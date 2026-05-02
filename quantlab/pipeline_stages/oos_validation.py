"""阶段 4: 样本外验证 —— Agent 驱动诊断分析。

保留全部数值计算（样本拆分 + IC计算），在此基础上由 AgentAnalyst
对每个因子的OOS表现进行定性诊断，并生成跨因子汇总和下一轮发现反馈。
LLM 不可用时回退到规则化诊断，不阻塞管线。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quantlab.metrics import compute_rank_ic

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class AgentOOSValidationStage(PipelineStage):
    """Agent 驱动的样本外验证阶段。

    数值计算层（始终执行）:
      - SampleSplitter 拆分训练/测试集
      - SafeFactorExecutor 执行因子
      - compute_rank_ic 计算训练/测试 IC
      - 成本调整 IC 计算

    Agent 分析层（LLM 可用时执行）:
      - 逐因子 OOS 诊断（健康/过拟合/结构断裂/信号不足）
      - 跨因子汇总（表现最好/最差的因子族）
      - 对下一轮因子发现的反馈建议

    反馈闭环: 分析结果写入 ctx._meta["oos_analysis"] 和
    ctx._meta["discovery_feedback"]，供 EvolutionStage 在下一轮读取。
    """

    def __init__(self, enable_agent: bool = True, agent_timeout: int = 90) -> None:
        self.enable_agent = enable_agent

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor
        from quantlab.factor_discovery.sample_split import SampleSplitter

        market_df = ctx.load_data()
        if market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        splitter = SampleSplitter(oos_months=6)
        split_result = splitter.split(market_df)
        if not split_result.sufficient:
            return {"status": "skipped", "reason": "数据不足，需要至少120天训练+40天测试"}

        store = PersistentFactorStore()
        entries = store.load_library_entries()
        approved = [e for e in entries if str(e.factor_spec.status) == "approved"]
        if not approved:
            return {"status": "skipped", "reason": "无已审批因子"}

        executor = SafeFactorExecutor()
        passed = 0
        failed = 0
        checks = []

        # ---- 数值计算层 ----
        for entry in approved:
            try:
                train_df = split_result.train_df
                test_df = split_result.test_df

                train_computed = executor.execute(entry.factor_spec, train_df)
                train_panel = train_computed.get("factor_panel")
                if train_panel is None or len(train_panel) == 0:
                    continue

                test_computed = executor.execute(entry.factor_spec, test_df)
                test_panel = test_computed.get("factor_panel")
                if test_panel is None or len(test_panel) == 0:
                    continue

                train_ic = compute_rank_ic(train_panel, train_df)["ic_mean"]
                test_ic = compute_rank_ic(test_panel, test_df)["ic_mean"]

                oos_passed = test_ic > 0.0 and (train_ic <= 0 or test_ic / (train_ic + 1e-10) > 0.5)

                cost_result = {}
                try:
                    from quantlab.trading.cost_model import compute_turnover_cost_impact
                    cost_result = compute_turnover_cost_impact(test_panel, test_df)
                except Exception:
                    pass

                checks.append({
                    "factor_id": entry.factor_spec.factor_id,
                    "factor_name": getattr(entry.factor_spec, "name", ""),
                    "factor_family": getattr(entry.factor_spec, "family", "unknown"),
                    "direction": getattr(entry.factor_spec, "direction", ""),
                    "train_ic": train_ic,
                    "test_ic": test_ic,
                    "oos_decay": round(1.0 - test_ic / (train_ic + 1e-10), 4) if train_ic > 0 else 0.0,
                    "cost_adj_ic": round(test_ic - cost_result.get("cost_adj_ic_penalty", 0), 4),
                    "turnover": cost_result.get("turnover", 0),
                    "cost_impact_bps": cost_result.get("cost_impact_bps", 0),
                    "passed": oos_passed,
                })

                if oos_passed:
                    passed += 1
                else:
                    failed += 1
                    logger.info("因子 %s OOS验证失败: train_ic=%.4f test_ic=%.4f",
                               entry.factor_spec.factor_id, train_ic, test_ic)

            except Exception as exc:
                checks.append({
                    "factor_id": entry.factor_spec.factor_id,
                    "factor_name": getattr(entry.factor_spec, "name", ""),
                    "factor_family": getattr(entry.factor_spec, "family", "unknown"),
                    "error": str(exc)[:200],
                    "passed": False,
                })
                failed += 1

        # ---- 市场概览（供 Agent 分析使用） ----
        market_summary = {
            "train_days": split_result.train_trading_days,
            "test_days": split_result.test_trading_days,
            "cutoff_date": str(split_result.cutoff_date.date()),
            "asset_count": split_result.test_assets,
        }

        # ---- Walk-Forward 验证层 ----
        walk_forward = {}
        try:
            walk_forward = _run_walk_forward(approved, market_df, executor)
            logger.info("Walk-Forward完成: %d/%d 稳定",
                       walk_forward.get("stable_count", 0),
                       walk_forward.get("total_tested", 0))
        except Exception as exc:
            logger.warning("Walk-Forward验证失败: %s", exc)
            walk_forward = {"status": "failed", "error": str(exc)[:200]}

        # ---- Agent 分析层 ----
        oos_analysis = {}
        discovery_feedback = {}

        if self.enable_agent and checks:
            try:
                from .agent_analyst import AgentAnalyst

                analyst = AgentAnalyst(timeout=self.agent_timeout)
                oos_analysis = analyst.analyze_oos(checks, market_summary)
                logger.info(
                    "Agent OOS 分析完成: status=%s, %d个因子已诊断",
                    oos_analysis.get("status", "unknown"),
                    len(oos_analysis.get("per_factor", [])),
                )

                # 生成跨轮反馈
                governance_snapshot = ctx._meta.get("governance_analysis", {})
                discovery_feedback = analyst.generate_feedback(oos_analysis, governance_snapshot)
            except Exception as exc:
                logger.warning("Agent OOS 分析失败: %s", exc)
                oos_analysis = {"status": "failed", "error": str(exc)[:200]}

        # ---- 反馈闭环: 写入 ctx._meta 供 EvolutionStage 读取 ----
        ctx._meta["oos_analysis"] = oos_analysis
        ctx._meta["discovery_feedback"] = discovery_feedback

        # ---- 保存分析记录 ----
        self._save_oos_record(checks, oos_analysis, discovery_feedback)

        return {
            "status": "success",
            "total": len(approved),
            "passed": passed,
            "failed": failed,
            "cutoff_date": str(split_result.cutoff_date.date()),
            "test_days": split_result.test_trading_days,
            "checks": checks,
            "walk_forward": walk_forward,
            "agent_analysis": oos_analysis,
            "discovery_feedback": discovery_feedback,
        }

def _run_walk_forward(
    entries: list,
    market_df: pd.DataFrame,
    executor: Any,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
) -> dict[str, Any]:
    """Walk-forward OOS validation across multiple rolling windows.

    For each window: train on [T-train_months, T), test on [T, T+test_months).
    Slides forward by step_months. Requires at least 2 complete windows.

    Returns stability metrics: OOS IC across windows, win rate, max drawdown.
    """
    from quantlab.metrics import compute_rank_ic

    dates = sorted(pd.to_datetime(market_df["date"]).unique())
    if len(dates) < (train_months + test_months) * 21:
        return {"status": "skipped", "reason": f"需要至少 {(train_months+test_months)*21} 个交易日"}

    # Generate windows
    windows = []
    start_idx = 0
    while True:
        train_end_idx = start_idx + train_months * 21  # ~21 trading days/month
        test_end_idx = train_end_idx + test_months * 21
        if test_end_idx > len(dates):
            break
        windows.append({
            "train_start": dates[start_idx],
            "train_end": dates[train_end_idx - 1],
            "test_start": dates[train_end_idx],
            "test_end": dates[min(test_end_idx, len(dates)) - 1],
        })
        start_idx += step_months * 21

    if len(windows) < 2:
        return {"status": "skipped", "reason": f"仅 {len(windows)} 个窗口，需至少2个"}

    # Run validation per factor
    wf_results: dict[str, dict] = {}
    for entry in entries:
        fid = entry.factor_spec.factor_id
        window_ics = []
        window_passed = []

        for w in windows:
            try:
                train_df = market_df[
                    (pd.to_datetime(market_df["date"]) >= w["train_start"]) &
                    (pd.to_datetime(market_df["date"]) <= w["train_end"])
                ]
                test_df = market_df[
                    (pd.to_datetime(market_df["date"]) >= w["test_start"]) &
                    (pd.to_datetime(market_df["date"]) <= w["test_end"])
                ]
                if len(train_df) < 100 or len(test_df) < 40:
                    continue

                train_computed = executor.execute(entry.factor_spec, train_df)
                train_panel = train_computed.get("factor_panel")
                test_computed = executor.execute(entry.factor_spec, test_df)
                test_panel = test_computed.get("factor_panel")
                if train_panel is None or test_panel is None:
                    continue

                train_ic = compute_rank_ic(train_panel, train_df)["ic_mean"]
                test_ic = compute_rank_ic(test_panel, test_df)["ic_mean"]
                window_ics.append({
                    "train_start": str(w["train_start"].date()),
                    "test_start": str(w["test_start"].date()),
                    "train_ic": round(train_ic, 6),
                    "test_ic": round(test_ic, 6),
                })
                window_passed.append(test_ic > 0)
            except Exception:
                pass

        if len(window_ics) >= 2:
            test_ics = [w["test_ic"] for w in window_ics]
            mean_oos = float(np.mean(test_ics)) if test_ics else 0.0
            std_oos = float(np.std(test_ics, ddof=1)) if len(test_ics) > 1 else 0.0
            win_rate = float(np.mean(window_passed)) if window_passed else 0.0
            # Max OOS IC drawdown across windows
            cumsum = np.cumsum(test_ics)
            peak = np.maximum.accumulate(cumsum)
            dd = np.max(peak - cumsum) if len(cumsum) > 0 else 0.0

            wf_results[fid] = {
                "n_windows": len(window_ics),
                "mean_oos_ic": round(mean_oos, 6),
                "std_oos_ic": round(std_oos, 6),
                "oos_ic_ir": round(mean_oos / max(std_oos, 1e-10), 4),
                "win_rate": round(win_rate, 4),
                "max_ic_drawdown": round(float(dd), 6),
                "stable": win_rate >= 0.6 and mean_oos > 0,
                "windows": window_ics,
            }

    stable_count = sum(1 for r in wf_results.values() if r.get("stable", False))
    return {
        "status": "success",
        "n_windows": len(windows),
        "total_tested": len(wf_results),
        "stable_count": stable_count,
        "stability_ratio": round(stable_count / max(len(wf_results), 1), 4),
        "per_factor": wf_results,
    }


def _save_oos_record(
        self,
        checks: list[dict],
        analysis: dict[str, Any],
        feedback: dict[str, Any],
    ) -> None:
        """保存 OOS 分析记录到 scheduler 目录。"""
        try:
            from quantlab.config import DATA_DIR

            record_dir = DATA_DIR / "scheduler"
            record_dir.mkdir(parents=True, exist_ok=True)
            record_path = record_dir / "oos_analysis.json"

            records = []
            if record_path.exists():
                try:
                    records = json.loads(record_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            records.append({
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "agent_analysis": analysis,
                "discovery_feedback": feedback,
            })
            records = records[-30:]

            record_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("保存 OOS 分析记录失败: %s", exc)
