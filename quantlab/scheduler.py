"""每日自动调度器 —— 因子工厂无人值守运行。

核心流程：
1. 增量刷新横截面数据（只拉新交易日）
2. 运行因子衰减监控（检查已有因子 IC 是否衰减）
3. 触发进化搜索（对衰减因子或按计划发掘新因子）
4. 交付标准自动筛选（只保留可卖因子）
5. 生成交付报告（可交付因子输出 JSON + Markdown）

使用方式：
    # 手动执行一次日常任务
    python -m quantlab.scheduler run_daily

    # 启动 Windows 计划任务（每日 18:30 执行）
    python -m quantlab.scheduler install_cron

    # 查看最近执行记录
    python -m quantlab.scheduler status
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH, DATA_DIR
from quantlab.factor_discovery.factor_enhancements import (
    ExperienceLoop,
    FactorCombiner,
    OrthogonalityGuide,
)

logger = logging.getLogger(__name__)

SCHEDULER_DIR = DATA_DIR / "scheduler"
SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = SCHEDULER_DIR / "daily_runs.json"


# ---------------------------------------------------------------------------
# 1. 执行记录
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DailyRunRecord:
    """一次日常执行记录。"""
    run_id: str
    run_date: str
    start_time: str
    end_time: str = ""
    status: str = "running"  # running / success / partial / failed

    # 各阶段结果摘要
    data_refresh: dict[str, Any] = field(default_factory=dict)
    decay_monitor: dict[str, Any] = field(default_factory=dict)
    evolution: dict[str, Any] = field(default_factory=dict)
    screening: dict[str, Any] = field(default_factory=dict)
    oos_validation: dict[str, Any] = field(default_factory=dict)
    combination: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)
    delivery_reports: list[str] = field(default_factory=list)

    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_run_log() -> list[dict[str, Any]]:
    if not RUN_LOG_PATH.exists():
        return []
    try:
        return json.loads(RUN_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_run_log(records: list[dict[str, Any]]) -> None:
    # 只保留最近 90 条
    records = records[-90:]
    RUN_LOG_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 2. 每日执行引擎
# ---------------------------------------------------------------------------

class DailyScheduler:
    """每日因子工厂调度引擎。"""

    def __init__(
        self,
        data_path: Path | str | None = None,
        directions: list[str] | None = None,
        evolution_rounds: int = 3,
        max_candidates_per_round: int = 5,
        use_adaptive_directions: bool = True,
    ) -> None:
        self.data_path = Path(data_path or DEFAULT_CROSS_SECTION_DATA_PATH)
        self.directions = directions or [
            "momentum_reversal",
            "quality_earnings",
            "volume_price_divergence",
            "volatility_regime",
            "liquidity_premium",
        ]
        self._fallback_directions = list(self.directions)
        self.evolution_rounds = evolution_rounds
        self.max_candidates_per_round = max_candidates_per_round
        self.use_adaptive_directions = use_adaptive_directions
        self.experience_loop = ExperienceLoop()
        self.orth_guide = OrthogonalityGuide()
        self.combiner = FactorCombiner()
        self.alert_bus = None  # set on first use

    def run_daily(self) -> DailyRunRecord:
        """执行一次完整的日常任务。"""
        now = datetime.now()
        record = DailyRunRecord(
            run_id=f"daily_{now.strftime('%Y%m%d_%H%M%S')}",
            run_date=now.strftime("%Y-%m-%d"),
            start_time=now.isoformat(),
        )

        logger.info("=== 每日因子工厂启动 === run_id=%s", record.run_id)

        try:
            # ---- 阶段 1: 增量数据刷新 ----
            record.data_refresh = self._refresh_data()
            logger.info("数据刷新完成: %s", record.data_refresh.get("status", "unknown"))

            # ---- 阶段 2: 因子衰减监控 ----
            record.decay_monitor = self._monitor_decay()
            logger.info("衰减监控完成: %d 因子需再发掘", record.decay_monitor.get("decayed_count", 0))

            # ---- 阶段 3: 进化搜索 ----
            record.evolution = self._run_evolution()
            logger.info("进化搜索完成: 新增 %d 因子", record.evolution.get("new_approved", 0))

            # ---- 阶段 4: 样本外验证 ----
            try:
                record.oos_validation = self._validate_oos()
                logger.info("OOS验证完成: 通过 %d / 失败 %d",
                           record.oos_validation.get("passed", 0),
                           record.oos_validation.get("failed", 0))
            except Exception as exc:
                logger.warning("OOS验证失败: %s", exc)
                record.oos_validation = {"status": "failed", "error": str(exc)[:200]}

            # ---- 阶段 5: 多因子组合 ----
            try:
                record.combination = self._combine_factors()
                logger.info("多因子组合完成: 组合IC=%.4f", record.combination.get("combined_ic", 0))
            except Exception as exc:
                logger.warning("多因子组合失败: %s", exc)
                record.combination = {"status": "failed", "error": str(exc)[:200]}

            # ---- 阶段 6: 交付标准筛选 ----
            record.screening = self._screen_deliverable()
            logger.info("筛选完成: %d 可交付因子", record.screening.get("deliverable_count", 0))

            # ---- 阶段 7: 因子库治理 ----
            try:
                record.governance = self._run_governance()
                logger.info("因子库治理完成: 归档 %d 个因子", record.governance.get("archived_count", 0))
            except Exception as exc:
                logger.warning("因子库治理失败: %s", exc)
                record.governance = {"status": "failed", "error": str(exc)[:200]}

            # ---- 阶段 8: 生成交付报告 ----
            record.delivery_reports = self._generate_delivery_reports(
                record.screening.get("deliverable_factor_ids", [])
            )
            logger.info("交付报告生成: %d 份", len(record.delivery_reports))

            record.status = "success"

        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)[:500]
            logger.error("日常任务失败: %s", exc)

        record.end_time = datetime.now().isoformat()

        # Emit alerts based on record
        alert_bus = self._get_alert_bus()
        if record.data_refresh.get("status") == "skipped":
            alert_bus.warning("数据刷新跳过", record.data_refresh.get("reason", ""), source="data_refresh")
        if record.decay_monitor.get("decayed_count", 0) > 0:
            alert_bus.warning("因子衰减告警", f"{record.decay_monitor.get('decayed_count')} 个因子已衰减", source="decay_monitor")
        if record.governance.get("crowding", {}).get("crowded_factor_ids"):
            alert_bus.warning("拥挤度告警", f"发现拥挤因子", source="crowding", ids=record.governance["crowding"]["crowded_factor_ids"])
        risk = record.governance.get("risk", {})
        if risk.get("breaches"):
            alert_bus.critical("风控告警", "; ".join(risk["breaches"][:3]), source="risk_control")
        if record.governance.get("regime", {}).get("current") == "bear":
            alert_bus.info("市场状态: 熊市", "因子表现可能出现系统性下降", source="regime")
        if record.status == "failed":
            alert_bus.critical("每日任务失败", record.error_message[:200], source="scheduler")

        alert_summary = alert_bus.summary()
        if alert_summary.get("critical", 0) > 0:
            record.status = "partial" if record.status == "success" else record.status
        if any(alert_summary.values()):
            logger.info("告警汇总: info=%d warning=%d critical=%d", alert_summary.get("info", 0), alert_summary.get("warning", 0), alert_summary.get("critical", 0))

        # 保存记录
        log = _load_run_log()
        log.append(record.to_dict())
        _save_run_log(log)

        logger.info("=== 每日因子工厂结束 === status=%s", record.status)
        return record

    # -- 阶段实现 --

    def _refresh_data(self) -> dict[str, Any]:
        """增量刷新横截面数据。"""
        try:
            from quantlab.data.tushare_provider import AkShareIncrementalProvider
            provider = AkShareIncrementalProvider()
            result = provider.refresh_cross_section(self.data_path)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.warning("增量刷新失败，尝试 Tushare Pro: %s", exc)
            try:
                from quantlab.data.tushare_provider import TushareProProvider
                provider = TushareProProvider()
                if provider.available:
                    result = provider.refresh_cross_section(self.data_path)
                    return {"status": "success_fallback", "result": result}
            except Exception as exc2:
                logger.warning("Tushare Pro 也失败: %s", exc2)
            return {"status": "skipped", "reason": str(exc)[:200]}

    def _monitor_decay(self) -> dict[str, Any]:
        """监控因子衰减。"""
        from quantlab.factor_discovery.decay_monitor import FactorDecayMonitor
        monitor = FactorDecayMonitor(data_path=self.data_path)
        return monitor.check_all()

    def _combine_factors(self) -> dict[str, Any]:
        """加载已审批因子并执行多因子组合。"""
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        entries = store.load_library_entries()
        approved = [e for e in entries if str(e.factor_spec.status) == "approved"]
        if not approved:
            return {"status": "skipped", "reason": "无已审批因子"}

        executor = SafeFactorExecutor()
        market_df = self._load_data()
        if market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        factor_panels: dict[str, "pd.Series"] = {}
        for entry in approved:
            try:
                computed = executor.execute(entry.factor_spec, market_df)
                fp = computed.get("factor_panel")
                if fp is not None and len(fp) > 0:
                    factor_panels[entry.factor_spec.factor_id] = fp
            except Exception as exc:
                logger.warning("因子 %s 执行失败: %s", entry.factor_spec.factor_id, exc)

        if len(factor_panels) < 2:
            return {"status": "skipped", "reason": f"可执行因子不足: {len(factor_panels)} 个"}

        try:
            result = self.combiner.combine(factor_panels, market_df, method="ic_weighted")
        except Exception as exc:
            return {"status": "failed", "error": str(exc)[:200]}

        # 持久化组合结果
        comb_dir = DATA_DIR / "assistant_data" / "combinations"
        comb_dir.mkdir(parents=True, exist_ok=True)
        comb_path = comb_dir / f"comb_{result.combination_id}.json"
        comb_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        return {
            "status": "success",
            "combined_ic": result.combined_ic,
            "combined_icir": result.combined_icir,
            "combined_rank_ic": result.combined_rank_ic,
            "factor_ids": result.factor_ids,
            "weights": result.weights,
            "method": result.method,
        }

    def _run_governance(self) -> dict[str, Any]:
        """因子库治理：归档过时/低效因子 + 生命周期 + 拥挤度检测 + 风控。"""
        from quantlab.factor_discovery.runtime import PersistentFactorStore
        from quantlab.factor_discovery.models import FactorLifecycleManager, FactorStatus
        store = PersistentFactorStore()
        stats_before = store.get_library_stats()
        result = store.archive_underperforming(min_observe_days=30, min_ic_threshold=0.015)
        stats_after = store.get_library_stats()

        # Factor lifecycle advancement
        lc_manager = FactorLifecycleManager()
        lifecycle_changes = []
        for entry in store.load_library_entries():
            current = FactorStatus(str(entry.factor_spec.status))
            recommended, reason = lc_manager.recommend(
                entry, days_since_eval=0, oos_ic=0, cost_adj_ic=0,
                crowding_score=result.get("crowding_scores", {}).get(entry.factor_spec.factor_id, 0),
            )
            if recommended != current:
                ok, msg = lc_manager.transition(current, recommended)
                if ok:
                    entry.factor_spec.status = recommended
                    store.upsert_library_entry(entry)
                    lifecycle_changes.append({"factor_id": entry.factor_spec.factor_id, "from": current.value, "to": recommended.value, "reason": reason})
        if lifecycle_changes:
            result["lifecycle_changes"] = lifecycle_changes
            logger.info("生命周期变更: %d 个因子", len(lifecycle_changes))
        result["stats_before"] = stats_before
        result["stats_after"] = stats_after
        result["status"] = "success"

        # Market regime detection
        try:
            from quantlab.factor_discovery.factor_enhancements import RegimeDetector
            rd = RegimeDetector()
            market_df = self._load_data()
            if not market_df.empty:
                regime_result = rd.detect(market_df)
                result["regime"] = {
                    "current": regime_result.current_regime,
                    "stats": regime_result.regime_stats,
                }
                self._current_regime = regime_result  # cache for other phases
                logger.info("市场状态: %s (牛市 %.0f%%, 熊市 %.0f%%)",
                           regime_result.current_regime,
                           regime_result.regime_stats.get("bull", {}).get("pct_days", 0) * 100,
                           regime_result.regime_stats.get("bear", {}).get("pct_days", 0) * 100)
        except Exception as exc:
            logger.warning("市场状态检测失败: %s", exc)
            result["regime"] = {"status": "failed", "error": str(exc)[:200]}

        # Crowding detection
        try:
            from quantlab.factor_discovery.factor_enhancements import CrowdingDetector
            detector = CrowdingDetector()
            crowding = detector.detect(correlation_threshold=0.6, min_factors=3)
            result["crowding"] = {
                "crowded_factor_ids": crowding.crowded_factor_ids,
                "clusters": len(crowding.clusters),
                "max_observed_corr": crowding.max_observed_corr,
                "avg_observed_corr": crowding.avg_observed_corr,
            }
            if crowding.crowded_factor_ids:
                logger.info("拥挤度检测: %d 个拥挤因子 (%d 个集群)", len(crowding.crowded_factor_ids), len(crowding.clusters))
        except Exception as exc:
            logger.warning("拥挤度检测失败: %s", exc)
            result["crowding"] = {"status": "failed", "error": str(exc)[:200]}

        # Risk control assessment
        try:
            from quantlab.trading.risk_control import RiskManager, RiskLimits
            rm = RiskManager()
            # Use combination weights if available
            comb_weights = result.get("weights", {})
            comb_ids = result.get("factor_ids", [])
            if comb_weights and comb_ids:
                # Load factor panels for risk assessment
                from quantlab.factor_discovery.runtime import SafeFactorExecutor
                executor = SafeFactorExecutor()
                factor_panels = {}
                entries = store.load_library_entries()
                for e in entries:
                    if e.factor_spec.factor_id in comb_ids:
                        try:
                            computed = executor.execute(e.factor_spec, market_df)
                            fp = computed.get("factor_panel")
                            if fp is not None:
                                factor_panels[e.factor_spec.factor_id] = fp
                        except Exception:
                            pass
                risk_report = rm.evaluate(comb_weights, factor_panels, market_df)
                result["risk"] = {
                    "risk_score": risk_report.risk_score,
                    "passed": risk_report.passed,
                    "recommendation": risk_report.recommendation,
                    "breaches": risk_report.breaches,
                }
                if risk_report.breaches:
                    logger.warning("风控告警 (score=%.2f): %s", risk_report.risk_score, "; ".join(risk_report.breaches[:3]))
            else:
                result["risk"] = {"status": "skipped", "reason": "无组合权重数据"}
        except Exception as exc:
            logger.warning("风控评估失败: %s", exc)
            result["risk"] = {"status": "failed", "error": str(exc)[:200]}

        return result

    def _run_evolution(self) -> dict[str, Any]:
        """运行因子进化搜索。"""
        from quantlab.factor_discovery.evolution import EvolutionConfig, FactorEvolutionLoop
        from quantlab.factor_discovery.runtime import PersistentFactorStore

        store = PersistentFactorStore()
        hub = self._load_data()
        if hub.empty:
            return {"status": "skipped", "reason": "数据为空"}

        # Cold-start bootstrap: inject seed factors when library is empty
        entries = store.load_library_entries()
        if not entries:
            logger.info("因子库为空，执行冷启动引导...")
            try:
                from quantlab.factor_discovery.seed_factors import bootstrap_seed_factors
                bootstrap_result = bootstrap_seed_factors(
                    market_df=hub, store=store, experience_loop=self.experience_loop,
                )
                logger.info("冷启动完成: 注入 %d 个种子因子", bootstrap_result.get("injected_count", 0))
            except Exception as exc:
                logger.warning("冷启动引导失败: %s", exc)

        total_approved = 0
        all_results = []

        # Adaptive direction selection
        effective_directions = list(self.directions)
        if self.use_adaptive_directions:
            try:
                direction_priorities = []
                for d in self.directions:
                    guidance = self.experience_loop.get_guidance(d)
                    orth_ctx = self.orth_guide.get_orthogonality_context(d)

                    total_recorded = guidance.get("total_recorded", 0)
                    direction_insight = guidance.get("direction_insight", "")
                    saturated = d in orth_ctx.get("saturated_directions", [])

                    win_rate = 0.0
                    successful = guidance.get("successful_patterns", [])
                    if total_recorded > 0:
                        useful_count = sum(1 for o in guidance.get("successful_patterns", [])
                                           if float(o.get("rank_ic", 0)) > 0.015)
                        win_rate = len(successful) / total_recorded if total_recorded > 0 else 0.0

                    priority = win_rate * 0.5 - (0.5 if saturated else 0.0)

                    direction_priorities.append({
                        "direction": d,
                        "priority": priority,
                        "win_rate": win_rate,
                        "total_recorded": total_recorded,
                        "saturated": saturated,
                        "insight": direction_insight[:100] if direction_insight else "",
                    })

                active = [dp for dp in direction_priorities if not dp["saturated"] or dp["win_rate"] > 0.1]
                if not active:
                    active = direction_priorities

                active.sort(key=lambda x: x["priority"], reverse=True)
                effective_directions = [dp["direction"] for dp in active]

                logger.info("自适应方向选择 (共%d个):", len(effective_directions))
                for dp in active:
                    logger.info("  %s: priority=%.3f win_rate=%.2f saturated=%s",
                               dp["direction"], dp["priority"], dp["win_rate"], dp["saturated"])

                # Build meta-learning param lookup
                meta_params: dict[str, dict] = {}
                for dp in active:
                    wr = dp["win_rate"]
                    if wr > 0.5:
                        meta_params[dp["direction"]] = {"rounds": self.evolution_rounds + 2, "candidates": self.max_candidates_per_round + 3}
                    elif wr > 0.2:
                        meta_params[dp["direction"]] = {"rounds": self.evolution_rounds, "candidates": self.max_candidates_per_round}
                    elif wr > 0.0:
                        meta_params[dp["direction"]] = {"rounds": max(1, self.evolution_rounds - 1), "candidates": self.max_candidates_per_round + 2}
                    else:
                        meta_params[dp["direction"]] = {"rounds": self.evolution_rounds, "candidates": self.max_candidates_per_round}

            except Exception as exc:
                logger.warning("自适应方向选择失败，回退到默认方向: %s", exc)
                effective_directions = list(self._fallback_directions)
                meta_params = {d: {"rounds": self.evolution_rounds, "candidates": self.max_candidates_per_round} for d in effective_directions}

        for direction in effective_directions:
            mp = meta_params.get(direction, {"rounds": self.evolution_rounds, "candidates": self.max_candidates_per_round})
            try:
                loop = FactorEvolutionLoop(
                    store=store,
                    config=EvolutionConfig(
                        max_rounds=mp["rounds"],
                        candidates_per_round=mp["candidates"],
                    ),
                )
                result = loop.run(direction=direction, market_df=hub)
                approved = result.get("approved_count", 0)
                total_approved += approved
                all_results.append({
                    "direction": direction,
                    "approved": approved,
                    "total_candidates": result.get("total_candidates", 0),
                    "best_score": result.get("best_score", 0.0),
                    "meta_rounds": mp["rounds"],
                    "meta_candidates": mp["candidates"],
                })
            except Exception as exc:
                logger.warning("方向 %s 进化失败: %s", direction, exc)
                all_results.append({"direction": direction, "error": str(exc)[:200]})

        return {
            "status": "success",
            "new_approved": total_approved,
            "directions": all_results,
            "adaptive_selection": self.use_adaptive_directions,
        }

    def _validate_oos(self) -> dict[str, Any]:
        """样本外验证：对已审批因子检查 OOS IC 衰减。"""
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor
        from quantlab.factor_discovery.sample_split import SampleSplitter

        market_df = self._load_data()
        if market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        splitter = SampleSplitter(oos_months=6)
        split_result = splitter.split(market_df)
        if not split_result.sufficient:
            return {"status": "skipped", "reason": "数据不足，需要至少120天训练+40天测试"}

        test_dates = set(pd.to_datetime(split_result.test_df["date"].unique()))
        store = PersistentFactorStore()
        entries = store.load_library_entries()
        approved = [e for e in entries if str(e.factor_spec.status) == "approved"]
        if not approved:
            return {"status": "skipped", "reason": "无已审批因子"}

        executor = SafeFactorExecutor()
        passed = 0
        failed = 0
        checks = []

        for entry in approved:
            try:
                train_df = split_result.train_df
                test_df = split_result.test_df

                # Train IC
                train_computed = executor.execute(entry.factor_spec, train_df)
                train_panel = train_computed.get("factor_panel")
                if train_panel is None or len(train_panel) == 0:
                    continue

                # Test IC
                test_computed = executor.execute(entry.factor_spec, test_df)
                test_panel = test_computed.get("factor_panel")
                if test_panel is None or len(test_panel) == 0:
                    continue

                # Compute train/test rank IC
                train_ic = self._compute_rank_ic(train_panel, train_df)
                test_ic = self._compute_rank_ic(test_panel, test_df)

                # OOS decay check: test IC > 0 and not decayed > 50% from train
                oos_passed = test_ic > 0.0 and (train_ic <= 0 or test_ic / (train_ic + 1e-10) > 0.5)

                # Cost impact on turnover
                cost_result = {}
                try:
                    from quantlab.trading.cost_model import compute_turnover_cost_impact
                    cost_result = compute_turnover_cost_impact(test_panel, test_df)
                except Exception:
                    pass

                checks.append({
                    "factor_id": entry.factor_spec.factor_id,
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
                    "error": str(exc)[:200],
                    "passed": False,
                })
                failed += 1

        return {
            "status": "success",
            "total": len(approved),
            "passed": passed,
            "failed": failed,
            "cutoff_date": str(split_result.cutoff_date.date()),
            "test_days": split_result.test_trading_days,
            "checks": checks,
        }

    def _compute_rank_ic(self, factor_panel, market_df) -> float:
        """计算 Rank IC。"""
        try:
            if hasattr(factor_panel, 'to_frame'):
                factor_panel = factor_panel
            if "date" not in market_df.columns or "asset" not in market_df.columns:
                return 0.0
            df = market_df[["date", "asset"]].copy()
            df["factor"] = factor_panel if hasattr(factor_panel, 'values') else factor_panel
            if "close" in market_df.columns:
                df = df.merge(market_df[["date", "asset", "close"]].assign(
                    close=lambda x: x.groupby("asset")["close"].shift(-5) / x["close"] - 1
                ), on=["date", "asset"], how="inner")
                ics = []
                for _, g in df.groupby("date"):
                    valid = g[["factor", "close"]].dropna()
                    if len(valid) >= 20:
                        ric = valid["factor"].rank().corr(valid["close"].rank(), method="pearson")
                        if not np.isnan(ric):
                            ics.append(ric)
                if ics:
                    return round(float(np.mean(ics)), 4)
            return 0.0
        except Exception:
            return 0.0

    def _screen_deliverable(self) -> dict[str, Any]:
        """筛选可交付因子。"""
        from quantlab.factor_discovery.delivery_screener import DeliveryScreener
        screener = DeliveryScreener(data_path=self.data_path)
        return screener.screen()

    def _generate_delivery_reports(self, factor_ids: list[str]) -> list[str]:
        """为可交付因子生成报告。"""
        if not factor_ids:
            return []

        from quantlab.factor_discovery.factor_report import FactorDeliveryReportGenerator
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        hub = self._load_data()
        if hub.empty:
            return []

        generator = FactorDeliveryReportGenerator()
        executor = SafeFactorExecutor()
        library = store.load_library_entries()
        reports = []

        for entry in library:
            if entry.factor_spec.factor_id not in factor_ids:
                continue
            try:
                computed = executor.execute(entry.factor_spec, hub)
                factor_panel = computed["factor_panel"]
                output_dir = str(DATA_DIR / "delivery_reports" / entry.factor_spec.factor_id)
                report = generator.generate(
                    factor_spec=entry.factor_spec,
                    factor_panel=factor_panel,
                    market_df=hub,
                    evaluation_report=entry.latest_report,
                    output_dir=output_dir,
                )
                reports.append(output_dir)
            except Exception as exc:
                logger.warning("因子 %s 报告生成失败: %s", entry.factor_spec.factor_id, exc)

        return reports

    def _get_alert_bus(self):
        if self.alert_bus is None:
            from quantlab.assistant.notifier import AlertBus
            self.alert_bus = AlertBus()
        return self.alert_bus

    def _load_data(self, apply_survivorship: bool = True) -> pd.DataFrame:
        """加载市场数据，可选应用幸存者偏差过滤。"""
        try:
            from quantlab.factor_discovery.datahub import DataHub
            hub = DataHub()
            df = hub.load(str(self.data_path), use_cache=False)
        except Exception:
            if self.data_path.exists():
                df = pd.read_csv(self.data_path)
            else:
                return pd.DataFrame()

        if apply_survivorship and not df.empty:
            try:
                from quantlab.factor_discovery.survivorship import apply_survivorship_filter
                df = apply_survivorship_filter(df)
            except Exception as exc:
                logger.warning("幸存者偏差过滤失败: %s", exc)

        return df


# ---------------------------------------------------------------------------
# 3. Windows 计划任务安装
# ---------------------------------------------------------------------------

def install_windows_task(task_name: str = "QuantAgentDaily", hour: int = 18, minute: int = 30) -> dict[str, str]:
    """安装 Windows 计划任务，每日定时执行。

    使用 schtasks 命令创建，无需管理员权限（当前用户作用域）。
    """
    python_exe = sys.executable
    project_root = str(Path(__file__).resolve().parent.parent)
    script = f'"{python_exe}" -m quantlab.scheduler run_daily'

    # 创建计划任务
    cmd = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", script,
        "/SC", "DAILY",
        "/ST", f"{hour:02d}:{minute:02d}",
        "/F",  # 强制覆盖
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"status": "installed", "task_name": task_name, "schedule": f"每日 {hour:02d}:{minute:02d}"}
        return {"status": "error", "message": result.stderr.strip()[:500]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:500]}


def remove_windows_task(task_name: str = "QuantAgentDaily") -> dict[str, str]:
    """移除 Windows 计划任务。"""
    cmd = ["schtasks", "/Delete", "/TN", task_name, "/F"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"status": "removed", "task_name": task_name}
        return {"status": "error", "message": result.stderr.strip()[:500]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:500]}


# ---------------------------------------------------------------------------
# 4. CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口。"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="quant-agent 每日调度器")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run_daily", help="执行一次日常任务")
    sub.add_parser("status", help="查看最近执行记录")

    cron_parser = sub.add_parser("install_cron", help="安装每日定时任务")
    cron_parser.add_argument("--hour", type=int, default=18)
    cron_parser.add_argument("--minute", type=int, default=30)

    sub.add_parser("remove_cron", help="移除定时任务")

    args = parser.parse_args()

    if args.command == "run_daily":
        scheduler = DailyScheduler()
        record = scheduler.run_daily()
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, default=str))

    elif args.command == "status":
        records = _load_run_log()
        if not records:
            print("暂无执行记录。")
        else:
            for r in records[-10:]:
                status_icon = {
                    "success": "[OK]",
                    "partial": "[~]",
                    "failed": "[!!]",
                    "running": "[..]",
                }.get(r.get("status", ""), "[?]")
                date = r.get("run_date", "?")
                new = r.get("evolution", {}).get("new_approved", 0)
                delivered = r.get("screening", {}).get("deliverable_count", 0)
                print(f"{status_icon} {date} | 新增={new} | 可交付={delivered} | {r.get('status', '')}")

    elif args.command == "install_cron":
        result = install_windows_task(hour=args.hour, minute=args.minute)
        print(json.dumps(result, ensure_ascii=False))

    elif args.command == "remove_cron":
        result = remove_windows_task()
        print(json.dumps(result, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
