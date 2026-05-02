"""阶段 6: 因子库治理 —— Agent 驱动综合解读。

保留全部数值检测（市场状态、拥挤度、衰减曲线、风控评估），
在此基础上由 AgentAnalyst 对治理结果进行整体解读：
  - 市场状态叙事（市况对因子库的影响）
  - 拥挤度机制分析（拥挤因子的共同特征）
  - 衰减模式解读（系统性的衰减趋势）
  - 前瞻性风控建议

LLM 不可用时回退到规则化摘要，不阻塞管线。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class AgentGovernanceStage(PipelineStage):
    """Agent 驱动的因子库治理阶段。

    数值检测层（始终执行）:
      - 生命周期管理（归档低表现因子、晋级成熟因子）
      - 市场状态检测（牛/熊/震荡）
      - 拥挤度检测（相关聚类 + 拥挤闭环）
      - 因子衰减曲线分析（IC 半衰期）
      - 风控评估（组合风险评分）

    Agent 分析层（LLM 可用时执行）:
      - 整体治理摘要
      - 市况对因子库的影响解读
      - 拥挤度背后的机制分析
      - 衰减模式与系统性风险识别
      - 前瞻性管理建议
    """

    def __init__(self, enable_agent: bool = True, agent_timeout: int = 90) -> None:
        self.enable_agent = enable_agent
        self.agent_timeout = agent_timeout

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.runtime import PersistentFactorStore
        from quantlab.factor_discovery.models import FactorLifecycleManager, FactorStatus

        store = PersistentFactorStore()
        stats_before = store.get_library_stats()
        result = store.archive_underperforming(min_observe_days=30, min_ic_threshold=0.015)
        stats_after = store.get_library_stats()

        # ---- 生命周期管理 ----
        lc_manager = FactorLifecycleManager()
        lifecycle_changes = []

        def _parse_status(s) -> FactorStatus:
            if isinstance(s, FactorStatus):
                return s
            raw = str(s).split(".")[-1]
            try:
                return FactorStatus(raw)
            except ValueError:
                return FactorStatus.DRAFT

        for entry in store.load_library_entries():
            current = _parse_status(entry.factor_spec.status)
            recommended, reason = lc_manager.recommend(
                entry, days_since_eval=0, oos_ic=0, cost_adj_ic=0,
                crowding_score=result.get("crowding_scores", {}).get(
                    entry.factor_spec.factor_id, 0
                ),
            )
            if recommended != current:
                ok, msg = lc_manager.transition(current, recommended)
                if ok:
                    entry.factor_spec.status = recommended
                    store.upsert_library_entry(entry)
                    lifecycle_changes.append({
                        "factor_id": entry.factor_spec.factor_id,
                        "from": current.value,
                        "to": recommended.value,
                        "reason": reason,
                    })
        if lifecycle_changes:
            result["lifecycle_changes"] = lifecycle_changes
            logger.info("生命周期变更: %d 个因子", len(lifecycle_changes))

        result["stats_before"] = stats_before
        result["stats_after"] = stats_after
        result["status"] = "success"

        # ---- 数值检测层 ----
        result["regime"] = _detect_regime(ctx)
        result["crowding"] = _detect_crowding(ctx, store, result)
        result["curves"] = _analyze_curves(ctx, store)
        result["risk"] = _assess_risk(ctx, store, result)

        # ---- Agent 分析层 ----
        governance_analysis = {}
        if self.enable_agent:
            try:
                from .agent_analyst import AgentAnalyst

                analyst = AgentAnalyst(timeout=self.agent_timeout)
                governance_analysis = analyst.analyze_governance(result)
                logger.info(
                    "Agent 治理分析完成: status=%s",
                    governance_analysis.get("status", "unknown"),
                )
            except Exception as exc:
                logger.warning("Agent 治理分析失败: %s", exc)
                governance_analysis = {"status": "failed", "error": str(exc)[:200]}

        result["agent_analysis"] = governance_analysis

        # ---- 反馈闭环: 写入 ctx._meta ----
        ctx._meta["governance_analysis"] = governance_analysis

        # ---- 保存治理分析记录 ----
        self._save_governance_record(result)

        return result

    def _save_governance_record(self, result: dict[str, Any]) -> None:
        """保存治理分析记录。"""
        try:
            from quantlab.config import DATA_DIR

            record_dir = DATA_DIR / "scheduler"
            record_dir.mkdir(parents=True, exist_ok=True)
            record_path = record_dir / "governance_analysis.json"

            records = []
            if record_path.exists():
                try:
                    records = json.loads(record_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            records.append({
                "timestamp": datetime.now().isoformat(),
                "regime": result.get("regime", {}),
                "crowding": result.get("crowding", {}),
                "curves": result.get("curves", {}),
                "risk": result.get("risk", {}),
                "lifecycle_changes": result.get("lifecycle_changes", []),
                "agent_analysis": result.get("agent_analysis", {}),
            })
            records = records[-30:]

            record_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("保存治理分析记录失败: %s", exc)


# ------------------------------------------------------------------
# 数值检测函数（保持原有逻辑不变）
# ------------------------------------------------------------------


def _detect_regime(ctx: PipelineContext) -> dict[str, Any]:
    try:
        from quantlab.factor_discovery.factor_enhancements import RegimeDetector
        rd = RegimeDetector()
        market_df = ctx.load_data()
        if not market_df.empty:
            regime_result = rd.detect(market_df)
            region = regime_result.current_regime
            bull_pct = regime_result.regime_stats.get("bull", {}).get("pct_days", 0) * 100
            bear_pct = regime_result.regime_stats.get("bear", {}).get("pct_days", 0) * 100
            logger.info("市场状态: %s (牛市 %.0f%%, 熊市 %.0f%%)", region, bull_pct, bear_pct)
            return {"current": region, "stats": regime_result.regime_stats}
    except Exception as exc:
        logger.warning("市场状态检测失败: %s", exc)
    return {"status": "failed"}


def _detect_crowding(ctx: PipelineContext, store: Any, result: dict[str, Any]) -> dict[str, Any]:
    try:
        from quantlab.factor_discovery.factor_enhancements import CrowdingDetector
        detector = CrowdingDetector()
        crowding = detector.detect(correlation_threshold=0.6, min_factors=3)
        crowding_result = {
            "crowded_factor_ids": crowding.crowded_factor_ids,
            "clusters": len(crowding.clusters),
            "max_observed_corr": crowding.max_observed_corr,
            "avg_observed_corr": crowding.avg_observed_corr,
        }
        if crowding.crowded_factor_ids:
            logger.info("拥挤度检测: %d 个拥挤因子 (%d 个集群)",
                       len(crowding.crowded_factor_ids), len(crowding.clusters))
            try:
                crowded_dirs = set()
                for fid in crowding.crowded_factor_ids:
                    for entry in store.load_library_entries():
                        if entry.factor_spec.factor_id == fid:
                            crowded_dirs.add(entry.factor_spec.family or fid[:8])
                            break
                avoid_dirs = sorted(crowded_dirs)[:5]
                logger.info("拥挤闭环: %d个因子降权, 引导避开方向: %s",
                           len(crowding.crowded_factor_ids), avoid_dirs)
                crowding_result["crowding_closed_loop"] = {
                    "penalized_count": len(crowding.crowded_factor_ids),
                    "avoid_directions": avoid_dirs,
                }
            except Exception as e2:
                logger.warning("拥挤闭环处理失败: %s", e2)
        return crowding_result
    except Exception as exc:
        logger.warning("拥挤度检测失败: %s", exc)
        return {"status": "failed", "error": str(exc)[:200]}


def _analyze_curves(ctx: PipelineContext, store: Any) -> dict[str, Any]:
    try:
        from quantlab.factor_discovery.factor_enhancements import FactorCurveAnalyzer
        from quantlab.factor_discovery.runtime import SafeFactorExecutor

        curve_analyzer = FactorCurveAnalyzer()
        entries = store.load_library_entries()
        approved = [e for e in entries if str(e.factor_spec.status) == "approved"]
        if not approved:
            return {"status": "skipped", "reason": "无已审批因子"}

        curve_market_df = ctx.load_data()
        if curve_market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        curve_executor = SafeFactorExecutor()
        curve_results: dict[str, dict] = {}
        for entry in approved:
            try:
                computed = curve_executor.execute(entry.factor_spec, curve_market_df)
                panel = computed.get("factor_panel")
                if panel is not None and not panel.empty:
                    decay = curve_analyzer.ic_decay_curve(
                        panel, curve_market_df, windows=[5, 10, 20, 40, 60]
                    )
                    curve_results[entry.factor_spec.factor_id] = decay
            except Exception as exc:
                logger.debug("因子 %s 曲线分析失败: %s", entry.factor_spec.factor_id, exc)

        result: dict[str, Any] = {"total": len(curve_results), "details": curve_results}
        if curve_results:
            half_life_factors = {
                fid: d["half_life"]
                for fid, d in curve_results.items()
                if d.get("half_life", 0) > 0
            }
            result["factors_with_half_life"] = len(half_life_factors)
            logger.info("因子曲线分析: %d 个因子, %d 个有半衰期",
                       len(curve_results), len(half_life_factors))
        return result
    except Exception as exc:
        logger.warning("因子曲线分析失败: %s", exc)
        return {"status": "failed", "error": str(exc)[:200]}


def _assess_risk(ctx: PipelineContext, store: Any, result: dict[str, Any]) -> dict[str, Any]:
    try:
        from quantlab.trading.risk_control import RiskManager
        from quantlab.factor_discovery.runtime import SafeFactorExecutor

        rm = RiskManager()
        comb_weights = result.get("weights", {})
        comb_ids = result.get("factor_ids", [])
        if not (comb_weights and comb_ids):
            return {"status": "skipped", "reason": "无组合权重数据"}

        market_df = ctx.load_data()
        executor = SafeFactorExecutor()
        factor_panels = {}
        for entry in store.load_library_entries():
            if entry.factor_spec.factor_id in comb_ids:
                try:
                    computed = executor.execute(entry.factor_spec, market_df)
                    fp = computed.get("factor_panel")
                    if fp is not None:
                        factor_panels[entry.factor_spec.factor_id] = fp
                except Exception:
                    pass

        risk_report = rm.evaluate(comb_weights, factor_panels, market_df)
        if risk_report.breaches:
            logger.warning("风控告警 (score=%.2f): %s",
                          risk_report.risk_score, "; ".join(risk_report.breaches[:3]))
        return {
            "risk_score": risk_report.risk_score,
            "passed": risk_report.passed,
            "recommendation": risk_report.recommendation,
            "breaches": risk_report.breaches,
        }
    except Exception as exc:
        logger.warning("风控评估失败: %s", exc)
        return {"status": "failed", "error": str(exc)[:200]}
