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

        # ---- 生命周期管理（使用真实OOS数据） ----
        lc_manager = FactorLifecycleManager()
        lifecycle_changes = []

        # 加载OOS验证结果以获得真实IC数据
        oos_lookup: dict[str, dict] = {}
        try:
            oos_records_path = DATA_DIR / "scheduler" / "oos_analysis.json"
            if oos_records_path.exists():
                oos_records = json.loads(oos_records_path.read_text(encoding="utf-8"))
                if oos_records:
                    for check in oos_records[-1].get("checks", []):
                        oos_lookup[check["factor_id"]] = check
        except Exception:
            pass

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
            fid = entry.factor_spec.factor_id
            oos_data = oos_lookup.get(fid, {})

            # 使用真实的OOS IC和cost-adj IC
            real_oos_ic = oos_data.get("test_ic", 0.0)
            real_cost_adj_ic = oos_data.get("cost_adj_ic", real_oos_ic)

            recommended, reason = lc_manager.recommend(
                entry,
                days_since_eval=0,
                oos_ic=real_oos_ic,
                cost_adj_ic=real_cost_adj_ic,
                crowding_score=result.get("crowding_scores", {}).get(fid, 0),
            )
            if recommended != current:
                ok, msg = lc_manager.transition(current, recommended)
                if ok:
                    entry.factor_spec.status = recommended
                    store.upsert_library_entry(entry)
                    lifecycle_changes.append({
                        "factor_id": fid,
                        "from": current.value,
                        "to": recommended.value,
                        "reason": reason,
                        "oos_ic": round(real_oos_ic, 6),
                        "cost_adj_ic": round(real_cost_adj_ic, 6),
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

        # ---- 因子健康监控 ----
        monitor_results = self._run_factor_monitor(ctx, store, result)
        result["factor_monitor"] = monitor_results
        self._emit_monitor_alerts(monitor_results)

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

    def _run_factor_monitor(
        self, ctx: PipelineContext, store: Any, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Run FactorMonitor on all library factors.

        Computes IC drift, crowding trend, and rolling Sharpe for each factor,
        returning a summary dict with per-factor health reports.
        """
        try:
            from quantlab.pipeline_stages.factor_monitor import FactorMonitor
            from quantlab.factor_discovery.runtime import SafeFactorExecutor
            from quantlab.metrics import compute_ic_sequence

            monitor = FactorMonitor()
            executor = SafeFactorExecutor()
            market_df = ctx.load_data()
            if market_df.empty:
                return {"status": "skipped", "reason": "数据为空"}

            entries = store.load_library_entries()
            if not entries:
                return {"status": "skipped", "reason": "因子库为空"}

            reports: dict[str, dict[str, Any]] = {}
            healthy_count = 0
            warning_count = 0
            critical_count = 0

            # Pre-fetch crowding data from current run's result
            crowding_data = result.get("crowding", {})

            for entry in entries:
                fid = entry.factor_spec.factor_id
                try:
                    # Execute factor to get factor panel
                    computed = executor.execute(entry.factor_spec, market_df)
                    factor_panel = computed.get("factor_panel")
                    if factor_panel is None or factor_panel.empty:
                        continue

                    # Compute IC sequence
                    ic_series = compute_ic_sequence(factor_panel, market_df)
                    if ic_series.empty:
                        ic_series = None

                    # Get crowding history from current crowding result
                    crowding_history: list[dict[str, Any]] = []
                    crowding_score_val = crowding_data.get("crowding_scores", {}).get(fid, 0.0)
                    if isinstance(crowding_score_val, (int, float)):
                        crowding_history = [{"crowding_score": float(crowding_score_val)}]

                    # Build NAV series from factor panel returns (simplified)
                    nav_series = _build_nav_from_factor_panel(factor_panel, market_df)
                    if nav_series is None or nav_series.empty:
                        continue

                    # Run all checks
                    health = monitor.run_all(
                        factor_id=fid,
                        ic_series=ic_series,
                        crowding_history=crowding_history,
                        nav_series=nav_series,
                    )
                    reports[fid] = health.to_dict()

                    if health.overall_health.value == "healthy":
                        healthy_count += 1
                    elif health.overall_health.value == "warning":
                        warning_count += 1
                    else:
                        critical_count += 1
                except Exception as exc:
                    logger.debug("因子 %s 健康监控失败: %s", fid, exc)

            return {
                "status": "success",
                "total_checked": len(reports),
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "reports": reports,
            }
        except Exception as exc:
            logger.warning("因子健康监控失败: %s", exc)
            return {"status": "failed", "error": str(exc)[:200]}

    def _emit_monitor_alerts(self, monitor_results: dict[str, Any]) -> None:
        """Feed factor monitor alerts to AlertBus."""
        reports = monitor_results.get("reports", {})
        if not reports:
            return
        try:
            from quantlab.assistant.notifier import AlertBus

            bus = AlertBus()
            for fid, report in reports.items():
                health = report.get("overall_health", "healthy")
                if health == "critical":
                    bus.critical(
                        f"因子 {fid} 健康状态: 严重",
                        "; ".join(report.get("recommendations", [])[:3]),
                        source="factor_monitor",
                        factor_id=fid,
                    )
                elif health == "warning":
                    bus.warning(
                        f"因子 {fid} 健康状态: 告警",
                        "; ".join(report.get("recommendations", [])[:3]),
                        source="factor_monitor",
                        factor_id=fid,
                    )
        except Exception as exc:
            logger.debug("监控告警发送失败: %s", exc)

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


def _build_nav_from_factor_panel(
    factor_panel: Any,
    market_df: Any,
    n_long: int = 50,
) -> Any:
    """Build a synthetic NAV series from a factor panel using long-top equal-weight.

    Used by FactorMonitor to estimate rolling Sharpe from factor signals.
    """
    import pandas as pd

    try:
        # Normalize to DataFrame
        if isinstance(factor_panel, pd.Series):
            fp = factor_panel.reset_index()
            if fp.shape[1] >= 3:
                fp.columns = ["date", "asset", "factor_value"]
            elif fp.shape[1] == 2:
                fp.columns = ["asset", "factor_value"]
            else:
                return None
        else:
            fp = factor_panel.copy()

        if "date" not in fp.columns or "asset" not in fp.columns:
            return None
        if "factor_value" not in fp.columns:
            val_cols = [c for c in fp.columns if c not in ("date", "asset")]
            if val_cols:
                fp = fp.rename(columns={val_cols[0]: "factor_value"})
            else:
                return None

        fp["date"] = pd.to_datetime(fp["date"])
        mkt = market_df.copy()
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt["asset"] = mkt["asset"].astype(str)
        fp["asset"] = fp["asset"].astype(str)

        if "close" not in mkt.columns:
            return None

        dates = sorted(fp["date"].unique())
        if len(dates) < 3:
            return None

        nav = pd.Series(1.0, index=[dates[0]], dtype=float)

        for i in range(len(dates)):
            date = dates[i]
            day_factors = fp[fp["date"] == date].copy()
            if day_factors.empty:
                continue

            day_factors = day_factors.dropna(subset=["factor_value"])
            n_select = min(n_long, len(day_factors))
            if n_select == 0:
                continue

            selected = day_factors.nlargest(n_select, "factor_value")
            target: dict[str, float] = {a: 1.0 / n_select for a in selected["asset"].tolist()}

            if i > 0 and dates[i - 1] in nav.index:
                prev_nav = float(nav[dates[i - 1]])
                # Compute period return
                period_ret = 0.0
                for asset, weight in target.items():
                    asset_mkt = mkt[mkt["asset"] == asset]
                    if asset_mkt.empty:
                        continue
                    prev_row = asset_mkt[asset_mkt["date"] == dates[i - 1]]
                    curr_row = asset_mkt[asset_mkt["date"] == date]
                    if prev_row.empty or curr_row.empty:
                        continue
                    prev_price = float(prev_row["close"].iloc[0])
                    curr_price = float(curr_row["close"].iloc[0])
                    if prev_price > 0:
                        period_ret += weight * (curr_price / prev_price - 1.0)

                nav[date] = prev_nav * (1.0 + period_ret)
            else:
                nav[date] = 1.0

        return nav.sort_index() if len(nav) > 1 else None
    except Exception as exc:
        logger.debug("构建因子NAV失败: %s", exc)
        return None
