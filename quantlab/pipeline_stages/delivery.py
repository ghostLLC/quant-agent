"""阶段 7-8: 交付筛选 + 纸交易 + Agent 驱动报告生成。

DeliveryScreeningStage / PaperTradingStage: 保持原有逻辑。
AgentDeliveryReportStage: 保留数值报告生成，新增 LLM 叙事报告
（执行摘要、因子故事、优劣势、市况适配、买方清单）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .base import DATA_DIR, PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class DeliveryScreeningStage(PipelineStage):
    """交付标准筛选。"""
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.delivery_screener import DeliveryScreener
        screener = DeliveryScreener(data_path=ctx.data_path)
        return screener.screen()


class PaperTradingStage(PipelineStage):
    """为可交付因子启动纸交易。"""
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        deliverable_ids = ctx._meta.get("deliverable_factor_ids", [])
        if not deliverable_ids:
            return {"status": "skipped", "reason": "无可交付因子"}

        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor
        from quantlab.trading.broker import PaperBroker, OrderManager

        market_df = ctx.load_data()
        if market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        store = PersistentFactorStore()
        executor = SafeFactorExecutor()
        library = store.load_library_entries()
        accounts = []

        output_dir = DATA_DIR / "assistant_data" / "paper_trading"
        output_dir.mkdir(parents=True, exist_ok=True)

        for entry in library:
            fid = entry.factor_spec.factor_id
            if fid not in deliverable_ids:
                continue
            try:
                computed = executor.execute(entry.factor_spec, market_df)
                factor_panel = computed.get("factor_panel")
                if factor_panel is None or len(factor_panel) == 0:
                    continue

                latest_date = market_df["date"].max() if "date" in market_df.columns else ""
                latest = market_df[market_df["date"] == latest_date] if latest_date else market_df
                prices: dict[str, float] = {}
                if "close" in latest.columns and "asset" in latest.columns:
                    for _, row in latest.iterrows():
                        prices[str(row["asset"])] = float(row["close"])

                broker = PaperBroker(initial_cash=1_000_000, account_id=f"paper_{fid}")
                broker.update_prices(prices)

                target_weights: dict[str, float] = {}
                factor_slice = (
                    factor_panel[factor_panel.index.get_level_values("date") == latest_date]
                    if latest_date
                    else factor_panel.iloc[-len(latest):]
                )
                if len(factor_slice) > 0:
                    ranked = factor_slice.rank(pct=True)
                    for asset, val in ranked.items():
                        if isinstance(asset, tuple):
                            asset = str(asset[1]) if len(asset) > 1 else str(asset[0])
                        if val > 0.0 and not pd.isna(val):
                            target_weights[str(asset)] = float(val)

                total_w = sum(target_weights.values())
                if total_w > 0:
                    target_weights = {a: w / total_w for a, w in target_weights.items()}

                orders = OrderManager(broker).rebalance(
                    target_weights, prices, reason=f"factor={fid}"
                )
                account = broker.get_account()
                accounts.append({
                    "factor_id": fid,
                    "account": account.to_dict(),
                    "orders": len(orders),
                })
            except Exception as exc:
                logger.warning("纸交易 %s 失败: %s", fid, exc)

        if accounts:
            log_path = output_dir / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path.write_text(
                json.dumps(accounts, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("纸交易: %d 个账户已启动, 日志: %s", len(accounts), log_path)

        return {
            "status": "success",
            "accounts": len(accounts),
            "factor_ids": [a["factor_id"] for a in accounts],
        }


class AgentDeliveryReportStage(PipelineStage):
    """Agent 驱动的交付报告生成。

    数值层: FactorDeliveryReportGenerator 生成完整数值报告（IC/绩效/容量/正交性等）。
    叙事层: AgentAnalyst 基于数值报告生成 LLM 研究叙事。
    输出: 合并数值报告 + 叙事报告，保存为 JSON + Markdown + HTML。
    """

    def __init__(self, enable_agent: bool = True, agent_timeout: int = 90) -> None:
        self.enable_agent = enable_agent
        self.agent_timeout = agent_timeout

    def run(self, ctx: PipelineContext) -> list[str]:
        deliverable_ids = ctx._meta.get("deliverable_factor_ids", [])
        if not deliverable_ids:
            return []

        from quantlab.factor_discovery.factor_report import FactorDeliveryReportGenerator
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        hub = ctx.load_data()
        if hub.empty:
            return []

        generator = FactorDeliveryReportGenerator()
        executor = SafeFactorExecutor()
        library = store.load_library_entries()
        reports = []

        # 构建因子库背景（供 Agent 分析 peer comparison 使用）
        library_context = self._build_library_context(library, deliverable_ids)

        # 获取异常检测结果（来自 DataRefreshStage）
        anomalies = ctx._meta.get("anomalies")

        for entry in library:
            if entry.factor_spec.factor_id not in deliverable_ids:
                continue
            try:
                computed = executor.execute(entry.factor_spec, hub)
                factor_panel = computed["factor_panel"]
                output_dir = str(DATA_DIR / "delivery_reports" / entry.factor_spec.factor_id)

                # ---- 数值报告 ----
                numerical_report = generator.generate(
                    factor_spec=entry.factor_spec,
                    factor_panel=factor_panel,
                    market_df=hub,
                    evaluation_report=entry.latest_report,
                    output_dir=output_dir,
                )
                report_dict = numerical_report.to_dict()

                # ---- 真实收益验证 ----
                real_return_validation = self._run_real_return_validation(
                    factor_id=entry.factor_spec.factor_id,
                    factor_panel=factor_panel,
                    market_df=hub,
                    report_dict=report_dict,
                )

                # ---- Agent 叙事报告 ----
                narrative = {}
                if self.enable_agent:
                    try:
                        from .agent_analyst import AgentAnalyst

                        analyst = AgentAnalyst(timeout=self.agent_timeout)
                        narrative = analyst.generate_narrative_report(
                            report_dict, library_context
                        )
                        logger.info(
                            "Agent 报告叙事完成: factor_id=%s status=%s",
                            entry.factor_spec.factor_id,
                            narrative.get("status", "unknown"),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Agent 报告叙事失败 factor_id=%s: %s",
                            entry.factor_spec.factor_id, exc,
                        )
                        narrative = {"status": "failed", "error": str(exc)[:200]}

                # ---- 基准因子对比 ----
                benchmark_comparison = self._run_benchmark_comparison(
                    factor_id=entry.factor_spec.factor_id,
                    factor_panel=factor_panel,
                    market_df=hub,
                )

                # ---- 合并并保存增强报告 ----
                enhanced = {
                    "numerical": report_dict,
                    "narrative": narrative,
                    "real_return_validation": real_return_validation,
                    "benchmark_comparison": benchmark_comparison,
                    "generated_at": datetime.now().isoformat(),
                }
                # 注入异常检测摘要
                if anomalies:
                    enhanced["anomaly_summary"] = anomalies
                self._save_enhanced_report(enhanced, output_dir, entry.factor_spec.factor_id)

                reports.append(output_dir)
            except Exception as exc:
                logger.warning(
                    "因子 %s 报告生成失败: %s",
                    entry.factor_spec.factor_id, exc,
                )

        return reports

    def _run_real_return_validation(
        self,
        factor_id: str,
        factor_panel: "pd.Series",
        market_df: "pd.DataFrame",
        report_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Run RealReturnEvaluator and compare to IC predictions."""
        try:
            from quantlab.factor_discovery.real_return import (
                RealReturnEvaluator,
                compare_to_ic,
            )

            # Convert factor_panel to DataFrame format for RealReturnEvaluator
            import pandas as pd
            if isinstance(factor_panel, pd.Series):
                fp = factor_panel.reset_index()
                if fp.shape[1] >= 3:
                    fp.columns = ["date", "asset", "factor_value"]
                elif fp.shape[1] == 2:
                    fp.columns = ["asset", "factor_value"]
                fp["date"] = pd.to_datetime(fp["date"])
            elif isinstance(factor_panel, pd.DataFrame):
                fp = factor_panel.copy()
                if "factor_value" not in fp.columns and fp.shape[1] >= 3:
                    val_cols = [c for c in fp.columns if c not in ("date", "asset")]
                    if val_cols:
                        fp = fp.rename(columns={val_cols[0]: "factor_value"})
            else:
                return {"status": "skipped", "reason": "因子面板格式错误"}

            evaluator = RealReturnEvaluator()
            rr_report = evaluator.evaluate(
                factor_panel=fp,
                market_df=market_df,
                n_long=50,
                initial_capital=1e8,
            )
            rr_report.factor_id = factor_id

            # Get IC stats from numerical report
            perf = report_dict.get("performance", {})
            ic_stats = {
                "rank_ic_mean": perf.get("rank_ic_mean", perf.get("ic_mean", 0.0)),
                "coverage": perf.get("coverage", 1.0),
            }

            comparison = compare_to_ic(rr_report, ic_stats)

            logger.info(
                "真实收益验证完成: factor_id=%s net_sharpe=%.3f gross_sharpe=%.3f verdict=%s",
                factor_id, rr_report.net_sharpe, rr_report.gross_sharpe,
                comparison.get("verdict", ""),
            )

            return {
                "status": "success",
                "report": rr_report.to_dict(),
                "ic_comparison": comparison,
            }
        except Exception as exc:
            logger.warning("真实收益验证失败 factor_id=%s: %s", factor_id, exc)
            return {"status": "failed", "error": str(exc)[:200]}

    def _build_real_return_block(self, validation: dict[str, Any]) -> list[str]:
        """Build markdown block for Real Return Validation section."""
        if validation.get("status") != "success":
            return []

        report = validation.get("report", {})
        comparison = validation.get("ic_comparison", {})

        block = ["## 真实收益验证 (Real Return Validation)", ""]

        # Core metrics
        block.append("### 组合回测绩效")
        block.append("")
        block.append(f"- **净夏普**: {report.get('net_sharpe', 0):.3f}")
        block.append(f"- **毛夏普**: {report.get('gross_sharpe', 0):.3f}")
        block.append(f"- **年化净收益**: {report.get('net_return_annual', 0):.2%}")
        block.append(f"- **年化毛收益**: {report.get('gross_return_annual', 0):.2%}")
        block.append(f"- **最大回撤**: {report.get('max_drawdown', 0):.2%}")
        block.append(f"- **年化波动**: {report.get('net_volatility_annual', 0):.2%}")
        block.append("")

        # Cost analysis
        block.append("### 交易成本分析")
        block.append("")
        block.append(f"- **成本侵蚀**: {report.get('cost_drag_pct', 0) * 100:.2f}%/年")
        block.append(f"- **平均换手率**: {report.get('avg_turnover', 0):.2%}")
        block.append(f"- **换手衰减比** (net/gross IC): {report.get('turnover_decay_ratio', 0):.3f}")
        block.append("")

        # IC comparison
        if comparison:
            block.append("### IC 与真实收益对比")
            block.append("")
            block.append(f"- **IC 预测夏普**: {comparison.get('ic_predicted_sharpe', 0):.3f}")
            block.append(f"- **实际夏普**: {comparison.get('actual_sharpe', 0):.3f}")
            block.append(f"- **相关偏差**: {comparison.get('correlation_bias', 0):.3f}")
            block.append(f"- **IC 效率**: {comparison.get('ic_inefficiency', 0):.1%} 的信号在交易中损失")
            block.append(f"- **判定**: {comparison.get('verdict', '')}")
            block.append("")

        # Capacity
        cap = report.get("capacity_estimate", {})
        if cap:
            block.append("### 容量估算")
            block.append("")
            block.append(f"- 日均容量: {cap.get('total_daily_capacity_yuan', 0):.0f} 元")
            block.append(f"- 月均容量: {cap.get('total_monthly_capacity_yuan', 0):.0f} 元")
            block.append(f"- 最大参与率: {cap.get('max_participation_rate', 0):.0%}")
            block.append(f"- 持仓数: {cap.get('max_stocks', 0)}")
            block.append("")

        # Attribution
        attr = report.get("attribution", {})
        if attr:
            block.append("### 收益归因")
            block.append("")
            block.append(f"- Alpha: {attr.get('alpha', 0):.4f}")
            block.append(f"- 市场 Beta: {attr.get('market_beta', 0):.3f}")
            block.append(f"- 残差占比: {attr.get('residual', 0):.1%}")
            block.append("")

        return block

    def _run_benchmark_comparison(
        self,
        factor_id: str,
        factor_panel: "pd.DataFrame",
        market_df: "pd.DataFrame",
    ) -> dict[str, Any]:
        """Compare a factor against the known benchmark factor registry."""
        try:
            from quantlab.factor_discovery.benchmark_factors import BenchmarkFactorRegistry
            registry = BenchmarkFactorRegistry()
            result = registry.compare_to_benchmarks(
                {factor_id: factor_panel}, market_df
            )
            correlations = result.get("correlations", {}).get(factor_id, {})
            top_matches = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            return {
                "status": "success",
                "benchmark_count": result.get("benchmark_count", 0),
                "top_correlations": [
                    {"benchmark": name, "correlation": round(corr, 4)}
                    for name, corr in top_matches
                ],
                "max_abs_correlation": round(result.get("max_abs_correlation", 0), 4),
            }
        except Exception as e:
            logger.debug("基准对比失败 factor_id=%s: %s", factor_id, e)
            return {"status": "skipped", "reason": str(e)[:100]}

    def _build_library_context(
        self, library: list, deliverable_ids: list[str]
    ) -> dict[str, Any]:
        """构建因子库背景信息。"""
        families: dict[str, int] = {}
        total = len(library)
        active = 0
        for entry in library:
            status = str(entry.factor_spec.status)
            if status in ("approved", "paper", "pilot", "live"):
                active += 1
            fam = getattr(entry.factor_spec, "family", "unknown")
            families[fam] = families.get(fam, 0) + 1

        return {
            "total_factors": total,
            "active_factors": active,
            "deliverable_count": len(deliverable_ids),
            "family_distribution": families,
        }

    def _save_enhanced_report(
        self, enhanced: dict[str, Any], output_dir: str, factor_id: str
    ) -> None:
        """保存增强后的报告（数值 + 叙事）。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON（完整）
        json_path = out / f"factor_delivery_{factor_id}_enhanced.json"
        json_path.write_text(
            json.dumps(enhanced, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Markdown（如果叙事存在则增强）
        md_path = out / f"factor_delivery_{factor_id}.md"
        if md_path.exists():
            existing = md_path.read_text(encoding="utf-8")
            modified = False
            # 注入 Agent 叙事
            if enhanced.get("narrative", {}).get("executive_summary"):
                existing = self._inject_narrative_into_markdown(existing, enhanced["narrative"])
                modified = True
            # 注入异常检测摘要
            if enhanced.get("anomaly_summary"):
                existing = self._inject_anomaly_into_markdown(existing, enhanced["anomaly_summary"])
                modified = True
            # 注入真实收益验证
            rr_validation = enhanced.get("real_return_validation", {})
            if rr_validation.get("status") == "success":
                rr_block = self._build_real_return_block(rr_validation)
                if rr_block:
                    existing = existing.rstrip("\n") + "\n\n" + "\n".join(rr_block) + "\n"
                    modified = True
            if modified:
                md_path.write_text(existing, encoding="utf-8")

    def _inject_narrative_into_markdown(
        self, original_md: str, narrative: dict[str, Any]
    ) -> str:
        """将 LLM 叙事注入到 Markdown 报告的开头。"""
        exec_summary = narrative.get("executive_summary", "")
        strengths = narrative.get("strengths", [])
        weaknesses = narrative.get("weaknesses", [])
        factor_story = narrative.get("factor_story", {})
        market_ctx = narrative.get("market_context", {})
        risk_assessment = narrative.get("risk_assessment", {})
        buyer_checklist = narrative.get("buyer_checklist", [])
        peer = narrative.get("peer_comparison", "")

        narrative_block = []

        if exec_summary:
            narrative_block.append("## 执行摘要 (AI 分析)")
            narrative_block.append("")
            narrative_block.append(exec_summary)
            narrative_block.append("")

        if factor_story:
            ei = factor_story.get("economic_intuition", "")
            me = factor_story.get("mechanism_explanation", "")
            if ei:
                narrative_block.append("### 经济学直觉")
                narrative_block.append("")
                narrative_block.append(ei)
                narrative_block.append("")
            if me:
                narrative_block.append("### 机制解释")
                narrative_block.append("")
                narrative_block.append(me)
                narrative_block.append("")

        if strengths:
            narrative_block.append("### 核心优势")
            narrative_block.append("")
            for s in strengths:
                narrative_block.append(f"- {s}")
            narrative_block.append("")

        if weaknesses:
            narrative_block.append("### 需关注的风险")
            narrative_block.append("")
            for w in weaknesses:
                narrative_block.append(f"- {w}")
            narrative_block.append("")

        if market_ctx:
            best = market_ctx.get("best_environments", [])
            worst = market_ctx.get("worst_environments", [])
            suitability = market_ctx.get("current_suitability", "")
            if best or worst:
                narrative_block.append("### 市场环境适配")
                narrative_block.append("")
                if best:
                    narrative_block.append(f"**最佳环境**: {', '.join(best)}")
                if worst:
                    narrative_block.append(f"**最差环境**: {', '.join(worst)}")
                if suitability:
                    narrative_block.append(f"**当前适用性**: {suitability}")
                narrative_block.append("")

        if risk_assessment:
            primary = risk_assessment.get("primary_risk", "")
            secondary = risk_assessment.get("secondary_risks", [])
            if primary:
                narrative_block.append(f"**首要风险**: {primary}")
            if secondary:
                narrative_block.append("**次要风险**:")
                for sr in secondary:
                    narrative_block.append(f"- {sr}")
            narrative_block.append("")

        if peer:
            narrative_block.append("### 同类比较")
            narrative_block.append("")
            narrative_block.append(peer)
            narrative_block.append("")

        if buyer_checklist:
            narrative_block.append("### 买方尽调清单")
            narrative_block.append("")
            for i, item in enumerate(buyer_checklist, 1):
                narrative_block.append(f"{i}. {item}")
            narrative_block.append("")

        if not narrative_block:
            return original_md

        # 将叙事块插入到第一个 "## " 章节标题之前
        narrative_text = "\n".join(narrative_block)
        lines = original_md.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                insert_idx = i
                break

        if insert_idx > 0:
            return "\n".join(lines[:insert_idx]) + "\n" + narrative_text + "\n" + "\n".join(lines[insert_idx:])
        return original_md + "\n" + narrative_text

    @staticmethod
    def _inject_anomaly_into_markdown(
        original_md: str, anomalies: dict[str, Any]
    ) -> str:
        """将异常检测摘要注入 Markdown 报告。"""
        summary = anomalies.get("summary", {})
        if not summary or summary.get("total_anomalies", 0) == 0:
            return original_md  # 无异常，不修改

        lines_block = ["## 数据异常检测 (AnomalyGuard)", ""]
        lines_block.append(f"- **总异常数**: {summary.get('total_anomalies', 0)}")
        lines_block.append(f"- 收盘价缺失: {summary.get('nan_in_close', 0) > 0}")
        lines_block.append(f"- 零成交量资产数: {summary.get('zero_volume_assets', 0)}")
        lines_block.append(f"- 价格跳变资产数: {summary.get('price_gap_assets', 0)}")
        lines_block.append(f"- 重复行数: {summary.get('duplicate_rows', 0)}")
        lines_block.append(f"- 未来日期行数: {summary.get('future_dates', 0)}")
        lines_block.append(f"- 疑似拆股事件: {summary.get('suspected_splits', 0)}")
        lines_block.append(f"- 疑似分红事件: {summary.get('suspected_dividends', 0)}")
        lines_block.append(f"- 停牌资产数: {summary.get('suspended_assets', 0)}")

        # 列出停牌资产
        suspended = anomalies.get("suspensions", [])
        if suspended:
            lines_block.append(f"- 停牌资产列表: {', '.join(suspended[:20])}")
            if len(suspended) > 20:
                lines_block.append(f"  (... 及其他 {len(suspended) - 20} 个资产)")

        lines_block.append("")

        anomaly_text = "\n".join(lines_block)
        lines = original_md.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                insert_idx = i
                break

        if insert_idx > 0:
            return "\n".join(lines[:insert_idx]) + "\n" + anomaly_text + "\n" + "\n".join(lines[insert_idx:])
        return original_md + "\n" + anomaly_text
