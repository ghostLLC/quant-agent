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

                # ---- 合并并保存增强报告 ----
                enhanced = {
                    "numerical": report_dict,
                    "narrative": narrative,
                    "generated_at": datetime.now().isoformat(),
                }
                self._save_enhanced_report(enhanced, output_dir, entry.factor_spec.factor_id)

                reports.append(output_dir)
            except Exception as exc:
                logger.warning(
                    "因子 %s 报告生成失败: %s",
                    entry.factor_spec.factor_id, exc,
                )

        return reports

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
        if md_path.exists() and enhanced.get("narrative", {}).get("executive_summary"):
            narrative = enhanced["narrative"]
            existing = md_path.read_text(encoding="utf-8")
            enhanced_md = self._inject_narrative_into_markdown(existing, narrative)
            md_path.write_text(enhanced_md, encoding="utf-8")

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
