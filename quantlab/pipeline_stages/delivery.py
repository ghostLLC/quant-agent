"""阶段 7-8: 交付筛选 + 纸交易 + 报告生成。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
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
                factor_slice = factor_panel[factor_panel.index.get_level_values("date") == latest_date] if latest_date else factor_panel.iloc[-len(latest):]
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

                orders = OrderManager(broker).rebalance(target_weights, prices, reason=f"factor={fid}")
                account = broker.get_account()
                accounts.append({
                    "factor_id": fid, "account": account.to_dict(), "orders": len(orders),
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

        return {"status": "success", "accounts": len(accounts), "factor_ids": [a["factor_id"] for a in accounts]}


class DeliveryReportStage(PipelineStage):
    """生成交付报告。"""
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

        for entry in library:
            if entry.factor_spec.factor_id not in deliverable_ids:
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
