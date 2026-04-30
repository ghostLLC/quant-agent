"""阶段 4: 样本外验证。"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from quantlab.metrics import compute_rank_ic

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class OOSValidationStage(PipelineStage):
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
