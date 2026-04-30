"""阶段 5: 多因子组合。"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from quantlab.factor_discovery.factor_enhancements import FactorCombiner
from quantlab.metrics import compute_rank_ic

from .base import DATA_DIR, PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class CombinationStage(PipelineStage):
    def __init__(self, combiner: FactorCombiner | None = None) -> None:
        self.combiner = combiner or FactorCombiner()

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        entries = store.load_library_entries()
        approved = [e for e in entries if str(e.factor_spec.status) == "approved"]
        if not approved:
            return {"status": "skipped", "reason": "无已审批因子"}

        executor = SafeFactorExecutor()
        market_df = ctx.load_data()
        if market_df.empty:
            return {"status": "skipped", "reason": "数据为空"}

        factor_panels: dict[str, pd.Series] = {}
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


class BenchmarkCompareStage(PipelineStage):
    """等权和市值加权基准比较。"""

    def run(self, ctx: PipelineContext) -> dict[str, float]:
        combination = {}  # populated from context in run_daily
        return _benchmark_compare(ctx, combination)


def _benchmark_compare(ctx: PipelineContext, combination: dict[str, Any]) -> dict[str, Any]:
    market_df = ctx.load_data()
    if market_df.empty:
        return {"status": "skipped", "reason": "数据为空"}

    comb_ic = combination.get("combined_ic", 0.0)

    try:
        if "date" in market_df.columns and "asset" in market_df.columns:
            df = market_df.copy()
            df["ew_factor"] = 1.0
            ew_ic = compute_rank_ic(df["ew_factor"], df)["ic_mean"]
        else:
            ew_ic = 0.0

        mc_col = None
        for col in ["total_mv", "circ_mv", "market_cap"]:
            if col in market_df.columns:
                mc_col = col
                break
        if mc_col and "date" in market_df.columns and "asset" in market_df.columns:
            df = market_df.copy()
            df["mcw_factor"] = df[mc_col].fillna(df[mc_col].median())
            mcw_ic = compute_rank_ic(df["mcw_factor"], df)["ic_mean"]
        else:
            mcw_ic = 0.0

        excess_ew = round(comb_ic - ew_ic, 4)
        excess_mcw = round(comb_ic - mcw_ic, 4)

        return {
            "ew_ic": round(ew_ic, 4),
            "mcw_ic": round(mcw_ic, 4),
            "factor_ic": round(comb_ic, 4),
            "excess_vs_ew": excess_ew,
            "excess_vs_mcw": excess_mcw,
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)[:200]}
