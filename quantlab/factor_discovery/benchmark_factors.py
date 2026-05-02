"""Benchmark Factor Registry —— 20+ academic/industry factor benchmarks expressed in Block DSL.

Categories:
- Value (5): Book/P, E/P, Low PB, Value Composite, Sales Proxy
- Momentum (4): 12m1m, 6m, 1m, 3m price momentum
- Size (3): Small cap (log market cap), float cap, volume proxy
- Quality (3): EP stability, price stability, turnover stability
- Low Volatility (3): 60d inverse vol, 20d inverse vol, inverse range
- Liquidity (2): Turnover-based, Amihud illiquidity
- A-Share Specific (3): Retail flow proxy, state ownership effect, small board premium

Each benchmark is expressed as a composable Block tree using factory functions
from blocks.py. BlockExecutor evaluates them; compute_rank_ic scores them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantlab.factor_discovery.blocks import (
    Block,
    BlockExecutor,
    combine,
    data,
    filter_block,
    relational,
    transform,
)

logger = logging.getLogger(__name__)

# ── Index-last variants that don't exist in Block DSL ──
# We use combine("sub", ..., ...) to negate where needed.


def _negate(block: Block) -> Block:
    """Negate a block: -1 * block."""
    return combine("mul", transform("constant", block, value=-1), block)


def _inverse(block: Block) -> Block:
    """Invert a block: 1 / block."""
    return combine("div", transform("constant", block, value=1), block)


def _rank(block: Block) -> Block:
    """Cross-sectional percentile rank."""
    return transform("rank", block)


# ── Benchmark Registry ───────────────────────────────────────────

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════════
    # Value (5)
    # ═══════════════════════════════════════════════════════════════
    "Book_to_Price": {
        "category": "value",
        "description": "close / pb approximates book value per share since PB = P/B",
        "direction": "higher_is_better",
        "data_fields": ["close", "pb"],
        "block": _rank(combine("div", data("close"), data("pb"))),
    },
    "Earnings_Yield": {
        "category": "value",
        "description": "1 / PE = earnings yield, higher E/P = cheaper",
        "direction": "higher_is_better",
        "data_fields": ["pe"],
        "block": _rank(_inverse(data("pe"))),
    },
    "Low_PB": {
        "category": "value",
        "description": "1 / PB, higher = lower price-to-book = cheaper",
        "direction": "higher_is_better",
        "data_fields": ["pb"],
        "block": _rank(_inverse(data("pb"))),
    },
    "Value_Composite": {
        "category": "value",
        "description": "Average cross-sectional rank of E/P and B/P",
        "direction": "higher_is_better",
        "data_fields": ["pe", "pb"],
        "block": _rank(combine(
            "add",
            transform("rank", _inverse(data("pe"))),
            transform("rank", _inverse(data("pb"))),
        )),
    },
    "Sales_Yield_Proxy": {
        "category": "value",
        "description": "turnover / market_cap as a crude sales-to-price proxy (higher turnover relative to size = higher activity per unit value)",
        "direction": "higher_is_better",
        "data_fields": ["turnover", "market_cap"],
        "block": _rank(combine("div", data("turnover"), data("market_cap"))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Momentum (4)
    # ═══════════════════════════════════════════════════════════════
    "Momentum_12m1m": {
        "category": "momentum",
        "description": "20-day price change (proxy for 12-month momentum excluding last month)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(transform("delta", data("close"), window=20)),
    },
    "Momentum_6m": {
        "category": "momentum",
        "description": "60-day price change approximating 6-month momentum",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(transform("delta", data("close"), window=60)),
    },
    "Momentum_1m": {
        "category": "momentum",
        "description": "5-day price change (short-term momentum / reversal boundary)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(transform("delta", data("close"), window=5)),
    },
    "Momentum_3m": {
        "category": "momentum",
        "description": "40-day price change approximating 3-month momentum",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(transform("delta", data("close"), window=40)),
    },

    # ═══════════════════════════════════════════════════════════════
    # Size (3)
    # ═══════════════════════════════════════════════════════════════
    "Size_Small": {
        "category": "size",
        "description": "Negative log market cap — smaller firms rank higher (Fama-French SMB)",
        "direction": "higher_is_better",
        "data_fields": ["market_cap"],
        "block": _rank(_negate(transform("log", data("market_cap")))),
    },
    "Size_FloatCap": {
        "category": "size",
        "description": "Negative log float market cap — free-float size effect",
        "direction": "higher_is_better",
        "data_fields": ["float_market_cap"],
        "block": _rank(_negate(transform("log", data("float_market_cap")))),
    },
    "Size_Volume": {
        "category": "size",
        "description": "Negative log trading volume — liquidity-size proxy",
        "direction": "higher_is_better",
        "data_fields": ["volume"],
        "block": _rank(_negate(transform("log", data("volume")))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Quality (3)
    # ═══════════════════════════════════════════════════════════════
    "Quality_EP_Stable": {
        "category": "quality",
        "description": "Z-score stabilized earnings yield — consistent profitability",
        "direction": "higher_is_better",
        "data_fields": ["pe"],
        "block": _rank(transform("zscore", _inverse(data("pe")))),
    },
    "Quality_Price_Stability": {
        "category": "quality",
        "description": "Inverse of 60-day price volatility — stable earners",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(_inverse(transform("ts_std", data("close"), window=60))),
    },
    "Quality_Margin_Proxy": {
        "category": "quality",
        "description": "close / turnover as a rough margin-to-activity proxy",
        "direction": "higher_is_better",
        "data_fields": ["close", "turnover"],
        "block": _rank(combine("div", data("close"), data("turnover"))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Low Volatility (3)
    # ═══════════════════════════════════════════════════════════════
    "Low_Vol_60d": {
        "category": "low_volatility",
        "description": "Inverse of 60-day return volatility — low-vol anomaly (Ang et al. 2006)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(_inverse(transform("ts_std", data("close"), window=60))),
    },
    "Low_Vol_20d": {
        "category": "low_volatility",
        "description": "Inverse of 20-day return volatility — shorter lookback variant",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(_inverse(transform("ts_std", data("close"), window=20))),
    },
    "Low_Vol_Daily_Range": {
        "category": "low_volatility",
        "description": "Inverse of absolute 5-day return — low daily range stocks",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(_inverse(transform("abs", transform("delta", data("close"), window=5)))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Liquidity (2)
    # ═══════════════════════════════════════════════════════════════
    "Turnover_Liquidity": {
        "category": "liquidity",
        "description": "Cross-sectional turnover rank — higher turnover = more liquid",
        "direction": "higher_is_better",
        "data_fields": ["turnover"],
        "block": _rank(data("turnover")),
    },
    "Amihud_Illiquidity": {
        "category": "liquidity",
        "description": "|daily return| / volume — Amihud (2002) illiquidity measure (higher = less liquid = premium)",
        "direction": "higher_is_better",
        "data_fields": ["close", "volume"],
        "block": _rank(combine(
            "div",
            transform("abs", combine("div", transform("delta", data("close"), window=1), data("close"))),
            data("volume"),
        )),
    },

    # ═══════════════════════════════════════════════════════════════
    # A-Share Specific (3)
    # ═══════════════════════════════════════════════════════════════
    "A_Share_Retail_Flow": {
        "category": "a_share_specific",
        "description": "turnover / 20d avg turnover — abnormal retail-driven volume proxy for A-share market",
        "direction": "higher_is_better",
        "data_fields": ["turnover"],
        "block": _rank(combine(
            "div",
            data("turnover"),
            transform("ts_mean", data("turnover"), window=20),
        )),
    },
    "A_Share_State_Ownership": {
        "category": "a_share_specific",
        "description": "float_market_cap / market_cap — lower free-float ratio suggests higher state/strategic ownership",
        "direction": "higher_is_better",
        "data_fields": ["float_market_cap", "market_cap"],
        "block": _rank(combine("div", data("float_market_cap"), data("market_cap"))),
    },
    "A_Share_Limit_Impact": {
        "category": "a_share_specific",
        "description": "20d max close / 20d min close — range ratio captures A-share limit-up/down effects",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine(
            "div",
            transform("ts_max", data("close"), window=20),
            transform("ts_min", data("close"), window=20),
        )),
    },

    # ═══════════════════════════════════════════════════════════════
    # A-Share Specific (expanded: +8)
    # ═══════════════════════════════════════════════════════════════
    "A_Share_Limit_Up_Proximity": {
        "category": "a_share_specific",
        "description": "Close distance to 20d high — proximity to recent high signals limit-up momentum in A-shares",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("div", data("close"), transform("ts_max", data("close"), window=20))),
    },
    "A_Share_Limit_Down_Risk": {
        "category": "a_share_specific",
        "description": "Close distance from 20d low — farther from low = lower limit-down risk",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("sub", data("close"), transform("ts_min", data("close"), window=20))),
    },
    "A_Share_ST_Proxy": {
        "category": "a_share_specific",
        "description": "Extreme PB proxy for distressed/ST stocks — very low PB signals potential delisting risk in A-shares",
        "direction": "higher_is_better",
        "data_fields": ["pb"],
        "block": _rank(transform("clip", data("pb"), lo=0.1, hi=10.0)),
    },
    "A_Share_Vol_Price_Coupling": {
        "category": "a_share_specific",
        "description": "20d corr(volume, |return|) — volume-return coupling signals retail-driven momentum in A-shares",
        "direction": "higher_is_better",
        "data_fields": ["close", "volume"],
        "block": _rank(transform(
            "ts_corr",
            data("volume"),
            window=20,
            _right_block=transform("abs", combine("div", transform("delta", data("close"), window=1), data("close"))).to_dict(),
        )),
    },
    "A_Share_Gap_Up_Frequency": {
        "category": "a_share_specific",
        "description": "Count of days where open > prev close * 1.02 in 20d — gap-up frequency signals bullish sentiment",
        "direction": "higher_is_better",
        "data_fields": ["open", "close"],
        "block": _rank(transform("ts_sum",
            transform("piecewise",
                combine("sub", combine("div", data("open"), transform("lag", data("close"), window=1)),
                       transform("constant", data("close"), value=1.02)),
                threshold=0,
            ), window=20)),
    },
    "A_Share_Overnight_Return": {
        "category": "a_share_specific",
        "description": "open / prev close - 1 — overnight gap captures A-share retail sentiment and policy news impact",
        "direction": "higher_is_better",
        "data_fields": ["open", "close"],
        "block": _rank(combine("sub",
            combine("div", data("open"), transform("lag", data("close"), window=1)),
            transform("constant", data("close"), value=1.0))),
    },
    "A_Share_Small_Cap_Effect": {
        "category": "a_share_specific",
        "description": "Negative log market cap — A-share small-cap premium is historically strong (shell value, retail preference)",
        "direction": "higher_is_better",
        "data_fields": ["market_cap"],
        "block": _rank(transform("log", transform("clip", data("market_cap"), lo=1e8, hi=1e13))),
    },
    "A_Share_Shell_Value_Proxy": {
        "category": "a_share_specific",
        "description": "Extreme small market cap + low PB composite — proxy for shell/restructuring value in A-shares",
        "direction": "higher_is_better",
        "data_fields": ["market_cap", "pb"],
        "block": _rank(combine("add",
            _negate(transform("log", transform("clip", data("market_cap"), lo=1e8, hi=1e13))),
            _negate(transform("clip", data("pb"), lo=0.1, hi=10.0)),
        )),
    },

    # ═══════════════════════════════════════════════════════════════
    # Momentum (expanded: +4)
    # ═══════════════════════════════════════════════════════════════
    "Momentum_Overnight_20d": {
        "category": "momentum",
        "description": "Average overnight return over 20 days — isolates overnight momentum from intraday noise",
        "direction": "higher_is_better",
        "data_fields": ["open", "close"],
        "block": _rank(transform("ts_mean",
            combine("sub", combine("div", data("open"), transform("lag", data("close"), window=1)),
                   transform("constant", data("close"), value=1.0)),
            window=20)),
    },
    "Momentum_Risk_Adjusted_60d": {
        "category": "momentum",
        "description": "60d return / 60d std — Sharpe-style momentum (risk-adjusted trend)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("div",
            combine("div", transform("delta", data("close"), window=60), transform("lag", data("close"), window=60)),
            transform("ts_std", combine("div", transform("delta", data("close"), window=1), data("close")), window=60),
        )),
    },
    "Momentum_High_Low_20d": {
        "category": "momentum",
        "description": "(close - 20d low) / (20d high - 20d low) — stochastic momentum (not pure price trend)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("div",
            combine("sub", data("close"), transform("ts_min", data("close"), window=20)),
            combine("sub", transform("ts_max", data("close"), window=20), transform("ts_min", data("close"), window=20)),
        )),
    },
    "Momentum_Volume_Confirmed_20d": {
        "category": "momentum",
        "description": "20d return * (volume / 20d avg volume) — volume-confirmed momentum",
        "direction": "higher_is_better",
        "data_fields": ["close", "volume"],
        "block": _rank(combine("mul",
            combine("div", transform("delta", data("close"), window=20), transform("lag", data("close"), window=20)),
            combine("div", data("volume"), transform("ts_mean", data("volume"), window=20)),
        )),
    },

    # ═══════════════════════════════════════════════════════════════
    # Reversal (expanded: +3)
    # ═══════════════════════════════════════════════════════════════
    "Reversal_Overnight_5d": {
        "category": "reversal",
        "description": "-1 * (5d overnight return avg) — overnight gap reversal (gap-up stocks tend to revert intraday)",
        "direction": "higher_is_better",
        "data_fields": ["open", "close"],
        "block": _negate(_rank(transform("ts_mean",
            combine("sub", combine("div", data("open"), transform("lag", data("close"), window=1)),
                   transform("constant", data("close"), value=1.0)),
            window=5))),
    },
    "Reversal_Extreme_Loser_20d": {
        "category": "reversal",
        "description": "Select stocks in bottom 20% of 20d return — extreme loser reversal effect",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _negate(_rank(combine(
            "div", transform("delta", data("close"), window=20),
            transform("lag", data("close"), window=20),
        ))),
    },
    "Reversal_Vol_Normalized_5d": {
        "category": "reversal",
        "description": "-5d return / 20d std — volatility-normalised short-term reversal",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _negate(_rank(combine("div",
            combine("div", transform("delta", data("close"), window=5), transform("lag", data("close"), window=5)),
            transform("ts_std", combine("div", transform("delta", data("close"), window=1), data("close")), window=20),
        ))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Behavioral Finance (+3)
    # ═══════════════════════════════════════════════════════════════
    "Behavioral_52_Week_High": {
        "category": "behavioral",
        "description": "close / 252d max close — George & Hwang (2004) 52-week high anchoring effect",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("div", data("close"), transform("ts_max", data("close"), window=252))),
    },
    "Behavioral_52_Week_Low": {
        "category": "behavioral",
        "description": "close / 252d min close — distance from 52-week low (anchoring from below)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(combine("div", data("close"), transform("ts_min", data("close"), window=252))),
    },
    "Behavioral_Disposition_Proxy": {
        "category": "behavioral",
        "description": "20d turnover / 60d return — high turnover on gains signals disposition effect selling pressure",
        "direction": "lower_is_better",
        "data_fields": ["turnover", "close"],
        "block": _negate(_rank(combine("div",
            transform("ts_mean", data("turnover"), window=20),
            combine("div", transform("delta", data("close"), window=60), transform("lag", data("close"), window=60)),
        ))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Event-Driven (+2)
    # ═══════════════════════════════════════════════════════════════
    "Event_PEAD_Proxy": {
        "category": "event_driven",
        "description": "|5d return - 20d return| — return discontinuity proxies for earnings/news surprise (PEAD effect)",
        "direction": "higher_is_better",
        "data_fields": ["close"],
        "block": _rank(transform("abs", combine("sub",
            combine("div", transform("delta", data("close"), window=5), transform("lag", data("close"), window=5)),
            combine("div", transform("delta", data("close"), window=20), transform("lag", data("close"), window=20)),
        ))),
    },
    "Event_Gap_Reversal": {
        "category": "event_driven",
        "description": "Gap-up days (>2% open vs prev close) followed by reversal — event-driven mean reversion",
        "direction": "higher_is_better",
        "data_fields": ["open", "close"],
        "block": _negate(_rank(transform("ts_mean",
            transform("piecewise",
                combine("sub", combine("div", data("open"), transform("lag", data("close"), window=1)),
                       transform("constant", data("close"), value=1.02)),
                threshold=0,
            ), window=5))),
    },

    # ═══════════════════════════════════════════════════════════════
    # Liquidity (expanded: +1)
    # ═══════════════════════════════════════════════════════════════
    "Liquidity_Pastor_Stambaugh_Proxy": {
        "category": "liquidity",
        "description": "sign(5d return) * 5d volume change — Pastor-Stambaugh (2003) liquidity beta proxy (order flow sensitivity)",
        "direction": "higher_is_better",
        "data_fields": ["close", "volume"],
        "block": _rank(combine("mul",
            transform("sign", combine("div", transform("delta", data("close"), window=5), transform("lag", data("close"), window=5))),
            combine("div", transform("delta", data("volume"), window=5), transform("lag", data("volume"), window=5)),
        )),
    },
}

# Total: 44 benchmark factors across 8 categories


class BenchmarkFactorRegistry:
    """Registry of academic/industry benchmark factors expressed in Block DSL.

    Each benchmark is a composable Block tree that can be evaluated against
    any market DataFrame via BlockExecutor, then scored via compute_rank_ic.

    Usage::

        registry = BenchmarkFactorRegistry()
        results = registry.evaluate_all(market_df)
        corr = registry.compare_to_benchmarks({"my_factor": panel}, market_df)
        registry.bootstrap_benchmarks(store, market_df)
    """

    def __init__(self) -> None:
        self._registry = dict(BENCHMARK_REGISTRY)

    # ── Core API ─────────────────────────────────────────────────

    def get_benchmark_names(self) -> list[str]:
        """Return all registered benchmark factor names."""
        return list(self._registry.keys())

    def get_benchmarks_by_category(self) -> dict[str, list[str]]:
        """Group benchmark names by category."""
        grouped: dict[str, list[str]] = {}
        for name, meta in self._registry.items():
            cat = meta.get("category", "unknown")
            grouped.setdefault(cat, []).append(name)
        return grouped

    def get_block(self, name: str) -> Block:
        """Return the Block tree for a named benchmark."""
        if name not in self._registry:
            raise KeyError(f"未知基准因子: {name}")
        return self._registry[name]["block"]

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate_all(
        self,
        market_df: pd.DataFrame,
        forward_days: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Compute every benchmark factor and return IC results.

        Args:
            market_df: Market data with date, asset, and required data fields.
            forward_days: Forward return horizon for IC calculation.

        Returns:
            Dict of benchmark_name → {rank_ic_mean, ic_ir, coverage, factor_panel, ...}
        """
        from quantlab.metrics import compute_rank_ic

        executor = BlockExecutor(date_col="date", asset_col="asset")
        results: dict[str, dict[str, Any]] = {}

        for name, meta in self._registry.items():
            try:
                block = meta["block"]
                required = set(meta.get("data_fields", []))
                available = set(market_df.columns)
                missing = required - available
                if missing:
                    logger.debug("基准 %s 缺少数据字段 %s，跳过", name, missing)
                    results[name] = {
                        "rank_ic_mean": 0.0,
                        "ic_ir": 0.0,
                        "coverage": 0.0,
                        "error": f"missing_fields: {sorted(missing)}",
                    }
                    continue

                factor_values = executor.execute(block, market_df)
                ic_data = compute_rank_ic(
                    factor_values, market_df, forward_days=forward_days,
                )
                results[name] = {
                    "rank_ic_mean": ic_data.get("rank_ic_mean", 0.0),
                    "ic_ir": ic_data.get("ic_ir", 0.0),
                    "coverage": ic_data.get("coverage", 0.0),
                    "factor_panel": factor_values,
                    "category": meta.get("category", ""),
                }
            except Exception as exc:
                logger.warning("基准 %s 计算失败: %s", name, exc)
                results[name] = {
                    "rank_ic_mean": 0.0,
                    "ic_ir": 0.0,
                    "coverage": 0.0,
                    "error": str(exc)[:200],
                }

        return results

    # ── Comparison ───────────────────────────────────────────────

    def compare_to_benchmarks(
        self,
        discovered_factors_panels: dict[str, pd.Series | pd.DataFrame],
        market_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Compute correlation matrix between discovered factors and benchmarks.

        Merges each discovered factor panel with each benchmark panel on (date, asset),
        then computes Pearson correlation between the aligned factor value columns.

        Args:
            discovered_factors_panels: {factor_id → Series or DataFrame with factor_value}
            market_df: Market data used to compute benchmarks.

        Returns:
            {factor_id → {benchmark_name → correlation}}
        """
        benchmarks = self.evaluate_all(market_df)
        correlation_matrix: dict[str, dict[str, float]] = {}

        for fid, panel in discovered_factors_panels.items():
            discovered_values = _extract_factor_column(panel)
            if discovered_values is None:
                correlation_matrix[fid] = {bm: 0.0 for bm in self._registry}
                continue

            row: dict[str, float] = {}
            for bm_name, bm_result in benchmarks.items():
                bm_panel = bm_result.get("factor_panel")
                if bm_panel is None:
                    row[bm_name] = 0.0
                    continue
                row[bm_name] = _correlate_panels(discovered_values, bm_panel)
            correlation_matrix[fid] = row

        return correlation_matrix

    # ── Bootstrap ────────────────────────────────────────────────

    def bootstrap_benchmarks(
        self,
        store: Any,
        market_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Inject all benchmark factors into a PersistentFactorStore as reference factors.

        Each benchmark is upserted as a FactorLibraryEntry with OBSERVE status
        so it appears in the factor library alongside discovered factors.

        Args:
            store: PersistentFactorStore instance.
            market_df: Market data for computing initial factor panels.

        Returns:
            {injected_count, benchmark_ids, ic_results}
        """
        from quantlab.factor_discovery.models import (
            FactorDirection,
            FactorEvaluationReport,
            FactorScorecard,
            FactorSpec,
            FactorStatus,
        )
        from quantlab.factor_discovery.runtime import FactorLibraryEntry

        existing_ids = {e.factor_spec.factor_id for e in store.load_library_entries()}

        direction_map = {
            "higher_is_better": FactorDirection.HIGHER_IS_BETTER,
            "lower_is_better": FactorDirection.LOWER_IS_BETTER,
        }

        executor = BlockExecutor(date_col="date", asset_col="asset")
        injected: list[str] = []
        ic_results: dict[str, dict[str, float]] = {}

        for name, meta in self._registry.items():
            bm_id = f"benchmark_{name.lower()}"
            if bm_id in existing_ids:
                logger.debug("基准 %s 已存在，跳过", name)
                continue

            try:
                block = meta["block"]
                factor_values = executor.execute(block, market_df)

                spec = FactorSpec(
                    factor_id=bm_id,
                    name=name,
                    version="v1_benchmark",
                    description=meta.get("description", ""),
                    hypothesis=meta.get("description", ""),
                    family=meta.get("category", "benchmark"),
                    direction=direction_map.get(
                        meta.get("direction", "higher_is_better"),
                        FactorDirection.UNKNOWN,
                    ),
                    status=FactorStatus.OBSERVE,
                    tags=["benchmark", "academic", meta.get("category", "")],
                    source="benchmark_registry",
                    created_from="benchmark",
                )

                entry = FactorLibraryEntry(
                    factor_spec=spec,
                    latest_report=FactorEvaluationReport(
                        report_id=f"bm_{bm_id}",
                        factor_spec=spec,
                        scorecard=FactorScorecard(),
                    ),
                    retention_reason=meta.get("description", ""),
                    semantic_name=name,
                    version="1.0",
                )
                store.upsert_library_entry(entry, factor_panel=factor_values)

                from quantlab.metrics import compute_rank_ic
                ic_data = compute_rank_ic(factor_values, market_df)
                ic_results[bm_id] = ic_data

                injected.append(bm_id)
                logger.info("基准注入: %s (%s) IC=%.4f", bm_id, name, ic_data.get("rank_ic_mean", 0.0))

            except Exception as exc:
                logger.warning("基准 %s 注入失败: %s", name, exc)

        return {
            "injected_count": len(injected),
            "benchmark_ids": injected,
            "ic_results": ic_results,
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the benchmark registry."""
        by_cat = self.get_benchmarks_by_category()
        return {
            "total_benchmarks": len(self._registry),
            "categories": {cat: len(names) for cat, names in by_cat.items()},
            "names_by_category": by_cat,
        }


# ── Internal helpers ────────────────────────────────────────────

def _extract_factor_column(panel: pd.Series | pd.DataFrame) -> pd.Series | None:
    """Extract a factor value Series from a panel, normalizing index to (date, asset)."""
    if isinstance(panel, pd.Series):
        return panel
    if isinstance(panel, pd.DataFrame):
        for col in ("factor_value", "factor", "factor_val"):
            if col in panel.columns:
                sub = panel[["date", "asset", col]].copy()
                sub = sub.dropna(subset=["date", "asset"])
                sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                sub["asset"] = sub["asset"].astype(str)
                return sub.set_index(["date", "asset"])[col]
        # Fallback: use last numeric column
        numeric = panel.select_dtypes(include=[np.number])
        if not numeric.empty and "date" in panel.columns and "asset" in panel.columns:
            col = numeric.columns[-1]
            sub = panel[["date", "asset", col]].copy()
            sub = sub.dropna(subset=["date", "asset"])
            sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
            sub["asset"] = sub["asset"].astype(str)
            return sub.set_index(["date", "asset"])[col]
    return None


def _correlate_panels(a: pd.Series, b: pd.Series | pd.DataFrame) -> float:
    """Compute Pearson correlation between two factor value Series after aligning on index."""
    try:
        b_values = _extract_factor_column(b) if not isinstance(b, pd.Series) else b
        if b_values is None:
            return 0.0

        # Align both to common MultiIndex
        a_aligned = a if isinstance(a.index, pd.MultiIndex) else a
        b_aligned = b_values if isinstance(b_values.index, pd.MultiIndex) else b_values

        merged = pd.DataFrame({"a": a_aligned, "b": b_aligned}).dropna()
        if len(merged) < 20:
            return 0.0
        if merged["a"].nunique(dropna=True) <= 1 or merged["b"].nunique(dropna=True) <= 1:
            return 0.0
        corr = merged[["a", "b"]].corr().iloc[0, 1]
        return round(float(corr), 4) if pd.notna(corr) else 0.0
    except Exception:
        return 0.0
