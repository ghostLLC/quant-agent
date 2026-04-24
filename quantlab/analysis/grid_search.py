from __future__ import annotations

from dataclasses import asdict
from itertools import product

import pandas as pd

from quantlab.backtest.engine import BacktestResult, run_long_only_backtest
from quantlab.config import BacktestConfig
from quantlab.strategies import get_strategy


def run_parameter_grid(
    price_df: pd.DataFrame,
    base_config: BacktestConfig,
    parameter_grid: dict[str, list],
    strategy_name: str = "ma_cross",
) -> tuple[pd.DataFrame, BacktestResult]:
    strategy = get_strategy(strategy_name)
    keys = list(parameter_grid.keys())
    rows: list[dict] = []
    best_result: BacktestResult | None = None
    best_score = None

    for values in product(*(parameter_grid[key] for key in keys)):
        overrides = dict(zip(keys, values))
        config = BacktestConfig(**{**asdict(base_config), **overrides})
        signal_df = strategy.generate_signals(price_df, config)
        result = run_long_only_backtest(signal_df, config)

        row = {**overrides, **result.metrics}
        rows.append(row)

        score = (result.metrics["annual_return"], result.metrics["sharpe"], -abs(result.metrics["max_drawdown"]))
        if best_score is None or score > best_score:
            best_score = score
            best_result = result

    summary_df = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    if best_result is None:
        raise ValueError("参数扫描未生成任何结果")
    return summary_df, best_result

