from __future__ import annotations

from itertools import product

import pandas as pd

from quantlab.backtest.engine import BacktestResult, run_long_only_backtest
from quantlab.config import BacktestConfig
from quantlab.strategies.base import StrategySignalConfig
from quantlab.strategies.ma_cross import generate_ma_signals


def run_parameter_grid(price_df: pd.DataFrame, base_config: BacktestConfig, parameter_grid: dict[str, list]) -> tuple[pd.DataFrame, BacktestResult]:
    keys = list(parameter_grid.keys())
    rows: list[dict] = []
    best_result: BacktestResult | None = None
    best_score = None

    for values in product(*(parameter_grid[key] for key in keys)):
        overrides = dict(zip(keys, values))
        config = BacktestConfig(**{**base_config.__dict__, **overrides})
        signal_config = StrategySignalConfig(
            short_window=config.short_window,
            long_window=config.long_window,
            enable_trend_filter=config.enable_trend_filter,
            trend_window=config.trend_window,
            enable_volatility_filter=config.enable_volatility_filter,
            volatility_window=config.volatility_window,
            max_volatility=config.max_volatility,
        )
        signal_df = generate_ma_signals(price_df, signal_config)
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
