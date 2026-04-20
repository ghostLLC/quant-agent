from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quantlab.analysis.grid_search import run_parameter_grid
from quantlab.backtest.engine import BacktestResult
from quantlab.config import BacktestConfig
from quantlab.strategies.base import StrategySignalConfig
from quantlab.strategies.ma_cross import generate_ma_signals


def split_train_test(price_df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.3 <= train_ratio <= 0.9:
        raise ValueError("train_ratio 需要位于 0.3 到 0.9 之间")
    split_index = int(len(price_df) * train_ratio)
    split_index = max(30, min(split_index, len(price_df) - 30))
    train_df = price_df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = price_df.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def _signal_config_from_backtest(config: BacktestConfig) -> StrategySignalConfig:
    return StrategySignalConfig(
        short_window=config.short_window,
        long_window=config.long_window,
        enable_trend_filter=config.enable_trend_filter,
        trend_window=config.trend_window,
        enable_volatility_filter=config.enable_volatility_filter,
        volatility_window=config.volatility_window,
        max_volatility=config.max_volatility,
    )


def run_train_test_validation(
    price_df: pd.DataFrame,
    base_config: BacktestConfig,
    parameter_grid: dict[str, list],
    backtest_runner,
) -> dict:
    train_df, test_df = split_train_test(price_df, base_config.train_ratio)
    train_summary, best_train_result = run_parameter_grid(train_df, base_config, parameter_grid)
    best_params = train_summary.iloc[0][list(parameter_grid.keys())].to_dict()

    tuned_config = BacktestConfig(**{**asdict(base_config), **best_params})
    test_signal = generate_ma_signals(test_df, _signal_config_from_backtest(tuned_config))
    test_result: BacktestResult = backtest_runner(test_signal, tuned_config)

    baseline_signal = generate_ma_signals(test_df, _signal_config_from_backtest(base_config))
    baseline_test_result: BacktestResult = backtest_runner(baseline_signal, base_config)

    overview = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": train_df["date"].min().strftime("%Y-%m-%d"),
        "train_end": train_df["date"].max().strftime("%Y-%m-%d"),
        "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
        "test_end": test_df["date"].max().strftime("%Y-%m-%d"),
    }

    return {
        "overview": overview,
        "train_summary": train_summary,
        "best_params": best_params,
        "train_best_result": best_train_result,
        "test_result": test_result,
        "baseline_test_result": baseline_test_result,
    }
