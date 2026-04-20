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


def _run_backtest_with_config(price_df: pd.DataFrame, config: BacktestConfig, backtest_runner) -> BacktestResult:
    signal_df = generate_ma_signals(price_df, _signal_config_from_backtest(config))
    return backtest_runner(signal_df, config)


def _validate_walk_forward_windows(price_df: pd.DataFrame, config: BacktestConfig) -> None:
    min_required = config.walk_forward_train_window + config.walk_forward_test_window
    if config.walk_forward_train_window < 60:
        raise ValueError("walk_forward_train_window 至少需要 60 个交易日")
    if config.walk_forward_test_window < 20:
        raise ValueError("walk_forward_test_window 至少需要 20 个交易日")
    if config.walk_forward_step_size < 20:
        raise ValueError("walk_forward_step_size 至少需要 20 个交易日")
    if len(price_df) < min_required:
        raise ValueError(f"数据长度不足以执行 walk-forward，需要至少 {min_required} 行")


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
    test_result = _run_backtest_with_config(test_df, tuned_config, backtest_runner)
    baseline_test_result = _run_backtest_with_config(test_df, base_config, backtest_runner)

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


def run_walk_forward_validation(
    price_df: pd.DataFrame,
    base_config: BacktestConfig,
    parameter_grid: dict[str, list],
    backtest_runner,
) -> dict:
    _validate_walk_forward_windows(price_df, base_config)

    train_window = base_config.walk_forward_train_window
    test_window = base_config.walk_forward_test_window
    step_size = base_config.walk_forward_step_size

    folds: list[dict] = []
    fold_rows: list[dict] = []
    aggregated_equity_frames: list[pd.DataFrame] = []
    baseline_equity_frames: list[pd.DataFrame] = []

    start_idx = 0
    fold_id = 1

    while start_idx + train_window + test_window <= len(price_df):
        train_df = price_df.iloc[start_idx : start_idx + train_window].copy().reset_index(drop=True)
        test_df = price_df.iloc[start_idx + train_window : start_idx + train_window + test_window].copy().reset_index(drop=True)

        train_summary, best_train_result = run_parameter_grid(train_df, base_config, parameter_grid)
        best_params = train_summary.iloc[0][list(parameter_grid.keys())].to_dict()
        tuned_config = BacktestConfig(**{**asdict(base_config), **best_params})

        test_result = _run_backtest_with_config(test_df, tuned_config, backtest_runner)
        baseline_result = _run_backtest_with_config(test_df, base_config, backtest_runner)

        fold_meta = {
            "fold_id": fold_id,
            "train_start": train_df["date"].min().strftime("%Y-%m-%d"),
            "train_end": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_df["date"].max().strftime("%Y-%m-%d"),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }

        fold_rows.append(
            {
                **fold_meta,
                **best_params,
                "train_annual_return": best_train_result.metrics["annual_return"],
                "train_sharpe": best_train_result.metrics["sharpe"],
                "test_annual_return": test_result.metrics["annual_return"],
                "test_total_return": test_result.metrics["total_return"],
                "test_sharpe": test_result.metrics["sharpe"],
                "test_max_drawdown": test_result.metrics["max_drawdown"],
                "baseline_test_annual_return": baseline_result.metrics["annual_return"],
                "baseline_test_total_return": baseline_result.metrics["total_return"],
                "baseline_test_sharpe": baseline_result.metrics["sharpe"],
                "baseline_test_max_drawdown": baseline_result.metrics["max_drawdown"],
            }
        )

        fold_equity = test_result.equity_curve[["date", "cum_return", "benchmark_return", "drawdown"]].copy()
        fold_equity["fold_id"] = fold_id
        fold_equity["strategy_label"] = "walk_forward_best"
        aggregated_equity_frames.append(fold_equity)

        baseline_equity = baseline_result.equity_curve[["date", "cum_return", "benchmark_return", "drawdown"]].copy()
        baseline_equity["fold_id"] = fold_id
        baseline_equity["strategy_label"] = "baseline"
        baseline_equity_frames.append(baseline_equity)

        folds.append(
            {
                **fold_meta,
                "best_params": best_params,
                "train_summary": train_summary,
                "train_best_result": best_train_result,
                "test_result": test_result,
                "baseline_test_result": baseline_result,
            }
        )

        start_idx += step_size
        fold_id += 1

    if not folds:
        raise ValueError("walk-forward 未生成任何窗口，请检查窗口大小与数据长度")

    fold_summary = pd.DataFrame(fold_rows)
    walk_forward_equity = pd.concat(aggregated_equity_frames, ignore_index=True)
    baseline_walk_forward_equity = pd.concat(baseline_equity_frames, ignore_index=True)

    metric_columns = [
        "train_annual_return",
        "train_sharpe",
        "test_annual_return",
        "test_total_return",
        "test_sharpe",
        "test_max_drawdown",
        "baseline_test_annual_return",
        "baseline_test_total_return",
        "baseline_test_sharpe",
        "baseline_test_max_drawdown",
    ]
    average_metrics = {
        key: round(float(fold_summary[key].mean()), 4)
        for key in metric_columns
        if key in fold_summary.columns
    }

    overview = {
        "fold_count": int(len(folds)),
        "train_window": int(train_window),
        "test_window": int(test_window),
        "step_size": int(step_size),
        "start_date": price_df["date"].min().strftime("%Y-%m-%d"),
        "end_date": price_df["date"].max().strftime("%Y-%m-%d"),
    }

    return {
        "overview": overview,
        "fold_summary": fold_summary,
        "folds": folds,
        "average_metrics": average_metrics,
        "walk_forward_equity": walk_forward_equity,
        "baseline_walk_forward_equity": baseline_walk_forward_equity,
    }
