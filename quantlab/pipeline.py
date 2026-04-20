from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quantlab.analysis.grid_search import run_parameter_grid
from quantlab.analysis.history_store import load_experiment_detail, load_experiment_history, save_experiment_record
from quantlab.analysis.validation import run_train_test_validation, run_walk_forward_validation
from quantlab.backtest.engine import BacktestResult, export_backtest_result, run_long_only_backtest
from quantlab.config import BacktestConfig, DEFAULT_DATA_PATH, DEFAULT_GRID
from quantlab.data.fetcher import update_hs300_etf_csv
from quantlab.data.loader import load_price_data, summarize_data
from quantlab.strategies.base import StrategySignalConfig
from quantlab.strategies.ma_cross import generate_ma_signals


def build_signal_config(config: BacktestConfig) -> StrategySignalConfig:
    return StrategySignalConfig(
        short_window=config.short_window,
        long_window=config.long_window,
        enable_trend_filter=config.enable_trend_filter,
        trend_window=config.trend_window,
        enable_volatility_filter=config.enable_volatility_filter,
        volatility_window=config.volatility_window,
        max_volatility=config.max_volatility,
    )


def run_single_backtest(data_path: str | Path = DEFAULT_DATA_PATH, config: BacktestConfig | None = None) -> tuple[BacktestResult, dict, Path, Path]:
    config = config or BacktestConfig()
    price_df = load_price_data(data_path)
    signal_df = generate_ma_signals(price_df, build_signal_config(config))
    result = run_long_only_backtest(signal_df, config)
    export_backtest_result(result, config.report_dir)
    history_path = save_experiment_record(
        experiment_type="single_backtest",
        config_payload=asdict(config),
        metrics_payload=result.metrics,
        notes={"data_path": str(data_path)},
        history_dir=config.history_dir,
    )
    return result, summarize_data(price_df), Path(data_path), history_path


def run_grid_experiment(data_path: str | Path = DEFAULT_DATA_PATH, config: BacktestConfig | None = None, parameter_grid: dict[str, list] | None = None) -> tuple[pd.DataFrame, BacktestResult, dict, Path]:
    config = config or BacktestConfig()
    parameter_grid = parameter_grid or DEFAULT_GRID
    price_df = load_price_data(data_path)
    summary_df, best_result = run_parameter_grid(price_df, config, parameter_grid)
    history_path = save_experiment_record(
        experiment_type="grid_search",
        config_payload={**asdict(config), "parameter_grid": parameter_grid},
        metrics_payload=best_result.metrics,
        notes={"best_params": summary_df.iloc[0][list(parameter_grid.keys())].to_dict()},
        history_dir=config.history_dir,
    )
    return summary_df, best_result, summarize_data(price_df), history_path


def run_train_test_experiment(data_path: str | Path = DEFAULT_DATA_PATH, config: BacktestConfig | None = None, parameter_grid: dict[str, list] | None = None) -> tuple[dict, dict, Path]:
    config = config or BacktestConfig()
    parameter_grid = parameter_grid or DEFAULT_GRID
    price_df = load_price_data(data_path)
    validation_result = run_train_test_validation(price_df, config, parameter_grid, run_long_only_backtest)
    overview = summarize_data(price_df)

    train_metrics = validation_result["train_best_result"].metrics
    test_metrics = validation_result["test_result"].metrics
    baseline_metrics = validation_result["baseline_test_result"].metrics

    history_path = save_experiment_record(
        experiment_type="train_test_validation",
        config_payload={**asdict(config), "parameter_grid": parameter_grid},
        metrics_payload={
            "train_annual_return": train_metrics["annual_return"],
            "test_annual_return": test_metrics["annual_return"],
            "test_sharpe": test_metrics["sharpe"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "baseline_test_annual_return": baseline_metrics["annual_return"],
            "baseline_test_sharpe": baseline_metrics["sharpe"],
        },
        notes={
            "overview": validation_result["overview"],
            "best_params": validation_result["best_params"],
        },
        history_dir=config.history_dir,
    )
    return validation_result, overview, history_path


def run_walk_forward_experiment(data_path: str | Path = DEFAULT_DATA_PATH, config: BacktestConfig | None = None, parameter_grid: dict[str, list] | None = None) -> tuple[dict, dict, Path]:
    config = config or BacktestConfig()
    parameter_grid = parameter_grid or DEFAULT_GRID
    price_df = load_price_data(data_path)
    walk_forward_result = run_walk_forward_validation(price_df, config, parameter_grid, run_long_only_backtest)
    overview = summarize_data(price_df)

    history_path = save_experiment_record(
        experiment_type="walk_forward_validation",
        config_payload={**asdict(config), "parameter_grid": parameter_grid},
        metrics_payload={
            "average_train_annual_return": walk_forward_result["average_metrics"].get("train_annual_return", 0.0),
            "average_test_annual_return": walk_forward_result["average_metrics"].get("test_annual_return", 0.0),
            "average_test_sharpe": walk_forward_result["average_metrics"].get("test_sharpe", 0.0),
            "average_test_max_drawdown": walk_forward_result["average_metrics"].get("test_max_drawdown", 0.0),
            "average_baseline_test_annual_return": walk_forward_result["average_metrics"].get("baseline_test_annual_return", 0.0),
            "average_baseline_test_sharpe": walk_forward_result["average_metrics"].get("baseline_test_sharpe", 0.0),
        },
        notes={
            "overview": walk_forward_result["overview"],
            "fold_summary": walk_forward_result["fold_summary"].to_dict(orient="records"),
        },
        history_dir=config.history_dir,
    )
    return walk_forward_result, overview, history_path


def refresh_market_data(data_path: str | Path = DEFAULT_DATA_PATH) -> dict:
    df = update_hs300_etf_csv(data_path)
    return summarize_data(df.assign(date=pd.to_datetime(df["date"])))


def get_experiment_history(config: BacktestConfig | None = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    return load_experiment_history(config.history_dir)


def get_experiment_detail(experiment_id: str, config: BacktestConfig | None = None) -> dict | None:
    config = config or BacktestConfig()
    return load_experiment_detail(experiment_id, config.history_dir)
