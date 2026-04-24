from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quantlab.analysis.grid_search import run_parameter_grid
from quantlab.backtest.engine import BacktestResult
from quantlab.config import BacktestConfig
from quantlab.strategies import get_strategy


def split_train_test(price_df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.3 <= train_ratio <= 0.9:
        raise ValueError("train_ratio 需要位于 0.3 到 0.9 之间")
    split_index = int(len(price_df) * train_ratio)
    split_index = max(30, min(split_index, len(price_df) - 30))
    train_df = price_df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = price_df.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def _run_backtest_with_config(
    price_df: pd.DataFrame,
    config: BacktestConfig,
    backtest_runner,
    strategy_name: str = "ma_cross",
) -> BacktestResult:
    strategy = get_strategy(strategy_name)
    signal_df = strategy.generate_signals(price_df, config)
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


def _bounded_score(value: float, upper_bound: float) -> float:
    if upper_bound <= 0:
        return 0.0
    normalized = max(0.0, min(1.0, value / upper_bound))
    return round(1 - normalized, 4)


def summarize_walk_forward_stability(fold_summary: pd.DataFrame) -> dict:
    if fold_summary.empty:
        return {
            "stability_score": 0.0,
            "stability_label": "暂无样本",
            "positive_test_ratio": 0.0,
            "beat_baseline_ratio": 0.0,
            "test_annual_return_std": 0.0,
            "test_max_drawdown_std": 0.0,
            "parameter_regime_count": 0,
            "dominant_parameter_ratio": 0.0,
        }

    positive_test_ratio = float((fold_summary["test_annual_return"] > 0).mean())
    beat_baseline_ratio = float((fold_summary["test_annual_return"] > fold_summary["baseline_test_annual_return"]).mean())
    test_annual_return_std = float(fold_summary["test_annual_return"].std(ddof=0) or 0.0)
    test_max_drawdown_std = float(fold_summary["test_max_drawdown"].std(ddof=0) or 0.0)

    parameter_columns = [
        col for col in ["short_window", "long_window", "enable_trend_filter", "stop_loss_pct"]
        if col in fold_summary.columns
    ]
    if parameter_columns:
        regime_counts = fold_summary[parameter_columns].astype(str).agg("|".join, axis=1).value_counts()
        parameter_regime_count = int(regime_counts.shape[0])
        dominant_parameter_ratio = float(regime_counts.iloc[0] / len(fold_summary))
    else:
        parameter_regime_count = 0
        dominant_parameter_ratio = 0.0

    score_components = {
        "positive_test_ratio": positive_test_ratio,
        "beat_baseline_ratio": beat_baseline_ratio,
        "return_stability": _bounded_score(test_annual_return_std, 0.30),
        "drawdown_stability": _bounded_score(abs(test_max_drawdown_std), 0.20),
        "parameter_consistency": dominant_parameter_ratio,
    }
    stability_score = round(
        score_components["positive_test_ratio"] * 0.28
        + score_components["beat_baseline_ratio"] * 0.28
        + score_components["return_stability"] * 0.18
        + score_components["drawdown_stability"] * 0.14
        + score_components["parameter_consistency"] * 0.12,
        4,
    )

    if stability_score >= 0.75:
        stability_label = "稳定性强"
    elif stability_score >= 0.55:
        stability_label = "稳定性中等"
    else:
        stability_label = "稳定性偏弱"

    return {
        "stability_score": stability_score,
        "stability_label": stability_label,
        "positive_test_ratio": round(positive_test_ratio, 4),
        "beat_baseline_ratio": round(beat_baseline_ratio, 4),
        "test_annual_return_std": round(test_annual_return_std, 4),
        "test_max_drawdown_std": round(test_max_drawdown_std, 4),
        "parameter_regime_count": parameter_regime_count,
        "dominant_parameter_ratio": round(dominant_parameter_ratio, 4),
        "score_components": score_components,
    }


def summarize_walk_forward_research_score(fold_summary: pd.DataFrame, average_metrics: dict, stability_summary: dict) -> dict:
    if fold_summary.empty:
        return {
            "research_score": 0.0,
            "research_label": "暂无样本",
            "score_components": {},
        }

    avg_test_annual_return = float(average_metrics.get("test_annual_return", 0.0))
    avg_test_sharpe = float(average_metrics.get("test_sharpe", 0.0))
    avg_test_max_drawdown = abs(float(average_metrics.get("test_max_drawdown", 0.0)))
    baseline_annual_return = float(average_metrics.get("baseline_test_annual_return", 0.0))
    excess_annual_return = avg_test_annual_return - baseline_annual_return

    score_components = {
        "return_quality": round(max(0.0, min(1.0, avg_test_annual_return / 0.25)), 4),
        "sharpe_quality": round(max(0.0, min(1.0, avg_test_sharpe / 1.5)), 4),
        "drawdown_quality": _bounded_score(avg_test_max_drawdown, 0.25),
        "excess_quality": round(max(0.0, min(1.0, excess_annual_return / 0.15)), 4),
        "stability_quality": float(stability_summary.get("stability_score", 0.0)),
    }
    research_score = round(
        score_components["return_quality"] * 0.30
        + score_components["sharpe_quality"] * 0.22
        + score_components["drawdown_quality"] * 0.16
        + score_components["excess_quality"] * 0.12
        + score_components["stability_quality"] * 0.20,
        4,
    )

    if research_score >= 0.78:
        research_label = "研究质量强"
    elif research_score >= 0.58:
        research_label = "研究质量中等"
    else:
        research_label = "研究质量偏弱"

    return {
        "research_score": research_score,
        "research_label": research_label,
        "avg_test_annual_return": round(avg_test_annual_return, 4),
        "avg_test_sharpe": round(avg_test_sharpe, 4),
        "avg_test_max_drawdown": round(avg_test_max_drawdown, 4),
        "avg_baseline_test_annual_return": round(baseline_annual_return, 4),
        "excess_annual_return": round(excess_annual_return, 4),
        "score_components": score_components,
    }


def run_train_test_validation(
    price_df: pd.DataFrame,
    base_config: BacktestConfig,
    parameter_grid: dict[str, list],
    backtest_runner,
    strategy_name: str = "ma_cross",
) -> dict:
    train_df, test_df = split_train_test(price_df, base_config.train_ratio)
    train_summary, best_train_result = run_parameter_grid(train_df, base_config, parameter_grid, strategy_name=strategy_name)
    best_params = train_summary.iloc[0][list(parameter_grid.keys())].to_dict()

    tuned_config = BacktestConfig(**{**asdict(base_config), **best_params})
    test_result = _run_backtest_with_config(test_df, tuned_config, backtest_runner, strategy_name=strategy_name)
    baseline_test_result = _run_backtest_with_config(test_df, base_config, backtest_runner, strategy_name=strategy_name)

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
    strategy_name: str = "ma_cross",
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

        train_summary, best_train_result = run_parameter_grid(train_df, base_config, parameter_grid, strategy_name=strategy_name)
        best_params = train_summary.iloc[0][list(parameter_grid.keys())].to_dict()
        tuned_config = BacktestConfig(**{**asdict(base_config), **best_params})

        test_result = _run_backtest_with_config(test_df, tuned_config, backtest_runner, strategy_name=strategy_name)
        baseline_result = _run_backtest_with_config(test_df, base_config, backtest_runner, strategy_name=strategy_name)

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
    stability_summary = summarize_walk_forward_stability(fold_summary)
    research_summary = summarize_walk_forward_research_score(fold_summary, average_metrics, stability_summary)

    parameter_columns = [
        col for col in ["short_window", "long_window", "enable_trend_filter", "stop_loss_pct"]
        if col in fold_summary.columns
    ]
    if parameter_columns:
        regime_evolution = fold_summary[["fold_id", *parameter_columns]].copy()
        regime_evolution["parameter_regime"] = regime_evolution[parameter_columns].astype(str).agg(" | ".join, axis=1)
    else:
        regime_evolution = pd.DataFrame(columns=["fold_id", "parameter_regime"])

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
        "stability_summary": stability_summary,
        "research_summary": research_summary,
        "regime_evolution": regime_evolution,
        "walk_forward_equity": walk_forward_equity,
        "baseline_walk_forward_equity": baseline_walk_forward_equity,
    }



