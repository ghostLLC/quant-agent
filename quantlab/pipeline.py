from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


from quantlab.analysis.grid_search import run_parameter_grid
from quantlab.analysis.history_store import load_experiment_detail, load_experiment_history, save_experiment_record
from quantlab.analysis.validation import run_train_test_validation, run_walk_forward_validation, summarize_walk_forward_research_score
from quantlab.backtest.engine import BacktestResult, export_backtest_result, run_long_only_backtest
from quantlab.config import BacktestConfig, DEFAULT_CROSS_SECTION_DATA_PATH, DEFAULT_DATA_PATH, DEFAULT_PRICE_DATA_PATH
from quantlab.data.fetcher import update_hs300_cross_section_csv, update_hs300_etf_csv
from quantlab.data.loader import load_cross_section_data, load_price_data, summarize_cross_section_data, summarize_data

from quantlab.strategies import get_default_strategy, get_strategy, list_strategies, resolve_parameter_grid




def _normalize_weight_map(weight_map: dict[str, float]) -> dict[str, float]:
    positive_map = {key: max(float(value), 0.0) for key, value in weight_map.items()}
    total = sum(positive_map.values())
    if total <= 0:
        equal_weight = round(1 / len(positive_map), 4) if positive_map else 0.0
        return {key: equal_weight for key in positive_map}
    return {key: round(value / total, 4) for key, value in positive_map.items()}



def _build_strategy_return_series(
    price_df: pd.DataFrame,
    config: BacktestConfig,
    strategy_name: str,
    best_params: dict[str, object],
) -> pd.Series:
    strategy_config = BacktestConfig(**{**asdict(config), **best_params})
    strategy = get_strategy(strategy_name)
    signal_df = strategy.generate_signals(price_df, strategy_config)
    backtest_result = run_long_only_backtest(signal_df, strategy_config)
    series = backtest_result.equity_curve[["date", "daily_return"]].copy()
    series["date"] = pd.to_datetime(series["date"])
    return series.set_index("date")["daily_return"].rename(strategy_name)



def _infer_portfolio_constraints(config: BacktestConfig, ranking: pd.DataFrame) -> dict[str, float]:
    candidate_count = max(1, min(3, len(ranking)))
    avg_drawdown = abs(float(ranking["train_test_max_drawdown"].head(candidate_count).mean())) if "train_test_max_drawdown" in ranking else 0.12
    avg_volatility = float(ranking["realized_volatility"].head(candidate_count).mean()) if "realized_volatility" in ranking else 0.18
    target_volatility = round(min(0.25, max(0.08, avg_volatility * 0.9 if avg_volatility > 0 else 0.16)), 4)
    max_drawdown_limit = round(min(0.3, max(0.08, avg_drawdown * 1.15 if avg_drawdown > 0 else 0.18)), 4)
    max_single_weight = round(min(0.7, max(0.45, 1 / candidate_count + 0.1)), 4)
    return {
        "target_volatility": target_volatility,
        "max_drawdown_limit": max_drawdown_limit,
        "max_single_weight": max_single_weight,
    }



def _solve_constrained_weights(
    ranking: pd.DataFrame,
    candidate_names: list[str],
    correlation_matrix: pd.DataFrame,
    constraints: dict[str, float],
) -> dict[str, object]:
    candidate_df = ranking[ranking["strategy_name"].isin(candidate_names)].copy()
    candidate_df = candidate_df.set_index("strategy_name").loc[candidate_names]
    base_scores = candidate_df["composite_score"].clip(lower=0.0001)
    normalized_scores = base_scores / base_scores.sum()
    penalties = 1 - correlation_matrix.abs().mean()
    adjusted_scores = normalized_scores * penalties.reindex(candidate_names).fillna(1.0).clip(lower=0.1)
    adjusted_scores = adjusted_scores / adjusted_scores.sum()

    min_weight = 0.1 if len(candidate_names) >= 2 else 1.0
    max_weight = float(constraints["max_single_weight"])
    clipped = adjusted_scores.clip(lower=min_weight, upper=max_weight)
    normalized = _normalize_weight_map(clipped.to_dict())

    weight_vector = np.array([normalized[name] for name in candidate_names], dtype=float)
    annualized_volatility = np.array(
        [float(candidate_df.loc[name, "realized_volatility"]) for name in candidate_names],
        dtype=float,
    )
    target_volatility = float(constraints["target_volatility"])
    portfolio_volatility = float(np.dot(weight_vector, annualized_volatility))
    if portfolio_volatility > 0 and portfolio_volatility > target_volatility:
        scale = target_volatility / portfolio_volatility
        scaled_map = {name: max(weight * scale, 0.05) for name, weight in normalized.items()}
        normalized = _normalize_weight_map(scaled_map)
        weight_vector = np.array([normalized[name] for name in candidate_names], dtype=float)
        portfolio_volatility = float(np.dot(weight_vector, annualized_volatility))

    expected_drawdown = float(
        np.dot(
            weight_vector,
            np.array([abs(float(candidate_df.loc[name, "train_test_max_drawdown"])) for name in candidate_names], dtype=float),
        )
    )
    diversified_correlation = float(
        np.dot(weight_vector, np.dot(correlation_matrix.loc[candidate_names, candidate_names].to_numpy(dtype=float), weight_vector))
    )
    diversification_score = round(max(0.0, min(1.0, 1 - max(0.0, diversified_correlation))), 4)

    constraint_flags = {
        "within_target_volatility": portfolio_volatility <= target_volatility + 1e-6,
        "within_drawdown_limit": expected_drawdown <= float(constraints["max_drawdown_limit"]) + 1e-6,
        "within_single_weight_limit": max(normalized.values()) <= max_weight + 1e-6,
    }
    return {
        "weights": normalized,
        "portfolio_volatility": round(portfolio_volatility, 4),
        "expected_max_drawdown": round(expected_drawdown, 4),
        "diversification_score": diversification_score,
        "constraint_flags": constraint_flags,
    }



def build_signal_config(config: BacktestConfig, strategy_name: str = "ma_cross"):

    strategy = get_strategy(strategy_name)
    return strategy.build_signal_config(config)


def run_single_backtest(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
    strategy_name: str = "ma_cross",
) -> tuple[BacktestResult, dict, Path, Path]:
    config = config or BacktestConfig()
    price_df = load_price_data(data_path)
    strategy = get_strategy(strategy_name)
    signal_df = strategy.generate_signals(price_df, config)
    result = run_long_only_backtest(signal_df, config)
    export_backtest_result(result, config.report_dir)
    history_path = save_experiment_record(
        experiment_type="single_backtest",
        config_payload=asdict(config),
        metrics_payload=result.metrics,
        notes={"data_path": str(data_path), "strategy_name": strategy.name},
        history_dir=config.history_dir,
    )
    return result, summarize_data(price_df), Path(data_path), history_path


def run_grid_experiment(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
    parameter_grid: dict[str, list] | None = None,
    strategy_name: str = "ma_cross",
) -> tuple[pd.DataFrame, BacktestResult, dict, Path]:
    config = config or BacktestConfig()
    strategy = get_strategy(strategy_name)
    parameter_grid = resolve_parameter_grid(strategy.name, parameter_grid)
    price_df = load_price_data(data_path)
    summary_df, best_result = run_parameter_grid(price_df, config, parameter_grid, strategy_name=strategy.name)
    history_path = save_experiment_record(
        experiment_type="grid_search",
        config_payload={**asdict(config), "parameter_grid": parameter_grid, "strategy_name": strategy.name},
        metrics_payload=best_result.metrics,
        notes={
            "best_params": summary_df.iloc[0][list(parameter_grid.keys())].to_dict(),
            "strategy_name": strategy.name,
        },
        history_dir=config.history_dir,
    )
    return summary_df, best_result, summarize_data(price_df), history_path


def run_train_test_experiment(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
    parameter_grid: dict[str, list] | None = None,
    strategy_name: str = "ma_cross",
) -> tuple[dict, dict, Path]:
    config = config or BacktestConfig()
    strategy = get_strategy(strategy_name)
    parameter_grid = resolve_parameter_grid(strategy.name, parameter_grid)
    price_df = load_price_data(data_path)
    validation_result = run_train_test_validation(price_df, config, parameter_grid, run_long_only_backtest, strategy_name=strategy.name)
    overview = summarize_data(price_df)

    train_metrics = validation_result["train_best_result"].metrics
    test_metrics = validation_result["test_result"].metrics
    baseline_metrics = validation_result["baseline_test_result"].metrics

    history_path = save_experiment_record(
        experiment_type="train_test_validation",
        config_payload={**asdict(config), "parameter_grid": parameter_grid, "strategy_name": strategy.name},
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
            "strategy_name": strategy.name,
        },
        history_dir=config.history_dir,
    )
    return validation_result, overview, history_path


def run_walk_forward_experiment(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
    parameter_grid: dict[str, list] | None = None,
    strategy_name: str = "ma_cross",
) -> tuple[dict, dict, Path]:
    config = config or BacktestConfig()
    strategy = get_strategy(strategy_name)
    parameter_grid = resolve_parameter_grid(strategy.name, parameter_grid)
    price_df = load_price_data(data_path)
    walk_forward_result = run_walk_forward_validation(price_df, config, parameter_grid, run_long_only_backtest, strategy_name=strategy.name)
    overview = summarize_data(price_df)
    stability_summary = walk_forward_result["stability_summary"]
    research_summary = walk_forward_result["research_summary"]

    history_path = save_experiment_record(
        experiment_type="walk_forward_validation",
        config_payload={**asdict(config), "parameter_grid": parameter_grid, "strategy_name": strategy.name},
        metrics_payload={
            "average_train_annual_return": walk_forward_result["average_metrics"].get("train_annual_return", 0.0),
            "average_test_annual_return": walk_forward_result["average_metrics"].get("test_annual_return", 0.0),
            "average_test_sharpe": walk_forward_result["average_metrics"].get("test_sharpe", 0.0),
            "average_test_max_drawdown": walk_forward_result["average_metrics"].get("test_max_drawdown", 0.0),
            "average_baseline_test_annual_return": walk_forward_result["average_metrics"].get("baseline_test_annual_return", 0.0),
            "average_baseline_test_sharpe": walk_forward_result["average_metrics"].get("baseline_test_sharpe", 0.0),
            "stability_score": stability_summary.get("stability_score", 0.0),
            "positive_test_ratio": stability_summary.get("positive_test_ratio", 0.0),
            "beat_baseline_ratio": stability_summary.get("beat_baseline_ratio", 0.0),
            "dominant_parameter_ratio": stability_summary.get("dominant_parameter_ratio", 0.0),
            "research_score": research_summary.get("research_score", 0.0),
            "excess_annual_return": research_summary.get("excess_annual_return", 0.0),
        },
        notes={
            "overview": walk_forward_result["overview"],
            "stability_summary": stability_summary,
            "research_summary": research_summary,
            "regime_evolution": walk_forward_result["regime_evolution"].to_dict(orient="records"),
            "fold_summary": walk_forward_result["fold_summary"].to_dict(orient="records"),
            "strategy_name": strategy.name,
        },
        history_dir=config.history_dir,
    )

    return walk_forward_result, overview, history_path






def refresh_market_data(data_path: str | Path = DEFAULT_PRICE_DATA_PATH) -> dict[str, object]:
    update_hs300_etf_csv(data_path)

    loaded = load_price_data(data_path)
    summary = summarize_data(loaded)
    summary["data_path"] = str(data_path)
    return summary




def refresh_cross_section_data(
    data_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    max_assets: int | None = None,
    index_symbol: str = "000300",
    pause_seconds: float = 0.2,
    resume: bool = True,
) -> dict[str, object]:
    df, refresh_report = update_hs300_cross_section_csv(
        output_path=data_path,
        start_date=start_date,
        end_date=end_date,
        max_assets=max_assets,
        index_symbol=index_symbol,
        pause_seconds=pause_seconds,
        resume=resume,
    )
    loaded = load_cross_section_data(data_path)
    summary = summarize_cross_section_data(loaded)
    summary["index_symbol"] = index_symbol
    summary["data_path"] = str(data_path)
    summary["refresh_report"] = refresh_report
    return summary


def incremental_refresh_cross_section_data(
    data_path: str | Path,
) -> dict[str, object]:
    """增量刷新横截面数据：只拉取现有数据之后的新交易日。"""
    from quantlab.data.tushare_provider import AkShareIncrementalProvider
    provider = AkShareIncrementalProvider()
    result = provider.refresh_cross_section(Path(data_path))
    return result






def get_experiment_history(config: BacktestConfig | None = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    return load_experiment_history(config.history_dir)



def get_experiment_detail(experiment_id: str, config: BacktestConfig | None = None) -> dict | None:
    config = config or BacktestConfig()
    return load_experiment_detail(experiment_id, config.history_dir)


def get_strategy_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": spec["name"],
            "title": spec["title"],
            "description": spec["description"],
            "tags": spec["tags"],
        }
        for spec in list_strategies()
    ]



def get_default_strategy_summary() -> dict[str, object]:
    strategy = get_default_strategy()
    return {
        "name": strategy.name,
        "title": strategy.title,
        "description": strategy.description,
        "tags": list(strategy.tags),
        "default_parameter_grid": {key: list(value) for key, value in strategy.default_parameter_grid.items()},
    }


def run_multi_strategy_compare(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
) -> dict[str, object]:
    config = config or BacktestConfig()
    price_df = load_price_data(data_path)
    strategy_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for item in list_strategies():
        strategy = get_strategy(str(item["name"]))
        parameter_grid = resolve_parameter_grid(strategy.name)
        summary_df, best_result, _, _ = run_grid_experiment(
            data_path=data_path,
            config=config,
            parameter_grid=parameter_grid,
            strategy_name=strategy.name,
        )
        validation_result, _, _ = run_train_test_experiment(
            data_path=data_path,
            config=config,
            parameter_grid=parameter_grid,
            strategy_name=strategy.name,
        )
        walk_forward_result, _, _ = run_walk_forward_experiment(
            data_path=data_path,
            config=config,
            parameter_grid=parameter_grid,
            strategy_name=strategy.name,
        )
        research_summary = walk_forward_result.get("research_summary", {})
        stability_summary = walk_forward_result.get("stability_summary", {})
        row = {
            "strategy_name": strategy.name,
            "strategy_title": strategy.title,
            "tags": list(strategy.tags),
            "grid_best_sharpe": best_result.metrics.get("sharpe", 0.0),
            "grid_best_annual_return": best_result.metrics.get("annual_return", 0.0),
            "train_test_sharpe": validation_result["test_result"].metrics.get("sharpe", 0.0),
            "train_test_annual_return": validation_result["test_result"].metrics.get("annual_return", 0.0),
            "train_test_max_drawdown": abs(validation_result["test_result"].metrics.get("max_drawdown", 0.0)),
            "walk_forward_research_score": research_summary.get("research_score", 0.0),
            "walk_forward_stability_score": stability_summary.get("stability_score", 0.0),
            "realized_volatility": float(validation_result["test_result"].equity_curve["daily_return"].std(ddof=0) * np.sqrt(config.trading_days_per_year)),
        }

        strategy_rows.append(row)
        detail_rows.append(
            {
                **row,
                "best_params": validation_result["best_params"],
                "research_label": research_summary.get("research_label", "未知"),
                "stability_label": stability_summary.get("stability_label", "未知"),
                "avg_test_max_drawdown": research_summary.get("avg_test_max_drawdown", 0.0),
                "excess_annual_return": research_summary.get("excess_annual_return", 0.0),
            }
        )


    ranking = pd.DataFrame(strategy_rows)
    if ranking.empty:
        raise ValueError("当前没有可比较的策略")
    ranking["composite_score"] = (
        ranking["grid_best_sharpe"] * 0.2
        + ranking["train_test_sharpe"] * 0.25
        + ranking["walk_forward_research_score"] * 0.35
        + ranking["walk_forward_stability_score"] * 0.2
    )
    ranking = ranking.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    best_strategy = ranking.iloc[0].to_dict()
    return {
        "overview": summarize_data(price_df),
        "strategy_count": int(len(ranking)),
        "ranking": ranking,
        "best_strategy": best_strategy,
        "details": detail_rows,
    }


def review_portfolio_construction(
    data_path: str | Path = DEFAULT_PRICE_DATA_PATH,

    config: BacktestConfig | None = None,
) -> dict[str, object]:
    config = config or BacktestConfig()
    compare_result = run_multi_strategy_compare(data_path=data_path, config=config)
    ranking = compare_result["ranking"].copy()
    price_df = load_price_data(data_path)
    candidate_count = max(1, min(3, len(ranking)))
    candidate_names = ranking.head(candidate_count)["strategy_name"].tolist()

    if not candidate_names:
        return {
            "portfolio_candidates": [],
            "allocation_principles": ["当前没有可用策略，无法构建组合层建议。"],
            "portfolio_assessment": {
                "candidate_count": 0,
                "average_research_score": 0.0,
                "average_stability_score": 0.0,
            },
            "compare_snapshot": compare_result["best_strategy"],
        }

    best_param_map = {
        str(item["strategy_name"]): dict(item.get("best_params", {}))
        for item in compare_result.get("details", [])
    }
    return_series = [
        _build_strategy_return_series(price_df, config, name, best_param_map.get(name, {}))
        for name in candidate_names
    ]
    returns_frame = pd.concat(return_series, axis=1).fillna(0.0)
    correlation_matrix = returns_frame.corr().fillna(0.0)
    constraints = _infer_portfolio_constraints(config, ranking)
    optimization_result = _solve_constrained_weights(ranking, candidate_names, correlation_matrix, constraints)

    portfolio_candidates: list[dict[str, object]] = []
    for index, name in enumerate(candidate_names):
        row = ranking[ranking["strategy_name"] == name].iloc[0]
        portfolio_candidates.append(
            {
                "strategy_name": name,
                "weight": optimization_result["weights"].get(name, 0.0),
                "role": "core" if index == 0 else "satellite",
                "composite_score": round(float(row.get("composite_score", 0.0)), 4),
                "expected_volatility": round(float(row.get("realized_volatility", 0.0)), 4),
                "expected_max_drawdown": round(float(row.get("train_test_max_drawdown", 0.0)), 4),
            }
        )

    avg_research_score = round(float(ranking["walk_forward_research_score"].mean()), 4)
    avg_stability_score = round(float(ranking["walk_forward_stability_score"].mean()), 4)
    blended_summary = summarize_walk_forward_research_score(
        fold_summary=pd.DataFrame([
            {
                "test_annual_return": avg_research_score,
                "baseline_test_annual_return": 0.0,
                "test_sharpe": avg_stability_score,
                "test_max_drawdown": -optimization_result["expected_max_drawdown"],
            }
        ]),
        average_metrics={
            "test_annual_return": avg_research_score,
            "test_sharpe": avg_stability_score,
            "test_max_drawdown": -optimization_result["expected_max_drawdown"],
            "baseline_test_annual_return": 0.0,
        },
        stability_summary={"stability_score": avg_stability_score},
    )

    return {
        "portfolio_candidates": portfolio_candidates,
        "allocation_principles": [
            "组合核心权重优先分配给综合分更高且约束内波动更可控的策略。",
            "卫星权重优先分配给与核心策略相关性较低的策略，用于缓解单策略失效。",
            "当前组合建议已显式纳入相关性、目标波动、最大回撤约束与单策略权重上限。",
        ],
        "portfolio_constraints": constraints,
        "correlation_matrix": correlation_matrix.round(4).to_dict(orient="index"),
        "portfolio_assessment": {
            "candidate_count": len(portfolio_candidates),
            "average_research_score": avg_research_score,
            "average_stability_score": avg_stability_score,
            "blended_research_summary": blended_summary,
            "target_volatility": constraints["target_volatility"],
            "expected_portfolio_volatility": optimization_result["portfolio_volatility"],
            "max_drawdown_limit": constraints["max_drawdown_limit"],
            "expected_portfolio_max_drawdown": optimization_result["expected_max_drawdown"],
            "diversification_score": optimization_result["diversification_score"],
            "constraint_flags": optimization_result["constraint_flags"],
        },
        "compare_snapshot": compare_result["best_strategy"],
    }


