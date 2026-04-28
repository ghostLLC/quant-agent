from __future__ import annotations

SUPPORTED_TASK_TYPES = {
    "single_backtest",
    "grid_search",
    "train_test_validation",
    "walk_forward_validation",
    "multi_strategy_compare",
    "portfolio_construction_review",
    "refresh_market_data",
    "refresh_cross_section_data",
    "experiment_history",
    "experiment_detail",
    "factor_discovery",
    "generate_factor_hypotheses",
    "factor_evolution",
    "multi_agent_discovery",
}


def normalize_task_type(task_type: str) -> str:
    normalized = (task_type or "").strip().lower()
    if normalized not in SUPPORTED_TASK_TYPES:
        available = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"不支持的研究任务类型：{task_type}。当前支持：{available}")
    return normalized


