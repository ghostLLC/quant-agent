from __future__ import annotations

from quantlab.config import BacktestConfig, DEFAULT_GRID
from quantlab.strategies.base import StrategySignalConfig, StrategySpec
from quantlab.strategies.channel_breakout import generate_channel_breakout_signals
from quantlab.strategies.ma_cross import generate_ma_signals


DEFAULT_STRATEGY_NAME = "ma_cross"


def build_ma_signal_config(config: BacktestConfig) -> StrategySignalConfig:
    return StrategySignalConfig(
        short_window=config.short_window,
        long_window=config.long_window,
        enable_trend_filter=config.enable_trend_filter,
        trend_window=config.trend_window,
        enable_volatility_filter=config.enable_volatility_filter,
        volatility_window=config.volatility_window,
        max_volatility=config.max_volatility,
    )


def build_channel_breakout_config(config: BacktestConfig) -> StrategySignalConfig:
    return StrategySignalConfig(
        short_window=config.short_window,
        long_window=config.long_window,
        enable_trend_filter=config.enable_trend_filter,
        trend_window=config.trend_window,
        enable_volatility_filter=config.enable_volatility_filter,
        volatility_window=config.volatility_window,
        max_volatility=config.max_volatility,
    )


MA_CROSS_STRATEGY = StrategySpec(
    name=DEFAULT_STRATEGY_NAME,
    title="双均线交叉",
    description="使用短均线与长均线交叉生成信号，并可叠加趋势过滤和波动率过滤。",
    signal_config_builder=build_ma_signal_config,
    signal_generator=generate_ma_signals,
    default_parameter_grid={key: list(value) for key, value in DEFAULT_GRID.items()},
    tags=("trend", "long_only", "baseline"),
)

CHANNEL_BREAKOUT_STRATEGY = StrategySpec(
    name="channel_breakout",
    title="通道突破",
    description="基于价格通道突破与回落退出生成信号，并可叠加趋势与波动率过滤。",
    signal_config_builder=build_channel_breakout_config,
    signal_generator=generate_channel_breakout_signals,
    default_parameter_grid={
        "short_window": [10, 20, 30],
        "long_window": [40, 60, 80],
        "enable_trend_filter": [False, True],
        "stop_loss_pct": [None, 0.05, 0.08],
    },
    tags=("breakout", "trend", "long_only"),
)


_STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    MA_CROSS_STRATEGY.name: MA_CROSS_STRATEGY,
    CHANNEL_BREAKOUT_STRATEGY.name: CHANNEL_BREAKOUT_STRATEGY,
}



def register_strategy(spec: StrategySpec) -> None:
    _STRATEGY_REGISTRY[spec.name] = spec


def get_strategy(name: str = DEFAULT_STRATEGY_NAME) -> StrategySpec:
    key = (name or DEFAULT_STRATEGY_NAME).strip()
    if key not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(f"未注册策略：{key}。当前可用策略：{available}")
    return _STRATEGY_REGISTRY[key]


def get_default_strategy() -> StrategySpec:
    return get_strategy(DEFAULT_STRATEGY_NAME)


def list_strategies() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "tags": list(spec.tags),
        }
        for spec in _STRATEGY_REGISTRY.values()
    ]


def resolve_parameter_grid(name: str = DEFAULT_STRATEGY_NAME, overrides: dict[str, list] | None = None) -> dict[str, list]:
    spec = get_strategy(name)
    parameter_grid = {key: list(value) for key, value in spec.default_parameter_grid.items()}
    if overrides:
        for key, value in overrides.items():
            if value:
                parameter_grid[key] = list(value)
    return parameter_grid
