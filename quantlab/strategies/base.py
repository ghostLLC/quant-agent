from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from quantlab.config import BacktestConfig


@dataclass
class StrategySignalConfig:
    short_window: int
    long_window: int
    enable_trend_filter: bool = False
    trend_window: int = 60
    enable_volatility_filter: bool = False
    volatility_window: int = 20
    max_volatility: float | None = None


SignalConfigBuilder = Callable[["BacktestConfig"], StrategySignalConfig]
SignalGenerator = Callable[[pd.DataFrame, StrategySignalConfig], pd.DataFrame]


@dataclass
class StrategySpec:
    name: str
    title: str
    description: str
    signal_config_builder: SignalConfigBuilder
    signal_generator: SignalGenerator
    default_parameter_grid: dict[str, list] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def build_signal_config(self, config: "BacktestConfig") -> StrategySignalConfig:
        return self.signal_config_builder(config)

    def generate_signals(self, price_df: pd.DataFrame, config: "BacktestConfig") -> pd.DataFrame:
        return self.signal_generator(price_df, self.build_signal_config(config))
