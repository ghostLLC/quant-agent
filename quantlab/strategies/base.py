from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategySignalConfig:
    short_window: int
    long_window: int
    enable_trend_filter: bool = False
    trend_window: int = 60
    enable_volatility_filter: bool = False
    volatility_window: int = 20
    max_volatility: float | None = None
