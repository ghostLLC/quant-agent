from __future__ import annotations

import pandas as pd

from quantlab.strategies.base import StrategySignalConfig


def generate_ma_signals(df: pd.DataFrame, config: StrategySignalConfig) -> pd.DataFrame:
    if config.short_window >= config.long_window:
        raise ValueError("short_window 必须小于 long_window")

    data = df.copy()
    data["ma_short"] = data["close"].rolling(window=config.short_window, min_periods=1).mean()
    data["ma_long"] = data["close"].rolling(window=config.long_window, min_periods=1).mean()
    data["trend_ma"] = data["close"].rolling(window=config.trend_window, min_periods=1).mean()
    data["returns"] = data["close"].pct_change().fillna(0.0)
    data["volatility"] = data["returns"].rolling(window=config.volatility_window, min_periods=1).std().fillna(0.0)

    base_buy = (data["ma_short"] > data["ma_long"]) & (
        data["ma_short"].shift(1) <= data["ma_long"].shift(1)
    )
    base_sell = (data["ma_short"] < data["ma_long"]) & (
        data["ma_short"].shift(1) >= data["ma_long"].shift(1)
    )

    if config.enable_trend_filter:
        base_buy = base_buy & (data["close"] > data["trend_ma"])

    if config.enable_volatility_filter and config.max_volatility is not None:
        base_buy = base_buy & (data["volatility"] <= config.max_volatility)

    data["signal"] = 0
    data.loc[base_buy, "signal"] = 1
    data.loc[base_sell, "signal"] = -1
    return data
