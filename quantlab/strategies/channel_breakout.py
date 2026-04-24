from __future__ import annotations

import pandas as pd

from quantlab.strategies.base import StrategySignalConfig


def generate_channel_breakout_signals(df: pd.DataFrame, config: StrategySignalConfig) -> pd.DataFrame:
    if config.short_window < 2:
        raise ValueError("short_window 至少需要 2")
    if config.long_window < config.short_window:
        raise ValueError("long_window 不能小于 short_window")

    data = df.copy()
    breakout_window = config.short_window
    exit_window = config.long_window

    data["upper_channel"] = data["close"].rolling(window=breakout_window, min_periods=breakout_window).max().shift(1)
    data["lower_channel"] = data["close"].rolling(window=exit_window, min_periods=exit_window).min().shift(1)
    data["trend_ma"] = data["close"].rolling(window=config.trend_window, min_periods=1).mean()
    data["returns"] = data["close"].pct_change().fillna(0.0)
    data["volatility"] = data["returns"].rolling(window=config.volatility_window, min_periods=1).std().fillna(0.0)

    base_buy = data["close"] > data["upper_channel"]
    base_sell = data["close"] < data["lower_channel"]

    if config.enable_trend_filter:
        base_buy = base_buy & (data["close"] > data["trend_ma"])

    if config.enable_volatility_filter and config.max_volatility is not None:
        base_buy = base_buy & (data["volatility"] <= config.max_volatility)

    data["signal"] = 0
    data.loc[base_buy.fillna(False), "signal"] = 1
    data.loc[base_sell.fillna(False), "signal"] = -1
    return data
