from __future__ import annotations

import pandas as pd


def generate_ma_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """基于收盘价生成双均线交易信号。"""
    if short_window >= long_window:
        raise ValueError("short_window 必须小于 long_window")

    data = df.copy()
    data["ma_short"] = data["close"].rolling(window=short_window, min_periods=1).mean()
    data["ma_long"] = data["close"].rolling(window=long_window, min_periods=1).mean()

    data["signal"] = 0
    buy_condition = (data["ma_short"] > data["ma_long"]) & (
        data["ma_short"].shift(1) <= data["ma_long"].shift(1)
    )
    sell_condition = (data["ma_short"] < data["ma_long"]) & (
        data["ma_short"].shift(1) >= data["ma_long"].shift(1)
    )

    data.loc[buy_condition, "signal"] = 1
    data.loc[sell_condition, "signal"] = -1
    return data
