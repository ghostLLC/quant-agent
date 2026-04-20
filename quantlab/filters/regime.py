from __future__ import annotations

import pandas as pd


def attach_regime_features(df: pd.DataFrame, trend_window: int, volatility_window: int) -> pd.DataFrame:
    data = df.copy()
    data["trend_ma"] = data["close"].rolling(window=trend_window, min_periods=1).mean()
    data["returns"] = data["close"].pct_change().fillna(0.0)
    data["volatility"] = data["returns"].rolling(window=volatility_window, min_periods=1).std().fillna(0.0)
    data["trend_state"] = (data["close"] > data["trend_ma"]).map({True: "uptrend", False: "downtrend"})
    return data
