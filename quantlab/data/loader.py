from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
CROSS_SECTION_REQUIRED_COLUMNS = ["date", "asset", "close", "volume"]


def load_price_data(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    numeric_columns = [col for col in REQUIRED_COLUMNS if col != "date"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    data = data.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return data


def load_cross_section_data(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [col for col in CROSS_SECTION_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"横截面数据缺少必要字段: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["asset"] = data["asset"].astype(str)

    numeric_columns = [
        column
        for column in ["open", "high", "low", "close", "volume", "amount", "market_cap", "amplitude", "pct_chg", "change", "turnover"]
        if column in data.columns
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "industry" not in data.columns:
        data["industry"] = "unknown"
    else:
        data["industry"] = data["industry"].fillna("unknown").astype(str)

    if "market_cap" not in data.columns:
        data["market_cap"] = pd.to_numeric(data["close"], errors="coerce") * pd.to_numeric(data["volume"], errors="coerce")

    data = data.sort_values(["date", "asset"]).drop_duplicates(subset=["date", "asset"]).reset_index(drop=True)
    data = data.dropna(subset=["close", "volume"]).reset_index(drop=True)
    return data


def summarize_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
        "end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
        "latest_close": float(df["close"].iloc[-1]) if not df.empty else None,
    }


def summarize_cross_section_data(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "asset_count": int(df["asset"].nunique()) if not df.empty and "asset" in df.columns else 0,
        "start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
        "end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
        "latest_trade_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
    }

