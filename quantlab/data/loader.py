from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


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


def summarize_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
        "end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
        "latest_close": float(df["close"].iloc[-1]) if not df.empty else None,
    }
