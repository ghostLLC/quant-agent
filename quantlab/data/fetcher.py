from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

ETF_SYMBOL = "510300"
ETF_SINA_SYMBOL = "sh510300"


def _normalize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"ETF 数据缺少必要字段: {missing}")

    normalized = data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return normalized


def fetch_from_sina(symbol: str = ETF_SINA_SYMBOL) -> pd.DataFrame:
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df.empty:
        raise ValueError(f"新浪源未获取到 ETF 数据: {symbol}")
    return _normalize_dataframe(df)


def fetch_from_eastmoney(symbol: str = ETF_SYMBOL, max_retries: int = 3, retry_delay: float = 2.0) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
            if df.empty:
                raise ValueError(f"未获取到 ETF 数据: {symbol}")
            data = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_chg",
                    "涨跌额": "change",
                    "换手率": "turnover",
                }
            )
            return _normalize_dataframe(data)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                raise RuntimeError(f"东方财富源抓取 ETF 数据失败，已重试 {max_retries} 次") from exc
            time.sleep(retry_delay)
    raise RuntimeError("东方财富源抓取 ETF 数据失败") from last_error


def fetch_hs300_etf_history() -> pd.DataFrame:
    try:
        return fetch_from_sina()
    except Exception:
        return fetch_from_eastmoney()


def update_hs300_etf_csv(output_path: str | Path) -> pd.DataFrame:
    data = fetch_hs300_etf_history()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    return data
