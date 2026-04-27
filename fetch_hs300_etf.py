from __future__ import annotations

from pathlib import Path

from quantlab.config import DEFAULT_PRICE_DATA_PATH
from quantlab.data.fetcher import update_hs300_etf_csv


def main() -> None:
    data = update_hs300_etf_csv(Path(DEFAULT_PRICE_DATA_PATH))
    print(f"已写入真实沪深300ETF数据: {DEFAULT_PRICE_DATA_PATH}")
    print(f"数据区间: {data['date'].iloc[0]} ~ {data['date'].iloc[-1]}")
    print(f"总行数: {len(data)}")


if __name__ == "__main__":
    main()

