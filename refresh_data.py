from __future__ import annotations

from quantlab.config import DEFAULT_DATA_PATH
from quantlab.pipeline import refresh_market_data


def main() -> None:
    summary = refresh_market_data(DEFAULT_DATA_PATH)
    print(f"已更新数据，共 {summary['rows']} 行，区间 {summary['start_date']} ~ {summary['end_date']}")


if __name__ == "__main__":
    main()
