from __future__ import annotations

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH
from quantlab.pipeline import refresh_cross_section_data


def main() -> None:
    summary = refresh_cross_section_data(DEFAULT_CROSS_SECTION_DATA_PATH)
    print(
        f"已更新横截面数据，共 {summary['rows']} 行、{summary['asset_count']} 个资产，"
        f"区间 {summary['start_date']} ~ {summary['end_date']}"
    )


if __name__ == "__main__":
    main()

