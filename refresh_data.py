"""Refresh the thick cross-section dataset incrementally.

Extends data/cross_section_thick.csv by pulling new OHLCV rows
from tushare for all existing assets (new dates only).

For a full historical extension: python build_dataset.py --full
To use HS300 as-is:          python build_dataset.py --use-snapshot
"""

from __future__ import annotations

from build_dataset import incremental_refresh, THICK_DATASET


def main() -> None:
    incremental_refresh(THICK_DATASET)


if __name__ == "__main__":
    main()
