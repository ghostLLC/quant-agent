from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LATEST_REPORT_DIR = REPORTS_DIR / "latest"
HISTORY_REPORT_DIR = REPORTS_DIR / "history"
DEFAULT_DATA_PATH = DATA_DIR / "hs300_etf.csv"
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_WALK_FORWARD_TRAIN_WINDOW = 504
DEFAULT_WALK_FORWARD_TEST_WINDOW = 126
DEFAULT_WALK_FORWARD_STEP_SIZE = 126


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    short_window: int = 5
    long_window: int = 20
    benchmark_symbol: str = "510300"
    trading_days_per_year: int = 252
    stop_loss_pct: float | None = None
    min_holding_days: int = 0
    enable_trend_filter: bool = False
    trend_window: int = 60
    enable_volatility_filter: bool = False
    volatility_window: int = 20
    max_volatility: float | None = None
    train_ratio: float = DEFAULT_TRAIN_RATIO
    walk_forward_train_window: int = DEFAULT_WALK_FORWARD_TRAIN_WINDOW
    walk_forward_test_window: int = DEFAULT_WALK_FORWARD_TEST_WINDOW
    walk_forward_step_size: int = DEFAULT_WALK_FORWARD_STEP_SIZE
    report_dir: Path = field(default_factory=lambda: LATEST_REPORT_DIR)
    history_dir: Path = field(default_factory=lambda: HISTORY_REPORT_DIR)


DEFAULT_GRID = {
    "short_window": [5, 10, 15],
    "long_window": [20, 30, 60],
    "enable_trend_filter": [False, True],
    "stop_loss_pct": [None, 0.05, 0.08],
}
