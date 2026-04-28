"""Shared test fixtures."""
from pathlib import Path
import sys

import pandas as pd
import pytest

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))


@pytest.fixture(scope="session")
def project_root():
    return _project_root


@pytest.fixture(scope="session")
def sample_market_df():
    """Small synthetic cross-section for unit tests."""
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    assets = ["000001", "000002", "000003", "000004", "000005"]
    rows = []
    for date in rng:
        for i, asset in enumerate(assets):
            base_price = 10 + i * 5
            noise = (hash(f"{date}{asset}") % 100 - 50) / 100.0
            rows.append({
                "date": date,
                "asset": asset,
                "close": base_price + noise,
                "volume": 1e6 + i * 5e5 + noise * 1e5,
                "industry": ["bank", "tech", "consumer", "industry", "energy"][i],
                "market_cap": (base_price + noise) * (1e6 + i * 5e5),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="session")
def sample_price_df():
    """Single-asset price series for backtest tests."""
    rng = pd.date_range("2023-01-01", periods=252, freq="B")
    import numpy as np
    close = 100 * np.cumprod(1 + np.random.default_rng(42).normal(0.001, 0.015, len(rng)))
    return pd.DataFrame({
        "date": rng,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1e7,
    })
