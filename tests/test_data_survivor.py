"""Tests for data quality monitoring, survivorship filtering, and multi-asset context."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from quantlab.factor_discovery.datahub import DataQualityMonitor, MultiAssetContext
from quantlab.factor_discovery.survivorship import SurvivorshipFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def long_history_df():
    """200-day, 50-asset synthetic cross-section with deterministic prices."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    assets = [f"{i:06d}" for i in range(50)]
    rows = []
    for d in dates:
        for a in assets:
            base = 10 + (int(a) % 50) * 2
            close = base + rng.normal(0, base * 0.02)
            rows.append({
                "date": d,
                "asset": a,
                "close": max(close, 0.01),
                "volume": rng.uniform(1e5, 1e7),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DataQualityMonitor
# ---------------------------------------------------------------------------


class TestDataQualityMonitor:
    def test_check_valid_csv(self, tmp_path):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "asset": ["000001"] * 10,
            "close": np.random.uniform(10, 100, 10),
            "volume": np.random.uniform(1e5, 1e7, 10),
        })
        path = tmp_path / "valid.csv"
        df.to_csv(path, index=False)
        result = DataQualityMonitor.check(str(path))
        assert result["status"] == "ok"
        assert result["avg_missing_rate"] < 1.0

    def test_check_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.csv"
        result = DataQualityMonitor.check(str(path))
        assert result["status"] == "error"

    def test_check_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("date,asset,close\n")
        result = DataQualityMonitor.check(str(path))
        assert result["status"] == "empty"


# ---------------------------------------------------------------------------
# DataQualityMonitor.quality_score
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_quality_score_perfect(self):
        report = {
            "status": "ok",
            "avg_missing_rate": 0.0,
            "stale_date": 0,
            "total_rows": 100,
            "total_columns": 5,
            "total_outliers": 0,
        }
        score = DataQualityMonitor.quality_score(report)
        assert score >= 0.9

    def test_quality_score_bad(self):
        report = {
            "status": "ok",
            "avg_missing_rate": 0.5,
            "stale_date": 100,
            "total_rows": 1000,
            "total_columns": 5,
            "total_outliers": 200,
        }
        score = DataQualityMonitor.quality_score(report)
        assert score < 0.5

    def test_quality_score_error_is_zero(self):
        score = DataQualityMonitor.quality_score({"status": "error"})
        assert score == 0.0

    def test_quality_score_empty_is_zero(self):
        score = DataQualityMonitor.quality_score({"status": "empty"})
        assert score == 0.0


# ---------------------------------------------------------------------------
# SurvivorshipFilter
# ---------------------------------------------------------------------------


class TestSurvivorshipFilter:
    def test_filter_normal(self, long_history_df):
        sf = SurvivorshipFilter(min_history_days=5, lookahead_days=5)
        result_df, summary = sf.filter(long_history_df)
        assert len(result_df) > 0
        assert len(result_df) <= len(long_history_df)
        # All assets present throughout, so retention should be high
        assert summary["retention_pct"] >= 90.0
        assert summary["new_listing_dropped_assets"] == 0
        assert summary["delisting_dropped_assets"] == 0

    def test_filter_new_listing(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        assets_normal = [f"{i:06d}" for i in range(10)]
        assets_late = [f"{i:06d}" for i in range(90, 95)]

        rows = []
        for d in dates:
            for a in assets_normal:
                rows.append({"date": d, "asset": a, "close": 50.0 + rng.normal(0, 1)})
            # Late assets only appear after 60 calendar days
            if d >= pd.Timestamp("2024-03-01"):
                for a in assets_late:
                    rows.append({"date": d, "asset": a, "close": 55.0 + rng.normal(0, 1)})

        df = pd.DataFrame(rows)
        sf = SurvivorshipFilter(min_history_days=30, lookahead_days=5)
        result_df, summary = sf.filter(df)

        remaining_assets = set(result_df["asset"].unique())
        for a in assets_late:
            assert a not in remaining_assets, f"Late-listing asset {a} should be dropped"
        assert summary["new_listing_dropped_assets"] >= len(assets_late)

    def test_filter_delisting(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        assets_normal = [f"{i:06d}" for i in range(10)]
        assets_delisted = [f"{i:06d}" for i in range(90, 92)]

        rows = []
        for d in dates:
            for a in assets_normal:
                rows.append({"date": d, "asset": a, "close": 50.0 + rng.normal(0, 1)})
            # Delisted assets disappear after an early cutoff
            if d < pd.Timestamp("2024-02-15"):
                for a in assets_delisted:
                    rows.append({"date": d, "asset": a, "close": 55.0 + rng.normal(0, 1)})

        df = pd.DataFrame(rows)
        sf = SurvivorshipFilter(min_history_days=5, lookahead_days=20)
        result_df, summary = sf.filter(df)

        remaining_assets = set(result_df["asset"].unique())
        for a in assets_delisted:
            assert a not in remaining_assets, f"Delisted asset {a} should be dropped"
        assert summary["delisting_dropped_assets"] >= len(assets_delisted)

    def test_filter_extreme_returns(self):
        """5-day return below -60% should set close to NaN."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        asset = "000050"
        closes = [100, 101, 102, 103, 104, 40, 41, 42, 43, 44]

        rows = [{"date": d, "asset": asset, "close": c} for d, c in zip(dates, closes)]
        # Add some normal assets to keep per-date count above min_assets_per_date
        for i in range(30):
            for d, c in zip(dates, np.linspace(20, 25, len(dates))):
                rows.append({"date": d, "asset": f"{i:06d}", "close": c + np.random.default_rng(42).normal(0, 0.1)})

        df = pd.DataFrame(rows)
        sf = SurvivorshipFilter(min_history_days=1, lookahead_days=1, min_assets_per_date=5)
        result_df, summary = sf.filter(df)

        # Row where close=40 (pct_change(5) ≈ -60%) should now be NaN
        asset_rows = result_df[result_df["asset"] == asset].sort_values("date")
        extreme_idx = asset_rows.index[asset_rows["close"].isna()]
        assert len(extreme_idx) > 0, "Expected at least one close value to be set to NaN"


# ---------------------------------------------------------------------------
# MultiAssetContext
# ---------------------------------------------------------------------------


class TestMultiAssetContext:
    def test_register(self, tmp_path):
        ctx = MultiAssetContext()
        ctx.register("equity_a", data_path=str(tmp_path / "equity.csv"),
                     store_dir=str(tmp_path / "store_a"))
        ctx.register("futures_b", data_path=str(tmp_path / "futures.csv"),
                     store_dir=str(tmp_path / "store_b"))
        classes = ctx.registered_classes()
        assert "equity_a" in classes
        assert "futures_b" in classes
        assert len(classes) == 2

    def test_get_hub_unknown(self):
        ctx = MultiAssetContext()
        with pytest.raises(KeyError, match="not registered"):
            ctx.get_hub("nonexistent_class")

    def test_get_store_unknown(self):
        ctx = MultiAssetContext()
        with pytest.raises(KeyError, match="not registered"):
            ctx.get_store("nonexistent_class")

    def test_register_all(self):
        ctx = MultiAssetContext()
        ctx.register_all_known_asset_classes()
        classes = ctx.registered_classes()
        assert len(classes) == 4
        assert "a_share_equity" in classes
        assert "index_future" in classes
        assert "commodity_future" in classes
        assert "convertible_bond" in classes
