"""Tests for GPU-accelerated factor IC computation."""
import numpy as np
import pandas as pd
import pytest

from quantlab.metrics.gpu_accelerator import GpuAccelerator


def _make_synth_data(n_dates=10, n_assets=50, seed=42):
    """Create synthetic market data and factor values for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}" for i in range(n_assets)]
    rows = []
    for d in dates:
        for a in assets:
            rows.append({
                "date": d,
                "asset": a,
                "close": rng.uniform(10, 100),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    # Generate factor values: flat Series aligned to df index
    factor_vals = rng.uniform(-2, 2, len(df))
    factor = pd.Series(factor_vals, index=df.index, name="factor")

    return df, factor


def _make_multifactor_data(n_dates=10, n_assets=50, n_factors=3, seed=42):
    """Create synthetic market data and a dict of factor panels."""
    df, _ = _make_synth_data(n_dates, n_assets, seed)
    factors: dict[str, pd.Series] = {}
    for i in range(n_factors):
        rng = np.random.default_rng(seed + i)
        vals = rng.uniform(-2, 2, len(df))
        factors[f"factor_{i}"] = pd.Series(vals, index=df.index, name=f"factor_{i}")
    return df, factors


class TestGpuAccelerator:
    """Tests for GPU-accelerated IC computation."""

    def test_gpu_available_returns_bool(self):
        """gpu_available() returns a boolean."""
        result = GpuAccelerator.gpu_available()
        assert isinstance(result, bool)

    def test_compute_rank_ic_gpu_returns_dict(self):
        """compute_rank_ic_gpu returns a dict with expected keys."""
        df, factor = _make_synth_data()
        result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=2)
        assert isinstance(result, dict)
        for key in ("ic_mean", "rank_ic_mean", "ic_ir", "coverage"):
            assert key in result
            assert isinstance(result[key], float)

    def test_compute_rank_ic_gpu_fallback_to_cpu(self):
        """When GPU is not available, compute_rank_ic_gpu calls CPU version
        by comparing results to the CPU compute_rank_ic directly."""
        from quantlab.metrics.ic_calculator import compute_rank_ic

        df, factor = _make_synth_data(seed=123)
        gpu_result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=2)
        cpu_result = compute_rank_ic(factor, df, forward_days=2)

        # Results should match (GPU path either uses GPU or falls back to CPU)
        for key in ("ic_mean", "rank_ic_mean", "ic_ir", "coverage"):
            assert abs(gpu_result[key] - cpu_result[key]) < 1e-6, (
                f"Mismatch for {key}: GPU={gpu_result[key]}, CPU={cpu_result[key]}"
            )

    def test_compute_rank_ic_gpu_consistency(self):
        """GPU and CPU results are structurally consistent."""
        df, factor = _make_synth_data(seed=456)

        result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=3)
        assert set(result.keys()) == {"ic_mean", "rank_ic_mean", "ic_ir", "coverage"}
        assert -1.0 <= result["ic_mean"] <= 1.0
        assert -1.0 <= result["rank_ic_mean"] <= 1.0
        assert result["coverage"] >= 0.0

    def test_batch_compute_ic_returns_dict(self):
        """batch_compute_ic returns correct dict structure."""
        df, factors = _make_multifactor_data(n_factors=2)
        result = GpuAccelerator.batch_compute_ic(factors, df, forward_days=2)
        assert isinstance(result, dict)
        assert len(result) == 2
        for fid in ("factor_0", "factor_1"):
            assert fid in result
            assert isinstance(result[fid], dict)
            assert "ic_mean" in result[fid]

    def test_batch_compute_ic_multiple_factors(self):
        """batch_compute_ic handles multiple factors correctly."""
        df, factors = _make_multifactor_data(n_factors=3, seed=789)
        result = GpuAccelerator.batch_compute_ic(factors, df, forward_days=2)
        assert set(result.keys()) == {"factor_0", "factor_1", "factor_2"}
        for fid, ic in result.items():
            assert -1.0 <= ic["ic_mean"] <= 1.0, f"{fid} ic_mean out of range: {ic['ic_mean']}"
            assert ic["coverage"] >= 0.0, f"{fid} coverage negative: {ic['coverage']}"

    def test_gpu_batch_parallel_processing(self):
        """batch_compute_ic processes all factor IDs."""
        df, factors = _make_multifactor_data(n_factors=5, seed=101)
        result = GpuAccelerator.batch_compute_ic(factors, df, forward_days=2)
        expected_ids = {f"factor_{i}" for i in range(5)}
        assert set(result.keys()) == expected_ids
        # Every factor should have a valid result
        for fid in expected_ids:
            assert isinstance(result[fid]["ic_mean"], float)
            assert isinstance(result[fid]["ic_ir"], float)

    def test_multiindex_factor_values(self):
        """compute_rank_ic_gpu handles MultiIndex (date, asset) factor Series."""
        df, _ = _make_synth_data(seed=202)
        # Create a MultiIndex factor Series
        rng = np.random.default_rng(seed=303)
        mi_index = pd.MultiIndex.from_frame(df[["date", "asset"]])
        factor_values = pd.Series(rng.uniform(-2, 2, len(df)), index=mi_index, name="factor")
        result = GpuAccelerator.compute_rank_ic_gpu(factor_values, df, forward_days=2)
        assert isinstance(result, dict)
        assert "ic_mean" in result

    def test_missing_close_column(self):
        """Returns zero result when close column is missing."""
        df = pd.DataFrame({"date": [1, 2], "asset": ["a", "b"]})
        factor = pd.Series([1.0, 2.0])
        result = GpuAccelerator.compute_rank_ic_gpu(factor, df)
        assert result["ic_mean"] == 0.0
        assert result["rank_ic_mean"] == 0.0

    def test_insufficient_samples(self):
        """Returns zero result when no date group has enough samples."""
        df, factor = _make_synth_data(n_dates=5, n_assets=3, seed=505)
        # min_samples=100 should be impossible with only 3 assets
        result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=2, min_samples=100)
        assert result["ic_mean"] == 0.0

    def test_auto_detect_asset_col(self):
        """Auto-detects asset column when asset_col is None."""
        df, factor = _make_synth_data(seed=606)
        result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=2, asset_col=None)
        assert isinstance(result, dict)
        assert "ic_mean" in result

    def test_auto_detect_ts_code_col(self):
        """Falls back to ts_code column when asset column is absent."""
        df, factor = _make_synth_data(seed=707)
        df = df.rename(columns={"asset": "ts_code"})
        result = GpuAccelerator.compute_rank_ic_gpu(factor, df, forward_days=2, asset_col=None)
        assert isinstance(result, dict)
        assert "ic_mean" in result
