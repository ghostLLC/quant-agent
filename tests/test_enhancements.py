"""Tests for factor_enhancements.py — experience loop, combiner, regime detector, crowding, curve analyzer, orthogonality guide."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantlab.factor_discovery.factor_enhancements import (
    CrowdingDetector,
    ExperienceLoop,
    FactorCombiner,
    FactorCurveAnalyzer,
    FactorOutcome,
    OrthogonalityGuide,
    RegimeDetector,
)


def make_synth_df(n_dates=60, n_assets=100, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}" for i in range(n_assets)]
    rows = []
    for d in dates:
        for a in assets:
            rows.append(
                {
                    "date": d,
                    "asset": a,
                    "close": rng.uniform(10, 100),
                    "industry": rng.choice(["A", "B", "C"]),
                    "volume": rng.uniform(1e5, 1e7),
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["asset", "date"])
    df["fwd_ret_5"] = df.groupby("asset")["close"].transform(lambda x: x.shift(-5) / x - 1)
    return df


def make_factor_outcome(
    outcome_id="test_1",
    direction="momentum",
    hypothesis_intuition="price momentum with volume",
    mechanism="delta + rank crossover",
    pseudocode="rank(ts_delta(close, 10)) * rank(volume)",
    input_fields=None,
    block_tree_desc="transform:rank(input:transform:ts_delta(field:close))",
    verdict="useful",
    rank_ic=0.05,
    ic_ir=0.5,
    coverage=0.95,
    risk_exposure=None,
):
    if input_fields is None:
        input_fields = ["close", "volume"]
    if risk_exposure is None:
        risk_exposure = {"market_cap": 0.05, "industry": 0.02, "momentum": 0.01}
    return FactorOutcome(
        outcome_id=outcome_id,
        direction=direction,
        hypothesis_intuition=hypothesis_intuition,
        mechanism=mechanism,
        pseudocode=pseudocode,
        input_fields=input_fields,
        block_tree_desc=block_tree_desc,
        verdict=verdict,
        rank_ic=rank_ic,
        ic_ir=ic_ir,
        coverage=coverage,
        risk_exposure=risk_exposure,
        run_id="run_test",
    )


# ---------------------------------------------------------------------------
# FactorOutcome
# ---------------------------------------------------------------------------

class TestFactorOutcome:
    def test_serialization_roundtrip(self):
        original = make_factor_outcome(
            outcome_id="roundtrip_test",
            direction="reversal",
            hypothesis_intuition="short-term mean reversion",
            mechanism="ts_zscore of close",
            pseudocode="-ts_zscore(close, 5)",
            input_fields=["close"],
            block_tree_desc="transform:ts_zscore(field:close,window=5)",
            verdict="marginal",
            rank_ic=0.025,
            ic_ir=0.30,
            coverage=0.88,
            risk_exposure={"market_cap": 0.10, "industry": 0.03, "momentum": 0.02},
        )
        d = original.to_dict()
        restored = FactorOutcome.from_dict(d)
        assert restored.outcome_id == original.outcome_id
        assert restored.direction == original.direction
        assert restored.hypothesis_intuition == original.hypothesis_intuition
        assert restored.mechanism == original.mechanism
        assert restored.pseudocode == original.pseudocode
        assert restored.input_fields == original.input_fields
        assert restored.block_tree_desc == original.block_tree_desc
        assert restored.verdict == original.verdict
        assert restored.rank_ic == original.rank_ic
        assert restored.ic_ir == original.ic_ir
        assert restored.coverage == original.coverage
        assert restored.risk_exposure == original.risk_exposure
        assert restored.run_id == original.run_id
        assert isinstance(restored, FactorOutcome)


# ---------------------------------------------------------------------------
# ExperienceLoop
# ---------------------------------------------------------------------------

class TestExperienceLoop:
    def test_record_and_retrieve(self, tmp_path):
        loop = ExperienceLoop(store_path=tmp_path / "exp_loop")
        o1 = make_factor_outcome(outcome_id="o1", verdict="useful", rank_ic=0.06, ic_ir=0.6)
        o2 = make_factor_outcome(outcome_id="o2", verdict="marginal", rank_ic=0.02, ic_ir=0.2)
        o3 = make_factor_outcome(outcome_id="o3", verdict="useless", rank_ic=0.001, ic_ir=0.01)
        for o in [o1, o2, o3]:
            loop.record(o)

        guidance = loop.get_guidance("momentum")
        assert guidance["total_recorded"] == 3
        assert len(guidance["successful_patterns"]) == 1  # only o1 is useful
        assert len(guidance["marginal_patterns"]) == 1  # o2 is marginal
        assert len(guidance["failed_patterns"]) == 1  # o3 is useless

    def test_empty(self, tmp_path):
        loop = ExperienceLoop(store_path=tmp_path / "exp_empty")
        guidance = loop.get_guidance("momentum")
        assert guidance["total_recorded"] == 0
        assert guidance["successful_patterns"] == []
        assert "暂无历史经验" in guidance["direction_insight"]

    def test_structure_stats(self, tmp_path):
        loop = ExperienceLoop(store_path=tmp_path / "exp_stats")

        o1 = make_factor_outcome(outcome_id="s1", block_tree_desc="delta+rank", verdict="useful")
        o2 = make_factor_outcome(outcome_id="s2", block_tree_desc="delta+rank", verdict="useful")
        o3 = make_factor_outcome(outcome_id="s3", block_tree_desc="ts_std+group_neutralize", verdict="useless")
        o4 = make_factor_outcome(outcome_id="s4", block_tree_desc="ts_std+group_neutralize", verdict="marginal")
        for o in [o1, o2, o3, o4]:
            loop.record(o)

        guidance = loop.get_guidance("momentum")
        stats = guidance["structure_stats"]

        delta_rank_key = next(k for k in stats if "delta+rank" in k)
        assert stats[delta_rank_key]["win_rate"] == 1.0
        assert stats[delta_rank_key]["total_attempts"] == 2
        assert stats[delta_rank_key]["useful_count"] == 2

        ts_std_key = next(k for k in stats if "ts_std+group" in k)
        assert stats[ts_std_key]["win_rate"] == 0.0
        assert stats[ts_std_key]["total_attempts"] == 2


# ---------------------------------------------------------------------------
# FactorCombiner
# ---------------------------------------------------------------------------

class TestFactorCombiner:
    @staticmethod
    def _make_factor_panel(df, seed=100):
        """Create a factor value Series indexed to match df, with some predictive power."""
        rng = np.random.default_rng(seed)
        factor_vals = df.groupby("asset")["close"].transform(
            lambda x: x.rolling(10, min_periods=3).mean() / x - 1
        )
        noise = pd.Series(rng.normal(0, 0.02, len(df)), index=df.index)
        return factor_vals.fillna(0) + noise.values

    def test_combine_empty_raises(self):
        combiner = FactorCombiner()
        df = make_synth_df(n_dates=60, n_assets=50)
        with pytest.raises(ValueError, match="为空"):
            combiner.combine({}, df)

    def test_combine_single_factor(self):
        combiner = FactorCombiner()
        df = make_synth_df(n_dates=60, n_assets=50)
        panel = self._make_factor_panel(df, seed=101)
        result = combiner.combine({"f1": panel}, df, method="equal_weight")
        assert len(result.factor_ids) == 1
        assert result.weights == {"f1": 1.0}

    def test_combine_ic_weighted(self):
        combiner = FactorCombiner()
        df = make_synth_df(n_dates=60, n_assets=50)
        p1 = self._make_factor_panel(df, seed=201)
        p2 = self._make_factor_panel(df, seed=202)
        p3 = self._make_factor_panel(df, seed=203)
        result = combiner.combine({"f1": p1, "f2": p2, "f3": p3}, df, method="ic_weighted")
        assert len(result.factor_ids) >= 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        assert isinstance(result.combined_ic, float)
        assert result.method == "ic_weighted"

    def test_combine_equal_weight(self):
        combiner = FactorCombiner()
        df = make_synth_df(n_dates=60, n_assets=50)
        # Use distinct factor constructions so orthogonal selection keeps all 3
        p1 = df.groupby("asset")["close"].transform(lambda x: x.rolling(20, min_periods=3).mean() / x - 1)
        p2 = df.groupby("asset")["close"].transform(lambda x: x.pct_change(5))
        p3 = df.groupby("asset")["volume"].transform(lambda x: x.rolling(10, min_periods=3).mean())
        result = combiner.combine(
            {"f1": p1, "f2": p2, "f3": p3}, df,
            method="equal_weight", correlation_threshold=0.99,
        )
        n = len(result.factor_ids)
        expected = round(1.0 / n, 4)
        for w in result.weights.values():
            assert w == expected

    def test_orthogonal_selection_reduces_correlated(self):
        combiner = FactorCombiner()
        df = make_synth_df(n_dates=60, n_assets=50)
        base = self._make_factor_panel(df, seed=400)
        # f2 is almost identical to f1 → highly correlated
        p1 = base
        p2 = base + pd.Series(np.random.default_rng(401).normal(0, 0.001, len(base)), index=base.index)
        # f3 is quite different
        p3 = self._make_factor_panel(df, seed=402)
        result = combiner.combine(
            {"f1": p1, "f2": p2, "f3": p3}, df,
            method="equal_weight", correlation_threshold=0.7,
        )
        assert len(result.factor_ids) <= 2  # f1+f2 are correlated, one gets dropped


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    @staticmethod
    def _make_trendy_df(n_dates=120, n_assets=20, seed=42):
        """Synthetic market with clear bull (first 40 days) and bear (last 40 days) regimes."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
        assets = [f"{i:06d}" for i in range(n_assets)]
        base_price = rng.uniform(10, 100, size=n_assets)

        rows = []
        for i, d in enumerate(dates):
            for j, a in enumerate(assets):
                if i < 40:
                    trend = 1 + 0.01 * i  # bull: ~+40% over 40 days
                elif i < 80:
                    trend = 1 + 0.01 * 40  # sideways: flat
                else:
                    trend = (1 + 0.01 * 40) * (1 - 0.01 * (i - 80))  # bear: declining
                noise = rng.normal(0, 0.005)
                close = base_price[j] * trend * (1 + noise)
                rows.append({"date": d, "asset": a, "close": close, "industry": rng.choice(["A", "B", "C"])})
        return pd.DataFrame(rows)

    def test_detect_bull_bear(self):
        df = self._make_trendy_df(n_dates=120, n_assets=20)
        detector = RegimeDetector(fast_ma=10, slow_ma=30)
        classification = detector.detect(df)

        assert "bull" in classification.regime_stats
        assert "bear" in classification.regime_stats
        assert "sideways" in classification.regime_stats
        assert classification.current_regime in ("bull", "bear", "sideways")

        bear_days = classification.regime_stats["bear"]["n_days"]
        bull_days = classification.regime_stats["bull"]["n_days"]
        # With 120 days minus slow_ma=30 buffer, we should have some bull and bear
        assert bull_days > 0 or bear_days > 0, f"Expected non-zero bull or bear days, got bull={bull_days}, bear={bear_days}"

    def test_regime_adjusted_ic_keys(self):
        df = self._make_trendy_df(n_dates=120, n_assets=20)
        dates = pd.date_range("2025-01-01", periods=120, freq="B")
        detector = RegimeDetector(fast_ma=10, slow_ma=30)
        classification = detector.detect(df)

        # Build factor values: simple momentum
        factor_vals = df.groupby("asset")["close"].transform(lambda x: x.pct_change(10).shift(-5))

        result = detector.regime_adjusted_ic(factor_vals, df, classification)
        assert "bull_ic" in result
        assert "bear_ic" in result
        assert "sideways_ic" in result
        assert "regime_adj_ic" in result
        assert "regime_adj_icir" in result
        assert isinstance(result["regime_adj_ic"], float)


# ---------------------------------------------------------------------------
# FactorCurveAnalyzer
# ---------------------------------------------------------------------------

class TestFactorCurveAnalyzer:
    def test_ic_decay_curve_keys(self):
        df = make_synth_df(n_dates=60, n_assets=50)
        analyzer = FactorCurveAnalyzer()
        factor = df.groupby("asset")["close"].transform(lambda x: x.pct_change(10))
        result = analyzer.ic_decay_curve(factor, df, windows=[5, 10, 20])
        assert "curves" in result
        assert "best_window" in result
        assert "best_ic" in result
        assert "half_life" in result
        assert set(result["curves"].keys()) == {5, 10, 20}


# ---------------------------------------------------------------------------
# CrowdingDetector
# ---------------------------------------------------------------------------

class TestCrowdingDetector:
    def test_detect_empty_store(self):
        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = []

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ):
            detector = CrowdingDetector()
            report = detector.detect(min_factors=2)
            assert report.distance_matrix == {}
            assert report.clusters == []
            assert report.crowding_scores == {}
            assert report.crowded_factor_ids == []
            assert report.max_observed_corr == 0.0
            assert report.avg_observed_corr == 0.0

    def test_detect_clusters(self, tmp_path):
        """3 correlated factor panels → clusters found."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        assets = [f"{i:06d}" for i in range(10)]
        base = rng.normal(0, 1, len(dates) * len(assets))

        def build_panel(noise_scale, seed):
            r = np.random.default_rng(seed)
            vals = base + r.normal(0, noise_scale, len(base))
            rows = []
            idx = 0
            for d in dates:
                for a in assets:
                    rows.append({"date": d, "asset": a, "factor_value": vals[idx]})
                    idx += 1
            return pd.DataFrame(rows)

        panel_dir = tmp_path / "panels"
        panel_dir.mkdir()
        p1 = build_panel(0.3, 501)
        p2 = build_panel(0.3, 502)  # highly correlated with p1 (same base + small noise)
        p3 = build_panel(2.0, 503)  # less correlated

        paths = []
        for i, panel in enumerate([p1, p2, p3], start=1):
            p = panel_dir / f"f{i}.csv"
            panel.to_csv(p, index=False)
            paths.append(p)

        mock_entry_1 = MagicMock()
        mock_entry_1.factor_spec.factor_id = "f1"
        mock_entry_1.factor_spec.status = "approved"
        mock_entry_1.panel_snapshot_path = str(paths[0])

        mock_entry_2 = MagicMock()
        mock_entry_2.factor_spec.factor_id = "f2"
        mock_entry_2.factor_spec.status = "approved"
        mock_entry_2.panel_snapshot_path = str(paths[1])

        mock_entry_3 = MagicMock()
        mock_entry_3.factor_spec.factor_id = "f3"
        mock_entry_3.factor_spec.status = "approved"
        mock_entry_3.panel_snapshot_path = str(paths[2])

        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = [mock_entry_1, mock_entry_2, mock_entry_3]

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ):
            detector = CrowdingDetector()
            report = detector.detect(correlation_threshold=0.6, min_factors=2)

        # f1 and f2 are highly correlated, should form a cluster or at least have high crowding score
        assert "f1" in report.crowding_scores
        assert "f2" in report.crowding_scores
        # The avg correlation between f1 and f2 should be high
        assert report.crowding_scores["f1"] > 0.5


# ---------------------------------------------------------------------------
# OrthogonalityGuide
# ---------------------------------------------------------------------------

class TestOrthogonalityGuide:
    def test_empty_experience(self, tmp_path):
        guide = OrthogonalityGuide(store_path=tmp_path / "ortho_empty")
        ctx = guide.get_orthogonality_context("momentum")
        assert ctx["total_existing_factors"] == 0
        assert ctx["covered_fields"] == []
        assert ctx["covered_structures"] == []
        assert "因子库为空" in ctx["orthogonality_hint"]

    def test_with_outcomes(self, tmp_path):
        # Pre-populate outcomes file
        store = tmp_path / "ortho_populated"
        store.mkdir()
        outcomes = [
            make_factor_outcome(
                outcome_id="o1", direction="momentum",
                input_fields=["close", "volume"], block_tree_desc="delta_rank",
                verdict="useful",
            ),
            make_factor_outcome(
                outcome_id="o2", direction="momentum",
                input_fields=["close"], block_tree_desc="ts_std",
                verdict="useful",
            ),
            make_factor_outcome(
                outcome_id="o3", direction="momentum",
                input_fields=["close", "volume"], block_tree_desc="delta_rank",
                verdict="marginal",
            ),
            make_factor_outcome(
                outcome_id="o4", direction="momentum",
                input_fields=["close"], block_tree_desc="ts_std",
                verdict="useless",
            ),
        ]
        db_file = store / "outcomes.json"
        db_file.write_text(
            json.dumps([o.to_dict() for o in outcomes], ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )

        guide = OrthogonalityGuide(store_path=store)
        ctx = guide.get_orthogonality_context("momentum")
        assert ctx["total_existing_factors"] == 4
        assert len(ctx["covered_fields"]) > 0
        assert len(ctx["covered_structures"]) > 0
        assert isinstance(ctx["saturated_directions"], list)
        assert isinstance(ctx["orthogonality_hint"], str)
