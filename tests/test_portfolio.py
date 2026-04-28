"""Tests for portfolio construction (post-bugfix)."""
import numpy as np
import pandas as pd

from quantlab.trading.portfolio import FactorPortfolioConstructor, PortfolioWeightScheme


class TestPortfolioConstructor:
    def make_panel(self, n_assets=20, n_dates=10):
        rng = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        rows = []
        for d in rng:
            for i in range(n_assets):
                rows.append({
                    "date": d,
                    "asset": f"{i + 1:06d}",
                    "factor_value": np.random.default_rng(i * 7 + d.day).normal(0, 1),
                })
        return pd.DataFrame(rows)

    def test_equal_weight_sums_to_one(self):
        panel = self.make_panel()
        ctor = FactorPortfolioConstructor(n_long=10, weight_scheme=PortfolioWeightScheme.EQUAL)
        weights = ctor.construct_weights(panel)
        for _, group in weights.groupby("date"):
            assert abs(group["weight"].sum() - 1.0) < 0.001

    def test_equal_weight_uniform(self):
        panel = self.make_panel()
        ctor = FactorPortfolioConstructor(n_long=10, weight_scheme=PortfolioWeightScheme.EQUAL)
        weights = ctor.construct_weights(panel)
        for _, group in weights.groupby("date"):
            assert len(group["weight"].unique()) == 1

    def test_score_weight_positive(self):
        panel = self.make_panel()
        ctor = FactorPortfolioConstructor(n_long=10, weight_scheme=PortfolioWeightScheme.SCORE)
        weights = ctor.construct_weights(panel)
        assert (weights["weight"] > 0).all()

    def test_all_schemes_produce_valid_weights(self):
        panel = self.make_panel()
        for scheme in PortfolioWeightScheme:
            ctor = FactorPortfolioConstructor(n_long=10, weight_scheme=scheme)
            weights = ctor.construct_weights(panel)
            if not weights.empty:
                for _, group in weights.groupby("date"):
                    assert abs(group["weight"].sum() - 1.0) < 0.001
                    assert (group["weight"] > 0).all()

    def test_empty_panel_returns_empty(self):
        ctor = FactorPortfolioConstructor(n_long=10)
        empty = pd.DataFrame(columns=["date", "asset", "factor_value"])
        result = ctor.construct_weights(empty)
        assert result.empty

    def test_weights_respect_min_max(self):
        panel = self.make_panel()
        ctor = FactorPortfolioConstructor(
            n_long=10,
            weight_scheme=PortfolioWeightScheme.EQUAL,
            min_single_weight=0.01,
            max_single_weight=0.3,
        )
        weights = ctor.construct_weights(panel)
        assert weights["weight"].min() >= 0.01
        assert weights["weight"].max() <= 0.3
