"""Tests for cost model and simulator."""
import pandas as pd
import pytest

from quantlab.trading.cost_model import AShareCostModel, CostModel
from quantlab.trading.portfolio import FactorPortfolioConstructor, PortfolioWeightScheme
from quantlab.trading.simulator import FactorPortfolioSimulator, SimulationResult


class TestCostModel:
    def test_buy_cost_positive(self):
        cm = AShareCostModel()
        rate = cm.buy_cost_rate(10000)
        assert rate > 0

    def test_sell_cost_higher_than_buy(self):
        cm = AShareCostModel()
        buy = cm.buy_cost_rate(10000)
        sell = cm.sell_cost_rate(10000)
        assert sell > buy  # stamp tax on sell side

    def test_min_commission_applied(self):
        cm = AShareCostModel()
        rate = cm.buy_cost_rate(100)  # tiny trade
        assert rate >= 5.0 / 100  # min commission applies

    def test_round_trip_cost(self):
        cm = AShareCostModel()
        rt = cm.round_trip_cost_rate(100000)
        expected_min = cm.commission_rate * 2 + cm.stamp_tax_rate
        assert rt >= expected_min

    def test_capacity_estimate(self):
        cm = AShareCostModel()
        volumes = pd.Series([1e6] * 100)
        cap = cm.estimate_capacity(volumes, max_stocks=50)
        assert cap["total_daily_capacity_yuan"] > 0
        assert cap["max_stocks"] == 50


class TestSimulator:
    def make_factor_panel(self, n_assets=20, n_dates=30):
        rng = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        rows = []
        for d in rng:
            for i in range(n_assets):
                rows.append({
                    "date": d,
                    "asset": f"{i + 1:06d}",
                    "close": 10 + i * 3 + (d.day % 3),
                    "volume": 1e6 + i * 5e5,
                    "factor_value": (i % 5) * 0.5,
                })
        return pd.DataFrame(rows)

    def test_simulator_runs_without_error(self):
        panel = self.make_factor_panel()
        market = panel[["date", "asset", "close", "volume"]].copy()
        sim = FactorPortfolioSimulator(n_long=5, weight_scheme=PortfolioWeightScheme.EQUAL)
        result = sim.simulate(panel, market)
        assert isinstance(result, SimulationResult)
        assert result.trading_days > 0

    def test_empty_panel_returns_empty_result(self):
        empty = pd.DataFrame(columns=["date", "asset", "factor_value"])
        market = pd.DataFrame(columns=["date", "asset", "close", "volume"])
        sim = FactorPortfolioSimulator(n_long=5)
        result = sim.simulate(empty, market)
        assert result.trading_days == 0

    def test_serialization(self):
        result = SimulationResult(
            annual_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            gross_return=0.20,
            net_return=0.15,
            simulation_period="2024-01-01 ~ 2024-12-31",
            trading_days=252,
        )
        d = result.to_dict()
        assert d["annual_return"] == 0.15
        assert d["sharpe_ratio"] == 1.2
