"""Tests for live multi-factor portfolio paper trading simulator."""
import numpy as np
import pandas as pd

from quantlab.trading.live_simulator import LiveSimulator
from quantlab.trading.portfolio import PortfolioWeightScheme


def make_market_df(n_assets=20, n_dates=5):
    """Create synthetic market data with date, asset, close, volume."""
    rng = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    rows = []
    for d in rng:
        for i in range(n_assets):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "asset": f"{i + 1:06d}",
                "close": 10.0 + i * 2.0 + d.day,
                "volume": 1_000_000 + i * 500_000,
            })
    return pd.DataFrame(rows)


def make_factor_panels(market_df):
    """Create synthetic factor panels dict with MultiIndex Series."""
    assets = sorted(market_df["asset"].unique())
    n_assets = len(assets)
    dates = sorted(market_df["date"].unique())

    # Factor 1: strong positive signal on first half of assets
    factor1_values = []
    for date in dates:
        for i in range(len(assets)):
            val = 1.0 if i < n_assets // 2 else -1.0
            val += np.random.default_rng(i * 7 + int(date[-2:])).normal(0, 0.1)
            factor1_values.append(val)

    # Factor 2: strong positive signal on last half of assets
    factor2_values = []
    for date in dates:
        for i in range(len(assets)):
            val = 1.0 if i >= n_assets // 2 else -1.0
            val += np.random.default_rng(i * 13 + int(date[-2:])).normal(0, 0.1)
            factor2_values.append(val)

    index_tuples = []
    for date in dates:
        for asset in assets:
            index_tuples.append((date, asset))

    idx = pd.MultiIndex.from_tuples(index_tuples, names=["date", "asset"])

    return {
        "factor_momentum": pd.Series(factor1_values, index=idx, name="factor_value"),
        "factor_value": pd.Series(factor2_values, index=idx, name="factor_value"),
    }


class TestLiveSimulator:
    def test_initialization(self):
        """LiveSimulator initializes with correct defaults."""
        ls = LiveSimulator()
        assert ls.nav_history.empty
        assert len(ls.positions) == 0
        assert len(ls.trades) == 0

    def test_run_daily_first_day(self, tmp_path):
        """First day run creates positions and NAV record."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        first_date = sorted(market_df["date"].unique())[0]

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)
        result = ls.run_daily(
            date=first_date,
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )

        assert result["date"] == first_date
        assert result["nav"] > 0
        assert result["n_positions"] > 0
        assert result["n_positions"] <= 10
        assert len(ls.nav_history) == 1
        assert len(ls.positions) > 0

    def test_run_daily_rebalance(self, tmp_path):
        """Second day rebalance adjusts positions."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        # Day 1
        ls.run_daily(
            date=dates[0],
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )

        # Day 2
        ls.run_daily(
            date=dates[1],
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )

        assert len(ls.nav_history) == 2
        assert len(ls.positions) > 0
        # Positions should have changed (rebalance happened)
        assert len(ls.trades) > 0

    def test_nav_history(self, tmp_path):
        """NAV history tracks correctly across multiple days."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        for date in dates:
            ls.run_daily(
                date=date,
                factor_panels=factor_panels,
                market_df=market_df,
                scheme=PortfolioWeightScheme.EQUAL,
                top_n=10,
            )

        nav_df = ls.nav_history
        assert len(nav_df) == len(dates)
        assert "date" in nav_df.columns
        assert "nav" in nav_df.columns
        assert "daily_return" in nav_df.columns
        assert "cash" in nav_df.columns
        assert "position_value" in nav_df.columns
        assert nav_df["nav"].iloc[0] > 0

    def test_generate_report(self, tmp_path):
        """Report generation returns correct structure."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        for date in dates:
            ls.run_daily(
                date=date,
                factor_panels=factor_panels,
                market_df=market_df,
                scheme=PortfolioWeightScheme.EQUAL,
                top_n=10,
            )

        report = ls.generate_report()
        assert "cumulative_return" in report
        assert "annual_return" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "daily_pnl" in report
        assert isinstance(report["daily_pnl"], list)
        assert "total_trades" in report
        assert "win_rate" in report
        assert "current_positions_summary" in report

    def test_persistence_roundtrip(self, tmp_path):
        """State survives save/load cycle."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        ls.run_daily(
            date=dates[0],
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )
        nav_after_first = len(ls.nav_history)
        positions_after_first = len(ls.positions)
        trades_after_first = len(ls.trades)

        # Create a new LiveSimulator loading from same directory
        ls2 = LiveSimulator(data_dir=data_dir)
        assert len(ls2.nav_history) == nav_after_first
        assert len(ls2.positions) == positions_after_first
        assert len(ls2.trades) == trades_after_first

    def test_mark_to_market(self, tmp_path):
        """Mark to market updates position values correctly."""
        market_df = make_market_df(n_assets=20, n_dates=5)
        factor_panels = make_factor_panels(market_df)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        # Run day 1 to establish positions
        ls.run_daily(
            date=dates[0],
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )

        nav_before = ls.nav_history["nav"].iloc[-1]

        # Day 2 market prices differ, NAV should update
        ls.run_daily(
            date=dates[1],
            factor_panels=factor_panels,
            market_df=market_df,
            scheme=PortfolioWeightScheme.EQUAL,
            top_n=10,
        )

        nav_after = ls.nav_history["nav"].iloc[-1]
        # NAV should have changed due to price movements
        assert nav_after != nav_before

    def test_empty_factor_panels(self, tmp_path):
        """Running with empty factor panels does not crash."""
        market_df = make_market_df(n_assets=5, n_dates=3)
        dates = sorted(market_df["date"].unique())

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        result = ls.run_daily(
            date=dates[0],
            factor_panels={},
            market_df=market_df,
            top_n=5,
        )
        assert result["date"] == dates[0]
        assert "nav" in result

    def test_no_market_data_for_date(self, tmp_path):
        """Running on a date with no market data returns empty summary."""
        market_df = make_market_df(n_assets=5, n_dates=3)
        factor_panels = make_factor_panels(market_df)

        data_dir = tmp_path / "assistant_data"
        ls = LiveSimulator(data_dir=data_dir)

        result = ls.run_daily(
            date="2099-01-01",
            factor_panels=factor_panels,
            market_df=market_df,
            top_n=5,
        )
        assert "warning" in result
