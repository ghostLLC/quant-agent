"""Live paper trading simulator for multi-factor portfolios.

Tracks positions, NAV, and P&L day by day. Persists state so it can
be resumed across scheduler runs -- this is NOT a backtest, it simulates
real-time forward trading.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.trading.broker import (
    OrderManager,
    OrderStatus,
    PaperBroker,
    Position,
)
from quantlab.trading.cost_model import AShareCostModel
from quantlab.trading.portfolio import PortfolioWeightScheme

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_CAPITAL = 1_000_000.0


class LiveSimulator:
    """Daily live paper trading simulator for multi-factor portfolios.

    Tracks positions, NAV, and P&L day by day. Persists state so it can
    be resumed across scheduler runs -- this is NOT a backtest, it simulates
    real-time forward trading.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        cost_model=None,
        portfolio_constructor=None,
        risk_manager=None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("assistant_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.data_dir / "live_portfolio.json"

        self.cost_model = cost_model or AShareCostModel()
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager

        self._initial_capital: float = DEFAULT_INITIAL_CAPITAL
        self._nav_history: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._last_run_date: str = ""

        self._broker: PaperBroker = PaperBroker(
            initial_cash=self._initial_capital,
        )

        self._load_state()

    # ── properties ──────────────────────────────────────────────────

    @property
    def nav_history(self) -> pd.DataFrame:
        """Return NAV history as DataFrame with columns: date, nav, daily_return, cash, position_value."""
        if not self._nav_history:
            return pd.DataFrame(
                columns=["date", "nav", "daily_return", "cash", "position_value"]
            )
        return pd.DataFrame(self._nav_history)

    @property
    def positions(self) -> dict[str, Position]:
        """Current positions keyed by asset."""
        pos_list = self._broker.get_positions()
        return {p.asset: p for p in pos_list}

    @property
    def trades(self) -> list[dict]:
        """List of all executed trades."""
        return list(self._trades)

    # ── daily execution ─────────────────────────────────────────────

    def run_daily(
        self,
        date: str,
        factor_panels: dict[str, pd.Series],
        market_df: pd.DataFrame,
        scheme: PortfolioWeightScheme = PortfolioWeightScheme.IC_WEIGHT,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Execute one day of live simulation.

        1. Combine factor signals into portfolio weights
        2. Compare target weights vs current positions
        3. Execute orders through PaperBroker (with cost model)
        4. Update positions, cash, NAV
        5. Track daily P&L
        6. Persist state to disk
        7. Return daily summary dict
        """
        # 0. Extract close prices for the day
        day_market = market_df[market_df["date"] == date]
        if day_market.empty:
            logger.warning("No market data for date %s, skipping.", date)
            return self._empty_summary(date)

        prices: dict[str, float] = {}
        for _, row in day_market.iterrows():
            asset = str(row["asset"])
            prices[asset] = float(row["close"])

        if not prices:
            logger.warning("No price data for date %s, skipping.", date)
            return self._empty_summary(date)

        # 1. Mark to market before trading
        self._broker.update_prices(prices)

        # 2. Combine factor signals into a single score per asset
        combined_scores = self._combine_factor_scores(factor_panels, date)
        if combined_scores.empty:
            logger.warning("No factor values for date %s, holding positions.", date)
            self._record_nav(date, self._broker.get_account().total_value)
            self._save_state()
            return self._make_summary(date)

        # 3. Rank by combined score, select top_n
        n_select = min(top_n, len(combined_scores.dropna()))
        if n_select == 0:
            self._record_nav(date, self._broker.get_account().total_value)
            self._save_state()
            return self._make_summary(date)

        selected_assets = combined_scores.dropna().nlargest(n_select).index.tolist()

        # 4. Compute target weights (equal weight among selected)
        weight = 1.0 / n_select
        target_weights: dict[str, float] = {asset: weight for asset in selected_assets}

        # 5. Execute rebalance via OrderManager
        order_mgr = OrderManager(self._broker)
        orders = order_mgr.rebalance(
            target_weights, prices, reason=f"live_rebalance_top_{n_select}"
        )

        # 6. Re-mark to market after trades
        self._broker.update_prices(prices)

        # 7. Record NAV
        account = self._broker.get_account()
        self._record_nav(date, account.total_value)

        # 8. Record trades
        for order in orders:
            if order.status == OrderStatus.FILLED:
                self._trades.append({
                    "date": date,
                    "asset": order.asset,
                    "side": order.side.value,
                    "quantity": order.filled_qty,
                    "price": order.filled_avg_price,
                    "cost": round(order.commission, 4),
                    "reason": order.reason,
                })

        # 9. Persist
        self._last_run_date = date
        self._save_state()

        return self._make_summary(date)

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        nav_df = self.nav_history
        if nav_df.empty or len(nav_df) < 2:
            return {
                "cumulative_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "daily_pnl": [],
                "total_trades": len(self._trades),
                "win_rate": 0.0,
                "current_positions_summary": [],
            }

        returns = nav_df["daily_return"].values
        final_nav = nav_df["nav"].iloc[-1]
        cumulative_return = (final_nav / self._initial_capital) - 1.0

        n_days = len(returns)
        ann_factor = 252.0 / max(n_days, 1)
        annual_return = (1.0 + cumulative_return) ** ann_factor - 1.0 if cumulative_return > -1.0 else -1.0

        ann_vol = float(np.std(returns[1:], ddof=1)) * np.sqrt(252.0) if len(returns) > 2 else 0.0
        sharpe = annual_return / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown from NAV curve
        nav_series = nav_df["nav"].values
        running_max = np.maximum.accumulate(nav_series)
        drawdowns = (nav_series - running_max) / running_max
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # Daily P&L
        daily_pnl: list[dict[str, Any]] = []
        for _, row in nav_df.iterrows():
            prev_nav = self._initial_capital
            if len(daily_pnl) > 0:
                prev_nav = daily_pnl[-1]["nav"]
            pnl = row["nav"] - prev_nav
            daily_pnl.append({
                "date": row["date"],
                "pnl": round(pnl, 2),
                "nav": row["nav"],
            })

        # Win rate (positive daily return days)
        if len(returns) > 1:
            win_rate = float((returns[1:] > 0).mean())
        else:
            win_rate = 0.0

        # Current positions summary
        positions_summary: list[dict[str, Any]] = []
        for p in self._broker.get_positions():
            positions_summary.append({
                "asset": p.asset,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            })

        return {
            "cumulative_return": round(cumulative_return, 6),
            "annual_return": round(annual_return, 6),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
            "daily_pnl": daily_pnl,
            "total_trades": len(self._trades),
            "win_rate": round(win_rate, 4),
            "current_positions_summary": positions_summary,
        }

    # ── persistence ─────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted state from data_dir/live_portfolio.json."""
        if not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse live_portfolio.json, starting fresh.")
            return

        self._initial_capital = float(state.get("initial_capital", DEFAULT_INITIAL_CAPITAL))
        self._broker.cash = float(state.get("cash", self._initial_capital))
        self._nav_history = state.get("nav_history", [])
        self._trades = state.get("trades", [])
        self._last_run_date = state.get("last_run_date", "")

        # Restore positions
        for asset, pos_data in state.get("positions", {}).items():
            self._broker._positions[asset] = Position(
                asset=asset,
                quantity=int(pos_data["quantity"]),
                avg_cost=float(pos_data["avg_cost"]),
                market_value=0.0,
                unrealized_pnl=0.0,
            )

        # Update broker initial_cash
        self._broker.initial_cash = self._initial_capital

    def _save_state(self) -> None:
        """Persist current state to data_dir/live_portfolio.json."""
        positions_serialized: dict[str, dict[str, Any]] = {}
        for asset, pos in self._broker._positions.items():
            positions_serialized[asset] = {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
            }

        state: dict[str, Any] = {
            "initial_capital": self._initial_capital,
            "cash": self._broker.cash,
            "positions": positions_serialized,
            "nav_history": self._nav_history,
            "trades": self._trades,
            "last_run_date": self._last_run_date,
        }

        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── internal helpers ────────────────────────────────────────────

    def _combine_factor_scores(
        self,
        factor_panels: dict[str, pd.Series],
        date: str,
    ) -> pd.Series:
        """Combine multiple factor panels into a single score per asset.

        Each factor's values are percentile-ranked cross-sectionally,
        then averaged across factors to produce a combined score.
        """
        if not factor_panels:
            return pd.Series(dtype=float)

        ranked_series: list[pd.Series] = []

        for _factor_id, panel in factor_panels.items():
            if panel.empty:
                continue

            # Extract values for the given date
            if isinstance(panel.index, pd.MultiIndex):
                try:
                    values = panel.xs(date, level="date", drop_level=True)
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                values = panel

            if values.empty:
                continue

            values = values.dropna()
            if len(values) < 2:
                continue

            # Cross-sectional percentile rank
            ranked = values.rank(pct=True)
            ranked.name = _factor_id
            ranked_series.append(ranked)

        if not ranked_series:
            return pd.Series(dtype=float)

        # Equal-weight average of percentile ranks across factors
        combined = pd.concat(ranked_series, axis=1).mean(axis=1)
        return combined.dropna().sort_values(ascending=False)

    def _record_nav(self, date: str, nav: float) -> None:
        """Record a NAV data point."""
        prev_nav = self._initial_capital
        if self._nav_history:
            prev_nav = self._nav_history[-1]["nav"]
        daily_return = (nav / prev_nav - 1.0) if prev_nav > 0 else 0.0

        account = self._broker.get_account()
        self._nav_history.append({
            "date": date,
            "nav": nav,
            "daily_return": round(daily_return, 8),
            "cash": account.cash,
            "position_value": account.position_value,
        })

    def _make_summary(self, date: str) -> dict[str, Any]:
        """Build a daily summary dict."""
        account = self._broker.get_account()
        return {
            "date": date,
            "nav": account.total_value,
            "cash": account.cash,
            "position_value": account.position_value,
            "n_positions": len(self._broker.get_positions()),
            "n_trades": 0,
        }

    def _empty_summary(self, date: str) -> dict[str, Any]:
        """Return summary when no data is available."""
        return {
            "date": date,
            "nav": self._broker.get_account().total_value,
            "cash": 0.0,
            "position_value": 0.0,
            "n_positions": 0,
            "n_trades": 0,
            "warning": "no market data",
        }

    def _mark_to_market(self, date: str, market_df: pd.DataFrame) -> None:
        """Update position market values based on current prices."""
        day_market = market_df[market_df["date"] == date]
        prices: dict[str, float] = {}
        if "close" in day_market.columns and "asset" in day_market.columns:
            for _, row in day_market.iterrows():
                prices[str(row["asset"])] = float(row["close"])
        if prices:
            self._broker.update_prices(prices)


def run_live_simulation(ctx, factor_panels: dict) -> dict:
    """Hook for delivery stage to run live simulation.

    Args:
        ctx: PipelineContext with data_path and cached market data.
        factor_panels: factor_id -> Series (MultiIndex [date, asset] or asset-indexed)

    Returns:
        Daily run summary dict.
    """
    market_df = ctx.load_data()
    if market_df.empty:
        return {"status": "skipped", "reason": "数据为空"}

    if "date" not in market_df.columns:
        return {"status": "skipped", "reason": "数据缺少 date 列"}

    latest_date = str(market_df["date"].max())
    sim = LiveSimulator()
    result = sim.run_daily(
        date=latest_date,
        factor_panels=factor_panels,
        market_df=market_df,
        scheme=PortfolioWeightScheme.IC_WEIGHT,
        top_n=50,
    )
    result["status"] = "success"
    return result
