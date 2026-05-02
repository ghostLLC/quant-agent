"""真实收益评估器 —— 将 IC 预测转化为实际组合收益验证。

核心功能:
  1. 全周期组合回测: long top N 股票，等权/月频调仓，计算扣费前后收益
  2. 成本归因: 佣金、印花税、滑点、冲击成本对收益的侵蚀
  3. IC 与实际收益对比: 检测 IC 是否高估或低估真实投资回报
  4. 容量估算: 基于日均成交量和参与率上限

设计参考:
  - WorldQuant Alpha 提交评估标准: 净IC > 毛IC 的验证逻辑
  - A股 T+1 约束: 当日建仓，次日才能卖出
  - 买方尽调: 因子容量、冲击成本、换手衰减是核心关注点
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.trading.cost_model import AShareCostModel, CostModel

logger = logging.getLogger(__name__)


@dataclass
class RealReturnReport:
    """因子真实收益评估报告。

    汇总全周期组合回测的绩效指标，对比 IC 预测与真实收益的偏差。
    """
    factor_id: str = ""

    # 核心绩效
    net_sharpe: float = 0.0
    gross_sharpe: float = 0.0
    net_return_annual: float = 0.0
    gross_return_annual: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    net_volatility_annual: float = 0.0

    # 成本与换手
    cost_drag_pct: float = 0.0        # 成本侵蚀的年化收益百分比
    avg_turnover: float = 0.0         # 单边平均换手率
    turnover_decay_ratio: float = 0.0 # net_IC / gross_IC，成本吞噬比例

    # 容量
    capacity_estimate: dict[str, Any] = field(default_factory=dict)

    # 归因 (简化版，不依赖 Barra 模型)
    attribution: dict[str, float] = field(default_factory=lambda: {
        "alpha": 0.0,
        "industry_beta": 0.0,
        "market_beta": 0.0,
        "residual": 0.0,
    })

    # 元信息
    n_rebalance_periods: int = 0
    avg_positions_held: float = 0.0
    simulation_period: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "net_sharpe": self.net_sharpe,
            "gross_sharpe": self.gross_sharpe,
            "net_return_annual": self.net_return_annual,
            "gross_return_annual": self.gross_return_annual,
            "max_drawdown": self.max_drawdown,
            "net_volatility_annual": self.net_volatility_annual,
            "cost_drag_pct": self.cost_drag_pct,
            "avg_turnover": self.avg_turnover,
            "turnover_decay_ratio": self.turnover_decay_ratio,
            "capacity_estimate": self.capacity_estimate,
            "attribution": self.attribution,
            "n_rebalance_periods": self.n_rebalance_periods,
            "avg_positions_held": self.avg_positions_held,
            "simulation_period": self.simulation_period,
        }

    def summary(self) -> str:
        lines = [
            f"=== 因子真实收益验证: {self.factor_id} ===",
            f"区间: {self.simulation_period}",
            f"年化收益: 毛 {self.gross_return_annual:.2%} / 净 {self.net_return_annual:.2%}",
            f"夏普: 毛 {self.gross_sharpe:.3f} / 净 {self.net_sharpe:.3f}",
            f"最大回撤: {self.max_drawdown:.2%}",
            f"平均换手: {self.avg_turnover:.2%} | 成本侵蚀: {self.cost_drag_pct:.2f}%/年",
            f"换手衰减比 (net/gross IC): {self.turnover_decay_ratio:.3f}",
        ]
        cap = self.capacity_estimate
        if cap:
            lines.append(
                f"容量估算: 日 {cap.get('total_daily_capacity_yuan', 0):.0f} 元 "
                f"({cap.get('max_stocks', 0)} 只)"
            )
        return "\n".join(lines)


class RealReturnEvaluator:
    """真实收益评估器 —— 从因子面板到组合真实回报。

    运行完整的多头组合回测：
    - 每月调仓，选前 N 只股票，等权
    - 计入 A 股真实交易成本（佣金、印花税、滑点、冲击成本）
    - 输出毛/净收益对比 + 成本归因

    使用方式:
        evaluator = RealReturnEvaluator()
        report = evaluator.evaluate(
            factor_panel=factor_df,
            market_df=market_df,
            n_long=50,
            initial_capital=1e8,
        )
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        trading_days_per_year: int = 252,
    ) -> None:
        self.cost_model = cost_model or AShareCostModel()
        self.trading_days_per_year = trading_days_per_year

    def evaluate(
        self,
        factor_panel: pd.DataFrame,
        market_df: pd.DataFrame,
        cost_model: CostModel | None = None,
        n_long: int = 50,
        initial_capital: float = 1e8,
    ) -> RealReturnReport:
        """对因子面板运行全周期真实收益评估。

        Args:
            factor_panel: 因子值 DataFrame，至少包含 date, asset, factor_value
            market_df: 市场数据，至少包含 date, asset, close, volume
            cost_model: 成本模型，None 使用默认 AShareCostModel
            n_long: 做多股票数量
            initial_capital: 初始资金

        Returns:
            RealReturnReport
        """
        cm = cost_model or self.cost_model

        # ---- 1. 数据准备 ----
        if factor_panel.empty:
            return RealReturnReport(factor_id="unknown", simulation_period="无数据")

        # Normalize columns: support both factor_value column and Series name
        panel = factor_panel.copy()
        if isinstance(panel, pd.Series):
            panel = panel.reset_index()
            if panel.shape[1] >= 3:
                panel.columns = ["date", "asset", "factor_value"]
            elif panel.shape[1] == 2:
                panel.columns = ["asset", "factor_value"]
            else:
                return RealReturnReport(factor_id="unknown", simulation_period="数据格式错误")

        required_cols = {"date", "asset"}
        missing = required_cols - set(panel.columns)
        if missing:
            # Try MultiIndex — already handled by reset_index above
            return RealReturnReport(factor_id="unknown", simulation_period="缺少必要列")

        if "factor_value" not in panel.columns:
            # Assume the third column is the factor value
            val_cols = [c for c in panel.columns if c not in ("date", "asset")]
            if val_cols:
                panel = panel.rename(columns={val_cols[0]: "factor_value"})

        if "factor_value" not in panel.columns:
            return RealReturnReport(factor_id="unknown", simulation_period="缺少因子值列")

        panel["date"] = pd.to_datetime(panel["date"])
        panel["asset"] = panel["asset"].astype(str)
        panel = panel.dropna(subset=["factor_value"])

        # Market data
        mkt = market_df.copy()
        if "date" not in mkt.columns or "asset" not in mkt.columns:
            return RealReturnReport(factor_id="unknown", simulation_period="市场数据缺少必要列")

        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt["asset"] = mkt["asset"].astype(str)

        # Required market columns
        for col in ["close", "volume"]:
            if col not in mkt.columns:
                logger.warning("市场数据缺少 %s 列，使用默认值", col)
                if col == "close":
                    mkt[col] = 10.0
                else:
                    mkt[col] = 1_000_000.0

        # ---- 2. 确定调仓日期 (每月第一个有因子数据的交易日) ----
        factor_dates = sorted(panel["date"].unique())
        if len(factor_dates) < 3:
            return RealReturnReport(
                factor_id="unknown",
                simulation_period=f"{factor_dates[0]} ~ {factor_dates[-1]}" if factor_dates else "无数据",
            )

        # Monthly rebalance: group dates by year-month, pick first date in each
        rebalance_dates: list[pd.Timestamp] = []
        for date in factor_dates:
            if not rebalance_dates or (
                date.year != rebalance_dates[-1].year
                or date.month != rebalance_dates[-1].month
            ):
                rebalance_dates.append(date)

        if len(rebalance_dates) < 2:
            return RealReturnReport(
                factor_id="unknown",
                simulation_period=f"{factor_dates[0]} ~ {factor_dates[-1]}",
            )

        # ---- 3. 逐月模拟 ----
        capital = float(initial_capital)
        total_cost = 0.0
        nav_history: list[float] = [capital]
        turnovers: list[float] = []
        prev_positions: dict[str, float] = {}
        period_dates: list[str] = []
        gross_returns: list[float] = []

        for i in range(len(rebalance_dates) - 1):
            rebal_date = rebalance_dates[i]
            next_rebal_date = rebalance_dates[i + 1]

            # Get factor values at rebalance date
            day_factors = panel[panel["date"] == rebal_date].copy()
            if day_factors.empty:
                continue

            # Select top N stocks
            day_factors = day_factors.dropna(subset=["factor_value"])
            n_select = min(n_long, len(day_factors))
            if n_select == 0:
                continue

            selected = day_factors.nlargest(n_select, "factor_value")
            selected_assets = selected["asset"].tolist()

            # Equal weight
            weight_per = 1.0 / n_select
            target_positions: dict[str, float] = {a: weight_per for a in selected_assets}

            # Compute turnover
            all_assets = set(prev_positions.keys()) | set(target_positions.keys())
            if prev_positions:
                turnover = sum(
                    abs(target_positions.get(a, 0.0) - prev_positions.get(a, 0.0))
                    for a in all_assets
                ) / 2.0
                turnovers.append(turnover)

                # Compute trading costs for transitions
                period_cost = 0.0
                for asset in all_assets:
                    old_w = prev_positions.get(asset, 0.0)
                    new_w = target_positions.get(asset, 0.0)
                    trade_value = abs(new_w - old_w) * capital
                    if trade_value <= 0:
                        continue

                    asset_mkt = mkt[mkt["asset"] == asset]
                    asset_mkt_rebal = asset_mkt[asset_mkt["date"] == rebal_date]
                    price = float(asset_mkt_rebal["close"].iloc[0]) if not asset_mkt_rebal.empty else 10.0
                    volume = float(asset_mkt_rebal["volume"].iloc[0]) if not asset_mkt_rebal.empty else 1e6

                    participation = min(
                        (trade_value / price) / max(volume, 1.0), 0.2
                    ) if volume > 0 else 0.0

                    if new_w > old_w:
                        cost_rate = cm.buy_cost_rate(trade_value, participation)
                    else:
                        cost_rate = cm.sell_cost_rate(trade_value, participation)

                    cost = trade_value * cost_rate
                    period_cost += cost

                total_cost += period_cost
                capital -= period_cost
            else:
                turnovers.append(0.0)

            # Compute period return (from rebal_date to next_rebal_date close)
            period_return = 0.0
            for asset, weight in target_positions.items():
                asset_mkt = mkt[mkt["asset"] == asset]
                if asset_mkt.empty:
                    continue

                start_row = asset_mkt[asset_mkt["date"] == rebal_date]
                end_row = asset_mkt[asset_mkt["date"] == next_rebal_date]

                if start_row.empty or end_row.empty:
                    # Try nearest dates
                    start_candidates = asset_mkt[asset_mkt["date"] >= rebal_date]
                    end_candidates = asset_mkt[asset_mkt["date"] <= next_rebal_date]
                    if start_candidates.empty or end_candidates.empty:
                        continue
                    start_price = float(start_candidates["close"].iloc[0])
                    end_price = float(end_candidates["close"].iloc[-1])
                else:
                    start_price = float(start_row["close"].iloc[0])
                    end_price = float(end_row["close"].iloc[0])

                if start_price > 0:
                    asset_return = (end_price / start_price) - 1.0
                else:
                    asset_return = 0.0
                period_return += weight * asset_return

            gross_returns.append(period_return)

            # Update capital
            capital = capital * (1.0 + period_return)
            nav_history.append(capital)
            period_dates.append(str(rebal_date.date()))

            prev_positions = target_positions.copy()

        # ---- 4. 计算绩效指标 ----
        if len(gross_returns) < 2:
            return RealReturnReport(
                factor_id="unknown",
                simulation_period=f"{rebalance_dates[0].date()} ~ {rebalance_dates[-1].date()}",
            )

        # Gross metrics from period returns
        period_arr = np.array(gross_returns)
        n_periods = len(period_arr)

        # Annualization: number of periods per year ≈ 12 (monthly)
        periods_per_year = min(12.0, float(n_periods) * 12.0 / max(float(len(rebalance_dates)), 1.0))
        if periods_per_year < 1.0:
            periods_per_year = 12.0

        cumulative_gross = np.prod(1.0 + period_arr)
        gross_total_return = float(cumulative_gross - 1.0)
        gross_return_annual = float((1.0 + gross_total_return) ** (periods_per_year / n_periods) - 1.0)

        gross_period_std = float(np.std(period_arr, ddof=1))
        gross_vol_annual = gross_period_std * np.sqrt(periods_per_year)
        gross_sharpe = gross_return_annual / gross_vol_annual if gross_vol_annual > 0 else 0.0

        # Net metrics
        final_nav = float(nav_history[-1])
        net_total_return = (final_nav / initial_capital) - 1.0
        net_return_annual = float((1.0 + net_total_return) ** (periods_per_year / n_periods) - 1.0)

        # Net volatility (approximate: same as gross since costs are small relative)
        net_vol_annual = gross_vol_annual
        net_sharpe = net_return_annual / net_vol_annual if net_vol_annual > 0 else 0.0

        # Cost drag: difference between gross and net annual return
        cost_drag_pct = max(0.0, gross_return_annual - net_return_annual)

        # Max drawdown from NAV
        nav_arr = np.array(nav_history)
        running_max = np.maximum.accumulate(nav_arr)
        drawdowns = (nav_arr - running_max) / running_max
        max_dd = float(drawdowns.min())

        # Turnover
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

        # Capacity estimate
        capacity: dict[str, Any] = {}
        if "volume" in mkt.columns:
            try:
                daily_volumes = mkt.groupby("asset")["volume"].mean()
                avg_daily_vol = float(daily_volumes.mean()) if len(daily_volumes) > 0 else 0.0
                per_stock = avg_daily_vol * 20.0 * 0.05  # avg price ~20, max 5% participation
                total_capacity = per_stock * n_long
                capacity = {
                    "per_stock_daily_capacity_yuan": round(per_stock, 0),
                    "total_daily_capacity_yuan": round(total_capacity, 0),
                    "total_monthly_capacity_yuan": round(total_capacity * 20, 0),
                    "max_participation_rate": 0.05,
                    "max_stocks": n_long,
                    "avg_daily_volume": round(avg_daily_vol, 0),
                }
            except Exception as exc:
                logger.debug("容量估算失败: %s", exc)

        # Turnover decay ratio: net_sharpe / gross_sharpe
        turnover_decay = net_sharpe / gross_sharpe if gross_sharpe > 0 else 0.0

        # Simple attribution
        attribution = self._compute_simple_attribution(
            period_arr, mkt, rebalance_dates, nav_arr, initial_capital, n_long,
        )

        # Simulation period
        sim_period = f"{rebalance_dates[0].date()} ~ {rebalance_dates[-1].date()}"

        return RealReturnReport(
            factor_id="unknown",
            net_sharpe=round(net_sharpe, 4),
            gross_sharpe=round(gross_sharpe, 4),
            net_return_annual=round(net_return_annual, 6),
            gross_return_annual=round(gross_return_annual, 6),
            max_drawdown=round(max_dd, 6),
            net_volatility_annual=round(net_vol_annual, 6),
            cost_drag_pct=round(cost_drag_pct, 6),
            avg_turnover=round(avg_turnover, 6),
            turnover_decay_ratio=round(turnover_decay, 6),
            capacity_estimate=capacity,
            attribution=attribution,
            n_rebalance_periods=n_periods,
            avg_positions_held=float(n_long),
            simulation_period=sim_period,
        )

    def _compute_simple_attribution(
        self,
        period_returns: np.ndarray,
        market_df: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
        nav_arr: np.ndarray,
        initial_capital: float,
        n_long: int,
    ) -> dict[str, float]:
        """Simplified performance attribution (no Barra model dependency).

        Decomposes returns into:
          - market_beta: sensitivity to equal-weight market return
          - alpha: residual after market beta removal
          - industry_beta: placeholder (0.0, requires industry classifications)
          - residual: unexplained variance
        """
        try:
            # Build equal-weight market return series for each period
            market_returns: list[float] = []
            for i in range(len(rebalance_dates) - 1):
                start_date = rebalance_dates[i]
                end_date = rebalance_dates[i + 1]
                start_mkt = market_df[market_df["date"] == start_date]
                end_mkt = market_df[market_df["date"] == end_date]
                if start_mkt.empty or end_mkt.empty:
                    market_returns.append(0.0)
                    continue
                # Equal-weight average return
                common = set(start_mkt["asset"].unique()) & set(end_mkt["asset"].unique())
                if not common:
                    market_returns.append(0.0)
                    continue
                rets = []
                for a in list(common)[:100]:
                    sp = start_mkt[start_mkt["asset"] == a]["close"]
                    ep = end_mkt[end_mkt["asset"] == a]["close"]
                    if not sp.empty and not ep.empty:
                        rets.append(float(ep.iloc[0] / sp.iloc[0] - 1.0))
                market_returns.append(float(np.mean(rets)) if rets else 0.0)

            if len(market_returns) < 3 or len(period_returns) < 3:
                return {"alpha": 0.0, "industry_beta": 0.0, "market_beta": 0.0, "residual": 0.0}

            min_len = min(len(period_returns), len(market_returns))
            pr = period_returns[:min_len]
            mr = np.array(market_returns[:min_len])

            # Market beta via OLS: period_return = alpha + beta * market_return + epsilon
            mr_centered = mr - mr.mean()
            if np.sum(mr_centered ** 2) < 1e-12:
                return {"alpha": 0.0, "industry_beta": 0.0, "market_beta": 0.0, "residual": 0.0}

            beta = float(np.sum(pr * mr_centered) / np.sum(mr_centered ** 2))
            alpha = float(pr.mean() - beta * mr.mean())

            # R-squared for residual
            predicted = alpha + beta * mr
            ss_res = float(np.sum((pr - predicted) ** 2))
            ss_tot = float(np.sum((pr - pr.mean()) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            residual = 1.0 - r_squared

            return {
                "alpha": round(alpha, 6),
                "industry_beta": 0.0,
                "market_beta": round(beta, 4),
                "residual": round(residual, 4),
            }
        except Exception:
            return {"alpha": 0.0, "industry_beta": 0.0, "market_beta": 0.0, "residual": 0.0}


def compare_to_ic(
    report: RealReturnReport,
    ic_stats: dict[str, float],
) -> dict[str, Any]:
    """将 IC 预测与实际组合收益进行对比。

    核心逻辑:
      - IC → Sharpe 的近似关系: Sharpe ≈ IC_annual / sqrt(N) * sqrt(breadth)
        其中 N 为持仓数，breadth 为年化调仓次数
      - 比较 IC 预测的 Sharpe 与组合实际 Sharpe
      - 计算 correlation_bias: IC 高估/低估真实收益的程度

    Args:
        report: RealReturnReport
        ic_stats: IC 统计字典，包含 rank_ic_mean 和 coverage

    Returns:
        dict with ic_predicted_sharpe, actual_sharpe, correlation_bias, ic_inefficiency
    """
    ic_mean = ic_stats.get("rank_ic_mean", ic_stats.get("ic_mean", 0.0))
    coverage = ic_stats.get("coverage", 1.0)

    if ic_mean <= 0:
        return {
            "ic_predicted_sharpe": 0.0,
            "actual_sharpe": report.net_sharpe,
            "correlation_bias": 0.0,
            "ic_inefficiency": 1.0,
            "verdict": "IC无预测能力",
        }

    # Breadth: annual number of independent bets
    # Monthly rebalance × N holdings gives a rough breadth estimate
    n_positions = max(report.avg_positions_held, 1.0)
    rebalance_freq = 12.0  # monthly
    breadth = n_positions * rebalance_freq

    # IC-implied Sharpe: IC * sqrt(breadth)
    ic_predicted_sharpe = ic_mean * np.sqrt(breadth)

    actual_sharpe = report.net_sharpe

    # Correlation bias: how much IC over/under-estimates real returns
    # positive = IC overestimates, negative = IC underestimates
    if ic_predicted_sharpe > 0:
        correlation_bias = (ic_predicted_sharpe - actual_sharpe) / ic_predicted_sharpe
    else:
        correlation_bias = 0.0

    # IC inefficiency: what fraction of IC signal is lost in execution
    if coverage > 0 and ic_mean > 0:
        ic_efficiency = actual_sharpe / ic_predicted_sharpe if ic_predicted_sharpe > 0 else 0.0
        ic_inefficiency = 1.0 - min(ic_efficiency, 1.0)
    else:
        ic_inefficiency = 1.0

    # Verdict
    if actual_sharpe > ic_predicted_sharpe * 1.2:
        verdict = "IC低估真实收益，因子可能存在非线性效应"
    elif actual_sharpe < ic_predicted_sharpe * 0.5:
        verdict = "IC严重高估真实收益，执行成本或市场摩擦过大"
    elif actual_sharpe < ic_predicted_sharpe * 0.8:
        verdict = "IC温和高估真实收益，属于正常摩擦范围"
    else:
        verdict = "IC预测与真实收益基本一致，因子可直接执行"

    return {
        "ic_predicted_sharpe": round(ic_predicted_sharpe, 4),
        "actual_sharpe": round(actual_sharpe, 4),
        "correlation_bias": round(correlation_bias, 4),
        "ic_inefficiency": round(ic_inefficiency, 4),
        "ic_mean": round(ic_mean, 4),
        "breadth_estimate": round(breadth, 0),
        "verdict": verdict,
    }
