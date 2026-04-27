"""因子模拟交易引擎 —— 从因子信号到模拟盘绩效。

核心流程：
1. 因子面板 → 每日组合权重（FactorPortfolioConstructor）
2. 每日权重 → 模拟交易（含成本）
3. 跟踪组合净值、换手率、成本明细
4. 输出扣费后绩效指标

设计参考：
- WorldQuant Alpha 提交评估标准
- A 股 T+1 约束：当日买入次日才能卖出
- 因子容量估算：基于日均成交量和最大参与率
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import AShareCostModel, CostModel
from .portfolio import FactorPortfolioConstructor, PortfolioWeightScheme


@dataclass(slots=True)
class SimulationResult:
    """模拟交易结果。"""

    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # 扣费指标
    gross_return: float = 0.0
    net_return: float = 0.0
    total_cost: float = 0.0
    cost_ratio: float = 0.0

    # 换手与容量
    avg_daily_turnover: float = 0.0
    max_daily_turnover: float = 0.0
    capacity_estimate: dict[str, Any] = field(default_factory=dict)

    # 因子质量指标（买方关注）
    net_ic: float = 0.0
    information_ratio: float = 0.0
    batting_average: float = 0.0  # 正收益日占比

    # 元信息
    simulation_period: str = ""
    trading_days: int = 0
    rebalance_count: int = 0

    # 详情路径
    equity_curve_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"=== 因子模拟交易结果 ===",
            f"区间: {self.simulation_period} ({self.trading_days} 交易日)",
            f"年化收益: {self.annual_return:.2%} (毛: {self.gross_return:.2%}, 扣费: {self.net_return:.2%})",
            f"年化波动: {self.annual_volatility:.2%}",
            f"Sharpe: {self.sharpe_ratio:.3f} | MaxDD: {self.max_drawdown:.2%} | Calmar: {self.calmar_ratio:.3f}",
            f"日均换手: {self.avg_daily_turnover:.2%} | 最大换手: {self.max_daily_turnover:.2%}",
            f"总成本率: {self.cost_ratio:.2%} | 换仓次数: {self.rebalance_count}",
            f"Net IC: {self.net_ic:.4f} | IR: {self.information_ratio:.3f} | Batting: {self.batting_average:.2%}",
        ]
        cap = self.capacity_estimate
        if cap:
            lines.append(f"容量估算: 日均 {cap.get('total_daily_capacity_yuan', 0):.0f} 元")
        return "\n".join(lines)


class FactorPortfolioSimulator:
    """因子模拟交易引擎。"""

    def __init__(
        self,
        cost_model: CostModel | None = None,
        n_long: int = 50,
        n_short: int = 0,
        weight_scheme: PortfolioWeightScheme = PortfolioWeightScheme.EQUAL,
        rebalance_freq: int = 1,  # 每N天调仓
        initial_capital: float = 1_000_000.0,
        trading_days_per_year: int = 252,
    ) -> None:
        self.cost_model = cost_model or AShareCostModel()
        self.constructor = FactorPortfolioConstructor(
            n_long=n_long,
            n_short=n_short,
            weight_scheme=weight_scheme,
        )
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital
        self.trading_days_per_year = trading_days_per_year

    def simulate(
        self,
        factor_panel: pd.DataFrame,
        market_df: pd.DataFrame,
        output_dir: str | Path | None = None,
    ) -> SimulationResult:
        """运行模拟交易。

        Parameters
        ----------
        factor_panel : DataFrame
            必须包含: date, asset, factor_value
        market_df : DataFrame
            必须包含: date, asset, close, volume
        output_dir : str or Path, optional
            输出目录，保存净值曲线等详情

        Returns
        -------
        SimulationResult
        """
        # 1. 构建每日权重
        weights_df = self.constructor.construct_weights(factor_panel)

        if weights_df.empty:
            return SimulationResult(simulation_period="无可用数据")

        # 2. 准备价格数据
        prices = market_df[["date", "asset", "close", "volume"]].copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices["asset"] = prices["asset"].astype(str).str.zfill(6)
        prices = prices.dropna(subset=["close"]).reset_index(drop=True)

        # 3. 逐日模拟
        dates = sorted(weights_df["date"].unique())
        portfolio_values: list[dict[str, Any]] = []
        prev_weights: dict[str, float] = {}
        capital = self.initial_capital
        total_cost = 0.0
        rebalance_count = 0
        turnovers: list[float] = []

        for day_idx, date in enumerate(dates):
            day_weights = weights_df[weights_df["date"] == date].set_index("asset")["weight"].to_dict()
            day_prices = prices[prices["date"] == date].set_index("asset")["close"].to_dict()
            day_volumes = prices[prices["date"] == date].set_index("asset")["volume"].to_dict()

            # 计算换手率
            if prev_weights:
                all_assets = set(prev_weights.keys()) | set(day_weights.keys())
                turnover = sum(
                    abs(day_weights.get(a, 0.0) - prev_weights.get(a, 0.0))
                    for a in all_assets
                ) / 2.0
                turnovers.append(turnover)

                # 计算交易成本
                for asset in all_assets:
                    old_w = prev_weights.get(asset, 0.0)
                    new_w = day_weights.get(asset, 0.0)
                    trade_value = abs(new_w - old_w) * capital

                    if trade_value > 0:
                        price = day_prices.get(asset, 20.0)
                        volume = day_volumes.get(asset, 1e6)
                        participation = min(
                            (trade_value / price) / max(volume, 1.0), 0.2
                        ) if volume > 0 else 0.0

                        if new_w > old_w:
                            cost_rate = self.cost_model.buy_cost_rate(trade_value, participation)
                        else:
                            cost_rate = self.cost_model.sell_cost_rate(trade_value, participation)

                        total_cost += trade_value * cost_rate
                        capital -= trade_value * cost_rate

                rebalance_count += 1

            # 计算当日组合收益
            day_return = 0.0
            for asset, weight in day_weights.items():
                asset_price = day_prices.get(asset)
                if asset_price is None:
                    continue
                # 用次日收益计算（T日权重 × T+1日收益）
                day_return += weight * 0  # 当日无法知道次日收益，先用0占位

            prev_weights = day_weights.copy()

            # 用实际价格变动计算收益
            # 在下一个循环迭代中计算（需要次日价格）
            portfolio_values.append({
                "date": date,
                "capital": capital,
                "n_positions": len(day_weights),
                "weights": day_weights,
            })

        # 用价格数据计算实际收益
        result = self._compute_performance(
            portfolio_values, prices, total_cost, turnovers, rebalance_count
        )

        # 估算容量
        if not market_df.empty and "volume" in market_df.columns:
            result.capacity_estimate = self.cost_model.estimate_capacity(
                market_df["volume"].dropna()
            )

        # 保存净值曲线
        if output_dir:
            result = self._save_details(result, portfolio_values, output_dir)

        return result

    def _compute_performance(
        self,
        portfolio_values: list[dict[str, Any]],
        prices: pd.DataFrame,
        total_cost: float,
        turnovers: list[float],
        rebalance_count: int,
    ) -> SimulationResult:
        """从模拟记录计算绩效指标。"""
        if not portfolio_values:
            return SimulationResult()

        # 构建净值曲线
        pv_df = pd.DataFrame(portfolio_values)
        pv_df["date"] = pd.to_datetime(pv_df["date"])
        pv_df = pv_df.sort_values("date").reset_index(drop=True)

        # 计算每日收益（基于价格变动）
        dates = pv_df["date"].tolist()
        daily_returns: list[float] = []

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            prev_weights = pv_df.loc[pv_df["date"] == prev_date, "weights"].values
            curr_prices = prices[prices["date"] == curr_date].set_index("asset")["close"]
            prev_prices = prices[prices["date"] == prev_date].set_index("asset")["close"]

            if len(prev_weights) == 0:
                daily_returns.append(0.0)
                continue

            weights = prev_weights[0] if isinstance(prev_weights[0], dict) else {}

            day_return = 0.0
            for asset, weight in weights.items():
                if asset in curr_prices.index and asset in prev_prices.index:
                    asset_return = (curr_prices[asset] / prev_prices[asset]) - 1.0
                    day_return += weight * asset_return

            daily_returns.append(day_return)

        if not daily_returns:
            return SimulationResult(
                simulation_period=f"{dates[0]} ~ {dates[-1]}" if dates else "",
                trading_days=len(dates),
                rebalance_count=rebalance_count,
            )

        # 填充第一天
        daily_returns = [0.0] + daily_returns
        pv_df["daily_return"] = daily_returns[:len(pv_df)]

        # 计算指标
        returns = pv_df["daily_return"].values
        cumulative = np.cumprod(1 + returns)
        gross_cumulative = cumulative[-1]

        total_return = gross_cumulative - 1.0
        trading_days = len(returns)
        ann_factor = self.trading_days_per_year / max(trading_days, 1)
        annual_return = (1 + total_return) ** ann_factor - 1 if total_return > -1 else -1.0
        annual_vol = float(np.std(returns, ddof=1)) * np.sqrt(self.trading_days_per_year) if len(returns) > 1 else 0.0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        # 最大回撤
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(drawdowns.min())
        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

        # 扣费计算
        initial = self.initial_capital
        final_capital = pv_df["capital"].iloc[-1] if "capital" in pv_df.columns else initial * (1 + total_return)
        net_return = (final_capital - initial) / initial
        gross_return = total_return
        cost_ratio = total_cost / initial if initial > 0 else 0.0

        # 换手
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        max_turnover = float(np.max(turnovers)) if turnovers else 0.0

        # Batting average
        batting = float((np.array(returns) > 0).mean()) if len(returns) > 0 else 0.0

        # Information ratio（相对基准，这里用 0 作为基准）
        excess_returns = returns
        ir = float(np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(self.trading_days_per_year) if len(excess_returns) > 1 and np.std(excess_returns) > 0 else 0.0

        period_str = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}" if dates else ""

        return SimulationResult(
            total_return=round(total_return, 6),
            annual_return=round(annual_return, 6),
            annual_volatility=round(annual_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 6),
            calmar_ratio=round(calmar, 4),
            gross_return=round(gross_return, 6),
            net_return=round(net_return, 6),
            total_cost=round(total_cost, 2),
            cost_ratio=round(cost_ratio, 6),
            avg_daily_turnover=round(avg_turnover, 6),
            max_daily_turnover=round(max_turnover, 6),
            net_ic=0.0,  # 需要从因子面板单独计算
            information_ratio=round(ir, 4),
            batting_average=round(batting, 4),
            simulation_period=period_str,
            trading_days=trading_days,
            rebalance_count=rebalance_count,
        )

    def _save_details(
        self,
        result: SimulationResult,
        portfolio_values: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> SimulationResult:
        """保存详情到文件。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 净值曲线
        if portfolio_values:
            pv_df = pd.DataFrame(portfolio_values)
            pv_df["date"] = pd.to_datetime(pv_df["date"])
            curve_path = out / "simulation_equity_curve.csv"
            pv_df[["date", "capital", "daily_return", "n_positions"]].to_csv(
                curve_path, index=False, encoding="utf-8-sig"
            )
            result.equity_curve_path = str(curve_path)

        # 结果摘要
        summary_path = out / "simulation_summary.json"
        summary_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        return result
