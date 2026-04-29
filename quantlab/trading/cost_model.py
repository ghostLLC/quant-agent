"""交易成本模型 —— A 股市场的真实交易成本建模。

成本构成：
1. 佣金（双向）：默认万三，最低 5 元
2. 印花税（卖出）：千一
3. 滑点：可配置，默认万二
4. 冲击成本：基于成交量参与率估算

设计参考：
- WorldQuant alpha 评估标准：净 IC（扣费后）比毛 IC 更重要
- A 股 T+1 制度约束
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CostModel:
    """通用交易成本模型。"""

    commission_rate: float = 0.0003   # 佣金费率（双边），万三
    commission_min: float = 5.0       # 最低佣金（元）
    stamp_tax_rate: float = 0.001     # 印花税率（卖出），千一
    slippage_rate: float = 0.0002     # 滑点率（单边），万二
    impact_coefficient: float = 0.1   # 冲击成本系数
    impact_decay: float = 0.5         # 冲击成本衰减（参与率越高衰减越快）

    def buy_cost_rate(self, trade_value: float, participation_rate: float = 0.0) -> float:
        """计算买入综合成本率。"""
        commission = max(self.commission_rate, self.commission_min / max(trade_value, 1.0))
        slippage = self.slippage_rate
        impact = self.impact_coefficient * (participation_rate ** self.impact_decay)
        return commission + slippage + impact

    def sell_cost_rate(self, trade_value: float, participation_rate: float = 0.0) -> float:
        """计算卖出综合成本率。"""
        commission = max(self.commission_rate, self.commission_min / max(trade_value, 1.0))
        slippage = self.slippage_rate
        impact = self.impact_coefficient * (participation_rate ** self.impact_decay)
        stamp_tax = self.stamp_tax_rate
        return commission + slippage + impact + stamp_tax

    def round_trip_cost_rate(self, trade_value: float = 100000.0, participation_rate: float = 0.0) -> float:
        """计算往返交易成本率（买入+卖出）。"""
        return (
            self.buy_cost_rate(trade_value, participation_rate)
            + self.sell_cost_rate(trade_value, participation_rate)
        )


@dataclass(slots=True)
class AShareCostModel(CostModel):
    """A 股专用成本模型，内置市场惯例默认值。"""

    commission_rate: float = 0.0003
    commission_min: float = 5.0
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0002
    impact_coefficient: float = 0.1
    impact_decay: float = 0.5

    def estimate_capacity(
        self,
        daily_volume: pd.Series,
        max_participation: float = 0.05,
        max_stocks: int = 50,
    ) -> dict:
        """估算因子的容量上限。

        容量 = min(单股容量) × 持仓股数
        单股容量 = 日均成交量 × 最大参与率
        """
        avg_volume = daily_volume.mean()
        avg_value = float(avg_volume) * 20.0  # 粗略均价 20 元
        per_stock_capacity = avg_value * max_participation
        total_capacity = per_stock_capacity * max_stocks
        return {
            "per_stock_daily_capacity_yuan": round(per_stock_capacity, 0),
            "total_daily_capacity_yuan": round(total_capacity, 0),
            "total_monthly_capacity_yuan": round(total_capacity * 20, 0),
            "max_participation_rate": max_participation,
            "max_stocks": max_stocks,
            "avg_daily_volume": round(float(avg_volume), 0),
        }


def compute_turnover_cost_impact(
    factor_values: pd.Series,
    market_df: pd.DataFrame,
    cost_model: CostModel | None = None,
    rebalance_days: int = 20,
    date_col: str = "date",
    asset_col: str = "asset",
) -> dict[str, float]:
    """计算因子周转率对交易成本的影响。

    Args:
        factor_values: 因子值 Series
        market_df: 市场数据
        cost_model: 成本模型（None 则使用 AShareCostModel 默认值）
        rebalance_days: 调仓周期（交易日）
        date_col: 日期列名
        asset_col: 资产列名

    Returns:
        dict: turnover, cost_impact_bps, cost_adj_ic_penalty
    """
    if cost_model is None:
        cost_model = AShareCostModel()

    # Compute factor turnover between rebalance dates
    aligned = market_df[[date_col, asset_col]].copy()
    aligned["factor"] = factor_values
    dates = sorted(aligned[date_col].unique())

    turnovers = []
    for i in range(rebalance_days, len(dates), rebalance_days):
        prev_date = dates[i - rebalance_days]
        curr_date = dates[i]
        prev = aligned[aligned[date_col] == prev_date].set_index(asset_col)["factor"]
        curr = aligned[aligned[date_col] == curr_date].set_index(asset_col)["factor"]
        common = prev.index.intersection(curr.index)
        if len(common) < 20:
            continue
        prev_rank = prev[common].rank(pct=True)
        curr_rank = curr[common].rank(pct=True)
        # Fraction of positions that change top/bottom quintile
        prev_top = set(prev_rank.nlargest(int(len(common) * 0.2)).index)
        curr_top = set(curr_rank.nlargest(int(len(common) * 0.2)).index)
        overlap = len(prev_top & curr_top)
        turnover = 1.0 - overlap / max(len(prev_top), 1)
        turnovers.append(turnover)

    if not turnovers:
        return {"turnover": 0.0, "cost_impact_bps": 0.0, "cost_adj_ic_penalty": 0.0}

    avg_turnover = float(np.mean(turnovers))

    # Cost impact = turnover × round_trip_cost × annualization
    round_trip = cost_model.round_trip_cost_rate()
    annual_turnover_rate = avg_turnover * (252 / rebalance_days)
    cost_impact_bps = annual_turnover_rate * round_trip * 10000  # basis points

    # IC penalty: roughly cost_impact / typical return dispersion
    # Typical cross-sectional return dispersion in A-shares ≈ 20% annual
    typical_dispersion = 0.20
    ic_penalty = cost_impact_bps / 10000 / typical_dispersion

    return {
        "turnover": round(float(avg_turnover), 4),
        "annual_turnover_rate": round(float(annual_turnover_rate), 2),
        "cost_impact_bps": round(float(cost_impact_bps), 2),
        "cost_adj_ic_penalty": round(float(ic_penalty), 4),
        "round_trip_cost_rate": round(float(round_trip), 6),
    }
