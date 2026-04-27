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
