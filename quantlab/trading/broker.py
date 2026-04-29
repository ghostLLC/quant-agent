"""券商交易接口 —— 抽象基类 + 纸交易实现。

设计思路（围绕 Agent 决策）：
- BrokerInterface 定义标准契约，Agent 通过此接口发单
- PaperBroker 提供无损验证环境，真实费率模拟
- 实盘券商通过继承 BrokerInterface 接入（预留）
- 所有订单经 OrderManager 统一管理，支持 Agent 查询和决策

不绑定特定券商，为多券商接入留扩展点。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    order_id: str
    asset: str
    side: OrderSide
    quantity: int
    price: float  # 限价，0 表示市价
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    commission: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reason: str = ""  # Agent 决策理由

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    asset: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Account:
    account_id: str
    total_value: float
    cash: float
    positions: list[Position]
    margin_used: float = 0.0

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "total_value": self.total_value,
            "cash": self.cash,
            "position_value": self.position_value,
            "margin_used": self.margin_used,
            "position_count": len(self.positions),
        }


class BrokerInterface(ABC):
    """券商抽象接口 —— Agent 通过此接口与市场交互。"""

    @abstractmethod
    def submit_order(self, asset: str, side: OrderSide, quantity: int, price: float = 0, reason: str = "") -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        ...

    @abstractmethod
    def get_account(self) -> Account:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order | None:
        ...


class PaperBroker(BrokerInterface):
    """纸交易券商 —— 模拟撮合，真实费率。

    从现有 CostModel 提取费率参数，确保纸交易结果接近实盘。
    """

    def __init__(self, initial_cash: float = 1_000_000, account_id: str = "") -> None:
        from quantlab.trading.cost_model import AShareCostModel
        self.initial_cash = initial_cash
        self.cost_model = AShareCostModel()
        self.account_id = account_id or f"paper_{uuid4().hex[:6]}"
        self.cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._prices: dict[str, float] = {}  # asset → latest price

    def update_prices(self, prices: dict[str, float]) -> None:
        """更新最新价格（由 Agent 在下单前调用）。"""
        self._prices.update(prices)

    def submit_order(self, asset: str, side: OrderSide, quantity: int, price: float = 0, reason: str = "") -> Order:
        order = Order(
            order_id=f"ord_{uuid4().hex[:8]}",
            asset=asset,
            side=side,
            quantity=quantity,
            price=price,
            reason=reason,
        )

        # Simulate fill
        fill_price = price if price > 0 else self._prices.get(asset, 0)
        if fill_price <= 0:
            order.status = OrderStatus.REJECTED
            self._orders[order.order_id] = order
            return order

        trade_value = fill_price * quantity * 100  # A股按手，100股/手

        if side == OrderSide.BUY:
            cost_rate = self.cost_model.buy_cost_rate(trade_value)
            total_cost = trade_value * (1 + cost_rate)
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                self._orders[order.order_id] = order
                return order
            self.cash -= total_cost
            order.commission = trade_value * cost_rate
        else:
            cost_rate = self.cost_model.sell_cost_rate(trade_value)
            self.cash += trade_value * (1 - cost_rate)
            order.commission = trade_value * cost_rate

        order.status = OrderStatus.FILLED
        order.filled_qty = quantity
        order.filled_avg_price = fill_price

        # Update position
        pos = self._positions.get(asset)
        if pos is None:
            self._positions[asset] = Position(asset=asset, quantity=quantity if side == OrderSide.BUY else -quantity,
                                              avg_cost=fill_price, market_value=0, unrealized_pnl=0)
        else:
            if side == OrderSide.BUY:
                total_qty = pos.quantity + quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * quantity) / max(total_qty, 1)
                pos.quantity = total_qty
            else:
                pos.quantity -= quantity
                if pos.quantity == 0:
                    del self._positions[asset]

        self._orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_positions(self) -> list[Position]:
        for asset, pos in self._positions.items():
            price = self._prices.get(asset, pos.avg_cost)
            pos.market_value = pos.quantity * price * 100
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity * 100
        return list(self._positions.values())

    def get_account(self) -> Account:
        positions = self.get_positions()
        total_value = self.cash + sum(p.market_value for p in positions)
        return Account(
            account_id=self.account_id,
            total_value=total_value,
            cash=self.cash,
            positions=positions,
        )

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)


class OrderManager:
    """订单管理器 —— Agent 决策与券商执行之间的协调层。

    职责：
    1. 接收 Agent 的信号（factor → target_weights）
    2. 计算与当前持仓的差异
    3. 生成订单列表
    4. 提交到 BrokerInterface
    """

    def __init__(self, broker: BrokerInterface, max_turnover_pct: float = 0.30) -> None:
        self.broker = broker
        self.max_turnover_pct = max_turnover_pct

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        reason: str = "",
    ) -> list[Order]:
        """根据目标权重生成调仓订单。"""
        self.broker.update_prices(prices) if hasattr(self.broker, 'update_prices') else None
        account = self.broker.get_account()
        total_value = account.total_value
        positions = {p.asset: p for p in self.broker.get_positions()}

        orders = []
        all_assets = set(target_weights.keys()) | set(positions.keys())

        for asset in all_assets:
            target_w = target_weights.get(asset, 0)
            current_pos = positions.get(asset)
            current_qty = current_pos.quantity if current_pos else 0
            price = prices.get(asset, 0)
            if price <= 0:
                continue

            target_qty = int(target_w * total_value / (price * 100))
            diff = target_qty - current_qty

            if diff == 0:
                continue

            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            order = self.broker.submit_order(asset, side, abs(diff), price, reason=reason)
            if order.status != OrderStatus.REJECTED:
                orders.append(order)

        return orders
