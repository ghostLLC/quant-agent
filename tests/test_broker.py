"""Tests for broker module: PaperBroker, OrderValidator, TradeLogger, BrokerFactory."""
import pytest
import pandas as pd

from quantlab.trading.broker import (
    BrokerFactory,
    Order,
    OrderSide,
    OrderStatus,
    OrderValidator,
    PaperBroker,
    TradeLogger,
    XtQuantBroker,
)


@pytest.fixture
def paper_broker(tmp_path):
    log_path = str(tmp_path / "trades.csv")
    broker = PaperBroker(initial_cash=1_000_000, trade_log_path=log_path)
    broker.update_prices({"000001": 50.0, "000002": 30.0})
    return broker


@pytest.fixture
def validator():
    return OrderValidator(universe=["000001", "000002"])


class TestPaperBroker:
    def test_paper_broker_buy(self, paper_broker):
        order = paper_broker.submit_order("000001", OrderSide.BUY, 100, price=50.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.filled_avg_price == 50.0
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0].asset == "000001"
        assert positions[0].quantity == 100
        assert paper_broker.cash < 1_000_000

    def test_paper_broker_sell(self, paper_broker):
        paper_broker.submit_order("000001", OrderSide.BUY, 100, price=50.0)
        cash_after_buy = paper_broker.cash
        order = paper_broker.submit_order("000001", OrderSide.SELL, 50, price=50.0)
        assert order.status == OrderStatus.FILLED
        assert order.commission > 0
        assert paper_broker.cash > cash_after_buy
        positions = paper_broker.get_positions()
        assert positions[0].quantity == 50

    def test_paper_broker_insufficient_cash(self, paper_broker):
        paper_broker.cash = 1000  # force insufficient
        order = paper_broker.submit_order("000001", OrderSide.BUY, 100, price=50.0)
        assert order.status == OrderStatus.REJECTED

    def test_paper_broker_insufficient_position(self, tmp_path):
        log_path = str(tmp_path / "trades.csv")
        broker = PaperBroker(
            initial_cash=1_000_000, universe=["000001", "000002"], trade_log_path=log_path
        )
        broker.update_prices({"000001": 50.0, "000002": 30.0})
        order = broker.submit_order("999999", OrderSide.SELL, 100, price=50.0)
        assert order.status == OrderStatus.REJECTED

    def test_paper_broker_get_account(self, paper_broker):
        paper_broker.submit_order("000001", OrderSide.BUY, 100, price=50.0)
        account = paper_broker.get_account()
        assert account.cash < 1_000_000
        assert account.position_value > 0
        assert account.total_value > 0
        assert account.total_value == pytest.approx(account.cash + account.position_value)


class TestOrderValidator:
    def test_order_validator_valid(self, validator):
        errors = validator.validate("000001", OrderSide.BUY, 100, 50.0)
        assert len(errors) == 0

    def test_order_validator_negative_qty(self, validator):
        errors = validator.validate("000001", OrderSide.BUY, 0, 50.0)
        assert any("数量" in e for e in errors)

    def test_order_validator_missing_asset(self, validator):
        errors = validator.validate("999999", OrderSide.BUY, 100, 50.0)
        assert any("交易池" in e for e in errors)


class TestTradeLogger:
    def test_trade_logger_write_read(self, tmp_path):
        log_path = tmp_path / "trades.csv"
        logger = TradeLogger(log_path=str(log_path))
        order = Order(
            order_id="test_001",
            asset="000001",
            side=OrderSide.BUY,
            quantity=100,
            price=50.0,
            status=OrderStatus.FILLED,
            filled_qty=100,
            filled_avg_price=50.0,
            commission=15.0,
        )
        logger.log(order)
        assert log_path.exists()
        df = pd.read_csv(log_path, dtype=str)
        assert list(df.columns) == ["timestamp", "asset", "side", "qty", "price", "commission", "reason"]
        assert len(df) == 1
        assert df.iloc[0]["asset"] == "000001"
        assert df.iloc[0]["side"] == "buy"


class TestBrokerFactory:
    def test_broker_factory_paper(self):
        broker = BrokerFactory.create("paper")
        assert isinstance(broker, PaperBroker)

    def test_broker_factory_xtquant(self):
        broker = BrokerFactory.create("xtquant")
        assert isinstance(broker, XtQuantBroker)

    def test_broker_factory_unknown(self):
        with pytest.raises(ValueError):
            BrokerFactory.create("unknown")
