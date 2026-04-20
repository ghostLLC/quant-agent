from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_price_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def run_long_only_backtest(
    signal_df: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    trading_days_per_year: int,
) -> BacktestResult:
    df = signal_df.copy().reset_index(drop=True)

    cash = initial_capital
    shares = 0.0
    position = 0
    trade_log: list[dict] = []
    equity_records: list[dict] = []

    for _, row in df.iterrows():
        trade_price = float(row["close"])
        signal = int(row["signal"])

        if signal == 1 and position == 0:
            shares = cash / (trade_price * (1 + commission_rate))
            commission = shares * trade_price * commission_rate
            cash = cash - shares * trade_price - commission
            position = 1
            trade_log.append(
                {
                    "date": row["date"],
                    "action": "buy",
                    "price": trade_price,
                    "shares": shares,
                    "commission": commission,
                }
            )

        elif signal == -1 and position == 1:
            gross_amount = shares * trade_price
            commission = gross_amount * commission_rate
            cash = cash + gross_amount - commission
            trade_log.append(
                {
                    "date": row["date"],
                    "action": "sell",
                    "price": trade_price,
                    "shares": shares,
                    "commission": commission,
                }
            )
            shares = 0.0
            position = 0

        total_equity = cash + shares * trade_price
        equity_records.append(
            {
                "date": row["date"],
                "close": trade_price,
                "signal": signal,
                "position": position,
                "cash": cash,
                "shares": shares,
                "equity": total_equity,
            }
        )

    equity_curve = pd.DataFrame(equity_records)
    equity_curve["daily_return"] = equity_curve["equity"].pct_change().fillna(0.0)
    equity_curve["cum_return"] = equity_curve["equity"] / initial_capital - 1
    equity_curve["rolling_max"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["rolling_max"] - 1

    total_return = equity_curve["equity"].iloc[-1] / initial_capital - 1
    periods = len(equity_curve)
    annual_return = (1 + total_return) ** (trading_days_per_year / periods) - 1 if periods > 0 else 0.0
    max_drawdown = equity_curve["drawdown"].min() if not equity_curve.empty else 0.0

    metrics = {
        "initial_capital": initial_capital,
        "ending_equity": round(float(equity_curve["equity"].iloc[-1]), 2),
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(annual_return), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "trade_count": int(len(trade_log)),
    }

    trades = pd.DataFrame(trade_log)
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trades=trades)


def format_metrics(metrics: dict) -> str:
    lines = [
        "=== 回测结果 ===",
        f"初始资金: RMB {metrics['initial_capital']:.2f}",
        f"期末权益: RMB {metrics['ending_equity']:.2f}",
        f"累计收益率: {metrics['total_return']:.2%}",
        f"年化收益率: {metrics['annual_return']:.2%}",
        f"最大回撤: {metrics['max_drawdown']:.2%}",
        f"总交易次数: {metrics['trade_count']}",
    ]
    return "\n".join(lines)

