from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import BacktestConfig


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signal_frame: pd.DataFrame
    summary: dict


@dataclass
class ExportArtifacts:
    metrics_path: Path
    equity_path: Path
    trades_path: Path
    signal_path: Path


def _compute_sharpe(returns: pd.Series, trading_days_per_year: int) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(trading_days_per_year) * returns.mean() / std)


def _compute_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    base = df["close"] / df["close"].iloc[0]
    return base - 1


def run_long_only_backtest(signal_df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    df = signal_df.copy().reset_index(drop=True)

    cash = config.initial_capital
    shares = 0.0
    position = 0
    holding_days = 0
    entry_price = None

    trade_log: list[dict] = []
    equity_records: list[dict] = []

    for _, row in df.iterrows():
        trade_price = float(row["close"])
        signal = int(row["signal"])
        executed_action = "hold"
        blocked_reason = ""

        if position == 1:
            holding_days += 1

        stop_triggered = False
        if position == 1 and config.stop_loss_pct is not None and entry_price is not None:
            drawdown_from_entry = trade_price / entry_price - 1
            if drawdown_from_entry <= -config.stop_loss_pct:
                stop_triggered = True

        if signal == 1 and position == 0:
            gross_shares = cash / (trade_price * (1 + config.commission_rate + config.slippage_rate))
            commission = gross_shares * trade_price * config.commission_rate
            slippage = gross_shares * trade_price * config.slippage_rate
            total_cost = gross_shares * trade_price + commission + slippage
            shares = gross_shares
            cash = cash - total_cost
            position = 1
            holding_days = 0
            entry_price = trade_price
            executed_action = "buy"
            trade_log.append(
                {
                    "date": row["date"],
                    "action": "buy",
                    "price": trade_price,
                    "shares": shares,
                    "commission": commission,
                    "slippage": slippage,
                    "reason": "signal",
                }
            )

        elif position == 1 and (stop_triggered or signal == -1):
            if not stop_triggered and holding_days < config.min_holding_days:
                blocked_reason = "min_holding_days"
            else:
                gross_amount = shares * trade_price
                commission = gross_amount * config.commission_rate
                slippage = gross_amount * config.slippage_rate
                cash = cash + gross_amount - commission - slippage
                executed_action = "sell_stop" if stop_triggered else "sell"
                trade_log.append(
                    {
                        "date": row["date"],
                        "action": executed_action,
                        "price": trade_price,
                        "shares": shares,
                        "commission": commission,
                        "slippage": slippage,
                        "reason": "stop_loss" if stop_triggered else "signal",
                    }
                )
                shares = 0.0
                position = 0
                holding_days = 0
                entry_price = None

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
                "holding_days": holding_days,
                "executed_action": executed_action,
                "blocked_reason": blocked_reason,
            }
        )

    equity_curve = pd.DataFrame(equity_records)
    equity_curve["daily_return"] = equity_curve["equity"].pct_change().fillna(0.0)
    equity_curve["cum_return"] = equity_curve["equity"] / config.initial_capital - 1
    equity_curve["rolling_max"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["rolling_max"] - 1
    equity_curve["benchmark_return"] = _compute_buy_and_hold(df)
    equity_curve["excess_return"] = equity_curve["cum_return"] - equity_curve["benchmark_return"]

    total_return = float(equity_curve["equity"].iloc[-1] / config.initial_capital - 1) if not equity_curve.empty else 0.0
    periods = len(equity_curve)
    annual_return = (1 + total_return) ** (config.trading_days_per_year / periods) - 1 if periods > 0 else 0.0
    max_drawdown = float(equity_curve["drawdown"].min()) if not equity_curve.empty else 0.0
    sharpe = _compute_sharpe(equity_curve["daily_return"], config.trading_days_per_year)
    benchmark_return = float(equity_curve["benchmark_return"].iloc[-1]) if not equity_curve.empty else 0.0
    excess_return = float(equity_curve["excess_return"].iloc[-1]) if not equity_curve.empty else 0.0
    win_rate = 0.0

    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        closed = trades[trades["action"].isin(["sell", "sell_stop"])].copy()
        opens = trades[trades["action"] == "buy"].reset_index(drop=True)
        closes = closed.reset_index(drop=True)
        paired_count = min(len(opens), len(closes))
        if paired_count > 0:
            pnl = closes.loc[: paired_count - 1, "price"].to_numpy() - opens.loc[: paired_count - 1, "price"].to_numpy()
            win_rate = float((pnl > 0).mean())

    metrics = {
        "initial_capital": round(float(config.initial_capital), 2),
        "ending_equity": round(float(equity_curve["equity"].iloc[-1]), 2) if not equity_curve.empty else round(float(config.initial_capital), 2),
        "total_return": round(total_return, 4),
        "annual_return": round(float(annual_return), 4),
        "max_drawdown": round(max_drawdown, 4),
        "benchmark_return": round(benchmark_return, 4),
        "excess_return": round(excess_return, 4),
        "sharpe": round(float(sharpe), 4),
        "trade_count": int(len(trade_log)),
        "win_rate": round(win_rate, 4),
    }

    summary = {
        "buy_signals": int((df["signal"] == 1).sum()),
        "sell_signals": int((df["signal"] == -1).sum()),
        "rows": int(len(df)),
    }
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trades=trades, signal_frame=df, summary=summary)


def export_backtest_result(result: BacktestResult, output_dir: str | Path) -> ExportArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = output / "metrics.csv"
    equity_path = output / "equity_curve.csv"
    trades_path = output / "trades.csv"
    signal_path = output / "signals.csv"

    pd.DataFrame([result.metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    result.equity_curve.to_csv(equity_path, index=False, encoding="utf-8-sig")
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    result.signal_frame.to_csv(signal_path, index=False, encoding="utf-8-sig")

    return ExportArtifacts(
        metrics_path=metrics_path,
        equity_path=equity_path,
        trades_path=trades_path,
        signal_path=signal_path,
    )


def format_metrics(metrics: dict) -> str:
    lines = [
        "=== 回测结果 ===",
        f"初始资金: RMB {metrics['initial_capital']:.2f}",
        f"期末权益: RMB {metrics['ending_equity']:.2f}",
        f"累计收益率: {metrics['total_return']:.2%}",
        f"年化收益率: {metrics['annual_return']:.2%}",
        f"基准收益率: {metrics['benchmark_return']:.2%}",
        f"超额收益率: {metrics['excess_return']:.2%}",
        f"Sharpe: {metrics['sharpe']:.2f}",
        f"最大回撤: {metrics['max_drawdown']:.2%}",
        f"胜率: {metrics['win_rate']:.2%}",
        f"总交易次数: {metrics['trade_count']}",
    ]
    return "\n".join(lines)
