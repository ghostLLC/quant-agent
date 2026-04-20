from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.backtest.engine import format_metrics
from quantlab.config import BacktestConfig
from quantlab.pipeline import run_single_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行沪深300 ETF 双均线回测")
    parser.add_argument("--data", required=True, help="CSV 数据文件绝对路径")
    parser.add_argument("--short-window", type=int, default=5, help="短均线窗口")
    parser.add_argument("--long-window", type=int, default=20, help="长均线窗口")
    parser.add_argument("--stop-loss", type=float, default=None, help="止损阈值，例如 0.05")
    parser.add_argument("--trend-filter", action="store_true", help="启用趋势过滤")
    parser.add_argument("--vol-filter", action="store_true", help="启用波动率过滤")
    parser.add_argument("--max-volatility", type=float, default=None, help="最大允许波动率")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    config = BacktestConfig(
        short_window=args.short_window,
        long_window=args.long_window,
        stop_loss_pct=args.stop_loss,
        enable_trend_filter=args.trend_filter,
        enable_volatility_filter=args.vol_filter,
        max_volatility=args.max_volatility,
    )

    result, data_summary, _, history_path = run_single_backtest(data_path, config)
    print(format_metrics(result.metrics))
    print(f"\n数据区间: {data_summary['start_date']} ~ {data_summary['end_date']}")
    print(f"数据行数: {data_summary['rows']}")
    print(f"历史记录: {history_path}")
    print(f"\n结果目录: {config.report_dir}")


if __name__ == "__main__":
    main()

