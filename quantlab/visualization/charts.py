from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_equity_figure(equity_curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=equity_curve["date"], y=equity_curve["cum_return"], name="策略累计收益", line=dict(width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=equity_curve["date"], y=equity_curve["benchmark_return"], name="买入并持有", line=dict(width=2, dash="dot")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=equity_curve["date"], y=equity_curve["drawdown"], name="回撤", fill="tozeroy", opacity=0.2),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.1),
        height=480,
    )
    fig.update_yaxes(title_text="收益率", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", secondary_y=True)
    return fig


def build_price_signal_figure(signal_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signal_frame["date"], y=signal_frame["close"], name="收盘价", line=dict(width=2.2)))
    fig.add_trace(go.Scatter(x=signal_frame["date"], y=signal_frame["ma_short"], name="短均线", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=signal_frame["date"], y=signal_frame["ma_long"], name="长均线", line=dict(width=1.5)))
    if "trend_ma" in signal_frame.columns:
        fig.add_trace(go.Scatter(x=signal_frame["date"], y=signal_frame["trend_ma"], name="趋势均线", line=dict(width=1.2, dash="dot")))

    buys = signal_frame[signal_frame["signal"] == 1]
    sells = signal_frame[signal_frame["signal"] == -1]
    fig.add_trace(go.Scatter(x=buys["date"], y=buys["close"], mode="markers", name="买入信号", marker=dict(size=8, symbol="triangle-up", color="#d14b4b")))
    fig.add_trace(go.Scatter(x=sells["date"], y=sells["close"], mode="markers", name="卖出信号", marker=dict(size=8, symbol="triangle-down", color="#1f8f55")))
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), height=460)
    return fig


def build_grid_scatter(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        summary_df,
        x="max_drawdown",
        y="annual_return",
        color="sharpe",
        size="trade_count",
        hover_data=[col for col in summary_df.columns if col not in {"annual_return", "max_drawdown"}],
        title="参数组合收益 / 回撤 / Sharpe 分布",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=500)
    fig.update_xaxes(title="最大回撤", tickformat=".0%")
    fig.update_yaxes(title="年化收益率", tickformat=".0%")
    return fig


def build_walk_forward_compare_figure(fold_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fold_summary["fold_id"],
            y=fold_summary["test_annual_return"],
            mode="lines+markers",
            name="Walk-forward 样本外年化",
            line=dict(width=2.6, color="#d14b4b"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fold_summary["fold_id"],
            y=fold_summary["baseline_test_annual_return"],
            mode="lines+markers",
            name="基线样本外年化",
            line=dict(width=2.2, dash="dot", color="#1f8f55"),
        )
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), height=420)
    fig.update_xaxes(title="窗口编号", dtick=1)
    fig.update_yaxes(title="年化收益率", tickformat=".0%")
    return fig


def build_history_metric_compare_figure(history_df: pd.DataFrame, metric_key: str) -> go.Figure:
    plot_df = history_df.copy()
    plot_df = plot_df.dropna(subset=[metric_key])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=320)
        return fig

    fig = px.bar(
        plot_df,
        x="timestamp",
        y=metric_key,
        color="experiment_type",
        hover_data=["experiment_id"],
        title=f"历史实验指标对比：{metric_key}",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    if "sharpe" in metric_key:
        fig.update_yaxes(title=metric_key)
    else:
        fig.update_yaxes(title=metric_key, tickformat=".0%")
    fig.update_xaxes(title="实验时间")
    return fig
