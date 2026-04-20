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


def build_walk_forward_detail_figure(fold_summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=fold_summary["fold_id"],
            y=fold_summary["test_total_return"],
            name="样本外累计收益",
            marker_color="#d14b4b",
            opacity=0.75,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=fold_summary["fold_id"],
            y=fold_summary["test_max_drawdown"],
            name="样本外最大回撤",
            mode="lines+markers",
            line=dict(color="#2563eb", width=2.2),
        ),
        secondary_y=True,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), height=420)
    fig.update_xaxes(title="窗口编号", dtick=1)
    fig.update_yaxes(title_text="累计收益", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="最大回撤", tickformat=".0%", secondary_y=True)
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


def build_history_scatter_figure(history_df: pd.DataFrame, x_metric: str, y_metric: str) -> go.Figure:
    plot_df = history_df.dropna(subset=[x_metric, y_metric]).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340)
        return fig

    hover_name = plot_df["experiment_id"] if "experiment_id" in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x=x_metric,
        y=y_metric,
        color="experiment_type" if "experiment_type" in plot_df.columns else None,
        hover_name=hover_name,
        hover_data=[col for col in ["timestamp", "fold_count", "primary_metric"] if col in plot_df.columns],
        title=f"历史实验双指标对比：{x_metric} vs {y_metric}",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    if "sharpe" not in x_metric:
        fig.update_xaxes(title=x_metric, tickformat=".0%")
    else:
        fig.update_xaxes(title=x_metric)
    if "sharpe" not in y_metric:
        fig.update_yaxes(title=y_metric, tickformat=".0%")
    else:
        fig.update_yaxes(title=y_metric)
    return fig


def build_stability_rank_figure(history_df: pd.DataFrame) -> go.Figure:
    plot_df = history_df.dropna(subset=["stability_score"]).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340)
        return fig

    plot_df = plot_df.sort_values("stability_score", ascending=False).head(12)
    fig = px.bar(
        plot_df,
        x="experiment_id",
        y="stability_score",
        color="experiment_type" if "experiment_type" in plot_df.columns else None,
        hover_data=[col for col in ["timestamp", "stability_label", "average_test_annual_return"] if col in plot_df.columns],
        title="稳定性评分 Top 实验",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    fig.update_yaxes(title="稳定性评分", range=[0, 1])
    fig.update_xaxes(title="实验 ID")
    return fig




def build_research_score_radar_figure(stability_summary: dict, research_summary: dict) -> go.Figure:
    categories = ["正收益占比", "跑赢基线占比", "参数一致性", "稳定性评分", "研究总分"]
    values = [
        float(stability_summary.get("positive_test_ratio", 0.0)),
        float(stability_summary.get("beat_baseline_ratio", 0.0)),
        float(stability_summary.get("dominant_parameter_ratio", 0.0)),
        float(stability_summary.get("stability_score", 0.0)),
        float(research_summary.get("research_score", 0.0)),
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            name="研究评审画像",
            line=dict(color="#d14b4b", width=2.4),
            fillcolor="rgba(209,75,75,0.22)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
        title="Walk-forward 研究评分画像",
    )
    return fig


def build_parameter_regime_figure(regime_df: pd.DataFrame) -> go.Figure:
    if regime_df.empty or "parameter_regime" not in regime_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340)
        return fig

    plot_df = regime_df.copy()
    plot_df["fold_label"] = plot_df["fold_id"].apply(lambda fold_id: f"窗口 {fold_id}")
    unique_regimes = plot_df["parameter_regime"].dropna().astype(str).unique().tolist()
    regime_map = {label: index + 1 for index, label in enumerate(unique_regimes)}
    plot_df["regime_code"] = plot_df["parameter_regime"].astype(str).map(regime_map)

    hover_columns = [
        col
        for col in ["short_window", "long_window", "enable_trend_filter", "stop_loss_pct", "parameter_regime"]
        if col in plot_df.columns
    ]
    fig = px.line(
        plot_df,
        x="fold_id",
        y="regime_code",
        markers=True,
        custom_data=hover_columns,
        title="参数 Regime 演化",
    )
    fig.update_traces(
        line=dict(width=2.4, color="#2563eb"),
        marker=dict(size=9, color="#d14b4b"),
        hovertemplate="<br>".join(
            [
                "窗口 %{x}",
                *[f"{col}: %{{customdata[{idx}]}}" for idx, col in enumerate(hover_columns)],
                "<extra></extra>",
            ]
        ),
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    fig.update_xaxes(title="窗口编号", dtick=1)
    fig.update_yaxes(
        title="参数 Regime",
        tickmode="array",
        tickvals=list(regime_map.values()),
        ticktext=list(regime_map.keys()),
    )
    return fig


def build_experiment_compare_figure(compare_df: pd.DataFrame, metric_keys: list[str]) -> go.Figure:
    if compare_df.empty or not metric_keys:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340)
        return fig

    plot_df = compare_df[["experiment_id", *metric_keys]].melt(id_vars="experiment_id", var_name="metric", value_name="value")
    plot_df = plot_df.dropna(subset=["value"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340)
        return fig

    fig = px.bar(
        plot_df,
        x="metric",
        y="value",
        color="experiment_id",
        barmode="group",
        title="多实验指标并排评审",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig







