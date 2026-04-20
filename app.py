from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from quantlab.config import BacktestConfig, DEFAULT_DATA_PATH
from quantlab.pipeline import (
    get_experiment_detail,
    get_experiment_history,
    refresh_market_data,
    run_grid_experiment,
    run_single_backtest,
    run_train_test_experiment,
    run_walk_forward_experiment,
)
from quantlab.visualization.charts import (
    build_equity_figure,
    build_experiment_compare_figure,
    build_grid_scatter,
    build_history_metric_compare_figure,
    build_history_scatter_figure,
    build_parameter_regime_figure,
    build_price_signal_figure,
    build_research_score_radar_figure,
    build_stability_rank_figure,
    build_walk_forward_compare_figure,
    build_walk_forward_detail_figure,
)



st.set_page_config(page_title="量化研究面板", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(209,75,75,0.08), transparent 30%),
            radial-gradient(circle at 85% 10%, rgba(33,99,235,0.10), transparent 28%),
            linear-gradient(180deg, #f7f8fc 0%, #eef1f7 100%);
    }
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #182848 100%);
        color: white;
        padding: 1.35rem 1.5rem;
        border-radius: 24px;
        box-shadow: 0 24px 80px rgba(17, 24, 39, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.45rem 0 0;
        color: rgba(255,255,255,0.78);
        line-height: 1.6;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
    }
    .subtle-panel {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 0.8rem;
    }
    .detail-card {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(148,163,184,0.16);
        min-height: 120px;
    }
    .detail-kicker {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .detail-value {
        color: #0f172a;
        font-size: 1.4rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def render_metric_cards(metrics: dict, labels: list[tuple[str, str]] | None = None) -> None:
    labels = labels or [
        ("累计收益", "total_return"),
        ("年化收益", "annual_return"),
        ("最大回撤", "max_drawdown"),
        ("Sharpe", "sharpe"),
        ("超额收益", "excess_return"),
    ]
    cols = st.columns(len(labels))
    for col, (title, key) in zip(cols, labels):
        value = metrics.get(key, 0)
        text = f"{value:.2f}" if key == "sharpe" else f"{value:.2%}"
        col.markdown(
            f"<div class='metric-card'><div style='color:#64748b;font-size:0.9rem'>{title}</div><div style='font-size:1.6rem;font-weight:800;color:#0f172a'>{text}</div></div>",
            unsafe_allow_html=True,
        )


def build_config_from_sidebar() -> tuple[BacktestConfig, Path]:
    with st.sidebar:
        st.markdown("## 策略控制台")
        data_path = st.text_input("数据文件路径", value=str(DEFAULT_DATA_PATH))
        short_window = st.slider("短均线", 3, 30, 5)
        long_window = st.slider("长均线", 10, 120, 20)
        initial_capital = st.number_input("初始资金", min_value=10000.0, value=100000.0, step=10000.0)
        commission_rate = st.number_input("手续费率", min_value=0.0, value=0.0003, step=0.0001, format="%.4f")
        slippage_rate = st.number_input("滑点率", min_value=0.0, value=0.0002, step=0.0001, format="%.4f")
        enable_trend_filter = st.toggle("启用趋势过滤", value=False)
        trend_window = st.slider("趋势均线", 20, 180, 60)
        enable_volatility_filter = st.toggle("启用波动率过滤", value=False)
        volatility_window = st.slider("波动率窗口", 5, 60, 20)
        max_volatility = st.number_input("最大允许波动率", min_value=0.0, value=0.03, step=0.005, format="%.3f")
        use_stop_loss = st.toggle("启用止损", value=False)
        stop_loss_pct = st.number_input("止损阈值", min_value=0.0, value=0.08, step=0.01, format="%.2f") if use_stop_loss else None
        min_holding_days = st.slider("最小持有天数", 0, 30, 0)
        train_ratio = st.slider("训练集占比", 0.5, 0.9, 0.7, step=0.05)
        st.markdown("### Walk-forward 设置")
        walk_forward_train_window = st.slider("训练窗口（交易日）", 252, 1000, 504, step=21)
        walk_forward_test_window = st.slider("测试窗口（交易日）", 42, 252, 126, step=21)
        walk_forward_step_size = st.slider("滚动步长（交易日）", 21, 252, 126, step=21)

    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        short_window=short_window,
        long_window=long_window,
        enable_trend_filter=enable_trend_filter,
        trend_window=trend_window,
        enable_volatility_filter=enable_volatility_filter,
        volatility_window=volatility_window,
        max_volatility=max_volatility if enable_volatility_filter else None,
        stop_loss_pct=stop_loss_pct,
        min_holding_days=min_holding_days,
        train_ratio=train_ratio,
        walk_forward_train_window=walk_forward_train_window,
        walk_forward_test_window=walk_forward_test_window,
        walk_forward_step_size=walk_forward_step_size,
    )
    return config, Path(data_path)


def render_downloads(config: BacktestConfig) -> None:
    d1, d2, d3 = st.columns(3)
    metrics_path = config.report_dir / "metrics.csv"
    equity_path = config.report_dir / "equity_curve.csv"
    trades_path = config.report_dir / "trades.csv"
    if metrics_path.exists():
        d1.download_button("下载指标", load_csv_bytes(str(metrics_path)), file_name="metrics.csv", use_container_width=True)
    if equity_path.exists():
        d2.download_button("下载净值曲线", load_csv_bytes(str(equity_path)), file_name="equity_curve.csv", use_container_width=True)
    if trades_path.exists():
        d3.download_button("下载交易记录", load_csv_bytes(str(trades_path)), file_name="trades.csv", use_container_width=True)


def _build_parameter_grid(short_values, long_values, trend_values, stop_values) -> dict[str, list]:
    return {
        "short_window": sorted(short_values),
        "long_window": sorted(long_values),
        "enable_trend_filter": trend_values,
        "stop_loss_pct": stop_values,
    }


def _format_metric_value(metric_key: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, float)):
        if "sharpe" in metric_key.lower():
            return f"{value:.2f}"
        return f"{value:.2%}"
    return str(value)


def _render_detail_summary(detail: dict) -> None:
    metrics = detail.get("metrics", {})
    notes = detail.get("notes", {})
    overview = notes.get("overview", {}) if isinstance(notes, dict) else {}
    stability_summary = notes.get("stability_summary", {}) if isinstance(notes, dict) else {}
    cards = [
        ("实验类型", detail.get("experiment_type", "—")),
        ("主指标", _format_metric_value("primary_metric", metrics.get("annual_return") or metrics.get("test_annual_return") or metrics.get("average_test_annual_return"))),
        ("窗口数量", overview.get("fold_count", "—")),
        ("记录时间", detail.get("timestamp", "—")),
    ]
    if stability_summary:
        cards.extend([
            ("稳定性评分", _format_metric_value("stability_score", stability_summary.get("stability_score"))),
            ("稳定性标签", stability_summary.get("stability_label", "—")),
        ])
    cols = st.columns(len(cards))
    for col, (title, value) in zip(cols, cards):
        col.markdown(
            f"<div class='detail-card'><div class='detail-kicker'>{title}</div><div class='detail-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )



def _render_history_detail(detail: dict | None) -> None:
    if not detail:
        st.info("未找到对应历史详情。")
        return

    _render_detail_summary(detail)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 配置快照")
        st.json(detail.get("config", {}), expanded=False)
    with c2:
        st.markdown("#### 指标快照")
        st.json(detail.get("metrics", {}), expanded=False)

    notes = detail.get("notes", {})
    if isinstance(notes, dict):
        stability_summary = notes.get("stability_summary", {})
        research_summary = notes.get("research_summary", {})
        regime_evolution = notes.get("regime_evolution", [])

        if stability_summary or research_summary:
            st.markdown("#### 稳定性 / 研究评分回看")
            review_col1, review_col2 = st.columns(2)
            with review_col1:
                st.plotly_chart(build_research_score_radar_figure(stability_summary, research_summary), use_container_width=True)
            with review_col2:
                regime_df = pd.DataFrame(regime_evolution) if regime_evolution else pd.DataFrame()
                st.plotly_chart(build_parameter_regime_figure(regime_df), use_container_width=True)

        if isinstance(notes.get("fold_summary"), list) and notes["fold_summary"]:
            fold_summary_df = pd.DataFrame(notes["fold_summary"])
            st.markdown("#### Walk-forward 明细")
            wf_col1, wf_col2 = st.columns(2)
            with wf_col1:
                st.plotly_chart(build_walk_forward_compare_figure(fold_summary_df), use_container_width=True)
            with wf_col2:
                st.plotly_chart(build_walk_forward_detail_figure(fold_summary_df), use_container_width=True)
            st.dataframe(fold_summary_df, use_container_width=True, height=260)

    st.markdown("#### 备注 / 扩展信息")
    st.json(notes, expanded=True)



def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>沪深300 ETF 量化研究面板</h1>
            <p>现在这套工作台已经支持真实数据更新、单次回测、参数实验、训练测试分离、walk-forward 验证和实验历史留痕。前端面板会跟随后端研究能力同步扩展。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config, data_path = build_config_from_sidebar()

    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        st.markdown("<div class='section-title'>数据总览</div>", unsafe_allow_html=True)
        if st.button("更新沪深300ETF数据", use_container_width=True):
            with st.spinner("正在抓取最新数据..."):
                summary = refresh_market_data(data_path)
            st.success(f"数据已更新：{summary['start_date']} ~ {summary['end_date']}，共 {summary['rows']} 行")

    with top_right:
        st.markdown("<div class='section-title'>当前配置</div>", unsafe_allow_html=True)
        st.code(
            json.dumps(
                {
                    "short_window": config.short_window,
                    "long_window": config.long_window,
                    "trend_filter": config.enable_trend_filter,
                    "volatility_filter": config.enable_volatility_filter,
                    "stop_loss_pct": config.stop_loss_pct,
                    "min_holding_days": config.min_holding_days,
                    "train_ratio": config.train_ratio,
                    "walk_forward_train_window": config.walk_forward_train_window,
                    "walk_forward_test_window": config.walk_forward_test_window,
                    "walk_forward_step_size": config.walk_forward_step_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

    tab1, tab2, tab3, tab4 = st.tabs(["单次回测", "参数对比实验", "验证实验", "实验历史"])

    with tab1:
        if st.button("运行单次回测", type="primary", use_container_width=True):
            with st.spinner("回测中..."):
                result, data_summary, _, history_path = run_single_backtest(data_path, config)
            st.success(f"回测完成：数据区间 {data_summary['start_date']} ~ {data_summary['end_date']}，历史记录已保存到 {history_path.name}")
            render_metric_cards(result.metrics)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(build_equity_figure(result.equity_curve), use_container_width=True)
            with col2:
                st.plotly_chart(build_price_signal_figure(result.signal_frame), use_container_width=True)

            st.markdown("<div class='section-title'>交易记录</div>", unsafe_allow_html=True)
            st.dataframe(result.trades, use_container_width=True, height=280)
            render_downloads(config)

    with tab2:
        st.markdown("<div class='section-title'>参数网格设置</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        short_values = g1.multiselect("短均线候选", [3, 5, 8, 10, 12, 15, 20], default=[5, 10, 15])
        long_values = g2.multiselect("长均线候选", [20, 30, 40, 60, 90, 120], default=[20, 30, 60])
        trend_values = g3.multiselect("趋势过滤", [False, True], default=[False, True], format_func=lambda x: "开启" if x else "关闭")
        stop_values = g4.multiselect("止损候选", [None, 0.05, 0.08, 0.1], default=[None, 0.05, 0.08], format_func=lambda x: "无止损" if x is None else f"{x:.0%}")

        if st.button("运行参数对比", use_container_width=True):
            grid = _build_parameter_grid(short_values, long_values, trend_values, stop_values)
            with st.spinner("正在扫描参数组合..."):
                summary_df, best_result, _, history_path = run_grid_experiment(data_path, config, grid)
            st.success(f"参数实验完成，共 {len(summary_df)} 组组合，历史记录已保存到 {history_path.name}")
            render_metric_cards(best_result.metrics)
            st.plotly_chart(build_grid_scatter(summary_df), use_container_width=True)
            st.markdown("<div class='section-title'>最佳组合结果</div>", unsafe_allow_html=True)
            st.plotly_chart(build_equity_figure(best_result.equity_curve), use_container_width=True)
            st.dataframe(summary_df, use_container_width=True, height=360)
            summary_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载参数实验结果", summary_bytes, file_name="grid_search_summary.csv", use_container_width=True)

    with tab3:
        mode = st.radio("验证模式", ["训练 / 测试验证", "Walk-forward 验证"], horizontal=True)
        v1, v2, v3, v4 = st.columns(4)
        val_short_values = v1.multiselect("验证短均线候选", [3, 5, 8, 10, 12, 15, 20], default=[5, 10, 15], key="val_short_values")
        val_long_values = v2.multiselect("验证长均线候选", [20, 30, 40, 60, 90, 120], default=[20, 30, 60], key="val_long_values")
        val_trend_values = v3.multiselect("验证趋势过滤", [False, True], default=[False, True], format_func=lambda x: "开启" if x else "关闭", key="val_trend_values")
        val_stop_values = v4.multiselect("验证止损候选", [None, 0.05, 0.08, 0.1], default=[None, 0.05, 0.08], format_func=lambda x: "无止损" if x is None else f"{x:.0%}", key="val_stop_values")
        grid = _build_parameter_grid(val_short_values, val_long_values, val_trend_values, val_stop_values)

        if mode == "训练 / 测试验证":
            st.markdown("<div class='subtle-panel'>先在训练集上找参数，再把最佳参数放到测试集看样本外表现，适合快速做一次样本外检查。</div>", unsafe_allow_html=True)
            if st.button("运行训练 / 测试验证", use_container_width=True):
                with st.spinner("正在执行训练 / 测试验证..."):
                    validation_result, _, history_path = run_train_test_experiment(data_path, config, grid)
                st.success(f"训练 / 测试验证完成，历史记录已保存到 {history_path.name}")

                overview = validation_result["overview"]
                best_params = validation_result["best_params"]
                st.markdown(
                    f"<div class='subtle-panel'><strong>训练集：</strong>{overview['train_start']} ~ {overview['train_end']}（{overview['train_rows']} 行）<br><strong>测试集：</strong>{overview['test_start']} ~ {overview['test_end']}（{overview['test_rows']} 行）<br><strong>训练集选出的最佳参数：</strong>{json.dumps(best_params, ensure_ascii=False)}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<div class='section-title'>测试集表现</div>", unsafe_allow_html=True)
                render_metric_cards(validation_result["test_result"].metrics)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 样本外最佳参数")
                    st.plotly_chart(build_equity_figure(validation_result["test_result"].equity_curve), use_container_width=True)
                with c2:
                    st.markdown("#### 样本外基线参数")
                    st.plotly_chart(build_equity_figure(validation_result["baseline_test_result"].equity_curve), use_container_width=True)

                compare_df = pd.DataFrame([
                    {"方案": "训练选优参数", **validation_result["test_result"].metrics},
                    {"方案": "当前基线参数", **validation_result["baseline_test_result"].metrics},
                ])
                st.dataframe(compare_df, use_container_width=True, height=220)
                st.markdown("<div class='section-title'>训练集参数扫描结果</div>", unsafe_allow_html=True)
                st.dataframe(validation_result["train_summary"], use_container_width=True, height=260)
        else:
            st.markdown("<div class='subtle-panel'>Walk-forward 会按滚动窗口重复执行“训练选参 → 样本外验证”，更接近真实研究流程，也比单次 train/test 更抗过拟合。</div>", unsafe_allow_html=True)
            if st.button("运行 Walk-forward 验证", use_container_width=True):
                with st.spinner("正在执行 Walk-forward 验证..."):
                    wf_result, _, history_path = run_walk_forward_experiment(data_path, config, grid)
                st.success(f"Walk-forward 验证完成，共 {wf_result['overview']['fold_count']} 个窗口，历史记录已保存到 {history_path.name}")

                avg = wf_result["average_metrics"]
                stability_summary = wf_result.get("stability_summary", {})
                research_summary = wf_result.get("research_summary", {})
                render_metric_cards(
                    {
                        "total_return": avg.get("test_total_return", 0.0),
                        "annual_return": avg.get("test_annual_return", 0.0),
                        "max_drawdown": avg.get("test_max_drawdown", 0.0),
                        "sharpe": avg.get("test_sharpe", 0.0),
                        "excess_return": avg.get("baseline_test_annual_return", 0.0),
                    },
                    labels=[
                        ("平均样本外收益", "total_return"),
                        ("平均样本外年化", "annual_return"),
                        ("平均最大回撤", "max_drawdown"),
                        ("平均 Sharpe", "sharpe"),
                        ("基线样本外年化", "excess_return"),
                    ],
                )

                st.markdown(
                    f"<div class='subtle-panel'><strong>全区间：</strong>{wf_result['overview']['start_date']} ~ {wf_result['overview']['end_date']}<br><strong>训练窗口：</strong>{wf_result['overview']['train_window']} 个交易日<br><strong>测试窗口：</strong>{wf_result['overview']['test_window']} 个交易日<br><strong>滚动步长：</strong>{wf_result['overview']['step_size']} 个交易日</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<div class='section-title'>稳定性 / 研究评分摘要</div>", unsafe_allow_html=True)
                summary_cols = st.columns(6)
                summary_cols[0].metric("稳定性评分", _format_metric_value("stability_score", stability_summary.get("stability_score")))
                summary_cols[1].metric("稳定性标签", stability_summary.get("stability_label", "—"))
                summary_cols[2].metric("研究总分", _format_metric_value("research_score", research_summary.get("research_score")))
                summary_cols[3].metric("研究标签", research_summary.get("research_label", "—"))
                summary_cols[4].metric("正收益窗口占比", _format_metric_value("positive_test_ratio", stability_summary.get("positive_test_ratio")))
                summary_cols[5].metric("跑赢基线占比", _format_metric_value("beat_baseline_ratio", stability_summary.get("beat_baseline_ratio")))

                wf_col1, wf_col2 = st.columns(2)
                with wf_col1:
                    st.plotly_chart(build_walk_forward_compare_figure(wf_result["fold_summary"]), use_container_width=True)
                with wf_col2:
                    st.plotly_chart(build_walk_forward_detail_figure(wf_result["fold_summary"]), use_container_width=True)

                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.plotly_chart(build_research_score_radar_figure(stability_summary, research_summary), use_container_width=True)
                with score_col2:
                    regime_df = wf_result.get("regime_evolution", pd.DataFrame())
                    st.plotly_chart(build_parameter_regime_figure(regime_df), use_container_width=True)

                component_df = pd.DataFrame([
                    {"维度": "稳定性评分", "数值": stability_summary.get("stability_score", 0.0)},
                    {"维度": "研究总分", "数值": research_summary.get("research_score", 0.0)},
                    {"维度": "参数一致性", "数值": stability_summary.get("dominant_parameter_ratio", 0.0)},
                    {"维度": "超额年化", "数值": research_summary.get("excess_annual_return", 0.0)},
                ])
                detail_col1, detail_col2 = st.columns([1.2, 1])
                with detail_col1:
                    st.dataframe(wf_result["fold_summary"], use_container_width=True, height=320)
                with detail_col2:
                    st.dataframe(component_df, use_container_width=True, height=320)


    with tab4:
        st.markdown("<div class='section-title'>实验历史</div>", unsafe_allow_html=True)
        history_df = get_experiment_history(config)
        if history_df.empty:
            st.info("当前还没有历史实验记录。先运行一次回测、参数实验、训练测试验证或 walk-forward 验证，这里就会出现记录。")
        else:
            filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
            experiment_types = sorted(history_df["experiment_type"].dropna().unique().tolist())
            selected_types = filter_col1.multiselect("实验类型筛选", experiment_types, default=experiment_types)
            metric_options = [col for col in [
                "annual_return",
                "test_annual_return",
                "average_test_annual_return",
                "sharpe",
                "test_sharpe",
                "average_test_sharpe",
                "primary_metric",
                "stability_score",
                "research_score",
                "positive_test_ratio",
                "beat_baseline_ratio",
                "dominant_parameter_ratio",
                "excess_annual_return",
            ] if col in history_df.columns]
            selected_metric = filter_col2.selectbox("历史指标对比图", metric_options, index=0 if metric_options else None)
            keyword = filter_col3.text_input("关键词筛选（ID / 备注）")
            stability_labels = sorted(history_df["stability_label"].dropna().unique().tolist()) if "stability_label" in history_df.columns else []
            selected_labels = filter_col4.multiselect("稳定性标签", stability_labels, default=stability_labels)
            research_labels = sorted(history_df["research_label"].dropna().unique().tolist()) if "research_label" in history_df.columns else []
            selected_research_labels = filter_col5.multiselect("研究标签", research_labels, default=research_labels)

            filtered_df = history_df.copy()
            if selected_types:
                filtered_df = filtered_df[filtered_df["experiment_type"].isin(selected_types)]
            if selected_labels and "stability_label" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["stability_label"].isin(selected_labels)]
            if selected_research_labels and "research_label" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["research_label"].isin(selected_research_labels)]

            if keyword:
                keyword_lower = keyword.lower()
                filtered_df = filtered_df[
                    filtered_df["experiment_id"].astype(str).str.lower().str.contains(keyword_lower)
                    | filtered_df["notes"].astype(str).str.lower().str.contains(keyword_lower)
                ]

            if "stability_score" in filtered_df.columns and filtered_df["stability_score"].notna().any():
                score_min = float(filtered_df["stability_score"].dropna().min())
                score_max = float(filtered_df["stability_score"].dropna().max())
                selected_range = st.slider(
                    "稳定性评分区间",
                    min_value=0.0,
                    max_value=1.0,
                    value=(max(0.0, score_min), min(1.0, score_max)),
                    step=0.01,
                )
                filtered_df = filtered_df[
                    filtered_df["stability_score"].isna()
                    | filtered_df["stability_score"].between(selected_range[0], selected_range[1])
                ]

            history_summary_cols = st.columns(6)
            history_summary_cols[0].metric("筛选后实验数", len(filtered_df))
            if "experiment_type" in filtered_df.columns:
                history_summary_cols[1].metric("实验类型数", filtered_df["experiment_type"].nunique())
            if "fold_count" in filtered_df.columns and filtered_df["fold_count"].notna().any():
                history_summary_cols[2].metric("最大窗口数", int(filtered_df["fold_count"].fillna(0).max()))
            if selected_metric and selected_metric in filtered_df.columns and filtered_df[selected_metric].notna().any():
                history_summary_cols[3].metric("当前指标均值", _format_metric_value(selected_metric, float(filtered_df[selected_metric].dropna().mean())))
            if "stability_score" in filtered_df.columns and filtered_df["stability_score"].notna().any():
                history_summary_cols[4].metric("平均稳定性", _format_metric_value("stability_score", float(filtered_df["stability_score"].dropna().mean())))
            if "research_score" in filtered_df.columns and filtered_df["research_score"].notna().any():
                history_summary_cols[5].metric("平均研究总分", _format_metric_value("research_score", float(filtered_df["research_score"].dropna().mean())))

            review_tab, compare_tab, detail_tab = st.tabs(["历史总览", "实验评审", "详情回看"])

            with review_tab:
                sort_columns = [col for col in ["timestamp", "primary_metric", "average_test_annual_return", "stability_score", "research_score", "fold_count"] if col in filtered_df.columns]
                default_sort_idx = sort_columns.index("timestamp") if "timestamp" in sort_columns else 0
                sort_col1, sort_col2 = st.columns([1, 1])
                sort_by = sort_col1.selectbox("排序字段", sort_columns, index=default_sort_idx if sort_columns else None)
                sort_desc = sort_col2.toggle("按降序排序", value=True)
                review_df = filtered_df.sort_values(sort_by, ascending=not sort_desc) if sort_by else filtered_df
                st.dataframe(review_df, use_container_width=True, height=320)

                if selected_metric:
                    st.plotly_chart(build_history_metric_compare_figure(review_df, selected_metric), use_container_width=True)

                scatter_metric_candidates = [col for col in metric_options if col in review_df.columns]
                if len(scatter_metric_candidates) >= 2:
                    scatter_col1, scatter_col2 = st.columns(2)
                    scatter_x = scatter_col1.selectbox("双指标对比 X 轴", scatter_metric_candidates, index=0, key="scatter_x")
                    scatter_y_default = 1 if len(scatter_metric_candidates) > 1 else 0
                    scatter_y = scatter_col2.selectbox("双指标对比 Y 轴", scatter_metric_candidates, index=scatter_y_default, key="scatter_y")
                    st.plotly_chart(build_history_scatter_figure(review_df, scatter_x, scatter_y), use_container_width=True)

            with compare_tab:
                st.markdown("<div class='subtle-panel'>这一层用于快速做实验评审：先看稳定性排行，再看研究总分，再把候选实验拉出来做并排指标比较。</div>", unsafe_allow_html=True)
                rank_col1, rank_col2 = st.columns(2)
                with rank_col1:
                    if "stability_score" in filtered_df.columns and filtered_df["stability_score"].notna().any():
                        st.plotly_chart(build_stability_rank_figure(filtered_df), use_container_width=True)
                    else:
                        st.info("当前筛选结果里还没有可用于排行的稳定性评分，先运行 walk-forward 验证会更有参考价值。")
                with rank_col2:
                    if "research_score" in filtered_df.columns and filtered_df["research_score"].notna().any():
                        research_rank_df = filtered_df.dropna(subset=["research_score"]).sort_values("research_score", ascending=False).head(12)
                        st.dataframe(
                            research_rank_df[[col for col in ["experiment_id", "experiment_type", "research_score", "research_label", "stability_score", "average_test_annual_return"] if col in research_rank_df.columns]],
                            use_container_width=True,
                            height=420,
                        )
                    else:
                        st.info("当前筛选结果里还没有研究总分，运行新版 walk-forward 后这里会显示研究排序。")

                compare_options = filtered_df["experiment_id"].tolist()
                default_compare = compare_options[: min(4, len(compare_options))]
                selected_compare_ids = st.multiselect("选择要并排评审的实验", compare_options, default=default_compare)
                compare_metric_candidates = [col for col in [
                    "primary_metric",
                    "annual_return",
                    "test_annual_return",
                    "average_test_annual_return",
                    "sharpe",
                    "test_sharpe",
                    "average_test_sharpe",
                    "stability_score",
                    "research_score",
                    "positive_test_ratio",
                    "beat_baseline_ratio",
                    "dominant_parameter_ratio",
                    "excess_annual_return",
                ] if col in filtered_df.columns]
                selected_compare_metrics = st.multiselect(
                    "评审指标",
                    compare_metric_candidates,
                    default=compare_metric_candidates[: min(5, len(compare_metric_candidates))],
                )

                if selected_compare_ids and selected_compare_metrics:
                    compare_df = filtered_df[filtered_df["experiment_id"].isin(selected_compare_ids)].copy()
                    st.plotly_chart(build_experiment_compare_figure(compare_df, selected_compare_metrics), use_container_width=True)
                    st.dataframe(
                        compare_df[[col for col in ["experiment_id", "experiment_type", "timestamp", "stability_label", "research_label", *selected_compare_metrics] if col in compare_df.columns]],
                        use_container_width=True,
                        height=260,
                    )
                else:
                    st.info("至少选中 1 个实验和 1 个评审指标，才能生成并排评审图。")


            with detail_tab:
                detail_options = filtered_df["experiment_id"].tolist()
                selected_experiment_id = st.selectbox("点开看详情", detail_options, index=0 if detail_options else None)
                if selected_experiment_id:
                    detail = get_experiment_detail(selected_experiment_id, config)
                    _render_history_detail(detail)

            history_csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载实验历史", history_csv, file_name="experiment_history.csv", use_container_width=True)



if __name__ == "__main__":
    main()

