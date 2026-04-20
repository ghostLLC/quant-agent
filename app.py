from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from quantlab.config import BacktestConfig, DEFAULT_DATA_PATH
from quantlab.pipeline import (
    get_experiment_history,
    refresh_market_data,
    run_grid_experiment,
    run_single_backtest,
    run_train_test_experiment,
)
from quantlab.visualization.charts import build_equity_figure, build_grid_scatter, build_price_signal_figure

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


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>沪深300 ETF 量化研究面板</h1>
            <p>现在这套工作台已经支持真实数据更新、单次回测、参数实验、训练测试分离和实验历史留痕。你可以把它当成一个持续进化的量化研究底座。</p>
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
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

    tab1, tab2, tab3, tab4 = st.tabs(["单次回测", "参数对比实验", "训练 / 测试验证", "实验历史"])

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
            grid = {
                "short_window": sorted(short_values),
                "long_window": sorted(long_values),
                "enable_trend_filter": trend_values,
                "stop_loss_pct": stop_values,
            }
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
        st.markdown("<div class='section-title'>训练 / 测试分离</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle-panel'>先在训练集上找参数，再把最佳参数放到测试集看样本外表现，避免只看全样本结果时的过拟合错觉。</div>", unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)
        train_short_values = v1.multiselect("训练短均线候选", [3, 5, 8, 10, 12, 15, 20], default=[5, 10, 15], key="train_short_values")
        train_long_values = v2.multiselect("训练长均线候选", [20, 30, 40, 60, 90, 120], default=[20, 30, 60], key="train_long_values")
        train_trend_values = v3.multiselect("训练趋势过滤", [False, True], default=[False, True], format_func=lambda x: "开启" if x else "关闭", key="train_trend_values")
        train_stop_values = v4.multiselect("训练止损候选", [None, 0.05, 0.08, 0.1], default=[None, 0.05, 0.08], format_func=lambda x: "无止损" if x is None else f"{x:.0%}", key="train_stop_values")

        if st.button("运行训练 / 测试验证", use_container_width=True):
            grid = {
                "short_window": sorted(train_short_values),
                "long_window": sorted(train_long_values),
                "enable_trend_filter": train_trend_values,
                "stop_loss_pct": train_stop_values,
            }
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

    with tab4:
        st.markdown("<div class='section-title'>实验历史</div>", unsafe_allow_html=True)
        history_df = get_experiment_history(config)
        if history_df.empty:
            st.info("当前还没有历史实验记录。先运行一次回测、参数实验或训练测试验证，这里就会出现记录。")
        else:
            st.dataframe(history_df, use_container_width=True, height=360)
            history_csv = history_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载实验历史", history_csv, file_name="experiment_history.csv", use_container_width=True)


if __name__ == "__main__":
    main()
