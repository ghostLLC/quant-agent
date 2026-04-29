"""因子交付报告生成器 —— WorldQuant 风格的因子提交标准。

买方关注的维度：
1. 因子定义与表达式（可复现）
2. IC 指标族（RankIC, ICIR, 正 IC 占比, 衰减曲线）
3. 扣费后组合绩效（净 Sharpe, 净收益, 换手率）
4. 容量估算（能容纳多少资金）
5. 正交性（与已知因子/风险因子的相关性）
6. 稳健性（不同市场阶段、样本外、walk-forward）
7. 风险提示（拥挤度、衰减速度、风格暴露）

输出格式：
- JSON: 结构化数据，供程序化接入
- Markdown: 人类可读报告，供买方尽调
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..trading.cost_model import AShareCostModel
from ..trading.portfolio import FactorPortfolioConstructor, PortfolioWeightScheme
from ..trading.simulator import FactorPortfolioSimulator, SimulationResult


@dataclass(slots=True)
class FactorDeliveryReport:
    """因子交付报告。"""

    # 因子身份
    factor_id: str = ""
    factor_name: str = ""
    factor_family: str = ""
    expression: str = ""
    hypothesis: str = ""
    direction: str = ""

    # IC 指标族
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    icir: float = 0.0
    ic_positive_ratio: float = 0.0
    decay_profile: dict[str, float] = field(default_factory=dict)

    # 扣费后组合绩效
    simulation: dict[str, Any] = field(default_factory=dict)

    # 容量
    capacity: dict[str, Any] = field(default_factory=dict)

    # 正交性
    correlation_to_known_factors: dict[str, float] = field(default_factory=dict)
    industry_exposure: dict[str, float] = field(default_factory=dict)
    market_cap_exposure: float = 0.0

    # 稳健性
    stability_score: float = 0.0
    sample_out_performance: float = 0.0
    walk_forward_sharpe: float = 0.0

    # 交易特征
    avg_daily_turnover: float = 0.0
    holding_period_days: float = 0.0
    round_trip_cost: float = 0.0

    # 风险提示
    risk_flags: list[str] = field(default_factory=list)

    # 元数据
    report_date: str = ""
    data_period: str = ""
    universe: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# 因子交付报告: {self.factor_name}",
            "",
            f"**Factor ID**: `{self.factor_id}` | **Family**: {self.factor_family} | **Direction**: {self.direction}",
            f"**Report Date**: {self.report_date} | **Universe**: {self.universe}",
            "",
            "## 1. 因子定义",
            "",
            f"**表达式**: `{self.expression}`",
            f"**假设**: {self.hypothesis}",
            "",
            "## 2. IC 指标族",
            "",
            f"| 指标 | 值 |",
            f"|------|------|",
            f"| Rank IC 均值 | {self.rank_ic_mean:.4f} |",
            f"| Rank IC 标准差 | {self.rank_ic_std:.4f} |",
            f"| ICIR | {self.icir:.3f} |",
            f"| IC 正比例 | {self.ic_positive_ratio:.1%} |",
            "",
        ]

        if self.decay_profile:
            lines.append("**衰减曲线**:")
            lines.append("")
            for horizon, ic_val in self.decay_profile.items():
                bar = "█" * max(1, int(abs(ic_val) * 200))
                lines.append(f"- {horizon}: {ic_val:.4f} {bar}")
            lines.append("")

        # 扣费绩效
        sim = self.simulation
        if sim:
            lines.extend([
                "## 3. 扣费后组合绩效",
                "",
                f"| 指标 | 值 |",
                f"|------|------|",
                f"| 年化收益（毛） | {sim.get('gross_return', 0):.2%} |",
                f"| 年化收益（扣费） | {sim.get('net_return', 0):.2%} |",
                f"| Sharpe | {sim.get('sharpe_ratio', 0):.3f} |",
                f"| 最大回撤 | {sim.get('max_drawdown', 0):.2%} |",
                f"| 日均换手 | {sim.get('avg_daily_turnover', 0):.2%} |",
                f"| 总成本率 | {sim.get('cost_ratio', 0):.2%} |",
                f"| IR | {sim.get('information_ratio', 0):.3f} |",
                f"| Batting Average | {sim.get('batting_average', 0):.1%} |",
                "",
            ])

        # 容量
        cap = self.capacity
        if cap:
            lines.extend([
                "## 4. 容量估算",
                "",
                f"| 指标 | 值 |",
                f"|------|------|",
                f"| 单股日容量 | ¥{cap.get('per_stock_daily_capacity_yuan', 0):,.0f} |",
                f"| 组合日容量 | ¥{cap.get('total_daily_capacity_yuan', 0):,.0f} |",
                f"| 组合月容量 | ¥{cap.get('total_monthly_capacity_yuan', 0):,.0f} |",
                "",
            ])

        # 正交性
        if self.correlation_to_known_factors:
            lines.extend(["## 5. 正交性", "", "| 已知因子 | 相关性 |", "|----------|--------|"])
            for factor_name, corr in self.correlation_to_known_factors.items():
                lines.append(f"| {factor_name} | {corr:.4f} |")
            lines.append("")

        if self.industry_exposure:
            lines.append(f"**行业暴露**: {json.dumps(self.industry_exposure, ensure_ascii=False)}")
            lines.append(f"**市值暴露**: {self.market_cap_exposure:.4f}")
            lines.append("")

        # 稳健性
        lines.extend([
            "## 6. 稳健性",
            "",
            f"| 指标 | 值 |",
            f"|------|------|",
            f"| 稳定性评分 | {self.stability_score:.3f} |",
            f"| 样本外超额 | {self.sample_out_performance:.2%} |",
            f"| Walk-forward Sharpe | {self.walk_forward_sharpe:.3f} |",
            "",
        ])

        # 交易特征
        lines.extend([
            "## 7. 交易特征",
            "",
            f"| 指标 | 值 |",
            f"|------|------|",
            f"| 日均换手率 | {self.avg_daily_turnover:.2%} |",
            f"| 平均持仓天数 | {self.holding_period_days:.1f} |",
            f"| 往返成本 | {self.round_trip_cost:.2%} |",
            "",
        ])

        # 风险提示
        if self.risk_flags:
            lines.extend(["## 8. ⚠️ 风险提示", ""])
            for flag in self.risk_flags:
                lines.append(f"- {flag}")
            lines.append("")

        return "\n".join(lines)


class FactorDeliveryReportGenerator:
    """因子交付报告生成器。"""

    def __init__(
        self,
        cost_model: AShareCostModel | None = None,
        n_long: int = 50,
        weight_scheme: PortfolioWeightScheme = PortfolioWeightScheme.EQUAL,
    ) -> None:
        self.cost_model = cost_model or AShareCostModel()
        self.n_long = n_long
        self.weight_scheme = weight_scheme

    def generate(
        self,
        factor_spec: Any,
        factor_panel: pd.DataFrame,
        market_df: pd.DataFrame,
        evaluation_report: Any | None = None,
        output_dir: str | Path | None = None,
        formats: list[str] | None = None,
    ) -> FactorDeliveryReport:
        """生成因子交付报告。

        Parameters
        ----------
        factor_spec : FactorSpec
            因子规格
        factor_panel : DataFrame
            因子面板，包含 date, asset, factor_value
        market_df : DataFrame
            市场数据，包含 date, asset, close, volume
        evaluation_report : FactorEvaluationReport, optional
            评估报告（如已有）
        output_dir : str or Path, optional
            输出目录

        Returns
        -------
        FactorDeliveryReport
        """
        panel = factor_panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.dropna(subset=["factor_value"]).reset_index(drop=True)

        # 1. IC 指标族
        ic_metrics = self._compute_ic_metrics(panel)

        # 2. 模拟交易
        simulator = FactorPortfolioSimulator(
            cost_model=self.cost_model,
            n_long=self.n_long,
            weight_scheme=self.weight_scheme,
        )
        sim_result = simulator.simulate(
            factor_panel=panel,
            market_df=market_df,
            output_dir=output_dir,
        )

        # 3. 容量估算
        capacity = self.cost_model.estimate_capacity(market_df["volume"].dropna())

        # 4. 正交性
        orthogonality = self._compute_orthogonality(panel)

        # 5. 交易特征
        trading_chars = self._compute_trading_characteristics(panel, ic_metrics)

        # 6. 风险提示
        risk_flags = self._identify_risks(ic_metrics, sim_result, orthogonality)

        # 7. 稳健性（从评估报告提取）
        stability = self._extract_stability(evaluation_report)

        # 构建报告
        report = FactorDeliveryReport(
            factor_id=getattr(factor_spec, "factor_id", "unknown"),
            factor_name=getattr(factor_spec, "name", "unknown"),
            factor_family=getattr(factor_spec, "family", "unknown"),
            expression=getattr(factor_spec, "expression", ""),
            hypothesis=getattr(factor_spec, "hypothesis", ""),
            direction=getattr(factor_spec, "direction", "unknown"),
            **ic_metrics,
            simulation=sim_result.to_dict(),
            capacity=capacity,
            **orthogonality,
            **stability,
            **trading_chars,
            risk_flags=risk_flags,
            report_date=datetime.now().strftime("%Y-%m-%d"),
            data_period=(
                f"{panel['date'].min().strftime('%Y-%m-%d')} ~ "
                f"{panel['date'].max().strftime('%Y-%m-%d')}"
            ) if not panel.empty else "",
            universe=getattr(getattr(factor_spec, "universe", None), "pool", "hs300"),
        )

        # 保存
        if output_dir:
            self._save_report(report, output_dir, formats=formats, factor_panel=factor_panel)

        return report

    def _compute_ic_metrics(self, panel: pd.DataFrame) -> dict[str, Any]:
        """计算 IC 指标族。"""
        if panel.empty or "forward_return" not in panel.columns:
            # 需要计算 forward return
            if "close" in panel.columns:
                panel = panel.copy()
                panel["forward_return"] = panel.groupby("asset")["close"].shift(-1) / panel["close"] - 1.0
                panel = panel.dropna(subset=["forward_return"])
            else:
                return {
                    "rank_ic_mean": 0.0,
                    "rank_ic_std": 0.0,
                    "icir": 0.0,
                    "ic_positive_ratio": 0.0,
                    "decay_profile": {},
                }

        # Rank IC 序列
        rank_ic_series = panel.groupby("date", sort=False).apply(
            lambda g: g["factor_value"].rank(pct=True).corr(
                g["forward_return"].rank(pct=True)
            ) if len(g) >= 5 else np.nan
        ).dropna()

        # IC 衰减
        decay = {}
        for h in [1, 5, 10, 20]:
            shifted = panel.copy()
            shifted["fwd_ret_h"] = shifted.groupby("asset")["close"].shift(-h) / shifted["close"] - 1.0
            shifted = shifted.dropna(subset=["fwd_ret_h", "factor_value"])
            if not shifted.empty:
                ic_h = shifted.groupby("date", sort=False).apply(
                    lambda g: g["factor_value"].rank(pct=True).corr(
                        g["fwd_ret_h"].rank(pct=True)
                    ) if len(g) >= 5 else np.nan
                ).dropna().mean()
                decay[f"{h}d"] = round(float(ic_h), 6) if pd.notna(ic_h) else 0.0

        return {
            "rank_ic_mean": round(float(rank_ic_series.mean()), 6) if not rank_ic_series.empty else 0.0,
            "rank_ic_std": round(float(rank_ic_series.std()), 6) if len(rank_ic_series) > 1 else 0.0,
            "icir": round(float(rank_ic_series.mean() / rank_ic_series.std()), 4) if len(rank_ic_series) > 1 and rank_ic_series.std() > 0 else 0.0,
            "ic_positive_ratio": round(float((rank_ic_series > 0).mean()), 4) if not rank_ic_series.empty else 0.0,
            "decay_profile": decay,
        }

    def _compute_orthogonality(self, panel: pd.DataFrame) -> dict[str, Any]:
        """计算与已知因子的正交性。"""
        exposure = {}

        # 行业暴露
        if "industry" in panel.columns:
            industry_means = panel.groupby("industry")["factor_value"].mean().abs()
            top_industries = industry_means.nlargest(3).to_dict()
            exposure["industry_exposure"] = {k: round(v, 4) for k, v in top_industries.items()}
        else:
            exposure["industry_exposure"] = {}

        # 市值暴露
        market_cap_corr = 0.0
        if "market_cap" in panel.columns:
            valid = panel.dropna(subset=["factor_value", "market_cap"])
            if len(valid) > 10:
                market_cap_corr = round(
                    float(valid["factor_value"].corr(np.log(valid["market_cap"].clip(lower=1.0)))), 4
                )
        exposure["market_cap_exposure"] = market_cap_corr

        # 与已知因子的相关性（基于因子库，目前为占位）
        exposure["correlation_to_known_factors"] = {}

        return exposure

    def _compute_trading_characteristics(self, panel: pd.DataFrame, ic_metrics: dict) -> dict[str, Any]:
        """计算交易特征。"""
        # 换手率（基于因子排名变动）
        turnovers = []
        dates = sorted(panel["date"].unique())
        for i in range(1, len(dates)):
            prev = panel[panel["date"] == dates[i - 1]].nlargest(self.n_long, "factor_value")["asset"]
            curr = panel[panel["date"] == dates[i]].nlargest(self.n_long, "factor_value")["asset"]
            if len(prev) > 0:
                turnover = 1.0 - len(set(prev) & set(curr)) / len(prev)
                turnovers.append(turnover)

        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

        # 平均持仓天数
        holding_days = 1.0 / max(avg_turnover, 0.01) if avg_turnover > 0 else 252.0

        # 往返成本
        round_trip = self.cost_model.round_trip_cost_rate()

        return {
            "avg_daily_turnover": round(avg_turnover, 6),
            "holding_period_days": round(holding_days, 1),
            "round_trip_cost": round(round_trip, 6),
        }

    def _identify_risks(
        self,
        ic_metrics: dict,
        sim_result: SimulationResult,
        orthogonality: dict,
    ) -> list[str]:
        """识别风险提示。"""
        flags: list[str] = []

        if ic_metrics.get("rank_ic_mean", 0) < 0.02:
            flags.append("Rank IC 偏弱（<0.02），信号强度有限")
        if ic_metrics.get("icir", 0) < 0.3:
            flags.append("ICIR 偏低（<0.3），信号稳定性不足")
        if ic_metrics.get("ic_positive_ratio", 0) < 0.55:
            flags.append("IC 正比例低于 55%，方向不稳定")

        # 衰减风险
        decay = ic_metrics.get("decay_profile", {})
        if decay:
            short_ic = decay.get("1d", 0)
            long_ic = decay.get("20d", 0)
            if abs(short_ic) > 0.01 and abs(long_ic) < abs(short_ic) * 0.3:
                flags.append("因子衰减过快（20d IC 不足 1d IC 的 30%），仅适合极短周期")

        # 换手风险
        if sim_result.avg_daily_turnover > 0.5:
            flags.append(f"日均换手率过高（{sim_result.avg_daily_turnover:.1%}），交易成本侵蚀严重")

        # 市值暴露
        mc_exp = orthogonality.get("market_cap_exposure", 0)
        if abs(mc_exp) > 0.3:
            flags.append(f"市值暴露较高（{mc_exp:.2f}），可能实质上在做大小盘轮动")

        # 容量风险
        cap = sim_result.capacity_estimate
        if cap and cap.get("total_daily_capacity_yuan", 0) < 5_000_000:
            flags.append(f"容量偏小（日容量 ¥{cap['total_daily_capacity_yuan']:,.0f}），大资金无法完全使用")

        # 扣费后衰减
        if sim_result.net_return < sim_result.gross_return * 0.5:
            flags.append("扣费后收益衰减超过 50%，因子的可交易性存疑")

        if not flags:
            flags.append("当前未检测到显著风险，但仍需持续监控衰减与拥挤度变化")

        return flags

    def _extract_stability(self, evaluation_report: Any | None) -> dict[str, Any]:
        """从评估报告提取稳健性指标。"""
        if evaluation_report is None:
            return {
                "stability_score": 0.0,
                "sample_out_performance": 0.0,
                "walk_forward_sharpe": 0.0,
            }

        scorecard = getattr(evaluation_report, "scorecard", None)
        return {
            "stability_score": float(getattr(scorecard, "stability_score", 0.0) or 0.0),
            "sample_out_performance": 0.0,
            "walk_forward_sharpe": 0.0,
        }

    def _save_report(
        self,
        report: FactorDeliveryReport,
        output_dir: str | Path,
        formats: list[str] | None = None,
        factor_panel: pd.DataFrame | None = None,
    ) -> None:
        """保存报告。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_dict = report.to_dict()

        if formats is None:
            formats = ["json", "md"]

        for fmt in formats:
            if fmt == "json":
                json_path = out / f"factor_delivery_{report.factor_id}.json"
                json_path.write_text(
                    json.dumps(report_dict, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            elif fmt == "md":
                md_path = out / f"factor_delivery_{report.factor_id}.md"
                md_path.write_text(report.to_markdown(), encoding="utf-8")
            elif fmt == "html":
                html_path = out / f"factor_delivery_{report.factor_id}.html"
                self.generate_html(report_dict, str(html_path))
            elif fmt in ("csv", "parquet"):
                if factor_panel is not None:
                    ext = fmt
                    panel_path = out / f"factor_panel_{report.factor_id}.{ext}"
                    self.export_factor_panel(factor_panel, str(panel_path), format=fmt)

    @staticmethod
    def generate_html(report_dict: dict[str, Any], output_path: str) -> None:
        """Generate a self-contained HTML factor delivery report.

        Parameters
        ----------
        report_dict : dict
            FactorDeliveryReport as a dict (from to_dict()).
        output_path : str
            File path for the output HTML.
        """
        name = report_dict.get("factor_name", "Unknown")
        fid = report_dict.get("factor_id", "unknown")
        family = report_dict.get("factor_family", "")
        direction = report_dict.get("direction", "")
        expression = report_dict.get("expression", "")
        hypothesis = report_dict.get("hypothesis", "")
        report_date = report_dict.get("report_date", "")
        universe = report_dict.get("universe", "")
        data_period = report_dict.get("data_period", "")

        rank_ic = report_dict.get("rank_ic_mean", 0)
        ic_std = report_dict.get("rank_ic_std", 0)
        icir = report_dict.get("icir", 0)
        ic_pos = report_dict.get("ic_positive_ratio", 0)
        decay = report_dict.get("decay_profile", {})

        sim = report_dict.get("simulation", {})
        cap = report_dict.get("capacity", {})
        corr = report_dict.get("correlation_to_known_factors", {})
        ind_exp = report_dict.get("industry_exposure", {})
        mc_exp = report_dict.get("market_cap_exposure", 0)
        stability = report_dict.get("stability_score", 0)
        so_perf = report_dict.get("sample_out_performance", 0)
        wf_sharpe = report_dict.get("walk_forward_sharpe", 0)
        turnover = report_dict.get("avg_daily_turnover", 0)
        holding = report_dict.get("holding_period_days", 0)
        rt_cost = report_dict.get("round_trip_cost", 0)
        risk_flags = report_dict.get("risk_flags", [])

        def esc(v: Any) -> str:
            return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def fmt_pct(v: float) -> str:
            return f"{v * 100:.2f}%"

        def fmt_float(v: float, d: int = 4) -> str:
            return f"{v:.{d}f}"

        # Build decay bars
        decay_rows = ""
        if decay:
            max_ic = max(abs(v) for v in decay.values()) or 0.01
            for horizon, ic_val in decay.items():
                bar_pct = min(abs(ic_val) / max_ic * 100, 100) if max_ic else 0
                color = "#2e7d32" if ic_val >= 0 else "#c62828"
                decay_rows += f"""
                <tr>
                    <td class="decay-label">{esc(horizon)}</td>
                    <td class="decay-value">{fmt_float(ic_val)}</td>
                    <td class="decay-bar-cell">
                        <div class="decay-bar" style="width:{bar_pct:.0f}%;background:{color};"></div>
                    </td>
                </tr>"""

        # Risk flags
        risk_html = ""
        if risk_flags:
            risk_items = "".join(f"<li>{esc(f)}</li>" for f in risk_flags)
            risk_html = f"""
            <h2>Risk Flags</h2>
            <ul class="risk-list">{risk_items}</ul>"""

        # Correlation table
        corr_rows = ""
        if corr:
            for fname, cval in corr.items():
                corr_rows += f"<tr><td>{esc(fname)}</td><td>{fmt_float(cval)}</td></tr>"

        # Industry exposure
        ind_rows = ""
        if ind_exp:
            for iname, ival in ind_exp.items():
                ind_rows += f"<tr><td>{esc(iname)}</td><td>{fmt_float(ival)}</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factor Delivery Report: {esc(name)}</title>
<style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        color: #212121;
        background: #fff;
        line-height: 1.5;
    }}
    h1 {{ font-size: 1.5rem; border-bottom: 2px solid #1565c0; padding-bottom: 8px; color: #1565c0; }}
    h2 {{ font-size: 1.15rem; margin-top: 28px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; color: #333; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 16px; }}
    .meta span {{ margin-right: 24px; }}
    .section {{ margin: 12px 0 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px; }}
    th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .decay-chart {{ margin: 12px 0; }}
    .decay-label {{ width: 60px; }}
    .decay-value {{ width: 80px; font-family: monospace; }}
    .decay-bar-cell {{ width: auto; }}
    .decay-bar {{ height: 16px; border-radius: 2px; min-width: 4px; }}
    .risk-list {{ color: #c62828; font-size: 0.9rem; }}
    .risk-list li {{ margin: 4px 0; }}
    .expression {{ font-family: monospace; background: #f5f5f5; padding: 8px 12px; border-radius: 4px; font-size: 0.9rem; }}
    .footer {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #e0e0e0; color: #999; font-size: 0.8rem; }}
    @media print {{
        body {{ padding: 0; max-width: none; }}
        h1 {{ page-break-before: avoid; }}
        table {{ page-break-inside: avoid; }}
    }}
</style>
</head>
<body>

<h1>Factor Delivery Report: {esc(name)}</h1>
<div class="meta">
    <span><strong>Factor ID:</strong> {esc(fid)}</span>
    <span><strong>Family:</strong> {esc(family)}</span>
    <span><strong>Direction:</strong> {esc(direction)}</span>
</div>
<div class="meta">
    <span><strong>Report Date:</strong> {esc(report_date)}</span>
    <span><strong>Universe:</strong> {esc(universe)}</span>
    <span><strong>Data Period:</strong> {esc(data_period)}</span>
</div>

<div class="section">
    <h2>Factor Definition</h2>
    <p><strong>Expression:</strong></p>
    <div class="expression">{esc(expression)}</div>
    <p><strong>Hypothesis:</strong> {esc(hypothesis)}</p>
</div>

<div class="section">
    <h2>IC Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Rank IC Mean</td><td>{fmt_float(rank_ic)}</td></tr>
        <tr><td>Rank IC Std</td><td>{fmt_float(ic_std)}</td></tr>
        <tr><td>ICIR</td><td>{fmt_float(icir, 3)}</td></tr>
        <tr><td>IC Positive Ratio</td><td>{fmt_pct(ic_pos)}</td></tr>
    </table>
</div>"""

        if decay:
            html += f"""
<div class="section">
    <h2>IC Decay Profile</h2>
    <div class="decay-chart">
        <table>
            <tr><th>Horizon</th><th>IC</th><th>Strength</th></tr>
            {decay_rows}
        </table>
    </div>
</div>"""

        if sim:
            html += f"""
<div class="section">
    <h2>Post-Cost Portfolio Performance</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Gross Return (ann.)</td><td>{fmt_pct(sim.get('gross_return', 0))}</td></tr>
        <tr><td>Net Return (ann.)</td><td>{fmt_pct(sim.get('net_return', 0))}</td></tr>
        <tr><td>Sharpe Ratio</td><td>{fmt_float(sim.get('sharpe_ratio', 0), 3)}</td></tr>
        <tr><td>Max Drawdown</td><td>{fmt_pct(sim.get('max_drawdown', 0))}</td></tr>
        <tr><td>Avg Daily Turnover</td><td>{fmt_pct(sim.get('avg_daily_turnover', 0))}</td></tr>
        <tr><td>Cost Ratio</td><td>{fmt_pct(sim.get('cost_ratio', 0))}</td></tr>
        <tr><td>Information Ratio</td><td>{fmt_float(sim.get('information_ratio', 0), 3)}</td></tr>
        <tr><td>Batting Average</td><td>{fmt_pct(sim.get('batting_average', 0))}</td></tr>
    </table>
</div>"""

        if cap:
            html += f"""
<div class="section">
    <h2>Capacity Estimate</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Per-Stock Daily Capacity</td><td>&yen;{cap.get('per_stock_daily_capacity_yuan', 0):,.0f}</td></tr>
        <tr><td>Total Daily Capacity</td><td>&yen;{cap.get('total_daily_capacity_yuan', 0):,.0f}</td></tr>
        <tr><td>Total Monthly Capacity</td><td>&yen;{cap.get('total_monthly_capacity_yuan', 0):,.0f}</td></tr>
    </table>
</div>"""

        if corr_rows:
            html += f"""
<div class="section">
    <h2>Orthogonality</h2>
    <table>
        <tr><th>Known Factor</th><th>Correlation</th></tr>
        {corr_rows}
    </table>
</div>"""

        if ind_rows:
            html += f"""
<div class="section">
    <h2>Exposures</h2>
    <p><strong>Industry Exposure:</strong></p>
    <table>
        <tr><th>Industry</th><th>Exposure</th></tr>
        {ind_rows}
    </table>
    <p><strong>Market Cap Exposure:</strong> {fmt_float(mc_exp)}</p>
</div>"""

        html += f"""
<div class="section">
    <h2>Robustness</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Stability Score</td><td>{fmt_float(stability, 3)}</td></tr>
        <tr><td>Sample-Out Performance</td><td>{fmt_pct(so_perf)}</td></tr>
        <tr><td>Walk-Forward Sharpe</td><td>{fmt_float(wf_sharpe, 3)}</td></tr>
    </table>
</div>

<div class="section">
    <h2>Trading Characteristics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Avg Daily Turnover</td><td>{fmt_pct(turnover)}</td></tr>
        <tr><td>Avg Holding Period (days)</td><td>{fmt_float(holding, 1)}</td></tr>
        <tr><td>Round-Trip Cost</td><td>{fmt_pct(rt_cost)}</td></tr>
    </table>
</div>
{risk_html}

<div class="footer">
    Generated by QuantLab Factor Factory on {esc(report_date)}.
</div>

</body>
</html>"""

        Path(output_path).write_text(html, encoding="utf-8")

    @staticmethod
    def export_factor_panel(
        factor_panel: pd.DataFrame,
        output_path: str,
        format: str = "csv",
    ) -> None:
        """Export factor panel DataFrame to CSV or Parquet.

        Parameters
        ----------
        factor_panel : DataFrame
            Factor panel with date, asset, factor_value columns.
        output_path : str
            File path for the output file.
        format : str
            Export format: 'csv' or 'parquet'.
        """
        if format == "csv":
            factor_panel.to_csv(output_path, index=False)
        elif format == "parquet":
            factor_panel.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'csv' or 'parquet'.")
