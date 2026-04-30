"""组合风控层 —— 仓位管理、回撤熔断、流动性过滤。

集成思路（围绕 Agent 思路）：
- 风控不是硬关卡，而是给因子组合评分降权
- RiskManager 输出 risk_penalty_score，接入 delivery screener
- 回撤熔断触发时通知调度器暂停新因子上线，而非直接清仓

独立于具体券商，与现有 portfolio.py / cost_model.py 互操作。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """风险限额配置。"""
    max_single_factor_weight: float = 0.15   # 单个因子最大组合权重
    max_industry_exposure: float = 0.30      # 单个行业最大暴露
    max_daily_turnover: float = 0.50         # 单日最大换手率
    max_drawdown_limit: float = 0.15         # 回撤熔断阈值
    min_market_cap_percentile: int = 10      # 流动市值最低分位数
    min_daily_volume: float = 1_000_000      # 最低日均成交额（元）


@dataclass
class RiskReport:
    """风控报告。"""
    passed: bool
    risk_score: float  # 0-1, higher = riskier
    breaches: list[str]
    position_limits: dict[str, Any]
    drawdown: dict[str, float]
    liquidity: dict[str, float]
    recommendation: str  # "approve" | "reduce" | "suspend"


class RiskManager:
    """组合风控管理器。

    在因子组合后、信号执行前运行。
    输出 risk_score 供 delivery screener 和调度器决策参考。
    """

    def __init__(self, limits: RiskLimits | None = None, date_col: str = "date", asset_col: str = "asset") -> None:
        self.limits = limits or RiskLimits()
        self.date_col = date_col
        self.asset_col = asset_col
        self._drawdown_history: list[float] = []

    def evaluate(
        self,
        combined_weights: dict[str, float],
        factor_panels: dict[str, pd.Series],
        market_df: pd.DataFrame,
        portfolio_returns: pd.Series | None = None,
    ) -> RiskReport:
        """全面风险评估。

        Args:
            combined_weights: factor_id → weight
            factor_panels: factor_id → factor_values
            market_df: 市场数据
            portfolio_returns: 组合收益序列（用于计算回撤）

        Returns:
            RiskReport with score and recommendation
        """
        breaches: list[str] = []
        risk_score = 0.0

        # 1. Position concentration check
        pos_limits = self._check_position_limits(combined_weights)
        risk_score += pos_limits.get("concentration_penalty", 0.0)

        # 2. Industry exposure check
        industry_check = self._check_industry_exposure(factor_panels, combined_weights, market_df)
        risk_score += industry_check.get("exposure_penalty", 0.0)

        # 3. Liquidity check
        liq_check = self._check_liquidity(market_df)
        risk_score += liq_check.get("liquidity_penalty", 0.0)

        # 4. Drawdown check
        dd_check = self._check_drawdown(portfolio_returns)
        risk_score += dd_check.get("drawdown_penalty", 0.0)

        # Cap risk_score at 1.0
        risk_score = min(risk_score, 1.0)

        # Collect breaches
        if pos_limits.get("breaches"):
            breaches.extend(pos_limits["breaches"])
        if industry_check.get("breaches"):
            breaches.extend(industry_check["breaches"])
        if liq_check.get("breaches"):
            breaches.extend(liq_check["breaches"])
        if dd_check.get("breaches"):
            breaches.extend(dd_check["breaches"])

        # Recommendation
        if risk_score > 0.5:
            recommendation = "suspend"
        elif risk_score > 0.25:
            recommendation = "reduce"
        else:
            recommendation = "approve"

        return RiskReport(
            passed=len(breaches) == 0,
            risk_score=round(risk_score, 4),
            breaches=breaches,
            position_limits=pos_limits,
            drawdown=dd_check,
            liquidity=liq_check,
            recommendation=recommendation,
        )

    def _check_position_limits(self, weights: dict[str, float]) -> dict[str, Any]:
        breaches = []
        penalty = 0.0
        for fid, w in weights.items():
            if w > self.limits.max_single_factor_weight:
                breaches.append(f"因子 {fid} 权重 {w:.2%} 超过上限 {self.limits.max_single_factor_weight:.0%}")
                penalty += 0.15
        return {"breaches": breaches, "concentration_penalty": round(penalty, 4), "max_weight": max(weights.values()) if weights else 0.0}

    def _check_industry_exposure(self, factor_panels: dict, weights: dict, market_df: pd.DataFrame) -> dict[str, Any]:
        if "industry" not in market_df.columns:
            return {"breaches": [], "exposure_penalty": 0.0, "note": "行业数据不可用"}

        breaches = []
        penalty = 0.0
        try:
            # Compute combined factor series
            combined = None
            for fid, fv in factor_panels.items():
                w = weights.get(fid, 0)
                if combined is None:
                    combined = fv.fillna(0) * w
                else:
                    combined += fv.fillna(0) * w

            if combined is not None:
                aligned = market_df[[self.date_col, self.asset_col, "industry"]].copy()
                aligned = aligned.merge(
                    combined.reset_index().rename(columns={combined.name or 0: "factor"}),
                    on=[self.date_col, self.asset_col], how="left"
                )
                industry_exposure = aligned.groupby("industry")["factor"].mean().abs()
                max_exp = float(industry_exposure.max()) if not industry_exposure.empty else 0.0
                if max_exp > self.limits.max_industry_exposure:
                    breaches.append(f"最大行业暴露 {max_exp:.3f} 超过上限 {self.limits.max_industry_exposure:.2f}")
                    penalty += 0.2
                return {"breaches": breaches, "exposure_penalty": round(penalty, 4), "max_industry_exposure": max_exp}
        except Exception as exc:
            logger.warning("行业暴露检查失败: %s", exc)

        return {"breaches": breaches, "exposure_penalty": 0.0}

    def _check_liquidity(self, market_df: pd.DataFrame) -> dict[str, Any]:
        breaches = []
        penalty = 0.0

        # Check market cap
        mc_col = None
        for col in ["circ_mv", "total_mv", "market_cap"]:
            if col in market_df.columns:
                mc_col = col
                break

        if mc_col:
            threshold = np.percentile(market_df[mc_col].dropna(), self.limits.min_market_cap_percentile)
            below_threshold = (market_df[mc_col] < threshold).sum()
            if below_threshold > 0:
                # Informational — not a hard breach
                penalty += 0.05
                breaches.append(f"市值最低 {self.limits.min_market_cap_percentile}% 分位以下有 {below_threshold} 条记录")

        # Check volume
        if "volume" in market_df.columns and "close" in market_df.columns:
            avg_value = (market_df["volume"] * market_df["close"]).mean()
            if avg_value < self.limits.min_daily_volume:
                breaches.append(f"日均成交额 {avg_value/1e6:.1f}M 低于最低 {self.limits.min_daily_volume/1e6:.1f}M")
                penalty += 0.1

        return {"breaches": breaches, "liquidity_penalty": round(penalty, 4)}

    def _check_drawdown(self, portfolio_returns: pd.Series | None) -> dict[str, Any]:
        breaches = []
        penalty = 0.0

        if portfolio_returns is None or len(portfolio_returns) == 0:
            return {"breaches": breaches, "drawdown_penalty": 0.0, "note": "无组合收益数据"}

        cum = (1 + portfolio_returns.fillna(0)).cumprod()
        running_max = cum.cummax()
        drawdown = (cum / (running_max + 1e-10)) - 1
        max_dd = float(abs(drawdown.min()))

        # Track history
        self._drawdown_history.append(max_dd)
        self._drawdown_history = self._drawdown_history[-20:]  # keep last 20

        if max_dd > self.limits.max_drawdown_limit:
            breaches.append(f"最大回撤 {max_dd:.2%} 超过熔断阈值 {self.limits.max_drawdown_limit:.0%}")
            penalty += 0.3

            # Escalation: if 3 consecutive periods above limit, increase penalty
            recent = self._drawdown_history[-3:]
            if len(recent) >= 3 and all(d > self.limits.max_drawdown_limit for d in recent):
                penalty += 0.2
                breaches.append("连续3期回撤超标，建议暂停新因子上线")

        return {
            "breaches": breaches,
            "drawdown_penalty": round(penalty, 4),
            "max_drawdown": round(float(max_dd), 4),
            "current_drawdown": round(float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0, 4),
        }
