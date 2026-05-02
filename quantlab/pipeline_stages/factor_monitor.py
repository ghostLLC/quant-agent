"""因子健康持续监控 —— 超越简单衰减检测的连续因子健康追踪。

检测维度:
  1. IC 漂移: 检测因子预测能力的长期退化趋势
  2. 拥挤度趋势: 检测因子是否被市场参与者持续涌入
  3. 滚动夏普: 检测因子实际收益的风险调整后表现变化
  4. 综合健康报告: 聚合三种检测，给出整体健康状态

设计参考:
  - WorldQuant 因子生命周期管理 (DRAFT → LIVE → RETIRED)
  - 买方因子尽调标准: 拥挤度监控、容量退化
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorHealth(str, Enum):
    """因子综合健康状态。"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FactorHealthReport:
    """因子健康综合报告。"""
    factor_id: str
    ic_drift: dict[str, Any] = field(default_factory=dict)
    crowding_trend: dict[str, Any] = field(default_factory=dict)
    sharpe_status: dict[str, Any] = field(default_factory=dict)
    overall_health: FactorHealth = FactorHealth.HEALTHY
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "ic_drift": self.ic_drift,
            "crowding_trend": self.crowding_trend,
            "sharpe_status": self.sharpe_status,
            "overall_health": self.overall_health.value,
            "recommendations": self.recommendations,
        }


class FactorMonitor:
    """因子性能持续监控器。

    对单个因子执行三项检测（IC 漂移、拥挤度趋势、滚动夏普），
    并汇总为 `FactorHealthReport`。

    使用方式:
        monitor = FactorMonitor()
        report = monitor.run_all(
            factor_id="momentum_01",
            ic_series=ic_daily_series,
            crowding_history=crowding_scores_list,
            nav_series=nav_daily_series,
        )
    """

    def detect_ic_drift(
        self,
        factor_id: str,
        ic_series: pd.Series,
        window: int = 60,
    ) -> dict[str, Any]:
        """检测 IC 漂移 —— 因子的预测能力是否在系统性地退化。

        Args:
            factor_id: 因子 ID
            ic_series: 日频 IC 序列，index 为日期，值域 [-1, 1]
            window: 滚动窗口（交易日），默认 60

        Returns:
            dict with current_ic, peak_ic, decline_pct, drift_flagged, drift_severity
        """
        if ic_series is None or len(ic_series) < window:
            return {
                "current_ic": 0.0,
                "peak_ic": 0.0,
                "decline_pct": 0.0,
                "drift_flagged": False,
                "drift_severity": "normal",
                "error": "insufficient_data",
            }

        try:
            ic = ic_series.dropna()
            if len(ic) < window:
                return {
                    "current_ic": float(ic.mean()) if len(ic) > 0 else 0.0,
                    "peak_ic": float(ic.mean()) if len(ic) > 0 else 0.0,
                    "decline_pct": 0.0,
                    "drift_flagged": False,
                    "drift_severity": "normal",
                    "error": "insufficient_data",
                }

            # Current: 最近 window 天的均值
            recent = ic.iloc[-window:]
            current_ic = float(recent.mean())

            # Peak: 历史最大滚动 window 天均值
            if len(ic) <= window:
                peak_ic = current_ic
            else:
                rolling = ic.rolling(window=window, min_periods=max(5, window // 2)).mean()
                peak_ic = float(rolling.max())

            # Decline percentage
            if peak_ic > 0 and current_ic > 0:
                decline_pct = (peak_ic - current_ic) / peak_ic * 100.0
            elif peak_ic <= 0 and current_ic <= 0:
                decline_pct = 0.0
            else:
                # Sign flip between current and peak means structural break
                decline_pct = 100.0

            # Flag and severity
            drift_flagged = decline_pct > 50.0 and current_ic > 0.0

            if decline_pct > 70.0:
                drift_severity = "critical"
            elif decline_pct > 50.0:
                drift_severity = "warning"
            else:
                drift_severity = "normal"

            result = {
                "current_ic": round(current_ic, 6),
                "peak_ic": round(peak_ic, 6),
                "decline_pct": round(decline_pct, 2),
                "drift_flagged": drift_flagged,
                "drift_severity": drift_severity,
            }

            if drift_flagged:
                logger.info(
                    "IC漂移告警 factor_id=%s decline=%.1f%% severity=%s current=%.4f peak=%.4f",
                    factor_id, decline_pct, drift_severity, current_ic, peak_ic,
                )

            return result
        except Exception as exc:
            logger.warning("IC漂移检测失败 factor_id=%s: %s", factor_id, exc)
            return {
                "current_ic": 0.0,
                "peak_ic": 0.0,
                "decline_pct": 0.0,
                "drift_flagged": False,
                "drift_severity": "normal",
                "error": str(exc)[:200],
            }

    def detect_crowding_trend(
        self,
        factor_id: str,
        crowding_scores_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """检测拥挤度趋势 —— 因子是否被市场参与者持续涌入。

        Args:
            factor_id: 因子 ID
            crowding_scores_history: 按时间排序的拥挤度记录列表。
                每条记录至少包含 "crowding_score" 字段。

        Returns:
            dict with current_crowding, trend_direction, consecutive_increases,
                 crowding_flagged
        """
        if not crowding_scores_history:
            return {
                "current_crowding": 0.0,
                "trend_direction": "stable",
                "consecutive_increases": 0,
                "crowding_flagged": False,
            }

        try:
            scores = []
            for entry in crowding_scores_history:
                if isinstance(entry, dict):
                    s = entry.get("crowding_score", entry.get("score", np.nan))
                elif isinstance(entry, (int, float)):
                    s = float(entry)
                else:
                    s = np.nan
                if not np.isnan(s):
                    scores.append(float(s))

            if not scores:
                return {
                    "current_crowding": 0.0,
                    "trend_direction": "stable",
                    "consecutive_increases": 0,
                    "crowding_flagged": False,
                }

            current = scores[-1]

            # Consecutive increases from the end
            consecutive = 0
            for i in range(len(scores) - 1, 0, -1):
                if scores[i] > scores[i - 1]:
                    consecutive += 1
                else:
                    break

            # Trend direction
            if len(scores) >= 3:
                recent_trend = np.polyfit(
                    range(max(0, len(scores) - 5), len(scores)),
                    scores[-5:],
                    1,
                )[0]
                if recent_trend > 0.005:
                    direction = "increasing"
                elif recent_trend < -0.005:
                    direction = "decreasing"
                else:
                    direction = "stable"
            else:
                direction = "stable"

            flagged = consecutive >= 5

            result = {
                "current_crowding": round(current, 4),
                "trend_direction": direction,
                "consecutive_increases": consecutive,
                "crowding_flagged": flagged,
            }

            if flagged:
                logger.info(
                    "拥挤度趋势告警 factor_id=%s consecutive=%d current=%.3f",
                    factor_id, consecutive, current,
                )

            return result
        except Exception as exc:
            logger.warning("拥挤度趋势检测失败 factor_id=%s: %s", factor_id, exc)
            return {
                "current_crowding": 0.0,
                "trend_direction": "stable",
                "consecutive_increases": 0,
                "crowding_flagged": False,
                "error": str(exc)[:200],
            }

    def monitor_rolling_sharpe(
        self,
        factor_id: str,
        nav_series: pd.Series,
        window: int = 60,
    ) -> dict[str, Any]:
        """监控滚动夏普比率 —— 因子实际收益的风险调整后表现。

        Args:
            factor_id: 因子 ID
            nav_series: 日频净值序列，index 为日期
            window: 滚动窗口（交易日），默认 60

        Returns:
            dict with current_sharpe, peak_sharpe, decline_pct, sharpe_flagged
        """
        if nav_series is None or len(nav_series) < max(window, 10):
            return {
                "current_sharpe": 0.0,
                "peak_sharpe": 0.0,
                "decline_pct": 0.0,
                "sharpe_flagged": False,
                "error": "insufficient_data",
            }

        try:
            nav = nav_series.dropna()
            if len(nav) < max(window, 10):
                return {
                    "current_sharpe": 0.0,
                    "peak_sharpe": 0.0,
                    "decline_pct": 0.0,
                    "sharpe_flagged": False,
                    "error": "insufficient_data",
                }

            # Daily returns
            returns = nav.pct_change().dropna()
            if len(returns) < window:
                return {
                    "current_sharpe": 0.0,
                    "peak_sharpe": 0.0,
                    "decline_pct": 0.0,
                    "sharpe_flagged": False,
                    "error": "insufficient_return_data",
                }

            # Rolling Sharpe: mean / std * sqrt(252)
            rolling_mean = returns.rolling(window=window, min_periods=max(5, window // 2)).mean()
            rolling_std = returns.rolling(window=window, min_periods=max(5, window // 2)).std()
            rolling_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252)

            rolling_sharpe = rolling_sharpe.dropna()
            if rolling_sharpe.empty:
                return {
                    "current_sharpe": 0.0,
                    "peak_sharpe": 0.0,
                    "decline_pct": 0.0,
                    "sharpe_flagged": False,
                }

            current_sharpe = float(rolling_sharpe.iloc[-1])
            peak_sharpe = float(rolling_sharpe.max())

            if peak_sharpe > 0:
                decline_pct = max(0.0, (peak_sharpe - current_sharpe) / peak_sharpe * 100.0)
            else:
                decline_pct = 0.0

            sharpe_flagged = current_sharpe < 0.0

            result = {
                "current_sharpe": round(current_sharpe, 4),
                "peak_sharpe": round(peak_sharpe, 4),
                "decline_pct": round(decline_pct, 2),
                "sharpe_flagged": sharpe_flagged,
            }

            if sharpe_flagged:
                logger.info(
                    "夏普告警 factor_id=%s current=%.3f peak=%.3f",
                    factor_id, current_sharpe, peak_sharpe,
                )

            return result
        except Exception as exc:
            logger.warning("滚动夏普监控失败 factor_id=%s: %s", factor_id, exc)
            return {
                "current_sharpe": 0.0,
                "peak_sharpe": 0.0,
                "decline_pct": 0.0,
                "sharpe_flagged": False,
                "error": str(exc)[:200],
            }

    def run_all(
        self,
        factor_id: str,
        ic_series: pd.Series,
        crowding_history: list[dict[str, Any]],
        nav_series: pd.Series,
    ) -> FactorHealthReport:
        """对单个因子执行三项健康检测，生成综合报告。

        Args:
            factor_id: 因子 ID
            ic_series: 日频 IC 序列
            crowding_history: 拥挤度历史记录列表
            nav_series: 日频净值序列

        Returns:
            FactorHealthReport 包含三项检测结果和综合判定
        """
        # 三项独立检测
        ic_drift = self.detect_ic_drift(factor_id, ic_series)
        crowding = self.detect_crowding_trend(factor_id, crowding_history)
        sharpe = self.monitor_rolling_sharpe(factor_id, nav_series)

        # 综合判定
        critical_count = 0
        warning_count = 0
        recommendations: list[str] = []

        # IC drift checks
        if ic_drift.get("drift_severity") == "critical":
            critical_count += 1
            recommendations.append(
                f"IC从峰值{ic_drift.get('peak_ic', 0):.4f}下降{ic_drift.get('decline_pct', 0):.0f}%，"
                f"建议立即评估是否需要归档"
            )
        elif ic_drift.get("drift_flagged"):
            warning_count += 1
            recommendations.append(
                f"IC持续下降{ic_drift.get('decline_pct', 0):.0f}%，建议加大监控频率"
            )

        # Crowding checks
        if crowding.get("crowding_flagged"):
            if crowding.get("consecutive_increases", 0) >= 8:
                critical_count += 1
            else:
                warning_count += 1
            recommendations.append(
                f"拥挤度连续{crowding.get('consecutive_increases')}期上升，"
                f"当前拥挤度{crowding.get('current_crowding', 0):.3f}，"
                f"建议控制仓位或寻求正交替代"
            )

        # Sharpe checks
        if sharpe.get("sharpe_flagged"):
            critical_count += 1
            recommendations.append(
                f"滚动夏普转负 ({sharpe.get('current_sharpe', 0):.3f})，"
                f"因子当前无正向风险调整收益"
            )
        elif sharpe.get("decline_pct", 0) > 50:
            warning_count += 1
            recommendations.append(
                f"夏普从峰值{sharpe.get('peak_sharpe', 0):.3f}下降{sharpe.get('decline_pct', 0):.0f}%"
            )

        # Overall health
        if critical_count >= 2:
            overall = FactorHealth.CRITICAL
        elif critical_count >= 1 or warning_count >= 2:
            overall = FactorHealth.WARNING
        else:
            overall = FactorHealth.HEALTHY

        if not recommendations:
            recommendations.append("因子各项指标正常，继续监控")

        return FactorHealthReport(
            factor_id=factor_id,
            ic_drift=ic_drift,
            crowding_trend=crowding,
            sharpe_status=sharpe,
            overall_health=overall,
            recommendations=recommendations,
        )
