"""因子组合构建 —— 从因子面板到投资组合权重。

支持的权重方案：
- equal_weight: 等权
- score_weight: 按因子得分加权
- ic_weight: 按 IC 加权（需历史 IC 序列）
- sector_neutral: 行业中性化

支持的选股方案：
- top_n: 选前 N 只
- threshold: 因子得分超过阈值的
- quantile: 选特定分位组（如最高/最低五分位）

设计参考：
- WorldQuant Alpha 平台标准：long-only, long-short, sector-neutral
- Barra 风险模型：行业/市值暴露控制
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class PortfolioWeightScheme(str, Enum):
    EQUAL = "equal"
    SCORE = "score"
    IC_WEIGHT = "ic_weight"
    SECTOR_NEUTRAL = "sector_neutral"


class FactorPortfolioConstructor:
    """因子信号 → 组合权重。"""

    def __init__(
        self,
        n_long: int = 50,
        n_short: int = 0,
        weight_scheme: PortfolioWeightScheme = PortfolioWeightScheme.EQUAL,
        sector_neutral: bool = False,
        max_single_weight: float = 0.05,
        min_single_weight: float = 0.005,
    ) -> None:
        self.n_long = n_long
        self.n_short = n_short
        self.weight_scheme = weight_scheme
        self.sector_neutral = sector_neutral
        self.max_single_weight = max_single_weight
        self.min_single_weight = min_single_weight

    def construct_weights(
        self,
        factor_panel: pd.DataFrame,
        ic_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """根据因子面板构建每日组合权重。

        Parameters
        ----------
        factor_panel : DataFrame
            必须包含: date, asset, factor_value
        ic_series : Series, optional
            历史每日 IC，用于 ic_weight 方案

        Returns
        -------
        DataFrame with columns: date, asset, weight, side
        """
        panel = factor_panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        if "factor_value" not in panel.columns:
            raise ValueError("factor_panel 必须包含 factor_value 列")

        all_weights: list[pd.DataFrame] = []

        for date, group in panel.groupby("date", sort=True):
            group = group.dropna(subset=["factor_value"]).copy()
            if len(group) < max(self.n_long, 5):
                continue

            # 排序
            sorted_group = group.sort_values("factor_value", ascending=False)

            # 多头
            long_assets = sorted_group.head(self.n_long)
            long_weights = self._compute_weights(long_assets, side="long", ic_series=ic_series)

            # 空头
            short_weights = pd.DataFrame()
            if self.n_short > 0:
                short_assets = sorted_group.tail(self.n_short)
                short_weights = self._compute_weights(short_assets, side="short", ic_series=ic_series)
                short_weights["weight"] = -short_weights["weight"]

            date_weights = pd.concat([long_weights, short_weights], ignore_index=True)
            date_weights["date"] = date
            all_weights.append(date_weights)

        if not all_weights:
            return pd.DataFrame(columns=["date", "asset", "weight", "side"])

        result = pd.concat(all_weights, ignore_index=True)
        result = result.sort_values(["date", "asset"]).reset_index(drop=True)
        return result

    def _compute_weights(
        self,
        assets: pd.DataFrame,
        side: str = "long",
        ic_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """计算单组资产权重。"""
        n = len(assets)
        if n == 0:
            return pd.DataFrame(columns=["asset", "weight", "side"])

        if self.weight_scheme == PortfolioWeightScheme.EQUAL:
            raw_weight = 1.0 / n

        elif self.weight_scheme == PortfolioWeightScheme.SCORE:
            scores = assets["factor_value"].abs()
            total = scores.sum()
            raw_weights = (scores / total).values if total > 0 else np.full(n, 1.0 / n)

        elif self.weight_scheme == PortfolioWeightScheme.IC_WEIGHT and ic_series is not None:
            # 用最近 20 日 IC 均值作为方向确认
            recent_ic = ic_series.tail(20).mean() if len(ic_series) >= 5 else 0.0
            if recent_ic < 0:
                scores = assets["factor_value"].rank(ascending=True)
            else:
                scores = assets["factor_value"].rank(ascending=False)
            total = scores.sum()
            raw_weights = (scores / total).values if total > 0 else np.full(n, 1.0 / n)

        elif self.weight_scheme == PortfolioWeightScheme.SECTOR_NEUTRAL:
            raw_weights = self._sector_neutral_weights(assets)
        else:
            raw_weight = 1.0 / n

        if self.weight_scheme in (PortfolioWeightScheme.EQUAL,):
            weights = np.full(n, raw_weight)
        elif isinstance(raw_weights := None, np.ndarray) or True:
            # 已经在上面赋值了
            if self.weight_scheme == PortfolioWeightScheme.EQUAL:
                weights = np.full(n, raw_weight)
            else:
                weights = raw_weights

        # 截断
        weights = np.clip(weights, self.min_single_weight, self.max_single_weight)
        weights = weights / weights.sum()  # 重新归一化

        result = pd.DataFrame({
            "asset": assets["asset"].values,
            "weight": weights,
            "side": side,
        })
        return result

    def _sector_neutral_weights(self, assets: pd.DataFrame) -> np.ndarray:
        """行业中性化权重。"""
        if "industry" not in assets.columns:
            return np.full(len(assets), 1.0 / len(assets))

        industries = assets["industry"].fillna("unknown").unique()
        n_industries = len(industries)
        per_industry_weight = 1.0 / n_industries

        weights = np.zeros(len(assets))
        for ind in industries:
            mask = assets["industry"].fillna("unknown") == ind
            n_in_ind = mask.sum()
            if n_in_ind > 0:
                weights[mask] = per_industry_weight / n_in_ind

        total = weights.sum()
        return weights / total if total > 0 else np.full(len(assets), 1.0 / len(assets))
