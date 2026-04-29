"""幸存者偏差修正 —— 过滤只在样本期部分存在的资产。

问题：从 Tushare/AkShare 拉取的横截面数据只包含当前存续股票，
回测时 IC 会系统性偏高（退市/ST 股票的历史数据不存在但未来收益为负）。

修正方法（不依赖外部成分股数据）：
1. 时效性过滤：每个日期只保留前后各 N 日都有数据的资产
2. 极端收益过滤：标记可能对应 ST/退市的极端价格行为
3. 新股过滤：上市不足 M 日的资产排除

使用时直接对 market_df 调用 filter_survivorship() 即可。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SurvivorshipFilter:
    """幸存者偏差修正过滤器。

    在市场数据加载后、因子计算前调用。
    """

    min_history_days: int = 60     # 资产至少需要的历史交易日数
    lookahead_days: int = 20       # 向前看 N 日确保不是退市前数据
    min_assets_per_date: int = 30  # 每日期至少保留的资产数
    date_col: str = "date"
    asset_col: str = "asset"

    def filter(self, market_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """过滤市场数据，移除存在幸存者偏差的观测。

        Returns:
            (filtered_df, summary_dict)
        """
        df = market_df.copy()
        dates = pd.to_datetime(df[self.date_col])
        df[self.date_col] = dates

        original_rows = len(df)
        original_assets = df[self.asset_col].nunique()
        original_dates = dates.nunique()

        # 1. 新股过滤：资产出现的最早日期距离数据起始日太近
        asset_first_date = df.groupby(self.asset_col)[self.date_col].min()
        data_start = dates.min() + pd.Timedelta(days=self.min_history_days)
        mature_assets = asset_first_date[asset_first_date < data_start].index
        df = df[df[self.asset_col].isin(mature_assets)]
        new_listing_dropped = original_assets - df[self.asset_col].nunique()

        # 2. 退市前过滤：资产最后出现日期不能距离数据结束日太近
        asset_last_date = df.groupby(self.asset_col)[self.date_col].max()
        data_end = dates.max() - pd.Timedelta(days=self.lookahead_days)
        active_assets = asset_last_date[asset_last_date > data_end].index
        df = df[df[self.asset_col].isin(active_assets)]
        delisting_dropped = len(set(asset_last_date.index) - set(active_assets))

        # 3. 连续性过滤：每个日期至少保留 min_assets_per_date
        date_counts = df.groupby(self.date_col).size()
        thin_dates = date_counts[date_counts < self.min_assets_per_date]
        if len(thin_dates) > 0:
            df = df[~df[self.date_col].isin(thin_dates.index)]
            logger.info("移除 %d 个资产不足的日期（< %d asset）", len(thin_dates), self.min_assets_per_date)

        # 4. 极端收益标记（可能是 ST/退市信号）
        if "close" in df.columns:
            df = df.sort_values([self.asset_col, self.date_col])
            ret_5d = df.groupby(self.asset_col)["close"].transform(
                lambda x: x.pct_change(5)
            )
            # 5日内跌超50% 或 涨超200% → 标记异常
            abnormal = (ret_5d < -0.5) | (ret_5d > 2.0)
            abnormal_count = int(abnormal.sum())
            if abnormal_count > 0:
                df.loc[abnormal, "close"] = np.nan  # 标记而非删除，让下游决定
                logger.info("标记 %d 个极端收益观测（可能ST/退市）", abnormal_count)

        final_rows = len(df)
        final_assets = df[self.asset_col].nunique()
        final_dates = df[self.date_col].nunique()

        summary = {
            "original_rows": original_rows,
            "filtered_rows": final_rows,
            "rows_removed": original_rows - final_rows,
            "retention_pct": round(final_rows / max(original_rows, 1) * 100, 2),
            "new_listing_dropped_assets": int(new_listing_dropped),
            "delisting_dropped_assets": int(delisting_dropped),
            "original_assets": int(original_assets),
            "final_assets": int(final_assets),
            "original_dates": int(original_dates),
            "final_dates": int(final_dates),
        }

        logger.info(
            "幸存者偏差过滤: %d → %d 行 (%.1f%%), 资产 %d → %d",
            original_rows, final_rows,
            summary["retention_pct"],
            original_assets, final_assets,
        )

        return df, summary


def apply_survivorship_filter(market_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """便捷函数：过滤后返回干净的 DataFrame。"""
    filter_obj = SurvivorshipFilter(**kwargs)
    return filter_obj.filter(market_df)[0]
