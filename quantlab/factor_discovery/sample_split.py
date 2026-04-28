"""样本外自动拆分 —— 6 个月预留测试集。

加载数据时自动判断最新日期 T_latest：
- 训练窗口：T_start ~ T_latest - 6M
- 测试窗口：T_latest - 6M ~ T_latest
- 因子发掘和进化只在训练窗口上跑
- 交付筛选时必须同时检查样本外 IC

使用方式：
    from quantlab.factor_discovery.sample_split import SampleSplitter
    
    splitter = SampleSplitter(oos_months=6)
    train_df, test_df = splitter.split(df)
    
    # 或直接获取 cutoff 日期
    cutoff = splitter.get_cutoff(df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """拆分结果。"""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cutoff_date: pd.Timestamp
    train_start: pd.Timestamp
    test_end: pd.Timestamp
    train_trading_days: int
    test_trading_days: int
    train_assets: int
    test_assets: int
    train_rows: int
    test_rows: int
    sufficient: bool  # 数据是否足够做有意义的拆分

    def summary(self) -> dict[str, Any]:
        return {
            "cutoff_date": str(self.cutoff_date.date()),
            "train_period": f"{self.train_start.date()} ~ {self.cutoff_date.date()}",
            "test_period": f"{self.cutoff_date.date()} ~ {self.test_end.date()}",
            "train_trading_days": self.train_trading_days,
            "test_trading_days": self.test_trading_days,
            "train_assets": self.train_assets,
            "test_assets": self.test_assets,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "sufficient": self.sufficient,
        }


class SampleSplitter:
    """样本外自动拆分器。

    默认预留最近 6 个月作为样本外测试集。
    要求训练集至少 120 个交易日（约 6 个月），测试集至少 40 个交易日（约 2 个月）。
    """

    MIN_TRAIN_DAYS = 120
    MIN_TEST_DAYS = 40

    def __init__(
        self,
        oos_months: int = 6,
        date_col: str = "date",
        asset_col: str = "asset",
    ):
        self.oos_months = oos_months
        self.date_col = date_col
        self.asset_col = asset_col

    def get_cutoff(self, df: pd.DataFrame) -> pd.Timestamp:
        """计算样本外起始日期。"""
        dates = pd.to_datetime(df[self.date_col])
        latest = dates.max()
        cutoff = latest - pd.DateOffset(months=self.oos_months)
        return cutoff

    def split(self, df: pd.DataFrame) -> SplitResult:
        """拆分数据集为训练集和测试集。"""
        dates = pd.to_datetime(df[self.date_col])
        cutoff = self.get_cutoff(df)

        train_df = df[dates <= cutoff].copy()
        test_df = df[dates > cutoff].copy()

        train_trading_days = pd.to_datetime(train_df[self.date_col]).nunique()
        test_trading_days = pd.to_datetime(test_df[self.date_col]).nunique()
        train_assets = train_df[self.asset_col].nunique() if self.asset_col in train_df.columns else 0
        test_assets = test_df[self.asset_col].nunique() if self.asset_col in test_df.columns else 0

        # 判断数据是否足够
        sufficient = (
            train_trading_days >= self.MIN_TRAIN_DAYS
            and test_trading_days >= self.MIN_TEST_DAYS
        )

        result = SplitResult(
            train_df=train_df,
            test_df=test_df,
            cutoff_date=cutoff,
            train_start=pd.to_datetime(dates.min()),
            test_end=pd.to_datetime(dates.max()),
            train_trading_days=train_trading_days,
            test_trading_days=test_trading_days,
            train_assets=train_assets,
            test_assets=test_assets,
            train_rows=len(train_df),
            test_rows=len(test_df),
            sufficient=sufficient,
        )

        if not sufficient:
            logger.warning(
                f"数据不足：训练集 {train_trading_days} 天（需 {self.MIN_TRAIN_DAYS}+），"
                f"测试集 {test_trading_days} 天（需 {self.MIN_TEST_DAYS}+）"
            )

        return result

    @staticmethod
    def oos_ic_check(
        factor_values: pd.Series,
        returns: pd.Series,
        test_dates: set[str] | set[pd.Timestamp] | None = None,
        date_series: pd.Series | None = None,
    ) -> dict[str, Any]:
        """计算样本外 IC。

        Args:
            factor_values: 因子值 Series（需与 returns 对齐）
            returns: 下期收益率 Series
            test_dates: 样本外日期集合
            date_series: 日期列（如果 test_dates 与 index 不对齐时需要）

        Returns:
            dict: 包含 oos_rank_ic, oos_ic_ir, oos_positive_ratio 等
        """
        if test_dates is not None and date_series is not None:
            mask = date_series.isin(test_dates)
            fv = factor_values[mask].astype(float)
            ret = returns[mask].astype(float)
        else:
            fv = factor_values.astype(float)
            ret = returns.astype(float)

        if len(fv) < 50:
            return {"oos_rank_ic": 0.0, "oos_ic_ir": 0.0, "oos_positive_ratio": 0.0, "oos_n_days": 0}

        # 按日计算 Rank IC
        valid = pd.DataFrame({"factor": fv, "return": ret}).dropna()
        if len(valid) < 50:
            return {"oos_rank_ic": 0.0, "oos_ic_ir": 0.0, "oos_positive_ratio": 0.0, "oos_n_days": 0}

        # 按日期分组计算 Spearman 相关系数
        if date_series is not None and test_dates is not None:
            dates_aligned = date_series[mask]
        else:
            dates_aligned = None

        if dates_aligned is not None and len(dates_aligned) == len(valid):
            valid["date"] = dates_aligned.values
            daily_ic = valid.groupby("date").apply(
                lambda g: g["factor"].corr(g["return"], method="spearman")
            ).dropna()
        else:
            # 无日期分组，计算整体 IC
            ic = valid["factor"].corr(valid["return"], method="spearman")
            daily_ic = pd.Series([ic])

        mean_ic = daily_ic.mean()
        std_ic = daily_ic.std()
        ic_ir = mean_ic / (std_ic + 1e-10)
        positive_ratio = (daily_ic > 0).mean()

        return {
            "oos_rank_ic": round(float(mean_ic), 6),
            "oos_ic_ir": round(float(ic_ir), 6),
            "oos_positive_ratio": round(float(positive_ratio), 4),
            "oos_n_days": len(daily_ic),
        }
