"""共享 Rank IC 计算 —— 所有管线统一使用此模块。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _align_factor(
    market_df: pd.DataFrame,
    factor_values: pd.Series,
    date_col: str,
    asset_col: str,
) -> pd.DataFrame:
    """将因子值对齐到市场数据，处理 MultiIndex Series。"""
    aligned = market_df.copy()
    if isinstance(factor_values.index, pd.MultiIndex):
        factor_flat = factor_values.reset_index()
        n_cols = len(factor_flat.columns)
        if n_cols == 3:
            factor_flat.columns = [date_col, asset_col, "factor"]
        elif n_cols >= 3:
            factor_flat = factor_flat.iloc[:, [0, 1, n_cols - 1]]
            factor_flat.columns = [date_col, asset_col, "factor"]
        else:
            factor_flat.columns = [date_col, "factor"]
            aligned = aligned.merge(factor_flat, on=date_col, how="left")
            return aligned
        aligned = aligned.merge(factor_flat, on=[date_col, asset_col], how="left")
    else:
        aligned["factor"] = factor_values
    return aligned


def compute_rank_ic(
    factor_values: pd.Series,
    market_df: pd.DataFrame,
    forward_days: int = 5,
    min_samples: int = 20,
    date_col: str = "date",
    asset_col: str | None = None,
    close_col: str = "close",
) -> dict[str, float]:
    """计算因子 Rank IC，返回标准指标字典。

    Args:
        factor_values: 因子值 Series（支持 MultiIndex (date, asset) 或普通索引）
        market_df: 市场数据，必须包含 date, asset (或 ts_code), close 列
        forward_days: 前向收益天数
        min_samples: 每截面最少样本数
    """
    try:
        if close_col not in market_df.columns:
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

        if asset_col is None:
            if "asset" in market_df.columns:
                asset_col = "asset"
            elif "ts_code" in market_df.columns:
                asset_col = "ts_code"
            else:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

        if date_col not in market_df.columns:
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

        aligned = _align_factor(market_df, factor_values, date_col, asset_col)
        aligned = aligned.sort_values([asset_col, date_col])
        aligned["fwd_ret"] = aligned.groupby(asset_col)[close_col].shift(-forward_days) / aligned[close_col] - 1

        rank_ics = []
        for _, group in aligned.groupby(date_col):
            valid = group[["factor", "fwd_ret"]].dropna()
            if len(valid) >= min_samples:
                ric = valid["factor"].rank().corr(valid["fwd_ret"].rank(), method="pearson")
                if not np.isnan(ric):
                    rank_ics.append(ric)

        if not rank_ics:
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

        ic_mean = float(np.mean(rank_ics))
        ic_std = float(np.std(rank_ics))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        coverage = float(factor_values.notna().mean())

        return {
            "ic_mean": round(ic_mean, 4),
            "rank_ic_mean": round(ic_mean, 4),
            "ic_ir": round(ic_ir, 4),
            "coverage": round(coverage, 4),
        }
    except Exception as exc:
        logger.warning(f"Rank IC 计算失败: {exc}")
        return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}


def compute_ic_sequence(
    factor_values: pd.Series,
    market_df: pd.DataFrame,
    forward_days: int = 5,
    min_samples: int = 20,
    date_col: str = "date",
    asset_col: str | None = None,
    close_col: str = "close",
) -> pd.Series:
    """计算每日 IC 序列（用于衰减分析、稳定性评估）。"""
    try:
        if close_col not in market_df.columns:
            return pd.Series(dtype=float)

        if asset_col is None:
            if "asset" in market_df.columns:
                asset_col = "asset"
            elif "ts_code" in market_df.columns:
                asset_col = "ts_code"
            else:
                return pd.Series(dtype=float)

        if date_col not in market_df.columns:
            return pd.Series(dtype=float)

        aligned = _align_factor(market_df, factor_values, date_col, asset_col)
        aligned = aligned.sort_values([asset_col, date_col])
        aligned["fwd_ret"] = aligned.groupby(asset_col)[close_col].shift(-forward_days) / aligned[close_col] - 1

        ics: dict[Any, float] = {}
        for date_val, group in aligned.groupby(date_col):
            valid = group[["factor", "fwd_ret"]].dropna()
            if len(valid) >= min_samples:
                ric = valid["factor"].rank().corr(valid["fwd_ret"].rank(), method="pearson")
                if not np.isnan(ric):
                    ics[date_val] = float(ric)

        return pd.Series(ics, name="rank_ic").sort_index()
    except Exception as exc:
        logger.warning(f"IC 序列计算失败: {exc}")
        return pd.Series(dtype=float)
