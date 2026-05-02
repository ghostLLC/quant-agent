"""管道阶段基础设施。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """管线共享上下文，在各阶段间传递。"""
    data_path: Path = field(default_factory=lambda: Path(DEFAULT_CROSS_SECTION_DATA_PATH))
    directions: list[str] = field(default_factory=lambda: [
        "momentum_reversal", "quality_earnings",
        "volume_price_divergence", "volatility_regime", "liquidity_premium",
    ])
    evolution_rounds: int = 3
    max_candidates_per_round: int = 5
    use_adaptive_directions: bool = True
    use_multi_agent: bool = True

    # Cached data / meta
    _market_df: pd.DataFrame | None = field(default=None, repr=False)
    _meta: dict[str, Any] = field(default_factory=dict, repr=False)
    _data_loaded: bool = field(default=False, repr=False)

    def load_data(self, apply_survivorship: bool = True, check_quality: bool = True) -> pd.DataFrame:
        """加载市场数据（带缓存，跨阶段复用，避免重复IO）。"""
        if self._market_df is not None:
            return self._market_df
        df = _load_market_data(self.data_path, apply_survivorship, check_quality)
        self._market_df = df
        self._data_loaded = True
        return df

    def invalidate_cache(self) -> None:
        """强制刷新缓存（仅在数据刷新后调用）。"""
        self._market_df = None
        self._data_loaded = False


class PipelineStage(ABC):
    """管线阶段的抽象基类。"""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        ...


def _load_market_data(data_path: Path, apply_survivorship: bool, check_quality: bool) -> pd.DataFrame:
    try:
        from quantlab.factor_discovery.datahub import DataHub
        hub = DataHub()
        df = hub.load(str(data_path), use_cache=False)
        if check_quality and not df.empty:
            qr = hub.check_quality(str(data_path))
            logger.info("数据质量检查: score=%.4f", qr.get("score", 0))
    except Exception:
        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            return pd.DataFrame()

    if apply_survivorship and not df.empty:
        try:
            from quantlab.factor_discovery.survivorship import apply_survivorship_filter
            df = apply_survivorship_filter(df)
        except Exception as exc:
            logger.warning("幸存者偏差过滤失败: %s", exc)

    return df
