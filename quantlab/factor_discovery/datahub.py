"""DataHub 统一数据抽象层 —— 收敛当前分散的数据操作。

核心能力：
1. 统一横截面数据加载、刷新、元数据管理
2. Provider 抽象：akshare / 雪球 / 本地CSV / 未来新数据源
3. 因子面板缓存与复用
4. 数据质量报告

设计借鉴：
- Qlib: DataProvider + Dataset 抽象
- RD-Agent-Q: Data-Centric 思路
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 数据质量报告
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DataQualityReport:
    """数据质量快照。"""
    source: str
    asset_count: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    total_rows: int = 0
    coverage: float = 0.0
    nan_ratio: float = 0.0
    industry_coverage: float = 0.0
    market_cap_coverage: float = 0.0
    metadata_asset_count: int = 0
    metadata_failed_assets: int = 0
    last_refreshed: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. 数据源 Provider 抽象
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """数据源抽象基类。"""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load_cross_section(self, data_path: Path, **kwargs) -> pd.DataFrame:
        ...

    @abstractmethod
    def refresh_cross_section(self, data_path: Path, **kwargs) -> dict[str, Any]:
        ...

    @abstractmethod
    def data_quality(self, data_path: Path) -> DataQualityReport:
        ...


class LocalCSVProvider(DataProvider):
    """本地 CSV 数据 Provider（当前主数据源）。"""

    def name(self) -> str:
        return "local_csv"

    def load_cross_section(self, data_path: Path, **kwargs) -> pd.DataFrame:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "asset" in df.columns:
            df["asset"] = df["asset"].astype(str)
        return df

    def refresh_cross_section(self, data_path: Path, **kwargs) -> dict[str, Any]:
        """刷新通过 quantlab.pipeline.refresh_cross_section_data 代理。"""
        from quantlab.pipeline import refresh_cross_section_data
        return refresh_cross_section_data(data_path=Path(data_path), **kwargs)

    def data_quality(self, data_path: Path) -> DataQualityReport:
        path = Path(data_path)
        if not path.exists():
            return DataQualityReport(source=self.name(), notes=["文件不存在"])

        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        asset_col = "asset" if "asset" in df.columns else None

        report = DataQualityReport(
            source=self.name(),
            total_rows=len(df),
            nan_ratio=round(float(df.isnull().mean().mean()), 4) if not df.empty else 1.0,
        )

        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            report.date_range_start = str(dates.min()) if pd.notna(dates.min()) else ""
            report.date_range_end = str(dates.max()) if pd.notna(dates.max()) else ""

        if asset_col and asset_col in df.columns:
            report.asset_count = int(df[asset_col].nunique())
            report.coverage = round(1.0 - report.nan_ratio, 4)

        # 检查元数据
        metadata_path = path.parent / path.name.replace(".csv", "_asset_metadata.csv")
        if metadata_path.exists():
            meta_df = pd.read_csv(metadata_path)
            report.metadata_asset_count = len(meta_df)
            if "industry" in meta_df.columns:
                known = (meta_df["industry"] != "unknown").sum()
                report.industry_coverage = round(known / max(len(meta_df), 1), 4)
            if "market_cap" in meta_df.columns:
                known = meta_df["market_cap"].notna().sum()
                report.market_cap_coverage = round(known / max(len(meta_df), 1), 4)

        # 检查刷新报告
        report_path = path.parent / path.name.replace(".csv", "_refresh_report.json")
        if report_path.exists():
            rp = json.loads(report_path.read_text(encoding="utf-8"))
            report.last_refreshed = rp.get("timestamp", "")
            report.metadata_failed_assets = rp.get("metadata_failed_assets", 0)
            if rp.get("industry_coverage") is not None:
                report.industry_coverage = rp["industry_coverage"]
            if rp.get("market_cap_coverage") is not None:
                report.market_cap_coverage = rp["market_cap_coverage"]

        return report


# ---------------------------------------------------------------------------
# 3. DataHub 主类
# ---------------------------------------------------------------------------

class DataHub:
    """统一数据访问层。

    使用方式：
        hub = DataHub()
        df = hub.load("D:/quant-agent/data/hs300_cross_section.csv")
        report = hub.quality("D:/quant-agent/data/hs300_cross_section.csv")
        hub.refresh("D:/quant-agent/data/hs300_cross_section.csv")
    """

    def __init__(self, default_provider: DataProvider | None = None) -> None:
        self._providers: dict[str, DataProvider] = {}
        self._cache: dict[str, pd.DataFrame] = {}
        self._default = default_provider or LocalCSVProvider()
        self.register_provider(self._default)

    def register_provider(self, provider: DataProvider) -> None:
        self._providers[provider.name()] = provider

    @property
    def default_provider(self) -> DataProvider:
        return self._default

    def load(
        self,
        data_path: str | Path,
        provider_name: str | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """加载横截面数据。"""
        key = str(data_path)
        if use_cache and key in self._cache:
            return self._cache[key]

        provider = self._providers.get(provider_name or "") or self._default
        df = provider.load_cross_section(Path(data_path), **kwargs)

        if use_cache:
            self._cache[key] = df

        return df

    def refresh(
        self,
        data_path: str | Path,
        provider_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """刷新数据。"""
        provider = self._providers.get(provider_name or "") or self._default
        result = provider.refresh_cross_section(Path(data_path), **kwargs)

        # 刷新后清除缓存
        key = str(data_path)
        self._cache.pop(key, None)

        return result

    def quality(
        self,
        data_path: str | Path,
        provider_name: str | None = None,
    ) -> DataQualityReport:
        """获取数据质量报告。"""
        provider = self._providers.get(provider_name or "") or self._default
        return provider.data_quality(Path(data_path))

    def invalidate_cache(self, data_path: str | Path | None = None) -> None:
        """清除缓存。"""
        if data_path is None:
            self._cache.clear()
        else:
            self._cache.pop(str(data_path), None)
