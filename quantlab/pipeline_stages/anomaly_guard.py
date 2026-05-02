"""Anomaly Guard — data anomaly detection and edge-case handling.

Runs sanity checks on market DataFrames before they enter the pipeline,
detecting NaN floods, zero-volume assets, extreme price gaps, duplicate
rows, future dates, suspected corporate actions, and suspended assets.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Aggregated anomaly detection results."""

    sanity: dict[str, Any] = field(default_factory=dict)
    corporate_actions: dict[str, Any] = field(default_factory=dict)
    suspensions: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnomalyGuard:
    """Detects data anomalies and handles edge cases in market DataFrames.

    Intended to run at the DataRefreshStage boundary so the pipeline
    has visibility into data quality before evolution and validation.

    Usage::

        guard = AnomalyGuard()
        report = guard.run_all(market_df)
        if report.summary["total_anomalies"] > 0:
            logger.warning("Anomalies detected: %s", report.summary)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> AnomalyReport:
        """Run all anomaly checks and return an aggregated report.

        Args:
            df: Market DataFrame with columns including date, asset, close,
                volume, open, high, low.

        Returns:
            AnomalyReport with sanity, corporate_actions, suspensions, and summary.
        """
        if df.empty:
            return AnomalyReport(
                sanity={"error": "empty_dataframe"},
                corporate_actions={},
                suspensions=[],
                summary={"total_anomalies": 1, "empty_dataframe": 1},
                timestamp=datetime.now().isoformat(),
            )

        sanity = self.check_data_sanity(df)
        corporate = self.check_corporate_actions(df)
        suspensions = self.handle_suspensions(df)

        total = (
            int(sanity.get("nan_in_close", 0) > 0)
            + int(sanity.get("zero_volume", 0) > 0)
            + len(sanity.get("price_gaps", []))
            + int(sanity.get("duplicate_rows", 0) > 0)
            + int(sanity.get("future_dates", 0) > 0)
            + len(corporate.get("suspected_splits", []))
            + len(corporate.get("suspected_dividends", []))
            + len(suspensions)
        )

        summary: dict[str, int] = {
            "total_anomalies": total,
            "nan_in_close": int(sanity.get("nan_in_close", 0) > 0),
            "zero_volume_assets": int(sanity.get("zero_volume", 0)),
            "price_gap_assets": len(sanity.get("price_gaps", [])),
            "duplicate_rows": sanity.get("duplicate_rows", 0),
            "future_dates": sanity.get("future_dates", 0),
            "suspected_splits": len(corporate.get("suspected_splits", [])),
            "suspected_dividends": len(corporate.get("suspected_dividends", [])),
            "suspended_assets": len(suspensions),
        }

        return AnomalyReport(
            sanity=sanity,
            corporate_actions=corporate,
            suspensions=suspensions,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

    def check_data_sanity(self, df: pd.DataFrame) -> dict[str, Any]:
        """Basic data sanity checks.

        Returns dict with:
            nan_in_close: count of rows where close is NaN
            zero_volume: count of unique assets with any zero-volume day
            price_gaps: list of {asset, date, pct_change} for >50% single-day jumps
            duplicate_rows: count of duplicate (date, asset) rows
            future_dates: count of dates beyond today
        """
        result: dict[str, Any] = {
            "nan_in_close": 0,
            "zero_volume": 0,
            "price_gaps": [],
            "duplicate_rows": 0,
            "future_dates": 0,
        }

        if df.empty:
            return result

        # NaN in close
        if "close" in df.columns:
            result["nan_in_close"] = int(df["close"].isna().sum())

        # Zero volume
        if "volume" in df.columns:
            zero_vol = df[df["volume"] == 0]
            if not zero_vol.empty and "asset" in df.columns:
                result["zero_volume"] = int(zero_vol["asset"].nunique())

        # Price gaps (>50% single-day change)
        if all(c in df.columns for c in ("close", "asset")):
            price_gaps = self._detect_price_gaps(df)
            result["price_gaps"] = price_gaps

        # Duplicate rows
        dup_cols = [c for c in ("date", "asset") if c in df.columns]
        if len(dup_cols) == 2:
            result["duplicate_rows"] = int(df.duplicated(subset=dup_cols).sum())

        # Future dates
        if "date" in df.columns:
            today = pd.Timestamp.now()
            try:
                dates = pd.to_datetime(df["date"])
                result["future_dates"] = int((dates > today).sum())
            except Exception:
                pass

        return result

    def check_corporate_actions(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect suspected corporate actions from price/volume patterns.

        Returns dict with:
            suspected_splits: list of {asset, date, ratio} where close dropped >40%
                              and volume spiked >3x day-over-day
            suspected_dividends: list of {asset, date, ratio} where close dropped
                                 1-10% with normal volume
        """
        result: dict[str, Any] = {
            "suspected_splits": [],
            "suspected_dividends": [],
        }

        required = {"date", "asset", "close", "volume"}
        if not required.issubset(df.columns):
            return result

        df_sorted = df.sort_values(["asset", "date"]).copy()

        # Compute day-over-day returns per asset
        df_sorted["_ret"] = df_sorted.groupby("asset")["close"].pct_change()
        df_sorted["_vol_ratio"] = df_sorted.groupby("asset")["volume"].pct_change()

        # Suspected splits: close drop > 40% AND volume spike > 3x (200% increase)
        split_mask = (df_sorted["_ret"] < -0.40) & (df_sorted["_vol_ratio"] > 2.0)
        splits = df_sorted[split_mask]
        for _, row in splits.iterrows():
            result["suspected_splits"].append({
                "asset": str(row["asset"]),
                "date": str(row["date"]),
                "ratio": round(float(row["_ret"]), 4),
            })

        # Suspected dividends: close drop 1-10% with normal volume (vol ratio < 1.5x)
        div_mask = (
            (df_sorted["_ret"] < -0.01)
            & (df_sorted["_ret"] > -0.10)
            & (df_sorted["_vol_ratio"].abs() < 0.5)
        )
        dividends = df_sorted[div_mask]
        for _, row in dividends.iterrows():
            result["suspected_dividends"].append({
                "asset": str(row["asset"]),
                "date": str(row["date"]),
                "ratio": round(float(row["_ret"]), 4),
            })

        return result

    def handle_suspensions(self, df: pd.DataFrame) -> list[str]:
        """Identify suspended assets (5+ consecutive days of zero volume or missing close).

        Returns list of asset identifiers to exclude.
        """
        required = {"date", "asset", "close", "volume"}
        if not required.issubset(df.columns):
            return []

        df_sorted = df.sort_values(["asset", "date"]).copy()
        excluded: list[str] = []

        for asset, group in df_sorted.groupby("asset"):
            asset_str = str(asset)
            # Mark days where close is NaN or volume is zero
            is_stalled = group["close"].isna() | (group["volume"] == 0)
            # Find runs of consecutive stalled days
            streak = 0
            max_streak = 0
            for val in is_stalled:
                if val:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            if max_streak >= 5:
                excluded.append(asset_str)

        return excluded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_price_gaps(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Find assets with >50% single-day price jumps."""
        gaps: list[dict[str, Any]] = []
        df_sorted = df.sort_values(["asset", "date"]).copy()
        df_sorted["_ret"] = df_sorted.groupby("asset")["close"].pct_change()
        extreme = df_sorted[df_sorted["_ret"].abs() > 0.50]
        for _, row in extreme.iterrows():
            gaps.append({
                "asset": str(row["asset"]),
                "date": str(row["date"]),
                "pct_change": round(float(row["_ret"]), 4),
            })
        return gaps
