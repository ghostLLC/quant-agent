"""阶段 1: 增量数据刷新 —— 集成 build_dataset.py 增量更新。

刷新策略:
  1. 调用 build_dataset.py --refresh 拉取新的 OHLCV 交易日
  2. 检测是否有新的财报数据（季度切换时），有则更新 PE/PB
  3. 失败时回退到缓存数据，不阻塞管线
  4. 刷新后运行 AnomalyGuard 检测数据异常
"""

from __future__ import annotations

import logging
from typing import Any

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class DataRefreshStage(PipelineStage):
    """增量数据刷新阶段。

    优先使用 build_dataset.py --refresh（tushare OHLCV 增量 + PE/PB 更新），
    失败时回退到 AkShareIncrementalProvider。
    刷新后运行 AnomalyGuard 检测数据异常。
    """

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        # 主路径: build_dataset.py --refresh
        try:
            from build_dataset import incremental_refresh, THICK_DATASET
            incremental_refresh(THICK_DATASET, THICK_DATASET)
            ctx.invalidate_cache()
            result: dict[str, Any] = {"status": "success", "method": "build_dataset_refresh"}
        except Exception as exc:
            logger.warning("厚数据集刷新失败: %s，尝试 AkShare 回退", exc)
            # 回退路径: AkShare
            try:
                from quantlab.data.tushare_provider import AkShareIncrementalProvider
                provider = AkShareIncrementalProvider()
                fallback_result = provider.refresh_cross_section(ctx.data_path)
                ctx.invalidate_cache()
                result = {"status": "success_fallback", "result": fallback_result}
            except Exception as exc2:
                logger.warning("AkShare 也失败: %s", exc2)
                # 最后回退: 不清空缓存，用旧数据继续
                logger.warning("所有数据源刷新失败，使用缓存数据继续管线")
                result = {"status": "skipped", "reason": "all sources failed"}

        # 刷新后运行异常检测
        self._run_anomaly_check(ctx)

        return result

    def _run_anomaly_check(self, ctx: PipelineContext) -> None:
        """Run AnomalyGuard on the refreshed data and store results in ctx._meta."""
        try:
            from .anomaly_guard import AnomalyGuard
            df = ctx.load_data(apply_survivorship=False, check_quality=False)
            if df.empty:
                logger.info("AnomalyGuard 跳过（数据为空）")
                return
            guard = AnomalyGuard()
            report = guard.run_all(df)
            ctx._meta["anomalies"] = report.to_dict()
            summary = report.summary
            if summary.get("total_anomalies", 0) > 0:
                logger.warning(
                    "AnomalyGuard 检测到 %d 项异常: nan_close=%s, zero_vol=%d, price_gaps=%d, "
                    "duplicates=%d, future_dates=%d, splits=%d, dividends=%d, suspended=%d",
                    summary["total_anomalies"],
                    summary.get("nan_in_close", 0),
                    summary.get("zero_volume_assets", 0),
                    summary.get("price_gap_assets", 0),
                    summary.get("duplicate_rows", 0),
                    summary.get("future_dates", 0),
                    summary.get("suspected_splits", 0),
                    summary.get("suspected_dividends", 0),
                    summary.get("suspended_assets", 0),
                )
            else:
                logger.info("AnomalyGuard 数据正常，未发现异常")
        except Exception as exc:
            logger.warning("AnomalyGuard 检测失败: %s", exc)
