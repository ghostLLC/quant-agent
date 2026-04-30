"""阶段 1: 增量数据刷新。"""

from __future__ import annotations

import logging
from typing import Any

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class DataRefreshStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        try:
            from quantlab.data.tushare_provider import AkShareIncrementalProvider
            provider = AkShareIncrementalProvider()
            result = provider.refresh_cross_section(ctx.data_path)
            ctx.invalidate_cache()
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.warning("增量刷新失败，尝试 Tushare Pro: %s", exc)
            try:
                from quantlab.data.tushare_provider import TushareProProvider
                provider = TushareProProvider()
                if provider.available:
                    result = provider.refresh_cross_section(ctx.data_path)
                    ctx.invalidate_cache()
                    return {"status": "success_fallback", "result": result}
            except Exception as exc2:
                logger.warning("Tushare Pro 也失败: %s", exc2)
            return {"status": "skipped", "reason": str(exc)[:200]}
