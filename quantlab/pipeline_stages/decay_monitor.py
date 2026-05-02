"""阶段 2: 因子衰减监控 —— 检测衰减并触发定向再进化。

检测因子衰减后，将衰减因子信息写入 ctx._meta，
供 EvolutionStage 读取并进行定向再进化。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class DecayMonitorStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.decay_monitor import FactorDecayMonitor
        monitor = FactorDecayMonitor(data_path=ctx.data_path)
        result = monitor.check_all()

        # 写入衰减因子信息到 ctx._meta，供 EvolutionStage 闭环使用
        decayed_factors = result.get("decayed_factors", [])
        if decayed_factors:
            logger.info("检测到 %d 个衰减因子，将触发定向再进化", len(decayed_factors))
            ctx._meta["decayed_factors"] = decayed_factors
            ctx._meta["decay_triggered_rediscovery"] = True

        return result
