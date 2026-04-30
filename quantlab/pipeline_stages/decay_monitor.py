"""阶段 2: 因子衰减监控。"""

from __future__ import annotations

import logging
from typing import Any

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class DecayMonitorStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.decay_monitor import FactorDecayMonitor
        monitor = FactorDecayMonitor(data_path=ctx.data_path)
        return monitor.check_all()
