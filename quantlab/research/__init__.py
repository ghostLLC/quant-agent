from __future__ import annotations

from typing import Any

from quantlab.research.models import ResearchPlan, ResearchTask, ResearchTaskResult
from quantlab.research.protocol import SUPPORTED_TASK_TYPES, normalize_task_type

__all__ = [
    "ResearchPlan",
    "ResearchTask",
    "ResearchTaskExecutor",
    "ResearchTaskResult",
    "SUPPORTED_TASK_TYPES",
    "normalize_task_type",
]


def __getattr__(name: str) -> Any:
    if name == "ResearchTaskExecutor":
        from quantlab.research.executor import ResearchTaskExecutor

        return ResearchTaskExecutor
    raise AttributeError(f"module 'quantlab.research' has no attribute {name!r}")

