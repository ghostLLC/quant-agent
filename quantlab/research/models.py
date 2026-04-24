from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResearchTask:
    task_type: str
    data_path: Path
    strategy_name: str = "ma_cross"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    parameter_grid: dict[str, list] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchTaskResult:
    task: ResearchTask
    summary: dict[str, Any]
    payload: dict[str, Any]
    history_path: str | None = None
    credibility: dict[str, Any] = field(default_factory=dict)



@dataclass
class ResearchPlan:
    goal: str
    tasks: list[ResearchTask] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
