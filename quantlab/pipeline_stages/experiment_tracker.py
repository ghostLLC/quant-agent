"""Experiment tracker — records full provenance for every factor and pipeline run.

Tracks:
  - Run metadata (run_id, timestamps, LLM config)
  - Per-stage decisions (directions explored, agents used, prompts)
  - Factor attribution (which direction/round/agent produced each factor)
  - Evaluation results (IC, OOS performance, screening outcome)

Stored in: data/scheduler/experiments/{run_id}.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from quantlab.config import DATA_DIR

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = DATA_DIR / "scheduler" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FactorProvenance:
    """Full attribution trail for a single factor."""
    factor_id: str
    direction: str
    source: str = ""  # "llm_multi_agent" | "template_evolution" | "seed"
    run_id: str = ""
    round_number: int = 0
    agent_role: str = ""  # R1, P1, etc.
    parent_factor_id: str = ""
    mutation_op: str = ""  # if evolved from parent
    prompt_summary: str = ""  # truncated prompt used
    knowledge_injected: list[str] = field(default_factory=list)
    block_tree_summary: str = ""
    expression: str = ""
    # Evaluation
    rank_ic_mean: float = 0.0
    icir: float = 0.0
    oos_ic: float = 0.0
    screening_passed: bool = False
    screening_fail_reasons: list[str] = field(default_factory=list)
    final_status: str = "draft"
    # Timestamps
    created_at: str = ""
    evaluated_at: str = ""


@dataclass
class ExperimentRun:
    """Full experiment record for one pipeline run."""
    run_id: str
    run_date: str = ""
    started_at: str = ""
    ended_at: str = ""
    status: str = "running"
    # Config
    llm_model: str = ""
    llm_available: bool = False
    directions_explored: list[str] = field(default_factory=list)
    adaptive_selection: bool = True
    # Results
    total_candidates: int = 0
    factors_approved: int = 0
    factors_screened: int = 0
    factors_deliverable: int = 0
    # Agent feedback context
    agent_feedback_used: dict[str, Any] = field(default_factory=dict)
    # Per-factor provenance
    factor_provenance: list[dict[str, Any]] = field(default_factory=list)
    # Stage summaries
    stage_results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Tracks experiment provenance across pipeline runs."""

    def __init__(self) -> None:
        self._current_run: ExperimentRun | None = None

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        run_date: str = "",
        llm_model: str = "",
        llm_available: bool = False,
        directions: list[str] | None = None,
    ) -> ExperimentRun:
        """Begin tracking a new pipeline run."""
        self._current_run = ExperimentRun(
            run_id=run_id,
            run_date=run_date or datetime.now().strftime("%Y-%m-%d"),
            started_at=datetime.now().isoformat(),
            llm_model=llm_model,
            llm_available=llm_available,
            directions_explored=directions or [],
        )
        return self._current_run

    def end_run(self, status: str = "success") -> None:
        """Finish the current run and persist."""
        if self._current_run is None:
            return
        self._current_run.ended_at = datetime.now().isoformat()
        self._current_run.status = status
        self._save()

    # ------------------------------------------------------------------
    # Factor attribution
    # ------------------------------------------------------------------

    def record_factor(
        self,
        factor_id: str,
        direction: str,
        source: str = "",
        agent_role: str = "",
        parent_factor_id: str = "",
        mutation_op: str = "",
        prompt_summary: str = "",
        knowledge_injected: list[str] | None = None,
        block_tree_summary: str = "",
        expression: str = "",
    ) -> None:
        """Record provenance for a newly created factor."""
        if self._current_run is None:
            return
        prov = FactorProvenance(
            factor_id=factor_id,
            direction=direction,
            source=source,
            run_id=self._current_run.run_id,
            agent_role=agent_role,
            parent_factor_id=parent_factor_id,
            mutation_op=mutation_op,
            prompt_summary=prompt_summary[:200],
            knowledge_injected=knowledge_injected or [],
            block_tree_summary=block_tree_summary[:100],
            expression=expression[:200],
            created_at=datetime.now().isoformat(),
        )
        self._current_run.factor_provenance.append(prov.to_dict())

    def record_evaluation(
        self,
        factor_id: str,
        rank_ic: float = 0.0,
        icir: float = 0.0,
        oos_ic: float = 0.0,
        screening_passed: bool = False,
        fail_reasons: list[str] | None = None,
        final_status: str = "draft",
    ) -> None:
        """Update evaluation results for a factor."""
        if self._current_run is None:
            return
        for prov in self._current_run.factor_provenance:
            if prov.get("factor_id") == factor_id:
                prov["rank_ic_mean"] = round(rank_ic, 6)
                prov["icir"] = round(icir, 4)
                prov["oos_ic"] = round(oos_ic, 6)
                prov["screening_passed"] = screening_passed
                prov["screening_fail_reasons"] = fail_reasons or []
                prov["final_status"] = final_status
                prov["evaluated_at"] = datetime.now().isoformat()
                return

    # ------------------------------------------------------------------
    # Stage results
    # ------------------------------------------------------------------

    def record_stage(self, stage_name: str, result: dict[str, Any]) -> None:
        """Record a stage's output summary."""
        if self._current_run is None:
            return
        # Store a lightweight summary
        summary = {}
        if stage_name == "evolution":
            summary = {
                "new_approved": result.get("new_approved", 0),
                "directions_count": len(result.get("directions", [])),
            }
        elif stage_name == "oos_validation":
            summary = {
                "total": result.get("total", 0),
                "passed": result.get("passed", 0),
                "failed": result.get("failed", 0),
            }
        elif stage_name == "governance":
            agent = result.get("agent_analysis", {})
            summary = {
                "regime": result.get("regime", {}).get("current", "unknown"),
                "crowding_severity": agent.get("crowding_analysis", {}).get("severity", "unknown"),
            }
        elif stage_name == "screening":
            summary = {
                "deliverable_count": result.get("deliverable_count", 0),
                "total_factors": result.get("total_factors", 0),
            }
        else:
            summary = {"status": result.get("status", "unknown")}

        self._current_run.stage_results[stage_name] = summary

    def set_agent_feedback(self, feedback: dict[str, Any]) -> None:
        """Record the agent feedback used for this run."""
        if self._current_run is None:
            return
        self._current_run.agent_feedback_used = feedback

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._current_run is None:
            return
        path = EXPERIMENTS_DIR / f"{self._current_run.run_id}.json"
        path.write_text(
            json.dumps(self._current_run.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("实验记录已保存: %s", path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @staticmethod
    def load_run(run_id: str) -> dict[str, Any] | None:
        """Load a specific experiment run."""
        path = EXPERIMENTS_DIR / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def list_runs(limit: int = 20) -> list[dict[str, Any]]:
        """List recent experiment runs."""
        runs = []
        for path in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                runs.append({
                    "run_id": data.get("run_id", ""),
                    "run_date": data.get("run_date", ""),
                    "status": data.get("status", ""),
                    "factors_approved": data.get("factors_approved", 0),
                    "factors_deliverable": data.get("factors_deliverable", 0),
                })
            except Exception:
                pass
            if len(runs) >= limit:
                break
        return runs

    @staticmethod
    def get_factor_history(factor_id: str) -> dict[str, Any] | None:
        """Find the provenance record for a specific factor."""
        for path in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for prov in data.get("factor_provenance", []):
                    if prov.get("factor_id") == factor_id:
                        return {"run_id": data["run_id"], **prov}
            except Exception:
                pass
        return None
