from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from enum import Enum
from typing import Any


class FactorDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    UNKNOWN = "unknown"


class AssetClass(str, Enum):
    """资产类别 —— 为多品种扩展提供类型锚点。"""
    A_SHARE_EQUITY = "a_share_equity"      # A股
    INDEX_FUTURE = "index_future"          # 股指期货
    COMMODITY_FUTURE = "commodity_future"  # 商品期货
    CONVERTIBLE_BOND = "convertible_bond"  # 可转债
    ETF = "etf"                            # ETF
    OPTION = "option"                      # 期权


class FactorStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    OBSERVE = "observe"          # under paper-tracking observation
    PAPER = "paper"              # paper-trading (OOS validated, no real money)
    PILOT = "pilot"              # small-position live trading
    LIVE = "live"                # full-position trading
    APPROVED = "approved"        # legacy alias → maps to PAPER
    REJECTED = "rejected"
    ARCHIVED = "archived"
    RETIRED = "retired"          # gracefully decommissioned prior live factor


class ResearchStage(str, Enum):
    IDEA = "idea"
    SPEC = "spec"
    COMPUTE = "compute"
    EVALUATE = "evaluate"
    VALIDATE = "validate"
    SCREEN = "screen"
    LIBRARY = "library"
    REVIEW = "review"


class AgentRole(str, Enum):
    CANDIDATE_GENERATOR = "candidate_generator"
    HYPOTHESIS_REVIEWER = "hypothesis_reviewer"
    SAFE_EXECUTOR = "safe_executor"
    FACTOR_EVALUATOR = "factor_evaluator"
    ROBUSTNESS_VALIDATOR = "robustness_validator"
    LIBRARY_GOVERNOR = "library_governor"


@dataclass(slots=True)
class FactorNode:
    node_type: str
    value: str | float | int | None = None
    children: list["FactorNode"] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorNode":
        return cls(
            node_type=str(payload.get("node_type", "")).strip(),
            value=payload.get("value"),
            children=[cls.from_dict(child) for child in payload.get("children", []) or []],
            params=dict(payload.get("params", {}) or {}),
        )


@dataclass(slots=True)
class FactorDependency:
    field_name: str
    source: str = "market_data"
    frequency: str = "1d"
    lookback: int | None = None
    required: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorDependency":
        return cls(
            field_name=str(payload.get("field_name", "")).strip(),
            source=str(payload.get("source", "market_data") or "market_data"),
            frequency=str(payload.get("frequency", "1d") or "1d"),
            lookback=payload.get("lookback"),
            required=bool(payload.get("required", True)),
            description=str(payload.get("description", "") or ""),
        )


@dataclass(slots=True)
class FactorPreprocessConfig:
    winsorize_method: str = "mad"
    winsorize_limit: float = 3.0
    normalization: str = "zscore"
    neutralization: list[str] = field(default_factory=lambda: ["industry", "market_cap"])
    fillna_method: str = "cross_section_median"
    liquidity_filter: str = "exclude_suspended_and_st"
    outlier_guard: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorPreprocessConfig":
        return cls(
            winsorize_method=str(payload.get("winsorize_method", "mad") or "mad"),
            winsorize_limit=float(payload.get("winsorize_limit", 3.0) or 3.0),
            normalization=str(payload.get("normalization", "zscore") or "zscore"),
            neutralization=list(payload.get("neutralization", ["industry", "market_cap"]) or []),
            fillna_method=str(payload.get("fillna_method", "cross_section_median") or "cross_section_median"),
            liquidity_filter=str(payload.get("liquidity_filter", "exclude_suspended_and_st") or "exclude_suspended_and_st"),
            outlier_guard=bool(payload.get("outlier_guard", True)),
        )


@dataclass(slots=True)
class FactorLabelSpec:
    return_horizon: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    return_type: str = "forward_excess_return"
    benchmark: str = "000300.SH"
    shift: int = 1
    tradability_delay: int = 1
    label_universe: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorLabelSpec":
        return cls(
            return_horizon=[int(item) for item in payload.get("return_horizon", [1, 5, 10, 20]) or [1, 5, 10, 20]],
            return_type=str(payload.get("return_type", "forward_excess_return") or "forward_excess_return"),
            benchmark=str(payload.get("benchmark", "000300.SH") or "000300.SH"),
            shift=int(payload.get("shift", 1) or 1),
            tradability_delay=int(payload.get("tradability_delay", 1) or 1),
            label_universe=payload.get("label_universe"),
        )


@dataclass(slots=True)
class FactorUniverseSpec:
    market: str = "cn_a"
    pool: str = "all_a"
    include_st: bool = False
    include_new_listing: bool = False
    min_listing_days: int = 120
    industry_schema: str = "sw_l1"
    rebalance_frequency: str = "1d"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorUniverseSpec":
        return cls(
            market=str(payload.get("market", "cn_a") or "cn_a"),
            pool=str(payload.get("pool", "all_a") or "all_a"),
            include_st=bool(payload.get("include_st", False)),
            include_new_listing=bool(payload.get("include_new_listing", False)),
            min_listing_days=int(payload.get("min_listing_days", 120) or 120),
            industry_schema=str(payload.get("industry_schema", "sw_l1") or "sw_l1"),
            rebalance_frequency=str(payload.get("rebalance_frequency", "1d") or "1d"),
        )


@dataclass(slots=True)
class SandboxPolicy:
    allowed_node_types: list[str] = field(
        default_factory=lambda: [
            "feature",
            "constant",
            "add",
            "sub",
            "mul",
            "div",
            "rank",
            "zscore",
            "delta",
            "lag",
            "mean",
            "std",
            "ts_rank",
            "min",
            "max",
            "clip",
        ]
    )
    max_tree_depth: int = 6
    max_children_per_node: int = 3
    max_lookback_window: int = 252
    forbid_raw_python_eval: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxPolicy":
        return cls(
            allowed_node_types=list(payload.get("allowed_node_types", cls().allowed_node_types) or cls().allowed_node_types),
            max_tree_depth=int(payload.get("max_tree_depth", 6) or 6),
            max_children_per_node=int(payload.get("max_children_per_node", 3) or 3),
            max_lookback_window=int(payload.get("max_lookback_window", 252) or 252),
            forbid_raw_python_eval=bool(payload.get("forbid_raw_python_eval", True)),
        )


@dataclass(slots=True)
class FactorExecutionSpec:
    compute_engine: str = "pandas"
    cache_key_template: str = "{factor_id}:{version}:{start}:{end}:{pool}"
    supports_cross_section: bool = True
    supports_time_series: bool = True
    chunk_size: int = 2000
    max_workers: int = 4
    sandbox_policy: SandboxPolicy = field(default_factory=SandboxPolicy)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorExecutionSpec":
        return cls(
            compute_engine=str(payload.get("compute_engine", "pandas") or "pandas"),
            cache_key_template=str(payload.get("cache_key_template", "{factor_id}:{version}:{start}:{end}:{pool}") or "{factor_id}:{version}:{start}:{end}:{pool}"),
            supports_cross_section=bool(payload.get("supports_cross_section", True)),
            supports_time_series=bool(payload.get("supports_time_series", True)),
            chunk_size=int(payload.get("chunk_size", 2000) or 2000),
            max_workers=int(payload.get("max_workers", 4) or 4),
            sandbox_policy=SandboxPolicy.from_dict(payload.get("sandbox_policy", {}) or {}),
        )


@dataclass(slots=True)
class RoleAssignment:
    role: AgentRole
    responsibility: str
    required_outputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FactorSpec:
    factor_id: str
    name: str
    version: str
    description: str
    hypothesis: str
    family: str
    direction: FactorDirection = FactorDirection.UNKNOWN
    expression: str = ""
    expression_tree: FactorNode | None = None
    template_name: str | None = None
    template_params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[FactorDependency] = field(default_factory=list)
    preprocess: FactorPreprocessConfig = field(default_factory=FactorPreprocessConfig)
    label: FactorLabelSpec = field(default_factory=FactorLabelSpec)
    universe: FactorUniverseSpec = field(default_factory=FactorUniverseSpec)
    execution: FactorExecutionSpec = field(default_factory=FactorExecutionSpec)
    tags: list[str] = field(default_factory=list)
    author: str = "agent"
    source: str = "factor_discovery_mode"
    created_from: str = "template"
    status: FactorStatus = FactorStatus.DRAFT
    asset_class: AssetClass = AssetClass.A_SHARE_EQUITY
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["status"] = self.status.value
        payload["asset_class"] = self.asset_class.value
        if self.expression_tree:
            payload["expression_tree"] = self.expression_tree.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorSpec":
        return cls(
            factor_id=str(payload.get("factor_id", "")).strip(),
            name=str(payload.get("name", "")).strip(),
            version=str(payload.get("version", "v1") or "v1"),
            description=str(payload.get("description", "") or ""),
            hypothesis=str(payload.get("hypothesis", "") or ""),
            family=str(payload.get("family", "generic") or "generic"),
            direction=FactorDirection(str(payload.get("direction", FactorDirection.UNKNOWN.value) or FactorDirection.UNKNOWN.value)),
            expression=str(payload.get("expression", "") or ""),
            expression_tree=FactorNode.from_dict(payload["expression_tree"]) if payload.get("expression_tree") else None,
            template_name=payload.get("template_name"),
            template_params=dict(payload.get("template_params", {}) or {}),
            dependencies=[FactorDependency.from_dict(item) for item in payload.get("dependencies", []) or []],
            preprocess=FactorPreprocessConfig.from_dict(payload.get("preprocess", {}) or {}),
            label=FactorLabelSpec.from_dict(payload.get("label", {}) or {}),
            universe=FactorUniverseSpec.from_dict(payload.get("universe", {}) or {}),
            execution=FactorExecutionSpec.from_dict(payload.get("execution", {}) or {}),
            tags=list(payload.get("tags", []) or []),
            author=str(payload.get("author", "agent") or "agent"),
            source=str(payload.get("source", "factor_discovery_mode") or "factor_discovery_mode"),
            created_from=str(payload.get("created_from", "template") or "template"),
            status=FactorStatus(str(payload.get("status", FactorStatus.DRAFT.value) or FactorStatus.DRAFT.value)),
            asset_class=AssetClass(str(payload.get("asset_class", AssetClass.A_SHARE_EQUITY.value) or AssetClass.A_SHARE_EQUITY.value)),
            notes=list(payload.get("notes", []) or []),
        )


@dataclass(slots=True)
class FactorResearchTask:
    task_id: str
    stage: ResearchStage
    title: str
    objective: str
    inputs: dict[str, Any] = field(default_factory=dict)
    output_key: str = ""
    required: bool = True
    owner_role: AgentRole | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage.value,
            "title": self.title,
            "objective": self.objective,
            "inputs": self.inputs,
            "output_key": self.output_key,
            "required": self.required,
            "owner_role": self.owner_role.value if self.owner_role else None,
        }


@dataclass(slots=True)
class FactorResearchPlan:
    plan_id: str
    title: str
    objective: str
    factor_spec: FactorSpec
    stages: list[FactorResearchTask] = field(default_factory=list)
    role_assignments: list[RoleAssignment] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    fallback_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "objective": self.objective,
            "factor_spec": self.factor_spec.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "role_assignments": [
                {
                    "role": item.role.value,
                    "responsibility": item.responsibility,
                    "required_outputs": item.required_outputs,
                }
                for item in self.role_assignments
            ],
            "acceptance_criteria": self.acceptance_criteria,
            "fallback_actions": self.fallback_actions,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ConstraintScorecard:
    originality_score: float | None = None
    hypothesis_alignment_score: float | None = None
    complexity_score: float | None = None
    safety_score: float | None = None


@dataclass(slots=True)
class FactorScorecard:
    ic_mean: float | None = None
    rank_ic_mean: float | None = None
    ic_ir: float | None = None
    quantile_monotonicity: float | None = None
    long_short_return: float | None = None
    turnover: float | None = None
    coverage: float | None = None
    decay: dict[str, float] = field(default_factory=dict)
    exposure_risk: dict[str, float] = field(default_factory=dict)
    correlation_to_library: dict[str, float] = field(default_factory=dict)
    stability_score: float | None = None
    tradability_score: float | None = None
    novelty_score: float | None = None
    composite_score: float | None = None


@dataclass(slots=True)
class FactorValidationResult:
    validation_name: str
    passed: bool
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SandboxValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    max_depth_seen: int = 0
    node_count: int = 0


@dataclass(slots=True)
class FactorExperience:
    experience_id: str
    factor_family: str
    pattern_type: str
    summary: str
    outcome: str
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorExperience":
        return cls(
            experience_id=str(payload.get("experience_id", "")).strip(),
            factor_family=str(payload.get("factor_family", "generic") or "generic"),
            pattern_type=str(payload.get("pattern_type", "observation") or "observation"),
            summary=str(payload.get("summary", "") or ""),
            outcome=str(payload.get("outcome", "observe") or "observe"),
            evidence=dict(payload.get("evidence", {}) or {}),
            tags=list(payload.get("tags", []) or []),
        )


@dataclass(slots=True)
class FactorMemorySnapshot:
    successful_patterns: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)
    regime_specific_findings: list[str] = field(default_factory=list)
    rejected_reason_clusters: list[str] = field(default_factory=list)
    related_experience_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FactorEvaluationReport:
    report_id: str
    factor_spec: FactorSpec
    scorecard: FactorScorecard
    constraint_scores: ConstraintScorecard = field(default_factory=ConstraintScorecard)
    validations: list[FactorValidationResult] = field(default_factory=list)
    memory_snapshot: FactorMemorySnapshot = field(default_factory=FactorMemorySnapshot)
    sandbox_validation: SandboxValidationResult | None = None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    decision: FactorStatus = FactorStatus.CANDIDATE
    decision_reason: str = ""
    next_actions: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "report_id": self.report_id,
                    "factor_spec": self.factor_spec.to_dict(),
                    "scorecard": asdict(self.scorecard),
                    "constraint_scores": asdict(self.constraint_scores),
                    "validations": [asdict(item) for item in self.validations],
                    "memory_snapshot": asdict(self.memory_snapshot),
                    "sandbox_validation": asdict(self.sandbox_validation) if self.sandbox_validation else None,
                    "strengths": self.strengths,
                    "weaknesses": self.weaknesses,
                    "decision": self.decision.value,
                    "decision_reason": self.decision_reason,
                    "next_actions": self.next_actions,
                    "artifacts": self.artifacts,
                },
                ensure_ascii=False,
                default=str,
            )
        )



@dataclass(slots=True)
class FactorLibraryEntry:
    factor_spec: FactorSpec
    latest_report: FactorEvaluationReport
    retention_reason: str
    review_cycle_days: int = 20
    dependencies: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    monitoring_metrics: list[str] = field(default_factory=lambda: ["rank_ic_mean", "ic_ir", "turnover", "originality_score"])
    experience_refs: list[str] = field(default_factory=list)
    panel_snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_spec": self.factor_spec.to_dict(),
            "latest_report": self.latest_report.to_dict(),
            "retention_reason": self.retention_reason,
            "review_cycle_days": self.review_cycle_days,
            "dependencies": self.dependencies,
            "supersedes": self.supersedes,
            "monitoring_metrics": self.monitoring_metrics,
            "experience_refs": self.experience_refs,
            "panel_snapshot_path": self.panel_snapshot_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorLibraryEntry":
        report_payload = dict(payload.get("latest_report", {}) or {})
        spec_payload = dict(payload.get("factor_spec", {}) or {})
        report = FactorEvaluationReport(
            report_id=str(report_payload.get("report_id", "")),
            factor_spec=FactorSpec.from_dict(report_payload.get("factor_spec", spec_payload)),
            scorecard=FactorScorecard(**dict(report_payload.get("scorecard", {}) or {})),
            constraint_scores=ConstraintScorecard(**dict(report_payload.get("constraint_scores", {}) or {})),
            validations=[FactorValidationResult(**item) for item in report_payload.get("validations", []) or []],
            memory_snapshot=FactorMemorySnapshot(**dict(report_payload.get("memory_snapshot", {}) or {})),
            sandbox_validation=SandboxValidationResult(**dict(report_payload.get("sandbox_validation", {}) or {})) if report_payload.get("sandbox_validation") else None,
            strengths=list(report_payload.get("strengths", []) or []),
            weaknesses=list(report_payload.get("weaknesses", []) or []),
            decision=FactorStatus(str(report_payload.get("decision", FactorStatus.CANDIDATE.value) or FactorStatus.CANDIDATE.value)),
            decision_reason=str(report_payload.get("decision_reason", "") or ""),
            next_actions=list(report_payload.get("next_actions", []) or []),
            artifacts=dict(report_payload.get("artifacts", {}) or {}),
        )
        return cls(
            factor_spec=FactorSpec.from_dict(spec_payload),
            latest_report=report,
            retention_reason=str(payload.get("retention_reason", "") or ""),
            review_cycle_days=int(payload.get("review_cycle_days", 20) or 20),
            dependencies=list(payload.get("dependencies", []) or []),
            supersedes=list(payload.get("supersedes", []) or []),
            monitoring_metrics=list(payload.get("monitoring_metrics", ["rank_ic_mean", "ic_ir", "turnover", "originality_score"]) or []),
            experience_refs=list(payload.get("experience_refs", []) or []),
            panel_snapshot_path=payload.get("panel_snapshot_path"),
        )


# ═══════════════════════════════════════════════════════════════════
# Factor Lifecycle Manager
# ═══════════════════════════════════════════════════════════════════

class FactorLifecycleManager:
    """因子生命周期管理器。

    状态流转（基于 Agent 决策而非硬规则）：
    DRAFT → CANDIDATE → OBSERVE → PAPER → PILOT → LIVE → RETIRED
                              ↓         ↓
                          REJECTED  ARCHIVED (from any state)

    升级条件（每个状态停留期收集证据）：
    - CANDIDATE → OBSERVE: 初始 IC > 0.015
    - OBSERVE → PAPER:    OOS IC 稳定 30+ 天
    - PAPER → PILOT:      纸交易 60 天 IC 稳定 + 无风控告警
    - PILOT → LIVE:       小仓位 90 天实盘验证 + 成本调整后 IC 仍 > 0
    - any → RETIRED:      衰减不可恢复 或 拥挤度过高
    """

    VALID_TRANSITIONS = {
        FactorStatus.DRAFT:     [FactorStatus.CANDIDATE, FactorStatus.REJECTED],
        FactorStatus.CANDIDATE: [FactorStatus.OBSERVE, FactorStatus.REJECTED],
        FactorStatus.OBSERVE:   [FactorStatus.PAPER, FactorStatus.REJECTED, FactorStatus.ARCHIVED],
        FactorStatus.PAPER:     [FactorStatus.PILOT, FactorStatus.OBSERVE, FactorStatus.ARCHIVED],
        FactorStatus.PILOT:     [FactorStatus.LIVE, FactorStatus.PAPER, FactorStatus.RETIRED],
        FactorStatus.LIVE:      [FactorStatus.RETIRED],
        FactorStatus.APPROVED:  [FactorStatus.PAPER, FactorStatus.OBSERVE, FactorStatus.ARCHIVED, FactorStatus.RETIRED],
        FactorStatus.REJECTED:  [FactorStatus.ARCHIVED],
        FactorStatus.ARCHIVED:  [],
        FactorStatus.RETIRED:   [FactorStatus.ARCHIVED],
    }

    def transition(
        self,
        current: FactorStatus,
        target: FactorStatus,
        evidence: dict | None = None,
    ) -> tuple[bool, str]:
        """尝试状态迁移。

        Args:
            current: 当前状态
            target: 目标状态
            evidence: 迁移证据 (e.g. {oob_ic: 0.03, days_stable: 45})

        Returns:
            (success, reason)
        """
        allowed = self.VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            return False, f"不允许从 {current.value} → {target.value}，允许: {[t.value for t in allowed]}"

        return True, f"状态迁移 {current.value} → {target.value} 已许可"

    def recommend(self, entry: "FactorLibraryEntry", days_since_eval: int = 0, oos_ic: float = 0.0,
                  cost_adj_ic: float = 0.0, crowding_score: float = 0.0,
                  regime_adj_ic: float = 0.0) -> tuple[FactorStatus, str]:
        """基于证据推荐下一步状态（Agent 决策辅助）。"""
        current = FactorStatus(str(entry.factor_spec.status))
        report = entry.latest_report

        ic = oos_ic or abs(getattr(getattr(report, 'scorecard', None), 'rank_ic_mean', 0) or 0)

        if current == FactorStatus.CANDIDATE:
            if ic > 0.015:
                return FactorStatus.OBSERVE, f"IC={ic:.4f} > 0.015，进入观察期"
            return current, f"IC={ic:.4f} 不足，保持候选"

        if current == FactorStatus.OBSERVE:
            if days_since_eval >= 30 and oos_ic > 0.01 and cost_adj_ic > 0.0:
                return FactorStatus.PAPER, f"OOS IC={oos_ic:.4f}, cost_adj_IC={cost_adj_ic:.4f}, 进入纸交易"
            if ic < 0.005 and days_since_eval > 60:
                return FactorStatus.REJECTED, f"长期低 IC={ic:.4f}，拒绝"
            return current, "观察中"

        if current == FactorStatus.PAPER:
            if days_since_eval >= 60 and oos_ic > 0.015 and crowding_score < 0.5:
                return FactorStatus.PILOT, f"纸交易稳定 (IC={oos_ic:.4f}, crowding={crowding_score:.2f})"
            if oos_ic < 0.0 and days_since_eval > 90:
                return FactorStatus.OBSERVE, "纸交易 IC 转负，回退观察"
            return current, "纸交易中"

        if current == FactorStatus.PILOT:
            if days_since_eval >= 90 and cost_adj_ic > 0.01 and regime_adj_ic > 0.0:
                return FactorStatus.LIVE, f"实盘验证通过 (cost_adj_IC={cost_adj_ic:.4f})"
            if cost_adj_ic < 0.0 and days_since_eval > 120:
                return FactorStatus.RETIRED, "成本调整后 IC 为负，建议退役"
            return current, "小仓验证中"

        if current == FactorStatus.LIVE:
            if ic < 0.005 and days_since_eval > 60:
                return FactorStatus.RETIRED, f"IC 衰减至 {ic:.4f}，建议退役"
            if crowding_score > 0.7:
                return FactorStatus.RETIRED, f"拥挤度过高 ({crowding_score:.2f})，建议退役"
            return current, "正常运行中"

        return current, "无推荐变更"
