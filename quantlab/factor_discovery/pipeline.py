from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    AgentRole,
    ConstraintScorecard,
    FactorDependency,
    FactorEvaluationReport,
    FactorExperience,
    FactorLibraryEntry,
    FactorResearchPlan,
    FactorResearchTask,
    FactorScorecard,
    FactorSpec,
    FactorStatus,
    FactorValidationResult,
    ResearchStage,
    RoleAssignment,
)
from .runtime import FactorExperienceMemory, PersistentFactorStore, SafeFactorExecutor


class FactorDiscoveryOrchestrator:
    """面向成熟量化因子发掘的闭环编排器。"""

    def __init__(
        self,
        memory: FactorExperienceMemory | None = None,
        executor: SafeFactorExecutor | None = None,
        store: PersistentFactorStore | None = None,
    ) -> None:
        self.store = store or PersistentFactorStore()
        self.memory = memory or self.store.build_memory() or self._build_default_memory()
        if not self.memory.export():
            self.memory = self._build_default_memory()
            for item in self.memory.export():
                self.store.append_experience(FactorExperience.from_dict(item))
        self.executor = executor or SafeFactorExecutor()

    def _build_default_memory(self) -> FactorExperienceMemory:
        return FactorExperienceMemory(
            experiences=[
                FactorExperience(
                    experience_id="exp_quality_success_001",
                    factor_family="quality",
                    pattern_type="successful_pattern",
                    summary="质量类因子在行业中性化后更稳定，极端行情下仍需叠加波动约束。",
                    outcome="success",
                    tags=["quality", "industry_neutral"],
                ),
                FactorExperience(
                    experience_id="exp_momentum_failure_001",
                    factor_family="momentum",
                    pattern_type="rejected_reason",
                    summary="高换手动量因子即便 IC 较高，也常因成本与拥挤暴露被淘汰。",
                    outcome="failure",
                    tags=["momentum", "turnover"],
                ),
                FactorExperience(
                    experience_id="exp_regime_quality_001",
                    factor_family="quality",
                    pattern_type="regime_finding",
                    summary="高波动阶段若没有行业和市值双重中性化，质量因子稳定性明显下降。",
                    outcome="success",
                    tags=["quality", "high_vol"],
                ),
            ]
        )

    def build_research_plan(self, spec: FactorSpec) -> FactorResearchPlan:
        tasks = [
            FactorResearchTask(
                task_id="spec_review",
                stage=ResearchStage.SPEC,
                title="检查因子定义、假设与依赖",
                objective="确认候选因子不仅能算，还和研究假设一致。",
                output_key="spec_review",
                owner_role=AgentRole.HYPOTHESIS_REVIEWER,
            ),
            FactorResearchTask(
                task_id="sandbox_guard",
                stage=ResearchStage.COMPUTE,
                title="执行安全校验",
                objective="对 expression_tree 做 AST 白名单、深度和窗口限制检查。",
                output_key="sandbox_validation",
                owner_role=AgentRole.SAFE_EXECUTOR,
            ),
            FactorResearchTask(
                task_id="compute_factor",
                stage=ResearchStage.COMPUTE,
                title="计算因子面板并完成预处理",
                objective="按统一链路完成缺失值处理、去极值、标准化和中性化。",
                output_key="factor_panel",
                owner_role=AgentRole.SAFE_EXECUTOR,
            ),
            FactorResearchTask(
                task_id="evaluate_single_factor",
                stage=ResearchStage.EVALUATE,
                title="输出单因子评分卡与约束评分",
                objective="计算收益指标，同时给出原创性、假设对齐与复杂度评分。",
                output_key="scorecard",
                owner_role=AgentRole.FACTOR_EVALUATOR,
            ),
            FactorResearchTask(
                task_id="validate_robustness",
                stage=ResearchStage.VALIDATE,
                title="做多阶段稳健性验证",
                objective="覆盖 train/test、walk-forward、市场阶段、股票池和参数扰动。",
                output_key="validations",
                owner_role=AgentRole.ROBUSTNESS_VALIDATOR,
            ),
            FactorResearchTask(
                task_id="screen_library",
                stage=ResearchStage.SCREEN,
                title="结合经验记忆和因子库做筛选决策",
                objective="判断是否保留、观察、淘汰，避免重复收录高度相关因子。",
                output_key="screening_decision",
                owner_role=AgentRole.LIBRARY_GOVERNOR,
            ),
            FactorResearchTask(
                task_id="persist_library",
                stage=ResearchStage.LIBRARY,
                title="沉淀因子库记录与复验计划",
                objective="写入因子元数据、保留理由、监控指标和经验引用。",
                output_key="library_entry",
                owner_role=AgentRole.LIBRARY_GOVERNOR,
            ),
        ]
        role_assignments = [
            RoleAssignment(role=AgentRole.CANDIDATE_GENERATOR, responsibility="提出候选因子、模板来源和参数设定。", required_outputs=["factor_spec"]),
            RoleAssignment(role=AgentRole.HYPOTHESIS_REVIEWER, responsibility="检查因子表达是否与假设一致，避免无解释的堆算子。", required_outputs=["spec_review"]),
            RoleAssignment(role=AgentRole.SAFE_EXECUTOR, responsibility="保证表达树只走安全白名单算子，并控制复杂度与窗口风险。", required_outputs=["sandbox_validation", "factor_panel"]),
            RoleAssignment(role=AgentRole.FACTOR_EVALUATOR, responsibility="输出收益、稳定性、可交易性和约束评分卡。", required_outputs=["scorecard"]),
            RoleAssignment(role=AgentRole.ROBUSTNESS_VALIDATOR, responsibility="做样本外、walk-forward、市场阶段与参数扰动验证。", required_outputs=["validations"]),
            RoleAssignment(role=AgentRole.LIBRARY_GOVERNOR, responsibility="基于经验记忆和冗余比较给出保留/观察/淘汰决策。", required_outputs=["screening_decision", "library_entry"]),
        ]
        return FactorResearchPlan(
            plan_id=f"plan::{spec.factor_id}::{spec.version}",
            title=f"{spec.name} 因子研究计划",
            objective="完成成熟量化因子研究闭环，从候选定义走到安全执行、评估、验证、筛选和入库。",
            factor_spec=spec,
            stages=tasks,
            role_assignments=role_assignments,
            acceptance_criteria=[
                "核心依赖字段完整且可计算",
                "表达树通过安全校验，不使用越界算子或过长窗口",
                "单因子评分卡包含 IC/RankIC/分层收益/换手/暴露/原创性/复杂度",
                "至少完成时间切片、样本外与 walk-forward 三类验证",
                "给出明确的保留/观察/淘汰决策以及因子库动作",
            ],
            fallback_actions=[
                "若安全校验失败，先收缩算子白名单或简化表达树",
                "若原创性不足，转为已有因子扩展分支而不是新入库",
                "若稳定性不足，增加更长窗口或补做高波动阶段中性化",
            ],
            metadata={
                "agent_mode": "factor_discovery_mode",
                "pipeline_version": "v3",
                "target_horizons": spec.label.return_horizon,
                "preprocess": asdict(spec.preprocess),
                "roles": [item.role.value for item in role_assignments],
            },
        )

    def _build_constraint_scores(self, spec: FactorSpec, artifacts: dict[str, Any], max_library_corr: float) -> ConstraintScorecard:
        tree_depth = artifacts.get("tree_depth", self._estimate_tree_depth(spec.expression_tree))
        originality = max(0.0, round(1.0 - max_library_corr, 4))
        hypothesis_alignment = self._score_hypothesis_alignment(spec, artifacts)
        complexity_penalty = min(tree_depth / max(spec.execution.sandbox_policy.max_tree_depth, 1), 1.0)
        complexity_score = round(1.0 - complexity_penalty * 0.55, 3)
        safety_score = 0.95 if spec.expression_tree else 0.0
        return ConstraintScorecard(
            originality_score=originality,
            hypothesis_alignment_score=hypothesis_alignment,
            complexity_score=complexity_score,
            safety_score=safety_score,
        )

    def _score_hypothesis_alignment(self, spec: FactorSpec, artifacts: dict[str, Any]) -> float:
        if not spec.hypothesis:
            return 0.45
        hypothesis = spec.hypothesis.lower()
        family = spec.family.lower()
        score = 0.6
        if family and family in hypothesis:
            score += 0.2
        monotonicity = float(artifacts.get("quantile_monotonicity", 0.0) or 0.0)
        score += min(0.2, max(0.0, monotonicity) * 0.2)
        return round(min(score, 1.0), 3)

    def _estimate_tree_depth(self, node: Any) -> int:
        if node is None:
            return 0
        if not getattr(node, "children", None):
            return 1
        return 1 + max(self._estimate_tree_depth(child) for child in node.children)

    def _build_validations(self, spec: FactorSpec, metrics: dict[str, Any], sandbox_validation: Any) -> list[FactorValidationResult]:
        decay = dict(metrics.get("decay", {}) or {})
        validations = [
            FactorValidationResult(
                validation_name="sandbox_guard",
                passed=sandbox_validation.passed,
                summary="表达树安全边界检查完成。",
                metrics={"node_count": sandbox_validation.node_count, "max_depth_seen": sandbox_validation.max_depth_seen},
                warnings=sandbox_validation.reasons,
            ),
            FactorValidationResult(
                validation_name="coverage_check",
                passed=float(metrics.get("coverage", 0.0) or 0.0) >= 0.7,
                summary="检查横截面覆盖率是否达到可研究标准。",
                metrics={"coverage": float(metrics.get("coverage", 0.0) or 0.0)},
            ),
            FactorValidationResult(
                validation_name="monotonicity_check",
                passed=float(metrics.get("quantile_monotonicity", 0.0) or 0.0) >= 0.55,
                summary="检查分层收益是否具备单调性。",
                metrics={"quantile_monotonicity": float(metrics.get("quantile_monotonicity", 0.0) or 0.0)},
            ),
            FactorValidationResult(
                validation_name="decay_profile",
                passed=bool(decay) and max(decay.values()) >= min(decay.values()),
                summary="检查不同持有期下的衰减曲线是否合理。",
                metrics=decay,
                warnings=["长周期衰减明显"] if decay and min(decay.values()) < 0 else [],
            ),
        ]
        if spec.preprocess.neutralization:
            validations.append(
                FactorValidationResult(
                    validation_name="neutralization_check",
                    passed=max(abs(float(value)) for value in dict(metrics.get("exposure_risk", {}) or {"dummy": 0.0}).values()) < 0.35,
                    summary="检查行业 / 市值暴露是否被压缩到合理范围。",
                    metrics=dict(metrics.get("exposure_risk", {}) or {}),
                )
            )
        return validations

    def _build_scorecard(self, metrics: dict[str, Any], constraint_scores: ConstraintScorecard) -> FactorScorecard:
        return FactorScorecard(
            ic_mean=float(metrics.get("ic_mean", 0.0) or 0.0),
            rank_ic_mean=float(metrics.get("rank_ic_mean", 0.0) or 0.0),
            ic_ir=float(metrics.get("ic_ir", 0.0) or 0.0),
            quantile_monotonicity=float(metrics.get("quantile_monotonicity", 0.0) or 0.0),
            long_short_return=float(metrics.get("long_short_return", 0.0) or 0.0),
            turnover=float(metrics.get("turnover", 0.0) or 0.0),
            coverage=float(metrics.get("coverage", 0.0) or 0.0),
            decay=dict(metrics.get("decay", {}) or {}),
            exposure_risk=dict(metrics.get("exposure_risk", {}) or {}),
            correlation_to_library=dict(metrics.get("correlation_to_library", {}) or {}),
            stability_score=float(metrics.get("stability_score", 0.0) or 0.0),
            tradability_score=float(metrics.get("tradability_score", 0.0) or 0.0),
            novelty_score=constraint_scores.originality_score,
            composite_score=float(metrics.get("composite_score", 0.0) or 0.0),
        )

    def evaluate(self, spec: FactorSpec, market_df: pd.DataFrame) -> FactorEvaluationReport:
        sandbox_validation = self.executor.validate_spec(spec)
        computed = self.executor.execute(spec, market_df)
        factor_panel = computed["factor_panel"]
        correlation_map, max_library_corr = self.store.summarize_library_overlap(factor_panel)
        metrics = evaluate_factor_metrics(factor_panel, spec)
        metrics["correlation_to_library"] = correlation_map
        constraint_scores = self._build_constraint_scores(spec, {**computed, **metrics}, max_library_corr)
        memory_snapshot = self.memory.summarize_for_factor(spec)
        metrics["novelty_score"] = constraint_scores.originality_score
        metrics["composite_score"] = _composite_score(metrics, constraint_scores)
        scorecard = self._build_scorecard(metrics, constraint_scores)
        validations = self._build_validations(spec, metrics, sandbox_validation)

        decision = FactorStatus.APPROVED
        reason = "核心指标达标，真实横截面评估通过，可进入正式因子库。"
        if not sandbox_validation.passed:
            decision = FactorStatus.REJECTED
            reason = "表达树未通过安全校验，禁止进入计算与入库阶段。"
        elif scorecard.coverage is not None and scorecard.coverage < 0.7:
            decision = FactorStatus.REJECTED
            reason = "覆盖率不足，无法进入成熟因子库。"
        elif (constraint_scores.originality_score or 0.0) < 0.45:
            decision = FactorStatus.OBSERVE
            reason = "与现有因子库相关性过高，先放入观察区而不是正式入库。"
        elif (scorecard.ic_ir or 0.0) < 0.25:
            decision = FactorStatus.OBSERVE
            reason = "ICIR 偏弱，先继续观察并补做更多阶段验证。"
        elif (scorecard.quantile_monotonicity or 0.0) < 0.55:
            decision = FactorStatus.OBSERVE
            reason = "分层收益单调性不足，需继续优化表达式或预处理链路。"

        return FactorEvaluationReport(
            report_id=f"report::{spec.factor_id}::{spec.version}",
            factor_spec=spec,
            scorecard=scorecard,
            constraint_scores=constraint_scores,
            validations=validations,
            memory_snapshot=memory_snapshot,
            sandbox_validation=sandbox_validation,
            strengths=_build_strengths(metrics, memory_snapshot),
            weaknesses=_build_weaknesses(metrics, constraint_scores, validations),
            decision=decision,
            decision_reason=reason,
            next_actions=[
                "复验更长持有期下的衰减曲线",
                "检查是否需要额外行业 / 市值中性化",
                "把新结论沉淀到长期经验库与因子 registry",
            ],
            artifacts={
                "computed_artifacts": {**computed, **metrics, "factor_panel_preview": factor_panel.head(20).to_dict(orient="records")},
                "industry_standard_checks": [
                    "winsorize",
                    "zscore",
                    "industry_neutralize",
                    "rank_ic",
                    "quantile_return",
                    "turnover",
                    "library_correlation",
                    "originality_constraint",
                    "hypothesis_alignment_constraint",
                    "complexity_constraint",
                    "sandbox_guard",
                ],
            },
        )

    def build_library_entry(self, report: FactorEvaluationReport, factor_panel: pd.DataFrame | None = None) -> FactorLibraryEntry:
        retention_reason = {
            FactorStatus.APPROVED: "单因子研究闭环通过，允许进入正式因子库并纳入监控。",
            FactorStatus.OBSERVE: "核心信号存在潜力，但需要更多验证或先降低复杂度后再升级。",
            FactorStatus.REJECTED: "当前不满足安全或入库标准，仅保留研究档案。",
        }.get(report.decision, "保留研究档案，等待下一轮评估。")
        entry = FactorLibraryEntry(
            factor_spec=report.factor_spec,
            latest_report=report,
            retention_reason=retention_reason,
            dependencies=[item.field_name for item in report.factor_spec.dependencies],
            supersedes=[],
            experience_refs=report.memory_snapshot.related_experience_ids,
        )
        self.store.upsert_library_entry(entry, factor_panel=factor_panel)
        return entry

    def run_closed_loop(self, spec: FactorSpec, market_df: pd.DataFrame) -> dict[str, Any]:
        plan = self.build_research_plan(spec)
        report = self.evaluate(spec, market_df)
        factor_panel_preview = pd.DataFrame(report.artifacts["computed_artifacts"].get("factor_panel_preview", []))
        library_entry = self.build_library_entry(report, factor_panel=factor_panel_preview if not factor_panel_preview.empty else None)
        experience = FactorExperience(
            experience_id=f"exp::{spec.factor_id}::{spec.version}",
            factor_family=spec.family,
            pattern_type="approved_factor" if report.decision == FactorStatus.APPROVED else "research_observation",
            summary=report.decision_reason,
            outcome="success" if report.decision == FactorStatus.APPROVED else "failure" if report.decision == FactorStatus.REJECTED else "observe",
            evidence={
                "report_id": report.report_id,
                "ic_mean": report.scorecard.ic_mean,
                "rank_ic_mean": report.scorecard.rank_ic_mean,
                "ic_ir": report.scorecard.ic_ir,
            },
            tags=list({spec.family, *spec.tags, report.decision.value}),
        )
        self.store.append_experience(experience)
        self.memory.add_experience(experience)
        return {
            "plan": plan.to_dict(),
            "report": report.to_dict(),
            "library_entry": library_entry.to_dict(),
            "memory_registry": self.memory.export(),
        }


def evaluate_factor_metrics(factor_panel: pd.DataFrame, spec: FactorSpec) -> dict[str, Any]:
    panel = factor_panel.copy()
    if panel.empty:
        raise ValueError("factor_panel 为空，无法评估。")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)
    close_col = "close" if "close" in panel.columns else None
    if close_col is None:
        raise ValueError("factor_panel 缺少 close 字段，无法计算 forward return。")
    horizon = int(spec.label.return_horizon[0] if spec.label.return_horizon else 1)
    shift = int(spec.label.shift or 1)
    panel["forward_return"] = panel.groupby("asset")[close_col].shift(-(horizon + shift - 1)) / panel.groupby("asset")[close_col].shift(-shift) - 1.0
    panel = panel.dropna(subset=["factor_value", "forward_return"]).reset_index(drop=True)
    if panel.empty:
        raise ValueError("缺少可用于评估的因子值或 forward return。")

    ic_series = panel.groupby("date", sort=False).apply(lambda group: _safe_corr(group["factor_value"], group["forward_return"]))
    rank_ic_series = panel.groupby("date", sort=False).apply(lambda group: _safe_corr(group["factor_value"].rank(pct=True), group["forward_return"].rank(pct=True)))
    ic_mean = float(ic_series.mean()) if not ic_series.dropna().empty else 0.0
    rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.dropna().empty else 0.0
    ic_std = float(ic_series.std(ddof=0)) if not ic_series.dropna().empty else 0.0
    ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0

    quantile_return, monotonicity = _evaluate_quantile(panel)
    turnover = _evaluate_turnover(panel)
    coverage = float(panel["factor_value"].notna().mean()) if not panel.empty else 0.0
    decay = _evaluate_decay(panel, spec.label.return_horizon)
    exposure_risk = _evaluate_exposure(panel)
    stability_score = _evaluate_stability(ic_series, rank_ic_series, decay)
    tradability_score = round(max(0.0, 1.0 - min(turnover, 1.2) / 1.2), 4)
    long_short_return = quantile_return.get("q5_minus_q1", 0.0)

    return {
        "ic_mean": round(ic_mean, 6),
        "rank_ic_mean": round(rank_ic_mean, 6),
        "ic_ir": round(ic_ir, 6),
        "quantile_monotonicity": round(monotonicity, 6),
        "long_short_return": round(float(long_short_return), 6),
        "turnover": round(float(turnover), 6),
        "coverage": round(float(coverage), 6),
        "decay": {key: round(float(value), 6) for key, value in decay.items()},
        "exposure_risk": {key: round(float(value), 6) for key, value in exposure_risk.items()},
        "stability_score": round(float(stability_score), 6),
        "tradability_score": round(float(tradability_score), 6),
        "quantile_returns": {key: round(float(value), 6) for key, value in quantile_return.items()},
    }


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return 0.0
    left_values = pd.to_numeric(joined.iloc[:, 0], errors="coerce")
    right_values = pd.to_numeric(joined.iloc[:, 1], errors="coerce")
    valid = pd.concat([left_values, right_values], axis=1).dropna()
    if len(valid) < 3:
        return 0.0
    if valid.iloc[:, 0].nunique(dropna=True) <= 1 or valid.iloc[:, 1].nunique(dropna=True) <= 1:
        return 0.0
    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
    return float(corr) if pd.notna(corr) else 0.0



def _evaluate_quantile(panel: pd.DataFrame, buckets: int = 5) -> tuple[dict[str, float], float]:
    bucket_rows: list[dict[str, Any]] = []
    for _, group in panel.groupby("date", sort=False):
        if group["factor_value"].nunique() < buckets:
            continue
        ranked = group[["asset", "factor_value", "forward_return"]].copy()
        ranked["bucket"] = pd.qcut(ranked["factor_value"], buckets, labels=False, duplicates="drop")
        bucket_summary = ranked.groupby("bucket")["forward_return"].mean().to_dict()
        bucket_rows.append(bucket_summary)
    if not bucket_rows:
        return {"q5_minus_q1": 0.0}, 0.0
    bucket_df = pd.DataFrame(bucket_rows).fillna(0.0)
    averaged = {f"q{int(column) + 1}": float(bucket_df[column].mean()) for column in bucket_df.columns}
    ordered_values = [averaged.get(f"q{index}", 0.0) for index in range(1, buckets + 1)]
    monotonic_steps = sum(1 for idx in range(len(ordered_values) - 1) if ordered_values[idx] <= ordered_values[idx + 1])
    monotonicity = monotonic_steps / max(1, buckets - 1)
    averaged["q5_minus_q1"] = ordered_values[-1] - ordered_values[0]
    return averaged, monotonicity


def _evaluate_turnover(panel: pd.DataFrame, buckets: int = 5) -> float:
    leaders: list[set[str]] = []
    for _, group in panel.groupby("date", sort=False):
        if group["factor_value"].nunique() < buckets:
            continue
        ranked = group[["asset", "factor_value"]].copy().sort_values("factor_value", ascending=False)
        leaders.append(set(ranked.head(max(1, len(ranked) // buckets))["asset"].astype(str)))
    if len(leaders) < 2:
        return 0.0
    turnovers = []
    for previous, current in zip(leaders[:-1], leaders[1:]):
        if not previous:
            continue
        overlap = len(previous & current) / len(previous)
        turnovers.append(1.0 - overlap)
    return float(np.mean(turnovers)) if turnovers else 0.0


def _evaluate_decay(panel: pd.DataFrame, horizons: list[int]) -> dict[str, float]:
    decay: dict[str, float] = {}
    for horizon in horizons:
        shifted = panel.copy()
        shifted["forward_return_h"] = shifted.groupby("asset")["close"].shift(-horizon) / shifted["close"] - 1.0
        shifted = shifted.dropna(subset=["forward_return_h", "factor_value"])
        decay[f"{horizon}d"] = _safe_corr(shifted["factor_value"], shifted["forward_return_h"])
    return decay


def _evaluate_exposure(panel: pd.DataFrame) -> dict[str, float]:
    exposure: dict[str, float] = {}
    if "market_cap" in panel.columns:
        exposure["market_cap"] = abs(_safe_corr(panel["factor_value"], np.log(panel["market_cap"].clip(lower=1.0))))
    if "industry" in panel.columns:
        industry_mean = panel.groupby("industry")["factor_value"].mean().abs()
        exposure["industry"] = float(industry_mean.max()) if not industry_mean.empty else 0.0
    return exposure or {"industry": 0.0, "market_cap": 0.0}


def _evaluate_stability(ic_series: pd.Series, rank_ic_series: pd.Series, decay: dict[str, float]) -> float:
    ic_positive_ratio = float((ic_series > 0).mean()) if len(ic_series) else 0.0
    rank_positive_ratio = float((rank_ic_series > 0).mean()) if len(rank_ic_series) else 0.0
    decay_ratio = 0.0
    if decay:
        values = list(decay.values())
        if values[0] != 0:
            decay_ratio = max(min(values[-1] / values[0], 1.0), -1.0)
    return round(max(0.0, min(1.0, 0.45 * ic_positive_ratio + 0.35 * rank_positive_ratio + 0.2 * max(decay_ratio, 0.0))), 6)


def _composite_score(metrics: dict[str, Any], constraint_scores: ConstraintScorecard) -> float:
    score = (
        0.2 * max(float(metrics.get("ic_mean", 0.0) or 0.0) * 10, 0.0)
        + 0.2 * max(float(metrics.get("rank_ic_mean", 0.0) or 0.0) * 10, 0.0)
        + 0.15 * max(float(metrics.get("ic_ir", 0.0) or 0.0), 0.0)
        + 0.15 * max(float(metrics.get("quantile_monotonicity", 0.0) or 0.0), 0.0)
        + 0.1 * max(float(metrics.get("stability_score", 0.0) or 0.0), 0.0)
        + 0.1 * max(float(metrics.get("tradability_score", 0.0) or 0.0), 0.0)
        + 0.1 * max(float(constraint_scores.originality_score or 0.0), 0.0)
    )
    return round(min(score, 1.0), 6)


def _build_strengths(metrics: dict[str, Any], memory_snapshot: Any) -> list[str]:
    strengths: list[str] = []
    if float(metrics.get("rank_ic_mean", 0.0) or 0.0) > 0.02:
        strengths.append("RankIC 为正，横截面排序能力可用。")
    if float(metrics.get("quantile_monotonicity", 0.0) or 0.0) >= 0.6:
        strengths.append("分层收益具备较好的单调性。")
    if memory_snapshot.successful_patterns:
        strengths.append("当前因子家族在经验库中存在可复用的成功模式。")
    return strengths or ["表达树已可在白名单约束下真实计算。"]


def _build_weaknesses(metrics: dict[str, Any], constraints: ConstraintScorecard, validations: list[FactorValidationResult]) -> list[str]:
    weaknesses: list[str] = []
    if float(metrics.get("turnover", 0.0) or 0.0) > 0.6:
        weaknesses.append("换手率偏高，实际交易成本压力较大。")
    if any(not item.passed for item in validations):
        weaknesses.append("部分验证未达标，需要继续补充稳健性验证。")
    if float(constraints.originality_score or 0.0) < 0.5:
        weaknesses.append("与现有因子库重合度偏高，原创性一般。")
    return weaknesses or ["当前暂无明显弱项，但仍需持续监控衰减。"]


def build_default_factor_pipeline(spec: FactorSpec, market_df: pd.DataFrame) -> dict[str, Any]:
    orchestrator = FactorDiscoveryOrchestrator()
    return orchestrator.run_closed_loop(spec, market_df=market_df)
