from __future__ import annotations

from typing import Any

from quantlab.research.models import ResearchPlan


class ResearchDecisionEvaluator:
    def __init__(self) -> None:
        self.phase_thresholds = {
            "plan": 0.7,
            "execution": 0.75,
            "conclusion": 0.75,
            "task": 0.65,
        }

    def evaluate_plan(self, research_plan: ResearchPlan) -> dict[str, Any]:
        task_types = [task.task_type for task in research_plan.tasks]
        is_factor_plan = "factor_discovery" in task_types or research_plan.metadata.get("agent_mode") == "factor_discovery_mode"
        if is_factor_plan:
            score_components = {
                "goal_clarity": 1.0 if (research_plan.goal or "").strip() else 0.0,
                "task_coverage": 1.0 if "factor_discovery" in task_types else 0.0,
                "rationale_completeness": round(min(1.0, len(research_plan.rationale) / 2), 4),
                "workflow_integrity": 1.0 if "factor_discovery" in task_types else 0.0,
                "portfolio_awareness": 1.0,
            }
        else:
            required_chain = {
                "single_backtest",
                "grid_search",
                "train_test_validation",
                "walk_forward_validation",
            }
            score_components = {
                "goal_clarity": 1.0 if (research_plan.goal or "").strip() else 0.0,
                "task_coverage": round(min(1.0, len(task_types) / 4), 4),
                "rationale_completeness": round(min(1.0, len(research_plan.rationale) / 3), 4),
                "workflow_integrity": 1.0 if required_chain.issubset(set(task_types)) else round(min(1.0, len(set(task_types) & required_chain) / 4), 4),
                "portfolio_awareness": 1.0 if any(task in {"multi_strategy_compare", "portfolio_construction_review"} for task in task_types) else 0.3,
            }
        score = round(
            score_components["goal_clarity"] * 0.2
            + score_components["task_coverage"] * 0.2
            + score_components["rationale_completeness"] * 0.15
            + score_components["workflow_integrity"] * 0.3
            + score_components["portfolio_awareness"] * 0.15,
            4,
        )
        return self._build_assessment(
            phase="plan",
            score=score,
            score_components=score_components,
            task_types=task_types,
        )

    def evaluate_execution(self, autopilot_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not autopilot_results:
            return self._build_assessment(
                phase="execution",
                score=0.2,
                score_components={
                    "executed_steps": 0.0,
                    "history_traceability": 0.0,
                    "metrics_availability": 0.0,
                    "strategy_scope": 0.0,
                },
                task_types=[],
            )

        executed_steps = len(autopilot_results)
        history_count = 0
        metrics_count = 0
        strategies = set()
        task_types: list[str] = []
        for item in autopilot_results:
            task_type = str(item.get("task_type", ""))
            task_types.append(task_type)
            strategies.add(str(item.get("strategy_name", "")))
            summary = item.get("summary", {}) or {}
            if summary.get("history_path"):
                history_count += 1
            if any(summary.get(key) for key in ["metrics", "best_metrics", "average_metrics", "research_summary", "credibility_assessment", "report", "scorecard"]):
                metrics_count += 1

        is_factor_execution = "factor_discovery" in task_types
        score_components = {
            "executed_steps": round(min(1.0, executed_steps / (1 if is_factor_execution else 4)), 4),
            "history_traceability": round(history_count / executed_steps, 4),
            "metrics_availability": round(metrics_count / executed_steps, 4),
            "strategy_scope": 1.0 if is_factor_execution or len(strategies) >= 2 else 0.55,
        }
        score = round(
            score_components["executed_steps"] * 0.35
            + score_components["history_traceability"] * 0.25
            + score_components["metrics_availability"] * 0.25
            + score_components["strategy_scope"] * 0.15,
            4,
        )
        return self._build_assessment(
            phase="execution",
            score=score,
            score_components=score_components,
            task_types=task_types,
        )

    def evaluate_conclusion(
        self,
        research_plan: ResearchPlan,
        tool_logs: list[dict[str, Any]],
        autopilot_results: list[dict[str, Any]],
        knowledge_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_types = [task.task_type for task in research_plan.tasks]
        is_factor_plan = "factor_discovery" in task_types or research_plan.metadata.get("agent_mode") == "factor_discovery_mode"
        score_components = {
            "plan_alignment": 1.0 if task_types else 0.0,
            "execution_support": 1.0 if autopilot_results else (0.5 if tool_logs else 0.2),
            "evidence_support": round(min(1.0, (len(tool_logs) + len(autopilot_results)) / 4), 4),
            "knowledge_support": 1.0 if knowledge_hits else 0.45,
            "portfolio_scope": 1.0 if is_factor_plan or any(task in {"multi_strategy_compare", "portfolio_construction_review"} for task in task_types) else 0.5,
        }
        score = round(
            score_components["plan_alignment"] * 0.2
            + score_components["execution_support"] * 0.3
            + score_components["evidence_support"] * 0.2
            + score_components["knowledge_support"] * 0.15
            + score_components["portfolio_scope"] * 0.15,
            4,
        )
        return self._build_assessment(
            phase="conclusion",
            score=score,
            score_components=score_components,
            task_types=task_types,
        )

    def evaluate_task_result(self, task_type: str, payload: dict[str, Any], history_path: str | None = None) -> dict[str, Any]:
        metrics_container = self._extract_metrics_container(payload)
        score_components = {
            "metrics_presence": 1.0 if metrics_container else 0.35,
            "history_traceability": 1.0 if history_path else 0.45,
            "result_completeness": round(min(1.0, len(payload) / 6), 4),
            "portfolio_depth": 1.0 if task_type in {"multi_strategy_compare", "portfolio_construction_review", "factor_discovery"} else 0.6,
        }
        score = round(
            score_components["metrics_presence"] * 0.35
            + score_components["history_traceability"] * 0.2
            + score_components["result_completeness"] * 0.25
            + score_components["portfolio_depth"] * 0.2,
            4,
        )
        return self._build_assessment(
            phase="task",
            score=score,
            score_components=score_components,
            task_types=[task_type],
        )

    def _extract_metrics_container(self, payload: dict[str, Any]) -> dict[str, Any]:
        for key in [
            "metrics",
            "best_metrics",
            "test_metrics",
            "average_metrics",
            "research_summary",
            "portfolio_assessment",
            "scorecard",
        ]:
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        report = payload.get("report")
        if isinstance(report, dict):
            scorecard = report.get("scorecard")
            if isinstance(scorecard, dict) and scorecard:
                return scorecard
        return {}

    def _build_assessment(
        self,
        phase: str,
        score: float,
        score_components: dict[str, float],
        task_types: list[str],
    ) -> dict[str, Any]:
        threshold = self.phase_thresholds.get(phase, 0.7)
        label = self._build_label(score)
        fail_reasons = self._classify_fail_reasons(score_components)
        hard_failures = [item for item in fail_reasons if item["severity"] == "hard"]
        gate_status = self._build_gate_status(score, threshold, hard_failures)
        risks, recommendations = self._build_risks_and_recommendations(score_components)
        scorecard = self._build_scorecard(score_components, threshold)

        return {
            "phase": phase,
            "score": score,
            "label": label,
            "acceptance_threshold": threshold,
            "gate_status": gate_status,
            "passed": gate_status == "pass",
            "score_components": score_components,
            "scorecard": scorecard,
            "task_types": task_types,
            "fail_reasons": fail_reasons,
            "hard_failures": hard_failures,
            "risks": risks,
            "recommendations": recommendations,
            "summary": self._build_summary(label, gate_status, fail_reasons),
        }

    def _build_label(self, score: float) -> str:
        if score >= 0.8:
            return "可信度强"
        if score >= 0.6:
            return "可信度中等"
        return "可信度偏弱"

    def _build_gate_status(self, score: float, threshold: float, hard_failures: list[dict[str, Any]]) -> str:
        if hard_failures:
            return "fail"
        if score >= threshold:
            return "pass"
        if score >= max(0.5, threshold - 0.1):
            return "review_required"
        return "fail"

    def _build_scorecard(self, score_components: dict[str, float], threshold: float) -> list[dict[str, Any]]:
        scorecard: list[dict[str, Any]] = []
        for name, value in score_components.items():
            status = "pass" if value >= threshold else ("review" if value >= max(0.5, threshold - 0.1) else "fail")
            scorecard.append(
                {
                    "dimension": name,
                    "score": value,
                    "status": status,
                }
            )
        return scorecard

    def _classify_fail_reasons(self, score_components: dict[str, float]) -> list[dict[str, Any]]:
        rules = [
            (
                "workflow_integrity",
                0.75,
                "PLAN_CHAIN_INCOMPLETE",
                "研究计划没有覆盖完整研究闭环。",
                "hard",
            ),
            (
                "goal_clarity",
                0.6,
                "GOAL_NOT_CLEAR",
                "研究目标表述不清，容易导致后续执行偏航。",
                "hard",
            ),
            (
                "history_traceability",
                0.6,
                "TRACEABILITY_WEAK",
                "关键步骤缺少历史留痕，复盘与审计能力不足。",
                "hard",
            ),
            (
                "metrics_availability",
                0.6,
                "METRICS_INSUFFICIENT",
                "结果缺少足够指标支撑，难以形成可靠判断。",
                "hard",
            ),
            (
                "execution_support",
                0.65,
                "EXECUTION_EVIDENCE_WEAK",
                "结论缺少足够执行证据支撑。",
                "hard",
            ),
            (
                "knowledge_support",
                0.5,
                "KNOWLEDGE_GROUNDING_WEAK",
                "缺少足够知识背景支撑，容易形成脱离上下文的判断。",
                "soft",
            ),
            (
                "portfolio_awareness",
                0.5,
                "PORTFOLIO_PERSPECTIVE_WEAK",
                "缺少组合层或上层约束视角。",
                "soft",
            ),
            (
                "portfolio_scope",
                0.5,
                "PORTFOLIO_SCOPE_WEAK",
                "总结阶段缺少组合层或完整研究闭环视角。",
                "soft",
            ),
            (
                "result_completeness",
                0.55,
                "RESULT_COMPLETENESS_WEAK",
                "任务结果信息不完整，后续复用价值有限。",
                "soft",
            ),
        ]
        reasons: list[dict[str, Any]] = []
        for dimension, threshold, code, message, severity in rules:
            value = score_components.get(dimension)
            if value is None:
                continue
            if value < threshold:
                reasons.append(
                    {
                        "dimension": dimension,
                        "threshold": threshold,
                        "score": value,
                        "code": code,
                        "message": message,
                        "severity": severity,
                    }
                )
        return reasons

    def _build_risks_and_recommendations(self, score_components: dict[str, float]) -> tuple[list[str], list[str]]:
        risks: list[str] = []
        recommendations: list[str] = []
        if score_components.get("workflow_integrity", 1.0) < 0.75:
            risks.append("研究流程不完整，容易在证据不足时提前下结论。")
            recommendations.append("补齐基线、调参、样本外与稳定性验证，或在因子场景下补齐安全执行与入库闭环。")
        if score_components.get("history_traceability", 1.0) < 0.6:
            risks.append("缺少历史留痕会削弱复盘、审计与后续自动化能力。")
            recommendations.append("确保每次关键执行都写入可追踪记录。")
        if score_components.get("metrics_availability", 1.0) < 0.6 or score_components.get("metrics_presence", 1.0) < 0.6:
            risks.append("缺少量化指标会让结论更像观点而不是证据。")
            recommendations.append("补充核心评分卡、对比结果或稳健性指标。")
        if score_components.get("knowledge_support", 1.0) < 0.5:
            risks.append("知识支撑不足时，建议降低自动化结论力度。")
            recommendations.append("增加知识库命中或引用历史研究记忆。")
        return risks, recommendations

    def _build_summary(self, label: str, gate_status: str, fail_reasons: list[dict[str, Any]]) -> str:
        if not fail_reasons:
            return f"{label}，当前阶段已通过验收。"
        lead_reason = fail_reasons[0]["message"]
        return f"{label}，当前阶段状态为 {gate_status}；主要问题：{lead_reason}"
