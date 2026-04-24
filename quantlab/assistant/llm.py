from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from quantlab.assistant.config import AssistantProfile
from quantlab.assistant.evaluator import ResearchDecisionEvaluator
from quantlab.assistant.knowledge_base import ProjectKnowledgeBase

from quantlab.assistant.memory import ConversationMemoryStore, ResearchMemoryStore
from quantlab.assistant.planner import ResearchPlanner
from quantlab.assistant.tools import AssistantToolRuntime
from quantlab.research.models import ResearchPlan, ResearchTask


@dataclass
class LLMSettings:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.2


class QuantPanelAssistant:
    def __init__(
        self,
        profile: AssistantProfile,
        runtime: AssistantToolRuntime,
        knowledge_base: ProjectKnowledgeBase,
        memory_store: ConversationMemoryStore,
        planner: ResearchPlanner | None = None,
        research_memory_store: ResearchMemoryStore | None = None,
    ) -> None:
        self.profile = profile
        self.runtime = runtime
        self.knowledge_base = knowledge_base
        self.memory_store = memory_store
        self.planner = planner or ResearchPlanner()
        self.research_memory_store = research_memory_store or ResearchMemoryStore()
        self.evaluator = ResearchDecisionEvaluator()
        self.profile.ensure_dirs()


    def chat(self, user_message: str) -> dict[str, Any]:
        self.memory_store.append("user", user_message)
        tool_logs: list[dict[str, Any]] = []
        summary = self.memory_store.get_summary() or self.memory_store.build_fallback_summary()
        knowledge_hits = self.knowledge_base.retrieve(user_message, limit=self.profile.max_knowledge_chunks)

        research_plan = self.planner.build_plan(
            user_message=user_message,
            data_path=self.runtime.data_path,
            strategy_name="ma_cross",
        )
        serialized_plan = self._serialize_plan(research_plan)
        plan_assessment = self.evaluator.evaluate_plan(research_plan)
        effective_plan = research_plan
        effective_plan_serialized = serialized_plan
        plan_execution_control = self._resolve_plan_execution(research_plan, plan_assessment)
        plan_execution_control = self._resolve_forced_execution_if_confirmed(
            user_message=user_message,
            plan_execution_control=plan_execution_control,
        )
        if plan_execution_control["effective_plan"] is not None:
            effective_plan = plan_execution_control["effective_plan"]
            effective_plan_serialized = self._serialize_plan(effective_plan)

        plan_feedback = self._build_plan_feedback(
            source_plan=research_plan,
            effective_plan=effective_plan,
            plan_assessment=plan_assessment,
            plan_execution_control=plan_execution_control,
        )
        decision_context = self._build_decision_context(
            user_message=user_message,
            knowledge_hits=knowledge_hits,
            research_plan=effective_plan,
            plan_feedback=plan_feedback,
        )

        self.research_memory_store.append_plan(
            goal=effective_plan_serialized["goal"],
            tasks=effective_plan_serialized["tasks"],
            rationale=effective_plan_serialized["rationale"],
            metadata={
                **effective_plan_serialized["metadata"],
                "credibility_assessment": plan_assessment,
                "plan_execution_control": plan_execution_control,
                "plan_feedback": plan_feedback,
                "source_plan": serialized_plan,
            },
            decision_context=decision_context,
        )
        self.research_memory_store.append_decision_record(
            decision_type="planning",
            summary=f"已为目标“{effective_plan.goal}”生成研究计划。",
            evidence=self._build_plan_evidence(effective_plan, knowledge_hits),
            metadata={
                "goal": effective_plan.goal,
                "plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                "task_count": len(effective_plan.tasks),
                "credibility_assessment": plan_assessment,
                "plan_execution_control": plan_execution_control,
                "plan_feedback": plan_feedback,
                "source_plan": serialized_plan,
            },
        )
        self.research_memory_store.append_decision_record(
            decision_type="plan_gate",
            summary=plan_execution_control["summary"],
            evidence=self._build_plan_gate_evidence(plan_assessment, research_plan, effective_plan),
            metadata={
                "goal": research_plan.goal,
                "source_plan_type": research_plan.metadata.get("plan_type", "unknown"),
                "effective_plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                "plan_execution_control": plan_execution_control,
                "plan_feedback": plan_feedback,
                "credibility_assessment": plan_assessment,
            },
        )


        research_snapshot = self.research_memory_store.latest_snapshot()

        autopilot_results = self._run_research_plan_if_needed(effective_plan, plan_execution_control)
        execution_assessment = self.evaluator.evaluate_execution(autopilot_results)
        if autopilot_results:


            tool_logs.extend(
                {
                    "tool": "execute_research_plan",
                    "arguments": {
                        "task_count": len(effective_plan.tasks),
                        "execution_mode": plan_execution_control.get("execution_mode", "full_autopilot"),
                    },
                    "result": autopilot_results,
                }
            )
            self.research_memory_store.append_artifact(
                "research_plan_execution",
                {
                    "goal": effective_plan.goal,
                    "plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                    "execution_mode": plan_execution_control.get("execution_mode", "full_autopilot"),
                    "credibility_assessment": execution_assessment,
                    "results": [self._summarize_tool_result(item.get("summary", {})) for item in autopilot_results],
                },
            )
            self.research_memory_store.append_decision_record(
                decision_type="execution",
                summary=f"已自动执行研究计划中的 {len(autopilot_results)} 个步骤。",
                evidence=self._build_execution_evidence(autopilot_results),
                metadata={
                    "goal": effective_plan.goal,
                    "plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                    "executed_task_count": len(autopilot_results),
                    "credibility_assessment": execution_assessment,
                    "plan_execution_control": plan_execution_control,
                },
            )



        conversation = self._build_conversation(
            user_message=user_message,
            summary=summary,
            knowledge_hits=knowledge_hits,
            research_plan=effective_plan,
            research_snapshot=self.research_memory_store.latest_snapshot(),
            autopilot_results=autopilot_results,
        )

        response_payload = self._request_completion(conversation)
        final_text = ""

        for _ in range(self.profile.tool_call_limit):
            tool_calls = self._extract_tool_calls(response_payload)
            if not tool_calls:
                final_text = self._extract_text(response_payload)
                break
            conversation.extend(response_payload.get("output", []))
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                tool_result = self.runtime.execute(tool_name, arguments)
                tool_logs.append({"tool": tool_name, "arguments": arguments, "result": tool_result})
                self.research_memory_store.append_artifact(
                    "tool_result",
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": self._summarize_tool_result(tool_result),
                    },
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.get("call_id"),
                        "output": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            response_payload = self._request_completion(conversation)
        else:
            final_text = self._extract_text(response_payload) or "我已经执行了一部分操作，但本轮工具调用达到上限。你可以继续追问，我会基于现有结果接着做。"

        if not final_text:
            final_text = self._extract_text(response_payload)
        if plan_feedback.get("message"):
            final_text = f"{plan_feedback['message']}\n\n{final_text}" if final_text else plan_feedback["message"]
        self.memory_store.append(
            "assistant",
            final_text,
            metadata={
                "tool_logs": tool_logs,
                "research_plan": effective_plan_serialized,
                "source_research_plan": serialized_plan,
                "plan_execution_control": plan_execution_control,
                "plan_feedback": plan_feedback,
                "autopilot_results": autopilot_results,
            },
        )

        self.memory_store.maybe_rollup_summary(trigger_messages=self.profile.summary_trigger_messages)
        self.research_memory_store.append_insight(
            title=self._build_insight_title(effective_plan.goal),
            content=final_text,
            metadata={
                "goal": effective_plan.goal,
                "tool_count": len(tool_logs),
                "knowledge_hit_count": len(knowledge_hits),
                "plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                "autopilot_executed": bool(autopilot_results),
                "plan_execution_mode": plan_execution_control.get("execution_mode", "blocked"),
                "plan_feedback": plan_feedback,
            },
        )
        conclusion_assessment = self.evaluator.evaluate_conclusion(
            research_plan=effective_plan,
            tool_logs=tool_logs,
            autopilot_results=autopilot_results,
            knowledge_hits=knowledge_hits,
        )
        self.research_memory_store.append_decision_record(
            decision_type="conclusion",
            summary=f"已输出研究结论：{self._build_insight_title(effective_plan.goal)}",
            evidence=self._build_conclusion_evidence(
                research_plan=effective_plan,
                tool_logs=tool_logs,
                autopilot_results=autopilot_results,
                knowledge_hits=knowledge_hits,
            ),
            metadata={
                "goal": effective_plan.goal,
                "plan_type": effective_plan.metadata.get("plan_type", "unknown"),
                "tool_count": len(tool_logs),
                "credibility_assessment": conclusion_assessment,
                "plan_execution_control": plan_execution_control,
                "plan_feedback": plan_feedback,
            },
        )
        latest_snapshot = self.research_memory_store.latest_snapshot()
        return {
            "answer": final_text,
            "tool_logs": tool_logs,
            "knowledge_hits": knowledge_hits,
            "summary": self.memory_store.get_summary(),
            "research_plan": effective_plan_serialized,
            "source_research_plan": serialized_plan,
            "plan_execution_control": plan_execution_control,
            "plan_feedback": plan_feedback,
            "research_snapshot": latest_snapshot,
            "autopilot_results": autopilot_results,
            "credibility_assessment": {
                "plan": plan_assessment,
                "execution": execution_assessment,
                "conclusion": conclusion_assessment,
            },
        }




    def _build_conversation(
        self,
        user_message: str,
        summary: str,
        knowledge_hits: list[dict],
        research_plan: ResearchPlan,
        research_snapshot: dict[str, Any],
        autopilot_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        recent_messages = self.memory_store.get_recent_messages(limit=self.profile.max_context_messages)
        system_text_parts = [self.profile.system_prompt]
        if summary:
            system_text_parts.append(f"已压缩的长期会话摘要：\n{summary}")
        if knowledge_hits:
            kb_text = "\n\n".join(
                [f"[{item['title']}]\n来源：{item['source']}\n{item['content']}" for item in knowledge_hits]
            )
            system_text_parts.append(f"当前可检索知识库片段：\n{kb_text}")

        system_text_parts.append(f"当前建议研究计划：\n{self._format_research_plan(research_plan)}")
        snapshot_text = self._format_research_snapshot(research_snapshot)
        if snapshot_text:
            system_text_parts.append(f"最近研究记忆快照：\n{snapshot_text}")
        if autopilot_results:
            system_text_parts.append(f"本轮已自动完成的研究步骤摘要：\n{self._format_autopilot_results(autopilot_results)}")

        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(system_text_parts)},
        ]
        for message in recent_messages[:-1]:
            conversation.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )
        conversation.append({"role": "user", "content": user_message})
        return conversation

    def _request_completion(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = f"{self.profile.base_url.rstrip('/')}/responses"
        payload = {
            "model": self.profile.model,
            "input": messages,
            "temperature": self.profile.temperature,
            "tools": self.runtime.describe_tools(),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.profile.api_key}",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"AI 接口调用失败：HTTP {exc.code} - {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"AI 接口调用失败：{exc}") from exc

    def _extract_tool_calls(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for item in payload.get("output", []):
            if item.get("type") == "function_call":
                arguments = item.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls.append(
                    {
                        "name": item.get("name"),
                        "arguments": arguments,
                        "call_id": item.get("call_id"),
                    }
                )
        return tool_calls

    def _extract_text(self, payload: dict[str, Any]) -> str:
        text_segments: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_segments.append(content.get("text", ""))
        return "\n".join(segment for segment in text_segments if segment).strip()

    def _serialize_plan(self, plan: ResearchPlan) -> dict[str, Any]:
        return {
            "goal": plan.goal,
            "tasks": [self._serialize_task(task) for task in plan.tasks],
            "rationale": list(plan.rationale),
            "metadata": dict(plan.metadata),
        }

    def _serialize_task(self, task: ResearchTask) -> dict[str, Any]:
        return {
            "task_type": task.task_type,
            "data_path": str(task.data_path),
            "strategy_name": task.strategy_name,
            "config_overrides": dict(task.config_overrides),
            "parameter_grid": dict(task.parameter_grid or {}),
            "metadata": dict(task.metadata),
        }

    def _format_research_plan(self, plan: ResearchPlan) -> str:
        if not plan.tasks:
            return "- 当前没有待执行研究任务。"
        lines = [f"目标：{plan.goal}"]
        for index, task in enumerate(plan.tasks, start=1):
            lines.append(f"{index}. {task.task_type} | 策略={task.strategy_name} | 数据={task.data_path}")
        if plan.rationale:
            lines.append("规划依据：")
            lines.extend([f"- {item}" for item in plan.rationale])
        if plan.metadata:
            lines.append(f"计划元数据：{json.dumps(plan.metadata, ensure_ascii=False)}")
        return "\n".join(lines)

    def _format_research_snapshot(self, snapshot: dict[str, Any]) -> str:
        lines: list[str] = []
        latest_plan = snapshot.get("latest_plan")
        latest_insight = snapshot.get("latest_insight")
        latest_artifact = snapshot.get("latest_artifact")
        latest_decision = snapshot.get("latest_decision")
        if latest_plan:
            lines.append(f"- 最近计划：{latest_plan.get('goal', '')}")
        if latest_insight:
            lines.append(f"- 最近洞察：{latest_insight.get('title', '')}")
        if latest_artifact:
            lines.append(f"- 最近产物：{latest_artifact.get('artifact_type', '')}")
        if latest_decision:
            lines.append(f"- 最近决策：{latest_decision.get('decision_type', '')} | {latest_decision.get('summary', '')}")
        return "\n".join(lines)

    def _build_plan_feedback(
        self,
        source_plan: ResearchPlan,
        effective_plan: ResearchPlan,
        plan_assessment: dict[str, Any],
        plan_execution_control: dict[str, Any],
    ) -> dict[str, Any]:
        execution_mode = str(plan_execution_control.get("execution_mode", "blocked"))
        recovery_assessment = plan_execution_control.get("recovery_assessment") or {}
        needs_confirmation = execution_mode in {"blocked", "blocked_after_replan"}
        user_status = self._build_user_facing_status(execution_mode)
        fail_reason_items = self._format_fail_reason_items(plan_assessment.get("fail_reasons", []))
        recommendations = [str(item) for item in plan_assessment.get("recommendations", [])[:3] if str(item).strip()]
        recovery_summary = self._build_recovery_summary(
            source_plan=source_plan,
            effective_plan=effective_plan,
            execution_mode=execution_mode,
            recovery_assessment=recovery_assessment,
        )
        next_actions = self._build_next_actions(execution_mode)
        confirmation_prompt = self._build_confirmation_prompt(execution_mode, effective_plan)
        message = self._build_feedback_message(
            source_plan=source_plan,
            effective_plan=effective_plan,
            execution_mode=execution_mode,
            fail_reason_items=fail_reason_items,
            recovery_summary=recovery_summary,
            recommendations=recommendations,
            needs_confirmation=needs_confirmation,
            confirmation_prompt=confirmation_prompt,
        )
        return {
            "user_status": user_status,
            "execution_mode": execution_mode,
            "needs_confirmation": needs_confirmation,
            "message": message,
            "confirmation_prompt": confirmation_prompt,
            "gate_summary": str(plan_execution_control.get("summary", "")).strip(),
            "fail_reasons": fail_reason_items,
            "recommendations": recommendations,
            "recovery_summary": recovery_summary,
            "next_actions": next_actions,
        }

    def _build_user_facing_status(self, execution_mode: str) -> str:
        mapping = {
            "full_autopilot": "正常自动执行",
            "limited_autopilot": "已降级执行",
            "replanned_autopilot": "已自动重规划后执行",
            "forced_autopilot": "已确认后强制执行",
            "blocked_after_replan": "重规划后仍被拦截",
            "blocked": "已拦截等待确认",
        }
        return mapping.get(execution_mode, "计划处理中")


    def _format_fail_reason_items(self, fail_reasons: list[dict[str, Any]]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for item in fail_reasons[:4]:
            if not isinstance(item, dict):
                continue
            formatted.append(
                {
                    "code": str(item.get("code", "UNKNOWN")),
                    "message": str(item.get("message", "计划未通过当前阶段验收。")),
                    "severity": str(item.get("severity", "soft")),
                }
            )
        return formatted

    def _build_recovery_summary(
        self,
        source_plan: ResearchPlan,
        effective_plan: ResearchPlan,
        execution_mode: str,
        recovery_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        if execution_mode not in {"replanned_autopilot", "blocked_after_replan"}:
            return {}
        return {
            "source_plan_type": str(source_plan.metadata.get("plan_type", "unknown")),
            "effective_plan_type": str(effective_plan.metadata.get("plan_type", "unknown")),
            "source_task_types": [task.task_type for task in source_plan.tasks],
            "effective_task_types": [task.task_type for task in effective_plan.tasks],
            "recovery_gate_status": str(recovery_assessment.get("gate_status", "unknown")),
            "recovery_score": recovery_assessment.get("score", 0.0),
        }

    def _build_next_actions(self, execution_mode: str) -> list[str]:
        mapping = {
            "full_autopilot": ["继续查看自动执行结果", "基于研究结果追问结论与风险"],
            "limited_autopilot": ["先查看有限验证结果", "如需完整链路可再确认放大执行范围"],
            "replanned_autopilot": ["先查看自动补齐后的研究结果", "如需保留原计划口径可要求对比原计划与重规划计划"],
            "forced_autopilot": ["重点核对强制执行结果的证据充分性", "如需更稳妥结论可要求补充复核或改写研究目标"],
            "blocked_after_replan": ["确认是否接受当前补齐后的计划再手动执行", "或先修改研究目标后重新规划"],
            "blocked": ["补充更明确的研究目标", "或明确确认是否仍要按当前低可信度计划继续执行"],
        }
        return mapping.get(execution_mode, ["继续补充研究目标", "重新发起规划"])


    def _build_confirmation_prompt(self, execution_mode: str, effective_plan: ResearchPlan) -> str:
        if execution_mode == "blocked_after_replan":
            return f"当前补齐后的计划仍未通过验收。若你仍想继续，我可以按 {effective_plan.metadata.get('plan_type', '当前计划')} 强制执行；是否继续？"
        if execution_mode == "blocked":
            return "当前计划可信度不足，默认已停止自动执行。若你仍想继续，我可以按当前计划强制执行；是否继续？"
        return ""

    def _build_feedback_message(
        self,
        source_plan: ResearchPlan,
        effective_plan: ResearchPlan,
        execution_mode: str,
        fail_reason_items: list[dict[str, str]],
        recovery_summary: dict[str, Any],
        recommendations: list[str],
        needs_confirmation: bool,
        confirmation_prompt: str,
    ) -> str:
        lines: list[str] = []
        lines.append(f"计划状态：{self._build_user_facing_status(execution_mode)}")
        if execution_mode == "limited_autopilot":
            lines.append("原计划没有被直接放行，系统已先缩成更保守的验证链路，避免一上来跑完整研究闭环。")
        elif execution_mode == "replanned_autopilot":
            lines.append("原计划没有直接通过，系统已自动补齐缺失研究链路，并按补齐后的计划继续执行。")
        elif execution_mode == "blocked_after_replan":
            lines.append("原计划未通过，且自动重规划后仍不达标，所以自动执行已被拦截。")
        elif execution_mode == "blocked":
            lines.append("原计划未通过验收，系统已停止自动执行，避免基于低可信度计划直接产出结论。")

        if fail_reason_items:
            lines.append("主要原因：")
            lines.extend([f"- {item['code']}：{item['message']}" for item in fail_reason_items])

        if recovery_summary:
            lines.append(
                f"重规划说明：{recovery_summary.get('source_plan_type', 'unknown')} → {recovery_summary.get('effective_plan_type', 'unknown')}"
            )
            lines.append(
                f"补齐任务：{', '.join(recovery_summary.get('effective_task_types', [])) or '无'}"
            )

        if recommendations:
            lines.append("建议动作：")
            lines.extend([f"- {item}" for item in recommendations])

        if needs_confirmation and confirmation_prompt:
            lines.append(f"确认请求：{confirmation_prompt}")
        return "\n".join(lines)

    def _build_decision_context(
        self,
        user_message: str,
        knowledge_hits: list[dict[str, Any]],
        research_plan: ResearchPlan,
        plan_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "user_message": user_message,
            "knowledge_hit_titles": [item.get("title", "") for item in knowledge_hits[:5]],
            "plan_type": research_plan.metadata.get("plan_type", "unknown"),
            "task_types": [task.task_type for task in research_plan.tasks],
            "plan_feedback": plan_feedback or {},
        }

    def _build_plan_evidence(self, research_plan: ResearchPlan, knowledge_hits: list[dict[str, Any]]) -> list[str]:
        evidence = [f"plan_type={research_plan.metadata.get('plan_type', 'unknown')}"]
        evidence.extend([f"task={task.task_type}" for task in research_plan.tasks])
        evidence.extend([f"knowledge={item.get('title', '')}" for item in knowledge_hits[:3] if item.get("title")])
        return evidence


    def _build_execution_evidence(self, autopilot_results: list[dict[str, Any]]) -> list[str]:
        evidence: list[str] = []
        for item in autopilot_results:
            summary = item.get("summary", {})
            history_path = summary.get("history_path")
            evidence.append(f"executed={item.get('task_type')}:{item.get('strategy_name')}")
            if history_path:
                evidence.append(f"history={history_path}")
        return evidence

    def _build_conclusion_evidence(
        self,
        research_plan: ResearchPlan,
        tool_logs: list[dict[str, Any]],
        autopilot_results: list[dict[str, Any]],
        knowledge_hits: list[dict[str, Any]],
    ) -> list[str]:
        evidence = [f"goal={research_plan.goal}", f"plan_type={research_plan.metadata.get('plan_type', 'unknown')}"]
        evidence.append(f"tool_logs={len(tool_logs)}")
        evidence.append(f"autopilot_results={len(autopilot_results)}")
        evidence.extend([f"knowledge={item.get('title', '')}" for item in knowledge_hits[:2] if item.get("title")])
        return evidence

    def _resolve_forced_execution_if_confirmed(
        self,
        user_message: str,
        plan_execution_control: dict[str, Any],
    ) -> dict[str, Any]:
        execution_mode = str(plan_execution_control.get("execution_mode", "blocked"))
        if execution_mode not in {"blocked", "blocked_after_replan"}:
            return plan_execution_control

        confirmation_context = self._get_pending_confirmation_context()
        if not confirmation_context:
            return plan_execution_control

        confirmation_intent = self._assess_force_execution_intent(user_message, confirmation_context)
        if not confirmation_intent.get("confirmed", False):
            return plan_execution_control

        forced_control = dict(plan_execution_control)
        effective_plan = forced_control.get("effective_plan")
        effective_plan_type = forced_control.get("effective_plan_type") or (
            effective_plan.metadata.get("plan_type", "unknown") if effective_plan is not None else "unknown"
        )
        forced_control.update(
            {
                "execution_mode": "forced_autopilot",
                "should_execute": True,
                "summary": f"用户已明确确认继续执行，系统已按 {effective_plan_type} 强制放行当前计划。",
                "forced_by_user_confirmation": True,
                "previous_execution_mode": execution_mode,
                "confirmation_message": user_message.strip(),
                "confirmation_intent": confirmation_intent,
            }
        )
        return forced_control

    def _get_pending_confirmation_context(self) -> dict[str, Any] | None:
        recent_messages = self.memory_store.get_recent_messages(limit=8)
        for message in reversed(recent_messages[:-1]):
            if message.get("role") != "assistant":
                continue
            metadata = message.get("metadata") or {}
            plan_feedback = metadata.get("plan_feedback") or {}
            execution_mode = metadata.get("plan_execution_control", {}).get("execution_mode")
            if not (
                plan_feedback.get("needs_confirmation")
                or execution_mode in {"blocked", "blocked_after_replan"}
                or plan_feedback.get("confirmation_prompt")
            ):
                break
            return {
                "confirmation_prompt": str(plan_feedback.get("confirmation_prompt", "")).strip(),
                "execution_mode": str(execution_mode or plan_feedback.get("execution_mode", "blocked")),
                "assistant_message": str(message.get("content", "")).strip(),
            }
        return None

    def _assess_force_execution_intent(
        self,
        user_message: str,
        confirmation_context: dict[str, Any],
    ) -> dict[str, Any]:
        raw_message = str(user_message or "").strip()
        normalized = self._normalize_intent_text(raw_message)
        if not normalized:
            return {"confirmed": False, "score": 0.0, "reason": "empty_message"}

        if self._contains_rejection_intent(normalized):
            return {"confirmed": False, "score": 0.0, "reason": "rejection_detected"}

        if self._looks_like_clarification_or_question(raw_message, normalized):
            return {"confirmed": False, "score": 0.1, "reason": "question_or_clarification"}

        score = 0.0
        reasons: list[str] = []
        prompt_normalized = self._normalize_intent_text(confirmation_context.get("confirmation_prompt", ""))
        assistant_message_normalized = self._normalize_intent_text(confirmation_context.get("assistant_message", ""))

        strong_positive_markers = [
            "强制执行",
            "确认执行",
            "继续执行",
            "继续跑",
            "继续做",
            "就按这个执行",
            "按当前计划执行",
            "按这个计划执行",
            "按补齐后的计划执行",
            "按当前方案执行",
            "继续推进",
            "继续往下跑",
            "继续往下做",
            "直接执行",
            "直接跑",
        ]
        if any(marker in normalized for marker in strong_positive_markers):
            score += 0.75
            reasons.append("strong_positive_marker")

        short_affirmations = {
            "继续",
            "确认",
            "可以",
            "行",
            "好",
            "好的",
            "嗯",
            "嗯嗯",
            "ok",
            "okay",
            "yes",
        }
        if normalized in short_affirmations:
            score += 0.45
            reasons.append("short_affirmation")

        positive_signal_groups = [
            ("继续", "执行"),
            ("确认", "执行"),
            ("继续", "计划"),
            ("按", "执行"),
            ("按", "计划"),
            ("继续", "跑"),
            ("继续", "做"),
            ("可以", "执行"),
            ("好", "执行"),
        ]
        for group in positive_signal_groups:
            if all(keyword in normalized for keyword in group):
                score += 0.25
                reasons.append(f"positive_group:{'+'.join(group)}")

        supportive_context_markers = ["就这样", "没问题", "可以继续", "按这个来", "照这个来"]
        if any(marker in normalized for marker in supportive_context_markers):
            score += 0.2
            reasons.append("supportive_context_marker")

        if prompt_normalized and any(keyword in prompt_normalized for keyword in ["继续", "执行", "强制"]):
            score += 0.15
            reasons.append("prompt_confirms_confirmation_context")
        if assistant_message_normalized and any(keyword in assistant_message_normalized for keyword in ["确认请求", "是否继续", "强制执行"]):
            score += 0.15
            reasons.append("assistant_message_confirms_context")

        condition_markers = ["如果", "但是", "不过", "前提是", "先", "等我", "稍后"]
        if any(marker in raw_message for marker in condition_markers):
            score -= 0.35
            reasons.append("conditional_or_deferred_reply")

        long_explanation_markers = ["因为", "原因", "风险", "依据", "解释"]
        if len(normalized) > 24 and any(marker in normalized for marker in long_explanation_markers):
            score -= 0.2
            reasons.append("explanatory_reply")

        confirmed = score >= 0.6
        return {
            "confirmed": confirmed,
            "score": round(score, 3),
            "reason": "confirmed_by_context_score" if confirmed else "insufficient_confirmation_score",
            "signals": reasons[:8],
            "context_execution_mode": confirmation_context.get("execution_mode", "blocked"),
        }

    def _contains_rejection_intent(self, normalized: str) -> bool:
        rejection_markers = [
            "不要执行",
            "先别执行",
            "先不要执行",
            "不用执行",
            "暂不执行",
            "不继续",
            "不跑",
            "别跑",
            "取消执行",
            "停止执行",
            "不用了",
            "算了",
            "等等",
            "等下",
            "再等等",
            "先放着",
        ]
        return any(marker in normalized for marker in rejection_markers)


    def _normalize_intent_text(self, text: str) -> str:
        normalized = str(text).strip().lower()
        punctuation = " \\t\\r\\n,.!?;:，。！？；：、~`'\"“”‘’()（）[]【】<>《》-_=+*/\\|"
        translation = str.maketrans("", "", punctuation)
        return normalized.translate(translation)

    def _looks_like_clarification_or_question(self, raw_message: str, normalized: str) -> bool:
        question_markers = [
            "?",
            "？",
            "为什么",
            "为何",
            "怎么",
            "如何",
            "吗",
            "能不能",
            "可不可以",
            "是否",
            "啥意思",
            "什么意思",
            "风险",
            "原因",
            "依据",
            "会怎样",
            "是不是",
        ]
        if any(marker in raw_message for marker in ["?", "？"]):
            return True
        if any(marker in normalized for marker in [self._normalize_intent_text(item) for item in question_markers if item not in {"?", "？"}]):
            return True

        clarification_markers = [
            "先说",
            "先讲",
            "先解释",
            "展开说说",
            "具体说说",
            "先看看",
            "先分析",
            "先判断",
            "先别急",
        ]
        return any(marker in normalized for marker in [self._normalize_intent_text(item) for item in clarification_markers])


    def _run_research_plan_if_needed(
        self,
        research_plan: ResearchPlan,
        plan_execution_control: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:

        if not research_plan.tasks:
            return []
        if plan_execution_control is not None and not plan_execution_control.get("should_execute", False):
            return []
        if len(research_plan.tasks) <= 1 and research_plan.metadata.get("plan_type") == "baseline_only":
            return []
        return self.runtime.execute_research_plan(research_plan.tasks)


    def _resolve_plan_execution(self, research_plan: ResearchPlan, plan_assessment: dict[str, Any]) -> dict[str, Any]:

        gate_status = plan_assessment.get("gate_status", "fail")
        original_plan_type = research_plan.metadata.get("plan_type", "unknown")

        if gate_status == "pass":
            return {
                "gate_status": gate_status,
                "execution_mode": "full_autopilot",
                "should_execute": True,
                "summary": f"计划评估通过，按 {original_plan_type} 正常进入自动执行。",
                "effective_plan": research_plan,
                "source_plan_type": original_plan_type,
                "effective_plan_type": original_plan_type,
            }

        if gate_status == "review_required":
            review_plan = self.planner.build_review_plan(research_plan, plan_assessment)
            return {
                "gate_status": gate_status,
                "execution_mode": "limited_autopilot",
                "should_execute": True,
                "summary": "计划评估要求复核，已自动降级为更保守的验证链路后再执行。",
                "effective_plan": review_plan,
                "source_plan_type": original_plan_type,
                "effective_plan_type": review_plan.metadata.get("plan_type", "unknown"),
            }

        hard_failures = plan_assessment.get("hard_failures", [])
        can_replan = bool(hard_failures) or plan_assessment.get("score", 0.0) >= 0.35
        if can_replan:
            recovery_plan = self.planner.build_recovery_plan(research_plan, plan_assessment)
            recovery_assessment = self.evaluator.evaluate_plan(recovery_plan)
            if recovery_assessment.get("gate_status") != "fail":
                return {
                    "gate_status": gate_status,
                    "execution_mode": "replanned_autopilot",
                    "should_execute": True,
                    "summary": "原计划未通过验收，已自动重规划并切换到补全后的研究链路执行。",
                    "effective_plan": recovery_plan,
                    "source_plan_type": original_plan_type,
                    "effective_plan_type": recovery_plan.metadata.get("plan_type", "unknown"),
                    "recovery_assessment": recovery_assessment,
                }
            return {
                "gate_status": gate_status,
                "execution_mode": "blocked_after_replan",
                "should_execute": False,
                "summary": "原计划未通过验收，且自动重规划后仍不达标，已拦截自动执行。",
                "effective_plan": recovery_plan,
                "source_plan_type": original_plan_type,
                "effective_plan_type": recovery_plan.metadata.get("plan_type", "unknown"),
                "recovery_assessment": recovery_assessment,
            }

        return {
            "gate_status": gate_status,
            "execution_mode": "blocked",
            "should_execute": False,
            "summary": "原计划未通过验收，已拦截自动执行，避免基于低可信度计划继续推进。",
            "effective_plan": research_plan,
            "source_plan_type": original_plan_type,
            "effective_plan_type": original_plan_type,
        }

    def _build_plan_gate_evidence(
        self,
        plan_assessment: dict[str, Any],
        research_plan: ResearchPlan,
        effective_plan: ResearchPlan,
    ) -> list[str]:
        evidence = [
            f"plan_gate={plan_assessment.get('gate_status', 'unknown')}",
            f"plan_score={plan_assessment.get('score', 0.0)}",
            f"source_plan_type={research_plan.metadata.get('plan_type', 'unknown')}",
            f"effective_plan_type={effective_plan.metadata.get('plan_type', 'unknown')}",
        ]
        for item in plan_assessment.get("fail_reasons", [])[:3]:
            if isinstance(item, dict) and item.get("code"):
                evidence.append(f"fail_reason={item['code']}")
        return evidence


    def _format_autopilot_results(self, results: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            summary = item.get("summary", {})
            metrics = summary.get("metrics") or summary.get("best_metrics") or summary.get("average_metrics")
            lines.append(f"{index}. {item.get('task_type')} | 策略={item.get('strategy_name')}")
            if metrics:
                lines.append(f"   指标摘要：{json.dumps(metrics, ensure_ascii=False)}")
            if summary.get("history_path"):
                lines.append(f"   历史记录：{summary.get('history_path')}")
        return "\n".join(lines)

    def _summarize_tool_result(self, tool_result: Any) -> dict[str, Any]:
        if not isinstance(tool_result, dict):
            return {"type": type(tool_result).__name__, "preview": str(tool_result)[:400]}
        summary: dict[str, Any] = {"keys": list(tool_result.keys())[:12]}
        if "history_path" in tool_result:
            summary["history_path"] = tool_result.get("history_path")
        if "metrics" in tool_result:
            summary["metrics"] = tool_result.get("metrics")
        if "best_metrics" in tool_result:
            summary["best_metrics"] = tool_result.get("best_metrics")
        if "average_metrics" in tool_result:
            summary["average_metrics"] = tool_result.get("average_metrics")
        if "overview" in tool_result:
            summary["overview"] = tool_result.get("overview")
        if "data_summary" in tool_result:
            summary["data_summary"] = tool_result.get("data_summary")
        return summary

    def _build_insight_title(self, goal: str) -> str:
        cleaned = (goal or "量化研究任务").strip()
        return f"研究结论：{cleaned[:40]}"
