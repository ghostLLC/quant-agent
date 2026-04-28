"""三团队多 Agent 协作框架 —— R1+R2 → P1+P2+P3 → T1+T2。

架构概览：
- Research 团队 (R1 生成 + R2 审查)：对抗循环，输出 FactorHypothesis
- Programming 团队 (P1 架构 + P2 积木组装 + P3 定制编码)：协作实现因子
- Testing 团队 (T1 回测 + T2 验证)：闭环验证因子质量

核心设计原则：
1. 每个 Agent 有明确的角色、输入输出契约
2. 团队内部有生成-审查对抗（R1↔R2）和分工协作（P1→P2→P3）
3. 团队间通过标准传递物（FactorHypothesis → ProgrammingPlan → TestResult）流转
4. LLM 驱动深度推理，非硬编码模板
5. 消息总线实现松耦合通信

设计借鉴：
- RD-Agent-Quant: Research → Development 两阶段迭代
- AlphaAgent: LLM + 正则化探索 + 对抗审查
- FactorMiner: Ralph Loop (Retrieve → Adapt → Learn)
- QuantaAlpha: trajectory-level 经验复用
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .blocks import (
    Block,
    BlockExecutor,
    CombineBlock,
    CustomRequest,
    DataBlock,
    FactorHypothesis,
    FilterBlock,
    ProgrammingPlan,
    RelationalBlock,
    TransformBlock,
    data,
    transform,
    combine,
)
from .hypothesis import FactorHypothesisGenerator, HypothesisCandidate, HypothesisRequest
from .models import (
    FactorDirection,
    FactorNode,
    FactorSpec,
    FactorStatus,
)
from .runtime import FactorExperienceMemory, PersistentFactorStore, SafeFactorExecutor
from .pipeline import FactorDiscoveryOrchestrator
from .sample_split import SampleSplitter
from .factor_enhancements import (
    CustomCodeGenerator,
    ExperienceLoop,
    FactorCombiner,
    FactorCombinationResult,
    FactorOutcome,
    OrthogonalityGuide,
    ParamSearchResult,
    ParameterSearcher,
    RiskNeutralizer,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. 消息总线 & Agent 基类
# ═══════════════════════════════════════════════════════════════════

class AgentRole(str, Enum):
    R1_HYPOTHESIS_GENERATOR = "r1_hypothesis_generator"
    R2_HYPOTHESIS_REVIEWER = "r2_hypothesis_reviewer"
    P1_ARCHITECT = "p1_architect"
    P2_BLOCK_ASSEMBLER = "p2_block_assembler"
    P3_CUSTOM_CODER = "p3_custom_coder"
    T1_BACKTESTER = "t1_backtester"
    T2_VALIDATOR = "t2_validator"


@dataclass
class AgentMessage:
    """Agent 间传递的消息。"""
    msg_id: str
    sender: AgentRole
    recipient: AgentRole | str  # "broadcast" 表示广播
    msg_type: str  # "hypothesis" | "review" | "plan" | "code" | "result" | "verdict"
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    thread_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "sender": self.sender.value,
            "recipient": self.recipient if isinstance(self.recipient, str) else self.recipient.value,
            "msg_type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
        }


class MessageBus:
    """简单的同步消息总线。"""

    def __init__(self) -> None:
        self._queue: list[AgentMessage] = []
        self._history: list[AgentMessage] = []

    def send(self, msg: AgentMessage) -> None:
        self._queue.append(msg)
        self._history.append(msg)

    def receive(self, recipient: AgentRole | str, msg_type: str | None = None) -> list[AgentMessage]:
        """取走指定接收者的消息。"""
        matched = []
        remaining = []
        for msg in self._queue:
            rec_match = (
                msg.recipient == recipient
                or (isinstance(msg.recipient, str) and msg.recipient == "broadcast")
                or (isinstance(recipient, str) and recipient == "broadcast")
            )
            type_match = msg_type is None or msg.msg_type == msg_type
            if rec_match and type_match:
                matched.append(msg)
            else:
                remaining.append(msg)
        self._queue = remaining
        return matched

    def history_for(self, role: AgentRole, limit: int = 50) -> list[dict]:
        """查看历史消息。"""
        results = []
        for msg in reversed(self._history):
            if msg.sender == role or msg.recipient == role or msg.recipient == "broadcast":
                results.append(msg.to_dict())
                if len(results) >= limit:
                    break
        return results


class BaseAgent(ABC):
    """Agent 基类。"""

    def __init__(self, role: AgentRole, bus: MessageBus, llm_client: "LLMClient | None" = None) -> None:
        self.role = role
        self.bus = bus
        self.llm = llm_client
        self._log: list[dict] = []

    def _log_action(self, action: str, detail: dict[str, Any]) -> None:
        self._log.append({"role": self.role.value, "action": action, "detail": detail})
        logger.debug(f"[{self.role.value}] {action}: {json.dumps(detail, ensure_ascii=False, default=str)[:200]}")

    def send(self, recipient: AgentRole | str, msg_type: str, payload: dict[str, Any], thread_id: str = "") -> None:
        msg = AgentMessage(
            msg_id=f"msg_{uuid4().hex[:8]}",
            sender=self.role,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
            thread_id=thread_id,
        )
        self.bus.send(msg)
        self._log_action("send", {"to": recipient if isinstance(recipient, str) else recipient.value, "type": msg_type})

    def receive(self, msg_type: str | None = None) -> list[AgentMessage]:
        return self.bus.receive(self.role, msg_type)

    @abstractmethod
    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """执行一轮动作，返回结果摘要。"""
        ...


# ═══════════════════════════════════════════════════════════════════
# 2. LLM 客户端 —— 统一推理接口
# ═══════════════════════════════════════════════════════════════════

class LLMClient:
    """轻量 LLM 客户端，兼容 OpenAI Responses API。"""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _load_from_env(self) -> None:
        """从 .env 或环境变量补充缺失的配置。"""
        import os
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
        if not self.base_url:
            self.base_url = os.environ.get("ASSISTANT_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
        if not self.api_key:
            self.api_key = os.environ.get("ASSISTANT_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        if not self.model:
            self.model = os.environ.get("ASSISTANT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o"))

    def chat(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        """单轮对话，返回文本。"""
        self._load_from_env()
        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM 未配置：请设置 ASSISTANT_BASE_URL 和 ASSISTANT_API_KEY")

        from urllib import error, request as urllib_req

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib_req.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib_req.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM 调用失败 HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM 调用失败: {exc}") from exc

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> dict[str, Any]:
        """单轮对话，期望返回 JSON。"""
        raw = self.chat(system_prompt, user_prompt, temperature)
        # 尝试提取 JSON 块
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试找到第一个 { 和最后一个 }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"LLM 返回非 JSON，原始内容: {text[:300]}")
            return {"raw_response": text, "parse_error": True}


# ═══════════════════════════════════════════════════════════════════
# 3. Research 团队 —— R1 (生成) + R2 (审查)
# ═══════════════════════════════════════════════════════════════════

R1_SYSTEM_PROMPT = """你是一个专业的量化因子假设生成专家（R1角色）。

你的职责是：
1. 根据研究方向和现有经验，生成有经济学直觉支撑的因子假设
2. 每个假设必须包含：直觉解释、作用机制、预期行为、风险因素
3. 假设必须可以用积木体系（Data/Transform/Combine/Relational/Filter）实现
4. 避免与已知因子同质化，追求新颖但有逻辑的方向
5. 必须遵守正交性约束，不在已有因子的饱和方向上重复
6. 优先使用使用不足的数据字段和未探索的积木组合

关键原则：
- 经验驱动：参考历史成功/失败模式，但不盲从
- 正交性优先：新假设必须与已有因子有实质区别
- 经济学直觉：每个假设必须有清晰的经济学故事
- 可执行性：必须能用积木算子实现

输出格式（严格 JSON）：
{
  "hypotheses": [
    {
      "direction": "因子方向关键词",
      "intuition": "经济学直觉，1-2句",
      "mechanism": "作用机制，为什么这个信号应该有效",
      "expected_behavior": "预期在横截面上的行为模式",
      "risk_factors": ["可能失效的情况1", "情况2"],
      "pseudocode": "伪代码，用积木算子描述",
      "input_fields": ["close", "volume", ...],
      "novelty_claim": "与已有因子的区别",
      "orthogonality_explanation": "为什么与已有因子正交"
    }
  ]
}"""

R2_SYSTEM_PROMPT = """你是一个审慎的因子假设审查专家（R2角色）。

你的职责是：
1. 审查 R1 生成的因子假设，评估其合理性、可执行性和新颖性
2. 指出逻辑漏洞、过度拟合风险、数据窥探嫌疑
3. 评估积木实现的可行性（输入字段是否可用、算子是否支持）
4. 检查正交性：假设是否与已有因子实质同质
5. 检查风险暴露：是否需要额外的中性化处理
6. 对每个假设给出 approve / revise / reject 决策

审查维度（按权重）：
- 逻辑合理性 (30%)：经济学直觉是否自洽，机制是否有学术/实证支撑
- 可行性 (20%)：能否用积木实现，输入字段是否可用
- 新颖性 (25%)：是否与常见因子不同，正交性是否足够
- 风险控制 (25%)：是否存在数据窥探、过拟合、风险暴露过大等问题

输出格式（严格 JSON）：
{
  "reviews": [
    {
      "hypothesis_index": 0,
      "decision": "approve|revise|reject",
      "strengths": ["优点1", "优点2"],
      "weaknesses": ["弱点1", "弱点2"],
      "suggestions": "改进建议（revise时必填）",
      "feasibility_score": 0.8,
      "novelty_score": 0.7,
      "logic_score": 0.9,
      "risk_score": 0.6,
      "orthogonality_score": 0.8,
      "needs_neutralization": ["market_cap", "industry"],
      "suggested_param_search": {"window": [5, 10, 20]}
    }
  ]
}"""


class R1HypothesisGenerator(BaseAgent):
    """R1: 因子假设生成 Agent（LLM 驱动）。"""

    def __init__(
        self,
        bus: MessageBus,
        llm_client: LLMClient | None = None,
        template_generator: FactorHypothesisGenerator | None = None,
    ) -> None:
        super().__init__(AgentRole.R1_HYPOTHESIS_GENERATOR, bus, llm_client)
        self.template_generator = template_generator or FactorHypothesisGenerator()
        self._orth_guide: OrthogonalityGuide | None = None  # 由编排器注入

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        direction = context.get("direction", "")
        max_candidates = context.get("max_candidates", 3)
        memory_context = context.get("memory_context", {})
        library_context = context.get("library_context", [])
        self._orth_guide = context.get("orth_guide") or self._orth_guide

        if self.llm and self.llm.api_key:
            hypotheses = self._generate_via_llm(direction, max_candidates, memory_context, library_context)
        else:
            hypotheses = self._generate_via_template(direction, max_candidates)

        # 包装为 FactorHypothesis 传递物
        results = []
        for h in hypotheses:
            results.append(h if isinstance(h, FactorHypothesis) else self._wrap_hypothesis(h, direction))

        # 发送给 R2 审查
        for i, hyp in enumerate(results):
            self.send(
                AgentRole.R2_HYPOTHESIS_REVIEWER,
                "hypothesis",
                hyp.to_dict(),
                thread_id=context.get("thread_id", ""),
            )

        self._log_action("generate_hypotheses", {"count": len(results), "direction": direction})
        return {"generated": len(results), "hypotheses": [h.to_dict() for h in results]}

    def _generate_via_llm(
        self,
        direction: str,
        max_candidates: int,
        memory_context: dict[str, Any],
        library_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """通过 LLM 深度推理生成因子假设，集成经验学习回路和正交性引导。"""
        # 构造经验上下文（增强版：包含结构统计和方向洞察）
        exp_summary = ""
        if memory_context:
            direction_insight = memory_context.get("direction_insight", "")
            structure_stats = memory_context.get("structure_stats", {})
            field_stats = memory_context.get("field_stats", {})
            successful = memory_context.get("successful_patterns", [])
            failed = memory_context.get("failed_patterns", [])
            marginal = memory_context.get("marginal_patterns", [])

            if direction_insight:
                exp_summary += f"\n【方向洞察】{direction_insight}"
            if successful:
                exp_summary += f"\n【成功模式】{json.dumps(successful[:3], ensure_ascii=False, default=str)}"
            if marginal:
                exp_summary += f"\n【边际模式】{json.dumps(marginal[:2], ensure_ascii=False, default=str)}"
            if failed:
                exp_summary += f"\n【失败模式（避开）】{json.dumps(failed[:2], ensure_ascii=False, default=str)}"
            if structure_stats:
                top_structs = list(structure_stats.items())[:5]
                exp_summary += f"\n【积木结构胜率】{json.dumps(dict(top_structs), ensure_ascii=False, default=str)}"
            if field_stats:
                exp_summary += f"\n【字段胜率】{json.dumps(field_stats, ensure_ascii=False, default=str)}"

        lib_summary = ""
        if library_context:
            lib_summary = f"\n已有因子库（前10个）：{json.dumps(library_context[:10], ensure_ascii=False, default=str)}"

        # 事前正交性引导
        orth_addon = ""
        if self._orth_guide:
            orth_addon = self._orth_guide.generate_orthogonality_prompt_addon(direction)

        # 可用积木算子
        from .blocks import OperatorRegistry
        ops_desc = []
        for op_name, op_def in OperatorRegistry.TRANSFORM_OPS.items():
            ops_desc.append(f"- {op_name}: {op_def['desc']}")
        for op_name, op_def in OperatorRegistry.COMBINE_OPS.items():
            ops_desc.append(f"- {op_name}: {op_def['desc']}")
        for op_name, op_def in OperatorRegistry.RELATIONAL_OPS.items():
            ops_desc.append(f"- {op_name}: {op_def['desc']}")

        # 可用数据字段
        available_fields = context_fields_description()

        # 事前正交性引导
        orth_addon = ""
        if self._orth_guide:
            orth_addon = self._orth_guide.generate_orthogonality_prompt_addon(direction)

        user_prompt = f"""研究方向：{direction}
需要生成 {max_candidates} 个因子假设。

可用数据字段：
{available_fields}

可用积木算子：
{chr(10).join(ops_desc)}

经验上下文：{exp_summary or '（暂无）'}
因子库：{lib_summary or '（空库）'}
{orth_addon}

请生成 {max_candidates} 个因子假设。每个假设必须：
1. 有清晰的经济学直觉
2. 用上述积木算子可以表达
3. 与已有因子有明显区别（遵守正交性约束）
4. 包含伪代码描述
5. 说明为何与已有因子正交
6. 考虑是否需要风险中性化（市值/行业/动量）
"""

        try:
            result = self.llm.chat_json(R1_SYSTEM_PROMPT, user_prompt, temperature=0.7)
            hypotheses = result.get("hypotheses", [])
            if not hypotheses and not result.get("parse_error"):
                # 尝试兼容不同格式
                hypotheses = result if isinstance(result, list) else []
            return hypotheses
        except Exception as exc:
            logger.warning(f"R1 LLM 生成失败，回退模板生成: {exc}")
            return self._generate_via_template(direction, max_candidates)

    def _generate_via_template(self, direction: str, max_candidates: int) -> list[FactorHypothesis]:
        """回退：使用已有的模板生成器。"""
        request = HypothesisRequest(
            research_direction=direction,
            max_candidates=max_candidates,
        )
        candidates = self.template_generator.generate(request)
        results = []
        for c in candidates:
            hyp = FactorHypothesis(
                hypothesis_id=c.spec.factor_id,
                direction=direction,
                intuition=c.rationale,
                mechanism=c.spec.hypothesis,
                expected_behavior=f"{c.family_match}族因子在横截面上的预测行为",
                risk_factors=["市场状态切换", "行业集中"],
                pseudocode=_expression_tree_to_pseudocode(c.spec.expression_tree),
                input_fields=[d.field_name for d in c.spec.dependencies],
            )
            results.append(hyp)
        return results

    def _wrap_hypothesis(self, raw: dict, direction: str) -> FactorHypothesis:
        """将 LLM 输出的原始字典包装为 FactorHypothesis。"""
        return FactorHypothesis(
            hypothesis_id=raw.get("hypothesis_id", f"hyp_{uuid4().hex[:8]}"),
            direction=raw.get("direction", direction),
            intuition=raw.get("intuition", ""),
            mechanism=raw.get("mechanism", ""),
            expected_behavior=raw.get("expected_behavior", ""),
            risk_factors=raw.get("risk_factors", []),
            pseudocode=raw.get("pseudocode", ""),
            input_fields=raw.get("input_fields", []),
        )


class R2HypothesisReviewer(BaseAgent):
    """R2: 因子假设审查 Agent（LLM 驱动对抗审查）。"""

    def __init__(self, bus: MessageBus, llm_client: LLMClient | None = None) -> None:
        super().__init__(AgentRole.R2_HYPOTHESIS_REVIEWER, bus, llm_client)
        self._pending: list[FactorHypothesis] = []

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        # 接收 R1 的假设
        messages = self.receive("hypothesis")
        for msg in messages:
            hyp = FactorHypothesis.from_dict(msg.payload)
            self._pending.append(hyp)

        if not self._pending:
            return {"reviewed": 0}

        reviews = []
        for hyp in self._pending:
            review = self._review_hypothesis(hyp, context)
            reviews.append(review)

            # 更新假设状态
            hyp.review_status = review["decision"]
            hyp.review_comments = review.get("suggestions", "")

            # 如果 approve 或 revise，传递给 P 团队
            if review["decision"] in ("approve", "revise"):
                hyp.iteration = context.get("iteration", 0) + 1
                self.send(
                    AgentRole.P1_ARCHITECT,
                    "approved_hypothesis",
                    hyp.to_dict(),
                    thread_id=context.get("thread_id", ""),
                )

        reviewed_count = len(reviews)
        self._pending.clear()
        self._log_action("review_hypotheses", {"count": reviewed_count})
        return {"reviewed": reviewed_count, "reviews": reviews}

    def _review_hypothesis(self, hyp: FactorHypothesis, context: dict[str, Any]) -> dict[str, Any]:
        """审查单个假设。优先 LLM，回退规则审查。"""
        if self.llm and self.llm.api_key:
            return self._review_via_llm(hyp, context)
        return self._review_via_rules(hyp)

    def _review_via_llm(self, hyp: FactorHypothesis, context: dict[str, Any]) -> dict[str, Any]:
        """通过 LLM 深度推理审查假设。"""
        available_fields = context_fields_description()
        user_prompt = f"""请审查以下因子假设：

方向：{hyp.direction}
直觉：{hyp.intuition}
机制：{hyp.mechanism}
预期行为：{hyp.expected_behavior}
风险因素：{hyp.risk_factors}
伪代码：{hyp.pseudocode}
输入字段：{hyp.input_fields}

可用数据字段：
{available_fields}

请评估这个假设的：
1. 逻辑合理性（经济学直觉是否自洽）
2. 可行性（能否用积木实现）
3. 新颖性（是否与常见因子不同）
4. 潜在风险（数据窥探、过拟合等）
"""

        try:
            result = self.llm.chat_json(R2_SYSTEM_PROMPT, user_prompt, temperature=0.2)
            reviews = result.get("reviews", [])
            if reviews:
                r = reviews[0]
                return {
                    "hypothesis_id": hyp.hypothesis_id,
                    "decision": r.get("decision", "revise"),
                    "strengths": r.get("strengths", []),
                    "weaknesses": r.get("weaknesses", []),
                    "suggestions": r.get("suggestions", ""),
                    "feasibility_score": float(r.get("feasibility_score", 0.5)),
                    "novelty_score": float(r.get("novelty_score", 0.5)),
                    "logic_score": float(r.get("logic_score", 0.5)),
                }
        except Exception as exc:
            logger.warning(f"R2 LLM 审查失败，回退规则审查: {exc}")

        return self._review_via_rules(hyp)

    def _review_via_rules(self, hyp: FactorHypothesis) -> dict[str, Any]:
        """规则引擎审查（回退方案）。"""
        scores = {"feasibility": 0.5, "novelty": 0.5, "logic": 0.5}
        strengths = []
        weaknesses = []

        # 检查直觉是否充分
        if len(hyp.intuition) > 10:
            strengths.append("直觉描述充分")
            scores["logic"] += 0.1
        else:
            weaknesses.append("直觉描述不充分")
            scores["logic"] -= 0.2

        # 检查机制是否清晰
        if len(hyp.mechanism) > 15:
            strengths.append("机制描述清晰")
            scores["logic"] += 0.1
        else:
            weaknesses.append("机制描述模糊")
            scores["logic"] -= 0.1

        # 检查输入字段是否可用
        available = {"close", "open", "high", "low", "volume", "vwap", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv", "adj_factor", "industry"}
        input_available = all(f in available for f in hyp.input_fields)
        if input_available:
            strengths.append("输入字段全部可用")
            scores["feasibility"] = 0.9
        else:
            missing = [f for f in hyp.input_fields if f not in available]
            weaknesses.append(f"输入字段不可用: {missing}")
            scores["feasibility"] -= 0.3

        # 检查风险因素
        if len(hyp.risk_factors) >= 2:
            strengths.append("风险因素考虑充分")
        else:
            weaknesses.append("风险因素考虑不足")

        # 综合决策
        avg_score = sum(scores.values()) / 3
        if avg_score >= 0.65:
            decision = "approve"
        elif avg_score >= 0.45:
            decision = "revise"
        else:
            decision = "reject"

        return {
            "hypothesis_id": hyp.hypothesis_id,
            "decision": decision,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": "请补充机制描述和风险因素" if decision == "revise" else "",
            "feasibility_score": round(min(1.0, max(0.0, scores["feasibility"])), 2),
            "novelty_score": round(min(1.0, max(0.0, scores["novelty"])), 2),
            "logic_score": round(min(1.0, max(0.0, scores["logic"])), 2),
        }


# ═══════════════════════════════════════════════════════════════════
# 4. Programming 团队 —— P1 (架构) + P2 (组装) + P3 (编码)
# ═══════════════════════════════════════════════════════════════════

P1_SYSTEM_PROMPT = """你是一个因子编程架构师（P1角色）。

你的职责是：
1. 将审查通过的因子假设转化为积木编程方案
2. 决定使用哪些 Data/Transform/Combine/Relational/Filter 积木
3. 判断哪些部分可以用标准积木实现，哪些需要定制编码
4. 输出 ProgrammingPlan，交给 P2 组装

输出格式（严格 JSON）：
{
  "programming_plan": {
    "factor_id": "...",
    "block_plan": [
      {"type": "data", "field": "close"},
      {"type": "transform", "op": "delta", "params": {"window": 5}, "input_idx": 0},
      {"type": "transform", "op": "rank", "input_idx": 1}
    ],
    "custom_requests": [],
    "integration_spec": "标准积木链可直接执行"
  }
}"""

P2_SYSTEM_PROMPT = """你是一个积木组装工头（P2角色）。

你的职责是：
1. 根据 P1 的 ProgrammingPlan，将积木描述组装为可执行的积木树
2. 如果有 CustomRequest，分发给 P3 处理
3. 验证积木树的类型一致性
4. 输出完整的积木树 JSON

输出格式（严格 JSON）：
{
  "block_tree": { ... },  // 积木树的序列化形式
  "custom_implementations": { ... },  // P3 产出的定制实现
  "validation_passed": true
}"""


class P1Architect(BaseAgent):
    """P1: 架构师，将假设转化为积木编程方案。"""

    def __init__(self, bus: MessageBus, llm_client: LLMClient | None = None) -> None:
        super().__init__(AgentRole.P1_ARCHITECT, bus, llm_client)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        # 接收 R2 审查通过的假设
        messages = self.receive("approved_hypothesis")
        if not messages:
            return {"planned": 0}

        plans = []
        for msg in messages:
            hyp = FactorHypothesis.from_dict(msg.payload)
            plan = self._design_plan(hyp, context)
            plans.append(plan)

            # 传递给 P2
            self.send(
                AgentRole.P2_BLOCK_ASSEMBLER,
                "programming_plan",
                {"hypothesis": hyp.to_dict(), "plan": asdict(plan)},
                thread_id=msg.thread_id,
            )

        self._log_action("design_plans", {"count": len(plans)})
        return {"planned": len(plans), "plans": [asdict(p) for p in plans]}

    def _design_plan(self, hyp: FactorHypothesis, context: dict[str, Any]) -> ProgrammingPlan:
        """设计积木编程方案。优先 LLM，回退规则解析。"""
        if self.llm and self.llm.api_key:
            plan = self._design_via_llm(hyp)
            if plan:
                return plan

        # 回退：从伪代码解析
        return self._design_via_pseudocode(hyp)

    def _design_via_llm(self, hyp: FactorHypothesis) -> ProgrammingPlan | None:
        """通过 LLM 设计编程方案。"""
        available_fields = context_fields_description()
        from .blocks import OperatorRegistry
        ops_desc = []
        for category in [OperatorRegistry.TRANSFORM_OPS, OperatorRegistry.COMBINE_OPS, OperatorRegistry.RELATIONAL_OPS]:
            for op_name, op_def in category.items():
                params_str = f" (参数: {op_def.get('params', [])})" if op_def.get("params") else ""
                ops_desc.append(f"- {op_name}: {op_def['desc']}{params_str}")

        user_prompt = f"""请为以下因子假设设计积木编程方案：

方向：{hyp.direction}
直觉：{hyp.intuition}
机制：{hyp.mechanism}
伪代码：{hyp.pseudocode}
输入字段：{hyp.input_fields}

可用数据字段：
{available_fields}

可用积木算子：
{chr(10).join(ops_desc)}

积木类型：
- DataBlock: 数据积木，提供原始字段
- TransformBlock: 变换积木（截面/时序/条件/分组/递推操作）
- CombineBlock: 组合积木（将两个积木合成）
- RelationalBlock: 关系积木（跨资产查表/组内聚合/截面回归）
- FilterBlock: 筛选积木（条件过滤/加权）

请设计 block_plan（积木列表）和 custom_requests（需要定制编码的部分）。
"""

        try:
            result = self.llm.chat_json(P1_SYSTEM_PROMPT, user_prompt, temperature=0.2)
            plan_data = result.get("programming_plan", {})
            return ProgrammingPlan(
                factor_id=hyp.hypothesis_id,
                block_plan=plan_data.get("block_plan", []),
                custom_requests=plan_data.get("custom_requests", []),
                integration_spec=plan_data.get("integration_spec", ""),
            )
        except Exception as exc:
            logger.warning(f"P1 LLM 设计失败，回退伪代码解析: {exc}")
            return None

    def _design_via_pseudocode(self, hyp: FactorHypothesis) -> ProgrammingPlan:
        """从伪代码简单解析积木方案（回退方案）。"""
        block_plan = []
        input_fields = hyp.input_fields or ["close"]

        # 添加数据积木
        for i, field_name in enumerate(input_fields):
            block_plan.append({"type": "data", "field": field_name, "idx": i})

        # 尝试从伪代码提取算子
        next_idx = len(input_fields)
        pseudocode = hyp.pseudocode.lower()

        # 简单的伪代码解析规则
        if "delta" in pseudocode or "差分" in pseudocode or "动量" in pseudocode or "momentum" in pseudocode:
            window = 5
            for w in [5, 10, 20]:
                if str(w) in pseudocode:
                    window = w
                    break
            block_plan.append({"type": "transform", "op": "delta", "params": {"window": window}, "input_idx": 0})
            next_idx += 1

        if "std" in pseudocode or "波动" in pseudocode or "volatility" in pseudocode:
            window = 20
            block_plan.append({"type": "transform", "op": "ts_std", "params": {"window": window}, "input_idx": 0})
            next_idx += 1

        if "rank" in pseudocode or "排名" in pseudocode or "排序" in pseudocode:
            block_plan.append({"type": "transform", "op": "rank", "input_idx": next_idx - 1})
            next_idx += 1

        if "zscore" in pseudocode or "标准化" in pseudocode:
            block_plan.append({"type": "transform", "op": "zscore", "input_idx": next_idx - 1})
            next_idx += 1

        if "group" in pseudocode or "行业" in pseudocode or "中性" in pseudocode:
            block_plan.append({"type": "transform", "op": "group_neutralize", "input_idx": next_idx - 1, "group_key": "industry"})
            next_idx += 1

        # 如果没有任何变换，加一个默认的 rank
        if next_idx <= len(input_fields):
            block_plan.append({"type": "transform", "op": "rank", "input_idx": 0})

        return ProgrammingPlan(
            factor_id=hyp.hypothesis_id,
            block_plan=block_plan,
            custom_requests=[],
            integration_spec="自动从伪代码解析",
        )


class P2BlockAssembler(BaseAgent):
    """P2: 积木组装工头，将方案转为可执行积木树。"""

    def __init__(self, bus: MessageBus, llm_client: LLMClient | None = None) -> None:
        super().__init__(AgentRole.P2_BLOCK_ASSEMBLER, bus, llm_client)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.receive("programming_plan")
        if not messages:
            return {"assembled": 0}

        results = []
        for msg in messages:
            payload = msg.payload
            hyp = FactorHypothesis.from_dict(payload["hypothesis"])
            plan = ProgrammingPlan(
                factor_id=payload["plan"]["factor_id"],
                block_plan=payload["plan"]["block_plan"],
                custom_requests=payload["plan"].get("custom_requests", []),
                integration_spec=payload["plan"].get("integration_spec", ""),
            )

            # 组装积木树
            block_tree, factor_spec = self._assemble(hyp, plan)

            if block_tree:
                hyp.block_tree = block_tree.to_dict()
                # 传递给 T1 回测
                self.send(
                    AgentRole.T1_BACKTESTER,
                    "assembled_factor",
                    {
                        "hypothesis": hyp.to_dict(),
                        "factor_spec": factor_spec.to_dict() if factor_spec else None,
                        "block_tree": block_tree.to_dict(),
                    },
                    thread_id=msg.thread_id,
                )
                results.append({"factor_id": hyp.hypothesis_id, "status": "assembled"})
            else:
                results.append({"factor_id": hyp.hypothesis_id, "status": "assembly_failed"})

        self._log_action("assemble_blocks", {"count": len(results)})
        return {"assembled": len(results), "results": results}

    def _assemble(self, hyp: FactorHypothesis, plan: ProgrammingPlan) -> tuple[Block | None, FactorSpec | None]:
        """将 ProgrammingPlan 组装为积木树和 FactorSpec。"""
        try:
            blocks: dict[int, Block] = {}

            for item in plan.block_plan:
                idx = item.get("idx", len(blocks))
                btype = item.get("type", "")

                if btype == "data":
                    blocks[idx] = data(item["field"])

                elif btype == "transform":
                    input_idx = item.get("input_idx", 0)
                    input_block = blocks.get(input_idx)
                    if input_block is None:
                        continue
                    op = item.get("op", "rank")
                    params = item.get("params", {})
                    group_key = item.get("group_key")
                    blocks[idx] = transform(op, input_block, group_key=group_key, **params)

                elif btype == "combine":
                    left_idx = item.get("left_idx", 0)
                    right_idx = item.get("right_idx", 1)
                    left_block = blocks.get(left_idx)
                    right_block = blocks.get(right_idx)
                    if left_block is None or right_block is None:
                        continue
                    op = item.get("op", "sub")
                    cond_block = None
                    if "cond_idx" in item:
                        cond_block = blocks.get(item["cond_idx"])
                    blocks[idx] = combine(op, left_block, right_block, cond=cond_block)

            if not blocks:
                return None, None

            # 取最后一个积木作为根
            root = blocks[max(blocks.keys())]

            # 构建 FactorSpec
            direction_map = {
                "momentum": FactorDirection.HIGHER_IS_BETTER,
                "reversal": FactorDirection.LOWER_IS_BETTER,
                "volatility": FactorDirection.LOWER_IS_BETTER,
            }
            factor_dir = FactorDirection.UNKNOWN
            dir_lower = hyp.direction.lower()
            for key, fd in direction_map.items():
                if key in dir_lower:
                    factor_dir = fd
                    break

            spec = FactorSpec(
                factor_id=hyp.hypothesis_id,
                name=f"{hyp.direction}_factor",
                version="v1",
                description=hyp.intuition,
                hypothesis=hyp.mechanism,
                family=_match_family(hyp.direction),
                direction=factor_dir,
                status=FactorStatus.CANDIDATE,
                tags=["multi_agent", "auto_generated"],
            )

            return root, spec

        except Exception as exc:
            logger.warning(f"积木组装失败: {exc}")
            return None, None


class P3CustomCoder(BaseAgent):
    """P3: 定制编码匠人，处理无法用标准积木实现的部分。集成 LLM 代码生成。"""

    def __init__(self, bus: MessageBus, llm_client: LLMClient | None = None) -> None:
        super().__init__(AgentRole.P3_CUSTOM_CODER, bus, llm_client)
        self._code_gen = CustomCodeGenerator(llm_client)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.receive("custom_request")
        market_df = context.get("market_df")
        results = []

        for msg in messages:
            req_payload = msg.payload

            # 尝试 LLM 代码生成 + 沙箱执行
            if market_df is not None:
                gen_result = self._code_gen.generate_and_execute(req_payload, market_df)
                results.append({
                    "request_id": req_payload.get("request_id", ""),
                    "status": gen_result.get("status", "deferred"),
                    "code": gen_result.get("code", ""),
                    "description": gen_result.get("description", ""),
                    "error": gen_result.get("error", ""),
                })

                # 如果生成成功，把 factor_values 传递回去
                if gen_result.get("status") == "success" and gen_result.get("factor_values") is not None:
                    self.send(
                        AgentRole.T1_BACKTESTER,
                        "custom_factor_values",
                        {
                            "request_id": req_payload.get("request_id", ""),
                            "factor_values": gen_result["factor_values"],
                        },
                        thread_id=msg.thread_id,
                    )
            else:
                results.append({
                    "request_id": req_payload.get("request_id", ""),
                    "status": "deferred",
                    "note": "无市场数据，定制代码无法执行",
                })

        self._log_action("custom_code_gen", {"count": len(results)})
        return {"coded": len(results), "results": results}


# ═══════════════════════════════════════════════════════════════════
# 5. Testing 团队 —— T1 (回测) + T2 (验证)
# ═══════════════════════════════════════════════════════════════════

class T1Backtester(BaseAgent):
    """T1: 回测运行 Agent，执行因子计算、风险中性化和参数搜索。"""

    def __init__(self, bus: MessageBus, neutralizer: RiskNeutralizer | None = None, param_searcher: ParameterSearcher | None = None) -> None:
        super().__init__(AgentRole.T1_BACKTESTER, bus)
        self.neutralizer = neutralizer or RiskNeutralizer(industry=True, market_cap=True, momentum=False)
        self.param_searcher = param_searcher

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.receive("assembled_factor")
        if not messages:
            return {"tested": 0}

        market_df = context.get("market_df")
        if market_df is None:
            return {"tested": 0, "error": "market_df not provided"}

        results = []
        for msg in messages:
            payload = msg.payload
            block_tree_dict = payload.get("block_tree")
            factor_spec_dict = payload.get("factor_spec")

            # 阶段1：参数搜索（如果启用）
            best_tree = block_tree_dict
            param_search_result = None
            if self.param_searcher and block_tree_dict:
                param_search_result = self.param_searcher.search(
                    block_tree_dict, market_df,
                    factor_id=payload["hypothesis"].get("hypothesis_id", ""),
                )
                if param_search_result.best_params:
                    best_tree = self.param_searcher._apply_params(block_tree_dict, param_search_result.best_params)

            # 阶段2：因子计算 + 风险中性化
            test_result = self._run_backtest(best_tree, factor_spec_dict, market_df)

            # 加入参数搜索结果
            if param_search_result:
                test_result["param_search"] = {
                    "best_params": param_search_result.best_params,
                    "best_ic": param_search_result.best_ic,
                    "best_icir": param_search_result.best_icir,
                    "total_trials": param_search_result.total_trials,
                }

            # 传递给 T2 验证
            self.send(
                AgentRole.T2_VALIDATOR,
                "test_result",
                {
                    "hypothesis_id": payload["hypothesis"].get("hypothesis_id", ""),
                    "factor_spec": factor_spec_dict,
                    "test_result": test_result,
                },
                thread_id=msg.thread_id,
            )
            results.append(test_result)

        self._log_action("run_backtests", {"count": len(results)})
        return {"tested": len(results), "results": results}

    def _run_backtest(
        self,
        block_tree_dict: dict | None,
        factor_spec_dict: dict | None,
        market_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """执行因子回测，含风险中性化。"""
        if not block_tree_dict:
            return {"status": "failed", "error": "block_tree is None"}

        try:
            # 反序列化积木树并执行
            root_block = Block.from_dict(block_tree_dict)
            executor = BlockExecutor()
            factor_values = executor.execute(root_block, market_df)

            if factor_values is None or len(factor_values) == 0:
                return {"status": "failed", "error": "factor_values is empty"}

            # 风险中性化
            risk_exposure = {}
            try:
                neutralized_values, risk_exposure = self.neutralizer.neutralize(factor_values, market_df)
                factor_values = neutralized_values
            except Exception as exc:
                logger.warning(f"风险中性化失败，使用原始值: {exc}")

            # 计算 IC
            ic_metrics = self._compute_ic(factor_values, market_df)

            return {
                "status": "success",
                "ic_mean": ic_metrics.get("ic_mean"),
                "rank_ic_mean": ic_metrics.get("rank_ic_mean"),
                "ic_ir": ic_metrics.get("ic_ir"),
                "coverage": ic_metrics.get("coverage"),
                "factor_count": len(factor_values.dropna()),
                "risk_exposure": risk_exposure,
            }

        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    def _compute_ic(self, factor_values: pd.Series, market_df: pd.DataFrame) -> dict[str, Any]:
        """计算 IC 指标。"""
        try:
            # 构造下一期收益标签
            if "close" not in market_df.columns:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            # 对齐索引
            aligned = market_df.copy()
            aligned["factor"] = factor_values

            # 计算下期收益
            if "ts_code" in aligned.columns:
                asset_col = "ts_code"
            elif "asset" in aligned.columns:
                asset_col = "asset"
            else:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            aligned = aligned.sort_values([asset_col, "date"])
            aligned["fwd_ret_5"] = aligned.groupby(asset_col)["close"].shift(-5) / aligned["close"] - 1

            # 截面 IC
            date_col = "date"
            rank_ics = []
            for date_val, group in aligned.groupby(date_col):
                valid = group[["factor", "fwd_ret_5"]].dropna()
                if len(valid) >= 20:
                    rank_ic = valid["factor"].rank().corr(valid["fwd_ret_5"].rank(), method="pearson")
                    if not np.isnan(rank_ic):
                        rank_ics.append(rank_ic)

            if not rank_ics:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            rank_ic_mean = float(np.mean(rank_ics))
            ic_std = float(np.std(rank_ics))
            ic_ir = rank_ic_mean / ic_std if ic_std > 0 else 0.0
            coverage = len(factor_values.dropna()) / len(factor_values) if len(factor_values) > 0 else 0.0

            return {
                "ic_mean": round(rank_ic_mean, 4),
                "rank_ic_mean": round(rank_ic_mean, 4),
                "ic_ir": round(ic_ir, 4),
                "coverage": round(float(coverage), 4),
            }

        except Exception as exc:
            logger.warning(f"IC 计算失败: {exc}")
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}


class T2Validator(BaseAgent):
    """T2: 结果验证 Agent，综合评估因子是否交付级。"""

    def __init__(self, bus: MessageBus, llm_client: LLMClient | None = None) -> None:
        super().__init__(AgentRole.T2_VALIDATOR, bus, llm_client)

    # 交付阈值
    IC_THRESHOLD_USEFUL = 0.03
    ICIR_THRESHOLD_USEFUL = 0.5
    IC_THRESHOLD_MARGINAL = 0.015
    COVERAGE_THRESHOLD = 0.7

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.receive("test_result")
        if not messages:
            return {"validated": 0}

        verdicts = []
        for msg in messages:
            payload = msg.payload
            test_result = payload.get("test_result", {})
            verdict = self._validate(test_result, payload.get("factor_spec"))
            verdicts.append({
                "hypothesis_id": payload.get("hypothesis_id", ""),
                "verdict": verdict,
                "test_result": test_result,
            })

        self._log_action("validate_factors", {"count": len(verdicts)})
        return {"validated": len(verdicts), "verdicts": verdicts}

    def _validate(self, test_result: dict[str, Any], factor_spec: dict | None = None) -> str:
        """验证因子质量，返回 useful / marginal / useless。"""
        if test_result.get("status") != "success":
            return "useless"

        rank_ic = abs(float(test_result.get("rank_ic_mean", 0.0)))
        ic_ir = abs(float(test_result.get("ic_ir", 0.0)))
        coverage = float(test_result.get("coverage", 0.0))

        if rank_ic >= self.IC_THRESHOLD_USEFUL and ic_ir >= self.ICIR_THRESHOLD_USEFUL and coverage >= self.COVERAGE_THRESHOLD:
            return "useful"
        elif rank_ic >= self.IC_THRESHOLD_MARGINAL or (rank_ic >= self.IC_THRESHOLD_USEFUL and ic_ir < self.ICIR_THRESHOLD_USEFUL):
            return "marginal"
        else:
            return "useless"


# ═══════════════════════════════════════════════════════════════════
# 6. 三团队编排器 —— 主入口
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MultiAgentConfig:
    """多 Agent 协作配置。"""
    max_r1_r2_rounds: int = 2  # R1↔R2 对抗最大轮数
    max_candidates_per_round: int = 3
    llm_temperature_r1: float = 0.7
    llm_temperature_r2: float = 0.2
    require_llm: bool = False  # 是否强制要求 LLM
    store_results: bool = True
    # 增强模块开关
    enable_risk_neutralization: bool = True  # 风险中性化
    enable_param_search: bool = True  # 参数搜索
    enable_experience_loop: bool = True  # 经验学习回路
    enable_orthogonality_guide: bool = True  # 事前正交性引导
    enable_factor_combination: bool = True  # 多因子组合
    enable_custom_code_gen: bool = True  # P3 定制代码生成
    param_search_trials: int = 20  # 参数搜索最大尝试数


class FactorMultiAgentOrchestrator:
    """三团队多 Agent 协作编排器（增强版）。

    工作流：
    1. Research 阶段：R1 生成假设（含经验学习 + 正交性引导）→ R2 审查 → 对抗循环
    2. Programming 阶段：P1 设计方案 → P2 组装积木 → P3 定制编码（含 LLM 代码生成）
    3. Testing 阶段：T1 回测运行（含风险中性化 + 参数搜索）→ T2 结果验证
    4. 组合阶段：多因子组合优化
    5. 经验沉淀：将结果反馈到经验学习回路
    """

    def __init__(
        self,
        config: MultiAgentConfig | None = None,
        llm_client: LLMClient | None = None,
        store: PersistentFactorStore | None = None,
    ) -> None:
        self.config = config or MultiAgentConfig()
        self.store = store or PersistentFactorStore()
        self.bus = MessageBus()
        self.llm = llm_client or LLMClient()

        # 增强模块
        self.experience_loop = ExperienceLoop()
        self.orth_guide = OrthogonalityGuide() if self.config.enable_orthogonality_guide else None
        self.factor_combiner = FactorCombiner() if self.config.enable_factor_combination else None
        param_searcher = ParameterSearcher(
            max_trials=self.config.param_search_trials,
            method="grid",
        ) if self.config.enable_param_search else None

        # 初始化所有 Agent
        self.r1 = R1HypothesisGenerator(self.bus, self.llm)
        self.r1._orth_guide = self.orth_guide
        self.r2 = R2HypothesisReviewer(self.bus, self.llm)
        self.p1 = P1Architect(self.bus, self.llm)
        self.p2 = P2BlockAssembler(self.bus, self.llm)
        self.p3 = P3CustomCoder(self.bus, self.llm)
        self.t1 = T1Backtester(self.bus, param_searcher=param_searcher)
        self.t2 = T2Validator(self.bus, self.llm)

    def run(
        self,
        direction: str,
        market_df: pd.DataFrame,
        config: MultiAgentConfig | None = None,
    ) -> dict[str, Any]:
        """执行完整的三团队协作流程。"""
        cfg = config or self.config
        run_id = f"run_{uuid4().hex[:8]}"
        start_time = time.time()

        logger.info(f"[{run_id}] 三团队协作启动，方向: {direction}")

        # 获取经验上下文（增强版：来自 ExperienceLoop）
        if cfg.enable_experience_loop:
            memory_context = self.experience_loop.get_guidance(direction)
        else:
            memory_context = self._get_memory_context(direction)
        library_context = self._get_library_context()

        all_hypotheses = []
        all_verdicts = []

        # ── Phase 1: Research (R1 + R2 对抗循环) ──
        for r1r2_round in range(cfg.max_r1_r2_rounds):
            logger.info(f"[{run_id}] Research Round {r1r2_round + 1}/{cfg.max_r1_r2_rounds}")

            # R1 生成
            r1_result = self.r1.act({
                "direction": direction,
                "max_candidates": cfg.max_candidates_per_round,
                "memory_context": memory_context,
                "library_context": library_context,
                "orth_guide": self.orth_guide,
                "thread_id": run_id,
                "iteration": r1r2_round,
            })

            # R2 审查
            r2_result = self.r2.act({
                "thread_id": run_id,
                "iteration": r1r2_round,
            })

            approved = [r for r in r2_result.get("reviews", []) if r.get("decision") in ("approve", "revise")]
            logger.info(f"[{run_id}] R1→R2 Round {r1r2_round + 1}: generated={r1_result.get('generated', 0)}, approved={len(approved)}")

            if not approved and r1r2_round > 0:
                break  # 第二轮还是没有 approve，停止

        # ── Phase 2: Programming (P1 → P2 → P3) ──
        p1_result = self.p1.act({"thread_id": run_id})
        p2_result = self.p2.act({"thread_id": run_id})
        p3_result = self.p3.act({"thread_id": run_id})

        logger.info(f"[{run_id}] Programming: planned={p1_result.get('planned', 0)}, assembled={p2_result.get('assembled', 0)}")

        # ── Phase 3: Testing (T1 → T2) ──
        t1_result = self.t1.act({"market_df": market_df, "thread_id": run_id})
        t2_result = self.t2.act({"thread_id": run_id})

        verdicts = t2_result.get("verdicts", [])
        useful_count = sum(1 for v in verdicts if v.get("verdict") == "useful")
        marginal_count = sum(1 for v in verdicts if v.get("verdict") == "marginal")
        useless_count = sum(1 for v in verdicts if v.get("verdict") == "useless")

        logger.info(f"[{run_id}] Testing: useful={useful_count}, marginal={marginal_count}, useless={useless_count}")

        elapsed = round(time.time() - start_time, 2)

        # 收集最终结果
        result = {
            "run_id": run_id,
            "direction": direction,
            "config": asdict(cfg),
            "elapsed_seconds": elapsed,
            "research": {
                "r1_generated": r1_result.get("generated", 0),
                "r2_reviewed": r2_result.get("reviewed", 0),
                "r2_reviews": r2_result.get("reviews", []),
            },
            "programming": {
                "p1_planned": p1_result.get("planned", 0),
                "p2_assembled": p2_result.get("assembled", 0),
            },
            "testing": {
                "t1_tested": t1_result.get("tested", 0),
                "t2_validated": t2_result.get("validated", 0),
                "useful": useful_count,
                "marginal": marginal_count,
                "useless": useless_count,
                "verdicts": verdicts,
            },
            "message_history": self.bus.history_for("broadcast", limit=100),
            "agent_logs": {
                "r1": self.r1._log[-10:],
                "r2": self.r2._log[-10:],
                "p1": self.p1._log[-10:],
                "p2": self.p2._log[-10:],
                "t1": self.t1._log[-10:],
                "t2": self.t2._log[-10:],
            },
        }

        # 保存结果
        if cfg.store_results:
            self._save_run_result(run_id, direction, result)

        return result

    def _get_memory_context(self, direction: str) -> dict[str, Any]:
        """获取经验记忆上下文。"""
        try:
            generator = FactorHypothesisGenerator(store=self.store)
            family = generator._match_family(direction)
            return generator._retrieve_memory(family)
        except Exception:
            return {}

    def _get_library_context(self) -> list[dict[str, Any]]:
        """获取因子库上下文。"""
        try:
            entries = self.store.load_library_entries()
            return [
                {
                    "factor_id": e.factor_spec.factor_id,
                    "name": e.factor_spec.name,
                    "family": e.factor_spec.family,
                    "composite_score": e.latest_report.scorecard.composite_score,
                }
                for e in entries[:20]
            ]
        except Exception:
            return []

    def _save_run_result(self, run_id: str, direction: str, result: dict[str, Any]) -> None:
        """保存运行结果到文件。"""
        try:
            from pathlib import Path
            output_dir = Path(__file__).resolve().parents[1] / "assistant_data" / "multi_agent_runs"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{run_id}__{direction[:24].strip().replace(' ', '_')}.json"
            (output_dir / fname).write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"保存运行结果失败: {exc}")


# ═══════════════════════════════════════════════════════════════════
# 7. 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _expression_tree_to_pseudocode(tree: FactorNode | None) -> str:
    """将 FactorNode 表达树转为伪代码。"""
    if tree is None:
        return ""
    parts = []
    if tree.node_type == "feature":
        return str(tree.value or "?")
    if tree.node_type == "constant":
        return str(tree.value or "0")
    children_str = ", ".join(_expression_tree_to_pseudocode(c) for c in tree.children)
    params_str = ""
    if tree.params:
        params_str = ", " + ", ".join(f"{k}={v}" for k, v in tree.params.items())
    return f"{tree.node_type}({children_str}{params_str})"


def _match_family(direction: str) -> str:
    """将研究方向匹配到因子族。"""
    lowered = direction.lower()
    families = {
        "momentum": ["动量", "趋势", "momentum", "trend", "延续"],
        "reversal": ["反转", "均值回归", "reversal", "mean_reversion", "超跌"],
        "volatility": ["波动", "风险", "volatility", "risk", "波动率"],
        "volume_price": ["量价", "换手", "成交量", "volume", "turnover", "资金流"],
        "liquidity": ["流动性", "换手率", "liquidity", "amihud"],
        "fundamental": ["基本面", "财务", "盈利", "估值", "fundamental", "earnings"],
    }
    for family_key, keywords in families.items():
        for kw in keywords:
            if kw in lowered:
                return family_key
    return "generic"


def context_fields_description() -> str:
    """返回可用数据字段的描述。"""
    return """- close: 收盘价
- open: 开盘价
- high: 最高价
- low: 最低价
- volume: 成交量
- vwap: 成交量加权均价
- turnover_rate: 换手率
- pe_ttm: 滚动市盈率
- pb: 市净率
- total_mv: 总市值
- circ_mv: 流通市值
- adj_factor: 复权因子
- industry: 行业分类（申万一级）"""
