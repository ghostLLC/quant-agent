from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantlab.config import BASE_DIR

ASSISTANT_DIR = BASE_DIR / "assistant_data"
KNOWLEDGE_DIR = ASSISTANT_DIR / "knowledge"
MEMORY_DIR = ASSISTANT_DIR / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
MEMORY_SUMMARIES_DIR = MEMORY_DIR / "summaries"
DEFAULT_ASSISTANT_MODEL = "gpt-5.4"
DEFAULT_LLM_BASE_URL = "https://tmlgb.me/v1"
DEFAULT_SYSTEM_PROMPT = """
你是专业量化研究 Agent，不是闲聊助手，也不只是面板操作代理。

你当前运行在“策略研究模式”下。
这个模式的边界是：围绕已有策略与已有研究工具链做规划、验证、比较和结论输出；默认不承担成熟因子发掘、自动生成新因子表达式或自主编写新策略代码的职责。

你的核心职责：
1. 把用户目标翻译成可执行的研究任务，先明确研究目标、约束、数据与策略对象。
2. 主动编排研究流程，优先形成“基线评估 → 参数搜索 → 样本外验证 → 稳定性验证 → 结论与下一步建议”的完整链路。
3. 每次给出结论时，都要说明依据来自哪里：当前配置、实验结果、历史记录、研究记忆或知识库。
4. 你的输出重点不是“代码是否跑通”，而是“研究决策是否可信、边界是否明确、风险是否讲清楚”。
5. 不夸大收益，不做确定性投资承诺；必须同时说明假设、风险、样本局限和后续验证建议。
6. 当信息不足时，先调用可用工具读取、验证或执行，再回答；不要凭空补结论。
7. 为避免上下文膨胀，你要依赖摘要记忆、结构化研究记录和检索结果，而不是反复塞入全部聊天记录。

你的回答规范：
- 先给研究判断，再给关键证据，再给下一步动作。
- 如果本轮已经自动执行了研究步骤，要显式说明完成了哪些环节、哪些结论仍待验证。
- 如果用户请求较宽泛，你要主动收敛成专业研究流程，而不是只返回零散建议。
""".strip()
ENV_FILE_PATH = BASE_DIR / ".env"


def _read_env_file(env_path: Path = ENV_FILE_PATH) -> dict[str, str]:
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        values[key.strip()] = cleaned
    return values


def _pick_first(mapping: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def load_assistant_profile_from_env() -> "AssistantProfile":
    env_file_values = _read_env_file()
    env_values = {**env_file_values, **os.environ}
    return AssistantProfile(
        base_url=_pick_first(env_values, ["ASSISTANT_BASE_URL", "OPENAI_BASE_URL"], DEFAULT_LLM_BASE_URL),
        api_key=_pick_first(env_values, ["ASSISTANT_API_KEY", "OPENAI_API_KEY"], ""),
        model=_pick_first(env_values, ["ASSISTANT_MODEL", "OPENAI_MODEL"], DEFAULT_ASSISTANT_MODEL),
    )


@dataclass
class AssistantProfile:
    model: str = DEFAULT_ASSISTANT_MODEL
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_context_messages: int = 10
    max_knowledge_chunks: int = 6
    max_history_items: int = 8
    summary_trigger_messages: int = 12
    temperature: float = 0.2
    tool_call_limit: int = 6
    memory_dir: Path = field(default_factory=lambda: MEMORY_DIR)
    knowledge_dir: Path = field(default_factory=lambda: KNOWLEDGE_DIR)

    def ensure_dirs(self) -> None:
        for path in [ASSISTANT_DIR, self.memory_dir, SESSIONS_DIR, MEMORY_SUMMARIES_DIR, self.knowledge_dir]:
            path.mkdir(parents=True, exist_ok=True)

