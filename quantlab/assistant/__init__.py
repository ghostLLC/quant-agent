from __future__ import annotations

from typing import Any

__all__ = [
    "AssistantToolRuntime",
    "ConversationMemoryStore",
    "LLMSettings",
    "ProjectKnowledgeBase",
    "QuantPanelAssistant",
    "ResearchMemoryStore",
    "ResearchPlanner",
]


def __getattr__(name: str) -> Any:
    if name == "ProjectKnowledgeBase":
        from quantlab.assistant.knowledge_base import ProjectKnowledgeBase

        return ProjectKnowledgeBase
    if name in {"LLMSettings", "QuantPanelAssistant"}:
        from quantlab.assistant.llm import LLMSettings, QuantPanelAssistant

        return {"LLMSettings": LLMSettings, "QuantPanelAssistant": QuantPanelAssistant}[name]
    if name in {"ConversationMemoryStore", "ResearchMemoryStore"}:
        from quantlab.assistant.memory import ConversationMemoryStore, ResearchMemoryStore

        return {"ConversationMemoryStore": ConversationMemoryStore, "ResearchMemoryStore": ResearchMemoryStore}[name]
    if name == "ResearchPlanner":
        from quantlab.assistant.planner import ResearchPlanner

        return ResearchPlanner
    if name == "AssistantToolRuntime":
        from quantlab.assistant.tools import AssistantToolRuntime

        return AssistantToolRuntime
    raise AttributeError(f"module 'quantlab.assistant' has no attribute {name!r}")

