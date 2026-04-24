from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quantlab.assistant.config import MEMORY_DIR, MEMORY_SUMMARIES_DIR, SESSIONS_DIR


class ConversationMemoryStore:
    def __init__(self, session_id: str, base_dir: Path | None = None) -> None:
        self.session_id = session_id
        self.base_dir = base_dir or SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        MEMORY_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        self.session_path = self.base_dir / f"{session_id}.json"
        self.summary_path = MEMORY_SUMMARIES_DIR / f"{session_id}.md"
        if not self.session_path.exists():
            self._write_payload({"session_id": session_id, "messages": [], "summary": ""})

    def load(self) -> dict:
        return json.loads(self.session_path.read_text(encoding="utf-8"))

    def append(self, role: str, content: str, metadata: dict | None = None) -> None:
        payload = self.load()
        payload.setdefault("messages", []).append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_payload(payload)

    def get_recent_messages(self, limit: int = 10) -> list[dict]:
        payload = self.load()
        messages = payload.get("messages", [])
        return messages[-limit:]

    def get_summary(self) -> str:
        payload = self.load()
        summary = payload.get("summary", "")
        if summary:
            return summary
        if self.summary_path.exists():
            return self.summary_path.read_text(encoding="utf-8")
        return ""

    def update_summary(self, summary: str) -> None:
        payload = self.load()
        payload["summary"] = summary.strip()
        self._write_payload(payload)
        self.summary_path.write_text(summary.strip(), encoding="utf-8")

    def build_fallback_summary(self, max_items: int = 8) -> str:
        messages = self.get_recent_messages(limit=max_items)
        if not messages:
            return ""
        lines = ["会话摘要："]
        for message in messages:
            role = "用户" if message["role"] == "user" else "助手"
            lines.append(f"- {role}：{message['content'][:180]}")
        summary = "\n".join(lines)
        self.update_summary(summary)
        return summary

    def maybe_rollup_summary(self, trigger_messages: int = 12) -> str:
        payload = self.load()
        messages = payload.get("messages", [])
        if len(messages) < trigger_messages:
            return payload.get("summary", "")
        latest = messages[-trigger_messages:]
        lines = ["长期记忆摘要："]
        for item in latest:
            role = "用户" if item["role"] == "user" else "助手"
            lines.append(f"- {role}：{item['content'][:150]}")
        summary = "\n".join(lines)
        self.update_summary(summary)
        return summary

    def _write_payload(self, payload: dict) -> None:
        self.session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ResearchMemoryStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (MEMORY_DIR / "research")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.base_dir / "research_memory.json"
        if not self.memory_path.exists():
            self._write_payload({"artifacts": [], "plans": [], "insights": [], "decisions": []})

    def load(self) -> dict[str, list[dict[str, Any]]]:
        payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
        payload.setdefault("artifacts", [])
        payload.setdefault("plans", [])
        payload.setdefault("insights", [])
        payload.setdefault("decisions", [])
        return payload

    def append_plan(
        self,
        goal: str,
        tasks: list[dict[str, Any]],
        rationale: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        decision_context: dict[str, Any] | None = None,
    ) -> None:
        payload = self.load()
        payload.setdefault("plans", []).append(
            {
                "goal": goal,
                "tasks": tasks,
                "rationale": rationale or [],
                "metadata": metadata or {},
                "decision_context": decision_context or {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_payload(payload)

    def append_insight(self, title: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        payload = self.load()
        payload.setdefault("insights", []).append(
            {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_payload(payload)

    def append_artifact(self, artifact_type: str, payload_item: dict[str, Any]) -> None:
        payload = self.load()
        payload.setdefault("artifacts", []).append(
            {
                "artifact_type": artifact_type,
                "payload": payload_item,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_payload(payload)

    def append_decision_record(
        self,
        decision_type: str,
        summary: str,
        evidence: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = self.load()
        payload.setdefault("decisions", []).append(
            {
                "decision_type": decision_type,
                "summary": summary,
                "evidence": evidence or [],
                "metadata": metadata or {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_payload(payload)

    def latest_snapshot(self) -> dict[str, Any]:
        payload = self.load()
        return {
            "latest_plan": payload.get("plans", [])[-1] if payload.get("plans") else None,
            "latest_insight": payload.get("insights", [])[-1] if payload.get("insights") else None,
            "latest_artifact": payload.get("artifacts", [])[-1] if payload.get("artifacts") else None,
            "latest_decision": payload.get("decisions", [])[-1] if payload.get("decisions") else None,
        }

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.memory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

