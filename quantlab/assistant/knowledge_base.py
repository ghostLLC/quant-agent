from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantlab.assistant.config import KNOWLEDGE_DIR
from quantlab.config import BASE_DIR


_TOKEN_SPLIT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    content: str
    tokens: set[str]

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
        }


class ProjectKnowledgeBase:
    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self.index_path = self.knowledge_dir / "kb_index.json"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def build_default_index(self, config_snapshot: dict | None = None, history_df: pd.DataFrame | None = None) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for file in sorted(self.knowledge_dir.glob("*.md")):
            if file.name.lower() == "readme.md":
                chunks.extend(self._chunk_markdown(file))
        if not chunks:
            readme_path = BASE_DIR / "README.md"
            if readme_path.exists():
                chunks.extend(self._chunk_markdown(readme_path, source_name="README.md"))

        if config_snapshot:
            chunks.append(self._build_config_chunk(config_snapshot))
        if history_df is not None and not history_df.empty:
            chunks.extend(self._build_history_chunks(history_df))
        self._save_index(chunks)
        return chunks

    def load(self) -> list[KnowledgeChunk]:
        if not self.index_path.exists():
            return []
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [
            KnowledgeChunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                title=item["title"],
                content=item["content"],
                tokens=set(item.get("tokens", [])),
            )
            for item in payload
        ]

    def retrieve(self, query: str, limit: int = 6) -> list[dict]:
        chunks = self.load()
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in chunks:
            overlap = len(query_tokens & chunk.tokens)
            length_penalty = max(1.0, math.log(len(chunk.content) + 10, 10))
            score = overlap / length_penalty
            if overlap > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk.to_payload() for _, chunk in scored[:limit]]
        if top_chunks:
            return top_chunks
        return [chunk.to_payload() for chunk in chunks[:limit]]

    def refresh(self, config_snapshot: dict | None = None, history_df: pd.DataFrame | None = None) -> list[KnowledgeChunk]:
        return self.build_default_index(config_snapshot=config_snapshot, history_df=history_df)

    def _save_index(self, chunks: Iterable[KnowledgeChunk]) -> None:
        payload = []
        for chunk in chunks:
            item = chunk.to_payload()
            item["tokens"] = sorted(chunk.tokens)
            payload.append(item)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _chunk_markdown(self, path: Path, source_name: str | None = None) -> list[KnowledgeChunk]:
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=# )|\n(?=## )|\n(?=### )", text)
        chunks: list[KnowledgeChunk] = []
        for index, section in enumerate(sections, start=1):
            content = section.strip()
            if not content:
                continue
            first_line = content.splitlines()[0].lstrip("# ").strip() or f"Section {index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{(source_name or path.name).replace('.', '_')}_{index}",
                    source=source_name or path.name,
                    title=first_line,
                    content=content,
                    tokens=self._tokenize(content),
                )
            )
        return chunks

    def _build_config_chunk(self, config_snapshot: dict) -> KnowledgeChunk:
        content = "当前面板配置快照：\n" + json.dumps(config_snapshot, ensure_ascii=False, indent=2)
        return KnowledgeChunk(
            chunk_id="runtime_config_snapshot",
            source="runtime",
            title="当前面板配置",
            content=content,
            tokens=self._tokenize(content),
        )

    def _build_history_chunks(self, history_df: pd.DataFrame) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        top_df = history_df.head(8).copy()
        for _, row in top_df.iterrows():
            summary = {
                "experiment_id": row.get("experiment_id"),
                "timestamp": row.get("timestamp"),
                "experiment_type": row.get("experiment_type"),
                "primary_metric": row.get("primary_metric"),
                "stability_score": row.get("stability_score"),
                "research_score": row.get("research_score"),
                "notes": row.get("notes"),
            }
            content = "最近实验摘要：\n" + json.dumps(summary, ensure_ascii=False, indent=2)
            chunk_id = f"history_{row.get('experiment_id', len(chunks))}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source="history",
                    title=f"历史实验 {row.get('experiment_id', 'unknown')}",
                    content=content,
                    tokens=self._tokenize(content),
                )
            )
        return chunks

    def _tokenize(self, text: str) -> set[str]:
        tokens = {token.lower() for token in _TOKEN_SPLIT_RE.split(text) if token}
        return {token for token in tokens if len(token) > 1}
