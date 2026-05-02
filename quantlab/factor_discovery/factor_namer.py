"""Factor Naming & Version Manager —— semantic naming + lineage tracking.

FactorNamer: Generates human-readable semantic names for discovered factors.
  - LLM path: Summarizes block tree, operators, and evaluation metrics.
  - Fallback: Structured name from ops, windows, and family.

FactorVersionManager: Manages factor version chains across evolutions.
  - Tracks parent-child relationships and auto-increments versions.
  - Persists to data/scheduler/factor_versions.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from quantlab.config import DATA_DIR

logger = logging.getLogger(__name__)

VERSIONS_PATH: Path = DATA_DIR / "scheduler" / "factor_versions.json"


# ── Factor Namer ────────────────────────────────────────────────

class FactorNamer:
    """Generate semantic factor names from specs and evaluation reports.

    Two paths:
    1. LLM: Ask LLM to summarize block tree + metrics into a concise name.
    2. Fallback: Construct from topological structure (ops/windows/family).
    """

    def generate_name(
        self,
        factor_spec: Any,
        evaluation_report: Any | None = None,
        llm_client: Any | None = None,
    ) -> str:
        """Generate a human-readable semantic name for a factor.

        Args:
            factor_spec: FactorSpec instance with family, direction, expression_tree.
            evaluation_report: Optional FactorEvaluationReport with scorecard metrics.
            llm_client: Optional LLMClient for AI-powered naming.

        Returns:
            Semantic name string, e.g. "momentum_long_20dCloseRank_ic004_20260502_1430"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Try LLM path first
        if llm_client is not None:
            try:
                name = self._llm_name(factor_spec, evaluation_report, llm_client)
                if name:
                    return name
            except Exception as exc:
                logger.debug("LLM 命名失败，使用回退: %s", exc)

        # Fallback: structural name
        return self._fallback_name(factor_spec, evaluation_report, timestamp)

    def _llm_name(
        self,
        factor_spec: Any,
        evaluation_report: Any | None,
        llm_client: Any,
    ) -> str | None:
        """Generate name via LLM summary."""
        family = getattr(factor_spec, "family", "generic")
        direction_value = getattr(factor_spec, "direction", None)
        direction = str(direction_value.value) if hasattr(direction_value, "value") else str(direction_value or "unknown")
        direction_short = "long" if "higher" in direction else ("short" if "lower" in direction else "neutral")

        ops_desc = self._extract_ops_description(factor_spec)

        ic_str = ""
        if evaluation_report is not None:
            sc = getattr(evaluation_report, "scorecard", None)
            if sc is not None:
                ric = getattr(sc, "rank_ic_mean", None) or 0.0
                if ric:
                    ic_str = f"IC={abs(ric):.3f}"

        prompt = (
            f"为以下量化因子生成一个简洁的语义名称（不超过60字符）。\n"
            f"因子家族: {family}\n"
            f"方向: {direction_short}\n"
            f"算子结构: {ops_desc}\n"
            f"评估指标: {ic_str}\n"
            f"\n命名格式: {{family}}_{{direction}}_{{key_feature}}\n"
            f"只返回名称字符串，不加任何解释。"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        try:
            # Use LLMClient.chat if available, otherwise try raw API
            if hasattr(llm_client, "chat"):
                raw = llm_client.chat(prompt).strip()
            elif hasattr(llm_client, "api_key") and llm_client.api_key:
                raw = self._raw_llm_call(llm_client, prompt)
            else:
                return None

            # Clean up: remove quotes, limit length
            name = raw.strip().strip('"').strip("'").strip()
            # Replace spaces with underscores, limit to 80 chars before timestamp
            name = name.replace(" ", "_").replace("-", "_")[:80]
            # Append timestamp
            full = f"{name}_{timestamp}"
            return full
        except Exception:
            return None

    def _raw_llm_call(self, llm_client: Any, prompt: str) -> str:
        """Make a raw LLM API call for naming."""
        import urllib.request

        base_url = getattr(llm_client, "base_url", "").rstrip("/")
        api_key = getattr(llm_client, "api_key", "")
        model = getattr(llm_client, "model", "default")

        if not base_url or not api_key:
            raise RuntimeError("LLM 配置不完整")

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.3,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _fallback_name(
        self,
        factor_spec: Any,
        evaluation_report: Any | None,
        timestamp: str,
    ) -> str:
        """Construct a structural name from block tree without LLM."""
        family = getattr(factor_spec, "family", "generic")
        direction_value = getattr(factor_spec, "direction", None)
        direction = str(direction_value.value) if hasattr(direction_value, "value") else str(direction_value or "unknown")
        direction_short = "long" if "higher" in direction else ("short" if "lower" in direction else "neutral")

        ops_parts = self._extract_op_sequence(factor_spec)

        ic_suffix = ""
        if evaluation_report is not None:
            sc = getattr(evaluation_report, "scorecard", None)
            if sc is not None:
                ric = getattr(sc, "rank_ic_mean", None) or 0.0
                if ric:
                    ic_suffix = f"_ic{abs(ric):.0f}"

        ops_str = "_".join(ops_parts[:4]) if ops_parts else "unknown"
        # Truncate to avoid excessively long names
        ops_str = ops_str[:50]
        name = f"{family}_{direction_short}_{ops_str}{ic_suffix}_{timestamp}"
        return name

    def _extract_ops_description(self, factor_spec: Any) -> str:
        """Extract a human-readable operator description from the factor's expression tree."""
        ops = self._extract_op_sequence(factor_spec)
        return " → ".join(ops) if ops else "unknown"

    def _extract_op_sequence(self, factor_spec: Any) -> list[str]:
        """Walk the expression tree (FactorNode or Block) and extract an ordered op sequence."""
        tree = getattr(factor_spec, "expression_tree", None)
        if tree is None:
            # Try block_tree attribute on the spec (used by seed factors)
            tree = getattr(factor_spec, "block_tree", None)

        if tree is None:
            return []

        parts: list[tuple[int, str]] = []  # (depth, description)

        # Case 1: FactorNode tree (node_type + children)
        if hasattr(tree, "node_type"):
            self._walk_factor_node(tree, 0, parts)
        # Case 2: Block tree (block_type + op + input_block/left/right)
        elif hasattr(tree, "block_type"):
            self._walk_block(tree, 0, parts)
        # Case 3: dict representation
        elif isinstance(tree, dict):
            self._walk_dict(tree, 0, parts)

        # Sort by depth, return descriptions
        parts.sort(key=lambda x: x[0])
        return [p[1] for p in parts]

    def _walk_factor_node(self, node: Any, depth: int, parts: list[tuple[int, str]]) -> None:
        """Walk a FactorNode tree."""
        nt = str(getattr(node, "node_type", "") or "")
        if nt == "feature":
            field = str(getattr(node, "value", "") or "")
            parts.append((depth, f"{field}"))
        elif nt == "constant":
            parts.append((depth, "const"))
        elif nt:
            window = (getattr(node, "params", {}) or {}).get("window", "")
            w_str = f"w{window}" if window else ""
            parts.append((depth, f"{nt}{w_str}"))

        for child in getattr(node, "children", []) or []:
            self._walk_factor_node(child, depth + 1, parts)

    def _walk_block(self, block: Any, depth: int, parts: list[tuple[int, str]]) -> None:
        """Walk a Block tree."""
        bt = str(getattr(block, "block_type", "") or "")

        if bt == "data":
            field = str(getattr(block, "field_name", "") or "")
            parts.append((depth, f"{field}"))
        elif bt == "transform":
            op = str(getattr(block, "op", "") or "")
            params = getattr(block, "params", {}) or {}
            w = params.get("window", "")
            w_str = f"w{w}" if w else ""
            parts.append((depth, f"{op}{w_str}"))
            inp = getattr(block, "input_block", None)
            if inp is not None:
                self._walk_block(inp, depth + 1, parts)
        elif bt == "combine":
            op = str(getattr(block, "op", "") or "")
            parts.append((depth, f"{op}"))
            left = getattr(block, "left", None)
            right = getattr(block, "right", None)
            if left is not None:
                self._walk_block(left, depth + 1, parts)
            if right is not None:
                self._walk_block(right, depth + 1, parts)

    def _walk_dict(self, d: dict, depth: int, parts: list[tuple[int, str]]) -> None:
        """Walk a dict-based tree representation."""
        bt = d.get("block_type", "")
        if bt == "data":
            field = d.get("field_name", "")
            parts.append((depth, f"{field}"))
        elif bt == "transform":
            op = d.get("op", "")
            params = d.get("params", {}) or {}
            w = params.get("window", "")
            w_str = f"w{w}" if w else ""
            parts.append((depth, f"{op}{w_str}"))
            inp = d.get("input_block")
            if isinstance(inp, dict):
                self._walk_dict(inp, depth + 1, parts)
        elif bt == "combine":
            op = d.get("op", "")
            parts.append((depth, f"{op}"))
            for side in ("left", "right"):
                child = d.get(side)
                if isinstance(child, dict):
                    self._walk_dict(child, depth + 1, parts)


# ── Factor Version Manager ──────────────────────────────────────

class FactorVersionManager:
    """Manages factor version chains across evolutionary iterations.

    Persists version records to data/scheduler/factor_versions.json.
    Each factor_id maps to a list of version entries tracking the
    parent chain, version number, and timestamp.

    Version format: "major.minor" (e.g. "1.0", "1.1", "2.0")
    - New factor: "1.0"
    - Updated existing: increment minor (1.0 → 1.1)
    - Explicit bump: major.minor+1 or user-supplied
    """

    def __init__(self, store_path: Path | str | None = None) -> None:
        self.store_path = Path(store_path) if store_path else VERSIONS_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.store_path.exists():
            self.store_path.write_text(
                json.dumps({}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def register(
        self,
        factor_id: str,
        parent_id: str | None = None,
        version_override: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a factor version, returning the assigned version string.

        If the factor_id already has version history, increments the minor version.
        If version_override is provided, uses that instead.

        Args:
            factor_id: Unique factor identifier.
            parent_id: Optional parent factor_id for lineage tracking.
            version_override: Explicit version to use instead of auto-increment.
            metadata: Optional metadata to store with this version entry.

        Returns:
            Assigned version string (e.g. "1.0", "1.1").
        """
        data = self._load()
        chain = data.get(factor_id, [])

        if version_override:
            version = version_override
        elif chain:
            # Increment minor from latest version
            latest = chain[-1].get("version", "1.0")
            parts = latest.split(".")
            major = int(parts[0]) if len(parts) >= 1 else 1
            minor = int(parts[1]) if len(parts) >= 2 else 0
            version = f"{major}.{minor + 1}"
        else:
            version = "1.0"

        entry: dict[str, Any] = {
            "factor_id": factor_id,
            "version": version,
            "parent_id": parent_id,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        chain.append(entry)
        data[factor_id] = chain
        self._save(data)

        logger.info("因子版本注册: %s v%s (parent=%s)", factor_id, version, parent_id or "none")
        return version

    def get_lineage(self, factor_id: str) -> list[dict[str, Any]]:
        """Return the full version lineage from root to current for a factor_id.

        Walks parent_id chain to reconstruct the ancestry tree.

        Returns:
            List of {factor_id, version, parent_id, timestamp} ordered root → current.
        """
        data = self._load()
        all_entries: dict[str, list[dict[str, Any]]] = {}

        # Build lookup: for each factor_id, get its latest version entry
        latest_map: dict[str, dict[str, Any]] = {}
        for fid, chain in data.items():
            if chain:
                latest_map[fid] = chain[-1]

        # Walk parent chain from factor_id to root
        lineage: list[dict[str, Any]] = []
        current_id = factor_id
        visited: set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            entry = latest_map.get(current_id)
            if entry is None:
                # Check if factor_id itself has a chain
                chain = data.get(current_id, [])
                if chain:
                    for e in chain:
                        lineage.append({
                            "factor_id": e.get("factor_id", current_id),
                            "version": e.get("version", "?"),
                            "parent_id": e.get("parent_id"),
                            "timestamp": e.get("timestamp", ""),
                        })
                    break
                else:
                    lineage.append({
                        "factor_id": current_id,
                        "version": "?",
                        "parent_id": None,
                        "timestamp": "",
                    })
                    break

            lineage.append({
                "factor_id": entry.get("factor_id", current_id),
                "version": entry.get("version", "?"),
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp", ""),
            })
            current_id = entry.get("parent_id", "")

        lineage.reverse()
        return lineage

    def get_latest_version(self, factor_id: str) -> str | None:
        """Return the latest version string for a factor_id, or None if unregistered."""
        data = self._load()
        chain = data.get(factor_id, [])
        if not chain:
            return None
        return chain[-1].get("version", "1.0")

    def get_all_versions(self, factor_id: str) -> list[str]:
        """Return all version strings for a factor_id in chronological order."""
        data = self._load()
        chain = data.get(factor_id, [])
        return [e.get("version", "?") for e in chain]

    def list_all(self) -> dict[str, str]:
        """Return {factor_id → latest_version} for all registered factors."""
        data = self._load()
        result: dict[str, str] = {}
        for fid, chain in data.items():
            if chain:
                result[fid] = chain[-1].get("version", "?")
        return result
