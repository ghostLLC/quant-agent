from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.assistant.config import MEMORY_DIR

from .models import (
    FactorEvaluationReport,
    FactorExperience,
    FactorLibraryEntry,
    FactorMemorySnapshot,
    FactorNode,
    FactorSpec,
    FactorStatus,
    SandboxValidationResult,
)


class SafeFactorExecutor:
    """结构化表达树的安全执行器。"""

    def validate_spec(self, spec: FactorSpec) -> SandboxValidationResult:
        if spec.execution.sandbox_policy.forbid_raw_python_eval and spec.expression and spec.expression_tree is None:
            return SandboxValidationResult(
                passed=False,
                reasons=["禁止直接执行原始字符串表达式，必须提供 expression_tree。"],
            )
        if spec.expression_tree is None:
            return SandboxValidationResult(
                passed=False,
                reasons=["缺少 expression_tree，无法进入安全执行链路。"],
            )
        reasons: list[str] = []
        node_count, max_depth_seen = self._walk_node(spec.expression_tree, 1, spec, reasons)
        dependency_names = {item.field_name for item in spec.dependencies}
        self._validate_dependency_usage(spec.expression_tree, dependency_names, reasons)
        return SandboxValidationResult(
            passed=not reasons,
            reasons=reasons,
            max_depth_seen=max_depth_seen,
            node_count=node_count,
        )

    def execute(self, spec: FactorSpec, market_df: pd.DataFrame) -> dict[str, Any]:
        validation = self.validate_spec(spec)
        if not validation.passed:
            raise ValueError("expression_tree 未通过安全校验：" + "；".join(validation.reasons))
        if market_df.empty:
            raise ValueError("市场数据为空，无法计算因子。")
        prepared = self._prepare_market_frame(market_df)

        # 统一通过 BlockExecutor 执行（FactorNode → Block 转换后计算）
        from quantlab.factor_discovery.blocks import factor_node_to_block, BlockExecutor
        block_tree = factor_node_to_block(spec.expression_tree)
        block_executor = BlockExecutor(date_col="date", asset_col="asset")
        values = block_executor.execute(block_tree, prepared)

        base_columns = [column for column in ["date", "asset", "close", "volume", "industry", "market_cap"] if column in prepared.columns]
        panel = prepared[base_columns].copy()
        panel["factor_value"] = pd.to_numeric(values, errors="coerce")
        panel = self._postprocess_panel(panel, spec)
        return {
            "factor_panel": panel,
            "tree_depth": validation.max_depth_seen,
            "node_count": validation.node_count,
            "coverage": round(float(panel["factor_value"].notna().mean()), 6) if not panel.empty else 0.0,
        }

    def _prepare_market_frame(self, market_df: pd.DataFrame) -> pd.DataFrame:
        frame = market_df.copy()
        required_columns = {"date", "asset"}
        if not required_columns.issubset(frame.columns):
            missing = sorted(required_columns - set(frame.columns))
            raise ValueError(f"市场数据缺少必要字段：{missing}")
        frame["date"] = pd.to_datetime(frame["date"])
        frame["asset"] = frame["asset"].astype(str)
        if "industry" not in frame.columns:
            frame["industry"] = "unknown"
        frame = frame.sort_values(["asset", "date"]).reset_index(drop=True)
        numeric_candidates = [column for column in frame.columns if column not in {"date", "asset", "industry"}]
        for column in numeric_candidates:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "market_cap" not in frame.columns and "close" in frame.columns and "volume" in frame.columns:
            frame["market_cap"] = frame["close"].astype(float) * frame["volume"].astype(float)
        return frame

    def _validate_dependency_usage(self, node: FactorNode, dependency_names: set[str], reasons: list[str]) -> None:
        if node.node_type == "feature" and node.value and dependency_names and str(node.value) not in dependency_names:
            reasons.append(f"特征 {node.value} 不在 dependencies 白名单中")
        for child in node.children:
            self._validate_dependency_usage(child, dependency_names, reasons)

    def _walk_node(
        self,
        node: FactorNode,
        depth: int,
        spec: FactorSpec,
        reasons: list[str],
    ) -> tuple[int, int]:
        policy = spec.execution.sandbox_policy
        node_count = 1
        max_depth_seen = depth
        if node.node_type not in policy.allowed_node_types:
            reasons.append(f"node_type={node.node_type} 不在白名单中")
        if depth > policy.max_tree_depth:
            reasons.append(f"表达树深度 {depth} 超过限制 {policy.max_tree_depth}")
        if len(node.children) > policy.max_children_per_node:
            reasons.append(
                f"node_type={node.node_type} 的 children 数量 {len(node.children)} 超过限制 {policy.max_children_per_node}"
            )
        lookback = node.params.get("window") or node.params.get("lookback")
        if isinstance(lookback, int) and lookback > policy.max_lookback_window:
            reasons.append(f"窗口长度 {lookback} 超过限制 {policy.max_lookback_window}")
        for child in node.children:
            child_count, child_depth = self._walk_node(child, depth + 1, spec, reasons)
            node_count += child_count
            max_depth_seen = max(max_depth_seen, child_depth)
        return node_count, max_depth_seen

    def _evaluate_node(self, node: FactorNode, frame: pd.DataFrame) -> pd.Series:
        node_type = node.node_type
        if node_type == "feature":
            column = str(node.value)
            if column not in frame.columns:
                raise ValueError(f"找不到特征列：{column}")
            return pd.to_numeric(frame[column], errors="coerce")
        if node_type == "constant":
            return pd.Series(float(node.value or 0.0), index=frame.index, dtype=float)

        child_series = [self._evaluate_node(child, frame) for child in node.children]

        if node_type == "add":
            return child_series[0] + child_series[1]
        if node_type == "sub":
            return child_series[0] - child_series[1]
        if node_type == "mul":
            return child_series[0] * child_series[1]
        if node_type == "div":
            denominator = child_series[1].replace(0, np.nan)
            return child_series[0] / denominator
        if node_type == "rank":
            return child_series[0].groupby(frame["date"]).rank(pct=True)
        if node_type == "zscore":
            return child_series[0].groupby(frame["date"]).transform(self._zscore)
        if node_type == "delta":
            window = int(node.params.get("window") or node.params.get("lookback") or 1)
            return child_series[0].groupby(frame["asset"]).diff(window)
        if node_type == "lag":
            window = int(node.params.get("window") or node.params.get("lookback") or 1)
            return child_series[0].groupby(frame["asset"]).shift(window)
        if node_type == "mean":
            window = int(node.params.get("window") or node.params.get("lookback") or 5)
            return child_series[0].groupby(frame["asset"]).transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
        if node_type == "std":
            window = int(node.params.get("window") or node.params.get("lookback") or 5)
            return child_series[0].groupby(frame["asset"]).transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).std(ddof=0))
        if node_type == "ts_rank":
            window = int(node.params.get("window") or node.params.get("lookback") or 10)
            return child_series[0].groupby(frame["asset"]).transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).apply(self._rolling_rank, raw=False))
        if node_type == "min":
            if len(child_series) == 1:
                window = int(node.params.get("window") or node.params.get("lookback") or 5)
                return child_series[0].groupby(frame["asset"]).transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).min())
            return pd.concat(child_series, axis=1).min(axis=1)
        if node_type == "max":
            if len(child_series) == 1:
                window = int(node.params.get("window") or node.params.get("lookback") or 5)
                return child_series[0].groupby(frame["asset"]).transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).max())
            return pd.concat(child_series, axis=1).max(axis=1)
        if node_type == "clip":
            lower = float(node.params.get("lower", -3.0))
            upper = float(node.params.get("upper", 3.0))
            return child_series[0].clip(lower=lower, upper=upper)
        raise ValueError(f"暂不支持的 node_type: {node_type}")


    def _postprocess_panel(self, panel: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
        result = panel.copy()
        result["factor_value"] = pd.to_numeric(result["factor_value"], errors="coerce")
        result = self._apply_fillna(result, spec)
        if spec.preprocess.outlier_guard:
            result["factor_value"] = result.groupby("date")["factor_value"].transform(
                lambda series: self._winsorize_series(series, spec.preprocess.winsorize_limit)
            )
        if spec.preprocess.normalization == "zscore":
            result["factor_value"] = result.groupby("date")["factor_value"].transform(self._zscore)
        if "industry" in result.columns and "industry" in spec.preprocess.neutralization:
            result["factor_value"] = result["factor_value"] - result.groupby(["date", "industry"])["factor_value"].transform("mean")
        if "market_cap" in result.columns and "market_cap" in spec.preprocess.neutralization:
            result["factor_value"] = result["factor_value"] - result.groupby("date")["market_cap"].transform(self._zscore).fillna(0.0) * 0.05
        result = result.sort_values(["date", "asset"]).reset_index(drop=True)
        return result

    def _apply_fillna(self, panel: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
        result = panel.copy()
        method = spec.preprocess.fillna_method
        if method == "cross_section_median":
            fill_values = result.groupby("date")["factor_value"].transform("median")
            result["factor_value"] = result["factor_value"].fillna(fill_values)
        elif method == "zero":
            result["factor_value"] = result["factor_value"].fillna(0.0)
        else:
            result["factor_value"] = result.groupby("asset")["factor_value"].ffill().bfill()
        return result

    @staticmethod
    def _winsorize_series(series: pd.Series, limit: float) -> pd.Series:
        clean = series.dropna()
        if clean.empty:
            return series
        median = clean.median()
        mad = (clean - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return series
        bound = limit * 1.4826 * mad
        return series.clip(lower=median - bound, upper=median + bound)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std is None or std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index, dtype=float)
        return (series - series.mean()) / std

    @staticmethod
    def _rolling_rank(values: pd.Series) -> float:
        clean = values.dropna()
        if clean.empty:
            return np.nan
        return float(clean.rank(pct=True).iloc[-1])


class FactorExperienceMemory:
    def __init__(self, experiences: list[FactorExperience] | None = None) -> None:
        self._experiences = experiences or []

    def add_experience(self, experience: FactorExperience) -> None:
        self._experiences.append(experience)

    def summarize_for_factor(self, spec: FactorSpec) -> FactorMemorySnapshot:
        matched = [item for item in self._experiences if item.factor_family == spec.family or spec.family in item.tags]
        successful_patterns = [item.summary for item in matched if item.outcome == "success"][:3]
        failed_patterns = [item.summary for item in matched if item.outcome == "failure"][:3]
        regime_specific_findings = [item.summary for item in matched if item.pattern_type == "regime_finding"][:3]
        rejected_reason_clusters = [item.summary for item in matched if item.pattern_type == "rejected_reason"][:3]
        return FactorMemorySnapshot(
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            regime_specific_findings=regime_specific_findings,
            rejected_reason_clusters=rejected_reason_clusters,
            related_experience_ids=[item.experience_id for item in matched[:10]],
        )

    def export(self) -> list[dict[str, object]]:
        return [asdict(item) for item in self._experiences]


class PersistentFactorStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (MEMORY_DIR / "factor_discovery")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experience_path = self.base_dir / "experience_registry.json"
        self.library_path = self.base_dir / "factor_library.json"
        self.panel_dir = self.base_dir / "panels"
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.experience_path.exists():
            self.experience_path.write_text(json.dumps({"experiences": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.library_path.exists():
            self.library_path.write_text(json.dumps({"entries": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_experiences(self) -> list[FactorExperience]:
        payload = json.loads(self.experience_path.read_text(encoding="utf-8"))
        return [FactorExperience.from_dict(item) for item in payload.get("experiences", []) or []]

    def append_experience(self, experience: FactorExperience) -> None:
        experiences = self.load_experiences()
        experiences = [item for item in experiences if item.experience_id != experience.experience_id]
        experiences.append(experience)
        self.experience_path.write_text(
            json.dumps({"experiences": [asdict(item) for item in experiences]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_library_entries(self) -> list[FactorLibraryEntry]:
        payload = json.loads(self.library_path.read_text(encoding="utf-8"))
        return [FactorLibraryEntry.from_dict(item) for item in payload.get("entries", []) or []]

    def upsert_library_entry(self, entry: FactorLibraryEntry, factor_panel: pd.DataFrame | None = None) -> None:
        entries = [item for item in self.load_library_entries() if item.factor_spec.factor_id != entry.factor_spec.factor_id]
        snapshot_path = entry.panel_snapshot_path
        if factor_panel is not None and not factor_panel.empty:
            snapshot_file = self.panel_dir / f"{entry.factor_spec.factor_id}__{entry.factor_spec.version}.csv"
            factor_panel.to_csv(snapshot_file, index=False, encoding="utf-8")
            snapshot_path = str(snapshot_file)
            entry.panel_snapshot_path = snapshot_path
        entries.append(entry)
        self.library_path.write_text(
            json.dumps({"entries": [item.to_dict() for item in entries]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def build_memory(self) -> FactorExperienceMemory:
        return FactorExperienceMemory(self.load_experiences())

    def library_correlation(self, factor_panel: pd.DataFrame, max_items: int = 8) -> dict[str, float]:
        if factor_panel.empty:
            return {}
        correlations: dict[str, float] = {}
        current = factor_panel[["date", "asset", "factor_value"]].copy()
        current["date"] = pd.to_datetime(current["date"], errors="coerce")
        current["asset"] = current["asset"].astype(str)
        current = current.rename(columns={"factor_value": "candidate_factor"})
        for entry in self.load_library_entries()[:max_items]:
            if not entry.panel_snapshot_path:
                continue
            snapshot_file = Path(entry.panel_snapshot_path)
            if not snapshot_file.exists():
                continue
            history_panel = pd.read_csv(snapshot_file)
            if not {"date", "asset", "factor_value"}.issubset(history_panel.columns):
                continue
            history_panel["date"] = pd.to_datetime(history_panel["date"], errors="coerce")
            history_panel["asset"] = history_panel["asset"].astype(str)
            merged = current.merge(
                history_panel[["date", "asset", "factor_value"]].rename(columns={"factor_value": "library_factor"}),
                on=["date", "asset"],
                how="inner",
            )
            if len(merged) < 20:
                continue
            pair = merged[["candidate_factor", "library_factor"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 20:
                continue
            if pair["candidate_factor"].nunique(dropna=True) <= 1 or pair["library_factor"].nunique(dropna=True) <= 1:
                continue
            corr = pair[["candidate_factor", "library_factor"]].corr().iloc[0, 1]
            if pd.notna(corr):
                correlations[entry.factor_spec.factor_id] = round(float(corr), 4)

        return correlations


    def summarize_library_overlap(self, factor_panel: pd.DataFrame) -> tuple[dict[str, float], float]:
        corr_map = self.library_correlation(factor_panel)
        max_corr = max(corr_map.values()) if corr_map else 0.0
        return corr_map, float(max_corr)

    def get_evolution_tree(self, factor_id: str) -> list[dict[str, Any]]:
        """追溯因子进化链（从当前因子沿 parent_factor_id 回溯至根）。

        返回按时间顺序排列的因子列表（根 → 当前因子）。
        """
        entries = self.load_library_entries()
        entry_map = {e.factor_spec.factor_id: e for e in entries}

        chain: list[dict[str, Any]] = []
        current_id = factor_id
        while current_id:
            entry = entry_map.get(current_id)
            if entry is None:
                break
            chain.append({
                "factor_id": entry.factor_spec.factor_id,
                "name": entry.factor_spec.name,
                "version": entry.factor_spec.version,
                "parent_factor_id": entry.factor_spec.parent_factor_id,
                "family": entry.factor_spec.family,
                "status": str(entry.factor_spec.status),
            })
            next_id = entry.factor_spec.parent_factor_id
            if not next_id or next_id == current_id:
                break
            current_id = next_id

        chain.reverse()
        return chain

    def get_all_roots(self) -> list[dict[str, Any]]:
        """返回所有根因子（无父因子的原始发现）。"""
        entries = self.load_library_entries()
        roots = [
            {
                "factor_id": e.factor_spec.factor_id,
                "name": e.factor_spec.name,
                "version": e.factor_spec.version,
                "family": e.factor_spec.family,
                "status": str(e.factor_spec.status),
            }
            for e in entries
            if not e.factor_spec.parent_factor_id
        ]
        return roots

    def get_library_stats(self) -> dict[str, int]:
        entries = self.load_library_entries()
        stats: dict[str, int] = {}
        for e in entries:
            s = str(e.factor_spec.status)
            stats[s] = stats.get(s, 0) + 1
        stats["total"] = len(entries)
        return stats

    def archive_underperforming(self, min_observe_days: int = 30, min_ic_threshold: float = 0.015) -> dict[str, Any]:
        """Archive OBSERVE factors older than min_observe_days and REJECTED factors.

        Returns summary of archived count and freed panel files.
        """
        from datetime import datetime, timezone
        entries = self.load_library_entries()
        now = datetime.now(timezone.utc)
        archived: list[str] = []
        cleaned_panels = 0

        for entry in entries:
            status = str(entry.factor_spec.status)
            should_archive = False

            if status == "rejected":
                should_archive = True
            elif status == "observe":
                report = entry.latest_report
                if report and hasattr(report, 'evaluated_at') and report.evaluated_at:
                    try:
                        eval_dt = datetime.fromisoformat(str(report.evaluated_at))
                        days_since = (now - eval_dt).days
                        if days_since > min_observe_days:
                            should_archive = True
                    except (ValueError, TypeError):
                        pass
                elif report and hasattr(report, 'scorecard'):
                    ic = abs(getattr(report.scorecard, 'rank_ic_mean', 0) or 0)
                    if ic < min_ic_threshold:
                        should_archive = True

            if should_archive:
                entry.factor_spec.status = FactorStatus.ARCHIVED
                archived.append(entry.factor_spec.factor_id)
                if entry.panel_snapshot_path:
                    panel_file = Path(entry.panel_snapshot_path)
                    if panel_file.exists():
                        try:
                            panel_file.unlink()
                            cleaned_panels += 1
                        except OSError:
                            pass

        if archived:
            entries_to_keep = [e for e in entries if str(e.factor_spec.status) != "archived"]
            self.library_path.write_text(
                json.dumps({"entries": [e.to_dict() for e in entries]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {"archived_count": len(archived), "archived_ids": archived, "panels_cleaned": cleaned_panels}

