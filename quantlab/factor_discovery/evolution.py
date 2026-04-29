"""自主搜索循环 —— 融合 QuantaAlpha 轨迹进化 + FactorMiner Ralph Loop + RD-Agent-Q 两阶段迭代。

核心能力：
1. 多轮自主因子发掘（假设→执行→评估→进化→再假设）
2. trajectory-level mutation 和 crossover
3. 经验记忆自动更新（成功模式 + 失败约束）
4. 因子-评估联合优化循环

设计借鉴：
- QuantaAlpha: trajectory-level mutation/crossover
- FactorMiner: Ralph Loop (Retrieve → Adapt → Learn → Plan → Harvest)
- RD-Agent-Q: Research → Development 两阶段
- AlphaAgent: 正则化探索 + 抗衰减评估
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .hypothesis import FactorHypothesisGenerator, HypothesisCandidate, HypothesisRequest
from .models import (
    FactorDirection,
    FactorEvaluationReport,
    FactorExperience,
    FactorLibraryEntry,
    FactorNode,
    FactorSpec,
    FactorStatus,
)
from .pipeline import FactorDiscoveryOrchestrator
from .runtime import FactorExperienceMemory, PersistentFactorStore, SafeFactorExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 轨迹定义 —— QuantaAlpha trajectory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrajectoryStep:
    """轨迹中的一个步骤。"""
    step_type: str  # "hypothesis" | "execute" | "evaluate" | "decision"
    factor_id: str
    factor_family: str
    composite_score: float | None = None
    rank_ic_mean: float | None = None
    ic_ir: float | None = None
    coverage: float | None = None
    decision: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """一条完整的因子发掘轨迹。"""
    trajectory_id: str
    direction: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    best_score: float = 0.0
    best_factor_id: str | None = None
    total_candidates: int = 0
    approved_count: int = 0
    observed_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "direction": self.direction,
            "steps": [asdict(s) for s in self.steps],
            "best_score": self.best_score,
            "best_factor_id": self.best_factor_id,
            "total_candidates": self.total_candidates,
            "approved_count": self.approved_count,
            "observed_count": self.observed_count,
            "rejected_count": self.rejected_count,
        }


# ---------------------------------------------------------------------------
# 2. 进化策略 —— QuantaAlpha mutation/crossover
# ---------------------------------------------------------------------------

class EvolutionStrategy:
    """进化策略：mutation + crossover。"""

    @staticmethod
    def mutate_tree(tree: FactorNode, mutation_rate: float = 0.3) -> FactorNode:
        """对表达树做随机变异：替换算子、调整窗口、交换子树。"""
        import copy, random

        node = copy.deepcopy(tree)
        if random.random() > mutation_rate:
            return node

        # 收集所有可变异节点
        nodes: list[tuple[FactorNode, FactorNode | None, int | None]] = []
        stack = [(node, None, None)]
        while stack:
            current, parent, child_idx = stack.pop()
            nodes.append((current, parent, child_idx))
            for i, child in enumerate(current.children):
                stack.append((child, current, i))

        if not nodes:
            return node

        # 随机选一个节点变异
        target, parent, child_idx = random.choice(nodes)
        mutation_type = random.choice(["change_window", "change_operator", "swap_children"])

        if mutation_type == "change_window" and target.params:
            # 调整窗口参数
            for key in ["window", "lookback"]:
                if key in target.params:
                    current_val = target.params[key]
                    delta = random.choice([-2, -1, 1, 2, 5])
                    target.params[key] = max(1, current_val + delta)

        elif mutation_type == "change_operator":
            # 替换算子（同类别）
            op_map = {
                "add": ["sub", "mul", "div"],
                "sub": ["add"],
                "mul": ["div"],
                "div": ["mul"],
                "mean": ["std", "min", "max"],
                "std": ["mean"],
                "delta": ["lag"],
                "lag": ["delta"],
                "rank": ["zscore"],
                "zscore": ["rank"],
            }
            alternatives = op_map.get(target.node_type)
            if alternatives:
                target.node_type = random.choice(alternatives)

        elif mutation_type == "swap_children" and len(target.children) >= 2:
            # 交换子节点顺序
            target.children.reverse()

        return node

    @staticmethod
    def crossover_trees(tree_a: FactorNode, tree_b: FactorNode) -> FactorNode:
        """交叉：从 tree_a 为主，替换一个子树为 tree_b 的子树。"""
        import copy, random

        result = copy.deepcopy(tree_a)
        donor_subtrees: list[FactorNode] = []
        stack = [copy.deepcopy(tree_b)]
        while stack:
            node = stack.pop()
            donor_subtrees.append(node)
            stack.extend(node.children)

        if not donor_subtrees:
            return result

        # 找到 result 中的替换点
        target_nodes: list[tuple[FactorNode, FactorNode | None, int | None]] = []
        stack = [(result, None, None)]
        while stack:
            current, parent, child_idx = stack.pop()
            target_nodes.append((current, parent, child_idx))
            for i, child in enumerate(current.children):
                stack.append((child, current, i))

        # 随机选一个非根节点替换
        non_root = [(n, p, i) for n, p, i in target_nodes if p is not None]
        if not non_root:
            return result

        _, parent, child_idx = random.choice(non_root)
        replacement = random.choice(donor_subtrees)
        if parent is not None and child_idx is not None:
            parent.children[child_idx] = copy.deepcopy(replacement)

        return result


# ---------------------------------------------------------------------------
# 3. 自主搜索循环主类
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EvolutionConfig:
    """进化搜索配置。"""
    max_rounds: int = 5
    candidates_per_round: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    score_threshold_approve: float = 0.55
    score_threshold_observe: float = 0.25
    early_stop_rounds: int = 3  # 连续N轮无改善则停止
    diversity_penalty: float = 0.15


class FactorEvolutionLoop:
    """因子自主搜索循环。

    工作流程（Ralph Loop + trajectory 进化）：
    1. Round 0: 初始假设生成
    2. 每轮: 执行 → 评估 → 进化 → 经验更新 → 下一轮假设
    3. 终止条件: 达到最大轮数 / 连续N轮无改善

    借鉴：
    - QuantaAlpha: trajectory-level 进化（mutate/crossover 最优因子）
    - FactorMiner: Ralph Loop (Retrieve → Adapt → Learn)
    - AlphaAgent: 正则化探索 + 抗衰减评估
    - RD-Agent-Q: Research(假设) → Development(执行) 两阶段
    """

    def __init__(
        self,
        orchestrator: FactorDiscoveryOrchestrator | None = None,
        hypothesis_generator: FactorHypothesisGenerator | None = None,
        store: PersistentFactorStore | None = None,
        config: EvolutionConfig | None = None,
    ) -> None:
        self.store = store or PersistentFactorStore()
        self.orchestrator = orchestrator or FactorDiscoveryOrchestrator(store=self.store)
        self.hypothesis_generator = hypothesis_generator or FactorHypothesisGenerator(store=self.store)
        self.config = config or EvolutionConfig()
        self.evolution = EvolutionStrategy()

    def run(
        self,
        direction: str,
        market_df: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> dict[str, Any]:
        """执行完整的进化搜索循环。"""
        cfg = config or self.config
        trajectory = Trajectory(
            trajectory_id=f"traj_{uuid4().hex[:8]}",
            direction=direction,
        )
        best_scores: list[float] = []
        all_round_results: list[dict[str, Any]] = []

        for round_idx in range(cfg.max_rounds):
            logger.info(f"进化搜索 Round {round_idx + 1}/{cfg.max_rounds}")

            # --- Research 阶段：生成候选 ---
            candidates = self._generate_candidates(
                direction=direction,
                round_idx=round_idx,
                trajectory=trajectory,
                cfg=cfg,
            )

            if not candidates:
                logger.info("没有更多候选因子，提前终止。")
                break

            # --- Development 阶段：执行 + 评估 ---
            round_results = self._evaluate_candidates(candidates, market_df)
            all_round_results.append({
                "round": round_idx + 1,
                "candidates": len(candidates),
                "results": round_results,
            })

            # --- Learn 阶段：更新经验 + 统计 ---
            approved, observed, rejected = self._update_from_results(round_results, trajectory)
            round_best = self._get_round_best_score(round_results)
            best_scores.append(round_best)
            trajectory.best_score = max(trajectory.best_score, round_best)

            logger.info(
                f"Round {round_idx + 1} 完成: "
                f"approved={approved}, observed={observed}, rejected={rejected}, "
                f"best_score={round_best:.4f}"
            )

            # --- 早停检查 ---
            if self._should_early_stop(best_scores, cfg.early_stop_rounds):
                logger.info(f"连续 {cfg.early_stop_rounds} 轮无改善，提前终止。")
                break

        return {
            "trajectory": trajectory.to_dict(),
            "rounds": all_round_results,
            "total_candidates": trajectory.total_candidates,
            "approved_count": trajectory.approved_count,
            "observed_count": trajectory.observed_count,
            "rejected_count": trajectory.rejected_count,
            "best_score": trajectory.best_score,
        }

    # -- 内部方法 --

    def _generate_candidates(
        self,
        direction: str,
        round_idx: int,
        trajectory: Trajectory,
        cfg: EvolutionConfig,
    ) -> list[HypothesisCandidate]:
        """每轮生成候选：Round 0 用假设生成器，后续用进化策略。"""
        if round_idx == 0:
            # Round 0: 初始假设
            request = HypothesisRequest(
                research_direction=direction,
                max_candidates=cfg.candidates_per_round,
                diversity_penalty=cfg.diversity_penalty,
            )
            return self.hypothesis_generator.generate(request)

        # 后续轮次：进化已评估的因子
        evolved: list[HypothesisCandidate] = []
        library_entries = self.store.load_library_entries()

        # 从库中取最优因子做 mutation
        scored_entries = sorted(
            library_entries,
            key=lambda e: e.latest_report.scorecard.composite_score or 0.0,
            reverse=True,
        )

        for entry in scored_entries[:cfg.candidates_per_round]:
            if entry.factor_spec.expression_tree is None:
                continue

            # Mutation
            mutated_tree = self.evolution.mutate_tree(
                entry.factor_spec.expression_tree,
                mutation_rate=cfg.mutation_rate,
            )
            mutated_spec = self._clone_spec_with_new_tree(
                entry.factor_spec, mutated_tree, suffix="_mutated"
            )
            evolved.append(HypothesisCandidate(
                spec=mutated_spec,
                rationale=f"从因子 {entry.factor_spec.factor_id} 变异而来",
                family_match=entry.factor_spec.family,
                novelty_score=0.5,
                exploration_bonus=0.1,
            ))

        # Crossover（如果有至少2个因子）
        if len(scored_entries) >= 2 and len(evolved) < cfg.candidates_per_round:
            tree_a = scored_entries[0].factor_spec.expression_tree
            tree_b = scored_entries[1].factor_spec.expression_tree
            if tree_a and tree_b:
                crossed_tree = self.evolution.crossover_trees(tree_a, tree_b)
                crossed_spec = self._clone_spec_with_new_tree(
                    scored_entries[0].factor_spec, crossed_tree, suffix="_crossover"
                )
                evolved.append(HypothesisCandidate(
                    spec=crossed_spec,
                    rationale=f"从 {scored_entries[0].factor_spec.factor_id} × {scored_entries[1].factor_spec.factor_id} 交叉而来",
                    family_match=crossed_spec.family,
                    novelty_score=0.6,
                    exploration_bonus=0.2,
                ))

        # 如果进化候选不够，补充新的假设
        if len(evolved) < cfg.candidates_per_round:
            request = HypothesisRequest(
                research_direction=direction,
                max_candidates=cfg.candidates_per_round - len(evolved),
                diversity_penalty=cfg.diversity_penalty,
            )
            fresh = self.hypothesis_generator.generate(request)
            evolved.extend(fresh)

        return evolved[:cfg.candidates_per_round]

    def _evaluate_candidates(
        self,
        candidates: list[HypothesisCandidate],
        market_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """对候选因子执行计算+评估。"""
        results = []
        executor = SafeFactorExecutor()

        for candidate in candidates:
            spec = candidate.spec
            try:
                # 执行因子计算
                exec_result = executor.execute(spec, market_df)
                factor_panel = exec_result["factor_panel"]

                # 通过 orchestrator 做完整评估
                closed_loop = self.orchestrator.run_closed_loop(
                    spec, market_df=factor_panel
                )
                report = closed_loop["report"]
                scorecard = report.get("scorecard", {})
                composite = float(scorecard.get("composite_score", 0.0) or 0.0)

                # 入库（至少 observe 级别）
                library_entry = closed_loop.get("library_entry")
                if library_entry:
                    self.store.upsert_library_entry(
                        FactorLibraryEntry.from_dict(library_entry),
                        factor_panel=factor_panel,
                    )

                results.append({
                    "factor_id": spec.factor_id,
                    "family": spec.family,
                    "composite_score": composite,
                    "rank_ic_mean": scorecard.get("rank_ic_mean"),
                    "ic_ir": scorecard.get("ic_ir"),
                    "coverage": scorecard.get("coverage"),
                    "decision": report.get("decision"),
                    "decision_reason": report.get("decision_reason"),
                    "status": "success",
                })

            except Exception as exc:
                logger.warning(f"因子 {spec.factor_id} 执行失败: {exc}")
                results.append({
                    "factor_id": spec.factor_id,
                    "family": spec.family,
                    "composite_score": 0.0,
                    "status": "failed",
                    "error": str(exc),
                })

        return results

    def _update_from_results(
        self,
        results: list[dict[str, Any]],
        trajectory: Trajectory,
    ) -> tuple[int, int, int]:
        """从评估结果更新经验和轨迹统计。"""
        approved = observed = rejected = 0

        for r in results:
            trajectory.total_candidates += 1
            decision = r.get("decision", "rejected")
            score = float(r.get("composite_score", 0.0) or 0.0)

            step = TrajectoryStep(
                step_type="evaluate",
                factor_id=r.get("factor_id", ""),
                factor_family=r.get("family", ""),
                composite_score=score,
                rank_ic_mean=r.get("rank_ic_mean"),
                ic_ir=r.get("ic_ir"),
                coverage=r.get("coverage"),
                decision=decision,
            )
            trajectory.steps.append(step)

            if decision == "approved":
                approved += 1
                if score > trajectory.best_score:
                    trajectory.best_score = score
                    trajectory.best_factor_id = r.get("factor_id")
                # 记录成功经验
                self._record_experience(r, "success", "successful_pattern")
            elif decision == "observe":
                observed += 1
                self._record_experience(r, "observe", "observation")
            else:
                rejected += 1
                self._record_experience(r, "failure", "rejected_reason")

        trajectory.approved_count += approved
        trajectory.observed_count += observed
        trajectory.rejected_count += rejected

        return approved, observed, rejected

    def _record_experience(self, result: dict[str, Any], outcome: str, pattern_type: str) -> None:
        """记录经验到持久存储。"""
        family = result.get("family", "generic")
        score = result.get("composite_score", 0.0)
        decision = result.get("decision", "unknown")
        ic = result.get("rank_ic_mean")

        summary_parts = [f"{family}族因子"]
        if ic is not None:
            summary_parts.append(f"RankIC={ic:.4f}")
        summary_parts.append(f"综合评分={score:.4f}")
        summary_parts.append(f"决策={decision}")

        if outcome == "failure":
            error = result.get("error", "")
            if error:
                summary_parts.append(f"失败原因: {error[:50]}")
        elif outcome == "success":
            summary_parts.append("通过评估，可入库")

        experience = FactorExperience(
            experience_id=f"exp_{uuid4().hex[:8]}",
            factor_family=family,
            pattern_type=pattern_type,
            summary="，".join(summary_parts),
            outcome=outcome,
            evidence={k: str(v) for k, v in result.items() if k != "error"},
            tags=[family, pattern_type],
        )
        self.store.append_experience(experience)

    def _clone_spec_with_new_tree(
        self, base: FactorSpec, new_tree: FactorNode, suffix: str = ""
    ) -> FactorSpec:
        """用新表达树克隆一个 FactorSpec。"""
        return FactorSpec(
            factor_id=f"factor_{uuid4().hex[:10]}",
            name=f"{base.name}{suffix}",
            version="v1",
            description=f"从 {base.factor_id} 进化而来",
            hypothesis=base.hypothesis,
            family=base.family,
            direction=base.direction,
            expression_tree=new_tree,
            template_name=f"{base.template_name or 'evolved'}{suffix}",
            dependencies=base.dependencies,
            preprocess=base.preprocess,
            label=base.label,
            universe=base.universe,
            execution=base.execution,
            tags=base.tags + ["evolved"],
            source="evolution_loop",
            created_from="evolution",
            parent_factor_id=base.factor_id,
        )

    @staticmethod
    def _get_round_best_score(results: list[dict[str, Any]]) -> float:
        scores = [float(r.get("composite_score", 0.0) or 0.0) for r in results if r.get("status") == "success"]
        return max(scores) if scores else 0.0

    @staticmethod
    def _should_early_stop(best_scores: list[float], patience: int) -> bool:
        if len(best_scores) < patience:
            return False
        recent = best_scores[-patience:]
        return all(s <= best_scores[-patience - 1] for s in recent)
