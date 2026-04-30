"""阶段 3: 因子进化搜索。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantlab.factor_discovery.factor_enhancements import ExperienceLoop, OrthogonalityGuide

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


def _record_evolution_experience(store: Any, experience_loop: Any, direction: str, evolution_result: dict[str, Any]) -> None:
    """从模板进化结果中提取真实值记录经验。"""
    try:
        from quantlab.factor_discovery.factor_enhancements import FactorOutcome

        entries = store.load_library_entries()
        approved_ids = set(evolution_result.get("approved_factor_ids", []))
        if not approved_ids:
            return

        for entry in entries:
            fid = entry.factor_spec.factor_id
            if fid not in approved_ids:
                continue

            block_tree_desc = _extract_tree_desc(entry.factor_spec)
            input_fields = [d.field_name for d in (entry.factor_spec.dependencies or [])]
            rank_ic = abs(entry.latest_report.scorecard.rank_ic_mean) if entry.latest_report else 0.0
            ic_ir = abs(entry.latest_report.scorecard.ic_ir) if entry.latest_report else 0.0

            outcome = FactorOutcome(
                outcome_id=f"evo_{fid}",
                direction=direction,
                hypothesis_intuition=entry.factor_spec.description or "",
                mechanism=entry.factor_spec.hypothesis or "",
                pseudocode=str(entry.factor_spec.expression_tree) if entry.factor_spec.expression_tree else "",
                input_fields=input_fields,
                block_tree_desc=block_tree_desc,
                verdict="useful" if rank_ic >= 0.03 else ("marginal" if rank_ic >= 0.015 else "useless"),
                rank_ic=round(rank_ic, 4),
                ic_ir=round(ic_ir, 4),
                coverage=round(float(entry.latest_report.scorecard.coverage) if entry.latest_report else 0.0, 4),
                risk_exposure=getattr(entry.latest_report.scorecard, 'risk_exposure', {}) if entry.latest_report else {},
                run_id=evolution_result.get("run_id", ""),
            )
            experience_loop.record(outcome)
            logger.info("经验记录: %s direction=%s ic=%.4f", fid, direction, rank_ic)
    except Exception as exc:
        logger.warning("经验提取失败: %s", exc)


def _extract_tree_desc(spec: Any) -> str:
    try:
        tree = spec.expression_tree
        if tree is None:
            return "template_unknown"
        if hasattr(tree, 'node_type'):
            parts = []
            q = [tree]
            while q:
                node = q.pop(0)
                if hasattr(node, 'node_type'):
                    parts.append(str(node.node_type) if node.node_type else "?")
                q.extend(getattr(node, 'children', []))
            return "→".join(parts[:8])[:60]
        if isinstance(tree, dict):
            bt = tree.get("block_type", tree.get("type", ""))
            op = tree.get("op", "")
            children = tree.get("children", tree.get("input_blocks", []))
            parts = [bt, op] if bt or op else ["dict"]
            for child in (children if isinstance(children, list) else [children]):
                if isinstance(child, dict):
                    parts.append(child.get("op", child.get("type", "")))
            return "→".join(p for p in parts if p)[:60]
        return str(tree)[:60]
    except Exception:
        return "template_unknown"


class EvolutionStage(PipelineStage):
    """因子进化搜索阶段。"""

    def __init__(
        self,
        experience_loop: ExperienceLoop | None = None,
        orth_guide: OrthogonalityGuide | None = None,
    ) -> None:
        self.experience_loop = experience_loop or ExperienceLoop()
        self.orth_guide = orth_guide or OrthogonalityGuide()

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from quantlab.factor_discovery.evolution import EvolutionConfig, FactorEvolutionLoop
        from quantlab.factor_discovery.runtime import PersistentFactorStore

        store = PersistentFactorStore()
        hub = ctx.load_data()
        if hub.empty:
            return {"status": "skipped", "reason": "数据为空"}

        # Cold-start bootstrap
        entries = store.load_library_entries()
        if not entries:
            logger.info("因子库为空，执行冷启动引导...")
            try:
                from quantlab.factor_discovery.seed_factors import bootstrap_seed_factors
                bootstrap_result = bootstrap_seed_factors(
                    market_df=hub, store=store, experience_loop=self.experience_loop,
                )
                logger.info("冷启动完成: 注入 %d 个种子因子", bootstrap_result.get("injected_count", 0))
            except Exception as exc:
                logger.warning("冷启动引导失败: %s", exc)

        total_approved = 0
        all_results = []

        # Adaptive direction selection
        effective_directions, meta_params = self._select_directions(ctx)

        # LLM multi-agent is the default; template evolution is the fallback
        llm_available = ctx.use_multi_agent and self._llm_available()
        if llm_available:
            logger.info("LLM 已配置，多智能体协作为默认因子发现路径")
        else:
            logger.info("LLM 未配置或已禁用，使用模板进化模式")

        for direction in effective_directions:
            mp = meta_params.get(direction, {"rounds": ctx.evolution_rounds, "candidates": ctx.max_candidates_per_round})
            try:
                result = None
                tried_llm = False
                if llm_available:
                    tried_llm = True
                    try:
                        result = self._run_multi_agent(direction, hub, store, mp)
                    except Exception as exc:
                        logger.warning("多智能体 %s 失败，回退模板进化: %s", direction, exc)
                        result = None

                if result is None:
                    loop = FactorEvolutionLoop(
                        store=store,
                        config=EvolutionConfig(
                            max_rounds=mp["rounds"],
                            candidates_per_round=mp["candidates"],
                        ),
                    )
                    result = loop.run(direction=direction, market_df=hub)
                    _record_evolution_experience(store, self.experience_loop, direction, result)

                approved = result.get("approved_count", 0)
                total_approved += approved
                all_results.append({
                    "direction": direction,
                    "approved": approved,
                    "tried_llm": tried_llm,
                    "total_candidates": result.get("total_candidates", 0),
                    "best_score": result.get("best_score", 0.0),
                    "meta_rounds": mp["rounds"],
                    "meta_candidates": mp["candidates"],
                })
            except Exception as exc:
                logger.warning("方向 %s 进化失败: %s", direction, exc)
                all_results.append({"direction": direction, "error": str(exc)[:200]})

        return {
            "status": "success",
            "new_approved": total_approved,
            "directions": all_results,
            "adaptive_selection": ctx.use_adaptive_directions,
        }

    def _select_directions(self, ctx: PipelineContext) -> tuple[list[str], dict[str, dict]]:
        effective = list(ctx.directions)
        meta_params: dict[str, dict] = {d: {"rounds": ctx.evolution_rounds, "candidates": ctx.max_candidates_per_round} for d in effective}

        if not ctx.use_adaptive_directions:
            return effective, meta_params

        try:
            direction_priorities = []
            for d in ctx.directions:
                guidance = self.experience_loop.get_guidance(d)
                orth_ctx = self.orth_guide.get_orthogonality_context(d)
                total_recorded = guidance.get("total_recorded", 0)
                saturated = d in orth_ctx.get("saturated_directions", [])
                win_rate = 0.0
                if total_recorded > 0:
                    useful_count = sum(1 for o in guidance.get("successful_patterns", [])
                                       if float(o.get("rank_ic", 0)) > 0.015)
                    win_rate = useful_count / total_recorded

                priority = win_rate * 0.5 - (0.5 if saturated else 0.0)
                direction_priorities.append({
                    "direction": d, "priority": priority, "win_rate": win_rate,
                    "total_recorded": total_recorded, "saturated": saturated,
                    "insight": guidance.get("direction_insight", "")[:100],
                })

            active = [dp for dp in direction_priorities if not dp["saturated"] or dp["win_rate"] > 0.1]
            if not active:
                active = direction_priorities
            active.sort(key=lambda x: x["priority"], reverse=True)
            effective = [dp["direction"] for dp in active]

            logger.info("自适应方向选择 (共%d个):", len(effective))
            for dp in active:
                logger.info("  %s: priority=%.3f win_rate=%.2f saturated=%s",
                           dp["direction"], dp["priority"], dp["win_rate"], dp["saturated"])

            for dp in active:
                wr = dp["win_rate"]
                if wr > 0.5:
                    meta_params[dp["direction"]] = {"rounds": ctx.evolution_rounds + 2, "candidates": ctx.max_candidates_per_round + 3}
                elif wr > 0.2:
                    meta_params[dp["direction"]] = {"rounds": ctx.evolution_rounds, "candidates": ctx.max_candidates_per_round}
                elif wr > 0.0:
                    meta_params[dp["direction"]] = {"rounds": max(1, ctx.evolution_rounds - 1), "candidates": ctx.max_candidates_per_round + 2}

        except Exception as exc:
            logger.warning("自适应方向选择失败，回退到默认方向: %s", exc)
            effective = list(ctx.directions)

        return effective, meta_params

    def _llm_available(self) -> bool:
        """Check if LLM is configured and reachable. No experience gate."""
        if not hasattr(self, '_cached_llm_available'):
            try:
                from quantlab.factor_discovery.multi_agent import LLMClient
                llm = LLMClient()
                llm._load_from_env()
                self._cached_llm_available = bool(llm.api_key)
            except Exception:
                self._cached_llm_available = False
        return self._cached_llm_available

    def _run_multi_agent(self, direction: str, hub: pd.DataFrame, store: Any, meta_params: dict) -> dict[str, Any]:
        from quantlab.factor_discovery.multi_agent import (
            FactorMultiAgentOrchestrator, MultiAgentConfig, LLMClient,
        )

        llm = LLMClient()
        llm._load_from_env()
        if not llm.api_key:
            raise RuntimeError("LLM 未配置，无法使用多智能体模式")

        cfg = MultiAgentConfig(
            max_r1_r2_rounds=2,
            max_candidates_per_round=min(meta_params.get("candidates", 5), 3),
            require_llm=True,
            enable_risk_neutralization=True,
            enable_param_search=True,
            enable_experience_loop=True,
            enable_orthogonality_guide=True,
            enable_factor_combination=True,
            enable_custom_code_gen=True,
            enable_knowledge_injection=True,
            param_search_trials=20,
        )

        orchestrator = FactorMultiAgentOrchestrator(config=cfg, llm_client=llm, store=store)
        ma_result = orchestrator.run(direction=direction, market_df=hub)

        testing = ma_result.get("testing", {})
        useful = testing.get("useful", 0)
        marginal = testing.get("marginal", 0)
        verdicts = testing.get("verdicts", [])

        best_score = 0.0
        for v in verdicts:
            tr = v.get("test_result", {})
            score = abs(float(tr.get("rank_ic_mean", 0)))
            if score > best_score:
                best_score = score

        logger.info("多智能体 %s: useful=%d marginal=%d best_ic=%.4f", direction, useful, marginal, best_score)
        return {
            "approved_count": useful + marginal,
            "total_candidates": len(verdicts),
            "best_score": best_score,
            "multi_agent_run_id": ma_result.get("run_id", ""),
        }
