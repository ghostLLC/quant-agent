from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from quantlab.assistant.config import MEMORY_DIR
from quantlab.assistant.evaluator import ResearchDecisionEvaluator
from quantlab.config import BacktestConfig
from quantlab.data.loader import load_cross_section_data, load_price_data, summarize_cross_section_data, summarize_data

from quantlab.factor_discovery import (
    DataHub,
    EvolutionConfig,
    FactorDependency,
    FactorDirection,
    FactorDiscoveryOrchestrator,
    FactorEvolutionLoop,
    FactorHypothesisGenerator,
    FactorMultiAgentOrchestrator,
    FactorNode,
    FactorSpec,
    HypothesisRequest,
    LLMClient,
    MultiAgentConfig,
)

from quantlab.pipeline import (
    get_experiment_detail,
    get_experiment_history,
    refresh_cross_section_data,
    refresh_market_data,
    review_portfolio_construction,
    run_grid_experiment,
    run_multi_strategy_compare,
    run_single_backtest,
    run_train_test_experiment,
    run_walk_forward_experiment,
)

from quantlab.research.models import ResearchTask, ResearchTaskResult
from quantlab.research.protocol import normalize_task_type


class ResearchTaskExecutor:
    def __init__(self, base_config: BacktestConfig, data_path: Path) -> None:
        self.base_config = base_config
        self.data_path = Path(data_path)
        self.evaluator = ResearchDecisionEvaluator()
        self.factor_history_dir = MEMORY_DIR / "factor_discovery_runs"
        self.factor_history_dir.mkdir(parents=True, exist_ok=True)
        self.datahub = DataHub()
        self.hypothesis_generator = FactorHypothesisGenerator()


    def _build_task_result(
        self,
        task: ResearchTask,
        payload: dict,
        history_path: str | None = None,
    ) -> ResearchTaskResult:
        payload = self._make_json_safe(payload)
        credibility = self.evaluator.evaluate_task_result(task.task_type, payload, history_path)
        summary = {**payload, "credibility_assessment": credibility}
        return ResearchTaskResult(
            task=task,
            summary=summary,
            payload=payload,
            history_path=history_path,
            credibility=credibility,
        )

    def execute(self, task: ResearchTask) -> ResearchTaskResult:
        task_type = normalize_task_type(task.task_type)
        active_data_path = Path(task.data_path or self.data_path)
        config = BacktestConfig(**{**asdict(self.base_config), **task.config_overrides})

        if task_type == "single_backtest":
            result, data_summary, _, history_path = run_single_backtest(
                active_data_path,
                config,
                strategy_name=task.strategy_name,
            )
            payload = {
                "data_summary": data_summary,
                "metrics": result.metrics,
                "trade_count": int(len(result.trades)),
            }
            return self._build_task_result(task, payload, str(history_path))

        if task_type == "grid_search":
            summary_df, best_result, data_summary, history_path = run_grid_experiment(
                active_data_path,
                config,
                parameter_grid=task.parameter_grid,
                strategy_name=task.strategy_name,
            )
            payload = {
                "data_summary": data_summary,
                "best_metrics": best_result.metrics,
                "top_rows": summary_df.head(10).to_dict(orient="records"),
                "combination_count": int(len(summary_df)),
            }
            return self._build_task_result(task, payload, str(history_path))

        if task_type == "train_test_validation":
            validation_result, overview, history_path = run_train_test_experiment(
                active_data_path,
                config,
                parameter_grid=task.parameter_grid,
                strategy_name=task.strategy_name,
            )
            payload = {
                "overview": overview,
                "best_params": validation_result["best_params"],
                "test_metrics": validation_result["test_result"].metrics,
                "baseline_test_metrics": validation_result["baseline_test_result"].metrics,
            }
            return self._build_task_result(task, payload, str(history_path))

        if task_type == "walk_forward_validation":
            wf_result, overview, history_path = run_walk_forward_experiment(
                active_data_path,
                config,
                parameter_grid=task.parameter_grid,
                strategy_name=task.strategy_name,
            )
            payload = {
                "overview": overview,
                "average_metrics": wf_result["average_metrics"],
                "stability_summary": wf_result.get("stability_summary", {}),
                "research_summary": wf_result.get("research_summary", {}),
                "fold_summary": wf_result["fold_summary"].head(12).to_dict(orient="records"),
            }
            return self._build_task_result(task, payload, str(history_path))

        if task_type == "multi_strategy_compare":
            compare_result = run_multi_strategy_compare(active_data_path, config)
            payload = {
                "overview": compare_result["overview"],
                "strategy_count": compare_result["strategy_count"],
                "ranking": compare_result["ranking"].to_dict(orient="records"),
                "best_strategy": compare_result["best_strategy"],
            }
            return self._build_task_result(task, payload)

        if task_type == "portfolio_construction_review":
            review_result = review_portfolio_construction(active_data_path, config)
            payload = dict(review_result)
            return self._build_task_result(task, payload)

        if task_type == "refresh_market_data":
            payload = refresh_market_data(active_data_path)
            return self._build_task_result(task, payload)

        if task_type == "refresh_cross_section_data":
            metadata = task.metadata or {}
            payload = refresh_cross_section_data(
                data_path=active_data_path,
                start_date=str(metadata.get("start_date") or "").strip() or None,
                end_date=str(metadata.get("end_date") or "").strip() or None,
                max_assets=int(metadata["max_assets"]) if metadata.get("max_assets") is not None else None,
                index_symbol=str(metadata.get("index_symbol") or "000300"),
                pause_seconds=float(metadata.get("pause_seconds") or 0.2),
                resume=bool(metadata.get("resume", True)),
            )
            return self._build_task_result(task, payload)


        if task_type == "experiment_history":
            metadata = task.metadata or {}
            limit = int(metadata.get("limit", 8) or 8)
            history = get_experiment_history(config)
            rows = history.head(limit).to_dict(orient="records") if not history.empty else []
            payload = {
                "rows": rows,
                "row_count": int(len(rows)),
                "limit": limit,
            }
            return self._build_task_result(task, payload)

        if task_type == "experiment_detail":

            experiment_id = str(task.metadata.get("experiment_id", "")).strip()
            if not experiment_id:
                raise ValueError("experiment_id 不能为空")
            detail = get_experiment_detail(experiment_id, config)
            if detail is None:
                raise ValueError(f"未找到实验 {experiment_id}")
            return self._build_task_result(task, detail)

        if task_type == "factor_discovery":
            payload, history_path = self._run_factor_discovery_task(task, active_data_path)
            return self._build_task_result(task, payload, history_path)

        if task_type == "generate_factor_hypotheses":
            payload = self._run_generate_factor_hypotheses_task(task)
            return self._build_task_result(task, payload)

        if task_type == "factor_evolution":
            payload, history_path = self._run_factor_evolution_task(task, active_data_path)
            return self._build_task_result(task, payload, history_path)

        if task_type == "multi_agent_discovery":
            payload, history_path = self._run_multi_agent_discovery_task(task, active_data_path)
            return self._build_task_result(task, payload, history_path)

        raise ValueError(f"未知研究任务类型：{task_type}")

    def _run_generate_factor_hypotheses_task(self, task: ResearchTask) -> dict[str, Any]:
        metadata = task.metadata or {}
        research_direction = str(metadata.get("research_direction") or metadata.get("factor_prompt") or "").strip()
        if not research_direction:
            raise ValueError("research_direction 不能为空")

        request = HypothesisRequest(
            research_direction=research_direction,
            max_candidates=int(metadata.get("max_candidates", 5) or 5),
            exclude_families=list(metadata.get("exclude_families") or []),
            focus_features=metadata.get("focus_features") or None,
        )
        candidates = self.hypothesis_generator.generate(request)
        return {
            "research_direction": research_direction,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "factor_id": candidate.spec.factor_id,
                    "name": candidate.spec.name,
                    "family": candidate.family_match,
                    "direction": candidate.spec.direction.value if candidate.spec.direction else "unknown",
                    "novelty_score": candidate.novelty_score,
                    "exploration_bonus": candidate.exploration_bonus,
                    "rationale": candidate.rationale,
                    "hypothesis": candidate.spec.hypothesis,
                    "dependencies": [dependency.field_name for dependency in candidate.spec.dependencies],
                    "expression_root": candidate.spec.expression_tree.node_type if candidate.spec.expression_tree else None,
                }
                for candidate in candidates
            ],
        }

    def _run_factor_evolution_task(self, task: ResearchTask, data_path: Path) -> tuple[dict[str, Any], str]:
        metadata = task.metadata or {}
        direction = str(metadata.get("direction") or metadata.get("factor_prompt") or "").strip()
        if not direction:
            raise ValueError("direction 不能为空")

        market_df, market_summary = self._load_factor_market_frame(data_path, allow_proxy=False)
        evolution_config = EvolutionConfig(
            max_rounds=int(metadata.get("max_rounds", 5) or 5),
            candidates_per_round=int(metadata.get("candidates_per_round", 5) or 5),
            mutation_rate=float(metadata.get("mutation_rate", 0.3) or 0.3),
            score_threshold_approve=float(metadata.get("score_threshold_approve", 0.55) or 0.55),
        )
        loop = FactorEvolutionLoop(
            hypothesis_generator=self.hypothesis_generator,
            config=evolution_config,
        )
        result = loop.run(direction=direction, market_df=market_df, config=evolution_config)
        trajectory = dict(result.get("trajectory", {}) or {})
        payload = {
            "direction": direction,
            "data_summary": market_summary,
            "total_candidates": result["total_candidates"],
            "approved_count": result["approved_count"],
            "observed_count": result["observed_count"],
            "rejected_count": result["rejected_count"],
            "best_score": result["best_score"],
            "rounds_completed": len(result.get("rounds", [])),
            "trajectory": trajectory,
            "rounds": result.get("rounds", []),
        }
        history_path = self._save_evolution_history(direction, payload)
        return payload, history_path

    def _run_factor_discovery_task(self, task: ResearchTask, data_path: Path) -> tuple[dict[str, Any], str]:


        spec = self._build_factor_spec(task)
        market_df, market_summary = self._load_factor_market_frame(data_path, allow_proxy=False)

        orchestrator = FactorDiscoveryOrchestrator()
        closed_loop = orchestrator.run_closed_loop(spec, market_df=market_df)
        report = closed_loop["report"]
        scorecard = dict(report.get("scorecard", {}) or {})
        research_summary = {
            "decision": report.get("decision"),
            "decision_reason": report.get("decision_reason"),
            "composite_score": scorecard.get("composite_score"),
            "rank_ic_mean": scorecard.get("rank_ic_mean"),
            "ic_ir": scorecard.get("ic_ir"),
            "coverage": scorecard.get("coverage"),
            "novelty_score": scorecard.get("novelty_score"),
        }
        payload = {
            "data_summary": market_summary,
            "factor_summary": {
                "factor_id": spec.factor_id,
                "name": spec.name,
                "family": spec.family,
                "direction": spec.direction.value,
                "dependencies": [item.field_name for item in spec.dependencies],
                "prompt": str(task.metadata.get("factor_prompt", "") or ""),
            },
            "research_summary": research_summary,
            "plan": closed_loop["plan"],
            "report": report,
            "library_entry": closed_loop["library_entry"],
            "memory_registry": closed_loop["memory_registry"],
        }
        history_path = self._save_factor_history(spec, payload)
        return payload, history_path

    def _build_factor_spec(self, task: ResearchTask) -> FactorSpec:
        payload = task.metadata.get("factor_spec")
        if payload:
            normalized = self._coerce_factor_spec_payload(payload)
            spec = FactorSpec.from_dict(normalized)
            if not spec.factor_id:
                spec.factor_id = f"factor_{uuid4().hex[:10]}"
            if not spec.name:
                spec.name = spec.factor_id
            return spec

        # 用 FactorHypothesisGenerator 替代硬编码模板
        prompt = str(task.metadata.get("factor_prompt", "") or task.task_type).strip()
        request = HypothesisRequest(
            research_direction=prompt,
            max_candidates=1,
        )
        candidates = self.hypothesis_generator.generate(request)


        if candidates:
            # 选综合评分最高的候选
            best = max(candidates, key=lambda c: c.novelty_score + c.exploration_bonus)
            spec = best.spec
            spec.tags = list(set(spec.tags + ["research_executor"]))
            return spec

        # 兜底：如果生成器没有产出，仍用默认动量模板
        factor_id = f"factor_{uuid4().hex[:10]}"
        expression_tree = FactorNode(
            node_type="rank",
            children=[
                FactorNode(
                    node_type="div",
                    children=[
                        FactorNode(node_type="delta", children=[FactorNode(node_type="feature", value="close")], params={"window": 5}),
                        FactorNode(node_type="lag", children=[FactorNode(node_type="feature", value="close")], params={"window": 5}),
                    ],
                )
            ],
        )
        dependencies = [FactorDependency(field_name="close", lookback=5, description="价格序列")]
        return FactorSpec(
            factor_id=factor_id,
            name="短周期价格动量",
            version="v1",
            description="使用近 5 日价格变化相对 5 日前价格的比例构造横截面动量。",
            hypothesis="短期价格动量在横截面上具备一定延续性，可作为基础候选因子。",
            family="momentum",
            direction=FactorDirection.HIGHER_IS_BETTER,
            expression=prompt,
            expression_tree=expression_tree,
            dependencies=dependencies,
            tags=["momentum", "fallback", "research_executor"],
            notes=["由 ResearchTaskExecutor 兜底生成（假设生成器无产出）。"],
        )

    def _coerce_factor_spec_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                raise ValueError("factor_spec 不能为空字符串")
            return json.loads(stripped)
        raise ValueError("factor_spec 仅支持 dict 或 JSON 字符串")

    def _load_factor_market_frame(self, data_path: Path, allow_proxy: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw = pd.read_csv(data_path)
        if {"date", "close", "volume"}.issubset(raw.columns) and (
            "asset" in raw.columns or "ts_code" in raw.columns
        ):
            market_df = load_cross_section_data(data_path)
            summary = summarize_cross_section_data(market_df)
            summary["source_mode"] = "direct_cross_section"
            return market_df, summary

        if not allow_proxy:
            raise ValueError(
                "当前数据不是正式横截面数据，已拒绝使用单资产代理样本继续研究。"
                "请先刷新或指定真实的 hs300_cross_section.csv 后再运行因子发掘/进化任务。"
            )

        single_asset_df = load_price_data(data_path)
        market_df = self._expand_single_asset_frame(single_asset_df)
        summary = {
            **summarize_data(single_asset_df),
            "source_mode": "proxy_cross_section_from_single_asset",
            "asset_count": int(market_df["asset"].nunique()),
            "rows": int(len(market_df)),
            "note": "当前原始样本是单资产行情，这里基于真实时间序列扩展为代理横截面样本，仅用于临时链路验证，不应用于正式研究结论。",
        }
        return market_df, summary



    def _expand_single_asset_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df.empty:
            raise ValueError("价格数据为空，无法构造横截面代理样本。")
        industries = ["bank", "tech", "consumer", "industry", "energy", "healthcare"]
        rows: list[pd.DataFrame] = []
        base = price_df.copy().reset_index(drop=True)
        date_index = np.arange(len(base), dtype=float)

        for index, industry in enumerate(industries):
            asset = f"proxy_{index + 1:02d}"
            price_scale = 1.0 + index * 0.035
            seasonal = 1.0 + 0.025 * np.sin(date_index / (5.0 + index))
            drift = 1.0 + (index - 2) * 0.00045 * date_index
            liquidity_curve = 1.0 + 0.12 * np.cos(date_index / (4.0 + index))
            asset_df = pd.DataFrame(
                {
                    "date": base["date"],
                    "asset": asset,
                    "close": base["close"].to_numpy(dtype=float) * price_scale * seasonal * drift,
                    "volume": np.maximum(base["volume"].to_numpy(dtype=float) * (1.0 + index * 0.18) * liquidity_curve, 1.0),
                    "industry": industry,
                }
            )
            asset_df["market_cap"] = asset_df["close"] * asset_df["volume"] * (800 + index * 120)
            rows.append(asset_df)

        market_df = pd.concat(rows, ignore_index=True)
        market_df["date"] = pd.to_datetime(market_df["date"])
        market_df = market_df.sort_values(["date", "asset"]).reset_index(drop=True)
        return market_df

    def _save_factor_history(self, spec: FactorSpec, payload: dict[str, Any]) -> str:
        target = self.factor_history_dir / f"{spec.factor_id}__{spec.version}.json"
        target.write_text(json.dumps(self._make_json_safe(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(target)

    def _save_evolution_history(self, direction: str, payload: dict[str, Any]) -> str:
        target = self.factor_history_dir / f"evolution__{direction[:24].strip().replace(' ', '_') or 'direction'}__{uuid4().hex[:8]}.json"
        target.write_text(json.dumps(self._make_json_safe(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(target)

    def _run_multi_agent_discovery_task(self, task: ResearchTask, data_path: Path) -> tuple[dict[str, Any], str]:
        """运行三团队多 Agent 协作因子发掘。"""
        metadata = task.metadata or {}
        direction = str(metadata.get("direction") or metadata.get("factor_prompt") or "").strip()
        if not direction:
            raise ValueError("direction 不能为空")

        market_df, market_summary = self._load_factor_market_frame(data_path, allow_proxy=False)

        ma_config = MultiAgentConfig(
            max_r1_r2_rounds=int(metadata.get("max_r1_r2_rounds", 2) or 2),
            max_candidates_per_round=int(metadata.get("max_candidates_per_round", 3) or 3),
        )

        llm = LLMClient()
        orchestrator = FactorMultiAgentOrchestrator(config=ma_config, llm_client=llm)
        result = orchestrator.run(direction=direction, market_df=market_df, config=ma_config)

        payload = {
            "direction": direction,
            "data_summary": market_summary,
            "run_id": result["run_id"],
            "elapsed_seconds": result["elapsed_seconds"],
            "research": result["research"],
            "programming": result["programming"],
            "testing": result["testing"],
        }
        history_path = self._save_evolution_history(direction, payload)
        return payload, history_path

    def _make_json_safe(self, payload: Any) -> Any:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

