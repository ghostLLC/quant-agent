from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.config import BacktestConfig
from quantlab.factor_discovery import (
    DataHub,
    EvolutionConfig,
    FactorEvolutionLoop,
    FactorHypothesisGenerator,
    FactorSpec,
    HypothesisRequest,
)
from quantlab.pipeline import get_default_strategy_summary, get_strategy_catalog
from quantlab.research.executor import ResearchTaskExecutor
from quantlab.research.models import ResearchTask



class AssistantToolRuntime:
    def __init__(self, config: BacktestConfig, data_path: Path) -> None:
        self.config = config
        self.data_path = data_path
        self.executor = ResearchTaskExecutor(config, data_path)
        self.datahub = DataHub()
        self.hypothesis_generator = FactorHypothesisGenerator()

    def describe_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "view_current_config",
                "description": "查看当前配置、数据路径与默认策略。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "list_strategies",
                "description": "查看当前可用策略列表与默认策略摘要。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "run_single_backtest",
                "description": "运行单次回测并返回核心指标摘要。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "run_grid_experiment",
                "description": "运行参数对比实验，可传入参数网格。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "short_window": {"type": "array", "items": {"type": "integer"}},
                        "long_window": {"type": "array", "items": {"type": "integer"}},
                        "enable_trend_filter": {"type": "array", "items": {"type": "boolean"}},
                        "stop_loss_pct": {"type": "array", "items": {"type": ["number", "null"]}},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "run_train_test_validation",
                "description": "运行训练/测试验证。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "run_walk_forward_validation",
                "description": "运行 Walk-forward 验证。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "run_multi_strategy_compare",
                "description": "比较当前可用策略的研究表现与稳定性。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "review_portfolio_construction",
                "description": "基于多策略研究结果输出组合构建与仓位分配建议。",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "run_factor_discovery",
                "description": "运行因子发掘闭环，可直接传因子提示词或结构化 factor_spec。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "factor_prompt": {"type": "string"},
                        "factor_spec": {
                            "oneOf": [
                                {"type": "object"},
                                {"type": "string"}
                            ]
                        },
                        "data_path": {"type": "string"}
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "refresh_cross_section_data",
                "description": "抓取并更新沪深300真实多资产横截面数据。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data_path": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "max_assets": {"type": "integer"},
                        "index_symbol": {"type": "string"},
                        "pause_seconds": {"type": "number"},
                        "resume": {"type": "boolean"}

                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "generate_factor_hypotheses",
                "description": "用因子假设生成器自动生成一批因子候选，支持族感知多样性和经验记忆引导。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "research_direction": {"type": "string", "description": "研究方向，如'波动率调整动量'、'量价背离'"},
                        "max_candidates": {"type": "integer", "description": "最大候选数，默认5"},
                        "exclude_families": {"type": "array", "items": {"type": "string"}, "description": "排除的因子族"},
                        "focus_features": {"type": "array", "items": {"type": "string"}, "description": "聚焦的特征列"},
                    },
                    "required": ["research_direction"],
                },
            },
            {
                "type": "function",
                "name": "run_factor_evolution",
                "description": "运行因子自主搜索进化循环：多轮假设→执行→评估→进化→再假设。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "description": "搜索方向，如'量价背离'"},
                        "data_path": {"type": "string"},
                        "max_rounds": {"type": "integer", "description": "最大进化轮数，默认5"},
                        "candidates_per_round": {"type": "integer", "description": "每轮候选数，默认5"},
                        "mutation_rate": {"type": "number", "description": "变异率，默认0.3"},
                        "score_threshold_approve": {"type": "number", "description": "通过阈值，默认0.55"},
                    },
                    "required": ["direction"],
                },
            },

            {
                "type": "function",
                "name": "list_experiment_history",
                "description": "列出最近的实验历史摘要。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "get_experiment_detail",
                "description": "读取某个实验的详情。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "string"},
                    },
                    "required": ["experiment_id"],
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "view_current_config":
            return self._view_current_config()
        if name == "list_strategies":
            return self._list_strategies()
        if name == "run_single_backtest":
            return self._run_single_backtest()
        if name == "run_grid_experiment":
            return self._run_grid_experiment(arguments)
        if name == "run_train_test_validation":
            return self._run_train_test_validation()
        if name == "run_walk_forward_validation":
            return self._run_walk_forward_validation()
        if name == "run_multi_strategy_compare":
            return self._run_multi_strategy_compare()
        if name == "review_portfolio_construction":
            return self._review_portfolio_construction()
        if name == "run_factor_discovery":
            return self._run_factor_discovery(arguments)
        if name == "refresh_cross_section_data":
            return self._refresh_cross_section_data(arguments)
        if name == "refresh_market_data":
            return self._refresh_market_data()
        if name == "generate_factor_hypotheses":
            return self._generate_factor_hypotheses(arguments)
        if name == "run_factor_evolution":
            return self._run_factor_evolution(arguments)

        if name == "list_experiment_history":
            return self._list_experiment_history(arguments)
        if name == "get_experiment_detail":
            return self._get_experiment_detail(arguments)
        raise ValueError(f"未知工具：{name}")


    def execute_research_plan(self, tasks: list[ResearchTask]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            result = self.executor.execute(task)
            payload = result.summary.copy()
            if "top_rows" in payload:
                payload["top_rows"] = self._normalize_records(payload.get("top_rows", []))
            if "fold_summary" in payload:
                payload["fold_summary"] = self._normalize_records(payload.get("fold_summary", []))
            if "rows" in payload:
                payload["rows"] = self._normalize_records(payload.get("rows", []))
            payload = self._normalize_factor_payload(payload)
            results.append(
                {
                    "task_type": task.task_type,
                    "strategy_name": task.strategy_name,
                    "summary": self._finalize_result(payload, result.history_path, result.credibility),
                }
            )

        return results

    def _view_current_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["data_path"] = str(self.data_path)
        payload["report_dir"] = str(payload.get("report_dir", ""))
        payload["history_dir"] = str(payload.get("history_dir", ""))
        payload["default_strategy"] = get_default_strategy_summary()
        return payload

    def _list_strategies(self) -> dict[str, Any]:
        return {
            "default_strategy": get_default_strategy_summary(),
            "strategies": get_strategy_catalog(),
        }

    def _run_single_backtest(self) -> dict[str, Any]:
        task = ResearchTask(task_type="single_backtest", data_path=self.data_path)
        result = self.executor.execute(task)
        return self._finalize_result(result.summary, result.history_path, result.credibility)

    def _run_grid_experiment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = ResearchTask(task_type="grid_search", data_path=self.data_path, parameter_grid=self._normalize_grid(arguments))
        result = self.executor.execute(task)
        payload = result.summary.copy()
        payload["top_rows"] = self._normalize_records(payload.get("top_rows", []))
        return self._finalize_result(payload, result.history_path, result.credibility)

    def _run_train_test_validation(self) -> dict[str, Any]:
        task = ResearchTask(task_type="train_test_validation", data_path=self.data_path)
        result = self.executor.execute(task)
        return self._finalize_result(result.summary, result.history_path, result.credibility)

    def _run_walk_forward_validation(self) -> dict[str, Any]:
        task = ResearchTask(task_type="walk_forward_validation", data_path=self.data_path)
        result = self.executor.execute(task)
        payload = result.summary.copy()
        payload["fold_summary"] = self._normalize_records(payload.get("fold_summary", []))
        return self._finalize_result(payload, result.history_path, result.credibility)

    def _run_multi_strategy_compare(self) -> dict[str, Any]:
        task = ResearchTask(task_type="multi_strategy_compare", data_path=self.data_path)
        result = self.executor.execute(task)
        payload = result.summary.copy()
        payload["ranking"] = self._normalize_records(payload.get("ranking", []))
        return self._finalize_result(payload, result.history_path, result.credibility)

    def _review_portfolio_construction(self) -> dict[str, Any]:
        task = ResearchTask(task_type="portfolio_construction_review", data_path=self.data_path)
        result = self.executor.execute(task)
        return self._finalize_result(result.summary, result.history_path, result.credibility)

    def _run_factor_discovery(self, arguments: dict[str, Any]) -> dict[str, Any]:
        factor_prompt = str(arguments.get("factor_prompt", "") or "").strip()
        factor_spec = arguments.get("factor_spec")
        data_path = Path(arguments.get("data_path") or self.data_path)
        metadata: dict[str, Any] = {}
        if factor_prompt:
            metadata["factor_prompt"] = factor_prompt
        if factor_spec:
            metadata["factor_spec"] = self._normalize_factor_spec_argument(factor_spec)
        task = ResearchTask(task_type="factor_discovery", data_path=data_path, metadata=metadata)
        result = self.executor.execute(task)
        payload = self._normalize_factor_payload(result.summary.copy())
        return self._finalize_result(payload, result.history_path, result.credibility)

    def _refresh_market_data(self) -> dict[str, Any]:
        task = ResearchTask(task_type="refresh_market_data", data_path=self.data_path)
        result = self.executor.execute(task)
        return self._finalize_result(result.summary, result.history_path, result.credibility)

    def _refresh_cross_section_data(self, arguments: dict[str, Any]) -> dict[str, Any]:

        data_path = Path(arguments.get("data_path") or self.data_path)
        metadata: dict[str, Any] = {}
        for key in ["start_date", "end_date", "max_assets", "index_symbol", "pause_seconds", "resume"]:

            if key in arguments and arguments.get(key) is not None:
                metadata[key] = arguments.get(key)
        task = ResearchTask(task_type="refresh_cross_section_data", data_path=data_path, metadata=metadata)
        result = self.executor.execute(task)
        return self._finalize_result(result.summary, result.history_path, result.credibility)

    def _list_experiment_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = ResearchTask(
            task_type="experiment_history",
            data_path=self.data_path,
            metadata={"limit": int(arguments.get("limit", 8) or 8)},
        )
        result = self.executor.execute(task)
        payload = result.summary.copy()
        payload["rows"] = self._normalize_records(payload.get("rows", []))
        return payload

    def _get_experiment_detail(self, arguments: dict[str, Any]) -> dict[str, Any]:

        experiment_id = str(arguments.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError("experiment_id 不能为空")
        task = ResearchTask(
            task_type="experiment_detail",
            data_path=self.data_path,
            metadata={"experiment_id": experiment_id},
        )
        return self.executor.execute(task).summary

    def _normalize_factor_spec_argument(self, factor_spec: Any) -> dict[str, Any] | str:
        if isinstance(factor_spec, FactorSpec):
            return factor_spec.to_dict()
        if isinstance(factor_spec, dict):
            return factor_spec
        if isinstance(factor_spec, str):
            stripped = factor_spec.strip()
            if not stripped:
                raise ValueError("factor_spec 不能为空字符串")
            return json.loads(stripped)
        raise ValueError("factor_spec 仅支持 dict、JSON 字符串或 FactorSpec")

    def _normalize_factor_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        if "report" in normalized and isinstance(normalized["report"], dict):
            computed = normalized["report"].get("artifacts", {}).get("computed_artifacts", {})
            if isinstance(computed, dict) and "factor_panel_preview" in computed:
                computed["factor_panel_preview"] = self._normalize_records(computed.get("factor_panel_preview", []))
        return normalized

    def _normalize_grid(self, arguments: dict[str, Any]) -> dict[str, list] | None:
        if not arguments:
            return None
        normalized: dict[str, list] = {}
        for key, value in arguments.items():
            if value:
                normalized[key] = list(value)
        return normalized or None

    def _normalize_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        frame = pd.DataFrame(records)
        if frame.empty:
            return []
        return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))

    def _generate_factor_hypotheses(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """用因子假设生成器生成候选因子。"""
        research_direction = str(arguments.get("research_direction", "")).strip()
        if not research_direction:
            raise ValueError("research_direction 不能为空")

        request = HypothesisRequest(
            research_direction=research_direction,
            max_candidates=int(arguments.get("max_candidates", 5) or 5),
            exclude_families=list(arguments.get("exclude_families") or []),
            focus_features=arguments.get("focus_features") or None,
        )

        candidates = self.hypothesis_generator.generate(request)

        result_candidates = []
        for c in candidates:
            entry = {
                "factor_id": c.spec.factor_id,
                "name": c.spec.name,
                "family": c.family_match,
                "direction": c.spec.direction.value if c.spec.direction else "unknown",
                "novelty_score": c.novelty_score,
                "exploration_bonus": c.exploration_bonus,
                "rationale": c.rationale,
                "hypothesis": c.spec.hypothesis,
                "dependencies": [d.field_name for d in c.spec.dependencies],
            }
            if c.spec.expression_tree:
                entry["expression_root"] = c.spec.expression_tree.node_type
            result_candidates.append(entry)

        return {
            "research_direction": research_direction,
            "candidate_count": len(result_candidates),
            "candidates": result_candidates,
        }

    def _run_factor_evolution(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """运行因子自主搜索进化循环。"""
        direction = str(arguments.get("direction", "")).strip()
        if not direction:
            raise ValueError("direction 不能为空")

        data_path = Path(arguments.get("data_path") or self.data_path)
        market_df = self.datahub.load(data_path, use_cache=True)

        evolution_config = EvolutionConfig(
            max_rounds=int(arguments.get("max_rounds", 5) or 5),
            candidates_per_round=int(arguments.get("candidates_per_round", 5) or 5),
            mutation_rate=float(arguments.get("mutation_rate", 0.3) or 0.3),
            score_threshold_approve=float(arguments.get("score_threshold_approve", 0.55) or 0.55),
        )

        loop = FactorEvolutionLoop(
            hypothesis_generator=self.hypothesis_generator,
            config=evolution_config,
        )

        result = loop.run(
            direction=direction,
            market_df=market_df,
            config=evolution_config,
        )

        # 序列化轨迹信息
        trajectory = result.get("trajectory", {})
        return {
            "direction": direction,
            "total_candidates": result["total_candidates"],
            "approved_count": result["approved_count"],
            "observed_count": result["observed_count"],
            "rejected_count": result["rejected_count"],
            "best_score": result["best_score"],
            "best_factor_id": trajectory.get("best_factor_id"),
            "rounds_completed": len(result.get("rounds", [])),
            "trajectory_id": trajectory.get("trajectory_id"),
        }

    def _finalize_result(
        self,
        payload: dict[str, Any],
        history_path: str | None,
        credibility: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(payload)
        if history_path:
            result["history_path"] = history_path
        if credibility:
            result["credibility_assessment"] = credibility
        return result
