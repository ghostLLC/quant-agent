from __future__ import annotations

from quantlab.strategies import DEFAULT_STRATEGY_NAME, list_strategies
from quantlab.research.models import ResearchPlan, ResearchTask


class ResearchPlanner:
    def build_plan(self, user_message: str, data_path: str, strategy_name: str = DEFAULT_STRATEGY_NAME) -> ResearchPlan:
        message = (user_message or "").strip()
        lowered = message.lower()
        available_strategies = [item["name"] for item in list_strategies()]
        selected_strategy = self._select_strategy(lowered, available_strategies, strategy_name)
        tasks: list[ResearchTask] = []
        rationale: list[str] = []
        metadata = {
            "planner_version": "v5",
            "agent_mode": "strategy_research_mode",
            "selected_strategy": selected_strategy,
            "available_strategies": available_strategies,
            "data_path": data_path,
        }

        if self._is_factor_goal(lowered):
            metadata.update(
                {
                    "plan_type": "factor_research_cycle",
                    "agent_mode": "factor_discovery_mode",
                }
            )
            tasks = [
                ResearchTask(
                    task_type="factor_discovery",
                    data_path=data_path,
                    strategy_name=selected_strategy,
                    metadata={"factor_prompt": message},
                )
            ]
            rationale.extend(
                [
                    "用户目标已经切到因子发掘，不该再退回单策略回测链路。",
                    "先走 expression_tree 安全执行、横截面评估、经验库与因子库沉淀的一体化闭环。",
                ]
            )
        elif any(keyword in lowered for keyword in ["组合", "仓位", "资产配置", "allocation", "portfolio"]):
            metadata["plan_type"] = "portfolio_research_cycle"
            tasks = [
                ResearchTask(task_type="multi_strategy_compare", data_path=data_path, strategy_name=selected_strategy),
                ResearchTask(task_type="portfolio_construction_review", data_path=data_path, strategy_name=selected_strategy),
            ]
            rationale.extend(
                [
                    "用户目标已经进入组合层，不应只停留在单策略回测。",
                    "先做多策略比较，再输出组合构建与仓位管理视角的研究评审。",
                ]
            )
        elif any(keyword in lowered for keyword in ["多策略", "策略比较", "比较策略", "strategy compare", "compare"]):
            metadata["plan_type"] = "multi_strategy_compare"
            tasks.append(ResearchTask(task_type="multi_strategy_compare", data_path=data_path, strategy_name=selected_strategy))
            rationale.append("用户关注策略横向比较，优先安排多策略比较任务。")
        elif any(keyword in lowered for keyword in ["从头", "全套", "完整", "系统", "agent", "研究流程"]):
            metadata["plan_type"] = "full_research_cycle"
            tasks = [
                ResearchTask(task_type="single_backtest", data_path=data_path, strategy_name=selected_strategy),
                ResearchTask(task_type="grid_search", data_path=data_path, strategy_name=selected_strategy),
                ResearchTask(task_type="train_test_validation", data_path=data_path, strategy_name=selected_strategy),
                ResearchTask(task_type="walk_forward_validation", data_path=data_path, strategy_name=selected_strategy),
            ]
            rationale.extend(
                [
                    "用户目标偏完整研究闭环，先跑基线、再调参、再做样本外、最后做稳定性验证。",
                    "这种顺序更接近量化 Agent 的标准研究链路，而不是单次工具调用。",
                ]
            )
        elif any(keyword in lowered for keyword in ["walk", "滚动", "稳定性"]):
            metadata["plan_type"] = "walk_forward_only"
            tasks.append(ResearchTask(task_type="walk_forward_validation", data_path=data_path, strategy_name=selected_strategy))
            rationale.append("用户关注稳定性或滚动表现，优先安排 walk-forward 验证。")
        elif any(keyword in lowered for keyword in ["训练", "测试", "样本外"]):
            metadata["plan_type"] = "train_test_only"
            tasks.append(ResearchTask(task_type="train_test_validation", data_path=data_path, strategy_name=selected_strategy))
            rationale.append("用户关注样本内外区分，优先安排训练/测试验证。")
        elif any(keyword in lowered for keyword in ["参数", "扫描", "网格"]):
            metadata["plan_type"] = "grid_search_only"
            tasks.append(ResearchTask(task_type="grid_search", data_path=data_path, strategy_name=selected_strategy))
            rationale.append("用户关注参数筛选，优先安排参数扫描。")
        else:
            metadata["plan_type"] = "baseline_only"
            tasks.append(ResearchTask(task_type="single_backtest", data_path=data_path, strategy_name=selected_strategy))
            rationale.append("默认先做单次回测，快速建立基线结果。")

        return ResearchPlan(goal=message or "执行量化研究任务", tasks=tasks, rationale=rationale, metadata=metadata)

    def build_review_plan(self, research_plan: ResearchPlan, assessment: dict[str, object]) -> ResearchPlan:
        data_path = self._extract_data_path(research_plan)
        selected_strategy = str(research_plan.metadata.get("selected_strategy") or DEFAULT_STRATEGY_NAME)
        task_types = [task.task_type for task in research_plan.tasks]

        if self._is_factor_plan(research_plan):
            limited_task_types = ["factor_discovery"]
        elif self._is_portfolio_goal(research_plan):
            limited_task_types = ["multi_strategy_compare", "portfolio_construction_review"]
        else:
            limited_task_types = ["single_backtest"]
            if "train_test_validation" in task_types:
                limited_task_types.append("train_test_validation")
            elif "walk_forward_validation" in task_types:
                limited_task_types.append("walk_forward_validation")

        tasks = [self._make_task(task_type, data_path, selected_strategy, research_plan) for task_type in limited_task_types]
        rationale = list(research_plan.rationale)
        rationale.append("原计划被标记为 review_required，先降级成更保守的验证链路，避免直接进入高成本自动执行。")

        metadata = {
            **dict(research_plan.metadata),
            "planner_version": "v5",
            "plan_type": "review_limited_cycle",
            "execution_mode": "limited_autopilot",
            "source_plan_type": research_plan.metadata.get("plan_type", "unknown"),
            "source_gate_status": assessment.get("gate_status", "review_required"),
        }
        return ResearchPlan(goal=research_plan.goal, tasks=tasks, rationale=rationale, metadata=metadata)

    def build_recovery_plan(self, research_plan: ResearchPlan, assessment: dict[str, object]) -> ResearchPlan:
        data_path = self._extract_data_path(research_plan)
        selected_strategy = str(research_plan.metadata.get("selected_strategy") or DEFAULT_STRATEGY_NAME)
        fail_reason_codes = [item.get("code", "unknown") for item in assessment.get("fail_reasons", []) if isinstance(item, dict)]

        if self._is_factor_plan(research_plan):
            task_types = ["factor_discovery"]
            plan_type = "factor_research_cycle"
            rationale = list(research_plan.rationale) + [
                "原计划未通过验收后，保留因子发掘主线，重新执行安全执行、评估和入库闭环。",
                "因子链路更依赖结构化 spec 与横截面验证，不适合自动降级回策略回测模式。",
            ]
        elif self._is_portfolio_goal(research_plan):
            task_types = [
                "single_backtest",
                "grid_search",
                "train_test_validation",
                "walk_forward_validation",
                "multi_strategy_compare",
                "portfolio_construction_review",
            ]
            plan_type = "portfolio_full_research_cycle"
            rationale = list(research_plan.rationale) + [
                "原计划未通过验收后，自动补齐单策略验证、多策略比较与组合层评审，形成完整组合研究闭环。",
                "这样可以先建立每个候选策略的可信证据，再进入组合构建与仓位约束讨论。",
            ]
        else:
            task_types = [
                "single_backtest",
                "grid_search",
                "train_test_validation",
                "walk_forward_validation",
            ]
            plan_type = "full_research_cycle"
            rationale = list(research_plan.rationale) + [
                "原计划未通过验收后，自动补齐基线、调参、样本外和稳定性验证，恢复完整研究闭环。",
                "这样可以避免因为只做单点分析而直接得出过早结论。",
            ]

        tasks = [self._make_task(task_type, data_path, selected_strategy, research_plan) for task_type in task_types]
        metadata = {
            **dict(research_plan.metadata),
            "planner_version": "v5",
            "plan_type": plan_type,
            "source_plan_type": research_plan.metadata.get("plan_type", "unknown"),
            "auto_replanned": True,
            "replan_reason": "assessment_failed",
            "source_gate_status": assessment.get("gate_status", "fail"),
            "source_fail_reason_codes": fail_reason_codes,
        }
        return ResearchPlan(goal=research_plan.goal, tasks=tasks, rationale=rationale, metadata=metadata)

    def _is_factor_goal(self, lowered_message: str) -> bool:
        factor_keywords = ["因子", "alpha", "factor", "横截面", "expression_tree", "单因子"]
        return any(keyword in lowered_message for keyword in factor_keywords)

    def _is_factor_plan(self, research_plan: ResearchPlan) -> bool:
        plan_type = str(research_plan.metadata.get("plan_type", ""))
        return plan_type == "factor_research_cycle" or any(task.task_type == "factor_discovery" for task in research_plan.tasks)

    def _select_strategy(self, lowered_message: str, available_strategies: list[str], fallback: str) -> str:
        for strategy_name in available_strategies:
            if strategy_name.lower() in lowered_message:
                return strategy_name
        if "突破" in lowered_message or "channel" in lowered_message:
            return "channel_breakout" if "channel_breakout" in available_strategies else fallback
        return fallback if fallback in available_strategies else DEFAULT_STRATEGY_NAME

    def _extract_data_path(self, research_plan: ResearchPlan) -> str:
        if research_plan.tasks:
            return str(research_plan.tasks[0].data_path)
        return str(research_plan.metadata.get("data_path", ""))

    def _is_portfolio_goal(self, research_plan: ResearchPlan) -> bool:
        plan_type = str(research_plan.metadata.get("plan_type", ""))
        goal = (research_plan.goal or "").lower()
        portfolio_keywords = ["组合", "仓位", "资产配置", "allocation", "portfolio", "多策略", "compare"]
        return plan_type in {"portfolio_research_cycle", "multi_strategy_compare", "portfolio_full_research_cycle"} or any(
            keyword in goal for keyword in portfolio_keywords
        )

    def _make_task(self, task_type: str, data_path: str, strategy_name: str, source_plan: ResearchPlan) -> ResearchTask:
        metadata = {}
        if task_type == "factor_discovery":
            metadata = {"factor_prompt": source_plan.goal}
        return ResearchTask(task_type=task_type, data_path=data_path, strategy_name=strategy_name, metadata=metadata)
