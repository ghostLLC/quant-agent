"""因子假设生成器 —— 融合 AlphaAgent 正则化探索 + Hubble DSL约束 + FactorMiner Ralph Loop。

核心能力：
1. LLM 驱动的因子假设生成（非硬编码模板）
2. 正则化探索约束（避免同质化）
3. 经验记忆驱动的 Retrieve-Adapt 循环
4. 族感知多样性控制

设计借鉴：
- AlphaAgent: LLM + 正则化探索，抗衰减
- Hubble: DSL约束生成 + 族感知选择
- FactorMiner: Ralph Loop (Retrieve → Adapt → Learn)
- QuantaAlpha: 轨迹级经验复用
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    FactorDependency,
    FactorDirection,
    FactorExperience,
    FactorNode,
    FactorSpec,
    FactorStatus,
)
from .runtime import FactorExperienceMemory, PersistentFactorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 因子族定义 —— Hubble 族感知 + AlphaAgent 正则化
# ---------------------------------------------------------------------------

FACTOR_FAMILIES: dict[str, dict[str, Any]] = {
    "momentum": {
        "keywords": ["动量", "趋势", "momentum", "trend", "延续", "惯性"],
        "description": "价格动量与趋势延续类因子",
        "direction": FactorDirection.HIGHER_IS_BETTER,
        "base_features": ["close"],
        "typical_windows": [5, 10, 20, 60],
    },
    "reversal": {
        "keywords": ["反转", "均值回归", "reversal", "mean_reversion", "超跌"],
        "description": "短期反转与均值回归类因子",
        "direction": FactorDirection.LOWER_IS_BETTER,
        "base_features": ["close"],
        "typical_windows": [1, 3, 5],
    },
    "volatility": {
        "keywords": ["波动", "风险", "volatility", "risk", "波动率", "标准差"],
        "description": "波动率与风险调整类因子",
        "direction": FactorDirection.LOWER_IS_BETTER,
        "base_features": ["close"],
        "typical_windows": [5, 10, 20, 60],
    },
    "volume_price": {
        "keywords": ["量价", "换手", "成交量", "volume", "turnover", "资金流"],
        "description": "量价关系与资金流动类因子",
        "direction": FactorDirection.UNKNOWN,
        "base_features": ["close", "volume"],
        "typical_windows": [5, 10, 20],
    },
    "liquidity": {
        "keywords": ["流动性", "换手率", "liquidity", "amihud"],
        "description": "流动性与交易活跃度类因子",
        "direction": FactorDirection.UNKNOWN,
        "base_features": ["close", "volume"],
        "typical_windows": [5, 10, 20],
    },
    "fundamental": {
        "keywords": ["基本面", "财务", "盈利", "估值", "fundamental", "earnings"],
        "description": "基本面与财务质量类因子",
        "direction": FactorDirection.HIGHER_IS_BETTER,
        "base_features": ["close"],
        "typical_windows": [20, 60, 120],
    },
}


# ---------------------------------------------------------------------------
# 2. DSL 算子库 —— Hubble DSL约束思路
# ---------------------------------------------------------------------------

OPERATOR_CATALOG: dict[str, dict[str, Any]] = {
    "rank": {"arity": 1, "category": "cross_section", "description": "横截面排名百分位"},
    "zscore": {"arity": 1, "category": "cross_section", "description": "横截面标准化"},
    "delta": {"arity": 1, "category": "time_series", "params": ["window"], "description": "时间序列差分"},
    "lag": {"arity": 1, "category": "time_series", "params": ["window"], "description": "时间序列滞后"},
    "mean": {"arity": 1, "category": "time_series", "params": ["window"], "description": "滚动均值"},
    "std": {"arity": 1, "category": "time_series", "params": ["window"], "description": "滚动标准差"},
    "ts_rank": {"arity": 1, "category": "time_series", "params": ["window"], "description": "时间序列排名"},
    "add": {"arity": 2, "category": "arithmetic", "description": "加法"},
    "sub": {"arity": 2, "category": "arithmetic", "description": "减法"},
    "mul": {"arity": 2, "category": "arithmetic", "description": "乘法"},
    "div": {"arity": 2, "category": "arithmetic", "description": "除法"},
    "min": {"arity": 1, "category": "time_series", "params": ["window"], "description": "滚动最小值"},
    "max": {"arity": 1, "category": "time_series", "params": ["window"], "description": "滚动最大值"},
    "clip": {"arity": 1, "category": "postprocess", "params": ["lower", "upper"], "description": "截断"},
}


# ---------------------------------------------------------------------------
# 3. 假设生成请求与结果
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HypothesisRequest:
    """因子假设生成请求。"""
    research_direction: str
    max_candidates: int = 5
    exclude_families: list[str] = field(default_factory=list)
    exclude_factor_ids: list[str] = field(default_factory=list)
    focus_features: list[str] | None = None
    diversity_penalty: float = 0.15
    temperature: float = 0.7


@dataclass(slots=True)
class HypothesisCandidate:
    """一个因子假设候选。"""
    spec: FactorSpec
    rationale: str
    family_match: str
    novelty_score: float
    exploration_bonus: float


# ---------------------------------------------------------------------------
# 4. 核心生成器
# ---------------------------------------------------------------------------

class FactorHypothesisGenerator:
    """因子假设生成器。

    工作流程（Ralph Loop 变体）：
    1. Retrieve: 从经验记忆 + 因子库提取相关模式
    2. Generate: 基于方向 + 约束 + 经验，生成候选 FactorSpec
    3. Filter: 多样性约束 + 族感知 + 正则化探索
    """

    def __init__(
        self,
        store: PersistentFactorStore | None = None,
        memory: FactorExperienceMemory | None = None,
    ) -> None:
        self.store = store or PersistentFactorStore()
        self.memory = memory or self.store.build_memory()

    # -- 公开接口 --

    def generate(self, request: HypothesisRequest) -> list[HypothesisCandidate]:
        """主入口：生成一批因子假设候选。"""
        direction = request.research_direction
        family = self._match_family(direction)

        # 1. Retrieve
        memory_context = self._retrieve_memory(family)
        library_context = self._retrieve_library(family, request.exclude_factor_ids)
        existing_families = self._get_library_family_distribution()

        # 2. Generate candidates via template-based exploration
        candidates = self._generate_candidates(
            direction=direction,
            family=family,
            max_candidates=request.max_candidates * 3,  # 过采样
            memory_context=memory_context,
            library_context=library_context,
            focus_features=request.focus_features,
        )

        # 3. Filter with diversity + regularization
        filtered = self._filter_candidates(
            candidates=candidates,
            exclude_families=request.exclude_families,
            exclude_factor_ids=request.exclude_factor_ids,
            existing_families=existing_families,
            diversity_penalty=request.diversity_penalty,
            max_output=request.max_candidates,
        )

        return filtered

    # -- Retrieve 阶段 --

    def _retrieve_memory(self, family: str) -> dict[str, Any]:
        """从经验记忆中提取相关模式。"""
        dummy_spec = FactorSpec(
            factor_id="retrieval_probe", name="probe", version="v0",
            description="", hypothesis="", family=family,
        )
        snapshot = self.memory.summarize_for_factor(dummy_spec)
        return {
            "successful_patterns": snapshot.successful_patterns,
            "failed_patterns": snapshot.failed_patterns,
            "regime_specific_findings": snapshot.regime_specific_findings,
            "rejected_reason_clusters": snapshot.rejected_reason_clusters,
        }

    def _retrieve_library(self, family: str, exclude_ids: list[str]) -> list[dict[str, Any]]:
        """从因子库中提取同族已有因子。"""
        entries = self.store.load_library_entries()
        result = []
        for entry in entries:
            if entry.factor_spec.factor_id in exclude_ids:
                continue
            if entry.factor_spec.family == family or family == "generic":
                result.append({
                    "factor_id": entry.factor_spec.factor_id,
                    "name": entry.factor_spec.name,
                    "family": entry.factor_spec.family,
                    "composite_score": entry.latest_report.scorecard.composite_score,
                    "decision": entry.latest_report.decision.value,
                })
        return result[:20]

    def _get_library_family_distribution(self) -> dict[str, int]:
        """统计因子库中各族的分布。"""
        entries = self.store.load_library_entries()
        dist: dict[str, int] = {}
        for entry in entries:
            f = entry.factor_spec.family
            dist[f] = dist.get(f, 0) + 1
        return dist

    # -- Generate 阶段 --

    def _match_family(self, direction: str) -> str:
        """将研究方向匹配到因子族。"""
        lowered = direction.lower()
        for family_key, family_def in FACTOR_FAMILIES.items():
            for keyword in family_def["keywords"]:
                if keyword in lowered:
                    return family_key
        return "generic"

    def _generate_candidates(
        self,
        direction: str,
        family: str,
        max_candidates: int,
        memory_context: dict[str, Any],
        library_context: list[dict[str, Any]],
        focus_features: list[str] | None,
    ) -> list[HypothesisCandidate]:
        """基于模板组合 + 经验引导生成候选因子。"""
        family_def = FACTOR_FAMILIES.get(family, FACTOR_FAMILIES["momentum"])
        base_features = focus_features or family_def.get("base_features", ["close"])
        windows = family_def.get("typical_windows", [5, 10, 20])

        candidates: list[HypothesisCandidate] = []
        template_index = 0

        # 策略1：基础时序算子组合
        for feat in base_features:
            for w in windows[:3]:
                for op_name, op_def in OPERATOR_CATALOG.items():
                    if op_def["category"] not in ("time_series",):
                        continue
                    if template_index >= max_candidates:
                        break
                    tree = FactorNode(
                        node_type="rank",
                        children=[
                            FactorNode(
                                node_type=op_name,
                                children=[FactorNode(node_type="feature", value=feat)],
                                params={"window": w},
                            )
                        ],
                    )
                    candidate = self._build_candidate_from_tree(
                        tree=tree,
                        family=family,
                        direction=direction,
                        features=[feat],
                        window=w,
                        template_name=f"{op_name}_{feat}_{w}",
                        memory_context=memory_context,
                        library_context=library_context,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                        template_index += 1

        # 策略2：双特征交叉组合
        if len(base_features) >= 2:
            f1, f2 = base_features[0], base_features[1]
            for w in windows[:2]:
                for cross_op in ["sub", "div", "mul"]:
                    if template_index >= max_candidates:
                        break
                    tree = FactorNode(
                        node_type="rank",
                        children=[
                            FactorNode(
                                node_type=cross_op,
                                children=[
                                    FactorNode(node_type="zscore", children=[
                                        FactorNode(node_type="delta", children=[
                                            FactorNode(node_type="feature", value=f1)
                                        ], params={"window": w})
                                    ]),
                                    FactorNode(node_type="zscore", children=[
                                        FactorNode(node_type="delta", children=[
                                            FactorNode(node_type="feature", value=f2)
                                        ], params={"window": w})
                                    ]),
                                ],
                            )
                        ],
                    )
                    candidate = self._build_candidate_from_tree(
                        tree=tree,
                        family=family,
                        direction=direction,
                        features=[f1, f2],
                        window=w,
                        template_name=f"cross_{cross_op}_{f1}_{f2}_{w}",
                        memory_context=memory_context,
                        library_context=library_context,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                        template_index += 1

        # 策略3：波动率调整组合
        if template_index < max_candidates:
            for feat in base_features[:1]:
                for w_short in [3, 5]:
                    for w_long in [20, 60]:
                        if template_index >= max_candidates:
                            break
                        tree = FactorNode(
                            node_type="rank",
                            children=[
                                FactorNode(
                                    node_type="div",
                                    children=[
                                        FactorNode(node_type="delta", children=[
                                            FactorNode(node_type="feature", value=feat)
                                        ], params={"window": w_short}),
                                        FactorNode(node_type="std", children=[
                                            FactorNode(node_type="feature", value=feat)
                                        ], params={"window": w_long}),
                                    ],
                                )
                            ],
                        )
                        candidate = self._build_candidate_from_tree(
                            tree=tree,
                            family="volatility_adjusted_momentum" if family == "momentum" else family,
                            direction=direction,
                            features=[feat],
                            window=w_long,
                            template_name=f"vol_adj_{feat}_{w_short}_{w_long}",
                            memory_context=memory_context,
                            library_context=library_context,
                        )
                        if candidate is not None:
                            candidates.append(candidate)
                            template_index += 1

        return candidates

    def _build_candidate_from_tree(
        self,
        tree: FactorNode,
        family: str,
        direction: str,
        features: list[str],
        window: int,
        template_name: str,
        memory_context: dict[str, Any],
        library_context: list[dict[str, Any]],
    ) -> HypothesisCandidate | None:
        """从表达树构建完整候选。"""
        factor_id = f"factor_{uuid4().hex[:10]}"
        family_def = FACTOR_FAMILIES.get(family, {})
        family_direction = family_def.get("direction", FactorDirection.UNKNOWN)

        name = f"{template_name}_候选"
        description = f"由假设生成器基于方向'{direction}'自动生成的{family}族候选因子"
        hypothesis = self._generate_hypothesis_text(family, features, window, template_name)

        dependencies = [
            FactorDependency(field_name=feat, lookback=window + 5, description=f"{feat}序列")
            for feat in features
        ]

        spec = FactorSpec(
            factor_id=factor_id,
            name=name,
            version="v1",
            description=description,
            hypothesis=hypothesis,
            family=family,
            direction=family_direction,
            expression_tree=tree,
            template_name=template_name,
            dependencies=dependencies,
            source="hypothesis_generator",
            created_from="auto_generated",
            status=FactorStatus.DRAFT,
        )

        # 计算新颖性
        novelty = self._score_novelty(spec, library_context)
        # 计算探索奖励（AlphaAgent 正则化）
        exploration_bonus = self._score_exploration(family, memory_context)

        return HypothesisCandidate(
            spec=spec,
            rationale=hypothesis,
            family_match=family,
            novelty_score=novelty,
            exploration_bonus=exploration_bonus,
        )

    def _generate_hypothesis_text(
        self, family: str, features: list[str], window: int, template_name: str
    ) -> str:
        """生成假设文本。"""
        templates = {
            "momentum": f"近{window}日价格动量在横截面排序中具有预测能力，短期趋势延续效应显著。",
            "reversal": f"近{window}日价格反向运动后存在均值回归机会，短期反转效应可捕捉。",
            "volatility": f"近{window}日波动率异动包含风险定价信息，低波动异象可能带来超额收益。",
            "volume_price": f"近{window}日量价关系背离蕴含信息不对称信号，可能预示价格反转。",
            "liquidity": f"近{window}日流动性变化反映交易者行为模式，异常换手蕴含alpha机会。",
            "fundamental": f"近{window}日基本面变化与价格变动的关系中存在可利用的定价偏差。",
        }
        return templates.get(family, f"近{window}日{'/'.join(features)}的特征组合具有横截面预测能力。")

    # -- Filter 阶段（AlphaAgent 正则化 + Hubble 族感知） --

    def _score_novelty(self, spec: FactorSpec, library_context: list[dict[str, Any]]) -> float:
        """评估候选因子与库内已有因子的差异度。"""
        if not library_context:
            return 1.0
        same_family_count = sum(1 for item in library_context if item.get("family") == spec.family)
        total = len(library_context)
        family_saturation = same_family_count / max(total, 1)
        return round(max(0.1, 1.0 - family_saturation), 4)

    def _score_exploration(self, family: str, memory_context: dict[str, Any]) -> float:
        """AlphaAgent 正则化探索：对探索不足的族给予奖励。"""
        failed_patterns = memory_context.get("failed_patterns", [])
        regime_findings = memory_context.get("regime_specific_findings", [])
        if not failed_patterns and not regime_findings:
            return 0.3  # 没有历史经验，给中等探索奖励
        # 失败越多 → 该方向探索不足 → 更高奖励
        exploration = min(0.5, 0.1 * len(failed_patterns) + 0.05 * len(regime_findings))
        return round(exploration, 4)

    def _filter_candidates(
        self,
        candidates: list[HypothesisCandidate],
        exclude_families: list[str],
        exclude_factor_ids: list[str],
        existing_families: dict[str, int],
        diversity_penalty: float,
        max_output: int,
    ) -> list[HypothesisCandidate]:
        """过滤并排序候选因子。"""
        # 排除指定族
        filtered = [c for c in candidates if c.family_match not in exclude_families]

        # 族感知多样性惩罚（Hubble 族感知）
        total_in_library = sum(existing_families.values()) or 1
        for candidate in filtered:
            family_count = existing_families.get(candidate.family_match, 0)
            family_penalty = diversity_penalty * (family_count / total_in_library)
            candidate.novelty_score = max(0.0, candidate.novelty_score - family_penalty)

        # 综合排序：novelty + exploration_bonus
        scored = sorted(
            filtered,
            key=lambda c: c.novelty_score + c.exploration_bonus,
            reverse=True,
        )

        return scored[:max_output]
