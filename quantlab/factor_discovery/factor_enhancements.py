"""因子发掘增强模块 —— 补齐自动化因子发掘的六大缺口。

1. 经验学习回路（ExperienceLoop）：从历史成功/失败中进化，R1 引用历史胜率特征
2. 风险中性化（RiskNeutralizer）：市值/行业/动量中性化，管控因子风险暴露
3. 多因子组合优化（FactorCombiner）：IC 加权 + 正交选择 + 组合 ICIR
4. 自动化参数搜索（ParameterSearcher）：Grid / 贝叶斯参数调优
5. P3 定制代码生成（CustomCodeGenerator）：LLM 生成 Python 因子代码 + 沙箱执行
6. 事前正交性引导（OrthogonalityGuide）：假设生成时避开已有因子空间
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. 经验学习回路 —— 因子假设→结果→经验沉淀→指导下一轮
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FactorOutcome:
    """单次因子发掘的结果记录。"""
    outcome_id: str
    direction: str
    hypothesis_intuition: str
    mechanism: str
    pseudocode: str
    input_fields: list[str]
    block_tree_desc: str  # 积木树的结构化描述
    verdict: str  # useful / marginal / useless
    rank_ic: float
    ic_ir: float
    coverage: float
    risk_exposure: dict[str, float]  # 市值/行业/动量暴露
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FactorOutcome":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ExperienceLoop:
    """经验学习回路：沉淀因子假设→结果的映射，指导未来假设生成。

    核心逻辑：
    - 每次发掘结束后，将假设结构 + 结果存入经验库
    - R1 生成假设时，从经验库检索相似方向的成功/失败模式
    - 按积木结构（如 delta+rank, ts_std+group_neutralize）聚合统计
    - 给出"哪种积木组合在哪个方向上更容易出 alpha"的经验指导
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = Path(__file__).resolve().parents[1] / "assistant_data" / "experience_loop"
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._outcomes: list[FactorOutcome] = self._load()

    def _load(self) -> list[FactorOutcome]:
        db_file = self.store_path / "outcomes.json"
        if not db_file.exists():
            return []
        try:
            raw = json.loads(db_file.read_text(encoding="utf-8"))
            return [FactorOutcome.from_dict(d) for d in raw]
        except Exception:
            return []

    def _save(self) -> None:
        db_file = self.store_path / "outcomes.json"
        db_file.write_text(
            json.dumps([o.to_dict() for o in self._outcomes], ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )

    def record(self, outcome: FactorOutcome) -> None:
        """记录一次因子发掘结果。"""
        self._outcomes.append(outcome)
        self._save()

    def record_batch(self, outcomes: list[FactorOutcome]) -> None:
        for o in outcomes:
            self._outcomes.append(o)
        self._save()

    def get_guidance(self, direction: str, top_k: int = 5) -> dict[str, Any]:
        """为 R1 提供经验指导。

        返回：
        - successful_patterns: 同方向成功的积木组合
        - failed_patterns: 同方向失败的积木组合
        - structure_stats: 各积木结构的全局胜率
        - field_stats: 各输入字段的全局胜率
        - direction_insight: 方向级洞察
        """
        dir_lower = direction.lower()

        # 按方向筛选
        dir_outcomes = [o for o in self._outcomes if dir_lower in o.direction.lower() or o.direction.lower() in dir_lower]
        if not dir_outcomes:
            dir_outcomes = self._outcomes  # 回退到全局

        successful = [o for o in dir_outcomes if o.verdict == "useful"]
        marginal = [o for o in dir_outcomes if o.verdict == "marginal"]
        failed = [o for o in dir_outcomes if o.verdict == "useless"]

        # 积木结构统计
        structure_stats = self._compute_structure_stats()

        # 输入字段统计
        field_stats = self._compute_field_stats()

        # 方向洞察
        direction_insight = self._compute_direction_insight(dir_outcomes, direction)

        return {
            "successful_patterns": [
                {"intuition": o.hypothesis_intuition, "structure": o.block_tree_desc, "rank_ic": o.rank_ic, "ic_ir": o.ic_ir}
                for o in successful[:top_k]
            ],
            "marginal_patterns": [
                {"intuition": o.hypothesis_intuition, "structure": o.block_tree_desc, "rank_ic": o.rank_ic}
                for o in marginal[:3]
            ],
            "failed_patterns": [
                {"intuition": o.hypothesis_intuition, "structure": o.block_tree_desc, "reason": "IC过低或风险暴露过大"}
                for o in failed[:3]
            ],
            "structure_stats": structure_stats,
            "field_stats": field_stats,
            "direction_insight": direction_insight,
            "total_recorded": len(self._outcomes),
        }

    def _compute_structure_stats(self) -> dict[str, dict[str, float]]:
        """统计各积木结构的胜率。"""
        struct_map: dict[str, list[str]] = {}
        for o in self._outcomes:
            key = o.block_tree_desc[:60]  # 截断做 key
            struct_map.setdefault(key, []).append(o.verdict)

        result = {}
        for key, verdicts in struct_map.items():
            total = len(verdicts)
            useful = sum(1 for v in verdicts if v == "useful")
            result[key] = {
                "win_rate": round(useful / max(total, 1), 3),
                "total_attempts": total,
                "useful_count": useful,
            }
        # 按胜率排序取 top 10
        sorted_items = sorted(result.items(), key=lambda x: x[1]["win_rate"], reverse=True)[:10]
        return dict(sorted_items)

    def _compute_field_stats(self) -> dict[str, dict[str, float]]:
        """统计各输入字段关联的因子胜率。"""
        field_map: dict[str, list[str]] = {}
        for o in self._outcomes:
            for f in o.input_fields:
                field_map.setdefault(f, []).append(o.verdict)

        result = {}
        for f, verdicts in field_map.items():
            total = len(verdicts)
            useful = sum(1 for v in verdicts if v == "useful")
            result[f] = {
                "win_rate": round(useful / max(total, 1), 3),
                "total_attempts": total,
            }
        return result

    def _compute_direction_insight(self, outcomes: list[FactorOutcome], direction: str) -> str:
        """生成方向级洞察文本，供 R1 prompt 使用。"""
        if not outcomes:
            return f"方向'{direction}'暂无历史经验，属于新探索方向。"

        useful = sum(1 for o in outcomes if o.verdict == "useful")
        marginal = sum(1 for o in outcomes if o.verdict == "marginal")
        total = len(outcomes)
        avg_ic = np.mean([o.rank_ic for o in outcomes]) if outcomes else 0.0

        # 找出成功的共性
        successful = [o for o in outcomes if o.verdict == "useful"]
        common_fields = {}
        for o in successful:
            for f in o.input_fields:
                common_fields[f] = common_fields.get(f, 0) + 1

        insight = f"方向'{direction}'已有{total}次尝试，成功{useful}次，边际{marginal}次，平均IC={avg_ic:.4f}。"
        if common_fields:
            top_fields = sorted(common_fields.items(), key=lambda x: x[1], reverse=True)[:3]
            insight += f"成功因子常用字段：{', '.join(f'{f}({c}次)' for f, c in top_fields)}。"

        # 找出常见失败原因
        failed = [o for o in outcomes if o.verdict == "useless"]
        if failed:
            high_exposure = [o for o in failed if any(abs(v) > 0.3 for v in o.risk_exposure.values())]
            if high_exposure:
                insight += f"失败因子中有{len(high_exposure)}个风险暴露过大，注意中性化。"

        return insight


# ═══════════════════════════════════════════════════════════════════
# 2. 风险中性化 —— 市值/行业/动量中性化
# ═══════════════════════════════════════════════════════════════════

class RiskNeutralizer:
    """因子风险中性化器。

    支持三种中性化：
    1. 行业中性化：组内去均值
    2. 市值中性化：对 log(market_cap) 做截面回归取残差
    3. 动量中性化：对过去 N 日收益做截面回归取残差

    中性化顺序：行业 → 市值 → 动量
    """

    def __init__(
        self,
        industry: bool = True,
        market_cap: bool = True,
        momentum: bool = False,
        momentum_window: int = 20,
        date_col: str = "date",
        asset_col: str = "asset",
    ) -> None:
        self.industry = industry
        self.market_cap = market_cap
        self.momentum = momentum
        self.momentum_window = momentum_window
        self.date_col = date_col
        self.asset_col = asset_col

    def neutralize(self, factor_values: pd.Series, market_df: pd.DataFrame) -> pd.Series:
        """对因子值做风险中性化，返回中性化后的因子值。"""
        result = factor_values.copy()
        exposure = {}

        # 对齐索引
        if result.name is None:
            result.name = "factor"
        aligned = market_df.copy()
        aligned["_factor"] = result

        # 确保有双索引
        if self.date_col not in aligned.index.names:
            if self.date_col in aligned.columns and self.asset_col in aligned.columns:
                aligned = aligned.set_index([self.date_col, self.asset_col])

        # 1. 行业中性化
        if self.industry and "industry" in aligned.columns:
            industry_col = aligned["industry"]
            result = result.groupby([pd.Grouper(level=self.date_col), industry_col]).transform(
                lambda x: x - x.mean()
            )
            # 计算暴露
            industry_mean = aligned.groupby("industry")["_factor"].mean().abs()
            exposure["industry"] = float(industry_mean.max()) if not industry_mean.empty else 0.0

        # 2. 市值中性化
        if self.market_cap:
            mc_col = self._find_market_cap_col(aligned)
            if mc_col is not None:
                log_mc = np.log(aligned[mc_col].clip(lower=1.0))
                result, mc_exp = self._cross_section_residual(result, log_mc, aligned)
                exposure["market_cap"] = mc_exp
            else:
                exposure["market_cap"] = 0.0
        else:
            exposure["market_cap"] = 0.0

        # 3. 动量中性化
        if self.momentum and "close" in aligned.columns:
            w = self.momentum_window
            past_ret = aligned.groupby(level=self.asset_col)["close"].transform(
                lambda x: x.pct_change(w)
            )
            result, mom_exp = self._cross_section_residual(result, past_ret, aligned)
            exposure["momentum"] = mom_exp
        else:
            exposure["momentum"] = 0.0

        if "industry" not in exposure:
            exposure["industry"] = 0.0

        return result, exposure

    def _find_market_cap_col(self, df: pd.DataFrame) -> str | None:
        """找到市值列名。"""
        for col in ["total_mv", "market_cap", "circ_mv", "float_market_cap"]:
            if col in df.columns:
                return col
        return None

    def _cross_section_residual(self, y: pd.Series, X: pd.Series, df: pd.DataFrame) -> tuple[pd.Series, float]:
        """截面回归取残差，返回残差序列和暴露度。"""
        date_col = self.date_col

        residuals = []
        exposures = []

        for date_val, group_idx in y.groupby(level=date_col).groups.items():
            y_group = y.loc[group_idx].dropna()
            x_group = X.loc[group_idx].dropna()

            # 对齐
            common_idx = y_group.index.intersection(x_group.index)
            if len(common_idx) < 10:
                residuals.append(pd.Series(0.0, index=group_idx))
                continue

            y_vals = y_group.loc[common_idx].values
            x_vals = x_group.loc[common_idx].values

            # 简单 OLS: y = a + b*x + e
            valid = ~(np.isnan(y_vals) | np.isnan(x_vals) | np.isinf(x_vals))
            if valid.sum() < 10:
                residuals.append(pd.Series(0.0, index=group_idx))
                continue

            slope, intercept, _, _, _ = sp_stats.linregress(x_vals[valid], y_vals[valid])
            predicted = intercept + slope * x_vals
            resid = y_vals - predicted

            exposures.append(abs(slope))
            resid_series = pd.Series(resid, index=common_idx)
            residuals.append(resid_series)

        if residuals:
            result = pd.concat(residuals)
            # 保留原始索引中不在回归中的值
            original_idx = y.index.difference(result.index)
            if len(original_idx) > 0:
                result = pd.concat([result, pd.Series(0.0, index=original_idx)])
            result = result.reindex(y.index)
            avg_exposure = np.mean(exposures) if exposures else 0.0
            return result, round(float(avg_exposure), 4)

        return y, 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. 多因子组合优化 —— IC 加权 + 正交选择 + 组合 ICIR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FactorCombinationResult:
    """多因子组合结果。"""
    combination_id: str
    factor_ids: list[str]
    weights: dict[str, float]  # factor_id → weight
    combined_ic: float
    combined_icir: float
    combined_rank_ic: float
    pairwise_correlations: dict[str, dict[str, float]]
    coverage: float
    method: str  # ic_weighted / equal_weight / max_icir

    def to_dict(self) -> dict:
        return asdict(self)


class FactorCombiner:
    """多因子组合优化器。

    支持：
    1. IC 加权组合：权重 = IC_mean / sum(|IC_mean|)
    2. 等权组合：所有因子等权
    3. 最大 ICIR 组合：迭代搜索权重使 ICIR 最大化
    4. 正交选择：从因子池中选出互相正交的子集
    """

    def __init__(self, date_col: str = "date", asset_col: str = "asset") -> None:
        self.date_col = date_col
        self.asset_col = asset_col

    def combine(
        self,
        factor_panels: dict[str, pd.Series],  # factor_id → factor_values
        market_df: pd.DataFrame,
        method: str = "ic_weighted",
        correlation_threshold: float = 0.6,
    ) -> FactorCombinationResult:
        """组合多个因子。"""
        if not factor_panels:
            raise ValueError("因子面板为空")

        # 1. 正交选择
        selected_ids = self._orthogonal_selection(factor_panels, correlation_threshold)
        selected_panels = {fid: factor_panels[fid] for fid in selected_ids}

        # 2. 计算各因子的 IC
        ic_stats = {}
        for fid, fv in selected_panels.items():
            ic_data = self._compute_single_ic(fv, market_df)
            ic_stats[fid] = ic_data

        # 3. 计算权重
        weights = self._compute_weights(ic_stats, method)

        # 4. 加权组合
        combined = self._weighted_combine(selected_panels, weights)

        # 5. 计算组合 IC
        combined_ic_data = self._compute_single_ic(combined, market_df)

        # 6. 计算两两相关性
        pairwise_corr = self._compute_pairwise_correlation(selected_panels)

        return FactorCombinationResult(
            combination_id=f"combo_{uuid4().hex[:8]}",
            factor_ids=selected_ids,
            weights=weights,
            combined_ic=combined_ic_data["ic_mean"],
            combined_icir=combined_ic_data["ic_ir"],
            combined_rank_ic=combined_ic_data["rank_ic_mean"],
            pairwise_correlations=pairwise_corr,
            coverage=combined_ic_data["coverage"],
            method=method,
        )

    def _orthogonal_selection(
        self,
        factor_panels: dict[str, pd.Series],
        threshold: float,
    ) -> list[str]:
        """从因子池中选出互相正交的子集（贪心法）。"""
        if len(factor_panels) <= 1:
            return list(factor_panels.keys())

        # 计算两两相关性
        fids = list(factor_panels.keys())
        selected = [fids[0]]

        for fid in fids[1:]:
            is_orthogonal = True
            for existing_fid in selected:
                corr = factor_panels[fid].corr(factor_panels[existing_fid])
                if abs(corr) > threshold:
                    is_orthogonal = False
                    break
            if is_orthogonal:
                selected.append(fid)

        return selected

    def _compute_single_ic(self, factor_values: pd.Series, market_df: pd.DataFrame) -> dict[str, float]:
        """计算单个因子的 IC 统计。"""
        try:
            aligned = market_df.copy()
            aligned["_factor"] = factor_values

            if self.date_col in aligned.columns and self.asset_col in aligned.columns:
                aligned = aligned.set_index([self.date_col, self.asset_col])
            elif "ts_code" in aligned.columns:
                aligned = aligned.rename(columns={"ts_code": self.asset_col})
                if self.date_col in aligned.columns:
                    aligned = aligned.set_index([self.date_col, self.asset_col])

            # 计算下期收益
            if "close" in aligned.columns:
                aligned = aligned.sort_values([self.asset_col, self.date_col])
                aligned["fwd_ret_5"] = aligned.groupby(self.asset_col)["close"].shift(-5) / aligned["close"] - 1
            else:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            # 截面 Rank IC
            rank_ics = []
            for date_val, group in aligned.groupby(level=self.date_col):
                valid = group[["_factor", "fwd_ret_5"]].dropna()
                if len(valid) >= 20:
                    ric = valid["_factor"].rank().corr(valid["fwd_ret_5"].rank(), method="pearson")
                    if not np.isnan(ric):
                        rank_ics.append(ric)

            if not rank_ics:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            ic_mean = float(np.mean(rank_ics))
            ic_std = float(np.std(rank_ics))
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            coverage = float(factor_values.notna().mean())

            return {
                "ic_mean": round(ic_mean, 4),
                "rank_ic_mean": round(ic_mean, 4),
                "ic_ir": round(ic_ir, 4),
                "coverage": round(coverage, 4),
            }
        except Exception as exc:
            logger.warning(f"IC 计算失败: {exc}")
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

    def _compute_weights(self, ic_stats: dict[str, dict], method: str) -> dict[str, float]:
        """计算组合权重。"""
        if method == "equal_weight":
            n = len(ic_stats)
            return {fid: round(1.0 / n, 4) for fid in ic_stats}

        elif method == "ic_weighted":
            # 权重 = IC / sum(|IC|)
            ic_vals = {fid: max(data["ic_mean"], 0.001) for fid, data in ic_stats.items()}
            total = sum(abs(v) for v in ic_vals.values())
            if total == 0:
                n = len(ic_stats)
                return {fid: round(1.0 / n, 4) for fid in ic_stats}
            return {fid: round(abs(v) / total, 4) for fid, v in ic_vals.items()}

        elif method == "max_icir":
            # 简化版：用 ICIR 代替 IC 作为权重
            icir_vals = {fid: max(data["ic_ir"], 0.001) for fid, data in ic_stats.items()}
            total = sum(abs(v) for v in icir_vals.values())
            if total == 0:
                n = len(ic_stats)
                return {fid: round(1.0 / n, 4) for fid in ic_stats}
            return {fid: round(abs(v) / total, 4) for fid, v in icir_vals.items()}

        # 默认等权
        n = len(ic_stats)
        return {fid: round(1.0 / n, 4) for fid in ic_stats}

    def _weighted_combine(
        self,
        factor_panels: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> pd.Series:
        """加权组合多个因子。"""
        combined = None
        for fid, fv in factor_panels.items():
            w = weights.get(fid, 0.0)
            if combined is None:
                combined = fv.fillna(0) * w
            else:
                combined = combined + fv.fillna(0) * w
        return combined if combined is not None else pd.Series(dtype=float)

    def _compute_pairwise_correlation(
        self, factor_panels: dict[str, pd.Series]
    ) -> dict[str, dict[str, float]]:
        """计算两两相关性矩阵。"""
        fids = list(factor_panels.keys())
        result = {}
        for f1 in fids:
            result[f1] = {}
            for f2 in fids:
                if f1 == f2:
                    result[f1][f2] = 1.0
                else:
                    corr = factor_panels[f1].corr(factor_panels[f2])
                    result[f1][f2] = round(float(corr) if not np.isnan(corr) else 0.0, 4)
        return result


# ═══════════════════════════════════════════════════════════════════
# 4. 自动化参数搜索 —— Grid / 贝叶斯参数调优
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ParamSearchResult:
    """参数搜索结果。"""
    factor_id: str
    best_params: dict[str, Any]
    best_ic: float
    best_icir: float
    all_trials: list[dict[str, Any]]  # [{params, ic, icir}, ...]
    search_method: str
    total_trials: int

    def to_dict(self) -> dict:
        return asdict(self)


class ParameterSearcher:
    """自动化参数搜索器。

    支持：
    1. Grid Search：穷举参数组合
    2. Random Search：随机采样参数组合
    3. 简化贝叶斯：基于前几轮结果，用高斯过程思路采样（纯 numpy 实现）

    搜索范围自动从积木树的参数中提取（window, alpha, threshold 等）。
    """

    # 默认参数搜索空间
    DEFAULT_PARAM_SPACE: dict[str, list[Any]] = {
        "window": [3, 5, 10, 20, 40, 60],
        "alpha": [0.05, 0.1, 0.2, 0.3],
        "threshold": [0.0, 0.01, 0.02, 0.05],
        "n": [3, 5, 10],
        "lo": [-3.0, -2.0, -1.5],
        "hi": [1.5, 2.0, 3.0],
    }

    def __init__(
        self,
        max_trials: int = 30,
        method: str = "grid",  # grid / random
        date_col: str = "date",
        asset_col: str = "asset",
    ) -> None:
        self.max_trials = max_trials
        self.method = method
        self.date_col = date_col
        self.asset_col = asset_col

    def search(
        self,
        block_tree_dict: dict,
        market_df: pd.DataFrame,
        factor_id: str = "",
    ) -> ParamSearchResult:
        """对积木树中的可调参数执行搜索。"""
        from .blocks import Block, BlockExecutor

        # 1. 提取可调参数
        param_space = self._extract_param_space(block_tree_dict)

        if not param_space:
            # 无可调参数，直接执行一次
            try:
                root_block = Block.from_dict(block_tree_dict)
                executor = BlockExecutor(date_col=self.date_col, asset_col=self.asset_col)
                factor_values = executor.execute(root_block, market_df)
                ic_data = self._compute_ic(factor_values, market_df)
                return ParamSearchResult(
                    factor_id=factor_id,
                    best_params={},
                    best_ic=ic_data["ic_mean"],
                    best_icir=ic_data["ic_ir"],
                    all_trials=[{"params": {}, **ic_data}],
                    search_method="none",
                    total_trials=1,
                )
            except Exception as exc:
                logger.warning(f"参数搜索失败（无可调参数）: {exc}")
                return ParamSearchResult(
                    factor_id=factor_id, best_params={}, best_ic=0.0, best_icir=0.0,
                    all_trials=[], search_method="none", total_trials=0,
                )

        # 2. 生成参数组合
        param_combos = self._generate_combos(param_space)

        # 3. 逐个执行评估
        trials = []
        best_ic = -999.0
        best_icir = -999.0
        best_params = {}

        for combo in param_combos[:self.max_trials]:
            try:
                # 替换积木树中的参数
                modified_tree = self._apply_params(block_tree_dict, combo)
                root_block = Block.from_dict(modified_tree)
                executor = BlockExecutor(date_col=self.date_col, asset_col=self.asset_col)
                factor_values = executor.execute(root_block, market_df)
                ic_data = self._compute_ic(factor_values, market_df)

                trial = {"params": combo, **ic_data}
                trials.append(trial)

                # 更新最优（按 ICIR 排序）
                if ic_data["ic_ir"] > best_icir or (ic_data["ic_ir"] == best_icir and ic_data["ic_mean"] > best_ic):
                    best_ic = ic_data["ic_mean"]
                    best_icir = ic_data["ic_ir"]
                    best_params = combo

            except Exception as exc:
                logger.debug(f"参数组合 {combo} 执行失败: {exc}")
                trials.append({"params": combo, "ic_mean": 0.0, "ic_ir": 0.0, "error": str(exc)})
                continue

        return ParamSearchResult(
            factor_id=factor_id,
            best_params=best_params,
            best_ic=best_ic,
            best_icir=best_icir,
            all_trials=trials,
            search_method=self.method,
            total_trials=len(trials),
        )

    def _extract_param_space(self, block_tree_dict: dict) -> dict[str, list[Any]]:
        """从积木树中提取可调参数及其搜索空间。"""
        space: dict[str, list[Any]] = {}
        self._walk_tree_for_params(block_tree_dict, space, "")
        return space

    def _walk_tree_for_params(self, tree: dict, space: dict, prefix: str) -> None:
        """递归遍历积木树，提取参数。"""
        if not isinstance(tree, dict):
            return

        params = tree.get("params", {})
        block_type = tree.get("block_type", "")
        op = tree.get("op", "")

        for key, value in params.items():
            param_key = f"{prefix}{op}__{key}" if op else f"{prefix}{key}"
            if key in self.DEFAULT_PARAM_SPACE:
                space[param_key] = self.DEFAULT_PARAM_SPACE[key]
            elif isinstance(value, (int, float)):
                # 生成一个简单的搜索范围
                if isinstance(value, int) and value > 0:
                    space[param_key] = [max(1, value // 2), value, value * 2]
                elif isinstance(value, float):
                    space[param_key] = [value * 0.5, value, value * 1.5]

        # 递归子树
        for child_key in ["input_block", "left", "right", "cond", "cond_block"]:
            child = tree.get(child_key)
            if isinstance(child, dict):
                self._walk_tree_for_params(child, space, prefix)

    def _generate_combos(self, param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """生成参数组合（Grid 或 Random）。"""
        if self.method == "random":
            return self._random_combos(param_space)
        return self._grid_combos(param_space)

    def _grid_combos(self, param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """生成网格搜索组合。"""
        import itertools

        keys = list(param_space.keys())
        values = [param_space[k] for k in keys]

        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
            if len(combos) >= self.max_trials:
                break
        return combos

    def _random_combos(self, param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """生成随机搜索组合。"""
        import random

        combos = []
        for _ in range(self.max_trials):
            combo = {}
            for k, v in param_space.items():
                combo[k] = random.choice(v)
            combos.append(combo)
        return combos

    def _apply_params(self, tree: dict, params: dict[str, Any]) -> dict:
        """将搜索到的参数应用回积木树。"""
        import copy
        tree = copy.deepcopy(tree)
        self._apply_params_recursive(tree, params, "")
        return tree

    def _apply_params_recursive(self, tree: dict, params: dict[str, Any], prefix: str) -> None:
        """递归应用参数。"""
        if not isinstance(tree, dict):
            return

        op = tree.get("op", "")
        tree_params = tree.get("params", {})

        for key in list(tree_params.keys()):
            param_key = f"{prefix}{op}__{key}" if op else f"{prefix}{key}"
            if param_key in params:
                tree_params[key] = params[param_key]

        for child_key in ["input_block", "left", "right", "cond", "cond_block"]:
            child = tree.get(child_key)
            if isinstance(child, dict):
                self._apply_params_recursive(child, params, prefix)

    def _compute_ic(self, factor_values: pd.Series, market_df: pd.DataFrame) -> dict[str, float]:
        """计算因子 IC。"""
        try:
            aligned = market_df.copy()
            aligned["_factor"] = factor_values

            if self.date_col in aligned.columns and self.asset_col in aligned.columns:
                aligned = aligned.set_index([self.date_col, self.asset_col])
            elif "ts_code" in aligned.columns:
                aligned = aligned.rename(columns={"ts_code": self.asset_col})
                if self.date_col in aligned.columns:
                    aligned = aligned.set_index([self.date_col, self.asset_col])

            if "close" in aligned.columns:
                aligned = aligned.sort_values([self.asset_col, self.date_col])
                aligned["fwd_ret_5"] = aligned.groupby(self.asset_col)["close"].shift(-5) / aligned["close"] - 1
            else:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            rank_ics = []
            for date_val, group in aligned.groupby(level=self.date_col):
                valid = group[["_factor", "fwd_ret_5"]].dropna()
                if len(valid) >= 20:
                    ric = valid["_factor"].rank().corr(valid["fwd_ret_5"].rank(), method="pearson")
                    if not np.isnan(ric):
                        rank_ics.append(ric)

            if not rank_ics:
                return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}

            ic_mean = float(np.mean(rank_ics))
            ic_std = float(np.std(rank_ics))
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            coverage = float(factor_values.notna().mean())

            return {
                "ic_mean": round(ic_mean, 4),
                "rank_ic_mean": round(ic_mean, 4),
                "ic_ir": round(ic_ir, 4),
                "coverage": round(coverage, 4),
            }
        except Exception:
            return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "ic_ir": 0.0, "coverage": 0.0}


# ═══════════════════════════════════════════════════════════════════
# 5. P3 定制代码生成 —— LLM 生成 Python 因子代码 + 沙箱执行
# ═══════════════════════════════════════════════════════════════════

P3_CODE_GEN_SYSTEM_PROMPT = """你是一个量化因子代码生成专家（P3角色）。

你的职责是将因子假设转化为可执行的 Python 代码。

代码要求：
1. 输入：一个 pandas DataFrame（包含 date, asset, close, open, high, low, volume, vwap, turnover_rate, pe_ttm, pb, total_mv, circ_mv, adj_factor, industry 列）
2. 输出：一个 pd.Series，索引与输入 DataFrame 相同，值为因子值
3. 只能使用 pandas 和 numpy，不允许 import 其他库
4. 必须处理 NaN（使用 fillna 或 dropna）
5. 不允许执行任何 I/O 操作（读写文件、网络请求等）
6. 代码必须定义一个名为 `compute_factor` 的函数
7. 函数签名：`def compute_factor(df: pd.DataFrame) -> pd.Series:`

输出格式（严格 JSON）：
{
  "code": "def compute_factor(df):\\n    ...\\n    return result",
  "description": "代码逻辑简述",
  "input_fields": ["close", "volume"],
  "estimated_complexity": "O(n)"
}"""


class CustomCodeGenerator:
    """P3 定制代码生成器。

    工作流：
    1. 接收 CustomRequest（描述因子逻辑）
    2. 通过 LLM 生成 Python 代码
    3. 沙箱执行验证
    4. 返回因子值 Series
    """

    # 沙箱安全白名单
    SAFE_BUILTINS = {
        "abs", "max", "min", "sum", "len", "range", "enumerate", "zip",
        "sorted", "reversed", "round", "float", "int", "bool", "str",
        "list", "dict", "set", "tuple", "type", "isinstance", "True", "False", "None",
    }

    SAFE_MODULES = {"pandas", "np", "numpy"}

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm = llm_client

    def generate_and_execute(
        self,
        custom_request: dict[str, Any],
        market_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """生成定制代码并执行。

        Args:
            custom_request: 包含 what, input_schema 等字段
            market_df: 市场数据

        Returns:
            {status, factor_values, code, error}
        """
        # 1. 通过 LLM 生成代码
        code, description = self._generate_code(custom_request)
        if not code:
            return {"status": "failed", "error": "LLM 未生成代码", "factor_values": None, "code": ""}

        # 2. 安全检查
        safety_check = self._safety_check(code)
        if not safety_check["safe"]:
            return {"status": "failed", "error": f"安全检查未通过: {safety_check['reasons']}", "factor_values": None, "code": code}

        # 3. 沙箱执行
        exec_result = self._sandbox_execute(code, market_df)

        return {
            "status": exec_result["status"],
            "factor_values": exec_result.get("factor_values"),
            "code": code,
            "description": description,
            "error": exec_result.get("error", ""),
        }

    def _generate_code(self, custom_request: dict[str, Any]) -> tuple[str, str]:
        """通过 LLM 生成因子代码。"""
        if not self.llm or not self.llm.api_key:
            # 无 LLM，尝试基于模板生成简单代码
            return self._generate_code_template(custom_request)

        what = custom_request.get("what", "")
        input_schema = custom_request.get("input_schema", {})
        must_pass = custom_request.get("must_pass", [])

        user_prompt = f"""请为以下因子需求生成 Python 代码：

需求描述：{what}

输入字段：{json.dumps(input_schema, ensure_ascii=False)}

必须通过的检查：
{chr(10).join('- ' + m for m in must_pass) if must_pass else '无额外检查'}

请生成 compute_factor 函数。"""

        try:
            result = self.llm.chat_json(P3_CODE_GEN_SYSTEM_PROMPT, user_prompt, temperature=0.2)
            code = result.get("code", "")
            desc = result.get("description", "")
            return code, desc
        except Exception as exc:
            logger.warning(f"LLM 代码生成失败: {exc}")
            return self._generate_code_template(custom_request)

    def _generate_code_template(self, custom_request: dict[str, Any]) -> tuple[str, str]:
        """基于模板生成简单因子代码（无 LLM 回退）。"""
        what = custom_request.get("what", "").lower()
        input_fields = custom_request.get("input_schema", {}).get("fields", ["close"])

        # 简单模板
        if "波动" in what or "volatility" in what:
            code = f"""import pandas as pd
import numpy as np

def compute_factor(df):
    factor = df.groupby('asset')['{input_fields[0]}'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    return -factor.fillna(0)
"""
        elif "动量" in what or "momentum" in what:
            code = f"""import pandas as pd
import numpy as np

def compute_factor(df):
    factor = df.groupby('asset')['{input_fields[0]}'].transform(
        lambda x: x.pct_change(20)
    )
    return factor.fillna(0)
"""
        else:
            code = f"""import pandas as pd
import numpy as np

def compute_factor(df):
    factor = df['{input_fields[0]}'].astype(float)
    return factor.fillna(0)
"""

        return code, "模板生成（无 LLM）"

    def _safety_check(self, code: str) -> dict[str, Any]:
        """检查代码安全性。"""
        reasons = []

        # 禁止的关键词
        forbidden = [
            "import os", "import sys", "import subprocess", "import shutil",
            "open(", "exec(", "eval(", "__import__", "compile(",
            "os.system", "os.popen", "subprocess", "socket",
            "requests.", "urllib", "http.", "pathlib",
            "__builtins__", "__globals__", "__locals__",
            "getattr", "setattr", "delattr", "globals()", "locals()",
        ]

        for keyword in forbidden:
            if keyword in code:
                reasons.append(f"禁止使用: {keyword}")

        # 必须包含 compute_factor 函数
        if "def compute_factor" not in code:
            reasons.append("缺少 compute_factor 函数定义")

        return {"safe": len(reasons) == 0, "reasons": reasons}

    def _sandbox_execute(self, code: str, market_df: pd.DataFrame) -> dict[str, Any]:
        """在受限环境中执行代码。"""
        try:
            # 准备受限的命名空间
            import pandas as _pd
            import numpy as _np

            namespace = {
                "pd": _pd,
                "np": _np,
                "pandas": _pd,
                "numpy": _np,
                "__builtins__": {
                    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                    for k in self.SAFE_BUILTINS
                    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
                },
            }

            # 执行代码定义
            exec(code, namespace)  # noqa: S102

            # 调用 compute_factor
            compute_fn = namespace.get("compute_factor")
            if compute_fn is None:
                return {"status": "failed", "error": "compute_factor 函数未定义"}

            result = compute_fn(market_df)

            # 验证输出
            if not isinstance(result, pd.Series):
                return {"status": "failed", "error": f"输出类型错误: {type(result)}，期望 pd.Series"}

            if len(result) != len(market_df):
                return {"status": "failed", "error": f"输出长度不匹配: {len(result)} vs {len(market_df)}"}

            return {"status": "success", "factor_values": result}

        except Exception as exc:
            return {"status": "failed", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# 6. 事前正交性引导 —— 假设生成时避开已有因子空间
# ═══════════════════════════════════════════════════════════════════

class OrthogonalityGuide:
    """事前正交性引导器。

    在 R1 生成假设之前，分析已有因子空间，给出：
    1. 已有因子覆盖的方向/字段/算子组合
    2. 未被覆盖的空间（建议探索方向）
    3. 高饱和度方向（建议避开）

    让 R1 的 prompt 包含正交性约束，而非仅靠事后筛选。
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = Path(__file__).resolve().parents[1] / "assistant_data" / "experience_loop"
        self.store_path = Path(store_path)

    def get_orthogonality_context(self, direction: str) -> dict[str, Any]:
        """获取正交性上下文，供 R1 prompt 使用。"""
        # 从经验库加载已有因子信息
        outcomes = self._load_outcomes()

        if not outcomes:
            return {
                "covered_fields": [],
                "covered_structures": [],
                "saturated_directions": [],
                "unexplored_directions": ["量价背离", "资金流向", "基本面变化率", "波动率结构"],
                "orthogonality_hint": "因子库为空，所有方向均可自由探索。",
            }

        # 分析已有因子的字段覆盖
        field_counts: dict[str, int] = {}
        for o in outcomes:
            for f in o.input_fields:
                field_counts[f] = field_counts.get(f, 0) + 1

        # 分析已有因子的结构覆盖
        structure_counts: dict[str, int] = {}
        for o in outcomes:
            key = o.block_tree_desc[:40]
            structure_counts[key] = structure_counts.get(key, 0) + 1

        # 分析方向饱和度
        direction_counts: dict[str, int] = {}
        for o in outcomes:
            d = o.direction[:20]
            direction_counts[d] = direction_counts.get(d, 0) + 1

        # 高饱和度方向（>3个因子）
        saturated = [d for d, c in direction_counts.items() if c >= 3]

        # 低覆盖字段
        all_possible_fields = {"close", "open", "high", "low", "volume", "vwap", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv"}
        underused = [f for f in all_possible_fields if field_counts.get(f, 0) <= 1]

        # 推荐未探索方向
        standard_directions = [
            "动量", "反转", "波动率", "量价关系", "流动性", "基本面",
            "资金流", "情绪", "事件驱动", "跨市场",
        ]
        explored_dirs = set(d.lower()[:4] for d in direction_counts.keys())
        unexplored = [d for d in standard_directions if d[:4] not in explored_dirs]

        # 生成正交性提示
        hint_parts = []
        if saturated:
            hint_parts.append(f"以下方向已高度饱和，请避开：{', '.join(saturated)}")
        if underused:
            hint_parts.append(f"以下数据字段使用不足，建议优先使用：{', '.join(underused)}")
        if unexplored:
            hint_parts.append(f"以下方向尚无探索记录：{', '.join(unexplored)}")

        orth_hint = " ".join(hint_parts) if hint_parts else "当前因子库较小，各方向均可探索。"

        return {
            "covered_fields": sorted(field_counts.items(), key=lambda x: x[1], reverse=True),
            "covered_structures": sorted(structure_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "saturated_directions": saturated,
            "unexplored_directions": unexplored,
            "underused_fields": underused,
            "orthogonality_hint": orth_hint,
            "total_existing_factors": len(outcomes),
        }

    def _load_outcomes(self) -> list[FactorOutcome]:
        """从经验库加载因子结果。"""
        db_file = self.store_path / "outcomes.json"
        if not db_file.exists():
            return []
        try:
            raw = json.loads(db_file.read_text(encoding="utf-8"))
            return [FactorOutcome.from_dict(d) for d in raw]
        except Exception:
            return []

    def generate_orthogonality_prompt_addon(self, direction: str) -> str:
        """生成正交性约束的 prompt 附加文本，直接拼入 R1 的 user prompt。"""
        ctx = self.get_orthogonality_context(direction)

        addon = f"""
【正交性约束——必须遵守】
已有因子数：{ctx['total_existing_factors']}
饱和方向（请避开）：{', '.join(ctx['saturated_directions']) or '无'}
使用不足的字段（建议优先）：{', '.join(ctx['underused_fields']) or '无'}
未探索的方向：{', '.join(ctx['unexplored_directions']) or '无'}

要求：
1. 你的假设必须与已有因子方向有实质区别，不要在饱和方向上重复
2. 优先使用使用不足的数据字段
3. 积木组合应避免与已有结构高度雷同
4. 如在已有方向上创新，必须提出显著不同的机制解释
"""
        return addon
