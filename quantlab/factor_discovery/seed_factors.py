"""种子因子库 —— 冷启动引导。

提供一组经过学术验证的基础因子，用 Block 系统构造，用于：
1. 冷启动时填充因子库，让经验回路和正交性引导有数据可查
2. 作为进化搜索的基准锚点

种子因子覆盖五大类：
- 动量 (momentum): 12-1月动量
- 反转 (reversal): 短期反转
- 价值 (value): 账面市值比
- 低波 (low_vol): 历史波动率倒数
- 规模 (size): 小盘溢价
- 换手 (turnover): 换手率异常
- 质量 (quality): 低PE溢价
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SeedFactorDef:
    factor_id: str
    name: str
    family: str
    direction: str  # higher_is_better / lower_is_better
    description: str
    intuition: str
    mechanism: str
    block_tree: dict[str, Any]  # serialized Block tree
    input_fields: list[str]


def build_seed_factors() -> list[SeedFactorDef]:
    """构建种子因子列表。

    使用已注册的 Block 算子构造：
    - 取反：mul(constant(value=-1), x)  等价于 -x
    - 倒数：div(constant(value=1), x)   等价于 1/x
    """
    from .blocks import combine, data, transform

    seeds: list[SeedFactorDef] = []

    # ── Momentum: 20日价格动量 ──
    mom_close = data("close")
    mom_delta = transform("delta", mom_close, window=20)
    mom_rank = transform("rank", mom_delta)
    seeds.append(SeedFactorDef(
        factor_id="seed_momentum_20d",
        name="20日动量因子",
        family="momentum",
        direction="higher_is_better",
        description="过去20个交易日的价格变化率，反映中期趋势延续性",
        intuition="过去表现好的股票在未来1-3个月倾向于继续表现好（Jegadeesh & Titman, 1993）",
        mechanism="信息渐进扩散 + 投资者反应不足 → 价格趋势延续",
        block_tree=mom_rank.to_dict(),
        input_fields=["close"],
    ))

    # ── Short-term reversal: 5日反转 (-delta) ──
    rev_close = data("close")
    rev_delta = transform("delta", rev_close, window=5)
    rev_neg = combine("mul", transform("constant", rev_delta, value=-1), rev_delta)
    rev_rank = transform("rank", rev_neg)
    seeds.append(SeedFactorDef(
        factor_id="seed_reversal_5d",
        name="5日反转因子",
        family="reversal",
        direction="higher_is_better",
        description="过去5日跌幅最大的股票短期反弹",
        intuition="短期过度反应后的均值回归（Lehmann, 1990/Jegadeesh, 1990）",
        mechanism="流动性冲击 + 过度反应 → 短期反弹，与动量互补",
        block_tree=rev_rank.to_dict(),
        input_fields=["close"],
    ))

    # ── Value: PB倒数 (1/pb) ──
    val_pb = data("pb")
    val_inv = combine("div", transform("constant", val_pb, value=1), val_pb)
    val_rank = transform("rank", val_inv)
    seeds.append(SeedFactorDef(
        factor_id="seed_value_bp",
        name="账面市值比因子（PB倒数）",
        family="value",
        direction="higher_is_better",
        description="市净率越低（1/PB越高），预期收益越高",
        intuition="低估值股票长期跑赢高估值（Fama-French HML, 1993）",
        mechanism="价值溢价来源于风险补偿或行为偏差",
        block_tree=val_rank.to_dict(),
        input_fields=["pb"],
    ))

    # ── Size: 小盘溢价 (-log(circ_mv)) ──
    size_mv = data("circ_mv")
    size_log = transform("log", size_mv)
    size_neg = combine("mul", transform("constant", size_log, value=-1), size_log)
    size_rank = transform("rank", size_neg)
    seeds.append(SeedFactorDef(
        factor_id="seed_size_small",
        name="小盘溢价因子",
        family="size",
        direction="higher_is_better",
        description="流通市值越小，预期收益越高",
        intuition="小市值股票风险溢价更高（Fama-French SMB, 1993）",
        mechanism="流动性风险 + 信息不对称 → 小盘溢价",
        block_tree=size_rank.to_dict(),
        input_fields=["circ_mv"],
    ))

    # ── Low volatility: 波动率倒数 (1/ts_std) ──
    lowvol_close = data("close")
    lowvol_std = transform("ts_std", lowvol_close, window=60)
    lowvol_inv = combine("div", transform("constant", lowvol_std, value=1), lowvol_std)
    lowvol_rank = transform("rank", lowvol_inv)
    seeds.append(SeedFactorDef(
        factor_id="seed_low_volatility",
        name="低波动率因子",
        family="low_volatility",
        direction="higher_is_better",
        description="过去60日波动率越低，未来收益越高",
        intuition="低波动率股票风险调整后收益更高（低波动率异象，Ang et al. 2006）",
        mechanism="杠杆约束 + 彩票偏好 → 高波动股被高估，低波动股被低估",
        block_tree=lowvol_rank.to_dict(),
        input_fields=["close"],
    ))

    # ── Turnover: 低换手 (-turnover_rate) ──
    to_data = data("turnover_rate")
    to_neg = combine("mul", transform("constant", to_data, value=-1), to_data)
    to_rank = transform("rank", to_neg)
    seeds.append(SeedFactorDef(
        factor_id="seed_turnover_low",
        name="低换手率因子",
        family="turnover",
        direction="higher_is_better",
        description="换手率越低，未来收益越高",
        intuition="高换手股票被过度交易，低换手股票被忽视（Datar et al. 1998）",
        mechanism="注意力偏差 + 交易成本 → 低换手股票隐含溢价",
        block_tree=to_rank.to_dict(),
        input_fields=["turnover_rate"],
    ))

    # ── Quality: 低PE (1/pe_ttm) ──
    qual_pe = data("pe_ttm")
    qual_inv = combine("div", transform("constant", qual_pe, value=1), qual_pe)
    qual_rank = transform("rank", qual_inv)
    seeds.append(SeedFactorDef(
        factor_id="seed_quality_low_pe",
        name="低市盈率因子",
        family="quality",
        direction="higher_is_better",
        description="市盈率越低（1/PE越高），预期收益越高",
        intuition="便宜股票长期跑赢（Basu, 1977），与价值因子互补",
        mechanism="盈利稳定性溢价 + 成长预期修正",
        block_tree=qual_rank.to_dict(),
        input_fields=["pe_ttm"],
    ))

    return seeds


def bootstrap_seed_factors(
    market_df: "pd.DataFrame | None" = None,
    store: "PersistentFactorStore | None" = None,
    experience_loop: "ExperienceLoop | None" = None,
) -> dict[str, Any]:
    """将种子因子注入因子库和经验回路。

    Args:
        market_df: 市场数据，用于计算种子因子的初始 IC
        store: 因子持久化存储（None 则创建）
        experience_loop: 经验回路（None 则跳过记录）

    Returns:
        dict: {injected_count, factor_ids, ic_results}
    """
    import pandas as pd
    from .blocks import BlockExecutor
    from .models import FactorDirection, FactorEvaluationReport, FactorScorecard, FactorSpec, FactorStatus
    from .runtime import FactorLibraryEntry, PersistentFactorStore

    if store is None:
        store = PersistentFactorStore()

    existing_ids = {e.factor_spec.factor_id for e in store.load_library_entries()}
    seeds = build_seed_factors()
    new_seeds = [s for s in seeds if s.factor_id not in existing_ids]

    if not new_seeds:
        logger.info("种子因子已存在，跳过冷启动引导")
        return {"injected_count": 0, "factor_ids": [], "ic_results": {}}

    direction_map = {
        "higher_is_better": FactorDirection.HIGHER_IS_BETTER,
        "lower_is_better": FactorDirection.LOWER_IS_BETTER,
    }

    executor = BlockExecutor()
    from .blocks import Block
    injected = []
    ic_results: dict[str, dict] = {}

    for seed in new_seeds:
        try:
            root = Block.from_dict(seed.block_tree)
            factor_values = executor.execute(root, market_df) if market_df is not None else None

            spec = FactorSpec(
                factor_id=seed.factor_id,
                name=seed.name,
                version="v1_seed",
                description=seed.description,
                hypothesis=seed.mechanism,
                family=seed.family,
                direction=direction_map.get(seed.direction, FactorDirection.UNKNOWN),
                status=FactorStatus.OBSERVE,
                tags=["seed", "academic", seed.family],
                source="cold_start_bootstrap",
                created_from="seed",
            )

            entry = FactorLibraryEntry(
                factor_spec=spec,
                latest_report=FactorEvaluationReport(
                    report_id=f"seeded_{seed.factor_id}",
                    factor_spec=spec,
                    scorecard=FactorScorecard(),
                ),
                retention_reason=seed.mechanism,
            )
            store.upsert_library_entry(entry, factor_panel=factor_values)

            if market_df is not None and factor_values is not None:
                ic_data = _compute_seed_ic(factor_values, market_df)
                ic_results[seed.factor_id] = ic_data

            injected.append(seed.factor_id)
            logger.info("种子因子注入: %s (%s)", seed.factor_id, seed.name)

        except Exception as exc:
            logger.warning("种子因子 %s 注入失败: %s", seed.factor_id, exc)

    # Record outcomes in experience loop
    if experience_loop is not None and injected:
        from .factor_enhancements import FactorOutcome
        outcomes = []
        for seed in new_seeds:
            if seed.factor_id not in injected:
                continue
            ic = ic_results.get(seed.factor_id, {})
            outcomes.append(FactorOutcome(
                outcome_id=f"seed_{seed.factor_id}",
                direction=seed.family,
                hypothesis_intuition=seed.intuition,
                mechanism=seed.mechanism,
                pseudocode=seed.description,
                input_fields=seed.input_fields,
                block_tree_desc=f"{seed.family}: {seed.name}",
                verdict="useful" if ic.get("rank_ic_mean", 0) > 0.015 else "marginal",
                rank_ic=ic.get("rank_ic_mean", 0) or 0.0,
                ic_ir=ic.get("ic_ir", 0) or 0.0,
                coverage=ic.get("coverage", 0) or 1.0,
                risk_exposure={},
            ))
        if outcomes:
            experience_loop.record_batch(outcomes)

    return {
        "injected_count": len(injected),
        "factor_ids": injected,
        "ic_results": ic_results,
    }


def _compute_seed_ic(factor_values: "pd.Series", market_df: "pd.DataFrame") -> dict[str, float]:
    """计算种子因子的 Rank IC。"""
    from quantlab.metrics import compute_rank_ic
    return compute_rank_ic(factor_values, market_df)
