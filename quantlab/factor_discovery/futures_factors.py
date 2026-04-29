"""P3-1 期货因子模板 —— 指数期货 α 因子种子模板。

提供：
1. 期货数据字段定义（量价 + 基差 + 期限结构）
2. 3 个 Block 系统兼容的种子因子模板
3. 注册到 MultiAssetContext 的占位数据路径

设计原则：轻量模板，不做真实数据拉取。
"""

from __future__ import annotations

from typing import Any

from quantlab.factor_discovery.blocks import (
    BLOCK_TYPE_COMBINE,
    BLOCK_TYPE_DATA,
    BLOCK_TYPE_TRANSFORM,
)

# ── 期货数据字段 ──────────────────────────────────────────────────

INDEX_FUTURE_FIELDS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "basis",           # 基差 (spot - future) / spot
    "term_structure",  # 期限结构 near_month / far_month - 1
]

# ── 种子因子模板 ──────────────────────────────────────────────────


def build_futures_seed_factors() -> dict[str, dict[str, Any]]:
    """构建 3 个期货因子 Block 模板。

    返回 {
        "basis_factor": {...},       # 升贴水因子
        "term_structure_factor": {...},  # 期限结构因子
        "oi_momentum_factor": {...},     # 持仓量动量因子
    }
    每个 value 是一个 Block 树的 dict 表示，可直接经 BlockExecutor 执行。
    """

    # 1. 升贴水因子: (spot - future) / spot → rank
    basis_factor = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_DATA,
            "field_name": "basis",
        },
    }

    # 2. 期限结构因子: near_month_close / far_month_close - 1 → rank
    term_structure_factor = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_DATA,
            "field_name": "term_structure",
        },
    }

    # 3. 持仓量动量因子: delta(open_interest, 5) → rank
    oi_momentum_factor = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_TRANSFORM,
            "op": "delta",
            "params": {"window": 5},
            "input_block": {
                "block_type": BLOCK_TYPE_DATA,
                "field_name": "open_interest",
            },
        },
    }

    return {
        "basis_factor": basis_factor,
        "term_structure_factor": term_structure_factor,
        "oi_momentum_factor": oi_momentum_factor,
    }


# ── 注册到 MultiAssetContext ──────────────────────────────────────


def register_futures_in_context(ctx: Any, data_path: str = "data/futures_placeholder.csv") -> None:
    """将期货品种注册到 MultiAssetContext（占位数据路径）。"""
    ctx.register("index_future", data_path=data_path, store_dir="assistant_data/futures")
