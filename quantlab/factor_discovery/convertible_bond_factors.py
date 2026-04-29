"""P3-3 可转债因子模板 —— 可转债 α 因子种子模板。

提供：
1. 可转债数据字段定义（价格 + 溢价率 + 转换价值 + Delta）
2. 5 个 Block 系统兼容的种子因子模板
3. 注册到 MultiAssetContext 的占位数据路径

设计原则：轻量模板，不做真实数据拉取。
"""

from __future__ import annotations

from typing import Any

from quantlab.factor_discovery.blocks import (
    BLOCK_TYPE_COMBINE,
    BLOCK_TYPE_DATA,
    BLOCK_TYPE_TRANSFORM,
    Block,
)

# ── 可转债数据字段 ──────────────────────────────────────────────────

CB_FIELDS: list[str] = [
    "close",
    "volume",
    "premium_rate",       # 转股溢价率
    "conversion_value",   # 转换价值
    "conversion_price",   # 转股价
    "bond_floor",         # 纯债价值（债底）
    "parity",             # 平价
    "delta",              # 可转债 Delta（对正股敏感度）
    "underlying_price",   # 正股价格
    "issuance_size",      # 发行规模
    "days_to_maturity",   # 剩余期限（天）
    "put_price",          # 回售价
    "call_price",         # 强赎触发价
]

# ── 种子因子模板 ──────────────────────────────────────────────────


def build_cb_seed_factors() -> dict[str, dict[str, Any]]:
    """构建 5 个可转债因子 Block 模板。

    返回 {
        "cb_premium": {...},        # 低溢价率因子
        "cb_delta": {...},          # 高 Delta 因子
        "cb_floor_buffer": {...},   # 债底缓冲因子
        "cb_issuance_size": {...},  # 小规模因子
        "cb_double_low": {...},     # 双低策略因子
    }
    每个 value 是一个 Block 树的 dict 表示，可直接经 BlockExecutor 执行。
    所有模板均通过 Block.from_dict() 反序列化验证。
    """

    # 辅助：构造 constant(value) 节点
    def _constant(value: float, ref_field: str = "close") -> dict:
        return {
            "block_type": BLOCK_TYPE_TRANSFORM,
            "op": "constant",
            "params": {"value": value},
            "input_block": {
                "block_type": BLOCK_TYPE_DATA,
                "field_name": ref_field,
            },
        }

    # 1. 低溢价率: -premium_rate → rank
    #    溢价率越低，转债对正股波动越敏感，上涨弹性越大
    cb_premium = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "mul",
            "left": _constant(-1.0),
            "right": {
                "block_type": BLOCK_TYPE_DATA,
                "field_name": "premium_rate",
            },
        },
    }

    # 2. 高 Delta: delta → rank
    #    Delta 越高转债越接近权益属性，与正股联动更强
    cb_delta = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_DATA,
            "field_name": "delta",
        },
    }

    # 3. 债底缓冲: (close - bond_floor) / bond_floor → rank
    #    价格相对债底的溢价越低，下行保护越强
    cb_floor_buffer = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "div",
            "left": {
                "block_type": BLOCK_TYPE_COMBINE,
                "op": "sub",
                "left": {
                    "block_type": BLOCK_TYPE_DATA,
                    "field_name": "close",
                },
                "right": {
                    "block_type": BLOCK_TYPE_DATA,
                    "field_name": "bond_floor",
                },
            },
            "right": {
                "block_type": BLOCK_TYPE_DATA,
                "field_name": "bond_floor",
            },
        },
    }

    # 4. 小规模: -log(issuance_size) → rank
    #    规模越小，分析师覆盖越少，定价效率越低，α 机会越大
    cb_issuance_size = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "mul",
            "left": _constant(-1.0),
            "right": {
                "block_type": BLOCK_TYPE_TRANSFORM,
                "op": "log",
                "params": {},
                "input_block": {
                    "block_type": BLOCK_TYPE_DATA,
                    "field_name": "issuance_size",
                },
            },
        },
    }

    # 5. 双低策略: combine(mul, rank(-premium_rate), rank(-close/underlying_price * 100)) → rank
    #    双低 = 低价格 + 低溢价率，综合选债的经典策略
    #    rank1: rank(-premium_rate)
    rank1 = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "mul",
            "left": _constant(-1.0),
            "right": {
                "block_type": BLOCK_TYPE_DATA,
                "field_name": "premium_rate",
            },
        },
    }

    #    rank2: rank(-close / underlying_price * 100)
    rank2 = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "mul",
            "left": {
                "block_type": BLOCK_TYPE_COMBINE,
                "op": "mul",
                "left": _constant(-1.0),
                "right": {
                    "block_type": BLOCK_TYPE_COMBINE,
                    "op": "div",
                    "left": {
                        "block_type": BLOCK_TYPE_DATA,
                        "field_name": "close",
                    },
                    "right": {
                        "block_type": BLOCK_TYPE_DATA,
                        "field_name": "underlying_price",
                    },
                },
            },
            "right": _constant(100.0),
        },
    }

    cb_double_low = {
        "block_type": BLOCK_TYPE_TRANSFORM,
        "op": "rank",
        "params": {},
        "input_block": {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": "mul",
            "left": rank1,
            "right": rank2,
        },
    }

    return {
        "cb_premium": cb_premium,
        "cb_delta": cb_delta,
        "cb_floor_buffer": cb_floor_buffer,
        "cb_issuance_size": cb_issuance_size,
        "cb_double_low": cb_double_low,
    }


# ── 因子方向 ──────────────────────────────────────────────────────

CB_DIRECTIONS: list[str] = ["premium", "delta", "floor", "size", "double_low"]


# ── 注册到 MultiAssetContext ──────────────────────────────────────


def register_cb_in_context(ctx: Any, data_path: str = "data/cb_placeholder.csv") -> None:
    """将可转债品种注册到 MultiAssetContext（占位数据路径）。"""
    ctx.register("convertible_bond", data_path=data_path, store_dir="assistant_data/convertible_bond")


# ── 自检：验证所有模板均可通过 Block 系统反序列化 ────────────────


def _validate_templates() -> list[str]:
    """验证所有种子模板。返回失败的模板 ID 列表。"""
    failures: list[str] = []
    templates = build_cb_seed_factors()
    for tid, tree_dict in templates.items():
        try:
            Block.from_dict(tree_dict)
        except Exception as exc:
            failures.append(f"{tid}: {exc}")
    return failures
