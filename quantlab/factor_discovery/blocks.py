"""因子积木体系 v2 —— 从模板组合到可拼装的 Block 系统。

三类基础积木 + 两类扩展积木：
- DataBlock:     数据积木，提供原始字段
- TransformBlock: 变换积木，对单序列做截面/时序/条件/分组/递推操作
- CombineBlock:  组合积木，将两个积木合成一个
- RelationalBlock: 关系积木，跨资产查表/组内聚合/截面回归
- FilterBlock:   筛选积木，条件过滤/加权

设计原则：
1. 每块积木输入输出类型严格（Series → Series）
2. 积木树可直接序列化为 JSON（供 Agent 传递）
3. BlockExecutor 统一执行，含类型检查和 NaN 保护
4. 编程 Agent 只需拼积木描述，不需要写 Python 代码
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 积木类型枚举 ──────────────────────────────────────────────

BLOCK_TYPE_DATA = "data"
BLOCK_TYPE_TRANSFORM = "transform"
BLOCK_TYPE_COMBINE = "combine"
BLOCK_TYPE_RELATIONAL = "relational"
BLOCK_TYPE_FILTER = "filter"


# ── 算子注册表 ────────────────────────────────────────────────

class OperatorRegistry:
    """所有可用算子的注册中心。"""

    # Transform 算子
    TRANSFORM_OPS = {
        # 截面类
        "rank": {"arity": 1, "group_aware": False, "desc": "截面排名（百分位）"},
        "zscore": {"arity": 1, "group_aware": False, "desc": "截面 Z-Score 标准化"},
        "quantile": {"arity": 1, "group_aware": False, "params": ["n"], "desc": "截面分位数分组（n 组）"},
        "top_n": {"arity": 1, "group_aware": False, "params": ["n"], "desc": "截面前 N 名标记为 1"},
        "bottom_n": {"arity": 1, "group_aware": False, "params": ["n"], "desc": "截面后 N 名标记为 1"},
        # 时序类
        "delta": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "时序差分 x_t - x_{t-w}"},
        "lag": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "时序滞后 x_{t-w}"},
        "ts_mean": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动均值"},
        "ts_std": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动标准差"},
        "ts_rank": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动时序排名"},
        "ts_max": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动最大值"},
        "ts_min": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动最小值"},
        "ts_sum": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动求和"},
        "ts_corr": {"arity": 2, "group_aware": False, "params": ["window"], "desc": "滚动相关系数（需第二输入）"},
        "ts_argmax": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动最大值位置"},
        "ts_argmin": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动最小值位置"},
        # 条件类
        "abs": {"arity": 1, "group_aware": False, "desc": "绝对值"},
        "sign": {"arity": 1, "group_aware": False, "desc": "符号函数"},
        "log": {"arity": 1, "group_aware": False, "desc": "自然对数（自动 clip 下界）"},
        "sigmoid": {"arity": 1, "group_aware": False, "desc": "1 / (1 + exp(-x))"},
        "clip": {"arity": 1, "group_aware": False, "params": ["lo", "hi"], "desc": "裁剪到 [lo, hi]"},
        "piecewise": {"arity": 1, "group_aware": False, "params": ["threshold"], "desc": "x > threshold → 1, else 0"},
        # 分组类
        "group_neutralize": {"arity": 1, "group_aware": True, "desc": "组内去均值（行业中性）"},
        "group_rank": {"arity": 1, "group_aware": True, "desc": "组内排名"},
        "group_zscore": {"arity": 1, "group_aware": True, "desc": "组内 Z-Score"},
        "group_top_n": {"arity": 1, "group_aware": True, "params": ["n"], "desc": "组内前 N 名"},
        "group_bottom_n": {"arity": 1, "group_aware": True, "params": ["n"], "desc": "组内后 N 名"},
        # 递推类
        "ema": {"arity": 1, "group_aware": False, "params": ["alpha"], "desc": "指数移动平均"},
        "rolling_ols_residual": {"arity": 1, "group_aware": False, "params": ["window"], "desc": "滚动 OLS 对市场回归取残差"},
        "constant": {"arity": 0, "group_aware": False, "params": ["value"], "desc": "常量值（忽略输入，返回固定标量）"},
    }

    # Combine 算子
    COMBINE_OPS = {
        "add": {"arity": 2, "desc": "a + b"},
        "sub": {"arity": 2, "desc": "a - b"},
        "mul": {"arity": 2, "desc": "a * b"},
        "div": {"arity": 2, "desc": "a / b（自动保护除零）"},
        "max": {"arity": 2, "desc": "max(a, b)"},
        "min": {"arity": 2, "desc": "min(a, b)"},
        "where": {"arity": 3, "desc": "where(cond, a, b)"},
    }

    # Relational 算子
    RELATIONAL_OPS = {
        "group_aggregate": {"arity": 1, "group_aware": True, "params": ["func"], "desc": "组内聚合后广播回来（mean/median/sum）"},
        "cross_section_ols": {"arity": 1, "group_aware": False, "params": ["factor_cols"], "desc": "截面回归取残差"},
        "event_filter": {"arity": 1, "group_aware": False, "params": ["event_col", "window"], "desc": "按事件日筛选后计算"},
    }

    # Filter 算子
    FILTER_OPS = {
        "where_condition": {"arity": 1, "desc": "条件筛选（只保留满足条件的行）"},
        "sample_weight": {"arity": 1, "desc": "加权（不筛选，只改变权重）"},
    }

    # Data 字段
    DATA_FIELDS = {
        # Price
        "open": {"category": "price", "desc": "开盘价"},
        "high": {"category": "price", "desc": "最高价"},
        "low": {"category": "price", "desc": "最低价"},
        "close": {"category": "price", "desc": "收盘价"},
        "vwap": {"category": "price", "desc": "成交量加权均价"},
        # Volume
        "volume": {"category": "volume", "desc": "成交量"},
        "amount": {"category": "volume", "desc": "成交额"},
        "turnover": {"category": "volume", "desc": "换手率"},
        # Fundamental
        "market_cap": {"category": "fundamental", "desc": "总市值"},
        "float_market_cap": {"category": "fundamental", "desc": "流通市值"},
        "pe": {"category": "fundamental", "desc": "市盈率"},
        "pb": {"category": "fundamental", "desc": "市净率"},
        # Derived
        "pct_chg": {"category": "derived", "desc": "涨跌幅"},
        "industry": {"category": "derived", "desc": "行业代码（分组键）"},
        "adj_factor": {"category": "derived", "desc": "复权因子"},
    }

    @classmethod
    def all_transform_ops(cls) -> list[str]:
        return list(cls.TRANSFORM_OPS.keys())

    @classmethod
    def all_combine_ops(cls) -> list[str]:
        return list(cls.COMBINE_OPS.keys())

    @classmethod
    def all_relational_ops(cls) -> list[str]:
        return list(cls.RELATIONAL_OPS.keys())

    @classmethod
    def all_filter_ops(cls) -> list[str]:
        return list(cls.FILTER_OPS.keys())

    @classmethod
    def all_data_fields(cls) -> list[str]:
        return list(cls.DATA_FIELDS.keys())

    @classmethod
    def op_info(cls, op_type: str, op_name: str) -> dict | None:
        if op_type == "transform":
            return cls.TRANSFORM_OPS.get(op_name)
        elif op_type == "combine":
            return cls.COMBINE_OPS.get(op_name)
        elif op_type == "relational":
            return cls.RELATIONAL_OPS.get(op_name)
        elif op_type == "filter":
            return cls.FILTER_OPS.get(op_name)
        return None


# ── 积木基类 ──────────────────────────────────────────────────

@dataclass
class Block(ABC):
    """积木基类。所有积木输出 pd.Series（对齐到截面索引：date × asset）。"""

    block_type: str

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @abstractmethod
    def required_fields(self) -> list[str]:
        """该积木树需要从 DataFrame 中读取的字段列表。"""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        bt = d.get("block_type")
        if bt == BLOCK_TYPE_DATA:
            return DataBlock.from_dict(d)
        elif bt == BLOCK_TYPE_TRANSFORM:
            return TransformBlock.from_dict(d)
        elif bt == BLOCK_TYPE_COMBINE:
            return CombineBlock.from_dict(d)
        elif bt == BLOCK_TYPE_RELATIONAL:
            return RelationalBlock.from_dict(d)
        elif bt == BLOCK_TYPE_FILTER:
            return FilterBlock.from_dict(d)
        raise ValueError(f"未知积木类型: {bt}")


@dataclass
class DataBlock(Block):
    """数据积木 —— 提供原始字段。"""
    field_name: str
    block_type: str = field(default=BLOCK_TYPE_DATA, init=False)

    def to_dict(self) -> dict:
        return {"block_type": BLOCK_TYPE_DATA, "field_name": self.field_name}

    @classmethod
    def from_dict(cls, d: dict) -> "DataBlock":
        return cls(field_name=d["field_name"])

    def required_fields(self) -> list[str]:
        return [self.field_name]


@dataclass
class TransformBlock(Block):
    """变换积木 —— 对单个输入做截面/时序/条件/分组/递推操作。"""
    op: str
    input_block: Block
    params: dict[str, Any] = field(default_factory=dict)
    group_key: str | None = None  # 分组键，如 "industry"
    block_type: str = field(default=BLOCK_TYPE_TRANSFORM, init=False)

    def to_dict(self) -> dict:
        return {
            "block_type": BLOCK_TYPE_TRANSFORM,
            "op": self.op,
            "input_block": self.input_block.to_dict(),
            "params": self.params,
            "group_key": self.group_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransformBlock":
        return cls(
            op=d["op"],
            input_block=Block.from_dict(d["input_block"]),
            params=d.get("params", {}),
            group_key=d.get("group_key"),
        )

    def required_fields(self) -> list[str]:
        fields = self.input_block.required_fields()
        if self.group_key and self.group_key not in fields:
            fields.append(self.group_key)
        return fields


@dataclass
class CombineBlock(Block):
    """组合积木 —— 将两个（或三个）积木合成一个。"""
    op: str
    left: Block
    right: Block
    cond: Block | None = None  # 仅 where 算子需要
    block_type: str = field(default=BLOCK_TYPE_COMBINE, init=False)

    def to_dict(self) -> dict:
        d = {
            "block_type": BLOCK_TYPE_COMBINE,
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }
        if self.cond is not None:
            d["cond"] = self.cond.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CombineBlock":
        return cls(
            op=d["op"],
            left=Block.from_dict(d["left"]),
            right=Block.from_dict(d["right"]),
            cond=Block.from_dict(d["cond"]) if "cond" in d else None,
        )

    def required_fields(self) -> list[str]:
        fields = self.left.required_fields() + self.right.required_fields()
        if self.cond:
            fields += self.cond.required_fields()
        return list(set(fields))


@dataclass
class RelationalBlock(Block):
    """关系积木 —— 跨资产查表/组内聚合/截面回归。"""
    op: str
    input_block: Block
    params: dict[str, Any] = field(default_factory=dict)
    group_key: str | None = None
    block_type: str = field(default=BLOCK_TYPE_RELATIONAL, init=False)

    def to_dict(self) -> dict:
        return {
            "block_type": BLOCK_TYPE_RELATIONAL,
            "op": self.op,
            "input_block": self.input_block.to_dict(),
            "params": self.params,
            "group_key": self.group_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationalBlock":
        return cls(
            op=d["op"],
            input_block=Block.from_dict(d["input_block"]),
            params=d.get("params", {}),
            group_key=d.get("group_key"),
        )

    def required_fields(self) -> list[str]:
        fields = self.input_block.required_fields()
        if self.group_key and self.group_key not in fields:
            fields.append(self.group_key)
        for fc in self.params.get("factor_cols", []):
            if fc not in fields:
                fields.append(fc)
        return fields


@dataclass
class FilterBlock(Block):
    """筛选积木 —— 条件过滤/加权。"""
    op: str
    input_block: Block
    cond_block: Block | None = None
    block_type: str = field(default=BLOCK_TYPE_FILTER, init=False)

    def to_dict(self) -> dict:
        d = {
            "block_type": BLOCK_TYPE_FILTER,
            "op": self.op,
            "input_block": self.input_block.to_dict(),
        }
        if self.cond_block is not None:
            d["cond_block"] = self.cond_block.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FilterBlock":
        return cls(
            op=d["op"],
            input_block=Block.from_dict(d["input_block"]),
            cond_block=Block.from_dict(d["cond_block"]) if "cond_block" in d else None,
        )

    def required_fields(self) -> list[str]:
        fields = self.input_block.required_fields()
        if self.cond_block:
            fields += self.cond_block.required_fields()
        return list(set(fields))


# ── 积木执行器 ────────────────────────────────────────────────

class BlockExecutor:
    """统一执行积木树，产出因子面板（date × asset → float）。

    输入 DataFrame 必须包含列: date, asset（或 ts_code）
    以及积木所需的数据字段。
    """

    def __init__(self, date_col: str = "date", asset_col: str = "asset"):
        self.date_col = date_col
        self.asset_col = asset_col

    def execute(self, block: Block, df: pd.DataFrame) -> pd.Series:
        """执行积木树，返回一个 Series（index = date × asset）。"""
        # 确保有双索引
        if df.index.name != self.date_col:
            if self.date_col in df.columns and self.asset_col in df.columns:
                df = df.set_index([self.date_col, self.asset_col])
            elif self.date_col in df.columns and "ts_code" in df.columns:
                df = df.rename(columns={"ts_code": self.asset_col})
                df = df.set_index([self.date_col, self.asset_col])

        result = self._exec_block(block, df)
        return result

    def _get_series(self, block: Block, df: pd.DataFrame) -> pd.Series:
        """递归执行积木树。"""
        return self._exec_block(block, df)

    def _exec_block(self, block: Block, df: pd.DataFrame) -> pd.Series:
        if isinstance(block, DataBlock):
            return self._exec_data(block, df)
        elif isinstance(block, TransformBlock):
            return self._exec_transform(block, df)
        elif isinstance(block, CombineBlock):
            return self._exec_combine(block, df)
        elif isinstance(block, RelationalBlock):
            return self._exec_relational(block, df)
        elif isinstance(block, FilterBlock):
            return self._exec_filter(block, df)
        raise ValueError(f"未知积木: {type(block)}")

    # ── Data ──

    def _exec_data(self, block: DataBlock, df: pd.DataFrame) -> pd.Series:
        if block.field_name not in df.columns:
            raise ValueError(f"DataFrame 缺少字段: {block.field_name}")
        return df[block.field_name].astype(float)

    # ── Transform ──

    def _exec_transform(self, block: TransformBlock, df: pd.DataFrame) -> pd.Series:
        inp = self._get_series(block.input_block, df)
        op = block.op
        params = block.params

        # 截面类
        if op == "rank":
            return inp.groupby(level=self.date_col).rank(pct=True)
        elif op == "zscore":
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
        elif op == "quantile":
            n = int(params.get("n", 5))
            return inp.groupby(level=self.date_col).transform(
                lambda x: pd.qcut(x.rank(method="first"), n, labels=False, duplicates="drop") / (n - 1)
            )
        elif op == "top_n":
            n = int(params.get("n", 10))
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x.rank(ascending=False) <= n).astype(float)
            )
        elif op == "bottom_n":
            n = int(params.get("n", 10))
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x.rank(ascending=True) <= n).astype(float)
            )

        # 时序类
        elif op == "delta":
            w = int(params.get("window", 1))
            return inp.groupby(level=self.asset_col).diff(w)
        elif op == "lag":
            w = int(params.get("window", 1))
            return inp.groupby(level=self.asset_col).shift(w)
        elif op == "ts_mean":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        elif op == "ts_std":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        elif op == "ts_rank":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).rank(pct=True)
            )
        elif op == "ts_max":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
        elif op == "ts_min":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).min()
            )
        elif op == "ts_sum":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
        elif op == "ts_argmax":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).apply(np.argmax, raw=True) / w
            )
        elif op == "ts_argmin":
            w = int(params.get("window", 20))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).apply(np.argmin, raw=True) / w
            )
        elif op == "ts_corr":
            # 特殊：需要第二输入
            w = int(params.get("window", 20))
            right_block = params.get("_right_block")
            if right_block:
                right_inp = self._get_series(Block.from_dict(right_block), df)
            else:
                # 如果没有第二输入，用自身（无意义但不会崩）
                right_inp = inp
            combined = pd.DataFrame({"left": inp, "right": right_inp})
            return combined.groupby(level=self.asset_col).apply(
                lambda g: g["left"].rolling(w, min_periods=1).corr(g["right"])
            ).droplevel(0) if len(combined) > 0 else inp * 0

        # 条件类
        elif op == "abs":
            return inp.abs()
        elif op == "sign":
            return np.sign(inp)
        elif op == "log":
            return np.log(inp.clip(lower=1e-10))
        elif op == "sigmoid":
            return 1.0 / (1.0 + np.exp(-inp.clip(-50, 50)))
        elif op == "clip":
            lo = float(params.get("lo", -3.0))
            hi = float(params.get("hi", 3.0))
            return inp.clip(lo, hi)
        elif op == "piecewise":
            threshold = float(params.get("threshold", 0))
            return (inp > threshold).astype(float)

        # 分组类
        elif op == "group_neutralize":
            gk = block.group_key or "industry"
            if gk in df.columns:
                group = df[gk]
                return inp.groupby([pd.Grouper(level=self.date_col), group]).transform(
                    lambda x: x - x.mean()
                )
            return inp
        elif op == "group_rank":
            gk = block.group_key or "industry"
            if gk in df.columns:
                group = df[gk]
                return inp.groupby([pd.Grouper(level=self.date_col), group]).rank(pct=True)
            return inp.groupby(level=self.date_col).rank(pct=True)
        elif op == "group_zscore":
            gk = block.group_key or "industry"
            if gk in df.columns:
                group = df[gk]
                return inp.groupby([pd.Grouper(level=self.date_col), group]).transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
        elif op == "group_top_n":
            gk = block.group_key or "industry"
            n = int(params.get("n", 3))
            if gk in df.columns:
                group = df[gk]
                return inp.groupby([pd.Grouper(level=self.date_col), group]).transform(
                    lambda x: (x.rank(ascending=False) <= n).astype(float)
                )
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x.rank(ascending=False) <= n).astype(float)
            )
        elif op == "group_bottom_n":
            gk = block.group_key or "industry"
            n = int(params.get("n", 3))
            if gk in df.columns:
                group = df[gk]
                return inp.groupby([pd.Grouper(level=self.date_col), group]).transform(
                    lambda x: (x.rank(ascending=True) <= n).astype(float)
                )
            return inp.groupby(level=self.date_col).transform(
                lambda x: (x.rank(ascending=True) <= n).astype(float)
            )

        # 条件类
        elif op == "constant":
            value = float(params.get("value", 0.0))
            return pd.Series(value, index=df.index, dtype=float)

        # 递推类
        elif op == "ema":
            alpha = float(params.get("alpha", 0.1))
            return inp.groupby(level=self.asset_col).transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
        elif op == "rolling_ols_residual":
            # 简化版：对截面均值做回归取残差
            w = int(params.get("window", 60))
            market_mean = inp.groupby(level=self.date_col).mean()
            market_aligned = market_mean.reindex(inp.index, level=self.date_col)
            residual = inp - market_aligned
            return residual.groupby(level=self.asset_col).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

        raise ValueError(f"未知 Transform 算子: {op}")

    # ── Combine ──

    def _exec_combine(self, block: CombineBlock, df: pd.DataFrame) -> pd.Series:
        left = self._get_series(block.left, df)
        right = self._get_series(block.right, df)
        op = block.op

        if op == "add":
            return left + right
        elif op == "sub":
            return left - right
        elif op == "mul":
            return left * right
        elif op == "div":
            return left / (right.replace(0, np.nan).abs().clip(lower=1e-10) * np.sign(right.replace(0, 1)))
        elif op == "max":
            return np.maximum(left, right)
        elif op == "min":
            return np.minimum(left, right)
        elif op == "where":
            cond = self._get_series(block.cond, df) if block.cond else (left > right)
            return pd.Series(np.where(cond > 0, left, right), index=left.index)

        raise ValueError(f"未知 Combine 算子: {op}")

    # ── Relational ──

    def _exec_relational(self, block: RelationalBlock, df: pd.DataFrame) -> pd.Series:
        inp = self._get_series(block.input_block, df)
        op = block.op
        params = block.params

        if op == "group_aggregate":
            func = params.get("func", "mean")
            gk = block.group_key or "industry"
            if gk in df.columns:
                group = df[gk]
                agg = inp.groupby([pd.Grouper(level=self.date_col), group]).transform(func)
                return agg
            return inp.groupby(level=self.date_col).transform(func)

        elif op == "cross_section_ols":
            # 简化：对指定因子做截面回归取残差
            factor_cols = params.get("factor_cols", [])
            if not factor_cols:
                return inp
            # 构造设计矩阵
            X_parts = []
            for fc in factor_cols:
                if fc in df.columns:
                    X_parts.append(df[fc].astype(float))
            if not X_parts:
                return inp
            # 每个截面做 OLS
            def _ols_residual(y_group):
                # 简化：减去截面均值
                return y_group - y_group.mean()
            return inp.groupby(level=self.date_col).transform(_ols_residual)

        elif op == "event_filter":
            event_col = params.get("event_col", "")
            window = int(params.get("window", 5))
            if event_col and event_col in df.columns:
                events = df[event_col].astype(float)
                mask = events.groupby(level=self.asset_col).transform(
                    lambda x: x.rolling(window, min_periods=1).max() > 0
                )
                return inp * mask.astype(float)
            return inp

        raise ValueError(f"未知 Relational 算子: {op}")

    # ── Filter ──

    def _exec_filter(self, block: FilterBlock, df: pd.DataFrame) -> pd.Series:
        inp = self._get_series(block.input_block, df)
        op = block.op

        if op == "where_condition":
            if block.cond_block:
                cond = self._get_series(block.cond_block, df)
                return inp.where(cond > 0, np.nan)
            return inp
        elif op == "sample_weight":
            if block.cond_block:
                weights = self._get_series(block.cond_block, df)
                return inp * weights.abs()
            return inp

        raise ValueError(f"未知 Filter 算子: {op}")


# ── 因子描述 & FactorHypothesis ───────────────────────────────

@dataclass
class FactorHypothesis:
    """调研团队 → 编程团队 → 测试团队 的标准传递物。"""
    hypothesis_id: str
    direction: str

    # 经济逻辑（调研团队）
    intuition: str = ""
    mechanism: str = ""
    expected_behavior: str = ""
    risk_factors: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # 数学表达
    pseudocode: str = ""
    input_fields: list[str] = field(default_factory=list)
    output_type: str = "cross_section"

    # 积木树（编程团队填充）
    block_tree: dict | None = None

    # 评审记录
    review_status: str = "pending"  # pending / approved / rejected / revised
    review_comments: str = ""
    iteration: int = 0

    # 编程团队
    implementation: str | None = None
    code_path: str | None = None

    # 测试团队
    test_result: dict | None = None
    verdict: str | None = None  # useful / marginal / useless

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FactorHypothesis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProgrammingPlan:
    """P1 架构师输出的编程方案。"""
    factor_id: str
    block_plan: list[dict] = field(default_factory=list)
    custom_requests: list[dict] = field(default_factory=list)
    integration_spec: str = ""


@dataclass
class CustomRequest:
    """P2 工头给 P3 匠人的定制工单。"""
    request_id: str
    factor_id: str
    what: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    must_pass: list[str] = field(default_factory=list)
    performance_hint: str = ""
    target_path: str = ""
    integration_interface: str = ""


# ── 便捷构造器 ────────────────────────────────────────────────

def data(field_name: str) -> DataBlock:
    """快捷构造数据积木。"""
    return DataBlock(field_name=field_name)


def transform(op: str, input_block: Block, **params) -> TransformBlock:
    """快捷构造变换积木。"""
    group_key = params.pop("group_key", None)
    return TransformBlock(op=op, input_block=input_block, params=params, group_key=group_key)


def combine(op: str, left: Block, right: Block, cond: Block | None = None) -> CombineBlock:
    """快捷构造组合积木。"""
    return CombineBlock(op=op, left=left, right=right, cond=cond)


def relational(op: str, input_block: Block, **params) -> RelationalBlock:
    """快捷构造关系积木。"""
    group_key = params.pop("group_key", None)
    return RelationalBlock(op=op, input_block=input_block, params=params, group_key=group_key)


def filter_block(op: str, input_block: Block, cond_block: Block | None = None) -> FilterBlock:
    """快捷构造筛选积木。"""
    return FilterBlock(op=op, input_block=input_block, cond_block=cond_block)


# ── FactorNode ↔ Block 转换器 ──────────────────────────────────

# FactorNode node_type → Block operator name mapping
_FACTOR_NODE_OP_MAP: dict[str, str] = {
    # Arithmetic (arity 2)
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    # Cross-section (arity 1)
    "rank": "rank",
    "zscore": "zscore",
    "quantile": "quantile",
    "group_neutralize": "group_neutralize",
    "group_rank": "group_rank",
    "group_zscore": "group_zscore",
    # Time-series (arity 1)
    "delta": "delta",
    "lag": "lag",
    "mean": "ts_mean",
    "std": "ts_std",
    "ts_rank": "ts_rank",
    "min": "ts_min",
    "max": "ts_max",
    "ts_sum": "ts_sum",
    "ts_argmax": "ts_argmax",
    "ts_argmin": "ts_argmin",
    "ema": "ema",
    "rolling_ols_residual": "rolling_ols_residual",
    # Post-process (arity 1)
    "abs": "abs",
    "sign": "sign",
    "log": "log",
    "sigmoid": "sigmoid",
    "clip": "clip",
    "piecewise": "piecewise",
    # Constant (arity 0)
    "constant": "constant",
}

# Arity: 1 = TransformBlock, 2 = CombineBlock, 0 = constant
_FACTOR_NODE_ARITY: dict[str, int] = {
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "rank": 1, "zscore": 1, "quantile": 1,
    "group_neutralize": 1, "group_rank": 1, "group_zscore": 1,
    "delta": 1, "lag": 1, "mean": 1, "std": 1, "ts_rank": 1,
    "min": 1, "max": 1, "ts_sum": 1,
    "ts_argmax": 1, "ts_argmin": 1, "ema": 1, "rolling_ols_residual": 1,
    "abs": 1, "sign": 1, "log": 1, "sigmoid": 1, "clip": 1, "piecewise": 1,
    "constant": 0,
}


def factor_node_to_block(fn: "FactorNode") -> Block:
    """将 FactorNode 表达树转换为 Block 积木树。

    映射规则：
    - feature → DataBlock
    - constant → TransformBlock(op="constant", params={"value": ...})
    - 单子节点算子（rank, zscore, delta, mean, std, ts_rank, clip 等）→ TransformBlock
    - 双子节点算子（add, sub, mul, div）→ CombineBlock
    """
    from .models import FactorNode as FN

    node_type = fn.node_type

    if node_type == "feature":
        return DataBlock(field_name=str(fn.value or ""))

    if node_type == "constant":
        return TransformBlock(
            op="constant",
            input_block=DataBlock(field_name="close"),  # dummy input, ignored
            params={"value": float(fn.value or 0.0)},
        )

    block_op = _FACTOR_NODE_OP_MAP.get(node_type)
    if block_op is None:
        raise ValueError(f"FactorNode 算子 {node_type} 无法映射到 Block")

    arity = _FACTOR_NODE_ARITY.get(node_type, len(fn.children))
    child_blocks = [factor_node_to_block(c) for c in fn.children]

    if arity == 0:
        return TransformBlock(
            op=block_op,
            input_block=DataBlock(field_name="close"),
            params={**fn.params},
        )

    if arity == 1:
        return TransformBlock(
            op=block_op,
            input_block=child_blocks[0],
            params={**fn.params},
        )

    # arity == 2
    return CombineBlock(
        op=block_op,
        left=child_blocks[0],
        right=child_blocks[1] if len(child_blocks) > 1 else child_blocks[0],
    )


def block_to_factor_node(block: Block) -> "FactorNode":
    """将 Block 积木树转换回 FactorNode 表达树。"""
    from .models import FactorNode as FN

    if isinstance(block, DataBlock):
        return FN(node_type="feature", value=block.field_name)

    if isinstance(block, TransformBlock):
        op = block.op
        if op == "constant":
            return FN(
                node_type="constant",
                value=float(block.params.get("value", 0.0)),
            )
        # Reverse map: ts_mean → mean, ts_std → std, etc.
        reverse_map = {
            "ts_mean": "mean", "ts_std": "std", "ts_rank": "ts_rank",
            "ts_min": "min", "ts_max": "max",
        }
        fn_type = reverse_map.get(op, op)
        return FN(
            node_type=fn_type,
            children=[block_to_factor_node(block.input_block)],
            params={**block.params},
        )

    if isinstance(block, CombineBlock):
        return FN(
            node_type=block.op,
            children=[block_to_factor_node(block.left), block_to_factor_node(block.right)],
        )

    raise ValueError(f"未知 Block 类型: {type(block)}")

