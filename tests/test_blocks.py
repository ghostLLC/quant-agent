"""Tests for the Block system and FactorNode ↔ Block conversion."""
import numpy as np
import pandas as pd
import pytest

from quantlab.factor_discovery.blocks import (
    data, transform, combine, relational, filter_block,
    BlockExecutor, Block, DataBlock, TransformBlock, CombineBlock,
    factor_node_to_block, block_to_factor_node,
)
from quantlab.factor_discovery.models import FactorNode


class TestBlockConstruction:
    def test_data_block_serialization(self):
        b = data("close")
        d = b.to_dict()
        assert d == {"block_type": "data", "field_name": "close"}
        restored = Block.from_dict(d)
        assert isinstance(restored, DataBlock)
        assert restored.field_name == "close"

    def test_transform_block_serialization(self):
        b = transform("rank", data("close"))
        d = b.to_dict()
        assert d["block_type"] == "transform"
        assert d["op"] == "rank"
        restored = Block.from_dict(d)
        assert isinstance(restored, TransformBlock)
        assert restored.op == "rank"

    def test_combine_block_serialization(self):
        b = combine("sub", transform("rank", data("close")), transform("rank", data("volume")))
        d = b.to_dict()
        assert d["block_type"] == "combine"
        assert d["op"] == "sub"
        restored = Block.from_dict(d)
        assert isinstance(restored, CombineBlock)
        assert restored.op == "sub"

    def test_nested_block_serialization(self):
        b = combine(
            "div",
            transform("ts_mean", data("close"), window=20),
            transform("ts_std", data("close"), window=20),
        )
        restored = Block.from_dict(b.to_dict())
        assert isinstance(restored, CombineBlock)
        assert isinstance(restored.left, TransformBlock)
        assert restored.left.params["window"] == 20


class TestBlockExecutor:
    def test_execute_data_block(self, sample_market_df):
        executor = BlockExecutor()
        result = executor.execute(data("close"), sample_market_df)
        assert len(result) == len(sample_market_df)
        assert result.notna().sum() > 0

    def test_execute_rank_transform(self, sample_market_df):
        executor = BlockExecutor()
        result = executor.execute(transform("rank", data("close")), sample_market_df)
        assert result.notna().sum() > 0
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_execute_zscore_transform(self, sample_market_df):
        executor = BlockExecutor()
        result = executor.execute(transform("zscore", data("close")), sample_market_df)
        assert result.notna().sum() > 0

    def test_execute_delta_transform(self, sample_market_df):
        executor = BlockExecutor()
        result = executor.execute(transform("delta", data("close"), window=1), sample_market_df)
        assert result.notna().sum() > 0

    def test_execute_combine_add(self, sample_market_df):
        executor = BlockExecutor()
        result = executor.execute(
            combine("add", data("close"), data("volume")), sample_market_df
        )
        assert len(result) == len(sample_market_df)

    def test_execute_complex_tree(self, sample_market_df):
        tree = combine(
            "div",
            transform("zscore", data("close")),
            transform("ts_std", data("volume"), window=20),
        )
        executor = BlockExecutor()
        result = executor.execute(tree, sample_market_df)
        assert result.notna().sum() > 0

    def test_constant_operator(self, sample_market_df):
        result = BlockExecutor().execute(
            transform("constant", data("close"), value=3.14),
            sample_market_df,
        )
        assert result.notna().all()
        assert abs(result.iloc[0] - 3.14) < 0.01


class TestFactorNodeBlockConversion:
    def test_feature_conversion(self):
        fn = FactorNode(node_type="feature", value="close")
        blk = factor_node_to_block(fn)
        assert isinstance(blk, DataBlock)
        assert blk.field_name == "close"
        back = block_to_factor_node(blk)
        assert back.node_type == "feature"
        assert back.value == "close"

    def test_constant_conversion(self):
        fn = FactorNode(node_type="constant", value=5.0)
        blk = factor_node_to_block(fn)
        assert isinstance(blk, TransformBlock)
        assert blk.op == "constant"
        assert blk.params["value"] == 5.0

    def test_rank_conversion(self):
        fn = FactorNode(
            node_type="rank",
            children=[FactorNode(node_type="feature", value="close")],
        )
        blk = factor_node_to_block(fn)
        assert isinstance(blk, TransformBlock)
        assert blk.op == "rank"
        back = block_to_factor_node(blk)
        assert back.node_type == "rank"

    def test_arithmetic_conversion(self):
        fn = FactorNode(
            node_type="add",
            children=[
                FactorNode(node_type="feature", value="close"),
                FactorNode(node_type="feature", value="volume"),
            ],
        )
        blk = factor_node_to_block(fn)
        assert isinstance(blk, CombineBlock)
        assert blk.op == "add"

    def test_deep_tree_roundtrip(self):
        fn = FactorNode(
            node_type="rank",
            children=[
                FactorNode(
                    node_type="div",
                    children=[
                        FactorNode(
                            node_type="delta",
                            children=[FactorNode(node_type="feature", value="close")],
                            params={"window": 5},
                        ),
                        FactorNode(
                            node_type="std",
                            children=[FactorNode(node_type="feature", value="close")],
                            params={"window": 20},
                        ),
                    ],
                )
            ],
        )
        blk = factor_node_to_block(fn)
        back = block_to_factor_node(blk)
        assert back.node_type == "rank"
        assert back.children[0].node_type == "div"

    def test_execution_equivalence(self, sample_market_df):
        """FactorNode→Block→execute should match Block direct execute."""
        fn = FactorNode(
            node_type="rank",
            children=[FactorNode(node_type="feature", value="close")],
        )
        block = data("close")
        block_exec = BlockExecutor()

        via_converter = block_exec.execute(factor_node_to_block(fn), sample_market_df)
        direct = block_exec.execute(transform("rank", block), sample_market_df)

        pd.testing.assert_series_equal(
            via_converter.sort_index(),
            direct.sort_index(),
        )

    def test_momentum_factor_conversion(self):
        """The standard momentum factor roundtrips correctly."""
        fn = FactorNode(
            node_type="rank",
            children=[
                FactorNode(
                    node_type="div",
                    children=[
                        FactorNode(
                            node_type="delta",
                            children=[FactorNode(node_type="feature", value="close")],
                            params={"window": 5},
                        ),
                        FactorNode(
                            node_type="lag",
                            children=[FactorNode(node_type="feature", value="close")],
                            params={"window": 5},
                        ),
                    ],
                )
            ],
        )
        blk = factor_node_to_block(fn)
        assert isinstance(blk, TransformBlock)  # rank wraps it
        assert blk.op == "rank"
        # Inner is a combine (div)
        assert isinstance(blk.input_block, CombineBlock)
        assert blk.input_block.op == "div"
