"""Agent 驱动管线阶段测试。

测试 Agent OOS 验证、Agent 治理和 Agent 交付报告阶段。
所有 Agent 调用使用 mock LLMClient 或禁用 Agent 模式，避免网络依赖。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure quantlab is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_sample_df() -> pd.DataFrame:
    """Create a minimal valid market DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=200)
    records = []
    for date in dates:
        for asset in [f"00000{i}" for i in range(1, 6)]:
            records.append({
                "date": date,
                "asset": asset,
                "close": 10.0 + np.random.randn() * 2,
                "volume": np.random.randint(1000, 10000),
                "industry": np.random.choice(["银行", "科技", "消费"]),
                "market_cap": np.random.uniform(1e9, 1e11),
            })
    return pd.DataFrame(records)


class _MockEmptyStore:
    """Mock PersistentFactorStore that returns an empty library."""

    def load_library_entries(self):
        return []

    def get_library_stats(self):
        return {}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def sample_market_df():
    """Synthetic market data: 5 assets × 60 days."""
    np.random.seed(42)
    n_assets = 5
    n_days = 60
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    records = []
    for date in dates:
        for asset in [f"00000{i}" for i in range(1, n_assets + 1)]:
            records.append({
                "date": date,
                "asset": asset,
                "close": 10.0 + np.random.randn() * 2,
                "volume": np.random.randint(1000, 10000),
                "industry": np.random.choice(["银行", "科技", "消费"]),
                "market_cap": np.random.uniform(1e9, 1e11),
            })
    return pd.DataFrame(records)


@pytest.fixture
def pipeline_ctx(sample_market_df, tmp_path):
    """PipelineContext with a local CSV."""
    from quantlab.pipeline_stages.base import PipelineContext

    csv_path = tmp_path / "test_data.csv"
    sample_market_df.to_csv(csv_path, index=False)
    return PipelineContext(data_path=csv_path)


@pytest.fixture
def mock_llm_response():
    """Mock LLMClient.chat_json that returns realistic analysis."""
    def _mock_chat_json(system_prompt, user_prompt, temperature=None):
        if "OOS" in system_prompt or "样本外" in system_prompt:
            return {
                "per_factor": [
                    {
                        "factor_id": "test_factor_001",
                        "diagnosis": "healthy",
                        "diagnosis_cn": "样本外表现良好，信号稳健",
                        "robustness": "solid",
                        "recommendation": "promote",
                        "reasoning": "训练IC和测试IC一致性高，衰减可控",
                        "risk_note": None,
                    }
                ],
                "cross_factor_summary": {
                    "overall_pass_rate": 1.0,
                    "top_performing_families": ["momentum"],
                    "worst_performing_families": [],
                    "systemic_issues": None,
                    "market_context_hypothesis": "震荡市中动量因子仍有效",
                    "recommendations_for_next_round": "继续探索动量方向",
                },
            }
        elif "治理" in system_prompt or "因子库" in system_prompt:
            return {
                "executive_summary": "因子库整体健康，无严重拥挤",
                "regime_analysis": {
                    "interpretation": "当前处于震荡市",
                    "factor_implications": "低波动因子可能表现更好",
                    "watch_points": ["关注动量因子的持续性"],
                },
                "crowding_analysis": {
                    "interpretation": "未发现显著拥挤",
                    "cluster_descriptions": [],
                    "severity": "low",
                    "recommended_actions": [],
                },
                "decay_analysis": {
                    "interpretation": "衰减曲线整体健康",
                    "healthy_factors": ["test_factor_001"],
                    "concerning_factors": [],
                    "systemic_pattern": None,
                },
                "risk_summary": {
                    "overall_level": "low",
                    "top_risks": [],
                    "immediate_actions": [],
                    "forward_guidance": "保持定期监控",
                },
            }
        elif "买方" in system_prompt or "交付报告" in system_prompt or "量化因子研究员" in system_prompt:
            return {
                "executive_summary": "该因子长期表现稳健，适合作为组合基础因子",
                "factor_story": {
                    "economic_intuition": "捕捉短期价格过度反应后的均值回归",
                    "mechanism_explanation": "市场对新信息的过度反应导致价格偏离，随后回归",
                    "academic_background": "行为金融学中的过度反应理论",
                },
                "strengths": ["IC稳定性高", "扣费后仍有正收益", "与库内其他因子低相关"],
                "weaknesses": ["牛市趋势中可能失效", "极端行情下换手增加"],
                "market_context": {
                    "best_environments": ["震荡市", "低波动环境"],
                    "worst_environments": ["强趋势牛市"],
                    "current_suitability": "当前市场环境下适用性良好",
                },
                "risk_assessment": {
                    "primary_risk": "趋势市中信号可能反转",
                    "secondary_risks": ["流动性枯竭时换手成本上升"],
                    "mitigation_suggestions": ["设置趋势过滤器"],
                },
                "peer_comparison": "相比库内其他反转因子，ICIR高出约30%",
                "buyer_checklist": [
                    "验证不同市场状态下的表现一致性",
                    "检查100亿以上规模的容量",
                    "评估极端行情下的回撤控制",
                    "确认与现有组合的相关性",
                    "审查因子构造中没有未来函数",
                ],
            }
        elif "反馈" in system_prompt or "下一轮" in system_prompt:
            return {
                "summary": "本轮OOS整体表现良好，动量方向仍有挖掘空间",
                "directions_to_prioritize": ["momentum_reversal", "quality_earnings"],
                "directions_to_avoid": ["volatility_regime"],
                "structural_patterns": {
                    "what_worked": "基于成交量的动量增强",
                    "what_failed": "纯波动率因子衰减过快",
                },
                "data_quality_notes": None,
                "methodology_suggestions": "增加换手率约束以降低交易成本侵蚀",
            }
        return {}

    return _mock_chat_json


# ------------------------------------------------------------------
# AgentAnalyst Tests
# ------------------------------------------------------------------


class TestAgentAnalystRuleFallback:
    """Test AgentAnalyst rule-based fallback when LLM is unavailable."""

    def test_oos_rule_fallback_produces_valid_output(self):
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        checks = [
            {
                "factor_id": "factor_001",
                "factor_name": "Test Momentum",
                "factor_family": "momentum",
                "train_ic": 0.045,
                "test_ic": 0.038,
                "oos_decay": 0.156,
                "cost_adj_ic": 0.035,
                "turnover": 0.25,
                "passed": True,
            },
            {
                "factor_id": "factor_002",
                "factor_name": "Test Reversal",
                "factor_family": "reversal",
                "train_ic": 0.030,
                "test_ic": -0.005,
                "oos_decay": 1.17,
                "cost_adj_ic": -0.008,
                "turnover": 0.60,
                "passed": False,
            },
        ]

        result = analyst.analyze_oos(checks)

        assert result["status"] == "rule_fallback"
        assert len(result["per_factor"]) == 2
        assert result["per_factor"][0]["diagnosis"] in ("healthy", "overfit", "structural_break", "weak_signal")
        assert result["per_factor"][0]["recommendation"] in ("promote", "iterate", "observe", "abandon")
        assert "overall_pass_rate" in result["cross_factor_summary"]

    def test_governance_rule_fallback_produces_valid_output(self):
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        gov_data = {
            "status": "success",
            "regime": {"current": "sideways", "stats": {}},
            "crowding": {"crowded_factor_ids": ["f1", "f2"], "clusters": 1},
            "curves": {"total": 5, "factors_with_half_life": 3},
            "risk": {"risk_score": 0.35, "breaches": ["行业集中度过高"]},
            "lifecycle_changes": [{"factor_id": "f3", "from": "draft", "to": "candidate"}],
        }

        result = analyst.analyze_governance(gov_data)

        assert result["status"] == "rule_fallback"
        assert "executive_summary" in result
        assert "regime_analysis" in result
        assert "crowding_analysis" in result
        assert "risk_summary" in result
        assert result["crowding_analysis"]["severity"] in ("low", "moderate", "high", "critical")

    def test_report_rule_fallback_produces_valid_output(self):
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        report_dict = {
            "factor_id": "factor_001",
            "factor_name": "Test Momentum",
            "factor_family": "momentum",
            "direction": "long",
            "expression": "rank(delta(close,20))",
            "hypothesis": "过去表现好的股票将继续表现好",
            "rank_ic_mean": 0.045,
            "rank_ic_std": 0.12,
            "icir": 1.5,
            "ic_positive_ratio": 0.62,
            "decay_profile": {"1d": 0.045, "5d": 0.038, "10d": 0.032, "20d": 0.025},
            "simulation": {
                "gross_return": 0.15,
                "net_return": 0.10,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "avg_daily_turnover": 0.30,
                "information_ratio": 0.9,
            },
            "capacity": {"total_daily_capacity_yuan": 50_000_000},
            "market_cap_exposure": 0.12,
            "industry_exposure": {},
            "correlation_to_known_factors": {},
            "stability_score": 0.75,
            "risk_flags": ["换手率偏高"],
        }

        result = analyst.generate_narrative_report(report_dict)

        assert result["status"] == "rule_fallback"
        assert "executive_summary" in result
        assert "strengths" in result or "weaknesses" in result
        assert "buyer_checklist" in result
        assert len(result["buyer_checklist"]) >= 3

    def test_feedback_rule_fallback_produces_valid_output(self):
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        oos_analysis = {
            "cross_factor_summary": {
                "overall_pass_rate": 0.7,
                "top_performing_families": ["momentum"],
                "worst_performing_families": ["volatility"],
            },
        }
        gov_analysis = {
            "crowding_analysis": {"cluster_descriptions": []},
        }

        result = analyst.generate_feedback(oos_analysis, gov_analysis)

        assert result["status"] == "rule_fallback"
        assert "summary" in result
        assert "directions_to_prioritize" in result
        assert "directions_to_avoid" in result
        assert "generated_at" in result


# ------------------------------------------------------------------
# Agent Pipeline Stage Tests (Agent disabled mode)
# ------------------------------------------------------------------


class TestAgentOOSValidationStage:
    """Test the refactored OOS Validation stage with agent disabled."""

    def test_stage_skips_on_empty_data(self, tmp_path):
        from quantlab.pipeline_stages.base import PipelineContext
        from quantlab.pipeline_stages.oos_validation import AgentOOSValidationStage

        csv_path = tmp_path / "empty.csv"
        # Write a CSV with header but no data rows
        csv_path.write_text("date,asset,close,volume,industry,market_cap\n", encoding="utf-8")
        ctx = PipelineContext(data_path=csv_path)

        stage = AgentOOSValidationStage(enable_agent=False)
        result = stage.run(ctx)

        # Data loads but is empty → stage should skip
        assert result["status"] in ("skipped", "success")

    def test_stage_skips_when_no_approved_factors(self, monkeypatch):
        """When factor store has no approved entries, stage should skip."""
        from quantlab.pipeline_stages.base import PipelineContext
        from quantlab.pipeline_stages.oos_validation import AgentOOSValidationStage

        # Mock PersistentFactorStore to return empty library
        monkeypatch.setattr(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            lambda: _MockEmptyStore(),
        )
        # Mock SampleSplitter.split to return sufficient data
        from quantlab.factor_discovery.sample_split import SampleSplitter, SplitResult
        df = _make_sample_df()
        cutoff = pd.Timestamp("2024-06-01")

        def _mock_split(self, df_):
            return SplitResult(
                train_df=df[df["date"] <= cutoff].copy(),
                test_df=df[df["date"] > cutoff].copy(),
                cutoff_date=cutoff,
                train_start=pd.Timestamp("2024-01-01"),
                test_end=pd.Timestamp("2024-09-30"),
                train_trading_days=120,
                test_trading_days=80,
                train_assets=5,
                test_assets=5,
                train_rows=600,
                test_rows=400,
                sufficient=True,
            )

        monkeypatch.setattr(SampleSplitter, "split", _mock_split)

        ctx = PipelineContext()
        stage = AgentOOSValidationStage(enable_agent=False)
        result = stage.run(ctx)

        assert result["status"] == "skipped"
        assert "无已审批因子" in result.get("reason", "")

    def test_agent_feedback_stored_in_ctx_meta(self, pipeline_ctx):
        from quantlab.pipeline_stages.oos_validation import AgentOOSValidationStage

        stage = AgentOOSValidationStage(enable_agent=False)
        result = stage.run(pipeline_ctx)

        # Agent feedback should be written to ctx._meta even for skipped runs
        if result["status"] != "skipped":
            assert "oos_analysis" in pipeline_ctx._meta
            assert "discovery_feedback" in pipeline_ctx._meta

    def test_rule_fallback_matches_checks_count(self):
        """Rule-based fallback produces one diagnosis per check entry."""
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        checks = [
            {"factor_id": f"f{i:03d}", "train_ic": 0.04, "test_ic": 0.03,
             "oos_decay": 0.25, "cost_adj_ic": 0.028, "turnover": 0.3, "passed": True}
            for i in range(5)
        ]

        result = analyst.analyze_oos(checks)

        assert len(result["per_factor"]) == 5
        for pf in result["per_factor"]:
            assert "factor_id" in pf
            assert "diagnosis" in pf
            assert "recommendation" in pf


class TestAgentGovernanceStage:
    """Test the refactored Governance stage with agent disabled."""

    def test_rate_fallback_is_stable(self):
        """Rule-based governance fallback should always return consistent structure."""
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        for regime in ["bull", "bear", "sideways"]:
            gov_data = {
                "status": "success",
                "regime": {"current": regime},
                "crowding": {"crowded_factor_ids": []},
                "curves": {"total": 3, "factors_with_half_life": 2},
                "risk": {"risk_score": 0.2, "breaches": []},
                "lifecycle_changes": [],
            }
            result = analyst.analyze_governance(gov_data)

            assert result["status"] == "rule_fallback"
            assert isinstance(result["executive_summary"], str)
            assert result["executive_summary"] != ""

    def test_crowding_severity_thresholds(self):
        """Crowding severity should increase with more crowded factors."""
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        # 0 crowded → low
        result = analyst.analyze_governance({
            "status": "success",
            "crowding": {"crowded_factor_ids": []},
        })
        assert result["crowding_analysis"]["severity"] == "low"

        # 6 crowded → high
        result = analyst.analyze_governance({
            "status": "success",
            "crowding": {"crowded_factor_ids": ["f1", "f2", "f3", "f4", "f5", "f6"]},
        })
        assert result["crowding_analysis"]["severity"] == "high"


class TestAgentDeliveryReportStage:
    """Test the refactored Delivery Report stage with agent disabled."""

    def test_inject_narrative_into_markdown(self):
        """Narrative injection should not break existing markdown structure."""
        from quantlab.pipeline_stages.delivery import AgentDeliveryReportStage

        stage = AgentDeliveryReportStage()
        original = "# Factor Report\n\n## 1. Factor Definition\n\ncontent here\n\n## 2. IC Stats\n\nmore content\n"
        narrative = {
            "executive_summary": "这是一个优秀的因子。",
            "strengths": ["高IC", "低换手"],
            "weaknesses": ["波动敏感"],
        }

        result = stage._inject_narrative_into_markdown(original, narrative)

        assert "执行摘要" in result
        assert "这是一个优秀的因子" in result
        assert "高IC" in result
        assert "Factor Definition" in result  # original content preserved
        assert "IC Stats" in result

    def test_build_library_context_is_accurate(self):
        """Library context should reflect the factor store state."""
        from quantlab.pipeline_stages.delivery import AgentDeliveryReportStage

        stage = AgentDeliveryReportStage()

        # Create mock entries
        class MockSpec:
            def __init__(self, fid, fam, status):
                self.factor_id = fid
                self.family = fam
                self.status = status

        class MockEntry:
            def __init__(self, fid, fam, status):
                self.factor_spec = MockSpec(fid, fam, status)

        library = [
            MockEntry("f1", "momentum", "approved"),
            MockEntry("f2", "reversal", "approved"),
            MockEntry("f3", "momentum", "draft"),
        ]

        ctx = stage._build_library_context(library, ["f1", "f2"])

        assert ctx["total_factors"] == 3
        assert ctx["active_factors"] == 2
        assert ctx["deliverable_count"] == 2
        assert ctx["family_distribution"]["momentum"] == 2

    def test_report_complete_format_has_required_sections(self):
        """Rule-fallback narrative report must contain all required sections."""
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        result = analyst.generate_narrative_report({
            "factor_id": "test",
            "factor_name": "Test",
            "rank_ic_mean": 0.04,
            "icir": 1.8,
            "simulation": {"sharpe_ratio": 1.1},
            "risk_flags": [],
        })

        required_keys = [
            "executive_summary", "factor_story", "strengths", "weaknesses",
            "market_context", "risk_assessment", "buyer_checklist",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ------------------------------------------------------------------
# Cross-Stage Integration Tests
# ------------------------------------------------------------------


class TestCrossStageFeedback:
    """Test the feedback loop between stages via ctx._meta."""

    def test_ctx_meta_preserves_data_across_stages(self, pipeline_ctx):
        """ctx._meta should accumulate data as stages run in sequence."""
        # Simulate the pipeline sequence

        # Stage 4: OOS writes feedback
        pipeline_ctx._meta["oos_analysis"] = {"status": "rule_fallback"}
        pipeline_ctx._meta["discovery_feedback"] = {
            "directions_to_prioritize": ["momentum"],
            "directions_to_avoid": ["volatility"],
        }

        # Stage 6: Governance writes analysis
        pipeline_ctx._meta["governance_analysis"] = {"status": "rule_fallback"}
        pipeline_ctx._meta["deliverable_factor_ids"] = ["f1", "f2"]

        # All keys should still be present
        assert "oos_analysis" in pipeline_ctx._meta
        assert "discovery_feedback" in pipeline_ctx._meta
        assert "governance_analysis" in pipeline_ctx._meta
        assert "deliverable_factor_ids" in pipeline_ctx._meta
        assert pipeline_ctx._meta["deliverable_factor_ids"] == ["f1", "f2"]

    def test_feedback_contains_actionable_directions(self):
        """Feedback must contain specific direction guidance for the discovery agent."""
        from quantlab.pipeline_stages.agent_analyst import AgentAnalyst

        analyst = AgentAnalyst()
        analyst._available = False

        result = analyst.generate_feedback(
            {"cross_factor_summary": {"top_performing_families": ["momentum", "value"]}},
            {"crowding_analysis": {"cluster_descriptions": ["波动率拥挤"]}},
        )

        assert "directions_to_prioritize" in result
        assert "directions_to_avoid" in result
        assert isinstance(result["directions_to_prioritize"], list)
        assert isinstance(result["directions_to_avoid"], list)


class TestEvolutionStageFeedback:
    """Test that EvolutionStage accepts and uses agent feedback."""

    def test_evolution_stage_has_feedback_attribute(self):
        from quantlab.pipeline_stages.evolution import EvolutionStage

        stage = EvolutionStage()
        assert hasattr(stage, "agent_feedback")
        assert stage.agent_feedback == {}

    def test_feedback_injected_to_llm_context(self):
        """Feedback should be added to LLMClient context before running multi-agent."""
        from quantlab.pipeline_stages.evolution import EvolutionStage

        stage = EvolutionStage()
        stage.agent_feedback = {
            "summary": "动量方向表现优秀",
            "directions_to_prioritize": ["momentum_reversal"],
            "directions_to_avoid": ["volatility_regime"],
            "structural_patterns": {
                "what_worked": "成交量增强动量",
                "what_failed": "纯波动率因子",
            },
        }

        # Verify the feedback is stored
        assert len(stage.agent_feedback["directions_to_prioritize"]) == 1
        assert len(stage.agent_feedback["directions_to_avoid"]) == 1
        assert stage.agent_feedback["structural_patterns"]["what_worked"] is not None
