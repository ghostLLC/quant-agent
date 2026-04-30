"""Tests for DailyScheduler and pipeline stages."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from quantlab.scheduler import DailyScheduler, DailyRunRecord
from quantlab.pipeline_stages import PipelineContext


def make_market_df(n_dates=200, n_assets=50):
    """Synthetic cross-section market DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}" for i in range(1, n_assets + 1)]
    rows = []
    for d in dates:
        for a in assets:
            rows.append({
                "date": d,
                "asset": a,
                "close": np.random.uniform(10, 100),
                "volume": np.random.uniform(1e5, 1e7),
                "industry": "test",
            })
    return pd.DataFrame(rows)


class TestDailySchedulerInit:
    def test_init_defaults(self):
        scheduler = DailyScheduler()
        assert scheduler.ctx.evolution_rounds == 3
        assert scheduler.ctx.use_adaptive_directions is True
        assert scheduler.ctx.use_multi_agent is True
        assert len(scheduler.ctx.directions) == 5
        assert "momentum_reversal" in scheduler.ctx.directions
        assert scheduler.ctx.max_candidates_per_round == 5

    def test_init_custom(self):
        scheduler = DailyScheduler(
            directions=["a"],
            evolution_rounds=5,
            use_adaptive_directions=False,
        )
        assert scheduler.ctx.directions == ["a"]
        assert scheduler.ctx.evolution_rounds == 5
        assert scheduler.ctx.use_adaptive_directions is False
        assert scheduler.ctx.use_multi_agent is True


class TestLoadData:
    def test_load_data_empty(self, tmp_path):
        """ctx.load_data returns empty DataFrame when file doesn't exist."""
        ctx = PipelineContext(data_path=tmp_path / "nonexistent.csv")
        with patch("quantlab.factor_discovery.datahub.DataHub", side_effect=Exception("no datahub")):
            result = ctx.load_data(apply_survivorship=False, check_quality=False)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_data_with_file(self, tmp_path):
        """ctx.load_data loads CSV correctly via fallback path."""
        csv_path = tmp_path / "test_data.csv"
        df_expected = make_market_df(10, 5)
        df_expected.to_csv(csv_path, index=False)

        ctx = PipelineContext(data_path=csv_path)
        with patch("quantlab.factor_discovery.datahub.DataHub", side_effect=Exception("no datahub")):
            result = ctx.load_data(apply_survivorship=False, check_quality=False)

        assert not result.empty
        assert "date" in result.columns
        assert "asset" in result.columns
        assert "close" in result.columns


class TestRefreshData:
    def test_refresh_data_fallback(self):
        """DataRefreshStage returns skipped when AkShare and Tushare both fail."""
        from quantlab.pipeline_stages import DataRefreshStage
        ctx = PipelineContext()
        stage = DataRefreshStage()
        with patch(
            "quantlab.data.tushare_provider.AkShareIncrementalProvider",
            side_effect=Exception("API down"),
        ):
            result = stage.run(ctx)
        assert result["status"] == "skipped"


class TestCombineFactors:
    def test_combine_factors_empty(self):
        """CombinationStage returns skipped when store is empty."""
        from quantlab.pipeline_stages import CombinationStage
        ctx = PipelineContext()
        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = []

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ):
            result = CombinationStage().run(ctx)

        assert result["status"] == "skipped"
        assert "无已审批因子" in result["reason"]

    def test_combine_factors_no_approved(self):
        """CombinationStage returns skipped when no approved factors."""
        from quantlab.pipeline_stages import CombinationStage
        ctx = PipelineContext()
        mock_entry = MagicMock()
        mock_entry.factor_spec.status = "draft"

        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = [mock_entry]

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ):
            result = CombinationStage().run(ctx)

        assert result["status"] == "skipped"
        assert "无已审批因子" in result["reason"]


class TestValidateOOS:
    def test_validate_oos_insufficient(self):
        """OOSValidationStage returns skipped when data has insufficient days."""
        from quantlab.pipeline_stages import OOSValidationStage
        ctx = PipelineContext()
        df = make_market_df(n_dates=50, n_assets=10)

        with patch.object(ctx, "load_data", return_value=df):
            result = OOSValidationStage().run(ctx)
        assert result["status"] == "skipped"


class TestRunGovernance:
    def test_run_governance_empty_store(self):
        """GovernanceStage handles empty PersistentFactorStore gracefully."""
        from quantlab.pipeline_stages import GovernanceStage
        ctx = PipelineContext()

        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = []
        mock_store.get_library_stats.return_value = {"total": 0}
        mock_store.archive_underperforming.return_value = {"crowding_scores": {}}

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ), patch(
            "quantlab.factor_discovery.models.FactorLifecycleManager",
            return_value=MagicMock(),
        ), patch(
            "quantlab.factor_discovery.factor_enhancements.RegimeDetector",
            side_effect=ImportError,
        ), patch(
            "quantlab.factor_discovery.factor_enhancements.CrowdingDetector",
            side_effect=ImportError,
        ), patch(
            "quantlab.factor_discovery.factor_enhancements.FactorCurveAnalyzer",
            side_effect=ImportError,
        ), patch(
            "quantlab.trading.risk_control.RiskManager",
            side_effect=ImportError,
        ), patch.object(
            ctx, "load_data", return_value=pd.DataFrame(),
        ):
            result = GovernanceStage().run(ctx)

        assert result["status"] == "success"
        assert "stats_before" in result
        assert "stats_after" in result


class TestDailyRunRecord:
    def test_daily_run_record_to_dict(self):
        """DailyRunRecord.to_dict() produces correct keys."""
        record = DailyRunRecord(
            run_id="daily_test",
            run_date="2025-01-01",
            start_time="2025-01-01T00:00:00",
        )
        d = record.to_dict()

        assert d["run_id"] == "daily_test"
        assert d["run_date"] == "2025-01-01"
        assert d["status"] == "running"
        for key in [
            "data_refresh", "decay_monitor", "evolution", "screening",
            "oos_validation", "combination", "governance", "paper_trading",
            "delivery_reports", "error_message", "end_time",
        ]:
            assert key in d, f"Missing key: {key}"


class TestEvolution:
    def test_cold_start_bootstrap_called(self):
        """EvolutionStage triggers bootstrap when store is empty."""
        from quantlab.pipeline_stages import EvolutionStage
        ctx = PipelineContext()
        df = make_market_df(200, 10)

        mock_store = MagicMock()
        mock_store.load_library_entries.return_value = []

        with patch(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            return_value=mock_store,
        ), patch(
            "quantlab.factor_discovery.seed_factors.bootstrap_seed_factors",
        ) as mock_bootstrap, patch(
            "quantlab.factor_discovery.evolution.FactorEvolutionLoop",
        ) as mock_loop_class, patch.object(
            ctx, "load_data", return_value=df,
        ):
            mock_bootstrap.return_value = {"injected_count": 3, "factor_ids": [], "ic_results": {}}
            mock_loop = MagicMock()
            mock_loop.run.return_value = {
                "approved_count": 1,
                "total_candidates": 2,
                "best_score": 0.03,
            }
            mock_loop_class.return_value = mock_loop

            EvolutionStage().run(ctx)
            mock_bootstrap.assert_called_once()


class TestBenchmark:
    def test_benchmark_compare(self):
        """_benchmark_compare computes ew_ic, mcw_ic, excess keys correctly."""
        from quantlab.pipeline_stages.combination import _benchmark_compare
        ctx = PipelineContext()
        df = make_market_df(200, 50)

        combination = {
            "combined_ic": 0.05,
            "combined_icir": 0.8,
            "combined_rank_ic": 0.04,
            "factor_ids": ["f1"],
            "weights": {"f1": 1.0},
            "method": "ic_weighted",
        }
        with patch.object(ctx, "load_data", return_value=df):
            result = _benchmark_compare(ctx, combination)

        assert "ew_ic" in result
        assert "mcw_ic" in result
        assert "factor_ic" in result
        assert "excess_vs_ew" in result
        assert "excess_vs_mcw" in result
        assert result["factor_ic"] == 0.05
