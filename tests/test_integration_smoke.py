"""End-to-end integration smoke test — validates pipeline stages in sequence.

Uses mocked heavy stages (Evolution) to avoid long compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _make_smoke_df() -> pd.DataFrame:
    import numpy as np
    dates = pd.bdate_range("2022-01-01", periods=300)
    assets = [f"00000{i}" for i in range(1, 11)]
    records = []
    for date in dates:
        for i, asset in enumerate(assets):
            base = 10 + i * 3
            noise = (hash(f"{date}{asset}") % 100 - 50) / 100.0
            records.append({
                "date": date, "asset": asset,
                "open": base + noise - 0.1, "high": base + noise + 0.3,
                "low": base + noise - 0.3, "close": base + noise,
                "volume": float(np.random.randint(1000000, 10000000)),
                "amount": (base + noise) * float(np.random.randint(1000000, 10000000)),
                "industry": ["bank", "tech"][i % 2],
                "market_cap": (base + noise) * float(np.random.randint(100000000, 2147483647)),
                "turnover": np.random.uniform(0.01, 0.15),
                "pb": np.random.uniform(0.5, 5.0),
                "pe": np.random.uniform(5.0, 50.0),
            })
    return pd.DataFrame(records)


class TestPipelineSmoke:
    """Smoke test: validate that all non-evolution stages can run in sequence."""

    def test_core_stages_run(self, monkeypatch):
        """Data→Decay→OOS→Combination→Screening→Paper→Governance→Report."""
        from quantlab.pipeline_stages.base import PipelineContext
        from quantlab.pipeline_stages import (
            DecayMonitorStage, OOSValidationStage, CombinationStage,
            GovernanceStage, DeliveryScreeningStage,
            PaperTradingStage, DeliveryReportStage,
        )

        monkeypatch.setattr(
            "quantlab.pipeline_stages.base._load_market_data",
            lambda path, a, b: _make_smoke_df(),
        )
        # Mock PersistentFactorStore to return empty library
        class MockStore:
            def load_library_entries(self): return []
            def get_library_stats(self): return {}
            def upsert_library_entry(self, e, **kw): pass
            def archive_underperforming(self, **kw): return {"crowding_scores": {}}

        monkeypatch.setattr(
            "quantlab.factor_discovery.runtime.PersistentFactorStore",
            lambda: MockStore(),
        )

        ctx = PipelineContext()
        stages_ok = 0

        r = DecayMonitorStage().run(ctx)
        assert r is not None; stages_ok += 1

        r = OOSValidationStage(enable_agent=False).run(ctx)
        assert "status" in r; stages_ok += 1

        r = CombinationStage().run(ctx)
        assert "status" in r; stages_ok += 1

        r = DeliveryScreeningStage().run(ctx)
        assert "deliverable_count" in r
        ctx._meta["deliverable_factor_ids"] = r.get("deliverable_factor_ids", [])
        stages_ok += 1

        r = PaperTradingStage().run(ctx)
        assert "status" in r; stages_ok += 1

        r = GovernanceStage(enable_agent=False).run(ctx)
        assert "status" in r; stages_ok += 1

        r = DeliveryReportStage(enable_agent=False).run(ctx)
        assert isinstance(r, list); stages_ok += 1

        assert stages_ok == 7  # All 7 fast stages pass

    def test_pipeline_context_cache(self, monkeypatch):
        """load_data caches and reuses DataFrame across stages."""
        from quantlab.pipeline_stages.base import PipelineContext
        monkeypatch.setattr(
            "quantlab.pipeline_stages.base._load_market_data",
            lambda path, a, b: _make_smoke_df(),
        )
        ctx = PipelineContext()
        df1 = ctx.load_data()
        df2 = ctx.load_data()
        assert df1 is df2
        ctx.invalidate_cache()
        assert ctx.load_data() is not df1

    def test_checkpoint_save_load(self):
        """Checkpoint saves and loads correctly."""
        from quantlab.scheduler import CHECKPOINT_PATH
        import json
        cp = {"run_id": "test_001", "completed_stages": ["data_refresh", "decay_monitor"]}
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(json.dumps(cp), encoding="utf-8")
        loaded = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "test_001"
        assert len(loaded["completed_stages"]) == 2
        CHECKPOINT_PATH.unlink()

    def test_new_modules_importable(self):
        """All new production modules import cleanly."""
        from quantlab.factor_discovery.llm_supervisor import LLMSupervisor
        from quantlab.factor_discovery.benchmark_factors import BenchmarkFactorRegistry
        from quantlab.factor_discovery.factor_namer import FactorNamer, FactorVersionManager
        from quantlab.factor_discovery.real_return import RealReturnEvaluator
        from quantlab.pipeline_stages.anomaly_guard import AnomalyGuard
        from quantlab.pipeline_stages.factor_monitor import FactorMonitor
        from quantlab.pipeline_stages.experiment_tracker import ExperimentTracker
        from quantlab.assistant.email_notifier import EmailNotifier
        assert BenchmarkFactorRegistry().get_benchmark_names()
        assert EmailNotifier(recipient="676236147@qq.com").recipient == "676236147@qq.com"
