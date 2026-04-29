"""Tests for MessageBus, AgentMessage, LLMClient context, and decay monitor helpers."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from quantlab.factor_discovery.multi_agent import (
    MessageBus, AgentMessage, AgentRole, LLMClient,
)
from quantlab.factor_discovery.decay_monitor import (
    record_daily_ic, get_ic_trend, check_decay_trend, ic_history_path,
)


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def temp_ic_csv(tmp_path, monkeypatch):
    p = tmp_path / "ic_history.csv"
    monkeypatch.setattr("quantlab.factor_discovery.decay_monitor.ic_history_path", p)
    return p


# ── MessageBus ──

class TestMessageBus:
    def test_send_receive(self, bus):
        msg1 = AgentMessage(
            msg_id="m1", sender=AgentRole.R1_HYPOTHESIS_GENERATOR,
            recipient=AgentRole.R2_HYPOTHESIS_REVIEWER,
            msg_type="hypothesis", payload={"data": 1},
        )
        msg2 = AgentMessage(
            msg_id="m2", sender=AgentRole.R1_HYPOTHESIS_GENERATOR,
            recipient=AgentRole.P1_ARCHITECT,
            msg_type="plan", payload={"data": 2},
        )
        bus.send(msg1)
        bus.send(msg2)

        r2_msgs = bus.receive(AgentRole.R2_HYPOTHESIS_REVIEWER)
        assert len(r2_msgs) == 1
        assert r2_msgs[0].msg_id == "m1"

        p1_msgs = bus.receive(AgentRole.P1_ARCHITECT)
        assert len(p1_msgs) == 1
        assert p1_msgs[0].msg_id == "m2"

    def test_broadcast(self, bus):
        # send broadcast → R2 receives it
        bus.send(AgentMessage(
            msg_id="m3", sender=AgentRole.R1_HYPOTHESIS_GENERATOR,
            recipient="broadcast",
            msg_type="announce", payload={"text": "hello"},
        ))
        r2_msgs = bus.receive(AgentRole.R2_HYPOTHESIS_REVIEWER)
        assert len(r2_msgs) == 1
        assert r2_msgs[0].msg_id == "m3"

        # send another broadcast → P1 receives it
        bus.send(AgentMessage(
            msg_id="m4", sender=AgentRole.R1_HYPOTHESIS_GENERATOR,
            recipient="broadcast",
            msg_type="announce", payload={"text": "world"},
        ))
        p1_msgs = bus.receive(AgentRole.P1_ARCHITECT)
        assert len(p1_msgs) == 1
        assert p1_msgs[0].msg_id == "m4"

        # history_for confirms both recipients saw broadcasts
        r2_history = bus.history_for(AgentRole.R2_HYPOTHESIS_REVIEWER)
        assert len(r2_history) == 2
        assert any(m["msg_id"] == "m3" for m in r2_history)
        p1_history = bus.history_for(AgentRole.P1_ARCHITECT)
        assert len(p1_history) >= 1
        assert any(m["msg_id"] == "m4" for m in p1_history)

    def test_history(self, bus):
        for i in range(5):
            bus.send(AgentMessage(
                msg_id=f"m{i}", sender=AgentRole.R1_HYPOTHESIS_GENERATOR,
                recipient=AgentRole.R2_HYPOTHESIS_REVIEWER,
                msg_type="hypothesis", payload={"i": i},
            ))
        # Also send one to a different recipient
        bus.send(AgentMessage(
            msg_id="m_other", sender=AgentRole.T1_BACKTESTER,
            recipient=AgentRole.T2_VALIDATOR,
            msg_type="result", payload={},
        ))

        r2_history = bus.history_for(AgentRole.R2_HYPOTHESIS_REVIEWER)
        assert len(r2_history) == 5

        t1_history = bus.history_for(AgentRole.T1_BACKTESTER)
        assert len(t1_history) == 1


# ── AgentMessage serialization ──

class TestAgentMessageSerialization:
    def test_to_dict_roundtrip(self):
        msg = AgentMessage(
            msg_id="test_1",
            sender=AgentRole.R2_HYPOTHESIS_REVIEWER,
            recipient=AgentRole.P1_ARCHITECT,
            msg_type="approved_hypothesis",
            payload={"score": 0.85, "notes": "good factor"},
            thread_id="thread_abc",
        )
        d = msg.to_dict()
        assert d["msg_id"] == "test_1"
        assert d["sender"] == "r2_hypothesis_reviewer"
        assert d["recipient"] == "p1_architect"
        assert d["msg_type"] == "approved_hypothesis"
        assert d["payload"] == {"score": 0.85, "notes": "good factor"}
        assert d["thread_id"] == "thread_abc"
        assert isinstance(d["timestamp"], float)


# ── LLMClient context management ──

class TestLLMClientContext:
    @pytest.fixture
    def client(self):
        return LLMClient()

    def test_add_context(self, client):
        client.add_context("R2", "rejected due to PB exposure")
        assert len(client.context) == 1
        assert client.context[0]["role"] == "R2"
        assert client.context[0]["content"] == "rejected due to PB exposure"

    def test_context_summary(self, client):
        client.add_context("R1", "generated momentum factor")
        client.add_context("R2", "approved with minor revisions")
        client.add_context("T2", "useful factor verified")
        summary = client.get_context_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "momentum" in summary

    def test_context_empty(self, client):
        summary = client.get_context_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "暂无" in summary or "历史" in summary

    def test_context_overflow(self, client):
        for i in range(25):
            client.add_context(f"agent_{i}", f"message number {i}")
        assert len(client.context) == 20
        # Oldest entries should be dropped; first remaining should be index 5
        assert client.context[0]["content"] == "message number 5"
        assert client.context[-1]["content"] == "message number 24"


# ── Decay monitor helpers ──

class TestDecayMonitorHelpers:
    def test_record_daily_ic(self, temp_ic_csv):
        record_daily_ic("factor_A", "2024-06-15", 0.032)
        assert temp_ic_csv.exists()
        df = pd.read_csv(temp_ic_csv)
        assert list(df.columns) == ["factor_id", "date", "rank_ic"]
        assert len(df) == 1
        assert df.iloc[0]["factor_id"] == "factor_A"
        assert df.iloc[0]["rank_ic"] == 0.032

    def test_get_ic_trend_insufficient(self, temp_ic_csv):
        # Write only 5 data points (< 20 min_days)
        rows = []
        for i in range(5):
            rows.append({"factor_id": "factor_B", "date": f"2024-01-{i+1:02d}", "rank_ic": 0.03})
        pd.DataFrame(rows).to_csv(temp_ic_csv, index=False)

        result = get_ic_trend("factor_B", min_days=20)
        assert result["n_days"] == 5
        assert "insufficient" in result.get("error", "")

    def test_check_decay_trend_decaying(self, temp_ic_csv):
        # Synthetic declining IC: 30 days, IC drops from ~0.05 to ~-0.01
        np.random.seed(42)
        n = 30
        dates = [f"2024-01-{i+1:02d}" for i in range(n)]
        t = np.arange(n)
        rank_ic = 0.05 - 0.0025 * t + np.random.normal(0, 0.003, n)
        rows = []
        for i in range(n):
            rows.append({"factor_id": "factor_C", "date": dates[i], "rank_ic": rank_ic[i]})
        pd.DataFrame(rows).to_csv(temp_ic_csv, index=False)

        result = check_decay_trend("factor_C", min_days=20)
        assert result == "decaying"

    def test_check_decay_trend_stable(self, temp_ic_csv):
        # Synthetic flat IC: 30 days, IC hovers around 0.03
        np.random.seed(42)
        n = 30
        dates = [f"2024-01-{i+1:02d}" for i in range(n)]
        rank_ic = 0.03 + np.random.normal(0, 0.005, n)
        rows = []
        for i in range(n):
            rows.append({"factor_id": "factor_D", "date": dates[i], "rank_ic": rank_ic[i]})
        pd.DataFrame(rows).to_csv(temp_ic_csv, index=False)

        result = check_decay_trend("factor_D", min_days=20)
        assert result == "stable"
