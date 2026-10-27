import os
import tempfile
import pytest

from klens.stream import LiveTranscriptStreamProcessor
from klens.models import SpectralMetrics


def test_stream_processor_basic_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_log.csv")
        processor = LiveTranscriptStreamProcessor(csv_log_path=csv_path)

        # Ingest round 1
        processor.ingest_turn(1, "A1", "Autonomous AI auditing and judge models reduce operational cost.")
        processor.ingest_turn(1, "A2", "Autonomous systems without human oversight pose security risks.")
        # Round 1 alone should return None because >= 2 rounds required
        m1 = processor.process_round_event(1)
        assert m1 is None

        # Ingest round 2
        processor.ingest_turn(2, "A1", "Judge models with small SLM verifiers satisfy cost metrics.")
        processor.ingest_turn(2, "A2", "Selective auditing with human escalations provides safety.")
        m2 = processor.process_round_event(2)
        assert m2 is not None
        assert isinstance(m2, SpectralMetrics)
        assert m2.round_id == 2
        assert m2.active_agents_count == 2
        assert 0.0 <= m2.lambda_2_magnitude <= 1.0
        assert m2.rogue_agent_detected in ["A1", "A2"]
        assert m2.orthogonal_agent_detected in ["A1", "A2"]
        assert m2.orthogonal_agent_detected == m2.rogue_agent_detected
        assert m2.topological_state in ["Convergence", "Factionalization", "Dissent", "Migration", "Dissolution"]
        assert len(m2.agent_cluster_assignments) == 2
        assert len(m2.agent_local_residuals) == 2

        # Compatibility checks (indexable and iterable like the reference tuple/list)
        assert len(m2) == 8
        assert m2[0] == 2
        assert m2[1] == 2

        # CSV output exists and has headers + 1 row
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0].startswith("round,active_agents_count,lambda_2_magnitude")


def test_sliding_window_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "sliding_log.csv")
        processor = LiveTranscriptStreamProcessor(
            csv_log_path=csv_path,
            sliding_window_rounds=2,
        )

        for r in range(1, 6):
            processor.ingest_turn(r, "A1", f"Statement on cost optimization round {r}")
            processor.ingest_turn(r, "A2", f"Statement on safety and oversight round {r}")
            m = processor.process_round_event(r)
            if r >= 2:
                assert m is not None
                assert m.round_id == r


def test_callbacks_fired():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "callback_log.csv")
        processor = LiveTranscriptStreamProcessor(csv_log_path=csv_path)

        round_events = []
        processor.add_round_callback(lambda m: round_events.append(m.round_id))

        processor.ingest_turn(1, "A1", "Safety first.")
        processor.ingest_turn(1, "A2", "Speed first.")
        processor.process_round_event(1)

        processor.ingest_turn(2, "A1", "Safety and speed compromise.")
        processor.ingest_turn(2, "A2", "Speed with safety guards.")
        processor.process_round_event(2)

        assert round_events == [2]
