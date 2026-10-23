import numpy as np
import pytest

from klens.spectral import predict_agent_trajectories
from klens.stream import LiveTranscriptStreamProcessor


def test_predict_agent_trajectories_basic():
    T, N, d = 4, 3, 4
    Z = np.zeros((T, N, d))
    speakers = ["Agent_A", "Agent_B", "Agent_C"]

    # Agent_C is contrarian with large disagreement, A and B are close
    for t in range(T):
        Z[t, 0, :] = [1.0, 0.0, 0.0, 0.0]
        Z[t, 1, :] = [0.9, 0.1, 0.0, 0.0]
        Z[t, 2, :] = [0.0, 0.0, 5.0, 2.0]

    pred_rogue, contraction, norms = predict_agent_trajectories(Z, speakers, steps_ahead=1)
    assert pred_rogue == "Agent_C"
    assert "Agent_C" in norms
    assert norms["Agent_C"] > norms["Agent_A"]
    assert norms["Agent_C"] > norms["Agent_B"]
    assert contraction > 0.0


def test_stream_processor_attaches_prediction():
    proc = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_pred_stream.csv",
        auto_init_csv=False,
    )

    proc.ingest_turn(1, "A1", "Autonomous judge model cost.")
    proc.ingest_turn(1, "A2", "Human auditing verification.")
    m1 = proc.process_round_event(1)
    assert m1 is None

    proc.ingest_turn(2, "A1", "Judge model auditing compromise.")
    proc.ingest_turn(2, "A2", "Strict opposition no compromise.")
    m2 = proc.process_round_event(2)

    assert m2 is not None
    assert m2.predicted_next_rogue in ["A1", "A2"]
    assert m2.predicted_contraction_ratio is not None
    assert m2.predicted_contraction_ratio > 0.0
