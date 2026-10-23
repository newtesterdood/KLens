import numpy as np
import pytest

from klens.models import CompressionConfig
from klens.spectral import (
    compress_to_faction_centroids,
    compute_koopman_metrics,
    detect_rogue_from_states,
)
from klens.stream import LiveTranscriptStreamProcessor


def test_primal_vs_dual_solver_equivalence():
    np.random.seed(42)
    T, N, d = 4, 3, 6
    Z = np.random.randn(T, N, d)

    l2_primal, t_primal, _, _ = compute_koopman_metrics(Z, solver="primal", seed=42)
    l2_dual, t_dual, _, _ = compute_koopman_metrics(Z, solver="dual", seed=42)

    assert abs(l2_primal - l2_dual) < 1e-4
    assert t_primal == t_dual


def test_compress_to_faction_centroids():
    T, N, d = 3, 30, 4
    speakers = [f"Agent_{i}" for i in range(N)]
    np.random.seed(42)

    # 3 natural clusters of 10 agents each
    Z = np.zeros((T, N, d))
    cluster_centers = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    for i in range(N):
        c_idx = i % 3
        noise = np.random.normal(0, 0.05, size=(T, d))
        Z[:, i, :] = cluster_centers[c_idx] + noise

    Z_macro, factions = compress_to_faction_centroids(Z, speakers, max_factions=3, seed=42)

    assert Z_macro.shape == (T, 3, d)
    assert len(factions) == 3
    # All 30 agents must be partitioned across the 3 factions
    total_members = sum(len(f.members) for f in factions)
    assert total_members == 30


def test_stream_processor_with_compression_toggle():
    # Test toggled OFF (default full compute)
    proc_full = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_full.csv",
        compression_config=CompressionConfig(enabled=False),
        auto_init_csv=False,
    )

    # Test toggled ON (compressed macro mode)
    proc_comp = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_comp.csv",
        compression_config=CompressionConfig(enabled=True, max_factions=2),
        auto_init_csv=False,
    )

    distinct_texts = [
        "Auditing models is the only safe procedure.",
        "Cost reduction must take absolute precedence.",
        "Human verification is required for all actions.",
        "Autonomous agents should execute without delay.",
    ]

    for r in [1, 2]:
        for i in range(4):
            txt = distinct_texts[i] + f" Round {r}"
            proc_full.ingest_turn(r, f"Agent_{i}", txt)
            proc_comp.ingest_turn(r, f"Agent_{i}", txt)

    m_full = proc_full.process_round_event(2)
    m_comp = proc_comp.process_round_event(2)

    assert m_full is not None
    assert m_comp is not None

    assert m_full.is_compressed is False
    assert m_comp.is_compressed is True
    assert m_comp.faction_count == 2


def test_large_agent_population_scaling_and_rogue_detection():
    # 60 agents partitioned into 4 distinct perspectives, with Agent_42 being an extreme rogue
    N = 60
    speakers = [f"Agent_{i}" for i in range(N)]
    proc = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_scale_60.csv",
        compression_config=CompressionConfig(enabled=True, max_factions=4),
        auto_init_csv=False,
    )

    perspectives = [
        "Strict regulatory auditing compliance.",
        "Aggressive latency optimization throughput.",
        "Minimal financial cost budget constraint.",
        "Balanced multi-layered consensus security.",
    ]

    for r in [1, 2]:
        for i, speaker in enumerate(speakers):
            if speaker == "Agent_42":
                text = "Completely discordant contrarian opposition to all points."
            else:
                p_text = perspectives[i % 4]
                text = f"{p_text} Round {r}"
            proc.ingest_turn(r, speaker, text)

    metrics = proc.process_round_event(2)
    assert metrics is not None
    assert metrics.active_agents_count == 60
    assert metrics.is_compressed is True
    # The rogue agent must be accurately identified as Agent_42 despite compression!
    assert metrics.rogue_agent_detected == "Agent_42"
    assert metrics.faction_count == 4
