import numpy as np
import pytest

from klens.spectral import (
    cluster_factions,
    compute_koopman_metrics,
    detect_rogue_agent,
    extract_centroid_keywords,
    lift_state,
)


def test_lift_state_dimensions():
    n_samples = 4
    d = 10
    num_rff = 50
    delta = np.random.randn(n_samples, d)
    W_f = np.random.randn(d, num_rff)
    b_f = np.random.uniform(0, 2 * np.pi, size=(num_rff,))

    lifted = lift_state(delta, W_f, b_f)
    # Expected columns: 1 (bias) + d (linear state) + num_rff (Fourier features)
    assert lifted.shape == (n_samples, 1 + d + num_rff)
    assert np.allclose(lifted[:, 0], 1.0)


def test_compute_koopman_metrics_insufficient_history():
    # Only 1 round
    Z = np.random.randn(1, 4, 10)
    l2, t_cert, delta, z_mean = compute_koopman_metrics(Z)
    assert l2 == 0.99
    assert t_cert == 999


def test_compute_koopman_metrics_insufficient_agents():
    # Only 1 agent
    Z = np.random.randn(5, 1, 10)
    l2, t_cert, delta, z_mean = compute_koopman_metrics(Z)
    assert l2 == 0.99
    assert t_cert == 999


def test_compute_koopman_metrics_contractive_system():
    # Synthetic contractive dynamics where disagreements decay exponentially
    T, N, d = 5, 4, 8
    Z = np.zeros((T, N, d))
    base_centroid = np.ones(d)
    for t in range(T):
        decay = 0.5 ** t
        for i in range(N):
            Z[t, i, :] = base_centroid + decay * (np.arange(d) * (i + 1) * 0.1)

    l2, t_cert, delta, z_mean = compute_koopman_metrics(Z)
    assert 0.0 <= l2 < 1.0
    assert 1 <= t_cert < 100
    assert delta.shape == (T, N * d)
    assert z_mean.shape == (T, d)


def test_detect_rogue_agent():
    T, N, d = 4, 3, 2
    speakers = ["Agent_A", "Agent_B", "Agent_C"]
    delta = np.zeros((T, N, d))
    # Agent_C has large disagreement, Agent_A and Agent_B have small disagreement
    delta[:, 0, :] = 0.1
    delta[:, 1, :] = 0.1
    delta[:, 2, :] = 5.0

    rogue_speaker, rogue_idx = detect_rogue_agent(delta.reshape(T, N * d), speakers, T, N, d)
    assert rogue_speaker == "Agent_C"
    assert rogue_idx == 2


def test_cluster_factions():
    speakers = ["Agent_A", "Agent_B", "Agent_C", "Agent_D"]
    # Suppose Agent_D is rogue (rogue_idx=3)
    # Agent_A and Agent_B are nearly identical
    # Agent_C is distinct
    final_states = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    factions = cluster_factions(final_states, speakers, rogue_idx=3, similarity_threshold=0.65)
    # Should produce 2 factions: F1:(Agent_A,Agent_B) and F2:(Agent_C)
    assert len(factions) == 2
    assert set(factions[0].members) == {"Agent_A", "Agent_B"}
    assert set(factions[1].members) == {"Agent_C"}


def test_extract_centroid_keywords():
    centroid = np.array([0.0, 0.8, 0.1, 0.9, 0.0])
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
    top_keywords = extract_centroid_keywords(centroid, vocab, top_k=3)
    assert top_keywords == ["delta", "beta", "gamma"]
