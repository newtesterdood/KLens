import numpy as np
import pytest

from klens.spectral import (
    classify_topological_state,
    cluster_factions,
    compute_koopman_metrics,
    compute_multi_cluster_residuals,
    detect_cluster_migrations,
    detect_orthogonal_agent,
    detect_orthogonal_from_states,
    detect_rogue_agent,
    detect_rogue_from_states,
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


def test_multi_cluster_residuals_resolves_phantom_center():
    # 4-4-2 distribution:
    # 4 agents in Faction A at [1.0, 0.0]
    # 4 agents in Faction B at [-1.0, 0.0]
    # 2 agents at the origin [0.0, 0.0] (the global mean)
    N = 10
    d = 2
    T = 2
    Z = np.zeros((T, N, d))
    for t in range(T):
        Z[t, 0:4, 0] = 1.0   # Faction A
        Z[t, 4:8, 0] = -1.0  # Faction B
        Z[t, 8:10, 0] = 0.0  # Origin agents (Phantom center)

    # Compute global mean vs multi-cluster local residual
    global_mean = Z.mean(axis=1) # [0.0, 0.0]
    global_residuals = np.linalg.norm(Z[0] - global_mean[0], axis=1)
    # Under global centroid, origin agents falsely have 0 residual, while cohesive faction members have 1.0!
    assert np.allclose(global_residuals[8:10], 0.0)
    assert np.allclose(global_residuals[0:4], 1.0)

    # Now verify multi-cluster local residuals
    local_residuals, assignments, centroids = compute_multi_cluster_residuals(Z, n_clusters=2, seed=42)
    # Faction A members have 0 distance to Faction A
    assert np.allclose(local_residuals[0, 0:4], 0.0, atol=1e-5)
    # Faction B members have 0 distance to Faction B
    assert np.allclose(local_residuals[0, 4:8], 0.0, atol=1e-5)
    # The 2 origin agents have distance 1.0 from both faction centroids
    assert np.allclose(local_residuals[0, 8:10], 1.0, atol=1e-5)
    # Proves local residuals solve the Phantom Center problem!


def test_detect_orthogonal_agent_transverse():
    T, N, d = 3, 5, 4
    speakers = [f"Agent_{i}" for i in range(N)]
    Z = np.zeros((T, N, d))

    # Agents 0..3 form two clusters moving along dimension 0 (collective drift)
    for t in range(T):
        drift = 0.5 * t
        Z[t, 0:2, 0] = 1.0 + drift
        Z[t, 2:4, 0] = -1.0 + drift

        # Agent 4 is an orthogonal catalyst moving along dimension 1 (transverse 90° subspace)
        Z[t, 4, 1] = 4.0 + 1.0 * t

    ortho_speaker, ortho_idx = detect_orthogonal_from_states(Z, speakers)
    assert ortho_speaker == "Agent_4"
    assert ortho_idx == 4

    # Verify backward compatible alias
    rogue_spk, rogue_i = detect_rogue_from_states(Z, speakers)
    assert rogue_spk == "Agent_4"
    assert rogue_i == 4


def test_detect_cluster_migrations():
    # 2 rounds, 4 agents. Agent 2 changes cluster assignment from round 1 to round 2
    speakers = ["A", "B", "C", "D"]
    assignments = np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 1], # C migrated from cluster 1 to cluster 0
    ])
    migrations = detect_cluster_migrations(assignments, speakers)
    assert len(migrations) == 1
    assert "C (Cluster 2 -> Cluster 1 at Round 2)" in migrations[0]


def test_classify_topological_state():
    # 1. Dissolution
    assert classify_topological_state(1.02, 2, [], 0.5, 0.2) == "Dissolution"
    # 2. Migration
    assert classify_topological_state(0.90, 2, ["A migrated"], 0.5, 0.2) == "Migration"
    # 3. Dissent
    assert classify_topological_state(0.88, 2, [], 2.5, 0.5) == "Dissent"
    # 4. Factionalization
    assert classify_topological_state(0.92, 3, [], 0.5, 0.4) == "Factionalization"
    # 5. Convergence
    assert classify_topological_state(0.65, 1, [], 0.2, 0.15) == "Convergence"

