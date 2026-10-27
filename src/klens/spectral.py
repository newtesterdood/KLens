from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from klens.models import Faction


def lift_state(delta_matrix: np.ndarray, W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    n_samples = delta_matrix.shape[0]
    bias = np.ones((n_samples, 1))
    rff = np.cos(delta_matrix @ W_f + b_f)
    return np.hstack([bias, delta_matrix, rff])


def compute_koopman_metrics(
    Z: np.ndarray,
    num_rff: int = 100,
    ridge_rho: float = 1e-6,
    epsilon: float = 0.05,
    unit_filter: float = 0.985,
    seed: int = 42,
    solver: str = "auto",
) -> Tuple[float, int, np.ndarray, np.ndarray]:
    T, N, d = Z.shape
    if T < 2 or N < 2:
        return 0.99, 999, np.zeros((T, N * d)), np.zeros((T, d))

    Z_flat = Z.reshape(T, N * d)

    Z_mean = Z.mean(axis=1, keepdims=True)
    Z_mean_tiled = np.tile(Z_mean, (1, N, 1)).reshape(T, N * d)
    Delta = Z_flat - Z_mean_tiled

    input_dim = N * d
    rng = np.random.RandomState(seed)
    W_f = rng.normal(0, 0.7 / np.sqrt(input_dim), size=(input_dim, num_rff))
    b_f = rng.uniform(0, 2 * np.pi, size=(num_rff,))

    Psi = lift_state(Delta[:-1, :], W_f, b_f)
    Psi_plus = lift_state(Delta[1:, :], W_f, b_f)

    obs_dim = Psi.shape[1]
    S = Psi.shape[0]

    use_dual = (solver == "dual") or (solver == "auto" and S < obs_dim and obs_dim > 200)

    if use_dual:
        G_dual = Psi @ Psi.T + ridge_rho * np.eye(S)
        A_dual = Psi_plus @ Psi.T
        K_dual = np.linalg.solve(G_dual, A_dual)
        eigenvalues, _ = np.linalg.eig(K_dual)
    else:
        G = Psi.T @ Psi + ridge_rho * np.eye(obs_dim)
        A = Psi.T @ Psi_plus
        K_hat = np.linalg.solve(G, A)
        eigenvalues, _ = np.linalg.eig(K_hat)

    filtered_mask = np.abs(eigenvalues) < unit_filter
    valid_eigs = eigenvalues[filtered_mask]

    if len(valid_eigs) == 0:
        valid_eigs = eigenvalues

    sub_dom_idx = int(np.argmax(np.abs(valid_eigs)))
    lambda_2 = valid_eigs[sub_dom_idx]
    abs_l2 = float(np.abs(lambda_2))

    if abs_l2 >= 1.0 - 1e-12:
        T_cert = 999
    elif abs_l2 <= 1e-12:
        T_cert = 1
    else:
        denom = -math.log(abs_l2)
        T_pred = math.log(1.0 / epsilon) / (denom + 1e-12)
        T_cert = int(math.ceil(T_pred))

    return abs_l2, T_cert, Delta, Z_mean[:, 0, :]


def compute_multi_cluster_residuals(
    Z: np.ndarray,
    n_clusters: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes multi-cluster local residuals d_i^{local}(t) = min_k ||z_i(t) - C_k(t)||
    to resolve the Phantom Center problem when factions >= 2.

    Returns:
        local_residuals: (T, N) array of minimum Euclidean distances to nearest cluster centroid.
        cluster_assignments: (T, N) integer cluster index per agent per round.
        centroids: (T, K, d) cluster centroids per round.
    """
    is_2d = False
    if Z.ndim == 2:
        is_2d = True
        Z = Z[np.newaxis, :, :]

    T, N, d = Z.shape
    speakers = [f"Agent_{i}" for i in range(N)]

    local_residuals = np.zeros((T, N))
    cluster_assignments = np.zeros((T, N), dtype=int)
    centroids_list = []

    for t in range(T):
        states_t = Z[t]

        # First, check for emergent cohesive multi-member factions via cosine similarity
        factions = cluster_factions(states_t, speakers, similarity_threshold=0.60)
        multi_member = [f for f in factions if len(f.members) >= 2]

        if len(multi_member) >= 1:
            viable_centers = np.array([
                states_t[[speakers.index(s) for s in f.members]].mean(axis=0)
                for f in multi_member
            ])
        else:
            k = n_clusters if n_clusters is not None else min(4, max(2, N // 3))
            unique_states = len(np.unique(states_t, axis=0))
            cur_k = max(1, min(k, unique_states))
            if cur_k < 2:
                viable_centers = states_t.mean(axis=0, keepdims=True)
            else:
                kmeans = KMeans(n_clusters=cur_k, random_state=seed, n_init="auto").fit(states_t)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                counts = np.bincount(labels, minlength=cur_k)
                viable_idx = np.where(counts >= 2)[0]
                if len(viable_idx) == 0:
                    viable_idx = np.arange(cur_k)
                viable_centers = centers[viable_idx]

        centroids_list.append(viable_centers)
        dists_to_clusters = np.linalg.norm(states_t[:, np.newaxis, :] - viable_centers[np.newaxis, :, :], axis=2)
        local_residuals[t] = np.min(dists_to_clusters, axis=1)
        cluster_assignments[t] = np.argmin(dists_to_clusters, axis=1)

    max_k_found = max(c.shape[0] for c in centroids_list)
    centroids_arr = np.zeros((T, max_k_found, d))
    for t, c in enumerate(centroids_list):
        centroids_arr[t, :c.shape[0], :] = c

    if is_2d:
        return local_residuals[0], cluster_assignments[0], centroids_arr[0]
    return local_residuals, cluster_assignments, centroids_arr


def detect_cluster_migrations(
    cluster_assignments: np.ndarray,
    speakers: Sequence[str],
) -> List[str]:
    """
    Tracks when an agent's nearest cluster assignment changes across consecutive rounds.
    """
    if cluster_assignments.ndim == 1:
        return []

    T, N = cluster_assignments.shape
    migrations: List[str] = []
    for t in range(1, T):
        for i in range(N):
            prev_c = cluster_assignments[t - 1, i]
            curr_c = cluster_assignments[t, i]
            if prev_c != curr_c:
                migrations.append(
                    f"{speakers[i]} (Cluster {prev_c + 1} -> Cluster {curr_c + 1} at Round {t + 1})"
                )
    return migrations


def detect_orthogonal_from_states(
    Z: np.ndarray,
    speakers: Sequence[str],
    seed: int = 42,
) -> Tuple[str, int]:
    """
    Detects the primary Orthogonal Agent: an agent that is persistently distant from all
    emergent cluster centroids AND whose trajectory is decoupled / transverse to collective drift.
    """
    T, N, d = Z.shape
    if N <= 1:
        return speakers[0], 0

    local_res, _, _ = compute_multi_cluster_residuals(Z, seed=seed)
    mean_residuals = local_res.mean(axis=0)

    # Transverse decoupling factor relative to collective group drift
    if T >= 2:
        drift = Z.mean(axis=1)[-1] - Z.mean(axis=1)[0]
        drift_norm = float(np.linalg.norm(drift))
        decoupling_factors = np.ones(N)
        if drift_norm > 1e-6:
            for i in range(N):
                vel = Z[-1, i] - Z[0, i]
                vel_norm = float(np.linalg.norm(vel))
                if vel_norm > 1e-6:
                    cos_theta = float(np.dot(vel, drift) / (vel_norm * drift_norm))
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                    # High transverse score when orthogonal to collective drift (cos_theta ~ 0)
                    transverse_score = 1.0 - abs(cos_theta)
                    decoupling_factors[i] = 1.0 + 0.4 * transverse_score
        scores = mean_residuals * decoupling_factors
    else:
        scores = mean_residuals

    ortho_idx = int(np.argmax(scores))
    return speakers[ortho_idx], ortho_idx


def detect_rogue_from_states(
    Z: np.ndarray,
    speakers: Sequence[str],
) -> Tuple[str, int]:
    """
    Backward-compatible alias pointing to detect_orthogonal_from_states.
    """
    return detect_orthogonal_from_states(Z, speakers)


def detect_orthogonal_agent(
    Delta: np.ndarray,
    speakers: Sequence[str],
    T: int,
    N: int,
    d: int,
    seed: int = 42,
) -> Tuple[str, int]:
    """
    Detects orthogonal agent from disagreement tensor Delta.
    """
    Z_proxy = Delta.reshape(T, N, d)
    return detect_orthogonal_from_states(Z_proxy, speakers, seed=seed)


def detect_rogue_agent(
    Delta: np.ndarray,
    speakers: Sequence[str],
    T: int,
    N: int,
    d: int,
) -> Tuple[str, int]:
    """
    Backward-compatible alias pointing to detect_orthogonal_agent.
    """
    return detect_orthogonal_agent(Delta, speakers, T, N, d)


def classify_topological_state(
    l2_magnitude: float,
    faction_count: int,
    migrations: List[str],
    max_residual: float,
    median_residual: float,
    predicted_contraction: Optional[float] = None,
) -> str:
    """
    Classifies collective reasoning into one of the 5 canonical Topological States:
    1. Dissolution: Dispersed or expanding entropy (|lambda_2| >= 1.0).
    2. Migration: Dynamic cluster reassignment across rounds.
    3. Factionalization: Distinct multi-cluster basins (K >= 2, slow decay).
    4. Dissent: Persistent isolation of an orthogonal or contrarian agent.
    5. Convergence: Collapsing consensus attractor (|lambda_2| < 0.85).
    """
    if l2_magnitude >= 1.0:
        return "Dissolution"
    if migrations:
        return "Migration"
    if max_residual > 1.8 * max(median_residual, 1e-4):
        return "Dissent"
    if faction_count >= 2 and l2_magnitude >= 0.85:
        return "Factionalization"
    return "Convergence"


def coarse_grain_faction_centroids(
    Z: np.ndarray,
    speakers: Sequence[str],
    max_factions: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Faction]]:
    T, N, d = Z.shape
    k = min(N, max_factions)

    if k >= N:
        factions = [Faction(id=i + 1, members=[speakers[i]]) for i in range(N)]
        return Z, factions

    agent_trajectories = Z.mean(axis=0)
    unique_count = len(np.unique(agent_trajectories, axis=0))
    if unique_count >= 2:
        k = max(2, min(k, unique_count))
    else:
        k = min(2, N)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(agent_trajectories)
    labels = kmeans.labels_

    Z_macro = np.zeros((T, k, d))
    factions: List[Faction] = []

    for cluster_id in range(k):
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) > 0:
            Z_macro[:, cluster_id, :] = Z[:, member_indices, :].mean(axis=1)
            members = [speakers[idx] for idx in member_indices]
        else:
            members = []
        factions.append(Faction(id=cluster_id + 1, members=members))

    return Z_macro, factions


def cluster_factions(
    final_states: np.ndarray,
    speakers: Sequence[str],
    rogue_idx: Optional[int] = None,
    similarity_threshold: float = 0.65,
    orthogonal_idx: Optional[int] = None,
) -> List[Faction]:
    N = len(speakers)
    sim_matrix = cosine_similarity(final_states)

    factions: List[Faction] = []
    visited = set()
    excluded_idx = orthogonal_idx if orthogonal_idx is not None else rogue_idx
    non_isolated_indices = [i for i in range(N) if i != excluded_idx]

    faction_counter = 1
    for i in non_isolated_indices:
        if i in visited:
            continue
        members = [speakers[i]]
        visited.add(i)
        for j in non_isolated_indices:
            if j not in visited and sim_matrix[i, j] >= similarity_threshold:
                members.append(speakers[j])
                visited.add(j)
        factions.append(Faction(id=faction_counter, members=members))
        faction_counter += 1

    return factions


def extract_centroid_keywords(
    final_centroid: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 4,
) -> List[str]:
    feature_arr = np.asarray(feature_names)
    top_k_indices = np.argsort(final_centroid)[::-1][:top_k]
    return [str(feature_arr[idx]) for idx in top_k_indices if final_centroid[idx] > 0]


def predict_agent_trajectories(
    Z: np.ndarray,
    speakers: Sequence[str],
    steps_ahead: int = 1,
    num_rff: int = 100,
    ridge_rho: float = 1e-6,
    seed: int = 42,
    solver: str = "auto",
) -> Tuple[Optional[str], float, dict[str, float]]:
    T, N, d = Z.shape
    if T < 2 or N < 2:
        return None, 1.0, {}

    Z_flat = Z.reshape(T, N * d)
    Z_mean = Z.mean(axis=1, keepdims=True)
    Z_mean_tiled = np.tile(Z_mean, (1, N, 1)).reshape(T, N * d)
    Delta = Z_flat - Z_mean_tiled

    input_dim = N * d
    rng = np.random.RandomState(seed)
    W_f = rng.normal(0, 0.7 / np.sqrt(input_dim), size=(input_dim, num_rff))
    b_f = rng.uniform(0, 2 * np.pi, size=(num_rff,))

    Psi = lift_state(Delta[:-1, :], W_f, b_f)
    Psi_plus = lift_state(Delta[1:, :], W_f, b_f)

    obs_dim = Psi.shape[1]
    S = Psi.shape[0]

    use_dual = (solver == "dual") or (solver == "auto" and S < obs_dim and obs_dim > 200)

    psi_current = lift_state(Delta[-1:, :], W_f, b_f)

    if use_dual:
        G_dual = Psi @ Psi.T + ridge_rho * np.eye(S)
        M_dual = np.linalg.solve(G_dual, Psi_plus)
        psi_pred = psi_current
        for _ in range(steps_ahead):
            psi_pred = (psi_pred @ Psi.T) @ M_dual
    else:
        G = Psi.T @ Psi + ridge_rho * np.eye(obs_dim)
        A = Psi.T @ Psi_plus
        K_hat = np.linalg.solve(G, A)
        if steps_ahead == 1:
            K_pow = K_hat
        else:
            K_pow = np.linalg.matrix_power(K_hat, steps_ahead)
        psi_pred = psi_current @ K_pow

    delta_pred_flat = psi_pred[0, 1:1 + N * d]
    delta_pred_3d = delta_pred_flat.reshape(N, d)

    agent_norms: dict[str, float] = {}
    for i, spk in enumerate(speakers):
        agent_norms[spk] = float(np.linalg.norm(delta_pred_3d[i]))

    predicted_next_agent = max(agent_norms.items(), key=lambda kv: kv[1])[0]

    current_total_norm = float(np.linalg.norm(Delta[-1]))
    pred_total_norm = float(np.linalg.norm(delta_pred_flat))
    contraction_ratio = (pred_total_norm / current_total_norm) if current_total_norm > 1e-12 else 1.0

    return predicted_next_agent, round(contraction_ratio, 4), agent_norms
