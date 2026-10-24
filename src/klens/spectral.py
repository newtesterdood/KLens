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


def detect_rogue_from_states(
    Z: np.ndarray,
    speakers: Sequence[str],
) -> Tuple[str, int]:
    T, N, d = Z.shape
    Z_mean = Z.mean(axis=1, keepdims=True)
    delta_3d = Z - Z_mean
    avg_disagreements = np.linalg.norm(delta_3d, axis=2).mean(axis=0)
    rogue_idx = int(np.argmax(avg_disagreements))
    return speakers[rogue_idx], rogue_idx


def detect_rogue_agent(
    Delta: np.ndarray,
    speakers: Sequence[str],
    T: int,
    N: int,
    d: int,
) -> Tuple[str, int]:
    delta_3d = Delta.reshape(T, N, d)
    avg_disagreements = np.linalg.norm(delta_3d, axis=2).mean(axis=0)
    rogue_idx = int(np.argmax(avg_disagreements))
    return speakers[rogue_idx], rogue_idx


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
    rogue_idx: int,
    similarity_threshold: float = 0.65,
) -> List[Faction]:
    N = len(speakers)
    sim_matrix = cosine_similarity(final_states)

    factions: List[Faction] = []
    visited = set()
    non_rogue_indices = [i for i in range(N) if i != rogue_idx]

    faction_counter = 1
    for i in non_rogue_indices:
        if i in visited:
            continue
        members = [speakers[i]]
        visited.add(i)
        for j in non_rogue_indices:
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

    predicted_next_rogue = max(agent_norms.items(), key=lambda kv: kv[1])[0]

    current_total_norm = float(np.linalg.norm(Delta[-1]))
    pred_total_norm = float(np.linalg.norm(delta_pred_flat))
    contraction_ratio = (pred_total_norm / current_total_norm) if current_total_norm > 1e-12 else 1.0

    return predicted_next_rogue, round(contraction_ratio, 4), agent_norms

