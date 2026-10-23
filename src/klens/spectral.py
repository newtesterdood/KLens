from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
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
