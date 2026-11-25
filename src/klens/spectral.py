from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import nnls
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

    max_delta_norm = float(np.max(np.linalg.norm(Delta, axis=1)))
    if max_delta_norm <= 1e-12:
        return 0.0, 1, Delta, Z_mean[:, 0, :]
    max_step_diff = float(np.max(np.linalg.norm(np.diff(Delta, axis=0), axis=1)))
    if max_step_diff <= 1e-12:
        return 1.0, 999, Delta, Z_mean[:, 0, :]

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
        if obs_dim > S:
            top_idx = np.argsort(np.abs(eigenvalues))[::-1][:S]
            eigenvalues = eigenvalues[top_idx]

    tol = max(1e-6, 1.0 - unit_filter)
    dist_to_one = np.abs(eigenvalues - 1.0)
    closest_idx = int(np.argmin(dist_to_one))
    if dist_to_one[closest_idx] <= tol and len(eigenvalues) > 1:
        valid_eigs = np.delete(eigenvalues, closest_idx)
    else:
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
        T_cert = min(999, int(math.ceil(T_pred)))

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


def compute_orthogonal_significance(
    Z: np.ndarray,
    speakers: Sequence[str],
    seed: int = 42,
    z_threshold: float = 2.0,
) -> Tuple[str, int, float, bool]:
    """
    Detects the primary Orthogonal Agent and evaluates statistical significance via
    robust Modified Z-Score (0.6745 * (score - median) / MAD).
    Returns (speaker, idx, modified_z_score, is_statistically_orthogonal).
    """
    T, N, d = Z.shape
    if N <= 1:
        return speakers[0], 0, 0.0, False

    local_res, _, _ = compute_multi_cluster_residuals(Z, seed=seed)
    mean_residuals = local_res.mean(axis=0)

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
                    transverse_score = 1.0 - abs(cos_theta)
                    decoupling_factors[i] = 1.0 + 0.4 * transverse_score
        scores = mean_residuals * decoupling_factors
    else:
        scores = mean_residuals

    ortho_idx = int(np.argmax(scores))
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    if mad > 1e-12:
        mod_z = float(0.6745 * (scores[ortho_idx] - med) / mad)
    else:
        std = float(np.std(scores))
        mod_z = float((scores[ortho_idx] - med) / std) if std > 1e-12 else 0.0

    mod_z_rounded = round(mod_z, 4)
    return speakers[ortho_idx], ortho_idx, mod_z_rounded, bool(mod_z >= z_threshold)


def detect_orthogonal_from_states(
    Z: np.ndarray,
    speakers: Sequence[str],
    seed: int = 42,
) -> Tuple[str, int]:
    """
    Detects the primary Orthogonal Agent: an agent that is persistently distant from all
    emergent cluster centroids AND whose trajectory is decoupled / transverse to collective drift.
    """
    speaker, idx, _, _ = compute_orthogonal_significance(Z, speakers, seed=seed)
    return speaker, idx


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


def compute_effective_channels_k_star(states: np.ndarray) -> float:
    """
    Computes the effective number of reasoning channels K* via spectral participation ratio.
    Accepts states of shape (N, d) or (T, N, d) (evaluates at final round states[-1]).
    """
    if states.ndim == 3:
        if states.shape[0] == 0:
            return 1.0
        X_raw = states[-1]
    elif states.ndim == 2:
        X_raw = states
    else:
        return 1.0

    N, d = X_raw.shape
    if N <= 1 or d == 0:
        return 1.0

    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    nonzero_mask = (norms[:, 0] > 1e-12)
    if not np.any(nonzero_mask):
        return 1.0

    X = np.zeros_like(X_raw, dtype=float)
    X[nonzero_mask] = X_raw[nonzero_mask] / norms[nonzero_mask]

    G = X @ X.T
    eigvals = np.linalg.eigvalsh(G)
    gamma = np.maximum(0.0, eigvals)

    s1 = float(np.sum(gamma))
    s2 = float(np.sum(gamma ** 2))
    if s2 <= 1e-12:
        return 1.0

    k_star = (s1 ** 2) / s2
    k_star_clipped = float(np.clip(k_star, 1.0, float(N)))
    return round(k_star_clipped, 4)


def extract_last_active_states(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns each agent's last non-zero state vector across rounds and a boolean mask
    indicating whether each agent was active at the terminal round.
    """
    if Z.ndim == 2:
        Z = Z[np.newaxis, :, :]
    elif Z.ndim != 3:
        raise ValueError(f"Expected 2D or 3D state array, got shape {Z.shape}")

    T, N, d = Z.shape
    if T == 0:
        return np.zeros((N, d), dtype=float), np.zeros((N,), dtype=bool)

    active_terminal_mask = np.linalg.norm(Z[-1], axis=1) > 1e-12
    last_active_states = np.zeros((N, d), dtype=float)

    for i in range(N):
        if active_terminal_mask[i]:
            last_active_states[i] = Z[-1, i]
            continue
        for t in range(T - 2, -1, -1):
            if np.linalg.norm(Z[t, i]) > 1e-12:
                last_active_states[i] = Z[t, i]
                break

    return last_active_states, active_terminal_mask


def compute_nnls_subspace_orthogonality(
    channel_vectors: Sequence[np.ndarray],
) -> List[float]:
    """
    Computes each active channel's exact non-negative conic orthogonal residual
    sin(theta_k^+) = min_{w >= 0} ||P_{-k} w - c_hat_k||_2 via NNLS.
    Returns 0.0 for zero-norm (inactive) channels.
    """
    K = len(channel_vectors)
    if K == 0:
        return []

    norms = [float(np.linalg.norm(v)) for v in channel_vectors]
    unit_vecs = [
        (np.asarray(v, dtype=float) / n) if n > 1e-12 else None
        for v, n in zip(channel_vectors, norms)
    ]

    sin_thetas: List[float] = []
    for k in range(K):
        t_hat = unit_vecs[k]
        if t_hat is None:
            sin_thetas.append(0.0)
            continue
        peers = [unit_vecs[j] for j in range(K) if j != k and unit_vecs[j] is not None]
        if not peers:
            sin_thetas.append(0.0)
            continue
        P = np.column_stack(peers)
        _, residual = nnls(P, t_hat)
        sin_thetas.append(round(float(np.clip(residual, 0.0, 1.0)), 4))

    return sin_thetas


def validate_koopman_walk_forward(
    Z: np.ndarray,
    num_rff: int = 100,
    ridge_rho: float = 1e-6,
    seed: int = 42,
    solver: str = "auto",
) -> Tuple[Optional[float], float]:
    """
    Evaluates out-of-sample 1-step EDMD forecast accuracy via walk-forward splits t = 2..T-1.
    Returns (mean_relative_error, fit_confidence) where fit_confidence = exp(-mean_relative_error).
    Returns (None, 0.0) when T < 3 (no held-out transition exists).
    """
    T, N, d = Z.shape
    if T < 3 or N < 2 or d == 0:
        return None, 0.0

    input_dim = N * d
    errors: List[float] = []
    for t in range(2, T):
        Z_sub = Z[:t]
        Z_with_target = Z[: t + 1]
        Delta_true = Z_with_target - Z_with_target.mean(axis=1, keepdims=True)
        target_vec = Delta_true[-1].reshape(-1)
        target_norm = float(np.linalg.norm(target_vec))
        if target_norm <= 1e-12:
            continue

        Z_flat = Z_sub.reshape(t, input_dim)
        Z_mean = Z_sub.mean(axis=1, keepdims=True)
        Delta = Z_flat - np.tile(Z_mean, (1, N, 1)).reshape(t, input_dim)

        rng = np.random.RandomState(seed)
        W_f = rng.normal(0, 0.7 / np.sqrt(input_dim), size=(input_dim, num_rff))
        b_f = rng.uniform(0, 2 * np.pi, size=(num_rff,))

        Psi = lift_state(Delta[:-1, :], W_f, b_f)
        Psi_plus = lift_state(Delta[1:, :], W_f, b_f)
        psi_curr = lift_state(Delta[-1:, :], W_f, b_f)

        obs_dim = Psi.shape[1]
        S = Psi.shape[0]
        use_dual = (solver == "dual") or (solver == "auto" and S < obs_dim and obs_dim > 200)

        if use_dual:
            G_dual = Psi @ Psi.T + ridge_rho * np.eye(S)
            psi_next = (psi_curr @ Psi.T) @ np.linalg.solve(G_dual, Psi_plus)
        else:
            G = Psi.T @ Psi + ridge_rho * np.eye(obs_dim)
            K_hat = np.linalg.solve(G, Psi.T @ Psi_plus)
            psi_next = psi_curr @ K_hat

        delta_pred = psi_next[0, 1 : 1 + input_dim]
        rel_err = float(np.linalg.norm(delta_pred - target_vec) / target_norm)
        errors.append(rel_err)

    if not errors:
        return None, 0.0

    mean_err = float(np.mean(errors))
    confidence = float(math.exp(-mean_err))
    return round(mean_err, 4), round(confidence, 4)
