import math
import numpy as np
import pytest

from klens.models import Faction, TranscriptTurn
from klens.semantic import synthesize_deliberation_topics
from klens.spectral import (
    compute_effective_channels_k_star,
    compute_koopman_metrics,
    compute_multi_cluster_residuals,
    compute_nnls_subspace_orthogonality,
    compute_orthogonal_significance,
    predict_agent_trajectories,
    validate_koopman_walk_forward,
)
from klens.encoders import TfidfTextEncoder


def _linear_decay_tensor(alpha: float, T: int = 8, N: int = 4, d: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.randn(N, d)
    base -= base.mean(axis=0, keepdims=True)
    return np.array([(alpha ** t) * base for t in range(T)], dtype=float)


def _rotational_tensor(r: float, theta: float, T: int = 10) -> np.ndarray:
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    cur = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=float)
    Z = np.zeros((T, 4, 2), dtype=float)
    for t in range(T):
        Z[t] = cur
        cur = r * (cur @ R.T)
    return Z


@pytest.mark.parametrize(
    "Z",
    [
        _linear_decay_tensor(0.5),
        _linear_decay_tensor(0.85),
        _linear_decay_tensor(1.0),
        _linear_decay_tensor(1.15),
        _linear_decay_tensor(1.35),
        _rotational_tensor(0.9, 0.6),
        _rotational_tensor(1.0, math.pi / 3),
    ],
)
def test_primal_dual_exact_equivalence_all_regimes(Z):
    speakers = [f"A{i}" for i in range(Z.shape[1])]
    l2_p, t_p, d_p, m_p = compute_koopman_metrics(Z, solver="primal")
    l2_d, t_d, d_d, m_d = compute_koopman_metrics(Z, solver="dual")
    assert l2_p == pytest.approx(l2_d, abs=1e-6)
    assert t_p == t_d
    assert np.allclose(d_p, d_d, atol=1e-12)
    assert np.allclose(m_p, m_d, atol=1e-12)

    p_spk, p_rat, p_norms = predict_agent_trajectories(Z, speakers, steps_ahead=2, solver="primal")
    d_spk, d_rat, d_norms = predict_agent_trajectories(Z, speakers, steps_ahead=2, solver="dual")
    assert p_rat == pytest.approx(d_rat, abs=1e-4)
    for spk in speakers:
        assert p_norms[spk] == pytest.approx(d_norms[spk], abs=1e-5)
    sorted_norms = sorted(p_norms.values(), reverse=True)
    if len(sorted_norms) < 2 or (sorted_norms[0] - sorted_norms[1]) > 1e-6:
        assert p_spk == d_spk

    wf_ep, wf_cp = validate_koopman_walk_forward(Z, solver="primal")
    wf_ed, wf_cd = validate_koopman_walk_forward(Z, solver="dual")
    assert wf_ep == pytest.approx(wf_ed, abs=1e-4)
    assert wf_cp == pytest.approx(wf_cd, abs=1e-4)


@pytest.mark.parametrize("alpha", [0.4, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("epsilon", [0.05, 0.01])
def test_koopman_contractive_eigenvalue_and_t_cert_formula(alpha, epsilon):
    Z = _linear_decay_tensor(alpha, T=9, N=4, d=5, seed=3)
    l2, t_cert, _, _ = compute_koopman_metrics(Z, epsilon=epsilon, num_rff=60)
    assert l2 == pytest.approx(alpha, abs=0.025)
    expected_t = min(999, int(math.ceil(math.log(1.0 / epsilon) / (-math.log(l2) + 1e-12))))
    assert t_cert == expected_t


def test_koopman_recovers_slowest_bottleneck_mode():
    T, N, d = 10, 4, 4
    v_slow = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]])
    v_fast = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
    Z = np.array([(0.88 ** t) * v_slow + (0.25 ** t) * v_fast for t in range(T)])

    l2, _, _, _ = compute_koopman_metrics(Z, num_rff=40)
    assert l2 == pytest.approx(0.88, abs=0.03)


def test_koopman_regime_boundary_rigor():
    Z_zero = np.ones((6, 4, 5)) * 3.14
    l2_z, t_z, _, _ = compute_koopman_metrics(Z_zero)
    assert l2_z == 0.0
    assert t_z == 1

    Z_stale = _linear_decay_tensor(1.0, T=6, N=4, d=5)
    l2_s, t_s, _, _ = compute_koopman_metrics(Z_stale)
    assert l2_s == 1.0
    assert t_s == 999

    Z_cycle = _rotational_tensor(1.0, math.pi / 3, T=10)
    l2_c, t_c, _, _ = compute_koopman_metrics(Z_cycle)
    assert l2_c == pytest.approx(1.0, abs=1e-3)
    assert t_c == 999

    Z_spiral = _rotational_tensor(0.85, 0.7, T=10)
    l2_sp, t_sp, _, _ = compute_koopman_metrics(Z_spiral)
    assert l2_sp == pytest.approx(0.85, abs=0.025)
    assert 15 <= t_sp <= 25

    Z_div = _linear_decay_tensor(1.2, T=8, N=4, d=5)
    l2_div, t_div, _, _ = compute_koopman_metrics(Z_div)
    assert l2_div > 1.0
    assert t_div == 999


def test_koopman_translation_invariance():
    Z = _linear_decay_tensor(0.72, T=8, N=5, d=6, seed=9)
    rng = np.random.RandomState(99)
    global_drift = rng.randn(Z.shape[0], 1, Z.shape[2]) * 10.0
    Z_shifted = Z + global_drift

    l2_a, t_a, delta_a, _ = compute_koopman_metrics(Z)
    l2_b, t_b, delta_b, _ = compute_koopman_metrics(Z_shifted)
    assert np.allclose(delta_a, delta_b, atol=1e-10)
    assert l2_a == pytest.approx(l2_b, abs=1e-9)
    assert t_a == t_b

    speakers = [f"S{i}" for i in range(Z.shape[1])]
    pa, ra, _ = predict_agent_trajectories(Z, speakers, steps_ahead=2)
    pb, rb, _ = predict_agent_trajectories(Z_shifted, speakers, steps_ahead=2)
    assert pa == pb
    assert ra == pytest.approx(rb, abs=1e-6)

    ea, ca = validate_koopman_walk_forward(Z)
    eb, cb = validate_koopman_walk_forward(Z_shifted)
    assert ea == pytest.approx(eb, abs=1e-6)
    assert ca == pytest.approx(cb, abs=1e-6)


def test_koopman_multistep_contraction_accuracy():
    alpha = 0.7
    Z = _linear_decay_tensor(alpha, T=8, N=4, d=5, seed=5)
    speakers = ["A", "B", "C", "D"]
    true_max_idx = int(np.argmax(np.linalg.norm(Z[-1] - Z[-1].mean(axis=0, keepdims=True), axis=1)))
    for k in (1, 2, 3):
        pred_spk, ratio, _ = predict_agent_trajectories(Z, speakers, steps_ahead=k, num_rff=40)
        assert pred_spk == speakers[true_max_idx]
        assert ratio == pytest.approx(alpha ** k, abs=0.04)


def test_walk_forward_strict_causality_and_noise_discrimination():
    Z_clean = _linear_decay_tensor(0.65, T=7, N=4, d=5, seed=12)
    err_prefix, conf_prefix = validate_koopman_walk_forward(Z_clean[:5], num_rff=24)

    Z_future_shock = Z_clean.copy()
    Z_future_shock[5:] += 50.0 * np.random.RandomState(1).randn(2, 4, 5)
    err_prefix_after, conf_prefix_after = validate_koopman_walk_forward(Z_future_shock[:5], num_rff=24)
    assert err_prefix == err_prefix_after
    assert conf_prefix == conf_prefix_after

    rng = np.random.RandomState(2026)
    Z_noise = rng.randn(7, 4, 5)
    err_clean, conf_clean = validate_koopman_walk_forward(Z_clean, num_rff=24)
    err_noise, conf_noise = validate_koopman_walk_forward(Z_noise, num_rff=24)
    assert err_clean is not None and err_noise is not None
    assert err_clean < 0.15
    assert conf_clean > 0.85
    assert err_clean < err_noise * 0.35
    assert conf_clean > conf_noise + 0.30


def test_k_star_axioms_and_closed_forms():
    rng = np.random.RandomState(17)
    X = rng.randn(7, 12)
    k0 = compute_effective_channels_k_star(X)
    assert 1.0 <= k0 <= min(X.shape)

    # Positive per-agent scaling invariance
    scales = rng.uniform(0.2, 8.0, size=(7, 1))
    assert compute_effective_channels_k_star(X * scales) == pytest.approx(k0, abs=1e-4)

    # Orthogonal rotation invariance Q in O(d)
    Q, _ = np.linalg.qr(rng.randn(12, 12))
    assert compute_effective_channels_k_star(X @ Q) == pytest.approx(k0, abs=1e-4)

    # Agent permutation invariance
    perm = rng.permutation(7)
    assert compute_effective_channels_k_star(X[perm]) == pytest.approx(k0, abs=1e-4)

    # Exact 2-vector closed form: K*(theta) = 2 / (1 + cos^2(theta))
    for theta in (0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2):
        V = np.array([[1.0, 0.0], [math.cos(theta), math.sin(theta)]])
        expected = 2.0 / (1.0 + math.cos(theta) ** 2)
        assert compute_effective_channels_k_star(V) == pytest.approx(expected, abs=1e-4)

    # Exact Simpson / Herfindahl inverse for mutually orthogonal factions of sizes (4, 2, 1)
    sizes = [4, 2, 1]
    rows = []
    for idx, sz in enumerate(sizes):
        vec = np.zeros(3)
        vec[idx] = 1.0
        rows.extend([vec] * sz)
    expected_simpson = (sum(sizes) ** 2) / sum(s ** 2 for s in sizes)
    assert compute_effective_channels_k_star(np.array(rows)) == pytest.approx(expected_simpson, abs=1e-4)


def test_nnls_orthogonality_conic_axioms():
    # Exact 2-vector acute angle identity: min_{w >= 0} ||w*u1 - u2|| = sin(phi)
    for phi in (0.15, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2):
        u1 = np.array([1.0, 0.0, 0.0])
        u2 = np.array([math.cos(phi), math.sin(phi), 0.0])
        sins = compute_nnls_subspace_orthogonality([u1, u2])
        assert sins[0] == pytest.approx(math.sin(phi), abs=1e-4)
        assert sins[1] == pytest.approx(math.sin(phi), abs=1e-4)

    # Conic vs linear span distinction: -u is outside positive cone of {+u} -> sin(theta^+) = 1.0
    sins_opp = compute_nnls_subspace_orthogonality([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    assert sins_opp == [1.0, 1.0]

    # Rotation, positive scale, and permutation equivariance
    rng = np.random.RandomState(42)
    vecs = [rng.randn(6) for _ in range(5)]
    base = compute_nnls_subspace_orthogonality(vecs)
    assert all(0.0 <= s <= 1.0 for s in base)

    Q, _ = np.linalg.qr(rng.randn(6, 6))
    rot = compute_nnls_subspace_orthogonality([v @ Q for v in vecs])
    scaled = compute_nnls_subspace_orthogonality([(i + 1.5) * v for i, v in enumerate(vecs)])
    perm = [3, 0, 4, 1, 2]
    permuted = compute_nnls_subspace_orthogonality([vecs[i] for i in perm])
    assert rot == pytest.approx(base, abs=1e-3)
    assert scaled == pytest.approx(base, abs=1e-3)
    assert permuted == pytest.approx([base[i] for i in perm], abs=1e-4)


def test_orthogonal_transverse_decoupling_and_mad_calibration():
    # Symmetric 3-way Phantom Center: 3 factions at 120 deg + 1 agent at origin
    T = 3
    angles = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    Z_tri = np.zeros((T, 10, 2))
    for t in range(T):
        for f_idx, ang in enumerate(angles):
            Z_tri[t, f_idx * 3 : (f_idx + 1) * 3] = [math.cos(ang), math.sin(ang)]
        Z_tri[t, 9] = [0.0, 0.0]
    res_tri, _, _ = compute_multi_cluster_residuals(Z_tri, n_clusters=3, seed=42)
    assert np.allclose(res_tri[0, :9], 0.0, atol=1e-5)
    assert res_tri[0, 9] == pytest.approx(1.0, abs=1e-4)

    # Transverse velocity decoupling: equal cluster residual, perpendicular velocity wins over parallel
    speakers = ["C0", "C1", "C2", "C3", "ParallelOut", "PerpOut"]
    Z_vel = np.zeros((4, 6, 3))
    for t in range(4):
        drift = 1.0 * t
        Z_vel[t, 0:4] = [drift, 0.0, 0.0]
        Z_vel[t, 4] = [drift, 2.0, 0.0]  # velocity = (1, 0, 0) parallel to drift
        Z_vel[t, 5] = [0.0, 0.0, 2.0 + 1.0 * t]  # velocity = (0, 0, 1) transverse to drift
    # Make PerpOut exact same distance (2.0) at t=0 and close mean residual by keeping perp amplitude around cluster
    Z_vel_eq = np.zeros((4, 6, 3))
    for t in range(4):
        drift = 0.5 * t
        Z_vel_eq[t, 0:4] = [drift, 0.0, 0.0]
        Z_vel_eq[t, 4] = [drift, 2.0, 0.0]
        Z_vel_eq[t, 5] = [0.0, 0.0, 2.0]
    # Give ParallelOut a slight parallel displacement and PerpOut an equal-norm transverse displacement
    Z_vel_eq[:, 4, 0] += np.linspace(0.0, 0.2, 4)
    Z_vel_eq[:, 5, 2] += np.linspace(0.0, 0.2, 4)
    spk_out, idx_out, _, _ = compute_orthogonal_significance(Z_vel_eq, speakers)
    assert spk_out == "PerpOut"
    assert idx_out == 5

    # MAD Z monotonicity as outlier distance increases
    z_scores = []
    for amp in (0.4, 2.0, 6.0):
        Z_amp = np.zeros((4, 6, 3))
        for t in range(4):
            for i in range(5):
                Z_amp[t, i] = [1.0 + 0.1 * t, 0.02 * i, 0.0]
            Z_amp[t, 5] = [0.0, 0.0, amp]
        _, _, mod_z, is_sig = compute_orthogonal_significance(Z_amp, [f"A{i}" for i in range(6)])
        z_scores.append((mod_z, is_sig))
    assert z_scores[0][0] < z_scores[1][0] < z_scores[2][0]
    assert z_scores[2][1] is True


def test_semantic_simplex_conservation_and_svd_axis():
    records = [
        TranscriptTurn(1, "Alpha", "Strict privacy encryption and zero telemetry compliance are mandatory."),
        TranscriptTurn(1, "Beta", "Strict privacy encryption and zero telemetry compliance protect users."),
        TranscriptTurn(1, "Gamma", "Fast cloud throughput and low latency scaling maximize revenue."),
        TranscriptTurn(1, "Delta", "Fast cloud throughput and low latency scaling win deployment."),
        TranscriptTurn(1, "Silent", "Opening procedural greeting."),
        TranscriptTurn(2, "Alpha", "Privacy encryption compliance prevents breaches."),
        TranscriptTurn(2, "Beta", "Privacy encryption compliance ensures regulatory trust."),
        TranscriptTurn(2, "Gamma", "Cloud throughput latency scaling drives adoption."),
        TranscriptTurn(2, "Delta", "Cloud throughput latency scaling reduces infrastructure cost."),
    ]
    rounds = [1, 2]
    speakers = ["Alpha", "Beta", "Delta", "Gamma", "Silent"]
    grid = {(r, s): "" for r in rounds for s in speakers}
    for rec in records:
        grid[(rec.round_id, rec.speaker)] += " " + rec.text

    enc = TfidfTextEncoder(max_features=20)
    Z, feat_names = enc.encode_grid(grid, rounds, speakers)
    factions = [
        Faction(id=1, members=["Alpha", "Beta"]),
        Faction(id=2, members=["Gamma", "Delta"]),
    ]
    synth = synthesize_deliberation_topics(
        records=records,
        Z=Z,
        feature_names=feat_names,
        speakers=speakers,
        rounds=rounds,
        factions=factions,
        rogue_speaker="Silent",
        vectorizer=enc.vectorizer,
    )
    active_weights = [f.consensus_weight for f in synth.factions if f.dynamical_role != "Inactive Panelist"]
    if synth.orthogonal_stance and synth.orthogonal_stance.dynamical_role != "Inactive Panelist":
        active_weights.append(synth.orthogonal_stance.consensus_weight)
    assert sum(active_weights) == pytest.approx(1.0, abs=1e-3)
    assert synth.orthogonal_stance is not None
    assert synth.orthogonal_stance.dynamical_role == "Inactive Panelist"
    assert synth.orthogonal_stance.consensus_weight == 0.0

    axis_lower = synth.center_of_disagreement.lower()
    assert any(tok in axis_lower for tok in ("privacy", "encryption", "compliance"))
    assert any(tok in axis_lower for tok in ("cloud", "latency", "scaling", "throughput"))
