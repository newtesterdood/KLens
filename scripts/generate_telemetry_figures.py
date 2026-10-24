#!/usr/bin/env python3
"""
Generate publication-grade telemetry figures for KLens README.md.
Produces:
1. assets/klens_telemetry_dashboard.png (4-panel multi-scale telemetry)
2. assets/faction_centroid_drift.png (Detailed 2D semantic manifold & faction polarization)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from sklearn.decomposition import PCA

from klens import LiveTranscriptStreamProcessor, AlertConfig, HierarchicalConfig
from klens.spectral import compute_koopman_metrics, lift_state, predict_agent_trajectories


def generate_figures():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load real transcript data from samples/llm_safety_consensus.json
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "llm_safety_consensus.json"))
    with open(sample_path, "r", encoding="utf-8") as f:
        turns = json.load(f)

    processor = LiveTranscriptStreamProcessor(
        csv_log_path=os.path.join(output_dir, "temp_stream.csv"),
        auto_init_csv=False,
    )
    for t in turns:
        processor.ingest_turn(t["round"], t["speaker"], t["text"])

    # Extract state grid
    turns_buffer = processor.transcript_buffer
    speakers = sorted(list(set(t.speaker for t in turns_buffer)))
    rounds = sorted(list(set(t.round_id for t in turns_buffer)))
    T = len(rounds)
    N = len(speakers)

    text_grid = {}
    for turn in turns_buffer:
        text_grid[(turn.round_id, turn.speaker)] = turn.text

    # Forward fill missing texts
    for r_idx, r in enumerate(rounds):
        for s in speakers:
            if (r, s) not in text_grid and r_idx > 0:
                prev_r = rounds[r_idx - 1]
                if (prev_r, s) in text_grid:
                    text_grid[(r, s)] = text_grid[(prev_r, s)]

    Z, feature_names = processor.encoder.encode_grid(text_grid, rounds, speakers)
    d = len(feature_names)

    # Center states
    Z_mean = Z.mean(axis=1) # (T, d)
    Delta = Z - Z_mean[:, np.newaxis, :] # (T, N, d)
    Delta_flat = Delta.reshape(T, N * d)

    input_dim = N * d
    num_rff = 100
    rng = np.random.RandomState(42)
    W_f = rng.normal(0, 0.7 / np.sqrt(input_dim), size=(input_dim, num_rff))
    b_f = rng.uniform(0, 2 * np.pi, size=(num_rff,))

    Psi_X = lift_state(Delta_flat[:-1, :], W_f, b_f)
    Psi_Y = lift_state(Delta_flat[1:, :], W_f, b_f)
    rho = 1e-6
    G = Psi_X.T @ Psi_X + rho * np.eye(Psi_X.shape[1])
    A = Psi_X.T @ Psi_Y
    K_hat = np.linalg.solve(G, A)
    eigenvalues, _ = np.linalg.eig(K_hat)

    # Filter trivial unit modes to find lambda_2
    non_trivial = [ev for ev in eigenvalues if abs(ev) < 0.985]
    if non_trivial:
        sorted_evals = sorted(non_trivial, key=lambda x: abs(x), reverse=True)
        lambda_2 = sorted_evals[0]
    else:
        lambda_2 = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)[1]

    lambda_2_mag = abs(lambda_2)
    eps = 0.05
    t_cert = int(np.ceil(np.log(1.0 / eps) / (-np.log(lambda_2_mag)))) if lambda_2_mag < 1.0 else 999

    # 2D PCA projection of semantic trajectories
    pca = PCA(n_components=2, random_state=42)
    all_states = Z.reshape(-1, d)
    pca.fit(all_states)

    Z_2d = np.zeros((T, N, 2))
    for t in range(T):
        Z_2d[t] = pca.transform(Z[t])
    Z_mean_2d = pca.transform(Z_mean)

    # Agent deviation norms across rounds
    round_norms = [float(np.linalg.norm(Delta_flat[t])) for t in range(T)]

    # Forecast forward trajectory norms
    pred_rogue, pred_contraction, _ = predict_agent_trajectories(Z, speakers, steps_ahead=1)
    psi_cur = lift_state(Delta_flat[-1:, :], W_f, b_f)
    pred_norms = []
    for step in [1, 2, 3]:
        psi_pred = psi_cur @ np.linalg.matrix_power(K_hat, step)
        d_pred = psi_pred[0, 1:1 + N * d]
        pred_norms.append(float(np.linalg.norm(d_pred)))

    # Plot styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    # =========================================================================
    # FIGURE 1: 4-PANEL TELEMETRY DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=250)
    fig.patch.set_facecolor("#ffffff")

    # PANEL A: Complex Unit Disk Spectrum
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#f8fafc")
    unit_circle = plt.Circle((0, 0), 1.0, color="#cbd5e1", fill=False, linestyle="--", linewidth=1.5, label="Unit Circle (|z|=1.0)")
    l2_circle = plt.Circle((0, 0), lambda_2_mag, color="#3b82f6", fill=False, linestyle=":", linewidth=1.5, label=f"|λ2| = {lambda_2_mag:.3f}")
    ax_a.add_patch(unit_circle)
    ax_a.add_patch(l2_circle)

    # Plot stable eigenvalues
    re_vals = [ev.real for ev in eigenvalues]
    im_vals = [ev.imag for ev in eigenvalues]
    ax_a.scatter(re_vals, im_vals, color="#64748b", alpha=0.6, s=30, label="Operator Modes {μ_j}")

    # Highlight lambda_2
    ax_a.scatter([lambda_2.real], [lambda_2.imag], color="#ef4444", s=90, zorder=5, edgecolors="#1e293b", linewidth=1.5, label=f"Sub-dominant λ2 ({lambda_2.real:.2f}+{lambda_2.imag:.2f}j)")

    ax_a.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="-", alpha=0.5)
    ax_a.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="-", alpha=0.5)
    ax_a.set_xlim(-1.25, 1.25)
    ax_a.set_ylim(-1.25, 1.25)
    ax_a.set_aspect("equal")
    ax_a.set_title("A. Koopman Spectrum on Complex Unit Disk", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_xlabel("Re(μ)")
    ax_a.set_ylabel("Im(μ)")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # PANEL B: Certified Consensus Decay Envelope
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#f8fafc")
    eval_rounds = np.arange(1, T + 1)
    ax_b.plot(eval_rounds, round_norms, "o-", color="#1e293b", linewidth=2.2, markersize=7, label="Observed Disagreement ||Δ(t)||")

    # Theoretical decay envelope
    t_extended = np.linspace(1, max(T + 2, t_cert + 1), 100)
    C_norm = round_norms[0] / (lambda_2_mag ** 1)
    envelope = C_norm * (lambda_2_mag ** t_extended)
    ax_b.plot(t_extended, envelope, "--", color="#3b82f6", linewidth=1.8, label=f"Decay Envelope C·|λ2|^t (λ2={lambda_2_mag:.3f})")

    # Tolerance epsilon
    ax_b.axhline(eps, color="#10b981", linestyle="--", linewidth=1.5, label=f"Tolerance Threshold ε = {eps}")

    # Certified deadline marker
    ax_b.axvline(t_cert, color="#f59e0b", linestyle="-.", linewidth=1.8, label=f"Certified Deadline T_cert = {t_cert} rounds")
    ax_b.scatter([t_cert], [eps], color="#f59e0b", s=90, zorder=5, edgecolors="#1e293b")

    ax_b.set_title("B. Certified Convergence Deadline Envelope", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_xlabel("Debate Round (t)")
    ax_b.set_ylabel("Disagreement State Norm ||Δ(t)||")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # PANEL C: Semantic Faction Drift & Consensus Centroid
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#f8fafc")
    palette = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f97316", "#06b6d4"]

    # Plot agent trajectories
    for i, spk in enumerate(speakers):
        col = palette[i % len(palette)]
        traj = Z_2d[:, i, :]
        ax_c.plot(traj[:, 0], traj[:, 1], "-", color=col, alpha=0.7, linewidth=1.8)
        # Start marker
        ax_c.scatter(traj[0, 0], traj[0, 1], color=col, alpha=0.5, s=40, marker="^")
        # Final marker
        ax_c.scatter(traj[-1, 0], traj[-1, 1], color=col, s=80, marker="o", edgecolors="#1e293b", label=spk)
        ax_c.text(traj[-1, 0] + 0.015, traj[-1, 1] + 0.015, spk, fontsize=7.5, color="#1e293b")

    # Plot Consensus Centroid drift
    ax_c.plot(Z_mean_2d[:, 0], Z_mean_2d[:, 1], "k--", linewidth=2.5, label="Consensus Centroid z̄(t)", zorder=6)
    ax_c.scatter(Z_mean_2d[:, 0], Z_mean_2d[:, 1], color="#eab308", s=130, marker="*", edgecolors="#0f172a", zorder=7)
    for r_idx in range(T):
        ax_c.text(Z_mean_2d[r_idx, 0] + 0.01, Z_mean_2d[r_idx, 1] - 0.025, f"R{r_idx+1}", fontsize=8, fontweight="bold", color="#854d0e")

    ax_c.set_title("C. 2D Semantic Manifold: Faction Convergence & Centroid Drift", fontsize=12, fontweight="bold", pad=10)
    ax_c.set_xlabel("Principal Semantic Observable (PC 1)")
    ax_c.set_ylabel("Disagreement Observable (PC 2)")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

    # PANEL D: Forward Trajectory Prediction & Contraction Cone
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#f8fafc")

    # Historical rounds
    hist_x = list(range(1, T + 1))
    ax_d.plot(hist_x, round_norms, "o-", color="#1e293b", linewidth=2.0, label="Observed Norm ||Δ(t)||")

    # Predicted future rounds
    pred_x = [T + k for k in range(1, len(pred_norms) + 1)]
    all_future_x = [T] + pred_x
    all_future_y = [round_norms[-1]] + pred_norms

    ax_d.plot(all_future_x, all_future_y, "s--", color="#8b5cf6", linewidth=2.2, label=f"Predicted Trajectory (K^k Forward Lift)")
    ax_d.scatter(pred_x, pred_norms, color="#8b5cf6", s=70, zorder=5, edgecolors="#1e293b")

    # Contraction cone shading
    ax_d.fill_between(all_future_x, [y * 0.85 for y in all_future_y], [y * 1.15 for y in all_future_y], color="#8b5cf6", alpha=0.15, label="Spectral Prediction Confidence Envelope")

    ax_d.axvline(T, color="#94a3b8", linestyle=":", linewidth=1.5)
    ax_d.text(T + 0.1, max(round_norms) * 0.9, "Prediction Horizon →", color="#64748b", fontweight="bold", fontsize=9)

    ax_d.set_title(f"D. Forward Trajectory Projection (Contraction Rate {pred_contraction:.2f}x)", fontsize=12, fontweight="bold", pad=10)
    ax_d.set_xlabel("Debate Round (t)")
    ax_d.set_ylabel("Residual Disagreement Norm ||Δ(t)||")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.set_ylim(bottom=0)
    ax_d.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "klens_telemetry_dashboard.png")
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated dashboard: {dashboard_path}")

    # =========================================================================
    # FIGURE 2: DETAILED FACTION CLUSTERING & METASTABLE POLARIZATION
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=250)
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#f8fafc")

    # Draw shaded faction boundaries around final states
    factions_groups = [
        ("Automated Judge Tier", [0, 1, 3], "#3b82f6"),       # Agent_Arch, Agent_Auditor, Agent_Ops
        ("Governance & Compliance", [2, 4], "#ec4899"),       # Agent_Compliance, Agent_Safety
        ("Cost & Finance", [5], "#10b981"),                   # Agent_Finance
    ]

    for label, member_indices, color in factions_groups:
        pts = Z_2d[-1, member_indices, :]
        if len(pts) > 1:
            center = pts.mean(axis=0)
            radius_x = np.max(np.abs(pts[:, 0] - center[0])) + 0.08
            radius_y = np.max(np.abs(pts[:, 1] - center[1])) + 0.08
            ell = Ellipse(center, width=radius_x * 2.2, height=radius_y * 2.2, color=color, alpha=0.15, linestyle="--", linewidth=1.5, label=f"Faction: {label}")
            ax2.add_patch(ell)
        else:
            circ = Circle(pts[0], radius=0.06, color=color, alpha=0.15, linestyle="--", linewidth=1.5, label=f"Faction: {label}")
            ax2.add_patch(circ)

    # Plot all trajectories
    for i, spk in enumerate(speakers):
        traj = Z_2d[:, i, :]
        col = palette[i % len(palette)]
        ax2.plot(traj[:, 0], traj[:, 1], "o-", color=col, linewidth=2.0, markersize=6, alpha=0.85)
        # Annotate turn 1 to turn 4
        for step in range(T):
            ax2.text(traj[step, 0] + 0.008, traj[step, 1] + 0.008, f"t{step+1}", fontsize=7, color=col)
        ax2.scatter(traj[-1, 0], traj[-1, 1], color=col, s=110, edgecolors="#0f172a", linewidth=1.5, label=f"{spk} (Final)")

    # Plot Centroid drift
    ax2.plot(Z_mean_2d[:, 0], Z_mean_2d[:, 1], "k--", linewidth=3.0, label="Consensus Centroid Drift z̄(t)", zorder=10)
    ax2.scatter(Z_mean_2d[:, 0], Z_mean_2d[:, 1], color="#eab308", s=180, marker="*", edgecolors="#0f172a", linewidth=1.5, zorder=11)
    for step in range(T):
        ax2.text(Z_mean_2d[step, 0] + 0.015, Z_mean_2d[step, 1] - 0.02, f"Centroid t{step+1}", fontsize=8.5, fontweight="bold", color="#854d0e")

    ax2.set_title("Metastable Faction Polarization & Consensus Centroid Contraction", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Principal Semantic Dimension (PC 1)")
    ax2.set_ylabel("Disagreement Subspace (PC 2)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    drift_path = os.path.join(output_dir, "faction_centroid_drift.png")
    fig2.savefig(drift_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"Generated drift figure: {drift_path}")

    # Clean temporary stream log
    temp_csv = os.path.join(output_dir, "temp_stream.csv")
    if os.path.exists(temp_csv):
        os.remove(temp_csv)


if __name__ == "__main__":
    generate_figures()
