import os
import tempfile
import numpy as np
import pytest

from klens import LiveTranscriptStreamProcessor
from klens.cli import main
from klens.vis import (
    _extract_cluster_events_and_flow,
    _extract_narrative_dynamics,
    _project_silence_aware_pca,
    export_interactive_timeline_html,
    plot_from_transcript_file,
    render_spectral_dashboard,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_render_spectral_dashboard_with_numpy():
    np.random.seed(42)
    T, N, d = 3, 4, 5
    Z = np.random.randn(T, N, d)
    speakers = [f"Agent_{i}" for i in range(N)]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "dashboard_test.png")
        fig = render_spectral_dashboard(
            source=Z,
            speakers=speakers,
            output_path=out_png,
            show=False,
            title="Unit Test Dashboard",
        )
        assert fig is not None
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000


def test_plot_from_transcript_file():
    sample_file = os.path.join(SAMPLES_DIR, "llm_safety_consensus.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "plot_out.png")
        proc, saved_path = plot_from_transcript_file(
            transcript_path=sample_file,
            output_path=out_png,
            show=False,
        )
        assert proc is not None
        assert proc.latest_metrics is not None
        assert saved_path == out_png
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000


def test_cli_plot_command(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "llm_safety_consensus.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "cli_plot.png")
        ret = main(["plot", sample_file, "--output", out_png])
        assert ret == 0
        assert os.path.exists(out_png)

        out = capsys.readouterr().out
        assert "Spectral telemetry dashboard saved" in out


def test_cli_run_with_plot_flag(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "boardroom_deadlock_polarization.jsonl")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "cli_run_plot.png")
        out_csv = os.path.join(tmpdir, "cli_run.csv")
        ret = main(["run", sample_file, "--output", out_csv, "--plot", out_png])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.exists(out_csv)


def test_cli_demo_with_plot_flag(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "demo_plot.png")
        out_csv = os.path.join(tmpdir, "demo.csv")
        ret = main(["demo", "--output", out_csv, "--plot", out_png])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.exists(out_csv)


def test_render_temporal_evolution_grid_and_html():
    sample_file = os.path.join(SAMPLES_DIR, "delaware_merger_deliberation.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "merger_evo.png")
        out_html = os.path.join(tmpdir, "merger_timeline.html")
        ret = main(["plot", sample_file, "--style", "evolution", "--output", out_png, "--html", out_html])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 10000
        assert os.path.exists(out_html)
        assert os.path.getsize(out_html) > 5000


def test_silence_aware_pca_and_html_flags():
    Z = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ])
    Z_2d, Z_display_2d, _, active_mask, has_spoken_mask = _project_silence_aware_pca(Z)
    assert active_mask.tolist() == [[True, True, False], [True, False, True]]
    assert has_spoken_mask.tolist() == [[True, True, False], [True, True, True]]
    assert np.allclose(Z_display_2d[1, 1], Z_2d[0, 1])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = os.path.join(tmpdir, "silent_check.html")
        export_interactive_timeline_html(Z, speakers=["Active", "EarlySilent", "LateJoin"], output_path=out_html)
        with open(out_html, "r", encoding="utf-8") as f:
            html_txt = f.read()
        assert "Silent (Carried Forward)" in html_txt
        assert '"active": false' in html_txt
        assert '"has_spoken": false' in html_txt


def test_render_trajectory_paths_dashboard():
    sample_file = os.path.join(SAMPLES_DIR, "debate_bush_clinton_perot_1992.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "bush_perot_paths.png")
        out_html = os.path.join(tmpdir, "bush_perot_paths.html")
        ret = main(["plot", sample_file, "--style", "paths", "--output", out_png, "--html", out_html])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 10000
        assert os.path.exists(out_html)
        with open(out_html, "r", encoding="utf-8") as f:
            html_txt = f.read()
        assert "concept_pins" in html_txt
        assert "claimsPanel" in html_txt
        assert "Center of Disagreement" in html_txt
        assert "four-trillion-dollar debt" in html_txt
        assert "inflection_events" in html_txt
        assert "wedge_terms" in html_txt
        assert "Cluster Lens" in html_txt
        assert "Alluvial Faction-Flow" in html_txt
        assert "highlightClaimTerms" in html_txt


def test_render_concept_flow_dashboard():
    sample_file = os.path.join(SAMPLES_DIR, "real_llm_debate_bitcoin.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "bitcoin_flow.png")
        ret = main(["plot", sample_file, "--style", "flow", "--output", out_png])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 10000


def test_render_narrative_dashboard_and_html():
    sample_file = os.path.join(SAMPLES_DIR, "real_llm_debate_bitcoin.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "bitcoin_narrative.png")
        out_html = os.path.join(tmpdir, "bitcoin_narrative.html")
        ret = main(["plot", sample_file, "--style", "narrative", "--output", out_png, "--html", out_html])
        assert ret == 0
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 10000
        assert os.path.exists(out_html)
        with open(out_html, "r", encoding="utf-8") as f:
            html_txt = f.read()
        assert "Narrative Insights" in html_txt
        assert "catalyst_events" in html_txt
        assert "concept_transfers" in html_txt
        assert "phase_rounds" in html_txt
        assert "renderNarrativeCanvas" in html_txt


def test_vis_new_styles_accept_raw_numpy():
    rng = np.random.RandomState(11)
    Z = rng.rand(4, 5, 8)
    speakers = [f"Node_{i}" for i in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        for style in ("paths", "flow", "narrative"):
            out_png = os.path.join(tmpdir, f"raw_{style}.png")
            fig = render_spectral_dashboard(Z, speakers=speakers, style=style, output_path=out_png)
            assert fig is not None
            assert os.path.exists(out_png)
            assert os.path.getsize(out_png) > 5000


def test_narrative_dynamics_behavioral_invariants():
    speakers = ["Anchor", "FollowerA", "FollowerB", "Dissenter"]
    rounds = [1, 2, 3]
    features = ["safety", "cost", "latency", "privacy", "audit"]
    Z = np.array([
        [
            [0.9, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.8],
        ],
        [
            [0.9, 0.4, 0.0, 0.0, 0.0],
            [0.6, 0.3, 0.4, 0.0, 0.0],
            [0.6, 0.3, 0.0, 0.4, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.8],
        ],
        [
            [0.9, 0.4, 0.0, 0.0, 0.0],
            [0.85, 0.38, 0.0, 0.0, 0.0],
            [0.85, 0.38, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.8],
        ],
    ], dtype=float)
    _, Z_display_2d, _, active_mask, has_spoken_mask = _project_silence_aware_pca(Z)
    round_clusters, _ = _extract_cluster_events_and_flow(
        source=Z,
        Z=Z,
        Z_display_2d=Z_display_2d,
        active_mask=active_mask,
        has_spoken_mask=has_spoken_mask,
        speakers=speakers,
        rounds=rounds,
        feature_names=features,
        rogue_idx=3,
    )
    nar = _extract_narrative_dynamics(
        source=Z,
        Z=Z,
        active_mask=active_mask,
        has_spoken_mask=has_spoken_mask,
        speakers=speakers,
        rounds=rounds,
        feature_names=features,
        rogue_idx=3,
        round_clusters=round_clusters,
    )
    profiles = {p["speaker"]: p for p in nar["speaker_profiles"]}
    assert profiles["Anchor"]["concession"] == 0.0
    assert profiles["Anchor"]["influence_pct"] > profiles["FollowerA"]["influence_pct"]
    assert profiles["Anchor"]["archetype"] == "Framing Anchor"
    assert profiles["Dissenter"]["archetype"] == "Unreconciled Dissenter"

    adopted_targets = {(tr["from_speaker"], tr["to_speaker"]): tr["terms"] for tr in nar["concept_transfers"]}
    assert any(tr[0] == "Anchor" and tr[1] in {"FollowerA", "FollowerB"} for tr in adopted_targets)

    # Orthogonally rotating centroid trajectory -> agenda pivot >= 45 deg
    Z_piv = np.array([
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.1, 0.0, 0.0], [1.0, 0.0, 0.1, 0.0]],
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.1, 0.0], [1.0, 1.0, 0.0, 0.1]],
    ], dtype=float)
    _, Z_piv_2d, _, act_p, spk_p = _project_silence_aware_pca(Z_piv)
    rc_piv, _ = _extract_cluster_events_and_flow(
        source=Z_piv, Z=Z_piv, Z_display_2d=Z_piv_2d, active_mask=act_p, has_spoken_mask=spk_p,
        speakers=["A", "B", "C"], rounds=[1, 2, 3], feature_names=["t0", "t1", "t2", "t3"],
    )
    nar_piv = _extract_narrative_dynamics(
        source=Z_piv, Z=Z_piv, active_mask=act_p, has_spoken_mask=spk_p,
        speakers=["A", "B", "C"], rounds=[1, 2, 3], feature_names=["t0", "t1", "t2", "t3"],
        rogue_idx=None, round_clusters=rc_piv,
    )
    assert nar_piv["phase_rounds"][2]["rotation_deg"] >= 80.0
    assert nar_piv["phase_rounds"][2]["is_pivot"] is True







