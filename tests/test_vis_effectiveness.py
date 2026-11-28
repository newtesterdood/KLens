import json
import os
import re
import tempfile
import numpy as np
import pytest

from klens.cli import main
from klens.stream import LiveTranscriptStreamProcessor
from klens.vis import (
    _extract_cluster_events_and_flow,
    _extract_concept_fields_3d,
    _extract_narrative_dynamics,
    _extract_spectral_rigor_data,
    _project_silence_aware_pca,
    export_interactive_timeline_html,
    render_concept_fields_dashboard,
    render_concept_fields_video,
    render_spectral_dashboard,
    render_spectral_rigor_dashboard,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def _load_processor(sample_filename: str) -> LiveTranscriptStreamProcessor:
    path = os.path.join(SAMPLES_DIR, sample_filename)
    proc = LiveTranscriptStreamProcessor(auto_init_csv=False)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            turns = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            turns = data if isinstance(data, list) else data.get("turns", [])
    rounds = sorted({int(t.get("round", t.get("round_id", 1))) for t in turns})
    for t in turns:
        proc.ingest_turn(int(t.get("round", t.get("round_id", 1))), str(t["speaker"]), str(t["text"]))
    for r in rounds:
        proc.process_round_event(r)
    return proc


def test_explanatory_anchor_conceder_false_consensus():
    speakers = ["Anchor", "ConcederA", "ConcederB", "FalseCons", "Dissenter"]
    rounds = [1, 2, 3, 4]
    features = ["safety", "audit", "latency", "cost", "privacy", "veto"]
    # Anchor stays fixed on (safety, audit); Conceders migrate from latency/cost -> Anchor's frame;
    # FalseCons shares safety/audit vocab (>0.05) but holds huge orthogonal privacy/veto weight (sin_theta >= 0.80);
    # Dissenter stays purely on veto.
    Z = np.array([
        [
            [0.85, 0.52, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.95, 0.30, 0.0, 0.0],
            [0.0, 0.0, 0.20, 0.98, 0.0, 0.0],
            [0.08, 0.08, 0.0, 0.0, 0.99, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.85, 0.52, 0.0, 0.0, 0.0, 0.0],
            [0.45, 0.25, 0.70, 0.15, 0.0, 0.0],
            [0.45, 0.25, 0.10, 0.75, 0.0, 0.0],
            [0.08, 0.08, 0.0, 0.0, 0.99, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.85, 0.52, 0.0, 0.0, 0.0, 0.0],
            [0.75, 0.45, 0.20, 0.05, 0.0, 0.0],
            [0.75, 0.45, 0.05, 0.20, 0.0, 0.0],
            [0.08, 0.08, 0.0, 0.0, 0.99, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.85, 0.52, 0.0, 0.0, 0.0, 0.0],
            [0.84, 0.51, 0.05, 0.0, 0.0, 0.0],
            [0.84, 0.51, 0.0, 0.05, 0.0, 0.0],
            [0.08, 0.08, 0.0, 0.0, 0.99, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    ], dtype=float)

    _, Z_disp, _, act, spk_m = _project_silence_aware_pca(Z)
    rc, _ = _extract_cluster_events_and_flow(
        source=Z, Z=Z, Z_display_2d=Z_disp, active_mask=act, has_spoken_mask=spk_m,
        speakers=speakers, rounds=rounds, feature_names=features, rogue_idx=4,
    )
    nar = _extract_narrative_dynamics(
        source=Z, Z=Z, active_mask=act, has_spoken_mask=spk_m,
        speakers=speakers, rounds=rounds, feature_names=features, rogue_idx=4, round_clusters=rc,
    )
    prof = {p["speaker"]: p for p in nar["speaker_profiles"]}

    assert prof["Anchor"]["archetype"] == "Framing Anchor"
    assert prof["Anchor"]["concession"] == 0.0
    assert prof["Anchor"]["influence_pct"] > prof["ConcederA"]["influence_pct"]
    assert prof["Dissenter"]["archetype"] == "Unreconciled Dissenter"
    assert prof["FalseCons"]["false_consensus"] is True
    assert prof["FalseCons"]["vocab_overlap_pct"] >= 45.0
    assert prof["FalseCons"]["sin_theta_pct"] >= 80.0
    assert any(
        tr["from_speaker"] == "Anchor" and tr["to_speaker"] in {"ConcederA", "ConcederB"}
        for tr in nar["concept_transfers"]
    )

    rigor = _extract_spectral_rigor_data(Z, speakers, rounds)
    order = rigor["ordered_speakers"]
    dmat = np.asarray(rigor["delta_cosine_matrix"])
    i_anc = order.index("Anchor")
    i_ca = order.index("ConcederA")
    i_dis = order.index("Dissenter")
    assert dmat[i_anc, i_ca] > 0.80
    assert abs(dmat[i_anc, i_dis]) < 0.05


def test_explanatory_polarization_merger_and_pivot():
    speakers = ["A1", "A2", "B1", "B2"]
    rounds = [1, 2, 3, 4]
    features = ["shared", "camp_a", "camp_b", "pivot_dim"]

    # 1. Polarization: start shared -> split into orthogonal Camp A vs Camp B
    Z_pol = np.array([
        [[0.95, 0.2, 0.1, 0.0], [0.95, 0.1, 0.2, 0.0], [0.95, 0.1, 0.2, 0.0], [0.95, 0.2, 0.1, 0.0]],
        [[0.70, 0.7, 0.0, 0.0], [0.70, 0.7, 0.0, 0.0], [0.70, 0.0, 0.7, 0.0], [0.70, 0.0, 0.7, 0.0]],
        [[0.20, 0.98, 0.0, 0.0], [0.20, 0.98, 0.0, 0.0], [0.20, 0.0, 0.98, 0.0], [0.20, 0.0, 0.98, 0.0]],
        [[0.05, 0.99, 0.0, 0.0], [0.05, 0.99, 0.0, 0.0], [0.05, 0.0, 0.99, 0.0], [0.05, 0.0, 0.99, 0.0]],
    ], dtype=float)
    rig_pol = _extract_spectral_rigor_data(Z_pol, speakers, rounds)
    assert rig_pol["round_metrics"][-1]["k_star"] > rig_pol["round_metrics"][0]["k_star"] + 0.5
    assert len(rig_pol["top_polarized_pairs"]) >= 1
    assert rig_pol["top_polarized_pairs"][0]["delta_cos"] < -0.5

    # 2. Merger + Agenda Pivot: two camps merge in R3 -> rotate 90 deg to pivot_dim in R4
    Z_mer = np.array([
        [[0.1, 0.99, 0.0, 0.0], [0.1, 0.99, 0.0, 0.0], [0.1, 0.0, 0.99, 0.0], [0.1, 0.0, 0.99, 0.0]],
        [[0.3, 0.90, 0.2, 0.0], [0.3, 0.90, 0.2, 0.0], [0.3, 0.2, 0.90, 0.0], [0.3, 0.2, 0.90, 0.0]],
        [[0.95, 0.25, 0.25, 0.0], [0.95, 0.25, 0.25, 0.0], [0.95, 0.25, 0.25, 0.0], [0.95, 0.25, 0.25, 0.0]],
        [[0.95, 0.25, 0.25, 0.95], [0.95, 0.25, 0.25, 0.95], [0.95, 0.25, 0.25, 0.95], [0.95, 0.25, 0.25, 0.95]],
    ], dtype=float)
    _, Z_m2d, _, act_m, spk_m = _project_silence_aware_pca(Z_mer)
    rc_m, ev_m = _extract_cluster_events_and_flow(
        source=Z_mer, Z=Z_mer, Z_display_2d=Z_m2d, active_mask=act_m, has_spoken_mask=spk_m,
        speakers=speakers, rounds=rounds, feature_names=features,
    )
    nar_m = _extract_narrative_dynamics(
        source=Z_mer, Z=Z_mer, active_mask=act_m, has_spoken_mask=spk_m,
        speakers=speakers, rounds=rounds, feature_names=features, rogue_idx=None, round_clusters=rc_m,
    )
    rig_mer = _extract_spectral_rigor_data(Z_mer, speakers, rounds)
    assert any(ev["event_type"] == "Collection" for ev in ev_m)
    assert rig_mer["round_metrics"][-1]["k_star"] < rig_mer["round_metrics"][0]["k_star"] - 0.5
    assert len(rig_mer["top_converged_pairs"]) >= 1
    assert rig_mer["top_converged_pairs"][0]["delta_cos"] > 0.5
    assert nar_m["phase_rounds"][3]["is_pivot"] is True
    assert nar_m["phase_rounds"][3]["rotation_deg"] >= 45.0


def test_rigor_walk_forward_and_stability_discrimination():
    rng = np.random.RandomState(42)
    base = rng.randn(4, 5)
    base -= base.mean(axis=0, keepdims=True)
    speakers = ["A", "B", "C", "D"]
    rounds = list(range(1, 8))

    Z_con = np.array([(0.6 ** t) * base for t in range(7)])
    Z_div = np.array([(1.25 ** t) * base for t in range(7)])
    Z_noi = rng.randn(7, 4, 5)

    r_con = _extract_spectral_rigor_data(Z_con, speakers, rounds, num_rff=30)
    r_div = _extract_spectral_rigor_data(Z_div, speakers, rounds, num_rff=30)
    r_noi = _extract_spectral_rigor_data(Z_noi, speakers, rounds, num_rff=30)

    assert r_con["round_metrics"][-1]["regime"] == "Contractive"
    assert r_con["walk_forward_error"] is not None and r_con["walk_forward_error"] < 0.2
    assert r_con["walk_forward_confidence"] > 0.8
    assert r_con["walk_forward_error"] < r_noi["walk_forward_error"] * 0.4
    assert r_con["walk_forward_confidence"] > r_noi["walk_forward_confidence"] + 0.3

    assert r_div["round_metrics"][-1]["regime"] == "Divergent"
    assert r_div["round_metrics"][-1]["lambda_2"] > 1.0
    assert r_div["round_metrics"][-1]["t_cert"] == 999
    sub_eigs = [e for e in r_div["terminal_eigenvalues"] if e["is_subdominant"]]
    assert len(sub_eigs) == 1 and sub_eigs[0]["mag"] > 1.0


def test_real_corpora_explanatory_contrast():
    proc_adv = _load_processor("real_llm_negotiation_targeted_adv.json")
    proc_coop = _load_processor("real_llm_negotiation_gpt4_coop.json")

    rig_adv = _extract_spectral_rigor_data(proc_adv.latest_Z, proc_adv.latest_speakers, proc_adv.latest_rounds)
    rig_coop = _extract_spectral_rigor_data(proc_coop.latest_Z, proc_coop.latest_speakers, proc_coop.latest_rounds)

    assert rig_adv["orthogonal_persistence"].get("Environmental League", 0) >= 5
    max_coop_pers = max(rig_coop["orthogonal_persistence"].values())
    assert rig_adv["orthogonal_persistence"]["Environmental League"] > max_coop_pers

    proc_ww = _load_processor("real_llm_werewolf_gpt5_vs_gemini.json")
    _, Z_w2d, _, act_w, spk_w = _project_silence_aware_pca(proc_ww.latest_Z)
    rc_w, _ = _extract_cluster_events_and_flow(
        source=proc_ww, Z=proc_ww.latest_Z, Z_display_2d=Z_w2d, active_mask=act_w, has_spoken_mask=spk_w,
        speakers=proc_ww.latest_speakers, rounds=proc_ww.latest_rounds,
        feature_names=proc_ww.latest_feature_names,
    )
    final_clusters = rc_w[-1]
    assert any({"Eve", "Grace", "Liam"}.issubset(set(c["members"])) for c in final_clusters)


def test_silence_projection_honesty_and_edge_degeneracies():
    # Edge cases: d=1, N=2, silent round, identical states -> finite coordinates & no crash
    for Z_edge in [
        np.array([[[1.0], [2.0]], [[1.5], [0.0]], [[1.8], [2.2]]]),
        np.ones((3, 3, 4)),
        np.zeros((3, 3, 4)),
    ]:
        Z_2d, Z_disp, Z_m2d, act, spk_m = _project_silence_aware_pca(Z_edge)
        assert np.all(np.isfinite(Z_2d))
        assert np.all(np.isfinite(Z_disp))
        assert np.all(np.isfinite(Z_m2d))
        rig = _extract_spectral_rigor_data(Z_edge, [f"A{i}" for i in range(Z_edge.shape[1])], [1, 2, 3])
        assert all(np.isfinite(r["disagreement_norm"]) for r in rig["round_metrics"])


def test_all_eight_styles_render_legible_nonblank_figures():
    import matplotlib.image as mpimg

    proc = _load_processor("debate_bush_clinton_perot_1992.json")
    expected_min_axes = {
        "telemetry": 4,
        "diagnostic": 4,
        "evolution": 5,
        "shortcut": 1,
        "paths": 4,
        "flow": 2,
        "narrative": 4,
        "rigor": 4,
        "fields": 4,
        "envelopes": 3,
        "footprints": 6,
        "ribbons": 2,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        for style, min_axes in expected_min_axes.items():
            out_png = os.path.join(tmpdir, f"bush_{style}.png")
            fig = render_spectral_dashboard(proc, style=style, output_path=out_png, show=False, dpi=120)
            assert fig is not None
            assert len(fig.axes) >= min_axes, f"Style {style} had {len(fig.axes)} axes (< {min_axes})"
            assert os.path.exists(out_png) and os.path.getsize(out_png) > 10000

            # Verify PNG pixels are rich, non-blank visual imagery
            img = mpimg.imread(out_png)
            assert img.ndim == 3 and img.shape[0] >= 300 and img.shape[1] >= 400
            assert float(np.std(img[:, :, :3])) > 0.05

            # Verify axes have finite bounds and titles
            titled_axes = [ax for ax in fig.axes if ax.get_title().strip()]
            assert len(titled_axes) >= 1, f"Style {style} has no panel titles"
            for ax in fig.axes:
                xl = ax.get_xlim()
                yl = ax.get_ylim()
                assert all(np.isfinite(v) for v in (*xl, *yl))


def test_interactive_html_payload_schema_and_rigor_tab():
    proc = _load_processor("real_llm_negotiation_targeted_adv.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = os.path.join(tmpdir, "adv_interactive.html")
        export_interactive_timeline_html(proc, output_path=out_html, title="Rigor HTML Test")
        with open(out_html, "r", encoding="utf-8") as f:
            html_txt = f.read()

        for expected_ui in [
            "2D Spatial Map",
            "Alluvial Faction-Flow",
            "Narrative Insights",
            "Spectral Rigor",
            "3D Concept Fields",
            "Positional Shapes",
            "renderRigorCanvas",
            "renderRigorPanel",
            "renderFields3DCanvas",
            "renderFieldsPanel",
            "renderPositions3DCanvas",
            "renderPositionsPanel",
        ]:
            assert expected_ui in html_txt

        m = re.search(r"const DATA = (\{.*?\});\nconst canvas =", html_txt, re.S)
        assert m is not None, "Could not locate embedded DATA JSON payload"
        payload = json.loads(m.group(1))

        assert len(payload["rounds"]) == len(proc.latest_rounds)
        assert payload["speakers"] == proc.latest_speakers
        assert "narrative" in payload and "rigor" in payload and "fields3d" in payload and "positions" in payload
        assert len(payload["fields3d"]["fields"]) >= 2
        assert len(payload["positions"]["speaker_profiles"]) == len(proc.latest_speakers)
        rig = payload["rigor"]
        assert len(rig["round_metrics"]) == len(proc.latest_rounds)
        assert len(rig["walk_forward_steps"]) == len(proc.latest_rounds) - 2
        assert len(rig["delta_cosine_matrix"]) == len(proc.latest_speakers)
        assert rig["orthogonal_persistence"]["Environmental League"] >= 5


def test_cli_rigor_style_run_and_plot(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "real_llm_werewolf_gpt5_vs_gemini.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "ww_rigor.csv")
        out_png1 = os.path.join(tmpdir, "ww_run_rigor.png")
        out_png2 = os.path.join(tmpdir, "ww_plot_rigor.png")
        out_html = os.path.join(tmpdir, "ww_rigor.html")

        assert main(["run", sample_file, "--output", out_csv, "--style", "rigor", "--plot", out_png1]) == 0
        assert os.path.exists(out_csv) and os.path.getsize(out_png1) > 10000

        assert main(["plot", sample_file, "--style", "rigor", "--output", out_png2, "--html", out_html]) == 0
        assert os.path.getsize(out_png2) > 10000 and os.path.getsize(out_html) > 10000

        # Direct API on raw numpy array
        rng = np.random.RandomState(3)
        Z = rng.rand(5, 4, 6)
        out_raw = os.path.join(tmpdir, "raw_rigor.png")
        fig = render_spectral_rigor_dashboard(Z, speakers=["A", "B", "C", "D"], output_path=out_raw)
        assert fig is not None and os.path.getsize(out_raw) > 8000


def test_concept_fields_planar_distinction_and_relations():
    # Plant 4 orthogonal concept groups: 2 large-variance floor axes (PC1, PC2) + 2 distinct PC3 strata (+z vs -z)
    # whose 2D floor projections sit near (0, 0), proving PC3 separates 2D shadow overlap!
    speakers = ["FloorPosX", "FloorNegX", "FloorPosY", "FloorNegY", "UpperStratum", "LowerStratum"]
    rounds = [1, 2, 3]
    features = ["taxes", "cuts", "healthcare", "jobs", "lobbyists", "gridlock", "tariffs", "nafta"]
    Z = np.zeros((3, 6, 8), dtype=float)
    for t in range(3):
        # High-variance floor axes along dims (0,1) vs (2,3)
        Z[t, 0] = [2.4, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        Z[t, 1] = [0.1, 0.1, 2.4, 2.2, 0.0, 0.0, 0.0, 0.0]
        Z[t, 2] = [2.1, 2.0, 2.1, 2.0, 0.0, 0.0, 0.0, 0.0]
        Z[t, 3] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
        # Medium-variance 3rd axis splitting UpperStratum (dims 4,5) vs LowerStratum (dims 6,7)
        Z[t, 4] = [1.1, 1.0, 1.1, 1.0, 1.5, 1.4, 0.0, 0.0]
        Z[t, 5] = [1.1, 1.0, 1.1, 1.0, 0.0, 0.0, 1.5, 1.4]

    fdata = _extract_concept_fields_3d(Z, speakers, rounds, features, max_fields=4, grid_res=40)
    s_end = {s["speaker"]: s for s in fdata["speaker_3d_rounds"][-1]}

    # UpperStratum and LowerStratum have nearly identical 2D floor coordinates (x, y)
    assert abs(s_end["UpperStratum"]["x"] - s_end["LowerStratum"]["x"]) < 0.05
    assert abs(s_end["UpperStratum"]["y"] - s_end["LowerStratum"]["y"]) < 0.05
    # Yet PC3 cleanly separates them onto opposite elevation strata (+z vs -z)!
    assert abs(s_end["UpperStratum"]["z"] - s_end["LowerStratum"]["z"]) > 1.2
    assert s_end["UpperStratum"]["z"] * s_end["LowerStratum"]["z"] < 0.0

    # Verify CLI --style fields and direct API
    sample_file = os.path.join(SAMPLES_DIR, "debate_bush_clinton_perot_1992.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "bush_fields.png")
        assert main(["plot", sample_file, "--style", "fields", "--output", out_png]) == 0
        assert os.path.exists(out_png) and os.path.getsize(out_png) > 12000

        out_raw = os.path.join(tmpdir, "raw_fields.png")
        fig = render_concept_fields_dashboard(Z, speakers=speakers, output_path=out_raw)
        assert fig is not None and os.path.getsize(out_raw) > 10000

        out_mp4 = os.path.join(tmpdir, "bush_3d_tour.mp4")
        saved_vid = render_concept_fields_video(
            Z, speakers=speakers, output_path=out_mp4, fps=10, duration_sec=2.4, width=800, height=460
        )
        assert saved_vid == out_mp4 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 15000
