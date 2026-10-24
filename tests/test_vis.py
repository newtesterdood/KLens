import os
import tempfile
import numpy as np
import pytest

from klens import LiveTranscriptStreamProcessor
from klens.cli import main
from klens.vis import plot_from_transcript_file, render_spectral_dashboard

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
