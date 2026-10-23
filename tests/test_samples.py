import os
import tempfile
import pytest

from klens.cli import main

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_sample_llm_safety_consensus(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "llm_safety_consensus.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_safety.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 6" in out
        assert "Round  4" in out
        assert "✓ Converging" in out


def test_sample_boardroom_deadlock(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "boardroom_deadlock_polarization.jsonl")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_deadlock.csv")
        ret = main(["run", sample_file, "--output", out_csv, "--t-cert-limit", "40"])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 8" in out
        assert "Round  4" in out


def test_sample_swarm_deliberation_hierarchical(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "swarm_deliberation_24agents.jsonl")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_swarm.csv")
        ret = main([
            "run",
            sample_file,
            "--hierarchical",
            "--max-factions",
            "4",
            "--solver",
            "auto",
            "--output",
            out_csv,
        ])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 24" in out
        assert "Hierarchical K=4" in out
