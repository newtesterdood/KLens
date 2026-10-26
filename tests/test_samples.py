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


def test_sample_scotus_bruen(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "scotus_bruen_oral_argument.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_scotus.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 12" in out
        assert "Round  6" in out
        assert "Second Amendment" in out


def test_sample_fomc_monetary_policy(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "fomc_monetary_policy_aug2011.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_fomc.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 25" in out
        assert "Round  5" in out
        assert "Monetary Policy" in out


def test_sample_real_llm_debates(capsys):
    real_samples = [
        ("real_llm_debate_des_moines.json", "Des Moines"),
        ("real_llm_debate_bitcoin.json", "Cryptocurrency Expert"),
        ("real_llm_debate_boolean_logic.json", "Computer Science Theorist"),
        ("real_llm_debate_bee_sting.json", "Entomologist"),
    ]
    for filename, expected_match in real_samples:
        sample_file = os.path.join(SAMPLES_DIR, filename)
        assert os.path.exists(sample_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, f"out_{filename}.csv")
            ret = main(["run", sample_file, "--output", out_csv])
            assert ret == 0
            assert os.path.exists(out_csv)

            out = capsys.readouterr().out
            assert "Agents: 3" in out
            assert "Round  7" in out
            assert expected_match in out


def test_sample_fda_adcom(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "fda_aducanumab_adcom.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_adcom.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 17" in out
        assert "Round  5" in out


def test_sample_delaware_merger(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "delaware_merger_deliberation.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_delaware.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 14" in out
        assert "Round  5" in out
        assert "Delaware Law" in out


def test_sample_puc_infrastructure(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "puc_infrastructure_hearing.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_puc.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 14" in out
        assert "Round  5" in out
        assert "Battery Storage" in out or "Decision 22-12-056" in out


def test_sample_city_council_policy(capsys):
    sample_file = os.path.join(SAMPLES_DIR, "city_council_policy_meeting.json")
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "out_council.csv")
        ret = main(["run", sample_file, "--output", out_csv])
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        assert "Agents: 14" in out
        assert "Round  5" in out
        assert "Parking and Density" in out or "Vote Aye" in out


