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


def _run_sample_test(
    capsys,
    filename: str,
    expected_substrings: list[str],
    extra_args: list[str] | None = None,
) -> None:
    sample_file = os.path.join(SAMPLES_DIR, filename)
    assert os.path.exists(sample_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, f"out_{filename}.csv")
        args = ["run", sample_file, "--output", out_csv, *(extra_args or [])]
        ret = main(args)
        assert ret == 0
        assert os.path.exists(out_csv)

        out = capsys.readouterr().out
        for expected in expected_substrings:
            assert expected in out


def test_sample_lincoln_douglas(capsys):
    _run_sample_test(
        capsys,
        "debate_lincoln_douglas_1858.json",
        [
            "Agents: 3",
            "Round  5",
            "Orthogonal: Thomas J. Turner",
            "Popular Sovereignty & Dred Scott",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_kennedy_nixon(capsys):
    _run_sample_test(
        capsys,
        "debate_kennedy_nixon_1960.json",
        [
            "Agents: 6",
            "Round  5",
            "Orthogonal: Howard K. Smith",
            "Social Security",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_baldwin_buckley(capsys):
    _run_sample_test(
        capsys,
        "debate_baldwin_buckley_1965.json",
        [
            "Agents: 5",
            "Round  5",
            "Orthogonal: Mr. Burford",
            "American Dream",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_bush_clinton_perot(capsys):
    _run_sample_test(
        capsys,
        "debate_bush_clinton_perot_1992.json",
        [
            "Agents: 7",
            "Round  5",
            "Orthogonal: Ross Perot",
            "Unreconciled Dissent",
            "Primary Attractor",
            "Inactive Panelist",
            "Washington Gridlock",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_darrow_bryan(capsys):
    _run_sample_test(
        capsys,
        "debate_darrow_bryan_scopes_1925.json",
        [
            "Agents: 5",
            "Round  5",
            "Orthogonal: Judge John T. Raulston",
            "Biblical Record",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_chomsky_foucault(capsys):
    _run_sample_test(
        capsys,
        "debate_chomsky_foucault_1971.json",
        [
            "Agents: 3",
            "Round  5",
            "Orthogonal: Noam Chomsky",
            "Proletariat Wages War",
            "Effective Reasoning Channels (K*)",
        ],
    )


def test_sample_buckley_vidal(capsys):
    _run_sample_test(
        capsys,
        "debate_buckley_vidal_1968.json",
        [
            "Agents: 3",
            "Round  4",
            "Orthogonal: William F. Buckley Jr.",
            "Unreconciled Dissent",
            "ORTHOGONAL_PERSISTENCE",
            "National Convention",
            "Effective Reasoning Channels (K*)",
        ],
    )


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


def test_sample_scotus_trump_v_us_2024(capsys):
    _run_sample_test(
        capsys,
        "scotus_trump_v_us_2024.json",
        ["Agents: 11", "Round 10", "Absolute Immunity"],
    )


def test_sample_fomc_emergency_covid_mar2020(capsys):
    _run_sample_test(
        capsys,
        "fomc_emergency_covid_mar2020.json",
        ["Agents: 21", "Round  8", "Hierarchical K=8"],
        extra_args=["--hierarchical"],
    )


def test_sample_senate_ai_oversight_2023(capsys):
    _run_sample_test(
        capsys,
        "senate_ai_oversight_hearing_2023.json",
        ["Agents: 10", "Round  8", "Chairman Peters"],
    )


def test_sample_kennedy_nixon_1960_full(capsys):
    _run_sample_test(
        capsys,
        "debate_kennedy_nixon_1960_full.json",
        ["Agents: 7", "Round  8", "Howard K. Smith"],
    )


def test_sample_bush_clinton_perot_1992_full(capsys):
    _run_sample_test(
        capsys,
        "debate_bush_clinton_perot_1992_full.json",
        ["Agents: 7", "Round 10", "Jim Lehrer"],
    )


def test_sample_real_llm_werewolf_gpt5_vs_gemini(capsys):
    _run_sample_test(
        capsys,
        "real_llm_werewolf_gpt5_vs_gemini.json",
        ["Agents: 6", "Round  7", "Alice"],
    )


def test_sample_real_llm_werewolf_wolves_win(capsys):
    _run_sample_test(
        capsys,
        "real_llm_werewolf_wolves_win.json",
        ["Agents: 5", "Round  7", "Oscar"],
    )


def test_sample_real_llm_negotiation_gpt4_coop(capsys):
    _run_sample_test(
        capsys,
        "real_llm_negotiation_gpt4_coop.json",
        ["Agents: 7", "Round 10", "SportCo"],
    )


def test_sample_real_llm_negotiation_targeted_adv(capsys):
    _run_sample_test(
        capsys,
        "real_llm_negotiation_targeted_adv.json",
        ["Agents: 6", "Round  9", "Environmental League"],
    )


