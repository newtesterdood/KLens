import json
import os
import tempfile
import pytest

from klens.cli import main


def test_cli_demo(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "demo_out.csv")
        ret = main(["demo", "--output", csv_path, "--t-cert-limit", "50"])
        assert ret == 0

        captured = capsys.readouterr()
        assert "KLens" in captured.out
        assert "Round  2" in captured.out
        assert "Round  3" in captured.out
        assert os.path.exists(csv_path)


def test_cli_run_json(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "turns.json")
        csv_path = os.path.join(tmpdir, "run_out.csv")

        turns_data = [
            {"round": 1, "speaker": "A1", "text": "Auditing is mandatory for model integrity."},
            {"round": 1, "speaker": "A2", "text": "Cost efficiency requires fewer audits."},
            {"round": 2, "speaker": "A1", "text": "Selective auditing is acceptable."},
            {"round": 2, "speaker": "A2", "text": "We agree on selective auditing compromise."},
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(turns_data, f)

        ret = main(["run", json_path, "--output", csv_path])
        assert ret == 0
        assert os.path.exists(csv_path)
