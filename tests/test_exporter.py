import csv
import os
import tempfile
import pytest

from klens.exporter import CSVLogExporter, DEFAULT_COLUMNS
from klens.models import Faction, SpectralMetrics


def test_csv_log_exporter_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "subdir", "test_spectral.csv")
        exporter = CSVLogExporter(csv_file)
        exporter.init_file()

        metrics = SpectralMetrics(
            round_id=2,
            active_agents_count=6,
            lambda_2_magnitude=0.93124,
            certified_deadline_rounds=43,
            rogue_agent_detected="Agent_1",
            faction_count=5,
            factions_summary="F1:(Agent_2) | F2:(Agent_3) | F3:(Agent_4) | F4:(Agent_5) | F5:(Agent_6)",
            centroid_top_keywords="judge;models;cost;auditing",
        )

        row = exporter.log_metrics(metrics)
        assert row[0] == 2
        assert row[1] == 6
        assert row[2] == 0.9312
        assert row[3] == 43
        assert row[4] == "Agent_1"
        assert row[5] == 5
        assert "Agent_2" in row[6]
        assert "judge;models;cost;auditing" == row[7]

        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert reader[0] == DEFAULT_COLUMNS
            assert len(reader) == 2
            assert reader[1][0] == "2"
            assert reader[1][1] == "6"
            assert reader[1][2] == "0.9312"
            assert reader[1][3] == "43"
            assert reader[1][4] == "Agent_1"
            assert reader[1][5] == "5"
            assert reader[1][6] == "F1:(Agent_2) | F2:(Agent_3) | F3:(Agent_4) | F4:(Agent_5) | F5:(Agent_6)"
            assert reader[1][7] == "judge;models;cost;auditing"
