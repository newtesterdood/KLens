from __future__ import annotations

import csv
import os
from typing import List, Sequence

from klens.models import SpectralMetrics

DEFAULT_COLUMNS: List[str] = [
    "round",
    "active_agents_count",
    "lambda_2_magnitude",
    "certified_deadline_rounds",
    "rogue_agent_detected",
    "faction_count",
    "factions_summary",
    "centroid_top_keywords",
]


class CSVLogExporter:
    def __init__(self, file_path: str = "spectral_debate_log.csv"):
        self.file_path = file_path
        self._initialized = False

    def init_file(self) -> None:
        target_dir = os.path.dirname(self.file_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(DEFAULT_COLUMNS)
        self._initialized = True

    def append_row(self, row: Sequence[object]) -> None:
        if not self._initialized and (not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0):
            self.init_file()
        else:
            self._initialized = True

        target_dir = os.path.dirname(self.file_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        with open(self.file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_metrics(self, metrics: SpectralMetrics) -> List[object]:
        row = metrics.to_csv_row()
        self.append_row(row)
        return row
