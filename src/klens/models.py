from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    level: AlertLevel
    code: str
    message: str
    round_id: int
    metric_value: Optional[float] = None


@dataclass
class AlertConfig:
    t_cert_limit: Optional[int] = 50
    lambda_2_warning_limit: float = 0.95
    lambda_2_critical_limit: float = 0.99
    rogue_persistence_rounds: int = 3
    enabled: bool = True


@dataclass
class HierarchicalConfig:
    enabled: bool = False
    max_factions: int = 8
    solver: str = "auto"
    seed: int = 42


# Backward compatibility alias
CompressionConfig = HierarchicalConfig



@dataclass(frozen=True)
class TranscriptTurn:
    round_id: int
    speaker: str
    text: str
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class Faction:
    id: int
    members: List[str]

    def format_summary(self) -> str:
        return f"F{self.id}:({','.join(self.members)})"


@dataclass
class SpectralMetrics:
    round_id: int
    active_agents_count: int
    lambda_2_magnitude: float
    certified_deadline_rounds: int
    rogue_agent_detected: str
    faction_count: int
    factions_summary: str
    centroid_top_keywords: str
    factions: List[Faction] = field(default_factory=list)
    centroid_keywords_list: List[str] = field(default_factory=list)
    is_converging: bool = True
    is_hierarchical: bool = False
    is_compressed: bool = False

    def __post_init__(self) -> None:
        if self.is_compressed and not self.is_hierarchical:
            self.is_hierarchical = self.is_compressed
        elif self.is_hierarchical and not self.is_compressed:
            self.is_compressed = self.is_hierarchical




    def to_csv_row(self) -> List[object]:
        return [
            self.round_id,
            self.active_agents_count,
            round(self.lambda_2_magnitude, 4),
            self.certified_deadline_rounds,
            self.rogue_agent_detected,
            self.faction_count,
            self.factions_summary,
            self.centroid_top_keywords,
        ]

    def __getitem__(self, item: int) -> object:
        return self.to_csv_row()[item]

    def __len__(self) -> int:
        return len(self.to_csv_row())

    def __iter__(self):
        return iter(self.to_csv_row())

