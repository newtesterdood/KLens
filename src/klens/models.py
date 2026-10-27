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
    orthogonal_persistence_rounds: int = 3
    enabled: bool = True

    def __post_init__(self):
        if self.orthogonal_persistence_rounds != 3 and self.rogue_persistence_rounds == 3:
            self.rogue_persistence_rounds = self.orthogonal_persistence_rounds
        elif self.rogue_persistence_rounds != 3 and self.orthogonal_persistence_rounds == 3:
            self.orthogonal_persistence_rounds = self.rogue_persistence_rounds


@dataclass
class HierarchicalConfig:
    enabled: bool = False
    max_factions: int = 8
    solver: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class TranscriptTurn:
    round_id: int
    speaker: str
    text: str
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class FactionStance:
    id: int
    members: List[str]
    topic: str
    keywords: List[str]
    representative_statement: str
    position_summary: str


@dataclass(frozen=True)
class DeliberationTopicSynthesis:
    center_of_disagreement: str
    center_of_disagreement_statement: str
    centroid_concept: str
    centroid_statement: str
    disagreement_poles: tuple[str, str]
    disagreement_keywords: List[str]
    factions: List[FactionStance]
    rogue_stance: Optional[FactionStance] = None
    orthogonal_stance: Optional[FactionStance] = None

    def __post_init__(self):
        if self.orthogonal_stance and not self.rogue_stance:
            object.__setattr__(self, "rogue_stance", self.orthogonal_stance)
        elif self.rogue_stance and not self.orthogonal_stance:
            object.__setattr__(self, "orthogonal_stance", self.rogue_stance)


@dataclass(frozen=True)
class Faction:
    id: int
    members: List[str]
    topic: str = ""
    representative_statement: str = ""

    def format_summary(self) -> str:
        return f"F{self.id}:({','.join(self.members)})"


@dataclass
class SpectralMetrics:
    round_id: int
    active_agents_count: int
    lambda_2_magnitude: float
    certified_deadline_rounds: int
    rogue_agent_detected: str = ""
    faction_count: int = 0
    factions_summary: str = ""
    centroid_top_keywords: str = ""
    factions: List[Faction] = field(default_factory=list)
    centroid_keywords_list: List[str] = field(default_factory=list)
    is_converging: bool = True
    is_hierarchical: bool = False
    predicted_next_rogue: Optional[str] = None
    predicted_contraction_ratio: Optional[float] = None
    topic_synthesis: Optional[DeliberationTopicSynthesis] = None
    center_of_disagreement: str = ""
    centroid_concept: str = ""
    centroid_statement: str = ""
    orthogonal_agent_detected: str = ""
    predicted_next_orthogonal: Optional[str] = None
    topological_state: str = "Convergence"
    agent_cluster_assignments: dict[str, int] = field(default_factory=dict)
    agent_local_residuals: dict[str, float] = field(default_factory=dict)
    agent_migrations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.orthogonal_agent_detected and not self.rogue_agent_detected:
            self.rogue_agent_detected = self.orthogonal_agent_detected
        elif self.rogue_agent_detected and not self.orthogonal_agent_detected:
            self.orthogonal_agent_detected = self.rogue_agent_detected
        if self.predicted_next_orthogonal and not self.predicted_next_rogue:
            self.predicted_next_rogue = self.predicted_next_orthogonal
        elif self.predicted_next_rogue and not self.predicted_next_orthogonal:
            self.predicted_next_orthogonal = self.predicted_next_rogue

    def to_csv_row(self) -> List[object]:
        return [
            self.round_id,
            self.active_agents_count,
            round(self.lambda_2_magnitude, 4),
            self.certified_deadline_rounds,
            self.orthogonal_agent_detected or self.rogue_agent_detected,
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
