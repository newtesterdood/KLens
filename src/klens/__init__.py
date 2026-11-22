from __future__ import annotations

from klens.alerts import AlertMonitor
from klens.encoders import TextEncoder, TfidfTextEncoder
from klens.exporter import CSVLogExporter, DEFAULT_COLUMNS
from klens.models import (
    Alert,
    AlertConfig,
    AlertLevel,
    DeliberationTopicSynthesis,
    Faction,
    FactionStance,
    HierarchicalConfig,
    SpectralMetrics,
    TranscriptTurn,
)
from klens.semantic import synthesize_deliberation_topics
from klens.spectral import (
    cluster_factions,
    coarse_grain_faction_centroids,
    compute_effective_channels_k_star,
    compute_koopman_metrics,
    compute_nnls_subspace_orthogonality,
    compute_orthogonal_significance,
    detect_rogue_agent,
    detect_rogue_from_states,
    extract_centroid_keywords,
    extract_last_active_states,
    lift_state,
    predict_agent_trajectories,
    validate_koopman_walk_forward,
)

from klens.stream import LiveTranscriptStreamProcessor
from klens.vis import plot_from_transcript_file, render_spectral_dashboard

__version__ = "0.2.0"

__all__ = [
    "Alert",
    "AlertConfig",
    "AlertLevel",
    "AlertMonitor",
    "CSVLogExporter",
    "DEFAULT_COLUMNS",
    "DeliberationTopicSynthesis",
    "Faction",
    "FactionStance",
    "HierarchicalConfig",
    "LiveTranscriptStreamProcessor",
    "SpectralMetrics",
    "TextEncoder",
    "TfidfTextEncoder",
    "TranscriptTurn",
    "cluster_factions",
    "coarse_grain_faction_centroids",
    "compute_effective_channels_k_star",
    "compute_koopman_metrics",
    "compute_nnls_subspace_orthogonality",
    "compute_orthogonal_significance",
    "detect_rogue_agent",
    "detect_rogue_from_states",
    "extract_centroid_keywords",
    "extract_last_active_states",
    "lift_state",
    "plot_from_transcript_file",
    "predict_agent_trajectories",
    "render_spectral_dashboard",
    "synthesize_deliberation_topics",
    "validate_koopman_walk_forward",
]

