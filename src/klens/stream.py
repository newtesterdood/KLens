from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from klens.alerts import AlertMonitor
from klens.encoders import TextEncoder, TfidfTextEncoder
from klens.exporter import CSVLogExporter
from klens.models import (
    Alert,
    AlertConfig,
    CompressionConfig,
    HierarchicalConfig,
    SpectralMetrics,
    TranscriptTurn,
)
from klens.spectral import (
    cluster_factions,
    compress_to_faction_centroids,
    compute_koopman_metrics,
    detect_rogue_agent,
    detect_rogue_from_states,
    extract_centroid_keywords,
    predict_agent_trajectories,
)



class LiveTranscriptStreamProcessor:
    def __init__(
        self,
        csv_log_path: str = "spectral_debate_log.csv",
        max_features: int = 30,
        alert_config: Optional[AlertConfig] = None,
        hierarchical_config: Optional[HierarchicalConfig] = None,
        compression_config: Optional[HierarchicalConfig] = None,
        encoder: Optional[TextEncoder] = None,
        sliding_window_rounds: Optional[int] = None,
        auto_init_csv: bool = True,
    ):
        self.csv_log_path = csv_log_path
        self.max_features = max_features
        self.sliding_window_rounds = sliding_window_rounds
        self.transcript_buffer: List[TranscriptTurn] = []
        self.encoder: TextEncoder = encoder or TfidfTextEncoder(max_features=max_features)
        self.alert_monitor = AlertMonitor(config=alert_config)
        self.hierarchical_config = hierarchical_config or compression_config or HierarchicalConfig()
        self.exporter = CSVLogExporter(file_path=csv_log_path)
        self.round_callbacks: List[Callable[[SpectralMetrics], None]] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    @property
    def compression_config(self) -> HierarchicalConfig:
        return self.hierarchical_config


        if auto_init_csv:
            self.init_csv_log()

    def init_csv_log(self) -> None:
        self.exporter.init_file()

    def add_round_callback(self, cb: Callable[[SpectralMetrics], None]) -> None:
        self.round_callbacks.append(cb)

    def add_alert_callback(self, cb: Callable[[Alert], None]) -> None:
        self.alert_callbacks.append(cb)

    def ingest_turn(
        self,
        round_id: int,
        speaker: str,
        text: str,
        timestamp: Optional[float] = None,
    ) -> None:
        self.transcript_buffer.append(
            TranscriptTurn(
                round_id=round_id,
                speaker=speaker,
                text=text,
                timestamp=timestamp,
            )
        )

    def process_round_event(self, current_round: int) -> Optional[SpectralMetrics]:
        if self.sliding_window_rounds and self.sliding_window_rounds > 0:
            min_round = max(1, current_round - self.sliding_window_rounds + 1)
            records = [
                r for r in self.transcript_buffer
                if min_round <= r.round_id <= current_round
            ]
        else:
            records = [r for r in self.transcript_buffer if r.round_id <= current_round]

        if not records:
            return None

        rounds = sorted(list(set(r.round_id for r in records)))
        speakers = sorted(list(set(r.speaker for r in records)))
        T, N = len(rounds), len(speakers)
        if T < 2 or N < 2:
            return None

        text_grid: Dict[tuple[int, str], str] = {(r, s): "" for r in rounds for s in speakers}
        for r in records:
            text_grid[(r.round_id, r.speaker)] += " " + r.text

        Z, feature_names = self.encoder.encode_grid(text_grid, rounds, speakers)
        d = len(feature_names)

        is_compressed = False
        if self.compression_config.enabled and N > self.compression_config.max_factions:
            Z_macro, factions = compress_to_faction_centroids(
                Z=Z,
                speakers=speakers,
                max_factions=self.compression_config.max_factions,
                seed=self.compression_config.seed,
            )
            l2_mag, t_cert, _, Z_mean = compute_koopman_metrics(
                Z_macro,
                solver=self.compression_config.solver,
                seed=self.compression_config.seed,
            )
            rogue_speaker, _ = detect_rogue_from_states(Z, speakers)
            is_compressed = True
        else:
            l2_mag, t_cert, Delta, Z_mean = compute_koopman_metrics(
                Z,
                solver=self.compression_config.solver,
                seed=self.compression_config.seed,
            )
            rogue_speaker, rogue_idx = detect_rogue_agent(Delta, speakers, T, N, d)
            final_states = Z[-1]
            factions = cluster_factions(
                final_states=final_states,
                speakers=speakers,
                rogue_idx=rogue_idx,
                similarity_threshold=0.65,
            )

        final_centroid = Z_mean[-1]
        centroid_keywords = extract_centroid_keywords(final_centroid, feature_names, top_k=4)

        factions_str = " | ".join([f.format_summary() for f in factions])
        centroid_str = ";".join(centroid_keywords)

        pred_rogue, pred_contraction, _ = predict_agent_trajectories(
            Z=Z,
            speakers=speakers,
            steps_ahead=1,
            seed=self.hierarchical_config.seed,
            solver=self.hierarchical_config.solver,
        )

        metrics = SpectralMetrics(
            round_id=current_round,
            active_agents_count=N,
            lambda_2_magnitude=l2_mag,
            certified_deadline_rounds=t_cert,
            rogue_agent_detected=rogue_speaker,
            faction_count=len(factions),
            factions_summary=factions_str,
            centroid_top_keywords=centroid_str,
            factions=factions,
            centroid_keywords_list=centroid_keywords,
            is_converging=(l2_mag < 1.0),
            is_hierarchical=is_compressed,
            predicted_next_rogue=pred_rogue,
            predicted_contraction_ratio=pred_contraction,
        )

        new_alerts = self.alert_monitor.evaluate(metrics)
        metrics.alerts = new_alerts

        self.exporter.log_metrics(metrics)


        for cb in self.round_callbacks:
            cb(metrics)

        for alert in new_alerts:
            for acb in self.alert_callbacks:
                acb(alert)

        return metrics
