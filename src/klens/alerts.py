from __future__ import annotations

from typing import List, Optional

from klens.models import Alert, AlertConfig, AlertLevel, SpectralMetrics


class AlertMonitor:
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.history: List[SpectralMetrics] = []

    def evaluate(self, metrics: SpectralMetrics) -> List[Alert]:
        if not self.config.enabled:
            return []

        alerts: List[Alert] = []
        r_id = metrics.round_id

        # 1. Divergence / Non-contractive dynamics
        if metrics.lambda_2_magnitude >= 1.0:
            alerts.append(
                Alert(
                    level=AlertLevel.CRITICAL,
                    code="DIVERGENT_DYNAMICS",
                    message=f"Debate dynamics are non-contractive (|lambda_2| = {metrics.lambda_2_magnitude:.4f} >= 1.0). Consensus is not guaranteed.",
                    round_id=r_id,
                    metric_value=metrics.lambda_2_magnitude,
                )
            )
        elif metrics.lambda_2_magnitude >= self.config.lambda_2_critical_limit:
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    code="CRITICAL_SPECTRAL_RADIUS",
                    message=f"Near-unit spectral radius (|lambda_2| = {metrics.lambda_2_magnitude:.4f} >= {self.config.lambda_2_critical_limit}). Severe resistance to consensus.",
                    round_id=r_id,
                    metric_value=metrics.lambda_2_magnitude,
                )
            )
        elif metrics.lambda_2_magnitude >= self.config.lambda_2_warning_limit:
            alerts.append(
                Alert(
                    level=AlertLevel.INFO,
                    code="SLOW_CONVERGENCE",
                    message=f"Slow consensus convergence mode (|lambda_2| = {metrics.lambda_2_magnitude:.4f} >= {self.config.lambda_2_warning_limit}).",
                    round_id=r_id,
                    metric_value=metrics.lambda_2_magnitude,
                )
            )

        # 2. Certified deadline threshold
        if (
            self.config.t_cert_limit is not None
            and metrics.certified_deadline_rounds > self.config.t_cert_limit
        ):
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    code="T_CERT_EXCEEDED",
                    message=(
                        f"Certified deadline T_cert ({metrics.certified_deadline_rounds} rounds) "
                        f"exceeds safety threshold of {self.config.t_cert_limit} rounds."
                    ),
                    round_id=r_id,
                    metric_value=float(metrics.certified_deadline_rounds),
                )
            )

        # 3. Persistent orthogonal / rogue agent
        limit = min(self.config.orthogonal_persistence_rounds, self.config.rogue_persistence_rounds)
        if self.history and limit > 1:
            streak = 1
            current_ortho = metrics.orthogonal_agent_detected or metrics.rogue_agent_detected
            for past in reversed(self.history):
                past_ortho = past.orthogonal_agent_detected or past.rogue_agent_detected
                if past_ortho == current_ortho:
                    streak += 1
                else:
                    break
            if streak >= limit:
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        code="ORTHOGONAL_PERSISTENCE",
                        message=(
                            f"Agent '{current_ortho}' flagged as orthogonal for {streak} "
                            f"consecutive rounds. Sustained transverse divergence detected."
                        ),
                        round_id=r_id,
                    )
                )
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        code="PERSISTENT_ROGUE",
                        message=(
                            f"Agent '{current_ortho}' flagged as rogue for {streak} "
                            f"consecutive rounds. Sustained divergence detected."
                        ),
                        round_id=r_id,
                    )
                )

        self.history.append(metrics)
        return alerts
