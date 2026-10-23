import pytest

from klens.alerts import AlertMonitor
from klens.models import AlertConfig, AlertLevel, SpectralMetrics


def create_dummy_metrics(
    round_id: int,
    l2: float = 0.90,
    t_cert: int = 25,
    rogue: str = "Agent_1",
) -> SpectralMetrics:
    return SpectralMetrics(
        round_id=round_id,
        active_agents_count=4,
        lambda_2_magnitude=l2,
        certified_deadline_rounds=t_cert,
        rogue_agent_detected=rogue,
        faction_count=3,
        factions_summary="F1:(Agent_2) | F2:(Agent_3)",
        centroid_top_keywords="compromise;safety",
    )


def test_alert_t_cert_exceeded():
    config = AlertConfig(t_cert_limit=40)
    monitor = AlertMonitor(config=config)

    # Within limit
    alerts_ok = monitor.evaluate(create_dummy_metrics(round_id=2, t_cert=35))
    assert not any(a.code == "T_CERT_EXCEEDED" for a in alerts_ok)

    # Exceeds limit
    alerts_exceeded = monitor.evaluate(create_dummy_metrics(round_id=3, t_cert=75))
    t_alerts = [a for a in alerts_exceeded if a.code == "T_CERT_EXCEEDED"]
    assert len(t_alerts) == 1
    assert t_alerts[0].level == AlertLevel.WARNING
    assert t_alerts[0].metric_value == 75.0


def test_alert_divergent_dynamics():
    monitor = AlertMonitor()
    alerts = monitor.evaluate(create_dummy_metrics(round_id=2, l2=1.02))
    div_alerts = [a for a in alerts if a.code == "DIVERGENT_DYNAMICS"]
    assert len(div_alerts) == 1
    assert div_alerts[0].level == AlertLevel.CRITICAL


def test_alert_persistent_rogue():
    config = AlertConfig(rogue_persistence_rounds=3)
    monitor = AlertMonitor(config=config)

    monitor.evaluate(create_dummy_metrics(round_id=2, rogue="Agent_Rogue"))
    monitor.evaluate(create_dummy_metrics(round_id=3, rogue="Agent_Rogue"))
    alerts_r4 = monitor.evaluate(create_dummy_metrics(round_id=4, rogue="Agent_Rogue"))

    rogue_alerts = [a for a in alerts_r4 if a.code == "PERSISTENT_ROGUE"]
    assert len(rogue_alerts) == 1
    assert "Agent_Rogue" in rogue_alerts[0].message
