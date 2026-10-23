#!/usr/bin/env python3
"""
Example: Live Transcript Streaming with Koopman Spectral Consensus Tracking
"""

from klens import (
    AlertConfig,
    LiveTranscriptStreamProcessor,
)

TURNS = [
    (1, "Agent_1", "We need autonomous LLM judge models with zero human auditing to minimize cost."),
    (1, "Agent_2", "Zero human auditing is unacceptable. We must have strict human verification layers."),
    (1, "Agent_3", "A hybrid tier could work: judge models for low risk, human auditing for high risk."),
    (1, "Agent_4", "Cost is the limiting factor. Specialized small language models (SLM) can handle the audit."),
    (1, "Agent_5", "I agree with Agent_3. Selective human auditing balances safety and operational cost."),
    (1, "Agent_6", "A compromise between automated judge models and SLM auditing gives optimal safety."),

    (2, "Agent_1", "Autonomous judge models should decide everything. Cost reduction is paramount."),
    (2, "Agent_2", "Without auditing, safety guarantees fail. We require mandatory human review checkpoints."),
    (2, "Agent_3", "Selective auditing with judge models reduces cost while maintaining compliance."),
    (2, "Agent_4", "SLM auditing alongside judge models is viable if we enforce sample review rates."),
    (2, "Agent_5", "A hybrid compromise where SLM models filter low risk and humans review edge cases works."),
    (2, "Agent_6", "Selective SLM auditing combined with judge models achieves consensus on cost and safety."),

    (3, "Agent_1", "I can accept hybrid SLM auditing if benchmark cost targets are strictly satisfied."),
    (3, "Agent_2", "I accept the hybrid compromise if selective human escalation remains for edge cases."),
    (3, "Agent_3", "Consensus: hybrid selective auditing combining judge models and SLM verification."),
    (3, "Agent_4", "Agreed on hybrid compromise: selective SLM verification with escalation."),
    (3, "Agent_5", "Agreed on hybrid compromise architecture with selective auditing."),
    (3, "Agent_6", "Full agreement reached on hybrid compromise selective SLM model architecture."),
]


def main():
    config = AlertConfig(t_cert_limit=50, lambda_2_warning_limit=0.95)
    processor = LiveTranscriptStreamProcessor(
        csv_log_path="out/spectral_debate_log.csv",
        alert_config=config,
    )

    processor.add_alert_callback(
        lambda a: print(f"  [ALERT {a.level.value}] {a.code}: {a.message}")
    )

    print("=== KLens Live Transcript Streaming Demo ===\n")
    rounds = sorted(list(set(t[0] for t in TURNS)))
    for r in rounds:
        round_turns = [t for t in TURNS if t[0] == r]
        for _, speaker, text in round_turns:
            processor.ingest_turn(round_id=r, speaker=speaker, text=text)

        metrics = processor.process_round_event(current_round=r)
        if metrics:
            print(f"--- Round {r} ---")
            print(f"Active Agents: {metrics.active_agents_count}")
            print(f"λ2 Magnitude:  {metrics.lambda_2_magnitude:.4f}")
            print(f"T_cert:        {metrics.certified_deadline_rounds} rounds")
            print(f"Rogue Agent:   {metrics.rogue_agent_detected}")
            print(f"Factions:      {metrics.factions_summary}")
            print(f"Centroid Top:  {metrics.centroid_top_keywords}")
        else:
            print(f"--- Round {r} (Buffering initial state) ---")
        print()

    print("Completed. CSV saved to out/spectral_debate_log.csv")


if __name__ == "__main__":
    main()
