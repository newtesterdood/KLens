from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from klens import __version__
from klens.models import AlertConfig, CompressionConfig, SpectralMetrics
from klens.stream import LiveTranscriptStreamProcessor

SAMPLE_DEBATE_TURNS = [
    # Round 1
    (1, "Agent_1", "We need autonomous LLM judge models with zero human auditing to minimize cost."),
    (1, "Agent_2", "Zero human auditing is unacceptable. We must have strict human verification layers."),
    (1, "Agent_3", "A hybrid tier could work: judge models for low risk, human auditing for high risk."),
    (1, "Agent_4", "Cost is the limiting factor. Specialized small language models (SLM) can handle the audit."),
    (1, "Agent_5", "I agree with Agent_3. Selective human auditing balances safety and operational cost."),
    (1, "Agent_6", "A compromise between automated judge models and SLM auditing gives optimal safety."),
    # Round 2
    (2, "Agent_1", "Autonomous judge models should decide everything. Cost reduction is paramount."),
    (2, "Agent_2", "Without auditing, safety guarantees fail. We require mandatory human review checkpoints."),
    (2, "Agent_3", "Selective auditing with judge models reduces cost while maintaining compliance."),
    (2, "Agent_4", "SLM auditing alongside judge models is viable if we enforce sample review rates."),
    (2, "Agent_5", "A hybrid compromise where SLM models filter low risk and humans review edge cases works."),
    (2, "Agent_6", "Selective SLM auditing combined with judge models achieves consensus on cost and safety."),
    # Round 3
    (3, "Agent_1", "I can accept hybrid SLM auditing if benchmark cost targets are strictly satisfied."),
    (3, "Agent_2", "I accept the hybrid compromise if selective human escalation remains for edge cases."),
    (3, "Agent_3", "Consensus: hybrid selective auditing combining judge models and SLM verification."),
    (3, "Agent_4", "Agreed on hybrid compromise: selective SLM verification with escalation."),
    (3, "Agent_5", "Agreed on hybrid compromise architecture with selective auditing."),
    (3, "Agent_6", "Full agreement reached on hybrid compromise selective SLM model architecture."),
]


def format_metrics_table_row(m: SpectralMetrics) -> str:
    status_indicator = "✓ Converging" if m.is_converging else "✗ Diverging"
    mode_str = f" [Compressed K={m.faction_count}]" if m.is_compressed else ""
    return (
        f"Round {m.round_id:2d} | Agents: {m.active_agents_count} | "
        f"λ2: {m.lambda_2_magnitude:.4f} | T_cert: {m.certified_deadline_rounds:3d} rounds | "
        f"Rogue: {m.rogue_agent_detected:<8} | Factions: {m.faction_count} | [{status_indicator}]{mode_str}"
    )


def run_demo(
    output_csv: str,
    t_cert_limit: int,
    compression_enabled: bool,
    max_factions: int,
    solver: str,
) -> int:
    mode_descr = f"Compressed (max_factions={max_factions})" if compression_enabled else "Uncompressed (full compute)"
    print(f"KLens v{__version__} - Spectral Multi-Agent Consensus Monitor")
    print(f"Simulating 6-Agent Debate across 3 rounds | Mode: {mode_descr} | Solver: {solver} | Log -> {output_csv}\n")

    alert_cfg = AlertConfig(t_cert_limit=t_cert_limit)
    comp_cfg = CompressionConfig(
        enabled=compression_enabled,
        max_factions=max_factions,
        solver=solver,
    )
    processor = LiveTranscriptStreamProcessor(
        csv_log_path=output_csv,
        alert_config=alert_cfg,
        compression_config=comp_cfg,
    )

    processor.add_alert_callback(
        lambda a: print(f"  [ALERT {a.level.value}] {a.code}: {a.message}")
    )

    rounds = sorted(list(set(t[0] for t in SAMPLE_DEBATE_TURNS)))
    for r in rounds:
        round_turns = [t for t in SAMPLE_DEBATE_TURNS if t[0] == r]
        for _, speaker, text in round_turns:
            processor.ingest_turn(round_id=r, speaker=speaker, text=text)

        metrics = processor.process_round_event(current_round=r)
        if metrics:
            print(format_metrics_table_row(metrics))
            print(f"  Centroid keywords: {metrics.centroid_top_keywords}")
            print(f"  Factions: {metrics.factions_summary}")
        else:
            print(f"Round {r:2d} | Buffered {len(round_turns)} turns (need >= 2 rounds to compute transfer operator)")
        print("-" * 80)

    print(f"\nCompleted debate simulation. Log written to: {os.path.abspath(output_csv)}")
    return 0


def run_file(
    transcript_file: str,
    output_csv: str,
    t_cert_limit: int,
    compression_enabled: bool,
    max_factions: int,
    solver: str,
) -> int:
    if not os.path.exists(transcript_file):
        print(f"Error: file not found: {transcript_file}", file=sys.stderr)
        return 1

    alert_cfg = AlertConfig(t_cert_limit=t_cert_limit)
    comp_cfg = CompressionConfig(
        enabled=compression_enabled,
        max_factions=max_factions,
        solver=solver,
    )
    processor = LiveTranscriptStreamProcessor(
        csv_log_path=output_csv,
        alert_config=alert_cfg,
        compression_config=comp_cfg,
    )

    processor.add_alert_callback(
        lambda a: print(f"  [ALERT {a.level.value}] {a.code}: {a.message}")
    )

    turns: List[dict] = []
    if transcript_file.endswith(".jsonl"):
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))
    else:
        with open(transcript_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            turns = data if isinstance(data, list) else data.get("turns", [])

    rounds_seen = set()
    for t in turns:
        r = int(t.get("round", t.get("round_id", 1)))
        speaker = str(t.get("speaker", "Unknown"))
        text = str(t.get("text", ""))
        processor.ingest_turn(round_id=r, speaker=speaker, text=text)
        rounds_seen.add(r)

    for r in sorted(list(rounds_seen)):
        metrics = processor.process_round_event(current_round=r)
        if metrics:
            print(format_metrics_table_row(metrics))
            print(f"  Keywords: {metrics.centroid_top_keywords}")
            print(f"  Factions: {metrics.factions_summary}")
        else:
            print(f"Round {r:2d} | Insufficient history for spectral decomposition (< 2 rounds)")

    print(f"\nProcessing complete. Output saved to: {os.path.abspath(output_csv)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="klens",
        description="Koopman EDMD Spectral Analysis for Multi-Agent Consensus",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run reference multi-agent debate simulation")
    demo_parser.add_argument(
        "--output",
        "-o",
        default="spectral_debate_log.csv",
        help="Path for CSV output log (default: spectral_debate_log.csv)",
    )
    demo_parser.add_argument(
        "--t-cert-limit",
        type=int,
        default=50,
        help="Certified deadline alert threshold (default: 50 rounds)",
    )
    demo_parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable faction centroid compression for high-agent scaling",
    )
    demo_parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when compression is enabled (default: 8)",
    )
    demo_parser.add_argument(
        "--solver",
        choices=["auto", "primal", "dual"],
        default="auto",
        help="Koopman solver mode: auto, primal, or dual (default: auto)",
    )

    run_parser = subparsers.add_parser("run", help="Process transcript JSON or JSONL file")
    run_parser.add_argument("transcript", help="Path to transcript JSON or JSONL file")
    run_parser.add_argument(
        "--output",
        "-o",
        default="spectral_debate_log.csv",
        help="Path for CSV output log (default: spectral_debate_log.csv)",
    )
    run_parser.add_argument(
        "--t-cert-limit",
        type=int,
        default=50,
        help="Certified deadline alert threshold (default: 50 rounds)",
    )
    run_parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable faction centroid compression for high-agent scaling",
    )
    run_parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when compression is enabled (default: 8)",
    )
    run_parser.add_argument(
        "--solver",
        choices=["auto", "primal", "dual"],
        default="auto",
        help="Koopman solver mode: auto, primal, or dual (default: auto)",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        return run_demo(
            args.output,
            args.t_cert_limit,
            args.compress,
            args.max_factions,
            args.solver,
        )
    elif args.command == "run":
        return run_file(
            args.transcript,
            args.output,
            args.t_cert_limit,
            args.compress,
            args.max_factions,
            args.solver,
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
