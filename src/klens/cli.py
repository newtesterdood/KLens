from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from klens import __version__
from klens.models import AlertConfig, HierarchicalConfig, SpectralMetrics
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
    mode_str = f" [Hierarchical K={m.faction_count}]" if m.is_hierarchical else ""
    detected_agent = m.orthogonal_agent_detected or m.rogue_agent_detected
    return (
        f"Round {m.round_id:2d} | Agents: {m.active_agents_count} | "
        f"λ2: {m.lambda_2_magnitude:.4f} | T_cert: {m.certified_deadline_rounds:3d} rounds | "
        f"Orthogonal: {detected_agent:<8} | Factions: {m.faction_count} | [{status_indicator}]{mode_str}"
    )


def format_semantic_synthesis_summary(m: SpectralMetrics) -> str:
    synth = m.topic_synthesis
    if not synth:
        return ""
    lines = [
        "\n=== Deliberation Semantic Synthesis ===",
        f"• Center of Disagreement: {synth.center_of_disagreement}",
        f"  \"{synth.center_of_disagreement_statement}\"",
        f"• Consensus Centroid Concept: {synth.centroid_concept}",
        f"  \"{synth.centroid_statement}\"",
        "\n--- Faction Topics & Positions ---",
    ]
    for f in synth.factions:
        lines.append(f"• Faction {f.id} ({len(f.members)} members): {f.topic}")
        lines.append(f"  Position: \"{f.representative_statement}\"")
    if synth.orthogonal_stance or synth.rogue_stance:
        r = synth.orthogonal_stance or synth.rogue_stance
        lines.append(f"\n--- Orthogonal / Dissent Position ({r.members[0]}) ---")
        lines.append(f"• Stance: {r.topic}")
        lines.append(f"  Position: \"{r.representative_statement}\"")
    return "\n".join(lines)


def run_demo(
    output_csv: str,
    t_cert_limit: int,
    hierarchical_enabled: bool,
    max_factions: int,
    solver: str,
    plot_path: Optional[str] = None,
    show_plot: bool = False,
) -> int:
    mode_descr = f"Hierarchical Coarse-Graining (max_factions={max_factions})" if hierarchical_enabled else "Micro-Level (full compute)"
    print(f"KLens v{__version__} - Spectral Multi-Agent Consensus Monitor")
    print(f"Simulating 6-Agent Debate across 3 rounds | Mode: {mode_descr} | Solver: {solver} | Log -> {output_csv}\n")

    alert_cfg = AlertConfig(t_cert_limit=t_cert_limit)
    hier_cfg = HierarchicalConfig(
        enabled=hierarchical_enabled,
        max_factions=max_factions,
        solver=solver,
    )
    processor = LiveTranscriptStreamProcessor(
        csv_log_path=output_csv,
        alert_config=alert_cfg,
        hierarchical_config=hier_cfg,
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
            pred_agent = metrics.predicted_next_orthogonal or metrics.predicted_next_rogue
            if pred_agent:
                pred_str = f"  Forecast (Round {r+1}): Projected Orthogonal -> {pred_agent}"
                if metrics.predicted_contraction_ratio is not None:
                    pred_str += f" | Contraction Rate -> {metrics.predicted_contraction_ratio:.2f}x"
                print(pred_str)
        else:
            print(f"Round {r:2d} | Buffered {len(round_turns)} turns (need >= 2 rounds to compute transfer operator)")
        print("-" * 80)

    if processor.latest_metrics and processor.latest_metrics.topic_synthesis:
        print(format_semantic_synthesis_summary(processor.latest_metrics))

    print(f"\nCompleted debate simulation. Log written to: {os.path.abspath(output_csv)}")

    if plot_path is not None or show_plot:
        from klens.vis import render_spectral_dashboard
        resolved_plot = plot_path if (plot_path and len(plot_path) > 0) else "out/demo_telemetry.png"
        render_spectral_dashboard(
            processor,
            output_path=resolved_plot if (plot_path is not None or not show_plot) else None,
            show=show_plot,
            title="KLens Reference Demo Deliberation",
        )
        if resolved_plot and (plot_path is not None or not show_plot):
            print(f"Spectral telemetry dashboard saved to: {os.path.abspath(resolved_plot)}")

    return 0


def run_file(
    transcript_file: str,
    output_csv: str,
    t_cert_limit: int,
    hierarchical_enabled: bool,
    max_factions: int,
    solver: str,
    plot_path: Optional[str] = None,
    show_plot: bool = False,
    plot_style: str = "telemetry",
    html_path: Optional[str] = None,
) -> int:
    if not os.path.exists(transcript_file):
        print(f"Error: file not found: {transcript_file}", file=sys.stderr)
        return 1

    alert_cfg = AlertConfig(t_cert_limit=t_cert_limit)
    hier_cfg = HierarchicalConfig(
        enabled=hierarchical_enabled,
        max_factions=max_factions,
        solver=solver,
    )
    processor = LiveTranscriptStreamProcessor(
        csv_log_path=output_csv,
        alert_config=alert_cfg,
        hierarchical_config=hier_cfg,
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
            pred_agent = metrics.predicted_next_orthogonal or metrics.predicted_next_rogue
            if pred_agent:
                pred_str = f"  Forecast (Round {r+1}): Projected Orthogonal -> {pred_agent}"
                if metrics.predicted_contraction_ratio is not None:
                    pred_str += f" | Contraction Rate -> {metrics.predicted_contraction_ratio:.2f}x"
                print(pred_str)
        else:
            print(f"Round {r:2d} | Insufficient history for spectral decomposition (< 2 rounds)")

    if processor.latest_metrics and processor.latest_metrics.topic_synthesis:
        print(format_semantic_synthesis_summary(processor.latest_metrics))

    print(f"\nProcessing complete. Output saved to: {os.path.abspath(output_csv)}")

    if plot_path is not None or show_plot:
        from klens.vis import render_spectral_dashboard
        if plot_path is not None and len(plot_path) > 0:
            resolved_plot = plot_path
        elif plot_path is not None:
            base = os.path.splitext(os.path.basename(transcript_file))[0]
            suffix = "_evolution.png" if plot_style == "evolution" else "_telemetry.png"
            resolved_plot = os.path.join("out", f"{base}{suffix}")
        else:
            resolved_plot = None

        render_spectral_dashboard(
            processor,
            output_path=resolved_plot,
            show=show_plot,
            title=f"KLens Telemetry: {os.path.basename(transcript_file)}",
            style=plot_style,
        )
        if resolved_plot:
            print(f"Spectral telemetry dashboard saved to: {os.path.abspath(resolved_plot)}")

    if html_path is not None:
        from klens.vis import export_interactive_timeline_html
        resolved_html = html_path if len(html_path) > 0 else os.path.join(
            "out", f"{os.path.splitext(os.path.basename(transcript_file))[0]}_interactive.html"
        )
        out_html = export_interactive_timeline_html(
            processor,
            output_path=resolved_html,
            title=f"KLens Interactive Evolution: {os.path.basename(transcript_file)}",
        )
        print(f"Interactive timeline HTML exported to: {out_html}")

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
        "--hierarchical",
        action="store_true",
        default=False,
        help="Enable hierarchical macro-faction coarse-graining for high-agent scaling",
    )
    demo_parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when hierarchical mode is enabled (default: 8)",
    )
    demo_parser.add_argument(
        "--solver",
        choices=["auto", "primal", "dual"],
        default="auto",
        help="Koopman solver mode: auto, primal, or dual (default: auto)",
    )
    demo_parser.add_argument(
        "--plot",
        nargs="?",
        const="out/demo_telemetry.png",
        default=None,
        help="Generate visual telemetry dashboard PNG (default: out/demo_telemetry.png)",
    )
    demo_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot in an interactive window",
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
        "--hierarchical",
        action="store_true",
        default=False,
        help="Enable hierarchical macro-faction coarse-graining for high-agent scaling",
    )
    run_parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when hierarchical mode is enabled (default: 8)",
    )
    run_parser.add_argument(
        "--solver",
        choices=["auto", "primal", "dual"],
        default="auto",
        help="Koopman solver mode: auto, primal, or dual (default: auto)",
    )
    run_parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        default=None,
        help="Generate visual telemetry dashboard PNG (default: out/<basename>_telemetry.png)",
    )
    run_parser.add_argument(
        "--style",
        choices=["telemetry", "diagnostic", "evolution", "shortcut"],
        default="telemetry",
        help="Plot style: telemetry, diagnostic, evolution, or shortcut (default: telemetry)",
    )
    run_parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        help="Export standalone interactive HTML5 evolution timeline (default: out/<basename>_interactive.html)",
    )
    run_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot in an interactive window",
    )

    plot_parser = subparsers.add_parser("plot", help="Load transcript file and generate visual dashboard")
    plot_parser.add_argument("transcript", help="Path to transcript JSON or JSONL file")
    plot_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path for output PNG (default: out/<basename>_telemetry.png)",
    )
    plot_parser.add_argument(
        "--style",
        choices=["telemetry", "diagnostic", "evolution", "shortcut"],
        default="telemetry",
        help="Plot style: telemetry, diagnostic, evolution, or shortcut (default: telemetry)",
    )
    plot_parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        help="Export standalone interactive HTML5 evolution timeline (default: out/<basename>_interactive.html)",
    )
    plot_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot in an interactive window",
    )
    plot_parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to also export CSV log",
    )
    plot_parser.add_argument(
        "--t-cert-limit",
        type=int,
        default=50,
        help="Certified deadline alert threshold (default: 50 rounds)",
    )
    plot_parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="Enable hierarchical macro-faction coarse-graining for high-agent scaling",
    )
    plot_parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when hierarchical mode is enabled (default: 8)",
    )
    plot_parser.add_argument(
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
            args.hierarchical,
            args.max_factions,
            args.solver,
            plot_path=args.plot,
            show_plot=args.show,
        )
    elif args.command == "run":
        return run_file(
            args.transcript,
            args.output,
            args.t_cert_limit,
            args.hierarchical,
            args.max_factions,
            args.solver,
            plot_path=args.plot,
            show_plot=args.show,
            plot_style=args.style,
            html_path=args.html,
        )
    elif args.command == "plot":
        temp_csv = args.csv or os.path.join("out", f"_tmp_{os.path.basename(args.transcript)}.csv")
        plot_target = args.output
        if not plot_target and not args.show and args.html is None:
            base = os.path.splitext(os.path.basename(args.transcript))[0]
            suffix = "_evolution.png" if args.style == "evolution" else "_telemetry.png"
            plot_target = os.path.join("out", f"{base}{suffix}")
        ret = run_file(
            args.transcript,
            temp_csv,
            args.t_cert_limit,
            args.hierarchical,
            args.max_factions,
            args.solver,
            plot_path=plot_target,
            show_plot=args.show,
            plot_style=args.style,
            html_path=args.html,
        )
        if args.csv is None and os.path.exists(temp_csv):
            try:
                os.remove(temp_csv)
            except OSError:
                pass
        return ret
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
