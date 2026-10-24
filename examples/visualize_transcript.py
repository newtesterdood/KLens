#!/usr/bin/env python3
"""
Simple File Loader: Loads any multi-agent transcript (JSON or JSONL),
computes Koopman spectral consensus metrics across debate rounds,
and renders the 4-panel visual telemetry dashboard.

Usage:
    python examples/visualize_transcript.py [path/to/transcript.json] [--output out/plot.png] [--show]
"""

import argparse
import os
import sys

from klens.vis import plot_from_transcript_file


def main():
    parser = argparse.ArgumentParser(
        description="Load a discussion transcript and render visual Koopman spectral telemetry",
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default="samples/llm_safety_consensus.json",
        help="Path to transcript JSON or JSONL file (default: samples/llm_safety_consensus.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save output dashboard PNG (default: out/<name>_telemetry.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot in an interactive window",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="Enable hierarchical macro-faction coarse-graining for large swarms",
    )
    parser.add_argument(
        "--max-factions",
        type=int,
        default=8,
        help="Max faction centroids when hierarchical mode is active (default: 8)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        print(f"Error: file not found: {args.transcript}", file=sys.stderr)
        return 1

    print(f"Loading transcript: {args.transcript}...")
    processor, saved_path = plot_from_transcript_file(
        transcript_path=args.transcript,
        output_path=args.output,
        show=args.show,
        hierarchical=args.hierarchical,
        max_factions=args.max_factions,
    )

    m = processor.latest_metrics
    if m:
        print("\n=== Deliberation Spectral Summary ===")
        print(f"Rounds:                {m.round_id}")
        print(f"Participants:          {m.active_agents_count}")
        print(f"Subdominant λ2:        {m.lambda_2_magnitude:.4f}")
        print(f"Certified Deadline:    {m.certified_deadline_rounds} rounds")
        print(f"Detected Rogue:        {m.rogue_agent_detected}")
        print(f"Faction Clusters:      {m.factions_summary}")
        if m.predicted_next_rogue:
            print(f"Projected Next Rogue:  {m.predicted_next_rogue} (Contraction: {m.predicted_contraction_ratio:.2f}x)")

    if saved_path:
        print(f"\nVisual telemetry dashboard saved to: {os.path.abspath(saved_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
