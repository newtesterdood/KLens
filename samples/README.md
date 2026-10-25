# KLens Sample Transcripts

This directory contains reference datasets and discussion transcripts for testing, benchmarking, and demonstrating KLens.

---

## Available Samples

| File | Format | Agents | Rounds | Scenario & Deliberation Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| `scotus_bruen_oral_argument.json` | JSON | 12 | 6 | **US Supreme Court Oral Argument (*NYSRA v. Bruen*, 2021)**<br>370 real human dialogue turns between 9 Supreme Court Justices and 3 Advocates. Demonstrates early deadline spike ($T_{\text{cert}} = 179$ rounds), constitutional text/history faction alignment, and Chief Justice procedural isolation. |
| `fomc_monetary_policy_aug2011.json` | JSON | 25 | 5 | **Federal Reserve FOMC Monetary Policy Meeting (Aug 9, 2011)**<br>258 real human debate turns across 25 Fed Governors and Presidents debating calendar-date forward guidance ("mid-2013"). Isolates persistent dissenter (Richard Fisher), flags sluggish consensus ($\lambda_2 = 0.9841$), and reveals 3-way dissent deadlock. |
| `clinical_slm_debate.json` | JSON | 6 | 4 | **Clinical SLM Deployment & Regulatory Audit (Rogue Benchmark)**<br>Six specialist agents debate deploying Small Language Models in clinical triage. Isolates contrarian dissenter (Agent 5) and demonstrates spectral faction identification. |
| `llm_safety_consensus.json` | JSON | 6 | 4 | **LLM AI Safety Evaluation Debate**<br>Six autonomous LLM agents debate judge models vs. human review. Demonstrates smooth contraction towards a hybrid architecture. |
| `boardroom_deadlock_polarization.jsonl` | JSONL | 8 | 4 | **Executive Committee Infrastructure Strategy**<br>Eight leaders debate cloud migration vs. on-prem. Demonstrates entrenched faction polarization, high spectral radius ($\|\lambda_2\| \ge 0.97$), and persistent contrarian dissent. |
| `swarm_deliberation_24agents.jsonl` | JSONL | 24 | 3 | **24-Agent Microservice Swarm Deliberation**<br>24 agents debate API protocols (gRPC vs. GraphQL vs. REST). Demonstrates **Hierarchical Coarse-Graining** (`--hierarchical --max-factions 4`) and Dual Snapshot EDMD solving. |

---

## Running the Samples

### 1. Analyze Real-World Human Discussion Traces

For full methodology and empirical findings on real human deliberation, see the [Human Deliberation Benchmarks Guide](../docs/HUMAN_DELIBERATION_BENCHMARKS.md).

```bash
# 1. SCOTUS Oral Argument (370 turns across 12 Justices & Counsel)
klens run samples/scotus_bruen_oral_argument.json \
  --plot out/scotus_bruen_telemetry.png \
  --output out/metrics_bruen.csv

# 2. Federal Reserve FOMC Rate Deliberation (258 turns across 25 Fed Leaders)
klens run samples/fomc_monetary_policy_aug2011.json \
  --plot out/fomc_telemetry.png \
  --output out/metrics_fomc.csv \
  --t-cert-limit 50

# 3. Direct 4-Panel Visualization Plot (One-Liner)
klens plot samples/scotus_bruen_oral_argument.json --output out/scotus_telemetry.png
```

### 2. Multi-Agent LLM Safety Debate (JSON)

```bash
klens run samples/llm_safety_consensus.json --output out/metrics_safety.csv
```

### 3. Boardroom Polarization & Deadlock (JSONL)

```bash
klens run samples/boardroom_deadlock_polarization.jsonl --output out/metrics_deadlock.csv --t-cert-limit 40
```

### 4. High-Agent Swarm with Hierarchical Coarse-Graining (JSONL)

```bash
# Run with 4 macro-faction centroids using the Dual Snapshot solver
klens run samples/swarm_deliberation_24agents.jsonl \
  --hierarchical \
  --max-factions 4 \
  --solver auto \
  --output out/metrics_swarm.csv
```

---

## Transcript Schema Specification

KLens supports both JSON and JSONL formats:

### Format A: JSON Array (`.json`)
```json
[
  {
    "round": 1,
    "speaker": "Agent_A",
    "text": "Statement for round 1."
  },
  {
    "round": 1,
    "speaker": "Agent_B",
    "text": "Counter-statement for round 1."
  }
]
```

### Format B: Line-Delimited JSON (`.jsonl`)
```json
{"round": 1, "speaker": "Agent_A", "text": "Statement for round 1."}
{"round": 1, "speaker": "Agent_B", "text": "Counter-statement for round 1."}
{"round": 2, "speaker": "Agent_A", "text": "Statement for round 2."}
{"round": 2, "speaker": "Agent_B", "text": "Counter-statement for round 2."}
```

### Required Fields per Turn

* `round` (or `round_id`): Integer index (1-indexed). Turns within the same round are grouped into that round's semantic state.
* `speaker`: String identifier for the agent or participant.
* `text`: Content spoken or generated during this turn.
