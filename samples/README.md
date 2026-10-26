# KLens Sample Transcripts

This directory contains reference datasets and discussion transcripts for testing, benchmarking, and demonstrating KLens.

---

## Available Samples

| File | Format | Agents | Rounds | Scenario & Deliberation Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| `scotus_bruen_oral_argument.json` | JSON | 12 | 6 | **US Supreme Court Oral Argument (*NYSRA v. Bruen*, 2021)**<br>370 real human dialogue turns between 9 Supreme Court Justices and 3 Advocates. Demonstrates early deadline spike ($T_{\text{cert}} = 179$ rounds), constitutional text/history faction alignment, and Chief Justice procedural isolation. |
| `fomc_monetary_policy_aug2011.json` | JSON | 25 | 5 | **Federal Reserve FOMC Monetary Policy Meeting (Aug 9, 2011)**<br>258 real human debate turns across 25 Fed Governors and Presidents debating calendar-date forward guidance ("mid-2013"). Isolates persistent dissenter (Richard Fisher), flags sluggish consensus ($\lambda_2 = 0.9841$), and reveals 3-way dissent deadlock. |
| `fda_aducanumab_adcom.json` | JSON | 17 | 5 | **FDA Advisory Committee Meeting (PCNS AdCom - Aducanumab, Nov 6, 2020)**<br>335 real human turns across 17 clinical voting members, biostatisticians, and agency officials evaluating conflicting Phase 3 trials (EMERGE 302 vs. ENGAGE 301). Demonstrates deadlock escalation ($T_{\text{cert}} = 148$ rounds) leading to an unprecedented 0–10–1 negative recommendation vote. |
| `real_llm_debate_des_moines.json` | JSON | 3 | 7 | **Real LLM Swarm Debate: Geography & Surfing (Hugging Face `Multi-Agent-LLMs/DEBATE`)**<br>35 turns across 3 expert LLM personas (*Geographer*, *Tourism Expert*, *Local Resident*). Real-world multi-round deliberation on geographical constraints with emergent consensus drift. |
| `real_llm_debate_bitcoin.json` | JSON | 3 | 7 | **Real LLM Swarm Debate: Physicality of Digital Assets (`Multi-Agent-LLMs/DEBATE`)**<br>35 turns across 3 expert LLM personas (*Cryptocurrency Expert*, *Digital Wallet Specialist*, *Physicist*). Deliberates hardware wallet semantics vs. intangible ledger assets. |
| `real_llm_debate_boolean_logic.json` | JSON | 3 | 7 | **Real LLM Swarm Debate: Boolean Algebra Computation (`Multi-Agent-LLMs/DEBATE`)**<br>35 turns across 3 expert LLM personas (*Computer Science Theorist*, *Computer Programmer*, *Mathematical Logician*). Formal logic debate mapping bitwise operations to universal computation. |
| `real_llm_debate_bee_sting.json` | JSON | 3 | 7 | **Real LLM Swarm Debate: Biological Mechanics (`Multi-Agent-LLMs/DEBATE`)**<br>35 turns across 3 expert LLM personas (*Entomologist*, *Biologist*, *Beekeeper*). Analysis of barbed stinger anatomy and autotomy consequences. |
| `clinical_slm_debate.json` | JSON | 6 | 4 | **Clinical SLM Deployment & Regulatory Audit (Synthetic Benchmark)**<br>Six specialist agents debate deploying Small Language Models in clinical triage. Demonstrates scripted contrarian dissenter (Agent 5). |
| `llm_safety_consensus.json` | JSON | 6 | 4 | **LLM AI Safety Evaluation Debate (Synthetic Benchmark)**<br>Six autonomous LLM agents debate judge models vs. human review. Demonstrates smooth contraction towards a hybrid architecture. |
| `boardroom_deadlock_polarization.jsonl` | JSONL | 8 | 4 | **Executive Committee Infrastructure Strategy (Synthetic Benchmark)**<br>Eight leaders debate cloud migration vs. on-prem. Demonstrates entrenched faction polarization and contrarian dissent. |
| `swarm_deliberation_24agents.jsonl` | JSONL | 24 | 3 | **24-Agent Microservice Swarm Deliberation (Synthetic Benchmark)**<br>24 agents debate API protocols. Demonstrates **Hierarchical Coarse-Graining** (`--hierarchical --max-factions 4`) and Dual Snapshot EDMD solving. |

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

# 3. FDA Advisory Committee Risk Deliberation (335 turns across 17 Clinical Experts)
klens run samples/fda_aducanumab_adcom.json \
  --plot out/adcom_telemetry.png \
  --output out/metrics_adcom.csv

# 4. Direct 4-Panel Visualization Plot (One-Liner)
klens plot samples/scotus_bruen_oral_argument.json --output out/scotus_telemetry.png
```

### 2. Analyze Real-World Multi-Agent LLM Debate Traces (`Multi-Agent-LLMs/DEBATE`)

These traces are sourced directly from the open-source [Hugging Face `Multi-Agent-LLMs/DEBATE`](https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE) benchmark repository, evaluating multi-persona reasoning dynamics across 7 continuous deliberation rounds:

```bash
# 1. Geography & Environment Debate (Des Moines Surfing)
klens run samples/real_llm_debate_des_moines.json --output out/real_llm_debate_des_moines.csv

# 2. Digital Assets & Hardware Wallets (Bitcoin Physicality)
klens run samples/real_llm_debate_bitcoin.json --output out/real_llm_debate_bitcoin.csv

# 3. Formal Logic & Computation (Boolean Algebra)
klens run samples/real_llm_debate_boolean_logic.json --output out/real_llm_debate_boolean_logic.csv

# 4. Biological Mechanics & Autotomy (Honey Bee Sting)
klens run samples/real_llm_debate_bee_sting.json --output out/real_llm_debate_bee_sting.csv
```

### 3. Synthetic Benchmark Datasets

```bash
# Multi-Agent LLM Safety Debate (JSON)
klens run samples/llm_safety_consensus.json --output out/metrics_safety.csv

# Boardroom Polarization & Deadlock (JSONL)
klens run samples/boardroom_deadlock_polarization.jsonl --output out/metrics_deadlock.csv --t-cert-limit 40

# High-Agent Swarm with Hierarchical Coarse-Graining (JSONL)
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
