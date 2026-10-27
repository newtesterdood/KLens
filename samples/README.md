# KLens Sample Transcripts

This directory contains reference datasets and discussion transcripts for testing, benchmarking, and demonstrating KLens.

---

## Available Samples

| File | Format | Agents | Rounds | Scenario & Deliberation Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| `scotus_bruen_oral_argument.json` | JSON | 12 | 6 | **US Supreme Court Oral Argument (*NYSRA v. Bruen*, 2021)**<br>370 real human dialogue turns between 9 Supreme Court Justices and 3 Advocates. Demonstrates early deadline spike ($T_{\text{cert}} = 179$ rounds), constitutional text/history faction alignment, and Chief Justice procedural isolation. |
| `fomc_monetary_policy_aug2011.json` | JSON | 25 | 5 | **Federal Reserve FOMC Monetary Policy Meeting (Aug 9, 2011)**<br>258 real human debate turns across 25 Fed Governors and Presidents debating calendar-date forward guidance ("mid-2013"). Isolates persistent dissenter (Richard Fisher), flags sluggish consensus ($\lambda_2 = 0.9841$), and reveals 3-way dissent deadlock. |
| `fda_aducanumab_adcom.json` | JSON | 17 | 5 | **FDA Advisory Committee Meeting (PCNS AdCom - Aducanumab, Nov 6, 2020)**<br>335 real human turns across 17 clinical voting members, biostatisticians, and agency officials evaluating conflicting Phase 3 trials (EMERGE 302 vs. ENGAGE 301). Demonstrates deadlock escalation ($T_{\text{cert}} = 148$ rounds) leading to an unprecedented 0–10–1 negative recommendation vote. |
| `delaware_merger_deliberation.json` | JSON | 14 | 5 | **Delaware Court of Chancery M&A Deliberation (*Twitter v. Musk*, 2022)**<br>56 high-stakes turns across board directors, management, legal counsel (Wachtell Lipton), and bankers evaluating Musk's $54.20 unsolicited buyout. Tracks poison pill adoption under *Unocal*, Revlon duties, deadlock during fake accounts/MAE dispute ($T_{\text{cert}} = 152$ rounds), and rapid contraction to specific performance resolution ($\lambda_2 = 0.7592, T_{\text{cert}} = 11$). |
| `puc_infrastructure_hearing.json` | JSON | 14 | 5 | **California Public Utilities Commission Rate Hearing (CPUC NEM 3.0 / D.22-12-056)**<br>45 real-world regulatory turns across 14 commissioners, judges, ratepayer advocates (TURN, Cal Advocates), utilities (SCE, PG&E), and solar coalitions (CALSSA, NRDC). Tracks $3.4B–$4.0B cost-shift debate, CAISO evening duck curve ramping, battery storage pairing mandates, deadlock ($T_{\text{cert}} = 127$ rounds), and unanimous 5–0 decision adoption ($\lambda_2 = 0.6119, T_{\text{cert}} = 7$). |
| `city_council_policy_meeting.json` | JSON | 14 | 5 | **New York City Council Stated Meeting (Contested 'City of Yes' Zoning & Infrastructure Package)**<br>48 municipal legislative turns across 14 council leaders, land use chairs, and borough representatives. Deliberates Universal Affordability Preference (UAP), outer-borough parking minimum elimination, sewer/stormwater capacity, and deadlock ($T_{\text{cert}} = 102$ rounds) resolved by a landmark $5.0 Billion capital infrastructure agreement leading to a 31–18 stated roll-call vote ($\lambda_2 = 0.6361, T_{\text{cert}} = 7$). |
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

# 4. Delaware Chancery M&A Merger Deliberation (56 turns across 14 Directors & Advisors)
klens run samples/delaware_merger_deliberation.json \
  --plot assets/delaware_merger_telemetry.png \
  --output out/metrics_delaware_merger.csv

# 4b. Generate Multi-Round Temporal Evolution Grid & Interactive HTML Timeline
klens plot samples/delaware_merger_deliberation.json \
  --style evolution \
  --output assets/delaware_merger_evolution.png \
  --html out/delaware_merger_interactive.html

# 5. California PUC Infrastructure Rate Hearing (45 turns across 14 Commissioners & Advocates)
klens run samples/puc_infrastructure_hearing.json \
  --plot assets/puc_hearing_telemetry.png \
  --output out/metrics_puc_hearing.csv

# 6. NYC City Council Stated Meeting (48 turns across 14 Council Members & Chairs)
klens run samples/city_council_policy_meeting.json \
  --plot assets/city_council_telemetry.png \
  --output out/metrics_city_council.csv

# 7. Direct 4-Panel Visualization Plot (One-Liner)
klens plot samples/delaware_merger_deliberation.json --output out/delaware_telemetry.png
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
