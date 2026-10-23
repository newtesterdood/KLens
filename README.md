# KLens: Koopman Spectral Consensus Tracking for Multi-Agent LLM Debates

**KLens** is a Python library and CLI tool for real-time dynamical analysis of multi-agent LLM debates. It applies **Koopman Operator Theory** via **Extended Dynamic Mode Decomposition (EDMD)** with **Random Fourier Features (RFF)** to quantify consensus convergence, identify polarized factions, flag rogue divergent agents, certify convergence deadlines ($T_{\text{cert}}$), and trigger automated anomaly alerts.

---

## Mathematical Foundations

### 1. State Formulation & Centering
For $N$ agents across rounds $t \in \{1, \dots, T\}$, each agent $i$ emits state $z_i(t) \in \mathbb{R}^d$ (via TF-IDF or dense embeddings).
- **Consensus Centroid**: $\bar{z}(t) = \frac{1}{N} \sum_{i=1}^N z_i(t) \in \mathbb{R}^d$
- **Disagreement State**: $\Delta_i(t) = z_i(t) - \bar{z}(t)$
- **Flattened Disagreement Vector**: $\Delta(t) = [\Delta_1(t), \dots, \Delta_N(t)] \in \mathbb{R}^{Nd}$

### 2. Koopman Operator & Random Fourier Features
The Koopman operator $\mathcal{K}$ advances observables linearly: $\mathcal{K} g(\Delta(t)) = g(\Delta(t+1))$.
To capture non-linear dynamics, states are lifted via Random Fourier Features:
$$\Psi(\Delta) = \begin{bmatrix} 1 & \Delta & \cos(\Delta W_f + b_f) \end{bmatrix}$$
where $W_f \sim \mathcal{N}\left(0, \frac{0.7}{\sqrt{Nd}}\right)$ and $b_f \sim \mathcal{U}(0, 2\pi)$.

### 3. Finite-Dimensional Approximation via Ridge Regression
Given data pairs $\Psi_t = \Psi(\Delta(t))$ and $\Psi_t^+ = \Psi(\Delta(t+1))$:
$$G = \Psi^T \Psi + \rho I, \quad A = \Psi^T \Psi^+$$
$$\hat{K} = G^{-1} A$$

### 4. Sub-Dominant Eigenvalue & Certified Deadline
From the spectrum of $\hat{K}$, filter trivial modes ($|\mu| < 0.985$) to isolate the sub-dominant eigenvalue $\lambda_2$.
The decay envelope satisfies $\|\Delta(t)\| \le C |\lambda_2|^t$.
For convergence tolerance $\epsilon > 0$ (default $\epsilon = 0.05$):
$$T_{\text{cert}} = \left\lceil \frac{\ln(1/\epsilon)}{-\ln|\lambda_2|} \right\rceil$$

### 5. Rogue Agent Detection & Faction Clustering
- **Rogue Detection**: $\arg\max_i \frac{1}{T} \sum_{t=1}^T \|\Delta_i(t)\|_2$
- **Faction Clustering**: Pairwise cosine similarity on non-rogue final states with threshold $\ge 0.65$.

---

## Installation

```bash
git clone https://github.com/newtesterdood/KLens.git
cd KLens
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quickstart

### Run Built-in 6-Agent Simulation

```bash
klens demo --output spectral_debate_log.csv --t-cert-limit 50
```

### Analyze a Transcript File (JSON or JSONL)

```bash
klens run path/to/transcript.json --output out/metrics.csv
```

---

## Python API Usage

```python
from klens import LiveTranscriptStreamProcessor, AlertConfig

# 1. Configure processor, alert thresholds, and scaling
config = AlertConfig(
    t_cert_limit=50,             # Alert if certified deadline > 50 rounds
    lambda_2_warning_limit=0.95, # Warning if spectral radius is sluggish
    lambda_2_critical_limit=0.99 # Critical alert if near divergence
)

# Configure scaling: toggle compression on or off
comp_config = CompressionConfig(
    enabled=False,        # Set False to "go big" (full compute on all agents), or True for macro-faction compression
    max_factions=8,       # Max macro-faction centroids when compression is enabled
    solver="auto",        # "auto" dynamically selects Snapshot Dual EDMD when S < P to prevent O(P^3) memory blowup
)

processor = LiveTranscriptStreamProcessor(
    csv_log_path="spectral_debate_log.csv",
    alert_config=config,
    compression_config=comp_config,
)

# 2. Register callbacks (optional)
processor.add_alert_callback(lambda a: print(f"[{a.level.value}] {a.code}: {a.message}"))

# 3. Stream turns in real-time
processor.ingest_turn(round_id=1, speaker="Agent_1", text="Judge models must handle all tasks.")
processor.ingest_turn(round_id=1, speaker="Agent_2", text="Human review is strictly mandatory.")

# 4. Trigger round evaluation event
metrics = processor.process_round_event(current_round=2)
if metrics:
    print(f"Sub-dominant eigenvalue: {metrics.lambda_2_magnitude:.4f}")
    print(f"Certified deadline:     {metrics.certified_deadline_rounds} rounds")
    print(f"Rogue agent:            {metrics.rogue_agent_detected}")
    print(f"Factions:               {metrics.factions_summary}")
    print(f"Compressed mode active: {metrics.is_compressed}")

```

---

## Scaling to Massive Agent Populations

KLens supports scaling across two operational regimes:

1. **Uncompressed Mode ("Go Big" / Default `enabled=False`)**:
   - Models every individual agent directly in the lifted observable space.
   - Ideal for small to medium groups ($N \le 50$) where full computational resolution across every agent is desired.
   - When $S < P$, the built-in **Dual (Snapshot) Solver** automates eigendecomposition via the snapshot Gram matrix $G_{\text{snap}} = \Psi \Psi^T \in \mathbb{R}^{S \times S}$, eliminating the $\mathcal{O}(P^3)$ memory/compute crash.

2. **Hierarchical Macro-Faction Compression (`enabled=True`)**:
   - Compresses $N$ agent trajectories into $K = \min(N, \text{max\_factions})$ macro-faction centroids.
   - Runs Koopman EDMD on the macro-centroid dynamics to certified convergence ($T_{\text{cert}}$).
   - **Zero Loss in Rogue Detection**: Computes individual agent deviations in an $\mathcal{O}(Nd)$ streaming residual pass against the consensus centroid, guaranteeing that outlier rogue agents are accurately detected across all $N$ agents.

### CLI Scaling Options

```bash
# Go big: full uncompressed resolution
klens demo --solver auto

# Scale up: enable faction compression with 4 macro-centroids
klens demo --compress --max-factions 4
```

---


## Output CSV Schema

Entries are appended to `spectral_debate_log.csv` at each round:

| Column | Type | Description |
| :--- | :--- | :--- |
| `round` | Integer | Debate round index |
| `active_agents_count` | Integer | Number of participating agents |
| `lambda_2_magnitude` | Float | Magnitude of subdominant eigenvalue $|\lambda_2|$ |
| `certified_deadline_rounds`| Integer | Certified deadline $T_{\text{cert}}$ |
| `rogue_agent_detected` | String | Agent with highest average disagreement norm |
| `faction_count` | Integer | Number of non-rogue agent clusters |
| `factions_summary` | String | Faction membership groupings |
| `centroid_top_keywords` | String | Semicolon-delimited top centroid keywords |

Example row:
```csv
round,active_agents_count,lambda_2_magnitude,certified_deadline_rounds,rogue_agent_detected,faction_count,factions_summary,centroid_top_keywords
2,6,0.9312,43,Agent_1,5,F1:(Agent_2) | F2:(Agent_3) | F3:(Agent_4) | F4:(Agent_5) | F5:(Agent_6),judge;models;cost;auditing
3,6,0.9608,75,Agent_2,5,F1:(Agent_1) | F2:(Agent_3) | F3:(Agent_4) | F5:(Agent_5) | F6:(Agent_6),hybrid;compromise;selective;slm
```

---

## Automated Alert Rules

`AlertMonitor` detects pathological debate dynamics:

1. **`T_CERT_EXCEEDED`**: $T_{\text{cert}}$ exceeds `t_cert_limit` rounds.
2. **`DIVERGENT_DYNAMICS`**: $|\lambda_2| \ge 1.0$ (non-contractive / divergent trajectory).
3. **`CRITICAL_SPECTRAL_RADIUS`**: $|\lambda_2| \ge 0.99$ (extreme stagnation).
4. **`SLOW_CONVERGENCE`**: $|\lambda_2| \ge 0.95$.
5. **`PERSISTENT_ROGUE`**: Same agent remains rogue for $K$ consecutive rounds.

---

## Verification & Tests

Run the full test suite with:

```bash
pytest -v
```


research references:

The primary papers that serve as the foundation for this code and its theoretical architecture are:

---

### 1. Primary Algorithmic Source
* **Paper:** *"Certifying Collective Reasoning in Multi-Agent Systems via Koopman Spectral Analysis"* — Nuzhat Khan and Indrakshi Dey [1, 2].
* **Core Concepts Used in the Code:**
  * **State Centering:** Subtracting the mean consensus vector (\\(\delta(t) = z(t) - \bar{z}(t)\\)) to remove trivial unit eigenvalues and isolate disagreement dynamics [3, 4].
  * **Random Fourier Feature (RFF) Lifting:** Lifting nonlinear belief trajectories into an observable feature space \\(\Psi(\delta)\\) [4, 5].
  * **Extended Dynamic Mode Decomposition (EDMD):** Solving for the finite-dimensional Koopman matrix estimate \\(\hat{\mathbf{K}}\\) via regularized ridge regression [3, 4, 6].
  * **Sub-dominant Eigenvalue Extraction (\\(\lambda_2\\)):** Filtering out residual near-unit modes (\\(|\lambda| \ge 0.985\\)) and isolating the slowest-decaying disagreement mode [4, 7].
  * **Certified Convergence Deadline (\\(T_{\text{cert}}\\)):** Calculating \\(T_{\text{cert}} = \left\lceil \frac{\ln(1/\epsilon)}{-\ln |\lambda_2|} \right\rceil\\) to guarantee consensus bounds [7, 8].
  * **Mode Attribution & Faction Diagnostics:** Using sub-dominant eigenvectors to identify slow-decaying agent partitions and metastable factions [7, 9].

---

### 2. Scaling, Diversity, & Faction Compression Source
* **Paper:** *"Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity"* — Yingxuan Yang et al. [10, 11].
* **Core Concepts Used in the Code:**
  * **Effective Channel Count (\\(K^*\\)):** Quantifying non-redundant reasoning paths rather than raw agent count \\(N\\) [12, 13].
  * **Diminishing Returns of Scale:** Demonstrating why scaling homogeneous agents leads to correlated output redundancy [14, 15].
  * **Centroid & PCA Compression:** Exploiting the principle that massive agent populations (\\(N = 10,000+\\)) naturally collapse into a small number of effective reasoning channels (factions), enabling \\(O(N)\\) scaling for thousand-agent traces [12, 16].

---

### 3. Disagreement & Selector Bottleneck Source
* **Paper:** *"When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines"* — Artem Maryanskyy et al. [17, 18].
* **Core Concepts Used in the Code:**
  * **Divergence & Selection Dynamics:** Analyzing how agent disagreement creates bottlenecks and demonstrating why pruning uncooperative or rogue entities dramatically accelerates consensus pipeline performance [17, 18].
