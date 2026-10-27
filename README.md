# KLens: Koopman Spectral Consensus Workbench & Trace Analysis Platform

**KLens** is an experimental research platform and trace analysis workbench developed to investigate whether **Koopman Operator Theory** can decode the latent dynamical geometry of multi-party discussion—enabling real-time tracking of text-based debate, issue emergence, faction crystallization, and consensus resolution.

---

### The Paradigm: Pointing the Lens at the Mirror

Recent foundational research has introduced rigorous mathematical frameworks to certify collective reasoning and model diversity in artificial multi-agent systems:
* **Khan & Dey (2024)** introduced Koopman spectral certification, state-centering, and Random Fourier Feature lifting to bound consensus deadlines in multi-agent networks.
* **Yang et al. (2024)** proved that massive agent swarms suffer from diminishing returns and collapse into a small number of effective reasoning channels (factions).
* **Maryanskyy et al. (2024)** established the selection bottleneck, proving that unresolved contrarian disagreements stall decision pipelines.

**KLens was built to point the lens made possible by their research directly back at the mirror.** Rather than treating discussion as static textual outputs, KLens treats deliberative processes—across synthetic LLMs, human deliberation traces, and hybrid human-AI debates—as continuous-time nonlinear dynamical systems. By projecting non-linear belief trajectories into an observable Hilbert space, KLens investigates whether operator spectra can uncover the universal dynamics of how collectives form factions, isolate contrarians, and converge on key issue resolutions.

> 📖 **Research Perspective**: Read the comprehensive cross-disciplinary whitepaper on this emerging area of study: [**Spectral Telemetry and Operator-Theoretic Foundations for Multi-Agent Deliberation**](docs/RESEARCH_PERSPECTIVE.md).

---

## Core Deliberation Inquiries


The KLens workbench provides experimental tooling to investigate four fundamental facets of multi-party deliberation:

1. **Text-Based Debate Trajectories**: Ingesting turn-by-turn dialogue embeddings to reconstruct centered disagreement manifolds $\Delta(t) = Z(t) - \bar{Z}(t)$ across arbitrary rounds.
2. **Issue Identification & Resolution**: Tracking consensus centroids $\bar{z}(t)$ as they drift through semantic space, isolating the exact coordinates and keywords where compromise solidifies.
3. **Faction Formation & Polarization**: Employing sub-dominant eigenvector mode attribution and cosine clustering to reveal emergent metastable ideological blocs before they openly declare opposition.
4. **Rogue Entity & Bottleneck Isolation**: Tracking micro-level residual deviations to pinpoint contrarian entities that anchor discussion, enabling principled pruning or mediation.


## Visual Telemetry & Spectral Diagnostics

KLens visualizes the dynamical geometry and operator spectra of multi-party deliberation across three complementary diagnostic dashboards:

### 1. Multi-Agent Spectral Analysis: Faction Identification & Rogue Detection

The primary diagnostic dashboard isolates contrarian agents, measures disagreement decay relative to the theoretical isolation threshold, visualizes faction cohesion via cosine similarity, and synthesizes operational spectral certificates:

![Multi-Agent Spectral Analysis](assets/multi_agent_spectral_analysis.png)

* **Panel 1: PCA Opinion Space with Landmark Centroid**: 2D projection of agent belief states alongside the evolving consensus centroid $\bar{z}(t)$, highlighting isolated rogue agents.
* **Panel 2: Agent Disagreement Magnitude $\|z_i(t) - \bar{z}(t)\|$**: Turn-by-turn disagreement norm plotted against the Rogue Isolation Threshold ($\mu + 1.5\sigma$).
* **Panel 3: Pairwise Cosine Similarity Heatmap**: 6x6 cohesion matrix revealing intra-faction alignment and inter-faction antagonism with exact cosine coefficients.
* **Panel 4: Monospace Spectral Diagnostic & Semantic Summary**: Real-time summary detailing the primary Center of Disagreement, Faction Topics & verbatim Positions, Rogue Dissent Position, Consensus Centroid statement, subdominant eigenvalue $\lambda_2$, certified deadline $T_{\text{cert}}$, and counterfactual pruning impact.

---

### 2. Consensus Acceleration via Centroid Token Injection (Spectral Shortcuts)

When deliberation stalls due to stubborn factions or rogue dissenters, KLens models the impact of injecting consensus centroid tokens $\bar{z}(t)$ directly into agent prompt contexts:

![Consensus Acceleration via Centroid Token Injection](assets/centroid_shortcut_comparison.png)

* **Baseline Deliberation (Sluggish Decay)**: Slow natural consensus with $|\lambda_2| = 0.941$ yielding an extended certified deadline of $T_{\text{cert}} = 49$ rounds.
* **Centroid Shortcut (Rapid Collapse)**: Dynamic injection of the consensus centroid token at round 8 breaks ideological deadlock, shifting the spectral radius to $|\lambda_2| = 0.724$ and contracting the certified deadline to just 13 rounds ($3.8\times$ speedup).

---

### 3. Unit Disk Spectrum & Convergence Envelope

For low-level dynamical systems tracking, KLens plots the Koopman eigenvalues on the complex unit disk and traces the empirical disagreement norm against the theoretical exponential decay envelope:

![KLens Telemetry Dashboard](assets/klens_telemetry_dashboard.png)

* **Koopman Spectrum on Complex Unit Disk (Panel A)**: Maps all eigenvalues $\{\mu_j\} \subset \mathbb{C}$ of the empirical transfer operator $\hat{K}$ relative to the unit disk ($|\mu| = 1.0$). Modes clustering near the unit circle ($|\lambda_2| \ge 0.985$) visually expose ideological deadlock and slow consensus decay.
* **Certified Convergence Deadline Envelope (Panel B)**: Plots empirical disagreement norm $\|\Delta(t)\|$ across debate rounds against the theoretical decay envelope $C |\lambda_2|^t$, with certified deadline $T_{\text{cert}}$ at $\epsilon = 0.05$.
* **2D Semantic Manifold & Centroid Drift (Panel C)**: Projects lifted state trajectories into principal observable coordinates (PC 1 vs. PC 2) and tracks the drift of the **Consensus Centroid $\bar{z}(t)$** (dashed gold path).
* **Forward Trajectory Projection & Contraction Cone (Panel D)**: Projects agent belief states forward $k$ rounds ($\hat{\Psi}_{t+k} = \Psi_t \hat{K}^k$) to forecast whether subsequent rounds will contract toward consensus or expand into deadlock.

---

## Human-Understandable Semantic Topic Synthesis

While Koopman operators track the continuous-time geometry of belief vectors in observable space, human facilitators and decision-makers require plain-English semantic clarity. KLens automatically decodes abstract state manifolds into four interpretable semantic layers:

1. **Center of Disagreement (Principal Contention Axis)**: Identifies the core ideological trade-off dividing the debate (via SVD on the centered disagreement matrix $\Delta(T) = U \Sigma V^T$).
2. **Faction Topics & Content of Positions**: Formulates the human-understandable concept each faction represents, accompanied by the verbatim representative statement spoken by its members.
3. **Consensus Centroid Concept & Statement**: Translates the moving consensus landmark $\bar{z}(t)$ into an authentic, readable resolution statement extracted from the discussion.
4. **Rogue Dissent Stance & Objection**: Identifies the exact contrarian stance and verbatim dissent statement anchoring consensus progress.

### Live Benchmark Outputs

#### 1. Clinical SLM Deployment Benchmark (`samples/clinical_slm_debate.json`)
Six specialist agents deliberate deploying Small Language Models in hospital triage. A contrarian regulatory entity objects to automated clinical decisions:

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Zero AI vs. Selective Judge Auditing
  "The primary debate axis centers on the trade-off between zero ai and selective judge auditing."
• Consensus Centroid Concept: Selective Judge Auditing & Hybrid SLM Orchestration
  "Agreement on hybrid SLM orchestration with selective judge auditing for clinical rigor."

--- Faction Topics & Positions ---
• Faction 1 (5 members): Selective Judge Auditing
  Position: "Agreement on hybrid SLM orchestration with selective judge auditing for clinical rigor."

--- Rogue Dissent Position (Agent 5 (Regulatory/Adversary)) ---
• Stance:   Zero AI
  Position: "AI must not touch clinical decisions under any condition."
```

#### 2. Executive Boardroom Infrastructure Benchmark (`samples/boardroom_deadlock_polarization.jsonl`)
Eight C-suite leaders debate public cloud migration vs. on-prem hardware. Seven leaders converge on hybrid architecture while the Infrastructure Director dissents:

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Cloud Services vs. Hybrid Implementation
  "The primary debate axis centers on the trade-off between cloud services and hybrid implementation."
• Consensus Centroid Concept: Hybrid Implementation & Hybrid Cloud
  "Security boundary approved for hybrid implementation."

--- Faction Topics & Positions ---
• Faction 1 (CFO): Capital Allocation Approved
  Position: "Capital allocation approved for hybrid implementation."
• Faction 2 (CISO): Security Boundary Approved
  Position: "Security boundary approved for hybrid implementation."
• Faction 3 (Director_Infra): Cloud Services Dissent
  Position: "No cloud services under any circumstances."
• Faction 4 (Lead_Architect): Approve Implementation
  Position: "Hybrid design finalized; seven of eight leaders approve implementation."
• Faction 5 (Lead_DBA): Database Replication
  Position: "Local database replication verified for hybrid boundary."
• Faction 6 (VP_Engineering): Architecture Rollout
  Position: "We are calling for final approval on the hybrid architecture rollout plan."
• Faction 7 (VP_Product): Product Releases
  Position: "Product roadmaps scheduled for hybrid cloud release."

--- Rogue Dissent Position (Head_DevOps) ---
• Stance:   Staged Deployment
  Position: "Infrastructure automation scripts staged for hybrid deployment."
```

#### 3. US Supreme Court Oral Argument Benchmark (`samples/scotus_bruen_oral_argument.json`)
370 verbatim human turns across 12 speakers (9 Supreme Court Justices and 3 Advocates in *NYSRA v. Bruen*). Early debate rounds exhibit severe friction ($\lambda_2 = 0.9833, T_{\text{cert}} = 179$ rounds) before collapsing toward the constitutional *Heller* text/history standard ($\lambda_2 = 0.7126, T_{\text{cert}} = 9$ rounds):

![SCOTUS Bruen Oral Argument Telemetry](assets/scotus_bruen_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Second Amendment vs. Other States
  "The primary debate axis centers on the trade-off between second amendment and other states."
• Consensus Centroid Concept: Second Amendment & I Think
  "As to the general question about Heller, we agree completely that the Court ought to apply the method from Heller, which we, like I think all the parties, take to be look to the text, history, and tradition of the Second Amendment right..."

--- Faction Topics & Positions ---
• Faction 3 (Brett M. Kavanaugh, Brian H. Fletcher): History and Tradition
  Position: "Clement responded to what you're saying on history and tradition by saying you have to look at carry laws more generally."
• Faction 6 (Neil Gorsuch): Text and History
  Position: "Others say, yes, text and history is appropriate, but we're not going to extend the Heller right until and unless the Court first does so through its own text and history analysis."
• Faction 7 (Paul D. Clement, Samuel A. Alito, Jr.): Second Amendment
  Position: "In a country with the Second Amendment as a fundamental right, simply having more firearms cannot be a problem and can't be a government interest just to put a cap on the number of firearms."

--- Rogue Dissent Position (John G. Roberts, Jr.) ---
• Stance:   Justice Gorsuch
  Position: "Justice Thomas?"  [Flagged: procedural moderator turns lie on an orthogonal subspace]
```

#### 4. Federal Reserve FOMC Monetary Policy Benchmark (`samples/fomc_monetary_policy_aug2011.json`)
258 substantive turns across 25 Federal Reserve policymakers debating explicit calendar-date forward guidance ("mid-2013"). KLens tracks mounting ideological resistance ($\lambda_2 = 0.9841, T_{\text{cert}} = 187$ rounds), isolates the historic 3-way dissent, and pinpoints Dallas Fed President Richard Fisher as the persistent contrarian:

![FOMC Monetary Policy Telemetry](assets/fomc_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Language From Alternative vs. Monetary Policy
  "The primary debate axis centers on the trade-off between language from alternative and monetary policy."
• Consensus Centroid Concept: Language From Alternative & Alternative B
  "If we choose not to clarify our anticipation, as in alternative B without any additional language, I think we need to explain why more clearly."

--- Faction Topics & Positions ---
• Faction 10 (Roll Call Vote): Historic 3-Way Dissent
  Position: "Bernanke: Yes, Dudley: Yes, Duke: Yes, Evans: Yes, Fisher: No, Kocherlakota: No, Plosser: No, Raskin: Yes, Tarullo: Yes, Yellen: Yes"
• Faction 13 (Esther George / Charles Evans): Alternative A Language
  Position: "If you’re going to take three dissents, then I think we should go to the language of alternative A and take out paragraph 4..."
• Faction 17 (Daniel Tarullo, Janet Yellen): Monetary Policy Tools
  Position: "I think that first sentence really does help us a little bit with that “range of policy tools.”"

--- Rogue Dissent Position (Fisher) ---
• Stance:   Persistent Dissenter (Dallas Fed President)
  Position: "Chairman, could you read the full sentence for me, please?"
```

#### 5. FDA Advisory Committee Risk Deliberation Benchmark (`samples/fda_aducanumab_adcom.json`)
335 turns across 17 clinical voting members, biostatisticians, and agency leadership evaluating conflicting Phase 3 trials for Aducanumab. KLens charts escalating deadlocked tension ($\lambda_2 = 0.9799, T_{\text{cert}} = 148$ rounds) culminating in an unprecedented 0–10–1 negative recommendation vote:

![FDA AdCom Telemetry](assets/fda_adcom_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: I Voted vs. I Just
  "The primary debate axis centers on the trade-off between i voted and i just."
• Consensus Centroid Concept: Thank You & I Think
  "But again I think that's the reason Thank you."

--- Faction Topics & Positions ---
• Faction 1 (Alexander, Kesselheim, Thambisetty): Efficacy Rejection
  Position: "don't think that the evidence in Study 302 provides substantial evidence of efficacy of effectiveness of this drug and the way that 301 was sliced and diced to support that further justifies that."
• Faction 2 (Duda, Jones, Onyike): Pharmacodynamic Effect Support
  Position: "I voted yes because I do believe there's strong evidence of a pharmacodynamic effect on Alzheimer disease pathology, specifically amyloid pathology..."
• Faction 5 (Samantha Budd Haeberlein / Biogen): Sponsor Defense
  Position: "That was absolutely the case through our investigations, that we treated both studies equally, and the resulting output of those investigations are indeed that Study 302 is robust and that Study 301 is a negative study."

--- Rogue Dissent Position (Bonner) ---
• Stance:   Vote Results (Designated Federal Officer)
  Position: "We will now move voting numbers to the voting breakout room to vote only." [Procedural outlier]
```

#### 6. Delaware Court of Chancery M&A Deliberation Benchmark (`samples/delaware_merger_deliberation.json`)
56 turns across Twitter directors, executive management, Wachtell Lipton legal counsel, and investment bankers evaluating Elon Musk's unsolicited $54.20/share ($44.0 Billion) buyout proposal. KLens tracks poison pill adoption under *Unocal*, mounting deadlock during the spam bot / Material Adverse Effect (MAE) dispute ($T_{\text{cert}} = 152$ rounds), and rapid phase collapse to specific performance contract enforcement ($\lambda_2 = 0.7592, T_{\text{cert}} = 11$ rounds):

![Delaware Merger Deliberation Telemetry](assets/delaware_merger_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Delaware Law vs. 20 Cash
  "The primary debate axis centers on the trade-off between delaware law and 20 cash."
• Consensus Centroid Concept: Delaware Law & Law and Delaware
  "This resolution represents an extraordinary vindication of independent corporate governance and fiduciary duty under Delaware law."

--- Faction Topics & Positions ---
• Faction 4 (Durban, Pichette, Zoellick): Delaware Law & Shareholder Liquidity
  Position: "This resolution represents an extraordinary vindication of independent corporate governance and fiduciary duty under Delaware law."
• Faction 5 (Elon Musk): Information Access & Bot Diligence Demands
  Position: "Twitter has actively obstructed my legal rights under Section 6.4 of the merger agreement..."
• Faction 7 (Martin Lipton): Material Adverse Effect (MAE) Doctrine
  Position: "Under Delaware Chancery law, the bar for establishing a Material Adverse Effect is extraordinarily high."

--- Rogue Dissent Position (William Savitt) ---
• Stance:   Specific Performance
  Position: "Faced with the certainty of a decree of specific performance following trial, Mr. Musk abandoned his pretextual claims..."
```

#### 7. California Public Utilities Commission Rate Proceeding (`samples/puc_infrastructure_hearing.json`)
45 regulatory turns across 14 commissioners, judges, ratepayer advocates (TURN, Cal Advocates), investor-owned utilities (SCE, PG&E), and solar coalitions (CALSSA, NRDC) evaluating the NEM 3.0 net billing reform. KLens charts prolonged deadlock over the $4.0B cost shift and grid connection fees ($T_{\text{cert}} = 127$ rounds), which collapses cleanly into unanimous en banc adoption of Decision 22-12-056 ($\lambda_2 = 0.6119, T_{\text{cert}} = 7$ rounds) upon introduction of battery storage export adders:

![CPUC Hearing Telemetry](assets/puc_hearing_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Battery Storage vs. Safety and Compliance Governance
  "The primary debate axis centers on the trade-off between battery storage and safety and compliance governance."
• Consensus Centroid Concept: Decision 22-12-056 & California Grid
  "The legacy net metering rules were essential to launch California's solar revolution, but they are no longer suited for a grid that requires flexible, dispatchable clean energy."

--- Faction Topics & Positions ---
• Faction 2 & 3 (CALSSA - Heavner, Del Chiaro): Industry Viability & Glide Path Adders
  Position: "CALSSA proposes that existing customers be grandfathered for 20 years, and that new adopters receive a transitional export credit adder..."
• Faction 4 (Commissioner Rechtschaffen): Evening Duck Curve Ramping & Avoided Cost Alignment
  Position: "The legacy net metering rules were essential to launch California's solar revolution, but they are no longer suited for a grid that requires flexible, dispatchable clean energy."
• Faction 5 (Commissioner Houck): Consumer Protection & Ratepayer Cross-Subsidy Reduction
  Position: "Reforming the tariff to true avoided cost stops a regressive $4 billion wealth transfer..."

--- Rogue Dissent Position (President Alice Reynolds) ---
• Stance:   Clean Energy Framework
  Position: "The revised decision before us reflects substantial improvements: it eliminates the proposed monthly grid access charge..." [Presiding facilitator]
```

#### 8. New York City Council Stated Meeting Benchmark (`samples/city_council_policy_meeting.json`)
48 municipal turns across 14 council leaders and borough delegates debating the contested "City of Yes for Housing Opportunity" zoning text amendment. KLens charts acute deadlock triggered by 38 Community Board rejections and outer-borough parking minimum disputes ($T_{\text{cert}} = 102$ rounds), resolved when Council leadership negotiates a $5.0 Billion Capital Infrastructure & Tenant Protection Agreement, leading to a contested 31–18 roll call adoption ($\lambda_2 = 0.6361, T_{\text{cert}} = 7$ rounds):

![City Council Telemetry](assets/city_council_telemetry.png)

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Parking and Density vs. Affordable Housing
  "The primary debate axis centers on the trade-off between parking and density and affordable housing."
• Consensus Centroid Concept: Vote Aye & I Vote
  "I vote aye."

--- Faction Topics & Positions ---
• Faction 1 (Alexa Avilés): Progressive Supply-Side Skepticism
  Position: "Market-rate supply-side deregulation will not solve the affordability crisis for families in Sunset Park."
• Faction 5 (Joseph Borelli, Kevin Riley): Outer-Borough Parking Preservation
  Position: "My constituents on Staten Island do not want higher density, traffic congestion, or lost parking."
• Faction 6 (Justin Brannan): Capital Infrastructure Front-Loading
  Position: "Housing without infrastructure is a disaster; housing with infrastructure is the future."
• Faction 9 (Rafael Salamanca Jr.): Land-Use Equity & Sewer Upgrades
  Position: "With $2 billion dedicated to sewer upgrades and deep affordability guarantees, this modified plan strikes the right balance..."

--- Rogue Dissent Position (Speaker Adrienne Adams) ---
• Stance:   City Council Stated Procedure
  Position: "The clerk will call the roll on Land Use Matter 0118-2024..." [Persistent rogue alert: presiding facilitator]
```

> **Detailed Guide**: For complete preprocessing pipelines, sourcing instructions, and mathematical interpretations on human deliberation traces, see [docs/HUMAN_DELIBERATION_BENCHMARKS.md](docs/HUMAN_DELIBERATION_BENCHMARKS.md).

---

### Empirical Findings: Human vs. Multi-Agent Deliberation Dynamics

By applying Koopman operator spectral analysis and SVD disagreement decomposition across verbatim human proceedings, real-world multi-agent LLM debates ([`Multi-Agent-LLMs/DEBATE`](https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE)), and synthetic scenario fixtures, KLens exposes four fundamental mathematical and behavioral distinctions:

| Deliberation Dimension | Real Multi-Agent LLM Swarms | Real-World Human Deliberations | Synthetic Fixtures |
| :--- | :--- | :--- | :--- |
| **Faction Topologies** | **Micro-Coalition (1–2 Factions)**<br>Collapses into 1 consensus bloc with isolated dissenter; high normative conformity. | **Hyper-Fragmentation (5–21 Factions)**<br>Escalates monotonically (e.g. FOMC 11 → 21 factions); persistent ideological dispersion. | **Isotropic / Single-Node**<br>Scripted single-member clusters until synchronous 1-round collapse. |
| **Spectral Dynamics** | **Rapid Contraction & Late Crystallization**<br>Starts contractive ($\lambda_2 = 0.00 \to 0.50, T_{\text{cert}} \le 10$), slowly rising only if a holdout persists. | **Metastable Deadlock & Bifurcations**<br>Immediate severe friction ($\lambda_2 \ge 0.85 - 0.98, T_{\text{cert}} > 100$) until sharp doctrinal collapse. | **Step-Function Collapse**<br>Holds near 0.97 until single-round collapse to consensus template or scripted deadlock. |
| **Outlier Mechanics** | **Persistent Substantive Holdouts**<br>Cleanly flags the same contrarian persona across 5+ rounds based on argument content. | **The "Moderator Effect"**<br>Flags institutional facilitators (Roberts, Bernanke) due to procedural brevity having low topic overlap. | **Scripted Rule Repetition**<br>Flags explicitly prompted contrarian agents repeating fixed assertions. |
| **Linguistic Geometry** | **Expository & Explicit Alignment**<br>65%–75% explicit tags (`[AGREE]`/`[DISAGREE]`); cohesive length (CV 0.50–0.87); 0%–17% questions; 0% sparsity. | **Heavy-Tailed & Inquisitive**<br>0% explicit tags; heavy-tailed turns (CV 1.65–2.21); 30%–40% questions; 30%–42% empty round slots. | **Uniform & Round-Robin**<br>Strict 10–12 word sentences (CV < 0.30); 0% questions; 1 turn per agent per round; Gini = 0.00. |

### Detailed Observations

#### 1. Faction Topologies & Clustering Dynamics: Hyper-Fragmentation vs. Micro-Coalition
* Real LLM swarms display a strong normative pull toward shared vocabulary and framing, collapsing into 1 or 2 factions within 2–3 rounds.
* Human institutions maintain high-dimensional ideological dispersion: in FOMC 2011, faction counts escalate from 11 to 21 across 5 rounds as members introduce local caveats and statutory conditions.

#### 2. Spectral Contraction: Early Contraction vs. Marginal Deadlock
* In real LLM traces (`samples/real_llm_debate_*.json`), the system starts highly contractive ($\lambda_2 = 0.0000, T_{\text{cert}} = 1$ in Round 2) before gently drifting to $\sim 0.70 - 0.95$ as positions solidify.
* In human debates, deliberations operate near marginal stability from Round 2 onwards ($\lambda_2 \ge 0.85 - 0.98, T_{\text{cert}} = 55 - 187$).

#### 3. Outlier Mechanics: Substantive Holdouts vs. The "Moderator Effect"
* In real LLM traces, the `PERSISTENT_ROGUE` detector cleanly isolates the genuine holdout (e.g., `Entomologist`, `Cryptocurrency Expert`, `Computer Science Theorist`, `Local Resident`) across 5+ consecutive rounds.
* In human proceedings, procedural turn-taking (*"Justice Thomas?"*, *"Thank you, counsel"*) produces short, low-TF-IDF vectors that distance moderators (Chief Justice Roberts, Chairman Bernanke) from substantive centroids, creating the "moderator effect".

#### 4. Linguistic Geometry & Stance Markers
* Real LLMs rely heavily on overt stance declarations (`[AGREE]`/`[DISAGREE]` in 65%–75% of turns) and consistent paragraph-length arguments (CV = 0.50–0.87).
* Humans communicate through rhetorical implication, heavy questioning (30%–40%), extreme turn-length disparity (CV = 1.65–2.21), and selective entry (30%–42% empty speaker slots per round).

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

![Metastable Faction Polarization & Consensus Centroid Contraction](assets/faction_centroid_drift.png)

*Figure: Metastable faction clustering and consensus centroid contraction during a 6-agent deliberation. The consensus centroid $\bar{z}(t)$ acts as a moving semantic attractor pulling factions toward resolution.*

### 6. Forward Agent Trajectory Prediction
Because the finite-dimensional Koopman operator $\hat{K}$ acts as a linear forward transition generator in the lifted observable space ($\Psi_{t+1} \approx \Psi_t \hat{K}$), KLens projects agent positions $k$ rounds ahead:
$$\hat{\Psi}(t+k) = \Psi(t) \hat{K}^k$$
Extracting the unlifted state indices $\hat{\Delta}(t+k) = \hat{\Psi}(t+k)[:, 1 : 1 + Nd]$ yields forward forecasting:
* **Projected Rogue Agent**: $\arg\max_i \|\hat{\Delta}_i(t+k)\|_2$, forecasting which participant will drift furthest into contrarian disagreement in future rounds.
* **Projected Contraction Rate**: $\frac{\|\hat{\Delta}(t+k)\|}{\|\Delta(t)\|}$, predicting whether upcoming debate rounds will contract toward consensus ($< 1.0$) or expand into deadlock ($> 1.0$).

### 7. Semantic Topic Synthesis & Position Content Extraction
To bridge geometric dynamics with human reasoning, KLens extracts human-interpretable concepts, the principal center of disagreement, faction positions, and the consensus centroid:
* **Center of Disagreement (Principal Contention Axis)**: Singular Value Decomposition of the centered disagreement matrix $\Delta(T) = U \Sigma V^T$ yields the dominant singular vector $v_1 \in \mathbb{R}^d$. The positive and negative loadings isolate the opposing ideological poles:
  $$\text{Pole}_+ = \text{TopKeywords}(v_1^+), \quad \text{Pole}_- = \text{TopKeywords}(v_1^-)$$
  Synthesizing the primary conflict: e.g. *"Zero AI vs. Selective Judge Auditing"*.
* **Faction Topics & Positions**: For each cluster $F_k$, distinctive weighting $\delta_{F_k} = z_{F_k} - \bar{z}$ isolates characteristic themes. Verbatim position statements are extracted via recency-weighted projection:
  $$s^*_{F_k} = \arg\max_{s \in \text{Turns}(F_k)} \cos(v_s, z_{F_k}) \cdot \left(1 + 0.35 \frac{r_s}{T}\right)$$
* **Consensus Centroid Concept & Statement**: Verbatim consensus compromise representing the moving landmark $\bar{z}$:
  $$s^*_{\text{centroid}} = \arg\max_{s} \cos(v_s, \bar{z}) \cdot \left(1 + 0.35 \frac{r_s}{T}\right)$$

```
=== Deliberation Semantic Synthesis ===
• Center of Disagreement: Zero AI vs. Selective Judge Auditing
  "The primary debate axis centers on the trade-off between zero ai and selective judge auditing."
• Consensus Centroid Concept: Selective Judge Auditing & Hybrid SLM Orchestration
  "Agreement on hybrid SLM orchestration with selective judge auditing for clinical rigor."

--- Faction Topics & Positions ---
• Faction 1 (5 members): Selective Judge Auditing
  Position: "Agreement on hybrid SLM orchestration with selective judge auditing for clinical rigor."

--- Rogue Dissent Position (Agent 5 (Regulatory/Adversary)) ---
• Stance:   Zero AI
  Position: "AI must not touch clinical decisions under any condition."
```

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

### 1. Render Visual Diagnostic Dashboard (One-Liner)

Pass any multi-agent transcript (JSON or JSONL) to `klens plot` or the standalone file loader script:

```bash
# Render 4-panel diagnostic dashboard matching the clinical debate benchmark
klens plot samples/clinical_slm_debate.json --output out/clinical_telemetry.png

# Or use the standalone Python file loader
python examples/visualize_transcript.py samples/clinical_slm_debate.json --output out/clinical_telemetry.png
```

### 2. Run Built-in 6-Agent Simulation

```bash
klens demo --plot out/demo_telemetry.png --output spectral_debate_log.csv --t-cert-limit 50
```

### 3. Analyze & Plot Transcript Files (JSON or JSONL)

KLens includes realistic deliberation traces in [samples/](samples/) covering diverse multi-agent topologies:

```bash
# 1. Clinical SLM Deployment & Regulatory Audit (JSON)
klens run samples/clinical_slm_debate.json --plot out/clinical_telemetry.png --output out/metrics_clinical.csv

# 2. Multi-Agent LLM Safety Consensus Debate (JSON)
klens run samples/llm_safety_consensus.json --plot out/safety_telemetry.png --output out/metrics_safety.csv

# 3. Executive Boardroom Polarization & Deadlock (JSONL)
klens run samples/boardroom_deadlock_polarization.jsonl --plot out/deadlock_telemetry.png --output out/metrics_deadlock.csv

# 4. 24-Agent Swarm with Hierarchical Coarse-Graining (JSONL)
klens run samples/swarm_deliberation_24agents.jsonl --hierarchical --max-factions 4 --plot out/swarm_telemetry.png --output out/metrics_swarm.csv

# 5. Real-World Human Deliberation: SCOTUS Bruen Oral Argument (JSON, 370 turns across 12 Justices & Advocates)
klens run samples/scotus_bruen_oral_argument.json --plot out/scotus_bruen_telemetry.png --output out/metrics_bruen.csv

# 6. Real-World Human Deliberation: Federal Reserve FOMC Monetary Policy Meeting (JSON, 258 turns across 25 Fed Policymakers)
klens run samples/fomc_monetary_policy_aug2011.json --plot out/fomc_telemetry.png --output out/metrics_fomc.csv --t-cert-limit 50
```

---

## Python API Usage

```python
from klens import (
    LiveTranscriptStreamProcessor,
    AlertConfig,
    HierarchicalConfig,
    plot_from_transcript_file,
)

# --- Option A: One-line transcript visualization ---
processor, saved_path = plot_from_transcript_file(
    transcript_path="samples/clinical_slm_debate.json",
    output_path="out/clinical_telemetry.png",
    show=False,
)

# --- Option B: Streaming ingestion & real-time monitoring ---
# 1. Configure processor, alert thresholds, and scaling
config = AlertConfig(
    t_cert_limit=50,             # Alert if certified deadline > 50 rounds
    lambda_2_warning_limit=0.95, # Warning if spectral radius is sluggish
    lambda_2_critical_limit=0.99 # Critical alert if near divergence
)

# Configure scaling: toggle hierarchical coarse-graining on or off
hier_config = HierarchicalConfig(
    enabled=False,        # Set False to "go big" (full compute on all agents), or True for macro-faction coarse-graining
    max_factions=8,       # Max macro-faction centroids when hierarchical mode is enabled
    solver="auto",        # "auto" dynamically selects Snapshot Dual EDMD when S < P to prevent O(P^3) memory blowup
)

processor = LiveTranscriptStreamProcessor(
    csv_log_path="spectral_debate_log.csv",
    alert_config=config,
    hierarchical_config=hier_config,
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
    print(f"Hierarchical mode active: {metrics.is_hierarchical}")

    # 5. Access human-readable semantic concepts and verbatim positions
    if metrics.topic_synthesis:
        synth = metrics.topic_synthesis
        print(f"Center of Disagreement: {synth.center_of_disagreement}")
        print(f"Consensus Centroid:     {synth.centroid_concept}")
        print(f"Consensus Statement:   \"{synth.centroid_statement}\"")
        for f in synth.factions:
            print(f"[{f.topic}] {f.members} -> \"{f.representative_statement}\"")
```

---

## Scaling to Massive Agent Populations

KLens supports scaling across two operational regimes:

1. **Micro-Level Resolution ("Go Big" / Default `enabled=False`)**:
   - Models every individual agent directly in the lifted observable space.
   - Ideal for small to medium groups ($N \le 50$) where full computational resolution across every agent is desired.
   - When $S < P$, the built-in **Dual (Snapshot) Solver** automates eigendecomposition via the snapshot Gram matrix $G_{\text{snap}} = \Psi \Psi^T \in \mathbb{R}^{S \times S}$, eliminating the $\mathcal{O}(P^3)$ memory/compute crash.

2. **Hierarchical Multi-Scale Decomposition & Macro-Faction Coarse-Graining (`enabled=True`)**:
   - Coarse-grains $N$ agent trajectories into $K = \min(N, \text{max\_factions})$ macro-faction centroids based on time-scale separation (fast intra-faction decay vs. slow inter-faction consensus drift).
   - Runs Koopman EDMD on the macro-centroid dynamics to certified convergence ($T_{\text{cert}}$).
   - **Zero Loss in Rogue Detection**: Computes individual agent deviations in an $\mathcal{O}(Nd)$ streaming residual pass against the consensus centroid, guaranteeing that outlier rogue agents are accurately detected across all $N$ agents.

### CLI Scaling Options

```bash
# Go big: full micro-level resolution
klens demo --solver auto

# Scale up: enable hierarchical coarse-graining with 4 macro-centroids
klens demo --hierarchical --max-factions 4
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

---

## Future Research Horizons

For active engineering epics, streaming ingestion protocols, and task tracking, see [BACKLOG.md](BACKLOG.md).

### 1. Live Telemetry & Real-Time Deliberation Visualization
The next phase of the KLens workbench integrates a live graphical cockpit for ongoing meetings and deliberations:
* **Streaming Inputs to Dynamic Dashboard**: Ingesting real-time WebSocket / SSE turns into an interactive web dashboard.
* **Complex Unit Disk Telemetry**: Real-time streaming plot of the operator eigenvalues $\{\mu_j\} \subset \mathbb{C}$. Clustering near the unit circle visually flags ideological deadlock.
* **Dynamic Manifold Animation**: 2D/3D low-dimensional projections tracking participant attraction, faction clustering, and consensus trajectory drift in real time.
* **$T_{\text{cert}}$ Early-Warning Gauges**: Live countdown cones alerting facilitators when debate dynamics turn non-contractive ($|\lambda_2| \ge 1.0$).


### 2. Multimodal Discussion Telemetry: Audio, Visual, and Verbal Signals
While text embeddings capture literal propositions, human and embodied multi-agent deliberations are deeply multimodal. A primary research inquiry for KLens is:

> *Do acoustic and visual signals help or hinder the predictive accuracy of Koopman spectral consensus certificates?*

* **Acoustic Channels ($z^{\text{audio}}$)**: Prosodic dynamics, pitch variance ($F_0$), vocal intensity, speech cadence, and conversational turn-taking latency.
* **Visual Channels ($z^{\text{visual}}$)**: Facial action unit activations (FACS), head pose kinematics, gaze direction, and visual attention synchrony.
* **The Cross-Modal Congruence Factor ($\Gamma_i(t)$)**:
  Measuring the alignment between verbal semantics, vocal prosody, and facial affect:
  $$\Gamma_i(t) = \frac{1}{3} \left[ \cos(z_i^v, z_i^a) + \cos(z_i^v, z_i^{vis}) + \cos(z_i^a, z_i^{vis}) \right]$$
* **The Congruence Experiment**: Investigating whether high multimodal congruence stabilizes the Koopman transfer operator, and whether cross-modal dissonance (e.g. verbal concession masked by acoustic tension or averted gaze) injects non-conservative perturbation energy into $\hat{K}$, exposing latent disagreement long before it appears in text.

---

## Research References & Attribution

The theoretical architecture and algorithmic foundations of KLens are grounded in the following primary papers:


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

### 2. Scaling, Diversity, & Faction Coarse-Graining Source
* **Paper:** *"Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity"* — Yingxuan Yang et al. [10, 11].
* **Core Concepts Used in the Code:**
  * **Effective Channel Count (\\(K^*\\)):** Quantifying non-redundant reasoning paths rather than raw agent count \\(N\\) [12, 13].
  * **Diminishing Returns of Scale:** Demonstrating why scaling homogeneous agents leads to correlated output redundancy [14, 15].
  * **Centroid Coarse-Graining & Dimension Reduction:** Exploiting the principle that massive agent populations (\\(N = 10,000+\\)) naturally collapse into a small number of effective reasoning channels (factions), enabling \\(O(N)\\) scaling for thousand-agent traces [12, 16].

---

### 3. Disagreement & Selector Bottleneck Source
* **Paper:** *"When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines"* — Artem Maryanskyy et al. [17, 18].
* **Core Concepts Used in the Code:**
  * **Divergence & Selection Dynamics:** Analyzing how agent disagreement creates bottlenecks and demonstrating why pruning uncooperative or rogue entities dramatically accelerates consensus pipeline performance [17, 18].

---

## License & Confidentiality Scope

* **Open-Source Software**: The KLens engine, streaming processor, CLI, and test suite are licensed under the permissive [MIT License](LICENSE).
* **Confidential Research Scope**: Proprietary research notes, novel deliberation experiments, conference telemetry designs, and private datasets residing in the `research/` directory are confidential, proprietary, and excluded from version control via `.gitignore`.

