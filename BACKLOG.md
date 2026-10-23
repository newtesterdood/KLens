# KLens Engineering & Research Backlog

This backlog tracks planned feature development, architectural enhancements, and research milestones for the KLens platform.

---

## High Priority: Streaming Inputs to a Dynamic Dashboard

### Epic Overview
Enable real-time, bidirectional streaming of multi-party deliberation inputs (text, audio, video telemetry) into an interactive, low-latency visual dashboard. This transitions KLens from an offline/batch trace analyzer into a live operational cockpit for ongoing multi-agent and human deliberations.

```
┌─────────────────────────────────────────┐
│           Live Deliberation             │
│  (LLM Debate / Meeting Bot / Simulator) │
└─────────────────────────────────────────┘
                     │
         Streaming WebSocket / SSE
                     │
                     ▼
┌─────────────────────────────────────────┐
│     KLens Async Streaming Ingestion     │
│  - Slide-buffer turn ingestion          │
│  - Online EDMD state updater            │
│  - Real-time forward trajectory forecast│
└─────────────────────────────────────────┘
                     │
          WebSocket Broadcast Bus
                     │
                     ▼
┌─────────────────────────────────────────┐
│        Dynamic Telemetry Dashboard      │
│  - Live Complex Unit Disk (λ2 modes)    │
│  - Real-time 2D/3D Faction Manifold     │
│  - T_cert Convergence Countdown Gauge   │
│  - Predictive Rogue Escalation Radar    │
│  - Cross-Modal Congruence Dial (Γ)      │
└─────────────────────────────────────────┘
```

---

### Work Breakdown & Sub-Tasks

#### 1. Streaming Ingestion Server (`klens.server`)
- [ ] **WebSocket & SSE Endpoints**: Build a lightweight async streaming server (`klens serve --port 8000`) supporting real-time turn streaming.
- [ ] **Streaming Wire Protocol**: Define JSON schemas for streaming events:
  - `turn.ingest`: Ingests an ongoing speech turn `{round_id, speaker, text, timestamp, audio_features, visual_features}`.
  - `round.finalize`: Signals round boundary and triggers incremental EDMD spectral update.
  - `spectral.telemetry`: Emits real-time eigenvalues, $T_{\text{cert}}$, rogue flags, and faction clusters.
  - `forecast.update`: Emits forward-projected agent positions and predicted rogue flags.
- [ ] **External Connectors**:
  - Connectors for OpenAI Realtime API / Gemini Live API multi-agent simulations.
  - Adapter for live meeting transcript streams (Zoom/Teams/Google Meet webhook ingestion).

#### 2. Online / Incremental Koopman State Updater
- [ ] **Recursive EDMD**: Implement rank-1 Woodbury/Sherman-Morrison updates for Gram matrix $G(t+1) = G(t) + \psi_t \psi_t^T$ to eliminate re-computing from scratch at every turn.
- [ ] **Sliding Window Stream Buffer**: Maintain configurable fixed-memory history windows for unbounded streaming sessions.

#### 3. Interactive Web Telemetry Dashboard (`klens-ui`)
- [ ] **Complex Unit Disk Visualizer**:
  - Interactive Canvas / SVG visualization of eigenvalues $\{\mu_j\}$ on the unit circle $|z|=1$.
  - Visual indicators for contractive modes ($|\mu| < 1.0$) vs. divergent/metastable modes ($|\mu| \ge 1.0$).
  - Animated decay cones illustrating the asymptotic decay governed by $\lambda_2$.
- [ ] **Real-Time Faction & Belief Manifold**:
  - Force-directed 2D/3D graph rendering participants as nodes and cosine similarities as elastic edges.
  - Dynamic clustering showing real-time polarization, faction consolidation, and agent defection.
- [ ] **Predictive Alert & Convergence Gauges**:
  - Heads-up display for $T_{\text{cert}}$ certified deadline countdown.
  - Rogue agent radar highlighting projected contrarians for upcoming rounds.
  - Contraction rate speedometer ($\|\hat{\Delta}(t+1)\| / \|\Delta(t)\|$).
- [ ] **Historical Playback & Time-Scrubber**:
  - Time-slider allowing researchers to scrub backward through past rounds of a debate to inspect historical phase transitions.

#### 4. Multimodal Telemetry Integration
- [ ] **Acoustic & Visual Feature Ingestion**:
  - Streaming payload support for fundamental frequency ($F_0$), speech intensity, and speaking rate.
  - Streaming facial action unit (FACS) activations and head kinematics.
- [ ] **Live Congruence Gauge**: Real-time display of the Cross-Modal Congruence Factor $\Gamma_i(t)$.

---

## Medium Priority: Algorithmic & Analytical Extensions

- [ ] **Kernelized Deep EDMD (Deep Koopman)**: Optional trainable neural observable dictionary $\Psi_\theta(z)$ for non-stationary linguistic manifolds.
- [ ] **Multi-Issue Manifold Slicing**: Decomposing consensus centroids along orthogonal topical sub-spaces to isolate which specific sub-issues reached consensus vs. stalled.
- [ ] **Automated Mediation Interventions**: System hooks triggering synthetic mediator agents to inject targeted prompts when $|\lambda_2| \ge 0.99$.

---

## Low Priority / Exploratory

- [ ] **Hardware Acceleration**: GPU-accelerated batch eigensolvers for ultra-massive agent populations ($N \ge 1,000,000$).
- [ ] **Export to OpenTelemetry**: Emitting Koopman spectral metrics as standard OpenTelemetry gauge metrics for enterprise observability stacks (Prometheus / Grafana).
