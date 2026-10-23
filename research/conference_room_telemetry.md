# Research Note: Conference Room Deliberation Telemetry (RTMP/RTSP)

> **Status:** Exploratory Research Idea (Held off from core roadmap)  
> **Classification:** Internal Research Concept  
> **Topic:** Physical Conference Room Sensor Ingestion & Privacy-Preserving Deliberation Analysis

---

## 1. Concept Overview

This document explores the integration of live audio/video telemetry from physical conference rooms (e.g. Cisco Webex RoomOS, Poly, Meeting Owl, Logitech Tap) into the KLens Koopman spectral consensus engine.

Instead of analyzing text alone, a real-time RTMP/RTSP/WebRTC feed from in-room cameras and beamforming ceiling microphones would be demuxed into:
1. **Verbal Channel ($z^{\text{verbal}}$)**: Real-time speaker-diarized speech-to-text.
2. **Acoustic Channel ($z^{\text{audio}}$)**: Prosodic dynamics (fundamental frequency $F_0$, energy, tempo, interruption latency).
3. **Visual Channel ($z^{\text{visual}}$)**: Facial action units (FACS), head pose kinematics, and gaze attention vectors.

---

## 2. Confidentiality, Privacy, and Standalone Guarantees

Connecting conference room hardware introduces significant confidentiality requirements:

### A. Strict On-Premise / Edge Processing
* **No Raw Audio/Video Exfiltration**: In a standalone deployment, raw RTMP streams must never leave the local network boundary or be transmitted to third-party cloud APIs.
* **Local Edge Inference**:
  * Speech-to-Text: Run locally using open weights (e.g. `whisper.cpp`, `faster-whisper`, or local vLLM).
  * Prosody Extraction: Run locally via `openSMILE` or `pyworld`.
  * Visual Affect: Run locally via `MediaPipe` or `Py-Feat`.

### B. Vector-Only Ingestion Boundary
* KLens receives only normalized, numerical embedding vectors $z_i(t) \in \mathbb{R}^d$.
* Raw speech recordings, video frames, and biometric identifiers are discarded at the edge demux buffer immediately after feature extraction.

### C. Zero External Phone-Home
* KLens contains zero telemetry, analytics beacons, or remote logging. All matrices, spectral logs, and CSV outputs are written strictly to local disk.

---

## 3. Exploratory Research Hypotheses

1. **Latent Dissensus Detection**: Does acoustic tension or averted gaze signal disagreement before a participant speaks or concedes verbally?
2. **Entrainment vs. Polarization**: Does acoustic pitch entrainment precede consensus resolution in physical meetings?
3. **Hardware Ingestion Viability**: Can a local workstation process multi-person audio diarization and facial tracking in real time ($< 500\text{ ms}$ latency) without saturating the Koopman streaming buffer?
