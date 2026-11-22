from __future__ import annotations

import json
import pytest
import numpy as np

from klens.encoders import TfidfTextEncoder
from klens.models import Faction, TranscriptTurn
from klens.semantic import (
    clean_title_casing,
    extract_candidate_phrases,
    extract_candidate_sentences,
    score_phrases_against_vector,
    synthesize_deliberation_topics,
)
from klens.spectral import (
    cluster_factions,
    detect_rogue_agent,
    extract_last_active_states,
)
from klens.stream import LiveTranscriptStreamProcessor


def test_clean_title_casing():
    assert clean_title_casing("hybrid slm orchestration") == "Hybrid SLM Orchestration"
    assert clean_title_casing("public cloud api velocity") == "Public Cloud API Velocity"
    assert clean_title_casing("fda compliance and ciso sign-off") == "FDA Compliance and CISO Sign-off"
    assert clean_title_casing("") == ""


def test_extract_candidate_sentences():
    records = [
        TranscriptTurn(round_id=1, speaker="A", text='We must deploy SLMs. Latency is critical!'),
        TranscriptTurn(round_id=1, speaker="B", text='Short.'),
    ]
    sentences = extract_candidate_sentences(records)
    assert len(sentences) == 2
    assert sentences[0] == ("We must deploy SLMs.", "A", 1)
    assert sentences[1] == ("Latency is critical!", "A", 1)


def test_semantic_synthesis_clinical_sample():
    with open("samples/clinical_slm_debate.json") as f:
        data = json.load(f)

    records = [
        TranscriptTurn(
            round_id=d.get("round", d.get("round_id", 1)),
            speaker=d["speaker"],
            text=d["text"],
        )
        for d in data
    ]
    rounds = sorted(list(set(d.round_id for d in records)))
    speakers = sorted(list(set(d.speaker for d in records)))
    text_grid = {(r, s): "" for r in rounds for s in speakers}
    for d in records:
        text_grid[(d.round_id, d.speaker)] += " " + d.text

    encoder = TfidfTextEncoder(max_features=40)
    Z, feature_names = encoder.encode_grid(text_grid, rounds, speakers)
    Delta = Z - Z.mean(axis=1, keepdims=True)
    rogue_spk, rogue_idx = detect_rogue_agent(Delta, speakers, len(rounds), len(speakers), len(feature_names))
    factions = cluster_factions(Z[-1], speakers, rogue_idx)

    synth = synthesize_deliberation_topics(
        records=records,
        Z=Z,
        feature_names=feature_names,
        speakers=speakers,
        rounds=rounds,
        factions=factions,
        rogue_speaker=rogue_spk,
        vectorizer=encoder.vectorizer,
    )

    assert "vs." in synth.center_of_disagreement
    assert len(synth.center_of_disagreement_statement) > 20
    assert len(synth.centroid_concept) > 0
    assert "orchestration" in synth.centroid_statement.lower() or "hybrid" in synth.centroid_statement.lower()

    assert len(synth.factions) >= 1
    for f in synth.factions:
        assert len(f.topic) > 0
        assert len(f.representative_statement) > 10

    assert synth.rogue_stance is not None
    assert synth.rogue_stance.members == ["Agent 5 (Regulatory/Adversary)"]
    assert len(synth.rogue_stance.topic) > 0
    assert len(synth.rogue_stance.representative_statement) > 10


def test_stream_processor_attaches_topic_synthesis():
    processor = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_semantic_stream.csv",
        auto_init_csv=False,
    )

    processor.ingest_turn(1, "Agent_A", "We must adopt public cloud to scale elastic workloads.")
    processor.ingest_turn(1, "Agent_B", "Cloud is unacceptable. Data security requires on-prem hardware.")
    processor.ingest_turn(2, "Agent_A", "A hybrid model running cloud frontends with private on-prem databases works.")
    processor.ingest_turn(2, "Agent_B", "We agree on the hybrid architecture with private on-prem databases.")

    m = processor.process_round_event(2)
    assert m is not None
    assert m.topic_synthesis is not None
    assert len(m.center_of_disagreement) > 0
    assert len(m.centroid_concept) > 0
    assert len(m.centroid_statement) > 0
    assert "hybrid" in m.centroid_statement.lower()


def test_channel_incorporation_and_attribution():
    processor = LiveTranscriptStreamProcessor(
        csv_log_path="out/test_channel_attribution.csv",
        auto_init_csv=False,
    )
    processor.ingest_turn(1, "Agent_A", "We must adopt public cloud to scale elastic workloads.")
    processor.ingest_turn(1, "Agent_B", "Cloud is unacceptable. Data security requires on-prem hardware.")
    processor.ingest_turn(1, "Agent_C", "Zero trust isolation must govern all communication.")
    processor.ingest_turn(2, "Agent_A", "A hybrid model running cloud frontends with private on-prem databases works.")
    processor.ingest_turn(2, "Agent_B", "We agree on the hybrid architecture with private on-prem databases.")
    processor.ingest_turn(2, "Agent_C", "We concede and accept hybrid architecture with on-prem databases.")

    m = processor.process_round_event(2)
    assert m is not None
    assert m.effective_channels_k_star >= 1.0

    synth = m.topic_synthesis
    assert synth is not None
    assert synth.effective_channels_k_star >= 1.0
    assert synth.initial_k_star >= 1.0
    assert synth.total_channels_identified >= 1

    valid_roles = {
        "Primary Attractor",
        "Conceding Coalition",
        "Synthesizing Compromiser",
        "Unreconciled Dissent",
        "Inactive Panelist",
    }

    all_stances = list(synth.factions)
    if synth.rogue_stance:
        all_stances.append(synth.rogue_stance)

    assert len(all_stances) == synth.total_channels_identified
    total_weight = sum(s.consensus_weight for s in all_stances)
    assert pytest.approx(1.0, abs=1e-3) == total_weight

    for stance in all_stances:
        assert stance.dynamical_role in valid_roles
        assert len(stance.contributed_terms) > 0
        assert stance.consensus_weight >= 0.0
        assert not stance.topic.startswith("Faction ")

    assert any(s.dynamical_role != "Inactive Panelist" and s.consensus_weight > 0.0 for s in all_stances)


def test_channel_roles_distinguish_attractor_dissent_and_inactive():
    speakers = ["Anchor_1", "Anchor_2", "Adapter", "Dissenter", "Absent"]
    feature_names = ["elastic", "security", "dissent", "legacy"]
    records = [
        TranscriptTurn(round_id=1, speaker="Anchor_1", text="Elastic scaling requires unified architecture."),
        TranscriptTurn(round_id=1, speaker="Anchor_2", text="Elastic scaling governs production stability."),
        TranscriptTurn(round_id=1, speaker="Adapter", text="Security controls must guard elastic deployment."),
        TranscriptTurn(round_id=1, speaker="Dissenter", text="Unilateral dissent blocks every proposal."),
        TranscriptTurn(round_id=1, speaker="Absent", text="Legacy hardware governance remains essential."),
        TranscriptTurn(round_id=2, speaker="Anchor_1", text="Elastic scaling remains our primary anchor."),
        TranscriptTurn(round_id=2, speaker="Anchor_2", text="Elastic scaling anchors the final consensus."),
        TranscriptTurn(round_id=2, speaker="Adapter", text="Security controls integrate with elastic scaling."),
        TranscriptTurn(round_id=2, speaker="Dissenter", text="Unilateral dissent rejects compromise."),
    ]
    Z = np.array([
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.9, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.4, 0.9, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ])
    factions = [
        Faction(id=1, members=["Anchor_1", "Anchor_2"]),
        Faction(id=2, members=["Adapter"]),
        Faction(id=3, members=["Absent"]),
    ]

    synth = synthesize_deliberation_topics(
        records=records,
        Z=Z,
        feature_names=feature_names,
        speakers=speakers,
        rounds=[1, 2],
        factions=factions,
        rogue_speaker="Dissenter",
    )

    by_id = {f.id: f for f in synth.factions}
    assert by_id[1].dynamical_role == "Primary Attractor"
    assert by_id[2].dynamical_role in {"Conceding Coalition", "Synthesizing Compromiser"}
    assert by_id[3].dynamical_role == "Inactive Panelist"
    assert by_id[3].consensus_weight == 0.0
    assert "legacy" in by_id[3].keywords
    assert not by_id[3].topic.startswith("Faction ")
    assert synth.rogue_stance is not None
    assert synth.rogue_stance.dynamical_role == "Unreconciled Dissent"

    active_weight_sum = sum(
        s.consensus_weight
        for s in [*synth.factions, synth.rogue_stance]
        if s.dynamical_role != "Inactive Panelist"
    )
    assert pytest.approx(1.0, abs=1e-4) == active_weight_sum


def test_stop_words_and_last_active_states_edge_cases():
    records = [
        TranscriptTurn(
            round_id=1,
            speaker="Senator",
            text="Mr. President, just very here there his her my your policy reform works.",
        ),
    ]
    phrases = extract_candidate_phrases(records)
    assert "Mr President" not in phrases
    assert "your policy" not in phrases
    assert "policy reform" in phrases

    empty_states, empty_mask = extract_last_active_states(np.zeros((0, 3, 2)))
    assert empty_states.shape == (3, 2)
    assert empty_mask.shape == (3,)
    assert not np.any(empty_mask)

    two_d_states, two_d_mask = extract_last_active_states(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert two_d_mask.tolist() == [True, False]
    assert np.allclose(two_d_states[0], [1.0, 0.0])
    assert np.allclose(two_d_states[1], [0.0, 0.0])

    Z = np.array([
        [[0.0, 2.0], [0.0, 0.0], [3.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]],
    ])
    last_states, terminal_mask = extract_last_active_states(Z)
    assert terminal_mask.tolist() == [False, False, True]
    assert np.allclose(last_states[0], [0.0, 2.0])
    assert np.allclose(last_states[1], [0.0, 0.0])
    assert np.allclose(last_states[2], [4.0, 0.0])

