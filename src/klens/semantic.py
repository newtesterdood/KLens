from __future__ import annotations

import re
from collections import Counter
from typing import Any, List, Optional, Sequence, Set, Tuple
import numpy as np

from klens.models import (
    DeliberationTopicSynthesis,
    Faction,
    FactionStance,
    TranscriptTurn,
)
from klens.spectral import compute_effective_channels_k_star

COMMON_ACRONYMS: Set[str] = {
    "slm", "llm", "ai", "gpu", "cpu", "api", "apis", "ciso", "cfo", "dba",
    "sla", "ci/cd", "hsm", "pii", "fda", "ops", "grpc", "rest", "mvp", "id",
    "qa", "pr", "sla", "slo",
}

STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on", "at",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "we", "they", "our", "all", "any", "this", "that", "these", "those", "it",
    "its", "can", "could", "will", "would", "should", "must", "not", "no", "if",
    "so", "as", "than", "more", "most", "some", "such", "only", "also", "then",
    "under", "per", "across", "upon", "within", "without", "between", "while",
    "during", "total", "each", "both", "one", "two", "three", "four", "five",
}


def clean_title_casing(text: str) -> str:
    tokens = text.strip().split()
    if not tokens:
        return ""
    words = []
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if token_lower in COMMON_ACRONYMS:
            words.append(token_lower.upper())
        elif i > 0 and token_lower in {"and", "with", "for", "of", "in", "to", "vs", "on", "at"}:
            words.append(token_lower)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def extract_candidate_sentences(records: Sequence[TranscriptTurn]) -> List[Tuple[str, str, int]]:
    sentences: List[Tuple[str, str, int]] = []
    for r in records:
        txt = r.text.strip()
        chunks = re.split(r"(?<=[.!?])\s+", txt)
        for chunk in chunks:
            cleaned = chunk.strip().strip('"').strip("'")
            if len(cleaned) >= 8:
                sentences.append((cleaned, r.speaker, r.round_id))
    return sentences


def extract_candidate_phrases(
    records: Sequence[TranscriptTurn],
    allowed_speakers: Optional[Set[str]] = None,
    stop_words: Optional[Set[str]] = None,
) -> Counter:
    stops = stop_words or STOP_WORDS
    phrase_counts: Counter = Counter()

    for r in records:
        if allowed_speakers is not None and r.speaker not in allowed_speakers:
            continue
        # Split into distinct sentences first to prevent n-grams bridging across punctuation
        for sent in re.split(r"[.!?]+", r.text):
            tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", sent)
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    chunk = tokens[i : i + n]
                    first = chunk[0].lower()
                    last = chunk[-1].lower()
                    if first not in stops and last not in stops:
                        phrase = " ".join(chunk)
                        phrase_counts[phrase] += 1

    return phrase_counts


def score_phrases_against_vector(
    phrases: Counter,
    target_vector: np.ndarray,
    vectorizer: Any,
    top_k: int = 2,
) -> List[str]:
    norm_target = float(np.linalg.norm(target_vector))
    if norm_target < 1e-12 or vectorizer is None:
        return []

    scored: List[Tuple[float, str]] = []
    for phrase, count in phrases.items():
        try:
            vec = vectorizer.transform([phrase]).toarray()[0]
        except Exception:
            continue
        norm_v = float(np.linalg.norm(vec))
        if norm_v > 1e-12:
            sim = float(np.dot(vec, target_vector)) / (norm_v * norm_target)
            if sim > 0:
                score = sim * (1.0 + 0.25 * np.log1p(count))
                scored.append((score, phrase))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[str] = []
    for _, phrase in scored:
        phrase_lower = phrase.lower()
        if any(phrase_lower in existing.lower() or existing.lower() in phrase_lower for existing in selected):
            continue
        selected.append(clean_title_casing(phrase))
        if len(selected) >= top_k:
            break

    return selected


def find_representative_statement(
    sentences: Sequence[Tuple[str, str, int]],
    target_vector: np.ndarray,
    vectorizer: Any,
    max_round: int,
    allowed_speakers: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    norm_target = float(np.linalg.norm(target_vector))
    if not sentences:
        return "", ""

    best_statement = sentences[-1][0]
    best_speaker = sentences[-1][1]
    best_score = -1.0

    candidates = [
        s for s in sentences
        if allowed_speakers is None or s[1] in allowed_speakers
    ]
    if not candidates:
        candidates = list(sentences)

    for text, speaker, r_id in candidates:
        if norm_target >= 1e-12 and vectorizer is not None:
            try:
                vec = vectorizer.transform([text]).toarray()[0]
                norm_v = float(np.linalg.norm(vec))
                if norm_v > 1e-12:
                    sim = float(np.dot(vec, target_vector)) / (norm_v * norm_target)
                else:
                    sim = 0.0
            except Exception:
                sim = 0.0
        else:
            sim = 0.5

        recency_weight = 1.0 + 0.35 * (r_id / max(1, max_round))
        score = sim * recency_weight

        if score > best_score:
            best_score = score
            best_statement = text
            best_speaker = speaker

    return best_statement, best_speaker


def _compute_channel_dynamics_and_weights(
    Z: np.ndarray,
    speakers: Sequence[str],
    channel_member_lists: List[List[str]],
    is_dissent_flags: List[bool],
) -> Tuple[List[float], List[str]]:
    T, N, d = Z.shape
    K = len(channel_member_lists)
    if K == 0:
        return [], []

    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    z_bar_0 = Z[0].mean(axis=0)
    z_bar_T = Z[-1].mean(axis=0)

    s_scores: List[float] = []
    d_self_list: List[float] = []
    d_pull_list: List[float] = []

    for k in range(K):
        m_list = channel_member_lists[k]
        m_idx = [speaker_to_idx[m] for m in m_list if m in speaker_to_idx]
        if m_idx:
            C_k_0 = Z[0, m_idx, :].mean(axis=0)
            C_k_T = Z[-1, m_idx, :].mean(axis=0)
        else:
            C_k_0 = z_bar_0
            C_k_T = z_bar_T

        norm_C = float(np.linalg.norm(C_k_T))
        norm_z = float(np.linalg.norm(z_bar_T))
        if norm_C > 1e-12 and norm_z > 1e-12:
            cos_sim = float(np.dot(C_k_T, z_bar_T) / (norm_C * norm_z))
        else:
            cos_sim = 0.0
        s_k = max(0.0, cos_sim)
        s_scores.append(s_k)

        d_self = float(np.linalg.norm(C_k_T - C_k_0))
        d_pull = float(np.linalg.norm(z_bar_0 - C_k_0) - np.linalg.norm(z_bar_T - C_k_0))
        d_self_list.append(d_self)
        d_pull_list.append(d_pull)

    s_sum = sum(s_scores)
    if s_sum > 1e-12:
        alphas = [s / s_sum for s in s_scores]
    else:
        alphas = [1.0 / K for _ in range(K)]

    roles: List[str] = []
    for k in range(K):
        is_dissent = is_dissent_flags[k] if k < len(is_dissent_flags) else False
        s_k = s_scores[k]
        d_self_k = d_self_list[k]
        d_pull_k = d_pull_list[k]

        if is_dissent and s_k < 0.50:
            role = "Unreconciled Dissent"
        elif d_pull_k > d_self_k + 0.05:
            role = "Primary Attractor"
        elif d_self_k > d_pull_k + 0.05:
            role = "Conceding Coalition"
        else:
            role = "Synthesizing Compromiser"
        roles.append(role)

    weights = [round(float(a), 4) for a in alphas]
    if len(weights) > 0:
        weights[-1] = round(1.0 - sum(weights[:-1]), 4)

    return weights, roles


def synthesize_deliberation_topics(
    records: Sequence[TranscriptTurn],
    Z: np.ndarray,
    feature_names: Sequence[str],
    speakers: Sequence[str],
    rounds: Sequence[int],
    factions: Sequence[Faction],
    rogue_speaker: Optional[str] = None,
    vectorizer: Optional[Any] = None,
) -> DeliberationTopicSynthesis:
    T, N, d = Z.shape
    max_round = max(rounds) if rounds else 1
    feature_arr = np.asarray(feature_names)

    sentences = extract_candidate_sentences(records)
    all_phrases = extract_candidate_phrases(records)

    # 1. Consensus Centroid Concept & Statement
    final_centroid = Z[-1].mean(axis=0)
    centroid_phrases = score_phrases_against_vector(all_phrases, final_centroid, vectorizer, top_k=2)
    top_centroid_kw_idx = np.argsort(final_centroid)[::-1][:4]
    top_centroid_kw = [str(feature_arr[i]) for i in top_centroid_kw_idx if final_centroid[i] > 0]

    if centroid_phrases:
        centroid_concept = " & ".join(centroid_phrases)
    elif top_centroid_kw:
        centroid_concept = clean_title_casing(" ".join(top_centroid_kw[:2]) + " Architecture")
    else:
        centroid_concept = "Consensus Resolution"

    centroid_statement, _ = find_representative_statement(
        sentences=sentences,
        target_vector=final_centroid,
        vectorizer=vectorizer,
        max_round=max_round,
    )

    # 2. Principal Disagreement Axis via SVD on Delta
    Delta = Z - Z.mean(axis=1, keepdims=True)
    Delta_latest = Delta[-1]

    try:
        _, _, Vt = np.linalg.svd(Delta_latest, full_matrices=False)
        v1 = Vt[0]
    except Exception:
        v1 = np.ones(d)

    pos_weights = np.clip(v1, 0, None)
    neg_weights = np.clip(-v1, 0, None)

    pole_a_phrases = score_phrases_against_vector(all_phrases, pos_weights, vectorizer, top_k=1)
    pole_b_phrases = score_phrases_against_vector(all_phrases, neg_weights, vectorizer, top_k=1)

    pole_a_kw = [str(feature_arr[i]) for i in np.argsort(pos_weights)[::-1] if pos_weights[i] > 0][:3]
    pole_b_kw = [str(feature_arr[i]) for i in np.argsort(neg_weights)[::-1] if neg_weights[i] > 0][:3]

    pole_a_name = pole_a_phrases[0] if pole_a_phrases else clean_title_casing(" ".join(pole_a_kw))
    pole_b_name = pole_b_phrases[0] if pole_b_phrases else clean_title_casing(" ".join(pole_b_kw))

    if not pole_a_name:
        pole_a_name = "Operational Deployment"
    if not pole_b_name:
        pole_b_name = "Safety and Compliance Governance"

    center_of_disagreement = f"{pole_a_name} vs. {pole_b_name}"
    center_of_disagreement_statement = (
        f"The primary debate axis centers on the trade-off between {pole_a_name.lower()} "
        f"and {pole_b_name.lower()}."
    )
    disagreement_keywords = (pole_a_kw + pole_b_kw)[:6]

    # 3. Channel Dynamics, Faction Topics, Keywords, and Positions
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    channel_member_lists: List[List[str]] = [list(f.members) for f in factions]
    is_dissent_flags: List[bool] = [False] * len(factions)
    has_rogue = bool(rogue_speaker and rogue_speaker in speaker_to_idx)
    if has_rogue and rogue_speaker:
        channel_member_lists.append([rogue_speaker])
        is_dissent_flags.append(True)

    weights, roles = _compute_channel_dynamics_and_weights(
        Z=Z,
        speakers=speakers,
        channel_member_lists=channel_member_lists,
        is_dissent_flags=is_dissent_flags,
    )

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    consensus_vocab_pool: Set[str] = set()
    for p in centroid_phrases:
        consensus_vocab_pool.add(p.lower())
        for token in re.findall(r"\b[A-Za-z0-9_-]+\b", p.lower()):
            if token not in STOP_WORDS and len(token) > 2:
                consensus_vocab_pool.add(token)
    for kw in top_centroid_kw:
        consensus_vocab_pool.add(kw.lower())
    for token in re.findall(r"\b[A-Za-z0-9_-]+\b", centroid_statement.lower()):
        if token not in STOP_WORDS and len(token) > 2:
            consensus_vocab_pool.add(token)

    def _extract_contributed_terms(
        c_phrases: List[str],
        c_kw: List[str],
        fallback_topic: str,
    ) -> List[str]:
        matched: List[str] = []
        for p in c_phrases:
            p_clean = p.strip()
            if not p_clean:
                continue
            p_lower = p_clean.lower()
            if p_lower in consensus_vocab_pool:
                if p_clean not in matched:
                    matched.append(p_clean)
            elif p_lower in feature_to_idx and final_centroid[feature_to_idx[p_lower]] > 0:
                if p_clean not in matched:
                    matched.append(p_clean)

        for k in c_kw:
            k_clean = k.strip()
            if not k_clean:
                continue
            k_lower = k_clean.lower()
            if k_lower in consensus_vocab_pool:
                if k_clean not in matched:
                    matched.append(k_clean)
            elif k_lower in feature_to_idx and final_centroid[feature_to_idx[k_lower]] > 0:
                if k_clean not in matched:
                    matched.append(k_clean)

        terms = matched[:3]
        if not terms:
            terms = [k for k in c_kw[:2] if k]
        if not terms:
            terms = [fallback_topic] if fallback_topic else ["consensus"]
        return terms

    faction_stances: List[FactionStance] = []
    for f_idx, f in enumerate(factions):
        member_set = set(f.members)
        member_indices = [speaker_to_idx[m] for m in f.members if m in speaker_to_idx]
        if member_indices:
            f_centroid = Z[-1, member_indices, :].mean(axis=0)
            f_delta = f_centroid - final_centroid
        else:
            f_centroid = final_centroid
            f_delta = np.zeros(d)

        f_distinctive_idx = np.argsort(f_delta)[::-1][:4]
        f_kw = [str(feature_arr[i]) for i in f_distinctive_idx if f_delta[i] > 0]
        if not f_kw:
            f_kw = [str(feature_arr[i]) for i in np.argsort(f_centroid)[::-1][:4] if f_centroid[i] > 0]

        # Extract phrases constrained to this faction's members
        f_phrases_counter = extract_candidate_phrases(records, allowed_speakers=member_set)
        f_phrases = score_phrases_against_vector(f_phrases_counter, np.clip(f_delta, 0, None), vectorizer, top_k=1)
        if not f_phrases:
            f_phrases = score_phrases_against_vector(f_phrases_counter, f_centroid, vectorizer, top_k=1)
        if not f_phrases:
            f_phrases = score_phrases_against_vector(all_phrases, f_centroid, vectorizer, top_k=1)

        if f_phrases:
            f_topic = f_phrases[0]
        elif f_kw:
            f_topic = clean_title_casing(" ".join(f_kw[:2]) + " Stance")
        else:
            f_topic = f"Faction {f.id} Stance"

        f_stmt, f_spk = find_representative_statement(
            sentences=sentences,
            target_vector=f_centroid,
            vectorizer=vectorizer,
            max_round=max_round,
            allowed_speakers=member_set,
        )

        pos_summary = f"Endorses {f_topic.lower()}: \"{f_stmt}\"" if f_stmt else f"Supports {f_topic}."

        f_weight = weights[f_idx] if f_idx < len(weights) else 0.0
        f_role = roles[f_idx] if f_idx < len(roles) else "Synthesizing Compromiser"
        f_contributed = _extract_contributed_terms(f_phrases, f_kw, f_topic)

        faction_stances.append(
            FactionStance(
                id=f.id,
                members=f.members,
                topic=f_topic,
                keywords=f_kw,
                representative_statement=f_stmt,
                position_summary=pos_summary,
                consensus_weight=f_weight,
                dynamical_role=f_role,
                contributed_terms=f_contributed,
            )
        )

    # 4. Rogue Entity Stance & Statement
    rogue_stance: Optional[FactionStance] = None
    if has_rogue and rogue_speaker:
        r_idx = speaker_to_idx[rogue_speaker]
        r_state = Z[-1, r_idx, :]
        r_delta = r_state - final_centroid

        r_distinctive_idx = np.argsort(r_delta)[::-1][:4]
        r_kw = [str(feature_arr[i]) for i in r_distinctive_idx if r_delta[i] > 0]
        if not r_kw:
            r_kw = [str(feature_arr[i]) for i in np.argsort(r_state)[::-1][:4] if r_state[i] > 0]

        # Extract phrases constrained strictly to rogue speaker
        r_phrases_counter = extract_candidate_phrases(records, allowed_speakers={rogue_speaker})
        r_phrases = score_phrases_against_vector(r_phrases_counter, np.clip(r_delta, 0, None), vectorizer, top_k=1)
        if not r_phrases:
            r_phrases = score_phrases_against_vector(r_phrases_counter, r_state, vectorizer, top_k=1)

        if r_phrases:
            r_topic = r_phrases[0]
        elif r_kw:
            r_topic = clean_title_casing(" ".join(r_kw[:2]) + " Dissent")
        else:
            r_topic = "Contrarian Dissent"

        r_stmt, _ = find_representative_statement(
            sentences=sentences,
            target_vector=r_delta if float(np.linalg.norm(r_delta)) > 1e-12 else r_state,
            vectorizer=vectorizer,
            max_round=max_round,
            allowed_speakers={rogue_speaker},
        )

        r_summary = f"Holds dissenting position on {r_topic.lower()}: \"{r_stmt}\""

        r_weight = weights[-1] if weights else 0.0
        r_role = roles[-1] if roles else "Unreconciled Dissent"
        r_contributed = _extract_contributed_terms(r_phrases, r_kw, r_topic)

        rogue_stance = FactionStance(
            id=99,
            members=[rogue_speaker],
            topic=r_topic,
            keywords=r_kw,
            representative_statement=r_stmt,
            position_summary=r_summary,
            consensus_weight=r_weight,
            dynamical_role=r_role,
            contributed_terms=r_contributed,
        )

    k_star_T = compute_effective_channels_k_star(Z[-1])
    k_star_0 = compute_effective_channels_k_star(Z[0])
    total_channels = len(faction_stances) + (1 if rogue_stance else 0)
    if total_channels == 0:
        total_channels = 1

    return DeliberationTopicSynthesis(
        center_of_disagreement=center_of_disagreement,
        center_of_disagreement_statement=center_of_disagreement_statement,
        centroid_concept=centroid_concept,
        centroid_statement=centroid_statement,
        disagreement_poles=(pole_a_name, pole_b_name),
        disagreement_keywords=disagreement_keywords,
        factions=faction_stances,
        rogue_stance=rogue_stance,
        effective_channels_k_star=k_star_T,
        initial_k_star=k_star_0,
        total_channels_identified=total_channels,
    )
