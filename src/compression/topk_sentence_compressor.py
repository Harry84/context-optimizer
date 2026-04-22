"""
v4 — Top-K Sentence Retrieval Compressor.

Combines sentence-level splitting of landmark turns with query-aware
top-K retrieval across ALL units — including landmark sentences.

Key insight: landmark detection is query-agnostic. A turn flagged as a
landmark because it contains a date slot is not necessarily relevant to a
query about prices. Hard-KEEPing all landmarks wastes context budget on
query-irrelevant facts.

v4 scores every unit (landmark sentences and non-landmark turns alike)
against the query. Landmark units receive a score boost (landmark_boost)
to bias towards them, but they compete in the same top-K pool. This means
query-irrelevant landmarks can be outcompeted by non-landmark units that
are more directly relevant to the current query.

Pipeline:
  1. Split landmark turns into sentences; detect landmark sentences
  2. Score ALL units against the query in one batch
  3. Apply landmark boost to landmark units
  4. Apply noise floor; keep top K% of ALL units (topk_sentence_fraction)
  5. Promote sandwiched COMPRESS sentences within landmark turns
  6. Merge same-turn same-disposition units
  7. Group into runs
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from src.compression.compressor import Run, _merge_singleton_compress_runs
from src.compression.sentence_splitter import split_sentences
from src.ingestion.models import OptimizerConfig, Turn
from src.landmarks.rule_detector import (
    _has_slot_signal,
    _is_assistant_offer,
    _is_conversation_close,
    _is_pure_filler,
    _STRONG_CONFIRMATION,
    _INTENT_VERB_PATTERNS,
    _ACTION_PATTERNS,
)
from src.scoring.keyword import keyword_scores
from src.scoring.semantic import semantic_scores
from src.scoring.recency import recency_scores
from src.scoring.scorer import _normalise


@dataclass
class Unit:
    """A scoreable unit — either a whole turn or a sentence from a landmark turn."""
    turn_index:         int
    sentence_idx:       int
    speaker:            str
    text:               str
    is_landmark:        bool = False
    landmark_type:      str | None = None
    score:              float = 0.0
    disposition:        str = ""
    from_landmark_turn: bool = False


def _effective(disposition: str) -> str:
    return "KEEP" if disposition in ("KEEP", "CANDIDATE") else "COMPRESS"


def _sentence_is_landmark(text: str, speaker: str) -> tuple[bool, str | None]:
    text_l = text.lower().strip()
    if _is_pure_filler(text):
        return False, None
    if speaker == "USER":
        if _STRONG_CONFIRMATION.search(text_l):
            return True, "decision"
        if _is_conversation_close(text):
            return True, "decision"
        for p in _INTENT_VERB_PATTERNS:
            if p.search(text_l):
                return True, "intent"
        if _has_slot_signal(text):
            return True, "intent"
    if speaker == "ASSISTANT":
        if _is_assistant_offer(text):
            return True, "decision"
        if _has_slot_signal(text):
            return True, "decision"
        for p in _ACTION_PATTERNS:
            if p.search(text_l):
                return True, "action_item"
    return False, None


def _build_units(history: list[Turn]) -> list[Unit]:
    """Non-landmark turns → one Unit. Landmark turns → one Unit per sentence."""
    units: list[Unit] = []
    for turn in history:
        if not turn.is_landmark:
            units.append(Unit(
                turn_index=turn.turn_index,
                sentence_idx=0,
                speaker=turn.speaker,
                text=turn.text,
                is_landmark=False,
                from_landmark_turn=False,
            ))
        else:
            sentences = split_sentences(turn.text)
            for idx, text in enumerate(sentences):
                is_lm, lm_type = _sentence_is_landmark(text, turn.speaker)
                units.append(Unit(
                    turn_index=turn.turn_index,
                    sentence_idx=idx,
                    speaker=turn.speaker,
                    text=text,
                    is_landmark=is_lm,
                    landmark_type=lm_type,
                    from_landmark_turn=True,
                ))
    return units


def _score_all_units(
    units: list[Unit],
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> None:
    """Score ALL units against the query — including landmark sentences.
    Landmark units get a boost but are not hard-scored at 1.0."""
    if not units:
        return

    texts        = [u.text for u in units]
    turn_indices = [u.turn_index for u in units]

    kw_scores  = _normalise(keyword_scores(query, texts))
    sem_scores = _normalise(semantic_scores(query, texts, config.embedding_model))
    rec_scores = _normalise(recency_scores(turn_indices, query_position, config.lambda_decay))

    w1, w2, w3 = 0.35, 0.50, 0.15

    for i, u in enumerate(units):
        base  = w1 * kw_scores[i] + w2 * sem_scores[i] + w3 * rec_scores[i]
        boost = config.landmark_boost if u.is_landmark else 0.0
        u.score = min(1.0, base + boost)


def _classify_units_topk(
    units: list[Unit],
    query_type: str,
    config: OptimizerConfig,
) -> list[Unit]:
    """
    Classify ALL units using topk_sentence_fraction and noise floor.

    Uses topk_sentence_fraction (not topk_fraction) because:
    - Unit count is higher after sentence splitting
    - Landmarks compete rather than hard-KEEP, so a higher fraction
      is needed to capture all genuinely relevant content
    """
    fraction  = config.topk_sentence_fraction[query_type]
    min_score = config.topk_min_score

    candidates = [u for u in units if u.score >= min_score]
    noise      = [u for u in units if u.score <  min_score]

    for u in noise:
        u.disposition = "COMPRESS"

    k = max(1, math.ceil(fraction * len(units)))
    ranked   = sorted(candidates, key=lambda u: u.score, reverse=True)
    keep_ids = {id(u) for u in ranked[:k]}

    for u in candidates:
        u.disposition = "KEEP" if id(u) in keep_ids else "COMPRESS"

    return units


def _promote_sandwiched_units(units: list[Unit]) -> list[Unit]:
    """Promote COMPRESS units sandwiched between KEEPs within a landmark turn."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        if u.from_landmark_turn:
            by_turn[u.turn_index].append(i)

    for indices in by_turn.values():
        if len(indices) <= 1:
            continue
        turn_units = [units[i] for i in indices]
        has_keep_before = False
        for j, u in enumerate(turn_units):
            if _effective(u.disposition) == "KEEP":
                has_keep_before = True
            elif u.disposition == "COMPRESS" and has_keep_before:
                has_keep_after = any(
                    _effective(turn_units[k].disposition) == "KEEP"
                    for k in range(j + 1, len(turn_units))
                )
                if has_keep_after:
                    u.disposition = "CANDIDATE"

    return units


def _merge_same_turn_units(units: list[Unit]) -> list[Unit]:
    """Merge consecutive same-turn same-effective-disposition units."""
    if not units:
        return []

    merged: list[Unit] = []
    for u in units:
        if (merged
                and merged[-1].turn_index == u.turn_index
                and merged[-1].speaker == u.speaker
                and _effective(merged[-1].disposition) == _effective(u.disposition)):
            merged[-1].text = merged[-1].text + " " + u.text
            if u.is_landmark and not merged[-1].is_landmark:
                merged[-1].is_landmark   = True
                merged[-1].landmark_type = u.landmark_type
            merged[-1].score = max(merged[-1].score, u.score)
        else:
            merged.append(Unit(
                turn_index=u.turn_index,
                sentence_idx=u.sentence_idx,
                speaker=u.speaker,
                text=u.text,
                is_landmark=u.is_landmark,
                landmark_type=u.landmark_type,
                score=u.score,
                disposition=u.disposition,
                from_landmark_turn=u.from_landmark_turn,
            ))
    return merged


def _units_to_runs(units: list[Unit]) -> list[Run]:
    if not units:
        return []

    units = _promote_sandwiched_units(units)
    units = _merge_same_turn_units(units)

    runs: list[Run] = []
    for u in units:
        eff = _effective(u.disposition)
        synthetic_turn = Turn(turn_index=u.turn_index, speaker=u.speaker, text=u.text)
        synthetic_turn.is_landmark   = u.is_landmark
        synthetic_turn.landmark_type = u.landmark_type
        synthetic_turn.score         = u.score
        synthetic_turn.disposition   = u.disposition

        if runs and runs[-1][0] == eff:
            runs[-1][1].append(synthetic_turn)
        else:
            runs.append((eff, [synthetic_turn]))

    runs = _merge_singleton_compress_runs(runs)
    return runs


def topk_sentence_runs(
    history: list[Turn],
    query: str,
    query_position: int,
    query_type: str,
    config: OptimizerConfig,
) -> list[Run]:
    """v4: split landmark turns → score all units → top-K across all units."""
    units = _build_units(history)
    _score_all_units(units, query, query_position, config)
    units = _classify_units_topk(units, query_type, config)
    return _units_to_runs(units)
