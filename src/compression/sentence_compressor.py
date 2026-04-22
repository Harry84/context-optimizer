"""
v2 Sentence-Level Compressor.

Replaces compressor.py's turn-level classify_turns() with sentence-level
classification. Landmark turns are split into sentences; only the sentences
that themselves match landmark patterns are hard-KEEPed. Non-landmark
sentences in a landmark turn are scored independently against the query.

Returns the same list[Run] type as compressor.py so the assembler is
completely unchanged.

Key differences from v1:
  v1: turn.is_landmark=True → entire turn KEEPed verbatim
  v2: turn.is_landmark=True → split into sentences → only landmark
      sentences KEEPed; filler sentences scored independently and compressed.
      Within a landmark turn, COMPRESS sentences sandwiched between KEEP
      sentences are promoted to KEEP to maintain turn coherence and prevent
      [context continues] bridges in the assembled output.
      Tighter sentence_thresholds used because individual sentences have
      less context, making semantic scores noisier than at turn level.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

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
class Sentence:
    """A single sentence extracted from a Turn."""
    turn_index:    int
    sentence_idx:  int
    speaker:       str
    text:          str
    is_landmark:   bool = False
    landmark_type: str | None = None
    score:         float = 0.0
    disposition:   str = ""


def _effective(disposition: str) -> str:
    """KEEP and CANDIDATE are both effectively KEEP for grouping purposes."""
    return "KEEP" if disposition in ("KEEP", "CANDIDATE") else "COMPRESS"


def _sentence_is_landmark(text: str, speaker: str) -> tuple[bool, str | None]:
    """Re-run landmark pattern matching at sentence level."""
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


def _split_turn_into_sentences(turn: Turn) -> list[Sentence]:
    """Split a landmark Turn into Sentence objects, re-running landmark
    patterns at sentence level so only triggering sentences are KEEPed."""
    raw_sentences = split_sentences(turn.text)
    sentences = []
    for idx, text in enumerate(raw_sentences):
        is_lm, lm_type = _sentence_is_landmark(text, turn.speaker)
        sentences.append(Sentence(
            turn_index=turn.turn_index,
            sentence_idx=idx,
            speaker=turn.speaker,
            text=text,
            is_landmark=is_lm,
            landmark_type=lm_type,
        ))
    return sentences


def _score_non_landmark_sentences(
    sentences: list[Sentence],
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> list[Sentence]:
    """Score non-landmark sentences independently against the query."""
    non_lm = [s for s in sentences if not s.is_landmark]

    if not non_lm:
        for s in sentences:
            s.score = 1.0
        return sentences

    texts        = [s.text for s in non_lm]
    turn_indices = [s.turn_index for s in non_lm]

    kw_scores  = _normalise(keyword_scores(query, texts))
    sem_scores = _normalise(semantic_scores(query, texts, config.embedding_model))
    rec_scores = _normalise(recency_scores(turn_indices, query_position, config.lambda_decay))

    w1, w2, w3 = 0.35, 0.50, 0.15

    score_map: dict[int, float] = {}
    for i, s in enumerate(non_lm):
        score_map[id(s)] = w1 * kw_scores[i] + w2 * sem_scores[i] + w3 * rec_scores[i]

    for s in sentences:
        s.score = 1.0 if s.is_landmark else score_map.get(id(s), 0.0)

    return sentences


def _classify_sentences(
    sentences: list[Sentence],
    query_type: str,
    config: OptimizerConfig,
) -> list[Sentence]:
    """Assign KEEP/CANDIDATE/COMPRESS using tighter sentence_thresholds."""
    high = config.sentence_thresholds[query_type]["high"]
    low  = config.sentence_thresholds[query_type]["low"]

    for s in sentences:
        if s.is_landmark:
            s.disposition = "KEEP"
        elif s.score >= high:
            s.disposition = "KEEP"
        elif s.score >= low:
            s.disposition = "CANDIDATE"
        else:
            s.disposition = "COMPRESS"

    return sentences


def _promote_sandwiched_sentences(sentences: list[Sentence]) -> list[Sentence]:
    """
    Within each landmark turn, promote COMPRESS sentences that are sandwiched
    between KEEP sentences to KEEP.

    A COMPRESS sentence flanked by KEEPs from the same turn causes alternating
    KEEP/COMPRESS/KEEP groups, which after assembly produce:
        [USER] kept sentence
        [ASSISTANT] [context continues]   ← bridge
        [USER] another kept sentence

    Promoting the sandwiched COMPRESS to KEEP allows all three to merge into
    one USER turn. The cost is keeping a few extra words; the benefit is a
    coherent thread structure with no spurious bridges.

    Only applies within individual turns — does not affect cross-turn grouping.
    """
    # Group sentences by turn_index for per-turn processing
    by_turn: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(sentences):
        by_turn[s.turn_index].append(i)

    for turn_idx, indices in by_turn.items():
        if len(indices) <= 1:
            continue

        turn_sentences = [sentences[i] for i in indices]

        # Only process landmark turns that were split
        if not any(s.is_landmark for s in turn_sentences):
            continue

        # Find sandwiched COMPRESS sentences:
        # a sentence at position j is sandwiched if there exists a KEEP
        # before it and a KEEP after it within the same turn
        has_keep_before = False
        for j, s in enumerate(turn_sentences):
            if _effective(s.disposition) == "KEEP":
                has_keep_before = True
            elif s.disposition == "COMPRESS" and has_keep_before:
                # Check if there's a KEEP after this position
                has_keep_after = any(
                    _effective(turn_sentences[k].disposition) == "KEEP"
                    for k in range(j + 1, len(turn_sentences))
                )
                if has_keep_after:
                    s.disposition = "CANDIDATE"  # promote to KEEP group

    return sentences


def _merge_same_turn_sentences(sentences: list[Sentence]) -> list[Sentence]:
    """
    Merge consecutive sentences from the same turn with the same effective
    disposition into a single sentence.
    """
    if not sentences:
        return []

    merged: list[Sentence] = []
    for s in sentences:
        if (merged
                and merged[-1].turn_index == s.turn_index
                and merged[-1].speaker == s.speaker
                and _effective(merged[-1].disposition) == _effective(s.disposition)):
            merged[-1].text = merged[-1].text + " " + s.text
            if s.is_landmark and not merged[-1].is_landmark:
                merged[-1].is_landmark   = True
                merged[-1].landmark_type = s.landmark_type
            merged[-1].score = max(merged[-1].score, s.score)
        else:
            merged.append(Sentence(
                turn_index=s.turn_index,
                sentence_idx=s.sentence_idx,
                speaker=s.speaker,
                text=s.text,
                is_landmark=s.is_landmark,
                landmark_type=s.landmark_type,
                score=s.score,
                disposition=s.disposition,
            ))

    return merged


def _sentences_to_runs(sentences: list[Sentence]) -> list[Run]:
    """
    Promote sandwiched sentences, merge same-turn sentences, then group
    into contiguous KEEP/COMPRESS runs.
    """
    if not sentences:
        return []

    sentences = _promote_sandwiched_sentences(sentences)
    sentences = _merge_same_turn_sentences(sentences)

    runs: list[Run] = []
    for s in sentences:
        eff = _effective(s.disposition)
        synthetic_turn = Turn(turn_index=s.turn_index, speaker=s.speaker, text=s.text)
        synthetic_turn.is_landmark   = s.is_landmark
        synthetic_turn.landmark_type = s.landmark_type
        synthetic_turn.score         = s.score
        synthetic_turn.disposition   = s.disposition

        if runs and runs[-1][0] == eff:
            runs[-1][1].append(synthetic_turn)
        else:
            runs.append((eff, [synthetic_turn]))

    runs = _merge_singleton_compress_runs(runs)
    return runs


def classify_turns_sentence_level(
    history: list[Turn],
    query: str,
    query_position: int,
    query_type: str,
    config: OptimizerConfig,
) -> list[Run]:
    """
    Sentence-level compression pipeline.

    Non-landmark turns: treated atomically (same as v1).
    Landmark turns: split → sentence-level landmark detection →
    independent scoring → classify → promote sandwiched sentences →
    merge same-turn same-disposition sentences → group into runs.
    """
    all_sentences: list[Sentence] = []

    for turn in history:
        if not turn.is_landmark:
            all_sentences.append(Sentence(
                turn_index=turn.turn_index,
                sentence_idx=0,
                speaker=turn.speaker,
                text=turn.text,
                is_landmark=False,
                score=turn.score,
            ))
        else:
            sentences = _split_turn_into_sentences(turn)
            sentences = _score_non_landmark_sentences(
                sentences, query, query_position, config
            )
            all_sentences.extend(sentences)

    all_sentences = _classify_sentences(all_sentences, query_type, config)
    return _sentences_to_runs(all_sentences)
