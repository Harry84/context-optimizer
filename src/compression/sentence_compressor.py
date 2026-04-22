"""
v2 Sentence-Level Compressor.

Replaces compressor.py's turn-level classify_turns() with sentence-level
classification. Landmark turns are split into sentences; only the sentences
that themselves match landmark patterns are hard-KEEPed. Non-landmark
sentences in a landmark turn are scored independently against the query.

Returns the same list[Run] type as compressor.py so the assembler is
completely unchanged.

Key difference from v1:
  v1: turn.is_landmark=True → entire turn KEEPed verbatim
  v2: turn.is_landmark=True → split into sentences → only landmark
      sentences KEEPed; filler sentences scored independently and compressed
"""

from __future__ import annotations

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


def _sentence_is_landmark(text: str, speaker: str) -> tuple[bool, str | None]:
    """
    Re-run landmark pattern matching at sentence level.
    Same logic as rule_detector._pass1 but on a single sentence.
    """
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
    """
    Split a landmark Turn into Sentence objects, re-running landmark
    patterns at sentence level so only triggering sentences are KEEPed.
    """
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
    """
    Score non-landmark sentences independently against the query.

    Landmark sentences get score=1.0 (hard KEEP).
    Non-landmark sentences get a proper keyword+semantic+recency score
    computed on their individual text — this is the key fix over v1,
    which incorrectly inherited the parent turn's (high) score.
    """
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

    # Use factual weights for sentence scoring — sentences are short,
    # keyword and semantic signals matter more than recency at this granularity
    w1, w2, w3 = 0.35, 0.50, 0.15

    score_map: dict[int, float] = {}
    for i, s in enumerate(non_lm):
        score_map[id(s)] = (
            w1 * kw_scores[i]
            + w2 * sem_scores[i]
            + w3 * rec_scores[i]
        )

    for s in sentences:
        if s.is_landmark:
            s.score = 1.0
        else:
            s.score = score_map.get(id(s), 0.0)

    return sentences


def _classify_sentences(
    sentences: list[Sentence],
    query_type: str,
    config: OptimizerConfig,
) -> list[Sentence]:
    """Assign KEEP/CANDIDATE/COMPRESS disposition to each sentence."""
    high = config.thresholds[query_type]["high"]
    low  = config.thresholds[query_type]["low"]

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


def _sentences_to_runs(sentences: list[Sentence]) -> list[Run]:
    """
    Group sentences into contiguous KEEP/COMPRESS runs.
    Sentences are re-wrapped as synthetic Turn objects for assembler compatibility.
    """
    if not sentences:
        return []

    runs: list[Run] = []
    for s in sentences:
        effective = "KEEP" if s.disposition in ("KEEP", "CANDIDATE") else "COMPRESS"

        synthetic_turn = Turn(turn_index=s.turn_index, speaker=s.speaker, text=s.text)
        synthetic_turn.is_landmark   = s.is_landmark
        synthetic_turn.landmark_type = s.landmark_type
        synthetic_turn.score         = s.score
        synthetic_turn.disposition   = s.disposition

        if runs and runs[-1][0] == effective:
            runs[-1][1].append(synthetic_turn)
        else:
            runs.append((effective, [synthetic_turn]))

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
    Sentence-level equivalent of compressor.classify_turns() + group_into_runs().

    Non-landmark turns: treated atomically (same as v1).
    Landmark turns: split into sentences, landmark patterns re-run per
    sentence, non-landmark sentences scored independently against query.

    Returns list[Run] — same type as compressor.group_into_runs().
    """
    all_sentences: list[Sentence] = []

    for turn in history:
        if not turn.is_landmark:
            # Non-landmark turn: atomic, inherit turn score
            s = Sentence(
                turn_index=turn.turn_index,
                sentence_idx=0,
                speaker=turn.speaker,
                text=turn.text,
                is_landmark=False,
                score=turn.score,
            )
            all_sentences.append(s)
        else:
            # Landmark turn: split, re-classify, score independently
            sentences = _split_turn_into_sentences(turn)
            sentences = _score_non_landmark_sentences(
                sentences, query, query_position, config
            )
            all_sentences.extend(sentences)

    all_sentences = _classify_sentences(all_sentences, query_type, config)
    return _sentences_to_runs(all_sentences)
