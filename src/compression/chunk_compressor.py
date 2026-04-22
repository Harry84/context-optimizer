"""
v5 — Chunk-Based Retrieval Compressor.

Scores overlapping multi-turn chunks against the query rather than
individual turns or sentences.

Motivation: the answer to a query is often spread across multiple
consecutive turns. Scoring turns individually misses this — a turn saying
"Okay I'm going with Virgin" scores moderately alone, but as part of a
chunk that includes the flight options and the decision reasoning it scores
very highly against "what flights were compared and what did the user decide?"

Approach:
  1. Build overlapping windows of chunk_size turns with chunk_stride step
  2. Score each chunk + each individual turn against the query (one batch)
  3. Each turn's score = 0.7 * max_chunk_score + 0.3 * individual_score
     The individual component penalises turns that ride a good chunk score
     despite being individually irrelevant (e.g. greetings co-chunked with
     relevant content)
  4. Apply scaled landmark boost: boost = landmark_boost * individual_score
     Relevant landmarks get a strong boost; irrelevant ones get almost none
  5. Top-K on blended scores using chunk_topk_fraction
  6. Landmarks compete in top-K — no hard-KEEP
"""

from __future__ import annotations

import math

from src.compression.compressor import Run, group_into_runs
from src.ingestion.models import OptimizerConfig, Turn
from src.scoring.keyword import keyword_scores
from src.scoring.semantic import semantic_scores
from src.scoring.recency import recency_scores
from src.scoring.scorer import _normalise

# Blend weights: chunk context vs individual turn relevance
_CHUNK_WEIGHT      = 0.7
_INDIVIDUAL_WEIGHT = 0.3


def _build_chunks(
    history: list[Turn],
    chunk_size: int,
    stride: int,
) -> list[tuple[str, list[int]]]:
    """
    Build overlapping multi-turn chunks from history.
    Returns list of (chunk_text, [turn_indices]).
    """
    chunks = []
    n = len(history)
    for start in range(0, n, stride):
        end = min(start + chunk_size, n)
        window = history[start:end]
        text = " ".join(f"{t.speaker}: {t.text}" for t in window)
        indices = [t.turn_index for t in window]
        chunks.append((text, indices))
        if end == n:
            break
    return chunks


def _score_turns_by_chunks(
    history: list[Turn],
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> None:
    """
    Score each turn using a blend of chunk-level and individual-level signals.

    All chunks AND individual turns scored in one batch for efficiency.

    final_score = 0.7 * max_chunk_score + 0.3 * individual_score
                + landmark_boost * individual_score  (if landmark)

    The chunk component captures answer-spanning relevance.
    The individual component prevents irrelevant turns from riding a good
    chunk score (e.g. a greeting co-chunked with relevant flight content).
    The scaled landmark boost rewards genuinely relevant landmarks without
    inflating query-irrelevant ones — a greeting landmark gets boost ≈ 0.15 × 0.3
    while a flight options landmark gets boost ≈ 0.85 × 0.3.

    Mutates turns in-place.
    """
    chunks = _build_chunks(history, config.chunk_size, config.chunk_stride)
    if not chunks:
        return

    chunk_texts   = [text for text, _ in chunks]
    chunk_indices = [indices for _, indices in chunks]
    turn_texts    = [t.text for t in history]
    turn_indices  = [t.turn_index for t in history]

    # Score chunks and individual turns in one combined batch
    all_texts = chunk_texts + turn_texts
    kw_all  = _normalise(keyword_scores(query, all_texts))
    sem_all = _normalise(semantic_scores(query, all_texts, config.embedding_model))

    n_chunks = len(chunks)
    n_turns  = len(history)

    chunk_last = [idxs[-1] for idxs in chunk_indices]
    rec_chunk  = _normalise(recency_scores(chunk_last, query_position, config.lambda_decay))
    rec_turn   = _normalise(recency_scores(turn_indices, query_position, config.lambda_decay))

    w1, w2, w3 = 0.35, 0.50, 0.15

    chunk_scores = [
        w1 * kw_all[i] + w2 * sem_all[i] + w3 * rec_chunk[i]
        for i in range(n_chunks)
    ]

    indiv_scores = [
        w1 * kw_all[n_chunks + i] + w2 * sem_all[n_chunks + i] + w3 * rec_turn[i]
        for i in range(n_turns)
    ]

    # Map max chunk score to each turn
    turn_max_chunk: dict[int, float] = {}
    for i, indices in enumerate(chunk_indices):
        for turn_idx in indices:
            turn_max_chunk[turn_idx] = max(
                turn_max_chunk.get(turn_idx, 0.0), chunk_scores[i]
            )

    # Blend and apply scaled landmark boost
    for i, turn in enumerate(history):
        chunk_s = turn_max_chunk.get(turn.turn_index, 0.0)
        indiv_s = indiv_scores[i]
        blended = _CHUNK_WEIGHT * chunk_s + _INDIVIDUAL_WEIGHT * indiv_s
        # Scale boost by individual score: relevant landmarks boosted strongly,
        # irrelevant landmarks (greetings etc.) boosted minimally
        boost = config.landmark_boost * indiv_s if turn.is_landmark else 0.0
        turn.score = min(1.0, blended + boost)


def classify_turns_chunk_topk(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
    """
    Classify turns using blended scores and top-K.
    Landmarks compete — no hard-KEEP.
    """
    fraction  = config.chunk_topk_fraction[query_type]
    min_score = config.topk_min_score

    candidates = [t for t in history if t.score >= min_score]
    noise      = [t for t in history if t.score <  min_score]

    for t in noise:
        t.disposition = "COMPRESS"

    k = max(1, math.ceil(fraction * len(history)))
    ranked   = sorted(candidates, key=lambda t: t.score, reverse=True)
    keep_ids = {id(t) for t in ranked[:k]}

    for t in candidates:
        t.disposition = "KEEP" if id(t) in keep_ids else "COMPRESS"

    return history


def chunk_topk_runs(
    history: list[Turn],
    query: str,
    query_position: int,
    query_type: str,
    config: OptimizerConfig,
) -> list[Run]:
    """v5: chunk + individual blended scoring → scaled landmark boost → top-K."""
    _score_turns_by_chunks(history, query, query_position, config)
    classify_turns_chunk_topk(history, query_type, config)
    return group_into_runs(history)
