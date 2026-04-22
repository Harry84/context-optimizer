"""
Stage 3 — Relevance Scoring Engine.

Orchestrates keyword, semantic, and recency scorers into a composite
score for each history turn against the current query.

CRITICAL: Only turns 0..query_position-1 are scored.
Turn query_position is the current query — not part of the context window.
"""

from __future__ import annotations

import numpy as np

from src.ingestion.models import OptimizerConfig, Turn
from src.scoring.keyword import keyword_scores
from src.scoring.query_classifier import QueryType, classify_query
from src.scoring.recency import recency_scores
from src.scoring.semantic import semantic_scores


def score_turns(
    history: list[Turn],
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> list[Turn]:
    """
    Score each history turn for relevance to the current query.
    Mutates turns in-place by setting turn.score.

    Args:
        history:        Turns 0..query_position-1 (pre-sliced by caller).
        query:          The current query string.
        query_position: Index of the query turn in the full conversation.
                        Used for recency decay calculation.
        config:         OptimizerConfig with weights, lambda, landmark_boost.

    Returns:
        The same list of turns with .score populated.
    """
    if not history:
        return history

    query_type = classify_query(query)
    weights    = config.weights[query_type]
    w1, w2, w3 = weights["keyword"], weights["semantic"], weights["recency"]

    texts        = [t.text for t in history]
    turn_indices = [t.turn_index for t in history]

    # Compute raw component scores
    kw_scores  = keyword_scores(query, texts)
    sem_scores = semantic_scores(query, texts, config.embedding_model)
    rec_scores = recency_scores(turn_indices, query_position, config.lambda_decay)

    # Normalise each component to [0, 1] across the history window
    kw_scores  = _normalise(kw_scores)
    sem_scores = _normalise(sem_scores)
    rec_scores = _normalise(rec_scores)

    # Combine and apply landmark boost
    for i, turn in enumerate(history):
        base_score = (
            w1 * kw_scores[i]
            + w2 * sem_scores[i]
            + w3 * rec_scores[i]
        )
        boost = config.landmark_boost if turn.is_landmark else 0.0
        turn.score = min(1.0, base_score + boost)

    return history


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not scores:
        return scores
    arr = np.array(scores, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return [0.5] * len(scores)   # all scores identical — return midpoint
    return ((arr - lo) / (hi - lo)).tolist()
