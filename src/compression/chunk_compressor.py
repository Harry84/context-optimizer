"""
v5 — Chunk-Based Retrieval Compressor.

Scores overlapping multi-turn chunks against the query rather than
individual turns or sentences.

Approach:
  1. Build overlapping windows of chunk_size turns with chunk_stride step
  2. Score each chunk + each individual turn against the query (one batch)
  3. Each turn's score = 0.7 * max_chunk_score + 0.3 * individual_score
  4. Apply scaled landmark boost: boost = landmark_boost * individual_score
  5. If query is airport-related: apply airport floor to turns mentioning
     IATA codes or airport names (not city names — too broad)
  6. Top-K on blended scores using chunk_topk_fraction
  7. Landmarks compete in top-K — no hard-KEEP
"""

from __future__ import annotations

import math
import re

from src.compression.compressor import Run, group_into_runs
from src.ingestion.models import OptimizerConfig, Turn
from src.scoring.keyword import keyword_scores
from src.scoring.semantic import semantic_scores
from src.scoring.recency import recency_scores
from src.scoring.scorer import _normalise

_CHUNK_WEIGHT        = 0.7
_INDIVIDUAL_WEIGHT   = 0.3
_AIRPORT_SCORE_FLOOR = 0.45

# Detects airport-related queries — used to decide whether to apply the floor
_AIRPORT_QUERY_RE = re.compile(
    r"\b(airport|terminal|hub|fly(ing)? (from|to|into|out of)|depart(ing)? from|"
    r"arriv(ing)? (at|in)|origin|destination)\b",
    re.IGNORECASE,
)

# Detects airport signals in turn text — IATA codes and airport names only.
# City names deliberately excluded: too broad and apply the floor to nearly
# every turn in a flight booking conversation, losing discriminating power.
_AIRPORT_TURN_RE = re.compile(
    r"\b("
    # IATA codes
    r"JFK|LHR|LGA|EWR|LAX|ORD|ATL|DFW|SFO|MIA|BOS|SEA|DEN|LAS|PHX|MCO|"
    r"IAH|MSP|DTW|PHL|FLL|BWI|DCA|IAD|MDW|HOU|SAN|TPA|PDX|STL|MCI|"
    r"LGW|STN|LCY|MAN|EDI|GLA|BHX|LPL|BRS|NCL|"
    r"CDG|ORY|AMS|FRA|MUC|ZRH|BCN|MAD|FCO|MXP|LIN|VIE|BRU|CPH|OSL|ARN|HEL|"
    r"DXB|AUH|DOH|SIN|NRT|HND|ICN|HKG|PEK|PVG|BKK|KUL|SYD|MEL|YYZ|YVR|GRU|EZE|"
    # Airport names / nicknames (not city names)
    r"Heathrow|Gatwick|Stansted|Luton|City Airport|"
    r"O'?Hare|Midway|Charles de Gaulle|Orly|Schiphol|"
    r"Newark|Kennedy|LaGuardia|Logan|Dulles|Reagan|"
    r"Changi|Narita|Haneda|Incheon|Pearson|"
    r"airport|terminal"
    r")\b",
    re.IGNORECASE,
)


def _is_airport_query(query: str) -> bool:
    """Return True if the query is asking about airports/terminals/routing."""
    return bool(_AIRPORT_QUERY_RE.search(query))


def _has_airport_signal(text: str) -> bool:
    """Return True if the turn mentions an IATA code or airport name."""
    return bool(_AIRPORT_TURN_RE.search(text))


def _build_chunks(
    history: list[Turn],
    chunk_size: int,
    stride: int,
) -> list[tuple[str, list[int]]]:
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
    Score each turn using chunk + individual blend with scaled landmark boost.
    Applies airport floor only when the query is airport-related.
    Mutates turns in-place.
    """
    chunks = _build_chunks(history, config.chunk_size, config.chunk_stride)
    if not chunks:
        return

    airport_query = _is_airport_query(query)

    chunk_texts   = [text for text, _ in chunks]
    chunk_indices = [indices for _, indices in chunks]
    turn_texts    = [t.text for t in history]
    turn_indices  = [t.turn_index for t in history]

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

    turn_max_chunk: dict[int, float] = {}
    for i, indices in enumerate(chunk_indices):
        for turn_idx in indices:
            turn_max_chunk[turn_idx] = max(
                turn_max_chunk.get(turn_idx, 0.0), chunk_scores[i]
            )

    for i, turn in enumerate(history):
        chunk_s = turn_max_chunk.get(turn.turn_index, 0.0)
        indiv_s = indiv_scores[i]
        blended = _CHUNK_WEIGHT * chunk_s + _INDIVIDUAL_WEIGHT * indiv_s
        boost   = config.landmark_boost * indiv_s if turn.is_landmark else 0.0
        score   = min(1.0, blended + boost)

        # Only apply airport floor when query is asking about airports
        if airport_query and _has_airport_signal(turn.text):
            score = max(score, _AIRPORT_SCORE_FLOOR)

        turn.score = score


def classify_turns_chunk_topk(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
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
    """v5: chunk + individual blend → scaled landmark boost → query-gated airport floor → top-K."""
    _score_turns_by_chunks(history, query, query_position, config)
    classify_turns_chunk_topk(history, query_type, config)
    return group_into_runs(history)
