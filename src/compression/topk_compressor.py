"""
v3 — Top-K Retrieval Compressor.

Instead of threshold-based classification, ranks all non-landmark turns
by their composite relevance score and keeps only the top K.

Rationale: for a query like "what was the price of the flight?", only
2-3 turns are genuinely relevant. Threshold-based approaches keep too many
turns because many score above the threshold. Top-K retrieval keeps exactly
the turns that matter — regardless of how many turns are in the conversation.

Pipeline:
  1. Landmarks → always KEEP (unchanged from v1/v2)
  2. Non-landmark turns below topk_min_score → always COMPRESS (noise floor)
  3. Remaining non-landmarks ranked by score descending, top K kept
  4. Rest → COMPRESS
  5. Grouping and assembly unchanged (same assembler)

K is determined by query type via config.topk_k:
  factual    — fewer turns (query has a specific answer)
  analytical — more turns (query requires reasoning across history)
  preference — intermediate

topk_min_score is a noise floor: turns scoring below it are always
compressed regardless of K, preventing low-quality crowdworker noise
from making it into the kept set just because K slots remain unfilled.
"""

from __future__ import annotations

from src.compression.compressor import Run, group_into_runs
from src.ingestion.models import OptimizerConfig, Turn


def classify_turns_topk(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
    """
    Classify turns using top-K retrieval with a noise floor.

    Landmarks: always KEEP regardless of score.
    Non-landmarks below topk_min_score: always COMPRESS (noise floor).
    Remaining non-landmarks: top K by score → KEEP, rest → COMPRESS.
    """
    k         = config.topk_k[query_type]
    min_score = config.topk_min_score

    landmarks     = [t for t in history if t.is_landmark]
    non_landmarks = [t for t in history if not t.is_landmark]

    # Always KEEP landmarks
    for t in landmarks:
        t.disposition = "KEEP"

    # Split non-landmarks into candidates (above floor) and noise (below floor)
    candidates = [t for t in non_landmarks if t.score >= min_score]
    noise      = [t for t in non_landmarks if t.score <  min_score]

    # Always COMPRESS noise
    for t in noise:
        t.disposition = "COMPRESS"

    # From candidates, keep top K by score
    ranked      = sorted(candidates, key=lambda t: t.score, reverse=True)
    keep_ids    = {id(t) for t in ranked[:k]}

    for t in candidates:
        t.disposition = "KEEP" if id(t) in keep_ids else "COMPRESS"

    return history


def topk_runs(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Run]:
    """
    Full top-K classification + run grouping.
    Returns list[Run] compatible with the assembler.
    """
    classified = classify_turns_topk(history, query_type, config)
    return group_into_runs(classified)
