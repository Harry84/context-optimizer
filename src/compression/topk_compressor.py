"""
v3 — Top-K Retrieval Compressor.

Instead of threshold-based classification, ranks all non-landmark turns
by their composite relevance score and keeps only the top K.

Rationale: for a query like "what was the price of the flight?", only
a small fraction of turns are genuinely relevant. Threshold-based approaches
keep too many turns. Top-K retrieval keeps exactly the turns that matter.

K is proportional to the number of non-landmark turns (topk_fraction),
not a fixed number — this ensures consistent token reduction regardless
of conversation length or landmark density.

Pipeline:
  1. Landmarks → always KEEP
  2. Non-landmarks below topk_min_score → always COMPRESS (noise floor)
  3. Remaining non-landmarks ranked by score descending
  4. Top K = ceil(topk_fraction × len(non_landmarks)) kept
  5. Rest → COMPRESS
"""

from __future__ import annotations
import math

from src.compression.compressor import Run, group_into_runs
from src.ingestion.models import OptimizerConfig, Turn


def classify_turns_topk(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
    """
    Classify turns using proportional top-K retrieval with a noise floor.

    K = ceil(topk_fraction[query_type] × number of non-landmark turns).
    This keeps token reduction consistent regardless of conversation length.
    """
    fraction  = config.topk_fraction[query_type]
    min_score = config.topk_min_score

    landmarks     = [t for t in history if t.is_landmark]
    non_landmarks = [t for t in history if not t.is_landmark]

    # Always KEEP landmarks
    for t in landmarks:
        t.disposition = "KEEP"

    # Split non-landmarks: candidates (above floor) vs noise (below floor)
    candidates = [t for t in non_landmarks if t.score >= min_score]
    noise      = [t for t in non_landmarks if t.score <  min_score]

    # Always COMPRESS noise
    for t in noise:
        t.disposition = "COMPRESS"

    # K is proportional to total non-landmark count (not just candidates)
    # so that fraction is meaningful relative to conversation size
    k = max(1, math.ceil(fraction * len(non_landmarks)))

    ranked   = sorted(candidates, key=lambda t: t.score, reverse=True)
    keep_ids = {id(t) for t in ranked[:k]}

    for t in candidates:
        t.disposition = "KEEP" if id(t) in keep_ids else "COMPRESS"

    return history


def topk_runs(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Run]:
    """Full top-K classification + run grouping."""
    classified = classify_turns_topk(history, query_type, config)
    return group_into_runs(classified)
