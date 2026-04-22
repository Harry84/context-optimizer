"""
BERTScore F1 metric.

Computes semantic similarity between the full-context answer and the
optimised-context answer. This is an LLM-independent, deterministic
measure of semantic content preservation.

Threshold: F1 >= 0.85 = acceptable semantic preservation.
Uses roberta-large as the reference model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_bertscore_available = True
try:
    from bert_score import score as _bert_score
except ImportError:
    _bertscore_available = False
    logger.warning(
        "bert-score not installed. BERTScore metric will return None. "
        "Install with: pip install bert-score"
    )


def compute_bertscore(
    answer_full: str,
    answer_opt: str,
    model_type: str = "roberta-large",
) -> float | None:
    """
    Compute BERTScore F1 between the full-context and optimised-context answers.

    Args:
        answer_full: Answer generated using full conversation history.
        answer_opt:  Answer generated using optimised context.
        model_type:  Reference model for BERTScore (default roberta-large).

    Returns:
        F1 score in [0, 1], or None if bert-score is not installed.
        F1 >= 0.85 indicates acceptable semantic preservation.
    """
    if not _bertscore_available:
        return None

    if not answer_full.strip() or not answer_opt.strip():
        return None

    try:
        P, R, F1 = _bert_score(
            cands=[answer_opt],
            refs=[answer_full],
            model_type=model_type,
            lang="en",
            verbose=False,
        )
        return float(F1[0])
    except Exception as e:
        logger.error("BERTScore computation failed: %s", e)
        return None
