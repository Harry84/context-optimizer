"""
Landmark recall metric.

Measures what fraction of slot-annotated turns (ground truth)
were correctly identified as landmarks by the detector.

This is an honest, independent evaluation of the landmark detector
using Taskmaster-2's human annotations as ground truth.

Note: Action items have no slot annotations in Taskmaster-2,
so they are excluded from this calculation. Action item detection
quality is reported separately (pattern-match coverage only).
"""

from __future__ import annotations

from src.ingestion.models import Turn


def landmark_recall(turns: list[Turn]) -> float:
    """
    Compute landmark recall against slot-annotation ground truth.

    Ground truth: turns with at least one slot annotation.
    Detected:     turns where is_landmark is True.

    Args:
        turns: History turns (0..Q-1) with both slots and is_landmark populated.

    Returns:
        Recall in [0, 1]. Returns 1.0 if no GT turns exist (vacuously true).
    """
    gt_indices       = {t.turn_index for t in turns if t.slots}
    detected_indices = {t.turn_index for t in turns if t.is_landmark}

    if not gt_indices:
        return 1.0

    return len(gt_indices & detected_indices) / len(gt_indices)


def landmark_stats(turns: list[Turn]) -> dict:
    """
    Return detailed landmark statistics for a turn list.
    Used in evaluation reporting.
    """
    gt_set       = {t.turn_index for t in turns if t.slots}
    detected_set = {t.turn_index for t in turns if t.is_landmark}
    promoted_set = {t.turn_index for t in turns if t.promoted}

    by_type = {"intent": 0, "decision": 0, "action_item": 0}
    for t in turns:
        if t.is_landmark and t.landmark_type:
            by_type[t.landmark_type] = by_type.get(t.landmark_type, 0) + 1

    return {
        "total_turns":      len(turns),
        "gt_turns":         len(gt_set),
        "detected":         len(detected_set),
        "promoted_pass2":   len(promoted_set),
        "recall":           landmark_recall(turns),
        "by_type":          by_type,
        "compressible":     sum(1 for t in turns if t.disposition == "COMPRESS"),
    }
