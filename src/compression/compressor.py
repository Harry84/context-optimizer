"""
Stage 4a — Turn Classification & Run Grouping.

After scoring, classifies each turn's disposition and groups
contiguous same-disposition turns into runs for summarisation.
"""

from __future__ import annotations

from src.ingestion.models import OptimizerConfig, Turn


# Type alias for clarity
Run = tuple[str, list[Turn]]   # (disposition, [turns])


def classify_turns(
    history: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
    """
    Assign a disposition to each turn based on landmark status and score.

    Dispositions:
        KEEP      — landmark turns (hard rule) or high relevance score
        CANDIDATE — moderate relevance; kept for structural integrity
        COMPRESS  — low relevance; will be summarised

    Args:
        history:    Scored history turns (0..Q-1).
        query_type: "factual" | "analytical" | "preference"
        config:     Contains per-query-type thresholds.

    Returns:
        Same list of turns with .disposition set.
    """
    high = config.thresholds[query_type]["high"]
    low  = config.thresholds[query_type]["low"]

    for turn in history:
        if turn.is_landmark:
            turn.disposition = "KEEP"           # hard rule — always verbatim
        elif turn.score >= high:
            turn.disposition = "KEEP"
        elif turn.score >= low:
            turn.disposition = "CANDIDATE"
        else:
            turn.disposition = "COMPRESS"

    return history


def group_into_runs(turns: list[Turn]) -> list[Run]:
    """
    Group consecutive turns by disposition into contiguous runs.

    Each run is a tuple of (disposition, [turns]).
    Runs are in strict chronological order.

    CANDIDATE turns are treated as KEEP for grouping — they are
    included verbatim but their run may be merged with adjacent KEEPs.

    Example:
        turns: KEEP KEEP COMPRESS COMPRESS KEEP COMPRESS COMPRESS
        runs:  [("KEEP",     [t0, t1]),
                ("COMPRESS", [t2, t3]),
                ("KEEP",     [t4]),
                ("COMPRESS", [t5, t6])]

    Each COMPRESS run becomes one LLM summarisation call.
    """
    if not turns:
        return []

    runs: list[Run] = []
    for turn in turns:
        # Treat CANDIDATE as KEEP for grouping purposes
        effective = "KEEP" if turn.disposition in ("KEEP", "CANDIDATE") else "COMPRESS"
        if runs and runs[-1][0] == effective:
            runs[-1][1].append(turn)
        else:
            runs.append((effective, [turn]))

    return runs
