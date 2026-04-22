"""
Stage 4a — Turn Classification & Run Grouping.
"""

from __future__ import annotations

from src.ingestion.models import OptimizerConfig, Turn

Run = tuple[str, list[Turn]]


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
    """
    high = config.thresholds[query_type]["high"]
    low  = config.thresholds[query_type]["low"]

    for turn in history:
        if turn.is_landmark:
            turn.disposition = "KEEP"
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
    CANDIDATE turns are treated as KEEP for grouping.

    Adjacent COMPRESS runs are merged together so that larger sub-threads
    are summarised in one LLM call rather than many small calls.
    Single-turn COMPRESS runs are merged into adjacent COMPRESS runs where
    possible, to avoid trivial LLM calls for single filler turns like "Okay."
    """
    if not turns:
        return []

    # Initial grouping
    runs: list[Run] = []
    for turn in turns:
        effective = "KEEP" if turn.disposition in ("KEEP", "CANDIDATE") else "COMPRESS"
        if runs and runs[-1][0] == effective:
            runs[-1][1].append(turn)
        else:
            runs.append((effective, list([turn])))

    # Merge single-turn COMPRESS runs into adjacent COMPRESS runs
    runs = _merge_singleton_compress_runs(runs)

    return runs


def _merge_singleton_compress_runs(runs: list[Run]) -> list[Run]:
    """
    Merge single-turn COMPRESS runs with the nearest adjacent COMPRESS run.
    If no adjacent COMPRESS run exists, keep it as-is.

    This reduces the number of LLM summarisation calls without dropping
    any content — single filler turns get absorbed into larger summaries.
    """
    if len(runs) <= 1:
        return runs

    merged: list[Run] = []
    i = 0
    while i < len(runs):
        disposition, turns = runs[i]

        if disposition == "COMPRESS" and len(turns) == 1:
            # Try to merge with previous COMPRESS run
            if merged and merged[-1][0] == "COMPRESS":
                merged[-1][1].extend(turns)
                i += 1
                continue
            # Try to merge with next COMPRESS run
            if i + 1 < len(runs) and runs[i + 1][0] == "COMPRESS":
                # Prepend to next run — handled on next iteration
                runs[i + 1] = ("COMPRESS", turns + runs[i + 1][1])
                i += 1
                continue

        merged.append((disposition, turns))
        i += 1

    return merged
