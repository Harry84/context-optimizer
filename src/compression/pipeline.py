"""
Full compression pipeline entry point.

compress() is the single function called by the evaluation harness.
It takes a Conversation, a query string, and the query's position in
the conversation, and returns an optimised [{role, content}] thread.
"""

from __future__ import annotations

import time

from src.compression.assembler import AssemblyStats, assemble, format_full_context
from src.compression.compressor import Run, classify_turns, group_into_runs
from src.compression.summariser import summarise_run
from src.ingestion.models import Conversation, OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query
from src.scoring.scorer import score_turns


def compress(
    conversation: Conversation,
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> tuple[list[dict], AssemblyStats, float]:
    """
    Full compression pipeline for a single (conversation, query) pair.

    Only turns[0..query_position-1] are used as context.
    Turn at query_position is the current query being answered.
    Turns after query_position are ignored (not yet in history).

    Args:
        conversation:   Full Conversation object (landmark-detected on first call).
        query:          The current query string.
        query_position: Index of the query turn in the conversation.
                        Must be > 0.
        config:         OptimizerConfig.

    Returns:
        (thread, stats, latency_ms) where:
          thread      — optimised [{role, content}] list for LLM consumption
          stats       — AssemblyStats with counts (for reporting)
          latency_ms  — wall-clock time for compression in milliseconds
    """
    assert query_position > 0, "query_position must be > 0 (need at least one history turn)"
    assert query_position <= len(conversation.turns), "query_position out of range"

    t_start = time.perf_counter()

    # 1. Ensure landmark detection has been run on the full conversation.
    #    The detector is idempotent — safe to call multiple times.
    detector = get_detector(config)
    detector.detect(conversation)

    # 2. Slice to history: turns 0..query_position-1 only.
    history = conversation.turns[:query_position]

    # 3. Classify query type (drives weights and thresholds).
    query_type = classify_query(query)

    # 4. Score all history turns against the query.
    scored_history = score_turns(
        history=history,
        query=query,
        query_position=query_position,
        config=config,
    )

    # 5. Assign disposition to each turn.
    classified = classify_turns(scored_history, query_type, config)

    # 6. Group into contiguous runs.
    runs: list[Run] = group_into_runs(classified)

    # 7. Summarise each COMPRESS run (one LLM call per run).
    summaries: dict[int, str] = {}
    for disposition, run_turns in runs:
        if disposition == "COMPRESS":
            summaries[id(run_turns)] = summarise_run(
                turns=run_turns,
                domain=conversation.domain,
                model=config.summarisation_model,
            )

    # 8. Assemble final thread.
    thread, stats = assemble(runs, summaries)

    latency_ms = (time.perf_counter() - t_start) * 1000

    return thread, stats, latency_ms


def full_context(
    conversation: Conversation,
    query_position: int,
) -> list[dict]:
    """
    Return the full uncompressed context for baseline comparison.
    Uses turns 0..query_position-1 with no compression.
    """
    history = conversation.turns[:query_position]
    return format_full_context(history)
