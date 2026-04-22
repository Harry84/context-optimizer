"""
Stage 4c — Thread Assembly & Integrity Check.

Assembles the final [{role, content}] thread from KEEP runs
and SUMMARY placeholders for COMPRESS runs.

Enforces structural integrity:
  - No consecutive same-role turns
  - No orphaned user turns (USER with no preceding ASSISTANT)
  - Valid alternating role structure expected by LLMs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ingestion.models import Turn

logger = logging.getLogger(__name__)

Run = tuple[str, list[Turn]]


@dataclass
class AssemblyStats:
    total_turns_in:    int = 0
    kept_verbatim:     int = 0
    compressed_runs:   int = 0
    summary_turns:     int = 0
    integrity_repairs: int = 0


def assemble(
    runs: list[Run],
    summaries: dict[int, str],
) -> tuple[list[dict], AssemblyStats]:
    """
    Build the final [{role, content}] thread.

    KEEP runs   → verbatim turns in chronological order.
    COMPRESS runs → single synthetic assistant [SUMMARY: ...] turn.

    Args:
        runs:      List of (disposition, [Turn]) from compressor.group_into_runs().
        summaries: Mapping from id(turns_list) → summary string.

    Returns:
        (thread, stats)
    """
    stats = AssemblyStats()
    thread: list[dict] = []

    for disposition, turns in runs:
        stats.total_turns_in += len(turns)

        if disposition == "KEEP":
            for turn in turns:
                role = "user" if turn.speaker == "USER" else "assistant"
                thread.append({"role": role, "content": turn.text})
                stats.kept_verbatim += 1

        elif disposition == "COMPRESS":
            summary_text = summaries.get(id(turns), "")
            if summary_text:
                thread.append({
                    "role": "assistant",
                    "content": f"[SUMMARY: {summary_text}]",
                })
                stats.summary_turns += 1
            stats.compressed_runs += 1

    thread, repairs = _integrity_check(thread)
    stats.integrity_repairs = repairs

    return thread, stats


def _integrity_check(thread: list[dict]) -> tuple[list[dict], int]:
    """
    Validate and repair structural issues in the assembled thread.

    Repairs applied:
    1. Consecutive same-role turns → merge content with a space
    2. Thread starts with an assistant turn → prepend placeholder user turn

    Note: a thread ending with a user turn is normal — the current query
    is posed after the context is assembled.

    Returns:
        (repaired_thread, repair_count)
    """
    if not thread:
        return thread, 0

    repairs = 0

    # Repair 1: merge consecutive same-role turns
    merged: list[dict] = []
    for msg in thread:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += " " + msg["content"]
            repairs += 1
            logger.debug("Merged consecutive %s turns", msg["role"])
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    # Repair 2: thread must not start with an assistant turn
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "[conversation start]"})
        repairs += 1
        logger.debug("Inserted placeholder user turn at start")

    if repairs > 0:
        logger.info("Assembly integrity: %d repair(s) made", repairs)

    return merged, repairs


def format_full_context(turns: list[Turn]) -> list[dict]:
    """
    Format a full list of turns as a [{role, content}] thread
    with no compression. Used for the full-context baseline in evaluation.
    """
    thread = [
        {"role": "user" if t.speaker == "USER" else "assistant", "content": t.text}
        for t in turns
    ]
    # Apply integrity check — fixes any structural issues in raw data too
    thread, _ = _integrity_check(thread)
    return thread
