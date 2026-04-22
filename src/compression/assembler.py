"""
Stage 4c — Thread Assembly & Integrity Check.

Assembles the final [{role, content}] thread from KEEP runs
and SUMMARY placeholders for COMPRESS runs.

Enforces structural integrity:
  - No consecutive same-role turns (merged)
  - Thread must start with a user turn
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

    if repairs > 0:
        logger.debug("Assembly integrity: %d repair(s) made", repairs)

    return thread, stats


def _integrity_check(thread: list[dict]) -> tuple[list[dict], int]:
    """
    Validate and repair structural issues in the assembled thread.

    Repairs:
    1. Consecutive same-role turns → merge content with a space
    2. Thread starts with an assistant turn → prepend placeholder user turn
    """
    if not thread:
        return thread, 0

    repairs = 0

    # Repair 1: merge consecutive same-role turns
    merged: list[dict] = []
    for msg in thread:
        if merged and merged[-1]["role"] == msg["role"]:
            # Only merge if content is meaningfully different
            if msg["content"].strip() and msg["content"].strip() != merged[-1]["content"].strip():
                merged[-1]["content"] += " " + msg["content"]
            repairs += 1
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    # Repair 2: thread must not start with an assistant turn
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "[conversation start]"})
        repairs += 1

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
    thread, _ = _integrity_check(thread)
    return thread
