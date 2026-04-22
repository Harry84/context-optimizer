"""
Stage 4c — Thread Assembly & Integrity Check.
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

    KEEP runs    → verbatim turns in chronological order.
    COMPRESS runs → single synthetic assistant [SUMMARY: ...] turn,
                    or nothing if the summariser returned empty (trivial run).
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
                    "role":    "assistant",
                    "content": f"[SUMMARY: {summary_text}]",
                })
                stats.summary_turns += 1
            stats.compressed_runs += 1

    thread, repairs = _integrity_check(thread)
    stats.integrity_repairs = repairs

    return thread, stats


def _is_substantially_contained(new: str, existing: str) -> bool:
    """
    Return True if new content is already substantially present in existing.
    Used to avoid appending near-duplicate assistant turns.
    """
    new_clean      = new.strip().lower()
    existing_clean = existing.strip().lower()
    if not new_clean:
        return True
    # If the new content is a substring of existing, skip it
    if new_clean in existing_clean:
        return True
    # If 80%+ of the words in new already appear in existing, skip it
    new_words      = set(new_clean.split())
    existing_words = set(existing_clean.split())
    if new_words and len(new_words & existing_words) / len(new_words) >= 0.8:
        return True
    return False


def _integrity_check(thread: list[dict]) -> tuple[list[dict], int]:
    """
    Validate and repair structural issues.

    Rules:
    1. Thread must start with a user turn.
    2. Consecutive ASSISTANT turns → merge, skipping near-duplicate content.
    3. Consecutive USER turns → insert a thin assistant bridge.
    """
    if not thread:
        return thread, 0

    repairs = 0
    merged: list[dict] = []

    for msg in thread:
        same_role = merged and merged[-1]["role"] == msg["role"]

        if same_role and msg["role"] == "assistant":
            new_content = msg["content"].strip()
            if new_content and not _is_substantially_contained(new_content, merged[-1]["content"]):
                merged[-1]["content"] += " " + new_content
            repairs += 1

        elif same_role and msg["role"] == "user":
            merged.append({"role": "assistant", "content": "[context continues]"})
            merged.append({"role": msg["role"], "content": msg["content"]})
            repairs += 1

        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    # Thread must not start with an assistant turn
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "[conversation start]"})
        repairs += 1

    return merged, repairs


def format_full_context(turns: list[Turn]) -> list[dict]:
    """Format full turn list as [{role, content}] with no compression."""
    thread = [
        {"role": "user" if t.speaker == "USER" else "assistant", "content": t.text}
        for t in turns
    ]
    thread, _ = _integrity_check(thread)
    return thread
