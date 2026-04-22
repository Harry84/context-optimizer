"""
Stage 4b — LLM Summarisation.

Summarises a COMPRESS run into a single ultra-concise sentence.
Short or trivial runs are dropped rather than summarised.
"""

from __future__ import annotations

import os
from openai import OpenAI

from src.ingestion.models import Turn

_client: OpenAI | None = None

# Minimum total characters in a COMPRESS run before we bother summarising.
# Runs shorter than this are dropped entirely — summary would be longer than original.
MIN_CHARS_TO_SUMMARISE = 200


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _format_turns(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        label = "User" if turn.speaker == "USER" else "Assistant"
        lines.append(f"{label}: {turn.text}")
    return "\n".join(lines)


def _total_chars(turns: list[Turn]) -> int:
    return sum(len(t.text.strip()) for t in turns)


def summarise_run(
    turns: list[Turn],
    domain: str = "flights",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Summarise a COMPRESS run into one tight phrase (≤15 words).

    Returns empty string if the run is too short or contains nothing
    meaningful — caller drops it entirely from the output.
    """
    if not turns:
        return ""

    if _total_chars(turns) < MIN_CHARS_TO_SUMMARISE:
        return ""

    formatted = _format_turns(turns)
    prompt = f"""Summarise these {domain} booking conversation turns in ONE short phrase of at most 15 words.
Only include facts useful for answering questions later (destinations, dates, prices, decisions).
Omit greetings, filler, confirmations, and repeated content entirely.
If there are no useful facts, reply with exactly: SKIP

Turns:
{formatted}

Summary (≤15 words):"""

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=30,
    )
    result = response.choices[0].message.content.strip()

    if result.upper() == "SKIP" or not result:
        return ""

    return result
