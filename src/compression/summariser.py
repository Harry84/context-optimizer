"""
Stage 4b — LLM Summarisation.

Summarises a COMPRESS run into a concise summary string.
Short or trivial runs are dropped rather than summarised.
"""

from __future__ import annotations

import os
from openai import OpenAI

from src.ingestion.models import Turn

_client: OpenAI | None = None

# Minimum total characters in a COMPRESS run before we bother summarising.
# Runs shorter than this are just dropped — they're pure filler.
MIN_CHARS_TO_SUMMARISE = 80


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
    Summarise a COMPRESS run into 1-2 sentences.

    Returns empty string if the run is too short to be worth summarising
    (caller will drop it entirely from the output).

    Only summarise if the run contains enough content to justify an LLM call —
    trivial filler like "Okay." / "Sure." / "Hold on." is just dropped.
    """
    if not turns:
        return ""

    # Drop trivial runs — not worth an LLM call or a summary placeholder
    if _total_chars(turns) < MIN_CHARS_TO_SUMMARISE:
        return ""

    formatted = _format_turns(turns)
    prompt = f"""The following turns are from a {domain} booking conversation.
Summarise in 1-2 concise sentences. Only include information that would be
useful context for answering a question about this conversation later.
Omit greetings, filler, and repeated content. If there is nothing meaningful
to preserve, respond with exactly: SKIP

Turns:
{formatted}

Summary:"""

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80,
    )
    result = response.choices[0].message.content.strip()

    # If model says nothing meaningful, drop it
    if result.upper() == "SKIP" or not result:
        return ""

    return result
