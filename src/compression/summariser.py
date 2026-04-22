"""
Stage 4b — LLM Summarisation.

Summarises a COMPRESS run (contiguous low-relevance turns) into
a single concise summary string. One LLM call per run.
"""

from __future__ import annotations

import os
from openai import OpenAI

from src.ingestion.models import Turn

_client: OpenAI | None = None


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


def summarise_run(
    turns: list[Turn],
    domain: str = "flights",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Summarise a contiguous run of low-relevance turns into 1-2 sentences.

    The summary preserves any constraints, options, or prices mentioned
    in the run. It does not introduce new information.

    Args:
        turns:  The turns in this COMPRESS run (non-empty).
        domain: Conversation domain for context in the prompt.
        model:  LLM model name (default gpt-4o-mini for cost efficiency).

    Returns:
        A 1-2 sentence summary string.
    """
    if not turns:
        return ""

    # If run is a single very short filler turn, skip LLM call
    if len(turns) == 1 and len(turns[0].text.strip()) < 20:
        return turns[0].text.strip()

    formatted = _format_turns(turns)
    prompt = f"""The following turns are from a {domain} booking conversation.
Summarise them in 1-2 sentences. Preserve any constraints, options, prices, 
or factual details mentioned. Do not invent anything not in the turns.

Turns:
{formatted}

Summary:"""

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()
