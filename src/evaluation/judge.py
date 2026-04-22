"""
LLM-as-Judge — evaluates answer quality on 4 dimensions (1-10 each):
  - Correctness
  - Completeness
  - Landmark consistency
  - Hallucination (10 = no hallucination)

Uses response_format=json_object to guarantee parseable output.
Each answer is evaluated independently against its own context.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


@dataclass
class JudgeScores:
    correctness:          float
    completeness:         float
    landmark_consistency: float
    hallucination:        float

    @property
    def mean(self) -> float:
        return (
            self.correctness
            + self.completeness
            + self.landmark_consistency
            + self.hallucination
        ) / 4.0


_RUBRIC_PROMPT = """You are evaluating an AI assistant's answer to a query about a booking conversation.

Score the answer on each dimension from 1 to 10:

1. correctness: Are factual claims in the answer accurate relative to the conversation context?
2. completeness: Does the answer address all parts of the query?
3. landmark_consistency: Does the answer respect stated intents, decisions, and action items from the conversation?
4. hallucination: Does the answer avoid introducing information not grounded in the provided context? (10 = no hallucination at all)

Query: {query}

Conversation context provided to the assistant:
---
{context_summary}
---

Answer being evaluated:
---
{answer}
---

Return a JSON object with these exact keys and integer values 1-10:
{{"correctness": N, "completeness": N, "landmark_consistency": N, "hallucination": N}}"""


def _judge_once(
    query: str,
    answer: str,
    context_summary: str,
    model: str,
) -> JudgeScores:
    """Single judge call using JSON mode for reliable parsing."""
    prompt = _RUBRIC_PROMPT.format(
        query=query,
        context_summary=context_summary[:2000],
        answer=answer,
    )
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        return JudgeScores(
            correctness=float(data["correctness"]),
            completeness=float(data["completeness"]),
            landmark_consistency=float(data["landmark_consistency"]),
            hallucination=float(data["hallucination"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Judge parse error: %s | raw: %s", e, raw)
        return JudgeScores(5.0, 5.0, 5.0, 5.0)  # neutral fallback


def _context_summary(thread: list[dict]) -> str:
    """Render a thread as a readable string for the judge prompt."""
    lines = []
    for msg in thread:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def judge_pair(
    query: str,
    answer_full: str,
    answer_opt: str,
    full_thread: list[dict],
    opt_thread: list[dict],
    model: str = "gpt-4o",
) -> tuple[JudgeScores, JudgeScores]:
    """
    Judge both answers independently against their own context.
    Each answer is evaluated once — no side-by-side comparison,
    so positional bias does not apply.

    Returns:
        (scores_full, scores_opt) — JudgeScores for each answer.
    """
    full_ctx = _context_summary(full_thread)
    opt_ctx  = _context_summary(opt_thread)

    scores_full = _judge_once(query, answer_full, full_ctx, model)
    scores_opt  = _judge_once(query, answer_opt,  opt_ctx,  model)

    return scores_full, scores_opt
