"""
Stage 5 — Evaluation Harness.

Runs the full optimizer pipeline against the full-context baseline
across a set of (conversation, query) pairs and collects metrics.

Metrics reported per query:
  - token_reduction_pct  : % fewer tokens in optimised context
  - quality_full         : LLM-as-judge mean score for full context
  - quality_opt          : LLM-as-judge mean score for optimised context
  - delta_quality        : quality_opt - quality_full
  - bertscore_f1         : semantic preservation between answers
  - landmark_recall      : fraction of GT landmark turns detected
  - compression_pct      : % of turns compressed (not kept verbatim)
  - integrity_repairs    : structural repairs made during assembly
  - latency_ms           : wall-clock compression time
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import pandas as pd
import tiktoken
from openai import OpenAI

from src.compression.assembler import format_full_context
from src.compression.pipeline import compress, full_context
from src.evaluation.bertscore_metric import compute_bertscore
from src.evaluation.judge import JudgeScores, judge_pair
from src.evaluation.landmark_recall import landmark_recall, landmark_stats
from src.ingestion.models import Conversation, OptimizerConfig
from src.landmarks.detector import get_detector

logger = logging.getLogger(__name__)

_llm_client: OpenAI | None = None
_tokeniser = tiktoken.get_encoding("cl100k_base")


def _get_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _llm_client


def _token_count(thread: list[dict]) -> int:
    total = 0
    for msg in thread:
        total += len(_tokeniser.encode(msg.get("content", "")))
    return total


def _generate_answer(
    thread: list[dict],
    query: str,
    model: str,
) -> str:
    """Generate an answer using the provided context thread."""
    messages = thread + [{"role": "user", "content": query}]
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return ""


@dataclass
class EvalQuery:
    """
    A single evaluation query for a conversation.

    query_position: the index of this query in the conversation turn list.
                    Only turns 0..query_position-1 are used as context.
    query_text:     the query string.
    query_type:     "factual" | "analytical" | "preference" (for reporting).
    """
    query_position: int
    query_text:     str
    query_type:     str = "factual"


@dataclass
class EvalResult:
    conversation_id:    str
    query_text:         str
    query_type:         str
    turns_in_history:   int
    full_tokens:        int
    opt_tokens:         int
    token_reduction_pct: float
    quality_full:       float
    quality_opt:        float
    delta_quality:      float
    bertscore_f1:       float | None
    landmark_recall:    float
    compression_pct:    float
    integrity_repairs:  int
    latency_ms:         float


def evaluate(
    conversations: list[Conversation],
    eval_queries:  dict[str, list[EvalQuery]],
    config:        OptimizerConfig,
) -> pd.DataFrame:
    """
    Run evaluation across all (conversation, query) pairs.

    Args:
        conversations: List of Conversation objects.
        eval_queries:  Dict mapping conversation_id → list of EvalQuery.
        config:        OptimizerConfig.

    Returns:
        DataFrame with one row per (conversation, query) pair.
    """
    # Pre-run landmark detection on all conversations
    detector = get_detector(config)
    for conv in conversations:
        detector.detect(conv)
        logger.info("Landmark detection: %s (%d turns)", conv.conversation_id, len(conv.turns))

    results: list[EvalResult] = []

    for conv in conversations:
        queries = eval_queries.get(conv.conversation_id, [])
        if not queries:
            logger.warning("No eval queries for %s — skipping", conv.conversation_id)
            continue

        for eq in queries:
            logger.info(
                "Evaluating %s | query_pos=%d | %s",
                conv.conversation_id, eq.query_position, eq.query_text[:60]
            )

            try:
                result = _evaluate_one(conv, eq, config)
                results.append(result)
                _log_result(result)
            except Exception as e:
                logger.error(
                    "Evaluation failed for %s / pos=%d: %s",
                    conv.conversation_id, eq.query_position, e
                )

    if not results:
        logger.warning("No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame([vars(r) for r in results])

    # Summary row
    _print_summary(df)

    return df


def _evaluate_one(
    conv: Conversation,
    eq:   EvalQuery,
    config: OptimizerConfig,
) -> EvalResult:

    # Full context baseline
    full_thread = full_context(conv, eq.query_position)
    full_tokens = _token_count(full_thread)

    # Optimised context
    opt_thread, assembly_stats, latency_ms = compress(
        conversation=conv,
        query=eq.query_text,
        query_position=eq.query_position,
        config=config,
    )
    opt_tokens = _token_count(opt_thread)

    token_reduction = (
        (full_tokens - opt_tokens) / full_tokens * 100
        if full_tokens > 0 else 0.0
    )

    # Generate answers
    answer_full = _generate_answer(full_thread, eq.query_text, config.judge_model)
    answer_opt  = _generate_answer(opt_thread,  eq.query_text, config.judge_model)

    # Judge both answers
    scores_full, scores_opt = judge_pair(
        query=eq.query_text,
        answer_full=answer_full,
        answer_opt=answer_opt,
        full_thread=full_thread,
        opt_thread=opt_thread,
        model=config.judge_model,
    )

    # BERTScore
    bertscore = compute_bertscore(answer_full, answer_opt)

    # Landmark recall over history
    history = conv.turns[:eq.query_position]
    lm_recall = landmark_recall(history)

    # Compression %
    total = len(history)
    compressed = sum(1 for t in history if t.disposition == "COMPRESS")
    compression_pct = (compressed / total * 100) if total > 0 else 0.0

    return EvalResult(
        conversation_id=conv.conversation_id,
        query_text=eq.query_text,
        query_type=eq.query_type,
        turns_in_history=total,
        full_tokens=full_tokens,
        opt_tokens=opt_tokens,
        token_reduction_pct=round(token_reduction, 1),
        quality_full=round(scores_full.mean, 2),
        quality_opt=round(scores_opt.mean, 2),
        delta_quality=round(scores_opt.mean - scores_full.mean, 2),
        bertscore_f1=round(bertscore, 3) if bertscore is not None else None,
        landmark_recall=round(lm_recall, 3),
        compression_pct=round(compression_pct, 1),
        integrity_repairs=assembly_stats.integrity_repairs,
        latency_ms=round(latency_ms, 1),
    )


def _log_result(r: EvalResult) -> None:
    logger.info(
        "  ✓ %s | reduction=%.0f%% | Δquality=%+.2f | BERTScore=%s | latency=%.0fms",
        r.conversation_id,
        r.token_reduction_pct,
        r.delta_quality,
        f"{r.bertscore_f1:.3f}" if r.bertscore_f1 else "n/a",
        r.latency_ms,
    )


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Conversations: {df['conversation_id'].nunique()}")
    print(f"Queries:       {len(df)}")
    print(f"\nToken reduction:   {df['token_reduction_pct'].mean():.1f}% (mean)")
    print(f"Quality (full):    {df['quality_full'].mean():.2f} (mean)")
    print(f"Quality (opt):     {df['quality_opt'].mean():.2f} (mean)")
    print(f"Δ Quality:         {df['delta_quality'].mean():+.2f} (mean)")

    if df["bertscore_f1"].notna().any():
        print(f"BERTScore F1:      {df['bertscore_f1'].mean():.3f} (mean)")
        pct_passing = (df["bertscore_f1"] >= 0.85).mean() * 100
        print(f"  ≥ 0.85 (passing): {pct_passing:.0f}%")

    print(f"Landmark recall:   {df['landmark_recall'].mean():.1%} (mean)")
    print(f"Compression:       {df['compression_pct'].mean():.1f}% turns (mean)")
    print(f"Latency:           {df['latency_ms'].mean():.0f}ms (mean)")

    # Acceptance bars
    print("\nACCEPTANCE:")
    reduction_ok = (df["token_reduction_pct"].mean() >= 40) and (df["token_reduction_pct"].mean() <= 60)
    quality_ok   = df["delta_quality"].mean() >= 0
    bertscore_ok = df["bertscore_f1"].mean() >= 0.85 if df["bertscore_f1"].notna().any() else None

    print(f"  Token reduction 40-60%: {'✓ PASS' if reduction_ok else '✗ FAIL'}")
    print(f"  Opt quality ≥ full:     {'✓ PASS' if quality_ok else '✗ FAIL'}")
    if bertscore_ok is not None:
        print(f"  BERTScore ≥ 0.85:       {'✓ PASS' if bertscore_ok else '✗ FAIL'}")

    print("=" * 70)
