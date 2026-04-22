"""
Full compression pipeline entry point.

compress() is the single function called by the evaluation harness.

Compression strategy selected via config.compression_strategy:
  "turn"          — v1 threshold-based turn-level
  "sentence"      — v2 sentence-level within landmark turns
  "topk"          — v3 proportional top-K turn retrieval
  "topk-sentence" — v4 top-K + sentence splitting of landmark turns
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
    assert query_position > 0
    assert query_position <= len(conversation.turns)

    t_start = time.perf_counter()

    detector = get_detector(config)
    detector.detect(conversation)

    history    = conversation.turns[:query_position]
    query_type = classify_query(query)

    scored_history = score_turns(
        history=history,
        query=query,
        query_position=query_position,
        config=config,
    )

    if config.compression_strategy == "sentence":
        from src.compression.sentence_compressor import classify_turns_sentence_level
        runs: list[Run] = classify_turns_sentence_level(
            history=scored_history,
            query=query,
            query_position=query_position,
            query_type=query_type,
            config=config,
        )
    elif config.compression_strategy == "topk":
        from src.compression.topk_compressor import topk_runs
        runs = topk_runs(scored_history, query_type, config)
    elif config.compression_strategy == "topk-sentence":
        from src.compression.topk_sentence_compressor import topk_sentence_runs
        runs = topk_sentence_runs(
            history=scored_history,
            query=query,
            query_position=query_position,
            query_type=query_type,
            config=config,
        )
    else:
        classified = classify_turns(scored_history, query_type, config)
        runs = group_into_runs(classified)

    summaries: dict[int, str] = {}
    for disposition, run_turns in runs:
        if disposition == "COMPRESS":
            summaries[id(run_turns)] = summarise_run(
                turns=run_turns,
                domain=conversation.domain,
                model=config.summarisation_model,
            )

    thread, stats = assemble(runs, summaries)
    latency_ms = (time.perf_counter() - t_start) * 1000
    return thread, stats, latency_ms


def full_context(
    conversation: Conversation,
    query_position: int,
) -> list[dict]:
    history = conversation.turns[:query_position]
    return format_full_context(history)
