"""
Context Optimizer — CLI entry point.

Usage:
    # Run full evaluation pipeline
    python main.py evaluate

    # Inspect a single conversation interactively
    python main.py inspect --conv-id dlg-cbfc519d-93e3-404d-9db5-c5fe35a5b765

    # Show corpus stats
    python main.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.compression.pipeline import compress, full_context
from src.evaluation.harness import EvalQuery, evaluate
from src.ingestion.loader import load_from_config
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_stats(config: OptimizerConfig) -> None:
    """Show corpus statistics."""
    convs = load_from_config(config)
    turn_counts = [len(c.turns) for c in convs]
    print(f"Corpus: {config.data_path}")
    print(f"Conversations (≥{config.min_turns} turns): {len(convs)}")
    if turn_counts:
        print(f"Turn counts — min: {min(turn_counts)} | max: {max(turn_counts)} | mean: {sum(turn_counts)/len(turn_counts):.1f}")


def cmd_inspect(config: OptimizerConfig, conv_id: str, query: str, query_pos: int) -> None:
    """
    Run compression on a single conversation and print the result.
    Useful for manual inspection before running full evaluation.
    """
    convs = load_from_config(config)
    conv  = next((c for c in convs if c.conversation_id == conv_id), None)

    if conv is None:
        print(f"Conversation {conv_id!r} not found.")
        sys.exit(1)

    print(f"\nConversation: {conv.conversation_id}")
    print(f"Total turns:  {len(conv.turns)}")
    print(f"Query pos:    {query_pos}")
    print(f"Query:        {query}")
    print(f"Query type:   {classify_query(query)}")

    # Detect landmarks
    detector = get_detector(config)
    detector.detect(conv)

    # Full context
    full_thread = full_context(conv, query_pos)
    print(f"\nFull context: {len(full_thread)} turns")

    # Optimised
    opt_thread, stats, latency = compress(conv, query, query_pos, config)
    print(f"Optimised:    {len(opt_thread)} turns | {latency:.0f}ms")
    print(f"Kept verbatim: {stats.kept_verbatim} | Summaries: {stats.summary_turns} | Repairs: {stats.integrity_repairs}")

    print("\n--- OPTIMISED CONTEXT ---")
    for msg in opt_thread:
        role = msg["role"].upper()
        print(f"[{role}] {msg['content'][:120]}")


def cmd_evaluate(config: OptimizerConfig) -> None:
    """
    Run evaluation on a hardcoded set of queries across sampled conversations.

    In a full implementation, eval_queries would be loaded from a file.
    Here we construct a minimal set for demonstration.
    """
    convs = load_from_config(config)

    if len(convs) < 10:
        logger.warning("Fewer than 10 conversations — evaluation may not be representative.")

    # Sample 10 conversations
    import random
    random.seed(42)
    eval_convs = random.sample(convs, min(10, len(convs)))

    # For each conversation, construct evaluation queries at meaningful positions.
    # Queries are placed at the turn where a booking decision is being made —
    # requiring understanding of earlier constraint-setting turns.
    eval_queries: dict[str, list[EvalQuery]] = {}
    for conv in eval_convs:
        n = len(conv.turns)
        # Place query at 75% through the conversation (post-comparison, pre-close)
        q_pos = max(5, int(n * 0.75))
        eval_queries[conv.conversation_id] = [
            EvalQuery(
                query_position=q_pos,
                query_text="What flights were compared and what did the user decide?",
                query_type="factual",
            ),
            EvalQuery(
                query_position=q_pos,
                query_text="Why did the user choose the flight they selected?",
                query_type="analytical",
            ),
        ]

    results_df = evaluate(eval_convs, eval_queries, config)

    # Save results
    out_path = "eval_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Optimizer")
    sub = parser.add_subparsers(dest="command")

    # Stats command
    sub.add_parser("stats", help="Show corpus statistics")

    # Inspect command
    p_inspect = sub.add_parser("inspect", help="Inspect a single conversation")
    p_inspect.add_argument("--conv-id",   required=True, help="Conversation ID")
    p_inspect.add_argument("--query",     required=True, help="Query string")
    p_inspect.add_argument("--query-pos", required=True, type=int, help="Query position (turn index)")

    # Evaluate command
    sub.add_parser("evaluate", help="Run full evaluation pipeline")

    # Global config overrides
    parser.add_argument("--data-path",  default="data/taskmaster2/flights.json")
    parser.add_argument("--min-turns",  default=20, type=int)
    parser.add_argument("--detector",   default="rules", choices=["rules", "embedding", "llm"])
    parser.add_argument("--summariser", default="gpt-4o-mini")
    parser.add_argument("--judge",      default="gpt-4o")

    args = parser.parse_args()

    config = OptimizerConfig(
        data_path=args.data_path,
        min_turns=args.min_turns,
        landmark_detector=args.detector,
        summarisation_model=args.summariser,
        judge_model=args.judge,
    )

    if args.command == "stats":
        cmd_stats(config)
    elif args.command == "inspect":
        cmd_inspect(config, args.conv_id, args.query, args.query_pos)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
