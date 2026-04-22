"""
Context Optimizer — CLI entry point.

Usage:
    python main.py stats
    python main.py inspect --conv-id dlg-xxx --query "..." --query-pos 50
    python main.py inspect --conv-id dlg-xxx --query "..." --query-pos 50 --dry-run
    python main.py inspect --conv-id dlg-xxx --query "..." --query-pos 50 --compare
    python main.py evaluate
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

from dotenv import load_dotenv
load_dotenv()

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login as hf_login
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.ingestion.loader import load_from_config
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query
from src.compression.compressor import classify_turns, group_into_runs
from src.compression.pipeline import compress, full_context
from src.evaluation.harness import EvalQuery, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DISPLAY_WIDTH = 100  # characters per line for content display


def _print_thread(thread: list[dict], title: str) -> None:
    print(f"\n{'='*DISPLAY_WIDTH}")
    print(f"  {title}  ({len(thread)} turns)")
    print(f"{'='*DISPLAY_WIDTH}")
    for msg in thread:
        role    = msg["role"].upper()
        content = msg["content"]
        label   = f"[{role}]"
        # Wrap long content across multiple lines
        indent  = " " * (len(label) + 1)
        lines   = []
        while content:
            lines.append(content[:DISPLAY_WIDTH - len(label) - 1])
            content = content[DISPLAY_WIDTH - len(label) - 1:]
            label   = indent.rstrip()
        print(f"[{msg['role'].upper()}] {lines[0]}")
        for line in lines[1:]:
            print(f"{indent}{line}")


def cmd_stats(config: OptimizerConfig) -> None:
    convs = load_from_config(config)
    turn_counts = [len(c.turns) for c in convs]
    print(f"Corpus: {config.data_path}")
    print(f"Conversations (≥{config.min_turns} turns): {len(convs)}")
    if turn_counts:
        print(f"Turn counts — min: {min(turn_counts)} | max: {max(turn_counts)} | mean: {sum(turn_counts)/len(turn_counts):.1f}")


def cmd_inspect(
    config: OptimizerConfig,
    conv_id: str,
    query: str,
    query_pos: int,
    dry_run: bool = False,
    compare: bool = False,
) -> None:
    from src.scoring.scorer import score_turns
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

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

    detector = get_detector(config)
    detector.detect(conv)

    full_thread = full_context(conv, query_pos)
    full_tokens = sum(len(enc.encode(m["content"])) for m in full_thread)
    print(f"\nFull context: {len(full_thread)} turns | {full_tokens} tokens")

    if dry_run:
        print("\n--- DRY RUN (no LLM calls) ---")
        history    = conv.turns[:query_pos]
        query_type = classify_query(query)
        scored        = score_turns(history, query, query_pos, config)
        classified    = classify_turns(scored, query_type, config)
        runs          = group_into_runs(classified)
        keep           = sum(1 for t in classified if t.disposition in ("KEEP", "CANDIDATE"))
        compress_count = sum(1 for t in classified if t.disposition == "COMPRESS")
        landmarks      = sum(1 for t in classified if t.is_landmark)
        compress_runs  = sum(1 for d, _ in runs if d == "COMPRESS")
        print(f"Landmarks detected: {landmarks}")
        print(f"Turns to KEEP:      {keep}")
        print(f"Turns to COMPRESS:  {compress_count} ({compress_runs} runs → {compress_runs} LLM calls)")
        print(f"Est. token reduction: ~{100*compress_count/len(history):.0f}%")
        print("\n--- PER-TURN DISPOSITIONS ---")
        for turn in classified:
            lm = f" [{turn.landmark_type.upper()}]" if turn.is_landmark else ""
            print(f"  [{turn.disposition:>8}] [{turn.speaker[:4]}] (score={turn.score:.2f}){lm} {turn.text[:80]}")
        return

    opt_thread, stats, latency = compress(conv, query, query_pos, config)
    opt_tokens = sum(len(enc.encode(m["content"])) for m in opt_thread)
    reduction  = (full_tokens - opt_tokens) / full_tokens * 100

    print(f"Optimised:     {len(opt_thread)} turns | {opt_tokens} tokens")
    print(f"Reduction:     {reduction:.1f}% tokens")
    print(f"Kept verbatim: {stats.kept_verbatim} | Summaries: {stats.summary_turns} | Repairs: {stats.integrity_repairs}")
    print(f"Latency:       {latency:.0f}ms")

    if compare:
        _print_thread(full_thread, "ORIGINAL (full context)")
        _print_thread(opt_thread,  "OPTIMISED")
    else:
        _print_thread(opt_thread, "OPTIMISED CONTEXT")


def cmd_evaluate(config: OptimizerConfig) -> None:
    import random
    convs = load_from_config(config)
    if len(convs) < 10:
        logger.warning("Fewer than 10 conversations — evaluation may not be representative.")
    random.seed(42)
    eval_convs = random.sample(convs, min(10, len(convs)))
    eval_queries: dict[str, list[EvalQuery]] = {}
    for conv in eval_convs:
        n     = len(conv.turns)
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
    out_path   = "eval_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Optimizer")
    sub    = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="Show corpus statistics")

    p_inspect = sub.add_parser("inspect", help="Inspect a single conversation")
    p_inspect.add_argument("--conv-id",   required=True)
    p_inspect.add_argument("--query",     required=True)
    p_inspect.add_argument("--query-pos", required=True, type=int)
    p_inspect.add_argument("--dry-run",   action="store_true",
                           help="Show scoring/dispositions without LLM calls")
    p_inspect.add_argument("--compare",   action="store_true",
                           help="Show original and optimised side by side")

    sub.add_parser("evaluate", help="Run full evaluation pipeline")

    parser.add_argument("--data-path",  default="data/taskmaster2/flights.json")
    parser.add_argument("--min-turns",  default=20, type=int)
    parser.add_argument("--detector",   default="rules", choices=["rules", "embedding", "llm"])
    parser.add_argument("--summariser", default="gpt-4o-mini")
    parser.add_argument("--judge",      default="gpt-4o")

    args   = parser.parse_args()
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
        cmd_inspect(
            config, args.conv_id, args.query, args.query_pos,
            dry_run=getattr(args, "dry_run", False),
            compare=getattr(args, "compare", False),
        )
    elif args.command == "evaluate":
        cmd_evaluate(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
