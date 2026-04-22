"""
Dry-run compression audit on synthetic conversations.
No API calls. Shows estimated token reduction before summarisation.

Usage:
    python utilities/synthetic_audit.py
    python utilities/synthetic_audit.py data/synthetic/synthetic_flights.json
    python utilities/synthetic_audit.py data/synthetic/synthetic_flights.json --min-turns 5
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

from src.ingestion.loader import load_corpus
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query
from src.scoring.scorer import score_turns
from src.compression.compressor import classify_turns

parser = argparse.ArgumentParser(description="Dry-run compression audit")
parser.add_argument("data_path", nargs="?", default="data/synthetic/synthetic_flights.json")
parser.add_argument("--min-turns", type=int, default=5)
parser.add_argument("--query", default="What flights were compared and what did the user decide?")
args = parser.parse_args()

QUERY      = args.query
QUERY_TYPE = classify_query(QUERY)

config   = OptimizerConfig(min_turns=args.min_turns, data_path=args.data_path)
convs    = load_corpus(args.data_path, min_turns=args.min_turns)
detector = get_detector(config)

print(f"Data: {args.data_path}  |  Conversations: {len(convs)}  |  Min turns: {args.min_turns}")
print(f"Query: {QUERY}\n")
print(f"{'Conv ID':<35} {'Trns':>4} {'qpos':>4} {'KEEP':>4} {'COMP':>4} {'Full':>6} {'Kept':>6} {'Comp':>6} {'MaxRed':>7}")
print("-" * 90)

totals = {"full": 0, "keep": 0, "comp": 0, "turns": 0}

for conv in convs:
    detector.detect(conv)
    n     = len(conv.turns)
    q_pos = max(5, int(n * 0.75))
    history = conv.turns[:q_pos]

    scored     = score_turns(history, QUERY, q_pos, config)
    classified = classify_turns(scored, QUERY_TYPE, config)

    keep = [t for t in classified if t.disposition in ("KEEP", "CANDIDATE")]
    comp = [t for t in classified if t.disposition == "COMPRESS"]

    full_tok = sum(len(enc.encode(t.text)) for t in history)
    keep_tok = sum(len(enc.encode(t.text)) for t in keep)
    comp_tok = sum(len(enc.encode(t.text)) for t in comp)
    max_red  = 100 * comp_tok / full_tok if full_tok > 0 else 0

    totals["full"]  += full_tok
    totals["keep"]  += keep_tok
    totals["comp"]  += comp_tok
    totals["turns"] += n

    print(f"{conv.conversation_id:<35} {n:>4} {q_pos:>4} {len(keep):>4} {len(comp):>4} "
          f"{full_tok:>6} {keep_tok:>6} {comp_tok:>6} {max_red:>6.1f}%")

print("-" * 90)
overall_max_red = 100 * totals["comp"] / totals["full"] if totals["full"] else 0
avg_tok_per_turn = totals["full"] / totals["turns"] if totals["turns"] else 0
print(f"\nMax possible reduction (summaries → 0 tokens): {overall_max_red:.1f}%")
print(f"Avg tokens per turn:                            {avg_tok_per_turn:.1f}")
print(f"Estimated net reduction (after ~15-tok summaries): {overall_max_red * 0.65:.0f}–{overall_max_red * 0.80:.0f}%")
