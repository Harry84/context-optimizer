"""
Dry-run compression audit — no API calls, no summarisation.

Loads conversations with ≥50 turns, runs landmark detection + scoring +
classification on each, and reports:
  - turns kept verbatim vs compressed
  - estimated token reduction (treats compressed turns as gone — no summary tokens)
  - distribution across the sample

Usage:
    python utilities/compression_audit.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import statistics

from dotenv import load_dotenv
load_dotenv(override=True)

import warnings
warnings.filterwarnings("ignore")

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

from src.ingestion.loader import load_from_config
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query
from src.scoring.scorer import score_turns
from src.compression.compressor import classify_turns, group_into_runs

QUERY      = "What flights were compared and what did the user decide?"
QUERY_TYPE = classify_query(QUERY)
SAMPLE_N   = 30
MIN_TURNS  = 50
SEED       = 42

config = OptimizerConfig(min_turns=MIN_TURNS)
convs  = load_from_config(config)

long_convs = [c for c in convs if len(c.turns) >= MIN_TURNS]
print(f"Corpus: {len(convs)} total, {len(long_convs)} with ≥{MIN_TURNS} turns")

random.seed(SEED)
sample = random.sample(long_convs, min(SAMPLE_N, len(long_convs)))

detector = get_detector(config)

reductions = []
keep_pcts  = []
comp_pcts  = []

print(f"\n{'Conv ID':<45} {'Turns':>5} {'Hist':>5} {'KEEP':>5} {'COMP':>5} {'Full tok':>9} {'Kept tok':>9} {'Est red%':>8}")
print("-" * 105)

for conv in sample:
    detector.detect(conv)

    n     = len(conv.turns)
    q_pos = max(5, int(n * 0.75))
    history = conv.turns[:q_pos]

    scored     = score_turns(history, QUERY, q_pos, config)
    classified = classify_turns(scored, QUERY_TYPE, config)

    keep_turns = [t for t in classified if t.disposition in ("KEEP", "CANDIDATE")]
    comp_turns = [t for t in classified if t.disposition == "COMPRESS"]

    full_tokens = sum(len(enc.encode(t.text)) for t in history)
    kept_tokens = sum(len(enc.encode(t.text)) for t in keep_turns)

    est_reduction = (full_tokens - kept_tokens) / full_tokens * 100 if full_tokens > 0 else 0

    reductions.append(est_reduction)
    keep_pcts.append(100 * len(keep_turns) / len(history))
    comp_pcts.append(100 * len(comp_turns) / len(history))

    print(
        f"{conv.conversation_id:<45} {n:>5} {q_pos:>5} "
        f"{len(keep_turns):>5} {len(comp_turns):>5} "
        f"{full_tokens:>9} {kept_tokens:>9} {est_reduction:>7.1f}%"
    )

print("-" * 105)
print(f"\nSUMMARY across {len(sample)} conversations:")
print(f"  Est. token reduction (pre-summary): {statistics.mean(reductions):.1f}% mean  |  {statistics.median(reductions):.1f}% median  |  {min(reductions):.1f}%–{max(reductions):.1f}% range")
print(f"  Turns KEPT:                         {statistics.mean(keep_pcts):.1f}% mean")
print(f"  Turns COMPRESSED:                   {statistics.mean(comp_pcts):.1f}% mean")
print(f"\nNote: actual reduction will be lower once summary tokens are added back.")
print(f"Target is 40-60% net reduction after summarisation.")
