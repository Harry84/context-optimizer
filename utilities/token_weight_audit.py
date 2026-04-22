"""
Token weight audit — per-conversation breakdown with diagnostics on WHY
some conversations compress poorly.

Usage:
    python utilities/token_weight_audit.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, warnings, statistics
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

from src.ingestion.loader import load_from_config
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector
from src.scoring.query_classifier import classify_query
from src.scoring.scorer import score_turns
from src.compression.compressor import classify_turns

QUERY      = "What flights were compared and what did the user decide?"
QUERY_TYPE = classify_query(QUERY)
SAMPLE_N   = 20
MIN_TURNS  = 50
SEED       = 42

config   = OptimizerConfig(min_turns=MIN_TURNS)
convs    = load_from_config(config)
sample   = random.Random(SEED).sample([c for c in convs if len(c.turns) >= MIN_TURNS], SAMPLE_N)
detector = get_detector(config)

rows = []

for conv in sample:
    detector.detect(conv)
    n     = len(conv.turns)
    q_pos = max(5, int(n * 0.75))
    history = conv.turns[:q_pos]

    scored     = score_turns(history, QUERY, q_pos, config)
    classified = classify_turns(scored, QUERY_TYPE, config)

    keep = [t for t in classified if t.disposition in ("KEEP", "CANDIDATE")]
    comp = [t for t in classified if t.disposition == "COMPRESS"]
    lm   = [t for t in classified if t.is_landmark]

    full_tok = sum(len(enc.encode(t.text)) for t in history)
    keep_tok = sum(len(enc.encode(t.text)) for t in keep)
    comp_tok = sum(len(enc.encode(t.text)) for t in comp)

    avg_keep_len = keep_tok / len(keep) if keep else 0
    avg_comp_len = comp_tok / len(comp) if comp else 0
    avg_turn_len = full_tok / len(history) if history else 0
    lm_rate      = len(lm) / len(history) if history else 0
    comp_tok_pct = 100 * comp_tok / full_tok if full_tok > 0 else 0
    scores       = [t.score for t in classified]
    high_score_pct = 100 * sum(1 for s in scores if s >= 0.65) / len(scores) if scores else 0

    rows.append({
        "id":             conv.conversation_id[-12:],  # truncate for display
        "hist":           q_pos,
        "keep_n":         len(keep),
        "comp_n":         len(comp),
        "full_tok":       full_tok,
        "comp_tok_pct":   comp_tok_pct,
        "avg_turn_tok":   avg_turn_len,
        "avg_keep_tok":   avg_keep_len,
        "avg_comp_tok":   avg_comp_len,
        "lm_rate_pct":    100 * lm_rate,
        "high_score_pct": high_score_pct,
    })

# Sort by comp_tok_pct ascending so worst compressors are at top
rows.sort(key=lambda r: r["comp_tok_pct"])

print(f"{'Conv (tail)':<14} {'Hist':>5} {'KEEP':>5} {'COMP':>5} {'Tok':>6} {'COMP tok%':>9} {'Avg tok/turn':>13} {'Avg tok/KEEP':>13} {'Avg tok/COMP':>13} {'LM%':>5} {'Hi-sc%':>7}")
print("-" * 120)

for r in rows:
    print(
        f"{r['id']:<14} {r['hist']:>5} {r['keep_n']:>5} {r['comp_n']:>5} "
        f"{r['full_tok']:>6} {r['comp_tok_pct']:>8.1f}% "
        f"{r['avg_turn_tok']:>12.1f} "
        f"{r['avg_keep_tok']:>12.1f} "
        f"{r['avg_comp_tok']:>12.1f} "
        f"{r['lm_rate_pct']:>4.0f}% "
        f"{r['high_score_pct']:>6.0f}%"
    )

print("-" * 120)

comp_pcts = [r["comp_tok_pct"] for r in rows]
print(f"\nMax possible token reduction (if summaries → 0):")
print(f"  Mean {statistics.mean(comp_pcts):.1f}%  |  Median {statistics.median(comp_pcts):.1f}%  |  Range {min(comp_pcts):.1f}%–{max(comp_pcts):.1f}%")
print(f"  ≥40%: {sum(1 for p in comp_pcts if p >= 40)}/{len(rows)} conversations")
print(f"  <25%: {sum(1 for p in comp_pcts if p < 25)}/{len(rows)} conversations")

print("\nDIAGNOSTICS — what drives poor compression (bottom of table):")
poor  = [r for r in rows if r["comp_tok_pct"] < 25]
good  = [r for r in rows if r["comp_tok_pct"] >= 40]
if poor and good:
    print(f"  Poor compressors (COMP tok% < 25):  avg turn length {statistics.mean(r['avg_turn_tok'] for r in poor):.1f} tok  |  LM rate {statistics.mean(r['lm_rate_pct'] for r in poor):.0f}%  |  Hi-score% {statistics.mean(r['high_score_pct'] for r in poor):.0f}%")
    print(f"  Good compressors (COMP tok% ≥ 40):  avg turn length {statistics.mean(r['avg_turn_tok'] for r in good):.1f} tok  |  LM rate {statistics.mean(r['lm_rate_pct'] for r in good):.0f}%  |  Hi-score% {statistics.mean(r['high_score_pct'] for r in good):.0f}%")
