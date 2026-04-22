"""
Inspect per-turn dispositions AND assembled output for the worst-compressing
conversations. No API calls — summaries shown as [SUM] placeholders.

Usage:
    python utilities/inspect_poor_compressors.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, warnings
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
from src.compression.compressor import classify_turns, group_into_runs

QUERY      = "What flights were compared and what did the user decide?"
QUERY_TYPE = classify_query(QUERY)
SAMPLE_N   = 20
MIN_TURNS  = 50
SEED       = 42
SHOW_WORST = 3

# Disposition abbreviations
DISP = {"KEEP": "K", "CANDIDATE": "C", "COMPRESS": "X"}
# Speaker abbreviations
SPK  = {"USER": "U", "ASSISTANT": "A"}

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
    runs       = group_into_runs(classified)

    full_tok = sum(len(enc.encode(t.text)) for t in history)
    comp_tok = sum(len(enc.encode(t.text)) for t in classified if t.disposition == "COMPRESS")
    comp_tok_pct = 100 * comp_tok / full_tok if full_tok > 0 else 0

    rows.append((comp_tok_pct, conv, history, classified, runs, q_pos, full_tok))

rows.sort(key=lambda r: r[0])

W = 100

for comp_tok_pct, conv, history, classified, runs, q_pos, full_tok in rows[:SHOW_WORST]:
    lm_count   = sum(1 for t in classified if t.is_landmark)
    keep_count = sum(1 for t in classified if t.disposition in ("KEEP", "CANDIDATE"))
    comp_count = sum(1 for t in classified if t.disposition == "COMPRESS")

    print()
    print("=" * W)
    print(f"{conv.conversation_id}  hist={q_pos} tok={full_tok} comp%={comp_tok_pct:.0f}% lm={lm_count} keep={keep_count} compress={comp_count}")
    print("=" * W)

    # Dispositions  (K=keep C=candidate X=compress)
    print("D S  sc  tk  lm          text")
    print("-" * W)
    for turn in classified:
        tok    = len(enc.encode(turn.text))
        lm_tag = (turn.landmark_type or "")[:10]
        if turn.promoted: lm_tag += "+"
        d = DISP.get(turn.disposition, "?")
        s = SPK.get(turn.speaker, "?")
        print(f"{d} {s} {turn.score:.2f} {tok:>3}  {lm_tag:<11} {turn.text[:60]}")

    # Assembled output
    print(f"\nASSEMBLED  (K=verbatim  S=summarised  D=dropped)")
    print("-" * W)
    keep_tok_total = 0
    for disposition, run_turns in runs:
        if disposition == "KEEP":
            for t in run_turns:
                tok = len(enc.encode(t.text))
                keep_tok_total += tok
                s   = SPK.get(t.speaker, "?")
                lm  = t.landmark_type[:6] if t.is_landmark else ""
                print(f"K {s} {tok:>3}tk {lm:<7} {t.text[:68]}")
        else:
            run_chars = sum(len(t.text.strip()) for t in run_turns)
            run_tok   = sum(len(enc.encode(t.text)) for t in run_turns)
            if run_chars >= 200:
                keep_tok_total += 15
                print(f"S - ~15tk         ({len(run_turns)}t {run_tok}tk → ~15tk summary)")
            else:
                print(f"D -   0tk         ({len(run_turns)}t {run_tok}tk → dropped)")

    est_red = 100 * (full_tok - keep_tok_total) / full_tok if full_tok > 0 else 0
    print(f"\n  output≈{keep_tok_total}tok  reduction≈{est_red:.0f}%")
