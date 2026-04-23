#!/usr/bin/env python3
"""
Corpus-wide landmark recall using the production rule detector.

Runs src/landmarks/rule_detector.py across all viable conversations
and reports GT recall against Taskmaster-2 slot annotations.

This is the authoritative recall measurement — use this number in docs,
not verify_classifiers.py which uses a standalone reimplementation that
may have drifted from the production detector.

Usage:
    python utilities/corpus_landmark_recall.py
    python utilities/corpus_landmark_recall.py --min-turns 20
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import load_corpus
from src.ingestion.models import OptimizerConfig
from src.landmarks.detector import get_detector

DATA_PATH = "data/taskmaster2/flights.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-wide landmark recall (production detector)")
    parser.add_argument("--min-turns", type=int, default=20)
    parser.add_argument("--data-path", default=DATA_PATH)
    args = parser.parse_args()

    print(f"Loading {args.data_path} (min_turns={args.min_turns})...")
    conversations = load_corpus(args.data_path, min_turns=args.min_turns)
    print(f"Loaded {len(conversations):,} conversations\n")

    config   = OptimizerConfig(data_path=args.data_path, min_turns=args.min_turns)
    detector = get_detector(config)

    total_turns    = 0
    gt_total       = 0
    gt_detected    = 0
    lm_total       = 0
    promoted_total = 0
    by_type        = {"intent": 0, "decision": 0, "action_item": 0, "conv_close": 0}

    for conv in conversations:
        detector.detect(conv)
        for turn in conv.turns:
            total_turns += 1
            is_gt = bool(turn.slots)
            if is_gt:
                gt_total += 1
                if turn.is_landmark:
                    gt_detected += 1
            if turn.is_landmark:
                lm_total += 1
                t = turn.landmark_type or ""
                if t in by_type:
                    by_type[t] += 1
            if turn.promoted:
                promoted_total += 1

    recall = gt_detected / gt_total if gt_total else 0.0

    print(f"{'='*55}")
    print(f"CORPUS-WIDE LANDMARK RECALL (production detector)")
    print(f"{'='*55}")
    print(f"Conversations:        {len(conversations):,}")
    print(f"Total turns:          {total_turns:,}")
    print(f"GT turns (slot ann.): {gt_total:,}")
    print(f"GT detected:          {gt_detected:,}")
    print(f"")
    print(f"GT Recall:            {recall:.1%}  ({gt_detected:,}/{gt_total:,})")
    print(f"")
    print(f"Detected landmarks:   {lm_total:,} ({100*lm_total/total_turns:.1f}% of all turns)")
    print(f"  Stated intents:     {by_type['intent']:,}")
    print(f"  Decisions:          {by_type['decision']:,}")
    print(f"  Action items:       {by_type['action_item']:,}")
    print(f"  Conv close:         {by_type['conv_close']:,}")
    print(f"  Promoted (pass 2):  {promoted_total:,}")
    print(f"Compressible turns:   {total_turns-lm_total:,} ({100*(total_turns-lm_total)/total_turns:.1f}%)")
    print(f"{'='*55}")
    print(f"(Action items have no slot annotations — excluded from GT recall)")


if __name__ == "__main__":
    main()
