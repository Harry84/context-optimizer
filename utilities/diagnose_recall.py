#!/usr/bin/env python3
"""
Show slot-annotated turns that the text-based detector MISSED.
Import detect_landmark directly from verify_classifiers.

Usage:
    python utilities/diagnose_recall.py
"""

import json
import re
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from verify_classifiers import detect_landmark, slot_annotated

DATA_PATH = "data/taskmaster2/flights.json"

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_convs = json.load(f)

    viable = [c for c in all_convs if len(c.get("utterances", [])) >= 20]

    missed = []
    caught = []

    for conv in viable:
        utterances = conv.get("utterances", [])
        prev_speaker, prev_text = "", ""
        for utt in utterances:
            speaker = utt.get("speaker", "")
            text    = utt.get("text", "").strip()
            slots   = [
                ann.get("name", "")
                for seg in utt.get("segments", [])
                for ann in seg.get("annotations", [])
            ]
            is_gt  = slot_annotated(utt)
            is_lm, lm_type, reason = detect_landmark(speaker, text, prev_speaker, prev_text)

            if is_gt and not is_lm:
                missed.append((speaker, text, slots))
            elif is_gt and is_lm:
                caught.append((speaker, text, slots))

            prev_speaker, prev_text = speaker, text

    total = len(missed) + len(caught)
    print(f"GT turns:  {total:,}")
    print(f"Caught:    {len(caught):,} ({100*len(caught)/total:.1f}%)")
    print(f"Missed:    {len(missed):,} ({100*len(missed)/total:.1f}%)")

    # Slot types in missed turns
    from collections import Counter
    missed_slots = Counter(s for _, _, slots in missed for s in slots)
    print(f"\nTop slot types in MISSED turns:")
    for slot, count in missed_slots.most_common(20):
        print(f"  {count:>5}  {slot}")

    # Split missed by speaker
    missed_user      = [(t, s) for sp, t, s in missed if sp == "USER"]
    missed_assistant = [(t, s) for sp, t, s in missed if sp == "ASSISTANT"]
    print(f"\nMissed USER turns:      {len(missed_user):,}")
    print(f"Missed ASSISTANT turns: {len(missed_assistant):,}")

    # Sample missed USER turns
    print(f"\n{'='*70}")
    print("SAMPLE MISSED USER TURNS")
    print(f"{'='*70}")
    random.seed(42)
    for text, slots in random.sample(missed_user, min(25, len(missed_user))):
        print(f"  [USER] {text[:100]}")
        print(f"    slots: {slots}")

    # Sample missed ASSISTANT turns
    print(f"\n{'='*70}")
    print("SAMPLE MISSED ASSISTANT TURNS")
    print(f"{'='*70}")
    for text, slots in random.sample(missed_assistant, min(25, len(missed_assistant))):
        print(f"  [ASST] {text[:100]}")
        print(f"    slots: {slots}")

if __name__ == "__main__":
    main()
