#!/usr/bin/env python3
"""
View conversations from Taskmaster-2.

Usage:
    python utilities/view_conversation.py --stats      # turn length distribution
    python utilities/view_conversation.py 1            # longest conversation
    python utilities/view_conversation.py 2            # 2nd longest
    python utilities/view_conversation.py dlg-abc123   # by id
"""

import json
import os
import sys

DATA_DIR = "data/taskmaster2"


def load_all():
    convs = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        domain = fname.replace(".json", "")
        for conv in data:
            conv["_domain"] = domain
            convs.append(conv)
    return convs


def display(conv):
    utterances = conv.get("utterances", [])
    print(f"Conversation ID : {conv.get('conversation_id')}")
    print(f"Instruction ID  : {conv.get('instruction_id')}")
    print(f"Domain          : {conv.get('_domain')}")
    print(f"Total turns     : {len(utterances)}")
    print()
    for utt in utterances:
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        slots = [
            ann.get("name", "")
            for seg in utt.get("segments", [])
            for ann in seg.get("annotations", [])
        ]
        label = "USER     " if speaker == "USER" else "ASSISTANT"
        print(f"  {label} | {text}")
        if slots:
            print(f"             [slots: {', '.join(slots)}]")
    print()


def stats(convs):
    counts = sorted(len(c["utterances"]) for c in convs)
    n = len(counts)
    print(f"Total conversations: {n}")
    print(f"Min / Max / Mean / Median: {counts[0]} / {counts[-1]} / {sum(counts)/n:.1f} / {counts[n//2]}")
    for t in [10, 20, 30, 50]:
        c = sum(1 for x in counts if x >= t)
        print(f">= {t:>3} turns: {c:>5} ({100*c/n:.1f}%)")
    from collections import Counter
    domains = Counter(c["_domain"] for c in convs)
    print("\nBy domain:")
    for d, cnt in domains.most_common():
        print(f"  {d}: {cnt}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    convs = load_all()

    if arg == "--stats":
        stats(convs)
        return

    sorted_convs = sorted(convs, key=lambda c: len(c["utterances"]), reverse=True)

    if arg.isdigit():
        rank = int(arg) - 1
        conv = sorted_convs[rank]
        print(f"Rank {rank+1} longest conversation:\n")
    else:
        conv = next((c for c in convs if c.get("conversation_id") == arg), None)
        if not conv:
            print(f"ID '{arg}' not found.")
            return

    display(conv)
    print("Tip: --stats | 1 | 2 | 3 | dlg-id")


if __name__ == "__main__":
    main()
