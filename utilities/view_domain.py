#!/usr/bin/env python3
"""View longest conversations from a specific domain."""

import json, os, sys

DATA_DIR = "data/taskmaster2"
domain = sys.argv[1] if len(sys.argv) > 1 else "flights"
rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1

fpath = os.path.join(DATA_DIR, f"{domain}.json")
with open(fpath, "r", encoding="utf-8") as f:
    convs = json.load(f)

convs_sorted = sorted(convs, key=lambda c: len(c.get("utterances", [])), reverse=True)
conv = convs_sorted[rank - 1]
utterances = conv.get("utterances", [])

print(f"Domain          : {domain}")
print(f"Rank            : {rank} longest")
print(f"Conversation ID : {conv.get('conversation_id')}")
print(f"Instruction ID  : {conv.get('instruction_id')}")
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

print(f"\nTip: python utilities/view_domain.py flights 2")
