#!/usr/bin/env python3
"""Quick diagnostic: print raw turns around a given position."""
import json, sys

conv_id  = "dlg-cbfc519d-93e3-404d-9db5-c5fe35a5b765"
path     = "data/taskmaster2/flights.json"
start    = int(sys.argv[1]) if len(sys.argv) > 1 else 33
end      = int(sys.argv[2]) if len(sys.argv) > 2 else 42

with open(path) as f:
    data = json.load(f)

conv = next(c for c in data if c["conversation_id"] == conv_id)
for i, utt in enumerate(conv["utterances"]):
    if start <= i <= end:
        print(f"{i:3d} [{utt['speaker']:9}] {utt['text']!r}")
