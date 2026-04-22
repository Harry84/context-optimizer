#!/usr/bin/env python3
"""Count viable conversations by domain and threshold."""

import json, os

DATA_DIR = "data/taskmaster2"
DOMAINS = ["flights", "hotels", "restaurant-search"]
THRESHOLD = 20

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".json"):
        continue
    domain = fname.replace(".json", "")
    with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
        convs = json.load(f)
    viable = [c for c in convs if len(c.get("utterances", [])) >= THRESHOLD]
    counts = sorted(len(c["utterances"]) for c in viable)
    if viable:
        print(f"{domain}: {len(viable)} conversations >= {THRESHOLD} turns | max {counts[-1]} | mean {sum(counts)/len(counts):.1f}")
    else:
        print(f"{domain}: 0 conversations >= {THRESHOLD} turns")
