#!/usr/bin/env python3
"""
Download raw Taskmaster-2 from google-research-datasets/taskmaster2.
Proper full-conversation version with alternating USER/ASSISTANT turns.

Usage:
    python download_taskmaster2.py
"""

import json
import os
from datasets import load_dataset

OUTPUT_DIR = "data/taskmaster2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔄 Loading google-research-datasets/taskmaster2...")
dataset = load_dataset("google-research-datasets/taskmaster2")

print("\n📊 Structure:")
print(dataset)

# Flatten all splits into one list
all_convs = []
for split_name, split in dataset.items():
    for row in split:
        utterances = row.get("utterances", [])
        all_convs.append({
            "conversation_id": row.get("conversation_id"),
            "instruction_id": row.get("instruction_id"),
            "utterances": utterances,
            "_split": split_name,
        })

# Turn length analysis
turn_counts = sorted([len(c["utterances"]) for c in all_convs])
n = len(turn_counts)

print(f"\n📏 Turn length analysis across {n} conversations:")
print(f"  Min / Max / Mean / Median: {min(turn_counts)} / {max(turn_counts)} / {sum(turn_counts)/n:.1f} / {turn_counts[n//2]}")
for t in [10, 20, 30, 50]:
    count = sum(1 for l in turn_counts if l >= t)
    print(f"  >= {t:>3} turns: {count:>5} ({100*count/n:.1f}%)")

# Domain breakdown via instruction_id prefix
from collections import Counter
domains = Counter()
for c in all_convs:
    iid = c.get("instruction_id") or ""
    domain = iid.split("-")[0] if iid else "unknown"
    domains[domain] += 1
print(f"\n📂 Domains (by instruction_id prefix):")
for domain, count in domains.most_common():
    print(f"  {domain}: {count}")

# Show a sample conversation
sample = all_convs[0]
print(f"\n🔍 Sample conversation ({sample['conversation_id']}, {len(sample['utterances'])} turns):")
for utt in sample["utterances"][:8]:
    print(f"  [{utt.get('speaker')}] {utt.get('text', '')[:120]}")

# Save
output_path = os.path.join(OUTPUT_DIR, "taskmaster2.jsonl")
print(f"\n💾 Saving to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    for conv in all_convs:
        f.write(json.dumps(conv) + "\n")

size_mb = os.path.getsize(output_path) / 1_000_000
print(f"   {len(all_convs)} conversations | {size_mb:.1f} MB")
print("\n✅ Done!")
