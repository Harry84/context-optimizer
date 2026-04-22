#!/usr/bin/env python3
"""Analyse Taskmaster2 turn lengths and speaker patterns."""

import json

with open("data/dialog2flow/Taskmaster2.jsonl", "r", encoding="utf-8") as f:
    record = json.loads(f.readline().strip())

dialogs = record["dialogs"]
print(f"Total dialogs: {len(dialogs)}")

turn_counts = []
speaker_issues = 0  # dialogs with consecutive same-speaker turns

for dialog_id, turns in dialogs.items():
    turn_counts.append(len(turns))
    speakers = [t["speaker"] for t in turns]
    for i in range(1, len(speakers)):
        if speakers[i] == speakers[i-1]:
            speaker_issues += 1
            break

turn_counts.sort()
n = len(turn_counts)
print(f"Dialogs with consecutive same-speaker turns: {speaker_issues}")
print(f"\nTurn length distribution:")
print(f"  Min:    {min(turn_counts)}")
print(f"  Max:    {max(turn_counts)}")
print(f"  Mean:   {sum(turn_counts)/n:.1f}")
print(f"  Median: {turn_counts[n//2]}")
for t in [10, 20, 30, 50]:
    count = sum(1 for l in turn_counts if l >= t)
    print(f"  >= {t:>3} turns: {count:>5} ({100*count/n:.1f}%)")

# Show a longer dialog as example
longest_id = max(dialogs.keys(), key=lambda k: len(dialogs[k]))
longest = dialogs[longest_id]
print(f"\nLongest dialog: {longest_id} ({len(longest)} turns)")
print(json.dumps(longest[:4], indent=2))
