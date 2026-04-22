#!/usr/bin/env python3
"""
Download dialog2flow dataset by fetching raw JSON files directly
from HuggingFace Hub. Saves each subdataset immediately after download
so progress is preserved if a later file crashes.

Usage:
    python download_dialog2flow.py
"""

import json
import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "sergioburdisso/dialog2flow-dataset"
OUTPUT_DIR = "data/dialog2flow"


def try_load(local_path, filepath):
    """Try to load a file as JSON or JSONL, print a preview first."""
    with open(local_path, "r", encoding="utf-8") as f:
        raw = f.read(300)
    print(f"   Preview: {repr(raw[:200])}")

    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try standard JSON
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            dialogs = []
            for v in data.values():
                if isinstance(v, list):
                    dialogs.extend(v)
            return dialogs if dialogs else [data]
    except json.JSONDecodeError:
        pass

    # Try JSONL
    try:
        dialogs = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                dialogs.append(json.loads(line))
        if dialogs:
            print(f"   Parsed as JSONL: {len(dialogs)} records")
            return dialogs
    except json.JSONDecodeError:
        pass

    print(f"   ⚠️  Could not parse {filepath}, skipping")
    return []


def save_subdataset(subdataset, dialogs):
    """Save a single subdataset's dialogs to its own JSONL file immediately."""
    out_path = os.path.join(OUTPUT_DIR, f"{subdataset}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps(d) + "\n")
    size_kb = os.path.getsize(out_path) / 1000
    print(f"   💾 Saved {out_path} ({len(dialogs)} records, {size_kb:.0f} KB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 Listing files in {REPO_ID}...")
    files = list(list_repo_files(REPO_ID, repo_type="dataset"))

    json_files = [f for f in files if f.endswith("data.json")]
    print(f"   Found {len(json_files)} data.json files:")
    for f in json_files:
        print(f"   - {f}")

    # Skip already-saved subdatasets so re-runs don't re-download
    already_done = {
        f.replace(".jsonl", "")
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".jsonl")
    }
    if already_done:
        print(f"\n   Skipping already saved: {sorted(already_done)}")

    total_dialogs = 0
    failed = []

    for filepath in json_files:
        subdataset = filepath.split("/")[0]

        if subdataset in already_done:
            print(f"\n⏭️  Skipping {subdataset} (already saved)")
            continue

        print(f"\n⬇️  {filepath}...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filepath,
                repo_type="dataset",
            )
            dialogs = try_load(local_path, filepath)

            for d in dialogs:
                if isinstance(d, dict):
                    d["_source"] = subdataset

            save_subdataset(subdataset, dialogs)
            total_dialogs += len(dialogs)
            print(f"   ✅ {subdataset}: {len(dialogs)} records")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed.append(filepath)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total records saved: {total_dialogs}")
    if failed:
        print(f"Failed files ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")

    # Show what's in the output dir
    print(f"\n📁 Files in {OUTPUT_DIR}:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1000
        print(f"   {fname} ({size_kb:.0f} KB)")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
