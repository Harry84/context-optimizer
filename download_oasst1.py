#!/usr/bin/env python3
"""
Download the OpenAssistant/oasst1 dataset and export it as JSONL.

Usage:
    python download_oasst1.py
"""

from datasets import load_dataset
import os

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("🔄 Loading dataset from HuggingFace…")
    ds = load_dataset("OpenAssistant/oasst1")

    # The dataset has only a 'train' split
    train_split = ds["train"]

    output_path = os.path.join(output_dir, "oasst1.jsonl")

    print(f"💾 Saving to {output_path} …")
    train_split.to_json(output_path, orient="records", lines=True)

    print("✅ Done! Dataset saved locally.")
    print(f"   Total rows: {len(train_split)}")
    print(f"   File size: ~{os.path.getsize(output_path) / 1_000_000:.1f} MB")

if __name__ == "__main__":
    main()
