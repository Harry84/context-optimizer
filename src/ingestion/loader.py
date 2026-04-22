"""
Stage 1 — Ingestion & Normalisation.

Loads Taskmaster-2 JSON, filters by turn count, normalises to
Conversation objects. Slot annotations extracted for evaluation use only.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.models import Conversation, OptimizerConfig, Turn


def load_corpus(
    path: str | Path,
    min_turns: int = 20,
) -> list[Conversation]:
    """
    Load a Taskmaster-2 domain JSON file and return conversations
    with at least min_turns turns.

    Args:
        path:      Path to a Taskmaster-2 domain JSON file
                   (e.g. data/taskmaster2/flights.json)
        min_turns: Minimum number of turns to include (default 20)

    Returns:
        List of Conversation objects, each with normalised Turn list.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_convs = json.load(f)

    conversations = []
    for raw in raw_convs:
        utterances = raw.get("utterances", [])
        if len(utterances) < min_turns:
            continue

        turns = [
            _normalise_turn(utt, idx)
            for idx, utt in enumerate(utterances)
        ]

        conversations.append(Conversation(
            conversation_id=raw.get("conversation_id", ""),
            instruction_id=raw.get("instruction_id", ""),
            turns=turns,
            domain=_infer_domain(path),
        ))

    return conversations


def _normalise_turn(raw: dict, index: int) -> Turn:
    """
    Normalise a raw Taskmaster-2 utterance dict into a Turn.

    Slot annotations are extracted and stored for evaluation ground-truth
    purposes only. They are never read by the landmark detector or scorer.
    """
    slots = [
        ann["name"]
        for seg in raw.get("segments", [])
        for ann in seg.get("annotations", [])
        if ann.get("name")
    ]
    return Turn(
        turn_index=index,
        speaker=raw.get("speaker", ""),
        text=raw.get("text", "").strip(),
        slots=slots,
    )


def _infer_domain(path: str | Path) -> str:
    return Path(path).stem  # e.g. "flights" from "flights.json"


def load_from_config(config: OptimizerConfig) -> list[Conversation]:
    """Convenience loader using config values."""
    return load_corpus(config.data_path, config.min_turns)
