"""Tests for ingestion and normalisation."""

import pytest
from src.ingestion.loader import load_corpus, _normalise_turn
from src.ingestion.models import Turn, Conversation


def _make_raw_utt(speaker: str, text: str, slots: list[str] | None = None) -> dict:
    slots = slots or []
    return {
        "speaker": speaker,
        "text": text,
        "segments": [
            {"annotations": [{"name": s}], "start_index": 0, "end_index": 1, "text": "x"}
            for s in slots
        ],
    }


def test_normalise_turn_basic():
    raw = _make_raw_utt("USER", "I need a flight to Orlando.", ["flight_search.destination1"])
    turn = _normalise_turn(raw, index=3)
    assert turn.turn_index == 3
    assert turn.speaker == "USER"
    assert turn.text == "I need a flight to Orlando."
    assert "flight_search.destination1" in turn.slots


def test_normalise_turn_no_slots():
    raw = _make_raw_utt("ASSISTANT", "Okay.")
    turn = _normalise_turn(raw, index=0)
    assert turn.slots == []
    assert turn.is_landmark is False
    assert turn.score == 0.0
    assert turn.disposition == ""


def test_normalise_turn_strips_whitespace():
    raw = _make_raw_utt("USER", "  Hello.  ")
    turn = _normalise_turn(raw, index=0)
    assert turn.text == "Hello."


def test_load_corpus_filters_by_min_turns(tmp_path):
    """load_corpus should exclude conversations below min_turns."""
    import json

    # Short conversation (3 turns) — should be excluded at min_turns=20
    short_conv = {
        "conversation_id": "short-1",
        "instruction_id": "flight-1",
        "utterances": [
            _make_raw_utt("USER", "Hi."),
            _make_raw_utt("ASSISTANT", "Hello."),
            _make_raw_utt("USER", "Bye."),
        ],
    }
    # Long enough conversation (25 turns)
    long_conv = {
        "conversation_id": "long-1",
        "instruction_id": "flight-1",
        "utterances": [
            _make_raw_utt("USER" if i % 2 == 0 else "ASSISTANT", f"Turn {i}.")
            for i in range(25)
        ],
    }

    data_file = tmp_path / "flights.json"
    data_file.write_text(json.dumps([short_conv, long_conv]))

    convs = load_corpus(str(data_file), min_turns=20)
    assert len(convs) == 1
    assert convs[0].conversation_id == "long-1"
    assert len(convs[0].turns) == 25


def test_load_corpus_domain_inferred(tmp_path):
    import json
    conv = {
        "conversation_id": "x",
        "instruction_id": "flight-1",
        "utterances": [_make_raw_utt("USER", f"Turn {i}.") for i in range(20)],
    }
    data_file = tmp_path / "flights.json"
    data_file.write_text(json.dumps([conv]))
    convs = load_corpus(str(data_file), min_turns=20)
    assert convs[0].domain == "flights"
