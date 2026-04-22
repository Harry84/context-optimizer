"""Tests for compression — turn classification, run grouping, and assembly."""

import pytest
from src.ingestion.models import OptimizerConfig, Turn
from src.compression.compressor import classify_turns, group_into_runs
from src.compression.assembler import assemble, format_full_context, _integrity_check


def _turn(index: int, speaker: str, text: str,
          is_landmark: bool = False, score: float = 0.0) -> Turn:
    t = Turn(turn_index=index, speaker=speaker, text=text)
    t.is_landmark = is_landmark
    t.score = score
    return t


# ─── Turn classification ──────────────────────────────────────────────────────

def test_landmark_always_keep():
    config = OptimizerConfig()
    turns = [_turn(0, "USER", "I need a flight.", is_landmark=True, score=0.0)]
    result = classify_turns(turns, "factual", config)
    assert result[0].disposition == "KEEP"


def test_high_score_keep():
    config = OptimizerConfig()
    turns = [_turn(0, "USER", "flight to Paris", score=0.8)]
    result = classify_turns(turns, "factual", config)
    assert result[0].disposition == "KEEP"


def test_low_score_compress():
    config = OptimizerConfig()
    turns = [_turn(0, "ASSISTANT", "Okay.", score=0.05)]
    result = classify_turns(turns, "factual", config)
    assert result[0].disposition == "COMPRESS"


def test_mid_score_candidate():
    config = OptimizerConfig()
    # factual thresholds: high=0.6, low=0.3
    turns = [_turn(0, "USER", "some turn", score=0.45)]
    result = classify_turns(turns, "factual", config)
    assert result[0].disposition == "CANDIDATE"


def test_thresholds_vary_by_query_type():
    config = OptimizerConfig()
    # Score of 0.5 is above analytical HIGH (0.5) but below factual HIGH (0.6)
    turns_a = [_turn(0, "USER", "text", score=0.5)]
    turns_f = [_turn(0, "USER", "text", score=0.5)]

    classify_turns(turns_a, "analytical", config)
    classify_turns(turns_f, "factual", config)

    assert turns_a[0].disposition == "KEEP"       # 0.5 >= 0.5 (analytical high)
    assert turns_f[0].disposition == "CANDIDATE"  # 0.5 < 0.6 (factual high), >= 0.3 (low)


# ─── Run grouping ────────────────────────────────────────────────────────────

def _classified(*dispositions) -> list[Turn]:
    turns = []
    for i, d in enumerate(dispositions):
        sp = "USER" if i % 2 == 0 else "ASSISTANT"
        t = _turn(i, sp, f"Turn {i}")
        t.disposition = d
        turns.append(t)
    return turns


def test_group_single_run():
    turns = _classified("KEEP", "KEEP", "KEEP")
    runs = group_into_runs(turns)
    assert len(runs) == 1
    assert runs[0][0] == "KEEP"
    assert len(runs[0][1]) == 3


def test_group_alternating():
    turns = _classified("KEEP", "KEEP", "COMPRESS", "COMPRESS", "KEEP", "COMPRESS")
    runs = group_into_runs(turns)
    dispositions = [r[0] for r in runs]
    assert dispositions == ["KEEP", "COMPRESS", "KEEP", "COMPRESS"]


def test_group_candidate_merged_with_keep():
    """CANDIDATE disposition is treated as KEEP for grouping."""
    turns = _classified("KEEP", "CANDIDATE", "KEEP")
    runs = group_into_runs(turns)
    assert len(runs) == 1
    assert runs[0][0] == "KEEP"


def test_group_empty():
    assert group_into_runs([]) == []


# ─── Assembly ────────────────────────────────────────────────────────────────

def _make_runs(specs: list[tuple[str, list[tuple[str, str]]]]):
    """Build runs list from (disposition, [(speaker, text)]) specs."""
    runs = []
    idx = 0
    for disposition, utterances in specs:
        turn_list = []
        for sp, tx in utterances:
            t = Turn(turn_index=idx, speaker=sp, text=tx)
            t.disposition = disposition
            turn_list.append(t)
            idx += 1
        runs.append((disposition, turn_list))
    return runs


def test_assemble_keep_verbatim():
    runs = _make_runs([
        ("KEEP", [("USER", "Hello."), ("ASSISTANT", "Hi there.")]),
    ])
    thread, stats = assemble(runs, summaries={})
    assert len(thread) == 2
    assert thread[0] == {"role": "user",      "content": "Hello."}
    assert thread[1] == {"role": "assistant",  "content": "Hi there."}
    assert stats.kept_verbatim == 2
    assert stats.summary_turns == 0


def test_assemble_compress_run_replaced():
    runs = _make_runs([
        ("COMPRESS", [("USER", "Some filler."), ("ASSISTANT", "More filler.")]),
    ])
    run_id = id(runs[0][1])
    summaries = {run_id: "The user asked about the flight and the assistant responded."}
    thread, stats = assemble(runs, summaries)

    assert len(thread) == 2   # placeholder + integrity inserts user at start
    assert any("[SUMMARY:" in msg["content"] for msg in thread)
    assert stats.summary_turns == 1


def test_assemble_mixed():
    runs = _make_runs([
        ("KEEP",     [("USER", "I need a flight to Paris.")]),
        ("COMPRESS", [("ASSISTANT", "Okay."), ("USER", "Sure.")]),
        ("KEEP",     [("ASSISTANT", "Here is the flight: departs 8AM.")]),
    ])
    run_id = id(runs[1][1])
    summaries = {run_id: "Routine exchange."}
    thread, stats = assemble(runs, summaries)

    roles = [m["role"] for m in thread]
    # Should not have consecutive same roles
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], f"Consecutive {roles[i]} at positions {i},{i+1}"


def test_integrity_check_merges_consecutive():
    thread = [
        {"role": "user",      "content": "Hello."},
        {"role": "user",      "content": "How are you?"},
        {"role": "assistant", "content": "Fine."},
    ]
    repaired, repairs = _integrity_check(thread)
    assert repairs == 1
    assert len(repaired) == 2
    assert "Hello." in repaired[0]["content"]
    assert "How are you?" in repaired[0]["content"]


def test_integrity_check_no_repairs_needed():
    thread = [
        {"role": "user",      "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
        {"role": "user",      "content": "Bye."},
    ]
    repaired, repairs = _integrity_check(thread)
    assert repairs == 0
    assert repaired == thread


def test_format_full_context():
    turns = [
        Turn(turn_index=0, speaker="USER",      text="I need a flight."),
        Turn(turn_index=1, speaker="ASSISTANT", text="Where to?"),
        Turn(turn_index=2, speaker="USER",      text="Paris."),
    ]
    thread = format_full_context(turns)
    assert len(thread) == 3
    assert thread[0]["role"] == "user"
    assert thread[1]["role"] == "assistant"
    assert thread[2]["role"] == "user"
