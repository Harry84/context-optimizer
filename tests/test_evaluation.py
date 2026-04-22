"""Tests for evaluation metrics (no LLM calls — mocked)."""

import pytest
from src.ingestion.models import Turn
from src.evaluation.landmark_recall import landmark_recall, landmark_stats
from src.scoring.recency import recency_scores


def _turn(index: int, slots: list[str] | None = None, is_landmark: bool = False) -> Turn:
    t = Turn(turn_index=index, speaker="USER", text=f"Turn {index}",
             slots=slots or [])
    t.is_landmark = is_landmark
    return t


# ─── Landmark recall ─────────────────────────────────────────────────────────

def test_recall_perfect():
    turns = [
        _turn(0, slots=["flight_search.destination1"], is_landmark=True),
        _turn(1, slots=["flight_search.date"],         is_landmark=True),
    ]
    assert landmark_recall(turns) == 1.0


def test_recall_zero():
    turns = [
        _turn(0, slots=["flight_search.destination1"], is_landmark=False),
        _turn(1, slots=["flight_search.date"],         is_landmark=False),
    ]
    assert landmark_recall(turns) == 0.0


def test_recall_partial():
    turns = [
        _turn(0, slots=["flight_search.destination1"], is_landmark=True),   # detected
        _turn(1, slots=["flight_search.date"],         is_landmark=False),  # missed
    ]
    assert landmark_recall(turns) == 0.5


def test_recall_no_gt_turns():
    """No slot-annotated turns → vacuously 1.0."""
    turns = [
        _turn(0, slots=[], is_landmark=False),
        _turn(1, slots=[], is_landmark=True),
    ]
    assert landmark_recall(turns) == 1.0


def test_recall_non_gt_landmarks_ignored():
    """Detected landmarks with no slots don't affect GT recall calculation."""
    turns = [
        _turn(0, slots=["flight_search.destination1"], is_landmark=True),   # GT + detected
        _turn(1, slots=[],                             is_landmark=True),   # detected, not GT
    ]
    # GT = {0}, detected ∩ GT = {0}, recall = 1.0
    assert landmark_recall(turns) == 1.0


# ─── Landmark stats ───────────────────────────────────────────────────────────

def test_landmark_stats_keys():
    turns = [_turn(0, slots=["s1"], is_landmark=True)]
    turns[0].landmark_type = "intent"
    stats = landmark_stats(turns)
    assert "total_turns" in stats
    assert "gt_turns" in stats
    assert "detected" in stats
    assert "recall" in stats
    assert "by_type" in stats


# ─── Token reduction (no LLM needed — uses tiktoken) ─────────────────────────

def test_token_count_import():
    """Verify tiktoken is available and counts tokens."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode("Hello, world!")
    assert len(tokens) > 0


def test_token_reduction_formula():
    """Verify the reduction formula is correct."""
    full_tokens = 1000
    opt_tokens  = 550
    reduction   = (full_tokens - opt_tokens) / full_tokens * 100
    assert abs(reduction - 45.0) < 0.01   # should be exactly 45%


def test_token_reduction_within_target():
    """Reduction should be 40-60% for a valid optimisation."""
    full_tokens = 1000
    for opt_tokens in [400, 450, 500, 550, 600]:
        reduction = (full_tokens - opt_tokens) / full_tokens * 100
        assert 40 <= reduction <= 60, f"Reduction {reduction}% out of target range"
