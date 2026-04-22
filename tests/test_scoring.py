"""Tests for relevance scoring."""

import pytest
from src.ingestion.models import OptimizerConfig, Turn
from src.scoring.keyword import keyword_scores
from src.scoring.query_classifier import classify_query
from src.scoring.recency import recency_scores
from src.scoring.scorer import score_turns


# ─── Query classifier ────────────────────────────────────────────────────────

def test_classify_factual():
    assert classify_query("What time does the flight depart?") == "factual"
    assert classify_query("How much did the ticket cost?") == "factual"


def test_classify_analytical():
    assert classify_query("Why did they choose Paris over Budapest?") == "analytical"
    assert classify_query("How do the two flights compare on layovers?") == "analytical"


def test_classify_preference():
    assert classify_query("Which flight would you recommend?") == "preference"
    assert classify_query("What would you suggest based on price?") == "preference"


def test_classify_default_analytical():
    # Ambiguous query → default analytical
    assert classify_query("Tell me about the conversation.") == "analytical"


# ─── Keyword scores ──────────────────────────────────────────────────────────

def test_keyword_scores_length():
    texts = ["I need a flight to Paris.", "The price is $500.", "Okay."]
    scores = keyword_scores("flight to Paris", texts)
    assert len(scores) == 3


def test_keyword_scores_relevant_turn_higher():
    texts = ["I need a flight to Paris.", "The weather is nice today."]
    scores = keyword_scores("flight to Paris", texts)
    assert scores[0] > scores[1]


def test_keyword_scores_empty_texts():
    scores = keyword_scores("flight", [])
    assert scores == []


def test_keyword_scores_in_range():
    texts = ["I want a nonstop flight.", "Okay.", "Business class please."]
    scores = keyword_scores("nonstop business class", texts)
    assert all(0.0 <= s <= 1.0 for s in scores)


# ─── Recency scores ──────────────────────────────────────────────────────────

def test_recency_scores_monotonic():
    """More recent turns should score higher."""
    indices = [0, 5, 10, 15]
    scores = recency_scores(indices, query_position=20, lambda_decay=0.05)
    assert len(scores) == 4
    # Each score should be higher than the previous (more recent = higher index)
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))


def test_recency_scores_latest_is_highest():
    indices = [0, 1, 2, 9]
    scores = recency_scores(indices, query_position=10)
    assert scores[-1] == max(scores)


def test_recency_scores_all_positive():
    scores = recency_scores([0, 5, 10], query_position=20)
    assert all(s > 0 for s in scores)


# ─── Composite scorer ─────────────────────────────────────────────────────────

def _make_turns(texts_and_speakers: list[tuple[str, str]]) -> list[Turn]:
    return [
        Turn(turn_index=i, speaker=sp, text=tx)
        for i, (sp, tx) in enumerate(texts_and_speakers)
    ]


def test_score_turns_sets_scores():
    config = OptimizerConfig()
    history = _make_turns([
        ("USER",      "I need a flight to Paris on March 14th."),
        ("ASSISTANT", "When would you like to return?"),
        ("USER",      "I'll return on March 21st."),
        ("ASSISTANT", "Okay."),
    ])
    scored = score_turns(history, "When is the flight to Paris?", query_position=4, config=config)
    assert all(0.0 <= t.score <= 1.0 for t in scored)
    # Every turn should have a score set
    assert all(t.score >= 0 for t in scored)


def test_score_turns_landmark_boost():
    config = OptimizerConfig(landmark_boost=0.3)
    history = _make_turns([
        ("USER", "Okay."),        # generic filler
        ("USER", "I'd like to fly to London on the 15th."),  # will be landmark
    ])
    # Manually set landmark on second turn
    history[1].is_landmark = True

    scored = score_turns(history, "Where is the flight going?", query_position=2, config=config)
    # Landmark turn should score higher due to boost
    assert scored[1].score >= scored[0].score


def test_score_turns_query_position_respected():
    """Only turns before query_position should be scored."""
    config = OptimizerConfig()
    history = _make_turns([
        ("USER", "I need a flight."),
        ("ASSISTANT", "Where to?"),
    ])
    # query_position=2 means both turns are history
    scored = score_turns(history, "Where is the flight?", query_position=2, config=config)
    assert len(scored) == 2


def test_score_turns_empty():
    config = OptimizerConfig()
    result = score_turns([], "any query", query_position=0, config=config)
    assert result == []
