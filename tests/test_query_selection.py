"""Tests for query selection — no real LLM calls, OpenAI client mocked."""

from unittest.mock import MagicMock, patch
import json
import pytest

from src.ingestion.models import Conversation, Turn
from src.evaluation.harness import QUERY_POOL, select_queries, EvalQuery, _conversation_snippet


def _make_conv(n_turns: int = 20) -> Conversation:
    turns = [
        Turn(
            turn_index=i,
            speaker="USER" if i % 2 == 0 else "ASSISTANT",
            text=f"I want to fly to Paris on March 10th." if i % 2 == 0 else "Sure, let me check.",
        )
        for i in range(n_turns)
    ]
    return Conversation(
        conversation_id="dlg-test-001",
        instruction_id="flight-001",
        turns=turns,
    )


def _mock_response(indices: list[int]) -> MagicMock:
    """Build a mock OpenAI response returning the given indices."""
    msg = MagicMock()
    msg.content = json.dumps({"indices": indices})
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ─── QUERY_POOL sanity ────────────────────────────────────────────────────────

def test_query_pool_not_empty():
    assert len(QUERY_POOL) >= 10


def test_query_pool_types():
    valid_types = {"factual", "analytical", "preference"}
    for query_text, query_type in QUERY_POOL:
        assert query_type in valid_types, f"Unknown type {query_type!r} for query {query_text!r}"
        assert len(query_text) > 10, "Query text too short"


def test_query_pool_has_all_types():
    types = {qt for _, qt in QUERY_POOL}
    assert "factual" in types
    assert "analytical" in types
    assert "preference" in types


# ─── select_queries — happy path ──────────────────────────────────────────────

@patch("src.evaluation.harness._get_client")
def test_select_queries_returns_two(mock_get_client):
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response([0, 8])
    conv    = _make_conv()
    queries = select_queries(conv, model="gpt-4o-mini")
    assert len(queries) == 2


@patch("src.evaluation.harness._get_client")
def test_select_queries_returns_eval_query_objects(mock_get_client):
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response([0, 8])
    conv    = _make_conv()
    queries = select_queries(conv, model="gpt-4o-mini")
    for q in queries:
        assert isinstance(q, EvalQuery)
        assert q.query_position > 0
        assert len(q.query_text) > 0
        assert q.query_type in {"factual", "analytical", "preference"}


@patch("src.evaluation.harness._get_client")
def test_select_queries_uses_pool_text(mock_get_client):
    indices = [2, 9]
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response(indices)
    conv    = _make_conv()
    queries = select_queries(conv, model="gpt-4o-mini")
    assert queries[0].query_text == QUERY_POOL[2][0]
    assert queries[1].query_text == QUERY_POOL[9][0]


@patch("src.evaluation.harness._get_client")
def test_select_queries_query_position_at_75pct(mock_get_client):
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response([0, 8])
    conv    = _make_conv(n_turns=60)
    queries = select_queries(conv, model="gpt-4o-mini")
    expected_pos = max(5, int(60 * 0.75))
    for q in queries:
        assert q.query_position == expected_pos


# ─── _conversation_snippet ────────────────────────────────────────────────────

def test_conversation_snippet_uses_last_n_turns():
    """Snippet should return the LAST n turns of the slice, not the first."""
    turns = [
        Turn(turn_index=i, speaker="USER", text=f"turn-{i}")
        for i in range(20)
    ]
    snippet = _conversation_snippet(turns, n_turns=5)
    assert "turn-19" in snippet
    assert "turn-15" in snippet
    assert "turn-0" not in snippet


def test_conversation_snippet_shorter_than_n_turns():
    """Snippet returns all turns when fewer than n_turns exist."""
    turns = [Turn(turn_index=i, speaker="USER", text=f"turn-{i}") for i in range(3)]
    snippet = _conversation_snippet(turns, n_turns=15)
    assert "turn-0" in snippet
    assert "turn-2" in snippet


# ─── select_queries — prompt uses turns near query position ───────────────────

def _make_long_conv(n_turns: int = 60) -> Conversation:
    """Conversation where early turns say 'early-turn-N' and late turns say 'late-turn-N'."""
    turns = []
    for i in range(n_turns):
        label = "early" if i < 15 else "late"
        turns.append(Turn(
            turn_index=i,
            speaker="USER" if i % 2 == 0 else "ASSISTANT",
            text=f"{label}-turn-{i}",
        ))
    return Conversation(conversation_id="dlg-long", instruction_id="x", turns=turns)


@patch("src.evaluation.harness._get_client")
def test_select_queries_prompt_uses_turns_near_query_position(mock_get_client):
    """For a long conversation, the LLM prompt should contain late turns, not early ones."""
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response([0, 8])
    conv = _make_long_conv(n_turns=60)
    select_queries(conv, model="gpt-4o-mini")

    call_args = mock_get_client.return_value.chat.completions.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]

    assert "late-turn" in prompt
    assert "early-turn" not in prompt


# ─── select_queries — fallback on bad LLM response ───────────────────────────

@patch("src.evaluation.harness._get_client")
def test_select_queries_fallback_on_invalid_json(mock_get_client):
    bad = MagicMock()
    bad.content = "not json at all"
    choice = MagicMock(); choice.message = bad
    resp   = MagicMock(); resp.choices = [choice]
    mock_get_client.return_value.chat.completions.create.return_value = resp
    queries = select_queries(_make_conv(), model="gpt-4o-mini")
    assert len(queries) == 2  # falls back to defaults [0, 8]
    assert queries[0].query_text == QUERY_POOL[0][0]
    assert queries[1].query_text == QUERY_POOL[8][0]


@patch("src.evaluation.harness._get_client")
def test_select_queries_fallback_on_out_of_range_index(mock_get_client):
    mock_get_client.return_value.chat.completions.create.return_value = _mock_response([0, 999])
    queries = select_queries(_make_conv(), model="gpt-4o-mini")
    assert len(queries) == 2
    assert queries[0].query_text == QUERY_POOL[0][0]


@patch("src.evaluation.harness._get_client")
def test_select_queries_fallback_on_api_error(mock_get_client):
    mock_get_client.return_value.chat.completions.create.side_effect = Exception("API down")
    queries = select_queries(_make_conv(), model="gpt-4o-mini")
    assert len(queries) == 2
    assert queries[0].query_text == QUERY_POOL[0][0]
