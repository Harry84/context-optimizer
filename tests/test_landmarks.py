"""Tests for landmark detection — rule-based + two-pass alignment."""

import pytest
from src.ingestion.models import Conversation, Turn
from src.landmarks.rule_detector import RuleLandmarkDetector


def _conv(utterances: list[tuple[str, str, list[str] | None]]) -> Conversation:
    turns = [
        Turn(turn_index=i, speaker=sp, text=tx, slots=sl or [])
        for i, (sp, tx, sl) in enumerate(utterances)
    ]
    return Conversation(conversation_id="test", instruction_id="test", turns=turns)


detector = RuleLandmarkDetector()


# ─── Stated intent tests ─────────────────────────────────────────────────────

def test_intent_explicit_verb():
    conv = _conv([("USER", "I'd like a flight to Orlando.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "intent"


def test_intent_slot_signal_price():
    conv = _conv([("USER", "My budget is under $1,000.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "intent"


def test_intent_slot_signal_airline():
    conv = _conv([("USER", "I'd prefer Delta if possible.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark


def test_intent_slot_signal_date():
    conv = _conv([("USER", "I need to fly on March 14th.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark


# ─── Decision tests ──────────────────────────────────────────────────────────

def test_decision_strong_confirmation():
    conv = _conv([("USER", "I'll take the 6AM flight.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_decision_assistant_offer():
    conv = _conv([("ASSISTANT", "I found a flight departing at 8:15 AM for $1505.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_decision_weak_confirmation_after_offer():
    conv = _conv([
        ("ASSISTANT", "The flight costs $1505. Is that okay?", None),
        ("USER", "Yes.", None),
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[1].is_landmark
    assert conv.turns[1].promoted


def test_decision_weak_confirmation_no_offer_not_promoted():
    conv = _conv([
        ("ASSISTANT", "Anything else I can help with?", None),
        ("USER", "Yes.", None),
    ])
    detector.detect(conv)
    assert not conv.turns[1].is_landmark


# ─── Conversation close tests ─────────────────────────────────────────────────

def test_conversation_close_that_will_be_all():
    """User signals end of conversation without explicit booking."""
    conv = _conv([("USER", "Okay. That will be all.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_conversation_close_im_done():
    conv = _conv([("USER", "Oh, I'm done.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_conversation_close_that_is_everything():
    conv = _conv([("USER", "I think that's everything I needed.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_conversation_close_nothing_else():
    conv = _conv([("USER", "Nothing else, thanks.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark


# ─── Action item tests ───────────────────────────────────────────────────────

def test_action_item_send():
    conv = _conv([("ASSISTANT", "I'll send you the flight details now.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "action_item"


def test_action_item_booking_confirmed():
    conv = _conv([("ASSISTANT", "Your tickets have been booked and confirmed.", None)])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type in ("action_item", "decision")


# ─── Filler tests ────────────────────────────────────────────────────────────

def test_filler_not_landmark():
    for text in ["Okay.", "Sure.", "Got it.", "Thank you.", "Bye."]:
        conv = _conv([("ASSISTANT", text, None)])
        detector.detect(conv)
        assert not conv.turns[0].is_landmark, f"Expected {text!r} to be filler"


# ─── Idempotency ─────────────────────────────────────────────────────────────

def test_detect_is_idempotent():
    conv = _conv([
        ("USER", "I need a flight to Paris.", None),
        ("ASSISTANT", "When would you like to travel?", None),
    ])
    detector.detect(conv)
    first  = [(t.is_landmark, t.landmark_type) for t in conv.turns]
    detector.detect(conv)
    second = [(t.is_landmark, t.landmark_type) for t in conv.turns]
    assert first == second


# ─── Cross-turn alignment (pass 2) ───────────────────────────────────────────

def test_pass2_pattern_b_echo():
    conv = _conv([
        ("USER", "I want to fly to London.", None),
        ("ASSISTANT", "So you want to fly to London, is that correct?", None),
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[1].is_landmark
