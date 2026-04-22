"""Tests for landmark detection — rule-based + two-pass alignment."""

import pytest
from src.ingestion.models import Conversation, Turn
from src.landmarks.rule_detector import RuleLandmarkDetector


def _conv(utterances: list[tuple[str, str, list[str] | None]]) -> Conversation:
    """Helper: build a minimal Conversation from (speaker, text, slots?) tuples."""
    turns = [
        Turn(
            turn_index=i,
            speaker=sp,
            text=tx,
            slots=sl or [],
        )
        for i, (sp, tx, sl) in enumerate(utterances)
    ]
    return Conversation(
        conversation_id="test",
        instruction_id="test",
        turns=turns,
    )


detector = RuleLandmarkDetector()


# ─── Stated intent tests ─────────────────────────────────────────────────────

def test_intent_explicit_verb():
    conv = _conv([("USER", "I'd like a flight to Orlando.", None)])
    detector.detect(conv)
    t = conv.turns[0]
    assert t.is_landmark
    assert t.landmark_type == "intent"


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
    conv = _conv([
        ("ASSISTANT", "I found a flight departing at 8:15 AM for $1505.", None)
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "decision"


def test_decision_weak_confirmation_after_offer():
    """Weak 'yes' after an ASSISTANT offer → both promoted to decision."""
    conv = _conv([
        ("ASSISTANT", "The flight costs $1505. Is that okay?", None),
        ("USER", "Yes.", None),
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark   # assistant offer
    assert conv.turns[1].is_landmark   # weak confirmation
    assert conv.turns[1].promoted


def test_decision_weak_confirmation_no_offer_not_promoted():
    """Bare 'yes' with no preceding offer → NOT promoted."""
    conv = _conv([
        ("ASSISTANT", "Anything else I can help with?", None),
        ("USER", "Yes.", None),
    ])
    detector.detect(conv)
    # The 'yes' should NOT be a landmark (no offer preceded it)
    assert not conv.turns[1].is_landmark


# ─── Action item tests ───────────────────────────────────────────────────────

def test_action_item_send():
    conv = _conv([
        ("ASSISTANT", "I'll send you the flight details now.", None)
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "action_item"


def test_action_item_booking_confirmed():
    conv = _conv([
        ("ASSISTANT", "Your tickets have been booked and confirmed.", None)
    ])
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[0].landmark_type == "action_item"


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
    first_result = [(t.is_landmark, t.landmark_type) for t in conv.turns]
    detector.detect(conv)
    second_result = [(t.is_landmark, t.landmark_type) for t in conv.turns]
    assert first_result == second_result


# ─── Cross-turn alignment (pass 2) ───────────────────────────────────────────

def test_pass2_pattern_b_echo():
    """USER constraint echoed by ASSISTANT → both promoted."""
    conv = _conv([
        ("USER", "I want to fly to London.", None),
        ("ASSISTANT", "So you want to fly to London, is that correct?", None),
    ])
    # Without pass 2, USER turn would be caught by slot signal.
    # ASSISTANT echo would also fire on slot signal.
    detector.detect(conv)
    assert conv.turns[0].is_landmark
    assert conv.turns[1].is_landmark
