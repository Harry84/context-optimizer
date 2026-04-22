"""
v1 Landmark Detector — Rule-based + Two-Pass Cross-Turn Alignment.

Detection operates on RAW TEXT AND SPEAKER ROLE ONLY.
Slot annotations are never used here — they are evaluation ground truth only.

Measured performance on Taskmaster-2 flights (1,692 conversations, ≥20 turns):
  GT recall:   86.6%  (20,092 / 23,190 slot-annotated turns detected)
  Landmark %:  46.4%  of all turns
  Compressible: 53.6% of all turns
  Promoted by pass 2: 879 turns (3.7% of all landmarks)

See ARCHITECTURE.md §5.2 and key_decisions.md KD-009, KD-010 for design rationale.
"""

from __future__ import annotations

import re

from src.ingestion.models import Conversation, Turn

# ─── Slot-value signals ──────────────────────────────────────────────────────
# Domain vocabulary indicating a turn carries real flight information.
# Used for both USER intent detection and ASSISTANT offer/decision detection.

SLOT_SIGNALS: list[str] = [
    # Times
    r"\b\d{1,2}:\d{2}\s*(a\.?m\.?|p\.?m\.?)\b",
    # Prices
    r"\$\s*\d+",
    r"\b(under|less than|up to|around|about|max|maximum|budget|not more than)\b.{0,15}\$?\d",
    # Airlines
    r"\b(delta|united|american airlines?|southwest|jetblue|british airways|"
    r"lufthansa|frontier|virgin|alaska|air canada|klm|air new zealand|"
    r"spirit|hawaiian|allegiant)\b",
    # Seat class
    r"\b(business|economy|first|coach|premium economy)\b",
    # Stops
    r"\b(non.?stop|nonstop|direct|layover|stopover|one stop|two stops?|no stops?)\b",
    # Trip type
    r"\b(one.?way|round.?trip|multi.?city)\b",
    # Travel verb + direction
    r"\b(fly(ing)?|flight|travel(l?ing)?|going|heading|depart(ing)?|leav(e|ing))\b.{0,40}\b(to|from)\b",
    r"\bfrom\b.{2,40}\bto\b",
    # Time of day
    r"\b(morning|afternoon|evening|early|late|overnight|red.?eye)\b",
    # Dates
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d+(st|nd|rd|th)\b",
    r"\b(today|tomorrow|next week|this weekend|next month)\b",
    # Seat location
    r"\b(aisle|window|middle)\b.{0,10}(seat)?\b",
    # Amenities
    r"\b(wi.?fi|meal|food|alcohol|entertainment|baggage|luggage)\b",
    # Passenger count
    r"\b\d+\s*(people|persons?|passengers?|adults?|tickets?|seats?)\b",
]

_SLOT_SIGNAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in SLOT_SIGNALS]


def _has_slot_signal(text: str) -> bool:
    return any(p.search(text) for p in _SLOT_SIGNAL_COMPILED)


# ─── Filler detection ────────────────────────────────────────────────────────

_FILLER_RE = re.compile(
    r"^(okay|ok|sure|alright|great|perfect|fine|yes|yeah|no|"
    r"hold on|wait|right|correct|got it|i see|uh|um|"
    r"could you (say|repeat)|i('m| am) sorry|"
    r"what (else|other)|anything else|is that all|"
    r"thank(s| you)|goodbye|bye|hello|hi|hey|"
    r"sounds good|that sounds good|that works)[\.\!\?,]?\s*$",
    re.IGNORECASE,
)


def _is_pure_filler(text: str) -> bool:
    """Short turns with no slot signals matching acknowledgement patterns."""
    return len(text.strip()) < 50 and not _has_slot_signal(text) and bool(_FILLER_RE.match(text.strip()))


# ─── Intent patterns (USER) ──────────────────────────────────────────────────

_INTENT_VERB_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(i'?d? ?(like|want|need|prefer)|i'?m (looking for|trying to find))\b",
    r"\b(looking for|searching for|need to (find|book|get)|can you (find|get|book))\b",
    r"\b(i want|i need|i'?d prefer|i'?d like)\b",
]]

# ─── ASSISTANT offer patterns ────────────────────────────────────────────────

_OFFER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bi (found|have found|'?ve found)\b.{0,80}\b(flight|seat|option|ticket)\b",
    r"\b(departs?|leaves?|departing|leaving)\b.{0,30}\b\d{1,2}:\d{2}",
    r"\b(arrives?|arriving|lands?|arrival)\b.{0,30}\b\d{1,2}:\d{2}",
    r"\b(costs?|price is|fare is|total is|that.?s)\b.{0,20}\$\s*\d",
    r"\$\s*\d{3,}",
    r"\b(here are|your options?|you can (leave|depart|fly|travel) at)\b",
    r"\byou (will|would) (leave|depart|fly|arrive|land|be back|return)\b",
    r"\b(option|choice|flight) (1|2|3|one|two|three)\b",
    r"\b(layover|stopover|connection)\b.{0,30}\b(in|at)\b",
    r"\b\d+ ?(hour|minute|hr|min).{0,10}(layover|stopover|connection)\b",
    r"\byou are all set\b",
    r"\b(tickets?|booking|reservation) (have been|has been|is) (booked|confirmed|processed)\b",
    r"\bdetails? (have been|will be|are being) (sent|emailed|forwarded)\b",
    r"\blooks? like (there (is|are)|we have)\b",
    r"\bthere (is|are) (only |just )?\d",
    r"\bi (see|have) \d+ (flights?|options?|times?|choices?)\b",
]]


def _is_assistant_offer(text: str) -> bool:
    return any(p.search(text) for p in _OFFER_PATTERNS)


# ─── Confirmation patterns ───────────────────────────────────────────────────

_STRONG_CONFIRMATION = re.compile(
    r"\b(i'?ll (take|go with|choose|pick|book|do that)|"
    r"let'?s (go with|do|take|book|choose)|"
    r"book (it|that|both|them)|go ahead (and book|with that)|"
    r"i'?d like (to book|to go with|that one))\b",
    re.IGNORECASE,
)

_WEAK_CONFIRMATION = re.compile(
    r"^(yes|yeah|yep|okay|ok|sure|alright|confirmed|deal|done|great|perfect|"
    r"fine|correct|that'?s (right|correct|fine|good)|works for me)[\.\!,]?\s*$",
    re.IGNORECASE,
)

# ─── Echo patterns (ASSISTANT confirming back a user constraint) ─────────────

_ECHO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(you said|you mentioned|you want|you('d| would) like|you('re| are) looking for)\b",
    r"\bso (you want|you('d| would) like|you need|you('re| are) looking)\b",
    r"(correct|right|is that right|is that correct)\??\s*$",
    r"\bjust to confirm\b",
]]


def _is_assistant_echo(text: str) -> bool:
    return _has_slot_signal(text) and any(p.search(text) for p in _ECHO_PATTERNS)


# ─── Action item patterns (ASSISTANT committing to action) ───────────────────

_ACTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(i'?ll|i will|i'?m going to)\b.{0,50}\b(send|book|confirm|process|arrange|reserve|email)\b",
    r"\b(i have|i'?ve)\b.{0,40}\b(booked|confirmed|reserved|sent|processed|emailed)\b",
    r"\blet me (look that up|find|check|search|pull that up|look into)\b",
    r"\bi'?ll (look that up|find that|check that|search for)\b",
    r"\bokay,? let me send\b",
]]


# ─── Pass 1: Individual turn scoring ─────────────────────────────────────────

def _pass1(turn: Turn) -> tuple[bool, str | None, str]:
    """
    Score a single turn independently using text signals only.
    Returns (is_landmark, landmark_type, reason).
    """
    text = turn.text

    if _is_pure_filler(text):
        return False, None, ""

    text_l = text.lower().strip()

    if turn.speaker == "USER":
        # Strong confirmation → decision
        if _STRONG_CONFIRMATION.search(text_l):
            return True, "decision", "strong confirmation"

        # Explicit intent verb → stated intent
        for p in _INTENT_VERB_PATTERNS:
            if p.search(text_l):
                return True, "intent", "intent verb"

        # Contains slot-value signal → stated intent
        if _has_slot_signal(text):
            return True, "intent", "slot signal in user turn"

    if turn.speaker == "ASSISTANT":
        # Presenting concrete options/details → decision
        if _is_assistant_offer(text):
            return True, "decision", "assistant offer"

        # Contains slot-value signal → decision
        if _has_slot_signal(text):
            return True, "decision", "slot signal in assistant turn"

        # Commitment to action → action item
        for p in _ACTION_PATTERNS:
            if p.search(text_l):
                return True, "action_item", "action pattern"

    return False, None, ""


# ─── Pass 2: Cross-turn alignment ────────────────────────────────────────────

def _pass2(turns: list[Turn]) -> None:
    """
    Promote turns based on cross-turn structural patterns.
    Mutates turns in-place. Sets promoted=True on promoted turns.

    Pattern A — Offer → Confirmation:
      ASSISTANT[i] makes offer AND USER[i+1] gives weak confirmation
      → both promoted to landmark:decision

    Pattern B — Constraint → Echo:
      USER[i] has slot signal AND ASSISTANT[i+1] echoes it
      → both promoted to landmark:intent
    """
    n = len(turns)
    for i in range(n - 1):
        curr = turns[i]
        nxt  = turns[i + 1]

        # Pattern A
        if (curr.speaker == "ASSISTANT"
                and nxt.speaker == "USER"
                and _WEAK_CONFIRMATION.match(nxt.text.strip())
                and _is_assistant_offer(curr.text)):

            if not curr.is_landmark:
                curr.is_landmark    = True
                curr.landmark_type  = "decision"
                curr.landmark_reason = "offer confirmed by next USER turn [align-A]"
                curr.promoted       = True

            if not nxt.is_landmark:
                nxt.is_landmark    = True
                nxt.landmark_type  = "decision"
                nxt.landmark_reason = "weak confirmation of ASSISTANT offer [align-A]"
                nxt.promoted       = True

        # Pattern B
        if (curr.speaker == "USER"
                and nxt.speaker == "ASSISTANT"
                and _has_slot_signal(curr.text)
                and _is_assistant_echo(nxt.text)):

            if not curr.is_landmark:
                curr.is_landmark    = True
                curr.landmark_type  = "intent"
                curr.landmark_reason = "constraint echoed by next ASSISTANT turn [align-B]"
                curr.promoted       = True

            if not nxt.is_landmark:
                nxt.is_landmark    = True
                nxt.landmark_type  = "intent"
                nxt.landmark_reason = "echo confirms prior USER constraint [align-B]"
                nxt.promoted       = True


# ─── Public detector class ────────────────────────────────────────────────────

class RuleLandmarkDetector:
    """
    Two-pass rule-based landmark detector.

    Pass 1: Score each turn individually using text signals.
    Pass 2: Promote turns based on cross-turn alignment patterns.

    Operates on raw text and speaker role only.
    Slot annotations are never read here.
    """

    def detect(self, conversation: Conversation) -> Conversation:
        """
        Annotate all turns in-place and return the conversation.
        Safe to call multiple times — resets annotations before running.
        """
        # Reset any previous annotations
        for turn in conversation.turns:
            turn.is_landmark     = False
            turn.landmark_type   = None
            turn.landmark_reason = ""
            turn.promoted        = False

        # Pass 1: individual scoring
        for turn in conversation.turns:
            is_lm, lm_type, reason = _pass1(turn)
            turn.is_landmark     = is_lm
            turn.landmark_type   = lm_type
            turn.landmark_reason = reason

        # Pass 2: cross-turn alignment
        _pass2(conversation.turns)

        return conversation
