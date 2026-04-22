"""
v1 Landmark Detector — Rule-based + Two-Pass Cross-Turn Alignment.

Detection operates on RAW TEXT AND SPEAKER ROLE ONLY.
Slot annotations are never used here — they are evaluation ground truth only.
"""

from __future__ import annotations

import re

from src.ingestion.models import Conversation, Turn

# ─── Slot-value signals ──────────────────────────────────────────────────────

SLOT_SIGNALS: list[str] = [
    r"\b\d{1,2}:\d{2}\s*(a\.?m\.?|p\.?m\.?)\b",
    r"\$\s*\d+",
    r"\b(under|less than|up to|around|about|max|maximum|budget|not more than)\b.{0,15}\$?\d",
    r"\b(delta|united|american airlines?|southwest|jetblue|british airways|"
    r"lufthansa|frontier|virgin|alaska|air canada|klm|air new zealand|"
    r"spirit|hawaiian|allegiant)\b",
    r"\b(business|economy|first|coach|premium economy)\b",
    r"\b(non.?stop|nonstop|direct|layover|stopover|one stop|two stops?|no stops?)\b",
    r"\b(one.?way|round.?trip|multi.?city)\b",
    r"\b(fly(ing)?|flight|travel(l?ing)?|going|heading|depart(ing)?|leav(e|ing))\b.{0,40}\b(to|from)\b",
    r"\bfrom\b.{2,40}\bto\b",
    r"\b(morning|afternoon|evening|early|late|overnight|red.?eye)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d+(st|nd|rd|th)\b",
    r"\b(today|tomorrow|next week|this weekend|next month)\b",
    r"\b(aisle|window|middle)\b.{0,10}(seat)?\b",
    r"\b(wi.?fi|meal|food|alcohol|entertainment|baggage|luggage)\b",
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
    return len(text.strip()) < 50 and not _has_slot_signal(text) and bool(_FILLER_RE.match(text.strip()))


# ─── Conversation close patterns (USER signalling end of conversation) ────────

_CONVERSATION_CLOSE = re.compile(
    r"\b("
    # Explicit completion signals
    r"that('s| is| will be)? all|"
    r"that('s| is)? everything|"
    r"that (does it|covers it)|"
    r"i('m| am) (done|finished|all set|good|set)|"
    r"we('re| are) (done|finished|all set)|"
    r"i think (that('s| is) (it|all|everything))|"
    r"i('ve| have) (got|gotten) (what|everything) i need(ed)?|"
    # Declining further help
    r"i (don't|do not) need (anything|nothing) else|"
    r"nothing else|"
    r"no (more|other) questions?|"
    r"no (thanks|thank you),? ?(that'?s? (all|it|everything|good|fine))?|"
    r"no,? i('m| am) (good|fine|all set|okay|ok)|"
    # Gratitude as closing signal
    r"thanks? (for (your |the )?(help|assistance|time|everything))|"
    r"thank you (for (your |the )?(help|assistance|time|everything))|"
    r"(really |very |much )?appreciated|"
    # Farewell signals
    r"(good)?bye|"
    r"have a (good|great|nice) (day|one|evening|afternoon)|"
    r"talk (to you |with you )?(soon|later)|"
    r"(take care|cheers|so long)|"
    # Casual wrap-ups
    r"that'?s? (it|all|everything|good|fine|perfect|great)|"
    r"i('m| am) (all set|good to go)|"
    r"sounds (good|great|perfect|fine)|"
    r"perfect,? (thanks?|thank you)"
    r")\b",
    re.IGNORECASE,
)


def _is_conversation_close(text: str) -> bool:
    """
    Returns True if a USER turn signals the conversation is ending.
    These are landmarks because they tell the LLM the conversation goal
    was met (or abandoned) and the state is resolved.
    """
    return bool(_CONVERSATION_CLOSE.search(text.strip()))


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

# ─── Echo patterns ───────────────────────────────────────────────────────────

_ECHO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(you said|you mentioned|you want|you('d| would) like|you('re| are) looking for)\b",
    r"\bso (you want|you('d| would) like|you need|you('re| are) looking)\b",
    r"(correct|right|is that right|is that correct)\??\s*$",
    r"\bjust to confirm\b",
]]


def _is_assistant_echo(text: str) -> bool:
    return _has_slot_signal(text) and any(p.search(text) for p in _ECHO_PATTERNS)


# ─── Action item patterns ─────────────────────────────────────────────────────

_ACTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(i'?ll|i will|i'?m going to)\b.{0,50}\b(send|book|confirm|process|arrange|reserve|email)\b",
    r"\b(i have|i'?ve)\b.{0,40}\b(booked|confirmed|reserved|sent|processed|emailed)\b",
    r"\blet me (look that up|find|check|search|pull that up|look into)\b",
    r"\bi'?ll (look that up|find that|check that|search for)\b",
    r"\bokay,? let me send\b",
]]


# ─── Pass 1: Individual turn scoring ─────────────────────────────────────────

def _pass1(turn: Turn) -> tuple[bool, str | None, str]:
    text   = turn.text
    text_l = text.lower().strip()

    if _is_pure_filler(text):
        return False, None, ""

    if turn.speaker == "USER":
        # Strong confirmation → decision
        if _STRONG_CONFIRMATION.search(text_l):
            return True, "decision", "strong confirmation"

        # Conversation close signal → decision
        if _is_conversation_close(text):
            return True, "decision", "conversation close"

        # Explicit intent verb → stated intent
        for p in _INTENT_VERB_PATTERNS:
            if p.search(text_l):
                return True, "intent", "intent verb"

        # Contains slot-value signal → stated intent
        if _has_slot_signal(text):
            return True, "intent", "slot signal in user turn"

    if turn.speaker == "ASSISTANT":
        if _is_assistant_offer(text):
            return True, "decision", "assistant offer"

        if _has_slot_signal(text):
            return True, "decision", "slot signal in assistant turn"

        for p in _ACTION_PATTERNS:
            if p.search(text_l):
                return True, "action_item", "action pattern"

    return False, None, ""


# ─── Pass 2: Cross-turn alignment ────────────────────────────────────────────

def _pass2(turns: list[Turn]) -> None:
    n = len(turns)
    for i in range(n - 1):
        curr = turns[i]
        nxt  = turns[i + 1]

        # Pattern A — Offer → Confirmation
        if (curr.speaker == "ASSISTANT"
                and nxt.speaker == "USER"
                and _WEAK_CONFIRMATION.match(nxt.text.strip())
                and _is_assistant_offer(curr.text)):

            if not curr.is_landmark:
                curr.is_landmark     = True
                curr.landmark_type   = "decision"
                curr.landmark_reason = "offer confirmed by next USER turn [align-A]"
                curr.promoted        = True

            if not nxt.is_landmark:
                nxt.is_landmark     = True
                nxt.landmark_type   = "decision"
                nxt.landmark_reason = "weak confirmation of ASSISTANT offer [align-A]"
                nxt.promoted        = True

        # Pattern B — Constraint → Echo
        if (curr.speaker == "USER"
                and nxt.speaker == "ASSISTANT"
                and _has_slot_signal(curr.text)
                and _is_assistant_echo(nxt.text)):

            if not curr.is_landmark:
                curr.is_landmark     = True
                curr.landmark_type   = "intent"
                curr.landmark_reason = "constraint echoed by next ASSISTANT turn [align-B]"
                curr.promoted        = True

            if not nxt.is_landmark:
                nxt.is_landmark     = True
                nxt.landmark_type   = "intent"
                nxt.landmark_reason = "echo confirms prior USER constraint [align-B]"
                nxt.promoted        = True


# ─── Public detector class ────────────────────────────────────────────────────

class RuleLandmarkDetector:
    """Two-pass rule-based landmark detector. Operates on raw text and speaker role only."""

    def detect(self, conversation: Conversation) -> Conversation:
        for turn in conversation.turns:
            turn.is_landmark     = False
            turn.landmark_type   = None
            turn.landmark_reason = ""
            turn.promoted        = False

        for turn in conversation.turns:
            is_lm, lm_type, reason = _pass1(turn)
            turn.is_landmark     = is_lm
            turn.landmark_type   = lm_type
            turn.landmark_reason = reason

        _pass2(conversation.turns)

        return conversation
