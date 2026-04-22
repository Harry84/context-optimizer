#!/usr/bin/env python3
"""
Visual verification of query classification and landmark detection logic.

DETECTION APPROACH: Two-pass cross-turn alignment.

Pass 1: Score each turn individually using slot-value signals and speaker role.
Pass 2: Promote turns based on cross-turn alignment:
  - ASSISTANT offers value → USER confirms = both promoted to landmark
  - USER states constraint → ASSISTANT echoes = both promoted to landmark

This is a behavioural/structural approach — works on any dataset without
annotations. Slot annotations are used ONLY for evaluation (GT recall).

Usage:
    python utilities/verify_classifiers.py              # 5 random conversations
    python utilities/verify_classifiers.py --longest    # 5 longest
    python utilities/verify_classifiers.py --index 3    # specific index
"""

import json
import re
import sys
import random
from dataclasses import dataclass, field

DATA_PATH = "data/taskmaster2/flights.json"

# ─── Query Classifier ─────────────────────────────────────────────────────────

FACTUAL_PATTERNS    = [r"\b(what|when|where|who|which|how much|how many|how long)\b",
                       r"\b(price|cost|fare|date|time|airport|airline|stop|layover|seat)\b"]
ANALYTICAL_PATTERNS = [r"\b(why|how|compare|explain|difference|reason|better|worse|pros|cons)\b"]
PREFERENCE_PATTERNS = [r"\b(recommend|suggest|best|prefer|which would|which should|what would you)\b"]

def classify_query(query: str) -> str:
    q = query.lower()
    if any(re.search(p, q) for p in ANALYTICAL_PATTERNS): return "analytical"
    if any(re.search(p, q) for p in PREFERENCE_PATTERNS): return "preference"
    if any(re.search(p, q) for p in FACTUAL_PATTERNS):    return "factual"
    return "analytical"

# ─── Slot-value signals (domain vocabulary) ───────────────────────────────────

SLOT_SIGNALS = [
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

def has_slot_signal(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in SLOT_SIGNALS)

def is_pure_filler(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 50 and not has_slot_signal(t):
        return bool(re.match(
            r"^(okay|ok|sure|alright|great|perfect|fine|yes|yeah|no|"
            r"hold on|wait|right|correct|got it|i see|uh|um|"
            r"could you (say|repeat)|i('m| am) sorry|"
            r"what (else|other)|anything else|is that all|"
            r"thank(s| you)|goodbye|bye|hello|hi|hey|"
            r"sounds good|that sounds good|that works)[\.\!\?,]?\s*$",
            t
        ))
    return False

# ─── Pass 1: Individual turn scoring ──────────────────────────────────────────

INTENT_VERB_PATTERNS = [
    r"\b(i'?d? ?(like|want|need|prefer)|i'?m (looking for|trying to find))\b",
    r"\b(looking for|searching for|need to (find|book|get)|can you (find|get|book))\b",
    r"\b(i want|i need|i'?d prefer|i'?d like)\b",
]

ASSISTANT_OFFER_PATTERNS = [
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
]

STRONG_CONFIRMATION = re.compile(
    r"\b(i'?ll (take|go with|choose|pick|book|do that)|"
    r"let'?s (go with|do|take|book|choose)|"
    r"book (it|that|both|them)|go ahead (and book|with that)|"
    r"i'?d like (to book|to go with|that one))\b",
    re.IGNORECASE
)

WEAK_CONFIRMATION = re.compile(
    r"^(yes|yeah|yep|okay|ok|sure|alright|confirmed|deal|done|great|perfect|"
    r"fine|correct|that'?s (right|correct|fine|good)|works for me)[\.\!,]?\s*$",
    re.IGNORECASE
)

ACTION_PATTERNS = [
    r"\b(i'?ll|i will|i'?m going to)\b.{0,50}\b(send|book|confirm|process|arrange|reserve|email)\b",
    r"\b(i have|i'?ve)\b.{0,40}\b(booked|confirmed|reserved|sent|processed|emailed)\b",
    r"\blet me (look that up|find|check|search|pull that up|look into)\b",
    r"\bi'?ll (look that up|find that|check that|search for)\b",
    r"\bokay,? let me send\b",
]

def is_assistant_offer(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in ASSISTANT_OFFER_PATTERNS)

# ASSISTANT echoing user constraints back ("You said X", "So you want X", "X, correct?")
ECHO_PATTERNS = [
    r"\b(you said|you mentioned|you want|you('d| would) like|you('re| are) looking for)\b",
    r"\bso (you want|you('d| would) like|you need|you('re| are) looking)\b",
    r"\b(confirm(ing)?|verify(ing)?).{0,30}(you want|you need|you('d| would) like)\b",
    r"(correct|right|is that right|is that correct)\??\s*$",
    r"\bjust to confirm\b",
]

def is_assistant_echo(text: str) -> bool:
    """ASSISTANT repeating back a user constraint for confirmation."""
    return (has_slot_signal(text) and
            any(re.search(p, text, re.IGNORECASE) for p in ECHO_PATTERNS))


@dataclass
class TurnResult:
    index: int
    speaker: str
    text: str
    is_landmark: bool = False
    landmark_type: str | None = None
    reason: str = ""
    pass1_landmark: bool = False    # result of pass 1 alone
    promoted: bool = False          # promoted by pass 2 alignment


def pass1_score(speaker: str, text: str) -> tuple[bool, str | None, str]:
    """Individual turn scoring — no cross-turn context."""
    if is_pure_filler(text):
        return False, None, ""

    text_l = text.lower().strip()

    if speaker == "USER":
        if STRONG_CONFIRMATION.search(text_l):
            return True, "decision", "strong confirmation"
        for p in INTENT_VERB_PATTERNS:
            if re.search(p, text_l, re.IGNORECASE):
                return True, "intent", "intent verb"
        if has_slot_signal(text):
            return True, "intent", "slot signal in user turn"

    if speaker == "ASSISTANT":
        if is_assistant_offer(text):
            return True, "decision", "assistant offer"
        if has_slot_signal(text):
            return True, "decision", "slot signal in assistant turn"
        for p in ACTION_PATTERNS:
            if re.search(p, text_l, re.IGNORECASE):
                return True, "action_item", "action pattern"

    return False, None, ""


# ─── Pass 2: Cross-turn alignment ─────────────────────────────────────────────

def pass2_align(results: list[TurnResult]) -> list[TurnResult]:
    """
    Promote turns based on cross-turn alignment signals.

    Pattern A — Offer → Confirmation:
      ASSISTANT[i] makes an offer AND USER[i+1] gives weak confirmation
      → both are landmarks (decision). The confirmation validates the offer;
        the offer is what the confirmation is accepting.

    Pattern B — User constraint → Assistant echo:
      USER[i] states something AND ASSISTANT[i+1] echoes it back
      → both are landmarks (intent confirmed). The echo validates that
        the user turn contained a real constraint.

    These patterns are dataset-agnostic — they rely on conversational
    structure, not annotations.
    """
    n = len(results)
    for i in range(n - 1):
        curr = results[i]
        nxt  = results[i + 1]

        # Pattern A: ASSISTANT offer → USER weak confirmation
        if (curr.speaker == "ASSISTANT"
                and nxt.speaker == "USER"
                and WEAK_CONFIRMATION.match(nxt.text.strip())
                and is_assistant_offer(curr.text)):

            if not curr.is_landmark:
                curr.is_landmark    = True
                curr.landmark_type  = "decision"
                curr.reason         = "offer confirmed by next USER turn [align-A]"
                curr.promoted       = True

            if not nxt.is_landmark:
                nxt.is_landmark   = True
                nxt.landmark_type = "decision"
                nxt.reason        = "weak confirmation of ASSISTANT offer [align-A]"
                nxt.promoted      = True

        # Pattern B: USER constraint → ASSISTANT echo
        if (curr.speaker == "USER"
                and nxt.speaker == "ASSISTANT"
                and has_slot_signal(curr.text)
                and is_assistant_echo(nxt.text)):

            if not curr.is_landmark:
                curr.is_landmark   = True
                curr.landmark_type = "intent"
                curr.reason        = "constraint echoed by next ASSISTANT turn [align-B]"
                curr.promoted      = True

            if not nxt.is_landmark:
                nxt.is_landmark   = True
                nxt.landmark_type = "intent"
                nxt.reason        = "echo confirms prior USER constraint [align-B]"
                nxt.promoted      = True

    return results


# ─── Full pipeline ────────────────────────────────────────────────────────────

def detect_landmarks(utterances: list[dict]) -> list[TurnResult]:
    """Run both passes and return annotated TurnResult list."""
    # Pass 1
    results = []
    for i, utt in enumerate(utterances):
        speaker = utt.get("speaker", "")
        text    = utt.get("text", "").strip()
        is_lm, lm_type, reason = pass1_score(speaker, text)
        r = TurnResult(
            index=i, speaker=speaker, text=text,
            is_landmark=is_lm, landmark_type=lm_type, reason=reason,
            pass1_landmark=is_lm,
        )
        results.append(r)

    # Pass 2
    results = pass2_align(results)
    return results


# ─── Evaluation helper ────────────────────────────────────────────────────────

def slot_annotated(utterance: dict) -> bool:
    return any(
        ann.get("name", "")
        for seg in utterance.get("segments", [])
        for ann in seg.get("annotations", [])
    )

# ─── Display ─────────────────────────────────────────────────────────────────

LANDMARK_ICONS = {"intent": "🎯", "decision": "✅", "action_item": "📋"}

def display_conversation(conv: dict, index: int):
    utterances = conv.get("utterances", [])
    gt_set = {i for i, u in enumerate(utterances) if slot_annotated(u)}
    results = detect_landmarks(utterances)

    detected_set = {r.index for r in results if r.is_landmark}
    promoted_set = {r.index for r in results if r.promoted}
    recall = len(gt_set & detected_set) / len(gt_set) if gt_set else 1.0

    lm_counts = {"intent": 0, "decision": 0, "action_item": 0}
    compressible = 0

    print(f"\n{'='*70}")
    print(f"Conv {index+1}: {conv.get('conversation_id')}")
    print(f"Instruction: {conv.get('instruction_id')} | Turns: {len(utterances)} | GT: {len(gt_set)}")
    print(f"{'='*70}")

    for r in results:
        gt_marker  = " [GT]"       if r.index in gt_set      else ""
        promo_mark = " [promoted]" if r.promoted              else ""
        label      = "USER     "   if r.speaker == "USER"     else "ASSISTANT"

        if r.is_landmark:
            lm_counts[r.landmark_type] += 1
            icon = LANDMARK_ICONS[r.landmark_type]
            print(f"\n  {icon} [{r.landmark_type.upper():11}] {label} | {r.text[:85]}{gt_marker}{promo_mark}")
            print(f"       reason: {r.reason}")
        else:
            compressible += 1
            print(f"  💨 {label} | {r.text[:80]}{gt_marker}")

    total_lm = sum(lm_counts.values())
    print(f"\n  📊 Landmarks: {total_lm} {lm_counts} | Promoted by alignment: {len(promoted_set)}")
    print(f"     Compressible: {compressible} ({100*compressible/len(utterances):.0f}%) | GT recall: {recall:.0%}")

    print(f"\n  🔍 Query classification:")
    for q in [
        "What was the final flight chosen?",
        "Why did they choose Paris over Budapest?",
        "Which flight would you recommend based on price?",
        "What time does the flight depart?",
        "How do the two options compare on layovers?",
    ]:
        print(f"     [{classify_query(q):>10}] {q}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_convs = json.load(f)

    viable = [c for c in all_convs if len(c.get("utterances", [])) >= 20]
    print(f"Loaded {len(viable)} conversations with ≥20 turns")

    if "--longest" in sys.argv:
        sample = sorted(viable, key=lambda c: len(c["utterances"]), reverse=True)[:5]
    elif "--index" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--index") + 1])
        sample = [viable[idx]]
    else:
        random.seed(42)
        sample = random.sample(viable, min(5, len(viable)))

    for i, conv in enumerate(sample):
        display_conversation(conv, i)

    # Corpus-wide stats
    print(f"\n\n{'='*70}")
    print("OVERALL STATS (two-pass cross-turn alignment)")
    print(f"{'='*70}")
    total = lm_total = intent_t = decision_t = action_t = promoted_t = 0
    gt_total = gt_detected = 0

    for conv in viable:
        utterances = conv.get("utterances", [])
        results    = detect_landmarks(utterances)
        gt_set     = {i for i, u in enumerate(utterances) if slot_annotated(u)}
        det_set    = {r.index for r in results if r.is_landmark}

        for r in results:
            total += 1
            if r.index in gt_set: gt_total += 1
            if r.is_landmark:
                lm_total += 1
                if r.landmark_type == "intent":      intent_t   += 1
                if r.landmark_type == "decision":    decision_t += 1
                if r.landmark_type == "action_item": action_t   += 1
                if r.promoted:                       promoted_t += 1
            if r.index in gt_set and r.index in det_set:
                gt_detected += 1

    recall = gt_detected / gt_total if gt_total else 0
    print(f"Total turns:          {total:,}")
    print(f"Detected landmarks:   {lm_total:,} ({100*lm_total/total:.1f}%)")
    print(f"  Stated intents:     {intent_t:,}")
    print(f"  Decisions:          {decision_t:,}")
    print(f"  Action items:       {action_t:,}")
    print(f"  Promoted (pass 2):  {promoted_t:,}")
    print(f"Compressible:         {total-lm_total:,} ({100*(total-lm_total)/total:.1f}%)")
    print(f"\nGround truth recall:  {recall:.1%}  ({gt_detected:,}/{gt_total:,})")
    print(f"(slot-annotated GT; action items have no GT annotations)")


if __name__ == "__main__":
    main()
