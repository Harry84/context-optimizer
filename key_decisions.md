# Key Decisions Log
## Context Optimizer — Running Record of Architectural & Design Choices

> **Purpose:** Records significant decisions made during the project with justification at the time of making them. Feeds directly into the Architecture Decisions Document, written report, and interview follow-up defence.

---

## KD-001 — Dataset Choice: Taskmaster-2, Flights Domain

**Decision:** Use Taskmaster-2 flights domain as the primary evaluation corpus.

**Justification:**
- **Full conversations preserved** — alternating USER/ASSISTANT turns intact with slot annotations per utterance. Unlike pre-processed derivatives (e.g. dialog2flow), the raw dataset retains conversation structure required for context optimisation.
- **Structurally analogous to LEC use case** — flight booking conversations exhibit the same dialogue patterns as trade logistics enquiries: user establishes a goal, states constraints across multiple turns, compares 2–3 options on price/time/routing, shifts requirements, and confirms a final choice. The domain vocabulary differs but the flow is identical.
- **Rich slot annotations as ground truth** — every utterance has segment-level slot annotations (`flight_search.destination1`, `flight1_detail.fare`, `flight1_detail.seating_class`, etc.). These serve as verifiable ground truth for landmark detection — we can measure whether our detector finds the right turns, not just assume it does.
- **Natural compressible noise** — inspection of real conversations reveals genuine filler (repeated assistant utterances, crowdworker instruction-reading, confirmatory one-word turns) that should be compressed. This is exactly the compression opportunity the system is designed to exploit.
- **Scale:** 1,692 conversations ≥20 turns in the flights domain alone. Mean 30.2 turns, max 85 turns.
- **Licence:** Creative Commons CC BY 4.0 — permissive, suitable for research.
- **Free, no API cost** — downloaded directly from `google-research-datasets/taskmaster2` on HuggingFace.

**Datasets evaluated and rejected:**
- **OASST1** — general-purpose Q&A. No transactional flow, no slots, no ground-truth landmarks.
- **Bitext customer support** — single-turn pairs, not conversations.
- **MultiWOZ** — average 8–15 turns, too short.
- **dialog2flow dataset** — pre-processed for embedding training; conversation structure stripped out.
- **Taskmaster-2 hotels/restaurant-search** — valid, deferred. Available for extended evaluation.
- **Real LEC transcripts** — not available.

---

## KD-002 — Turn Threshold: ≥20 Turns

**Decision:** Filter corpus to conversations with ≥20 turns. The assignment's "50+ messages" threshold is not applied literally.

**Justification:**
- Taskmaster-2 has a mean of 20.2 turns and median of 17. Only 1.5% of conversations reach 50 turns — applying that threshold literally would leave an unrepresentatively thin corpus.
- Task-oriented dialogues are informationally denser than casual chat. A 20-turn flight booking conversation contains as much load-bearing context as 40+ turns of general conversation.
- At ≥20 turns, 1,692 conversations are available in the flights domain — sufficient for a representative evaluation set.

---

## KD-003 — No Tree Reconstruction Required

**Decision:** Taskmaster-2 requires no tree reconstruction. Conversations are pre-structured as linear sequences.

**Justification:**
- Unlike OASST1 (flat node list requiring parent→child tree reconstruction), Taskmaster-2 stores each conversation as a self-contained JSON object with an `utterances` array in chronological order.

---

## KD-004 — Embedding Model: Local all-MiniLM-L6-v2

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` locally for all semantic similarity. CPU-only — no CUDA required.

**Justification:**
- Zero API cost — called per message across all conversations
- Deterministic — same weights, same output, fully reproducible
- Fast on CPU — ~80MB, viable on a developer laptop; inference is not the bottleneck (OpenAI API calls are)
- Competitive on semantic similarity benchmarks; sufficient for retrieval-style scoring
- CPU-only torch from PyPI installs cleanly on any machine — GPU acceleration is not needed for this use case and would add deployment complexity for assessors

**Alternatives considered:**
- `text-embedding-3-small` (OpenAI): rejected — per-call cost accumulates
- `all-mpnet-base-v2`: 4× slower, overkill
- TF-IDF only: rejected — misses semantic similarity

---

## KD-005 — LLM-as-Judge: Rubric Design & Bias Mitigation

**Decision:** 4-dimension rubric (correctness, completeness, landmark consistency, hallucination), temperature=0, independent per-answer evaluation, plus BERTScore F1 as an independent objective metric.

**Rubric dimensions:**
- **Correctness** — are factual claims accurate relative to the conversation?
- **Completeness** — does the answer address all parts of the query?
- **Landmark consistency** — does the answer respect stated intents, decisions, and action items from the conversation?
- **Hallucination** — does the answer introduce information not grounded in context? (10 = none)

**Bias mitigations:**
- Each answer evaluated independently against its own context (not side-by-side) — eliminates positional bias
- Temperature=0 — same inputs produce same scores every run
- Different judge model from generation model — reduces self-serving bias
- BERTScore F1 (roberta-large, local) — LLM-independent semantic preservation metric; threshold ≥ 0.85

---

## KD-006 — Hard Mode Features are Must Have

**Decision:** Adaptive compression strategy and landmark detection (stated intents, decisions, action items) are Must Have, not stretch goals.

**Justification:**
- Assignment explicitly labels these as "hard mode signals we'll look for"
- Adaptive strategy makes compression thresholds principled rather than arbitrary
- Landmark detection prevents the optimizer compressing away the most critical turns
- Taskmaster-2 slot annotations provide ground truth for measuring landmark detection accuracy

---

## KD-007 — Landmark Framing: Assignment Terminology Adopted

**Decision:** Use the assignment's exact framing — **stated intents, decisions, action items** — as the three landmark categories.

**Justification:**
- The assignment brief explicitly names these three categories as "hard mode signals." Using their language signals the brief was read carefully and addressed directly.
- All three are naturally present in Taskmaster-2 flights conversations with verifiable ground truth via slot annotations (except action items, which have no annotation — limitation documented).

---

## KD-008 — Phased Approach: Taskmaster-2 Flights First, LEC Domain Second

**Decision:** Phase 1 proves the optimizer on Taskmaster-2 flights. Phase 2 adapts to LEC with synthetic customer service conversations. Synthetic generation deferred.

**Justification:**
- Taskmaster-2 flights is structurally analogous to LEC logistics — same optimizer logic applies; only domain vocabulary and landmark taxonomy differ
- Pipeline is dataset-agnostic — Phase 2 requires new data and updated landmark labels, not architectural changes

---

## KD-009 — Landmark Detection: Slot Annotations as Evaluation Only, Not Detection

**Decision:** The landmark detector operates on raw text and speaker role only. Slot annotations are used exclusively as independent ground truth for measuring recall.

**Justification:**
- Using slot annotations for detection would be circular: 100% recall on Taskmaster-2, 0% on any unannotated dataset including real LEC conversations.
- Text-based detection generalises to any conversational dataset without pre-annotation.
- Validated empirically: text-based detection with cross-turn alignment achieved 86.6% recall while remaining generalisable.

**Limitation acknowledged:** Action items have no slot annotations in Taskmaster-2. Their detection recall cannot be measured against ground truth and is reported separately as pattern-match coverage only.

---

## KD-010 — Two-Pass Cross-Turn Alignment for Landmark Detection

**Decision:** Two-pass approach: Pass 1 scores each turn individually; Pass 2 promotes turns based on cross-turn patterns.

**Patterns:**
- **Pattern A (Offer→Confirmation):** ASSISTANT[i] makes offer AND USER[i+1] gives weak confirmation → both promoted to `decision`
- **Pattern B (Constraint→Echo):** USER[i] has slot signal AND ASSISTANT[i+1] echoes it → both promoted to `intent`

**Measured results (1,692 conversations, ≥20 turns):**
- GT recall: **86.6%**
- Landmark rate: **46.4%** of all turns
- Compressible: **53.6%** of all turns
- Turns promoted by pass 2: 879 (3.7% of all landmarks)

---

## KD-011 — Landmark Detection: Why Not LLM or Embeddings?

**Decision:** Rule-based with cross-turn alignment as v1. LLM and embedding approaches documented as explicit v2 upgrade paths.

| Approach | Recall | Cost/conv | Latency | Generalisable |
|---|---|---|---|---|
| Rules + two-pass (v1) | 86.6% measured | Free | <5ms | With new patterns |
| Embedding similarity (v2) | ~80–88% est. | Free | ~50ms | With new prototypes |
| LLM per conversation | ~92% est. | ~$0.01–0.03 | ~5s | Yes |

**Why not LLM:** Net cost problem — detection cost eats into compression savings. At rule-based detection, break-even is ≥2 downstream calls per context. With LLM detection, ≥3–5.

**Why not embeddings in v1:** Recall is an estimate until measured. Rule-based 86.6% is a known quantity. Embedding v2 is the highest-priority next upgrade.

---

## KD-012 — Data Quality: Crowdworker Noise Handling

**Decision:** Apply two data-quality fixes at load time rather than filtering affected conversations.

**Problem observed:** Taskmaster-2 crowdworker transcripts contain two types of noise:
1. **Intra-turn sentence repetition** — identical sentence appearing twice within one utterance (e.g. "There is a 7 AM flight. There is a 7 AM flight. Both cost $1500.")
2. **Cross-turn near-duplication** — consecutive ASSISTANT turns where turn N is a strict substring of turn N+1 (crowdworker re-stated then completed)

**Fixes:**
1. `_dedup_sentences()` in `loader.py` — removes consecutively repeated sentences within a single turn at load time using sentence-boundary splitting
2. `_smart_merge()` in `assembler.py` — when merging consecutive ASSISTANT turns, detects substring containment in both directions:
   - New content substring of existing → skip (already have it)
   - Existing content substring of new → replace with new (new is a superset)
   - 80%+ word overlap → skip (near-duplicate)
   - Otherwise → append

**Justification:** Filtering conversations with noise would remove ~15% of the corpus unnecessarily. The noise is a known Taskmaster-2 data quality issue, not a signal. Fixing at load time is transparent and auditable.

**Limitation:** These fixes handle structural duplication. They do not fix semantic repetition where the crowdworker rephrased the same information differently — this is less common and harder to detect without semantic comparison.

---

## KD-013 — Conversation Close Signals as Landmarks

**Decision:** USER turns signalling conversation end are classified as `decision` landmarks and always kept verbatim.

**Motivation:** During manual inspection of the dlg-cbfc519d conversation, `"Okay. That will be all."` and `"Oh, I'm done."` were being compressed — they scored low because they contain no slot-value signals. But they are load-bearing context: they tell the LLM that the user's goal was met (or abandoned) and the conversation state is resolved. Without them, a query like "Why did the user stop comparing options?" has no anchor.

**Signals detected (in `rule_detector.py`):**
- Explicit completion: "that's all", "that will be all", "I'm done", "I'm finished", "that does it"
- Declining further help: "nothing else", "no more questions", "no thanks"
- Gratitude as closing: "thanks for your help", "thank you for your assistance", "appreciated"
- Farewells: "goodbye", "bye", "have a good day", "take care", "talk soon"
- Casual wrap-ups: "that's it", "sounds good", "I'm all set", "good to go"

**Classification:** `decision` (goal state resolved, not just acknowledged).

**Trade-off acknowledged:** Some of these signals (e.g. "sounds good") may fire mid-conversation in non-closing contexts, creating false positives. The cost is conservative — keeping a turn that could have been compressed. This is the preferred failure mode for a context optimizer.

---

## KD-014 — Summariser Minimum Threshold

**Decision:** COMPRESS runs shorter than 80 characters total are dropped entirely rather than summarised.

**Motivation:** During inspect runs, single-turn filler runs ("Oh, sure thing." = 16 chars) were producing LLM summaries longer than the original content — defeating the purpose of compression. The summariser prompt now includes a SKIP instruction for trivial content, and a 80-char minimum prevents the API call entirely for pure filler.

**Effect:** Fewer LLM calls per conversation (reducing cost and latency), cleaner output (no [SUMMARY: Oh, sure thing.] placeholders), and tighter token budgets.

---

## KD-015 — Assembly Integrity: Merge vs. Bridge for Consecutive Same-Role Turns

**Decision:** Consecutive ASSISTANT turns are merged (smart merge); consecutive USER turns get a `[context continues]` assistant bridge rather than being merged.

**Justification:**
- **ASSISTANT merges:** Multiple consecutive ASSISTANT turns arise when a SUMMARY turn is followed by a verbatim ASSISTANT turn (or vice versa). Merging them into one ASSISTANT turn is semantically correct — the LLM sees one coherent context block. Smart merge prevents duplication.
- **USER bridges:** Consecutive USER turns arise when the raw data has multiple short user utterances (e.g. "February 14th." / "February 15th." on separate turns). Merging them into one USER turn would misrepresent the conversational structure and lose the question-answer pairing. A thin bridge `[context continues]` preserves the USER turns as distinct while maintaining valid role alternation.

---

## KD-016 — Turn-Level vs. Sentence-Level Compression

**Decision:** v1 implements turn-level compression. Sentence-level compression is documented as the highest-priority v2 upgrade.

**Problem discovered during evaluation:** The landmark detector correctly flags a turn as load-bearing when any sentence within it contains a slot signal or intent verb. The entire turn then survives verbatim — including sentences that are pure filler. For example:

> *"An onboard bar! That sounds fun. Okay I'm going to go with Virgin I think. Although — hold on — what about the American Airlines one?"*

This turn contains a booking decision ("go with Virgin") so is correctly flagged as a landmark. But the first sentence ("An onboard bar! That sounds fun.") is pure filler that adds noise to the context. Under turn-level compression, both survive together.

**Measured impact:** Dry-run audits on both Taskmaster-2 and realistic synthetic conversations show that COMPRESS runs hold only ~11–33% of total tokens. Even compressing all non-landmark turns to zero, the maximum achievable token reduction is ~25–33% on these datasets. The 40–60% target assumes more compressible noise than either dataset contains at the turn level.

**Why turn-level was correct for v1:**
- Simpler to reason about and test — the unit of compression is the same unit as the conversation
- Safe by default — no risk of splitting a landmark sentence from its context mid-turn
- Sufficient to demonstrate the architecture and prove quality preservation

**The v2 fix — sentence-level compression:**
1. At load time (or at scoring time), split each turn into constituent sentences using sentence-boundary detection
2. Score each sentence independently against the query using the same keyword + semantic + recency composite
3. Classify sentences as KEEP or COMPRESS rather than whole turns
4. Landmark sentences always KEEP; non-landmark sentences in the same turn can be compressed
5. Reassemble kept sentences into the turn before assembling the thread

**Expected impact of sentence-level compression:** Token reduction of 40–55% on realistic conversational data, with the same quality preservation guarantees. The landmark detection logic is unchanged — it still operates at turn level to identify which turns contain load-bearing content, but the compressor now has sub-turn granularity to discard the surrounding filler.

**Estimated implementation effort:** 2–3 days. The scorer, assembler, and pipeline interfaces require minor extension; no architectural changes.

---

*Last updated: April 2026*
