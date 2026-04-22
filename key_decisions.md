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

**Example conversation (dlg-cbfc519d, 85 turns):**
- Stated intent: "I need a flight to Europe" — load-bearing, must be preserved
- Constraint-setting across turns: dates, time of day, price range, seat class — all slot-annotated ground truth
- Comparison: Paris 6AM $2880 economy 1 stop vs Budapest 7AM $1500 2 stops — decision point
- Mid-conversation noise: user reads crowdworker instructions aloud — genuine compressible filler
- Decision: "6AM flight, Paris" — must be preserved
- Action item: "Let me send you the flight details" — assistant commitment, must be preserved
- This single conversation contains all landmark categories the assignment specifies: stated intents, decisions, action items — plus genuine noise to compress.

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
- Task-oriented dialogues are informationally denser than casual chat. A 20-turn flight booking conversation — establishing origin, destination, dates, price, seat class, comparing options — contains as much load-bearing context as 40+ turns of general conversation. The 50-turn figure in the assignment was written with chat-style data in mind.
- At ≥20 turns, 1,692 conversations are available in the flights domain — sufficient for a representative evaluation set.
- Documented honestly in the report and defended on information density grounds, not convenience.

**Alternatives considered:**
- 50-turn threshold literally: rejected — 195 conversations total across all domains, unrepresentative
- Chaining conversations to inflate length: rejected — destroys coherence, undermines evaluation

---

## KD-003 — No Tree Reconstruction Required

**Decision:** Taskmaster-2 requires no tree reconstruction. Conversations are pre-structured as linear sequences.

**Justification:**
- Unlike OASST1 (flat node list requiring parent→child tree reconstruction), Taskmaster-2 stores each conversation as a self-contained JSON object with an `utterances` array in chronological order.
- Preprocessing is: load per-domain JSON, filter by turn count, normalise to internal `Conversation` format.

---

## KD-004 — Embedding Model: Local all-MiniLM-L6-v2

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` locally for all semantic similarity.

**Justification:**
- Zero API cost — called per message across all conversations
- Deterministic — same weights, same output, fully reproducible
- Fast on CPU — ~80MB, viable on a developer laptop
- Competitive on semantic similarity benchmarks; sufficient for retrieval-style scoring

**Alternatives considered:**
- `text-embedding-3-small` (OpenAI): rejected — per-call cost accumulates
- `all-mpnet-base-v2`: 4× slower, overkill
- TF-IDF only: rejected — misses semantic similarity

---

## KD-005 — LLM-as-Judge: Rubric Design & Bias Mitigation

**Decision:** 4-dimension rubric (correctness, completeness, landmark consistency, hallucination), temperature=0, order-swap bias mitigation, plus BERTScore F1 as an independent objective metric.

**Rubric dimensions:**
- **Correctness** — are factual claims accurate relative to the conversation?
- **Completeness** — does the answer address all parts of the query?
- **Landmark consistency** — does the answer respect stated intents, decisions, and action items from the conversation?
- **Hallucination** — does the answer introduce information not grounded in context? (10 = none)

**Bias mitigations:**
- **Order-swap:** Each pair evaluated twice with order swapped and scores averaged — cancels positional bias
- **Temperature=0:** Same inputs produce same scores every run
- **Different judge model:** Reduces self-serving bias
- **BERTScore F1:** LLM-independent semantic preservation metric; threshold F1 ≥ 0.85

---

## KD-006 — Hard Mode Features are Must Have

**Decision:** Adaptive compression strategy and landmark detection (stated intents, decisions, action items) are Must Have, not stretch goals.

**Justification:**
- Assignment explicitly labels these as "hard mode signals we'll look for"
- Adaptive strategy makes compression thresholds principled rather than arbitrary
- Landmark detection prevents the optimizer compressing away the most critical turns
- Taskmaster-2 slot annotations provide ground truth for measuring landmark detection accuracy — we can report recall/precision, not just assert it works

---

## KD-007 — Landmark Framing: Assignment Terminology Adopted

**Decision:** Use the assignment's exact framing — **stated intents, decisions, action items** — as the three landmark categories. Do not rename or reframe.

**Justification:**
- The assignment brief explicitly names these three categories as "hard mode signals." Using their language signals the brief was read carefully and addressed directly.
- All three are naturally present in Taskmaster-2 flights conversations:
  - **Stated intents** — "I need a flight to Europe", "early morning preferred" — slot-annotated turns, verifiable ground truth
  - **Decisions** — "Paris, 6AM flight", "economy class" — confirmation turns, identifiable by speaker pattern and slot annotations
  - **Action items** — "Let me send you the flight details", "I'll book that for you" — assistant commitment turns, detectable via pattern matching on assistant utterances
- Slot annotations in Taskmaster-2 cover stated intents and decisions with ground truth. Action items require pattern-based detection on assistant turns — a known limitation noted in the report.

---

## KD-008 — Phased Approach: Taskmaster-2 Flights First, LEC Domain Second

**Decision:** Phase 1 proves the optimizer on Taskmaster-2 flights. Phase 2 adapts to LEC with synthetic customer service conversations. Synthetic generation deferred.

**Justification:**
- Taskmaster-2 flights is structurally analogous to LEC logistics — same optimizer logic applies; only domain vocabulary and landmark taxonomy differ
- Pipeline is dataset-agnostic — Phase 2 requires new data and updated landmark labels, not architectural changes
- Synthetic generation at volume requires LLM cost unjustified at this stage
- Hotels and restaurant-search remain available for extended evaluation with no additional downloads

---

## KD-009 — Landmark Detection: Slot Annotations as Evaluation Only, Not Detection

**Decision:** The landmark detector operates on raw text and speaker role only. Slot annotations are used exclusively as independent ground truth for measuring recall — they play no role in detection logic.

**Justification:**
- Using slot annotations for detection would be circular: the system would perform perfectly on Taskmaster-2 but fail entirely on any unannotated dataset, including real LEC conversations.
- The detector must generalise. Text-based detection — slot-value signals (prices, times, airlines, seat classes), intent verb patterns, confirmation patterns, and cross-turn alignment — works on any conversational dataset without pre-annotation.
- Slot annotations then serve their correct role: an independent ground truth that lets us measure how well the text-based detector performs. We can report recall honestly.
- This separation was validated empirically: initial slot-based detection scored 100% recall trivially; text-based detection with cross-turn alignment achieved 86.6% recall while remaining generalisable.

**Limitation acknowledged:** Action items (ASSISTANT commitment turns) have no slot annotations in Taskmaster-2. Their detection recall cannot be measured against ground truth. This is noted in the evaluation report; action item recall is reported separately as pattern-match coverage, not GT recall.

---

## KD-010 — Two-Pass Cross-Turn Alignment for Landmark Detection

**Decision:** Landmark detection uses a two-pass approach. Pass 1 scores each turn individually using text signals. Pass 2 promotes turns based on cross-turn alignment patterns.

**The two alignment patterns:**

**Pattern A — Offer → Confirmation:**
If ASSISTANT[i] presents a concrete option (time, price, airline, flight details) AND USER[i+1] gives a weak confirmation ("yes", "okay", "sure"), both turns are promoted to landmark (decision). The confirmation validates the offer; without it the offer might just be information. Together they constitute a decision point.

**Pattern B — User constraint → Assistant echo:**
If USER[i] contains a slot-value signal AND ASSISTANT[i+1] echoes it back ("You said you want X, is that correct?", "So you're flying from X to Y?"), both turns are promoted to landmark (intent confirmed). The echo validates that the user turn contained a real constraint rather than a question or filler.

**Justification:**
- Single-turn heuristics have an inherent limitation: a bare "yes" is noise in isolation but a decision when it follows a flight offer. Cross-turn context resolves this ambiguity reliably.
- This approach is dataset-agnostic — it relies on conversational structure (offer-acceptance, constraint-echo) rather than domain vocabulary. The same patterns apply to LEC customer service conversations and any other task-oriented dialogue.
- It directly addresses the most common false negative category identified in diagnostic analysis: ASSISTANT offer turns and USER weak-confirmation turns that were missed by pass 1 alone.
- The promotion step is transparent and auditable — each promoted turn records its reason, so the system's decisions can be inspected and debugged.

**Measured results (across 1,692 flights conversations, ≥20 turns):**
- Ground truth recall: **86.6%** (20,092 / 23,190 slot-annotated turns detected)
- Landmark rate: **46.4%** of all turns (23,675 / 51,066)
- Compressible: **53.6%** of all turns — within the 40-60% target range
- Turns promoted by pass 2: 879 (3.7% of all landmarks)

**Known false positive pattern:** ASSISTANT clarifying questions containing domain vocabulary (e.g. "Are you flying round trip or one way?") are occasionally marked as decisions because they contain slot signals. These are compressible turns being incorrectly kept. This is a conservative error — keeping too much rather than dropping landmarks — and is the preferred failure mode for a context optimizer. Noted as a future improvement: add a question filter that exempts short clarifying ASSISTANT questions from the slot-signal rule.

**Alternatives considered:**
- Single-pass with lookahead context: rejected — requires passing future turns into the scoring function, complicating the interface
- LLM-based alignment classification: rejected for v1 — see KD-011
- Slot annotation as primary signal with text as fallback: rejected — see KD-009; not generalisable

---

## KD-011 — Landmark Detection: Why Not LLM or Embeddings?

**Decision:** Use rule-based text signals and cross-turn alignment (KD-010) as the v1 landmark detector. LLM-based detection and embedding similarity are explicitly deferred to v2 as documented upgrade paths, not dismissed.

**The honest quality comparison:**

| Approach | Recall (est.) | Cost per conv | Latency | Generalisable | Deterministic |
|---|---|---|---|---|---|
| Rules + two-pass alignment (v1) | 86.6% (measured) | Free | <5ms | With new patterns | Yes |
| Embedding similarity to prototypes (v2) | ~80–88% | Free | ~50ms | With new prototypes | Yes |
| LLM per turn (batch) | ~95% | ~$0.04–0.10 | ~30s | Yes | At temp=0 |
| LLM per conversation (single call) | ~92% | ~$0.01–0.03 | ~5s | Yes | At temp=0 |

**Why not LLM detection in v1:**
- **Net cost problem:** The optimizer exists to save token cost. If landmark detection requires an LLM call per conversation (~$0.01–0.10), this eats significantly into compression savings — especially for short conversations where the full-context token cost is already low. The net cost analysis only works if detection is cheap.
- **Latency:** One LLM call per conversation adds 2–10s to assembly latency. At 1,692 evaluation conversations this becomes hours of evaluation time.
- **Circular dependency risk:** Using the same model family for detection and for answering creates a risk that the detector's biases align with the answerer's biases, inflating quality metrics.
- **86.6% recall is sufficient for v1:** The false negatives are predominantly short clarifying questions (e.g. "Are there layovers?") that, if dropped, have minimal impact on answer quality — the surrounding context makes them recoverable.

**Why not embedding similarity in v1:**
- Embedding similarity to landmark prototypes would likely match or slightly exceed rule-based recall (~80–88%), but requires: (a) a curated prototype set per domain, (b) a similarity threshold tuned per domain, and (c) the embedding model loaded at detection time (already loaded for scoring, so marginal cost is low).
- The main reason to defer: we need to validate the prototype approach carefully before relying on it. Rule-based detection with measured 86.6% recall is a known quantity; embedding similarity recall is an estimate until measured.
- This is the highest-priority v2 upgrade — see "What I'd Ship Next" roadmap.

**The v2 upgrade path:**
1. Take the highest-confidence rule-detected landmarks (those with multiple signals firing) as seed prototypes
2. Embed all seed prototypes with `all-MiniLM-L6-v2` — already loaded for scoring, zero additional dependency
3. For each turn, compute max cosine similarity to intent/decision/action_item prototype clusters
4. Landmark if similarity > threshold (tune on held-out conversations)
5. Measure recall against slot annotations — compare directly to v1's 86.6%

This upgrade is domain-adaptive: new domains (LEC) only need new prototypes, not new regex patterns. It's also the natural bridge to LLM-based detection — prototypes can be generated by asking an LLM "give me 20 examples of stated intents in a customer service context."

**LLM detection as an optional mode:**
LLM-based detection is architected as a plug-in via `LANDMARK_DETECTOR=llm` env var. When enabled, it batches all turns from a conversation into a single structured prompt, classifying each turn in one call. This is the right mode for: very long conversations (500+ turns) where missing a landmark is expensive, production systems where detection latency is acceptable, and new domains with no prototype examples yet.

---

*Last updated: April 2026*
*Add new entries as decisions are made during architecture and implementation.*
