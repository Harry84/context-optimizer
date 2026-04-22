# Key Decisions Log
## Context Optimizer — Running Record of Architectural & Design Choices

> **Purpose:** Records significant decisions made during the project with justification at the time of making them. Feeds directly into the Architecture Decisions Document, written report, and interview follow-up defence.

---

## KD-001 — Dataset Choice: Taskmaster-2, Flights Domain

**Decision:** Use Taskmaster-2 flights domain as the primary evaluation corpus.

**Justification:**
- **Full conversations preserved** — alternating USER/ASSISTANT turns intact with slot annotations per utterance.
- **Structurally analogous to LEC use case** — flight booking conversations exhibit the same dialogue patterns as trade logistics enquiries: user establishes a goal, states constraints across multiple turns, compares 2–3 options, shifts requirements, and confirms a final choice.
- **Rich slot annotations as ground truth** — every utterance has segment-level slot annotations. These serve as verifiable ground truth for landmark detection.
- **Natural compressible noise** — genuine filler (repeated assistant utterances, crowdworker instruction-reading, confirmatory one-word turns) that the system is designed to compress.
- **Scale:** 1,692 conversations ≥20 turns. Mean 30.2 turns, max 85.
- **Licence:** Creative Commons CC BY 4.0.

**Datasets evaluated and rejected:** OASST1 (no slots), Bitext (single-turn), MultiWOZ (too short), dialog2flow (structure stripped), real LEC transcripts (not available).

---

## KD-002 — Turn Threshold: ≥20 Turns

**Decision:** Filter corpus to conversations with ≥20 turns. The assignment's "50+ messages" threshold is not applied literally.

**Justification:** Taskmaster-2 has a mean of 20.2 turns. Only 1.5% reach 50 turns — applying that threshold literally would leave an unrepresentatively thin corpus. Task-oriented dialogues are informationally denser than casual chat.

---

## KD-003 — No Tree Reconstruction Required

**Decision:** Taskmaster-2 requires no tree reconstruction. Conversations are pre-structured as linear sequences.

---

## KD-004 — Embedding Model: Local all-MiniLM-L6-v2

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` locally for all semantic similarity.

**Justification:** Zero API cost, deterministic, fast on CPU, competitive on semantic similarity benchmarks.

**Alternatives rejected:** `text-embedding-3-small` (API cost), `all-mpnet-base-v2` (4× slower, overkill), TF-IDF only (misses semantic similarity).

---

## KD-005 — LLM-as-Judge: Rubric Design & Bias Mitigation

**Decision:** 4-dimension rubric (correctness, completeness, landmark consistency, hallucination), temperature=0, independent per-answer evaluation, plus BERTScore F1 as an independent objective metric.

**Bias mitigations:** Independent evaluation per answer (no side-by-side, eliminates positional bias); temperature=0; different judge model from generation model; BERTScore F1 (roberta-large, local) as deterministic second signal.

**Judge variance discovered:** Even at temperature=0, gpt-4o produces non-deterministic scores across identical runs due to floating-point non-determinism in the backend. Full-context quality scores varied by up to 0.3 points between runs, making mean Δ quality unstable. BERTScore is the stable signal for cross-run comparison. Multiple-run averaging is the correct mitigation.

---

## KD-006 — Hard Mode Features are Must Have

**Decision:** Adaptive compression strategy and landmark detection are Must Have, not stretch goals.

---

## KD-007 — Landmark Framing: Assignment Terminology Adopted

**Decision:** Use the assignment's exact framing — **stated intents, decisions, action items** — as the three landmark categories.

---

## KD-008 — Phased Approach: Taskmaster-2 Flights First, LEC Domain Second

**Decision:** Phase 1 proves the optimizer on Taskmaster-2 flights. Phase 2 adapts to LEC. Synthetic generation deferred.

---

## KD-009 — Landmark Detection: Slot Annotations as Evaluation Only, Not Detection

**Decision:** The landmark detector operates on raw text and speaker role only. Slot annotations are used exclusively as independent ground truth for measuring recall.

**Justification:** Using slot annotations for detection would be circular — 100% recall on Taskmaster-2, 0% on any unannotated dataset.

---

## KD-010 — Two-Pass Cross-Turn Alignment for Landmark Detection

**Decision:** Two-pass approach: Pass 1 scores each turn individually; Pass 2 promotes turns based on cross-turn patterns.

**Measured results:** GT recall 86.6%, landmark rate 46.4%, compressible 53.6%, 879 turns promoted by pass 2 (3.7% of all landmarks).

---

## KD-011 — Landmark Detection: Why Not LLM or Embeddings?

**Decision:** Rule-based with cross-turn alignment as v1. LLM and embedding approaches documented as v2 upgrade paths.

| Approach | Recall | Cost/conv | Latency |
|---|---|---|---|
| Rules + two-pass (v1) | 86.6% measured | Free | <5ms |
| Embedding similarity (v2) | ~80–88% est. | Free | ~50ms |
| LLM per conversation | ~92% est. | ~$0.01–0.03 | ~5s |

**Why not LLM:** Net cost problem — detection cost eats into compression savings.

---

## KD-012 — Data Quality: Crowdworker Noise Handling

**Decision:** Apply two data-quality fixes at load time rather than filtering affected conversations.

**Fixes:** `_dedup_sentences()` in `loader.py` (intra-turn repetition); `_smart_merge()` in `assembler.py` (cross-turn near-duplication with substring and 80% word-overlap detection).

---

## KD-013 — Conversation Close Signals as Landmarks

**Decision:** USER turns signalling conversation end are classified as `decision` landmarks.

**Motivation:** "Okay. That will be all." and "Oh, I'm done." score low (no slot signals) but are load-bearing — they tell the LLM the conversation state is resolved.

---

## KD-014 — Summariser Minimum Threshold

**Decision:** COMPRESS runs shorter than 200 characters are dropped entirely rather than summarised.

**Motivation:** Single-turn filler runs were producing LLM summaries longer than the original content. Drop threshold prevents trivial API calls.

---

## KD-015 — Assembly Integrity: Merge vs. Bridge for Consecutive Same-Role Turns

**Decision:** Consecutive ASSISTANT turns merged (smart merge); consecutive USER turns get `[context continues]` bridge.

**Justification:** Merging USER turns would misrepresent conversational structure and lose question-answer pairing. Bridge preserves USER turns as distinct while maintaining valid role alternation.

---

## KD-016 — Turn-Level vs. Sentence-Level Compression

**Decision:** v1 implements turn-level compression. v2 implements sentence-level, achieving 34.1% token reduction on realistic conversations.

**Problem:** Landmark turns contain both load-bearing sentences and filler. Turn-level compression keeps both. Sentence-level compression scores each sentence independently and only keeps the load-bearing ones.

**v2 fixes applied:** NLTK punkt sentence splitting; per-sentence landmark re-detection; batch scoring of non-landmark sentences; sandwiched COMPRESS sentence promotion; same-turn sentence merging on effective disposition before assembly.

**Why the structural fixes were non-obvious:** Splitting turns into sentences creates consecutive same-speaker synthetic turns, which the assembler's integrity check repairs with `[context continues]` bridges — inflating the output. The three-step fix (merge on effective disposition, promote sandwiched sentences, batch before assembly) only emerged from carefully reading the assembled output.

---

## KD-017 — Top-K Retrieval: Landmarks Compete Rather Than Hard-KEEP

**Decision:** In v3–v5, landmarks receive a score boost but are not hard-KEEPed — they compete in the same top-K pool as non-landmark turns.

**Motivation:** Landmark detection is query-agnostic. A turn flagged as a landmark because it contains a date slot is not relevant to a query about prices. Hard-KEEPing all landmarks wastes context budget on query-irrelevant facts.

**Trade-off:** The landmark boost (+0.3, or scaled by individual score in v4/v5) biases the ranking toward landmarks without guaranteeing their survival. Query-irrelevant landmarks can be compressed. The risk is that a genuinely relevant landmark might score below the top-K threshold if the conversation contains many high-scoring non-landmark turns — this is the documented failure mode for airport queries.

**In v1/v2:** landmarks remain hard-KEEPed. These strategies are conservative by design.

---

## KD-018 — Chunk-Based Scoring: The Unit of Relevance is an Exchange, Not a Turn

**Decision:** v5 scores overlapping multi-turn chunks (chunk_size=6, stride=2) rather than individual turns.

**Motivation:** The answer to a query is often spread across multiple consecutive turns. Scoring turns individually misses this — "Okay I'm going with Virgin" scores 0.49 alone but scores 0.91 as part of a chunk containing the flight options, the deliberation, and the decision. This is a meaningful architectural insight: the unit of relevance for a conversational query is an exchange, not a single turn.

**Blended scoring:** `score = 0.7 × max_chunk_score + 0.3 × individual_score`. The chunk component captures answer-spanning relevance; the individual component prevents irrelevant turns (e.g. greetings containing the word "flight") from riding a high chunk score. The landmark boost is scaled by individual score (`boost × indiv_score`) so query-irrelevant landmarks get minimal boost.

**Chunk parameters:** chunk_size=6 selected after testing chunk_size=4 (too small — final comparison turns fall outside high-scoring chunks) and chunk_size=6 (captures full flight comparison exchange including options, deliberation, and decision). stride=2 gives 66% overlap between adjacent chunks.

**Results:** 43.4% token reduction — first strategy to hit the 40–60% target on the real Taskmaster-2 corpus. BERTScore 0.935–0.947, 100% ≥ 0.85. Two persistent failures on airport queries.

---

## KD-019 — Airport Floor: Query-Gated Domain Signal

**Decision:** When the query is airport-related, turns mentioning IATA codes or airport names receive a minimum score of 0.45, regardless of their chunk or individual score.

**Motivation:** Airport queries ("what airports were involved?") consistently failed in v3 and v5 because the airport-mentioning turns scored below the top-K threshold. The chunk containing airport references often also contains unrelated filler, diluting its score. A targeted floor prevents the relevant turns from being compressed.

**Implementation details:**
- Query gate: `_AIRPORT_QUERY_RE` detects queries containing "airport", "terminal", "fly from/to/into", "depart from", "arrive at"
- Turn gate: `_AIRPORT_TURN_RE` detects IATA codes and airport names only — city names deliberately excluded. Adding city names (London, New York etc.) applied the floor to nearly every turn in a flight booking conversation, losing discriminating power.
- Floor value: 0.45 — above `topk_min_score=0.30` (guaranteed to survive the noise filter) but below most genuinely high-scoring turns (0.6–0.9), so airport turns compete rather than dominate.

**Remaining limitation:** Two conversations still produce Δ quality ≈ -3 to -5 on airport queries. The airport floor partially mitigates (worst case improved from Δ-6.25 to Δ-2.75) but does not eliminate the failure. The root cause is that the airport-mentioning turns in those specific conversations are in chunks that score poorly regardless of the floor, and the floor alone cannot overcome the top-K competition from other high-scoring turns.

---

*Last updated: April 2026*
