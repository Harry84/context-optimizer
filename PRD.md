# Product Requirements Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 0.5  
**Date:** April 2026  
**Status:** Approved for Architecture Phase

---

## 1. Problem Statement

LLM agents operating over long conversations face a hard trade-off: include the full history and waste tokens on noise, or truncate naively and lose critical context. Neither is acceptable in production. This system solves it by intelligently selecting and compressing conversation history sent to the model for each query — reducing cost and improving signal-to-noise without degrading answer quality.

The target application is LEC (London Export Corporation), where customer service agents handle multi-turn enquiries involving trade logistics, order tracking, complaints, and onboarding. These conversations are long, topic-shifting, and require the agent to track the customer's evolving intent across many turns.

---

## 2. Goals & Success Criteria

| Goal | Success Criterion |
|---|---|
| Context reduction | 40–60% token reduction vs. naive full-history |
| Quality preservation | Optimised context scores ≥ full context on LLM-as-judge rubric |
| Semantic preservation | BERTScore F1 ≥ 0.85 between full-context and optimised-context answers |
| Structural correctness | No orphaned turns in output; valid role alternation maintained |
| Measurability | Evaluation across ≥10 conversations; report token %, quality score, BERTScore, latency |

**Non-goals (v1):** streaming assembly, fine-tuned classifiers, production serving infrastructure, multi-path evaluation, synthetic LEC data generation.

---

## 3. Phased Approach

### Phase 1 — Proof of Concept on Taskmaster-2 Flights (this submission)
Demonstrate the full optimizer pipeline on real task-oriented dialogues. This proves:
- The architecture is sound and measurably correct on real multi-turn data
- Landmark detection (stated intents, decisions, action items) works and can be evaluated against ground-truth slot annotations
- Adaptive compression produces meaningful token reduction without quality loss
- The evaluation framework produces honest, reproducible numbers

**Why Taskmaster-2 flights:** 1,692 conversations ≥20 turns, mean 30 turns, max 85. Full USER/ASSISTANT turn structure preserved. Slot annotations per utterance provide ground truth for landmark detection. Conversation structure — goal establishment, multi-option comparison, constraint evolution, final decision — is directly analogous to LEC trade logistics enquiries.

### Phase 2 — LEC Domain Adaptation (future work)
Adapt to LEC customer service conversations using a small set of synthetically generated dialogues. Introduces LEC-specific landmark taxonomy (`track_order`, `raise_complaint`, `request_quote`, `confirm_booking`, `escalate_issue`) and action/commitment detection. The pipeline is dataset-agnostic — Phase 2 requires only new data and updated landmark labels, not architectural changes.

---

## 4. Dataset & Preprocessing

**Source:** Google Research Taskmaster-2, flights domain (`data/taskmaster2/flights.json`).  
**Format:** Per-conversation JSON objects with `conversation_id`, `instruction_id`, and `utterances` array.  
**Each utterance:** `speaker` (USER/ASSISTANT), `text`, `segments` (slot annotations with span indices).

### 4.1 Preprocessing Pipeline

1. Load `flights.json` — no tree reconstruction required, conversations are pre-structured linear sequences
2. Filter to conversations with **≥20 turns**
3. Normalise each turn to `{turn_index, speaker, text, slots[], is_landmark}`
4. Pre-compute slot presence per turn from `segments` annotations — used as ground truth for landmark evaluation

**Turn threshold rationale:** Task-oriented dialogues are informationally denser than casual chat. A 20-turn flight booking conversation contains as much load-bearing context as 40+ turns of general conversation. Only 1.5% of Taskmaster-2 conversations reach 50 turns; the 50-turn figure in the assignment was written with chat-style data in mind. Documented in `key_decisions.md` (KD-002).

---

## 5. System Architecture Overview

```
flights.json
    │
    ▼
[Stage 1] Ingestion & Normalisation
    │  Load, filter ≥20 turns, normalise to Conversation objects
    │  Pre-compute slot presence as landmark ground truth
    ▼
[Stage 2] Relevance Scoring Engine
    │  Classify query: factual / analytical / preference
    │  Per-turn composite score against query:
    │    keyword match + semantic similarity + recency decay + landmark boost
    ▼
[Stage 3] Landmark Detection & Compression
    │  Detect: stated intents, decisions, action items
    │  Landmark turns → verbatim (hard rule, overrides score)
    │  High-score turns → verbatim
    │  Low-score runs → LLM-summarised [SUMMARY] turns
    │  Assemble: chronological order, valid role alternation
    ▼
[Stage 4] Evaluation Harness
       Full-context vs. optimised-context answers
       Metrics: token reduction %, LLM-as-judge (4 dimensions),
                BERTScore F1, landmark recall, assembly latency
```

---

## 6. Functional Requirements

### 6.1 Must Have

| ID | Requirement |
|---|---|
| FR-01 | Load Taskmaster-2 flights; filter ≥20 turns; normalise to Conversation objects |
| FR-02 | Pre-compute slot presence per turn from segment annotations (landmark ground truth) |
| FR-03 | Accept `(conversation, query)` as input; return optimised conversation thread |
| FR-04 | Composite relevance scoring: keyword match + semantic similarity + recency decay + landmark boost |
| FR-05 | Local embeddings via `sentence-transformers/all-MiniLM-L6-v2` — no API cost |
| FR-06 | **Landmark detection:** identify and preserve stated intents, decisions, and action items verbatim |
| FR-07 | **Adaptive compression:** classify query as factual / analytical / preference; apply strategy-specific scoring weights and thresholds |
| FR-08 | LLM-based summarisation of low-relevance sub-threads into `[SUMMARY: …]` turns |
| FR-09 | Output passes structural integrity check: valid role alternation, no orphaned turns |
| FR-10 | Evaluation: token reduction %, LLM-as-judge (4-dimension rubric), BERTScore F1, landmark detection recall vs. slot annotations, latency — across ≥10 conversations |
| FR-11 | Prove optimised context mean quality ≥ full context mean quality |

### 6.2 Stretch Goals

| ID | Requirement |
|---|---|
| FR-12 | Net cost analysis: summarisation LLM cost vs. tokens saved |
| FR-13 | Extended evaluation on hotels + restaurant-search domains |
| FR-14 | Phase 2: synthetic LEC conversations with action/commitment detection |

---

## 7. Relevance Scoring

Each turn receives a composite score `S(t, q)` against the current query `q`:

```
S(t, q) = w1·keyword(t,q) + w2·semantic(t,q) + w3·recency(t) + landmark_boost(t)
```

| Signal | Method | Notes |
|---|---|---|
| **Keyword match** | TF-IDF weighted token overlap | Higher weight for rare terms |
| **Semantic similarity** | Cosine similarity, `all-MiniLM-L6-v2` | Local, no API cost |
| **Recency decay** | `exp(-λ · (N - turn_index))`, λ=0.05 initial | Tunable; recent turns score higher |
| **Landmark boost** | +0.3 if turn classified as landmark | Binary; overrides low score in assembly |

Weights `w1, w2, w3` adjusted per query type in v1. v2/v4/v5 use fixed weights `(0.35, 0.50, 0.15)` — the factual profile — regardless of query type. Query type still drives thresholds and top-K fractions.

---

## 8. Adaptive Compression Strategy

Query classified before scoring; drives weight and threshold adjustments:

| Query Type | Detection Signals | Compression Behaviour |
|---|---|---|
| **Factual** | Wh-questions, entity/date references, "what/when/where" | Conservative — high semantic weight; landmarks anywhere in history preserved |
| **Analytical** | "why", "how", "compare", "explain", reasoning keywords | Moderate — preserve reasoning chains and comparisons; compress filler aggressively |
| **Preference** | "which would you", "recommend", "best option" | Recency-heavy — recent turns dominate; older context deprioritised unless landmark |

---

## 9. Landmark Detection

Landmark turns are preserved verbatim regardless of relevance score. Three categories per the assignment specification:

| Category | Definition | Detection Method | Ground Truth Available |
|---|---|---|---|
| **Stated intents** | User establishes or shifts their goal | Slot-annotated USER turns (`flight_search.*` slots) + pattern matching | ✅ Taskmaster-2 slot annotations |
| **Decisions** | User or assistant confirms a choice | Confirmation patterns on slot-bearing turns; `flight1_detail.*` slots | ✅ Taskmaster-2 slot annotations |
| **Action items** | Assistant commits to an action | Pattern matching on ASSISTANT turns ("I'll book", "let me send", "I will") | ❌ No slot annotations; pattern-based only |

Slot annotations cover stated intents and decisions with ground truth — landmark detection recall can be measured against them. Action items require pattern-based detection on assistant turns; this is noted as a limitation in the evaluation report.

---

## 10. Compression & Assembly Rules

1. Landmark turns (any category) → verbatim, always
2. Turns with `S(t, q) ≥ high_threshold` → verbatim
3. Contiguous runs of low-score, non-landmark turns → single `[SUMMARY: …]` turn via LLM
4. Final thread: strict chronological order, valid USER/ASSISTANT alternation, no dropped turns without summary placeholder

---

## 11. Evaluation Design

### 11.1 LLM-as-Judge Rubric

**Bias mitigations:** independent evaluation (full and optimised answers judged separately against their own contexts — no side-by-side presentation, so positional bias does not apply), temperature=0, different judge model from generation model.

**Rubric (1–10 per dimension, independently scored):**

| Dimension | What is assessed |
|---|---|
| **Correctness** | Are factual claims accurate relative to the conversation? |
| **Completeness** | Does the answer address all parts of the query? |
| **Landmark consistency** | Does the answer respect stated intents, decisions, and action items? |
| **Hallucination** | Does the answer introduce ungrounded information? (10 = none) |

### 11.2 Additional Metrics

| Metric | Method |
|---|---|
| **Token reduction %** | `(full_tokens − opt_tokens) / full_tokens` |
| **BERTScore F1** | Between `A_full` and `A_opt`; threshold ≥ 0.85 |
| **Landmark recall** | Slot-annotated turns recovered by landmark detector vs. total slot-annotated turns |
| **Assembly latency** | Wall-clock ms from `(conversation, query)` to optimised thread |

### 11.3 Results Table Format

| Conv | Turns | Full Tok | Opt Tok | Reduction | Full Q | Opt Q | ΔQ | BERTScore | LM Recall | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| dlg-xxx | 45 | 8,200 | 4,300 | 48% | 8.3 | 8.5 | +0.2 | 0.89 | 0.84 | 290ms |

**Acceptance bar:** optimised mean quality ≥ full mean quality AND BERTScore F1 ≥ 0.85. Underperforming cases reported, not hidden.

---

## 12. Stack & Model Choices

| Component | Choice | Rationale |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, fast, deterministic |
| Keyword scoring | TF-IDF via `scikit-learn` | Lightweight, deterministic |
| Landmark detection | Slot annotations + pattern matching | Ground truth available for stated intents and decisions |
| Query classification | Rule-based classifier (LLM fallback) | Deterministic by default |
| Summarisation LLM | OpenAI-compatible API (env-configurable) | Quality matters; cost tracked |
| Judge LLM | OpenAI-compatible API, different model | Reduces self-serving bias |
| Semantic preservation | BERTScore F1 (local) | Objective, LLM-independent |

---

## 13. Out of Scope (v1)

- Synthetic LEC conversation generation
- Fine-tuned landmark classifier
- Streaming context assembly
- Production deployment / serving infrastructure
- Multi-domain evaluation (hotels, restaurant-search deferred to stretch)

---

## 14. Deliverables

1. **GitHub repository** — main branch runnable, README with setup, tests passing
2. **Written report** (≤2 pages) — what was built, what broke, honest failure modes
3. **Architecture decisions document** — chosen stack, rejected alternatives, trade-offs
4. **"What I'd ship next"** — 3–5 concrete roadmap items with justification
5. **AI-usage note** — transparent accounting of AI-assisted vs. authored work
