# Product Requirements Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 0.3  
**Date:** April 2026  
**Status:** Approved for Architecture Phase

---

## 1. Problem Statement

LLM agents operating over long conversations face a hard trade-off: include the full history and waste tokens on noise, or truncate naively and lose critical context. Neither is acceptable in production. This system solves it by intelligently selecting and compressing the conversation history sent to the model for each query — reducing cost and improving signal-to-noise without degrading answer quality.

---

## 2. Goals & Success Criteria

| Goal | Success Criterion |
|---|---|
| Context reduction | 40–60% token reduction vs. naive full-history |
| Quality preservation | Optimised context scores ≥ full context on LLM-as-judge rubric |
| Semantic preservation | BERTScore F1 ≥ 0.85 between full-context and optimised-context answers |
| Structural correctness | No broken tool-call chains, no orphaned turns in output |
| Measurability | Evaluation across ≥10 conversations; report token %, quality score, BERTScore, latency |

**Non-goals (v1):** streaming assembly for 1000+ turns, fine-tuned landmark classifiers, production serving infrastructure, multi-path branch evaluation.

---

## 3. Dataset & Preprocessing

**Source:** OpenAssistant/oasst1 (~137MB JSONL, ~84k message rows, downloaded locally at `data/oasst1.jsonl`).

The dataset is a flat list of message nodes — not pre-assembled conversations. Each row contains:
- `message_id`, `parent_id` — tree linkage
- `role` — `prompter` or `assistant`
- `text` — message content
- `lang` — language tag
- `rank`, `rank_end` — human preference ranking among sibling branches
- Various quality metadata fields

### 3.1 Tree Reconstruction Pipeline

Conversation trees must be explicitly reconstructed before any context optimisation can occur:

1. Parse JSONL into a message store keyed by `message_id`
2. Build parent→children adjacency from `parent_id` links; root nodes have `parent_id = null`
3. At each branching point, select the **highest-ranked child** (lowest `rank` value per OASST1 convention) — this gives one canonical linear thread per tree
4. Walk each tree depth-first along the highest-ranked path, materialising a linear sequence of turns
5. Filter to threads with **≥50 turns** — this is the working corpus
6. Normalise each turn to `{turn_index, role, content, message_id, metadata}`

**Branching strategy rationale:** OASST1 trees branch at many nodes (multiple human responses, multiple assistant responses). For v1 we take the single highest-ranked path per tree. This keeps tree reconstruction simple and unambiguous while still producing naturalistic, high-quality multi-turn conversations. Multi-path evaluation is deferred to future work.

**Expected yield:** OASST1 contains ~9k conversation trees. The majority of deep threads are English multi-turn exchanges. We anticipate 50–200 threads meeting the ≥50 turn threshold after filtering.

---

## 4. System Architecture Overview

The system runs as four sequential stages:

```
Raw JSONL
    │
    ▼
[Stage 1] Ingestion & Tree Reconstruction
    │  Reconstruct trees, select highest-ranked path,
    │  filter ≥50 turns, normalise to Conversation objects
    ▼
[Stage 2] Relevance Scoring Engine
    │  Per-message composite score against current query:
    │  keyword match + semantic similarity + recency decay + landmark boost
    │  + query type classification (factual / analytical / preference)
    ▼
[Stage 3] Compression & Assembly
    │  High-score messages → kept verbatim
    │  Landmarks & tool-call turns → always verbatim (hard rule)
    │  Low-score sub-threads → LLM-summarised into [summary] turns
    │  Enforce chronological order + referential integrity
    ▼
[Stage 4] Evaluation Harness
       Full-context vs. optimised-context → LLM-as-judge + BERTScore
       Report: token reduction %, quality score, semantic preservation, latency
```

---

## 5. Functional Requirements

### 5.1 Must Have

| ID | Requirement |
|---|---|
| FR-01 | Reconstruct OASST1 conversation trees from flat JSONL; select highest-ranked path at each branch |
| FR-02 | Filter corpus to threads ≥50 turns; persist as structured Conversation objects |
| FR-03 | Accept `(conversation, query)` as input; return optimised conversation thread |
| FR-04 | Composite relevance scoring: keyword match + semantic similarity + recency decay + landmark boost |
| FR-05 | Local embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic similarity — no API cost |
| FR-06 | LLM-based summarisation of low-relevance sub-threads into synthetic `[SUMMARY: …]` turns |
| FR-07 | Verbatim preservation of landmark turns (decisions, commitments, action items) and any tool-call turns |
| FR-08 | Output passes structural integrity check: valid role alternation, no orphaned turns, no broken references |
| FR-09 | Evaluation framework reporting token reduction %, LLM-as-judge quality, BERTScore F1, and latency across ≥10 conversations |
| FR-10 | Demonstrate optimised context mean quality ≥ full context mean quality across eval set |
| FR-11 | **Adaptive compression strategy:** classify each query as factual / analytical / preference and apply strategy-specific thresholds and compression rules |
| FR-12 | **Landmark detection:** identify and tag messages containing decisions, commitments, stated intents, and action items; preserve these verbatim regardless of relevance score |

### 5.2 Stretch Goals

| ID | Requirement |
|---|---|
| FR-13 | Net cost analysis: track summarisation LLM call cost vs. tokens saved — report whether optimisation is net-positive |
| FR-14 | 500-message stress test with documented failure modes and latency profile |

---

## 6. Relevance Scoring Detail

Each message receives a composite score `S(m, q)` against the current query `q`:

```
S(m, q) = w1·keyword(m,q) + w2·semantic(m,q) + w3·recency(m) + landmark_boost(m)
```

| Signal | Method | Notes |
|---|---|---|
| **Keyword match** | TF-IDF weighted token overlap | Higher weight for rare terms |
| **Semantic similarity** | Cosine similarity, `all-MiniLM-L6-v2` embeddings | Run locally, no API cost |
| **Recency decay** | `exp(-λ · (N - turn_index))`, λ=0.05 initial | Tunable; recent turns score higher |
| **Landmark boost** | +0.3 if message classified as landmark | Binary; regex heuristics in v1 |

Weights `w1, w2, w3` are tunable per query type (see §7).

---

## 7. Adaptive Compression Strategy

Query type is classified before scoring. This classification drives threshold and weight adjustments:

| Query Type | Detection Signal | Compression Behaviour |
|---|---|---|
| **Factual** | Wh-questions, specific entity references, date/number queries | Conservative: lower compression, high semantic weight — facts can be buried anywhere in history |
| **Analytical** | "why", "how", "compare", "explain", reasoning keywords | Moderate: landmark-heavy — decisions and reasoning chains preserved, filler compressed aggressively |
| **Preference** | "what do you think", "which would you recommend", user preference phrases | Aggressive recency: recent turns dominate, older context deprioritised unless landmark |

Classification is performed by a lightweight LLM call (or rule-based classifier as fallback) at the start of each optimisation request.

---

## 8. Landmark Detection

Landmark messages are preserved verbatim regardless of relevance score. In v1, detection uses regex heuristics across the following categories:

| Category | Example Trigger Phrases |
|---|---|
| **Decision** | "we decided", "going with", "final answer is", "let's do", "I've chosen" |
| **Commitment** | "I will", "I'll make sure", "you can count on", "I promise", "by [date]" |
| **Stated intent** | "my goal is", "I want to", "I'm planning to", "the objective is" |
| **Action item** | "action item:", "TODO:", "next step:", "follow up on", "don't forget" |

Landmark classification is a binary flag. Future work: replace heuristics with a fine-tuned classifier or small LLM pass.

---

## 9. Compression & Assembly Rules

1. Messages with `S(m, q) ≥ high_threshold` → included verbatim
2. Messages classified as landmarks → included verbatim (overrides score)
3. Messages containing tool calls or tool results → included verbatim (hard rule)
4. Contiguous runs of low-score messages → collapsed into a single `[SUMMARY: …]` assistant turn via LLM call
5. Final thread must:
   - Maintain strict chronological (turn index) order
   - Alternate roles correctly (prompter / assistant)
   - Replace any dropped message referenced by a later kept message with a summary placeholder

---

## 10. Evaluation Design

### 10.1 LLM-as-Judge — Rubric & Bias Mitigation

LLM-as-judge is used for quality scoring but is implemented carefully to address known failure modes:

**Known failure modes addressed:**

| Bias | Mitigation |
|---|---|
| Positional bias (favours first answer seen) | Each pair evaluated twice with order swapped; scores averaged |
| Verbosity bias (longer = better) | Rubric dimensions are scored independently; length is not a dimension |
| Self-serving bias (model favours its own output) | Judge model differs from generation model where possible |
| Inconsistency across runs | Temperature=0 for judge; same prompt template every time |

**Rubric (scored 1–10 per dimension, independently):**

| Dimension | What is assessed |
|---|---|
| **Correctness** | Are factual claims accurate relative to the conversation? |
| **Completeness** | Does the answer address all parts of the query? |
| **Decision consistency** | Does the answer respect commitments and decisions made earlier in the conversation? |
| **Hallucination** | Does the answer introduce information not grounded in context? (10 = no hallucination) |

Final quality score = mean of four dimensions.

### 10.2 Semantic Preservation (Objective Metric)

To provide an LLM-independent measure of whether the optimised context preserves the semantic content of the full-context answer:

- Generate answer `A_full` using full conversation history
- Generate answer `A_opt` using optimised context
- Compute **BERTScore F1** between `A_full` and `A_opt`
- Threshold: BERTScore F1 ≥ 0.85 indicates acceptable semantic preservation

BERTScore is deterministic, does not require an API call after initial model load, and is well-validated in the NLP literature for semantic similarity. It measures token-level alignment using contextual embeddings rather than surface overlap, making it robust to paraphrasing.

### 10.3 Conversations & Queries

**Conversations:** ≥10 reconstructed threads, varied length (50–200+ turns) and topic domain  
**Queries:** 2–3 multi-step queries per conversation; one factual, one analytical, one preference where possible

### 10.4 Results Table

| Conv | Turns | Full Tokens | Opt Tokens | Reduction % | Full Quality | Opt Quality | Δ Quality | BERTScore F1 | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| conv_001 | 67 | 12,400 | 6,800 | 45% | 8.4 | 8.6 | +0.2 | 0.91 | 340 |
| … | | | | | | | | | |

**Acceptance bar:** optimised context mean quality ≥ full context mean quality, AND mean BERTScore F1 ≥ 0.85. Cases where optimised underperforms are reported, not hidden.

---

## 11. Embedding & Model Choices

| Component | v1 Choice | Rationale |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, fast, no API cost, well-validated for retrieval-style similarity |
| Keyword scoring | TF-IDF via `scikit-learn` | Lightweight, deterministic, no external calls |
| Summarisation LLM | OpenAI-compatible API (configurable via env var) | Quality matters; cost tracked for net cost analysis |
| Judge LLM | OpenAI-compatible API, different model from generation where possible | Reduces self-serving bias |
| Semantic preservation | BERTScore F1 (`bert-score` library, local) | Objective, LLM-independent, deterministic |

---

## 12. Open Questions (Resolved)

| Question | Decision |
|---|---|
| Branching strategy | Highest-ranked path per branch (v1); multi-path deferred |
| Embedding model | Local `all-MiniLM-L6-v2`; OpenAI embeddings deferred |
| λ for recency decay | Start at 0.05; sweep 0.01–0.1 during eval tuning |
| Landmark detection | Regex heuristics in v1; LLM classifier as stretch goal |
| Summarisation model | OpenAI-compatible API; model configurable via env var |
| Hard mode signals | Adaptive strategy and landmark detection are Must Have, not stretch |
| Judge reliability | Mitigated via order-swap, rubric decomposition, and independent BERTScore metric |

---

## 13. Out of Scope (v1)

- Streaming / incremental context assembly
- Fine-tuned landmark classifier
- Multi-path branch evaluation
- Concurrent / multi-session handling
- Production deployment / serving infrastructure

---

## 14. Deliverables

1. **GitHub repository** — main branch runnable, README with setup, tests passing
2. **Written report** (≤2 pages) — what was built, what broke, honest failure modes
3. **Architecture decisions document** — chosen stack, rejected alternatives, trade-offs
4. **"What I'd ship next"** — 3–5 concrete roadmap items with justification
5. **AI-usage note** — transparent accounting of AI-assisted vs. authored work
