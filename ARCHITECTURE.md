# Architecture Decisions Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 1.3
**Date:** April 2026
**Status:** Implemented

---

## 1. System Overview

The Context Optimizer takes a multi-turn conversation and a current query, and returns a compressed conversation thread that preserves the information the LLM needs to answer the query — and discards what it doesn't.

```
Input:  Conversation (N turns) + Query string (at position Q in the conversation)
Output: Optimised [{role, content}] thread covering turns 0..Q-1 (40–60% fewer tokens)
```

**Critical constraint:** Only turns *before* the current query position are considered. Turn Q itself is the query being answered — it is not part of the context window. This mirrors real agent behaviour.

```
Ingestion → Landmark Detection → Relevance Scoring → Compression & Assembly → Evaluation
```

Each stage has a clean interface. Components are swappable without touching adjacent stages.

---

## 2. Project Structure

```
context_optimizer/
├── src/
│   ├── ingestion/
│   │   ├── loader.py                 # Load, filter, normalise, dedup
│   │   └── models.py                 # Conversation, Turn, OptimizerConfig dataclasses
│   ├── scoring/
│   │   ├── scorer.py                 # Composite relevance scorer
│   │   ├── keyword.py                # TF-IDF keyword match
│   │   ├── semantic.py               # Embedding cosine similarity (MiniLM-L6-v2)
│   │   ├── recency.py                # Exponential decay
│   │   └── query_classifier.py       # Factual / analytical / preference
│   ├── landmarks/
│   │   ├── detector.py               # Pluggable detector interface + factory
│   │   └── rule_detector.py          # v1: rule-based + two-pass alignment
│   ├── compression/
│   │   ├── compressor.py             # Classify dispositions, group into runs
│   │   ├── summariser.py             # LLM summarisation (one call per run)
│   │   ├── assembler.py              # Assemble thread, smart merge, integrity check
│   │   └── pipeline.py               # compress() entry point
│   └── evaluation/
│       ├── harness.py                # Full evaluation loop, acceptance bars
│       ├── judge.py                  # LLM-as-judge, 4-dimension rubric
│       ├── bertscore_metric.py       # BERTScore F1 (local)
│       └── landmark_recall.py        # Recall vs. slot annotation GT
├── utilities/                        # Inspection and download scripts
├── tests/                            # pytest suite (no LLM calls)
├── main.py                           # CLI: stats, inspect, evaluate
├── .env                              # API keys (gitignored)
└── .env.example                      # Template (committed)
```

---

## 3. Core Data Model

```python
@dataclass
class Turn:
    turn_index: int
    speaker: str                      # "USER" or "ASSISTANT"
    text: str
    slots: list[str]                  # Slot names from Taskmaster-2 annotations
                                      # EVALUATION GROUND TRUTH ONLY
    is_landmark: bool = False
    landmark_type: str | None = None  # "intent" | "decision" | "action_item"
    landmark_reason: str = ""         # Human-readable reason (auditable)
    promoted: bool = False            # True if promoted by cross-turn alignment
    score: float = 0.0                # Set by relevance scorer
    disposition: str = ""             # "KEEP" | "CANDIDATE" | "COMPRESS"
```

---

## 4. Stage 1 — Ingestion & Normalisation

**Module:** `src/ingestion/loader.py`

Load Taskmaster-2 JSON, filter by turn count, apply data-quality fixes, normalise to `Conversation` objects.

**Data quality fixes applied at load time:**
- `_dedup_sentences()` — removes consecutively repeated sentences within a single utterance. Taskmaster-2 crowdworkers occasionally re-stated a sentence before completing the thought. This is structural noise, not signal.

**What is NOT fixed at load time:**
- Cross-turn near-duplication (consecutive turns where one is a subset of the other) — handled in the assembler's smart merge during assembly.

---

## 5. Stage 2 — Landmark Detection

**Module:** `src/landmarks/`

Classifies each turn as landmark or not before relevance scoring, so landmark boost can be applied in scoring.

### 5.1 Detector Interface (pluggable)

```python
class LandmarkDetector(Protocol):
    def detect(self, conversation: Conversation) -> Conversation: ...
```

Active detector selected via `OptimizerConfig.landmark_detector` ("rules" | "embedding" | "llm").

### 5.2 v1 — Rule-Based + Two-Pass Alignment

**Pass 1 — individual turn scoring:**

| Signal | Landmark type |
|---|---|
| USER: explicit intent verb ("I'd like", "I need") | `intent` |
| USER: slot-value signal (price, date, airline, seat class, stops) | `intent` |
| USER: strong confirmation ("I'll take the 6AM flight") | `decision` |
| USER: conversation-close signal ("that's all", "goodbye", "I'm done") | `decision` |
| ASSISTANT: offer pattern (specific price/time/flight details) | `decision` |
| ASSISTANT: slot-value signal | `decision` |
| ASSISTANT: action commitment ("I'll send", "tickets confirmed") | `action_item` |
| Short pure filler ("okay", "sure", "hold on") | Not a landmark |

**Conversation-close signals (added after manual inspection):**
Turns like "Okay. That will be all." and "Oh, I'm done." contain no slot signals but are load-bearing — they tell the LLM the user's goal was met or abandoned and the conversation is resolved. They are classified as `decision` landmarks and always kept verbatim. Signals include explicit completions, farewell phrases, gratitude closings, and casual wrap-ups ("sounds good", "I'm all set").

**Pass 2 — cross-turn alignment:**
- Pattern A (Offer→Confirmation): ASSISTANT[i] offer + USER[i+1] weak confirmation ("yes", "okay") → both `decision`
- Pattern B (Constraint→Echo): USER[i] slot signal + ASSISTANT[i+1] echo ("so you want X, correct?") → both `intent`

**Measured: 86.6% GT recall, 46.4% landmark rate, 53.6% compressible.**

### 5.3 v2 / LLM modes

Documented as upgrade paths in `key_decisions.md KD-011`. Not implemented in v1.

---

## 6. Stage 3 — Relevance Scoring

**Module:** `src/scoring/scorer.py`

Operates only on turns `0..Q-1` where Q is the query position. Future turns are never seen.

### 6.1 Query Classification

Rule-based classifier assigns "factual", "analytical", or "preference" — drives weight selection.

### 6.2 Composite Score

```
S(t, q) = w1·keyword(t,q) + w2·semantic(t,q) + w3·recency(t,Q) + landmark_boost(t)
```

- `keyword` — TF-IDF cosine similarity (vectoriser fitted on this conversation's history)
- `semantic` — cosine similarity of MiniLM-L6-v2 embeddings (all turns encoded in one batch)
- `recency` — `exp(-0.05 · (Q - t.turn_index))` — recent turns score higher
- `landmark_boost` — +0.3 for landmark turns, applied after normalisation

All components normalised to [0,1] before combining.

**Weights by query type:**

| Query Type | keyword | semantic | recency |
|---|---|---|---|
| Factual | 0.3 | 0.5 | 0.2 |
| Analytical | 0.2 | 0.4 | 0.4 |
| Preference | 0.2 | 0.3 | 0.5 |

---

## 7. Stage 4 — Compression & Assembly

### 7.1 Turn Classification (`compressor.py`)

```
is_landmark → KEEP (hard rule)
score ≥ high → KEEP
score ≥ low  → CANDIDATE (kept verbatim; treated as KEEP for grouping)
score < low  → COMPRESS
```

Thresholds: factual (0.6/0.3), analytical (0.5/0.25), preference (0.45/0.2).

### 7.2 Run Grouping

Contiguous same-disposition turns grouped into runs. Single-turn COMPRESS runs merged into adjacent COMPRESS runs where possible to reduce LLM call count.

### 7.3 Summarisation (`summariser.py`)

One LLM call per COMPRESS run. Runs shorter than 80 characters total are **dropped silently** — no LLM call, no placeholder. This prevents summaries longer than the original content and eliminates cost for pure filler turns.

Prompt instructs the model to: summarise in 1-2 sentences, preserve constraints/prices/options, omit greetings and filler, and return "SKIP" if nothing meaningful.

### 7.4 Assembly (`assembler.py`)

Builds final `[{role, content}]` thread:
- KEEP/CANDIDATE runs → verbatim turns in order
- COMPRESS runs → single `[SUMMARY: ...]` assistant turn (or nothing if summariser returned empty)

**Smart merge** (`_smart_merge`): when consecutive ASSISTANT turns are merged (summary + verbatim, or two verbatim turns):
- New content substring of existing → skip (already present)
- Existing content substring of new → replace (new is a superset — handles crowdworker repeat-then-complete pattern)
- 80%+ word overlap → skip (near-duplicate)
- Otherwise → append

**Integrity check** repairs:
- Consecutive ASSISTANT turns → smart merge
- Consecutive USER turns → insert `[context continues]` assistant bridge (preserves USER turns as distinct)
- Thread starting with ASSISTANT → prepend `[conversation start]` user turn

### 7.5 Pipeline Entry Point (`pipeline.py`)

`compress(conversation, query, query_position, config)` — main entry point.
Runs all stages and returns `(thread, AssemblyStats, latency_ms)`.

---

## 8. Stage 5 — Evaluation

**Module:** `src/evaluation/harness.py`

Runs full vs. optimised context for each (conversation, query) pair, collects all metrics, prints acceptance bar summary, saves `eval_results.csv`.

### 8.1 Metrics

| Metric | Method | Target |
|---|---|---|
| Token reduction % | `(full - opt) / full` | 40–60% |
| Quality Δ (LLM judge) | Mean of 4 dimensions, opt minus full | ≥ 0 |
| BERTScore F1 | roberta-large, local | ≥ 0.85 |
| Landmark recall | `\|detected ∩ GT\| / \|GT\|` | Reported |
| Latency | Wall-clock ms for compress() | Reported |

### 8.2 LLM-as-Judge

Each answer evaluated independently against its own context (not side-by-side). Temperature=0. 4 dimensions: correctness, completeness, landmark consistency, hallucination (10 = no hallucination).

---

## 9. CLI

```
python main.py stats                         # corpus statistics
python main.py inspect --conv-id X --query Y --query-pos N          # full compression
python main.py inspect --conv-id X --query Y --query-pos N --dry-run # no API calls
python main.py inspect --conv-id X --query Y --query-pos N --compare # side-by-side
python main.py evaluate                      # full evaluation pipeline
```

---

## 10. Rejected Alternatives

- **RAG-style retrieval** — ignores turn order, cannot enforce landmark preservation for turns dissimilar to query
- **Sliding window truncation** — drops critical early context (original intent)
- **Summarise everything** — loses precision on exact values (prices, dates, flight numbers)
- **LLM as default landmark detector** — net cost problem; 86.6% rule-based recall sufficient for v1
- **Merging consecutive USER turns** — misrepresents conversational structure; bridge approach chosen instead

---

## 11. What Breaks at Scale (500+ Turns)

| Failure Mode | Root Cause | Mitigation |
|---|---|---|
| Embedding latency | Linear with turns | Batch encode once; cached |
| Summarisation cost | One call per COMPRESS run | Short-run drop threshold; singleton merging |
| Landmark detection drift | Regex degrades on unusual phrasing | Upgrade to embedding detector (v2) |
| Assembly integrity failures | More anomalies in long conversations | Smart merge + integrity check with logging |
| Context window exceeded | Summarised thread still too long | Hard token cap; second summarisation pass |

---

## 12. Net Cost Analysis

```
Cost    = summarisation_runs × avg_tokens × cost/token
Savings = (full_tokens - opt_tokens) × cost/token × downstream_calls

Net     = Savings - Cost
```

Break-even with rule-based detection: **≥2 downstream calls per optimised context.**
Break-even with LLM detection: ≥3–5 downstream calls.

---

## 13. Dependencies

| Package | Purpose | Where |
|---|---|---|
| `sentence-transformers` | Semantic embeddings | Local (CPU) |
| `scikit-learn` | TF-IDF | Local |
| `bert-score` | BERTScore F1 | Local |
| `tiktoken` | Token counting | Local |
| `openai` | Summarisation + judge | API |
| `python-dotenv` | `.env` loading with override | Local |
| `pandas` | Evaluation results | Local |
| `pytest` | Test runner | Local |
