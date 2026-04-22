# Architecture Decisions Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 1.4
**Date:** April 2026
**Status:** Implemented

---

## 1. System Overview

The Context Optimizer takes a multi-turn conversation and a current query, and returns a compressed conversation thread that preserves the information the LLM needs to answer the query — and discards what it doesn't.

```
Input:  Conversation (N turns) + Query string (at position Q in the conversation)
Output: Optimised [{role, content}] thread covering turns 0..Q-1 (24–34% fewer tokens)
```

**Critical constraint:** Only turns *before* the current query position are considered. Turn Q itself is the query being answered — it is not part of the context window. This mirrors real agent behaviour.

Each stage has a clean interface. Components are swappable without touching adjacent stages.

---

## 2. Pipeline Diagram

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        A["Conversation\n(N turns)"]
        B["Query + Position Q"]
    end

    subgraph S1["Stage 1 — Ingestion"]
        C["Load Taskmaster-2 JSON\nFilter ≥ min_turns\nDedup sentences\nNormalise to Turn objects"]
    end

    subgraph S2["Stage 2 — Landmark Detection\n(zero API cost, < 5ms)"]
        D["Pass 1: Per-turn pattern matching\n· Slot-value signals → intent\n· Offer patterns → decision\n· Strong confirmations → decision\n· Conversation-close → decision\n· Action verbs → action_item\n· Pure filler → not landmark"]
        E["Pass 2: Cross-turn alignment\n· Offer→Confirmation → both decision\n· Constraint→Echo → both intent"]
        D --> E
    end

    subgraph S3["Stage 3 — Relevance Scoring"]
        F["Classify query\nfactual / analytical / preference"]
        G["Score turns 0..Q-1 only\nKeyword: TF-IDF cosine\nSemantic: MiniLM-L6-v2 cosine\nRecency: exp decay\nLandmark boost: +0.3"]
        F --> G
    end

    subgraph S4["Stage 4 — Compression & Assembly"]
        H{{"Compression\nstrategy?"}}

        subgraph V1["v1 — Turn-Level"]
            I["Classify each turn\nKEEP / CANDIDATE / COMPRESS\n(landmark → always KEEP)"]
            J["Group into runs\nmerge singleton COMPRESS runs"]
            I --> J
        end

        subgraph V2["v2 — Sentence-Level"]
            K["Split landmark turns\ninto sentences (NLTK punkt)"]
            L["Re-run landmark patterns\nper sentence"]
            M["Score all non-landmark\nsentences in one batch"]
            N["Classify sentences\n(tighter thresholds)"]
            O["Promote sandwiched\nCOMPRESS → CANDIDATE"]
            P["Merge same-turn\nsame-disposition sentences"]
            K --> L --> M --> N --> O --> P
        end

        H -->|turn| V1
        H -->|sentence| V2

        Q["Summarise COMPRESS runs\ngpt-4o-mini · ≤15 words\nDrop runs < 200 chars"]
        R["Assemble thread\nSmart merge consecutive ASSISTANT\nBridge consecutive USER\nGuarantee valid role alternation"]

        V1 --> Q
        V2 --> Q
        Q --> R
    end

    subgraph S5["Stage 5 — Evaluation"]
        S["Select queries\ngpt-4o-mini picks 2 from\n14-item pool per conversation"]
        T["Generate answers\nFull context vs Optimised\ngpt-4o · temperature=0"]
        U["LLM-as-Judge\nCorrectness · Completeness\nLandmark consistency · Hallucination\ngpt-4o · temperature=0"]
        V["BERTScore F1\nroberta-large · local\nthreshold ≥ 0.85"]
        W["Report\nToken reduction %\nΔ Quality · BERTScore\nLandmark recall · Latency"]
        S --> T --> U --> W
        T --> V --> W
    end

    subgraph OUTPUT["Output"]
        X["Optimised thread\n[{role, content}]\nvalid for LLM consumption"]
        Y["eval_results.csv\nAcceptance bars"]
    end

    A --> S1
    B --> S1
    S1 --> S2
    S2 --> S3
    B --> S3
    S3 --> S4
    S4 --> S5
    R --> X
    S5 --> Y

    style INPUT fill:#1e293b,stroke:#475569,color:#f1f5f9
    style OUTPUT fill:#1e293b,stroke:#475569,color:#f1f5f9
    style S1 fill:#0f172a,stroke:#334155,color:#94a3b8
    style S2 fill:#0f172a,stroke:#334155,color:#94a3b8
    style S3 fill:#0f172a,stroke:#334155,color:#94a3b8
    style S4 fill:#0f172a,stroke:#334155,color:#94a3b8
    style S5 fill:#0f172a,stroke:#334155,color:#94a3b8
    style V1 fill:#172554,stroke:#1d4ed8,color:#bfdbfe
    style V2 fill:#14532d,stroke:#16a34a,color:#bbf7d0
```

---

## 3. Project Structure

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
│   │   ├── compressor.py             # v1 turn-level: classify + group into runs
│   │   ├── sentence_compressor.py    # v2 sentence-level: split, score, classify, merge
│   │   ├── sentence_splitter.py      # NLTK punkt sentence boundary detection
│   │   ├── summariser.py             # LLM summarisation (one call per run)
│   │   ├── assembler.py              # Assemble thread, smart merge, integrity check
│   │   └── pipeline.py               # compress() entry point — selects v1 or v2
│   └── evaluation/
│       ├── harness.py                # Full evaluation loop, query selection, acceptance bars
│       ├── judge.py                  # LLM-as-judge, 4-dimension rubric
│       ├── bertscore_metric.py       # BERTScore F1 (local)
│       └── landmark_recall.py        # Recall vs. slot annotation GT
├── utilities/                        # Inspection and audit scripts
├── tests/                            # pytest suite (73 tests, no LLM calls)
├── main.py                           # CLI: stats, inspect, evaluate
├── .env                              # API keys (gitignored)
└── .env.example                      # Template (committed)
```

---

## 4. Core Data Model

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

## 5. Stage 1 — Ingestion & Normalisation

**Module:** `src/ingestion/loader.py`

Load Taskmaster-2 JSON, filter by turn count, apply data-quality fixes, normalise to `Conversation` objects.

**Data quality fixes applied at load time:**
- `_dedup_sentences()` — removes consecutively repeated sentences within a single utterance. Taskmaster-2 crowdworkers occasionally re-stated a sentence before completing the thought. This is structural noise, not signal.

**What is NOT fixed at load time:**
- Cross-turn near-duplication (consecutive turns where one is a subset of the other) — handled in the assembler's smart merge during assembly.

---

## 6. Stage 2 — Landmark Detection

**Module:** `src/landmarks/`

Classifies each turn as landmark or not before relevance scoring, so landmark boost can be applied in scoring.

### 6.1 Detector Interface (pluggable)

```python
class LandmarkDetector(Protocol):
    def detect(self, conversation: Conversation) -> Conversation: ...
```

Active detector selected via `OptimizerConfig.landmark_detector` ("rules" | "embedding" | "llm").

### 6.2 v1 — Rule-Based + Two-Pass Alignment

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

**Pass 2 — cross-turn alignment:**
- Pattern A (Offer→Confirmation): ASSISTANT[i] offer + USER[i+1] weak confirmation ("yes", "okay") → both `decision`
- Pattern B (Constraint→Echo): USER[i] slot signal + ASSISTANT[i+1] echo ("so you want X, correct?") → both `intent`

**Measured: 86.6% GT recall, 46.4% landmark rate, 53.6% compressible.**

### 6.3 v2 / LLM modes

Documented as upgrade paths in `key_decisions.md KD-011`. Not implemented in v1.

---

## 7. Stage 3 — Relevance Scoring

**Module:** `src/scoring/scorer.py`

Operates only on turns `0..Q-1` where Q is the query position. Future turns are never seen.

### 7.1 Query Classification

Rule-based classifier assigns "factual", "analytical", or "preference" — drives weight selection.

### 7.2 Composite Score

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

## 8. Stage 4 — Compression & Assembly

### 8.1 v1 — Turn-Level (`compressor.py`)

```
is_landmark → KEEP (hard rule)
score ≥ high → KEEP
score ≥ low  → CANDIDATE (kept verbatim; treated as KEEP for grouping)
score < low  → COMPRESS
```

Thresholds: factual (0.72/0.45), analytical (0.65/0.40), preference (0.60/0.35).

Contiguous same-disposition turns grouped into runs. Single-turn COMPRESS runs merged into adjacent COMPRESS runs to reduce LLM call count.

### 8.2 v2 — Sentence-Level (`sentence_compressor.py`)

For landmark turns only:
1. Split into sentences using NLTK punkt tokeniser
2. Re-run landmark patterns at sentence level — only triggering sentences are hard-KEEPed
3. Score all non-landmark sentences in one batch (keyword + semantic + recency)
4. Classify using tighter `sentence_thresholds` (factual 0.80/0.60)
5. Promote COMPRESS sentences sandwiched between KEEPs in the same turn → prevents structural fragmentation
6. Merge consecutive same-disposition sentences from the same turn back into one

Non-landmark turns are treated atomically (same as v1).

Selected via `--compression-strategy turn|sentence`.

### 8.3 Summarisation (`summariser.py`)

One gpt-4o-mini call per COMPRESS run. Runs shorter than 200 characters dropped silently. Summary capped at ≤15 words / 30 tokens.

### 8.4 Assembly (`assembler.py`)

- KEEP/CANDIDATE runs → verbatim turns in chronological order
- COMPRESS runs → single `[SUMMARY: ...]` assistant turn (or dropped if summariser returns empty)
- Smart merge: consecutive ASSISTANT turns merged, substring/near-duplicate detection
- Integrity check: consecutive USER turns bridged with `[context continues]`; thread always starts with user

---

## 9. Stage 5 — Evaluation

**Module:** `src/evaluation/harness.py`

### 9.1 Query Selection

Per-conversation: gpt-4o-mini reads first 15 turns and selects the 2 most answerable queries from a 14-item pool (factual, analytical, preference types). Falls back to defaults on parse error.

### 9.2 Metrics

| Metric | Method | Target |
|---|---|---|
| Token reduction % | `(full - opt) / full` | 40–60% |
| Quality Δ (LLM judge) | Mean of 4 dimensions, opt minus full | ≥ 0 |
| BERTScore F1 | roberta-large, local | ≥ 0.85 |
| Landmark recall | `|detected ∩ GT| / |GT|` | Reported |
| Latency | Wall-clock ms for compress() | Reported |

### 9.3 LLM-as-Judge

Each answer evaluated independently against its own context. Temperature=0. `response_format=json_object` enforced to prevent parse failures. 4 dimensions: correctness, completeness, landmark consistency, hallucination (10 = none).

---

## 10. CLI

```
python main.py stats
python main.py inspect --conv-id X --query Y --query-pos N
python main.py inspect --conv-id X --query Y --query-pos N --dry-run
python main.py inspect --conv-id X --query Y --query-pos N --compare
python main.py --compression-strategy sentence inspect --conv-id X --query Y --query-pos N
python main.py evaluate
python main.py --compression-strategy sentence evaluate
```

---

## 11. Rejected Alternatives

- **RAG-style retrieval** — ignores turn order, cannot enforce landmark preservation for turns dissimilar to query
- **Sliding window truncation** — drops critical early context (original intent)
- **Summarise everything** — loses precision on exact values (prices, dates, flight numbers)
- **LLM as default landmark detector** — net cost problem; 86.6% rule-based recall sufficient for v1
- **Merging consecutive USER turns** — misrepresents conversational structure; bridge approach chosen instead

---

## 12. What Breaks at Scale (500+ Turns)

| Failure Mode | Root Cause | Mitigation |
|---|---|---|
| Embedding latency | Linear with turns | Batch encode once; cached |
| Summarisation cost | One call per COMPRESS run | Short-run drop threshold; singleton merging |
| Landmark detection drift | Regex degrades on unusual phrasing | Upgrade to embedding detector (v2) |
| Assembly integrity failures | More anomalies in long conversations | Smart merge + integrity check with logging |
| Context window exceeded | Summarised thread still too long | Hard token cap; second summarisation pass |
| v2 sentence scoring latency | Large sentence batch on CPU | GPU deployment or embedding cache reuse |

---

## 13. Net Cost Analysis

```
Cost    = summarisation_runs × avg_tokens × cost/token
Savings = (full_tokens - opt_tokens) × cost/token × downstream_calls

Net     = Savings - Cost
```

Break-even with rule-based detection: **≥2 downstream calls per optimised context.**
Break-even with LLM detection: ≥3–5 downstream calls.

---

## 14. Dependencies

| Package | Purpose | Where |
|---|---|---|
| `sentence-transformers` | Semantic embeddings (MiniLM-L6-v2) | Local (CPU) |
| `scikit-learn` | TF-IDF | Local |
| `bert-score` | BERTScore F1 | Local |
| `tiktoken` | Token counting | Local |
| `nltk` | Sentence boundary detection (punkt) | Local |
| `langchain-text-splitters` | NLP utilities (planned embedding work) | Local |
| `openai` | Summarisation + judge + query selection | API |
| `python-dotenv` | `.env` loading with override | Local |
| `pandas` | Evaluation results | Local |
| `pytest` | Test runner | Local |
