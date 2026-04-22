# Architecture Decisions Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 1.1
**Date:** April 2026
**Status:** Approved for Implementation

---

## 1. System Overview

The Context Optimizer takes a multi-turn conversation and a current query, and returns a compressed conversation thread that preserves the information the LLM needs to answer the query — and discards what it doesn't.

```
Input:  Conversation (N turns) + Query string
Output: Optimised conversation thread (40–60% fewer tokens)
```

The system is a four-stage pipeline: **Ingestion → Landmark Detection → Scoring → Compression → Evaluation**. Each stage has a clean interface; components are swappable without touching other stages. This is deliberate — the landmark detector in particular is designed with a pluggable interface so it can be upgraded from rules to embeddings to LLM without changing the rest of the pipeline.

---

## 2. Project Structure

```
context_optimizer/
├── data/
│   └── taskmaster2/
│       └── flights.json              # Primary corpus
├── src/
│   ├── ingestion/
│   │   ├── loader.py                 # Load and filter conversations
│   │   └── models.py                 # Conversation, Turn dataclasses
│   ├── scoring/
│   │   ├── scorer.py                 # Composite relevance scorer
│   │   ├── keyword.py                # TF-IDF keyword match
│   │   ├── semantic.py               # Embedding cosine similarity
│   │   ├── recency.py                # Exponential decay
│   │   └── query_classifier.py       # Factual / analytical / preference
│   ├── landmarks/
│   │   ├── detector.py               # Pluggable landmark detector interface
│   │   ├── rule_detector.py          # v1: rule-based + two-pass alignment
│   │   ├── embedding_detector.py     # v2: prototype embedding similarity
│   │   └── llm_detector.py           # optional: LLM-based (env flag)
│   ├── compression/
│   │   ├── compressor.py             # Select turns, summarise low-value runs
│   │   ├── summariser.py             # LLM summarisation calls
│   │   └── assembler.py              # Assemble final thread, integrity check
│   └── evaluation/
│       ├── harness.py                # Run full vs. optimised, collect metrics
│       ├── judge.py                  # LLM-as-judge with order-swap
│       ├── bertscore_metric.py       # BERTScore F1 computation
│       └── landmark_recall.py        # Recall vs. slot annotation ground truth
├── utilities/
│   ├── view_conversation.py
│   ├── view_domain.py
│   ├── corpus_stats.py
│   ├── verify_classifiers.py         # Visual verification of detector
│   └── diagnose_recall.py            # Missed turn analysis
├── tests/
│   ├── test_ingestion.py
│   ├── test_scoring.py
│   ├── test_landmarks.py
│   ├── test_compression.py
│   └── test_evaluation.py
├── main.py                           # CLI entry point
├── PRD.md
├── ARCHITECTURE.md                   # This document
├── key_decisions.md
└── README.md
```

---

## 3. Core Data Model

Everything in the pipeline flows through two dataclasses. Defining these first ensures all stages share a common contract.

```python
@dataclass
class Turn:
    turn_index: int
    speaker: str                      # "USER" or "ASSISTANT"
    text: str
    slots: list[str]                  # Slot names from Taskmaster-2 annotations
                                      # (evaluation ground truth only — not used in detection)
    is_landmark: bool = False         # Set by landmark detector
    landmark_type: str | None = None  # "intent" | "decision" | "action_item"
    landmark_reason: str = ""         # Human-readable reason (auditable)
    promoted: bool = False            # Promoted by cross-turn alignment pass
    score: float = 0.0                # Set by relevance scorer
    summary: str | None = None        # Set if turn is replaced by a summary

@dataclass
class Conversation:
    conversation_id: str
    instruction_id: str
    turns: list[Turn]
    domain: str = "flights"
```

**Decision:** Use dataclasses over dicts throughout. Provides IDE support, type safety, and prevents silent key errors. The `landmark_reason` and `promoted` fields are deliberate — every detection decision is auditable, which is important for debugging and for the evaluation report.

---

## 4. Stage 1 — Ingestion & Normalisation

**Module:** `src/ingestion/loader.py`

**Responsibility:** Load raw Taskmaster-2 JSON, filter by turn count, normalise to `Conversation` objects, extract slot annotations per turn.

```python
def load_corpus(path: str, min_turns: int = 20) -> list[Conversation]:
    ...

def normalise_turn(raw: dict, index: int) -> Turn:
    slots = [
        ann["name"]
        for seg in raw.get("segments", [])
        for ann in seg.get("annotations", [])
    ]
    return Turn(
        turn_index=index,
        speaker=raw["speaker"],
        text=raw["text"],
        slots=slots,          # stored for evaluation GT only
    )
```

**No tree reconstruction required.** Taskmaster-2 conversations are pre-structured linear sequences. Slot annotations are extracted at load time and stored on each `Turn` — they are never read by the landmark detector, only by the evaluation harness.

---

## 5. Stage 2 — Landmark Detection

**Module:** `src/landmarks/detector.py`

**Responsibility:** Classify each turn as landmark or not, and assign a type. Runs before scoring so landmark boost can be applied. The detector interface is pluggable — the active implementation is selected via config.

### 5.1 Detector Interface

```python
class LandmarkDetector(Protocol):
    def detect(self, conversation: Conversation) -> Conversation:
        """Annotate turns in-place with is_landmark, landmark_type, reason."""
        ...
```

Active detector selected by `OptimizerConfig.landmark_detector`:
- `"rules"` — v1 rule-based + two-pass alignment (default)
- `"embedding"` — v2 prototype embedding similarity
- `"llm"` — LLM-based batch classification (optional, high quality, higher cost)

### 5.2 v1: Rule-Based + Two-Pass Alignment (`rule_detector.py`)

**Why rules, not LLM or embeddings:** See §11.5 and KD-011. In summary: 86.6% measured recall at zero marginal cost and <5ms latency. LLM detection adds $0.01–0.10/conversation and 2–10s latency, undermining the net cost case for compression. Embedding similarity is the natural v2 upgrade — it needs prototype curation before deployment.

**Pass 1 — Individual turn scoring using text signals:**

Slot-value signals (domain vocabulary indicating information content):
- Times (`\d{1,2}:\d{2}\s*(am|pm)`), prices (`\$\d+`), airlines, seat classes, stop preferences, trip types, locations, dates, amenities

A USER turn containing any slot-value signal → **stated intent** candidate.
An ASSISTANT turn containing any slot-value signal or offer pattern → **decision** candidate.
An ASSISTANT turn matching commitment patterns ("I'll send", "let me find", "tickets confirmed") → **action item**.

Pure filler turns (short, no slot signals, matching acknowledgement patterns) → always compressible.

**Pass 2 — Cross-turn alignment:**

```
Pattern A — Offer → Confirmation:
  ASSISTANT[i] presents concrete option (time/price/airline)
  AND USER[i+1] gives weak confirmation ("yes", "okay", "sure")
  → both promoted to landmark:decision

Pattern B — Constraint → Echo:
  USER[i] contains slot-value signal
  AND ASSISTANT[i+1] echoes it back ("You said X?", "So you want Y?")
  → both promoted to landmark:intent
```

These patterns are dataset-agnostic — they rely on conversational structure, not domain vocabulary. They resolve the core ambiguity of single-turn scoring: a bare "yes" is noise alone but a decision when it follows a flight offer.

**Measured performance on Taskmaster-2 flights corpus (1,692 conversations, ≥20 turns):**
- GT recall: **86.6%** (20,092 / 23,190 slot-annotated turns detected)
- Landmark rate: **46.4%** of all turns
- Compressible: **53.6%** of all turns
- Turns promoted by pass 2: 879 (3.7% of all landmarks)

**Known limitation:** ASSISTANT clarifying questions containing domain vocabulary (e.g. "Are you flying round trip or one way?") are occasionally kept as false positives. This is a conservative error — keeping too much rather than dropping landmarks. Preferred failure mode for a context optimizer.

### 5.3 v2: Embedding Similarity (`embedding_detector.py`)

**When to use:** When adapting to a new domain with different vocabulary, or when improving recall above 86.6% without LLM cost.

**Approach:**
1. Seed prototypes from highest-confidence rule detections (multiple signals firing simultaneously)
2. Embed prototypes with `all-MiniLM-L6-v2` — already loaded for relevance scoring, zero additional model dependency
3. For each turn, compute max cosine similarity to each landmark category's prototype cluster
4. Classify as landmark if similarity > threshold (tuned per category on held-out conversations)

**Advantages over rules:**
- Handles paraphrasing and domain vocabulary shifts without new regex patterns
- New domains (LEC) need only new prototypes, not new code
- Prototypes can be generated cheaply: ask an LLM for 20 examples of stated intents in context X

**Estimated recall:** ~80–88% (comparable to v1, better generalisation). To be measured against slot annotations when implemented.

### 5.4 LLM-Based Detection (`llm_detector.py`)

**When to use:** Enabled via `LANDMARK_DETECTOR=llm` env var. Appropriate for: 500+ turn conversations where missing a landmark is expensive, new domains with no prototype examples, production systems where detection latency is acceptable.

**Implementation:** Batches all turns from a conversation into a single structured prompt:
```
Below are turns from a task-oriented conversation. 
Classify each turn as one of: stated_intent | decision | action_item | compressible

A stated_intent is a USER turn establishing or shifting a goal or constraint.
A decision is a USER or ASSISTANT turn confirming a specific choice.
An action_item is an ASSISTANT turn committing to a concrete next step.
Everything else is compressible.

[TURN 0] USER: Hi, I'd like to get a flight to Orlando...
[TURN 1] ASSISTANT: When in March?
...

Return JSON: [{"turn": 0, "type": "stated_intent"}, ...]
```

**One LLM call per conversation** — not per turn. Estimated cost: $0.01–0.03/conversation at gpt-4o-mini pricing. Estimated recall: ~92–95%.

**Why not v1 default:** At 1,692 evaluation conversations × 3 queries each = ~5,000 optimisation runs, LLM detection at $0.02/run = ~$100 evaluation cost. Rule-based detection achieves 86.6% recall at $0 — sufficient for v1.

---

## 6. Stage 3 — Relevance Scoring Engine

**Module:** `src/scoring/scorer.py`

**Responsibility:** Assign each turn a relevance score against the current query. Classify query type first; use type to set component weights.

### 6.1 Query Classification

Rule-based classifier (LLM fallback configurable via `QUERY_CLASSIFIER=llm`):

| Query Type | Detection Signals |
|---|---|
| **Factual** | what/when/where/who/which, specific entity or date reference |
| **Analytical** | why/how/compare/explain/difference/reason |
| **Preference** | recommend/suggest/best/which would/prefer |
| **Default** | Analytical — safest, preserves reasoning context |

### 6.2 Composite Scoring

```
S(t, q) = w1·keyword(t,q) + w2·semantic(t,q) + w3·recency(t) + landmark_boost(t)
```

**Weights by query type:**

| Query Type | w1 (keyword) | w2 (semantic) | w3 (recency) |
|---|---|---|---|
| Factual | 0.3 | 0.5 | 0.2 |
| Analytical | 0.2 | 0.4 | 0.4 |
| Preference | 0.2 | 0.3 | 0.5 |

- **Keyword match:** TF-IDF cosine similarity between query and turn
- **Semantic similarity:** `all-MiniLM-L6-v2` cosine similarity; embeddings cached per conversation
- **Recency decay:** `exp(-λ · (N - turn_index))`, λ=0.05, tunable
- **Landmark boost:** +0.3 if `turn.is_landmark` — ensures landmark turns always clear the high threshold

**Decision: additive weighted sum.** Evaluation set too small to train a learned scorer reliably. Interpretable weights, transparent, tunable via harness.

---

## 7. Stage 4 — Compression & Assembly

**Module:** `src/compression/`

### 7.1 Selection Logic (`compressor.py`)

```
For each turn t:
    if t.is_landmark:              → KEEP verbatim (hard rule)
    elif t.score >= HIGH_THRESH:   → KEEP verbatim
    elif t.score >= LOW_THRESH:    → CANDIDATE
    else:                          → COMPRESS
```

Thresholds by query type:

| Query Type | HIGH_THRESH | LOW_THRESH |
|---|---|---|
| Factual | 0.6 | 0.3 |
| Analytical | 0.5 | 0.25 |
| Preference | 0.45 | 0.2 |

Contiguous COMPRESS runs → single LLM summarisation call.

### 7.2 Summarisation (`summariser.py`)

One LLM call per contiguous run. Prompt instructs the model to preserve constraints, options, and decisions while omitting filler. Batching minimises API calls and cost.

### 7.3 Assembly (`assembler.py`)

1. Replace COMPRESS runs with synthetic `[SUMMARY: <text>]` ASSISTANT turns
2. Sort by `turn_index` — strict chronological order
3. Integrity check: no consecutive same-speaker turns, no orphaned USER turns
4. Return as `[{role, content}]` — valid LLM message format

---

## 8. Stage 5 — Evaluation Harness

**Module:** `src/evaluation/harness.py`

```python
for conv in eval_set:           # ≥10 conversations
    for query in queries[conv]: # 2-3 queries: factual, analytical, preference
        full_thread = format_full(conv)
        opt_thread  = optimizer.run(conv, query)
        A_full = llm.answer(full_thread, query)
        A_opt  = llm.answer(opt_thread,  query)

        token_reduction = tokens(full_thread - opt_thread) / tokens(full_thread)
        quality_full    = judge.score(query, A_full)   # order-swap applied
        quality_opt     = judge.score(query, A_opt)
        bertscore       = bertscore_f1(A_full, A_opt)
        lm_recall       = landmark_recall(conv)
        latency         = time_compression(conv, query)
```

**LLM-as-judge rubric (4 dimensions, 1–10 each, temperature=0, order-swapped):**
- Correctness, Completeness, Landmark consistency, Hallucination

**BERTScore F1** (`roberta-large`, local): objective semantic preservation metric. Threshold ≥ 0.85.

**Landmark recall:** `|detected ∩ GT| / |GT|` where GT = slot-annotated turns. Action items excluded (no GT annotations).

---

## 9. Interfaces Between Stages

```python
load_corpus(path, min_turns)     -> list[Conversation]
detector.detect(conversation)    -> Conversation        # annotates is_landmark in-place
score_turns(conversation, query) -> Conversation        # annotates score in-place
compress(conversation, query)    -> list[dict]          # [{role, content}]
evaluate(conversations, queries) -> pd.DataFrame        # results table
```

---

## 10. Configuration

```python
@dataclass
class OptimizerConfig:
    # Landmark detection
    landmark_detector: str = "rules"     # "rules" | "embedding" | "llm"
    # Scoring
    lambda_decay: float = 0.05
    landmark_boost: float = 0.3
    weights: dict = field(default_factory=lambda: {
        "factual":    {"keyword": 0.3, "semantic": 0.5, "recency": 0.2},
        "analytical": {"keyword": 0.2, "semantic": 0.4, "recency": 0.4},
        "preference": {"keyword": 0.2, "semantic": 0.3, "recency": 0.5},
    })
    thresholds: dict = field(default_factory=lambda: {
        "factual":    {"high": 0.6, "low": 0.3},
        "analytical": {"high": 0.5, "low": 0.25},
        "preference": {"high": 0.45, "low": 0.2},
    })
    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    summarisation_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    query_classifier: str = "rules"      # "rules" | "llm"
    # Corpus
    min_turns: int = 20
    data_path: str = "data/taskmaster2/flights.json"
```

All tunable parameters in one place. No magic numbers scattered through code.

---

## 11. Rejected Alternatives

### 11.1 RAG-style Retrieval

Pure embedding retrieval ignores turn order and structural constraints. Cannot preserve landmarks that are dissimilar to the query text (e.g. "Yes." confirming a decision). **Rejected.**

### 11.2 Sliding Window Truncation

Drops critical early context. The user's original destination request is in turn 3; without it, "which flight did we choose?" is unanswerable. **Rejected** — this is the exact failure mode the system prevents.

### 11.3 Learned Scoring Model

Evaluation set (≥10 conversations) too small to train without overfitting. Additive weighted sum is transparent and appropriate at this scale. **Rejected for v1.**

### 11.4 Summarise Everything

Summaries lose precision on exact values (prices, dates, flight numbers) — the most critical facts in a booking conversation. Verbatim landmark preservation is non-negotiable. **Rejected.**

### 11.5 LLM or Embeddings as Default Landmark Detector

An LLM would achieve ~92–95% recall vs. our measured 86.6%. Embedding similarity would achieve ~80–88%. Both are better quality — but at a cost that matters for v1:

- **LLM cost:** ~$0.01–0.10/conversation. Across 1,692 evaluation conversations this is $17–169 for detection alone, before compression or judging costs. This undermines the net cost case.
- **LLM latency:** 2–10s per conversation. Makes evaluation slow and production latency unacceptable for short sessions.
- **Embedding similarity:** Zero marginal cost (model already loaded), but needs prototype curation before deployment. Recall is estimated, not measured.

**Decision:** Rule-based detection with 86.6% measured recall at zero cost is the right v1 default. Embedding similarity is the highest-priority v2 upgrade. LLM detection is architected as an optional mode (`LANDMARK_DETECTOR=llm`). See §5.3, §5.4, and KD-011 for full analysis.

---

## 12. What Breaks at Scale (500+ Messages)

| Failure Mode | Root Cause | Mitigation |
|---|---|---|
| Embedding latency | Linear with turns | Batch encode all turns once; cache per conversation |
| Summarisation cost | One LLM call per COMPRESS run | Cap batch size; merge adjacent runs before calling |
| Landmark drift | Pattern-based detection degrades on unusual phrasing | Log and report recall separately; upgrade to embedding detector |
| Assembly integrity failures | Long conversations more likely to have anomalies | Integrity checker with repair logic; log all repairs |
| Context window exceeded | Summarised thread still too long | Hard token cap on output; trigger second summarisation pass |
| LLM detector cost explosion | Per-conversation call at 500 turns is expensive | Cap conversation length before LLM detection; chunk if needed |

---

## 13. Net Cost Analysis (Stretch)

```
Cost of optimisation =
    landmark_detection_cost          # $0 for rules/embedding; ~$0.02 for LLM
  + summarisation_calls × tokens × cost_per_token

Savings =
    (full_tokens - opt_tokens) × cost_per_token × production_calls_per_context

Net saving = Savings - Cost of optimisation
```

Break-even: if each optimised context is used for ≥2 downstream LLM calls, the system is net-positive with rule-based detection. With LLM detection, break-even requires ≥3–5 downstream calls.

---

## 14. Testing Strategy

| Test | Coverage |
|---|---|
| `test_ingestion.py` | Load, filter, normalise, slot extraction |
| `test_scoring.py` | Each component independently; composite; weight adjustments by query type |
| `test_landmarks.py` | Rule detector recall vs. known annotations; pass 2 alignment patterns; action item patterns |
| `test_compression.py` | Selection logic; assembly integrity; summary insertion |
| `test_evaluation.py` | Token counting; BERTScore; landmark recall calculation |

Integration test: full pipeline on 3 known conversations; assert token reduction in [0.3, 0.7] and valid LLM message format output.

---

## 15. Dependency Summary

| Package | Purpose | Local/API |
|---|---|---|
| `sentence-transformers` | Semantic embeddings (scoring + v2 landmark detector) | Local |
| `scikit-learn` | TF-IDF keyword scoring | Local |
| `bert-score` | BERTScore F1 evaluation | Local |
| `tiktoken` | Token counting | Local |
| `openai` | Summarisation + judge LLM + optional LLM detector | API |
| `pytest` | Test runner | Local |
| `pandas` | Evaluation results table | Local |
| `datasets` / `huggingface-hub` | Download only, not runtime | — |
