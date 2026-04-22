# Architecture Decisions Document
## Intelligent Context Optimizer for Multi-Turn Agents

**Version:** 1.2
**Date:** April 2026
**Status:** Approved for Implementation

---

## 1. System Overview

The Context Optimizer takes a multi-turn conversation and a current query, and returns a compressed conversation thread that preserves the information the LLM needs to answer the query — and discards what it doesn't.

```
Input:  Conversation (N turns) + Query string (at position Q in the conversation)
Output: Optimised conversation thread covering turns 0..Q-1 (40–60% fewer tokens)
```

**Critical constraint:** Only turns *before* the current query position are considered. Turn Q itself is the query being answered — it is not part of the context window. This mirrors real agent behaviour: the LLM sees prior history and the current query, not future turns.

The system is a five-stage pipeline:

```
Ingestion → Landmark Detection → Relevance Scoring → Compression & Assembly → Evaluation
```

Each stage has a clean interface. Components are swappable without touching adjacent stages. The landmark detector in particular has a pluggable interface so it can be upgraded from rules → embeddings → LLM without changing the rest of the pipeline.

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
│   │   └── models.py                 # Conversation, Turn, OptimizerConfig dataclasses
│   ├── scoring/
│   │   ├── scorer.py                 # Composite relevance scorer (orchestrates below)
│   │   ├── keyword.py                # TF-IDF keyword match
│   │   ├── semantic.py               # Embedding cosine similarity
│   │   ├── recency.py                # Exponential decay
│   │   └── query_classifier.py       # Factual / analytical / preference
│   ├── landmarks/
│   │   ├── detector.py               # Pluggable landmark detector interface
│   │   ├── rule_detector.py          # v1: rule-based + two-pass alignment
│   │   ├── embedding_detector.py     # v2: prototype embedding similarity
│   │   └── llm_detector.py           # optional: LLM-based batch classification
│   ├── compression/
│   │   ├── compressor.py             # Score turns, group into runs, classify each run
│   │   ├── summariser.py             # LLM summarisation of COMPRESS runs
│   │   └── assembler.py              # Assemble final thread, integrity check
│   └── evaluation/
│       ├── harness.py                # Run full vs. optimised, collect all metrics
│       ├── judge.py                  # LLM-as-judge with order-swap bias mitigation
│       ├── bertscore_metric.py       # BERTScore F1 computation
│       └── landmark_recall.py        # Recall vs. slot annotation ground truth
├── utilities/
│   ├── view_conversation.py
│   ├── view_domain.py
│   ├── corpus_stats.py
│   ├── verify_classifiers.py         # Visual verification of detector output
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

```python
@dataclass
class Turn:
    turn_index: int
    speaker: str                      # "USER" or "ASSISTANT"
    text: str
    slots: list[str]                  # Slot names from Taskmaster-2 annotations
                                      # (evaluation ground truth ONLY — never used in detection)
    is_landmark: bool = False         # Set by landmark detector
    landmark_type: str | None = None  # "intent" | "decision" | "action_item"
    landmark_reason: str = ""         # Human-readable reason (auditable)
    promoted: bool = False            # True if promoted by cross-turn alignment pass
    score: float = 0.0                # Set by relevance scorer against current query
    disposition: str = ""             # "KEEP" | "COMPRESS" | "CANDIDATE" — set by compressor

@dataclass
class Conversation:
    conversation_id: str
    instruction_id: str
    turns: list[Turn]
    domain: str = "flights"

@dataclass
class OptimizerConfig:
    landmark_detector: str = "rules"       # "rules" | "embedding" | "llm"
    query_classifier: str  = "rules"       # "rules" | "llm"
    lambda_decay: float    = 0.05
    landmark_boost: float  = 0.3
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
    embedding_model:      str = "all-MiniLM-L6-v2"
    summarisation_model:  str = "gpt-4o-mini"
    judge_model:          str = "gpt-4o"
    min_turns:            int = 20
    data_path:            str = "data/taskmaster2/flights.json"
```

The `disposition` field on `Turn` is new — it makes the compressor's decision explicit and auditable on each turn, not just implied by whether it appears in the output.

---

## 4. Stage 1 — Ingestion & Normalisation

**Module:** `src/ingestion/loader.py`

Load Taskmaster-2 JSON, filter by turn count, normalise to `Conversation` objects. Slot annotations extracted at load time and stored on each `Turn` for evaluation use only.

```python
def load_corpus(path: str, min_turns: int = 20) -> list[Conversation]: ...

def normalise_turn(raw: dict, index: int) -> Turn:
    slots = [
        ann["name"]
        for seg in raw.get("segments", [])
        for ann in seg.get("annotations", [])
    ]
    return Turn(turn_index=index, speaker=raw["speaker"], text=raw["text"], slots=slots)
```

No tree reconstruction required. Taskmaster-2 conversations are pre-structured linear sequences.

---

## 5. Stage 2 — Landmark Detection

**Module:** `src/landmarks/`

Classifies each turn as landmark or not *before* relevance scoring, so landmark boost can be applied. The active detector is selected via `OptimizerConfig.landmark_detector`.

### 5.1 Detector Interface (pluggable)

```python
class LandmarkDetector(Protocol):
    def detect(self, conversation: Conversation) -> Conversation:
        """Annotate all turns in-place. Returns the same conversation."""
        ...
```

### 5.2 v1 — Rule-Based + Two-Pass Alignment (`rule_detector.py`)

**Pass 1:** Score each turn individually using text signals.
- USER turn with slot-value signal (price, time, airline, seat class, date, stops) → `stated_intent` candidate
- ASSISTANT turn with offer signal (specific values, "I found", "you will leave at") → `decision` candidate
- ASSISTANT turn with commitment pattern ("I'll send", "tickets confirmed") → `action_item`
- Short pure filler ("okay", "sure", "hold on") with no slot signal → always compressible

**Pass 2:** Promote turns based on cross-turn alignment.
- Pattern A (Offer→Confirmation): ASSISTANT[i] makes offer AND USER[i+1] gives weak confirmation → both `decision`
- Pattern B (Constraint→Echo): USER[i] has slot signal AND ASSISTANT[i+1] echoes it back → both `intent`

**Measured: 86.6% GT recall, 46.4% landmark rate, 53.6% compressible.**

### 5.3 v2 — Embedding Similarity (`embedding_detector.py`)

Cosine similarity to prototype embeddings per landmark category. Uses `all-MiniLM-L6-v2` already loaded for scoring. Handles domain vocabulary shifts without new regex. New domains need only new prototypes.

### 5.4 LLM Batch Detection (`llm_detector.py`)

One LLM call per conversation, all turns classified in a single structured prompt. ~92–95% estimated recall. Enabled via `LANDMARK_DETECTOR=llm`. Higher cost ($0.01–0.03/conversation) — see KD-011 for full cost analysis.

---

## 6. Stage 3 — Relevance Scoring

**Module:** `src/scoring/scorer.py`

**Critical:** Scoring operates only on turns 0..Q-1 where Q is the position of the current query in the conversation. Turn Q and any subsequent turns are excluded — the optimizer only sees history available to the agent at query time.

### 6.1 Query Classification

```python
def classify_query(query: str) -> Literal["factual", "analytical", "preference"]: ...
```

Rule-based (LLM fallback via `QUERY_CLASSIFIER=llm`). Classification drives weight selection for all downstream scoring.

### 6.2 Scoring Each Turn

For each turn t in turns[0..Q-1], compute composite score against query q:

```
S(t, q) = w1 · keyword(t, q)
         + w2 · semantic(t, q)
         + w3 · recency(t, Q)
         + landmark_boost(t)
```

Where:
- `keyword(t, q)` — TF-IDF cosine similarity; vectoriser fitted on turns[0..Q-1] so IDF reflects this conversation's vocabulary
- `semantic(t, q)` — cosine similarity of `all-MiniLM-L6-v2` embeddings; turn embeddings computed once and cached
- `recency(t, Q)` — `exp(-λ · (Q - t.turn_index))` where Q is current query position, λ=0.05; turns closer to Q score higher
- `landmark_boost(t)` — +0.3 if `t.is_landmark`, 0.0 otherwise

**Weights by query type:**

| Query Type | w1 (keyword) | w2 (semantic) | w3 (recency) |
|---|---|---|---|
| Factual | 0.3 | 0.5 | 0.2 |
| Analytical | 0.2 | 0.4 | 0.4 |
| Preference | 0.2 | 0.3 | 0.5 |

Scores are normalised to [0, 1] after weighting. Landmark boost is applied after normalisation.

---

## 7. Stage 4 — Compression & Assembly

**Module:** `src/compression/`

### 7.1 Turn Classification (`compressor.py`)

After scoring, each turn receives a `disposition`:

```python
def classify_turns(
    turns: list[Turn],
    query_type: str,
    config: OptimizerConfig,
) -> list[Turn]:
    high = config.thresholds[query_type]["high"]
    low  = config.thresholds[query_type]["low"]

    for turn in turns:
        if turn.is_landmark:
            turn.disposition = "KEEP"           # hard rule — always verbatim
        elif turn.score >= high:
            turn.disposition = "KEEP"
        elif turn.score >= low:
            turn.disposition = "CANDIDATE"      # kept for structural integrity
        else:
            turn.disposition = "COMPRESS"
    return turns
```

Thresholds by query type:

| Query Type | HIGH | LOW |
|---|---|---|
| Factual | 0.6 | 0.3 |
| Analytical | 0.5 | 0.25 |
| Preference | 0.45 | 0.2 |

### 7.2 Run Grouping

Contiguous turns with the same disposition are grouped into runs. This is the unit of summarisation:

```python
def group_into_runs(turns: list[Turn]) -> list[tuple[str, list[Turn]]]:
    """
    Group consecutive turns by disposition.
    Returns list of (disposition, [turns]) tuples in chronological order.

    Example:
        turns: KEEP KEEP COMPRESS COMPRESS COMPRESS KEEP COMPRESS
        runs:  [("KEEP", [t0,t1]),
                ("COMPRESS", [t2,t3,t4]),
                ("KEEP", [t5]),
                ("COMPRESS", [t6])]
    """
    runs = []
    for turn in turns:
        if runs and runs[-1][0] == turn.disposition:
            runs[-1][1].append(turn)
        else:
            runs.append((turn.disposition, [turn]))
    return runs
```

Each COMPRESS run becomes exactly one LLM summarisation call. Grouping before calling minimises API calls and cost.

### 7.3 Summarisation (`summariser.py`)

```python
def summarise_run(turns: list[Turn], domain: str) -> str:
    """One LLM call per COMPRESS run."""
    prompt = f"""These turns are from a {domain} booking conversation.
Summarise in 1-2 sentences. Preserve any constraints, options, or prices mentioned.
Do not invent information not in the turns.

{format_turns(turns)}

Summary:"""
    ...
```

### 7.4 Assembly (`assembler.py`)

```python
def assemble(runs: list[tuple[str, list[Turn]]], summaries: dict[int, str]) -> list[dict]:
    """
    Build final [{role, content}] thread from runs.
    KEEP/CANDIDATE runs → verbatim turns in order
    COMPRESS runs → single synthetic ASSISTANT [SUMMARY: ...] turn
    """
    thread = []
    for disposition, turns in runs:
        if disposition == "COMPRESS":
            run_id = id(turns)
            thread.append({
                "role": "assistant",
                "content": f"[SUMMARY: {summaries[run_id]}]"
            })
        else:
            for turn in turns:
                role = "user" if turn.speaker == "USER" else "assistant"
                thread.append({"role": role, "content": turn.text})

    return integrity_check(thread)
```

**Integrity check** (`assembler.py`): validates no consecutive same-role turns, no orphaned user turns without preceding assistant context. Repairs logged and counted for evaluation report.

### 7.5 Full Compression Pipeline

```python
def compress(
    conversation: Conversation,
    query: str,
    query_position: int,
    config: OptimizerConfig,
) -> list[dict]:
    """
    Full compression pipeline for a single (conversation, query) pair.

    query_position: index of the query turn in the conversation.
                    Only turns[0..query_position-1] are used as context.
    """
    # 1. Slice to history only
    history = conversation.turns[:query_position]

    # 2. Score all history turns against the query
    query_type = classify_query(query)
    scored     = score_turns(history, query, query_type, config)

    # 3. Classify each turn's disposition
    classified = classify_turns(scored, query_type, config)

    # 4. Group into contiguous runs
    runs = group_into_runs(classified)

    # 5. Summarise COMPRESS runs (one LLM call per run)
    summaries = {}
    for disposition, turns in runs:
        if disposition == "COMPRESS":
            summaries[id(turns)] = summarise_run(turns, conversation.domain)

    # 6. Assemble final thread
    return assemble(runs, summaries)
```

The `query_position` parameter is the explicit enforcement of the "turns 0..Q-1 only" constraint. It is always passed by the evaluation harness and cannot be omitted.

---

## 8. Stage 5 — Evaluation Harness

**Module:** `src/evaluation/harness.py`

For each conversation, the harness selects evaluation queries from *within* the conversation at positions where the answer requires understanding of prior context. This ensures the evaluation is testing something real — not asking questions whose answer is entirely in the query turn itself.

```python
for conv in eval_set:                          # ≥10 conversations
    for q_idx, query in eval_queries[conv]:    # 2-3 queries per conversation
                                               # q_idx = turn index of query
        # Full context: all turns before query
        full_thread = format_full(conv.turns[:q_idx])

        # Optimised context: compressed turns before query
        opt_thread  = compress(conv, query, query_position=q_idx, config=config)

        # Generate answers using both contexts
        A_full = llm.answer(full_thread, query)
        A_opt  = llm.answer(opt_thread,  query)

        # Metrics
        token_reduction = (token_count(full_thread) - token_count(opt_thread)) \
                          / token_count(full_thread)
        quality_full    = judge.score(query, A_full, conv)   # order-swap applied
        quality_opt     = judge.score(query, A_opt,  conv)
        bertscore_f1    = bertscore(A_full, A_opt)
        lm_recall       = landmark_recall(conv.turns[:q_idx])
        latency_ms      = time_ms(compress, conv, query, q_idx, config)
```

### 8.1 Query Selection

Queries are selected or constructed to require information from prior turns:
- **Factual:** "What price was quoted for the nonstop flight?" (answer is in an earlier turn)
- **Analytical:** "Why did the user switch from business to economy class?" (requires earlier context)
- **Preference:** "Based on the conversation, which option would you recommend?" (requires comparison turns)

For Taskmaster-2 evaluation, queries are constructed semi-automatically: for each conversation, we identify the turn where the final booking is confirmed, then construct queries whose answers depend on earlier constraint-setting and comparison turns.

### 8.2 LLM-as-Judge

4-dimension rubric (1–10 each), temperature=0, order-swapped:
- **Correctness** — factual accuracy relative to conversation
- **Completeness** — addresses all parts of the query
- **Landmark consistency** — respects stated intents, decisions, action items
- **Hallucination** — no ungrounded information (10 = none)

### 8.3 Metrics Summary

| Metric | Method | Acceptance bar |
|---|---|---|
| Token reduction % | `(full - opt) / full` | 40–60% |
| Quality (LLM judge) | Mean of 4 dimensions, order-swapped | opt ≥ full |
| BERTScore F1 | `roberta-large`, local | ≥ 0.85 |
| Landmark recall | `|detected ∩ GT| / |GT|` | Reported, not a pass/fail |
| Assembly latency | Wall-clock ms | Reported |

---

## 9. Stage Interfaces

```python
# Stage 1
load_corpus(path, min_turns)                    -> list[Conversation]

# Stage 2 — landmark detection (in-place)
detector.detect(conversation)                   -> Conversation

# Stage 3 — relevance scoring (in-place, history slice only)
score_turns(turns, query, query_type, config)   -> list[Turn]

# Stage 4 — compression (returns new structure)
compress(conversation, query, query_position, config) -> list[dict]

# Stage 5 — evaluation
evaluate(conversations, eval_queries, config)   -> pd.DataFrame
```

---

## 10. Rejected Alternatives

### 10.1 RAG-style Retrieval

Pure embedding retrieval ignores turn order and cannot enforce structural constraints. Cannot preserve landmark turns that are dissimilar to the query text (e.g. "Yes." confirming a decision). **Rejected.**

### 10.2 Sliding Window Truncation

Drops critical early context. The user's original destination request is in turn 3; without it, "which flight did we choose?" is unanswerable. **Rejected.**

### 10.3 Learned Scoring Model

Evaluation set too small to train without overfitting. Additive weighted sum is transparent and appropriate at this scale. **Rejected for v1.**

### 10.4 Summarise Everything

Summaries lose precision on exact values (prices, dates, flight numbers). Verbatim landmark preservation is non-negotiable. **Rejected.**

### 10.5 LLM or Embeddings as Default Landmark Detector

86.6% measured recall at zero cost is the right v1 default. LLM detection adds $0.01–0.10/conversation — see §5.4 and KD-011. Embedding similarity is the highest-priority v2 upgrade.

---

## 11. What Breaks at Scale (500+ Turns)

| Failure Mode | Root Cause | Mitigation |
|---|---|---|
| Embedding latency | Linear with turns | Batch encode once per conversation; cache |
| Summarisation cost | One call per COMPRESS run | Cap batch size; merge short adjacent runs |
| Landmark detection drift | Pattern-based degrades on unusual phrasing | Log recall; upgrade to embedding detector |
| Assembly integrity failures | More anomalies in long conversations | Integrity checker with repair logic; log repairs |
| Context window exceeded | Summarised thread still too long | Hard token cap; trigger second summarisation pass |
| LLM detector cost at scale | Per-conversation cost × 500 turns | Chunk conversation; cap before LLM detection |

---

## 12. Net Cost Analysis

```
Cost    = landmark_detection_cost + (summarisation_runs × avg_tokens × cost/token)
Savings = (full_tokens - opt_tokens) × cost/token × downstream_calls_per_context

Net     = Savings - Cost
```

Break-even: ≥2 downstream calls per optimised context with rule-based detection. With LLM detection, ≥3–5 downstream calls.

---

## 13. Testing Strategy

| Test | Coverage |
|---|---|
| `test_ingestion.py` | Load, filter, normalise, slot extraction |
| `test_scoring.py` | Each component; composite; weight adjustments; query position slicing |
| `test_landmarks.py` | Rule detector; pass 2 alignment; action item patterns |
| `test_compression.py` | Run grouping; selection logic; assembly integrity; summary insertion |
| `test_evaluation.py` | Token counting; BERTScore; landmark recall; query position enforcement |

Integration test: full pipeline on 3 known conversations; assert token reduction in [0.3, 0.7] and output is valid LLM message format.

---

## 14. Dependencies

| Package | Purpose | Local/API |
|---|---|---|
| `sentence-transformers` | Semantic embeddings (scoring + v2 detector) | Local |
| `scikit-learn` | TF-IDF keyword scoring | Local |
| `bert-score` | BERTScore F1 | Local |
| `tiktoken` | Token counting | Local |
| `openai` | Summarisation + judge + optional LLM detector | API |
| `pandas` | Evaluation results table | Local |
| `pytest` | Test runner | Local |
| `datasets` / `huggingface-hub` | Download only, not runtime | — |
