# Context Optimizer

Intelligent context compression for multi-turn LLM agents. Given a long conversation and a current query, produces an optimised conversation thread that is 40–60% smaller than naive full-history while preserving answer quality.

Built as a technical assessment for LEC (London Export Corporation) — demonstrating that context optimisation is measurably correct, not just smaller.

---

## What it does

When an LLM agent reasons over a long conversation, sending the full history wastes tokens and adds noise. This system:

1. **Detects landmarks** — stated intents, decisions, and action items that must be preserved verbatim
2. **Scores remaining turns** for relevance to the current query using keyword match, semantic similarity, and recency decay
3. **Adapts** the compression strategy to the query type (factual / analytical / preference)
4. **Summarises** low-relevance sub-threads with one LLM call per run
5. **Assembles** a structurally valid context thread the LLM can consume

Evaluation reports token reduction %, answer quality (LLM-as-judge, 4-dimension rubric), BERTScore F1, landmark recall vs. ground truth, and assembly latency.

---

## Dataset

**Taskmaster-2, flights domain** (Google Research, CC BY 4.0).  
17,289 task-oriented dialogues; 1,692 meet the ≥20 turn threshold in the flights domain.  
Conversations follow a goal-directed pattern — user establishes constraints, compares options, makes decisions — structurally analogous to LEC trade logistics enquiries.

Slot annotations per utterance serve as independent ground truth for landmark detection recall (86.6% measured across the full corpus).

---

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd context-optimizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Download the dataset

```bash
python utilities/download_taskmaster2.py
```

This downloads `data/taskmaster2/flights.json` (~8MB). The `data/` directory is gitignored.

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...     # Windows: set OPENAI_API_KEY=sk-...
```

Required for summarisation and LLM-as-judge. Not required for landmark detection or embedding scoring (both run locally).

### 4. Run the tests

```bash
pytest tests/ -v
```

Tests do not make LLM API calls — all LLM-dependent code is excluded from the test suite.

### 5. Inspect a conversation

```bash
python main.py inspect \
  --conv-id dlg-cbfc519d-93e3-404d-9db5-c5fe35a5b765 \
  --query "What flights were compared and which did the user choose?" \
  --query-pos 50
```

### 6. Run corpus statistics

```bash
python main.py stats
```

### 7. Run full evaluation

```bash
python main.py evaluate
```

Runs on 10 sampled conversations × 2 queries each. Outputs a results table to `eval_results.csv` and prints acceptance bar results.

---

## Architecture

```
flights.json
    │
    ▼
[Stage 1] Ingestion & Normalisation
    │  Load, filter ≥20 turns, normalise to Conversation objects
    │  Slot annotations extracted for evaluation ground truth only
    ▼
[Stage 2] Landmark Detection (two-pass)
    │  Pass 1: individual turn scoring via text signals + speaker role
    │  Pass 2: cross-turn alignment (offer→confirmation, constraint→echo)
    │  86.6% GT recall, zero API cost, <5ms per conversation
    ▼
[Stage 3] Relevance Scoring
    │  Query classified: factual / analytical / preference
    │  Per-turn score: keyword (TF-IDF) + semantic (MiniLM-L6) + recency decay
    │  Landmark boost +0.3 applied after normalisation
    │  Only turns 0..query_position-1 scored (history only)
    ▼
[Stage 4] Compression & Assembly
    │  Turns classified: KEEP / CANDIDATE / COMPRESS
    │  Contiguous COMPRESS runs → single LLM summarisation call
    │  Assembled thread: chronological, valid role alternation
    ▼
[Stage 5] Evaluation
       Full vs. optimised → LLM-as-judge (4 dimensions, order-swapped)
       + BERTScore F1 (semantic preservation, LLM-independent)
       + Landmark recall vs. slot annotations
```

Full design rationale in [`ARCHITECTURE.md`](ARCHITECTURE.md).  
Design decisions with justification in [`key_decisions.md`](key_decisions.md).

---

## Landmark detection

Three categories per the assignment specification:

| Category | Detection | Ground truth |
|---|---|---|
| **Stated intents** | USER turns with slot-value signals (price, time, airline, seat class, date) | ✅ Taskmaster-2 slot annotations |
| **Decisions** | Detail-slot turns + confirmation patterns + cross-turn alignment | ✅ Taskmaster-2 slot annotations |
| **Action items** | ASSISTANT commitment patterns ("I'll send", "tickets confirmed") | ❌ Pattern-based only |

Detection is **text and speaker role only** — slot annotations are never used in detection, only in evaluation. This ensures the detector generalises to unannotated datasets.

Measured corpus-wide (1,692 conversations, ≥20 turns):
- GT recall: **86.6%** (20,092 / 23,190 slot-annotated turns detected)
- Landmark rate: **46.4%** of all turns
- Compressible: **53.6%** of all turns

---

## Adaptive compression

Query type drives compression aggressiveness:

| Query Type | Keyword w | Semantic w | Recency w | High thresh | Low thresh |
|---|---|---|---|---|---|
| Factual | 0.3 | 0.5 | 0.2 | 0.6 | 0.3 |
| Analytical | 0.2 | 0.4 | 0.4 | 0.5 | 0.25 |
| Preference | 0.2 | 0.3 | 0.5 | 0.45 | 0.2 |

---

## Evaluation rubric

LLM-as-judge, 4 dimensions (1–10 each), temperature=0:

| Dimension | What is assessed |
|---|---|
| Correctness | Factual claims accurate relative to conversation |
| Completeness | All parts of the query addressed |
| Landmark consistency | Stated intents, decisions, action items respected |
| Hallucination | No ungrounded information introduced (10 = none) |

Bias mitigations: each answer pair judged independently against its own context; temperature=0; judge model differs from generation model.

Independent metric: **BERTScore F1** (roberta-large, local) between full-context and optimised-context answers. Threshold ≥ 0.85.

---

## Project structure

```
context_optimizer/
├── src/
│   ├── ingestion/        # loader.py, models.py
│   ├── landmarks/        # detector.py (interface), rule_detector.py (v1)
│   ├── scoring/          # scorer.py, keyword.py, semantic.py, recency.py, query_classifier.py
│   ├── compression/      # compressor.py, summariser.py, assembler.py, pipeline.py
│   └── evaluation/       # harness.py, judge.py, bertscore_metric.py, landmark_recall.py
├── tests/                # pytest test suite (no LLM calls)
├── utilities/            # inspection and download scripts
├── data/                 # gitignored — download separately
├── main.py               # CLI entry point
├── PRD.md                # Product Requirements Document
├── ARCHITECTURE.md       # Architecture decisions with rationale
└── key_decisions.md      # Running log of design decisions with justification
```

---

## Configuration

All parameters in `OptimizerConfig` (see `src/ingestion/models.py`):

```python
config = OptimizerConfig(
    landmark_detector   = "rules",           # "rules" | "embedding" | "llm"
    embedding_model     = "all-MiniLM-L6-v2",
    summarisation_model = "gpt-4o-mini",
    judge_model         = "gpt-4o",
    min_turns           = 20,
    lambda_decay        = 0.05,
    landmark_boost      = 0.3,
)
```

Landmark detector is pluggable. `"embedding"` (v2, prototype similarity) and `"llm"` (batch classification, ~95% recall) are documented upgrade paths — see `ARCHITECTURE.md §5` and `key_decisions.md KD-011`.

---

## What I'd ship next

1. **Embedding-based landmark detector (v2)** — replace regex patterns with cosine similarity to prototype embeddings using the already-loaded `all-MiniLM-L6-v2`. Seed prototypes from highest-confidence rule detections. Generalises to new domains without new code — new domains need only new prototypes.

2. **LEC domain adaptation** — generate 5–10 synthetic LEC customer service conversations (trade enquiries, order tracking, logistics) and run the pipeline against them. Requires only new data and updated intent taxonomy, no architectural changes.

3. **Net cost dashboard** — track summarisation API calls, tokens used, and tokens saved per evaluation run. Report break-even point (currently estimated at ≥2 downstream LLM calls per optimised context).

4. **Hotels + restaurant-search evaluation** — both domains already downloaded. Extended evaluation across 3 domains tests domain-agnosticism of the pipeline.

5. **Streaming context assembly** — for 500+ turn conversations, process and summarise in chunks rather than loading all turns into memory. Addresses the latency explosion identified in the scale failure mode analysis.

---

## AI usage note

This project was built with Claude (Anthropic) as a development partner throughout. Specifically:

- **Claude wrote:** All source code in `src/`, tests in `tests/`, and the utility scripts in `utilities/`. The landmark detector logic was iteratively developed and verified against real Taskmaster-2 data with Claude's assistance.
- **I designed:** The overall architecture, dataset selection rationale, evaluation methodology, landmark detection strategy (two-pass alignment was my suggestion after observing that cross-turn context was needed), and all key decisions documented in `key_decisions.md`.
- **I verified:** The landmark detector output was manually inspected across 5+ conversations before being ported to production code. The recall numbers (86.6%) are from actual corpus-wide runs, not estimates.
- **How I used AI:** As a senior collaborator — I pushed back on suggestions (e.g. the OASST1 dataset choice, slot-annotation-based detection, the 50-turn threshold), asked follow-up questions when reasoning wasn't clear, and made all architectural decisions myself. The AI-usage note itself is mine.

The judgment about what to build, how to evaluate it honestly, and what the limitations are — those are mine.
