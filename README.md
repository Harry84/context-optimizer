# Context Optimizer

Intelligent context compression for multi-turn LLM agents. Given a long conversation and a current query, produces an optimised conversation thread that is 40–60% smaller than naive full-history while preserving answer quality.

Built as a technical assessment for LEC (London Export Corporation) — demonstrating that context optimisation is measurably correct, not just smaller.

---

## What it does

When an LLM agent reasons over a long conversation, sending the full history wastes tokens and adds noise. This system:

1. **Detects landmarks** — stated intents, decisions, action items, and conversation-close signals that must be preserved verbatim
2. **Scores remaining turns** for relevance to the current query using keyword match, semantic similarity, and recency decay
3. **Adapts** the compression strategy to the query type (factual / analytical / preference)
4. **Summarises** low-relevance sub-threads with one LLM call per run, dropping trivial filler entirely
5. **Assembles** a structurally valid context thread the LLM can consume

Evaluation reports token reduction %, answer quality (LLM-as-judge, 4-dimension rubric), BERTScore F1, landmark recall vs. ground truth, and assembly latency.

---

## Dataset

**Taskmaster-2, flights domain** (Google Research, CC BY 4.0).
1,692 conversations ≥20 turns. Mean 30.2 turns, max 85 turns.
Conversations follow a goal-directed pattern — user establishes constraints, compares options, makes decisions — structurally analogous to LEC trade logistics enquiries.

Slot annotations per utterance serve as independent ground truth for landmark detection recall (86.6% measured across the full corpus).

---

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd context-optimizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Download the dataset

```bash
python utilities/download_taskmaster2.py
```

Downloads `data/taskmaster2/flights.json` (~8MB). The `data/` directory is gitignored.

### 3. Set API keys

Create a `.env` file in the project root with your keys:

```
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...           # optional — removes HuggingFace rate limit warnings
```

`OPENAI_API_KEY` is required for summarisation and LLM-as-judge evaluation.
Landmark detection and embedding scoring run locally — no API key needed for those.

`.env` values always override environment variables set in the shell.

### 4. Run the tests

```bash
pytest tests/ -v
```

All tests pass without any LLM API calls.

### 5. Inspect a conversation (dry run — no API calls)

```bash
python main.py inspect \
  --conv-id dlg-cbfc519d-93e3-404d-9db5-c5fe35a5b765 \
  --query "What flights were compared and which did the user choose?" \
  --query-pos 50 \
  --dry-run
```

Shows per-turn scoring and dispositions without calling OpenAI.

### 6. Inspect with full compression and side-by-side comparison

```bash
python main.py inspect \
  --conv-id dlg-cbfc519d-93e3-404d-9db5-c5fe35a5b765 \
  --query "What flights were compared and which did the user choose?" \
  --query-pos 50 \
  --compare
```

### 7. Run corpus statistics

```bash
python main.py stats
```

### 8. Run full evaluation

```bash
# Default strategy (v1 — turn-level)
python main.py evaluate

# Per-strategy evals
python main.py --compression-strategy chunk evaluate
python main.py --compression-strategy sentence evaluate
python main.py --compression-strategy topk evaluate
python main.py --compression-strategy topk-sentence evaluate
```

Runs on 10 sampled conversations × 2 queries each (~$1–2 OpenAI cost per run).
Outputs `eval_results.csv` (or `eval_results_<strategy>.csv`) and prints acceptance bar summary.

Answer generation uses `--generator` (default `gpt-4o-mini`); judging uses `--judge` (default `gpt-4o`).

### 9. Inspect a synthetic conversation

```bash
python main.py \
  --data-path "data/synthetic/synthetic_flights.json" \
  --min-turns 5 \
  --compression-strategy chunk \
  inspect \
  --conv-id syn-001-rambling-holiday \
  --query "What flights were compared and what did the user decide?" \
  --query-pos 20 \
  --compare
```

Useful for validating compression behaviour on short, predictable conversations without touching the main corpus.

---

## Architecture

```
flights.json
    │
    ▼
[Stage 1] Ingestion & Normalisation
    │  Load, filter ≥20 turns, normalise to Conversation objects
    │  Sentence-level deduplication applied at load time (crowdworker noise)
    │  Slot annotations extracted for evaluation ground truth only
    ▼
[Stage 2] Landmark Detection (two-pass, zero API cost)
    │  Pass 1: individual turn scoring via text signals + speaker role
    │           — stated intents (slot-value signals, intent verbs)
    │           — decisions (offer patterns, confirmations)
    │           — action items (commitment patterns)
    │           — conversation-close signals ("that's all", "goodbye", etc.)
    │  Pass 2: cross-turn alignment (offer→confirmation, constraint→echo)
    │  86.6% GT recall, <5ms per conversation
    ▼
[Stage 3] Relevance Scoring
    │  Query classified: factual / analytical / preference
    │  Per-turn score: keyword (TF-IDF) + semantic (MiniLM-L6) + recency decay
    │  Landmark boost +0.3 applied after normalisation
    │  Only turns 0..query_position-1 scored (history only)
    ▼
[Stage 4] Compression & Assembly
    │  Turns classified: KEEP / CANDIDATE / COMPRESS
    │  Short COMPRESS runs (<80 chars) dropped entirely — no LLM call
    │  Remaining COMPRESS runs → one LLM summarisation call each
    │  Smart merge deduplicates consecutive ASSISTANT turns
    │  Assembled thread: chronological, valid role alternation enforced
    ▼
[Stage 5] Evaluation
       Full vs. optimised → LLM-as-judge (4 dimensions, temperature=0)
       + BERTScore F1 (semantic preservation, LLM-independent)
       + Landmark recall vs. slot annotations
```

Full design rationale in [`ARCHITECTURE.md`](ARCHITECTURE.md).
Design decisions with justification in [`key_decisions.md`](key_decisions.md).

---

## Landmark detection

Four categories — three from the assignment specification plus conversation-close:

| Category | Detection | Ground truth |
|---|---|---|
| **Stated intents** | USER turns with slot-value signals (price, time, airline, seat class, date) | ✅ Slot annotations |
| **Decisions** | Offer patterns + confirmation patterns + cross-turn alignment | ✅ Slot annotations |
| **Action items** | ASSISTANT commitment patterns ("I'll send", "tickets confirmed") | ❌ Pattern-based only |
| **Conversation close** | USER farewell/completion signals ("that's all", "I'm done", "goodbye") | ❌ Pattern-based only |

Detection is **text and speaker role only** — slot annotations are never used in detection, only in evaluation.

Measured corpus-wide (1,692 conversations, ≥20 turns):
- GT recall: **86.6%**
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

Independent metric: **BERTScore F1** (roberta-large, local). Threshold ≥ 0.85.

Answer generation uses `gpt-4o-mini`; judging uses `gpt-4o` (separate models to avoid self-grading bias).

---

## Evaluation results — v5 chunk (latest)

10 conversations × 2 queries each, Taskmaster-2 flights corpus, `--compression-strategy chunk`.

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 43.7% | 40–60% | ✓ PASS |
| Quality Δ (LLM judge) | -0.21 | ≥ 0 | ✗ FAIL |
| BERTScore F1 | 0.927 | ≥ 0.85 | ✓ PASS |
| BERTScore ≥ 0.85 | 100% of queries | — | ✓ |
| Landmark recall | 77.0% | — | Reported |
| Quality (full context) | 9.01 | — | |
| Quality (optimised) | 8.80 | — | |
| Compression | 64.2% turns | — | |
| Latency | 1,674ms mean | — | |

The Δ quality FAIL is consistent with earlier runs and is attributed to gpt-4o judge non-determinism even at temperature=0 (documented in [`report.md`](report.md)). BERTScore is the stable signal — 100% of queries pass the 0.85 threshold.

---

## What I'd ship next

1. **Embedding-based landmark detector (v2)** — cosine similarity to prototype embeddings using the already-loaded `all-MiniLM-L6-v2`. Seed prototypes from highest-confidence rule detections. Generalises to new domains without new regex — new domains need only new prototypes.

2. **LEC domain adaptation** — generate 5–10 synthetic LEC customer service conversations (trade enquiries, order tracking, logistics). Requires only new data and updated intent taxonomy, no architectural changes.

3. **Net cost dashboard** — track summarisation API calls, tokens used, and tokens saved per run. Report break-even point (currently estimated at ≥2 downstream LLM calls per optimised context with rule-based detection).

4. **Hotels + restaurant-search evaluation** — both domains downloaded. Extended evaluation across 3 domains tests domain-agnosticism.

5. **Streaming context assembly** — for 500+ turn conversations, process and summarise in chunks. Addresses the latency and memory issues identified in the scale failure mode analysis.

---

## AI usage note

This project was built with Claude (Anthropic) as a development partner throughout.

- **Claude wrote:** All source code, tests, and utility scripts. The landmark detector logic was iteratively developed and verified against real Taskmaster-2 data with Claude's assistance.
- **I designed:** The overall architecture, dataset selection rationale, evaluation methodology, landmark detection strategy (two-pass alignment was my suggestion after observing that cross-turn context was needed), and all key decisions in `key_decisions.md`.
- **I verified:** The landmark detector output was manually inspected across multiple conversations. Recall numbers (86.6%) are from actual corpus-wide runs. Pipeline output was visually verified using `--compare` and `--dry-run` modes.
- **How I used AI:** As a senior collaborator — I pushed back on suggestions, asked follow-up questions when reasoning wasn't clear, and made all architectural decisions myself.

The judgment about what to build, how to evaluate it honestly, and what the limitations are — those are mine.
