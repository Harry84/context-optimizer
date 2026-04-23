# Intelligent Context Optimizer — Written Report
## Technical Assessment, April 2026

---

## What I Built

The Context Optimizer takes a multi-turn conversation and a current query and returns a compressed conversation thread that preserves the information an LLM needs to answer the query — and discards what it doesn't. The pipeline has five stages: ingestion and normalisation, landmark detection, relevance scoring, compression and assembly, and evaluation. Five compression strategies are implemented: turn-level (v1), sentence-level (v2), top-K retrieval (v3), top-K sentence (v4), and chunk-based retrieval (v5).

**Landmark detection** runs first, before scoring, using a two-pass rule-based detector. Pass 1 classifies each turn individually by text signals and speaker role — stated intents (slot-value signals: price, date, airline, seat class), decisions (offer patterns, strong confirmations, cross-turn offer→confirmation pairs), and action items (commitment verbs). Pass 2 promotes adjacent turn pairs using cross-turn alignment: an ASSISTANT offer followed by a USER weak confirmation ("yes", "okay") promotes both to `decision`; a USER constraint echoed back by the ASSISTANT promotes both to `intent`. Landmarks are hard-preserved in v1/v2 — they bypass the relevance scorer and are always kept verbatim. In v3–v5, landmarks receive a score boost but compete in the same top-K pool — query-irrelevant landmarks can be compressed. Measured against Taskmaster-2 slot annotations as ground truth: 86.6% recall across 1,692 conversations.

**Relevance scoring** classifies the query as factual, analytical, or preference — this drives different weight profiles across keyword (TF-IDF), semantic (MiniLM-L6-v2 cosine similarity), and recency (exponential decay) components. Landmarks receive a +0.3 boost after normalisation. Only history turns (0..query\_position-1) are scored; future turns are never seen.

**Compression — v1 (turn-level):** classifies each turn as KEEP, CANDIDATE, or COMPRESS using score thresholds, groups consecutive COMPRESS turns into runs, and makes one gpt-4o-mini summarisation call per run (capped at ≤15 words). Default strategy.

**Compression — v2 (sentence-level):** for landmark turns, splits the text into sentences using NLTK's punkt tokeniser, re-runs landmark pattern matching at sentence level, and scores non-landmark sentences independently against the query. Sentences with the same effective disposition from the same turn are merged back before assembly to prevent structural fragmentation.

**Compression — v3 (top-K retrieval):** non-landmark turns ranked by composite score descending; top K% kept, rest compressed. Landmarks receive a boost but are not hard-KEEPed. A noise floor (`topk_min_score=0.30`) prevents low-quality turns filling K slots.

**Compression — v4 (top-K sentence):** combines v2 sentence splitting with v3 top-K. All units (landmark sentences and non-landmark turns) scored against the query and ranked together. Landmarks receive a scaled boost (boost × individual score) to prevent query-irrelevant landmarks from inflating their score.

**Compression — v5 (chunk-based retrieval):** scores overlapping multi-turn chunks (chunk_size=6, stride=2) against the query rather than individual turns. Each turn's score = 0.7 × max chunk score + 0.3 × individual score. The chunk component captures answer-spanning relevance — turns that are part of a highly relevant multi-turn exchange score well even if they individually seem weak. The individual component prevents irrelevant turns from riding a high chunk score. Landmark boost is scaled by individual score. A query-gated airport floor (score ≥ 0.45) is applied to turns mentioning IATA codes or airport names when the query is airport-related.

**Assembly** enforces structural validity across all strategies: consecutive same-role turns are merged or bridged, and the thread is guaranteed to start with a user turn.

**Evaluation** runs across 10 conversations × 2 queries each, with per-conversation query selection from a 14-item pool. Both full-context and optimised answers are generated and evaluated by a gpt-4o judge on a 4-dimension rubric (correctness, completeness, landmark consistency, hallucination). BERTScore F1 (roberta-large, local) provides a deterministic independent quality signal.

---

## Results

**v1 evaluation** — 10 conversations (≥50 turns, Taskmaster-2 flights domain), 20 query pairs:

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 24.1% | 40–60% | ✗ |
| Quality Δ (LLM judge) | +0.01 | ≥ 0 | ✓ |
| BERTScore F1 | 0.940 | ≥ 0.85 | ✓ |
| BERTScore ≥ 0.85 | 100% of queries | — | ✓ |
| Landmark recall | 77.0% | — | Reported |
| Latency | 628ms mean | — | Reported |

**v2 spot-check** — single realistic 20-turn synthetic conversation:

| Metric | Result |
|---|---|
| Token reduction | 34.1% |
| Turn reduction | 40% (20 → 12 turns) |
| Latency | ~10s (CPU) |

**v3 evaluation** — same 10 conversations, proportional top-K:

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 21.0% | 40–60% | ✗ |
| Quality Δ (LLM judge) | -0.06 | ≥ 0 | ✗ |
| BERTScore F1 | 0.952 | ≥ 0.85 | ✓ |
| Latency | 462ms mean | — | Reported |

**v5 evaluation** — same 10 conversations, chunk-based retrieval (chunk_size=6, stride=2):

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 43.7% | 40–60% | ✓ |
| Quality Δ (LLM judge) | -0.21* | ≥ 0 | ✗ |
| BERTScore F1 | 0.927 | ≥ 0.85 | ✓ |
| BERTScore ≥ 0.85 | 100% of queries | — | ✓ |
| Quality (full context) | 9.01 | — | |
| Quality (optimised) | 8.80 | — | |
| Latency | 1,674ms mean | — | Reported |

*Prior runs reported Δ of -0.31 to -0.56, but those used the same model (gpt-4o) for both answer generation and judging — a methodology error. These results use gpt-4o-mini for generation and gpt-4o for judging (separate models). The full-context quality score dropped from ~9.5 to 9.01 as a result — expected, since gpt-4o-mini produces weaker answers than gpt-4o. The Δ quality figure remains an unreliable signal due to gpt-4o judge non-determinism at temperature=0; BERTScore is the stable indicator — 0.927 with 100% of queries passing the 0.85 threshold.

**v5 is the first strategy to hit the 40–60% token reduction target on the real Taskmaster-2 corpus.** v1 remains the recommended default for production use because its failure mode is conservative.

---

## What Broke and Why

**Token reduction shortfall (v1, v2, v3).** The 40–60% target is not met with threshold-based or fixed top-K strategies. The root cause is Taskmaster-2's unusually high landmark density (~46% of turns flagged) — slot values are repeated verbatim on nearly every turn in crowdworker transcripts. Landmark turns hold ~67% of total tokens. Even compressing every non-landmark turn to zero, maximum achievable reduction is ~33%.

**Top-K quality regression (v3, v5).** Two conversations consistently produce Δ quality ≈ -3 to -6 on airport queries — the relevant airport turns scored below the top-K threshold and were compressed away. The root cause is that chunk scoring captures multi-turn relevance well for most query types, but airport-related content can score lower than other high-relevance chunks when the chunk containing the airport mention also contains unrelated filler. The query-gated airport floor partially mitigates this (improving the worst case from Δ-6.25 to Δ-2.75) but does not eliminate it. This is a known limitation documented in key_decisions.md.

**LLM judge variance.** The gpt-4o judge is non-deterministic even at temperature=0 due to floating-point non-determinism in the backend. Full-context quality scores fluctuated by up to 0.3 points between identical runs, making the mean Δ quality unstable. This is a fundamental limitation of LLM-as-judge evaluation. BERTScore is the stable signal and should be weighted accordingly.

**v2 latency.** Sentence-level scoring produces ~10s latency on CPU. Fixable by reusing turn-level embeddings from the scorer rather than re-encoding at sentence level.

**Judge JSON parsing (fixed).** Initial evaluation returned all 5.0 scores due to missing `response_format=json_object`. Fixed.

**Fixed evaluation queries (fixed).** Initial evaluation used the same two queries for every conversation. Fixed by per-conversation query selection from a 14-item pool.

---

## What I Learned

The most important finding is about what makes a scoring signal useful for compression. Individual turn scoring (v1–v3) fails when the relevant content is spread across multiple turns — no single turn scores high enough to survive top-K. Chunk scoring (v5) solves this by scoring multi-turn windows, so a turn saying "Okay I'm going with Virgin" that individually scores 0.49 scores 0.93 as part of a chunk that includes the flight options and deliberation. This is a meaningful architectural insight: **the unit of relevance is not a turn, it is an exchange**.

The second finding is about failure mode asymmetry. Conservative compression (v1, v2) over-keeps — it preserves more context than necessary but never drops critical content. Aggressive compression (v3, v4, v5) under-keeps — it can drop critical content when that content doesn't rank in the top K. For a production system serving real users, conservative failure is strongly preferable.

The LLM judge variance finding is important for anyone building evaluation pipelines: at temperature=0, gpt-4o still produces different scores across runs for identical inputs. Multiple runs and averaged scores are needed for stable Δ quality measurements. BERTScore, being deterministic, is a more reliable signal for tracking compression quality across strategy iterations.

---

## What I'd Ship Next

1. **Hybrid v1+v5 strategy.** Use v1 threshold-based classification as the safety net. If the output exceeds a token budget, apply chunk-based top-K to trim further. Gets the reduction guarantees of v5 without the quality risk on edge cases.

2. **Full v2 evaluation.** Run the 10-conversation evaluation suite with `--compression-strategy sentence` to get quality numbers. The v2 spot-check shows 34.1% token reduction with structural coherence — the full eval numbers are the missing piece.

3. **Multi-run judge averaging.** Run each (conversation, query) pair through the judge 3 times and average. This stabilises the Δ quality metric and makes cross-strategy comparisons meaningful.

4. **Embedding-based landmark detector.** Replace regex patterns with cosine similarity to prototype embeddings. New domains need only new prototypes, not new code. Addresses the 13.4% of landmarks currently missed.

5. **Multi-domain evaluation.** Taskmaster-2 hotels and restaurant-search domains already downloaded. Running across three domains tests domain-agnosticism without architectural changes.

---

## AI Usage Note

This project was built with Claude (Anthropic) as a development partner throughout.

**Claude wrote:** All source code in `src/`, tests in `tests/`, and the utility scripts in `utilities/`. All five compression strategies, the assembly fixes, and the evaluation harness were implemented with Claude's assistance.

**I designed:** The overall architecture, dataset selection rationale, evaluation methodology, the two-pass landmark detection strategy, and all key decisions in `key_decisions.md`. The insight that chunk-based scoring was needed — because the answer to a query is spread across multiple turns, not contained in any single turn — was mine. The decision to remove hard-KEEP for landmarks in v3–v5 and let them compete in top-K was mine, as was the realisation that the landmark boost needed to be scaled by individual score to prevent query-irrelevant landmarks from dominating.

**I verified:** Landmark detector output was manually inspected across multiple conversations. The 86.6% recall figure is from an actual corpus-wide run. Every compression output was read carefully using `--compare` mode before accepting it. The LLM judge variance was identified by comparing identical runs and noticing the full-context scores fluctuating — not the optimised scores.

**How I used AI:** As a senior collaborator — I pushed back when explanations were incomplete, made all architectural decisions myself, and identified every failure mode by reading the actual output rather than accepting summary statistics.

The judgment about what to build, how to evaluate it honestly, and what the failure modes mean — those are mine.
