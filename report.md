# Intelligent Context Optimizer — Written Report
## Technical Assessment, April 2026

---

## What I Built

The Context Optimizer takes a multi-turn conversation and a current query and returns a compressed conversation thread that preserves the information an LLM needs to answer the query — and discards what it doesn't. The pipeline has five stages: ingestion and normalisation, landmark detection, relevance scoring, compression and assembly, and evaluation. Three compression strategies are implemented: turn-level (v1), sentence-level (v2), and top-K retrieval (v3).

**Landmark detection** runs first, before scoring, using a two-pass rule-based detector. Pass 1 classifies each turn individually by text signals and speaker role — stated intents (slot-value signals: price, date, airline, seat class), decisions (offer patterns, strong confirmations, cross-turn offer→confirmation pairs), and action items (commitment verbs). Pass 2 promotes adjacent turn pairs using cross-turn alignment: an ASSISTANT offer followed by a USER weak confirmation ("yes", "okay") promotes both to `decision`; a USER constraint echoed back by the ASSISTANT promotes both to `intent`. Landmarks are hard-preserved — they bypass the relevance scorer and are always kept verbatim. Measured against Taskmaster-2 slot annotations as ground truth: 86.6% recall across 1,692 conversations.

**Relevance scoring** classifies the query as factual, analytical, or preference — this drives different weight profiles across keyword (TF-IDF), semantic (MiniLM-L6-v2 cosine similarity), and recency (exponential decay) components. Landmarks receive a +0.3 boost after normalisation. Only history turns (0..query\_position-1) are scored; future turns are never seen.

**Compression — v1 (turn-level):** classifies each turn as KEEP, CANDIDATE, or COMPRESS, groups consecutive COMPRESS turns into runs, and makes one gpt-4o-mini summarisation call per run (capped at ≤15 words). Runs shorter than 200 characters are dropped silently.

**Compression — v2 (sentence-level):** for landmark turns, splits the text into sentences using NLTK's punkt tokeniser, re-runs landmark pattern matching at sentence level to identify which specific sentences triggered the landmark, and scores non-landmark sentences independently against the query. Sentences with the same effective disposition from the same turn are merged back before assembly to prevent structural fragmentation.

**Compression — v3 (top-K retrieval):** landmarks always KEEP; non-landmark turns ranked by composite score descending; top K (proportional to conversation length) kept, rest compressed. A noise floor (`topk_min_score=0.30`) prevents low-quality turns filling K slots. Designed to guarantee token reduction independent of landmark density.

**Assembly** enforces structural validity across all strategies: consecutive same-role turns are merged or bridged, and the thread is guaranteed to start with a user turn.

**Evaluation** runs across 10 conversations × 2 queries each, with per-conversation query selection from a 14-item pool. Both full-context and optimised answers are generated and evaluated by a gpt-4o judge on a 4-dimension rubric. BERTScore F1 (roberta-large, local) provides an independent quality signal.

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
| Latency | ~10s (CPU, unbatched) |

**v3 evaluation** — same 10 conversations, proportional top-K (factual=20%, analytical=35%, preference=25%):

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 21.0% | 40–60% | ✗ |
| Quality Δ (LLM judge) | -0.06 | ≥ 0 | ✗ |
| BERTScore F1 | 0.952 | ≥ 0.85 | ✓ |
| Latency | 462ms mean | — | Reported |

v1 is the recommended default. Quality is preserved and the failure mode is conservative — it over-keeps rather than under-keeps.

---

## What Broke and Why

**Token reduction shortfall (v1, v2, v3).** The 40–60% target is not met on Taskmaster-2 with any strategy. The root cause is consistent: Taskmaster-2's crowdworker transcripts have unusually high landmark density (~46% of turns flagged) because slot values are repeated verbatim on nearly every turn. Landmark turns hold ~67% of total tokens. Even compressing every non-landmark turn to zero, maximum achievable reduction is ~33%. This is not representative of real production conversations, where landmark density is typically ~30%.

**Top-K quality regression (v3).** One conversation produced Δ quality = -6.25 on a "what airports were involved?" query — the relevant airport turns did not rank in the top 20% of non-landmark turns for that specific query, so they were compressed away. This exposes the fundamental risk of hard-K retrieval: relevant content can be lost when it does not rank highly against a specific query even though it would be retained by landmark detection or threshold-based classification. v3 achieves higher reduction on long conversations with low landmark density (46% on our test conversation) but is not safe as a default because the failure mode is to silently drop relevant content rather than conservatively over-keep it.

**v2 latency.** Sentence-level scoring produces ~10s latency on CPU. Fixable by batching all non-landmark sentences into a single embedding call.

**Judge JSON parsing (fixed).** Initial evaluation returned all 5.0 scores due to missing `response_format=json_object`. Fixed.

**Fixed evaluation queries (fixed).** Initial evaluation used the same two queries for every conversation. Fixed by per-conversation query selection from a 14-item pool.

---

## What I Learned

The most important finding is about failure modes in compression strategies. Threshold-based compression (v1) fails conservatively — it over-keeps, reducing less than the target but never dropping critical content. Top-K retrieval (v3) fails aggressively — it can drop content that is genuinely relevant but does not rank highly for a specific query. For a production system, conservative failure is strongly preferable: an answer based on a slightly larger context is better than an answer that hallucinates because relevant context was discarded.

The second finding is that token reduction and quality preservation are in tension in a way that depends on the conversation structure. Dense, efficient conversations (where nearly every turn is load-bearing) compress less — and that is the correct behaviour. The system should not compress content just to hit a number.

Implementing sentence-level compression surfaced a structural problem that was not anticipated in the v1 design: splitting turns into sentences creates consecutive same-speaker synthetic turns which the assembler repairs with bridges, inflating the output. The three-step fix (merge on effective disposition, promote sandwiched sentences, batch turns before assembly) only emerged from carefully reading the assembled output and identifying what was wrong.

---

## What I'd Ship Next

1. **Hybrid v1+v3 strategy.** Use threshold-based classification as the safety net, but cap the output at a token budget. If the threshold-based output exceeds the budget, apply top-K to the non-landmark turns to trim further. This gets the reduction guarantees of v3 without the quality risk.

2. **Full v2 evaluation.** Run the 10-conversation evaluation suite with `--compression-strategy sentence` to get quality numbers. The v2 spot-check shows 34.1% token reduction with structural coherence — the quality numbers are the missing piece.

3. **Embedding-based landmark detector.** Replace regex patterns with cosine similarity to prototype embeddings. New domains need only new prototypes, not new code. Addresses the 13.4% of landmarks currently missed.

4. **Net cost dashboard.** Track tokens consumed by summariser and judge per run. Report break-even explicitly. Currently estimated at ≥2 downstream calls per optimised context.

5. **Multi-domain evaluation.** Taskmaster-2 hotels and restaurant-search domains already downloaded. Running across three domains tests domain-agnosticism without architectural changes.

---

## AI Usage Note

This project was built with Claude (Anthropic) as a development partner throughout.

**Claude wrote:** All source code in `src/`, tests in `tests/`, and the utility scripts in `utilities/`. The landmark detector, sentence-level compressor, top-K compressor, and all assembly fixes were implemented with Claude's assistance.

**I designed:** The overall architecture, dataset selection rationale, evaluation methodology, the two-pass landmark detection strategy, and all key decisions in `key_decisions.md`. The decision to implement top-K retrieval and to document its failure mode honestly rather than accepting the result were mine.

**I verified:** Landmark detector output was manually inspected across multiple conversations. The 86.6% recall figure is from an actual corpus-wide run. The token reduction analysis was driven by me asking the right questions about the numbers. The v3 quality regression was identified by reading the CSV row by row, not from the summary.

**How I used AI:** As a senior collaborator — I pushed back when explanations were incomplete, made all architectural decisions myself, and identified every failure mode by reading the actual output rather than accepting summary statistics.

The judgment about what to build, how to evaluate it honestly, and what the failure modes mean — those are mine.
