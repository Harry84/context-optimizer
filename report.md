# Intelligent Context Optimizer — Written Report
## Technical Assessment, April 2026

---

## What I Built

The Context Optimizer takes a multi-turn conversation and a current query and returns a compressed conversation thread that preserves the information an LLM needs to answer the query — and discards what it doesn't. The pipeline has five stages: ingestion and normalisation, landmark detection, relevance scoring, compression and assembly, and evaluation.

**Landmark detection** runs first, before scoring, using a two-pass rule-based detector. Pass 1 classifies each turn individually by text signals and speaker role — stated intents (slot-value signals: price, date, airline, seat class), decisions (offer patterns, strong confirmations, cross-turn offer→confirmation pairs), and action items (commitment verbs). Pass 2 promotes adjacent turn pairs using cross-turn alignment: an ASSISTANT offer followed by a USER weak confirmation ("yes", "okay") promotes both to `decision`; a USER constraint echoed back by the ASSISTANT promotes both to `intent`. Landmarks are hard-preserved — they bypass the relevance scorer and are always kept verbatim. Measured against Taskmaster-2 slot annotations as ground truth: 86.6% recall across 1,692 conversations.

**Relevance scoring** classifies the query as factual, analytical, or preference — this drives different weight profiles across keyword (TF-IDF), semantic (MiniLM-L6-v2 cosine similarity), and recency (exponential decay) components. Landmarks receive a +0.3 boost after normalisation. Only history turns (0..query\_position-1) are scored; future turns are never seen.

**Compression** classifies each turn as KEEP, CANDIDATE, or COMPRESS, groups consecutive COMPRESS turns into runs, and makes one gpt-4o-mini summarisation call per run (capped at ≤15 words). Runs shorter than 200 characters are dropped silently — no LLM call, no placeholder. Assembly enforces structural validity: consecutive same-role turns are merged or bridged, and the thread is guaranteed to start with a user turn.

**Evaluation** runs across 10 conversations × 2 queries each. Queries are selected per conversation by a gpt-4o-mini call that reads the first 15 turns and picks the two most answerable questions from a 14-item pool — factual, analytical, and preference types. Both full-context and optimised answers are generated with gpt-4o and evaluated by a gpt-4o judge on a 4-dimension rubric (correctness, completeness, landmark consistency, hallucination). BERTScore F1 (roberta-large, local) provides an independent LLM-free quality signal.

---

## Results

Evaluation across 10 conversations (≥50 turns, Taskmaster-2 flights domain), 20 query pairs:

| Metric | Result | Target | Status |
|---|---|---|---|
| Token reduction | 24.1% | 40–60% | ✗ |
| Quality Δ (LLM judge) | +0.01 | ≥ 0 | ✓ |
| BERTScore F1 | 0.940 | ≥ 0.85 | ✓ |
| BERTScore ≥ 0.85 | 100% of queries | — | ✓ |
| Landmark recall | 77.0% | — | Reported |
| Latency | 628ms mean | — | Reported |

Quality is preserved. The optimised context produces answers statistically indistinguishable from full-context answers (Δ+0.01 on a 10-point scale, BERTScore 0.940). The token reduction target is not met — and the reason is specific and worth explaining.

---

## What Broke and Why

**Token reduction shortfall.** The 40–60% target is not met on Taskmaster-2. After thorough investigation — auditing token weight distributions across 30 conversations, inspecting per-turn dispositions on the worst compressors, and testing on synthetic realistic conversations — the root cause is clear: **compression granularity is at the turn level, but landmark content is at the sentence level.**

When a turn contains a booking decision ("I'll go with Virgin") alongside filler ("An onboard bar! That sounds fun."), the entire turn is hard-preserved as a landmark. The filler sentences survive verbatim because there is no mechanism to split them from the load-bearing sentence. Across the corpus, COMPRESS-classified turns hold only 11–33% of total tokens — the remaining tokens are in KEEP turns, many of which contain significant filler.

This is a design choice made consciously for v1 (see KD-016): turn-level compression is simpler to implement, easier to test, and safe by default. It is sufficient to prove the architecture and the quality-preservation result. But it sets a ceiling on token reduction that this dataset cannot push past.

A secondary factor: Taskmaster-2 has an unusually high landmark density (46.4% of turns flagged, vs ~30% on cleaner conversational data) because crowdworker transcripts repeat slot values verbatim on nearly every turn. This pushes more turns into the hard-KEEP category than a production conversation would.

**Judge JSON parsing (fixed).** The initial evaluation run returned all 5.0 scores — the neutral fallback — because the judge prompt did not use `response_format=json_object`. The model was occasionally wrapping its response in markdown fences or adding a preamble, causing the JSON parser to fail silently. Fixed by adding `response_format={"type": "json_object"}` to the judge call.

**Fixed evaluation queries (fixed).** The initial evaluation used the same two queries for every conversation regardless of content. A conversation where the user never compared airlines was being asked "What airlines did the user compare?" — the judge was scoring an unanswerable question, inflating variance. Fixed by adding a per-conversation query selector that reads the first 15 turns and picks the two most answerable questions from a 14-item pool.

---

## What I Learned

The most useful insight is about where token reduction actually comes from. It is not the number of turns compressed — it is the token weight of those turns. Compressing 51% of turns saves almost nothing if those turns average 5 tokens each and the kept turns average 12. The compression opportunity is in the long, rambling, low-relevance passages — not in the short filler turns that Taskmaster-2's telegraphic crowdworker style produces.

Sentence-level compression is the right next step (KD-016). The landmark detector correctly identifies which turns contain load-bearing content. The missing piece is sub-turn granularity: score each sentence independently, keep the sentences that matter, compress the rest. The architecture supports this without structural changes — it is a compressor-level modification.

The quality result is the more important result. A context optimizer that shrinks context and degrades answer quality is useless. Showing that optimised context produces answers equivalent to full-context answers — on independently selected, conversation-appropriate queries, judged by a separate model on four dimensions, with a local BERTScore check as a second opinion — is the substantive claim. That result holds cleanly.

---

## What I'd Ship Next

1. **Sentence-level compression (v2 compressor).** Split turns at sentence boundaries, score each sentence independently, only keep high-scoring sentences. Expected to push token reduction to 40–55% on realistic conversational data. Estimated 2–3 days. This is the single highest-priority item.

2. **Embedding-based landmark detector (v2 detector).** Replace regex patterns with cosine similarity to prototype embeddings using the already-loaded MiniLM-L6-v2. New domains need only new prototypes, not new code. Addresses the 13.4% of landmarks currently missed by the rule-based detector.

3. **Net cost dashboard.** Track tokens consumed by the summariser and judge per evaluation run. Report break-even explicitly: at what number of downstream LLM calls does compression pay for itself? Currently estimated at ≥2 downstream calls per optimised context with rule-based detection.

4. **Multi-domain evaluation.** Taskmaster-2 hotels and restaurant-search domains are already downloaded. Running evaluation across three domains tests domain-agnosticism without any architectural changes — only the query pool needs extending.

5. **Streaming assembly for 500+ turn conversations.** Current implementation loads all turns into memory before scoring. For very long conversations, process in chunks: score and compress the first N turns, emit the summary, then process the next N. Addresses the latency and memory pressure identified in the scale failure-mode analysis.

---

## AI Usage Note

This project was built with Claude (Anthropic) as a development partner throughout.

**Claude wrote:** All source code in `src/`, tests in `tests/`, and the utility scripts in `utilities/`. The landmark detector logic was iteratively developed and refined with Claude's assistance across multiple inspection cycles on real Taskmaster-2 data.

**I designed:** The overall architecture, dataset selection rationale, evaluation methodology, the two-pass landmark detection strategy (cross-turn alignment was my suggestion after observing that single-turn scoring missed offer→confirmation pairs), and all key decisions documented in `key_decisions.md`. The decision to investigate token reduction failure modes rigorously — rather than accepting the first plausible explanation — was mine.

**I verified:** Landmark detector output was manually inspected across multiple conversations using `--dry-run` and `--compare` modes before being accepted. The 86.6% recall figure is from an actual corpus-wide run. The token reduction analysis (compression audit, token weight audit, poor compressor inspection) was driven by me asking the right questions about the numbers.

**How I used AI:** As a senior collaborator — I pushed back when explanations were incomplete, asked follow-up questions when numbers didn't make sense, and made all architectural decisions myself. When the evaluation returned all 5.0 scores, I identified it as a parsing failure and directed the fix. When the token reduction was 15%, I drove the investigation that identified turn-level granularity as the root cause rather than accepting a threshold tweak.

The judgment about what to build, how to evaluate it honestly, and what the failure modes mean — those are mine.
