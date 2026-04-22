# Key Decisions Log
## Context Optimizer — Running Record of Architectural & Design Choices

> **Purpose:** This document records significant decisions made during the project, with justification at the time of making them. Useful for the Architecture Decisions Document deliverable, the written report, and interview follow-up questions.

---

## KD-001 — Dataset Choice: OpenAssistant/oasst1

**Decision:** Use the OpenAssistant oasst1 dataset as the source of multi-turn conversations for evaluation.

**Justification:**
- **Length:** oasst1 contains ~9,000 conversation trees, many with deep branching. After reconstructing trees along the highest-ranked path (see KD-002), a meaningful number of threads reach 50+ turns — enough to build a non-trivial evaluation corpus without synthetic data.
- **Quality:** The dataset was collected via a coordinated human annotation effort (Open Assistant project, 2022–2023) with explicit quality ratings per message. This means we are not working with scraped or synthetic conversations — the turns reflect genuine multi-turn human reasoning and instruction-following behaviour.
- **Popularity & legitimacy:** oasst1 is widely used in the open-source LLM research community (used to train several open models). Using it signals familiarity with standard NLP benchmarking practice.
- **Licence:** CC BY 4.0 — permissive, suitable for research and demonstration.
- **Local availability:** Already downloaded at `data/oasst1.jsonl` (~137MB), avoiding repeated API calls or download dependencies during development.

**Alternatives considered:**
- Synthetic conversations: rejected — too easy to game the evaluation, less realistic turn structure.
- ShareGPT: rejected — licence ambiguity, less structured metadata.
- Custom conversations: rejected — insufficient volume for ≥10-conversation evaluation without significant manual effort.

**Trade-offs:**
- oasst1 conversations are human↔AI, not human↔human — this means they naturally lack certain landmark types (e.g. scheduling commitments between two people). Acceptable for v1; noted as a limitation in the written report.
- The ≥50 turn filter may yield a smaller corpus than expected if most deep threads are non-English. Mitigation: filter to English (`lang == "en"`) first, then apply turn threshold.

---

## KD-002 — Tree Reconstruction Strategy: Highest-Ranked Path

**Decision:** At each branching point in the OASST1 tree, select the child with the lowest `rank` value (= highest human preference rank). This materialises one canonical linear thread per conversation tree.

**Justification:**
- oasst1 is structured as a tree, not a list. Multiple human responses and multiple assistant responses can exist at each node. We must choose one path to get a coherent linear conversation.
- The `rank` field encodes human preference — lower rank = humans rated it more highly. Following the highest-ranked path gives us the "best" version of each conversation: the responses that human annotators found most useful, which is a reasonable proxy for realistic, high-quality dialogue.
- Keeps the reconstruction algorithm simple and deterministic: no randomness, no ambiguity.

**Alternatives considered:**
- Random path selection: rejected — non-deterministic, harder to reproduce evaluation results.
- All paths: rejected — combinatorial explosion, and we don't need it to demonstrate context optimisation.
- Longest path: rejected — length does not correlate with quality in oasst1.

**Trade-offs:**
- We discard a large amount of the dataset (all non-highest-ranked branches). Acceptable for v1 — we only need ≥10 conversations for the evaluation, not the full corpus.
- Multi-path evaluation (e.g. testing whether the optimizer works equally well on lower-quality branches) is noted as a future direction.

---

## KD-003 — Corpus Filtering: ≥50 Turn Threshold

**Decision:** Only retain reconstructed conversation threads with 50 or more turns for the evaluation corpus.

**Justification:**
- The assignment specification explicitly requires "50+ messages" to constitute a long conversation that stresses naive full-history approaches.
- Below 50 turns, the token cost of full-history is low enough that optimisation provides marginal benefit — making it harder to demonstrate meaningful reduction.
- 50 turns is also long enough to naturally contain multiple topic shifts, digressions, and landmark events (decisions, commitments), giving the optimizer meaningful material to work with.

**Trade-offs:**
- May yield a small corpus. Mitigation: lower threshold to 30 if fewer than 10 conversations pass the filter after English-language filtering, and document this in the report.

---

## KD-004 — Embedding Model: Local sentence-transformers (all-MiniLM-L6-v2)

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` running locally for all semantic similarity computations.

**Justification:**
- **Zero API cost:** Embedding is called for every message in every conversation during scoring. At scale this would be expensive via OpenAI API. Local inference eliminates this cost entirely.
- **Determinism:** Local model with fixed weights gives identical embeddings on every run, making evaluation reproducible.
- **Speed:** MiniLM-L6-v2 is a small model (~80MB) optimised for sentence similarity tasks. Inference is fast on CPU, viable on a developer laptop.
- **Quality:** Despite its size, all-MiniLM-L6-v2 performs competitively on semantic textual similarity benchmarks (SBERT leaderboard). Sufficient for retrieval-style scoring.

**Alternatives considered:**
- `text-embedding-3-small` (OpenAI): rejected for v1 — per-call cost adds up at 50–500 messages × multiple conversations × multiple queries.
- `all-mpnet-base-v2`: better quality but 4× slower and larger. Not needed for this task.
- TF-IDF only (no embeddings): rejected — misses semantic similarity entirely, would over-rely on exact keyword overlap.

**Trade-offs:**
- MiniLM may miss subtle semantic relationships that a larger model would catch. Accepted — the scoring signal is combined with keyword match and recency decay, so no single signal needs to be perfect.
- Future work: swap in OpenAI embeddings as a configurable option and compare quality.

---

## KD-005 — LLM-as-Judge: Rubric Design & Bias Mitigation

**Decision:** Use a decomposed 4-dimension rubric (correctness, completeness, decision consistency, hallucination), evaluated at temperature=0, with order-swap bias mitigation, supplemented by BERTScore F1 as an independent semantic preservation metric.

**Justification:**

**Why a decomposed rubric rather than a holistic score?**
- A single "how good is this answer" score conflates multiple concerns. A judge can find an answer correct but incomplete, or complete but hallucinating. Decomposing forces the judge to reason about each axis independently, reducing the chance that one salient feature dominates the score.
- Decomposed rubrics are more reproducible — less dependent on the judge's implicit weighting of trade-offs.

**Why order-swap?**
- Positional bias in LLM judges is well-documented (Wang et al., 2023; Zheng et al., 2023). The judge systematically favours whichever answer it sees first. By evaluating each pair twice (full-context first; optimised-context first) and averaging, we cancel this effect.

**Why BERTScore F1 as a separate metric?**
- LLM-as-judge is not fully objective. BERTScore provides an independent, deterministic, LLM-free measure of whether the optimised-context answer preserves the semantic content of the full-context answer.
- It measures token-level alignment using contextual embeddings — robust to paraphrasing, unlike BLEU or exact match.
- Threshold of F1 ≥ 0.85 is calibrated to typical paraphrase-level similarity in NLP literature.

**Why temperature=0 for the judge?**
- Reproducibility. We want the same inputs to produce the same scores every time so the evaluation table is stable.

**Alternatives considered:**
- Single holistic score: rejected — too coarse, hides the source of quality differences.
- Human evaluation: rejected — too slow and expensive for ≥10 conversations × multiple queries.
- ROUGE/BLEU: rejected — surface-level metrics, poor proxy for semantic quality.

---

## KD-006 — Hard Mode Features are Must Have

**Decision:** Adaptive compression strategy (factual/analytical/preference query classification) and landmark detection (decisions, commitments, action items) are treated as core Must Have requirements, not stretch goals.

**Justification:**
- The assignment specification explicitly labels these as "hard mode signals we'll look for" — language that implies they are differentiating features, not optional extras.
- Adaptive strategy is what makes the system defensible: without it, the compression thresholds are arbitrary. With it, there is a principled reason why a factual query preserves more context than an analytical one.
- Landmark detection is what prevents the system from compressing away the most important information. Without it, the optimizer could discard a key decision and produce an incoherent answer — the worst possible failure mode.
- Both features are implementable with reasonable effort in v1 (heuristics + lightweight classification) and create a clear upgrade path for future work (fine-tuned classifier, LLM-based landmark detection).

---

*Last updated: April 2026*
*Add new entries as decisions are made during architecture and implementation phases.*
