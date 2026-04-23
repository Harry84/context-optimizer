# Context_Optimizer — Codebase Audit Report
_Date: 2026-04-23 | Scope: All source, docs, tests, config_

---

## LLM Evaluator Step — End-to-End Trace (Priority Focus)

Before the full findings, here is the complete flow of the evaluator and every inconsistency found within it.

### How it works (code truth)

1. `cmd_evaluate` (`main.py:227–244`) samples 10 conversations and calls `evaluate(eval_convs, {}, config)` with an **empty** query dict — the query override path is never exercised from the CLI.
2. `evaluate()` (`harness.py:184–230`) runs landmark detection, then per-conversation calls `select_queries(conv, model=config.summarisation_model)` to pick 2 questions from `QUERY_POOL`.
3. `select_queries` (`harness.py:88–139`) asks the model to pick indices from 14 pooled queries; sets `query_position = max(5, int(n * 0.75))` — but the selector only sees the **first 15 turns** regardless of where the query position lands.
4. `_evaluate_one` (`harness.py:233–282`):
   - Builds `full_thread` and `opt_thread, stats, latency = compress(...)`.
   - Generates `answer_full` and `answer_opt` using **`config.judge_model`** (gpt-4o).
   - Calls `judge_pair(...)` also using **`config.judge_model`** — same model generates and judges.
   - Computes BERTScore and landmark recall separately.
   - Records `delta_quality = round(scores_opt.mean - scores_full.mean, 2)`.
5. `judge_pair` (`judge.py:119–141`) calls `_judge_once` **once each** for full and opt — no order-swap.
6. `JudgeScores.mean` = `(correctness + completeness + landmark_consistency + hallucination) / 4.0` — unweighted, relying on the prompt rubric to make all four "higher is better" (hallucination: 10 = none).

### Consistency verdict

The arithmetic aggregation is internally consistent end-to-end. The scores in `eval_results*.csv` correctly reflect what the code does. However, **two methodology bugs** mean the numbers don't reflect what the PRD/ARCHITECTURE.md says:

- The **same model** generates answers and judges them (violates PRD §11.1 and KD-005 bias mitigation claim).
- **No order-swap** is implemented (PRD §11.1 claims it is).

---

## Findings by Severity

### CRITICAL

---

**C1 — ~~Real API keys committed to disk~~ (resolved) — stale `.env.example` docs reference**
- `.env` is in `.gitignore` and has never been committed (confirmed via `git log`). Keys are not exposed.
- `README.md:55` and `ARCHITECTURE.md:171–172` referenced a `.env.example` file that does not exist.
- **Fixed:** Both docs updated — README now says "Create a `.env` file", ARCHITECTURE.md tree no longer lists `.env.example`.
- Severity downgraded to **LOW** (docs-only).

---

**C2 — Judge model is also the answer-generation model (eval methodology violation)**
- Files: `src/evaluation/harness.py:247–256`, `src/ingestion/models.py:108`
- `_generate_answer(full_thread, eq.query_text, config.judge_model)` — gpt-4o generates both answers.
- `judge_pair(..., model=config.judge_model)` — the same gpt-4o judges them.
- This directly violates:
  - `PRD.md:181`: "different judge model from generation model"
  - `key_decisions.md:52` (KD-005): same claim
  - `ARCHITECTURE.md:395`: "Judge LLM | different model"
- There is no `generator_model` field on `OptimizerConfig` and no `--generator` CLI flag.
- **Every quality score in `eval_results*.csv` was produced under this conflict.**
- **Action:** Add `OptimizerConfig.generator_model` (default `gpt-4o-mini`) and use it in `_generate_answer`. Keep `judge_model = gpt-4o`.

---

**C3 — Documented order-swap bias mitigation is not implemented**
- Files: `src/evaluation/judge.py:129–130`, `PRD.md:181`, `key_decisions.md:52`
- `PRD.md:181`: "order-swap (each pair evaluated twice, scores averaged)".
- Code: each answer is judged exactly **once**, independently. `judge.py:129–130` explicitly comments "no side-by-side comparison, so positional bias does not apply."
- `key_decisions.md` (KD-005) documents the switch to independent evaluation — so the intent changed — but the PRD was never updated. Anyone reading the PRD believes a bias mitigation exists that does not.
- **Action:** Either implement order-swap, or update `PRD.md:181` to match KD-005.

---

### HIGH

---

**H1 — Landmark recall headline number contradicts measured results**
- Files: `README.md:30,172–174`, `ARCHITECTURE.md:238`, `report.md:41`, `eval_results.csv`
- README.md and ARCHITECTURE.md claim **86.6% landmark recall**.
- `report.md` table shows **77.0%**.
- `eval_results.csv` mean is **≈76.6%** — consistent with the report, not the README.
- No script in the repo produces the 86.6% figure. It appears to be from an uncommitted measurement or is stale.
- **Action:** Re-run the measurement and commit the script, or correct README.md and ARCHITECTURE.md to 77%.

---

**H2 — Test count claims are mutually contradictory**
- `README.md:73`: "All **46** tests pass"
- `tests/README.md:3`: "**55** tests across 6 files"
- `ARCHITECTURE.md:169`: "pytest suite (**73** tests)"
- Actual count: **~52** test functions across 6 test files.
- **Action:** Run `pytest --collect-only -q | tail -1` and update all three documents.

---

**H3 — `compression_pct` is misleading for sentence/topk-sentence strategies**
- File: `src/evaluation/harness.py:264`
- `compressed = sum(1 for t in history if t.disposition == "COMPRESS")` counts turn-level `disposition`.
- For `sentence` and `topk-sentence`, `.disposition` is written on synthetic sentence-level `Turn` objects inside the compressor — the original `history` turns' `.disposition` is never updated. Result: `compression_pct` is effectively **0 for sentence-level strategies**, making the CSV column misleading for v2 and v4.
- **Action:** Either propagate disposition back to original turns, or add a caveat to CSV column docs.

---

**H4 — v3 `topk` strategy hard-KEEPs landmarks — contradicts docs claiming it does not**
- File: `src/compression/topk_compressor.py:44–49`
- Code: `for t in landmarks: t.disposition = "KEEP"` — hard-KEEP is active in v3.
- `ARCHITECTURE.md:276` (§8.3): "No hard-KEEP for landmarks — they receive +0.3 boost but compete in top-K pool."
- `key_decisions.md:151–152` (KD-017): "In v3–v5, landmarks receive a score boost but are not hard-KEEPed."
- v4 (`topk_sentence_compressor.py`) and v5 (`chunk_compressor.py`) correctly do **not** hard-KEEP.
- **Action:** Either remove the hard-KEEP from `topk_compressor.py:44–49` (making v3 match the docs), or update ARCHITECTURE.md/KD-017 to say v3 does hard-KEEP.

---

**H5 — Scoring weights are hardcoded in v2/v4/v5, ignoring query-type config**
- Files: `src/compression/sentence_compressor.py:133`, `topk_sentence_compressor.py:142`, `chunk_compressor.py:124`
- All three hardcode `w1, w2, w3 = 0.35, 0.50, 0.15` regardless of query type.
- Only v1 (`scorer.py`) uses `config.weights[query_type]` as documented.
- Contradicts `README.md:183–186`, `ARCHITECTURE.md:252–256`, `PRD.md:138` — all of which show a per-query-type weight table.
- **Action:** Either thread `config.weights[query_type]` through v2/v4/v5, or update docs to state that v2+ use a fixed weight profile.

---

**H6 — `query_type` in CSV is pool label; compressor uses `classify_query()` output — may disagree**
- Files: `src/evaluation/harness.py:269`, `src/compression/pipeline.py:40`
- `EvalResult.query_type` (what's in the CSV) is `eq.query_type` — the hand-label from `QUERY_POOL`.
- Inside `compress()`, `query_type = classify_query(query)` is recomputed from text independently.
- If the two disagree (e.g. a "preference" query whose text triggers the factual regex), the weights and thresholds actually used differ from what the CSV column shows.
- **Action:** Pass `eq.query_type` into `compress()` and use it directly instead of re-classifying.

---

**H7 — `select_queries` selector sees only first 15 turns regardless of query position**
- File: `src/evaluation/harness.py:101–110,130`
- The selector prompt receives `_conversation_snippet(..., n_turns=15)`, but `q_pos = max(5, int(n * 0.75))`.
- In a 63-turn conversation, the query is positioned at turn 47 — but the selector never sees turns 16–46. It cannot judge which questions are actually answerable at that position.
- **Action:** Pass a snippet ending at `q_pos` rather than always taking the first 15 turns.

---

### MEDIUM

---

**M1 — `JudgeScores.mean` relies on prompt convention for hallucination inversion, not code**
- File: `src/evaluation/judge.py:40–47`
- `mean = (correctness + completeness + landmark_consistency + hallucination) / 4.0`
- The rubric makes hallucination "10 = no hallucination" (higher is better). This works only if the judge reliably follows the inversion convention. Any judge misinterpretation (returning 2 for "no hallucination found") silently corrupts the mean.
- **Action (low urgency):** Invert in code — store hallucination raw and compute mean as `(c + co + lc + (11 - h)) / 4.0` to make the inversion explicit and model-independent.

---

**M2 — Parse-error fallback of `(5.0, 5.0, 5.0, 5.0)` is silent in CSV output**
- File: `src/evaluation/judge.py:107`
- A JSON parse failure returns neutral 5.0 for all dimensions. With ~20 queries per run, one failure noticeably shifts the reported means.
- No `parse_error: bool` flag exists in `EvalResult` — you cannot distinguish a genuine neutral score from a fallback.
- **Action:** Add `judge_parse_error: bool` to `EvalResult` and set it on fallback.

---

**M3 — README.md threshold table is stale — does not match `OptimizerConfig` defaults**
- Files: `README.md:184–186`, `src/ingestion/models.py:63–67`
- README shows e.g. factual thresholds as `keep=0.6, compress=0.3`.
- Code has `factual high=0.72, low=0.45`.
- `ARCHITECTURE.md:264` and `test_compression.py:28,45` correctly match the code (0.72/0.45).
- **Action:** Update the README.md threshold table.

---

**M4 — `_merge_singleton_compress_runs` mutates input `runs` list during iteration**
- File: `src/compression/compressor.py:93–96`
- When a singleton COMPRESS run has an adjacent next COMPRESS run, it does `runs[i+1] = ...` — mutating the source list, not the output `merged` list. This works under current call patterns but is fragile.
- **Action:** Refactor to work on a copy, or add a comment documenting the in-place side effect.

---

**M5 — `score_turns` is called unconditionally in `pipeline.py` for all strategies**
- File: `src/compression/pipeline.py:42–47`
- For `sentence`/`topk-sentence`, the computed `turn.score` is largely unused (sentence-level scoring overwrites it). For `chunk`, `_score_turns_by_chunks` fully overwrites it.
- Wasted LLM-free compute on every call — not a correctness bug.
- **Action (low urgency):** Gate the `score_turns` call on strategy if performance becomes a concern.

---

**M6 — Singleton `SentenceTransformer` model cache ignores model name**
- File: `src/scoring/semantic.py:14`
- `_MODEL: SentenceTransformer | None = None` is a process-level singleton. A second call with a different `model_name` silently reuses the first model.
- Currently all callers pass `"all-MiniLM-L6-v2"` so no bug — but the behaviour is undocumented and fragile.
- **Action:** Add an assertion or keyed cache `dict[str, SentenceTransformer]`.

---

### LOW

---

**L1 — Dead `import numpy as np` in `keyword.py:10`**
- `np` is never used. Remove the import.

---

**L2 — Dead `langchain-text-splitters` dependency in `pyproject.toml:19`**
- `sentence_splitter.py` docstring references LangChain "for the embedding work planned in v2" but no project source file imports it.
- **Action:** Remove from `pyproject.toml` unless a future version needs it.

---

**L3 — `eval_queries` parameter of `evaluate()` is effectively dead code**
- File: `src/evaluation/harness.py:192`, `main.py:240`
- `evaluate()` accepts per-conversation query overrides but `main.py` always passes `{}`. The CLI exposes no way to populate this dict.
- **Action:** Either wire up a `--queries-file` CLI arg or document it as an internal API.

---

**L4 — Airport floor regex comment "City names deliberately excluded" is slightly misleading**
- File: `src/compression/chunk_compressor.py:42–60`
- The floor applies when query contains `airport|terminal|flight|boarding|gate` and turn text contains same. The comment implies city-name exclusion is a meaningful precision choice, but common words like "airport" are broad enough that the exclusion adds little discriminating value.
- No bug; cosmetic documentation clarity.

---

**L5 — `_ACTION_PATTERNS` and `_OFFER_PATTERNS` overlap in `rule_detector.py`**
- File: `src/landmarks/rule_detector.py:110–180`
- "tickets have been booked" matches `_OFFER_PATTERNS` (→ `decision`); "I've booked" matches `_ACTION_PATTERNS` (→ `action_item`). Semantically equivalent utterances get different landmark types.
- `test_landmarks.py:122–127` acknowledges this ambiguity by accepting either type.
- **Action (low urgency):** Consolidate overlapping patterns or document the type precedence explicitly.

---

**L6 — `recency.py:34` exponent can exceed 1.0 if `idx > query_position`**
- `math.exp(-lambda_decay * (query_position - idx))` — if `idx > query_position`, result > 1.
- In practice, callers always slice `history[:query_position]`, so `idx < query_position` always holds. No bug, but add a `assert idx < query_position` to make the invariant explicit.

---

**L7 — Test count claims never reconciled with runner output**
- See H2. Additionally, `tests/README.md:200–206` acknowledges that `judge.py` and `summariser.py` have no mocked LLM tests — this is consistent with design intent but should be noted for future CI coverage.

---

## Summary Table

| ID | File(s) | Issue | Severity |
|----|---------|-------|----------|
| C1 | `README.md`, `ARCHITECTURE.md` | Stale `.env.example` reference (docs fixed; keys never committed) | ~~CRITICAL~~ LOW |
| C2 | `harness.py:247–256` | Same model generates answers and judges them | **CRITICAL** |
| C3 | `judge.py`, `PRD.md:181` | Order-swap documented but not implemented | **CRITICAL** |
| H1 | `README.md:30`, `eval_results.csv` | 86.6% landmark recall contradicts measured 77% | **HIGH** |
| H2 | `README.md:73`, `ARCHITECTURE.md:169` | Test count claims: 46 vs 55 vs 73 (actual ~52) | **HIGH** |
| H3 | `harness.py:264` | `compression_pct` always 0 for sentence strategies | **HIGH** |
| H4 | `topk_compressor.py:44–49` | v3 hard-KEEPs landmarks; docs say it does not | **HIGH** |
| H5 | `sentence_compressor.py:133` et al. | Scoring weights hardcoded, query-type config ignored | **HIGH** |
| H6 | `harness.py:269`, `pipeline.py:40` | CSV `query_type` may differ from type used in compress | **HIGH** |
| H7 | `harness.py:101–130` | Selector sees only first 15 turns regardless of q_pos | **HIGH** |
| M1 | `judge.py:40–47` | Hallucination inversion relies on prompt, not code | **MEDIUM** |
| M2 | `judge.py:107` | Silent 5.0 fallback undetectable in CSV | **MEDIUM** |
| M3 | `README.md:184–186` | Threshold table doesn't match `models.py` defaults | **MEDIUM** |
| M4 | `compressor.py:93–96` | `_merge_singleton_compress_runs` mutates input list | **MEDIUM** |
| M5 | `pipeline.py:42–47` | `score_turns` called for strategies that discard result | **MEDIUM** |
| M6 | `semantic.py:14` | Model cache ignores model name argument | **MEDIUM** |
| L1 | `keyword.py:10` | Unused `import numpy as np` | LOW |
| L2 | `pyproject.toml:19` | Dead `langchain-text-splitters` dependency | LOW |
| L3 | `harness.py:192` | `eval_queries` parameter is dead from CLI | LOW |
| L4 | `chunk_compressor.py:42` | Airport regex comment slightly misleading | LOW |
| L5 | `rule_detector.py:110–180` | Overlapping action/offer patterns → inconsistent types | LOW |
| L6 | `recency.py:34` | Missing assertion for `idx < query_position` invariant | LOW |
| L7 | `tests/README.md` | No mocked LLM tests for judge/summariser | LOW |

---

## What Is NOT a Problem

- All function calls resolve to real definitions — no hallucinated imports or undefined functions found.
- All imports are valid.
- Variable scoping is clean throughout.
- The `JudgeScores` arithmetic aggregation is internally consistent end-to-end.
- BERTScore and landmark recall computations are independent of the judge bugs and appear correct.
- `_merge_singleton_compress_runs` test coverage confirms expected behaviour (despite the mutation style).
- `_dedup_sentences` regex in `loader.py:49–63` correctly avoids splitting on decimal numbers.
