# Test Suite

55 tests across 6 files. No LLM API calls anywhere — all external dependencies are either mocked, use local fixtures, or test pure logic. Tests run in ~3 seconds.

---

## test_ingestion.py (5 tests)

Tests `src/ingestion/loader.py` — loading and normalising Taskmaster-2 JSON into `Conversation` objects.

| Test | What it checks |
|---|---|
| `test_normalise_turn_basic` | A raw Taskmaster-2 utterance dict is correctly converted to a `Turn` — speaker, text, turn_index, and slot names all extracted correctly |
| `test_normalise_turn_no_slots` | A turn with no segment annotations gets an empty slots list; all other fields initialised to defaults (is_landmark=False, score=0.0, disposition="") |
| `test_normalise_turn_strips_whitespace` | Leading/trailing whitespace is stripped from turn text at load time |
| `test_load_corpus_filters_by_min_turns` | A corpus file with one short (3 turns) and one long (25 turns) conversation: only the long one is returned when min_turns=20 |
| `test_load_corpus_domain_inferred` | The domain is inferred from the filename stem — `flights.json` → domain="flights" |

---

## test_landmarks.py (13 tests)

Tests `src/landmarks/rule_detector.py` — the two-pass rule-based landmark detector. All tests use synthetic conversations built from plain text; no slot annotations used (detector is text-only).

**Stated intent detection (pass 1)**

| Test | What it checks |
|---|---|
| `test_intent_explicit_verb` | "I'd like a flight to Orlando" → landmark:intent via explicit want/need verb |
| `test_intent_slot_signal_price` | "My budget is under $1,000" → landmark:intent via price signal |
| `test_intent_slot_signal_airline` | "I'd prefer Delta" → landmark:intent via airline name signal |
| `test_intent_slot_signal_date` | "I need to fly on March 14th" → landmark:intent via date signal |

**Decision detection (pass 1)**

| Test | What it checks |
|---|---|
| `test_decision_strong_confirmation` | "I'll take the 6AM flight" → landmark:decision via strong confirmation pattern |
| `test_decision_assistant_offer` | ASSISTANT presenting a price + time → landmark:decision via offer pattern |

**Cross-turn alignment (pass 2)**

| Test | What it checks |
|---|---|
| `test_decision_weak_confirmation_after_offer` | ASSISTANT offers a price, USER says "Yes." → both turns promoted to landmark:decision; USER turn has promoted=True |
| `test_decision_weak_confirmation_no_offer_not_promoted` | ASSISTANT asks a neutral question, USER says "Yes." → USER turn NOT promoted (no offer preceded it) |
| `test_pass2_pattern_b_echo` | USER states a constraint, ASSISTANT echoes it back with "is that correct?" → both turns promoted to landmark:intent |

**Action items**

| Test | What it checks |
|---|---|
| `test_action_item_send` | "I'll send you the flight details now" → landmark:action_item |
| `test_action_item_booking_confirmed` | "Your tickets have been booked and confirmed" → is_landmark=True; may be classified as action_item or decision depending on which pattern fires first — both are correct |

**Other**

| Test | What it checks |
|---|---|
| `test_filler_not_landmark` | Short acknowledgement turns ("Okay.", "Sure.", "Thank you.") → not landmarks |
| `test_detect_is_idempotent` | Running detect() twice on the same conversation produces identical results — no state accumulation |

---

## test_scoring.py (15 tests)

Tests `src/scoring/` — query classification, keyword scoring, recency decay, and the composite scorer.

**Query classifier**

| Test | What it checks |
|---|---|
| `test_classify_factual` | "What time does the flight depart?" → factual; "How much did the ticket cost?" → factual |
| `test_classify_analytical` | "Why did they choose Paris?" → analytical; "How do the flights compare?" → analytical |
| `test_classify_preference` | "Which flight would you recommend?" → preference |
| `test_classify_default_analytical` | Ambiguous query with no classification signals → defaults to analytical (the safe default that preserves the most context) |

**Keyword scorer (TF-IDF)**

| Test | What it checks |
|---|---|
| `test_keyword_scores_length` | Returns one score per input turn |
| `test_keyword_scores_relevant_turn_higher` | A turn mentioning "flight to Paris" scores higher than an unrelated turn for query "flight to Paris" |
| `test_keyword_scores_empty_texts` | Empty input returns empty list without error |
| `test_keyword_scores_in_range` | All scores are in [0, 1] |

**Recency decay**

| Test | What it checks |
|---|---|
| `test_recency_scores_monotonic` | Scores increase strictly with turn index — more recent turns always score higher |
| `test_recency_scores_latest_is_highest` | The turn closest to the query position has the highest score |
| `test_recency_scores_all_positive` | All recency scores are positive (exp decay is always > 0) |

**Composite scorer**

| Test | What it checks |
|---|---|
| `test_score_turns_sets_scores` | After scoring, all turns have a score in [0, 1] |
| `test_score_turns_landmark_boost` | A manually-flagged landmark turn scores ≥ a non-landmark turn after the +0.3 boost is applied |
| `test_score_turns_query_position_respected` | Scorer operates on the pre-sliced history list; returns same number of turns as input |
| `test_score_turns_empty` | Empty history returns empty list without error |

---

## test_compression.py (10 tests)

Tests `src/compression/compressor.py` and `src/compression/assembler.py` — turn classification, run grouping, and thread assembly.

**Turn classification**

| Test | What it checks |
|---|---|
| `test_landmark_always_keep` | A landmark turn (score=0.0) gets disposition=KEEP regardless of score — landmarks are always kept |
| `test_high_score_keep` | A turn with score=0.8 gets disposition=KEEP (above factual high threshold of 0.72) |
| `test_low_score_compress` | A turn with score=0.05 gets disposition=COMPRESS (below factual low threshold of 0.45) |
| `test_mid_score_candidate` | A turn with score=0.55 gets disposition=CANDIDATE (between factual thresholds 0.45–0.72) |
| `test_thresholds_vary_by_query_type` | Score of 0.5: KEEP for analytical (threshold=0.65), CANDIDATE for factual (threshold=0.72) — adaptive thresholds work |

**Run grouping**

| Test | What it checks |
|---|---|
| `test_group_single_run` | Three consecutive KEEP turns → one run of length 3 |
| `test_group_alternating` | KEEP/KEEP/COMPRESS/COMPRESS/KEEP/COMPRESS → four runs with correct dispositions |
| `test_group_candidate_merged_with_keep` | KEEP/CANDIDATE/KEEP → one merged KEEP run (CANDIDATE treated as KEEP for grouping) |
| `test_group_empty` | Empty input → empty output without error |

**Assembly and integrity**

| Test | What it checks |
|---|---|
| `test_assemble_keep_verbatim` | KEEP run → turns appear verbatim as {role, content} dicts in correct order |
| `test_assemble_compress_run_replaced` | COMPRESS run with a summary → replaced by a single [SUMMARY: ...] assistant turn; integrity check prepends user placeholder |
| `test_assemble_mixed` | Mixed KEEP/COMPRESS/KEEP → no consecutive same-role turns in output |
| `test_integrity_check_merges_consecutive` | Two consecutive user turns → merged into one; repair count = 1 |
| `test_integrity_check_no_repairs_needed` | Alternating user/assistant/user → returned unchanged; repair count = 0 |
| `test_format_full_context` | Three turns → [{role, content}] list with correct role mapping |

---

## test_evaluation.py (9 tests)

Tests `src/evaluation/landmark_recall.py` and token counting logic. No LLM calls.

**Landmark recall (vs slot-annotation ground truth)**

| Test | What it checks |
|---|---|
| `test_recall_perfect` | All slot-annotated turns detected → recall = 1.0 |
| `test_recall_zero` | No slot-annotated turns detected → recall = 0.0 |
| `test_recall_partial` | 1 of 2 slot-annotated turns detected → recall = 0.5 |
| `test_recall_no_gt_turns` | No slot-annotated turns exist → vacuous recall = 1.0 (nothing to miss) |
| `test_recall_non_gt_landmarks_ignored` | A detected landmark with no slots doesn't inflate the denominator — only slot-annotated turns count as ground truth |
| `test_landmark_stats_keys` | `landmark_stats()` returns a dict with all expected keys |

**Token counting**

| Test | What it checks |
|---|---|
| `test_token_count_import` | tiktoken is installed and can encode text |
| `test_token_reduction_formula` | The reduction formula `(full - opt) / full * 100` is arithmetically correct |
| `test_token_reduction_within_target` | Reductions of 40%, 45%, 50%, 55%, 60% all fall within the target range |

---

## test_query_selection.py (9 tests)

Tests `src/evaluation/harness.py` — the query pool and `select_queries()` function. OpenAI client is mocked throughout; no real API calls made.

**Query pool sanity**

| Test | What it checks |
|---|---|
| `test_query_pool_not_empty` | Pool contains at least 10 queries |
| `test_query_pool_types` | Every query has a valid type (factual/analytical/preference) and non-trivial text |
| `test_query_pool_has_all_types` | All three query types are represented in the pool |

**select_queries — happy path**

| Test | What it checks |
|---|---|
| `test_select_queries_returns_two` | Always returns exactly 2 queries |
| `test_select_queries_returns_eval_query_objects` | Returns `EvalQuery` objects with valid position, text, and type |
| `test_select_queries_uses_pool_text` | Selected queries use the exact text from QUERY_POOL at the returned indices |
| `test_select_queries_query_position_at_75pct` | query_position is set to int(n_turns * 0.75), clamped to ≥5 |

**select_queries — fallback on bad LLM response**

| Test | What it checks |
|---|---|
| `test_select_queries_fallback_on_invalid_json` | Unparseable LLM response → falls back to default indices [0, 8] |
| `test_select_queries_fallback_on_out_of_range_index` | Index 999 returned → falls back to defaults (index out of pool range) |
| `test_select_queries_fallback_on_api_error` | API exception → falls back to defaults without raising |

---

## What is NOT tested here

- LLM summarisation (`src/compression/summariser.py`) — requires OpenAI API key
- LLM-as-judge (`src/evaluation/judge.py`) — requires OpenAI API key
- BERTScore computation (`src/evaluation/bertscore_metric.py`) — requires bert-score model download
- Full pipeline integration (`src/compression/pipeline.py`) — requires both of the above
- Semantic similarity scoring (`src/scoring/semantic.py`) — tested implicitly via `test_scoring.py` which imports the scorer; model downloads on first run

These are tested manually via `python main.py inspect` and `python main.py evaluate`.
