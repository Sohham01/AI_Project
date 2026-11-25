# V6.3 — Issue–Solution Mapping & Resolution Evaluation Report

> **Evaluation Honesty Notice:** Metrics reported in Sections 2 and 3 below are evaluated against `issue_solution_eval_v1.csv` (`n = 64`) and `resolution_eval_v1.csv` (`n = 40`), which are explicitly labelled **`DEVELOPMENT_SET_ONLY_V1`** (used during V6.0/V6.3 rule and calibration design). Blind independent `v2` evaluation sheets (`180` linking pairs across all 8 issues + `120` threads with chronological timelines) have been frozen with `0%` `v1` overlap under [`data/annotations/v6_3_package/linking_and_resolution_v2/`](../data/annotations/v6_3_package/linking_and_resolution_v2/) and await human annotation before claiming generalization on unseen threads.

---

## 1. P0-B Resolution Engine Fixes (`ml/linking/resolution_engine.py`)

V6.3 resolved four structural reliability flaws in `ExplainableResolutionEngine`:

1. **Dynamic Deterministic Reference Timestamp (`resolve_reference_time`):** Removed the hard-coded `2026-04-15` fallback and disallowed implicit machine wall-clock calls (`datetime.now()`). Reference time resolves deterministically from `explicit_reference_time → dataset_observation_timestamp → corpus_max_observation_timestamp / thread_latest_activity_timestamp`, stamped verbatim inside `timestamp_semantics` (`reference_time = 2026-11-05T05:08:44+00:00`, `corpus_max_observation_timestamp` for `v6.1.0`).
2. **Five-Tier Discourse Evidence Separation (`agreement` ≠ Confirmation):**
   - `proposed_solution`: Linked Phase 4 solution candidates (`matching_score >= 0.56`).
   - `endorsement_evidence`: `agreement` responses and corroborating approvals; recorded transparently without ever elevating `proposed_only` into `reported_success` or `Resolved`.
   - `attempt_evidence`: Reports indicating a user tried an option (`"tried calling"`, `"applied"`, `"submitted"`) without confirming resolution.
   - `pending_or_failure_evidence`: Unanswered portals, unanswered calls/emails, rejections, or explicit failures.
   - `success_evidence`: Explicit first-person/OP outcome confirmation guarded against questions (`_QUESTION_GUARD_RE`), advisor credentials, and negations (`_NEGATED_SUCCESS_RE`).
3. **Chronological Conflict & Precedence Rules:**
   - **Regression after prior success** (e.g., initial offer/fix followed later by rejection/recurrence, particularly by OP) overrides prior success into `reported_pending` + `Unresolved` with explicit `contradiction_details`.
   - **Chronological recovery after initial failure** (same actor initially fails/pending then explicitly confirms success later) resolves cleanly to `reported_success` + `Resolved` with reduced historical retry penalty (`0.05`) and documented transition trail.
   - **Multi-report actor conflict** (distinct commenters reporting opposing outcomes without OP failure override) caps status at `Partially Resolved`.
   - **Attempted action + persistent problem** routes to `reported_pending` + `Unresolved`.
4. **Separated False Overclaim Telemetry (`evaluate_resolution_predictions`):** Reports `false_reported_success_rate`, `false_resolved_verdict_rate`, and `false_partially_resolved_rate` individually alongside `combined_false_resolution_overclaim_rate` so partial overclaims cannot hide behind binary resolution checks.

Regression unit tests: `10 passed, 0 failed` across [`backend/tests/test_resolution_engine_v63.py`](../backend/tests/test_resolution_engine_v63.py) and [`backend/tests/test_phase4_phase5_linking_resolution.py`](../backend/tests/test_phase4_phase5_linking_resolution.py).

---

## 2. Issue–Solution Mapping Development-Set Evaluation (`v1`, `n = 64` Pairs)

Evaluated across full reconstructed corpus links (`951` high-confidence linked edges reconstructed via `ContextualIssueSolutionMapper`, never truncated top-12 UI slices) in [`ml/artifacts/v6_3_experiments/linking_resolution_eval/linking_resolution_v1_dev_eval.json`](../ml/artifacts/v6_3_experiments/linking_resolution_eval/linking_resolution_v1_dev_eval.json):

| Metric (`n = 64` Development Set Pairs, 8 Issues) | Value | Confusion / Ranking Breakdown |
|---|---:|---|
| **Pairwise Precision** (`th = 0.56`) | `87.88%` | `TP = 29, FP = 4` |
| **Pairwise Recall** | `85.29%` | `FN = 5, TN = 26` |
| **Pairwise F1** | **`86.57%`** | False-Positive Rate = `13.33%` |
| **MRR / P@1 / P@3** | `1.0000 / 1.0000 / 1.0000` | Every issue's top-3 contains relevant actions |
| **P@5 / Recall@3 / Recall@5** | `0.8250 / 0.7125 / 0.9688` | — |
| **NDCG@3 / NDCG@5** | `0.8368 / 0.9120` | — |

---

## 3. Resolution Engine Development-Set Evaluation (`v1`, `n = 40` Threads) & Corpus (`n = 365`)

### 3.1 Development-Set Metrics (`n = 40` Threads)
- **4-State Accuracy:** **`82.50%`** *(+2.50 pp over V6.0 baseline `80.00%`)*
- **4-State Macro F1:** **`84.43%`** *(+1.91 pp over V6.0 baseline `82.52%`)*
- **4-State Weighted F1:** **`82.19%`**
- **Unresolved Binary Precision / Recall / F1:** `97.06% / 100.00% / 98.51%`
- **Evidence Retrieval Accuracy:** **`90.00%`**
- **False Claim Breakdown (`n = 33` non-success / unresolved gold threads):**
  - `false_reported_success_rate`: **`0.00%` (`0/33`)**
  - `false_resolved_verdict_rate`: **`0.00%` (`0/33`)**
  - `false_partially_resolved_rate`: **`0.00%` (`0/33`)**
  - `combined_false_resolution_overclaim_rate`: **`0.00%` (`0/33`)**

| State (`v1` Dev) | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `reported_success` | 7 | 1.0000 | 0.8571 | **0.9231** |
| `reported_pending` | 12 | 0.6875 | 0.9167 | **0.7857** |
| `proposed_only` | 13 | 0.8889 | 0.6154 | **0.7273** |
| `insufficient_evidence` | 8 | 0.8889 | 1.0000 | **0.9412** |

### 3.2 Community Breakdown (`v1` Dev Set With Low-Support Warnings)
Communities with fewer than 5 threads (`low_support_warning = true`) must not be over-interpreted:
- `r/IIMSambalpurUG` (`n = 15`): Accuracy `93.33%`, Macro F1 `87.13%`
- `r/MBAIndia` (`n = 6`): Accuracy `66.67%`, Macro F1 `87.50%`
- `r/iim_CAT` (`n = 6`): Accuracy `83.33%`, Macro F1 `88.89%`
- Low-support warnings (`n < 5`): `r/CATPrep` (`n=4`), `r/CATStudyRoom` (`n=4`), `r/Indian_Academia` (`n=3`), `r/CATpreparation` (`n=1`), `r/IIMCATPreparation` (`n=1`).

### 3.3 Full `v6.1.0` Corpus Behavior Under P0-B Rules (`n = 365` Threads)
- **States:** `reported_success = 12`, `reported_pending = 55`, `proposed_only = 147`, `insufficient_evidence = 151`
- **Verdicts:** `Resolved = 8`, `Partially Resolved = 4`, `Unresolved = 202`, `Insufficient Evidence = 151`
- **Contradictory / pending follow-up threads surfaced:** **`32`** threads (up from `26` before explicit attempt/regression separation).

---

## 4. Frozen Independent Blind `v2` Evaluation Protocol

Generated by [`ml/experiments/v6_3/prepare_independent_linking_resolution_eval.py`](../ml/experiments/v6_3/prepare_independent_linking_resolution_eval.py) across the combined `816`-thread real corpus using **confidence-independent** lexical/structural strata (zero model scores, `0%` overlap with `v1`):

- **Issue–Solution `v2` Blind Package (`180` pairs):** `179` from `v6_3_expansion` + `1` from `v6_1_baseline`; stratified evenly across all 8 issue domains (`20–23` pairs per domain) balancing `actionable_lexical`, `trap_or_social`, and `general_discussion`. CSV SHA-256: `7607459c9d35229a4e3a7a303d1fde640725620666014b21d37d174d9dbdf61e`.
- **Resolution `v2` Blind Package (`120` threads):** `60` from `v6_1_baseline` + `60` from `v6_3_expansion`; `20` each from `outcome_or_pending_cues`, `action_or_advice_cues`, and `low_signal_or_open`, with chronological `thread_timeline_digest`. CSV SHA-256: `de37a61bf61f56bcdc7b69886237cc0d54d7a8f96dd908c20ea94ae2d68d4c01`.
- **Manifest:** [`v2_protocol_freeze_manifest.json`](../data/annotations/v6_3_package/linking_and_resolution_v2/v2_protocol_freeze_manifest.json).
