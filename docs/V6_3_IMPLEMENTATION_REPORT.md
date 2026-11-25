# Reddit Community Intelligence (RCI) — V6.3 Implementation Report

**Cycle:** RCI V6.3 — XGBoost + Reliability Fixes + Dataset Expansion + Research Validation  
**Baseline Commit Reference:** `e20287d`  
**Preservation Invariants Verified:** `data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv` (`459` rows, `367` trainval / `92` frozen test across `37` unseen threads, `SHA-256 = 8415bdacbb9661071d3b88150f651b2fe919381c1aefd02ac43589890ba325cd`), `v5_frozen_splits.json` (`test_sha256 = eb26459fc0d87b0631d6c0930a7a32a7ff6175b66258fcc3025c4faf805f7283`), `data/live_corpus/corpus.json` (`v6.1.0`), `ml/artifacts/live_corpus/topic_discovery.json` (`nmf_v1`), canonical 6-class `response` order, and `REQUIRED_ARCHITECTURES` (`benchmark_verifier.py`) remain **100% intact**.

---

## A. Executive Summary

V6.3 addresses the major research-validity, reproducibility, dataset-balance, and benchmarking gaps identified in [V6_3_REPOSITORY_AUDIT.md](./V6_3_REPOSITORY_AUDIT.md) without disrupting deployed V6.1/V6.2 zero-ML Studio serving:

1. **P0 Reliability Fixes:** Eliminated partial-slice embedding cache collisions (`v2.0` full-corpus streaming SHA-256 + model fingerprint in `ml/topics/bertopic/embeddings.py`) and removed hardcoded timestamps, `agreement`-as-resolution elevation, and order-blind contradiction scoring in `ml/linking/resolution_engine.py`.
2. **Real Corpus Rebalancing (`365 → 816` Threads, `3,557 → 7,570` Comments):** Collected `+451` authentic public Reddit threads (`+4,013` comments) into `data/corpus_v6_3/new_unannotated_corpus_v6_3.json` with `0` frozen-test overlaps, cutting `r/IIMSambalpurUG` concentration from `57.81% → 25.86%` and introducing real Bengali-script (`13`), Banglish (`95`), and Roman-Odia (`14`) comments.
3. **Experimental XGBoost Benchmarking:** Evaluated `XGBoostBaselineClassifier` (`xgb_sparse_tfidf` and `xgb_svd_linguistic`) strictly on 5-fold `post_id`-grouped CV (`trainval n = 367`, `3` seeds) alongside LogReg, ANN, TextCNN, and saved Transformer OOF metrics without querying the frozen `n = 92` test split.
4. **BERTopic Multi-Seed Stability Audit:** Audited `5–7` seeds across `6` baseline configs (`n = 3,459`) and `3` expanded-corpus configs (`n = 7,152`), exposing `eom` collapse modes (`1/7–2/7` baseline, `1/5–2/5` expanded) vs. `0` collapses under `leaf` selection (`NMI = 0.7697–0.8090`).
5. **Blind Manual Annotation & Independent `v2` Protocol Packages:** Prepared `data/annotations/v6_3_package/` with `iaa_blind_round2/` (`80` items × 2 annotators), `multilingual_bn_or_v2/` (`117` candidates), and `linking_and_resolution_v2/` (`180` pairs + `120` threads, `0%` `v1` dev overlap) ready for human work.

---

## B. Reliability Fixes Completed

### B.1 P0-A — Full-Corpus Embedding Cache Identity (`ml/topics/bertopic/embeddings.py`)
- **Defect Resolved:** Legacy cache keying hashed only `texts[:200] + texts[-100:]` in a 16-hex digest and ignored model revision hashes, preprocessing versions, and corpus versions — allowing silent stale cache hits when middle documents or weights changed.
- **V6.3 Implementation:**
  - Replaced legacy keyer with `EMBEDDING_CACHE_SCHEMA_VERSION = "v2.0"` streaming length-prefixed UTF-8 SHA-256 over **every** text in exact order (or explicit sorted order when `order_independent=True`).
  - Added `resolve_model_fingerprint()` hashing local checkpoint config/headers and HuggingFace `refs/main` + explicit `model_revision`, combined with `preprocessing_version` and `corpus_version`.
  - Isolated v2 cache storage under `<cache_dir>/v2/` with strict JSON `.meta.json` validation so unversioned v1 `.npy` files can never masquerade as valid v2 hits.
  - Added single-slot `_ACTIVE_ENCODER` instance reuse + `gc.collect()` to prevent repeated heavyweight model reloads across linking/topic runs.
- **Verification:** `5 passed, 0 failed` (`test_embedding_cache_identity.py` + `test_bertopic_pipeline.py`).

### B.2 P0-B — Dynamic Timestamps, Conflict Precedence & Confirmation Semantics (`ml/linking/resolution_engine.py`)
- **Defects Resolved:** Removed hardcoded `datetime(2026, 4, 15)` fallback; stopped implicit wall-clock usage; prevented `agreement` comments from counting as resolution confirmation; separated outcome reversal (`success → failure`) vs. retry recovery (`failure → success`) and multi-user disputes; and separated false `reported_success`, `Resolved`, and `Partially Resolved` metrics.
- **V6.3 Implementation:**
  - `resolve_reference_time()` deterministically selects `explicit_reference_time → dataset_observation_timestamp → corpus_max_observation_timestamp / thread_latest_activity_timestamp`, stamping full `timestamp_semantics` on outputs.
  - Five explicit discourse tiers: `proposed_solution`, `endorsement_evidence` (`agreement`, never upgrades `proposed_only`), `attempt_evidence`, `pending/failure/contradictory_evidence`, and `success_evidence` (negation + question + credential guarded).
  - Chronological conflict rules: OP/later regression overrides prior success into `reported_pending` / `Unresolved`; same-actor later recovery resolves to `reported_success` / `Resolved`; multi-speaker conflicts cap at `Partially Resolved`.
- **Verification:** `10 passed, 0 failed` (`test_resolution_engine_v63.py` + `test_phase4_phase5_linking_resolution.py`).

---

## C. Annotation Reliability & IAA Framework

Created [`ml/evaluation/annotation_reliability.py`](../ml/evaluation/annotation_reliability.py) and ran [`ml/experiments/v6_3/audit_and_prepare_iaa.py`](../ml/experiments/v6_3/audit_and_prepare_iaa.py):

- **Honest Audit of Pre-Existing Files ([`existing_annotation_audit.json`](../ml/artifacts/v6_3_experiments/annotation_audit/existing_annotation_audit.json)):**
  - `master_labeled_v1.csv` (`459` rows): single-curator rubric provenance (`rubric_curated_real_reddit_v1`); contains `0` paired second-human judgments.
  - `iaa_50_human_annotator_studio.csv` (`50` rows): 1 blind human (`researcher_1`) vs. V2 Pass-1 rubric (`0/5` threshold gates passed; Track B V2 remains blocked).
  - `iaa_50_human_annotator_b_studio.csv` & `iaa_double_annotation_50.csv`: completely blank templates.
  - True multi-human IAA is explicitly marked `PENDING_SECOND_INDEPENDENT_ANNOTATOR` with zero fabricated numbers.
- **Multi-Rater IAA & Adjudication Toolkit Implemented (`4/4` unit tests passing in `test_annotation_reliability.py`):**
  - Supports $K \ge 2$ annotators, round IDs, `TAXONOMY_VERSION = "rci_taxonomy_v2.0"`, and `GUIDELINE_VERSION = "ANNOTATION_GUIDELINES_v2.0"`.
  - Computes pairwise Cohen's $\kappa$, multi-rater Fleiss' $\kappa$, general coincidence-matrix Krippendorff's $\alpha$ with missingness support, MASI multi-label $\alpha$, per-class binary agreement warnings on low-prevalence labels, and non-destructive `merge_independent_annotations_and_adjudicate()`.
  - Blindness checker `verify_blind_sheet()` verifies zero peer or model columns exist before distribution.

---

## D. Dataset Expansion Summary

Script: [`backend/expand_real_corpus_v63.py`](../backend/expand_real_corpus_v63.py) · Detailed Report: [`docs/V6_3_DATASET_EXPANSION_REPORT.md`](./V6_3_DATASET_EXPANSION_REPORT.md) · Data Dictionary: [`docs/V6_3_CORPUS_DATA_DICTIONARY.md`](./V6_3_CORPUS_DATA_DICTIONARY.md).

- **Corpus Scale Growth:** Expanded cleanly from `365` baseline threads (`3,557` comments) to **`816` combined unique real threads (`7,570` comments)** via `+451` newly collected threads (`+4,013` comments, `+123.56%` thread growth) stored separately in `data/corpus_v6_3/new_unannotated_corpus_v6_3.json`.
- **Institutional Rebalancing:** `r/IIMSambalpurUG` share dropped from **`57.81%` (`211/365`) → `25.86%` (`211/816`)**, complemented by `r/MBAIndia` (`195`), `r/IIMCATPreparation` (`119`), `r/CATPrep` (`63`), `r/Indian_Academia` (`52`), `r/CATpreparation` (`50`), `r/CATStudyRoom` (`47`), `r/kolkata` (`24`), `r/Bhubaneswar` (`23`), `r/iim_CAT` (`18`), and `r/westbengal` (`14`).
- **Deterministic Filtering & Zero-Leakage Guarantee:** Filtered `236` bot/AutoMod comments, `182` duplicates, `52` empty threads, `17` GIF/media comments, `9` off-topic posts, `6` deleted/removed comments, and `1` spam item. All `37` frozen test threads (`526` comments) were checked via ID, exact/normalized title-body hash, and bigram similarity: **`0` quarantined, `0` frozen-test leakage**, and **`0` PII/URLs/usernames** verified by `test_dataset_expansion_and_leakage_v63.py` (`3/3` passed).

---

## E. Multilingual Coverage Update

Script: [`ml/experiments/v6_3/sample_bengali_odia_candidates.py`](../ml/experiments/v6_3/sample_bengali_odia_candidates.py) · Artifact: [`coverage_and_eval_report.json`](../ml/artifacts/v6_3_experiments/multilingual_bn_or/coverage_and_eval_report.json).

- **Existing V1 Baseline Preserved Unchanged:** `multilingual_eval_real_v1.csv` (`n = 84`: `39 en`, `34 hi-en`, `5 hi`, `5 und`, `1 or-en`, `0 bn/bn-en`) remains untouched (`88.10%` acc, `85.29%` Macro F1, `100.0%` script acc), strictly separate from the `24` synthetic diagnostic probes.
- **V6.3 Authentic Bengali & Odia Expansion Scan:** Scanning all `3,876` eligible new comments alongside `3,162` baseline comments surfaced **`19` native Bengali-script**, **`93` Romanized Bengali (Banglish)**, **`23` Romanized Odia**, and **`597` mixed-language control** raw candidates (`0` native Odia Unicode `0B00–0B7F` found even across Bhubaneswar/Odisha threads).
- **V2 Candidate Package Assembled (`117` Items):** `14` `bengali_native_script`, `45` `romanized_bengali`, `23` `romanized_odia`, and `35` `mixed_language_control` exported with explicit `language_candidate_detector_hypothesis` warnings and blank `gold_*` targets. Genuine `bn`/`bn-en`/`or-en` generalization metrics will be reported only after human annotation of `v2`.

---

## F. XGBoost Experimental Benchmark

Module: [`ml/models/xgboost_baseline.py`](../ml/models/xgboost_baseline.py) · Runner: [`ml/experiments/v6_programme/run_xgboost_benchmark.py`](../ml/experiments/v6_programme/run_xgboost_benchmark.py) · Full Report: [`docs/V6_3_XGBOOST_BENCHMARK_REPORT.md`](./V6_3_XGBOOST_BENCHMARK_REPORT.md).

Evaluated strictly on identical `post_id`-grouped 5-fold CV over `trainval (n = 367, 163 threads)` across `3` seeds (`42, 101, 202`), leaving the frozen `n = 92` test set **untouched**:

| Model Family | 5-Task Mean OOF F1 | Mean Fold Std | Mean ECE | Total Train Time (5 tasks) | `issue` | `sentiment` | `emotion` | `response` | `solution` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Transformers (`MuRIL-DAPT-WWM + XLM-R` OOF Ref.)** | **`63.05%`** | — | — | ~522s (GPU) | **`58.61%`** | **`69.22%`** | **`61.58%`** | **`49.66%`** | **`76.17%`** |
| **ANN (2-Layer MLP)** | `50.50%` | `±2.84%` | `0.0631` | `13.74s` | `48.59%` | **`53.65%`** | `44.51%` | **`43.11%`** | `62.63%` |
| **Logistic Regression** | `49.20%` | `±5.01%` | `0.0813` | `4.33s` | **`49.88%`** | `51.92%` | **`49.36%`** | `38.46%` | `56.36%` |
| **XGBoost (SVD-128 + 14 Ling. Cues)** | `48.57%` | `±4.45%` | **`0.0187`** | `29.88s` | `39.16%` | `51.93%` | `44.81%` | `39.77%` | **`67.16%`** |
| **XGBoost (30k Sparse CSR TF-IDF)** | `47.92%` | `±5.89%` | `0.0355` | `127.68s` | `43.64%` | `49.38%` | `45.63%` | `37.66%` | `63.27%` |
| **1D TextCNN** | `42.46%` | `±4.00%` | `0.0671` | `16.02s` | `30.22%` | `49.27%` | `34.62%` | `38.79%` | `59.38%` |

- **Key Finding:** `xgb_svd_linguistic` achieved the strongest non-transformer OOF score on **`solution` (`67.16%`)** and best raw calibration (**`ECE = 0.0187`**), yet struggled on rare multi-label `issue` classes (`academic_workload`, `extracurriculars = 0.00` F1 on sparse tree splits at `n ~ 293` rows/fold). XGBoost remains **strictly experimental** (`3/3` unit tests pass in `test_xgboost_baseline.py`).

---

## G. BERTopic Stability Findings

Runner: [`ml/experiments/v6_3/run_bertopic_stability.py`](../ml/experiments/v6_3/run_bertopic_stability.py) · Full Report: [`docs/V6_3_BERTOPIC_STABILITY_REPORT.md`](./V6_3_BERTOPIC_STABILITY_REPORT.md).

- **Baseline Corpus (`n = 3,459` Docs, `5–7` Seeds):**
  - `C0_v1_baseline_eom` (`e5-small + mcs=25, eom`): `[35, 31, 2, 30, 33, 34, 32]`, **`1/7` collapse (`seed=202`)**, mean outliers `30.5% ± 12.5%` (`38.16%` at canonical `seed=42`), NMI `0.5757`.
  - `C1_eom_mcs35` (`eom`): **`2/7` collapses** (`[20, 2, 2, 20, 23, 20, 20]`), NMI `0.4335`.
  - `C2_leaf_mcs25` (`e5-small + leaf`): **`0/7` collapses** (`[37, 35, 35, 34, 34, 39, 34]`), NMI `0.7697`, non-outlier ARI **`0.9332`**, outliers `41.2% ± 2.3%`.
  - `C5_minilm_leaf_mcs25` (`MiniLM + leaf`): **`0/5` collapses** (`[43, 45, 42, 42, 41]`), highest NMI **`0.8090`**, non-outlier ARI **`0.9467`**, outliers `35.3% ± 3.7%`.
- **Expanded V6.3 Corpus (`n = 7,152` Docs, `5` Seeds):**
  - `eom` collapsed at `seed=42` (`1/5` collapse under `C0`, `2/5` collapses under `C1`), whereas `C2_leaf_mcs25_expanded_v63` achieved **`0/5` collapses** (`[57, 51, 52, 50, 49]`, non-outlier ARI `0.9072`, outliers `59.1%`).
- **Decision:** Keep **`nmf_v1`** (`topic_discovery.json`, 10 deterministic topics, untouched) as RCI's default production topic model and keep BERTopic as the interactive exploratory companion.

---

## H. Issue–Solution Mapping & Resolution Evaluation

Runner: [`ml/experiments/v6_3/evaluate_linking_and_resolution_v63.py`](../ml/experiments/v6_3/evaluate_linking_and_resolution_v63.py) · Full Report: [`docs/V6_3_LINKING_AND_RESOLUTION_EVAL_REPORT.md`](./V6_3_LINKING_AND_RESOLUTION_EVAL_REPORT.md).

- **Development Set `v1` Updated Performance (`DEVELOPMENT_SET_ONLY_V1`):**
  - **Issue–Solution (`n = 64` pairs, `951` reconstructed corpus links):** Pairwise F1 = **`86.57%`**, Precision = `87.88%`, Recall = `85.29%`, MRR = `1.00`, `P@3 = 1.00`.
  - **Resolution (`n = 40` threads, P0-B engine):** 4-State Accuracy = **`82.50%`**, Macro F1 = **`84.43%`**, Weighted F1 = **`82.19%`**, Unresolved F1 = **`98.51%`**, Evidence Retrieval = **`90.00%`**, and **`0.00%` (`0/33`)** across `false_reported_success_rate`, `false_resolved_verdict_rate`, and `false_partially_resolved_rate`.
  - **Corpus `n = 365` Distribution Under P0-B:** `12` `reported_success`, `55` `reported_pending`, `147` `proposed_only`, `151` `insufficient_evidence` (`8` `Resolved`, `4` `Partially Resolved`, `202` `Unresolved`, `151` `Insufficient Evidence`, `32` contradictory/pending follow-up threads).
- **Independent Blind `v2` Protocol Frozen (`0%` `v1` Overlap):** `180` linking pairs + `120` threads sampled confidence-independently across all `816` threads and `8` issue domains (`data/annotations/v6_3_package/linking_and_resolution_v2/`).

---

## I. Manual Annotation Package Ready for Human Work

Complete operator guide: [`data/annotations/v6_3_package/README_ANNOTATION_PACKAGE.md`](../data/annotations/v6_3_package/README_ANNOTATION_PACKAGE.md)

1. **`iaa_blind_round2/`** (`80` comments × 2 blind annotators `.csv` + `.xlsx`, `0` frozen-test overlap, `rci_taxonomy_v2.0`, adjudication template).
2. **`multilingual_bn_or_v2/`** (`117` authentic candidate comments `.csv` + `.jsonl` + `.xlsx`: `14` BN script, `45` Banglish, `23` Roman-Odia, `35` mixed control).
3. **`linking_and_resolution_v2/`** (`180` blind issue–solution pairs + `120` blind thread timelines `.csv` + `.xlsx`, `0` `v1` overlap, protocol freeze manifest).

---

## J. Remaining Limitations and Next Research Cycle (V7)

1. **Human Annotation Completion Needed Before Claims:** Until two human annotators complete `iaa_blind_round2/`, `multilingual_bn_or_v2/`, and `linking_and_resolution_v2/`, true two-human IAA, representative real-world Bengali/Odia LID accuracy, and unseen `v2` resolution/linking generalization remain **pending** (zero estimated/synthetic numbers reported).
2. **Corpus Language Availability:** Even with `r/kolkata`, `r/westbengal`, and `r/Bhubaneswar` queries (`HTTP 429/422` rate limits respected cleanly), native Odia script (`U+0B00–U+0B7F`) did not appear in retrieved public threads (`0` docs; `or` appears only as `or-en` Roman-Odia).
3. **Production Studio Serving Pinned to `v6.1.0`:** Live Studio APIs (`data/live_corpus/corpus.json` and `ml/artifacts/live_corpus/`) remain untouched until Round-2 labeled data allows retraining and blind `v2` re-verification.

---

## V6.3 Changelog Summary

- **Added:** `ml/evaluation/annotation_reliability.py`, `ml/models/xgboost_baseline.py`, `backend/expand_real_corpus_v63.py`, `ml/experiments/v6_programme/run_xgboost_benchmark.py`, `ml/experiments/v6_3/{audit_and_prepare_iaa,sample_bengali_odia_candidates,run_bertopic_stability,prepare_independent_linking_resolution_eval,evaluate_linking_and_resolution_v63}.py`, 5 new unit/regression test suites under `backend/tests/`, V6.3 corpus (`data/corpus_v6_3/`), blind annotation bundles (`data/annotations/v6_3_package/`), and 7 V6.3 technical/implementation docs.
- **Fixed:** P0-A streaming full-corpus v2.0 embedding cache + model fingerprint (`ml/topics/bertopic/embeddings.py`, `pipeline.py`); P0-B dynamic `resolve_reference_time()`, `agreement` ≠ confirmation, chronological reversal precedence, and separated false-overclaim metrics (`ml/linking/resolution_engine.py`).
- **Updated:** `backend/requirements-ml.txt` (`xgboost==3.4.1`), `.github/workflows/ci.yml` (`test-ml-optional` job + expansion corpus PII scan), `README.md` + historical V6 report addenda for strict OOF & IAA epistemic transparency.
