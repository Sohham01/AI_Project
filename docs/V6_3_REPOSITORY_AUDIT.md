# RCI V6.3 — Pre-Implementation Repository Audit & Verified Baseline

**Audit Date:** 2026-11-25  
**Git HEAD:** `e20287d` (`fix(ci): guard optional torch/bertopic unit test and track lightweight topic model registry`)  
**Python Environment:** `D:\AI_Project\backend\.venv\Scripts\python.exe` (`Python 3.14.6`, CUDA enabled on `NVIDIA GeForce RTX 4060 Laptop GPU`)  
**Baseline Test Result:** `pytest backend/tests -q` → **`51 passed, 0 failed`** (`75.22s`)  
**Upstream Archive Reachability:** `ArcticShiftClient` (`https://arctic-shift.photon-reddit.com`) verified reachable (`OK`).

---

## 1. Preserved User Assets & Frozen Benchmark Invariants

- **Untracked User Deliverables Preserved Untouched:**
  - `data/annotations/templates/RCI_Manual_Annotation_Template.xlsx`
  - `docs/academic_submission/RCI_V6_Team_Presentation_Guide.pdf`
- **Frozen Supervised Benchmark (Read-Only):**
  - `data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv` (`459` real comments across `200` threads; `367` 5-fold group CV train/val across `163` threads; `92` untouched frozen test comments across `37` unseen threads):
    - `SHA256 = 8415bdacbb9661071d3b88150f651b2fe919381c1aefd02ac43589890ba325cd`
  - `ml/artifacts/v5_experiments/manifests/v5_frozen_splits.json`:
    - `test_sha256 = eb26459fc0d87b0631d6c0930a7a32a7ff6175b66258fcc3025c4faf805f7283`
- **Preserved Production Serving & Topic Baselines:**
  - `data/live_corpus/corpus.json` (`365` threads, `3,557` comments across 8 subreddits: `IIMSambalpurUG` `211`, `MBAIndia` `46`, `IIMCATPreparation` `41`, `iim_CAT` `18`, `CATPrep` `18`, `CATStudyRoom` `15`, `CATpreparation` `12`, `Indian_Academia` `4`)
  - `ml/artifacts/live_corpus/topic_discovery.json` (`nmf_v1`, 10 topics, `3,459` eligible documents)
  - `ml/artifacts/live_corpus/evaluation.json` and `ml/artifacts/live_corpus/predictions.json`
  - Canonical 6-class `response` order: `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
  - Canonical verification architectures (`REQUIRED_ARCHITECTURES` in `ml/evaluation/benchmark_verifier.py`): `logistic_regression`, `ann_mlp`, `cnn_textcnn`, `transformer_muril_dapt_v3`.

---

## 2. End-to-End Data & ML Pipeline Map

1. **Ingestion & Sanitization:** `backend/app/services/ingestion/arctic_shift.py`, `backend/app/services/ingestion/collection.py`, `backend/run_arctic_shift_pilot.py`, `backend/import_reddit_export.py` → outputs validated `format_version=1` JSON + `review_manifest.jsonl` with author stripping and PII redaction (`<EMAIL>`, `<PHONE>`, `<USER_MENTION>`).
2. **Multilingual LID & Informal Normalization:** `backend/app/services/nlp/indian_reddit_lid.py`, `script_detector.py`, `code_mix_detector.py`, `preprocessing/context_normalizer.py`.
3. **Splitting & 5-Task Supervised Training:** `ml/data/splitting.py` (`post_id` group-aware), `ml/experiments/v5_optimization/common.py`, `ml/models/neural_baselines.py` (LogReg, ANN/MLP, TextCNN), `ml/models/transformer/` (MuRIL-DAPT-WWM, XLM-RoBERTa).
4. **Dual Topic Discovery:** `backend/app/services/nlp/topic_discovery/nmf_topic_discoverer.py` (`nmf_v1` default) alongside `ml/topics/bertopic/` (`bertopic_v1` companion via `intfloat/multilingual-e5-small`).
5. **Contextual Issue–Solution Mapping & Resolution Analysis:** `ml/linking/contextual_mapper.py` (E5 + TF-IDF + parent/thread structure + detector confidence − negation penalty) and `ml/linking/resolution_engine.py` (4 canonical states + 4 verdict tiers).
6. **Evaluation, Integrity & Serving:** `ml/evaluation/benchmark_verifier.py` (`7/7` checks), `backend/app/api/demo.py`, `api/index.py` (slim zero-ML Vercel runtime).

---

## 3. Defects & Research Gaps Confirmed During Audit (Targets for V6.3 Sprints)

| Sprint | Target Area / File | Confirmed Gap / Defect in `e20287d` |
|---|---|---|
| **P0-A** | `ml/topics/bertopic/embeddings.py` | `_cache_key()` hashes only `texts[:200]` and `texts[-100:]`, ignoring middle documents (`texts[200:-100]`), truncating to 16 hex chars, and omitting model revision/weight fingerprint, preprocessing version, and corpus version. |
| **P0-B** | `ml/linking/resolution_engine.py` | (1) Hard-codes fallback `datetime(2026, 4, 15, tzinfo=timezone.utc)`; (2) treats `pred_resp == "agreement"` identically to outcome confirmation and elevates `proposed_only` to `"Partially Resolved"` when `best_sol_rel >= 0.72` without execution/outcome evidence; (3) ignores chronological reversal (later failure following earlier success, or later recovery after earlier failure); (4) reports only `"Resolved"` false resolution rate rather than reporting false `reported_success` and false `Partially Resolved` rates separately. |
| **P0-C** | `ml/experiments/v5_optimization/score_human_iaa.py`, `data/annotations/` | `real_reddit_labeled_v1` (`459` rows) has 1 rubric curator (`0` 2-human pairs); `iaa_50_human_annotator_studio.csv` captures 1 human reviewer vs V2 Pass-1 rubric (`0/5` threshold gates passed); `iaa_50_human_annotator_b_studio.csv` and `iaa_double_annotation_50.csv` are completely blank. Fleiss' $\kappa$, per-class multi-label warnings, taxonomy versioning, and blind Round-2 workbook generation need unified support. |
| **P1-A** | `data/live_corpus/corpus.json` & ingestion | Baseline corpus has `365` threads (`3,557` comments) with `57.8%` (`211/365`) concentrated in `r/IIMSambalpurUG`; needs versioned real expansion targeting `700–1,000` unique threads while strictly isolating from the frozen `92`-comment / `37`-thread test set. |
| **P1-B** | `data/annotations/multilingual_eval_real_v1.csv` | `n = 84` real comments contain `0` Bengali-script, `0` Odia-script, `0` Banglish, and `1` Romanized Odia sample (`cmt_01936`). Needs expanded candidate sampling from the expanded real corpus separated cleanly into native-script vs Romanized vs synthetic probes. |
| **P1-C** | `ml/models/`, `backend/requirements-ml.txt` | Missing experimental `XGBoost` benchmark across all 5 tasks (both sparse TF-IDF CSR and SVD + linguistic feature representations) under strict `n=367` 5-fold group-aware CV without touching `requirements.txt` or production selection. |
| **P1-D** | `ml/topics/bertopic/` | `docs/BERTOPIC_IMPLEMENTATION_AND_COMPARISON.md` tested only 3 seeds (`[42, 101, 202]`) with `seed=202` collapsing to `2` clusters and `38.2%` outlier rate; needs systematic `>= 5`-seed stability across UMAP/HDBSCAN configurations (e.g., `leaf` vs `eom`) and on the expanded V6.3 corpus. |
| **P1-E** | `data/annotations/issue_solution_eval_v1.csv`, `resolution_eval_v1.csv` | Existing `64`-pair and `40`-thread eval sets are V6.0 development benchmarks; need strictly disjoint, confidence-independent blind `v2` evaluation packages (`150–200` linking pairs, `100+` threads) plus separated false-claim reporting. |
| **P2** | `.github/workflows/ci.yml`, `README.md`, `docs/` | Base CI runs only slim `backend/requirements.txt`; need a parallel `test-ml-optional` job covering `xgboost` + `bertopic` + P0/P1 regression tests, plus full epistemic alignment across README and technical reports. |
