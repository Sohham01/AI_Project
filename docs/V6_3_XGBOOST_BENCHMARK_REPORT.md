# V6.3 — Experimental XGBoost 5-Fold Group-Aware CV Benchmark Report

> **Experimental Governance & Frozen Test Protection:** XGBoost (`tree_method="hist"`) was evaluated strictly as an **experimental comparator** (`experimental_comparator_not_promoted_to_production`) across identical `post_id`-grouped 5-fold cross-validation (`trainval n = 367` across `163` threads) and `3` random seeds (`[42, 101, 202]`). Per V6.3 protocol, the frozen `n = 92` real held-out test split (`37` unseen threads) was **intentionally NOT evaluated**, production V6.1 (`MuRIL-DAPT-WWM + XLM-R`) remains unchanged, and `REQUIRED_ARCHITECTURES` in the benchmark verifier remains unchanged.

---

## 1. Exact Task Formulations & Hyperparameters

To ensure a strictly fair comparison with `LogisticRegression`, `ANN (MLP)`, `1D TextCNN`, and `MuRIL-DAPT-WWM + XLM-R`, XGBoost mirrors RCI's exact schema (`ml/experiments/v5_optimization/common.py`) without altering labels or objectives:

| Task | Objective / Structure | Classes / Labels | XGBoost Hyperparameters |
|---|---|---|---|
| **`issue`** | 8 independent `binary:logistic` estimators (multi-label OVR + per-label OOF F1 threshold tuning) | `academic_workload`, `administrative_communication`, `campus_infrastructure`, `curriculum_teaching`, `extracurriculars`, `financial_concerns`, `hostel_facilities`, `placements` | `tree_method="hist"`, `n_estimators=300`, `max_depth=4`, `learning_rate=0.06`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.5`, `reg_lambda=2.0`, `early_stopping_rounds=25`, capped `scale_pos_weight <= 6.0` |
| **`sentiment`** | 3-class `multi:softprob` with inverse-frequency sample weights | `negative`, `neutral`, `positive` | Same histogram boosting & early stopping |
| **`emotion`** | 6 independent `binary:logistic` estimators (multi-label OVR + per-label OOF threshold tuning) | `confidence`, `curiosity`, `frustration`, `gratitude`, `hope`, `stress` | Same histogram boosting & early stopping |
| **`response`** | 6-class `multi:softprob` preserving canonical RCI taxonomy order | `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument` | Same histogram boosting & early stopping |
| **`solution`** | Single `binary:logistic` estimator + OOF F1 threshold tuning | `not_solution`, `solution` | Same histogram boosting & early stopping |

### Two Tested Feature Pipelines (`ml/models/xgboost_baseline.py`)
1. **`xgb_sparse_tfidf`:** Fit inside each CV training fold only (zero fold leakage): Word (`1–2` n-gram, `max_features=12,000`, `sublinear_tf=True`) + Character (`3–5` char_wb n-gram, `max_features=18,000`) sparse CSR TF-IDF (`30,000` max dims).
2. **`xgb_svd_linguistic`:** Fit inside each CV training fold only: `128`-component `TruncatedSVD` over the same Word+Char TF-IDF matrix concatenated with `14` deterministic linguistic/discourse/script cues (character length, word count, punctuation counts `?` / `!`, Hinglish marker density, advice cue matches, experience cue matches, agreement cue matches, counterargument cue matches, URL/email presence, emoji count, native Indic script count).

---

## 2. 5-Task Aggregate 5-Fold Group-Aware OOF CV Summary (`n = 367`)

Source artifacts: [`ml/artifacts/v6_3_experiments/xgboost_v1/xgboost_cv_benchmark.json`](../ml/artifacts/v6_3_experiments/xgboost_v1/xgboost_cv_benchmark.json) and saved 5-fold validation OOF metrics from [`ml/artifacts/live_corpus/evaluation.json`](../ml/artifacts/live_corpus/evaluation.json).

| Architecture | Feature Representation | 5-Task Mean OOF F1 | Mean Fold Std | Mean 5-Bin ECE | Total 5-Task Train Time | Inference (ms/doc) | Avg Model Size |
|---|---|---:|---:|---:|---:|---:|---:|
| **Deployed Transformers (`MuRIL-DAPT-WWM + XLM-R`)** *(Ref. OOF)* | `parent_pair` / comment contextual subwords | **`63.05%`** | — | *(0.059 test)* | ~522s (GPU) | ~5.47 ms | ~237.6M / 278M params |
| **ANN (2-Layer MLP)** | Dense TF-IDF (3,072 dims) | `50.50%` | `±2.84%` | `0.0631` | `13.74s` | `0.167 ms` | `3,165.6 KB` |
| **Logistic Regression** | CSR Word+Char TF-IDF (30k max) | `49.20%` | `±5.01%` | `0.0813` | `4.33s` | `0.167 ms` | `309.9 KB` |
| **XGBoost (Experimental Dense)** | `TruncatedSVD-128` + `14` Ling. Cues | `48.57%` | `±4.45%` | **`0.0187`** | `29.88s` | `0.270 ms` | `3,916.4 KB` |
| **XGBoost (Experimental Sparse)** | CSR Word+Char TF-IDF (30k max) | `47.92%` | `±5.89%` | `0.0355` | `127.68s` | `0.222 ms` | `599.8 KB` |
| **1D TextCNN (Kim 2014)** | Token Embeddings (`k=2..5`) | `42.46%` | `±4.00%` | `0.0671` | `16.02s` | **`0.030 ms`** | `1,125.3 KB` |

---

## 3. Per-Task 5-Fold OOF Macro F1 (%) & Seed/Fold Variance

| Task | LogReg | ANN (MLP) | 1D TextCNN | XGB (Sparse TF-IDF) | XGB (SVD + Ling.) | Deployed Transformer OOF Ref. |
|---|---:|---:|---:|---:|---:|---:|
| **`issue`** *(8-label)* | `49.88%` (±4.24 fold) | `48.59%` (±1.62 fold) | `30.22%` (±2.75 fold) | `43.64%` *(±0.23 seed / ±5.57 fold)* | `39.16%` *(±0.92 seed / ±3.48 fold)* | **`58.61%`** *(MuRIL-DAPT-WWM)* |
| **`sentiment`** *(3-class)* | `51.92%` (±6.55 fold) | `53.65%` (±2.94 fold) | `49.27%` (±5.16 fold) | `49.38%` *(±0.03 seed / ±6.86 fold)* | `51.93%` *(±1.23 seed / ±5.02 fold)* | **`69.22%`** *(MuRIL-DAPT-WWM)* |
| **`emotion`** *(6-label)* | `49.36%` (±3.62 fold) | `44.51%` (±5.07 fold) | `34.62%` (±8.20 fold) | `45.63%` *(±0.23 seed / ±4.76 fold)* | `44.81%` *(±0.30 seed / ±5.49 fold)* | **`61.58%`** *(MuRIL-DAPT-WWM)* |
| **`response`** *(6-class)* | `38.46%` (±4.27 fold) | `43.11%` (±1.95 fold) | `38.79%` (±1.51 fold) | `37.66%` *(±1.58 seed / ±6.16 fold)* | `39.77%` *(±1.76 seed / ±3.41 fold)* | **`49.66%`** *(XLM-R)* |
| **`solution`** *(binary)* | `56.36%` (±6.38 fold) | `62.63%` (±2.60 fold) | `59.38%` (±2.39 fold) | `63.27%` *(±1.32 seed / ±6.10 fold)* | **`67.16%`** *(±2.31 seed / ±4.83 fold)* | **`76.17%`** *(XLM-R)* |

### Key Per-Class & Diagnostic Observations
1. **Where XGBoost Wins Among Non-Transformer Models (`solution`):** Combining `TruncatedSVD-128` semantic components with `14` explicit discourse/action cues enables `xgb_svd_linguistic` to reach **`67.16%` OOF F1** on `solution` (`+10.80 pp` above LogReg `56.36%`, `+4.53 pp` above ANN `62.63%`, `+7.78 pp` above TextCNN `59.38%`), while maintaining the lowest 5-bin ECE (**`0.0135`**).
2. **Where Sparse Linear / Dense ANN Win (`issue`, `emotion`, `response`):**
   - On sparse multi-label **`issue` (`8` labels)**, both XGBoost variants scored **`0.00` F1** on low-support rare labels (`academic_workload` `n=16`, `extracurriculars` `n=27`), whereas LogReg (`49.88%`) and ANN (`48.59%`) retained recall across rare keywords because linear weights accumulate weak evidence additively across thousands of sparse terms rather than thresholding single tree splits on `n ~ 293` training rows per fold.
   - Similarly on **`emotion`**, `xgb_sparse_tfidf` achieved strong precision/recall on lexical expressions (`gratitude = 92.68%` F1, `curiosity = 54.21%`), while rare/implicit affect (`stress = 15.38%`) hurt its macro average compared to LogReg (`49.36%`) and contextual MuRIL-DAPT-WWM (`61.58%`).
   - Across all 5 tasks, `xgb_svd_linguistic` had **lowest raw probability calibration error (`mean ECE = 0.0187`)** among classical/shallow baselines.

---

## 4. Architectural Verdict & Governance Decision

- **Retain XGBoost Strictly as an Experimental Benchmark (`not_promoted`):**
  1. Across all 5 tasks combined, Transformers remain **`+14.48 pp` ahead** of `xgb_svd_linguistic` (`63.05%` vs. `48.57%` OOF Macro F1), and both ANN (`50.50%`) and LogReg (`49.20%`) edge out XGBoost on macro average due to sparse-text rare-class behavior.
  2. XGBoost inference (`0.22–0.27 ms`) is slower than TextCNN (`0.03 ms`) and LogReg (`0.17 ms`), and sparse TF-IDF training (`127.68s`) is `29.5×` slower than LogReg (`4.33s`) without beating LogReg overall.
  3. Re-testing after Round-2 human annotation expands labeled training data beyond `n = 367` may change dense/tree tradeoffs on `solution` and `sentiment`, making `XGBoostBaselineClassifier` worth keeping in the experimental suite alongside future larger labeled splits.
