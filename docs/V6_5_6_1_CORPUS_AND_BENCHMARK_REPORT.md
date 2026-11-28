# V6.5 — Real Corpus Expansion, 6:1 Train/Test Protocol & 7-Fold (`6:1`) Benchmark Report

> **Protocol Integrity Guard:** `data/annotations/real_reddit_labeled_v1/` (`367` trainval / `92` frozen test across `37` unseen threads), `data/live_corpus/corpus.json` (`v6.1`), `data/corpus_v6_3/new_unannotated_corpus_v6_3.json` (`v6.3`), and `data/corpus_v6_4/new_iim_corpus_v6_4.json` (`v6.4`) remain **100% immutable**. All V6.5 threads are strictly unannotated real discussions (`annotation_status = "unannotated"`), with zero frozen-test leakage and zero fabricated human labels.

## 1. Real Corpus Growth (`v6.1.0` → `v6.3.0` → `v6.4.0` → `v6.5.0`)

| Cohort | Threads | Human Comments | Notes |
|---|---:|---:|---|
| **Baseline (`v6.1.0`)** | `365` | `3,557` | Production corpus + `459` V1 labeled comments |
| **V6.3 Expansion (`v6.3.0`)** | `+451` | `+4,013` | Broad MBA / regional higher-ed discussions |
| **V6.4 IIM Expansion (`v6.4.0`)** | `+526` | `+9,108` | Dedicated IIM + IPMAT + IIMCalcutta communities |
| **V6.5 Campus/Pain-Point Expansion (`v6.5.0`)** | **`+482`** | **`+8,482`** | BLACKI/CAP/New-IIM waitlist, fees, hostel, placements |
| **Combined V6.5 Real Research Corpus** | **`1,824`** | **`25,160`** | `0` leaked/quarantined threads |

## 2. Two-Tier 6:1 Train/Test Protocol

### 2.1 Combined Corpus Thread-Grouped `6:1` Split (`corpus_split_6_1_manifest.json`)

| Partition | Threads | Thread Share | Comments | Comment Share | Pinned Guards |
|---|---:|---:|---:|---:|---|
| **`train_pool`** | **`1,563`** | `85.69%` | `21,543` | `85.62%` | All `163` V1 trainval threads pinned |
| **`test_pool`** | **`261`** | `14.31%` | `3,617` | `14.38%` | All `37` frozen V1 test threads pinned (`0` overlap) |
| **Observed Ratio** | **`5.9885 : 1`** | — | **`5.9560 : 1`** | — | Target: `6.0000 : 1` (`85.71% / 14.29%`) |

### 2.2 Supervised `+185` Blind Human Annotation Target (`552 : 92 = 6.00 : 1`)

- **Why `+185`:** Existing V1 labeled data stands at `367` trainval : `92` frozen test (`3.99 : 1`). Adding **`185` real comments** (`129` diverse `train_pool` threads; `0` frozen-test/test-pool overlap) raises supervised training data to `367 + 185 = 552` against the untouched `92` frozen test comments (**`6.00 : 1`**).
- **Blind Files (`pending_human_annotation`):** [`blind_185_v6_5_annotator_1.csv`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_v6_5_annotator_1.csv), [`blind_185_v6_5_annotator_2.csv`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_v6_5_annotator_2.csv), and [`blind_185_adjudication_template.csv`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_adjudication_template.csv).

## 3. 7-Fold (`6:1` Per-Fold) vs. 5-Fold (`4:1`) OOF Benchmark (`trainval`, `n = 367`)

Switching from 5-fold (`4:1`, `80%` train / `20%` val) to **7-fold `post_id`-grouped CV (`6:1`, `85.7%` train / `14.3%` val)** increases per-fold training exposure without touching the frozen `n = 92` test set (`3` seeds).

| Model Family | 7-Fold (`6:1`) Mean OOF F1 | 5-Fold (`4:1`) Mean OOF F1 | Δ (7F vs 5F) | 7-Fold ECE | 7-Fold Train Time | Latency / doc |
|---|---:|---:|---:|---:|---:|---:|
| **ANN (2-Layer MLP)** | **`52.01%`** | `50.50%` | `+1.51 pp` | `0.0706` | `22.66s` | `0.242 ms` |
| **PyTorch BiGRU + Additive Attention** | **`50.61%`** | `49.85%` | `+0.76 pp` | `0.0693` | `81.40s` | `0.072 ms` |
| **Logistic Regression (Word+Char TF-IDF)** | **`50.46%`** | `49.20%` | `+1.26 pp` | `0.0869` | `8.45s` | `0.225 ms` |
| **XGBoost (Dense SVD+Ling)** | **`49.39%`** | `48.57%` | `+0.82 pp` | `0.0225` | `64.43s` | `0.480 ms` |
| **XGBoost (Sparse TF-IDF)** | **`48.47%`** | `47.92%` | `+0.55 pp` | `0.0286` | `216.57s` | `0.337 ms` |
| **Keras 3 Multi-Task (`KERAS_BACKEND='torch'`)** | **`45.96%`** | `45.12%` | `+0.84 pp` | `0.0400` | `52.91s` | `0.631 ms` |
| **1D TextCNN (Kim 2014)** | **`44.09%`** | `42.46%` | `+1.63 pp` | `0.0645` | `31.96s` | `0.065 ms` |

### Per-Task 7-Fold (`6:1`) OOF Macro F1 (%)

| Task | ANN (MLP) | PyTorch BiGRU-Attn | LogReg | XGB (SVD+Ling) | XGB (Sparse) | Keras 3 MT (Torch) | TextCNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **issue** | `47.52%` | `40.80%` | `49.51%` | `38.34%` | `46.30%` | `29.72%` | `31.72%` |
| **sentiment** | `55.32%` | `54.60%` | `51.84%` | `52.95%` | `46.97%` | `48.40%` | `50.86%` |
| **emotion** | `46.68%` | `46.83%` | `48.17%` | `43.89%` | `45.51%` | `37.52%` | `32.08%` |
| **response** | `46.88%` | `43.75%` | `43.71%` | `42.42%` | `39.98%` | `42.09%` | `41.81%` |
| **solution** | `63.65%` | `67.09%` | `59.08%` | `69.33%` | `63.57%` | `72.06%` | `63.98%` |
