# V6.7 — Controlled Comparison: TF-IDF vs. Dense Multilingual Embeddings vs. Hybrid Fusion

> **Zero-Leakage Protocol:** Evaluated across all **5 RCI tasks** (`issue`, `sentiment`, `emotion`, `response`, `solution`) using **3-seed × 7-fold (`6:1` per-fold) `post_id`-grouped OOF CV** on `trainval` (`n = 367`, `163` threads) and single-pass locked evaluation on frozen `test` (`n = 92`, `37` threads). All TF-IDF vectorizers, `TruncatedSVD` projections, anisotropy mean-centering vectors, scalers, temperatures, and thresholds are fit strictly inside each training fold `tr_idx`.

## 1. Executive Summary — All 14 Arms Ranked by 7-Fold (`6:1`) OOF Mean F1

| Model Family | Arm | Feature Representation | 7-Fold OOF Mean | OOF ECE | Frozen Test Mean (`n = 92`) | Test ECE | Total Latency / Doc |
|---|---|---|---:|---:|---:|---:|---:|
| PyTorch MLP | `mlp_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | **`64.49%`** | `0.0393` | **`59.36%`** | `0.0529` | `18.55 ms` |
| PyTorch MLP | `mlp_hybrid_gated` | Hybrid Gated (SVD-128 TF-IDF + Tri-Emb) | **`63.79%`** | `0.0362` | **`58.45%`** | `0.0360` | `18.90 ms` |
| Logistic Regression | `logreg_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | **`61.87%`** | `0.0461` | **`59.49%`** | `0.0760` | `18.54 ms` |
| Logistic Regression | `logreg_hybrid_tfidf_emb` | Hybrid (TF-IDF + Tri-Emb + Cues) | **`61.45%`** | `0.0404` | **`59.56%`** | `0.0486` | `18.83 ms` |
| Logistic Regression | `logreg_emb_xlmr_dapt10x` | Embedding (XLM-R-DAPT10x-768) | **`59.46%`** | `0.0540` | **`57.60%`** | `0.0823` | `8.30 ms` |
| Logistic Regression | `logreg_emb_e5_small` | Embedding (E5-Small-384) | **`58.56%`** | `0.0516` | **`53.89%`** | `0.0663` | `5.72 ms` |
| Logistic Regression | `logreg_emb_muril_dapt10x` | Embedding (MuRIL-DAPT10x-768) | **`58.31%`** | `0.0500` | **`57.61%`** | `0.0681` | `4.52 ms` |
| Logistic Regression | `logreg_emb_minilm` | Embedding (MiniLM-384) | **`54.23%`** | `0.0579` | **`50.06%`** | `0.0750` | `4.73 ms` |
| PyTorch MLP | `mlp_tfidf` | TF-IDF (Word+Char) | **`52.07%`** | `0.0775` | **`44.18%`** | `0.0947` | `0.22 ms` |
| XGBoost (hist) | `xgb_hybrid_svd_emb` | Hybrid (SVD-64 TF-IDF + E5-384 + Cues) | **`51.34%`** | `0.0235` | **`45.85%`** | `0.0529` | `6.28 ms` |
| Logistic Regression | `logreg_tfidf_word` | TF-IDF (Word 1-2g) | **`51.00%`** | `0.1075` | **`50.69%`** | `0.1251` | `0.04 ms` |
| Logistic Regression | `logreg_tfidf_word_char` | TF-IDF (Word+Char) | **`50.86%`** | `0.0757` | **`50.50%`** | `0.0812` | `0.16 ms` |
| XGBoost (hist) | `xgb_emb_e5` | Embedding (E5-Small-384 + Cues) | **`50.58%`** | `0.0213` | **`46.17%`** | `0.0602` | `5.89 ms` |
| XGBoost (hist) | `xgb_tfidf_sparse` | TF-IDF (Word+Char CSR) | **`48.48%`** | `0.0329` | **`49.24%`** | `0.0485` | `0.29 ms` |
| *End-to-End Fine-Tuned Reference* | *`v6.6_max_oof`* | *Unfrozen 4-Ep LLRD DAPT Transformers* | *`67.31%`* | *—* | *`64.44%`* | *`0.0653`* | *`~14.8 ms`* |

## 2. Per-Task 7-Fold (`6:1`) OOF Macro F1 Matrix (`trainval`, `n = 367`, 3-Seed Mean)

| Arm | Representation | Issue | Sentiment | Emotion | Response | Solution | **5-Task OOF Mean** |
|---|---|---:|---:|---:|---:|---:|---:|
| `logreg_tfidf_word` | TF-IDF (Word 1-2g) | `48.66%` | `53.45%` | `51.32%` | `40.74%` | `60.84%` | **`51.00%`** |
| `logreg_tfidf_word_char` | TF-IDF (Word+Char) | `46.38%` | `51.85%` | `48.61%` | `43.57%` | `63.89%` | **`50.86%`** |
| `logreg_emb_minilm` | Embedding (MiniLM-384) | `51.89%` | `61.19%` | `45.20%` | `45.74%` | `67.13%` | **`54.23%`** |
| `logreg_emb_e5_small` | Embedding (E5-Small-384) | `54.38%` | `59.96%` | `54.02%` | `50.54%` | `73.91%` | **`58.56%`** |
| `logreg_emb_muril_dapt10x` | Embedding (MuRIL-DAPT10x-768) | `53.61%` | `62.90%` | `55.13%` | `48.55%` | `71.35%` | **`58.31%`** |
| `logreg_emb_xlmr_dapt10x` | Embedding (XLM-R-DAPT10x-768) | `52.26%` | `62.17%` | `58.25%` | `52.62%` | `72.00%` | **`59.46%`** |
| `logreg_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | `55.64%` | `64.52%` | `59.24%` | `55.40%` | `74.54%` | **`61.87%`** |
| `logreg_hybrid_tfidf_emb` | Hybrid (TF-IDF + Tri-Emb + Cues) | `52.84%` | `63.51%` | `61.34%` | `56.26%` | `73.32%` | **`61.45%`** |
| `xgb_tfidf_sparse` | TF-IDF (Word+Char CSR) | `45.47%` | `48.61%` | `45.22%` | `41.68%` | `61.41%` | **`48.48%`** |
| `xgb_emb_e5` | Embedding (E5-Small-384 + Cues) | `38.59%` | `57.24%` | `41.92%` | `42.83%` | `72.32%` | **`50.58%`** |
| `xgb_hybrid_svd_emb` | Hybrid (SVD-64 TF-IDF + E5-384 + Cues) | `38.56%` | `58.16%` | `43.32%` | `45.59%` | `71.08%` | **`51.34%`** |
| `mlp_tfidf` | TF-IDF (Word+Char) | `46.91%` | `55.86%` | `45.68%` | `47.88%` | `64.01%` | **`52.07%`** |
| `mlp_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | `54.27%` | `69.25%` | `62.37%` | `59.03%` | `77.51%` | **`64.49%`** |
| `mlp_hybrid_gated` | Hybrid Gated (SVD-128 TF-IDF + Tri-Emb) | `52.96%` | `67.72%` | `60.83%` | `60.36%` | `77.06%` | **`63.79%`** |

## 3. Per-Task Post-Lock Frozen Test Macro F1 Matrix (`n = 92`, `37` Threads)

| Arm | Family | Issue | Sentiment | Emotion | Response | Solution | **5-Task Test Mean** |
|---|---|---:|---:|---:|---:|---:|---:|
| `logreg_tfidf_word` | Logistic Regression | `43.48%` | `61.70%` | `43.55%` | `38.76%` | `65.95%` | **`50.69%`** |
| `logreg_tfidf_word_char` | Logistic Regression | `42.72%` | `60.52%` | `46.95%` | `38.17%` | `64.16%` | **`50.50%`** |
| `logreg_emb_minilm` | Logistic Regression | `46.97%` | `60.05%` | `41.62%` | `43.26%` | `58.39%` | **`50.06%`** |
| `logreg_emb_e5_small` | Logistic Regression | `47.04%` | `54.85%` | `54.38%` | `51.00%` | `62.17%` | **`53.89%`** |
| `logreg_emb_muril_dapt10x` | Logistic Regression | `46.25%` | `58.84%` | `53.83%` | `54.67%` | `74.44%` | **`57.61%`** |
| `logreg_emb_xlmr_dapt10x` | Logistic Regression | `47.85%` | `58.05%` | `56.86%` | `53.98%` | `71.24%` | **`57.60%`** |
| `logreg_emb_tri_1920` | Logistic Regression | `45.81%` | `56.20%` | `62.41%` | `60.65%` | `72.37%` | **`59.49%`** |
| `logreg_hybrid_tfidf_emb` | Logistic Regression | `48.52%` | `55.87%` | `56.25%` | `56.34%` | `80.83%` | **`59.56%`** |
| `xgb_tfidf_sparse` | XGBoost (hist) | `42.43%` | `51.08%` | `49.25%` | `40.27%` | `63.15%` | **`49.24%`** |
| `xgb_emb_e5` | XGBoost (hist) | `27.02%` | `55.53%` | `26.06%` | `52.52%` | `69.74%` | **`46.17%`** |
| `xgb_hybrid_svd_emb` | XGBoost (hist) | `27.79%` | `56.56%` | `28.99%` | `49.97%` | `65.95%` | **`45.85%`** |
| `mlp_tfidf` | PyTorch MLP | `38.75%` | `50.37%` | `42.64%` | `36.06%` | `53.10%` | **`44.18%`** |
| `mlp_emb_tri_1920` | PyTorch MLP | `50.72%` | `59.53%` | `58.98%` | `55.24%` | `72.34%` | **`59.36%`** |
| `mlp_hybrid_gated` | PyTorch MLP | `43.80%` | `62.42%` | `58.31%` | `55.38%` | `72.34%` | **`58.45%`** |

## 4. Head-to-Head Family Deltas (Dense Embedding & Hybrid vs. Pure TF-IDF)

| Comparison | Classifier Family | Δ 7-Fold OOF (5-Task Mean) | Δ Frozen Test (`n = 92` Mean) |
|---|---|---:|---:|
| `logreg_emb_tri_1920` vs. `logreg_tfidf_word_char` | Logistic Regression | **`+11.01 pp`** | **`+8.99 pp`** |
| `logreg_hybrid_tfidf_emb` vs. `logreg_tfidf_word_char` | Logistic Regression | **`+10.59 pp`** | **`+9.06 pp`** |
| `xgb_emb_e5` vs. `xgb_tfidf_sparse` | XGBoost (hist) | **`+2.10 pp`** | **`-3.07 pp`** |
| `xgb_hybrid_svd_emb` vs. `xgb_tfidf_sparse` | XGBoost (hist) | **`+2.86 pp`** | **`-3.39 pp`** |
| `mlp_emb_tri_1920` vs. `mlp_tfidf` | PyTorch MLP | **`+12.42 pp`** | **`+15.18 pp`** |
| `mlp_hybrid_gated` vs. `mlp_tfidf` | PyTorch MLP | **`+11.72 pp`** | **`+14.27 pp`** |

## 5. Linguistic Regime Breakdown: Pure English (`n = 327`) vs. Code-Mixed Hinglish / Indic (`n = 40`)

| Arm | Representation | Pure English OOF Mean | Code-Mixed / Indic OOF Mean | Gain vs. Word TF-IDF on Code-Mix |
|---|---|---:|---:|---:|
| `logreg_tfidf_word` | TF-IDF (Word 1-2g) | `50.60%` | **`46.24%`** | **`+0.00 pp`** |
| `logreg_tfidf_word_char` | TF-IDF (Word+Char) | `50.27%` | **`40.88%`** | **`-5.36 pp`** |
| `logreg_emb_e5_small` | Embedding (E5-Small-384) | `58.85%` | **`49.47%`** | **`+3.23 pp`** |
| `logreg_emb_xlmr_dapt10x` | Embedding (XLM-R-DAPT10x-768) | `59.03%` | **`49.67%`** | **`+3.43 pp`** |
| `logreg_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | `61.95%` | **`52.33%`** | **`+6.09 pp`** |
| `logreg_hybrid_tfidf_emb` | Hybrid (TF-IDF + Tri-Emb + Cues) | `61.32%` | **`50.17%`** | **`+3.93 pp`** |
| `mlp_tfidf` | TF-IDF (Word+Char) | `52.15%` | **`42.19%`** | **`-4.05 pp`** |
| `mlp_emb_tri_1920` | Embedding (Tri-Encoder 1920d) | `64.45%` | **`53.96%`** | **`+7.72 pp`** |
| `mlp_hybrid_gated` | Hybrid Gated (SVD-128 TF-IDF + Tri-Emb) | `64.60%` | **`57.15%`** | **`+10.91 pp`** |

## 6. Learned Lexical vs. Semantic Feature Gate (`mlp_hybrid_gated`)

Inside `HybridGatedTFIDFEmbeddingMLP`, the sigmoid gate $g = \sigma(W_g [h_{\text{tfidf}} \,\|\, h_{\text{emb}}])$ measures the empirical reliance on lexical TF-IDF vs. dense embeddings per task:

| Task | Learned TF-IDF Gate Share | Learned Embedding Gate Share |
|---|---:|---:|
| **issue** | `32.1%` | **`67.9%`** |
| **sentiment** | `40.6%` | **`59.4%`** |
| **emotion** | `32.8%` | **`67.2%`** |
| **response** | `38.5%` | **`61.5%`** |
| **solution** | `42.4%` | **`57.6%`** |
