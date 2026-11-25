# V6.4 — Experimental PyTorch BiGRU-Attention & Keras 3 Multi-Task Benchmark Report

> **Python 3.14 & Keras 3 Backend Architecture:** Because TensorFlow provides no Python 3.14 wheels, RCI runs **Keras 3.15.1 natively on `KERAS_BACKEND="torch"`** alongside custom PyTorch `nn.Module` architectures on CUDA/CPU. All models were evaluated strictly via identical 5-fold `post_id`-grouped CV over `trainval (n = 367, 163 threads)` across `3` seeds; the frozen `n = 92` real test split remains untouched.

## 1. Unified 7-Family 5-Fold OOF Cross-Validation Leaderboard (`n = 367`)

| Model Family | Framework / Features | 5-Task Mean OOF F1 | Mean Fold Std | Mean ECE | 5-Task Train Time | Latency / doc |
|---|---|---:|---:|---:|---:|---:|
| **Transformers (`MuRIL-DAPT-WWM + XLM-R`)** *(Ref. OOF)* | HuggingFace PyTorch subwords | **`63.05%`** | — | — | ~522s (GPU) | ~5.47 ms |
| **ANN (2-Layer MLP)** | PyTorch Dense TF-IDF (3,072 dims) | `50.50%` | `±2.84%` | `0.0631` | `13.74s` | `0.167 ms` |
| **PyTorch BiGRU + Additive Attention** | Sequence Emb + BiGRU + Masked Attn + MaxPool (273,673 params) | `49.85%` | `±4.63%` | `0.0650` | `37.28s` | `0.030 ms` |
| **Logistic Regression** | scikit-learn CSR Word+Char TF-IDF (30k) | `49.20%` | `±5.01%` | `0.0813` | `4.33s` | `0.167 ms` |
| **XGBoost (Dense SVD+Ling)** | TruncatedSVD-128 + 14 Ling. Cues | `48.57%` | `±4.45%` | `0.0187` | `29.88s` | `0.270 ms` |
| **XGBoost (Sparse TF-IDF)** | CSR Word+Char TF-IDF (30k) | `47.92%` | `±5.89%` | `0.0355` | `127.68s` | `0.222 ms` |
| **Keras 3 Multi-Task (`KERAS_BACKEND='torch'`)** | Functional Residual MLP + 5 Heads (933,720 joint params) | `45.12%` | `±6.22%` | `0.0493` | `25.20s` | `0.373 ms` |
| **1D TextCNN (Kim 2014)** | PyTorch Token Embeddings (k=2..5) | `42.46%` | `±4.00%` | `0.0671` | `16.02s` | `0.030 ms` |

## 2. Per-Task 5-Fold OOF Macro F1 (%) Comparison

| Task | Keras 3 MT (Torch) | PyTorch BiGRU-Attn | ANN (MLP) | TextCNN | LogReg | XGB (SVD+Ling) | Transformer OOF |
|---|---:|---:|---:|---:|---:|---:|---:|
| **issue** | `30.97%` | `42.81%` | `48.59%` | `30.22%` | `49.88%` | `39.16%` | **`58.61%`** |
| **sentiment** | `48.78%` | `52.84%` | `53.65%` | `49.27%` | `51.92%` | `51.93%` | **`69.22%`** |
| **emotion** | `34.94%` | `46.53%` | `44.51%` | `34.62%` | `49.36%` | `44.81%` | **`61.58%`** |
| **response** | `39.62%` | `41.74%` | `43.11%` | `38.79%` | `38.46%` | `39.77%` | **`49.66%`** |
| **solution** | `71.31%` | `65.31%` | `62.63%` | `59.38%` | `56.36%` | `67.16%` | **`76.17%`** |

## 3. Architectural Takeaways & Governance Decision
- **PyTorch BiGRU + Additive Attention (`49.85%` mean F1, `ECE = 0.0650`):** Outperforms `1D TextCNN` (`42.46%`) on **all five tasks** (`+12.59 pp` Issue, `+11.91 pp` Emotion, `+5.93 pp` Solution, `+3.57 pp` Sentiment, `+2.95 pp` Response), beating `LogReg` (`49.20%`) and both `XGBoost` configurations (`47.92%` / `48.57%`) on 5-task average while providing inspectable token-level attention weights (`get_attention_weights`).
- **Keras 3 Multi-Task (`KERAS_BACKEND="torch"`, `71.31%` Solution, `ECE = 0.0493`):** One shared Functional residual trunk (`933,720` joint params across all 5 heads) achieves **`71.31%` OOF Macro F1 on `solution`** — the strongest non-transformer `solution` score across all tested architectures (`+4.15 pp` above XGBoost Dense `67.16%`, `+8.68 pp` above ANN `62.63%`, `+14.95 pp` above LogReg `56.36%`). Adding per-task projection towers, rare-label row weights, and validation-only temperature scaling cut mean ECE from `0.1923 → 0.0493`. On sparse 8-label `issue` (`30.97%`), joint optimization on `n = 367` samples exhibits multi-task negative transfer relative to single-task models.
- **Experimental Governance:** Neither architecture replaces deployed V6.1 `MuRIL-DAPT-WWM + XLM-R` (`63.05%` OOF CV / `64.62%–65.01%` strict OOF test); both remain experimental benchmarks ready for re-evaluation once Round-2 human annotations expand labeled data beyond `n = 367`.
