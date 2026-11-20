# Model Upgrades & Immediate Real-Data Ingestion Guide

**Repository:** `D:\AI_Project`  
**Benchmark Date:** 2026-11-19  
**GPU Hardware Verified:** NVIDIA GeForce RTX 4060 Laptop GPU (`torch 2.14.1+cu132`, `transformers 5.19.0`, local offline `google/muril-base-cased` cache)  
**Benchmark Artifact (Isolated):** `ml/artifacts/model_upgrade_benchmark/benchmark_report.json`

> [!IMPORTANT]
> **Scientific Scope Guard:** Benchmark metrics below were measured on the held-out, thread-grouped synthetic split (`group_by="post_id"`, `896` train / `192` validation / `192` test, seed `20261116`, zero cross-split leakage) plus a deterministic Reddit-style typo/Hinglish-noise stress test. Frozen `ml/artifacts/demo_synthetic/` and audited `ml/artifacts/demo_synthetic_v2/` remain untouched.

---

## 1. Model Improvements Implemented

### A. Upgraded Classical Multilingual Models (CPU-Fast, Vercel/Docker-Safe)
Added opt-in architectural upgrades across all five classifiers without breaking any V1 contracts:
1. **Word `(1, 2)` + Character `char_wb (3, 5)` `FeatureUnion` TF-IDF:** Captures Hinglish transliteration drift, mobile typos (`hostl`, `placemnt`, `schedul`), and Devanagari/Indic morphological suffixes.
2. **Class-Balanced One-vs-Rest Weighting (`class_weight="balanced"`):** Fixes the V1 `IssueModelConfig` 7:1 negative-class suppression that crushed `academic_workload` recall to `2.5%`.
3. **Validation-Only Per-Label Threshold Calibration (`calibrate_thresholds()`):** Tunes each multi-label sigmoid threshold $\tau_l \in [0.15, 0.85]$ strictly on the held-out validation split to maximize per-label $F_1$ (never touching the test split).

### B. Upgraded GPU MuRIL Transformer (`ml/models/transformer/muril_classifier.py`)
Extended `google/muril-base-cased` from Response-only to **all five RCI tasks** (`sentiment`, `emotion`, `issue`, `response`, `solution`) with:
- Automatic offline snapshot discovery (`D:\AI_Project_Transformer_Temp\hf_home`).
- CUDA `bfloat16` Automatic Mixed Precision (`AMP`) on RTX 4060 (`batch_size=16`, ~45s per task for 6 epochs).
- Decoupled AdamW weight decay (0 decay on biases & LayerNorm), 10% linear warmup + cosine LR decay, gradient clipping (`1.0`).
- `BCEWithLogitsLoss(pos_weight=...)` + validation-only per-label threshold calibration for multi-label tasks (`issue`, `emotion`).
- Best-validation checkpoint restoration (maximizing validation Macro F1 with lower validation loss tie-breaking).

### C. Held-Out Benchmark Results (`n = 192` Test Comments, `24` Unseen Threads)

#### Issue Identification (8-Label Multi-Label — Primary Improvement Target)

| Model | Clean Macro F1 | Clean Micro F1 | Clean Subset Acc | Clean Hamming Loss | `academic_workload` F1 | Noisy Reddit-Style Macro F1 | Train Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **V1 Baseline (`tfidf_logistic`)** | `0.5854` | `0.5778` | `0.4063` | `0.0742` | `0.0488` | `0.3811` | `0.10s` CPU |
| **V2 Classical (`Word+Char + Balanced + Val Cal`)** | **`0.7638`** (`+17.8%`) | **`0.7740`** | **`0.7135`** (`+30.7%`) | `0.0521` | **`0.7222`** (`+67.3%`) | **`0.5695`** (`+18.8%`) | `0.27s` CPU |
| **MuRIL v2 GPU (Comment Only, `bf16`)** | **`0.7800`** (`+19.5%`) | **`0.7909`** | **`0.7344`** (`+32.8%`) | `0.0540` | `0.6739` | **`0.6535`** (`+27.2%`) | `47.4s` GPU |
| **MuRIL v2 GPU + Parent Post Title Context** | **`0.7981`** (`+21.3%`) | **`0.8146`** | **`0.7396`** (`+33.3%`) | **`0.0495`** | `0.6739` | **`0.6892`** (`+30.8%`) | `46.9s` GPU |

#### All Five Tasks Summary (Clean & Noisy Held-Out Macro F1)

| Task | V1 Clean | V2 Classical Clean | MuRIL GPU Clean | V1 Noisy | V2 Classical Noisy | MuRIL GPU Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Issue** (`8-label`) | `0.5854` | `0.7638` | **`0.7800`** (`0.7981` w/ title) | `0.3811` | `0.5695` | **`0.6535`** (`0.6892` w/ title) |
| **Emotion** (`6-label`) | `0.9857` | `0.9857` | **`1.0000`** | `0.9815` | **`0.9884`** | `0.9748` |
| **Sentiment** (`3-class`) | `1.0000` | `1.0000` | `0.9947`* | `0.9946` | **`1.0000`** | `0.9600`* |
| **Response** (`6-class`) | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Solution** (`binary`) | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

*\*Note: During the benchmark run, Sentiment/Emotion tied at `val_macro_f1 = 1.0000` on epoch 1 and kept epoch 1 before margin convergence; `val_loss` tie-breaking has now been added to `MuRILTaskClassifier.fit()` so tied epochs always keep the lower-loss, sharper-margin checkpoint.*

Re-run the benchmark anytime:
```powershell
D:\AI_Project\backend\.venv\Scripts\python.exe D:\AI_Project\backend\benchmark_model_upgrades.py --epochs 6
```

---

## 2. Live Network Diagnostics: Why Unauthenticated Free Endpoints Failed

Live probes run from this machine on 2026-11-19 confirmed:
- **`www.reddit.com/*.json` & `old.reddit.com/*.json` (unauthenticated):** HTTP `403` block page.
- **`www.reddit.com/r/*/.rss`:** HTTP `404` / `429` throttling.
- **`arctic-shift.photon-reddit.com` REST API:** Immediate `ConnectionResetError(10054)` from Python `requests` on this network.
- **`api.pullpush.io`:** Cloudflare HTTP `521` (origin down).
- **Local `.env` status:** `DECODO_API_TOKEN`, `DATABASE_URL`, `HERMES_API_KEY`, `HERMES_API_URL` are already set.

---

## 3. Three Ways to Ingest Real Reddit Data Immediately (Fastest First)

### Path 1 — Zero-Cost, Zero-API-Key Browser Thread `.json` / Arctic Shift `.jsonl` Importer (Works in 60 Seconds!)
1. Open target threads in your normal logged-in browser (e.g., `r/MBAIndia`, `r/CATpreparation`, `r/IIMSambalpur`), append `.json` to the thread URL, and press `Ctrl+S` into `D:\AI_Project\data\raw_reddit\` — **OR** download `.jsonl` dumps from https://arctic-shift.photon-reddit.com/download-tool or Academic Torrents into `D:\AI_Project\data\raw_reddit\`.
2. Convert, strip PII, validate against `static_export.py`, and generate blank-label human annotation CSVs in one command:
   ```powershell
   D:\AI_Project\backend\.venv\Scripts\python.exe D:\AI_Project\backend\import_reddit_export.py `
     --input data/raw_reddit/ `
     --output data/real_exports/iim_sambalpur_v1.json `
     --dataset-id real-iim-sambalpur-20261119-v1 `
     --community IIMSambalpur `
     --source-provider browser-reddit-json `
     --export-candidates data/annotations/real_candidates/
   ```

### Path 2 — Bounded Decodo Universal Pilot (Uses Your Existing `DECODO_API_TOKEN`)
1. Verify remaining credits in your Decodo dashboard, then inspect the dry-run plan (0 credits used):
   ```powershell
   D:\AI_Project\backend\.venv\Scripts\python.exe D:\AI_Project\backend\run_decodo_pilot.py `
     --url https://www.reddit.com/r/IIMSambalpur/
   ```
2. Run a single bounded request (`max_posts=10`, `max_comments=25`):
   ```powershell
   D:\AI_Project\backend\.venv\Scripts\python.exe D:\AI_Project\backend\run_decodo_pilot.py `
     --url https://www.reddit.com/r/IIMSambalpur/ `
     --max-posts 10 --max-comments 25 `
     --output data/real_exports/decodo_pilot_v1.json `
     --export-candidates data/annotations/real_candidates/ `
     --live --yes
   ```

### Path 3 — Official Reddit OAuth2 API (`RedditOAuthProvider` — Free, 100 QPM, Full Comment Trees)
Implemented in `backend/app/services/ingestion/providers/reddit_oauth.py` and registered in `IngestionManager` as `"reddit_oauth"`.
Once you add `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` to `backend/.env`, `RedditOAuthProvider` authenticates via `oauth.reddit.com` and retrieves posts + nested comment trees directly into `PostData`/`CommentData` without hitting unauthenticated `403` blocks.
