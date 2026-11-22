# Phase 0 — V5 Pre-Optimization Repository, Dataset & Benchmark Audit

**Date:** 2026-11-22  
**Repository:** `https://github.com/Sohham01/AI_Project`  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM), PyTorch `2.14.1+cu132`, `transformers 5.19.0`, `scikit-learn 1.9.1`, `scipy 1.18.1`  
**Scope:** Complete read-only reconciliation of the V5 codebase, datasets, splits, DAPT pretraining protocol, calibration mechanics, and error distributions prior to controlled isolated experimentation under `ml/artifacts/v5_experiments/`.

---

## 1. Reconciliation Summary (Working Tree vs. Prior Audit Claims)

Every figure from the prior audit was checked directly against `data/annotations/real_reddit_labeled_v1/`, `data/live_corpus/corpus.json`, `ml/artifacts/live_corpus/evaluation.json`, and `backend/build_live_corpus_artifacts.py`:

| Metric / Property | Prior Audit Report | Verified Working-Tree State | Match? |
|---|---|---|---|
| Labelled real comments | 459 comments / 200 threads | 459 comments / 200 distinct `post_id`s (`master_labeled_v1.csv`) | Exact |
| Train / Val / Test split | 298 / 69 / 92 | 298 (132 threads) / 69 (31 threads) / 92 (37 threads) | Exact |
| Cross-task split identity | Unverified | **All 5 tasks share the exact same 298/69/92 `document_id` and `post_id` partitions** | Verified |
| MuRIL-DAPT v3 mean test macro F1 | ~59.55% | **59.55%** (`0.59546`; Production Hybrid **59.48%**, `0.59485`) | Exact |
| Corpus size (`data/live_corpus/corpus.json`) | 365 threads / 3,557 comments | 365 threads / 3,557 comments across 8 subreddits | Exact |
| Parent-comment join coverage | Unverified | **458 / 459** match `corpus.json` via `comment_id == reddit_id` (`cmt_03561` unmatched); **254 / 459 (55.3%)** have a real parent comment (`parent_reddit_id`) and **459 / 459 (100%)** have `post_title` | Verified |

> **Critical correction on annotation independence:** All 5 task CSVs annotate the **exact same 459 underlying Reddit comments** (not 5 × 459 independent comments). Multi-task learning therefore provides 5 distinct supervisory signals per comment across 298 training comments (367 train+val comments), not 1,490 independent comments.

---

## 2. Exact V5 Production Architecture & Hyperparameter Inventory

Verified in `backend/build_live_corpus_artifacts.py`, `ml/models/transformer/muril_classifier.py`, and `ml/artifacts/live_corpus/evaluation.json`:

| Task | Paradigm & Vocab | V5 Training Input | Production Model (`evaluation.json`) | Blend ($\alpha_{\text{MuRIL}} / \alpha_{\text{TF-IDF}}$) | Temp $T$ | Val Macro F1 | Test Macro F1 ($n=92$) | Raw → Cal 5-Bin ECE |
|---|---|---|---|---:|---:|---:|---:|---:|
| **Issue** | Multi-label (8 fixed domains) | `"{post_title} \| {text}"` (string concat) | `muril_dapt_v3_plus_title` | `1.00 / 0.00` | `1.00` | `71.89%` | `45.72%` | `0.060 → 0.060` |
| **Sentiment** | 3-class (`negative`, `neutral`, `positive`) | `{text}` only | `muril_dapt_v3_pooled` | `1.00 / 0.00` | `1.70` | `69.91%` | `59.50%` | `0.267 → 0.143` |
| **Emotion** | Multi-label (6 fixed emotions) | `{text}` only | `muril_dapt_v3_pooled` | `1.00 / 0.00` | `1.06` | `62.58%` | `62.31%` | `0.028 → 0.028` |
| **Response** | 6-class canonical fixed order | `{text}` only | `muril_dapt_v3_tfidf_hybrid` | `0.90 / 0.10` | `1.56` | `57.58%` | `53.79%` (MuRIL `55.20%`) | `0.165 → 0.072` |
| **Solution** | Binary (`not_solution`, `solution`) | `{text}` only | `muril_dapt_v3_tfidf_hybrid` | `0.80 / 0.20` | `1.12` | `86.60%` | `76.10%` (MuRIL `75.00%`) | `0.073 → 0.021` |

### Shared V5 MuRIL-DAPT v3 Fine-Tuning Recipe (`build_live_corpus_artifacts.py:518-536`)
- **Backbone:** Local DAPT checkpoint `$RCI_MODEL_CACHE_DIR/muril_dapt_reddit` (`google/muril-base-cased`, 237.6M params).
- **Pooling & Head:** `[CLS] + Attention-Masked Mean + Max` (`3 × 768 = 2304` dim) → `LayerNorm` → 5-branch Multi-Sample Dropout (`p = 0.2`) → single `Linear(2304, num_labels)`.
- **Optimization:** `AdamW` with LLRD (`0.92`), `lr = 4e-5`, `head_lr_multiplier = 4.0`, `weight_decay = 0.01`, `batch_size = 8`, `epochs = 6`, linear warmup `10%` + cosine decay to `5%`, `max_length = 160`, `bfloat16` AMP on RTX 4060.
- **Losses:**
  - Single-label (`sentiment`, `response`, `solution`): `CrossEntropyLoss` with inverse-frequency class weights (`n_total / (n_cls * count)`) and `label_smoothing = 0.05`.
  - Multi-label (`issue`, `emotion`): `BCEWithLogitsLoss` with capped `pos_weight = min(neg / pos, 2.5)` and `label_smoothing = 0.0`.
- **Checkpoint Selection:** Single-label chooses best epoch by `val_macro_f1` (tie-break `val_loss`); multi-label chooses best epoch by lowest `val_loss` (fixed 0.5 threshold).
- **Calibration & Blend:** Scalar temperature $T \in [1.0, 4.0]$ minimizing validation NLL (`T >= 1.0` non-sharpening); hybrid blend $\alpha \in \{0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$ (`min_muril_alpha = 1.0` for `emotion`) and 33-point per-label threshold grid `[0.10, 0.90]` tuned strictly on the single `n = 69` validation split.

---

## 3. Critical Scientific Findings & Root Causes Found in Phase 0

### Finding A (CRITICAL — Section 3C): V5 DAPT Exposed Held-Out Test Threads
Inspecting `backend/build_live_corpus_artifacts.py:1049-1180` and `ml/models/transformer/muril_dapt.py:151-157`:
1. `unlabeled_dapt_texts` collects titles, bodies, and comments from **all 365 corpus threads** *before* supervised train/val/test partitions are computed.
2. `run_muril_mlm_dapt()` partitions `3,159` deduplicated texts randomly by document (`2,780` train / `379` val) — **not grouped by `post_id`**, and **without excluding the 37 supervised held-out test threads**.
3. **Exact leakage measurement:** All **37 held-out test threads** (`571 / 3,557` corpus comments) had unlabeled text inside the V5 DAPT pool.
4. **Protocol fix for all V5 optimization experiments:** First freeze the 37 held-out test `post_id`s; completely exclude all 37 test threads from DAPT training and DAPT validation; group DAPT validation by `post_id` (`328` eligible threads → `~288` DAPT train threads / `~40` DAPT val threads); record exact thread/doc manifests and assert `len(dapt_threads & test_threads) == 0`. Retain V5's transductive DAPT strictly labeled as `V5-Transductive` for comparison only.

### Finding B (CRITICAL — Root Cause of Issue's −26.2 pp Val→Test Drop): Zero Positive Validation Support on Rare Labels
Expanding the pipe-delimited multi-label arrays across the V5 `298 / 69 / 92` split reveals why V5 validation scores on `Issue` (`71.9% → 45.7%`) and `Emotion` (`stress`, `frustration`) were severely misleading:

| Task & Label | Train Positives ($n=298$) | Val Positives ($n=69$) | Test Positives ($n=92$) | V5 Tuned $\tau$ | V5 Test F1 |
|---|---:|---:|---:|---:|---:|
| `issue: academic_workload` | 16 | **0** | 7 | `0.500` (untuned) | **`0.000`** |
| `issue: hostel_facilities` | 27 | **0** | 6 | `0.500` (untuned) | `0.667` |
| `issue: curriculum_teaching` | 30 | **1** | 5 | `0.625` (overfit to 1 ex.) | **`0.000`** |
| `issue: extracurriculars` | 25 | **2** | 2 | `0.225` | `0.400` |
| `issue: campus_infrastructure` | 27 | **3** | 8 | `0.200` | `0.444` |
| `issue: placements` | 69 | 21 | 29 | **`0.175`** (steals 13 FPs) | `0.607` (`P=0.450`) |
| `issue: financial_concerns` | 73 | 17 | 20 | `0.500` | `0.778` |
| `issue: administrative_communication` | 122 | 33 | 34 | `0.500` | `0.762` |
| `emotion: gratitude` | 20 | **2** | 9 | `0.500` | `0.889` |
| `emotion: confidence` | 28 | **4** | 10 | `0.200` | `0.609` |
| `emotion: hope` | 26 | **4** | 10 | `0.450` | `0.778` |
| `emotion: stress` | 21 | **5** | 5 | **`0.125`** (low prec.) | `0.211` (`P=0.143`) |
| `emotion: frustration` | 56 | 15 | 16 | `0.325` | `0.364` |
| `emotion: curiosity` | 78 | 18 | 22 | `0.775` | `0.889` |
| `response: counterargument` | 38 | **6** | 9 | argmax | **`0.000`** (6→`opinion`) |
| `response: agreement` | 31 | 7 | 7 | argmax | `0.800` |
| `response: solution` | 27 | 11 | 10 | argmax | `0.429` (`R=0.300`, 5→`advice`) |
| `response: advice` | 37 | 9 | 18 | argmax | `0.714` |
| `response: personal_experience` | 57 | 9 | 20 | argmax | `0.718` |
| `response: opinion` | 108 | 27 | 28 | argmax | `0.567` |
| `solution: solution` | 50 | 18 | 18 | `0.500` | `0.636` (12 FPs) |
| `solution: not_solution` | 248 | 51 | 74 | `0.500` | `0.886` |

**Key Diagnostic Insights:**
1. Because `ml/data/splitting.py` stratified `Issue` using composite string signatures (`"academic_workload|placements"`), rare labels got **`0` validation instances (`academic_workload`, `hostel_facilities`) or `1–2` validation instances (`curriculum_teaching`, `extracurriculars`)** — making V5's single-split `val_macro_f1` artificially inflated and making 33-point per-label threshold tuning (`0.125`, `0.175`, `0.625`) pure overfitting noise.
2. In `Response`, `counterargument` (`0.000 F1`) and `solution` (`R=0.300`, 5 confused with `advice`) receive **no parent comment or post title context**, despite 55.3% of labeled rows having a real parent comment and 100% having a post title.
3. In `Sentiment`, `neutral ↔ negative` accounts for 20 of 37 errors and raw ECE (`0.267`) is the worst across all 5 tasks.

### Finding C — Token Length & Truncation Audit (`google/muril-base-cased`)
Measured directly on all 459 labelled comments with joined real parent comments from `corpus.json`:

| Input Configuration | Median | p90 | p95 | Max | Truncated `>160` | Truncated `>256` | Truncated `>384` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Comment only | 23 | 90 | 129 | 191 | `0.4%` | `0.0%` | `0.0%` |
| Reply `[SEP]` Post Title | 37 | 110 | 139 | 234 | `1.3%` | `0.0%` | `0.0%` |
| Reply `[SEP]` Parent Comment (or Title) | 40 | 125 | 164 | 866 | `5.0%` | `1.7%` | `0.9%` |
| Reply `[SEP]` Title + Parent Comment | 49 | 139 | 178 | 909 | `5.7%` | `2.2%` | `1.1%` |

- At `max_length = 160` (V5 default), comment-only inputs rarely truncate (`0.4%`), but parent-context pairs truncate `5.0–5.7%` of long discussions. At `max_length = 256` with `truncation="only_second"` (preserving 100% of the reply while bounding long parent comments), reply loss drops to **`0.0%`** and overall truncation drops to `1.7–2.2%`.

### Finding D — Local Environment & Available Unlabeled Pool
- **Installed:** `torch 2.14.1+cu132` (CUDA RTX 4060, 8.59 GB VRAM), `transformers 5.19.0`, `scikit-learn 1.9.1`, `scipy 1.18.1`.
- **Not installed (implement natively in clean Python/PyTorch):** `iterstrat` (we will implement a deterministic **Group-Preserving Iterative Multi-Label Stratifier** in `ml/experiments/v5_optimization/common.py`) and `peft` (we will implement a clean **pure-PyTorch LoRA adapter** on transformer attention projections without external dependencies).
- **Local cached models:** `google/muril-base-cased` and V5 transductive `muril_dapt_reddit` in `$RCI_MODEL_CACHE_DIR`.
- **Eligible Unlabeled Real Pool (strictly disjoint from the 37 frozen V5 test threads):** `2,986` non-test corpus comments across `328` threads (`166` completely unlabeled threads with `930` comments + `2,056` unlabeled comments in partially labeled non-test threads).

---

## 4. Frozen V5 Baseline Lock & Isolation Contract

- **Frozen Test Partition:** Exact 92 `document_id`s across 37 `post_id`s from `Seed = 20261116` (`real_reddit_labeled_v1`), saved to `ml/artifacts/v5_experiments/manifests/v5_frozen_splits.json` and **locked read-only** during Phases 1–7 model development.
- **Canonical Protection:** `ml/artifacts/live_corpus/`, `data/live_corpus/`, `data/annotations/real_reddit_labeled_v1/`, and `$RCI_MODEL_CACHE_DIR/muril_dapt_reddit` will **never be modified or overwritten**.
- **Strict Inductive Cleanliness:** Before running Phase 1–4 comparisons on clean inductive representations, we will also produce a strictly inductive DAPT checkpoint (excluding all 37 held-out test threads!) and track BOTH inductive and transductive-baseline comparisons transparently.
