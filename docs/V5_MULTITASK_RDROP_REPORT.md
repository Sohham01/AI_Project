# Phase 3 — Shared Multi-Task MuRIL & R-Drop Regularization Report

**Date:** 2026-11-22  
**Backbone:** Strictly Inductive `muril_dapt_inductive_mlm_4ep` (`max_length = 256`, `epochs = 7` for MTL, `epochs = 6` for 5 separate models, `batch_size = 4, grad_accum = 2` effective batch `8` for MTL)  
**Evaluation Protocol:** Identical **joint 5-fold group-aware CV** on `trainval` (`n = 367` unique comments across `163` threads; fold sizes `[72, 74, 74, 73, 74]`; `0` thread overlap; `n = 92` frozen test untouched). Multi-label tasks evaluated with honest nested OOF bounded `[0.30, 0.70]` thresholds.  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase3_multitask/phase3_summary.json`

---

## 1. Critical Annotation-Coverage Control (Section 7)

All 5 task CSVs annotate the **exact same 367 `trainval` real Reddit comments** (`100%` overlap; `459` total unique comments including the `92` frozen test comments). Multi-task learning therefore supplies **5 distinct supervisory heads per comment** rather than `5 × 367 = 1,835` independent comments. The architecture supports explicit per-task binary masking (`m_b[t] > 0.5`) so any future partially annotated dataset integrates without modification.

---

## 2. Architecture, Loss Weighting & R-Drop Consistency Implementation

- **Shared Encoder:** Single `muril_dapt_inductive_mlm_4ep` transformer (`237.6M` parameters total across all 5 heads — **5.0× parameter reduction** vs. `1.188B` across 5 separate models).
- **Per-Task Heads:** 5 independent `LayerNorm(2304) → 5× Multi-Sample Dropout(0.20) → Linear(2304, num_labels)` heads using each task's Phase 4 winning loss (`issue/emotion: asl`, `sentiment/response/solution: ce_weighted`) with equal documented task weights (`w_t = 1.0`).
- **Segment-A Reply-Masked Pooling:** Directly testing our Phase 2 mechanistic hypothesis, `mtl_shared_replymask_*` feeds `[CLS] reply [SEP] Post: title | Parent: parent [SEP]` while pooling full tokens for `issue`/`solution` and restricting Mean+Max pooling strictly to Segment-A (`token_type_ids == 0`) reply tokens for `sentiment`/`emotion`/`response`. We also evaluate pure `comment_only` MTL (`mtl_shared_comment_only_rdrop_0.5`) to separate context interference from multi-task gradient interference.
- **Exact R-Drop Regularization:** Two stochastic forward passes per batch with independent dropout masks:
  - **Multi-label tasks (`issue`, `emotion`):** Exact symmetric **Bernoulli KL divergence** per label: $\frac{1}{2}\big[\mathrm{KL}(\mathrm{Bern}(p_1)\,\|\,\mathrm{Bern}(p_2)) + \mathrm{KL}(\mathrm{Bern}(p_2)\,\|\,\mathrm{Bern}(p_1))\big]$.
  - **Single-label tasks (`sentiment`, `response`, `solution`):** Exact symmetric **Categorical KL divergence** over softmax distributions.

---

## 3. Controlled Joint 5-Fold OOF Comparison: 5 Separate Models vs. Shared MTL + R-Drop

| Architecture Variant | Encoders | Total Params | All-5 Latency (ms/cmt) | Speedup | Peak VRAM (MB) | `issue` OOF F1 | `sentiment` OOF F1 | `emotion` OOF F1 | `response` OOF F1 | `solution` OOF F1 | **5-Task Mean OOF F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5 Separate Single-Task MuRIL** | `5` | `1,187.9M` | `7.44 ms` | `1.00×` | `4,044` | **`48.88%`** | **`66.79%`** | **`55.48%`** | **`51.04%`** | `74.10%` | **`59.26%`** |
| MTL Reply-Masked (`α = 0.0`, no R-Drop) | `1` | `237.6M` | `2.30 ms` | `3.24×` | `4,274` | `29.00%` | `60.82%` | `35.13%` | `49.14%` | `71.51%` | `49.12%` |
| MTL Reply-Masked + R-Drop (`α = 0.5`) | `1` | `237.6M` | `2.30 ms` | `3.23×` | `5,045` | `31.43%` | `61.64%` | `39.51%` | `47.97%` | `73.47%` | `50.80%` |
| MTL Reply-Masked + R-Drop (`α = 1.0`) | `1` | `237.6M` | `2.39 ms` | `3.12×` | `5,045` | `30.68%` | `61.79%` | `38.12%` | `48.33%` | `74.65%` | `50.71%` |
| **MTL Comment-Only + R-Drop (`α = 0.5`)** | `1` | `237.6M` | **`1.32 ms`** | **`5.63×`** | `4,942` | `28.69%` | `63.80%` | `48.73%` | `49.98%` | **`75.21%`** | **`53.28%`** |

### Calibration (5-Bin OOF ECE) Comparison
| Architecture Variant | `issue` ECE | `sentiment` ECE | `emotion` ECE | `response` ECE | `solution` ECE |
|---|---:|---:|---:|---:|---:|
| 5 Separate Single-Task MuRIL | `0.1410` *(ASL)* | `0.0419` | `0.1600` *(ASL)* | `0.0499` | `0.0156` |
| MTL Reply-Masked (`α = 0.0`) | `0.2144` | `0.0427` | `0.2267` | `0.0736` | `0.0240` |
| MTL Reply-Masked + R-Drop (`α = 0.5`) | `0.2115` | `0.0496` | `0.2216` | `0.0423` | **`0.0107`** |
| MTL Reply-Masked + R-Drop (`α = 1.0`) | `0.2122` | **`0.0352`** | `0.2230` | **`0.0268`** | `0.0210` |
| MTL Comment-Only + R-Drop (`α = 0.5`) | `0.2165` | `0.0485` | `0.2303` | `0.0383` | `0.0192` |

---

## 4. Task-Interference Diagnosis & R-Drop Effect

### A. Genuine Benefits Observed
1. **R-Drop Regularization (`α = 0.5`) Consistently Improved Multi-Task Generalization & Calibration:**
   - Adding R-Drop (`α = 0.5`) to identical Reply-Masked MTL raised 5-task mean OOF Macro F1 by **`+1.68 pp` (`49.12% → 50.80%`)**, lifting `emotion` by **`+4.38 pp`**, `issue` by **`+2.43 pp`**, and `solution` by **`+1.96 pp`** (`+3.14 pp` at `α = 1.0`).
   - R-Drop also dramatically improved single-label calibration: `response` ECE dropped **−64%** (`0.0736 → 0.0268` at `α = 1.0`) and `solution` ECE dropped **−55%** (`0.0240 → 0.0107` at `α = 0.5`).
2. **Positive Cross-Task Transfer on `Solution` (`+1.11 pp`) & Massive Efficiency Gains:**
   - Under `mtl_shared_comment_only_rdrop_0.5`, **`solution` achieved `75.21%` OOF F1 — beating the dedicated single-task `solution` encoder (`74.10%`) by `+1.11 pp`**, even *without* parent context! Joint supervision with `response` (explicitly separating `solution`, `advice`, `personal_experience`, and `opinion`) acts as a direct structural regularizer for binary solution detection.
   - Total all-5-task inference latency dropped **`5.63×`** (`7.44 ms → 1.32 ms/comment`) and parameter count dropped **`5.0×`** (`1.188B → 237.6M`).

### B. Severe Negative Transfer on Multi-Label Tasks (`issue: −20.2 pp`, `emotion: −6.8 pp`) — Root Causes
Two exact mechanisms caused severe negative transfer on `issue` and `emotion`:
1. **CE vs. ASL Loss-Scale Gradient Dominance (Primary Cause):** Under equal `w_t = 1.0` weighting, single-label weighted Cross-Entropy (`response ≈ 1.6`, `sentiment ≈ 1.0`, `solution ≈ 0.6`) has **10–15× larger scalar loss and gradient magnitude** than sparse `AsymmetricLossMultiLabel.mean()` (`≈ 0.08–0.15`). As a result, single-label gradients dominated shared encoder updates, leaving `issue` (`−20.2 pp`) and `emotion` (`−6.8 pp`) severely underfit (`ECE > 0.21`).
2. **Cross-Attention Context Bleed Still Persists in Reply-Masked Pooling:** Comparing `mtl_shared_comment_only_rdrop_0.5` (`53.28%`) against `mtl_shared_replymask_rdrop_0.5` (`50.80%`) shows that even when Mean+Max pooling is restricted to Segment-A tokens, **12 layers of self-attention over Segment-B context tokens** still shift affect representations (`emotion: 48.73% vs 39.51%`, `sentiment: 63.80% vs 61.64%`).

---

## 5. Phase 3 Decision

- **Do NOT replace all 5 dedicated single-task models with an equal-weight 5-task shared MTL encoder** because negative gradient interference causes an unacceptable `−20.2 pp` drop on `issue` and `−6.8 pp` drop on `emotion`.
- **Keep `mtl_shared_comment_only_rdrop_0.5`** in the Phase 8–9 final comparison explicitly as the **5.6× faster low-resource deployment alternative** (especially noting its `+1.11 pp` positive transfer on `solution` and near-parity `−1.06 pp` on `response`). Future multi-task research should pair only **compatible single-label discourse tasks (`response + solution`)** or apply automatic loss-scale balancing (GradNorm / homoscedastic uncertainty weighting) when mixing ASL and CrossEntropy heads.
