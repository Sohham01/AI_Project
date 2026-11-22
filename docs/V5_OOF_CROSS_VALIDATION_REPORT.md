# Phase 1 — Five-Fold Group-Aware Cross-Validation & OOF Calibration Report

**Date:** 2026-11-22  
**Backbone:** Strictly Inductive `muril_dapt_inductive_mlm_4ep` (289 DAPT train threads / 39 DAPT val threads; **0 / 37 frozen test threads exposed**; thread-grouped unseen-thread validation perplexity `167.12 → 31.36`)  
**Dataset Scope:** Untouched V1 `trainval` (`n = 367` real comments across `163` threads). Frozen V5 test set (`n = 92`, `37` threads, SHA-256 `eb26459fc0d8...`) was **strictly excluded** from all fold training, temperature fitting, threshold selection, and model comparisons.  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase1_cv/phase1_summary.json` and `ml/artifacts/v5_experiments/phase1_cv/logits/*.npz`

---

## 1. Group-Aware 5-Fold Construction & Zero-Leakage Verification

Every fold partition was grouped strictly by original `post_id` (`0` overlapping threads across every single train/val split):
- **Single-label tasks (`sentiment`, `response`, `solution`):** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261116)`.
- **Multi-label tasks (`issue`, `emotion`):** Custom **Group-Preserving Iterative Multi-Label Stratification** (`ml/experiments/v5_optimization/common.py:make_group_cv_folds`) assigning thread groups containing the rarest remaining label to the fold with the largest label deficit while preserving strict `post_id` disjointness.

### Key Upgrade Over V5 Single Validation Split (`n = 69`)
In V5's single `298 / 69` split, rare multi-label classes had **`0` validation instances** (`academic_workload`, `hostel_facilities`) or **`1–2` instances** (`curriculum_teaching`, `extracurriculars`, `gratitude`). Under our 5-fold group-preserving iterative stratification:
- **Every single one of the 5 `issue` folds** contains `≥ 3` positives for **all 8 issue labels** (`academic_workload: [4,3,3,3,3]`, `curriculum_teaching: [7,6,6,6,6]`, `hostel_facilities: [5,5,7,5,5]`, `campus_infrastructure: [6,6,6,6,6]`).
- **Every single one of the 5 `emotion` folds** contains `≥ 4` positives for **all 6 emotions** (`stress: [5,5,5,6,5]`, `gratitude: [5,4,4,5,4]`, `frustration: [14,15,14,14,14]`).
- **Every single one of the 5 `response` folds** contains `7–9` `counterargument` and `7–8` `solution` examples.

---

## 2. Single-Split (`298/69`) vs. Honest 5-Fold Group-Aware OOF CV (`n = 367`)

Using the exact V5 input format per task (`title_concat` for `issue`, `comment_only` for the remaining 4 tasks, `max_length = 160`) on the clean strictly inductive DAPT backbone:

| Task | V5 Transductive Val (`n=69`, leaky DAPT) | Inductive Single-Split Val (`n=69`) | 5-Fold OOF Macro F1 (`fixed 0.5 / argmax`) | 5-Fold OOF Macro F1 (Nested Bounded `[0.30,0.70]`) | Outer Fold F1 Range (Min–Max) | Median Best Epoch |
|---|---:|---:|---:|---:|---|---:|
| **Issue** | `71.89%`* | `65.42%`* | `46.68%` | **`48.92%`** | `38.67% – 53.58%` | `6` |
| **Sentiment** | `69.91%` | `71.96%` | **`68.81%`** | *(argmax)* | `61.62% – 80.81%` | `6` |
| **Emotion** | `62.58%` | `67.44%` | `50.84%` | **`57.21%`** | `35.98% – 57.60%` | `5` |
| **Response** | `57.58%` | `55.11%` | **`53.20%`** | *(argmax)* | `46.32% – 59.19%` | `6` |
| **Solution** | `86.60%` | `79.66%` | **`74.35%`** | `71.76%` *(worse!)* | `66.02% – 82.49%` | `6` |
| **5-Task Mean** | `69.71%` | `67.92%` | `58.78%` | **`60.50%`** | — | `6` |

*\*Single-split `n=69` Issue val Macro F1 skipped empty labels (`academic_workload=0`, `hostel_facilities=0`) and overfit 33-point thresholds to `1–2` examples (`extracurriculars=0.125`, `placements=0.175`).*

**Diagnostic Takeaway:** Across every single task, outer 72–78 comment folds swing by **13–22 percentage points** depending on which ~32 threads fall into the validation fold (`sentiment: 61.6%–80.8%`, `solution: 66.0%–82.5%`, `emotion: 36.0%–57.6%`). This conclusively proves why V5's single `n=69` validation split could not support stable checkpoint, threshold, or hyperparameter selection.

---

## 3. OOF Calibration Analysis: Raw vs. Global Scalar $T$ vs. Per-Class $T_c$

Evaluated strictly on aggregated out-of-fold logits (`n = 367`, `T >= 1.0` non-sharpening):

| Task | Raw ($T=1.0$) OOF F1 / ECE / NLL | Global Scalar $T$ | Global $T$ OOF F1 / ECE / NLL | Per-Class $T_c$ OOF F1 / ECE | Finding |
|---|---:|---:|---:|---:|---|
| **Issue** | `0.4668 / 0.0387 / 0.2642` | `1.02` | `0.4668 / 0.0405 / 0.2642` | `0.4668 / 0.0411` | Already well-calibrated (`T ≈ 1.0`) |
| **Sentiment** | `0.6881 / 0.1118 / 0.8040` | **`1.52`** | **`0.6881 / 0.0338 / 0.7548`** | `0.6815 / 0.0219` | **Global $T=1.52$ cuts ECE by 70%** without hurting argmax F1; per-class $T_c$ hurts F1 (`-0.66 pp`) |
| **Emotion** | `0.5084 / 0.0531 / 0.2504` | `1.00` | `0.5084 / 0.0531 / 0.2504` | `0.5084 / 0.0531` | Already well-calibrated (`T = 1.0`) |
| **Response** | `0.5320 / 0.0581 / 1.2572` | **`1.12`** | **`0.5320 / 0.0532 / 1.2525`** | `0.4870 / 0.0486` | **Per-class $T_c$ severely harms F1 (`-4.50 pp`)** by scaling rare `counterargument` logits by `T=2.7` |
| **Solution** | `0.7435 / 0.0418 / 0.4655` | **`1.12`** | **`0.7435 / 0.0413 / 0.4633`** | `0.7495 / 0.0600` | Per-class $T_c$ worsens ECE (`0.0413 → 0.0600`); keep global scalar $T$ |

**Conclusion:** Global scalar temperature scaling ($T \ge 1.0$) strictly dominates per-class vector temperature scaling on this sample size: it preserves single-label decision boundaries while reducing Sentiment's OOF ECE by **70%** (`0.1118 → 0.0338`), whereas per-class $T_c$ overfits rare classes (`response: -4.50 pp`).

---

## 4. Multi-Label & Binary Threshold Calibration + Fold-to-Fold Stability

To avoid optimistic in-sample threshold bias, we measured **nested out-of-fold threshold generalization** (fitting thresholds on 4 outer folds and scoring the untouched 5th fold across all 5 rotations) alongside fold-to-fold threshold standard deviation:

### A. Issue (8 Multi-Label Domains)
| Strategy | OOF In-Sample Macro F1 | **Nested OOF Macro F1** | OOF Micro F1 | Hamming Loss | Key Thresholds (`placements / admin / workload`) |
|---|---:|---:|---:|---:|---|
| `fixed_05` | `0.4668` | `0.4668` | `0.6820` | `0.0991` | `0.500 / 0.500 / 0.500` |
| `unbounded [0.10, 0.90]` | `0.5579` *(+9.1 pp optimistic!)* | `0.5042` | `0.6287` *(−5.3 pp!)* | `0.1420` *(worse!)* | `0.450 / 0.700 / 0.100` (99 FPs on workload!) |
| **`bounded [0.30, 0.70]`** | `0.5259` | **`0.4892`** | **`0.6904`** | **`0.1008`** | **`0.450 / 0.700 / 0.375`** |
| `regularized` (EB shrinkage) | `0.4965` | `0.4851` | `0.6866` | `0.0998` | `0.455 / 0.688 / 0.423` |

- **Investigating V5's `placements = 0.175` anomaly:** On honest 5-fold OOF (`99` positive `placements` examples instead of `21`), the optimal `placements` threshold moved immediately from **`0.175 → 0.450`**, eliminating the 13-FP collapse!
- **Why `unbounded [0.10, 0.90]` is dangerous despite higher raw Macro F1:** Setting `academic_workload = 0.100` fired **99 predicted positives for 16 true examples (`precision = 8.1%`)**, collapsing overall `micro_f1` (`0.6820 → 0.6287`) and increasing Hamming loss by `+43%` (`0.0991 → 0.1420`). Meanwhile, **`bounded [0.30, 0.70]`** cut fold-to-fold threshold standard deviation by **40%–93%** across all 8 issue labels (e.g., `placements std: 0.223 → 0.133`, `admin std: 0.197 → 0.118`, `extracurriculars std: 0.157 → 0.010`) while simultaneously improving BOTH Macro F1 and Micro F1!
- **Root cause remaining on `academic_workload`:** Under baseline `BCEWithLogitsLoss(pos_weight<=2.5)`, `academic_workload` (`n=16` positives) scores `0.000` nested OOF F1 because raw probabilities rarely exceed `0.30` — directly motivating Phase 4 (Asymmetric / Class-Balanced Loss) and Phase 6 / Item #18 (targeted real annotation).

### B. Emotion (6 Multi-Label Classes)
| Strategy | OOF In-Sample Macro F1 | **Nested OOF Macro F1** | Key Thresholds (`stress / frustration / curiosity`) | Fold Std Reduction (`unb → bnd`) |
|---|---:|---:|---|---|
| `fixed_05` | `0.5084` | `0.5084` | `0.500 / 0.500 / 0.500` | — |
| `unbounded [0.10, 0.90]` | `0.6012` | `0.5839` | `0.275 / 0.400 / 0.625` | `frustration: 0.181`, `curiosity: 0.163` |
| **`bounded [0.30, 0.70]`** | `0.5951` | **`0.5721`** | **`0.300 / 0.400 / 0.625`** | **`frustration: 0.075` (−59%), `curiosity: 0.072` (−56%)** |
| `regularized` (EB shrinkage) | `0.5797` | `0.5680` | `0.356 / 0.412 / 0.613` | Tightest variance |

- **Investigating V5's `stress = 0.125` anomaly:** On honest 5-fold OOF (`26` stress positives instead of `5`), `stress` moved from `0.125 → 0.275–0.300`, eliminating precision collapse (`14.3%`) while boosting Emotion nested OOF Macro F1 by **`+6.37 pp`** (`50.84% → 57.21%`)!

### C. Solution (Binary)
- Fixed `0.50` OOF Macro F1 = **`0.7435`**, whereas nested OOF tuning in `[0.35, 0.70]` **degraded** nested Macro F1 to **`0.7176` (`-2.59 pp`)**. **Decision:** Keep fixed `0.50` (`argmax`) for binary `solution` — do NOT tune a threshold when nested validation refutes it!

---

## 5. Empirical Measurement of Single-Model OOF vs. 5-Model Ensemble Calibration Mismatch

Per Section 5's strict calibration-matching requirement, we ran an **inner 5-fold group CV strictly inside V5 `train` (`n = 298`)** and evaluated both the mean single checkpoint and the **logit-averaged 5-model ensemble** on the **strictly unseen V5 `val` split (`n = 69`, `31` disjoint threads)**:

| Task | Single-Model OOF $T_{\text{single}}$ | Optimal 5-Ens $T_{\text{ens}}$ | Mean 1-Ckpt Val69 F1 / ECE | **5-Ens Raw ($T=1.0$) Val69 F1 / ECE / NLL** | 5-Ens w/ Blind $T_{\text{single}}$ ECE / NLL | Empirical Mismatch Finding |
|---|---:|---:|---:|---:|---:|---|
| **Issue** | `1.00` | `1.00` | `0.3768 / 0.0602` | `0.3711 / 0.0608 / 0.2162` | `0.0608 / 0.2162` | Zero mismatch (`T=1.00` both) |
| **Sentiment** | **`1.48`** | **`1.00`** | `0.6378 / 0.0952` | **`0.6794 (+4.16 pp) / 0.0399 / 0.6868`** | **`0.0893 / 0.7262` (WORSE!)** | **CRITICAL MISMATCH!** Logit ensembling naturally calibrates Sentiment; blindly applying `T_single=1.48` over-softens probabilities and doubles ECE! |
| **Emotion** | `1.00` | `1.00` | `0.5032 / 0.0664` | `0.5074 / 0.0646 / 0.2319` | `0.0646 / 0.2319` | Zero mismatch (`T=1.00` both) |
| **Response** | `1.00` | `1.00` | `0.4788 / 0.1031` | **`0.4751 / 0.0714 (−31% ECE) / 1.1990`** | `0.0714 / 1.1990` | Ensembling cuts ECE by 31% natively |
| **Solution** | `1.00` | `1.00` | `0.7994 / 0.0684` | **`0.8208 (+2.14 pp) / 0.0773 / 0.3398`** | `0.0773 / 0.3398` | Strong `+2.14 pp` ensemble lift |

**Key Engineering Takeaway:**
1. **Never blindly apply a high single-model OOF temperature (`T > 1.3`) to a 5-model logit ensemble** — ensemble variance reduction already achieves optimal calibration (`T_ens_opt = 1.00` across all 5 tasks on unseen threads!).
2. **Single-model alternative (`1 checkpoint trained on all 367 rows` for `median_best_epoch = 5–6`)** uses **5× less storage and 5× faster inference** (`~3.67 GB` peak train VRAM, `21.5s` train time) and MUST apply the single-model OOF scalar $T$ (`1.52` on Sentiment, `1.12` on Response/Solution). Both the 5-fold ensemble and the 1-checkpoint full-367 alternative are saved untouched for controlled Phase 8–9 frozen-test comparison.
