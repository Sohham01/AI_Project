# Phase 4 — Task-Specific Loss Function Ablation & Hard-Negative Analysis

**Date:** 2026-11-22  
**Backbone & Inputs:** Strictly Inductive `muril_dapt_inductive_mlm_4ep` (`max_length = 256`), using each task's Phase 2 OOF-selected context mode (`issue → title_pair`, `solution → parent_pair`, `sentiment / emotion / response → comment_only`).  
**Evaluation Protocol:** Identical 5-fold group-aware OOF CV on `trainval` (`n = 367`, `163` threads; `0` leakage; `n = 92` frozen test untouched). Multi-label losses are compared using honest **nested out-of-fold bounded `[0.30, 0.70]` thresholds** so calibration shifts never inflate in-sample F1.  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase4_losses/phase4_summary.json`

---

## 1. Multi-Label Tasks: Issue & Emotion (`bce`, `bce_pos`, `cb_bce`, `asl`)

### A. Issue (8 Fixed Domains, `title_pair`)
| Loss Function | Nested OOF Macro F1 | Fixed-0.5 Macro F1 | Raw ECE | Cal ECE ($T$) | `academic_workload` | `curriculum_teaching` | `extracurriculars` | `placements` | `hostel_facilities` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bce_pos` (V5 baseline) | `46.92%` | `44.76%` | `0.0388` | `0.0423` (`1.04`) | `0.000` | `0.187` | `0.063` | `0.702` | **`0.704`** |
| `bce` (unweighted) | `44.88%` | `39.39%` | `0.0202` | **`0.0151`** (`1.14`) | `0.000` | **`0.245`** | `0.000` | `0.651` | `0.696` |
| `cb_bce` (eff.-num $\beta=0.95$) | `45.27%` | `42.73%` | **`0.0151`** | `0.0187` (`1.04`) | `0.000` | `0.175` | `0.114` | `0.667` | `0.694` |
| **`asl` ($\gamma_-=2, m=0.05$)** | **`48.68%`** | **`45.57%`** | `0.1424` | `0.1424` (`1.00`) | **`0.073`** | `0.226` | **`0.171`** | **`0.733`** | `0.680` |

### B. Emotion (6 Fixed Emotions, `comment_only`)
| Loss Function | Nested OOF Macro F1 | Fixed-0.5 Macro F1 | Raw ECE | Cal ECE ($T$) | `stress` | `frustration` | `hope` | `curiosity` | `confidence` | `gratitude` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bce_pos` (V5 baseline) | `55.94%` | `52.73%` | `0.0513` | `0.0513` (`1.00`) | `0.256` | `0.485` | `0.394` | `0.867` | **`0.554`** | `0.800` |
| `bce` (unweighted) | `49.75%` | `43.16%` | **`0.0100`** | **`0.0100`** (`1.00`) | `0.000` | `0.477` | `0.421` | `0.877` | `0.448` | `0.762` |
| `cb_bce` (eff.-num $\beta=0.95$) | `55.22%` | `45.15%` | `0.0220` | `0.0220` (`1.00`) | `0.250` | `0.477` | `0.426` | `0.864` | `0.407` | **`0.889`** |
| **`asl` ($\gamma_-=2, m=0.05$)** | **`57.87%`** | **`55.77%`** | `0.1619` | `0.1619` (`1.00`) | **`0.417`** | **`0.519`** | **`0.441`** | **`0.878`** | `0.479` | `0.739` |

### Key Finding on Asymmetric Loss (ASL) — Direct Answer to Section 8A
1. **ASL delivers the strongest multi-label discrimination and rare-class recovery on BOTH tasks:**
   - **Emotion:** `stress` jumped **`+16.0 pp` (`0.256 → 0.417`)**, `frustration` rose **`+3.4 pp` (`0.485 → 0.519`)**, `hope` rose **`+4.7 pp` (`0.394 → 0.441`)**, and overall nested OOF Macro F1 improved **`+1.93 pp` (`55.94% → 57.87%`)**.
   - **Issue:** ASL was the **only** loss to break `0.000` nested OOF F1 on `academic_workload` (`0.073`), lifted `extracurriculars` by **`+10.9 pp` (`0.063 → 0.171`)**, `curriculum_teaching` by **`+3.9 pp` (`0.187 → 0.226`)**, and `placements` to **`0.733`**, lifting overall nested OOF Macro F1 by **`+1.76 pp` (`46.92% → 48.68%`)**.
2. **CRITICAL CALIBRATION TRADE-OFF EXPOSED:** ASL **severely degrades probability calibration** (`Issue ECE: 0.0423 → 0.1424`, `Emotion ECE: 0.0513 → 0.1619` — ~3.2× worse!). Because ASL's asymmetric probability shift (`clip = 0.05`, $\gamma_- = 2$) down-weights easy negatives, negative logits sit closer to zero (`p ≈ 0.20–0.35` instead of `0.02–0.05`), and multiplicative temperature scaling (`T >= 1.0`) cannot remove an additive logit offset.
   - **Engineering implication:** ASL wins on ranking and rare-class F1 (carried forward for `issue` and `emotion`), **but** any deployment exposing calibrated confidence scores for ASL models MUST apply per-label Platt/affine calibration $\sigma(a z + b)$ (`b < 0`) rather than scalar $T$ alone — or fall back to `cb_bce` / `bce_pos` when strict ECE `< 0.05` is required.

---

## 2. Single-Label Multiclass Tasks: Response & Sentiment

### A. Response (6 Canonical Classes, `comment_only`)
| Loss Function | OOF Macro F1 | Raw ECE | Cal ECE ($T$) | `counterargument` F1 | `counterarg → opinion` Errors | `solution` F1 | `solution → advice` Errors | `advice` F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`ce_weighted` (V5 baseline)** | **`52.92%`** | `0.0622` | `0.0605` (`1.10`) | **`0.2857`** | **`14`** | **`0.5275`** | **`2`** | **`0.6383`** |
| `ce` (unweighted) | `49.07%` | `0.0584` | `0.0326` (`1.14`) | `0.1818` *(−10.4 pp!)* | `26` | `0.4444` | `2` | `0.5977` |
| `logit_adj` ($\tau=0.75$) | `52.74%` | **`0.0314`** | **`0.0283`** (`1.08`) | `0.2667` | **`14`** | `0.5217` | **`2`** | `0.6154` |
| `focal_weighted` ($\gamma=2$) | `51.88%` | `0.0600` | `0.0600` (`1.00`) | `0.2381` | `19` | `0.4789` | `4` | `0.5870` |

- **Takeaway:** Unweighted `ce` collapses rare `counterargument` (`0.286 → 0.182`, nearly doubling `counterargument → opinion` confusions `14 → 26`). **`ce_weighted`** retains the highest OOF Macro F1 (`52.92%`), while **`logit_adj`** matches it within `0.18 pp` (`52.74%`) while **cutting ECE by 53% (`0.0605 → 0.0283`)** and lifting `personal_experience` (`0.686 → 0.721`) and `opinion` (`0.604 → 0.640`).

### B. Sentiment (3 Classes, `comment_only`)
| Loss Function | OOF Macro F1 | Raw ECE | Cal ECE ($T$) | `negative` F1 | `neutral` F1 | `positive` F1 | `neg ↔ neu` Mutual Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`ce_weighted` (V5 baseline)** | **`68.78%`** | `0.0846` | **`0.0363`** (`1.54`) | **`0.7000`** | `0.7227` | **`0.6405`** | `56` |
| `ce` (unweighted) | `67.87%` | `0.1211` | `0.0489` (`1.68`) | `0.6870` | **`0.7447`** | `0.6043` | `56` |
| `logit_adj` ($\tau=0.75$) | `67.55%` | `0.1186` | `0.0421` (`1.62`) | `0.6612` | `0.7317` | `0.6335` | `56` |
| `focal_weighted` ($\gamma=2$) | `67.19%` | **`0.0654`** | `0.0654` (`1.00`) | `0.6769` | `0.7233` | `0.6154` | `56` |

- **Takeaway:** Notice every single loss produces **exactly 56 `negative ↔ neutral` mutual confusions** — confirming that Sentiment's neutral/negative boundary is a **semantic/data-boundary problem**, not a loss-reweighting problem! `ce_weighted` wins on both Macro F1 (`68.78%`) and calibrated ECE (`0.0363`).

---

## 3. Binary Solution Detection & Hard-Negative Error Analysis (`parent_pair`)

| Loss Function | OOF Macro F1 | `solution` F1 | Raw ECE | Cal ECE ($T$) | TP / FP / FN | Multi-Clause ($\ge 3$) FPs |
|---|---:|---:|---:|---:|---|---:|
| **`ce_weighted` (V5 baseline)** | **`76.22%`** | **`0.6131`** | `0.0410` | `0.0410` (`1.00`) | `42 / 27 / 26` | `15 / 27` (`55.6%`) |
| `ce` (unweighted) | `74.43%` | `0.5714` | `0.0384` | **`0.0131`** (`1.30`) | `34 / 17 / 34` | `9 / 17` (`52.9%`) |
| `focal` ($\gamma=2$) | `74.40%` | `0.5690` | `0.0659` | `0.0659` (`1.00`) | `33 / 15 / 35` | `7 / 15` (`46.7%`) |
| `focal_weighted` ($\gamma=2$) | `75.90%` | `0.6087` | `0.0963` | `0.0963` (`1.00`) | `42 / 28 / 26` | `15 / 28` (`53.6%`) |

### Hard-Negative Breakdown of `solution` False Positives (`n = 27`, `ce_weighted + parent_pair`)
Breaking down all 27 OOF false positives by their ground-truth `response` label and discourse structure reveals **four distinct hard-negative failure modes**:
1. **Opinions & Long Multi-Clause Critiques (`9 / 27` FPs, `33%`):** Multi-clause analytical commentary (`mean 3.74 clauses`) that sounds authoritative without prescribing an operational fix (e.g., `doc-cmt_00915`: 12-clause, 121-word `counterargument` about Baby IIM packages, $p = 0.847$).
2. **Procedural Personal Experiences & Follow-Up Questions (`8 / 27` FPs, `30%`):** Comments mentioning websites, emails, or portals while recounting an anecdote or asking for help (e.g., `doc-cmt_01180`: *"I did that too bro they sent the website link and asked me to check in the website"*, $p = 0.947$; `doc-cmt_02188`: *"We have not received any reply from one week, can u please share email address..."*, $p = 0.844$).
3. **General Advice (`5 / 27` FPs, `19%`):** Substantially reduced from V5 once `parent_pair` context was added, though multi-clause advice remains a secondary FP source.
4. **Parent-Context Bleed on Short Gratitude/Agreement (`2 / 27` FPs, `7%`):** E.g., `doc-cmt_02215` (*"Thanks man, appreciate it"*, $p = 0.838$) — the **parent comment** was a concrete solution, and full-sequence Mean+Max pooling over `parent_pair` bled the parent's solution signal into the short thank-you reply!

---

## 4. Phase 4 Per-Task Loss Decisions Carried Forward

Selected strictly via 5-fold OOF CV:
- **`issue`:** `asl` (`+1.76 pp` nested OOF F1; only loss >0 on `academic_workload`; caveat: requires post-hoc Platt calibration for raw probabilities)
- **`emotion`:** `asl` (`+1.93 pp` nested OOF F1; `stress +16.0 pp`, `frustration +3.4 pp`; same calibration caveat)
- **`sentiment`:** `ce_weighted` (highest OOF F1 `68.78%` & lowest cal ECE `0.0363`)
- **`response`:** `ce_weighted` (highest OOF F1 `52.92%` & `counterargument` `0.286`; `logit_adj` noted as strong low-ECE `0.0283` runner-up)
- **`solution`:** `ce_weighted` (highest OOF F1 `76.22%` with `parent_pair`)
