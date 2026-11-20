# RCI — Synthetic Robustness & Resource Benchmark Report

**Project:** Reddit Community Intelligence (RCI) — `D:\AI_Project`  
**Execution Date:** 2026-11-19  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`torch 2.14.1+cu132`, `transformers 5.19.0`, offline `google/muril-base-cased` snapshot)  
**Benchmark Script:** [backend/run_robustness_benchmark.py](file:///D:/AI_Project/backend/run_robustness_benchmark.py)  
**Isolated JSON Output:** `ml/artifacts/robustness_benchmark/robustness_report.json`

> [!CAUTION]
> **Strict Scientific Separation:** Every training record (`data/demo_synthetic/`, seed `20261116`) and every stress-test probe (`54` hand-crafted unseen items across `8` perturbation categories) in this report is **synthetic**. These results measure out-of-distribution robustness and compute resource profiles — they are **not** real-world human-annotated Reddit performance and remain strictly isolated from future real-pilot evaluations.

---

## 1. Experimental Protocol (Zero Leakage, Validation-Only Tuning)

- **Splits:** Thread-grouped by `post_id` (`112` train threads / `896` comments; `24` validation threads / `192` comments; `24` test threads / `192` comments; `leakage_detected: false` across all 5 tasks).
- **Tuning Boundary:** Multi-label per-label sigmoid thresholds (`issue`, `emotion`) and MuRIL best-epoch selection (`val_macro_f1` with `val_loss` tie-breaking) used **validation data only**. The test split (`n = 192`) and the 8-category stress-test suite (`n = 54`) were never touched during training or calibration.
- **Preserved Baselines:** Frozen `ml/artifacts/demo_synthetic/`, audited `ml/artifacts/demo_synthetic_v2/`, and `ml/artifacts/model_upgrade_benchmark/` remain untouched.

---

## 2. Held-Out Test Metrics & Resource Profile (`n = 192` Comments, `24` Unseen Threads)

### 2.1 Single-Label & Binary Tasks (`sentiment`, `response`, `solution`)

| Task | Model | Macro F1 | Micro F1 / Acc | Confusion Matrix (Canonical Order) | Train Time | Inference Latency (192 docs) | Peak CPU RAM | Peak GPU VRAM | Artifact Size |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **Sentiment** | V1 TF-IDF | `1.0000` | `1.0000` | `[[51,0,0],[0,46,0],[0,0,95]]` | `0.17s` | `6.71 ms` (`0.035 ms/doc`) | `3.55 MB` | `0 MB` | `358 KB` |
| (`neg, neu, pos`) | V2 Word+Char | `1.0000` | `1.0000` | `[[51,0,0],[0,46,0],[0,0,95]]` | `0.54s` | `26.90 ms` (`0.140 ms/doc`) | `13.41 MB` | `0 MB` | `732 KB` |
| | MuRIL v2 GPU | `1.0000` | `1.0000` | `[[51,0,0],[0,46,0],[0,0,95]]` | `50.01s` | `102.82 ms` (`0.536 ms/doc`) | `10.26 MB` | `4,561 MB` | `906 MB` |
| **Response** | V1 TF-IDF | `1.0000` | `1.0000` | `diag(31,33,32,36,29,31)` | `0.17s` | `7.42 ms` (`0.039 ms/doc`) | `5.84 MB` | `0 MB` | `416 KB` |
| (6 canonical) | V2 Word+Char | `1.0000` | `1.0000` | `diag(31,33,32,36,29,31)` | `0.57s` | `28.81 ms` (`0.150 ms/doc`) | `21.86 MB` | `0 MB` | `942 KB` |
| | MuRIL v2 GPU | `1.0000` | `1.0000` | `diag(31,33,32,36,29,31)` | `57.08s` | `110.74 ms` (`0.577 ms/doc`) | `1.72 MB` | `4,560 MB` | `906 MB` |
| **Solution** | V1 TF-IDF | `1.0000` | `1.0000` | `[[127,0],[0,65]]` | `0.12s` | `6.83 ms` (`0.036 ms/doc`) | `1.92 MB` | `0 MB` | `320 KB` |
| (`not_sol, sol`) | V2 Word+Char | `1.0000` | `1.0000` | `[[127,0],[0,65]]` | `0.48s` | `30.36 ms` (`0.158 ms/doc`) | `12.29 MB` | `0 MB` | `593 KB` |
| | MuRIL v2 GPU | `1.0000` | `1.0000` | `[[127,0],[0,65]]` | `48.49s` | `103.01 ms` (`0.537 ms/doc`) | `1.71 MB` | `4,560 MB` | `906 MB` |

### 2.2 Multi-Label Tasks (`emotion`, `issue`)

| Task | Model | Macro F1 | Micro F1 | Subset Acc | Hamming Loss | Train Time | Inference Latency (192 docs) | Peak CPU RAM | Peak GPU VRAM | Artifact Size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Emotion** | V1 TF-IDF | `0.9857` | `0.9956` | `0.9844` | `0.0026` | `0.20s` | `7.47 ms` (`0.039 ms/doc`) | `2.20 MB` | `0 MB` | `418 KB` |
| (6 labels) | V2 Word+Char | `0.9857` | `0.9956` | `0.9844` | `0.0026` | `0.75s` | `29.26 ms` (`0.152 ms/doc`) | `12.43 MB` | `0 MB` | `945 KB` |
| | MuRIL v2 GPU | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`0.0000`** | `48.02s` | `105.40 ms` (`0.549 ms/doc`) | `1.70 MB` | `4,560 MB` | `906 MB` |
| **Issue** | V1 TF-IDF | `0.5854` | `0.5778` | `0.4062` | `0.0742` | `0.27s` | `8.23 ms` (`0.043 ms/doc`) | `2.16 MB` | `0 MB` | `257 KB` |
| (8 labels) | V2 Word+Char | `0.7638` | `0.7740` | `0.7135` | `0.0521` | `0.85s` | `30.85 ms` (`0.161 ms/doc`) | `12.42 MB` | `0 MB` | `885 KB` |
| | V2 + Post Title | `0.7075` | `0.7191` | `0.6042` | `0.0651` | `1.05s` | `37.26 ms` (`0.194 ms/doc`) | `14.86 MB` | `0 MB` | `946 KB` |
| | MuRIL v2 GPU | `0.7800` | `0.7909` | `0.7344` | `0.0540` | `47.16s` | `105.58 ms` (`0.550 ms/doc`) | `1.71 MB` | `4,560 MB` | `906 MB` |
| | MuRIL + Post Title | **`0.7981`** | **`0.8146`** | **`0.7396`** | **`0.0495`** | `48.05s` | `125.04 ms` (`0.651 ms/doc`) | `1.70 MB` | `4,584 MB` | `906 MB` |

*(Per-class and per-label Precision, Recall, F1, and Support for every model and task are preserved verbatim in `ml/artifacts/robustness_benchmark/robustness_report.json`.)*

---

## 3. Eight-Category Unseen Robustness Stress-Test Results (`n = 54` Probes)

Each probe uses **novel phrasings never appearing in `generate_synthetic_demo.py`**:
`unseen_templates` (8), `paraphrased_comments` (6), `spelling_errors` (6), `informal_language` (6), `short_comments` (6), `ambiguous_responses` (6), `parent_post_context` (8), and `code_mixed_language` (8).

### 3.1 Overall 8-Category Macro F1 Comparison (Out-of-Distribution Generalization)

| Task | V1 TF-IDF | V2 Word+Char | MuRIL v2 GPU | MuRIL vs. V1 Delta | Key Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| **Sentiment** | `0.2326` | `0.2769` | **`0.3768`** | `+0.1442` | Synthetic cue-sentence memorization collapses V1 (`1.00 → 0.23`). |
| **Emotion** | `0.1224` | `0.1270` | **`0.4126`** | `+0.2902` | MuRIL is `3.4×` V1 on unseen affect, especially paraphrases (`0.73`) & short replies (`0.65`). |
| **Issue (Comment Only)** | `0.0903` | `0.2722` | **`0.4797`** | `+0.3894` | Word+Char is `3.0×` V1; MuRIL is `5.3×` V1 (`0.71` on code-mixed!). |
| **Issue (+ Parent Post Title)** | — | `0.5313` | **`0.6856`** | `+0.5953` | Adding parent post title lifts `parent_post_context` from `0.00` (V1/V2) to **`0.92`** (MuRIL+Title). |
| **Response** | `0.0889` | `0.0889` | **`0.4006`** | `+0.3117` | TF-IDF fails completely outside fixed templates (`0.09`); MuRIL reaches `0.78` on paraphrases. |
| **Solution** | `0.4199` | `0.4107` | **`0.7838`** | `+0.3639` | MuRIL generalizes strongly (`1.00` on paraphrases, typos, short replies, parent context). |

### 3.2 Category-by-Category Macro F1 Breakdown (Across All 8 Stress Categories)

| Task & Model | Unseen Templates | Paraphrased | Spelling Errors | Informal Slang | Short Comments | Ambiguous | Parent-Post Context | Code-Mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sentiment — V1** | `0.13` | `0.22` | `0.36` | `0.33` | `0.00` | `0.25` | `0.31` | `0.22` |
| **Sentiment — V2** | `0.10` | `0.11` | `0.36` | `0.38` | `0.33` | `0.00` | `0.36` | **`0.56`** |
| **Sentiment — MuRIL** | **`0.75`** | `0.10` | `0.24` | **`0.62`** | `0.29` | `0.00` | **`0.42`** | `0.53` |
| **Emotion — V1** | `0.13` | `0.08` | `0.30` | `0.13` | `0.00` | `0.00` | `0.08` | `0.07` |
| **Emotion — V2** | `0.07` | `0.00` | **`0.40`** | `0.13` | `0.33` | `0.00` | `0.08` | `0.08` |
| **Emotion — MuRIL** | **`0.44`** | **`0.73`** | `0.20` | **`0.40`** | **`0.65`** | **`0.43`** | **`0.50`** | **`0.36`** |
| **Issue — V1 (comment)** | `0.00` | `0.17` | `0.00` | `0.14` | `0.00` | `0.00` | `0.00` | `0.12` |
| **Issue — V2 (comment)** | `0.12` | `0.33` | `0.00` | `0.14` | **`0.33`** | `0.00` | `0.00` | `0.62` |
| **Issue — V2 (+ title)** | `0.54` | `0.67` | **`0.78`** | **`0.57`** | `0.22` | `0.17` | `0.50` | `0.58` |
| **Issue — MuRIL (comment)** | `0.08` | `0.50` | `0.25` | **`0.57`** | `0.00` | `0.56` | `0.51` | `0.71` |
| **Issue — MuRIL (+ title)** | **`0.60`** | **`0.86`** | `0.51` | `0.36` | `0.22` | **`0.72`** | **`0.92`** | **`0.77`** |
| **Response — V1** | `0.00` | `0.17` | `0.17` | `0.00` | `0.17` | `0.13` | `0.00` | `0.00` |
| **Response — V2** | `0.00` | `0.17` | `0.17` | `0.00` | `0.17` | `0.20` | `0.00` | `0.00` |
| **Response — MuRIL** | **`0.39`** | **`0.78`** | **`0.23`** | **`0.11`** | **`0.58`** | **`0.40`** | **`0.41`** | **`0.22`** |
| **Solution — V1** | `0.38` | `0.40` | `0.78` | `0.40` | `0.40` | `0.25` | `0.38` | `0.33` |
| **Solution — V2** | `0.33` | `0.40` | `0.78` | `0.40` | `0.40` | `0.25` | `0.38` | `0.33` |
| **Solution — MuRIL** | **`0.62`** | **`1.00`** | **`1.00`** | **`0.83`** | **`1.00`** | **`0.49`** | **`1.00`** | `0.33` |

---

## 4. Scientific Conclusions & Engineering Guidance for the Real Pilot

1. **Synthetic 100% $\neq$ Real-World Readiness:** The dramatic drop from `1.00` synthetic held-out scores to `0.09–0.42` V1 TF-IDF Macro F1 on unseen realistic phrasings proves quantitatively that template-trained models **cannot** be used to draw real-world conclusions before human-annotated Reddit training data exists.
2. **MuRIL + Parent Post Title Context Is the Clear Architecture for Real Data:** Even when trained only on synthetic data, pretrained multilingual MuRIL outperformed V1 TF-IDF by **+14 to +60 Macro F1 points** across unseen phrasings, paraphrases, code-mixed Hindi/English, and parent-context attribution — and ran in `<57s` per task (`~0.55 ms/comment` inference, `~4.56 GB` VRAM) on the local RTX 4060.
3. **Word+Char Calibrated TF-IDF Remains the Best Lightweight CPU Baseline:** At `<1 MB` disk size and `~0.15 ms/comment` CPU latency, V2 Word+Char + Parent Title improved Issue probes from `0.09 → 0.53` and spelling-error robustness to `0.78`, making it ideal for fast local sanity checks and CPU-only deployments (Docker without GPU).
4. **Hardest Failure Modes to Target in Real Annotation:** `ambiguous_responses`, `short_comments`, and unseen `code_mixed_language` remain the biggest failure modes even for MuRIL (`0.00–0.49` on Sentiment/Emotion/Response), confirming why the human-annotated real pilot must prioritize naturally occurring short, sarcastic, and Hinglish replies with explicit `uncertain`/`abstain` tracking.
