# Phase 6 — Genuine Human Inter-Annotator Agreement (IAA) & Adjudication Report

**Date:** 2026-11-22  
**Status:** `COMPLETE` (`50 / 50` blind items reviewed by human annotator in local Annotation Studio)  
**Primary Comparison:** `Human Reviewer (Blind Studio)` (`iaa_50_human_annotator_studio.csv`) vs. `V2 Rubric Pass-1` (`data/annotations/real_reddit_labeled_v2/`)  
**Secondary Reference:** `Human Reviewer (Blind Studio)` vs. `Full-Arctic-Shift-Context Pass-2` (`iaa_50_ai_arctic_shift_context_audit.csv`)  
**Coverage:** `50 / 50` items (`49` both labeled; `1` review-status disagreement on `doc-v2-cmt_01823`; review-status agreement `98.0%`)  
**Phase 6 Validation Gate Passed:** **`NO (0 / 5 tasks passed)` — V2 expansion (`real_reddit_labeled_v2/`) is BLOCKED from canonical promotion until rubric calibration & adjudication.**

> [!IMPORTANT]
> **Why This Result Matters Scientifically:** The IAA gate worked exactly as designed. Measuring genuine blind human agreement before promoting `real_reddit_labeled_v2/` prevented silent rubric drift from entering production models and exposed six specific, measurable taxonomy boundary discrepancies that also explain V5's historical error patterns (neutral-vs-negative sentiment and multi-clause advice-vs-solution false positives).

---

## 1. Task-Level Genuine Human IAA Summary (`n = 49` both labeled)

### Primary Gate Evaluation: Human Blind Studio vs. V2 Rubric Pass-1
| Task | Paradigm | $n$ | Raw / Exact % | Cohen's $\kappa$ | Krippendorff's $\alpha$ | Phase 6 Gate | Result |
|---|---|---:|---:|---:|---:|---:|---|
| **sentiment** | Single-label | 49 | 59.18% | `0.2615` | `0.2518` | $\ge 0.70$ | **FAIL** |
| **response** | Single-label | 49 | 59.18% | `0.4592` | `0.4596` | $\ge 0.65$ | **FAIL** |
| **solution** | Single-label | 49 | 71.43% | `0.3043` | `0.2354` | $\ge 0.70$ | **FAIL** |
| **issue** | Multi-label | 49 | 42.86% (Jaccard `0.5221`) | *(per-label below)* | MASI `0.3904` / Micro `0.5776` / Macro `0.5949` | $\ge 0.65$ | **FAIL** |
| **emotion** | Multi-label | 49 | 48.98% (Jaccard `0.5612`) | *(per-label below)* | MASI `0.3926` / Micro `0.5375` / Macro `0.4724` | $\ge 0.65$ | **FAIL** |

### Secondary Reference: Human Blind Studio vs. Full-Arctic-Shift-Context Pass-2
| Task | Raw / Exact % | Cohen's $\kappa$ | Krippendorff's $\alpha$ | Interpretation |
|---|---:|---:|---:|---|
| **sentiment** | 63.27% | `0.3333` | `0.3277` | Slightly higher (`+0.072 κ`), same neutral-vs-negative gap |
| **response** | 53.06% | `0.3721` | `0.3724` | Same `opinion` vs `counterargument` / `solution` shift |
| **solution** | 71.43% | `0.3043` | `0.2354` | Identical one-sided `[[30, 0], [14, 5]]` boundary gap |
| **issue** | 40.82% (Jaccard `0.4711`) | *(per-label)* | MASI `0.3382` / Micro `0.5047` | Same `campus_infrastructure` / `financial_concerns` boundary |
| **emotion** | 51.02% (Jaccard `0.5646`) | *(per-label)* | MASI `0.3721` / Micro `0.4723` | Same factual `confidence` vs blank gap |

*(Because Human agreed similarly with both narrow-context Pass-1 and full-thread Arctic Shift Pass-2, the low $\kappa$/$\alpha$ is driven by **rubric definition differences**, not missing thread context.)*

---

## 2. Per-Label Multi-Label Agreement (Human vs. V2 Pass-1)

| Task | Label | Human Count | Pass-1 Count | Cohen's $\kappa$ | Krippendorff's $\alpha$ | Reliability Tier |
|---|---|---:|---:|---:|---:|---|
| `issue` | `academic_workload` | 1 | 1 | `1.0000` | `1.0000` | Exact match (rare) |
| `issue` | `extracurriculars` | 1 | 1 | `1.0000` | `1.0000` | Exact match (rare) |
| `issue` | `hostel_facilities` | 5 | 6 | `0.8977` | `0.8986` | **Strong** |
| `issue` | `administrative_communication` | 11 | 12 | `0.6025` | `0.6064` | Moderate–Substantial |
| `issue` | `placements` | 13 | 21 | `0.5625` | `0.5542` | Moderate (Pass-1 broader) |
| `issue` | `curriculum_teaching` | 7 | 10 | `0.3636` | `0.3660` | Fair |
| `issue` | `financial_concerns` | 10 | 7 | `0.3636` | `0.3660` | Fair (scope drift) |
| `issue` | `campus_infrastructure` | 4 | 0 | `0.0000` | `-0.0319` | Boundary conflict vs `hostel_facilities` |
| `emotion` | `frustration` | 9 | 10 | `0.8043` | `0.8061` | **Strong** |
| `emotion` | `curiosity` | 13 | 8 | `0.5822` | `0.5801` | Moderate |
| `emotion` | `confidence` | 15 | 6 | `0.3651` | `0.3401` | Fair (Human tags factual assertions) |
| `emotion` | `stress` | 6 | 4 | `0.3348` | `0.3386` | Fair |
| `emotion` | `hope` | 3 | 3 | `0.2899` | `0.2971` | Fair |
| `emotion` | `gratitude` | 0 | 0 | `1.0000` | `1.0000` | Absent in sample |

---

## 3. Distribution Shift Check: Human Blind Studio vs. V1 & V2 Datasets

| Dataset | `negative` | `neutral` | `positive` | `solution` (binary) | `response: opinion` | `response: personal_exp` |
|---|---:|---:|---:|---:|---:|---:|
| `master_labeled_v1.csv` (`n=459`) | `35.3%` | `43.8%` | `20.9%` | `18.7%` | `35.5%` | `18.7%` |
| `master_labeled_v2.csv` (`n=707`) | `35.4%` | `46.3%` | `18.4%` | `16.5%` | `35.1%` | `18.7%` |
| **Human Blind Studio (`n=50`)** | **`12.0%`** | **`68.0%`** | `20.0%` | **`38.0%`** | **`46.0%`** | **`10.0%`** |

---

## 4. Six Systematic Disagreement Patterns Diagnosed from Confusion Matrices

1. **One-sided `solution` definition gap (`[[30, 0], [14, 5]]`):**
   - Human agreed on **5 / 5 (`100%`)** of Pass-1's `solution` labels, but marked **14 additional items** as `solution` (`38.0%` vs `10.2%`).
   - Those 14 items are primarily direct factual AMA answers (`doc-v2-cmt_01739`, `01744`, `01759`, `01807`, `01811`), broad decision/career advice (`00551`, `01621`, `01878`, `00413`, `00690`), and status observations (`02196`, `02619`, `02654`, `02017`).
   - **Rule to lock down:** Decide explicitly whether *"informative answer to a question"* counts as `solution`, or keep `ANNOTATION_GUIDELINES.md`'s narrow *"concrete actionable fix/workaround/resource"* rule.
2. **Neutral vs. Negative polarity threshold (`[[4, 2, 0], [10, 21, 2], [1, 5, 4]]`):**
   - **10 of 49 items** were tagged `neutral` by Human and `negative` by Pass-1 (skeptical pushbacks like *"Alone internship cannot provide u good placement"*, complaints like *"It's all really confusing"*, sarcastic callouts like *"Try harder"*, loan warnings like *"honey trap"*).
   - Human reserved `negative` largely for overt disgust/mockery (`12.0%`), whereas V1/V2 rubric tagged critical/dissatisfied remarks `negative` (`35.3%`).
3. **`counterargument` vs. `opinion` & `agreement` (`response` confusion matrix):**
   - **5 items** tagged `counterargument` by Pass-1 were tagged `opinion` by Human (`00211`, `00212`, `00414`, `00721`, `00917` — implicit corrections/debates without explicit *"I disagree"*).
   - **3 items** tagged `agreement` by Human were tagged `opinion` by Pass-1 (`00413`, `00690`, `03524`).
4. **`campus_infrastructure` vs. `hostel_facilities` (`issue` `κ = 0.00`):**
   - Human tagged mess hygiene (`03524`, `03530`, `03531`) and dorm cooking (`02059`) under `campus_infrastructure`, whereas `ANNOTATION_GUIDELINES.md` routes mess/hostel/dorm strictly to `hostel_facilities` and reserves `campus_infrastructure` for Wi-Fi, classrooms, library, labs, transport, and campus facilities.
5. **`financial_concerns` scope drift (`issue` `κ = 0.36`):**
   - Human tagged finance-sector careers/internships (`01831`, `03501`) and general MBA ROI/brand value (`00690`, `02016`) as `financial_concerns`, whereas the rubric limits `financial_concerns` to tuition fees, loans, scholarships, refunds, and financial aid.
6. **Factual assertion vs. `confidence` (`emotion` `κ = 0.37`):**
   - Human tagged `15` items with `confidence` (including plain factual statements like *"see the mail"*, *"Yes there is for st sc and obc"*, *"A requires 4yrs and B 5yrs"*), whereas the rubric leaves purely factual/procedural statements blank (`""`). Conversely, **`frustration` achieved strong agreement (`κ = 0.8043`)**.

---

## 5. Governance Decision & Next Steps
- **Release Gate Decision:** Keep **Track A (`data/annotations/real_reddit_labeled_v1/`, untouched `459` rows, strictly inductive zero-leakage `65.28%` neural / `64.38%` hybrid on frozen `n=92`)** as the sole clean, promotion-eligible benchmark. Do **not** merge `real_reddit_labeled_v2/` into canonical production until human adjudication resolves the 6 boundary rules above.
- **Disagreements File Ready for Adjudication (`45 / 50` items):** `ml/artifacts/v5_experiments/active_learning/iaa_disagreements_adjudication.csv`
- **Raw Machine-Readable Metrics:** `ml/artifacts/v5_experiments/active_learning/iaa_results.json`
