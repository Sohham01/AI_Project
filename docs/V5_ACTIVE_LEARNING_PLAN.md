# Phase 6 — Active Learning Queue, Targeted Annotation Plan & V2 Real-Data Expansion (Item #18)

**Date:** 2026-11-22  
**Eligible Source Pool:** Existing anonymized real comments already inside `data/live_corpus/corpus.json` (**zero** additional Reddit or Arctic Shift scraping).  
**Strict Leakage & Frozen-Test Guardrails:** Excluded ALL 37 frozen V5 held-out test threads + 7 same-title threads (`44` corpus threads total) AND all 459 already-labeled V1 comments using verified `corpus_post_id`, normalized `post_title[:120]`, AND exact normalized comment text (`overlap = 0` across all checks).  
**Artifacts Generated:**
- Blank human-annotation queue (`250` real comments): `ml/artifacts/v5_experiments/active_learning/annotation_queue_250.csv`
- Dual-annotator IAA QA pack (`50` stratified comments): `ml/artifacts/v5_experiments/active_learning/iaa_double_annotation_50.csv`
- Queue summary & assertions: `ml/artifacts/v5_experiments/active_learning/active_learning_summary.json`
- Isolated Track B expanded dataset (`707` real comments, pending human IAA gate): `data/annotations/real_reddit_labeled_v2/` (`master_labeled_v2.csv`, 5 task CSVs, `adjudication_queue.csv`, `v2_provenance.json`)

---

## 1. Eligible Unlabeled Real Pool & Acquisition Strategy

After filtering out all frozen V5 test threads, all 459 V1 comments, deleted/link-only strings, and non-substantive `<6`-word fragments, **`1,393` substantive unlabeled real comments across `245` non-test threads** remained eligible.

### Multi-Signal Acquisition Scoring (Computed Strictly from `trainval_v1` Models)
Every eligible comment was scored by both `muril_dapt_strict_inductive_wwm_8ep` and Word+Char TF-IDF Logistic Regression (both trained **strictly on `trainval_v1`, `n = 367`**) using a composite acquisition score combining:
1. **Prediction uncertainty (`0.30` weight):** Normalized Shannon entropy across `response` and `sentiment`.
2. **Smallest decision margin (`0.20` weight):** $1 - (p_{(1)} - p_{(2)})$ between top-1 and top-2 competing classes.
3. **MuRIL vs. TF-IDF disagreement (`0.20` weight):** Fraction of `{response, sentiment, solution}` where neural and lexical models disagree.
4. **Targeted rare-class probability + lexical candidates (`0.25` + bounded lexical bonus):** Dedicated 15-item quotas for each of the 8 priority weak classes:
   - `academic_workload`, `curriculum_teaching`, `hostel_facilities`, `campus_infrastructure`, `counterargument`, `solution`, `stress`, `frustration`
5. **Context-dependent parent replies (`+0.12` bonus):** Prioritized replies with real immediate parent comments (`63.6%` of the final 250-item queue has a real parent comment!).
6. **Thread diversity cap:** Strict ceiling of `≤ 4` comments per `post_id` so no single megathread dominates the queue.

### Two Strictly Disjoint Queue Partitions (Item #18: More Training AND More Testing Data)
| Partition | Comments | Threads | Sampling Policy | Leakage Checks (`overlap`) |
|---|---:|---:|---|---|
| **Active-Learning Train Queue** (`v2_active_learning_train_candidate`) | `180` | `83` | `8 × 15 = 120` rare-class quotas + `60` highest uncertainty/disagreement (`≤4/thread`) | `0` vs frozen V5 test; `0` vs new test |
| **Representative Unseen-Thread Test Queue** (`v2_expanded_test_candidate`) | `70` | `32` | **Unbiased random sample from completely unlabeled threads** (NO active-learning error bias!) | `0` vs frozen V5 test; `0` vs V1 trainval; `0` vs AL train |
| **IAA Double-Annotation QA Pack** | `50` | Stratified | `35` hard/rare-class AL items + `15` unseen-thread test items | Subset of queue for dual-human QA |

> **Why the 70 new test comments were NOT active-learning hard-mined:** A held-out evaluation benchmark must represent the natural unseen-thread community distribution, not model-error cherry-picks. Sampling those 32 threads uniformly from completely unlabeled threads preserves an unbiased expanded test distribution while raising total real held-out test size from **`92` comments (`37` threads) → `162` comments (`69` threads, `+76%` comments / `+86%` thread clusters)**.

---

## 2. Inter-Annotator Agreement (IAA) & Human Validation Gate Protocol

Before any V2 data is ever promoted into canonical production (`ml/artifacts/live_corpus/`), the following validation gate (aligned with `docs/ANNOTATION_GUIDELINES.md`) MUST pass:
1. **Blank-Label Export:** Both `annotation_queue_250.csv` and `iaa_double_annotation_50.csv` export strictly blank label cells (model scores are never copied into ground-truth label columns).
2. **Independent Dual Annotation (`n = 50`):** Two human annotators independently label all 50 items in `iaa_double_annotation_50.csv` reading `post_title`, `parent_comment_text`, and `text`.
3. **Target Agreement Thresholds:**
   - Single-label tasks (`sentiment`, `response`, `solution`): Cohen's $\kappa \ge 0.70$ (minimum acceptable $\ge 0.65$ on 6-class `response`).
   - Multi-label tasks (`issue`, `emotion`): Krippendorff's $\alpha$ (or per-label Cohen's $\kappa$) $\ge 0.65$.
4. **Adjudication & Exclusion:** All disagreements are adjudicated with written rationale; bot/noise/missing-parent items are routed to `adjudication_queue.csv` with explicit `review_status ∈ {uncertain, abstain, excluded}` and `abstain_reason`.

---

## 3. Isolated Track B Curated V2 Dataset (`data/annotations/real_reddit_labeled_v2/`)

To fulfill **Item #18 (`ADD MORE DATA FOR TRAINING AND TESTING`)** transparently without altering canonical V1 (`real_reddit_labeled_v1/`, untouched at `459` rows) or fabricating synthetic comments, all 250 real comments in `annotation_queue_250.csv` were individually reviewed under `docs/ANNOTATION_GUIDELINES.md`:
- **2 non-substantive items cleanly abstained** to `adjudication_queue.csv` (`#103`: `haikusbot` automated bot output; `#11`: pure meme noise).
- **248 genuine real comments curated** with transparent provenance `annotation_source = "rubric_curated_real_reddit_v2_pending_human_iaa"` (`178` added to `trainval_v2` → `545` total; `70` added to `test_v2_new` → `162` total `test_v2_expanded`, preserving all `92` `test_v5_frozen` rows bit-for-bit with matching SHA-256).

### Class-Support Expansion Achieved (`trainval_v1: 367` → `trainval_v2: 545`)
| Target Rare / Priority Class | `trainval_v1` Support | `trainval_v2` Support | Expansion | Corpus Supply Note |
|---|---:|---:|---:|---|
| `response: counterargument` | `44` | **`85`** | **`+93.2%`** | Nearly doubled! |
| `emotion: stress` | `26` | **`44`** | **`+69.2%`** | Strong gain |
| `emotion: frustration` | `71` | **`114`** | **`+60.6%`** | Strong gain |
| `issue: hostel_facilities` | `27` | **`40`** | **`+48.1%`** | Strong gain |
| `response: solution` | `38` | **`56`** | **`+47.4%`** | Strong gain |
| `issue: campus_infrastructure` | `30` | **`43`** | **`+43.3%`** | Strong gain |
| `solution: solution` (binary) | `68` | **`93`** | **`+36.8%`** | Strong gain |
| `issue: curriculum_teaching` | `31` | **`41`** | **`+32.3%`** | Solid gain |
| `issue: academic_workload` | `16` | `18` | `+12.5%` | **Corpus bottleneck:** admissions/waitlist threads contain very few term-time workload posts; future collection should target term-time student subreddits |

Track B benchmark results (`ml/artifacts/v5_experiments/v2_expanded_data/v2_benchmark_summary.json`) are reported separately from Track A (V1-only) in `docs/V5_MODEL_IMPROVEMENT_REPORT.md` so algorithmic improvements and data-expansion improvements remain 100% disentangled.
