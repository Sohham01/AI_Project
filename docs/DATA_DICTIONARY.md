# Deliverable 2 — Reddit Dataset Data Dictionary & Annotation Sample-Size Justification

> **Dataset Family:** `iim-sambalpur-real-reddit-v6` (`v6.0.0`, 100% real anonymized public Reddit discussions, `0` synthetic rows, `0` PII).  
> **Core Files Covered:**
> 1. `data/live_corpus/corpus.json` — Cleaned, deduplicated, pseudonymized Reddit threads & comments (`365` posts, `3,557` comments)
> 2. `data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv` (+ task CSVs `issue.csv`, `sentiment.csv`, `emotion.csv`, `response.csv`, `solution.csv`) — Manually curated real ground-truth dataset (`459` comments across `200` distinct threads)
> 3. `ml/artifacts/live_corpus/predictions.json` — Precomputed calibrated 5-task neural predictions across all `3,557` comments
> 4. `ml/artifacts/live_corpus/topic_discovery.json` — Unsupervised 10-cluster NMF topic model & per-comment topic assignments
> 5. `ml/artifacts/live_corpus/community_intelligence.json` — TF-IDF cosine Issue–Solution links (`861` asserted edges), ranked solution clusters, and Task 5/6 ablation tables
> 6. `ml/artifacts/live_corpus/evaluation.json` — 4-architecture 5-fold benchmarks, DAPT history, error analysis, calibration curves, and 7/7 integrity verifier output

---

## 1. Anonymization, Privacy & Raw-to-Cleaned Lineage

- **Collection Source:** Keyless public Reddit historical archive (`ArcticShiftProvider`) targeting IIM Sambalpur and Indian MBA/CAT communities across **8 subreddits**: `r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, and `r/Indian_Academia`.
- **Raw → Cleaned Hygiene Steps:**
  1. Removed AutoModerator / RemindMeBot stickies, deleted/removed placeholders (`[deleted]`, `[removed]`), GIF-only embeds, and zero-comment threads.
  2. Stripped all personally identifiable or linkable attributes: author usernames, `author_fullname`, emails (`<EMAIL>`), phone numbers, `u/` mentions (`<MENTION>`), raw `reddit.com/r/...` URLs, and raw Reddit base-36 IDs.
  3. Replaced every thread and comment ID deterministically with opaque surrogate IDs (`post_0001`..`post_0365`, `cmt_00001`..`cmt_03557`) while preserving exact parent-child reply tree topology (`2,046 / 3,557` comments linked to their verified immediate parent reply).
  4. Verified automatically on every CI run via regex PII/URL/username gates (`backend/tests/test_live_corpus_integrity.py`).

---

## 2. Schema 1 — Cleaned Anonymized Corpus (`data/live_corpus/corpus.json`)

### 2.1 Top-Level Keys
| Field | Type | Constraints / Description |
|---|---|---|
| `metadata` | `object` | Corpus provenance: `dataset_version` (`"6.0.0"`), `dataset_id` (`"iim-sambalpur-real-reddit-v6"`), `synthetic` (`false`), `post_count` (`365`), `comment_count` (`3557`), `source_provider` (`"arctic_shift"`), `generated_at` (ISO-8601 UTC), `notice` (provenance disclosure) |
| `posts` | `array<Post>` | Array of `365` pseudonymized discussion threads |

### 2.2 `Post` Object (`posts[]`)
| Field | Type | Nullable | Constraints / Allowed Values | Description |
|---|---|---:|---|---|
| `reddit_id` | `string` | No | `^post_\d{4,}$` (unique) | Opaque pseudonymized thread identifier |
| `subreddit` | `string` | No | 8 active Indian MBA/IIM subreddits | Source community name |
| `title` | `string` | No | PII-redacted UTF-8 | Post headline |
| `body` | `string` | Yes | PII-redacted UTF-8 (may be `""`) | Post selftext body |
| `created_at` | `string` | No | ISO-8601 UTC timestamp | Original thread timestamp |
| `score` | `integer` | No | Reddit net upvote score | Community upvote engagement |
| `issue_key` | `string` | No | 1 of 8 canonical issue slugs | Dominant thread issue domain |
| `issue_category` | `string` | No | Human-readable issue label | Display title for `issue_key` |
| `resolution_state` | `string` | No | `reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence` | Explicit first-person outcome state (strictly separated from advice/proposals) |
| `resolution_evidence_comment_id` | `string` | Yes | `^cmt_\d{5,}$` or `null` | Comment containing explicit follow-up outcome statement |
| `comments` | `array<Comment>` | No | Non-empty nested comment list | Pseudonymized human comments in thread |

### 2.3 `Comment` Object (`posts[].comments[]`)
| Field | Type | Nullable | Constraints / Allowed Values | Description |
|---|---|---:|---|---|
| `document_id` | `string` | No | Unique document key (`cmt_XXXXX`) | Primary join key linking `corpus.json`, `predictions.json`, and `topic_discovery.json` |
| `reddit_id` | `string` | No | `^cmt_\d{5,}$` (unique) | Opaque pseudonymized comment ID |
| `parent_reddit_id` | `string` | Yes | `^cmt_\d{5,}$` or `null` | Immediate parent comment ID (`null` for top-level replies to the post) |
| `body` | `string` | No | PII-redacted UTF-8 | Original sanitized comment text (never destructively overwritten) |
| `created_at` | `string` | No | ISO-8601 UTC timestamp | Comment timestamp |
| `score` | `integer` | No | Integer | Net upvote score |
| `language` | `string` | No | `en`, `hi`, `hi-en`, `bn`, `bn-en`, `or`, `or-en`, `ta`, `te`, `kn`, `ml`, `gu`, `pa`, `und` | Indian-Reddit language/code-mix tag |
| `script` | `string` | No | `latin`, `devanagari`, `bengali`, `odia`, `mixed`, `unknown` | Unicode writing script |
| `is_code_mixed` | `boolean` | No | `true` / `false` | Whether comment switches between English and an Indic language |
| `response_type` | `string` | No | 6 canonical roles (fixed order) | Predicted canonical response class |
| `is_solution` | `boolean` | No | `true` / `false` | Predicted binary actionable-solution flag |
| `resolution_evidence_type` | `string` | Yes | `reported_success`, `reported_pending`, or `null` | Explicit first-person resolution signal if present |
| `synthetic` | `boolean` | No | Always `false` | Enforces zero-synthetic provenance |

---

## 3. Schema 2 — Manually Curated Labeled Dataset (`data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv`)

Every row represents **one unique real anonymized Reddit comment** annotated jointly across all 5 NLP tasks (so `459` unique comments = `2,295` task-level annotations, NOT 2,295 independent comments).

| Column | Type | Paradigm | Allowed Values / Format |
|---|---|---|---|
| `document_id` | `string` | ID | Unique labeled record ID (`doc-real-cmt_XXXXX`) |
| `post_id` | `string` | Group Key | Original thread ID (`200` unique threads; strictly enforces zero thread leakage across splits/folds) |
| `comment_id` | `string` | ID | Pseudonymized comment ID |
| `subreddit` | `string` | Metadata | Subreddit name |
| `post_title` | `string` | Context | Thread title (never includes child/descendant replies) |
| `text` | `string` | Input | Sanitized original Reddit comment text |
| `issues` | `pipe-delimited string` | **Multi-Label (8)** | Subset of `{academic_workload, administrative_communication, campus_infrastructure, curriculum_teaching, extracurriculars, financial_concerns, hostel_facilities, placements}` (empty allowed if off-domain) |
| `sentiment` | `enum` | **Single-Label (3)** | `negative`, `neutral`, `positive` |
| `emotions` | `pipe-delimited string` | **Multi-Label (6)** | Subset of `{confidence, curiosity, frustration, gratitude, hope, stress}` (empty allowed for neutral facts) |
| `response` | `enum` | **Single-Label (6, Fixed Order)** | `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument` |
| `solution` | `enum` | **Binary (2)** | `not_solution`, `solution` |

*(Task-specific CSV exports `issue.csv`, `sentiment.csv`, `emotion.csv`, `response.csv`, `solution.csv` mirror `master_labeled_v1.csv` for modular training loaders. Archived Track B expansion `data/annotations/real_reddit_labeled_v2/` (`707` rows) is preserved strictly outside production because its 50-item blind Human IAA audit failed `0/5` threshold gates.)*

---

## 4. Schema 3 — Precomputed Prediction & Analytics Artifacts (`ml/artifacts/live_corpus/`)

| File | Key Structure | Purpose |
|---|---|---|
| `predictions.json` | `tasks.{sentiment, emotion, issue, response, solution}[document_id]` | Per-comment predicted label(s), calibrated probabilities, `confidence`, `confidence_tier` (`high >= 0.80`, `medium 0.55–0.79`, `low < 0.55`), `confidence_margin`, `abstained` flag |
| `topic_discovery.json` | `result.topics[]` (`10` NMF clusters: `topic_id`, `label`, `top_words`, `weights`) + `result.documents[]` (`comment_id`, `dominant_topic_id`, `dominant_topic_score`) | Unsupervised Task 7 topic discovery fit directly on `3,557` real comments (`8,000`-term TF-IDF) |
| `community_intelligence.json` | `linker_summary`, `by_issue`, `issue_solution_clusters`, `task5_task6_ablations` | Precomputed Task 10 `TfidfCosineIssueSolutionLinker` edges (`861` asserted links across `158` threads), ranked community solution themes per issue, and Task 5/6 ablation tables |
| `evaluation.json` | `tasks`, `architecture_comparison`, `dapt_summary`, `error_analysis`, `benchmark_verification`, `labeled_data_provenance`, `human_iaa_audit` | Full apples-to-apples 4-architecture 5-fold benchmarks, strictly inductive WWM DAPT convergence curve, empirical confusion diagnostics, 5-bin calibration reliability curves, and 7/7 integrity audit |

---

## 5. Justification: How Many Samples Were Manually Annotated (`459`) and Why? (PDF Task 4, p. 7)

1. **Multi-Task Annotation Density (`459` unique comments = `2,295` individual task decisions across `200` distinct threads):** Because every comment is jointly annotated across all 5 schemas (8 multi-label issues, 3 sentiments, 6 multi-label emotions, 6 canonical response roles, and binary actionable solution) with thread-title verification against `docs/ANNOTATION_GUIDELINES.md`, each comment requires ~60–90 seconds of careful reading — totaling ~8–11 hours of focused manual curation for `459` high-fidelity real comments without relying on synthetic LLM label generation.
2. **Thread-Grouped Split Coverage (`200` distinct threads → `367` 5-fold CV / `92` frozen unseen-thread test):**
   - Preventing thread leakage requires splitting strictly by `post_id`, not randomly by comment. Sampling `459` comments stratified across `200` distinct threads (`~2.3` comments/thread) ensures the held-out test set spans **`37` completely unseen discussion threads (`92` comments, `20.0%`)**, providing a statistically meaningful unseen-thread benchmark while leaving **`367` comments across `163` threads** for 5-fold group-aware CV (`~294` train / `~73` val per fold).
   - Sampling `12.9%` of the entire `3,557`-comment real corpus (`54.8%` of all `365` threads represented!) gives strong topic diversity across all 8 subreddits and all 8 issue domains.
3. **Why We Stopped Promotion at `459` (V1) Rather Than Uncritically Promoting `707` (V2):**
   - We actively explored expanding the training pool to `707` comments (`+248` uncertainty-sampled rows in Track B V2). However, before promoting V2, we ran a **genuine 50-item blind Human IAA audit** via our local Annotation Studio (`docs/V5_HUMAN_IAA_REPORT.md`).
   - Because `0/5` tasks cleared the $\kappa \ge 0.70 / 0.65$ gate due to 6 rubric boundary ambiguities (documented in `ANNOTATION_GUIDELINES.md` Section 8), we intentionally kept the clean, audited **`459`-comment V1 benchmark** in production rather than trading annotation consistency for raw sample count.
4. **Honest Limitations of `n = 459` (`n = 92` Frozen Test):** Rare classes such as `academic_workload` (`7` test support), `curriculum_teaching` (`5` test support), and `counterargument` (`9` test support) have wide bootstrap 95% confidence intervals (`±10–15 pp`), which is why we report thread-clustered bootstrap CIs (`n = 2,000`), use 5-fold OOF CV for all model selection, and provide a calibrated 30-item blind IAA batch to unlock V2 cleanly once dual-annotator agreement passes.
