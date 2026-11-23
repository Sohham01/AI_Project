# Deliverable 2 — Reddit Dataset Package, Data Dictionary & Provenance Governance

> **Document ID:** `RCI-V6.2-D2` · **Navigation:** [← Deliverable 1 (Video)](./DELIVERABLE_1_VIDEO_PACKAGE.md) · [Submission Index](./README.md) · [Deliverable 3 (Prototype) →](./DELIVERABLE_3_PROTOTYPE_GUIDE.md)  
> **Data Protection & Redistribution Notice:** No restricted, quarantined, or personally identifiable raw Reddit user records are bundled into public submission assets. All usernames are excluded, Reddit identifiers are pseudonymized (`post_XXXX`, `cmt_XXXXX`), AutoMod/GIF noise is filtered transparently, and the four provenance authorization scopes are recorded separately below without inferring permission from ingestion.

---

## 1. Separated Four-Field Provenance Authorizations

| Provenance Field | Recorded Status | Governance Rationale |
|---|---|---|
| **`collection_authorization`** | `public_reddit_archive_read_only_anonymized_snapshot` | Collected via read-only public Arctic Shift archive snapshot of public IIM/MBA subreddits; zero live collection or private-subreddit scraping performed in V6. |
| **`research_use_authorization`** | `academic_non_commercial_research_analysis` | Authorized strictly for academic NLP research, error analysis, and student-community insight discovery. |
| **`model_training_authorization`** | `internal_research_benchmarking_and_dapt_evaluation_only` | Used strictly for internal DAPT pretraining (with all 44 frozen-test/same-title threads strictly excluded) and supervised group-aware CV benchmarking; not licensed for commercial model redistribution. |
| **`public_redistribution_permission`** | `aggregated_metrics_and_anonymized_excerpts_only_no_usernames_or_restricted_records` | Public web studio and submission reports distribute only aggregated statistics, pseudonymized IDs, and brief anonymized research excerpts; zero raw usernames, PII, or restricted third-party dumps are redistributed. |

---

## 2. Corpus Overview & Subreddit Composition

- **Corpus File:** `data/live_corpus/corpus.json` (`dataset_version: 6.1.0`, `synthetic: false`)
- **Total Volume:** **365 discussion threads** and **3,557 anonymized comments** (mean `9.75` comments/thread; `3,459` eligible non-AutoMod comments for topic discovery).
- **Subreddit Distribution (`8` communities):**
  - `r/IIMSambalpurUG`: `211` threads (`57.8%`)
  - `r/MBAIndia`: `46` threads (`12.6%`)
  - `r/IIMCATPreparation`: `41` threads (`11.2%`)
  - `r/CATPrep`: `18` threads (`4.9%`)
  - `r/iim_CAT`: `18` threads (`4.9%`)
  - `r/CATStudyRoom`: `15` threads (`4.1%`)
  - `r/CATpreparation`: `12` threads (`3.3%`)
  - `r/Indian_Academia`: `4` threads (`1.1%`)

---

## 3. Dataset Schema & Data Dictionary

### 3.1 Post Record (`corpus.json -> posts[]`)

| Field | Type | Description |
|---|---|---|
| `reddit_id` | `string` | Pseudonymized thread identifier (`post_0001` .. `post_0365`). |
| `title` | `string` | Anonymized original thread title (original text preserved). |
| `body` | `string` | Anonymized original selftext body (may be empty for title-only posts). |
| `subreddit` | `string` | Public subreddit name (without `r/` prefix). |
| `created_at` | `ISO-8601 string` | UTC timestamp of post creation. |
| `score` | `integer` | Public upvote score at snapshot time. |
| `num_comments` | `integer` | Number of included anonymized comments in thread. |
| `issue_key` | `string` | Primary canonical issue domain key (one of 8 domains). |
| `issue_category` | `string` | Human-readable issue domain title. |
| `resolution_state` | `enum` | One of `reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`. |
| `resolution_evidence_comment_id` | `string \| null` | Comment ID backing explicit outcome evidence, if present. |
| `comments` | `array<Comment>` | Nested list of anonymized comments preserving `parent_reddit_id`. |

### 3.2 Comment Record (`posts[].comments[]`)

| Field | Type | Description |
|---|---|---|
| `reddit_id` | `string` | Pseudonymized comment ID (`cmt_00001` .. `cmt_03557`). |
| `document_id` | `string` | Canonical cross-artifact join key (`comment-cmt_XXXXX`). |
| `parent_reddit_id` | `string \| null` | Parent comment ID (or `null` for direct reply to OP), preserving full thread hierarchy. |
| `body` | `string` | Original unmodified anonymized comment text (never overwritten during normalization). |
| `created_at` | `ISO-8601 string` | UTC timestamp of comment creation. |
| `language` / `script` / `is_code_mixed` | `string / string / bool` | Output of `IndianRedditLID` separating language from Unicode script. |

---

## 4. Collection & Non-Destructive Preprocessing Summary

1. **PII & Noise Filtering:** Removed AutoModerator/bot boilerplate, GIF-only embeds, and `[deleted]` stubs prior to topic modeling (`3,557 → 3,459` topic-eligible comments); stripped usernames and replaced URLs/links cleanly where required while retaining `original_text` on every record.
2. **Language & Script Separation (`indian_reddit_lid.py`):** Distinguishes Latin vs. Devanagari/Bengali/Odia scripts, detects Romanized Indic lexical inventories (`hi-en`, `hi`, `bn-en`, `or-en`) with proper-noun guards (`Bodh Gaya`), segments multi-language spans (`78` comments with cross-sentence language switches), and explicitly abstains (`346` comments) on numeric/emoji/terse utterances.
3. **Non-Destructive Normalization:** Separate normalized views (`normalize_romanized_indic`, `apply_informal_normalization`) track exact edits in `transformations` metadata without mutating `original_text`.

---

## 5. Language & Script Distribution (`n = 3,557` Comments)

| Language Label | Comments | Share | Dominant Script |
|---|---:|---:|---|
| `en` (English) | 2,883 | 81.05% | Latin (`3,496` total Latin across corpus) |
| `hi-en` (Hinglish / Code-Mixed) | 317 | 8.91% | Latin (`316`) + Mixed Devanagari-Latin (`1`: `cmt_01610`) |
| `und` (Abstained / Numeric / Emoji / Terse) | 314 | 8.83% | Latin (`254`) + Unknown (`60`) |
| `hi` (Pure Romanized Hindi) | 42 | 1.18% | Latin |
| `or-en` (Roman-Odia + English) | 1 | 0.03% | Latin (`cmt_01936`) |
| Native `bn` / `or` script | 0 | 0.00% | Documented corpus limitation (tested separately on synthetic diagnostic probes) |

---

## 6. Annotation Guidelines, Canonical Taxonomies & Labeled Benchmarks

### 6.1 Fixed Canonical Taxonomies (Never Altered)
- **Response Classification (6 classes, locked order):** `1. agreement`, `2. personal_experience`, `3. opinion`, `4. advice`, `5. solution`, `6. counterargument`.
- **Issue Classification (8 multi-label domains):** `academic_workload`, `administrative_communication`, `campus_infrastructure`, `curriculum_teaching`, `extracurriculars`, `financial_concerns`, `hostel_facilities`, `placements`.
- **Emotion Classification (6 multi-label classes):** `confidence`, `curiosity`, `frustration`, `gratitude`, `hope`, `stress`.
- **Sentiment (3 classes):** `negative`, `neutral`, `positive`.
- **Solution (binary):** `not_solution`, `solution`.
- **Resolution States (4 distinct states):** `reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`.

### 6.2 Annotated Datasets Inventory (Strictly Separated Real vs. Synthetic)

| Dataset Artifact | Scope | Size | Split / Separation Policy |
|---|---|---|---|
| `data/annotations/real_reddit_labeled_v1/` | Track A canonical 5-task benchmark | `459` real comments (`200` threads) | `367` `trainval` (5-fold `post_id`-grouped CV) + `92` **frozen untouched test** (`37` threads, SHA-256 locked). |
| `data/annotations/real_reddit_labeled_v2/` | Track B expanded real dataset | `707` real comments | `545` trainval + `162` test (`92` frozen + `70` unseen threads); pending full dual-human IAA merge gate. |
| `data/annotations/multilingual_eval_real_v1.csv` | Phase 2 real LID & span eval | `84` real comments | Manually reviewed real comments (`88.1%` acc, `95.8%` selective acc, `100%` script acc). |
| `data/annotations/multilingual_eval_synthetic_probes_v1.csv` | Phase 2 synthetic capability probe | `24` synthetic probes | **Strictly separated diagnostic** covering Devanagari, Bengali, Odia, Banglish, Roman-Odia. |
| `data/annotations/issue_solution_eval_v1.csv` | Issue–solution mapping eval | `64` real candidate pairs (`8` per issue) | Manually graded `0/1/2` (`verification_status: human_reviewed_v1`). |
| `data/annotations/resolution_eval_v1.csv` | Resolution state & contradiction eval | `40` real threads | Stratified across all 4 states + contradictory follow-ups (`0.0%` false resolution claims). |
| `data/annotations/bertopic_human_review_v1.csv` | BERTopic v1 cluster review & labeling | `35` clusters (`105` top-3 centroid rows) | Manually reviewed & refined all 35 labels (`4.51/5.0` mean coherence; `13` substantive, `12` logistics, `9` conversational, `1` URL-media). |

### 6.3 Illustrative Anonymized Cleaned Samples
- **English Advice (`cmt_00016`, `post_0004`, `financial_concerns`):** *"If you are withdrawing 15 days before classes start, they have to do a full refund except for charges to the tune of 1k..."*
- **Hinglish Contradictory Pending (`cmt_03132`, `post_0319`, `administrative_communication`):** *"Ye log phone bhi nhi utha te hai 🫩"* (`hi-en`, Latin, normalized variant `nhi → nahi`, emoji `🫩` preserved).
- **Mixed Devanagari-Latin (`cmt_01610`, `post_0158`):** *"Hn,hn chor unko, congrats ,tum party do bhai, गोलगप्पे khilao ab to ;)"* (`hi-en`, `mixed` script).
- **Romanized Odia-English (`cmt_01936`, `post_0183`):** *"Bhai nit ra try kar \n Ae sabu kana pasuchu"* (`or-en`, Latin, multi-span).
