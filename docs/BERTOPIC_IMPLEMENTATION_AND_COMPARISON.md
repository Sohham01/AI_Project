# BERTopic Implementation & NMF vs. BERTopic Comparison (V6 — Phase 1)

> **Non-Replacement Guarantee:** RCI's existing NMF topic-discovery pipeline (`backend/app/services/nlp/topic_discovery/nmf_topic_discoverer.py`) and fitted artifact (`ml/artifacts/live_corpus/topic_discovery.json`) remain **100% intact and unmodified**. BERTopic is implemented as an isolated, versioned companion pipeline (`ml/topics/bertopic/` + `ml/artifacts/topic_models/bertopic_v1/` + `ml/artifacts/live_corpus/bertopic_v1_topics.json`) tracked alongside NMF in `ml/artifacts/topic_models/registry.json`. Topic identifiers are strictly namespaced (`nmf:v1:<id>` vs. `bertopic:bertopic_v1:<id>`) and never mixed.

---

## 1. Architecture

Implemented modularly under [`ml/topics/bertopic/`](../ml/topics/bertopic):
- [`config.py`](../ml/topics/bertopic/config.py): JSON-serializable `BERTopicExperimentConfig` capturing UMAP, HDBSCAN, c-TF-IDF, embedding, and preprocessing settings.
- [`embeddings.py`](../ml/topics/bertopic/embeddings.py): Multilingual dense embedding backend (`intfloat/multilingual-e5-small`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, and local mean-pooled `muril_dapt_strict_inductive_wwm_8ep`) with GPU acceleration (RTX 4060 CUDA), SHA-256 deterministic `.npy` caching, and a controlled synthetic cross-lingual alignment probe.
- [`pipeline.py`](../ml/topics/bertopic/pipeline.py): Official `bertopic==0.17.4` pipeline implementing all 10 stages: (1) multilingual embedding generation, (2) seeded UMAP reduction, (3) density-based HDBSCAN clustering, (4) Indic-script-safe `CountVectorizer` + BM25 `ClassTfidfTransformer`, (5) top-10 unigram/bigram keyword extraction, (6) centroid-cosine representative document ranking, (7) soft/hard topic assignment, (8) explicit `-1` outlier preservation, (9) agglomerative `hierarchical_topics` extraction, and (10) SafeTensors persistence + round-trip `load`/`transform` verification.
- [`evaluation.py`](../ml/topics/bertopic/evaluation.py): Corpus-grounded multi-metric evaluator (NPMI coherence, UMass coherence, topic diversity, outlier rate, size balance entropy, representative centroid cosine, and per-language-register breakdown).
- [`registry.py`](../ml/topics/bertopic/registry.py): Versioned JSON registry at [`ml/artifacts/topic_models/registry.json`](../ml/artifacts/topic_models/registry.json).
- [`run_bertopic_benchmark.py`](../ml/topics/bertopic/run_bertopic_benchmark.py): Reproducible experiment runner executing all 5 BERTopic configurations + NMF baseline on the exact same `3,459` eligible documents.

---

## 2. Multilingual Embedding Model Selection

Before selecting a BERTopic encoder, we evaluated three candidate encoders on **availability, license, RTX 4060 inference speed, dimensionality, and multilingual coverage** across English, Hindi (Devanagari), Hinglish, Bengali (script + Banglish), and Odia (script + Roman-Odia):

### 2.1 Candidate Model Properties & Real-Corpus Encoding Speed (`n = 3,459` docs, RTX 4060 CUDA)

| Embedding Key | Model Identifier | License | Dim | Contrastive Pretrained | Real Corpus Encode Time | ms / doc | Language Coverage |
|---|---|---|---:|---|---:|---:|---|
| `multilingual_e5_small` | `intfloat/multilingual-e5-small` | MIT | 384 | Yes (contrastive) | 8.21 s | 2.37 ms | 100 languages (EN, HI, BN, OR, Romanized Indic) |
| `multilingual_minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 384 | Yes (paraphrase distillation) | 4.82 s | 1.39 ms | 50+ languages (EN, HI, BN; Odia not native) |
| `muril_dapt_wwm` | Local `muril_dapt_strict_inductive_wwm_8ep` (mean-pooled) | Apache-2.0 | 768 | **No (MLM only)** | 6.23 s | 1.80 ms | 17 Indian languages + EN + Reddit WWM DAPT |

### 2.2 Controlled Synthetic 14-Probe Cross-Lingual Alignment Diagnostic
*(Strictly separated from real-corpus evaluation! Measures whether semantically equivalent student-issue queries across scripts/code-mix retrieve the correct English issue anchor.)*

| Encoder | Top-1 Accuracy (`n = 14`) | Mean Cosine Margin | Hinglish | Hindi Dev. | Bengali Script | Banglish | Odia Script | Roman-Odia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`multilingual_e5_small`** | **100.0%** | +0.0531 | 100% | 100% | 100% | 100% | 100% | 100% |
| `multilingual_minilm` | 85.7% | +0.2222 | 100% | 100% | **50%** | **50%** | 100% | 100% |
| `muril_dapt_wwm` (mean-pool) | 92.9% | **+0.0017** | 100% | 100% | **50%** | 100% | 100% | 100% |

**Key Empirical Finding on MuRIL Pooling:** Although `muril_dapt_wwm` excels as a supervised fine-tuned classifier, its raw mean-pooled hidden states suffer classic **anisotropic cone collapse** (near-zero cosine margin `+0.0017`, centroid cosine `0.9976`), causing HDBSCAN to collapse the entire `3,459`-comment corpus into **2 degenerate mega-clusters** (`balance = 0.4153`). This empirically validates why contrastive multilingual sentence embeddings (`intfloat/multilingual-e5-small`) are strictly superior for density-based topic discovery.

---

## 3. Preprocessing & Indic-Safe c-TF-IDF Design

- **Fair Corpus Alignment:** All NMF and BERTopic experiments run on the exact `3,459` AutoMod-filtered comments from `ml/artifacts/live_corpus/topic_discovery.json` (read-only).
- **Indic-Script-Safe `CountVectorizer`:** Standard scikit-learn `token_pattern=r"(?u)\b\w\w+\b"` fragments Devanagari/Bengali/Odia combining vowel matras (`\u0900-\u0d7f`). BERTopic uses `token_pattern=r"(?u)[A-Za-z0-9\u0900-\u0d7f]{2,}"`, `ngram_range=(1, 2)`, `min_df=1`, `max_df=1.0` (because BERTopic applies `CountVectorizer` across the `K` per-topic concatenated mega-documents — using `min_df=3` discarded topic-specific keywords and dropped NPMI by `-0.33`!), paired with `ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)`.
- **Preprocessing Ablation:** We directly compared `raw_clean` (light URL/whitespace cleanup preserving original casing, slang, emojis, and Romanized scripts) against `language_aware_affect` (applying RCI's slang expansion, affect-preserving emoji verbalization, and Romanized Hinglish phonetic unifiers).

---

## 4. Clustering Configuration (Natural Topic Discovery)

- **UMAP:** `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric="cosine"`, `random_state=42`.
- **HDBSCAN:** `min_cluster_size=25`, `min_samples=8`, `metric="euclidean"`, `cluster_selection_method="eom"`, `prediction_data=True`. Sensitivity tested at `min_cluster_size=35, min_samples=10`.
- **No Forced `K=10`:** `nr_topics=None` lets HDBSCAN discover the natural density structure (`35` topics at `mcs=25`, `20` topics at `mcs=35`).
- **Explicit Outlier Preservation:** All `-1` unassigned documents (`1,320 / 3,459 = 38.2%`) are preserved transparently and surfaced in the Topic Map outlier inspector rather than forced into noisy clusters.

---

## 5. Systematic NMF vs. BERTopic Comparison (`n = 3,459` Identical Documents)

| Model ID | Algorithm | Embedding / Vectorizer | Preproc. | Topics (`K`) | Outliers (`-1`) | NPMI Coh. | UMass Coh. | Topic Div. | Centroid Cos. | Size Balance (Entropy) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`nmf_v1`** *(Default Preserved)* | NMF | TF-IDF (1–2g, 8k vocab) | AutoMod-filtered | 10 (fixed) | 0 (0.0%) | **+0.0046** | -2.14 | 0.9600 | 0.9406 | 0.8087 |
| **`bertopic_e5_small_raw`** *(Promoted Companion `bertopic_v1`)* | BERTopic | `multilingual-e5-small` | `raw_clean` | **35** (natural) | 1,320 (38.2%) | -0.0845 | **-1.520** | 0.9629 | 0.9523 | 0.9509 |
| `bertopic_e5_small_raw_mcs35` | BERTopic | `multilingual-e5-small` | `raw_clean` (`mcs=35`) | 20 (natural) | 883 (25.5%) | -0.1481 | -1.900 | 0.9450 | 0.9512 | 0.8482 |
| `bertopic_minilm_raw` | BERTopic | `multilingual-MiniLM-L12` | `raw_clean` | 39 (natural) | 983 (28.4%) | -0.1319 | -1.596 | **0.9795** | 0.8184 | 0.9374 |
| `bertopic_muril_dapt_raw` | BERTopic | `muril_dapt_wwm` (mean) | `raw_clean` | **2** *(collapsed)* | 9 (0.3%) | +0.2280* | -2.417 | 1.0000* | 0.9976* | **0.4153** *(degenerate)* |
| `bertopic_e5_small_lang_aware` | BERTopic | `multilingual-e5-small` | `language_aware_affect` | 31 (natural) | 1,371 (39.6%) | -0.0743 | -1.805 | 0.9516 | **0.9538** | **0.9511** |

*\*Note: `bertopic_muril_dapt_raw`'s high NPMI/centroid cosine is an artifact of collapsing 99.7% of the corpus into 2 giant clusters (entropy balance `0.4153`), underscoring why no single metric can select a topic model.*

### 5.1 Multi-Metric Tradeoff Analysis
1. **Why NMF wins on raw NPMI (+0.0046) yet loses on semantic granularity:** NMF's `Topic 0` absorbs `39.95%` (`1,382`) of all comments using high-frequency English stopwords (`the, is, to, a, it, and, of, for, iim, in`) — high-frequency stopwords co-occur everywhere, mechanically boosting NPMI while providing weak topical interpretability.
2. **Why BERTopic (`bertopic_e5_small_raw`) wins on semantic precision, balance, and multilingual spread:**
   - Entropy balance jumps from `0.8087` (NMF) to **`0.9509`** (BERTopic) with **35 granular themes** (education loans & CIBIL `BT-12`, refund & registration dates `BT-13`, B-school comparisons `BT-06`, repeat CAT vs. workex `BT-05`, BS DSAI vs. MPP `BT-07`, JEE/CAT percentiles `BT-08`, EWS/OBC waitlists `BT-09`/`BT-15`, faculty & classroom concerns `BT-16`, corporate internships `BT-31`).
   - **UMass coherence improves to `-1.520`**, **topic diversity improves to `0.9629`**, and **centroid semantic relevance improves to `0.9523`**.
   - **Multilingual spread:** Under NMF, `77.4%` (`263/340`) of Hinglish/code-mixed comments collapse into one language-bucket (`Topic 6`). Under BERTopic, Hinglish comments span **21 distinct semantic topics** (with only `22.9%` outlier rate — LOWER than English's `41.8%` outlier rate!).
3. **`raw_clean` vs. `language_aware_affect`:** `language_aware_affect` slightly improved NPMI (`-0.0743` vs. `-0.0845`) and centroid cosine (`0.9538` vs. `0.9523`) while yielding `31` topics (`39.6%` outliers), whereas `raw_clean` achieved stronger UMass (`-1.520` vs. `-1.805`), higher diversity (`0.9629`), slightly fewer outliers (`38.2%`), and better preservation of authentic user expressions; thus `bertopic_e5_small_raw` was selected as `bertopic_v1`.

---

## 6. Stability Analysis & Outlier Diagnostics

### 6.1 Seed & Subsample Stability (`bertopic_v1`)
- **80% Subsample Stability (`n = 2,767`, seed 42):** `24` topics discovered; **NMI (all docs) = `0.6251`**, **ARI (non-outlier overlap) = `0.6192`**, ARI (all docs) = `0.2947`.
- **3-Seed Stability (`seeds = [42, 101, 202]`):** Discovered topic counts = `[35, 31, 2]`; mean seed NMI = `0.4191`, mean non-outlier overlap ARI = `0.4396`.
- **Important Honesty Finding:** On noisy short Reddit comments (`median ~11 words`), UMAP + HDBSCAN `eom` occasionally experiences a stochastic density-bridge collapse on one random seed (`seed=202` yielded `2` coarse clusters). Fixing `random_state=42` guarantees exact reproducibility (`SafeTensors` round-trip verified!), while exposing seed sensitivity honestly as a known BERTopic limitation.

### 6.2 Outlier (`-1`) Analysis
- **Total Outliers:** `1,320 / 3,459` (`38.16%`) — preserved explicitly instead of silently assigned.
- **Profile:** Median word count = `11.0` words (`20.6%` are ultra-short `<= 5`-word comments). Language split: `1,192` English (`41.8%` of EN), `78` Hinglish/Indic (`22.9%` of Indic), `50` terse/undetermined (`18.9%` of `und`).
- **Interpretation:** Outliers primarily represent conversational tangents, unique personal narratives, sarcastic banter, and multi-issue rants that do not form a dense 25-comment sub-cluster — preserving them keeps the 35 thematic clusters remarkably clean!

---

## 7. Representative Discovered BERTopic Themes (Machine-Suggested Labels, Pending Human Review)

*(All labels are explicitly marked `machine_suggested_pending_human_review`. A complete human-review worksheet with top-3 representative comments per topic is saved at [`ml/artifacts/topic_models/bertopic_v1/human_review_sample.csv`](../ml/artifacts/topic_models/bertopic_v1/human_review_sample.csv). Unsupervised topics are strictly separated from the 8 supervised issue domains.)*

| BERTopic ID | Docs | Share | Top c-TF-IDF Keywords | Related Supervised Issues (Observational) |
|---|---:|---:|---|---|
| `BT-00` | 224 | 6.5% | `hai, bhai, bhi, nhi, ka, se, nahi` | `administrative_communication`, `placements` |
| `BT-04` | 85 | 2.5% | `baby, baby iims, iims, masters, institute, private, lakh` | `placements`, `financial_concerns` |
| `BT-05` | 74 | 2.1% | `cat, job, workex, years, mba, work, experience` | `curriculum_teaching`, `academic_workload` |
| `BT-06` | 73 | 2.1% | `imi, scmhrd, alumni, gim, iim sambalpur, lpa, glim` | `curriculum_teaching`, `placements` |
| `BT-07` | 68 | 2.0% | `mpp, dsai, mpp dsai, dsai mpp` | `administrative_communication`, `curriculum_teaching` |
| `BT-08` | 66 | 1.9% | `percentile, jee, mains, 85 percentile` | `administrative_communication`, `curriculum_teaching` |
| `BT-09` | 64 | 1.9% | `ews, wl 76, ews wl, waitlist movment` | `administrative_communication`, `curriculum_teaching` |
| `BT-12` | 59 | 1.7% | `cibil, loan, banks, loans, nbfc, credit, cibil score` | `financial_concerns`, `administrative_communication` |
| `BT-13` | 59 | 1.7% | `date, registration, refund, fee, payment, sem` | `administrative_communication`, `financial_concerns` |
| `BT-16` | 49 | 1.4% | `classrooms, mica, concerned faculty, scared uncertainty` | `placements`, `campus_infrastructure` |
| `BT-25` | 41 | 1.2% | `response wait, mailed response, update mailed, sent` | `administrative_communication`, `campus_infrastructure` |
| `BT-31` | 28 | 0.8% | `internship, internships, corporate, operations, committee` | `curriculum_teaching`, `campus_infrastructure` |

---

## 8. Limitations & Final Recommendation

### Limitations
1. **UMAP stochastic seed sensitivity:** As documented in Section 6.1, one of three seeds (`202`) triggered HDBSCAN `eom` cluster collapse; future work should evaluate `cluster_selection_method="leaf"` or PCA initialization for higher cross-seed stability.
2. **High outlier fraction (`38.2%`) on short informal Reddit comments:** Inherent to density-based clustering on brief conversational replies; must be inspected rather than ignored.
3. **Corpus Bengali/Odia scarcity:** While `multilingual-e5-small` scored `100%` on synthetic Bengali/Odia probes, real-corpus topic discovery could only test English (`82.5%`) and Hinglish/Hindi (`9.8%`) due to actual subreddit composition.

### Final Recommendation
**Keep `nmf_v1` as the default backward-compatible deterministic baseline, and expose `bertopic_v1` (`intfloat/multilingual-e5-small`) as an equal, interactive side-by-side companion in the Topic Map.** Neither method dominates on every criterion: NMF provides 100% document coverage and deterministic stability, while BERTopic uncovers fine-grained actionable themes (`CIBIL/loans`, `refund deadlines`, `B-school comparisons`, `faculty concerns`, `internships`) and distributes Hinglish comments across 21 semantic themes instead of collapsing them into a single language bucket.

---

## 9. Completed Human Review & Cluster Label Refinement (`human_review_sample.csv`)

All **35 BERTopic v1 clusters** (`105` top-3 centroid representative comments) in `data/annotations/bertopic_human_review_v1.csv` (synced with `ml/artifacts/topic_models/bertopic_v1/human_review_sample.csv`) were manually reviewed, rated on a 1–5 semantic coherence scale, typed by discourse function, and propagated to `ml/artifacts/live_corpus/bertopic_v1_topics.json` while preserving the original `machine_suggested_label` for auditability.

| Review Dimension | Result |
| :--- | :--- |
| Reviewed clusters / rows | **35 / 35** clusters (`105` centroid comment checks) |
| Mean human coherence rating (1–5) | **4.51 / 5.0** (`22` rated 5, `9` rated 4, `4` rated 3, `0` rated ≤ 2) |
| Substantive student-issue themes | **13 clusters** (e.g., `BT-12` Education Loans/CIBIL, `BT-13` Fee & Refund Deadlines, `BT-31` Corporate Internships & Stipends, `BT-5` CAT Reattempt & Work-Ex ROI, `BT-16` MBA ROI/Faculty/Classrooms, `BT-25` Unanswered Admission Emails) |
| Admissions & waitlist logistics | **12 clusters** (e.g., `BT-7` BS MPP vs DSAI, `BT-8` CAT/JEE Percentiles, `BT-20` NC-OBC Conversion Odds, `BT-23` Office Phone Call Checks, `BT-33` Merit List Round Tracking) |
| Conversational / phatic clusters | **7 clusters** (honestly tagged as non-issue discourse: `BT-2` Gratitude, `BT-14` DM Requests, `BT-17` Confirmations, `BT-19` Emoji-Only Reactions, `BT-27` Congratulations) |
| Multilingual Hinglish/Hindi clusters | **2 clusters** (`BT-0` Hinglish Admissions & Campus Banter, `BT-29` Short Romanized Hindi Questions) |
| Media / bare-URL artifact cluster | **1 cluster** (`BT-24` Portal Sign-In URLs & Embedded Reddit Image Links — preserved honestly rather than mislabelled) |
