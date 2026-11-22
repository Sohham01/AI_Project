# Deliverable 4 — AI Architecture Document (`RCI v6.1.0`)

> **Specification Alignment:** Follows the exact **11 required sections** of Deliverable 4 (PDF Section 19, pp. 18–19) and Task 12 (pp. 12–13), grounded strictly in the deployed **Reddit Community Intelligence (RCI) `v6.1.0`** strictly inductive system.

---

## 1. Problem Definition

Higher-education institutions lack an automated, continuous way to synthesize candid student and prospective-student discourse from Reddit, where posts mix English, Hindi, Hinglish, Banglish, Odia, Gen-Z slang, and emojis within a single thread. Rather than stopping at polarity counts, RCI models the complete community discourse chain:

$$\text{Issue} \longrightarrow \text{Discussion} \longrightarrow \text{Community Response} \longrightarrow \text{Linked Solution} \longrightarrow \text{Resolution Evidence}$$

Operational definitions and six mandatory boundary tie-breakers are codified in `docs/ANNOTATION_GUIDELINES.md`:
- **8 Multi-Label Issue Domains:** `academic_workload`, `administrative_communication`, `campus_infrastructure`, `curriculum_teaching`, `extracurriculars`, `financial_concerns`, `hostel_facilities`, `placements`.
- **3 Sentiment Polarities:** `negative`, `neutral`, `positive`.
- **6 Multi-Label Emotions:** `confidence`, `curiosity`, `frustration`, `gratitude`, `hope`, `stress`.
- **6 Canonical Community Response Roles (Locked Exact Order):** `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
- **Binary Actionable Solution Detection:** `solution` vs. `not_solution`.
- **Explicit Proposal-vs-Resolution Separation:** 4 thread outcome states (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`) plus quantitative issue-level criteria (`Resolved` / `Partially Resolved` / `Unresolved`).

---

## 2. Data Acquisition

- **Ingestion Layer (`backend/app/services/ingestion/`):** Modular provider architecture supporting keyless historical public archives (`ArcticShiftProvider`), static JSON exports (`StaticExportProvider`), and optional Reddit OAuth (`RedditOAuthProvider`), unified behind `IngestionManager`.
- **Real Public Corpus (`data/live_corpus/corpus.json`, `v6.1.0`):** `365` anonymized threads (`post_0001`..`post_0365`) and `3,557` human comments (`cmt_00001`..`cmt_03557`, `2,046` with verified immediate parent-comment links) across **8 subreddits**: `r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, and `r/Indian_Academia`.
- **Privacy & Hygiene Controls:** AutoModerator/RemindMeBot stickies, deleted/GIF-only comments, and zero-comment threads are dropped; all author handles, emails, phone numbers, `u/` mentions, raw Reddit URLs, and raw base-36 Reddit IDs are deterministically redacted/pseudonymized and verified in CI.
- **Supervised Ground Truth (`data/annotations/real_reddit_labeled_v1/`):** `459` manually curated real Reddit comments spanning `200` distinct threads (`367` 5-fold CV trainval / `92` frozen unseen-thread real test across `37` threads; `0` synthetic rows). See `docs/DATA_DICTIONARY.md` for full schemas and sample-size rationale.

---

## 3. Data Preprocessing

Implemented as deterministic, non-destructive stages in `backend/app/services/nlp/preprocessing/` (`original_text` is never destroyed):
1. **Unicode NFC & Zero-Width Cleaning** (`unicode_normalizer.py`) — preserves Devanagari, Bengali, Odia, and Latin characters intact.
2. **Whitespace, URL/Mention & Repeated-Char Normalization** (`whitespace_normalizer.py`, `url_mention_normalizer.py`, `repeated_char_normalizer.py`) — collapses `soooo` → `soo` without collapsing natural pairs (`good`, `??`, `!!`).
3. **Punctuation Normalization** (`punctuation_normalizer.py`) — preserves `?`, `??`, `!`, `!!` as curiosity/affect cues.
4. **Empirical Task 6 Findings (`docs/TASK5_TASK6_PREPROCESSING_LID_REPORT.md`):**
   - Conventional **stop-word removal + stemming + punctuation removal** severely hurts multilingual Reddit NLP (**`-1.40 pp`** on TF-IDF and **`-14.34 pp`** on `MuRIL-DAPT-WWM` Affect F1) by destroying negation (`not`/`no`) and Hinglish morphemes, so it is **explicitly rejected**.
   - **Emoji Normalization (`emoji_normalizer.py`):** Preserves emojis instead of stripping them, supporting both deterministic `<EMOJI_name>` tokens (best for TF-IDF: `+0.75 pp`) and natural-language semantic affect verbalization (`😭 → sadness frustration`, `💀 → extreme amusement disbelief`, `🤡 → sarcasm mockery`, `🔥 → strong approval`), which lifts `MuRIL-DAPT-WWM` OOF Sentiment by `+2.27 pp` and Emotion by `+1.23 pp`.
   - **Conservative Gen-Z Slang Handling (`slang_normalizer.py` v1.1.0):** Expands unambiguous abbreviations (`idk`, `ngl`, `fr`, `tbh`, `imo`, `btw`, `cfbr`) while intentionally preserving tone markers (`bro`, `bhai`, `yaar`, `lol`) and polysemous MBA words (`mid` in *"mid 30s"*, `based` in *"based on"*, `cooked`) to avoid false corruption, surfacing all detected cues via `detect_slang_terms()`.

---

## 4. Language Handling

Implemented in pure Python stdlib (`backend/app/services/nlp/indian_reddit_lid.py`) for identical execution offline, on Docker, and on Vercel serverless:
- **Multi-Script Unicode Detection:** Distinguishes `latin`, `devanagari` (`hi`), `bengali` (`bn`), `odia` (`or`), `gurmukhi`, `gujarati`, `tamil`, `telugu`, `kannada`, `malayalam`, and `mixed` script-switching (e.g., Odia + English *"ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro"* → `or-en`, `mixed`).
- **Romanized Code-Mix Detection:** Uses curated collision-free markers (`STRONG_HINGLISH`, `STRONG_BANGLISH`, `STRONG_ROMAN_ODIA`) paired with English/MBA anchors to detect `hi-en` (`315` live comments), `hi` (`23`), `or-en` (`2`), `en` (`2,898`), and `und` (`319`) with **zero bogus European labels**.
- **Existing-LID Limitation Proof (Task 5):** On our `3,557` real comments, off-the-shelf `langdetect 1.0.9` emits bogus non-Indian tags (`id`, `sw`, `so`, `et`, `tl`, `cy`, `vi`, etc.) on **`22.9%` (`815/3,557`) overall** and **`60.6%` (`206/340`) of code-mixed comments** (tagging the other `39.4%` as English, `0%` correctly as code-mixed). On the 12-item multilingual probe suite covering all 5 PDF languages and scripts, **`IndianRedditLID` scores `100.0%` vs. `langdetect` `33.3%`**.

---

## 5. Model Architecture

RCI `v6.1.0` uses a **decoupled multi-stage architecture** where each task is served by its **strictly OOF-selected, zero-leakage 5-fold pure neural ensemble** alongside unsupervised topic modeling and thread-scoped issue–solution linking:

| Component / Task | Production Architecture (`v6.1.0`) | Input Context | Loss & Calibration | Parameters |
|---|---|---|---|---:|
| **Domain Adaptation (DAPT)** | `google/muril-base-cased` + `##`-aligned Whole-Word Masking (`8` ep, strictly inductive `0/44` test overlap) | Unlabeled non-test threads (`2,135` train / `347` val) | WWM MLM (`val PPL 88.32 → 25.23`) | `237.6M` |
| **Issue (8-label)** | `MuRIL-DAPT-WWM` (5-fold `[CLS]+Mean+Max` + 5× MSD) | `parent_pair` (true parent reply or post title via `[SEP]`) | Asymmetric Loss (ASL) + OOF Monotonic Platt | `237.6M` |
| **Sentiment (3-class)** | `MuRIL-DAPT-WWM` (5-fold `[CLS]+Mean+Max` + 5× MSD) | `comment_only` | Class-Weighted CE + Native 5-Fold `T=1.0` | `237.6M` |
| **Emotion (6-label)** | `MuRIL-DAPT-WWM` (5-fold `[CLS]+Mean+Max` + 5× MSD) | `comment_only` | Asymmetric Loss (ASL) + OOF Monotonic Platt | `237.6M` |
| **Response (6-class)** | **`xlm-roberta-base`** *(OOF winner, zero DAPT)* (5-fold) | `comment_only` | Class-Weighted CE + Native 5-Fold `T=1.0` | `278.0M` |
| **Solution (binary)** | **`xlm-roberta-base`** *(OOF winner, zero DAPT)* (5-fold) | `comment_only` | Class-Weighted CE + Native 5-Fold `T=1.0` | `278.0M` |
| **Topic Discovery (Task 7)** | Unsupervised **10-Cluster NMF** (`8,000`-term TF-IDF) | All `3,557` real comments | Frobenius NMF | — |
| **Issue–Solution Linking (Task 10)** | `TfidfCosineIssueSolutionLinker` (`same_post_only=True`, `τ=0.45`) + Theme Clustering | Thread issue + candidate solutions | Weighted cosine blend (`859` links / `162` threads) | — |
| **4-Arch Baselines (Benchmarked)** | 5-Fold Word+Char TF-IDF LogReg, 2-Layer GELU+LN ANN, 1D Multi-Kernel TextCNN (`k=2..5`) | Identical `post_id`-grouped folds | Apples-to-apples benchmarked every build | `113.5K–821.4K` |

---

## 6. Hardware Architecture & GPU Necessity Justification (Task 12, p. 12)

```text
Offline Research Workstation (Training & Batch Inference)
  Public Archive (Arctic Shift) ──► CPU Hygiene/PII Redaction ──► NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, bf16 AMP)
                                                                    ├─ WWM DAPT MLM (8 epochs: ~6.8 min, 5.58 GB VRAM)
                                                                    ├─ 5-Task × 5-Fold Ensembles (~8.7 min total, 4.0–4.6 GB VRAM)
                                                                    └─ Length-Sorted Batch Inference on 3,557 comments (~19s total)
                                                                               │
                                                                               ▼  Precomputed Read-Only JSON Artifacts (~9.2 MB total)
Production Serving Tier (Zero Runtime GPU Required)
  Vercel Serverless (Python 3.14, 256 MB RAM, 0 GPU)  ──or──  Docker Compose (FastAPI + PostgreSQL 16, CPU only)
  Serves all 10 Studio views + Live Multilingual Analyzer in <15 ms/request at $0 GPU hosting cost.
```

### Is a GPU Actually Necessary? (Empirical Evidence)
- **For Offline Training & DAPT — YES (Strongly Justified):**
  - 8-epoch WWM DAPT of `237.6M`-param MuRIL takes **`~6.8 minutes`** in `bf16` on an 8 GB RTX 4060 (`~5.58 GB` peak VRAM); on a modern 8-core laptop CPU, equivalent masked language modeling exceeds **~3–5 hours**.
  - Training all 5 supervised 5-fold ensembles (`25` transformer checkpoints, `max_length=256`) takes **`~8.7 minutes` (`522 s`)** on GPU vs. **~4–6 hours** on CPU — enabling disciplined multi-phase ablations (context, losses, DAPT, backbones, R-Drop, Stage 3 tuning) that would be impractical on CPU.
  - GPU inference achieves **`5.47 ms/comment`** across full 5-fold Transformer ensembles (`~1.1 ms` per single checkpoint) with length-sorted batching, vs. `~48–90 ms/comment` per checkpoint on CPU.
- **For Production Dashboard Serving — NO (Architecturally Decoupled):**
  - Because Reddit community intelligence is analyzed in scheduled batches rather than sub-millisecond trading loops, offline GPU batch compilation writes compact read-only JSON artifacts (`9.2 MB` total). Both Vercel serverless (`requirements.txt`: `fastapi` + `pydantic` only) and Docker run purely on **CPU with `<10 ms` API latency and zero GPU operational cost**, while real-time comment inspection (`POST /api/demo/analyze`) runs pure-stdlib Task 5/6 NLP + sparse V6 corpus-neighbor projection in `<15 ms`.

---

## 7. Software Architecture

- **Backend (`backend/app/`):** FastAPI (`0.141`) + Pydantic v2 (`2.13`) + SQLAlchemy 2.0 + Alembic migrations (optional PostgreSQL 16 for live ingestion) + `SecurityHeadersMiddleware` (`CSP`, `X-Frame-Options: DENY`, `nosniff`, `COOP`).
- **Studio API (`/api/demo/*`):** 12 read-only / stateless endpoints (`health`, `summary`, `discussions`, `discussions/{id}`, `issues`, `topics`, `sentiment`, `responses`, `resolution`, `comparisons`, `models`, `error-analysis`, `insights`, `POST /analyze`).
- **Frontend (`backend/app/static/`):** Zero-build ES-module research studio (Dusk Midnight & Daylight Bone themes, `10` interactive views including Overview with Model Architecture & Community Network diagrams, Issues, Discussions, Topic Map, Sentiment, Responses, Resolution with Tasks 10–11 matrices, Compare & Lab with Cross-IIM/Language/Live Analyzer, and Models & Data).
- **Human-in-the-Loop Workbench (`backend/run_annotation_studio.py`):** Loopback-only (`127.0.0.1:8765`) blind Annotation Studio for multi-annotator IAA audits.

---

## 8. Training

- **Leak-Free Partitioning:** Group-split strictly by original `post_id` (`367` trainval across `163` threads / `92` frozen held-out test across `37` threads). All `44` frozen-test and same-title threads (`749` comments) are purged before WWM DAPT (`0` train/test and `0` val/test overlap).
- **5-Fold Group-Aware Cross-Validation:** `StratifiedGroupKFold` (single-label) and Group-Preserving Iterative Stratification (multi-label `Issue`, `Emotion`) inside `trainval (367)`.
- **Optimization Schedule:** AdamW (`weight_decay = 0.01`), `lr = 4e-5` with LLRD (`0.92`) and `4×` head LR multiplier, cosine decay with `10%` warmup, `batch_size = 8`, `max_length = 256`, `6` epochs with validation-loss/F1 early checkpointing, gradient clipping `1.0`, `bf16` mixed precision.
- **Reproducibility Command:** `python backend/build_live_corpus_artifacts.py --dataset-version 6.1.0 --dapt-epochs 8 --muril-epochs 6` reproduces every artifact from tracked real files (`0` synthetic rows, `0` hardcoded paths).

---

## 9. Evaluation

Apples-to-apples 5-fold group-aware CV on `trainval (367)`, evaluated once on the untouched frozen real held-out test set (`n = 92`, `37` unseen threads):

| Architecture (All 5-Fold Group CV) | Issue | Solution | Sentiment | Response | Emotion | **5-Task Mean F1** | Mean 5-Bin ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression *(Word+Char TF-IDF)* | `39.80%` | `64.16%` | `60.88%` | `38.17%` | `46.55%` | `49.91%` | `0.098` |
| ANN *(2-Layer GELU+LN MLP)* | `35.79%` | `55.28%` | `49.09%` | `35.64%` | `42.29%` | `43.62%` | `0.093` |
| CNN *(1D Multi-Kernel TextCNN)* | `27.70%` | `60.65%` | `56.62%` | `37.72%` | `27.77%` | `42.09%` | `0.094` |
| Transformer (`MuRIL-DAPT-WWM` Strict) | `60.83%` | `74.44%` | `65.11%` | `53.52%` | `61.04%` | `62.99%` | `0.053` |
| **Deployed V6.1 (`MuRIL-WWM + XLM-R`)** | **`60.83%`** | **`84.57%`** | **`65.11%`** | **`56.42%`** | **`61.04%`** | **`65.59%`** | **`0.059`** |

- **Gain Over Historical Transductive V5 (`59.48%`):** **`+6.11 pp`** while simultaneously *eliminating* all test-thread DAPT leakage (`0/44`).
- **Automated Verification:** `7 / 7` Pipeline Integrity Verifier checks pass (`zero_split_leakage`, `real_labeled_provenance`, `four_architecture_coverage`, `metric_reconciliation`, `calibration_and_probability_bounds`, `dapt_heldout_perplexity_check`, `zero_pii_and_pseudonymized_ids`) + `2,196` offline automated tests passing.

---

## 10. Deployment Architecture

Three strict trust boundaries:
1. **Public Read-Only Cloud (`Vercel Serverless` via `api/index.py` & `vercel.json`):** Installs only `fastapi` + `pydantic`, bundles read-only JSON artifacts + static UI (`<10 ms` responses, zero DB/Reddit outbound calls, strict HTTP security headers).
2. **Self-Hosted Container Stack (`Docker Compose` via `Dockerfile` + `docker-compose.yml`):** Non-root `appuser` Uvicorn web service (`/health` probe) + PostgreSQL 16 bound strictly to `127.0.0.1:5432` (never LAN-exposed).
3. **Offline ML & Annotation Boundary:** GPU training (`build_live_corpus_artifacts.py`) and blind human annotation (`127.0.0.1:8765`) run strictly locally; CI (`.github/workflows/ci.yml`) gates every PR/push on tests, Vercel entrypoint audit, and zero-PII checks.

---

## 11. Limitations

1. **Multilingual Corpus Skew:** While `IndianRedditLID` handles Hindi, Bengali, Odia, Hinglish, Banglish, and Roman-Odia (`100%` probe accuracy), real public Reddit discussions around IIM Sambalpur/CAT are heavily English (`81.5%`) and Hinglish (`9.5%`), with only `2` Roman-Odia comments and `0` native Bengali/Odia script posts in the scraped archive.
2. **Small Rare-Class Test Support (`n = 92`):** Classes like `academic_workload` (`7` test rows), `curriculum_teaching` (`5`), and `counterargument` (`9`) carry wide bootstrap confidence intervals (`±10–15 pp`), and `counterargument` remains our hardest individual class (`~0.24` F1).
3. **Human IAA Governance Hold on Track B V2:** Our genuine 50-item blind human review failed `0/5` threshold gates (`κ = 0.26–0.46`, MASI `α = 0.39`) due to 6 rubric boundary ambiguities. Production therefore remains locked on clean Track A V1 (`459` rows) until a calibrated dual-annotator pass clears `κ >= 0.70 / 0.65` using `ANNOTATION_GUIDELINES.md` Section 8 and `iaa_calibration_30.csv`.
4. **Live Comment Analyzer Inference Mode:** To keep Vercel serverless at `0 GPU` and `<15 ms`, `POST /api/demo/analyze` runs deterministic Task 5/6 NLP live and projects task scores from the `k = 12` nearest precomputed V6 neural predictions (explicitly disclosed in the UI rather than pretending to execute 500M-parameter GPU Transformers per request).
5. **Observational Resolution Evidence:** Explicit first-person follow-up confirmation appears in only `4.7%` (`17/365`) of threads — reflecting Reddit's drop-off behavior rather than proof that unconfirmed proposals failed.
