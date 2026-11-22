# Reddit Community Intelligence (RCI)

[![CI — Backend Tests & Vercel Entrypoint Audit](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab?style=flat&labelColor=1e2235)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat&labelColor=1e2235)
![Corpus v6.1.0](https://img.shields.io/badge/corpus-v6.1.0%20%C2%B7%20365%20threads%20%C2%B7%203%2C557%20comments-4338ca?style=flat&labelColor=1e2235)
![Pipeline Integrity 7/7](https://img.shields.io/badge/pipeline%20integrity-7%2F7%20checks-15803d?style=flat&labelColor=1e2235)
![License MIT](https://img.shields.io/badge/license-MIT-4a526e?style=flat&labelColor=1e2235)

**Multilingual research studio for anonymized & pseudonymized public Reddit discussions** around **IIM Sambalpur** and Indian MBA/CAT communities. **V6.1 (`v6.1.0`)** promotes our **100% strictly inductive, zero-leakage Track A 5-fold ensembles** (`MuRIL-DAPT-WWM-8ep` for Issue/Sentiment/Emotion + `XLM-RoBERTa-base` for Response & Solution; `0/44` test-thread DAPT overlap; `0` synthetic training/test rows), apples-to-apples **4-architecture 5-fold benchmarking** (Logistic Regression, ANN/MLP, 1D TextCNN, Transformer), **OOF per-label monotonic Platt calibration** (`0.059` mean 5-bin ECE), **TF-IDF Cosine Issue–Solution Linking** (`859` links across `162` threads) + **Quantitative Unresolved Issue Criteria**, **Cross-IIM (`19` IIMs) & Cross-Language analytics**, a **Live Multilingual Analyzer**, and **10-cluster NMF topic discovery** — served via a zero-build FastAPI studio on both **Vercel** and **Docker Compose**.

[Live Studio (Vercel)](https://rci-ten.vercel.app/app/) · [System Architecture](#system-architecture) · [ML Pipeline](#multilingual-nlp-pipeline) · [4-Architecture Real-Data Benchmarks](#4-architecture-benchmarks-on-real-held-out-reddit-comments) · [Deployment & Security](#deployment--security) · [Quickstart](#quickstart)

![Reddit Community Intelligence Studio Overview](docs/demo_screenshots/overview-v3-desktop.png)

---

## At a Glance

| Dimension | Current Production State (`v6.1.0` — Strictly Inductive Track A) |
|---|---|
| **Real Pseudonymized Corpus** | `iim-sambalpur-real-reddit-v6` — **365 pseudonymized threads (`post_0001`..`post_0365`)**, **3,557 human comments (`cmt_00001`..`cmt_03557`, `2,046` with verified parent replies)** across **8 active subreddits** (`r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, `r/Indian_Academia`); all raw Reddit IDs, URLs, and usernames stripped |
| **Noise, Privacy & Language Tagging** | AutoMod/RemindMeBot stickies + GIF-only embeds + empty threads removed; `IndianRedditLID` identifies `2,898` English, `340` Hinglish/Indic code-mixed (`315` `hi-en`, `23` `hi`, `2` `or-en`; `9.6%`), and `319` terse comments (`0` bogus European tags vs. `22.9%` overall / `60.6%` Indic error rate in `langdetect`) |
| **100% Real Labeled Training & Test Set** | Tracked [data/annotations/real_reddit_labeled_v1/](data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv) — **459 curated real Reddit comments across 200 distinct threads** (`367` 5-fold group-aware CV trainval / `92` untouched frozen real held-out test across `37` unseen threads, `0` synthetic rows). A 50-item blind human IAA audit on Track B V2 failed `0/5` threshold gates ([V5_HUMAN_IAA_REPORT.md](docs/V5_HUMAN_IAA_REPORT.md)), so Track B V2 stays archived/blocked and production serves strictly verified Track A V1 |
| **Strictly Inductive WWM DAPT & NMF** | **MuRIL-DAPT-WWM-8ep** (`##`-aligned Whole-Word Masking, **44 frozen-test & same-title threads / 749 comments strictly excluded, `0` overlap**) trained on `2,135` unlabeled texts reduced **unseen thread-grouped WWM validation perplexity `88.32 → 25.23`** on `347` unseen texts; **10-cluster NMF** (`8,000`-term vocab) fit directly on real comments |
| **V6.1 Production Models & Calibration** | **5-fold pure neural ensembles (`65.59%` frozen test mean F1, `+6.11 pp` vs. transductive V5 `59.48%`!)**: `MuRIL-DAPT-WWM` (`parent_pair + ASL + OOF Platt` on Issue, `ASL + OOF Platt` on Emotion, `CE-weighted` on Sentiment) + **`XLM-RoBERTa-base`** (zero DAPT, OOF winner on **Solution `84.57%`** & **Response `56.42%`**!). Achieves **`0.059` mean 5-bin ECE**, `72.2%` live mean confidence (`915` high / `2,459` medium / `183` low), and `859` cosine-linked issue–solution edges across `162` threads |
| **Automated Verification** | **2,196 offline tests** (`973` subtests) + **7/7 automated Pipeline Integrity Checks** + strict zero-test-thread DAPT & IAA governance assertions + slim Vercel zero-ML import audit + HTTP security headers + zero-PII/URL/secret CI gates |

---

## System Architecture

Public Reddit threads enter through keyless Arctic Shift or static-archive ingestion, pass deterministic PII redaction, bot/GIF removal, and opaque ID pseudonymization, and feed both the **strictly inductive V6 (`MuRIL-DAPT-WWM + XLM-R`) + NMF** pipeline and optional PostgreSQL storage. Precomputed read-only JSON artifacts power the Vercel and Docker studio with zero outbound Reddit calls at request time, while a local loopback-only **Annotation Studio** (`127.0.0.1:8765`) supports blind multi-annotator human IAA review.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-architecture-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-architecture.svg">
  <img alt="End-to-end anonymized Reddit intelligence platform architecture" src="docs/diagrams/rci-architecture.svg">
</picture>

*Editable HTML sources live in [docs/diagrams/src/](docs/diagrams/src/).*

### Research Integrity & Epistemic Boundaries

- **100% Reproducible From Tracked Real Files:** Legacy synthetic bundles (`data/demo_synthetic/`, `ml/artifacts/demo_synthetic*`) are retired; running `python backend/build_live_corpus_artifacts.py` reproduces the entire `v6.0.0` strictly inductive pipeline directly from tracked `data/live_corpus/corpus.json` and `data/annotations/real_reddit_labeled_v1/` with zero synthetic rows, zero test-thread DAPT leakage (`0/44`), and zero hardcoded absolute paths.
- **Transparent Label Provenance & Honest IAA Governance:** The `459` labeled real comments (`367` 5-fold CV trainval / `92` frozen real test) were individually curated comment-by-comment against [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md). When a 50-item blind human IAA review of the Track B V2 expansion scored `0/5` passing gates ([V5_HUMAN_IAA_REPORT.md](docs/V5_HUMAN_IAA_REPORT.md)), Track B V2 was intentionally archived/blocked rather than silently promoted — production serves strictly verified Track A V1 only.
- **6 Canonical Response Classes (fixed order):** `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
- **Proposal vs. Resolution Separation:** Predicted `advice` / `solution` comments (`73.9%`, `241/326` cross-task agreement between `response=="solution"` and `is_solution==True`) are tracked independently from explicit first-person follow-up resolution signals (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`).
- **Pipeline Integrity Scope:** The 7 automated checks in [benchmark_verifier.py](ml/evaluation/benchmark_verifier.py) verify split isolation, real-data provenance, 4-architecture coverage, confusion-matrix reconciliation, non-sharpening `T ≥ 1.0` bounds, held-out DAPT perplexity reduction, and pseudonymized zero-PII/URL hygiene — they do not claim multi-human gold certification.

---

## Multilingual NLP Pipeline

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-ml-pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-ml-pipeline.svg">
  <img alt="Domain-adaptive MuRIL-DAPT-WWM and XLM-R V6 multilingual NLP pipeline" src="docs/diagrams/rci-ml-pipeline.svg">
</picture>

1. **Strictly Inductive WWM DAPT MLM ([muril_dapt.py](ml/models/transformer/muril_dapt.py)):** Continued `##`-aligned Whole-Word Masking MLM of `google/muril-base-cased` for `8` epochs on `2,135` unlabeled Reddit texts (**`44` frozen-test & same-title threads / `749` comments excluded, `0` train/test & `0` val/test overlap**), evaluated on `347` disjoint unseen-thread texts (**Epoch 0 `88.32` → Epoch 8 `25.23` held-out validation perplexity**).
2. **V6 Per-Task 5-Fold Pure Neural Ensembles ([build_live_corpus_artifacts.py](backend/build_live_corpus_artifacts.py), [muril_classifier.py](ml/models/transformer/muril_classifier.py)):** Attention-masked `[CLS] + Mean + Max` pooling, LayerNorm, 5-branch Multi-Sample Dropout, and LLRD (`0.92`) trained across 5 `post_id`-grouped folds on `trainval (367)`. **Issue** adds true parent-comment context (`parent_pair`, `2,046/3,557` live replies) + **Asymmetric Loss (ASL)** + **OOF per-label monotonic Platt calibration** (`0.012` ECE); **Emotion** uses ASL + OOF Platt (`0.007` ECE); **Sentiment** and **Response** use class-weighted CE (`T=1.0`); **Solution** promotes **`xlm-roberta-base`** (zero DAPT, OOF & frozen-test winner at **`84.57%`**).
3. **4-Architecture Comparative Suite ([neural_baselines.py](ml/models/neural_baselines.py)):** Apples-to-apples 5-fold group-aware CV evaluation of **Logistic Regression** (Word+Char TF-IDF), **ANN** (2-Layer GELU+LayerNorm MLP), **CNN** (1D Multi-Kernel TextCNN `k=2..5`), **MuRIL-DAPT-WWM**, and **Deployed V6 (`MuRIL-WWM + XLM-R`)** on identical `post_id`-grouped splits (`367` 5-fold CV / `92` frozen held-out test across `37` unseen threads).
4. **Data-Driven Error Analysis & Integrity Verifier ([error_analysis.py](ml/evaluation/error_analysis.py), [benchmark_verifier.py](ml/evaluation/benchmark_verifier.py)):** Computes confusion matrices, strictly empirical sample diagnostics for every confused class pair (actual word lengths, OOV rates, Hinglish cues, confidence margins, and quoted examples — zero canned lookup text), 5-bin calibration reliability curves on the `n = 92` real test split, and 7 pipeline integrity assertions.
5. **Cross-Model Insight Engine ([cross_model_insights.py](backend/app/services/nlp/cross_model_insights.py)):** Synthesizes all 5 supervised models, 10 NMF topics (`8,000` terms), resolution evidence, calibrated confidence, and real benchmarks into the Issue Friction & Priority Quadrant and 5 dynamically consistent briefings.

---

## 4-Architecture Benchmarks on Real Held-Out Reddit Comments

Primary evaluation is strictly **inductive (`0/44` test-thread DAPT overlap)** on the untouched frozen **`n = 92` real Reddit comments across `37` unseen threads** (all 4 architectures trained apples-to-apples via identical 5-fold group-aware CV on `trainval n = 367` of `real_reddit_labeled_v1`, `0` synthetic rows; auxiliary 54-probe stress diagnostic is observational only and never used for model selection; see full ablation reports in [V5_MODEL_IMPROVEMENT_REPORT.md](docs/V5_MODEL_IMPROVEMENT_REPORT.md)):

### Aggregate Comparison (5-Task Mean on Untouched Frozen Real Test, `n = 92`)

| Architecture (All 5-Fold Group CV) | Real Held-Out Test F1 *(Primary)* | Auxiliary 54-Probe F1 *(Obs.)* | Mean 5-Bin ECE | Latency / Comment | Parameters |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** *(5-Fold Word+Char TF-IDF)* | `49.91%` | `33.38%` | `0.098` | `0.69 ms` | `113.5K` |
| **ANN** *(5-Fold 2-Layer MLP · GELU + LayerNorm)* | `43.62%` | `40.87%` | `0.093` | `0.72 ms` | `821.4K` |
| **CNN** *(5-Fold 1D Multi-Kernel TextCNN `k=2..5`)* | `42.09%` | `26.05%` | `0.094` | `0.13 ms` | `339.7K` |
| **Transformer (`MuRIL-DAPT-WWM` Strict Inductive)** | **`62.99%`** | `53.76%` | **`0.053`** | `5.42 ms` | `237.6M` |
| **Deployed V6.1 (`MuRIL-WWM + XLM-R`, Pure Neural)** | **`65.59%`** | **`54.07%`** | **`0.059`** | `5.47 ms` | `237.6M / 278M` |

### Per-Task Real Held-Out Test Macro F1 (`n = 92` Unseen-Thread Real Comments)

| Task | LogReg | ANN (MLP) | 1D TextCNN | **MuRIL-WWM** | **Deployed V6.1** | Calibration | Calib. ECE |
|---|---:|---:|---:|---:|---:|---|---:|
| **Issue** *(8-label · `parent_pair` · ASL)* | `39.80%` | `35.79%` | `27.70%` | **`60.83%`** | **`60.83%`** *(MuRIL-WWM)* | OOF Platt | **`0.012`** |
| **Solution** *(binary · CE-weighted)* | `64.16%` | `55.28%` | `60.65%` | `74.44%` | **`84.57%`** *(XLM-R)* | 5-Fold `T=1.0` | `0.104` |
| **Sentiment** *(3-class · CE-weighted)* | `60.88%` | `49.09%` | `56.62%` | **`65.11%`** | **`65.11%`** *(MuRIL-WWM)* | 5-Fold `T=1.0` | `0.104` |
| **Response** *(6 classes · CE-weighted)* | `38.17%` | `35.64%` | `37.72%` | `53.52%` | **`56.42%`** *(XLM-R)* | 5-Fold `T=1.0` | **`0.066`** |
| **Emotion** *(6-label · ASL)* | `46.55%` | `42.29%` | `27.77%` | **`61.04%`** | **`61.04%`** *(MuRIL-WWM)* | OOF Platt | **`0.007`** |

> **Why V6.1 beats both baselines and historical transductive V5 (`59.48%`):** With **zero test-thread DAPT overlap (`0/44`)**, V6.1 lifts 5-task frozen test Macro F1 to **`65.59%`** (**`+6.11 pp`** over transductive V5, **`+15.68 pp`** over 5-fold LogReg, **`+21.97 pp`** over ANN, **`+23.50 pp`** over CNN), led by strictly OOF-selected improvements on **Issue (`45.7% → 60.8%`)**, **Solution (`76.1% → 84.6%`, `XLM-R`)**, **Sentiment (`59.5% → 65.1%`)**, and **Response (`53.8% → 56.4%`, `XLM-R`, ECE `0.085 → 0.066`)**, while monotonic OOF Platt calibration cuts ASL ECE **`12×–22×`**. Offline batch inference compiles predictions to read-only JSON so Vercel/Docker serve responses in `<10 ms` with zero runtime GPU cost.

---

## Deployment & Security

RCI separates public read-only analytics, self-hosted container persistence, and offline GPU/annotation workflows across three explicit trust boundaries:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-deployment-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-deployment.svg">
  <img alt="Dual Vercel serverless, Docker Compose, and offline workstation topology" src="docs/diagrams/rci-deployment.svg">
</picture>

### Security & Protection Controls

- **Pseudonymized Zero-PII Public Artifacts:** Usernames, emails, phone numbers, `u/` handles, raw Reddit base36 IDs, and `reddit.com/r/...` URLs are stripped and replaced with opaque surrogate IDs (`post_0001`, `cmt_00001`), asserted on every push/PR in [.github/workflows/ci.yml](.github/workflows/ci.yml).
- **Defensive HTTP Security Headers:** Shared `SecurityHeadersMiddleware` ([security.py](backend/app/core/security.py)) and [vercel.json](vercel.json) enforce `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`, `Cross-Origin-Opener-Policy: same-origin`, and strict `Content-Security-Policy` (`frame-ancestors 'none'`, `object-src 'none'`).
- **Hardened Container & Loopback Isolation:** [Dockerfile](Dockerfile) executes Uvicorn under non-root `appuser` with a `/health` probe; [docker-compose.yml](docker-compose.yml) binds PostgreSQL strictly to `127.0.0.1:5432` (never exposed on LAN); [run_annotation_studio.py](backend/run_annotation_studio.py) binds strictly to `127.0.0.1:8765`.
- **Repository Protection:** `main` branch protection enforces the CI check (`strict: true`) and blocks force pushes/deletions; Dependabot vulnerability alerts, automated security fixes, and grouped weekly updates ([dependabot.yml](.github/dependabot.yml)) are active. See [SECURITY.md](SECURITY.md) for the full policy.

---

## Quickstart

### 1. Docker Compose (Recommended — FastAPI + PostgreSQL 16)

```bash
docker compose up -d --build
docker compose exec web alembic upgrade head
```

Open **http://localhost:8000/app/** (`http://localhost:8000/docs` for OpenAPI, `http://localhost:8000/health` for liveness).

### 2. Local Python 3.14

```bash
python -m venv backend/.venv
# Windows: .\backend\.venv\Scripts\activate  |  macOS/Linux: source backend/.venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 3. Reproducing the Full `v6.0.0` Strictly-Inductive ML Pipeline (100% Real Data)

```bash
# Optional: set RCI_MODEL_CACHE_DIR to store HuggingFace/DAPT weights outside the repo
python backend/build_live_corpus_artifacts.py --dataset-version 6.0.0 --dapt-epochs 8 --muril-epochs 6
```

### 4. Local Human Annotation Studio (`/annotate`) & Offline Tests

```bash
python backend/run_annotation_studio.py --host 127.0.0.1 --port 8765
pytest backend/ --ignore=backend/test_migrations_postgres.py -q
```

---

## Core API Endpoints

All studio analytics routes are read-only and serve identical contracts on Vercel (`api/index.py`) and Docker (`backend/app/main.py`):

| Endpoint | Purpose |
|---|---|
| `GET /app/` | Interactive Dusk / Daylight research studio UI |
| `GET /health`, `GET /health/db` | Liveness probe and optional PostgreSQL connectivity check |
| `GET /api/demo/summary` | Corpus KPIs, calibrated confidence summary, Cross-Model Insight Engine briefings, Issue Priority Matrix, and recent threads |
| `GET /api/demo/discussions` & `/{id}` | Paginated thread search, `?confidence=high\|medium\|low\|review` filtering, and threaded comment confidence audit |
| `GET /api/demo/issues`, `/topics` | 8-label issue breakdown and 10-cluster NMF topic map |
| `GET /api/demo/sentiment`, `/responses`, `/resolution` | Sentiment/emotion trends, 6 canonical response roles, linked community solution clusters (Task 10), quantitative unresolved criteria (Task 11), and explicit resolution audit |
| `GET /api/demo/comparisons`, `POST /api/demo/analyze` | Cross-IIM (`19` IIMs) & Cross-Language analytics (Q8), Task 5–6 studies, and real-time zero-GPU multilingual comment analyzer |
| `GET /api/demo/models` | 4-architecture real-test benchmark, held-out DAPT perplexity convergence, 7/7 Pipeline Integrity Checks, and per-task evaluation |
| `GET /api/demo/error-analysis` | Data-driven per-task confusion matrices, empirical failure diagnostics, 5-bin reliability curves, Transformer recoveries, and residual errors |
| `GET /api/demo/insights` | Cross-Model Insight Engine briefings, Issue Friction & Priority Quadrant data, and corpus confidence distribution |

---

## Repository Layout

```text
api/                                   # Slim Vercel serverless FastAPI entrypoint (api/index.py)
backend/
  app/
    api/demo.py                        # Read-only Studio API (/api/demo/*)
    core/                              # Lazy DB engine/session proxies + HTTP SecurityHeadersMiddleware
    models/ & schemas/                 # SQLAlchemy 2.0 ORM + Pydantic v2 contracts
    services/ingestion/                # Arctic Shift, Reddit OAuth, Static Export providers
    services/nlp/                      # Preprocessing, script/Hinglish detection, NMF, classical NLP, Cross-Model Insight Engine
    static/                            # Dusk / Daylight interactive studio & SVG visualizations (zero JS build step)
  alembic/                             # Linear PostgreSQL migration history
  annotation_studio_ui/                # Local loopback human annotation workbench UI
  tests/                               # 2,193 unit, contract, benchmark, integrity, security, and PII tests
  build_live_corpus_artifacts.py       # Reproducible 100%-real DAPT MLM + 4-arch benchmark + calibration + error analysis builder
data/
  annotations/real_reddit_labeled_v1/  # Tracked 459 curated real anonymized Reddit comments across 200 threads (0 synthetic rows)
  live_corpus/                         # Pseudonymized public Reddit corpus (corpus.json, v6.0.0)
ml/
  artifacts/live_corpus/               # Precomputed v6 predictions, NMF topics, real 4-arch benchmarks & data-driven error analysis
  evaluation/                          # Data-driven Error Analysis module & 7-check Pipeline Integrity Verifier
  models/                              # LogReg / ANN (MLP) / 1D TextCNN baselines + MuRIL-DAPT-WWM / XLM-R classifiers
docs/
  diagrams/                            # Editorial Light/Dark SVGs + self-contained HTML sources
  demo_screenshots/                    # Studio UI preview captures
  *.md                                 # Pilot guide, annotation guidelines, V5/V6 optimization, Human IAA & gap analysis reports
```

---

## Documentation, Formal Deliverables & Contributing

- **Deliverable 1 (Video Script):** [DELIVERABLE_1_VIDEO_SCRIPT.md](docs/DELIVERABLE_1_VIDEO_SCRIPT.md)
- **Deliverable 2 (Data Dictionary & Sample-Size Rationale):** [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) · [Human Annotation Guidelines (v2.0)](docs/ANNOTATION_GUIDELINES.md)
- **Deliverable 3 (Prototype Walkthrough Notebook):** [notebooks/RCI_V6_Prototype_Walkthrough.ipynb](notebooks/RCI_V6_Prototype_Walkthrough.ipynb)
- **Deliverable 4 (11-Section AI Architecture):** [DELIVERABLE_4_AI_ARCHITECTURE.md](docs/DELIVERABLE_4_AI_ARCHITECTURE.md)
- **Deliverable 5 (Mathematical Foundations & Pseudocode):** [DELIVERABLE_5_MATH_AND_PSEUDOCODE.md](docs/DELIVERABLE_5_MATH_AND_PSEUDOCODE.md)
- **Deliverable 6 (Final Insight Report):** [DELIVERABLE_6_FINAL_INSIGHT_REPORT.md](docs/DELIVERABLE_6_FINAL_INSIGHT_REPORT.md)
- **Research Audits:** [Task 5 & 6 LID/Preprocessing Report](docs/TASK5_TASK6_PREPROCESSING_LID_REPORT.md) · [V6 Stage 3 Tuning Report](docs/V6_STAGE3_MODEL_TUNING_REPORT.md) · [Human IAA Governance Report](docs/V5_HUMAN_IAA_REPORT.md) · [Problem Statement Gap Analysis](docs/PROBLEM_STATEMENT_GAP_ANALYSIS.md)
- **Policies:** [Security & Data Privacy Policy](SECURITY.md) · [Contributing Guide](CONTRIBUTING.md) · [Local Demo Runbook](docs/LOCAL_DEMO_RUNBOOK.md)

## License

MIT — see [LICENSE](LICENSE).
