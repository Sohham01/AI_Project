# Reddit Community Intelligence (RCI)

[![CI — Backend Tests & Vercel Entrypoint Audit](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab?style=flat&labelColor=1e2235)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat&labelColor=1e2235)
![Corpus v5.0.0](https://img.shields.io/badge/corpus-v5.0.0%20%C2%B7%20365%20threads%20%C2%B7%203%2C557%20comments-4338ca?style=flat&labelColor=1e2235)
![Pipeline Integrity 7/7](https://img.shields.io/badge/pipeline%20integrity-7%2F7%20checks-15803d?style=flat&labelColor=1e2235)
![License MIT](https://img.shields.io/badge/license-MIT-4a526e?style=flat&labelColor=1e2235)

**Multilingual research studio for anonymized & pseudonymized public Reddit discussions** around **IIM Sambalpur** and Indian MBA/CAT communities. Trained and evaluated **100% on real curated Reddit comments** (`0` synthetic rows) with unsupervised domain-adaptive pretraining (**MuRIL-DAPT v3** validated on held-out unseen texts), apples-to-apples **4-architecture benchmarking** (Logistic Regression, ANN/MLP, 1D TextCNN, Transformer), **non-sharpening calibrated confidence (`T ≥ 1.0`)**, **data-driven Error Analysis**, a **Cross-Model Insight Engine**, **10-cluster NMF topic discovery**, and **explicit resolution-evidence tracking** — served via a zero-build FastAPI studio on both **Vercel** and **Docker Compose**.

[Live Studio (Vercel)](https://rci-ten.vercel.app/app/) · [System Architecture](#system-architecture) · [ML Pipeline](#multilingual-nlp-pipeline) · [4-Architecture Real-Data Benchmarks](#4-architecture-benchmarks-on-real-held-out-reddit-comments) · [Deployment & Security](#deployment--security) · [Quickstart](#quickstart)

![Reddit Community Intelligence Studio Overview](docs/demo_screenshots/overview-v3-desktop.png)

---

## At a Glance

| Dimension | Current Production State (`v5.0.0`) |
|---|---|
| **Real Pseudonymized Corpus** | `iim-sambalpur-real-reddit-v5` — **365 pseudonymized threads (`post_0001`..`post_0365`)**, **3,557 human comments (`cmt_00001`..`cmt_03557`)** across **8 active subreddits** (`r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, `r/Indian_Academia`); all raw Reddit IDs, URLs, and usernames stripped |
| **Noise, Privacy & Language Tagging** | `365` AutoMod/RemindMeBot stickies + `8` GIF-only embeds + `119` empty threads removed; Indian-Reddit language tagging identifies `2,914` English, `324` Hinglish/Hindi (`9.1%`), and `319` terse/undetermined comments (`0` bogus European language tags) |
| **100% Real Labeled Training & Test Set** | Tracked [data/annotations/real_reddit_labeled_v1/](data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv) — **459 rubric-curated real Reddit comments across 200 distinct threads** (`298` train / `69` val / `92` held-out real test, strictly `post_id`-grouped, `0` synthetic rows; multi-human κ adjudication pending via local Annotation Studio) |
| **Unsupervised DAPT MLM & NMF** | **MuRIL-DAPT MLM** trained on `2,780` unlabeled Reddit texts reduced **held-out validation perplexity `94.35 → 20.12`** on `379` disjoint unseen texts; **10-cluster NMF** (`8,000`-term vocab) fit directly on real comments |
| **Calibration & Cross-Model Engine** | Validation-only non-sharpening temperature scaling (`T ≥ 1.0`) achieving **`0.065` mean 5-bin ECE** across all 5 tasks (`70.8%` live mean confidence; `861` high / `2,410` medium / `286` low), Issue Friction & Priority Quadrant, and 5 dynamically consistent cross-model briefings |
| **Automated Verification** | **2,193 offline tests** (`973` subtests) + **7/7 automated Pipeline Integrity Checks** + 15 Vercel route checks + HTTP security headers + zero-PII/URL/secret/gitlink CI gates |

---

## System Architecture

Public Reddit threads enter through keyless Arctic Shift or static-archive ingestion, pass deterministic PII redaction, bot/GIF removal, and opaque ID pseudonymization, and feed both the **MuRIL-DAPT v3 + NMF** pipeline and optional PostgreSQL storage. Precomputed read-only JSON artifacts power the Vercel and Docker studio with zero outbound Reddit calls at request time, while a local loopback-only **Annotation Studio** (`127.0.0.1:8765`) supports multi-annotator human review.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-architecture-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-architecture.svg">
  <img alt="End-to-end anonymized Reddit intelligence platform architecture" src="docs/diagrams/rci-architecture.svg">
</picture>

*Editable HTML sources live in [docs/diagrams/src/](docs/diagrams/src/).*

### Research Integrity & Epistemic Boundaries

- **100% Reproducible From Tracked Real Files:** Legacy synthetic bundles (`data/demo_synthetic/`, `ml/artifacts/demo_synthetic*`) are retired; running `python backend/build_live_corpus_artifacts.py` reproduces the entire `v5.0.0` pipeline directly from tracked `data/live_corpus/corpus.json` and `data/annotations/real_reddit_labeled_v1/` with zero synthetic rows and zero hardcoded absolute paths.
- **Transparent Label Provenance:** The `459` labeled real comments were individually curated comment-by-comment against [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) to eliminate synthetic training/testing data; multi-annotator human κ adjudication remains tracked as pending via the local Annotation Studio.
- **6 Canonical Response Classes (fixed order):** `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
- **Proposal vs. Resolution Separation:** Predicted `advice` / `solution` comments (`69.9%` cross-task agreement between `response=="solution"` and `is_solution==True`) are tracked independently from explicit first-person follow-up resolution signals (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`).
- **Pipeline Integrity Scope:** The 7 automated checks in [benchmark_verifier.py](ml/evaluation/benchmark_verifier.py) verify split isolation, real-data provenance, 4-architecture coverage, confusion-matrix reconciliation, non-sharpening `T ≥ 1.0` bounds, held-out DAPT perplexity reduction, and pseudonymized zero-PII/URL hygiene — they do not claim multi-human gold certification.

---

## Multilingual NLP Pipeline

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-ml-pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-ml-pipeline.svg">
  <img alt="Domain-adaptive MuRIL v3 and calibrated hybrid NLP pipeline" src="docs/diagrams/rci-ml-pipeline.svg">
</picture>

1. **Held-Out Validated DAPT MLM ([muril_dapt.py](ml/models/transformer/muril_dapt.py)):** Continued masked language modeling of `google/muril-base-cased` on `2,780` unlabeled Reddit texts, evaluated against `379` disjoint unseen Reddit texts (**pre-DAPT Epoch 0 `94.35` → Epoch 4 `20.12` held-out validation perplexity**).
2. **MuRIL v3 Pooled Classifier ([muril_classifier.py](ml/models/transformer/muril_classifier.py)):** Attention-masked `[CLS] + Mean + Max` pooling, LayerNorm, 5-branch Multi-Sample Dropout, LLRD (`0.92`), multi-label `val_loss` early stopping, and validation-only **non-sharpening temperature scaling (`T ≥ 1.0`)**.
3. **4-Architecture Comparative Suite ([neural_baselines.py](ml/models/neural_baselines.py)):** Apples-to-apples evaluation of **Logistic Regression** (Word+Char TF-IDF), **ANN** (2-Layer GELU+LayerNorm MLP), **CNN** (1D Multi-Kernel TextCNN `k=2..5`), and **MuRIL-DAPT v3 Transformer** (+ validation-calibrated Hybrid with `α ≥ 0.50`) on identical `post_id`-grouped real Reddit splits (`298 / 69 / 92`).
4. **Data-Driven Error Analysis & Integrity Verifier ([error_analysis.py](ml/evaluation/error_analysis.py), [benchmark_verifier.py](ml/evaluation/benchmark_verifier.py)):** Computes confusion matrices, strictly empirical sample diagnostics for every confused class pair (actual word lengths, OOV rates, Hinglish cues, confidence margins, and quoted examples — zero canned lookup text), 5-bin calibration reliability curves on the `n = 92` real test split, and 7 pipeline integrity assertions.
5. **Cross-Model Insight Engine ([cross_model_insights.py](backend/app/services/nlp/cross_model_insights.py)):** Synthesizes all 5 supervised models, 10 NMF topics (`8,000` terms), resolution evidence, calibrated confidence, and real benchmarks into the Issue Friction & Priority Quadrant and 5 dynamically consistent briefings.

---

## 4-Architecture Benchmarks on Real Held-Out Reddit Comments

Primary evaluation is strictly on **`n = 92` held-out real Reddit comments from unseen threads** (`post_id`-grouped `65/15/20` split of `459` real comments across `200` threads; `0` synthetic rows; auxiliary 54-probe stress diagnostic is observational only and never used for model selection):

### Aggregate Comparison (5-Task Mean on Real Held-Out Reddit Test, `n = 92`)

| Architecture | Real Held-Out Test F1 *(Primary)* | Auxiliary 54-Probe F1 *(Obs.)* | Mean 5-Bin ECE | Latency / Comment | Parameters |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** *(Word+Char TF-IDF)* | `47.52%` | `34.48%` | `0.094` | `0.14 ms` | `91K` |
| **ANN** *(2-Layer MLP · GELU + LayerNorm)* | `43.48%` | `43.22%` | `0.109` | `0.13 ms` | `821K` |
| **CNN** *(1D Multi-Kernel TextCNN `k=2..5`)* | `42.29%` | `25.34%` | `0.069` | `0.03 ms` | `306K` |
| **Transformer (`MuRIL-DAPT v3`)** | **`59.55%`** | **`51.51%`** | `0.076` | `1.28 ms` | `237.6M` |
| **Production Hybrid** *(MuRIL + TF-IDF, `α ≥ 0.5`)* | **`59.48%`** | `51.10%` | **`0.065`** | `1.45 ms` | `237.7M` |

### Per-Task Real Held-Out Test Macro F1 (`n = 92` Unseen-Thread Real Comments)

| Task | LogReg | ANN (MLP) | 1D TextCNN | **MuRIL-DAPT v3** | Production Hybrid | Temp (`T ≥ 1`) | Calib. ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Issue** *(8-label + title)* | `41.51%` | `34.28%` | `35.01%` | **`45.72%`** | **`45.72%`** *(`α=1.0`)* | `1.00` | `0.060` |
| **Solution** *(binary)* | `62.09%` | `56.27%` | `59.33%` | **`75.00%`** | **`76.10%`** *(`α=0.8`)* | `1.12` | `0.021` |
| **Sentiment** *(3-class)* | `58.50%` | `51.64%` | `51.61%` | **`59.50%`** | **`59.50%`** *(`α=1.0`)* | `1.70` | `0.143` |
| **Response** *(6 classes)* | `36.98%` | `36.18%` | `32.46%` | **`55.20%`** | **`53.79%`** *(`α=0.9`)* | `1.56` | `0.072` |
| **Emotion** *(6-label)* | `38.53%` | `39.02%` | `33.03%` | **`62.31%`** | **`62.31%`** *(`α=1.0`)* | `1.06` | `0.028` |

> **Why we chose Transformer (`MuRIL-DAPT v3`):** On genuine, messy, unseen-thread Indian student Reddit discussions (`n = 92` real held-out test comments), `MuRIL-DAPT v3` beats Logistic Regression, 2-layer ANN, and 1D TextCNN on **all 5 out of 5 tasks** — lifting mean Macro F1 by **`+12.03 pp` over LogReg, `+16.07 pp` over ANN, and `+17.26 pp` over CNN**, with the largest gains on pragmatic 6-class **Response (`+18.2 pp`)**, multi-label **Emotion (`+23.3 pp`)**, and binary **Solution (`+14.0 pp`)**, while the calibrated hybrid achieves the **lowest mean 5-bin ECE (`0.065`)**. Offline batch inference compiles predictions to read-only JSON so Vercel/Docker serve responses in `<10 ms` with zero runtime GPU cost.

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

### 3. Reproducing the Full `v5.0.0` ML Pipeline (100% Real Data)

```bash
# Optional: set RCI_MODEL_CACHE_DIR to store HuggingFace/DAPT weights outside the repo
python backend/build_live_corpus_artifacts.py --dataset-version 5.0.0 --dapt-epochs 4 --muril-epochs 12
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
| `GET /api/demo/sentiment`, `/responses`, `/resolution` | Sentiment/emotion trends, 6 canonical response roles, and proposal vs. resolution audit |
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
  live_corpus/                         # Pseudonymized public Reddit corpus (corpus.json, v5.0.0)
ml/
  artifacts/live_corpus/               # Precomputed v5 predictions, NMF topics, real 4-arch benchmarks & data-driven error analysis
  evaluation/                          # Data-driven Error Analysis module & 7-check Pipeline Integrity Verifier
  models/                              # LogReg / ANN (MLP) / 1D TextCNN baselines + MuRIL-DAPT v3 Transformer
docs/
  diagrams/                            # Editorial Light/Dark SVGs + self-contained HTML sources
  demo_screenshots/                    # Studio UI preview captures
  *.md                                 # Pilot guide, annotation guidelines, audit & benchmark reports
```

---

## Documentation & Contributing

- [Security & Data Privacy Policy](SECURITY.md) · [Contributing Guide](CONTRIBUTING.md)
- [Real-Data Pilot Guide](docs/REAL_DATA_PILOT_GUIDE.md) · [Human Annotation Guidelines](docs/ANNOTATION_GUIDELINES.md)
- [Robustness Benchmark Report](docs/ROBUSTNESS_BENCHMARK_REPORT.md) · [Model Upgrade & Real Ingestion Notes](docs/MODEL_UPGRADE_AND_REAL_INGESTION.md)
- [ML Integrity Audit](docs/ML_INTEGRITY_AUDIT.md) · [Local Demo Runbook](docs/LOCAL_DEMO_RUNBOOK.md)

## License

MIT — see [LICENSE](LICENSE).
