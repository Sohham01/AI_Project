# Reddit Community Intelligence (RCI)

[![CI — Backend Tests & Vercel Entrypoint Audit](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sohham01/AI_Project/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab?style=flat&labelColor=1e2235)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat&labelColor=1e2235)
![Corpus v4.0.0](https://img.shields.io/badge/corpus-v4.0.0%20%C2%B7%20366%20threads%20%C2%B7%203%2C562%20comments-4338ca?style=flat&labelColor=1e2235)
![Benchmarks Verified 7/7](https://img.shields.io/badge/benchmarks-verified%207%2F7-15803d?style=flat&labelColor=1e2235)
![License MIT](https://img.shields.io/badge/license-MIT-4a526e?style=flat&labelColor=1e2235)

**Multilingual research studio for anonymized public Reddit discussions** around **IIM Sambalpur** and Indian MBA/CAT communities. Combines unsupervised domain-adaptive pretraining (**MuRIL-DAPT v3**), apples-to-apples **4-architecture benchmarking** (Logistic Regression, ANN/MLP, 1D TextCNN, Transformer), **calibrated prediction confidence**, a structured **Error Analysis Studio**, a **Cross-Model Insight Engine**, **10-cluster NMF topic discovery**, **6 canonical community response roles**, and **explicit resolution-evidence tracking** — served exclusively on real anonymized Reddit data via a zero-build FastAPI studio on both **Vercel** and **Docker Compose**.

[Live Studio (Vercel)](https://rci-ten.vercel.app/app/) · [System Architecture](#system-architecture) · [ML Pipeline](#multilingual-nlp-pipeline) · [4-Architecture Benchmarks](#4-architecture-benchmarks-why-transformer) · [Deployment & Security](#deployment--security) · [Quickstart](#quickstart)

![Reddit Community Intelligence Studio Overview](docs/demo_screenshots/overview-v3-desktop.png)

---

## At a Glance

| Dimension | Current Production State (`v4.0.0`) |
|---|---|
| **Real Anonymized Corpus** | `iim-sambalpur-arctic-shift-v4` — **366 anonymized threads**, **3,562 human comments** across **8 active subreddits** (`r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, `r/Indian_Academia`); synthetic demo bundles fully retired |
| **Noise & Privacy Filtering** | Author usernames stripped, inline emails/phones/`u/` mentions redacted, `360` AutoModerator stickies + `8` GIF-only embeds + `118` empty threads removed deterministically |
| **Supervised Tasks (5)** | 8-label **Issue**, binary **Solution**, 3-class **Sentiment**, 6-class **Response**, 6-label **Emotion** |
| **Unsupervised Topics & DAPT** | **10-cluster NMF** (`4,542` vocabulary terms) + unsupervised **MuRIL-DAPT MLM** on `3,271` unique unlabeled Reddit texts (**perplexity `37.19 → 14.86`**) |
| **Confidence & Cross-Model Engine** | Temperature-calibrated confidence scores (`59.6%` corpus mean; `high ≥ 80%`, `medium 55–79%`, `low < 55%` human-review queue), Issue Friction & Priority Quadrant, and 5 synthesized cross-model briefings |
| **Automated Verification** | **2,192 offline tests** (`973` subtests) + **7/7 automated benchmark integrity checks** + 15 Vercel route checks + HTTP security headers + zero-PII/secret/gitlink CI gates |

---

## System Architecture

Public Reddit threads enter through keyless Arctic Shift or static-archive ingestion, pass deterministic PII redaction and bot/GIF filtering, and feed both the **MuRIL-DAPT v3 + NMF** pipeline and optional PostgreSQL storage. Precomputed read-only JSON artifacts power the Vercel and Docker studio with zero outbound Reddit calls at request time, while a local loopback-only **Annotation Studio** (`127.0.0.1:8765`) collects human gold labels to retrain future iterations.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-architecture-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-architecture.svg">
  <img alt="End-to-end anonymized Reddit intelligence platform architecture" src="docs/diagrams/rci-architecture.svg">
</picture>

*Editable HTML sources live in [docs/diagrams/src/](docs/diagrams/src/).*

### Research Integrity & Epistemic Boundaries

- **6 Canonical Response Classes (fixed order):** `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
- **Proposal vs. Resolution Separation:** Predicted `advice` / `solution` comments are tracked independently from explicit first-person resolution outcomes (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`).
- **Model-Inferred & Confidence Transparency:** Real-corpus predictions shown in the studio carry calibrated per-comment and per-thread confidence badges (`high` / `medium` / `low` / `review`) and are explicitly surfaced as model-inferred outputs while human gold annotation proceeds via the local Annotation Studio.
- **Zero Synthetic Showcase Data:** Retired legacy `data/demo_synthetic/` and `ml/artifacts/demo_synthetic*` bundles; the studio, API, Docker image, and Vercel deployment exclusively serve verified anonymized real Reddit threads (`data/live_corpus/`).

---

## Multilingual NLP Pipeline

To handle noisy Indian student Reddit discourse (abbreviations, spelling variants, Hinglish code-mixing, and short thread replies) without leaking test targets, RCI pairs unsupervised domain adaptation with multi-architecture evaluation, structured error analysis, and validation-only calibration:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-ml-pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-ml-pipeline.svg">
  <img alt="Domain-adaptive MuRIL v3 and calibrated hybrid NLP pipeline" src="docs/diagrams/rci-ml-pipeline.svg">
</picture>

1. **Unsupervised Domain-Adaptive Pretraining ([muril_dapt.py](ml/models/transformer/muril_dapt.py)):** Continued masked language modeling of `google/muril-base-cased` strictly on `3,271` unique unlabeled anonymized Reddit texts (**perplexity `37.19 → 14.86`** across 4 epochs without fabricating any labels).
2. **MuRIL v3 Pooled Classifier ([muril_classifier.py](ml/models/transformer/muril_classifier.py)):** Attention-masked `[CLS] + Mean + Max` pooling, LayerNorm, 5-branch Multi-Sample Dropout, Layer-wise Learning Rate Decay (`LLRD = 0.90`), single-label smoothing (`0.05`), and validation-only scalar temperature scaling.
3. **4-Architecture Comparative Suite ([neural_baselines.py](ml/models/neural_baselines.py)):** Apples-to-apples evaluation of **Logistic Regression** (Word+Char TF-IDF), **ANN** (2-Layer GELU+LayerNorm MLP), **CNN** (1D Multi-Kernel TextCNN, `k=2,3,4,5`), and **MuRIL-DAPT v3 Transformer** (+ validation-calibrated Hybrid) on identical `post_id`-grouped splits and 54 unseen probes.
4. **Error Analysis, Calibration & Automated Benchmark Verifier ([error_analysis.py](ml/evaluation/error_analysis.py), [benchmark_verifier.py](ml/evaluation/benchmark_verifier.py)):** Generates confusion matrices, ranked top confused class pairs, raw vs. calibrated reliability curves (ECE/Brier), High/Medium/Low confidence-tier accuracy, Transformer win recoveries, residual error taxonomy, and 7 automated integrity assertions.
5. **Cross-Model Insight Engine ([cross_model_insights.py](backend/app/services/nlp/cross_model_insights.py)):** Synthesizes all 5 supervised models, 10 NMF topics (`4,542` vocabulary terms), resolution evidence, calibrated confidence, and OOD benchmarks into the Issue Friction & Priority Quadrant and actionable studio briefings.

---

## 4-Architecture Benchmarks: Why Transformer?

Evaluated across all 5 tasks on identical `post_id`-grouped zero-leakage splits (`seed = 20261116`) and the untouched **54-probe unseen 8-category stress benchmark** (verified **`0` exact / 5-gram / 6-gram probe contamination**; **7/7 automated benchmark verification checks passing**):

### Aggregate Comparison (Mean across All 5 Tasks)

| Architecture | Held-out Test Macro F1 | Unseen 54-Probe OOD F1 | Generalization Drop | Mean Calib. ECE | Latency / Comment | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** *(Word+Char TF-IDF)* | `95.04%` | `57.11%` | `-37.94 pp` | `0.109` | `0.12 ms` | `181K` |
| **ANN** *(2-Layer MLP · GELU + LayerNorm)* | `95.18%` | `52.56%` | `-42.62 pp` | `0.300` | `0.13 ms` | `821K` |
| **CNN** *(1D Multi-Kernel TextCNN `k=2..5`)* | `96.59%` | `49.31%` | `-47.29 pp` | `0.238` | `0.04 ms` | `381K` |
| **Transformer (`MuRIL-DAPT v3`)** | **`97.28%`** | **`68.58%`** | **`-28.70 pp`** | `0.210` | `0.54 ms` | `237.6M` |
| **Production Hybrid** *(MuRIL + TF-IDF)* | **`97.57%`** | `64.50%` | `-33.07 pp` | `0.142` | `0.66 ms` | `237.8M` |

### Per-Task Unseen 54-Probe OOD Macro F1

| Task | LogReg | ANN (MLP) | 1D TextCNN | **MuRIL-DAPT v3** | Production Hybrid | Why Transformer Wins |
|---|---:|---:|---:|---:|---:|---|
| **Issue** *(8-label + title)* | `92.63%` | `84.93%` | `75.25%` | **`93.36%`** | **`94.90%`** *(`α=0.65`)* | Handles paraphrased domain phrasing & parent context |
| **Solution** *(binary)* | `69.90%` | `69.71%` | `59.03%` | **`75.00%`** | **`75.00%`** *(`α=1.00`)* | Distinguishes actionable steps from passive opinions |
| **Sentiment** *(3-class)* | `40.54%` | `39.27%` | `44.26%` | **`64.12%`** | `36.85%` *(`α=0.00`)* | `+19.9` to `+24.9 pp` over baselines on slang & subtle polarity |
| **Response** *(6 classes)* | `52.26%` | `38.86%` | `46.69%` | **`58.29%`** | **`63.64%`** *(`α=0.30`)* | Captures discourse stance; hybrid cuts ECE `0.286 → 0.116` |
| **Emotion** *(6-label)* | `30.20%` | `30.05%` | `21.30%` | **`52.11%`** | **`52.11%`** *(`α=1.00`)* | `+21.9` to `+30.8 pp` over bag-of-words/scratch neural models |

> **Why we chose Transformer (`MuRIL-DAPT v3`):** On held-out in-domain splits, all four architectures look deceptively close (`95.0–97.3%`). Under real unseen language shift (**54 unseen stress probes**), bag-of-words Logistic Regression, ANN, and scratch-trained 1D CNN drop **`37.9–47.3 pp`**, whereas **`MuRIL-DAPT v3` beats every non-transformer architecture on all 5 tasks (`+11.47 pp` over LogReg, `+16.02 pp` over ANN, `+19.27 pp` over CNN)** — driven by massive gains on **paraphrased comments (`77.0%` vs `35.0–52.9%`)**, **informal Reddit slang (`74.7%` vs `52.5–58.7%`)**, **unseen templates (`73.7%` vs `45.1–53.3%`)**, and **parent-post context (`70.4%` vs `40.4–54.6%`)**. Offline batch inference compiles predictions to read-only JSON so Vercel/Docker serve responses in `<10 ms` with zero runtime GPU cost.

---

## Deployment & Security

RCI separates public read-only analytics, self-hosted container persistence, and offline GPU/annotation workflows across three explicit trust boundaries:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/rci-deployment-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/diagrams/rci-deployment.svg">
  <img alt="Dual Vercel serverless, Docker Compose, and offline workstation topology" src="docs/diagrams/rci-deployment.svg">
</picture>

### Security & Protection Controls

- **Zero-PII Public Artifacts:** Usernames are never stored; emails, phone numbers, and `u/` handles are redacted and asserted on every push/PR in [.github/workflows/ci.yml](.github/workflows/ci.yml).
- **Defensive HTTP Security Headers:** Shared `SecurityHeadersMiddleware` ([security.py](backend/app/core/security.py)) and [vercel.json](vercel.json) enforce `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`, `Cross-Origin-Opener-Policy: same-origin`, and strict `Content-Security-Policy` (`frame-ancestors 'none'`, `object-src 'none'`).
- **Hardened Container & Loopback Isolation:** [Dockerfile](Dockerfile) executes Uvicorn under non-root `appuser` with a `/health` probe; [docker-compose.yml](docker-compose.yml) parameterizes Postgres credentials; [run_annotation_studio.py](backend/run_annotation_studio.py) binds strictly to `127.0.0.1:8765` and renders corpus fields via pure DOM `textContent`.
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

### 3. Local Human Annotation Studio (`/annotate`)

Launch the local-only workbench (keyboard triage shortcuts, explicit `uncertain` / `abstain` workflow, atomic CSV writes, live class-quota readiness tracking):

```bash
python backend/run_annotation_studio.py --host 127.0.0.1 --port 8765
```

### 4. Offline Verification Suite

```bash
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
| `GET /api/demo/models` | 4-architecture benchmark comparison, DAPT perplexity convergence, 7/7 benchmark verification, and per-task evaluation |
| `GET /api/demo/error-analysis` | Structured per-task confusion matrices, top confused pairs, calibration reliability bins (ECE/Brier), Transformer wins, and residual failure taxonomy |
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
    services/ingestion/                # Arctic Shift, Reddit OAuth, Static Export, Decodo providers
    services/nlp/                      # Preprocessing, script detection, NMF, classical NLP, Cross-Model Insight Engine
    static/                            # Dusk / Daylight interactive studio & SVG visualizations (zero JS build step)
  alembic/                             # Linear PostgreSQL migration history
  annotation_studio_ui/                # Local loopback human annotation workbench UI
  tests/                               # 2,192 unit, contract, benchmark, integrity, security, and PII tests
  build_live_corpus_artifacts.py       # End-to-end DAPT MLM + 4-arch benchmark + calibration + error analysis builder
data/
  live_corpus/                         # Sanitized real public Reddit corpus (corpus.json, v4.0.0)
ml/
  artifacts/live_corpus/               # Precomputed v4 predictions, NMF topics, 4-arch benchmarks & error analysis
  evaluation/                          # Error Analysis module & 7-check Benchmark Verifier
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
