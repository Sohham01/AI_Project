# Local synthetic demo runbook

## Scope and provenance

**Reddit Community Intelligence** is served locally by the existing FastAPI application. Its demo endpoints read only generated files from `data/demo_synthetic/` and `ml/artifacts/demo_synthetic_v2/` (with the original pre-audit artifacts preserved untouched in `ml/artifacts/demo_synthetic/`); they do not use PostgreSQL, contact Reddit, or call external APIs.

Persistent banner: **SYNTHETIC DEMO DATA — NOT REAL REDDIT DATA**. Every post, comment, identifier, scenario and label is fabricated. The corpus is not evidence about IIM Sambalpur or any real student community.

Dataset metadata is in `data/demo_synthetic/corpus.json`:

- Dataset ID/version: `reddit-community-intelligence-demo` / `1.0.0`
- Seed: `20261116`
- 160 generated posts and 1,280 generated comments (8 per thread)
- Eight scenario issue categories and all six canonical response classes
- Explicit fabricated provenance: `synthetic_generator`, `synthetic_only`
- Generation timestamp is recorded separately; generated text, IDs and scenario labels are seeded

Labels are assigned by the generator's scenario rules, independently of model predictions. The six response classes and their order remain unchanged: `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.

## Start the local app

Requirements: the existing Python environment at `backend/.venv`; no frontend package manager, credentials, database, or environment variables are required for the synthetic demo.

In PowerShell:

```powershell
Set-Location D:\AI_Project\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/app/>. The root URL retains the existing health-style JSON response. API documentation is at <http://127.0.0.1:8000/docs>.

The app does not generate data or train models on startup. Keep `AUTO_CREATE_SCHEMA` unset/false; setting it true opts the existing application into schema creation. The synthetic demo endpoints do not use the database. Do not call `/health/db` for this local synthetic demo.

## Generate or retrain explicitly

Generated files already exist. The commands below refuse to overwrite a non-empty demo directory unless `--force` is supplied. Regeneration changes only the synthetic demo corpus; retraining changes only the isolated synthetic artifacts.

```powershell
Set-Location D:\AI_Project\backend
.\.venv\Scripts\python.exe generate_synthetic_demo.py --force
.\.venv\Scripts\python.exe train_demo_baselines.py --force
```

Run training after regenerating the corpus. Application restart alone never rewrites these files. Data files: `data/demo_synthetic/`. Model weights, registry, predictions, topic outputs and evaluation: `ml/artifacts/demo_synthetic/`.

## Implemented views

- **Overview** — generated post/comment counts, issue-frequency counts, model-predicted sentiment and response distributions, resolution-state counts and dataset provenance.
- **Discussion explorer** — search over post/comment text; community, scenario issue, sentiment, emotion and response filters; 12-thread pagination; complete thread detail with parent context and confidence values.
- **Issues and topics** — generated scenario-category totals, topic assignments, representative generated comments and links to their threads. NMF terms are unsupervised, noisy and not validated research themes.
- **Sentiment and emotion** — model predictions by issue and generated month. The timeline is synthetic.
- **Community responses** — all six canonical response types, predicted counts and examples.
- **Solution and resolution** — scenario-labeled solution proposals, model outputs and explicit generated outcome evidence are separated. A proposed solution is not treated as proof of resolution. Missing reports are not asserted as real-world failures.
- **Model evaluation** — held-out metrics, class counts, per-class/per-label metrics and confusion matrices where applicable. Marked synthetic-only.

## Local API

All prototype routes are under `/api/demo`:

- `GET /health`
- `GET /summary`
- `GET /discussions` — `q`, `community`, `issue`, `sentiment`, `emotion`, `response`, `page`, `page_size`
- `GET /discussions/{post_id}`
- `GET /issues`
- `GET /topics`
- `GET /sentiment`
- `GET /responses`
- `GET /resolution`
- `GET /models`

Response models are Pydantic-validated. Filter values are checked; pagination is bounded to 1–100 items per page.

## Baseline training notes

The existing `Trainer` and task models were used for five synthetic-only baselines: sentiment, emotion, issue, response and solution. Each uses post/thread-grouped train/validation/test splits (896 / 192 / 192 records) with seed `20261116`; the recorded leakage checks report no cross-split leakage. Evaluation JSON includes the exact per-task metrics and limitations. Predictions shown on discussion pages are in-sample predictions on synthetic demo records; evaluation results are computed on the held-out test split. High scores do not imply generalization beyond this generator.

No pretrained model or dataset was downloaded. No real collection, credential transmission, database connectivity check, or migration was performed.

## Frontend implementation note

The repository had no usable frontend package manifest or installed React/Vite stack. To keep the app locally runnable without package downloads or network access, the frontend is a dependency-free HTML/CSS/JavaScript application served by FastAPI; there is no React or npm build step. Charts use accessible, data-derived HTML/CSS bars rather than an external chart library. This is the main deviation from the preferred React architecture.

## Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_demo_api.py
.\.venv\Scripts\python.exe -m pytest -q --ignore=test_migrations_postgres.py
```

The PostgreSQL migration test is excluded because this prototype must not touch a configured database. Browser checks and screenshots are recorded separately in the delivery summary.

## Known limitations

- All data and outcome reports are generated scenarios, not observations or real student statements.
- Baseline predictions are trained on this synthetic generator family; distribution, topic, emotion and response outputs are not validated on genuine Reddit content.
- Existing NMF topic terms are bag-of-words output and can be noisy, especially across code-mixed and Devanagari text.
- No authorized real-data ingestion or real-world model evaluation is included. Any future collection requires appropriate authorization and ethics review.
- The frontend is not React because dependencies were absent and external package retrieval was not used.

## Deployment (Docker & Vercel)

The repository includes configuration for both **Docker** (full backend + classical ML stack + PostgreSQL) and **Vercel** (zero-config serverless hosting of the Interactive Research Studio and `/api/demo/*` API).

### 1. Docker & Docker Compose (Full Stack)

Run the full FastAPI + scikit-learn + PostgreSQL stack using [Dockerfile](file:///D:/AI_Project/Dockerfile) and [docker-compose.yml](file:///D:/AI_Project/docker-compose.yml):

```powershell
Set-Location D:\AI_Project
docker compose up --build
```

Or build and run the standalone web container without PostgreSQL (the synthetic demo routes do not require a database):

```powershell
Set-Location D:\AI_Project
docker build -t rci-app .
docker run --rm -p 8000:8000 rci-app
```

Then open <http://127.0.0.1:8000/app/>.

### 2. Vercel (Serverless Web & Demo API)

The repository root includes [vercel.json](file:///D:/AI_Project/vercel.json), [api/index.py](file:///D:/AI_Project/api/index.py), and a slim root [requirements.txt](file:///D:/AI_Project/requirements.txt) (`fastapi` + `pydantic`) tailored for Vercel's Python serverless runtime and 250 MB bundle limit:

1. Import the GitHub repository (`Sohham01/AI_Project`) at <https://vercel.com/new>, or run `vercel --prod` from `D:\AI_Project`.
2. Keep the Root Directory set to `./` (repository root). No build command or environment variables are required for the synthetic demo.
3. Visiting `/` automatically redirects to `/app/`, serving the Hallmark Dusk Interactive Research Studio backed by `/api/demo/*` and the bundled `data/demo_synthetic/` and `ml/artifacts/demo_synthetic_v2/` artifacts.
