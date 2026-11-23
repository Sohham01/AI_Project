# Deliverable 3 — AI / NLP Prototype Guide (Application, APIs, Artifacts & Verification)

> **Document ID:** `RCI-V6.2-D3` · **Status:** Verified Production Prototype (`2,204/2,204` offline tests · `16/16` Vercel zero-ML checks · Dual-theme browser QA passed)  
> **Navigation:** [← Deliverable 2 (Dataset)](./DELIVERABLE_2_DATASET_PACKAGE.md) · [Submission Index](./README.md) · [Deliverable 4 (Architecture) →](./AI_ARCHITECTURE.md)

---

## 1. Running the Application

### Option A — Docker Compose (Recommended Full Local Stack)
```bash
docker compose up -d --build web
# Open http://localhost:8000/  (automatically redirects browsers to /app)
```

### Option B — Local Python Virtualenv (`backend/.venv`)
```bash
# Windows PowerShell
$env:PYTHONPATH = "backend"
.\backend\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000

# macOS / Linux
PYTHONPATH=backend ./backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Option C — Slim Serverless Deployment (Vercel)
- **ASGI Entrypoint:** [`api/index.py`](../../api/index.py) (identical security, GZip compression, CDN `Cache-Control`, and HTML root redirect parity with [`backend/app/main.py`](../../backend/app/main.py)).
- **Zero-ML Serving Footprint:** Root [`requirements.txt`](../../requirements.txt) installs `fastapi` + `pydantic` only (`0 MB` runtime ML libraries). Heavy offline training, embeddings, BERTopic clustering, and 5-fold Transformers emit compact read-only JSON serving artifacts under `ml/artifacts/live_corpus/`.

---

## 2. Repository Structure (Core V6.2 Modules)

```text
AI_Project/
├── api/index.py                                    # Slim Vercel ASGI entrypoint (GZip + Security + / -> /app)
├── backend/
│   ├── app/
│   │   ├── main.py                                 # FastAPI app + static SPA mount + GZipMiddleware
│   │   ├── core/security.py                        # CSP, security headers & CDN Cache-Control middleware
│   │   ├── api/demo.py                             # Read-only V6.2 Studio API (13 endpoints + compact mode)
│   │   ├── services/nlp/
│   │   │   ├── indian_reddit_lid.py                # Pure-stdlib span LID, script separation & Romanization
│   │   │   ├── preprocessing/
│   │   │   │   ├── slang_normalizer.py             # Base informal/Gen-Z lexicon
│   │   │   │   ├── emoji_normalizer.py             # Affect-preserving emoji semantics
│   │   │   │   └── context_normalizer.py           # Context-aware slang guards (mid / based / cooked)
│   │   │   ├── topic_discovery/nmf_topic_discoverer.py         # Preserved NMF v1 baseline
│   │   │   ├── issue_solution_linking/tfidf_cosine_linker.py   # Preserved V5 lexical linker
│   │   │   ├── resolution_detection/evidence_based_resolver.py # Preserved V1 deterministic resolver
│   │   │   ├── community_intelligence.py           # Cross-IIM, cross-language & V6 live analyzer
│   │   │   └── cross_model_insights.py             # Evidence-linked Cross-Model Insight Engine v2
│   │   └── static/                                 # Zero-build ES Module SPA (10 views, Dusk Dark / Daylight Light, CSV/MD export)
│   └── tests/                                      # Automated API, BERTopic, LID, linking, resolution & PII tests
├── data/
│   ├── live_corpus/corpus.json                     # 365 anonymized threads, 3,557 comments (v6.1.0)
│   └── annotations/                                # Tracked human evaluation sets (V1 459, LID 84, Pairs 64, Res 40, BERTopic 35)
├── ml/
│   ├── topics/bertopic/                            # Modular BERTopic v1 (config, embeddings, pipeline, eval, registry)
│   ├── linking/
│   │   ├── contextual_mapper.py                    # Multilingual E5 + negation-aware mapper + agglomerative clusters
│   │   └── resolution_engine.py                    # 8-signal explainable unresolved-issue engine
│   ├── experiments/v6_programme/                   # Reproducible V6 evaluation & ablation runners
│   └── artifacts/
│       ├── live_corpus/                            # Compact serving JSONs (NMF, BERTopic v1, E5 Map v2, Resolution v2, Eval, Preds)
│       └── topic_models/                           # Local SafeTensors BERTopic v1 + registry.json (gitignored weights)
└── docs/academic_submission/                       # Consolidated Deliverables 1–6
```

---

## 3. Backend API Surface (`/api/demo/*`)

All read-only GET endpoints emit `Content-Encoding: gzip` (≥ 1 KB) and `Cache-Control: public, max-age=300, stale-while-revalidate=3600`.

| Endpoint | Method | Wire Size (GZip) | Purpose & V6.2 Capabilities |
| :--- | :---: | :---: | :--- |
| `/` | `GET` | `0.1 KB` | Smart content negotiation: `307 → /app` for browsers (`Accept: text/html`); `200 JSON` for API clients. |
| `/api/demo/health` | `GET` | `0.4 KB` | Liveness, corpus counts (`365` posts, `3,557` comments), and dataset version. |
| `/api/demo/summary` | `GET` | `11.4 KB` | Reconciled tally KPIs, V6 resolution summary (`resolution_v2_summary`), 4-field provenance, community network, and evidence-linked insights. |
| `/api/demo/topics?model=nmf\|bertopic` | `GET` | `4.7 / 37.1 KB` | Backward-compatible NMF v1 default (`10` factors) + explicit **BERTopic v1** companion (`35` human-reviewed clusters, `1,320` `-1` outliers, comparison table). |
| `/api/demo/issues` | `GET` | `15.2 KB` | 8 issue domains enriched with Multilingual-E5 matched solutions, semantic clusters, and explainable resolution verdicts. |
| `/api/demo/discussions` & `/{post_id}` | `GET` | `2.7 KB` | Filterable discussions & nested reply trees enriched with span-level LID/script/abstention and parent context. |
| `/api/demo/sentiment` | `GET` | `0.8 KB` | Monthly polarity spline, multi-label emotions, issue polarity matrix, and language-register breakdown. |
| `/api/demo/responses` | `GET` | `1.8 KB` | Locked 6-class canonical distribution, 3-example switcher, and difficult boundary diagnostics. |
| `/api/demo/resolution?compact=true` | `GET` | `90.6 KB` | **11.8× wire reduction** vs. legacy uncompressed `1,068.7 KB`; exposes V6 8-signal explainable matrix, `26` contradictory threads, and lazy-expandable audit rows. |
| `/api/demo/models` | `GET` | `25.6 KB` | Strict OOF Governance Audit, 4-architecture benchmark, BERTopic vs. NMF table, 4-field provenance, latency profile, and subsystem benchmarks. |
| `/api/demo/insights` | `GET` | `7.0 KB` | Standalone evidence-linked cross-model insight cards (separated `observation` vs. `interpretation`). |
| `/api/demo/comparisons` | `GET` | `4.1 KB` | Cross-IIM (`19` IIMs), Cross-Language register comparison, and LID/preprocessing benchmark tables. |
| `/api/demo/analyze` | `POST` | `< 3 KB` | Real-time zero-GPU multilingual lab: span-level LID, context-aware slang disambiguation diff, emoji affect decoding, and `k=12` corpus projection. |

---

## 4. Sample Input / Output

### 4.1 `GET /api/demo/topics?model=bertopic` (Verified Excerpt)
```json
{
  "model": "bertopic",
  "model_info": {
    "model_id": "bertopic_v1",
    "embedding_model": "intfloat/multilingual-e5-small",
    "preprocessing_mode": "raw_clean",
    "n_documents": 3459,
    "n_topics": 35,
    "outlier_count": 1320,
    "outlier_rate": 0.3816,
    "human_review_summary": {
      "reviewed_topics": 35,
      "mean_coherence_rating_1_to_5": 4.51,
      "review_source": "data/annotations/bertopic_human_review_v1.csv"
    }
  },
  "items": [
    {
      "topic_id": 12,
      "namespaced_topic_id": "bertopic:bertopic_v1:12",
      "label": "BT-12: Education Loans, CIBIL Score & Fee Financing",
      "machine_suggested_label": "BT-12: cibil · loan · banks · loans",
      "label_status": "human_reviewed",
      "human_coherence_rating": 5,
      "cluster_type": "substantive",
      "document_count": 59,
      "keywords": ["cibil", "loan", "banks", "loans", "nbfc", "credit", "cibil score"]
    }
  ]
}
```

### 4.2 `POST /api/demo/analyze` (Context-Aware Slang & Span-LID Excerpt)
- **Input:** `{"text": "After mid-semester exams my quiz score is totally cooked, though the placement cell is based on merit."}`
- **Verified Output Highlights:**
  - `language_detection`: `language: "en"`, `script: "latin"`, `confidence: 0.90`, `spans: [...]`
  - `preprocessing.context_slang_transformations`:
    - `mid` → `literal_preserved` *(temporal compound `mid-semester`)*
    - `based` → `literal_preserved` *(prepositional phrase `based on`)*
    - `cooked` → `context_expanded` → `"overwhelmed doomed"` *(Gen-Z evaluative predicate)*

---

## 5. Studio Views & SaaS Research Workflows

All 10 studio views support **Dusk Midnight (Dark)** and **Daylight Bone (Light)** themes, `Ctrl+K` keyboard command palette (with focus trap & ARIA active-descendant), shareable URL query parameters, and client-side zero-server exports (`UTF-8 BOM CSV` and `Executive Brief .md`):

1. **Overview (`#/overview`):** Reconciled 4-metric Tally Strip (`20.0%` actionable replies; `46 / 365` explicit outcomes with `12` confirmed / `26` contradictory), Evidence-Linked Insight Explorer + `Export brief (.md)`, Tripartite Community Network, Issue Friction Quadrant, and Monthly Sentiment Spline.
2. **Issues (`#/issues`):** Sortable 8-domain ledger + sticky Deep Dive Inspector with URL-synced `?selected=` state and Multilingual-E5 semantic solution clusters.
3. **Discussions (`#/explorer`):** Master-detail split workbench with `j` / `k` keyboard stepping, 7 filter dimensions, and `Export page CSV`.
4. **Conversation Audit (`#/thread/:id`):** Nested parent-child reply tree, `?comment=` deep-link scroll/highlight anchors, span-level LID badges, and collapsible 5-task probability distributions.
5. **Topic Map (`#/topics`):** URL-synced (`?model=&tab=&topic=`) **NMF v1 vs. BERTopic v1** switcher, `T0..T9` / `BT-0..BT-13` proportional constellation SVG, discourse-type filters (`Substantive`, `Logistics`, `Conversational`, `URL / Media`), human-reviewed coherence ratings (`4.51/5`), explicit `-1` Outlier Inspector, side-by-side comparison table, and `Export CSV`.
6. **Sentiment (`#/sentiment`):** Interactive monthly polarity spline, multi-label emotion prevalence, language-register sentiment table, and issue polarity matrix.
7. **Responses (`#/responses`):** Locked 6-class canonical radial wheel, interactive 3-example cards, and difficult boundary analysis.
8. **Resolution (`#/resolution`):** Unified **V6 Explainable Engine | Legacy V5 Baseline** switcher, ruled 5-cell verdict stat strip, interactive **Contradiction & Outcome Triage Queue** (`26` contradictory / `12` confirmed) with `Jump to evidence →` comment deep-links, paginated 365-thread audit table (`25`/page) with lazy expansion, and `Export CSV`.
9. **Compare & Lab (`#/compare`):** URL-synced (`?tab=`) Cross-IIM (`19` IIMs + `Export CSV`), Languages & NLP Benchmarks, and **V6 Live Multilingual Analyzer** with Span-LID chips and Context-Aware Slang Disambiguation Diff.
10. **Models & Data (`#/models`):** Pipeline Integrity bar (`7/7`), Strict OOF Governance Audit stat strip, 4-architecture dumbbell chart, BERTopic vs. NMF comparison table, DAPT perplexity convergence, calibration reliability plots, inference latency profile, and 4-field Data Provenance Vault.

---

## 6. Automated Verification Suite

```bash
# Full offline workspace test suite (2,204 passed, 0 failed, 973 subtests passed)
pytest -q --ignore=ml/experiments --ignore=backend/test_migrations_postgres.py

# Targeted V6 pipeline, linking, resolution & API contract tests
PYTHONPATH=backend pytest backend/tests/test_bertopic_pipeline.py backend/tests/test_phase4_phase5_linking_resolution.py backend/tests/test_demo_api.py backend/test_app_startup.py -q
```

- **Slim Vercel Zero-ML Audit:** All `16 / 16` endpoints on [`api/index.py`](../../api/index.py) verified HTTP `200` (and `307 → /app` for HTML root requests) with zero ML packages installed at serving time.
- **Dual-Theme Browser QA:** Verified across both Dusk Dark and Daylight Bone Light modes (`0` console errors, `0` HTTP ≥ 400 requests).
