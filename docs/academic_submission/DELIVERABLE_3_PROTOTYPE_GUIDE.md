# Deliverable 3 — AI/NLP Prototype Guide (Application, APIs, Artifacts & Verification)

---

## 1. Running the Application Locally

### Option A — Docker Compose (Full Local Stack)
```powershell
docker compose up -d --build web
# Open http://localhost:8000/app
```

### Option B — Local Python Virtualenv (`backend/.venv`)
```powershell
$env:PYTHONPATH = "D:\AI_Project\backend"
D:\AI_Project\backend\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/app
```

### Option C — Slim Serverless Deployment (Vercel)
- Entrypoint: [`api/index.py`](file:///D:/AI_Project/api/index.py)
- Root [`requirements.txt`](file:///D:/AI_Project/requirements.txt): `fastapi` + `pydantic` ONLY (`0 MB` ML libraries at request time — all heavy embeddings, BERTopic models, and Transformers run offline and emit compact read-only JSON serving artifacts bundled from `ml/artifacts/live_corpus/`).

---

## 2. Repository Structure (Key V6 Modules)

```text
D:\AI_Project\
├── api/index.py                               # Slim Vercel ASGI entrypoint
├── backend/
│   ├── app/
│   │   ├── main.py                            # FastAPI app + static SPA mount
│   │   ├── api/demo.py                        # Read-only V6 research studio API (13 endpoints)
│   │   ├── services/nlp/
│   │   │   ├── indian_reddit_lid.py           # Phase 2: Pure-stdlib LID, script, spans, Romanization
│   │   │   ├── preprocessing/
│   │   │   │   ├── slang_normalizer.py        # Base slang lexicon
│   │   │   │   ├── emoji_normalizer.py        # Affect-preserving emoji semantics
│   │   │   │   └── context_normalizer.py      # Phase 3: Context-aware Gen-Z disambiguation (mid/based/cooked)
│   │   │   ├── topic_discovery/nmf_topic_discoverer.py  # Preserved NMF v1 baseline
│   │   │   ├── issue_solution_linking/tfidf_cosine_linker.py  # Preserved V5 lexical linker
│   │   │   ├── resolution_detection/evidence_based_resolver.py # Preserved V1 deterministic resolver
│   │   │   ├── community_intelligence.py      # Cross-IIM, cross-language & live analyzer
│   │   │   └── cross_model_insights.py        # Phase 7: Evidence-linked Cross-Model Insight Engine v2
│   │   └── static/                            # Native ES Module SPA (9+ interactive Studio views)
│   └── tests/                                 # 2,200+ automated offline pytest tests
├── ml/
│   ├── topics/bertopic/                       # Phase 1: Modular BERTopic (config, embeddings, pipeline, eval, registry)
│   ├── linking/
│   │   ├── contextual_mapper.py               # Phase 4: Multilingual E5 + negation-aware mapper + clustering
│   │   └── resolution_engine.py               # Phase 5: 8-signal explainable unresolved-issue engine
│   ├── experiments/v6_programme/              # Reproducible Phase 2–5 benchmark scripts
│   └── artifacts/
│       ├── live_corpus/                       # Read-only serving JSONs (NMF, BERTopic v1, E5 Map v2, Resolution v2, Eval, Preds)
│       ├── topic_models/                      # Full SafeTensors BERTopic v1 + registry.json
│       └── v6_programme/                      # Phase 2–5 JSON evaluation reports
└── docs/academic_submission/                  # Consolidated Deliverables 1–6
```

---

## 3. Backend API Surface (`/api/demo/*`)

| Endpoint | Method | Purpose & V6 Additions |
|---|---|---|
| `/api/demo/health` | `GET` | Liveness, corpus counts (`365` posts, `3,557` comments), dataset version. |
| `/api/demo/summary` | `GET` | Tally metrics, 4-field provenance authorizations, issue priority matrix, community network, and **Phase 7 evidence-linked insights**. |
| `/api/demo/topics?model=nmf\|bertopic` | `GET` | Backward-compatible NMF v1 default (`10` factors) + explicit **BERTopic v1** switch (`35` clusters, `-1` outliers, hierarchy, side-by-side comparison). |
| `/api/demo/issues` | `GET` | 8 issue domains enriched with **Phase 4 contextual E5 matched solutions, semantic clusters, and Phase 5 resolution verdicts**. |
| `/api/demo/discussions` & `/discussions/{post_id}` | `GET` | Filterable discussions & nested reply trees enriched with **Phase 2 span-level LID/script/abstention** & parent context. |
| `/api/demo/sentiment` | `GET` | Monthly polarity spline, multi-label emotions, issue matrix, and **language-aware sentiment & emotion breakdown**. |
| `/api/demo/responses` | `GET` | Locked 6-class canonical distribution, 3-example switcher, and **difficult classification boundary diagnostics**. |
| `/api/demo/resolution` | `GET` | 4 resolution states, **Phase 5 8-signal explainable resolution matrix, contradictory follow-ups (`26` threads), and evidence chains**. |
| `/api/demo/models` | `GET` | **Phase 6 Strict OOF Governance Audit**, 4-architecture benchmark, BERTopic vs. NMF table, 4-field provenance, latency profile, and subsystem evals. |
| `/api/demo/insights` | `GET` | Standalone access to evidence-linked cross-model insight briefings. |
| `/api/demo/comparisons` | `GET` | Cross-IIM (19 IIMs) and Cross-Language comparison analytics. |
| `/api/demo/analyze` | `POST` | Live interactive multilingual comment analyzer (LID, spans, context slang/emoji normalization, nearest-neighbor reference). |

---

## 4. Sample Input / Output

### 4.1 `GET /api/demo/topics?model=bertopic` (Excerpt)
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
    "outlier_rate": 0.3816
  },
  "items": [
    {
      "topic_id": 12,
      "namespaced_topic_id": "bertopic:bertopic_v1:12",
      "label": "BT-12: cibil · loan · banks · loans",
      "label_status": "machine_suggested_pending_human_review",
      "document_count": 59,
      "keywords": ["cibil", "loan", "banks", "loans", "nbfc", "credit", "cibil score"]
    }
  ]
}
```

### 4.2 `POST /api/demo/analyze` Input & Response (Excerpt)
Input: `{"text": "Bhai hostel mess food literally mid hai 😭. Check AICTE guidelines for refund."}`  
Output includes `language_detection` (`language: "hi-en"`, `script: "latin"`, `has_multi_language_spans: true`, `spans: [...]`), `preprocessing` (`original_text` preserved, context-expanded `mid → mediocre`, emoji `😭 → sadness frustration`), and retrieved corpus neighbors.

---

## 5. Visualizations Across the 9+ Research Views

1. **Overview (`#/overview`):** Evidence-Linked Insight Explorer (expandable observations, interpretations, supporting posts & quoted comments), Interactive Tripartite Community Network, Honest Linear Issue Friction Quadrant, Monthly Sentiment Spline.
2. **Issues (`#/issues`):** Sortable 8-Domain Table + Deep Dive Inspector showing Phase 4 Multilingual E5 Matched Solutions & Semantic Solution Clusters.
3. **Discussions (`#/explorer`) & 4. Conversation Audit (`#/thread/:id`):** Nested Parent-Child Reply Trees with Span-Level Language/Script/Abstention Badges and 5-Task Probability Bars.
5. **Topic Map (`#/topics`):** Explicit **NMF v1 vs. BERTopic v1** Switcher, Proportional Constellation SVG, Searchable Cards, Explicit `-1` Outlier Inspector, and Comparison Table.
6. **Sentiment (`#/sentiment`):** Interactive Spline Area Chart, Multi-Label Emotion Bars, Language-Aware Sentiment Table, Issue Polarity Matrix.
7. **Responses (`#/responses`):** Animated 6-Class Radial Wheel, Interactive 3-Example Role Cards, and Difficult Boundary Analysis.
8. **Resolution (`#/resolution`):** V6 Explainable Issue → Solution → Contradiction → Resolution Engine, 4-State Funnel, Semantic Solution Clusters, and Expandable Thread Audit.
9. **Models & Data (`#/models`):** Pipeline Integrity Banner, Strict OOF Governance Audit, 4-Architecture Dumbbell Chart, BERTopic vs. NMF Table, DAPT Perplexity Curve, Calibration Reliability Plots, Inference Latency, and 4-Field Provenance Vault.
10. **Compare & Lab (`#/compare`):** Cross-IIM Comparison, Cross-Language Comparison, and Live Multilingual Analyzer.

---

## 6. Automated Test Suite

- BERTopic pipeline, SafeTensors round-trip, reproducibility, outliers, NMF preservation & `/api/demo/topics`: `backend/tests/test_bertopic_pipeline.py`
- Multilingual LID/spans, Romanized normalization, context-aware slang/emoji disambiguation, Phase 4 negation & ranking, Phase 5 all-5-scenario resolution: `backend/tests/test_phase4_phase5_linking_resolution.py`
- Full offline regression suite (`2,200+` tests):
  ```powershell
  $env:PYTHONPATH="D:\AI_Project\backend"
  D:\AI_Project\backend\.venv\Scripts\pytest.exe D:\AI_Project\backend\ --ignore=D:\AI_Project\backend\test_migrations_postgres.py -q
  ```
- Slim Vercel zero-ML runtime audit verifying all 10 core endpoints return HTTP 200 when `numpy`, `sklearn`, `torch`, `transformers`, `bertopic`, `umap`, `hdbscan`, `sentence_transformers`, and `langdetect` are blocked.
