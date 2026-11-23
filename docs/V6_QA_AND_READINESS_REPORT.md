# Reddit Community Intelligence (RCI) — V6 QA & Presentation Readiness Report

**Date:** 2026-11-23  
**Workspace:** `D:\AI_Project` (Branch: `main`, uncommitted local V6 programme changes preserved; zero commits, zero pushes, zero deployments, zero DB migrations, zero `.env` edits, zero external Reddit/Arctic Shift requests)

---

## 1. End-to-End Verification Matrix

| Verification Gate | Status | Verified Result & Evidence |
|---|---|---|
| **Preservation Guardrails** | **PASS** | `ml/artifacts/live_corpus/topic_discovery.json` (NMF v1, 10 factors, 3,459 docs) 100% untouched; V5 historical scores preserved; canonical 6-class response order untouched; `.env` & DB migrations untouched; zero git commit/push/deploy performed. |
| **Phase 1 — BERTopic Unit, Reproducibility, Outliers & API** | **PASS** | `backend/tests/test_bertopic_pipeline.py` (`3/3` passed): SafeTensors fit/save/load/transform round-trip, identical-seed reproducibility, explicit `-1` outlier preservation (`1,320` docs), `registry.json` (`nmf_v1` default + `bertopic_v1` companion), backward-compatible `/api/demo/topics?model=nmf\|bertopic`. |
| **Phase 2–5 Automated Unit & Contract Suite** | **PASS** | `backend/tests/test_phase4_phase5_linking_resolution.py` (`5/5` passed): Span-level LID & Bodh Gaya guard, non-destructive Romanized normalization, context-aware `mid`/`based`/`cooked` disambiguation + affect emojis, Phase 4 negation/placeholder penalties & ranking evaluator, Phase 5 all-5 resolution scenarios (`proposed_only`, `reported_success`, `reported_pending`, `contradictory`, `insufficient_evidence`). |
| **Full Offline Regression Suite** | **PASS** | `pytest backend/ --ignore=backend/test_migrations_postgres.py -q` → **`2,204 passed, 0 failed, 973 subtests passed`** in `42.97s`. |
| **Slim Vercel Zero-ML Runtime Audit** | **PASS** | Simulated Vercel `requirements.txt` (`fastapi` + `pydantic` only) with `numpy`, `scipy`, `sklearn`, `torch`, `transformers`, `bertopic`, `umap`, `hdbscan`, `sentence_transformers`, and `langdetect` blocked → **`10/10` core `/api/demo/*` endpoints returned HTTP 200**. |
| **Static ES Module Import/Export Audit** | **PASS** | All `backend/app/static/js/*.js` and `views/*.js` cross-checked → **`0` missing exports**. |
| **Docker Build & Local Stack Health** | **PASS** | `docker compose up -d --build web` succeeded cleanly (`rci-postgres Healthy`, `rci-web Started` on `http://localhost:8000/app`). |
| **Live Browser 9-View Visual & Interactive Audit** | **PASS** | Verified on `http://localhost:8000/app` across all 9 views (`Overview`, `Topic Map` NMF↔BERTopic + Outliers `-1` + Comparison, `Issues`, `Discussions`, `Thread Audit`, `Sentiment`, `Responses`, `Resolution`, `Models & Data`) + `768px` responsive viewport: **`0` console errors, `0` failed network requests (`35/35` HTTP 200)**. |

---

## 2. Quantitative Benchmark Summary Across All V6 Phases

| Phase | Deliverable / Engine | Real-Data Benchmark | Verified Headline Metric |
|---|---|---|---|
| **Phase 0** | Pre-Implementation Audit | 17 assignment criteria | [`docs/V6_PRE_IMPLEMENTATION_AUDIT.md`](file:///D:/AI_Project/docs/V6_PRE_IMPLEMENTATION_AUDIT.md) |
| **Phase 1** | BERTopic v1 alongside NMF v1 | `3,459` identical corpus docs | `35` natural clusters, Diversity `0.9629`, Balance `0.9509`, `38.2%` explicit `-1` outliers, `80%` subsample NMI `0.6251`; E5-small `100%` on 14 synthetic cross-lingual probes (vs. MuRIL-DAPT mean-pool anisotropic 2-cluster collapse). |
| **Phase 2** | Span-Level LID & Romanization | `n = 84` manually reviewed real comments | Accuracy **`88.10%`**, Macro F1 **`85.29%`**, Selective Acc (confident) **`95.83%`**, Script Acc **`100.0%`**, Code-Mix F1 **`95.52%`**, Multi-Span F1 **`90.91%`**. (`n = 24` synthetic Bengali/Odia/Devanagari probes `100%` reported strictly separately). |
| **Phase 3** | 5-Task Slang & Emoji Ablation | `n = 367` 5-fold GroupKFold OOF (`100` MuRIL fine-tunes + `20` TF-IDF) | Context-aware disambiguation preserved `24` literal uses (`based on`, `mid sem`) on corpus. **TF-IDF:** `original` best (`50.08%`, normalization `−0.46 to −1.00 pp`). **Neural MuRIL-DAPT-WWM:** `slang_normalized` **`+1.01 pp`** (`Solution +6.08 pp`), `slang_emoji_semantic` **`+0.90 pp`** (`Sentiment +2.22 pp`, `Issue +1.35 pp`), while `Response` (`−1.33 pp`) prefers raw text. |
| **Phase 4** | Contextual E5 Issue–Solution Mapper | `n = 64` manually graded real pairs (`8` issues) | `951` corpus links (`268` penalized) → `282` semantic clusters. Pairwise F1 **`86.57%`** (`+6.57 pp` vs. V5 TF-IDF `80.00%`), **`P@3 = 100.0%`**, **`MRR = 1.00`**, FPR **`13.33%`** (`−3.34 pp`). |
| **Phase 5** | Explainable Unresolved-Issue Engine | `n = 40` manually reviewed real threads + `365` corpus | 4-state Acc **`80.00%`**, Macro F1 **`82.52%`**, **False Resolution Claims `0.00%` (`0/33`)**, Unresolved F1 **`98.51%`**, Evidence Retrieval **`87.50%`**, `26` contradictory threads detected. |
| **Phase 6** | Strict OOF Evaluation Audit | `n = 92` frozen real test (`37` threads, `0/44` DAPT overlap) | Reports strictly unbiased OOF-selected headlines: **`64.62%` 3-seed mean** / **`65.01%` V6.0 pure neural** / **`64.38%` strict Stage 3** vs. V5 `59.48%`; explicitly flags `65.59%` V6.1 as post-hoc test-inspected. |
| **Phase 7** | Evidence-Linked Cross-Model Insights | `365` threads / `3,557` comments | `6` cards separating `observation` vs. `interpretation` with clickable `supporting_posts`, quoted `supporting_comments`, NMF+BERTopic `topic_refs`, confidence, and `evidence_limitations`. |
| **Phase 8 & 9** | 9-View Integration & Academic Package | 7 docs in `docs/academic_submission/` | All 9 Studio views integrated & verified; Deliverables 1–6 consolidated (`DELIVERABLE_1` explicitly notes MP4 recording pending). |

---

## 3. Presentation Readiness Checklist & Remaining Human Actions

- **Ready Now:**
  - Live interactive local studio at `http://localhost:8000/app` (`rci-web` + `rci-postgres` healthy).
  - Complete academic submission package under [`docs/academic_submission/`](file:///D:/AI_Project/docs/academic_submission/).
  - All `2,204` automated offline tests and Vercel zero-ML compatibility verified green.
- **Explicit Pending Human Actions (Intentionally Not Automated):**
  1. **Record the 3-minute scenario video** using the script and shot list in [`DELIVERABLE_1_VIDEO_PACKAGE.md`](file:///D:/AI_Project/docs/academic_submission/DELIVERABLE_1_VIDEO_PACKAGE.md).
  2. **Review & optionally refine BERTopic machine-suggested labels** in [`ml/artifacts/topic_models/bertopic_v1/human_review_sample.csv`](file:///D:/AI_Project/ml/artifacts/topic_models/bertopic_v1/human_review_sample.csv).
  3. **Review local changes (`git diff` / `git status`) before deciding whether to commit, push, or deploy** (held back per Part 2 constraints).
