# Evidence-Linked Cross-Model Community Insight Engine (V6 — Phase 7)

> **Core Objective:** Transform RCI from disconnected NLP charts into an **evidence-linked research intelligence engine** where every insight synthesizes topics, issues, sentiment, emotions, response roles, contextual solutions, and resolution outcomes — and attaches traceable post IDs, quoted comments, calibrated confidence, separated observations vs. interpretations, and explicit evidence limitations.

---

## 1. Architecture & Runtime Compatibility

Implemented in pure Python stdlib (zero heavy ML dependencies; runs identically in offline QA, Docker, and Vercel serverless) inside [`backend/app/services/nlp/cross_model_insights.py`](file:///D:/AI_Project/backend/app/services/nlp/cross_model_insights.py) (`engine_version: "2.0.0"`), served via `GET /api/demo/insights` and `GET /api/demo/summary`.

### Inputs Connected
1. **Dual Topic Discovery:** Preserved `NMF v1` (`10` factors) + Companion `BERTopic v1` (`35` semantic multilingual E5 clusters + `-1` outliers).
2. **Five Supervised NLP Tasks:** Issue (8 domains), Sentiment (3 classes), Emotion (6 multi-label), Response (6 canonical categories), Solution (binary).
3. **Phase 4 Contextual Issue–Solution Mapping:** `951` linked solutions across `282` semantic solution clusters (`issue_solution_map_v2.json`).
4. **Phase 5 Explainable Unresolved-Issue Detection:** 4 distinct resolution states + `26` threads with explicit contradictory/failed follow-ups (`resolution_engine_v2.json`).
5. **Phase 6 Strict OOF Governance Audit:** Unbiased strict-OOF benchmarks (`64.62%` 3-seed / `65.01%` V6.0 pure neural) with explicit disclosure of post-hoc Stage 3 bias.

---

## 2. Mandatory Evidence Contract (Every Insight Card)

Every card returned by `build_cross_model_insights()` satisfies the strict evidence schema:

| Field | Purpose & Integrity Guardrail |
|---|---|
| `observation` | Strictly factual corpus counts, shares, or cross-tabulations derived from the active dataset artifacts — zero speculative wording or causal claims. |
| `interpretation` | Clearly demarcated cautious analytical synthesis explaining *why* the observation matters without presenting model predictions as verified facts. |
| `supporting_posts` | Up to 4 clickable Reddit thread references (`post_id`, `post_title`, `subreddit`, `issue`, `resolution_state`) linking directly to `#/thread/<post_id>`. |
| `supporting_comments` | Up to 3 actual quoted anonymized comments (`comment_id`, `post_id`, `text`, `language`, `script`, `sentiment`, `response_type`, `confidence`) supporting the slice. |
| `topic_refs` | Explicit namespaced NMF (`nmf:v1:<id>`) and BERTopic (`bertopic:bertopic_v1:<id>`) topic references + top keywords (never conflated). |
| `issue_refs` | Linked canonical supervised issue domain keys. |
| `model_confidence` | Mean calibrated model confidence on the exact slice backing the insight. |
| `evidence_limitations` | Explicit domain/statistical caveat (e.g., observation-window censoring on `proposed_only`, rare-class bootstrap CI widths, non-causality). |

---

## 3. Six Evidence-Linked Insight Briefings Synthesized

1. **Friction Hotspot (`Issue × Sentiment × Emotion × Topics × Solutions × Resolution`):** Surfaces the highest-friction issue domain (`administrative_communication` / `placements`), pairing negative share, dominant emotion, NMF + BERTopic refs, Phase 4 matched solution count, quoted negative replies, and clickable thread drill-downs.
2. **Resolution & Evidence Gap (`Response × Solution × Phase 4 Links × Phase 5 Contradictory Follow-Ups`):** Contrasts `951` contextually matched peer solutions against only `12` explicit `reported_success` threads and `26` threads with contradictory/failed follow-ups (e.g., unanswered admissions emails/calls).
3. **Emotion × Response Role Dynamics:** Quantifies exact co-occurrence between `curiosity`/`frustration` and actionable `advice`/`solution` roles with explicit non-causal framing and quoted examples.
4. **Strict OOF-Selection 4-Model Benchmark & Multilingual Signal:** Reports the unbiased strict-OOF 3-seed (`64.62%`) and V6.0 (`65.01%`) benchmarks against LogReg/ANN/TextCNN and historical V5 (`59.48%`), linked to real Hinglish/code-mixed evidence comments (`10.1%` of corpus).
5. **NMF vs. BERTopic Semantic Discovery:** Compares `35` natural BERTopic themes (`CIBIL/loans`, `refund deadlines`, `B-school comparisons`, `internships`) + `38.2%` explicit `-1` outliers alongside NMF's `10` preserved baseline factors.
6. **Confidence & Human-in-the-Loop Triage Queue:** Surfaces high-confidence shares alongside low-confidence comments (`needs_review`) linking directly to `#/explorer?confidence=low` for human audit.

---

## 4. Application Integration (No Second Disconnected Dashboard)

The upgraded **Evidence-Linked Insight Explorer** is embedded directly inside the existing **Overview (`#/overview`)** studio view — every insight card expands to reveal separated **Observation vs. Interpretation**, clickable **Supporting Discussions (`#/thread/<post_id>`)**, **Quoted Evidence Comments**, **NMF/BERTopic Topic Badges**, **Calibrated Confidence**, and **Evidence Limitations**.
