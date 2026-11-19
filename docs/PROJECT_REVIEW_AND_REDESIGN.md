# Reddit Community Intelligence (RCI) — Full Project Review, Redesign Specification & Verification Report

**Repository:** `D:\AI_Project`
**Application Entry Point:** `http://127.0.0.1:8000/app/` (API namespace: `/api/demo`)
**Review Scope:** Complete technical, product, ML evaluation, and UX/UI audit of the existing RCI repository, followed by full frontend redesign architecture, design system specification, implementation, and verification.
**Dataset Integrity Notice:** All live demo views, API responses, predictions, and evaluation metrics inspected in this document operate strictly on the isolated synthetic pilot dataset (`data/demo_synthetic/`, `ml/artifacts/demo_synthetic/`, seed `20261116`). **SYNTHETIC DEMO DATA — NOT REAL REDDIT COMMUNITY DATA.**

---

## A. Executive Assessment

### What the project currently does

**Reddit Community Intelligence (RCI)** is an end-to-end research intelligence platform and NLP pipeline designed around the analytical sequence:

> **ISSUE → DISCUSSION → COMMUNITY RESPONSE → SOLUTION → RESOLUTION EVIDENCE**

Originating from a multilingual higher-education community analysis brief (*"AI-Powered Discovery of Student Issues and Solutions from Multilingual Reddit Discussions"*), the repository has matured well past a toy script into a structured, modular backend and ML training architecture with an isolated synthetic pilot demo:

- **Synthetic pilot corpus (`data/demo_synthetic/`):** Reproducibly generated (`backend/generate_synthetic_demo.py`, seed `20261116`) with **160 fictional posts** and **1,280 comments** (8 comments per thread), covering 8 scenario issue categories, the **6 canonical community response classes** (`agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`), 3 sentiments, 6 multi-label emotions, 4 explicit resolution states, and English (`en`, 457), Devanagari Hindi (`hi`, 529), and Latin-script Hinglish (`hinglish`, 294) comments with preserved `parent_reddit_id` relationships.
- **NLP & ML framework (`backend/app/services/nlp/`, `ml/`):** Implements language/script/code-mix detection, multilingual preprocessing, NMF topic discovery, multi-label issue classification, sentiment & emotion classification, 6-class response classification, binary solution detection, issue-solution linking, evidence-based resolution detection, and a thread-scoped evidence handoff adapter (`thread_evidence.py`). A unified `Trainer` (`ml/training/trainer.py`) supports thread-grouped (`post_id`) splits, leakage detection, artifact serialization, and model registry tracking across classical TF-IDF baselines and additional neural architectures (`response_mlp`, `text_cnn`, `muril`).
- **Isolated demo API (`backend/app/api/demo.py`):** Exposes 10 read-only, Pydantic-validated FastAPI endpoints under `/api/demo` serving only generated files from `data/demo_synthetic/` and `ml/artifacts/demo_synthetic/` without touching PostgreSQL or external services.
- **Existing local frontend (`backend/app/static/`):** A functional, dependency-free HTML/CSS/JS prototype (`index.html`, `styles.css`, `app.js`) with 7 views served directly by FastAPI at `/app/`.

### Strongest capabilities

1. **Rigorous research-integrity guardrails in the backend:** Explicit `synthetic_only: true` provenance flags across all API schemas, thread-grouped (`post_id`) splits with zero cross-split leakage, isolated artifact directories (`ml/artifacts/demo_synthetic/`), and strict preservation of the six canonical response categories and their order.
2. **Rich, well-typed backend payloads:** Endpoints already expose far more analytical depth than the original UI displays—including per-comment probability distributions across all 5 ML tasks, language/script/code-mix metadata, NMF keyword weights, 3 response examples per class, full proposal/advice text arrays on `/resolution`, and held-out validation + test reports with warnings on `/models`.
3. **Strong offline test discipline:** 38 test modules covering ingestion contracts, NLP modules, ML splitting/evaluation, Alembic migration safety, and synthetic demo API validation.

### Key limitations and overall maturity

- **Overall maturity:** *Backend & ML Pipeline: Late Alpha / Research Prototype (strong engineering hygiene).* *Frontend Product Experience (pre-redesign): Early Prototype with P0 correctness and usability defects.* *Data Readiness: Synthetic Pilot Only (no authorized real Reddit data or production DB-backed analytical API).*
- **Critical honesty gaps in pre-redesign UI:** Several frontend and artifact-reporting defects actively distort the underlying data—most notably displaying **`0.0%` Test subset accuracy** for the Emotion and Issue models when the metric is actually `null` (not emitted), misaligning an 8-column Issues table under 6 headers, rendering the Monthly Sentiment chart completely invisible due to missing CSS classes, and printing all-zero per-class metrics for the Solution baseline without explanation.
- **Unimplemented production scope:** The `/api/v1/*` authenticated admin/pipeline surface (`docs/API_SURFACE_PROPOSAL.md`) remains unimplemented; Bengali, Odia, Banglish, Romanized Odia, emoji semantics, and Gen-Z slang from the academic brief are absent from the synthetic pilot; and real-world human-annotated datasets are not yet collected or benchmarked.

---

## B. Technical Review

### 1. Architecture & Code Organization

- **Clean separation of concerns in the backend:** FastAPI entry (`backend/app/main.py`) separates liveness (`/health`), database health (`/health/db`), static frontend mounting (`/app/`, `/assets`), and demo routes (`/api/demo`). Core NLP modules live cleanly under `backend/app/services/nlp/<stage>/`, and generic ML orchestration lives under `ml/`.
- **Empty `frontend/` directory:** The top-level `frontend/` folder contains no files, `package.json`, or build tooling. The live application is served exclusively from `backend/app/static/`.
- **Architectural decision — Native ES Modules vs. React Migration:**
  - *Decision:* **Retain a zero-dependency native ES Module architecture inside `backend/app/static/` (`index.html`, `styles.css`, `app.js`, and modular `js/` + `js/views/` files), rather than introducing React/Vite/npm.**
  - *Justification:* (1) No `package.json` or frontend toolchain exists in the repository; introducing React/Vite adds build-step friction, `node_modules` bloat, and supply-chain dependency risk in an offline-first academic project. (2) FastAPI's `StaticFiles` mounts `/assets` natively with zero build configuration, so `uvicorn app.main:app` immediately serves modular ES files cleanly. (3) The application's state graph is straightforward (read-only analytical views, hash routing, cached JSON endpoints, client-side filtering/sorting, and crisp custom inline SVG charts) where modular ES6 + semantic HTML + CSS custom properties delivers instant load times, zero runtime overhead, and superior maintainability without rewriting working infrastructure.

### 2. Backend API Design (`backend/app/api/demo.py`)

- **Strengths:** Clean Pydantic v2 response models (`HealthResponse`, `SummaryResponse`, `DiscussionListResponse`, `DiscussionDetailResponse`, `ItemsResponse`, `SentimentResponse`, `ModelsResponse`); strict filter validation returning HTTP `422` for unknown values; bounded pagination (`page_size` 1–100); `@lru_cache(maxsize=1)` in-memory loading.
- **Verified minor backend defects (documented, preserved to avoid breaking contracts):**
  1. `backend/app/api/demo.py#L299-L302`: Contains a dead dictionary comprehension (`for task in ()`) and an `O(N × M)` linear search (`next((r for r in data["comments"] ...))`) inside `discussion_detail()`.
  2. `backend/app/api/demo.py#L188`: `_summary_data()` checks `item.get("topic_id")` on `topic_assignments` instead of `item.get("dominant_topic_id")` (compare `#L322`). It accidentally succeeds only because `assigned or topic.get("document_count", 0)` falls back to the pre-computed `document_count`.
  3. `backend/app/api/demo.py#L254`: Validates the `emotion` query parameter against generator gold labels (`row["emotions"]`) while filtering against model predictions (`r["predictions"]["emotion"]["emotions"]`). Works cleanly on the current corpus because all 6 emotion labels appear in both, but could diverge on noisier datasets.
  4. `backend/app/api/demo.py#L234`: `/discussions` does not accept a `resolution_state` query parameter (though `resolution_state` is returned in every discussion item and filterable on `/resolution`).

### 3. ML Integration, Evaluation Anomalies & Data Integrity

#### Verified Root Cause #1 — Emotion & Issue "0.0% Subset Accuracy" vs. 98.6% Macro F1
Tracing `backend/app/services/nlp/sentiment_emotion/models/evaluation.py` → `ml/evaluation/reports.py` → `ml/artifacts/demo_synthetic/evaluation.json` → `backend/app/static/app.js` establishes the exact two-part cause:
1. **Evaluator omission (`ml/evaluation/reports.py#L173-L186`):** Although `EmotionEvaluationReport` computes `subset_accuracy`, `hamming_accuracy`, and `hamming_loss` (defined in `sentiment_emotion/models/evaluation.py#L382-L384`), the unified report extractor `_extract_metrics()` for `is_multilabel=True` extracts only `per_label`, `macro_f1`, and `micro_f1`. Consequently, `ml/artifacts/demo_synthetic/evaluation.json` serializes `"accuracy": null`, `"subset_accuracy": null`, `"hamming_loss": null`, and `"hamming_accuracy": null` for both `emotion` (`macro_f1: 0.985714`, `micro_f1: 0.995582`) and `issue` (`macro_f1: 0.585376`, `micro_f1: 0.577778`).
2. **Frontend null-to-zero coercion (`backend/app/static/app.js#L31` & `#L290`):** The helper `percent = (value) => Number.isFinite(Number(value)) ? ... : "Not reported"` fails on `null` because JavaScript evaluates `Number(null)` as `0`, and `Number.isFinite(0)` is `true`. Line 290 passes `metrics.accuracy` (which is `null`) under the label `"Test subset accuracy"`, producing a fabricated **`0.0%`** display.

#### Verified Root Cause #2 — Solution Baseline `per_class` (0% / Support 0) & Zero Confusion Matrix
- In `backend/train_demo_baselines.py#L60-L89`, `heldout_class_metrics()` normalizes gold and predicted binary solution values to `{"true"}` and `{"false"}`, then scores them against `trained_model.labels` (`["not_solution", "solution"]`). Because `"solution" not in {"true"}` and `"not_solution" not in {"false"}`, every cell in `per_class` and `confusion_matrix` evaluates to `0` (`support: 0`, `[[0, 0], [0, 0]]`) despite `test.metrics.accuracy = 1.0`, `macro_f1 = 1.0`, `precision = 1.0`, `recall = 1.0`, and `n_samples = 192`.
- Because existing model artifacts must not be overwritten, the redesigned UI detects this zero-support artifact anomaly programmatically and renders an honest methodological disclosure alongside the valid held-out aggregate metrics and dataset class balance (`False: 854`, `True: 426`).

#### Verified Data Generation & Unsupervised NMF Artifacts
- **Sequential Domain Timestamping (`backend/generate_synthetic_demo.py#L139-L146`):** Posts are generated domain-by-domain at 2-day intervals from `2025-01-03` (`placements`) to `2025-11-17` (`financial_concerns`). Consequently, `summary.issue_trends` (`+20` Financial Concerns, `+20` Administrative Communication, `-20` Extracurriculars, `-15` Hostel Facilities, `0` for earlier domains) is purely an artifact of sequential generation order—not real community momentum.
- **NMF Tokenization on Devanagari & Unfiltered Stopwords (`ml/artifacts/demo_synthetic/topic_discovery.json`):** Topics 0, 2, and 5 list single Devanagari consonants (`र`, `क`, `ए`, `न`, `प`, `त`, `द`, `ष`, `म`, `ह`) because default word-regex tokenization splits Unicode combining vowel signs (`\u093E–\u094D`) as word boundaries. Topic 1 (`592` comments / `46.2%`) clusters English function words (`the`, `that`, `for`, `would`), while Topics 3, 4, 6, and 7 capture language/script and response-role templates rather than substantive issue themes.
- **Minor Language-Tagging Bug (`backend/generate_synthetic_demo.py#L98-L103`):** `_language()` classifies any text containing a Devanagari character as `("hi", "devanagari", is_code_mixed=False, ["hi"])`, even when an English sentence receives a Devanagari sentiment cue (e.g., `syn-comment-0001-03`).
- **In-Sample Display Scope:** `predictions.json` contains predictions over all 1,280 comments (including the 896 training records), while `evaluation.json` reports metrics on the held-out 192-comment test split (`group_by="post_id"`, zero leakage). This distinction must stay explicit across every analytical view.

### 4. Testing & Maintainability

- `backend/tests/test_demo_api.py` and `backend/tests/test_synthetic_demo_integrity.py` thoroughly verify Pydantic contract validation, filter 422 responses, seed reproducibility, canonical response ordering, and artifact isolation.
- Pre-redesign frontend had zero modularization—all HTML rendering, state management, API fetching, and formatting lived in a single 344-line `app.js` file using innerHTML string concatenation without hash routing or reusable chart abstractions.

---

## C. Product Review

### Does the current application communicate its purpose clearly?

Partially. The pre-redesign UI gets the **epistemic posture right** (consistent synthetic warnings and an explicit 5-step evidence rail on Overview), which is a strong foundation. However, as an analytical research product for faculty and institutional stakeholders, it falls short in four core workflows:

1. **Broken `Issue → Discussion → Solution → Resolution` drilldown:** Clicking an issue on the Issues page jumps to Explorer without resetting pagination or clearing stale filters, and users cannot inspect an issue's sentiment profile, resolution breakdown, or linked proposals without manually hopping across four separate tabs.
2. **Overview fails to surface actionable variation:** Because every synthetic issue category contains exactly 20 posts, the Overview "Issue frequency" chart renders eight identical 100%-width horizontal bars—wasting prime viewport real estate while hiding the metrics that *actually vary* (actionable solution/advice density, sentiment balance, emotion prevalence, and 4-state resolution outcomes).
3. **Solutions & Resolution hides the actual solutions:** `/api/demo/resolution` returns every thread's proposed solution comments, advice comments, actionable recommendations, and per-comment solution model probabilities—yet the pre-redesign UI discarded all proposal and advice text and showed only bare counts (`2 scenario-labeled proposals`).
4. **Multilingual & thread-structure strengths are buried:** Comment-parent chains (depth 0 to 6) were flattened to a single indent level, and script/code-mix metadata (`devanagari` vs `latin`, `hinglish` code-mixing) and per-task confidence distributions were largely invisible.

---

## D. UX/UI Review

| Area | Pre-Redesign Findings & Specific Evidence |
| --- | --- |
| **Information Hierarchy** | Overview stacks two full-width header strips (evidence rail + 4 metric blocks) above identical-length issue bars and hides dataset provenance at the very bottom. Issues and Topics—two fundamentally different analytical artifacts (scenario labels vs. unsupervised NMF)—are jammed onto one long scroll page. |
| **Navigation** | 7 views in a dark navy sidebar using inconsistent Unicode glyphs (`◫`, `⌕`, `▤`, `◒`, `↳`, `◎`, `▥`) that render inconsistently across fonts/OSes. No URL hash routing (`window.location.hash`), so page reload or browser Back loses the current view/thread/filters. No global search shortcut. |
| **Visual Consistency** | Undefined CSS classes cause silent failures: `.stacked-segment.positive/neutral/negative` have no styles (invisible monthly chart), and `.status-tag.positive/neutral/negative` fall back to uncolored gray tags on thread comments. |
| **Layout & Data Density** | Desktop uses wide cards with sparse horizontal space in some views (Responses shows only 1 of 3 examples per card in a 2-column grid) while cramping the 8-column Issues table under 6 headers. `resolution-mobile.png` shows the intro sentence colliding directly against `160 synthetic threads` without spacing (`"proposal only."160 synthetic threads`). |
| **Typography** | Clean system sans-serif and tabular numerals on primary metrics, but weak distinction between technical identifiers (`syn-post-0008`, `placements`, `reported_success`) and editorial prose—no monospace treatment for IDs/keys and tiny `10px` table headers/metadata that strain legibility. |
| **Charts & Visualizations** | Div-width bars without axis ticks, percentage shares, legends on several charts, or hover tooltips. Monthly sentiment stacked chart is completely blank. NMF `keyword_weights` and `document_ratio` are not visualized at all. |
| **Filtering & Searching** | Explorer requires pressing "Apply filters" with no "Clear filters" button, no removable filter pills, no active-filter count, and no `page=1` reset when jumping from Issues. Resolution filtering hides rows via DOM `.hidden` without an empty-state message when 0 rows match! |
| **Empty, Loading & Error States** | Plain text single-line loading banner (`Loading generated results…`), no skeleton structure, and Resolution view shows an empty white table header with no guidance when filters match zero threads. |
| **Responsive Behaviour** | Below `760px`, the sidebar collapses into a horizontally scrolling tab strip that clips labels mid-word (`Sentimen|` in `overview-mobile.png`), hides the dataset stamp completely, and leaves wide tables cramped on `390px` screens. |
| **Accessibility** | Good baseline (`skip-link`, `:focus-visible`, `aria-current="page"`, `aria-live="polite"`, `44px` target heights on buttons). Gaps: non-descriptive icon glyphs, no `aria-expanded` mobile drawer, `10px` body/table metadata below comfortable reading size, `<button>` wrapping multi-line block cards without clear focus boundaries, and missing `scope="col"` on most tables. |

---

## E. Prioritized Improvement Roadmap

### P0 — Critical (Implemented in This Pass)
1. **P0-1: Fix `null` → `0.0%` coercion on Model Evaluation (`percent()` & multilabel metrics)**
   - *Problem:* `percent(null)` rendered `Test subset accuracy: 0.0%` for Emotion (`98.6%` macro F1) and Issue (`58.5%` macro F1).
   - *Impact:* Misleads researchers into thinking multilabel exact-match accuracy is zero.
   - *Solution:* Strictly check `value === null || value === undefined || value === ""` in `js/format.js`; display `Not emitted` for `subset_accuracy` with an explanatory tooltip referencing `ml/evaluation/reports.py`, highlight `macro_f1` & `micro_f1`, and render evaluator warnings verbatim.
2. **P0-2: Fix 8-cell vs. 6-header misalignment in Issues table**
   - *Problem:* Issues table rendered 8 `<td>` cells under 6 `<th>` headers, shifting advice/recommendation/resolution columns.
   - *Impact:* Corrupted numerical attribution across all 8 issue categories.
   - *Solution:* Give Issues Explorer its own dedicated view with a properly structured, sortable 8-column evidence table (`Issue Category`, `Posts`, `Comments`, `Solution Comments`, `Advice Comments`, `Actionable Total`, `Reported Success`, `Unconfirmed Outcome`) plus inline issue inspection.
3. **P0-3: Fix invisible Monthly Sentiment stacked chart & unstyled comment sentiment badges**
   - *Problem:* Missing CSS fills for `.stacked-segment.*` and `.status-tag.positive/neutral/negative`.
   - *Impact:* Monthly timeline chart rendered blank; comment sentiments lacked semantic visual contrast.
   - *Solution:* Replace with semantic SVG charts using centralized palette tokens and dedicated positive/neutral/negative badge tokens.
4. **P0-4: Detect and honestly disclose Solution model zero-support `per_class` / `confusion_matrix` anomaly**
   - *Problem:* `evaluation.json` stores `support: 0` and `[[0,0],[0,0]]` for Solution due to `train_demo_baselines.py` boolean-vs-string label mismatch, while top-level `accuracy` & `macro_f1` are `1.0`.
   - *Impact:* Contradictory metrics destroy stakeholder trust.
   - *Solution:* Programmatically detect zero total support when `n_samples > 0` and display a clear methodological callout documenting the artifact serialization issue without modifying frozen model artifacts.
5. **P0-5: Fix Issue-to-Explorer pagination bug, add filter reset & zero-match empty states**
   - *Problem:* Cross-view issue drilldown retained stale `explorerPage` and prior filters; Explorer and Resolution lacked clear-filter controls and empty states.
   - *Impact:* Broken core research journey (`Issue → Discussions`).
   - *Solution:* Reset pagination to page 1 on every filter/drilldown change, add active removable filter chips + "Clear all filters", and render informative empty states everywhere.

### P1 — High (Implemented in This Pass)
1. **P1-1: Upgrade Information Architecture to 8 cohesive research views** (`Overview`, `Issues Explorer`, `Discussion Explorer` + `Discussion Detail`, `Topic Intelligence`, `Sentiment & Emotion`, `Community Responses`, `Solutions & Resolution`, `Models & Dataset`) with grouped navigation (`Discovery`, `Analysis`, `Provenance & ML`).
2. **P1-2: Unlock under-utilized API payloads** — Render thread 6-role composition mini-bars on Explorer cards, full 5-task probability distributions + language/script/code-mix tags + Scenario-vs-Prediction comparison on Discussion Detail, NMF `keyword_weights` & `document_ratio` charts on Topic Intelligence, all 3 examples per response class on Community Responses, expandable proposal/advice texts & model probabilities on Solutions & Resolution, and Validation-vs-Test comparisons + warnings on Models & Dataset.
3. **P1-3: True hierarchical parent-child thread view** — Render progressive bounded reply nesting with vertical thread connectors, clickable parent-jump chips (`↳ Replying to #02`), flat vs. threaded toggle, in-thread filters (by role, sentiment, solution, language), and clickable Evidence Jump from sidebar to exact resolution comment.
4. **P1-4: Transparent methodological disclosures for synthetic artifacts** — Explicitly explain sequential domain timestamping on temporal views, Devanagari regex-tokenization & stopword behavior on NMF Topic Intelligence, and in-sample display vs. held-out test evaluation across all views.
5. **P1-5: Hash-based URL routing & global quick search** — Support deep links (`#/overview`, `#/issues`, `#/explorer?issue=...`, `#/thread/syn-post-0001`, etc.), browser Back/Forward history, and topbar `/` quick search.
6. **P1-6: Redesign Overview for genuine analytical hierarchy** — Replace 8 identical post-count bars with an **Issue × Actionable Recommendations & Resolution Status** analytical breakdown, plus Sentiment, Emotion, Canonical 6-Class Response composition, and interactive Evidence Chain navigation.
7. **P1-7: Responsive slide-over mobile drawer & accessible data layouts** — Replace clipped mobile tab strip with a proper accessible navigation drawer (`aria-expanded`, Escape-to-close, backdrop), responsive mobile table cards / clean scroll wrappers, and verified zero overflow at `1440px`, `1280px`, `768px`, and `390px`.

### P2 — Medium (Implemented in This Pass & Tracked)
1. **P2-1 (Implemented):** One-click copyable identifiers (`post_id`, `comment_id`, `dataset_id`, `registry_id`) with subtle toast feedback, plus interactive glossary tooltips for ML/research terms (`Macro F1`, `Micro F1`, `Subset Accuracy`, `Thread-Grouped Split`, `NMF`, `Proposed vs. Reported Outcome`).
2. **P2-2 (Implemented):** Client-side column sorting on Issues Explorer, Sentiment-by-Issue, and Solutions & Resolution tables, plus structured loading skeletons and retryable API error panels.
3. **P2-3 (Tracked for Backend Hygiene):** Remove dead loop in `backend/app/api/demo.py#L299-L302`, index comments by ID (`O(1)` lookup), fix `#L188` `dominant_topic_id` lookup, and fix mixed English+Devanagari `is_code_mixed` classification in `backend/generate_synthetic_demo.py#L98-L103`.

### P3 — Later (Post-Pilot / Real-Data Production Track)
1. **P3-1:** Acquire authorized real Reddit community corpus with ethics/privacy review and human multi-annotator labeling (with inter-annotator agreement reporting).
2. **P3-2:** Update `ml/evaluation/reports.py::_extract_metrics()` to propagate `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, and fix `backend/train_demo_baselines.py#L64-L65` boolean-to-label mapping prior to future retraining.
3. **P3-3:** Upgrade topic discovery tokenization for Unicode Indic scripts (Devanagari/Bengali/Odia grapheme clusters) and semantic multilingual embeddings (e.g., BERTopic / Sentence-BERT).
4. **P3-4:** Benchmark neural baselines (`response_mlp`, `text_cnn`, `muril`) on ≥300 manually annotated real comments and implement the authenticated `/api/v1/*` PostgreSQL-backed API (`docs/API_SURFACE_PROPOSAL.md`).
5. **P3-5:** Expand preprocessing and dataset coverage to Bengali, Odia, Banglish, Romanized Odia, emoji semantic tokens, and Gen-Z slang per the full research brief.

---

## F. Redesigned Information Architecture & Visual Design System

### 1. Information Architecture (8 Core Views + Thread Detail)

| Group | View | Hash Route | Primary Endpoints Consumed | Research Question Answered |
| --- | --- | --- | --- | --- |
| **Discovery** | **Overview** | `#/overview` | `/summary`, `/health` | What corpus is loaded, how does the 5-stage chain distribute across issues, sentiment, emotions, responses, and resolution evidence? |
| **Discovery** | **Issues Explorer** | `#/issues` | `/issues`, `/sentiment`, `/resolution`, `/discussions` | Which issue categories carry the highest actionable recommendation density, negative sentiment, or unconfirmed outcomes? |
| **Discovery** | **Discussion Explorer** | `#/explorer` | `/discussions`, `/summary` | Which threads match specific keywords, communities, issues, sentiments, emotions, or response roles? |
| **Discovery** | **Discussion Detail** | `#/thread/:id` | `/discussions/:id` | How does a thread unfold across parent-child replies, what are scenario labels vs. model predictions, and what resolution evidence exists? |
| **Discovery** | **Topic Intelligence** | `#/topics` | `/topics`, `/issues` | What weighted term clusters does unsupervised NMF extract, how much of the corpus do they cover, and what are their tokenization limits? |
| **Analysis** | **Sentiment & Emotion** | `#/sentiment` | `/sentiment`, `/summary` | How do predicted polarity (single-label sentiment) and affect (multi-label emotions) vary across issues and generated months? |
| **Analysis** | **Community Responses** | `#/responses` | `/responses`, `/summary` | How are comments distributed across the 6 fixed canonical response classes (`agreement` → `counterargument`), with concrete classified examples? |
| **Analysis** | **Solutions & Resolution** | `#/resolution` | `/resolution`, `/issues` | Which threads contain proposed solutions/advice vs. explicit first-person resolution outcomes (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`)? |
| **Provenance & ML** | **Models & Dataset** | `#/models` | `/models`, `/summary`, `/health` | What is the exact synthetic provenance, how do all 5 TF-IDF baselines perform on held-out thread-grouped validation/test splits, and what limitations apply? |

### 2. Visual Design System Specification (Hallmark **Dusk** — Smoked-Glass Interactive Research Studio)

Following a full from-scratch architecture pass (`theme.mjs deal --genre modern-minimal`, deal `3/3`, seed `accessible-visual-community-intelligence-980b·a2543c58·2026-11-19·r0`) and Hallmark static + visual audits (`sloplint.mjs --genre modern-minimal --scope app --kept`), the frontend was completely rebuilt from scratch on the **Dusk** design system (`dark / grotesk-sans / cool`), with a live **Dusk Midnight (dark default) ↔ Daylight Bone (light)** theme switcher:

- **Colour Palette (OKLCH CSS Custom Properties in `backend/app/static/tokens.css` — zero color literals in `styles.css`):**
  - *Dusk Midnight Ground & Smoked-Glass Surfaces:* `--color-paper` (`oklch(14% 0.015 265)`), `--color-paper-2` (`oklch(18.5% 0.018 265)`), `--color-paper-3` (`oklch(23.5% 0.020 265)`), `--color-surface` (`oklch(19.5% 0.018 265 / 0.78)` with `backdrop-filter: blur(18px)`), `--color-surface-solid` (`oklch(19% 0.018 265)`), `--color-inset` (`oklch(12.2% 0.014 265)`).
  - *Daylight Bone Light Mode (`:root[data-theme="light"]`):* Warm bone ground (`oklch(96.2% 0.008 80)`), elevated porcelain decks (`oklch(99.2% 0.004 80)`), and deep slate ink (`oklch(17% 0.018 265)`).
  - *Soft Fog Ink & Hairline Rules:* `--color-ink` (`oklch(93.5% 0.008 260)`), `--color-ink-2` (`oklch(82% 0.012 260)`), `--color-muted` (`oklch(70% 0.014 260)`, `>= 4.8:1` AAA/AA contrast on both `--color-paper` and `--color-surface-solid`, and `>= 3.4:1` non-text field border contrast), `--color-rule` (`1px` hairline).
  - *Bi-Tonal Accents:* Periwinkle primary `--color-accent` (`oklch(68% 0.145 268)`) and warm apricot secondary `--color-accent-2` (`oklch(77% 0.135 55)`).
  - *Harmonized OKLCH Semantic & Canonical 6-Class Response Palette (Fixed Order):* `agreement` (`hue 248`), `personal_experience` (`hue 292`), `opinion` (`muted fog hue 260`), `advice` (`cyan-teal hue 200`), `solution` (`mint-emerald hue 158`), `counterargument` (`apricot-coral hue 48`).
- **Typography (100% Local Offline-Safe Stack — Zero External CDNs):**
  - Display & Section Titles: `"Plus Jakarta Sans", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif` (`--display-weight: 500`, tight `-0.024em` tracking, sentence case, zero tracked uppercase eyebrows).
  - UI Body, Controls & Tabular Numerals: `"Inter", system-ui, -apple-system, "Segoe UI", sans-serif` with `font-variant-numeric: tabular-nums` on `.num`, `td.num`, `th.num`, and SVG chart readouts.
  - Technical Identifiers & Code Keys Only (`--font-outlier`): `"JetBrains Mono", "Cascadia Code", "Consolas", ui-monospace, monospace` restricted strictly to `code`, keyboard shortcut badges (`Ctrl K`, `Esc`), and canonical code keys.
- **Spacing, Geometry & Craft:** Strict 4px spacing scale (`--space-3xs` through `--space-4xl`), softly rounded Dusk surfaces (`--radius-card: 16px`, `--radius-inner: 12px`, `--radius-pill: 9999px`), single-layer card discipline (zero card-in-card nesting; inner collections use hairline-divided rows `.spotlight-row`, `.ruled-row`, `.tally-cell`, `.eval-cell`), single filled primary button recipe (`.btn-primary`), `>= 3:1` input field borders (`--color-muted`), outline focus rings (`outline-color: var(--color-focus)`), 850ms-hover / 0ms-focus glossary tooltips, and 44px touch targets under `@media (pointer: coarse), (max-width: 640px)`.
- **Navigation, Command Palette & Studio Shell:**
  - *Single Labeled `204px` Studio Rail (`>1023px`):* Always-visible icon + text labels (`Overview`, `Issues`, `Discussions`, `Topic map`, `Sentiment`, `Responses`, `Resolution`, `Models & data`), theme switcher (`Daylight mode` / `Dusk dark mode`), and synthetic dataset stamp (`v1.0 · Seed 20261116`).
  - *Top Command Bar & Synthetic Provenance Ribbon:* Breadcrumb trail, global `/` search input (`q` field preserved), `Ctrl+K` Command Palette trigger, clickable `Synthetic pilot · v1.0` provenance popover pill, and live API health badge (`Local API ready`, kept visible on mobile `390px`).
  - *Frosted-Glass `Ctrl+K` Command Palette Modal:* Instant keyboard-navigable jump across all 8 views, 8 issue domains, 6 canonical response roles, and live debounced `/api/demo/discussions` thread search.
  - *Master-Detail Discussion Workbench (`#/explorer`):* Left filterable thread stream with 6-role mini composition strips paired with a sticky right live preview pane featuring 3 interactive sub-tabs (`Evidence`, `All 8 replies`, `Model signals`).
- **Signature Interactive SVG Visualizations & Motion Engine (`js/charts.js`, `js/motion.js`):**
  1. **Interactive 3-Stage Sankey Research Flow (`#/overview`):** Maps `8 Issue Domains → 6 Canonical Response Classes → 4 Resolution States` with opaque Column-2 pill backings (zero ribbon-behind-text collision), hover path highlighting, floating glass tooltip, keyboard navigation, and click-to-drilldown.
  2. **Interactive Multi-Series Smooth Spline Area Chart (`#/overview`, `#/sentiment`):** Cubic Catmull-Rom / Bezier polarity curves (`Positive`, `Neutral`, `Negative`) across Jan–Nov 2025 with vertical SVG gradients, animated stroke draw-in, series filter pills, and a live hovering vertical crosshair + glowing point markers + monthly breakdown tooltip.
  3. **Interactive 8-Orb NMF Topic Constellation (`#/topics`):** Radial cluster map sized by comment share and colored by script character (`English stopwords`, `Hinglish phrasing`, `Devanagari fragments`) with click-to-inspect weighted NMF terms.
  4. **Interactive 6-Segment Radial Canonical Response Wheel (`#/responses`):** Hover-expanding SVG donut arcs in exact canonical 1–6 order with live center readout and 3-example tab switchers per category.
  5. **Dusk Ruled Tally Strip (`tallyStrip`) & GPU-Safe Motion Hydration (`js/motion.js`):** Single elevated surface parted by `1px` hairlines with `requestAnimationFrame` cubic-out number count-ups, 36px inline SVG ratio rings on percentage metrics, and automatic `prefers-reduced-motion: reduce` compliance.
- **Responsive Breakpoints:**
  - `≥1280px` (Desktop `1440px` / Laptop `1280px`): Full `204px` labeled studio rail, asymmetric 2-column analytical decks, 4-column ruled tally strip, and Master-Detail Discussion Workbench.
  - `768px–1023px` (Tablet `768px`): Slide-over navigation drawer, fluid full-width Sankey SVG (`min-width: 0`), unclipped `256px` search bar, and 2-column analytical grids.
  - `≤640px` (Mobile `390px`): Slide-over drawer + persistent 5-tab mobile bottom dock (`Overview`, `Discussions`, `Sentiment`, `Resolution`, `More`), visible `Local API ready` status pill, `44px` touch targets, and zero horizontal page overflow.

---

## G. Verification, Visual Evidence & Final Assessment

### 1. Implemented Frontend Architecture

All frontend work was implemented from scratch directly inside `backend/app/static/` with zero external runtime dependencies, zero CDN/font requests, and full compatibility with FastAPI's existing `/app/` and `/assets` static mounts:

- `backend/app/static/tokens.css` & `backend/app/static/base.css` — Hallmark Dusk OKLCH design tokens (`:root` Dusk Midnight + `:root[data-theme="light"]` Daylight Bone), typography registers, focus/selection/scrollbar theming, and base reset.
- `backend/app/static/index.html` — Semantic Dusk studio shell with `204px` labeled navigation rail (all 9 links preserved), command bar with breadcrumb, `/` quick search (`q` field preserved), `Ctrl+K` Command Palette modal, synthetic provenance ribbon & popover, mobile slide-over drawer + bottom dock, inline SVG favicon, and ARIA live toast region.
- `backend/app/static/styles.css` — Dusk smoked-glass decks, single-surface hairline-parted `.tally-strip` and `.resolution-state-grid`, flat hairline-divided `.spotlight-row` / `.ruled-row` lists (zero card-in-card nesting), interactive SVG chart styles, Master-Detail Discussion Workbench, hierarchical comment tree, token-driven confusion matrix heatmap, and responsive rules (`1440px`, `1280px`, `768px`, `390px`).
- `backend/app/static/app.js` — Hash router (`#/overview`, `#/issues`, `#/explorer`, `#/thread/:id`, `#/topics`, `#/sentiment`, `#/responses`, `#/resolution`, `#/models`), live API hydration, `Ctrl+K` Command Palette controller, Dusk Midnight ↔ Daylight Bone theme switcher (`localStorage`), and automatic motion hydration.
- `backend/app/static/js/motion.js` — GPU-safe `requestAnimationFrame` count-up animator, SVG ring/path draw-in controller, and floating glass chart tooltip positioning (`prefers-reduced-motion` aware).
- `backend/app/static/js/api.js`, `format.js`, `icons.js`, `charts.js`, `components.js` — Cached API client, strict null-safe `percent()` formatter, canonical 6-class taxonomy constants, inline SVG icons, interactive SVG visualizations (Sankey, smooth spline with crosshair, NMF constellation, radial response wheel, benchmark bars), and Dusk UI primitives (`tallyStrip`, `deck`, `callout`, `glossaryTip`, `copyableId`).
- `backend/app/static/js/views/` — Nine focused view modules: `overview.js`, `issues.js`, `explorer.js`, `thread.js`, `topics.js`, `sentiment.js`, `responses.js`, `resolution.js`, `models.js`.

### 2. Test & Quality Verification Report

| Suite / Check | Scope | Result | Details |
| --- | --- | --- | --- |
| **Hallmark Static Craft Sweep (`sloplint.mjs`)** | `backend/app/static/` (`--genre modern-minimal --scope app --kept`) | **0 FAIL, 0 REFLEX, 0 WARN** (down from `56 FAIL, 172 REFLEX, 8 WARN`) | Preserved `links 9/9`, `fields 1/1`, zero color/font literals outside `tokens.css`, strict 4px scale, zero eyebrows or tracked caps, single primary button recipe, tabular numerals on all figures |
| **Fresh-Context Hallmark Dusk Finish Review** | Independent subagent review against `dusk.md`, `dusk.jpg`, `gates.md`, before/after screenshots, and CSS/HTML | **8 / 8 findings resolved** (verified in scoring round) | Added opaque Column-2 Sankey label pills + `>= 12px` SVG text + fluid tablet Sankey, eliminated all card-in-card nesting across views, raised input borders to `>= 3:1` (`--color-muted`) + added palette `:focus-visible` ring, replaced 4-tile KPI boxes with Dusk's hairline-parted `.tally-strip`, unified desktop nav into a single labeled `204px` rail + kept mobile `.api-pill` visible, restricted `--font-outlier` strictly to `code`/kbd badges, removed `.preview-inner` re-entry fade, and removed body gradient / spring overshoot |
| **Backend Pytest Suite (Focused)** | `tests/test_demo_api.py`, `tests/test_synthetic_demo_integrity.py`, `test_app_startup.py` | **19 passed, 0 failed** | Pydantic contracts, filter 422 validation, seed reproducibility, canonical response taxonomy, isolated artifact checks, app startup (zero backend `.py` or artifact files modified) |
| **Frontend JS Syntax Check** | All 15 ES modules under `backend/app/static/` via `node --check` | **15 / 15 passed** | Zero syntax errors |
| **Format & Canonical Unit Assertions** | `js/format.js` (`percent(null)`, `percent(0)`, `CANONICAL_RESPONSE_TYPES`) | **All assertions passed** | Verified `percent(null) === "Not emitted"` and exact 6-class canonical order |
| **Live Headless Chrome CDP E2E Suite** | Real FastAPI server (`http://127.0.0.1:8000/app/`, reused pre-existing instance) | **16 / 16 functional checks passed** | Verified Dusk Midnight Overview + 4-cell tally strip + 18-node Sankey + 3-series spline, `Ctrl+K` Command Palette modal, Dusk Midnight ↔ Daylight Bone theme switcher, Issues 8 interactive pods + 9=9 table toggle, Discussion Workbench split preview + 3 live tabs + filtered pagination, Conversation Audit 8 comments / 6 nested links / scenario toggle / 5-task probabilities, Topic Constellation 8 interactive orbs + Devanagari cluster inspection, Sentiment Pulse 11-month crosshair spline + 8 issue matrices, Canonical Responses 1–6 order + radial wheel + 18 example tab buttons, Solutions & Resolution 4-state ruled bar + expandable proposal/advice drawer, Models & Provenance `Not emitted` + Solution mismatch disclosure, 404 error alert & retry, Mobile 390px bottom dock & slide-over drawer |
| **Console & Network Cleanliness** | Full E2E + responsive sweep | **0 console errors, 0 uncaught exceptions, 0 failed requests** | Inline SVG favicon eliminated `/favicon.ico` 404 |
| **Responsive Overflow Verification** | `1440×900` (9 routes), `1280×800` (7 routes), `768×1024` (6 routes), `390×844` (9 routes) | **31 / 31 passed (`overflowX: false`)** | Zero horizontal page overflow across all 4 target breakpoints |

### 3. Visual Evidence (`docs/demo_screenshots/redesign/`)

19 Dusk studio screenshots captured and visually inspected from the live application:

- **Desktop (`1440px`):** `overview-desktop.png` (Dusk Midnight), `overview-light-desktop.png` (Daylight Bone), `command-palette-desktop.png` (`Ctrl+K` modal), `issues-desktop.png`, `explorer-desktop.png`, `thread-desktop.png`, `topics-desktop.png`, `sentiment-desktop.png`, `responses-desktop.png`, `resolution-desktop.png`, `resolution-audit-desktop.png`, `models-desktop.png`.
- **Tablet (`768px`):** `overview-tablet-768.png`.
- **Mobile (`390px`):** `overview-mobile.png`, `mobile-drawer.png`, `issues-mobile.png`, `explorer-mobile.png`, `thread-mobile.png`, `models-mobile.png`.

### 4. Honest Final Assessment

#### What improved
- **Research integrity & correctness:** Eliminated every P0 misleading display—Emotion & Issue `subset_accuracy` now honestly reports `Not emitted` (with explanation of `ml/evaluation/reports.py`), the Solution baseline zero-support artifact anomaly is transparently diagnosed, the Issues table aligns all columns cleanly, and the Monthly Sentiment chart renders crisp stacked SVG columns.
- **Analytical depth without backend risk:** Unlocked rich pre-existing API payloads (thread response mini-strips, 5-task probability distributions, Scenario-vs-Prediction comparison, language/script/code-mix badges, NMF keyword weights, all 3 response examples per class, expandable proposal/advice texts & binary probabilities on Solutions & Resolution, validation vs. test metrics, and evaluator warnings) while keeping all `/api/demo/*` contracts and model artifacts untouched.
- **Cohesive research UX:** Elevated RCI from a flat prototype into an 8-view editorial research workbench with hash routing, global `/` search, true hierarchical parent-child reply chains with jump-to-parent links, copyable technical IDs, glossary tooltips, and an accessible slide-over mobile drawer.

#### What remains incomplete / known limitations
- The underlying corpus remains 100% synthetic (160 posts, 1,280 comments, seed `20261116`), with equal 20-thread issue sizing and sequential domain timestamps.
- Backend hygiene items noted in Section B (e.g., `ml/evaluation/reports.py` omitting `subset_accuracy`, `train_demo_baselines.py` boolean-vs-string solution label accounting, NMF Devanagari vowel-matra tokenization, dead loop in `demo.py#L299-L302`) were intentionally left unmodified to preserve frozen artifacts and backend stability.
- The live PostgreSQL-backed `/api/v1/*` operational endpoints (`docs/API_SURFACE_PROPOSAL.md`) are not yet implemented.

#### Suitability for a faculty / institutional demonstration
**Yes — ready for confident academic and institutional demonstration** as a transparent research pipeline pilot. The interface communicates RCI's five-stage `Issue → Discussion → Community Response → Proposed Solution → Resolution Evidence` methodology with clarity, visual restraint, and full epistemic honesty about synthetic data provenance and model limitations.

#### What is required before using genuine community data
1. **Authorized data collection & ethics approval:** Collect genuine multilingual Reddit threads (including Bengali, Odia, Banglish, Romanized Odia, emoji, and Gen-Z slang per the academic brief) under applicable platform terms and institutional ethics review, with PII scrubbing.
2. **Human annotation & inter-annotator agreement:** Replace generator scenario rules with multi-annotator ground truth across all tasks (≥300+ balanced annotated rows per class before enabling MuRIL/ANN/CNN benchmarking).
3. **Evaluation & tokenization fixes before retraining:** Patch `ml/evaluation/reports.py` (propagate `subset_accuracy` / `hamming_loss`), patch `train_demo_baselines.py` (boolean-to-string label alignment for `solution`), and adopt Unicode grapheme-aware / neural embedding topic modeling for Indic scripts.
4. **Production API & inference pipeline:** Implement authenticated `/api/v1/*` database-backed endpoints with out-of-sample inference (never in-sample predictions on training threads) and verified discussion-completeness checks before asserting resolution states.

