# Problem Statement vs. RCI V6 (`v6.0.0`) — Gap Analysis, Missing Deliverables & Tuning Roadmap

> **Reference Specification:** *Group Project: AI-Powered Discovery of Student Issues and Solutions from Multilingual Reddit Discussions* (19-page official project brief).  
> **Evaluated Product State:** **Reddit Community Intelligence (RCI) `v6.0.0`** (`iim-sambalpur-real-reddit-v6`, strictly inductive Track A 5-fold pure neural ensembles, `65.01%` frozen held-out test Macro F1, `7/7` Pipeline Integrity Checks passing, `2,193` offline tests passing, zero synthetic training/test rows, zero test-thread DAPT leakage).

---

## 1. Executive Scorecard

| Evaluation Area | PDF Reference | Current RCI V6 Status | Gap Severity |
|---|---|---|---|
| **Core NLP Innovation (`Issue → Discussion → Response → Solution`)** | Sec. 2–3, Tasks 7–9 | **Exceeds Spec** — 5 dedicated Transformer classifiers (`MuRIL-DAPT-WWM` + `XLM-R`), 6 canonical response roles in exact order, 10-cluster NMF discovery, 4-state proposal-vs-resolution separation | Done |
| **Real Anonymized Reddit Corpus & Privacy** | Task 3 (pp. 5–6) | **Exceeds Spec** — 365 threads, 3,557 comments, 8 subreddits, full PII/ID/URL pseudonymization + CI hygiene gates | Minor (multilingual breadth) |
| **Manual Annotation & Governance** | Task 1 & 4 (pp. 4, 6–7) | **Strong (with Transparent Audit)** — `459` curated real rows + `ANNOTATION_GUIDELINES.md` + local Annotation Studio + 50-item blind human IAA audit (`0/5` gates passed → Track B V2 honestly archived/blocked) | Medium (`Solution type` column, sample-size rationale doc, IAA calibration) |
| **Language, Script, Slang & Emoji Handling** | Tasks 5–6 (pp. 7–9) | **Partial** — English + Hinglish (`324` comments, `9.1%`) works cleanly, but Bengali/Odia/Banglish/Roman-Odia coverage & wiring, missing emoji/slang entries (`💀`, `🤡`, `fr`, `mid`, `cooked`, `bro`, `bhai`, `lol`, `based`), and Task 6 preprocessing ablation are incomplete | **High (Explicit Rubric Ask)** |
| **Issue–Solution Linking, Solution Clustering & Unresolved Criteria** | Tasks 10–11, Sec. 18 (pp. 11–12, 17) | **Partial** — `TfidfCosineIssueSolutionLinker` exists & is tested, but is **not wired** into the live V6 pipeline/dashboard; dashboard lacks ranked **"Most Common Community Solutions"** clusters per issue and explicit **Resolved / Partially Resolved / Unresolved** quantitative criteria table | **High (Core Spec Feature)** |
| **Cross-IIM & Cross-Language Comparative Analytics** | Sec. 1 Q8 (p. 2) | **Not Yet Exposed** — `corpus.json` already mentions 21 distinct IIMs and `324` Hinglish vs. `2,914` English comments, but no UI view compares issues/sentiment/solutions across IIMs or languages | **Medium-High** |
| **Formal Academic Deliverables (1–6)** | Sec. 19 (pp. 18–19) | **3/6 Complete, 3/6 Fragmented or Missing** — Deliverable 3 (Prototype) is production-grade; Deliverables 1 (Video), 2 (Data Dictionary), 4 (Consolidated 11-Section Architecture Doc), 5 (Math + Pseudocode Doc), and 6 (7-Question Final Insight Report) still need clean standalone artifacts | **Critical for Submission** |

---

## 2. Task-by-Task Compliance Audit (Tasks 1–14 & 8 Research Questions)

### A. Where RCI V6 Already Meets or Exceeds the Specification

1. **Task 1 — Define the Problem (`docs/ANNOTATION_GUIDELINES.md`):** Formal operational definitions and boundary rules exist for issues (8 domains), sentiment (3 classes), emotion (6 labels), canonical community response roles (6 classes), actionable solutions, and proposal-vs-resolution distinction.
2. **Task 3 — Data Collection (`data/live_corpus/corpus.json`):** `365` real threads (`post_0001`..`post_0365`) and `3,557` human comments (`cmt_00001`..`cmt_03557`, `2,046` with verified immediate parent-comment links) across `8` active subreddits (`r/IIMSambalpurUG`, `r/MBAIndia`, `r/IIMCATPreparation`, `r/iim_CAT`, `r/CATPrep`, `r/CATStudyRoom`, `r/CATpreparation`, `r/Indian_Academia`), containing post titles, bodies, comments, dates, subreddits, parent/comment hierarchy, upvotes, and strictly zero PII/raw Reddit IDs/URLs.
3. **Task 7 — Discover Major Issues (`ml/artifacts/live_corpus/topic_discovery.json`):** Unsupervised **10-cluster NMF** (`8,000`-term TF-IDF vocabulary) discovers latent community themes directly from the real corpus without pre-supplied categories, visualized in the Studio's `Topics` view.
4. **Task 8 — Multi-Label Issue & Emotion Identification (`backend/build_live_corpus_artifacts.py`):** Multi-label `MuRIL-DAPT-WWM` with true `parent_pair` reply context, Asymmetric Loss (ASL), and OOF per-label monotonic Platt calibration reaches **`60.83%` frozen test Macro F1 (`0.012` ECE)** on 8 issues and **`61.04%` (`0.007` ECE)** on 6 emotions. Across the live corpus: `administrative_communication` (`1,995`), `placements` (`732`), `curriculum_teaching` (`557`), `financial_concerns` (`406`), `campus_infrastructure` (`297`), `extracurriculars` (`229`), `academic_workload` (`119`), `hostel_facilities` (`110`); emotions: `curiosity` (`1,075`), `frustration` (`804`), `stress` (`366`), `hope` (`274`), `gratitude` (`210`), `confidence` (`202`).
5. **Task 9 — Identify Solutions, Advice & 6 Canonical Community Responses:**
   - All 6 exact canonical roles are classified in fixed order: `agreement` (`662`), `personal_experience` (`552`), `opinion` (`1,455`), `advice` (`321`), `solution` (`326`), `counterargument` (`241`).
   - Binary Solution Detection uses **`xlm-roberta-base`** (5-fold, zero DAPT), achieving **`84.57%` frozen test Macro F1** (`+8.47 pp` over V5, `+10.13 pp` over `MuRIL-DAPT-WWM`), flagging `535` actionable solution comments (`15.0%`) with **`73.9%` (`241/326`)** cross-task agreement with `response=="solution"`.
6. **Model Rigor Beyond the Brief:** Strictly inductive zero-leakage WWM DAPT (`44` test/same-title threads / `749` comments excluded; perplexity `88.32 → 25.23` on `347` unseen texts), apples-to-apples 5-fold 4-architecture benchmarking (`LogReg 49.91%`, `ANN 43.62%`, `CNN 42.09%`, `MuRIL-WWM 62.99%`, `Deployed V6 65.01%`), data-driven empirical error analysis, 7/7 automated Pipeline Integrity checks, dual Vercel + hardened Docker deployment, and `2,193` passing automated tests.

---

### B. Concrete Functional & Research Gaps Found in Code and Data

#### Gap 1 — Task 5: Bengali, Odia, Banglish, Roman-Odia Coverage & Existing-LID Limitation Study (PDF pp. 1, 6–8)
- **Observed Corpus Reality:** Current `corpus.json` contains `2,914` English, `324` Hinglish/Hindi (`9.1%`), and `319` terse/undetermined comments (`scripts`: `latin 3,496`, `unknown 60`, `mixed 1`, `0` pure Devanagari/Bengali/Odia script comments; only `1` Banglish lexical hit and `4` Roman-Odia lexical hits).
- **Code Wiring Gap:** `backend/app/services/nlp/code_mix_detector.py` already defines `_BENGALI_ROMAN_WORDS` and `_ODIA_ROMAN_WORDS` plus Unicode script detectors for Devanagari, Bengali, and Odia, **but `detect_indian_reddit_language()` inside `backend/build_live_corpus_artifacts.py` does not invoke them** when tagging live corpus comments.
- **Missing Required Study (PDF p. 8):** *"Students should investigate the limitations of existing language identification models on such data."* We previously observed `langdetect` misclassifying ~19% of Hinglish Reddit comments as Welsh/French/Somali, but there is no reproducible benchmark script + table documenting `langdetect` / `fastText` failure rates on our real Hinglish/code-mixed comments vs. our custom Indian-Reddit detector.

#### Gap 2 — Task 6: Gen-Z Slang & Emoji Coverage + Mandatory Preprocessing Ablation (PDF pp. 8–9)
- **Exact Coverage Audit Against PDF Examples:**
  - `backend/app/services/nlp/preprocessing/slang_normalizer.py` (`SLANG_LEXICON`, 31 entries) covers only **3 of 10** terms listed on PDF p. 8 (`ngl`, `idk`, `btw`) and is **missing 7** (`bro`, `bhai`, `fr`, `lol`, `mid`, `based`, `cooked` — note `mid` appears `11` times, `fr` `5` times, `cooked` `2` times in the real corpus) and contains a duplicate `"imo"` key.
  - `backend/app/services/nlp/preprocessing/emoji_normalizer.py` (`_EMOJI_MAP`, 59 entries) covers **3 of 5** emojis on PDF pp. 8–9 (`😭`, `😂`, `🔥`) and is **missing 2** (`💀` — `6` hits in corpus, featured in PDF Tasks 1 & 6; `🤡` — `7` hits, featured in PDF Task 6).
- **Missing Required Empirical Ablation (PDF pp. 8–9):** The brief explicitly asks two experimental questions that do not yet have a controlled experiment report:
  1. *Does conventional preprocessing (**stop-word removal + stemming + punctuation removal**) actually help or hurt on informal multilingual Reddit text?*
  2. *Should **emojis be removed** or **converted into semantic affect tokens** (`😭 → sadness/frustration`, `🔥 → strong approval`, `💀 → extreme amusement/disbelief`)?*

#### Gap 3 — Task 10: Issue–Solution Linking & Ranked "Most Common Community Solutions" Clustering (PDF pp. 4, 11, 17)
- **Existing Unwired Component:** `backend/app/services/nlp/issue_solution_linking/tfidf_cosine_linker.py` (`TfidfCosineIssueSolutionLinker`) is implemented and unit-tested, **but never called in `build_live_corpus_artifacts.py` or surfaced in `/api/demo/*` and the Studio UI**.
- **Missing Solution Theme Clustering:** Currently the `Resolution` view tracks per-thread proposed solutions and 4 thread resolution states (`reported_success: 17`, `reported_pending: 25`, `proposed_only: 208`, `insufficient_evidence: 115`), **but does not group solution comments into ranked themes per issue** (e.g., *Financial Aid / Education Loan*, *Peer & Alumni Mentoring*, *CAT/GDPI Prep Strategy*, *Off-Campus / Mess Alternatives* as required on PDF pp. 4, 11, 17).

#### Gap 4 — Task 11: Quantitative `Resolved / Partially Resolved / Unresolved` Issue Criteria (PDF pp. 4, 12, 17)
- **What Exists:** Thread-level evidence states (`reported_success`, `reported_pending`, `proposed_only`, `insufficient_evidence`) and the `Issue Friction & Priority Quadrant` (mention volume vs. negative sentiment & unresolved ratio).
- **What Is Missing:** PDF p. 12 explicitly asks for a **per-issue table** (`Issue | Discussion Volume | Solutions Detected | Resolution Strength | Status`) backed by an explicit, reproducible **quantitative decision rule** classifying each issue domain into `Resolved`, `Partially Resolved`, or `Unresolved` (e.g., based on solution density, verified `reported_success` rate, and negative-sentiment persistence).

#### Gap 5 — Question #8: Cross-IIM & Cross-Language Comparative View (PDF p. 2)
- **Untapped Signal Already in `corpus.json`:** Direct entity scanning shows **21 distinct IIMs** mentioned across the corpus (`IIM Sambalpur 431`, `IIM Mumbai 78`, `IIM Sirmaur 36`, `IIM Jammu 25`, `IIM Vizag 23`, `IIM Raipur 21`, `IIM Nagpur 19`, `IIM Udaipur 14`, `IIM Kashipur 14`, `IIM Indore 13`, `IIM Amritsar 12`, `IIM Lucknow/Shillong/Rohtak 10` each, etc.) alongside `324` Hinglish/Hindi vs. `2,914` English comments.
- **UI Gap:** No dashboard tab or API endpoint currently slices issues, sentiment, emotion, or solution density **by mentioned IIM** or **by language register (English vs. Hinglish/Hindi)** to directly answer Research Question #8 (*"How do discussions differ across topics, languages and IIMs?"*).

#### Gap 6 — Task 4: Optional `Solution Type` Label & Explicit Sample-Size Justification (PDF pp. 6–7)
- `master_labeled_v1.csv` covers all 5 core supervised tasks on `459` real comments (`200` threads), which exceeds standard binary-only schemes, but does not include a separate `solution_type` categorical column (such as `financial_aid`, `prep_strategy`, `institutional_escalation`, `peer_mentoring`, `alternative_option`) or a standalone written justification of *"How many samples were manually annotated (`459`) and why?"* (statistical power, `post_id`-grouped split constraints, rare-class support, and human annotation budget).

---

## 3. Formal Deliverables Checklist (PDF Section 19, pp. 18–19)

| Deliverable | PDF Requirement | Current Status | Action Required |
|---|---|---|---|
| **Deliverable 1** — Real-Life Scenario Video | 2–4 minute video demonstrating the multilingual Reddit analyst overload problem + proposed AI pipeline output | **Missing** | Produce a shot-by-shot storyboard + narration script; record screen-capture walkthrough of the live RCI V6 Studio + sample raw multilingual Reddit chaos |
| **Deliverable 2** — Reddit Dataset | Raw dataset/sample, Cleaned dataset, **Data Dictionary**, Annotation guidelines, Manually annotated sample (+ why `459` samples) | **80% Done** | Add **`docs/DATA_DICTIONARY.md`** documenting every field, type, constraint, and enum across raw archive sample, `corpus.json`, `predictions.json`, `topic_discovery.json`, `evaluation.json`, and `master_labeled_v1.csv` + explicit sample-size rationale |
| **Deliverable 3** — AI/NLP Prototype | Working **notebook/application**, source code, model, sample predictions, visualizations/dashboard | **95% Done** | Full FastAPI Studio + Docker + Vercel + models + predictions are live; add an executable **`notebooks/RCI_V6_Prototype_Walkthrough.ipynb`** (inspects corpus, runs preprocessing, inspects V6 metrics/error analysis, demonstrates inference) for notebook-first reviewers |
| **Deliverable 4** — AI Architecture Document | Unified doc with all **11 exact sections** (1. Problem def → 5. Model arch → **6. Hardware arch + GPU necessity justification** → 7. Software arch → 8. Training → 9. Evaluation → 10. Deployment → 11. Limitations) | **Fragmented** | Consolidate `README.md` + `docs/V5_*.md` into one clean **`docs/DELIVERABLE_4_AI_ARCHITECTURE.md`** covering all 11 numbered sections, including empirical **GPU vs. CPU justification** (`8-epoch WWM DAPT: ~6.3 min on RTX 4060 vs. hours on CPU`; single-comment inference `~1.82–5.39 ms` GPU vs. `~48.9 ms` CPU; static JSON serving `<10 ms` CPU at zero runtime GPU cost) |
| **Deliverable 5** — Mathematical Foundation + Pseudocode | Mathematical formulation, algorithms, **only equations actually used**, complete pseudocode matching V6, and explicit math-to-code mapping | **Missing** | Write **`docs/DELIVERABLE_5_MATH_AND_PSEUDOCODE.md`** covering exact V6 math: Word+Char TF-IDF, WordPiece/SentencePiece embeddings, Scaled Dot-Product Self-Attention, `[CLS]+Mean+Max` Attention Pooling, Class-Weighted CE & Softmax, Asymmetric Loss (ASL), Monotonic OOF Platt Calibration, NMF, Cosine Similarity (Issue–Solution Linker), Macro-F1 / 5-Bin ECE / Cohen's $\kappa$ / MASI Krippendorff's $\alpha$, plus complete V6 pseudocode mapped to exact file/function paths |
| **Deliverable 6** — Final Insight Report | Executive report answering all **7 exact questions** from p. 19 (including *"How reliable is the AI's analysis?"*) | **Missing** | Write **`docs/DELIVERABLE_6_FINAL_INSIGHT_REPORT.md`** populated with verified V6 (`v6.0.0`) corpus findings, top issues, emotions, solution themes, unresolved issues, Cross-IIM/language contrasts, and an honest AI reliability section (`65.01%` frozen test F1, `0.062` ECE, per-class residual error analysis, and `0/5` Human IAA governance audit) |

---

## 4. High-Impact Product Features & Genuine Model Tuning Roadmap

### Tier 1 — Highest-Impact Deliverables & Rubric Completeness (Recommended Next)
1. **Generate Deliverables 2, 4, 5, and 6 Markdown Reports (`docs/DATA_DICTIONARY.md`, `docs/DELIVERABLE_4_AI_ARCHITECTURE.md`, `docs/DELIVERABLE_5_MATH_AND_PSEUDOCODE.md`, `docs/DELIVERABLE_6_FINAL_INSIGHT_REPORT.md`) + Video Storyboard (`docs/DELIVERABLE_1_VIDEO_SCRIPT.md`):** Closes every formal grading artifact gap using 100% live-verified V6 data.
2. **Wire Issue–Solution Linking, Solution Clustering & Quantitative Unresolved Table (Tasks 10–11):**
   - Integrate `TfidfCosineIssueSolutionLinker` + per-issue TF-IDF/NMF solution theme clustering into `build_live_corpus_artifacts.py` and `/api/demo/resolution`.
   - Expose a clean **"Most Common Community Solutions"** ranked card per issue and an explicit **`Resolved / Partially Resolved / Unresolved` + Resolution Strength (`High / Moderate / Low`)** table with documented formula criteria in the Studio UI.
3. **Add Cross-IIM & Cross-Language Comparative Analytics (Research Question #8):**
   - Add an IIM entity extractor (for the 21 already-present IIM mentions) and language-register comparator (English vs. Hinglish/Hindi) in `/api/demo/comparisons` and a focused Studio sub-section/view.
4. **Upgrade Preprocessing & Run Controlled Task 5 + Task 6 Ablations:**
   - Wire Bengali/Odia/Banglish/Roman-Odia detection from `code_mix_detector.py` into live corpus tagging; add missing PDF emojis (`💀`, `🤡`) and slang (`bro`, `bhai`, `fr`, `lol`, `mid`, `based`, `cooked`) while fixing the duplicate `"imo"` key.
   - Run & document two controlled ablations on the V1 5-fold CV splits (never touching test for selection):
     1. Existing LID (`langdetect`) vs. custom Indian-Reddit detector on Hinglish/code-mixed comments.
     2. Raw text vs. stopword+stemming+punctuation removal vs. emoji stripping vs. emoji-to-semantic-token conversion.
5. **Interactive "Live Comment Analyzer" Demo Widget (Great for Deliverable 1 Video & Live Demo):**
   - Add a lightweight live sandbox tab/panel (running on local Docker or lightweight classical/cached fallback on Vercel) where a presenter types a Hinglish + emoji + slang comment (e.g., *"Bro hostel life 💀... mess food toh literally mid... try applying for financial aid"*) and sees real-time Language/Script detection, Emoji/Slang normalization, Issue, Sentiment, Emotion, Response Role, and Solution probability.

### Tier 2 — Genuine ML Model Improvements (Strictly Non-Fabricated, Evidence-Backed)

All tuning suggestions below are grounded in our empirical V5/V6 ablation results (`docs/V5_MODEL_IMPROVEMENT_REPORT.md`, `docs/V5_HUMAN_IAA_REPORT.md`, `docs/V5_DAPT_BACKBONE_ABLATION.md`, `docs/V5_MULTITASK_RDROP_REPORT.md`) — zero fabricated labels, zero test-set tuning:

1. **Human IAA Rubric Calibration → Unlock Track B V2 (`+178` Active-Learning Training Rows):**
   - **Why it is the #1 genuine lever:** In Phase 7, adding the `178` uncertainty-sampled training rows lifted **Response Macro F1 by `+6.56 pp`** (`53.5% → 59.9%`, statistically significant, `p < 0.01`) and tightened bootstrap 95% confidence intervals by **`15%–26%`**.
   - **Required gate:** Resolve the 6 concrete boundary ambiguities identified in `docs/V5_HUMAN_IAA_REPORT.md` (notably `neutral` vs. `negative` factual complaints, `advice` vs. `solution` specificity, `opinion` vs. `counterargument`, and multi-label `Issue` primary-vs-incidental scope), run a fresh 30-comment blind dual-annotator pass until all 5 tasks clear $\kappa \ge 0.70$ / MASI $\alpha \ge 0.65$, and only then promote V2.
2. **Test `xlm-roberta-base` on `Response` Classification (5-Fold + Multi-Seed):**
   - In Phase 3 backbone ablations, `xlm-roberta-base` (zero DAPT) achieved a 3-seed OOF/test mean of **`58.22% ± 0.41%`** on `Response` vs. `MuRIL-DAPT-WWM`'s **`54.64% ± 1.10%`**, and already proved a **`+10.13 pp`** win on `Solution` (`84.57%` vs. `74.44%`). Evaluating `XLM-R` (or strict-inductive WWM-DAPT `XLM-R`) on `Response` via 5-fold OOF CV is the highest-yield pure architecture lever for our hardest task (`53.52%`).
3. **2-Encoder Shared Multi-Task Learning (`Sentiment + Response + Solution` with R-Drop):**
   - Phase 5 showed that sharing one encoder across `Sentiment + Response + Solution` with symmetric KL R-Drop regularization hit **`69.66%` Sentiment** and **`57.52%` Response** while running **`5.6×` faster** than 5 separate single-task encoders. Combining an `XLM-R` or `MuRIL-WWM` shared 3-task head with separate `Issue` (`parent_pair + ASL + Platt`) and `Emotion` (`ASL + Platt`) heads could simultaneously improve `Sentiment`/`Response` F1 and cut inference latency/memory in half.
4. **Span-Masked Reply-Only Pooling (`mask_segment_b`) for `Response` (`counterargument`):**
   - Naive `parent_pair` concatenation helped `Issue` (`+15.1 pp`) because topic is inherited from the parent post, but hurt `Response` because pooling over parent tokens diluted the reply's own stance. Cross-attending to `[parent] [SEP] [reply]` while restricting `[Mean + Max]` pooling strictly to **Segment B (reply tokens only)** lets the model verify disagreement with the parent (`counterargument`!) without leaking parent sentiment/role.
5. **Targeted Data Collection for Rare Classes & Regional Languages (Strictly Excluded from Frozen Test):**
   - **`academic_workload` (`16` train examples) & `counterargument` (`9` frozen test support):** Collecting term-time IIM MBA curriculum/rigor threads and genuinely Bengali/Odia/Banglish/Roman-Odia student discussions (from regional Indian college/CAT subreddits) addresses both corpus representation gaps (PDF Task 3/5) and rare-class under-support — while keeping the `92`-comment V1 frozen test set untouched as a historical continuity benchmark.
