# Contextual Issue–Solution Mapping Engine Report (V6 — Phase 4)

> **Research Integrity & Separation Notice:** All `951` corpus-wide issue–solution links produced by the V6 Contextual Mapper are explicitly labelled `verification_status: "model_inferred"` in [`ml/artifacts/live_corpus/issue_solution_map_v2.json`](file:///D:/AI_Project/ml/artifacts/live_corpus/issue_solution_map_v2.json). The human-reviewed `64`-pair evaluation benchmark is stored separately in [`data/annotations/issue_solution_eval_v1.csv`](file:///D:/AI_Project/data/annotations/issue_solution_eval_v1.csv) (`verification_status: "human_reviewed_v1"`). V5's lexical `TfidfCosineIssueSolutionLinker` remains 100% intact as the lexical baseline.

---

## 1. End-to-End Analytical Workflow

Implemented in [`ml/linking/contextual_mapper.py`](file:///D:/AI_Project/ml/linking/contextual_mapper.py) and executed by [`ml/experiments/v6_programme/run_phase4_phase5_linking_resolution.py`](file:///D:/AI_Project/ml/experiments/v6_programme/run_phase4_phase5_linking_resolution.py):

```
ISSUE DOMAIN (8 canonical domains + rich domain description)
  ↓
RELATED DISCUSSION (post title + body + thread context)
  ↓
RELEVANT COMMENTS (with parent_reddit_id context preserved)
  ↓
UPSTREAM RESPONSE TYPE & SOLUTION PROBABILITY (preserved, never overwritten)
  ↓
MULTI-SIGNAL CONTEXTUAL SCORING (E5-small + TF-IDF + Structure + Detector - Negation Penalty)
  ↓
SEMANTIC SOLUTION CLUSTERS (cosine AgglomerativeClustering per issue)
  ↓
TRACEABLE 9-FIELD EVIDENCE LINKS -> PHASE 5 RESOLUTION ENGINE
```

---

## 2. Multi-Signal Matching Formulation (Never Cosine Alone)

For every candidate triple `(issue_id, post_id, solution_comment_id)` with parent context `parent_context` and comment text `x`:

1. **Multilingual Sentence Embedding Similarity (`s_emb`, weight `0.38`):** L2-normalized cosine similarity using `intfloat/multilingual-e5-small` between the structured query `query: Issue (<name>): <description> | Thread: <post_title> | Parent: <parent_context>` and `passage: <comment_text>`, rescaled linearly across the E5 effective range `[0.68, 0.92] → [0, 1]`.
2. **Lexical TF-IDF Cosine (`s_lex`, weight `0.14`):** Same-thread 1–2 gram TF-IDF cosine between `post_title + parent_context` and `comment_text` (scaled by `min(1, 2.2 × cos)`).
3. **Structural & Parent-Context Proximity (`s_struct`, weight `0.18`):** Base same-thread alignment (`0.45`) + direct comment-level issue tag match (`+0.30` or `+0.15` if `p_issue >= 0.25`) + direct reply to OP (`+0.15`) or explicit parent reply (`+0.10`) + actionable procedural cues (`+0.10`).
4. **Upstream Detector Confidence (`s_det`, weight `0.30`):** `0.60 × solution_probability + 0.40 × role_prior(response_type)` where `solution=1.00, advice=0.78, personal_experience=0.48, opinion=0.30, agreement=0.15, counterargument=0.12`.
5. **Negation, Contradiction & Non-Actionable Penalty (`p_neg ∈ [0.0, 0.55]`):** Explicitly subtracts penalties for:
   - Ultra-short replies (`< 4 words`, `-0.28`)
   - Non-actionable social/DM/placeholder replies (`cfbr`, `idk`, `dm me`, `thanks`, `[Removed by Reddit]`, pure URL/GIF links, `-0.28`)
   - Negation/dismissal without an actionable alternative (`doesn't work`, `not worth`, `bad idea`, `scam`, `no chance`, `-0.25`; softened to `-0.08` if accompanied by a substantive alternative recommendation!)
   - `counterargument` (`-0.14` / `-0.06` with action) or pure `agreement` (`-0.18`) response roles.

$$\text{matching\_score} = \text{clip}\!\left(0.38\, s_{\text{emb}} + 0.14\, s_{\text{lex}} + 0.18\, s_{\text{struct}} + 0.30\, s_{\text{det}} - p_{\text{neg}},\; 0,\; 1\right)$$

A link is asserted iff `matching_score >= 0.56` AND `p_neg < 0.28`. Every link records all 9 required fields: `issue_id`, `post_id`, `solution_comment_id`, `parent_context`, `matching_score`, `model_confidence`, `evidence_text`, `matching_rationale`, and `verification_status` (`"model_inferred"`), plus the complete `signal_breakdown`.

---

## 3. Full Real-Corpus Mapping & Semantic Solution Clusters (`365` Threads / `3,557` Comments)

- **Candidate pairs evaluated:** `1,392`
- **High-confidence linked solutions (`model_inferred`):** **`951`** across all 8 issue domains
- **Candidates penalized for negation / dismissal / non-actionability:** **`268`**
- **Semantic Solution Clusters:** **`282`** distinct clusters via cosine `AgglomerativeClustering(linkage="average", distance_threshold=0.14)` so semantically similar proposals group together without collapsing distinct actionable advice.

| Issue Domain | Linked Solutions | Semantic Solution Clusters | Example Recurring Solution Themes |
|---|---:|---:|---|
| `administrative_communication` | 297 | 86 | AICTE refund guidelines, portal/email verification, category WL benchmarks, admitted batch groups |
| `placements` | 236 | 52 | LinkedIn alumni verification (`Open to Work`), B-school placement comparisons, CV/live-project prep |
| `curriculum_teaching` | 150 | 39 | Executive MBA vs. 2-yr PGP paths, BS DSAI vs. MPP comparisons, marketing/finance domain prep |
| `financial_concerns` | 134 | 39 | 15-day full refund rules (max ₹1k deduction), scholarship/aid options, education loan & CIBIL fixes |
| `extracurriculars` | 47 | 22 | Clubs/committees/case competitions for profile growth, PI gap-year explanation prep |
| `academic_workload` | 41 | 11 | Structured CAT retake vs. workex pacing, live projects & domain study habits |
| `campus_infrastructure` | 33 | 23 | Laptop hardware specifications (Discord curated list), local healthcare/connectivity facts (VIMSAR 2km) |
| `hostel_facilities` | 13 | 10 | Electric kettle room workaround, hostel room allotment process upon arrival |

---

## 4. Human-Reviewed Benchmark: V5 TF-IDF Baseline vs. V6 Contextual E5 Mapper (`n = 64` Real Pairs)

Evaluated on [`data/annotations/issue_solution_eval_v1.csv`](file:///D:/AI_Project/data/annotations/issue_solution_eval_v1.csv) (`8` candidate pairs per issue × `8` issue domains = `64` real corpus pairs graded `0 = irrelevant/non-actionable/contradictory`, `1 = partial helpful advice`, `2 = strong actionable solution`):

| Metric (`n = 64` real pairs across 8 issues) | V5 TF-IDF Baseline (`th = 0.35`) | V6 Contextual Multilingual E5 + Negation (`th = 0.56`) | Delta (V6 − V5) |
|---|---:|---:|---:|
| **Pairwise Precision** | 83.87% | **87.88%** | **+4.01 pp** |
| **Pairwise Recall** | 76.47% | **85.29%** | **+8.82 pp** |
| **Pairwise F1** | 80.00% | **86.57%** | **+6.57 pp** |
| **False-Positive Rate (FPR)** | 16.67% | **13.33%** | **−3.34 pp** *(better)* |
| **MRR** | 1.0000 | **1.0000** | 0.0000 |
| **Precision@1** | 1.0000 | **1.0000** | 0.0000 |
| **Precision@3** | 95.83% | **100.00%** | **+4.17 pp** |
| **Precision@5** | 80.00% | **82.50%** | **+2.50 pp** |
| **Recall@3** | 68.13% | **71.25%** | **+3.12 pp** |
| **Recall@5** | 93.75% | **96.88%** | **+3.13 pp** |
| **NDCG@3** | **0.8743** | 0.8368 | −0.0375 *(honest tradeoff)* |
| **NDCG@5** | **0.9289** | 0.9120 | −0.0169 *(honest tradeoff)* |

### Key Analytical Takeaways
1. **Substantial Pairwise F1 (+6.57 pp) & FPR Reduction (−3.34 pp):** Combining `multilingual-e5-small` embeddings with parent context and explicit negation/placeholder penalties eliminates false positives like `[ Removed by Reddit ]` (`cmt_01201`), bare image URLs (`cmt_03192`), 2-word replies (`"Clear it"`), and sarcastic dismissals, while recalling multilingual/paraphrased solutions that share few exact surface words with the thread title.
2. **Honest NDCG@3 Tradeoff:** On 2 issues where both `gold=1` (partial advice) and `gold=2` (strong solution) were relevant, V6 ranked a longer `gold=1` advice comment slightly ahead of a concise `gold=2` comment, nudging graded NDCG@3 (`0.8368` vs `0.8743`) slightly below TF-IDF while achieving **`100.0%` binary Precision@3**!
3. **Why Cross-Encoder Reranking Was Not Added:** With `MRR = 1.00`, `P@3 = 1.00`, and `86.57%` Pairwise F1 already achieved at `~2.4 ms/doc` using our shared `multilingual-e5-small` bi-encoder, adding a second heavy cross-encoder checkpoint would increase GPU/disk complexity without a justified top-3 ranking gain.
