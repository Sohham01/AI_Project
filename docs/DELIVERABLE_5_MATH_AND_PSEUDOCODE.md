# Deliverable 5 — Mathematical Foundations, Complete Pseudocode & Code Translation

> **Scope Rule (PDF Task 13 & Sec. 19):** Every equation below corresponds strictly to a component **actually implemented and executed** in the Reddit Community Intelligence (RCI) prototype. No unrelated textbook formulas are included; Section 3 maps every formula directly to its exact source file and symbol.

---

## 1. Mathematical Formulation of the RCI Pipeline

### 1.1 Lexical Representation — Word + Character Sublinear TF-IDF (Baselines, NMF & Issue–Solution Linker)

Used in the Logistic Regression baseline, 2-layer ANN/MLP baseline, Unsupervised NMF Topic Discovery, and the `TfidfCosineIssueSolutionLinker`. For term $t$ (word $1\text{–}2$-gram or character boundary $3\text{–}5$-gram) in document $d$ from a corpus of $N$ documents with document frequency $\mathrm{df}(t)$:

$$\mathrm{TF}_{\text{sub}}(t, d) = \begin{cases} 1 + \ln f_{t, d} & \text{if } f_{t, d} > 0 \\ 0 & \text{otherwise} \end{cases}, \qquad \mathrm{IDF}(t) = \ln\!\left(\frac{1 + N}{1 + \mathrm{df}(t)}\right) + 1$$

$$x_{d, t} = \frac{\mathrm{TF}_{\text{sub}}(t, d)\,\mathrm{IDF}(t)}{\sqrt{\sum_{t'} \left[\mathrm{TF}_{\text{sub}}(t', d)\,\mathrm{IDF}(t')\right]^2}} \in \mathbb{R}_{\ge 0}$$

Character $n$-grams (`char_wb`, $n \in \{3, 4, 5\}$) capture Indian-Reddit spelling noise (`"waitlst"`, `"accha"/"achha"`, `"nhi"/"nahi"`).

---

### 1.2 Contextual Multilingual Embeddings & Scaled Dot-Product Self-Attention (`MuRIL-DAPT-WWM` & `XLM-RoBERTa-base`)

Each input sequence (either single comment $x$ or context pair $\langle \text{reply}, \text{parent\_context} \rangle$ separated by `[SEP]` / `</s>`) is tokenized via WordPiece (`google/muril-base-cased`, $|V| = 197{,}285$) or SentencePiece (`xlm-roberta-base`, $|V| = 250{,}002$) into $L$ token indices $(w_1, \dots, w_L)$, mapped to initial representations $H^{(0)} \in \mathbb{R}^{L \times d}$ ($d = 768$, $12$ Transformer layers, $h = 12$ heads, $d_k = d / h = 64$).

At each layer $\ell \in \{1, \dots, 12\}$ and head $j \in \{1, \dots, 12\}$:

$$Q_j = H^{(\ell-1)} W_j^Q, \qquad K_j = H^{(\ell-1)} W_j^K, \qquad V_j = H^{(\ell-1)} W_j^V$$

$$\mathrm{Attention}(Q_j, K_j, V_j) = \mathrm{softmax}\!\left(\frac{Q_j K_j^\top}{\sqrt{d_k}} + M_{\text{pad}}\right) V_j$$

where $M_{\text{pad}}$ masks padding positions with $-\infty$, followed by multi-head concatenation, residual LayerNorm, and a GELU feed-forward block.

---

### 1.3 Strictly Inductive Whole-Word Masking DAPT Objective (`MuRIL-DAPT-WWM`)

Before supervised fine-tuning, `google/muril-base-cased` undergoes Domain-Adaptive Pretraining (DAPT) on strictly unseen-test-excluded unlabeled Reddit texts $\mathcal{U}_{\text{train}}$ (`44` frozen-test & same-title threads excluded; zero overlap). Unlike random subword masking, **`##`-aligned Whole-Word Masking (WWM)** masks entire contiguous WordPiece words $\mathcal{M}(x)$ (15% whole-word masking rate; 80% `[MASK]`, 10% random token, 10% unchanged) so Hinglish and MBA terms (`waitlist`, `sambalpur`, `placement`, `kaisa`) must be reconstructed holistically:

$$\mathcal{L}_{\text{MLM}}(\theta) = -\frac{1}{|\mathcal{M}(x)|} \sum_{i \in \mathcal{M}(x)} \ln P_\theta\!\left(w_i \mid x_{\setminus \mathcal{M}(x)}\right)$$

Held-out domain adaptation quality is tracked on thread-disjoint unseen Reddit validation texts $\mathcal{U}_{\text{val}}$ (`n = 347`) via validation perplexity:

$$\mathrm{PPL}(\mathcal{U}_{\text{val}}) = \exp\!\left(\frac{1}{|\mathcal{U}_{\text{val}}|} \sum_{x \in \mathcal{U}_{\text{val}}} \mathcal{L}_{\text{MLM}}(x)\right)$$

---

### 1.4 Attention-Masked `[CLS] + Mean + Max` Pooling, Span-Masked Reply Pooling & Multi-Sample Dropout

From final contextual states $H = H^{(12)} \in \mathbb{R}^{L \times d}$ with binary pooling mask $m \in \{0, 1\}^L$:

- **Standard Full-Sequence Mask** (`comment_only` or `Issue parent_pair`): $m_i = \mathbb{I}[\text{token } i \text{ is non-padding}]$.
- **Span-Masked Reply-Only Mask** (`reply_only_pool = True`, for context-aware stance classification): all 12 layers cross-attend to $\langle \text{reply}, \text{parent} \rangle$, but pooling restricts $m_i = \mathbb{I}[\text{token } i \text{ precedes first }\texttt{[SEP]}/\texttt{</s>}]$ so parent affect cannot bleed into pooled reply features.

The concatenated representation $u \in \mathbb{R}^{3d}$ ($2{,}304$-dim) passes through LayerNorm and $K = 5$ parallel Multi-Sample Dropout branches ($p = 0.20$) whose logits are averaged:

$$h_{\text{cls}} = H_0, \qquad h_{\text{mean}} = \frac{\sum_{i=1}^{L} m_i H_i}{\max\!\left(\sum_{i=1}^{L} m_i,\, 10^{-6}\right)}, \qquad h_{\text{max}, r} = \max_{i:\, m_i = 1} H_{i, r}$$

$$u = \mathrm{LayerNorm}\!\left([h_{\text{cls}};\, h_{\text{mean}};\, h_{\text{max}}]\right), \qquad z = \frac{1}{5} \sum_{k=1}^{5} \left(W\,\mathrm{Dropout}_k(u) + b\right)$$

**Layer-Wise Learning Rate Decay (LLRD, $\lambda = 0.92$):** Layer $\ell \in \{0, \dots, 11\}$ uses $\eta_\ell = \eta_{\text{base}} \cdot 0.92^{11 - \ell}$, embeddings use $\eta_{\text{base}} \cdot 0.92^{12}$, and the classification head uses $4\,\eta_{\text{base}}$ with cosine decay and $10\%$ linear warmup.

---

### 1.5 Task Output Heads & Imbalanced Loss Functions

#### A. Single-Label Multiclass & Binary Tasks (`Sentiment`, `Response`, `Solution`)
Predicted class distribution via Softmax:

$$P(y = c \mid x) = \frac{\exp(z_c / T)}{\sum_{j=1}^{C} \exp(z_j / T)}, \qquad T \ge 1.0$$

Trained with inverse-frequency **Class-Weighted Label-Smoothed Cross-Entropy** ($\varepsilon = 0.05$, $w_c = \frac{N}{C \cdot N_c}$):

$$\mathcal{L}_{\text{CE-W}}(z, y) = -\sum_{c=1}^{C} w_c\, q_c \ln P(y = c \mid x), \qquad q_c = (1 - \varepsilon)\,\mathbb{I}[c = y] + \frac{\varepsilon}{C}$$

#### B. Multi-Label Imbalanced Tasks (`Issue` — 8 labels, `Emotion` — 6 labels)
Each label $c \in \{1, \dots, C\}$ outputs an independent Bernoulli logit $z_c$ with probability $p_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}$. Because multi-label real Reddit threads are heavily negative-skewed (e.g., `academic_workload` and `hostel_facilities` are positive in $<5\%$ of comments), we train with **Asymmetric Loss (ASL)** ($\gamma_+ = 0$, $\gamma_- = 2.0$, probability shift clip $m = 0.05$):

$$p_{c, m} = \max(p_c - m,\, 0)$$

$$\mathcal{L}_{\text{ASL}}(z, y) = -\frac{1}{C} \sum_{c=1}^{C} \Big[\, y_c (1 - p_c)^{\gamma_+} \ln(p_c) + (1 - y_c)\, p_{c, m}^{\gamma_-} \ln(1 - p_{c, m}) \,\Big]$$

#### C. Multi-Task R-Drop Consistency Regularization (Evaluated in Phase 3 & Stage 3)
Two stochastic dropout forward passes produce distributions $P_1, P_2$; total loss adds symmetric KL divergence:

$$\mathcal{L}_{\text{R-Drop}} = \frac{1}{2}\big(\mathcal{L}_{\text{task}}(z^{(1)}, y) + \mathcal{L}_{\text{task}}(z^{(2)}, y)\big) + \frac{\alpha}{2}\Big(D_{\text{KL}}(P_1 \| P_2) + D_{\text{KL}}(P_2 \| P_1)\Big)$$

---

### 1.6 Post-Hoc Probability Calibration & 5-Fold Ensembling

Across $F = 5$ `post_id`-grouped folds, ensemble test/corpus logits average the 5 fold models: $\bar{z}(x) = \frac{1}{5} \sum_{f=1}^{5} z^{(f)}(x)$.

1. **Non-Sharpening Temperature Scaling (`Sentiment`, `Response`, `Solution`):** Constrained $T \ge 1.0$ minimizing validation NLL (native 5-fold ensembling achieves $T = 1.0$ without overconfidence).
2. **OOF Per-Label Monotonic Platt Calibration (`Issue`, `Emotion` ASL heads):** Because ASL shifts negative logits upward to suppress easy negatives, raw $\sigma(z_c)$ is systematically overconfident ($\text{ECE} \approx 0.15\text{–}0.16$). Strictly on out-of-fold logits $z^{\text{OOF}}_{i, c}$, we fit per-label logistic regression with positive slope constraint $a_c \ge 0.05$:

$$\hat{p}_c(x) = \sigma\!\left(a_c \bar{z}_c(x) + b_c\right), \qquad a_c > 0$$

Because $a_c > 0$ is strictly monotonic, ROC-AUC/ranking is preserved while ECE drops by $12\times\text{–}22\times$. Regularized per-label decision thresholds $\tau_c \in [0.25, 0.70]$ are tuned strictly on OOF calibrated probabilities with empirical-Bayes shrinkage toward $0.50$ proportional to positive support $n_c$:

$$\tau_c^{\text{reg}} = \mathrm{clip}\!\left(\frac{n_c}{n_c + 10}\tau_c^{\text{OOF}} + \frac{10}{n_c + 10}(0.50),\; 0.25,\; 0.70\right)$$

3. **Empty-Set Multi-Label Confidence:** When an annotatable multi-label head predicts the empty set ($\hat{Y} = \emptyset$), calibrated confidence reflects certainty that *no* label applies: $\mathrm{conf} = 1 - \max_c \hat{p}_c(x)$; otherwise $\mathrm{conf} = \frac{1}{|\hat{Y}|}\sum_{c \in \hat{Y}} \hat{p}_c(x)$.

---

### 1.7 Unsupervised Topic Discovery — Non-Negative Matrix Factorization (NMF, Task 7)

Given the non-negative document-term TF-IDF matrix $X \in \mathbb{R}_{\ge 0}^{N \times V}$ ($N = 3{,}557$ comments, $V \le 8{,}000$ terms) and $K = 10$ latent topics, NMF solves:

$$\min_{W \ge 0,\; H \ge 0} \; \frac{1}{2}\|X - W H\|_F^2$$

where $W \in \mathbb{R}_{\ge 0}^{N \times K}$ gives each comment's topic membership weights (dominant topic $\arg\max_k W_{i, k}$) and $H \in \mathbb{R}_{\ge 0}^{K \times V}$ gives the weighted top terms per discovered topic.

---

### 1.8 Thread-Scoped Issue–Solution Linking — Blended Cosine Similarity (Task 10)

For every same-thread candidate pair $(u_{\text{issue}}, v_{\text{sol}})$ with L2-normalized TF-IDF vectors $\mathbf{a}, \mathbf{b}$:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2} \in [0, 1]$$

$$\mathrm{link\_score}(u, v) = w_{\text{sim}}\cos(\theta) + w_{\text{prob}}\,p_{\text{sol}}(v) + w_{\text{thr}}\,r_{\text{thread}}(u, v) + w_{\text{role}}\,r_{\text{role}}(v)$$

with normalized weights summing to $1$ ($w_{\text{sim}}=0.45$, $w_{\text{prob}}=0.25$, $w_{\text{thr}}=0.20$, $w_{\text{role}}=0.10$) and link assertion $\mathrm{linked} = \mathbb{I}[\mathrm{link\_score} \ge 0.45]$. Boundary confidence is explicitly defined as an uncalibrated distance heuristic: $\mathrm{conf}_{\text{link}} = \mathrm{clip}(1 - 2|\mathrm{link\_score} - 0.45|,\, 0,\, 1)$.

---

### 1.9 Evaluation, Uncertainty & Inter-Annotator Agreement Metrics

1. **Macro F1:** $\mathrm{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}$.
2. **5-Bin Expected Calibration Error (ECE):** Partition predictions into $B = 5$ equal-width confidence bins $B_b$:

$$\mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N}\left|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\right|$$

3. **Thread-Clustered Bootstrap 95% CI ($B = 2{,}000$):** Resample entire thread clusters (`post_id`) with replacement rather than individual comments, preserving within-thread correlation when computing paired $\Delta \mathrm{F1}$ p-values.
4. **Cohen's $\kappa$ (Single-Label IAA):** $\kappa = \frac{p_o - p_e}{1 - p_e}$.
5. **MASI-Weighted Krippendorff's $\alpha$ (Multi-Label IAA):** Uses set distance $d_{\text{MASI}}(A, B) = 1 - M(A, B)\frac{|A \cap B|}{|A \cup B|}$ where monotonicity weight $M(A, B) \in \{1.0, 0.67, 0.33, 0.0\}$ penalizes partial vs. subset vs. disjoint multi-label overlaps: $\alpha = 1 - \frac{D_o}{D_e}$.

---

## 2. Complete End-to-End Pseudocode (Matching Actual Production Architecture)

```text
ALGORITHM RCI_PRODUCTION_PIPELINE(raw_reddit_archive, labeled_v1_csv)

INPUT:
  raw_reddit_archive   // Keyless Arctic Shift public posts + comments (8 subreddits)
  labeled_v1_csv       // 459 manually curated real comments across 200 distinct threads

OUTPUT:
  corpus.json, predictions.json, topic_discovery.json,
  community_intelligence.json, evaluation.json

BEGIN

  // STAGE 1: Ingestion, Hygiene & Zero-PII Pseudonymization
  posts <- FILTER_OUT(raw_reddit_archive, bots, stickies, deleted, gif_only, empty_threads)
  FOR EACH post IN posts:
    REDACT_PII(post.title, post.body)               // strip emails, phones, u/handles, reddit URLs
    post.id <- PSEUDONYMIZE_POST(post)              // post_0001 .. post_0365
    FOR EACH comment IN post.comments:
      REDACT_PII(comment.body)
      comment.id <- PSEUDONYMIZE_COMMENT(comment)   // cmt_00001 .. cmt_03557
      RESOLVE_PARENT_LINK(comment.parent_id, post.comments)
      lid <- INDIAN_REDDIT_LID(comment.body)        // en, hi, hi-en, bn-en, or-en, script, code_mixed
      comment.language, comment.script, comment.is_code_mixed <- lid
    END FOR
  END FOR

  // STAGE 2: Leak-Free Supervised Split & Strictly Inductive WWM DAPT
  (trainval_367, frozen_test_92) <- GROUP_SPLIT_BY_POST_ID(labeled_v1_csv, test_ratio=0.20)
  ASSERT INTERSECT(POST_IDS(trainval_367), POST_IDS(frozen_test_92)) == EMPTY

  excluded_threads <- POST_IDS(frozen_test_92) UNION SAME_TITLE_THREADS(frozen_test_92)
  dapt_pool <- [text FOR post IN posts IF post.id NOT IN excluded_threads]
  (dapt_train, dapt_val) <- GROUP_SPLIT_BY_POST_ID(dapt_pool, val_ratio=0.14)
  ASSERT INTERSECT(dapt_pool_threads, excluded_threads) == EMPTY

  muril_wwm <- TRAIN_WWM_MLM(base="google/muril-base-cased",
                             train=dapt_train, val=dapt_val, epochs=8, mask_prob=0.15)
  RECORD_HELD_OUT_PERPLEXITY_CURVE(muril_wwm, dapt_val)

  // STAGE 3: Unsupervised Topic Discovery (Task 7)
  X_tfidf <- FIT_SUBLINEAR_TFIDF(all_comments.body, max_features=8000, ngram=(1, 2))
  (W, H) <- FIT_NMF(X_tfidf, n_topics=10)
  WRITE topic_discovery.json (10 discovered topics + per-comment dominant topic)

  // STAGE 4: Apples-to-Apples 5-Fold Group-Aware CV & Per-Task Neural Training
  FOR EACH task IN [issue, sentiment, emotion, response, solution]:
    folds <- GROUP_CV_FOLDS(trainval_367, group_by="post_id", k=5)
             // StratifiedGroupKFold (single-label) or Iterative Group Stratification (multi-label)
    TRAIN_5FOLD_BASELINES(task, folds, [LogReg_TFIDF, ANN_MLP, TextCNN_1D])

    backbone, ctx_mode, loss_fn, prep_fn <- SELECT_OOF_WINNER_SPEC(task)
    FOR fold f = 1 .. 5:
      pairs_tr <- BUILD_INPUT_PAIRS(folds[f].train, ctx_mode, prep_fn)
      pairs_va <- BUILD_INPUT_PAIRS(folds[f].val,   ctx_mode, prep_fn)
      model_f  <- TRAIN_POOLED_TRANSFORMER(backbone, pairs_tr, loss_fn,
                                           pool="[CLS]+Mean+Max", msd=5, llrd=0.92)
      oof_logits[folds[f].val] <- INFER_LENGTH_SORTED(model_f, pairs_va)
      test_logits_f[f]         <- INFER_LENGTH_SORTED(model_f, frozen_test_92)
      corpus_logits_f[f]       <- INFER_LENGTH_SORTED(model_f, all_comments)
    END FOR

    ens_test_logits   <- MEAN(test_logits_f[1..5])
    ens_corpus_logits <- MEAN(corpus_logits_f[1..5])

    IF task IN {issue, emotion}:                    // ASL multi-label heads
      platt_params <- FIT_OOF_MONOTONIC_PLATT(oof_logits, y_trainval)
      oof_probs    <- APPLY_PLATT(oof_logits, platt_params)
      taus         <- TUNE_REGULARIZED_OOF_THRESHOLDS(oof_probs, y_trainval, [0.25, 0.70])
      test_probs   <- APPLY_PLATT(ens_test_logits, platt_params)
      corpus_probs <- APPLY_PLATT(ens_corpus_logits, platt_params)
    ELSE:                                           // Single-label CE-weighted heads
      T            <- FIT_NON_SHARPENING_TEMPERATURE(oof_logits, y_trainval, T >= 1.0)
      test_probs   <- SOFTMAX(ens_test_logits / T)
      corpus_probs <- SOFTMAX(ens_corpus_logits / T)
    END IF

    EVALUATE_ONCE_ON_FROZEN_TEST(task, test_probs, taus)
    COMPUTE_EMPIRICAL_ERROR_ANALYSIS_AND_5BIN_ECE(task, test_probs)
  END FOR

  // STAGE 5: Issue-Solution Linking, Clustering, Resolution & Unresolved Matrix (Tasks 10-11)
  FOR EACH post IN posts:
    post.resolution_state <- DETECT_EXPLICIT_FIRST_PERSON_OUTCOME(post.comments)
                             // reported_success | reported_pending | proposed_only | insufficient_evidence
  END FOR

  links <- TFIDF_COSINE_ISSUE_SOLUTION_LINKER(posts, predicted_solutions, same_post_only=TRUE, tau=0.45)
  solution_clusters <- CLUSTER_LINKED_SOLUTIONS_BY_ISSUE(links)
  unresolved_matrix <- APPLY_QUANTITATIVE_UNRESOLVED_RULES(posts, predictions, links)

  // STAGE 6: Cross-IIM, Cross-Language & Pipeline Integrity Gate
  cross_iim_stats  <- AGGREGATE_BY_IIM_ENTITY(posts, predictions)
  cross_lang_stats <- AGGREGATE_BY_LANGUAGE_REGISTER(predictions)
  integrity_report <- VERIFY_7_PIPELINE_INTEGRITY_CHECKS(corpus, predictions, evaluation)
  ASSERT integrity_report.passed_checks == 7

  WRITE corpus.json, predictions.json, community_intelligence.json, evaluation.json
  SERVE_READ_ONLY_FASTAPI_STUDIO(Docker + Vercel, <10 ms latency, 0 runtime GPU)

END
```

---

## 3. Exact Mathematics-to-Code Implementation Map

| Mathematical Component | Source File | Implementing Function / Class |
|---|---|---|
| Word+Char Sublinear TF-IDF | `backend/app/services/nlp/issue_identification/tfidf_logistic.py` & `ml/experiments/v5_optimization/run_task5_task6_ablations.py` | `TfidfVectorizer(sublinear_tf=True)` + `FeatureUnion` |
| Indian-Reddit Language & Script ID | `backend/app/services/nlp/indian_reddit_lid.py` | `detect_script_profile()`, `detect_indian_reddit_language()` |
| Emoji Semantic Affect & Gen-Z Slang | `backend/app/services/nlp/preprocessing/emoji_normalizer.py`, `slang_normalizer.py` | `normalize_emojis()`, `normalize_emojis_semantic_affect()`, `normalize_slang()`, `detect_slang_terms()` |
| `##`-Aligned WWM DAPT MLM | `ml/experiments/v5_optimization/run_dapt_inductive.py` | `WholeWordMaskingCollator`, `run_strict_inductive_wwm_dapt()` |
| `[CLS]+Mean+Max`, Reply-Only Mask & 5× MSD | `ml/experiments/v5_optimization/common.py` | `PooledTransformerClassifier.forward()` |
| LLRD ($\lambda = 0.92$) & Non-Sharpening $T \ge 1.0$ | `ml/models/transformer/muril_classifier.py` | `_build_llrd_param_groups()`, `_calibrate_temperature()` |
| Class-Weighted CE & Asymmetric Loss (ASL) | `ml/experiments/v5_optimization/common.py` | `AsymmetricLossMultiLabel`, `_make_criterion()` |
| Multi-Task R-Drop Symmetric KL | `ml/experiments/v5_optimization/common.py` & `run_stage3_tuning.py` | `_categorical_sym_kl()`, `_bernoulli_sym_kl()`, `Shared3TaskModel` |
| Group-Preserving 5-Fold CV | `ml/experiments/v5_optimization/common.py` | `make_group_cv_folds()` |
| OOF Monotonic Platt & Regularized Thresholds | `backend/build_live_corpus_artifacts.py` & `common.py` | `_fit_oof_platt()`, `tune_multilabel_thresholds()` |
| 10-Cluster NMF Topic Discovery | `backend/app/services/nlp/topic_discovery/nmf_topic_model.py` | `NmfTopicModel.fit_transform()` |
| Blended Cosine Issue–Solution Linker | `backend/app/services/nlp/issue_solution_linking/tfidf_cosine_linker.py` | `TfidfCosineIssueSolutionLinker.link()` |
| Solution Clustering, Unresolved Criteria & Q8 | `backend/app/services/nlp/community_intelligence.py` & `build_community_intelligence_artifact.py` | `build_unresolved_issue_matrix()`, `build_issue_solution_clusters()`, `build_cross_iim_comparison()`, `build_cross_language_comparison()` |
| 5-Bin ECE, Empirical Error Analysis & 7-Check Verifier | `ml/evaluation/error_analysis.py`, `ml/evaluation/benchmark_verifier.py` | `compute_calibration_metrics()`, `verify_benchmark_artifact()` |
| Thread-Clustered Bootstrap, Cohen's $\kappa$ & MASI $\alpha$ | `ml/experiments/v5_optimization/common.py`, `score_human_iaa.py` | `thread_clustered_bootstrap_ci()`, `masi_distance()`, `krippendorff_alpha()` |
