# Deliverable 5b — Complete RCI V6 System Pseudocode

> This pseudocode documents the **exact implemented V6 pipeline** in `D:\AI_Project` (module names and artifacts match actual repository files).

---

```text
================================================================================
ALGORITHM 1: RCI V6 END-TO-END COMMUNITY INTELLIGENCE PIPELINE
================================================================================
INPUT:
  Corpus C = {posts P_1..P_365, comments c_1..c_3557 with parent_reddit_id, created_at}
  Preserved NMF Artifact A_nmf = ml/artifacts/live_corpus/topic_discovery.json (READ-ONLY)
  Supervised 5-Task Predictions Y_hat = ml/artifacts/live_corpus/predictions.json
  Strict OOF Evaluation Ledger E_eval = ml/artifacts/live_corpus/evaluation.json

--------------------------------------------------------------------------------
STAGE 1: MULTILINGUAL LANGUAGE, SCRIPT, SPAN & NON-DESTRUCTIVE NORMALIZATION
Module: backend/app/services/nlp/indian_reddit_lid.py
        backend/app/services/nlp/preprocessing/context_normalizer.py
--------------------------------------------------------------------------------
FOR each comment c in C:
  1. original_text <- c.body   // INVARIANT: NEVER MUTATED
  2. script_profile <- detect_script_profile(original_text)
       // Counts Latin, Devanagari, Bengali, Odia, etc. Unicode codepoints
  3. spans <- []
     FOR each sentence/newline/danda segment s with [start, end) in original_text:
       s_res <- classify_segment(s)
         // Native Indic script -> {hi|bn|or} or {lang-en} if mixed with Latin
         // Romanized Indic markers with proper-noun guard ("Bodh Gaya" != Hindi "gaya")
         // Pure Romanized Indic requires len(english_anchors) == 0
         // Low-evidence numeric/emoji/terse -> language="und", abstain=True
       spans.append({text: s, start, end, ...s_res})
  4. lid[c] <- reconcile_document_and_spans(classify_segment(original_text), spans)
       // Never forces multi-span code-switched comments into one monolingual label
  5. roman_norm[c] <- normalize_romanized_indic(original_text)
       // Separate normalized string + transformations[] metadata
  6. informal_norm[c] <- apply_informal_normalization(original_text, mode)
       // Context guards: "mid sem" / "based on" / "cooked food" preserved;
       // Gen-Z "is mid" -> "mediocre", "we are cooked" -> "overwhelmed doomed";
       // Emojis verbalized to affect phrases (never stripped by default)

--------------------------------------------------------------------------------
STAGE 2: DUAL TOPIC DISCOVERY (PRESERVED NMF v1 + COMPANION BERTopic v1)
Module: ml/topics/bertopic/{embeddings.py, pipeline.py, evaluation.py, registry.py}
--------------------------------------------------------------------------------
1. LOAD eligible_docs (n = 3,459 AutoMod-filtered comments) from A_nmf (READ-ONLY)
2. EVALUATE NMF v1 baseline on eligible_docs -> metrics_nmf (NEVER overwrite A_nmf)
3. FOR embedding_key in {multilingual_e5_small, multilingual_minilm, muril_dapt_wwm}:
     E_mat <- encode_documents(eligible_docs, embedding_key, L2_normalize=True, cache=True)
     umap_model <- UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", seed=42)
     hdbscan_model <- HDBSCAN(min_cluster_size=25, min_samples=8, method="eom", prediction_data=True)
     vec_model <- CountVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0,
                                  token_pattern=INDIC_MULTILINGUAL_TOKEN_PATTERN)
     ctfidf_model <- ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
     model <- BERTopic(umap_model, hdbscan_model, vec_model, ctfidf_model, nr_topics=None)
     topics, probs <- model.fit_transform(eligible_docs, embeddings=E_mat)
     // Explicitly preserve -1 outliers; never force K=10
     metrics <- evaluate_topic_model(eligible_docs, topics, keywords, languages, E_mat)
     REGISTER run in ml/artifacts/topic_models/registry.json
4. SELECT bertopic_v1 via multi-metric profile (NPMI, UMass, diversity, centroid cosine,
   size entropy balance, outlier rate, multilingual probe accuracy) -> bertopic_e5_small_raw
5. RUN 3-seed (42, 101, 202) & 80%-subsample stability checks (ARI, NMI)
6. EXTRACT hierarchical_topics(eligible_docs) + human_review_sample.csv
7. SAVE SafeTensors model -> ml/artifacts/topic_models/bertopic_v1/
   VERIFY load() + transform() round-trip
   EMIT compact serving JSON -> ml/artifacts/live_corpus/bertopic_v1_topics.json

--------------------------------------------------------------------------------
STAGE 3: STRICTLY INDUCTIVE 5-TASK SUPERVISED CLASSIFICATION (OFFLINE TRAINING)
Module: ml/experiments/v5_optimization/ + backend/build_live_corpus_artifacts.py
--------------------------------------------------------------------------------
1. EXCLUDE all 44 frozen-test & same-title threads (749 comments) from DAPT -> 0 overlap
2. TRAIN ##-aligned WWM DAPT on 2,135 unlabeled train docs (8 epochs, RTX 4060 bf16)
3. PARTITION trainval_v1 (n = 367) into 5 post_id-grouped folds (zero thread leakage)
4. FOR task in {issue, sentiment, emotion, response, solution}:
     IF task == "issue":
       input_format <- [CLS] comment [SEP] parent_or_title [SEP] (max_len=256, only_second)
       backbone <- muril_dapt_strict_inductive_wwm_8ep; loss <- ASL(gamma_neg=2, m=0.05)
     ELSE IF task in {"sentiment", "emotion"}:
       input_format <- [CLS] comment [SEP]
       backbone <- muril_dapt_strict_inductive_wwm_8ep
       loss <- CE_weighted (sentiment) / ASL (emotion)
     ELSE IF task in {"response", "solution"}:
       input_format <- [CLS] comment [SEP]
       backbone <- muril_dapt_wwm or xlm-roberta-base (OOF-selected); loss <- CE_weighted
     COLLECT 5-fold OOF logits -> FIT bounded thresholds [0.30,0.70] & OOF Platt calibration
     // STRICT GOVERNANCE: Report OOF-only selection scores on frozen test (n=92)
5. INFER over all 3,557 corpus comments -> ml/artifacts/live_corpus/predictions.json

--------------------------------------------------------------------------------
STAGE 4: CONTEXTUAL MULTILINGUAL ISSUE-SOLUTION MAPPING & CLUSTERING
Module: ml/linking/contextual_mapper.py
--------------------------------------------------------------------------------
1. candidates <- []
   FOR each post P and each comment c in P:
     IF Y_hat.solution[c].is_solution OR Y_hat.response[c].type in {solution, advice}
        OR Y_hat.solution[c].prob >= 0.42:
       FOR each issue I in (Y_hat.issue[c].labels UNION {P.issue_key}):
         candidates.append((I, P, c, parent_context(c)))
2. ENCODE query_emb  <- E5("query: Issue " + desc[I] + " | Thread: " + P.title + " | Parent: " + ctx)
   ENCODE sol_emb    <- E5("passage: " + c.body)
   COMPUTE tfidf_cos <- TFIDF_cosine(P.title + ctx, c.body)
3. FOR each candidate (I, P, c):
     s_emb    <- clip((dot(query_emb, sol_emb) - 0.68) / 0.24, 0, 1)
     s_lex    <- min(1, 2.2 * tfidf_cos)
     s_struct <- 0.45 + 0.30*I[I in pred_issues] + 0.15*I[direct_reply_to_op] + 0.10*I[action_cues]
     s_det    <- 0.60 * sol_prob + 0.40 * ROLE_WEIGHT[response_type]
     p_neg    <- negation_contradiction_and_placeholder_penalty(c.body, response_type)
     score    <- clip(0.38*s_emb + 0.14*s_lex + 0.18*s_struct + 0.30*s_det - p_neg, 0, 1)
     linked   <- (score >= 0.56) AND (p_neg < 0.28)
     EMIT 9-field record with verification_status = "model_inferred" + signal_breakdown
4. FOR each issue domain I:
     CLUSTER linked solution embeddings via AgglomerativeClustering(metric="cosine",
             linkage="average", distance_threshold=0.14) -> semantic_solution_clusters
5. SAVE -> ml/artifacts/live_corpus/issue_solution_map_v2.json

--------------------------------------------------------------------------------
STAGE 5: EXPLAINABLE UNRESOLVED ISSUE DETECTION
Module: ml/linking/resolution_engine.py
--------------------------------------------------------------------------------
FOR each thread P in C:
  1. linked_sols <- {s in Stage4.linked : s.post_id == P.reddit_id}
  2. SCAN P.title, P.body, and all comments c in P for:
       success_ev       <- matches(REPORTED_SUCCESS_RE) AND NOT matches(QUESTION_GUARD_RE)
       pending_ev       <- matches(REPORTED_PENDING_RE)
       proc_failure_ev  <- matches(PROCEDURAL_FAILURE_RE)    // e.g. "didn't respond", "not picking up"
       rhet_dispute_ev  <- matches(RHETORICAL_DISPUTE_RE)    // e.g. "fake news", "misleading"
       counter_sol_ev   <- {c : c.parent_id in linked_sols AND response_type == "counterargument"}
       confirm_ev       <- {c : c.parent_id in linked_sols AND matches(CONFIRM_RE | agreement)}
       contradictory_ev <- proc_failure_ev UNION rhet_dispute_ev UNION counter_sol_ev
  3. ASSIGN canonical 4-state resolution_state:
       IF |success_ev| > 0 AND |proc_failure_ev| == 0  -> "reported_success"
       ELSE IF |pending_ev| > 0 OR |proc_failure_ev| > 0 -> "reported_pending"
       ELSE IF |linked_sols| > 0                       -> "proposed_only"
       ELSE                                            -> "insufficient_evidence"
  4. COMPUTE explainable score S_resolution (capped at 0.58 if |success_ev| == 0;
     discussion volume NEVER increases S_resolution, only confidence volume guard)
  5. MAP to Verdict in {Resolved, Partially Resolved, Unresolved, Insufficient Evidence}
  6. ATTACH full evidence trail (issue, proposed_solution, reported_outcome, follow_up_evidence,
     contradictory_evidence, timestamps, confidence, plain-English explanation)
SAVE -> ml/artifacts/live_corpus/resolution_engine_v2.json

--------------------------------------------------------------------------------
STAGE 6: EVIDENCE-LINKED CROSS-MODEL INSIGHT ENGINE & ZERO-ML SERVING
Module: backend/app/services/nlp/cross_model_insights.py + backend/app/api/demo.py
--------------------------------------------------------------------------------
1. LOAD compact read-only artifacts in pure Python stdlib (zero ML libs -> Vercel-safe)
2. SYNTHESIZE 6 cross-model insight cards joining NMF + BERTopic + 5 classifiers +
   Phase 4 matched solutions + Phase 5 contradictory/resolution trails + Phase 6 strict OOF audit
3. ATTACH to every card: separated {observation, interpretation}, supporting_posts[],
   supporting_comments[], topic_refs[], issue_refs[], model_confidence, evidence_limitations
4. SERVE via /api/demo/* to the 9-view native ES Module Studio SPA
================================================================================
```
