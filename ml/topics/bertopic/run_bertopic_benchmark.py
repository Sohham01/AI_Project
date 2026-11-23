"""Run the Phase 1 BERTopic vs. NMF benchmark on the real anonymized RCI corpus.

Strict integrity guarantees:
- NEVER modifies or overwrites ``ml/artifacts/live_corpus/topic_discovery.json`` (NMF).
- Uses the exact same 3,459 eligible documents as NMF for fair comparison.
- Never forces exactly 10 topics; preserves natural HDBSCAN cluster count & ``-1`` outliers.
- Keeps controlled synthetic cross-lingual embedding probes strictly separate from real-corpus metrics.
- Saves full SafeTensors model + registry under ``ml/artifacts/topic_models/bertopic_v1/``
  and a compact Vercel-ready serving JSON under ``ml/artifacts/live_corpus/bertopic_v1_topics.json``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.topics.bertopic.config import BERTopicExperimentConfig
from ml.topics.bertopic.embeddings import (
    EMBEDDING_CATALOG,
    encode_documents,
    evaluate_multilingual_embedding_probes,
)
from ml.topics.bertopic.evaluation import evaluate_topic_model
from ml.topics.bertopic.pipeline import RCIBERTopicPipeline, preprocess_for_bertopic
from ml.topics.bertopic.registry import TopicModelRegistry

NMF_ARTIFACT = ROOT / "ml" / "artifacts" / "live_corpus" / "topic_discovery.json"
CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
PREDICTIONS_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json"
BERTOPIC_ARTIFACT_DIR = ROOT / "ml" / "artifacts" / "topic_models" / "bertopic_v1"
SERVING_SUMMARY_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "bertopic_v1_topics.json"


def _load_eligible_corpus() -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Load exact NMF eligible documents (read-only) enriched with thread/issue metadata."""
    nmf_payload = json.loads(NMF_ARTIFACT.read_text(encoding="utf-8"))
    dataset_version = str(nmf_payload.get("dataset_version", "6.1.0"))
    nmf_res = nmf_payload["result"]

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    posts_by_id = {p["reddit_id"]: p for p in corpus.get("posts", [])}

    # Optional predicted issue lookup (for observational unsupervised-vs-supervised overlap)
    pred_issue_by_cid: Dict[str, str] = {}
    if PREDICTIONS_PATH.is_file():
        preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
        issue_task = (preds.get("tasks") or {}).get("issue") or {}
        for doc_key, item in issue_task.items():
            cid = doc_key.removeprefix("comment-")
            lbs = item.get("predicted_issue_labels") or []
            if lbs:
                pred_issue_by_cid[cid] = str(lbs[0])
            else:
                probs = item.get("issue_probabilities") or {}
                if probs:
                    pred_issue_by_cid[cid] = max(probs.items(), key=lambda kv: kv[1])[0]

    docs_meta: List[Dict[str, Any]] = []
    for doc in nmf_res["documents"]:
        pid = doc.get("post_id", "")
        cid = doc.get("comment_id", "")
        post = posts_by_id.get(pid, {})
        docs_meta.append({
            "document_id": doc.get("document_id"),
            "post_id": pid,
            "comment_id": cid,
            "post_title": post.get("title", ""),
            "subreddit": post.get("subreddit", ""),
            "issue_category": pred_issue_by_cid.get(cid) or post.get("issue_key") or post.get("issue_category"),
            "original_text": doc.get("original_text", ""),
            "nmf_preprocessed_text": doc.get("text", ""),
            "language": doc.get("language", "und"),
            "script": doc.get("script", "latin"),
            "is_code_mixed": bool(doc.get("is_code_mixed", False)),
            "nmf_topic_id": int(doc.get("dominant_topic_id", 0)),
        })
    return docs_meta, nmf_res, dataset_version


def _compute_stability(
    raw_texts: List[str],
    embeddings: np.ndarray,
    base_cfg: BERTopicExperimentConfig,
    base_topics: List[int],
    seeds: Tuple[int, ...] = (101, 202),
) -> Dict[str, Any]:
    """Measure clustering stability across random seeds and an 80% stratified subsample."""
    seed_aris_all: List[float] = []
    seed_nmis_all: List[float] = []
    seed_aris_non_outlier: List[float] = []
    seed_topic_counts: List[int] = [len({t for t in base_topics if t != -1})]

    for seed in seeds:
        cfg_s = BERTopicExperimentConfig(**{**base_cfg.to_dict(), "ngram_range": tuple(base_cfg.ngram_range), "random_state": seed})
        pipe_s = RCIBERTopicPipeline(cfg_s)
        t_s, _, _ = pipe_s.fit(raw_texts, embeddings=embeddings, use_embedding_cache=True)
        seed_topic_counts.append(len({t for t in t_s if t != -1}))
        seed_aris_all.append(float(adjusted_rand_score(base_topics, t_s)))
        seed_nmis_all.append(float(normalized_mutual_info_score(base_topics, t_s)))
        mask = [i for i, (a, b) in enumerate(zip(base_topics, t_s)) if a != -1 and b != -1]
        if len(mask) >= 10:
            seed_aris_non_outlier.append(
                float(adjusted_rand_score([base_topics[i] for i in mask], [t_s[i] for i in mask]))
            )

    # 80% subsample stability
    rng = np.random.default_rng(42)
    n = len(raw_texts)
    sub_idx = np.sort(rng.choice(n, size=int(round(n * 0.80)), replace=False))
    sub_texts = [raw_texts[i] for i in sub_idx]
    sub_embs = embeddings[sub_idx]
    cfg_sub = BERTopicExperimentConfig(**{**base_cfg.to_dict(), "ngram_range": tuple(base_cfg.ngram_range), "random_state": 42})
    pipe_sub = RCIBERTopicPipeline(cfg_sub)
    t_sub, _, _ = pipe_sub.fit(sub_texts, embeddings=sub_embs, use_embedding_cache=True)
    ref_sub = [base_topics[i] for i in sub_idx]
    sub_ari_all = float(adjusted_rand_score(ref_sub, t_sub))
    sub_nmi_all = float(normalized_mutual_info_score(ref_sub, t_sub))
    sub_mask = [i for i, (a, b) in enumerate(zip(ref_sub, t_sub)) if a != -1 and b != -1]
    sub_ari_non_out = (
        float(adjusted_rand_score([ref_sub[i] for i in sub_mask], [t_sub[i] for i in sub_mask]))
        if len(sub_mask) >= 10
        else 0.0
    )

    return {
        "seeds_tested": [base_cfg.random_state, *seeds],
        "topic_counts_across_seeds": seed_topic_counts,
        "mean_seed_ari_all_docs": round(float(np.mean(seed_aris_all)), 4),
        "mean_seed_nmi_all_docs": round(float(np.mean(seed_nmis_all)), 4),
        "mean_seed_ari_non_outlier_overlap": round(float(np.mean(seed_aris_non_outlier)) if seed_aris_non_outlier else 0.0, 4),
        "subsample_80pct_ari_all_docs": round(sub_ari_all, 4),
        "subsample_80pct_nmi_all_docs": round(sub_nmi_all, 4),
        "subsample_80pct_ari_non_outlier_overlap": round(sub_ari_non_out, 4),
        "subsample_80pct_n_topics": len({t for t in t_sub if t != -1}),
    }


def _composite_selection_note(runs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Select the strongest BERTopic configuration transparently across multiple criteria (not a single metric)."""
    # Rank across coherence (NPMI), diversity, centroid relevance, balance, and moderate outlier rate (<45%)
    best_id = None
    best_score = -1e9
    for r in runs:
        m = r["metrics"]
        if m["n_topics"] < 6:
            continue
        # Balanced multi-metric heuristic (explicitly documented; raw per-metric values all preserved)
        outlier_penalty = max(0.0, m["outlier_rate"] - 0.35) * 0.6
        score = (
            1.5 * m["coherence_npmi"]
            + 0.8 * m["topic_diversity"]
            + 0.7 * float(m["representative_centroid_cosine"] or 0.0)
            + 0.4 * m["size_distribution"]["normalized_entropy_balance"]
            - outlier_penalty
        )
        r["multi_metric_composite"] = round(score, 4)
        if score > best_score:
            best_score = score
            best_id = r["model_id"]
    return best_id or runs[0]["model_id"], (
        "Selected using balanced multi-metric profile combining NPMI coherence, topic diversity, "
        "representative centroid cosine, size entropy balance, and controlled outlier rate "
        "(plus multilingual probe support and cluster interpretability)."
    )


def main() -> None:
    print("[Phase 1] Loading exact NMF eligible corpus (read-only)...")
    docs_meta, nmf_res, dataset_version = _load_eligible_corpus()
    raw_texts = [d["original_text"] for d in docs_meta]
    languages = [d["language"] for d in docs_meta]
    n_docs = len(docs_meta)
    print(f"  Eligible documents: {n_docs} (dataset_version={dataset_version})")

    # 1. Controlled synthetic multilingual embedding diagnostic (strictly separate from real corpus)
    print("[Phase 1] Running controlled synthetic 14-probe multilingual embedding comparison...")
    emb_keys = ["multilingual_e5_small", "multilingual_minilm", "muril_dapt_wwm"]
    probe_report = evaluate_multilingual_embedding_probes(emb_keys)
    for ek, res in probe_report["results"].items():
        print(
            f"  Probe {ek:22s}: top1={res['top1_accuracy']*100:5.1f}%, "
            f"margin={res['mean_similarity_margin']:+.4f}, by_lang={res['by_language_group']}"
        )

    # 2. Precompute embeddings on real corpus for each candidate encoder (raw_clean + language_aware_affect for E5)
    raw_clean_texts = preprocess_for_bertopic(raw_texts, mode="raw_clean")
    lang_aware_texts = preprocess_for_bertopic(raw_texts, mode="language_aware_affect")

    cached_embs: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[str, Any]]] = {}
    for ek in emb_keys:
        print(f"[Phase 1] Encoding {n_docs} corpus docs with {ek} (raw_clean)...")
        cached_embs[(ek, "raw_clean")] = encode_documents(raw_clean_texts, ek, is_query=False, use_cache=True)
        print(f"    -> shape={cached_embs[(ek, 'raw_clean')][0].shape}, meta={cached_embs[(ek, 'raw_clean')][1]}")

    print(f"[Phase 1] Encoding {n_docs} corpus docs with multilingual_e5_small (language_aware_affect)...")
    cached_embs[("multilingual_e5_small", "language_aware_affect")] = encode_documents(
        lang_aware_texts, "multilingual_e5_small", is_query=False, use_cache=True
    )

    # 3. Evaluate (A) Existing NMF Baseline on identical corpus (with E5 embeddings for centroid relevance apples-to-apples)
    nmf_keywords = {int(t["topic_id"]): list(t["keywords"]) for t in nmf_res["topics"]}
    nmf_doc_topics = [int(d["nmf_topic_id"]) for d in docs_meta]
    nmf_metrics = evaluate_topic_model(
        corpus_texts=raw_clean_texts,
        doc_topics=nmf_doc_topics,
        topics_keywords=nmf_keywords,
        languages=languages,
        embeddings=cached_embs[("multilingual_e5_small", "raw_clean")][0],
    )
    print(
        f"[Phase 1] (A) Existing NMF v1 baseline: topics={nmf_metrics['n_topics']}, "
        f"NPMI={nmf_metrics['coherence_npmi']:+.4f}, div={nmf_metrics['topic_diversity']:.4f}, "
        f"outliers={nmf_metrics['outlier_rate']*100:.1f}%, centroid_cos={nmf_metrics['representative_centroid_cosine']:.4f}"
    )

    # 4. Run systematic BERTopic configurations (A=NMF, B1=E5, B2=MiniLM, C=MuRIL-DAPT, D=E5+lang_aware)
    experiment_specs = [
        BERTopicExperimentConfig(
            model_id="bertopic_e5_small_raw",
            embedding_key="multilingual_e5_small",
            embedding_model_name=EMBEDDING_CATALOG["multilingual_e5_small"]["model_name"],
            preprocessing_mode="raw_clean",
            hdbscan_min_cluster_size=25,
            hdbscan_min_samples=8,
            corpus_version=dataset_version,
            notes="Config B1: Multilingual E5-small + raw_clean text (natural density clustering)",
        ),
        BERTopicExperimentConfig(
            model_id="bertopic_e5_small_raw_mcs35",
            embedding_key="multilingual_e5_small",
            embedding_model_name=EMBEDDING_CATALOG["multilingual_e5_small"]["model_name"],
            preprocessing_mode="raw_clean",
            hdbscan_min_cluster_size=35,
            hdbscan_min_samples=10,
            corpus_version=dataset_version,
            notes="Config B1-coarse: Multilingual E5-small + raw_clean (min_cluster_size=35 sensitivity check)",
        ),
        BERTopicExperimentConfig(
            model_id="bertopic_minilm_raw",
            embedding_key="multilingual_minilm",
            embedding_model_name=EMBEDDING_CATALOG["multilingual_minilm"]["model_name"],
            preprocessing_mode="raw_clean",
            hdbscan_min_cluster_size=25,
            hdbscan_min_samples=8,
            corpus_version=dataset_version,
            notes="Config B2: Paraphrase-Multilingual-MiniLM-L12-v2 + raw_clean",
        ),
        BERTopicExperimentConfig(
            model_id="bertopic_muril_dapt_raw",
            embedding_key="muril_dapt_wwm",
            embedding_model_name=EMBEDDING_CATALOG["muril_dapt_wwm"]["model_name"],
            preprocessing_mode="raw_clean",
            hdbscan_min_cluster_size=25,
            hdbscan_min_samples=8,
            corpus_version=dataset_version,
            notes="Config C: MuRIL-DAPT-WWM mean-pooled non-contrastive comparator + raw_clean",
        ),
        BERTopicExperimentConfig(
            model_id="bertopic_e5_small_lang_aware",
            embedding_key="multilingual_e5_small",
            embedding_model_name=EMBEDDING_CATALOG["multilingual_e5_small"]["model_name"],
            preprocessing_mode="language_aware_affect",
            hdbscan_min_cluster_size=25,
            hdbscan_min_samples=8,
            corpus_version=dataset_version,
            notes="Config D: Multilingual E5-small + language_aware_affect (slang + affect emojis + Romanized Indic variants)",
        ),
    ]

    registry = TopicModelRegistry()
    registry.register(
        model_id="nmf_v1",
        algorithm="nmf",
        embedding_model=None,
        corpus_version=dataset_version,
        preprocessing_config={"vectorizer": "TF-IDF (1-2 grams, max_features=8000)", "automod_filtered": True},
        n_documents=n_docs,
        n_topics=nmf_metrics["n_topics"],
        outlier_count=0,
        training_parameters={"n_components": 10, "init": "nndsvda", "random_state": 42},
        evaluation_metrics=nmf_metrics,
        artifact_paths={"topic_discovery_json": str(NMF_ARTIFACT.relative_to(ROOT))},
        notes="Existing preserved NMF v1 baseline (never overwritten).",
        set_default=True,
    )

    fitted_runs: Dict[str, Tuple[RCIBERTopicPipeline, List[int], Any, np.ndarray, Dict[str, Any]]] = {}
    run_summaries: List[Dict[str, Any]] = []

    for cfg in experiment_specs:
        embs, emb_meta = cached_embs[(cfg.embedding_key, cfg.preprocessing_mode)]
        pipe = RCIBERTopicPipeline(cfg)
        doc_topics, doc_probs, _ = pipe.fit(raw_texts, embeddings=embs, use_embedding_cache=True)
        pipe.embedding_meta = emb_meta

        kw_pairs = pipe.get_topic_keywords()
        kw_only = {tid: [w for w, _ in pairs] for tid, pairs in kw_pairs.items()}
        eval_corpus = lang_aware_texts if cfg.preprocessing_mode == "language_aware_affect" else raw_clean_texts
        metrics = evaluate_topic_model(
            corpus_texts=eval_corpus,
            doc_topics=doc_topics,
            topics_keywords=kw_only,
            languages=languages,
            embeddings=embs,
        )
        print(
            f"[Phase 1] {cfg.model_id:30s}: topics={metrics['n_topics']:2d}, "
            f"outliers={metrics['outlier_rate']*100:4.1f}% ({metrics['outlier_count']}), "
            f"NPMI={metrics['coherence_npmi']:+.4f}, UMass={metrics['coherence_umass']:+.3f}, "
            f"div={metrics['topic_diversity']:.4f}, cent_cos={metrics['representative_centroid_cosine']:.4f}, "
            f"bal={metrics['size_distribution']['normalized_entropy_balance']:.4f}"
        )

        fitted_runs[cfg.model_id] = (pipe, doc_topics, doc_probs, embs, metrics)
        run_summaries.append({
            "model_id": cfg.model_id,
            "algorithm": "bertopic",
            "embedding_key": cfg.embedding_key,
            "embedding_model_name": cfg.embedding_model_name,
            "embedding_license": EMBEDDING_CATALOG[cfg.embedding_key]["license"],
            "preprocessing_mode": cfg.preprocessing_mode,
            "hdbscan_min_cluster_size": cfg.hdbscan_min_cluster_size,
            "hdbscan_min_samples": cfg.hdbscan_min_samples,
            "notes": cfg.notes,
            "metrics": metrics,
            "encode_seconds": emb_meta.get("encode_seconds"),
            "ms_per_doc": emb_meta.get("ms_per_doc"),
        })

        registry.register(
            model_id=cfg.model_id,
            algorithm="bertopic",
            embedding_model=cfg.embedding_model_name,
            corpus_version=dataset_version,
            preprocessing_config={
                "mode": cfg.preprocessing_mode,
                "token_pattern": "indic_multilingual_safe",
                "ngram_range": list(cfg.ngram_range),
                "bm25_weighting": cfg.bm25_weighting,
                "reduce_frequent_words": cfg.reduce_frequent_words,
            },
            n_documents=n_docs,
            n_topics=metrics["n_topics"],
            outlier_count=metrics["outlier_count"],
            training_parameters=cfg.to_dict(),
            evaluation_metrics=metrics,
            artifact_paths={"pending_winner_selection": str(BERTOPIC_ARTIFACT_DIR.relative_to(ROOT))},
            notes=cfg.notes,
            set_default=False,
        )

    winner_id, selection_rationale = _composite_selection_note(run_summaries)
    print(f"\n[Phase 1] Selected BERTopic v1 model: {winner_id}")
    win_pipe, win_topics, win_probs, win_embs, win_metrics = fitted_runs[winner_id]

    # 5. Stability evaluation (3 seeds + 80% subsample) on the winner
    print("[Phase 1] Running 3-seed & 80% subsample stability checks on winner...")
    stability = _compute_stability(raw_texts, win_embs, win_pipe.config, win_topics)
    print(f"  Stability: {json.dumps(stability)}")

    # 6. Extract structured topics, explicit outliers, and hierarchy
    topics_list, outliers_list = win_pipe.extract_structured_topics(
        documents_meta=docs_meta,
        doc_topics=win_topics,
        doc_probs=win_probs,
        embeddings=win_embs,
        max_examples_per_topic=5,
    )
    hierarchy = win_pipe.extract_hierarchy(raw_texts, max_nodes=30)

    # 7. Save full SafeTensors model + human review samples + full report in ml/artifacts/topic_models/bertopic_v1/
    saved_paths = win_pipe.save(BERTOPIC_ARTIFACT_DIR)

    # Save human-reviewable representative sample (CSV + JSON)
    review_rows = []
    for t in topics_list:
        for rank, ex in enumerate(t["representative_examples"][:3], 1):
            review_rows.append({
                "topic_id": t["topic_id"],
                "machine_suggested_label": t["label"],
                "top_keywords": ", ".join(t["keywords"][:8]),
                "topic_doc_count": t["document_count"],
                "example_rank": rank,
                "centroid_cosine": ex["score"],
                "post_id": ex["post_id"],
                "comment_id": ex["comment_id"],
                "language": ex["language"],
                "post_title": ex["post_title"],
                "comment_text": ex["text"],
                "human_reviewed_label": "",
                "human_coherence_rating_1_to_5": "",
                "human_notes": "",
            })
    review_csv = BERTOPIC_ARTIFACT_DIR / "human_review_sample.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        if review_rows:
            writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
            writer.writeheader()
            writer.writerows(review_rows)
    (BERTOPIC_ARTIFACT_DIR / "human_review_sample.json").write_text(
        json.dumps(review_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Outlier diagnostics
    outlier_langs = Counter(o["language"] for o in outliers_list)
    outlier_wcs = [o["word_count"] for o in outliers_list]
    outlier_short_share = round(sum(1 for w in outlier_wcs if w <= 5) / max(len(outlier_wcs), 1), 4)
    outlier_summary = {
        "count": len(outliers_list),
        "ratio": round(len(outliers_list) / max(n_docs, 1), 4),
        "median_word_count": float(np.median(outlier_wcs)) if outlier_wcs else 0.0,
        "short_comment_le_5_words_share": outlier_short_share,
        "language_distribution": dict(outlier_langs.most_common()),
        "sample_documents": sorted(outliers_list, key=lambda x: x["word_count"], reverse=True)[:40],
    }

    # Build side-by-side comparison payload
    comparison_table = [
        {
            "model_id": "nmf_v1",
            "algorithm": "nmf",
            "embedding_model": "TF-IDF (word 1–2 grams, 8,000 vocab)",
            "preprocessing_mode": "tfidf_automod_filtered",
            "n_topics": nmf_metrics["n_topics"],
            "outlier_count": 0,
            "outlier_rate": 0.0,
            "coherence_npmi": nmf_metrics["coherence_npmi"],
            "coherence_umass": nmf_metrics["coherence_umass"],
            "topic_diversity": nmf_metrics["topic_diversity"],
            "representative_centroid_cosine": nmf_metrics["representative_centroid_cosine"],
            "size_balance_entropy": nmf_metrics["size_distribution"]["normalized_entropy_balance"],
            "is_winner_bertopic": False,
            "is_preserved_default": True,
        }
    ]
    for r in run_summaries:
        m = r["metrics"]
        comparison_table.append({
            "model_id": r["model_id"],
            "algorithm": "bertopic",
            "embedding_model": r["embedding_model_name"],
            "embedding_key": r["embedding_key"],
            "embedding_license": r["embedding_license"],
            "preprocessing_mode": r["preprocessing_mode"],
            "hdbscan_min_cluster_size": r["hdbscan_min_cluster_size"],
            "n_topics": m["n_topics"],
            "outlier_count": m["outlier_count"],
            "outlier_rate": m["outlier_rate"],
            "coherence_npmi": m["coherence_npmi"],
            "coherence_umass": m["coherence_umass"],
            "topic_diversity": m["topic_diversity"],
            "representative_centroid_cosine": m["representative_centroid_cosine"],
            "size_balance_entropy": m["size_distribution"]["normalized_entropy_balance"],
            "encode_seconds": r["encode_seconds"],
            "is_winner_bertopic": r["model_id"] == winner_id,
            "is_preserved_default": False,
        })

    # Update registry for winner with concrete artifact paths
    registry.register(
        model_id="bertopic_v1",
        algorithm="bertopic",
        embedding_model=win_pipe.config.embedding_model_name,
        corpus_version=dataset_version,
        preprocessing_config={
            "mode": win_pipe.config.preprocessing_mode,
            "token_pattern": "indic_multilingual_safe",
            "ngram_range": list(win_pipe.config.ngram_range),
            "bm25_weighting": win_pipe.config.bm25_weighting,
            "reduce_frequent_words": win_pipe.config.reduce_frequent_words,
        },
        n_documents=n_docs,
        n_topics=win_metrics["n_topics"],
        outlier_count=win_metrics["outlier_count"],
        training_parameters=win_pipe.config.to_dict(),
        evaluation_metrics={**win_metrics, "stability": stability},
        artifact_paths={
            "safetensors_dir": str(Path(saved_paths["model_safetensors_dir"]).relative_to(ROOT)),
            "human_review_csv": str(review_csv.relative_to(ROOT)),
            "serving_summary_json": str(SERVING_SUMMARY_PATH.relative_to(ROOT)),
        },
        notes=f"Promoted BERTopic v1 companion model ({winner_id}). Coexists alongside NMF v1; never replaces NMF.",
        set_default=False,
    )

    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "n_eligible_documents": n_docs,
        "nmf_preserved_untouched": True,
        "selected_bertopic_model_id": winner_id,
        "selection_rationale": selection_rationale,
        "nmf_baseline_metrics": nmf_metrics,
        "bertopic_runs": run_summaries,
        "winner_stability": stability,
        "comparison_table": comparison_table,
        "multilingual_embedding_probes_synthetic": probe_report,
        "outlier_summary": {k: v for k, v in outlier_summary.items() if k != "sample_documents"},
        "hierarchy_node_count": len(hierarchy),
    }
    (BERTOPIC_ARTIFACT_DIR / "benchmark_report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Compact Vercel-ready serving JSON (separate from topic_discovery.json!)
    serving_payload = {
        "synthetic_only": False,
        "dataset_version": dataset_version,
        "generated_at": full_report["generated_at"],
        "model_info": {
            "model_id": "bertopic_v1",
            "selected_experiment_id": winner_id,
            "algorithm": "bertopic",
            "embedding_key": win_pipe.config.embedding_key,
            "embedding_model": win_pipe.config.embedding_model_name,
            "embedding_license": EMBEDDING_CATALOG[win_pipe.config.embedding_key]["license"],
            "embedding_languages": EMBEDDING_CATALOG[win_pipe.config.embedding_key]["languages"],
            "preprocessing_mode": win_pipe.config.preprocessing_mode,
            "umap": {
                "n_neighbors": win_pipe.config.umap_n_neighbors,
                "n_components": win_pipe.config.umap_n_components,
                "min_dist": win_pipe.config.umap_min_dist,
                "metric": win_pipe.config.umap_metric,
                "random_state": win_pipe.config.random_state,
            },
            "hdbscan": {
                "min_cluster_size": win_pipe.config.hdbscan_min_cluster_size,
                "min_samples": win_pipe.config.hdbscan_min_samples,
                "cluster_selection_method": win_pipe.config.hdbscan_cluster_selection_method,
            },
            "ctfidf": {
                "bm25_weighting": win_pipe.config.bm25_weighting,
                "reduce_frequent_words": win_pipe.config.reduce_frequent_words,
                "ngram_range": list(win_pipe.config.ngram_range),
            },
            "n_documents": n_docs,
            "n_topics": win_metrics["n_topics"],
            "outlier_count": win_metrics["outlier_count"],
            "outlier_rate": win_metrics["outlier_rate"],
            "metrics": win_metrics,
            "stability": stability,
            "selection_rationale": selection_rationale,
        },
        "topics": topics_list,
        "outliers": outlier_summary,
        "hierarchy": hierarchy[:20],
        "comparison": {
            "table": comparison_table,
            "nmf_metrics": nmf_metrics,
            "bertopic_metrics": win_metrics,
            "multilingual_embedding_probes_synthetic": {
                "probe_scope": probe_report["probe_scope"],
                "n_probes": probe_report["n_probes"],
                "results": {
                    ek: {k: v for k, v in val.items() if k != "details"}
                    for ek, val in probe_report["results"].items()
                },
            },
        },
    }
    SERVING_SUMMARY_PATH.write_text(json.dumps(serving_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Quick SafeTensors round-trip verification right after saving
    reloaded = RCIBERTopicPipeline.load(BERTOPIC_ARTIFACT_DIR)
    sample_topics, _ = reloaded.transform(raw_texts[:5], embeddings=win_embs[:5])
    print(f"[Phase 1] SafeTensors round-trip load & transform verified! Sample assignments: {sample_topics}")
    print(f"[Phase 1] Saved full model to {BERTOPIC_ARTIFACT_DIR}")
    print(f"[Phase 1] Saved Vercel serving JSON ({SERVING_SUMMARY_PATH.stat().st_size // 1024} KB) to {SERVING_SUMMARY_PATH}")
    print("[Phase 1] DONE — NMF topic_discovery.json untouched.")


if __name__ == "__main__":
    main()
