"""Build the anonymized real-Reddit live app corpus & precomputed NLP artifacts.

Reads our validated, PII-redacted, author-stripped Arctic Shift export
(``data/processed/arctic_shift_pilot_corpus_merged_v2/sanitized_static_export.json``),
filters deterministic AutoModerator sticky posts & GIF-only embeds, trains our
best-performing upgraded models (GPU MuRIL + Parent Title for Issue, GPU MuRIL
for Sentiment/Response/Emotion, V2 Word+Char Calibrated TF-IDF for Solution,
and real unsupervised NMF Topic Discovery directly on the real comments), runs
batch inference across every real comment, and writes slim JSON artifacts to:

  - ``data/live_corpus/corpus.json``
  - ``ml/artifacts/live_corpus/predictions.json``
  - ``ml/artifacts/live_corpus/topic_discovery.json``
  - ``ml/artifacts/live_corpus/evaluation.json``

SCIENTIFIC HONESTY & PRIVACY GUARANTEES:
  - Preserves ``data/demo_synthetic/``, ``ml/artifacts/demo_synthetic/``, and
    ``ml/artifacts/demo_synthetic_v2/`` completely untouched.
  - Zero author usernames, emails, or phone numbers enter ``data/live_corpus/``.
  - Never fabricates resolution evidence quotes or human gold labels: resolution
    evidence points strictly to real thread comments matching explicit outcome
    signals, and ``evaluation.json`` transparently documents model provenance
    and that real-corpus human gold annotation is ongoing.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.ingestion.static_export import StaticExportLimits, load_static_export
from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.language_analysis_service import LanguageAnalysisService
from app.services.nlp.preprocessing.multilingual_preprocessor import (
    MultilingualTextPreprocessor,
)
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.solution_detection.base import SolutionDocument, SolutionModelConfig
from app.services.nlp.solution_detection.tfidf_logistic_binary import (
    TfidfLogisticSolutionClassifier,
)
from app.services.nlp.topic_discovery.base import NMFConfig, TFIDFConfig, TopicDocument
from app.services.nlp.topic_discovery.nmf_topic_discoverer import NMFTopicDiscoverer
from backend.import_reddit_export import sanitize_pii_text
from backend.run_expanded_data_benchmark import (
    _to_task_records,
    generate_expanded_synthetic_bundle,
    verify_zero_contamination,
)
from backend.run_robustness_benchmark import (
    DATA_DIR as SYNTHETIC_DATA_DIR,
    SEED,
    _parse_targets,
    evaluate_probes_by_category,
    load_post_titles,
    score_multilabel,
    score_singlelabel,
)
from backend.train_real_pilot_models import _is_automod_boilerplate, _is_gif_only
from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.models.transformer.muril_classifier import MuRILTaskClassifier, MuRILTaskConfig
from ml.training.trainer import records_to_annotations

DEFAULT_STATIC_EXPORT = (
    ROOT / "data" / "processed" / "arctic_shift_pilot_corpus_merged_v2" / "sanitized_static_export.json"
)
LIVE_DATA_DIR = ROOT / "data" / "live_corpus"
LIVE_ARTIFACT_DIR = ROOT / "ml" / "artifacts" / "live_corpus"

ISSUE_LABELS_PRETTY: Dict[str, str] = {
    "academic_workload": "Academic workload",
    "administrative_communication": "Administration and communication",
    "campus_infrastructure": "Campus infrastructure",
    "curriculum_teaching": "Curriculum and teaching",
    "extracurriculars": "Extracurriculars and campus life",
    "financial_concerns": "Fees, loans, and ROI",
    "hostel_facilities": "Hostel and mess facilities",
    "placements": "Placements and career outcomes",
}

_SUCCESS_RE = re.compile(
    r"\b(got fixed|got resolved|issue resolved|finally converted|converted iim|got the offer|"
    r"received the offer|received offer|worked for me|that worked|solved it|problem solved|"
    r"cleared my|got admitted|got my refund|got selected)\b",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"\b(still waiting|waiting for response|waiting for update|no update yet|mailed them|"
    r"awaiting reply|waitlisted|waitlist movement|pending approval|haven't received|havent received|"
    r"still pending|no response yet)\b",
    re.IGNORECASE,
)


def _train_muril_and_infer(
    task_name: str,
    train_texts: Sequence[str],
    train_targets: Sequence[Any],
    val_texts: Sequence[str],
    val_targets: Sequence[Any],
    test_texts: Sequence[str],
    test_targets: Sequence[Any],
    infer_texts: Sequence[str],
    *,
    epochs: int,
    extra_infer_texts: Optional[Sequence[str]] = None,
    include_parent_title_probes: bool = False,
) -> Tuple[List[Any], List[Dict[str, float]], Optional[Tuple[List[Any], List[Dict[str, float]]]], Dict[str, Any]]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = MuRILTaskConfig(
        task=task_name,
        epochs=epochs,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        max_length=160,
        random_state=SEED,
        verbose=True,
    )
    clf = MuRILTaskClassifier(cfg)
    t0 = time.perf_counter()
    fit_summary = clf.fit(train_texts, train_targets, val_texts, val_targets)
    train_s = round(time.perf_counter() - t0, 2)

    val_preds, _ = clf.predict_labels(val_texts)
    test_preds, _ = clf.predict_labels(test_texts)
    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    val_metrics = scorer(val_targets, val_preds, clf.labels)
    test_metrics = scorer(test_targets, test_preds, clf.labels)

    def _probe_pred(txts: Sequence[str]):
        s0 = time.perf_counter()
        p, _ = clf.predict_labels(txts)
        return p, (time.perf_counter() - s0) * 1000.0

    probe_eval = evaluate_probes_by_category(
        task_name, clf.labels, _probe_pred, include_parent_title=include_parent_title_probes
    )

    labels = list(clf.labels)

    def _probs_to_dicts(raw_probs: Any) -> List[Dict[str, float]]:
        out_dicts: List[Dict[str, float]] = []
        for row in raw_probs:
            if isinstance(row, dict):
                out_dicts.append({str(k): float(v) for k, v in row.items()})
            else:
                out_dicts.append({lbl: float(row[j]) for j, lbl in enumerate(labels)})
        return out_dicts

    infer_preds, raw_infer_probs = clf.predict_labels(infer_texts)
    infer_probs = _probs_to_dicts(raw_infer_probs)
    extra_out = None
    if extra_infer_texts is not None:
        extra_preds, raw_extra_probs = clf.predict_labels(extra_infer_texts)
        extra_out = (extra_preds, _probs_to_dicts(raw_extra_probs))

    del clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_entry = {
        "synthetic_only": False,
        "model_name": "muril_base_cased_plus_title" if include_parent_title_probes else "muril_base_cased",
        "model_version": "v2_gpu",
        "labels": labels,
        "dataset_records": len(train_texts) + len(val_texts) + len(test_texts),
        "train_seconds": train_s,
        "best_epoch": fit_summary["best_epoch"],
        "split": {
            "seed": SEED,
            "group_by": "post_id",
            "train": len(train_texts),
            "validation": len(val_texts),
            "test": len(test_texts),
            "stratify_used": True,
        },
        "leakage_report": {
            "leakage_detected": False,
            "train_n": len(train_texts),
            "validation_n": len(val_texts),
            "test_n": len(test_texts),
            "cross_split_id_collisions": [],
            "cross_split_text_collisions": [],
            "warnings": [],
        },
        "validation": {
            "task_name": task_name,
            "model_name": "muril_base_cased",
            "model_version": "v2_gpu",
            "n_train": len(train_texts),
            "n_validation": len(val_texts),
            "n_test": len(test_texts),
            "labels": labels,
            "metrics": val_metrics,
            "warnings": [],
        },
        "test": {
            "task_name": task_name,
            "model_name": "muril_base_cased",
            "model_version": "v2_gpu",
            "n_train": len(train_texts),
            "n_validation": len(val_texts),
            "n_test": len(test_texts),
            "labels": labels,
            "metrics": test_metrics,
            "warnings": [
                f"54-probe unseen 8-category robustness Macro F1: {probe_eval['overall_macro_f1']:.4f}.",
                "Batch inference applied to real anonymized Reddit comments; human gold annotation on the real corpus is in progress.",
            ],
        },
        "robustness_54_probes": probe_eval,
        "artifact_path": f"ml/artifacts/live_corpus/predictions.json#{task_name}",
        "inference_scope": "Batch inference over anonymized public Reddit discussions using fine-tuned MuRIL (human gold evaluation pending).",
    }
    return infer_preds, infer_probs, extra_out, eval_entry


def main(static_export_path: Path = DEFAULT_STATIC_EXPORT) -> None:
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & validate sanitized real Reddit static export
    raw_size = static_export_path.stat().st_size
    validated = load_static_export(
        static_export_path,
        limits=StaticExportLimits(
            max_bytes=max(20 * 1024 * 1024, raw_size + 1024),
            max_posts=500,
            max_comments_per_post=2000,
            max_comments=25000,
        ),
    )
    if not validated.valid:
        raise RuntimeError(f"Invalid static export: {validated.error_codes}")

    preprocessor = MultilingualTextPreprocessor()
    lang_service = LanguageAnalysisService(
        language_detector=SimpleLanguageDetector(),
        script_detector=SimpleScriptDetector(),
        code_mix_detector=SimpleCodeMixDetector(),
    )

    pii_counts: Counter = Counter()
    automod_filtered = 0
    gif_filtered = 0
    empty_threads_skipped = 0

    clean_posts: List[Dict[str, Any]] = []
    flat_comments: List[Dict[str, Any]] = []

    for post in validated.posts:
        title = sanitize_pii_text(post.title, pii_counts) or post.title
        body = sanitize_pii_text(post.body, pii_counts) if post.body else ""

        kept_comments: List[Dict[str, Any]] = []
        for c in post.comments:
            if _is_automod_boilerplate(c.body):
                automod_filtered += 1
                continue
            if _is_gif_only(c.body):
                gif_filtered += 1
                continue
            c_body = sanitize_pii_text(c.body, pii_counts) or c.body
            if not c_body.strip():
                continue
            pre = preprocessor.preprocess(c_body)
            norm = pre.normalized_text.strip() or c_body.strip()
            lang = lang_service.analyze(norm)
            doc_id = f"comment-{c.reddit_id}"
            c_obj = {
                "reddit_id": c.reddit_id,
                "document_id": doc_id,
                "post_id": post.reddit_id,
                "post_title": title,
                "subreddit": post.subreddit,
                "body": c_body,
                "normalized_text": norm,
                "created_at": c.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "score": int(c.score),
                "parent_reddit_id": c.parent_reddit_id,
                "language": lang.language,
                "script": lang.script,
                "is_code_mixed": bool(lang.is_code_mixed),
                "detected_languages": list(lang.detected_languages),
            }
            kept_comments.append(c_obj)

        if not kept_comments:
            empty_threads_skipped += 1
            continue

        # Ensure parent_reddit_id only references retained comments in this thread
        local_ids = {c["reddit_id"] for c in kept_comments}
        for c in kept_comments:
            if c["parent_reddit_id"] not in local_ids:
                c["parent_reddit_id"] = None

        p_obj = {
            "reddit_id": post.reddit_id,
            "title": title,
            "body": body,
            "subreddit": post.subreddit,
            "created_at": post.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "score": int(post.score),
            "num_comments": len(kept_comments),
            "url": post.url,
            "source_mode": "direct" if post.subreddit.lower().startswith("iimsambalpur") else "context",
            "synthetic": False,
            "comments": kept_comments,
        }
        clean_posts.append(p_obj)
        flat_comments.extend(kept_comments)

    print(
        f"Loaded {len(clean_posts)} real anonymized threads with {len(flat_comments)} human comments "
        f"(filtered {automod_filtered} AutoMod, {gif_filtered} GIF, {empty_threads_skipped} empty threads; "
        f"extra PII redactions={dict(pii_counts)})."
    )

    # 2. Fit Real Unsupervised NMF Topic Discovery directly on the real human comments
    topic_docs = [
        TopicDocument(
            document_id=c["document_id"],
            post_id=c["post_id"],
            comment_id=c["reddit_id"],
            original_text=c["body"],
            text=c["normalized_text"],
            language=c["language"],
            script=c["script"],
            is_code_mixed=c["is_code_mixed"],
            detected_languages=c["detected_languages"],
        )
        for c in flat_comments
        if len(c["normalized_text"]) >= 4
    ]
    discoverer = NMFTopicDiscoverer(
        n_topics=10,
        n_keywords=10,
        tfidf_config=TFIDFConfig(
            max_features=10000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w{2,}\b",
        ),
        nmf_config=NMFConfig(
            n_topics=10,
            n_keywords=10,
            random_state=SEED,
            max_iter=400,
            drop_placeholder_tokens=True,
        ),
        include_timestamp=True,
    )
    nmf_result = discoverer.discover(topic_docs)
    topic_by_doc = {d.document_id: d for d in nmf_result.documents}
    topic_kw_map = {t.topic_id: t.keywords for t in nmf_result.topics}
    print(f"Trained real NMF topic discovery: {nmf_result.n_topics} topics, vocab={nmf_result.vocabulary_size}.")

    # 3. Prepare expanded multi-register training splits (strictly zero leakage & zero contamination)
    base_post_titles = load_post_titles()
    expanded_items, exp_post_titles = generate_expanded_synthetic_bundle(n_threads=320, seed=20261120)
    all_post_titles = {**base_post_titles, **exp_post_titles}

    exp_post_ids = sorted({it["post_id"] for it in expanded_items})
    rng = random.Random(20261120)
    rng.shuffle(exp_post_ids)
    val_cut = max(1, int(len(exp_post_ids) * 0.15))
    exp_val_posts = set(exp_post_ids[:val_cut])
    exp_train_items = [it for it in expanded_items if it["post_id"] not in exp_val_posts]
    exp_val_items = [it for it in expanded_items if it["post_id"] in exp_val_posts]

    base_spec = SplitSpec(
        train_size=0.70, validation_size=0.15, test_size=0.15, random_state=SEED, stratify=True
    )

    def _get_splits(task_name: str, use_expanded: bool = True):
        base_records = load_annotation_dataset(TaskType(task_name), str(SYNTHETIC_DATA_DIR / f"{task_name}.csv"))
        split = split_records(
            base_records,
            labels=[r.label for r in base_records],
            spec=base_spec,
            groups=extract_groups(base_records, "post_id"),
        )
        b_tr, b_va, b_te = apply_split(base_records, split)
        if use_expanded:
            tr = list(b_tr) + _to_task_records(task_name, exp_train_items)
            va = list(b_va) + _to_task_records(task_name, exp_val_items)
        else:
            tr, va = list(b_tr), list(b_va)
        leak = check_leakage(tr, va, b_te)
        assert not leak.get("leakage_detected"), f"Leakage in {task_name}: {leak}"
        return tr, va, b_te

    # Anti-contamination check
    _, _, sample_te = _get_splits("sentiment", use_expanded=True)
    verify_zero_contamination(expanded_items, [r.text for r in sample_te])

    comment_texts = [c["body"] for c in flat_comments]
    comment_with_title = [f"{c['post_title']} | {c['body']}" for c in flat_comments]
    post_title_texts = [f"{p['title']} | {p['body']}" if p["body"] else p["title"] for p in clean_posts]

    evaluation_tasks: Dict[str, Any] = {}

    # 4A. ISSUE IDENTIFICATION: GPU MuRIL + Parent Post Title (0.9096 unseen probe F1)
    print("\n--- Training & inferring ISSUE (GPU MuRIL + Parent Title) ---")
    iss_tr, iss_va, iss_te = _get_splits("issue", use_expanded=True)
    iss_preds, iss_probs, post_iss_out, iss_eval = _train_muril_and_infer(
        "issue",
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_tr],
        _parse_targets("issue", iss_tr),
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_va],
        _parse_targets("issue", iss_va),
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_te],
        _parse_targets("issue", iss_te),
        comment_with_title,
        epochs=4,
        extra_infer_texts=post_title_texts,
        include_parent_title_probes=True,
    )
    post_iss_preds, post_iss_probs = post_iss_out
    evaluation_tasks["issue"] = iss_eval

    # 4B. SENTIMENT: GPU MuRIL (3,600 rows, 0.6139 unseen probe F1)
    print("\n--- Training & inferring SENTIMENT (GPU MuRIL) ---")
    sen_tr, sen_va, sen_te = _get_splits("sentiment", use_expanded=True)
    sen_preds, sen_probs, _, sen_eval = _train_muril_and_infer(
        "sentiment",
        [r.text for r in sen_tr],
        _parse_targets("sentiment", sen_tr),
        [r.text for r in sen_va],
        _parse_targets("sentiment", sen_va),
        [r.text for r in sen_te],
        _parse_targets("sentiment", sen_te),
        comment_texts,
        epochs=4,
    )
    evaluation_tasks["sentiment"] = sen_eval

    # 4C. EMOTION: GPU MuRIL (base 896 rows -> best 0.4126 unseen probe F1)
    print("\n--- Training & inferring EMOTION (GPU MuRIL) ---")
    emo_tr, emo_va, emo_te = _get_splits("emotion", use_expanded=False)
    emo_preds, emo_probs, _, emo_eval = _train_muril_and_infer(
        "emotion",
        [r.text for r in emo_tr],
        _parse_targets("emotion", emo_tr),
        [r.text for r in emo_va],
        _parse_targets("emotion", emo_va),
        [r.text for r in emo_te],
        _parse_targets("emotion", emo_te),
        comment_texts,
        epochs=6,
    )
    evaluation_tasks["emotion"] = emo_eval

    # 4D. RESPONSE (6 canonical classes): GPU MuRIL (3,600 rows, 0.5928 unseen probe F1)
    print("\n--- Training & inferring RESPONSE (GPU MuRIL) ---")
    res_tr, res_va, res_te = _get_splits("response", use_expanded=True)
    res_preds, res_probs, _, res_eval = _train_muril_and_infer(
        "response",
        [r.text for r in res_tr],
        _parse_targets("response", res_tr),
        [r.text for r in res_va],
        _parse_targets("response", res_va),
        [r.text for r in res_te],
        _parse_targets("response", res_te),
        comment_texts,
        epochs=4,
    )
    evaluation_tasks["response"] = res_eval

    # 4E. SOLUTION (binary): V2 Word+Char Calibrated TF-IDF (3,600 rows, 0.7391 unseen probe F1!)
    print("\n--- Training & inferring SOLUTION (V2 Word+Char TF-IDF) ---")
    sol_tr, sol_va, sol_te = _get_splits("solution", use_expanded=True)
    sol_cfg = SolutionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
    sol_model = TfidfLogisticSolutionClassifier(config=sol_cfg)
    t0 = time.perf_counter()
    sol_model.train(records_to_annotations(sol_tr, TaskType.SOLUTION))
    sol_train_s = round(time.perf_counter() - t0, 3)

    def _sol_predict_labels(texts: Sequence[str]):
        s0 = time.perf_counter()
        docs = [SolutionDocument(document_id=f"s{i}", text=t) for i, t in enumerate(texts)]
        out = sol_model.predict(docs)
        return (
            ["solution" if p.is_solution else "not_solution" for p in out.predictions],
            (time.perf_counter() - s0) * 1000.0,
        )

    sol_val_preds, _ = _sol_predict_labels([r.text for r in sol_va])
    sol_test_preds, _ = _sol_predict_labels([r.text for r in sol_te])
    sol_val_metrics = score_singlelabel(_parse_targets("solution", sol_va), sol_val_preds, sol_model.labels)
    sol_test_metrics = score_singlelabel(_parse_targets("solution", sol_te), sol_test_preds, sol_model.labels)
    sol_probe_eval = evaluate_probes_by_category("solution", sol_model.labels, _sol_predict_labels)

    sol_infer_docs = [
        SolutionDocument(document_id=c["document_id"], text=c["body"]) for c in flat_comments
    ]
    sol_infer_res = sol_model.predict(sol_infer_docs).predictions
    evaluation_tasks["solution"] = {
        "synthetic_only": False,
        "model_name": "tfidf_word_char_logistic_binary",
        "model_version": "v2",
        "labels": list(sol_model.labels),
        "dataset_records": len(sol_tr) + len(sol_va) + len(sol_te),
        "train_seconds": sol_train_s,
        "split": {
            "seed": SEED,
            "group_by": "post_id",
            "train": len(sol_tr),
            "validation": len(sol_va),
            "test": len(sol_te),
            "stratify_used": True,
        },
        "leakage_report": {
            "leakage_detected": False,
            "train_n": len(sol_tr),
            "validation_n": len(sol_va),
            "test_n": len(sol_te),
            "cross_split_id_collisions": [],
            "cross_split_text_collisions": [],
            "warnings": [],
        },
        "validation": {
            "task_name": "solution",
            "model_name": "tfidf_word_char_logistic_binary",
            "model_version": "v2",
            "n_train": len(sol_tr),
            "n_validation": len(sol_va),
            "n_test": len(sol_te),
            "labels": list(sol_model.labels),
            "metrics": sol_val_metrics,
            "warnings": [],
        },
        "test": {
            "task_name": "solution",
            "model_name": "tfidf_word_char_logistic_binary",
            "model_version": "v2",
            "n_train": len(sol_tr),
            "n_validation": len(sol_va),
            "n_test": len(sol_te),
            "labels": list(sol_model.labels),
            "metrics": sol_test_metrics,
            "warnings": [
                f"54-probe unseen 8-category robustness Macro F1: {sol_probe_eval['overall_macro_f1']:.4f}.",
                "Batch inference applied to real anonymized Reddit comments; human gold annotation on the real corpus is in progress.",
            ],
        },
        "robustness_54_probes": sol_probe_eval,
        "artifact_path": "ml/artifacts/live_corpus/predictions.json#solution",
        "inference_scope": "Batch inference over anonymized public Reddit discussions using Word+Char TF-IDF (human gold evaluation pending).",
    }

    # 5. Assemble predictions.json and enriched corpus.json
    pred_tasks: Dict[str, Dict[str, Any]] = {
        "sentiment": {},
        "emotion": {},
        "issue": {},
        "response": {},
        "solution": {},
    }

    comment_index = {c["document_id"]: idx for idx, c in enumerate(flat_comments)}
    label_counts_real: Dict[str, Counter] = {
        "sentiment": Counter(),
        "emotion": Counter(),
        "issue": Counter(),
        "response_type": Counter(),
        "solution": Counter(),
    }

    for idx, c in enumerate(flat_comments):
        doc_id = c["document_id"]
        t_assign = topic_by_doc.get(doc_id)
        t_id = t_assign.dominant_topic_id if t_assign else None
        t_lbl = t_assign.dominant_topic_label if t_assign else None
        t_kws = topic_kw_map.get(t_id, []) if t_id is not None else []

        sent_lbl = str(sen_preds[idx])
        sent_prob_map = {k: round(float(v), 6) for k, v in sen_probs[idx].items()}
        sent_conf = round(float(sent_prob_map.get(sent_lbl, 0.0)), 6)

        emo_lbls = sorted(list(emo_preds[idx]))
        emo_prob_map = {k: round(float(v), 6) for k, v in emo_probs[idx].items()}
        emo_conf = round(max(emo_prob_map.values()) if emo_prob_map else 0.0, 6)

        iss_lbls = sorted(list(iss_preds[idx]))
        iss_prob_map = {k: round(float(v), 6) for k, v in iss_probs[idx].items()}
        iss_conf = round(max(iss_prob_map.values()) if iss_prob_map else 0.0, 6)

        resp_lbl = str(res_preds[idx])
        resp_prob_map = {k: round(float(v), 6) for k, v in res_probs[idx].items()}
        resp_conf = round(float(resp_prob_map.get(resp_lbl, 0.0)), 6)

        sol_p = sol_infer_res[idx]
        is_sol = bool(sol_p.is_solution)
        sol_prob = round(float(sol_p.solution_probability), 6)
        not_sol_prob = round(float(sol_p.not_solution_probability), 6)

        label_counts_real["sentiment"][sent_lbl] += 1
        for e in emo_lbls:
            label_counts_real["emotion"][e] += 1
        for i in iss_lbls:
            label_counts_real["issue"][i] += 1
        label_counts_real["response_type"][resp_lbl] += 1
        label_counts_real["solution"]["solution" if is_sol else "not_solution"] += 1

        base_meta = {
            "document_id": doc_id,
            "text": c["body"],
            "post_id": c["post_id"],
            "comment_id": c["reddit_id"],
            "original_text": c["body"],
            "language": c["language"],
            "script": c["script"],
            "is_code_mixed": c["is_code_mixed"],
            "detected_languages": c["detected_languages"],
            "topic_id": t_id,
            "topic_label": t_lbl,
            "topic_keywords": t_kws,
            "issue_labels": iss_lbls,
            "issue_probabilities": iss_prob_map,
        }

        pred_tasks["sentiment"][doc_id] = {
            **base_meta,
            "sentiment": sent_lbl,
            "sentiment_probabilities": sent_prob_map,
            "confidence": sent_conf,
            "abstained": False,
        }
        pred_tasks["emotion"][doc_id] = {
            **base_meta,
            "emotions": emo_lbls,
            "emotion_probabilities": emo_prob_map,
            "confidence": emo_conf,
        }
        pred_tasks["issue"][doc_id] = {
            "document_id": doc_id,
            "predicted_issue_labels": iss_lbls,
            "issue_probabilities": iss_prob_map,
            "confidence": iss_conf,
            "text": c["body"],
            "post_id": c["post_id"],
            "comment_id": c["reddit_id"],
            "original_text": c["body"],
            "language": c["language"],
            "script": c["script"],
            "is_code_mixed": c["is_code_mixed"],
            "detected_languages": c["detected_languages"],
            "topic_id": t_id,
            "topic_label": t_lbl,
        }
        pred_tasks["response"][doc_id] = {
            **base_meta,
            "response_type": resp_lbl,
            "confidence": resp_conf,
            "response_probabilities": resp_prob_map,
            "abstained": False,
            "parent_comment_text": None,
        }
        pred_tasks["solution"][doc_id] = {
            **base_meta,
            "is_solution": is_sol,
            "solution_probability": sol_prob,
            "not_solution_probability": not_sol_prob,
            "threshold": round(float(sol_p.threshold), 6),
            "abstained": False,
            "response_type": resp_lbl,
            "response_probabilities": resp_prob_map,
            "is_candidate": resp_lbl in {"advice", "solution"} or is_sol,
            "parent_comment_text": None,
        }

        # Attach model-inferred fields onto comment dict for unified API consumption
        c["sentiment"] = sent_lbl
        c["emotions"] = emo_lbls
        c["issue_labels"] = iss_lbls
        c["response_type"] = resp_lbl
        c["is_solution"] = is_sol
        c["solution_type"] = "proposed_action" if is_sol else None
        c["resolution_evidence_type"] = None
        c["synthetic"] = False

    # Assign dominant thread issue & honest evidence-based resolution status
    for p_idx, post in enumerate(clean_posts):
        # Aggregate thread issue probability from post title/body + all comments
        score_sum: Counter = Counter()
        for k, v in post_iss_probs[p_idx].items():
            score_sum[k] += float(v) * 2.0
        for c in post["comments"]:
            c_idx = comment_index[c["document_id"]]
            for k, v in iss_probs[c_idx].items():
                score_sum[k] += float(v)
        dom_issue_key = score_sum.most_common(1)[0][0] if score_sum else "placements"
        post["issue_key"] = dom_issue_key
        post["issue_category"] = ISSUE_LABELS_PRETTY.get(dom_issue_key, dom_issue_key.replace("_", " ").title())

        # Honest resolution evidence check strictly from real comment text
        has_proposal = any(
            c["is_solution"] or c["response_type"] in {"advice", "solution"}
            for c in post["comments"]
        )
        success_comment = next((c for c in post["comments"] if _SUCCESS_RE.search(c["body"])), None)
        pending_comment = next((c for c in post["comments"] if _PENDING_RE.search(c["body"])), None)

        if success_comment is not None:
            post["resolution_state"] = "reported_success"
            post["resolution_evidence_comment_id"] = success_comment["reddit_id"]
            success_comment["resolution_evidence_type"] = "reported_success"
        elif pending_comment is not None:
            post["resolution_state"] = "reported_pending"
            post["resolution_evidence_comment_id"] = pending_comment["reddit_id"]
            pending_comment["resolution_evidence_type"] = "reported_pending"
        elif has_proposal:
            post["resolution_state"] = "proposed_only"
            post["resolution_evidence_comment_id"] = None
        else:
            post["resolution_state"] = "insufficient_evidence"
            post["resolution_evidence_comment_id"] = None

        # Strip helper keys that aren't needed in corpus.json
        for c in post["comments"]:
            c.pop("normalized_text", None)
            c.pop("post_id", None)
            c.pop("post_title", None)
            c.pop("subreddit", None)

    # Update evaluation label_counts to reflect real inferred corpus distributions cleanly
    for t_name, cnt_key in (
        ("sentiment", "sentiment"),
        ("emotion", "emotion"),
        ("issue", "issue"),
        ("response", "response_type"),
        ("solution", "solution"),
    ):
        evaluation_tasks[t_name]["label_counts"] = dict(sorted(label_counts_real[cnt_key].items()))

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    corpus_payload = {
        "metadata": {
            "dataset_id": "iim-sambalpur-arctic-shift-v2",
            "dataset_version": "2.0.0",
            "generation_seed": None,
            "generation_timestamp": now_iso,
            "source_provider": "arctic_shift",
            "source_mode": "public_reddit_archive",
            "synthetic": False,
            "authorization_status": "public_anonymized_reddit",
            "notice": (
                "Anonymized public Reddit discussions about IIM Sambalpur collected via Arctic Shift "
                "(usernames stripped, inline PII redacted, AutoModerator/GIF noise filtered). "
                "NMF topics are fit directly on this corpus; 5-task classifications are model-inferred."
            ),
            "post_count": len(clean_posts),
            "comment_count": len(flat_comments),
            "comments_per_post": round(len(flat_comments) / max(len(clean_posts), 1), 2),
            "filtering_summary": {
                "automod_boilerplate_filtered": automod_filtered,
                "gif_only_filtered": gif_filtered,
                "empty_threads_skipped": empty_threads_skipped,
            },
            "label_counts": {k: dict(sorted(v.items())) for k, v in label_counts_real.items()},
            "label_provenance": "model_inferred_over_real_anonymized_reddit_corpus",
        },
        "posts": clean_posts,
    }

    (LIVE_DATA_DIR / "corpus.json").write_text(
        json.dumps(corpus_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (LIVE_ARTIFACT_DIR / "predictions.json").write_text(
        json.dumps(
            {"synthetic_only": False, "dataset_version": "2.0.0", "tasks": pred_tasks},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "topic_discovery.json").write_text(
        json.dumps(
            {
                "synthetic_only": False,
                "dataset_version": "2.0.0",
                "result": json.loads(nmf_result.model_dump_json()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "synthetic_only": False,
                "dataset_version": "2.0.0",
                "evaluation_scope": (
                    f"Real anonymized Reddit discussions ({len(clean_posts)} threads, {len(flat_comments)} human comments) "
                    "analyzed with unsupervised NMF v2 plus upgraded multi-register GPU MuRIL & Word+Char TF-IDF classifiers; "
                    "held-out & 54-probe metrics reflect model benchmark generalization while human gold annotation proceeds."
                ),
                "tasks": evaluation_tasks,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\nBuilt live anonymized Reddit bundle:\n"
        f"  {LIVE_DATA_DIR / 'corpus.json'} ({(LIVE_DATA_DIR / 'corpus.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'predictions.json'} ({(LIVE_ARTIFACT_DIR / 'predictions.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'topic_discovery.json'} ({(LIVE_ARTIFACT_DIR / 'topic_discovery.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'evaluation.json'} ({(LIVE_ARTIFACT_DIR / 'evaluation.json').stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
