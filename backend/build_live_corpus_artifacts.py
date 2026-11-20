"""Build the anonymized real-Reddit live app corpus & upgraded v3 NLP artifacts.

Reads our validated, PII-redacted, author-stripped Arctic Shift export,
filters deterministic AutoModerator sticky posts & GIF-only embeds, then runs:

  1. **Unsupervised Domain-Adaptive Pretraining (DAPT / MLM)** of
     ``google/muril-base-cased`` strictly on unlabeled anonymized public Reddit
     posts & comments (adapting representations to Indian MBA/CAT/IIM Sambalpur
     vocabulary, Hinglish, and Reddit slang without inventing any labels).
  2. **Real Unsupervised NMF Topic Discovery** directly over the anonymized
     human comments.
  3. **Upgraded 5-Task Hybrid Ensemble** (``MuRIL-DAPT v3`` with Attention-Masked
     ``[CLS]+Mean+Max`` pooling, 5-branch Multi-Sample Dropout, Layer-wise LR
     Decay, Label Smoothing, and validation-only Temperature Scaling +
     ``V2 Word+Char Calibrated TF-IDF`` blended with validation-only ``alpha``
     and threshold calibration).
  4. **Batch Inference** across every real anonymized comment and thread,
     writing slim, fast JSON artifacts to:

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
    signals, and ``evaluation.json`` transparently documents model provenance,
    DAPT perplexity, individual vs. hybrid ensemble validation/test/54-probe
    metrics, and that real-corpus human gold annotation is ongoing.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.ingestion.static_export import StaticExportLimits, load_static_export
from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.issue_identification.base import IssueDocument, IssueModelConfig
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)
from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.language_analysis_service import LanguageAnalysisService
from app.services.nlp.preprocessing.multilingual_preprocessor import (
    MultilingualTextPreprocessor,
)
from app.services.nlp.response_classification.base import (
    ResponseDocument,
    ResponseModelConfig,
)
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.response_classification.tfidf_logistic_multiclass import (
    TfidfLogisticResponseClassifier,
)
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.sentiment_emotion.base import (
    EmotionModelConfig,
    SentimentEmotionDocument,
    SentimentModelConfig,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
    TfidfLogisticEmotionClassifier,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment import (
    TfidfLogisticSentimentClassifier,
)
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
    _with_text,
    evaluate_probes_by_category,
    load_post_titles,
    score_multilabel,
    score_singlelabel,
)
from backend.train_real_pilot_models import _is_automod_boilerplate, _is_gif_only
from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.models.transformer.muril_classifier import (
    MuRILTaskClassifier,
    MuRILTaskConfig,
    _calibrate_multilabel_thresholds,
    _macro_f1_multilabel,
    _macro_f1_single,
)
from ml.models.transformer.muril_dapt import DAPTConfig, run_muril_mlm_dapt
from ml.training.trainer import records_to_annotations

DEFAULT_STATIC_EXPORT_V3 = (
    ROOT / "data" / "processed" / "arctic_shift_pilot_corpus_merged_v3" / "sanitized_static_export.json"
)
DEFAULT_STATIC_EXPORT_V2 = (
    ROOT / "data" / "processed" / "arctic_shift_pilot_corpus_merged_v2" / "sanitized_static_export.json"
)
DAPT_OUTPUT_DIR = Path("D:/AI_Project_Transformer_Temp/muril_dapt_reddit")
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


def _train_tfidf_v2(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    labels: Sequence[str],
    *,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
) -> Tuple[Any, float]:
    """Train V2 Word+Char Calibrated TF-IDF aligned to ``labels``."""
    task = TaskType(task_name)
    tr_adj = _with_text(train_rows, train_texts)
    va_adj = _with_text(val_rows, val_texts)
    train_ann = records_to_annotations(tr_adj, task)
    val_ann = records_to_annotations(va_adj, task)

    if task_name == "issue":
        model = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
        )
    elif task_name == "emotion":
        model = TfidfLogisticEmotionClassifier(
            config=EmotionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
        )
    elif task_name == "sentiment":
        model = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
        )
    elif task_name == "response":
        model = TfidfLogisticResponseClassifier(
            config=ResponseModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
        )
    else:
        model = TfidfLogisticSolutionClassifier(
            config=SolutionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
        )

    t0 = time.perf_counter()
    model.train(train_ann)
    if hasattr(model, "calibrate_thresholds"):
        model.calibrate_thresholds(val_ann)
    return model, round(time.perf_counter() - t0, 3)


def _tfidf_predict_proba(
    model: Any,
    task_name: str,
    texts: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    """Return an ``(N, len(labels))`` float32 probability matrix aligned to ``labels``."""
    if not texts:
        return np.zeros((0, len(labels)), dtype=np.float32)

    if task_name == "issue":
        docs = [IssueDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
        preds = model.predict(docs).predictions
        prob_dicts = [p.issue_probabilities for p in preds]
    elif task_name == "emotion":
        docs = [SentimentEmotionDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
        preds = model.predict(docs).predictions
        prob_dicts = [p.emotion_probabilities for p in preds]
    elif task_name == "sentiment":
        docs = [SentimentEmotionDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
        preds = model.predict(docs).predictions
        prob_dicts = [p.sentiment_probabilities for p in preds]
    elif task_name == "response":
        docs = [ResponseDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
        preds = model.predict(docs).predictions
        prob_dicts = [p.response_probabilities for p in preds]
    else:
        docs = [SolutionDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
        preds = model.predict(docs).predictions
        prob_dicts = [
            {
                "not_solution": float(p.not_solution_probability),
                "solution": float(p.solution_probability),
            }
            for p in preds
        ]

    mat = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for row_idx, d in enumerate(prob_dicts):
        for col_idx, lbl in enumerate(labels):
            mat[row_idx, col_idx] = float(d.get(lbl, 0.0))
    return mat


def _encode_targets_matrix(targets: Sequence[Any], labels: Sequence[str], multilabel: bool) -> np.ndarray:
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    if multilabel:
        mat = np.zeros((len(targets), len(labels)), dtype=np.float32)
        for r, row in enumerate(targets):
            for lbl in row:
                if lbl in idx_map:
                    mat[r, idx_map[lbl]] = 1.0
        return mat
    return np.array([idx_map[str(lbl)] for lbl in targets], dtype=np.int64)


def _decode_probs(
    probs: np.ndarray,
    labels: Sequence[str],
    multilabel: bool,
    thresholds: Optional[Dict[str, float]] = None,
    binary_threshold: Optional[float] = None,
) -> List[Any]:
    if multilabel:
        taus = [float((thresholds or {}).get(lbl, 0.5)) for lbl in labels]
        return [
            sorted([lbl for j, lbl in enumerate(labels) if float(row[j]) > taus[j]])
            for row in probs
        ]
    if binary_threshold is not None and len(labels) == 2 and labels[1] == "solution":
        return ["solution" if float(row[1]) >= binary_threshold else "not_solution" for row in probs]
    return [labels[int(np.argmax(row))] for row in probs]


def _probs_nll(probs: np.ndarray, y_true: np.ndarray, multilabel: bool) -> float:
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    if multilabel:
        return float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def _select_hybrid_blend_on_validation(
    p_muril_val: np.ndarray,
    p_tfidf_val: np.ndarray,
    val_targets: Sequence[Any],
    labels: Sequence[str],
    multilabel: bool,
    is_binary_solution: bool,
    *,
    min_muril_alpha: float = 0.0,
) -> Tuple[float, Dict[str, float], Optional[float], float]:
    """Select hybrid blend weight ``alpha`` strictly on held-out validation data.

    ``P_blend = alpha * P_muril + (1 - alpha) * P_tfidf``.
    Never looks at held-out test data or unseen robustness probes.
    """
    y_val = _encode_targets_matrix(val_targets, labels, multilabel)
    # Include pure TF-IDF (0.0), balanced blends, and pure MuRIL (1.0).
    # When F1 ties within 1e-4, prefer the lower validation NLL (better calibrated).
    candidate_alphas = tuple(
        a for a in (0.0, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0) if a >= min_muril_alpha
    ) or (1.0,)
    best_alpha = 0.75
    best_f1 = -1.0
    best_nll = float("inf")
    best_thresholds: Dict[str, float] = {lbl: 0.5 for lbl in labels}
    best_bin_tau: Optional[float] = 0.5 if is_binary_solution else None

    for alpha in candidate_alphas:
        p_blend = (alpha * p_muril_val + (1.0 - alpha) * p_tfidf_val).astype(np.float32)
        nll = _probs_nll(p_blend, y_val, multilabel)
        if multilabel:
            tuned = _calibrate_multilabel_thresholds(y_val, p_blend, labels, default_threshold=0.5)
            f1, _ = _macro_f1_multilabel(y_val, p_blend, [tuned[lbl] for lbl in labels])
            bin_tau = None
        elif is_binary_solution:
            tuned = {lbl: 0.5 for lbl in labels}
            best_local_f1 = -1.0
            bin_tau = 0.5
            for tau in [round(0.30 + 0.025 * i, 3) for i in range(17)]:  # 0.30..0.70
                pred_idx = (p_blend[:, 1] >= tau).astype(np.int64)
                sc = _macro_f1_single(y_val, pred_idx, 2)
                if (sc > best_local_f1 + 1e-12) or (
                    abs(sc - best_local_f1) <= 1e-12 and abs(tau - 0.5) < abs(bin_tau - 0.5)
                ):
                    best_local_f1 = sc
                    bin_tau = tau
            f1 = best_local_f1
        else:
            tuned = {lbl: 0.5 for lbl in labels}
            bin_tau = None
            pred_idx = np.argmax(p_blend, axis=1)
            f1 = _macro_f1_single(y_val, pred_idx, len(labels))

        better = (f1 > best_f1 + 1e-4) or (abs(f1 - best_f1) <= 1e-4 and nll < best_nll - 1e-6)
        if better:
            best_f1 = f1
            best_nll = nll
            best_alpha = float(alpha)
            best_thresholds = dict(tuned)
            best_bin_tau = bin_tau

    return best_alpha, best_thresholds, best_bin_tau, best_f1


def _probs_to_dicts(probs_mat: np.ndarray, labels: Sequence[str]) -> List[Dict[str, float]]:
    return [{lbl: round(float(row[j]), 6) for j, lbl in enumerate(labels)} for row in probs_mat]


def _train_hybrid_and_infer(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    test_texts: Sequence[str],
    infer_texts: Sequence[str],
    *,
    backbone_model_id: str,
    epochs: int,
    extra_infer_texts: Optional[Sequence[str]] = None,
    include_parent_title_probes: bool = False,
) -> Tuple[
    List[Any],
    List[Dict[str, float]],
    Optional[Tuple[List[Any], List[Dict[str, float]]]],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Train MuRIL-DAPT v3 + V2 Word+Char TF-IDF, blend on validation, evaluate, and infer."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multilabel = task_name in {"emotion", "issue"}
    is_binary_solution = task_name == "solution"
    scorer = score_multilabel if multilabel else score_singlelabel

    train_targets = _parse_targets(task_name, train_rows)
    val_targets = _parse_targets(task_name, val_rows)
    test_targets = _parse_targets(task_name, test_rows)

    # 1. Train upgraded MuRIL-DAPT v3 (CLS+Mean+Max pooling, Multi-Sample Dropout, LLRD, Label Smoothing, Temp Scaling)
    cfg = MuRILTaskConfig(
        task=task_name,
        model_id=backbone_model_id,
        epochs=epochs,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        head_lr_multiplier=3.0,
        llrd=0.90,
        pooling="cls_mean_max",
        multi_sample_dropout=5,
        dropout_rate=0.2,
        label_smoothing=0.05 if not multilabel else 0.0,
        calibrate_temperature=True,
        max_length=160,
        random_state=SEED,
        verbose=True,
    )
    muril_clf = MuRILTaskClassifier(cfg)
    t0 = time.perf_counter()
    fit_summary = muril_clf.fit(train_texts, train_targets, val_texts, val_targets)
    muril_train_s = round(time.perf_counter() - t0, 2)
    labels = list(muril_clf.labels)

    # 2. Train complementary V2 Word+Char Calibrated TF-IDF on identical split
    tfidf_clf, tfidf_train_s = _train_tfidf_v2(
        task_name, train_rows, val_rows, labels, train_texts=train_texts, val_texts=val_texts
    )

    # 3. Compute validation & test probabilities for both models
    p_mu_val = muril_clf.predict_proba(val_texts)
    p_tf_val = _tfidf_predict_proba(tfidf_clf, task_name, val_texts, labels)

    p_mu_test = muril_clf.predict_proba(test_texts)
    p_tf_test = _tfidf_predict_proba(tfidf_clf, task_name, test_texts, labels)

    # 4. Calibrate hybrid blend weight alpha + thresholds STRICTLY on validation split
    # Multi-label affect (emotion) suffers severe OOD recall collapse under bag-of-words
    # TF-IDF dilution, so keep alpha=1.0 (pure MuRIL-DAPT v3) for emotion.
    alpha, blend_thresholds, blend_bin_tau, _ = _select_hybrid_blend_on_validation(
        p_mu_val,
        p_tf_val,
        val_targets,
        labels,
        multilabel,
        is_binary_solution,
        min_muril_alpha=1.0 if task_name == "emotion" else 0.0,
    )

    p_blend_val = (alpha * p_mu_val + (1.0 - alpha) * p_tf_val).astype(np.float32)
    p_blend_test = (alpha * p_mu_test + (1.0 - alpha) * p_tf_test).astype(np.float32)

    val_preds = _decode_probs(p_blend_val, labels, multilabel, blend_thresholds, blend_bin_tau)
    test_preds = _decode_probs(p_blend_test, labels, multilabel, blend_thresholds, blend_bin_tau)
    val_metrics = scorer(val_targets, val_preds, labels)
    test_metrics = scorer(test_targets, test_preds, labels)

    # Also score pure MuRIL-DAPT v3 and pure V2 TF-IDF on the 54 unseen probes for transparent comparison
    def _mu_probe_fn(txts: Sequence[str]):
        s0 = time.perf_counter()
        pr, _ = muril_clf.predict_labels(txts)
        return pr, (time.perf_counter() - s0) * 1000.0

    def _tf_probe_fn(txts: Sequence[str]):
        s0 = time.perf_counter()
        pm = _tfidf_predict_proba(tfidf_clf, task_name, txts, labels)
        taus = getattr(tfidf_clf, "per_label_thresholds", None) if multilabel else None
        pr = _decode_probs(pm, labels, multilabel, taus, getattr(tfidf_clf, "threshold", 0.5) if is_binary_solution else None)
        return pr, (time.perf_counter() - s0) * 1000.0

    def _hybrid_probe_fn(txts: Sequence[str]):
        s0 = time.perf_counter()
        pm_mu = muril_clf.predict_proba(txts)
        pm_tf = _tfidf_predict_proba(tfidf_clf, task_name, txts, labels)
        pm_bl = (alpha * pm_mu + (1.0 - alpha) * pm_tf).astype(np.float32)
        pr = _decode_probs(pm_bl, labels, multilabel, blend_thresholds, blend_bin_tau)
        return pr, (time.perf_counter() - s0) * 1000.0

    mu_probe = evaluate_probes_by_category(
        task_name, labels, _mu_probe_fn, include_parent_title=include_parent_title_probes
    )
    tf_probe = evaluate_probes_by_category(
        task_name, labels, _tf_probe_fn, include_parent_title=include_parent_title_probes
    )
    hybrid_probe = evaluate_probes_by_category(
        task_name, labels, _hybrid_probe_fn, include_parent_title=include_parent_title_probes
    )

    print(
        f"  [{task_name.upper()} SUMMARY] val_alpha={alpha:.2f} temp={muril_clf.temperature:.2f} | "
        f"54-probe Macro F1: TFIDF-v2={tf_probe['overall_macro_f1']:.4f}, "
        f"MuRIL-DAPT-v3={mu_probe['overall_macro_f1']:.4f}, "
        f"Hybrid={hybrid_probe['overall_macro_f1']:.4f} | "
        f"heldout_test={test_metrics['macro_f1']:.4f}",
        flush=True,
    )

    # 5. Batch infer on real anonymized Reddit comments (+ optional post titles)
    p_mu_inf = muril_clf.predict_proba(infer_texts)
    p_tf_inf = _tfidf_predict_proba(tfidf_clf, task_name, infer_texts, labels)
    p_bl_inf = (alpha * p_mu_inf + (1.0 - alpha) * p_tf_inf).astype(np.float32)
    infer_preds = _decode_probs(p_bl_inf, labels, multilabel, blend_thresholds, blend_bin_tau)
    infer_prob_dicts = _probs_to_dicts(p_bl_inf, labels)

    extra_out = None
    if extra_infer_texts is not None:
        p_mu_ex = muril_clf.predict_proba(extra_infer_texts)
        p_tf_ex = _tfidf_predict_proba(tfidf_clf, task_name, extra_infer_texts, labels)
        p_bl_ex = (alpha * p_mu_ex + (1.0 - alpha) * p_tf_ex).astype(np.float32)
        extra_preds = _decode_probs(p_bl_ex, labels, multilabel, blend_thresholds, blend_bin_tau)
        extra_out = (extra_preds, _probs_to_dicts(p_bl_ex, labels))

    del muril_clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_hybrid = 0.0 < alpha < 1.0
    if include_parent_title_probes:
        model_name = "muril_dapt_v3_plus_title_hybrid" if is_hybrid else "muril_dapt_v3_plus_title"
    elif alpha == 0.0:
        model_name = "tfidf_word_char_logistic_v2"
    elif alpha == 1.0:
        model_name = "muril_dapt_v3_pooled"
    else:
        model_name = "muril_dapt_v3_tfidf_hybrid"

    eval_entry = {
        "synthetic_only": False,
        "model_name": model_name,
        "model_version": "v3_dapt_gpu",
        "labels": labels,
        "dataset_records": len(train_texts) + len(val_texts) + len(test_texts),
        "train_seconds": round(muril_train_s + tfidf_train_s, 2),
        "best_epoch": fit_summary["best_epoch"],
        "temperature": fit_summary["temperature"],
        "hybrid_blend": {
            "alpha_muril": alpha,
            "alpha_tfidf": round(1.0 - alpha, 4),
            "binary_threshold": blend_bin_tau,
            "per_label_thresholds": blend_thresholds if multilabel else {},
            "individual_54_probe_macro_f1": {
                "tfidf_word_char_v2": tf_probe["overall_macro_f1"],
                "muril_dapt_v3": mu_probe["overall_macro_f1"],
                "hybrid_selected": hybrid_probe["overall_macro_f1"],
            },
        },
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
            "model_name": model_name,
            "model_version": "v3_dapt_gpu",
            "n_train": len(train_texts),
            "n_validation": len(val_texts),
            "n_test": len(test_texts),
            "labels": labels,
            "metrics": val_metrics,
            "warnings": [],
        },
        "test": {
            "task_name": task_name,
            "model_name": model_name,
            "model_version": "v3_dapt_gpu",
            "n_train": len(train_texts),
            "n_validation": len(val_texts),
            "n_test": len(test_texts),
            "labels": labels,
            "metrics": test_metrics,
            "warnings": [
                (
                    f"54-probe unseen 8-category robustness Macro F1: {hybrid_probe['overall_macro_f1']:.4f} "
                    f"(MuRIL-DAPT-v3={mu_probe['overall_macro_f1']:.4f}, TFIDF-v2={tf_probe['overall_macro_f1']:.4f}, "
                    f"val-calibrated alpha={alpha:.2f}, T={fit_summary['temperature']:.2f})."
                ),
                "Batch inference applied to real anonymized Reddit comments; human gold annotation on the real corpus is in progress.",
            ],
        },
        "robustness_54_probes": hybrid_probe,
        "artifact_path": f"ml/artifacts/live_corpus/predictions.json#{task_name}",
        "inference_scope": (
            f"Batch inference over anonymized public Reddit discussions using {model_name} "
            f"(DAPT-adapted MuRIL v3 + Word+Char TF-IDF; human gold evaluation pending)."
        ),
    }
    meta_out = {
        "alpha": alpha,
        "binary_threshold": blend_bin_tau,
        "per_label_thresholds": blend_thresholds,
    }
    return infer_preds, infer_prob_dicts, extra_out, eval_entry, meta_out


def main(
    static_export_path: Optional[Path] = None,
    *,
    dataset_id: str = "iim-sambalpur-arctic-shift-v3",
    dataset_version: str = "3.0.0",
    n_topics: int = 10,
    dapt_epochs: int = 4,
    muril_epochs: int = 4,
    expanded_threads: int = 360,
    skip_dapt: bool = False,
) -> None:
    if static_export_path is None:
        static_export_path = (
            DEFAULT_STATIC_EXPORT_V3 if DEFAULT_STATIC_EXPORT_V3.is_file() else DEFAULT_STATIC_EXPORT_V2
        )

    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & validate sanitized real Reddit static export
    raw_size = static_export_path.stat().st_size
    validated = load_static_export(
        static_export_path,
        limits=StaticExportLimits(
            max_bytes=max(50 * 1024 * 1024, raw_size + 1024),
            max_posts=2000,
            max_comments_per_post=2000,
            max_comments=100000,
        ),
    )
    if not validated.valid:
        raise RuntimeError(f"Invalid static export {static_export_path}: {validated.error_codes}")

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
    unlabeled_dapt_texts: List[str] = []

    for post in validated.posts:
        title = sanitize_pii_text(post.title, pii_counts) or post.title
        body = sanitize_pii_text(post.body, pii_counts) if post.body else ""
        if title:
            unlabeled_dapt_texts.append(f"{title}. {body}".strip() if body else title)

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
            unlabeled_dapt_texts.append(c_body)
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

    sub_counts = Counter(p["subreddit"] for p in clean_posts)
    print(
        f"Loaded {len(clean_posts)} real anonymized threads with {len(flat_comments)} human comments "
        f"across {len(sub_counts)} subreddits ({dict(sub_counts.most_common())}); "
        f"filtered {automod_filtered} AutoMod, {gif_filtered} GIF, {empty_threads_skipped} empty threads; "
        f"extra PII redactions={dict(pii_counts)}.",
        flush=True,
    )

    # 2. Unsupervised Domain-Adaptive Pretraining (DAPT / MLM) on unlabeled real Reddit text
    dapt_summary: Optional[Dict[str, Any]] = None
    if not skip_dapt and len(unlabeled_dapt_texts) >= 40:
        print(
            f"\n--- Stage 2: Unsupervised MuRIL-DAPT MLM on {len(unlabeled_dapt_texts)} unlabeled real Reddit texts ---",
            flush=True,
        )
        dapt_summary = run_muril_mlm_dapt(
            unlabeled_dapt_texts,
            DAPTConfig(
                output_dir=str(DAPT_OUTPUT_DIR),
                epochs=dapt_epochs,
                batch_size=16,
                learning_rate=5e-5,
                max_length=160,
                random_state=SEED,
                verbose=True,
            ),
        )
        backbone_id = str(DAPT_OUTPUT_DIR)
        print(
            f"Completed DAPT MLM: {dapt_summary['unlabeled_documents']} unique docs, "
            f"perplexity {dapt_summary['initial_perplexity']} -> {dapt_summary['final_perplexity']} "
            f"({dapt_summary['elapsed_seconds']}s).",
            flush=True,
        )
    else:
        backbone_id = "google/muril-base-cased"

    # 3. Fit Real Unsupervised NMF Topic Discovery directly on the real human comments
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
        n_topics=n_topics,
        n_keywords=10,
        tfidf_config=TFIDFConfig(
            max_features=12000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w{2,}\b",
        ),
        nmf_config=NMFConfig(
            n_topics=n_topics,
            n_keywords=10,
            random_state=SEED,
            max_iter=400,
            drop_placeholder_tokens=True,
        ),
        include_timestamp=True,
    )
    nmf_result = discoverer.discover(topic_docs)
    topic_by_doc = {d.document_id: d for d in nmf_result.documents}
    print(
        f"Trained real NMF topic discovery: {nmf_result.n_topics} topics, vocab={nmf_result.vocabulary_size}.",
        flush=True,
    )

    # 4. Prepare zero-noise, zero-contamination expanded training splits
    base_post_titles = load_post_titles()
    expanded_items, exp_post_titles = generate_expanded_synthetic_bundle(
        n_threads=expanded_threads, seed=20261120
    )
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

    def _get_splits(task_name: str):
        base_records = load_annotation_dataset(TaskType(task_name), str(SYNTHETIC_DATA_DIR / f"{task_name}.csv"))
        split = split_records(
            base_records,
            labels=[r.label for r in base_records],
            spec=base_spec,
            groups=extract_groups(base_records, "post_id"),
        )
        b_tr, b_va, b_te = apply_split(base_records, split)
        tr = list(b_tr) + _to_task_records(task_name, exp_train_items)
        va = list(b_va) + _to_task_records(task_name, exp_val_items)
        leak = check_leakage(tr, va, b_te)
        assert not leak.get("leakage_detected"), f"Leakage in {task_name}: {leak}"
        return tr, va, b_te

    _, _, sample_te = _get_splits("sentiment")
    contamination = verify_zero_contamination(expanded_items, [r.text for r in sample_te])
    print(f"Contamination audit passed: {contamination}", flush=True)

    comment_texts = [c["body"] for c in flat_comments]
    comment_with_title = [f"{c['post_title']} | {c['body']}" for c in flat_comments]
    post_title_texts = [f"{p['title']} | {p['body']}" if p["body"] else p["title"] for p in clean_posts]

    evaluation_tasks: Dict[str, Any] = {}

    # 4A. ISSUE: MuRIL-DAPT v3 + Parent Title + Word+Char Hybrid
    print("\n--- Training & inferring ISSUE (MuRIL-DAPT v3 + Parent Title + Hybrid) ---", flush=True)
    iss_tr, iss_va, iss_te = _get_splits("issue")
    iss_preds, iss_probs, post_iss_out, iss_eval, _ = _train_hybrid_and_infer(
        "issue",
        iss_tr,
        iss_va,
        iss_te,
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_tr],
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_va],
        [f"{all_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_te],
        comment_with_title,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
        extra_infer_texts=post_title_texts,
        include_parent_title_probes=True,
    )
    assert post_iss_out is not None
    post_iss_preds, post_iss_probs = post_iss_out
    evaluation_tasks["issue"] = iss_eval

    # 4B. SENTIMENT: MuRIL-DAPT v3 + Word+Char Hybrid
    print("\n--- Training & inferring SENTIMENT (MuRIL-DAPT v3 + Hybrid) ---", flush=True)
    sen_tr, sen_va, sen_te = _get_splits("sentiment")
    sen_preds, sen_probs, _, sen_eval, _ = _train_hybrid_and_infer(
        "sentiment",
        sen_tr,
        sen_va,
        sen_te,
        [r.text for r in sen_tr],
        [r.text for r in sen_va],
        [r.text for r in sen_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["sentiment"] = sen_eval

    # 4C. EMOTION: MuRIL-DAPT v3 + Word+Char Hybrid (now trained on zero-noise expanded emotion corpus!)
    print("\n--- Training & inferring EMOTION (MuRIL-DAPT v3 + Hybrid) ---", flush=True)
    emo_tr, emo_va, emo_te = _get_splits("emotion")
    emo_preds, emo_probs, _, emo_eval, _ = _train_hybrid_and_infer(
        "emotion",
        emo_tr,
        emo_va,
        emo_te,
        [r.text for r in emo_tr],
        [r.text for r in emo_va],
        [r.text for r in emo_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["emotion"] = emo_eval

    # 4D. RESPONSE (6 canonical classes): MuRIL-DAPT v3 + Word+Char Hybrid
    print("\n--- Training & inferring RESPONSE (MuRIL-DAPT v3 + Hybrid) ---", flush=True)
    res_tr, res_va, res_te = _get_splits("response")
    res_preds, res_probs, _, res_eval, _ = _train_hybrid_and_infer(
        "response",
        res_tr,
        res_va,
        res_te,
        [r.text for r in res_tr],
        [r.text for r in res_va],
        [r.text for r in res_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["response"] = res_eval

    # 4E. SOLUTION (binary): MuRIL-DAPT v3 + Word+Char Hybrid (combines MuRIL semantic + TFIDF typo resilience!)
    print("\n--- Training & inferring SOLUTION (MuRIL-DAPT v3 + Hybrid) ---", flush=True)
    sol_tr, sol_va, sol_te = _get_splits("solution")
    sol_preds, sol_probs, _, sol_eval, sol_meta = _train_hybrid_and_infer(
        "solution",
        sol_tr,
        sol_va,
        sol_te,
        [r.text for r in sol_tr],
        [r.text for r in sol_va],
        [r.text for r in sol_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["solution"] = sol_eval
    sol_tau = float(sol_meta.get("binary_threshold") or 0.5)

    # 5. Assemble slim predictions.json and enriched corpus.json
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

        sent_lbl = str(sen_preds[idx])
        sent_prob_map = {k: round(float(v), 4) for k, v in sen_probs[idx].items()}
        sent_conf = round(float(sent_prob_map.get(sent_lbl, 0.0)), 4)

        emo_lbls = sorted(list(emo_preds[idx]))
        emo_prob_map = {k: round(float(v), 4) for k, v in emo_probs[idx].items()}
        emo_conf = round(max(emo_prob_map.values()) if emo_prob_map else 0.0, 4)

        iss_lbls = sorted(list(iss_preds[idx]))
        iss_prob_map = {k: round(float(v), 4) for k, v in iss_probs[idx].items()}
        iss_conf = round(max(iss_prob_map.values()) if iss_prob_map else 0.0, 4)

        resp_lbl = str(res_preds[idx])
        resp_prob_map = {k: round(float(v), 4) for k, v in res_probs[idx].items()}
        resp_conf = round(float(resp_prob_map.get(resp_lbl, 0.0)), 4)

        is_sol = str(sol_preds[idx]) == "solution"
        sol_prob = round(float(sol_probs[idx].get("solution", 0.0)), 4)
        not_sol_prob = round(float(sol_probs[idx].get("not_solution", 1.0 - sol_prob)), 4)

        label_counts_real["sentiment"][sent_lbl] += 1
        for e in emo_lbls:
            label_counts_real["emotion"][e] += 1
        for i in iss_lbls:
            label_counts_real["issue"][i] += 1
        label_counts_real["response_type"][resp_lbl] += 1
        label_counts_real["solution"]["solution" if is_sol else "not_solution"] += 1

        pred_tasks["sentiment"][doc_id] = {
            "sentiment": sent_lbl,
            "sentiment_probabilities": sent_prob_map,
            "confidence": sent_conf,
            "abstained": False,
        }
        pred_tasks["emotion"][doc_id] = {
            "emotions": emo_lbls,
            "emotion_probabilities": emo_prob_map,
            "confidence": emo_conf,
        }
        pred_tasks["issue"][doc_id] = {
            "predicted_issue_labels": iss_lbls,
            "issue_probabilities": iss_prob_map,
            "confidence": iss_conf,
        }
        pred_tasks["response"][doc_id] = {
            "response_type": resp_lbl,
            "confidence": resp_conf,
            "response_probabilities": resp_prob_map,
            "abstained": False,
            "topic_id": t_id,
            "topic_label": t_lbl,
        }
        pred_tasks["solution"][doc_id] = {
            "is_solution": is_sol,
            "solution_probability": sol_prob,
            "not_solution_probability": not_sol_prob,
            "threshold": round(sol_tau, 4),
            "abstained": False,
            "response_type": resp_lbl,
            "is_candidate": resp_lbl in {"advice", "solution"} or is_sol,
        }

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
        score_sum: Counter = Counter()
        for k, v in post_iss_probs[p_idx].items():
            score_sum[k] += float(v) * 3.0
        for c in post["comments"]:
            c_idx = comment_index[c["document_id"]]
            for k, v in iss_probs[c_idx].items():
                score_sum[k] += float(v)
        dom_issue_key = score_sum.most_common(1)[0][0] if score_sum else "placements"
        post["issue_key"] = dom_issue_key
        post["issue_category"] = ISSUE_LABELS_PRETTY.get(dom_issue_key, dom_issue_key.replace("_", " ").title())

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

        for c in post["comments"]:
            c.pop("normalized_text", None)
            c.pop("post_id", None)
            c.pop("post_title", None)
            c.pop("subreddit", None)

    for t_name, cnt_key in (
        ("sentiment", "sentiment"),
        ("emotion", "emotion"),
        ("issue", "issue"),
        ("response", "response_type"),
        ("solution", "solution"),
    ):
        evaluation_tasks[t_name]["label_counts"] = dict(sorted(label_counts_real[cnt_key].items()))

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sub_list = ", ".join(f"r/{s}" for s, _ in sub_counts.most_common())
    corpus_payload = {
        "metadata": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_seed": None,
            "generation_timestamp": now_iso,
            "source_provider": "arctic_shift",
            "source_mode": "public_reddit_archive",
            "synthetic": False,
            "authorization_status": "public_anonymized_reddit",
            "subreddits": [s for s, _ in sub_counts.most_common()],
            "notice": (
                f"Anonymized public Reddit discussions about IIM Sambalpur collected via Arctic Shift "
                f"across {len(sub_counts)} subreddits ({sub_list}) with author usernames stripped, "
                f"inline PII redacted, and AutoModerator/GIF noise filtered. NMF topics and MuRIL-DAPT "
                f"MLM domain adaptation are trained directly on this corpus; 5-task classifications are model-inferred."
            ),
            "post_count": len(clean_posts),
            "comment_count": len(flat_comments),
            "comments_per_post": round(len(flat_comments) / max(len(clean_posts), 1), 2),
            "filtering_summary": {
                "automod_boilerplate_filtered": automod_filtered,
                "gif_only_filtered": gif_filtered,
                "empty_threads_skipped": empty_threads_skipped,
            },
            "dapt_summary": dapt_summary,
            "label_counts": {k: dict(sorted(v.items())) for k, v in label_counts_real.items()},
            "label_provenance": "model_inferred_over_real_anonymized_reddit_corpus",
        },
        "posts": clean_posts,
    }

    (LIVE_DATA_DIR / "corpus.json").write_text(
        json.dumps(corpus_payload, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "predictions.json").write_text(
        json.dumps(
            {"synthetic_only": False, "dataset_version": dataset_version, "tasks": pred_tasks},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "topic_discovery.json").write_text(
        json.dumps(
            {
                "synthetic_only": False,
                "dataset_version": dataset_version,
                "result": json.loads(nmf_result.model_dump_json()),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "synthetic_only": False,
                "dataset_version": dataset_version,
                "dapt_summary": dapt_summary,
                "evaluation_scope": (
                    f"Real anonymized Reddit discussions ({len(clean_posts)} threads, {len(flat_comments)} human comments "
                    f"across {len(sub_counts)} subreddits) analyzed with unsupervised NMF v3, unsupervised MuRIL-DAPT MLM "
                    f"adaptation, and validation-calibrated MuRIL-DAPT v3 + Word+Char TF-IDF hybrid ensembles; "
                    f"held-out & 54-probe metrics reflect benchmark generalization while human gold annotation proceeds."
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
        f"\nBuilt live anonymized Reddit v3 bundle:\n"
        f"  {LIVE_DATA_DIR / 'corpus.json'} ({(LIVE_DATA_DIR / 'corpus.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'predictions.json'} ({(LIVE_ARTIFACT_DIR / 'predictions.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'topic_discovery.json'} ({(LIVE_ARTIFACT_DIR / 'topic_discovery.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'evaluation.json'} ({(LIVE_ARTIFACT_DIR / 'evaluation.json').stat().st_size / 1024:.1f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Sanitized format_version=1 static export JSON")
    parser.add_argument("--dataset-id", default="iim-sambalpur-arctic-shift-v3")
    parser.add_argument("--dataset-version", default="3.0.0")
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--dapt-epochs", type=int, default=4)
    parser.add_argument("--muril-epochs", type=int, default=4)
    parser.add_argument("--expanded-threads", type=int, default=360)
    parser.add_argument("--skip-dapt", action="store_true")
    args = parser.parse_args()
    main(
        static_export_path=args.input,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        n_topics=args.n_topics,
        dapt_epochs=args.dapt_epochs,
        muril_epochs=args.muril_epochs,
        expanded_threads=args.expanded_threads,
        skip_dapt=args.skip_dapt,
    )
