"""Build the anonymized real-Reddit live app corpus & v5 NLP artifacts (100% real data).

Reads our sanitized, PII-redacted public Reddit export (or falls back cleanly to
tracked ``data/live_corpus/corpus.json`` so any clean clone can reproduce the
entire pipeline), strips raw Reddit IDs/URLs into opaque pseudonymized IDs
(``post_0001``, ``cmt_00001``), removes AutoModerator/RemindMeBot/GIF noise,
applies Indian-Reddit-aware English/Hinglish/Hindi language detection, and runs:

  1. **Unsupervised Domain-Adaptive Pretraining (DAPT / MLM)** of
     ``google/muril-base-cased`` strictly on unlabeled real Reddit texts with a
     disjoint 12% held-out validation split measuring genuine pre- vs. post-DAPT
     held-out perplexity.
  2. **Real Unsupervised 10-Cluster NMF Topic Discovery** directly on the real
     anonymized comments.
  3. **Apples-to-Apples 4-Architecture Supervised Benchmark** trained and
     evaluated strictly on ``post_id``-grouped splits (65% train / 15% val / 20%
     held-out real test) of our tracked real curated Reddit dataset
     (``data/annotations/real_reddit_labeled_v1/``, 459 comments across 200
     distinct threads — ZERO synthetic training or test rows):
       - Logistic Regression (Word+Char TF-IDF)
       - ANN (2-Layer GELU+LayerNorm MLP)
       - CNN (1D Multi-Kernel TextCNN, k=2,3,4,5)
       - Transformer (``MuRIL-DAPT v3`` Pooled + Multi-Sample Dropout + LLRD +
         non-sharpening validation temperature scaling ``T >= 1.0``)
       - Production Hybrid (validation-only blend ``alpha >= 0.5``)
  4. **Data-Driven Error Analysis & 5-Bin Calibration** computed directly on the
     real held-out Reddit test split (zero canned lookup explanations), plus an
     auxiliary 54-probe stress diagnostic (never used for model selection).
  5. **7-Check Automated Pipeline Integrity Verification** before writing slim
     read-only JSON artifacts to ``data/live_corpus/`` and
     ``ml/artifacts/live_corpus/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
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
from app.services.nlp.issue_identification.base import IssueDocument, IssueModelConfig
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)
from app.services.nlp.preprocessing.multilingual_preprocessor import (
    MultilingualTextPreprocessor,
)
from app.services.nlp.response_classification.base import (
    ResponseDocument,
    ResponseModelConfig,
)
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
from backend.run_robustness_benchmark import (
    ROBUSTNESS_PROBES,
    SEED,
    _parse_targets,
    _with_text,
    evaluate_probes_by_category,
    score_multilabel,
    score_singlelabel,
)
from backend.train_real_pilot_models import _is_automod_boilerplate, _is_gif_only
from ml.configs import TaskType
from ml.data.dataset import TrainingRecord, load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.evaluation.benchmark_verifier import verify_benchmark_artifact
from ml.evaluation.error_analysis import (
    analyze_architecture_recoveries_and_errors,
    classify_confidence_tier,
    compute_calibration_metrics,
    compute_confusion_and_top_pairs,
    probability_margin,
)
from ml.models.neural_baselines import (
    ANNMLPClassifier,
    LogisticRegressionBaseline,
    NeuralBaselineConfig,
    TextCNNClassifier,
)
from ml.models.transformer.muril_classifier import (
    MuRILTaskClassifier,
    MuRILTaskConfig,
    _calibrate_multilabel_thresholds,
    _macro_f1_multilabel,
    _macro_f1_single,
)
from ml.models.transformer.muril_dapt import (
    DAPTConfig,
    default_dapt_output_dir,
    run_muril_mlm_dapt,
)
from ml.training.trainer import records_to_annotations

DEFAULT_STATIC_EXPORT_V4 = (
    ROOT / "data" / "processed" / "arctic_shift_pilot_corpus_merged_v4" / "sanitized_static_export.json"
)
DEFAULT_REAL_LABELED_DIR = ROOT / "data" / "annotations" / "real_reddit_labeled_v1"
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
_BOT_EXTRA_RE = re.compile(
    r"(RemindMeBot|I will be messaging you in|follows platform-wide Reddit Rules)",
    re.IGNORECASE,
)
_REDDIT_URL_RE = re.compile(r"https?://(?:www\.|old\.)?reddit\.com/\S+", re.IGNORECASE)
_ALPHA_TOK_RE = re.compile(r"[a-zA-Z]+")

# Unambiguous Romanized Hindi / Hinglish markers (strictly excluding English collisions like 'the', 'me', 'par', 'scene')
_STRONG_HINGLISH = frozenset({
    "hai", "hain", "nahi", "nhi", "kya", "kyu", "kyun", "kaise", "bhai", "bhaiya",
    "yaar", "toh", "bhi", "mein", "mera", "tera", "aur", "lekin", "agar", "accha",
    "achha", "acha", "theek", "thik", "karo", "karna", "karke", "karega", "karenge",
    "gaya", "gayi", "gaye", "dekho", "dekhna", "dekha", "bola", "bolna", "padho",
    "padhna", "padhai", "mila", "mile", "milta", "chahiye", "paisa", "paise",
    "sahi", "galat", "bahut", "bohot", "zyada", "abhi", "pehle", "phir", "kuch",
    "kuchh", "koi", "kaun", "wala", "wali", "wale", "kaam", "banda", "bande",
    "jugaad", "haan", "shukriya", "hoga", "hogi", "honge", "raha", "rahi", "rahe",
    "hota", "hoti", "jao", "jaana", "jaao", "rha", "rhi", "rhe", "krna", "kro",
    "mtlb", "matlab", "shayad", "bilkul", "thoda", "batao", "pucha", "pata",
    "aayi", "aayegi", "aaegi", "aagyi", "baat", "yaha", "waha", "kitna", "kab",
    "udhar", "idhar", "unka", "unko", "unhe", "apna", "apne", "khud", "lagta",
    "chalo", "fir", "yhi", "kiska", "kisko", "kahan", "dhyan", "bkl", "abe",
    "arre", "chutiyapa", "dost", "paglu", "bhoot", "jaruri", "isliye", "kyuki",
    "taaki", "kudh", "sabhi", "horaha", "uthare", "utha", "nahin", "harkatein",
    "dekhkar", "karwa", "hogyi", "karun", "sakti", "mujhe", "farak", "padhta",
    "hoon", "aap", "dekh", "wli", "phele", "daali", "ismi", "khilao", "chor",
    "bhar", "bahar", "puch", "karle", "lelo", "bata", "hum", "tum", "woh", "yeh",
})
_ENGLISH_OR_MBA_ANCHORS = frozenset({
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "have",
    "not", "but", "from", "they", "will", "would", "could", "should", "what", "when",
    "where", "which", "about", "there", "just", "like", "more", "some", "very", "only",
    "also", "can", "all", "any", "get", "got", "one", "how", "why", "yes", "no", "ok",
    "okay", "same", "true", "agree", "thanks", "thank", "congrats", "congratulations",
    "please", "pls", "bro", "sir", "check", "mail", "call", "email", "link", "dm",
    "iim", "cat", "xat", "gmat", "jee", "ipmat", "wl", "waitlist", "waitlisted",
    "fees", "fee", "loan", "roi", "lpa", "ctc", "sip", "placement", "placements",
    "hostel", "mess", "campus", "mba", "ug", "bs", "dsai", "mpp", "sambalpur",
    "mumbai", "lucknow", "shillong", "udaipur", "trichy", "raipur", "ranchi",
    "sirmaur", "jammu", "bodhgaya", "kashipur", "vizag", "nagpur", "tapmi", "gim",
    "glim", "imi", "imt", "fore", "nmims", "isb", "iit", "nit", "college", "clg",
    "offer", "list", "round", "merit", "gen", "general", "obc", "ews", "sc", "st",
    "gem", "gnem", "gef", "gnef", "profile", "percentile", "score", "rank", "cutoff",
    "interview", "pi", "gd", "wat", "cap", "sap", "batch", "sem", "semester", "term",
    "classes", "prof", "faculty", "exam", "exams", "cgpa", "refund", "portal", "seat",
    "admission", "admissions", "convert", "converted", "rejected", "cfbr", "imo",
    "tbh", "idk", "ig", "rn", "fr", "ngl", "lol", "lmao", "wtf", "fyi", "ama",
})


def detect_indian_reddit_language(text: str, script_detector: SimpleScriptDetector) -> Dict[str, Any]:
    """Accurate Indian-Reddit language & Hinglish code-mix classifier (zero fake Welsh/French)."""
    s_res = script_detector.detect(text)
    primary_script = str(getattr(s_res, "script", "latin") or "latin").lower()
    raw = str(text or "")

    # 1. Native Indic scripts first
    indic_script_map = {
        "devanagari": "hi",
        "bengali": "bn",
        "odia": "or",
        "oriya": "or",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "gujarati": "gu",
        "gurmukhi": "pa",
    }
    has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in raw)
    latin_toks = [t.lower() for t in _ALPHA_TOK_RE.findall(raw)]
    if has_devanagari or primary_script in indic_script_map:
        indic_lang = "hi" if has_devanagari else indic_script_map.get(primary_script, "hi")
        if len(latin_toks) >= 2:
            return {
                "language": f"{indic_lang}-en",
                "script": "mixed" if latin_toks else primary_script,
                "is_code_mixed": True,
                "detected_languages": [indic_lang, "en"],
            }
        return {
            "language": indic_lang,
            "script": primary_script,
            "is_code_mixed": False,
            "detected_languages": [indic_lang],
        }

    # 2. Romanized Hindi / Hinglish detection via clean unambiguous marker set
    n_tok = len(latin_toks)
    hi_hits = [t for t in latin_toks if t in _STRONG_HINGLISH]
    en_hits = [t for t in latin_toks if t in _ENGLISH_OR_MBA_ANCHORS]

    if hi_hits:
        hi_ratio = len(hi_hits) / max(n_tok, 1)
        if len(hi_hits) >= 2 or (n_tok <= 12 and len(hi_hits) >= 1):
            if hi_ratio >= 0.55 and len(en_hits) <= 1:
                return {
                    "language": "hi",
                    "script": "latin",
                    "is_code_mixed": False,
                    "detected_languages": ["hi"],
                }
            return {
                "language": "hi-en",
                "script": "latin",
                "is_code_mixed": True,
                "detected_languages": ["hi", "en"],
            }

    # 3. English / Indian MBA-Reddit English vs short undetermined replies
    if en_hits or n_tok >= 4:
        return {
            "language": "en",
            "script": "latin",
            "is_code_mixed": False,
            "detected_languages": ["en"],
        }

    return {
        "language": "und",
        "script": primary_script if primary_script else "unknown",
        "is_code_mixed": False,
        "detected_languages": [],
    }


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
    min_muril_alpha: float = 0.50,
) -> Tuple[float, Dict[str, float], Optional[float], float]:
    """Select hybrid blend weight ``alpha >= min_muril_alpha`` strictly on real validation split.

    Keeping ``min_muril_alpha >= 0.50`` ensures the production model always preserves
    the domain-adapted Transformer backbone rather than collapsing to bag-of-words TF-IDF.
    Never looks at held-out test data or auxiliary diagnostic probes.
    """
    y_val = _encode_targets_matrix(val_targets, labels, multilabel)
    candidate_alphas = tuple(
        a for a in (0.50, 0.60, 0.70, 0.80, 0.90, 1.0) if a >= min_muril_alpha
    ) or (1.0,)
    best_alpha = 1.0
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
            for tau in [round(0.30 + 0.025 * i, 3) for i in range(17)]:
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
    test_subreddits: Sequence[str],
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
    """Train 4 architectures on real curated Reddit splits, calibrate on val, evaluate on real test, and infer."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multilabel = task_name in {"emotion", "issue"}
    is_binary_solution = task_name == "solution"
    scorer = score_multilabel if multilabel else score_singlelabel

    train_targets = _parse_targets(task_name, train_rows)
    val_targets = _parse_targets(task_name, val_rows)
    test_targets = _parse_targets(task_name, test_rows)

    # 1. Train MuRIL-DAPT v3 (small-batch few-shot recipe per Mosbach et al. ICLR 2021:
    #    batch_size=8, lr=4e-5, cosine warmup, LLRD, best-validation checkpointing, T>=1.0)
    cfg = MuRILTaskConfig(
        task=task_name,
        model_id=backbone_model_id,
        epochs=epochs,
        batch_size=8,
        eval_batch_size=32,
        learning_rate=4e-5,
        head_lr_multiplier=4.0,
        llrd=0.92,
        pooling="cls_mean_max",
        multi_sample_dropout=5,
        dropout_rate=0.2,
        label_smoothing=0.05 if not multilabel else 0.0,
        max_pos_weight=2.5,
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

    # 2. Train complementary V2 Word+Char Calibrated TF-IDF on identical real split
    tfidf_clf, tfidf_train_s = _train_tfidf_v2(
        task_name, train_rows, val_rows, labels, train_texts=train_texts, val_texts=val_texts
    )

    # 2B. Train & evaluate apples-to-apples Logistic Regression, ANN (MLP), and 1D TextCNN baselines
    lr_clf = LogisticRegressionBaseline(multilabel=multilabel, random_state=SEED)
    lr_clf.fit(train_texts, train_targets, val_texts, val_targets, labels)
    t_lr0 = time.perf_counter()
    p_lr_test = lr_clf.predict_proba(test_texts)
    lr_lat_ms = ((time.perf_counter() - t_lr0) * 1000.0) / max(len(test_texts), 1)
    lr_test_preds = lr_clf.predict(test_texts)
    lr_test_m = scorer(test_targets, lr_test_preds, labels)
    lr_probe_m = evaluate_probes_by_category(
        task_name,
        labels,
        lambda txts: (lr_clf.predict(txts), 0.0),
        include_parent_title=include_parent_title_probes,
    )
    lr_cal = compute_calibration_metrics(
        test_targets, p_lr_test, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )

    ann_clf = ANNMLPClassifier(
        multilabel=multilabel,
        config=NeuralBaselineConfig(epochs=20, batch_size=32, learning_rate=2e-3, dropout=0.35, random_state=SEED),
    )
    ann_clf.fit(train_texts, train_targets, val_texts, val_targets, labels)
    t_ann0 = time.perf_counter()
    p_ann_test = ann_clf.predict_proba(test_texts)
    ann_lat_ms = ((time.perf_counter() - t_ann0) * 1000.0) / max(len(test_texts), 1)
    ann_test_preds = ann_clf.predict(test_texts)
    ann_test_m = scorer(test_targets, ann_test_preds, labels)
    ann_probe_m = evaluate_probes_by_category(
        task_name,
        labels,
        lambda txts: (ann_clf.predict(txts), 0.0),
        include_parent_title=include_parent_title_probes,
    )
    ann_cal = compute_calibration_metrics(
        test_targets, p_ann_test, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )
    ann_params = ann_clf.param_count_
    ann_train_s = ann_clf.train_seconds_
    ann_clf.cleanup()

    cnn_clf = TextCNNClassifier(
        multilabel=multilabel,
        config=NeuralBaselineConfig(epochs=22, batch_size=32, learning_rate=2e-3, dropout=0.40, random_state=SEED),
    )
    cnn_clf.fit(train_texts, train_targets, val_texts, val_targets, labels)
    t_cnn0 = time.perf_counter()
    p_cnn_test = cnn_clf.predict_proba(test_texts)
    cnn_lat_ms = ((time.perf_counter() - t_cnn0) * 1000.0) / max(len(test_texts), 1)
    cnn_test_preds = cnn_clf.predict(test_texts)
    cnn_test_m = scorer(test_targets, cnn_test_preds, labels)
    cnn_probe_m = evaluate_probes_by_category(
        task_name,
        labels,
        lambda txts: (cnn_clf.predict(txts), 0.0),
        include_parent_title=include_parent_title_probes,
    )
    cnn_cal = compute_calibration_metrics(
        test_targets, p_cnn_test, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )
    cnn_params = cnn_clf.param_count_
    cnn_train_s = cnn_clf.train_seconds_
    cnn_clf.cleanup()

    # 3. Evaluate MuRIL-DAPT v3 on real validation & held-out real test split (raw T=1.0 vs calibrated T>=1.0)
    p_mu_val = muril_clf.predict_proba(val_texts)
    p_tf_val = _tfidf_predict_proba(tfidf_clf, task_name, val_texts, labels)

    orig_temp = max(1.0, float(muril_clf.temperature))
    muril_clf._temperature = 1.0
    p_mu_test_uncal = muril_clf.predict_proba(test_texts)
    muril_clf._temperature = orig_temp

    t_mu0 = time.perf_counter()
    p_mu_test = muril_clf.predict_proba(test_texts)
    mu_lat_ms = ((time.perf_counter() - t_mu0) * 1000.0) / max(len(test_texts), 1)
    mu_test_preds, _ = muril_clf.predict_labels(test_texts)
    mu_test_m = scorer(test_targets, mu_test_preds, labels)
    mu_params = sum(p.numel() for p in muril_clf._model.parameters()) if muril_clf._model is not None else 237_556_736
    mu_cal_uncal = compute_calibration_metrics(
        test_targets, p_mu_test_uncal, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )
    mu_cal = compute_calibration_metrics(
        test_targets, p_mu_test, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )

    t_tf0 = time.perf_counter()
    p_tf_test = _tfidf_predict_proba(tfidf_clf, task_name, test_texts, labels)
    tf_lat_ms = ((time.perf_counter() - t_tf0) * 1000.0) / max(len(test_texts), 1)

    # 4. Calibrate hybrid blend alpha >= 0.50 strictly on real validation split (never touches test or probes!)
    alpha, blend_thresholds, blend_bin_tau, _ = _select_hybrid_blend_on_validation(
        p_mu_val,
        p_tf_val,
        val_targets,
        labels,
        multilabel,
        is_binary_solution,
        min_muril_alpha=1.0 if task_name == "emotion" else 0.50,
    )

    p_blend_val = (alpha * p_mu_val + (1.0 - alpha) * p_tf_val).astype(np.float32)
    p_blend_test = (alpha * p_mu_test + (1.0 - alpha) * p_tf_test).astype(np.float32)

    val_preds = _decode_probs(p_blend_val, labels, multilabel, blend_thresholds, blend_bin_tau)
    test_preds = _decode_probs(p_blend_test, labels, multilabel, blend_thresholds, blend_bin_tau)
    val_metrics = scorer(val_targets, val_preds, labels)
    test_metrics = scorer(test_targets, test_preds, labels)
    hyb_lat_ms = mu_lat_ms + tf_lat_ms
    hyb_cal = compute_calibration_metrics(
        test_targets, p_blend_test, labels, multilabel=multilabel, n_bins=5, source_split="heldout_real_test"
    )

    # Auxiliary 54-probe stress diagnostic (strictly observational, never used for selection!)
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

    # Data-driven Error Analysis strictly on the REAL held-out Reddit test split
    heldout_err = compute_confusion_and_top_pairs(
        test_targets, test_preds, labels, multilabel=multilabel, texts=test_texts, probs=p_blend_test
    )
    arch_recovery = analyze_architecture_recoveries_and_errors(
        task_name=task_name,
        probe_texts=test_texts,
        probe_categories=test_subreddits,
        y_true=test_targets,
        model_preds_by_arch={
            "logistic_regression": lr_test_preds,
            "ann_mlp": ann_test_preds,
            "cnn_textcnn": cnn_test_preds,
            "transformer_muril_dapt_v3": mu_test_preds,
            "production_hybrid": test_preds,
        },
        model_probs_by_arch={
            "logistic_regression": p_lr_test,
            "ann_mlp": p_ann_test,
            "cnn_textcnn": p_cnn_test,
            "transformer_muril_dapt_v3": p_mu_test,
            "production_hybrid": p_blend_test,
        },
        labels=labels,
        multilabel=multilabel,
        target_arch="transformer_muril_dapt_v3",
        train_texts=train_texts,
    )

    arch_comparison_task = {
        "logistic_regression": {
            "label": "Logistic Regression (Word+Char TF-IDF)",
            "family": "linear",
            "test_macro_f1": round(float(lr_test_m["macro_f1"]), 4),
            "probe_macro_f1": round(float(lr_probe_m["overall_macro_f1"]), 4),
            "probe_by_category": lr_probe_m.get("by_category", {}),
            "param_count": int(lr_clf.param_count_),
            "train_seconds": float(lr_clf.train_seconds_),
            "inference_ms_per_comment": round(float(lr_lat_ms), 3),
            "calibration_ece": float(lr_cal["ece"]),
        },
        "ann_mlp": {
            "label": "ANN (2-Layer MLP · GELU + LayerNorm)",
            "family": "ann",
            "test_macro_f1": round(float(ann_test_m["macro_f1"]), 4),
            "probe_macro_f1": round(float(ann_probe_m["overall_macro_f1"]), 4),
            "probe_by_category": ann_probe_m.get("by_category", {}),
            "param_count": int(ann_params),
            "train_seconds": float(ann_train_s),
            "inference_ms_per_comment": round(float(ann_lat_ms), 3),
            "calibration_ece": float(ann_cal["ece"]),
        },
        "cnn_textcnn": {
            "label": "CNN (1D Multi-Kernel TextCNN k=2..5)",
            "family": "cnn",
            "test_macro_f1": round(float(cnn_test_m["macro_f1"]), 4),
            "probe_macro_f1": round(float(cnn_probe_m["overall_macro_f1"]), 4),
            "probe_by_category": cnn_probe_m.get("by_category", {}),
            "param_count": int(cnn_params),
            "train_seconds": float(cnn_train_s),
            "inference_ms_per_comment": round(float(cnn_lat_ms), 3),
            "calibration_ece": float(cnn_cal["ece"]),
        },
        "transformer_muril_dapt_v3": {
            "label": "Transformer (MuRIL-DAPT v3 Pooled)",
            "family": "transformer",
            "test_macro_f1": round(float(mu_test_m["macro_f1"]), 4),
            "probe_macro_f1": round(float(mu_probe["overall_macro_f1"]), 4),
            "probe_by_category": mu_probe.get("by_category", {}),
            "param_count": int(mu_params),
            "train_seconds": float(muril_train_s),
            "inference_ms_per_comment": round(float(mu_lat_ms), 3),
            "calibration_ece": float(mu_cal["ece"]),
            "uncalibrated_ece": float(mu_cal_uncal["ece"]),
            "temperature": round(orig_temp, 3),
        },
        "production_hybrid": {
            "label": "Production Hybrid (MuRIL-DAPT v3 + TF-IDF)",
            "family": "hybrid",
            "test_macro_f1": round(float(test_metrics["macro_f1"]), 4),
            "probe_macro_f1": round(float(hybrid_probe["overall_macro_f1"]), 4),
            "probe_by_category": hybrid_probe.get("by_category", {}),
            "param_count": int(mu_params + lr_clf.param_count_),
            "train_seconds": round(float(muril_train_s + tfidf_train_s), 2),
            "inference_ms_per_comment": round(float(hyb_lat_ms), 3),
            "calibration_ece": float(hyb_cal["ece"]),
            "alpha_muril": alpha,
        },
    }

    error_analysis_task = {
        "task": task_name,
        "multilabel": multilabel,
        "evaluation_split": f"heldout_real_test (n={len(test_texts)})",
        "heldout_confusion": heldout_err,
        "probe_confusion": heldout_err,
        "calibration": {
            "n_samples": len(test_texts),
            "source_split": "heldout_real_test",
            "temperature": round(orig_temp, 3),
            "uncalibrated_ece": mu_cal_uncal["ece"],
            "calibrated_ece": hyb_cal["ece"],
            "transformer_calibrated_ece": mu_cal["ece"],
            "brier_score": hyb_cal["brier_score"],
            "bins": hyb_cal["bins"],
            "uncalibrated_bins": mu_cal_uncal["bins"],
            "confidence_tiers": hyb_cal["confidence_tiers"],
        },
        "architecture_analysis": arch_recovery,
    }

    print(
        f"  [{task_name.upper()} REAL HELD-OUT TEST F1 (n={len(test_texts)})] "
        f"LogReg={lr_test_m['macro_f1']:.4f} | ANN={ann_test_m['macro_f1']:.4f} | "
        f"CNN={cnn_test_m['macro_f1']:.4f} | MuRIL-DAPT-v3={mu_test_m['macro_f1']:.4f} | "
        f"Hybrid(a={alpha:.2f})={test_metrics['macro_f1']:.4f} | "
        f"ECE {mu_cal_uncal['ece']:.3f}->{hyb_cal['ece']:.3f} (T={orig_temp:.2f}) | "
        f"Aux 54-probe: MuRIL={mu_probe['overall_macro_f1']:.3f} Hyb={hybrid_probe['overall_macro_f1']:.3f}",
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
    elif alpha == 1.0:
        model_name = "muril_dapt_v3_pooled"
    else:
        model_name = "muril_dapt_v3_tfidf_hybrid"

    eval_entry = {
        "synthetic_only": False,
        "model_name": model_name,
        "model_version": "v5_real_reddit_gpu",
        "labels": labels,
        "dataset_records": len(train_texts) + len(val_texts) + len(test_texts),
        "train_seconds": round(muril_train_s + tfidf_train_s, 2),
        "best_epoch": fit_summary["best_epoch"],
        "temperature": round(orig_temp, 3),
        "hybrid_blend": {
            "alpha_muril": alpha,
            "alpha_tfidf": round(1.0 - alpha, 4),
            "binary_threshold": blend_bin_tau,
            "per_label_thresholds": blend_thresholds if multilabel else {},
            "individual_heldout_test_macro_f1": {
                "logistic_regression": round(float(lr_test_m["macro_f1"]), 4),
                "ann_mlp": round(float(ann_test_m["macro_f1"]), 4),
                "cnn_textcnn": round(float(cnn_test_m["macro_f1"]), 4),
                "muril_dapt_v3": round(float(mu_test_m["macro_f1"]), 4),
                "hybrid_selected": round(float(test_metrics["macro_f1"]), 4),
            },
            "auxiliary_54_probe_macro_f1": {
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
            "model_version": "v5_real_reddit_gpu",
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
            "model_version": "v5_real_reddit_gpu",
            "n_train": len(train_texts),
            "n_validation": len(val_texts),
            "n_test": len(test_texts),
            "labels": labels,
            "metrics": test_metrics,
            "warnings": [
                (
                    f"Primary evaluation is on {len(test_texts)} held-out real Reddit comments from unseen threads "
                    f"(Macro F1={test_metrics['macro_f1']:.4f}, MuRIL-DAPT-v3={mu_test_m['macro_f1']:.4f}, "
                    f"LogReg={lr_test_m['macro_f1']:.4f}, ANN={ann_test_m['macro_f1']:.4f}, CNN={cnn_test_m['macro_f1']:.4f}, "
                    f"val-only alpha={alpha:.2f}, T={orig_temp:.2f})."
                ),
                (
                    "Trained and tested strictly on 459 rubric-curated real anonymized Reddit comments "
                    "(data/annotations/real_reddit_labeled_v1/); multi-annotator human kappa adjudication is ongoing."
                ),
            ],
        },
        "robustness_54_probes": hybrid_probe,
        "artifact_path": f"ml/artifacts/live_corpus/predictions.json#{task_name}",
        "inference_scope": (
            f"Batch inference over anonymized public Reddit discussions using {model_name} "
            f"trained strictly on curated real Reddit comments (0 synthetic rows; multi-human adjudication pending)."
        ),
    }
    meta_out = {
        "alpha": alpha,
        "binary_threshold": blend_bin_tau,
        "per_label_thresholds": blend_thresholds,
        "architecture_comparison": arch_comparison_task,
        "error_analysis": error_analysis_task,
    }
    return infer_preds, infer_prob_dicts, extra_out, eval_entry, meta_out


def _load_raw_posts(static_export_path: Optional[Path]) -> Tuple[List[Dict[str, Any]], str]:
    """Load posts from sanitized static export if present, else cleanly from tracked corpus.json."""
    candidate = static_export_path or DEFAULT_STATIC_EXPORT_V4
    if candidate.is_file():
        raw_size = candidate.stat().st_size
        validated = load_static_export(
            candidate,
            limits=StaticExportLimits(
                max_bytes=max(50 * 1024 * 1024, raw_size + 1024),
                max_posts=2000,
                max_comments_per_post=2000,
                max_comments=100000,
            ),
        )
        if not validated.valid:
            raise RuntimeError(f"Invalid static export {candidate}: {validated.error_codes}")
        raw_posts = []
        for p in validated.posts:
            raw_posts.append(
                {
                    "orig_id": p.reddit_id,
                    "title": p.title,
                    "body": p.body or "",
                    "subreddit": p.subreddit,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "score": int(p.score or 0),
                    "comments": [
                        {
                            "orig_id": c.reddit_id,
                            "body": c.body or "",
                            "created_at": c.created_at.isoformat() if c.created_at else None,
                            "score": int(c.score or 0),
                            "parent_orig_id": c.parent_reddit_id,
                        }
                        for c in p.comments
                    ],
                }
            )
        return raw_posts, str(candidate.name)

    fallback = LIVE_DATA_DIR / "corpus.json"
    if not fallback.is_file():
        raise FileNotFoundError(f"Neither {candidate} nor {fallback} exists.")
    blob = json.loads(fallback.read_text(encoding="utf-8"))
    raw_posts = []
    for p in blob.get("posts") or []:
        raw_posts.append(
            {
                "orig_id": p["reddit_id"],
                "title": p.get("title") or "",
                "body": p.get("body") or "",
                "subreddit": p.get("subreddit") or "unknown",
                "created_at": p.get("created_at"),
                "score": int(p.get("score") or 0),
                "comments": [
                    {
                        "orig_id": c["reddit_id"],
                        "body": c.get("body") or "",
                        "created_at": c.get("created_at"),
                        "score": int(c.get("score") or 0),
                        "parent_orig_id": c.get("parent_reddit_id"),
                    }
                    for c in p.get("comments") or []
                ],
            }
        )
    return raw_posts, "data/live_corpus/corpus.json"


def _load_real_labeled_metadata(labeled_dir: Path) -> Tuple[Dict[str, str], Dict[str, str], int, int]:
    """Read post_title & subreddit maps from tracked master_labeled_v1.csv."""
    master = labeled_dir / "master_labeled_v1.csv"
    if not master.is_file():
        raise FileNotFoundError(f"Missing real labeled dataset at {master}")
    post_titles: Dict[str, str] = {}
    doc_subreddits: Dict[str, str] = {}
    n_rows = 0
    with master.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            post_titles[row["post_id"]] = row.get("post_title") or ""
            doc_subreddits[row["document_id"]] = row.get("subreddit") or "unknown"
    return post_titles, doc_subreddits, n_rows, len(post_titles)


def main(
    static_export_path: Optional[Path] = None,
    *,
    labeled_dir: Path = DEFAULT_REAL_LABELED_DIR,
    dataset_id: str = "iim-sambalpur-real-reddit-v5",
    dataset_version: str = "5.0.0",
    n_topics: int = 10,
    dapt_epochs: int = 4,
    muril_epochs: int = 6,
    skip_dapt: bool = False,
) -> None:
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dapt_dir = Path(default_dapt_output_dir())

    # 1. Load real anonymized Reddit posts & pseudonymize all IDs / strip raw URLs
    raw_posts, source_label = _load_raw_posts(static_export_path)
    raw_posts.sort(key=lambda p: (p.get("created_at") or "", p["orig_id"]))

    preprocessor = MultilingualTextPreprocessor()
    script_detector = SimpleScriptDetector()

    pii_counts: Counter = Counter()
    automod_filtered = 0
    gif_filtered = 0
    empty_threads_skipped = 0

    clean_posts: List[Dict[str, Any]] = []
    flat_comments: List[Dict[str, Any]] = []
    unlabeled_dapt_texts: List[str] = []

    post_counter = 0
    cmt_counter = 0

    for post in raw_posts:
        title = _REDDIT_URL_RE.sub("[REDDIT_LINK]", sanitize_pii_text(post["title"], pii_counts) or post["title"]).strip()
        body = _REDDIT_URL_RE.sub("[REDDIT_LINK]", sanitize_pii_text(post["body"], pii_counts) or "").strip() if post["body"] else ""

        kept_raw_cmts: List[Dict[str, Any]] = []
        for c in sorted(post["comments"], key=lambda x: (x.get("created_at") or "", x["orig_id"])):
            raw_b = c["body"]
            if _is_automod_boilerplate(raw_b) or _BOT_EXTRA_RE.search(raw_b):
                automod_filtered += 1
                continue
            if _is_gif_only(raw_b):
                gif_filtered += 1
                continue
            c_body = _REDDIT_URL_RE.sub("[REDDIT_LINK]", sanitize_pii_text(raw_b, pii_counts) or raw_b).strip()
            if not c_body:
                continue
            kept_raw_cmts.append({**c, "clean_body": c_body})

        if not kept_raw_cmts:
            empty_threads_skipped += 1
            continue

        post_counter += 1
        pseudo_pid = f"post_{post_counter:04d}"
        if title:
            unlabeled_dapt_texts.append(f"{title}. {body}".strip() if body else title)

        orig_to_pseudo_cmt: Dict[str, str] = {}
        for c in kept_raw_cmts:
            cmt_counter += 1
            orig_to_pseudo_cmt[c["orig_id"]] = f"cmt_{cmt_counter:05d}"

        kept_comments: List[Dict[str, Any]] = []
        for c in kept_raw_cmts:
            pseudo_cid = orig_to_pseudo_cmt[c["orig_id"]]
            c_body = c["clean_body"]
            pre = preprocessor.preprocess(c_body)
            norm = pre.normalized_text.strip() or c_body
            lang = detect_indian_reddit_language(c_body, script_detector)
            doc_id = f"comment-{pseudo_cid}"
            unlabeled_dapt_texts.append(c_body)
            parent_pseudo = orig_to_pseudo_cmt.get(c.get("parent_orig_id") or "")
            c_obj = {
                "reddit_id": pseudo_cid,
                "document_id": doc_id,
                "post_id": pseudo_pid,
                "post_title": title,
                "subreddit": post["subreddit"],
                "body": c_body,
                "normalized_text": norm,
                "created_at": c["created_at"],
                "score": int(c["score"]),
                "parent_reddit_id": parent_pseudo,
                "language": lang["language"],
                "script": lang["script"],
                "is_code_mixed": bool(lang["is_code_mixed"]),
                "detected_languages": list(lang["detected_languages"]),
            }
            kept_comments.append(c_obj)
            flat_comments.append(c_obj)

        clean_posts.append(
            {
                "reddit_id": pseudo_pid,
                "title": title,
                "body": body,
                "subreddit": post["subreddit"],
                "created_at": post["created_at"],
                "score": int(post["score"]),
                "num_comments": len(kept_comments),
                "url": None,
                "source_mode": "public_anonymized_reddit_pseudonymized",
                "synthetic": False,
                "comments": kept_comments,
            }
        )

    sub_counts = Counter(p["subreddit"] for p in clean_posts)
    lang_counts = Counter(c["language"] for c in flat_comments)
    print(
        f"Loaded {len(clean_posts)} pseudonymized real threads with {len(flat_comments)} human comments "
        f"from {source_label} across {len(sub_counts)} subreddits; "
        f"filtered {automod_filtered} bot/AutoMod, {gif_filtered} GIF, {empty_threads_skipped} empty threads; "
        f"languages={dict(lang_counts.most_common())}.",
        flush=True,
    )

    # 2. Unsupervised DAPT MLM with held-out validation perplexity
    dapt_summary: Optional[Dict[str, Any]] = None
    saved_dapt_summary_path = dapt_dir / "dapt_summary.json"
    if skip_dapt and dapt_dir.is_dir() and saved_dapt_summary_path.is_file():
        saved = json.loads(saved_dapt_summary_path.read_text(encoding="utf-8"))
        if saved.get("val_documents") and saved.get("initial_val_perplexity") is not None:
            dapt_summary = saved
            backbone_id = str(dapt_dir)
            print(
                f"\n--- Stage 2: Reusing held-out-validated MuRIL-DAPT MLM checkpoint "
                f"(val_ppl {dapt_summary['initial_val_perplexity']} -> {dapt_summary['final_val_perplexity']} "
                f"on {dapt_summary['val_documents']} unseen texts) ---",
                flush=True,
            )
    if dapt_summary is None and len(unlabeled_dapt_texts) >= 40:
        print(
            f"\n--- Stage 2: Unsupervised MuRIL-DAPT MLM on {len(unlabeled_dapt_texts)} unlabeled real Reddit texts (with 12% held-out val split) ---",
            flush=True,
        )
        dapt_summary = run_muril_mlm_dapt(
            unlabeled_dapt_texts,
            DAPTConfig(
                output_dir=str(dapt_dir),
                epochs=dapt_epochs,
                batch_size=8,
                grad_accum_steps=2,
                val_fraction=0.12,
                learning_rate=5e-5,
                max_length=160,
                random_state=SEED,
                verbose=True,
            ),
        )
        backbone_id = str(dapt_dir)
        print(
            f"Completed DAPT MLM: {dapt_summary['train_documents']} train / {dapt_summary['val_documents']} held-out val docs, "
            f"held-out val_ppl {dapt_summary['initial_val_perplexity']} -> {dapt_summary['final_val_perplexity']} "
            f"({dapt_summary['elapsed_seconds']}s).",
            flush=True,
        )
    elif dapt_summary is None:
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
        tfidf_config=TFIDFConfig(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            max_features=8000,
            sublinear_tf=True,
        ),
        nmf_config=NMFConfig(
            n_topics=n_topics,
            top_n_terms=10,
            random_state=42,
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

    # 4. Load 100% REAL curated Reddit labeled dataset (ZERO synthetic training/test rows!)
    labeled_post_titles, labeled_doc_subs, total_real_labeled, distinct_labeled_threads = (
        _load_real_labeled_metadata(labeled_dir)
    )
    split_spec = SplitSpec(
        train_size=0.65, validation_size=0.15, test_size=0.20, random_state=SEED, stratify=True
    )

    def _get_real_splits(task_name: str) -> Tuple[List[TrainingRecord], List[TrainingRecord], List[TrainingRecord]]:
        records = load_annotation_dataset(TaskType(task_name), str(labeled_dir / f"{task_name}.csv"))
        split = split_records(
            records,
            labels=[r.label for r in records],
            spec=split_spec,
            groups=extract_groups(records, "post_id"),
        )
        tr, va, te = apply_split(records, split)
        leak = check_leakage(tr, va, te)
        assert not leak.get("leakage_detected"), f"Leakage in {task_name}: {leak}"
        return list(tr), list(va), list(te)

    comment_texts = [c["body"] for c in flat_comments]
    comment_with_title = [f"{c['post_title']} | {c['body']}" for c in flat_comments]
    post_title_texts = [f"{p['title']} | {p['body']}" if p["body"] else p["title"] for p in clean_posts]

    evaluation_tasks: Dict[str, Any] = {}
    arch_comp_by_task: Dict[str, Any] = {}
    error_analysis_by_task: Dict[str, Any] = {}

    # 4A. ISSUE
    print("\n--- Training & evaluating ISSUE on real curated Reddit splits ---", flush=True)
    iss_tr, iss_va, iss_te = _get_real_splits("issue")
    iss_preds, iss_probs, post_iss_out, iss_eval, iss_meta = _train_hybrid_and_infer(
        "issue",
        iss_tr,
        iss_va,
        iss_te,
        [f"{labeled_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_tr],
        [f"{labeled_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_va],
        [f"{labeled_post_titles.get(r.post_id or '', '')} | {r.text}" for r in iss_te],
        [labeled_doc_subs.get(r.document_id, "unknown") for r in iss_te],
        comment_with_title,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
        extra_infer_texts=post_title_texts,
        include_parent_title_probes=True,
    )
    assert post_iss_out is not None
    post_iss_preds, post_iss_probs = post_iss_out
    evaluation_tasks["issue"] = iss_eval
    arch_comp_by_task["issue"] = iss_meta["architecture_comparison"]
    error_analysis_by_task["issue"] = iss_meta["error_analysis"]

    # 4B. SENTIMENT
    print("\n--- Training & evaluating SENTIMENT on real curated Reddit splits ---", flush=True)
    sen_tr, sen_va, sen_te = _get_real_splits("sentiment")
    sen_preds, sen_probs, _, sen_eval, sen_meta = _train_hybrid_and_infer(
        "sentiment",
        sen_tr,
        sen_va,
        sen_te,
        [r.text for r in sen_tr],
        [r.text for r in sen_va],
        [r.text for r in sen_te],
        [labeled_doc_subs.get(r.document_id, "unknown") for r in sen_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["sentiment"] = sen_eval
    arch_comp_by_task["sentiment"] = sen_meta["architecture_comparison"]
    error_analysis_by_task["sentiment"] = sen_meta["error_analysis"]

    # 4C. EMOTION
    print("\n--- Training & evaluating EMOTION on real curated Reddit splits ---", flush=True)
    emo_tr, emo_va, emo_te = _get_real_splits("emotion")
    emo_preds, emo_probs, _, emo_eval, emo_meta = _train_hybrid_and_infer(
        "emotion",
        emo_tr,
        emo_va,
        emo_te,
        [r.text for r in emo_tr],
        [r.text for r in emo_va],
        [r.text for r in emo_te],
        [labeled_doc_subs.get(r.document_id, "unknown") for r in emo_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["emotion"] = emo_eval
    arch_comp_by_task["emotion"] = emo_meta["architecture_comparison"]
    error_analysis_by_task["emotion"] = emo_meta["error_analysis"]

    # 4D. RESPONSE
    print("\n--- Training & evaluating RESPONSE (6 classes) on real curated Reddit splits ---", flush=True)
    res_tr, res_va, res_te = _get_real_splits("response")
    res_preds, res_probs, _, res_eval, res_meta = _train_hybrid_and_infer(
        "response",
        res_tr,
        res_va,
        res_te,
        [r.text for r in res_tr],
        [r.text for r in res_va],
        [r.text for r in res_te],
        [labeled_doc_subs.get(r.document_id, "unknown") for r in res_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["response"] = res_eval
    arch_comp_by_task["response"] = res_meta["architecture_comparison"]
    error_analysis_by_task["response"] = res_meta["error_analysis"]

    # 4E. SOLUTION
    print("\n--- Training & evaluating SOLUTION (binary) on real curated Reddit splits ---", flush=True)
    sol_tr, sol_va, sol_te = _get_real_splits("solution")
    sol_preds, sol_probs, _, sol_eval, sol_meta = _train_hybrid_and_infer(
        "solution",
        sol_tr,
        sol_va,
        sol_te,
        [r.text for r in sol_tr],
        [r.text for r in sol_va],
        [r.text for r in sol_te],
        [labeled_doc_subs.get(r.document_id, "unknown") for r in sol_te],
        comment_texts,
        backbone_model_id=backbone_id,
        epochs=muril_epochs,
    )
    evaluation_tasks["solution"] = sol_eval
    arch_comp_by_task["solution"] = sol_meta["architecture_comparison"]
    error_analysis_by_task["solution"] = sol_meta["error_analysis"]
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

    resp_sol_total = 0
    resp_sol_agree = 0

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
        sol_conf = round(sol_prob if is_sol else not_sol_prob, 4)

        if resp_lbl == "solution":
            resp_sol_total += 1
            if is_sol:
                resp_sol_agree += 1

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
            "confidence_tier": classify_confidence_tier(sent_conf),
            "confidence_margin": probability_margin(list(sent_prob_map.values())),
            "abstained": False,
        }
        pred_tasks["emotion"][doc_id] = {
            "emotions": emo_lbls,
            "emotion_probabilities": emo_prob_map,
            "confidence": emo_conf,
            "confidence_tier": classify_confidence_tier(emo_conf),
        }
        pred_tasks["issue"][doc_id] = {
            "predicted_issue_labels": iss_lbls,
            "issue_probabilities": iss_prob_map,
            "confidence": iss_conf,
            "confidence_tier": classify_confidence_tier(iss_conf),
        }
        pred_tasks["response"][doc_id] = {
            "response_type": resp_lbl,
            "confidence": resp_conf,
            "confidence_tier": classify_confidence_tier(resp_conf),
            "confidence_margin": probability_margin(list(resp_prob_map.values())),
            "response_probabilities": resp_prob_map,
            "abstained": False,
            "topic_id": t_id,
            "topic_label": t_lbl,
        }
        pred_tasks["solution"][doc_id] = {
            "is_solution": is_sol,
            "solution_probability": sol_prob,
            "not_solution_probability": not_sol_prob,
            "confidence": sol_conf,
            "confidence_tier": classify_confidence_tier(sol_conf),
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
        ("issue", "issue"),
        ("solution", "solution"),
        ("sentiment", "sentiment"),
        ("response", "response_type"),
        ("emotion", "emotion"),
    ):
        evaluation_tasks[t_name]["real_corpus_prediction_counts"] = dict(label_counts_real[cnt_key])

    corpus_payload = {
        "metadata": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "synthetic": False,
            "source_provider": "arctic_shift",
            "source_mode": "public_anonymized_reddit_pseudonymized",
            "authorization_status": "public_anonymized_reddit",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "post_count": len(clean_posts),
            "comment_count": len(flat_comments),
            "subreddits": sorted(sub_counts.keys()),
            "subreddit_counts": dict(sub_counts.most_common()),
            "language_counts": dict(lang_counts.most_common()),
            "filtering_summary": {
                "automod_and_bot_comments_removed": automod_filtered,
                "gif_only_comments_removed": gif_filtered,
                "empty_threads_removed": empty_threads_skipped,
                "extra_pii_redactions": dict(pii_counts),
                "pseudonymized_ids": True,
                "raw_reddit_urls_stripped": True,
            },
            "dapt_summary": dapt_summary,
            "notice": (
                f"Real anonymized & pseudonymized public Reddit corpus ({len(clean_posts)} threads, "
                f"{len(flat_comments)} human comments across {len(sub_counts)} subreddits; zero usernames, "
                f"zero raw Reddit IDs/URLs). Models trained & evaluated strictly on {total_real_labeled} "
                f"rubric-curated real Reddit comments (0 synthetic rows); multi-human adjudication ongoing."
            ),
        },
        "posts": clean_posts,
    }

    # Aggregate 4-architecture benchmark summary across all 5 tasks
    arch_keys = (
        "logistic_regression",
        "ann_mlp",
        "cnn_textcnn",
        "transformer_muril_dapt_v3",
        "production_hybrid",
    )
    arch_aggregate: Dict[str, Any] = {}
    probe_cat_agg: Dict[str, Dict[str, List[float]]] = {a: {} for a in arch_keys}
    task_names = ("issue", "solution", "sentiment", "response", "emotion")
    for a in arch_keys:
        entries = [arch_comp_by_task[t][a] for t in task_names if a in arch_comp_by_task.get(t, {})]
        for e in entries:
            for cat_name, cat_stats in (e.get("probe_by_category") or {}).items():
                score = cat_stats.get("macro_f1") if isinstance(cat_stats, dict) else None
                if score is not None:
                    probe_cat_agg[a].setdefault(cat_name, []).append(float(score))
        arch_aggregate[a] = {
            "label": entries[0]["label"] if entries else a,
            "family": entries[0]["family"] if entries else a,
            "mean_test_macro_f1": round(sum(e["test_macro_f1"] for e in entries) / max(len(entries), 1), 4),
            "mean_probe_macro_f1": round(sum(e["probe_macro_f1"] for e in entries) / max(len(entries), 1), 4),
            "generalization_gap": round(
                (sum(e["test_macro_f1"] - e["probe_macro_f1"] for e in entries)) / max(len(entries), 1), 4
            ),
            "mean_calibration_ece": round(sum(e["calibration_ece"] for e in entries) / max(len(entries), 1), 4),
            "total_train_seconds": round(sum(e["train_seconds"] for e in entries), 2),
            "mean_inference_ms": round(sum(e["inference_ms_per_comment"] for e in entries) / max(len(entries), 1), 3),
            "param_count": entries[0]["param_count"] if entries else 0,
            "by_category_mean_f1": {
                c: round(sum(vals) / len(vals), 4) for c, vals in sorted(probe_cat_agg[a].items())
            },
        }

    t_test = arch_aggregate["transformer_muril_dapt_v3"]["mean_test_macro_f1"]
    h_test = arch_aggregate["production_hybrid"]["mean_test_macro_f1"]
    lr_test = arch_aggregate["logistic_regression"]["mean_test_macro_f1"]
    ann_test = arch_aggregate["ann_mlp"]["mean_test_macro_f1"]
    cnn_test = arch_aggregate["cnn_textcnn"]["mean_test_macro_f1"]
    t_ood = arch_aggregate["transformer_muril_dapt_v3"]["mean_probe_macro_f1"]
    h_ood = arch_aggregate["production_hybrid"]["mean_probe_macro_f1"]

    why_transformer = {
        "selected_architecture": "MuRIL-DAPT v3 Transformer (+ validation-calibrated Word+Char TF-IDF Hybrid, alpha >= 0.50)",
        "primary_eval_split": "heldout_real_reddit_test",
        "mean_test_f1_transformer": t_test,
        "mean_test_f1_hybrid": h_test,
        "mean_ood_f1_transformer": t_ood,
        "mean_ood_f1_hybrid": h_ood,
        "gain_vs_logreg_pp": round((t_test - lr_test) * 100.0, 2),
        "gain_vs_ann_pp": round((t_test - ann_test) * 100.0, 2),
        "gain_vs_cnn_pp": round((t_test - cnn_test) * 100.0, 2),
        "hybrid_gain_vs_logreg_pp": round((h_test - lr_test) * 100.0, 2),
        "key_reasons": [
            {
                "title": "Real unseen-thread Reddit benchmark (0 synthetic data)",
                "evidence": (
                    f"Trained and tested strictly on {total_real_labeled} curated real Reddit comments across "
                    f"{distinct_labeled_threads} distinct threads (post_id-grouped 65/15/20): MuRIL-DAPT v3 achieves "
                    f"{t_test*100:.1f}% mean held-out real test Macro F1 (Hybrid {h_test*100:.1f}%) vs. "
                    f"LogReg {lr_test*100:.1f}%, ANN {ann_test*100:.1f}%, CNN {cnn_test*100:.1f}%."
                ),
            },
            {
                "title": "Auxiliary 54-probe stress diagnostic (never used for selection)",
                "evidence": (
                    f"On the untouched 54-probe stress diagnostic across slang, paraphrases, typos, and Hinglish, "
                    f"MuRIL-DAPT v3 retains {t_ood*100:.1f}% mean Macro F1 (Hybrid {h_ood*100:.1f}%) compared to "
                    f"LogReg {arch_aggregate['logistic_regression']['mean_probe_macro_f1']*100:.1f}%, "
                    f"ANN {arch_aggregate['ann_mlp']['mean_probe_macro_f1']*100:.1f}%, and "
                    f"CNN {arch_aggregate['cnn_textcnn']['mean_probe_macro_f1']*100:.1f}%."
                ),
            },
            {
                "title": "Held-out validated DAPT MLM domain adaptation",
                "evidence": (
                    f"Masked Language Modeling on {dapt_summary['train_documents']:,} unlabeled real Reddit texts "
                    f"reduced held-out validation perplexity from {dapt_summary['initial_val_perplexity']} -> "
                    f"{dapt_summary['final_val_perplexity']} on {dapt_summary['val_documents']:,} disjoint unseen texts."
                    if dapt_summary
                    else "Domain-adaptive MLM adapts MuRIL to Indian MBA/CAT vocabulary."
                ),
            },
            {
                "title": "Non-sharpening calibration (T >= 1.0) & static serving",
                "evidence": (
                    f"Validation temperature scaling strictly clamped to T >= 1.0 yields honest probability bounds "
                    f"(mean held-out ECE {arch_aggregate['production_hybrid']['mean_calibration_ece']:.3f}), and "
                    f"precomputed artifacts serve Vercel/Docker in <10 ms with zero runtime GPU cost."
                ),
            },
        ],
    }

    architecture_comparison_payload = {
        "benchmark_protocol": (
            f"Apples-to-apples evaluation across all 5 tasks on identical post_id-grouped 65/15/20 splits "
            f"of {total_real_labeled} curated real Reddit comments ({distinct_labeled_threads} distinct threads, "
            f"seed={SEED}, 0 synthetic rows). Auxiliary 54-probe stress scores are observational only."
        ),
        "primary_metric": "test_macro_f1",
        "architectures_order": list(arch_keys),
        "aggregate": arch_aggregate,
        "per_task": arch_comp_by_task,
        "why_transformer": why_transformer,
    }

    labeled_provenance = {
        "source_path": "data/annotations/real_reddit_labeled_v1",
        "annotation_source": "rubric_curated_real_reddit_v1_pending_multi_human_kappa",
        "labeled_real_comments": total_real_labeled,
        "distinct_threads": distinct_labeled_threads,
        "synthetic_training_rows": 0,
        "split_strategy": "post_id_grouped_65_15_20",
        "multi_annotator_human_kappa_completed": False,
    }

    resp_sol_consistency = {
        "response_solution_count": resp_sol_total,
        "agreed_with_solution_model": resp_sol_agree,
        "agreement_rate": round(resp_sol_agree / max(resp_sol_total, 1), 4),
    }

    predictions_payload = {"synthetic_only": False, "dataset_version": dataset_version, "tasks": pred_tasks}
    evaluation_payload = {
        "synthetic_only": False,
        "dataset_version": dataset_version,
        "labeled_data_provenance": labeled_provenance,
        "dapt_summary": dapt_summary,
        "response_solution_consistency": resp_sol_consistency,
        "architecture_comparison": architecture_comparison_payload,
        "error_analysis": error_analysis_by_task,
        "evaluation_scope": (
            f"100% real anonymized & pseudonymized Reddit corpus ({len(clean_posts)} threads, {len(flat_comments)} "
            f"human comments across {len(sub_counts)} subreddits). All 4 architectures trained and evaluated strictly "
            f"on post_id-grouped splits of {total_real_labeled} rubric-curated real Reddit comments (0 synthetic rows), "
            f"with held-out DAPT validation perplexity, 5-bin held-out calibration (T >= 1.0), and data-driven error analysis."
        ),
        "tasks": evaluation_tasks,
    }

    verification_report = verify_benchmark_artifact(
        evaluation_payload, predictions=predictions_payload, corpus=corpus_payload
    )
    evaluation_payload["benchmark_verification"] = verification_report
    print(
        f"\n[PIPELINE INTEGRITY VERIFIER] verified={verification_report['verified']} "
        f"({verification_report['passed_checks']}/{verification_report['total_checks']} checks passed)",
        flush=True,
    )
    for chk in verification_report["checks"]:
        print(f"  [{'PASS' if chk['passed'] else 'FAIL'}] {chk['id']}: {chk['detail']}", flush=True)
    print(
        f"[CROSS-TASK CONSISTENCY] response=='solution' agreeing with is_solution==True: "
        f"{resp_sol_agree}/{resp_sol_total} ({resp_sol_consistency['agreement_rate']*100:.1f}%)",
        flush=True,
    )
    print(f"[REAL CORPUS SENTIMENT COUNTS] {dict(label_counts_real['sentiment'])}", flush=True)

    (LIVE_DATA_DIR / "corpus.json").write_text(
        json.dumps(corpus_payload, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (LIVE_ARTIFACT_DIR / "predictions.json").write_text(
        json.dumps(predictions_payload, separators=(",", ":"), ensure_ascii=False) + "\n",
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
        json.dumps(evaluation_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"\nBuilt 100%-real anonymized Reddit v{dataset_version} bundle:\n"
        f"  {LIVE_DATA_DIR / 'corpus.json'} ({(LIVE_DATA_DIR / 'corpus.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'predictions.json'} ({(LIVE_ARTIFACT_DIR / 'predictions.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'topic_discovery.json'} ({(LIVE_ARTIFACT_DIR / 'topic_discovery.json').stat().st_size / 1024:.1f} KB)\n"
        f"  {LIVE_ARTIFACT_DIR / 'evaluation.json'} ({(LIVE_ARTIFACT_DIR / 'evaluation.json').stat().st_size / 1024:.1f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Sanitized format_version=1 static export JSON (falls back to data/live_corpus/corpus.json)")
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_REAL_LABELED_DIR, help="Tracked directory of real labeled per-task CSVs")
    parser.add_argument("--dataset-id", default="iim-sambalpur-real-reddit-v5")
    parser.add_argument("--dataset-version", default="5.0.0")
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--dapt-epochs", type=int, default=4)
    parser.add_argument("--muril-epochs", type=int, default=12)
    parser.add_argument("--skip-dapt", action="store_true")
    args = parser.parse_args()
    main(
        static_export_path=args.input,
        labeled_dir=args.labeled_dir,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        n_topics=args.n_topics,
        dapt_epochs=args.dapt_epochs,
        muril_epochs=args.muril_epochs,
        skip_dapt=args.skip_dapt,
    )
