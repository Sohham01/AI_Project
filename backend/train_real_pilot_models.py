"""Real-data pilot model trainer:
  1. Unsupervised NMF Topic Discovery (runs immediately on sanitized real Reddit
     exports — requires ZERO labels).
  2. Supervised 5-Task Training & Benchmarking (Sentiment, Emotion, Issue,
     Response, Solution) comparing Baseline V1 TF-IDF, Upgraded V2 Word+Char
     Calibrated TF-IDF, and optional GPU MuRIL — STRICTLY gated by human
     annotation readiness (``validate_pilot_annotations.py``).

SCIENTIFIC INTEGRITY & ISOLATION GUARANTEES:
  - Never fabricates ground-truth labels for unlabeled Reddit comments.
  - Never converts ``uncertain`` / ``abstain`` / ``excluded`` items into
    training classes.
  - Always uses thread-grouped (``post_id``) train/val/test splits with
    ``check_leakage()`` verification and validation-only threshold calibration.
  - Writes all artifacts into an isolated gitignored directory
    (default: ``ml/artifacts/real_arctic_shift_v1/``). Never modifies
    ``ml/artifacts/demo_synthetic/``, ``ml/artifacts/demo_synthetic_v2/``,
    ``ml/artifacts/registry.json``, or ``/api/demo``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
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
from app.services.nlp.solution_detection.base import (
    SolutionDocument,
    SolutionModelConfig,
)
from app.services.nlp.solution_detection.tfidf_logistic_binary import (
    TfidfLogisticSolutionClassifier,
)
from app.services.nlp.topic_discovery.base import NMFConfig, TFIDFConfig, TopicDocument
from app.services.nlp.topic_discovery.nmf_topic_discoverer import NMFTopicDiscoverer
from backend.validate_pilot_annotations import validate_and_split_pilot_csv
from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import (
    SplitSpec,
    apply_split,
    check_leakage,
    extract_groups,
    split_records,
)
from ml.training.trainer import records_to_annotations

TASKS = ("sentiment", "emotion", "issue", "response", "solution")
DEFAULT_SEED = 20261119
_GIF_ONLY_RE = re.compile(r"^\s*!\[gif\]\([^)]+\)\s*$", re.IGNORECASE)


def _is_automod_boilerplate(text: str) -> bool:
    low = (text or "").lower()
    return (
        "this action was performed automatically" in low
        or "i am a bot, and this action was performed automatically" in low
        or ("i am a bot" in low and "contact the moderators" in low)
    )


def _is_gif_only(text: str) -> bool:
    return bool(_GIF_ONLY_RE.fullmatch((text or "").strip()))


# ---------------------------------------------------------------------------
# Metrics (shared exact definitions matching robustness benchmark)
# ---------------------------------------------------------------------------


def score_multilabel(
    gold: Sequence[Sequence[str]],
    preds: Sequence[Sequence[str]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    l_count = len(labels)
    exact = 0
    hamming_errors = 0
    total_tp = total_fp = total_fn = 0
    per_label: Dict[str, Dict[str, Any]] = {}
    f1s: List[float] = []

    gold_sets = [set(g) for g in gold]
    pred_sets = [set(p) for p in preds]

    for g_set, p_set in zip(gold_sets, pred_sets):
        if g_set == p_set:
            exact += 1
        for lbl in labels:
            if (lbl in g_set) != (lbl in p_set):
                hamming_errors += 1

    for lbl in labels:
        tp = sum((lbl in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fp = sum((lbl not in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fn = sum((lbl in g) and (lbl not in p) for g, p in zip(gold_sets, pred_sets))
        support = sum(lbl in g for g in gold_sets)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_label[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_denom) if micro_denom > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {
        "n_samples": n,
        "multilabel": True,
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "subset_accuracy": round(exact / n, 6) if n > 0 else 0.0,
        "hamming_loss": round(hamming_errors / (n * l_count), 6) if (n * l_count) > 0 else 0.0,
        "per_label": per_label,
    }


def score_singlelabel(
    gold: Sequence[str],
    preds: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, preds))
    f1s: List[float] = []
    per_class: Dict[str, Dict[str, Any]] = {}
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for g, p in zip(gold, preds):
        if g in label_idx and p in label_idx:
            cm[label_idx[g]][label_idx[p]] += 1

    for lbl in labels:
        tp = sum((g == lbl) and (p == lbl) for g, p in zip(gold, preds))
        fp = sum((g != lbl) and (p == lbl) for g, p in zip(gold, preds))
        fn = sum((g == lbl) and (p != lbl) for g, p in zip(gold, preds))
        support = sum(g == lbl for g in gold)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_class[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }

    acc = correct / n if n > 0 else 0.0
    return {
        "n_samples": n,
        "multilabel": False,
        "accuracy": round(acc, 6),
        "micro_f1": round(acc, 6),
        "macro_f1": round(float(np.mean(f1s)), 6) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": cm,
    }


def _parse_targets(task_name: str, rows: Sequence[Any]) -> List[Any]:
    if task_name in {"emotion", "issue"}:
        return [
            sorted(
                part.strip().lower()
                for part in str(r.label or "").replace(",", "|").split("|")
                if part.strip()
            )
            for r in rows
        ]
    return [str(r.label).strip().lower() for r in rows]


# ---------------------------------------------------------------------------
# Stage A: Unsupervised Topic Discovery on Real Sanitized Exports
# ---------------------------------------------------------------------------


def train_unsupervised_topic_model(
    static_export_path: Path,
    out_dir: Path,
    *,
    n_topics: int = 8,
    n_keywords: int = 10,
    stop_words: Optional[str] = "english",
    include_post_titles: bool = False,
    filter_bot_noise: bool = True,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Fit ``NMFTopicDiscoverer`` on validated sanitized real Reddit exports.

    Requires zero ground-truth labels and preserves full provenance.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_size = static_export_path.stat().st_size
    limits = StaticExportLimits(
        max_bytes=max(20 * 1024 * 1024, raw_size + 1024),
        max_posts=500,
        max_comments_per_post=2000,
        max_comments=25000,
    )
    validated = load_static_export(static_export_path, limits=limits)
    if not validated.valid:
        raise ValueError(
            f"Static export failed validation ({len(validated.error_codes)} error(s)): "
            f"{list(validated.error_codes[:10])}"
        )

    preprocessor = MultilingualTextPreprocessor()
    lang_service = LanguageAnalysisService(
        language_detector=SimpleLanguageDetector(),
        script_detector=SimpleScriptDetector(),
        code_mix_detector=SimpleCodeMixDetector(),
    )

    topic_docs: List[TopicDocument] = []
    subreddit_counts: Counter = Counter()
    language_counts: Counter = Counter()
    skipped_short = 0
    automod_filtered = 0
    gif_only_filtered = 0

    for post in validated.posts:
        subreddit_counts[post.subreddit] += 1
        if include_post_titles:
            post_text = f"{post.title} {post.body or ''}".strip()
            pre = preprocessor.preprocess(post_text)
            if len(pre.normalized_text.strip()) >= 4:
                lang = lang_service.analyze(pre.normalized_text)
                language_counts[lang.language] += 1
                topic_docs.append(
                    TopicDocument(
                        document_id=f"post:{post.reddit_id}",
                        post_id=post.reddit_id,
                        comment_id=None,
                        original_text=post_text,
                        text=pre.normalized_text,
                        language=lang.language,
                        script=lang.script,
                        is_code_mixed=lang.is_code_mixed,
                        detected_languages=list(lang.detected_languages),
                    )
                )
        for comment in post.comments:
            if filter_bot_noise and _is_automod_boilerplate(comment.body):
                automod_filtered += 1
                continue
            if filter_bot_noise and _is_gif_only(comment.body):
                gif_only_filtered += 1
                continue
            pre = preprocessor.preprocess(comment.body)
            norm = pre.normalized_text.strip()
            if len(norm) < 4:
                skipped_short += 1
                continue
            lang = lang_service.analyze(norm)
            language_counts[lang.language] += 1
            topic_docs.append(
                TopicDocument(
                    document_id=f"comment:{comment.reddit_id}",
                    post_id=post.reddit_id,
                    comment_id=comment.reddit_id,
                    original_text=comment.body,
                    text=norm,
                    language=lang.language,
                    script=lang.script,
                    is_code_mixed=lang.is_code_mixed,
                    detected_languages=list(lang.detected_languages),
                )
            )

    if len(topic_docs) < 2:
        raise ValueError(
            f"Need at least 2 usable documents for NMF topic discovery, got {len(topic_docs)}."
        )

    effective_n_topics = max(2, min(n_topics, len(topic_docs) // 3 if len(topic_docs) >= 6 else 2))
    min_df = 2 if len(topic_docs) >= 25 else 1
    max_df = 0.85 if len(topic_docs) >= 10 else 1.0
    sw = None if (stop_words is None or stop_words.lower() == "none") else stop_words

    tfidf_cfg = TFIDFConfig(
        max_features=10000,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        stop_words=sw,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w{2,}\b",
    )
    nmf_cfg = NMFConfig(
        n_topics=effective_n_topics,
        n_keywords=n_keywords,
        random_state=seed,
        max_iter=400,
        drop_placeholder_tokens=True,
    )

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    discoverer = NMFTopicDiscoverer(
        n_topics=effective_n_topics,
        n_keywords=n_keywords,
        tfidf_config=tfidf_cfg,
        nmf_config=nmf_cfg,
        include_timestamp=True,
    )
    result = discoverer.discover(topic_docs)
    fit_seconds = round(time.perf_counter() - t0, 3)
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    topic_artifact_path = out_dir / "topic_discovery.json"
    topic_artifact_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    topics_compact = [
        {
            "topic_id": t.topic_id,
            "label": t.label,
            "document_count": t.document_count,
            "document_ratio": t.document_ratio,
            "keywords": t.keywords,
        }
        for t in result.topics
    ]

    source_meta = validated.metadata.get("source") if isinstance(validated.metadata, dict) else {}
    if not isinstance(source_meta, dict):
        source_meta = {}
    return {
        "status": "trained",
        "model": f"{result.topic_model}_{result.topic_model_version}",
        "requires_labels": False,
        "dataset_id": (validated.metadata or {}).get("dataset_id"),
        "source_provider": source_meta.get("source_provider"),
        "static_export_path": str(static_export_path),
        "artifact_path": str(topic_artifact_path),
        "artifact_size_kb": round(topic_artifact_path.stat().st_size / 1024.0, 2),
        "threads_loaded": len(validated.posts),
        "documents_modeled": result.n_documents,
        "include_post_titles": include_post_titles,
        "filter_bot_noise": filter_bot_noise,
        "automod_boilerplate_filtered": automod_filtered,
        "gif_only_filtered": gif_only_filtered,
        "skipped_short_comments": skipped_short,
        "vocabulary_size": result.vocabulary_size,
        "n_topics": result.n_topics,
        "fit_seconds": fit_seconds,
        "peak_cpu_mem_mb": round(peak_mem_bytes / (1024.0 * 1024.0), 2),
        "subreddit_thread_counts": dict(sorted(subreddit_counts.items())),
        "detected_language_counts": dict(sorted(language_counts.items())),
        "topics": topics_compact,
        "warnings": list(dict.fromkeys(result.warnings)),
        "warning_occurrences": len(result.warnings),
        "limitations": (
            "TF-IDF + NMF is an unsupervised bag-of-words topic model: it captures "
            "lexical co-occurrence without word order, synonymy, or cross-lingual "
            "alignment, and short Hinglish comments can trigger noisy language tags."
        ),
    }


# ---------------------------------------------------------------------------
# Stage B: Strictly Gated Supervised Training (Real Human-Annotated Pilot)
# ---------------------------------------------------------------------------


def _build_classical_docs(task_name: str, texts: Sequence[str]) -> List[Any]:
    if task_name == "issue":
        return [IssueDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
    if task_name in {"sentiment", "emotion"}:
        return [SentimentEmotionDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
    if task_name == "response":
        return [ResponseDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]
    return [SolutionDocument(document_id=f"d{i}", text=t) for i, t in enumerate(texts)]


def _predict_classical(model: Any, task_name: str, texts: Sequence[str]) -> Tuple[List[Any], float]:
    docs = _build_classical_docs(task_name, texts)
    t0 = time.perf_counter()
    res = model.predict(docs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if task_name == "issue":
        return [p.predicted_issue_labels for p in res.predictions], elapsed_ms
    if task_name == "emotion":
        return [p.emotions for p in res.predictions], elapsed_ms
    if task_name == "sentiment":
        return [p.sentiment or "" for p in res.predictions], elapsed_ms
    if task_name == "response":
        return [p.response_type or "" for p in res.predictions], elapsed_ms
    return ["solution" if p.is_solution else "not_solution" for p in res.predictions], elapsed_ms


def _fit_and_eval_classical(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    *,
    upgraded: bool,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    task = TaskType(task_name)
    if task_name == "issue":
        cfg = IssueModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else IssueModelConfig()
        model = TfidfLogisticIssueClassifier(config=cfg)
    elif task_name == "emotion":
        cfg = EmotionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else EmotionModelConfig()
        model = TfidfLogisticEmotionClassifier(config=cfg)
    elif task_name == "sentiment":
        cfg = SentimentModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else SentimentModelConfig()
        model = TfidfLogisticSentimentClassifier(config=cfg)
    elif task_name == "response":
        cfg = ResponseModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else ResponseModelConfig()
        model = TfidfLogisticResponseClassifier(config=cfg)
    else:
        cfg = SolutionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else SolutionModelConfig()
        model = TfidfLogisticSolutionClassifier(config=cfg)

    train_ann = records_to_annotations(train_rows, task)
    val_ann = records_to_annotations(val_rows, task)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model.train(train_ann)
    calibrated_thresholds = None
    if upgraded and hasattr(model, "calibrate_thresholds") and val_ann:
        calibrated_thresholds = model.calibrate_thresholds(val_ann)
    train_seconds = round(time.perf_counter() - t0, 4)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    test_texts = [r.text for r in test_rows]
    gold_test = _parse_targets(task_name, test_rows)
    test_preds, infer_ms = _predict_classical(model, task_name, test_texts)

    artifact_kb = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        artifact_kb = round(save_path.stat().st_size / 1024.0, 2)

    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    return {
        "labels": list(model.labels),
        "calibrated_thresholds": calibrated_thresholds,
        "artifact_path": str(save_path) if save_path is not None else None,
        "resources": {
            "train_seconds": train_seconds,
            "test_inference_ms_total": round(infer_ms, 3),
            "test_inference_ms_per_comment": round(infer_ms / max(len(test_texts), 1), 4),
            "peak_cpu_mem_mb": round(peak_mem / (1024.0 * 1024.0), 2),
            "peak_gpu_vram_mb": 0.0,
            "artifact_size_kb": artifact_kb,
        },
        "held_out_test": scorer(gold_test, test_preds, model.labels),
    }


def _fit_and_eval_muril(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    *,
    epochs: int,
    seed: int,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    import torch
    from ml.models.transformer.muril_classifier import MuRILTaskClassifier, MuRILTaskConfig

    tr_t = [r.text for r in train_rows]
    va_t = [r.text for r in val_rows]
    te_t = [r.text for r in test_rows]
    tr_y = _parse_targets(task_name, train_rows)
    va_y = _parse_targets(task_name, val_rows)
    te_y = _parse_targets(task_name, test_rows)

    cfg = MuRILTaskConfig(
        task=task_name,
        epochs=epochs,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        max_length=160,
        random_state=seed,
        verbose=True,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    t0 = time.perf_counter()
    clf = MuRILTaskClassifier(cfg)
    fit_summary = clf.fit(tr_t, tr_y, va_t, va_y)
    train_seconds = round(time.perf_counter() - t0, 3)
    _, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram = (
        round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        if torch.cuda.is_available()
        else 0.0
    )

    t_inf = time.perf_counter()
    test_preds, _ = clf.predict_labels(te_t)
    infer_ms = (time.perf_counter() - t_inf) * 1000.0

    artifact_kb = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        clf.save(save_dir)
        artifact_kb = round(
            sum(p.stat().st_size for p in save_dir.rglob("*") if p.is_file()) / 1024.0,
            2,
        )

    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    return {
        "labels": list(clf.labels),
        "device": fit_summary["device"],
        "amp_dtype": fit_summary["amp_dtype"],
        "best_epoch": fit_summary["best_epoch"],
        "best_val_macro_f1": fit_summary["best_val_macro_f1"],
        "calibrated_thresholds": fit_summary["per_label_thresholds"] if clf.is_multilabel else None,
        "artifact_path": str(save_dir) if save_dir is not None else None,
        "resources": {
            "train_seconds": train_seconds,
            "test_inference_ms_total": round(infer_ms, 2),
            "test_inference_ms_per_comment": round(infer_ms / max(len(te_t), 1), 3),
            "peak_cpu_mem_mb": round(peak_cpu / (1024.0 * 1024.0), 2),
            "peak_gpu_vram_mb": peak_vram,
            "artifact_size_kb": artifact_kb,
        },
        "held_out_test": scorer(te_y, test_preds, clf.labels),
    }


def _thread_grouped_split(
    records: Sequence[Any],
    *,
    seed: int,
) -> Tuple[List[Any], List[Any], List[Any], Dict[str, Any]]:
    groups = extract_groups(records, "post_id")
    for stratify in (True, False):
        try:
            spec = SplitSpec(
                train_size=0.70,
                validation_size=0.15,
                test_size=0.15,
                random_state=seed,
                stratify=stratify,
            )
            split = split_records(
                records,
                labels=[r.label for r in records],
                spec=spec,
                groups=groups,
            )
            train_rows, val_rows, test_rows = apply_split(records, split)
            leak = check_leakage(train_rows, val_rows, test_rows)
            if leak.get("leakage_detected"):
                raise RuntimeError(f"Thread leakage detected: {leak}")
            return (
                train_rows,
                val_rows,
                test_rows,
                {
                    "group_by": "post_id",
                    "stratified": stratify,
                    "leakage_detected": False,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "n_test": len(test_rows),
                    "n_train_threads": len({r.post_id for r in train_rows if r.post_id}),
                    "n_val_threads": len({r.post_id for r in val_rows if r.post_id}),
                    "n_test_threads": len({r.post_id for r in test_rows if r.post_id}),
                },
            )
        except ValueError:
            if not stratify:
                raise
    raise RuntimeError("Failed to compute thread-grouped split.")  # pragma: no cover


def train_supervised_real_pilot(
    annotation_csv_path: Path,
    out_dir: Path,
    *,
    min_comments: int = 60,
    min_threads: int = 10,
    min_per_class: int = 3,
    include_muril: bool = False,
    muril_epochs: int = 6,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Validate human annotations and train all 5 supervised tasks ONLY when ready."""
    out_dir.mkdir(parents=True, exist_ok=True)
    task_csv_dir = out_dir / "validated_task_csvs"
    validation = validate_and_split_pilot_csv(
        annotation_csv_path,
        out_dir=task_csv_dir,
        min_labeled_comments=min_comments,
        min_labeled_threads=min_threads,
        min_examples_per_class=min_per_class,
    )

    if not validation["ready_for_training"]:
        return {
            "status": "blocked_awaiting_human_labels",
            "ready_for_training": False,
            "annotation_csv": str(annotation_csv_path),
            "labeled_comments": validation["labeled_comments"],
            "labeled_threads": validation["labeled_threads"],
            "unreviewed_count": validation["unreviewed_count"],
            "adjudication_queue_count": validation["adjudication_queue_count"],
            "readiness_blockers": validation["readiness_blockers"],
            "validation_errors": validation["errors"],
            "note": (
                "Supervised training did not run because genuine human annotations "
                "have not yet passed readiness thresholds. Labels were NOT fabricated."
            ),
        }

    models_dir = out_dir / "models"
    task_reports: Dict[str, Any] = {}
    for task_name in TASKS:
        csv_file = Path(validation["written_files"][task_name])
        records = load_annotation_dataset(TaskType(task_name), str(csv_file))
        train_rows, val_rows, test_rows, split_info = _thread_grouped_split(
            records, seed=seed
        )
        v1 = _fit_and_eval_classical(
            task_name,
            train_rows,
            val_rows,
            test_rows,
            upgraded=False,
            save_path=models_dir / f"{task_name}_v1_tfidf.joblib",
        )
        v2 = _fit_and_eval_classical(
            task_name,
            train_rows,
            val_rows,
            test_rows,
            upgraded=True,
            save_path=models_dir / f"{task_name}_v2_word_char_calibrated.joblib",
        )
        entry: Dict[str, Any] = {
            "split": split_info,
            "baseline_v1_tfidf": v1,
            "upgraded_v2_word_char_calibrated": v2,
        }
        if include_muril:
            entry["muril_v2_gpu"] = _fit_and_eval_muril(
                task_name,
                train_rows,
                val_rows,
                test_rows,
                epochs=muril_epochs,
                seed=seed,
                save_dir=models_dir / f"muril_{task_name}",
            )
        task_reports[task_name] = entry

    small_pilot_caveat = (
        "Small pilot sample (<250 labeled comments): metrics validate pipeline "
        "end-to-end execution rather than statistically stable population generalization."
        if validation["labeled_comments"] < 250
        else None
    )
    return {
        "status": "trained",
        "ready_for_training": True,
        "evaluation_provenance": "real_human_annotated_pilot",
        "annotation_csv": str(annotation_csv_path),
        "labeled_comments": validation["labeled_comments"],
        "labeled_threads": validation["labeled_threads"],
        "adjudication_queue_count": validation["adjudication_queue_count"],
        "small_pilot_caveat": small_pilot_caveat,
        "tasks": task_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-export",
        default="data/processed/arctic_shift_pilot_corpus_v1/sanitized_static_export.json",
        help="Validated format_version=1 real static export JSON for unsupervised topic modeling",
    )
    parser.add_argument(
        "--annotations",
        default="data/annotations/arctic_shift_pilot_corpus_v1/real_annotation_candidates.csv",
        help="Human-annotated unified 5-task CSV for strictly gated supervised training",
    )
    parser.add_argument(
        "--out-dir",
        default="ml/artifacts/real_arctic_shift_v1",
        help="Isolated gitignored artifact directory (never overwrites demo_synthetic/registry)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "unsupervised", "supervised"),
        default="auto",
        help="auto: train unsupervised now + supervised only if human-label gate passes",
    )
    parser.add_argument("--n-topics", type=int, default=8, help="Target NMF topic count (default 8)")
    parser.add_argument("--n-keywords", type=int, default=10, help="Keywords per NMF topic (default 10)")
    parser.add_argument("--stop-words", default="english", help="TF-IDF stop_words ('english' or 'none')")
    parser.add_argument("--include-post-titles", action="store_true", help="Include post titles+bodies alongside comments in NMF")
    parser.add_argument("--keep-bot-noise", action="store_true", help="Keep AutoModerator boilerplate & GIF-only comments in NMF")
    parser.add_argument("--include-muril", action="store_true", help="Also fine-tune GPU MuRIL when supervised gate passes")
    parser.add_argument("--muril-epochs", type=int, default=6)
    parser.add_argument("--min-comments", type=int, default=60)
    parser.add_argument("--min-threads", type=int, default=10)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": args.mode,
        "out_dir": str(out_dir),
        "isolated_from_demo_and_registry": True,
        "unsupervised_topic_discovery": None,
        "supervised_models": None,
    }

    if args.mode in ("auto", "unsupervised"):
        export_path = Path(args.static_export)
        if not export_path.exists():
            print(f"Error: static export not found: {export_path}", file=sys.stderr)
            return 2
        summary["unsupervised_topic_discovery"] = train_unsupervised_topic_model(
            export_path,
            out_dir,
            n_topics=args.n_topics,
            n_keywords=args.n_keywords,
            stop_words=args.stop_words,
            include_post_titles=args.include_post_titles,
            filter_bot_noise=not args.keep_bot_noise,
            seed=args.seed,
        )

    if args.mode in ("auto", "supervised"):
        ann_path = Path(args.annotations)
        if not ann_path.exists():
            if args.mode == "supervised":
                print(f"Error: annotations CSV not found: {ann_path}", file=sys.stderr)
                return 2
            summary["supervised_models"] = {
                "status": "skipped_annotations_file_missing",
                "annotation_csv": str(ann_path),
            }
        else:
            summary["supervised_models"] = train_supervised_real_pilot(
                ann_path,
                out_dir,
                min_comments=args.min_comments,
                min_threads=args.min_threads,
                min_per_class=args.min_per_class,
                include_muril=args.include_muril,
                muril_epochs=args.muril_epochs,
                seed=args.seed,
            )

    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.mode == "supervised":
        sup = summary["supervised_models"] or {}
        if sup.get("status") != "trained":
            return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
