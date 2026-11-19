"""
Unified evaluation reports for the ML framework.

Translates task-specific classifier evaluation outputs into a single
``EvaluationReport`` shape that the trainer, artifact, and registry all use.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

from ml.evaluation.metrics import ClassificationMetricSet


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """A single evaluation run result, unified across all tasks.

    This is the shape the trainer, artifact, and registry all see — regardless
    of whether the underlying task is single-label sentiment, multi-label
    issue, or binary solution detection.
    """

    task_name: str
    model_name: str
    model_version: str
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0
    labels: List[str] = field(default_factory=list)
    metrics: ClassificationMetricSet = field(default_factory=lambda: ClassificationMetricSet([]))
    is_meaningful: bool = False
    data_split_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # Optional timestamp.
    timestamp: Optional[str] = None

    def metrics_serializable(self) -> Dict[str, Any]:
        if self.metrics is None:
            return {}
        return self.metrics.metrics_serializable()

    def summary_dict(self) -> Dict[str, Any]:
        """A concise summary for display / logging."""
        return {
            "task": self.task_name,
            "model": f"{self.model_name} {self.model_version}",
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "is_meaningful": self.is_meaningful,
            "labels": self.labels,
            "metrics": self.metrics_serializable(),
            "warnings": self.warnings,
        }


def build_evaluation_report(
    task_name: str,
    model_name: str,
    model_version: str,
    n_train: int,
    n_validation: int,
    n_test: int,
    labels: List[str],
    raw_report: Any,
    is_multilabel: bool = False,
    model_is_trained: bool = True,
    leakage_detected: bool = False,
    include_timestamp: bool = False,
    timestamp: Optional[str] = None,
) -> EvaluationReport:
    """Build a unified ``EvaluationReport`` from a task-specific raw report.

    Args:
        raw_report: the output of the model's ``evaluate()`` method.  The
            framework inspects it for known shapes (metric dicts, warning lists,
            etc.) and translates what it can into the unified metric set.
    """
    metrics = _extract_metrics(raw_report, labels, is_multilabel)
    is_meaningful = (
        model_is_trained
        and metrics.is_meaningful
        and not leakage_detected
        and n_test >= 5
    )

    warnings: List[str] = []
    if leakage_detected:
        warnings.append(
            "DATA LEAKAGE DETECTED: evaluation set overlaps training set. "
            "Metrics may be inflated."
        )
    if raw_report is not None:
        _collect_warnings(raw_report, warnings)
    if not model_is_trained:
        warnings.append("model is not trained")
    if not is_meaningful:
        warnings.append(
            "evaluation is not meaningful: see is_meaningful=False and metric reasons"
        )

    return EvaluationReport(
        task_name=task_name,
        model_name=model_name,
        model_version=model_version,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        labels=labels,
        metrics=metrics,
        is_meaningful=is_meaningful,
        data_split_metadata={
            "n_train": n_train,
            "n_validation": n_validation,
            "n_test": n_test,
            "leakage_detected": leakage_detected,
        },
        warnings=warnings,
        timestamp=timestamp,
    )


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dictionary key from *obj* without coercing None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_field(obj: Any, name: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _extract_metrics(
    raw_report: Any,
    labels: List[str],
    is_multilabel: bool,
) -> ClassificationMetricSet:
    """Try to extract a metric set from a task-specific evaluation report.

    The framework knows the major evaluation report types from the existing NLP
    modules.  When the type is unknown it falls back to an empty metric set
    with a warning.  Missing metrics remain ``None`` and are never coerced to 0.
    """
    metrics = ClassificationMetricSet(labels=labels, multilabel=is_multilabel)

    if raw_report is None:
        metrics.reason_empty = "raw_report is None"
        return metrics

    # Propagate explicit non-meaningful reason from task evaluators.
    raw_meaningful = _get_field(raw_report, "is_meaningful")
    raw_reason = _get_field(raw_report, "reason")
    if raw_meaningful is False:
        reason_text = str(raw_reason or "raw evaluation report marked is_meaningful=False")
        if any(tok in reason_text.lower() for tok in ("class", "target", "label", "example of")):
            metrics.reason_insufficient_classes = reason_text
        else:
            metrics.reason_empty = reason_text

    # Sample count.
    n = _get_field(raw_report, "n_samples")
    if n is None:
        n = _get_field(raw_report, "n_documents")
    if n is not None:
        metrics.n_samples = int(n)

    threshold = _get_field(raw_report, "threshold")
    if threshold is not None:
        metrics.threshold = float(threshold)

    if is_multilabel:
        # Multi-label tasks (emotion, issue):
        # Do NOT populate single-label accuracy, weighted_f1, per_class, or confusion_matrix.
        macro_f1 = _get_field(raw_report, "macro_f1")
        micro_f1 = _get_field(raw_report, "micro_f1")
        subset_acc = _get_field(raw_report, "subset_accuracy")
        h_loss = _get_field(raw_report, "hamming_loss")
        h_acc = _get_field(raw_report, "hamming_accuracy")
        prec = _get_field(raw_report, "precision")
        if prec is None:
            prec = _get_field(raw_report, "macro_precision")
        rec = _get_field(raw_report, "recall")
        if rec is None:
            rec = _get_field(raw_report, "macro_recall")

        metrics.macro_f1 = float(macro_f1) if macro_f1 is not None else None
        metrics.micro_f1 = float(micro_f1) if micro_f1 is not None else None
        metrics.subset_accuracy = float(subset_acc) if subset_acc is not None else None
        metrics.hamming_loss = float(h_loss) if h_loss is not None else None
        metrics.hamming_accuracy = float(h_acc) if h_acc is not None else None
        metrics.precision = float(prec) if prec is not None else None
        metrics.recall = float(rec) if rec is not None else None

        raw_per_label = _get_field(raw_report, "per_label")
        if isinstance(raw_per_label, dict) and raw_per_label:
            for label, vals in raw_per_label.items():
                if not isinstance(vals, dict):
                    continue
                metrics.per_label[label] = {
                    k: (int(v) if k == "support" and v is not None else float(v) if isinstance(v, (int, float)) else v)
                    for k, v in vals.items()
                }
        else:
            pl_f1 = _get_field(raw_report, "per_label_f1") or {}
            pl_prec = _get_field(raw_report, "per_label_precision") or {}
            pl_rec = _get_field(raw_report, "per_label_recall") or {}
            pl_sup = _get_field(raw_report, "per_label_support") or {}
            if isinstance(pl_f1, dict) or isinstance(pl_sup, dict):
                ordered_labels = list(labels) + [
                    k for k in (list(pl_sup.keys()) + list(pl_f1.keys()))
                    if k not in labels
                ]
                seen = set()
                for lbl in ordered_labels:
                    if lbl in seen:
                        continue
                    seen.add(lbl)
                    if lbl in pl_f1 or lbl in pl_sup or lbl in pl_prec or lbl in pl_rec:
                        sup_val = pl_sup.get(lbl)
                        metrics.per_label[lbl] = {
                            "precision": float(pl_prec[lbl]) if pl_prec.get(lbl) is not None else None,
                            "recall": float(pl_rec[lbl]) if pl_rec.get(lbl) is not None else None,
                            "f1": float(pl_f1[lbl]) if pl_f1.get(lbl) is not None else None,
                            "support": int(sup_val) if sup_val is not None else 0,
                        }
    else:
        # Single-label & binary tasks (sentiment, response, solution):
        # Do NOT populate multi-label subset_accuracy, hamming_loss, hamming_accuracy, or per_label.
        acc = _get_field(raw_report, "accuracy")
        macro_f1 = _get_field(raw_report, "macro_f1")
        if macro_f1 is None:
            macro_f1 = _get_field(raw_report, "f1_score")
        if macro_f1 is None and len(labels) == 2:
            macro_f1 = _get_field(raw_report, "f1")
        micro_f1 = _get_field(raw_report, "micro_f1")
        weighted_f1 = _get_field(raw_report, "weighted_f1")
        prec = _get_field(raw_report, "precision")
        if prec is None:
            prec = _get_field(raw_report, "macro_precision")
        rec = _get_field(raw_report, "recall")
        if rec is None:
            rec = _get_field(raw_report, "macro_recall")

        metrics.accuracy = float(acc) if acc is not None else None
        metrics.macro_f1 = float(macro_f1) if macro_f1 is not None else None
        metrics.micro_f1 = float(micro_f1) if micro_f1 is not None else None
        metrics.weighted_f1 = float(weighted_f1) if weighted_f1 is not None else None
        metrics.precision = float(prec) if prec is not None else None
        metrics.recall = float(rec) if rec is not None else None

        raw_per_class = _get_field(raw_report, "per_class")
        if isinstance(raw_per_class, dict) and raw_per_class:
            for cls_name, vals in raw_per_class.items():
                if not isinstance(vals, dict):
                    continue
                metrics.per_class[cls_name] = {
                    k: (int(v) if k == "support" and v is not None else float(v) if isinstance(v, (int, float)) else v)
                    for k, v in vals.items()
                }
        else:
            pc_f1 = _get_field(raw_report, "per_class_f1") or {}
            pc_prec = _get_field(raw_report, "per_class_precision") or {}
            pc_rec = _get_field(raw_report, "per_class_recall") or {}
            pc_sup = _get_field(raw_report, "per_class_support") or {}
            if isinstance(pc_f1, dict) or isinstance(pc_sup, dict):
                class_order = _get_field(raw_report, "class_order") or list(labels)
                ordered_classes = list(class_order) + [
                    k for k in (list(pc_sup.keys()) + list(pc_f1.keys()))
                    if k not in class_order
                ]
                seen = set()
                for cls_name in ordered_classes:
                    if cls_name in seen:
                        continue
                    seen.add(cls_name)
                    if cls_name in pc_f1 or cls_name in pc_sup or cls_name in pc_prec or cls_name in pc_rec:
                        sup_val = pc_sup.get(cls_name)
                        metrics.per_class[cls_name] = {
                            "precision": float(pc_prec[cls_name]) if pc_prec.get(cls_name) is not None else None,
                            "recall": float(pc_rec[cls_name]) if pc_rec.get(cls_name) is not None else None,
                            "f1": float(pc_f1[cls_name]) if pc_f1.get(cls_name) is not None else None,
                            "support": int(sup_val) if sup_val is not None else 0,
                        }

        cm = _get_field(raw_report, "confusion_matrix")
        if cm is not None:
            metrics.confusion_matrix = [list(map(int, row)) for row in cm]
            cm_labels = (
                _get_field(raw_report, "confusion_matrix_labels")
                or _get_field(raw_report, "class_order")
                or list(labels)
            )
            metrics.confusion_matrix_labels = [str(l) for l in cm_labels]

    # Collect any explicit reason fields.
    for attr in ("reason_empty", "reason_insufficient_classes",
                 "reason_invalid_shape", "reason_label_mismatch"):
        if _has_field(raw_report, attr):
            val = _get_field(raw_report, attr)
            if val:
                setattr(metrics, attr, str(val))

    return metrics


def _collect_warnings(report: Any, out: List[str]) -> None:
    """Pull warning strings from a task-specific report into the unified list."""
    warnings_list = _get_field(report, "warnings")
    if isinstance(warnings_list, list):
        for w in warnings_list:
            if w and w not in out:
                out.append(str(w))
    single_warning = _get_field(report, "warning")
    if isinstance(single_warning, str):
        if single_warning and single_warning not in out:
            out.append(single_warning)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


class EvaluationReportWriter:
    """Write evaluation reports to disk for later inspection."""

    def __init__(self, output_dir: str = "ml/artifacts") -> None:
        self.output_dir = output_dir

    def write_report(self, report: EvaluationReport, suffix: str = "") -> str:
        """Write a report as JSON to ``ml/artifacts/``."""
        os.makedirs(self.output_dir, exist_ok=True)
        safe_task = report.task_name.replace("/", "_").replace("\\", "_")
        safe_model = report.model_name.replace("/", "_").replace("\\", "_")
        base = f"eval_{safe_task}_{safe_model}{suffix}".strip("_")
        path = os.path.join(self.output_dir, base + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.summary_dict(), f, indent=2, default=str)
        return path


# ---------------------------------------------------------------------------
# Class distribution reporting
# ---------------------------------------------------------------------------


def report_class_distribution(
    label_counts: Dict[str, int],
    total: int,
) -> Dict[str, Any]:
    """Return a class distribution summary for logging / reporting."""
    if total == 0:
        return {"total": 0, "distribution": {}, "warning": "empty dataset"}
    dist: Dict[str, Dict[str, Any]] = {}
    for label, count in sorted(label_counts.items()):
        dist[label] = {
            "count": count,
            "percentage": round(count / total, 4),
        }
    # Imbalance warning: if the majority class covers more than 80%.
    max_count = max(label_counts.values()) if label_counts else 0
    majority_label = max(label_counts, key=label_counts.get) if label_counts else ""
    imbalance = max_count / total if total > 0 else 0
    warning = None
    if imbalance > 0.8:
        warning = (
            f"class imbalance: {majority_label!r} covers {imbalance:.0%} of the "
            f"dataset. Accuracy is inflated by this skew; prefer macro-F1 and "
            f"per-class recall when judging the model."
        )
    return {
        "total": total,
        "distribution": dist,
        "imbalance_ratio": round(imbalance, 4),
        "majority_class": majority_label,
        "majority_share": round(imbalance, 4),
        "warning": warning,
    }
