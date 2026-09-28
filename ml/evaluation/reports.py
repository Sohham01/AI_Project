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


def _extract_metrics(
    raw_report: Any,
    labels: List[str],
    is_multilabel: bool,
) -> ClassificationMetricSet:
    """Try to extract a metric set from a task-specific evaluation report.

    The framework knows the major evaluation report types from the existing NLP
    modules.  When the type is unknown it falls back to an empty metric set
    with a warning.
    """
    metrics = ClassificationMetricSet(labels=labels, multilabel=is_multilabel)

    if raw_report is None:
        metrics.reason_empty = "raw_report is None"
        return metrics

    # ---- Known report types from the existing NLP modules ----

    # Single-label: sentiment, response.
    # These have a ``metrics`` dict or similar attributes.
    if hasattr(raw_report, "accuracy"):
        metrics.accuracy = float(raw_report.accuracy) if raw_report.accuracy is not None else None
    if hasattr(raw_report, "macro_f1"):
        metrics.macro_f1 = float(raw_report.macro_f1) if raw_report.macro_f1 is not None else None
    if hasattr(raw_report, "micro_f1"):
        metrics.micro_f1 = float(raw_report.micro_f1) if raw_report.micro_f1 is not None else None
    if hasattr(raw_report, "weighted_f1"):
        metrics.weighted_f1 = float(raw_report.weighted_f1) if raw_report.weighted_f1 is not None else None
    if hasattr(raw_report, "precision"):
        metrics.precision = float(raw_report.precision) if raw_report.precision is not None else None
    if hasattr(raw_report, "recall"):
        metrics.recall = float(raw_report.recall) if raw_report.recall is not None else None
    if hasattr(raw_report, "n_samples") or hasattr(raw_report, "n_documents"):
        n = getattr(raw_report, "n_samples", getattr(raw_report, "n_documents", None))
        if n is not None:
            metrics.n_samples = int(n)

    # Multi-label: emotion, issue.
    if is_multilabel:
        if hasattr(raw_report, "per_label") and isinstance(getattr(raw_report, "per_label"), dict):
            for label, vals in raw_report.per_label.items():
                if label not in metrics.per_label:
                    metrics.per_label[label] = {}
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None:
                            metrics.per_label[label][k] = float(v)
        if hasattr(raw_report, "macro_f1"):
            metrics.macro_f1 = float(raw_report.macro_f1) if raw_report.macro_f1 is not None else None
        if hasattr(raw_report, "micro_f1"):
            metrics.micro_f1 = float(raw_report.micro_f1) if raw_report.micro_f1 is not None else None

    # Binary: solution detection.
    if not is_multilabel and len(labels) == 2:
        if hasattr(raw_report, "f1_score"):
            metrics.macro_f1 = float(raw_report.f1_score) if raw_report.f1_score is not None else None
        if hasattr(raw_report, "precision"):
            metrics.precision = float(raw_report.precision) if raw_report.precision is not None else None
        if hasattr(raw_report, "recall"):
            metrics.recall = float(raw_report.recall) if raw_report.recall is not None else None

    # Collect any reason fields.
    for attr in ("reason_empty", "reason_insufficient_classes",
                 "reason_invalid_shape", "reason_label_mismatch"):
        if hasattr(raw_report, attr):
            val = getattr(raw_report, attr)
            if val:
                setattr(metrics, attr, str(val))

    return metrics


def _collect_warnings(report: Any, out: List[str]) -> None:
    """Pull warning strings from a task-specific report into the unified list."""
    if hasattr(report, "warnings") and isinstance(report.warnings, list):
        for w in report.warnings:
            if w and w not in out:
                out.append(str(w))
    if hasattr(report, "warning") and isinstance(report.warning, str):
        if report.warning and report.warning not in out:
            out.append(report.warning)


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
