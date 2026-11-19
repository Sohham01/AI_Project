"""
Unified evaluation metrics for the ML framework.

Provides a single metric suite that handles both single-label and multi-label
classification, with clear guards: metrics are None when the data is too thin
to support them meaningfully, and the suite never fabricates numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    multilabel: bool = False,
    threshold: float = 0.5,
) -> "ClassificationMetricSet":
    """Compute a standard metric suite for classifier output.

    Args:
        y_true: ground truth.  For single-label this is a 1-D array of class
            names (or None for missing).  For multi-label this is a 2-D binary
            matrix (n_samples, n_labels).
        y_pred: predictions in the same shape as ``y_true``.
        labels: ordered list of label names.
        multilabel: if True the arrays are treated as multi-label.
        threshold: per-label decision threshold for multi-label.

    Returns:
        ClassificationMetricSet with all computable metrics.  Metrics that
            cannot be computed are None with a reason.
    """
    n_labels = len(labels)
    result = ClassificationMetricSet(labels=labels, multilabel=multilabel)

    if y_true.size == 0 or y_pred.size == 0:
        result.reason_empty = "empty input"
        return result

    if multilabel:
        return _compute_multilabel_metrics(result, y_true, y_pred, labels, threshold)

    return _compute_singlelabel_metrics(result, y_true, y_pred, labels)


def _compute_singlelabel_metrics(
    result: ClassificationMetricSet,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> ClassificationMetricSet:
    """Metrics for mutually exclusive single-label classification."""
    # Flatten to 1-D string arrays, handling None/missing.
    raw_true = _flatten_single(y_true, labels)
    raw_pred = _flatten_single(y_pred, labels)

    if len(raw_true) != len(raw_pred):
        result.reason_invalid_shape = (
            f"y_true length ({len(raw_true)}) != y_pred length ({len(raw_pred)})"
        )
        return result

    # Keep only rows with a non-null gold label; abstained (None) predictions on
    # valid gold rows are retained as misses via a sentinel outside `labels`.
    valid_pairs = [
        (t, "__abstained__" if p is None else p)
        for t, p in zip(raw_true, raw_pred)
        if t is not None and str(t).strip() != ""
    ]

    if len(valid_pairs) == 0:
        result.reason_empty = "no valid labels after flattening"
        return result

    true_series = [t for t, _ in valid_pairs]
    pred_series = [p for _, p in valid_pairs]

    present_true = sorted(set(true_series))
    present_labels = sorted(set(true_series) | {p for p in pred_series if p != "__abstained__"})
    if len(present_true) < 2 and len(present_labels) < 2:
        result.reason_insufficient_classes = (
            f"only {len(present_labels)} class(es) present in evaluation data; "
            f"metrics requiring class diversity are not meaningful"
        )
        # Accuracy is still computable with 1 class.
        result.accuracy = float(np.mean(np.array(true_series) == np.array(pred_series)))
        result.n_samples = len(true_series)
        return result

    result.n_samples = len(true_series)

    # Accuracy.
    result.accuracy = float(accuracy_score(true_series, pred_series, normalize=True))

    # Active labels for macro/micro/weighted averaging: never let an empty class
    # (0 gold, 0 predicted) silently inflate macro averages.
    active_labels = [l for l in labels if l in present_labels] or present_labels

    # F1 variants.
    result.macro_f1 = float(
        f1_score(true_series, pred_series, labels=active_labels, average="macro", zero_division=0)
    )
    result.micro_f1 = float(
        f1_score(true_series, pred_series, labels=active_labels, average="micro", zero_division=0)
    )
    result.weighted_f1 = float(
        f1_score(
            true_series, pred_series, labels=active_labels, average="weighted", zero_division=0
        )
    )

    # Precision/recall (macro).
    result.precision = float(
        precision_score(
            true_series, pred_series, labels=active_labels, average="macro", zero_division=0
        )
    )
    result.recall = float(
        recall_score(true_series, pred_series, labels=active_labels, average="macro", zero_division=0)
    )

    # Per-class.
    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    for label in labels:
        yt = [1 if t == label else 0 for t in true_series]
        yp = [1 if p == label else 0 for p in pred_series]
        support = sum(1 for t in yt if t == 1)
        predicted_pos = sum(1 for p in yp if p == 1)
        if support == 0 and predicted_pos == 0:
            per_class[label] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
            }
            continue
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        precision = round(tp / (tp + fp), 6) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        recall_l = round(tp / (tp + fn), 6) if (tp + fn) > 0 else (0.0 if predicted_pos == 0 and support > 0 else None)
        f1_l = (
            round(2 * precision * recall_l / (precision + recall_l), 6)
            if (precision is not None and recall_l is not None and (precision + recall_l) > 0)
            else (0.0 if support > 0 or predicted_pos > 0 else None)
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall_l,
            "f1": f1_l,
            "support": support,
        }
    result.per_class = per_class

    # Confusion matrix across the ordered label vocabulary present in gold or predictions.
    cm_labels = [l for l in labels if l in present_labels] or list(labels)
    cm = confusion_matrix(
        true_series, pred_series, labels=cm_labels
    )
    result.confusion_matrix = cm.tolist()
    result.confusion_matrix_labels = cm_labels

    return result


def _compute_multilabel_metrics(
    result: ClassificationMetricSet,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    threshold: float,
) -> ClassificationMetricSet:
    """Metrics for multi-label classification."""
    if y_true.ndim != 2 or y_pred.ndim != 2:
        result.reason_invalid_shape = (
            f"expected 2-D arrays for multi-label, got y_true.shape={y_true.shape}, "
            f"y_pred.shape={y_pred.shape}"
        )
        return result

    if y_true.shape[0] != y_pred.shape[0]:
        result.reason_invalid_shape = (
            f"sample count mismatch: y_true has {y_true.shape[0]} rows, "
            f"y_pred has {y_pred.shape[0]} rows"
        )
        return result

    if y_true.shape[1] == 0 or len(labels) == 0:
        result.reason_empty = "no labels provided for multi-label evaluation"
        return result

    if y_true.shape[1] != len(labels):
        result.reason_label_mismatch = (
            f"label count mismatch: y_true has {y_true.shape[1]} columns, "
            f"labels has {len(labels)} entries"
        )
        # Still try to compute whatever we can.
        actual_n = min(y_true.shape[1], len(labels))
        y_true = y_true[:, :actual_n]
        y_pred = y_pred[:, :actual_n]
        active_labels = labels[:actual_n]
    else:
        active_labels = labels

    result.n_samples = y_true.shape[0]
    result.n_labels = len(active_labels)
    result.threshold = threshold

    # Micro F1.
    result.micro_f1 = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )
    # Macro F1.
    result.macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    # Per-label precision/recall.
    result.precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.recall = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Per-label F1.
    per_label: Dict[str, Dict[str, Optional[float]]] = {}
    for i, label in enumerate(active_labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if yt.sum() == 0 and yp.sum() == 0:
            per_label[label] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": int(yt.sum()),
                "reason": "label never present in gold or prediction",
            }
            continue
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_l = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_l = (
            2 * precision * recall_l / (precision + recall_l)
            if (precision + recall_l) > 0
            else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall_l, 6),
            "f1": round(f1_l, 6),
            "support": int(yt.sum()),
        }
    result.per_label = per_label

    # Subset accuracy (exact match).
    exact = np.all(y_true == y_pred, axis=1)
    result.subset_accuracy = float(exact.mean()) if exact.size > 0 else None

    # Hamming loss.
    result.hamming_loss = float(hamming_loss(y_true, y_pred))

    # Hamming accuracy (1 - hamming_loss).
    result.hamming_accuracy = float(1.0 - hamming_loss(y_true, y_pred))

    return result


def _flatten_single(arr: np.ndarray, labels: List[str]) -> List[Optional[str]]:
    """Flatten a classifier output array to a list of label strings or None."""
    out: List[Optional[str]] = []
    arr = np.asarray(arr)
    if arr.ndim == 1:
        for v in arr:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(None)
            else:
                out.append(str(v))
    elif arr.ndim == 2:
        # Probability matrix: take argmax.
        for row in arr:
            if row.size == 0:
                out.append(None)
            else:
                idx = int(np.argmax(row))
                if idx < len(labels):
                    out.append(labels[idx])
                else:
                    out.append(None)
    else:
        return [None] * max(arr.shape[0], 1)
    return out


# ---------------------------------------------------------------------------
# Metric set
# ---------------------------------------------------------------------------


class ClassificationMetricSet:
    """A collection of metrics for one evaluation run.

    Metrics that cannot be computed are None with a reason.  The set is never
    fabricated — if the data is too thin the reason explains why.
    """

    def __init__(
        self,
        labels: List[str],
        multilabel: bool = False,
    ) -> None:
        self.labels = labels
        self.multilabel = multilabel

        # Always-present.
        self.n_samples: Optional[int] = None
        self.n_labels: int = len(labels)

        # Single-label metrics.
        self.accuracy: Optional[float] = None
        self.macro_f1: Optional[float] = None
        self.micro_f1: Optional[float] = None
        self.weighted_f1: Optional[float] = None
        self.precision: Optional[float] = None
        self.recall: Optional[float] = None

        # Multi-label extras.
        self.threshold: Optional[float] = None
        self.subset_accuracy: Optional[float] = None
        self.hamming_loss: Optional[float] = None
        self.hamming_accuracy: Optional[float] = None

        # Per-class / per-label breakdowns.
        self.per_class: Dict[str, Dict[str, Optional[float]]] = {}
        self.per_label: Dict[str, Dict[str, Optional[float]]] = {}

        # Confusion matrix (single-label only).
        self.confusion_matrix: Optional[List[List[int]]] = None
        self.confusion_matrix_labels: List[str] = []

        # Reasons why metrics may be missing.
        self.reason_empty: Optional[str] = None
        self.reason_insufficient_classes: Optional[str] = None
        self.reason_invalid_shape: Optional[str] = None
        self.reason_label_mismatch: Optional[str] = None

    @property
    def is_meaningful(self) -> bool:
        """Whether this metric set represents a real evaluation."""
        if self.reason_empty:
            return False
        if self.reason_insufficient_classes:
            return False
        if self.n_samples is not None and self.n_samples < 5:
            return False
        return True

    def metrics_serializable(self) -> Dict[str, Any]:
        """Return a JSON-serialisable subset of metrics."""
        out: Dict[str, Any] = {
            "n_samples": self.n_samples,
            "multilabel": self.multilabel,
            "is_meaningful": self.is_meaningful,
        }
        if self.accuracy is not None:
            out["accuracy"] = round(self.accuracy, 6)
        if self.macro_f1 is not None:
            out["macro_f1"] = round(self.macro_f1, 6)
        if self.micro_f1 is not None:
            out["micro_f1"] = round(self.micro_f1, 6)
        if self.weighted_f1 is not None:
            out["weighted_f1"] = round(self.weighted_f1, 6)
        if self.precision is not None:
            out["precision"] = round(self.precision, 6)
        if self.recall is not None:
            out["recall"] = round(self.recall, 6)
        if self.subset_accuracy is not None:
            out["subset_accuracy"] = round(self.subset_accuracy, 6)
        if self.hamming_loss is not None:
            out["hamming_loss"] = round(self.hamming_loss, 6)
        if self.hamming_accuracy is not None:
            out["hamming_accuracy"] = round(self.hamming_accuracy, 6)
        if self.per_class:
            out["per_class"] = {
                k: dict(v)
                for k, v in self.per_class.items()
            }
        if self.per_label:
            out["per_label"] = {
                k: dict(v)
                for k, v in self.per_label.items()
            }
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix
            out["confusion_matrix_labels"] = self.confusion_matrix_labels
        if self.reason_empty:
            out["reason_empty"] = self.reason_empty
        if self.reason_insufficient_classes:
            out["reason_insufficient_classes"] = self.reason_insufficient_classes
        if self.reason_invalid_shape:
            out["reason_invalid_shape"] = self.reason_invalid_shape
        if self.reason_label_mismatch:
            out["reason_label_mismatch"] = self.reason_label_mismatch
        return out

    def __repr__(self) -> str:
        return (
            f"ClassificationMetricSet(labels={self.labels}, "
            f"n_samples={self.n_samples}, "
            f"macro_f1={self.macro_f1}, "
            f"is_meaningful={self.is_meaningful})"
        )
