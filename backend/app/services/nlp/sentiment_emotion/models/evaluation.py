"""
Evaluation for sentiment (single-label) and emotion (multi-label).

The two dimensions are scored differently because they are different tasks,
and using the wrong metric family for either would mislead:

**Sentiment is single-label**, so ordinary multiclass metrics apply and are
reported: accuracy, macro/micro F1, precision, recall, per-class metrics and a
confusion matrix.  Accuracy is legitimate here (each document has exactly one
gold class), but it is still the metric most inflated by class imbalance, so
macro-F1 is reported alongside it and per-class support is always shown.

**Emotion is multi-label**, so the metric rules are the multi-label ones:
micro-F1 pools every (document, label) decision and is the headline number on
imbalanced data; macro-F1 exposes per-label failure; subset accuracy (exact
set match) is the strictest number and is always <= micro-F1 on genuinely
multi-label data; and "hamming accuracy" is reported *labelled as inflated*,
because it counts an all-zero true and predicted row as fully correct.

Nothing is fabricated.  If the evaluation set is too small, or lacks label
diversity, ``is_meaningful`` is False, ``reason`` says exactly why, and every
metric field stays ``None``.  Per-class / per-label support is still reported,
because that is a fact about the data rather than a claim about the model.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

#: Below this many documents an evaluation set is too small for the numbers to
#: mean anything.
MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS = 10
#: Below this many classes / labels the averages are not interpretable.
MIN_CLASSES_FOR_MEANINGFUL_METRICS = 2
#: Below this many examples for a class its F1 is not interpretable.
MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1 = 2


# ---------------------------------------------------------------------------
# Sentiment (single-label)
# ---------------------------------------------------------------------------


class SentimentEvaluationReport(BaseModel):
    """
    Outcome of evaluating predicted sentiment classes against gold annotations.

    Metric fields are ``Optional[float]``: they are ``None`` when the data is
    too small to support them, instead of being filled with a number that looks
    authoritative but is not.
    """

    is_meaningful: bool = False
    reason: Optional[str] = None

    n_documents: int = 0
    n_classes: int = 0
    #: Class order used by ``confusion_matrix`` rows/columns.
    class_order: List[str] = Field(default_factory=list)
    #: Documents where the model abstained (winning class below the floor).
    #: Counted as misses, so ``accuracy`` accounts for them.
    n_abstained: int = 0
    #: Documents whose gold class the model cannot emit at all (unseen class).
    #: Counted as misses; excluded from the per-class metrics.
    n_unscoreable: int = 0

    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    micro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    micro_recall: Optional[float] = None

    per_class_f1: Dict[str, float] = Field(default_factory=dict)
    per_class_precision: Dict[str, float] = Field(default_factory=dict)
    per_class_recall: Dict[str, float] = Field(default_factory=dict)
    per_class_support: Dict[str, int] = Field(default_factory=dict)
    per_class_predicted: Dict[str, int] = Field(default_factory=dict)

    #: Rows and columns follow ``class_order``; withheld when not meaningful.
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)

    def confusion_matrix_as_rows(self) -> List[str]:
        """The confusion matrix rendered as labelled rows for terminal display."""
        if self.confusion_matrix is None:
            return []
        width = max([len(name) for name in self.class_order] + [4])
        lines: List[str] = []
        for name, row in zip(self.class_order, self.confusion_matrix):
            cells = "  ".join(f"{value:>4}" for value in row)
            lines.append(f"    {name:<{width}}  {cells}")
        return lines

    def format_summary(self) -> str:
        """Human-readable summary that never overstates the numbers."""
        lines = [
            "Sentiment evaluation",
            f"  documents        : {self.n_documents}",
            f"  classes          : {self.n_classes}",
        ]
        if self.n_abstained:
            lines.append(f"  abstained        : {self.n_abstained}")
        if self.n_unscoreable:
            lines.append(
                f"  unscoreable gold : {self.n_unscoreable} (class not in the model)"
            )
        if not self.is_meaningful:
            lines.append(f"  STATUS           : NOT MEANINGFUL - {self.reason}")
            lines.append(
                "  metrics          : withheld (would be misleading at this data size)"
            )
            return "\n".join(lines)

        lines.extend(
            [
                f"  accuracy         : {self.accuracy:.4f}  (single-label, so valid here)",
                f"  macro-F1         : {self.macro_f1:.4f}",
                f"  micro-F1         : {self.micro_f1:.4f}",
                f"  macro-precision  : {self.macro_precision:.4f}",
                f"  micro-precision  : {self.micro_precision:.4f}",
                f"  macro-recall     : {self.macro_recall:.4f}",
                f"  micro-recall     : {self.micro_recall:.4f}",
                "",
                f"  confusion matrix (rows = actual, columns = predicted, order: "
                f"{', '.join(self.class_order)}):",
            ]
        )
        lines.extend(self.confusion_matrix_as_rows())
        lines.extend(["", "  per-class F1:"])
        for name in sorted(self.per_class_f1):
            lines.append(
                f"    {name:<24} F1={self.per_class_f1[name]:.4f}  "
                f"support={self.per_class_support.get(name, 0)}"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def evaluate_sentiment(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str],
) -> SentimentEvaluationReport:
    """
    Compute single-label sentiment metrics.

    ``y_true`` / ``y_pred`` are sequences of class names.  Two special cases are
    handled honestly rather than silently:

    - a prediction of ``None`` means the model **abstained**;
    - a gold value of ``None`` means the document's gold class is not in
      ``class_names``, i.e. the model could not possibly have produced it.

    Both are counted as misses, and abstentions are excluded from the
    confusion matrix (its row sums fall below the reported support by exactly
    that amount, which the warnings say out loud).  Dropping such rows would
    quietly improve the numbers.
    """
    truth: List[Optional[str]] = [
        None if value is None else str(value) for value in list(y_true)
    ]
    predicted: List[Optional[str]] = [
        None if value is None else str(value) for value in list(y_pred)
    ]

    if len(truth) != len(predicted):
        raise ValueError(
            f"y_true has {len(truth)} entries but y_pred has {len(predicted)}"
        )

    names = list(class_names)
    n_documents = len(truth)

    # Keep the model's class order, then append any class that only appears in
    # the evaluation data, so nothing is silently dropped from the matrix.
    observed = {value for value in truth if value is not None} | {
        value for value in predicted if value is not None
    }
    order = [name for name in names if name in observed] + [
        name for name in sorted(observed) if name not in names
    ]

    support = {name: sum(1 for value in truth if value == name) for name in order}
    predicted_counts = {
        name: sum(1 for value in predicted if value == name) for name in order
    }
    n_abstained = sum(1 for value in predicted if value is None)
    n_unscoreable = sum(1 for value in truth if value is None)

    warnings: List[str] = []

    if n_documents < MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS:
        return SentimentEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_documents} evaluation document(s); at least "
                f"{MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS} are required before "
                "sentiment metrics are reported"
            ),
            n_documents=n_documents,
            n_classes=len(order),
            class_order=order,
            n_abstained=n_abstained,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    if len(order) < MIN_CLASSES_FOR_MEANINGFUL_METRICS:
        return SentimentEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {len(order)} sentiment class(es) present in the evaluation "
                f"set; at least {MIN_CLASSES_FOR_MEANINGFUL_METRICS} are required "
                "for macro-averages and a confusion matrix to mean anything"
            ),
            n_documents=n_documents,
            n_classes=len(order),
            class_order=order,
            n_abstained=n_abstained,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    unpredicted = [name for name in order if predicted_counts[name] == 0]
    if unpredicted:
        warnings.append(
            f"class(es) {', '.join(unpredicted)} were never predicted on the "
            "evaluation set; macro-averages are penalised accordingly"
        )
    thin = [
        name
        for name in order
        if 0 < support[name] < MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1
    ]
    if thin:
        warnings.append(
            f"class(es) {', '.join(thin)} have fewer than "
            f"{MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1} examples; their per-class F1 "
            "is unstable and should not be quoted"
        )
    if n_abstained:
        warnings.append(
            f"{n_abstained} of {n_documents} document(s) were abstained on "
            "(winning class below the confidence floor) and counted as misses; "
            "confusion-matrix row sums are below their support by that amount"
        )
    if n_unscoreable:
        warnings.append(
            f"{n_unscoreable} of {n_documents} document(s) have a gold class the "
            "model cannot emit; they are counted as misses and excluded from the "
            "per-class metrics, which are therefore computed on the remainder"
        )

    index = {name: position for position, name in enumerate(order)}
    n_classes = len(order)

    # Indicator form, so one code path serves the aggregates and the per-class
    # metrics.  Abstentions and unscoreable rows stay all-zero, i.e. they match
    # no gold class, which is what makes them misses.
    gold = np.zeros((n_documents, n_classes), dtype=int)
    guesses = np.zeros((n_documents, n_classes), dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    correct = 0
    for row, (actual, guess) in enumerate(zip(truth, predicted)):
        if actual is not None:
            gold[row, index[actual]] = 1
        if guess is not None:
            guesses[row, index[guess]] = 1
        if actual is not None and guess is not None and actual == guess:
            correct += 1
            matrix[index[actual], index[guess]] += 1

    return SentimentEvaluationReport(
        is_meaningful=True,
        reason=None,
        n_documents=n_documents,
        n_classes=n_classes,
        class_order=order,
        n_abstained=n_abstained,
        n_unscoreable=n_unscoreable,
        accuracy=round(correct / n_documents, 6) if n_documents else 0.0,
        macro_f1=round(
            float(f1_score(gold, guesses, average="macro", zero_division=0)), 6
        ),
        micro_f1=round(
            float(f1_score(gold, guesses, average="micro", zero_division=0)), 6
        ),
        macro_precision=round(
            float(precision_score(gold, guesses, average="macro", zero_division=0)), 6
        ),
        micro_precision=round(
            float(precision_score(gold, guesses, average="micro", zero_division=0)), 6
        ),
        macro_recall=round(
            float(recall_score(gold, guesses, average="macro", zero_division=0)), 6
        ),
        micro_recall=round(
            float(recall_score(gold, guesses, average="micro", zero_division=0)), 6
        ),
        per_class_f1={
            name: round(
                float(f1_score(gold[:, position], guesses[:, position], zero_division=0)),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_precision={
            name: round(
                float(
                    precision_score(
                        gold[:, position], guesses[:, position], zero_division=0
                    )
                ),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_recall={
            name: round(
                float(
                    recall_score(
                        gold[:, position], guesses[:, position], zero_division=0
                    )
                ),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_support=support,
        per_class_predicted=predicted_counts,
        confusion_matrix=matrix.tolist(),
        classification_report=classification_report(
            np.asarray([index[value] if value is not None else -1 for value in truth]),
            np.asarray([index[value] if value is not None else -1 for value in predicted]),
            labels=list(range(n_classes)),
            target_names=order,
            zero_division=0,
            output_dict=True,
        ),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Emotion (multi-label)
# ---------------------------------------------------------------------------


class EmotionEvaluationReport(BaseModel):
    """
    Outcome of evaluating predicted emotion labels against gold annotations.

    Same honesty rules as the sentiment report: metric fields are ``None`` when
    the data cannot support them.
    """

    is_meaningful: bool = False
    reason: Optional[str] = None

    n_documents: int = 0
    n_labels: int = 0

    micro_f1: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_precision: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    subset_accuracy: Optional[float] = None
    hamming_accuracy: Optional[float] = None
    hamming_loss: Optional[float] = None

    per_label_f1: Dict[str, float] = Field(default_factory=dict)
    per_label_precision: Dict[str, float] = Field(default_factory=dict)
    per_label_recall: Dict[str, float] = Field(default_factory=dict)
    per_label_support: Dict[str, int] = Field(default_factory=dict)
    per_label_predicted: Dict[str, int] = Field(default_factory=dict)

    classification_report: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)

    def format_summary(self) -> str:
        """Human-readable summary that never overstates the numbers."""
        lines = [
            "Emotion evaluation",
            f"  documents        : {self.n_documents}",
            f"  labels           : {self.n_labels}",
        ]
        if not self.is_meaningful:
            lines.append(f"  STATUS           : NOT MEANINGFUL - {self.reason}")
            lines.append(
                "  metrics          : withheld (would be misleading at this data size)"
            )
            return "\n".join(lines)

        lines.extend(
            [
                f"  micro-F1         : {self.micro_f1:.4f}",
                f"  macro-F1         : {self.macro_f1:.4f}",
                f"  micro-precision  : {self.micro_precision:.4f}",
                f"  macro-precision  : {self.macro_precision:.4f}",
                f"  macro-recall     : {self.macro_recall:.4f}",
                f"  subset accuracy  : {self.subset_accuracy:.4f}  (exact label-set match)",
                f"  hamming accuracy : {self.hamming_accuracy:.4f}  (per-label, inflated on sparse data)",
                "",
                "  per-label F1:",
            ]
        )
        for name in sorted(self.per_label_f1):
            lines.append(
                f"    {name:<24} F1={self.per_label_f1[name]:.4f}  "
                f"support={self.per_label_support.get(name, 0)}"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def _as_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.astype(int)


def evaluate_emotions(
    y_true: Any,
    y_pred: Any,
    label_names: Sequence[str],
) -> EmotionEvaluationReport:
    """
    Compute multi-label metrics for emotion label matrices.

    ``y_true`` / ``y_pred`` are ``(n_documents, n_labels)`` binary matrices
    whose columns line up with ``label_names``.
    """
    truth = _as_matrix(y_true)
    pred = _as_matrix(y_pred)

    if truth.shape != pred.shape:
        raise ValueError(
            f"y_true shape {truth.shape} does not match y_pred shape {pred.shape}"
        )

    names = list(label_names)
    if truth.shape[1] != len(names):
        raise ValueError(
            f"label_names has {len(names)} entries but the matrices have "
            f"{truth.shape[1]} columns"
        )

    n_documents, n_labels = truth.shape
    support = {name: int(truth[:, i].sum()) for i, name in enumerate(names)}
    predicted = {name: int(pred[:, i].sum()) for i, name in enumerate(names)}

    warnings: List[str] = []

    if n_documents < MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS:
        return EmotionEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_documents} evaluation document(s); at least "
                f"{MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS} are required before "
                "emotion metrics are reported"
            ),
            n_documents=n_documents,
            n_labels=n_labels,
            per_label_support=support,
            per_label_predicted=predicted,
            warnings=warnings,
        )

    if n_labels < MIN_CLASSES_FOR_MEANINGFUL_METRICS:
        return EmotionEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_labels} emotion label(s) evaluated; emotion "
                f"recognition is multi-label, so at least "
                f"{MIN_CLASSES_FOR_MEANINGFUL_METRICS} are required for "
                "macro-averages to mean anything"
            ),
            n_documents=n_documents,
            n_labels=n_labels,
            per_label_support=support,
            per_label_predicted=predicted,
            warnings=warnings,
        )

    zero_support = [name for name in names if support[name] == 0]
    if zero_support:
        warnings.append(
            f"label(s) {', '.join(sorted(zero_support))} have zero support in the "
            "evaluation set; their F1 is reported as 0.0 by convention and does "
            "not reflect model quality"
        )
    unpredicted = [name for name in names if predicted[name] == 0]
    if unpredicted:
        warnings.append(
            f"label(s) {', '.join(sorted(unpredicted))} were never predicted on "
            "the evaluation set; macro-averages are penalised accordingly"
        )
    thin = [
        name
        for name in names
        if 0 < support[name] < MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1
    ]
    if thin:
        warnings.append(
            f"label(s) {', '.join(thin)} have fewer than "
            f"{MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1} positive examples; their "
            "per-label F1 is unstable and should not be quoted"
        )
    no_emotion_gold = int((truth.sum(axis=1) == 0).sum())
    no_emotion_pred = int((pred.sum(axis=1) == 0).sum())
    if no_emotion_gold == 0:
        warnings.append(
            "no document in the evaluation set is annotated with zero emotions; "
            "the model's ability to correctly emit an empty emotion set is "
            "therefore untested, and hamming accuracy is optimistic"
        )
    if no_emotion_pred:
        warnings.append(
            f"{no_emotion_pred} of {n_documents} document(s) received no emotion "
            "above the threshold; check whether the threshold is too high for "
            "this corpus"
        )

    return EmotionEvaluationReport(
        is_meaningful=True,
        reason=None,
        n_documents=n_documents,
        n_labels=n_labels,
        micro_f1=round(
            float(f1_score(truth, pred, average="micro", zero_division=0)), 6
        ),
        macro_f1=round(
            float(f1_score(truth, pred, average="macro", zero_division=0)), 6
        ),
        micro_precision=round(
            float(precision_score(truth, pred, average="micro", zero_division=0)), 6
        ),
        macro_precision=round(
            float(precision_score(truth, pred, average="macro", zero_division=0)), 6
        ),
        macro_recall=round(
            float(recall_score(truth, pred, average="macro", zero_division=0)), 6
        ),
        subset_accuracy=round(
            float(np.all(truth == pred, axis=1).mean()) if n_documents else 0.0, 6
        ),
        hamming_accuracy=round(float(1.0 - hamming_loss(truth, pred)), 6),
        hamming_loss=round(float(hamming_loss(truth, pred)), 6),
        per_label_f1={
            name: round(
                float(f1_score(truth[:, i], pred[:, i], zero_division=0)), 6
            )
            for i, name in enumerate(names)
        },
        per_label_precision={
            name: round(
                float(precision_score(truth[:, i], pred[:, i], zero_division=0)), 6
            )
            for i, name in enumerate(names)
        },
        per_label_recall={
            name: round(
                float(recall_score(truth[:, i], pred[:, i], zero_division=0)), 6
            )
            for i, name in enumerate(names)
        },
        per_label_support=support,
        per_label_predicted=predicted,
        classification_report=classification_report(
            truth, pred, target_names=names, zero_division=0, output_dict=True
        ),
        warnings=warnings,
    )


__all__ = [
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_CLASSES_FOR_MEANINGFUL_METRICS",
    "MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1",
    "SentimentEvaluationReport",
    "evaluate_sentiment",
    "EmotionEvaluationReport",
    "evaluate_emotions",
]
