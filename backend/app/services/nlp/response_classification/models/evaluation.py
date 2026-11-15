"""
Evaluation for response classification.

Response classification is **single-label multiclass**, so ordinary multiclass
metrics apply and are all reported: accuracy, macro/micro/weighted F1, macro and
weighted precision and recall, per-class metrics, a confusion matrix and a
scikit-learn classification report.

What each headline number means here, and which one to quote:

- **accuracy** is legitimate for a single-label task (each comment has exactly
  one gold role) and is the most familiar number - but it is the number most
  inflated by class imbalance, because a model that always answers
  ``"agreement"`` scores the share of agreements.  Quote it only next to
  per-class support.
- **macro-F1** weights every response type equally.  On a six-class task where
  ``agreement`` and ``opinion`` dominate real annotation corpora and ``advice``
  / ``solution`` are comparatively rare, macro-F1 is the number that refuses to
  let the model look good by ignoring four classes.
- **weighted-F1** averages per-class F1 weighted by class support, i.e. it sits
  between micro and macro.  It is reported because it is the fairest single
  number when class frequency is itself meaningful.
- **micro-F1 equals accuracy** for single-label multiclass problems: every
  comment contributes exactly one gold and one predicted label, so pooling them
  counts the same agreements accuracy does.  That identity is asserted in the
  tests rather than left to the reader, because a divergence would mean the
  aggregates and the per-class metrics are computed over different data.

Confusion matrix
----------------
The matrix is over ``class_order`` and is the most useful artefact here, because
the confusions are the interesting part of this task: ``advice`` and ``solution``
overlap heavily, and ``opinion`` and ``counterargument`` are separated mostly by
intent.  A classifier that cannot separate those two pairs is telling you
something the headline F1 hides.

Honesty rules
-------------
Nothing is fabricated.  If the evaluation set is too small, or lacks class
diversity, ``is_meaningful`` is False, ``reason`` says exactly why, and every
metric field stays ``None``.  Per-class support is still reported, because that
is a fact about the data rather than a claim about the model.

Two special cases are counted rather than dropped:

- a prediction of ``None`` means the model **abstained**;
- a gold value of ``None`` means the comment's gold role is one the model cannot
  emit at all (an out-of-taxonomy label, or a type it was never trained on).

Both are misses, and they are excluded from the confusion matrix - whose row sums
therefore fall below the reported support by exactly that amount, which the
warnings state explicitly.  Dropping the rows would quietly improve the numbers.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

#: Below this many documents an evaluation set is too small for the numbers to
#: mean anything.
MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS = 10
#: Below this many classes the averages are not interpretable.
MIN_CLASSES_FOR_MEANINGFUL_METRICS = 2
#: Below this many examples for a class its F1 is not interpretable.
MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1 = 2


class ResponseEvaluationReport(BaseModel):
    """
    Outcome of evaluating predicted response types against gold annotations.

    Metric fields are ``Optional[float]``: they are ``None`` when the data is
    too small to support them, instead of being filled with a number that looks
    authoritative but is not.
    """

    is_meaningful: bool = False
    reason: Optional[str] = None

    n_documents: int = 0
    n_classes: int = 0
    #: Class order used by ``confusion_matrix`` rows and columns.
    class_order: List[str] = Field(default_factory=list)
    #: Comments where the model abstained (winning class below the floor).
    #: Counted as misses, so ``accuracy`` accounts for them.
    n_abstained: int = 0
    #: Comments whose gold role the model cannot emit at all.
    #: Counted as misses; excluded from the per-class metrics.
    n_unscoreable: int = 0

    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    micro_precision: Optional[float] = None
    weighted_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    micro_recall: Optional[float] = None
    weighted_recall: Optional[float] = None

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
            "Response classification evaluation",
            f"  documents        : {self.n_documents}",
            f"  response types   : {self.n_classes}",
        ]
        if self.n_abstained:
            lines.append(f"  abstained        : {self.n_abstained}")
        if self.n_unscoreable:
            lines.append(
                f"  unscoreable gold : {self.n_unscoreable} (role not in the model)"
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
                f"  macro-F1         : {self.macro_f1:.4f}  (every type counts equally)",
                f"  micro-F1         : {self.micro_f1:.4f}  (== accuracy here)",
                f"  weighted-F1      : {self.weighted_f1:.4f}  (per-class F1 by support)",
                f"  macro-precision  : {self.macro_precision:.4f}",
                f"  micro-precision  : {self.micro_precision:.4f}",
                f"  weighted-precision: {self.weighted_precision:.4f}",
                f"  macro-recall     : {self.macro_recall:.4f}",
                f"  micro-recall     : {self.micro_recall:.4f}",
                f"  weighted-recall  : {self.weighted_recall:.4f}",
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


def _average(
    metric,
    gold: np.ndarray,
    guesses: np.ndarray,
    average: str,
) -> float:
    """
    Score one averaging mode.

    ``average="weighted"`` needs no extra argument: scikit-learn derives the
    per-class weights from the gold support itself, so passing a ``weights=``
    list here is an error, not a refinement.
    """
    return float(metric(gold, guesses, average=average, zero_division=0))


def evaluate_response_classification(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str],
) -> ResponseEvaluationReport:
    """
    Compute single-label multiclass metrics for response classification.

    ``y_true`` / ``y_pred`` are sequences of response-type names.  ``None`` is
    allowed in either and is handled honestly rather than silently: a ``None``
    prediction is an abstention, a ``None`` gold value is a role the model cannot
    emit, and both are counted as misses.
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
        return ResponseEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_documents} evaluation document(s); at least "
                f"{MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS} are required before "
                "response-classification metrics are reported"
            ),
            n_documents=n_documents,
            n_classes=len(order),
            class_order=order,
            n_abstained=n_abstained,
            n_unscoreable=n_unscoreable,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    if len(order) < MIN_CLASSES_FOR_MEANINGFUL_METRICS:
        return ResponseEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {len(order)} response type(s) present in the evaluation set; "
                f"at least {MIN_CLASSES_FOR_MEANINGFUL_METRICS} are required for "
                "macro-averages and a confusion matrix to mean anything"
            ),
            n_documents=n_documents,
            n_classes=len(order),
            class_order=order,
            n_abstained=n_abstained,
            n_unscoreable=n_unscoreable,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    unpredicted = [name for name in order if predicted_counts[name] == 0]
    if unpredicted:
        warnings.append(
            f"response type(s) {', '.join(unpredicted)} were never predicted on the "
            "evaluation set; macro-averages are penalised accordingly"
        )
    thin = [
        name
        for name in order
        if 0 < support[name] < MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1
    ]
    if thin:
        warnings.append(
            f"response type(s) {', '.join(thin)} have fewer than "
            f"{MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1} examples; their per-class F1 "
            "is unstable and should not be quoted"
        )
    if n_abstained:
        warnings.append(
            f"{n_abstained} of {n_documents} comment(s) were abstained on (winning "
            "class below the confidence floor) and counted as misses; "
            "confusion-matrix row sums are below their support by that amount"
        )
    if n_unscoreable:
        warnings.append(
            f"{n_unscoreable} of {n_documents} comment(s) have a gold role the model "
            "cannot emit; they are counted as misses and excluded from the per-class "
            "metrics, which are therefore computed on the remainder"
        )

    index = {name: position for position, name in enumerate(order)}
    n_classes = len(order)

    # Indicator form, so one code path serves the aggregates and the per-class
    # metrics.  Abstentions and unscoreable rows stay all-zero, i.e. they match no
    # gold class, which is what makes them misses.
    gold = np.zeros((n_documents, n_classes), dtype=int)
    guesses = np.zeros((n_documents, n_classes), dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    correct = 0
    for row, (actual, guess) in enumerate(zip(truth, predicted)):
        if actual is not None:
            gold[row, index[actual]] = 1
        if guess is not None:
            guesses[row, index[guess]] = 1
        # Every scorable row lands in the matrix, correct or not: a confusion
        # matrix that only recorded the diagonal would be an accuracy counter
        # wearing a matrix's name.  The interesting cells in this task are the
        # off-diagonal ones - advice vs solution, opinion vs counterargument.
        if actual is not None and guess is not None:
            matrix[index[actual], index[guess]] += 1
            if actual == guess:
                correct += 1

    support_counts = [support[name] for name in order]
    # Every gold row unscoreable means every accuracy/F1 denominator is zero.
    # Withhold the metrics instead of emitting a confident 0.0.
    if not any(support_counts):
        warnings.append(
            "no evaluation comment has a gold response type this model can score"
        )
        return ResponseEvaluationReport(
            is_meaningful=False,
            reason=(
                "no evaluation comment has a gold response type this model can "
                "score, so accuracy and every F1 are undefined"
            ),
            n_documents=n_documents,
            n_classes=n_classes,
            class_order=order,
            n_abstained=n_abstained,
            n_unscoreable=n_unscoreable,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    return ResponseEvaluationReport(
        is_meaningful=True,
        reason=None,
        n_documents=n_documents,
        n_classes=n_classes,
        class_order=order,
        n_abstained=n_abstained,
        n_unscoreable=n_unscoreable,
        accuracy=round(correct / n_documents, 6) if n_documents else 0.0,
        macro_f1=round(_average(f1_score, gold, guesses, "macro"), 6),
        micro_f1=round(_average(f1_score, gold, guesses, "micro"), 6),
        weighted_f1=round(
            _average(f1_score, gold, guesses, "weighted"), 6
        ),
        macro_precision=round(_average(precision_score, gold, guesses, "macro"), 6),
        micro_precision=round(_average(precision_score, gold, guesses, "micro"), 6),
        weighted_precision=round(
            _average(precision_score, gold, guesses, "weighted"), 6
        ),
        macro_recall=round(_average(recall_score, gold, guesses, "macro"), 6),
        micro_recall=round(_average(recall_score, gold, guesses, "micro"), 6),
        weighted_recall=round(
            _average(recall_score, gold, guesses, "weighted"), 6
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
