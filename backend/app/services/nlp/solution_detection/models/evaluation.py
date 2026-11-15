"""
Evaluation for binary solution detection.

Which number to quote
---------------------
The positive class is ``solution``, and its **F1** is the number that matters.
Everything else is context for it:

- **accuracy** is the most familiar figure and the most misleading one here.
  Most comments contain no solution, so an always-``False`` model scores the
  negative share and looks excellent while finding nothing at all.  It is
  reported - because it is asked for, and because it is easy to misread - but
  never alone, and never without support counts.
- **precision** is the number a downstream Issue-Solution linker should optimise
  first.  A false positive asserts that a problem was answered when it was not,
  and that error propagates into the knowledge base.  A missed solution is a
  retrieved-comment that nobody needed; a wrong one is a wrong fact.
- **recall** bounds how much of the actionable content is even reachable.  Low
  recall is recoverable (widen the net, raise the threshold trade later), low
  precision is not (the links are already published).
- **F1** is their harmonic mean and is only a useful summary when precision and
  recall are close.  When they are not, report them separately rather than
  quoting the mean.

The confusion matrix
--------------------
Axis order is canonical ``[not_solution, solution]``::

    [[true_negative,  false_positive],
     [false_negative, true_positive]]

The four counts are also exposed as named fields, because indexing a 2x2 matrix
is exactly the sort of thing that quietly produces a transposed metric.  The
off-diagonal cells are the finding: a large ``false_positive`` cell means the
model asserts solutions where there are none, which is the failure that would
break Issue-Solution Linking.

Honesty rules
-------------
Nothing is fabricated.  If the evaluation set is too small, or lacks one of the
two targets, ``is_meaningful`` is False, ``reason`` says exactly why, and every
metric field stays ``None``.  Support counts are still reported, because those
are facts about the data rather than claims about the model.

Two special cases are counted rather than dropped:

- ``y_pred = None`` means the model **abstained** (inside the optional band);
- ``y_true = None`` means the gold value is not scoreable - an unannotated
  comment, or a target the model cannot emit.

Both count as misses for accuracy, and both are excluded from the confusion
matrix and the per-class metrics, which are therefore computed on the scorable
remainder.  The warnings state exactly how many, because dropping rows quietly
improves the numbers.
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

from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    TARGET_LABELS,
    label_to_bool,
    parse_bool_cell,
)

#: Below this many documents an evaluation set is too small for the numbers to
#: mean anything.
MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS = 10
#: A binary detector needs at least this many examples of the positive class
#: before its precision/recall are measurements rather than arithmetic.
MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS = 2
#: Below this many examples of either target its own F1 is unstable.
MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1 = 2


class SolutionEvaluationReport(BaseModel):
    """
    Outcome of evaluating predicted solution flags against gold annotations.

    Metric fields are ``Optional[float]``: ``None`` when the data cannot support
    them, rather than a number that looks authoritative and is not.
    """

    is_meaningful: bool = False
    reason: Optional[str] = None

    n_documents: int = 0
    #: Documents where both a gold and a predicted value existed.
    n_scorable: int = 0
    n_classes: int = 2
    #: Axis order for ``confusion_matrix`` rows and columns.
    class_order: List[str] = Field(default_factory=list)
    #: Comments the model abstained on.  Misses for accuracy.
    n_abstained: int = 0
    #: Comments whose gold target was not scoreable.  Misses for accuracy.
    n_unscoreable: int = 0

    accuracy: Optional[float] = None
    #: Precision for the positive class: of the comments called a solution, how
    #: many really were one.  This is the number a linker should optimise first.
    precision: Optional[float] = None
    #: Recall for the positive class (solutions found, out of all solutions).
    recall: Optional[float] = None
    #: Harmonic mean of ``precision`` and ``recall``.
    f1: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_precision: Optional[float] = None
    weighted_recall: Optional[float] = None
    weighted_f1: Optional[float] = None

    #: Named so a 2x2 matrix never has to be indexed by eye.
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    per_class_f1: Dict[str, float] = Field(default_factory=dict)
    per_class_precision: Dict[str, float] = Field(default_factory=dict)
    per_class_recall: Dict[str, float] = Field(default_factory=dict)
    per_class_support: Dict[str, int] = Field(default_factory=dict)
    per_class_predicted: Dict[str, int] = Field(default_factory=dict)

    #: 2x2 over ``class_order``; withheld when not meaningful.
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

    @property
    def precision_recall_gap(self) -> Optional[float]:
        """
        ``recall - precision``; ``None`` when withheld.

        A large gap is the honest signal that F1 is summarising two different
        behaviours rather than one.  Positive means the model is conservative
        (misses solutions it could have found); negative means it over-fires.
        """
        if self.precision is None or self.recall is None:
            return None
        return round(self.recall - self.precision, 6)

    def format_summary(self) -> str:
        """Human-readable summary that never overstates the numbers."""
        lines = [
            "Solution detection evaluation",
            f"  documents        : {self.n_documents}",
            f"  scorable         : {self.n_scorable}",
            f"  support          : "
            f"{self.per_class_support.get(LABEL_NOT_SOLUTION, 0)} not_solution, "
            f"{self.per_class_support.get(LABEL_SOLUTION, 0)} solution",
        ]
        if self.n_abstained:
            lines.append(
                f"  abstained        : {self.n_abstained} (banded positives, counted "
                "as misses)"
            )
        if self.n_unscoreable:
            lines.append(
                f"  unscoreable gold : {self.n_unscoreable} (not scoreable, counted "
                "as misses)"
            )
        if not self.is_meaningful:
            lines.append(f"  STATUS           : NOT MEANINGFUL - {self.reason}")
            lines.append(
                "  metrics          : withheld (would be misleading at this data size)"
            )
            return "\n".join(lines)

        lines.extend(
            [
                f"  accuracy         : {self.accuracy:.4f}"
                "  (most comments are not solutions, so this flatters)",
                f"  precision        : {self.precision:.4f}"
                "  (positive class: of the comments called a solution, how many were)",
                f"  recall           : {self.recall:.4f}"
                "  (positive class: solutions found / all solutions)",
                f"  F1               : {self.f1:.4f}  (quote with precision AND recall)",
                f"  macro-F1         : {self.macro_f1:.4f}",
                f"  weighted-F1      : {self.weighted_f1:.4f}",
                f"  macro-precision  : {self.macro_precision:.4f}",
                f"  macro-recall     : {self.macro_recall:.4f}",
                f"  weighted-precision: {self.weighted_precision:.4f}",
                f"  weighted-recall  : {self.weighted_recall:.4f}",
                "",
                f"  TP={self.true_positives}  FP={self.false_positives}  "
                f"FN={self.false_negatives}  TN={self.true_negatives}",
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


def _average(metric, gold: np.ndarray, guesses: np.ndarray, average: str) -> float:
    """
    Score one averaging mode.

    ``average="weighted"`` needs no extra argument: scikit-learn derives the
    per-class weights from the gold support itself, so passing a ``weights=``
    list here is an error, not a refinement.
    """
    return float(metric(gold, guesses, average=average, zero_division=0))


def _as_target(value: Any) -> Optional[bool]:
    """
    Normalise one gold or predicted cell to a boolean, or ``None`` if it cannot be.

    Coercing with ``bool()`` would be actively harmful here: ``bool("maybe")`` is
    ``True``, so a mistyped or half-filled annotation cell would be scored as a
    confident *positive* and quietly inflate every rate.  ``None`` is the honest
    answer, and it routes the row into ``n_unscoreable`` where the warnings state
    how many rows were set aside.

    Accepted: real booleans (including ``numpy.bool_``), the canonical label
    strings, the boolean spellings :func:`parse_bool_cell` understands, and 0/1.
    Everything else is unscoreable.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        as_label = label_to_bool(value)
        if as_label is not None:
            return as_label
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return parse_bool_cell(value)
        except ValueError:
            return None
    if isinstance(value, str):
        try:
            return parse_bool_cell(value)
        except ValueError:
            return None
    return None


def evaluate_solution_detection(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[str]] = None,
) -> SolutionEvaluationReport:
    """
    Compute binary metrics for solution detection.

    ``y_true`` / ``y_pred`` are sequences of booleans (``is_solution``).  ``None``
    is allowed in either and is handled honestly rather than silently: a ``None``
    prediction is an abstention, a ``None`` gold value is not scoreable, and both
    count as misses.  Values that are neither booleans nor recognised boolean
    spellings are treated as ``None`` rather than coerced - see :func:`_as_target`.
    """
    truth: List[Optional[bool]] = [_as_target(value) for value in list(y_true)]
    predicted: List[Optional[bool]] = [_as_target(value) for value in list(y_pred)]

    if len(truth) != len(predicted):
        raise ValueError(
            f"y_true has {len(truth)} entries but y_pred has {len(predicted)}"
        )

    names = list(labels) if labels else list(TARGET_LABELS)
    n_documents = len(truth)

    support = {
        LABEL_NOT_SOLUTION: sum(1 for value in truth if value is False),
        LABEL_SOLUTION: sum(1 for value in truth if value is True),
    }
    predicted_counts = {
        LABEL_NOT_SOLUTION: sum(1 for value in predicted if value is False),
        LABEL_SOLUTION: sum(1 for value in predicted if value is True),
    }
    n_abstained = sum(1 for value in predicted if value is None)
    n_unscoreable = sum(1 for value in truth if value is None)
    order = [name for name in names if name in (LABEL_NOT_SOLUTION, LABEL_SOLUTION)] or [
        LABEL_NOT_SOLUTION,
        LABEL_SOLUTION,
    ]

    def withheld(reason: str, warnings: List[str]) -> SolutionEvaluationReport:
        return SolutionEvaluationReport(
            is_meaningful=False,
            reason=reason,
            n_documents=n_documents,
            n_scorable=sum(
                1 for a, b in zip(truth, predicted) if a is not None and b is not None
            ),
            n_classes=2,
            class_order=order,
            n_abstained=n_abstained,
            n_unscoreable=n_unscoreable,
            per_class_support=support,
            per_class_predicted=predicted_counts,
            warnings=warnings,
        )

    if n_documents < MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS:
        return withheld(
            f"only {n_documents} evaluation document(s); at least "
            f"{MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS} are required before "
            "solution-detection metrics are reported",
            [],
        )

    # A binary detector cannot be measured with one target missing: precision and
    # recall for the positive class would be 0.0 or undefined, and a confident
    # 0.0 here would be indistinguishable from a genuinely bad model.
    missing = [
        name
        for name, count in support.items()
        if count == 0
    ]
    if missing:
        return withheld(
            f"the evaluation set contains no example of {', '.join(missing)}; a "
            "binary detector needs both targets before its precision, recall and "
            "F1 are interpretable",
            [],
        )

    warnings: List[str] = []
    if support[LABEL_SOLUTION] < MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS:
        warnings.append(
            f"only {support[LABEL_SOLUTION]} solution example(s) in the evaluation "
            "set; precision and recall for the positive class will swing wildly on "
            "a single comment and should not be quoted"
        )
    for name, count in sorted(support.items()):
        if 0 < count < MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1:
            warnings.append(
                f"{name} has {count} example(s); its per-class F1 is unstable and "
                "should not be quoted on its own"
            )
    if n_abstained:
        warnings.append(
            f"{n_abstained} of {n_documents} comment(s) had no usable predicted value - "
            "the detector abstained inside the band, or the value could not be read "
            "as a boolean - and are counted as misses; the confusion matrix and "
            "per-class metrics exclude them"
        )
    if n_unscoreable:
        warnings.append(
            f"{n_unscoreable} of {n_documents} comment(s) have a gold value that is "
            "not scoreable; they are counted as misses and excluded from the "
            "confusion matrix and per-class metrics, which are therefore computed "
            "on the remainder"
        )
    if predicted_counts[LABEL_SOLUTION] == 0:
        warnings.append(
            "no comment was asserted as a solution, so recall is 0 by construction; "
            "the threshold is probably too high for this corpus"
        )
    if predicted_counts[LABEL_NOT_SOLUTION] == 0:
        warnings.append(
            f"every comment was asserted as a solution, so recall is 1.0 by "
            f"construction and precision collapses to the solution base rate "
            f"({predicted_counts[LABEL_SOLUTION]}/{n_documents}); the threshold is "
            "probably too low, or the detector has collapsed onto the positive class"
        )

    # Scorable subset: both a gold and a predicted value.  Indicator form, so one
    # code path serves the aggregates and the per-class metrics; excluded rows
    # stay all-zero, which is what makes them misses.
    scorable_pairs = [
        (a, b) for a, b in zip(truth, predicted) if a is not None and b is not None
    ]
    n_scorable = len(scorable_pairs)
    if not n_scorable:
        return withheld(
            "no evaluation comment has both a gold and a predicted value, so "
            "accuracy and every rate are undefined",
            warnings,
        )

    gold = np.asarray([1 if a else 0 for a, _ in scorable_pairs], dtype=int)
    guesses = np.asarray([1 if b else 0 for _, b in scorable_pairs], dtype=int)

    # Named counts, computed rather than read back off the matrix.
    tp = int(np.sum((gold == 1) & (guesses == 1)))
    fp = int(np.sum((gold == 0) & (guesses == 1)))
    fn = int(np.sum((gold == 1) & (guesses == 0)))
    tn = int(np.sum((gold == 0) & (guesses == 0)))

    # Every scorable row lands in the matrix, correct or not: a confusion matrix
    # that only recorded the diagonal would be an accuracy counter wearing a
    # matrix's name.
    matrix = [[tn, fp], [fn, tp]]

    indicator_gold = np.zeros((n_scorable, 2), dtype=int)
    indicator_guess = np.zeros((n_scorable, 2), dtype=int)
    indicator_gold[np.arange(n_scorable), gold] = 1
    indicator_guess[np.arange(n_scorable), guesses] = 1

    # Accuracy counts every document, so abstentions and unscoreable rows are
    # misses rather than silently excluded.
    correct_all = sum(
        1 for a, b in zip(truth, predicted) if a is not None and b is not None and a == b
    )

    return SolutionEvaluationReport(
        is_meaningful=True,
        reason=None,
        n_documents=n_documents,
        n_scorable=n_scorable,
        n_classes=2,
        class_order=order,
        n_abstained=n_abstained,
        n_unscoreable=n_unscoreable,
        accuracy=round(correct_all / n_documents, 6) if n_documents else 0.0,
        precision=round(float(precision_score(gold, guesses, zero_division=0)), 6),
        recall=round(float(recall_score(gold, guesses, zero_division=0)), 6),
        f1=round(float(f1_score(gold, guesses, zero_division=0)), 6),
        macro_precision=round(_average(precision_score, indicator_gold, indicator_guess, "macro"), 6),
        macro_recall=round(_average(recall_score, indicator_gold, indicator_guess, "macro"), 6),
        macro_f1=round(_average(f1_score, indicator_gold, indicator_guess, "macro"), 6),
        weighted_precision=round(_average(precision_score, indicator_gold, indicator_guess, "weighted"), 6),
        weighted_recall=round(_average(recall_score, indicator_gold, indicator_guess, "weighted"), 6),
        weighted_f1=round(_average(f1_score, indicator_gold, indicator_guess, "weighted"), 6),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        per_class_f1={
            name: round(
                float(f1_score(indicator_gold[:, position], indicator_guess[:, position], zero_division=0)),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_precision={
            name: round(
                float(precision_score(indicator_gold[:, position], indicator_guess[:, position], zero_division=0)),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_recall={
            name: round(
                float(recall_score(indicator_gold[:, position], indicator_guess[:, position], zero_division=0)),
                6,
            )
            for position, name in enumerate(order)
        },
        per_class_support=support,
        per_class_predicted=predicted_counts,
        confusion_matrix=matrix,
        classification_report=classification_report(
            gold,
            guesses,
            labels=[0, 1],
            target_names=[LABEL_NOT_SOLUTION, LABEL_SOLUTION],
            zero_division=0,
            output_dict=True,
        ),
        warnings=warnings,
    )
