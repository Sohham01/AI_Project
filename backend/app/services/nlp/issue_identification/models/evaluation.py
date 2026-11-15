"""
Multi-label evaluation for issue identification.

Multi-label evaluation is easy to get wrong, so the rules here are explicit:

- **micro** averages pool all (document, label) decisions, so a rare label
  counts only as much as its true-positive evidence allows.  Micro-F1 is the
  right headline number for imbalanced multi-label data.
- **macro** averages each label equally, so a label nobody predicted drags
  macro-F1 down hard.  Macro-F1 exposes per-label failure, it does not
  describe overall quality.
- **subset accuracy** (exact-set match) is the strictest number: it only
  counts a document correct when *every* label is right.  On genuinely
  multi-label data it is always <= micro-F1 and usually near zero, so it is
  reported but never presented as "accuracy".
- **accuracy** (fraction of correct label decisions) is reported as
  ``hamming_accuracy`` and is usually misleadingly high on sparse multi-label
  data — it counts a document with all-zero true and predicted labels as
  fully correct.

Nothing is fabricated.  If the evaluation set is too small, or a label has
almost no support, ``is_meaningful`` is False and ``reason`` says why, and
the metric fields stay ``None``.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.metrics import precision_score, recall_score

#: Below this many documents an evaluation set is too small for the numbers to
#: mean anything.
MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS = 10
#: Below this many positive examples for a label its F1 is not interpretable.
MIN_POSITIVES_FOR_STABLE_PER_LABEL_F1 = 2
#: Number of labels required before macro-averages are reported.
MIN_LABELS_FOR_MEANINGFUL_METRICS = 2


class IssueEvaluationReport(BaseModel):
    """
    Outcome of evaluating predicted issue labels against gold annotations.

    Metric fields are ``Optional[float]``: they are ``None`` when the data is
    too small to support them, instead of being filled with a number that
    looks authoritative but is not.
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
    average_precision: Optional[float] = None

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
            "Issue identification evaluation",
            f"  documents        : {self.n_documents}",
            f"  labels           : {self.n_labels}",
        ]
        if not self.is_meaningful:
            lines.append(f"  STATUS           : NOT MEANINGFUL - {self.reason}")
            lines.append("  metrics          : withheld (would be misleading at this data size)")
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
        for label in sorted(self.per_label_f1):
            support = self.per_label_support.get(label, 0)
            lines.append(
                f"    {label:<24} F1={self.per_label_f1[label]:.4f}  support={support}"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def _as_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.astype(int)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    label_names: Sequence[str],
) -> IssueEvaluationReport:
    """
    Compute multi-label metrics for a set of binary label matrices.

    ``y_true`` / ``y_pred`` are ``(n_documents, n_labels)`` binary matrices whose
    columns line up with ``label_names``.  Metrics are withheld (rather than
    invented) when the evaluation set is too small to support them.
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

    # --- Data-adequacy gate: report honestly instead of emitting noise ---
    if n_documents < MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS:
        return IssueEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_documents} evaluation document(s); at least "
                f"{MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS} are required before "
                "issue metrics are reported"
            ),
            n_documents=n_documents,
            n_labels=n_labels,
            per_label_support=support,
            per_label_predicted=predicted,
            warnings=warnings,
        )

    if n_labels < MIN_LABELS_FOR_MEANINGFUL_METRICS:
        return IssueEvaluationReport(
            is_meaningful=False,
            reason=(
                f"only {n_labels} label(s) evaluated; issue identification is "
                f"multi-label, so at least {MIN_LABELS_FOR_MEANINGFUL_METRICS} "
                "are required for macro-averages to mean anything"
            ),
            n_documents=n_documents,
            n_labels=n_labels,
            per_label_support=support,
            per_label_predicted=predicted,
            warnings=warnings,
        )

    # A label that never occurs in the gold data cannot be scored meaningfully;
    # zero_division=0 keeps sklearn from raising, and the warning explains it.
    degenerate = [name for name in names if support[name] == 0]
    if degenerate:
        warnings.append(
            f"label(s) {', '.join(sorted(degenerate))} have zero support in the "
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
        if 0 < support[name] < MIN_POSITIVES_FOR_STABLE_PER_LABEL_F1
    ]
    if thin:
        warnings.append(
            f"label(s) {', '.join(sorted(thin))} have fewer than "
            f"{MIN_POSITIVES_FOR_STABLE_PER_LABEL_F1} positive examples; their "
            "per-label F1 is unstable and should not be quoted"
        )

    average_precision = f1_score(
        truth, pred, average="macro", zero_division=0
    )

    return IssueEvaluationReport(
        is_meaningful=True,
        reason=None,
        n_documents=n_documents,
        n_labels=n_labels,
        micro_f1=round(float(f1_score(truth, pred, average="micro", zero_division=0)), 6),
        macro_f1=round(average_precision, 6),
        micro_precision=round(
            float(precision_score(truth, pred, average="micro", zero_division=0)), 6
        ),
        macro_precision=round(
            float(precision_score(truth, pred, average="macro", zero_division=0)), 6
        ),
        macro_recall=round(
            float(recall_score(truth, pred, average="macro", zero_division=0)), 6
        ),
        # Exact set match: every label of a document must be correct.
        subset_accuracy=round(
            float(np.all(truth == pred, axis=1).mean()) if n_documents else 0.0, 6
        ),
        hamming_accuracy=round(
            float(1.0 - hamming_loss(truth, pred)), 6
        ),
        hamming_loss=round(float(hamming_loss(truth, pred)), 6),
        average_precision=round(average_precision, 6),
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
