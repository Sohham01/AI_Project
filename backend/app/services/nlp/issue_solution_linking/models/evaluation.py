"""
Evaluation against human-annotated link pairs.

The central rule
----------------
**No annotated link dataset exists in this project, so no link precision, recall
or F1 may be quoted.**  :func:`evaluate_linking` requires a non-empty
:class:`~.models.annotation.LinkCorpus` and refuses otherwise.  There is no
fallback, no synthetic "example" metric, and no default corpus.

This is the first stage in the project with no trained model to score against a
labelled test set, and that gap is structural rather than incidental:

- Every earlier stage classifies a single comment against per-comment labels, so
  ``test_*.py`` can assert the *mechanics* and a labelled CSV can assert the
  *quality*.
- A linker asserts a relation between two comments.  Its mechanics are testable;
  its quality is not, because quality is exactly the question of whether the
  relation a human annotated is the relation the model asserted.  Only a human
  can answer that, so only a human-labelled file can measure it.

Consequently this module is infrastructure for an annotation round that has not
happened yet.  The tests exercise it with annotations declared inline and
explicitly synthetic - they verify that the arithmetic and the refusal behaviour
are correct, and they are never a measurement of the linker.

One subtlety worth stating: a metric over a *subset* of pairs is not a metric
over the system.  Candidates the linker never scored (rejected by thread
placement) can be unlinkable in the annotations, and a run that scores fewer
pairs can look better purely by declining to be judged on hard ones.  So
:meth:`LinkingEvaluation.uncovered_positives` and ``unscored_annotated_pairs``
are reported alongside every metric, and ``format_report`` prints them.
"""

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.services.nlp.issue_solution_linking.models.annotation import LinkCorpus
from app.services.nlp.issue_solution_linking.models.result import LinkPrediction, LinkingResult


def _confusion(
    result: LinkingResult, corpus: LinkCorpus
) -> Dict[str, Any]:
    """
    Match a run's scored links against annotated pairs.

    Matching is on the full triple ``(issue_document_id, issue_label,
    solution_document_id)`` because that is the unit the annotation schema
    defines.  Matching on document ids alone would silently conflate two labels
    of one issue with a single judgement.
    """
    predicted: Dict[str, bool] = {}
    for link in result.links:
        key = f"{link.issue_document_id}|{link.issue_label}|{link.solution_document_id}"
        # A duplicate key can only arise from duplicate inputs; a link is
        # asserted if *any* of its instances cleared the threshold.
        predicted[key] = predicted.get(key, False) or bool(link.linked)

    truth: Dict[str, bool] = {}
    duplicate_judgements: List[str] = []
    for annotation in corpus.annotations:
        key = annotation.pair_key()
        if key in truth:
            if truth[key] != bool(annotation.linked):
                duplicate_judgements.append(key)
            continue
        truth[key] = bool(annotation.linked)

    true_positive = false_positive = true_negative = false_negative = 0
    for key, annotated in truth.items():
        asserted = predicted.get(key)
        if asserted is None:
            continue  # never scored; counted separately
        if annotated and asserted:
            true_positive += 1
        elif not annotated and asserted:
            false_positive += 1
        elif not annotated and not asserted:
            true_negative += 1
        else:
            false_negative += 1

    unscored = sorted(key for key in truth if key not in predicted)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "annotated_positive": sum(1 for value in truth.values() if value),
        "annotated_negative": sum(1 for value in truth.values() if not value),
        "annotated_pairs": len(truth),
        "scored_annotated_pairs": len(truth) - len(unscored),
        "unscored_annotated_pairs": unscored,
        "conflicting_judgements": sorted(set(duplicate_judgements)),
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """Ratio that is ``None`` when undefined, never 0.0 masquerading as a score."""
    if denominator == 0:
        return None
    return numerator / denominator


class LinkingEvaluation(BaseModel):
    """
    Metrics for one linking run against an annotated corpus.

    Every ratio field is ``Optional[float]``.  A corpus with no annotated
    negatives has no measurable precision-specificity behaviour, and reporting
    ``0.0`` there would be a fabricated number; ``None`` with an explanation is
    the honest value.
    """

    model_name: str
    model_version: str
    link_threshold: float

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    annotated_positive: int = 0
    annotated_negative: int = 0
    annotated_pairs: int = 0
    scored_annotated_pairs: int = 0

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    specificity: Optional[float] = None
    accuracy: Optional[float] = None

    #: Annotated pairs the linker never produced a candidate for.
    unscored_annotated_pairs: List[str] = Field(default_factory=list)
    #: Annotated positives that were never scored - the metric's blind spot.
    uncovered_positives: List[str] = Field(default_factory=list)
    #: Triples annotated both ways by a human.
    conflicting_judgements: List[str] = Field(default_factory=list)
    #: Explicit reminder attached to every report produced here.
    note: str = (
        "Computed against an explicitly supplied link-annotation corpus. "
        "No such dataset ships with this project; any figure here describes "
        "the supplied corpus only, and the annotation set itself is small."
    )

    @property
    def is_degenerate_corpus(self) -> bool:
        """True when the corpus cannot support a meaningful measurement."""
        return self.annotated_pairs == 0 or self.annotated_positive == 0

    def format_report(self) -> str:
        def show(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        lines = [
            "Issue-Solution Linking - evaluation against annotated pairs",
            f"  model              : {self.model_name} {self.model_version}",
            f"  link_threshold     : {self.link_threshold}",
            f"  annotated pairs    : {self.annotated_pairs} "
            f"({self.annotated_positive} positive / {self.annotated_negative} negative)",
            f"  scored of annotated: {self.scored_annotated_pairs}",
            f"  TP / FP / TN / FN  : {self.true_positive} / {self.false_positive} "
            f"/ {self.true_negative} / {self.false_negative}",
            f"  precision          : {show(self.precision)}",
            f"  recall             : {show(self.recall)}",
            f"  f1                 : {show(self.f1)}",
            f"  specificity        : {show(self.specificity)}",
            f"  accuracy           : {show(self.accuracy)}",
        ]
        if self.uncovered_positives:
            lines.append(
                f"  ! {len(self.uncovered_positives)} annotated positive(s) were never "
                "scored as candidates, so recall is optimistic; not one of these "
                "confusions could have been avoided by the linker."
            )
        if self.unscored_annotated_pairs:
            lines.append(
                f"  ! {len(self.unscored_annotated_pairs)} annotated pair(s) produced no "
                "candidate at all (see the result's candidate_rejections)."
            )
        if self.conflicting_judgements:
            lines.append(
                f"  ! {len(self.conflicting_judgements)} pair(s) were annotated both "
                "linked and not linked; those judgements are mutually exclusive and "
                "are counted as the first one seen."
            )
        if self.is_degenerate_corpus:
            lines.append(
                "  ! corpus has no annotated positives, so precision/recall/F1 are "
                "undefined and accuracy alone says very little."
            )
        lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def evaluate_linking(
    result: LinkingResult,
    corpus: LinkCorpus,
    on_conflict: str = "report",
) -> LinkingEvaluation:
    """
    Score a linking run against explicit human annotations.

    Raises ``ValueError`` for an empty corpus.  This refusal is the entire point
    of the function's existence: a caller that reaches for a metric with nothing
    annotated should get an error, not a number.

    ``on_conflict`` handles triples annotated both ``linked=True`` and
    ``linked=False``, which is a real annotator disagreement rather than a data
    bug:

    - ``"report"`` (default): count the first judgement, list the pair in
      ``conflicting_judgements`` and warn in the report.
    - ``"raise"``: refuse to produce any metric over a corpus that contradicts
      itself, on the grounds that there is no single right answer for it.
    """
    if on_conflict not in ("report", "raise"):
        raise ValueError(
            f"on_conflict must be 'report' or 'raise', got {on_conflict!r}"
        )
    if corpus is None or len(corpus) == 0:
        raise ValueError(
            "cannot evaluate linking without annotated link pairs. This project "
            "ships no link-annotation dataset, so no link precision/recall may be "
            "reported. Annotate candidate pairs per "
            "LinkAnnotation(issue_document_id, issue_label, solution_document_id, "
            "linked, rationale) and pass them in explicitly."
        )
    corpus.validate()

    counts = _confusion(result, corpus)
    conflicts = counts["conflicting_judgements"]
    if conflicts and on_conflict == "raise":
        raise ValueError(
            f"{len(conflicts)} annotated pair(s) have contradictory judgements "
            f"({', '.join(conflicts[:3])}"
            f"{', ...' if len(conflicts) > 3 else ''}). Pass on_conflict='report' "
            "to score the corpus anyway, with the disagreement reported."
        )

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    uncovered = sorted(
        key
        for key in counts["unscored_annotated_pairs"]
        if key in set(corpus.positive_keys())
    )

    return LinkingEvaluation(
        model_name=result.model_name,
        model_version=result.model_version,
        link_threshold=result.link_threshold,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        annotated_positive=counts["annotated_positive"],
        annotated_negative=counts["annotated_negative"],
        annotated_pairs=counts["annotated_pairs"],
        scored_annotated_pairs=counts["scored_annotated_pairs"],
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=_ratio(tn, tn + fp),
        accuracy=_ratio(tp + tn, tp + tn + fp + fn),
        unscored_annotated_pairs=list(counts["unscored_annotated_pairs"]),
        uncovered_positives=uncovered,
        conflicting_judgements=conflicts,
    )


def threshold_sweep(
    result: LinkingResult,
    corpus: LinkCorpus,
    thresholds: Sequence[float],
) -> List[LinkingEvaluation]:
    """
    Evaluate the same run at several thresholds, without re-running the model.

    The score is computed once; only the boolean decision changes.  This is the
    supported way to pick a threshold, and it is *not* a substitute for a
    threshold chosen against a corpus that does not exist yet.
    """
    if corpus is None or len(corpus) == 0:
        raise ValueError("threshold_sweep requires an explicit annotated corpus")
    evaluations: List[LinkingEvaluation] = []
    for threshold in thresholds:
        swept = result.model_copy(update={"links": result.links_at_threshold(threshold)})
        swept = swept.model_copy(update={"link_threshold": threshold})
        evaluations.append(evaluate_linking(swept, corpus))
    return evaluations
