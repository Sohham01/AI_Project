"""
Evaluation against human-adjudicated resolution outcomes.

The central rule
----------------
**No annotated resolution dataset exists in this project, so no accuracy, macro
F1 or confusion matrix may be quoted.**  :func:`evaluate_resolution` requires a
non-empty :class:`~.models.annotation.ResolutionCorpus` and refuses otherwise.
There is no fallback and no synthetic "example" metric.

The tests that exercise this module declare annotations inline and explicitly
synthetic.  They verify the arithmetic and the refusal behaviour.  They are not,
and must not be reported as, a measurement of the stage.

Guards that matter more than usual here
--------------------------------------
A resolution metric is unusually easy to publish misleadingly, because every
number it produces looks like a clean count.  Four guards are applied, all of
them reported alongside the metrics:

1. **A corpus that covers only one status cannot measure much.**  If gold is
   100% ``unresolved``, accuracy is 1.0 for a resolver that always says
   ``unresolved`` - a perfect score for a model that learned nothing.  Every
   report therefore prints the gold distribution, and
   :attr:`ResolutionEvaluation.single_status_corpus` flags the degenerate case.

2. **Macro F1 needs support on both sides.**  It is the mean of per-status F1
   over statuses that gold actually contains.  A status with no gold examples has
   no precision, and averaging in a fabricated 0.0 would penalise the model for a
   gap in the annotation set rather than in its predictions.
   :attr:`ResolutionEvaluation.statuses_without_gold_support` names the excluded
   statuses, and ``macro_f1`` is ``None`` when gold covers no status at all.

3. **A metric over a subset is not a metric over the stage.**  Results the
   corpus does not cover are reported as ``unmatched_results``, and annotations
   with no corresponding result as ``unmatched_annotations``.  A run that assessed
   two easy issues and was silent about the rest can otherwise look perfect.

4. **Multi-label matching must not launder one label's verdict into another.**  An
   annotation without a label is matched by ``issue_id`` alone, which is sound
   only when the result carries a single label.  When it carries several, the
   match is refused and counted in :attr:`ResolutionEvaluation.ambiguous_matches`
   rather than attributed to whichever label sorted first.
"""

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.services.nlp.resolution_detection.base import ResolutionStatus, STATUS_ORDER
from app.services.nlp.resolution_detection.models.annotation import (
    ResolutionAnnotation,
    ResolutionCorpus,
)
from app.services.nlp.resolution_detection.models.result import (
    ResolutionBatchResult,
    ResolutionResult,
)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """Ratio that is ``None`` when undefined, never 0.0 masquerading as a score."""
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if (precision + recall) == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class PerStatusMetrics(BaseModel):
    """Precision, recall and F1 for one status, with the counts they came from."""

    status: ResolutionStatus
    #: Gold examples with this status.
    support: int = 0
    #: Results predicted with this status.
    predicted: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    @property
    def has_gold_support(self) -> bool:
        return self.support > 0

    def format_line(self) -> str:
        def show(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        return (
            f"  {self.status.value:<20} support={self.support:<5} "
            f"predicted={self.predicted:<5} "
            f"P={show(self.precision):<7} R={show(self.recall):<7} F1={show(self.f1)}"
        )


class ResolutionEvaluation(BaseModel):
    """
    Metrics for one detection run against an adjudicated corpus.

    Every ratio field is ``Optional[float]``.  A status with no gold examples has
    no measurable recall, and reporting ``0.0`` there would be a fabricated number
    that reads as a failure of the model rather than a gap in the annotations.
    """

    model_name: str
    model_version: str
    thresholds: Dict[str, float] = Field(default_factory=dict)

    annotated_pairs: int = 0
    matched_pairs: int = 0

    correct: int = 0
    accuracy: Optional[float] = None

    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None

    per_status: Dict[str, PerStatusMetrics] = Field(default_factory=dict)
    #: ``{gold_status: {predicted_status: count}}`` - the confusion matrix.
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    gold_status_counts: Dict[str, int] = Field(default_factory=dict)
    predicted_status_counts: Dict[str, int] = Field(default_factory=dict)

    #: Results the corpus does not cover, and annotations with no result.
    unmatched_results: List[str] = Field(default_factory=list)
    unmatched_annotations: List[str] = Field(default_factory=list)
    #: Unlabelled annotations that could not be matched safely.
    ambiguous_matches: List[str] = Field(default_factory=list)
    #: Pairs annotated with more than one gold status.
    conflicting_judgements: List[str] = Field(default_factory=list)
    #: Gold statuses with no examples, and therefore no per-status metrics.
    statuses_without_gold_support: List[str] = Field(default_factory=list)
    #: Results whose verdict rests on a discussion below the minimum volume.
    thin_evidence_results: List[str] = Field(default_factory=list)

    note: str = (
        "Computed against an explicitly supplied resolution-annotation corpus. No "
        "such dataset ships with this project; any figure here describes the "
        "supplied corpus only, and a corpus is not a sample of real discussions."
    )

    @property
    def distinct_gold_statuses(self) -> int:
        return sum(1 for count in self.gold_status_counts.values() if count > 0)

    @property
    def single_status_corpus(self) -> bool:
        """
        True when gold contains only one status.

        A 20-row corpus of all-``unresolved`` issues reports accuracy 1.0 for a
        resolver that always answers "unresolved".  Flagged loudly rather than
        left for a reader to notice.
        """
        return 0 < self.distinct_gold_statuses <= 1

    @property
    def is_degenerate_corpus(self) -> bool:
        return self.annotated_pairs == 0 or self.distinct_gold_statuses == 0

    def confusion_as_rows(self) -> List[str]:
        """The confusion matrix as aligned text rows: gold down, predicted across."""
        statuses = [status.value for status in STATUS_ORDER]
        header = "  gold \\ predicted  " + "".join(f"{name[:12]:>14}" for name in statuses)
        rows = [header]
        for gold in statuses:
            cells = self.confusion_matrix.get(gold, {})
            rows.append(
                f"  {gold:<16}" + "".join(f"{cells.get(pred, 0):>14}" for pred in statuses)
            )
        return rows

    def format_report(self) -> str:
        def show(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        lines = [
            "Resolution Detection - evaluation against adjudicated outcomes",
            f"  model            : {self.model_name} {self.model_version}",
            f"  thresholds       : unresolved<{self.thresholds.get('unresolved_upper', 0):.2f}"
            f" <= partial <{self.thresholds.get('resolved_lower', 0):.2f} <= resolved",
            f"  annotated pairs  : {self.annotated_pairs} (matched: {self.matched_pairs})",
            f"  correct          : {self.correct}",
            f"  accuracy         : {show(self.accuracy)}",
            f"  macro F1         : {show(self.macro_f1)}",
            f"  weighted F1      : {show(self.weighted_f1)}",
            "",
            "  per-status:",
        ]
        for status in STATUS_ORDER:
            metrics = self.per_status.get(status.value)
            if metrics is None:
                continue
            lines.append(metrics.format_line())
        lines.append("")
        lines.append("  confusion matrix (gold rows, predicted columns):")
        lines.extend(self.confusion_as_rows())

        distribution = ", ".join(
            f"{name}={count}" for name, count in self.gold_status_counts.items()
        )
        lines.append(f"  gold distribution: {distribution}")
        if self.single_status_corpus:
            lines.append(
                "  ! every gold label is the same status, so accuracy is not "
                "informative: a resolver that always predicts that status would "
                "score 1.0. Treat these numbers as a smoke test, not a result."
            )
        if self.statuses_without_gold_support:
            lines.append(
                f"  ! no gold examples for: {', '.join(self.statuses_without_gold_support)}. "
                "Their per-status precision/recall/F1 are undefined and excluded from "
                "macro F1 rather than counted as 0."
            )
        if self.thin_evidence_results:
            lines.append(
                f"  ! {len(self.thin_evidence_results)} correct/incorrect verdict(s) here "
                "rest on discussions below the minimum volume; their scores are weaker "
                "evidence than the accuracy figure suggests."
            )
        if self.unmatched_results:
            lines.append(
                f"  ! {len(self.unmatched_results)} result(s) were not covered by the "
                "corpus. A metric over a subset is not a metric over the stage."
            )
        if self.unmatched_annotations:
            lines.append(
                f"  ! {len(self.unmatched_annotations)} annotated issue(s) had no "
                "corresponding result (absent from the evidence, or filtered out)."
            )
        if self.ambiguous_matches:
            lines.append(
                f"  ! {len(self.ambiguous_matches)} unlabelled annotation(s) matched a "
                "multi-label result and were excluded: attributing a document-level "
                "judgement to one of several labels would be a guess."
            )
        if self.conflicting_judgements:
            lines.append(
                f"  ! {len(self.conflicting_judgements)} pair(s) carry more than one gold "
                "status; the first was used and the disagreement is reported."
            )
        lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def evaluate_resolution(
    results: ResolutionBatchResult,
    corpus: ResolutionCorpus,
    on_conflict: str = "report",
) -> ResolutionEvaluation:
    """
    Score a detection run against explicit human adjudications.

    Raises ``ValueError`` for an empty corpus.  That refusal is the point of the
    function: a caller reaching for a metric with nothing annotated should get an
    error, not a number.

    ``on_conflict`` handles issue-label pairs annotated with more than one gold
    status: ``"report"`` (default) uses the first and names the pair;
    ``"raise"`` refuses to score a corpus that contradicts itself.
    """
    if on_conflict not in ("report", "raise"):
        raise ValueError(f"on_conflict must be 'report' or 'raise', got {on_conflict!r}")
    if corpus is None or len(corpus) == 0:
        raise ValueError(
            "cannot evaluate resolution without adjudicated gold statuses. This "
            "project ships no resolution-annotation dataset, so no accuracy, F1 or "
            "confusion matrix may be reported. Annotate issues per "
            "ResolutionAnnotation(issue_id, issue_label, gold_status, rationale) and "
            "pass them in explicitly."
        )
    corpus.validate()

    conflicts = corpus.conflicting_pair_keys()
    if conflicts and on_conflict == "raise":
        raise ValueError(
            f"{len(conflicts)} annotated pair(s) have contradictory gold statuses "
            f"({', '.join(conflicts[:3])}"
            f"{', ...' if len(conflicts) > 3 else ''}). Pass on_conflict='report' to "
            "score the corpus anyway, with the disagreement reported."
        )

    # Index results by pair key, and remember multi-label documents so an
    # unlabelled annotation cannot be attributed to one arbitrary label.
    result_by_key: Dict[str, ResolutionResult] = {}
    labels_by_issue: Dict[str, List[str]] = {}
    for result in results.results:
        result_by_key[result.pair_key()] = result
        labels_by_issue.setdefault(result.issue_id, []).append(result.issue_label)

    confusion: Dict[str, Dict[str, int]] = {
        gold.value: {predicted.value: 0 for predicted in STATUS_ORDER} for gold in STATUS_ORDER
    }
    gold_counts: Dict[str, int] = {status.value: 0 for status in STATUS_ORDER}
    predicted_counts: Dict[str, int] = {status.value: 0 for status in STATUS_ORDER}

    matched: Dict[str, ResolutionResult] = {}
    ambiguous: List[str] = []
    unmatched_annotations: List[str] = []
    seen_keys: set = set()
    correct = 0

    for annotation in corpus.annotations:
        key = annotation.pair_key()
        if key in seen_keys:
            # A duplicate gold status; the first wins.  Counted as one annotation.
            continue
        seen_keys.add(key)

        gold = ResolutionStatus(annotation.gold_status)
        gold_counts[gold.value] += 1

        result = result_by_key.get(key)
        if result is None and not annotation.is_labelled():
            candidates = labels_by_issue.get(annotation.issue_id, [])
            if len(candidates) == 1:
                result = result_by_key.get(f"{annotation.issue_id}|{candidates[0]}")
            elif len(candidates) > 1:
                ambiguous.append(annotation.issue_id)
                gold_counts[gold.value] -= 1
                seen_keys.discard(key)
                continue
        if result is None:
            unmatched_annotations.append(key)
            continue

        matched[key] = result
        predicted = ResolutionStatus(result.status)
        predicted_counts[predicted.value] += 1
        confusion[gold.value][predicted.value] += 1
        if gold == predicted:
            correct += 1

    matched_count = len(matched)
    per_status: Dict[str, PerStatusMetrics] = {}
    f1_values: List[float] = []
    weighted_numerator = 0.0
    for status in STATUS_ORDER:
        gold_support = gold_counts[status.value]
        tp = confusion[status.value][status.value]
        # The *column* sum: how many results were predicted with this status,
        # across every gold class.  Using the row sum here would report the gold
        # support twice and make precision wrong whenever the two differ.
        predicted_total = sum(
            confusion[gold.value][status.value] for gold in STATUS_ORDER
        )
        fp = predicted_total - tp
        fn = gold_support - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _f1(precision, recall)
        per_status[status.value] = PerStatusMetrics(
            status=status,
            support=gold_support,
            predicted=predicted_total,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        if gold_support > 0 and f1 is not None:
            f1_values.append(f1)
            weighted_numerator += f1 * gold_support

    return ResolutionEvaluation(
        model_name=results.model_name,
        model_version=results.model_version,
        thresholds=dict(results.thresholds),
        annotated_pairs=len(corpus),
        matched_pairs=matched_count,
        correct=correct,
        accuracy=_ratio(correct, matched_count),
        macro_f1=(sum(f1_values) / len(f1_values)) if f1_values else None,
        weighted_f1=(
            weighted_numerator / sum(
                per_status[status.value].support for status in STATUS_ORDER
            )
            if f1_values
            else None
        ),
        per_status=per_status,
        confusion_matrix=confusion,
        gold_status_counts=gold_counts,
        predicted_status_counts=predicted_counts,
        unmatched_results=sorted(
            key for key in result_by_key if key not in matched
        ),
        unmatched_annotations=sorted(set(unmatched_annotations)),
        ambiguous_matches=sorted(set(ambiguous)),
        conflicting_judgements=conflicts,
        statuses_without_gold_support=[
            status.value for status in STATUS_ORDER if gold_counts[status.value] == 0
        ],
        thin_evidence_results=sorted(
            result.pair_key() for result in matched.values() if not result.evidence_sufficient
        ),
    )


def evaluate_results_list(
    results: Sequence[ResolutionResult],
    corpus: ResolutionCorpus,
    model_name: str = "",
    model_version: str = "",
    thresholds: Optional[Dict[str, float]] = None,
) -> ResolutionBatchResult:
    """Wrap a plain list of results into a batch so the evaluator can consume it."""
    return ResolutionBatchResult(
        model_name=model_name,
        model_version=model_version,
        results=list(results),
        thresholds=dict(thresholds or {}),
    )
