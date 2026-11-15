"""
Resolution-annotation schema: the shape a *future* human-annotated resolution
dataset takes.

Why this exists and why there is no data
-----------------------------------------
Resolution is the first stage whose output is a **judgement about a discussion**,
and no unit test can supply that judgement.  The earlier stages classify a single
comment, so "is this right?" has an answer - the human-annotated label on that
comment.  Here, correctness means "does a human reading this whole thread agree
that the issue is resolved?", and only a human reading the thread can say.

So the schema is defined in full, ready for an annotation round - and the data
file is deliberately absent::

    no resolution annotations exist -> no accuracy, F1 or confusion matrix may be quoted

That is not caution for its own sake.  An accuracy number computed against
labels the author invented, or against a fixture the author wrote, is a number
that gets quoted in a report as though it measured the system.  The evaluation
module requires an explicit corpus
(:func:`~.models.evaluation.evaluate_resolution` needs
:class:`ResolutionAnnotation` records) and refuses to produce anything without
one.  The tests that exercise it use annotations declared inline and explicitly
synthetic; they verify arithmetic and refusal behaviour and are never a
measurement.

The schema
----------
One row per annotated issue-label pair::

    issue_id,issue_label,gold_status,rationale

``gold_status`` is one of ``resolved``, ``partially_resolved``, ``unresolved``,
and a blank is refused.  A missing judgement silently defaulting to ``unresolved``
would bias every metric computed over the corpus in one direction, and a
resolution *rate* is exactly the number most likely to be quoted.

``issue_label`` is optional in the schema but present in practice.  A
multi-label issue document needs one judgement per label: "fees" and "roi" in the
same comment can resolve differently, and a single gold status per document would
force them to agree.  When the label is omitted the annotation is matched against
results by ``issue_id`` alone, which is only sound for single-label issues.
"""

import csv
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from app.services.nlp.resolution_detection.base import ResolutionStatus
from app.services.nlp.resolution_detection.models.labels import (
    parse_gold_status,
    parse_bool_cell,
)

#: Canonical column order for a resolution-annotation CSV.
RESOLUTION_CSV_COLUMNS = [
    "issue_id",
    "issue_label",
    "gold_status",
    "rationale",
]

#: Relative path where real resolution annotations would live.
DEFAULT_RESOLUTION_CSV = os.path.join("data", "annotations", "resolution_annotations.csv")


class ResolutionAnnotation(BaseModel):
    """
    One human judgement about one issue-label pair.

    ``gold_status`` is the whole point of the record: a human's reading of whether
    the discussion resolved the issue.  ``rationale`` is the field that makes a
    disagreement actionable later, so it is encouraged but not required.
    """

    issue_id: str
    gold_status: ResolutionStatus
    #: Which predicted label the judgement is about.  ``None`` means the
    #: annotation applies to the document as a whole, which is only meaningful
    #: for a single-label issue.
    issue_label: Optional[str] = None

    rationale: Optional[str] = None
    #: Free-form provenance: who adjudicated, and when.  Opaque to the evaluator.
    annotated_by: Optional[str] = None
    annotated_at: Optional[str] = None
    #: Optional per-record annotator confidence, if they gave one.
    annotator_confidence: Optional[float] = None
    #: Set when two annotators disagreed and this row is the adjudicated value.
    adjudicated: bool = False

    @field_validator("issue_id")
    @classmethod
    def _require_issue_id(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("resolution annotation requires a non-empty issue_id")
        return value

    @field_validator("issue_label")
    @classmethod
    def _reject_blank_label(cls, value: Optional[str]) -> Optional[str]:
        """
        A blank label is refused rather than treated as "unlabelled".

        The two mean different things - "this document has one label we did not
        record" versus "this annotator left the field empty" - and silently
        treating the second as the first would let a half-finished annotation
        round masquerade as a deliberate document-level judgement.
        """
        if value is not None and not str(value).strip():
            raise ValueError(
                "resolution annotation has a blank issue_label. Either give a real "
                "label or omit the field entirely; a blank label is not the same as "
                "an unlabelled issue."
            )
        return value

    @field_validator("annotator_confidence")
    @classmethod
    def _check_annotator_confidence(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"annotator_confidence={value} must be in [0.0, 1.0]"
            )
        return value

    @field_validator("gold_status", mode="before")
    @classmethod
    def _require_gold_status(cls, value: Any) -> Any:
        """
        A judgement must be one of the three statuses.

        Enforced before construction so an unusable record cannot exist in memory
        and then be counted by a metric.  A blank defaulting to ``unresolved``
        would bias every resolution rate computed over the corpus in one
        direction.
        """
        return parse_gold_status(value, "resolution gold_status")

    def pair_key(self) -> str:
        """
        Stable identity for matching against a result.

        The same rule as :meth:`IssueResolutionEvidence.pair_key`: an explicit
        label participates, an absent one does not.  Using different rules on the
        two sides would silently fail to match a labelled result against an
        unlabelled annotation.
        """
        if self.issue_label is None or not str(self.issue_label).strip():
            return self.issue_id
        return f"{self.issue_id}|{str(self.issue_label).strip()}"

    def is_labelled(self) -> bool:
        """True when the judgement is about a specific label rather than a document."""
        return self.issue_label is not None and bool(str(self.issue_label).strip())

    def validate_annotation(self) -> bool:
        """
        Re-check the record's shape.  Read-only, like every other stage.

        A validator that wrote back would quietly rewrite the caller's data
        mid-run; the training and evaluation paths must never be the thing that
        edits the annotation file's contents.
        """
        if not str(self.issue_id or "").strip():
            raise ValueError("resolution annotation requires a non-empty issue_id")
        if self.issue_label is not None and not str(self.issue_label).strip():
            raise ValueError(
                f"resolution annotation {self.issue_id!r} has a blank issue_label. Either "
                "give a real label or omit the field entirely; a blank label is not "
                "the same as an unlabelled issue."
            )
        if self.annotator_confidence is not None and not 0.0 <= float(
            self.annotator_confidence
        ) <= 1.0:
            raise ValueError(
                f"resolution annotation {self.issue_id!r} has annotator_confidence="
                f"{self.annotator_confidence}, which must be in [0.0, 1.0]"
            )
        return bool(self.gold_status)

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "issue_id": self.issue_id,
            "issue_label": self.issue_label or "",
            "gold_status": ResolutionStatus(self.gold_status).value,
            "rationale": self.rationale or "",
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "ResolutionAnnotation":
        if not str(row.get("issue_id") or "").strip():
            raise ValueError("resolution row is missing the required 'issue_id'")
        label = row.get("issue_label")
        label = (str(label).strip() or None) if label is not None else None
        if label is not None and not label:
            raise ValueError(
                f"resolution row for issue {row.get('issue_id')!r} has a blank "
                "'issue_label'. Omit the column's value entirely for an unlabelled issue."
            )
        annotation = cls(
            issue_id=str(row["issue_id"]).strip(),
            issue_label=label,
            gold_status=parse_gold_status(
                row.get("gold_status"), f"resolution {row.get('issue_id')!r} gold_status"
            ),
            rationale=(str(row["rationale"]).strip() or None) if row.get("rationale") else None,
            annotated_by=(str(row["annotated_by"]).strip() or None)
            if row.get("annotated_by")
            else None,
            annotated_at=(str(row["annotated_at"]).strip() or None)
            if row.get("annotated_at")
            else None,
        )
        annotation.validate_annotation()
        return annotation


class ResolutionCorpus(BaseModel):
    """
    A collection of :class:`ResolutionAnnotation` records.

    Duplicates are reported rather than silently de-duplicated.  Two annotators
    disagreeing about one issue is the most valuable thing this dataset can
    contain; collapsing it would delete the disagreement, and an
    inter-annotator agreement figure computed over a corpus that had quietly
    hidden its conflicts would be meaningless.
    """

    annotations: List[ResolutionAnnotation] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[ResolutionAnnotation]:
        return iter(self.annotations)

    @property
    def pairs(self) -> List[ResolutionAnnotation]:
        return list(self.annotations)

    def keys(self) -> List[str]:
        return sorted(annotation.pair_key() for annotation in self.annotations)

    def status_counts(self) -> Dict[str, int]:
        """Gold label distribution.  Reported with every metric, always."""
        counts = {status.value: 0 for status in ResolutionStatus}
        for annotation in self.annotations:
            counts[ResolutionStatus(annotation.gold_status).value] += 1
        return counts

    def conflicting_pair_keys(self) -> List[str]:
        """
        Issue-label pairs annotated with more than one gold status.

        A genuine annotator disagreement rather than a data bug, so it is
        reported and the corpus is not refused here.  Quoting a metric over a
        corpus that contradicts itself needs that number to be visible, and
        ``on_conflict="raise"`` is available for callers who would rather not
        score it at all.
        """
        seen: Dict[str, set] = {}
        for annotation in self.annotations:
            seen.setdefault(annotation.pair_key(), set()).add(
                ResolutionStatus(annotation.gold_status).value
            )
        return sorted(key for key, values in seen.items() if len(values) > 1)

    def unlabelled_document_keys(self) -> List[str]:
        """
        Annotations that judge a document without naming a label.

        These can only be matched against single-label results.  Flagged by the
        evaluator when the corresponding result carries several labels, because
        matching them would attribute one label's verdict to another.
        """
        return sorted(
            annotation.issue_id
            for annotation in self.annotations
            if not annotation.is_labelled()
        )

    def validate(self) -> None:
        for annotation in self.annotations:
            annotation.validate_annotation()

    @classmethod
    def from_csv(cls, path: str) -> "ResolutionCorpus":
        """
        Load resolution annotations from a CSV.

        Accepts a full path or one relative to the repository root.  This
        function has never been run on real data: no resolution-annotation file
        exists in this project, by design.  Shipping the loader means an
        annotation round does not also have to invent a format on top of an
        already-built stage.
        """
        resolved = path
        if not os.path.exists(resolved):
            candidate = os.path.join("..", path)
            if os.path.exists(candidate):
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"resolution annotation CSV not found: {path!r}. Expected columns: "
                    f"{', '.join(RESOLUTION_CSV_COLUMNS)}. Real resolution annotations "
                    f"would live at {DEFAULT_RESOLUTION_CSV!r} and are produced by a "
                    "human annotation round; this project ships the schema and the "
                    "loader, never a fabricated dataset."
                )

        rows: List[ResolutionAnnotation] = []
        with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(
                    f"resolution annotation CSV {path!r} is empty (no header row)"
                )
            missing = set(RESOLUTION_CSV_COLUMNS[:3]) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"resolution annotation CSV {path!r} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if all((value or "").strip() == "" for value in row.values() if value):
                    continue
                try:
                    rows.append(ResolutionAnnotation.from_csv_row(row))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(annotations=rows)

    def to_csv(self, path: str) -> str:
        """Write the corpus in the documented format.  Never called on real data."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESOLUTION_CSV_COLUMNS)
            writer.writeheader()
            for annotation in self.annotations:
                writer.writerow(annotation.to_csv_row())
        return path

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> "ResolutionCorpus":
        return cls(annotations=[ResolutionAnnotation(**record) for record in records])


def parse_adjudicated_cell(value: Any, subject: str = "adjudicated") -> bool:
    """Thin wrapper over the shared boolean parser, for an ``adjudicated`` column."""
    return parse_bool_cell(value, subject)
