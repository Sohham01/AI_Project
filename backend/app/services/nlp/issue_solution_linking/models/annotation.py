"""
Link-annotation schema: the shape a *future* human-annotated link dataset takes.

Why this exists and why there is no data
-----------------------------------------
Linking is the first stage in this project whose output cannot be validated by
unit tests alone.  Every earlier stage trains on annotated per-comment labels, so
"is this right?" has an answer.  A linker asserts a *relation*, and the only way
to know whether a relation is right is for a human to say so.  So the schema is
defined here, in full, ready for an annotation round - and the file itself is
deliberately absent.

    no link annotations exist -> no link precision/recall may be quoted

That is not caution for its own sake.  A precision number computed against labels
the author invented, or against a fixture the author wrote, is a number that
will be quoted in a report as though it measured the system.  The evaluation
module in this package refuses to produce one without an explicit dataset
(:func:`~.models.evaluation.evaluate_linking` needs
:class:`LinkAnnotation` records), and the tests that exercise it use
synthetic annotations clearly labelled as such.

The schema
----------
One row per annotated candidate pair::

    issue_document_id,issue_label,solution_document_id,linked,rationale

``linked`` is a real boolean and a blank is refused, for the same reason the
solution stage refuses a blank ``is_solution``: a missing judgement silently
becoming a negative would train nothing and quietly inflate whatever metric used
it.  There is no default, and there is no "unknown" - an undecided pair is not
annotated, and the count of annotated pairs is reported with the metric.
"""

import csv
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.services.nlp.issue_solution_linking.models.labels import parse_bool_cell

#: Canonical column order for a link-annotation CSV.
LINK_CSV_COLUMNS = [
    "issue_document_id",
    "issue_label",
    "solution_document_id",
    "linked",
    "rationale",
]

#: Relative path where real link annotations would live.
DEFAULT_LINK_CSV = os.path.join("data", "annotations", "issue_solution_links.csv")


def parse_linked_cell(value: Any, document_id: Optional[str] = None) -> bool:
    """
    Parse a ``linked`` cell, refusing a blank or unrecognised value.

    A thin, domain-worded wrapper over the shared
    :func:`~.labels.parse_bool_cell`, so the linker and every other stage agree on
    exactly which spellings count as a boolean and on what a blank means.  The
    same discipline as the per-comment stages: an unrecognised judgement is an
    error, never a ``False``.
    """
    subject = f"link {document_id!r} 'linked'" if document_id else "linked value"
    return parse_bool_cell(value, subject)


class LinkAnnotation(BaseModel):
    """
    One human judgement about one (issue, label, solution) pair.

    ``linked=True`` asserts that the solution comment addresses the issue; the
    linker treats the annotation as ground truth and never reads it as anything
    else.  ``rationale`` is the field that makes a disagreement actionable later,
    so it is encouraged but not required.
    """

    issue_document_id: str
    issue_label: str
    solution_document_id: str
    linked: bool

    rationale: Optional[str] = None
    #: Free-form provenance: who annotated it, and when.  Opaque to the linker.
    annotated_by: Optional[str] = None
    annotated_at: Optional[str] = None
    #: Optional per-pair confidence from the annotator, if they gave one.
    annotator_confidence: Optional[float] = None

    def pair_key(self) -> str:
        """Stable identity of this triple, for matching against a run's links."""
        return f"{self.issue_document_id}|{self.issue_label}|{self.solution_document_id}"

    def validate_annotation(self) -> bool:
        """
        Re-check the shape of the record.  Read-only, like every other stage.

        A ``linked`` value is already strictly typed by Pydantic; this exists so
        the loader, the evaluator and a human filling a dict all go through one
        place, and so a blank label is caught with the offending id in the
        message.
        """
        if not self.issue_label.strip():
            raise ValueError(
                f"link annotation {self.pair_key()!r} has a blank issue_label. A link "
                "is always about a specific predicted label, so an unlabelled link "
                "cannot be matched to a run."
            )
        if not self.issue_document_id.strip() or not self.solution_document_id.strip():
            raise ValueError("link annotation requires both document ids")
        return bool(self.linked)

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "issue_document_id": self.issue_document_id,
            "issue_label": self.issue_label,
            "solution_document_id": self.solution_document_id,
            "linked": "true" if self.linked else "false",
            "rationale": self.rationale or "",
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "LinkAnnotation":
        pair = (
            f"{row.get('issue_document_id')}|{row.get('issue_label')}|"
            f"{row.get('solution_document_id')}"
        )
        if not str(row.get("issue_document_id") or "").strip():
            raise ValueError("link row is missing the required 'issue_document_id'")
        if not str(row.get("issue_label") or "").strip():
            raise ValueError("link row is missing the required 'issue_label'")
        if not str(row.get("solution_document_id") or "").strip():
            raise ValueError("link row is missing the required 'solution_document_id'")
        annotation = cls(
            issue_document_id=str(row["issue_document_id"]).strip(),
            issue_label=str(row["issue_label"]).strip(),
            solution_document_id=str(row["solution_document_id"]).strip(),
            linked=parse_linked_cell(row.get("linked"), pair),
            rationale=(str(row["rationale"]).strip() or None)
            if row.get("rationale")
            else None,
            annotated_by=(str(row["annotated_by"]).strip() or None)
            if row.get("annotated_by")
            else None,
            annotated_at=(str(row["annotated_at"]).strip() or None)
            if row.get("annotated_at")
            else None,
        )
        annotation.validate_annotation()
        return annotation


class LinkCorpus(BaseModel):
    """
    A collection of :class:`LinkAnnotation` records.

    Duplicate triples are reported rather than silently de-duplicated: two
    annotators disagreeing about one pair is the most valuable thing this
    dataset can contain, and collapsing it would delete the disagreement.
    """

    annotations: List[LinkAnnotation] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[LinkAnnotation]:
        return iter(self.annotations)

    @property
    def pairs(self) -> List[LinkAnnotation]:
        return list(self.annotations)

    def positive_keys(self) -> List[str]:
        """Pair keys a human marked as linked, sorted."""
        return sorted(a.pair_key() for a in self.annotations if a.linked)

    def negative_keys(self) -> List[str]:
        """Pair keys a human marked as not linked, sorted."""
        return sorted(a.pair_key() for a in self.annotations if not a.linked)

    def conflicting_pair_keys(self) -> List[str]:
        """
        Triples annotated both ``linked=True`` and ``linked=False``.

        A genuine annotator disagreement rather than a data bug, so it is
        reported and the corpus is *not* refused here - the evaluation report
        says how many there are, and quoting a metric over a corpus that
        contradicts itself needs that number to be visible.
        """
        seen: Dict[str, set] = {}
        for annotation in self.annotations:
            seen.setdefault(annotation.pair_key(), set()).add(bool(annotation.linked))
        return sorted(key for key, values in seen.items() if len(values) > 1)

    def validate(self) -> None:
        for annotation in self.annotations:
            annotation.validate_annotation()

    @classmethod
    def from_csv(cls, path: str) -> "LinkCorpus":
        """
        Load link annotations from a CSV.

        Accepts a full path or a path relative to the repository root.  This
        function has never been run on real data: no link-annotation file exists
        in this project, by design.  Shipping a loader for a dataset nobody has
        labelled is the point - the annotation round should not have to invent a
        format on top of an already-built linker.
        """
        resolved = path
        if not os.path.exists(resolved):
            candidate = os.path.join("..", path)
            if os.path.exists(candidate):
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"link annotation CSV not found: {path!r}. Expected columns: "
                    f"{', '.join(LINK_CSV_COLUMNS)} (one linked value per row). "
                    f"Real link annotations would live at {DEFAULT_LINK_CSV!r} and "
                    "are produced by a human annotation round; this project ships "
                    "the schema and the loader, never a fabricated dataset."
                )

        rows: List[LinkAnnotation] = []
        with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"link annotation CSV {path!r} is empty (no header row)")
            missing = set(LINK_CSV_COLUMNS[:4]) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"link annotation CSV {path!r} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if all((value or "").strip() == "" for value in row.values() if value):
                    continue
                try:
                    rows.append(LinkAnnotation.from_csv_row(row))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(annotations=rows)

    def to_csv(self, path: str) -> str:
        """Write the corpus in the documented format.  Never called on real data."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LINK_CSV_COLUMNS)
            writer.writeheader()
            for annotation in self.annotations:
                writer.writerow(annotation.to_csv_row())
        return path

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> "LinkCorpus":
        return cls(annotations=[LinkAnnotation(**record) for record in records])
