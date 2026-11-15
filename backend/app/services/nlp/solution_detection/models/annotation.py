"""
Annotation model and corpus loader for solution detection.

``SolutionAnnotation`` is the hand-off format between human annotators and the
model.  Unlike the response-classification stage, the target is **not** a
taxonomy: it is a single boolean, ``is_solution``, and it comes from the data
rather than from a project-assigned list of roles.

What is and is not learned
--------------------------
``is_solution`` is the only target.  ``solution_type`` is **free-form annotation
data** - there is deliberately no module-level list of categories, because a
fixed list would encode one community's vocabulary into a general model.  It is
validated for shape, reported for coverage, and reserved for the future
solution-extraction stage::

    document_id,text,is_solution,solution_type
    ann-001,"apply for financial aid",true,financial_aid
    ann-002,"you should contact the placement office",true,placement_preparation
    ann-003,"same bro",false,
    ann-004,"try applying to the scholarship before the deadline",true,scholarship

Note the second row.  ``advice`` and ``solution`` are both ``is_solution=true``
when they contain something actionable, and the response role is a separate
annotation column - the two stages never collapse into one another.

A blank ``is_solution`` cell is refused, never read as ``False``.  A missing
annotation silently becoming a negative example would teach the detector to
ignore solutions, which is the most damaging possible error in this package.

``text`` is expected to be ALREADY preprocessed (the pipeline's
``normalized_text``) so the model never re-runs preprocessing.  Real annotations
are not fabricated here: this package ships the schema and the loader, and
``data/annotations/solution_annotations.csv`` is created by the annotation
workflow.
"""

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    distinct_solution_types,
    normalise_solution_type,
    parse_bool_cell,
    solution_type_counts,
    target_label,
    validate_solution_type,
)

#: Canonical column order.  Extra columns in a user's CSV are ignored; missing
#: optional columns fall back to documented defaults.
CSV_COLUMNS = [
    "document_id",
    "text",
    "is_solution",
    "solution_type",
    "post_id",
    "comment_id",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
    "response_type",
    "rationale",
    "evidence",
]

#: Relative path used by the project for real annotation data.
DEFAULT_ANNOTATION_CSV = os.path.join("data", "annotations", "solution_annotations.csv")


def _split_multi(value: Optional[str]) -> List[str]:
    """Parse a pipe-separated cell into a clean list of non-empty values."""
    if not value:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _join_multi(values: Sequence[str]) -> str:
    return "|".join(values)


class SolutionAnnotation(BaseModel):
    """
    One manually annotated solution-detection example.

    ``is_solution`` is required and must be a real boolean.  Pydantic's own
    boolean parsing accepts ``"true"``/``"false"``/``"yes"``/``"no"``/``"1"``/``"0"``
    and rejects anything else, so a typo becomes a validation error rather than a
    silent ``False``.

    ``solution_type`` is optional and free-form.  Two combinations are
    contradictory and are reported as corpus-level warnings rather than refused,
    because both are legitimate states of an annotation in progress:

    - ``is_solution=True`` with no ``solution_type`` - a real solution whose
      category the annotator has not settled yet.  Useful to know before quoting
      category coverage.
    - ``is_solution=False`` with a ``solution_type`` - an annotator noting the
      subject matter of a comment that is not a solution ("about financial aid,
      but only complaining").  A signal to review the guideline, not an error.
    """

    document_id: str
    text: str = ""
    is_solution: bool

    #: Free-form category.  Never a fixed list, never learned in V1.
    solution_type: Optional[str] = None
    #: Why the annotator decided as they did - the most useful field for
    #: re-adjudicating disagreements.
    rationale: Optional[str] = None
    #: A quoted span or link supporting the decision, if the annotator has one.
    evidence: Optional[str] = None

    # Source traceability
    post_id: Optional[str] = None
    comment_id: Optional[str] = None

    # Language-analysis metadata
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    # Optional upstream context
    topic_id: Optional[int] = None
    issue_labels: List[str] = Field(default_factory=list)
    #: Optional response role, recorded for triage.  A separate column, so the
    #: role and the boolean can be cross-tabulated without one implying the
    #: other.
    response_type: Optional[str] = None

    def normalised_solution_type(self) -> Optional[str]:
        """The category with whitespace removed, or ``None`` when absent."""
        return normalise_solution_type(self.solution_type)

    def validate_annotation(self) -> bool:
        """
        Re-check the target and the free-form field; raise on a real problem.

        Returns the validated boolean.  The ``is_solution`` field is already
        strictly typed by Pydantic, so this exists mainly to validate
        ``solution_type``'s shape and to give one place for the training path to
        call.

        Deliberately **read-only**.  ``train()`` validates every annotation it is
        handed, so a validator that wrote back would quietly rewrite the
        caller's data mid-run - the training path must never be the thing that
        edits the annotation file's contents.  Use
        :meth:`normalised_solution_type` for the trimmed value.
        """
        validate_solution_type(self.solution_type, self.document_id)
        if not self.text or not self.text.strip():
            raise ValueError(
                f"annotation {self.document_id!r} has no text. Solution detection "
                "reads the comment body, so a row with only metadata cannot be "
                "a training example."
            )
        return bool(self.is_solution)

    def to_solution_document(self):
        """Project this annotation onto a ``SolutionDocument`` for inference."""
        from app.services.nlp.solution_detection.base import SolutionDocument

        return SolutionDocument(
            document_id=self.document_id,
            text=self.text,
            post_id=self.post_id,
            comment_id=self.comment_id,
            original_text=self.text,
            language=self.language,
            script=self.script,
            is_code_mixed=self.is_code_mixed,
            detected_languages=list(self.detected_languages),
            topic_id=self.topic_id,
            issue_labels=list(self.issue_labels),
            response_type=self.response_type,
        )

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "SolutionAnnotation":
        """Build one annotation from a CSV row.  Unknown columns are ignored."""
        if row.get("document_id") in (None, ""):
            raise ValueError("annotation row is missing the required 'document_id'")

        document_id = str(row["document_id"]).strip()
        # Refuses blank / unrecognised rather than defaulting to False.
        is_solution = parse_bool_cell(row.get("is_solution"), document_id)

        topic_id = row.get("topic_id")
        if topic_id in (None, ""):
            topic_value: Optional[int] = None
        else:
            try:
                topic_value = int(topic_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"annotation {document_id!r} has a non-integer topic_id "
                    f"{topic_id!r}"
                ) from exc

        annotation = cls(
            document_id=document_id,
            text=str(row.get("text") or ""),
            is_solution=is_solution,
            solution_type=validate_solution_type(
                row.get("solution_type"), document_id
            ),
            post_id=(str(row["post_id"]).strip() or None) if row.get("post_id") else None,
            comment_id=(
                str(row["comment_id"]).strip() or None
            ) if row.get("comment_id") else None,
            language=(str(row.get("language") or "unknown").strip() or "unknown"),
            script=(str(row.get("script") or "unknown").strip() or "unknown"),
            is_code_mixed=str(row.get("is_code_mixed", "")).strip().lower()
            in {"1", "true", "yes", "y", "t"},
            detected_languages=_split_multi(row.get("detected_languages")),
            topic_id=topic_value,
            response_type=(
                str(row["response_type"]).strip() or None
            ) if row.get("response_type") else None,
            rationale=(
                str(row["rationale"]).strip() or None
            ) if row.get("rationale") else None,
            evidence=(
                str(row["evidence"]).strip() or None
            ) if row.get("evidence") else None,
        )
        annotation.validate_annotation()
        return annotation

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "is_solution": "true" if self.is_solution else "false",
            "solution_type": self.normalised_solution_type() or "",
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": _join_multi(self.detected_languages),
            "response_type": self.response_type or "",
            "rationale": self.rationale or "",
            "evidence": self.evidence or "",
        }


class SolutionCorpus(BaseModel):
    """
    A collection of :class:`SolutionAnnotation` records.

    Duplicate ``document_id`` values are never silently de-duplicated - dropping
    rows during annotation is a data-loss bug.  They are split into two reported
    groups because they mean different things:

    - :meth:`conflicting_document_ids` - the same comment labelled both
      ``is_solution=true`` and ``is_solution=false``.  The corpus contradicts
      itself and ``train()`` refuses it; there is no defensible way to pick a
      winner automatically.
    - :meth:`duplicate_document_ids` - the same comment labelled the same way
      twice.  Harmless but worth knowing, since it silently up-weights that
      comment in training.
    """

    annotations: List[SolutionAnnotation] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    @property
    def document_ids(self) -> List[str]:
        return [a.document_id for a in self.annotations]

    @property
    def is_solutions(self) -> List[bool]:
        """The training targets, in annotation order."""
        return [bool(a.is_solution) for a in self.annotations]

    def duplicate_document_ids(self) -> List[str]:
        """Document ids appearing more than once, in first-seen order."""
        seen = set()
        duplicates: List[str] = []
        for annotation in self.annotations:
            if annotation.document_id in seen and annotation.document_id not in duplicates:
                duplicates.append(annotation.document_id)
            seen.add(annotation.document_id)
        return duplicates

    def conflicting_document_ids(self) -> List[str]:
        """
        Document ids annotated with both ``is_solution=True`` and ``False``.

        A genuine annotation error: the same comment cannot both provide and not
        provide a solution, and silently picking one would make the training
        signal depend on row order.
        """
        seen: Dict[str, set] = {}
        for annotation in self.annotations:
            seen.setdefault(annotation.document_id, set()).add(bool(annotation.is_solution))
        return sorted(key for key, values in seen.items() if len(values) > 1)

    def class_counts(self) -> Dict[str, int]:
        """
        Annotations per target, in canonical order (``not_solution``, ``solution``).

        Both keys are always present so a consumer can index them without a
        ``.get(..., 0)`` guard; a zero is informative.
        """
        counts = {LABEL_NOT_SOLUTION: 0, LABEL_SOLUTION: 0}
        for annotation in self.annotations:
            counts[target_label(bool(annotation.is_solution))] += 1
        return counts

    @property
    def labels(self) -> List[str]:
        """Distinct targets present, in canonical order for determinism."""
        present = {target_label(bool(a.is_solution)) for a in self.annotations}
        return [
            name
            for name in (LABEL_NOT_SOLUTION, LABEL_SOLUTION)
            if name in present
        ]

    @property
    def missing_targets(self) -> List[str]:
        """Canonical targets with no annotated example."""
        present = set(self.labels)
        return [
            name
            for name in (LABEL_NOT_SOLUTION, LABEL_SOLUTION)
            if name not in present
        ]

    def observed_solution_types(self) -> List[str]:
        """
        The free-form ``solution_type`` values actually present, sorted.

        Reported, never learned.  Sorted rather than first-seen so the training
        report is byte-identical across runs regardless of CSV row order.
        """
        return distinct_solution_types(
            [a.solution_type for a in self.annotations]
        )

    def solution_type_counts(self) -> Dict[str, int]:
        """How many annotations carry each ``solution_type`` (absent excluded)."""
        return solution_type_counts([a.solution_type for a in self.annotations])

    def positives_without_type(self) -> List[str]:
        """
        ``is_solution=True`` rows with no ``solution_type``.

        Not an error - a real solution whose category is undecided - but worth
        knowing before quoting category coverage, so it is reported as a warning.
        """
        return [
            a.document_id
            for a in self.annotations
            if a.is_solution and not a.normalised_solution_type()
        ]

    def negatives_with_type(self) -> List[str]:
        """
        ``is_solution=False`` rows that nonetheless carry a ``solution_type``.

        Legitimate while annotating (the annotator may be recording the subject
        matter of a comment that is not a solution), but a signal that the
        guideline needs clarifying, so it is reported.
        """
        return [
            a.document_id
            for a in self.annotations
            if not a.is_solution and a.normalised_solution_type()
        ]

    def validate(self) -> None:
        """Validate every annotation; raises ``ValueError`` on the first problem."""
        for annotation in self.annotations:
            annotation.validate_annotation()

    def validate_unique_document_ids(self) -> None:
        """
        Refuse a corpus where one document carries two different targets.

        Repeated rows with the *same* target are left alone (reported by
        :meth:`duplicate_document_ids` instead).
        """
        conflicts = self.conflicting_document_ids()
        if conflicts:
            detail = ", ".join(conflicts)
            raise ValueError(
                f"annotation contract violated: document_id(s) {detail} appear more "
                "than once with different is_solution values. Solution detection is "
                "binary, so one comment has exactly one answer; resolve the "
                "duplicate rows in the annotation file (or give the rows distinct "
                "document_ids) before training."
            )

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "SolutionCorpus":
        """
        Load annotations from a CSV file.

        Accepts either a full path or a path relative to the repository root, so
        ``from_csv("data/annotations/solution_annotations.csv")`` works from
        either the repository root or ``backend/``.  Missing optional columns
        default; unknown extra columns are ignored.
        """
        resolved = path
        if not os.path.exists(resolved):
            candidate = os.path.join("..", path)
            if os.path.exists(candidate):
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"annotation CSV not found: {path!r}. Expected columns: "
                    f"{', '.join(CSV_COLUMNS)} (one is_solution value per row). "
                    f"Real annotations live at {DEFAULT_ANNOTATION_CSV!r} relative "
                    "to the repository root, and are created by the human "
                    "annotation workflow - this package does not ship any."
                )

        rows: List[SolutionAnnotation] = []
        with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"annotation CSV {path!r} is empty (no header row)")
            missing = {"document_id", "text", "is_solution"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"annotation CSV {path!r} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if all((value or "").strip() == "" for value in row.values() if value):
                    continue  # skip fully blank lines
                try:
                    rows.append(SolutionAnnotation.from_csv_row(row))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(annotations=rows)

    def to_csv(self, path: str) -> str:
        """Write the corpus to a CSV in the documented format."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for annotation in self.annotations:
                writer.writerow(annotation.to_csv_row())
        return path

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "SolutionCorpus":
        return cls(annotations=[SolutionAnnotation(**record) for record in records])
