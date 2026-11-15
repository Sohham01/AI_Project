"""
Annotation model and corpus loader for response classification.

``ResponseAnnotation`` is the hand-off format between human annotators and the
model.  Unlike the upstream stages, the label space is NOT free-form: the six
response types are defined by the project assignment, so a value outside them is
a labelling error and :meth:`ResponseAnnotation.validate_response_type` refuses
it rather than letting a seventh class into the model.

Single-label by design.  A comment plays one *primary* role:

    "same bro"                                -> "agreement"
    "I faced this last year"                  -> "personal_experience"
    "you should contact the placement office" -> "advice"
    "apply for the scholarship"               -> "solution"

This is the annotation contract that makes duplicate ``document_id`` handling
decidable (see :meth:`ResponseCorpus.conflicting_document_ids`): the same
comment annotated twice with two *different* roles is a self-contradictory
corpus, and is refused at training time rather than averaged away.

CSV format (see docs/response_classification.md for the full specification)::

    document_id,text,response_type,post_id,comment_id,language,script,is_code_mixed,detected_languages
    ann-001,"same bro",agreement,post-1,c-101,en,latin,false,en
    ann-002,"apply for the scholarship",solution,post-1,c-102,en,latin,false,en

``text`` is expected to be ALREADY preprocessed (the pipeline's
``normalized_text``) so the model never re-runs preprocessing.  Real Reddit
annotations are not fabricated here: this package ships the schema and the
loader, and ``data/annotations/response_annotations.csv`` is created by the
annotation workflow.
"""

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES,
    is_valid_response_type,
    validate_response_type,
)

#: Canonical column order.  Extra columns in a user's CSV are ignored; missing
#: optional columns fall back to documented defaults.
CSV_COLUMNS = [
    "document_id",
    "text",
    "response_type",
    "post_id",
    "comment_id",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

#: Relative path used by the project for real annotation data.
DEFAULT_ANNOTATION_CSV = os.path.join("data", "annotations", "response_annotations.csv")

_TRUTHY = {"1", "true", "yes", "y", "t"}


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Interpret the usual truthy spellings used in annotation CSVs."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in _TRUTHY


def _split_multi(value: Optional[str]) -> List[str]:
    """Parse a pipe-separated cell into a clean list of non-empty values."""
    if not value:
        return []
    return [
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    ]


def _join_multi(values: Sequence[str]) -> str:
    return "|".join(values)


class ResponseAnnotation(BaseModel):
    """
    One manually annotated response-role example.

    ``response_type`` must be one of the six V1 response types.  It is required
    and must be non-blank: response classification is single-label, so there is
    no "no role" value to fall back on.  A comment whose role is genuinely
    unclear is annotated as ``opinion`` by whatever the project's annotation
    guideline says - the decision belongs in the guideline, not in this code.
    """

    document_id: str
    text: str = ""
    response_type: str

    # Source traceability
    post_id: Optional[str] = None
    comment_id: Optional[str] = None

    # Language-analysis metadata
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    # Optional upstream context, when the annotator or pipeline had it
    topic_id: Optional[int] = None

    def normalised_response_type(self) -> str:
        """The class with surrounding whitespace removed, so spacing never leaks."""
        return self.response_type.strip()

    def validate_response_type(self) -> str:
        """
        Raise ValueError when the class is missing or outside the V1 taxonomy.

        Returns the normalised class so callers can use the checked value.
        """
        return validate_response_type(self.response_type, self.document_id)

    def to_response_document(self):
        """Project this annotation onto a ``ResponseDocument`` for inference."""
        from app.services.nlp.response_classification.base import ResponseDocument

        return ResponseDocument(
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
        )

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "ResponseAnnotation":
        """Build one annotation from a CSV row.  Unknown columns are ignored."""
        if row.get("document_id") in (None, ""):
            raise ValueError("annotation row is missing the required 'document_id'")
        if row.get("response_type") in (None, ""):
            raise ValueError("annotation row is missing the required 'response_type'")

        document_id = str(row["document_id"]).strip()
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

        return cls(
            document_id=document_id,
            text=str(row.get("text") or ""),
            response_type=str(row["response_type"]).strip(),
            post_id=(str(row["post_id"]).strip() or None) if row.get("post_id") else None,
            comment_id=(
                str(row["comment_id"]).strip() or None
            ) if row.get("comment_id") else None,
            language=(str(row.get("language") or "unknown").strip() or "unknown"),
            script=(str(row.get("script") or "unknown").strip() or "unknown"),
            is_code_mixed=_parse_bool(row.get("is_code_mixed")),
            detected_languages=_split_multi(row.get("detected_languages")),
            topic_id=topic_value,
        )

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "response_type": self.normalised_response_type(),
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": _join_multi(self.detected_languages),
        }


class ResponseCorpus(BaseModel):
    """
    A collection of :class:`ResponseAnnotation` records.

    Duplicate ``document_id`` values are never silently de-duplicated - dropping
    rows during annotation is a data-loss bug.  They are separated into two
    reported groups instead, because they mean different things:

    - :meth:`conflicting_document_ids` - the same comment labelled with two
      *different* roles.  The corpus contradicts itself, and ``train()`` refuses
      it.  There is no defensible way to pick a winner automatically.
    - :meth:`duplicate_document_ids` - the same comment labelled the same way
      twice.  Harmless but worth knowing, since it silently up-weights that
      comment in training.
    """

    annotations: List[ResponseAnnotation] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    @property
    def document_ids(self) -> List[str]:
        return [a.document_id for a in self.annotations]

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
        Document ids annotated with two or more *different* response types.

        This is a genuine annotation error rather than a duplicate row: one
        comment cannot play two primary roles, and silently picking one of them
        would make the training signal depend on row order.
        """
        seen: Dict[str, set] = {}
        for annotation in self.annotations:
            seen.setdefault(annotation.document_id, set()).add(
                annotation.normalised_response_type()
            )
        return sorted(key for key, values in seen.items() if len(values) > 1)

    def class_counts(self) -> Dict[str, int]:
        """
        Number of annotated documents per class, in canonical V1 order.

        Canonical rather than alphabetical so the counts line up with the
        probability vector and the confusion matrix; any label outside the
        taxonomy is appended in sorted order rather than dropped.
        """
        counts: Dict[str, int] = {}
        for annotation in self.annotations:
            key = annotation.normalised_response_type()
            counts[key] = counts.get(key, 0) + 1
        present = [name for name in RESPONSE_TYPES if name in counts]
        present.extend(sorted(set(counts) - set(RESPONSE_TYPES)))
        return {name: counts[name] for name in present}

    @property
    def labels(self) -> List[str]:
        """Distinct classes present, in canonical V1 order for determinism."""
        present = {a.normalised_response_type() for a in self.annotations}
        return [name for name in RESPONSE_TYPES if name in present] + sorted(
            present - set(RESPONSE_TYPES)
        )

    def missing_response_types(self) -> List[str]:
        """V1 response types with no annotated example, in canonical order."""
        present = set(self.labels)
        return [name for name in RESPONSE_TYPES if name not in present]

    def validate(self) -> None:
        """Validate every annotation; raises ValueError on the first problem."""
        for annotation in self.annotations:
            annotation.validate_response_type()

    def validate_unique_document_ids(self) -> None:
        """
        Refuse a corpus where one document carries two different roles.

        Repeated rows with the *same* label are left alone (they are reported by
        :meth:`duplicate_document_ids` instead), because the annotation contract
        here is one primary role per comment.
        """
        conflicts = self.conflicting_document_ids()
        if conflicts:
            detail = ", ".join(conflicts)
            raise ValueError(
                f"annotation contract violated: document_id(s) {detail} appear more "
                "than once with different response types. Response classification is "
                "single-label, so one comment has exactly one primary role. Resolve "
                "the duplicate rows in the annotation file (or give the rows distinct "
                "document_ids) before training."
            )

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "ResponseCorpus":
        """
        Load annotations from a CSV file.

        Accepts either a full path or a path relative to the repository root, so
        ``from_csv("data/annotations/response_annotations.csv")`` just works from
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
                    f"{', '.join(CSV_COLUMNS)} (one response type per row)"
                )

        rows: List[ResponseAnnotation] = []
        with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"annotation CSV {path!r} is empty (no header row)")
            missing = {"document_id", "text", "response_type"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"annotation CSV {path!r} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if all((value or "").strip() == "" for value in row.values() if value):
                    continue  # skip fully blank lines
                try:
                    rows.append(ResponseAnnotation.from_csv_row(row))
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
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "ResponseCorpus":
        return cls(annotations=[ResponseAnnotation(**record) for record in records])


def is_valid_class(value: Optional[str]) -> bool:
    """Re-exported for convenience: is ``value`` one of the six V1 types?"""
    return is_valid_response_type(value)
