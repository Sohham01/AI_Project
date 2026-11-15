"""
Annotation model for manually labelled issue examples.

The issue label set is *not* defined in code — it is defined by the people
annotating the corpus.  ``IssueAnnotation`` is the hand-off format between
human annotators and the model.

Multi-label by design: a document may carry zero, one, or many issues.

    "fees are way too high and ROI feels questionable"  -> ["fees", "roi"]
    "placement season is extremely stressful"            -> ["placement_stress"]
    "hostel rooms are too small and mess food is bad"    -> ["hostel", "mess_food"]

CSV format (see docs/issue_identification.md for the full specification)::

    document_id,text,labels,language,script,is_code_mixed,detected_languages
    ann-001,"fees are way too high and ROI feels questionable",fees|roi,en,latin,false,en
    ann-002,"hostel rooms are too small and mess food is bad",hostel|mess_food,en,latin,false,en

``labels`` uses a pipe separator so that a multi-label cell never collides
with CSV comma separation.  ``text`` is expected to be ALREADY preprocessed
(the pipeline's ``normalized_text``) so the model never re-runs preprocessing.
"""

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

#: Separator used inside CSV cells for multi-valued fields.
MULTI_VALUE_SEPARATOR = "|"

#: Canonical column order.  Extra columns in a user's CSV are ignored; missing
#: optional columns fall back to documented defaults.
CSV_COLUMNS = [
    "document_id",
    "text",
    "labels",
    "post_id",
    "comment_id",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

#: Relative path used by the project for real annotation data.
DEFAULT_ANNOTATION_CSV = os.path.join("data", "annotations", "issue_annotations.csv")


def _split_multi(value: Optional[str]) -> List[str]:
    """Parse a pipe-separated cell into a clean list of non-empty values."""
    if not value:
        return []
    return [part.strip() for part in value.split(MULTI_VALUE_SEPARATOR) if part.strip()]


def _join_multi(values: Sequence[str]) -> str:
    return MULTI_VALUE_SEPARATOR.join(values)


class IssueAnnotation(BaseModel):
    """
    One manually annotated issue example.

    ``labels`` may contain any number of entries, including none.  Label
    strings are free-form and language-neutral: nothing here assumes the
    taxonomy is English-only, or even that it is IIM-specific.
    """

    document_id: str
    text: str = ""
    labels: List[str] = Field(default_factory=list)

    # Source traceability
    post_id: Optional[str] = None
    comment_id: Optional[str] = None

    # Language-analysis metadata
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    # Optional topic-discovery context, when the annotator or pipeline had it
    topic_id: Optional[int] = None

    @property
    def is_multi_label(self) -> bool:
        return len(self.labels) > 1

    def normalised_labels(self) -> List[str]:
        """Labels de-duplicated and sorted, so ordering never affects training."""
        return sorted({label.strip() for label in self.labels if label.strip()})

    def validate_labels(self) -> None:
        """Raise ValueError if the label cell is unusable (blank, or separators left in)."""
        for label in self.labels:
            stripped = label.strip()
            if not stripped:
                raise ValueError(
                    f"annotation {self.document_id!r} contains an empty issue label"
                )
            if MULTI_VALUE_SEPARATOR in stripped:
                raise ValueError(
                    f"annotation {self.document_id!r} has a label containing "
                    f"{MULTI_VALUE_SEPARATOR!r} ({stripped!r}); multi-label cells "
                    f"must be split on that separator"
                )

    def to_issue_document(self):
        """Project this annotation onto an ``IssueDocument`` for inference."""
        from app.services.nlp.issue_identification.base import IssueDocument

        return IssueDocument(
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
    def from_csv_row(cls, row: Dict[str, Any]) -> "IssueAnnotation":
        """Build one annotation from a CSV row.  Unknown columns are ignored."""
        if row.get("document_id") in (None, ""):
            raise ValueError("annotation row is missing the required 'document_id'")
        topic_id = row.get("topic_id")
        if topic_id in (None, ""):
            topic_value: Optional[int] = None
        else:
            try:
                topic_value = int(topic_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"annotation {row.get('document_id')!r} has a non-integer "
                    f"topic_id {topic_id!r}"
                ) from exc

        mixed = row.get("is_code_mixed", "")
        if isinstance(mixed, bool):
            is_code_mixed = mixed
        else:
            is_code_mixed = str(mixed).strip().lower() in {"1", "true", "yes", "y"}

        return cls(
            document_id=str(row["document_id"]).strip(),
            text=str(row.get("text") or ""),
            labels=_split_multi(row.get("labels")),
            post_id=(str(row["post_id"]).strip() or None) if row.get("post_id") else None,
            comment_id=(
                str(row["comment_id"]).strip() or None
            ) if row.get("comment_id") else None,
            language=(str(row.get("language") or "unknown").strip() or "unknown"),
            script=(str(row.get("script") or "unknown").strip() or "unknown"),
            is_code_mixed=is_code_mixed,
            detected_languages=_split_multi(row.get("detected_languages")),
            topic_id=topic_value,
        )

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "labels": _join_multi(self.labels),
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": _join_multi(self.detected_languages),
        }


class AnnotationCorpus(BaseModel):
    """
    A collection of :class:`IssueAnnotation` records.

    Duplicate ``document_id`` values are preserved rather than silently
    de-duplicated, and reported through :meth:`duplicate_document_ids` so a
    caller can decide.  Dropping rows during annotation is a data-loss bug.
    """

    annotations: List[IssueAnnotation] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    @property
    def document_ids(self) -> List[str]:
        return [a.document_id for a in self.annotations]

    def duplicate_document_ids(self) -> List[str]:
        """Document ids that appear more than once, in first-seen order."""
        seen = set()
        duplicates: List[str] = []
        for annotation in self.annotations:
            if annotation.document_id in seen and annotation.document_id not in duplicates:
                duplicates.append(annotation.document_id)
            seen.add(annotation.document_id)
        return duplicates

    def label_counts(self) -> Dict[str, int]:
        """Number of annotated documents per label (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for annotation in self.annotations:
            for label in set(annotation.labels):
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def labels(self) -> List[str]:
        """Every distinct label in the corpus, sorted for determinism."""
        return sorted({label for a in self.annotations for label in a.labels})

    def validate(self) -> None:
        """Validate every annotation; raises ValueError on the first problem."""
        for annotation in self.annotations:
            annotation.validate_labels()

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "AnnotationCorpus":
        """
        Load annotations from a CSV file.

        Accepts either a full path or a path relative to the repository root,
        so ``from_csv("data/annotations/issue_annotations.csv")`` just works.
        Missing optional columns default; unknown extra columns are ignored.
        """
        resolved = path
        if not os.path.exists(resolved):
            candidate = os.path.join("..", path)
            if os.path.exists(candidate):
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"annotation CSV not found: {path!r}. Expected columns: "
                    f"{', '.join(CSV_COLUMNS)} (labels separated by "
                    f"{MULTI_VALUE_SEPARATOR!r})"
                )

        rows: List[IssueAnnotation] = []
        with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"annotation CSV {path!r} is empty (no header row)")
            missing = {"document_id", "text", "labels"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"annotation CSV {path!r} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if all((value or "").strip() == "" for value in row.values() if value):
                    continue  # skip fully blank lines
                try:
                    rows.append(IssueAnnotation.from_csv_row(row))
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
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "AnnotationCorpus":
        return cls(annotations=[IssueAnnotation(**record) for record in records])
