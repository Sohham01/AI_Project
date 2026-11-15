"""
Annotation models for sentiment and emotion.

Both label sets are defined by the people annotating the corpus, not by this
code.  ``SentimentAnnotation`` and ``EmotionAnnotation`` are the hand-off
format between human annotators and the models, and the vocabulary of the
``sentiment`` / ``emotions`` fields is free-form: the class set is whatever
the annotation guideline says, and it is allowed to change between rounds.

**Sentiment** is single-label.  Every annotated document carries exactly one
overall polarity, so a blank value is an error rather than "no opinion" — a
document with no opinion is annotated with whatever the project's taxonomy
calls that (conventionally ``neutral``).  The code does not impose that name.

**Emotion** is multi-label, because a text can express several feelings at
once, and may also carry *no* emotion at all:

    "placement pressure hai aur hostel ka stress bhi hai"
        -> ["stress", "frustration"]
    "I am nervous but also grateful for the support"
        -> ["anxiety", "gratitude"]
    "the mess serves rice and dal"
        -> []                       # no emotion; valuable negative supervision

No final emotion taxonomy is treated as domain truth anywhere in this package.
Sentiment and emotion are annotated independently, which is what makes it
impossible for a negative polarity to imply an emotional state:

    sentiment = negative, emotion = frustration     # supported by annotation
    sentiment = negative, emotion = disappointment  # a different feeling
    sentiment = negative, emotion = (none)         # complaint, no emotion

CSV formats (see docs/sentiment_emotion.md for the full specification)::

    document_id,text,sentiment,...
    ann-001,"this placement season is amazing",positive,...

    document_id,text,emotions,...
    ann-001,"placement pressure hai aur hostel ka stress bhi hai",stress|frustration,...

``emotions`` uses a pipe separator so a multi-value cell never collides with
CSV comma separation.  ``text`` is expected to be ALREADY preprocessed (the
pipeline's ``normalized_text``) so neither model re-runs preprocessing.
"""

import os
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from app.services.nlp.sentiment_emotion.models._csv import (
    MULTI_VALUE_SEPARATOR,
    join_multi,
    parse_bool,
    parse_optional_int,
    read_annotation_rows,
    split_multi,
    write_annotation_rows,
)

#: Canonical column order for the sentiment annotation file.
SENTIMENT_CSV_COLUMNS = [
    "document_id",
    "text",
    "sentiment",
    "post_id",
    "comment_id",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

#: Canonical column order for the emotion annotation file.
EMOTION_CSV_COLUMNS = [
    "document_id",
    "text",
    "emotions",
    "post_id",
    "comment_id",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

#: Relative paths used by the project for real annotation data.
DEFAULT_SENTIMENT_CSV = os.path.join("data", "annotations", "sentiment_annotations.csv")
DEFAULT_EMOTION_CSV = os.path.join("data", "annotations", "emotion_annotations.csv")


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


class SentimentAnnotation(BaseModel):
    """
    One manually annotated sentiment example.

    ``sentiment`` is a single free-form class name.  It is required and must
    be non-blank, because sentiment is mutually exclusive: a document has one
    overall polarity.  Nothing here restricts the value to a built-in list.
    """

    document_id: str
    text: str = ""
    sentiment: str

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

    def normalised_sentiment(self) -> str:
        """The class with surrounding whitespace removed, so spacing never leaks."""
        return self.sentiment.strip()

    def validate_sentiment(self) -> None:
        """Raise ValueError when the class is missing or unusable."""
        if not self.sentiment or not self.sentiment.strip():
            raise ValueError(
                f"annotation {self.document_id!r} has no sentiment label. "
                "Sentiment is single-label, so every annotated document needs "
                "exactly one class; use whatever the annotation guideline "
                "defines for 'no opinion expressed' rather than leaving it blank."
            )
        if MULTI_VALUE_SEPARATOR in self.sentiment:
            raise ValueError(
                f"annotation {self.document_id!r} has sentiment "
                f"{self.sentiment!r} containing {MULTI_VALUE_SEPARATOR!r}; "
                "sentiment is single-label, so use one class per document and "
                "annotate emotions separately in the emotion file"
            )

    def to_document(self):
        """Project this annotation onto a ``SentimentEmotionDocument``."""
        from app.services.nlp.sentiment_emotion.base import SentimentEmotionDocument

        return SentimentEmotionDocument(
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
    def from_csv_row(cls, row: Dict[str, Any]) -> "SentimentAnnotation":
        if row.get("document_id") in (None, ""):
            raise ValueError("annotation row is missing the required 'document_id'")
        if row.get("sentiment") in (None, ""):
            raise ValueError("annotation row is missing the required 'sentiment'")
        return cls(
            document_id=str(row["document_id"]).strip(),
            text=str(row.get("text") or ""),
            sentiment=str(row["sentiment"]).strip(),
            post_id=(str(row["post_id"]).strip() or None) if row.get("post_id") else None,
            comment_id=(
                str(row["comment_id"]).strip() or None
            ) if row.get("comment_id") else None,
            language=(str(row.get("language") or "unknown").strip() or "unknown"),
            script=(str(row.get("script") or "unknown").strip() or "unknown"),
            is_code_mixed=parse_bool(row.get("is_code_mixed")),
            detected_languages=split_multi(row.get("detected_languages")),
            topic_id=parse_optional_int(
                row.get("topic_id"), "topic_id", str(row["document_id"]).strip()
            ),
        )

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "sentiment": self.normalised_sentiment(),
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": join_multi(self.detected_languages),
        }


class SentimentCorpus(BaseModel):
    """
    A collection of :class:`SentimentAnnotation` records.

    Duplicate ``document_id`` values are preserved rather than silently
    de-duplicated, and reported through :meth:`duplicate_document_ids` so a
    caller can decide.  Dropping rows during annotation is a data-loss bug.
    """

    annotations: List[SentimentAnnotation] = Field(default_factory=list)

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

    def class_counts(self) -> Dict[str, int]:
        """Number of annotated documents per class (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.normalised_sentiment()] = (
                counts.get(annotation.normalised_sentiment(), 0) + 1
            )
        return dict(sorted(counts.items()))

    @property
    def labels(self) -> List[str]:
        """Every distinct class in the corpus, sorted for determinism."""
        return sorted({a.normalised_sentiment() for a in self.annotations})

    def validate(self) -> None:
        """Validate every annotation; raises ValueError on the first problem."""
        for annotation in self.annotations:
            annotation.validate_sentiment()

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "SentimentCorpus":
        """Load sentiment annotations from a CSV (full or repo-relative path)."""
        rows: List[SentimentAnnotation] = []
        for line_number, row in read_annotation_rows(
            path,
            required_columns=("document_id", "text", "sentiment"),
            column_hint=(
                f"{', '.join(SENTIMENT_CSV_COLUMNS)} (one sentiment class per row)"
            ),
        ):
            try:
                rows.append(SentimentAnnotation.from_csv_row(row))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(annotations=rows)

    def to_csv(self, path: str) -> str:
        """Write the corpus to a CSV in the documented format."""
        return write_annotation_rows(
            path,
            SENTIMENT_CSV_COLUMNS,
            (annotation.to_csv_row() for annotation in self.annotations),
        )

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "SentimentCorpus":
        return cls(annotations=[SentimentAnnotation(**record) for record in records])


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------


class EmotionAnnotation(BaseModel):
    """
    One manually annotated emotion example.

    ``emotions`` may hold any number of labels, including none.  A document
    with no listed emotion is valid and is valuable negative supervision for
    every label.  Label strings are free-form and language-neutral.
    """

    document_id: str
    text: str = ""
    emotions: List[str] = Field(default_factory=list)

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

    @property
    def is_multi_label(self) -> bool:
        return len(self.emotions) > 1

    def normalised_emotions(self) -> List[str]:
        """Labels de-duplicated and sorted, so ordering never affects training."""
        return sorted({label.strip() for label in self.emotions if label.strip()})

    def validate_emotions(self) -> None:
        """Raise ValueError if an emotion label is unusable (blank, or has separators)."""
        for label in self.emotions:
            stripped = label.strip()
            if not stripped:
                raise ValueError(
                    f"annotation {self.document_id!r} contains an empty emotion label"
                )
            if MULTI_VALUE_SEPARATOR in stripped:
                raise ValueError(
                    f"annotation {self.document_id!r} has an emotion label "
                    f"containing {MULTI_VALUE_SEPARATOR!r} ({stripped!r}); "
                    "multi-label cells must be split on that separator"
                )

    def to_document(self):
        """Project this annotation onto a ``SentimentEmotionDocument``."""
        from app.services.nlp.sentiment_emotion.base import SentimentEmotionDocument

        return SentimentEmotionDocument(
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
    def from_csv_row(cls, row: Dict[str, Any]) -> "EmotionAnnotation":
        if row.get("document_id") in (None, ""):
            raise ValueError("annotation row is missing the required 'document_id'")
        return cls(
            document_id=str(row["document_id"]).strip(),
            text=str(row.get("text") or ""),
            emotions=split_multi(row.get("emotions")),
            post_id=(str(row["post_id"]).strip() or None) if row.get("post_id") else None,
            comment_id=(
                str(row["comment_id"]).strip() or None
            ) if row.get("comment_id") else None,
            language=(str(row.get("language") or "unknown").strip() or "unknown"),
            script=(str(row.get("script") or "unknown").strip() or "unknown"),
            is_code_mixed=parse_bool(row.get("is_code_mixed")),
            detected_languages=split_multi(row.get("detected_languages")),
            topic_id=parse_optional_int(
                row.get("topic_id"), "topic_id", str(row["document_id"]).strip()
            ),
        )

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "emotions": join_multi(self.emotions),
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": join_multi(self.detected_languages),
        }


class EmotionCorpus(BaseModel):
    """
    A collection of :class:`EmotionAnnotation` records.

    Multi-label by construction, so :attr:`n_documents_without_emotions` and
    :attr:`n_multi_label_documents` are both first-class counts: documents with
    no emotion and documents with several are normal, not edge cases.
    """

    annotations: List[EmotionAnnotation] = Field(default_factory=list)

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
        """Number of annotated documents per emotion (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for annotation in self.annotations:
            for label in set(annotation.emotions):
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def labels(self) -> List[str]:
        """Every distinct emotion in the corpus, sorted for determinism."""
        return sorted({label for a in self.annotations for label in a.emotions})

    @property
    def n_documents_without_emotions(self) -> int:
        return sum(1 for a in self.annotations if not a.normalised_emotions())

    @property
    def n_multi_label_documents(self) -> int:
        return sum(1 for a in self.annotations if len(a.normalised_emotions()) > 1)

    def validate(self) -> None:
        """Validate every annotation; raises ValueError on the first problem."""
        for annotation in self.annotations:
            annotation.validate_emotions()

    # -- CSV interop ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "EmotionCorpus":
        """Load emotion annotations from a CSV (full or repo-relative path)."""
        rows: List[EmotionAnnotation] = []
        for line_number, row in read_annotation_rows(
            path,
            required_columns=("document_id", "text", "emotions"),
            column_hint=(
                f"{', '.join(EMOTION_CSV_COLUMNS)} (emotions separated by "
                f"{MULTI_VALUE_SEPARATOR!r})"
            ),
        ):
            try:
                rows.append(EmotionAnnotation.from_csv_row(row))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(annotations=rows)

    def to_csv(self, path: str) -> str:
        """Write the corpus to a CSV in the documented format."""
        return write_annotation_rows(
            path,
            EMOTION_CSV_COLUMNS,
            (annotation.to_csv_row() for annotation in self.annotations),
        )

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "EmotionCorpus":
        return cls(annotations=[EmotionAnnotation(**record) for record in records])


__all__ = [
    "MULTI_VALUE_SEPARATOR",
    "SENTIMENT_CSV_COLUMNS",
    "EMOTION_CSV_COLUMNS",
    "DEFAULT_SENTIMENT_CSV",
    "DEFAULT_EMOTION_CSV",
    "SentimentAnnotation",
    "SentimentCorpus",
    "EmotionAnnotation",
    "EmotionCorpus",
]
