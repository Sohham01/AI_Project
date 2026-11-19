"""
Dataset abstraction for the ML training framework.

Provides a unified ``TrainingRecord`` model that carries text, a label, and
optional metadata, plus utilities to load records from synthetic sources or
from real annotation files when they exist.

Do NOT fabricate annotations.  This module can generate SYNTHETIC demo data
explicitly labelled as such, and can load real annotation CSVs through the
existing corpus loaders, but it never invents annotation data and presents it
as real.

Design
------
Each classifier task has its own annotation type (SentimentAnnotation,
EmotionAnnotation, IssueAnnotation, ResponseAnnotation, SolutionAnnotation)
with its own CSV schema and validation rules.  The ML framework does not
replace those loaders - it uses them.  ``TrainingRecord`` is the generic
transport the splitter and trainer use so they do not need to know which
annotation type they are handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Generic training record
# ---------------------------------------------------------------------------


@dataclass
class TrainingRecord:
    """One annotated document for training or evaluation.

    ``label`` is task-specific: a single class name for single-label tasks,
    or a pipe-joined string of labels for multi-label tasks (matching the
    convention already used by the existing annotation loaders).  The trainer
    and splitter never interpret ``label`` - they pass it through to the
    classifier's own annotation type.
    """

    document_id: str
    text: str
    label: str
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = field(default_factory=list)
    topic_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for CSV/JSON transport."""
        d: Dict[str, Any] = {
            "document_id": self.document_id,
            "text": self.text,
            "label": self.label,
            "post_id": self.post_id or "",
            "comment_id": self.comment_id or "",
            "language": self.language,
            "script": self.script,
            "is_code_mixed": "true" if self.is_code_mixed else "false",
            "detected_languages": "|".join(self.detected_languages),
            "topic_id": str(self.topic_id) if self.topic_id is not None else "",
            "metadata": dict(self.metadata),
        }
        return d

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "TrainingRecord":
        """Deserialize from a dict (e.g. a CSV row)."""
        lang = str(row.get("language") or "unknown").strip() or "unknown"
        scr = str(row.get("script") or "unknown").strip() or "unknown"
        mixed = str(row.get("is_code_mixed", "")).strip().lower() in {"1", "true", "yes", "y"}
        dl = str(row.get("detected_languages") or "").strip()
        # Support both old-style flat metadata (keys spread at top level) and
        # new-style nested "metadata" key.
        known_keys = {
            "document_id", "text", "label", "post_id", "comment_id",
            "language", "script", "is_code_mixed", "detected_languages", "topic_id",
        }
        nested_meta = row.get("metadata")
        if isinstance(nested_meta, dict):
            metadata = nested_meta
        else:
            metadata = {k: v for k, v in row.items() if k not in known_keys}
        return cls(
            document_id=str(row.get("document_id") or "").strip(),
            text=str(row.get("text") or "").strip(),
            label=str(row.get("label") or "").strip(),
            post_id=(str(row.get("post_id") or "").strip() or None),
            comment_id=(str(row.get("comment_id") or "").strip() or None),
            language=lang,
            script=scr,
            is_code_mixed=mixed,
            detected_languages=[v for v in dl.split("|") if v.strip()] if dl else [],
            topic_id=_parse_optional_int(row.get("topic_id")),
            metadata=metadata,
        )


def _parse_optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def split_label(value: str) -> List[str]:
    """Split a ``TrainingRecord.label`` into its constituent labels.

    Single-label tasks return ``[value]``.  Multi-label tasks use the pipe
    separator already established by the annotation loaders.
    """
    value = value.strip()
    if not value:
        return []
    if "|" in value:
        return [v.strip() for v in value.split("|") if v.strip()]
    return [value]


def join_label(labels: Sequence[str]) -> str:
    """Join labels back into a ``TrainingRecord.label`` string."""
    labels = [str(l).strip() for l in labels if str(l).strip()]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return "|".join(labels)


# ---------------------------------------------------------------------------
# Label set abstraction
# ---------------------------------------------------------------------------


class LabelSet(Protocol):
    """Anything that can supply the labels known to a trained model."""

    @property
    def labels(self) -> List[str]: ...


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------


def make_synthetic_records(
    n: int,
    label_options: List[str],
    single_label: bool = True,
    seed: int = 42,
) -> List[TrainingRecord]:
    """Create SYNTHETIC demo records for plumbing checks.

    These are NOT real annotations.  Any demo or test that uses them must
    clearly label them as synthetic.

    Args:
        n: number of records to generate.
        label_options: pool of labels to draw from.
        single_label: if True each record gets exactly one label; if False
            each record gets 0-3 labels joined with pipe.
        seed: random seed for reproducibility.
    """
    import random

    rng = random.Random(seed)
    records: List[TrainingRecord] = []
    for i in range(n):
        did = f"synthetic-{i:04d}"
        text = _synthetic_text(rng)
        if single_label:
            label = rng.choice(label_options)
        else:
            k = rng.randint(0, min(3, len(label_options)))
            chosen = sorted(rng.sample(label_options, k)) if k > 0 else []
            label = "|".join(chosen)
        records.append(
            TrainingRecord(
                document_id=did,
                text=f"{text} [{did}]",
                label=label,
                language=rng.choice(["en", "hi", "bn", "or", "unknown"]),
                script=rng.choice(["latin", "devanagari", "bengali", "odia", "unknown"]),
                is_code_mixed=rng.random() < 0.3,
                detected_languages=[
                    l for l in rng.sample(
                        ["en", "hi", "bn", "or"], k=rng.randint(1, 3)
                    )
                ],
            )
        )
    return records


def _synthetic_text(rng: random.Random) -> str:
    """Generate a plausible-but-fake multilingual text snippet."""
    templates = [
        "placement season is tough this year and the PPO process feels unclear",
        "bhai CAT exam ki taiyari kaise kare coaching zaroori hai ya nhi",
        "हॉस्टल का मेस खाना अच्छा नहीं है charges बहुत zyada hai",
        "hostel mess food quality is really bad and charges are too high",
        "fees aur ROI ka scene kya hai IIM Sambalpur MBA return justify karta hai",
        "आईआईएम संबलपुर में प्लेसमेंट बहुत अच्छा है",
        "ক্যাট পরীক্ষার প্রস্তুতি খুব কঠিন কিন্তু প্লেসমেন্ট ভালো হয়েছে",
        "summer internship ke liye company kaise select kare projects mast hote hai",
        "bro placement ka kya scene hai PPO mila ya nhi confusion bahut hai",
        "the fees are very high and the ROI is not worth it compared to other colleges",
    ]
    return rng.choice(templates)


# ---------------------------------------------------------------------------
# Real annotation loading
# ---------------------------------------------------------------------------


class AnnotationNotFoundError(Exception):
    """Raised when a real annotation file does not exist and no synthetic
    fallback is requested."""


def load_annotation_dataset(
    task_name: str,
    path: str,
    base_dir: Optional[str] = None,
) -> List[TrainingRecord]:
    """Load real annotations for *task_name* from *path*.

    Delegates to the existing corpus loaders in the NLP modules.  Returns a
    list of ``TrainingRecord`` objects that preserve the document identity and
    metadata from the annotation file.

    Args:
        task_name: one of ``sentiment``, ``emotion``, ``issue``, ``response``,
            ``solution``.
        path: path to the annotation CSV (relative to *base_dir* or absolute).
        base_dir: directory to resolve relative *path* against.  Defaults to
            the current working directory.

    Raises:
        AnnotationNotFoundError: if the file does not exist.
        ValueError: if the file is malformed or has missing required columns.
    """
    base = base_dir or os.getcwd()
    full_path = os.path.join(base, path) if not os.path.isabs(path) else path

    if not os.path.exists(full_path):
        raise AnnotationNotFoundError(
            f"Annotation dataset not found: {full_path!r} "
            f"(task={task_name!r}, path={path!r}). "
            "Real annotation files are created by the human annotation workflow "
            "and are not shipped with the project.  Synthetic data can be generated "
            "with ``make_synthetic_records()`` for plumbing checks only."
        )

    # Delegate to the existing loaders.
    if task_name == "sentiment":
        from app.services.nlp.sentiment_emotion.models.annotation import (
            SentimentCorpus,
        )
        corpus = SentimentCorpus.from_csv(full_path)
        return [_sentiment_to_record(a) for a in corpus.annotations]

    if task_name == "emotion":
        from app.services.nlp.sentiment_emotion.models.annotation import (
            EmotionCorpus,
        )
        corpus = EmotionCorpus.from_csv(full_path)
        return [_emotion_to_record(a) for a in corpus.annotations]

    if task_name == "issue":
        from app.services.nlp.issue_identification.models.annotation import (
            AnnotationCorpus,
        )
        corpus = AnnotationCorpus.from_csv(full_path)
        return [_issue_to_record(a) for a in corpus.annotations]

    if task_name == "response":
        from app.services.nlp.response_classification.models.annotation import (
            ResponseCorpus,
        )
        corpus = ResponseCorpus.from_csv(full_path)
        return [_response_to_record(a) for a in corpus.annotations]

    if task_name == "solution":
        from app.services.nlp.solution_detection.models.annotation import (
            SolutionCorpus,
        )
        corpus = SolutionCorpus.from_csv(full_path)
        return [_solution_to_record(a) for a in corpus.annotations]

    raise ValueError(
        f"Unknown task {task_name!r}. Known tasks: sentiment, emotion, issue, "
        f"response, solution."
    )


def _sentiment_to_record(a) -> TrainingRecord:
    return TrainingRecord(
        document_id=a.document_id,
        text=a.text,
        label=a.normalised_sentiment(),
        post_id=a.post_id,
        comment_id=a.comment_id,
        language=a.language,
        script=a.script,
        is_code_mixed=a.is_code_mixed,
        detected_languages=list(a.detected_languages),
        topic_id=a.topic_id,
    )


def _emotion_to_record(a) -> TrainingRecord:
    return TrainingRecord(
        document_id=a.document_id,
        text=a.text,
        label="|".join(a.normalised_emotions()),
        post_id=a.post_id,
        comment_id=a.comment_id,
        language=a.language,
        script=a.script,
        is_code_mixed=a.is_code_mixed,
        detected_languages=list(a.detected_languages),
        topic_id=a.topic_id,
    )


def _issue_to_record(a) -> TrainingRecord:
    return TrainingRecord(
        document_id=a.document_id,
        text=a.text,
        label="|".join(a.normalised_labels()),
        post_id=a.post_id,
        comment_id=a.comment_id,
        language=a.language,
        script=a.script,
        is_code_mixed=a.is_code_mixed,
        detected_languages=list(a.detected_languages),
        topic_id=a.topic_id,
    )


def _response_to_record(a) -> TrainingRecord:
    return TrainingRecord(
        document_id=a.document_id,
        text=a.text,
        label=a.normalised_response_type(),
        post_id=a.post_id,
        comment_id=a.comment_id,
        language=a.language,
        script=a.script,
        is_code_mixed=a.is_code_mixed,
        detected_languages=list(a.detected_languages),
        topic_id=a.topic_id,
    )


def _solution_to_record(a) -> TrainingRecord:
    from app.services.nlp.solution_detection.models.labels import target_label

    return TrainingRecord(
        document_id=a.document_id,
        text=a.text,
        label=target_label(bool(a.is_solution)),
        post_id=a.post_id,
        comment_id=a.comment_id,
        language=a.language,
        script=a.script,
        is_code_mixed=a.is_code_mixed,
        detected_languages=list(a.detected_languages),
        topic_id=a.topic_id,
        metadata={"solution_type": a.solution_type},
    )
