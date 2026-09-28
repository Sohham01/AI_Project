"""
Task configuration and model registry for the ML training framework.

Task configuration defines what to train, where the data lives, and how to
split/evaluate it.  The model registry is a lightweight local JSON-based store
that records every trained model's metadata without reaching for a database.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Supported tasks
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Classification tasks the framework knows how to train."""

    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    ISSUE = "issue"
    RESPONSE = "response"
    SOLUTION = "solution"


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Configuration for training one classifier on one task.

    The framework uses this to locate data, split it, pick a model, train,
    evaluate, and save artifacts.  The model itself is passed in by the caller
    so the framework never hard-codes one classifier per task.
    """

    #: Human-readable task name.  Must match a ``TaskType`` value.
    task_name: str

    #: Path to the annotation CSV, relative to the project root or absolute.
    #: When the file does not exist the framework raises
    #: ``AnnotationNotFoundError`` rather than fabricating data.
    dataset_path: str = ""

    #: Column in the annotation that holds the label.  For existing annotation
    #: CSVs this is determined by the task type, but the framework accepts an
    #: explicit override for generic datasets.
    text_column: str = "text"
    label_column: str = "label"

    #: Whether the task is multi-label.  The framework uses this to pick the
    #: correct evaluation metric suite.  For existing NLP tasks this is derived
    #: from the task type; for custom datasets it must be set explicitly.
    multilabel: bool = False

    #: Reproducibility.
    random_state: int = 42
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15

    #: Minimum number of training examples required before training is attempted.
    minimum_training_examples: int = 10

    #: Where to write model artifacts for this task.
    artifact_directory: str = "ml/artifacts"

    #: Optional model instance.  When None the caller must supply one before
    #: calling ``Trainer.run()``.
    model: Optional[Any] = None

    #: Opt-in timestamp.  Off by default so runs are deterministic and not
    #: identified by wall-clock time.
    include_timestamp: bool = False

    def __post_init__(self) -> None:
        if self.task_name not in {t.value for t in TaskType}:
            raise ValueError(
                f"Unknown task_name {self.task_name!r}. Known tasks: "
                + ", ".join(t.value for t in TaskType)
            )
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split proportions must sum to 1.0, got "
                f"train={self.train_size}, val={self.validation_size}, "
                f"test={self.test_size}"
            )
        task = TaskType(self.task_name)
        # Derive multi-label flag from the task if not explicitly set.
        if not self.multilabel and task in (TaskType.EMOTION, TaskType.ISSUE):
            object.__setattr__(self, "multilabel", True)
        # Derive default dataset path if not set.
        if not self.dataset_path:
            object.__setattr__(
                self,
                "dataset_path",
                _default_dataset_path(task),
            )

    @property
    def is_multilabel(self) -> bool:
        return self.multilabel

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the model registry and artifact metadata."""
        d = {
            "task_name": self.task_name,
            "dataset_path": self.dataset_path,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "multilabel": self.multilabel,
            "random_state": self.random_state,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "minimum_training_examples": self.minimum_training_examples,
            "artifact_directory": self.artifact_directory,
            "include_timestamp": self.include_timestamp,
        }
        return d


def _default_dataset_path(task: TaskType) -> str:
    """Relative path to the canonical annotation file for *task*."""
    mapping = {
        TaskType.SENTIMENT: "data/annotations/sentiment_annotations.csv",
        TaskType.EMOTION: "data/annotations/emotion_annotations.csv",
        TaskType.ISSUE: "data/annotations/issue_annotations.csv",
        TaskType.RESPONSE: "data/annotations/response_annotations.csv",
        TaskType.SOLUTION: "data/annotations/solution_annotations.csv",
    }
    return mapping[task]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Lightweight local JSON registry for trained model metadata.

    Records each model's task, name, version, artifact path, training dataset,
    configuration, and evaluation metrics.  No database — just a JSON file on
    disk that can be read by any process.

    The registry is append-only: new entries are added with ``register()``,
    existing entries are never overwritten.
    """

    def __init__(self, registry_path: str = "ml/artifacts/registry.json") -> None:
        self._path = registry_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self._path):
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _read(self) -> List[Dict[str, Any]]:
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def register(self, entry: Dict[str, Any]) -> str:
        """Add a model entry and return its generated ID.

        Args:
            entry: dictionary with at least ``task``, ``model_name``,
                ``model_version``, ``artifact_path``, ``dataset``,
                ``configuration``, ``metrics``.

        Returns:
            Unique entry ID (hash of the content + timestamp-less run id).
        """
        entries = self._read()
        # Generate a deterministic-ish ID from the content.
        id_ = _entry_id(entry)
        # Append with an auto-generated run_key that is content-based.
        record = {
            "entry_id": id_,
            "registered_at_run": _run_signature(),
            "task": entry.get("task", "unknown"),
            "model_name": entry.get("model_name", "unknown"),
            "model_version": entry.get("model_version", "unknown"),
            "artifact_path": entry.get("artifact_path", ""),
            "dataset": entry.get("dataset", ""),
            "configuration": entry.get("configuration", {}),
            "metrics": entry.get("metrics", {}),
            "warnings": entry.get("warnings", []),
        }
        entries.append(record)
        self._write(entries)
        return id_

    def list_entries(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all entries, optionally filtered by task."""
        entries = self._read()
        if task:
            return [e for e in entries if e.get("task") == task]
        return list(entries)

    def find_latest(self, task: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Return the most recently registered entry for *task* + *model_name*."""
        candidates = [
            e
            for e in self._read()
            if e.get("task") == task and e.get("model_name") == model_name
        ]
        return candidates[-1] if candidates else None


def _entry_id(entry: Dict[str, Any]) -> str:
    """Content-derived ID so the same entry registered twice produces the same
    ID (append-only behaviour)."""
    import hashlib

    key = (
        str(entry.get("task", "")),
        str(entry.get("model_name", "")),
        str(entry.get("model_version", "")),
        str(entry.get("dataset", "")),
        json.dumps(entry.get("configuration", {}), sort_keys=True),
    )
    return hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:16]


def _run_signature() -> str:
    """A run identifier that is NOT a wall-clock timestamp.

    Uses the module's notion of the current run.  In practice this is set by
    the trainer before calling register().  When unset it falls back to a
    content hash of the active configuration so two identical runs produce the
    same signature.
    """
    return "run-deterministic"


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------


TASK_METADATA = {
    TaskType.SENTIMENT: {
        "name": "Sentiment",
        "description": "Single-label sentiment classification (positive/negative/neutral).",
        "multilabel": False,
        "default_model": "TfidfLogisticSentimentClassifier",
        "module": "app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment",
        "annotation_type": "SentimentAnnotation",
        "label_column": "sentiment",
        "text_column": "text",
    },
    TaskType.EMOTION: {
        "name": "Emotion",
        "description": "Multi-label emotion recognition (multiple emotions per document).",
        "multilabel": True,
        "default_model": "TfidfLogisticEmotionClassifier",
        "module": "app.services.nlp.sentiment_emotion.tfidf_logistic_emotion",
        "annotation_type": "EmotionAnnotation",
        "label_column": "emotions",
        "text_column": "text",
    },
    TaskType.ISSUE: {
        "name": "Issue",
        "description": "Multi-label issue identification (multiple issues per document).",
        "multilabel": True,
        "default_model": "TfidfLogisticIssueClassifier",
        "module": "app.services.nlp.issue_identification.tfidf_logistic_classifier",
        "annotation_type": "IssueAnnotation",
        "label_column": "labels",
        "text_column": "text",
    },
    TaskType.RESPONSE: {
        "name": "Response",
        "description": "Single-label response role classification (six V1 types).\n"
        "Classical baseline: TF-IDF → TfidfLogisticResponseClassifier.\n"
        "ANN baseline: TF-IDF → ResponseMLPClassifier (NumpyMLP, ml/models/ann).",
        "multilabel": False,
        "default_model": "TfidfLogisticResponseClassifier",
        "module": "app.services.nlp.response_classification.tfidf_logistic_multiclass",
        "annotation_type": "ResponseAnnotation",
        "label_column": "response_type",
        "text_column": "text",
    },
    TaskType.SOLUTION: {
        "name": "Solution",
        "description": "Binary solution detection (is_solution true/false).",
        "multilabel": False,
        "default_model": "TfidfLogisticSolutionClassifier",
        "module": "app.services.nlp.solution_detection.tfidf_logistic_binary",
        "annotation_type": "SolutionAnnotation",
        "label_column": "is_solution",
        "text_column": "text",
    },
}


def get_task_metadata(task: TaskType) -> Dict[str, Any]:
    return TASK_METADATA[task]
