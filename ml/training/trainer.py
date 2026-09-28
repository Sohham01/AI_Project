"""
Generic trainer for the ML framework.

The trainer orchestrates the existing NLP classifiers without modifying them.
It loads annotation data, splits it deterministically, trains the supplied model
through its public ``train()`` / ``evaluate()`` / ``predict()`` API, evaluates
on validation and test sets, saves a joblib artifact, and records the run in the
model registry.

The trainer never reaches into a classifier's internals.  If a future model
implements the same public interface it can be plugged in without changing the
trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import joblib

from ml.configs import TaskConfig, TaskType, ModelRegistry
from ml.data.dataset import (
    AnnotationNotFoundError,
    TrainingRecord,
    load_annotation_dataset,
    make_synthetic_records,
    split_label,
)
from ml.data.splitting import (
    SplitResult,
    SplitSpec,
    apply_split,
    check_leakage,
    split_records,
)
from ml.evaluation.reports import (
    EvaluationReport,
    EvaluationReportWriter,
    build_evaluation_report,
)
from ml.evaluation.metrics import ClassificationMetricSet


# ---------------------------------------------------------------------------
# Input dataset (pre-split)
# ---------------------------------------------------------------------------


@dataclass
class LoadedDataset:
    """A dataset that has been loaded but not yet split."""

    records: List[TrainingRecord]
    task: TaskType
    source: str  # "real" or "synthetic"
    n_records: int
    labels: List[str] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_meaningful(self) -> bool:
        return self.n_records >= 10 and len(self.labels) >= 2

    def verify_no_missing_labels(self, split: SplitResult) -> List[str]:
        """Check that every label seen in evaluation also appears in training."""
        train_labels = set()
        for idx in split.train:
            if idx < len(self.records):
                train_labels.update(split_label(self.records[idx].label))
        eval_labels = set()
        for idx in split.validation + split.test:
            if idx < len(self.records):
                eval_labels.update(split_label(self.records[idx].label))
        missing = eval_labels - train_labels
        if missing:
            return [f"labels in evaluation but not training: {sorted(missing)}"]
        return []


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Mutable state passed through the training pipeline."""

    config: TaskConfig
    task: TaskType
    dataset: Optional[LoadedDataset] = None
    split: Optional[SplitResult] = None
    model: Optional[Any] = None
    trained_model: Optional[Any] = None
    leakage_report: Dict[str, Any] = field(default_factory=dict)
    validation_report: Optional[EvaluationReport] = None
    test_report: Optional[EvaluationReport] = None
    warnings: List[str] = field(default_factory=list)
    artifact_path: Optional[str] = None
    registry_id: Optional[str] = None
    include_timestamp: bool = False
    _timestamp: Optional[str] = None

    @property
    def timestamp(self) -> Optional[str]:
        if self._timestamp is None and self.include_timestamp:
            self._timestamp = datetime.now(timezone.utc).isoformat()
        return self._timestamp

    def all_warnings(self) -> List[str]:
        w = list(self.warnings)
        if self.dataset:
            w.extend(self.dataset.warnings)
        if self.split:
            w.extend(self.split.warnings)
        if self.leakage_report.get("warnings"):
            w.extend(self.leakage_report["warnings"])
        if self.validation_report:
            w.extend(self.validation_report.warnings)
        if self.test_report:
            w.extend(self.test_report.warnings)
        return w


# ---------------------------------------------------------------------------
# Model protocol — what the trainer needs from any model
# ---------------------------------------------------------------------------


class TrainableModel(Protocol):
    """The minimal interface the trainer requires from a model.

    The trainer calls these methods and passes through the results.  It does
    NOT inspect the model's internals or call sklearn directly.
    """

    name: str
    version: str
    is_trained: bool
    threshold: float
    labels: List[str]

    def train(self, annotations: Sequence[Any]) -> Any: ...
    def predict(self, documents: Sequence[Any], threshold: Optional[float] = None) -> Any: ...
    def evaluate(self, annotations: Sequence[Any], threshold: Optional[float] = None) -> Any: ...


# ---------------------------------------------------------------------------
# Model factory — maps TaskType to the existing classifiers
# ---------------------------------------------------------------------------


def build_model(task: TaskType, config: Optional[Any] = None) -> TrainableModel:
    """Instantiate the default classical baseline for *task*.

    Returns an untrained model instance.  The caller is responsible for calling
    ``train()`` before ``predict()`` or ``evaluate()``.
    """
    if task == TaskType.SENTIMENT:
        from app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment import (
            TfidfLogisticSentimentClassifier,
        )
        return TfidfLogisticSentimentClassifier(config=config)

    if task == TaskType.EMOTION:
        from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
            TfidfLogisticEmotionClassifier,
        )
        return TfidfLogisticEmotionClassifier(config=config)

    if task == TaskType.ISSUE:
        from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
            TfidfLogisticIssueClassifier,
        )
        return TfidfLogisticIssueClassifier(config=config)

    if task == TaskType.RESPONSE:
        from app.services.nlp.response_classification.tfidf_logistic_multiclass import (
            TfidfLogisticResponseClassifier,
        )
        return TfidfLogisticResponseClassifier(config=config)

    if task == TaskType.SOLUTION:
        from app.services.nlp.solution_detection.tfidf_logistic_binary import (
            TfidfLogisticSolutionClassifier,
        )
        return TfidfLogisticSolutionClassifier(config=config)

    raise ValueError(f"No default model for task {task.value!r}")


# ---------------------------------------------------------------------------
# Annotation adapter — converts TrainingRecord -> task-specific annotation
# ---------------------------------------------------------------------------


def records_to_annotations(
    records: Sequence[TrainingRecord],
    task: TaskType,
) -> List[Any]:
    """Convert a list of ``TrainingRecord`` to the task's annotation type.

    This is needed because the existing classifiers accept their own annotation
    types (SentimentAnnotation, EmotionAnnotation, ...) and reject everything
    else.  The adapter rebuilds the right type from the generic record.
    """
    if task == TaskType.SENTIMENT:
        from app.services.nlp.sentiment_emotion.models.annotation import (
            SentimentAnnotation,
        )
        return [
            SentimentAnnotation(
                document_id=r.document_id,
                text=r.text,
                sentiment=r.label,
                post_id=r.post_id,
                comment_id=r.comment_id,
                language=r.language,
                script=r.script,
                is_code_mixed=r.is_code_mixed,
                detected_languages=r.detected_languages,
                topic_id=r.topic_id,
            )
            for r in records
        ]

    if task == TaskType.EMOTION:
        from app.services.nlp.sentiment_emotion.models.annotation import (
            EmotionAnnotation,
        )
        # emotion/ issue are multi-label; preserve the sorted ordering used by
        # the source annotation's normalised_* methods.
        return [
            EmotionAnnotation(
                document_id=r.document_id,
                text=r.text,
                emotions=sorted(set(split_label(r.label))),
                post_id=r.post_id,
                comment_id=r.comment_id,
                language=r.language,
                script=r.script,
                is_code_mixed=r.is_code_mixed,
                detected_languages=r.detected_languages,
                topic_id=r.topic_id,
            )
            for r in records
        ]

    if task == TaskType.ISSUE:
        from app.services.nlp.issue_identification.models.annotation import (
            IssueAnnotation,
        )
        return [
            IssueAnnotation(
                document_id=r.document_id,
                text=r.text,
                labels=sorted(set(split_label(r.label))),
                post_id=r.post_id,
                comment_id=r.comment_id,
                language=r.language,
                script=r.script,
                is_code_mixed=r.is_code_mixed,
                detected_languages=r.detected_languages,
                topic_id=r.topic_id,
            )
            for r in records
        ]

    if task == TaskType.RESPONSE:
        from app.services.nlp.response_classification.models.annotation import (
            ResponseAnnotation,
        )
        return [
            ResponseAnnotation(
                document_id=r.document_id,
                text=r.text,
                response_type=r.label,
                post_id=r.post_id,
                comment_id=r.comment_id,
                language=r.language,
                script=r.script,
                is_code_mixed=r.is_code_mixed,
                detected_languages=r.detected_languages,
                topic_id=r.topic_id,
            )
            for r in records
        ]

    if task == TaskType.SOLUTION:
        from app.services.nlp.solution_detection.models.annotation import (
            SolutionAnnotation,
        )
        return [
            SolutionAnnotation(
                document_id=r.document_id,
                text=r.text,
                is_solution=(r.label.strip().lower() in {"true", "1", "yes"}),
                solution_type=r.metadata.get("solution_type"),
                post_id=r.post_id,
                comment_id=r.comment_id,
                language=r.language,
                script=r.script,
                is_code_mixed=r.is_code_mixed,
                detected_languages=r.detected_languages,
                topic_id=r.topic_id,
                issue_labels=[],
                response_type=None,
            )
            for r in records
        ]

    raise ValueError(f"Cannot convert records to annotations for unknown task {task.value!r}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrate training, evaluation, and artifact saving for one task.

    Usage::

        config = TaskConfig(task_name="sentiment")
        trainer = Trainer(config)
        result = trainer.run()
        print(result.test_report.metrics)
    """

    def __init__(self, config: TaskConfig) -> None:
        if config.task_name not in {t.value for t in TaskType}:
            raise ValueError(
                f"Unknown task {config.task_name!r}. "
                f"Known tasks: {', '.join(t.value for t in TaskType)}"
            )
        self.config = config
        self.task = TaskType(config.task_name)
        self.state = PipelineState(config=config, task=self.task)
        self._model: Optional[TrainableModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, use_synthetic: bool = False) -> LoadedDataset:
        """Load the dataset for the configured task.

        Args:
            use_synthetic: if True and the real annotation file is missing,
                generate SYNTHETIC demo data.  Synthetic data is always labelled
                as such and is never presented as real project performance.

        Returns:
            LoadedDataset with the records and metadata.
        """
        task = self.task
        path = self.config.dataset_path

        records: List[TrainingRecord]
        source: str
        warnings: List[str] = []

        try:
            records = load_annotation_dataset(
                task_name=task.value,
                path=path,
            )
            source = "real"
        except AnnotationNotFoundError as exc:
            if use_synthetic:
                warnings.append(
                    f"REAL annotation file not found ({path}): {exc}. "
                    "Generating SYNTHETIC DEMO data for plumbing verification only. "
                    "These metrics are NOT project performance."
                )
                label_options = _default_label_options(task)
                records = make_synthetic_records(
                    n=self.config.minimum_training_examples + 20,
                    label_options=label_options,
                    single_label=not self.config.is_multilabel,
                    seed=self.config.random_state,
                )
                source = "synthetic"
            else:
                raise

        if not records:
            raise ValueError(
                f"Dataset for task {task.value!r} is empty after loading "
                f"({path}).  At least one annotated record is required."
            )

        # Derive labels and counts.
        all_labels: List[str] = []
        counts: Dict[str, int] = {}
        for r in records:
            for lbl in split_label(r.label):
                all_labels.append(lbl)
                counts[lbl] = counts.get(lbl, 0) + 1

        dataset = LoadedDataset(
            records=records,
            task=task,
            source=source,
            n_records=len(records),
            labels=sorted(set(all_labels)),
            label_counts=dict(sorted(counts.items())),
            warnings=warnings,
        )

        if not dataset.is_meaningful:
            dataset.warnings.append(
                f"dataset may be too small or lack label diversity: "
                f"{dataset.n_records} records, {len(dataset.labels)} labels. "
                f"At least 10 records and 2 labels are recommended."
            )

        self.state.dataset = dataset
        return dataset

    def split_dataset(
        self,
        dataset: Optional[LoadedDataset] = None,
        spec: Optional[SplitSpec] = None,
    ) -> SplitResult:
        """Split a loaded dataset into train/validation/test index lists.

        Args:
            dataset: dataset to split.  Defaults to the last loaded dataset.
            spec: split specification.  Defaults to config-driven proportions.

        Returns:
            SplitResult with index lists.
        """
        ds = dataset or self.state.dataset
        if ds is None:
            raise RuntimeError("no dataset loaded; call load_dataset() first")

        spec = spec or SplitSpec(
            train_size=self.config.train_size,
            validation_size=self.config.validation_size,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=not self.config.is_multilabel,
            min_train_examples=max(1, self.config.minimum_training_examples),
            min_validation_examples=1,
            min_test_examples=1,
        )

        split = split_records(
            records=ds.records,
            labels=[r.label for r in ds.records],
            spec=spec,
        )
        self.state.split = split
        return split

    def train(self, model: Optional[TrainableModel] = None) -> Any:
        """Train the model on the training split.

        Args:
            model: model to train.  Defaults to the model supplied in the
                config, or the default baseline for the task.

        Returns:
            The model's training report (task-specific type).
        """
        if self.state.split is None:
            raise RuntimeError("no split available; call split_dataset() first")
        if self.state.dataset is None:
            raise RuntimeError("no dataset loaded; call load_dataset() first")

        if model is not None:
            self._model = model
        elif self._model is None:
            self._model = self._resolve_model()

        if self._model is None:
            raise RuntimeError("no model available for training")

        # Convert training records to the model's annotation type.
        train_indices = self.state.split.train
        train_records = [self.state.dataset.records[i] for i in train_indices]
        annotations = records_to_annotations(train_records, self.task)

        if len(annotations) < self.config.minimum_training_examples:
            raise ValueError(
                f"training set has only {len(annotations)} examples, "
                f"but minimum_training_examples={self.config.minimum_training_examples}"
            )

        # Train.
        training_report = self._model.train(annotations)

        # Leakage check: ensure no evaluation record is in training.
        self.state.trained_model = self._model
        return training_report

    def evaluate_validation(self) -> EvaluationReport:
        """Evaluate the trained model on the validation split.

        Returns:
            Unified EvaluationReport.
        """
        return self._evaluate_split("validation")

    def evaluate_test(self) -> EvaluationReport:
        """Evaluate the trained model on the test split.

        Returns:
            Unified EvaluationReport.
        """
        return self._evaluate_split("test")

    def save_artifact(self, model: Optional[Any] = None) -> str:
        """Save the trained model + metadata to a joblib artifact.

        Args:
            model: model to save.  Defaults to the trained model.

        Returns:
            Path to the saved artifact.
        """
        model = model or self.state.trained_model
        if model is None:
            raise RuntimeError("no trained model to save")

        os.makedirs(self.config.artifact_directory, exist_ok=True)

        # Deterministic artifact name.
        name = self._artifact_name(model)
        path = os.path.join(self.config.artifact_directory, name + ".joblib")

        metadata = self._build_artifact_metadata(model)

        joblib.dump({"model": model, "metadata": metadata}, path)

        self.state.artifact_path = path
        return path

    def run(
        self,
        model: Optional[TrainableModel] = None,
        use_synthetic: bool = False,
        split_spec: Optional[SplitSpec] = None,
        save_artifact_flag: bool = True,
        register: bool = True,
    ) -> Dict[str, Any]:
        """Run the full training pipeline end-to-end.

        Pipeline::

            load_dataset()
                -> split_dataset()
                    -> leak_check()
                        -> train()
                            -> evaluate_validation()
                                -> evaluate_test()
                                    -> save_artifact()
                                        -> register()

        Args:
            model: model to train.  When None the default baseline for the
                task is used.
            use_synthetic: fall back to synthetic data when real annotations
                are missing.
            split_spec: override the default split.
            save_artifact_flag: whether to write a joblib artifact.
            register: whether to record the run in the model registry.

        Returns:
            Dictionary with the full pipeline result including training report,
            validation report, test report, leakage report, and artifact path.
        """
        # 1. Load.
        dataset = self.load_dataset(use_synthetic=use_synthetic)

        # 2. Split.
        split = self.split_dataset(spec=split_spec)

        # 3. Leakage check.
        train_recs, val_recs, test_recs = apply_split(dataset.records, split)
        self.state.leakage_report = check_leakage(train_recs, val_recs, test_recs)

        if self.state.leakage_report["leakage_detected"]:
            self.state.warnings.append(
                "DATA LEAKAGE DETECTED in split — evaluation results may be inflated. "
                "See leakage_report for details."
            )

        # 4. Train.
        training_report = self.train(model=model)

        # 5. Validate.
        self.state.validation_report = self.evaluate_validation()

        # 6. Test.
        self.state.test_report = self.evaluate_test()

        # 7. Save artifact.
        if save_artifact_flag:
            self.save_artifact()

        # 8. Register.
        if register:
            self._register()

        return self._build_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self) -> TrainableModel:
        if self.config.model is not None:
            return self.config.model
        return build_model(self.task)

    def _evaluate_split(self, which: str) -> EvaluationReport:
        if self.state.trained_model is None:
            raise RuntimeError("model not trained; call train() first")
        if self.state.split is None or self.state.dataset is None:
            raise RuntimeError("no split or dataset available")

        idx_key = "validation" if which == "validation" else "test"
        indices = getattr(self.state.split, idx_key)
        if not indices:
            return EvaluationReport(
                task_name=self.task.value,
                model_name=self._model.name if self._model else "unknown",
                model_version=self._model.version if self._model else "unknown",
                is_meaningful=False,
                metrics={},
                warnings=[f"no {which} split available"],
            )

        eval_records = [self.state.dataset.records[i] for i in indices]
        annotations = records_to_annotations(eval_records, self.task)

        # Call the model's own evaluate().
        raw_report = self.state.trained_model.evaluate(annotations)

        # Build the unified report.
        leak_check = self.state.leakage_report
        return build_evaluation_report(
            task_name=self.task.value,
            model_name=self.state.trained_model.name,
            model_version=self.state.trained_model.version,
            n_train=len(self.state.split.train),
            n_validation=len(self.state.split.validation),
            n_test=len(self.state.split.test),
            labels=self.state.trained_model.labels,
            raw_report=raw_report,
            is_multilabel=self.config.is_multilabel,
            model_is_trained=self.state.trained_model.is_trained,
            leakage_detected=leak_check.get("leakage_detected", False),
            include_timestamp=self.state.include_timestamp,
            timestamp=self.state.timestamp,
        )

    def _artifact_name(self, model: Any) -> str:
        """Deterministic artifact name from task + model identity."""
        return f"{self.task.value}__{model.name}__{model.version}"

    def _build_artifact_metadata(self, model: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "task": self.task.value,
            "model_name": model.name,
            "model_version": model.version,
            "artifact_path": self.state.artifact_path,
            "dataset": self.config.dataset_path,
            "source": self.state.dataset.source if self.state.dataset else "unknown",
            "n_training_records": len(self.state.split.train) if self.state.split else 0,
            "n_validation_records": len(self.state.split.validation) if self.state.split else 0,
            "n_test_records": len(self.state.split.test) if self.state.split else 0,
            "labels": model.labels,
            "configuration": self.config.to_dict(),
            "split": (
                {
                    "random_state": self.state.split.random_state,
                    "stratify_used": self.state.split.stratify_used,
                    "stratify_available": self.state.split.stratify_available,
                    "train_size": self.config.train_size,
                    "validation_size": self.config.validation_size,
                    "test_size": self.config.test_size,
                }
                if self.state.split
                else {}
            ),
            "trained_at": self.state.timestamp,
        }
        return meta

    def _register(self) -> None:
        registry = ModelRegistry()
        entry = {
            "task": self.task.value,
            "model_name": self.state.trained_model.name,
            "model_version": self.state.trained_model.version,
            "artifact_path": self.state.artifact_path or "",
            "dataset": self.config.dataset_path,
            "configuration": self.config.to_dict(),
            "metrics": self._registry_metrics(),
            "warnings": self.state.all_warnings(),
        }
        self.state.registry_id = registry.register(entry)

    def _registry_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if self.state.validation_report:
            metrics["validation"] = self.state.validation_report.metrics_serializable()
        if self.state.test_report:
            metrics["test"] = self.state.test_report.metrics_serializable()
        return metrics

    def _build_result(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "config": self.config.to_dict(),
            "dataset": {
                "source": self.state.dataset.source if self.state.dataset else "unknown",
                "n_records": self.state.dataset.n_records if self.state.dataset else 0,
                "labels": self.state.dataset.labels if self.state.dataset else [],
                "label_counts": self.state.dataset.label_counts
                if self.state.dataset
                else {},
            },
            "split": (
                {
                    "random_state": self.state.split.random_state,
                    "stratify_used": self.state.split.stratify_used,
                    "stratify_available": self.state.split.stratify_available,
                    "n_train": len(self.state.split.train),
                    "n_validation": len(self.state.split.validation),
                    "n_test": len(self.state.split.test),
                }
                if self.state.split
                else {}
            ),
            "leakage_report": self.state.leakage_report,
            "training_report": (
                self.state.trained_model.training_report
                if self.state.trained_model
                else None
            ),
            "validation_report": self.state.validation_report,
            "test_report": self.state.test_report,
            "artifact_path": self.state.artifact_path,
            "registry_id": self.state.registry_id,
            "warnings": self.state.all_warnings(),
            "timestamp": self.state.timestamp,
        }


def _default_label_options(task: TaskType) -> List[str]:
    """Default label pool for synthetic data generation."""
    mapping = {
        TaskType.SENTIMENT: ["positive", "negative", "neutral"],
        TaskType.EMOTION: ["frustration", "anxiety", "gratitude", "anger", "hope"],
        TaskType.ISSUE: ["fees", "hostel", "mess_food", "placement", "roi", "cat_prep"],
        TaskType.RESPONSE: [
            "agreement",
            "personal_experience",
            "opinion",
            "advice",
            "solution",
            "counterargument",
        ],
        TaskType.SOLUTION: ["true", "false"],
    }
    return mapping.get(task, ["label_a", "label_b", "label_c"])
