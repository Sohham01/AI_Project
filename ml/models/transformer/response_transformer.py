"""MuRIL-based six-class response-role classifier.

Uses the existing ResponseClassification contract. This module does not
create annotations or claim metrics from unannotated data.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.services.nlp.response_classification.base import (
    ResponseClassification,
    ResponseDocument,
    ResponseModelConfig,
)
from app.services.nlp.response_classification.models.annotation import (
    ResponseAnnotation,
    ResponseCorpus,
)
from app.services.nlp.response_classification.models.evaluation import (
    ResponseEvaluationReport,
    evaluate_response_classification,
)
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponsePrediction,
    ResponseTrainingReport,
)


class MuRILResponseConfig(ResponseModelConfig):
    """Training and inference settings for the MuRIL response classifier."""

    model_id: str = "google/muril-base-cased"
    max_length: int = 128
    batch_size: int = 2
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    device: str = "auto"
    use_class_weights: bool = True
    verbose: bool = True


class _TextLabelDataset(Dataset):
    """Small dataset wrapper; tokenization happens per batch."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in ("torch", "transformers", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class MuRILResponseClassifier(ResponseClassification):
    """Fine-tunable MuRIL encoder with a mutually exclusive six-class head."""

    _NAME = "muril_transformer"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[MuRILResponseConfig] = None,
    ) -> None:
        self._config = config or MuRILResponseConfig()
        if self._config.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self._config.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self._config.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self._config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self._device: Optional[torch.device] = None
        self._tokenizer = None
        self._model = None
        self._labels: List[str] = []
        self._trained = False
        self._trained_at: Optional[str] = None
        self._training_ids: set[str] = set()
        self._training_texts: set[str] = set()
        self._training_loss_by_epoch: List[float] = []

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def expected_labels(self) -> List[str]:
        return list(RESPONSE_TYPES)

    @property
    def is_trained(self) -> bool:
        return self._trained and self._model is not None

    @property
    def threshold(self) -> float:
        return float(self._config.threshold)

    @property
    def config(self) -> MuRILResponseConfig:
        return self._config

    @property
    def missing_response_types(self) -> List[str]:
        learned = set(self._labels)
        return [label for label in RESPONSE_TYPES if label not in learned]

    def _resolve_device(self) -> torch.device:
        requested = self._config.device.strip().lower()

        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device setting: {self._config.device!r}") from exc

    def _load_for_labels(self, labels: Sequence[str]) -> None:
        """Load pretrained encoder weights with a new classifier head."""
        self._device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_id)

        label_to_id = {label: index for index, label in enumerate(labels)}
        id_to_label = {index: label for label, index in label_to_id.items()}

        hf_config = AutoConfig.from_pretrained(self._config.model_id)
        hf_config.num_labels = len(labels)
        hf_config.label2id = label_to_id
        hf_config.id2label = id_to_label
        hf_config.problem_type = "single_label_classification"

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._config.model_id,
            config=hf_config,
        ).to(self._device)

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "MuRILResponseClassifier has not been trained. Call "
                "train() with real, annotated ResponseAnnotation records first."
            )

    def _check_threshold(self, threshold: Optional[float]) -> float:
        value = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        return value

    def _batch_encoding(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        assert self._tokenizer is not None
        assert self._device is not None

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self._device) for key, value in encoded.items()}

    def train(
        self,
        annotations: Sequence[ResponseAnnotation],
    ) -> ResponseTrainingReport:
        records = list(annotations or [])
        if not records:
            raise ValueError("Training requires at least one annotation.")

        corpus = ResponseCorpus(annotations=records)
        corpus.validate()
        corpus.validate_unique_document_ids()

        empty_ids = [r.document_id for r in records if not r.text.strip()]
        if empty_ids:
            raise ValueError(
                "Training text must not be blank. Empty document IDs: "
                + ", ".join(empty_ids[:10])
            )

        counts = Counter(record.validate_response_type() for record in records)
        labels_seen = set(counts)
        ordered_labels = [label for label in RESPONSE_TYPES if label in labels_seen]
        missing = [label for label in RESPONSE_TYPES if label not in labels_seen]

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"Only {len(records)} training documents; at least "
                f"{self._config.min_training_documents} are required."
            )

        if len(ordered_labels) < 2:
            raise ValueError(
                "MuRIL needs at least two distinct response types to train a classifier."
            )

        low_count = {
            label: counts[label]
            for label in ordered_labels
            if counts[label] < self._config.min_examples_per_class
        }
        if low_count:
            raise ValueError(
                "Insufficient examples per observed class. At least "
                f"{self._config.min_examples_per_class} are required per class; "
                f"found {low_count}."
            )

        if missing and self._config.require_all_response_types:
            raise ValueError(
                "Training corpus is missing required response types: "
                + ", ".join(missing)
            )

        self._trained = False
        self._labels = ordered_labels
        self._training_loss_by_epoch = []

        # Seed before loading the model so the new classification head
        # and subsequent stochastic training operations are reproducible.
        torch.manual_seed(self._config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.random_state)

        self._load_for_labels(self._labels)
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None

        label_to_id = {label: index for index, label in enumerate(self._labels)}
        texts = [record.text for record in records]
        targets = [label_to_id[record.validate_response_type()] for record in records]
        dataset = _TextLabelDataset(texts, targets)

        generator = torch.Generator()
        generator.manual_seed(self._config.random_state)

        def collate_batch(items):
            batch_texts, batch_labels = zip(*items)
            encoded = self._batch_encoding(batch_texts)
            target_tensor = torch.tensor(
                batch_labels, dtype=torch.long, device=self._device
            )
            return encoded, target_tensor

        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_batch,
        )

        class_weights = None
        if self._config.use_class_weights:
            total = len(records)
            n_classes = len(self._labels)
            weights = [
                total / (n_classes * counts[label])
                for label in self._labels
            ]
            class_weights = torch.tensor(
                weights, dtype=torch.float32, device=self._device
            )

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        self._model.train()

        for epoch in range(self._config.epochs):
            loss_total = 0.0
            examples_seen = 0

            for encoded, target_tensor in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self._model(**encoded).logits
                loss = criterion(logits, target_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                batch_count = int(target_tensor.shape[0])
                loss_total += float(loss.detach().item()) * batch_count
                examples_seen += batch_count

            epoch_loss = loss_total / max(examples_seen, 1)
            self._training_loss_by_epoch.append(epoch_loss)

            if self._config.verbose:
                print(
                    f"MuRIL epoch {epoch + 1}/{self._config.epochs}: "
                    f"training_loss={epoch_loss:.4f}"
                )

        self._model.eval()
        self._trained = True
        self._trained_at = datetime.now(timezone.utc).isoformat()
        self._training_ids = {str(record.document_id) for record in records}
        self._training_texts = {str(record.text) for record in records}

        warnings: List[str] = []
        if missing:
            warnings.append(
                "No training examples for response types: " + ", ".join(missing)
            )

        duplicates = corpus.duplicate_document_ids()
        if duplicates:
            warnings.append(
                "Repeated document IDs with the same label were retained: "
                + ", ".join(duplicates[:10])
            )

        if max(counts.values()) / len(records) >= self._config.imbalance_warning_share:
            warnings.append(
                "One response type dominates the training data; inspect class balance."
            )

        all_tokens = {
            token
            for text in texts
            for token in text.split()
        }

        return ResponseTrainingReport(
            model_name=self.name,
            model_version=self.version,
            n_training_documents=len(records),
            n_classes=len(self._labels),
            labels=list(self._labels),
            expected_labels=list(RESPONSE_TYPES),
            missing_response_types=missing,
            examples_per_class={label: counts[label] for label in self._labels},
            majority_class_share=max(counts.values()) / len(records),
            majority_class=max(self._labels, key=lambda label: counts[label]),
            n_features=len(all_tokens),
            classes_per_document=1,
            threshold=self.threshold,
            config=self._config.model_dump(mode="json"),
            library_versions=_library_versions(),
            trained_at=self._trained_at,
            warnings=warnings,
        )

    def predict(
        self,
        comments: Sequence[ResponseDocument],
        threshold: Optional[float] = None,
    ) -> ResponseClassificationResult:
        self._check_trained()
        confidence_floor = self._check_threshold(threshold)
        docs = list(comments or [])

        assert self._model is not None
        assert self._device is not None

        predictions: List[ResponsePrediction] = []
        predicted_counts = Counter()

        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(docs), self._config.batch_size):
                batch_docs = docs[start : start + self._config.batch_size]
                encoded = self._batch_encoding([doc.text for doc in batch_docs])
                logits = self._model(**encoded).logits
                probability_rows = torch.softmax(logits, dim=-1).cpu().tolist()

                for doc, row in zip(batch_docs, probability_rows):
                    winning_index = max(range(len(row)), key=lambda i: row[i])
                    confidence = float(row[winning_index])
                    winning_label = self._labels[winning_index]
                    abstained = confidence <= confidence_floor
                    response_type = None if abstained else winning_label

                    probabilities = {
                        label: round(float(row[index]), 6)
                        for index, label in enumerate(self._labels)
                    }

                    if response_type is not None:
                        predicted_counts[response_type] += 1

                    predictions.append(
                        ResponsePrediction(
                            document_id=doc.document_id,
                            response_type=response_type,
                            confidence=round(confidence, 6),
                            response_probabilities=probabilities,
                            abstained=abstained,
                            text=doc.text,
                            post_id=doc.post_id,
                            comment_id=doc.comment_id,
                            original_text=doc.original_text,
                            language=doc.language,
                            script=doc.script,
                            is_code_mixed=doc.is_code_mixed,
                            detected_languages=list(doc.detected_languages),
                            topic_id=doc.topic_id,
                            topic_label=doc.topic_label,
                            topic_keywords=list(doc.topic_keywords),
                            issue_labels=list(doc.issue_labels),
                            issue_probabilities=dict(doc.issue_probabilities),
                            parent_comment_text=doc.parent_comment_text,
                        )
                    )

        missing = self.missing_response_types
        warnings = []
        if missing:
            warnings.append(
                "The model has not learned these response types: "
                + ", ".join(missing)
            )

        return ResponseClassificationResult(
            model_name=self.name,
            model_version=self.version,
            threshold=confidence_floor,
            predictions=predictions,
            n_documents=len(docs),
            known_response_types=list(self._labels),
            missing_response_types=missing,
            unpredicted_response_types=[
                label for label in self._labels if predicted_counts[label] == 0
            ],
            n_abstained=sum(1 for prediction in predictions if prediction.abstained),
            n_documents_with_context=sum(1 for doc in docs if doc.has_context),
            config=self._config.model_dump(mode="json"),
            library_versions=_library_versions(),
            trained_at=self._trained_at,
            warnings=warnings,
        )

    def evaluate(
        self,
        annotations: Sequence[ResponseAnnotation],
        threshold: Optional[float] = None,
    ) -> ResponseEvaluationReport:
        self._check_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_response_classification([], [], self.labels)

        known = set(self._labels)
        gold: List[Optional[str]] = []
        unlearned_types = set()

        for record in records:
            label = record.validate_response_type()
            if label in known:
                gold.append(label)
            else:
                gold.append(None)
                unlearned_types.add(label)

        result = self.predict(
            [record.to_response_document() for record in records],
            threshold=threshold,
        )
        predicted = [prediction.response_type for prediction in result.predictions]
        report = evaluate_response_classification(gold, predicted, self._labels)

        if unlearned_types:
            report.warnings.append(
                "Evaluation includes response types the model cannot emit: "
                + ", ".join(sorted(unlearned_types))
                + "; those examples count as unscoreable gold labels."
            )

        if any(
            str(record.document_id) in self._training_ids
            or str(record.text) in self._training_texts
            for record in records
        ):
            report.warnings.append(
                "Evaluation data overlaps training IDs or text; results may "
                "reflect memorisation rather than generalisation."
            )

        return report

    @classmethod
    def from_artifact(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> "MuRILResponseClassifier":
        """Reconstruct a trained classifier from a native artifact directory."""
        destination = Path(path)
        metadata_path = destination / "metadata.json"

        if not destination.is_dir() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"MuRIL artifact directory or metadata.json not found: {destination}"
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        saved_labels = list(metadata.get("labels", []))
        canonical_labels = list(RESPONSE_TYPES)
        if (
            not saved_labels
            or len(saved_labels) != len(set(saved_labels))
            or any(label not in canonical_labels for label in saved_labels)
            or saved_labels != [label for label in canonical_labels if label in saved_labels]
        ):
            raise ValueError("Artifact contains an invalid or non-canonical label list.")

        config_values = dict(metadata.get("config", {}))
        if device is not None:
            config_values["device"] = device

        config = MuRILResponseConfig(**config_values)
        instance = cls(config=config)
        instance._device = instance._resolve_device()
        instance._tokenizer = AutoTokenizer.from_pretrained(
            str(destination),
            local_files_only=True,
        )
        instance._model = AutoModelForSequenceClassification.from_pretrained(
            str(destination),
            local_files_only=True,
        ).to(instance._device)

        if instance._model.config.num_labels != len(saved_labels):
            raise ValueError(
                "Artifact model output size does not match its saved label list."
            )

        instance._labels = saved_labels
        instance._trained = True
        instance._trained_at = metadata.get("trained_at")
        instance._training_loss_by_epoch = list(
            metadata.get("training_loss_by_epoch", [])
        )
        return instance

    def save_artifact_directory(self, path: str) -> str:
        """Explicit directory-artifact hook for the shared ML trainer."""
        return self.save_artifact(path)

    def save_artifact(self, path: str) -> str:
        """Save weights/tokenizer natively to a model directory."""
        self._check_trained()
        assert self._model is not None
        assert self._tokenizer is not None

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(destination), safe_serialization=True)
        self._tokenizer.save_pretrained(str(destination))

        metadata = {
            "model_name": self.name,
            "model_version": self.version,
            "model_id": self._config.model_id,
            "labels": list(self._labels),
            "expected_labels": list(RESPONSE_TYPES),
            "missing_response_types": self.missing_response_types,
            "threshold": self.threshold,
            "trained_at": self._trained_at,
            "config": self._config.model_dump(mode="json"),
            "library_versions": _library_versions(),
            "training_loss_by_epoch": list(self._training_loss_by_epoch),
        }

        with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        return str(destination)
