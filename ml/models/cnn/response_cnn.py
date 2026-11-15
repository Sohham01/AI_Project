"""
TextCNN response-classification adapter — implements ``ResponseClassification``
and plugs into the existing Trainer via explicit model injection for
``TaskType.RESPONSE``.

Architecture: tokenisation -> trainable embedding -> Conv1D -> ReLU ->
GlobalMaxPool1D -> Dense -> Softmax.

Owns the tokenizer end to end: ``fit()`` on TRAINING texts only,
``transform()`` on validation/test.  The fitted tokenizer is saved inside
the joblib artifact so a loaded model can score new comments.

No completed NLP package is modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

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
from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES,
    response_type_order,
)
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponsePrediction,
    ResponseTrainingReport,
)
from app.services.nlp.response_classification import (
    library_versions as _response_library_versions,
)

from ml.models.cnn.text_cnn import SimpleTokenizer, TextCNN


def _lib_versions() -> Dict[str, str]:
    base = _response_library_versions()
    return {**base, "numpy": np.__version__}


def _tokenize_for_mlp(text: str) -> str:
    """Strip placeholder/HTML-ish tokens that the logistic/ANN models already drop."""
    import re

    return re.sub(r"<[^<>]{1,64}>", "", text)


class TextCNNResponseClassifier(ResponseClassification):
    """TextCNN for V1 response role classification.

    task = "response"
    model = "text_cnn_response"
    version = "v1"

    Hyperparameters (defaults chosen for CPU testing):

    - embedding_dim = 32
    - n_filters = 32
    - kernel_size = 3
    - hidden_dim = 32
    - dropout = 0.3
    - learning_rate = 0.01
    - momentum = 0.9
    - epochs = 20
    - batch_size = 16
    - max_seq_len = 64
    - max_vocab_size = 5000
    - random_state = 42
    - clip_gradient = 5.0
    """

    _NAME = "text_cnn_response"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[ResponseModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
        # CNN architecture / optimisation
        embedding_dim: int = 32,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 20,
        batch_size: int = 16,
        max_seq_len: int = 64,
        max_vocab_size: int = 5000,
        random_state: int = 42,
        clip_gradient: float = 5.0,
        verbose: bool = False,
        # Data-adequacy guards (mirrored from the logistic/ANN adapters)
        min_training_documents: int = 5,
        min_examples_per_class: int = 2,
        require_all_response_types: bool = False,
        imbalance_warning_share: float = 0.8,
    ) -> None:
        self._config = config or ResponseModelConfig()
        self._threshold = self._validate_threshold(
            threshold if threshold is not None else self._config.threshold
        )
        self._include_timestamp = include_timestamp

        self._embedding_dim = embedding_dim
        self._n_filters = n_filters
        self._kernel_size = kernel_size
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epochs = epochs
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._max_vocab_size = max_vocab_size
        self._random_state = random_state
        self._clip_gradient = clip_gradient
        self._verbose = verbose
        self._min_training_documents = min_training_documents
        self._min_examples_per_class = min_examples_per_class
        self._require_all_response_types = require_all_response_types
        self._imbalance_warning_share = imbalance_warning_share

        # Fitted state.
        self._tokenizer: Optional[SimpleTokenizer] = None
        self._model: Optional[TextCNN] = None
        self._trained = False
        self._labels: List[str] = []
        self._class_index: Dict[str, int] = {}
        self._index_class: List[str] = []
        self._n_features = 0
        self._training_summary: Optional[ResponseTrainingReport] = None
        self._warnings: List[str] = []
        self._training_ids: frozenset = frozenset()
        self._training_texts: frozenset = frozenset()

    # ------------------------------------------------------------------
    # ResponseClassification contract
    # ------------------------------------------------------------------

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
    def missing_response_types(self) -> List[str]:
        return [name for name in RESPONSE_TYPES if name not in self._class_index]

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def config(self) -> ResponseModelConfig:
        return self._config

    @property
    def training_report(self) -> Optional[ResponseTrainingReport]:
        return self._training_summary

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    @property
    def model(self) -> TextCNN:
        """Read-only access to the underlying TextCNN (None before train)."""
        return self._model

    @property
    def tokenizer(self) -> SimpleTokenizer:
        """Read-only access to the fitted tokenizer (None before train)."""
        return self._tokenizer

    def train(
        self,
        annotations: Sequence[ResponseAnnotation],
    ) -> ResponseTrainingReport:
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = ResponseCorpus(annotations=records)
        corpus.validate()
        corpus.validate_unique_document_ids()

        present = set(corpus.labels)
        learned = response_type_order(present)
        self._class_index = {name: position for position, name in enumerate(learned)}
        self._index_class = list(learned)
        self._labels = list(learned)
        y = np.asarray(
            [self._class_index[r.normalised_response_type()] for r in records]
        )

        texts = [_tokenize_for_mlp(record.text) for record in records]

        # Reset previous fit.
        self._model = None
        self._trained = False
        self._warnings = []
        self._training_ids = frozenset(str(r.document_id) for r in records)
        self._training_texts = frozenset(str(r.text) for r in records)

        # Fit tokenizer on TRAINING texts only.
        self._tokenizer = SimpleTokenizer(
            max_vocab_size=self._max_vocab_size,
            max_seq_len=self._max_seq_len,
            lowercase=True,
        )
        self._tokenizer.fit(texts)
        X = self._tokenizer.encode(texts)
        self._n_features = int(X.shape[1])

        if self._n_features == 0:
            raise ValueError(
                "tokenizer produced zero-length sequences; check input texts and "
                "max_seq_len."
            )

        seed = self._random_state
        self._model = TextCNN(
            vocab_size=self._tokenizer.vocab_size,
            n_classes=len(self._labels),
            emb_dim=self._embedding_dim,
            n_filters=self._n_filters,
            kernel_size=self._kernel_size,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
            learning_rate=self._learning_rate,
            momentum=self._momentum,
            epochs=self._epochs,
            batch_size=self._batch_size,
            max_seq_len=self._max_seq_len,
            random_state=seed,
            clip_gradient=self._clip_gradient,
            verbose=self._verbose,
        )
        self._model.fit(X, y, class_index=self._class_index)

        self._trained = True

        counts = corpus.class_counts()
        majority_share = (
            round(max(counts.values()) / len(records), 6) if counts else 0.0
        )
        majority = (
            max(sorted(counts.items()), key=lambda kv: kv[1])[0]
            if counts
            else None
        )
        missing = corpus.missing_response_types()
        if missing:
            self._warnings.append(
                f"no training examples for {len(missing)} of "
                f"{len(RESPONSE_TYPES)} response type(s) "
                f"({', '.join(missing)}); the model will never emit them and its "
                "probability vector will cover only the annotated types."
            )
        if majority_share > self._imbalance_warning_share and majority_share > 0:
            self._warnings.append(
                f"class imbalance: {majority!r} covers {majority_share:.0%} of the "
                "corpus. Accuracy is inflated by that skew; judge macro-F1 and "
                "per-class recall, not accuracy."
            )

        report = ResponseTrainingReport(
            model_name=self._NAME,
            model_version=self._VERSION,
            n_training_documents=len(records),
            n_classes=len(learned),
            labels=learned,
            expected_labels=list(RESPONSE_TYPES),
            missing_response_types=missing,
            examples_per_class={
                name: counts.get(name, 0) for name in learned
            },
            majority_class_share=majority_share,
            majority_class=majority,
            n_features=self._n_features,
            classes_per_document=1,
            threshold=self._threshold,
            config=self._config_dict(),
            library_versions=_lib_versions(),
            trained_at=self._timestamp(),
            warnings=list(self._warnings),
        )
        self._training_summary = report

        duplicates = corpus.duplicate_document_ids()
        if duplicates:
            report.warnings.append(
                f"duplicate document_ids in training data: {', '.join(duplicates)}; "
                "every row was used, so those comments are effectively up-weighted"
            )
        return report

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        comments: Sequence[ResponseDocument],
        threshold: Optional[float] = None,
    ) -> ResponseClassificationResult:
        self._require_trained()
        docs = list(comments or [])
        floor = self._validate_threshold(
            threshold if threshold is not None else self._threshold
        )
        known = self.labels
        result_warnings: List[str] = list(self._warnings)

        if not docs:
            return ResponseClassificationResult(
                model_name=self._NAME,
                model_version=self._VERSION,
                threshold=floor,
                predictions=[],
                n_documents=0,
                known_response_types=known,
                missing_response_types=self.missing_response_types,
                config=self._config_dict(),
                library_versions=_lib_versions(),
                trained_at=(
                    self._training_summary.trained_at
                    if self._training_summary
                    else None
                ),
                warnings=result_warnings,
            )

        texts = [
            _tokenize_for_mlp(doc.text) for doc in docs
        ]
        X = self._tokenizer.encode(texts)
        proba = self._model.predict_proba(X)  # (n_docs, K)

        predictions: List[ResponsePrediction] = []
        for index, doc in enumerate(docs):
            row = proba[index]
            class_probabilities = {
                label: round(float(row[position]), 6)
                for position, label in enumerate(known)
            }
            top = (
                max(sorted(class_probabilities.items()), key=lambda kv: kv[1])[0]
                if class_probabilities
                else None
            )
            confidence = class_probabilities.get(top, 0.0) if top else 0.0
            abstained = top is None or confidence <= floor
            predictions.append(
                ResponsePrediction(
                    document_id=doc.document_id,
                    response_type=None if abstained else top,
                    confidence=round(confidence, 6),
                    response_probabilities=class_probabilities,
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

        counts: Dict[str, int] = {}
        for prediction in predictions:
            if prediction.response_type:
                counts[prediction.response_type] = (
                    counts.get(prediction.response_type, 0) + 1
                )

        return ResponseClassificationResult(
            model_name=self._NAME,
            model_version=self._VERSION,
            threshold=floor,
            predictions=predictions,
            n_documents=len(docs),
            known_response_types=known,
            missing_response_types=self.missing_response_types,
            unpredicted_response_types=[
                name for name in known if counts.get(name, 0) == 0
            ],
            n_abstained=sum(1 for p in predictions if p.abstained),
            n_documents_with_context=sum(1 for doc in docs if doc.has_context),
            config=self._config_dict(),
            library_versions=_lib_versions(),
            trained_at=(
                self._training_summary.trained_at
                if self._training_summary
                else None
            ),
            warnings=result_warnings,
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        annotations: Sequence[ResponseAnnotation],
        threshold: Optional[float] = None,
    ) -> ResponseEvaluationReport:
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_response_classification([], [], self.labels)

        known = set(self.labels)
        gold: List[Optional[str]] = []
        out_of_model: List[str] = []
        for record in records:
            value = record.validate_response_type()
            if value in known:
                gold.append(value)
            else:
                gold.append(None)
                out_of_model.append(value)

        documents = [record.to_response_document() for record in records]
        result = self.predict(documents, threshold=threshold)
        predicted = [p.response_type for p in result.predictions]

        report = evaluate_response_classification(gold, predicted, self.labels)
        if out_of_model:
            report.warnings.append(
                f"evaluation set contains response type(s) this model cannot emit "
                f"({', '.join(sorted(set(out_of_model)))}); those comments were "
                "scored against the types the model can actually predict"
            )
        overlap = self._self_scoring_warning(records)
        if overlap:
            report.warnings.append(overlap)
        return report

    # ------------------------------------------------------------------
    # Serialization / state
    # ------------------------------------------------------------------

    def save_artifact(self, path: str) -> str:
        """Save tokenizer + TextCNN to a joblib artifact."""
        import joblib
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "tokenizer": self._tokenizer,
                "metadata": self._config_dict(),
            },
            path,
        )
        return path

    def get_state(self) -> Dict[str, Any]:
        """Serializable snapshot (for inspect/debug)."""
        if not self._trained or self._model is None:
            raise RuntimeError("model not trained; call train() first")
        return {
            "name": self._NAME,
            "version": self._VERSION,
            "tokenizer_config": self._tokenizer.save(),
            "n_features": self._n_features,
            "n_classes": len(self._labels),
            "labels": list(self._labels),
            "class_index": dict(self._class_index),
            "index_class": list(self._index_class),
            "embedding_dim": self._embedding_dim,
            "n_filters": self._n_filters,
            "kernel_size": self._kernel_size,
            "hidden_dim": self._hidden_dim,
            "dropout": self._dropout,
            "learning_rate": self._learning_rate,
            "momentum": self._momentum,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "max_seq_len": self._max_seq_len,
            "max_vocab_size": self._max_vocab_size,
            "random_state": self._random_state,
            "seed": self._random_state,
            "threshold": self._threshold,
            "model_state": self._model.get_state(),
            "trained_at": self._timestamp(),
            "library_versions": _lib_versions(),
            "config": self._config_dict(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _timestamp(self) -> Optional[str]:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat() if self._include_timestamp else None

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                f"{self._NAME} has not been trained; call train(annotations) first."
            )

    def _validate_training_data(
        self, records: List[ResponseAnnotation]
    ) -> None:
        if not records:
            raise ValueError(
                "no training annotations supplied; at least one annotated comment "
                "is required"
            )
        if len({str(r.document_id) for r in records}) < len(records):
            raise ValueError("duplicate document_ids in training data")
        if len(records) < self._min_training_documents:
            raise ValueError(
                f"only {len(records)} training document(s), but "
                f"min_training_documents={self._min_training_documents}"
            )
        counts: Dict[str, int] = {}
        for r in records:
            label = r.validate_response_type()
            counts[label] = counts.get(label, 0) + 1
        for label, c in counts.items():
            if c < self._min_examples_per_class:
                raise ValueError(
                    f"response type {label!r} has only {c} example(s), but "
                    f"min_examples_per_class={self._min_examples_per_class}"
                )
        if self._require_all_response_types:
            missing = [name for name in RESPONSE_TYPES if name not in counts]
            if missing:
                raise ValueError(
                    f"training data does not cover all six V1 response types; "
                    f"missing: {', '.join(missing)}"
                )

    def _self_scoring_warning(
        self, records: List[ResponseAnnotation]
    ) -> Optional[str]:
        if not records or not self._training_ids:
            return None
        by_id = sum(1 for r in records if str(r.document_id) in self._training_ids)
        by_text = sum(1 for r in records if str(r.text) in self._training_texts)
        overlap = max(by_id, by_text)
        if overlap == 0:
            return None
        share = overlap / len(records)
        if overlap == len(records):
            return (
                f"every one of these {len(records)} evaluation comment(s) is in the "
                "training set — the scores report memorisation, not generalisation."
            )
        return (
            f"{overlap} of {len(records)} evaluation comment(s) are also in the "
            f"training set ({share:.0%}); scores are inflated by that overlap."
        )

    def _validate_threshold(self, value: Optional[float]) -> float:
        if value is None:
            return 0.5
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "name": self._NAME,
            "version": self._VERSION,
            "embedding_dim": self._embedding_dim,
            "n_filters": self._n_filters,
            "kernel_size": self._kernel_size,
            "hidden_dim": self._hidden_dim,
            "dropout": self._dropout,
            "learning_rate": self._learning_rate,
            "momentum": self._momentum,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "max_seq_len": self._max_seq_len,
            "max_vocab_size": self._max_vocab_size,
            "random_state": self._random_state,
            "clip_gradient": self._clip_gradient,
            "verbose": self._verbose,
            "n_features": self._n_features,
            "n_classes": len(self._labels),
            "labels": list(self._labels),
            "class_index": dict(self._class_index),
            "tfidf_max_features": self._config.max_features,
            "tfidf_ngram_range": self._config.ngram_range,
            "min_training_documents": self._min_training_documents,
            "min_examples_per_class": self._min_examples_per_class,
            "require_all_response_types": self._require_all_response_types,
            "imbalance_warning_share": self._imbalance_warning_share,
            "library_versions": _lib_versions(),
            "trained_at": self._timestamp(),
        }
