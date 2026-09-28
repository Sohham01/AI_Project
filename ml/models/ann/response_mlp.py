"""
Response-classification adapter that wraps ``NumpyMLP`` and presents the
same public interface the ML Trainer expects for the classical baselines.

This module imports types from the frozen ``response_classification`` package
for data contracts only — it does NOT modify or depend on those implementations
beyond reading annotation/result/result types.

The adapter owns TF-IDF featurisation end to end: fit on training texts,
transform validation/test through the same fitted vectorizer, and save the
vectorizer inside the artifact so a loaded model can score new comments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    ResponseTrainingReport,
)
from app.services.nlp.response_classification import (
    library_versions as _response_library_versions,
)

from ml.models.ann.numpy_mlp import NumpyMLP


def _lib_versions() -> Dict[str, str]:
    base = _response_library_versions()
    return {**base, "numpy": np.__version__}


class ResponseMLPClassifier(ResponseClassification):
    """
    TF-IDF -> NumpyMLP(256, ReLU, Dropout, 64, ReLU, 6, Softmax) for response.

    The first neural baseline in the ANN experiment track.  It plugs into the
    existing Trainer via the ``ResponseClassification`` interface, so the same
    split, evaluation, artifact and registry code used for the logistic baseline
    also runs this model.

    Usage::

        clf = ResponseMLPClassifier(
            hidden1=256,
            hidden2=64,
            dropout=0.3,
            learning_rate=0.01,
            momentum=0.9,
            epochs=40,
            batch_size=16,
            random_state=42,
        )
        report = clf.train(annotations)      # synthetic demo only at this stage
        result = clf.predict(comments)
        metrics = clf.evaluate(held_out)

    Thread context fields (``parent_comment_text``, ``topic_id``,
    ``issue_labels``) are accepted and echoed back but deliberately NOT read:
    V1 classifies each comment on its own text, matching the logistic baseline.
    """

    _NAME = "response_mlp_numpy"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[ResponseModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
        # MLP architecture / optimisation
        hidden1: int = 256,
        hidden2: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 40,
        batch_size: int = 16,
        random_state: int = 42,
        # Gradient clipping
        clip_gradient: float = 5.0,
        # Verbosity
        verbose: bool = False,
        # Override seed
        seed: Optional[int] = None,
        # TF-IDF knobs (mirrored from ResponseModelConfig for feature-space parity)
        max_features: int = 20000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 1,
        max_df: float = 1.0,
        stop_words: Optional[Any] = None,
        lowercase: bool = True,
        token_pattern: str = r"(?u)\b\w+\b",
        norm: str = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        # Data-adequacy guards
        min_training_documents: int = 5,
        min_examples_per_class: int = 2,
        require_all_response_types: bool = False,
        imbalance_warning_share: float = 0.8,
        drop_placeholder_tokens: bool = True,
    ) -> None:
        # Shared TF-IDF config (stored for the ``config`` property so callers
        # see a sensible ResponseModelConfig object).
        base_tfidf = config or ResponseModelConfig()
        self._config = ResponseModelConfig(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            token_pattern=token_pattern,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            C=base_tfidf.C,
            max_iter=base_tfidf.max_iter,
            solver=base_tfidf.solver,
            class_weight=base_tfidf.class_weight,
            random_state=base_tfidf.random_state,
            threshold=base_tfidf.threshold,
            min_training_documents=base_tfidf.min_training_documents,
            min_examples_per_class=base_tfidf.min_examples_per_class,
            require_all_response_types=base_tfidf.require_all_response_types,
            imbalance_warning_share=base_tfidf.imbalance_warning_share,
            drop_placeholder_tokens=base_tfidf.drop_placeholder_tokens,
        )

        # Hyper-parameters forwarded to NumpyMLP.
        self._hidden1 = hidden1
        self._hidden2 = hidden2
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epochs = epochs
        self._batch_size = batch_size
        self._random_state = random_state
        self._clip_gradient = clip_gradient
        self._verbose = verbose
        self._seed = seed

        # TF-IDF knobs forwarded to NumpyMLP via the shared config.
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._sublinear_tf = sublinear_tf
        self._min_df = min_df
        self._max_df = max_df
        self._stop_words = stop_words
        self._lowercase = lowercase
        self._token_pattern = token_pattern
        self._norm = norm
        self._use_idf = use_idf
        self._smooth_idf = smooth_idf

        # Data-adequacy guards forwarded to NumpyMLP.
        self._min_training_documents = min_training_documents
        self._min_examples_per_class = min_examples_per_class
        self._require_all_response_types = require_all_response_types
        self._imbalance_warning_share = imbalance_warning_share
        self._drop_placeholder_tokens = drop_placeholder_tokens

        self._threshold = self._validate_threshold(
            threshold if threshold is not None else self._config.threshold
        )
        self._include_timestamp = include_timestamp

        # Fitted state: None until train().
        self._mlp: Optional[NumpyMLP] = None
        self._vectorizer: Any = None
        self._vocabulary_: Optional[Dict[str, int]] = None
        self._n_features: int = 0
        self._class_index: Dict[str, int] = {}
        self._labels: List[str] = []
        self._trained = False
        self._training_summary: Optional[ResponseTrainingReport] = None
        self._warnings: List[str] = []
        self._training_ids: frozenset = frozenset()
        self._training_texts: frozenset = frozenset()

    # ------------------------------------------------------------------
    # Public interface (ResponseClassification contract)
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

    def train(
        self,
        annotations: Sequence[ResponseAnnotation],
    ) -> ResponseTrainingReport:
        """Fit TF-IDF + NumpyMLP over the (possibly partial) set of V1 types."""
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = ResponseCorpus(annotations=records)
        corpus.validate()
        corpus.validate_unique_document_ids()

        present = set(corpus.labels)
        learned = response_type_order(present)
        self._class_index = {name: position for position, name in enumerate(learned)}
        self._labels = list(learned)
        y = np.asarray([self._class_index[r.normalised_response_type()] for r in records])

        texts = [self._featurize_text(record.text) for record in records]

        # Reset previous fit.
        self._mlp = None
        self._trained = False
        self._warnings = []
        self._training_ids = frozenset(str(r.document_id) for r in records)
        self._training_texts = frozenset(str(r.text) for r in records)

        mc = self._config
        self._vectorizer = self._build_vectorizer()
        X = self._fit_vectorizer(texts)
        self._n_features = int(X.shape[1])

        if self._n_features == 0:
            raise ValueError(
                "TF-IDF produced zero features from the training texts after "
                "filtering.  Check that ``drop_placeholder_tokens`` is not "
                "discarding every term, and that the texts are non-empty."
            )

        seed = self._seed if self._seed is not None else self._random_state
        self._mlp = NumpyMLP(
            input_dim=self._n_features,
            n_classes=len(self._labels),
            hidden1=self._hidden1,
            hidden2=self._hidden2,
            dropout=self._dropout,
            learning_rate=self._learning_rate,
            momentum=self._momentum,
            epochs=self._epochs,
            batch_size=self._batch_size,
            random_state=seed,
            clip_gradient=self._clip_gradient,
            verbose=self._verbose,
        )
        self._mlp.fit(X, y)

        self._trained = True

        counts = corpus.class_counts()
        majority_share = (
            round(max(counts.values()) / len(records), 6) if counts else 0.0
        )
        majority = (
            max(sorted(counts.items()), key=lambda kv: kv[1])[0] if counts else None
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
            examples_per_class={name: counts.get(name, 0) for name in learned},
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
        """Predict one response role per comment with the full probability vector."""
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

        X = self._vectorizer.transform(
            [self._featurize_text(doc.text) for doc in docs]
        )
        if hasattr(X, "toarray"):
            X = X.toarray()
        proba = self._mlp.predict_proba(X)  # (K, n_docs)

        from app.services.nlp.response_classification.models.result import (
            ResponsePrediction,
        )

        predictions: List[ResponsePrediction] = []
        for index, doc in enumerate(docs):
            row = proba[:, index]
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
        """Score predictions against gold annotations."""
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
        """Save the TF-IDF vectorizer + MLP to a joblib artifact."""
        import joblib

        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self._mlp,
                "vectorizer": self._vectorizer,
                "metadata": self._config_dict(),
            },
            path,
        )
        return path

    def get_state(self) -> Dict[str, Any]:
        """Serializable snapshot of the trained model (for inspect/debug)."""
        if not self._trained or self._mlp is None:
            raise RuntimeError("model not trained; call train() first")
        return {
            "name": self._NAME,
            "version": self._VERSION,
            "n_features": self._n_features,
            "n_classes": len(self._labels),
            "labels": list(self._labels),
            "class_index": dict(self._class_index),
            "hidden1": self._hidden1,
            "hidden2": self._hidden2,
            "dropout": self._dropout,
            "learning_rate": self._learning_rate,
            "momentum": self._momentum,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "random_state": self._random_state,
            "seed": self._seed,
            "threshold": self._threshold,
            "mlp_state": self._mlp.get_state(),
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

    def _validate_training_data(self, records: List[ResponseAnnotation]) -> None:
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

    def _featurize_text(self, text: str) -> str:
        if not self._drop_placeholder_tokens:
            return text
        import re

        return re.sub(r"<[^<>]{1,64}>", "", text)

    def _build_vectorizer(self) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=self._ngram_range,
            sublinear_tf=self._sublinear_tf,
            min_df=self._min_df,
            max_df=self._max_df,
            stop_words=self._stop_words,
            lowercase=self._lowercase,
            token_pattern=self._token_pattern,
            norm=self._norm,
            use_idf=self._use_idf,
            smooth_idf=self._smooth_idf,
        )

    def _fit_vectorizer(self, texts: List[str]) -> np.ndarray:
        self._vectorizer.fit(texts)
        self._vocabulary_ = dict(self._vectorizer.vocabulary_)
        return self._vectorizer.transform(texts).toarray()

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "name": self._NAME,
            "version": self._VERSION,
            "hidden1": self._hidden1,
            "hidden2": self._hidden2,
            "dropout": self._dropout,
            "learning_rate": self._learning_rate,
            "momentum": self._momentum,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "random_state": self._random_state,
            "clip_gradient": self._clip_gradient,
            "verbose": self._verbose,
            "seed": self._seed,
            "threshold": self._threshold,
            "n_features": self._n_features,
            "n_classes": len(self._labels),
            "labels": list(self._labels),
            "class_index": dict(self._class_index),
            "tfidf_max_features": self._max_features,
            "tfidf_ngram_range": self._ngram_range,
            "tfidf_sublinear_tf": self._sublinear_tf,
            "tfidf_min_df": self._min_df,
            "tfidf_max_df": self._max_df,
            "tfidf_lowercase": self._lowercase,
            "tfidf_token_pattern": self._token_pattern,
            "tfidf_norm": self._norm,
            "tfidf_use_idf": self._use_idf,
            "tfidf_smooth_idf": self._smooth_idf,
            "min_training_documents": self._min_training_documents,
            "min_examples_per_class": self._min_examples_per_class,
            "require_all_response_types": self._require_all_response_types,
            "imbalance_warning_share": self._imbalance_warning_share,
            "drop_placeholder_tokens": self._drop_placeholder_tokens,
            "library_versions": _lib_versions(),
            "trained_at": self._timestamp(),
        }

    def _validate_threshold(self, value: Optional[float]) -> float:
        if value is None:
            return 0.5
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v
