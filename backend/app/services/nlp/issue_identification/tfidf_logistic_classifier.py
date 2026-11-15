"""
Baseline multi-label issue classifier: TF-IDF -> One-vs-Rest Logistic Regression.

Why this shape
--------------
1. TF-IDF gives each document a sparse, L2-normalised feature vector, so
   documents of very different length (a one-line comment vs a long rant) are
   comparable, and corpus-wide filler terms are down-weighted by IDF.
2. Issue labels are NOT mutually exclusive — "fees are too high and ROI is
   questionable" is genuinely two issues.  A softmax would be wrong: it
   normalises over classes that are assumed to compete.  Instead we train one
   INDEPENDENT binary classifier per label (One-vs-Rest), each producing an
   independent sigmoid probability:

       p(y_k = 1 | x) = sigmoid(w_k^T x + b_k) = 1 / (1 + e^(-(w_k^T x + b_k)))

3. A label is emitted when its sigmoid output exceeds a configurable
   threshold.  Because the sigmoids are independent, several labels can fire
   on the same document — which is exactly the multi-label behaviour we want.

The label set is learned from ``IssueAnnotation`` data at ``train()`` time.
Nothing in this module knows what "fees" or "hostel" means, and no IIM issue
taxonomy is hard-coded; ``MultiLabelBinarizer`` derives the label space from
whatever the annotators actually used.

Determinism: TF-IDF is deterministic, ``lbfgs`` is deterministic, and
``random_state`` is pinned for any solver that would otherwise sample.
"""

import re
import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

from app.services.nlp.issue_identification.base import (
    IssueDocument,
    IssueIdentification,
    IssueModelConfig,
)
from app.services.nlp.issue_identification.models.annotation import (
    AnnotationCorpus,
    IssueAnnotation,
)
from app.services.nlp.issue_identification.models.evaluation import (
    IssueEvaluationReport,
    evaluate_predictions,
)
from app.services.nlp.issue_identification.models.result import (
    IssueIdentificationResult,
    IssuePrediction,
    IssueTrainingReport,
)

# Placeholders emitted by the preprocessing stage: <URL>, <MENTION>,
# <EMOJI_loudly_crying_face>, ...
_PLACEHOLDER_RE = re.compile(r"<[^<>]{1,64}>")

# Topic-derived features are injected with these prefixes so they can never
# collide with real vocabulary terms.
_TOPIC_FEATURE_PREFIX = "topicid"
_TOPIC_KEYWORD_PREFIX = "topickey"


class TfidfLogisticIssueClassifier(IssueIdentification):
    """
    Multi-label issue classifier over already-preprocessed text.

    Usage:
        classifier = TfidfLogisticIssueClassifier(threshold=0.5)
        report = classifier.train(annotations)          # labels learned here
        result = classifier.predict(issues)             # multi-label output
    """

    _NAME = "tfidf_logistic_multilabel"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[IssueModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._config = self._validate_config(config or IssueModelConfig())
        self._threshold = self._validate_threshold(
            self._config.threshold if threshold is None else threshold
        )
        # Opt-in only: a wall-clock field would break reproducibility.
        self._include_timestamp = include_timestamp

        self._vectorizer = self._build_vectorizer()
        self._binarizer: Optional[MultiLabelBinarizer] = None
        self._model: Optional[OneVsRestClassifier] = None
        self._trained = False
        self._training_summary: Optional[IssueTrainingReport] = None
        self._warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def labels(self) -> List[str]:
        """Issue labels learned from annotation data (empty before training)."""
        if self._binarizer is None:
            return []
        return [str(label) for label in self._binarizer.classes_]

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def config(self) -> IssueModelConfig:
        return self._config

    @property
    def training_report(self) -> Optional[IssueTrainingReport]:
        return self._training_summary

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        annotations: Sequence[IssueAnnotation],
        validation_split: bool = False,
    ) -> IssueTrainingReport:
        """
        Learn the issue label set and train one binary classifier per label.

        ``annotations`` are manually labelled examples.  Every label found in
        the data becomes a predictable label; there is no pre-defined list.
        """
        records = list(annotations or [])
        self._validate_training_data(records)

        if validation_split:
            records = self._hold_out(records)

        corpus = AnnotationCorpus(annotations=records)
        corpus.validate()

        labels_seen: List[List[str]] = [
            annotation.normalised_labels() for annotation in records
        ]
        binarizer = MultiLabelBinarizer()
        Y = binarizer.fit_transform(labels_seen)

        texts = [self._featurize(annotation) for annotation in records]

        # Reset any previous fit so train() is safely repeatable.
        self._vectorizer = self._build_vectorizer()
        self._model = None
        self._trained = False
        self._warnings = []

        try:
            X = self._vectorizer.fit_transform(texts)
        except ValueError as exc:
            raise ValueError(
                f"annotation texts produced no usable TF-IDF features: {exc}. "
                "Check the text content, or the min_df / ngram_range / "
                "stop_words configuration."
            ) from exc

        base = LogisticRegression(
            C=self._config.C,
            max_iter=self._config.max_iter,
            solver=self._config.solver,
            class_weight=self._config.class_weight,
            random_state=self._config.random_state,
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model = OneVsRestClassifier(base, n_jobs=1)
            model.fit(X, Y)
            self._model = model
        self._binarizer = binarizer
        self._trained = True

        for warning in caught:
            self._warnings.append(f"{warning.category.__name__}: {warning.message}")

        positive_counts = {
            str(label): int(Y[:, index].sum())
            for index, label in enumerate(binarizer.classes_)
        }

        report = IssueTrainingReport(
            model_name=self._NAME,
            model_version=self._VERSION,
            n_training_documents=len(records),
            n_labels=len(binarizer.classes_),
            labels=[str(label) for label in binarizer.classes_],
            positive_examples_per_label=positive_counts,
            n_features=int(len(self._vectorizer.vocabulary_)),
            n_documents_without_labels=sum(1 for labels in labels_seen if not labels),
            n_multi_label_documents=sum(1 for labels in labels_seen if len(labels) > 1),
            threshold=self._threshold,
            config=self.config_dict(),
            library_versions=self._library_versions(),
            trained_at=(
                datetime.now(timezone.utc).isoformat() if self._include_timestamp else None
            ),
            warnings=list(self._warnings),
        )
        self._training_summary = report

        duplicates = corpus.duplicate_document_ids()
        if duplicates:
            report.warnings.append(
                f"duplicate_document_ids in training data: {', '.join(duplicates)}; "
                "every row was used, so those documents are effectively up-weighted"
            )
        return report

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        issues: Sequence[IssueDocument],
        threshold: Optional[float] = None,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> IssueIdentificationResult:
        """
        Predict issue labels for each document.

        ``threshold`` overrides the model default for this call only.
        ``candidate_labels`` lets a caller pass the full taxonomy it expects;
        labels the model never trained on are reported in
        ``result.unseen_labels`` instead of being silently ignored.
        """
        self._require_trained()
        docs = list(issues or [])
        effective_threshold = self._validate_threshold(
            self._threshold if threshold is None else threshold
        )

        known = self.labels
        unseen = sorted(
            {
                str(label)
                for label in (candidate_labels or [])
                if str(label) not in known
            }
        )

        result_warnings: List[str] = list(self._warnings)
        if unseen:
            result_warnings.append(
                f"unseen_labels: {', '.join(unseen)} were supplied but never "
                "appear in the training annotations, so they cannot be predicted"
            )

        if not docs:
            return IssueIdentificationResult(
                model_name=self._NAME,
                model_version=self._VERSION,
                threshold=effective_threshold,
                predictions=[],
                n_documents=0,
                known_labels=known,
                unseen_labels=unseen,
                config=self.config_dict(),
                library_versions=self._library_versions(),
                trained_at=(
                    self._training_summary.trained_at
                    if self._training_summary
                    else None
                ),
                warnings=result_warnings,
            )

        texts = [self._featurize_document(doc) for doc in docs]
        X = self._vectorizer.transform(texts)
        probabilities = self._model.predict_proba(X)  # independent sigmoids

        predictions: List[IssuePrediction] = []
        for index, doc in enumerate(docs):
            row = probabilities[index]
            label_probabilities = {
                label: round(float(row[position]), 6)
                for position, label in enumerate(known)
            }
            # Deterministic ordering, and ties resolve to the lexicographically
            # first label because sorted() precedes the max() key.
            predicted = sorted(
                label
                for label, probability in label_probabilities.items()
                if probability > effective_threshold
            )
            confidence = round(max(label_probabilities.values()), 6) if label_probabilities else 0.0
            predictions.append(
                IssuePrediction(
                    document_id=doc.document_id,
                    predicted_issue_labels=predicted,
                    issue_probabilities=label_probabilities,
                    confidence=confidence,
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
                )
            )

        return IssueIdentificationResult(
            model_name=self._NAME,
            model_version=self._VERSION,
            threshold=effective_threshold,
            predictions=predictions,
            n_documents=len(docs),
            known_labels=known,
            unseen_labels=unseen,
            n_documents_without_issue=sum(
                1 for p in predictions if not p.predicted_issue_labels
            ),
            n_multi_label_documents=sum(
                1 for p in predictions if len(p.predicted_issue_labels) > 1
            ),
            config=self.config_dict(),
            library_versions=self._library_versions(),
            trained_at=(
                self._training_summary.trained_at if self._training_summary else None
            ),
            warnings=result_warnings,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        annotations: Sequence[IssueAnnotation],
        threshold: Optional[float] = None,
    ) -> IssueEvaluationReport:
        """
        Score predictions against gold annotations.

        Metrics are withheld (with a stated reason) when the evaluation set is
        too small — no fabricated numbers.
        """
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_predictions(
                np.zeros((0, len(self.labels))), np.zeros((0, len(self.labels))), self.labels
            )
        if self._binarizer is None:  # pragma: no cover - guarded by _require_trained
            raise RuntimeError("classifier binarizer missing")

        # Restrict the gold matrix to the labels this model actually knows, so
        # an evaluation set carrying extra labels is handled honestly.
        known = set(self.labels)
        gold: List[List[str]] = []
        for annotation in records:
            annotation.validate_labels()
            gold.append(sorted({label for label in annotation.labels if label in known}))

        y_true = self._binarizer.transform(gold)
        documents = [annotation.to_issue_document() for annotation in records]
        result = self.predict(documents, threshold=threshold)
        y_pred = self._binarizer.transform(
            [prediction.predicted_issue_labels for prediction in result.predictions]
        )
        return evaluate_predictions(y_true, y_pred, self.labels)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self._config.max_features,
            min_df=self._config.min_df,
            max_df=self._config.max_df,
            ngram_range=tuple(self._config.ngram_range),
            stop_words=self._config.stop_words,
            lowercase=self._config.lowercase,
            sublinear_tf=self._config.sublinear_tf,
            token_pattern=self._config.token_pattern,
            norm=self._config.norm,
            use_idf=self._config.use_idf,
            smooth_idf=self._config.smooth_idf,
        )

    def _featurize(self, annotation: IssueAnnotation) -> str:
        """Training-side view of a record (text plus optional topic features)."""
        text = self._strip_placeholders(annotation.text)
        if not self._config.use_topic_features:
            return text
        return self._append_topic_features(
            text, annotation.topic_id, []
        )

    def _featurize_document(self, document: IssueDocument) -> str:
        """Inference-side view of a record. Never mutates the input."""
        text = self._strip_placeholders(document.text)
        if not self._config.use_topic_features:
            return text
        return self._append_topic_features(
            text, document.topic_id, document.topic_keywords
        )

    def _strip_placeholders(self, text: str) -> str:
        """
        Build the string handed to the vectoriser, on a copy.

        <URL>, <MENTION> and <EMOJI_...> tokens are artefacts of the
        preprocessing stage.  Issue labels describe problems, not emotions, and
        these tokens appear across almost every document, so they add noise
        rather than signal.  Disable via
        ``IssueModelConfig(drop_placeholder_tokens=False)``.
        """
        value = text or ""
        if self._config.drop_placeholder_tokens:
            value = _PLACEHOLDER_RE.sub(" ", value)
        return value

    def _append_topic_features(
        self,
        text: str,
        topic_id: Optional[int],
        topic_keywords: Sequence[str],
    ) -> str:
        """
        Optionally inject weak topic-derived features.

        Prefixed tokens (``topicid3``, ``topickeyhostel``) cannot collide with
        ordinary vocabulary.  Off by default: topic ids are corpus-run
        specific and would not survive a re-fit of the topic model.
        """
        parts = [text]
        if topic_id is not None:
            parts.append(f"{_TOPIC_FEATURE_PREFIX}{topic_id}")
        for keyword in topic_keywords:
            cleaned = str(keyword).strip()
            if cleaned:
                parts.append(f"{_TOPIC_KEYWORD_PREFIX}{cleaned}")
        return " ".join(parts)

    def _hold_out(
        self, records: List[IssueAnnotation]
    ) -> List[IssueAnnotation]:
        """
        Drop labels that are too rare to train, then keep the documents that
        still have at least ``min_labels_per_document`` labels.

        Guarantees the remaining matrix is trainable (no all-zero column) while
        being fully deterministic: rarity is judged by frequency, then
        alphabetically, and documents are visited in input order.
        """
        counts: Dict[str, int] = {}
        for annotation in records:
            for label in set(annotation.normalised_labels()):
                counts[label] = counts.get(label, 0) + 1

        minimum = max(1, self._config.min_positive_examples_per_label)
        kept_labels = {
            label for label, count in counts.items() if count >= minimum
        }
        min_labels_per_document = min(2, len(kept_labels))
        return [
            annotation
            for annotation in records
            if len(kept_labels & set(annotation.normalised_labels()))
            >= min_labels_per_document
        ]

    def _require_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(
                f"{self._NAME} has not been trained; call train(annotations) "
                "with annotated issue examples before predict()/evaluate()"
            )
        check_is_fitted(self._model)

    def config_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob that affects the output."""
        return {
            "threshold": self._threshold,
            "tfidf": {
                "max_features": self._config.max_features,
                "min_df": self._config.min_df,
                "max_df": self._config.max_df,
                "ngram_range": list(self._config.ngram_range),
                "stop_words": self._config.stop_words,
                "lowercase": self._config.lowercase,
                "sublinear_tf": self._config.sublinear_tf,
                "token_pattern": self._config.token_pattern,
                "norm": self._config.norm,
                "use_idf": self._config.use_idf,
                "smooth_idf": self._config.smooth_idf,
            },
            "logistic_regression": {
                "C": self._config.C,
                "max_iter": self._config.max_iter,
                "solver": self._config.solver,
                "class_weight": self._config.class_weight,
                "random_state": self._config.random_state,
            },
            "data_guards": {
                "min_training_documents": self._config.min_training_documents,
                "min_positive_examples_per_label": (
                    self._config.min_positive_examples_per_label
                ),
            },
            "features": {
                "use_topic_features": self._config.use_topic_features,
                "drop_placeholder_tokens": self._config.drop_placeholder_tokens,
            },
        }

    @staticmethod
    def _library_versions() -> Dict[str, str]:
        return {"scikit-learn": sklearn.__version__, "numpy": np.__version__}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_training_data(self, records: List[IssueAnnotation]) -> None:
        """Reject training runs that cannot honestly produce a usable model."""
        if not records:
            raise ValueError(
                "no training data: train() needs IssueAnnotation examples. "
                "Populate data/annotations/issue_annotations.csv (see "
                "docs/issue_identification.md for the format)."
            )

        for record in records:
            if not isinstance(record, IssueAnnotation):
                raise TypeError(
                    "train() expects IssueAnnotation records, got "
                    f"{type(record).__name__}"
                )
            record.validate_labels()

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"insufficient training data: {len(records)} annotated document(s) "
                f"but at least {self._config.min_training_documents} are required. "
                "A multi-label model fitted on fewer documents than labels will "
                "memorise rather than generalise; annotate more examples first."
            )

        labels = sorted({label for r in records for label in r.normalised_labels()})
        if len(labels) < 2:
            raise ValueError(
                f"insufficient training data: only {len(labels)} distinct issue "
                "label(s) found. Issue identification is a multi-label task and "
                "needs at least 2 labels to be meaningful; annotate a broader "
                "sample of the corpus."
            )

        minimum = self._config.min_positive_examples_per_label
        counts: Dict[str, int] = {}
        for record in records:
            for label in set(record.normalised_labels()):
                counts[label] = counts.get(label, 0) + 1

        too_rare = sorted(
            label for label, count in counts.items() if count < minimum
        )
        if too_rare:
            detail = ", ".join(f"{label} ({counts[label]})" for label in too_rare)
            raise ValueError(
                f"insufficient training data: label(s) {detail} have fewer than "
                f"{minimum} positive example(s). Either annotate more examples "
                "for these labels or lower "
                "IssueModelConfig.min_positive_examples_per_label if the "
                "rarity is genuine."
            )

        # One-vs-Rest needs both classes present per label, otherwise the
        # binary sub-problem is degenerate and sklearn cannot fit it.
        n_documents = len(records)
        all_positive = sorted(label for label, count in counts.items() if count == n_documents)
        if all_positive:
            raise ValueError(
                f"degenerate training data: label(s) {', '.join(all_positive)} are "
                "attached to every annotated document, so no negative example "
                "exists and the binary classifier for them cannot be fitted."
            )

    @staticmethod
    def _validate_threshold(threshold: Any) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be a number, got {type(threshold).__name__}"
            )
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be strictly between 0 and 1, got {value}")
        return value

    @classmethod
    def _validate_config(cls, config: IssueModelConfig) -> IssueModelConfig:
        if config.max_features is not None and config.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {config.max_features}")

        for name, value in (("min_df", config.min_df), ("max_df", config.max_df)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an int or a float")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} as int must be >= 0, got {value}")
            if isinstance(value, float) and not (0.0 < value <= 1.0):
                raise ValueError(f"{name} as float must be in (0.0, 1.0], got {value}")

        low, high = config.ngram_range
        if low < 1 or high < low:
            raise ValueError(
                f"ngram_range must satisfy 1 <= min_n <= max_n, got {config.ngram_range}"
            )

        if config.norm not in ("l1", "l2", None):
            raise ValueError(f"norm must be 'l1', 'l2' or None, got {config.norm!r}")

        try:
            re.compile(config.token_pattern)
        except re.error as exc:
            raise ValueError(f"invalid token_pattern: {exc}") from exc

        stop = config.stop_words
        if stop is not None and not isinstance(stop, (str, list)):
            raise ValueError(
                f"stop_words must be None, 'english' or a list, got {type(stop).__name__}"
            )
        if isinstance(stop, str) and stop != "english":
            raise ValueError(
                "only the built-in 'english' stop-word list is supported (no "
                f"language-specific lists), got {stop!r}"
            )
        if isinstance(stop, list) and not all(isinstance(word, str) for word in stop):
            raise ValueError("stop_words list must contain only strings")

        if config.C <= 0:
            raise ValueError(f"C must be > 0 (inverse regularisation), got {config.C}")
        if config.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {config.max_iter}")
        if config.class_weight is not None and config.class_weight not in (
            "balanced",
        ):
            raise ValueError(
                f"class_weight must be None or 'balanced', got {config.class_weight!r}"
            )
        if config.min_training_documents < 2:
            raise ValueError(
                f"min_training_documents must be >= 2, got {config.min_training_documents}"
            )
        if config.min_positive_examples_per_label < 1:
            raise ValueError(
                "min_positive_examples_per_label must be >= 1, got "
                f"{config.min_positive_examples_per_label}"
            )
        cls._validate_threshold(config.threshold)
        return config
