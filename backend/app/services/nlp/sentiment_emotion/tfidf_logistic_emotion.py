"""
Baseline multi-label emotion classifier: TF-IDF -> One-vs-Rest Logistic Regression.

Why this shape
--------------
1. TF-IDF gives each document a sparse, L2-normalised feature vector so a
   one-line comment and a long rant are comparable.
2. Emotions are **not** mutually exclusive.  A document can express several at
   once:

       "placement pressure hai aur hostel ka stress bhi hai"
           -> ["stress", "frustration"]

   A softmax would be wrong: it normalises over classes assumed to compete and
   forces the outputs to sum to 1, so a softmax head cannot express that
   example at all.  Instead one INDEPENDENT binary classifier is trained per
   emotion, each producing an independent sigmoid:

       p(y_k = 1 | x) = sigmoid(w_k^T x + b_k) = 1 / (1 + e^(-(w_k^T x + b_k)))

3. An emotion is emitted when its sigmoid output exceeds a configurable
   threshold.  Because the sigmoids are independent, several can fire on the
   same document — and none may, which is also correct, since a document may
   express no recognisable emotion.

**Emotion is never inferred from sentiment.**  This module has no sentiment
input, no sentiment labels and no sentiment model.  ``stress`` appears in the
output because ``EmotionAnnotation`` data said so for that document, never
because the text was negative.  A negative document legitimately yields no
emotion, or an emotion that is not the one sentiment would suggest.

The label set is learned from ``EmotionAnnotation`` data at ``train()`` time.
No emotion taxonomy is treated as domain truth in code: whatever the
annotators used is the label set, and it is expected to evolve between
annotation rounds.

Determinism: TF-IDF is deterministic, ``lbfgs`` is deterministic, labels are
sorted by ``MultiLabelBinarizer``, and ``random_state`` is pinned.
"""

import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

from app.services.nlp.sentiment_emotion.base import (
    EmotionAnalysis,
    EmotionModelConfig,
    SentimentEmotionDocument,
)
from app.services.nlp.sentiment_emotion.models.annotation import (
    EmotionAnnotation,
    EmotionCorpus,
)
from app.services.nlp.sentiment_emotion.models.evaluation import (
    EmotionEvaluationReport,
    evaluate_emotions,
)
from app.services.nlp.sentiment_emotion.models.result import (
    EmotionAnalysisResult,
    EmotionPrediction,
    EmotionTrainingReport,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_base import (
    TfidfLogisticTextModel,
    library_versions,
)


class TfidfLogisticEmotionClassifier(TfidfLogisticTextModel, EmotionAnalysis):
    """
    Multi-label emotion classifier over already-preprocessed text.

    Usage:
        classifier = TfidfLogisticEmotionClassifier(threshold=0.5)
        report = classifier.train(annotations)   # label set learned here
        result = classifier.predict(documents)   # emotion set per document
    """

    _NAME = "tfidf_logistic_multilabel"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[EmotionModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._config = self._validate_config(config or EmotionModelConfig())
        self._threshold = self._validate_threshold(
            self._config.threshold if threshold is None else threshold
        )
        # Opt-in only: a wall-clock field would break reproducibility.
        self._include_timestamp = include_timestamp

        self._vectorizer = self._build_vectorizer()
        self._binarizer: Optional[MultiLabelBinarizer] = None
        self._model: Optional[OneVsRestClassifier] = None
        self._per_label_thresholds: Dict[str, float] = {}
        self._trained = False
        self._training_summary: Optional[EmotionTrainingReport] = None
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
        """Emotion labels learned from annotation data (empty before training)."""
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
    def config(self) -> EmotionModelConfig:
        return self._config

    @property
    def training_report(self) -> Optional[EmotionTrainingReport]:
        return self._training_summary

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        annotations: Sequence[EmotionAnnotation],
    ) -> EmotionTrainingReport:
        """
        Learn the emotion label set and fit one binary classifier per label.

        ``annotations`` are manually labelled examples.  Every label found in
        the data becomes a predictable emotion; there is no pre-defined list and
        no opinion lexicon is consulted.
        """
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = EmotionCorpus(annotations=records)
        corpus.validate()

        labels_seen: List[List[str]] = [
            annotation.normalised_emotions() for annotation in records
        ]
        binarizer = MultiLabelBinarizer()
        Y = binarizer.fit_transform(labels_seen)

        texts = [
            self._featurize_text(annotation.text, annotation.topic_id, [])
            for annotation in records
        ]

        # Reset any previous fit so train() is safely repeatable.
        self._model = None
        self._per_label_thresholds = {}
        self._trained = False
        self._warnings = []
        X = self._fit_vectorizer(texts)
        self._record_training_fingerprints(records)

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

        if corpus.n_multi_label_documents == 0:
            self._warnings.append(
                "no training document carries more than one emotion, so the "
                "multi-label path (several sigmoids above threshold on one "
                "document) is untested by this corpus"
            )

        positive_counts = {
            str(label): int(Y[:, index].sum())
            for index, label in enumerate(binarizer.classes_)
        }

        report = EmotionTrainingReport(
            model_name=self._NAME,
            model_version=self._VERSION,
            n_training_documents=len(records),
            n_labels=len(binarizer.classes_),
            labels=[str(label) for label in binarizer.classes_],
            positive_examples_per_label=positive_counts,
            n_features=int(X.shape[1]),
            n_documents_without_emotions=sum(1 for labels in labels_seen if not labels),
            n_multi_label_documents=sum(1 for labels in labels_seen if len(labels) > 1),
            threshold=self._threshold,
            config=self.config_dict(),
            library_versions=library_versions(),
            trained_at=(
                datetime.now(timezone.utc).isoformat() if self._include_timestamp else None
            ),
            warnings=list(self._warnings),
        )
        self._training_summary = report

        duplicates = corpus.duplicate_document_ids()
        if duplicates:
            report.warnings.append(
                f"duplicate document_ids in training data: {', '.join(duplicates)}; "
                "every row was used, so those documents are effectively up-weighted"
            )
        return report

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        documents: Sequence[SentimentEmotionDocument],
        threshold: Optional[float] = None,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> EmotionAnalysisResult:
        """
        Predict a set of emotions per document.

        ``threshold`` overrides the per-label decision boundary for this call
        only.  ``candidate_labels`` lets a caller pass the full taxonomy it
        expects; labels the model never trained on are reported in
        ``result.unseen_labels`` instead of being silently ignored.
        """
        self._require_trained()
        docs = list(documents or [])
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
            return EmotionAnalysisResult(
                model_name=self._NAME,
                model_version=self._VERSION,
                threshold=effective_threshold,
                predictions=[],
                n_documents=0,
                known_emotions=known,
                unseen_labels=unseen,
                config=self.config_dict(),
                library_versions=library_versions(),
                trained_at=(
                    self._training_summary.trained_at if self._training_summary else None
                ),
                warnings=result_warnings,
            )

        texts = [
            self._featurize_text(doc.text, doc.topic_id, doc.topic_keywords)
            for doc in docs
        ]
        X = self._vectorizer.transform(texts)
        probabilities = self._model.predict_proba(X)  # independent sigmoids

        predictions: List[EmotionPrediction] = []
        for index, doc in enumerate(docs):
            row = probabilities[index]
            label_probabilities = {
                label: round(float(row[position]), 6)
                for position, label in enumerate(known)
            }
            # Deterministic ordering; ties resolve to the lexicographically
            # first label because sorted() precedes the comparison.
            predicted = sorted(
                label
                for label, probability in label_probabilities.items()
                if probability > (
                    effective_threshold
                    if threshold is not None
                    else self._per_label_thresholds.get(label, effective_threshold)
                )
            )
            confidence = (
                round(max(label_probabilities.values()), 6)
                if label_probabilities
                else 0.0
            )
            predictions.append(
                EmotionPrediction(
                    document_id=doc.document_id,
                    emotions=predicted,
                    emotion_probabilities=label_probabilities,
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
                    topic_keywords=list(doc.topic_keywords),
                    issue_labels=list(doc.issue_labels),
                    issue_probabilities=dict(doc.issue_probabilities),
                )
            )

        return EmotionAnalysisResult(
            model_name=self._NAME,
            model_version=self._VERSION,
            threshold=effective_threshold,
            predictions=predictions,
            n_documents=len(docs),
            known_emotions=known,
            unseen_labels=unseen,
            n_documents_without_emotion=sum(
                1 for p in predictions if not p.emotions
            ),
            n_multi_label_documents=sum(1 for p in predictions if len(p.emotions) > 1),
            config=self.config_dict(),
            library_versions=library_versions(),
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
        annotations: Sequence[EmotionAnnotation],
        threshold: Optional[float] = None,
    ) -> EmotionEvaluationReport:
        """
        Score predictions against gold annotations.

        Metrics are withheld (with a stated reason) when the evaluation set is
        too small — no fabricated numbers.
        """
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_emotions(
                np.zeros((0, len(self.labels))),
                np.zeros((0, len(self.labels))),
                self.labels,
            )
        if self._binarizer is None:  # pragma: no cover - guarded by _require_trained
            raise RuntimeError("classifier binarizer missing")

        # Restrict the gold matrix to the labels this model actually knows, so
        # an evaluation set carrying extra labels is handled honestly.
        known = set(self.labels)
        gold: List[List[str]] = []
        for annotation in records:
            annotation.validate_emotions()
            gold.append(sorted({label for label in annotation.emotions if label in known}))

        y_true = self._binarizer.transform(gold)
        documents = [annotation.to_document() for annotation in records]
        result = self.predict(documents, threshold=threshold)
        y_pred = self._binarizer.transform(
            [prediction.emotions for prediction in result.predictions]
        )
        report = evaluate_emotions(y_true, y_pred, self.labels)
        overlap = self._self_scoring_warning(records)
        if overlap:
            report.warnings.append(overlap)
        return report

    @property
    def per_label_thresholds(self) -> Dict[str, float]:
        """Optional validation-calibrated thresholds keyed by emotion label."""
        return dict(self._per_label_thresholds)

    def calibrate_thresholds(
        self,
        validation_annotations: Sequence[EmotionAnnotation],
        grid: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Tune per-label sigmoid thresholds on a held-out validation split."""
        self._require_trained()
        records = list(validation_annotations or [])
        if not records or self._binarizer is None:
            return dict(self._per_label_thresholds)

        candidates = [
            self._validate_threshold(value)
            for value in (
                grid
                if grid is not None
                else [round(0.15 + 0.05 * i, 2) for i in range(15)]
            )
        ]
        known = set(self.labels)
        gold: List[List[str]] = []
        for annotation in records:
            annotation.validate_emotions()
            gold.append(sorted({label for label in annotation.emotions if label in known}))
        y_true = self._binarizer.transform(gold)
        docs = [annotation.to_document() for annotation in records]
        texts = [
            self._featurize_text(doc.text, doc.topic_id, doc.topic_keywords)
            for doc in docs
        ]
        X = self._vectorizer.transform(texts)
        probabilities = self._model.predict_proba(X)

        tuned: Dict[str, float] = {}
        for col, label in enumerate(self.labels):
            truth = y_true[:, col]
            if int(truth.sum()) == 0:
                tuned[label] = self._threshold
                continue
            probs = probabilities[:, col]
            best_tau = self._threshold
            best_f1 = -1.0
            for tau in candidates:
                pred = (probs > tau).astype(int)
                tp = int(((truth == 1) & (pred == 1)).sum())
                fp = int(((truth == 0) & (pred == 1)).sum())
                fn = int(((truth == 1) & (pred == 0)).sum())
                denom = 2 * tp + fp + fn
                f1 = (2.0 * tp / denom) if denom > 0 else 0.0
                if (f1 > best_f1 + 1e-12) or (
                    abs(f1 - best_f1) <= 1e-12
                    and abs(tau - self._threshold) < abs(best_tau - self._threshold)
                ):
                    best_f1 = f1
                    best_tau = tau
            tuned[label] = best_tau

        self._per_label_thresholds = tuned
        return dict(tuned)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(
                f"{self._NAME} has not been trained; call train(annotations) "
                "with annotated emotion examples before predict()/evaluate()"
            )
        check_is_fitted(self._model)

    def _validate_training_data(self, records: List[EmotionAnnotation]) -> None:
        """Reject training runs that cannot honestly produce a usable model."""
        if not records:
            raise ValueError(
                "no training data: train() needs EmotionAnnotation examples. "
                "Populate data/annotations/emotion_annotations.csv (see "
                "docs/sentiment_emotion.md for the format)."
            )

        for record in records:
            if not isinstance(record, EmotionAnnotation):
                raise TypeError(
                    "train() expects EmotionAnnotation records, got "
                    f"{type(record).__name__}"
                )
            record.validate_emotions()

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"insufficient training data: {len(records)} annotated document(s) "
                f"but at least {self._config.min_training_documents} are required. "
                "A multi-label model fitted on fewer documents than labels will "
                "memorise rather than generalise; annotate more examples first."
            )

        labels = sorted({label for r in records for label in r.normalised_emotions()})
        if len(labels) < 2:
            raise ValueError(
                f"insufficient label variation: only {len(labels)} emotion label(s) "
                "found. Emotion recognition is multi-label, so at least 2 labels "
                "are required for it to be meaningful; annotate a broader sample "
                "of the corpus. Note that emotion labels must come from emotion "
                "annotation - they are never derived from sentiment."
            )

        minimum = self._config.min_positive_examples_per_label
        counts: Dict[str, int] = {}
        for record in records:
            for label in set(record.normalised_emotions()):
                counts[label] = counts.get(label, 0) + 1

        too_rare = sorted(label for label, count in counts.items() if count < minimum)
        if too_rare:
            detail = ", ".join(f"{label} ({counts[label]})" for label in too_rare)
            raise ValueError(
                f"insufficient examples per label: label(s) {detail} have fewer "
                f"than {minimum} positive example(s). Either annotate more examples "
                "for these labels or lower "
                "EmotionModelConfig.min_positive_examples_per_label if the rarity "
                "is genuine."
            )

        # One-vs-Rest needs both classes present per label, otherwise the binary
        # sub-problem is degenerate and sklearn cannot fit it.
        n_documents = len(records)
        all_positive = sorted(label for label, count in counts.items() if count == n_documents)
        if all_positive:
            raise ValueError(
                f"degenerate training data: label(s) {', '.join(all_positive)} are "
                "attached to every annotated document, so no negative example "
                "exists and the binary classifier for them cannot be fitted."
            )

    def config_dict(self) -> Dict[str, Any]:
        return self._config_dict(
            self._config,
            self._threshold,
            {
                "min_training_documents": self._config.min_training_documents,
                "min_positive_examples_per_label": (
                    self._config.min_positive_examples_per_label
                ),
            },
        )

    @classmethod
    def _validate_config(cls, config: EmotionModelConfig) -> EmotionModelConfig:
        cls._validate_shared_config(config)
        if config.min_positive_examples_per_label < 1:
            raise ValueError(
                "min_positive_examples_per_label must be >= 1, got "
                f"{config.min_positive_examples_per_label}"
            )
        return config
