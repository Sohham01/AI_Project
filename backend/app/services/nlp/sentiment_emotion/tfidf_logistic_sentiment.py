"""
Baseline single-label sentiment classifier: TF-IDF -> Logistic Regression.

Why this shape
--------------
1. TF-IDF gives each document a sparse, L2-normalised feature vector, so a
   one-line comment and a long rant are comparable, and corpus-wide filler
   terms are down-weighted by IDF.  ``ngram_range=(1, 2)`` matters more here
   than in topic modelling, because "not good" and "not bad" differ only by a
   bigram.
2. Sentiment is **mutually exclusive**: a text has one overall polarity, so
   the class probabilities are a normalised distribution and the prediction is
   the arg-max.  That is why this model is a single multiclass
   ``LogisticRegression`` and not a One-vs-Rest ensemble.  (The emotion model
   in the sibling module is the opposite case, and uses One-vs-Rest sigmoids.)
3. The threshold is a **confidence floor**, not a class boundary: there is
   only one winner, so a threshold cannot select between classes.  When the top
   class falls below the floor the model abstains (``sentiment=None``) instead
   of guessing or substituting a fallback class that the annotation data never
   taught it.

The class set is learned from ``SentimentAnnotation`` data at ``train()`` time.
Nothing here knows what "positive" or "neutral" means, and no opinion
lexicon is shipped: whatever the annotators used is the class set, and the
taxonomy is expected to change between annotation rounds.

Determinism: TF-IDF is deterministic, ``lbfgs`` is deterministic, classes are
sorted before encoding, and ``random_state`` is pinned for any solver that
would otherwise sample.
"""

import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from app.services.nlp.sentiment_emotion.base import (
    SentimentAnalysis,
    SentimentEmotionDocument,
    SentimentModelConfig,
)
from app.services.nlp.sentiment_emotion.models.annotation import (
    SentimentAnnotation,
    SentimentCorpus,
)
from app.services.nlp.sentiment_emotion.models.evaluation import (
    SentimentEvaluationReport,
    evaluate_sentiment,
)
from app.services.nlp.sentiment_emotion.models.result import (
    SentimentAnalysisResult,
    SentimentPrediction,
    SentimentTrainingReport,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_base import (
    TfidfLogisticTextModel,
    library_versions,
)


class TfidfLogisticSentimentClassifier(TfidfLogisticTextModel, SentimentAnalysis):
    """
    Single-label sentiment classifier over already-preprocessed text.

    Usage:
        classifier = TfidfLogisticSentimentClassifier(threshold=0.5)
        report = classifier.train(annotations)   # class set learned here
        result = classifier.predict(documents)  # one class per document
    """

    _NAME = "tfidf_logistic"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[SentimentModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._config = self._validate_config(config or SentimentModelConfig())
        self._threshold = self._validate_threshold(
            self._config.threshold if threshold is None else threshold
        )
        # Opt-in only: a wall-clock field would break reproducibility.
        self._include_timestamp = include_timestamp

        self._vectorizer = self._build_vectorizer()
        self._encoder: Optional[LabelEncoder] = None
        self._model: Optional[LogisticRegression] = None
        self._trained = False
        self._training_summary: Optional[SentimentTrainingReport] = None
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
        """Sentiment classes learned from annotation data (empty before training)."""
        if self._encoder is None:
            return []
        return [str(label) for label in self._encoder.classes_]

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def config(self) -> SentimentModelConfig:
        return self._config

    @property
    def training_report(self) -> Optional[SentimentTrainingReport]:
        return self._training_summary

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        annotations: Sequence[SentimentAnnotation],
    ) -> SentimentTrainingReport:
        """
        Learn the sentiment class set and fit a multiclass classifier.

        ``annotations`` are manually labelled examples.  Every class found in
        the data becomes a predictable class; there is no pre-defined list.
        """
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = SentimentCorpus(annotations=records)
        corpus.validate()

        encoder = LabelEncoder()
        encoder.fit([record.normalised_sentiment() for record in records])
        y = encoder.transform([record.normalised_sentiment() for record in records])

        texts = [
            self._featurize_text(record.text, record.topic_id, [])
            for record in records
        ]

        # Reset any previous fit so train() is safely repeatable.
        self._model = None
        self._trained = False
        self._warnings = []
        X = self._fit_vectorizer(texts)
        self._record_training_fingerprints(records)

        model = LogisticRegression(
            C=self._config.C,
            max_iter=self._config.max_iter,
            solver=self._config.solver,
            class_weight=self._config.class_weight,
            random_state=self._config.random_state,
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.fit(X, y)
            self._model = model
        self._encoder = encoder
        self._trained = True

        for warning in caught:
            self._warnings.append(f"{warning.category.__name__}: {warning.message}")

        class_names = [str(label) for label in encoder.classes_]
        counts = corpus.class_counts()
        majority_share = round(max(counts.values()) / len(records), 6)

        if majority_share > self._config.imbalance_warning_share:
            majority = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
            self._warnings.append(
                f"class imbalance: {majority!r} covers {majority_share:.0%} of the "
                "corpus, so accuracy would be inflated and a trivial "
                f"always-{majority!r} classifier would score highly. Consider "
                "SentimentModelConfig(class_weight='balanced') and weight macro-F1 "
                "over accuracy when judging this model."
            )

        report = SentimentTrainingReport(
            model_name=self._NAME,
            model_version=self._VERSION,
            n_training_documents=len(records),
            n_classes=len(encoder.classes_),
            labels=class_names,
            examples_per_class={name: counts.get(name, 0) for name in class_names},
            majority_class_share=majority_share,
            n_features=int(X.shape[1]),
            classes_per_document=1,
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
    ) -> SentimentAnalysisResult:
        """
        Predict one sentiment class per document.

        ``threshold`` overrides the confidence floor for this call only.  A
        document whose winning class does not clear the floor is reported with
        ``sentiment=None`` and ``abstained=True``; the full probability vector
        is still returned so a consumer can decide differently.
        """
        self._require_trained()
        docs = list(documents or [])
        floor = self._validate_threshold(
            self._threshold if threshold is None else threshold
        )

        result_warnings: List[str] = list(self._warnings)

        if not docs:
            return SentimentAnalysisResult(
                model_name=self._NAME,
                model_version=self._VERSION,
                threshold=floor,
                predictions=[],
                n_documents=0,
                known_sentiments=self.labels,
                config=self.config_dict(),
                library_versions=library_versions(),
                trained_at=(
                    self._training_summary.trained_at if self._training_summary else None
                ),
                warnings=result_warnings,
            )

        texts = [
            self._featurize_text(
                doc.text, doc.topic_id, doc.topic_keywords
            )
            for doc in docs
        ]
        X = self._vectorizer.transform(texts)
        probabilities = self._model.predict_proba(X)

        known = self.labels
        predictions: List[SentimentPrediction] = []
        for index, doc in enumerate(docs):
            row = probabilities[index]
            class_probabilities = {
                label: round(float(row[position]), 6)
                for position, label in enumerate(known)
            }
            # Deterministic arg-max: sorted() precedes max(), so a tie resolves
            # to the lexicographically first class.
            top = (
                max(sorted(class_probabilities.items()), key=lambda kv: kv[1])[0]
                if class_probabilities
                else None
            )
            confidence = class_probabilities.get(top, 0.0) if top else 0.0
            abstained = top is None or confidence <= floor
            predictions.append(
                SentimentPrediction(
                    document_id=doc.document_id,
                    sentiment=None if abstained else top,
                    sentiment_probabilities=class_probabilities,
                    confidence=round(confidence, 6),
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
                )
            )

        return SentimentAnalysisResult(
            model_name=self._NAME,
            model_version=self._VERSION,
            threshold=floor,
            predictions=predictions,
            n_documents=len(docs),
            known_sentiments=known,
            n_abstained=sum(1 for p in predictions if p.abstained),
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
        annotations: Sequence[SentimentAnnotation],
        threshold: Optional[float] = None,
    ) -> SentimentEvaluationReport:
        """
        Score predictions against gold annotations.

        Metrics are withheld (with a stated reason) when the evaluation set is
        too small or lacks class diversity — no fabricated numbers.
        """
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_sentiment([], [], self.labels)
        if self._encoder is None:  # pragma: no cover - guarded by _require_trained
            raise RuntimeError("classifier label encoder missing")

        # Restrict the gold classes to those this model actually knows, so an
        # evaluation set carrying an unseen class is handled honestly rather
        # than being scored against a class that cannot be produced.  The list
        # stays aligned with the predictions: a document whose gold class is
        # unseen is marked ``None``, which evaluate_sentiment() counts as a
        # miss.  Dropping the row instead would flatter the metrics.
        known = set(self.labels)
        gold: List[Optional[str]] = []
        dropped: List[str] = []
        for record in records:
            record.validate_sentiment()
            value = record.normalised_sentiment()
            if value in known:
                gold.append(value)
            else:
                gold.append(None)
                dropped.append(value)

        documents = [record.to_document() for record in records]
        result = self.predict(documents, threshold=threshold)
        predicted = [p.sentiment for p in result.predictions]

        report = evaluate_sentiment(gold, predicted, self.labels)
        if dropped:
            report.warnings.append(
                f"evaluation set contains class(es) the model was never trained "
                f"on ({', '.join(sorted(set(dropped)))}); those documents were "
                "scored against the classes the model can actually emit"
            )
        overlap = self._self_scoring_warning(records)
        if overlap:
            report.warnings.append(overlap)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(
                f"{self._NAME} has not been trained; call train(annotations) "
                "with annotated sentiment examples before predict()/evaluate()"
            )
        check_is_fitted(self._model)

    def _validate_training_data(self, records: List[SentimentAnnotation]) -> None:
        """Reject training runs that cannot honestly produce a usable model."""
        if not records:
            raise ValueError(
                "no training data: train() needs SentimentAnnotation examples. "
                "Populate data/annotations/sentiment_annotations.csv (see "
                "docs/sentiment_emotion.md for the format)."
            )

        for record in records:
            if not isinstance(record, SentimentAnnotation):
                raise TypeError(
                    "train() expects SentimentAnnotation records, got "
                    f"{type(record).__name__}"
                )
            record.validate_sentiment()

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"insufficient training data: {len(records)} annotated document(s) "
                f"but at least {self._config.min_training_documents} are required. "
                "A sentiment model fitted on a handful of documents will memorise "
                "rather than generalise; annotate more examples first."
            )

        classes = sorted({record.normalised_sentiment() for record in records})
        if len(classes) < 2:
            raise ValueError(
                f"insufficient label variation: only {len(classes)} sentiment "
                f"class(es) found ({', '.join(classes) if classes else 'none'}). "
                "Sentiment classification needs at least 2 classes to be "
                f"meaningful; annotate a broader sample of the corpus."
            )

        minimum = self._config.min_examples_per_class
        counts: Dict[str, int] = {}
        for record in records:
            counts[record.normalised_sentiment()] = (
                counts.get(record.normalised_sentiment(), 0) + 1
            )
        too_rare = sorted(name for name, count in counts.items() if count < minimum)
        if too_rare:
            detail = ", ".join(f"{name} ({counts[name]})" for name in too_rare)
            raise ValueError(
                f"insufficient examples per class: class(es) {detail} have fewer "
                f"than {minimum} example(s). A class with almost no evidence will "
                "be predicted arbitrarily; annotate more examples for these "
                "classes or lower SentimentModelConfig.min_examples_per_class if "
                "the rarity is genuine."
            )

    def config_dict(self) -> Dict[str, Any]:
        return self._config_dict(
            self._config,
            self._threshold,
            {
                "min_training_documents": self._config.min_training_documents,
                "min_examples_per_class": self._config.min_examples_per_class,
                "imbalance_warning_share": self._config.imbalance_warning_share,
            },
        )

    @classmethod
    def _validate_config(cls, config: SentimentModelConfig) -> SentimentModelConfig:
        cls._validate_shared_config(config)
        if config.min_examples_per_class < 1:
            raise ValueError(
                "min_examples_per_class must be >= 1, got "
                f"{config.min_examples_per_class}"
            )
        share = config.imbalance_warning_share
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ValueError("imbalance_warning_share must be a number")
        if not 0.0 < share <= 1.0:
            raise ValueError(
                f"imbalance_warning_share must be in (0.0, 1.0], got {share}"
            )
        return config
