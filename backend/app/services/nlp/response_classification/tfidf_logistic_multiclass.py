"""
Baseline response classifier: TF-IDF -> multiclass Logistic Regression.

Why this shape
--------------
1. TF-IDF gives each comment a sparse, L2-normalised feature vector, so
   "same bro" and a forty-line reply are comparable.  ``ngram_range=(1, 2)``
   matters more here than in topic modelling because a large share of response
   role is carried by function-word phrases: "same bro", "true that", "i agree
   with you", "actually though".  Those are the phrases a unigram model misses.
2. The six roles are **mutually exclusive**: a comment plays one primary role in
   the discussion it replies to.  So the class probabilities are a single
   normalised distribution and the prediction is the arg-max.  That is why this
   is one multiclass ``LogisticRegression`` with a softmax head, and not a
   One-vs-Rest sigmoid ensemble - six independent sigmoids would answer "which
   roles does this comment touch?" and would lose the competition between them.
3. ``threshold`` is a **confidence floor**, not a class boundary.  There is only
   one winner, so a threshold cannot select between classes.  When the top class
   falls below the floor the model abstains (``response_type=None``) instead of
   guessing, because a wrong role is worse downstream than an absent one: an
   "advice" comment mistaken for a "solution" would be counted as evidence that
   somebody answered the problem.
4. ``class_weight="balanced"`` by default.  In a real annotation corpus
   ``agreement`` and ``opinion`` are cheap to collect and hugely over-represented,
   while ``advice`` and ``solution`` are rarer.  Unweighted training therefore
   drifts towards the majority role and reports a high accuracy while being close
   to useless on the rare roles - which are the interesting ones.  Balancing
   trades a little majority-class accuracy for per-class recall; that trade is
   configurable, because sometimes a genuine majority role is worth optimising.

The label space is the six V1 response types from
:mod:`app.services.nlp.response_classification.models.labels` - fixed by the
assignment rather than discovered from the data.  A model only emits the types it
was actually trained on, and reports the rest as ``missing_response_types``.

Determinism: TF-IDF is deterministic, ``lbfgs`` is deterministic, class indices
come from the fixed taxonomy tuple rather than from a fitted encoder, and
``random_state`` is pinned for any solver that would otherwise sample.  Two runs
on the same data produce identical probability vectors.
"""

import re
import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from app.services.nlp.response_classification.base import (
    ResponseClassification,
    ResponseDocument,
    ResponseModelConfig,
)
from app.services.nlp.response_classification.models.annotation import (
    DEFAULT_ANNOTATION_CSV,
    ResponseAnnotation,
    ResponseCorpus,
)
from app.services.nlp.response_classification.models.evaluation import (
    ResponseEvaluationReport,
    evaluate_response_classification,
)
from app.services.nlp.response_classification.models.labels import (
    EXPECTED_NUMBER_OF_RESPONSE_TYPES,
    RESPONSE_TYPES,
    response_type_order,
)
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponsePrediction,
    ResponseTrainingReport,
)

# Placeholders emitted by the preprocessing stage: <URL>, <MENTION>,
# <EMOJI_loudly_crying_face>, ...
PLACEHOLDER_RE = re.compile(r"<[^<>]{1,64}>")


def library_versions() -> Dict[str, str]:
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__}


class TfidfLogisticResponseClassifier(ResponseClassification):
    """
    Six-class response-role classifier over already-preprocessed comment text.

    Usage:
        classifier = TfidfLogisticResponseClassifier(threshold=0.5)
        report = classifier.train(annotations)      # refuses bad/insufficient data
        result = classifier.predict(comments)      # one role + full distribution
        metrics = classifier.evaluate(held_out)    # None where data is too thin

    Thread context (``parent_comment_text``, ``topic_id``, ``issue_labels``) is
    accepted, echoed back and deliberately **not** read: V1 classifies the
    comment on its own text.
    """

    _NAME = "tfidf_logistic_multiclass"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[ResponseModelConfig] = None,
        threshold: Optional[float] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._config = self._validate_config(config or ResponseModelConfig())
        self._threshold = self._validate_threshold(
            self._config.threshold if threshold is None else threshold
        )
        # Opt-in only: a wall-clock field would break reproducibility.
        self._include_timestamp = include_timestamp

        self._vectorizer = self._build_vectorizer()
        self._model: Optional[LogisticRegression] = None
        self._class_index: Dict[str, int] = {}
        self._trained = False
        self._training_summary: Optional[ResponseTrainingReport] = None
        self._warnings: List[str] = []
        #: Fingerprints of the documents used for training, used only to detect
        #: self-scoring in evaluate().
        self._training_ids: frozenset = frozenset()
        self._training_texts: frozenset = frozenset()

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
        """Response types this model can emit; empty before training."""
        return list(self._class_index)

    @property
    def expected_labels(self) -> List[str]:
        """All six V1 response types, whether or not this model can emit them."""
        return list(RESPONSE_TYPES)

    @property
    def missing_response_types(self) -> List[str]:
        """V1 response types with no training data."""
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
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        annotations: Sequence[ResponseAnnotation],
    ) -> ResponseTrainingReport:
        """
        Fit the six-class model over the annotated response roles.

        Every label is checked against the fixed V1 taxonomy first, so a typo
        cannot become a seventh class.  Refuses an empty corpus, too few
        documents, a class with too little evidence, and a corpus where one
        document carries two different roles.
        """
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = ResponseCorpus(annotations=records)
        corpus.validate()
        corpus.validate_unique_document_ids()

        # Fixed taxonomy order -> class indices.  Not a fitted encoder: the label
        # space does not depend on the data, so the mapping cannot drift.
        present = set(corpus.labels)
        learned = response_type_order(present)
        self._class_index = {name: position for position, name in enumerate(learned)}
        y = np.asarray([self._class_index[r.normalised_response_type()] for r in records])

        texts = [self._featurize_text(record.text) for record in records]

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
        self._trained = True

        # sklearn encodes classes in ascending index order, which is the
        # taxonomy order used to build _class_index.  Verify rather than assume:
        # a silent misalignment would attach probabilities to the wrong roles.
        if [int(value) for value in model.classes_] != list(range(len(learned))):
            raise ValueError(
                f"fitted model class order {list(model.classes_)} does not match the "
                f"V1 taxonomy order {list(range(len(learned)))}; probabilities "
                "could not be attributed to the right response types"
            )

        for warning in caught:
            self._warnings.append(f"{warning.category.__name__}: {warning.message}")

        counts = corpus.class_counts()
        majority_share = round(max(counts.values()) / len(records), 6)
        majority = max(sorted(counts.items()), key=lambda kv: kv[1])[0]

        missing = corpus.missing_response_types()
        if missing:
            self._warnings.append(
                f"no training examples for {len(missing)} of "
                f"{EXPECTED_NUMBER_OF_RESPONSE_TYPES} response type(s) "
                f"({', '.join(missing)}); the model will never emit them and its "
                "probability vector will cover only the annotated types. Judge it "
                "on those classes alone, and annotate the rest before quoting a "
                "project-wide score."
            )

        if majority_share > self._config.imbalance_warning_share:
            self._warnings.append(
                f"class imbalance: {majority!r} covers {majority_share:.0%} of the "
                "corpus. Accuracy is inflated by that skew, and a trivial "
                f"always-{majority!r} classifier would score "
                f"{majority_share:.0%}. Judge this model on macro-F1 and per-class "
                "recall, not accuracy."
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
            n_features=int(len(self._vectorizer.vocabulary_)),
            classes_per_document=1,
            threshold=self._threshold,
            config=self.config_dict(),
            library_versions=library_versions(),
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
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        comments: Sequence[ResponseDocument],
        threshold: Optional[float] = None,
    ) -> ResponseClassificationResult:
        """
        Predict one response role per comment.

        ``threshold`` overrides the confidence floor for this call only.  A
        comment whose winning role does not clear the floor is reported with
        ``response_type=None`` and ``abstained=True``; the full probability vector
        is still returned so a consumer can decide differently without re-running
        inference.
        """
        self._require_trained()
        docs = list(comments or [])
        floor = self._validate_threshold(
            self._threshold if threshold is None else threshold
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
                config=self.config_dict(),
                library_versions=library_versions(),
                trained_at=(
                    self._training_summary.trained_at if self._training_summary else None
                ),
                warnings=result_warnings,
            )

        X = self._vectorizer.transform([self._featurize_text(doc.text) for doc in docs])
        probabilities = self._model.predict_proba(X)

        predictions: List[ResponsePrediction] = []
        for index, doc in enumerate(docs):
            row = probabilities[index]
            class_probabilities = {
                label: round(float(row[position]), 6)
                for position, label in enumerate(known)
            }
            # Deterministic arg-max: sorted() precedes max(), so a tie resolves
            # to the first class in canonical V1 order.
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
            unpredicted_response_types=[name for name in known if counts.get(name, 0) == 0],
            n_abstained=sum(1 for p in predictions if p.abstained),
            n_documents_with_context=sum(1 for doc in docs if doc.has_context),
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
        annotations: Sequence[ResponseAnnotation],
        threshold: Optional[float] = None,
    ) -> ResponseEvaluationReport:
        """
        Score predictions against gold annotations.

        Metrics are withheld (with a stated reason) when the evaluation set is
        too small or lacks class diversity - no fabricated numbers.
        """
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_response_classification([], [], self.labels)

        # Gold roles outside the model's class set are kept as None, which
        # evaluate_response_classification() counts as a miss.  Dropping those
        # rows instead would flatter the metrics.
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
                f"({', '.join(sorted(set(out_of_model)))}); those comments were scored "
                "against the types the model can actually predict"
            )
        overlap = self._self_scoring_warning(records)
        if overlap:
            report.warnings.append(overlap)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _timestamp(self) -> Optional[str]:
        return datetime.now(timezone.utc).isoformat() if self._include_timestamp else None

    def _require_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(
                f"{self._NAME} has not been trained; call train(annotations) with "
                f"annotated response roles before predict()/evaluate(). Expected "
                f"labels: {', '.join(RESPONSE_TYPES)}"
            )
        check_is_fitted(self._model)

    def _record_training_fingerprints(self, records: List[ResponseAnnotation]) -> None:
        """
        Remember which documents trained this model.

        Used exclusively to warn when ``evaluate()`` is handed the training data
        back, which reports memorisation rather than generalisation.
        """
        self._training_ids = frozenset(str(r.document_id) for r in records)
        self._training_texts = frozenset(str(r.text) for r in records)

    def _self_scoring_warning(self, records: List[ResponseAnnotation]) -> Optional[str]:
        """
        Warn when an evaluation set overlaps the training data.

        A model scored on its own training comments reports what it memorised,
        not what it can do, and on a small corpus that number is often a perfect
        1.0 - which is how synthetic metrics end up presented as project
        performance.  This is a warning rather than a refusal, because spotting
        the overlap is informative; dropping the rows silently is not.
        """
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
                "training data, so these metrics measure memorisation and are NOT "
                "held-out performance. Re-annotate a disjoint evaluation set before "
                "quoting any number from this run."
            )
        return (
            f"{overlap} of {len(records)} evaluation comment(s) ({share:.0%}) also "
            "appear in the training data, which inflates these metrics; they are "
            "not a clean held-out measurement."
        )

    # -- Featurisation ------------------------------------------------

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

    def _strip_placeholders(self, text: str) -> str:
        """
        Build the string handed to the vectoriser, on a copy.

        ``<URL>``, ``<MENTION>`` and ``<EMOJI_...>`` are artefacts of the
        preprocessing stage.  They are stripped here because a thread's comments
        share URLs and mentions, so the tokens recur often enough to become
        high-order terms with no role signal, and because ``<EMOJI_x>`` is a lossy
        proxy - one token can stand for many visually distinct emojis.

        Note that the *document* keeps its placeholders: the text handed to the
        model is a copy, so ``original_text`` and ``text`` are returned untouched.
        Disable with ``ResponseModelConfig(drop_placeholder_tokens=False)``.
        """
        value = text or ""
        if self._config.drop_placeholder_tokens:
            value = PLACEHOLDER_RE.sub(" ", value)
        return value

    def _featurize_text(self, text: str) -> str:
        """
        Single featurisation path used by both training and inference.

        Comment text only.  ``parent_comment_text``, ``topic_id`` and
        ``issue_labels`` are deliberately not read here: V1 is a text-only
        classifier, and the optional context fields exist so a context-aware
        model can be added later without changing the public interface.
        """
        return self._strip_placeholders(text)

    def _fit_vectorizer(self, texts: List[str]) -> Any:
        """Fit TF-IDF, converting sklearn's failure into an actionable error."""
        self._vectorizer = self._build_vectorizer()
        try:
            return self._vectorizer.fit_transform(texts)
        except ValueError as exc:
            raise ValueError(
                f"annotation texts produced no usable TF-IDF features: {exc}. "
                "Check the text content, or the min_df / ngram_range / "
                "stop_words configuration."
            ) from exc

    # -- Validation ----------------------------------------------------

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

    def _validate_training_data(self, records: List[ResponseAnnotation]) -> None:
        """Reject training runs that cannot honestly produce a usable model."""
        if not records:
            raise ValueError(
                "no training data: train() needs ResponseAnnotation examples. "
                f"Populate {DEFAULT_ANNOTATION_CSV} (see "
                "docs/response_classification.md for the format). Valid response "
                f"types: {', '.join(RESPONSE_TYPES)}"
            )

        for record in records:
            if not isinstance(record, ResponseAnnotation):
                raise TypeError(
                    "train() expects ResponseAnnotation records, got "
                    f"{type(record).__name__}"
                )
            # Also enforces the fixed six-type taxonomy.
            record.validate_response_type()

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"insufficient training data: {len(records)} annotated comment(s) "
                f"but at least {self._config.min_training_documents} are required. "
                "A six-class model fitted on a handful of comments will memorise "
                "rather than generalise; annotate more examples first."
            )

        counts: Dict[str, int] = {}
        for record in records:
            key = record.normalised_response_type()
            counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2:
            present = ", ".join(sorted(counts)) or "none"
            raise ValueError(
                f"insufficient label variation: only {len(counts)} response type(s) "
                f"found ({present}). Response classification needs at least 2 types "
                f"to be meaningful; annotate across the six ({', '.join(RESPONSE_TYPES)})."
            )

        minimum = self._config.min_examples_per_class
        too_rare = sorted(name for name, count in counts.items() if count < minimum)
        if too_rare:
            detail = ", ".join(f"{name} ({counts[name]})" for name in too_rare)
            raise ValueError(
                f"insufficient examples per class: response type(s) {detail} have "
                f"fewer than {minimum} example(s). A class with almost no evidence "
                "will be predicted arbitrarily; annotate more examples for these "
                "types or lower ResponseModelConfig.min_examples_per_class if the "
                "rarity is genuine."
            )

        if self._config.require_all_response_types:
            missing = [name for name in RESPONSE_TYPES if name not in counts]
            if missing:
                raise ValueError(
                    f"incomplete taxonomy: no training examples for "
                    f"{', '.join(missing)}. ResponseModelConfig."
                    "require_all_response_types=True, so all "
                    f"{EXPECTED_NUMBER_OF_RESPONSE_TYPES} response types must be "
                    "annotated before training."
                )

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
                "min_examples_per_class": self._config.min_examples_per_class,
                "require_all_response_types": self._config.require_all_response_types,
                "imbalance_warning_share": self._config.imbalance_warning_share,
            },
            "features": {
                "drop_placeholder_tokens": self._config.drop_placeholder_tokens,
            },
            "context": {
                "uses_parent_comment": False,
                "uses_topic_metadata": False,
                "uses_issue_metadata": False,
            },
        }

    @classmethod
    def _validate_config(cls, config: ResponseModelConfig) -> ResponseModelConfig:
        if config.max_features is not None and config.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {config.max_features}")

        for name, value in (("min_df", config.min_df), ("max_df", config.max_df)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an int or a float")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} as int must be >= 0, got {value}")
            if isinstance(value, float) and not (0.0 < value <= 1.0):
                raise ValueError(f"{name} as float must be in (0.0, 1.0], got {value}")

        # min_df is a floor and max_df is a ceiling; if the floor sits above the
        # ceiling the vectorizer can never keep a term, and the failure only
        # surfaces later as a confusing "empty vocabulary" error.
        numeric_df = [
            value
            for value in (config.min_df, config.max_df)
            if not isinstance(value, bool) and isinstance(value, (int, float))
        ]
        if len(numeric_df) == 2 and numeric_df[0] > numeric_df[1]:
            raise ValueError(
                f"min_df ({config.min_df}) must not exceed max_df ({config.max_df})"
            )

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

        # No Hindi/Bengali/Odia stop-word lists are supplied. In Romanized
        # code-mixed text those function words carry the code-switch structure,
        # so removing them by default would be a silent quality loss.
        stop = config.stop_words
        if stop is not None and not isinstance(stop, (str, list)):
            raise ValueError(
                f"stop_words must be None, 'english' or a list, got {type(stop).__name__}"
            )
        if isinstance(stop, str) and stop != "english":
            raise ValueError(
                "stop_words: only the built-in 'english' stop-word list is "
                "supported (no language-specific lists), got "
                f"{stop!r}. Leave stop_words=None to keep every script's "
                "function words."
            )
        if isinstance(stop, list) and not all(isinstance(word, str) for word in stop):
            raise ValueError("stop_words list must contain only strings")

        if config.C <= 0:
            raise ValueError(f"C must be > 0 (inverse regularisation), got {config.C}")
        if config.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {config.max_iter}")
        if config.class_weight is not None and config.class_weight != "balanced":
            raise ValueError(
                f"class_weight must be None or 'balanced', got {config.class_weight!r}"
            )
        if config.min_training_documents < 2:
            raise ValueError(
                f"min_training_documents must be >= 2, got {config.min_training_documents}"
            )
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
        cls._validate_threshold(config.threshold)
        return config
