"""
Baseline solution detector: TF-IDF -> binary Logistic Regression (sigmoid).

Why this shape
--------------
1. The target is **binary**, so the head is a single sigmoid rather than a
   six-way softmax.  Concretely: scikit-learn's two-class ``predict_proba``
   applies softmax over two columns, and for K=2 softmax is algebraically
   identical to a sigmoid::

       softmax([z0, z1])[1] = e^z1 / (e^z0 + e^z1)
                           = 1 / (1 + e^(z0 - z1))
                           = sigmoid(z1 - z0)

   so ``predict_proba(X)[:, 1]`` *is* ``sigma(w.x + b)`` for a single weight
   vector.  No separate head is needed and the documented mathematics is exact.

2. TF-IDF with ``ngram_range=(1, 2)`` because a large share of the actionable
   signal in these comments is phrase-shaped: "try to", "you should", "you must",
   "apply for", "contact the".  A unigram model reads "should" and "must" as
   generic words rather than as deontic markers of advice, and loses the class.

3. ``threshold`` is a **decision boundary**, unlike the multiclass stage where a
   threshold could only trigger abstention.  Here it is what turns a probability
   into the boolean a linker consumes::

       is_solution = p >= threshold

   Below it, ``is_solution`` is ``False`` because that is a real answer the
   corpus supports - not because the model is silent.  Pretending otherwise
   would make ``n_abstained`` mean "the negative class", which is not
   information.

4. Abstention is **opt-in** and lives *above* the boundary
   (``abstain_threshold``).  The band ``threshold <= p < abstain_threshold`` is
   where the model is positive-leaning but unconvincing - exactly the region in
   which an Issue-Solution link should not be asserted.  Default is ``None``,
   so evaluation is clean by default and a non-zero ``n_abstained`` means
   something.

5. ``class_weight="balanced"`` by default.  Most comments contain no solution,
   so an unweighted model converges on "no", reports high accuracy and finds
   nothing - the failure that would silently break Issue-Solution Linking
   downstream.  Balancing trades a little negative-class recall for positive-class
   recall.  The trade is configurable, because a corpus with a genuine majority of
   solutions is a legitimate case.

What is NOT used
----------------
- ``solution_type`` - free annotation data.  It is reported for coverage and
  never learned.  There is no category list in this package.
- ``parent_comment_text``, ``topic_*``, ``issue_*`` - carried and echoed, never
  read.  A linker will need the thread structure; this model must not quietly
  depend on it.
- ``response_type`` / ``response_probabilities`` - recorded for **candidate
  triage only**.  They never influence ``is_solution``; a test enforces that.

Determinism: TF-IDF is deterministic, ``lbfgs`` is deterministic, class indices
come from the fixed binary target order rather than from a fitted encoder, and
``random_state`` is pinned.  Two runs on the same data produce identical
probabilities.
"""

import re
import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_is_fitted

from app.services.nlp.solution_detection.base import (
    SolutionDetection,
    SolutionDocument,
    SolutionModelConfig,
)
from app.services.nlp.solution_detection.models.annotation import (
    DEFAULT_ANNOTATION_CSV,
    SolutionAnnotation,
    SolutionCorpus,
)
from app.services.nlp.solution_detection.models.candidate import (
    classify_response_candidate,
)
from app.services.nlp.solution_detection.models.evaluation import (
    SolutionEvaluationReport,
    evaluate_solution_detection,
)
from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    NEGATIVE_CLASS_INDEX,
    POSITIVE_CLASS_INDEX,
    TARGET_LABELS,
    target_index,
    target_label,
)
from app.services.nlp.solution_detection.models.result import (
    SolutionDetectionResult,
    SolutionPrediction,
    SolutionTrainingReport,
)

# Placeholders emitted by the preprocessing stage: <URL>, <MENTION>,
# <EMOJI_loudly_crying_face>, ...
PLACEHOLDER_RE = re.compile(r"<[^<>]{1,64}>")


class _Inherit:
    """Sentinel for 'not supplied' where ``None`` is itself a meaningful value.

    ``predict(abstain_threshold=None)`` must mean *disable abstention for this
    run*, so the parameter cannot use ``None`` as its own "unset" marker.  A
    dedicated sentinel keeps the two distinguishable; a class (not ``object()``)
    so the repr is readable if it ever leaks into an error message.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<inherit>"


INHERIT = _Inherit()


def library_versions() -> Dict[str, str]:
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__}


class TfidfLogisticSolutionClassifier(SolutionDetection):
    """
    Binary solution detector over already-preprocessed comment text.

    Usage::

        classifier = TfidfLogisticSolutionClassifier(threshold=0.5)
        report = classifier.train(annotations)      # refuses bad/insufficient data
        result = classifier.predict(comments)      # is_solution + probability
        metrics = classifier.evaluate(held_out)    # None where data is too thin

    Thread context (``parent_comment_text``, ``topic_id``, ``issue_labels``) is
    accepted, echoed back and deliberately **not** read: V1 tests the comment on
    its own text.  ``response_type`` is recorded for candidate triage and never
    influences the boolean.
    """

    _NAME = "tfidf_logistic_binary"
    _VERSION = "v1"

    def __init__(
        self,
        config: Optional[SolutionModelConfig] = None,
        threshold: Optional[float] = None,
        abstain_threshold: Optional[float] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._config = self._validate_config(config or SolutionModelConfig())
        self._threshold = self._validate_threshold(
            self._config.threshold if threshold is None else threshold
        )
        # None means "inherit the config"; an explicit value overrides it.
        self._abstain_threshold = self._validate_abstain_threshold(
            self._config.abstain_threshold if abstain_threshold is None else abstain_threshold
        )
        if (
            self._abstain_threshold is not None
            and self._abstain_threshold < self._threshold
        ):
            raise ValueError(
                f"abstain_threshold ({self._abstain_threshold}) must be >= threshold "
                f"({self._threshold}); the abstention band sits above the decision "
                "boundary by construction"
            )
        # Opt-in only: a wall-clock field would break reproducibility.
        self._include_timestamp = include_timestamp

        self._vectorizer = self._build_vectorizer()
        self._model: Optional[LogisticRegression] = None
        self._trained = False
        self._training_summary: Optional[SolutionTrainingReport] = None
        self._warnings: List[str] = []
        #: Fingerprints of the training documents, used only to detect
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
        """The two targets, in canonical order (empty before training)."""
        return list(TARGET_LABELS) if self._trained else []

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def abstain_threshold(self) -> Optional[float]:
        return self._abstain_threshold

    @property
    def config(self) -> SolutionModelConfig:
        return self._config

    @property
    def training_report(self) -> Optional[SolutionTrainingReport]:
        return self._training_summary

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        annotations: Sequence[SolutionAnnotation],
    ) -> SolutionTrainingReport:
        """
        Fit the binary detector over annotated ``is_solution`` values.

        ``solution_type`` is validated for shape and reported, never learned.
        Refuses an empty corpus, a missing label, a single-target corpus, a
        target with too little evidence, unusable vocabulary, and a corpus where
        one document carries two different answers.
        """
        records = list(annotations or [])
        self._validate_training_data(records)

        corpus = SolutionCorpus(annotations=records)
        corpus.validate()
        corpus.validate_unique_document_ids()

        # Fixed binary target order -> class indices.  Not a fitted encoder: the
        # label space cannot drift, so index 1 always means "solution".
        y = np.asarray([target_index(bool(r.is_solution)) for r in records])
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

        # sklearn encodes classes in ascending index order, which is the binary
        # target order used above.  Verify rather than assume: a silent
        # misalignment would invert the meaning of every probability.
        if [int(value) for value in model.classes_] != [
            NEGATIVE_CLASS_INDEX,
            POSITIVE_CLASS_INDEX,
        ]:
            raise ValueError(
                f"fitted model class order {list(model.classes_)} does not match the "
                f"expected binary order {[NEGATIVE_CLASS_INDEX, POSITIVE_CLASS_INDEX]}; "
                "the probability for 'solution' could not be attributed correctly"
            )

        for warning in caught:
            self._warnings.append(f"{warning.category.__name__}: {warning.message}")

        counts = corpus.class_counts()
        majority = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
        majority_share = round(max(counts.values()) / len(records), 6)

        if majority_share > self._config.imbalance_warning_share:
            self._warnings.append(
                f"class imbalance: {majority!r} covers {majority_share:.0%} of the "
                "corpus. Most comments contain no solution, so this skew is expected "
                f"in real data - but an always-{majority!r} model would score "
                f"{majority_share:.0%}. Judge this model on the positive-class F1 and "
                "recall, not accuracy."
            )

        # A positive with no category is a real solution whose type is undecided.
        # Not an error, but category coverage is unquotable until it is resolved.
        untyped = corpus.positives_without_type()
        if untyped:
            self._warnings.append(
                f"{len(untyped)} positive annotation(s) carry no solution_type "
                f"({', '.join(untyped[:5])}{'...' if len(untyped) > 5 else ''}); they "
                "are trained on normally, but any statement about solution-type "
                "coverage is incomplete until they are typed"
            )

        # A negative carrying a type is usually an annotator recording the
        # subject matter of a comment that is not a solution.  Legitimate, but it
        # is a signal the guideline needs to address.
        mislabelled = corpus.negatives_with_type()
        if mislabelled:
            self._warnings.append(
                f"{len(mislabelled)} annotation(s) have is_solution=false but a "
                f"solution_type ({', '.join(mislabelled[:5])}"
                f"{'...' if len(mislabelled) > 5 else ''}). If that means the "
                "annotator was recording the comment's subject rather than judging "
                "it a solution, the guideline should say so; solution_type is never "
                "learned either way"
            )

        report = SolutionTrainingReport(
            model_name=self._NAME,
            model_version=self._VERSION,
            n_training_documents=len(records),
            n_classes=2,
            labels=list(TARGET_LABELS),
            examples_per_class={name: counts.get(name, 0) for name in TARGET_LABELS},
            majority_class_share=majority_share,
            majority_class=majority,
            observed_solution_types=corpus.observed_solution_types(),
            solution_type_counts=corpus.solution_type_counts(),
            n_features=int(X.shape[1]),
            threshold=self._threshold,
            abstain_threshold=self._abstain_threshold,
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
        comments: Sequence[SolutionDocument],
        threshold: Optional[float] = None,
        abstain_threshold: Any = INHERIT,
    ) -> SolutionDetectionResult:
        """
        Decide whether each comment provides a solution.

        ``threshold`` overrides the decision boundary for this call only.

        ``abstain_threshold`` overrides the abstention band.  Omit it to use the
        configured value; pass ``None`` to disable abstention for one run without
        mutating the instance.  ``0`` is rejected rather than silently treated as
        "disabled", because an empty band is far more likely a mistake than an
        intention.
        """
        self._require_trained()
        docs = list(comments or [])
        floor = self._validate_threshold(
            self._threshold if threshold is None else threshold
        )
        band = (
            self._abstain_threshold
            if abstain_threshold is INHERIT
            else self._validate_abstain_threshold(abstain_threshold)
        )

        result_warnings: List[str] = list(self._warnings)

        if not docs:
            return SolutionDetectionResult(
                model_name=self._NAME,
                model_version=self._VERSION,
                threshold=floor,
                abstain_threshold=band,
                predictions=[],
                n_documents=0,
                config=self.config_dict(),
                library_versions=library_versions(),
                trained_at=(
                    self._training_summary.trained_at if self._training_summary else None
                ),
                warnings=result_warnings,
            )

        X = self._vectorizer.transform([self._featurize_text(doc.text) for doc in docs])
        probabilities = self._model.predict_proba(X)

        predictions: List[SolutionPrediction] = []
        for index, doc in enumerate(docs):
            row = probabilities[index]
            p_solution = round(float(row[POSITIVE_CLASS_INDEX]), 6)
            p_not = round(float(row[NEGATIVE_CLASS_INDEX]), 6)
            is_solution = p_solution >= floor
            # The band is the weak-positive region: the boundary says yes, but not
            # confidently enough to assert a link downstream.
            abstained = bool(
                is_solution and band is not None and p_solution < band
            )
            if abstained:
                is_solution = False
            predictions.append(
                SolutionPrediction(
                    document_id=doc.document_id,
                    is_solution=is_solution,
                    solution_probability=p_solution,
                    not_solution_probability=p_not,
                    threshold=floor,
                    abstained=abstained,
                    response_type=doc.response_type,
                    response_probabilities=dict(doc.response_probabilities),
                    # Triage only: this never feeds is_solution above.
                    is_candidate=classify_response_candidate(
                        doc.response_type
                    ).is_candidate,
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

        n_solutions = sum(1 for p in predictions if p.is_solution)
        n_candidates = sum(1 for p in predictions if p.is_candidate)
        n_candidates_not_solutions = sum(
            1 for p in predictions if p.is_candidate and not p.is_solution
        )
        # Only documents that actually carried a role can *disagree* with it.  A
        # comment with no response_type is unassessed, not contradicted, and
        # counting it here would raise a false alarm on every corpus where the
        # response stage has not been run yet.
        n_solutions_not_candidates = sum(
            1
            for p in predictions
            if p.is_solution and not p.is_candidate and p.response_type
        )

        # The two stage disagreements are the interesting cross-stage signal, so
        # they are surfaced rather than smoothed over.
        if n_candidates_not_solutions:
            result_warnings.append(
                f"{n_candidates_not_solutions} comment(s) were solution candidates "
                "by response role but the detector did not assert a solution. This is "
                "expected - advice is a candidate, not a guarantee - and the rate is "
                "worth watching, because a persistently high value means the two "
                "stages disagree about what counts as actionable."
            )
        if n_solutions_not_candidates:
            result_warnings.append(
                f"{n_solutions_not_candidates} comment(s) were asserted as "
                "solutions without being solution candidates by response role. Either "
                "the response role is mislabelled, or the comment is solution-shaped "
                "without being advice-shaped; worth adjudicating before linking."
            )

        return SolutionDetectionResult(
            model_name=self._NAME,
            model_version=self._VERSION,
            threshold=floor,
            abstain_threshold=band,
            predictions=predictions,
            n_documents=len(docs),
            n_solutions=n_solutions,
            n_not_solutions=len(docs) - n_solutions,
            n_abstained=sum(1 for p in predictions if p.abstained),
            n_candidates=n_candidates,
            n_candidates_not_solutions=n_candidates_not_solutions,
            n_solutions_not_candidates=n_solutions_not_candidates,
            n_documents_with_context=sum(1 for doc in docs if doc.has_context),
            n_documents_with_response_context=sum(
                1 for doc in docs if doc.has_response_context
            ),
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
        annotations: Sequence[SolutionAnnotation],
        threshold: Optional[float] = None,
    ) -> SolutionEvaluationReport:
        """
        Score predictions against gold annotations.

        Metrics are withheld (with a stated reason) when the evaluation set is
        too small or lacks one of the two targets - no fabricated numbers.
        """
        self._require_trained()
        records = list(annotations or [])
        if not records:
            return evaluate_solution_detection([], [], list(TARGET_LABELS))

        # Gold values are booleans by construction; anything else is treated as
        # unscoreable rather than coerced, so a malformed row cannot inflate a
        # metric.
        gold: List[Optional[bool]] = []
        malformed: List[str] = []
        for record in records:
            if isinstance(record.is_solution, bool):
                gold.append(bool(record.is_solution))
            else:
                gold.append(None)
                malformed.append(record.document_id)

        documents = [record.to_solution_document() for record in records]
        result = self.predict(documents, threshold=threshold)
        predicted: List[Optional[bool]] = [
            None if p.abstained else p.is_solution for p in result.predictions
        ]

        report = evaluate_solution_detection(gold, predicted, list(TARGET_LABELS))
        if malformed:
            report.warnings.append(
                f"evaluation set contains annotation(s) with a non-boolean "
                f"is_solution ({', '.join(malformed)}); they were counted as "
                "unscoreable rather than coerced to a boolean"
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
                "annotated is_solution values before predict()/evaluate(). The "
                f"targets are: {', '.join(TARGET_LABELS)}"
            )
        check_is_fitted(self._model)

    def _record_training_fingerprints(self, records: List[SolutionAnnotation]) -> None:
        """
        Remember which documents trained this model.

        Used exclusively to warn when ``evaluate()`` is handed the training data
        back, which reports memorisation rather than generalisation.
        """
        self._training_ids = frozenset(str(r.document_id) for r in records)
        self._training_texts = frozenset(str(r.text) for r in records)

    def _self_scoring_warning(self, records: List[SolutionAnnotation]) -> Optional[str]:
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

    def _build_vectorizer(self) -> Any:
        word_vectorizer = TfidfVectorizer(
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
        if not getattr(self._config, "use_char_ngrams", False):
            return word_vectorizer
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(self._config.char_ngram_range),
            max_features=self._config.char_max_features,
            min_df=self._config.min_df,
            max_df=self._config.max_df,
            lowercase=self._config.lowercase,
            sublinear_tf=self._config.sublinear_tf,
            norm=self._config.norm,
            use_idf=self._config.use_idf,
            smooth_idf=self._config.smooth_idf,
        )
        return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    def _strip_placeholders(self, text: str) -> str:
        """
        Build the string handed to the vectoriser, on a copy.

        ``<URL>``, ``<MENTION>`` and ``<EMOJI_...>`` are artefacts of the
        preprocessing stage.  They are stripped here because a thread's comments
        share URLs and mentions, so the tokens recur often enough to become
        high-order terms with no solution signal, and because ``<EMOJI_x>`` is a
        lossy proxy - one token can stand for many visually distinct emojis.
        Stripping is also a safety property: a comment that is *only* a URL
        cannot then masquerade as a training example.

        Note that the *document* keeps its placeholders: the text handed to the
        model is a copy, so ``original_text`` and ``text`` are returned untouched.
        Disable with ``SolutionModelConfig(drop_placeholder_tokens=False)``.
        """
        value = text or ""
        if self._config.drop_placeholder_tokens:
            value = PLACEHOLDER_RE.sub(" ", value)
        return value

    def _featurize_text(self, text: str) -> str:
        """
        Single featurisation path used by both training and inference.

        Comment text only.  ``parent_comment_text``, ``topic_*``, ``issue_*`` and
        ``response_type`` are deliberately not read here: V1 is a text-only
        detector, and the optional context fields exist so a context-aware model
        can be added later without changing the public interface.
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

    @staticmethod
    def _validate_abstain_threshold(value: Any) -> Optional[float]:
        """
        ``None`` disables abstention.  Anything else must be a real probability.

        ``0`` is refused on purpose: the band is ``threshold <= p <
        abstain_threshold``, so ``0`` would make it empty and the model would never
        abstain.  That is almost certainly a mistake, and silently accepting it
        would look like a working configuration.  Pass ``None`` to disable.
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "abstain_threshold must be a number or None, got "
                f"{type(value).__name__}"
            )
        number = float(value)
        if not 0.0 < number <= 1.0:
            hint = (
                " Pass None to disable abstention." if number == 0.0 else ""
            )
            raise ValueError(
                f"abstain_threshold must be in (0.0, 1.0] or None, got {number}.{hint}"
            )
        return number

    def _validate_training_data(self, records: List[SolutionAnnotation]) -> None:
        """Reject training runs that cannot honestly produce a usable model."""
        if not records:
            raise ValueError(
                "no training data: train() needs SolutionAnnotation examples. "
                f"Populate {DEFAULT_ANNOTATION_CSV} (see "
                "docs/solution_detection.md for the format). The target is "
                f"is_solution (true/false), and the canonical labels are: "
                f"{', '.join(TARGET_LABELS)}"
            )

        for record in records:
            if not isinstance(record, SolutionAnnotation):
                raise TypeError(
                    "train() expects SolutionAnnotation records, got "
                    f"{type(record).__name__}"
                )
            # Enforces a real boolean target and a usable text body.  A blank or
            # unrecognised is_solution is refused, never read as False.
            record.validate_annotation()

        if len(records) < self._config.min_training_documents:
            raise ValueError(
                f"insufficient training data: {len(records)} annotated comment(s) "
                f"but at least {self._config.min_training_documents} are required. "
                "A binary detector fitted on a handful of comments will memorise "
                "rather than generalise; annotate more examples first."
            )

        counts = {LABEL_NOT_SOLUTION: 0, LABEL_SOLUTION: 0}
        for record in records:
            counts[target_label(bool(record.is_solution))] += 1

        present = [name for name in TARGET_LABELS if counts[name] > 0]
        if len(present) < 2:
            found = ", ".join(present) or "none"
            raise ValueError(
                f"only one target class present: the corpus is {found} only "
                f"({counts[present[0]] if present else 0} annotation(s)). Solution "
                "detection is binary - a corpus of only solutions, or only "
                "non-solutions, cannot teach the difference, because the model "
                f"would have nothing to contrast against. Annotate both "
                f"{LABEL_NOT_SOLUTION} and {LABEL_SOLUTION} examples."
            )

        minimum = self._config.min_examples_per_class
        too_rare = sorted(name for name, count in counts.items() if count < minimum)
        if too_rare:
            detail = ", ".join(f"{name} ({counts[name]})" for name in too_rare)
            raise ValueError(
                f"insufficient examples per class: target(s) {detail} have fewer "
                f"than {minimum} example(s). A target with almost no evidence will "
                "be predicted arbitrarily; annotate more examples for it or lower "
                "SolutionModelConfig.min_examples_per_class if the rarity is "
                "genuine."
            )

    def config_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob that affects the output."""
        return {
            "threshold": self._threshold,
            "abstain_threshold": self._abstain_threshold,
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
                "head": "sigmoid (binary: two-class softmax is algebraically identical)",
            },
            "data_guards": {
                "min_training_documents": self._config.min_training_documents,
                "min_examples_per_class": self._config.min_examples_per_class,
                "imbalance_warning_share": self._config.imbalance_warning_share,
            },
            "features": {
                "drop_placeholder_tokens": self._config.drop_placeholder_tokens,
                "learns_is_solution": True,
                "learns_solution_type": False,
            },
            "context": {
                "uses_parent_comment": False,
                "uses_topic_metadata": False,
                "uses_issue_metadata": False,
                "uses_response_type_for_decision": False,
                "uses_response_type_for_candidate_triage": True,
            },
        }

    @classmethod
    def _validate_config(cls, config: SolutionModelConfig) -> SolutionModelConfig:
        if config.max_features is not None and config.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {config.max_features}")

        for name, value in (("min_df", config.min_df), ("max_df", config.max_df)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an int or a float")
            # An int is a document count, so 0 is not a meaningful floor (and
            # scikit-learn refuses it); a float is a proportion.
            if isinstance(value, int) and value < 1:
                raise ValueError(
                    f"{name} as an int must be >= 1 document, got {value}"
                )
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

        # No Hindi/Bengali/Odia stop-word lists are supplied.  In Romanized
        # code-mixed text those function words carry the code-switch structure,
        # so removing them by default would be a silent quality loss.  It would
        # also be a semantic loss here specifically: "you should", "you must" and
        # "try to" are the class's most reliable features.
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
                "function words, which are the actionable-signal markers in this "
                "task."
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
        if config.abstain_threshold is not None:
            cls._validate_abstain_threshold(config.abstain_threshold)
            if config.abstain_threshold < config.threshold:
                raise ValueError(
                    f"abstain_threshold ({config.abstain_threshold}) must be >= "
                    f"threshold ({config.threshold})"
                )
        return config
