"""
Base interfaces, configuration and input models for response classification.

Response classification asks one question: **what role does this comment play
in the discussion?**

    "same bro"                              -> agreement
    "I faced this last year"                -> personal_experience
    "I think this college is better"        -> opinion
    "you should contact the placement office" -> advice
    "apply for the scholarship"             -> solution
    "actually the fees aren't that high"    -> counterargument

How this differs from the stages before it
------------------------------------------
Each completed stage answers a different question, and none of them can answer
this one:

    language / script / code-mix : what language is this written in?
    preprocessing                : what does a clean version of it look like?
    topic discovery              : what is this discussion about?
    issue identification         : what problem is being complained about?
    sentiment / emotion          : how does the writer feel about it?
    response classification      : what is the writer DOING here?

Two distinctions are worth stating explicitly, because they are the ones that
get muddled in practice:

- **vs sentiment.** "apply for the scholarship" may be positive or negative
  depending on the writer's mood, and "same bro" carries almost no polarity at
  all.  Polarity and role are independent: the same role can be expressed
  happily or angrily.  So a response role is never derived from sentiment, and
  a sentiment value is never used as evidence for a role.
- **vs issues.** An issue label says *what is wrong*; a response role says *what
  the commenter is doing about it* - or not doing anything about it.  "Fees are
  too high" is an issue complaint; "same bro" is agreement; "actually the fees
  aren't that high compared with X" is a counterargument, and may well be about
  a completely different problem.

Why it follows issue identification: a role is only meaningful relative to
something being discussed.  Agreement and counterargument are inherently
dialogic - they respond to a claim - so the discussion (its topic and its
issues) is the substrate.  Both are carried on the input document as optional
metadata, and both are echoed back untouched.

Design decisions
----------------
- **The six labels are explicit.**  They are defined by the assignment, not
  discovered from the data, so they live in :mod:`.models.labels` and an
  annotation carrying a seventh value is refused.  This is intentionally the
  opposite of the annotation-driven label sets used upstream.
- **Softmax, not sigmoid.**  The six roles are mutually exclusive: a comment
  plays one primary role.  The probabilities therefore form a distribution
  summing to 1, and the prediction is the arg-max.  (A One-vs-Rest sigmoid
  ensemble would emit six independent scores that could all fire at once, which
  would answer a different question - "which roles does this comment touch?" -
  and would lose the competition between them.)
- **Context is carried but not used.**  ``parent_comment_text`` exists on the
  input model so a future context-aware classifier can be added without changing
  the public interface, but the V1 featuriser reads comment text alone.  Two
  comments differing only in their parent must receive identical predictions
  from a V1 model; a test enforces that rather than merely documenting it.
- **Input is ALREADY-PREPROCESSED text.**  This stage never imports or re-runs
  the preprocessing pipeline; upstream stages own normalisation.
- **No source text is modified.**  Every prediction echoes the document's text
  and metadata back unchanged.
- The abstract interface is the seam for a future neural classifier (a
  fine-tuned multilingual transformer with a six-way classification head, which
  could genuinely use the parent comment).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from app.services.nlp.response_classification.models.annotation import (
    ResponseAnnotation,
)
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponseTrainingReport,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class ResponseDocument(BaseModel):
    """
    One comment to be classified.

    ``text`` MUST already be preprocessed (i.e. the ``normalized_text`` produced
    by ``MultilingualTextPreprocessor``).  ``original_text`` is carried for
    traceability only and is never fed to the model.

    Thread context is modelled but optional.  ``parent_comment_text``,
    ``topic_id`` and ``issue_labels`` all default to ``None``/empty and are
    ignored by the V1 classifier, so training and prediction work on comment
    text alone; they are carried so that a future context-aware model can be
    introduced without changing this public interface.  Every one of them is
    echoed back on the prediction for traceability.
    """

    document_id: str
    text: str = ""

    # Source traceability
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None

    # Language-analysis metadata (from LanguageAnalysisService)
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    # Optional topic-discovery context
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_keywords: List[str] = Field(default_factory=list)

    # Optional issue-identification context
    issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)

    # Optional thread context. Present for future context-aware models; NOT read
    # by the V1 classifier, which uses comment text alone.
    parent_comment_text: Optional[str] = None

    @property
    def has_context(self) -> bool:
        """True when any optional context was supplied (not used by V1)."""
        return bool(
            self.parent_comment_text
            or self.topic_id is not None
            or self.issue_labels
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ResponseModelConfig(BaseModel):
    """
    Configuration for the TF-IDF + multiclass logistic-regression baseline.

    Multilingual defaults are deliberate and mirror the topic-discovery and
    sentiment/emotion stages:
    - ``stop_words`` defaults to ``None``.  No Hindi/Bengali/Odia stop-word
      removal is provided or applied by default, because in Romanized
      code-mixed text those function words carry the code-switch structure -
      and in a discussion-role task, short function-word comments ("same bro",
      "true that") are *signal*, not noise.
    - ``token_pattern`` keeps every script (``\\w`` under ``re.UNICODE``
      matches Devanagari / Bengali / Odia as well as Latin).
    - ``lowercase=True`` folds case so "Same" and "same" are one term.

    ``class_weight`` defaults to ``"balanced"``, unlike the upstream stages.
    See the note on the attribute for why that matters here.
    """

    # --- TF-IDF ---
    max_features: Optional[int] = 20000
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: Optional[Any] = None
    lowercase: bool = True
    sublinear_tf: bool = True
    token_pattern: str = r"(?u)\b\w+\b"
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True

    # --- Logistic regression (single multiclass model, softmax output) ---
    C: float = 4.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    #: Weight classes inversely to their frequency.  Set to ``None`` to disable.
    #: See the class docstring and docs/response_classification.md for why this
    #: stage defaults to ``"balanced"``.
    class_weight: Optional[str] = "balanced"
    random_state: int = 42

    # --- Decision rule ---
    #: Confidence floor on the winning class, NOT a class boundary: with a single
    #: softmax winner there is nothing for a threshold to choose between.  Below
    #: the floor the model abstains (``response_type=None``).
    threshold: float = 0.5

    # --- Data-adequacy guards (never ship a model trained on too little data) ---
    min_training_documents: int = 5
    min_examples_per_class: int = 2
    #: When True, ``train()`` refuses data that does not cover all six V1
    #: response types.  When False (the default) a missing class is reported in
    #: the training report and warned about, so an incrementally-annotated corpus
    #: can still be used - at the cost of a smaller probability vector.
    require_all_response_types: bool = False
    #: Above this share of the corpus held by a single class, the training
    #: report warns (it does not refuse) that the model may be near-trivial.
    imbalance_warning_share: float = 0.8

    # --- Preprocessing-stage placeholders (<URL>, <EMOJI_...>) ---
    drop_placeholder_tokens: bool = True


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ResponseClassification(ABC):
    """
    Abstract interface for single-label response classification.

    Any implementation (TF-IDF + multiclass logistic regression, a fine-tuned
    multilingual transformer with a six-way head, a prompt-based LLM
    classifier) must be trainable from :class:`ResponseAnnotation` data and must
    return a :class:`ResponseClassificationResult` carrying the full
    probability vector.

    The label space is the six V1 response types, not whatever the training data
    happens to contain.  Implementations may only emit a subset of them (the
    ones they were actually trained on), and must report the difference via the
    result's ``missing_response_types``.
    """

    @abstractmethod
    def train(
        self,
        annotations: Sequence[ResponseAnnotation],
    ) -> ResponseTrainingReport:
        """Fit the classifier over the six V1 response types."""
        ...

    @abstractmethod
    def predict(
        self,
        comments: Sequence[ResponseDocument],
        threshold: Optional[float] = None,
    ) -> ResponseClassificationResult:
        """Predict one response type per comment, with the full distribution."""
        ...

    @abstractmethod
    def evaluate(
        self,
        annotations: Sequence[ResponseAnnotation],
        threshold: Optional[float] = None,
    ):
        """Score predictions against gold annotations, withholding weak metrics."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"tfidf_logistic_multiclass"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """The response types this model can emit; empty before training."""

    @property
    @abstractmethod
    def expected_labels(self) -> List[str]:
        """All six V1 response types, whether or not this model can emit them."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        ...

    @property
    @abstractmethod
    def threshold(self) -> float:
        ...

    @property
    @abstractmethod
    def config(self) -> ResponseModelConfig:
        ...
