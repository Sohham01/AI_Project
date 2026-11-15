"""
Base interfaces, configuration and input models for SENTIMENT AND EMOTION
analysis.

Sentiment and emotion are different dimensions and are modelled separately.

    Sentiment  : the polarity of the text      -> one mutually exclusive class
                 (conventionally positive / negative / neutral)
    Emotion    : the emotional state expressed  -> a *set* of labels, and only
                 where annotation data supports it

    "placement season is horrible"
        sentiment = negative
        emotion   = frustration
    "I was worried but ultimately it worked out well"
        sentiment = positive          (overall polarity)
        emotion   = anxiety           (an emotion is NOT implied by the polarity)

Two consequences drive the whole design:

1. **Emotion is never inferred from sentiment.** A model is not allowed to
   guess "negative => frustration".  Each dimension has its own annotation
   type, its own ABC, its own classifier and its own training run, so nothing
   can leak across.  The same document legitimately produces
   ``sentiment=negative, emotion=frustration`` *or*
   ``sentiment=negative, emotion=disappointment`` *or*
   ``sentiment=negative`` with no emotion at all, and only annotation data can
   decide which.

2. **The label sets are annotation-driven, never hard-coded.** No module here
   knows what "positive" or "frustration" means.  Sentiment classes are learned
   from ``SentimentAnnotation`` data and emotions from ``EmotionAnnotation``
   data, so the taxonomy is whatever the annotators agreed on and may evolve
   between annotation rounds.  Nothing in this package would let a
   domain-specific label bypass the annotation corpus.

Why this is a separate stage, after issue identification:
- Issue identification says *what problem* is being reported; sentiment says
  *how the writer feels about it*; emotion says *which feeling*.  A single
  complaint can be negative with no emotional label at all, so the three
  stages genuinely carry independent information.
- Sentiment is mutually exclusive by construction (a text has one overall
  polarity), so its probabilities are a normalised distribution.  Emotion is
  multi-label, so it uses independent One-vs-Rest sigmoids.  Modelling them
  with one shared head would force one of those two truths to be wrong.
- Emotion annotation is expensive and subjective, so it is the stage most
  likely to be launched with partial coverage.  Keeping it separate means a
  project can ship sentiment and add emotion later without disturbing it.

Input is ALREADY-preprocessed text.  This stage neither imports nor re-runs the
preprocessing pipeline, and never mutates raw text, ``normalized_text``, topic
output or issue predictions.  The abstract interfaces are the seam for future
neural implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from app.services.nlp.sentiment_emotion.models.annotation import (
    EmotionAnnotation,
    SentimentAnnotation,
)
from app.services.nlp.sentiment_emotion.models.result import (
    EmotionAnalysisResult,
    EmotionTrainingReport,
    SentimentAnalysisResult,
    SentimentTrainingReport,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class SentimentEmotionDocument(BaseModel):
    """
    One unit of text for sentiment and/or emotion analysis.

    ``text`` MUST already be preprocessed, i.e. the ``normalized_text``
    produced by the upstream pipeline.  ``original_text`` is carried for
    traceability only and is never fed to either model.

    Topic metadata and issue predictions are optional context.  They are
    carried through to the result so a downstream consumer can join the
    dimensions, but they are not used as features unless explicitly enabled,
    and they are never written back to an earlier stage's output.
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

    # Optional issue-identification output, carried for downstream joins only
    issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)

    @property
    def has_issue(self) -> bool:
        return len(self.issue_labels) > 0


# ---------------------------------------------------------------------------
# Shared feature configuration
# ---------------------------------------------------------------------------


class BaseTextFeatureConfig(BaseModel):
    """
    The TF-IDF + logistic-regression knobs shared by both dimensions.

    Defaults are deliberate and mirror the earlier NLP stages:

    - ``stop_words`` defaults to ``None``.  No Hindi/Bengali/Odia stop-word
      list is applied automatically, because in Romanized code-mixed text
      those function words carry the code-switch structure, and because a
      sentiment lexicon is not a stop-word list.  Only ``None``,
      ``"english"`` and an explicit user-supplied list are accepted.
    - ``token_pattern`` keeps every script (``\\w`` under ``re.UNICODE``
      matches Devanagari / Bengali / Odia as well as Latin).
    - ``lowercase=True`` folds case so ``PPO`` and ``ppo`` are one term.
    - ``ngram_range=(1, 2)`` keeps bigrams, which matter more here than in
      topic modelling: "not good" and "not bad" differ only by a bigram.
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

    # --- Logistic regression ---
    C: float = 4.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: Optional[str] = None
    random_state: int = 42

    # --- Decision rule ---
    threshold: float = 0.5

    # --- Data-adequacy guards (never ship a model trained on too little data) ---
    min_training_documents: int = 5

    # --- Features ---
    use_topic_features: bool = False
    drop_placeholder_tokens: bool = True


class SentimentModelConfig(BaseTextFeatureConfig):
    """
    Configuration for the single-label (mutually exclusive) sentiment model.

    ``threshold`` means something different here than in the emotion model: it
    is a **confidence floor** on the winning class, not a class boundary.
    Sentiment has exactly one winner, so a threshold cannot select between
    classes.  When the top class is below the floor the model *abstains*
    (``sentiment=None``) rather than inventing a class.
    """

    min_examples_per_class: int = 2
    #: Above this share of the corpus for a single class, the training report
    #: warns (it does not refuse) that the model is likely to be trivial.
    imbalance_warning_share: float = 0.8


class EmotionModelConfig(BaseTextFeatureConfig):
    """
    Configuration for the multi-label emotion model.

    ``threshold`` is the per-label decision boundary on each independent
    sigmoid, so several emotions can fire on one document.
    """

    min_positive_examples_per_label: int = 2


# ---------------------------------------------------------------------------
# Interfaces — one per dimension, on purpose
# ---------------------------------------------------------------------------


class SentimentAnalysis(ABC):
    """
    Abstract interface for single-label sentiment classification.

    Implementations answer one question per document: what is the overall
    polarity of this text?  The class set is learned from
    :class:`SentimentAnnotation` data at ``train()`` time.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"tfidf_logistic"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """Sentiment classes learned from annotation data; empty before training."""

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
    def config(self) -> SentimentModelConfig:
        ...

    @abstractmethod
    def train(
        self,
        annotations: Sequence[SentimentAnnotation],
    ) -> SentimentTrainingReport:
        """Learn the sentiment class set and fit the classifier."""

    @abstractmethod
    def predict(
        self,
        documents: Sequence[SentimentEmotionDocument],
        threshold: Optional[float] = None,
    ) -> SentimentAnalysisResult:
        """Predict one sentiment class per document."""

    @abstractmethod
    def evaluate(
        self,
        annotations: Sequence[SentimentAnnotation],
        threshold: Optional[float] = None,
    ):
        """Score predictions against gold annotations, withholding weak metrics."""


class EmotionAnalysis(ABC):
    """
    Abstract interface for multi-label emotion recognition.

    Implementations answer a different question: which emotional states does
    this text express?  A document may carry zero, one or many emotions, and
    the label set is learned from :class:`EmotionAnnotation` data.

    Nothing in this interface receives a sentiment prediction.  Emotion labels
    must come from emotion annotation, never from the polarity of the text.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"tfidf_logistic_multilabel"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """Emotion labels learned from annotation data; empty before training."""

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
    def config(self) -> EmotionModelConfig:
        ...

    @abstractmethod
    def train(
        self,
        annotations: Sequence[EmotionAnnotation],
    ) -> EmotionTrainingReport:
        """Learn the emotion label set and fit one binary classifier per label."""

    @abstractmethod
    def predict(
        self,
        documents: Sequence[SentimentEmotionDocument],
        threshold: Optional[float] = None,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> EmotionAnalysisResult:
        """Predict a set of emotions per document."""

    @abstractmethod
    def evaluate(
        self,
        annotations: Sequence[EmotionAnnotation],
        threshold: Optional[float] = None,
    ):
        """Score predictions against gold annotations, withholding weak metrics."""
