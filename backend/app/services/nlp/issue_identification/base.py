"""
Base interfaces, configuration and input models for issue identification.

Topic discovery answers "what is this discussion generally about?".
Issue identification answers "what specific problem, concern, complaint
or difficulty is being expressed?".

    "IIM life is amazing but the workload is killing me"
      topic = campus life / academics      (topic discovery)
      issue = workload                     (issue identification)

Why this is a separate stage, and why it follows topic discovery:
- A topic is a *subject*; an issue is a *grievance about that subject*. They
  are different label spaces with different lifecycles.
- Topics are discovered unsupervised from the corpus, so they change
  whenever the corpus changes.  Issue labels are a *taxonomy that humans
  define and annotate*, so they must be stable and versioned.
- Topic identity is model-derived ("Topic 3" + keywords); issue identity is
  annotation-derived ("fees", "roi", "placement_stress").
- Downstream analytics want both: "the hostel topic is dominated by the
  hostel + mess_food issues" is a far more useful sentence than either alone.

Design decisions:
- Issue labels are NOT hard-coded.  The label set is whatever the annotation
  data contains, learned at train() time.  The system never ships a
  pre-decided IIM issue taxonomy.
- Multi-label by construction.  "fees are too high and ROI is questionable"
  is genuinely two issues.  One-vs-Rest with sigmoid outputs is used
  precisely because softmax would force a mutually-exclusive choice.
- Input is ALREADY-PREPROCESSED text.  This stage never imports or re-runs
  the preprocessing pipeline; upstream stages own normalisation.
- Every result carries model name, version and threshold/configuration
  metadata so a prediction can be reproduced and audited later.
- The abstract interface is the seam for a future neural classifier
  (fine-tuned multilingual transformer with a multi-label head).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from app.services.nlp.issue_identification.models.annotation import IssueAnnotation
from app.services.nlp.issue_identification.models.result import (
    IssueIdentificationResult,
    IssueTrainingReport,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class IssueDocument(BaseModel):
    """
    One unit of text for issue identification.

    ``text`` MUST already be preprocessed (i.e. the ``normalized_text``
    produced by MultilingualTextPreprocessor).  ``original_text`` is carried
    for traceability only and is never fed to the model.

    Topic metadata is optional.  By default the model uses text alone; set
    ``IssueModelConfig.use_topic_features = True`` to also use ``topic_id`` /
    ``topic_keywords`` as weak features.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IssueModelConfig(BaseModel):
    """
    Configuration for the TF-IDF + One-vs-Rest logistic-regression baseline.

    Multilingual defaults are deliberate and mirror the topic-discovery stage:
    - ``stop_words`` defaults to ``None``.  No Hindi/Bengali/Odia stop-word
      removal is provided or applied by default, because in Romanized
      code-mixed text those function words carry the code-switch structure.
    - ``token_pattern`` keeps every script (``\\w`` under ``re.UNICODE``
      matches Devanagari / Bengali / Odia as well as Latin).
    - ``lowercase=True`` folds case so "PPO" and "ppo" are one term.
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

    # --- Logistic regression (per label, inside OneVsRestClassifier) ---
    C: float = 4.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: Optional[str] = None  # e.g. "balanced" for imbalanced labels
    random_state: int = 42

    # --- Decision rule ---
    threshold: float = 0.5

    # --- Data-adequacy guards (never ship a model trained on too little data) ---
    min_training_documents: int = 5
    min_positive_examples_per_label: int = 2

    # --- Optional character n-gram features (code-mixed / Hinglish / Indic) ---
    use_char_ngrams: bool = False
    char_ngram_range: Tuple[int, int] = (3, 5)
    char_max_features: Optional[int] = 30000

    # --- Optional topic-derived features ---
    use_topic_features: bool = False

    # --- Preprocessing-stage placeholders (<URL>, <EMOJI_...>) ---
    drop_placeholder_tokens: bool = True


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class IssueIdentification(ABC):
    """
    Abstract interface for multi-label issue identification.

    Any implementation (TF-IDF + OvR logistic regression, a fine-tuned
    multilingual transformer with a multi-label head, a prompt-based LLM
    classifier) must be trainable from ``IssueAnnotation`` data and must
    return an ``IssueIdentificationResult``.

    The label set is always derived from the annotation data, never from a
    hard-coded taxonomy.
    """

    @abstractmethod
    def train(self, annotations: Sequence[IssueAnnotation]) -> IssueTrainingReport:
        """Learn the issue label set and a multi-label model from annotations."""
        ...

    @abstractmethod
    def predict(
        self,
        issues: Sequence[IssueDocument],
        threshold: Optional[float] = None,
    ) -> IssueIdentificationResult:
        """Predict issue labels with per-label probabilities."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model name, e.g. "tfidf_logistic_multilabel"."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string, e.g. "v1"."""
        ...

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """The issue labels this model can predict (learned from annotations)."""
        ...

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        ...
