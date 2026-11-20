"""
Base interfaces, configuration and input models for solution detection.

What this stage answers
-----------------------
    "Does this comment provide an actionable solution/recommendation?"

One question, one boolean.  Compare the response-classification stage, which asks
"what role does this comment play?" and answers with one of six roles.  Both run
on the same comments, both are needed, and neither substitutes for the other:

    comment                                     response role    is_solution
    ------------------------------------------  ---------------  -----------
    "same bro"                                  agreement        False
    "I faced this last year"                    personal_exp     False
    "I think the placement process is broken"   opinion          False
    "you should contact the placement office"   advice           ??? candidate
    "apply for the scholarship and submit the    solution         ???
     required documents"
    "actually fees aren't that high"            counterargument  False

The two ???s are the entire point.  ``advice`` and ``solution`` make a comment
eligible for this stage, but eligibility is not a verdict: only the trained
detector supplies ``is_solution``.  See :mod:`.models.candidate` for the
mechanism that keeps the two apart.

Relationship to issue-solution linking
--------------------------------------
This stage answers a per-comment question and stops there.  Deciding *which
issue* a detected solution answers is Issue-Solution Linking, which is a
different, later stage and is deliberately not implemented here.  ``post_id`` and
``issue_labels`` are therefore carried and echoed, never used as features: a
linker will need the thread structure, and this model must not quietly depend on
it.  Similarly, Resolution Detection (did the problem actually get fixed?) is out
of scope - a comment can propose a solution that nobody used.

Independence from the response stage
------------------------------------
``response_type`` and ``response_probabilities`` are accepted as optional
*neutral* context: this package defines its own
:class:`SolutionDocument` and never imports the response-classification
package.  A response stage is a separately deployable component, and coupling
the two would make either one impossible to replace.  An architecture test
enforces the absence of that import.

Design decisions
----------------
- **Sigmoid, because the target is binary.**  With K=2, scikit-learn's two-class
  softmax is algebraically identical to a single sigmoid, so
  ``predict_proba(X)[:, 1]`` *is* ``sigma(w.x + b)``.  No separate head is
  needed, and the documented mathematics is exact rather than approximate.
- **The threshold is a decision boundary.**  Unlike the multiclass stage, where a
  floor can only trigger abstention, here the threshold is what turns a
  probability into a boolean.  Below it, ``is_solution`` is ``False`` because
  that is a real answer the corpus supports - not because the model is silent.
- **Abstention is opt-in and lives above the boundary.**  ``abstain_threshold``,
  when set, covers the narrow band ``threshold <= p < abstain_threshold``: the
  boundary says yes, but not confidently enough to assert a link downstream.
  Default is ``None`` (never abstain), so evaluation is clean by default and
  ``n_abstained`` means something when it is non-zero.
- **``solution_type`` is never learned.**  It is free annotation data, reported
  for coverage and reserved for future solution extraction.  There is no
  hard-coded category list anywhere in this package.
- **Input is already-preprocessed text.**  This stage neither imports nor re-runs
  the preprocessing pipeline, and mutates nothing it is given.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from app.services.nlp.solution_detection.models.annotation import (
    SolutionAnnotation,
)
from app.services.nlp.solution_detection.models.result import (
    SolutionDetectionResult,
    SolutionTrainingReport,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class SolutionDocument(BaseModel):
    """
    One comment to be tested for a solution.

    ``text`` MUST already be preprocessed (the pipeline's ``normalized_text``).
    ``original_text`` is carried for traceability and is never fed to the model.

    Response context is optional and advisory.  ``response_type`` and
    ``response_probabilities`` are recorded on the prediction and used to compute
    *candidate* status, which is a triage signal - never an input to the boolean.
    Supplying a ``response_type`` of ``"advice"`` marks the comment as a
    candidate; it does not make ``is_solution`` true.

    Every field is treated as read-only: the classifier transforms text into a
    separate string and never writes back to the document.
    """

    document_id: str
    text: str = ""

    # Source traceability
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None

    # Language-analysis metadata (from the language stage)
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    # Optional topic-discovery context
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_keywords: List[str] = Field(default_factory=list)

    # Optional issue-identification context.  Carried for a future linker; the
    # detector never reads it, so no "did this solve that?" claim is implied.
    issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)

    # Optional response-classification context.  Plain, neutral data: this
    # package does not import the response stage and does not assume its taxonomy.
    response_type: Optional[str] = None
    response_probabilities: Dict[str, float] = Field(default_factory=dict)

    # Optional thread context.  Present for a future context-aware detector; NOT
    # read by V1, which uses comment text alone.
    parent_comment_text: Optional[str] = None

    @property
    def has_context(self) -> bool:
        """True when any optional context was supplied (not used by V1)."""
        return bool(
            self.parent_comment_text
            or self.topic_id is not None
            or self.issue_labels
        )

    @property
    def has_response_context(self) -> bool:
        """True when a response role was supplied by an upstream stage."""
        return bool(self.response_type and str(self.response_type).strip())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SolutionModelConfig(BaseModel):
    """
    Configuration for the TF-IDF + binary logistic-regression baseline.

    The multilingual defaults mirror the earlier stages, and the reasoning
    applies more strongly here than elsewhere: a solution detector leans on
    imperative and deontic markers, and those markers are exactly what an
    English stop-word list throws away.

    - ``stop_words`` defaults to ``None`` because "you should", "you must" and
      "try to" carry the actionable signal.  Removing them would delete the
      class's most reliable feature.
    - ``token_pattern`` keeps every script (``\\w`` under ``re.UNICODE`` matches
      Devanagari / Bengali / Odia as well as Latin), which is what lets a
      Bengali "আবেদন করুন" ("apply") reach the same vectoriser as its English
      counterpart.
    - ``lowercase=True`` folds case defensively.

    ``class_weight`` defaults to ``"balanced"``, matching the response stage.  It
    matters here for a specific reason: most comments contain no solution, so an
    unweighted model converges on "no" and looks accurate while being unable to
    find a single one - the failure mode that would silently break Issue-Solution
    Linking downstream.
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

    # --- Logistic regression (binary; two-class softmax == sigmoid) ---
    C: float = 4.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    #: Weight the two targets inversely to their frequency.  ``None`` disables.
    #: See the class docstring for why the default is "balanced".
    class_weight: Optional[str] = "balanced"
    random_state: int = 42

    # --- Decision rule ---
    #: Decision boundary on P(solution).  ``is_solution = p >= threshold``.
    #: Unlike the multiclass stage there is no class to select between: this is
    #: the probability that turns into a boolean.
    threshold: float = 0.5
    #: Opt-in abstention band.  When set (and >= threshold), a comment with
    #: ``threshold <= p < abstain_threshold`` is reported ``is_solution=False,
    #: abstained=True``: the boundary says yes, but not confidently enough to
    #: assert a link.  ``None`` (default) never abstains, which keeps
    #: ``n_abstained`` meaningful and evaluation clean.
    abstain_threshold: Optional[float] = None

    # --- Data-adequacy guards (never ship a model trained on too little data) ---
    min_training_documents: int = 10
    min_examples_per_class: int = 2
    #: Above this share of the corpus held by one target, warn (never refuse):
    #: the model may be near-trivial and accuracy meaningless.  0.8 rather than
    #: 0.9 because a 5:1 skew already lets an always-majority baseline dominate
    #: any report that quotes only accuracy.
    imbalance_warning_share: float = 0.8

    # --- Optional character n-gram features (code-mixed / Hinglish / Indic) ---
    use_char_ngrams: bool = False
    char_ngram_range: Tuple[int, int] = (3, 5)
    char_max_features: Optional[int] = 30000

    # --- Preprocessing-stage placeholders (<URL>, <EMOJI_...>) ---
    drop_placeholder_tokens: bool = True


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class SolutionDetection(ABC):
    """
    Abstract interface for binary solution detection.

    Any implementation (TF-IDF + logistic regression, a fine-tuned encoder with a
    one-unit head, a prompt-based LLM judge) must be trainable from
    :class:`SolutionAnnotation` data and must return a
    :class:`SolutionDetectionResult` carrying the full probability.

    Implementations learn ``is_solution`` alone.  ``solution_type`` is free
    annotation metadata, reported but never used as a target or a feature in V1.
    """

    @abstractmethod
    def train(
        self,
        annotations: Sequence[SolutionAnnotation],
    ) -> SolutionTrainingReport:
        """Fit the binary detector over annotated ``is_solution`` values."""
        ...

    @abstractmethod
    def predict(
        self,
        comments: Sequence[SolutionDocument],
        threshold: Optional[float] = None,
        abstain_threshold: Any = None,
    ) -> SolutionDetectionResult:
        """
        Predict whether each comment provides a solution, with the probability.

        Implementations may document their own sentinel for "abstain_threshold
        not supplied"; the baseline distinguishes it from an explicit ``None``
        (which disables abstention) via a module-level ``INHERIT`` sentinel.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        annotations: Sequence[SolutionAnnotation],
        threshold: Optional[float] = None,
    ) -> Any:
        """Score predictions against gold annotations, withholding weak metrics."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"tfidf_logistic_binary"``."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""
        ...

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
    def abstain_threshold(self) -> Optional[float]:
        ...

    @property
    @abstractmethod
    def config(self) -> SolutionModelConfig:
        ...
