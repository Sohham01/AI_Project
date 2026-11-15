"""
Base interfaces, configuration and result models for topic discovery.

Topic discovery answers the question: "what are the major recurring subjects
of discussion in this corpus?"  It does NOT answer "is this post
complaining about something?"  Issue identification is a separate stage that
consumes the topics produced here (see docs/topic_discovery.md).

Design decisions:
- Input is ALREADY-PREPROCESSED text.  This stage never re-runs the
  preprocessing pipeline; upstream stages own normalisation.
- Topic identity comes from the model, never from a hard-coded category list.
  A topic is "Topic <id>" plus the terms the model learned for it.
- Every result carries model name + version + configuration so a run can be
  reproduced or audited later.
- The abstract interface exists so an embedding-based implementation
  (Sentence-BERT + clustering, BERTopic) can be dropped in unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class TopicDocument(BaseModel):
    """
    One unit of text for topic modelling.

    ``text`` MUST already be preprocessed (i.e. it is the ``normalized_text``
    produced by MultilingualTextPreprocessor).  ``original_text`` is carried
    purely for traceability and is never used by the model.

    ``document_id`` is a stable identifier; ``post_id`` / ``comment_id`` keep
    the result mappable back to the Reddit source row.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TFIDFConfig(BaseModel):
    """
    TF-IDF vectoriser configuration.

    Multilingual defaults are deliberate:
    - ``stop_words`` defaults to ``None``.  Applying an English stop-word list
      to code-mixed text is a mixed blessing, and applying Hindi/Bengali/Odia
      stop-word lists to Romanized text is actively destructive (function
      words like "hai", "ka", "kya" carry the code-switch structure).  No
      language-specific stop-word removal happens unless asked for.
    - ``token_pattern`` keeps every script: ``\\w`` under the ``re.UNICODE``
      flag matches Devanagari / Bengali / Odia as well as Latin, so native-script
      terms survive into the vocabulary.
    - ``lowercase`` folds case so "IIM" and "iim" are the same term.  Domain
      acronyms (IIM, CAT, PPO, ROI, MBA) are never stop-words and therefore
      always survive as features.
    """

    max_features: Optional[int] = 20000
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: Optional[Any] = None  # None | "english" | list[str]
    lowercase: bool = True
    sublinear_tf: bool = True
    token_pattern: str = r"(?u)\b\w+\b"
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True


class NMFConfig(BaseModel):
    """
    Non-negative Matrix Factorisation configuration.

    ``init="nndsvda"`` uses a deterministic SVD-based initialisation, and
    ``random_state`` is pinned so that the stochastic fallback path is
    reproducible too.  With Frobenius loss and the CD solver the factorisation
    is fully deterministic run-to-run on the same input.
    """

    n_topics: int = 8
    n_keywords: int = 10
    init: str = "nndsvda"
    solver: str = "cd"
    beta_loss: str = "frobenius"
    random_state: int = 42
    max_iter: int = 300
    tol: float = 1e-4
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0

    # Drop <URL>, <MENTION>, <EMOJI_...> placeholder tokens before vectorising.
    # They are normalisation artifacts, not lexical content, and would
    # otherwise form artificial cross-corpus "topics".
    drop_placeholder_tokens: bool = True


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TopicInfo(BaseModel):
    """A single discovered topic."""

    topic_id: int
    label: str  # "Topic <id>" — derived from the model, never a hard-coded category
    keywords: List[str]
    keyword_weights: List[float]
    document_count: int
    document_ratio: float


class DocumentTopicAssignment(BaseModel):
    """
    Topic distribution for one document.

    ``topic_distribution`` maps topic id -> weight and sums to 1.0
    (NMF's W is L2-row-normalised after the factorisation).
    """

    document_id: str
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None
    text: str
    language: str
    script: str
    is_code_mixed: bool
    detected_languages: List[str] = Field(default_factory=list)

    topic_distribution: Dict[int, float]
    dominant_topic_id: int
    dominant_topic_label: str
    dominant_topic_score: float


class TopicDiscoveryResult(BaseModel):
    """Structured output of a topic discovery run."""

    topic_model: str
    topic_model_version: str

    n_topics: int
    n_documents: int
    vocabulary_size: int

    topics: List[TopicInfo]
    documents: List[DocumentTopicAssignment]

    # Reproducibility: every knob that can change the output.
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)

    discovered_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def get_document(self, document_id: str) -> Optional[DocumentTopicAssignment]:
        """Return the first assignment for ``document_id`` (or None)."""
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None

    def topic_by_id(self, topic_id: int) -> Optional[TopicInfo]:
        """Return the topic with ``topic_id`` (or None)."""
        for topic in self.topics:
            if topic.topic_id == topic_id:
                return topic
        return None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TopicDiscovery(ABC):
    """
    Abstract interface for automatic topic discovery.

    Any implementation (NMF, LDA, BERTopic, Sentence-BERT + clustering) must
    return a TopicDiscoveryResult and must derive topic labels from model
    output rather than a predefined category list.
    """

    @abstractmethod
    def discover(
        self,
        documents: Sequence[TopicDocument],
        n_topics: Optional[int] = None,
    ) -> TopicDiscoveryResult:
        """
        Discover topics over ``documents``.

        ``documents`` must already be preprocessed.  ``n_topics`` overrides the
        discoverer's default when provided.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model name, e.g. "nmf"."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string, e.g. "v1"."""
        ...
