"""
Baseline topic discovery: TF-IDF -> NMF -> topic keywords -> document topics.

Why this pipeline
-----------------
1. TF-IDF turns each preprocessed document into a sparse, L2-normalised
   document-term matrix X (documents x terms).  Common words get down-weighted
   by IDF, so domain terms such as "placement", "PPO", "hostel" dominate.
2. NMF factorises X approximately as X ~= W . H with W >= 0 and H >= 0.
   The non-negativity constraint is what makes the factors interpretable:
   every entry is an "amount of", so a row of H is literally a bag of words
   describing one topic, and a row of W is a mixture of those topics.
3. The top-weighted terms of each row of H become that topic's keywords.
4. Each row of W is the document's topic distribution; the argmax row is the
   document's dominant topic.

Nothing about the IIM domain is hard-coded: there is no category list, no
seed keyword list, and no hand-written topic label.  "Topic 0" means
"whatever topic the model learned first", and the keywords explain it.

Known limitations are documented in docs/topic_discovery.md rather than
hidden here: TF-IDF/NMF is a bag-of-words model, so it has no word order, no
synonymy, and no cross-lingual alignment.  That is precisely why the
TopicDiscovery interface exists.
"""

import re
import warnings as _warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sklearn
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.nlp.topic_discovery.base import (
    DocumentTopicAssignment,
    NMFConfig,
    TFIDFConfig,
    TopicDiscovery,
    TopicDiscoveryResult,
    TopicDocument,
    TopicInfo,
)

# Placeholders emitted by the preprocessing stage: <URL>, <MENTION>,
# <EMOJI_loudly_crying_face>, ...
_PLACEHOLDER_RE = re.compile(r"<[^<>]{1,64}>")

# Fallback token pattern kept for reference / documentation purposes.
_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w+\b"


class NMFTopicDiscoverer(TopicDiscovery):
    """
    NMF-based topic discovery over already-preprocessed documents.

    Usage:
        discoverer = NMFTopicDiscoverer(n_topics=8)
        result = discoverer.discover(documents)
        for topic in result.topics:
            print(topic.label, topic.keywords, topic.document_count)

    Input is never modified: the discoverer works on internal copies and
    returns the caller's text untouched.
    """

    _NAME = "nmf"
    _VERSION = "v1"

    def __init__(
        self,
        n_topics: Optional[int] = None,
        n_keywords: Optional[int] = None,
        tfidf_config: Optional[TFIDFConfig] = None,
        nmf_config: Optional[NMFConfig] = None,
        include_timestamp: bool = False,
    ) -> None:
        self._nmf_config = nmf_config or NMFConfig()
        self._tfidf_config = tfidf_config or TFIDFConfig()

        self._tfidf_config = self._validate_tfidf(self._tfidf_config)
        self._nmf_config = self._validate_nmf(self._nmf_config)

        # Precedence: explicit argument > config object > config default (8).
        # `n_topics=None` means "not specified by the caller", so an
        # NMFConfig(n_topics=...) is not silently overridden by the default.
        self._n_topics = self._validate_n_topics(
            self._nmf_config.n_topics if n_topics is None else n_topics
        )
        self._n_keywords = self._validate_n_keywords(
            self._nmf_config.n_keywords if n_keywords is None else n_keywords
        )

        # Opt-in: a wall-clock timestamp would make results non-reproducible,
        # so it is excluded by default and only added when a caller asks.
        self._include_timestamp = include_timestamp

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
    def n_topics(self) -> int:
        """Default number of topics for this discoverer."""
        return self._n_topics

    def discover(
        self,
        documents: Sequence[TopicDocument],
        n_topics: Optional[int] = None,
    ) -> TopicDiscoveryResult:
        docs = list(documents or [])

        n_topics_effective = self._n_topics if n_topics is None else self._validate_n_topics(n_topics)
        result_warnings: List[str] = []

        if not docs:
            # Empty corpus: a well-formed, empty result rather than a crash.
            result_warnings.append("empty_corpus: no documents supplied")
            return self._build_result(
                topics=[],
                documents=[],
                vocabulary_size=0,
                n_topics=n_topics_effective,
                n_documents=0,
                result_warnings=result_warnings,
            )

        duplicate_ids = self._find_duplicate_ids(docs)
        if duplicate_ids:
            result_warnings.append(
                f"duplicate_document_ids: {len(duplicate_ids)} id(s) appear more "
                f"than once; every document is still assigned a topic"
            )

        texts = [self._model_text(doc.text, result_warnings) for doc in docs]

        # --- TF-IDF representation ---
        vectorizer = TfidfVectorizer(
            max_features=self._tfidf_config.max_features,
            min_df=self._tfidf_config.min_df,
            max_df=self._tfidf_config.max_df,
            ngram_range=tuple(self._tfidf_config.ngram_range),
            stop_words=self._tfidf_config.stop_words,
            lowercase=self._tfidf_config.lowercase,
            sublinear_tf=self._tfidf_config.sublinear_tf,
            token_pattern=self._tfidf_config.token_pattern,
            norm=self._tfidf_config.norm,
            use_idf=self._tfidf_config.use_idf,
            smooth_idf=self._tfidf_config.smooth_idf,
        )

        try:
            X = vectorizer.fit_transform(texts)
        except ValueError as exc:
            raise ValueError(
                f"corpus produced no usable TF-IDF features: {exc}. "
                "Check the text content, or the min_df / ngram_range / "
                "stop_words configuration."
            ) from exc

        vocabulary_size = int(len(vectorizer.vocabulary_))
        if vocabulary_size < n_topics_effective:
            raise ValueError(
                f"too-small corpus: TF-IDF produced {vocabulary_size} term(s) "
                f"but {n_topics_effective} topics were requested; NMF needs at "
                "least as many features as topics. Lower n_topics, increase "
                "min_df flexibility, or use a larger corpus."
            )

        feature_names = np.asarray(vectorizer.get_feature_names_out())

        if n_topics_effective > len(docs):
            raise ValueError(
                f"too-small corpus: {len(docs)} document(s) cannot support "
                f"{n_topics_effective} topics; NMF needs n_topics <= n_documents. "
                "Lower n_topics or use a larger corpus."
            )

        # --- NMF factorisation: X ~= W . H ---
        model = NMF(
            n_components=n_topics_effective,
            init=self._nmf_config.init,
            solver=self._nmf_config.solver,
            beta_loss=self._nmf_config.beta_loss,
            random_state=self._nmf_config.random_state,
            max_iter=self._nmf_config.max_iter,
            tol=self._nmf_config.tol,
            alpha_W=self._nmf_config.alpha_W,
            alpha_H=self._nmf_config.alpha_H,
            l1_ratio=self._nmf_config.l1_ratio,
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", ConvergenceWarning)
            W = model.fit_transform(X)          # (n_documents, n_topics)
            H = model.components_               # (n_topics, n_features)

        for w in caught:
            result_warnings.append(
                f"{w.category.__name__}: NMF did not fully converge in "
                f"{self._nmf_config.max_iter} iterations; consider raising max_iter"
            )

        topics = self._build_topics(H, feature_names, W)
        assignments = self._build_assignments(docs, W)

        return self._build_result(
            topics=topics,
            documents=assignments,
            vocabulary_size=vocabulary_size,
            n_topics=n_topics_effective,
            n_documents=len(docs),
            result_warnings=result_warnings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _model_text(self, text: str, result_warnings: List[str]) -> str:
        """
        Build the string fed to the vectoriser.

        This is a *copy*; the caller's document and its text are never
        modified.  Placeholder tokens are dropped because they are artefacts
        of the preprocessing stage, not topic-bearing vocabulary.
        """
        value = text or ""
        if self._nmf_config.drop_placeholder_tokens:
            stripped = _PLACEHOLDER_RE.sub(" ", value)
            if stripped != value:
                result_warnings.append(
                    "placeholders_dropped: <URL>/<MENTION>/<EMOJI_...> tokens "
                    "were excluded from the TF-IDF representation"
                )
            value = stripped
        return value

    @staticmethod
    def _find_duplicate_ids(docs: List[TopicDocument]) -> List[str]:
        seen = set()
        duplicates = []
        for doc in docs:
            if doc.document_id in seen and doc.document_id not in duplicates:
                duplicates.append(doc.document_id)
            seen.add(doc.document_id)
        return duplicates

    def _build_topics(
        self,
        H: np.ndarray,
        feature_names: np.ndarray,
        W: np.ndarray,
    ) -> List[TopicInfo]:
        """Turn each row of H into a topic with its highest-weight terms."""
        n_docs = W.shape[0]
        dominant = np.argmax(W, axis=1)
        counts = np.bincount(dominant, minlength=H.shape[0])

        topics: List[TopicInfo] = []
        for topic_id in range(H.shape[0]):
            row = H[topic_id]
            # Deterministic ordering: weight descending, term ascending.
            order = np.lexsort((feature_names, -row))[: self._n_keywords]
            keywords = [str(feature_names[i]) for i in order]
            weights = [round(float(row[i]), 6) for i in order]
            count = int(counts[topic_id])
            topics.append(
                TopicInfo(
                    topic_id=topic_id,
                    label=self._topic_label(topic_id),
                    keywords=keywords,
                    keyword_weights=weights,
                    document_count=count,
                    document_ratio=round(count / n_docs, 6) if n_docs else 0.0,
                )
            )
        return topics

    def _build_assignments(
        self,
        docs: List[TopicDocument],
        W: np.ndarray,
    ) -> List[DocumentTopicAssignment]:
        """Each row of W is one document's topic distribution."""
        assignments: List[DocumentTopicAssignment] = []
        for idx, doc in enumerate(docs):
            row = W[idx]
            # NMF rows sum to 1 because the residual is Frobenius-normalised;
            # normalise defensively so the invariant holds unconditionally.
            total = float(row.sum())
            if total > 0:
                normalised = row / total
            else:
                normalised = np.zeros_like(row)
            # argmax ties resolve to the lowest topic id -> deterministic.
            dominant_idx = int(np.argmax(normalised))
            assignments.append(
                DocumentTopicAssignment(
                    document_id=doc.document_id,
                    post_id=doc.post_id,
                    comment_id=doc.comment_id,
                    original_text=doc.original_text,
                    text=doc.text,
                    language=doc.language,
                    script=doc.script,
                    is_code_mixed=doc.is_code_mixed,
                    detected_languages=list(doc.detected_languages),
                    topic_distribution={
                        topic_id: round(float(normalised[topic_id]), 6)
                        for topic_id in range(normalised.shape[0])
                    },
                    dominant_topic_id=dominant_idx,
                    dominant_topic_label=self._topic_label(dominant_idx),
                    dominant_topic_score=round(float(normalised[dominant_idx]), 6),
                )
            )
        return assignments

    @staticmethod
    def _topic_label(topic_id: int) -> str:
        """
        Neutral, model-derived label.

        Deliberately generic: assigning semantic names would mean hard-coding
        an IIM category list, which the assignment forbids.
        """
        return f"Topic {topic_id}"

    def _build_result(
        self,
        topics: List[TopicInfo],
        documents: List[DocumentTopicAssignment],
        vocabulary_size: int,
        n_topics: int,
        n_documents: int,
        result_warnings: List[str],
    ) -> TopicDiscoveryResult:
        return TopicDiscoveryResult(
            topic_model=self._NAME,
            topic_model_version=self._VERSION,
            n_topics=n_topics,
            n_documents=n_documents,
            vocabulary_size=vocabulary_size,
            topics=topics,
            documents=documents,
            config=self.config_dict(n_topics),
            library_versions={"scikit-learn": sklearn.__version__, "numpy": np.__version__},
            discovered_at=(
                datetime.now(timezone.utc).isoformat() if self._include_timestamp else None
            ),
            warnings=result_warnings,
        )

    def config_dict(self, n_topics: Optional[int] = None) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob affecting the output."""
        cfg: Dict[str, Any] = {
            "n_topics": self._n_topics if n_topics is None else n_topics,
            "n_keywords": self._n_keywords,
            "tfidf": {
                "max_features": self._tfidf_config.max_features,
                "min_df": self._tfidf_config.min_df,
                "max_df": self._tfidf_config.max_df,
                "ngram_range": list(self._tfidf_config.ngram_range),
                "stop_words": (
                    self._tfidf_config.stop_words
                    if isinstance(self._tfidf_config.stop_words, (list, str))
                    else self._tfidf_config.stop_words
                ),
                "lowercase": self._tfidf_config.lowercase,
                "sublinear_tf": self._tfidf_config.sublinear_tf,
                "token_pattern": self._tfidf_config.token_pattern,
                "norm": self._tfidf_config.norm,
                "use_idf": self._tfidf_config.use_idf,
                "smooth_idf": self._tfidf_config.smooth_idf,
            },
            "nmf": {
                "init": self._nmf_config.init,
                "solver": self._nmf_config.solver,
                "beta_loss": self._nmf_config.beta_loss,
                "random_state": self._nmf_config.random_state,
                "max_iter": self._nmf_config.max_iter,
                "tol": self._nmf_config.tol,
                "alpha_W": self._nmf_config.alpha_W,
                "alpha_H": self._nmf_config.alpha_H,
                "l1_ratio": self._nmf_config.l1_ratio,
                "drop_placeholder_tokens": self._nmf_config.drop_placeholder_tokens,
            },
        }
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_n_topics(n_topics: Any) -> int:
        if isinstance(n_topics, bool) or not isinstance(n_topics, int):
            raise ValueError(f"n_topics must be an int, got {type(n_topics).__name__}")
        if n_topics < 1:
            raise ValueError(f"n_topics must be >= 1, got {n_topics}")
        return n_topics

    @staticmethod
    def _validate_n_keywords(n_keywords: Any) -> int:
        if isinstance(n_keywords, bool) or not isinstance(n_keywords, int):
            raise ValueError(
                f"n_keywords must be an int, got {type(n_keywords).__name__}"
            )
        if n_keywords < 1:
            raise ValueError(f"n_keywords must be >= 1, got {n_keywords}")
        return n_keywords

    @staticmethod
    def _validate_tfidf(cfg: TFIDFConfig) -> TFIDFConfig:
        if cfg.max_features is not None and cfg.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {cfg.max_features}")

        if isinstance(cfg.min_df, bool) or not isinstance(cfg.min_df, (int, float)):
            raise ValueError("min_df must be an int or float")
        if isinstance(cfg.max_df, bool) or not isinstance(cfg.max_df, (int, float)):
            raise ValueError("max_df must be an int or float")
        if isinstance(cfg.min_df, int) and cfg.min_df < 0:
            raise ValueError(f"min_df as int must be >= 0, got {cfg.min_df}")
        if isinstance(cfg.max_df, int) and cfg.max_df < 0:
            raise ValueError(f"max_df as int must be >= 0, got {cfg.max_df}")
        for name, value in (("min_df", cfg.min_df), ("max_df", cfg.max_df)):
            if isinstance(value, float) and not (0.0 < value <= 1.0):
                raise ValueError(
                    f"{name} as float must be in (0.0, 1.0], got {value}"
                )

        lo, hi = cfg.ngram_range
        if lo < 1 or hi < lo:
            raise ValueError(
                f"ngram_range must satisfy 1 <= min_n <= max_n, got {cfg.ngram_range}"
            )

        if cfg.norm not in ("l1", "l2", None):
            raise ValueError(f"norm must be 'l1', 'l2' or None, got {cfg.norm!r}")

        try:
            re.compile(cfg.token_pattern)
        except re.error as exc:
            raise ValueError(f"invalid token_pattern: {exc}") from exc

        stop = cfg.stop_words
        if stop is not None and not isinstance(stop, (str, list)):
            raise ValueError(
                "stop_words must be None, 'english', or a list of terms; got "
                f"{type(stop).__name__}"
            )
        if isinstance(stop, str) and stop != "english":
            raise ValueError(
                "only the built-in 'english' stop-word list is supported "
                f"(no language-specific lists), got {stop!r}"
            )
        if isinstance(stop, list) and not all(isinstance(w, str) for w in stop):
            raise ValueError("stop_words list must contain only strings")

        return cfg

    @staticmethod
    def _validate_nmf(cfg: NMFConfig) -> NMFConfig:
        NMFTopicDiscoverer._validate_n_topics(cfg.n_topics)
        NMFTopicDiscoverer._validate_n_keywords(cfg.n_keywords)

        if cfg.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {cfg.max_iter}")
        if cfg.tol < 0:
            raise ValueError(f"tol must be >= 0, got {cfg.tol}")
        if cfg.init not in ("nndsvda", "nndsvd", "random", "custom"):
            raise ValueError(f"unsupported init {cfg.init!r}")
        if cfg.beta_loss not in ("frobenius", "kullback-leibler"):
            raise ValueError(f"unsupported beta_loss {cfg.beta_loss!r}")
        if cfg.alpha_W < 0 or cfg.alpha_H < 0:
            raise ValueError("alpha_W / alpha_H must be >= 0")
        if not 0.0 <= cfg.l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {cfg.l1_ratio}")
        return cfg
