"""
Shared TF-IDF plumbing for the sentiment and emotion classifiers.

Both dimensions featurise text the same way — TF-IDF with a script-agnostic
token pattern, a copy that has preprocessing placeholders stripped, and
optionally weak topic-derived tokens.  That logic lives here so the two
classifiers differ only where they genuinely do: sentiment fits one
multiclass model, emotion fits one binary model per label.

Nothing in this module knows a sentiment class or an emotion label.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Placeholders emitted by the preprocessing stage: <URL>, <MENTION>,
# <EMOJI_loudly_crying_face>, ...
PLACEHOLDER_RE = re.compile(r"<[^<>]{1,64}>")

# Topic-derived features use these prefixes so they can never collide with
# ordinary vocabulary terms.
TOPIC_FEATURE_PREFIX = "topicid"
TOPIC_KEYWORD_PREFIX = "topickey"


def library_versions() -> Dict[str, str]:
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__}


class TfidfLogisticTextModel:
    """
    Featurisation and validation shared by both dimensions.

    Not an ABC and not exported as a public interface: it is the internal base
    of :class:`TfidfLogisticSentimentClassifier` and
    :class:`TfidfLogisticEmotionClassifier`.
    """

    _config: Any
    _vectorizer: TfidfVectorizer
    #: Fingerprints of the documents used for training, used only to detect
    #: self-scoring in evaluate().
    _training_ids: set = frozenset()  # type: ignore[assignment]
    _training_texts: set = frozenset()  # type: ignore[assignment]

    # -- Training-set fingerprinting ------------------------------------

    def _record_training_fingerprints(self, records: List[Any]) -> None:
        """
        Remember which documents trained this model.

        Used exclusively to warn when ``evaluate()`` is handed the training
        data back, which reports memorisation rather than generalisation.
        """
        self._training_ids = frozenset(str(r.document_id) for r in records)
        self._training_texts = frozenset(str(r.text) for r in records)

    def _self_scoring_warning(self, records: List[Any]) -> Optional[str]:
        """
        Warn when an evaluation set overlaps the training data.

        A model scored on its own training documents reports what it memorised,
        not what it can do, and on a small corpus that number is often a
        perfect 1.0 — which is how synthetic metrics end up being presented as
        project performance.  This is a warning rather than a refusal, because
        spotting the overlap is informative; dropping the rows silently is not.
        """
        if not records or not self._training_ids:
            return None
        by_id = sum(
            1 for r in records if str(r.document_id) in self._training_ids
        )
        by_text = sum(1 for r in records if str(r.text) in self._training_texts)
        overlap = max(by_id, by_text)
        if overlap == 0:
            return None
        share = overlap / len(records)
        if overlap == len(records):
            return (
                f"every one of these {len(records)} evaluation document(s) is in "
                "the training data, so these metrics measure memorisation and are "
                "NOT held-out performance. Re-annotate a disjoint evaluation set "
                "before quoting any number from this run."
            )
        return (
            f"{overlap} of {len(records)} evaluation document(s) ({share:.0%}) "
            "also appear in the training data, which inflates these metrics; "
            "they are not a clean held-out measurement."
        )

    # -- Featurisation -------------------------------------------------

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
        preprocessing stage.  They are stripped here because they are frequent
        enough across a corpus to become high-order terms that carry no
        reliable affect signal, and because ``<EMOJI_x>`` is a lossy proxy —
        one token can stand for many visually distinct emojis.

        Note that the *document* keeps its placeholders: the text handed to the
        model is a copy, so ``original_text`` and ``text`` are returned
        untouched.  Disable with
        ``SentimentModelConfig(drop_placeholder_tokens=False)`` /
        ``EmotionModelConfig(drop_placeholder_tokens=False)`` to test whether
        the placeholders carry any signal.
        """
        value = text or ""
        if self._config.drop_placeholder_tokens:
            value = PLACEHOLDER_RE.sub(" ", value)
        return value

    def _append_topic_features(
        self,
        text: str,
        topic_id: Optional[int],
        topic_keywords: Sequence[str],
    ) -> str:
        """
        Optionally inject weak topic-derived features.

        Prefixed tokens (``topicid3``, ``topickeyhostel``) cannot collide with
        ordinary vocabulary.  Off by default: topic ids are corpus-run specific
        and would not survive a re-fit of the topic model.
        """
        parts = [text]
        if topic_id is not None:
            parts.append(f"{TOPIC_FEATURE_PREFIX}{topic_id}")
        for keyword in topic_keywords or []:
            cleaned = str(keyword).strip()
            if cleaned:
                parts.append(f"{TOPIC_KEYWORD_PREFIX}{cleaned}")
        return " ".join(parts)

    def _featurize_text(
        self,
        text: str,
        topic_id: Optional[int],
        topic_keywords: Sequence[str],
    ) -> str:
        """Single featurisation path used by both training and inference."""
        cleaned = self._strip_placeholders(text)
        if not self._config.use_topic_features:
            return cleaned
        return self._append_topic_features(cleaned, topic_id, topic_keywords)

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

    @classmethod
    def _validate_shared_config(cls, config: Any) -> Any:
        """Validate the knobs both dimensions share, with actionable errors."""
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
        if (
            isinstance(config.min_df, float)
            and isinstance(config.max_df, float)
            and config.min_df > config.max_df
        ):
            raise ValueError(
                f"min_df ({config.min_df}) must not exceed max_df ({config.max_df})"
            )
        if (
            isinstance(config.min_df, int)
            and isinstance(config.max_df, int)
            and config.min_df > config.max_df
        ):
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
                "only the built-in 'english' stop-word list is supported (no "
                f"language-specific lists), got {stop!r}"
            )
        if isinstance(stop, list) and not all(isinstance(word, str) for word in stop):
            raise ValueError("stop_words list must contain only strings")

        if config.C <= 0:
            raise ValueError(f"C must be > 0 (inverse regularisation), got {config.C}")
        if config.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {config.max_iter}")
        if config.class_weight is not None and config.class_weight not in ("balanced",):
            raise ValueError(
                f"class_weight must be None or 'balanced', got {config.class_weight!r}"
            )
        if config.min_training_documents < 2:
            raise ValueError(
                f"min_training_documents must be >= 2, got {config.min_training_documents}"
            )
        cls._validate_threshold(config.threshold)
        return config

    @staticmethod
    def _config_dict(
        config: Any,
        threshold: float,
        data_guards: Dict[str, Any],
    ) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob that affects the output."""
        return {
            "threshold": threshold,
            "tfidf": {
                "max_features": config.max_features,
                "min_df": config.min_df,
                "max_df": config.max_df,
                "ngram_range": list(config.ngram_range),
                "stop_words": config.stop_words,
                "lowercase": config.lowercase,
                "sublinear_tf": config.sublinear_tf,
                "token_pattern": config.token_pattern,
                "norm": config.norm,
                "use_idf": config.use_idf,
                "smooth_idf": config.smooth_idf,
            },
            "logistic_regression": {
                "C": config.C,
                "max_iter": config.max_iter,
                "solver": config.solver,
                "class_weight": config.class_weight,
                "random_state": config.random_state,
            },
            "data_guards": data_guards,
            "features": {
                "use_topic_features": config.use_topic_features,
                "drop_placeholder_tokens": config.drop_placeholder_tokens,
            },
        }
