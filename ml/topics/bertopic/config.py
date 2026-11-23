"""Configuration structures for reproducible BERTopic experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class BERTopicExperimentConfig:
    """Complete, JSON-serialisable hyperparameter specification for one BERTopic run."""

    model_id: str
    embedding_key: str  # e.g., "multilingual_e5_small" | "multilingual_minilm" | "muril_dapt_wwm"
    embedding_model_name: str
    preprocessing_mode: str = "raw_clean"  # raw_clean | language_aware_affect
    min_word_count: int = 3
    # UMAP parameters (seeded for reproducibility)
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    # HDBSCAN parameters (natural topic discovery — never forced to 10!)
    hdbscan_min_cluster_size: int = 25
    hdbscan_min_samples: int = 8
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    # c-TF-IDF & CountVectorizer parameters (applied to per-topic concatenated mega-docs!)
    ngram_range: Tuple[int, int] = (1, 2)
    vectorizer_min_df: int = 1
    vectorizer_max_df: float = 1.0
    bm25_weighting: bool = True
    reduce_frequent_words: bool = True
    top_n_words: int = 10
    nr_topics: Optional[int] = None  # None = natural density-discovered topic count
    calculate_probabilities: bool = True
    random_state: int = 42
    corpus_version: str = "6.1.0"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        return d
