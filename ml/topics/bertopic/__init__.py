"""BERTopic topic-discovery package (isolated alongside NMF; never replaces NMF)."""

from ml.topics.bertopic.config import BERTopicExperimentConfig
from ml.topics.bertopic.embeddings import (
    EMBEDDING_CATALOG,
    encode_documents,
    evaluate_multilingual_embedding_probes,
)
from ml.topics.bertopic.evaluation import (
    compute_representative_relevance,
    compute_topic_coherence_npmi_umass,
    compute_topic_diversity,
    evaluate_topic_model,
)
from ml.topics.bertopic.pipeline import RCIBERTopicPipeline, preprocess_for_bertopic
from ml.topics.bertopic.registry import TopicModelRegistry

BERTopicPipeline = RCIBERTopicPipeline

__all__ = [
    "BERTopicExperimentConfig",
    "BERTopicPipeline",
    "RCIBERTopicPipeline",
    "TopicModelRegistry",
    "EMBEDDING_CATALOG",
    "encode_documents",
    "evaluate_multilingual_embedding_probes",
    "compute_topic_coherence_npmi_umass",
    "compute_topic_diversity",
    "compute_representative_relevance",
    "evaluate_topic_model",
    "preprocess_for_bertopic",
]
