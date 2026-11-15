"""
Topic discovery stage.

Discovers the major recurring subjects of discussion in a corpus WITHOUT any
hard-coded category list.  Baseline implementation: TF-IDF -> NMF.

This stage is deliberately separate from issue identification:
- Topic discovery answers "what is being discussed?"
- Issue identification answers "is this a problem, and of what kind?"

Topic discovery runs on already-preprocessed text and never mutates raw
Reddit text or preprocessing output.

See docs/topic_discovery.md for the TF-IDF / NMF formulation, the
multilingual trade-offs, and the roadmap to embedding-based discovery.
"""

from app.services.nlp.topic_discovery.base import (
    DocumentTopicAssignment,
    NMFConfig,
    TFIDFConfig,
    TopicDiscovery,
    TopicDiscoveryResult,
    TopicDocument,
    TopicInfo,
)
from app.services.nlp.topic_discovery.nmf_topic_discoverer import (
    NMFTopicDiscoverer,
)

__all__ = [
    "TopicDocument",
    "TopicDiscovery",
    "TopicDiscoveryResult",
    "TopicInfo",
    "DocumentTopicAssignment",
    "TFIDFConfig",
    "NMFConfig",
    "NMFTopicDiscoverer",
]
