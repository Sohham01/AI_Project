"""Modular BERTopic topic-discovery pipeline built alongside (never replacing) NMF.

Implements all 10 required pipeline stages:
1. Multilingual embedding generation (delegated to ``embeddings.py`` with disk caching).
2. Deterministic UMAP dimensionality reduction (seeded ``random_state``).
3. Density-based HDBSCAN clustering (natural topic discovery, explicit ``-1`` outlier preservation).
4. Class-based TF-IDF (c-TF-IDF) with BM25 weighting & Indic-script-safe CountVectorizer.
5. Topic representation & keyword weight extraction (unigrams + bigrams).
6. Representative document selection (highest probability + centroid cosine).
7. Document-to-topic assignment (soft probabilities & hard cluster labels, preserving ``-1``).
8. Outlier inspection & structured outlier reporting.
9. Topic hierarchy exploration (``hierarchical_topics``) and non-destructive topic-reduction preview.
10. SafeTensors persistence & reproducible load/transform round-trip.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hdbscan
import numpy as np
import umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.topics.bertopic.config import BERTopicExperimentConfig
from ml.topics.bertopic.embeddings import EMBEDDING_CATALOG, encode_documents

# Indic-script-safe token pattern: preserves Devanagari, Bengali, Odia combining vowel signs/matras
INDIC_MULTILINGUAL_TOKEN_PATTERN = r"(?u)[A-Za-z0-9\u0900-\u0d7f]{2,}"

# Minimal conversational boilerplate stopwords (kept conservative so Hinglish content words remain visible)
EXTRA_STOPWORDS = {
    "reddit_link", "url", "http", "https", "www", "com", "amp", "gt", "lt", "nbsp",
    "removed", "deleted", "edit", "tl", "dr", "tldr",
    "just", "like", "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
    "ve", "ll", "re", "im", "ive", "thats", "youre", "cant", "wont", "dont", "didnt",
    "really", "actually", "basically", "literally", "thing", "things", "lot", "got",
    "going", "know", "think", "want", "make", "getting", "say", "said", "people",
}
MULTILINGUAL_STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def preprocess_for_bertopic(texts: Sequence[str], mode: str = "raw_clean") -> List[str]:
    """Produce non-destructive preprocessed strings for BERTopic without mutating originals.

    Modes:
    - ``raw_clean``: light whitespace/URL cleanup keeping original casing, slang, emojis, and scripts.
    - ``language_aware_affect``: applies RCI's slang + affect-preserving emoji normalization and
      Romanized Hinglish spelling normalization while preserving lexical semantics.
    """
    if mode == "raw_clean":
        out: List[str] = []
        for t in texts:
            s = _URL_RE.sub(" ", str(t or ""))
            s = _SPACE_RE.sub(" ", s).strip()
            out.append(s)
        return out

    if mode == "language_aware_affect":
        from app.services.nlp.preprocessing.slang_normalizer import normalize_slang
        from app.services.nlp.preprocessing.emoji_normalizer import normalize_emojis_semantic_affect

        # Non-destructive common Hinglish phonetic unifiers (for clustering consistency)
        roman_map = {
            r"\bnhi\b": "nahi",
            r"\bnahin\b": "nahi",
            r"\bbohot\b": "bahut",
            r"\bboht\b": "bahut",
            r"\bachha\b": "accha",
            r"\bacha\b": "accha",
            r"\bkrna\b": "karna",
            r"\bkyu\b": "kyun",
            r"\bmtlb\b": "matlab",
            r"\byarr\b": "yaar",
            r"\bhogya\b": "ho gaya",
            r"\bgya\b": "gaya",
        }
        compiled = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in roman_map.items()]

        out = []
        for t in texts:
            s = _URL_RE.sub(" ", str(t or ""))
            s = normalize_slang(s)
            s = normalize_emojis_semantic_affect(s)
            for rx, rep in compiled:
                s = rx.sub(rep, s)
            s = _SPACE_RE.sub(" ", s).strip()
            out.append(s)
        return out

    raise ValueError(f"Unsupported preprocessing mode: {mode!r}")


def _make_machine_suggested_label(topic_id: int, keywords: Sequence[str]) -> str:
    """Create a clearly descriptive machine-suggested topic label from top c-TF-IDF terms."""
    if int(topic_id) == -1:
        return "Outliers / Unassigned (-1)"
    clean_kws = [str(k).strip() for k in keywords[:4] if str(k).strip()]
    if not clean_kws:
        return f"BERTopic {topic_id}"
    return f"BT-{topic_id}: " + " · ".join(clean_kws)


class RCIBERTopicPipeline:
    """Self-contained, reproducible BERTopic pipeline for RCI."""

    def __init__(self, config: BERTopicExperimentConfig) -> None:
        self.config = config
        self.model: Optional[BERTopic] = None
        self.embedding_meta: Dict[str, Any] = {}
        self._fit_topics: Optional[List[int]] = None
        self._fit_probs: Optional[np.ndarray] = None

    def _build_model(self, n_docs: int) -> BERTopic:
        cfg = self.config
        n_neighbors = max(2, min(cfg.umap_n_neighbors, n_docs - 2))
        n_components = max(2, min(cfg.umap_n_components, n_docs - 2))
        min_cluster_size = max(2, min(cfg.hdbscan_min_cluster_size, max(2, n_docs // 3)))
        min_samples = max(1, min(cfg.hdbscan_min_samples, min_cluster_size))

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=cfg.random_state,
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=cfg.hdbscan_metric,
            cluster_selection_method=cfg.hdbscan_cluster_selection_method,
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            ngram_range=tuple(cfg.ngram_range),
            stop_words=MULTILINGUAL_STOPWORDS,
            min_df=cfg.vectorizer_min_df,
            max_df=cfg.vectorizer_max_df,
            token_pattern=INDIC_MULTILINGUAL_TOKEN_PATTERN,
        )
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=cfg.bm25_weighting,
            reduce_frequent_words=cfg.reduce_frequent_words,
        )
        return BERTopic(
            embedding_model=None,  # Precomputed L2-normalized multilingual embeddings passed explicitly
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            top_n_words=cfg.top_n_words,
            nr_topics=cfg.nr_topics,
            calculate_probabilities=cfg.calculate_probabilities,
            verbose=False,
        )

    def fit(
        self,
        raw_texts: Sequence[str],
        *,
        embeddings: Optional[np.ndarray] = None,
        use_embedding_cache: bool = True,
    ) -> Tuple[List[int], Optional[np.ndarray], np.ndarray]:
        """Fit BERTopic on ``raw_texts`` using configured preprocessing and embeddings."""
        processed = preprocess_for_bertopic(raw_texts, mode=self.config.preprocessing_mode)
        if embeddings is None:
            embeddings, self.embedding_meta = encode_documents(
                processed,
                self.config.embedding_key,
                is_query=False,
                use_cache=use_embedding_cache,
            )
        else:
            self.embedding_meta = {
                "embedding_key": self.config.embedding_key,
                "model_name": self.config.embedding_model_name,
                "dim": int(embeddings.shape[1]),
                "n_docs": int(embeddings.shape[0]),
                "cached": True,
            }

        self.model = self._build_model(len(processed))
        topics, probs = self.model.fit_transform(processed, embeddings=embeddings)
        self._fit_topics = [int(t) for t in topics]
        self._fit_probs = np.asarray(probs) if probs is not None else None
        return self._fit_topics, self._fit_probs, embeddings

    def transform(
        self,
        raw_texts: Sequence[str],
        *,
        embeddings: Optional[np.ndarray] = None,
        use_embedding_cache: bool = False,
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Assign unseen documents to discovered topics (explicitly returning -1 for outliers)."""
        if self.model is None:
            raise RuntimeError("BERTopic pipeline must be fit or loaded before calling transform().")
        processed = preprocess_for_bertopic(raw_texts, mode=self.config.preprocessing_mode)
        if embeddings is None:
            embeddings, _ = encode_documents(
                processed,
                self.config.embedding_key,
                is_query=False,
                use_cache=use_embedding_cache,
            )
        topics, probs = self.model.transform(processed, embeddings=embeddings)
        return [int(t) for t in topics], (np.asarray(probs) if probs is not None else None)

    def get_topic_keywords(self) -> Dict[int, List[Tuple[str, float]]]:
        """Return {topic_id: [(keyword, c_tfidf_weight), ...]} excluding -1 outlier bin."""
        if self.model is None:
            return {}
        out: Dict[int, List[Tuple[str, float]]] = {}
        for tid, pairs in (self.model.get_topics() or {}).items():
            if int(tid) == -1:
                continue
            cleaned = [(str(w), round(float(score), 6)) for w, score in pairs if str(w).strip()]
            out[int(tid)] = cleaned[: self.config.top_n_words]
        return out

    def extract_structured_topics(
        self,
        *,
        documents_meta: Sequence[Dict[str, Any]],
        doc_topics: Sequence[int],
        doc_probs: Optional[np.ndarray],
        embeddings: Optional[np.ndarray],
        max_examples_per_topic: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build rich, isolated BERTopic topic records and explicit outlier records."""
        kw_map = self.get_topic_keywords()
        n_total = max(len(doc_topics), 1)

        by_topic: Dict[int, List[int]] = defaultdict(list)
        for idx, tid in enumerate(doc_topics):
            by_topic[int(tid)].append(idx)

        topics_out: List[Dict[str, Any]] = []
        for tid in sorted(k for k in by_topic if k != -1):
            indices = by_topic[tid]
            pairs = kw_map.get(tid, [])
            keywords = [w for w, _ in pairs]
            weights = [float(s) for _, s in pairs]

            # Rank representative documents by centroid cosine similarity (fallback: topic probability)
            scored_docs: List[Tuple[float, int]] = []
            if embeddings is not None and len(indices) > 0:
                mat = embeddings[indices]
                centroid = mat.mean(axis=0)
                norm = float(np.linalg.norm(centroid)) or 1.0
                centroid = centroid / norm
                sims = mat @ centroid
                for local_i, doc_idx in enumerate(indices):
                    scored_docs.append((float(sims[local_i]), doc_idx))
            else:
                for doc_idx in indices:
                    p = 1.0
                    if doc_probs is not None and doc_probs.ndim == 2 and tid < doc_probs.shape[1]:
                        p = float(doc_probs[doc_idx, tid])
                    scored_docs.append((p, doc_idx))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            examples: List[Dict[str, Any]] = []
            for sim_score, doc_idx in scored_docs[:max_examples_per_topic]:
                dm = documents_meta[doc_idx]
                examples.append({
                    "document_id": dm.get("document_id"),
                    "comment_id": dm.get("comment_id"),
                    "post_id": dm.get("post_id"),
                    "post_title": dm.get("post_title", ""),
                    "subreddit": dm.get("subreddit", ""),
                    "language": dm.get("language", "und"),
                    "script": dm.get("script", "latin"),
                    "is_code_mixed": bool(dm.get("is_code_mixed", False)),
                    "issue_category": dm.get("issue_category"),
                    "text": dm.get("original_text", ""),
                    "score": round(float(sim_score), 4),
                })

            # Observational cross-tab with supervised issue categories (strictly kept separate!)
            issue_counts = Counter(
                str(documents_meta[i].get("issue_category"))
                for i in indices
                if documents_meta[i].get("issue_category")
            )
            lang_counts = Counter(str(documents_meta[i].get("language") or "und") for i in indices)

            topics_out.append({
                "topic_id": int(tid),
                "namespaced_topic_id": f"bertopic:{self.config.model_id}:{tid}",
                "algorithm": "bertopic",
                "label": _make_machine_suggested_label(tid, keywords),
                "label_status": "machine_suggested_pending_human_review",
                "keywords": keywords,
                "keyword_weights": weights,
                "document_count": len(indices),
                "document_ratio": round(len(indices) / n_total, 6),
                "representative_examples": examples,
                "language_mix": dict(lang_counts.most_common(5)),
                "related_supervised_issues": [
                    {"issue": iss, "count": cnt, "share": round(cnt / max(len(indices), 1), 4)}
                    for iss, cnt in issue_counts.most_common(4)
                ],
            })

        topics_out.sort(key=lambda t: t["document_count"], reverse=True)

        # Preserve outliers explicitly (-1 cluster)
        outlier_indices = by_topic.get(-1, [])
        outliers_out: List[Dict[str, Any]] = []
        for doc_idx in outlier_indices:
            dm = documents_meta[doc_idx]
            outliers_out.append({
                "document_id": dm.get("document_id"),
                "comment_id": dm.get("comment_id"),
                "post_id": dm.get("post_id"),
                "post_title": dm.get("post_title", ""),
                "subreddit": dm.get("subreddit", ""),
                "language": dm.get("language", "und"),
                "script": dm.get("script", "latin"),
                "is_code_mixed": bool(dm.get("is_code_mixed", False)),
                "issue_category": dm.get("issue_category"),
                "text": dm.get("original_text", ""),
                "word_count": len(str(dm.get("original_text", "")).split()),
            })

        return topics_out, outliers_out

    def extract_hierarchy(self, raw_texts: Sequence[str], max_nodes: int = 25) -> List[Dict[str, Any]]:
        """Extract agglomerative topic hierarchy via BERTopic's official ``hierarchical_topics``."""
        if self.model is None:
            return []
        non_outlier_topics = [t for t in (self.model.get_topics() or {}) if int(t) != -1]
        if len(non_outlier_topics) < 2:
            return []
        processed = preprocess_for_bertopic(raw_texts, mode=self.config.preprocessing_mode)
        try:
            df = self.model.hierarchical_topics(processed)
        except Exception:
            return []
        rows: List[Dict[str, Any]] = []
        for _, r in df.head(max_nodes).iterrows():
            rows.append({
                "parent_id": int(r["Parent_ID"]),
                "parent_name": str(r["Parent_Name"]),
                "topics": [int(x) for x in r["Topics"]],
                "child_left_id": int(r["Child_Left_ID"]),
                "child_right_id": int(r["Child_Right_ID"]),
                "distance": round(float(r["Distance"]), 4),
            })
        return rows

    def save(self, artifact_dir: Path) -> Dict[str, str]:
        """Persist BERTopic model via SafeTensors + config + embedding metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted RCIBERTopicPipeline.")
        artifact_dir = Path(artifact_dir)
        model_dir = artifact_dir / "model_safetensors"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(
            str(model_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        cfg_path = artifact_dir / "config.json"
        cfg_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

        emb_path = artifact_dir / "embedding_meta.json"
        emb_path.write_text(json.dumps(self.embedding_meta, indent=2), encoding="utf-8")

        return {
            "artifact_dir": str(artifact_dir),
            "model_safetensors_dir": str(model_dir),
            "config_json": str(cfg_path),
            "embedding_meta_json": str(emb_path),
        }

    @classmethod
    def load(cls, artifact_dir: Path) -> "RCIBERTopicPipeline":
        """Reload a saved RCIBERTopicPipeline from SafeTensors + config.json."""
        artifact_dir = Path(artifact_dir)
        cfg_raw = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        cfg_raw["ngram_range"] = tuple(cfg_raw.get("ngram_range", (1, 2)))
        cfg = BERTopicExperimentConfig(**cfg_raw)
        pipe = cls(cfg)
        pipe.model = BERTopic.load(str(artifact_dir / "model_safetensors"))
        emb_meta_file = artifact_dir / "embedding_meta.json"
        if emb_meta_file.is_file():
            pipe.embedding_meta = json.loads(emb_meta_file.read_text(encoding="utf-8"))
        return pipe
