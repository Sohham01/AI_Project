"""Multi-metric topic-model evaluation suite (fair comparison of NMF vs. BERTopic).

Computes strictly corpus-grounded metrics without requiring gensim:
- NPMI Topic Coherence (document co-occurrence Normalized Pointwise Mutual Information)
- UMass Coherence (log co-occurrence ratio)
- Topic Diversity (unique words across top-K keywords of all topics)
- Topic Count, Outlier Count & Outlier Ratio (-1 cluster)
- Topic Size Distribution (min, median, max, normalized Shannon entropy balance)
- Representative-Document Centroid Relevance (mean cosine similarity to topic centroid)
- Language-Register Breakdown (English vs. Hinglish/Indic vs. Terse outlier & assignment shares)
- Unsupervised-Topic vs. Supervised-Issue Cross-Tabulation (observational overlap only)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z0-9\u0900-\u0d7f]{2,}")


def _doc_token_sets(texts: Sequence[str]) -> List[set[str]]:
    out: List[set[str]] = []
    for t in texts:
        toks = [w.lower() for w in _WORD_RE.findall(str(t or ""))]
        tset = set(toks)
        # Also include bigrams so bigram keywords from NMF / BERTopic score accurately
        for i in range(len(toks) - 1):
            tset.add(f"{toks[i]} {toks[i + 1]}")
        out.append(tset)
    return out


def compute_topic_coherence_npmi_umass(
    topics_keywords: Sequence[Sequence[str]],
    corpus_texts: Sequence[str],
    top_k: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute corpus-level NPMI and UMass coherence over topic top-K keywords."""
    doc_sets = _doc_token_sets(corpus_texts)
    n_docs = max(len(doc_sets), 1)

    needed = {kw.lower().strip() for kws in topics_keywords for kw in kws[:top_k] if kw and kw.strip()}
    term_docs: Dict[str, set[int]] = {w: set() for w in needed}
    for doc_idx, tset in enumerate(doc_sets):
        for w in needed:
            if w in tset:
                term_docs[w].add(doc_idx)

    per_topic_npmi: List[float] = []
    per_topic_umass: List[float] = []

    for kws in topics_keywords:
        words = [kw.lower().strip() for kw in kws[:top_k] if kw and kw.strip()]
        npmis: List[float] = []
        umasses: List[float] = []
        for i in range(len(words)):
            di = term_docs.get(words[i], set())
            pi = len(di) / n_docs
            for j in range(i + 1, len(words)):
                dj = term_docs.get(words[j], set())
                pj = len(dj) / n_docs
                co = len(di & dj)
                pij = co / n_docs
                if pij > 0 and pi > 0 and pj > 0:
                    pmi = math.log((pij + eps) / (pi * pj + eps))
                    npmi = pmi / (-math.log(pij + eps))
                else:
                    npmi = -1.0 if (pi > 0 and pj > 0) else 0.0
                npmis.append(float(np.clip(npmi, -1.0, 1.0)))
                # UMass uses log((D(wi, wj) + 1) / D(wi))
                denom = max(len(di), 1)
                umasses.append(math.log((co + 1.0) / denom))
        per_topic_npmi.append(round(float(np.mean(npmis)) if npmis else 0.0, 4))
        per_topic_umass.append(round(float(np.mean(umasses)) if umasses else 0.0, 4))

    return {
        "mean_npmi": round(float(np.mean(per_topic_npmi)) if per_topic_npmi else 0.0, 4),
        "median_npmi": round(float(np.median(per_topic_npmi)) if per_topic_npmi else 0.0, 4),
        "mean_umass": round(float(np.mean(per_topic_umass)) if per_topic_umass else 0.0, 4),
        "per_topic_npmi": per_topic_npmi,
        "per_topic_umass": per_topic_umass,
    }


def compute_topic_diversity(topics_keywords: Sequence[Sequence[str]], top_k: int = 10) -> float:
    """Fraction of unique words across top-K words of all non-outlier topics."""
    all_words = [kw.lower().strip() for kws in topics_keywords for kw in kws[:top_k] if kw and kw.strip()]
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 4)


def compute_representative_relevance(
    embeddings: Optional[np.ndarray],
    doc_topics: Sequence[int],
    top_m: int = 5,
) -> Dict[str, Any]:
    """Mean cosine similarity of top-M representative documents to each topic centroid."""
    if embeddings is None or len(embeddings) == 0:
        return {"mean_centroid_cosine": None, "per_topic": {}}
    by_topic: Dict[int, List[int]] = defaultdict(list)
    for idx, tid in enumerate(doc_topics):
        if tid != -1:
            by_topic[int(tid)].append(idx)
    per_topic: Dict[str, float] = {}
    vals: List[float] = []
    for tid, indices in sorted(by_topic.items()):
        mat = embeddings[indices]
        centroid = mat.mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        centroid = centroid / norm
        sims = mat @ centroid
        top_sims = np.sort(sims)[::-1][: min(top_m, len(sims))]
        score = float(np.mean(top_sims))
        per_topic[str(tid)] = round(score, 4)
        vals.append(score)
    return {
        "mean_centroid_cosine": round(float(np.mean(vals)) if vals else 0.0, 4),
        "per_topic": per_topic,
    }


def evaluate_topic_model(
    *,
    corpus_texts: Sequence[str],
    doc_topics: Sequence[int],
    topics_keywords: Dict[int, List[str]],
    languages: Optional[Sequence[str]] = None,
    embeddings: Optional[np.ndarray] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compute the full Phase 1 multi-metric evaluation bundle for any topic model (NMF or BERTopic)."""
    n_docs = len(doc_topics)
    outlier_count = sum(1 for t in doc_topics if int(t) == -1)
    assigned_topics = [int(t) for t in doc_topics if int(t) != -1]
    counts = Counter(assigned_topics)
    valid_tids = sorted(tid for tid in topics_keywords if int(tid) != -1 and counts.get(int(tid), 0) > 0)
    kw_lists = [topics_keywords[tid] for tid in valid_tids]

    coh = compute_topic_coherence_npmi_umass(kw_lists, corpus_texts, top_k=top_k)
    diversity = compute_topic_diversity(kw_lists, top_k=top_k)
    rep_rel = compute_representative_relevance(embeddings, doc_topics)

    sizes = [counts[tid] for tid in valid_tids]
    if sizes and len(sizes) > 1:
        probs = np.array(sizes, dtype=float) / max(sum(sizes), 1)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        norm_balance = round(entropy / math.log(len(sizes)), 4)
    else:
        norm_balance = 0.0

    # Language-specific assignment & outlier breakdown
    lang_breakdown: Dict[str, Any] = {}
    if languages is not None and len(languages) == n_docs:
        by_reg: Dict[str, List[int]] = defaultdict(list)
        for lang, tid in zip(languages, doc_topics):
            lg = str(lang or "und").lower()
            reg = "english" if lg == "en" else ("code_mixed_indic" if lg in {"hi", "hi-en", "bn", "bn-en", "or", "or-en"} else "terse_und")
            by_reg[reg].append(int(tid))
        for reg, tids in sorted(by_reg.items()):
            n_r = max(len(tids), 1)
            out_r = sum(1 for t in tids if t == -1)
            dist = Counter(t for t in tids if t != -1)
            lang_breakdown[reg] = {
                "n_docs": len(tids),
                "outlier_count": out_r,
                "outlier_rate": round(out_r / n_r, 4),
                "distinct_topics_covered": len(dist),
                "top_topics": [{"topic_id": tid, "count": cnt} for tid, cnt in dist.most_common(3)],
            }

    return {
        "n_documents": n_docs,
        "n_topics": len(valid_tids),
        "outlier_count": outlier_count,
        "outlier_rate": round(outlier_count / max(n_docs, 1), 4),
        "coherence_npmi": coh["mean_npmi"],
        "coherence_npmi_median": coh["median_npmi"],
        "coherence_umass": coh["mean_umass"],
        "topic_diversity": diversity,
        "representative_centroid_cosine": rep_rel["mean_centroid_cosine"],
        "size_distribution": {
            "min": int(min(sizes)) if sizes else 0,
            "median": float(np.median(sizes)) if sizes else 0.0,
            "max": int(max(sizes)) if sizes else 0,
            "normalized_entropy_balance": norm_balance,
        },
        "per_topic_coherence_npmi": {str(tid): coh["per_topic_npmi"][i] for i, tid in enumerate(valid_tids)},
        "language_breakdown": lang_breakdown,
    }
