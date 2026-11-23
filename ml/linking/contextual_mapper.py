"""Phase 4 — Contextual Multilingual Embedding Issue–Solution Mapping Engine.

Connects the full RCI pipeline into one evidence-oriented analytical system:
  ISSUE -> RELATED DISCUSSION -> RELEVANT COMMENTS -> RESPONSE TYPE
  -> SOLUTION CANDIDATES -> SEMANTIC SOLUTION GROUPS -> RESOLUTION EVIDENCE

Key advances over the V5 TF-IDF cosine baseline (which remains intact!):
1. Multilingual dense sentence embeddings (``intfloat/multilingual-e5-small``)
   encoding enriched Issue + Post + Parent Context vs. Candidate Solution text.
2. Multi-signal fusion (never cosine alone): dense semantic similarity + lexical
   TF-IDF cosine + thread/parent-context structural proximity + upstream
   response/solution detector probability.
3. Explicit negation, contradiction, and non-actionable reply penalty (down-ranking
   ``counterargument`` replies, *"doesn't work"* / *"don't join"* / *"no idea"* /
   *"cfbr"* / *"check dm"* false positives).
4. Semantic solution clustering per issue using cosine AgglomerativeClustering so
   similar recommendations group together without collapsing distinct advice.
5. Full 9-field evidence traceability and strict separation between
   ``verification_status = "model_inferred"`` and human-reviewed ground truth.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ISSUE_DESCRIPTIONS: Dict[str, str] = {
    "administrative_communication": (
        "Admissions waitlists, CAP/SAP portals, merit lists, offer emails, category cutoffs, "
        "refund deadlines, AICTE guidelines, document verification, and institute communication delays."
    ),
    "placements": (
        "Summer internships (SIP), final placements, median CTC packages, recruiter profiles, "
        "company shortlists, CV preparation, mock interviews, and career outcomes."
    ),
    "financial_concerns": (
        "MBA tuition fees, education loans, bank interest rates, scholarships, seat-booking refunds, "
        "return on investment (ROI), and fee affordability."
    ),
    "curriculum_teaching": (
        "MBA curriculum quality, faculty teaching effectiveness, BS DSAI vs MPP courses, "
        "electives, pedagogy, case methodology, and academic rigour."
    ),
    "campus_infrastructure": (
        "Permanent campus construction, classrooms, library, healthcare access, city connectivity, "
        "location accessibility, Wi-Fi, and campus facilities."
    ),
    "hostel_facilities": (
        "Hostel room conditions, single vs shared accommodation, mess food quality, dining hygiene, "
        "electricity, AC, laundry, and residential living experience."
    ),
    "extracurriculars": (
        "Student clubs, placement/academic committees, case competitions, fests, peer networking, "
        "sports, and campus student life."
    ),
    "academic_workload": (
        "Heavy coursework, surprise quizzes, case study preparation, tight deadlines, exam stress, "
        "sleep deprivation, and managing MBA academic pressure."
    ),
}

# Explicit negation / contradiction / non-actionable patterns that should NOT be linked as solutions
_CONTRADICTION_PATTERNS = [
    re.compile(r"\b(doesn'?t|does not|didn'?t|did not|won'?t|will not|never)\s+(work|help|solve|matter|fix)\b", re.I),
    re.compile(r"\b(bad idea|terrible idea|not worth|waste of (money|time)|scam|trap|avoid joining|don'?t join|do not join)\b", re.I),
    re.compile(r"\b(no solution|nothing works|impossible|not possible|no chance|zero chance|give up)\b", re.I),
    re.compile(r"\b(wrong advice|false info|misleading|stop lying|not true|bullshit)\b", re.I),
]

_NON_ACTIONABLE_PATTERNS = [
    re.compile(r"^\s*(cfbr|commenting for better reach|following|same|same bro|remindme|bump|upvoted)\b", re.I),
    re.compile(r"\b(no idea|idk|i don'?t know|no clue|not sure|can'?t say|hard to say)\b", re.I),
    re.compile(r"^\s*(dm|dm me|check dm|please dm|can i dm|message me|inbox)\b", re.I),
    re.compile(r"^\s*(thanks|thank you|thx|congrats|congratulations|all the best|good luck)\b", re.I),
    re.compile(r"^\s*(\[?\s*(removed|deleted)(\s+by\s+reddit)?\s*\]?|https?://\S+|!\[gif\]\([^)]+\))\s*$", re.I),
]

_ACTIONABLE_CUES = re.compile(
    r"\b(try|check|contact|email|mail|call|apply|prepare|practice|connect|reach out|refer|guideline|"
    r"aicte|portal|refund|loan|bank|alumni|senior|mentor|mock|cv|resume|linkedin|compare|option|"
    r"wait|withdraw|submit|certificate|document|verify|talk to|speak with|choose|prefer|focus|"
    r"karo|karna|chahiye|dekh|bhej|pooch|puch|try kar)\b",
    re.I,
)

_ROLE_WEIGHTS: Dict[str, float] = {
    "solution": 1.00,
    "advice": 0.78,
    "personal_experience": 0.48,
    "opinion": 0.30,
    "agreement": 0.15,
    "counterargument": 0.12,
}

_KW_STOP = set(ENGLISH_STOP_WORDS) | {
    "iim", "college", "colleges", "students", "student", "people", "just", "like",
    "don", "doesn", "really", "actually", "going", "good", "better", "best", "make",
    "know", "think", "want", "need", "time", "year", "years", "bro", "bhai", "yaar",
    "hai", "hain", "bhi", "toh", "kya", "nahi", "nhi", "kar", "karo", "aur", "mein",
}


def compute_negation_and_non_actionable_penalty(
    text: str,
    response_type: str = "",
) -> Tuple[float, List[str]]:
    """Compute transparent penalty in [0.0, 0.55] for contradictory or non-actionable comments."""
    raw = str(text or "").strip()
    reasons: List[str] = []
    penalty = 0.0

    words = raw.split()
    has_action = bool(_ACTIONABLE_CUES.search(raw))

    if len(words) < 4:
        penalty += 0.28
        reasons.append("ultra_short_reply(<4 words)")

    for pat in _NON_ACTIONABLE_PATTERNS:
        if pat.search(raw) and not (has_action and len(words) >= 14):
            penalty += 0.28
            reasons.append("non_actionable_social_or_dm_reply")
            break

    for pat in _CONTRADICTION_PATTERNS:
        if pat.search(raw):
            # If comment also gives an actionable alternative ("don't join X, instead prepare for CAT / try AICTE"), soften penalty
            if has_action and len(words) >= 18:
                penalty += 0.08
                reasons.append("cautionary_negation_with_alternative")
            else:
                penalty += 0.25
                reasons.append("negation_or_dismissal_without_actionable_fix")
            break

    if response_type == "counterargument":
        penalty += 0.14 if not has_action else 0.06
        reasons.append("counterargument_response_role")
    elif response_type == "agreement" and not has_action:
        penalty += 0.18
        reasons.append("pure_agreement_no_solution_content")

    return round(min(0.55, penalty), 4), reasons


def _extract_cluster_keywords(texts: Sequence[str], top_k: int = 5) -> List[str]:
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=sorted(_KW_STOP),
            max_features=300,
            token_pattern=r"(?u)[a-zA-Z]{3,}",
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [str(terms[i]) for i in top_idx if scores[i] > 0]
    except ValueError:
        return []


class ContextualIssueSolutionMapper:
    """Contextual multilingual embedding + thread/parent + negation-aware Issue–Solution Mapper."""

    def __init__(
        self,
        *,
        embedding_key: str = "multilingual_e5_small",
        link_threshold: float = 0.56,
        cluster_distance_threshold: float = 0.26,
    ) -> None:
        self.embedding_key = embedding_key
        self.link_threshold = float(link_threshold)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.weights = {
            "embedding_cosine": 0.38,
            "lexical_tfidf_cosine": 0.14,
            "structural_context": 0.18,
            "detector_confidence": 0.30,
        }

    def score_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        use_embedding_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Score a list of candidate (issue, post, parent_context, solution_comment) records.

        Each input dict should include:
          issue_id, post_id, post_title, post_body, solution_comment_id, comment_text,
          parent_context, is_direct_reply_to_op, response_type, solution_probability,
          predicted_issue_labels, issue_probability, language, created_at
        """
        if not candidates:
            return []

        from ml.topics.bertopic.embeddings import encode_documents

        # Build rich issue+thread+parent queries and solution passages
        issue_queries: List[str] = []
        sol_passages: List[str] = []
        for c in candidates:
            iss = str(c["issue_id"])
            desc = ISSUE_DESCRIPTIONS.get(iss, iss.replace("_", " "))
            title = str(c.get("post_title") or "").strip()
            p_ctx = str(c.get("parent_context") or "").strip()[:260]
            q_txt = f"Issue ({iss.replace('_', ' ')}): {desc} | Thread: {title}"
            if p_ctx and p_ctx != title:
                q_txt += f" | Parent: {p_ctx}"
            issue_queries.append(q_txt)

            s_txt = str(c.get("comment_text") or "").strip()
            sol_passages.append(s_txt)

        # 1. Multilingual dense sentence embeddings (cached L2-normalized)
        q_embs, _ = encode_documents(issue_queries, self.embedding_key, is_query=True, use_cache=use_embedding_cache)
        s_embs, _ = encode_documents(sol_passages, self.embedding_key, is_query=False, use_cache=use_embedding_cache)
        raw_cos = np.sum(q_embs * s_embs, axis=1)
        # E5 multilingual cosines typically span [0.68, 0.92]; rescale meaningfully to [0, 1]
        if "e5" in self.embedding_key:
            emb_norm = np.clip((raw_cos - 0.68) / 0.24, 0.0, 1.0)
        else:
            emb_norm = np.clip((raw_cos - 0.15) / 0.65, 0.0, 1.0)

        # 2. Lexical TF-IDF cosine baseline signal (thread title + parent vs solution text)
        tfidf_corpus = [
            f"{c.get('post_title', '')} {str(c.get('parent_context') or '')[:220]}" for c in candidates
        ] + sol_passages
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True, min_df=1)
        X_all = vec.fit_transform(tfidf_corpus)
        n = len(candidates)
        X_q = X_all[:n]
        X_s = X_all[n:]
        tfidf_cos = np.asarray(X_q.multiply(X_s).sum(axis=1)).ravel()

        scored: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            text = sol_passages[i]
            resp_type = str(c.get("response_type") or "")
            sol_prob = float(c.get("solution_probability") or 0.0)
            iss = str(c["issue_id"])
            pred_issues = list(c.get("predicted_issue_labels") or [])
            iss_prob = float(c.get("issue_probability") or 0.0)

            # 3. Structural & parent-context proximity signal
            struct = 0.45  # Same-thread baseline
            if iss in pred_issues:
                struct += 0.30
            elif iss_prob >= 0.25:
                struct += 0.15
            if c.get("is_direct_reply_to_op"):
                struct += 0.15
            elif c.get("parent_context"):
                struct += 0.10
            if _ACTIONABLE_CUES.search(text):
                struct += 0.10
            struct = min(1.0, struct)

            # 4. Upstream detector confidence signal
            role_w = _ROLE_WEIGHTS.get(resp_type, 0.25)
            det_conf = 0.60 * sol_prob + 0.40 * role_w

            # 5. Negation / contradiction / non-actionable penalty
            penalty, pen_reasons = compute_negation_and_non_actionable_penalty(text, response_type=resp_type)

            # Composite score
            s_emb = float(emb_norm[i])
            s_lex = float(min(1.0, tfidf_cos[i] * 2.2))  # boost sparse TF-IDF cosine into usable range
            raw_score = (
                self.weights["embedding_cosine"] * s_emb
                + self.weights["lexical_tfidf_cosine"] * s_lex
                + self.weights["structural_context"] * struct
                + self.weights["detector_confidence"] * det_conf
                - penalty
            )
            match_score = round(float(np.clip(raw_score, 0.0, 1.0)), 4)
            linked = bool(match_score >= self.link_threshold and penalty < 0.28)

            # Build plain-English matching rationale
            rat_parts = [
                f"Multilingual E5 cosine={float(raw_cos[i]):.3f} (norm {s_emb:.2f})",
                f"TF-IDF cosine={float(tfidf_cos[i]):.3f}",
                f"role={resp_type or 'unknown'} (sol_prob={sol_prob:.2f})",
            ]
            if iss in pred_issues:
                rat_parts.append(f"comment directly tagged '{iss}'")
            if pen_reasons:
                rat_parts.append(f"penalized (-{penalty:.2f}: {', '.join(pen_reasons)})")
            rationale = "; ".join(rat_parts)

            scored.append({
                "issue_id": iss,
                "post_id": c["post_id"],
                "post_title": c.get("post_title", ""),
                "solution_comment_id": c["solution_comment_id"],
                "parent_comment_id": c.get("parent_comment_id"),
                "parent_context": str(c.get("parent_context") or c.get("post_title") or "")[:280],
                "matching_score": match_score,
                "tfidf_baseline_score": round(float(0.55 * s_lex + 0.45 * det_conf), 4),
                "linked": linked,
                "model_confidence": round(float(det_conf), 4),
                "response_type": resp_type,
                "solution_probability": round(sol_prob, 4),
                "language": c.get("language", "en"),
                "created_at": c.get("created_at", ""),
                "evidence_text": text,
                "matching_rationale": rationale,
                "verification_status": "model_inferred",
                "signal_breakdown": {
                    "raw_embedding_cosine": round(float(raw_cos[i]), 4),
                    "normalized_embedding_score": round(s_emb, 4),
                    "raw_tfidf_cosine": round(float(tfidf_cos[i]), 4),
                    "structural_context_score": round(struct, 4),
                    "detector_confidence_score": round(det_conf, 4),
                    "negation_contradiction_penalty": penalty,
                    "penalty_reasons": pen_reasons,
                },
                "_embedding": s_embs[i],
            })
        return scored

    def cluster_solutions_by_issue(
        self,
        linked_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group semantically similar linked solutions per issue without hiding distinct recommendations."""
        by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in linked_records:
            if rec.get("linked"):
                by_issue[rec["issue_id"]].append(rec)

        clusters_by_issue: Dict[str, List[Dict[str, Any]]] = {}
        for iss, items in sorted(by_issue.items()):
            if not items:
                clusters_by_issue[iss] = []
                continue
            embs = np.stack([it["_embedding"] for it in items], axis=0)
            if len(items) == 1:
                labels = np.array([0])
            else:
                # Cosine distance = 1 - cosine_similarity; tune threshold so distinct recs stay separate
                sim = np.clip(embs @ embs.T, -1.0, 1.0)
                dist = np.clip(1.0 - sim, 0.0, 2.0)
                # E5 cosine similarities among different MBA advices sit around 0.78–0.85,
                # whereas near-duplicate recommendations sit >= 0.87 (dist <= 0.13)
                thresh = 0.14 if "e5" in self.embedding_key else self.cluster_distance_threshold
                cl = AgglomerativeClustering(
                    n_clusters=None,
                    metric="precomputed",
                    linkage="average",
                    distance_threshold=thresh,
                )
                labels = cl.fit_predict(dist)

            grouped: Dict[int, List[int]] = defaultdict(list)
            for idx, cid in enumerate(labels):
                grouped[int(cid)].append(idx)

            cluster_list: List[Dict[str, Any]] = []
            for cid, idxs in grouped.items():
                members = [items[i] for i in idxs]
                members.sort(key=lambda x: x["matching_score"], reverse=True)
                kws = _extract_cluster_keywords([m["evidence_text"] for m in members], top_k=5)
                rep = members[0]
                label = " · ".join(kws[:3]).title() if kws else f"Solution Pattern {cid + 1}"
                cluster_list.append({
                    "cluster_id": f"{iss}_sol_cluster_{cid}",
                    "issue_id": iss,
                    "cluster_label": label,
                    "keywords": kws,
                    "solution_count": len(members),
                    "distinct_threads": len({m["post_id"] for m in members}),
                    "mean_matching_score": round(float(np.mean([m["matching_score"] for m in members])), 4),
                    "representative_solution": {
                        "solution_comment_id": rep["solution_comment_id"],
                        "post_id": rep["post_id"],
                        "post_title": rep["post_title"],
                        "matching_score": rep["matching_score"],
                        "evidence_text": rep["evidence_text"][:320],
                        "verification_status": rep["verification_status"],
                    },
                    "supporting_comment_ids": [m["solution_comment_id"] for m in members[:10]],
                    "supporting_post_ids": sorted({m["post_id"] for m in members})[:10],
                })

            cluster_list.sort(key=lambda x: (x["solution_count"], x["mean_matching_score"]), reverse=True)
            clusters_by_issue[iss] = cluster_list

        return clusters_by_issue


def evaluate_issue_solution_ranking(
    scored_pairs: Sequence[Dict[str, Any]],
    score_field: str = "matching_score",
    threshold: float = 0.56,
) -> Dict[str, Any]:
    """Compute Precision@k, Recall@k, MRR, NDCG@k, Pairwise P/R/F1, and FPR on annotated pairs.

    Expects each record in ``scored_pairs`` to have ``gold_relevant`` (0 or 1, or graded 0/1/2)
    and ``issue_id`` for per-issue ranking metrics.
    """
    y_true = [1 if int(r["gold_relevant"]) >= 1 else 0 for r in scored_pairs]
    y_pred = [1 if float(r[score_field]) >= threshold else 0 for r in scored_pairs]

    tp = sum(1 for g, p in zip(y_true, y_pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(y_true, y_pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(y_true, y_pred) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(y_true, y_pred) if g == 0 and p == 0)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Group by issue_id for ranking metrics (P@k, R@k, MRR, NDCG@k)
    by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored_pairs:
        by_issue[str(r["issue_id"])].append(r)

    p_at = {1: [], 3: [], 5: []}
    r_at = {1: [], 3: [], 5: []}
    ndcg_at = {3: [], 5: []}
    rrs: List[float] = []

    for iss, grp in sorted(by_issue.items()):
        ranked = sorted(grp, key=lambda x: float(x[score_field]), reverse=True)
        rels = [int(x["gold_relevant"]) for x in ranked]
        bin_rels = [1 if rv >= 1 else 0 for rv in rels]
        n_pos = sum(bin_rels)
        if n_pos == 0:
            continue

        # MRR
        rr = 0.0
        for rank_idx, b in enumerate(bin_rels, 1):
            if b == 1:
                rr = 1.0 / rank_idx
                break
        rrs.append(rr)

        for k in (1, 3, 5):
            top_k = bin_rels[: min(k, len(bin_rels))]
            p_at[k].append(sum(top_k) / max(len(top_k), 1))
            r_at[k].append(sum(top_k) / n_pos)

        for k in (3, 5):
            kk = min(k, len(rels))
            dcg = sum((2 ** rels[idx] - 1) / math.log2(idx + 2) for idx in range(kk))
            ideal = sorted(rels, reverse=True)[:kk]
            idcg = sum((2 ** ideal[idx] - 1) / math.log2(idx + 2) for idx in range(kk))
            ndcg_at[k].append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "n_pairs": len(scored_pairs),
        "n_issues_evaluated": len(rrs),
        "threshold": threshold,
        "pairwise_precision": round(prec, 4),
        "pairwise_recall": round(rec, 4),
        "pairwise_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mrr": round(float(np.mean(rrs)) if rrs else 0.0, 4),
        "precision_at_1": round(float(np.mean(p_at[1])) if p_at[1] else 0.0, 4),
        "precision_at_3": round(float(np.mean(p_at[3])) if p_at[3] else 0.0, 4),
        "precision_at_5": round(float(np.mean(p_at[5])) if p_at[5] else 0.0, 4),
        "recall_at_1": round(float(np.mean(r_at[1])) if r_at[1] else 0.0, 4),
        "recall_at_3": round(float(np.mean(r_at[3])) if r_at[3] else 0.0, 4),
        "recall_at_5": round(float(np.mean(r_at[5])) if r_at[5] else 0.0, 4),
        "ndcg_at_3": round(float(np.mean(ndcg_at[3])) if ndcg_at[3] else 0.0, 4),
        "ndcg_at_5": round(float(np.mean(ndcg_at[5])) if ndcg_at[5] else 0.0, 4),
    }
