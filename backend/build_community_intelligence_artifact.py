"""Offline builder for `ml/artifacts/live_corpus/community_intelligence.json`.

Runs the real `TfidfCosineIssueSolutionLinker` across all anonymized live threads
to link detected community solutions to identified issues (Task 10), clusters
linked solutions into ranked "Most Common Community Solutions" per issue domain,
and bundles the verified Task 5 (LID) & Task 6 (Preprocessing) ablation summary
for zero-GPU, pure-stdlib serving in FastAPI (Docker + Vercel).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.nlp.community_intelligence import (
    CANONICAL_ISSUES,
    ISSUE_DISPLAY_LABELS,
    _SOLUTION_THEME_RULES,
    _STOP_LITE,
    _TOKEN_RE,
)
from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence,
    LinkModelConfig,
    SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.tfidf_cosine_linker import (
    TfidfCosineIssueSolutionLinker,
)

CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
PRED_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json"
TASK56_PATH = ROOT / "ml" / "artifacts" / "v5_experiments" / "task5_task6_ablations.json"
OUT_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "community_intelligence.json"


def build_community_intelligence_artifact() -> Dict[str, Any]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    preds = json.loads(PRED_PATH.read_text(encoding="utf-8"))["tasks"]
    posts = corpus["posts"]

    post_by_id = {p["reddit_id"]: p for p in posts}
    cmt_by_id: Dict[str, Dict[str, Any]] = {}

    issue_evidences: List[IssueEvidence] = []
    solution_evidences: List[SolutionEvidence] = []

    for post in posts:
        pid = post["reddit_id"]
        cmts = post.get("comments") or []

        # Aggregate thread-level active issues from post primary issue + predicted comment issues
        thread_issue_counts: Counter[str] = Counter()
        if post.get("issue_key") in ISSUE_DISPLAY_LABELS:
            thread_issue_counts[post["issue_key"]] += 2
        for c in cmts:
            doc_id = c["document_id"]
            cmt_by_id[doc_id] = {**c, "post_id": pid, "post_title": post["title"]}
            iss_p = preds.get("issue", {}).get(doc_id, {})
            for lbl in iss_p.get("predicted_issue_labels") or []:
                if lbl in ISSUE_DISPLAY_LABELS:
                    thread_issue_counts[lbl] += 1

        thread_labels = [lbl for lbl, _ in thread_issue_counts.most_common(3)] or [post["issue_key"]]
        post_text = f"{post.get('title', '')} {post.get('body') or ''}".strip()
        issue_evidences.append(
            IssueEvidence(
                document_id=f"issue::{pid}",
                post_id=pid,
                original_text=post_text,
                normalized_text=post_text.lower(),
                issue_labels=thread_labels,
            )
        )

        # Every predicted solution or advice comment is a candidate for thread-scoped linking
        for c in cmts:
            doc_id = c["document_id"]
            sol_p = preds.get("solution", {}).get(doc_id, {})
            resp_p = preds.get("response", {}).get(doc_id, {})
            is_sol = bool(sol_p.get("is_solution"))
            sol_prob = float(sol_p.get("solution_probability") or 0.0)
            resp_type = resp_p.get("response_type")
            if is_sol or resp_type in {"solution", "advice"} or sol_prob >= 0.40:
                body = str(c.get("body") or "").strip()
                solution_evidences.append(
                    SolutionEvidence(
                        document_id=doc_id,
                        post_id=pid,
                        comment_id=c["reddit_id"],
                        original_text=body,
                        normalized_text=body.lower(),
                        is_solution=is_sol,
                        solution_probability=sol_prob,
                        response_type=resp_type,
                        parent_comment_id=c.get("parent_reddit_id"),
                        language=str(c.get("language") or "en"),
                        script=str(c.get("script") or "latin"),
                        is_code_mixed=bool(c.get("is_code_mixed", False)),
                    )
                )

    linker = TfidfCosineIssueSolutionLinker(
        config=LinkModelConfig(
            same_post_only=True,
            link_threshold=0.45,
            min_solution_probability=0.35,
            include_non_solution_candidates=True,
        )
    )
    result = linker.link(issue_evidences, solution_evidences)
    asserted_links = [lnk for lnk in result.links if lnk.linked]

    # Summarize per-issue linked evidence + ranked community solution clusters
    links_by_issue: Dict[str, List[Any]] = defaultdict(list)
    for lnk in asserted_links:
        if lnk.issue_label in ISSUE_DISPLAY_LABELS:
            links_by_issue[lnk.issue_label].append(lnk)

    by_issue_stats: Dict[str, Any] = {}
    issue_solution_clusters: Dict[str, Any] = {}

    for issue_key in CANONICAL_ISSUES:
        iss_links = links_by_issue.get(issue_key, [])
        # Deduplicate by solution comment ID keeping highest link_score
        best_by_sol: Dict[str, Any] = {}
        for lnk in iss_links:
            sid = lnk.solution_document_id
            if sid not in best_by_sol or lnk.link_score > best_by_sol[sid].link_score:
                best_by_sol[sid] = lnk

        unique_links = sorted(best_by_sol.values(), key=lambda x: -x.link_score)
        threads_covered = {lnk.issue_post_id for lnk in unique_links if lnk.issue_post_id}
        mean_cos = (
            sum(float(lnk.similarity_score) for lnk in unique_links) / len(unique_links)
            if unique_links
            else 0.0
        )
        mean_score = (
            sum(float(lnk.link_score) for lnk in unique_links) / len(unique_links)
            if unique_links
            else 0.0
        )

        by_issue_stats[issue_key] = {
            "issue_key": issue_key,
            "issue_label": ISSUE_DISPLAY_LABELS[issue_key],
            "linked_solutions": len(unique_links),
            "threads_with_linked_solution": len(threads_covered),
            "mean_cosine_similarity": round(mean_cos, 4),
            "mean_link_score": round(mean_score, 4),
            "top_linked_pairs": [
                {
                    "post_id": lnk.issue_post_id,
                    "post_title": post_by_id.get(lnk.issue_post_id, {}).get("title", ""),
                    "comment_id": cmt_by_id.get(lnk.solution_document_id, {}).get("reddit_id", lnk.solution_document_id),
                    "solution_text": str(cmt_by_id.get(lnk.solution_document_id, {}).get("body", ""))[:240],
                    "cosine_similarity": round(float(lnk.similarity_score), 4),
                    "link_score": round(float(lnk.link_score), 4),
                    "solution_probability": round(float(lnk.solution_probability), 4),
                    "response_type": lnk.signals.get("response_type"),
                }
                for lnk in unique_links[:5]
            ],
        }

        # Cluster linked solutions into ranked "Most Common Community Solutions" themes
        rules = _SOLUTION_THEME_RULES.get(issue_key, [])
        buckets: Dict[str, List[Any]] = defaultdict(list)
        for lnk in unique_links:
            cmt = cmt_by_id.get(lnk.solution_document_id, {})
            toks = set(_TOKEN_RE.findall(str(cmt.get("body") or "").lower()))
            best_theme = "General Peer Guidance & Experience"
            best_hits = 0
            for theme_name, kws in rules:
                hits = sum(1 for kw in kws if kw in toks)
                if hits > best_hits:
                    best_hits = hits
                    best_theme = theme_name
            buckets[best_theme].append((best_hits, lnk, cmt))

        n_unique = max(len(unique_links), 1)
        clusters = []
        for theme_name, entries in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            entries.sort(key=lambda x: (-x[0], -float(x[1].link_score)))
            kw_counter: Counter[str] = Counter()
            for _, _, cmt in entries:
                for tok in _TOKEN_RE.findall(str(cmt.get("body") or "").lower()):
                    if tok not in _STOP_LITE and len(tok) >= 3:
                        kw_counter[tok] += 1
            mean_theme_score = sum(float(e[1].link_score) for e in entries) / len(entries)
            mean_theme_cos = sum(float(e[1].similarity_score) for e in entries) / len(entries)
            clusters.append({
                "theme": theme_name,
                "count": len(entries),
                "share": round(len(entries) / n_unique, 4),
                "mean_link_score": round(mean_theme_score, 4),
                "mean_cosine_similarity": round(mean_theme_cos, 4),
                "top_terms": [w for w, _ in kw_counter.most_common(6)],
                "examples": [
                    {
                        "comment_id": e[2].get("reddit_id", e[1].solution_document_id),
                        "post_id": e[1].issue_post_id,
                        "post_title": e[2].get("post_title", ""),
                        "text": str(e[2].get("body", ""))[:240],
                        "link_score": round(float(e[1].link_score), 3),
                        "cosine_similarity": round(float(e[1].similarity_score), 3),
                        "solution_probability": round(float(e[1].solution_probability), 3),
                        "response_type": e[1].signals.get("response_type"),
                    }
                    for e in entries[:2]
                ],
            })

        issue_solution_clusters[issue_key] = {
            "issue_key": issue_key,
            "issue_label": ISSUE_DISPLAY_LABELS[issue_key],
            "linked_solution_count": len(unique_links),
            "threads_covered": len(threads_covered),
            "mean_cosine_similarity": round(mean_cos, 4),
            "mean_link_score": round(mean_score, 4),
            "clusters": clusters[:5],
        }

    unique_linked_sols = {lnk.solution_document_id for lnk in asserted_links}
    unique_linked_posts = {lnk.issue_post_id for lnk in asserted_links if lnk.issue_post_id}
    cfg_dump = linker.config_dict
    linker_summary = {
        "model_name": linker.name,
        "model_version": linker.version,
        "same_post_only": True,
        "link_threshold": linker.link_threshold,
        "weights": {
            "semantic_cosine": cfg_dump.get("similarity_weight", cfg_dump.get("semantic_weight", 0.45)),
            "solution_probability": cfg_dump.get("solution_probability_weight", 0.25),
            "thread_relevance": cfg_dump.get("thread_weight", cfg_dump.get("thread_relevance_weight", 0.20)),
            "response_role": cfg_dump.get("response_role_weight", 0.10),
        },
        "total_issue_evidences": len(issue_evidences),
        "total_solution_candidates": len(solution_evidences),
        "total_candidate_pairs": len(result.links),
        "total_asserted_links": len(asserted_links),
        "unique_linked_solutions": len(unique_linked_sols),
        "threads_with_linked_solutions": len(unique_linked_posts),
        "mean_cosine_similarity": round(
            sum(float(lnk.similarity_score) for lnk in asserted_links) / max(len(asserted_links), 1), 4
        ),
        "mean_link_score": round(
            sum(float(lnk.link_score) for lnk in asserted_links) / max(len(asserted_links), 1), 4
        ),
    }

    task56_summary = None
    if TASK56_PATH.is_file():
        task56_summary = json.loads(TASK56_PATH.read_text(encoding="utf-8"))

    payload = {
        "dataset_version": corpus["metadata"]["dataset_version"],
        "synthetic_only": bool(corpus["metadata"].get("synthetic", False)),
        "linker_summary": linker_summary,
        "by_issue": by_issue_stats,
        "issue_solution_clusters": issue_solution_clusters,
        "task5_task6_ablations": task56_summary,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built {OUT_PATH.name}: {linker_summary['total_asserted_links']} asserted links "
        f"({linker_summary['unique_linked_solutions']} unique solutions across "
        f"{linker_summary['threads_with_linked_solutions']}/{len(posts)} threads, "
        f"mean cos={linker_summary['mean_cosine_similarity']:.3f}, "
        f"mean blend={linker_summary['mean_link_score']:.3f})"
    )
    return payload


if __name__ == "__main__":
    build_community_intelligence_artifact()
