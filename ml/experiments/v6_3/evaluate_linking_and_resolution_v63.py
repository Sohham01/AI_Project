"""Sprint P1-E (Part B) — Reproducible Evaluator for Issue–Solution Linking & Resolution Engine.

Evaluates the updated P0-B ExplainableResolutionEngine and ContextualIssueSolutionMapper
against:
  1. Existing v1 human-reviewed sets (`n=64` linking pairs, `n=40` threads) —
     explicitly labeled as ``human_reviewed_development_set_v1`` (NOT independent test).
  2. Independent v2 human-reviewed sets (`v2_independent_eval`, once annotated).

Reconstructs the full corpus linked-solution candidate edges via
``ContextualIssueSolutionMapper`` (with v2 cached embeddings) rather than truncated
``top_matched_solutions[:12]`` UI slices, and reports per-class P/R/F1, macro & weighted
F1, FP/FN samples, evidence retrieval, separated false ``reported_success`` vs false
``Partially Resolved`` claim rates, and community/language breakdowns.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from ml.linking.contextual_mapper import (
    ISSUE_DESCRIPTIONS,
    ContextualIssueSolutionMapper,
    evaluate_issue_solution_ranking,
)
from ml.linking.resolution_engine import (
    ExplainableResolutionEngine,
    evaluate_resolution_predictions,
)

BASELINE_CORPUS = ROOT / "data" / "live_corpus" / "corpus.json"
SERVING_MAP_V2 = ROOT / "ml" / "artifacts" / "live_corpus" / "issue_solution_map_v2.json"
PREDICTIONS_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json"
V1_RES_CSV = ROOT / "data" / "annotations" / "resolution_eval_v1.csv"
V1_ISS_CSV = ROOT / "data" / "annotations" / "issue_solution_eval_v1.csv"
OUT_DIR = ROOT / "ml" / "artifacts" / "v6_3_experiments" / "linking_resolution_eval"


def run_dev_and_status_evaluation() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(BASELINE_CORPUS.read_text(encoding="utf-8"))
    preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))["tasks"]
    serving_map = json.loads(SERVING_MAP_V2.read_text(encoding="utf-8"))

    posts = corpus.get("posts", [])
    comments_by_post: Dict[str, List[Dict[str, Any]]] = {}
    post_meta: Dict[str, Dict[str, Any]] = {}
    comment_lookup: Dict[str, Dict[str, Any]] = {}
    parent_text_by_cid: Dict[str, str] = {}

    for p in posts:
        pid = p["reddit_id"]
        post_meta[pid] = p
        cmts = p.get("comments", [])
        by_cid_local = {c["reddit_id"]: c for c in cmts}
        enriched = []
        for c in cmts:
            cid = c["reddit_id"]
            dk = "comment-" + cid
            c_pred = {t: preds.get(t, {}).get(dk, {}) for t in ("issue", "sentiment", "emotion", "response", "solution")}
            p_cid = c.get("parent_reddit_id")
            parent_text_by_cid[cid] = by_cid_local[p_cid]["body"] if (p_cid and p_cid in by_cid_local) else p["title"]
            rec = {**c, "post_id": pid, "post_title": p["title"], "predictions": c_pred}
            enriched.append(rec)
            comment_lookup[cid] = rec
        comments_by_post[pid] = enriched

    # Build full corpus candidate links matching Phase 4 protocol (v2 disk cached embeddings)
    candidates: List[Dict[str, Any]] = []
    seen_pairs = set()
    for p in posts:
        pid = p["reddit_id"]
        post_issue = p.get("issue_key") or "administrative_communication"
        for c in comments_by_post[pid]:
            cid = c["reddit_id"]
            body = str(c.get("body") or "").strip()
            if not body or body.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
                continue
            cp = c["predictions"]
            rt = str(cp["response"].get("response_type") or "")
            sp = float(cp["solution"].get("solution_probability") or 0.0)
            is_sol = bool(cp["solution"].get("is_solution"))
            pred_issues = list(cp["issue"].get("predicted_issue_labels") or [])
            iss_probs = cp["issue"].get("issue_probabilities") or {}
            if not (is_sol or rt in ("solution", "advice") or sp >= 0.42):
                continue
            lid = detect_indian_reddit_language(body)
            for iss in (set(pred_issues) | {post_issue}):
                if iss not in ISSUE_DESCRIPTIONS or (iss, pid, cid) in seen_pairs:
                    continue
                seen_pairs.add((iss, pid, cid))
                candidates.append({
                    "issue_id": iss,
                    "post_id": pid,
                    "post_title": p["title"],
                    "post_body": p.get("body") or "",
                    "solution_comment_id": cid,
                    "parent_comment_id": c.get("parent_reddit_id"),
                    "parent_context": parent_text_by_cid.get(cid, p["title"]),
                    "is_direct_reply_to_op": not bool(c.get("parent_reddit_id")),
                    "comment_text": body,
                    "response_type": rt,
                    "solution_probability": sp,
                    "predicted_issue_labels": pred_issues,
                    "issue_probability": float(iss_probs.get(iss, 0.0)),
                    "language": lid["language"],
                    "created_at": c.get("created_at", ""),
                })

    mapper = ContextualIssueSolutionMapper(embedding_key="multilingual_e5_small", link_threshold=0.56)
    scored_all = mapper.score_candidates(candidates, use_embedding_cache=True)
    linked_all = [r for r in scored_all if r["linked"]]

    # Evaluate Issue-Solution v1 (n = 64)
    iss_cands: List[Dict[str, Any]] = []
    iss_gold_meta: List[Dict[str, Any]] = []
    with V1_ISS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            pid = r["post_id"]
            cid = r["solution_comment_id"]
            iss = r["issue_id"]
            p = post_meta[pid]
            c = comment_lookup[cid]
            cp = c["predictions"]
            rt = str(cp["response"].get("response_type") or "")
            sp = float(cp["solution"].get("solution_probability") or 0.0)
            pred_issues = list(cp["issue"].get("predicted_issue_labels") or [])
            iss_probs = cp["issue"].get("issue_probabilities") or {}
            iss_cands.append({
                "issue_id": iss,
                "post_id": pid,
                "post_title": p["title"],
                "post_body": p.get("body") or "",
                "solution_comment_id": cid,
                "parent_comment_id": c.get("parent_reddit_id"),
                "parent_context": parent_text_by_cid.get(cid, p["title"]),
                "is_direct_reply_to_op": not bool(c.get("parent_reddit_id")),
                "comment_text": c["body"],
                "response_type": rt,
                "solution_probability": sp,
                "predicted_issue_labels": pred_issues,
                "issue_probability": float(iss_probs.get(iss, 0.0)),
                "language": detect_indian_reddit_language(c["body"])["language"],
                "created_at": c.get("created_at", ""),
            })
            iss_gold_meta.append({"gold_relevant": int(r["gold_relevant"]), "subreddit": p.get("subreddit", "unknown")})

    iss_scored = mapper.score_candidates(iss_cands, use_embedding_cache=True)
    for rec, gm in zip(iss_scored, iss_gold_meta):
        rec["gold_relevant"] = gm["gold_relevant"]
        rec["subreddit"] = gm["subreddit"]
    iss_v1_metrics = evaluate_issue_solution_ranking(iss_scored, score_field="matching_score", threshold=0.56)

    # Evaluate Resolution Engine on full linked_all across 365 threads & v1 (n=40)
    engine = ExplainableResolutionEngine()
    res_full = engine.analyze_corpus(posts, comments_by_post, linked_all)
    pred_by_pid = {t["post_id"]: t for t in res_full["threads"]}

    gold_v1: List[Dict[str, Any]] = []
    with V1_RES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            gold_v1.append({
                "post_id": r["post_id"],
                "gold_resolution_state": r["gold_resolution_state"],
                "gold_is_unresolved": str(r["gold_is_unresolved"]).lower() == "true",
                "gold_has_contradiction": str(r.get("gold_has_contradiction", "")).lower() == "true",
                "subreddit": post_meta.get(r["post_id"], {}).get("subreddit", "unknown"),
            })

    v1_res_metrics = evaluate_resolution_predictions(gold_v1, pred_by_pid)

    # Evidence retrieval check on v1
    ev_ok = 0
    ev_tot = 0
    for r in gold_v1:
        pt = pred_by_pid[r["post_id"]]
        gs = r["gold_resolution_state"]
        ev_tot += 1
        if gs == "reported_success" and pt["success_evidence"]:
            ev_ok += 1
        elif gs == "reported_pending" and (pt["pending_evidence"] or pt["contradictory_evidence"]):
            ev_ok += 1
        elif gs == "proposed_only" and pt["proposed_solution"]:
            ev_ok += 1
        elif gs == "insufficient_evidence" and pt["resolution_state"] == "insufficient_evidence":
            ev_ok += 1

    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in gold_v1:
        by_sub[row["subreddit"]].append(row)
    sub_breakdown = {
        sub: {
            "n_threads": len(rows),
            "accuracy": evaluate_resolution_predictions(rows, pred_by_pid)["four_state_accuracy"],
            "macro_f1": evaluate_resolution_predictions(rows, pred_by_pid)["four_state_macro_f1"],
            "low_support_warning": len(rows) < 5,
        }
        for sub, rows in sorted(by_sub.items())
    }

    report = {
        "evaluation_scope": "DEVELOPMENT_SET_ONLY_V1 (used during V6.0/V6.3 rule development — NOT independent blind evaluation!)",
        "independent_v2_status": "PENDING_HUMAN_REVIEW (data/annotations/v6_3_package/linking_and_resolution_v2/)",
        "timestamp_semantics": res_full["timestamp_semantics"],
        "full_corpus_linked_solutions": len(linked_all),
        "corpus_365_summary": res_full["overall_summary"],
        "resolution_v1_dev_metrics": {
            **v1_res_metrics,
            "evidence_retrieval_accuracy": round(ev_ok / max(ev_tot, 1), 4),
        },
        "resolution_v1_by_community": sub_breakdown,
        "issue_solution_v1_dev_metrics": iss_v1_metrics,
    }
    (OUT_DIR / "linking_resolution_v1_dev_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    rep = run_dev_and_status_evaluation()
    print(json.dumps({
        "scope": rep["evaluation_scope"],
        "full_corpus_linked_solutions": rep["full_corpus_linked_solutions"],
        "v1_res_accuracy": rep["resolution_v1_dev_metrics"]["four_state_accuracy"],
        "v1_res_macro_f1": rep["resolution_v1_dev_metrics"]["four_state_macro_f1"],
        "v1_res_evidence_retrieval": rep["resolution_v1_dev_metrics"]["evidence_retrieval_accuracy"],
        "false_claims": rep["resolution_v1_dev_metrics"]["false_claim_breakdown"],
        "v1_iss_pairwise_f1": rep["issue_solution_v1_dev_metrics"]["pairwise_f1"],
        "v1_iss_p3": rep["issue_solution_v1_dev_metrics"]["precision_at_3"],
        "corpus_365": rep["corpus_365_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
