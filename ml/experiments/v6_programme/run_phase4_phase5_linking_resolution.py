"""Phases 4 & 5 Runner — Contextual Issue–Solution Mapping + Explainable Unresolved-Issue Detection.

Executes end-to-end on the real anonymized 365-thread / 3,557-comment RCI corpus:
1. Runs ``ContextualIssueSolutionMapper`` (multilingual E5-small + lexical TF-IDF +
   parent/thread context + upstream detector confidence - negation/contradiction penalty)
   and semantic solution clustering across all 8 issue domains.
2. Evaluates V6 Contextual Mapper vs. V5 TF-IDF baseline on a manually reviewed
   64-pair real benchmark (``data/annotations/issue_solution_eval_v1.csv``) reporting
   P@k, R@k, MRR, NDCG@k, Pairwise P/R/F1, and False-Positive Rate.
3. Runs ``ExplainableResolutionEngine`` across all 365 threads, preserving all 4
   distinct states (``reported_success``, ``reported_pending``, ``proposed_only``,
   ``insufficient_evidence``), contradictory follow-up evidence, elapsed time, and
   transparent verdict scoring (``Resolved``, ``Partially Resolved``, ``Unresolved``,
   ``Insufficient Evidence``).
4. Evaluates Phase 5 against a manually reviewed 40-thread resolution benchmark
   (``data/annotations/resolution_eval_v1.csv``), measuring 4-state accuracy/F1,
   false-resolution claim rate, unresolved-issue F1, and evidence retrieval accuracy.
5. Saves compact Vercel-ready serving JSON artifacts:
   - ``ml/artifacts/live_corpus/issue_solution_map_v2.json``
   - ``ml/artifacts/live_corpus/resolution_engine_v2.json``
   and full experiment metrics in ``ml/artifacts/v6_programme/``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    RESOLUTION_STATES,
    ExplainableResolutionEngine,
)

CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
PREDICTIONS_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json"
ISS_EVAL_CSV = ROOT / "data" / "annotations" / "issue_solution_eval_v1.csv"
RES_EVAL_CSV = ROOT / "data" / "annotations" / "resolution_eval_v1.csv"
SERVING_MAP_V2 = ROOT / "ml" / "artifacts" / "live_corpus" / "issue_solution_map_v2.json"
SERVING_RES_V2 = ROOT / "ml" / "artifacts" / "live_corpus" / "resolution_engine_v2.json"
EVAL_OUT_DIR = ROOT / "ml" / "artifacts" / "v6_programme"

# ---------------------------------------------------------------------------
# Phase 4: Manually reviewed 64 real (issue_id, post_id, comment_id) pairs
# gold_relevant: 2 = strong actionable solution addressing the issue,
#                1 = partial / conditional helpful advice addressing the issue,
#                0 = irrelevant / non-actionable / contradictory / sarcasm / removed
# ---------------------------------------------------------------------------
PHASE4_HUMAN_PAIRS: List[Tuple[str, str, str, int, str]] = [
    # academic_workload (8)
    ("academic_workload", "post_0029", "cmt_00277", 2, "Actionable MBA preparation advice: live projects, PORs, CV building"),
    ("academic_workload", "post_0029", "cmt_00237", 2, "Actionable domain prep strategy for FMCG/marketing coursework"),
    ("academic_workload", "post_0058", "cmt_00550", 1, "Conditional exam-prep advice (repeat CAT with workex)"),
    ("academic_workload", "post_0061", "cmt_00554", 1, "Exam prep & repeat recommendation"),
    ("academic_workload", "post_0107", "cmt_01073", 1, "Repeat CAT target score advice"),
    ("academic_workload", "post_0043", "cmt_00422", 0, "Bare number '8 LPA' — non-actionable"),
    ("academic_workload", "post_0066", "cmt_00647", 0, "'Please dm me' — non-actionable DM redirect"),
    ("academic_workload", "post_0191", "cmt_02018", 0, "'You also get an IIM TAG BROTHER' — hype, not a solution"),
    # administrative_communication (8)
    ("administrative_communication", "post_0070", "cmt_00726", 2, "Direct factual resolution on IIM Sambalpur sectional cutoff criteria"),
    ("administrative_communication", "post_0153", "cmt_01507", 2, "Actionable coordination step: create admitted batch WhatsApp group"),
    ("administrative_communication", "post_0004", "cmt_00014", 2, "Actionable refund remedy: check AICTE refund guidelines before classes start"),
    ("administrative_communication", "post_0006", "cmt_00023", 1, "Specific category waitlist movement benchmark (~300 SBP, ~3000 Sirmaur)"),
    ("administrative_communication", "post_0307", "cmt_02955", 0, "Ultra-short 2-word 'Clear it' — non-actionable"),
    ("administrative_communication", "post_0295", "cmt_02867", 0, "Clarifying question 'You general?' — not a solution"),
    ("administrative_communication", "post_0285", "cmt_02776", 0, "Bare number '93.27' — not a solution"),
    ("administrative_communication", "post_0069", "cmt_00695", 0, "Pure gratitude/agreement 'haanji bhai, dhanyawad'"),
    # campus_infrastructure (8)
    ("campus_infrastructure", "post_0212", "cmt_02193", 2, "Actionable laptop hardware breakdown for coding/analytics coursework"),
    ("campus_infrastructure", "post_0212", "cmt_02194", 2, "Direct pointer to curated laptop specs list on campus Discord"),
    ("campus_infrastructure", "post_0014", "cmt_00072", 2, "Concrete local campus-access clarification: VIMSAR hospital 2km away"),
    ("campus_infrastructure", "post_0052", "cmt_00519", 1, "Comparative campus/B-school choice advice"),
    ("campus_infrastructure", "post_0087", "cmt_00883", 0, "Dismissive critique calling new IIMs a 'ponzi scheme' — not a solution"),
    ("campus_infrastructure", "post_0155", "cmt_01524", 0, "Emoji-only '🙏' — non-actionable"),
    ("campus_infrastructure", "post_0363", "cmt_03533", 0, "Sarcastic dismissal of mess hygiene complaints"),
    ("campus_infrastructure", "post_0362", "cmt_03525", 0, "Exclamation 'Wakkkkkk thuuuu' — non-actionable"),
    # curriculum_teaching (8)
    ("curriculum_teaching", "post_0106", "cmt_01064", 2, "Structured program-path advice: Executive MBA after 5 years workex"),
    ("curriculum_teaching", "post_0077", "cmt_00763", 2, "Detailed comparative ranking of BA/MBA curriculum programs across colleges"),
    ("curriculum_teaching", "post_0106", "cmt_01065", 1, "Concise executive program alternative after 1 year workex"),
    ("curriculum_teaching", "post_0029", "cmt_00248", 1, "Informed peer comparison of IIM Udaipur academic trajectory"),
    ("curriculum_teaching", "post_0077", "cmt_00781", 0, "Well-wish 'Hope you get the best college' — non-actionable"),
    ("curriculum_teaching", "post_0029", "cmt_00262", 0, "Question '100% placement this year?' — not a solution"),
    ("curriculum_teaching", "post_0181", "cmt_01903", 0, "Mocking reply 'chat gpt ahh' — non-actionable"),
    ("curriculum_teaching", "post_0064", "cmt_00589", 0, "Dismissive opinion 'only IIM ABC tag helps' — no actionable guidance"),
    # extracurriculars (8)
    ("extracurriculars", "post_0088", "cmt_00900", 2, "Actionable PI & profile-gap mitigation strategy: build structured explanations"),
    ("extracurriculars", "post_0170", "cmt_01775", 2, "Direct factual campus-life & room-allotment guidance from current student"),
    ("extracurriculars", "post_0013", "cmt_00064", 2, "Detailed alum guidance on leveraging clubs, committees & case competitions"),
    ("extracurriculars", "post_0077", "cmt_00793", 1, "Interview cross-questioning preparedness advice"),
    ("extracurriculars", "post_0232", "cmt_02414", 0, "Status remark 'I'm also joining' — not a solution"),
    ("extracurriculars", "post_0170", "cmt_01764", 0, "Short gratitude 'Ohkk Thankss' — non-actionable"),
    ("extracurriculars", "post_0232", "cmt_02398", 0, "Personal remark 'Bhai Mai khud ladki hu😭?' — not a solution"),
    ("extracurriculars", "post_0358", "cmt_03478", 0, "Single-word social reply 'Chalo' — non-actionable"),
    # financial_concerns (8)
    ("financial_concerns", "post_0004", "cmt_00016", 2, "Concrete AICTE 15-day full-refund guideline (max 1k deduction) + ROI analysis"),
    ("financial_concerns", "post_0083", "cmt_00809", 2, "Actionable scholarship/financial-aid guidance (25% coverage options)"),
    ("financial_concerns", "post_0217", "cmt_02208", 1, "Payment portal troubleshooting guidance (temporary site glitch)"),
    ("financial_concerns", "post_0183", "cmt_01942", 1, "Direct factual clarification on SC scholarship availability vs 26L fee"),
    ("financial_concerns", "post_0122", "cmt_01201", 0, "'[ Removed by Reddit ]' placeholder — must never be linked!"),
    ("financial_concerns", "post_0357", "cmt_03453", 0, "Rhetorical question 'Are these institutions placement agencies?'"),
    ("financial_concerns", "post_0185", "cmt_01969", 0, "Short remark 'I pay the fees' — non-actionable"),
    ("financial_concerns", "post_0191", "cmt_02016", 0, "Meta-argument accusing OP of defaming program — not a financial solution"),
    # hostel_facilities (8)
    ("hostel_facilities", "post_0194", "cmt_02059", 2, "Practical hostel workaround: electric kettle allowed for light cooking"),
    ("hostel_facilities", "post_0194", "cmt_02054", 1, "Clear hostel rule clarification ('U can't cook inside room')"),
    ("hostel_facilities", "post_0013", "cmt_00064", 1, "Firsthand alum mess food quality assessment vs other colleges"),
    ("hostel_facilities", "post_0170", "cmt_01775", 1, "Firsthand hostel room allotment procedure upon campus arrival"),
    ("hostel_facilities", "post_0322", "cmt_03192", 0, "Bare image URL only — must never be linked as solution!"),
    ("hostel_facilities", "post_0362", "cmt_03523", 0, "Sarcastic joke about rat getting VIP buffet access"),
    ("hostel_facilities", "post_0322", "cmt_03193", 0, "Reaction question 'Ain't tht worm ?' — not a solution"),
    ("hostel_facilities", "post_0363", "cmt_03537", 0, "One-word joke 'Ratatouille' — non-actionable"),
    # placements (8)
    ("placements", "post_0036", "cmt_00369", 2, "Clear placement/location/ROI ranking (FORE < SBP < IMI Delhi)"),
    ("placements", "post_0077", "cmt_00787", 2, "Specific recruiter-backed college selection guidance (BITSOM / IMI D DE Shaw)"),
    ("placements", "post_0288", "cmt_02799", 2, "Actionable alternative career/program recommendation over secondary programs"),
    ("placements", "post_0031", "cmt_00334", 2, "Concrete due-diligence method: verify 2023-25 alumni 'Open to Work' on LinkedIn"),
    ("placements", "post_0052", "cmt_00527", 1, "Pragmatic advice to sit both interviews and decide after converts"),
    ("placements", "post_0029", "cmt_00229", 0, "Question 'What helped you clear GDs' — not a solution"),
    ("placements", "post_0104", "cmt_01053", 0, "One-word agreement 'Sure' — non-actionable"),
    ("placements", "post_0064", "cmt_00589", 0, "Dismissive counterargument 'only IIM ABC helps' — no actionable path"),
]

# ---------------------------------------------------------------------------
# Phase 5: Manually reviewed 40 real corpus threads for resolution evaluation
# Format: (post_id, gold_state, gold_is_unresolved, has_contradiction, note)
# gold_is_unresolved = True if NOT confirmed resolved (reported_pending, proposed_only, insufficient_evidence)
# ---------------------------------------------------------------------------
PHASE5_HUMAN_THREADS: List[Tuple[str, str, bool, bool, str]] = [
    # Reported success (8 genuine confirmed outcome threads)
    ("post_0131", "reported_success", False, False, "Confirmed WL-8 General offer letter received on 29th June"),
    ("post_0155", "reported_success", False, False, "Multiple students confirm receiving Merit List offer today"),
    ("post_0157", "reported_success", False, False, "Confirmed WL-46 converted and received offer in Merit List 2"),
    ("post_0158", "reported_success", False, False, "OP + commenters confirm final admission conversion ('Finally ho gya')"),
    ("post_0107", "reported_success", False, False, "Commenter confirms off-campus analytics course led to placement"),
    ("post_0170", "reported_success", False, False, "Alum confirms completed year + MPP offer letter confirmation"),
    ("post_0135", "reported_pending", True, False, "OP reports portal shows 'Your offer has been rejected' after EWS WL-7 — unresolved!"),
    ("post_0132", "reported_success", False, False, "OP confirms final selection for IIM Sambalpur MBA Core"),
    # Reported pending / contradictory unresolved (12 threads)
    ("post_0150", "reported_pending", True, True, "2nd list delayed; callers report admin 'Did not pick up'"),
    ("post_0151", "reported_pending", True, True, "Loan/withdrawal email unanswered by admin ('didnt respond')"),
    ("post_0182", "reported_pending", True, True, "3rd list unanswered; 'They are not picking up the call'"),
    ("post_0185", "reported_pending", True, True, "Refund deadline undisclosed + 'no refund after registration' contradiction"),
    ("post_0299", "reported_pending", True, True, "NC-OBC WL-1 stuck + 'mailed them today they didn't respond'"),
    ("post_0319", "reported_pending", True, True, "Waitlist movement uncertain + 'Ye log phone bhi nhi utha te hai'"),
    ("post_0145", "reported_pending", True, False, "Merit list 2 release date still pending"),
    ("post_0147", "reported_pending", True, False, "Asking for updates on merit list 2 — no official resolution"),
    ("post_0024", "reported_pending", True, False, "Portal showing registration closed / pending access"),
    ("post_0138", "reported_pending", True, False, "OBC WL-20 waiting on admission decision"),
    ("post_0146", "reported_pending", True, False, "Portal link reactivated — awaiting outcome"),
    ("post_0112", "proposed_only", True, False, "Waitlist movement discussion — 'U got the offer?' is a question, NOT confirmed success!"),
    # Proposed only (12 threads — solutions/advice proposed, NO confirmed outcome!)
    ("post_0001", "proposed_only", True, False, "Advice to repeat CAT vs NMIMS HR — no outcome reported"),
    ("post_0002", "proposed_only", True, False, "College comparison advice (Nagpur/XLRI) — no decision reported"),
    ("post_0004", "proposed_only", True, False, "AICTE refund rules & TAPMI comparison proposed — no refund outcome reported"),
    ("post_0006", "proposed_only", True, False, "Waitlist benchmarks shared — outcome unresolved"),
    ("post_0010", "proposed_only", True, False, "CGPA conversion form advice proposed — no confirmation"),
    ("post_0014", "proposed_only", True, False, "Campus critique + local hospital clarification — unresolved"),
    ("post_0028", "proposed_only", True, False, "Advice to retake CAT given — no outcome reported"),
    ("post_0031", "proposed_only", True, False, "Advice on CAT retake + LinkedIn check — no decision reported"),
    ("post_0064", "proposed_only", True, True, "Contested placement debate with counterarguments calling claims 'fake news'"),
    ("post_0087", "proposed_only", True, True, "Contested alumni post with contradictory credibility replies"),
    ("post_0104", "proposed_only", True, True, "Age/MBA advice with realistic counter-experience ('wish grit alone was enough')"),
    ("post_0194", "proposed_only", True, False, "Hostel cooking workarounds (kettle) proposed — no follow-up confirmation"),
    # Insufficient evidence (8 threads — sparse/no evaluable solution or outcome)
    ("post_0003", "insufficient_evidence", True, False, "Single sarcastic comment on Sambalpur waitlist"),
    ("post_0005", "insufficient_evidence", True, False, "Only 'Profile?' and numeric score '94/93.2/9.6'"),
    ("post_0007", "insufficient_evidence", True, False, "Brief coaching opinions — no concrete solution or outcome"),
    ("post_0008", "insufficient_evidence", True, False, "Brief speculation about CAP opt-out forms — no confirmed resolution"),
    ("post_0009", "insufficient_evidence", True, False, "Single venting comment about CAP interviews"),
    ("post_0011", "insufficient_evidence", True, False, "Brief cutoff guesses — no actionable solution or outcome"),
    ("post_0012", "insufficient_evidence", True, False, "Short profile comparison banter — no solution/outcome"),
    ("post_0017", "insufficient_evidence", True, False, "One-word joke prompt — insufficient resolution evidence"),
]


def _prf(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    cm = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(y_true, y_pred):
        cm[g][p] += 1
    per_cls = {}
    f1s = []
    for lb in labels:
        tp = cm[lb][lb]
        fp = sum(cm[g][lb] for g in labels if g != lb)
        fn = sum(cm[lb][p] for p in labels if p != lb)
        sup = sum(cm[lb].values())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
        per_cls[lb] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "support": sup}
        if sup > 0:
            f1s.append(f1)
    acc = sum(1 for g, p in zip(y_true, y_pred) if g == p) / max(len(y_true), 1)
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4), "per_class": per_cls, "confusion_matrix": cm}


def main() -> None:
    print("[Phases 4 & 5] Loading real anonymized corpus & V6 predictions...")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))["tasks"]
    posts = corpus.get("posts", [])

    posts_by_id = {p["reddit_id"]: p for p in posts}
    comments_by_post: Dict[str, List[Dict[str, Any]]] = {}
    comment_lookup: Dict[str, Dict[str, Any]] = {}
    parent_text_by_cid: Dict[str, str] = {}

    for p in posts:
        pid = p["reddit_id"]
        cmts = p.get("comments", [])
        by_id_local = {c["reddit_id"]: c for c in cmts}
        enriched_cmts = []
        for c in cmts:
            cid = c["reddit_id"]
            dk = "comment-" + cid
            c_pred = {t: preds.get(t, {}).get(dk, {}) for t in ("issue", "sentiment", "emotion", "response", "solution")}
            p_cid = c.get("parent_reddit_id")
            p_txt = by_id_local[p_cid]["body"] if (p_cid and p_cid in by_id_local) else p["title"]
            parent_text_by_cid[cid] = p_txt
            c_enr = {**c, "post_id": pid, "post_title": p["title"], "predictions": c_pred}
            enriched_cmts.append(c_enr)
            comment_lookup[cid] = c_enr
        comments_by_post[pid] = enriched_cmts

    # -----------------------------------------------------------------------
    # PHASE 4A: Full-corpus contextual candidate generation & mapping
    # -----------------------------------------------------------------------
    print("[Phase 4] Generating contextual candidate pairs across all 8 issues...")
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
            c_pred = c["predictions"]
            rt = str(c_pred["response"].get("response_type") or "")
            sp = float(c_pred["solution"].get("solution_probability") or 0.0)
            is_sol = bool(c_pred["solution"].get("is_solution"))
            pred_issues = list(c_pred["issue"].get("predicted_issue_labels") or [])
            iss_probs = c_pred["issue"].get("issue_probabilities") or {}

            # Candidate if upstream response/solution model signals solution/advice OR moderately plausible candidate
            if not (is_sol or rt in ("solution", "advice") or sp >= 0.42):
                continue

            target_issues = set(pred_issues) | {post_issue}
            lid = detect_indian_reddit_language(body)
            for iss in target_issues:
                if iss not in ISSUE_DESCRIPTIONS:
                    continue
                key = (iss, pid, cid)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
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

    print(f"  Candidate issue-solution pairs: {len(candidates)}")
    mapper = ContextualIssueSolutionMapper(embedding_key="multilingual_e5_small", link_threshold=0.56)
    scored_all = mapper.score_candidates(candidates, use_embedding_cache=True)
    linked_all = [r for r in scored_all if r["linked"]]
    penalized_count = sum(1 for r in scored_all if r["signal_breakdown"]["negation_contradiction_penalty"] > 0)
    print(f"  Linked high-confidence pairs: {len(linked_all)} / {len(scored_all)} (penalized candidates: {penalized_count})")

    clusters_by_issue = mapper.cluster_solutions_by_issue(linked_all)
    for iss, clist in clusters_by_issue.items():
        n_l = sum(1 for r in linked_all if r["issue_id"] == iss)
        print(f"    {iss:30s}: {n_l:3d} linked solutions -> {len(clist):2d} semantic solution clusters")

    # -----------------------------------------------------------------------
    # PHASE 4B: Evaluate on 64 Manually Reviewed Real Pairs (V5 TF-IDF vs V6 Contextual)
    # -----------------------------------------------------------------------
    print("\n[Phase 4] Evaluating V6 Contextual Mapper vs. V5 TF-IDF baseline on 64 real annotated pairs...")
    eval_cands = []
    eval_csv_rows = []
    for iss, pid, cid, gold_rel, note in PHASE4_HUMAN_PAIRS:
        p = posts_by_id[pid]
        c = comment_lookup[cid]
        c_pred = c["predictions"]
        rt = str(c_pred["response"].get("response_type") or "")
        sp = float(c_pred["solution"].get("solution_probability") or 0.0)
        pred_issues = list(c_pred["issue"].get("predicted_issue_labels") or [])
        iss_probs = c_pred["issue"].get("issue_probabilities") or {}
        eval_cands.append({
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
        eval_csv_rows.append({
            "issue_id": iss,
            "post_id": pid,
            "solution_comment_id": cid,
            "post_title": p["title"],
            "parent_context": parent_text_by_cid.get(cid, p["title"])[:220],
            "comment_text": c["body"],
            "gold_relevant": gold_rel,
            "verification_status": "human_reviewed_v1",
            "human_note": note,
        })

    ISS_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ISS_EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eval_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(eval_csv_rows)

    eval_scored = mapper.score_candidates(eval_cands, use_embedding_cache=False)
    for rec, (_, _, _, gold_rel, note) in zip(eval_scored, PHASE4_HUMAN_PAIRS):
        rec["gold_relevant"] = gold_rel
        rec["human_note"] = note

    v6_metrics = evaluate_issue_solution_ranking(eval_scored, score_field="matching_score", threshold=0.56)
    v5_tfidf_metrics = evaluate_issue_solution_ranking(eval_scored, score_field="tfidf_baseline_score", threshold=0.35)

    print("  V5 TF-IDF Baseline :", json.dumps({k: v for k, v in v5_tfidf_metrics.items() if k != "confusion"}))
    print("  V6 Contextual E5   :", json.dumps({k: v for k, v in v6_metrics.items() if k != "confusion"}))

    # Strip numpy _embedding before serializing serving artifact
    def _clean_rec(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in r.items() if k != "_embedding"}

    top_links_by_issue: Dict[str, List[Dict[str, Any]]] = {}
    for iss in sorted(ISSUE_DESCRIPTIONS):
        iss_links = sorted(
            [_clean_rec(r) for r in linked_all if r["issue_id"] == iss],
            key=lambda x: x["matching_score"],
            reverse=True,
        )
        top_links_by_issue[iss] = iss_links[:25]

    serving_map = {
        "synthetic_only": False,
        "dataset_version": corpus["metadata"].get("dataset_version", "6.1.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "contextual_multilingual_issue_solution_mapper",
            "version": "v2",
            "embedding_model": "intfloat/multilingual-e5-small",
            "link_threshold": mapper.link_threshold,
            "weights": mapper.weights,
            "negation_contradiction_aware": True,
            "clustering_method": "AgglomerativeClustering (cosine distance, average linkage)",
        },
        "summary": {
            "total_candidates_evaluated": len(scored_all),
            "total_linked_solutions": len(linked_all),
            "penalized_candidates": penalized_count,
            "distinct_threads_with_links": len({r["post_id"] for r in linked_all}),
            "distinct_solution_clusters": sum(len(v) for v in clusters_by_issue.values()),
            "verification_status_policy": "All corpus links are explicitly labeled 'model_inferred'; human-reviewed evaluation pairs are stored separately in data/annotations/issue_solution_eval_v1.csv.",
        },
        "evaluation_benchmark_human_reviewed": {
            "dataset_path": str(ISS_EVAL_CSV.relative_to(ROOT)),
            "n_pairs": len(PHASE4_HUMAN_PAIRS),
            "v5_tfidf_baseline": v5_tfidf_metrics,
            "v6_contextual_embedding": v6_metrics,
        },
        "by_issue": {
            iss: {
                "issue_id": iss,
                "description": ISSUE_DESCRIPTIONS[iss],
                "linked_solution_count": sum(1 for r in linked_all if r["issue_id"] == iss),
                "distinct_threads": len({r["post_id"] for r in linked_all if r["issue_id"] == iss}),
                "cluster_count": len(clusters_by_issue.get(iss, [])),
                "semantic_solution_clusters": clusters_by_issue.get(iss, [])[:10],
                "top_matched_solutions": top_links_by_issue.get(iss, [])[:12],
            }
            for iss in sorted(ISSUE_DESCRIPTIONS)
        },
    }
    SERVING_MAP_V2.write_text(json.dumps(serving_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {SERVING_MAP_V2} ({SERVING_MAP_V2.stat().st_size // 1024} KB)")

    # -----------------------------------------------------------------------
    # PHASE 5A: Explainable Unresolved-Issue Detection across all 365 threads
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Running ExplainableResolutionEngine across all 365 threads...")
    res_engine = ExplainableResolutionEngine()
    res_full = res_engine.analyze_corpus(posts, comments_by_post, [_clean_rec(r) for r in linked_all])
    print("  Overall 365-thread summary:", json.dumps(res_full["overall_summary"]))
    for row in res_full["by_issue"]:
        print(
            f"    {row['issue_id']:28s} | {row['verdict']:18s} | threads={row['threads']:3d} "
            f"| succ={row['state_distribution']['reported_success']:2d} "
            f"| pend={row['state_distribution']['reported_pending']:2d} "
            f"| prop={row['state_distribution']['proposed_only']:3d} "
            f"| insuff={row['state_distribution']['insufficient_evidence']:2d} "
            f"| contr_threads={row['contradictory_threads']:2d}"
        )

    # -----------------------------------------------------------------------
    # PHASE 5B: Evaluate Phase 5 on 40 Manually Reviewed Threads
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Evaluating on 40 manually reviewed real threads...")
    thread_by_pid = {t["post_id"]: t for t in res_full["threads"]}
    y_st_true, y_st_pred = [], []
    y_unres_true, y_unres_pred = [], []
    false_res_claims = 0
    non_resolved_total = 0
    ev_retrieved_ok = 0
    ev_applicable = 0
    res_csv_rows = []
    res_errors = []

    for pid, g_state, g_unres, g_contr, note in PHASE5_HUMAN_THREADS:
        pred_t = thread_by_pid[pid]
        p_state = pred_t["resolution_state"]
        p_verdict = pred_t["verdict"]
        p_unres = p_verdict in ("Unresolved", "Insufficient Evidence", "Partially Resolved") and p_state != "reported_success"

        y_st_true.append(g_state)
        y_st_pred.append(p_state)
        y_unres_true.append(1 if g_unres else 0)
        y_unres_pred.append(1 if p_unres else 0)

        if g_unres:
            non_resolved_total += 1
            if p_state == "reported_success" or p_verdict == "Resolved":
                false_res_claims += 1

        # Evidence retrieval quality check: did engine retrieve concrete evidence matching the gold state?
        if g_state == "reported_success":
            ev_applicable += 1
            if pred_t["success_evidence"]:
                ev_retrieved_ok += 1
        elif g_state == "reported_pending":
            ev_applicable += 1
            if pred_t["pending_evidence"] or pred_t["contradictory_evidence"]:
                ev_retrieved_ok += 1
        elif g_state == "proposed_only":
            ev_applicable += 1
            if pred_t["proposed_solution"]:
                ev_retrieved_ok += 1

        if p_state != g_state:
            res_errors.append({
                "post_id": pid,
                "post_title": pred_t["post_title"][:90],
                "gold_state": g_state,
                "pred_state": p_state,
                "pred_verdict": p_verdict,
                "pred_score": pred_t["resolution_score"],
                "note": note,
            })

        res_csv_rows.append({
            "post_id": pid,
            "issue_id": pred_t["issue_id"],
            "post_title": pred_t["post_title"],
            "gold_resolution_state": g_state,
            "gold_is_unresolved": g_unres,
            "gold_has_contradiction": g_contr,
            "model_pred_state": p_state,
            "model_pred_verdict": p_verdict,
            "model_resolution_score": pred_t["resolution_score"],
            "verification_status": "human_reviewed_v1",
            "human_note": note,
        })

    with RES_EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(res_csv_rows)

    st_metrics = _prf(y_st_true, y_st_pred)
    tp_u = sum(1 for g, p in zip(y_unres_true, y_unres_pred) if g == 1 and p == 1)
    fp_u = sum(1 for g, p in zip(y_unres_true, y_unres_pred) if g == 0 and p == 1)
    fn_u = sum(1 for g, p in zip(y_unres_true, y_unres_pred) if g == 1 and p == 0)
    pr_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) else 0.0
    rc_u = tp_u / (tp_u + fn_u) if (tp_u + fn_u) else 0.0
    f1_u = (2 * pr_u * rc_u / (pr_u + rc_u)) if (pr_u + rc_u) else 0.0

    res_eval_report = {
        "dataset_path": str(RES_EVAL_CSV.relative_to(ROOT)),
        "n_threads": len(PHASE5_HUMAN_THREADS),
        "four_state_classification": st_metrics,
        "false_resolution_claim_count": false_res_claims,
        "false_resolution_claim_rate": round(false_res_claims / max(non_resolved_total, 1), 4),
        "unresolved_detection": {
            "precision": round(pr_u, 4),
            "recall": round(rc_u, 4),
            "f1": round(f1_u, 4),
            "tp": tp_u,
            "fp": fp_u,
            "fn": fn_u,
        },
        "evidence_retrieval_accuracy": round(ev_retrieved_ok / max(ev_applicable, 1), 4),
        "errors": res_errors,
    }
    print("  4-state accuracy:", st_metrics["accuracy"], "macro F1:", st_metrics["macro_f1"])
    print("  False resolution claim rate:", f"{res_eval_report['false_resolution_claim_rate']*100:.2f}% ({false_res_claims}/{non_resolved_total})")
    print("  Unresolved detection F1:", res_eval_report["unresolved_detection"]["f1"])
    print("  Evidence retrieval accuracy:", res_eval_report["evidence_retrieval_accuracy"])

    # Compact serving artifact for Resolution view (all 8 issues + top 60 richest thread chains)
    featured_threads = sorted(
        res_full["threads"],
        key=lambda t: (
            t["signals"]["contradictory_followups_count"] > 0,
            t["resolution_state"] == "reported_success",
            t["signals"]["solution_oriented_comments"],
            t["signals"]["discussion_volume"],
        ),
        reverse=True,
    )[:60]

    serving_res = {
        "synthetic_only": False,
        "dataset_version": corpus["metadata"].get("dataset_version", "6.1.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scoring_policy": res_full["scoring_policy"],
        "overall_summary": res_full["overall_summary"],
        "evaluation_benchmark_human_reviewed": res_eval_report,
        "by_issue": res_full["by_issue"],
        "featured_thread_trails": featured_threads,
    }
    SERVING_RES_V2.write_text(json.dumps(serving_res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {SERVING_RES_V2} ({SERVING_RES_V2.stat().st_size // 1024} KB)")

    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_OUT_DIR / "phase4_issue_solution_eval.json").write_text(
        json.dumps(
            {
                "summary": serving_map["summary"],
                "v5_tfidf_baseline": v5_tfidf_metrics,
                "v6_contextual_embedding": v6_metrics,
                "scored_eval_pairs": [_clean_rec(r) for r in eval_scored],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (EVAL_OUT_DIR / "phase5_resolution_eval.json").write_text(
        json.dumps(
            {
                "overall_summary": res_full["overall_summary"],
                "by_issue": [{k: v for k, v in row.items() if not k.startswith("sample_")} for row in res_full["by_issue"]],
                "human_eval": res_eval_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("[Phases 4 & 5] DONE!")


if __name__ == "__main__":
    main()
