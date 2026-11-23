"""Phase 5 — Explainable Unresolved Issue Detection Engine.

Formalizes and preserves the four distinct thread-level resolution states:
  - ``reported_success``
  - ``reported_pending``
  - ``proposed_only``
  - ``insufficient_evidence``

And defines transparent, explainable issue-level & thread-level verdict scoring
using eight explicit signals (never equating a proposed solution or discussion
volume with confirmed resolution!):
  1. Discussion volume (context/evidence weight ONLY — never proof of resolution)
  2. Solution-oriented comment count & density
  3. Linked solution relevance (from Phase 4 Contextual Issue–Solution Mapper)
  4. Explicit reported-success signals
  5. Follow-up confirmation signals
  6. Time elapsed between post creation and latest comment / observation date
  7. Evidence strength (via ``EvidenceBasedResolver`` volume guard + support)
  8. Contradictory follow-up signals (e.g., *"tried that, didn't work"*,
     *"still no response"*, *"they refused"*, direct counterarguments to solutions)

Every assessment preserves a full traceable evidence chain:
  original issue -> related discussions -> proposed solution -> reported outcome
  -> follow-up evidence -> contradictory evidence -> timestamp -> confidence -> explanation
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

RESOLUTION_STATES: Tuple[str, ...] = (
    "reported_success",
    "reported_pending",
    "proposed_only",
    "insufficient_evidence",
)

VERDICT_TIERS: Tuple[str, ...] = (
    "Resolved",
    "Partially Resolved",
    "Unresolved",
    "Insufficient Evidence",
)

# Explicit outcome-confirmation patterns (strict: generic "thanks", advisor credentials like
# "I converted IIM L last year", or interrogative questions NEVER prove thread resolution!)
_REPORTED_SUCCESS_RE = re.compile(
    r"\b(it worked|that worked|this worked|worked for me|problem solved|issue (?:is )?resolved|"
    r"(?:i|we|finally|now)\s+(?:also\s+|actually\s+|just\s+)?got\s+(?:my\s+|the\s+)?(?:refund|offer|offer letter|admission|selected)|"
    r"finally\s+got\s+(?:selected|in|the\s+offer|admitted)|"
    r"refund (?:received|processed|came)|finally converted|converted (?:today|the waitlist|my waitlist|wl)|"
    r"admission confirmed|loan (?:approved|sanctioned)|resolved now|sorted now|"
    r"cleared the (?:waitlist|wl)|helped me (?:get|convert|resolve|clear)|that actually helped|"
    r"finally ho gya|finally ho gaya|update:\s*(?:got|converted|resolved|worked))\b",
    re.I,
)
_QUESTION_GUARD_RE = re.compile(r"\?\s*$|\b(did you|u got|anyone got|have you got|will i get|is it)\b", re.I)

# Explicit pending / unresolved status follow-ups (includes OP pending title cues)
_REPORTED_PENDING_RE = re.compile(
    r"\b(still waiting|still pending|no update yet|no response yet|haven'?t received|hasn'?t moved|"
    r"not yet (?:released|updated|received|declared|refunded)|waiting for (?:mail|email|offer|result|refund|list|reply)|"
    r"any updates?(?:\s+on)?|still stuck|portal (?:not|hasn'?t) updated|still showing|not showing yet|"
    r"offer has been rejected|registration closed|merit list \d*\s*when|list \d*\s*when|waitlist.*\bwhen\b|"
    r"abhi tak nhi|abhi tak nahi|koi update nahi|mail nahi aaya)\b",
    re.I,
)

# Explicit procedural/action failure follow-ups (solution tried or contact attempted -> failed/no response)
_PROCEDURAL_FAILURE_RE = re.compile(
    r"\b(tried (?:that|this|calling|mailing|emailing).{0,40}(?:didn'?t|did not|no reply|no response|nothing|failed|refused)|"
    r"(?:didn'?t|did not|doesn'?t|does not)\s+(?:work|help|reply|respond|pick up|refund)|"
    r"phone bhi nhi utha|they (?:refused|denied|rejected|ignored)|not picking up|"
    r"got rejected anyway|still (?:no reply|no response|not working|didn'?t work|rejected)|"
    r"deducted anyway|no refund)\b",
    re.I,
)

# Rhetorical / credibility pushback (records contradiction penalty without changing proposed_only -> pending)
_RHETORICAL_DISPUTE_RE = re.compile(
    r"\b(that'?s (?:not true|wrong|false|incorrect|outdated)|wrong info|misleading|fake news)\b",
    re.I,
)
_CONTRADICTORY_FOLLOWUP_RE = re.compile(
    f"(?:{_PROCEDURAL_FAILURE_RE.pattern})|(?:{_RHETORICAL_DISPUTE_RE.pattern})",
    re.I,
)

# Follow-up agreement/confirmation specifically validating a prior proposed solution
_FOLLOWUP_CONFIRM_RE = re.compile(
    r"(?:\b(can confirm|this is accurate|second this|agreed,? (this|that)|true,? (follow|check|try)|"
    r"same worked|did the same|yes,? (aicte|they have to|you can)|exactly what i did)\b|(?<!\S)\+1\b)",
    re.I,
)


def _parse_iso(ts: str) -> Optional[datetime]:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def assess_thread_resolution(
    post: Dict[str, Any],
    comments: Sequence[Dict[str, Any]],
    linked_solutions: Sequence[Dict[str, Any]],
    *,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Produce a fully explainable, evidence-tracked resolution assessment for one thread."""
    post_id = str(post.get("reddit_id") or "")
    post_title = str(post.get("title") or "")
    issue_id = str(post.get("issue_key") or post.get("issue_category") or "administrative_communication")
    post_dt = _parse_iso(str(post.get("created_at") or ""))
    ref_dt = reference_time or datetime(2026, 4, 15, tzinfo=timezone.utc)

    cmts = list(comments or [])
    by_cid = {str(c.get("reddit_id") or c.get("comment_id") or ""): c for c in cmts}
    sol_links = [s for s in linked_solutions if s.get("post_id") == post_id and s.get("linked")]
    sol_link_ids = {s["solution_comment_id"] for s in sol_links}

    # Chronological timestamps & elapsed time
    cmt_dts = [_parse_iso(str(c.get("created_at") or "")) for c in cmts]
    valid_dts = [d for d in cmt_dts if d is not None]
    latest_dt = max(valid_dts) if valid_dts else post_dt
    discussion_span_hours = (
        round((latest_dt - post_dt).total_seconds() / 3600.0, 2)
        if (post_dt and latest_dt and latest_dt >= post_dt)
        else 0.0
    )
    days_since_post = (
        round(max(0.0, (ref_dt - post_dt).total_seconds() / 86400.0), 1)
        if post_dt
        else 0.0
    )

    success_evidence: List[Dict[str, Any]] = []
    pending_evidence: List[Dict[str, Any]] = []
    contradictory_evidence: List[Dict[str, Any]] = []
    explicit_failure_evidence: List[Dict[str, Any]] = []
    followup_confirmations: List[Dict[str, Any]] = []

    # Check OP post title + body for explicit OP-reported outcome or pending status
    op_text = f"{post_title} {str(post.get('body') or '')[:300]}".strip()
    if _REPORTED_SUCCESS_RE.search(post_title) and not _QUESTION_GUARD_RE.search(post_title):
        success_evidence.append({
            "comment_id": post_id,
            "text": post_title[:280],
            "timestamp": str(post.get("created_at") or ""),
            "signal": "op_title_reported_outcome_success",
        })
    if _REPORTED_PENDING_RE.search(op_text):
        pending_evidence.append({
            "comment_id": post_id,
            "text": post_title[:280],
            "timestamp": str(post.get("created_at") or ""),
            "signal": "op_reported_pending_status",
        })

    for c in cmts:
        cid = str(c.get("reddit_id") or c.get("comment_id") or "")
        body = str(c.get("body") or c.get("text") or "").strip()
        parent_id = str(c.get("parent_reddit_id") or "")
        pred_resp = str(((c.get("predictions") or {}).get("response") or {}).get("response_type") or c.get("response_type") or "")
        ts = str(c.get("created_at") or "")

        is_procedural_fail = bool(_PROCEDURAL_FAILURE_RE.search(body))
        is_rhetorical_dispute = bool(_RHETORICAL_DISPUTE_RE.search(body))
        is_counter_to_sol = bool(parent_id in sol_link_ids and pred_resp == "counterargument")
        if is_procedural_fail or is_rhetorical_dispute or is_counter_to_sol:
            sig = (
                "procedural_failure_or_no_response"
                if is_procedural_fail
                else ("rhetorical_contradiction_dispute" if is_rhetorical_dispute else "contradictory_reply_to_solution")
            )
            item = {
                "comment_id": cid,
                "parent_comment_id": parent_id or None,
                "text": body[:280],
                "timestamp": ts,
                "signal": sig,
            }
            contradictory_evidence.append(item)
            if is_procedural_fail:
                explicit_failure_evidence.append(item)
        if _REPORTED_SUCCESS_RE.search(body) and not _QUESTION_GUARD_RE.search(body):
            success_evidence.append({
                "comment_id": cid,
                "text": body[:280],
                "timestamp": ts,
                "signal": "explicit_reported_outcome_success",
            })
        if _REPORTED_PENDING_RE.search(body):
            pending_evidence.append({
                "comment_id": cid,
                "text": body[:280],
                "timestamp": ts,
                "signal": "explicit_pending_or_no_response_status",
            })
        if parent_id in sol_link_ids and (_FOLLOWUP_CONFIRM_RE.search(body) or pred_resp == "agreement"):
            followup_confirmations.append({
                "comment_id": cid,
                "parent_comment_id": parent_id,
                "text": body[:240],
                "timestamp": ts,
                "signal": "reply_confirming_proposed_solution",
            })

    # Determine canonical 4-state resolution_state (preserving exact distinction!)
    has_solutions = len(sol_links) > 0
    has_success = len(success_evidence) > 0
    has_pending = len(pending_evidence) > 0
    has_contradiction = len(contradictory_evidence) > 0
    has_explicit_failure = len(explicit_failure_evidence) > 0

    if has_success and not has_explicit_failure:
        state = "reported_success"
    elif has_pending or has_explicit_failure:
        state = "reported_pending"
    elif has_solutions:
        state = "proposed_only"
    else:
        state = "insufficient_evidence"

    # Compute transparent 8-signal explainable score in [0, 1]
    # Note: discussion_volume NEVER increases the score directly — it only gates confidence!
    n_comments = len(cmts)
    sol_density = min(1.0, len(sol_links) / max(n_comments, 1)) if n_comments else 0.0
    mean_sol_rel = (
        round(sum(s["matching_score"] for s in sol_links) / len(sol_links), 4)
        if sol_links
        else 0.0
    )
    best_sol_rel = max((s["matching_score"] for s in sol_links), default=0.0)
    success_sig = 1.0 if has_success else 0.0
    confirm_sig = min(1.0, len(followup_confirmations) / 2.0)
    contradiction_pen = min(0.45, 0.22 * len(contradictory_evidence) + (0.12 if has_pending else 0.0))

    # Explainable resolution score formula:
    # - Without reported_success, max attainable score is capped at 0.58 (so proposed_only NEVER reaches Resolved!)
    raw_score = (
        0.42 * success_sig
        + 0.24 * best_sol_rel
        + 0.14 * min(1.0, sol_density * 2.5)
        + 0.12 * confirm_sig
        + 0.08 * mean_sol_rel
        - contradiction_pen
    )
    if not has_success:
        raw_score = min(raw_score, 0.58)
    resolution_score = round(max(0.0, min(1.0, raw_score)), 4)

    # Map to the 4 transparent verdict tiers (strictly requiring outcome evidence for Resolved!)
    if n_comments < 2 and not has_solutions and not has_success and not has_pending:
        verdict = "Insufficient Evidence"
    elif state == "reported_success" and resolution_score >= 0.62 and not has_contradiction:
        verdict = "Resolved"
    elif (state == "reported_success" and has_contradiction) or (
        has_solutions and (len(followup_confirmations) >= 1 or (best_sol_rel >= 0.72 and not has_pending and not has_contradiction))
    ):
        verdict = "Partially Resolved"
    elif state in ("proposed_only", "reported_pending") or has_contradiction:
        verdict = "Unresolved"
    else:
        verdict = "Insufficient Evidence"

    # Evidence strength / confidence (volume guard prevents overclaiming on tiny threads)
    volume_guard = min(1.0, n_comments / 6.0)
    evidence_items = len(sol_links) + len(success_evidence) + len(pending_evidence) + len(contradictory_evidence)
    confidence = round(max(0.15, min(0.96, 0.30 + 0.40 * volume_guard + 0.30 * min(1.0, evidence_items / 4.0))), 4)

    # Human-readable explanation
    exp_parts = [
        f"State='{state}' -> Verdict='{verdict}' (score={resolution_score:.2f}, confidence={confidence:.2f}).",
        f"Thread has {n_comments} comments over {discussion_span_hours:.1f}h ({days_since_post:.0f}d elapsed) and {len(sol_links)} contextually matched solution(s) (best relevance={best_sol_rel:.2f}).",
    ]
    if has_success:
        exp_parts.append(f"Explicit outcome confirmation found in {len(success_evidence)} comment(s).")
    else:
        exp_parts.append("No explicit reported-success confirmation found; proposed solutions are NOT assumed to have succeeded.")
    if has_contradiction:
        exp_parts.append(f"Contradictory/failed follow-up evidence detected in {len(contradictory_evidence)} comment(s) (penalty -{contradiction_pen:.2f}).")
    if has_pending:
        exp_parts.append(f"Pending/no-response status reported in {len(pending_evidence)} comment(s).")

    top_sol = max(sol_links, key=lambda s: s["matching_score"]) if sol_links else None
    reported_outcome_item = (
        success_evidence[0]
        if success_evidence
        else (contradictory_evidence[0] if contradictory_evidence else (pending_evidence[0] if pending_evidence else None))
    )

    return {
        "post_id": post_id,
        "post_title": post_title,
        "issue_id": issue_id,
        "subreddit": post.get("subreddit", ""),
        "resolution_state": state,
        "verdict": verdict,
        "resolution_score": resolution_score,
        "confidence": confidence,
        "timestamp": str(post.get("created_at") or ""),
        "latest_comment_timestamp": latest_dt.isoformat() if latest_dt else "",
        "signals": {
            "discussion_volume": n_comments,
            "solution_oriented_comments": len(sol_links),
            "solution_density": round(sol_density, 4),
            "mean_solution_relevance": mean_sol_rel,
            "best_solution_relevance": round(best_sol_rel, 4),
            "reported_success_count": len(success_evidence),
            "followup_confirmations_count": len(followup_confirmations),
            "pending_signals_count": len(pending_evidence),
            "contradictory_followups_count": len(contradictory_evidence),
            "discussion_span_hours": discussion_span_hours,
            "days_since_post": days_since_post,
            "evidence_strength": round(volume_guard, 4),
            "contradiction_penalty": round(contradiction_pen, 4),
        },
        "proposed_solution": (
            {
                "comment_id": top_sol["solution_comment_id"],
                "matching_score": top_sol["matching_score"],
                "text": top_sol["evidence_text"][:320],
                "matching_rationale": top_sol["matching_rationale"],
            }
            if top_sol
            else None
        ),
        "reported_outcome": reported_outcome_item,
        "follow_up_evidence": followup_confirmations[:3],
        "contradictory_evidence": contradictory_evidence[:3],
        "pending_evidence": pending_evidence[:3],
        "success_evidence": success_evidence[:3],
        "explanation": " ".join(exp_parts),
        "verification_status": "model_inferred",
    }


class ExplainableResolutionEngine:
    """Thread-level and Issue-domain-level explainable unresolved-issue analyzer."""

    def analyze_corpus(
        self,
        posts: Sequence[Dict[str, Any]],
        comments_by_post: Dict[str, List[Dict[str, Any]]],
        linked_solutions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        thread_assessments: List[Dict[str, Any]] = []
        for p in posts:
            pid = str(p.get("reddit_id") or "")
            ass = assess_thread_resolution(p, comments_by_post.get(pid, []), linked_solutions)
            thread_assessments.append(ass)

        # Aggregate strictly by issue domain
        by_issue_threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ta in thread_assessments:
            by_issue_threads[ta["issue_id"]].append(ta)

        issue_summaries: List[Dict[str, Any]] = []
        for iss, tlist in sorted(by_issue_threads.items()):
            n_t = max(len(tlist), 1)
            state_counts = Counter(t["resolution_state"] for t in tlist)
            verdict_counts = Counter(t["verdict"] for t in tlist)
            contrad_threads = sum(1 for t in tlist if t["signals"]["contradictory_followups_count"] > 0)
            total_sol_links = sum(t["signals"]["solution_oriented_comments"] for t in tlist)
            total_comments = sum(t["signals"]["discussion_volume"] for t in tlist)
            mean_score = round(sum(t["resolution_score"] for t in tlist) / n_t, 4)
            mean_conf = round(sum(t["confidence"] for t in tlist) / n_t, 4)

            confirmed_rate = state_counts["reported_success"] / n_t
            unresolved_rate = (state_counts["reported_pending"] + state_counts["proposed_only"] + state_counts["insufficient_evidence"]) / n_t
            contradiction_rate = contrad_threads / n_t

            # Transparent issue-level status verdict
            if n_t < 3 or (state_counts["insufficient_evidence"] / n_t) >= 0.70:
                issue_verdict = "Insufficient Evidence"
            elif confirmed_rate >= 0.35 and contradiction_rate < 0.25:
                issue_verdict = "Resolved"
            elif confirmed_rate >= 0.10 or (verdict_counts["Partially Resolved"] / n_t) >= 0.25:
                issue_verdict = "Partially Resolved"
            else:
                issue_verdict = "Unresolved"

            # Representative evidence trails (one resolved/partial if any, one unresolved/contradictory)
            top_unresolved = sorted(
                [t for t in tlist if t["verdict"] == "Unresolved"],
                key=lambda x: (x["signals"]["contradictory_followups_count"], x["signals"]["discussion_volume"]),
                reverse=True,
            )[:3]
            top_resolved = sorted(
                [t for t in tlist if t["resolution_state"] == "reported_success"],
                key=lambda x: x["resolution_score"],
                reverse=True,
            )[:2]

            issue_summaries.append({
                "issue_id": iss,
                "verdict": issue_verdict,
                "threads": len(tlist),
                "total_comments": total_comments,
                "matched_solutions": total_sol_links,
                "mean_resolution_score": mean_score,
                "mean_confidence": mean_conf,
                "confirmed_success_rate": round(confirmed_rate, 4),
                "unresolved_rate": round(unresolved_rate, 4),
                "contradiction_rate": round(contradiction_rate, 4),
                "contradictory_threads": contrad_threads,
                "state_distribution": {st: state_counts.get(st, 0) for st in RESOLUTION_STATES},
                "verdict_distribution": {vt: verdict_counts.get(vt, 0) for vt in VERDICT_TIERS},
                "explanation": (
                    f"{iss}: {issue_verdict} across {len(tlist)} threads — "
                    f"{state_counts['reported_success']} reported_success ({confirmed_rate*100:.1f}%), "
                    f"{state_counts['proposed_only']} proposed_only, "
                    f"{state_counts['reported_pending']} reported_pending, "
                    f"{state_counts['insufficient_evidence']} insufficient_evidence, "
                    f"with contradictory/failed follow-ups in {contrad_threads} thread(s)."
                ),
                "sample_unresolved_trails": top_unresolved,
                "sample_confirmed_trails": top_resolved,
            })

        issue_summaries.sort(key=lambda x: (x["unresolved_rate"], x["threads"]), reverse=True)

        overall_states = Counter(t["resolution_state"] for t in thread_assessments)
        overall_verdicts = Counter(t["verdict"] for t in thread_assessments)
        return {
            "scoring_policy": {
                "states": list(RESOLUTION_STATES),
                "verdict_tiers": list(VERDICT_TIERS),
                "guardrails": [
                    "Proposed solutions NEVER earn 'Resolved' without explicit reported_success confirmation (score capped at 0.58).",
                    "Discussion volume NEVER increases resolution_score; it only scales the confidence volume guard.",
                    "Contradictory follow-ups and counterarguments directly penalize resolution_score (-0.22 per signal, max -0.45).",
                ],
            },
            "overall_summary": {
                "total_threads": len(thread_assessments),
                "state_distribution": {st: overall_states.get(st, 0) for st in RESOLUTION_STATES},
                "verdict_distribution": {vt: overall_verdicts.get(vt, 0) for vt in VERDICT_TIERS},
                "threads_with_contradictory_followups": sum(
                    1 for t in thread_assessments if t["signals"]["contradictory_followups_count"] > 0
                ),
                "threads_with_reported_success": overall_states.get("reported_success", 0),
                "threads_proposed_only": overall_states.get("proposed_only", 0),
            },
            "by_issue": issue_summaries,
            "threads": thread_assessments,
        }
