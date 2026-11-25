"""Phase 5 — Explainable Unresolved Issue Detection Engine (V6.3 Chronological & Precedence Update).

Formalizes and preserves the four distinct thread-level resolution states:
  - ``reported_success``
  - ``reported_pending``
  - ``proposed_only``
  - ``insufficient_evidence``

Explicitly separates the five resolution-discourse stages (never equating agreement or
a proposed solution with attempted action or confirmed outcome!):
  1. Proposed recommendation (linked solution / advice)
  2. Agreement or endorsement (``agreement`` response, ``+1``, generic endorsement)
  3. Attempted action (e.g., *"I mailed admissions today"*, *"Tried calling them"*)
  4. Reported outcome / status (pending wait, procedural failure, contradictory dispute)
  5. Explicit confirmation of success (first-person resolution or confirmed recovery)

Uses dynamic, reproducible reference timestamps (explicit ``reference_time`` -> dataset
``observation_timestamp`` -> latest thread/corpus activity timestamp; never a hard-coded
historical date and never implicit ``datetime.now()``) and deterministic chronological
precedence rules when thread evidence conflicts.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

PRECEDENCE_RULES: Tuple[str, ...] = (
    "later_failure_or_pending_overrides_prior_success",
    "chronological_recovery_after_initial_failure",
    "conflicting_outcomes_capped_at_partial",
    "confirmed_success_supersedes_initial_pending",
    "clean_reported_success",
    "reported_pending_or_attempt_or_failure_unresolved",
    "proposed_solution_only_unresolved",
    "insufficient_evidence_default",
)

# Explicit outcome-confirmation patterns (strict: generic "thanks", generic positive
# sentiment, or interrogative questions NEVER prove thread resolution!)
_REPORTED_SUCCESS_RE = re.compile(
    r"\b(it worked|that worked|this worked|worked for me|problem solved|issue (?:is )?resolved|"
    r"(?:i|we|finally|now)\s+(?:also\s+|actually\s+|just\s+)?got\s+(?:my\s+|the\s+)?(?:refund|offer|offer letter|admission|selected)|"
    r"finally\s+got\s+(?:selected|in|the\s+offer|admitted)|"
    r"refund (?:received|processed|came)|finally converted|converted (?:today|the waitlist|my waitlist|wl)|"
    r"admission confirmed|loan (?:approved|sanctioned)|resolved now|sorted now|"
    r"cleared the (?:waitlist|wl)|helped me (?:get|convert|resolve|clear)|that actually helped|"
    r"finally ho gya|finally ho gaya|update:\s*(?:got|converted|resolved|worked|issue resolved))\b",
    re.I,
)
_NEGATED_SUCCESS_RE = re.compile(
    r"\b(?:not|never|haven'?t|hasn'?t|didn'?t|did not|no|still no)\s+"
    r"(?:yet\s+)?(?:actually\s+|really\s+)?(?:worked|resolved|solved|received|converted|approved|sanctioned|got)\b",
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

# Explicit procedural/action failure follow-ups (solution tried or contact attempted -> failed/persists)
_PROCEDURAL_FAILURE_RE = re.compile(
    r"\b(tried (?:that|this|calling|mailing|emailing).{0,45}(?:didn'?t|did not|no reply|no response|nothing|failed|refused|still|persists|not)|"
    r"(?:didn'?t|did not|doesn'?t|does not)\s+(?:work|help|reply|respond|pick up|refund)|"
    r"phone bhi nhi utha|they (?:refused|denied|rejected|ignored)|not picking up|"
    r"got rejected anyway|still (?:no reply|no response|not working|didn'?t work|rejected|unresolved|persists|broken|broken\b|not fixed|pending)|"
    r"issue (?:still )?persists|problem persists|still broken|failed again|"
    r"deducted anyway|no refund)\b",
    re.I,
)

# Rhetorical / credibility pushback (records contradiction penalty without conflating disagreement with outcome)
_RHETORICAL_DISPUTE_RE = re.compile(
    r"\b(that'?s (?:not true|wrong|false|incorrect|outdated)|wrong info|misleading|fake news)\b",
    re.I,
)

# Explicit attempted action (actor tried a step, outcome not yet confirmed)
_ATTEMPTED_ACTION_RE = re.compile(
    r"\b(?:i|we)\s+(?:just\s+|already\s+|today\s+)?(?:tried|mailed|emailed|called|applied|submitted|filed|"
    r"contacted|wrote to|logged a ticket|raised a ticket|followed (?:this|that))\b",
    re.I,
)

# Concrete peer outcome confirmation (someone followed a proposed solution AND confirms outcome worked)
_PEER_OUTCOME_CONFIRM_RE = re.compile(
    r"\b(same worked|did the same and (?:got|it worked|resolved)|worked for our batch|"
    r"followed (?:this|that|it) and (?:got|it worked|resolved)|exactly what i did and it worked)\b",
    re.I,
)

# Agreement / endorsement WITHOUT outcome execution confirmation (kept strictly separate!)
_ENDORSEMENT_ONLY_RE = re.compile(
    r"(?:\b(can confirm|this is accurate|second this|agreed,? (?:this|that)|true,? (?:follow|check|try)|"
    r"totally agree|same bro|same here)\b|(?<!\S)\+1\b)",
    re.I,
)

_RECOVERY_CUE_RE = re.compile(
    r"\b(update:|finally|after escalating|worked after|eventually|resolved now)\b",
    re.I,
)


def _parse_iso(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def resolve_reference_time(
    reference_time: Optional[Union[datetime, str]],
    post: Dict[str, Any],
    comments: Sequence[Dict[str, Any]],
    *,
    observation_timestamp: Optional[Union[datetime, str]] = None,
) -> Tuple[Optional[datetime], str]:
    """Deterministically resolve the observation reference timestamp (never hardcoded, never implicit wall clock).

    Resolution order:
      1. Explicit ``reference_time`` argument.
      2. Explicit ``observation_timestamp`` argument or ``post["observation_timestamp"]``.
      3. Latest valid timestamp observed across ``post["created_at"]`` and all ``comments[*]["created_at"]``.
      4. ``(None, "unavailable")`` when no valid timestamps exist.
    """
    explicit_dt = _parse_iso(reference_time)
    if explicit_dt is not None:
        return explicit_dt, "explicit_reference_time"

    obs_raw = observation_timestamp if observation_timestamp is not None else post.get("observation_timestamp")
    obs_dt = _parse_iso(obs_raw)
    if obs_dt is not None:
        return obs_dt, "dataset_observation_timestamp"

    candidates: List[datetime] = []
    post_dt = _parse_iso(post.get("created_at"))
    if post_dt is not None:
        candidates.append(post_dt)
    for c in comments or []:
        c_dt = _parse_iso(c.get("created_at"))
        if c_dt is not None:
            candidates.append(c_dt)
    if candidates:
        return max(candidates), "thread_latest_activity_timestamp"

    return None, "unavailable"


def _actor_key(post: Dict[str, Any], comment: Optional[Dict[str, Any]], text: str) -> str:
    if comment is None:
        return "op"
    if bool(comment.get("is_submitter")) or (
        post.get("author_hash") and comment.get("author_hash") == post.get("author_hash")
    ):
        return "op"
    if re.search(r"^\s*(?:update:|edit:|op here\b)", text, re.I):
        return "op"
    cid = str(comment.get("reddit_id") or comment.get("comment_id") or "")
    return f"commenter:{cid}"


def assess_thread_resolution(
    post: Dict[str, Any],
    comments: Sequence[Dict[str, Any]],
    linked_solutions: Sequence[Dict[str, Any]],
    *,
    reference_time: Optional[Union[datetime, str]] = None,
    observation_timestamp: Optional[Union[datetime, str]] = None,
) -> Dict[str, Any]:
    """Produce a chronological, explainable, 5-stage resolution assessment for one thread."""
    post_id = str(post.get("reddit_id") or "")
    post_title = str(post.get("title") or "")
    issue_id = str(post.get("issue_key") or post.get("issue_category") or "administrative_communication")
    post_dt = _parse_iso(post.get("created_at"))

    cmts = list(comments or [])
    ref_dt, ref_source = resolve_reference_time(
        reference_time,
        post,
        cmts,
        observation_timestamp=observation_timestamp,
    )

    sol_links = [s for s in linked_solutions if s.get("post_id") == post_id and s.get("linked")]
    sol_link_ids = {s["solution_comment_id"] for s in sol_links}

    # Chronological ordering & elapsed time
    indexed_comments: List[Tuple[int, Optional[datetime], Dict[str, Any]]] = []
    for idx, c in enumerate(cmts):
        indexed_comments.append((idx, _parse_iso(c.get("created_at")), c))
    indexed_comments.sort(
        key=lambda item: (
            item[1] if item[1] is not None else datetime.max.replace(tzinfo=timezone.utc),
            item[0],
        )
    )

    valid_dts = [d for _, d, _ in indexed_comments if d is not None]
    latest_dt = max(valid_dts) if valid_dts else post_dt
    discussion_span_hours = (
        round((latest_dt - post_dt).total_seconds() / 3600.0, 2)
        if (post_dt and latest_dt and latest_dt >= post_dt)
        else 0.0
    )
    days_since_post = (
        round(max(0.0, (ref_dt - post_dt).total_seconds() / 86400.0), 1)
        if (post_dt and ref_dt)
        else 0.0
    )

    success_evidence: List[Dict[str, Any]] = []
    pending_evidence: List[Dict[str, Any]] = []
    attempt_evidence: List[Dict[str, Any]] = []
    endorsement_evidence: List[Dict[str, Any]] = []
    contradictory_evidence: List[Dict[str, Any]] = []
    explicit_failure_evidence: List[Dict[str, Any]] = []
    rhetorical_dispute_evidence: List[Dict[str, Any]] = []

    # Timeline tuples: (sort_dt, seq_rank, event_kind, item_dict)
    timeline: List[Tuple[datetime, int, str, Dict[str, Any]]] = []
    epoch_floor = datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Check OP post title + body for explicit OP-reported outcome or pending status
    op_text = f"{post_title} {str(post.get('body') or '')[:300]}".strip()
    op_dt_key = post_dt or epoch_floor
    if (
        _REPORTED_SUCCESS_RE.search(post_title)
        and not _QUESTION_GUARD_RE.search(post_title)
        and not _NEGATED_SUCCESS_RE.search(post_title)
    ):
        ev = {
            "comment_id": post_id,
            "actor": "op",
            "text": post_title[:280],
            "timestamp": str(post.get("created_at") or ""),
            "signal": "op_title_reported_outcome_success",
            "seq": 0,
        }
        success_evidence.append(ev)
        timeline.append((op_dt_key, 0, "success", ev))
    if _REPORTED_PENDING_RE.search(op_text):
        ev = {
            "comment_id": post_id,
            "actor": "op",
            "text": post_title[:280],
            "timestamp": str(post.get("created_at") or ""),
            "signal": "op_reported_pending_status",
            "seq": 0,
        }
        pending_evidence.append(ev)
        timeline.append((op_dt_key, 0, "pending", ev))

    for order_pos, (orig_idx, c_dt, c) in enumerate(indexed_comments, start=1):
        cid = str(c.get("reddit_id") or c.get("comment_id") or "")
        body = str(c.get("body") or c.get("text") or "").strip()
        parent_id = str(c.get("parent_reddit_id") or c.get("parent_comment_id") or "")
        pred_resp = str(
            ((c.get("predictions") or {}).get("response") or {}).get("response_type")
            or c.get("response_type")
            or ""
        )
        ts = str(c.get("created_at") or "")
        actor = _actor_key(post, c, body)
        dt_key = c_dt or post_dt or epoch_floor

        has_negated_success = bool(_NEGATED_SUCCESS_RE.search(body))
        is_procedural_fail = bool(_PROCEDURAL_FAILURE_RE.search(body)) or has_negated_success
        is_rhetorical_dispute = bool(_RHETORICAL_DISPUTE_RE.search(body))
        is_counter_to_sol = bool(parent_id in sol_link_ids and pred_resp == "counterargument")

        if is_procedural_fail or is_rhetorical_dispute or is_counter_to_sol:
            sig = (
                "procedural_failure_or_no_response"
                if is_procedural_fail
                else (
                    "rhetorical_contradiction_dispute"
                    if is_rhetorical_dispute
                    else "contradictory_reply_to_solution"
                )
            )
            item = {
                "comment_id": cid,
                "parent_comment_id": parent_id or None,
                "actor": actor,
                "text": body[:280],
                "timestamp": ts,
                "signal": sig,
                "seq": order_pos,
            }
            contradictory_evidence.append(item)
            if is_procedural_fail:
                explicit_failure_evidence.append(item)
                timeline.append((dt_key, order_pos, "failure", item))
            else:
                rhetorical_dispute_evidence.append(item)

        has_explicit_success = bool(
            (
                _REPORTED_SUCCESS_RE.search(body)
                or (parent_id in sol_link_ids and _PEER_OUTCOME_CONFIRM_RE.search(body))
            )
            and not _QUESTION_GUARD_RE.search(body)
            and not has_negated_success
            and not is_procedural_fail
        )
        if has_explicit_success:
            ev = {
                "comment_id": cid,
                "parent_comment_id": parent_id or None,
                "actor": actor,
                "text": body[:280],
                "timestamp": ts,
                "signal": "explicit_reported_outcome_success",
                "seq": order_pos,
            }
            success_evidence.append(ev)
            timeline.append((dt_key, order_pos, "success", ev))

        if _REPORTED_PENDING_RE.search(body) and not has_explicit_success:
            ev = {
                "comment_id": cid,
                "actor": actor,
                "text": body[:280],
                "timestamp": ts,
                "signal": "explicit_pending_or_no_response_status",
                "seq": order_pos,
            }
            pending_evidence.append(ev)
            timeline.append((dt_key, order_pos, "pending", ev))

        if _ATTEMPTED_ACTION_RE.search(body) and not has_explicit_success and not is_procedural_fail:
            ev = {
                "comment_id": cid,
                "parent_comment_id": parent_id or None,
                "actor": actor,
                "text": body[:280],
                "timestamp": ts,
                "signal": "attempted_action_unconfirmed_outcome",
                "seq": order_pos,
            }
            attempt_evidence.append(ev)
            timeline.append((dt_key, order_pos, "attempt", ev))

        # Agreement / endorsement is recorded strictly as endorsement (NOT outcome confirmation!)
        is_endorse = bool(
            (pred_resp == "agreement" or _ENDORSEMENT_ONLY_RE.search(body))
            and not has_explicit_success
            and not is_procedural_fail
            and not is_rhetorical_dispute
        )
        if is_endorse and (parent_id in sol_link_ids or pred_resp == "agreement"):
            endorsement_evidence.append(
                {
                    "comment_id": cid,
                    "parent_comment_id": parent_id or None,
                    "actor": actor,
                    "text": body[:240],
                    "timestamp": ts,
                    "signal": "endorsement_without_outcome_confirmation",
                    "seq": order_pos,
                }
            )

    # ------------------------------------------------------------------
    # Chronological & Conflict Precedence State Resolution
    # ------------------------------------------------------------------
    has_solutions = len(sol_links) > 0
    has_success = len(success_evidence) > 0
    has_pending = len(pending_evidence) > 0
    has_attempt = len(attempt_evidence) > 0
    has_explicit_failure = len(explicit_failure_evidence) > 0
    has_contradiction = len(contradictory_evidence) > 0

    success_events = [e for e in timeline if e[2] == "success"]
    failure_events = [e for e in timeline if e[2] == "failure"]
    pending_events = [e for e in timeline if e[2] == "pending"]
    unresolved_events = failure_events + pending_events

    precedence_rule = "insufficient_evidence_default"
    conflict_mode = "none"

    if has_success:
        last_succ_dt, last_succ_seq, _, last_succ_item = max(
            success_events, key=lambda x: (x[0], x[1])
        )
        later_failures = [
            e for e in failure_events if (e[0], e[1]) > (last_succ_dt, last_succ_seq)
        ]
        # Ignore same-timestamp OP title vs OP title; consider strictly later comment pending signals
        later_pending = [
            e
            for e in pending_events
            if (e[0], e[1]) > (last_succ_dt, last_succ_seq) and e[3].get("comment_id") != post_id
        ]
        distinct_actors = {
            e[3].get("actor") for e in (success_events + failure_events) if e[3].get("actor")
        }

        if later_failures or later_pending:
            has_multiple_independent_successes = len(success_events) >= 2 and len(distinct_actors) >= 2
            op_later_fail = any(e[3].get("actor") == "op" for e in (later_failures + later_pending))
            if has_multiple_independent_successes and not op_later_fail and later_failures:
                state = (
                    "reported_pending"
                    if len(failure_events) >= len(success_events)
                    else "reported_success"
                )
                precedence_rule = "conflicting_outcomes_capped_at_partial"
                conflict_mode = "multi_report_conflict"
            else:
                state = "reported_pending"
                precedence_rule = "later_failure_or_pending_overrides_prior_success"
                conflict_mode = "regression_after_success"
        elif failure_events:
            last_fail_dt, last_fail_seq, _, last_fail_item = max(
                failure_events, key=lambda x: (x[0], x[1])
            )
            same_actor_recovery = (
                last_succ_item.get("actor") == last_fail_item.get("actor")
                or last_succ_item.get("parent_comment_id") == last_fail_item.get("comment_id")
                or bool(_RECOVERY_CUE_RE.search(str(last_succ_item.get("text") or "")))
            )
            if (
                len(failure_events) == 1
                and same_actor_recovery
                and (last_succ_dt, last_succ_seq) > (last_fail_dt, last_fail_seq)
            ):
                state = "reported_success"
                precedence_rule = "chronological_recovery_after_initial_failure"
                conflict_mode = "resolved_after_retry"
            else:
                state = (
                    "reported_pending"
                    if len(failure_events) >= len(success_events)
                    else "reported_success"
                )
                precedence_rule = "conflicting_outcomes_capped_at_partial"
                conflict_mode = "multi_report_conflict"
        elif pending_events:
            state = "reported_success"
            precedence_rule = "confirmed_success_supersedes_initial_pending"
        else:
            state = "reported_success"
            precedence_rule = "clean_reported_success"
    elif has_pending or has_explicit_failure or has_attempt:
        state = "reported_pending"
        precedence_rule = "reported_pending_or_attempt_or_failure_unresolved"
    elif has_solutions:
        state = "proposed_only"
        precedence_rule = "proposed_solution_only_unresolved"
    else:
        state = "insufficient_evidence"
        precedence_rule = "insufficient_evidence_default"

    # Compute transparent 8-signal explainable score in [0, 1]
    # Discussion volume & endorsement alone NEVER yield Resolved!
    n_comments = len(cmts)
    sol_density = min(1.0, len(sol_links) / max(n_comments, 1)) if n_comments else 0.0
    mean_sol_rel = (
        round(sum(s["matching_score"] for s in sol_links) / len(sol_links), 4)
        if sol_links
        else 0.0
    )
    best_sol_rel = max((s["matching_score"] for s in sol_links), default=0.0)
    success_sig = 1.0 if state == "reported_success" else (0.35 if has_success else 0.0)
    # Only peer outcome execution confirms raise score; generic agreement gets 0 confirmation boost
    peer_outcome_count = sum(
        1
        for ev in success_evidence
        if ev.get("parent_comment_id") in sol_link_ids
    )
    confirm_sig = min(1.0, peer_outcome_count / 2.0)
    if conflict_mode == "resolved_after_retry":
        # Earlier resolved failure stays logged in evidence but does not block confirmed recovery
        contradiction_pen = 0.05
    else:
        contradiction_pen = min(
            0.55,
            0.24 * len(contradictory_evidence)
            + (0.12 if has_pending and state != "reported_success" else 0.0)
            + (0.15 if conflict_mode in {"regression_after_success", "multi_report_conflict"} else 0.0),
        )

    raw_score = (
        0.44 * success_sig
        + 0.24 * best_sol_rel
        + 0.14 * min(1.0, sol_density * 2.5)
        + 0.10 * confirm_sig
        + 0.08 * mean_sol_rel
        - contradiction_pen
    )
    if state != "reported_success":
        raw_score = min(raw_score, 0.55)
    if conflict_mode == "multi_report_conflict":
        raw_score = min(raw_score, 0.58)
    resolution_score = round(max(0.0, min(1.0, raw_score)), 4)

    # Map to 4 transparent verdict tiers with strict epistemic guardrails
    if (
        n_comments < 2
        and not has_solutions
        and not has_success
        and not has_pending
        and not has_attempt
        and not has_contradiction
    ):
        verdict = "Insufficient Evidence"
    elif (
        conflict_mode == "multi_report_conflict"
        or (state == "reported_success" and rhetorical_dispute_evidence)
    ):
        verdict = "Partially Resolved"
    elif state == "reported_success":
        if resolution_score >= 0.62 and (
            not has_explicit_failure or conflict_mode == "resolved_after_retry"
        ):
            verdict = "Resolved"
        else:
            verdict = "Partially Resolved"
    elif state in ("proposed_only", "reported_pending") or has_contradiction or has_attempt:
        verdict = "Unresolved"
    else:
        verdict = "Insufficient Evidence"

    # Evidence strength / confidence (volume guard prevents overclaiming on tiny threads)
    volume_guard = min(1.0, n_comments / 6.0)
    evidence_items = (
        len(sol_links)
        + len(success_evidence)
        + len(pending_evidence)
        + len(attempt_evidence)
        + len(contradictory_evidence)
    )
    confidence = round(
        max(0.15, min(0.96, 0.30 + 0.40 * volume_guard + 0.30 * min(1.0, evidence_items / 4.0))),
        4,
    )

    exp_parts = [
        f"State='{state}' -> Verdict='{verdict}' (score={resolution_score:.2f}, confidence={confidence:.2f}, rule={precedence_rule}).",
        f"Thread has {n_comments} comments over {discussion_span_hours:.1f}h ({days_since_post:.1f}d vs {ref_source}) and {len(sol_links)} linked solution(s) (best relevance={best_sol_rel:.2f}).",
    ]
    if has_success:
        exp_parts.append(f"Explicit reported-success signal(s) found in {len(success_evidence)} item(s).")
    else:
        exp_parts.append("No explicit reported-success confirmation found; proposed solutions and agreement endorsements never count as confirmed resolution.")
    if endorsement_evidence:
        exp_parts.append(f"Recorded {len(endorsement_evidence)} agreement/endorsement reply(s) separately without inflating resolution state.")
    if attempt_evidence:
        exp_parts.append(f"Detected {len(attempt_evidence)} attempted-action report(s) awaiting confirmed outcome.")
    if has_contradiction:
        exp_parts.append(f"Contradictory/failed follow-up evidence detected in {len(contradictory_evidence)} item(s) (penalty -{contradiction_pen:.2f}).")
    if has_pending:
        exp_parts.append(f"Pending/no-response status reported in {len(pending_evidence)} item(s).")

    top_sol = max(sol_links, key=lambda s: s["matching_score"]) if sol_links else None
    if state == "reported_pending" and (explicit_failure_evidence or pending_evidence or attempt_evidence):
        reported_outcome_item = (
            explicit_failure_evidence[-1]
            if explicit_failure_evidence
            else (pending_evidence[-1] if pending_evidence else attempt_evidence[-1])
        )
    elif success_evidence:
        reported_outcome_item = success_evidence[-1]
    elif contradictory_evidence:
        reported_outcome_item = contradictory_evidence[-1]
    elif pending_evidence:
        reported_outcome_item = pending_evidence[-1]
    elif attempt_evidence:
        reported_outcome_item = attempt_evidence[-1]
    else:
        reported_outcome_item = None

    return {
        "post_id": post_id,
        "post_title": post_title,
        "issue_id": issue_id,
        "subreddit": post.get("subreddit", ""),
        "resolution_state": state,
        "verdict": verdict,
        "precedence_rule": precedence_rule,
        "conflict_mode": conflict_mode,
        "resolution_score": resolution_score,
        "confidence": confidence,
        "timestamp": str(post.get("created_at") or ""),
        "latest_comment_timestamp": latest_dt.isoformat() if latest_dt else "",
        "reference_time": ref_dt.isoformat() if ref_dt else "",
        "reference_time_source": ref_source,
        "signals": {
            "discussion_volume": n_comments,
            "solution_oriented_comments": len(sol_links),
            "solution_density": round(sol_density, 4),
            "mean_solution_relevance": mean_sol_rel,
            "best_solution_relevance": round(best_sol_rel, 4),
            "reported_success_count": len(success_evidence),
            "endorsement_only_count": len(endorsement_evidence),
            "attempted_action_count": len(attempt_evidence),
            "followup_confirmations_count": peer_outcome_count,
            "pending_signals_count": len(pending_evidence),
            "contradictory_followups_count": len(contradictory_evidence),
            "explicit_failure_count": len(explicit_failure_evidence),
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
        "follow_up_evidence": endorsement_evidence[:3],
        "endorsement_evidence": endorsement_evidence[:5],
        "attempt_evidence": attempt_evidence[:5],
        "contradictory_evidence": contradictory_evidence[:5],
        "explicit_failure_evidence": explicit_failure_evidence[:5],
        "pending_evidence": pending_evidence[:5],
        "success_evidence": success_evidence[:5],
        "explanation": " ".join(exp_parts),
        "verification_status": "model_inferred",
    }


def evaluate_resolution_predictions(
    gold_records: Sequence[Dict[str, Any]],
    pred_by_post_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute 4-state P/R/F1 plus separated false reported_success & false Partially Resolved claim rates."""
    if not gold_records:
        return {"n_threads": 0, "error": "empty_gold_records"}

    y_true: List[str] = []
    y_pred: List[str] = []
    cm: Dict[str, Dict[str, int]] = {g: {p: 0 for p in RESOLUTION_STATES} for g in RESOLUTION_STATES}

    non_success_gold = 0
    false_reported_success = 0
    unresolved_gold = 0
    false_resolved_verdict = 0
    false_partial_on_unresolved = 0
    unresolved_tp = 0
    unresolved_fp = 0
    unresolved_fn = 0

    false_positives_detail: List[Dict[str, Any]] = []
    false_negatives_detail: List[Dict[str, Any]] = []

    for row in gold_records:
        pid = str(row["post_id"])
        g_state = str(row["gold_resolution_state"])
        g_unres = bool(row.get("gold_is_unresolved", g_state != "reported_success"))
        pred = pred_by_post_id.get(pid) or {}
        p_state = str(pred.get("resolution_state") or "insufficient_evidence")
        p_verdict = str(pred.get("verdict") or "Insufficient Evidence")
        p_unres = p_state != "reported_success" and p_verdict != "Resolved"

        y_true.append(g_state)
        y_pred.append(p_state)
        cm.setdefault(g_state, {p: 0 for p in RESOLUTION_STATES})
        cm[g_state][p_state] = cm[g_state].get(p_state, 0) + 1

        if g_state != "reported_success":
            non_success_gold += 1
            if p_state == "reported_success":
                false_reported_success += 1
                false_positives_detail.append(
                    {"post_id": pid, "gold_state": g_state, "pred_state": p_state, "pred_verdict": p_verdict, "claim_type": "false_reported_success"}
                )

        if g_unres:
            unresolved_gold += 1
            if p_verdict == "Resolved":
                false_resolved_verdict += 1
                false_positives_detail.append(
                    {"post_id": pid, "gold_state": g_state, "pred_state": p_state, "pred_verdict": p_verdict, "claim_type": "false_resolved_verdict"}
                )
            elif p_verdict == "Partially Resolved":
                false_partial_on_unresolved += 1
                false_positives_detail.append(
                    {"post_id": pid, "gold_state": g_state, "pred_state": p_state, "pred_verdict": p_verdict, "claim_type": "false_partially_resolved"}
                )

        if g_unres and p_unres:
            unresolved_tp += 1
        elif (not g_unres) and p_unres:
            unresolved_fp += 1
        elif g_unres and (not p_unres):
            unresolved_fn += 1

        if g_state == "reported_success" and p_state != "reported_success":
            false_negatives_detail.append(
                {"post_id": pid, "gold_state": g_state, "pred_state": p_state, "pred_verdict": p_verdict}
            )

    per_class: Dict[str, Any] = {}
    macro_f1s: List[float] = []
    weighted_sum = 0.0
    total_n = len(gold_records)

    for lbl in RESOLUTION_STATES:
        tp = cm[lbl][lbl]
        fp = sum(cm[g][lbl] for g in RESOLUTION_STATES if g != lbl)
        fn = sum(cm[lbl][p] for p in RESOLUTION_STATES if p != lbl)
        sup = sum(cm[lbl].values())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
        per_class[lbl] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "support": sup}
        if sup > 0:
            macro_f1s.append(f1)
            weighted_sum += f1 * sup

    acc = sum(1 for g, p in zip(y_true, y_pred) if g == p) / max(total_n, 1)
    unres_prec = unresolved_tp / (unresolved_tp + unresolved_fp) if (unresolved_tp + unresolved_fp) else 0.0
    unres_rec = unresolved_tp / (unresolved_tp + unresolved_fn) if (unresolved_tp + unresolved_fn) else 0.0
    unres_f1 = (2.0 * unres_prec * unres_rec / (unres_prec + unres_rec)) if (unres_prec + unres_rec) else 0.0

    return {
        "n_threads": total_n,
        "four_state_accuracy": round(acc, 4),
        "four_state_macro_f1": round(sum(macro_f1s) / max(len(macro_f1s), 1), 4),
        "four_state_weighted_f1": round(weighted_sum / max(total_n, 1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "unresolved_binary": {
            "precision": round(unres_prec, 4),
            "recall": round(unres_rec, 4),
            "f1": round(unres_f1, 4),
        },
        "false_claim_breakdown": {
            "non_success_gold_count": non_success_gold,
            "false_reported_success_count": false_reported_success,
            "false_reported_success_rate": round(false_reported_success / max(non_success_gold, 1), 4),
            "unresolved_gold_count": unresolved_gold,
            "false_resolved_verdict_count": false_resolved_verdict,
            "false_resolved_verdict_rate": round(false_resolved_verdict / max(unresolved_gold, 1), 4),
            "false_partially_resolved_count": false_partial_on_unresolved,
            "false_partially_resolved_rate": round(false_partial_on_unresolved / max(unresolved_gold, 1), 4),
            "combined_false_resolution_overclaim_count": false_resolved_verdict + false_partial_on_unresolved,
            "combined_false_resolution_overclaim_rate": round(
                (false_resolved_verdict + false_partial_on_unresolved) / max(unresolved_gold, 1), 4
            ),
        },
        "false_positives_sample": false_positives_detail[:10],
        "false_negatives_sample": false_negatives_detail[:10],
    }


class ExplainableResolutionEngine:
    """Thread-level and Issue-domain-level explainable unresolved-issue analyzer."""

    def analyze_corpus(
        self,
        posts: Sequence[Dict[str, Any]],
        comments_by_post: Dict[str, List[Dict[str, Any]]],
        linked_solutions: Sequence[Dict[str, Any]],
        *,
        reference_time: Optional[Union[datetime, str]] = None,
        observation_timestamp: Optional[Union[datetime, str]] = None,
    ) -> Dict[str, Any]:
        # Derive deterministic corpus-wide observation timestamp if none explicitly supplied
        effective_ref: Optional[datetime] = _parse_iso(reference_time)
        ref_source = "explicit_reference_time"
        if effective_ref is None and observation_timestamp is not None:
            effective_ref = _parse_iso(observation_timestamp)
            ref_source = "dataset_observation_timestamp"
        if effective_ref is None:
            all_dts: List[datetime] = []
            for p in posts:
                p_dt = _parse_iso(p.get("created_at"))
                if p_dt is not None:
                    all_dts.append(p_dt)
                pid = str(p.get("reddit_id") or "")
                for c in comments_by_post.get(pid, []):
                    c_dt = _parse_iso(c.get("created_at"))
                    if c_dt is not None:
                        all_dts.append(c_dt)
            if all_dts:
                effective_ref = max(all_dts)
                ref_source = "corpus_max_observation_timestamp"
            else:
                ref_source = "unavailable"

        thread_assessments: List[Dict[str, Any]] = []
        for p in posts:
            pid = str(p.get("reddit_id") or "")
            ass = assess_thread_resolution(
                p,
                comments_by_post.get(pid, []),
                linked_solutions,
                reference_time=effective_ref,
            )
            if effective_ref is not None and ref_source != "explicit_reference_time":
                ass["reference_time_source"] = ref_source
            thread_assessments.append(ass)

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
            unresolved_rate = (
                state_counts["reported_pending"]
                + state_counts["proposed_only"]
                + state_counts["insufficient_evidence"]
            ) / n_t
            contradiction_rate = contrad_threads / n_t

            if n_t < 3 or (state_counts["insufficient_evidence"] / n_t) >= 0.70:
                issue_verdict = "Insufficient Evidence"
            elif confirmed_rate >= 0.35 and contradiction_rate < 0.25:
                issue_verdict = "Resolved"
            elif confirmed_rate >= 0.10 or (verdict_counts["Partially Resolved"] / n_t) >= 0.25:
                issue_verdict = "Partially Resolved"
            else:
                issue_verdict = "Unresolved"

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

            issue_summaries.append(
                {
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
                }
            )

        issue_summaries.sort(key=lambda x: (x["unresolved_rate"], x["threads"]), reverse=True)

        overall_states = Counter(t["resolution_state"] for t in thread_assessments)
        overall_verdicts = Counter(t["verdict"] for t in thread_assessments)
        return {
            "timestamp_semantics": {
                "reference_time": effective_ref.isoformat() if effective_ref else "",
                "reference_time_source": ref_source,
                "policy": "Derived deterministically from explicit reference_time, dataset observation_timestamp, or max corpus activity timestamp; never uses hard-coded dates or implicit wall clock.",
            },
            "scoring_policy": {
                "states": list(RESOLUTION_STATES),
                "verdict_tiers": list(VERDICT_TIERS),
                "precedence_rules": list(PRECEDENCE_RULES),
                "guardrails": [
                    "Proposed solutions and agreement endorsements NEVER earn Resolved or Partially Resolved without outcome evidence.",
                    "Later explicit failure or pending reports override earlier success claims chronologically.",
                    "Conflicting multi-user outcomes cap verdict at Partially Resolved (never Resolved).",
                    "Discussion volume NEVER increases resolution_score; it only scales the confidence volume guard.",
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
