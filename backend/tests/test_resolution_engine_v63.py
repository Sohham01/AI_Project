"""Sprint P0-B Regression Tests — Dynamic Timestamps, Confirmation Semantics & Chronological Precedence."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.linking.resolution_engine import (
    ExplainableResolutionEngine,
    assess_thread_resolution,
    evaluate_resolution_predictions,
)


def _linked(post_id: str, comment_id: str = "c_sol", score: float = 0.85) -> list[dict]:
    return [
        {
            "post_id": post_id,
            "solution_comment_id": comment_id,
            "linked": True,
            "matching_score": score,
            "evidence_text": "Email accounts desk with bank sanction letter.",
            "matching_rationale": "High similarity",
        }
    ]


def test_dynamic_reference_time_not_hardcoded_and_deterministic() -> None:
    post = {
        "reddit_id": "p_ts",
        "title": "Hostel WiFi issue",
        "created_at": "2027-01-01T00:00:00Z",
    }
    comments = [
        {"reddit_id": "c1", "body": "Still waiting for IT team fix.", "created_at": "2027-01-06T00:00:00Z"}
    ]

    # Without explicit reference_time, uses latest thread activity timestamp (NOT 2026-04-15!)
    res_auto = assess_thread_resolution(post, comments, [])
    assert res_auto["reference_time_source"] == "thread_latest_activity_timestamp"
    assert res_auto["signals"]["days_since_post"] == 5.0

    # With explicit reference_time -> exact deterministic elapsed days
    ref = datetime(2027, 1, 11, 0, 0, tzinfo=timezone.utc)
    res_exp = assess_thread_resolution(post, comments, [], reference_time=ref)
    assert res_exp["reference_time_source"] == "explicit_reference_time"
    assert res_exp["signals"]["days_since_post"] == 10.0

    # Corpus engine derives corpus_max_observation_timestamp deterministically
    eng = ExplainableResolutionEngine()
    corp = eng.analyze_corpus([post], {"p_ts": comments}, [])
    assert corp["timestamp_semantics"]["reference_time_source"] == "corpus_max_observation_timestamp"
    assert corp["threads"][0]["signals"]["days_since_post"] == 5.0


def test_agreement_and_proposed_high_score_never_count_as_resolved() -> None:
    post = {"reddit_id": "p_agr", "title": "How to apply for fee extension?", "created_at": "2026-07-01T00:00:00Z"}
    comments = [
        {
            "reddit_id": "c_sol",
            "body": "Email accounts desk with your SBI sanction letter.",
            "response_type": "solution",
            "created_at": "2026-07-01T02:00:00Z",
        },
        {
            "reddit_id": "c_agr1",
            "parent_reddit_id": "c_sol",
            "body": "Agreed this is accurate +1",
            "response_type": "agreement",
            "created_at": "2026-07-01T03:00:00Z",
        },
        {
            "reddit_id": "c_agr2",
            "parent_reddit_id": "c_sol",
            "body": "Same bro totally agree",
            "response_type": "agreement",
            "created_at": "2026-07-01T04:00:00Z",
        },
    ]
    res = assess_thread_resolution(post, comments, _linked("p_agr", "c_sol", score=0.92))
    assert res["resolution_state"] == "proposed_only"
    assert res["verdict"] == "Unresolved"
    assert res["signals"]["endorsement_only_count"] == 2
    assert res["signals"]["followup_confirmations_count"] == 0
    assert res["resolution_score"] <= 0.55


def test_attempted_action_and_persisting_failure_stay_unresolved() -> None:
    post = {"reddit_id": "p_att", "title": "Portal login error", "created_at": "2026-07-01T00:00:00Z"}
    c_attempt = [
        {"reddit_id": "c_sol", "body": "Email IT support with screenshot.", "created_at": "2026-07-01T01:00:00Z"},
        {"reddit_id": "c2", "body": "I mailed them today following this.", "created_at": "2026-07-01T02:00:00Z"},
    ]
    r_att = assess_thread_resolution(post, c_attempt, _linked("p_att", "c_sol"))
    assert r_att["resolution_state"] == "reported_pending"
    assert r_att["verdict"] == "Unresolved"
    assert r_att["signals"]["attempted_action_count"] == 1

    c_persist = c_attempt + [
        {"reddit_id": "c3", "body": "Tried that but issue still persists and didn't work.", "created_at": "2026-07-02T00:00:00Z"},
    ]
    r_pers = assess_thread_resolution(post, c_persist, _linked("p_att", "c_sol"))
    assert r_pers["resolution_state"] == "reported_pending"
    assert r_pers["verdict"] == "Unresolved"
    assert r_pers["signals"]["contradictory_followups_count"] >= 1


def test_chronological_success_then_failure_vs_failure_then_recovery() -> None:
    post = {"reddit_id": "p_chr", "title": "Mess AC and water issue", "created_at": "2026-07-01T00:00:00Z"}
    linked = _linked("p_chr", "c_sol", score=0.80)

    # Case 1: Success reported at T+1d, later failure at T+3d -> later failure overrides!
    c_reg = [
        {"reddit_id": "c_sol", "body": "Raise a maintenance ticket.", "created_at": "2026-07-01T01:00:00Z"},
        {"reddit_id": "c_s", "parent_reddit_id": "c_sol", "body": "It worked for me yesterday!", "created_at": "2026-07-02T00:00:00Z"},
        {"reddit_id": "c_f", "body": "Update: still broken today, no refund or fix.", "created_at": "2026-07-04T00:00:00Z"},
    ]
    r_reg = assess_thread_resolution(post, c_reg, linked)
    assert r_reg["resolution_state"] == "reported_pending"
    assert r_reg["verdict"] == "Unresolved"
    assert r_reg["precedence_rule"] == "later_failure_or_pending_overrides_prior_success"

    # Case 2: Single initial failure at T+1d, same actor / OP explicit recovery later at T+4d -> Resolved
    c_rec = [
        {"reddit_id": "c_sol", "body": "Email the registrar escalating the ticket.", "created_at": "2026-07-01T01:00:00Z"},
        {"reddit_id": "c_f", "is_submitter": True, "body": "Mailed them, no reply yet.", "created_at": "2026-07-02T00:00:00Z"},
        {"reddit_id": "c_f2", "is_submitter": True, "body": "They didn't respond initially.", "created_at": "2026-07-02T02:00:00Z"},
        {"reddit_id": "c_ok", "is_submitter": True, "parent_reddit_id": "c_f2", "body": "Update: finally got my refund today, problem solved!", "created_at": "2026-07-05T00:00:00Z"},
    ]
    # Note: 2 failure/pending events? Wait c_f is only pending_evidence, c_f2 is 1 explicit failure -> recovery applies!
    r_rec = assess_thread_resolution(post, c_rec, linked)
    assert r_rec["resolution_state"] == "reported_success"
    assert r_rec["verdict"] == "Resolved"
    assert r_rec["precedence_rule"] == "chronological_recovery_after_initial_failure"


def test_conflicting_reports_and_generic_positive_sentiment_and_false_claim_report() -> None:
    post = {"reddit_id": "p_conf", "title": "Refund dispute", "created_at": "2026-07-01T00:00:00Z"}
    c_conflict = [
        {"reddit_id": "c_sol", "body": "Cite AICTE guidelines.", "created_at": "2026-07-01T01:00:00Z"},
        {"reddit_id": "c1", "body": "Refund received after mailing them.", "created_at": "2026-07-02T00:00:00Z"},
        {"reddit_id": "c2", "body": "Tried mailing them, they didn't work and refused me.", "created_at": "2026-07-03T00:00:00Z"},
        {"reddit_id": "c3", "body": "Issue resolved for my friend.", "created_at": "2026-07-04T00:00:00Z"},
        {"reddit_id": "c4", "body": "Same no refund for me, they didn't respond.", "created_at": "2026-07-04T06:00:00Z"},
    ]
    r_conf = assess_thread_resolution(post, c_conflict, _linked("p_conf", "c_sol"))
    # Equal successes (2) and failures (2) across different commenters -> capped at Partially Resolved, NEVER Resolved!
    assert r_conf["verdict"] == "Partially Resolved"
    assert r_conf["verdict"] != "Resolved"
    assert r_conf["precedence_rule"] == "conflicting_outcomes_capped_at_partial"

    # Generic positive sentiment without solution or outcome -> Insufficient Evidence
    r_gen = assess_thread_resolution(
        {"reddit_id": "p_gen", "title": "Campus vibes?", "created_at": "2026-07-01T00:00:00Z"},
        [
            {"reddit_id": "c1", "body": "Great peers and awesome faculty vibes!", "created_at": "2026-07-01T02:00:00Z"},
            {"reddit_id": "c2", "body": "Love the environment overall.", "created_at": "2026-07-01T03:00:00Z"},
        ],
        [],
    )
    assert r_gen["resolution_state"] == "insufficient_evidence"
    assert r_gen["verdict"] == "Insufficient Evidence"

    # Separate reporting of false reported_success vs false Partially Resolved claims
    gold = [
        {"post_id": "p1", "gold_resolution_state": "proposed_only", "gold_is_unresolved": True},
        {"post_id": "p2", "gold_resolution_state": "reported_pending", "gold_is_unresolved": True},
    ]
    preds = {
        "p1": {"resolution_state": "proposed_only", "verdict": "Partially Resolved"},
        "p2": {"resolution_state": "reported_success", "verdict": "Resolved"},
    }
    metrics = evaluate_resolution_predictions(gold, preds)
    fc = metrics["false_claim_breakdown"]
    assert fc["false_reported_success_rate"] == 0.5
    assert fc["false_resolved_verdict_rate"] == 0.5
    assert fc["false_partially_resolved_rate"] == 0.5
    assert fc["combined_false_resolution_overclaim_rate"] == 1.0
