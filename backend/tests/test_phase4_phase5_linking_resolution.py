"""Phase 2, 3, 4 & 5 Automated Unit & Contract Tests (Multilingual LID, Slang/Emoji, Linking, Resolution)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.nlp.indian_reddit_lid import (
    detect_indian_reddit_language,
    detect_language_spans,
    normalize_romanized_indic,
)
from app.services.nlp.preprocessing.context_normalizer import apply_informal_normalization
from ml.linking.contextual_mapper import (
    compute_negation_and_non_actionable_penalty,
    evaluate_issue_solution_ranking,
)
from ml.linking.resolution_engine import assess_thread_resolution


def test_multilingual_lid_span_detection_and_script_separation() -> None:
    # Romanized Hindi must NEVER be labelled English
    r_hi = detect_indian_reddit_language("Thik hai bhai")
    assert r_hi["language"] == "hi"
    assert r_hi["script"] == "latin"

    # Multi-span English + Hinglish comment must expose multi-language spans
    r_multi = detect_indian_reddit_language(
        "Thank God I didn't receive a call. Laakhon rupiye bach gaye."
    )
    assert r_multi["language"] == "hi-en"
    assert r_multi["is_code_mixed"] is True
    assert r_multi["has_multi_language_spans"] is True
    spans = detect_language_spans("Thank God I didn't receive a call. Laakhon rupiye bach gaye.")
    assert len(spans) == 2
    assert spans[0]["language"] == "en"
    assert spans[1]["language"] == "hi"

    # Proper noun "Bodh Gaya" must NOT falsely trigger Hindi verb "gaya"
    r_bg = detect_indian_reddit_language("XIMB > Vizag > Sambalpur > Bodh gaya > Sirmaur")
    assert r_bg["language"] == "en"

    # Native scripts & Romanized Bengali/Odia
    assert detect_indian_reddit_language("हॉस्टल और मेस का खाना खराब है")["script"] == "devanagari"
    assert detect_indian_reddit_language("এই কলেজের হোস্টেল কেমন?")["script"] == "bengali"
    assert detect_indian_reddit_language("ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro")["language"] == "or-en"
    assert detect_indian_reddit_language("Bhai nit ra try kar\nAe sabu kana pasuchu")["language"] == "or-en"

    # Numeric/emoji-only must explicitly abstain
    r_num = detect_indian_reddit_language("94/93.2/9.6")
    assert r_num["language"] == "und"
    assert r_num["abstain"] is True


def test_romanized_normalization_preserves_original_text() -> None:
    orig = "Bhai fees bohot high hai aur reply nhi aaya abhi tak"
    res = normalize_romanized_indic(orig)
    assert res["original_text"] == orig
    assert "bahut" in res["normalized_text"]
    assert "nahi" in res["normalized_text"]
    assert len(res["transformations"]) >= 2


def test_context_aware_slang_and_emoji_disambiguation() -> None:
    # Literal uses MUST be preserved untouched
    lit_mid = apply_informal_normalization("Stagnant from mid Dec to mid sem", mode="slang_normalized")
    assert "mid Dec" in lit_mid["normalized_text"] and "mediocre" not in lit_mid["normalized_text"]

    lit_based = apply_informal_normalization("Shortlist is based on CAT percentile", mode="slang_normalized")
    assert "based on" in lit_based["normalized_text"] and "boldly agreeable" not in lit_based["normalized_text"]

    lit_cooked = apply_informal_normalization("Canteen serves freshly cooked food", mode="slang_normalized")
    assert "cooked food" in lit_cooked["normalized_text"] and "doomed" not in lit_cooked["normalized_text"]

    # Gen-Z slang uses MUST expand contextually + emojis preserved semantically
    slang_res = apply_informal_normalization(
        "Mess food is literally mid tbh and we are totally cooked 😭🤡",
        mode="slang_emoji_semantic",
    )
    assert slang_res["original_text"].endswith("😭🤡")
    assert "mediocre" in slang_res["normalized_text"]
    assert "overwhelmed doomed" in slang_res["normalized_text"]
    assert "to be honest" in slang_res["normalized_text"]
    assert "sadness frustration" in slang_res["normalized_text"]
    assert "sarcasm mockery" in slang_res["normalized_text"]


def test_issue_solution_negation_penalties_and_ranking_metrics() -> None:
    # Contradictory, removed, DM-only, and ultra-short comments must be penalized
    p_rm, _ = compute_negation_and_non_actionable_penalty("[ Removed by Reddit ]", "solution")
    assert p_rm >= 0.25
    p_dm, _ = compute_negation_and_non_actionable_penalty("Please dm me", "advice")
    assert p_dm >= 0.25
    p_neg, _ = compute_negation_and_non_actionable_penalty("This doesn't work and is a waste of money.", "counterargument")
    assert p_neg >= 0.25
    # Actionable solution has zero penalty
    p_ok, _ = compute_negation_and_non_actionable_penalty(
        "Check the official AICTE 15-day full refund guideline and email admissions with your receipt.",
        "solution",
    )
    assert p_ok == 0.0

    # Ranking evaluator sanity check
    pairs = [
        {"issue_id": "placements", "matching_score": 0.85, "gold_relevant": 2},
        {"issue_id": "placements", "matching_score": 0.72, "gold_relevant": 1},
        {"issue_id": "placements", "matching_score": 0.20, "gold_relevant": 0},
    ]
    ev = evaluate_issue_solution_ranking(pairs, threshold=0.56)
    assert ev["pairwise_f1"] == 1.0
    assert ev["false_positive_rate"] == 0.0
    assert ev["mrr"] == 1.0


def test_resolution_engine_all_five_scenarios() -> None:
    base_post = {
        "reddit_id": "p_test",
        "title": "Refund not received after seat withdrawal",
        "body": "Withdrew 2 weeks ago, still waiting.",
        "issue_key": "financial_concerns",
        "created_at": "2026-06-01T10:00:00Z",
    }
    linked = [{
        "post_id": "p_test",
        "solution_comment_id": "c_sol",
        "linked": True,
        "matching_score": 0.78,
        "evidence_text": "Email the registrar citing AICTE 15-day refund guidelines.",
        "matching_rationale": "High semantic alignment",
    }]

    # 1. Proposed-only MUST NEVER be called Resolved!
    r_prop = assess_thread_resolution(
        {**base_post, "body": "How do I get a fee refund?"},
        [{"reddit_id": "c_sol", "body": "Email the registrar citing AICTE 15-day refund guidelines.", "created_at": "2026-06-01T12:00:00Z"}],
        linked,
    )
    assert r_prop["resolution_state"] == "proposed_only"
    assert r_prop["verdict"] != "Resolved"
    assert r_prop["resolution_score"] <= 0.58

    # 2. Reported success (explicit confirmation, no contradictions -> Resolved)
    r_succ = assess_thread_resolution(
        {**base_post, "body": "How do I get a fee refund?"},
        [
            {"reddit_id": "c_sol", "body": "Email the registrar citing AICTE 15-day refund guidelines.", "created_at": "2026-06-01T12:00:00Z"},
            {"reddit_id": "c_ok", "parent_reddit_id": "c_sol", "body": "Update: that worked, I finally got my refund today!", "created_at": "2026-06-05T12:00:00Z"},
        ],
        linked,
    )
    assert r_succ["resolution_state"] == "reported_success"
    assert r_succ["verdict"] == "Resolved"

    # 3. Pending status
    r_pend = assess_thread_resolution(
        base_post,
        [{"reddit_id": "c1", "body": "Any update? Still waiting for refund email.", "created_at": "2026-06-02T10:00:00Z"}],
        [],
    )
    assert r_pend["resolution_state"] == "reported_pending"
    assert r_pend["verdict"] == "Unresolved"

    # 4. Contradictory follow-up evidence (solution tried -> failed/no response!)
    r_contr = assess_thread_resolution(
        {**base_post, "body": "How do I get a fee refund?"},
        [
            {"reddit_id": "c_sol", "body": "Email the registrar citing AICTE 15-day refund guidelines.", "created_at": "2026-06-01T12:00:00Z"},
            {"reddit_id": "c_fail", "parent_reddit_id": "c_sol", "body": "Tried mailing them twice, they didn't respond and are not picking up calls.", "created_at": "2026-06-04T12:00:00Z"},
        ],
        linked,
    )
    assert r_contr["resolution_state"] == "reported_pending"
    assert r_contr["signals"]["contradictory_followups_count"] >= 1
    assert r_contr["verdict"] == "Unresolved"

    # 5. Insufficient evidence
    r_insuff = assess_thread_resolution(
        {**base_post, "title": "Profile check?", "body": "94/93/9"},
        [{"reddit_id": "c0", "body": "Nice profile", "created_at": "2026-06-01T11:00:00Z"}],
        [],
    )
    assert r_insuff["resolution_state"] == "insufficient_evidence"
    assert r_insuff["verdict"] == "Insufficient Evidence"
