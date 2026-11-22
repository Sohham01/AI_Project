"""Tests for Stage 1 upgrades: Indian-Reddit LID (Task 5), Emoji/Slang (Task 6),
Issue-Solution Linking & Unresolved Criteria (Tasks 10-11), Cross-IIM &
Cross-Language Comparisons (Q8), and Live Multilingual Analyzer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from app.services.nlp.preprocessing.emoji_normalizer import (
    extract_emojis,
    normalize_emojis,
    normalize_emojis_semantic_affect,
    strip_emojis,
)
from app.services.nlp.preprocessing.slang_normalizer import (
    detect_slang_terms,
    normalize_slang,
)


def test_indian_reddit_lid_handles_all_pdf_languages_and_scripts() -> None:
    cases = [
        ("Bhai IIM ABC ka summer placement scene kaisa hai 😭", "hi-en", "latin", True),
        ("ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro", "or-en", "mixed", True),
        ("IIM e giye ki actually ROI pawa jay?", "bn-en", "latin", True),
        ("Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl.", "hi-en", "latin", True),
        ("IIM jaana worth it hai kya?", "hi-en", "latin", True),
        ("यह कॉलेज का प्लेसमेंट कैसा है?", "hi", "devanagari", False),
        ("এই কলেজের হোস্টেল কেমন?", "bn", "bengali", False),
        ("ଆଜିର କ୍ଲାସ୍ ବହୁତ ଭଲ ଥିଲା", "or", "odia", False),
        ("Try applying for financial aid.", "en", "latin", False),
    ]
    for text, exp_lang, exp_script, exp_cm in cases:
        res = detect_indian_reddit_language(text)
        assert res["language"] == exp_lang, (text, res)
        assert res["script"] == exp_script, (text, res)
        assert res["is_code_mixed"] is exp_cm, (text, res)


def test_pdf_emoji_and_slang_enhancements_preserve_backward_compatibility() -> None:
    assert "<EMOJI_skull>" in normalize_emojis("mess food is 💀")
    assert "<EMOJI_clown_face>" in normalize_emojis("rejected at 99%ile 🤡")
    assert "extreme amusement disbelief" in normalize_emojis_semantic_affect("mess food is 💀")
    assert "sadness frustration" in normalize_emojis_semantic_affect("fees too high 😭")
    assert "strong approval" in normalize_emojis_semantic_affect("placements 🔥")
    assert strip_emojis("mess food is 💀") == "mess food is"
    extracted = extract_emojis("hostel 💀 fees 😭")
    assert len(extracted) == 2

    # Slang: unambiguous abbreviations expand; tone/context-sensitive terms preserved safely
    assert normalize_slang("fr bro based on mid 30s ngl") == "for real bro based on mid 30s not gonna lie"
    detected = {item["term"]: item["action"] for item in detect_slang_terms("fr bro bhai mid based cooked ngl lol")}
    assert detected["fr"] == "expanded"
    assert detected["ngl"] == "expanded"
    assert detected["bro"] == "preserved_tone"
    assert detected["bhai"] == "preserved_tone"
    assert detected["lol"] == "preserved_tone"
    assert detected["mid"] == "context_preserved"
    assert detected["based"] == "context_preserved"
    assert detected["cooked"] == "context_preserved"


def test_resolution_comparisons_and_live_analyze_endpoints() -> None:
    client = TestClient(app)

    res = client.get("/api/demo/resolution")
    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) >= 300
    matrix = body["unresolved_matrix"]
    assert len(matrix["rows"]) == 8
    for row in matrix["rows"]:
        assert 0.0 <= row["linked_thread_coverage"] <= 1.0
        assert row["status"] in {"Resolved", "Partially Resolved", "Unresolved"}
        assert row["resolution_strength"] in {"High", "Moderate", "Low"}
    clusters = body["issue_solution_clusters"]
    assert len(clusters["by_issue"]) == 8
    assert body["linker_summary"]["total_asserted_links"] >= 100

    cmp_res = client.get("/api/demo/comparisons")
    assert cmp_res.status_code == 200
    cmp_body = cmp_res.json()
    assert cmp_body["cross_iim"]["total_distinct_iims_detected"] >= 15
    assert len(cmp_body["cross_iim"]["iims"]) >= 8
    assert len(cmp_body["cross_language"]["registers"]) == 3

    an_res = client.post(
        "/api/demo/analyze",
        json={"text": "Bro hostel life 💀... mess food toh literally mid... try food court near campus ngl"},
    )
    assert an_res.status_code == 200
    an_body = an_res.json()
    assert an_body["language_detection"]["language"] == "hi-en"
    assert an_body["language_detection"]["is_code_mixed"] is True
    assert len(an_body["preprocessing"]["emojis_detected"]) == 1
    assert len(an_body["preprocessing"]["slang_detected"]) >= 3
    assert an_body["predictions"]["sentiment"]["label"] in {"negative", "neutral", "positive"}
    assert len(an_body["nearest_v6_neighbors"]) >= 1
