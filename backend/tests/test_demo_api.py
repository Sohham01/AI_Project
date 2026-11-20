from __future__ import annotations

import re

from fastapi import HTTPException

from app.api import demo


def test_demo_endpoint_payloads_validate_and_mark_provenance() -> None:
    demo._data.cache_clear()
    listing = demo.discussions("", "", "", "", "", "", 1, 12)
    assert listing["items"], "Expected live corpus discussions"
    first_post_id = listing["items"][0]["id"]

    expected = [
        (demo.HealthResponse, demo.demo_health()),
        (demo.SummaryResponse, demo.summary()),
        (demo.DiscussionListResponse, listing),
        (demo.DiscussionDetailResponse, demo.discussion_detail(first_post_id)),
        (demo.ItemsResponse, demo.issues()),
        (demo.ItemsResponse, demo.topics()),
        (demo.SentimentResponse, demo.sentiment()),
        (demo.ItemsResponse, demo.responses()),
        (demo.ItemsResponse, demo.resolution()),
        (demo.ModelsResponse, demo.models()),
    ]
    for response_model, payload in expected:
        response_model.model_validate(payload)

    summary = demo.summary()
    assert summary["synthetic_only"] is False
    assert summary["provenance"]["authorization_status"] == "public_anonymized_reddit"
    assert summary["provenance"]["source_provider"] == "arctic_shift"
    assert summary["provenance"]["dapt_summary"] is not None
    assert summary["provenance"]["filtering_summary"] is not None
    assert summary["posts"] >= 70
    assert summary["comments"] >= 700
    assert summary["topic_count"] == 10


def test_discussion_search_filters_and_pagination() -> None:
    demo._data.cache_clear()
    first_page = demo.discussions("", "", "", "", "", "", 1, 10)
    second_page = demo.discussions("", "", "", "", "", "", 2, 10)
    assert first_page["total"] >= 20
    assert len(first_page["items"]) == 10
    assert len(second_page["items"]) == 10
    assert first_page["items"][0]["id"] != second_page["items"][0]["id"]

    searched = demo.discussions("sambalpur", "", "", "", "", "", 1, 100)
    assert searched["total"] > 0

    summary = demo.summary()
    top_comm = summary["community_distribution"][0]
    community = demo.discussions("", top_comm["label"], "", "", "", "", 1, 200)
    assert community["total"] == top_comm["count"]

    top_issue = summary["issue_distribution"][0]
    by_issue = demo.discussions("", "", top_issue["key"], "", "", "", 1, 200)
    assert by_issue["total"] == top_issue["posts"]


def test_invalid_demo_filters_return_422() -> None:
    for kwargs in (
        {"issue": "not-a-valid-issue"},
        {"sentiment": "not-a-sentiment"},
        {"response": "not-a-response"},
        {"community": "not-a-valid-community"},
    ):
        params = {"q": "", "community": "", "issue": "", "sentiment": "", "emotion": "", "response": "", "page": 1, "page_size": 12}
        params.update(kwargs)
        try:
            demo.discussions(**params)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"Expected 422 for invalid filter: {kwargs}")


def test_resolution_evidence_requires_explicit_outcome_text() -> None:
    rows = demo.resolution()["items"]
    assert rows
    valid_states = {"reported_success", "reported_pending", "proposed_only", "insufficient_evidence"}
    for row in rows:
        assert row["state"] in valid_states
        if row["state"] in {"reported_success", "reported_pending"}:
            assert row["evidence_text"]
            assert row["evidence_type"] == row["state"]
        else:
            assert row["evidence_text"] is None


def test_live_corpus_api_exposes_zero_usernames_or_raw_pii() -> None:
    demo._data.cache_clear()
    user_mention_re = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,24}\b")
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    listing = demo.discussions("", "", "", "", "", "", 1, 100)
    for item in listing["items"]:
        detail = demo.discussion_detail(item["id"])
        assert "author" not in detail["post"]
        assert "author_fullname" not in detail["post"]
        post_text = f"{detail['post'].get('title') or ''} {detail['post'].get('body') or ''}"
        assert not user_mention_re.search(post_text)
        assert not email_re.search(post_text)
        for comment in detail["comments"]:
            assert "author" not in comment
            assert "author_fullname" not in comment
            body = comment.get("body") or ""
            assert not user_mention_re.search(body)
            assert not email_re.search(body)
