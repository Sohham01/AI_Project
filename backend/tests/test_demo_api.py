from __future__ import annotations

from fastapi import HTTPException

from app.api import demo


def test_demo_endpoint_payloads_validate_and_mark_provenance() -> None:
    demo._data.cache_clear()
    expected = [
        (demo.HealthResponse, demo.demo_health()),
        (demo.SummaryResponse, demo.summary()),
        (demo.DiscussionListResponse, demo.discussions("", "", "", "", "", "", 1, 12)),
        (demo.DiscussionDetailResponse, demo.discussion_detail("syn-post-0001")),
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
    assert summary["synthetic_only"] is True
    assert summary["provenance"]["authorization_status"] == "synthetic_only"
    assert summary["posts"] == 160
    assert summary["comments"] == 1280


def test_discussion_search_filters_and_pagination() -> None:
    first_page = demo.discussions("", "", "placements", "", "", "", 1, 10)
    second_page = demo.discussions("", "", "placements", "", "", "", 2, 10)
    assert first_page["total"] == 20
    assert len(first_page["items"]) == 10
    assert len(second_page["items"]) == 10
    assert first_page["items"][0]["id"] != second_page["items"][0]["id"]

    searched = demo.discussions("offer comparison guide", "", "", "", "", "", 1, 100)
    assert searched["total"] > 0
    community = demo.discussions("", "IIMSambalpur", "", "", "", "", 1, 100)
    assert community["total"] == 160


def test_invalid_demo_filters_return_422() -> None:
    for kwargs in (
        {"issue": "not-a-synthetic-issue"},
        {"sentiment": "not-a-sentiment"},
        {"response": "not-a-response"},
        {"community": "not-a-synthetic-community"},
    ):
        params = {"q": "", "community": "", "issue": "", "sentiment": "", "emotion": "", "response": "", "page": 1, "page_size": 12}
        params.update(kwargs)
        try:
            demo.discussions(**params)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"Expected 422 for invalid filter: {kwargs}")


def test_resolution_evidence_requires_explicit_synthetic_outcome() -> None:
    rows = demo.resolution()["items"]
    states = {row["state"] for row in rows}
    assert states == {"reported_success", "reported_pending", "proposed_only", "insufficient_evidence"}
    for row in rows:
        if row["state"] in {"reported_success", "reported_pending"}:
            assert row["evidence_text"]
            assert row["evidence_type"] == row["state"]
        else:
            assert row["evidence_text"] is None
