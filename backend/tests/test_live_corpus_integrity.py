"""Integrity, privacy, and scientific honesty checks for the 100%-real live corpus bundle."""

from __future__ import annotations

import json
from pathlib import Path
import re

from app.api.demo import RESPONSE_TYPES, _data
from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES as CANONICAL_RESPONSE_TYPES,
)
from ml.evaluation.benchmark_verifier import verify_benchmark_artifact

ROOT = Path(__file__).resolve().parents[2]
LIVE_DATA = ROOT / "data" / "live_corpus"
LIVE_ARTIFACTS = ROOT / "ml" / "artifacts" / "live_corpus"
REAL_LABELED = ROOT / "data" / "annotations" / "real_reddit_labeled_v1"

_USER_RE = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,24}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_REDDIT_URL_RE = re.compile(r"https?://(?:www\.|old\.)?reddit\.com/r/", re.I)
_PSEUDO_POST_RE = re.compile(r"^post_\d{4,}$")
_PSEUDO_CMT_RE = re.compile(r"^cmt_\d{5,}$")
_ALLOWED_LANGS = {"en", "hi", "hi-en", "bn", "bn-en", "or", "or-en", "ta", "ta-en", "te", "te-en", "kn", "ml", "gu", "pa", "und"}


def test_synthetic_bundles_removed_and_real_labeled_data_tracked() -> None:
    assert not (ROOT / "data" / "demo_synthetic").exists()
    assert not (ROOT / "ml" / "artifacts" / "demo_synthetic").exists()
    assert not (ROOT / "ml" / "artifacts" / "demo_synthetic_v2").exists()
    assert (LIVE_DATA / "corpus.json").is_file()
    assert (LIVE_ARTIFACTS / "evaluation.json").is_file()
    assert (LIVE_ARTIFACTS / "predictions.json").is_file()
    assert (LIVE_ARTIFACTS / "topic_discovery.json").is_file()
    for task_csv in ("master_labeled_v1.csv", "issue.csv", "sentiment.csv", "emotion.csv", "response.csv", "solution.csv"):
        assert (REAL_LABELED / task_csv).is_file(), f"Missing tracked real labeled file: {task_csv}"


def test_real_corpus_pseudonymized_zero_pii_and_clean_languages() -> None:
    corpus = json.loads((LIVE_DATA / "corpus.json").read_text(encoding="utf-8"))
    metadata = corpus["metadata"]
    posts = corpus["posts"]
    comments = [c for p in posts for c in p["comments"]]

    assert metadata["synthetic"] is False
    assert metadata["post_count"] == len(posts) >= 300
    assert metadata["comment_count"] == len(comments) >= 3000
    assert len({p["reddit_id"] for p in posts}) == len(posts)
    assert len({c["reddit_id"] for c in comments}) == len(comments)

    assert set(RESPONSE_TYPES) == set(CANONICAL_RESPONSE_TYPES)
    hinglish_or_hi = 0
    for post in posts:
        assert _PSEUDO_POST_RE.match(post["reddit_id"]), f"Non-pseudonymized post id: {post['reddit_id']}"
        assert not post.get("url"), f"Raw URL exposed on {post['reddit_id']}"
        assert "author" not in post
        assert "author_fullname" not in post
        blob = f"{post.get('title', '')} {post.get('body', '')}"
        assert not _USER_RE.search(blob)
        assert not _EMAIL_RE.search(blob)
        assert not _REDDIT_URL_RE.search(blob)
        for comment in post["comments"]:
            assert _PSEUDO_CMT_RE.match(comment["reddit_id"]), f"Non-pseudonymized comment id: {comment['reddit_id']}"
            assert "author" not in comment
            assert comment["synthetic"] is False
            assert comment["response_type"] in CANONICAL_RESPONSE_TYPES
            assert comment["language"] in _ALLOWED_LANGS, f"Bogus language tag: {comment['language']}"
            if comment["language"] in {"hi", "hi-en"}:
                hinglish_or_hi += 1
            body = comment.get("body", "")
            assert not _USER_RE.search(body)
            assert not _EMAIL_RE.search(body)
            assert not _REDDIT_URL_RE.search(body)
    assert hinglish_or_hi >= 100, f"Expected real Hinglish/Hindi comments detected, got {hinglish_or_hi}"


def test_live_artifacts_pass_all_seven_pipeline_integrity_checks() -> None:
    evaluation = json.loads((LIVE_ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    predictions = json.loads((LIVE_ARTIFACTS / "predictions.json").read_text(encoding="utf-8"))
    corpus = json.loads((LIVE_DATA / "corpus.json").read_text(encoding="utf-8"))

    assert evaluation["synthetic_only"] is False
    assert evaluation["dataset_version"] == "6.1.0"
    assert evaluation["labeled_data_provenance"]["synthetic_training_rows"] == 0
    assert evaluation["labeled_data_provenance"]["labeled_real_comments"] >= 400
    assert evaluation["labeled_data_provenance"]["v2_expansion_blocked_by_iaa_gate"] is True
    assert evaluation["dapt_summary"]["zero_test_thread_leakage"] is True
    assert int(evaluation["dapt_summary"]["excluded_frozen_test_threads"]) >= 37

    # Ensure every task has realistic non-sharpening temperature T >= 1.0 and uses MuRIL (alpha >= 0.5)
    for t_name, t_obj in evaluation["tasks"].items():
        assert float(t_obj["temperature"]) >= 1.0, (t_name, t_obj["temperature"])
        assert float(t_obj["hybrid_blend"]["alpha_muril"]) >= 0.5, (t_name, t_obj["hybrid_blend"])

    report = verify_benchmark_artifact(evaluation, predictions=predictions, corpus=corpus)
    assert report["verified"] is True, report
    assert report["passed_checks"] == report["total_checks"] == 7


def test_demo_store_loads_real_corpus_with_confidence_profiles() -> None:
    store = _data()
    assert store["metadata"]["synthetic"] is False
    assert len(store["posts"]) >= 300
    assert len(store["comments"]) >= 3000
    sample = store["comments"][0]
    assert "confidence_profile" in sample
    assert 0.0 <= sample["confidence_profile"]["overall_confidence"] <= 1.0
    assert sample["confidence_profile"]["confidence_tier"] in {"high", "medium", "low"}
