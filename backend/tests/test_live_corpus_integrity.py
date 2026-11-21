"""Integrity & scientific honesty checks for the real anonymized live corpus bundle."""

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

_USER_RE = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,24}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def test_synthetic_bundles_removed_and_live_corpus_active() -> None:
    assert not (ROOT / "data" / "demo_synthetic").exists()
    assert not (ROOT / "ml" / "artifacts" / "demo_synthetic").exists()
    assert not (ROOT / "ml" / "artifacts" / "demo_synthetic_v2").exists()
    assert (LIVE_DATA / "corpus.json").is_file()
    assert (LIVE_ARTIFACTS / "evaluation.json").is_file()
    assert (LIVE_ARTIFACTS / "predictions.json").is_file()
    assert (LIVE_ARTIFACTS / "topic_discovery.json").is_file()


def test_real_corpus_zero_pii_and_frozen_response_taxonomy() -> None:
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
    for post in posts:
        assert "author" not in post
        assert "author_fullname" not in post
        blob = f"{post.get('title', '')} {post.get('body', '')}"
        assert not _USER_RE.search(blob)
        assert not _EMAIL_RE.search(blob)
        for comment in post["comments"]:
            assert "author" not in comment
            assert comment["synthetic"] is False
            assert comment["response_type"] in CANONICAL_RESPONSE_TYPES
            assert not _USER_RE.search(comment.get("body", ""))
            assert not _EMAIL_RE.search(comment.get("body", ""))


def test_live_artifacts_pass_all_seven_benchmark_verification_checks() -> None:
    evaluation = json.loads((LIVE_ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    predictions = json.loads((LIVE_ARTIFACTS / "predictions.json").read_text(encoding="utf-8"))
    corpus = json.loads((LIVE_DATA / "corpus.json").read_text(encoding="utf-8"))

    assert evaluation["synthetic_only"] is False
    assert evaluation["dataset_version"] == "4.0.0"

    report = verify_benchmark_artifact(evaluation, predictions=predictions, corpus=corpus)
    assert report["verified"] is True, report
    assert report["passed_checks"] == report["total_checks"] == 7

    agg = evaluation["architecture_comparison"]["aggregate"]
    assert agg["transformer_muril_dapt_v3"]["mean_probe_macro_f1"] > agg["logistic_regression"]["mean_probe_macro_f1"]
    assert agg["transformer_muril_dapt_v3"]["mean_probe_macro_f1"] > agg["ann_mlp"]["mean_probe_macro_f1"]
    assert agg["transformer_muril_dapt_v3"]["mean_probe_macro_f1"] > agg["cnn_textcnn"]["mean_probe_macro_f1"]


def test_demo_store_loads_real_corpus_with_confidence_profiles() -> None:
    store = _data()
    assert store["metadata"]["synthetic"] is False
    assert len(store["posts"]) >= 300
    assert len(store["comments"]) >= 3000
    sample = store["comments"][0]
    assert "confidence_profile" in sample
    assert 0.0 <= sample["confidence_profile"]["overall_confidence"] <= 1.0
    assert sample["confidence_profile"]["confidence_tier"] in {"high", "medium", "low"}
