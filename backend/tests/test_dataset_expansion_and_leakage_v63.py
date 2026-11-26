"""Sprint P1-A Tests — Corpus Expansion Schema, Dedup, Zero-PII & Frozen Test Leakage Guards."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.expand_real_corpus_v63 import (
    check_thread_leakage_and_duplication,
    is_valid_comment_text,
    load_baseline_and_frozen_guards,
)


def test_leakage_and_deduplication_guards_protect_frozen_test() -> None:
    guards = load_baseline_and_frozen_guards()
    assert guards["baseline_posts"] == 365
    assert len(guards["test_pids"]) == 37
    assert len(guards["test_comment_hashes"]) >= 50

    baseline = json.loads((ROOT / "data" / "live_corpus" / "corpus.json").read_text(encoding="utf-8"))
    # Exact baseline post title -> rejected as baseline duplicate
    sample_post = baseline["posts"][0]
    rej_dup, _ = check_thread_leakage_and_duplication(
        title=sample_post["title"],
        body=sample_post.get("body") or "",
        comment_bodies=[],
        guards=guards,
        seen_new_title_hashes=set(),
        seen_new_title120=set(),
    )
    assert rej_dup == "duplicate_of_baseline_corpus"

    # Unseen title containing a frozen test comment body -> accepted ONLY with dapt_and_train_excluded=True
    posts_by_id = {p["reddit_id"]: p for p in baseline["posts"]}
    long_test_cmt = next(
        c["body"]
        for pid in sorted(guards["test_pids"])
        for c in posts_by_id[pid]["comments"]
        if len(c["body"].split()) >= 8
    )

    rej_leak, excluded = check_thread_leakage_and_duplication(
        title="Brand new distinct thread title for leakage unit test",
        body="",
        comment_bodies=[long_test_cmt],
        guards=guards,
        seen_new_title_hashes=set(),
        seen_new_title120=set(),
    )
    assert rej_leak is None
    assert excluded is True


def test_spam_bot_and_deleted_comment_filtering() -> None:
    assert is_valid_comment_text("[deleted]")[0] is False
    assert is_valid_comment_text("RemindMe! 3 days")[0] is False
    assert is_valid_comment_text("![gif](giphy|12345)")[0] is False
    assert is_valid_comment_text("Check AICTE 15-day refund guidelines with admissions.")[0] is True


def test_v63_expanded_corpus_zero_pii_and_quarantine_invariants_when_present() -> None:
    new_corpus = ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"
    if not new_corpus.is_file():
        return
    payload = json.loads(new_corpus.read_text(encoding="utf-8"))
    assert payload["metadata"]["annotation_status"] == "unannotated"
    assert payload["metadata"]["synthetic_only"] is False

    user_re = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,24}\b")
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    url_re = re.compile(r"https?://(?:www\.|old\.)?reddit\.com/r/", re.I)

    for p in payload["posts"]:
        assert p["annotation_status"] == "unannotated"
        assert "author" not in p
        blob = f"{p.get('title', '')} {p.get('body', '')}"
        assert not user_re.search(blob)
        assert not email_re.search(blob)
        assert not url_re.search(blob)
        for c in p["comments"]:
            assert "author" not in c
            cb = c.get("body", "")
            assert not user_re.search(cb)
            assert not email_re.search(cb)
            assert not url_re.search(cb)
