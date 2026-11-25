"""V6.4 Tests — IIM Corpus Expansion Scope Guard, Deduplication & Zero-PII Invariants."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.expand_iim_corpus_v64 import is_iim_relevant, load_v64_guards


def test_iim_relevance_and_v64_additive_guards() -> None:
    assert is_iim_relevant("IIMKashipur", "M23 looking for hookup, ons, fwb", "", []) is False
    assert is_iim_relevant("Btechtards", "Which coding laptop to buy for DSA?", "", []) is False
    assert is_iim_relevant("IIMCalcutta", "Doubt regarding hostel fees and SIP placement", "", []) is True
    assert is_iim_relevant("IPMATtards", "IIM Indore IPMAT interview tips and waitlist movement", "", []) is True

    guards = load_v64_guards()
    assert guards["baseline_posts"] == 365
    assert len(guards["test_pids"]) == 37


def test_v64_expanded_iim_corpus_zero_pii_and_no_prior_overlap_when_present() -> None:
    p_v64 = ROOT / "data" / "corpus_v6_4" / "new_iim_corpus_v6_4.json"
    if not p_v64.is_file():
        return
    payload = json.loads(p_v64.read_text(encoding="utf-8"))
    assert payload["metadata"]["annotation_status"] == "unannotated"
    assert payload["metadata"]["synthetic_only"] is False

    base_titles = {
        re.sub(r"\s+", " ", str(p.get("title") or "")).strip().lower()[:120]
        for src in (ROOT / "data" / "live_corpus" / "corpus.json", ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json")
        if src.is_file()
        for p in json.loads(src.read_text(encoding="utf-8")).get("posts", [])
    }

    user_re = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,24}\b")
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    url_re = re.compile(r"https?://(?:www\.|old\.)?reddit\.com/r/", re.I)

    for post in payload["posts"]:
        assert post["annotation_status"] == "unannotated"
        assert "author" not in post
        t120 = re.sub(r"\s+", " ", str(post.get("title") or "")).strip().lower()[:120]
        assert t120 not in base_titles, "V6.4 must contain zero duplicate titles from V6.1 or V6.3"
        blob = f"{post.get('title', '')} {post.get('body', '')}"
        assert not user_re.search(blob) and not email_re.search(blob) and not url_re.search(blob)
        for c in post["comments"]:
            assert "author" not in c
            cb = c.get("body", "")
            assert not user_re.search(cb) and not email_re.search(cb) and not url_re.search(cb)
