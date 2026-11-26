"""V6.5 Tests — Real Corpus Expansion, 6:1 Thread-Grouped Split & +185 Blind Annotation Invariants."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.expand_corpus_v65 import load_v65_guards
from ml.evaluation.annotation_reliability import verify_blind_sheet


def test_v65_additive_guards_include_v61_v63_v64() -> None:
    guards = load_v65_guards()
    assert guards["baseline_posts"] == 365
    assert guards["v63_posts"] == 451
    assert guards["v64_posts"] == 526
    assert len(guards["test_pids"]) == 37


def test_v65_corpus_zero_pii_and_no_prior_overlap_when_present() -> None:
    p_v65 = ROOT / "data" / "corpus_v6_5" / "new_corpus_v6_5.json"
    if not p_v65.is_file():
        return
    payload = json.loads(p_v65.read_text(encoding="utf-8"))
    assert payload["metadata"]["annotation_status"] == "unannotated"
    assert payload["metadata"]["synthetic_only"] is False

    prior_titles = {
        re.sub(r"\s+", " ", str(p.get("title") or "")).strip().lower()[:120]
        for src in (
            ROOT / "data" / "live_corpus" / "corpus.json",
            ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json",
            ROOT / "data" / "corpus_v6_4" / "new_iim_corpus_v6_4.json",
        )
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
        assert t120 not in prior_titles, "V6.5 must contain zero duplicate titles from V6.1, V6.3, or V6.4"
        blob = f"{post.get('title', '')} {post.get('body', '')}"
        assert not user_re.search(blob) and not email_re.search(blob) and not url_re.search(blob)
        for c in post["comments"]:
            assert "author" not in c
            cb = c.get("body", "")
            assert not user_re.search(cb) and not email_re.search(cb) and not url_re.search(cb)


def test_v65_6_1_split_and_blind_185_package_leak_free_when_present() -> None:
    split_path = ROOT / "data" / "corpus_v6_5" / "corpus_split_6_1_manifest.json"
    pkg_dir = ROOT / "data" / "annotations" / "v6_5_package" / "blind_185_to_6_1"
    if not split_path.is_file() or not pkg_dir.is_dir():
        return

    spl = json.loads(split_path.read_text(encoding="utf-8"))
    tr_ids = set(spl["train_post_ids"])
    te_ids = set(spl["test_post_ids"])
    assert len(tr_ids & te_ids) == 0
    assert spl["leakage_checks"]["thread_overlap_count"] == 0
    assert spl["leakage_checks"]["all_37_frozen_v1_test_threads_in_test_pool"] is True
    assert spl["leakage_checks"]["all_163_v1_trainval_threads_in_train_pool"] is True
    ratio = spl["observed_ratios"]["thread_train_to_test"]
    assert 5.9 <= ratio <= 6.1, f"Expected ~6:1 thread split ratio, got {ratio}"

    manifest = json.loads((pkg_dir / "blind_185_manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotation_status"] == "pending_human_annotation"
    assert manifest["synthetic_labels_generated"] is False
    assert manifest["n_new_items"] == 185
    assert abs(manifest["resulting_train_to_frozen_test_ratio"] - 6.0) < 1e-6
    assert manifest["frozen_test_overlap_count"] == 0
    assert manifest["test_pool_overlap_count"] == 0

    with (pkg_dir / "blind_185_v6_5_annotator_1.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 185
    ok, errs = verify_blind_sheet(rows)
    assert ok, f"Blind 185 CSV failed verification: {errs}"
    assert not any(r["post_id"] in te_ids for r in rows)
