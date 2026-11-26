"""V6.5 6:1 Thread-Grouped Corpus Split & +185 Blind Annotation Package Builder.

1. Builds a deterministic, strictly leak-free **6:1 train : test thread partition**
   (`~85.71%` train_pool / `~14.29%` test_pool) across all combined real Reddit
   threads (`v6.1 + v6.3 + v6.4 + v6.5`), pinning all 163 V1 trainval threads into
   `train_pool` and all 37 frozen V1 test threads (plus any DAPT-excluded thread)
   into `test_pool`. Outputs `data/corpus_v6_5/corpus_split_6_1_manifest.json`.

2. Samples **185 net-new unannotated real Reddit comments** strictly from `train_pool`
   threads (zero overlap with V1, V2, or any `test_pool` / frozen thread) to create
   a blind double-annotation package (`367 V1 trainval + 185 = 552` vs `92 frozen test`
   -> exact **6.00 : 1** labeled train:test target once human-annotated).
   Zero synthetic or machine labels are fabricated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from app.services.nlp.preprocessing.multilingual_preprocessor import MultilingualTextPreprocessor
from ml.evaluation.annotation_reliability import (
    GUIDELINE_VERSION,
    TAXONOMY_VERSION,
    verify_blind_sheet,
)
from ml.experiments.v5_optimization.common import load_v1_records_and_frozen_splits

CORPUS_FILES = [
    ("v6_1", ROOT / "data" / "live_corpus" / "corpus.json"),
    ("v6_3", ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"),
    ("v6_4", ROOT / "data" / "corpus_v6_4" / "new_iim_corpus_v6_4.json"),
    ("v6_5", ROOT / "data" / "corpus_v6_5" / "new_corpus_v6_5.json"),
]
V2_MASTER_CSV = ROOT / "data" / "annotations" / "real_reddit_labeled_v2" / "master_labeled_v2.csv"
TEMPLATE_XLSX = ROOT / "data" / "annotations" / "templates" / "RCI_Manual_Annotation_Template.xlsx"
SPLIT_MANIFEST_OUT = ROOT / "data" / "corpus_v6_5" / "corpus_split_6_1_manifest.json"
PKG_DIR = ROOT / "data" / "annotations" / "v6_5_package" / "blind_185_to_6_1"


def _norm_text(s: Any) -> str:
    return " ".join(str(s or "").split()).strip().lower()


def _text_sha(s: Any) -> str:
    return hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()


def _export_blind_xlsx(rows: List[Dict[str, Any]], out_xlsx: Path) -> None:
    if not TEMPLATE_XLSX.is_file():
        return
    try:
        import openpyxl
    except ImportError:
        return

    wb = openpyxl.load_workbook(TEMPLATE_XLSX)
    ws = wb["02_Annotation_Sheet"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col_map = {h: idx for idx, h in enumerate(headers, start=1) if h}
    for r_i, item in enumerate(rows, start=2):
        for k, v in item.items():
            if k in col_map:
                ws.cell(row=r_i, column=col_map[k], value=v)
    wb.save(out_xlsx)


def build_6_1_corpus_split_and_package(target_blind_n: int = 185) -> Dict[str, Any]:
    """Build deterministic 6:1 thread split + 185-row blind human annotation package."""
    v1_data = load_v1_records_and_frozen_splits()
    v1_manifest = v1_data["manifest"]
    v1_trainval_pids: Set[str] = set(v1_manifest["trainval_corpus_post_ids"])
    v1_test_pids: Set[str] = set(v1_manifest["test_corpus_post_ids"])
    assert len(v1_test_pids) == 37
    assert not (v1_trainval_pids & v1_test_pids)

    v1_text_hashes: Set[str] = {_text_sha(r["text"]) for r in v1_data["all"]}
    v1_comment_ids: Set[str] = {str(r["corpus_comment_id"]) for r in v1_data["all"] if r.get("corpus_comment_id")}

    v2_text_hashes: Set[str] = set()
    v2_comment_ids: Set[str] = set()
    if V2_MASTER_CSV.is_file():
        with V2_MASTER_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                v2_text_hashes.add(_text_sha(row.get("text", "")))
                cid = (row.get("comment_id") or "").strip()
                if cid:
                    v2_comment_ids.add(cid)

    # Load all combined threads across V6.1..V6.5
    all_threads: List[Dict[str, Any]] = []
    seen_pids: Set[str] = set()
    for cohort, path in CORPUS_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required corpus artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for post in payload.get("posts", []):
            pid = str(post["reddit_id"])
            if pid in seen_pids:
                raise AssertionError(f"Duplicate thread ID across cohorts: {pid}")
            seen_pids.add(pid)
            all_threads.append({"cohort": cohort, "post": post})

    # Partition pinned vs assignable unannotated threads
    train_threads: List[Dict[str, Any]] = []
    test_threads: List[Dict[str, Any]] = []
    assignable: List[Dict[str, Any]] = []

    for rec in all_threads:
        p = rec["post"]
        pid = str(p["reddit_id"])
        if pid in v1_test_pids or bool(p.get("dapt_and_train_excluded", False)):
            test_threads.append(rec)
        elif pid in v1_trainval_pids:
            train_threads.append(rec)
        else:
            assignable.append(rec)

    total_n = len(all_threads)
    target_test_n = max(len(test_threads), round(total_n / 7.0))
    need_test = max(0, target_test_n - len(test_threads))

    # Stratified round-robin allocation across subreddits so 6:1 spans all communities cleanly
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in sorted(
        assignable,
        key=lambda x: (
            str(x["post"].get("subreddit") or ""),
            str(x["post"].get("created_at") or ""),
            str(x["post"]["reddit_id"]),
        ),
    ):
        by_sub[str(rec["post"].get("subreddit") or "unknown")].append(rec)

    total_assignable = len(assignable)
    test_frac = need_test / float(max(total_assignable, 1))
    picked_test_ids: Set[str] = set()

    # Select proportional deterministic 1/7th indices inside each subreddit
    for sub in sorted(by_sub.keys()):
        items = by_sub[sub]
        k = round(len(items) * test_frac)
        if k <= 0 and len(items) >= 7:
            k = 1
        if k > 0:
            step = len(items) / float(k)
            for j in range(k):
                idx = min(int(j * step + step * 0.5), len(items) - 1)
                picked_test_ids.add(str(items[idx]["post"]["reddit_id"]))

    # Top up or trim deterministically so total test_pool hits exact round(N / 7)
    ordered_assignable = [r for sub in sorted(by_sub.keys()) for r in by_sub[sub]]
    if len(picked_test_ids) < need_test:
        for rec in ordered_assignable:
            pid = str(rec["post"]["reddit_id"])
            if pid not in picked_test_ids:
                picked_test_ids.add(pid)
                if len(picked_test_ids) == need_test:
                    break
    elif len(picked_test_ids) > need_test:
        for rec in reversed(ordered_assignable):
            pid = str(rec["post"]["reddit_id"])
            if pid in picked_test_ids:
                picked_test_ids.remove(pid)
                if len(picked_test_ids) == need_test:
                    break

    for rec in ordered_assignable:
        pid = str(rec["post"]["reddit_id"])
        if pid in picked_test_ids:
            test_threads.append(rec)
        else:
            train_threads.append(rec)

    train_ids = sorted(str(r["post"]["reddit_id"]) for r in train_threads)
    test_ids = sorted(str(r["post"]["reddit_id"]) for r in test_threads)
    assert not (set(train_ids) & set(test_ids)), "6:1 split leaked threads across train_pool and test_pool!"
    assert v1_test_pids.issubset(set(test_ids)), "Frozen V1 test threads missing from test_pool!"
    assert v1_trainval_pids.issubset(set(train_ids)), "V1 trainval threads missing from train_pool!"

    train_comments = sum(len(r["post"].get("comments") or []) for r in train_threads)
    test_comments = sum(len(r["post"].get("comments") or []) for r in test_threads)
    by_cohort = Counter()
    for r in train_threads:
        by_cohort[f"{r['cohort']}_train_threads"] += 1
    for r in test_threads:
        by_cohort[f"{r['cohort']}_test_threads"] += 1

    split_manifest = {
        "split_protocol": "v6.5_thread_grouped_6_to_1",
        "target_ratio": "6:1 (85.71% train_pool : 14.29% test_pool)",
        "total_threads": total_n,
        "total_comments": train_comments + test_comments,
        "train_pool": {
            "threads": len(train_ids),
            "comments": train_comments,
            "share_threads_pct": round(100.0 * len(train_ids) / max(total_n, 1), 2),
            "share_comments_pct": round(100.0 * train_comments / max(train_comments + test_comments, 1), 2),
            "pinned_v1_trainval_threads": len(v1_trainval_pids),
        },
        "test_pool": {
            "threads": len(test_ids),
            "comments": test_comments,
            "share_threads_pct": round(100.0 * len(test_ids) / max(total_n, 1), 2),
            "share_comments_pct": round(100.0 * test_comments / max(train_comments + test_comments, 1), 2),
            "pinned_v1_frozen_test_threads": len(v1_test_pids),
        },
        "observed_ratios": {
            "thread_train_to_test": round(len(train_ids) / max(len(test_ids), 1), 4),
            "comment_train_to_test": round(train_comments / max(test_comments, 1), 4),
        },
        "by_cohort_counts": dict(sorted(by_cohort.items())),
        "leakage_checks": {
            "thread_overlap_count": 0,
            "all_37_frozen_v1_test_threads_in_test_pool": True,
            "all_163_v1_trainval_threads_in_train_pool": True,
        },
        "train_post_ids": train_ids,
        "test_post_ids": test_ids,
    }
    SPLIT_MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST_OUT.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Part B: Build +185 Blind Human Annotation Sheet (367 + 185 = 552 : 92 = 6:1)
    # ------------------------------------------------------------------
    prep = MultilingualTextPreprocessor()
    candidate_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_hashes: Set[str] = set(v1_text_hashes) | set(v2_text_hashes)

    for rec in train_threads:
        if rec["cohort"] == "v6_1":
            # Focus expansion annotations on v6.3 + v6.4 + v6.5 real threads outside V1/V2
            continue
        post = rec["post"]
        pid = str(post["reddit_id"])
        sub = str(post.get("subreddit") or "unknown")
        title = str(post.get("title") or "")
        comments = list(post.get("comments") or [])
        cmt_body_by_id = {str(c.get("reddit_id")): str(c.get("body") or "") for c in comments}

        # Filter substantive comments (>= 45 chars, <= 1,800 chars, not duplicate)
        for c in comments:
            cid = str(c.get("reddit_id") or "")
            body = str(c.get("body") or "").strip()
            if cid in v1_comment_ids or cid in v2_comment_ids:
                continue
            if len(body) < 45 or len(body) > 1800:
                continue
            bh = _text_sha(body)
            if bh in seen_hashes:
                continue
            seen_hashes.add(bh)
            lid = c.get("language") or detect_indian_reddit_language(body)["language"]
            p_cid = c.get("parent_reddit_id")
            p_txt = cmt_body_by_id.get(str(p_cid), "") if p_cid else ""
            p_status = "available" if (p_cid and p_txt) else ("unavailable" if p_cid else "top_level")
            lang_group = "indic" if (c.get("is_code_mixed") or lid in {"hi", "hi-en", "bn", "bn-en", "or", "or-en"}) else "en"
            bkey = f"{rec['cohort']}|{sub}|{lang_group}"
            candidate_buckets[bkey].append({
                "cohort": rec["cohort"],
                "post_id": pid,
                "comment_id": cid,
                "subreddit": sub,
                "post_title": title,
                "parent_comment_id": p_cid or "",
                "parent_text": p_txt,
                "parent_context_status": p_status,
                "created_at": str(c.get("created_at") or post.get("created_at") or ""),
                "score": int(c.get("score") or 0),
                "text": body,
            })

    for bkey in candidate_buckets:
        # Prefer high-engagement comments with parent context where available, deterministic tie-break
        candidate_buckets[bkey].sort(
            key=lambda x: (
                0 if x["parent_context_status"] == "available" else 1,
                -min(abs(x["score"]), 50),
                x["comment_id"],
            )
        )

    picked: List[Dict[str, Any]] = []
    per_thread_cnt: Counter = Counter()
    sorted_keys = sorted(candidate_buckets.keys())
    idx_pos = 0
    # Round-robin pass (max 2 comments per thread to maximize thread diversity)
    while len(picked) < target_blind_n:
        progressed = False
        for k in sorted_keys:
            bucket = candidate_buckets[k]
            while idx_pos < len(bucket):
                cand = bucket[idx_pos]
                if per_thread_cnt[cand["post_id"]] < 2:
                    picked.append(cand)
                    per_thread_cnt[cand["post_id"]] += 1
                    progressed = True
                    break
                # scan further inside this bucket on next step
                bucket.pop(idx_pos)
            if len(picked) >= target_blind_n:
                break
        if not progressed:
            break
        idx_pos += 1

    # Fallback relax per-thread cap to 3 if bucket exhaustion occurs
    if len(picked) < target_blind_n:
        picked_cids = {x["comment_id"] for x in picked}
        for k in sorted_keys:
            for cand in candidate_buckets[k]:
                if cand["comment_id"] in picked_cids:
                    continue
                if per_thread_cnt[cand["post_id"]] < 3:
                    picked.append(cand)
                    picked_cids.add(cand["comment_id"])
                    per_thread_cnt[cand["post_id"]] += 1
                    if len(picked) >= target_blind_n:
                        break
            if len(picked) >= target_blind_n:
                break

    assert len(picked) == target_blind_n, f"Expected {target_blind_n} blind rows, got {len(picked)}"
    # Verify ZERO overlap with test_pool or V1
    test_pid_set = set(test_ids)
    assert not any(x["post_id"] in test_pid_set or x["post_id"] in v1_test_pids for x in picked)

    fieldnames = [
        "document_id", "post_id", "comment_id", "subreddit", "post_title",
        "parent_comment_id", "parent_text", "parent_context_status", "created_at", "score",
        "original_text", "normalized_text", "language", "script", "is_code_mixed",
        "detected_languages", "sentiment", "emotions", "issues", "response_type",
        "is_solution", "review_status", "abstain_reason", "annotator_id", "annotator_notes",
    ]
    blind_rows: List[Dict[str, Any]] = []
    strat_sub = Counter()
    strat_cohort = Counter()
    strat_lang = Counter()
    for i, item in enumerate(picked, start=1):
        lid = detect_indian_reddit_language(item["text"])
        pre = prep.preprocess(item["text"])
        strat_sub[item["subreddit"]] += 1
        strat_cohort[item["cohort"]] += 1
        strat_lang[lid["language"]] += 1
        blind_rows.append({
            "document_id": f"v65-b185-{i:03d}-{item['comment_id']}",
            "post_id": item["post_id"],
            "comment_id": item["comment_id"],
            "subreddit": item["subreddit"],
            "post_title": item["post_title"],
            "parent_comment_id": item["parent_comment_id"],
            "parent_text": item["parent_text"],
            "parent_context_status": item["parent_context_status"],
            "created_at": item["created_at"],
            "score": item["score"],
            "original_text": item["text"],
            "normalized_text": pre.normalized_text,
            "language": lid["language"],
            "script": lid["script"],
            "is_code_mixed": "true" if lid["is_code_mixed"] else "false",
            "detected_languages": "|".join(lid.get("detected_languages") or [lid["language"]]),
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        })

    ok_blind, blind_errs = verify_blind_sheet(blind_rows)
    assert ok_blind, f"V6.5 blind 185 sheet failed blind verification: {blind_errs}"

    PKG_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("blind_185_v6_5_annotator_1.csv", "blind_185_v6_5_annotator_2.csv"):
        with (PKG_DIR / fname).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(blind_rows)

    _export_blind_xlsx(blind_rows, PKG_DIR / "blind_185_v6_5_annotator_1.xlsx")
    _export_blind_xlsx(blind_rows, PKG_DIR / "blind_185_v6_5_annotator_2.xlsx")

    adj_fields = [
        "document_id", "post_id", "post_title", "parent_text", "original_text",
        "annotator_1_labels", "annotator_2_labels",
        "adjudicated_status", "adjudicated_sentiment", "adjudicated_emotions",
        "adjudicated_issues", "adjudicated_response", "adjudicated_solution",
        "adjudicator_id", "adjudication_notes",
    ]
    with (PKG_DIR / "blind_185_adjudication_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=adj_fields).writeheader()

    pkg_manifest = {
        "package_id": "v6_5_blind_185_to_6_1",
        "annotation_status": "pending_human_annotation",
        "synthetic_labels_generated": False,
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": GUIDELINE_VERSION,
        "n_new_items": len(blind_rows),
        "unique_threads": len(per_thread_cnt),
        "existing_v1_trainval": len(v1_data["trainval"]),
        "existing_v1_frozen_test": len(v1_data["test"]),
        "resulting_labeled_train_target": len(v1_data["trainval"]) + len(blind_rows),
        "resulting_train_to_frozen_test_ratio": round(
            (len(v1_data["trainval"]) + len(blind_rows)) / max(len(v1_data["test"]), 1), 4
        ),
        "frozen_test_overlap_count": 0,
        "test_pool_overlap_count": 0,
        "blind_verified": True,
        "sampling_summary": {
            "by_cohort": dict(strat_cohort.most_common()),
            "by_subreddit": dict(strat_sub.most_common()),
            "by_language": dict(strat_lang.most_common()),
        },
        "files": [
            "blind_185_v6_5_annotator_1.csv",
            "blind_185_v6_5_annotator_1.xlsx",
            "blind_185_v6_5_annotator_2.csv",
            "blind_185_v6_5_annotator_2.xlsx",
            "blind_185_adjudication_template.csv",
        ],
    }
    (PKG_DIR / "blind_185_manifest.json").write_text(json.dumps(pkg_manifest, indent=2), encoding="utf-8")

    return {"split_manifest": split_manifest, "annotation_package": pkg_manifest}


def main() -> None:
    res = build_6_1_corpus_split_and_package(target_blind_n=185)
    sm = res["split_manifest"]
    ap = res["annotation_package"]
    print(json.dumps({
        "corpus_6_1": {
            "total_threads": sm["total_threads"],
            "train_threads": sm["train_pool"]["threads"],
            "test_threads": sm["test_pool"]["threads"],
            "thread_ratio": sm["observed_ratios"]["thread_train_to_test"],
            "comment_ratio": sm["observed_ratios"]["comment_train_to_test"],
        },
        "blind_185_package": {
            "n_new_items": ap["n_new_items"],
            "unique_threads": ap["unique_threads"],
            "target_ratio": ap["resulting_train_to_frozen_test_ratio"],
            "frozen_test_overlap": ap["frozen_test_overlap_count"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
