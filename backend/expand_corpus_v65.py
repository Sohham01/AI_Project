"""V6.5 Real Reddit Corpus Expansion (Reproducible, Leak-Free, Strictly Additive).

Expands the combined real research corpus beyond V6.1 (365 threads), V6.3
(451 threads), and V6.4 (526 threads) across IIM campus-specific discussions
(BLACKI, New/Baby IIMs, IPM/IPMAT, CAP/SAP rounds, waitlists, fees, hostels,
placements) and higher-education communities via authorized public Arctic Shift
archive snapshots (`ArcticShiftProvider`).

Strictly additive & leak-free:
  - Leaves `data/live_corpus/corpus.json` (v6.1),
    `data/corpus_v6_3/new_unannotated_corpus_v6_3.json` (v6.3), and
    `data/corpus_v6_4/new_iim_corpus_v6_4.json` (v6.4) 100% untouched.
  - Excludes any thread matching frozen V1 test `post_id`s, normalized titles,
    5-gram title shingles (Jaccard >= 0.72), or existing comment hashes.
  - Outputs strictly to `data/corpus_v6_5/` and
    `ml/artifacts/v6_5_experiments/corpus_expansion/`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.schemas.ingestion import CommentData
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import safe_failure_metadata
from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from backend.expand_iim_corpus_v64 import V63_CORPUS_PATH, _TRUSTED_IIM_SUBS, is_iim_relevant
from backend.expand_real_corpus_v63 import (
    _adaptive_sleep,
    _call_with_retry,
    _norm_text,
    _shingles,
    _text_sha256,
    check_thread_leakage_and_duplication,
    clean_and_redact,
    is_valid_comment_text,
    load_baseline_and_frozen_guards,
)

V64_CORPUS_PATH = ROOT / "data" / "corpus_v6_4" / "new_iim_corpus_v6_4.json"
STAGING_DIR = ROOT / "data" / "processed" / "arctic_shift_v6_5_staging"
OUT_CORPUS_DIR = ROOT / "data" / "corpus_v6_5"
OUT_AUDIT_DIR = ROOT / "ml" / "artifacts" / "v6_5_experiments" / "corpus_expansion"

V65_COLLECTION_PLANS: List[Dict[str, Any]] = [
    # Direct historical windows on diverse MBA/CAT/IIM communities (avoid slow full-text queries where 422 occurs)
    {"subreddit": "MBAIndia", "mode": "direct", "query": None, "limit": 100, "after": "2021-01-01", "before": "2023-06-01"},
    {"subreddit": "IIMCATPreparation", "mode": "direct", "query": None, "limit": 100, "after": "2021-01-01", "before": "2023-01-01"},
    {"subreddit": "CATPrep", "mode": "direct", "query": None, "limit": 100, "after": "2021-01-01", "before": "2024-06-01"},
    {"subreddit": "CATStudyRoom", "mode": "direct", "query": None, "limit": 100, "after": "2021-01-01", "before": "2024-06-01"},
    {"subreddit": "IIMRanchi", "mode": "direct", "query": None, "limit": 60, "after": "2020-01-01", "before": None},
    {"subreddit": "IIMIndore", "mode": "direct", "query": None, "limit": 60, "after": "2020-01-01", "before": None},
    {"subreddit": "IIMKozhikode", "mode": "direct", "query": None, "limit": 60, "after": "2020-01-01", "before": None},
    {"subreddit": "IIMBangalore", "mode": "direct", "query": None, "limit": 60, "after": "2020-01-01", "before": None},
    {"subreddit": "MBAIndia", "mode": "title", "query": "IIM", "limit": 80, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "title", "query": "waitlist", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "title", "query": "BLACKI", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "IPMATtards", "mode": "title", "query": "Ranchi", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "IPMATtards", "mode": "title", "query": "Bodh Gaya", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "IPMATtards", "mode": "title", "query": "Jammu", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "Indian_Academia", "mode": "title", "query": "IIM", "limit": 80, "after": "1825d", "before": None},
    {"subreddit": "Btechtards", "mode": "title", "query": "IIM", "limit": 70, "after": "1460d", "before": None},
    # Specific IIM campuses on r/CATpreparation using fast title search
    {"subreddit": "CATpreparation", "mode": "title", "query": "IIM Bangalore", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "CATpreparation", "mode": "title", "query": "IIM Ranchi", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "CATpreparation", "mode": "title", "query": "IIM Raipur", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "CATpreparation", "mode": "title", "query": "IIM Rohtak", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "CATpreparation", "mode": "title", "query": "CAP round", "limit": 70, "after": "1460d", "before": None},
    {"subreddit": "CATpreparation", "mode": "title", "query": "IIM Bodh Gaya", "limit": 60, "after": "1460d", "before": None},
]


def _ingest_prior_corpus(path: Path, guards: Dict[str, Any], prefix: str) -> None:
    posts_cnt = 0
    cmts_cnt = 0
    subs: Counter = Counter()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for p in payload.get("posts", []):
            posts_cnt += 1
            sub = str(p.get("subreddit") or "")
            subs[sub] += 1
            pid = str(p.get("reddit_id") or "")
            t_norm = _norm_text(p.get("title"))
            b_norm = _norm_text(p.get("body"))
            guards["existing_title_hashes"].add(_text_sha256(f"{t_norm}|{b_norm}"))
            if t_norm:
                guards["existing_title120"].add(t_norm[:120])
                sh = _shingles(t_norm, 5)
                if sh:
                    guards["existing_title_shingles"].append((pid, sh))
            for c in p.get("comments", []):
                cmts_cnt += 1
                guards["existing_comment_hashes"].add(_text_sha256(str(c.get("body") or "")))
    guards[f"{prefix}_posts"] = posts_cnt
    guards[f"{prefix}_comments"] = cmts_cnt
    guards[f"{prefix}_subreddits"] = subs


def load_v65_guards() -> Dict[str, Any]:
    """Extend V6.1 + frozen test guards with V6.3 and V6.4 threads so V6.5 stays strictly additive."""
    guards = load_baseline_and_frozen_guards()
    _ingest_prior_corpus(V63_CORPUS_PATH, guards, "v63")
    _ingest_prior_corpus(V64_CORPUS_PATH, guards, "v64")
    return guards


def expand_corpus_v65(
    *,
    target_new_threads: int = 580,
    max_comments_per_post: int = 80,
    request_delay: float = 1.15,
    fetch_live: bool = True,
) -> Dict[str, Any]:
    """Collect net-new IIM/higher-ed threads into isolated `data/corpus_v6_5/`."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    guards = load_v65_guards()
    provider = ArcticShiftProvider() if fetch_live else None

    staged_posts_path = STAGING_DIR / "staged_posts_with_comments.jsonl"
    raw_cache_by_id: Dict[str, Dict[str, Any]] = {}
    if staged_posts_path.is_file():
        for line in staged_posts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                raw_cache_by_id[rec["source_id_sha"]] = rec

    search_log: List[Dict[str, Any]] = []
    search_candidates: Dict[str, Dict[str, Any]] = {}
    seen_title120_stage: Set[str] = {
        _norm_text(r.get("title", ""))[:120] for r in raw_cache_by_id.values() if r.get("title")
    }

    if fetch_live and provider is not None:
        print(
            f"[V6.5] Searching Arctic Shift across {len(V65_COLLECTION_PLANS)} IIM/MBA plans (existing staged: {len(raw_cache_by_id)})...",
            flush=True,
        )
        for plan_idx, plan in enumerate(V65_COLLECTION_PLANS, start=1):
            if len(raw_cache_by_id) + len(search_candidates) >= target_new_threads + 260:
                break
            sub = plan["subreddit"]
            query = plan["query"]
            mode = plan["mode"]
            use_title = mode == "title"
            try:
                def _search_fn():
                    kw: Dict[str, Any] = {
                        "subreddit": sub,
                        "limit": plan["limit"],
                        "after": plan["after"],
                        "sort": "desc",
                        "timeout": 25.0,
                    }
                    if plan["before"]:
                        kw["before"] = plan["before"]
                    if query:
                        kw["title" if use_title else "query"] = query
                    return provider.search_posts(**kw)

                try:
                    posts = _call_with_retry(provider, _search_fn, max_retries=1, base_delay=request_delay)
                except Exception as first_exc:
                    _, first_code = safe_failure_metadata(first_exc)
                    if query and not use_title and first_code != 429:
                        use_title = True
                        posts = _call_with_retry(provider, _search_fn, max_retries=1, base_delay=request_delay)
                    else:
                        raise

                added = 0
                for p in posts:
                    if int(p.num_comments or 0) < 2:
                        continue
                    src_sha = hashlib.sha256(f"t3_{p.reddit_id}".encode("utf-8")).hexdigest()[:24]
                    if src_sha in raw_cache_by_id or src_sha in search_candidates:
                        continue
                    t_norm120 = _norm_text(p.title)[:120]
                    if t_norm120 in guards["existing_title120"] or t_norm120 in seen_title120_stage:
                        continue
                    seen_title120_stage.add(t_norm120)
                    search_candidates[src_sha] = {
                        "source_id_sha": src_sha,
                        "raw_id_temp": p.reddit_id,
                        "subreddit": p.subreddit,
                        "title": p.title,
                        "body": p.body or "",
                        "created_at": p.created_at.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
                        if p.created_at.tzinfo is None
                        else p.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "score": int(p.score or 0),
                        "num_comments": int(p.num_comments or 0),
                        "query": query or "<direct>",
                    }
                    added += 1
                search_log.append({
                    "plan_idx": plan_idx,
                    "subreddit": sub,
                    "mode": mode,
                    "query": query or "<direct>",
                    "returned": len(posts),
                    "new_queued": added,
                    "status": "ok",
                })
                print(
                    f"  [{plan_idx}/{len(V65_COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) -> returned={len(posts)}, new={added}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                cat, code = safe_failure_metadata(exc)
                search_log.append({
                    "plan_idx": plan_idx,
                    "subreddit": sub,
                    "mode": mode,
                    "query": query or "<direct>",
                    "status": "failed",
                    "error": f"{cat}:{code}",
                })
                print(
                    f"  [{plan_idx}/{len(V65_COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) FAILED: {cat}:{code}",
                    flush=True,
                )
                if code == 429:
                    time.sleep(5.0)
            _adaptive_sleep(provider, request_delay, time.sleep)

        cand_list = list(search_candidates.values())
        cand_list.sort(
            key=lambda x: (
                0 if x["subreddit"].lower() in _TRUSTED_IIM_SUBS else 1,
                -min(x["num_comments"], 40),
                x["source_id_sha"],
            )
        )
        needed = max(0, target_new_threads - len(raw_cache_by_id))
        to_fetch = cand_list[: max(needed + 80, needed)]
        print(
            f"[V6.5] Fetching comment trees for {len(to_fetch)} candidate threads (staged={len(raw_cache_by_id)})...",
            flush=True,
        )

        pii_stage_counts: Counter = Counter()
        with staged_posts_path.open("a", encoding="utf-8") as stage_f:
            for idx, item in enumerate(to_fetch, start=1):
                if len(raw_cache_by_id) >= target_new_threads + 45:
                    break
                pid_raw = item.pop("raw_id_temp")
                try:
                    cmts: List[CommentData] = _call_with_retry(
                        provider,
                        lambda: provider.get_comments(pid_raw, timeout=20.0, limit=max_comments_per_post),
                        max_retries=1,
                        base_delay=request_delay,
                    )
                except Exception as exc:  # noqa: BLE001
                    cat, code = safe_failure_metadata(exc)
                    search_log.append(
                        {"stage": "comments", "source_id_sha": item["source_id_sha"], "status": "failed", "error": f"{cat}:{code}"}
                    )
                    _adaptive_sleep(provider, request_delay, time.sleep)
                    continue

                cid_map = {
                    c.reddit_id: hashlib.sha256(f"t1_{c.reddit_id}".encode("utf-8")).hexdigest()[:24]
                    for c in cmts
                }
                norm_comments = []
                for c in cmts:
                    c_body = clean_and_redact(c.body, pii_stage_counts)
                    parent_sha = cid_map.get(c.parent_reddit_id) if c.parent_reddit_id else None
                    p_status = (
                        "top_level"
                        if c.parent_reddit_id is None
                        else ("available" if parent_sha is not None else "unavailable")
                    )
                    norm_comments.append({
                        "comment_id_sha": cid_map[c.reddit_id],
                        "parent_id_sha": parent_sha,
                        "parent_context_status": p_status,
                        "body": c_body,
                        "created_at": c.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "score": int(c.score or 0),
                    })

                staged_rec = {
                    **item,
                    "title": clean_and_redact(item["title"], pii_stage_counts),
                    "body": clean_and_redact(item["body"], pii_stage_counts),
                    "comments": norm_comments,
                }
                raw_cache_by_id[staged_rec["source_id_sha"]] = staged_rec
                stage_f.write(json.dumps(staged_rec, ensure_ascii=False) + "\n")
                if idx % 40 == 0 or idx == len(to_fetch):
                    print(f"  Fetched {idx}/{len(to_fetch)} threads (staged usable={len(raw_cache_by_id)})", flush=True)
                _adaptive_sleep(provider, request_delay, time.sleep)

    # Deterministic Filtering, Dedup, Leakage Check & Packaging
    dedup_stats: Counter = Counter()
    pii_total: Counter = Counter()
    seen_new_title_hashes: Set[str] = set()
    seen_new_title120: Set[str] = set()
    seen_new_comment_hashes: Set[str] = set()

    new_posts_out: List[Dict[str, Any]] = []
    excluded_leakage_log: List[Dict[str, Any]] = []
    lang_counter: Counter = Counter()
    script_counter: Counter = Counter()
    sub_counter: Counter = Counter()
    sub_comments_counter: Counter = Counter()

    sorted_staged = sorted(
        raw_cache_by_id.values(),
        key=lambda x: (x.get("created_at") or "", x["source_id_sha"]),
    )
    next_post_num = 1
    next_cmt_num = 1

    for item in sorted_staged:
        title = clean_and_redact(item.get("title", ""), pii_total)
        body = clean_and_redact(item.get("body", ""), pii_total)
        sub = str(item.get("subreddit") or "").strip()

        valid_cmts: List[Dict[str, Any]] = []
        for c in item.get("comments") or []:
            ok, reason = is_valid_comment_text(c.get("body", ""))
            if not ok:
                dedup_stats[f"comment_rejected:{reason}"] += 1
                continue
            cb_clean = clean_and_redact(c["body"], pii_total)
            ch = _text_sha256(cb_clean)
            if ch in guards["existing_comment_hashes"] or ch in seen_new_comment_hashes:
                dedup_stats["comment_rejected:duplicate_text"] += 1
                continue
            seen_new_comment_hashes.add(ch)
            valid_cmts.append({**c, "body": cb_clean})

        if not valid_cmts:
            dedup_stats["thread_rejected:zero_valid_comments"] += 1
            continue
        if not is_iim_relevant(sub, title, body, [c["body"] for c in valid_cmts]):
            dedup_stats["thread_rejected:off_topic_scope"] += 1
            continue

        rej_reason, dapt_excluded = check_thread_leakage_and_duplication(
            title=title,
            body=body,
            comment_bodies=[c["body"] for c in valid_cmts],
            guards=guards,
            seen_new_title_hashes=seen_new_title_hashes,
            seen_new_title120=seen_new_title120,
        )
        if rej_reason:
            dedup_stats[f"thread_rejected:{rej_reason}"] += 1
            continue

        seen_new_title_hashes.add(_text_sha256(f"{_norm_text(title)}|{_norm_text(body)}"))
        seen_new_title120.add(_norm_text(title)[:120])

        post_id = f"post_v65_{next_post_num:04d}"
        next_post_num += 1
        if dapt_excluded:
            excluded_leakage_log.append({
                "post_id": post_id,
                "subreddit": sub,
                "reason": "matched_frozen_test_title_or_comment_similarity",
            })

        sha_to_new_cid: Dict[str, str] = {}
        assigned_cmts: List[Dict[str, Any]] = []
        ordered_cmts = sorted(valid_cmts, key=lambda z: (z.get("created_at") or "", z["comment_id_sha"]))
        for c in ordered_cmts:
            sha_to_new_cid[c["comment_id_sha"]] = f"cmt_v65_{next_cmt_num:05d}"
            next_cmt_num += 1

        for c in ordered_cmts:
            new_cid = sha_to_new_cid[c["comment_id_sha"]]
            p_sha = c.get("parent_id_sha")
            parent_cid = sha_to_new_cid.get(p_sha) if p_sha else None
            p_status = "top_level" if not p_sha else ("available" if parent_cid else "unavailable")
            lid = detect_indian_reddit_language(c["body"])
            lang_counter[lid["language"]] += 1
            script_counter[lid["script"]] += 1
            sub_comments_counter[sub] += 1
            assigned_cmts.append({
                "reddit_id": new_cid,
                "parent_reddit_id": parent_cid,
                "parent_context_status": p_status,
                "body": c["body"],
                "created_at": c.get("created_at") or "",
                "score": int(c.get("score") or 0),
                "language": lid["language"],
                "script": lid["script"],
                "is_code_mixed": bool(lid["is_code_mixed"]),
                "has_multi_language_spans": bool(lid.get("has_multi_language_spans", False)),
            })

        sub_counter[sub] += 1
        new_posts_out.append({
            "reddit_id": post_id,
            "subreddit": sub,
            "title": title,
            "body": body,
            "created_at": item.get("created_at") or "",
            "score": int(item.get("score") or 0),
            "num_comments": len(assigned_cmts),
            "annotation_status": "unannotated",
            "dapt_and_train_excluded": bool(dapt_excluded),
            "provenance": {
                "source_provider": "arctic_shift_public_archive",
                "corpus_version": "6.5.0-iim-expansion",
                "search_query": item.get("query", "<direct>"),
                "source_id_sha": item["source_id_sha"],
            },
            "comments": assigned_cmts,
        })

    new_comments_count = sum(len(p["comments"]) for p in new_posts_out)
    dates_all = [p["created_at"] for p in new_posts_out if p.get("created_at")]

    v65_payload = {
        "metadata": {
            "dataset_id": "rci-real-reddit-expansion-v6.5",
            "corpus_version": "6.5.0",
            "annotation_status": "unannotated",
            "synthetic_only": False,
            "source_provider": "arctic_shift_public_archive",
            "prior_corpus_reference": (
                "data/live_corpus/corpus.json (365) + "
                "data/corpus_v6_3/new_unannotated_corpus_v6_3.json (451) + "
                "data/corpus_v6_4/new_iim_corpus_v6_4.json (526)"
            ),
            "total_new_threads": len(new_posts_out),
            "total_new_comments": new_comments_count,
            "dapt_and_train_excluded_threads": len(excluded_leakage_log),
            "date_range_utc": {"min": min(dates_all) if dates_all else "", "max": max(dates_all) if dates_all else ""},
        },
        "posts": new_posts_out,
    }
    (OUT_CORPUS_DIR / "new_corpus_v6_5.json").write_text(
        json.dumps(v65_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    combined_threads = guards["baseline_posts"] + guards["v63_posts"] + guards["v64_posts"] + len(new_posts_out)
    combined_comments = guards["baseline_comments"] + guards["v63_comments"] + guards["v64_comments"] + new_comments_count
    combined_subs = guards["baseline_subreddits"] + guards["v63_subreddits"] + guards["v64_subreddits"] + sub_counter

    prior_threads = guards["baseline_posts"] + guards["v63_posts"] + guards["v64_posts"]
    audit = {
        "corpus_version": "6.5.0",
        "before_vs_after": {
            "baseline_v6_1": {"threads": guards["baseline_posts"], "comments": guards["baseline_comments"]},
            "v6_3_expansion": {"threads": guards["v63_posts"], "comments": guards["v63_comments"]},
            "v6_4_iim_expansion": {"threads": guards["v64_posts"], "comments": guards["v64_comments"]},
            "v6_5_expansion": {"threads": len(new_posts_out), "comments": new_comments_count},
            "combined_v6_5_total": {"threads": combined_threads, "comments": combined_comments},
            "net_growth_over_v6_4_pct": round(100.0 * len(new_posts_out) / max(prior_threads, 1), 2),
        },
        "new_v6_5_subreddits": {
            sub: {
                "new_threads": cnt,
                "new_comments": sub_comments_counter.get(sub, 0),
                "combined_threads": combined_subs.get(sub, cnt),
                "combined_share_pct": round(100.0 * combined_subs.get(sub, cnt) / max(combined_threads, 1), 2),
            }
            for sub, cnt in sub_counter.most_common()
        },
        "combined_top_subreddits": {
            sub: {"threads": cnt, "share_pct": round(100.0 * cnt / max(combined_threads, 1), 2)}
            for sub, cnt in combined_subs.most_common()
        },
        "language_distribution_new_comments": dict(lang_counter.most_common()),
        "script_distribution_new_comments": dict(script_counter.most_common()),
        "date_range_utc": v65_payload["metadata"]["date_range_utc"],
        "deduplication_and_filtering": dict(sorted(dedup_stats.items())),
        "pii_redactions": dict(sorted(pii_total.items())),
        "leakage_audit": {
            "frozen_test_threads_checked": len(guards["test_pids"]),
            "frozen_test_comments_checked": len(guards["test_comment_hashes"]),
            "threads_quarantined_from_dapt_train": len(excluded_leakage_log),
            "quarantined_details": excluded_leakage_log,
            "frozen_test_leakage_into_train_pool": 0,
        },
        "search_log": search_log,
    }
    (OUT_CORPUS_DIR / "combined_corpus_v6_5_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_AUDIT_DIR / "v6_5_expansion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-new-threads", type=int, default=580)
    parser.add_argument("--max-comments-per-post", type=int, default=80)
    parser.add_argument("--delay", type=float, default=1.15)
    parser.add_argument("--offline-reprocess", action="store_true")
    args = parser.parse_args()

    rep = expand_corpus_v65(
        target_new_threads=args.target_new_threads,
        max_comments_per_post=args.max_comments_per_post,
        request_delay=args.delay,
        fetch_live=not args.offline_reprocess,
    )
    print(json.dumps(rep["before_vs_after"], indent=2))
    print("New V6.5 subreddits:", list(rep["new_v6_5_subreddits"].items())[:12])
    print("Leakage quarantined:", rep["leakage_audit"]["threads_quarantined_from_dapt_train"])


if __name__ == "__main__":
    main()
