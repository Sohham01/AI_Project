"""V6.4 IIM-Related Real Reddit Corpus Expansion (Reproducible, Leak-Free).

Expands the unannotated real research corpus beyond V6.1 (365 threads) and V6.3
(451 new threads -> 816 combined) specifically across dedicated IIM communities
(`r/IPMATtards`, `r/IIMCalcutta`, `r/IIMLucknow`, `r/IIMMumbai`, `r/IIMIndore`, etc.)
and IIM-targeted discussions (BLACKI, New/Baby IIMs, campus life, placements,
fees, waitlists, IPM, admissions) across `r/CATpreparation`, `r/MBAIndia`,
`r/IIMCATPreparation`, `r/CATPrep`, `r/CATStudyRoom`, `r/iim_CAT`,
`r/Indian_Academia`, and `r/Btechtards`.

Reuses all PII redaction, bot/GIF/spam filtering, and frozen-test leakage checks
directly from `backend/expand_real_corpus_v63.py`, keeping both V6.1 (`corpus.json`)
and V6.3 (`new_unannotated_corpus_v6_3.json`) strictly untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.schemas.ingestion import CommentData
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import safe_failure_metadata
from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from backend.expand_real_corpus_v63 import (
    BASELINE_CORPUS_PATH,
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

V63_CORPUS_PATH = ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"
STAGING_DIR = ROOT / "data" / "processed" / "arctic_shift_v6_4_staging"
OUT_CORPUS_DIR = ROOT / "data" / "corpus_v6_4"
OUT_AUDIT_DIR = ROOT / "ml" / "artifacts" / "v6_4_experiments" / "iim_expansion"

_TRUSTED_IIM_SUBS = {
    "ipmattards",
    "iimcalcutta",
    "iimlucknow",
    "iimmumbai",
    "iimindore",
    "iimbangalore",
    "iimkozhikode",
    "iimranchi",
    "iimrohtak",
    "iimsambalpur",
    "iimsambalpurug",
    "iimcatpreparation",
    "iim_cat",
}

_IIM_STRICT_RE = re.compile(
    r"\b(iim|iims|blacki|baby\s*iim|new\s*iim|ipm|ipmat|jipmat|cap\s*round|sap\s*round|"
    r"cat\s*percentile|waitlist|merit\s*list|shortlist|gdpi|wat\b|sip\b|placecom|"
    r"pgp\b|pgdm|mba|b-?school|bschool|ahmedabad|bangalore|calcutta|lucknow|kozhikode|"
    r"indore|shillong|udaipur|trichy|raipur|ranchi|rohtak|bodh\s*gaya|jammu|sirmaur|"
    r"visakhapatnam|vizag|nagpur|sambalpur|amritsar|hostel|mess|campus|fees?\b|loan)\b",
    re.IGNORECASE,
)

# Strictly require explicit IIM mention on broad hubs (like Btechtards / Indian_Academia)
_EXPLICIT_IIM_RE = re.compile(
    r"\b(iim|iims|iima|iimb|iimc|iiml|iimk|iimi|iimm|iims|blacki|baby\s*iim|new\s*iim|ipmat|jipmat)\b",
    re.IGNORECASE,
)

IIM_COLLECTION_PLANS: List[Dict[str, Any]] = [
    # Dedicated IIM communities (live-probed on Arctic Shift)
    {"subreddit": "IPMATtards", "mode": "direct", "query": None, "limit": 100, "after": "1095d", "before": None},
    {"subreddit": "IPMATtards", "mode": "direct", "query": None, "limit": 100, "after": "2023-01-01", "before": "2025-07-01"},
    {"subreddit": "IPMATtards", "mode": "context", "query": "IIM Indore", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "IPMATtards", "mode": "context", "query": "hostel", "limit": 50, "after": "1460d", "before": None},
    {"subreddit": "IPMATtards", "mode": "context", "query": "fees", "limit": 50, "after": "1460d", "before": None},
    {"subreddit": "IIMCalcutta", "mode": "direct", "query": None, "limit": 80, "after": "1460d", "before": None},
    {"subreddit": "IIMLucknow", "mode": "direct", "query": None, "limit": 80, "after": "1460d", "before": None},
    {"subreddit": "IIMMumbai", "mode": "direct", "query": None, "limit": 50, "after": "1460d", "before": None},
    {"subreddit": "IIMIndore", "mode": "direct", "query": None, "limit": 50, "after": "1460d", "before": None},
    {"subreddit": "IIMRanchi", "mode": "direct", "query": None, "limit": 50, "after": "1460d", "before": None},
    # Institute-specific & issue-focused IIM discussions across MBA/CAT forums
    {"subreddit": "CATpreparation", "mode": "context", "query": "BLACKI", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "Baby IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "New IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Ahmedabad", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Bangalore", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Calcutta", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Udaipur", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Trichy", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "IIM Shillong", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "BLACKI", "limit": 80, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "Baby IIM", "limit": 80, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "IIM Shillong", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "IIM Udaipur", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "IIM Mumbai", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "IIM Kozhikode", "limit": 60, "after": "1460d", "before": None},
    {"subreddit": "IIMCATPreparation", "mode": "direct", "query": None, "limit": 100, "after": "2023-01-01", "before": "2025-10-01"},
    {"subreddit": "IIMCATPreparation", "mode": "context", "query": "IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "CATPrep", "mode": "context", "query": "IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "CATStudyRoom", "mode": "context", "query": "IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "iim_CAT", "mode": "direct", "query": None, "limit": 80, "after": "2023-01-01", "before": "2025-09-01"},
    {"subreddit": "Indian_Academia", "mode": "context", "query": "IIM", "limit": 80, "after": "2022-01-01", "before": "2025-07-01"},
    {"subreddit": "Btechtards", "mode": "context", "query": "IIM", "limit": 60, "after": "1095d", "before": None},
]


def is_iim_relevant(subreddit: str, title: str, body: str, comments: Sequence[str]) -> bool:
    """Enforce IIM + higher-education topical relevance without allowing off-topic hookup/spam posts."""
    sub_l = subreddit.lower()
    blob = f"{title} {body[:600]} {' '.join(comments[:4])}"
    if sub_l in {"btechtards", "indian_academia", "iimkashipur"}:
        return bool(_EXPLICIT_IIM_RE.search(blob)) and bool(_IIM_STRICT_RE.search(blob))
    if sub_l in _TRUSTED_IIM_SUBS:
        return len(_norm_text(title)) >= 6 and bool(_IIM_STRICT_RE.search(blob))
    return bool(_EXPLICIT_IIM_RE.search(blob))


def load_v64_guards() -> Dict[str, Any]:
    """Extend V6.1 + frozen test guards with all existing V6.3 threads so V6.4 stays strictly additive."""
    guards = load_baseline_and_frozen_guards()
    v63_posts = 0
    v63_cmts = 0
    v63_subs: Counter = Counter()

    if V63_CORPUS_PATH.is_file():
        v63 = json.loads(V63_CORPUS_PATH.read_text(encoding="utf-8"))
        for p in v63.get("posts", []):
            v63_posts += 1
            sub = str(p.get("subreddit") or "")
            v63_subs[sub] += 1
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
                v63_cmts += 1
                guards["existing_comment_hashes"].add(_text_sha256(str(c.get("body") or "")))

    guards["v63_posts"] = v63_posts
    guards["v63_comments"] = v63_cmts
    guards["v63_subreddits"] = v63_subs
    return guards


def expand_iim_corpus(
    *,
    target_new_threads: int = 550,
    max_comments_per_post: int = 80,
    request_delay: float = 0.8,
    fetch_live: bool = True,
) -> Dict[str, Any]:
    """Collect net-new IIM-related threads into isolated `data/corpus_v6_4/`, with zero frozen-test leakage."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    guards = load_v64_guards()
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
            f"[V6.4-IIM] Searching IIM communities (existing staged: {len(raw_cache_by_id)})...",
            flush=True,
        )
        for plan_idx, plan in enumerate(IIM_COLLECTION_PLANS, start=1):
            if len(raw_cache_by_id) + len(search_candidates) >= target_new_threads + 220:
                break
            sub = plan["subreddit"]
            query = plan["query"]
            mode = plan["mode"]
            use_title = False
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
                except Exception:
                    if query and not use_title:
                        use_title = True
                        posts = _call_with_retry(provider, _search_fn, max_retries=1, base_delay=request_delay)
                    else:
                        raise

                added = 0
                for p in posts:
                    if int(p.num_comments or 0) < 1:
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
                    f"  [{plan_idx}/{len(IIM_COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) -> returned={len(posts)}, new={added}",
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
                    f"  [{plan_idx}/{len(IIM_COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) FAILED: {cat}:{code}",
                    flush=True,
                )
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
        to_fetch = cand_list[: max(needed + 60, needed)]
        print(
            f"[V6.4-IIM] Fetching comment trees for {len(to_fetch)} candidate threads (staged={len(raw_cache_by_id)})...",
            flush=True,
        )

        pii_stage_counts: Counter = Counter()
        with staged_posts_path.open("a", encoding="utf-8") as stage_f:
            for idx, item in enumerate(to_fetch, start=1):
                if len(raw_cache_by_id) >= target_new_threads + 40:
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
                if idx % 30 == 0 or idx == len(to_fetch):
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

        post_id = f"post_v64_{next_post_num:04d}"
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
            sha_to_new_cid[c["comment_id_sha"]] = f"cmt_v64_{next_cmt_num:05d}"
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
                "corpus_version": "6.4.0-iim-expansion",
                "search_query": item.get("query", "<direct>"),
                "source_id_sha": item["source_id_sha"],
            },
            "comments": assigned_cmts,
        })

    new_comments_count = sum(len(p["comments"]) for p in new_posts_out)
    dates_all = [p["created_at"] for p in new_posts_out if p.get("created_at")]

    v64_payload = {
        "metadata": {
            "dataset_id": "rci-real-reddit-iim-expansion-v6.4",
            "corpus_version": "6.4.0",
            "annotation_status": "unannotated",
            "synthetic_only": False,
            "source_provider": "arctic_shift_public_archive",
            "prior_corpus_reference": "data/live_corpus/corpus.json (365) + data/corpus_v6_3/new_unannotated_corpus_v6_3.json (451)",
            "total_new_threads": len(new_posts_out),
            "total_new_comments": new_comments_count,
            "dapt_and_train_excluded_threads": len(excluded_leakage_log),
            "date_range_utc": {"min": min(dates_all) if dates_all else "", "max": max(dates_all) if dates_all else ""},
        },
        "posts": new_posts_out,
    }
    (OUT_CORPUS_DIR / "new_iim_corpus_v6_4.json").write_text(
        json.dumps(v64_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    combined_threads = guards["baseline_posts"] + guards["v63_posts"] + len(new_posts_out)
    combined_comments = guards["baseline_comments"] + guards["v63_comments"] + new_comments_count
    combined_subs = guards["baseline_subreddits"] + guards["v63_subreddits"] + sub_counter

    audit = {
        "corpus_version": "6.4.0",
        "before_vs_after": {
            "baseline_v6_1": {"threads": guards["baseline_posts"], "comments": guards["baseline_comments"]},
            "v6_3_expansion": {"threads": guards["v63_posts"], "comments": guards["v63_comments"]},
            "v6_4_iim_expansion": {"threads": len(new_posts_out), "comments": new_comments_count},
            "combined_v6_4_total": {"threads": combined_threads, "comments": combined_comments},
            "net_growth_over_v6_3_pct": round(100.0 * len(new_posts_out) / max(guards["baseline_posts"] + guards["v63_posts"], 1), 2),
        },
        "new_v6_4_subreddits": {
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
        "date_range_utc": v64_payload["metadata"]["date_range_utc"],
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
    (OUT_CORPUS_DIR / "combined_corpus_v6_4_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_AUDIT_DIR / "iim_expansion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-new-threads", type=int, default=550)
    parser.add_argument("--max-comments-per-post", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--offline-reprocess", action="store_true")
    args = parser.parse_args()

    rep = expand_iim_corpus(
        target_new_threads=args.target_new_threads,
        max_comments_per_post=args.max_comments_per_post,
        request_delay=args.delay,
        fetch_live=not args.offline_reprocess,
    )
    print(json.dumps(rep["before_vs_after"], indent=2))
    print("New V6.4 subreddits:", list(rep["new_v6_4_subreddits"].items())[:10])
    print("Leakage quarantined:", rep["leakage_audit"]["threads_quarantined_from_dapt_train"])


if __name__ == "__main__":
    main()
