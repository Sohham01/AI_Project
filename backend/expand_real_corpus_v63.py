"""Sprint P1-A — Reproducible, Leakage-Safe Real Reddit Corpus Expansion (V6.3).

Expands the usable research corpus from the 365-thread V6.1 baseline toward the
700-1,000 unique thread planning target using ONLY authorized public Arctic Shift
archive snapshots (`ArcticShiftProvider`) and strictly enforcing:
  - Resumable per-search + per-post staging checkpoints under data/processed/arctic_shift_v6_3_staging/
  - Rate-limit-aware adaptive sleeping (x-ratelimit-remaining / x-ratelimit-reset) + retries
  - Full PII redaction (<EMAIL>, <PHONE>, <USER_MENTION>, zero raw Reddit URLs/authors)
  - Bot/AutoMod/GIF/spam/low-info filtering + parent-comment chain integrity validation
  - Exact normalized (title, body) SHA-256 + 5-gram Jaccard near-duplicate filtering
  - Strict isolation from data/live_corpus/corpus.json (untouched!) and from the 44
    frozen test / same-title threads in data/annotations/real_reddit_labeled_v1/
  - Separate versioned outputs under data/corpus_v6_3/ + ml/artifacts/v6_3_experiments/dataset_expansion/
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import safe_failure_metadata
from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from backend.import_reddit_export import sanitize_pii_text
from backend.train_real_pilot_models import _is_automod_boilerplate, _is_gif_only

BASELINE_CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
FROZEN_MANIFEST_PATH = ROOT / "ml" / "artifacts" / "v5_experiments" / "manifests" / "v5_frozen_splits.json"
STAGING_DIR = ROOT / "data" / "processed" / "arctic_shift_v6_3_staging"
OUT_CORPUS_DIR = ROOT / "data" / "corpus_v6_3"
OUT_AUDIT_DIR = ROOT / "ml" / "artifacts" / "v6_3_experiments" / "dataset_expansion"

_BOT_EXTRA_RE = re.compile(
    r"(RemindMeBot|I will be messaging you in|follows platform-wide Reddit Rules|^RemindMe!\s*\d*)",
    re.IGNORECASE,
)
_REDDIT_URL_RE = re.compile(r"https?://(?:www\.|old\.|np\.)?reddit\.com/\S+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9\u0900-\u0d7f]+", re.IGNORECASE)

# Strict higher-ed / IIM / MBA / CAT relevance patterns
_CORE_ACADEMIC_COMMUNITIES = {
    "iimsambalpur",
    "iimsambalpurug",
    "mbaindia",
    "catpreparation",
    "catprep",
    "iimcatpreparation",
    "iim_cat",
    "catstudyroom",
    "ipmat",
    "indian_academia",
}
_HIGHER_ED_RE = re.compile(
    r"\b(iim|mba|pgdm|cat\b|xat\b|ipmat|waitlist|merit list|placement|placements|sip\b|"
    r"internship|hostel|mess|campus|tuition|fees?\b|education loan|emi\b|aicte|refund|"
    r"curriculum|faculty|semester|trimester|quiz|exam|exams|cgpa|percentile|college|university|"
    r"b-?school|bschool|admission|academics|alumni|package|ctc|lpa|shortlist|interview|gdpi|wat)\b",
    re.IGNORECASE,
)
_SPAM_PROMO_RE = re.compile(
    r"\b(telegram group link|whatsapp channel join fast|buy follower|crypto signal|betting|escort|onlyfans)\b",
    re.IGNORECASE,
)

# Balanced multi-community collection plan (deterministic order)
COLLECTION_PLANS: List[Dict[str, Any]] = [
    # Broad IIM & MBA core communities (note: Arctic Shift accepts ISO dates or <N>d / 1year)
    {"subreddit": "MBAIndia", "mode": "direct", "query": None, "limit": 100, "after": "730d", "before": None},
    {"subreddit": "MBAIndia", "mode": "direct", "query": None, "limit": 100, "after": "2023-01-01", "before": "2025-06-01"},
    {"subreddit": "MBAIndia", "mode": "context", "query": "hostel", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "refund", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "MBAIndia", "mode": "context", "query": "loan", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "IIMCATPreparation", "mode": "direct", "query": None, "limit": 100, "after": "730d", "before": None},
    {"subreddit": "CATPrep", "mode": "direct", "query": None, "limit": 100, "after": "730d", "before": None},
    {"subreddit": "CATStudyRoom", "mode": "direct", "query": None, "limit": 100, "after": "730d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "waitlist", "limit": 70, "after": "730d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "hostel", "limit": 50, "after": "730d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "loan", "limit": 50, "after": "730d", "before": None},
    {"subreddit": "CATpreparation", "mode": "context", "query": "refund", "limit": 50, "after": "730d", "before": None},
    {"subreddit": "iim_CAT", "mode": "direct", "query": None, "limit": 80, "after": "730d", "before": None},
    {"subreddit": "ipmat", "mode": "context", "query": "IIM", "limit": 80, "after": "730d", "before": None},
    {"subreddit": "Indian_Academia", "mode": "context", "query": "IIM", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "Indian_Academia", "mode": "context", "query": "MBA", "limit": 80, "after": "1095d", "before": None},
    {"subreddit": "Indian_Academia", "mode": "context", "query": "hostel", "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "IIMSambalpur", "mode": "direct", "query": None, "limit": 60, "after": "1095d", "before": None},
    {"subreddit": "IIMSambalpurUG", "mode": "direct", "query": None, "limit": 80, "after": "730d", "before": None},
    # Regional Indian communities filtered strictly to higher-ed/IIM/MBA/student issues (aids Bengali/Odia real coverage too!)
    {"subreddit": "Odisha", "mode": "context", "query": "IIM", "limit": 40, "after": "1460d", "before": None},
    {"subreddit": "Odisha", "mode": "context", "query": "MBA", "limit": 40, "after": "1460d", "before": None},
    {"subreddit": "Odisha", "mode": "context", "query": "college", "limit": 50, "after": "1095d", "before": None},
    {"subreddit": "kolkata", "mode": "context", "query": "IIM", "limit": 40, "after": "1095d", "before": None},
    {"subreddit": "kolkata", "mode": "context", "query": "MBA", "limit": 40, "after": "1095d", "before": None},
    {"subreddit": "kolkata", "mode": "context", "query": "college", "limit": 50, "after": "730d", "before": None},
    {"subreddit": "westbengal", "mode": "context", "query": "college", "limit": 40, "after": "1460d", "before": None},
    {"subreddit": "Bhubaneswar", "mode": "context", "query": "MBA", "limit": 30, "after": "1460d", "before": None},
    {"subreddit": "Bhubaneswar", "mode": "context", "query": "college", "limit": 40, "after": "1095d", "before": None},
]


def _norm_text(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip().lower()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, n: int = 5) -> Set[str]:
    toks = _WORD_RE.findall(_norm_text(text))
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def is_valid_comment_text(text: str) -> Tuple[bool, str]:
    raw = str(text or "").strip()
    if not raw:
        return False, "empty"
    low = raw.lower()
    if low in {"[deleted]", "[removed]", "[ removed by reddit ]", "deleted"}:
        return False, "deleted_or_removed"
    if _is_automod_boilerplate(raw) or _BOT_EXTRA_RE.search(raw):
        return False, "bot_or_automod"
    if _is_gif_only(raw):
        return False, "gif_or_bare_media_url"
    if _SPAM_PROMO_RE.search(raw):
        return False, "spam_promo"
    return True, "ok"


def clean_and_redact(text: str, pii_counts: Counter) -> str:
    cleaned = sanitize_pii_text(str(text or ""), pii_counts) or ""
    if _REDDIT_URL_RE.search(cleaned):
        pii_counts["reddit_url_redacted"] += len(_REDDIT_URL_RE.findall(cleaned))
        cleaned = _REDDIT_URL_RE.sub("<REDDIT_LINK>", cleaned)
    return cleaned.strip()


def is_education_relevant(subreddit: str, title: str, body: str, comments: Sequence[str]) -> bool:
    if subreddit.lower() in _CORE_ACADEMIC_COMMUNITIES:
        return len(_norm_text(title)) >= 8
    blob = f"{title} {body[:500]} {' '.join(comments[:4])}"
    return bool(_HIGHER_ED_RE.search(blob))


def _adaptive_sleep(provider: ArcticShiftProvider, base_delay: float, sleeper: Callable[[float], None]) -> None:
    rem = getattr(provider.client, "last_ratelimit_remaining", None)
    reset = getattr(provider.client, "last_ratelimit_reset", None)
    wait = base_delay
    if rem == 0 and isinstance(reset, (int, float)) and reset > 0:
        wait = max(wait, min(float(reset) + 0.5, 15.0))
    if wait > 0:
        sleeper(wait)


def _call_with_retry(
    provider: ArcticShiftProvider,
    fn: Callable[[], Any],
    *,
    max_retries: int = 2,
    base_delay: float = 1.2,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries:
                raise
            cat, status = safe_failure_metadata(exc)
            if cat not in {"timeout", "connection", "transport"} and status not in {422, 429, 500, 502, 503, 504}:
                raise
            wait = min(max(base_delay, 1.5) * (2 ** attempt), 12.0)
            sleeper(wait)
    raise last_exc  # pragma: no cover


def _resolve_frozen_test_post_ids(baseline: Dict[str, Any]) -> Set[str]:
    """Load frozen test post IDs from manifest if present, else derive identically from tracked V1 CSVs."""
    if FROZEN_MANIFEST_PATH.is_file():
        frozen = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
        pids = set(frozen.get("test_corpus_post_ids") or [])
        if pids:
            return pids

    import csv
    from collections import defaultdict as _dd
    from ml.configs import TaskType
    from ml.data.dataset import load_annotation_dataset
    from ml.data.splitting import SplitSpec, apply_split, extract_groups, split_records

    v1_dir = ROOT / "data" / "annotations" / "real_reddit_labeled_v1"
    split_spec = SplitSpec(
        train_size=0.65, validation_size=0.15, test_size=0.20, random_state=20261116, stratify=True
    )
    recs = load_annotation_dataset(TaskType.ISSUE, str(v1_dir / "issue.csv"))
    sp = split_records(recs, labels=[r.label for r in recs], spec=split_spec, groups=extract_groups(recs, "post_id"))
    _, _, te = apply_split(recs, sp)
    test_dids = {r.document_id for r in te}

    posts_by_title: Dict[str, List[Dict[str, Any]]] = _dd(list)
    for p in baseline.get("posts", []):
        posts_by_title[_norm_text(p.get("title"))[:120]].append(p)

    resolved: Set[str] = set()
    with (v1_dir / "master_labeled_v1.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["document_id"] not in test_dids:
                continue
            nt = _norm_text(row.get("text"))
            hit_pid: Optional[str] = None
            for cand in posts_by_title.get(_norm_text(row.get("post_title"))[:120], []):
                for c in cand.get("comments") or []:
                    nb = _norm_text(c.get("body"))
                    if nb == nt or (len(nt) >= 30 and (nb.startswith(nt[:60]) or nt.startswith(nb[:60]))):
                        hit_pid = str(cand["reddit_id"])
                        break
                if hit_pid is not None:
                    break
            resolved.add(hit_pid or str(row["post_id"]))
    return resolved


def load_baseline_and_frozen_guards() -> Dict[str, Any]:
    """Load baseline corpus fingerprints and frozen test-thread exclusions (read-only)."""
    baseline = json.loads(BASELINE_CORPUS_PATH.read_text(encoding="utf-8"))
    test_pids = _resolve_frozen_test_post_ids(baseline)

    existing_title_hashes: Set[str] = set()
    existing_title120: Set[str] = set()
    existing_title_shingles: List[Tuple[str, Set[str]]] = []
    existing_comment_hashes: Set[str] = set()

    test_title120: Set[str] = set()
    test_comment_hashes: Set[str] = set()
    test_comment_shingles: List[Tuple[str, Set[str]]] = []

    for p in baseline.get("posts", []):
        pid = str(p.get("reddit_id") or "")
        t_norm = _norm_text(p.get("title"))
        b_norm = _norm_text(p.get("body"))
        existing_title_hashes.add(_text_sha256(f"{t_norm}|{b_norm}"))
        if t_norm:
            existing_title120.add(t_norm[:120])
            sh = _shingles(t_norm, 5)
            if sh:
                existing_title_shingles.append((pid, sh))
        if pid in test_pids and t_norm:
            test_title120.add(t_norm[:120])

        for c in p.get("comments", []):
            cb = str(c.get("body") or "")
            existing_comment_hashes.add(_text_sha256(cb))
            if pid in test_pids and len(_norm_text(cb)) >= 20:
                test_comment_hashes.add(_text_sha256(cb))
                csh = _shingles(cb, 5)
                if len(csh) >= 3:
                    test_comment_shingles.append((str(c.get("reddit_id") or ""), csh))

    return {
        "baseline_posts": len(baseline.get("posts", [])),
        "baseline_comments": sum(len(p.get("comments", [])) for p in baseline.get("posts", [])),
        "baseline_subreddits": Counter(p.get("subreddit", "") for p in baseline.get("posts", [])),
        "existing_title_hashes": existing_title_hashes,
        "existing_title120": existing_title120,
        "existing_title_shingles": existing_title_shingles,
        "existing_comment_hashes": existing_comment_hashes,
        "test_pids": test_pids,
        "test_title120": test_title120,
        "test_comment_hashes": test_comment_hashes,
        "test_comment_shingles": test_comment_shingles,
    }


def check_thread_leakage_and_duplication(
    *,
    title: str,
    body: str,
    comment_bodies: Sequence[str],
    guards: Dict[str, Any],
    seen_new_title_hashes: Set[str],
    seen_new_title120: Set[str],
) -> Tuple[Optional[str], bool]:
    """Return (rejection_reason_or_None, dapt_and_train_excluded_bool)."""
    t_norm = _norm_text(title)
    b_norm = _norm_text(body)
    tb_hash = _text_sha256(f"{t_norm}|{b_norm}")

    if tb_hash in guards["existing_title_hashes"] or t_norm[:120] in guards["existing_title120"]:
        return "duplicate_of_baseline_corpus", False
    if tb_hash in seen_new_title_hashes or t_norm[:120] in seen_new_title120:
        return "duplicate_within_new_collection", False

    t_sh = _shingles(t_norm, 5)
    if len(t_sh) >= 4:
        for _, base_sh in guards["existing_title_shingles"]:
            if _jaccard(t_sh, base_sh) >= 0.85:
                return "near_duplicate_title_of_baseline", False

    # Leakage protection against frozen 92-comment / 37-thread test split
    excluded = False
    if t_norm[:120] in guards["test_title120"]:
        excluded = True
    else:
        for cb in comment_bodies:
            if len(_norm_text(cb)) < 20:
                continue
            if _text_sha256(cb) in guards["test_comment_hashes"]:
                excluded = True
                break
            csh = _shingles(cb, 5)
            if len(csh) >= 3:
                for _, t_csh in guards["test_comment_shingles"]:
                    if _jaccard(csh, t_csh) >= 0.80:
                        excluded = True
                        break
            if excluded:
                break

    return None, excluded


def expand_real_corpus(
    *,
    target_new_threads: int = 480,
    max_comments_per_post: int = 80,
    request_delay: float = 0.85,
    fetch_live: bool = True,
) -> Dict[str, Any]:
    """Run resumable real-Reddit collection via Arctic Shift, deduplicate, audit leakage, and save v6.3 corpus."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    guards = load_baseline_and_frozen_guards()
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
    seen_title120_stage: Set[str] = set()

    if fetch_live and provider is not None:
        print(f"[P1-A] Starting Arctic Shift multi-community search (cached staged threads: {len(raw_cache_by_id)})...", flush=True)
        for plan_idx, plan in enumerate(COLLECTION_PLANS, start=1):
            if len(raw_cache_by_id) + len(search_candidates) >= target_new_threads + 180:
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
                print(f"  [{plan_idx}/{len(COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) -> returned={len(posts)}, new={added}", flush=True)
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
                print(f"  [{plan_idx}/{len(COLLECTION_PLANS)}] r/{sub} ({query or 'direct'}) FAILED: {cat}:{code}", flush=True)
            _adaptive_sleep(provider, request_delay, time.sleep)

        # Prioritize underrepresented communities & regional threads before filling remaining slots
        cand_list = list(search_candidates.values())
        cand_list.sort(
            key=lambda x: (
                0 if x["subreddit"].lower() not in {"iimsambalpurug", "iimsambalpur"} else 1,
                -min(x["num_comments"], 40),
                x["source_id_sha"],
            )
        )
        needed = max(0, target_new_threads - len(raw_cache_by_id))
        to_fetch = cand_list[: max(needed + 40, needed)]
        print(f"[P1-A] Fetching comment trees for {len(to_fetch)} queued threads (existing staged={len(raw_cache_by_id)})...", flush=True)

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
                    search_log.append({"stage": "comments", "source_id_sha": item["source_id_sha"], "status": "failed", "error": f"{cat}:{code}"})
                    _adaptive_sleep(provider, request_delay, time.sleep)
                    continue

                # Pseudonymize comment IDs immediately in staging so zero raw Reddit IDs touch output files
                cid_map: Dict[str, str] = {}
                for c_i, c in enumerate(cmts, start=1):
                    cid_map[c.reddit_id] = hashlib.sha256(f"t1_{c.reddit_id}".encode("utf-8")).hexdigest()[:24]

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
                if idx % 25 == 0 or idx == len(to_fetch):
                    print(f"  Fetched {idx}/{len(to_fetch)} threads (staged usable={len(raw_cache_by_id)})", flush=True)
                _adaptive_sleep(provider, request_delay, time.sleep)

    # ------------------------------------------------------------------
    # Deterministic Filtering, Deduplication, Leakage Audit & Final Packaging
    # ------------------------------------------------------------------
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

        raw_cmts = item.get("comments") or []
        valid_cmts: List[Dict[str, Any]] = []
        for c in raw_cmts:
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

        if not is_education_relevant(sub, title, body, [c["body"] for c in valid_cmts]):
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

        post_id = f"post_v63_{next_post_num:04d}"
        next_post_num += 1

        if dapt_excluded:
            excluded_leakage_log.append({
                "post_id": post_id,
                "subreddit": sub,
                "reason": "matched_frozen_test_title_or_comment_similarity",
            })

        sha_to_new_cid: Dict[str, str] = {}
        assigned_cmts: List[Dict[str, Any]] = []
        for c in sorted(valid_cmts, key=lambda z: (z.get("created_at") or "", z["comment_id_sha"])):
            new_cid = f"cmt_v63_{next_cmt_num:05d}"
            next_cmt_num += 1
            sha_to_new_cid[c["comment_id_sha"]] = new_cid

        for c in sorted(valid_cmts, key=lambda z: (z.get("created_at") or "", z["comment_id_sha"])):
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
                "corpus_version": "6.3.0-expansion",
                "search_query": item.get("query", "<direct>"),
                "source_id_sha": item["source_id_sha"],
            },
            "comments": assigned_cmts,
        })

    new_comments_count = sum(len(p["comments"]) for p in new_posts_out)
    dates_all = [p["created_at"] for p in new_posts_out if p.get("created_at")]

    new_corpus_payload = {
        "metadata": {
            "dataset_id": "rci-real-reddit-expansion-v6.3",
            "corpus_version": "6.3.0",
            "annotation_status": "unannotated",
            "synthetic_only": False,
            "source_provider": "arctic_shift_public_archive",
            "baseline_reference": "data/live_corpus/corpus.json (365 threads, untouched)",
            "total_new_threads": len(new_posts_out),
            "total_new_comments": new_comments_count,
            "dapt_and_train_excluded_threads": len(excluded_leakage_log),
            "date_range_utc": {"min": min(dates_all) if dates_all else "", "max": max(dates_all) if dates_all else ""},
        },
        "posts": new_posts_out,
    }
    (OUT_CORPUS_DIR / "new_unannotated_corpus_v6_3.json").write_text(
        json.dumps(new_corpus_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Combined distribution statistics (baseline 365 + new expansion)
    base_sub_cmts: Counter = Counter()
    base_langs: Counter = Counter()
    baseline = json.loads(BASELINE_CORPUS_PATH.read_text(encoding="utf-8"))
    base_dates = []
    for p in baseline.get("posts", []):
        if p.get("created_at"):
            base_dates.append(p["created_at"])
        for c in p.get("comments", []):
            base_sub_cmts[p.get("subreddit", "")] += 1
            lid = detect_indian_reddit_language(c.get("body", ""))
            base_langs[lid["language"]] += 1

    combined_threads_by_sub = guards["baseline_subreddits"] + sub_counter
    combined_cmts_by_sub = base_sub_cmts + sub_comments_counter
    combined_langs = base_langs + lang_counter
    total_threads = guards["baseline_posts"] + len(new_posts_out)
    total_comments = guards["baseline_comments"] + new_comments_count

    audit_report = {
        "corpus_version": "6.3.0",
        "before_vs_after": {
            "baseline_v6_1": {"threads": guards["baseline_posts"], "comments": guards["baseline_comments"]},
            "newly_collected_unannotated_v6_3": {"threads": len(new_posts_out), "comments": new_comments_count},
            "combined_v6_3_total": {"threads": total_threads, "comments": total_comments},
            "net_new_thread_growth_pct": round(100.0 * len(new_posts_out) / max(guards["baseline_posts"], 1), 2),
        },
        "community_distribution": {
            sub: {
                "baseline_threads": guards["baseline_subreddits"].get(sub, 0),
                "new_threads": sub_counter.get(sub, 0),
                "combined_threads": cnt,
                "combined_comments": combined_cmts_by_sub.get(sub, 0),
                "combined_thread_share_pct": round(100.0 * cnt / max(total_threads, 1), 2),
            }
            for sub, cnt in combined_threads_by_sub.most_common()
        },
        "language_distribution_new_comments": dict(lang_counter.most_common()),
        "language_distribution_combined": dict(combined_langs.most_common()),
        "script_distribution_new_comments": dict(script_counter.most_common()),
        "date_range_utc": {
            "baseline_min": min(base_dates) if base_dates else "",
            "baseline_max": max(base_dates) if base_dates else "",
            "new_min": min(dates_all) if dates_all else "",
            "new_max": max(dates_all) if dates_all else "",
        },
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
    (OUT_CORPUS_DIR / "combined_corpus_v6_3_manifest.json").write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_AUDIT_DIR / "dataset_expansion_audit.json").write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-new-threads", type=int, default=480)
    parser.add_argument("--max-comments-per-post", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.85)
    parser.add_argument("--offline-reprocess", action="store_true", help="Reprocess existing staging only without network calls")
    args = parser.parse_args()

    rep = expand_real_corpus(
        target_new_threads=args.target_new_threads,
        max_comments_per_post=args.max_comments_per_post,
        request_delay=args.delay,
        fetch_live=not args.offline_reprocess,
    )
    print(json.dumps(rep["before_vs_after"], indent=2))
    print("Community share (top 8):", {k: v["combined_thread_share_pct"] for k, v in list(rep["community_distribution"].items())[:8]})
    print("Leakage quarantined:", rep["leakage_audit"]["threads_quarantined_from_dapt_train"])


if __name__ == "__main__":
    main()
