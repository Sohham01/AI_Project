"""Rate-limit-aware Arctic Shift real-data pilot collector, sanitizer, and
blank-label 5-task annotation exporter.

Connects the existing Arctic Shift provider (``ArcticShiftProvider``) cleanly
to RCI's canonical privacy, static-export validation, and human-annotation
pipeline:

  ArcticShiftProvider (posts + comment trees, keyless)
    -> IIM Sambalpur relevance scope check (direct + context communities)
    -> Inline PII redaction (<EMAIL>, <PHONE>, <USER_MENTION>; zero author fields)
    -> Validated format_version=1 static export JSON (load_static_export)
    -> Blank-label 5-task human review CSV + canonical review_manifest.jsonl
    -> Optional opt-in PostgreSQL persistence (--persist --yes)

Never modifies data/demo_synthetic/, ml/artifacts/demo_synthetic_v2/, or /api/demo.

Examples::

    # Bounded live pilot (15 real IIM Sambalpur threads, up to 25 comments/thread):
    python backend/run_arctic_shift_pilot.py \\
        --max-posts 15 --max-comments 25 --after 1year \\
        --output data/processed/arctic_shift_pilot/sanitized_static_export.json \\
        --export-candidates data/annotations/arctic_shift_pilot
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.schemas.ingestion import PostData
from app.services.ingestion.collection import (
    DEFAULT_CONTEXT_SUBREDDITS,
    DEFAULT_DIRECT_SUBREDDITS,
    DEFAULT_TARGET_ALIASES,
    HARD_MAX_COMMENTS_PER_POST,
    HARD_MAX_POSTS,
    _is_retryable,
    is_target_relevant,
)
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import ProviderRequestError, safe_failure_metadata
from app.services.ingestion.static_export import StaticExportLimits, load_static_export
from backend.import_reddit_export import (
    export_blank_annotation_candidates,
    sanitize_pii_text,
)


def _polite_sleep(
    provider: ArcticShiftProvider,
    base_delay: float,
    sleeper: Callable[[float], None],
) -> None:
    """Sleep politely between Arctic Shift requests.

    Only extends beyond ``base_delay`` when ``x-ratelimit-remaining == 0``;
    normal 200 OK responses still carry ``x-ratelimit-reset`` (sliding window
    countdown) and should not trigger a full-window sleep.
    """
    remaining = getattr(provider.client, "last_ratelimit_remaining", None)
    reset = getattr(provider.client, "last_ratelimit_reset", None)
    wait = base_delay
    if remaining == 0 and isinstance(reset, (int, float)) and reset > 0:
        wait = max(wait, min(float(reset) + 1.0, 20.0))
    if wait > 0:
        sleeper(wait)


def _call_arctic_with_retries(
    provider: ArcticShiftProvider,
    fn: Callable[[], Any],
    *,
    max_retries: int,
    base_delay: float,
    sleeper: Callable[[float], None],
    label: str,
) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            reset = getattr(provider.client, "last_ratelimit_reset", None)
            backoff = min(max(base_delay, 2.0) * (2 ** attempt), 15.0)
            if isinstance(reset, (int, float)) and 0 < reset <= 15.0:
                backoff = max(backoff, float(reset) + 1.0)
            cat, status = safe_failure_metadata(exc)
            print(
                f"  [retry {attempt + 1}/{max_retries}] {label} failed "
                f"({cat} status={status}); waiting {backoff:.1f}s...",
                flush=True,
            )
            sleeper(backoff)
    raise last_exc  # pragma: no cover


def _sanitize_post_to_static_dict(
    post: PostData,
    pii_counts: Counter,
    parent_counts: Counter,
) -> dict[str, Any]:
    title = sanitize_pii_text(post.title, pii_counts) or post.title
    body = sanitize_pii_text(post.body, pii_counts) if post.body is not None else None
    comment_ids = {c.reddit_id for c in post.comments}
    sanitized_comments: List[dict[str, Any]] = []
    for c in post.comments:
        c_body = sanitize_pii_text(c.body, pii_counts) or c.body
        if c.parent_reddit_id is None:
            parent_counts["top_level"] += 1
        elif c.parent_reddit_id in comment_ids:
            parent_counts["available"] += 1
        else:
            parent_counts["unavailable"] += 1
        sanitized_comments.append(
            {
                "comment_id": c.reddit_id,
                "body": c_body,
                "created_at": c.created_at.replace(tzinfo=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "score": int(c.score),
                "parent_comment_id": c.parent_reddit_id,
            }
        )
    sanitized_comments.sort(key=lambda item: (item["created_at"], item["comment_id"]))
    return {
        "post_id": post.reddit_id,
        "subreddit": post.subreddit,
        "title": title,
        "body": body,
        "created_at": post.created_at.replace(tzinfo=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "score": int(post.score),
        "num_comments": int(post.num_comments),
        "url": post.url,
        "comments": sanitized_comments,
    }


def collect_arctic_shift_pilot(
    *,
    direct_subreddits: Sequence[str] = DEFAULT_DIRECT_SUBREDDITS,
    context_subreddits: Sequence[str] = DEFAULT_CONTEXT_SUBREDDITS,
    query: str = "IIM Sambalpur",
    after: Optional[str] = "1year",
    max_posts: int = 15,
    max_posts_per_source: Optional[int] = None,
    max_comments_per_post: int = 25,
    min_comments: int = 0,
    search_field: str = "auto",
    request_delay: float = 2.5,
    max_retries: int = 2,
    timeout: float = 30.0,
    provider: Optional[ArcticShiftProvider] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Tuple[List[PostData], Dict[str, Any]]:
    """Collect a bounded, relevant set of posts + comment trees via Arctic Shift."""
    if not 1 <= max_posts <= HARD_MAX_POSTS:
        raise ValueError(f"max_posts must be between 1 and {HARD_MAX_POSTS}")
    if max_posts_per_source is not None and not 1 <= max_posts_per_source <= max_posts:
        raise ValueError(f"max_posts_per_source must be between 1 and {max_posts}")
    if not 1 <= max_comments_per_post <= HARD_MAX_COMMENTS_PER_POST:
        raise ValueError(
            f"max_comments_per_post must be between 1 and {HARD_MAX_COMMENTS_PER_POST}"
        )
    if search_field not in {"auto", "title", "query"}:
        raise ValueError("search_field must be one of: auto, title, query")

    prov = provider if provider is not None else ArcticShiftProvider()
    retained: Dict[str, PostData] = {}
    source_stats: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    truncated_threads: List[str] = []

    def _run_search(
        subreddit: str,
        mode: str,
        use_title_fallback: bool = False,
    ) -> None:
        remaining = max_posts - len(retained)
        if remaining <= 0:
            return
        source_cap = min(remaining, max_posts_per_source) if max_posts_per_source else remaining
        fetch_limit = min(max(source_cap * 2, 10), 100)
        entry: Dict[str, Any] = {
            "subreddit": subreddit,
            "mode": mode,
            "search_field": "none" if mode == "direct" else ("title" if use_title_fallback else "query"),
            "posts_returned": 0,
            "posts_retained": 0,
            "status": "ok",
        }

        def _do_call():
            kwargs: Dict[str, Any] = {
                "subreddit": subreddit,
                "limit": fetch_limit,
                "after": after,
                "sort": "desc",
                "timeout": timeout,
            }
            if mode == "context":
                if use_title_fallback:
                    kwargs["title"] = query
                else:
                    kwargs["query"] = query
            return prov.search_posts(**kwargs)

        # On context full-text query=, try once before falling back immediately
        # to the much faster title-only index if Arctic Shift returns HTTP 422 timeout.
        effective_retries = 0 if (mode == "context" and not use_title_fallback) else max_retries
        try:
            batch = _call_arctic_with_retries(
                prov,
                _do_call,
                max_retries=effective_retries,
                base_delay=request_delay,
                sleeper=sleeper,
                label=f"{mode}:r/{subreddit}({entry['search_field']})",
            )
        except Exception as exc:  # noqa: BLE001
            cat, status = safe_failure_metadata(exc)
            # On massive context subreddits where full-text query times out (HTTP 422),
            # automatically try the faster title-only index once.
            if mode == "context" and not use_title_fallback and cat == "timeout":
                print(
                    f"  [fallback] r/{subreddit} full-text query timed out; "
                    f"trying faster title-only search...",
                    flush=True,
                )
                _polite_sleep(prov, request_delay, sleeper)
                _run_search(subreddit, mode, use_title_fallback=True)
                return
            entry["status"] = "failed"
            entry["failure_category"] = cat
            entry["status_code"] = status
            errors.append(
                {
                    "stage": "search_posts",
                    "subreddit": subreddit,
                    "mode": mode,
                    "failure_category": cat,
                    "status_code": status,
                }
            )
            source_stats.append(entry)
            _polite_sleep(prov, request_delay, sleeper)
            return

        entry["posts_returned"] = len(batch)
        for post in batch:
            if len(retained) >= max_posts or entry["posts_retained"] >= source_cap:
                break
            if post.reddit_id in retained:
                continue
            if min_comments > 0 and post.num_comments < min_comments:
                continue
            if mode == "context":
                combined = f"{post.title} {post.body or ''}"
                if not is_target_relevant(combined, aliases=DEFAULT_TARGET_ALIASES):
                    continue
            retained[post.reddit_id] = post
            entry["posts_retained"] += 1

        source_stats.append(entry)
        print(
            f"  [search] r/{subreddit} ({mode}/{entry['search_field']}): "
            f"returned={entry['posts_returned']} retained={entry['posts_retained']} "
            f"(total={len(retained)}/{max_posts})",
            flush=True,
        )
        _polite_sleep(prov, request_delay, sleeper)

    # 1. Direct institution communities first (highest precision)
    for sub in direct_subreddits:
        _run_search(sub, "direct")

    # 2. Context MBA/CAT communities with institution-specific query + relevance gate
    use_title_first = search_field == "title"
    for sub in context_subreddits:
        _run_search(sub, "context", use_title_fallback=use_title_first)

    # 3. Retrieve nested comment trees ONLY for retained posts
    ordered_posts = sorted(retained.values(), key=lambda p: p.created_at, reverse=True)
    comments_fetched_total = 0
    for idx, post in enumerate(ordered_posts, start=1):
        if post.num_comments == 0:
            post.comments = []
            continue
        try:
            comments = _call_arctic_with_retries(
                prov,
                lambda pid=post.reddit_id: prov.get_comments(
                    pid, limit=max_comments_per_post, timeout=timeout
                ),
                max_retries=max_retries,
                base_delay=request_delay,
                sleeper=sleeper,
                label=f"comments:{post.reddit_id}",
            )
            post.comments = comments
            comments_fetched_total += len(comments)
            if prov.last_comments_truncated:
                truncated_threads.append(post.reddit_id)
            print(
                f"  [comments {idx}/{len(ordered_posts)}] r/{post.subreddit} "
                f"{post.reddit_id}: {len(comments)} comments "
                f"(reported={post.num_comments}, truncated={prov.last_comments_truncated})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            cat, status = safe_failure_metadata(exc)
            post.comments = []
            errors.append(
                {
                    "stage": "get_comments",
                    "post_id": post.reddit_id,
                    "subreddit": post.subreddit,
                    "failure_category": cat,
                    "status_code": status,
                }
            )
            print(
                f"  [comments {idx}/{len(ordered_posts)}] {post.reddit_id} "
                f"FAILED ({cat} status={status})",
                flush=True,
            )
        if idx < len(ordered_posts):
            _polite_sleep(prov, min(request_delay, 1.5), sleeper)

    collection_summary = {
        "provider": "arctic_shift",
        "query": query,
        "after": after,
        "max_posts": max_posts,
        "max_comments_per_post": max_comments_per_post,
        "posts_retained": len(ordered_posts),
        "comments_retrieved": comments_fetched_total,
        "truncated_comment_threads": truncated_threads,
        "sources": source_stats,
        "errors": errors,
        "status": "partial_failure" if errors and ordered_posts else ("failed" if errors else "ok"),
    }
    return ordered_posts, collection_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_arctic_shift_pilot",
        description="Collect a bounded real Reddit pilot via Arctic Shift, sanitize PII, validate static_export, and generate blank-label 5-task annotation candidates.",
    )
    parser.add_argument(
        "--direct-subreddit",
        action="append",
        dest="direct_subreddits",
        help=f"Direct subreddit(s) (default: {list(DEFAULT_DIRECT_SUBREDDITS)})",
    )
    parser.add_argument(
        "--context-subreddit",
        action="append",
        dest="context_subreddits",
        help=f"Context subreddit(s) (default: {list(DEFAULT_CONTEXT_SUBREDDITS)})",
    )
    parser.add_argument("--no-direct", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--query", default="IIM Sambalpur", help="Institution search phrase for context communities")
    parser.add_argument("--after", default="1year", help="Lower date bound (default: 1year; pass 'none' for all-time)")
    parser.add_argument("--max-posts", type=int, default=15, help="Max posts to collect (1..200, default 15)")
    parser.add_argument("--max-posts-per-source", type=int, default=None, help="Optional cap per subreddit to balance direct and context sources")
    parser.add_argument("--max-comments", type=int, default=25, help="Max comments per post (1..2000, default 25)")
    parser.add_argument("--min-comments", type=int, default=0, help="Skip posts reporting fewer than N comments (default 0)")
    parser.add_argument("--search-field", default="auto", choices=["auto", "title", "query"], help="Context search field (default: auto; use 'title' for fast index)")
    parser.add_argument("--delay", type=float, default=2.5, help="Polite base delay in seconds between requests (default 2.5)")
    parser.add_argument("--max-retries", type=int, default=2, help="Bounded transient retries (default 2)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds (default 30)")
    parser.add_argument("--dataset-id", default="arctic-shift-pilot-v1", help="Dataset version ID")
    parser.add_argument(
        "--output",
        default="data/processed/arctic_shift_pilot/sanitized_static_export.json",
        help="Output path for sanitized validated format_version=1 JSON",
    )
    parser.add_argument(
        "--export-candidates",
        default="data/annotations/arctic_shift_pilot",
        help="Directory for blank-label 5-task annotation CSVs + manifest sidecar",
    )
    parser.add_argument("--persist", action="store_true", help="Also persist posts/comments to PostgreSQL (requires --yes)")
    parser.add_argument("--yes", action="store_true", help="Explicit confirmation for --persist")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.persist and not args.yes:
        print("Error: --persist requires --yes.", file=sys.stderr)
        return 2

    direct = [] if args.no_direct else (args.direct_subreddits or list(DEFAULT_DIRECT_SUBREDDITS))
    context = [] if args.no_context else (args.context_subreddits or list(DEFAULT_CONTEXT_SUBREDDITS))
    after_val = None if (args.after or "").strip().lower() == "none" else args.after

    print(
        f"Starting Arctic Shift bounded pilot: direct={direct}, context={context}, "
        f"query={args.query!r}, after={after_val!r}, max_posts={args.max_posts}, "
        f"max_comments={args.max_comments}"
    )

    posts, collection_summary = collect_arctic_shift_pilot(
        direct_subreddits=direct,
        context_subreddits=context,
        query=args.query,
        after=after_val,
        max_posts=args.max_posts,
        max_posts_per_source=args.max_posts_per_source,
        max_comments_per_post=args.max_comments,
        min_comments=args.min_comments,
        search_field=args.search_field,
        request_delay=args.delay,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    if not posts:
        print(json.dumps({"collection_summary": collection_summary}, indent=2))
        print("No posts retrieved from Arctic Shift.", file=sys.stderr)
        return 2

    pii_counts: Counter = Counter()
    parent_counts: Counter = Counter()
    static_posts = [
        _sanitize_post_to_static_dict(p, pii_counts, parent_counts) for p in posts
    ]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content_hash = hashlib.sha256(
        json.dumps(static_posts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    source_mode = "direct" if direct and not context else "context"
    primary_community = direct[0] if direct else (context[0] if context else "IIMSambalpur")

    payload = {
        "format_version": 1,
        "metadata": {
            "dataset_id": args.dataset_id,
            "source": {
                "community": primary_community,
                "source_reference": f"arctic-shift:sha256:{content_hash}",
                "source_provider": "arctic_shift",
                "source_mode": source_mode,
                "source_url": "https://arctic-shift.photon-reddit.com/api",
                "collected_at": now_iso,
            },
            "completeness": {
                "comments_complete": False,
            },
            "authorization": {
                "authorization_status": "pending-review",
                "authorization_reference": "operator-arctic-shift-pilot",
            },
        },
        "posts": static_posts,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    limits = StaticExportLimits(
        max_bytes=max(10 * 1024 * 1024, out_path.stat().st_size + 1024),
        max_posts=max(200, len(static_posts)),
        max_comments_per_post=max(2000, args.max_comments),
        max_comments=max(5000, collection_summary["comments_retrieved"] + 1),
    )
    validated = load_static_export(out_path, limits=limits)

    candidate_summary = None
    if args.export_candidates and validated.valid:
        candidate_summary = export_blank_annotation_candidates(
            static_posts,
            Path(args.export_candidates),
            source_provider="arctic_shift",
            source_mode=source_mode,
        )

    persistence_summary = None
    if args.persist and args.yes:
        from app.core.database import SessionLocal
        from app.services.ingestion.persistence import PersistenceService

        session = SessionLocal()
        try:
            persistence_summary = PersistenceService(session).save_posts(
                posts, provider_name="arctic_shift"
            )
        finally:
            session.close()

    report = {
        "dataset_id": args.dataset_id,
        "generated_at": now_iso,
        "output_path": str(out_path),
        "collection_summary": collection_summary,
        "parent_context_breakdown": {
            "top_level": parent_counts["top_level"],
            "available": parent_counts["available"],
            "unavailable": parent_counts["unavailable"],
        },
        "pii_redactions": {
            "email_redactions": pii_counts["email_redactions"],
            "phone_redactions": pii_counts["phone_redactions"],
            "user_mention_redactions": pii_counts["user_mention_redactions"],
        },
        "validation_summary": validated.as_summary(),
        "candidate_export_summary": candidate_summary,
        "persistence_summary": persistence_summary,
    }

    report_path = out_path.parent / "dataset_quality_provenance_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if validated.valid else 3


if __name__ == "__main__":
    sys.exit(main())
