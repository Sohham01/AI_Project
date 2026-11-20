"""Bounded Decodo Universal pilot runner (dry-run by default).

Implements the exact single-request pilot protocol from
``docs/small_dataset_pilot.md`` using ``DecodoProvider`` and
``DECODO_API_TOKEN`` from ``backend/.env`` or the process environment.

Zero billable requests are ever made unless BOTH ``--live`` and ``--yes`` are
explicitly supplied. Raw HTML and API keys are never printed or written.

Examples::

    # 1. Inspect the planned bounded request (dry-run, zero credits used):
    python backend/run_decodo_pilot.py --url https://www.reddit.com/r/IIMSambalpur/

    # 2. Execute ONE bounded Universal scrape after confirming account quota:
    python backend/run_decodo_pilot.py \\
        --url https://www.reddit.com/r/IIMSambalpur/ \\
        --max-posts 10 --max-comments 25 \\
        --output data/real_exports/decodo_pilot_v1.json \\
        --export-candidates data/annotations/real_candidates/ \\
        --live --yes
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
from app.schemas.ingestion import PostData
from app.services.ingestion.providers.decodo import DecodoProvider, _validated_reddit_url
from app.services.ingestion.static_export import StaticExportLimits, load_static_export


def _post_to_static_dict(post: PostData) -> dict[str, Any]:
    return {
        "post_id": post.reddit_id,
        "subreddit": post.subreddit,
        "title": post.title,
        "body": post.body,
        "created_at": post.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score": int(post.score or 0),
        "num_comments": int(post.num_comments or 0),
        "url": post.url,
        "comments": [
            {
                "comment_id": c.reddit_id,
                "body": c.body,
                "created_at": c.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "score": int(c.score or 0),
                "parent_comment_id": c.parent_reddit_id,
            }
            for c in post.comments
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_decodo_pilot",
        description="Bounded single-request Decodo Universal Reddit pilot (dry-run by default).",
    )
    parser.add_argument(
        "--url",
        default="https://www.reddit.com/r/IIMSambalpur/",
        help="Approved Reddit subreddit or post thread URL (default: https://www.reddit.com/r/IIMSambalpur/)",
    )
    parser.add_argument("--max-posts", type=int, default=10, help="Local post cap (1..20, default 10)")
    parser.add_argument("--max-comments", type=int, default=25, help="Local comment cap per post (1..100, default 25)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds (default 20)")
    parser.add_argument("--output", default="data/real_exports/decodo_pilot_v1.json", help="Output static-export JSON path")
    parser.add_argument("--dataset-id", default="decodo-pilot-v1", help="Dataset version ID")
    parser.add_argument("--export-candidates", default=None, help="Optional dir to write blank-label annotation candidate CSV")
    parser.add_argument("--live", action="store_true", help="Send the live Decodo /v1/scrape request (requires --yes)")
    parser.add_argument("--yes", action="store_true", help="Confirm billable single-request Decodo execution")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv(_BACKEND_DIR / ".env", override=False)
    args = build_parser().parse_args(argv)

    if not 1 <= args.max_posts <= 20:
        print("Error: --max-posts must be between 1 and 20 for bounded pilots.", file=sys.stderr)
        return 2
    if not 1 <= args.max_comments <= 100:
        print("Error: --max-comments must be between 1 and 100 for bounded pilots.", file=sys.stderr)
        return 2

    try:
        subreddit, post_id = _validated_reddit_url(args.url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    provider = DecodoProvider()
    mode = "scrape_url" if post_id is not None else "scrape_source"
    plan_summary = {
        "mode": mode,
        "url": args.url,
        "subreddit": subreddit,
        "post_id": post_id,
        "proxy_pool": provider.proxy_pool,
        "headless": provider.headless,
        "max_posts": 1 if post_id is not None else args.max_posts,
        "max_comments_per_post": args.max_comments,
        "timeout_seconds": args.timeout,
        "token_configured": bool(provider.api_token),
        "live": bool(args.live and args.yes),
    }
    print(json.dumps({"pilot_plan": plan_summary}, indent=2))

    if not args.live:
        print("\nDry-run complete (0 credits consumed). Re-run with --live --yes after checking your Decodo quota.")
        return 0
    if not args.yes:
        print("Error: --live requires explicit --yes confirmation.", file=sys.stderr)
        return 2
    if not provider.api_token:
        print("Error: DECODO_API_TOKEN is not configured in environment or backend/.env.", file=sys.stderr)
        return 2

    if post_id is not None:
        posts: List[PostData] = [
            provider.scrape_url(
                url=args.url,
                comment_limit=args.max_comments,
                timeout=args.timeout,
            )
        ]
    else:
        posts = provider.scrape_source(
            url=args.url,
            max_posts=args.max_posts,
            comment_limit=args.max_comments,
            timeout=args.timeout,
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_mode = "direct" if subreddit.casefold() in {"iimsambalpur", "iimsambalpurug"} else "context"
    static_posts = [_post_to_static_dict(p) for p in posts]
    payload = {
        "format_version": 1,
        "metadata": {
            "dataset_id": args.dataset_id,
            "source": {
                "community": subreddit,
                "source_reference": f"decodo-universal:{now_iso}",
                "source_provider": "decodo-universal",
                "source_mode": source_mode,
                "source_url": args.url,
                "collected_at": now_iso,
            },
            "completeness": {
                "comments_complete": False,
            },
            "authorization": {
                "authorization_status": "pending-review",
                "authorization_reference": "operator-decodo-pilot",
            },
        },
        "posts": static_posts,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    validated = load_static_export(out_path, limits=StaticExportLimits())
    print(
        json.dumps(
            {
                "posts_retrieved": len(posts),
                "comments_retrieved": sum(len(p.comments) for p in posts),
                "parser_diagnostics": getattr(provider, "last_diagnostics", []),
                "validation_summary": validated.as_summary(),
                "output_path": str(out_path),
            },
            indent=2,
        )
    )
    if args.export_candidates and static_posts:
        from backend.import_reddit_export import export_blank_annotation_candidates

        export_blank_annotation_candidates(static_posts, Path(args.export_candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
