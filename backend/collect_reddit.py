"""
Bounded Reddit collection CLI.

Runs the canonical ingestion providers through
:class:`app.services.ingestion.manager.IngestionManager` under the V1 research
scope (IIM Sambalpur).  Preview/dry-run is the default: nothing is written to
the database unless ``--persist`` **and** ``--yes`` are both given, and even
then the caller must keep within the hard safety caps.

Examples
--------
Preview the V1 scope (at most 5 posts, no writes)::

    python collect_reddit.py

Searches the direct communities (``IIMSambalpur`` and its UG subreddit) for
bounded recent posts and the context communities (``MBAIndia``, ``CATPrep``,
``CATpreparation``) only for the institution-specific aliases, retaining only
IIM Sambalpur-relevant data::

    python collect_reddit.py --direct-subreddit IIMSambalpur \\
        --direct-subreddit IIMSambalpurUG \\
        --context-subreddit MBAIndia --context-subreddit CATPrep \\
        --context-subreddit CATpreparation \\
        --term "IIM Sambalpur" --max-posts 5

Widen the scope (broader IIM/student discussions are reported but excluded
unless explicitly added)::

    python collect_reddit.py --term "IIM Sambalpur" --include-scope iim_broad

Persist a small, confirmed batch::

    python collect_reddit.py --max-posts 5 --persist --yes

Never pass credentials on the command line - the providers read what they need
from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

# Ensure imports work when run as ``python backend/collect_reddit.py`` from the
# repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
for _path in (_PROJECT_ROOT, _BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.services.ingestion.collection import (
    DEFAULT_CONTEXT_SUBREDDITS,
    DEFAULT_DIRECT_SUBREDDITS,
    DEFAULT_KEEP_SCOPES,
    DEFAULT_MAX_COMMENTS_PER_POST,
    DEFAULT_MAX_POSTS,
    DEFAULT_TARGET_ALIASES,
    HARD_MAX_ALIASES,
    HARD_MAX_COMMENTS_PER_POST,
    HARD_MAX_POSTS,
    HARD_MAX_PROVIDER_SEARCH_DISPATCHES,
    HARD_MAX_RETRIES,
    HARD_MAX_SEARCH_TASKS,
    HARD_MAX_SOURCES,
    CollectionPlan,
    CollectionSource,
    SourceMode,
    run_collection,
)

#: Direct institution communities: bounded recent-post retrieval, no query.
DEFAULT_SUBREDDITS = list(DEFAULT_DIRECT_SUBREDDITS)

#: Context communities: only ever searched with an institution-specific phrase.
DEFAULT_CONTEXT_SUBREDDITS = list(DEFAULT_CONTEXT_SUBREDDITS)

#: Institution-specific target phrases (never a bare "IIM"/"MBA"/"Sambalpur").
DEFAULT_TERMS = list(DEFAULT_TARGET_ALIASES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_reddit",
        description=(
            "Bounded, dry-run-by-default Reddit collection for the IIM "
            "Sambalpur community-intelligence scope."
        ),
    )
    parser.add_argument(
        "--direct-subreddit",
        action="append",
        dest="direct_subreddits",
        help=(
            "institution community to retrieve bounded recent posts from "
            f"(repeatable; approved names only; at most {HARD_MAX_SOURCES} total "
            f"sources; default {DEFAULT_SUBREDDITS})"
        ),
    )
    parser.add_argument(
        "--context-subreddit",
        action="append",
        dest="context_subreddits",
        help=(
            "general community to search with the target phrases (repeatable; "
            f"at most {HARD_MAX_SOURCES} total sources; default "
            f"{DEFAULT_CONTEXT_SUBREDDITS})"
        ),
    )
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="skip the direct institution communities",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="skip the context communities",
    )
    parser.add_argument(
        "--subreddit",
        action="append",
        dest="subreddits",
        help=(
            "legacy alias for --context-subreddit (repeatable; searched with "
            "the target phrases)"
        ),
    )
    parser.add_argument(
        "--term",
        action="append",
        dest="terms",
        help=(
            "institution-specific phrase, forwarded server-side "
            f"(repeatable; at most {HARD_MAX_ALIASES}; default {DEFAULT_TERMS})"
        ),
    )
    parser.add_argument("--after", default="30d", help="lower time bound (default 30d)")
    parser.add_argument("--before", default=None, help="upper time bound (optional)")
    parser.add_argument("--sort", default="desc", choices=["asc", "desc"])
    parser.add_argument(
        "--max-posts",
        type=int,
        default=DEFAULT_MAX_POSTS,
        help=(
            f"maximum unique in-scope posts to collect "
            f"(default {DEFAULT_MAX_POSTS}, hard cap {HARD_MAX_POSTS})"
        ),
    )
    parser.add_argument(
        "--max-posts-per-source",
        type=int,
        default=None,
        help=(
            "maximum in-scope posts kept from any single community "
            "(default: no per-source limit beyond --max-posts)"
        ),
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=DEFAULT_MAX_COMMENTS_PER_POST,
        help=(
            f"maximum comments kept per post "
            f"(default {DEFAULT_MAX_COMMENTS_PER_POST}, "
            f"hard cap {HARD_MAX_COMMENTS_PER_POST})"
        ),
    )
    parser.add_argument(
        "--include-scope",
        action="append",
        dest="include_scopes",
        help=(
            "additional relevance scope to keep, e.g. iim_broad, student_broad "
            "(repeatable; does not change the default IIM Sambalpur focus)"
        ),
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="fetch posts only; do not request comment threads",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="seconds to wait between provider requests (default 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "per-request timeout in seconds (default: provider default, 60s)"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help=(
            "bounded retries for transient provider timeouts/rate limits "
            f"(default 1, hard cap {HARD_MAX_RETRIES}); at most "
            f"{HARD_MAX_SEARCH_TASKS} logical tasks and "
            f"{HARD_MAX_PROVIDER_SEARCH_DISPATCHES} provider search dispatches "
            "per plan"
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="write collected posts/comments to PostgreSQL (requires --yes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="explicit confirmation for --persist",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON instead of human-readable text",
    )
    return parser


def plan_from_args(args: argparse.Namespace) -> CollectionPlan:
    if args.no_direct:
        direct: List[str] = []
    else:
        direct = list(args.direct_subreddits) if args.direct_subreddits else list(
            DEFAULT_DIRECT_SUBREDDITS
        )

    if args.no_context:
        context: List[str] = []
    else:
        context = []
        if args.context_subreddits:
            context.extend(args.context_subreddits)
        # ``--subreddit`` is kept as a backwards-compatible context alias.
        if args.subreddits:
            context.extend(args.subreddits)
        if not context:
            context = list(DEFAULT_CONTEXT_SUBREDDITS)
        context = list(dict.fromkeys(context))

    terms = tuple(args.terms) if args.terms else tuple(DEFAULT_TARGET_ALIASES)
    keep_scopes = tuple(DEFAULT_KEEP_SCOPES) + tuple(args.include_scopes or ())
    # de-duplicate while preserving order
    keep_scopes = tuple(dict.fromkeys(keep_scopes))

    sources = [
        CollectionSource(subreddit=sub, mode=SourceMode.DIRECT) for sub in direct
    ]
    sources += [
        CollectionSource(subreddit=sub, mode=SourceMode.CONTEXT, terms=terms)
        for sub in context
    ]

    return CollectionPlan(
        sources=sources,
        aliases=terms,
        after=args.after,
        before=args.before,
        sort=args.sort,
        max_posts=args.max_posts,
        max_posts_per_source=args.max_posts_per_source,
        max_comments_per_post=args.max_comments,
        keep_scopes=keep_scopes,
        fetch_comments=not args.no_comments,
        request_delay=args.request_delay,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        dry_run=not args.persist,
        persist=args.persist,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    plan = plan_from_args(args)
    if plan.persist and not args.yes:
        parser.error("--persist requires --yes (permanent database writes)")

    try:
        plan.validate()
    except ValueError as exc:
        parser.error(str(exc))

    from app.services.ingestion.manager import IngestionManager

    manager = IngestionManager()
    session = None
    try:
        persistence = None
        if plan.persist and not plan.dry_run:
            from app.core.database import SessionLocal
            from app.services.ingestion.persistence import PersistenceService

            session = SessionLocal()
            persistence = PersistenceService(session)

        print(plan.describe())
        report = run_collection(
            manager,
            plan,
            persistence=persistence,
            echo=(None if args.json else print),
        )
    finally:
        if session is not None:
            session.close()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        for line in report.summary_lines():
            print(line)

    # A partial result is useful to a caller, but must not look like complete
    # success to shell automation. Preserve status 2 for an empty run.
    if report.errors:
        return 3
    if report.posts_kept == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
