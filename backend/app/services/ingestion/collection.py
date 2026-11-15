"""
Bounded Reddit collection workflow for the V1 research scope.

This module is deliberately a thin, guard-railed *driver* on top of the
existing ingestion architecture.  It does not fetch Reddit data itself - it
asks :class:`~app.services.ingestion.manager.IngestionManager` to run the
canonical provider operations (``search_posts`` / ``get_comments``) and only
adds the research-workflow concerns on top:

* a narrow, explicit V1 scope (IIM Sambalpur first, broader IIM/student
  discussions tagged but **not** silently included);
* two source modes so requests stay targeted:

  - **direct** communities (``IIMSambalpur`` and its undergraduate subreddit):
    bounded *recent-post* retrieval with **no** query filter.  The community
    itself is the relevance signal, so the institution name does not have to
    appear in every post.
  - **context** communities (``MBAIndia``, ``CATPrep``, ``CATpreparation``):
    only ever a server-side ``query`` search for an institution-specific
    phrase, followed by a local target-relevance check.  These are never
    retrieved unfiltered.

* hard safety caps on how much is ever requested;
* dry-run by default, permanent persistence strictly opt-in;
* de-duplication (by Reddit id, across aliases and communities) and thread
  (parent/child) preservation;
* comments requested **only** for posts that passed the target-relevance
  filter;
* bounded retry/backoff for transient provider timeouts / rate limits;
* per-subreddit and overall reporting;
* partial failures reported per source instead of aborting the run.

Nothing here reads or prints credentials, database URLs or user-identifying
fields.  ``PostData`` / ``CommentData`` carry no author, so there is no
identifying field to strip.  Original post/comment text and source metadata
are preserved verbatim.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.services.ingestion.providers.base import safe_failure_metadata

# ---------------------------------------------------------------------------
# Safety caps
# ---------------------------------------------------------------------------

#: Hard ceilings.  A request for more than these is refused, not clamped: a
#: caller that thinks it wants 5000 posts has misunderstood the workflow, and
#: silently downgrading would hide that.
HARD_MAX_POSTS = 200
HARD_MAX_COMMENTS_PER_POST = 2000
HARD_MAX_RETRIES = 5
MAX_RETRY_DELAY_SECONDS = 30.0
# Thus a logical task waits at most 5 x 30 = 150 seconds across all retries.

# The CLI's supported five-community workflow fits within these limits:
# 5 sources x 4 aliases = 20 logical searches. Every logical search may be
# dispatched at most once plus HARD_MAX_RETRIES retries, so no valid plan can
# issue more than 20 x (5 + 1) = 120 provider search dispatches.
HARD_MAX_SOURCES = 5
HARD_MAX_ALIASES = 4
HARD_MAX_SEARCH_TASKS = 20
HARD_MAX_PROVIDER_SEARCH_DISPATCHES = (
    HARD_MAX_SEARCH_TASKS * (HARD_MAX_RETRIES + 1)
)

#: Bounded defaults.  The first live preview stays small on purpose.
DEFAULT_MAX_POSTS = 5
DEFAULT_MAX_COMMENTS_PER_POST = 100

#: V1 focus.  Only these scopes are collected by default; others are counted
#: and reported so the research scope is never broadened behind the operator's
#: back.
DEFAULT_KEEP_SCOPES: Tuple[str, ...] = ("iim_sambalpur",)

RELEVANCE_SCOPES: Tuple[str, ...] = (
    "iim_sambalpur",
    "iim_broad",
    "student_broad",
    "other",
)

_SAMBALPUR_SUBREDDITS = {
    "iimsambalpur",
    "iim_sambalpur",
    "iim-sambalpur",
    "iimsambalpurug",
}

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

#: Direct (institution) communities: retrieve bounded recent posts with no
#: query filter.  The subreddit itself is the relevance signal.
DEFAULT_DIRECT_SUBREDDITS: Tuple[str, ...] = ("IIMSambalpur", "IIMSambalpurUG")
_APPROVED_DIRECT_SUBREDDITS = frozenset(
    subreddit.casefold() for subreddit in DEFAULT_DIRECT_SUBREDDITS
)

#: Context communities: only searched with an institution-specific phrase.
DEFAULT_CONTEXT_SUBREDDITS: Tuple[str, ...] = (
    "MBAIndia",
    "CATPrep",
    "CATpreparation",
)

#: Institution-specific target phrases.  Deliberately never includes a bare
#: ``IIM``, ``MBA`` or ``Sambalpur`` - those match too much and would drag in
#: unrelated discussion.
DEFAULT_TARGET_ALIASES: Tuple[str, ...] = (
    "IIM Sambalpur",
    "IIMSambalpur",
    "IIM-Sambalpur",
    "IIM Sambalpur UG",
)

#: Normalised forms that are too generic to be a target query on their own.
FORBIDDEN_TARGET_TERMS = frozenset({"iim", "mba", "sambalpur"})


class SourceMode(str, Enum):
    """How a community is searched."""

    DIRECT = "direct"
    CONTEXT = "context"


@dataclass
class CollectionSource:
    """One community and the mode used to search it."""

    subreddit: str
    mode: SourceMode = SourceMode.CONTEXT
    #: Context only.  Empty means "use the plan's target aliases".
    terms: Tuple[str, ...] = ()


def _normalize_term_key(term: str) -> str:
    """Case/separator-insensitive key used to reject generic target terms."""
    return re.sub(r"[^a-z0-9]+", "", (term or "").lower())


def _alias_to_pattern(alias: str) -> str:
    """Build a separator-tolerant regex fragment for one alias.

    ``IIM Sambalpur``, ``IIMSambalpur`` and ``IIM-Sambalpur`` all collapse to
    the same fragment; ``IIM Sambalpur UG`` additionally requires ``UG``.
    """
    parts = re.findall(r"[a-z0-9]+", (alias or "").lower())
    if not parts:
        raise ValueError(f"target alias {alias!r} has no alphanumeric content")
    return r"[\s\-_]*".join(parts)


def _compile_target_pattern(aliases: Iterable[str]) -> "re.Pattern[str]":
    fragments: List[str] = []
    for alias in aliases:
        fragment = _alias_to_pattern(alias)
        if fragment not in fragments:
            fragments.append(fragment)
    if not fragments:
        raise ValueError("at least one target alias is required")
    # Longest first so multi-word aliases win over their own prefixes.
    fragments.sort(key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(fragments) + r")\b", re.IGNORECASE)


_DEFAULT_TARGET_PATTERN = _compile_target_pattern(DEFAULT_TARGET_ALIASES)


def is_target_relevant(
    text: str,
    aliases: Sequence[str] = DEFAULT_TARGET_ALIASES,
) -> bool:
    """Return True only when *text* mentions an institution-specific alias.

    This is intentionally stricter than :func:`classify_relevance`: it requires
    an actual ``IIM Sambalpur``-style phrase (with flexible ``space``/``-``/
    ``_`` separators).  A bare ``IIM``, ``MBA`` or ``Sambalpur`` is **not** a
    match, so context-community results are validated locally before being
    retained.
    """
    if tuple(aliases) == DEFAULT_TARGET_ALIASES:
        pattern = _DEFAULT_TARGET_PATTERN
    else:
        pattern = _compile_target_pattern(tuple(aliases))
    return bool(pattern.search(text or ""))


# ---------------------------------------------------------------------------
# Relevance scoping
# ---------------------------------------------------------------------------


def classify_relevance(text: str, subreddit: Optional[str] = None) -> str:
    """Bucket input text into one exclusive research-scope category.

    The tiers are ordered from most to least directly relevant to the V1
    question ("what are IIM Sambalpur students discussing?"):

    ``iim_sambalpur``
        Lowercased text plus subreddit contains the literal substring
        ``"sambalpur"``, or the normalized subreddit exactly matches a member
        of ``_SAMBALPUR_SUBREDDITS``.
    ``iim_broad``
        If the first category did not match: text plus subreddit contains the
        literal substring ``"indian institute of management"`` or ``"iim"``.
    ``student_broad``
        If neither earlier category matched: text plus subreddit contains one
        of ``"cat "``, ``" cat"``, ``"mba"``, ``"student"``, ``"placement"``
        or ``"campus"``.
    ``other``
        Everything else.

    This is ordered substring matching, not semantic classification. In a
    collection report, ``relevance_counts`` counts eligible context/legacy
    search results after missing-ID rejection and after duplicate IDs already
    retained have been skipped, but before target gating and keep-scope
    filtering. A repeated out-of-scope result may be counted more than once,
    because it was never added to the retained-ID set. Direct source results
    are not included. Broad categories are counted even when later excluded.
    """
    blob = f"{text or ''} {subreddit or ''}".lower()
    sub = (subreddit or "").strip().lower()
    if "sambalpur" in blob or sub in _SAMBALPUR_SUBREDDITS:
        return "iim_sambalpur"
    if "indian institute of management" in blob or "iim" in blob:
        return "iim_broad"
    if any(
        keyword in blob
        for keyword in ("cat ", " cat", "mba", "student", "placement", "campus")
    ):
        return "student_broad"
    return "other"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _validate_target_terms(terms: Sequence[str], *, kind: str) -> None:
    if not terms:
        raise ValueError(f"at least one {kind} is required")
    for term in terms:
        key = _normalize_term_key(term)
        if not key:
            raise ValueError(f"empty {kind} is not allowed")
        if key in FORBIDDEN_TARGET_TERMS:
            raise ValueError(
                f"{kind} {term!r} is too generic; use an institution-specific "
                f"phrase such as {list(DEFAULT_TARGET_ALIASES)}"
            )


def _logical_search_task_count(
    sources: Sequence[CollectionSource],
    subreddits: Sequence[str],
    terms: Sequence[str],
    aliases: Sequence[str],
) -> int:
    if sources:
        return sum(
            1 if source.mode == SourceMode.DIRECT else len(source.terms or aliases)
            for source in sources
        )
    return len(subreddits) * len(terms)


@dataclass
class CollectionPlan:
    """A validated description of one bounded collection run.

    Two equivalent ways to describe the scope:

    * ``sources`` - explicit direct/context communities (preferred); or
    * ``subreddits`` + ``terms`` - the legacy flat form, where every subreddit
      is searched with every term (kept for backwards compatibility).
    """

    subreddits: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    sources: List[CollectionSource] = field(default_factory=list)
    aliases: Tuple[str, ...] = DEFAULT_TARGET_ALIASES
    after: str = "30d"
    before: Optional[str] = None
    sort: str = "desc"
    max_posts: int = DEFAULT_MAX_POSTS
    max_posts_per_source: Optional[int] = None
    max_comments_per_post: int = DEFAULT_MAX_COMMENTS_PER_POST
    provider: str = "arctic_shift"
    keep_scopes: Tuple[str, ...] = DEFAULT_KEEP_SCOPES
    fetch_comments: bool = True
    request_delay: float = 1.0
    timeout_seconds: Optional[float] = None
    max_retries: int = 0
    retry_backoff: float = 2.0
    dry_run: bool = True
    persist: bool = False

    @property
    def maximum_provider_search_dispatches(self) -> int:
        """Upper bound for this plan, valid after :meth:`validate` succeeds."""
        return _logical_search_task_count(
            self.sources, self.subreddits, self.terms, self.aliases
        ) * (self.max_retries + 1)

    def validate(self) -> None:
        """Refuse unsafe or malformed plans with an actionable message."""
        source_count = len(self.sources) if self.sources else len(self.subreddits)
        if source_count > HARD_MAX_SOURCES:
            raise ValueError(f"at most {HARD_MAX_SOURCES} sources are allowed")
        if len(self.aliases) > HARD_MAX_ALIASES:
            raise ValueError(
                f"at most {HARD_MAX_ALIASES} search aliases are allowed"
            )

        if self.sources:
            for source in self.sources:
                if not isinstance(source.subreddit, str) or not source.subreddit.strip():
                    raise ValueError("every collection source needs a subreddit")
                if source.mode == SourceMode.DIRECT:
                    if source.subreddit.casefold() not in _APPROVED_DIRECT_SUBREDDITS:
                        approved = ", ".join(DEFAULT_DIRECT_SUBREDDITS)
                        raise ValueError(
                            f"direct mode is limited to approved direct communities: "
                            f"{approved}; use context mode for broader communities"
                        )
                elif source.mode == SourceMode.CONTEXT:
                    terms = source.terms or self.aliases
                    if len(terms) > HARD_MAX_ALIASES:
                        raise ValueError(
                            f"at most {HARD_MAX_ALIASES} search aliases are allowed "
                            "per context source"
                        )
                    _validate_target_terms(
                        terms, kind="context search term"
                    )
                elif source.mode != SourceMode.DIRECT:
                    raise ValueError(f"unknown source mode {source.mode!r}")
        else:
            if not self.subreddits:
                raise ValueError("at least one subreddit is required")
            if not self.terms:
                raise ValueError("at least one search term is required")
            if len(self.terms) > HARD_MAX_ALIASES:
                raise ValueError(
                    f"at most {HARD_MAX_ALIASES} search aliases are allowed"
                )
            _validate_target_terms(self.terms, kind="search term")

        _validate_target_terms(self.aliases, kind="target alias")

        if self.persist and self.dry_run:
            raise ValueError(
                "persist=True is incompatible with dry_run=True; an explicit "
                "non-dry run is required to write to the database"
            )
        if not 1 <= self.max_posts <= HARD_MAX_POSTS:
            raise ValueError(
                f"max_posts must be between 1 and the hard cap "
                f"{HARD_MAX_POSTS} (got {self.max_posts})"
            )
        if self.max_posts_per_source is not None and not (
            1 <= self.max_posts_per_source <= self.max_posts
        ):
            raise ValueError(
                f"max_posts_per_source must be between 1 and max_posts "
                f"({self.max_posts}); got {self.max_posts_per_source}"
            )
        if not 1 <= self.max_comments_per_post <= HARD_MAX_COMMENTS_PER_POST:
            raise ValueError(
                f"max_comments_per_post must be between 1 and the hard cap "
                f"{HARD_MAX_COMMENTS_PER_POST} (got {self.max_comments_per_post})"
            )
        if self.request_delay < 0:
            raise ValueError("request_delay cannot be negative")
        if not 0 <= self.max_retries <= HARD_MAX_RETRIES:
            raise ValueError(
                f"max_retries must be between 0 and {HARD_MAX_RETRIES}"
            )
        if not 0 <= self.retry_backoff <= MAX_RETRY_DELAY_SECONDS:
            raise ValueError(
                f"retry_backoff must be between 0 and {MAX_RETRY_DELAY_SECONDS:g} seconds"
            )
        task_count = _logical_search_task_count(
            self.sources, self.subreddits, self.terms, self.aliases
        )
        if task_count > HARD_MAX_SEARCH_TASKS:
            raise ValueError(
                f"at most {HARD_MAX_SEARCH_TASKS} logical search tasks are allowed"
            )
        if self.maximum_provider_search_dispatches > HARD_MAX_PROVIDER_SEARCH_DISPATCHES:
            raise ValueError(
                "plan exceeds the hard limit of "
                f"{HARD_MAX_PROVIDER_SEARCH_DISPATCHES} provider search dispatches"
            )
        unknown = [s for s in self.keep_scopes if s not in RELEVANCE_SCOPES]
        if unknown:
            raise ValueError(
                f"unknown keep scope(s) {unknown}; valid: {list(RELEVANCE_SCOPES)}"
            )

    def describe(self) -> str:
        mode = "PERSIST" if (self.persist and not self.dry_run) else "DRY-RUN"
        if self.sources:
            scope = f"sources={len(self.sources)}"
        else:
            scope = f"subreddits={len(self.subreddits)} terms={len(self.terms)}"
        return (
            f"[{mode}] {scope} aliases={len(self.aliases)} "
            f"after={self.after} before={self.before} sort={self.sort} "
            f"max_posts={self.max_posts} "
            f"max_posts_per_source={self.max_posts_per_source} "
            f"max_comments_per_post={self.max_comments_per_post} "
            f"keep_scopes={list(self.keep_scopes)} "
            f"max_retries={self.max_retries} "
            f"max_search_dispatches={self.maximum_provider_search_dispatches}"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SourceReport:
    """Per-subreddit counts for one run - never includes credentials or PII.

    Query counters count logical subreddit/query jobs. ``queries_retried``
    counts jobs with at least one retry once; ``search_attempts`` counts every
    provider dispatch, including retries. Failure entries represent each
    failed dispatch, including failures recovered by a later retry.
    """

    subreddit: str
    mode: str
    queries_attempted: int = 0
    queries_succeeded: int = 0
    queries_failed: int = 0
    queries_retried: int = 0
    search_attempts: int = 0
    search_attempt_failures: List[Dict[str, Any]] = field(default_factory=list)
    posts_returned: int = 0
    posts_retained: int = 0
    posts_rejected: int = 0
    duplicates: int = 0
    comment_trees_requested: int = 0
    comments_returned: int = 0
    comments_retrieved: int = 0
    comments_rejected: int = 0
    duplicate_comments: int = 0
    comments_skipped_cap: int = 0
    comments_collected: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subreddit": self.subreddit,
            "mode": self.mode,
            "queries_attempted": self.queries_attempted,
            "queries_succeeded": self.queries_succeeded,
            "queries_failed": self.queries_failed,
            "queries_retried": self.queries_retried,
            "search_attempts": self.search_attempts,
            "search_attempt_failures": list(self.search_attempt_failures),
            "posts_returned": self.posts_returned,
            "posts_retained": self.posts_retained,
            "posts_rejected": self.posts_rejected,
            "duplicates": self.duplicates,
            "comment_trees_requested": self.comment_trees_requested,
            "comments_returned": self.comments_returned,
            "comments_retrieved": self.comments_retrieved,
            "comments_rejected": self.comments_rejected,
            "duplicate_comments": self.duplicate_comments,
            "comments_skipped_cap": self.comments_skipped_cap,
            "comments_collected": self.comments_collected,
            "errors": list(self.errors),
        }


def _format_search_attempt_failure(failure: Dict[str, Any]) -> str:
    status = failure["status_code"]
    status_part = f" status={status}" if status is not None else ""
    retry_part = " retrying=yes" if failure["retrying"] else " retrying=no"
    return (
        "search attempt failed "
        f"mode={failure['source_mode']} "
        f"subreddit={failure['subreddit']} "
        f"query_index={failure['query_index']} "
        f"attempt={failure['attempt']} "
        f"category={failure['failure_category']}"
        f"{status_part}{retry_part}"
    )


@dataclass
class CollectionReport:
    """Counts and failures for one run - never includes credentials or PII.

    Query counters count logical subreddit/query jobs. Each attempted job ends
    as one success or failure; ``queries_retried`` counts distinct jobs with
    at least one retry, while ``search_attempts`` counts all provider
    dispatches. Per-source counter sums reconcile with global counters.
    """

    dry_run: bool
    queries_attempted: int = 0
    queries_succeeded: int = 0
    queries_failed: int = 0
    queries_retried: int = 0
    search_attempts: int = 0
    search_attempt_failures: List[Dict[str, Any]] = field(default_factory=list)
    posts_fetched: int = 0
    posts_kept: int = 0
    duplicate_posts: int = 0
    out_of_scope_posts: int = 0
    posts_rejected: int = 0
    comment_trees_requested: int = 0
    comments_collected: int = 0
    comments_returned: int = 0
    comments_retrieved: int = 0
    comments_rejected: int = 0
    comments_skipped_cap: int = 0
    duplicate_comments: int = 0
    persisted_posts: int = 0
    relevance_counts: Dict[str, int] = field(default_factory=dict)
    per_source: Dict[str, SourceReport] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    preview: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "status": "partial_failure" if self.errors else "complete",
            "queries_attempted": self.queries_attempted,
            "queries_succeeded": self.queries_succeeded,
            "queries_failed": self.queries_failed,
            "queries_retried": self.queries_retried,
            "search_attempts": self.search_attempts,
            "search_attempt_failures": list(self.search_attempt_failures),
            "posts_returned": self.posts_fetched,
            "posts_retained": self.posts_kept,
            "posts_rejected": self.posts_rejected,
            "duplicates": self.duplicate_posts,
            "comment_trees_requested": self.comment_trees_requested,
            "comments_returned": self.comments_returned,
            "comments_retrieved": self.comments_retrieved,
            "comments_rejected": self.comments_rejected,
            "comments_skipped_cap": self.comments_skipped_cap,
            # legacy field names, retained for existing consumers
            "posts_fetched": self.posts_fetched,
            "posts_kept": self.posts_kept,
            "duplicate_posts": self.duplicate_posts,
            "out_of_scope_posts": self.out_of_scope_posts,
            "comments_collected": self.comments_collected,
            "duplicate_comments": self.duplicate_comments,
            "persisted_posts": self.persisted_posts,
            "relevance_counts": dict(self.relevance_counts),
            "per_source": {
                key: source.to_dict() for key, source in self.per_source.items()
            },
            "errors": list(self.errors),
            "preview": list(self.preview),
        }

    def summary_lines(self) -> List[str]:
        mode = "DRY-RUN (no writes)" if self.dry_run else "LIVE"
        lines = [
            f"Mode: {mode}",
            f"Queries attempted: {self.queries_attempted}",
            f"Queries succeeded: {self.queries_succeeded}",
            f"Queries failed: {self.queries_failed}",
            f"Queries retried (one or more retries): {self.queries_retried}",
            f"Search attempts including retries: {self.search_attempts}",
            f"Posts returned: {self.posts_fetched}",
            f"Posts kept (in scope): {self.posts_kept}",
            f"Posts rejected (irrelevant/out of scope): {self.posts_rejected}",
            f"Duplicate posts skipped: {self.duplicate_posts}",
            f"Out-of-scope posts skipped: {self.out_of_scope_posts}",
            f"Comment trees requested: {self.comment_trees_requested}",
            f"Comments returned by provider: {self.comments_returned}",
            f"Comments retrieved: {self.comments_retrieved}",
            f"Comments rejected (missing/invalid id): {self.comments_rejected}",
            f"Comments collected: {self.comments_collected}",
            f"Comments skipped by per-post cap: {self.comments_skipped_cap}",
            f"Duplicate comments skipped: {self.duplicate_comments}",
            f"Relevance: {dict(self.relevance_counts)}",
        ]
        if self.per_source:
            lines.append("Per source:")
            for key in sorted(self.per_source):
                source = self.per_source[key]
                lines.append(
                    f"  [{key}] mode={source.mode} "
                    f"queries_attempted={source.queries_attempted} "
                    f"succeeded={source.queries_succeeded} "
                    f"failed={source.queries_failed} "
                    f"retried={source.queries_retried} "
                    f"search_attempts={source.search_attempts} "
                    f"returned={source.posts_returned} "
                    f"retained={source.posts_retained} "
                    f"rejected={source.posts_rejected} "
                    f"duplicates={source.duplicates} "
                    f"comment_trees={source.comment_trees_requested} "
                    f"comments_returned={source.comments_returned} "
                    f"comments={source.comments_retrieved} "
                    f"comments_rejected={source.comments_rejected} "
                    f"comments_duplicates={source.duplicate_comments} "
                    f"comments_collected={source.comments_collected} "
                    f"comments_capped={source.comments_skipped_cap}"
                )
        if self.persist_requested():
            lines.append(f"Posts persisted: {self.persisted_posts}")
        if self.errors:
            lines.append(f"Partial failures ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        else:
            lines.append("Partial failures: none")
        if self.search_attempt_failures:
            lines.append(f"Failed search attempts ({len(self.search_attempt_failures)}):")
            for failure in self.search_attempt_failures:
                lines.append(f"  - {_format_search_attempt_failure(failure)}")
        return lines

    def persist_requested(self) -> bool:
        return not self.dry_run


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _SearchTask:
    """A single bounded provider search (one subreddit / one query)."""

    subreddit: str
    mode: str
    query: Optional[str]
    gate_target: bool
    retain_all: bool
    source_key: str
    aliases: Tuple[str, ...]
    query_index: int


def _build_search_tasks(plan: CollectionPlan) -> List[_SearchTask]:
    tasks: List[_SearchTask] = []
    query_indices: Dict[str, int] = {}

    def next_index(source_key: str) -> int:
        query_indices[source_key] = query_indices.get(source_key, 0) + 1
        return query_indices[source_key]

    if plan.sources:
        for source in plan.sources:
            if source.mode == SourceMode.DIRECT:
                # Direct communities are retrieved unfiltered: the community is
                # the relevance signal, not the text.
                tasks.append(
                    _SearchTask(
                        subreddit=source.subreddit,
                        mode=source.mode.value,
                        query=None,
                        gate_target=False,
                        retain_all=True,
                        source_key=source.subreddit,
                        aliases=plan.aliases,
                        query_index=next_index(source.subreddit),
                    )
                )
            else:
                for term in (source.terms or plan.aliases):
                    tasks.append(
                        _SearchTask(
                            subreddit=source.subreddit,
                            mode=source.mode.value,
                            query=term,
                            gate_target=True,
                            retain_all=False,
                            source_key=source.subreddit,
                            aliases=plan.aliases,
                            query_index=next_index(source.subreddit),
                        )
                    )
    else:
        # Legacy flat form: every subreddit searched with every term, filtered
        # by scope only (no explicit target gate) to preserve prior behaviour.
        for subreddit in plan.subreddits:
            for term in plan.terms:
                tasks.append(
                    _SearchTask(
                        subreddit=subreddit,
                        mode=SourceMode.CONTEXT.value,
                        query=term,
                        gate_target=False,
                        retain_all=False,
                        source_key=subreddit,
                        aliases=plan.aliases,
                        query_index=next_index(subreddit),
                    )
                )
    return tasks


def _source_report(
    report: CollectionReport, key: str, subreddit: str, mode: str
) -> SourceReport:
    source = report.per_source.get(key)
    if source is None:
        source = SourceReport(subreddit=subreddit, mode=mode)
        report.per_source[key] = source
    return source


def _dedupe_comment_ids(comments: Iterable[Any]) -> Tuple[List[Any], int, int]:
    seen: set[str] = set()
    unique: List[Any] = []
    duplicates = 0
    rejected = 0
    for comment in comments:
        reddit_id = getattr(comment, "reddit_id", None)
        if not isinstance(reddit_id, str) or not reddit_id.strip():
            rejected += 1
            continue
        if reddit_id in seen:
            duplicates += 1
            continue
        seen.add(reddit_id)
        unique.append(comment)
    return unique, duplicates, rejected


def _is_retryable(exc: BaseException) -> bool:
    """Conservatively decide whether an error is worth a bounded retry.

    Only clearly transient failures are retried: timeouts, connection drops and
    rate-limiting responses.  Logic errors and validation errors are surfaced
    immediately.
    """
    category, status = safe_failure_metadata(exc)
    if category in {"timeout", "connection", "connect_timeout", "read_timeout"}:
        return True
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return True
    exc_name = type(exc).__name__.lower()
    if exc_name in {
        "timeout",
        "readtimeout",
        "connecttimeout",
        "connectionerror",
        "connectionreseterror",
    }:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "slow down",
            "too many requests",
            "rate limit",
            "429",
            "503",
        )
    )


def _call_with_retries(
    manager: Any,
    plan: CollectionPlan,
    operation: str,
    emit: Callable[[str], None],
    sleep: Callable[[float], None],
    on_attempt: Optional[Callable[[int], None]] = None,
    on_failure: Optional[Callable[[int, BaseException, bool], None]] = None,
    on_retry: Optional[Callable[[], None]] = None,
    **kwargs: Any,
) -> Any:
    attempts = plan.max_retries + 1
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        attempt_number = attempt + 1
        if on_attempt is not None:
            on_attempt(attempt_number)
        try:
            return manager.execute_operation(
                operation, provider=plan.provider, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - retried or re-raised
            last_exc = exc
            will_retry = attempt + 1 < attempts and _is_retryable(exc)
            if on_failure is not None:
                on_failure(attempt_number, exc, will_retry)
            if not will_retry:
                raise
            if on_retry is not None:
                on_retry()
            delay = min(
                plan.retry_backoff * (2 ** attempt), MAX_RETRY_DELAY_SECONDS
            )
            emit(f"retry {operation} in {delay:.1f}s after {type(exc).__name__}")
            if delay:
                sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises


def run_collection(
    manager: Any,
    plan: CollectionPlan,
    persistence: Optional[Any] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    echo: Optional[Callable[[str], None]] = None,
) -> CollectionReport:
    """Execute *plan* through the canonical ingestion manager.

    Args:
        manager: an :class:`IngestionManager` (or any object exposing
            ``execute_operation(operation, provider=..., **kwargs)``).
        plan: the validated :class:`CollectionPlan`.
        persistence: optional object with ``save_post_and_comments(post)``.
            Required when ``plan.persist`` is set; ignored in dry-run.
        sleeper: injectable sleep (defaults to :func:`time.sleep`) so tests do
            not actually wait between requests.
        echo: injectable progress sink (defaults to no-op).

    Returns:
        A :class:`CollectionReport`.  Provider failures are captured in
        ``report.errors`` and never abort the whole run.
    """
    plan.validate()
    if plan.persist and not plan.dry_run and persistence is None:
        raise ValueError("persist was requested but no persistence service supplied")

    sleep = sleeper if sleeper is not None else time.sleep
    emit = echo if echo is not None else (lambda _msg: None)

    report = CollectionReport(dry_run=plan.dry_run)
    for scope in RELEVANCE_SCOPES:
        report.relevance_counts.setdefault(scope, 0)

    posts: "Dict[str, Tuple[Any, str, str]]" = {}
    retained_per_source: Dict[str, int] = {}

    for task in _build_search_tasks(plan):
        if len(posts) >= plan.max_posts:
            break
        if (
            plan.max_posts_per_source is not None
            and retained_per_source.get(task.source_key, 0) >= plan.max_posts_per_source
        ):
            continue

        source = _source_report(report, task.source_key, task.subreddit, task.mode)
        source.queries_attempted += 1
        report.queries_attempted += 1
        emit(f"search request started mode={task.mode}")
        retry_count = 0
        last_failure: Optional[Dict[str, Any]] = None

        def on_search_attempt(_attempt_number: int) -> None:
            source.search_attempts += 1
            report.search_attempts += 1

        def on_search_failure(
            attempt_number: int, exc: BaseException, retrying: bool
        ) -> None:
            nonlocal last_failure
            category, status = safe_failure_metadata(exc)
            last_failure = {
                "source_mode": task.mode,
                "subreddit": task.subreddit,
                "query_index": task.query_index,
                "attempt": attempt_number,
                "failure_category": category,
                "status_code": status,
                "retrying": retrying,
            }
            report.search_attempt_failures.append(last_failure)
            source.search_attempt_failures.append(last_failure)
            emit(_format_search_attempt_failure(last_failure))

        def on_search_retry() -> None:
            nonlocal retry_count
            if retry_count == 0:
                report.queries_retried += 1
                source.queries_retried += 1
            retry_count += 1

        try:
            request_limit = plan.max_posts - len(posts)
            if plan.max_posts_per_source is not None:
                request_limit = min(
                    request_limit,
                    plan.max_posts_per_source
                    - retained_per_source.get(task.source_key, 0),
                )
            results = _call_with_retries(
                manager,
                plan,
                "search_posts",
                emit,
                sleep,
                on_attempt=on_search_attempt,
                on_failure=on_search_failure,
                on_retry=on_search_retry,
                subreddit=task.subreddit,
                query=task.query,
                limit=request_limit,
                after=plan.after,
                before=plan.before,
                sort=plan.sort,
                timeout=plan.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            report.queries_failed += 1
            source.queries_failed += 1
            category, status = safe_failure_metadata(exc)
            status_part = f" status={status}" if status is not None else ""
            used_attempts = last_failure["attempt"] if last_failure else 1
            message = (
                f"search failed mode={task.mode} subreddit={task.subreddit} "
                f"query_index={task.query_index} attempts={used_attempts} "
                f"category={category}{status_part}"
            )
            report.errors.append(message)
            source.errors.append(message)
            if plan.request_delay:
                sleep(plan.request_delay)
            continue

        report.queries_succeeded += 1
        source.queries_succeeded += 1

        for post in results or []:
            report.posts_fetched += 1
            source.posts_returned += 1
            reddit_id = getattr(post, "reddit_id", None)
            if not reddit_id:
                message = "search result without reddit_id rejected"
                report.errors.append(message)
                source.errors.append(message)
                report.posts_rejected += 1
                source.posts_rejected += 1
                continue
            if reddit_id in posts:
                report.duplicate_posts += 1
                source.duplicates += 1
                continue

            reported_subreddit = getattr(post, "subreddit", None)
            if task.retain_all:
                if (
                    not isinstance(reported_subreddit, str)
                    or not reported_subreddit.strip()
                    or reported_subreddit.strip().casefold()
                    != task.subreddit.casefold()
                ):
                    # Direct-source provenance must be explicit; never infer
                    # institutional evidence from the requested source.
                    report.posts_rejected += 1
                    source.posts_rejected += 1
                    continue
                post_subreddit = reported_subreddit
            else:
                # Preserve the established context-mode fallback behavior.
                post_subreddit = reported_subreddit or task.subreddit
            text = f"{getattr(post, 'title', '')} {getattr(post, 'body', '') or ''}"

            if task.retain_all:
                # Direct community: retain without requiring the name in text.
                relevance = "iim_sambalpur"
            else:
                relevance = classify_relevance(text, post_subreddit)
                report.relevance_counts[relevance] = (
                    report.relevance_counts.get(relevance, 0) + 1
                )
                if task.gate_target and not is_target_relevant(text, task.aliases):
                    report.posts_rejected += 1
                    source.posts_rejected += 1
                    continue

            if not task.retain_all and relevance not in plan.keep_scopes:
                report.out_of_scope_posts += 1
                report.posts_rejected += 1
                source.posts_rejected += 1
                continue

            if len(posts) >= plan.max_posts:
                break
            if (
                plan.max_posts_per_source is not None
                and retained_per_source.get(task.source_key, 0)
                >= plan.max_posts_per_source
            ):
                break

            posts[reddit_id] = (post, relevance, task.source_key)
            retained_per_source[task.source_key] = (
                retained_per_source.get(task.source_key, 0) + 1
            )

        if plan.request_delay:
            sleep(plan.request_delay)

    for reddit_id, (post, relevance, source_key) in posts.items():
        source = report.per_source.get(source_key)
        comments: List[Any] = []
        if plan.fetch_comments:
            if source is not None:
                source.comment_trees_requested += 1
            report.comment_trees_requested += 1
            emit("comments request started")
            try:
                comments = _call_with_retries(
                    manager,
                    plan,
                    "get_comments",
                    emit,
                    sleep,
                    post_id=reddit_id,
                    timeout=plan.timeout_seconds,
                    limit=plan.max_comments_per_post,
                ) or []
            except Exception as exc:  # noqa: BLE001 - partial failure
                message = f"comments request failed ({type(exc).__name__})"
                report.errors.append(message)
                if source is not None:
                    source.errors.append(message)
                comments = []
            comments = list(comments)
            returned_comments = len(comments)
            comments, duplicate_comments, rejected_comments = _dedupe_comment_ids(comments)
            report.comments_returned += returned_comments
            report.duplicate_comments += duplicate_comments
            report.comments_rejected += rejected_comments
            report.comments_retrieved += len(comments)
            if source is not None:
                source.comments_returned += returned_comments
                source.duplicate_comments += duplicate_comments
                source.comments_rejected += rejected_comments
                source.comments_retrieved += len(comments)
            if len(comments) > plan.max_comments_per_post:
                skipped_by_cap = len(comments) - plan.max_comments_per_post
                report.comments_skipped_cap += skipped_by_cap
                if source is not None:
                    source.comments_skipped_cap += skipped_by_cap
                comments = comments[: plan.max_comments_per_post]
            if plan.request_delay:
                sleep(plan.request_delay)

        post.comments = comments
        report.comments_collected += len(comments)
        if source is not None:
            source.comments_collected += len(comments)
        report.preview.append(
            {
                "reddit_id": reddit_id,
                "subreddit": getattr(post, "subreddit", ""),
                "relevance": relevance,
                "source_mode": source.mode if source is not None else "",
                "num_comments": len(comments),
            }
        )

    report.posts_kept = len(posts)
    for key, retained in retained_per_source.items():
        source = report.per_source.get(key)
        if source is not None:
            source.posts_retained = retained

    if plan.persist and not plan.dry_run:
        for reddit_id, (post, _relevance, _source_key) in posts.items():
            try:
                persistence.save_post_and_comments(post)
                report.persisted_posts += 1
            except Exception as exc:  # noqa: BLE001 - partial failure
                report.errors.append(f"persistence failed ({type(exc).__name__})")

    return report
