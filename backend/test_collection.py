"""
Tests for the bounded Reddit collection workflow.

All provider calls are faked: no network, no database.  The tests pin the
safety behaviour that matters (dry-run default, explicit persistence, hard
caps, de-duplication, scope discipline, partial-failure reporting).
"""

from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.ingestion.collection import (
    DEFAULT_CONTEXT_SUBREDDITS,
    DEFAULT_DIRECT_SUBREDDITS,
    DEFAULT_KEEP_SCOPES,
    DEFAULT_TARGET_ALIASES,
    HARD_MAX_COMMENTS_PER_POST,
    HARD_MAX_POSTS,
    HARD_MAX_PROVIDER_SEARCH_DISPATCHES,
    HARD_MAX_RETRIES,
    HARD_MAX_SEARCH_TASKS,
    CollectionReport,
    CollectionPlan,
    CollectionSource,
    SourceMode,
    classify_relevance,
    is_target_relevant,
    run_collection,
)
from app.services.ingestion.providers.base import ProviderOperationError
import collect_reddit


def _post(reddit_id, title="t", body="", subreddit="IIMSambalpur"):
    return SimpleNamespace(
        reddit_id=reddit_id,
        title=title,
        body=body,
        subreddit=subreddit,
        comments=[],
    )


def _comment(reddit_id, parent=None):
    return SimpleNamespace(
        reddit_id=reddit_id,
        body=f"body {reddit_id}",
        parent_reddit_id=parent,
    )


class FakeManager:
    """Records calls and returns configured results per operation."""

    def __init__(self, posts=None, comments=None, fail_search=False, fail_comments=False):
        self._posts = posts if posts is not None else []
        self._comments = comments if comments is not None else []
        self.fail_search = fail_search
        self.fail_comments = fail_comments
        self.calls = []

    def execute_operation(self, operation, provider=None, **kwargs):
        self.calls.append((operation, provider, kwargs))
        if operation == "search_posts":
            if self.fail_search:
                raise RuntimeError("search exploded")
            return list(self._posts)
        if operation == "get_comments":
            if self.fail_comments:
                raise RuntimeError("comments exploded")
            return list(self._comments)
        raise AssertionError(f"unexpected operation {operation}")


class FakePersistence:
    def __init__(self, fail_for=()):
        self.saved = []
        self.fail_for = set(fail_for)

    def save_post_and_comments(self, post):
        if post.reddit_id in self.fail_for:
            raise RuntimeError("persist exploded")
        self.saved.append(post.reddit_id)


class TestClassification(unittest.TestCase):
    def test_sambalpur_is_top_scope(self):
        self.assertEqual(
            classify_relevance("Is IIM Sambalpur worth it?", "CATpreparation"),
            "iim_sambalpur",
        )

    def test_iim_subreddit_is_top_scope(self):
        self.assertEqual(
            classify_relevance("hello", "IIMSambalpur"), "iim_sambalpur"
        )

    def test_iim_broad(self):
        self.assertEqual(
            classify_relevance("Which IIM is best", "Indian_Academia"),
            "iim_broad",
        )

    def test_student_broad(self):
        self.assertEqual(
            classify_relevance("CAT prep tips", "CATpreparation"),
            "student_broad",
        )

    def test_other(self):
        self.assertEqual(classify_relevance("weather today", "india"), "other")


class TestPlanValidation(unittest.TestCase):
    def _plan(self, **kwargs):
        base = dict(subreddits=["IIMSambalpur"], terms=["IIM Sambalpur"])
        base.update(kwargs)
        return CollectionPlan(**base)

    def test_defaults_are_safe(self):
        plan = self._plan()
        self.assertTrue(plan.dry_run)
        self.assertFalse(plan.persist)
        plan.validate()

    def test_persist_with_dry_run_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(persist=True, dry_run=True).validate()

    def test_missing_subreddits_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(subreddits=[]).validate()

    def test_missing_terms_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(terms=[]).validate()

    def test_max_posts_hard_cap(self):
        with self.assertRaises(ValueError):
            self._plan(max_posts=HARD_MAX_POSTS + 1).validate()

    def test_max_comments_hard_cap(self):
        with self.assertRaises(ValueError):
            self._plan(max_comments_per_post=HARD_MAX_COMMENTS_PER_POST + 1).validate()

    def test_unknown_keep_scope_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(keep_scopes=("not_a_scope",)).validate()

    def test_max_retries_has_hard_ceiling(self):
        with self.assertRaises(ValueError):
            self._plan(max_retries=HARD_MAX_RETRIES + 1).validate()

    def test_retry_backoff_has_hard_ceiling(self):
        with self.assertRaises(ValueError):
            self._plan(retry_backoff=31).validate()


class TestRunCollection(unittest.TestCase):
    def _plan(self, **kwargs):
        base = dict(
            subreddits=["IIMSambalpur"],
            terms=["IIM Sambalpur"],
            max_posts=5,
            max_comments_per_post=2,
            request_delay=0.0,
        )
        base.update(kwargs)
        return CollectionPlan(**base)

    def test_dry_run_never_persists(self):
        manager = FakeManager(posts=[_post("p1"), _post("p2")])
        persistence = FakePersistence()
        report = run_collection(manager, self._plan(), persistence=persistence)
        self.assertEqual(report.posts_kept, 2)
        self.assertEqual(persistence.saved, [])
        self.assertTrue(report.dry_run)
        self.assertEqual(report.persisted_posts, 0)

    def test_explicit_persist_writes(self):
        manager = FakeManager(posts=[_post("p1"), _post("p2")])
        persistence = FakePersistence()
        plan = self._plan(dry_run=False, persist=True, fetch_comments=False)
        report = run_collection(manager, plan, persistence=persistence)
        self.assertEqual(report.persisted_posts, 2)
        self.assertEqual(sorted(persistence.saved), ["p1", "p2"])

    def test_persist_without_service_raises(self):
        manager = FakeManager(posts=[_post("p1")])
        plan = self._plan(dry_run=False, persist=True)
        with self.assertRaises(ValueError):
            run_collection(manager, plan, persistence=None)

    def test_dedup_posts_by_reddit_id(self):
        manager = FakeManager(posts=[_post("p1"), _post("p1"), _post("p2")])
        report = run_collection(manager, self._plan(), persistence=None)
        self.assertEqual(report.posts_kept, 2)
        self.assertEqual(report.duplicate_posts, 1)

    def test_out_of_scope_excluded_but_counted(self):
        manager = FakeManager(
            posts=[
                _post("p1", title="IIM Sambalpur placement"),
                _post("p2", title="Which IIM is best", subreddit="Indian_Academia"),
            ]
        )
        report = run_collection(manager, self._plan(), persistence=None)
        self.assertEqual(report.posts_kept, 1)
        self.assertEqual(report.out_of_scope_posts, 1)
        self.assertEqual(report.relevance_counts["iim_sambalpur"], 1)
        self.assertEqual(report.relevance_counts["iim_broad"], 1)

    def test_include_scope_keeps_broader(self):
        manager = FakeManager(
            posts=[_post("p2", title="Which IIM is best", subreddit="Indian_Academia")]
        )
        plan = self._plan(keep_scopes=tuple(DEFAULT_KEEP_SCOPES) + ("iim_broad",))
        report = run_collection(manager, plan, persistence=None)
        self.assertEqual(report.posts_kept, 1)

    def test_comments_deduped_and_capped(self):
        manager = FakeManager(
            posts=[_post("p1")],
            comments=[_comment("c1"), _comment("c2"), _comment("c1"), _comment("c3")],
        )
        report = run_collection(manager, self._plan(), persistence=None)
        self.assertEqual(report.comments_collected, 2)  # capped at 2
        self.assertEqual(report.comments_retrieved, 3)
        self.assertEqual(report.comments_skipped_cap, 1)
        self.assertEqual(report.duplicate_comments, 1)
        self.assertEqual(report.per_source["IIMSambalpur"].comments_skipped_cap, 1)
        self.assertEqual(report.comments_returned, 4)
        self.assertEqual(report.comments_rejected, 0)
        self.assertEqual(
            report.comments_returned,
            report.duplicate_comments + report.comments_rejected + report.comments_retrieved,
        )
        self.assertEqual(
            report.comments_retrieved,
            report.comments_collected + report.comments_skipped_cap,
        )
        source = report.per_source["IIMSambalpur"]
        self.assertEqual(source.comments_returned, 4)
        self.assertEqual(source.duplicate_comments, 1)
        self.assertEqual(source.comments_collected, 2)

    def test_comment_accounting_rejects_missing_ids_and_reconciles_per_source(self):
        plan = self._plan(
            sources=[
                CollectionSource("IIMSambalpur", SourceMode.DIRECT),
                CollectionSource("IIMSambalpurUG", SourceMode.DIRECT),
            ],
            max_comments_per_post=1,
        )

        class SourceCommentsManager(FakeManager):
            def execute_operation(self, operation, provider=None, **kwargs):
                self.calls.append((operation, provider, kwargs))
                if operation == "search_posts":
                    if kwargs["subreddit"] == "IIMSambalpur":
                        return [_post("p1", subreddit="IIMSambalpur")]
                    return [_post("p2", subreddit="IIMSambalpurUG")]
                if operation == "get_comments":
                    if kwargs["post_id"] == "p1":
                        return [_comment("c1"), _comment("c1"), _comment(None), _comment("c2")]
                    return [_comment("c3")]
                raise AssertionError(operation)

        report = run_collection(SourceCommentsManager(), plan)
        self.assertEqual(report.comments_returned, 5)
        self.assertEqual(report.duplicate_comments, 1)
        self.assertEqual(report.comments_rejected, 1)
        self.assertEqual(report.comments_retrieved, 3)
        self.assertEqual(report.comments_skipped_cap, 1)
        self.assertEqual(report.comments_collected, 2)
        self.assertEqual(
            sum(s.comments_returned for s in report.per_source.values()),
            report.comments_returned,
        )
        self.assertEqual(
            sum(s.comments_collected for s in report.per_source.values()),
            report.comments_collected,
        )
        for field in (
            "comments_returned",
            "comments_retrieved",
            "comments_rejected",
            "duplicate_comments",
            "comments_skipped_cap",
            "comments_collected",
        ):
            self.assertEqual(
                sum(getattr(source, field) for source in report.per_source.values()),
                getattr(report, field),
                field,
            )
        for source in report.per_source.values():
            self.assertEqual(
                source.comments_returned,
                source.duplicate_comments + source.comments_rejected + source.comments_retrieved,
            )
            self.assertEqual(
                source.comments_retrieved,
                source.comments_collected + source.comments_skipped_cap,
            )

    def test_comment_request_is_bounded_by_plan_cap(self):
        manager = FakeManager(posts=[_post("p1")])
        run_collection(manager, self._plan(max_comments_per_post=7))
        request = next(call for call in manager.calls if call[0] == "get_comments")
        self.assertEqual(request[2]["limit"], 7)

    def test_provider_failure_secrets_are_not_reported_or_echoed(self):
        class SecretFailureManager(FakeManager):
            def execute_operation(self, operation, provider=None, **kwargs):
                if operation == "search_posts":
                    raise RuntimeError("Authorization: Bearer top-secret-token")
                return super().execute_operation(operation, provider, **kwargs)

        output = []
        report = run_collection(SecretFailureManager(), self._plan(), echo=output.append)
        combined = " ".join(report.errors + output + report.summary_lines())
        self.assertNotIn("top-secret-token", combined)
        self.assertEqual(report.to_dict()["status"], "partial_failure")

    def test_partial_alias_search_has_explicit_partial_status(self):
        plan = CollectionPlan(
            sources=[CollectionSource("CATpreparation", SourceMode.CONTEXT,
                                      ("IIM Sambalpur", "IIM Sambalpur MBA"))],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )

        class OneAliasFails(FakeManager):
            def execute_operation(self, operation, provider=None, **kwargs):
                if operation == "search_posts":
                    if kwargs["query"] == "IIM Sambalpur":
                        raise RuntimeError("alias failed")
                    return [_post("p1", title="IIM Sambalpur MBA", subreddit="CATpreparation")]
                return super().execute_operation(operation, provider, **kwargs)

        report = run_collection(OneAliasFails(), plan)
        self.assertEqual(report.posts_kept, 1)
        self.assertTrue(report.errors)
        self.assertEqual(report.to_dict()["status"], "partial_failure")
        self.assertEqual((report.queries_attempted, report.queries_succeeded,
                          report.queries_failed, report.queries_retried), (2, 1, 1, 0))
        self.assertEqual(report.search_attempts, 2)
        failure = report.search_attempt_failures[0]
        self.assertEqual(
            (failure["source_mode"], failure["subreddit"], failure["query_index"],
             failure["attempt"], failure["failure_category"], failure["retrying"]),
            ("context", "CATpreparation", 1, 1, "runtime_error", False),
        )
        summary = "\n".join(report.summary_lines())
        self.assertIn("mode=context subreddit=CATpreparation query_index=1 attempt=1", summary)
        self.assertNotIn("IIM Sambalpur", summary)
        source = report.per_source["CATpreparation"]
        self.assertEqual((source.queries_attempted, source.queries_succeeded,
                          source.queries_failed, source.queries_retried), (2, 1, 1, 0))

    def test_search_failure_reports_safe_category_and_http_status(self):
        class HttpFailure(RuntimeError):
            failure_category = "http_status"
            status_code = 503

            def __str__(self):
                return "https://secret.invalid/?token=never-show-this response body"

        class FailingManager(FakeManager):
            def execute_operation(self, operation, provider=None, **kwargs):
                if operation == "search_posts":
                    raise HttpFailure()
                return super().execute_operation(operation, provider, **kwargs)

        plan = self._plan(sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
                          max_retries=0, fetch_comments=False)
        report = run_collection(FailingManager(), plan)
        failure = report.search_attempt_failures[0]
        self.assertEqual((failure["failure_category"], failure["status_code"]),
                         ("http_status", 503))
        self.assertNotIn("never-show-this", str(report.to_dict()))
        self.assertNotIn("secret.invalid", str(report.to_dict()))

    def test_relevance_counts_measure_results_before_scope_exclusion(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource("IIMSambalpur", SourceMode.DIRECT),
                CollectionSource("AskReddit", SourceMode.CONTEXT, ("IIM Sambalpur",)),
            ],
            max_posts=10,
            request_delay=0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={
                ("IIMSambalpur", None): [_post("direct", title="No keywords")],
                ("AskReddit", "IIM Sambalpur"): [
                    _post("target", title="IIM Sambalpur campus", subreddit="AskReddit"),
                    _post("broad", title="Which IIM is best", subreddit="AskReddit"),
                    _post("student", title="CAT preparation", subreddit="AskReddit"),
                    _post("other", title="weather report", subreddit="AskReddit"),
                    # Rejected IDs are not added to the retained-post dedupe set,
                    # so a repeated out-of-scope result is classified again.
                    _post("broad", title="Which IIM is best", subreddit="AskReddit"),
                ],
            }
        )
        report = run_collection(manager, plan)
        self.assertEqual(
            report.relevance_counts,
            {"iim_sambalpur": 1, "iim_broad": 2, "student_broad": 1, "other": 1},
        )
        self.assertEqual(report.posts_rejected, 4)
        self.assertEqual(report.posts_kept, 2)

    def test_post_cap_enforced(self):
        manager = FakeManager(posts=[_post(f"p{i}") for i in range(20)])
        report = run_collection(manager, self._plan(max_posts=3), persistence=None)
        self.assertEqual(report.posts_kept, 3)

    def test_partial_search_failure_recorded(self):
        manager = FakeManager(fail_search=True)
        report = run_collection(manager, self._plan(), persistence=None)
        self.assertEqual(report.posts_kept, 0)
        self.assertTrue(report.errors)

    def test_partial_comment_failure_keeps_post(self):
        manager = FakeManager(posts=[_post("p1")], fail_comments=True)
        report = run_collection(manager, self._plan(), persistence=None)
        self.assertEqual(report.posts_kept, 1)
        self.assertEqual(report.comments_collected, 0)
        self.assertTrue(report.errors)

    def test_persist_partial_failure_recorded(self):
        manager = FakeManager(posts=[_post("p1"), _post("p2")])
        persistence = FakePersistence(fail_for={"p2"})
        plan = self._plan(dry_run=False, persist=True, fetch_comments=False)
        report = run_collection(manager, plan, persistence=persistence)
        self.assertEqual(report.persisted_posts, 1)
        self.assertTrue(report.errors)

    def test_report_json_serialisable(self):
        import json

        manager = FakeManager(posts=[_post("p1")])
        report = run_collection(manager, self._plan(), persistence=None)
        json.dumps(report.to_dict())


class TestCli(unittest.TestCase):
    def test_defaults_are_five_posts(self):
        args = collect_reddit.build_parser().parse_args([])
        plan = collect_reddit.plan_from_args(args)
        self.assertEqual(plan.max_posts, 5)
        self.assertTrue(plan.dry_run)
        self.assertEqual(list(plan.keep_scopes), list(DEFAULT_KEEP_SCOPES))

    def test_cli_returns_nonzero_for_partial_results(self):
        partial = CollectionReport(
            dry_run=True, posts_kept=1,
            errors=["search request failed (RuntimeError)"],
        )
        with patch.object(sys, "argv", ["collect_reddit.py", "--json"]), patch.object(
            collect_reddit, "run_collection", return_value=partial
        ):
            self.assertEqual(collect_reddit.main(), 3)

    def test_cli_prints_search_attempt_context(self):
        import io
        from contextlib import redirect_stdout

        report = CollectionReport(
            dry_run=True,
            queries_attempted=1,
            queries_failed=1,
            search_attempts=1,
            errors=["partial search failure"],
            search_attempt_failures=[{
                "source_mode": "context",
                "subreddit": "CATpreparation",
                "query_index": 2,
                "attempt": 1,
                "failure_category": "http_status",
                "status_code": 503,
                "retrying": False,
            }],
        )
        output = io.StringIO()
        with patch.object(sys, "argv", ["collect_reddit.py"]), patch.object(
            collect_reddit, "run_collection", return_value=report
        ), redirect_stdout(output):
            result = collect_reddit.main()
        self.assertEqual(result, 3)
        self.assertIn("mode=context subreddit=CATpreparation query_index=2", output.getvalue())
        self.assertIn("attempt=1 category=http_status status=503", output.getvalue())

    def test_persist_flag_requires_yes(self):
        with self.assertRaises(SystemExit):
            collect_reddit.main(["--persist"])

    def test_persist_session_closes_after_success(self):
        import io
        from contextlib import redirect_stdout

        session = MagicMock()
        report = CollectionReport(dry_run=False, posts_kept=1)
        with (
            patch("app.core.database.SessionLocal", return_value=session),
            patch("app.services.ingestion.persistence.PersistenceService"),
            patch("app.services.ingestion.manager.IngestionManager"),
            patch.object(collect_reddit, "run_collection", return_value=report),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(collect_reddit.main(["--persist", "--yes"]), 0)
        session.close.assert_called_once_with()

    def test_persist_session_closes_after_collection_exception(self):
        import io
        from contextlib import redirect_stdout

        session = MagicMock()
        with (
            patch("app.core.database.SessionLocal", return_value=session),
            patch("app.services.ingestion.persistence.PersistenceService"),
            patch("app.services.ingestion.manager.IngestionManager"),
            patch.object(
                collect_reddit,
                "run_collection",
                side_effect=RuntimeError("persistence failed"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "persistence failed"):
                collect_reddit.main(["--persist", "--yes"])
        session.close.assert_called_once_with()


class RoutingManager:
    """Fake manager keyed by ``(subreddit, query)`` for search operations."""

    def __init__(self, posts_by_key=None, errors_by_key=None, comments=None):
        self.posts_by_key = posts_by_key or {}
        self.errors_by_key = errors_by_key or {}
        self.comments = comments if comments is not None else []
        self.calls = []

    def execute_operation(self, operation, provider=None, **kwargs):
        self.calls.append((operation, provider, kwargs))
        if operation == "search_posts":
            key = (kwargs.get("subreddit"), kwargs.get("query"))
            if key in self.errors_by_key:
                raise self.errors_by_key[key]
            return list(self.posts_by_key.get(key, []))
        if operation == "get_comments":
            return list(self.comments)
        raise AssertionError(f"unexpected operation {operation}")

    def search_keys(self):
        return [
            (kwargs.get("subreddit"), kwargs.get("query"))
            for operation, _provider, kwargs in self.calls
            if operation == "search_posts"
        ]

    def comment_post_ids(self):
        return [
            kwargs.get("post_id")
            for operation, _provider, kwargs in self.calls
            if operation == "get_comments"
        ]


class TestTargetRelevance(unittest.TestCase):
    def test_alias_variants_match(self):
        for text in (
            "is IIM Sambalpur worth it?",
            "join IIMSambalpur this year",
            "IIM-Sambalpur campus review",
            "IIM Sambalpur UG admissions",
            "iim sambalpur",
        ):
            self.assertTrue(is_target_relevant(text), text)

    def test_generic_and_other_iim_do_not_match(self):
        for text in (
            "Which IIM is best",
            "Sambalpur is a city in Odisha",
            "MBA placements 2026",
            "IIM Ahmedabad review",
            "IIM Bangalore cutoff",
            "CAT preparation tips",
        ):
            self.assertFalse(is_target_relevant(text), text)

    def test_matches_inside_longer_ug_slug(self):
        self.assertTrue(is_target_relevant("r/IIMSambalpurUG is active"))


class TestSourceValidation(unittest.TestCase):
    def test_generic_context_term_rejected(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource("CATpreparation", SourceMode.CONTEXT, ("IIM",))
            ]
        )
        with self.assertRaises(ValueError):
            plan.validate()

    def test_generic_legacy_term_rejected(self):
        plan = CollectionPlan(subreddits=["CATpreparation"], terms=["Sambalpur"])
        with self.assertRaises(ValueError):
            plan.validate()

    def test_direct_source_needs_no_terms(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)]
        )
        plan.validate()

    def test_approved_direct_name_matching_is_case_insensitive(self):
        plan = CollectionPlan(
            sources=[CollectionSource("iimsambalpurug", SourceMode.DIRECT)]
        )
        plan.validate()

    def test_unapproved_direct_source_is_rejected_before_provider_call(self):
        manager = FakeManager(posts=[_post("wrong", title="unrelated")])
        plan = CollectionPlan(
            sources=[CollectionSource("AskReddit", SourceMode.DIRECT)],
            request_delay=0,
        )

        with self.assertRaisesRegex(ValueError, "approved direct"):
            run_collection(manager, plan)

        self.assertEqual(manager.calls, [])

    def test_excessive_sources_are_rejected_before_provider_call(self):
        manager = FakeManager()
        plan = CollectionPlan(
            sources=[
                CollectionSource("IIMSambalpur", SourceMode.DIRECT)
                for _ in range(6)
            ],
            request_delay=0,
        )

        with self.assertRaisesRegex(ValueError, "at most 5 sources"):
            run_collection(manager, plan)

        self.assertEqual(manager.calls, [])

    def test_excessive_aliases_are_rejected(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource(
                    "AskReddit",
                    SourceMode.CONTEXT,
                    DEFAULT_TARGET_ALIASES + ("IIM Sambalpur admissions",),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "at most 4 search aliases"):
            plan.validate()

    def test_per_source_limit_bounds(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=3,
            max_posts_per_source=4,
        )
        with self.assertRaises(ValueError):
            plan.validate()


class TestSourceModes(unittest.TestCase):
    def test_direct_retrieves_without_query_and_retains_without_name(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={
                ("IIMSambalpur", None): [
                    _post("p1", title="hello everyone", subreddit="IIMSambalpur")
                ]
            }
        )
        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 1)
        self.assertIn(("IIMSambalpur", None), manager.search_keys())
        self.assertEqual(report.per_source["IIMSambalpur"].mode, "direct")

    def test_direct_results_from_an_unexpected_community_are_rejected(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            request_delay=0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={
                ("IIMSambalpur", None): [
                    _post("foreign", title="unrelated", subreddit="AskReddit")
                ]
            }
        )

        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 0)
        self.assertEqual(report.posts_rejected, 1)
        self.assertEqual(report.relevance_counts["iim_sambalpur"], 0)

    def test_direct_results_without_matching_reported_community_are_rejected(self):
        invalid_reported_subreddits = (None, "", "   ", "AskReddit")
        approved_sources = ("IIMSambalpur", "IIMSambalpurUG")

        for requested in approved_sources:
            for reported in invalid_reported_subreddits:
                with self.subTest(requested=requested, reported=reported):
                    plan = CollectionPlan(
                        sources=[CollectionSource(requested, SourceMode.DIRECT)],
                        max_posts=5,
                        request_delay=0,
                        fetch_comments=False,
                    )
                    manager = RoutingManager(
                        posts_by_key={
                            (requested, None): [
                                _post("unproven", title="No provenance", subreddit=reported)
                            ]
                        }
                    )

                    report = run_collection(manager, plan)

                    self.assertEqual(report.posts_kept, 0)
                    self.assertEqual(report.posts_rejected, 1)
                    self.assertEqual(report.relevance_counts["iim_sambalpur"], 0)
                    self.assertEqual(report.per_source[requested].posts_retained, 0)
                    self.assertEqual(manager.comment_post_ids(), [])

    def test_direct_results_match_approved_community_case_insensitively(self):
        matching_reported_subreddits = {
            "IIMSambalpur": "iImSaMbAlPuR",
            "IIMSambalpurUG": "iImSaMbAlPuRuG",
        }

        for requested, reported in matching_reported_subreddits.items():
            with self.subTest(requested=requested, reported=reported):
                plan = CollectionPlan(
                    sources=[CollectionSource(requested, SourceMode.DIRECT)],
                    max_posts=5,
                    request_delay=0,
                    fetch_comments=False,
                )
                manager = RoutingManager(
                    posts_by_key={
                        (requested, None): [
                            _post("matching", title="No name in text", subreddit=reported)
                        ]
                    }
                )

                report = run_collection(manager, plan)

                self.assertEqual(report.posts_kept, 1)
                self.assertEqual(report.posts_rejected, 0)
                self.assertEqual(report.relevance_counts["iim_sambalpur"], 0)
                self.assertEqual(report.per_source[requested].posts_retained, 1)

    def test_context_uses_server_side_query_and_gates_irrelevant(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource(
                    "CATpreparation", SourceMode.CONTEXT, ("IIM Sambalpur",)
                )
            ],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={
                ("CATpreparation", "IIM Sambalpur"): [
                    _post("k1", title="IIM Sambalpur fees", subreddit="CATpreparation"),
                    _post("r1", title="Which IIM is best", subreddit="CATpreparation"),
                ]
            }
        )
        report = run_collection(manager, plan)

        self.assertIn(("CATpreparation", "IIM Sambalpur"), manager.search_keys())
        self.assertEqual(report.posts_kept, 1)
        self.assertEqual(report.posts_rejected, 1)
        self.assertEqual(report.per_source["CATpreparation"].mode, "context")

    def test_comments_requested_only_for_retained_posts(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource(
                    "CATpreparation", SourceMode.CONTEXT, ("IIM Sambalpur",)
                )
            ],
            max_posts=5,
            request_delay=0.0,
        )
        manager = RoutingManager(
            posts_by_key={
                ("CATpreparation", "IIM Sambalpur"): [
                    _post("keep", title="IIM Sambalpur", subreddit="CATpreparation"),
                    _post("drop", title="random chatter", subreddit="CATpreparation"),
                ]
            },
            comments=[_comment("c1")],
        )
        report = run_collection(manager, plan)

        self.assertEqual(manager.comment_post_ids(), ["keep"])
        self.assertEqual(report.comment_trees_requested, 1)

    def test_dedup_across_aliases_and_communities(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource(
                    "CATpreparation",
                    SourceMode.CONTEXT,
                    ("IIM Sambalpur", "IIMSambalpur"),
                ),
                CollectionSource("MBAIndia", SourceMode.CONTEXT, ("IIM Sambalpur",)),
            ],
            max_posts=10,
            request_delay=0.0,
            fetch_comments=False,
        )
        duplicate = _post("dup", title="IIM Sambalpur", subreddit="CATpreparation")
        manager = RoutingManager(
            posts_by_key={
                ("CATpreparation", "IIM Sambalpur"): [duplicate],
                ("CATpreparation", "IIMSambalpur"): [duplicate],
                ("MBAIndia", "IIM Sambalpur"): [duplicate],
            }
        )
        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 1)
        self.assertEqual(report.duplicate_posts, 2)

    def test_per_source_limit_enforced(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=10,
            max_posts_per_source=2,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={("IIMSambalpur", None): [_post(f"p{i}") for i in range(5)]}
        )
        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 2)
        self.assertEqual(report.per_source["IIMSambalpur"].posts_retained, 2)
        search_call = next(call for call in manager.calls if call[0] == "search_posts")
        self.assertEqual(search_call[2]["limit"], 2)

    def test_timeout_forwarded_to_provider(self):
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
            timeout_seconds=12.5,
        )
        manager = RoutingManager(
            posts_by_key={("IIMSambalpur", None): [_post("p1")]}
        )
        run_collection(manager, plan)

        search_kwargs = [
            kwargs for operation, _p, kwargs in manager.calls
            if operation == "search_posts"
        ][0]
        self.assertEqual(search_kwargs["timeout"], 12.5)

    def test_original_text_and_metadata_preserved(self):
        body = "  weird   spacing\n\nkeep  "
        post = _post(
            "p1", title="Title  with  spaces", body=body, subreddit="IIMSambalpur"
        )
        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={("IIMSambalpur", None): [post]}
        )
        run_collection(manager, plan)

        self.assertEqual(post.body, body)
        self.assertEqual(post.title, "Title  with  spaces")
        self.assertEqual(post.subreddit, "IIMSambalpur")

    def test_report_is_json_serialisable_with_per_source(self):
        import json

        plan = CollectionPlan(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={("IIMSambalpur", None): [_post("p1")]}
        )
        report = run_collection(manager, plan)

        payload = report.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["per_source"]["IIMSambalpur"]["posts_retained"], 1)


class _FlakyManager:
    def __init__(self, failures, posts):
        self.failures = failures
        self.posts = posts
        self.calls = 0

    def execute_operation(self, operation, provider=None, **kwargs):
        self.calls += 1
        if operation == "search_posts" and self.calls <= self.failures:
            raise RuntimeError(
                "Arctic Shift request failed Status: 422 "
                "Response: Timeout. Maybe slow down a bit"
            )
        if operation == "search_posts":
            return list(self.posts)
        return []


class TestRetriesAndFailures(unittest.TestCase):
    def _direct_plan(self, **kwargs):
        base = dict(
            sources=[CollectionSource("IIMSambalpur", SourceMode.DIRECT)],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        base.update(kwargs)
        return CollectionPlan(**base)

    def test_transient_timeout_is_retried(self):
        delays = []
        manager = _FlakyManager(failures=1, posts=[_post("p1")])
        plan = self._direct_plan(max_retries=2, retry_backoff=1.0)

        report = run_collection(manager, plan, sleeper=delays.append)

        self.assertEqual(report.posts_kept, 1)
        self.assertEqual(manager.calls, 2)
        self.assertTrue(delays)
        self.assertFalse(report.errors)
        self.assertEqual((report.queries_attempted, report.queries_succeeded,
                          report.queries_failed, report.queries_retried,
                          report.search_attempts), (1, 1, 0, 1, 2))
        self.assertEqual(len(report.search_attempt_failures), 1)
        self.assertEqual(report.search_attempt_failures[0]["attempt"], 1)
        self.assertTrue(report.search_attempt_failures[0]["retrying"])
        source = report.per_source["IIMSambalpur"]
        self.assertEqual((source.queries_attempted, source.queries_succeeded,
                          source.queries_failed, source.queries_retried,
                          source.search_attempts), (1, 1, 0, 1, 2))

    def test_structured_http_status_is_retryable_and_preserved_in_diagnostics(self):
        class StatusFlakyManager:
            def __init__(self):
                self.calls = 0

            def execute_operation(self, operation, provider=None, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderOperationError(
                        "arctic_shift", "search_posts", "http_status", 503
                    )
                return [_post("p1")]

        manager = StatusFlakyManager()
        report = run_collection(
            manager,
            self._direct_plan(max_retries=1, retry_backoff=0),
            sleeper=lambda _s: None,
        )
        self.assertEqual(manager.calls, 2)
        self.assertEqual(report.queries_succeeded, 1)
        self.assertEqual(report.queries_failed, 0)
        self.assertEqual(report.queries_retried, 1)
        self.assertEqual(report.search_attempt_failures[0]["failure_category"], "http_status")
        self.assertEqual(report.search_attempt_failures[0]["status_code"], 503)

    def test_non_retryable_error_is_not_retried(self):
        class Boom:
            def __init__(self):
                self.calls = 0

            def execute_operation(self, *args, **kwargs):
                self.calls += 1
                raise ValueError("bad query")

        manager = Boom()
        report = run_collection(manager, self._direct_plan(max_retries=3))

        self.assertEqual(manager.calls, 1)
        self.assertEqual(report.posts_kept, 0)
        self.assertTrue(report.errors)

    def test_repeated_timeout_recorded_after_retries_exhausted(self):
        manager = _FlakyManager(failures=10, posts=[_post("p1")])
        report = run_collection(
            manager, self._direct_plan(max_retries=1), sleeper=lambda _s: None
        )
        self.assertEqual(manager.calls, 2)
        self.assertEqual(report.posts_kept, 0)
        self.assertTrue(report.errors)
        self.assertEqual((report.queries_attempted, report.queries_succeeded,
                          report.queries_failed, report.queries_retried,
                          report.search_attempts), (1, 0, 1, 1, 2))
        self.assertEqual(
            [(f["attempt"], f["retrying"]) for f in report.search_attempt_failures],
            [(1, True), (2, False)],
        )

    def test_retry_backoff_is_capped(self):
        manager = _FlakyManager(failures=10, posts=[])
        delays = []
        run_collection(
            manager,
            self._direct_plan(max_retries=5, retry_backoff=30),
            sleeper=delays.append,
        )
        self.assertEqual(delays, [30.0] * 5)

    def test_partial_failure_across_sources_continues(self):
        plan = CollectionPlan(
            sources=[
                CollectionSource("IIMSambalpur", SourceMode.DIRECT),
                CollectionSource(
                    "CATpreparation", SourceMode.CONTEXT, ("IIM Sambalpur",)
                ),
            ],
            max_posts=5,
            request_delay=0.0,
            fetch_comments=False,
        )
        manager = RoutingManager(
            posts_by_key={
                ("CATpreparation", "IIM Sambalpur"): [
                    _post("p2", title="IIM Sambalpur", subreddit="CATpreparation")
                ]
            },
            errors_by_key={("IIMSambalpur", None): RuntimeError("boom")},
        )
        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 1)
        self.assertTrue(report.errors)
        self.assertEqual(report.per_source["IIMSambalpur"].posts_retained, 0)
        self.assertTrue(report.per_source["IIMSambalpur"].errors)
        self.assertEqual((report.queries_attempted, report.queries_succeeded,
                          report.queries_failed), (2, 1, 1))
        self.assertEqual(
            sum(s.queries_attempted for s in report.per_source.values()),
            report.queries_attempted,
        )
        self.assertEqual(
            sum(s.queries_succeeded for s in report.per_source.values()),
            report.queries_succeeded,
        )
        self.assertEqual(
            sum(s.queries_failed for s in report.per_source.values()),
            report.queries_failed,
        )
        self.assertEqual(
            sum(s.queries_retried for s in report.per_source.values()),
            report.queries_retried,
        )
        self.assertEqual(
            sum(s.search_attempts for s in report.per_source.values()),
            report.search_attempts,
        )
        self.assertEqual(
            sum(len(s.search_attempt_failures) for s in report.per_source.values()),
            len(report.search_attempt_failures),
        )
        self.assertEqual(report.queries_attempted,
                         report.queries_succeeded + report.queries_failed)


class TestCliModes(unittest.TestCase):
    def test_default_plan_has_both_modes(self):
        args = collect_reddit.build_parser().parse_args([])
        plan = collect_reddit.plan_from_args(args)
        modes = {(source.subreddit, source.mode) for source in plan.sources}
        self.assertIn(("IIMSambalpur", SourceMode.DIRECT), modes)
        self.assertIn(("IIMSambalpurUG", SourceMode.DIRECT), modes)
        self.assertIn(("MBAIndia", SourceMode.CONTEXT), modes)
        self.assertIn(("CATpreparation", SourceMode.CONTEXT), modes)
        for subreddit in DEFAULT_CONTEXT_SUBREDDITS:
            self.assertIn((subreddit, SourceMode.CONTEXT), modes)
        plan.validate()

    def test_no_direct_disables_direct_communities(self):
        args = collect_reddit.build_parser().parse_args(["--no-direct"])
        plan = collect_reddit.plan_from_args(args)
        self.assertTrue(all(s.mode == SourceMode.CONTEXT for s in plan.sources))

    def test_custom_direct_subreddit_overrides_default(self):
        args = collect_reddit.build_parser().parse_args(
            ["--direct-subreddit", "IIMSambalpur"]
        )
        plan = collect_reddit.plan_from_args(args)
        direct = [s.subreddit for s in plan.sources if s.mode == SourceMode.DIRECT]
        self.assertEqual(direct, ["IIMSambalpur"])

    def test_cli_rejects_unapproved_direct_source_before_manager_creation(self):
        with patch(
            "app.services.ingestion.manager.IngestionManager"
        ) as manager_class:
            with self.assertRaises(SystemExit):
                collect_reddit.main(
                    ["--direct-subreddit", "AskReddit", "--no-context"]
                )
        manager_class.assert_not_called()

    def test_cli_rejects_excessive_aliases_before_manager_creation(self):
        argv = ["--no-direct"]
        for term in DEFAULT_TARGET_ALIASES + ("IIM Sambalpur admissions",):
            argv.extend(("--term", term))
        with patch(
            "app.services.ingestion.manager.IngestionManager"
        ) as manager_class:
            with self.assertRaises(SystemExit):
                collect_reddit.main(argv)
        manager_class.assert_not_called()

    def test_cli_rejects_excessive_sources_before_manager_creation(self):
        argv = ["--no-context"]
        for _ in range(6):
            argv.extend(("--direct-subreddit", "IIMSambalpur"))
        with patch(
            "app.services.ingestion.manager.IngestionManager"
        ) as manager_class:
            with self.assertRaises(SystemExit):
                collect_reddit.main(argv)
        manager_class.assert_not_called()

    def test_default_terms_are_institution_specific(self):
        args = collect_reddit.build_parser().parse_args([])
        plan = collect_reddit.plan_from_args(args)
        for term in plan.aliases:
            self.assertNotIn(term.strip().lower(), {"iim", "mba", "sambalpur"})


class TestSearchWorkBounds(unittest.TestCase):
    def _five_context_sources(self, terms=DEFAULT_TARGET_ALIASES):
        return [
            CollectionSource(name, SourceMode.CONTEXT, tuple(terms))
            for name in (
                "MBAIndia",
                "CATPrep",
                "CATpreparation",
                "AskReddit",
                "Indian_Academia",
            )
        ]

    def test_empty_results_stop_at_validated_task_count(self):
        plan = CollectionPlan(
            sources=self._five_context_sources(),
            request_delay=0,
            fetch_comments=False,
            max_retries=0,
        )
        manager = RoutingManager()

        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 0)
        self.assertEqual(report.queries_attempted, HARD_MAX_SEARCH_TASKS)
        self.assertEqual(len(manager.search_keys()), HARD_MAX_SEARCH_TASKS)

    def test_irrelevant_results_do_not_trigger_extra_queries_or_comments(self):
        source = "AskReddit"
        terms = tuple(DEFAULT_TARGET_ALIASES)
        manager = RoutingManager(
            posts_by_key={
                (source, term): [
                    _post(f"irrelevant-{index}", "unrelated chatter", subreddit=source)
                ]
                for index, term in enumerate(terms)
            }
        )
        plan = CollectionPlan(
            sources=[CollectionSource(source, SourceMode.CONTEXT, terms)],
            request_delay=0,
            fetch_comments=True,
            max_retries=0,
        )

        report = run_collection(manager, plan)

        self.assertEqual(report.posts_kept, 0)
        self.assertEqual(report.posts_rejected, len(terms))
        self.assertEqual(len(manager.search_keys()), len(terms))
        self.assertEqual(manager.comment_post_ids(), [])

    def test_maximum_provider_dispatch_bound_is_120(self):
        sources = self._five_context_sources()
        errors = {
            (source.subreddit, term): RuntimeError("timeout")
            for source in sources
            for term in DEFAULT_TARGET_ALIASES
        }
        manager = RoutingManager(errors_by_key=errors)
        plan = CollectionPlan(
            sources=sources,
            request_delay=0,
            fetch_comments=False,
            max_retries=5,
            retry_backoff=0,
        )

        report = run_collection(manager, plan, sleeper=lambda _delay: None)

        self.assertEqual(report.queries_attempted, HARD_MAX_SEARCH_TASKS)
        self.assertEqual(
            plan.maximum_provider_search_dispatches,
            HARD_MAX_PROVIDER_SEARCH_DISPATCHES,
        )
        self.assertEqual(
            report.search_attempts, HARD_MAX_PROVIDER_SEARCH_DISPATCHES
        )
        self.assertEqual(
            len(manager.search_keys()), HARD_MAX_PROVIDER_SEARCH_DISPATCHES
        )


if __name__ == "__main__":
    unittest.main()
