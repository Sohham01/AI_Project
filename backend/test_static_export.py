"""Offline tests for bounded authorized static-export intake."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.services.ingestion.static_export import (
    StaticExportLimits,
    load_static_export,
)
from app.services.pilot import (
    prepare_authorized_static_pilot,
    prepare_end_to_end_pilot,
)


def _metadata(*, authorization_status: str = "pending-review") -> dict:
    return {
        "dataset_id": "approved-static-test-v1",
        "source": {
            "community": "IIMSambalpur",
            "source_reference": "supplier-reference-001",
            "source_provider": "authorized-static-export",
            "source_mode": "direct",
            "source_url": "https://www.reddit.com/r/IIMSambalpur/",
            "snapshot_at": "2026-01-15T12:00:00Z",
        },
        "authorization": {
            "authorization_status": authorization_status,
            "authorization_reference": "approval-record-001",
        },
    }


def _post(*, post_id: str = "abc123", comments: list[dict] | None = None) -> dict:
    return {
        "post_id": post_id,
        "subreddit": "IIMSambalpur",
        "title": "A supplied static title",
        "body": "A supplied static body",
        "created_at": "2026-01-15T12:00:00Z",
        "score": 4,
        "num_comments": len(comments or []),
        "url": f"https://www.reddit.com/r/IIMSambalpur/comments/{post_id}/title/",
        "comments": comments or [],
    }


def _comment(*, comment_id: str = "c1", parent: str | None = None) -> dict:
    return {
        "comment_id": comment_id,
        "body": "A supplied comment\nwith multiple lines and Unicode प्रश्न",
        "created_at": "2026-01-15T12:10:00+00:00",
        "score": 2,
        "parent_comment_id": parent,
    }


class StaticExportTests(unittest.TestCase):
    def write_file(self, name: str, value: object, *, encoding: str = "utf-8") -> Path:
        directory = Path(self.tempdir.name)
        path = directory / name
        if isinstance(value, str):
            path.write_text(value, encoding=encoding)
        else:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding=encoding)
        return path

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def test_json_normalizes_records_and_preserves_metadata(self) -> None:
        posts = [_post(comments=[_comment(), _comment(comment_id="c2", parent="c1")])]
        result = load_static_export(
            self.write_file("export.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": posts,
            })
        )

        self.assertTrue(result.valid, result.as_summary())
        self.assertTrue(result.usable_for_pilot, result.as_summary())
        self.assertEqual(result.posts_seen, 1)
        self.assertEqual(result.comments_seen, 2)
        self.assertEqual(result.posts[0].reddit_id, "abc123")
        self.assertEqual(result.posts[0].comments[1].parent_reddit_id, "c1")
        self.assertEqual(result.metadata["authorization"]["authorization_reference"], "approval-record-001")
        self.assertEqual(result.posts[0].created_at.isoformat(), "2026-01-15T12:00:00")

    def test_jsonl_metadata_post_and_comment_records_are_supported(self) -> None:
        lines = [
            {"record_type": "metadata", "format_version": 1, "metadata": _metadata()},
            {"record_type": "post", "post": _post(comments=[])},
            {"record_type": "comment", "post_id": "abc123", "comment": _comment()},
        ]
        path = self.write_file("export.jsonl", "\n".join(json.dumps(line) for line in lines))
        result = load_static_export(path)

        self.assertTrue(result.valid, result.as_summary())
        self.assertEqual(len(result.posts), 1)
        self.assertEqual([row.reddit_id for row in result.posts[0].comments], ["c1"])

    def test_boolean_format_version_is_rejected(self) -> None:
        result = load_static_export(
            self.write_file("boolean-version.json", {
                "format_version": True,
                "metadata": _metadata(),
                "posts": [_post(comments=[])],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("unsupported_format_version", result.error_codes)

    def test_identical_jsonl_duplicate_posts_warn_and_deduplicate(self) -> None:
        post = _post(comments=[])
        lines = [
            {"record_type": "metadata", "format_version": 1, "metadata": _metadata()},
            {"record_type": "post", "post": post},
            {"record_type": "post", "post": post},
        ]
        result = load_static_export(
            self.write_file("duplicate-posts.jsonl", "\n".join(json.dumps(line) for line in lines))
        )

        self.assertTrue(result.valid, result.as_summary())
        self.assertEqual(len(result.posts), 1)
        self.assertIn("duplicate_post_id", result.warning_codes)

    def test_conflicting_jsonl_duplicate_posts_fail_closed(self) -> None:
        first = _post(comments=[])
        second = _post(comments=[])
        second["title"] = "Conflicting synthetic title"
        lines = [
            {"record_type": "metadata", "format_version": 1, "metadata": _metadata()},
            {"record_type": "post", "post": first},
            {"record_type": "post", "post": second},
        ]
        result = load_static_export(
            self.write_file("conflicting-posts.jsonl", "\n".join(json.dumps(line) for line in lines))
        )

        self.assertFalse(result.valid)
        self.assertIn("conflicting_post_id", result.error_codes)

    def test_comment_id_repeated_under_different_posts_fails_closed(self) -> None:
        shared = _comment(comment_id="shared-comment")
        result = load_static_export(
            self.write_file("cross-post-comment.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [
                    _post(post_id="post-one", comments=[shared]),
                    _post(post_id="post-two", comments=[dict(shared)]),
                ],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("duplicate_comment_id_across_posts", result.error_codes)

    def test_multi_node_parent_cycle_fails_closed(self) -> None:
        first = _comment(comment_id="cycle-one", parent="cycle-two")
        second = _comment(comment_id="cycle-two", parent="cycle-one")
        result = load_static_export(
            self.write_file("parent-cycle.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [_post(comments=[first, second])],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("comment_parent_cycle", result.error_codes)

    def test_static_export_enters_existing_in_memory_pilot_without_provider_call(self) -> None:
        path = self.write_file(
            "export.json",
            {"format_version": 1, "metadata": _metadata(), "posts": [_post(comments=[_comment()])]},
        )
        result = load_static_export(path)
        source = result.metadata["source"]
        pilot = prepare_end_to_end_pilot(
            result.posts,
            source_provider=source["source_provider"],
            source_mode=source["source_mode"],
        )

        self.assertEqual(pilot.summary["status"], "ready_for_review")
        self.assertEqual(pilot.summary["posts_received"], 1)
        self.assertEqual(pilot.summary["comments_received"], 1)
        self.assertFalse(pilot.summary["writes_performed"])
        self.assertFalse(pilot.summary["training_performed"])
        self.assertFalse(pilot.summary["discussion_complete_asserted"])
        self.assertEqual(pilot.candidate_rows[0]["source_provider"], "authorized-static-export")

    def test_pending_review_cannot_cross_authorized_static_pilot_gate(self) -> None:
        result = load_static_export(
            self.write_file("pending-export.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [_post(comments=[_comment()])],
            })
        )

        self.assertTrue(result.usable_for_pilot)
        with self.assertRaises(PermissionError):
            prepare_authorized_static_pilot(result, approval_confirmed=True)

    def test_explicit_approval_allows_pilot_and_retains_safe_provenance(self) -> None:
        result = load_static_export(
            self.write_file("approved-export.json", {
                "format_version": 1,
                "metadata": _metadata(authorization_status="approved"),
                "posts": [_post(comments=[_comment()])],
            })
        )

        pilot = prepare_authorized_static_pilot(result, approval_confirmed=True)

        self.assertEqual(pilot.summary["status"], "ready_for_review")
        self.assertEqual(
            pilot.summary["dataset_provenance"],
            {
                "dataset_id": "approved-static-test-v1",
                "source_reference": "supplier-reference-001",
                "source_community": "IIMSambalpur",
                "source_provider": "authorized-static-export",
                "source_mode": "direct",
                "authorization_reference": "approval-record-001",
                "authorization_status": "approved",
                "completeness": {
                    "limitation_codes": ["comments_completeness_unknown"],
                },
            },
        )

    def test_explicit_confirmation_is_required_even_with_approved_metadata(self) -> None:
        result = load_static_export(
            self.write_file("approved-without-confirmation.json", {
                "format_version": 1,
                "metadata": _metadata(authorization_status="approved"),
                "posts": [_post(comments=[_comment()])],
            })
        )

        with self.assertRaises(PermissionError):
            prepare_authorized_static_pilot(result, approval_confirmed=False)

    def test_missing_body_and_url_are_represented_without_defaults(self) -> None:
        post = _post(comments=[_comment()])
        post.pop("body")
        post.pop("url")
        result = load_static_export(
            self.write_file("missing-optional.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [post],
            })
        )

        self.assertTrue(result.usable_for_pilot, result.as_summary())
        self.assertIsNone(result.posts[0].body)
        self.assertIsNone(result.posts[0].url)

    def test_missing_numeric_fields_are_reported_and_block_pilot_use(self) -> None:
        post = _post(comments=[_comment()])
        post.pop("score")
        post.pop("num_comments")
        comment = post["comments"][0]
        comment.pop("score")
        result = load_static_export(
            self.write_file("missing-numeric.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [post],
            })
        )

        self.assertTrue(result.valid, result.as_summary())
        self.assertFalse(result.usable_for_pilot)
        self.assertIn("post.score", result.lossy_default_fields)
        self.assertIn("comment.score", result.lossy_default_fields)
        self.assertIn("post_score_missing", result.warning_codes)

    def test_unresolved_parent_is_preserved_as_a_warning(self) -> None:
        result = load_static_export(
            self.write_file("unresolved-parent.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [_post(comments=[_comment(parent="not-returned")])],
            })
        )

        self.assertTrue(result.usable_for_pilot, result.as_summary())
        self.assertEqual(result.posts[0].comments[0].parent_reddit_id, "not-returned")
        self.assertIn("unresolved_comment_parent", result.warning_codes)

    def test_identical_duplicates_are_deduplicated(self) -> None:
        post = _post(comments=[_comment()])
        result = load_static_export(
            self.write_file("duplicate.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [post, post],
            })
        )

        self.assertTrue(result.valid, result.as_summary())
        self.assertEqual(len(result.posts), 1)
        self.assertIn("duplicate_post_id", result.warning_codes)

    def test_conflicting_duplicate_ids_fail_closed(self) -> None:
        first = _post(comments=[])
        second = _post(comments=[])
        second["title"] = "Conflicting title"
        result = load_static_export(
            self.write_file("conflict.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [first, second],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("conflicting_post_id", result.error_codes)

    def test_conflicting_comment_ids_and_post_parent_ids_fail_closed(self) -> None:
        first = _comment()
        second = _comment()
        second["body"] = "A conflicting comment"
        invalid_parent = _comment(comment_id="c3", parent="t3_abc123")
        result = load_static_export(
            self.write_file("comment-conflict.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [_post(comments=[first, second, invalid_parent])],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("conflicting_comment_id", result.error_codes)
        self.assertIn("comment_parent_invalid", result.error_codes)

    def test_invalid_timestamp_and_malformed_record_are_reported(self) -> None:
        post = _post(comments=[])
        post["created_at"] = "2026-01-15T12:00:00"
        result = load_static_export(
            self.write_file("invalid-time.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [post, "not-an-object"],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("post_record_invalid", result.error_codes)

    def test_empty_dataset_is_not_pilot_usable(self) -> None:
        result = load_static_export(
            self.write_file("empty.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [],
            })
        )

        self.assertFalse(result.valid)
        self.assertIn("posts_empty", result.error_codes)
        self.assertFalse(result.usable_for_pilot)

    def test_record_limits_are_errors_without_silent_truncation(self) -> None:
        result = load_static_export(
            self.write_file("too-many.json", {
                "format_version": 1,
                "metadata": _metadata(),
                "posts": [_post(post_id="p1"), _post(post_id="p2")],
            }),
            limits=StaticExportLimits(max_posts=1),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.posts, [])
        self.assertIn("post_limit_exceeded", result.error_codes)

    def test_invalid_encoding_and_csv_are_rejected(self) -> None:
        invalid = Path(self.tempdir.name) / "invalid.json"
        invalid.write_bytes(b"\xff\xfe")
        self.assertIn("utf8_encoding_required", load_static_export(invalid).error_codes)

        csv_path = Path(self.tempdir.name) / "export.csv"
        csv_path.write_text("post_id,title\nabc123,title\n", encoding="utf-8")
        self.assertIn("unsupported_format", load_static_export(csv_path).error_codes)


if __name__ == "__main__":
    unittest.main()
