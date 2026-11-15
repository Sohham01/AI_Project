"""Bounded, local-only intake for an authorized Reddit static export.

This module reads JSON or JSONL files supplied by an operator.  It never calls
a provider, writes a file, touches persistence, or decides whether a dataset is
approved for annotation or training.  The returned ``PostData`` values use the
same naive-UTC ingestion convention as the existing providers; the original
dataset-level provenance and authorization record remain in ``metadata``.

The accepted contract is documented in ``docs/static_dataset_intake.md``.
CSV is deliberately not accepted here: a flat post/comment file needs an
additional grouping and provenance contract before it can be handled without
guessing relationships or source metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.schemas.ingestion import CommentData, PostData


APPROVED_AUTHORIZATION_STATUS = "approved"


@dataclass(frozen=True)
class StaticExportLimits:
    """Resource limits applied before records enter the in-memory pipeline."""

    max_bytes: int = 10 * 1024 * 1024
    max_posts: int = 200
    max_comments_per_post: int = 2000
    max_comments: int = 5000


@dataclass(frozen=True)
class StaticExportDiagnostic:
    """Sanitized validation result; never contains source text or file content."""

    code: str
    severity: str = "error"
    record_kind: str | None = None
    record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "record_kind": self.record_kind,
            "record_id": self.record_id,
        }


@dataclass
class StaticExportResult:
    """In-memory result of one bounded static-export validation."""

    posts: list[PostData] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[StaticExportDiagnostic, ...] = ()
    format: str | None = None
    posts_seen: int = 0
    comments_seen: int = 0
    lossy_default_fields: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.diagnostics)

    @property
    def usable_for_pilot(self) -> bool:
        """Whether normalized records can enter the pilot without lossy defaults."""

        return self.valid and not self.lossy_default_fields and bool(self.posts)

    @property
    def authorization_approved(self) -> bool:
        """Whether the supplied metadata has the explicit approved status.

        This is metadata evidence only.  The authorized pilot helper also
        requires an explicit operator confirmation and does not establish legal
        permission independently.
        """

        authorization = self.metadata.get("authorization")
        return (
            isinstance(authorization, dict)
            and authorization.get("authorization_status") == APPROVED_AUTHORIZATION_STATUS
            and isinstance(authorization.get("authorization_reference"), str)
            and bool(authorization["authorization_reference"].strip())
        )

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.diagnostics if issue.severity == "error")

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.diagnostics if issue.severity == "warning")

    def as_summary(self) -> dict[str, Any]:
        """Return safe counts and codes for operator logs."""

        return {
            "format": self.format,
            "valid": self.valid,
            "usable_for_pilot": self.usable_for_pilot,
            "posts_seen": self.posts_seen,
            "posts_accepted": len(self.posts),
            "comments_seen": self.comments_seen,
            "comments_accepted": sum(len(post.comments) for post in self.posts),
            "errors": list(self.error_codes),
            "warnings": list(self.warning_codes),
            "lossy_default_fields": list(self.lossy_default_fields),
        }


def _issue(
    diagnostics: list[StaticExportDiagnostic],
    code: str,
    *,
    severity: str = "error",
    record_kind: str | None = None,
    record_id: str | None = None,
) -> None:
    diagnostics.append(
        StaticExportDiagnostic(
            code=code,
            severity=severity,
            record_kind=record_kind,
            record_id=record_id,
        )
    )


def _text(value: Any, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError("missing text")
        return None
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    result = value.strip()
    if required and not result:
        raise ValueError("blank text")
    return result if result else None


def _identifier(value: Any, prefix: str | None = None) -> str:
    value = _text(value, required=True)
    assert value is not None
    if prefix:
        if value.startswith(("t1_", "t3_")):
            if not value.startswith(f"{prefix}_"):
                raise ValueError("wrong identifier prefix")
            value = value[len(prefix) + 1 :]
    if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid identifier")
    return value


def _timestamp(value: Any) -> datetime:
    """Parse a timezone-qualified source time into the ingestion UTC convention."""

    if isinstance(value, bool):
        raise ValueError("invalid timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("invalid timestamp")
        return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp timezone required")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _integer(value: Any, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid integer")
    if nonnegative and value < 0:
        raise ValueError("invalid nonnegative integer")
    return value


def _url(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or any(char.isspace() for char in value):
        raise ValueError("invalid URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid URL") from exc
    return value


def _validate_limits(limits: StaticExportLimits) -> None:
    for name in ("max_bytes", "max_posts", "max_comments_per_post", "max_comments"):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_metadata(metadata: Any, diagnostics: list[StaticExportDiagnostic]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        _issue(diagnostics, "metadata_missing")
        return {}

    source = metadata.get("source")
    authorization = metadata.get("authorization")
    if not isinstance(source, dict):
        _issue(diagnostics, "source_metadata_missing")
        source = {}
    if not isinstance(authorization, dict):
        _issue(diagnostics, "authorization_metadata_missing")
        authorization = {}

    required = (
        ("dataset_id", metadata),
        ("community", source),
        ("source_reference", source),
        ("source_provider", source),
        ("source_mode", source),
        ("authorization_status", authorization),
        ("authorization_reference", authorization),
    )
    for name, container in required:
        try:
            value = _text(container.get(name), required=True)
        except (AttributeError, ValueError):
            _issue(diagnostics, f"metadata_{name}_missing")
            continue
        if name == "source_mode" and value not in {"direct", "context"}:
            _issue(diagnostics, "metadata_source_mode_invalid")

    for name in ("snapshot_at", "collected_at"):
        if name in source and source.get(name) is not None:
            try:
                _timestamp(source[name])
            except (TypeError, ValueError, OverflowError, OSError):
                _issue(diagnostics, f"metadata_{name}_invalid")
    if "source_url" in source and source.get("source_url") is not None:
        try:
            _url(source["source_url"])
        except (TypeError, ValueError):
            _issue(diagnostics, "metadata_source_url_invalid")

    completeness = metadata.get("completeness")
    if not isinstance(completeness, dict) or "comments_complete" not in completeness:
        _issue(diagnostics, "comments_completeness_unknown", severity="warning")
    elif not isinstance(completeness.get("comments_complete"), bool):
        _issue(diagnostics, "metadata_comments_complete_invalid")
    elif not completeness["comments_complete"]:
        _issue(diagnostics, "comments_incomplete", severity="warning")

    # Preserve only the supplied metadata; no approval status is inferred.
    return dict(metadata)


def _parse_json_payload(payload: Any, diagnostics: list[StaticExportDiagnostic]) -> tuple[dict[str, Any], list[Any]]:
    if not isinstance(payload, dict):
        _issue(diagnostics, "json_top_level_object_required")
        return {}, []
    if type(payload.get("format_version")) is not int or payload.get("format_version") != 1:
        _issue(diagnostics, "unsupported_format_version")
    posts = payload.get("posts")
    if not isinstance(posts, list):
        _issue(diagnostics, "posts_array_missing")
        posts = []
    return _validate_metadata(payload.get("metadata"), diagnostics), posts


def _parse_jsonl_payload(text: str, diagnostics: list[StaticExportDiagnostic]) -> tuple[dict[str, Any], list[Any]]:
    metadata: dict[str, Any] = {}
    posts: dict[str, dict[str, Any]] = {}
    post_order: list[str] = []
    comments: list[tuple[str, Any]] = []
    saw_metadata = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            _issue(diagnostics, "blank_jsonl_line", record_kind="line")
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError, RecursionError):
            _issue(diagnostics, "invalid_jsonl", record_kind="line")
            continue
        if not isinstance(record, dict):
            _issue(diagnostics, "jsonl_record_object_required", record_kind="line")
            continue
        record_type = record.get("record_type")
        if record_type == "metadata":
            if saw_metadata:
                _issue(diagnostics, "duplicate_metadata_record", record_kind="metadata")
            elif line_number != 1:
                _issue(diagnostics, "metadata_record_must_be_first", record_kind="metadata")
            saw_metadata = True
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if type(record.get("format_version")) is not int or record.get("format_version") != 1:
                _issue(diagnostics, "unsupported_format_version", record_kind="metadata")
        elif record_type == "post":
            raw_post = record.get("post")
            if not isinstance(raw_post, dict):
                _issue(diagnostics, "jsonl_post_object_required", record_kind="post")
                continue
            try:
                post_id = _identifier(raw_post.get("post_id"), "t3")
            except ValueError:
                _issue(diagnostics, "post_id_invalid", record_kind="post")
                continue
            if post_id in posts:
                if posts[post_id] == raw_post:
                    _issue(
                        diagnostics,
                        "duplicate_post_id",
                        severity="warning",
                        record_kind="post",
                        record_id=post_id,
                    )
                else:
                    _issue(
                        diagnostics,
                        "conflicting_post_id",
                        record_kind="post",
                        record_id=post_id,
                    )
            else:
                posts[post_id] = dict(raw_post)
                post_order.append(post_id)
        elif record_type == "comment":
            post_id = record.get("post_id")
            raw_comment = record.get("comment")
            if not isinstance(post_id, str) or not isinstance(raw_comment, dict):
                _issue(diagnostics, "jsonl_comment_record_invalid", record_kind="comment")
                continue
            comments.append((post_id, raw_comment))
        else:
            _issue(diagnostics, "jsonl_record_type_invalid", record_kind="line")

    if not saw_metadata:
        _issue(diagnostics, "metadata_record_missing", record_kind="metadata")
    for post_id, raw_comment in comments:
        try:
            normalized_post_id = _identifier(post_id, "t3")
        except ValueError:
            _issue(diagnostics, "comment_post_id_invalid", record_kind="comment")
            continue
        post = posts.get(normalized_post_id)
        if post is None:
            _issue(diagnostics, "comment_post_missing", record_kind="comment")
            continue
        post.setdefault("comments", []).append(raw_comment)
    return _validate_metadata(metadata, diagnostics), [posts[post_id] for post_id in post_order]


def _normalize_comment(
    raw_comment: Any,
    *,
    diagnostics: list[StaticExportDiagnostic],
    lossy_fields: set[str],
) -> CommentData | None:
    if not isinstance(raw_comment, dict):
        _issue(diagnostics, "comment_record_invalid", record_kind="comment")
        return None
    try:
        comment_id = _identifier(raw_comment.get("comment_id"), "t1")
    except ValueError:
        _issue(diagnostics, "comment_id_invalid", record_kind="comment")
        return None
    try:
        body = _text(raw_comment.get("body"), required=True)
        created_at = _timestamp(raw_comment.get("created_at"))
    except (TypeError, ValueError, OverflowError, OSError):
        _issue(diagnostics, "comment_record_invalid", record_kind="comment", record_id=comment_id)
        return None
    try:
        parent_value = raw_comment.get("parent_comment_id")
        parent_id = None if parent_value in (None, "") else _identifier(parent_value, "t1")
        if parent_id == comment_id:
            raise ValueError("self parent")
    except (TypeError, ValueError, OverflowError, OSError):
        _issue(diagnostics, "comment_parent_invalid", record_kind="comment", record_id=comment_id)
        return None
    try:
        score_missing = "score" not in raw_comment or raw_comment.get("score") is None
        score = 0 if score_missing else _integer(raw_comment.get("score"))
    except (TypeError, ValueError, OverflowError, OSError):
        _issue(diagnostics, "comment_record_invalid", record_kind="comment", record_id=comment_id)
        return None
    if score_missing:
        _issue(
            diagnostics,
            "comment_score_missing",
            severity="warning",
            record_kind="comment",
            record_id=comment_id,
        )
        lossy_fields.add("comment.score")
    return CommentData(
        reddit_id=comment_id,
        body=body,
        created_at=created_at,
        score=score,
        parent_reddit_id=parent_id,
    )


def _normalize_post(
    raw_post: Any,
    *,
    diagnostics: list[StaticExportDiagnostic],
    lossy_fields: set[str],
    limits: StaticExportLimits,
    global_comments: dict[str, str],
) -> tuple[PostData | None, int]:
    if not isinstance(raw_post, dict):
        _issue(diagnostics, "post_record_invalid", record_kind="post")
        return None, 0
    try:
        post_id = _identifier(raw_post.get("post_id"), "t3")
    except ValueError:
        _issue(diagnostics, "post_id_invalid", record_kind="post")
        return None, 0
    try:
        title = _text(raw_post.get("title"), required=True)
        subreddit = _text(raw_post.get("subreddit"), required=True)
        assert title is not None and subreddit is not None
        if any(char.isspace() for char in subreddit) or "/" in subreddit:
            raise ValueError("invalid subreddit")
        body = _text(raw_post.get("body"))
        created_at = _timestamp(raw_post.get("created_at"))
        url = _url(raw_post.get("url"))
        score_missing = "score" not in raw_post or raw_post.get("score") is None
        score = 0 if score_missing else _integer(raw_post.get("score"))
        count_missing = "num_comments" not in raw_post or raw_post.get("num_comments") is None
        num_comments = (
            0 if count_missing else _integer(raw_post.get("num_comments"), nonnegative=True)
        )
    except (TypeError, ValueError, OverflowError, OSError):
        _issue(diagnostics, "post_record_invalid", record_kind="post", record_id=post_id)
        return None, 0

    raw_comments = raw_post.get("comments", [])
    if "comments" not in raw_post:
        _issue(
            diagnostics,
            "comments_field_missing",
            severity="warning",
            record_kind="post",
            record_id=post_id,
        )
    elif not isinstance(raw_comments, list):
        _issue(diagnostics, "comments_array_required", record_kind="post", record_id=post_id)
        return None, 0
    if len(raw_comments) > limits.max_comments_per_post:
        _issue(
            diagnostics,
            "comment_limit_exceeded",
            record_kind="post",
            record_id=post_id,
        )
        return None, len(raw_comments)

    if score_missing:
        _issue(
            diagnostics,
            "post_score_missing",
            severity="warning",
            record_kind="post",
            record_id=post_id,
        )
        lossy_fields.add("post.score")
    if count_missing:
        _issue(
            diagnostics,
            "post_comment_count_missing",
            severity="warning",
            record_kind="post",
            record_id=post_id,
        )
        lossy_fields.add("post.num_comments")

    comments: list[CommentData] = []
    seen_comments: dict[str, CommentData] = {}
    for raw_comment in raw_comments:
        comment = _normalize_comment(
            raw_comment,
            diagnostics=diagnostics,
            lossy_fields=lossy_fields,
        )
        if comment is None:
            continue
        previous = seen_comments.get(comment.reddit_id)
        if previous is not None:
            if previous != comment:
                _issue(
                    diagnostics,
                    "conflicting_comment_id",
                    record_kind="comment",
                    record_id=comment.reddit_id,
                )
            else:
                _issue(
                    diagnostics,
                    "duplicate_comment_id",
                    severity="warning",
                    record_kind="comment",
                    record_id=comment.reddit_id,
                )
            continue
        seen_comments[comment.reddit_id] = comment
        comments.append(comment)

        previous_post_id = global_comments.get(comment.reddit_id)
        if previous_post_id is not None and previous_post_id != post_id:
            _issue(
                diagnostics,
                "duplicate_comment_id_across_posts",
                record_kind="comment",
                record_id=comment.reddit_id,
            )
        else:
            global_comments[comment.reddit_id] = post_id

    returned_ids = set(seen_comments)
    for comment in comments:
        if comment.parent_reddit_id and comment.parent_reddit_id not in returned_ids:
            _issue(
                diagnostics,
                "unresolved_comment_parent",
                severity="warning",
                record_kind="comment",
                record_id=comment.reddit_id,
            )
    _detect_parent_cycles(comments, diagnostics)
    if num_comments > len(comments):
        _issue(
            diagnostics,
            "fewer_comments_than_reported",
            severity="warning",
            record_kind="post",
            record_id=post_id,
        )

    return (
        PostData(
            reddit_id=post_id,
            title=title,
            body=body,
            subreddit=subreddit,
            created_at=created_at,
            score=score,
            num_comments=num_comments,
            url=url,
            comments=comments,
        ),
        len(raw_comments),
    )


def _detect_parent_cycles(
    comments: list[CommentData],
    diagnostics: list[StaticExportDiagnostic],
) -> None:
    """Reject every reachable multi-node parent cycle without recursion."""

    parent_by_id = {
        comment.reddit_id: comment.parent_reddit_id
        for comment in comments
        if comment.parent_reddit_id in {item.reddit_id for item in comments}
    }
    completed: set[str] = set()
    reported: set[str] = set()
    for start in parent_by_id:
        if start in completed:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start
        while current is not None and current in parent_by_id:
            if current in positions:
                cycle = path[positions[current]:]
                for comment_id in cycle:
                    if comment_id not in reported:
                        _issue(
                            diagnostics,
                            "comment_parent_cycle",
                            record_kind="comment",
                            record_id=comment_id,
                        )
                        reported.add(comment_id)
                break
            if current in completed:
                break
            positions[current] = len(path)
            path.append(current)
            current = parent_by_id[current]
        completed.update(path)


def load_static_export(
    path: str | Path,
    *,
    limits: StaticExportLimits | None = None,
) -> StaticExportResult:
    """Read and validate one local JSON or JSONL static export.

    The function returns a result rather than raising for source-data problems,
    so callers can show bounded diagnostic codes without echoing source values.
    Invalid limits are programmer/configuration errors and raise ``ValueError``.
    """

    limits = limits or StaticExportLimits()
    _validate_limits(limits)
    diagnostics: list[StaticExportDiagnostic] = []
    try:
        source_path = Path(path)
    except TypeError:
        _issue(diagnostics, "path_invalid")
        return StaticExportResult(diagnostics=tuple(diagnostics))

    suffix = source_path.suffix.lower()
    if suffix == ".ndjson":
        suffix = ".jsonl"
    format_name = suffix.lstrip(".") or None
    if suffix not in {".json", ".jsonl"}:
        _issue(diagnostics, "unsupported_format")
        return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)
    try:
        if not source_path.is_file():
            _issue(diagnostics, "source_file_missing")
            return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)
        raw = source_path.read_bytes()
    except OSError:
        _issue(diagnostics, "source_file_unreadable")
        return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)
    if len(raw) > limits.max_bytes:
        _issue(diagnostics, "source_size_limit_exceeded")
        return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        _issue(diagnostics, "utf8_encoding_required")
        return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)

    if suffix == ".json":
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, RecursionError):
            _issue(diagnostics, "invalid_json")
            return StaticExportResult(diagnostics=tuple(diagnostics), format=format_name)
        metadata, raw_posts = _parse_json_payload(payload, diagnostics)
    else:
        metadata, raw_posts = _parse_jsonl_payload(text, diagnostics)

    posts_seen = len(raw_posts)
    comments_seen = sum(
        len(item.get("comments", []))
        for item in raw_posts
        if isinstance(item, dict) and isinstance(item.get("comments", []), list)
    )
    if posts_seen > limits.max_posts:
        _issue(diagnostics, "post_limit_exceeded")
    if posts_seen == 0:
        _issue(diagnostics, "posts_empty")
    if comments_seen > limits.max_comments:
        _issue(diagnostics, "comment_limit_exceeded")

    posts: list[PostData] = []
    seen_posts: dict[str, PostData] = {}
    global_comments: dict[str, str] = {}
    lossy_fields: set[str] = set()
    if posts_seen <= limits.max_posts and comments_seen <= limits.max_comments:
        for raw_post in raw_posts:
            post, _ = _normalize_post(
                raw_post,
                diagnostics=diagnostics,
                lossy_fields=lossy_fields,
                limits=limits,
                global_comments=global_comments,
            )
            if post is None:
                continue
            previous = seen_posts.get(post.reddit_id)
            if previous is not None:
                if previous != post:
                    _issue(
                        diagnostics,
                        "conflicting_post_id",
                        record_kind="post",
                        record_id=post.reddit_id,
                    )
                else:
                    _issue(
                        diagnostics,
                        "duplicate_post_id",
                        severity="warning",
                        record_kind="post",
                        record_id=post.reddit_id,
                    )
                continue
            seen_posts[post.reddit_id] = post
            posts.append(post)

    return StaticExportResult(
        posts=posts,
        metadata=metadata,
        diagnostics=tuple(diagnostics),
        format=format_name,
        posts_seen=posts_seen,
        comments_seen=comments_seen,
        lossy_default_fields=tuple(sorted(lossy_fields)),
    )


__all__ = [
    "StaticExportDiagnostic",
    "StaticExportLimits",
    "StaticExportResult",
    "APPROVED_AUTHORIZATION_STATUS",
    "load_static_export",
]
