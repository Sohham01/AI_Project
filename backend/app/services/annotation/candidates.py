"""
Annotation-candidate export for response classification.

Lives outside the ``response_classification`` package on purpose: that package
is architecturally isolated from the database and the preprocessing pipeline
(enforced by its own tests), while this hand-off legitimately touches both.

The response taxonomy is hand-annotated, so the pipeline needs a clean hand-off
between *raw collected comments* and *a CSV a human fills in*.  This module
produces that hand-off and nothing more:

* every candidate keeps the **original** comment text untouched, and - when a
  preprocessor is supplied - a separate ``normalized_text`` produced by the
  existing pipeline;
* ``response_type`` is always blank: model predictions are never copied into
  ground truth;
* enough context is included (post title, parent comment text, identifiers,
  subreddit, score, timestamps, language metadata) to annotate reliably.

It does not touch :class:`ResponseAnnotation`; :func:`to_training_rows` converts
finished candidates into the existing training-CSV shape (``text`` = the
normalized text) so the schema stays exactly as it is.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES,
    is_valid_response_type,
)

#: Human-review column order.  ``text`` is the ORIGINAL text; ``normalized_text``
#: is what the model will eventually see.
CANDIDATE_COLUMNS: List[str] = [
    "document_id",
    "text",
    "normalized_text",
    "response_type",
    "post_id",
    "comment_id",
    "parent_comment_id",
    "parent_text",
    "post_title",
    "subreddit",
    "score",
    "created_at",
    "source_url",
    "source_provider",
    "source_mode",
    "parent_context_status",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

# These are the columns used by the original candidate exporter.  The newer
# review-context columns are optional so an older, already-authorized static
# candidate export can still be converted without losing its training fields.
REQUIRED_CANDIDATE_COLUMNS: List[str] = [
    "document_id",
    "text",
    "normalized_text",
    "response_type",
    "post_id",
    "comment_id",
    "parent_comment_id",
    "parent_text",
    "post_title",
    "subreddit",
    "score",
    "created_at",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
]

_BOOLEAN_TRUE = {"1", "true", "yes", "y", "t"}
_BOOLEAN_FALSE = {"0", "false", "no", "n", "f"}
_SOURCE_MODES = {"direct", "context"}
_PARENT_CONTEXT_STATUSES = {"top_level", "available", "unavailable", "unknown"}

_REQUIRED_KEYS = ("comment_id", "text")


class CandidateValidationError(ValueError):
    """Raised when candidate rows cannot be safely written/consumed."""


def _as_bool(value: Any, *, field: str = "is_code_mixed") -> Optional[bool]:
    """Parse an optional boolean without turning bad input into ``False``."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _BOOLEAN_TRUE:
            return True
        if token in _BOOLEAN_FALSE:
            return False
    raise CandidateValidationError(
        f"invalid {field} value {value!r}; expected one of "
        f"{sorted(_BOOLEAN_TRUE | _BOOLEAN_FALSE)}"
    )


def _optional_text(value: Any) -> str:
    """Preserve missing optional text as an empty CSV cell, never ``'None'``."""
    if value is None:
        return ""
    return str(value).strip()


def _optional_integer(value: Any, *, field: str) -> str:
    """Return a canonical integer cell while preserving an absent value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if isinstance(value, bool):
        raise CandidateValidationError(
            f"invalid {field} value {value!r}; expected an integer or missing value"
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        token = value.strip()
        try:
            return str(int(token))
        except ValueError as exc:
            raise CandidateValidationError(
                f"invalid {field} value {value!r}; expected an integer or missing value"
            ) from exc
    raise CandidateValidationError(
        f"invalid {field} value {value!r}; expected an integer or missing value"
    )


def _validate_optional_metadata(row: Mapping[str, Any], index: int) -> List[str]:
    problems: List[str] = []
    score = row.get("score")
    if score not in (None, ""):
        try:
            _optional_integer(score, field="score")
        except CandidateValidationError as exc:
            problems.append(f"row {index}: {exc}")

    bool_value = row.get("is_code_mixed")
    if bool_value not in (None, ""):
        try:
            _as_bool(bool_value)
        except CandidateValidationError as exc:
            problems.append(f"row {index}: {exc}")

    source_mode = _optional_text(row.get("source_mode"))
    if source_mode and source_mode not in _SOURCE_MODES:
        problems.append(
            f"row {index}: invalid source_mode {source_mode!r}; "
            f"expected one of {sorted(_SOURCE_MODES)}"
        )

    parent_status = _optional_text(row.get("parent_context_status"))
    if parent_status and parent_status not in _PARENT_CONTEXT_STATUSES:
        problems.append(
            f"row {index}: invalid parent_context_status {parent_status!r}; "
            f"expected one of {sorted(_PARENT_CONTEXT_STATUSES)}"
        )
    return problems


def build_candidate_rows(
    comments: Sequence[Mapping[str, Any]],
    *,
    preprocessor: Optional[Any] = None,
    language_service: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Turn collected comment mappings into review rows.

    Args:
        comments: mappings with ``document_id``, ``comment_id`` and ``text`` (the
            original body). Optional keys: ``post_id``, ``parent_comment_id``,
            ``parent_text``, ``post_title``, ``subreddit``, ``score``,
            ``created_at``, ``source_url``/``url``, ``source_provider``,
            ``source_mode`` and ``parent_context_status``.
        preprocessor: optional :class:`MultilingualTextPreprocessor`.  When
            given, ``normalized_text`` is filled from the existing pipeline;
            the original ``text`` is never modified.
        language_service: optional :class:`LanguageAnalysisService`.  When
            given, ``language``/``script``/``is_code_mixed``/
            ``detected_languages`` are filled from real analysis instead of
            defaults.

    Returns:
        One dict per comment, with a blank ``response_type``.
    """
    rows: List[Dict[str, Any]] = []
    for index, comment in enumerate(comments):
        missing = [k for k in (*_REQUIRED_KEYS, "document_id") if not comment.get(k)]
        if missing:
            raise CandidateValidationError(
                f"candidate #{index} is missing required field(s): {missing}"
            )

        comment_id = str(comment["comment_id"]).strip()
        original = str(comment["text"])
        document_id = str(comment["document_id"]).strip()

        language = _optional_text(comment.get("language"))
        script = _optional_text(comment.get("script"))
        is_code_mixed = _as_bool(comment.get("is_code_mixed"))
        detected = comment.get("detected_languages")
        if detected is None or detected == "":
            detected = []
        if isinstance(detected, str):
            detected = [d for d in detected.split("|") if d.strip()]

        if language_service is not None:
            analysis = language_service.analyze(original)
            language = analysis.language
            script = analysis.script
            is_code_mixed = analysis.is_code_mixed
            detected = list(analysis.detected_languages)

        normalized = ""
        if preprocessor is not None:
            result = preprocessor.preprocess(
                original,
                language=language,
                script=script,
                is_code_mixed=is_code_mixed,
                detected_languages=list(detected),
            )
            normalized = result.normalized_text

        rows.append(
            {
                "document_id": document_id,
                "text": original,
                "normalized_text": normalized,
                # Never pre-fill: this is the human's job.
                "response_type": "",
                "post_id": _optional_text(comment.get("post_id")),
                "comment_id": comment_id,
                "parent_comment_id": _optional_text(comment.get("parent_comment_id")),
                "parent_text": _optional_text(comment.get("parent_text")),
                "post_title": _optional_text(comment.get("post_title")),
                "subreddit": _optional_text(comment.get("subreddit")),
                "score": _optional_integer(comment.get("score"), field="score"),
                "created_at": _optional_text(comment.get("created_at")),
                "source_url": _optional_text(
                    comment.get("source_url", comment.get("url"))
                ),
                "source_provider": _optional_text(comment.get("source_provider")),
                "source_mode": _optional_text(comment.get("source_mode")),
                "parent_context_status": _optional_text(
                    comment.get("parent_context_status")
                ),
                "language": language,
                "script": script,
                "is_code_mixed": ""
                if is_code_mixed is None
                else ("true" if is_code_mixed else "false"),
                "detected_languages": "|".join(str(d) for d in detected),
            }
        )
    return rows


def validate_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return a list of problems with *rows* (empty list means valid).

    This is the check a reviewer runs before trusting an export:
    duplicate/blank identifiers, empty original text, and - importantly - a
    non-blank ``response_type`` (which would mean a prediction leaked into the
    ground truth).
    """
    problems: List[str] = []
    seen_ids: Dict[str, int] = {}
    seen_comment_ids: Dict[str, int] = {}
    for index, row in enumerate(rows):
        document_id = str(row.get("document_id") or "").strip()
        comment_id = str(row.get("comment_id") or "").strip()
        if not document_id:
            problems.append(f"row {index}: missing document_id")
        if not comment_id:
            problems.append(f"row {index}: missing comment_id")
        if document_id:
            if document_id in seen_ids:
                problems.append(
                    f"row {index}: duplicate document_id {document_id!r} "
                    f"(first seen at row {seen_ids[document_id]})"
                )
            else:
                seen_ids[document_id] = index
        if comment_id:
            if comment_id in seen_comment_ids:
                problems.append(
                    f"row {index}: duplicate comment_id {comment_id!r} "
                    f"(first seen at row {seen_comment_ids[comment_id]})"
                )
            else:
                seen_comment_ids[comment_id] = index
        if not str(row.get("text") or "").strip():
            problems.append(f"row {index}: empty original text")
        if str(row.get("response_type") or "").strip():
            problems.append(
                f"row {index}: response_type is pre-filled "
                f"({row.get('response_type')!r}); candidates must ship blank"
            )
        problems.extend(_validate_optional_metadata(row, index))
    return problems


def candidate_context_warnings(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Report missing review context without inventing or rejecting records.

    These warnings are intentionally separate from schema validation: a
    top-level comment may legitimately have no parent, while a missing post,
    timestamp, URL or provenance still needs a human release decision.
    """
    warnings: List[str] = []
    for index, row in enumerate(rows):
        document_id = _optional_text(row.get("document_id")) or f"row {index}"
        if not _optional_text(row.get("post_id")):
            warnings.append(f"{document_id}: missing post_id/thread context")
        if not _optional_text(row.get("created_at")):
            warnings.append(f"{document_id}: missing source timestamp")
        if not _optional_text(row.get("source_url")):
            warnings.append(f"{document_id}: missing source_url")
        if not _optional_text(row.get("source_provider")):
            warnings.append(f"{document_id}: missing source_provider")
        if not _optional_text(row.get("source_mode")):
            warnings.append(f"{document_id}: missing source_mode")

        parent_status = _optional_text(row.get("parent_context_status"))
        parent_id = _optional_text(row.get("parent_comment_id"))
        parent_text = _optional_text(row.get("parent_text"))
        if not parent_status:
            warnings.append(f"{document_id}: missing parent_context_status")
        elif parent_status == "available" and (not parent_id or not parent_text):
            warnings.append(
                f"{document_id}: parent_context_status=available but parent "
                "identity or text is missing"
            )
        elif parent_status == "top_level" and (parent_id or parent_text):
            warnings.append(
                f"{document_id}: top_level parent_context_status conflicts with "
                "parent fields"
            )
        elif parent_status == "unavailable":
            warnings.append(f"{document_id}: parent context unavailable")
    return warnings


def audit_training_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Classify candidate rows before conversion without inventing labels.

    The existing :func:`to_training_rows` return contract is unchanged. This
    report makes its skip behavior visible to a pilot operator and lets the
    conversion CLI fail closed on identity/schema errors.
    """
    report: Dict[str, Any] = {
        "accepted": [],
        "blank_response_labels": [],
        "invalid_response_labels": [],
        "missing_normalized_text": [],
        "missing_document_ids": [],
        "duplicate_document_ids": [],
        "missing_comment_ids": [],
        "duplicate_comment_ids": [],
        "empty_original_text": [],
        "metadata_problems": [],
    }
    seen_documents: Dict[str, int] = {}
    seen_comments: Dict[str, int] = {}
    for index, row in enumerate(rows):
        document_id = _optional_text(row.get("document_id"))
        comment_id = _optional_text(row.get("comment_id"))
        label = _optional_text(row.get("response_type"))
        normalized = _optional_text(row.get("normalized_text"))
        original = _optional_text(row.get("text"))

        if not document_id:
            report["missing_document_ids"].append(index)
        elif document_id in seen_documents:
            report["duplicate_document_ids"].append(document_id)
        else:
            seen_documents[document_id] = index
        if not comment_id:
            report["missing_comment_ids"].append(index)
        elif comment_id in seen_comments:
            report["duplicate_comment_ids"].append(comment_id)
        else:
            seen_comments[comment_id] = index
        if not original:
            report["empty_original_text"].append(index)

        if not label:
            report["blank_response_labels"].append(index)
        elif not is_valid_response_type(label):
            report["invalid_response_labels"].append(label)
        if not normalized:
            report["missing_normalized_text"].append(index)
        report["metadata_problems"].extend(_validate_optional_metadata(row, index))

        if (
            document_id
            and comment_id
            and label
            and is_valid_response_type(label)
            and normalized
            and original
        ):
            report["accepted"].append(index)

    report["fatal"] = any(
        report[key]
        for key in (
            "missing_document_ids",
            "duplicate_document_ids",
            "missing_comment_ids",
            "duplicate_comment_ids",
            "empty_original_text",
            "invalid_response_labels",
            "metadata_problems",
        )
    )
    report["excluded"] = len(rows) - len(report["accepted"])
    return report


def write_candidates_csv(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    validate: bool = True,
) -> str:
    """Write *rows* to *path* in :data:`CANDIDATE_COLUMNS` order.

    Raises :class:`CandidateValidationError` (before writing) when *validate*
    is set and the rows have problems, so a bad export never hits disk.
    """
    if validate:
        problems = validate_candidate_rows(rows)
        if problems:
            raise CandidateValidationError(
                "candidate export failed validation:\n  - "
                + "\n  - ".join(problems)
            )

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    # O_EXCL makes the create-and-claim operation atomic.  An existence check
    # followed by open(..., "w") would still allow another process to win the
    # race and truncate a previously written review batch.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    fd: Optional[int] = None
    created = False
    try:
        fd = os.open(path, flags, 0o666)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd = None
            writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in CANDIDATE_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if fd is not None:
            os.close(fd)
        if created:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise
    return path


def read_candidates_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def to_training_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Convert annotated candidates to the existing training-CSV shape.

    The model consumes ``text`` = the preprocessed ``normalized_text`` (see the
    response annotation contract).  Rows without a valid ``response_type`` or
    without normalized text are skipped, because they are not trainable - this
    function never invents a label.
    """
    out: List[Dict[str, str]] = []
    for row in rows:
        response_type = str(row.get("response_type") or "").strip()
        normalized = str(row.get("normalized_text") or "").strip()
        if not is_valid_response_type(response_type):
            continue
        if not normalized:
            continue
        detected = row.get("detected_languages") or ""
        if isinstance(detected, str):
            detected = "|".join(d for d in detected.split("|") if d.strip())
        is_code_mixed = _as_bool(row.get("is_code_mixed"))
        out.append(
            {
                "document_id": str(row.get("document_id") or ""),
                "text": normalized,
                "response_type": response_type,
                "post_id": str(row.get("post_id") or ""),
                "comment_id": str(row.get("comment_id") or ""),
                "language": _optional_text(row.get("language")),
                "script": _optional_text(row.get("script")),
                "is_code_mixed": ""
                if is_code_mixed is None
                else ("true" if is_code_mixed else "false"),
                "detected_languages": detected,
            }
        )
    return out


def load_comment_candidates(session: Any) -> List[Dict[str, Any]]:
    """Load comment candidates from the database with parent/post context.

    Reads only non-identifying fields (no author information exists on the
    models).  Imports are local so this module stays importable without a
    configured database.
    """
    from sqlalchemy import select

    from app.models.reddit_comment import RedditComment
    from app.models.reddit_post import RedditPost

    comments = list(session.execute(select(RedditComment)).scalars())
    if not comments:
        return []

    posts = {
        post.id: post
        for post in session.execute(select(RedditPost)).scalars()
    }
    by_id = {c.id: c for c in comments}

    candidates: List[Dict[str, Any]] = []
    for comment in comments:
        parent = by_id.get(comment.parent_comment_id)
        post = posts.get(comment.post_id)
        candidates.append(
            {
                "document_id": f"comment-{comment.reddit_id}",
                "comment_id": comment.reddit_id,
                "text": comment.body,
                "post_id": post.reddit_id if post else "",
                "parent_comment_id": parent.reddit_id if parent else "",
                "parent_text": parent.body if parent else "",
                "post_title": post.title if post else "",
                "subreddit": post.subreddit if post else "",
                "score": comment.score,
                "created_at": comment.created_at.isoformat()
                if comment.created_at
                else "",
                "source_url": post.url if post else "",
                "parent_context_status": (
                    "top_level"
                    if comment.parent_comment_id is None
                    else ("available" if parent else "unavailable")
                ),
            }
        )
    return candidates


__all__ = [
    "CANDIDATE_COLUMNS",
    "REQUIRED_CANDIDATE_COLUMNS",
    "CandidateValidationError",
    "build_candidate_rows",
    "validate_candidate_rows",
    "candidate_context_warnings",
    "audit_training_candidate_rows",
    "write_candidates_csv",
    "read_candidates_csv",
    "to_training_rows",
    "load_comment_candidates",
    "RESPONSE_TYPES",
]
