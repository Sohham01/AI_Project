"""Review-manifest creation and validation for annotation candidates.

The candidate CSV is deliberately a small, review-facing hand-off.  This
module owns the separate JSONL sidecar used to prove which source facts were
reviewed and whether a dataset was explicitly approved.  It does not read the
database, call a provider, or copy source text into a manifest record.

The public contract is intentionally made from standard-library types so an
offline review job can use it without starting the application.  A manifest
record has four top-level sections::

    {"manifest_version": "1.0", "document_id": "...",
     "source": {...}, "review": {...}, "adjudication": {...},
     "dataset_approval": {...}}

``source`` contains facts supplied by the candidate export.  ``review`` and
``adjudication`` contain human decisions.  ``dataset_approval`` is a separate
authorization gate; a row being reviewed is never treated as dataset approval.
The writer serializes one compact JSON object per line and never overwrites an
existing path.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence


# A released sidecar must identify the contract that produced it.  Callers may
# validate a different explicitly named version, but generated records always
# use this version unless asked otherwise.
MANIFEST_VERSION = "1.0"
CURRENT_MANIFEST_VERSION = MANIFEST_VERSION


# These are source facts only.  ``parent_text``, ``post_title``, ``text`` and
# ``normalized_text`` are intentionally absent: the review context belongs in
# the approved context bundle, not in this audit sidecar.
SOURCE_FIELDS = (
    "source_provider",
    "source_mode",
    "source_subreddit",
    "post_id",
    "comment_id",
    "parent_comment_id",
    "source_url",
    "source_created_at",
    "score",
    "parent_context_status",
    "context_snapshot_ref",
    "source_record_hash",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
)

OPTIONAL_PROVENANCE_FIELDS = (
    "source_provider",
    "source_mode",
    "source_subreddit",
    "post_id",
    "comment_id",
    "source_url",
    "source_created_at",
    "parent_context_status",
    "context_snapshot_ref",
    "source_record_hash",
)

REVIEW_FIELDS = (
    "status",
    "response_label",
    "annotator_1_ref",
    "annotator_1_label",
    "annotator_1_reviewed_at",
    "annotator_1_notes_ref",
    "annotator_2_ref",
    "annotator_2_label",
    "annotator_2_reviewed_at",
    "annotator_2_notes_ref",
    "exclusion_reason",
)

ADJUDICATION_FIELDS = (
    "status",
    "ref",
    "label",
    "reviewed_at",
    "reason_ref",
)

DATASET_APPROVAL_FIELDS = (
    "status",
    "dataset_version",
    "guideline_version",
    "approval_ref",
    "approved_at",
    "training_inclusion",
)

# Flat field names from the proposal are exposed as a compatibility constant
# for callers that build or inspect hand-authored rows.  The JSONL writer uses
# the nested sections above so category boundaries cannot be confused.
MANIFEST_FIELDS = (
    "manifest_version",
    "dataset_version",
    "document_id",
    "source_provider",
    "source_mode",
    "source_subreddit",
    "post_id",
    "comment_id",
    "parent_comment_id",
    "source_url",
    "source_created_at",
    "context_snapshot_ref",
    "source_record_hash",
    "guideline_version",
    "review_status",
    "exclusion_reason",
    "annotator_1_ref",
    "annotator_1_label",
    "annotator_1_reviewed_at",
    "annotator_1_notes_ref",
    "annotator_2_ref",
    "annotator_2_label",
    "annotator_2_reviewed_at",
    "annotator_2_notes_ref",
    "adjudicator_ref",
    "adjudicated_label",
    "adjudicated_at",
    "adjudication_reason_ref",
    "approval_ref",
    "approved_at",
    "training_inclusion",
)

REVIEW_STATUSES = frozenset(
    {
        "unreviewed",
        "annotated_1",
        "annotated_2",
        "adjudication_required",
        "adjudicated",
        "excluded",
        "approved_for_training",
    }
)
ADJUDICATION_STATUSES = frozenset(
    {"pending", "required", "adjudicated", "resolved", "not_required"}
)
APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
TRAINING_INCLUSION_VALUES = frozenset({"pending", "include", "exclude"})

SOURCE_PROVENANCE_FIELDS = SOURCE_FIELDS
REVIEW_STATUS_VALUES = REVIEW_STATUSES
DATASET_APPROVAL_STATUS_VALUES = APPROVAL_STATUSES


# Candidate CSV names and the canonical source-fact names in the manifest.
# The aliases keep older candidate exports usable without guessing a value.
_SOURCE_ALIASES = {
    "source_provider": ("source_provider",),
    "source_mode": ("source_mode",),
    "source_subreddit": ("source_subreddit", "subreddit"),
    "post_id": ("post_id", "thread_id"),
    "comment_id": ("comment_id",),
    "parent_comment_id": ("parent_comment_id",),
    "source_url": ("source_url", "url"),
    "source_created_at": ("source_created_at", "created_at", "timestamp"),
    "score": ("score",),
    "parent_context_status": ("parent_context_status",),
    "context_snapshot_ref": ("context_snapshot_ref",),
    "source_record_hash": ("source_record_hash", "source_hash"),
    "language": ("language",),
    "script": ("script",),
    "is_code_mixed": ("is_code_mixed",),
    "detected_languages": ("detected_languages",),
}


_MISSING = object()


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _as_document_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation result.

    Issue details contain field names, stable identifiers, and fixed rule
    descriptions only.  They never contain candidate text or any other raw
    source-content value.
    """

    code: str
    message: str
    document_id: Optional[str] = None
    field: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "code": self.code,
            "message": self.message,
            "document_id": self.document_id,
            "field": self.field,
        }

    def __str__(self) -> str:
        location = []
        if self.document_id:
            location.append(f"document_id={self.document_id!r}")
        if self.field:
            location.append(f"field={self.field}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}{suffix}: {self.message}"

    def __contains__(self, value: object) -> bool:
        """Allow simple substring checks while retaining structured issues."""

        return str(value) in str(self)

    def __getitem__(self, key: str) -> Optional[str]:
        return self.as_dict()[key]


@dataclass
class ValidationReport:
    """Errors and non-fatal warnings produced by a manifest check."""

    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    approval_checked: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def approved(self) -> bool:
        """Whether the explicit training-approval gate passed."""

        return self.approval_checked and self.valid

    @property
    def error_codes(self) -> list[str]:
        return [issue.code for issue in self.errors]

    @property
    def warning_codes(self) -> list[str]:
        return [issue.code for issue in self.warnings]

    def add_error(
        self,
        code: str,
        *,
        document_id: Optional[str] = None,
        field: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self.errors.append(
            ValidationIssue(
                code=code,
                message=message or _default_issue_message(code),
                document_id=document_id,
                field=field,
            )
        )

    def add_warning(
        self,
        code: str,
        *,
        document_id: Optional[str] = None,
        field: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self.warnings.append(
            ValidationIssue(
                code=code,
                message=message or _default_issue_message(code),
                document_id=document_id,
                field=field,
            )
        )

    def extend(self, other: "ValidationReport") -> None:
        existing_errors = set(self.errors)
        existing_warnings = set(self.warnings)
        for issue in other.errors:
            if issue not in existing_errors:
                self.errors.append(issue)
                existing_errors.add(issue)
        for issue in other.warnings:
            if issue not in existing_warnings:
                self.warnings.append(issue)
                existing_warnings.add(issue)
        self.approval_checked = self.approval_checked or other.approval_checked

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "approved": self.approved,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }

    def raise_for_errors(self, prefix: str = "manifest validation failed") -> None:
        if self.errors:
            raise ManifestValidationError(self, prefix=prefix)

    raise_if_invalid = raise_for_errors


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot safely be built, read, or written."""

    def __init__(
        self,
        report: ValidationReport,
        *,
        prefix: str = "manifest validation failed",
    ) -> None:
        self.report = report
        self.errors = report.errors
        self.issues = report.errors
        self.warnings = report.warnings
        details = []
        for issue in report.errors:
            location = []
            if issue.document_id:
                location.append(f"document_id={issue.document_id!r}")
            if issue.field:
                location.append(f"field={issue.field}")
            suffix = f" ({', '.join(location)})" if location else ""
            details.append(f"{issue.code}{suffix}: {issue.message}")
        super().__init__(prefix + ": " + "; ".join(details))


class ManifestRecords(list):
    """List-compatible build result carrying non-fatal provenance warnings."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]] = (),
        *,
        report: Optional[ValidationReport] = None,
    ) -> None:
        super().__init__(records)
        self.report = report or ValidationReport()

    @property
    def warnings(self) -> List[ValidationIssue]:
        return self.report.warnings

    @property
    def warning_messages(self) -> List[str]:
        return [issue.message for issue in self.report.warnings]


def _default_issue_message(code: str) -> str:
    messages = {
        "invalid_candidate_record": "candidate row must be a mapping",
        "missing_candidate_document_id": "candidate document_id is required",
        "duplicate_candidate_document_id": "candidate document_id must be unique",
        "candidate_source_conflict": "candidate source aliases disagree",
        "candidate_row_width": "candidate CSV row has an unexpected width",
        "candidate_header": "candidate CSV must include document_id",
        "invalid_source_section": "source section must be an object",
        "invalid_manifest_record": "manifest row must be an object",
        "missing_manifest_version": "manifest_version is required",
        "unsupported_manifest_version": "manifest_version is not supported",
        "missing_manifest_document_id": "manifest document_id is required",
        "duplicate_manifest_document_id": "manifest document_id must be unique",
        "manifest_source_conflict": "manifest source aliases disagree",
        "invalid_iso_timestamp": "timestamp must be an ISO-8601 string",
        "missing_optional_provenance": "optional provenance is missing",
        "missing_required_provenance": "required provenance is missing",
        "unknown_required_provenance_field": "required provenance field is not in the manifest contract",
        "blank_response_label": "blank response labels are review-only",
        "invalid_review_status": "review status is not recognized",
        "invalid_adjudication_status": "adjudication status is not recognized",
        "invalid_approval_status": "dataset approval status is not recognized",
        "invalid_training_inclusion": "training inclusion is not recognized",
        "excluded_missing_reason": "excluded records require an exclusion reason",
        "inclusion_conflicts_with_exclusion": "excluded records cannot be included",
        "missing_manifest_for_candidate": "candidate has no manifest row",
        "orphan_manifest_document_id": "manifest document_id has no candidate row",
        "source_field_conflict": "shared source fields conflict",
        "source_timestamp_conflict": "source timestamps must match exactly",
        "not_approved": "record is not approved for training",
        "review_not_complete": "record review is not complete",
        "adjudication_required": "record requires adjudication",
        "record_excluded": "excluded records are not approved",
        "dataset_approval_required": "explicit dataset approval is required",
        "dataset_approval_incomplete": "dataset approval is incomplete",
        "dataset_approval_rejected": "dataset approval is rejected",
        "approved_missing_dataset_version": "approved training data requires a dataset version",
        "approval_missing_inclusion": "approved records require training_inclusion=include",
        "serialization_failed": "manifest record could not be serialized",
        "invalid_jsonl": "manifest line is not valid JSON",
        "jsonl_not_object": "manifest JSONL lines must be objects",
        "blank_jsonl_line": "blank JSONL lines are not allowed",
    }
    return messages.get(code, "manifest validation rule failed")


def _choose_alias(
    record: Mapping[str, Any],
    nested: Mapping[str, Any],
    field_name: str,
    aliases: Sequence[str],
    report: Optional[ValidationReport],
    *,
    document_id: Optional[str],
    conflict_code: str,
) -> Any:
    """Choose a canonical value and flag contradictory aliases.

    Blank aliases do not erase a nonblank value.  Nonblank values are compared
    exactly; in particular timestamps are never parsed and normalized here.
    """

    candidates: List[tuple[str, Any]] = []
    if field_name in nested:
        candidates.append((field_name, nested[field_name]))
    for alias in aliases:
        if alias in record:
            candidates.append((alias, record[alias]))

    if not candidates:
        return None
    chosen = candidates[0][1]
    for _name, value in candidates:
        if _is_blank(chosen) and not _is_blank(value):
            chosen = value
    nonblank = [value for _name, value in candidates if not _is_blank(value)]
    if report is not None and len(nonblank) > 1:
        first = nonblank[0]
        if any(value != first for value in nonblank[1:]):
            report.add_error(
                conflict_code,
                document_id=document_id,
                field=field_name,
            )
    return chosen


def _extract_source(
    record: Mapping[str, Any],
    report: Optional[ValidationReport] = None,
    *,
    document_id: Optional[str] = None,
    conflict_code: str = "candidate_source_conflict",
) -> dict[str, Any]:
    nested_value = record.get("source", _MISSING)
    if nested_value is not _MISSING and not isinstance(nested_value, Mapping):
        if report is not None:
            report.add_error(
                "invalid_source_section",
                document_id=document_id,
                field="source",
            )
        nested: Mapping[str, Any] = {}
    else:
        nested = _mapping(nested_value)

    source: dict[str, Any] = {}
    for field_name in SOURCE_FIELDS:
        source[field_name] = _choose_alias(
            record,
            nested,
            field_name,
            _SOURCE_ALIASES[field_name],
            report,
            document_id=document_id,
            conflict_code=conflict_code,
        )
    return source


def _required_source_field_names(required_source_fields: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Normalize policy-supplied required source fields without inventing values."""

    if required_source_fields is None:
        return ()
    if isinstance(required_source_fields, str):
        return (required_source_fields,)
    try:
        return tuple(dict.fromkeys(str(field) for field in required_source_fields))
    except TypeError:
        return (str(required_source_fields),)


def _validate_required_source_field_names(
    required_source_fields: Sequence[str],
    report: ValidationReport,
) -> None:
    for field_name in required_source_fields:
        if field_name not in SOURCE_FIELDS:
            report.add_error(
                "unknown_required_provenance_field",
                field=field_name,
            )


def _check_required_source_fields(
    source: Mapping[str, Any],
    report: ValidationReport,
    *,
    document_id: Optional[str],
    required_source_fields: Sequence[str],
) -> None:
    for field_name in required_source_fields:
        if field_name in SOURCE_FIELDS and _is_blank(source.get(field_name)):
            report.add_error(
                "missing_required_provenance",
                document_id=document_id,
                field=field_name,
            )


def _read_csv_candidates(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a candidate CSV while allowing omitted newer optional columns."""

    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, restkey="__extra_fields__", restval=None)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames or "document_id" not in fieldnames:
                report = ValidationReport()
                report.add_error("candidate_header", field="document_id")
                report.raise_for_errors("candidate CSV validation failed")
            if len(fieldnames) != len(set(fieldnames)):
                report = ValidationReport()
                report.add_error("candidate_header")
                report.raise_for_errors("candidate CSV validation failed")
            for _line_number, row in enumerate(reader, start=2):
                if row.get("__extra_fields__") is not None:
                    report = ValidationReport()
                    report.add_error("candidate_row_width")
                    report.raise_for_errors("candidate CSV validation failed")
                rows.append(dict(row))
    except csv.Error:
        # Do not expose the parser's raw line in a privacy-sensitive error.
        report = ValidationReport()
        report.add_error("candidate_row_width")
        raise ManifestValidationError(
            report,
            prefix="candidate CSV validation failed",
        ) from None
    return rows


def _coerce_candidates(source: Any) -> list[Mapping[str, Any]]:
    if isinstance(source, (str, os.PathLike, Path)):
        return _read_csv_candidates(source)
    if isinstance(source, Mapping):
        return [source]
    try:
        rows = list(source)
    except TypeError:
        report = ValidationReport()
        report.add_error("invalid_candidate_record")
        raise ManifestValidationError(report, prefix="candidate validation failed") from None
    return rows


def _validate_timestamp(
    value: Any,
    report: ValidationReport,
    *,
    document_id: Optional[str],
    field: str,
) -> None:
    if _is_blank(value):
        return
    if not isinstance(value, str):
        report.add_error("invalid_iso_timestamp", document_id=document_id, field=field)
        return
    try:
        # Parsing is validation-only.  The original string remains untouched
        # and join comparison below uses exact string equality.
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        report.add_error("invalid_iso_timestamp", document_id=document_id, field=field)


def _warn_missing_provenance(
    source: Mapping[str, Any],
    report: ValidationReport,
    *,
    document_id: Optional[str],
    required_source_fields: Sequence[str] = (),
) -> None:
    for field_name in OPTIONAL_PROVENANCE_FIELDS:
        if field_name in required_source_fields:
            continue
        if _is_blank(source.get(field_name)):
            report.add_warning(
                "missing_optional_provenance",
                document_id=document_id,
                field=field_name,
            )


def validate_candidate_records(
    source: Any,
    *,
    required_source_fields: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Validate candidate identity and report optional-provenance warnings.

    Only ``document_id`` is a required manifest join identity.  Other source
    identifiers are conditional provenance and therefore produce warnings when
    absent.  Candidate text is deliberately never inspected or emitted in an
    issue.
    """

    rows = _coerce_candidates(source)
    report = ValidationReport()
    required_fields = _required_source_field_names(required_source_fields)
    _validate_required_source_field_names(required_fields, report)
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            report.add_error("invalid_candidate_record")
            continue
        document_id = _as_document_id(row.get("document_id"))
        issue_id = document_id or f"row-{index}"
        if not document_id:
            report.add_error(
                "missing_candidate_document_id",
                document_id=issue_id,
                field="document_id",
            )
        elif document_id in seen:
            report.add_error(
                "duplicate_candidate_document_id",
                document_id=document_id,
                field="document_id",
            )
        else:
            seen[document_id] = index

        source_values = _extract_source(
            row,
            report,
            document_id=document_id or issue_id,
            conflict_code="candidate_source_conflict",
        )
        _validate_timestamp(
            source_values.get("source_created_at"),
            report,
            document_id=document_id or issue_id,
            field="source_created_at",
        )
        _check_required_source_fields(
            source_values,
            report,
            document_id=document_id or issue_id,
            required_source_fields=required_fields,
        )
        _warn_missing_provenance(
            source_values,
            report,
            document_id=document_id or issue_id,
            required_source_fields=required_fields,
        )

        response_label = row.get("response_type", row.get("response_label"))
        if _is_blank(response_label):
            report.add_warning(
                "blank_response_label",
                document_id=document_id or issue_id,
                field="response_type",
            )
    return report


def load_candidate_records(source: Any) -> list[Mapping[str, Any]]:
    """Load candidate mappings from memory or a local CSV path."""

    return _coerce_candidates(source)


read_candidate_records = load_candidate_records


def _new_review() -> dict[str, Any]:
    values = {field_name: None for field_name in REVIEW_FIELDS}
    values["status"] = "unreviewed"
    return values


def _new_adjudication() -> dict[str, Any]:
    return {field_name: None for field_name in ADJUDICATION_FIELDS}


def _new_dataset_approval() -> dict[str, Any]:
    values = {field_name: None for field_name in DATASET_APPROVAL_FIELDS}
    values["status"] = "pending"
    values["training_inclusion"] = "pending"
    return values


def _new_manifest_record(
    candidate: Mapping[str, Any],
    *,
    manifest_version: str,
) -> dict[str, Any]:
    document_id = _as_document_id(candidate.get("document_id"))
    source = _extract_source(candidate)
    # Keep the insertion order obvious for callers inspecting a record.  The
    # writer also canonicalizes with sorted JSON keys for byte-stable output.
    return {
        "manifest_version": manifest_version,
        "document_id": document_id,
        "source": source,
        "review": _new_review(),
        "adjudication": _new_adjudication(),
        "dataset_approval": _new_dataset_approval(),
    }


def build_manifest_records(
    candidates: Any = None,
    *,
    candidate_records: Any = None,
    candidate_csv_path: str | os.PathLike[str] | None = None,
    manifest_version: str = MANIFEST_VERSION,
    required_source_fields: Optional[Iterable[str]] = None,
) -> ManifestRecords:
    """Create one unreviewed manifest record per candidate.

    ``candidates`` may be a sequence/iterator of mappings, one mapping, or a
    local candidate CSV path.  Legacy CSVs may omit newer optional context
    columns; absent values remain ``None`` and produce warnings. A caller may
    supply policy-specific ``required_source_fields``; missing values then
    become structured errors rather than being fabricated.
    """

    supplied = [
        value
        for value in (candidates, candidate_records, candidate_csv_path)
        if value is not None
    ]
    if len(supplied) != 1:
        raise TypeError("provide exactly one of candidates, candidate_records, or candidate_csv_path")
    rows = _coerce_candidates(supplied[0])
    required_fields = _required_source_field_names(required_source_fields)
    report = validate_candidate_records(
        rows,
        required_source_fields=required_fields,
    )
    if _is_blank(manifest_version):
        report.add_error("missing_manifest_version", field="manifest_version")
    report.raise_for_errors("candidate-to-manifest validation failed")
    records = [
        _new_manifest_record(row, manifest_version=str(manifest_version))
        for row in rows
        if isinstance(row, Mapping)
    ]
    return ManifestRecords(records, report=report)


def build_manifest(candidates: Any = None, **kwargs: Any) -> ManifestRecords:
    """Compatibility alias for :func:`build_manifest_records`."""

    return build_manifest_records(candidates, **kwargs)


def build_review_manifest(candidates: Any = None, **kwargs: Any) -> ManifestRecords:
    """Descriptive alias for :func:`build_manifest_records`."""

    return build_manifest_records(candidates, **kwargs)


create_manifest_records = build_manifest_records
create_review_manifest = build_manifest_records
create_manifest = build_manifest_records


def _first_value(
    nested: Mapping[str, Any],
    record: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    found: list[Any] = []
    for name in names:
        if name in nested:
            found.append(nested[name])
    for name in names:
        if name in record:
            found.append(record[name])
    if not found:
        return None
    for value in found:
        if not _is_blank(value):
            return value
    return found[0]


def _extract_review(
    record: Mapping[str, Any],
    *,
    document_id: Optional[str],
    report: Optional[ValidationReport] = None,
) -> dict[str, Any]:
    nested_value = record.get("review", _MISSING)
    if nested_value is not _MISSING and not isinstance(nested_value, Mapping):
        if report is not None:
            report.add_error("invalid_manifest_record", document_id=document_id, field="review")
        nested: Mapping[str, Any] = {}
    else:
        nested = _mapping(nested_value)

    aliases: dict[str, Sequence[str]] = {
        "status": ("review_status",),
        "response_label": ("response_label", "response_type", "label"),
        "annotator_1_ref": ("annotator_1_ref",),
        "annotator_1_label": ("annotator_1_label",),
        "annotator_1_reviewed_at": ("annotator_1_reviewed_at",),
        "annotator_1_notes_ref": ("annotator_1_notes_ref",),
        "annotator_2_ref": ("annotator_2_ref",),
        "annotator_2_label": ("annotator_2_label",),
        "annotator_2_reviewed_at": ("annotator_2_reviewed_at",),
        "annotator_2_notes_ref": ("annotator_2_notes_ref",),
        "exclusion_reason": ("exclusion_reason",),
    }
    values: dict[str, Any] = {}
    for field_name in REVIEW_FIELDS:
        values[field_name] = _first_value(nested, record, (field_name, *aliases[field_name]))
    return values


def _extract_adjudication(
    record: Mapping[str, Any],
    *,
    document_id: Optional[str],
    report: Optional[ValidationReport] = None,
) -> dict[str, Any]:
    nested_value = record.get("adjudication", _MISSING)
    if nested_value is not _MISSING and not isinstance(nested_value, Mapping):
        if report is not None:
            report.add_error(
                "invalid_manifest_record",
                document_id=document_id,
                field="adjudication",
            )
        nested: Mapping[str, Any] = {}
    else:
        nested = _mapping(nested_value)
    aliases = {
        "status": ("adjudication_status",),
        "ref": ("adjudicator_ref",),
        "label": ("adjudicated_label",),
        "reviewed_at": ("adjudicated_at",),
        "reason_ref": ("adjudication_reason_ref",),
    }
    return {
        field_name: _first_value(nested, record, (field_name, *aliases[field_name]))
        for field_name in ADJUDICATION_FIELDS
    }


def _extract_dataset_approval(
    record: Mapping[str, Any],
    *,
    document_id: Optional[str],
    report: Optional[ValidationReport] = None,
) -> dict[str, Any]:
    nested_value = record.get("dataset_approval", _MISSING)
    if nested_value is _MISSING:
        nested_value = record.get("approval", _MISSING)
    if nested_value is not _MISSING and not isinstance(nested_value, Mapping):
        if report is not None:
            report.add_error(
                "invalid_manifest_record",
                document_id=document_id,
                field="dataset_approval",
            )
        nested: Mapping[str, Any] = {}
    else:
        nested = _mapping(nested_value)
    aliases: dict[str, Sequence[str]] = {
        "status": ("dataset_approval_status", "approval_status"),
        "dataset_version": ("dataset_version",),
        "guideline_version": ("guideline_version",),
        "approval_ref": ("approval_ref",),
        "approved_at": ("approved_at",),
        "training_inclusion": ("training_inclusion",),
    }
    values = {
        field_name: _first_value(nested, record, (field_name, *aliases[field_name]))
        for field_name in DATASET_APPROVAL_FIELDS
    }
    if "approved" in nested and _is_blank(values["status"]):
        values["status"] = "approved" if nested["approved"] is True else "pending"
    return values


def _canonical_manifest_record(
    record: Mapping[str, Any],
    report: Optional[ValidationReport] = None,
) -> dict[str, Any]:
    document_id = _as_document_id(record.get("document_id"))
    source = _extract_source(
        record,
        report,
        document_id=document_id or None,
        conflict_code="manifest_source_conflict",
    )
    review = _extract_review(record, document_id=document_id or None, report=report)
    adjudication = _extract_adjudication(
        record,
        document_id=document_id or None,
        report=report,
    )
    dataset_approval = _extract_dataset_approval(
        record,
        document_id=document_id or None,
        report=report,
    )
    return {
        "manifest_version": record.get("manifest_version"),
        "document_id": document_id,
        "source": source,
        "review": review,
        "adjudication": adjudication,
        "dataset_approval": dataset_approval,
    }


def _manifest_rows(records: Any) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return [records]
    try:
        return list(records)
    except TypeError as exc:
        report = ValidationReport()
        report.add_error("invalid_manifest_record")
        raise ManifestValidationError(report) from exc


def _validate_manifest_rows(
    records: Any,
    *,
    expected_version: Optional[str] = MANIFEST_VERSION,
    warnings: bool = True,
    required_source_fields: Optional[Iterable[str]] = None,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], ValidationReport]:
    rows = _manifest_rows(records)
    report = ValidationReport()
    required_fields = _required_source_field_names(required_source_fields)
    _validate_required_source_field_names(required_fields, report)
    canonical: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            report.add_error("invalid_manifest_record")
            continue
        document_id = _as_document_id(row.get("document_id"))
        issue_id = document_id or f"row-{index}"
        normalized = _canonical_manifest_record(row, report)
        canonical.append(normalized)

        version = normalized.get("manifest_version")
        if _is_blank(version):
            report.add_error(
                "missing_manifest_version",
                document_id=issue_id,
                field="manifest_version",
            )
        elif expected_version is not None and version != expected_version:
            report.add_error(
                "unsupported_manifest_version",
                document_id=issue_id,
                field="manifest_version",
            )
        if not document_id:
            report.add_error(
                "missing_manifest_document_id",
                document_id=issue_id,
                field="document_id",
            )
        elif document_id in seen:
            report.add_error(
                "duplicate_manifest_document_id",
                document_id=document_id,
                field="document_id",
            )
        else:
            seen[document_id] = index

        source = normalized["source"]
        _validate_timestamp(
            source.get("source_created_at"),
            report,
            document_id=document_id or issue_id,
            field="source_created_at",
        )
        _check_required_source_fields(
            source,
            report,
            document_id=document_id or issue_id,
            required_source_fields=required_fields,
        )
        if warnings:
            _warn_missing_provenance(
                source,
                report,
                document_id=document_id or issue_id,
                required_source_fields=required_fields,
            )

        review = normalized["review"]
        status = review.get("status")
        if not _is_blank(status) and status not in REVIEW_STATUSES:
            report.add_error(
                "invalid_review_status",
                document_id=document_id or issue_id,
                field="review.status",
            )
        label = _final_review_label(normalized)
        if _is_blank(label):
            if status in {"adjudicated", "approved_for_training"}:
                report.add_error(
                    "blank_response_label",
                    document_id=document_id or issue_id,
                    field="review.response_label",
                )
            else:
                report.add_warning(
                    "blank_response_label",
                    document_id=document_id or issue_id,
                    field="review.response_label",
                )
        if status == "excluded" and _is_blank(review.get("exclusion_reason")):
            report.add_error(
                "excluded_missing_reason",
                document_id=document_id or issue_id,
                field="review.exclusion_reason",
            )

        adjudication = normalized["adjudication"]
        adjudication_status = adjudication.get("status")
        if (
            not _is_blank(adjudication_status)
            and adjudication_status not in ADJUDICATION_STATUSES
        ):
            report.add_error(
                "invalid_adjudication_status",
                document_id=document_id or issue_id,
                field="adjudication.status",
            )
        _validate_timestamp(
            adjudication.get("reviewed_at"),
            report,
            document_id=document_id or issue_id,
            field="adjudication.reviewed_at",
        )

        approval = normalized["dataset_approval"]
        approval_status = approval.get("status")
        if not _is_blank(approval_status) and approval_status not in APPROVAL_STATUSES:
            report.add_error(
                "invalid_approval_status",
                document_id=document_id or issue_id,
                field="dataset_approval.status",
            )
        inclusion = approval.get("training_inclusion")
        if not _is_blank(inclusion) and inclusion not in TRAINING_INCLUSION_VALUES:
            report.add_error(
                "invalid_training_inclusion",
                document_id=document_id or issue_id,
                field="dataset_approval.training_inclusion",
            )
        if status == "excluded" and inclusion == "include":
            report.add_error(
                "inclusion_conflicts_with_exclusion",
                document_id=document_id or issue_id,
                field="dataset_approval.training_inclusion",
            )
        _validate_timestamp(
            approval.get("approved_at"),
            report,
            document_id=document_id or issue_id,
            field="dataset_approval.approved_at",
        )
        for field_name in (
            "annotator_1_reviewed_at",
            "annotator_2_reviewed_at",
        ):
            _validate_timestamp(
                review.get(field_name),
                report,
                document_id=document_id or issue_id,
                field=f"review.{field_name}",
            )
    return rows, canonical, report


def validate_manifest(
    records: Any,
    *,
    candidates: Any = None,
    expected_version: Optional[str] = MANIFEST_VERSION,
    required_source_fields: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Validate manifest identity, schema states, and optional candidate join."""

    required_fields = _required_source_field_names(required_source_fields)
    _rows, _canonical, report = _validate_manifest_rows(
        records,
        expected_version=expected_version,
        required_source_fields=required_fields,
    )
    if candidates is not None:
        join_report = validate_candidate_manifest_join(
            candidates,
            records,
            expected_version=expected_version,
            required_source_fields=required_fields,
        )
        report.extend(join_report)
    return report


def _source_values_equal(field_name: str, left: Any, right: Any) -> bool:
    # Timestamps are intentionally raw string comparisons.  Thus ``Z`` and
    # ``+00:00`` remain distinct source facts even when they denote the same
    # instant.
    if field_name == "source_created_at":
        return isinstance(left, str) and isinstance(right, str) and left == right
    return left == right


def validate_candidate_manifest_join(
    candidates: Any,
    manifest_records: Any,
    *,
    expected_version: Optional[str] = MANIFEST_VERSION,
    required_source_fields: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Validate a one-to-one candidate/manifest join and shared source facts."""

    required_fields = _required_source_field_names(required_source_fields)
    candidate_rows = _coerce_candidates(candidates)
    candidate_report = validate_candidate_records(
        candidate_rows,
        required_source_fields=required_fields,
    )
    _manifest_rows_value, manifest_canonical, manifest_report = _validate_manifest_rows(
        manifest_records,
        expected_version=expected_version,
        required_source_fields=required_fields,
    )
    report = ValidationReport()
    report.extend(candidate_report)
    report.extend(manifest_report)

    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            continue
        document_id = _as_document_id(row.get("document_id"))
        if document_id and document_id not in candidate_by_id:
            candidate_by_id[document_id] = row

    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_canonical:
        document_id = _as_document_id(row.get("document_id"))
        if document_id and document_id not in manifest_by_id:
            manifest_by_id[document_id] = row

    for document_id in sorted(set(candidate_by_id) - set(manifest_by_id)):
        report.add_error(
            "missing_manifest_for_candidate",
            document_id=document_id,
            field="document_id",
        )
    for document_id in sorted(set(manifest_by_id) - set(candidate_by_id)):
        report.add_error(
            "orphan_manifest_document_id",
            document_id=document_id,
            field="document_id",
        )

    for document_id in sorted(set(candidate_by_id) & set(manifest_by_id)):
        candidate = candidate_by_id[document_id]
        candidate_source = _extract_source(candidate)
        manifest_source = manifest_by_id[document_id]["source"]
        for field_name in SOURCE_FIELDS:
            left = candidate_source.get(field_name)
            right = manifest_source.get(field_name)
            if _is_blank(left) or _is_blank(right):
                continue
            if not _source_values_equal(field_name, left, right):
                report.add_error(
                    "source_timestamp_conflict"
                    if field_name == "source_created_at"
                    else "source_field_conflict",
                    document_id=document_id,
                    field=field_name,
                )
    return report


def validate_join(candidates: Any, manifest_records: Any, **kwargs: Any) -> ValidationReport:
    """Compatibility alias for :func:`validate_candidate_manifest_join`."""

    return validate_candidate_manifest_join(candidates, manifest_records, **kwargs)


validate_manifest_join = validate_candidate_manifest_join
validate_candidates = validate_candidate_records


def _final_review_label(record: Mapping[str, Any]) -> Any:
    review = _extract_review(record, document_id=None)
    if not _is_blank(review.get("response_label")):
        return review.get("response_label")
    adjudication = _extract_adjudication(record, document_id=None)
    if not _is_blank(adjudication.get("label")):
        return adjudication.get("label")
    # Flat proposal rows may carry the final task decision in an independent
    # annotator label when the review status is already approved.  Prefer the
    # first nonblank independent label only as a compatibility fallback; a
    # disagreement should be represented by an adjudication record instead.
    for field_name in ("annotator_1_label", "annotator_2_label"):
        if not _is_blank(review.get(field_name)):
            return review.get(field_name)
    return None


def _global_approval_is_explicit(approval: Any) -> tuple[bool, Optional[str]]:
    if not isinstance(approval, Mapping):
        return False, None
    status = approval.get("status", approval.get("approval_status"))
    if _is_blank(status) and approval.get("approved") is True:
        status = "approved"
    # The flat proposal has no separate status field.  A review-independent
    # approval reference and timestamp together are an explicit approval
    # record, while either field alone remains incomplete.
    if (
        (status in {None, "", "pending"})
        and not _is_blank(approval.get("approval_ref"))
        and not _is_blank(approval.get("approved_at"))
    ):
        status = "approved"
    return status == "approved", status


def validate_manifest_for_training(
    records: Any,
    *,
    candidates: Any = None,
    dataset_approval: Any = None,
    expected_version: Optional[str] = MANIFEST_VERSION,
    required_source_fields: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Apply the review gate and the separate explicit dataset gate.

    A row is eligible only when it is explicitly ``approved_for_training``, has
    a nonblank final label, and requests ``training_inclusion=include``.  A
    global ``dataset_approval`` mapping may authorize the complete batch; when
    it is omitted, every row must carry its own explicit approved gate.
    ``excluded`` and ``adjudication_required`` are always not approved.
    """

    required_fields = _required_source_field_names(required_source_fields)
    if candidates is None:
        _rows, canonical, report = _validate_manifest_rows(
            records,
            expected_version=expected_version,
            required_source_fields=required_fields,
        )
    else:
        report = validate_candidate_manifest_join(
            candidates,
            records,
            expected_version=expected_version,
            required_source_fields=required_fields,
        )
        _rows, canonical, _manifest_report = _validate_manifest_rows(
            records,
            expected_version=expected_version,
            warnings=False,
            required_source_fields=required_fields,
        )

    global_approved, global_status = _global_approval_is_explicit(dataset_approval)
    if dataset_approval is not None and not global_approved:
        code = (
            "dataset_approval_rejected"
            if global_status == "rejected"
            else "dataset_approval_incomplete"
        )
        report.add_error(code, field="dataset_approval")
    if global_approved and _is_blank(
        _mapping(dataset_approval).get("dataset_version")
    ):
        report.add_error(
            "approved_missing_dataset_version",
            field="dataset_approval.dataset_version",
        )

    for record in canonical:
        document_id = _as_document_id(record.get("document_id")) or None
        review = record["review"]
        adjudication = record["adjudication"]
        approval = record["dataset_approval"]
        status = review.get("status")
        inclusion = approval.get("training_inclusion")
        label = _final_review_label(record)

        if status == "excluded":
            report.add_error("record_excluded", document_id=document_id, field="review.status")
        elif status == "adjudication_required" or adjudication.get("status") in {
            "required",
            "pending",
        }:
            report.add_error(
                "adjudication_required",
                document_id=document_id,
                field="review.status",
            )
        elif status != "approved_for_training":
            report.add_error(
                "review_not_complete",
                document_id=document_id,
                field="review.status",
            )

        if _is_blank(label):
            report.add_error(
                "blank_response_label",
                document_id=document_id,
                field="review.response_label",
            )
        if inclusion != "include":
            report.add_error(
                "approval_missing_inclusion",
                document_id=document_id,
                field="dataset_approval.training_inclusion",
            )

        if not global_approved:
            row_approved, row_status = _global_approval_is_explicit(approval)
            if row_approved and _is_blank(approval.get("dataset_version")):
                report.add_error(
                    "approved_missing_dataset_version",
                    document_id=document_id,
                    field="dataset_approval.dataset_version",
                )
            elif not row_approved:
                if row_status == "rejected":
                    report.add_error(
                        "dataset_approval_rejected",
                        document_id=document_id,
                        field="dataset_approval.status",
                    )
                else:
                    report.add_error(
                        "dataset_approval_required",
                        document_id=document_id,
                        field="dataset_approval.status",
                    )

    report.approval_checked = True
    return report


def validate_for_training(records: Any, **kwargs: Any) -> ValidationReport:
    """Compatibility alias for :func:`validate_manifest_for_training`."""

    return validate_manifest_for_training(records, **kwargs)


def _canonicalize_for_write(record: Mapping[str, Any]) -> dict[str, Any]:
    # Validation has already checked fields.  This second canonicalization is
    # deliberately side-effect free and drops raw candidate text if a caller
    # supplied a mixed row by mistake.
    return _canonical_manifest_record(record)


def _serialize_record(record: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        report = ValidationReport()
        report.add_error("serialization_failed")
        raise ManifestValidationError(report, prefix="manifest serialization failed") from None
    return (text + "\n").encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("manifest write made no progress")
        offset += written


def write_manifest(
    path: str | os.PathLike[str],
    records: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    candidates: Any = None,
    candidate_records: Any = None,
    candidate_csv_path: str | os.PathLike[str] | None = None,
    manifest_version: str = MANIFEST_VERSION,
    expected_version: Optional[str] = None,
    required_source_fields: Optional[Iterable[str]] = None,
    validate: bool = True,
    require_approval: bool = False,
    dataset_approval: Any = None,
    overwrite: bool = False,
) -> str:
    """Exclusively create a validated JSONL manifest.

    ``overwrite`` is accepted for caller compatibility but intentionally does
    not weaken the exclusive-create contract: an existing destination is
    always preserved and raises :class:`FileExistsError`.
    """

    del overwrite  # O_EXCL is the safety contract, regardless of caller flag.
    del validate  # Manifest creation is always validation-before-create.
    required_fields = _required_source_field_names(required_source_fields)
    if records is None:
        supplied_candidates = [
            value
            for value in (candidates, candidate_records, candidate_csv_path)
            if value is not None
        ]
        if len(supplied_candidates) != 1:
            raise TypeError(
                "records or exactly one candidate input is required"
            )
        # Materialize once so a generator is not consumed by construction and
        # then observed as empty by the candidate↔manifest join validation.
        candidate_rows = _coerce_candidates(supplied_candidates[0])
        built = build_manifest_records(
            candidate_rows,
            manifest_version=manifest_version,
            required_source_fields=required_fields,
        )
        records_list: list[Mapping[str, Any]] = list(built)
        candidates_for_validation = candidate_rows
    else:
        records_list = [records] if isinstance(records, Mapping) else list(records)
        candidate_inputs = [
            value
            for value in (candidates, candidate_records, candidate_csv_path)
            if value is not None
        ]
        if len(candidate_inputs) > 1:
            raise TypeError("provide at most one candidate input")
        candidates_for_validation = candidate_inputs[0] if candidate_inputs else None

    validation_version = manifest_version if expected_version is None else expected_version
    if require_approval:
        report = validate_manifest_for_training(
            records_list,
            candidates=candidates_for_validation,
            dataset_approval=dataset_approval,
            expected_version=validation_version,
            required_source_fields=required_fields,
        )
    elif candidates_for_validation is not None:
        report = validate_candidate_manifest_join(
            candidates_for_validation,
            records_list,
            expected_version=validation_version,
            required_source_fields=required_fields,
        )
    else:
        report = validate_manifest(
            records_list,
            expected_version=validation_version,
            required_source_fields=required_fields,
        )
    report.raise_for_errors()

    # Canonicalize and serialize before claiming the destination.  A bad
    # record therefore cannot create an empty file even when validate=False.
    payloads = [_serialize_record(_canonicalize_for_write(record)) for record in records_list]

    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target)) or os.curdir
    os.makedirs(directory, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd: Optional[int] = None
    created = False
    success = False
    try:
        fd = os.open(target, flags, 0o666)
        created = True
        for payload in payloads:
            _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        success = True
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if created and not success:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
    return target


def write_jsonl_manifest(path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]], **kwargs: Any) -> str:
    """Compatibility alias for :func:`write_manifest`."""

    return write_manifest(path, records, **kwargs)


def write_manifest_from_candidates(
    path: str | os.PathLike[str],
    candidates: Any,
    **kwargs: Any,
) -> str:
    """Build and exclusively write a manifest from candidate rows/CSV."""

    return write_manifest(path, candidates=candidates, **kwargs)


def read_manifest(
    path: str | os.PathLike[str],
    *,
    validate: bool = True,
    expected_version: Optional[str] = MANIFEST_VERSION,
    required_source_fields: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Read JSONL records back without exposing raw line contents in errors."""

    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for _line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    report = ValidationReport()
                    report.add_error("blank_jsonl_line")
                    report.raise_for_errors("manifest read failed")
                try:
                    value = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    report = ValidationReport()
                    report.add_error("invalid_jsonl")
                    raise ManifestValidationError(report, prefix="manifest read failed") from None
                if not isinstance(value, dict):
                    report = ValidationReport()
                    report.add_error("jsonl_not_object")
                    report.raise_for_errors("manifest read failed")
                records.append(value)
    except UnicodeDecodeError:
        report = ValidationReport()
        report.add_error("invalid_jsonl")
        raise ManifestValidationError(report, prefix="manifest read failed") from None

    if validate:
        required_fields = _required_source_field_names(required_source_fields)
        validate_manifest(
            records,
            expected_version=expected_version,
            required_source_fields=required_fields,
        ).raise_for_errors("manifest read validation failed")
    return records


def read_jsonl_manifest(path: str | os.PathLike[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Compatibility alias for :func:`read_manifest`."""

    return read_manifest(path, **kwargs)


write_manifest_jsonl = write_manifest
read_manifest_jsonl = read_manifest
write_jsonl = write_manifest
read_jsonl = read_manifest


__all__ = [
    "ADJUDICATION_FIELDS",
    "ADJUDICATION_STATUSES",
    "APPROVAL_STATUSES",
    "CURRENT_MANIFEST_VERSION",
    "DATASET_APPROVAL_FIELDS",
    "MANIFEST_FIELDS",
    "MANIFEST_VERSION",
    "ManifestRecords",
    "ManifestValidationError",
    "OPTIONAL_PROVENANCE_FIELDS",
    "REVIEW_FIELDS",
    "REVIEW_STATUSES",
    "REVIEW_STATUS_VALUES",
    "SOURCE_FIELDS",
    "SOURCE_PROVENANCE_FIELDS",
    "TRAINING_INCLUSION_VALUES",
    "DATASET_APPROVAL_STATUS_VALUES",
    "ValidationIssue",
    "ValidationReport",
    "build_manifest",
    "build_manifest_records",
    "build_review_manifest",
    "create_manifest",
    "create_manifest_records",
    "create_review_manifest",
    "load_candidate_records",
    "read_candidate_records",
    "read_jsonl_manifest",
    "read_manifest",
    "read_manifest_jsonl",
    "read_jsonl",
    "validate_candidate_manifest_join",
    "validate_candidate_records",
    "validate_for_training",
    "validate_join",
    "validate_manifest",
    "validate_manifest_for_training",
    "validate_manifest_join",
    "validate_candidates",
    "write_jsonl_manifest",
    "write_manifest",
    "write_manifest_jsonl",
    "write_jsonl",
    "write_manifest_from_candidates",
]
