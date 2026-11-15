"""Offline preparation bridge for the first bounded end-to-end pilot.

This module wires existing components together without creating files, touching
PostgreSQL, calling a provider, loading artifacts, or fitting models:

``PostData`` snapshots -> aware downstream records -> language/preprocessing
pipeline -> review candidates -> in-memory review manifest -> join validation.

The bridge makes the ingestion timestamp convention explicit at the boundary:
provider ``datetime`` values are naive UTC, while the downstream contract
requires timezone-aware timestamps. It does not claim thread completeness or
model readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.schemas.ingestion import CommentData, PostData
from app.services.annotation.candidates import build_candidate_rows
from app.services.annotation.manifest import (
    ManifestRecords,
    ValidationReport,
    build_manifest_records,
    validate_candidate_manifest_join,
    validate_candidate_records,
)
from app.services.nlp.pipeline import DownstreamPipeline, PipelineResult
from app.services.ingestion.static_export import StaticExportResult


def _aware_utc(value: datetime) -> datetime:
    """Acknowledge the provider's naive-UTC convention at the NLP boundary."""

    if not isinstance(value, datetime):
        raise ValueError("source timestamp is required")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _source_mode(value: str) -> str:
    if value not in {"direct", "context"}:
        raise ValueError("source_mode must be direct or context")
    return value


def _safe_dataset_provenance(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep only bounded dataset references in the in-memory pilot summary."""

    if value is None:
        return None
    allowed_text = (
        "dataset_id",
        "source_reference",
        "source_community",
        "source_provider",
        "source_mode",
        "authorization_reference",
        "authorization_status",
    )
    result: dict[str, Any] = {}
    for field_name in allowed_text:
        field_value = value.get(field_name)
        if field_value is not None:
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"dataset provenance field {field_name} must be nonblank text")
            result[field_name] = field_value.strip()
    completeness = value.get("completeness")
    if completeness is not None:
        if not isinstance(completeness, Mapping):
            raise ValueError("dataset completeness provenance must be an object")
        safe_completeness: dict[str, Any] = {}
        if "comments_complete" in completeness:
            if not isinstance(completeness["comments_complete"], bool):
                raise ValueError("comments_complete provenance must be boolean")
            safe_completeness["comments_complete"] = completeness["comments_complete"]
        limitations = completeness.get("limitation_codes", ())
        if not isinstance(limitations, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in limitations
        ):
            raise ValueError("completeness limitation codes must be text")
        safe_completeness["limitation_codes"] = sorted(set(item.strip() for item in limitations))
        result["completeness"] = safe_completeness
    return result


def _static_export_provenance(export: StaticExportResult) -> dict[str, Any]:
    source = export.metadata["source"]
    authorization = export.metadata["authorization"]
    limitation_codes = sorted(
        {
            issue.code
            for issue in export.diagnostics
            if issue.code
            in {
                "comments_completeness_unknown",
                "comments_incomplete",
                "unresolved_comment_parent",
                "fewer_comments_than_reported",
            }
        }
    )
    completeness = export.metadata.get("completeness", {})
    safe_completeness: dict[str, Any] = {
        "limitation_codes": limitation_codes,
    }
    if isinstance(completeness, Mapping) and isinstance(
        completeness.get("comments_complete"), bool
    ):
        safe_completeness["comments_complete"] = completeness["comments_complete"]
    return {
        "dataset_id": export.metadata["dataset_id"],
        "source_reference": source["source_reference"],
        "source_community": source["community"],
        "source_provider": source["source_provider"],
        "source_mode": source["source_mode"],
        "authorization_reference": authorization["authorization_reference"],
        "authorization_status": authorization["authorization_status"],
        "completeness": safe_completeness,
    }


def _comment_rows(
    posts: Sequence[PostData],
    *,
    source_provider: str,
    source_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for post in posts:
        comments_by_id = {comment.reddit_id: comment for comment in post.comments}
        for comment in post.comments:
            parent = (
                comments_by_id.get(comment.parent_reddit_id)
                if comment.parent_reddit_id
                else None
            )
            parent_status = (
                "top_level"
                if comment.parent_reddit_id is None
                else ("available" if parent is not None else "unavailable")
            )
            rows.append(
                {
                    "document_id": f"comment-{comment.reddit_id}",
                    "comment_id": comment.reddit_id,
                    "text": comment.body,
                    "response_type": "",
                    "post_id": post.reddit_id,
                    "parent_comment_id": comment.parent_reddit_id or "",
                    "parent_text": parent.body if parent is not None else "",
                    "post_title": post.title,
                    "subreddit": post.subreddit,
                    "score": comment.score,
                    "created_at": _aware_utc(comment.created_at).isoformat(),
                    "source_url": post.url or "",
                    "source_provider": source_provider,
                    "source_mode": source_mode,
                    "parent_context_status": parent_status,
                }
            )
    return rows


def _pipeline_rows(
    posts: Sequence[PostData],
    *,
    source_mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for post in posts:
        comments_by_id = {comment.reddit_id: comment for comment in post.comments}
        for comment in post.comments:
            parent = (
                comments_by_id.get(comment.parent_reddit_id)
                if comment.parent_reddit_id
                else None
            )
            records.append(
                {
                    "document_id": f"comment-{comment.reddit_id}",
                    "text": comment.body,
                    "original_text": comment.body,
                    "created_at": _aware_utc(comment.created_at),
                    "source_mode": source_mode,
                    "source_subreddit": post.subreddit,
                    "post_id": post.reddit_id,
                    "comment_id": comment.reddit_id,
                    "parent_comment_id": comment.parent_reddit_id,
                    "parent_reddit_id": (
                        f"t1_{comment.parent_reddit_id}"
                        if comment.parent_reddit_id
                        else None
                    ),
                    "parent_comment_text": parent.body if parent is not None else None,
                    "url": post.url,
                }
            )
    return records


@dataclass
class PilotPreparationResult:
    """In-memory outputs and checks for one synthetic or authorized snapshot."""

    pipeline_result: PipelineResult
    candidate_rows: list[dict[str, Any]]
    manifest_records: ManifestRecords
    candidate_report: ValidationReport
    join_report: ValidationReport
    summary: dict[str, Any]


def prepare_end_to_end_pilot(
    posts: Sequence[PostData],
    *,
    source_provider: str,
    source_mode: str,
    pipeline: Optional[DownstreamPipeline] = None,
    required_source_fields: Optional[Iterable[str]] = None,
    dataset_provenance: Optional[Mapping[str, Any]] = None,
) -> PilotPreparationResult:
    """Exercise the offline handoff without writing candidates or manifests.

    The returned candidate rows and manifest records are intentionally in
    memory. Callers must make a separate, authorized decision before exporting
    anything to disk or starting annotation.
    """

    if not isinstance(source_provider, str) or not source_provider.strip():
        raise ValueError("source_provider is required")
    mode = _source_mode(source_mode)
    safe_provenance = _safe_dataset_provenance(dataset_provenance)
    if not isinstance(posts, Sequence):
        raise TypeError("posts must be a bounded sequence")

    pipeline_instance = pipeline if pipeline is not None else DownstreamPipeline()
    pipeline_rows = _pipeline_rows(posts, source_mode=mode)
    pipeline_result = pipeline_instance.run(pipeline_rows, discussion_complete=False)

    candidate_input = _comment_rows(
        posts,
        source_provider=source_provider.strip(),
        source_mode=mode,
    )
    candidate_rows = build_candidate_rows(
        candidate_input,
        preprocessor=pipeline_instance.preprocessor,
        language_service=pipeline_instance.language_service,
    )
    candidate_report = validate_candidate_records(candidate_rows)
    manifest_records = build_manifest_records(
        candidate_rows,
    )
    join_report = validate_candidate_manifest_join(
        candidate_rows,
        manifest_records,
        required_source_fields=required_source_fields,
    )

    summary = {
        "status": "ready_for_review"
        if candidate_report.valid and join_report.valid
        else "blocked",
        "posts_received": len(posts),
        "comments_received": sum(len(post.comments) for post in posts),
        "candidate_rows": len(candidate_rows),
        "manifest_rows": len(manifest_records),
        "pipeline_status": pipeline_result.status,
        "pipeline_models": {
            stage.name: stage.status for stage in pipeline_result.stages
        },
        "candidate_errors": candidate_report.error_codes,
        "candidate_warnings": candidate_report.warning_codes,
        "manifest_warnings": [issue.code for issue in manifest_records.warnings],
        "join_errors": join_report.error_codes,
        "writes_performed": False,
        "training_performed": False,
        "discussion_complete_asserted": False,
        "dataset_provenance": safe_provenance,
    }
    return PilotPreparationResult(
        pipeline_result=pipeline_result,
        candidate_rows=candidate_rows,
        manifest_records=manifest_records,
        candidate_report=candidate_report,
        join_report=join_report,
        summary=summary,
    )


def prepare_authorized_static_pilot(
    export: StaticExportResult,
    *,
    approval_confirmed: bool,
    pipeline: Optional[DownstreamPipeline] = None,
    required_source_fields: Optional[Iterable[str]] = None,
) -> PilotPreparationResult:
    """Pass a structurally valid static export through an explicit approval gate.

    ``approval_confirmed`` is an operator assertion from the responsible
    project authority.  The metadata status must independently be exactly
    ``approved``.  Neither condition establishes legal permission by itself.
    """

    if not isinstance(export, StaticExportResult):
        raise TypeError("export must be a StaticExportResult")
    if approval_confirmed is not True:
        raise PermissionError("explicit static-export approval is required")
    if not export.usable_for_pilot:
        raise ValueError("static export is not structurally usable for the pilot")
    if not export.authorization_approved:
        raise PermissionError("static export authorization status is not approved")

    source = export.metadata["source"]
    return prepare_end_to_end_pilot(
        export.posts,
        source_provider=source["source_provider"],
        source_mode=source["source_mode"],
        pipeline=pipeline,
        required_source_fields=required_source_fields,
        dataset_provenance=_static_export_provenance(export),
    )


__all__ = [
    "PilotPreparationResult",
    "prepare_authorized_static_pilot",
    "prepare_end_to_end_pilot",
]
