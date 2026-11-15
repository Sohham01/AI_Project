# REVIEW MANIFEST — IMPLEMENTED V1 CONTRACT

**Implementation status:** The offline JSONL writer and validator live in
`backend/app/services/annotation/manifest.py`. This document still requires
human approval for the project's source-retention, reviewer-reference, and
dataset-approval policy; implementing the contract does not approve any data.

## Purpose

The current task training CSVs are intentionally small label-bearing contracts.
They do not preserve enough source provenance or human-review state to establish
that a final dataset is authorized, traceable, or adjudicated. This contract
defines a **separate sidecar manifest** for human review. It is not consumed by
the current task loaders, is not a training dataset, and must not be appended to
the templates without a reviewed loader/schema change.

The sidecar is versioned and joined to a task CSV by `document_id`. A row may be
present in the manifest while excluded from training; exclusion is part of the
review record, not a reason to delete the source audit trail. The implementation
serializes one compact JSON object per line and never overwrites an existing
destination.

## Implemented JSONL shape

Each record has these top-level keys, in addition to the stable join key:

```json
{
  "manifest_version": "1.0",
  "document_id": "doc-123",
  "source": {
    "source_provider": null,
    "source_mode": null,
    "post_id": null,
    "comment_id": null,
    "source_url": null,
    "source_created_at": null,
    "context_snapshot_ref": null,
    "source_record_hash": null
  },
  "review": {
    "status": "unreviewed",
    "response_label": null,
    "annotator_1_ref": null,
    "annotator_1_label": null,
    "annotator_1_reviewed_at": null,
    "annotator_1_notes_ref": null,
    "annotator_2_ref": null,
    "annotator_2_label": null,
    "annotator_2_reviewed_at": null,
    "annotator_2_notes_ref": null,
    "exclusion_reason": null
  },
  "adjudication": {
    "status": null,
    "ref": null,
    "label": null,
    "reviewed_at": null,
    "reason_ref": null
  },
  "dataset_approval": {
    "status": "pending",
    "dataset_version": null,
    "guideline_version": null,
    "approval_ref": null,
    "approved_at": null,
    "training_inclusion": "pending"
  }
}
```

The writer copies source facts from candidate mappings or a local candidate CSV
when present, but never copies `text`, `normalized_text`, parent text, or post
title into this audit sidecar. Missing optional source fields remain `null` and
produce structured `missing_optional_provenance` warnings. Source timestamps
must parse as ISO-8601, while candidate/manifest timestamp comparisons retain
the original strings exactly; `Z` and `+00:00` are not silently normalized. The
sample above is abbreviated: the writer emits every key in `SOURCE_FIELDS`,
including `source_subreddit`, `comment_id`, parent/context status, score, and
language metadata, with `null` for unavailable values. It copies a supplied
source hash but does not compute one from raw text or fetch additional source
data; a release owner can require a supplied hash with
`required_source_fields=("source_record_hash",)`.

Use the standard-library API below for an offline batch:

```python
from app.services.annotation.manifest import (
    build_manifest_records,
    validate_candidate_manifest_join,
    validate_manifest_for_training,
    write_manifest,
)

records = build_manifest_records(candidate_csv_path)
write_manifest(output_path, records)
```

`build_manifest_records()` accepts in-memory candidate rows or a local CSV path.
`validate_candidate_manifest_join()` enforces a one-to-one `document_id` join,
checks shared source facts and exact timestamps, and reports orphan/missing
records. `validate_manifest_for_training()` is a stricter gate: it requires
completed review, a final label, `training_inclusion=include`, and explicit
dataset approval. Required provenance is policy-driven through
`required_source_fields`; it is never filled by inference. These functions do
not access the database or providers. The minimal implemented approval gate also
requires an approved `dataset_version`; approval reference/time and any stricter
authority or reviewer policy remain human decisions.

## Logical field inventory

The table uses logical field names for readability. In the serialized JSONL,
source fields are nested under `source`, review fields under `review`,
adjudication fields under `adjudication`, and dataset-approval fields under
`dataset_approval`.

| Field | Type | Required | Purpose |
| --- | --- | --- | --- |
| `manifest_version` | string | Yes | Version of this sidecar contract; the current writer emits `1.0`. |
| `dataset_approval.dataset_version` | string | Yes before freeze | Human-approved batch/version identifier. |
| `document_id` | string | Yes | Exact join key to the task CSV; stable and unique within the dataset version. |
| `source_provider` | string | Yes when known | Provider/source name recorded without inventing one when unknown. |
| `source_mode` | string | Yes when known | Explicit provenance mode, such as `direct` or `context`; never inferred from a subreddit name. |
| `source_subreddit` | string | Yes when known | Community/subreddit from the source record. |
| `post_id` | string | Yes when known | Enclosing thread/post identity. |
| `comment_id` | string | Conditional | Source comment identity for comment records. |
| `parent_comment_id` | string | Conditional | Parent comment identity for replies; blank only for top-level comments or genuinely unavailable parents. |
| `source_url` | string | Conditional | Authoritative source permalink when the provider supplies one. Do not reconstruct it from hints. |
| `source_created_at` | ISO-8601 string | Yes when supplied | Original source timestamp, retaining its known offset/UTC convention. |
| `context_snapshot_ref` | string | Yes for review | Pointer/hash to the retained review-context bundle without copying raw text into the manifest. |
| `source_record_hash` | string | Policy-required for release | Stable hash of the approved source-record representation used for review; never fabricated by the exporter. |
| `guideline_version` | string | Required after review | Task guideline/taxonomy version used by the annotator. |
| `review_status` | enum | Yes | `unreviewed`, `annotated_1`, `annotated_2`, `adjudication_required`, `adjudicated`, `excluded`, or `approved_for_training`. |
| `exclusion_reason` | string | Conditional | Required for `excluded`; examples include insufficient context, duplicate, source mismatch, or policy exclusion. |
| `annotator_1_ref` | privacy-preserving string | Conditional | Approved reviewer reference, not a raw personal identifier. |
| `annotator_1_label` | task-specific string | Conditional | Independent decision before adjudication; may be blank only for an explicitly unresolved review. |
| `annotator_1_reviewed_at` | ISO-8601 string | Conditional | Time of independent review. |
| `annotator_1_notes_ref` | string | Optional | Pointer to review notes kept under the approved privacy policy. |
| `annotator_2_ref` | privacy-preserving string | Conditional | Second independent reviewer reference. |
| `annotator_2_label` | task-specific string | Conditional | Second independent decision. |
| `annotator_2_reviewed_at` | ISO-8601 string | Conditional | Time of second independent review. |
| `adjudicator_ref` | privacy-preserving string | Conditional | Reviewer who resolved a disagreement. |
| `adjudicated_label` | task-specific string | Conditional | Final decision after adjudication; must not overwrite independent labels. |
| `adjudicated_at` | ISO-8601 string | Conditional | Adjudication time. |
| `adjudication_reason_ref` | string | Conditional | Pointer to the rationale/guideline rule used. |
| `approval_ref` | string | Conditional | Human approval record for dataset inclusion. |
| `approved_at` | ISO-8601 string | Conditional | Approval time. |
| `training_inclusion` | enum | Yes before freeze | `include`, `exclude`, or `pending`. |

## Why this is separate from the current loaders

The existing ingestion models provide only part of the source contract:

- `PostData` contains post ID, title/body, subreddit, timestamp, URL, and
  comments.
- `CommentData` contains comment ID, body, timestamp, score, and parent ID.
- The response candidate export adds original text, normalized text, parent text,
  post title, subreddit, score, and `created_at` for that specific workflow.
- The five task annotation CSV schemas do not include source URL, timestamp,
  subreddit, parent text/ID, provider/source mode, review status, or approval
  fields.

Therefore the manifest is a compatibility boundary, not an assertion that
current loaders preserve all fields. The implemented reader/writer/validator
keeps the sidecar and training CSV separate and requires an exact
`document_id` join. Any future task-loader extension must update loaders, tests,
and documentation together rather than silently appending sidecar fields.

## Status transitions

```text
unreviewed
  -> annotated_1
  -> annotated_2
  -> adjudication_required (when decisions disagree or context is insufficient)
  -> adjudicated (when a reviewer resolves the case)
  -> approved_for_training
```

Any state may move to `excluded` with a documented reason. `approved_for_training`
requires `training_inclusion=include`, a valid task label in the task CSV, source
traceability, and completion of the dataset approval checklist. A manifest state
does not make a task label valid; the task loader remains authoritative for CSV
shape and label semantics.

## Privacy and retention

Do not put usernames, credentials, access tokens, or unnecessary raw personal
data in this manifest. Use approved reviewer references and pointers/hashes for
large context or note bundles. Retain only the source context required for a
defensible annotation and the project's privacy policy.
