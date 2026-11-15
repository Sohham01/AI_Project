# Authorized Static-Dataset Intake

## Status

This is a local-only intake path for a bounded, user-supplied Reddit export. It
does not collect data, call a provider, write an export, approve a dataset, or
start annotation/training. The absence of genuine Reddit data in this
repository remains unresolved until an authorized source file is supplied.

Structural validity and authorization are separate decisions. A structurally
valid export may have `authorization_status="pending-review"` and may be
inspected, but it cannot enter the approved static-pilot workflow until the
metadata status is exactly `approved` and the responsible operator explicitly
confirms approval. This code-level gate does not establish legal rights or
replace the supplier's actual authorization.

## Supported formats

`app.services.ingestion.static_export.load_static_export()` accepts UTF-8 JSON
and UTF-8 JSONL/NDJSON. It rejects CSV for now because a flat CSV does not
define comment grouping and dataset-level provenance without an additional
contract. The loader uses only the Python standard library plus the existing
`PostData`/`CommentData` models.

The input file is read-only. It is bounded by `StaticExportLimits` before
records are admitted to the in-memory result. The default limits are:

| Limit | Default |
| --- | ---: |
| File size | 10 MiB |
| Posts | 200 |
| Comments per post | 2,000 |
| Comments overall | 5,000 |

Callers may lower or raise these limits explicitly for an approved local
sample. A limit violation is an error; the loader does not silently truncate
the source.

## JSON contract

The top level must contain an integer `format_version: 1`, a `metadata` object,
and a `posts` array. Boolean `true` is not accepted as a version value.

```json
{
  "format_version": 1,
  "metadata": {
    "dataset_id": "supplier-chosen-version",
    "source": {
      "community": "IIMSambalpur",
      "source_reference": "supplier-record-or-snapshot-reference",
      "source_provider": "supplier-or-original-provider",
      "source_mode": "direct",
      "source_url": "https://www.reddit.com/r/IIMSambalpur/",
      "collected_at": "2026-01-15T12:00:00Z",
      "snapshot_at": "2026-01-15T12:00:00Z"
    },
    "completeness": {
      "comments_complete": false
    },
    "authorization": {
      "authorization_status": "approved-or-pending",
      "authorization_reference": "approval-record-reference"
    }
  },
  "posts": [
    {
      "post_id": "abc123",
      "subreddit": "IIMSambalpur",
      "title": "Original title",
      "body": "Original post body or null",
      "created_at": "2026-01-15T12:00:00Z",
      "score": 4,
      "num_comments": 1,
      "url": "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title/",
      "comments": [
        {
          "comment_id": "def456",
          "body": "Original comment body",
          "created_at": "2026-01-15T12:10:00Z",
          "score": 2,
          "parent_comment_id": null
        }
      ]
    }
  ]
}
```

### Required source and authorization metadata

The loader requires the following values and preserves them without deciding
whether they are true or sufficient:

- `metadata.dataset_id`;
- `metadata.source.community`;
- `metadata.source.source_reference`;
- `metadata.source.source_provider`;
- `metadata.source.source_mode`, exactly `direct` or `context`;
- `metadata.authorization.authorization_status`;
- `metadata.authorization.authorization_reference`.

The authorization object is evidence supplied by the dataset owner. Its
presence does not turn a pending or unreviewed export into an approved dataset.
The loader does not infer an owner, license, source URL, collection date, or
approval state.

`source_url`, `collected_at`, and `snapshot_at` are optional metadata. If
present, URLs must be HTTP(S), and timestamps must be timezone-qualified ISO
8601 strings or UTC epoch values. A timezone-qualified timestamp is normalized
to the existing provider convention of naive UTC in `PostData` and
`CommentData`; the supplied metadata remains available in the returned result.

`metadata.completeness.comments_complete` is optional. When absent, the loader
reports `comments_completeness_unknown`; when explicitly false, it reports
`comments_incomplete`. Neither the absence of the field nor a true value proves
that every Reddit comment was available. A missing parent is always retained
and reported separately.

## Record requirements

Required to create a normalized post:

- nonblank `post_id`, `subreddit`, and `title`;
- timezone-qualified `created_at` or a numeric UTC epoch;
- `comments` as an array when the field is supplied.

Required to create a normalized comment:

- nonblank `comment_id` and `body`;
- timezone-qualified `created_at` or a numeric UTC epoch.

`parent_comment_id: null` means top-level. A non-null value must identify a
comment; it is retained even when that parent is absent from the bounded
export, and `unresolved_comment_parent` is reported. The loader never replaces
an absent parent with the post ID or a fabricated identifier.

`body`, `url`, `score`, and `num_comments` are optional in the source contract.
The existing normalized schemas represent missing numeric values with their
historical defaults (`0`), so the loader reports the corresponding
`*_missing` warning and records the field in `lossy_default_fields`. A result
with such fields is not `usable_for_pilot`; this prevents a caller from
mistaking a schema default for an observed score or count. Missing post body or
URL remains representable as `None`.

Duplicate identical nested-JSON IDs are removed with a warning. Conflicting
records with the same ID produce an error. A comment ID repeated under
different posts produces `duplicate_comment_id_across_posts` and makes the
result invalid. Parent cycles produce `comment_parent_cycle` diagnostics and
make the result invalid. Malformed records are rejected individually and
reported by bounded diagnostic code; source text is never included in the
diagnostics.

## JSONL contract

The first line must be a metadata record with integer `format_version: 1`:

```json
{"record_type":"metadata","format_version":1,"metadata":{...}}
{"record_type":"post","post":{"post_id":"abc123", "subreddit":"IIMSambalpur", "title":"...", "created_at":"2026-01-15T12:00:00Z", "score":4, "num_comments":0}}
{"record_type":"comment","post_id":"abc123","comment":{"comment_id":"def456", "body":"...", "created_at":"2026-01-15T12:10:00Z", "score":2, "parent_comment_id":null}}
```

The `post_id` on a comment line links it to a post record. Comments for a
missing post are rejected. Repeated JSONL post IDs are handled consistently
with nested JSON: identical records produce a warning and are deduplicated;
conflicting records produce an error. JSONL post and comment records may occur
in either order after the metadata line because comments are associated after
the file is read.

## Workflow when a file is supplied

1. Obtain the file and the source/authorization record from the supplier. Do
   not fill missing provenance from memory or from the requested subreddit.
2. Run `load_static_export(path, limits=...)` and retain only its safe summary,
   diagnostics, metadata, and counts for the intake review.
3. Require `result.usable_for_pilot` and separately review the authorization,
   source reference, completeness limitations, and any warnings.
4. For a genuine static export, use the explicit authorization gate. The
   metadata status must be exactly `approved`, and the responsible operator
   must supply `approval_confirmed=True`:

   ```python
   from app.services.ingestion.static_export import load_static_export
   from app.services.pilot import prepare_authorized_static_pilot

   result = load_static_export("/approved/local/export.json")
   pilot = prepare_authorized_static_pilot(
       result,
       approval_confirmed=True,
   )
   ```

   A pending or structurally invalid export raises before records reach the
   pilot. The pilot summary retains bounded dataset ID, source reference,
   provider, source mode, authorization reference/status, and completeness
   limitation codes. Full source text is not copied into provenance metadata.
   The existing post/comment and candidate/manifest contracts remain unchanged.
5. Inspect the structured summaries. Language detection and preprocessing can
   run without trained models; model stages remain unavailable unless an
   explicitly approved model is injected.
6. Defer candidate export, manifest writing, human annotation, adjudication,
   dataset approval, and supervised training until those separate approvals
   exist.

The lower-level `prepare_end_to_end_pilot()` remains a generic in-memory bridge
for synthetic or already-authorized normalized records. It is not an
authorization mechanism. Genuine static exports must use
`prepare_authorized_static_pilot()`.

This workflow proves static-import compatibility when run with a supplied
genuine export. Synthetic tests prove only the adapter, integrity checks, gate,
and handoff behavior; they do not establish genuine Reddit compatibility or
source authorization.
