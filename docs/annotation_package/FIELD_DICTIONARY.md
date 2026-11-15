# DRAFT — HUMAN REVIEW REQUIRED

## Field dictionary for the current training CSV contracts

This dictionary describes the fields accepted by the existing loaders. It does
not add provenance or review fields to those loaders. Blank template files are
under `data/annotations/templates/`.

### Shared fields

| Field | Type in CSV | Required by loader | Meaning and validation |
| --- | --- | --- | --- |
| `document_id` | nonblank string | Yes | Stable corpus identity. Must be unique in an approved dataset version. The loaders preserve duplicates for inspection rather than universally rejecting them. |
| `text` | string | Yes | Text presented to the task model. Existing task docs expect normalized/preprocessed text for training; preserve original text outside this CSV in the review material. Blank text is not a usable training example even where a low-level loader can construct it. |
| `post_id` | string | No | Enclosing source post/thread identity. Preserve when available; use an empty cell only when the source unit genuinely has no post ID. It is used for group-aware leakage control, not as a label. |
| `comment_id` | string | No | Source comment identity when the document is a comment. Preserve the distinction between post and comment. |
| `language` | string | No | Language metadata, default `unknown`. Not a task label. |
| `script` | string | No | Script metadata, default `unknown`. Not a task label. |
| `is_code_mixed` | boolean token | No | Existing parsers accept common truthy spellings and otherwise parse false; use `true`/`false` explicitly. |
| `detected_languages` | pipe-separated string | No | Metadata list using `|`, for example `hi|en`. Blank means no detected list, not “unreviewable.” |

The current CSV loaders ignore unknown extra columns in several task packages,
but the templates intentionally use the canonical column lists. Do not rely on
ignored columns for provenance or review state.

## Task fields

### Sentiment: `sentiment_annotations.template.csv`

| Field | Type | Required | Allowed values |
| --- | --- | --- | --- |
| `sentiment` | nonblank string | Yes for final training | Free-form/annotator-defined by the current schema. A pipe is rejected because sentiment is single-label. A controlled taxonomy is a human decision, not imposed by code. |

The current `SentimentAnnotation` schema requires a nonblank single value when
validated. There is no final-training abstention value.

### Emotion: `emotion_annotations.template.csv`

| Field | Type | Required | Allowed values |
| --- | --- | --- | --- |
| `emotions` | pipe-separated string | Yes as a column; cell may be blank | Free-form/annotator-defined labels. `stress|frustration` means two labels. A blank cell is a valid empty emotion set. Empty parts are ignored by the low-level parser, so the approval check must reject malformed cells such as `stress||frustration`. |

The current schema permits zero, one, or many emotion labels. A valid empty set
must be distinguished from insufficient context in the review sidecar.

### Issue: `issue_annotations.template.csv`

| Field | Type | Required | Allowed values |
| --- | --- | --- | --- |
| `labels` | pipe-separated string | Yes as a column; cell may be blank | Free-form/annotator-defined issue names. A blank cell means no issue labels. |

The current issue annotation schema accepts zero, one, or many labels and does
not define an English or domain taxonomy in code. The approved guideline must
define the vocabulary and boundaries before training.

### Response: `response_annotations.template.csv`

| Field | Type | Required | Allowed values |
| --- | --- | --- | --- |
| `response_type` | single string | Yes for final training | Exactly one of the canonical ordered values below. |

Canonical response order:

1. `agreement`
2. `personal_experience`
3. `opinion`
4. `advice`
5. `solution`
6. `counterargument`

The current `ResponseAnnotation` validator rejects blank or unrecognized values.
The candidate-review workflow may keep `response_type` blank before
adjudication, but the converter skips blank rows and the final training schema
does not support an abstention value.

### Solution: `solution_annotations.template.csv`

| Field | Type | Required | Allowed values |
| --- | --- | --- | --- |
| `is_solution` | boolean token | Yes | `true`/`false` or another spelling accepted by `parse_bool_cell`; use lowercase `true`/`false` for reproducibility. Blank and unrecognized values are rejected, not treated as false. |
| `solution_type` | optional string | No | Free-form annotation metadata. It is reported for coverage and is not learned by the V1 classifier. |
| `response_type` | optional string | No | Optional upstream/context metadata; it does not determine `is_solution`. If populated, use the canonical response vocabulary. |
| `rationale` | optional string | No | Human explanation of the decision; accepted by the annotation schema but not used as the V1 target. |
| `evidence` | optional string | No | Supporting span/reference if the review process uses one; not used as the V1 target. |

The solution classifier learns only `is_solution`. A helpful or sympathetic
comment is not automatically a solution; the decision requires an actionable,
issue-relevant recommendation or fix.

## Review-manifest fields are separate

The current training CSVs cannot carry the complete review/provenance contract.
The implemented V1 sidecar is written as JSONL by
`backend/app/services/annotation/manifest.py` and is documented in
`REVIEW_MANIFEST_PROPOSAL.md`. It keeps these categories separate:

| JSONL section | Contents | Initial state from a candidate export |
| --- | --- | --- |
| `source` | Source/provider facts, identities, timestamps, context status, and hashes | Copied when present; missing optional fields remain `null` and produce warnings |
| `review` | Review status, task label, annotator references/labels/times, notes reference, and exclusion reason | `status="unreviewed"`; all decision fields unset |
| `adjudication` | Adjudicator reference, final label, decision time, and reason reference | All fields unset |
| `dataset_approval` | Dataset/guideline versions, approval reference/time, and training inclusion | `status="pending"`; `training_inclusion="pending"` |

The sidecar is joined to candidates and task CSVs only by unique, nonblank
`document_id`. `validate_candidate_manifest_join()` reports missing, orphan,
duplicate, contradictory-source, and exact-timestamp conflicts. It accepts
legacy candidate CSVs that omit newer optional context columns. Raw source text
is never copied into the manifest or included in validation errors. The
source-policy caller may pass `required_source_fields=(...)`; those omissions
are `missing_required_provenance` errors, while other conditional omissions
remain `missing_optional_provenance` warnings. These fields must not be silently
added to a final training CSV or treated as loader-supported without an approved
schema change.

## Response-candidate review fields

The response candidate CSV is a review-only hand-off, not one of the five final
training schemas. In addition to the shared/task fields above, its current
review columns include:

| Field | Type | Required for a released pilot | Meaning |
| --- | --- | --- | --- |
| `normalized_text` | string | Yes before conversion | Separate preprocessor output; never silently falls back to original text. |
| `parent_text` | string | When parent is available | Parent body supplied for review context. |
| `post_title` | string | When available | Enclosing post title. |
| `score` | integer string | Optional | Blank means unavailable; `0` is a genuine score. Malformed supplied values are rejected. |
| `created_at` | string | Yes when available | Source timestamp; not replaced with current time. |
| `source_url` | string | Yes when available | Authoritative source permalink; not reconstructed from hints. |
| `source_provider` | string | Yes when known | Provider/source identity. |
| `source_mode` | `direct`/`context` | Yes when known | Explicit provenance mode. |
| `parent_context_status` | enum | Yes for a released batch | `top_level`, `available`, `unavailable`, or `unknown`; makes missing parent retrieval visible. |

The candidate writer also requires unique nonblank `document_id` and
`comment_id`, while allowing multiple rows to share one `parent_comment_id`.
`response_type` must be blank until human review. The candidate-to-manifest join
is by unique `document_id`; use the manifest builder and validator before
releasing a batch.

## Type and formatting checklist

- Use UTF-8 CSV with the header in the template's exact order.
- Quote text containing commas, quotes, or newlines using normal CSV quoting.
- Use `|` only inside a multilabel cell; do not use it to create a new class.
- Do not use `None`, `null`, `unknown`, `unclear`, or `not_applicable` as a
  substitute for an unapproved task label.
- Preserve source IDs as strings; do not renumber rows as identities.
- Keep original and normalized text separate in review materials. The current
  task training CSV uses the normalized text contract documented by each task.
