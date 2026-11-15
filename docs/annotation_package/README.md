# DRAFT — HUMAN REVIEW REQUIRED

## Human annotation preparation package

**Status:** preparation only. This package contains blank, schema-compatible
templates and review instructions. It contains no labels, no approved dataset,
no reviewer identities, no approval decision, and no model-training data.

This package does **not** replace `data/annotations/language_annotations.csv`,
which is preserved unchanged and is not a supervised training dataset. Do not
copy its rows into any of the templates.

## Package contents

| Path | Purpose |
| --- | --- |
| `data/annotations/templates/sentiment_annotations.template.csv` | Header-only sentiment training-template contract |
| `data/annotations/templates/emotion_annotations.template.csv` | Header-only emotion training-template contract |
| `data/annotations/templates/issue_annotations.template.csv` | Header-only issue training-template contract |
| `data/annotations/templates/response_annotations.template.csv` | Header-only response training-template contract |
| `data/annotations/templates/solution_annotations.template.csv` | Header-only solution training-template contract |
| `backend/app/services/annotation/manifest.py` | Offline JSONL review-manifest builder, validator, reader, and exclusive writer |
| `backend/app/services/pilot.py` | In-memory ingestion-to-pipeline-to-candidate/manifest preparation bridge; never writes outputs |
| `docs/annotation_package/FIELD_DICTIONARY.md` | Field meanings, types, requiredness, and allowed values |
| `docs/annotation_package/REVIEW_MANIFEST_PROPOSAL.md` | Implemented V1 provenance/review sidecar contract; still subject to human policy approval |
| `docs/small_dataset_pilot.md` | Bounded Universal-scraper pilot request, credit, provenance, and acceptance plan |

The five CSVs intentionally contain only their canonical headers. A header-only
file is a template, not an approved empty corpus and not evidence that the task
has annotations.

## Source-of-truth contracts inspected

The templates follow the constants and loaders in the existing code:

| Task | Current loader | Training label field | Semantics |
| --- | --- | --- | --- |
| Sentiment | `SentimentCorpus.from_csv` | `sentiment` | One nonblank annotator-defined label |
| Emotion | `EmotionCorpus.from_csv` | `emotions` | Pipe-separated multilabel cell; blank means no listed emotion |
| Issue | `AnnotationCorpus.from_csv` | `labels` | Pipe-separated multilabel cell; blank means no issue labels |
| Response | `ResponseCorpus.from_csv` | `response_type` | Exactly one canonical V1 role |
| Solution | `SolutionCorpus.from_csv` | `is_solution` | Required boolean; `solution_type` is free-form metadata |

The existing `TaskType` members remain exactly `SENTIMENT`, `EMOTION`, `ISSUE`,
`RESPONSE`, and `SOLUTION`. No template adds a task or changes a model label
contract.

## How a reviewed dataset is created

1. Copy the appropriate template to the exact task path expected by
   `ml/configs/config.py`, only after the source sample and review plan have
   been approved. Do not overwrite an existing file.
2. Populate source identity and task labels only from an authorized review
   sample. Preserve the original source text separately from any normalized
   text used by a model.
3. Build a fresh review sidecar with
   `app.services.annotation.manifest.build_manifest_records()` from the
   authorized candidate CSV or in-memory rows, then write it with
   `write_manifest()`. The current training CSV loaders do not consume the
   sidecar fields.
4. Have annotators work independently before seeing another annotator's label.
   Use blank/adjudication status in the review sidecar for uncertain or
   insufficient-context rows. Do not force an ambiguous row into a class.
5. Run the task-specific schema checks and the record-level checklist below.
6. Resolve duplicates, disagreements, exclusions, and taxonomy changes before
   producing a final training CSV. The final training CSV must contain only
   rows eligible for the approved training version.
7. Validate the candidate-to-manifest join, freeze the approved dataset version
   and its manifest, and only then perform a deterministic, thread-aware split.
   Keep validation separate from the untouched test set.

The repository now contains an offline manifest validator, but it does not make
the human decision itself. `validate_manifest_for_training()` requires completed
review, a final task label, `training_inclusion=include`, and explicit dataset
approval. Human policy still determines who may approve the batch and which
provenance fields are mandatory for release.

## Response-candidate pilot procedure

The response candidate exporter is an offline hand-off only. For a pilot, use
an already-authorized, bounded static post/comment export. Do not invoke the
database-backed `load_comment_candidates()` helper for a pilot: it currently
selects all comments and all posts into memory and has no bounded batch
parameter. No database or provider request is part of this procedure.

The static input must provide, at minimum, nonblank string `document_id`,
`comment_id`, and original `text`. A released review batch should also provide
`post_id`, `parent_comment_id`/`parent_text` when available,
`parent_context_status`, `post_title`, `subreddit`, `created_at`,
`source_url`, `source_provider`, and explicit `source_mode` (`direct` or
`context`) where known. Missing values remain blank and are surfaced as review
warnings; the exporter does not invent IDs, timestamps, URLs, scores, parents,
or provenance. A genuine score of `0` is preserved; a missing score stays
blank. Invalid scores and supplied boolean metadata are rejected.

Candidate review columns are:

```text
document_id,text,normalized_text,response_type,post_id,comment_id,
parent_comment_id,parent_text,post_title,subreddit,score,created_at,
source_url,source_provider,source_mode,parent_context_status,language,script,
is_code_mixed,detected_languages
```

`response_type` is always blank in a candidate export. The optional source and
parent-context columns are review-only extensions; the final training CSV
header remains unchanged. The stable `document_id` is the unambiguous join key
to the review manifest. `build_manifest_records()` accepts this candidate CSV,
and `validate_candidate_manifest_join()` checks that the sidecar has exactly
one row per candidate ID, agrees on shared source facts, and preserves
timestamp strings exactly. The manifest writer creates a separate JSONL
sidecar with an exclusive destination claim; it does not modify the candidate
CSV.

The writer validates rows before creating the destination and claims a new path
with an exclusive create. Existing destinations are never overwritten. A pilot
must choose a fresh output path and preserve the static input unchanged.

## Review-manifest implementation

The sidecar is versioned JSONL (`manifest_version="1.0"`) with separate
`source`, `review`, `adjudication`, and `dataset_approval` sections. A fresh
record starts as `review.status="unreviewed"`,
`adjudication.status=null`, and `training_inclusion="pending"`; annotator,
adjudicator, label, approval, and timestamp fields remain unset. Raw candidate
text is not copied into the manifest.

The implementation accepts a local candidate CSV or in-memory mappings and
supports legacy candidate CSVs that omit newer optional context columns. Use
these checks before releasing a batch:

```python
from app.services.annotation.manifest import (
    build_manifest_records,
    validate_candidate_manifest_join,
    validate_manifest_for_training,
    write_manifest,
)

records = build_manifest_records(candidate_csv_path)
validate_candidate_manifest_join(candidate_csv_path, records).raise_for_errors()
write_manifest(manifest_path, records)
```

The default builder treats conditional provenance as a warning because the
required fields depend on the authorized source and review policy. A release
owner can make a field mandatory without inventing a value by passing, for
example, `required_source_fields=("source_record_hash", "context_snapshot_ref")`
to the builder, join validator, or writer. Missing fields then become
`missing_required_provenance` errors; optional omissions remain
`missing_optional_provenance` warnings. Unknown field names are rejected as
contract errors.

`validate_manifest_for_training()` is a separate, stricter gate. It rejects
blank final labels, incomplete review, required adjudication, exclusions, and
missing/rejected dataset approval, and requires an approved dataset version. It
does not replace human authorization or the task-specific CSV loaders.

Before releasing a candidate batch to annotators, check:

- [ ] The static source export is authorized, bounded, immutable, and separately
      retained.
- [ ] `document_id` and `comment_id` are unique and nonblank; shared parent IDs
      are allowed.
- [ ] Every row has the available post/thread context and explicit parent status;
      unavailable context is visible rather than repaired.
- [ ] Source timestamp, source URL, provider, and provenance mode are present or
      explicitly documented as unavailable.
- [ ] Original and normalized text remain separate.
- [ ] Score and boolean metadata pass strict validation; missing optional values
      remain missing.
- [ ] `response_type` is blank for every candidate.
- [ ] The candidate-to-manifest join is one-to-one by `document_id`.
- [ ] The manifest validator reports no source-field, timestamp, orphan, or
      missing-row errors.
- [ ] The output path did not exist before the exclusive create.

The converter is not an approval mechanism. It reports blank labels and missing
normalized text as exclusions, rejects invalid labels and duplicate identities,
and must only be run after human review, adjudication, manifest validation, and
explicit dataset approval.

## Annotation unit and context

The unit is one source document identified by a stable `document_id`. For
response and solution tasks, the document is normally a comment and must retain
its `comment_id`, enclosing `post_id`, and parent context when available. For
sentiment, emotion, and issue tasks, the reviewed unit may be a post or comment,
but the same stable identity and thread relationship must be retained.

Annotators should receive the smallest context needed to make a defensible
decision without merging threads: the original text, post/thread identity,
subreddit/community, source URL when available, source timestamp, and the
parent comment or post context for replies. Do not replace original text with
normalized text. The current training CSVs preserve only a subset of this
context; the review sidecar is the explicit place for the rest until a
reviewed loader/schema extension is approved.

## General decision rules

- Read the full available context before deciding.
- Never use a model prediction as a ground-truth label.
- Never invent a seventh response role, an issue taxonomy, an emotion taxonomy,
  a sentiment class, or a solution category.
- Keep a valid empty multilabel set distinct from an item that cannot be judged.
- If evidence is missing, truncated, unintelligible, or genuinely ambiguous,
  mark the row for adjudication/exclusion in the review sidecar instead of
  forcing a label. The final training CSV has no universal abstention column;
  unresolved rows therefore must not enter the final approved training file.
- Preserve exact canonical spellings and pipe-separated multilabel formatting.
- Do not claim agreement statistics until independent labels exist and have been
  measured.

## Task-specific annotation guidance

### 1. Sentiment classification

The implemented schema is single-label and does not impose a fixed vocabulary.
Every final training row needs one nonblank `sentiment` value. The following is
only a **draft taxonomy for human approval**, not an implementation change:

- `positive`: favorable overall evaluation or outcome;
- `negative`: unfavorable overall evaluation or outcome;
- `neutral`: factual, mixed-without-dominant-polarity, or non-evaluative text.

Annotators must use the approved taxonomy version, not this proposal, if the
project chooses another vocabulary. Do not infer sentiment from emotion, topic,
or a single strongly worded phrase when the overall text is mixed. If the text
cannot be interpreted, leave the task label unresolved in the review manifest
and exclude it from the final training CSV; do not invent an abstention class.

### 2. Emotion classification

The implemented schema is multilabel. `emotions` is a pipe-separated list such
as `stress|frustration`; a blank cell is a valid **no listed emotion** outcome.
Emotion is independent of sentiment: negative text may have one, several, or no
emotion labels.

The emotion vocabulary is annotator-defined and must be approved before labeling.
Annotators should label only emotions supported by the text and context. A
factual or procedural statement can validly have an empty emotion set. If the
meaning is unavailable or ambiguous, use the review sidecar's adjudication or
exclusion status; do not use an empty cell to hide an unannotatable item.

### 3. Issue identification

The implemented schema is multilabel. `labels` is a pipe-separated list of
annotator-defined issue names; a blank cell is a valid **no issue labels**
outcome. The issue vocabulary must be approved from the annotation guideline;
the code deliberately does not hard-code one.

Label an actual problem, request, or question grounded in the text. Do not label
every general discussion statement, preference, praise, or response role as an
issue. If multiple independent problems are expressed, retain each approved
issue label. If the text does not provide enough context to decide whether a
problem is present, record adjudication/exclusion status separately rather than
silently treating it as no issue.

### 4. Response classification

Response classification is single-label and must use this exact canonical order
and spelling:

1. `agreement`
2. `personal_experience`
3. `opinion`
4. `advice`
5. `solution`
6. `counterargument`

Use the primary conversational role, not sentiment. `agreement` endorses a
prior claim without substantive lived detail. `personal_experience` reports the
writer's own lived event. `opinion` states a judgment not clearly positioned
against a specific prior claim. `advice` recommends a next step while leaving
important choices or details unresolved. `solution` presents a concrete,
issue-matched fix. `counterargument` rejects, corrects, qualifies, or contests
an identifiable prior claim.

For mixed roles, choose the dominant contribution only when context supports a
consistent choice. For missing parent context, unintelligible code-mixed text,
emoji-only acknowledgements, or unresolved boundary cases, leave the candidate
blank for adjudication. The final `ResponseAnnotation` schema requires one of
the six labels; blank rows must be excluded before training.

### 5. Solution detection

The implemented target is the boolean `is_solution`, written as an accepted
boolean spelling such as `true` or `false`. Blank or unrecognized values are
rejected; blank is not a negative example. `solution_type` is optional,
free-form annotation metadata and is never learned as the V1 target.

Mark `true` only when the text provides an actionable solution or recommendation
for the relevant problem. Helpful tone, agreement, sympathy, a personal story,
or a vague suggestion is not automatically an actionable solution. `advice` and
`solution` response roles are separate annotations; response role must not be
used to derive `is_solution`.

If the comment cannot be interpreted or the issue context is missing, track it
as unresolved/excluded in the review sidecar. Do not turn uncertainty into
`false`.

## Illustrative examples

The examples in this package are decision illustrations only. They are not
project records, are not inserted into any annotation CSV, and must not be used
as evidence of data volume or model accuracy.

| Task | Illustrative text | Draft decision |
| --- | --- | --- |
| Sentiment | “The placement process was excellent.” | `positive` only if that draft taxonomy is approved |
| Emotion | “The exam is on Monday.” | blank `emotions` (no supported emotion) |
| Issue | “The hostel water supply fails every evening.” | an approved issue label such as `hostel` only after taxonomy approval |
| Response | “I faced this last year and the office fixed it.” | `personal_experience` |
| Response | “Email the registrar with your fee receipt.” | `solution` if the context makes it a concrete fix; otherwise adjudicate |
| Solution | “Same, I had the same problem.” | `false` unless the text contains an actionable fix |

## Record-level validation checklist

Complete this checklist for every row before it can enter a candidate review
batch, and repeat the applicable parts before a final training export:

- [ ] `document_id` is present, stable, nonblank, and unique within the dataset
      version.
- [ ] `post_id` and `comment_id` identify the source unit when applicable; a
      reply retains its `parent_comment_id` in the review manifest.
- [ ] The row has authorized source provenance and a traceable source URL or an
      explicit documented reason why the source has no URL.
- [ ] Source timestamp, community/subreddit, and thread identity are recorded
      in the review manifest when available.
- [ ] Original text is preserved; normalized text, if used, is a separate field.
- [ ] Required text is nonblank and not a truncated placeholder.
- [ ] `language`, `script`, `is_code_mixed`, and `detected_languages` are
      metadata, not replacements for a human task label.
- [ ] The task label uses the exact current schema and allowed structure.
- [ ] Multilabel cells use `|` with no empty or duplicated entries, except that
      blank `emotions`/`labels` has the documented valid-empty meaning.
- [ ] A blank/uncertain row is marked adjudication or exclusion in the review
      manifest and is not mistaken for a valid negative or empty multilabel row.
- [ ] Independent annotator status is recorded before another annotator's label
      is revealed.
- [ ] Any disagreement, duplicate identity, source mismatch, or missing context
      is linked to an adjudication decision.

## Human review and adjudication procedure

### A. Prepare the review batch

1. Select only authorized source records and freeze a batch identifier.
2. Create one review-manifest row per training `document_id`; retain the source
   post/thread, parent, URL, timestamp, subreddit, and provenance references.
3. Give each annotator the same source context and task-guideline version, but
   do not expose other annotators' decisions.
4. Use the blank templates only as task outputs. Keep review status, reviewer
   identity, notes, and provenance in the separate manifest until a schema
   extension is approved.

### B. Independent annotation

1. Each annotator records one independent decision per task where the item is
   eligible.
2. An annotator may mark `adjudication_required` or
   `excluded_insufficient_context` in the manifest instead of forcing a label.
3. Record reviewer identity using the project's approved privacy-preserving
   identifier, not a raw personal identifier in the training CSV.
4. Record review date/time in the manifest using an explicit timezone.
5. Do not edit an earlier independent decision in place; preserve the original
   value and record later adjudication as a separate event.

### C. Disagreement detection and adjudication

1. Compare independent labels by `document_id` after both annotators finish.
2. Treat different single labels, different boolean targets, and differing
   multilabel sets as disagreements. A valid empty emotion/issue set is a real
   decision and must not be treated as missing.
3. A senior reviewer or agreed adjudication panel reads the same source context,
   records the decision and rationale, and identifies the guideline rule used.
4. If the panel cannot decide, keep the row out of the final training CSV and
   mark it unresolved rather than inventing a label.
5. Summarize disagreement counts and agreement statistics only after the
   independent records exist. Do not backfill agreement from an adjudicated
   label alone.

### D. Version and freeze

1. Version the task guideline and taxonomy whenever a boundary or label meaning
   changes.
2. Record the manifest hash, training-CSV hash, guideline version, source batch,
   reviewer/adjudication status, and exclusions.
3. Freeze the approved dataset version before splitting or fitting.
4. Never silently merge a later taxonomy into an earlier training dataset; start
   a new version and document the migration decision.

## Dataset approval checklist

Training authorization requires a human owner to complete every applicable item:

- [ ] The source was authorized for annotation and the collection/use purpose is
      documented.
- [ ] Every included row has a traceable stable identity and source manifest.
- [ ] The final CSV uses exactly the task loader's required columns and types.
- [ ] The current task taxonomy/version has an explicit human approval.
- [ ] All included rows have final review/adjudication status.
- [ ] Unresolved, insufficient-context, duplicate, and source-mismatch rows are
      excluded or explicitly approved under a documented exception.
- [ ] Class counts, multilabel counts, empty-set counts, exclusions, and missing
      values have been reviewed.
- [ ] Duplicate IDs and exact duplicate texts have been checked.
- [ ] Near-duplicate and related-record policy is documented and applied.
- [ ] Thread-aware grouping is used where comments or related source records
      could leak across splits.
- [ ] Deterministic train/validation/test proportions and seed are recorded.
- [ ] The split is group-aware where applicable and frozen before fitting.
- [ ] Validation remains separate from the untouched held-out test set.
- [ ] Any minimum sample/class policy is explicitly approved for this task; no
      threshold was relaxed merely to make training run.
- [ ] The approved dataset version, manifest, guideline version, and hashes are
      archived together.
- [ ] Training is authorized for the named task and intended model baseline.
- [ ] A new, run-specific artifact directory and separate registry destination
      have been reserved; existing artifacts will not be overwritten.

Completing this checklist does not itself establish model quality. It authorizes
the next controlled training/evaluation step only after the human owner signs
off.

## Current implementation gaps and compatibility notes

These are documented discrepancies, not silently changed by this package:

1. The current task training CSVs do not include source URL, timestamp,
   subreddit, parent ID, source provider/mode, reviewer identity, review status,
   adjudication rationale, or approval version. See
   `REVIEW_MANIFEST_PROPOSAL.md`; do not append those fields to a production
   training CSV without a reviewed loader/schema change.
2. The current final schemas have no universal abstention or
   `adjudication_required` field. Blank response/sentiment/solution values are
   invalid for final training; blank emotion/issue cells have valid empty-set
   meanings. The sidecar status is therefore required to distinguish
   “valid empty” from “not annotatable.”
3. `ml/data/readiness.py` currently implements a readiness schema only for the
   response task. Other tasks still require their task-specific loaders plus a
   human review of class distributions and provenance.
4. Existing loaders expose duplicate-ID inspection or validation helpers, but
   not every loader automatically refuses every duplicate or missing-text case
   at CSV load time. The approval checklist must run those checks before
   training.
5. `preflight.py` is currently response-oriented and does not certify human
   approval. A successful preflight is not an approval decision.

## Latest offline verification recorded for this draft

The focused annotation and manifest command was run on **2026-10-21** using the project
virtual environment:

```powershell
Set-Location D:\AI_Project
$python = "D:\AI_Project\backend\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"
$env:PYTHONDONTWRITEBYTECODE = "1"

& $python -m unittest backend.test_annotation_templates backend.test_annotation_export backend.test_convert_annotation_candidates backend.test_annotation_manifest -q
```

Result:

```text
Ran 55 tests
OK
Focused annotation and manifest tests exit=0
```

These tests use header-only templates and small temporary fixtures only. They do
not read or modify `data/annotations/language_annotations.csv`, do not write a
real task dataset or manifest, and do not train or load a model. Additional static
checks for this draft passed: `py_compile` for the annotation modules/tests (exit 0),
`pip check` (`No broken requirements found`, exit 0), `git diff --check` (exit 0),
and relevant modified/untracked-file whitespace checks with zero whitespace
diagnostics. Git line-ending notices were not treated as whitespace failures.

**Stop condition:** this package is ready for human review, not for approval or
training. Do not populate the templates or begin collection as part of this
milestone.
