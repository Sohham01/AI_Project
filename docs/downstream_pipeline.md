# Offline downstream pipeline

## Verified architecture

This is an in-memory library integration, not a new HTTP endpoint, worker,
database job, or collection path. `backend/app/main.py` currently exposes only
root/liveness/database-health routes. `frontend/` and `analytics/` are empty in
this checkout; the presence of backend NLP packages does not imply a working UI
or end-to-end reporting application.

| Stage | Existing entry point and contract | Requirements / current handoff |
| --- | --- | --- |
| Normalized source records | `PostData` / `CommentData` in `app/schemas/ingestion.py` | Carry source IDs, content, timestamps and parent links, but not a complete downstream document/provenance envelope. Comments need their enclosing post identity and independently established source mode. |
| Language | `LanguageAnalysisService.analyze(str)` -> `FullLanguageAnalysis` | Uses existing langdetect, script and code-mix detectors; local, no project-trained model or DB. Short/code-mixed language results remain baseline estimates, not validated accuracy. |
| Preprocessing | `MultilingualTextPreprocessor.preprocess(...)` -> `PreprocessingResult` | Deterministic local normalization with separate original/normalized text. No translation. |
| Topics | `NMFTopicDiscoverer.discover(Sequence[TopicDocument])` -> `TopicDiscoveryResult` | Fits TF-IDF and NMF on each call. Not an inference-only API over an existing fitted artifact; intentionally not called by this orchestrator. |
| Issues | `predict(Sequence[IssueDocument])` -> `IssueIdentificationResult` | Requires a trained classifier; multilabel taxonomy learned from annotations. Optional topic features need topic context. |
| Sentiment / emotion | Separate `predict(Sequence[SentimentEmotionDocument])` -> `SentimentAnalysisResult` / `EmotionAnalysisResult` | Separate trained models and annotation-defined label spaces. Emotion is not derived from polarity. |
| Responses | `predict(Sequence[ResponseDocument])` -> `ResponseClassificationResult` | Trained comment-role model. Canonical order: agreement, personal_experience, opinion, advice, solution, counterargument. Missing trained classes remain explicit, not zero-filled. |
| Solutions | `predict(Sequence[SolutionDocument])` -> `SolutionDetectionResult` | Separate trained binary detector for comments. Advice/solution response roles supply context, not a positive solution verdict. |
| Linking | `link(Sequence[IssueEvidence], Sequence[SolutionEvidence])` -> `LinkingResult` | Thread-scoped adapter supports explicitly injected **no-fit** implementations of the existing interface. The built-in `TfidfCosineIssueSolutionLinker` always fits a vocabulary, so its invocation, including subclasses, is blocked with `linker_requires_fitting`. No default linker is created. |
| Resolution | `detect(Sequence[IssueResolutionEvidence])` -> `ResolutionBatchResult` | Injected resolver receives validated per-issue/label evidence only after explicit discussion completeness and complete issue/solution prediction processing. Requires configured minimum discussion volume and a supported strong pair-specific link; a link is not proof of resolution. No default resolver is created. |
| Reporting | Existing per-stage result helpers and `ml/evaluation` reports | No integrated application analytics/reporting entry point. Pipeline output provides record associations, model batch metadata and explicit stage coverage for a later reporting adapter. |

The existing annotation candidate builder can run language/preprocessing on
mappings without a database. Its database loader/export CLI is a separate path.
The annotation converter/readiness workflow is unchanged; predictions are never
copied into human labels by this pipeline.

`ml/training/trainer.py` is a **training** orchestrator, not a downstream
inference runner. It can save joblib/native artifacts and register metadata.
Registry entries are not evidence that an artifact is present, genuine, or
approved for inference. Searches for common binary artifact extensions in
`ml/artifacts/` and `backend/ml/` found none; this is not exhaustive artifact
validation or evidence of approval. No binary was deserialized. MuRIL additionally
needs torch/transformers and encoder/tokenizer weights; its training path can
download weights. This integration does not import, load, download or train it.

The five existing `TaskType` members remain SENTIMENT, EMOTION, ISSUE, RESPONSE
and SOLUTION. Readiness checks currently support response annotations only;
data volume alone is not evidence of annotation genuineness or model quality.

## Contained improvement

`app.services.nlp.pipeline.DownstreamPipeline.run(...)` accepts a sequence of
flat normalized record mappings. It reuses the existing language/preprocessor
and Document/Result schemas, then invokes **injected, already-trained** models:

1. Validate and deduplicate records, keeping the first valid document ID.
2. Analyze language, script and code mix.
3. Normalize text; block inference for records whose language/preprocessing failed
   or whose normalized text is empty.
4. Report topic discovery unavailable (it would fit a model).
5. Run issues, then independent sentiment and emotion.
6. Run response and solution models on comments only.
7. Adapt existing predictions to thread-scoped linking and resolution evidence;
   call injected interfaces when eligible and report unavailable/excluded work.

Issue predictions are passed to the existing context fields of the subsequent
per-document models. Response predictions are passed to solution detection.
If an optional upstream prediction is missing, dependent context stays empty;
its stage report still records the failure/unavailability. Models requiring
unavailable topic features are not called.

### Input contract

Required: nonblank string `document_id`, `text`, `source_subreddit`, an explicit
`source_mode` (`direct` or `context`), timezone-aware `created_at`, and nonblank
string `post_id`. Timestamps may be aware datetime objects or ISO timestamps with
an explicit offset. Naive timestamps and numeric epochs are rejected rather
than silently assigned a timezone.

`comment_id` identifies a comment; omit it for a post. Optional
`parent_comment_id`, `parent_reddit_id`, `parent_comment_text`, `url`, and
`original_text` are preserved. Parent fields require a comment identity.
Document IDs are independent corpus keys, not reconstructed Reddit IDs.
Only the two approved institutional communities can carry direct provenance,
case-insensitively. Context remains context even if its subreddit name happens
to match a direct community. No source mode is inferred from a name.

The envelope uses `source_subreddit` to distinguish explicit provenance from a
request hint. A caller adapting existing `PostData`/`CommentData` must explicitly
map their `subreddit`, `reddit_id`, and `body` fields and supply known provenance.
The orchestrator does not fetch missing metadata or guess IDs/relationships.
It does not accept an ORM object or collector report as a substitute for this
contract. Original record text is kept unchanged; if `original_text` is absent,
the supplied `text` is also retained as the source copy.

### Results and failure behavior

`PipelineResult` exposes `records`, `stages`, `batch_outputs`, `invalid_records`,
`duplicate_document_ids`, `warnings`, `thread_evidence` and an overall `status`.

- Accepted records retain source provenance, IDs, timestamps and parent links
  separately from predictions. Per-record outputs remain associated by document
  ID, not by model result position.
- `batch_outputs` stores each successful full model result once, including
  name/version/configuration/warnings. It is not copied into every record.
- Invalid rows report input index and field names, never payloads or exception
  messages. Stage errors expose bounded categories and exception types only.
- Missing classifiers and `is_trained != True` are `unavailable`; readiness exceptions
  and prediction/result-contract failures are `failed`. Healthy independent
  stages continue. No fallback training or fake prediction is performed.
- Result schema, exact prediction-ID coverage, counts, and echoed identity/text
  are checked before publishing any batch. Missing, duplicated, foreign, or
  source-contradicting predictions invalidate that stage atomically.
- Classifiers receive deep-copied documents, so mutations cannot change the
  authoritative source-derived validation inputs. Optional raw provenance and
  context echoes are checked before neutral schemas discard extra fields;
  cyclic/oversized prediction payloads are rejected before serialization.
- Response probabilities retain the supplied values and learned-class coverage;
  their keys use canonical order. Untrained classes are never assigned made-up
  probabilities.
- An unavailable stage is **not** a negative prediction. A zero solution count
  from a missing detector cannot be interpreted as "no solutions found."
- Stage states include `completed`, `partial_failure`, `failed`, `unavailable`
  and `skipped`. Unavailable stages make the overall outcome
  `partial_failure`, even if all supported stages work. Empty/all-invalid input
  returns `invalid_input` and skipped stages, never a successful analysis.

`topic_model` remains accepted and uncalled with
`topic_discovery_requires_fitting`. `linking_model` and `resolution_model` now
use their actual `link(...)` / `detect(...)` interfaces, never `predict(...)` or
an `is_trained` check. Non-interface objects are explicitly unavailable.

## Thread-scoped evidence handoff

`app.services.nlp.thread_evidence.ThreadEvidenceAdapter.process(...)` accepts
an **already-produced `PipelineResult`**. It does not analyze language,
preprocess, load an artifact, or run a classifier. The same adapter is available
through `DownstreamPipeline.process_thread_evidence(...)`; `run(...)` uses it
after the five per-document stages and returns it as `result.thread_evidence`.

### Scope, authority, and validation

- Thread identity is exactly `(source_mode, source_subreddit.casefold(), post_id)`.
  Matching post IDs in different provenance modes or communities never merge.
  Thread keys, record IDs, issue labels, links, and resolution outputs have stable
  ordering independent of input order.
- Each authoritative record is revalidated with the existing normalized input
  contract, including aware timestamps. No timestamp, provenance, ID, URL, or
  parent is filled in from a prediction. `ThreadEvidenceReport.records` preserves
  the source spelling, source/normalized text, URLs, times, and parent identity;
  it excludes raw per-record prediction/error payloads.
- Duplicate corpus IDs or source identities, known cross-thread parent references,
  conflicting `t1_`/`t3_` parents or supplied parent text, parent cycles, parent-after-child times,
  and comments before a supplied post invalidate the affected thread. A missing
  parent outside the supplied records is not guessed or fetched.
- Actual `IssuePrediction`, `ResponsePrediction`, and `SolutionPrediction` schemas
  validate record predictions. Required identity/text/language echoes must match
  the source-derived model input. Contradictory optional provenance/time/URL
  echoes, topic context, malformed probabilities, duplicate/unsupported issue
  labels, noncanonical response roles, and incoherent solution decisions are
  rejected. Decision fields must be explicitly present; neutral schema defaults
   cannot manufacture processed negative assessments. This also applies at the
   `run()` classifier result boundary. Response and solution predictions are comments only. Advice never
   becomes a positive solution by its response role.
- Prediction payload copying rejects cycles, unsupported values, depth over 32,
  or more than 10,000 visited items rather than truncating or repairing them.
  A malformed prediction remains a local rejection, not lost source data or an
  exception that prevents healthy threads from being processed.
- Batches are optional when existing per-record predictions are supplied. If a
  batch is present its schema, document count, ID coverage, model identity, and
  per-record predictions must agree. Foreign/duplicate IDs or malformed batch
  metadata invalidate that stage's handoff. A local batch/record echo mismatch
  excludes its own record/thread, preserving independent healthy threads.
- The linker receives only issue-bearing records and positive, nonabstained
  solution comments. Returned links must reference those exact inputs and labels,
  preserve their metadata/probabilities, carry consistent finite scores and model
  identity, and have coherent counters. Repeated/foreign triples and contradictory
  metadata reject the thread's linking result atomically. Source evidence and
  arguments are deep-copied so a model cannot mutate the authoritative envelope.
- Valid scored candidates remain in `linking_result` for audit. Only positive,
  non-self links with positive semantic similarity and a solution timestamp **not
  before** its issue enter `supported_links`. Zero-vector flags also prevent
  semantic support. Below-threshold/unsupported/chronologically invalid links
  have explicit rejection counts and never reach the resolver.

### Resolution guard and denominators

`discussion_complete` defaults to `False` (unknown coverage). A caller must
explicitly supply literal `True`, or a mapping from exact normalized thread keys
to literal `True`, to assert that the supplied discussion is complete. This is a
caller assertion, not something inferred from a collection cap or from the
absence of errors.

Even with that assertion, resolution requires a valid issue prediction for
**every supplied record**, including records with no issue labels, and a valid
solution prediction for **every supplied comment**, including negatives. Upstream
invalid records, dropped duplicate IDs, or invalid source envelopes prevent a
complete-coverage claim because their thread coverage cannot be established.
Missing predictions, unavailable models, no positive solutions, no supported
links, and linking failures produce exclusions, **not `unresolved` verdicts**.
Each eligible `(issue document ID, issue label)` needs its own supported strong
link and the resolver's configured minimum discussion volume. The default actual
resolver requires three comments and a strong-link threshold of 0.6. Invalid
resolver configuration, insufficient volume and missing strong support produce
explicit exclusions (`invalid_resolver_config`, `insufficient_discussion_volume`,
`no_supported_strong_links`) before `detect(...)`, not negative verdicts.

For each pair, the denominator is the number of unique supplied comments in its
thread (including an issue-bearing comment, excluding the post).
`discussion_volume` equals that denominator. `solution_candidate_count` is the
number of validated comment-level solution assessments, including negative
assessments, as defined by the existing resolution evidence schema. Linked and
strong counts are unique solution IDs for that pair; strength also requires the
resolver's configured `strong_link_threshold`. Scores and detector probabilities
come only from those pair-specific supported links. The canonical six-role
histogram counts validated comment responses for the whole thread; abstentions
and absent classifications do not invent roles. The existing scalar
`experience_count` maps to canonical `personal_experience`.

Each evidence record passes the resolver's real `derive_evidence` validation
before `detect(...)`. Returned resolution IDs must cover exactly the supplied
pairs, preserve the evidence and post identity, and echo its derived counts,
densities, temporal metadata, score components and `evidence_sufficient` consistent
with the validated volume/configuration. Failures publish no verdicts
for that thread and do not stop other threads. Resolver statuses remain the
existing three canonical labels; they are heuristic assessments, not measured
outcomes or proof that a proposed fix worked.

### Coverage and failure reports

`ThreadEvidenceResult` contains ordered per-thread reports plus aggregate
`EvidenceCoverage`. Counts are computed from actual source records, validated
predictions, returned candidates, accepted links, and calls; the adapter does not
copy a model's summary counters into coverage.

- `records_received/validated/rejected`, `threads_received/validated`, and
  `threads_complete/partial` describe source validation and final handoff coverage.
  Unplaceable source rows use sanitized `invalid_record_indexes`; their thread
  IDs are not invented. Upstream losses are separately counted in
  `upstream_invalid_records` and `upstream_duplicate_ids`.
- `issue_predictions_received/validated` counts all issue assessments;
  `issues_received/validated` counts unique issue-bearing documents;
  `issue_labels_received/validated` counts document-label pairs.
  `issues_linking_eligible` and `threads_linking_eligible` require valid issue and
  positive solution evidence, independently of model availability.
- `solution_predictions_received/validated` includes all comment assessments;
  `solutions_received/validated` counts asserted positives. Response prediction
  counts and the fixed-order histogram expose the actual role coverage.
- `link_attempts/failures` counts thread calls; `links_returned/validated/supported`
  and `link_rejections` distinguish output-contract failures from unsupported
  scored candidates. `rejection_reasons` names observed adapter rejections; it
  does not claim to count unobservable candidates a linker never returned.
- `resolution_pairs_received`, `resolution_eligible`, `resolution_excluded`, per-pair `excluded_reasons`, and
  aggregate `resolution_excluded_reasons` describe evidence eligibility.
  Reasons can overlap; the excluded-pair count is unique. Eligible evidence may
  still have an unavailable resolver. `resolution_attempts/failures` counts calls;
  `resolution_pairs_attempted/failed` and `resolution_outputs` count label-level
  records. Per-pair reports expose denominators and unique linked/strong IDs.

`discussion_complete`, `prediction_processing_complete`, and `coverage_complete`
are separate flags. Final complete coverage also requires all received issue
pairs to produce outputs and no validation/unavailable/failed handoff operation.
Unknown coverage is partial. Empty input is deterministic and `skipped`, with
zero counts and no model calls. Topic unavailability still makes a full
`DownstreamPipeline.run(...)` overall status partial even when its thread handoff
is complete.

### Calling the library

With the backend package on `PYTHONPATH`, a caller can process an already-loaded
list without any persistence:

```python
from app.services.nlp.pipeline import DownstreamPipeline

pipeline = DownstreamPipeline()  # existing local language/preprocessing only
result = pipeline.run(normalized_records)
# Inspect result.stages before interpreting result.records as analysis evidence.
# Pass approved already-trained model instances with issue_model=..., etc.

# Reuse already-produced record predictions without rerunning classifiers:
handoff = DownstreamPipeline(
    linking_model=approved_no_fit_linker,  # implements IssueSolutionLinking
    resolution_model=approved_resolver,   # implements ResolutionDetection
).process_thread_evidence(
    result,
    discussion_complete={("direct", "iimsambalpur", "known-post-id"): True},
)
# run(..., discussion_complete=...) uses the same handoff and injected models.
```

There is deliberately no implicit artifact discovery/deserialization. Loading
untrusted joblib/pickle files is unsafe. Artifact provenance, training readiness,
version compatibility and permission to process retained content require review
before an inference model is injected. `is_trained=True` is a technical readiness
signal, not a quality certification.

## Verification of this change

Using PowerShell from the repository root:

```powershell
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"
$env:PYTHONDONTWRITEBYTECODE = "1"
& "D:\AI_Project\backend\.venv\Scripts\python.exe" -m unittest backend.test_nlp_pipeline backend.test_thread_evidence -q
```

The earlier integration report recorded **70 tests** in the focused handoff
check: 37 pipeline tests and 33 thread-evidence contract tests. The final audit
expanded this selection to **78 distinct tests: 40 pipeline and 38 evidence**;
each module passed independently, including in the final consolidated rerun.
They exercise typed no-fit interface mocks, cross-thread/metadata rejection,
completeness and full prediction-coverage gating, valid per-label handoff,
per-thread failures (including cyclic/oversized prediction payloads), derived
counts, deterministic empty/order behavior, and
`run()` integration. The existing smoke check uses local language detectors and
preprocessing for English, Hindi, Bengali, Odia and Romanized text.

This focused command covers only these two new/updated test modules. Separate
provider and safe downstream regression checks are recorded with the integration
verification below. No classifier training/fitting, corpus-fitting linker/topic
tests, neural tests, model loading, or model artifact writes are part of these
checks. These are contract/smoke checks, not model-performance claims.

The earlier integration selection had **318 ingestion/pipeline** plus **376
safe downstream** IDs, with no overlap (**694 unique tests**). Independent
enumeration confirmed those pre-audit selection sizes; they are historical,
not current final counts. The final audit selection is **331 + 376 = 707 unique
tests**, with zero overlap after canonicalizing import aliases. On **2026-10-16**,
all 19 final selections passed in two wrappers that launched each selection in a
separate subprocess (707 executions, zero failed selections, every process exit
code 0). Repeated development/red-green runs are not extra unique tests, and
inventory counts are not treated as proof of execution success.

The audit reproduced and fixed insufficient resolution gating, mutable classifier
input authority, raw contradictory provenance lost during schema parsing, and
ambiguous Decodo target paths before paid dispatch. It also isolated a manager
test from ambient credentials: its initial run had one error among 13 tests;
after fixture correction and an added configured-target regression, all 14 passed.
The production manager and standalone resolver were unchanged.

Fresh `py_compile` passed for 16 scoped Python files using temporary bytecode;
`pip check` and tracked/relevant-untracked whitespace checks passed. Exact
selectors, counts and release gates are recorded in `backend/docs/ingestion.md`,
under "Final independent audit". Live requests to all providers were forbidden;
provider availability, credentials, access, quotas and costs remain unverified.

No collection, PostgreSQL connection, migration, annotation export, model
artifact write or candidate overwrite is part of this integration. No files
were staged, committed, reset or restored. Unrelated changes remain user-owned.

## Remaining milestone

Supply a reviewed no-fit linker and trusted inference artifacts for the five
supervised tasks, and establish discussion completeness at the calling boundary.
The built-in TF-IDF linker remains explicitly unavailable in this no-fit flow.
Separately obtain/review genuine human annotations and evaluate held-out outcomes;
training and quality evaluation need their own authorization. The adapter does
not establish classifier/linker accuracy, recover omitted thread records, or
turn a capped collection into a complete discussion.
