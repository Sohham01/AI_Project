# Ingestion System Documentation

## Overview

The ingestion layer normalizes provider results into the shared `PostData` and `CommentData` contracts. Live-source readiness is separate from offline adapter tests; mocked tests do not establish that a vendor account, quota, endpoint, or current response contract works.

## Providers

- **Arctic Shift**: Historical post/comment snapshots, bounded to one response per operation. External availability is not established by offline tests.
- **RSS**: Single-feed Reddit post discovery from RSS2 or namespaced Atom; not a comment source.
- **Decodo**: An offline-contract-tested adapter for Decodo Universal. It sends an explicit source URL, proxy pool, and rendering configuration; it does not send deprecated Reddit-specific targets such as `reddit_post` or `reddit_subreddit`. The application parses returned HTML into `PostData`/`CommentData`, preserves supplied post/comment/parent IDs, timestamps, scores, URLs and subreddit provenance, and reports partial-thread markers without claiming completeness.
- **Hermes**: Reddit-specific provider remains disabled. No custom Hermes `/search`, `/scrape`, `/comments`, or `/discover` gateway routes are assumed to exist. Hermes is the orchestration/review layer, not a Reddit data source unless a documented tool contract is configured.

The Decodo adapter makes one synchronous `/scrape` request per bounded source URL using Bearer API-key authentication. `scrape_url()` returns one normalized post for compatibility; `scrape_source()` returns a bounded list for a subreddit-page snapshot. Neither operation paginates, expands load-more controls, persists data, or proves live retrieval. The first live Universal response must be inspected before deciding whether one page is sufficient or whether additional explicitly capped requests are needed.

This is an HTTP Web Data API adapter, not the separate `@decodo/mcp-server`. Target URLs are checked before the potentially billable boundary: dot segments, encoded separators/backslashes, controls, malformed escapes and ambiguous double encoding are rejected without a request. Safe Unicode slugs remain supported; the original validated URL is submitted unchanged.

The video reference demonstrates a Hermes-invoked reusable skill/helper that calls Decodo's Web Scraping API, with the agent supplying a request and interpreting results. Its live demo is Amazon price comparison, not a Reddit collection run; it does not establish Reddit pagination or a comment limit. The closest project mapping is for Hermes to plan and validate URL inputs while the deterministic Decodo adapter extracts into the existing schema. That orchestration is not wired end-to-end here; Hermes's unverified HTTP draft remains disabled.

## Limits and cost

Decodo documents `/scrape` pricing per 1,000 requests. The project defaults to `proxy_pool=premium` and `headless=html`, but both are configurable through `DECODO_PROXY_POOL` and `DECODO_HEADLESS` or explicit provider configuration. Do not assume an unsuccessful scrape is free: verify the account's current plan, remaining quota, proxy settings and dashboard billing before any live run. No live request or account-entitlement check was made as part of this change.

A proposed first Universal pilot is one explicitly approved subreddit-page request, initially capped at 10 parsed posts and 25 comments per post, followed by inspection of the returned HTML and diagnostics. Expansion to 10–20 retained posts, additional source pages, or separate post-page requests requires an explicit budget decision after that first response. Do not treat a page response as archive completeness or silently add pagination.

## Configuration and tests

The Decodo adapter reads `DECODO_API_TOKEN`, optional `DECODO_API_URL`, `DECODO_PROXY_POOL`, and `DECODO_HEADLESS` from the process environment. If using `.env`, the caller must load it first. The endpoint defaults to `https://data.decodo.com/v1/scrape`; overrides must identify that same HTTPS host/path, without credentials, query parameters or fragments. Use a **Web Data API Bearer key**, not an old Web Scraping API Basic token. Do not put credentials in source, logs or chat. Tests inject synthetic keys and mocked responses only.

From the repository root, using the existing environment:

```powershell
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"
$env:PYTHONDONTWRITEBYTECODE = "1"
& "D:\AI_Project\backend\.venv\Scripts\python.exe" -m unittest backend.test_arctic_shift backend.test_arctic_contract backend.test_decodo backend.test_decodo_html backend.test_rss backend.test_ingestion_manager backend.test_ingestion_contract backend.test_collection backend.test_pilot_integration -q
```

The collection tests use fake managers/persistence, including for CLI persistence tests; this command does not start a real collection or connect to PostgreSQL. Do not substitute unrestricted test discovery: other repository tests fit/train models or require a database.

## Shared normalization and failure contract

- `PostData` and `CommentData` remain unchanged. Providers do not infer `source_mode`; the collector/caller must establish direct versus context provenance separately.
- Ingestion timestamps are **naive UTC**, not local time. Epoch/offset values are converted to UTC; already-naive values supported by Arctic/Decodo retain the ingestion UTC convention. Required dates, IDs and subreddits are never replaced with current time, titles or request hints. RSS instead requires a source date with an explicit known zone.
- A top-level comment has `parent_reddit_id=None`; a nested reply has the parent's **bare comment ID**. The containing post supplies thread identity. These canonical fields do not encode discussion completeness.
- RSS has no observed score/comment count; the shared schema's zeros are defaults, **not measured counts or proof of no comments**.
- `ProviderRequestError` exposes bounded category and optional HTTP status. `invalid_json`, `invalid_xml` and `invalid_payload` remain distinct through `IngestionManager` wrapping. Errors do not include request URLs, tokens or response bodies. A valid empty discovery response is different from an invalid payload.
- Registry order remains `arctic_shift`, `rss`, `hermes`, `decodo`; capabilities still control dispatch. Hermes remains disabled. Use explicit provider selection for a bounded request; an automatic manager dispatch can try several capable providers. Decodo exposes both `scrape_url` (one-post compatibility) and `scrape_source` (bounded HTML-page parsing).
- Each adapter makes one request with a finite timeout, disables redirects, and does not automatically retry, paginate, expand `more` nodes, or fetch linked content. The existing collector has its own separately bounded retry policy; adapter changes do not disable or expand it.

URL provenance differs by adapter: Arctic preserves an optional returned `url` (which need not be a discussion permalink), RSS preserves the returned Reddit permalink, and Decodo retains the validated request URL after verifying returned post/subreddit identity. These are not interchangeable inferred URLs. The `requests` timeouts are connection/read timeouts, not an absolute end-to-end deadline or a response-byte/memory budget.

The downstream pipeline requires **timezone-aware** timestamps and an explicit provenance envelope. Its adapter does not silently convert naive ingestion values: a caller must explicitly acknowledge the ingestion UTC convention at that boundary and supply the post/parent context. See `docs/downstream_pipeline.md`. None of these capped provider snapshots establishes the `discussion_complete` assertion needed for resolution.

## Arctic Shift snapshot policy

`search_posts` forwards documented subreddit-scoped `query` (title/selftext search), `after`, `before`, `sort`, `limit` and `timeout`; a nonblank query requires an explicit subreddit. No implicit `CATpreparation` scope is added. Absolute bounds are compared/serialized in UTC; supported relative bounds remain relative on the wire, with calendar-aware month/year validation. The default lower bound is `30d`.

- Search: default 25, integer limit 1–100. One server page, deduplicated by post ID and locally capped. Conflicting duplicates or a returned subreddit outside an explicit request fail closed.
- Comments: default 100, integer request/output limit 1–25,000. Nested and flattened parents are checked against the requested root; duplicates do not inflate counts, cycles and contradictory parents fail, and valid replies under deleted parents are retained. Deleted/removed/empty parent bodies are not returned as content.
- The HTTP client defaults to a finite 60-second timeout; search/tree calls allow a validated positive finite override. No implicit pagination around tied timestamps occurs.
- `last_search_complete` and `last_comments_complete` remain `False`, even on a short/empty response. `last_comments_truncated` flags observed collapsed branches or local truncation, not every possible upstream omission. No completeness guarantee follows from an absent warning.
- Missing/malformed response `data`, required metadata, or tree nodes produces `invalid_payload` instead of a false empty success. A valid empty post-ID lookup is reported as not found.

The upstream README currently documents `query`; previous HTTP 422 failures are **not proven to mean that `query` is unsupported**. Endpoint availability and historical coverage remain unverified.

## Decodo Universal response and truncation policy

The request is `POST /v1/scrape` with a body shaped as:

```json
{
  "url": "https://www.reddit.com/r/IIMSambalpur/",
  "proxy_pool": "premium",
  "headless": "html"
}
```

No deprecated Reddit-specific target is sent. The response may be a raw HTML
body or a JSON envelope whose `content` contains HTML; the parser also retains
support for structured offline fixtures. `scrape_source()` parses up to the
caller cap (default 10 posts) and `scrape_url()` keeps the existing one-post
operation contract. HTTP/envelope/parser failures, missing required post fields,
and contradictory identities fail explicitly.

The HTML parser recognizes explicit post/comment attributes and field slots,
deduplicates identical IDs, rejects conflicting duplicates, preserves supplied
timestamps/scores/URLs, and leaves missing optional values absent. Missing
parents, load-more markers, malformed structure, upstream counts larger than
the returned set, and local caps produce static diagnostics. The default
comment cap is 25 per post; this is a local output cap, not a server-side
budget. Diagnostics never include raw HTML or credentials. Their absence does
not establish complete thread coverage.

## RSS2 / Atom discovery policy

RSS requires a valid RSS2 channel; Atom requires its standard namespace. Entry identity/subreddit must come from an actual returned HTTPS Reddit thread permalink, not the requested feed URL, caller hints, title or body. Explicit guid/id or subreddit category conflicts are rejected. Generic feeds without the supported Reddit permalink are not normalized as Reddit posts.

- Required: nonblank title, valid post permalink and publication timestamp. RSS uses zoned RFC822 `pubDate`. Atom prefers `published`, otherwise the actual `updated`; the latter may **not** be the original creation time. Missing dates never become the current time.
- Optional HTML/CDATA/XHTML content is preserved. Linked `content src`, enclosures and feed `next` links are not fetched.
- Default 25 unique posts; integer limit 0–100; zero performs no request. Dedupe precedes the output cap, keeping the first matching post; conflicting subreddit identities fail. Entries after the cap are not validated or fetched.
- One request, fixed 30-second timeout, no redirects/retries/pagination. Valid empty supported feeds return `[]`; malformed XML produces `invalid_xml`, while unsupported shapes or invalid entry metadata produce `invalid_payload`.

RSS does not itself filter to the five collection communities or establish direct/context provenance. Callers must retain returned source metadata and apply their approved-source policy explicitly.

## Contract references and live-check status

Authoritative references used for the offline adapter review:

- Arctic Shift API README: <https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md>
- Decodo migration/auth: <https://help.decodo.com/docs/web-data-api-migration-guide>
- Reddit post template: <https://help.decodo.com/docs/web-scraping-api-reddit-post>
- Real-time envelope: <https://help.decodo.com/docs/web-data-api-scrape-real-time-requests>
- Status codes: <https://help.decodo.com/docs/web-data-api-status-codes>
- Billing: <https://help.decodo.com/docs/web-data-api-billing>

No live provider smoke was performed. Live requests to every provider were explicitly prohibited in the final audit; credentials, access, quotas, entitlement, cost and upstream behavior remain unverified. This work does not demonstrate live retrieval, archive/window completeness, production NLP accuracy, or successful end-to-end persistence/reporting. Reference URLs above were retained from the earlier contract review, not refetched in the final audit.

## Current offline pilot-preparation check (2026-10-24)

The Universal HTML parser, Decodo request migration, and in-memory pilot bridge
were checked with the existing ingestion, pipeline, annotation, and manifest
contracts:

```powershell
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"
$env:PYTHONDONTWRITEBYTECODE = "1"
& "D:\AI_Project\backend\.venv\Scripts\python.exe" -m unittest backend.test_arctic_shift backend.test_arctic_contract backend.test_decodo backend.test_decodo_html backend.test_rss backend.test_ingestion_manager backend.test_ingestion_contract backend.test_collection backend.test_pilot_integration backend.test_nlp_pipeline backend.test_thread_evidence backend.test_annotation_templates backend.test_annotation_export backend.test_convert_annotation_candidates backend.test_annotation_manifest -q
```

Result: **400 tests passed, exit code 0**. The run uses synthetic HTML,
mocked provider boundaries, temporary in-memory/fixture data, and no live
provider or database operations. `scrape_source()` is an explicit bounded
capability; the first live Universal response must still be inspected before
choosing any additional page or request budget.

## Earlier integration verification (pre-final-audit; 2026-10-15)

The preceding integration report recorded **318 tests** for this command. Before audit edits, independent test-ID enumeration confirmed the selection contained 318 distinct tests; the current expanded counts are recorded in the final audit below.

```powershell
& "D:\AI_Project\backend\.venv\Scripts\python.exe" -m unittest backend.test_arctic_shift backend.test_arctic_contract backend.test_decodo backend.test_rss backend.test_ingestion_manager backend.test_ingestion_contract backend.test_collection backend.test_nlp_pipeline backend.test_thread_evidence -q
```

| Area | Tests |
| --- | ---: |
| Arctic Shift adapter and HTTP/tree contracts | 64 |
| Decodo | 44 |
| RSS | 42 |
| Manager and shared ingestion contract | 22 |
| Fake-manager collection/CLI safety | 76 |
| Pipeline and thread evidence | 70 |

The preceding report recorded **376 safe downstream regressions** for the second command, giving **694 tests** across the two selections. Independent pre-edit enumeration confirmed 376 distinct IDs in this selection and **zero intersection** with the first selection, including after normalizing `backend.` import aliases. Thus 694 is the correct unique-test selection size for that pre-audit tree, not a count to reuse after new regressions.

```powershell
& "D:\AI_Project\backend\.venv\Scripts\python.exe" -m unittest backend.test_language_detection backend.test_preprocessing backend.test_readiness backend.test_ml_evaluation backend.test_ml_splitting backend.test_resolution_detection backend.test_issue_solution_linking.TestEvidenceModels backend.test_issue_solution_linking.TestConfig backend.test_issue_solution_linking.TestCandidateGeneration backend.test_issue_solution_linking.TestAnnotationSchema -q
```

Only the four named linking test classes were selected; scoring/evaluation/result fixtures that invoke the corpus-fitting linker, training/topic/neural suites, and database integration suites were not run. CSV round trips use synthetic temporary fixtures, not project annotations. Negative CLI cases intentionally print argument errors; the combined run exited successfully.

`py_compile` passed for 16 scoped Python files with bytecode confined to a temporary directory and removed afterward. `python -m pip check` reported no broken requirements. `git diff --check` and separate no-index checks covering the scoped untracked files reported no whitespace errors; Git emitted line-ending normalization notices only. No collection, live HTTP, PostgreSQL access, migrations, model fitting/training, model artifact writes, annotation/source overwrites, or git staging/commits were performed in this integration pass.

## Final independent audit

### Findings and narrow corrections

- The evidence adapter could publish complete coverage/verdicts with two comments although the actual resolver reported insufficient evidence (default minimum: three). It now checks valid resolver configuration, configured volume and strong pair-specific support before calling the resolver, and verifies returned `evidence_sufficient`. Exclusions are not `unresolved` verdicts; the standalone resolver was not modified.
- A classifier could mutate its input documents and have the pipeline validate echoed text against that mutation. Classifiers now receive deep copies while source-derived validation inputs remain authoritative.
- Contradictory optional raw provenance could disappear when neutral prediction schemas dropped extra fields. Raw echoes and bounded/cycle-safe prediction values are checked before schema normalization/serialization. Invalid classifier batches still fail atomically; independent stages continue.
- Decodo accepted ambiguous dot-segment/encoded-separator post paths. Validation now rejects these and related ambiguous/control/escape forms before `requests.post`; positive Unicode cases remain covered.
- One manager test depended on ambient Decodo configuration. Its initial independent run produced **13 tests, one error** with a synthetic configured key. The fixture now explicitly isolates configuration and adds a configured-invalid-target/no-request regression; the final **14 tests pass**. Production manager dispatch/order was not changed.

Adversarial reproductions and new red/green regression runs were performed before the corrections. Their repeated executions are not additional unique tests. No claim is made that every development invocation passed.

### Final independently executed selections

Each selector below was executed in a separate subprocess as:

```powershell
$python = "D:\AI_Project\backend\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python -m unittest <selector-from-table> -q
```

The fresh final rerun on **2026-10-16** used two wrappers, with every selector launched in its own subprocess. Each wrapper captured stdout/stderr, printed the actual count and process exit code, and would print captured failure output if a selector failed. All **19 selectors exited 0: 707 executions, zero failed selections**. Synthetic provider credentials and mocked HTTP were used; the approved temporary directory was verified before use and test fixtures remained outside the workspace. This is not unrestricted repository discovery.

| Selector (all prefixed `backend.`) | Tests |
| --- | ---: |
| `test_arctic_shift` | 14 |
| `test_arctic_contract` | 50 |
| `test_decodo` | 48 |
| `test_rss` | 42 |
| `test_ingestion_manager` | 14 |
| `test_ingestion_contract` | 9 |
| `test_collection` | 76 |
| `test_nlp_pipeline` | 40 |
| `test_thread_evidence` | 38 |
| **Ingestion/pipeline subtotal** | **331** |
| `test_language_detection` | 33 |
| `test_preprocessing` | 66 |
| `test_readiness` | 17 |
| `test_ml_evaluation` | 14 |
| `test_ml_splitting` | 16 |
| `test_resolution_detection` | 166 |
| `test_issue_solution_linking.TestEvidenceModels` | 5 |
| `test_issue_solution_linking.TestConfig` | 17 |
| `test_issue_solution_linking.TestCandidateGeneration` | 24 |
| `test_issue_solution_linking.TestAnnotationSchema` | 18 |
| **Safe downstream subtotal** | **376** |
| **Final total** | **707** |

Independent enumeration after the fresh rerun flattened the actual loaded suites, checked loader errors, and canonicalized IDs as module (without an optional `backend.` prefix), class qualified name and test method. Final selection sizes are **331/331 unique** and **376/376 unique**, with **intersection 0** and **union 707**. This inventory agrees with the successful subprocess summaries; inventory alone was not used as execution evidence. The audit added 13 methods: Decodo 4, pipeline 3, evidence 5, manager 1. This reconciles the historical 694 with the final 707; retries/subtests do not inflate the unique-method total. Earlier runs and delegated red/green runs are excluded from the final consolidated execution count, not claimed absent.

Sorted canonical-ID union SHA-256 (identity/discovery evidence, not a code or results hash):

- Pre-audit 694-ID selection: `1cd4c29e36242653df70cc99f382a56fc03ab5de5bc95f25a5734ae9952118a3`
- Final 707-ID selection: `18f4e1f73cb38a6b9f7265a65913944d98389341116ae1d0529a340fa8e3945a`

### Fresh static checks and preservation

- `py_compile` passed for 16 scoped implementation/test files, without application imports; bytecode was written only to an approved temporary directory and removed.
- `python -m pip check`: `No broken requirements found`, exit 0.
- `git diff --check`: exit 0. Relevant untracked pipeline/evidence implementations, tests, shared/Arctic contract tests and downstream documentation were separately checked with `git diff --no-index --check -- NUL <path>`: no whitespace diagnostics. Status 1 indicates differing content, not a whitespace defect; line-ending notices were not treated as failures.
- The working tree already had extensive tracked/untracked changes. Audit writes were limited to the nine files listed below; no destructive Git operation, staging or commit was used. No independent file-hash baseline was captured, so this is scoped-action preservation evidence, not a byte-for-byte certification of every unrelated file.

Audit-touched files: `backend/app/services/nlp/pipeline.py`, `backend/app/services/nlp/thread_evidence.py`, `backend/app/services/ingestion/providers/decodo.py`, `backend/test_nlp_pipeline.py`, `backend/test_thread_evidence.py`, `backend/test_decodo.py`, `backend/test_ingestion_manager.py`, `backend/docs/ingestion.md`, `docs/downstream_pipeline.md`. Existing Arctic/RSS/base/schema/collection changes are not attributed to this audit. The earlier swarm board was absent in the final audit and was not recreated.

### Release gates: offline contracts pass; production release remains blocked

1. **Live provider operation is unverified.** No Arctic Shift, RSS, Reddit or Decodo request was allowed. Account/credential validity, quotas, entitlement, billing, current responses and availability need a separately authorized bounded check. A failed Decodo request must not be assumed free.
2. **Source coverage is not discussion completeness.** Single responses, caps, omitted parents and unexpanded branches do not establish a full thread. Resolution needs independently established completeness, complete prediction processing, configured discussion volume and strong pair-specific support.
3. **No approved model readiness/quality claim.** Searches for common artifact binary extensions found none in the inspected artifact directories; no binary was loaded. Trusted artifacts, annotation genuineness, held-out validation and model approval remain separate requirements.
4. **The bundled linker is not no-fit.** `TfidfCosineIssueSolutionLinker.link()` calls `fit_transform` and is blocked before invocation (including subclasses). A reviewed no-fit implementation is still needed; topic discovery also fits and remains unavailable.
5. **The application boundary is incomplete.** Ingestion naive UTC and downstream aware timestamps need an explicit validated bridge, including authoritative provenance and thread/parent context. The in-memory pipeline is not wired into a complete collection-to-inference API/UI/reporting flow. No PostgreSQL/persistence/migration/end-to-end deployment validation was performed.

No new implementation milestone, collection, training, data export or live pilot was started by this audit.
