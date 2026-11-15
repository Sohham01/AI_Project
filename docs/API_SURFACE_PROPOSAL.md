# API Surface Proposal (Phase H)

**Status:** PROPOSAL ONLY — no routes are implemented in this phase.
**Revision:** 2 (staged-implementation review)
**Date:** 2026-10-11
**Scope:** the *minimum useful* HTTP surface, restricted to capabilities the
repository actually supports today. No product/invented endpoints, no
superficial frontend.

> This revision resolves the access-control conflict, defers the synchronous
> collection endpoint, tightens status output, constrains annotation readiness
> to `task=response`, defines outcome/error contracts, specifies DB-level
> pagination, and records the verified Alembic base-table gap with a
> disposable-database test plan.
>
> **Verified findings** (reproduced against disposable databases only) are
> marked ✅. Unverified statements are marked ⚠️ (assumption, needs a test).

---

## 0. Verified findings vs. assumptions

| # | Finding | Status |
| --- | --- | --- |
| F1 | No authentication/authorization exists anywhere in `app/` (no `Depends`, no middleware, no API-key check, no user model). | ✅ grep over `app/**.py` |
| F2 | Provider timeouts are per-request only: Arctic Shift `60s`, RSS `30s`, Decodo `60s`, Hermes `120s`; `run_collection` has **no overall deadline**. | ✅ source |
| F3 | `load_comment_candidates` runs `select(RedditComment)` (all rows) then loads all posts; **no** `.limit()` / `.offset()` anywhere in `app/`. | ✅ source |
| F4 | Readiness supports `task="response"` only; any other task raises `ValueError`. | ✅ `ml/data/readiness.py::DEFAULT_TASK_SCHEMAS` |
| F5 | Alembic chain is linear `e9fa8685c983`(root, `down_revision=None`) → `966e480ef159` → `17cb1444081b` → `8f1300a2a8f0`. | ✅ migration files |
| F6 | Base tables `reddit_posts`, `reddit_comments`, `data_sources` are **never** created by a migration; the root migration only `add_column` + `create_foreign_key` on the pre-existing `reddit_posts`. | ✅ migration files |
| F7 | Fresh `alembic upgrade head` on an empty disposable SQLite DB fails at the first migration: `no such table: reddit_posts` (only `alembic_version` remains). | ✅ reproduced |
| F8 | With base tables pre-created, migrations fail on SQLite at `create_foreign_key`: `No support for ALTER of constraints in SQLite dialect`. | ✅ reproduced |
| F9 | `GET /health/db` currently returns raw `f"{type(exc).__name__}: {exc}"` to an unauthenticated caller, which can disclose host/DB details. | ✅ `app/main.py` |
| F10 | `test_app_startup.py` asserts a single head and that the **NLP** tables are created by a migration, but does **not** assert base tables are created by migrations. | ✅ source |

Implication of F8: any migration test that actually exercises `e9fa8685c983`
must use a **disposable PostgreSQL** database, not SQLite. SQLite can only be
used for a *structural* (AST/string) assertion.

---

## 1. Current state (audit)

| Area | Reality in the repo |
| --- | --- |
| Framework | FastAPI `0.141.1`, Starlette `1.6.0`, Uvicorn `0.53.0`. |
| App entry | `backend/app/main.py` — a single module; `app/api/` is **empty**. |
| Existing routes | `GET /`, `GET /health`, `GET /health/db` only. |
| Auth | **None** (F1). |
| DB | SQLAlchemy 2.0 + Alembic. `get_engine()`/`get_sessionmaker()` lazy; `ConfigurationError` when `DATABASE_URL` is missing. |
| Background work | **None.** No worker/queue/scheduler/in-process runner. |
| Schema layer | `app/schemas/` = `ingestion.py` only. |
| Collection | `IngestionManager.execute_operation(...)` + `CollectionPlan`/`run_collection` (dry-run default). |
| Annotation readiness | `ml.data.readiness.analyse_annotation_csv` / `ReadinessReport`. |
| Pipeline/model state | `ml.configs.ModelRegistry`, `TaskConfig`, `Trainer`. |

---

## 2. Access-control boundary (resolves the scheduling conflict)

The previous revision mixed "loopback-only", "env flag", and "required auth",
and scheduled some admin endpoints **before** any authentication existed. That
is inconsistent and untestable. This section defines one boundary.

### 2.1 Two tiers only

- **public** — no credentials; may expose only non-operational liveness data.
- **admin** — requires a valid admin credential.

**Explicitly rejected as access control:**

- *Loopback / source-IP checks* — spoofable behind proxies and untestable in
  process; **not** authentication.
- *Environment feature flags* (e.g. `ENABLE_COLLECTION_API`) — these are
  **feature gates** (defense in depth), not authentication. They may disable an
  endpoint, but their presence never authorizes a request.

### 2.2 Mechanism: admin API key dependency (testable, no user model needed)

Add `app/api/deps.py::require_admin` — a FastAPI dependency:

1. Read expected secret from `ADMIN_API_KEY` env.
   - If unset/blank or shorter than a minimum length (e.g. 32 chars) →
     **fail closed**: `503 {"detail":"admin API is not configured","code":"admin_not_configured"}`.
2. Read the presented key from the `X-Admin-Key` request header.
3. Compare with `hmac.compare_digest` (constant time).
4. Missing/wrong → `401 {"detail":"missing or invalid admin credentials","code":"unauthorized"}`.

Properties:

- No users, sessions, JWT, or DB dependency — just an env secret and a header.
- Unit-testable by injecting the header value and patching `ADMIN_API_KEY`;
  no network, no DB.
- `ADMIN_API_KEY` is a secret: never logged, never echoed, never returned.

All **admin** routers declare `dependencies=[Depends(require_admin)]`.

### 2.3 Which tier each route uses

| Route | Tier |
| --- | --- |
| `GET /health` | public |
| `GET /api/v1/status` | public (**fields limited to §3.1 public subset**) |
| `GET /health/db` | **admin** after this change (see B3) |
| `GET /api/v1/admin/status` | admin |
| `GET /api/v1/annotations/readiness` | admin |
| `GET /api/v1/annotations/candidates` | admin |
| `GET /api/v1/pipeline/status` | admin |
| `POST /api/v1/collection/preview` | admin **and deferred** (§5) |

**Consequence for sequencing:** the `require_admin` dependency must land
*before* the first admin endpoint. This resolves the original conflict: only
`GET /api/v1/status` (public subset) and the health routes are scheduled ahead
of authentication; every admin endpoint comes after.

### 2.4 Access-boundary test matrix

| Case | Expectation |
| --- | --- |
| `ADMIN_API_KEY` unset, call admin route | `503 admin_not_configured` |
| `ADMIN_API_KEY` set, no `X-Admin-Key` | `401 unauthorized` |
| wrong key | `401 unauthorized` |
| correct key | handler runs |
| public route, any/no key | handler runs; response contains no admin fields |

---

## 3. Conventions

- Prefix new routes `/api/v1` via `include_router` in `main.py`; new routers in
  `app/api/` (`status.py`, `admin_status.py`, `annotations.py`, `pipeline.py`).
- Request/response models live in `app/schemas/`.
- Every list response is capped; every DB/outbound call has an explicit
  timeout.
- One error shape everywhere: `{"detail": str, "code": str}` (§6).

---

## 4. Endpoint specification

Endpoint status values: **proposed**, **deferred**. Nothing is approved yet.

### 4.1 `GET /api/v1/status` — public subset  (status: proposed)

- **Purpose:** minimal liveness/identity for clients and uptime checks.
- **Auth:** public.
- **Effect:** read-only, no DB, no outbound, no filesystem.
- **Public response (the *only* fields publicly exposed):**

  ```json
  { "service": "ok", "version": "1.0.0" }
  ```

- **Never public:** `database`, `auto_create_schema`, `collection_api_enabled`,
  `model_registry`, entry counts, dataset paths, or any exception detail.
- **Validation/timeout:** none needed (constant response).
- **Reuses:** the app `version` already declared in `main.py`.
- **Acceptance tests:**
  - Returns exactly `{service, version}`; no other keys.
  - Response body does not contain the substrings `db`, `registry`, `key`.
  - Works with `DATABASE_URL` unset (import-time and request-time).

### 4.2 `GET /api/v1/admin/status` — operational details  (status: proposed)

- **Purpose:** operator-facing operational state (previously mixed into
  `/api/v1/status`).
- **Auth:** **admin**.
- **Effect:** read-only; a `SELECT 1` DB probe (≤2s) and registry read; no
  writes, no outbound.
- **Response:**

  ```json
  {
    "database": "connected | not_configured | unavailable",
    "auto_create_schema": false,
    "collection_api_enabled": false,
    "model_registry": {"present": true, "entries": 13}
  }
  ```

- **Sanitization:** on DB failure return the *category*, never the raw
  exception text. `not_configured` carries no URL.
- **Reuses:** `get_engine`, `ConfigurationError`, `ModelRegistry`.
- **Acceptance tests:**
  - Unauthenticated → `401`; unconfigured key → `503`; valid key → `200`.
  - DB missing → `database=="not_configured"`, **no** secret echoed.
  - Registry file missing → `model_registry.present==false`, still `200`.

### 4.3 `GET /api/v1/annotations/readiness`  (status: proposed)

- **Purpose:** expose `preflight.py` results for CI/ops gating.
- **Auth:** **admin**.
- **Effect:** read-only local file; no DB, no outbound.
- **Task scope (resolves review #3):** **`task="response"` only** for the first
  release. Verified (F4): `ml/data/readiness.py::DEFAULT_TASK_SCHEMAS` defines
  only `response`; other tasks raise `ValueError`, and there is **no** safe
  task→file mapping in the codebase. Any other `task` → `422 unsupported_task`
  with the allowed list (`["response"]`).
- **File path:** server-configured (`ANNOTATION_CSV`), **never** a query
  parameter (path-traversal risk).
- **Request:** `task` (default/only `response`), `group_by` (default
  `post_id`).
- **Response:** `ReadinessReport.to_dict()` + a `muril` block
  (`eligible`, `blockers`, `min_total`, `min_per_class`).
- **Acceptance tests:**
  - `?task=sentiment` → `422` listing `response`.
  - Missing configured file → `200`, `verdict=="not_ready"`, non-empty `errors`.
  - Small file → `verdict=="warn"` and a warning that metrics are **not** real.
  - ≥300 balanced rows → `muril.eligible==true`.
  - Unauthenticated → `401`.

### 4.4 `GET /api/v1/annotations/candidates`  (status: proposed)

- **Purpose:** return annotation **candidate rows** for a human review tool.
- **Auth:** **admin**.
- **Effect:** read-only DB query; no file write, no outbound.
- **Pagination at the database level (resolves review #6):** verified (F3) that
  `load_comment_candidates` currently loads *every* comment and *every* post.
  The endpoint **must not** reuse that behaviour as-is. Required change to
  `app/services/annotation/candidates.py`:

  ```python
  # page the comments in SQL
  stmt = select(RedditComment).order_by(RedditComment.id)
  if post_reddit_id:
      post_pk = session.scalar(
          select(RedditPost.id).where(RedditPost.reddit_id == post_reddit_id)
      )
      if post_pk is None:
          return []
      stmt = stmt.where(RedditComment.post_id == post_pk)
  stmt = stmt.limit(limit).offset(offset)          # <- enforced in the DB
  comments = list(session.execute(stmt).scalars())
  # fetch only the posts and parent comments referenced by this page
  ```

  - `limit`: default `100`, hard max `500`; `offset` non-negative.
  - `post_id` filter is pushed into the `WHERE` clause.
  - Parent text is resolved only for the parent ids present on the page (no
    full-table load).
  - A separate `select(func.count())` provides `total` without materialising
    rows.
- **Raw discussion text controls:** the response contains verbatim user text,
  so:
  - admin-only (never public);
  - `Cache-Control: no-store`;
  - response is **not** logged (the request logger records route + duration +
    status only, never bodies);
  - `limit` capped as above so a single call cannot exfiltrate the corpus;
  - no author/identity fields exist on the models, and none are added.
- **Response:** `{"count": n, "total": m, "rows": [ {CANDIDATE_COLUMNS...} ]}`,
  `response_type` always `""`.
- **Acceptance tests (with a fake session / tiny fixture):**
  - Query string contains `LIMIT` (assert the compiled SQL), proving DB-level
    paging (guards against a regression to full-table load).
  - `limit>500` → `422`; empty DB → `200 {"count":0,"total":0}`.
  - Every row has `response_type==""`; response headers include `no-store`.
  - Unauthenticated → `401`.

### 4.5 `GET /api/v1/pipeline/status`  (status: proposed)

- **Purpose:** report registered tasks/models and recorded registry metadata.
- **Auth:** **admin**.
- **Effect:** read-only registry read; never opens artifact binaries.
- **Request:** optional `task` filter (validated against `TaskType`).
- **Response:**

  ```json
  {
    "tasks": [
      {"task": "response",
       "available_models": ["muril", "response_mlp", "text_cnn", "tfidf_logistic"],
       "registry_entries": [
         {"model": "tfidf_logistic", "version": "...", "trained_at": "...",
          "metrics": {"accuracy": 0.0}, "synthetic": true}
       ]}
    ]
  }
  ```

  `synthetic`/dataset provenance is surfaced so demo metrics cannot be
  presented as real.
- **Acceptance tests:** empty registry → `200` empty lists;
  `available_models("response")` returns the four names; unauthenticated → `401`.

### 4.6 `POST /api/v1/collection/preview`  (status: **DEFERRED**)

**Decision (resolves review #4):** defer. The current synchronous service
cannot guarantee an overall deadline, so exposing it request-bound is unsafe.

Evidence (F2): each request may block up to its provider timeout
(Arctic Shift 60s), and `run_collection` issues `|subreddits| × |terms|` search
requests plus one comment request per kept post, with `request_delay` sleeps in
between. Worst case exceeds any reasonable HTTP deadline.

Required enablers before this endpoint can be scheduled:

1. Inject a configurable per-request timeout through
   `IngestionManager` → provider → client (today only the client default 60s
   applies).
2. Add a monotonic **deadline** to `run_collection` (e.g. `deadline_seconds`);
   before each request compute `per_request = min(provider_timeout,
   remaining)`, and stop starting new requests once exhausted; report
   `outcome="partial"` with a `deadline_exceeded` note.
3. Only then may a request-bound preview promise a bounded wall clock
   (e.g. ≤30s). For sustained use it still belongs behind a worker/queue.

No `/collection` **persist** endpoint is proposed at all.

---

## 5. Failure semantics and error contract (resolves review #5)

### 5.1 Collection outcomes are distinct

When preview is eventually implemented, a single `outcome` field is returned:

| Outcome | Condition | HTTP |
| --- | --- | --- |
| `complete` | `posts_kept > 0` and `errors == []` | `200` |
| `partial` | `posts_kept > 0` and `errors != []` | `200` |
| `empty` | `posts_kept == 0` and `errors == []` (nothing matched / all out of scope) | `200` |
| `failed` | `posts_kept == 0` and `errors != []` (all sources failed) | `502` |

`partial`/`empty` are **not** errors; `failed` is. This prevents the previous
ambiguity where a total outage and a normal empty result both produced `200`
with `posts_kept == 0`.

### 5.2 One sanitized error contract

Every non-2xx body:

```json
{ "detail": "human-readable, secret-free", "code": "short_snake_case" }
```

- **Request-validation errors** (`RequestValidationError`): a handler maps to
  `422 {"detail":"request validation failed","code":"validation_error","errors":[{"loc":[...],"msg":...,"type":...}]}`.
  Only `loc`/`msg`/`type` are echoed — never submitted values.
- **Domain errors** via registered handlers:
  - `ConfigurationError` → `503 configuration_error`
  - plan/limit `ValueError` → `422 invalid_parameter`
  - `UnsupportedOperation` → `501 unsupported_operation`
  - unknown `Exception` → `500 internal_error` with a generic detail and a
    correlation id; the stack trace is logged server-side **without** bodies
    or secrets and never returned.
- No handler may echo `DATABASE_URL`, tokens, provider response bodies, or
  `.env` contents.

---

## 6. Cross-cutting requirements

- **Timeouts:** DB probe ≤2s; any future outbound route bounded per §4.6.
- **Rate limiting:** admin routes are credential-gated; add rate limiting when
  preview is un-deferred.
- **`/health/db` hardening (B3):** either move detailed DB status behind
  `require_admin`, or return only `{"database":"connected"}` /
  `{"database":"unavailable"}` with **no** exception string. Current behaviour
  (F9) leaks exception text publicly.
- **Logging:** route + duration + status only; never request/response bodies.
- **Secrets:** `ADMIN_API_KEY` and `DATABASE_URL` are never logged or returned.

---

## 7. Alembic base-table gap (resolves review #7)

### 7.1 Verified history

```
e9fa8685c983  (root, down_revision=None)  add_column reddit_posts.source_id
                                          + create_foreign_key -> data_sources
                                          + create_index
966e480ef159  create nlp_analysis_posts/comments; alter reddit_posts.source_id
17cb1444081b  add detected_languages to nlp tables
8f1300a2a8f0  add normalized_text/preprocessing_version to nlp tables   (head)
```

- **F6:** no migration contains `create_table('reddit_posts'`,
  `('reddit_comments'`, or `('data_sources'`. The root migration assumes they
  already exist.
- **F7 (reproduced):** fresh `upgrade head` on an empty DB fails at
  `e9fa8685c983`: `no such table: reddit_posts`.
- **F8 (reproduced):** on SQLite, even with base tables present, migrations
  fail at `create_foreign_key` (no ALTER-of-constraint support).

### 7.2 Recommended fix (staged; not done here)

Add a new **root** migration `0001_create_base_tables` that creates
`data_sources`, `reddit_posts`, `reddit_comments` (guarded by
`sa.inspect(bind).has_table(...)` for idempotency), then re-parent
`e9fa8685c983.down_revision` to it. Chain becomes
`0001 → e9fa8685c983 → 966e480ef159 → 17cb1444081b → 8f1300a2a8f0` with the
head unchanged.

- **Fresh DB:** `upgrade head` now creates base tables first, then the rest.
- **Existing DB:** already stamped at `8f1300a2a8f0`; the inserted root is
  *behind* the current version, so `upgrade head` is a no-op and data is
  untouched. The guard makes the migration safe even if re-run. Do **not**
  mutate the already-applied `e9fa8685c983` (mutating applied migrations is
  unsafe for existing installs).
- Document in `backend/docs/` that pre-migration databases were bootstrapped
  out-of-band and why the baseline migration exists.

### 7.3 Disposable-database test plan (never the configured DB)

Hard rule for all these tests: read **`TEST_DATABASE_URL`**, never
`DATABASE_URL`. If `TEST_DATABASE_URL` is unset, or equals `DATABASE_URL`,
**skip** and emit a loud message. Tests create/drop their **own** schema/DB and
never connect to the working database.

| Test | DB | Steps | Assertion |
| --- | --- | --- | --- |
| T-STRUCT (offline) | none | Parse migrations (extends the existing `TestMigrationGraph` pattern) | The three base tables appear in a `create_table(...)` call |
| T-FRESH | disposable **PostgreSQL** | create empty DB → `alembic upgrade head` (env `DATABASE_URL=TEST_DATABASE_URL`) | all 5 tables exist; expected columns/FKs/indexes present; `alembic current` == single head |
| T-EXISTING | disposable **PostgreSQL** | `Base.metadata.create_all()` → `alembic stamp 8f1300a2a8f0` → `alembic upgrade head` | no-op; head unchanged; all 5 tables intact; no data loss |

- T-FRESH/T-EXISTING must use PostgreSQL (F8): SQLite cannot execute
  `create_foreign_key`/`alter_column`. Gate them behind `TEST_DATABASE_URL`.
- T-STRUCT runs in the normal offline suite and will **fail until** the
  baseline migration is added — i.e. it is the regression guard for this gap.
  It must not be added before the fix, or the suite goes red.

---

## 8. Preservation constraints

This pass changes **no** application behaviour: no routes implemented, no
migration added/edited, no collection/annotation/preflight/ML code changed, no
tests weakened, files not staged/committed, secrets not exposed. The proposal
reuses existing contracts (`CollectionPlan`/`run_collection`, `ReadinessReport`,
`build_candidate_rows`, `ModelRegistry`, `available_models`).

---

## 9. Approval status, blockers, sequence

### 9.1 Endpoint approval status

| Endpoint | Tier | Status |
| --- | --- | --- |
| `GET /api/v1/status` (public subset) | public | proposed — candidate for first approval |
| `GET /api/v1/admin/status` | admin | proposed |
| `GET /api/v1/annotations/readiness` (`task=response`) | admin | proposed |
| `GET /api/v1/annotations/candidates` | admin | proposed (needs DB-level paging) |
| `GET /api/v1/pipeline/status` | admin | proposed |
| `POST /api/v1/collection/preview` | admin | **deferred** (needs deadline support) |
| `POST /collection` persist, `/annotations/export`, `/pipeline/train`, auth/login, frontend | — | **not proposed** |

### 9.2 Outstanding blockers

| ID | Blocker | Blocks |
| --- | --- | --- |
| B1 | `require_admin` dependency not implemented | all admin endpoints |
| B2 | `load_comment_candidates` loads all rows (F3) | candidates endpoint |
| B3 | `/health/db` leaks exception text publicly (F9) | public surface hardening |
| B4 | No overall deadline in `run_collection` (F2) | collection preview |
| B5 | Base-table migrations missing (F6/F7) | fresh installs / T-FRESH |
| B6 | No disposable-Postgres test fixture | T-FRESH/T-EXISTING |
| B7 | `httpx` not installed → no `fastapi.testclient` | route-level tests (call handlers directly, as the existing tests do) |

### 9.3 Smallest recommended implementation sequence

1. **`require_admin` dependency** (`app/api/deps.py`) + test matrix (§2.4) and
   sanitize `/health/db` (B3). No endpoint exposed yet.
2. **`GET /api/v1/status`** (public subset).
3. **`GET /api/v1/admin/status`** and **`GET /api/v1/pipeline/status`**.
4. **`GET /api/v1/annotations/readiness`** (`task=response` only).
5. **`GET /api/v1/annotations/candidates`** after the DB-level paging change
   (B2).
6. **Baseline migration fix + disposable-Postgres tests** (B5/B6) —
   independent track; can run in parallel.
7. `POST /api/v1/collection/preview` only after B4 is resolved.

### 9.4 First endpoint that is safe to implement, and the gate for the next

**Implement first:** `GET /api/v1/status` returning **only**
`{"service","version"}`. It needs no auth, no DB, no outbound call, no
filesystem read, and cannot leak operational detail.

**Before the next endpoint** (`GET /api/v1/admin/status`), the following must
be complete:

1. `require_admin` implemented and passing the full §2.4 access-boundary test
   matrix (fail-closed on missing/short `ADMIN_API_KEY`, `401` on bad key,
   constant-time compare).
2. `/health/db` hardened so public callers receive no exception text (B3).

Only once those are green may the first admin endpoint be added, and every
subsequent admin endpoint reuses the same dependency.
