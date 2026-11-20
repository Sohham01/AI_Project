# RCI — First Authorized Real-Data Pilot Guide

**Project:** Reddit Community Intelligence (RCI) — `D:\AI_Project`  
**Milestone:** Controlled 10–20 Thread Authorized Real-Data Pilot (Path 1: Static Export Intake)  
**Date:** 2026-11-19

---

## 1. Current Repository Data Status (Honest Inventory)

As of 2026-11-19, **zero genuine authorized Reddit export files are present in `D:\AI_Project\data\raw\`**.

| Directory | Contents Right Now | Status |
| --- | --- | --- |
| `data/demo_synthetic/` | 160 generated posts, 1,280 generated comments (seed `20261116`) | Synthetic prototype only (preserved untouched) |
| `data/raw/` | Empty | **Awaiting operator-supplied authorized export (10–20 threads)** |
| `docs/pilot_examples/` | Sanitized intake example generated from a **clearly labeled synthetic fixture** | End-to-end intake/privacy/manifest verification example only — **not real Reddit data** |

> [!CAUTION]
> **No Fabricated Ingestion:** Because no authorized real-world export currently exists on disk, no real-data ingestion or real-data model training was executed in this pass. All intake, privacy-redaction, schema-validation, and annotation-conversion tooling is complete, tested (`69/69` pilot/intake/annotation tests passing), and ready for the operator to supply the first authorized batch.

---

## 2. Exact Files Needed for the Next Step

To start the first real-world pilot (**target: 10–20 relevant public discussion threads** from IIM Sambalpur, MBA, or CAT communities obtained through permitted access — **no automated scraping, authentication bypass, or rate-limit evasion**):

1. Place your legitimately obtained raw export file(s) inside a read-only input directory outside version control (already gitignored by `.gitignore`):
   ```text
   D:\AI_Project\data\raw\pilot_v1\
   ```
2. Any of these **four authorized formats** are supported directly by [backend/import_reddit_export.py](file:///D:/AI_Project/backend/import_reddit_export.py):
   - **Format 1 — Reddit Thread JSON (`.json`):** One file per thread (`[post_listing, comment_listing]`).
   - **Format 2 — Reddit Subreddit / Search Listing JSON (`.json`):** `{"kind": "Listing", "data": {"children": [...]}}`.
   - **Format 3 — Arctic Shift / Pushshift Archive JSONL (`.jsonl` / `.ndjson`) or JSON envelope (`{"data": [...]}`):** Separate or combined post and comment records.
   - **Format 4 — Pre-structured `format_version: 1` Static Export JSON/JSONL:** Matching [docs/static_dataset_intake.md](file:///D:/AI_Project/docs/static_dataset_intake.md).
3. **Permitted scope & privacy checklist before placing files in `data/raw/pilot_v1/`:**
   - Only public discussions legitimately obtained for academic research.
   - Exclude private messages, moderator logs, doxxing threads, or sensitive personal/medical/financial-account disclosures.
   - Keep `data/raw/pilot_v1/` strictly separate from sanitized outputs (`data/processed/pilot_v1/`) and annotation sheets (`data/annotations/pilot_v1/`).

---

## 3. End-to-End Pilot Execution Workflow

### Step 1 — Sanitize, Validate & Generate Blank-Label Candidates

Run [backend/import_reddit_export.py](file:///D:/AI_Project/backend/import_reddit_export.py) (read-only on `data/raw/pilot_v1/`):

```powershell
Set-Location D:\AI_Project
.\backend\.venv\Scripts\python.exe backend\import_reddit_export.py `
  --input data\raw\pilot_v1 `
  --output data\processed\pilot_v1\sanitized_static_export.json `
  --dataset-id real-pilot-20261119-v1 `
  --community IIMSambalpur `
  --source-provider authorized-static-export `
  --source-mode direct `
  --authorization-status pending-review `
  --authorization-reference "<your-irb-or-project-approval-ref>" `
  --export-candidates data\annotations\pilot_v1 `
  --quality-report data\processed\pilot_v1\dataset_quality_provenance_report.json
```

**What this guarantees automatically:**
- **Author & PII Minimization:** Strips `author`, `author_fullname`, and author flair keys; redacts inline email addresses (`<EMAIL>`), 10–13 digit phone numbers (`<PHONE>`), and Reddit username mentions `u/...` / `/u/...` (`<USER_MENTION>`) without corrupting academic numbers (`CAT 2026`, percentiles, fee amounts).
- **Zero Silent Fabrication:** Absent `score` or `num_comments` values are omitted rather than turned into `0`, allowing [static_export.py](file:///D:/AI_Project/backend/app/services/ingestion/static_export.py) to flag `lossy_default_fields` honestly. Naive/invalid timestamps, `[deleted]`/`[removed]` comments, blank bodies, duplicate IDs, orphaned comments, and unresolved parent references are explicitly counted in `dataset_quality_provenance_report.json`.
- **Canonical Artifacts Produced (All Separate from Raw Input):**
  1. `data/processed/pilot_v1/sanitized_static_export.json` — Validated `format_version: 1` export.
  2. `data/processed/pilot_v1/dataset_quality_provenance_report.json` — Safe counts, SHA-256 hash, PII redaction totals, parent-context breakdown, and validation summary (zero raw text).
  3. `data/annotations/pilot_v1/real_annotation_candidates.csv` — Unified 5-task human review sheet with **blank labels** plus `review_status` and `abstain_reason`.
  4. `data/annotations/pilot_v1/response_candidates.csv` + `review_manifest.jsonl` — Canonical v1.0 review manifest sidecar joined on unique `document_id` and verified by `validate_candidate_manifest_join()`.

### Step 2 — Inspect Quality Report & Confirm Authorization Gate

Before starting annotation, inspect `dataset_quality_provenance_report.json`. Once the responsible project authority approves the sanitized batch, update `metadata.authorization.authorization_status` to `"approved"` and verify the explicit gate in [backend/app/services/pilot.py](file:///D:/AI_Project/backend/app/services/pilot.py#L295-L326):

```python
from app.services.ingestion.static_export import load_static_export
from app.services.pilot import prepare_authorized_static_pilot

export = load_static_export("data/processed/pilot_v1/sanitized_static_export.json")
pilot = prepare_authorized_static_pilot(export, approval_confirmed=True)
assert pilot.summary["status"] == "ready_for_review"
```

### Step 3 — Human Annotation (No Model-Generated Labels)

Annotators label `data/annotations/pilot_v1/real_annotation_candidates.csv` strictly following [docs/ANNOTATION_GUIDELINES.md](file:///D:/AI_Project/docs/ANNOTATION_GUIDELINES.md):
- Set `review_status = labeled` only when confident across all applicable tasks.
- Set `review_status = uncertain` / `abstain` / `excluded` + controlled `abstain_reason` whenever context is missing or ambiguous (never write `unknown`/`none` into label cells).

### Step 4 — Validate Annotations, Route Adjudication & Check Training Readiness Gate

Run [backend/validate_pilot_annotations.py](file:///D:/AI_Project/backend/validate_pilot_annotations.py):

```powershell
.\backend\.venv\Scripts\python.exe backend\validate_pilot_annotations.py `
  --input data\annotations\pilot_v1\real_annotation_candidates.csv `
  --output-dir data\annotations\pilot_v1\validated `
  --min-comments 60 --min-threads 10 --min-per-class 3
```

- Routes all `uncertain`, `abstain`, and `excluded` rows to `adjudication_queue.csv` (never silently converted into normal labels).
- Writes the 5 canonical training CSVs (`sentiment.csv`, `emotion.csv`, `issue.csv`, `response.csv`, `solution.csv`) matching `data/annotations/templates/*.template.csv` **only** when zero schema errors exist.
- Reports `ready_for_training: false` with exact `readiness_blockers` until minimum labeled threads (`>= 10`), comments (`>= 60`), and per-class counts (`>= 3`) are met. **Do not retrain until `ready_for_training: true`.**

### Step 5 — Leakage-Free Real-Pilot Benchmarking (Once Annotated)

Once `ready_for_training: true`:
1. Split strictly by `group_by="post_id"` (70% train / 15% val / 15% test) so no two comments from the same thread cross splits.
2. Calibrate multi-label thresholds (`issue`, `emotion`) and select best MuRIL epoch **strictly on validation data** (never touch test data for tuning).
3. Save real-pilot models into an isolated artifact directory (e.g., `ml/artifacts/real_pilot_v1/`, gitignored by default) — **never** overwrite `ml/artifacts/demo_synthetic/`, `ml/artifacts/demo_synthetic_v2/`, or `ml/artifacts/registry.json`.

---

### Live-Verified Arctic Shift Collector (`backend/run_arctic_shift_pilot.py`)

Live verification against `https://arctic-shift.photon-reddit.com/api` confirmed Arctic Shift is **free, keyless, and operational** for direct and context communities (`r/IIMSambalpur`, `r/IIMSambalpurUG`, `r/MBAIndia`, `r/CATPrep`, `r/CATpreparation`), with two upstream behaviors now handled automatically by [arctic_shift.py](file:///D:/AI_Project/backend/app/services/ingestion/arctic_shift.py) and [run_arctic_shift_pilot.py](file:///D:/AI_Project/backend/run_arctic_shift_pilot.py):

1. **HTTP 422 Server Timeout / Rate Throttling:** Arctic Shift returns `HTTP 422 {"data": null, "error": "Timeout. Maybe slow down a bit"}` with `x-ratelimit-reset` when full-text `query=` scans a large subreddit or exceeds the sliding window. `ArcticShiftClient` now classifies this as a retryable `timeout` (`status=422`, leaking zero body text), records `x-ratelimit-reset` / `x-ratelimit-remaining`, and `run_arctic_shift_pilot.py` automatically falls back from `query=` to the faster `title=` index.
2. **Relative Crosspost URLs:** `ArcticShiftNormalizer._url()` normalizes valid relative `/r/<sub>/comments/<id>/...` paths to canonical `https://www.reddit.com/...` URLs while still rejecting malformed URLs.

Run a bounded, balanced multi-community live pilot (gitignored outputs, 0 DB writes by default):

```powershell
Set-Location D:\AI_Project
.\backend\.venv\Scripts\python.exe backend\run_arctic_shift_pilot.py `
  --max-posts 18 --max-posts-per-source 6 --max-comments 25 `
  --after 1year --delay 2.0 --max-retries 1 `
  --dataset-id real-arctic-shift-pilot-20261119-v1 `
  --output data\processed\arctic_shift_pilot_v1\sanitized_static_export.json `
  --export-candidates data\annotations\arctic_shift_pilot_v1
```

---

## 4. Sanitized Export Example & Validation Results

To demonstrate and verify the complete intake workflow without fabricating real Reddit ingestion, we ran `backend/import_reddit_export.py` on a clearly labeled **synthetic intake fixture** (`synthetic-intake-example-v1`) containing both browser thread JSON and Arctic Shift JSONL formats plus deliberate edge cases (usernames, emails, Indian phone numbers, `[deleted]` comment, missing `score`, duplicate comment ID, and an omitted parent comment).

**Generated Example Bundle under `docs/pilot_examples/` (Synthetic Fixture Only):**
- [sanitized_static_export_example.json](file:///D:/AI_Project/docs/pilot_examples/sanitized_static_export_example.json)
- [real_annotation_candidates.csv](file:///D:/AI_Project/docs/pilot_examples/real_annotation_candidates.csv) (blank labels + `review_status` + `abstain_reason`)
- [response_candidates.csv](file:///D:/AI_Project/docs/pilot_examples/response_candidates.csv)
- [review_manifest.jsonl](file:///D:/AI_Project/docs/pilot_examples/review_manifest.jsonl) (manifest v1.0, `status: unreviewed`, `dataset_approval: pending`, zero raw text)
- [dataset_quality_provenance_report_example.json](file:///D:/AI_Project/docs/pilot_examples/dataset_quality_provenance_report_example.json)

**Verified Example Report Highlights:**
- `pii_redactions`: `10` author metadata fields stripped, `3` emails redacted (`<EMAIL>`), `1` phone number redacted (`<PHONE>`, while `"CAT 2026"` stayed untouched), `2` username mentions redacted (`<USER_MENTION>`).
- `diagnostics_and_rejections`: `1` `[deleted]` comment rejected, `1` duplicate comment deduplicated, `1` missing `score` preserved unfabricated (`lossy_default_fields: ["comment.score"]`, `usable_for_pilot: false`), `1` unresolved parent flagged (`parent_context_status: unavailable`).
- `candidate_validation_valid: true`, `manifest_join_valid: true`.

---

## 5. Public Demo & Deployment Boundary Guarantee (`/api/demo`)

**Question:** *Can real-data ingestion, sanitization, annotation, and offline real-pilot benchmarking be completed without changing the existing `/api/demo` contract?*

**Answer: YES — with 100% isolation.**

1. [backend/app/api/demo.py](file:///D:/AI_Project/backend/app/api/demo.py#L20-L26) hard-pins `DATA_DIR = ROOT / "data" / "demo_synthetic"` and `ARTIFACT_DIR = ROOT / "ml" / "artifacts" / "demo_synthetic_v2"`.
2. Real-pilot raw exports (`data/raw/`), sanitized exports (`data/processed/`), annotation workbooks (`data/annotations/pilot_v1/`), and future real-pilot model artifacts (`ml/artifacts/real_pilot_v1/`) live in completely separate directories that are **gitignored** (`.gitignore` lines 190, 210–213) and **never read** by `/api/demo/*` or included in Vercel's `includeFiles` glob (`vercel.json`).
3. No real usernames, raw Reddit bodies, or unapproved pilot records are ever exposed through public FastAPI/Vercel endpoints.
4. Any future decision to surface aggregated, privacy-reviewed real-data insights in the web UI must be built on a separate authenticated/governed API namespace (never mutating the synthetic-only `/api/demo` contract).
