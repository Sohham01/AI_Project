# Security & Data Privacy Policy

## Privacy & Anonymization Guarantees

Reddit Community Intelligence (RCI) enforces strict privacy boundaries across its ingestion, storage, and public API layers:

- **Zero author attribution in public artifacts:** Author usernames (`author`, `author_fullname`) are stripped before sanitization and never committed to `data/live_corpus/` or returned by `/api/demo/*`.
- **Inline PII redaction:** Email addresses, phone numbers, and Reddit `u/` mentions are redacted to `<EMAIL>`, `<PHONE>`, and `<USER_MENTION>` and audited on every push/PR via `backend/tests/test_demo_api.py` and `.github/workflows/ci.yml`.
- **Raw & pilot export isolation:** Raw archives (`data/raw/`, `data/processed/`), local human annotation sheets (`data/annotations/*pilot*/`), `.env` secrets, and GPU checkpoints (`*.safetensors`, `*.pt`) are excluded from Git (`.gitignore`) and Docker (`.dockerignore`).
- **Read-only public studio:** The deployed FastAPI studio (`/api/demo/*`) makes zero outbound network requests to Reddit and requires no live credentials at request time.

## Application & Repository Security Controls

- **HTTP security headers (`backend/app/core/security.py` & `vercel.json`):** Every response from both Docker and Vercel emits `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`, `Cross-Origin-Opener-Policy: same-origin`, and a strict `Content-Security-Policy` (`frame-ancestors 'none'`, `object-src 'none'`).
- **Non-root container runtime:** `Dockerfile` runs Uvicorn under unprivileged `appuser` with an integrated `/health` probe; `docker-compose.yml` parameterizes PostgreSQL credentials via `${POSTGRES_USER}`, `${POSTGRES_PASSWORD}`, and `${POSTGRES_DB}`.
- **Local-only annotation workbench:** `backend/run_annotation_studio.py` binds exclusively to loopback (`127.0.0.1:8765`), writes CSV updates atomically, and uses pure DOM `textContent` rendering for corpus fields.
- **GitHub repository protection & supply-chain hygiene:**
  - `main` branch protection enforces the `Backend pytest, PII audit & Vercel API smoke check` CI check (`strict: true`) and blocks force pushes and branch deletion.
  - Dependabot vulnerability alerts, automated security updates, and weekly grouped updates (`.github/dependabot.yml`) are enabled.
  - CI runs with least-privilege `contents: read` permissions and verifies security headers, zero-PII compliance, zero tracked `.env`/key/token leaks, and zero orphaned gitlinks on every push and pull request.

## Reporting a Vulnerability or Privacy Concern

If you discover a security vulnerability, credential exposure, or unredacted personal information in any committed file or live endpoint:

1. **Do not** open a public issue with sensitive details or raw PII.
2. Report privately via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`) or contact the repository maintainer directly.
3. Include the affected file path or endpoint and steps to reproduce; confirmed issues are prioritized for immediate remediation and re-deployment.
