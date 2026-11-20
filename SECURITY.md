# Security & Data Privacy Policy

## Privacy & Anonymization Guarantees

Reddit Community Intelligence (RCI) enforces strict privacy boundaries across its ingestion, storage, and public API layers:

- **Zero author attribution in public artifacts:** Author usernames (`author`, `author_fullname`) are stripped before sanitization and never committed to `data/live_corpus/` or returned by `/api/demo/*`.
- **Inline PII redaction:** Email addresses, phone numbers, and Reddit `u/` mentions are redacted to `<EMAIL>`, `<PHONE>`, and `<USER_MENTION>` and continuously audited by automated tests (`backend/tests/test_demo_api.py` and `.github/workflows/ci.yml`).
- **Raw & pilot export isolation:** Raw archives (`data/raw/`, `data/processed/`), local human annotation sheets (`data/annotations/*pilot*/`), `.env` secrets, and GPU checkpoints (`*.safetensors`, `*.pt`) are excluded from Git (`.gitignore`) and Docker (`.dockerignore`).
- **Read-only public studio:** The deployed FastAPI studio (`/api/demo/*`) performs zero outbound network requests to Reddit and requires no live credentials at request time.

## Reporting a Vulnerability or Privacy Concern

If you discover a security vulnerability, credential exposure, or unredacted personal information in any committed file or live endpoint:

1. **Do not** open a public issue with sensitive details or raw PII.
2. Report privately via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`) or contact the repository maintainer directly.
3. Include the affected file path or endpoint and steps to reproduce; confirmed issues are prioritized for immediate remediation and re-deployment.
