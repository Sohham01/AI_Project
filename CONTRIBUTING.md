# Contributing to Reddit Community Intelligence (RCI)

Thank you for contributing to RCI. Every change must uphold the project's scientific integrity, privacy boundaries, and test guarantees.

## Core Invariants

1. **Canonical 6-Class Response Taxonomy (Exact Names & Order):**
   `agreement`, `personal_experience`, `opinion`, `advice`, `solution`, `counterargument`.
   Never rename, merge, or reorder these classes across training, API responses, or UI views.
2. **Zero Cross-Split Thread Leakage & Zero Probe Contamination:**
   - All train/validation/test splits must be grouped by `post_id` and verified with `check_leakage()`.
   - Never tune hyperparameters, thresholds, temperature, or ensemble weights on the test split or the 54 unseen robustness probes.
3. **Never Fabricate Human Gold Labels or Resolution Quotes:**
   - Real Reddit comments must keep blank labels until reviewed by human annotators (with explicit `uncertain` / `abstain` workflows via `backend/run_annotation_studio.py`).
   - Proposed solutions (`advice` / `solution`) must remain strictly separated from explicit first-person follow-up resolution signals.
4. **Real-Corpus-Only Production Bundle & Automated Benchmark Verification:**
   - Keep `data/live_corpus/` and `ml/artifacts/live_corpus/` strictly free of synthetic demo bundles and PII; every artifact update must pass all 7 checks in `ml/evaluation/benchmark_verifier.py` and `backend/tests/test_live_corpus_integrity.py`.
5. **Privacy Hygiene:**
   - Never commit `.env`, raw exports (`data/processed/`), pilot annotation CSVs (`data/annotations/*pilot*/`), or unredacted usernames/PII.

## Local Development Workflow

```bash
# 1. Install base dependencies (includes pytest + httpx)
pip install -r backend/requirements.txt

# 2. Run the full offline test suite
pytest backend/ --ignore=backend/test_migrations_postgres.py -q

# 3. Verify Docker build & health
docker compose up -d --build
```

All pull requests targeting `main` automatically run `.github/workflows/ci.yml` (full backend `pytest` suite + 15-route Vercel entrypoint, 4-architecture benchmark verification, security headers, and zero-PII/secret guard).
