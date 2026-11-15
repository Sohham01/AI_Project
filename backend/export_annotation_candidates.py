"""
Export collected Reddit comments into a human annotation-review CSV.

Reads real comments from PostgreSQL, keeps the original text, adds the existing
pipeline's ``normalized_text`` and language metadata, and leaves
``response_type`` blank for a human to fill in.  No model prediction is ever
written into the ground-truth column.

Usage::

    python export_annotation_candidates.py
    python export_annotation_candidates.py --output data/annotations/response_candidates.csv

After a human annotates ``response_type``, convert to the training CSV shape
(``text`` = normalized text) with::

    from app.services.annotation.candidates import (
        read_candidates_csv, to_training_rows,
    )
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

# Ensure imports work when run as ``python backend/export_annotation_candidates.py``
# from the repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
for _path in (_PROJECT_ROOT, _BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.services.annotation.candidates import (
    build_candidate_rows,
    candidate_context_warnings,
    load_comment_candidates,
    validate_candidate_rows,
    write_candidates_csv,
)

DEFAULT_OUTPUT = "data/annotations/response_candidates.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_annotation_candidates",
        description="Export DB comments as a blank response_type review CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output CSV path (default {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip the preprocessing pipeline; leave normalized_text empty",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="skip language analysis; leave language metadata at defaults",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from app.core.database import SessionLocal

    session = SessionLocal()
    try:
        raw = load_comment_candidates(session)
    finally:
        session.close()

    if not raw:
        print("No comments in the database; nothing to export.")
        return 2

    preprocessor = None
    if not args.no_normalize:
        from app.services.nlp.preprocessing.multilingual_preprocessor import (
            MultilingualTextPreprocessor,
        )

        preprocessor = MultilingualTextPreprocessor()

    language_service = None
    if not args.no_language:
        from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
        from app.services.nlp.language import SimpleLanguageDetector
        from app.services.nlp.language_analysis_service import (
            LanguageAnalysisService,
        )
        from app.services.nlp.script_detector import SimpleScriptDetector

        language_service = LanguageAnalysisService(
            language_detector=SimpleLanguageDetector(),
            script_detector=SimpleScriptDetector(),
            code_mix_detector=SimpleCodeMixDetector(),
        )

    rows = build_candidate_rows(
        raw,
        preprocessor=preprocessor,
        language_service=language_service,
    )

    problems = validate_candidate_rows(rows)
    if problems:
        print("Export validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    context_warnings = candidate_context_warnings(rows)
    if context_warnings:
        print("Review-context warnings (not silently repaired):")
        for warning in context_warnings:
            print(f"  - {warning}")

    write_candidates_csv(args.output, rows, validate=False)
    print(f"Wrote {len(rows)} candidates to {args.output}")
    print("response_type is blank - fill it in before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
