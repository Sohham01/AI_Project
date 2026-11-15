"""Convert reviewed candidate CSV rows into the response training schema.

This is an offline file-to-file operation. It does not initialize application
services, connect to PostgreSQL, or call external services.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from typing import Optional, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND_DIR = os.path.dirname(__file__)
for _path in (_PROJECT_ROOT, _BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.services.annotation.candidates import (
    CANDIDATE_COLUMNS,
    REQUIRED_CANDIDATE_COLUMNS,
    audit_training_candidate_rows,
    to_training_rows,
)
from app.services.nlp.response_classification.models.annotation import CSV_COLUMNS
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES

DEFAULT_INPUT = os.path.join("data", "annotations", "response_candidates.csv")
DEFAULT_OUTPUT = os.path.join("data", "annotations", "response_annotations.csv")


def convert_file(input_path: str, output_path: str, *, overwrite: bool = False) -> dict:
    """Validate and atomically convert a candidate CSV without touching input."""
    source = os.path.abspath(input_path)
    target = os.path.abspath(output_path)
    if source == target:
        raise ValueError("input and output paths must be different")
    if os.path.exists(target) and not overwrite:
        raise FileExistsError(f"output already exists: {target}; pass --overwrite to replace")

    with open(source, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, restkey="__extra_fields__", restval=None, strict=True)
        columns = list(reader.fieldnames or [])
        if len(columns) != len(set(columns)):
            raise ValueError("duplicate candidate CSV column names")
        missing = [name for name in REQUIRED_CANDIDATE_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"missing required candidate column(s): {', '.join(missing)}")
        rows = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            if row.get("__extra_fields__") is not None or any(
                row.get(name) is None for name in REQUIRED_CANDIDATE_COLUMNS
            ):
                raise ValueError(f"row width mismatch at line {line_number}")
            did = (row.get("document_id") or "").strip()
            if not did:
                raise ValueError(f"missing document_id at line {line_number}")
            if did in seen:
                raise ValueError(f"duplicate document_id {did!r} at line {line_number} (duplicate/conflicting annotation)")
            seen.add(did)
            rows.append(row)

    audit = audit_training_candidate_rows(rows)
    if audit["fatal"]:
        details = []
        for key in (
            "missing_document_ids",
            "duplicate_document_ids",
            "missing_comment_ids",
            "duplicate_comment_ids",
            "empty_original_text",
            "invalid_response_labels",
            "metadata_problems",
        ):
            if audit[key]:
                label = (
                    "invalid response_type"
                    if key == "invalid_response_labels"
                    else key
                )
                details.append(f"{label}={audit[key]}")
        raise ValueError("candidate validation failed: " + "; ".join(details))

    skipped_unlabelled = len(audit["blank_response_labels"])
    skipped_blank_text = len(audit["missing_normalized_text"])
    usable = [rows[index] for index in audit["accepted"]]
    converted = to_training_rows(usable)
    counts = Counter(row["response_type"] for row in converted)

    directory = os.path.dirname(target) or os.curdir
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".annotation-convert-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(converted)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_path, target)
        else:
            # Hard-link creation fails atomically if another process made target.
            os.link(temp_path, target)
            os.unlink(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return {
        "input_rows": len(rows),
        "written": len(converted),
        "skipped_unlabelled": skipped_unlabelled,
        "skipped_blank_normalized_text": skipped_blank_text,
        "excluded": audit["excluded"],
        "exclusion_report": {
            "blank_response_labels": skipped_unlabelled,
            "missing_normalized_text": skipped_blank_text,
        },
        "label_counts": {label: counts[label] for label in RESPONSE_TYPES if counts[label]},
        "output": target,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline candidate-to-training CSV conversion.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"candidate CSV (default {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"training CSV (default {DEFAULT_OUTPUT}; existing files are protected)")
    parser.add_argument("--overwrite", action="store_true", help="explicitly allow replacing the output CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = convert_file(args.input, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2
    print(f"Input rows: {report['input_rows']}")
    print(f"Written: {report['written']}")
    print(f"Skipped unlabelled: {report['skipped_unlabelled']}")
    print(f"Skipped blank normalized_text: {report['skipped_blank_normalized_text']}")
    print(f"Excluded: {report['excluded']}")
    print(f"Exclusion report: {report['exclusion_report']}")
    print(f"Label counts: {report['label_counts']}")
    print(f"Output: {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
