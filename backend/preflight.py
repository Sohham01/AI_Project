"""
Training-data readiness pre-flight.

Reports whether an annotation CSV is usable for training, whether
leakage-safe splitting is feasible, and whether MuRIL fine-tuning would even
be justified.  It never trains a model and never prints a performance number,
so a small or synthetic file can never be mistaken for a real result.

Usage::

    python preflight.py
    python preflight.py --csv data/annotations/response_annotations.csv
    python preflight.py --csv data/annotations/response_candidates.csv --json

Exit codes: ``0`` ready or warnings, ``2`` not ready / file missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

# Ensure imports work when run as ``python backend/preflight.py`` from the repo
# root (the ``ml`` package lives at the project root).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
for _path in (_PROJECT_ROOT, _BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ml.data.readiness import (
    MURIL_MIN_RECORDS_PER_CLASS,
    MURIL_MIN_TOTAL_RECORDS,
    analyse_annotation_csv,
    can_train_muril,
)

DEFAULT_CSV = "data/annotations/response_annotations.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight",
        description="Check annotation data readiness before training.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"annotation CSV to inspect (default {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--task",
        default="response",
        help="task schema to validate against (default response)",
    )
    parser.add_argument(
        "--group-by",
        default="post_id",
        help="column identifying a thread/group for leakage-safe splitting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    report = analyse_annotation_csv(
        args.csv,
        task=args.task,
        group_by=args.group_by,
    )
    eligible, blockers = can_train_muril(report)

    if args.json:
        payload = report.to_dict()
        payload["muril"] = {
            "eligible": eligible,
            "blockers": blockers,
            "min_total": MURIL_MIN_TOTAL_RECORDS,
            "min_per_class": MURIL_MIN_RECORDS_PER_CLASS,
        }
        print(json.dumps(payload, indent=2))
    else:
        for line in report.summary_lines():
            print(line)
        print()
        print(
            "MuRIL gate: "
            + ("PASS" if eligible else "BLOCKED")
            + f" (needs >= {MURIL_MIN_TOTAL_RECORDS} records and "
            f">= {MURIL_MIN_RECORDS_PER_CLASS} per class)"
        )
        if blockers:
            print("  blockers:")
            for blocker in blockers:
                print(f"    - {blocker}")
        print()
        print(
            "Reminder: synthetic or small-sample metrics are software checks, "
            "NOT real model performance. Do not report them as results."
        )

    return 0 if report.verdict in ("ready", "warn") else 2


if __name__ == "__main__":
    sys.exit(main())
