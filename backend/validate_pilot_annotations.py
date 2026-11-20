"""Validate human-annotated pilot review CSVs, route uncertain/abstained items
to an adjudication queue, and emit canonical 5-task training CSVs only when
validation and minimum annotation readiness gates pass.

Never fabricates labels and never silently converts ``uncertain``/``abstain``/
``excluded`` items into normal classes (e.g. ``neutral``, ``not_solution``, or
empty multi-label sets).
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.solution_detection.models.labels import TARGET_LABELS

CANONICAL_SENTIMENTS = ("negative", "neutral", "positive")
CANONICAL_EMOTIONS = (
    "confidence",
    "curiosity",
    "frustration",
    "gratitude",
    "hope",
    "stress",
)
CANONICAL_ISSUES = (
    "academic_workload",
    "administrative_communication",
    "campus_infrastructure",
    "curriculum_teaching",
    "extracurriculars",
    "financial_concerns",
    "hostel_facilities",
    "placements",
)
CANONICAL_RESPONSES = tuple(RESPONSE_TYPES)
CANONICAL_SOLUTIONS = tuple(TARGET_LABELS)  # ("not_solution", "solution")

VALID_REVIEW_STATUSES = {"", "unreviewed", "labeled", "uncertain", "abstain", "excluded"}
FORBIDDEN_LABEL_TOKENS = {
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "unclear",
    "uncertain",
    "abstain",
    "not_applicable",
    "skip",
}
VALID_ABSTAIN_REASONS = {
    "insufficient_context",
    "missing_parent",
    "ambiguous_sarcasm",
    "mixed_or_borderline",
    "off_topic_noise",
    "untranslatable",
    "pii_risk",
    "other",
}


def _parse_multilabel_cell(
    raw: str,
    allowed: Sequence[str],
    *,
    field: str,
    doc_id: str,
    errors: List[Dict[str, Any]],
) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = [p.strip().lower() for p in text.split("|")]
    if any(not p for p in parts):
        errors.append({"document_id": doc_id, "field": field, "code": "malformed_pipe_delimiter"})
        return []
    allowed_set = set(allowed)
    cleaned: List[str] = []
    for part in parts:
        if part in FORBIDDEN_LABEL_TOKENS:
            errors.append(
                {
                    "document_id": doc_id,
                    "field": field,
                    "code": "forbidden_placeholder_in_label_cell",
                    "hint": "Use review_status='uncertain' or 'abstain' instead of writing placeholders into label cells.",
                }
            )
            continue
        if part not in allowed_set:
            errors.append({"document_id": doc_id, "field": field, "code": f"unknown_label:{part}"})
            continue
        if part not in cleaned:
            cleaned.append(part)
    return sorted(cleaned)


def _parse_single_cell(
    raw: str,
    allowed: Sequence[str],
    *,
    field: str,
    doc_id: str,
    errors: List[Dict[str, Any]],
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    val = (raw or "").strip().lower()
    if not val:
        errors.append({"document_id": doc_id, "field": field, "code": "missing_required_label"})
        return None
    if "|" in val:
        errors.append({"document_id": doc_id, "field": field, "code": "pipe_forbidden_in_single_label"})
        return None
    if val in FORBIDDEN_LABEL_TOKENS:
        errors.append(
            {
                "document_id": doc_id,
                "field": field,
                "code": "forbidden_placeholder_in_label_cell",
                "hint": "Use review_status='uncertain' or 'abstain' instead of writing placeholders into label cells.",
            }
        )
        return None
    if aliases and val in aliases:
        val = aliases[val]
    if val not in set(allowed):
        errors.append({"document_id": doc_id, "field": field, "code": f"unknown_label:{val}"})
        return None
    return val


def validate_and_split_pilot_csv(
    csv_path: Path,
    *,
    out_dir: Optional[Path] = None,
    min_labeled_comments: int = 60,
    min_labeled_threads: int = 10,
    min_examples_per_class: int = 3,
) -> Dict[str, Any]:
    """Validate a unified 5-task human review CSV and optionally emit clean task CSVs."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    errors: List[Dict[str, Any]] = []
    status_counts: Counter = Counter()
    abstain_reasons: Counter = Counter()
    seen_doc_ids: set[str] = set()

    labeled_rows: List[Dict[str, Any]] = []
    adjudication_rows: List[Dict[str, str]] = []
    unreviewed_count = 0

    task_counts: Dict[str, Counter] = {
        "sentiment": Counter(),
        "emotion": Counter(),
        "issue": Counter(),
        "response": Counter(),
        "solution": Counter(),
    }

    for idx, row in enumerate(rows, start=2):
        doc_id = (row.get("document_id") or "").strip()
        if not doc_id:
            errors.append({"row": idx, "field": "document_id", "code": "missing_document_id"})
            continue
        if doc_id in seen_doc_ids:
            errors.append({"document_id": doc_id, "field": "document_id", "code": "duplicate_document_id"})
            continue
        seen_doc_ids.add(doc_id)

        status = (row.get("review_status") or "").strip().lower()
        if status not in VALID_REVIEW_STATUSES:
            errors.append(
                {
                    "document_id": doc_id,
                    "field": "review_status",
                    "code": f"invalid_review_status:{status}",
                }
            )
            continue

        if status in {"", "unreviewed"}:
            status_counts["unreviewed"] += 1
            unreviewed_count += 1
            # Guard: warn if labels were filled without setting review_status='labeled'
            any_filled = any(
                (row.get(col) or "").strip()
                for col in ("sentiment", "emotions", "issues", "response_type", "is_solution")
            )
            if any_filled:
                errors.append(
                    {
                        "document_id": doc_id,
                        "field": "review_status",
                        "code": "labels_present_without_labeled_status",
                    }
                )
            continue

        if status in {"uncertain", "abstain", "excluded"}:
            status_counts[status] += 1
            reason = (row.get("abstain_reason") or "").strip().lower()
            if not reason:
                errors.append(
                    {
                        "document_id": doc_id,
                        "field": "abstain_reason",
                        "code": "abstain_or_uncertain_requires_reason",
                    }
                )
            elif reason not in VALID_ABSTAIN_REASONS:
                errors.append(
                    {
                        "document_id": doc_id,
                        "field": "abstain_reason",
                        "code": f"unknown_abstain_reason:{reason}",
                    }
                )
            else:
                abstain_reasons[reason] += 1
            # Routed exclusively to adjudication queue — NEVER converted into a training label!
            adjudication_rows.append(dict(row))
            continue

        # status == "labeled"
        status_counts["labeled"] += 1
        norm_text = (row.get("normalized_text") or "").strip()
        if not norm_text:
            errors.append({"document_id": doc_id, "field": "normalized_text", "code": "blank_normalized_text"})
            continue

        row_errors_before = len(errors)
        sentiment = _parse_single_cell(
            row.get("sentiment", ""),
            CANONICAL_SENTIMENTS,
            field="sentiment",
            doc_id=doc_id,
            errors=errors,
        )
        emotions = _parse_multilabel_cell(
            row.get("emotions", ""),
            CANONICAL_EMOTIONS,
            field="emotions",
            doc_id=doc_id,
            errors=errors,
        )
        issues = _parse_multilabel_cell(
            row.get("issues", ""),
            CANONICAL_ISSUES,
            field="issues",
            doc_id=doc_id,
            errors=errors,
        )
        response_type = _parse_single_cell(
            row.get("response_type", ""),
            CANONICAL_RESPONSES,
            field="response_type",
            doc_id=doc_id,
            errors=errors,
        )
        solution_label = _parse_single_cell(
            row.get("is_solution", ""),
            CANONICAL_SOLUTIONS,
            field="is_solution",
            doc_id=doc_id,
            errors=errors,
            aliases={"true": "solution", "false": "not_solution"},
        )

        if len(errors) == row_errors_before:
            task_counts["sentiment"][sentiment] += 1
            for e in emotions:
                task_counts["emotion"][e] += 1
            if not emotions:
                task_counts["emotion"]["<empty_set>"] += 1
            for iss in issues:
                task_counts["issue"][iss] += 1
            if not issues:
                task_counts["issue"]["<empty_set>"] += 1
            task_counts["response"][response_type] += 1
            task_counts["solution"][solution_label] += 1
            labeled_rows.append(
                {
                    "document_id": doc_id,
                    "text": norm_text,
                    "post_id": (row.get("post_id") or "").strip(),
                    "comment_id": (row.get("comment_id") or "").strip(),
                    "language": (row.get("language") or "unknown").strip() or "unknown",
                    "script": (row.get("script") or "unknown").strip() or "unknown",
                    "is_code_mixed": (row.get("is_code_mixed") or "false").strip().lower(),
                    "detected_languages": (row.get("detected_languages") or "").strip(),
                    "sentiment": sentiment,
                    "emotions": "|".join(emotions),
                    "issues": "|".join(issues),
                    "response_type": response_type,
                    "is_solution_bool": "true" if solution_label == "solution" else "false",
                    "solution_label": solution_label,
                    "annotator_notes": (row.get("annotator_notes") or "").strip(),
                }
            )

    labeled_threads = len({r["post_id"] for r in labeled_rows if r["post_id"]})
    readiness_blockers: List[str] = []
    if errors:
        readiness_blockers.append(f"schema_validation_errors:{len(errors)}")
    if len(labeled_rows) < min_labeled_comments:
        readiness_blockers.append(
            f"insufficient_labeled_comments:{len(labeled_rows)}<{min_labeled_comments}"
        )
    if labeled_threads < min_labeled_threads:
        readiness_blockers.append(
            f"insufficient_labeled_threads:{labeled_threads}<{min_labeled_threads}"
        )
    for task_name, req_labels in (
        ("sentiment", CANONICAL_SENTIMENTS),
        ("response", CANONICAL_RESPONSES),
        ("solution", CANONICAL_SOLUTIONS),
    ):
        under = [
            lbl for lbl in req_labels if task_counts[task_name].get(lbl, 0) < min_examples_per_class
        ]
        if under:
            readiness_blockers.append(
                f"{task_name}_underrepresented_classes:{','.join(under)}"
            )

    written_files: Dict[str, str] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Always write adjudication queue when uncertain/abstain/excluded rows exist
        if adjudication_rows:
            adj_path = out_dir / "adjudication_queue.csv"
            with adj_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(adjudication_rows[0].keys()))
                writer.writeheader()
                writer.writerows(adjudication_rows)
            written_files["adjudication_queue"] = str(adj_path)

        # Write canonical 5-task training CSVs only when zero schema errors and >= 1 labeled row
        if not errors and labeled_rows:
            specs = [
                (
                    "sentiment",
                    ["document_id", "text", "sentiment", "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages"],
                    lambda r: {
                        "document_id": r["document_id"],
                        "text": r["text"],
                        "sentiment": r["sentiment"],
                        "post_id": r["post_id"],
                        "comment_id": r["comment_id"],
                        "language": r["language"],
                        "script": r["script"],
                        "is_code_mixed": r["is_code_mixed"],
                        "detected_languages": r["detected_languages"],
                    },
                ),
                (
                    "emotion",
                    ["document_id", "text", "emotions", "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages"],
                    lambda r: {
                        "document_id": r["document_id"],
                        "text": r["text"],
                        "emotions": r["emotions"],
                        "post_id": r["post_id"],
                        "comment_id": r["comment_id"],
                        "language": r["language"],
                        "script": r["script"],
                        "is_code_mixed": r["is_code_mixed"],
                        "detected_languages": r["detected_languages"],
                    },
                ),
                (
                    "issue",
                    ["document_id", "text", "labels", "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages"],
                    lambda r: {
                        "document_id": r["document_id"],
                        "text": r["text"],
                        "labels": r["issues"],
                        "post_id": r["post_id"],
                        "comment_id": r["comment_id"],
                        "language": r["language"],
                        "script": r["script"],
                        "is_code_mixed": r["is_code_mixed"],
                        "detected_languages": r["detected_languages"],
                    },
                ),
                (
                    "response",
                    ["document_id", "text", "response_type", "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages"],
                    lambda r: {
                        "document_id": r["document_id"],
                        "text": r["text"],
                        "response_type": r["response_type"],
                        "post_id": r["post_id"],
                        "comment_id": r["comment_id"],
                        "language": r["language"],
                        "script": r["script"],
                        "is_code_mixed": r["is_code_mixed"],
                        "detected_languages": r["detected_languages"],
                    },
                ),
                (
                    "solution",
                    ["document_id", "text", "is_solution", "solution_type", "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages", "response_type", "rationale", "evidence"],
                    lambda r: {
                        "document_id": r["document_id"],
                        "text": r["text"],
                        "is_solution": r["is_solution_bool"],
                        "solution_type": "",
                        "post_id": r["post_id"],
                        "comment_id": r["comment_id"],
                        "language": r["language"],
                        "script": r["script"],
                        "is_code_mixed": r["is_code_mixed"],
                        "detected_languages": r["detected_languages"],
                        "response_type": r["response_type"],
                        "rationale": r["annotator_notes"],
                        "evidence": "",
                    },
                ),
            ]
            for task_key, cols, mapper in specs:
                out_csv = out_dir / f"{task_key}.csv"
                with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=cols)
                    writer.writeheader()
                    for rec in labeled_rows:
                        writer.writerow(mapper(rec))
                written_files[task_key] = str(out_csv)

    return {
        "input_csv": str(csv_path),
        "total_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "abstain_reasons": dict(sorted(abstain_reasons.items())),
        "labeled_comments": len(labeled_rows),
        "labeled_threads": labeled_threads,
        "adjudication_queue_count": len(adjudication_rows),
        "unreviewed_count": unreviewed_count,
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:50],
        "task_label_distributions": {
            k: dict(sorted(v.items())) for k, v in task_counts.items()
        },
        "ready_for_training": len(readiness_blockers) == 0,
        "readiness_blockers": readiness_blockers,
        "written_files": written_files,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Human-annotated unified review CSV")
    parser.add_argument("--output-dir", default=None, help="Directory to write adjudication queue and canonical task CSVs")
    parser.add_argument("--min-comments", type=int, default=60)
    parser.add_argument("--min-threads", type=int, default=10)
    parser.add_argument("--min-per-class", type=int, default=3)
    args = parser.parse_args(argv)

    summary = validate_and_split_pilot_csv(
        Path(args.input),
        out_dir=Path(args.output_dir) if args.output_dir else None,
        min_labeled_comments=args.min_comments,
        min_labeled_threads=args.min_threads,
        min_examples_per_class=args.min_per_class,
    )
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
