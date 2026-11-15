"""
Training-data readiness pre-flight checks.

Before a model is trained the annotation file has to be *inspected*, because
the failure mode this project actually hit is training a six-class model on a
handful of synthetic rows and then quoting the resulting numbers as if they
were real.  This module answers, from the CSV alone:

* how many records are there, and how many per class;
* are required columns present; are there duplicate ids/texts or empty bodies;
* how many threads/groups exist, and is leakage-safe (group-aware) splitting
  even possible;
* does the data meet the bar for real training - and specifically for MuRIL,
  which needs far more genuine annotations, not fewer.

It never trains anything and never reports a performance number: a small or
synthetic file yields ``warn``/``not_ready``, never a glowing "ready".
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Below this many records, any metric is noise rather than a result.
MIN_TOTAL_RECORDS = 60
#: Below this many records in any present class, the class is not learnable.
MIN_RECORDS_PER_CLASS = 6
#: MuRIL is a large pretrained model: it needs many more genuine examples and
#: every class present, or fine-tuning it is wasted compute.
MURIL_MIN_TOTAL_RECORDS = 300
MURIL_MIN_RECORDS_PER_CLASS = 30
#: Group-aware splitting needs at least this many distinct groups.
MIN_GROUPS_FOR_GROUP_SPLIT = 3


@dataclass(frozen=True)
class TaskSchema:
    """Which columns a task's annotation CSV must provide."""

    required: Tuple[str, ...]
    label_column: str
    single_label: bool = True


#: Known schemas.  Only ``response`` is wired here because it is the V1 focus;
#: other tasks can be added without changing the analysis logic.
DEFAULT_TASK_SCHEMAS: Dict[str, TaskSchema] = {
    "response": TaskSchema(
        required=("document_id", "text", "response_type"),
        label_column="response_type",
        single_label=True,
    ),
}


def _response_label_space() -> Tuple[str, ...]:
    """The six V1 response types, imported lazily to keep ``ml`` standalone."""
    from app.services.nlp.response_classification.models.labels import (
        RESPONSE_TYPES,
    )

    return tuple(RESPONSE_TYPES)


@dataclass
class ReadinessReport:
    path: str
    task: str
    label_column: str
    total_records: int = 0
    missing_columns: List[str] = field(default_factory=list)
    duplicate_document_ids: List[str] = field(default_factory=list)
    duplicate_texts: int = 0
    empty_content: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    unknown_labels: List[str] = field(default_factory=list)
    missing_classes: List[str] = field(default_factory=list)
    thread_count: int = 0
    multi_comment_threads: int = 0
    ungrouped_records: int = 0
    split_feasible: bool = False
    split_notes: List[str] = field(default_factory=list)
    muril_eligible: bool = False
    muril_blockers: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verdict: str = "not_ready"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "task": self.task,
            "label_column": self.label_column,
            "total_records": self.total_records,
            "missing_columns": list(self.missing_columns),
            "duplicate_document_ids": list(self.duplicate_document_ids),
            "duplicate_texts": self.duplicate_texts,
            "empty_content": self.empty_content,
            "label_counts": dict(self.label_counts),
            "unknown_labels": list(self.unknown_labels),
            "missing_classes": list(self.missing_classes),
            "thread_count": self.thread_count,
            "multi_comment_threads": self.multi_comment_threads,
            "ungrouped_records": self.ungrouped_records,
            "split_feasible": self.split_feasible,
            "split_notes": list(self.split_notes),
            "muril_eligible": self.muril_eligible,
            "muril_blockers": list(self.muril_blockers),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "verdict": self.verdict,
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"File: {self.path}",
            f"Task: {self.task} (label column: {self.label_column})",
            f"Verdict: {self.verdict.upper()}",
            f"Records: {self.total_records}",
            f"Missing columns: {self.missing_columns or 'none'}",
            f"Duplicate document_ids: {len(self.duplicate_document_ids)}",
            f"Duplicate texts: {self.duplicate_texts}",
            f"Empty content: {self.empty_content}",
            f"Label counts: {self.label_counts}",
            f"Missing classes: {self.missing_classes or 'none'}",
            f"Unknown labels: {self.unknown_labels or 'none'}",
            (
                f"Threads: {self.thread_count} "
                f"({self.multi_comment_threads} with >1 comment; "
                f"{self.ungrouped_records} ungrouped)"
            ),
            f"Leakage-safe split feasible: {self.split_feasible}",
        ]
        lines.extend(f"  split: {note}" for note in self.split_notes)
        lines.append(
            f"MuRIL eligible: {self.muril_eligible}"
            + (f" ({'; '.join(self.muril_blockers)})" if self.muril_blockers else "")
        )
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return lines


def _read_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, restkey="__extra_fields__", restval=None, strict=True)
        fieldnames = list(reader.fieldnames or [])
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("duplicate CSV column names")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if row.get("__extra_fields__") is not None or any(
                value is None for key, value in row.items() if key != "__extra_fields__"
            ):
                raise ValueError(f"malformed CSV row width at line {line_number}")
            rows.append({key: value for key, value in row.items() if key != "__extra_fields__"})
    return fieldnames, rows


def analyse_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    schema: TaskSchema,
    label_space: Sequence[str],
    path: str = "<records>",
    task: str = "response",
    group_by: str = "post_id",
) -> ReadinessReport:
    """Analyse already-loaded rows and build a :class:`ReadinessReport`.

    Pure and synchronous: everything the preflight reports comes from these
    rows, so it can be unit-tested without a database or network.
    """
    report = ReadinessReport(
        path=path,
        task=task,
        label_column=schema.label_column,
    )

    report.missing_columns = [
        col for col in schema.required if col not in set(fieldnames)
    ]
    if report.missing_columns:
        report.errors.append(
            f"missing required column(s): {', '.join(report.missing_columns)}"
        )
        report.verdict = "not_ready"
        return report

    label_space_set = set(label_space)
    label_counts: Counter = Counter()
    seen_document_ids: set[str] = set()
    duplicate_ids: List[str] = []
    seen_texts: set[str] = set()
    group_members: Dict[str, int] = defaultdict(int)

    for row in rows:
        document_id = str(row.get("document_id") or "").strip()
        text = str(row.get("text") or "").strip()
        label = str(row.get(schema.label_column) or "").strip()

        if document_id:
            if document_id in seen_document_ids and document_id not in duplicate_ids:
                duplicate_ids.append(document_id)
            seen_document_ids.add(document_id)

        if not text:
            report.empty_content += 1
        else:
            if text in seen_texts:
                report.duplicate_texts += 1
            seen_texts.add(text)

        if label:
            label_counts[label] += 1
            if label not in label_space_set:
                report.unknown_labels.append(label)

        group_value = str(row.get(group_by) or "").strip()
        if group_value:
            group_members[group_value] += 1
        else:
            report.ungrouped_records += 1

    report.total_records = len(rows)
    report.duplicate_document_ids = duplicate_ids
    # Present classes in canonical order, unknown labels appended.
    present = [name for name in label_space if name in label_counts]
    present.extend(sorted(set(label_counts) - label_space_set))
    report.label_counts = {name: label_counts[name] for name in present}
    report.missing_classes = [
        name for name in label_space if label_counts.get(name, 0) == 0
    ]
    report.unknown_labels = sorted(set(report.unknown_labels))

    report.thread_count = len(group_members)
    report.multi_comment_threads = sum(
        1 for count in group_members.values() if count > 1
    )

    _evaluate_split(report, group_by)
    _evaluate_muril(report, label_space)
    _evaluate_verdict(report)
    return report


def _evaluate_split(report: ReadinessReport, group_by: str) -> None:
    """Decide whether leakage-safe splitting is feasible."""
    if report.total_records < 3:
        report.split_feasible = False
        report.split_notes.append(
            "fewer than 3 records: train/validation/test cannot all be populated"
        )
        return

    if report.thread_count and (
        report.thread_count >= MIN_GROUPS_FOR_GROUP_SPLIT
    ):
        report.split_feasible = True
        report.split_notes.append(
            f"group-aware split on {group_by!r}: {report.thread_count} groups "
            "(whole threads stay together)"
        )
        if report.multi_comment_threads == 0:
            report.split_notes.append(
                "no thread has more than one comment - grouping has no effect "
                "yet, record-level splitting would be equivalent"
            )
        if report.ungrouped_records:
            report.split_notes.append(
                f"{report.ungrouped_records} record(s) lack {group_by!r}; they "
                "fall back to their own document identity"
            )
    else:
        report.split_feasible = True
        report.split_notes.append(
            f"record-level split only: {report.thread_count} group(s) present "
            f"(group-aware split needs >= {MIN_GROUPS_FOR_GROUP_SPLIT})"
        )
        report.warnings.append(
            "house thread context is thin; comments from one thread may "
            "straddle train/test (leakage risk)"
        )


def _evaluate_muril(
    report: ReadinessReport, label_space: Sequence[str]
) -> None:
    """MuRIL requires far more genuine data than the classical baselines."""
    blockers: List[str] = []
    if report.total_records < MURIL_MIN_TOTAL_RECORDS:
        blockers.append(
            f"{report.total_records} records < required {MURIL_MIN_TOTAL_RECORDS}"
        )
    for name in label_space:
        count = report.label_counts.get(name, 0)
        if count < MURIL_MIN_RECORDS_PER_CLASS:
            blockers.append(
                f"class {name!r} has {count} < required "
                f"{MURIL_MIN_RECORDS_PER_CLASS}"
            )
    report.muril_blockers = blockers
    report.muril_eligible = not blockers


def _evaluate_verdict(report: ReadinessReport) -> None:
    hard_fail = (
        report.total_records == 0
        or bool(report.missing_columns)
        or len(report.label_counts) < 2
        or bool(report.unknown_labels)
    )
    if hard_fail:
        if report.total_records == 0:
            report.errors.append("no records found")
        if len(report.label_counts) < 2:
            report.errors.append(
                "fewer than two classes present: not a classification problem"
            )
        if report.unknown_labels:
            report.errors.append(
                f"labels outside the taxonomy: {report.unknown_labels}"
            )
        report.verdict = "not_ready"
        return

    warnings: List[str] = []
    if report.total_records < MIN_TOTAL_RECORDS:
        warnings.append(
            f"only {report.total_records} records (< {MIN_TOTAL_RECORDS}); "
            "any metric computed from this file is NOT a real performance "
            "result"
        )
    sparse = [
        name
        for name, count in report.label_counts.items()
        if count < MIN_RECORDS_PER_CLASS
    ]
    if sparse:
        warnings.append(
            f"class(es) below {MIN_RECORDS_PER_CLASS} examples: {sparse}"
        )
    if report.missing_classes:
        warnings.append(
            f"missing class(es): {report.missing_classes}"
        )
    if report.duplicate_document_ids:
        warnings.append(
            f"{len(report.duplicate_document_ids)} duplicate document_id(s)"
        )
    if report.duplicate_texts:
        warnings.append(f"{report.duplicate_texts} duplicate text(s)")
    if report.empty_content:
        warnings.append(f"{report.empty_content} empty content row(s)")
    if not report.split_feasible:
        warnings.append("leakage-safe splitting is not feasible")

    report.warnings = list(report.warnings) + warnings
    report.verdict = "warn" if warnings else "ready"


def analyse_annotation_csv(
    path: str,
    *,
    task: str = "response",
    label_space: Optional[Sequence[str]] = None,
    group_by: str = "post_id",
) -> ReadinessReport:
    """Analyse an annotation CSV from disk."""
    schema = DEFAULT_TASK_SCHEMAS.get(task)
    if schema is None:
        raise ValueError(
            f"no readiness schema for task {task!r}; known: "
            f"{sorted(DEFAULT_TASK_SCHEMAS)}"
        )
    if label_space is None:
        label_space = _response_label_space()

    if not os.path.exists(path):
        report = ReadinessReport(path=path, task=task, label_column=schema.label_column)
        report.errors.append(f"annotation CSV not found: {path!r}")
        report.verdict = "not_ready"
        return report

    try:
        fieldnames, rows = _read_rows(path)
    except (csv.Error, ValueError) as exc:
        report = ReadinessReport(path=path, task=task, label_column=schema.label_column)
        report.errors.append(f"malformed annotation CSV: {exc}")
        report.verdict = "not_ready"
        return report
    return analyse_records(
        rows,
        fieldnames=fieldnames,
        schema=schema,
        label_space=label_space,
        path=path,
        task=task,
        group_by=group_by,
    )


def can_train_muril(report: ReadinessReport) -> Tuple[bool, List[str]]:
    """Return ``(eligible, blockers)`` for MuRIL fine-tuning.

    Mirrors :attr:`ReadinessReport.muril_eligible` but kept as a function so
    callers can gate on it explicitly and surface the blockers.
    """
    return report.muril_eligible, list(report.muril_blockers)


__all__ = [
    "TaskSchema",
    "DEFAULT_TASK_SCHEMAS",
    "ReadinessReport",
    "analyse_records",
    "analyse_annotation_csv",
    "can_train_muril",
    "MIN_TOTAL_RECORDS",
    "MIN_RECORDS_PER_CLASS",
    "MURIL_MIN_TOTAL_RECORDS",
    "MURIL_MIN_RECORDS_PER_CLASS",
    "MIN_GROUPS_FOR_GROUP_SPLIT",
]
