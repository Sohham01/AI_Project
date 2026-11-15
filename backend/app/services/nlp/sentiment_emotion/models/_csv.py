"""
Shared CSV helpers for the sentiment and emotion annotation files.

Kept in one place so the two annotation corpora in this stage parse, validate
and serialise exactly like the earlier ``issue_identification`` annotation
infrastructure: UTF-8 with BOM tolerance, ``|`` as the multi-value separator
inside a cell, blank lines skipped, unknown columns ignored, and a
``file:line`` prefix on every parse error.
"""

import csv
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

#: Separator used inside CSV cells for multi-valued fields.  A pipe cannot
#: collide with CSV comma separation, so ``a|b`` stays one cell.
MULTI_VALUE_SEPARATOR = "|"

_TRUTHY = {"1", "true", "yes", "y", "t"}


def split_multi(value: Optional[str]) -> List[str]:
    """Parse a pipe-separated cell into a clean list of non-empty values."""
    if not value:
        return []
    return [
        part.strip()
        for part in str(value).split(MULTI_VALUE_SEPARATOR)
        if part.strip()
    ]


def join_multi(values: Sequence[str]) -> str:
    return MULTI_VALUE_SEPARATOR.join(values)


def parse_bool(value: Any, default: bool = False) -> bool:
    """Interpret the usual truthy spellings used in annotation CSVs."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in _TRUTHY


def parse_optional_int(
    value: Any,
    field: str,
    document_id: Optional[str] = None,
) -> Optional[int]:
    """Parse an optional integer cell, raising a located error when malformed."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        subject = f"annotation {document_id!r} " if document_id else ""
        raise ValueError(
            f"{subject}has a non-integer {field} {value!r}"
        ) from exc


def resolve_annotation_path(path: str) -> str:
    """
    Accept a full path or a path relative to the repository root.

    ``from_csv("data/annotations/emotion_annotations.csv")`` therefore works
    from either the repository root or from ``backend/``.
    """
    if os.path.exists(path):
        return path
    candidate = os.path.join("..", path)
    if os.path.exists(candidate):
        return candidate
    return path


def read_annotation_rows(
    path: str,
    required_columns: Sequence[str],
    column_hint: str,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    Yield ``(line_number, row)`` pairs from an annotation CSV.

    Raises :class:`FileNotFoundError` naming the expected columns, and
    :class:`ValueError` prefixed with ``file:line`` for any malformed row, so
    an annotator can jump straight to the offending cell.
    """
    resolved = resolve_annotation_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"annotation CSV not found: {path!r}. Expected columns: "
            f"{column_hint}"
        )

    with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"annotation CSV {path!r} is empty (no header row)")
        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"annotation CSV {path!r} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            if all((value or "").strip() == "" for value in row.values() if value):
                continue  # skip fully blank lines
            yield line_number, row


def write_annotation_rows(
    path: str,
    columns: Sequence[str],
    rows: Iterator[Dict[str, str]],
) -> str:
    """Write annotation rows in the documented column order, creating parents."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
