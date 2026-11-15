"""
Shared parsing and numeric-guarding helpers for Resolution Detection.

One job that matters more here than in the earlier stages: **every number that
reaches a resolution verdict must be finite and in range, or refused.**  A
resolution score is a published claim, and a ``NaN`` propagating into a
comparison does not produce an error - it produces a status chosen by whichever
branch happened to see it first.  So the guards here raise rather than
substitute, and the resolver additionally clamps its derived terms.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.services.nlp.resolution_detection.base import ResolutionStatus, parse_status

_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool_cell(value: Any, subject: str = "value") -> bool:
    """
    Parse a boolean annotation cell, refusing blanks and unrecognised tokens.

    Accepts real bools and the strings ``1/0``, ``true/false``, ``t/f``,
    ``yes/no``, ``y/n``, ``on/off`` in any case.  A blank is an error, never a
    ``False``: an undecided row that silently became "unresolved" would bias every
    metric computed over the corpus.
    """
    if value is None:
        raise ValueError(_blank_message(subject))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(_bad_bool_message(subject, value))
    text = str(value).strip()
    if not text:
        raise ValueError(_blank_message(subject))
    token = text.lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(_bad_bool_message(subject, value))


def _blank_message(subject: str) -> str:
    return (
        f"{subject} has no value. Every annotated row needs an explicit boolean; "
        "an undecided row is simply not annotated, and defaulting it to 'no' "
        "would silently turn a missing judgement into a negative label."
    )


def _bad_bool_message(subject: str, value: Any) -> str:
    accepted = ", ".join(sorted(_TRUE_TOKENS | _FALSE_TOKENS))
    return f"{subject} has value {value!r}, which is not a boolean. Accepted: {accepted}."


def check_finite(value: Any, subject: str) -> float:
    """
    Coerce to ``float`` and reject NaN and infinity.

    Not a stylistic check.  ``float("nan") < 0.5`` is False, so a NaN silently
    lands a value above *every* threshold; a NaN is a data bug that would
    otherwise become a confident verdict.
    """
    if isinstance(value, bool):
        raise ValueError(f"{subject} must be a number, got the boolean {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{subject} has value {value!r}, which is not a number.") from exc
    if math.isnan(number):
        raise ValueError(f"{subject} has value NaN, which is not a usable measurement.")
    if math.isinf(number):
        raise ValueError(f"{subject} has value {value}, which is infinite.")
    return number


def check_unit_interval(value: Any, subject: str) -> float:
    """Coerce to a finite float in [0, 1]."""
    number = check_finite(value, subject)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{subject} must be in [0.0, 1.0], got {number}.")
    return number


def check_count(value: Any, subject: str) -> int:
    """
    Coerce to a non-negative integer.

    Rejects negatives explicitly rather than clamping them to zero: a negative
    comment count is an upstream bug, and clamping would hide it while still
    producing a plausible-looking density.
    """
    if isinstance(value, bool):
        raise ValueError(f"{subject} must be an integer count, got the boolean {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{subject} must be a finite integer, got {value!r}")
        if not float(value).is_integer():
            raise ValueError(f"{subject} must be a whole number, got {value!r}")
        value = int(value)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{subject} has value {value!r}, which is not an integer.") from exc
    if number < 0:
        raise ValueError(f"{subject} must be non-negative, got {number}.")
    return number


def clamp_unit(value: float) -> float:
    """Clamp a finite float into [0, 1].  Used on derived terms, not on inputs."""
    return max(0.0, min(1.0, float(value)))


def normalise_count_mapping(
    mapping: Optional[Dict[str, Any]],
    subject: str = "response_counts",
) -> Dict[str, int]:
    """
    Coerce a count histogram to ``{str: int}`` with non-negative values, sorted.

    Blank role names are dropped rather than kept as an empty key, and negative
    counts are refused rather than clamped.
    """
    if not mapping:
        return {}
    out: Dict[str, int] = {}
    for key, value in mapping.items():
        name = str(key or "").strip()
        if not name:
            continue
        out[name] = check_count(value, f"{subject}[{name!r}]")
    return dict(sorted(out.items()))


def parse_gold_status(value: Any, subject: str = "gold_status") -> ResolutionStatus:
    """Parse a gold status cell.  Thin wrapper over :func:`parse_status`."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"{subject} has no value. An annotated issue needs one of 'resolved', "
            "'partially_resolved' or 'unresolved'; an undecided issue is simply "
            "not annotated."
        )
    return parse_status(value)


def normalise_status_sequence(
    values: Optional[Iterable[Any]], subject: str = "statuses"
) -> List[ResolutionStatus]:
    """Parse an iterable of statuses, preserving order and duplicates."""
    if not values:
        return []
    return [parse_gold_status(value, f"{subject}[{index}]") for index, value in enumerate(values)]


def status_key(status: Any) -> str:
    """The canonical string for a status, for dict keys and confusion matrices."""
    return parse_status(status).value


def confusion_cell(status: Any) -> Tuple[str, str]:
    """(gold, predicted) as canonical strings."""
    parsed = parse_status(status)
    return parsed.value, parsed.value


def stable_sorted_scores(scores: Optional[Sequence[Any]], subject: str) -> List[float]:
    """
    Validate a list of scores: finite, unit interval, returned in input order.

    Order is preserved rather than sorted because the caller may care about
    ranking; the resolver takes max/mean itself.
    """
    if not scores:
        return []
    return [check_unit_interval(score, f"{subject}[{index}]") for index, score in enumerate(scores)]
