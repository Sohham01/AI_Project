"""
Shared parsing and label-normalisation helpers.

Two jobs, both of which every consumer needs and none of which should be
re-implemented per module:

1. **Strict boolean parsing for annotation CSVs.**  A blank or unrecognised
   ``linked`` / ``is_solution`` cell is an *error*, not a ``False``.  This is the
   single most consequential parsing decision in the project: a dataset where a
   quarter of the judgements silently defaulted to "no" produces a precision
   figure that looks like a measurement and is not one.  Failing loudly at load
   time costs one afternoon; being wrong in a published number costs a system.
2. **Canonical label ordering.**  Labels are trimmed, blanks dropped,
   de-duplicated and sorted, so the same evidence always produces the same link
   ordering regardless of the order an upstream stage happened to emit them in.

Kept separate from :mod:`base` on purpose: this is validation plumbing, not the
contract between the linker and its callers.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool_cell(value: Any, subject: str = "value") -> bool:
    """
    Parse a boolean annotation cell, refusing blanks and unrecognised tokens.

    Accepts real bools, and the strings ``1/0``, ``true/false``, ``t/f``,
    ``yes/no``, ``y/n``, ``on/off`` in any case.  Everything else raises.
    """
    if value is None:
        raise ValueError(_blank_message(subject))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(_bad_message(subject, value))
    text = str(value).strip()
    if not text:
        raise ValueError(_blank_message(subject))
    token = text.lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(_bad_message(subject, value))


def _blank_message(subject: str) -> str:
    return (
        f"{subject} has no value. Every annotated row needs an explicit boolean; "
        "an undecided row is simply not annotated, and defaulting it to 'no' "
        "would silently turn a missing judgement into a negative label."
    )


def _bad_message(subject: str, value: Any) -> str:
    accepted = ", ".join(sorted(_TRUE_TOKENS | _FALSE_TOKENS))
    return f"{subject} has value {value!r}, which is not a boolean. Accepted: {accepted}."


def parse_float_cell(
    value: Any,
    subject: str = "value",
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Parse a float cell, refusing blanks, non-numbers and out-of-range values."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"{subject} has no value. Expected a number"
            + (f" in [{minimum}, {maximum}]" if minimum is not None and maximum is not None else "")
            + "."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{subject} has value {value!r}, which is not a number.") from exc
    if number != number:  # NaN
        raise ValueError(f"{subject} has value {value!r}, which is not a number (NaN).")
    if minimum is not None and number < minimum:
        raise ValueError(f"{subject} has value {number}, which is below {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{subject} has value {number}, which is above {maximum}.")
    return number


def normalise_labels(labels: Optional[Iterable[Any]]) -> List[str]:
    """
    Trimmed, de-duplicated, alphabetically ordered labels.

    Sorted rather than first-seen so two runs over identical data produce
    byte-identical output regardless of upstream ordering; de-duplicated so one
    label repeated by an upstream stage cannot become two identical link
    candidates.
    """
    if not labels:
        return []
    seen: Dict[str, None] = {}
    for label in labels:
        text = str(label).strip() if label is not None else ""
        if text:
            seen.setdefault(text, None)
    return sorted(seen)


def normalise_probabilities(
    probabilities: Optional[Dict[str, Any]],
    minimum: float = 0.0,
    maximum: float = 1.0,
    subject: str = "probabilities",
) -> Dict[str, float]:
    """
    Coerce a probability mapping to ``{str: float}``, sorted by key.

    Values are validated into range and coerced, because a probability that
    arrives as ``"0.82"`` is fine and one that arrives as ``"high"`` is a
    wiring bug that should not reach a score.
    """
    if not probabilities:
        return {}
    out: Dict[str, float] = {}
    for key, value in probabilities.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = parse_float_cell(
            value, subject=f"{subject}[{name!r}]", minimum=minimum, maximum=maximum
        )
    return dict(sorted(out.items()))


def stable_unique(values: Sequence[Any]) -> List[str]:
    """Trimmed, de-duplicated ids in **input** order (not sorted)."""
    seen: Dict[str, None] = {}
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            seen.setdefault(text, None)
    return list(seen)


def clean_optional_text(value: Any) -> str:
    """Trim any value to a string, mapping ``None``/blank to ``""``."""
    if value is None:
        return ""
    return str(value).strip()


def strip_placeholder_tokens(text: Optional[str]) -> str:
    """
    Remove upstream preprocessing placeholders from a string, for model input.

    The placeholders are already present in ``normalized_text`` because
    preprocessing ran upstream; this stage owns neither that decision nor the
    text.  It strips them from the **copy it feeds to TF-IDF** only, so the
    echoed source text stays byte-identical to what upstream produced.

    Matching is literal and greedy on the tag shape (``<NAME>``), with a
    whitespace tidy-up afterwards, because a partial removal (``<URL``) would
    leave a fragment that the tokeniser then treats as a real term.
    """
    if not text:
        return ""
    import re

    cleaned = re.sub(r"<[A-Za-z_][A-Za-z0-9_]*>", " ", str(text))
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_issue_text(
    text: str,
    include_label: bool,
    label: str,
    drop_placeholders: bool,
) -> str:
    """
    Build the text TF-IDF sees for one (issue, label) pair.

    The label is appended only when ``include_label`` is explicitly enabled,
    because a one-token label weighted like a sentence is a link manufactured
    from a single shared word.
    """
    body = strip_placeholder_tokens(text) if drop_placeholders else str(text or "")
    if include_label and label:
        return f"{body} {label}".strip()
    return body
