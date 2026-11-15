"""
The binary target, and the free-form ``solution_type`` annotation field.

The label space
---------------
Solution detection has exactly two targets, and they are not a taxonomy:

    not_solution  (0)   the comment offers no actionable recommendation
    solution      (1)   the comment provides an actionable solution/recommendation

This is deliberately a *narrow* target.  The assignment's six response roles are
a different question answered by a different stage (see
``app.services.nlp.response_classification``); this package must not import that
package, so it re-derives only the two booleans it needs from the annotation
file.  The canonical order below fixes the row/column order of the confusion
matrix, so two runs produce byte-identical reports:

    [[true_negative,  false_positive],
     [false_negative, true_positive]]

Why ``solution_type`` is NOT a taxonomy
---------------------------------------
``solution_type`` is free annotation data and is **never learned in V1**.  There
deliberately is no module-level list of "financial_aid", "placement_preparation",
etc. - a fixed list would be an IIM-specific assumption baked into a general
model, and every new community would need a code change and a version bump to
express a category its annotators invented.

So this module provides only *normalisation* and *reporting* for the field:

- :func:`normalise_solution_type` - strip whitespace, treat blank as absent
- :func:`distinct_solution_types` - the sorted set actually present in a corpus
- :func:`validate_solution_type` - a present value must be a non-blank string

The classifier learns ``is_solution`` alone.  ``solution_type`` rides along on
annotations, is echoed into the training report so a reviewer can see the
category coverage that exists, and is reserved for the future solution-extraction
stage described in docs/solution_detection.md.
"""

from typing import Any, Dict, List, Optional, Sequence

#: Human-readable name of the negative target.
LABEL_NOT_SOLUTION = "not_solution"

#: Human-readable name of the positive target.  The downstream consumer cares
#: about this one, so it is the default positive class in every metric.
LABEL_SOLUTION = "solution"

#: The two targets, in canonical order.  Position in this tuple IS the integer
#: class the logistic regression is fitted on, which is what keeps the confusion
#: matrix stable and the two-class softmax identical to a sigmoid.
TARGET_LABELS = (LABEL_NOT_SOLUTION, LABEL_SOLUTION)

#: The integer class index meaning "provides a solution".
POSITIVE_CLASS_INDEX = 1

#: The integer class index meaning "provides no solution".
NEGATIVE_CLASS_INDEX = 0

#: Acceptable spellings when reading a boolean annotation cell, so a CSV written
#: by hand or exported from a spreadsheet loads without a crash.
_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def target_label(value: bool) -> str:
    """Map a boolean target onto its canonical label name."""
    return LABEL_SOLUTION if value else LABEL_NOT_SOLUTION


def target_index(value: bool) -> int:
    """Map a boolean target onto its integer class index."""
    return POSITIVE_CLASS_INDEX if value else NEGATIVE_CLASS_INDEX


def label_to_bool(label: Optional[str]) -> Optional[bool]:
    """
    Map a canonical label name back onto its boolean value.

    Returns ``None`` for anything that is not one of the two labels, so a caller
    can count it as unscoreable instead of guessing.
    """
    if not isinstance(label, str):
        return None
    normalised = label.strip().lower()
    if normalised == LABEL_SOLUTION:
        return True
    if normalised == LABEL_NOT_SOLUTION:
        return False
    return None


def parse_bool_cell(
    value: Any,
    document_id: Optional[str] = None,
) -> bool:
    """
    Parse an ``is_solution`` cell from a CSV row.

    Raises ``ValueError`` for a blank or unrecognised value rather than defaulting
    to ``False``.  Defaulting would be the single most damaging failure mode in
    this package: a missing annotation would silently become a *negative*
    training example, teaching the detector to ignore solutions.
    """
    subject = f"annotation {document_id!r}" if document_id else "value"
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"{subject} has no is_solution value. Every annotated comment needs "
            "exactly one of 'true' or 'false'; a blank cell is a missing "
            "annotation, not a negative example."
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(
            f"{subject} has is_solution {value!r}; expected one of "
            f"{sorted(_TRUE_TOKENS | _FALSE_TOKENS)}"
        )
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    raise ValueError(
        f"{subject} has is_solution {value!r}, which is not a boolean. Accepted "
        f"spellings: {', '.join(sorted(_TRUE_TOKENS | _FALSE_TOKENS))}. Note that "
        "an unrecognised value is refused rather than read as False, because a "
        "silently mislabelled positive would teach the detector to miss solutions."
    )


# ---------------------------------------------------------------------------
# solution_type: free-form, validated for shape, never learned
# ---------------------------------------------------------------------------


def normalise_solution_type(value: Optional[str]) -> Optional[str]:
    """
    Return the trimmed ``solution_type``, or ``None`` when absent or blank.

    Deliberately permissive about *content*: any non-blank string is a legal
    category, because the taxonomy belongs to the annotators.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_solution_type(
    value: Optional[str],
    document_id: Optional[str] = None,
) -> Optional[str]:
    """
    Return the normalised ``solution_type``, or raise ``ValueError``.

    Only shape is checked.  There is no allowed-value list, on purpose.
    """
    subject = f"annotation {document_id!r}" if document_id else "value"
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{subject} has solution_type {value!r} of type "
            f"{type(value).__name__}; solution_type is free-form annotation text "
            "and must be a string when present (or left empty)"
        )
    text = value.strip()
    if not text:
        return None
    return text


def distinct_solution_types(values: Sequence[Optional[str]]) -> List[str]:
    """
    Sorted set of non-blank ``solution_type`` values.

    Sorted rather than first-seen so the training report is byte-identical across
    runs regardless of annotation-file row order.
    """
    seen: set = set()
    for value in values:
        normalised = normalise_solution_type(value)
        if normalised:
            seen.add(normalised)
    return sorted(seen)


def solution_type_counts(values: Sequence[Optional[str]]) -> Dict[str, int]:
    """How many annotations carry each ``solution_type`` (absent excluded)."""
    counts: Dict[str, int] = {}
    for value in values:
        normalised = normalise_solution_type(value)
        if not normalised:
            continue
        counts[normalised] = counts.get(normalised, 0) + 1
    return dict(sorted(counts.items()))
