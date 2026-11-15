"""
The six response types, defined centrally and explicitly.

This is the one stage whose label set is *not* annotation-derived.  The
assignment defines the taxonomy, so it is written down here once and every other
module refers to it:

    A. agreement          "same bro"
    B. personal_experience "I faced this last year"
    C. opinion            "I think this college is better"
    D. advice             "you should contact the placement office"
    E. solution           "apply for the scholarship"
    F. counterargument    "actually the fees aren't that high compared with..."

That is a deliberate difference from ``issue_identification`` and
``sentiment_emotion``, where the labels are whatever the annotators recorded.
Here the taxonomy is a project requirement, so:

- the six names are a fixed, ordered, versioned contract;
- an annotation carrying anything else is a data-entry error and is refused
  rather than learned as a seventh class;
- no V1 class may be added, removed or renamed without bumping the model
  version, because stored predictions are compared against this list.

The classes are mutually exclusive by construction.  A comment plays one
*primary* role in the discussion it replies to, so a single multiclass model
with a softmax head is the right shape: the six probabilities form a
distribution that sums to 1, and the prediction is the arg-max.

Where the boundaries genuinely blur is a real annotation-guideline problem, not
a modelling one.  "You should apply for the scholarship" is an instruction
aimed at the reader; whether that is *advice* or a *solution* depends on the
guideline, and the guideline - not this code - has to decide.  Nothing here
resolves that ambiguity, because doing so silently would bake one annotator's
reading into the model.
"""

from typing import Dict, List, Optional, Sequence, Tuple

#: The complete V1 taxonomy, in the assignment's own order (A-F).
#: The order is canonical, not incidental: it fixes the column order of the
#: probability vector and of the confusion matrix, so two runs of the same model
#: produce byte-identical output and two results can be diffed.
RESPONSE_TYPES: Tuple[str, ...] = (
    "agreement",
    "personal_experience",
    "opinion",
    "advice",
    "solution",
    "counterargument",
)

#: Number of response types in V1.  Not a tuning knob - a V1 model that emits a
#: different number of classes is a different model.
EXPECTED_NUMBER_OF_RESPONSE_TYPES = len(RESPONSE_TYPES)

#: One-line meaning of each class, used by the demo and the documentation so
#: that no module has to restate the taxonomy in prose and drift from it.
RESPONSE_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "agreement": "endorses what the parent or another commenter said",
    "personal_experience": "reports the writer's own past experience",
    "opinion": "states a belief or judgement, without lived evidence",
    "advice": "recommends an action to somebody else",
    "solution": "presents the answer or fix outright",
    "counterargument": "pushes back on the claim being discussed",
}

#: The assignment's letter for each class, kept for annotator-facing output.
RESPONSE_TYPE_CODES: Dict[str, str] = {
    name: chr(ord("A") + position) for position, name in enumerate(RESPONSE_TYPES)
}

_FROZEN = frozenset(RESPONSE_TYPES)


def is_valid_response_type(value: Optional[str]) -> bool:
    """True when ``value`` is one of the six V1 response types."""
    if not isinstance(value, str):
        return False
    return value.strip() in _FROZEN


def validate_response_type(value: Optional[str], document_id: Optional[str] = None) -> str:
    """
    Return the normalised response type, or raise ``ValueError``.

    The error names the offending document and lists the six valid values,
    because the realistic mistake here is a typo ("personal experience",
    "personal-exp") rather than a deliberate seventh class.
    """
    subject = f"annotation {document_id!r} has" if document_id else "value"
    if value is None or not str(value).strip():
        raise ValueError(
            f"{subject} no response_type. Response classification is "
            "single-label, so every annotated comment needs exactly one role; "
            f"valid values are: {', '.join(RESPONSE_TYPES)}."
        )
    normalised = str(value).strip()
    if normalised not in _FROZEN:
        raise ValueError(
            f"{subject} response_type {value!r}, which is not one of the six V1 "
            f"response types ({', '.join(RESPONSE_TYPES)}). These are defined by "
            "the project assignment rather than learned from the data, so an "
            "unrecognised value is a labelling error - check for whitespace, "
            "hyphens or plurals, or annotate the comment as `opinion` when the "
            "role is genuinely unclear."
        )
    return normalised


def unknown_response_types(values: Sequence[Optional[str]]) -> List[str]:
    """
    The subset of ``values`` outside the taxonomy, de-duplicated and sorted.

    Blank values are excluded: a missing label and an unrecognised one are
    different mistakes, and ``validate_response_type`` reports the missing one
    with its own message.  Reporting ``''`` here would blame the label set for
    a blank cell.
    """
    return sorted(
        {
            str(value).strip()
            for value in values
            if value is not None
            and str(value).strip()
            and str(value).strip() not in _FROZEN
        }
    )


def response_type_order(candidate: Sequence[str]) -> List[str]:
    """
    Sort ``candidate`` into canonical V1 order.

    Used for every probability vector, confusion matrix and class list, so
    ordering is decided by this tuple rather than by dict iteration, by
    ``LabelEncoder`` internals or by whatever order a CSV happened to use.
    """
    wanted = set(candidate)
    ordered = [name for name in RESPONSE_TYPES if name in wanted]
    # Anything outside the taxonomy keeps the caller's labels visible instead of
    # being dropped, but can only appear if validation was bypassed.
    ordered.extend(sorted(wanted - set(RESPONSE_TYPES)))
    return ordered
