"""
Candidate logic: which response roles are *worth checking* for a solution.

The distinction this module exists to protect
---------------------------------------------
Response classification answers "what role does this comment play?".  Solution
detection answers "does this comment provide an actionable
solution/recommendation?".  They overlap but are not the same question:

    "apply for the scholarship and submit the documents"  -> response: solution
    "you should contact the placement office"              -> response: advice
    "same bro"                                             -> response: agreement
    "I think the process is broken"                        -> response: opinion

``advice`` and ``solution`` are both *candidates* - the comment at least tries to
help.  But a candidate is not an answer:

    "you should probably try emailing them, not sure though"
    -> candidate (advice), and quite possibly NOT a solution

A link to an issue gets asserted when this stage says ``is_solution=True``, and
that boolean comes only from the trained detector's prediction.  This module
never produces it.

Independence
------------
The response-role names below are plain strings defined in *this* package.  This
module does not import the response-classification stage, and must not: the two
stages are independently replaceable, and a shared enum would make swapping
either one a breaking change for the other.  Because the names are configurable
(:func:`classify_response_candidate` accepts ``candidate_types``), a future
taxonomy revision is a configuration change here rather than a code edit.

The type system is the safety mechanism
---------------------------------------
:class:`CandidateAssessment` carries ``is_solution: Optional[bool]`` which is
**always ``None``**.  A caller physically cannot read a solution verdict out of a
candidate assessment without running the detector and calling
:meth:`CandidateAssessment.with_prediction`.  That is the whole point: "advice is
a candidate" can never be written as "advice is a solution", because the field
that would express the second statement does not exist here.
"""

from typing import Any, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

#: The response-role name for a direct solution, as a plain string.
RESPONSE_TYPE_SOLUTION = "solution"

#: The response-role name for advice, as a plain string.
RESPONSE_TYPE_ADVICE = "advice"

#: Roles worth running the solution detector on, in canonical order.
#:
#: ``opinion``, ``agreement``, ``personal_experience`` and ``counterargument``
#: are excluded on purpose: none of them recommends an action.  Personal
#: experience is the tempting one - "I fixed this by applying for X" is
#: experience-shaped and solution-shaped at once - which is exactly why the
#: detector is run on it and the annotation guideline resolves it, rather than a
#: string match deciding.
DEFAULT_CANDIDATE_RESPONSE_TYPES = (RESPONSE_TYPE_ADVICE, RESPONSE_TYPE_SOLUTION)


def _normalise_response_type(value: Optional[str]) -> Optional[str]:
    """
    Case- and whitespace-insensitive response-role name.

    A response stage is a separate deployable component; normalising here means
    a downstream rename to "Advice" cannot silently make every comment a
    non-candidate.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() or None


def _normalise_candidate_types(
    candidate_types: Optional[Iterable[Any]],
) -> List[str]:
    """Canonical, de-duplicated, sorted candidate role names."""
    if candidate_types is None:
        return list(DEFAULT_CANDIDATE_RESPONSE_TYPES)
    seen = {
        _normalise_response_type(name)
        for name in candidate_types
        if name is not None and str(name).strip()
    }
    return sorted(seen)  # type: ignore[arg-type]


class CandidateAssessment(BaseModel):
    """
    What the response role says about solution-detection *eligibility*.

    ``is_solution`` is fixed at ``None`` for every assessment this module
    produces.  Only :meth:`with_prediction` can set it, and it requires an
    explicit detector verdict to be passed in.  That asymmetry is deliberate: it
    makes "advice implies solution" a type error rather than a review comment.
    """

    #: The response role that was assessed, normalised.  ``None`` when the
    #: upstream stage supplied no role at all.
    response_type: Optional[str] = None
    #: True when ``response_type`` is the direct-solution role.
    is_solution_response: bool = False
    #: True when ``response_type`` is the advice role.
    is_advice_response: bool = False
    #: True when this comment is worth running the solution detector on.
    is_candidate: bool = False
    #: The candidate roles in force, canonical order.
    candidate_types: List[str] = Field(default_factory=list)
    #: Plain-language justification, for audit trails and the demo.
    reason: str = ""

    #: Always ``None`` on an assessment built here.  See the class docstring.
    is_solution: Optional[bool] = None

    def with_prediction(self, is_solution: bool) -> "CandidateAssessment":
        """
        Attach a detector verdict, producing a *new* assessment.

        Used by the classifier to record "this was a candidate, and the detector
        said no" without mutating the eligibility record.  The original stays
        eligible with ``is_solution=None``, so the two facts never get confused.
        """
        return CandidateAssessment(
            response_type=self.response_type,
            is_solution_response=self.is_solution_response,
            is_advice_response=self.is_advice_response,
            is_candidate=self.is_candidate,
            candidate_types=list(self.candidate_types),
            reason=self.reason,
            is_solution=bool(is_solution),
        )

    @property
    def has_detector_verdict(self) -> bool:
        """True only once a detector prediction has been attached."""
        return self.is_solution is not None


def classify_response_candidate(
    response_type: Optional[str],
    candidate_types: Optional[Sequence[str]] = None,
) -> CandidateAssessment:
    """
    Decide whether a response role makes the comment a solution *candidate*.

    Returns a :class:`CandidateAssessment` whose ``is_solution`` is always
    ``None``.  This function answers eligibility, never the solution question -
    a comment labelled ``advice`` is a candidate, which says nothing about
    whether it contains a usable solution.
    """
    roles = _normalise_candidate_types(candidate_types)
    normalised = _normalise_response_type(response_type)

    if normalised is None:
        return CandidateAssessment(
            response_type=None,
            candidate_types=roles,
            is_candidate=False,
            reason=(
                "no response_type supplied, so eligibility is unknown. This stage "
                "does not infer it: run the response classifier first, or treat "
                "the comment as unassessed rather than as a non-candidate."
            ),
        )

    is_solution_role = normalised == RESPONSE_TYPE_SOLUTION
    is_advice_role = normalised == RESPONSE_TYPE_ADVICE
    is_candidate = normalised in roles

    if is_candidate:
        if is_solution_role:
            reason = (
                f"response_type {normalised!r} is a direct solution, so the comment "
                "is a candidate. The detector still decides: a comment can be a "
                "solution in role while being too vague to be actionable."
            )
        else:
            reason = (
                f"response_type {normalised!r} is advice, so the comment is a "
                "candidate worth checking - but advice is not automatically a "
                "solution. 'You should probably try emailing them' is advice and "
                "may contain nothing actionable. is_solution comes from the "
                "detector, not from this role."
            )
    else:
        reason = (
            f"response_type {normalised!r} is not a solution-candidate role "
            f"({', '.join(roles)}), so the comment is not expected to provide a "
            "solution. The detector can still be run on it; this is a prior, not "
            "a prohibition."
        )

    return CandidateAssessment(
        response_type=normalised,
        is_solution_response=is_solution_role,
        is_advice_response=is_advice_role,
        is_candidate=is_candidate,
        candidate_types=roles,
        reason=reason,
    )


def is_solution_candidate(
    response_type: Optional[str],
    candidate_types: Optional[Sequence[str]] = None,
) -> bool:
    """Boolean form of :func:`classify_response_candidate`."""
    return classify_response_candidate(response_type, candidate_types).is_candidate


def is_solution_response_type(
    response_type: Optional[str],
) -> bool:
    """True when ``response_type`` is the direct-solution role."""
    return _normalise_response_type(response_type) == RESPONSE_TYPE_SOLUTION


def is_advice_response_type(
    response_type: Optional[str],
) -> bool:
    """True when ``response_type`` is the advice role."""
    return _normalise_response_type(response_type) == RESPONSE_TYPE_ADVICE
