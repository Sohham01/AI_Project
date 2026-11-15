"""
Derivation and validation of the numbers a resolution verdict is computed from.

Why this is a separate module
-----------------------------
The temptation in a stage like this is to compute the score inline in the
resolver.  That hides two things worth isolating:

1. **What was supplied versus what was derived.**  A caller may pass
   ``best_link_score`` directly, or pass ``link_scores`` and let the stage
   summarise.  When both are present, ``link_scores`` wins.  Which one actually
   drove the verdict is not obvious from the evidence model, and getting it
   wrong would mean a score nobody can reproduce, so it is resolved explicitly
   here and reported in the result's provenance.
2. **Whether the record is even coherent.**  A linked-solution count larger than
   the comment count, a strong-link count above the linked count, a density
   outside [0, 1] - each of these describes an impossible issue.  The resolver
   would happily compute a plausible score from them, which is the problem: the
   impossible record would be laundered into a confident verdict.  So every such
   inconsistency is refused here, with a message naming the fields involved.

The Densities
-------------
::

    solution_density        = linked_solution_count / max(total_comments, 1)
    strong_solution_density = strong_link_count    / max(total_comments, 1)

The ``max(total_comments, 1)`` is a divide-by-zero guard, and it is a real
boundary rather than a theoretical one: a discussion with zero comments is
exactly the record that reaches a resolver first in a fresh deployment, and
0/0 must not produce a crash or a NaN.

The consequence is worth stating plainly, because it is a limitation of the
definition rather than of the implementation: with ``total_comments == 0`` the
denominator becomes 1, so any nonzero linked-solution count yields density 1.0.
Such a record is internally inconsistent and is refused by
:func:`derive_evidence`, so the clamp only ever applies to a genuinely empty
discussion with no links, where the density is 0.0 either way.

Only ``linked_solution_count`` and ``strong_link_count`` - evidence attached to
*this* issue and *this* label - reach the score.  ``solution_candidate_count``
(a count of comments assessed as possible solutions) is carried for reporting
only: a candidate that was never linked to this issue is evidence about the
discussion, not about whether this issue got fixed, and letting it raise the
score would let any busy thread look resolved.
"""

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel

from app.services.nlp.resolution_detection.base import IssueResolutionEvidence
from app.services.nlp.resolution_detection.models.labels import (
    check_count,
    check_finite,
    check_unit_interval,
    clamp_unit,
    normalise_count_mapping,
    stable_sorted_scores,
)


class DerivedEvidence(BaseModel):
    """
    The four score terms plus the context needed to interpret the verdict.

    Every field is guaranteed to be in range by :func:`derive_evidence` rather
    than by the caller's construction, so this is a computed view over one
    evidence record, not a second source of truth a caller could populate.
    """

    # -- the four score terms, each in [0, 1] --------------------------------
    solution_density: float = 0.0
    strong_solution_density: float = 0.0
    best_link_score: float = 0.0
    solution_probability_mean: float = 0.0

    # -- context, not scored ---------------------------------------------------
    effective_discussion_volume: int = 0
    total_comments: int = 0
    linked_solution_count: int = 0
    strong_link_count: int = 0
    solution_candidate_count: int = 0
    mean_link_score: float = 0.0
    solution_probability_max: float = 0.0
    link_count: int = 0

    # -- provenance: what the caller supplied vs. what was derived ------------
    #: ``"link_scores"``, ``"best_link_score"`` or ``"none"``.
    link_strength_source: str = "none"
    #: ``"solution_probabilities"``, ``"solution_probability_mean"`` or ``"none"``.
    probability_source: str = "none"
    #: ``"supplied"`` or ``"derived"`` for solution_density.
    density_source: str = "derived"

    def as_components(self) -> Dict[str, float]:
        """The four terms, keyed to match ``ResolutionModelConfig.weight_map()``."""
        return {
            "solution_density": self.solution_density,
            "strong_solution_density": self.strong_solution_density,
            "best_link_score": self.best_link_score,
            "solution_probability_mean": self.solution_probability_mean,
        }

    def as_signal_summary(self) -> Dict[str, Any]:
        """Compact counts, for the result's ``signals`` block."""
        return {
            "effective_discussion_volume": self.effective_discussion_volume,
            "total_comments": self.total_comments,
            "linked_solution_count": self.linked_solution_count,
            "strong_link_count": self.strong_link_count,
            "solution_candidate_count": self.solution_candidate_count,
            "link_count": self.link_count,
            "mean_link_score": self.mean_link_score,
            "solution_probability_max": self.solution_probability_max,
            "link_strength_source": self.link_strength_source,
            "probability_source": self.probability_source,
            "density_source": self.density_source,
        }


def validate_evidence(evidence: IssueResolutionEvidence) -> None:
    """
    Refuse an internally inconsistent record.

    Every message names the fields involved, because the caller's first question
    is "which of my numbers is wrong?" and a generic validation error would send
    them looking in the wrong place.
    """
    subject = f"evidence {evidence.pair_key()!r}"

    if not str(evidence.issue_id or "").strip():
        raise ValueError(f"{subject} has an empty issue_id")
    if not str(evidence.issue_label or "").strip():
        raise ValueError(
            f"{subject} has an empty issue_label. One record is one (issue, label) "
            "pair; a multi-label issue needs one record per label so each is "
            "resolved independently."
        )

    total = check_count(evidence.total_comments, f"{subject} total_comments")
    volume = check_count(evidence.effective_discussion_volume, f"{subject} discussion_volume")
    linked = check_count(evidence.linked_solution_count, f"{subject} linked_solution_count")
    strong = check_count(evidence.strong_link_count, f"{subject} strong_link_count")
    candidates = check_count(
        evidence.solution_candidate_count, f"{subject} solution_candidate_count"
    )
    check_count(evidence.agreement_count, f"{subject} agreement_count")
    check_count(evidence.experience_count, f"{subject} experience_count")
    check_count(evidence.opinion_count, f"{subject} opinion_count")
    check_count(evidence.advice_count, f"{subject} advice_count")
    check_count(evidence.solution_count, f"{subject} solution_count")
    check_count(evidence.counterargument_count, f"{subject} counterargument_count")
    normalise_count_mapping(evidence.response_counts, f"{subject} response_counts")

    if linked > total:
        raise ValueError(
            f"{subject} has linked_solution_count={linked} > total_comments={total}. "
            "Linked solutions are comments in the discussion, so this is impossible; "
            "either total_comments is understated or a link crosses discussions."
        )
    if strong > linked:
        raise ValueError(
            f"{subject} has strong_link_count={strong} > "
            f"linked_solution_count={linked}. Strong links are a subset of linked "
            "solutions, so strong cannot exceed linked."
        )
    if candidates < linked:
        raise ValueError(
            f"{subject} has solution_candidate_count={candidates} < "
            f"linked_solution_count={linked}. Every linked solution is a candidate, "
            "so the candidate count cannot be smaller."
        )
    if evidence.discussion_volume is not None and volume < total:
        raise ValueError(
            f"{subject} has discussion_volume={volume} < total_comments={total}. "
            "The discussion volume is the denominator of every density, so it "
            "cannot be smaller than the comment count it summarises."
        )

    if evidence.solution_density is not None:
        supplied = check_unit_interval(evidence.solution_density, f"{subject} solution_density")
        derived = linked / max(total, 1)
        # Compared with a tolerance, not for equality: the caller may have rounded.
        if abs(supplied - derived) > 1e-6:
            raise ValueError(
                f"{subject} has solution_density={supplied} but "
                f"linked_solution_count/total_comments = {derived}. Supply either the "
                "density or the counts, not an inconsistent pair."
            )

    for name in ("best_link_score", "mean_link_score"):
        value = getattr(evidence, name)
        if value is not None:
            check_unit_interval(value, f"{subject} {name}")
    for name in ("solution_probability_mean", "solution_probability_max"):
        value = getattr(evidence, name)
        if value is not None:
            check_unit_interval(value, f"{subject} {name}")

    stable_sorted_scores(evidence.link_scores, f"{subject} link_scores")
    stable_sorted_scores(evidence.solution_probabilities, f"{subject} solution_probabilities")

    if evidence.link_scores is not None and evidence.best_link_score is not None:
        if abs(max(evidence.link_scores) - float(evidence.best_link_score)) > 1e-6:
            raise ValueError(
                f"{subject} has best_link_score={evidence.best_link_score} but "
                f"max(link_scores)={max(evidence.link_scores)}. Supply one or the other."
            )

    if evidence.temporal is not None:
        check_count(evidence.temporal.follow_up_count, f"{subject} temporal.follow_up_count")


def derive_evidence(evidence: IssueResolutionEvidence) -> DerivedEvidence:
    """
    Validate a record, then compute the four score terms from it.

    Validation happens first, always: computing before checking would produce a
    score from an impossible record and only then complain, which is the wrong
    order - the caller would be looking at a number that should never have
    existed.
    """
    validate_evidence(evidence)

    subject = f"evidence {evidence.pair_key()!r}"
    total = int(evidence.total_comments)
    volume = int(evidence.effective_discussion_volume)
    denominator = max(total, 1)

    density = evidence.linked_solution_count / denominator
    strong_density = evidence.strong_link_count / denominator

    link_scores = stable_sorted_scores(evidence.link_scores, f"{subject} link_scores")
    if link_scores:
        best = max(link_scores)
        mean = sum(link_scores) / len(link_scores)
        link_source = "link_scores"
    else:
        best = float(evidence.best_link_score or 0.0)
        mean = float(evidence.mean_link_score or 0.0)
        link_source = "best_link_score" if evidence.best_link_score is not None else "none"

    probabilities = stable_sorted_scores(
        evidence.solution_probabilities, f"{subject} solution_probabilities"
    )
    if probabilities:
        probability_mean = sum(probabilities) / len(probabilities)
        probability_max = max(probabilities)
        probability_source = "solution_probabilities"
    else:
        probability_mean = float(evidence.solution_probability_mean or 0.0)
        probability_max = float(evidence.solution_probability_max or 0.0)
        probability_source = (
            "solution_probability_mean"
            if evidence.solution_probability_mean is not None
            else "none"
        )

    density_source = "supplied" if evidence.solution_density is not None else "derived"

    # Every term is clamped.  With consistent evidence the clamp is a no-op, and
    # it exists so a floating-point overshoot at exactly 1.0 cannot push a score
    # above the resolved band.
    return DerivedEvidence(
        solution_density=clamp_unit(check_finite(density, f"{subject} solution_density")),
        strong_solution_density=clamp_unit(
            check_finite(strong_density, f"{subject} strong_solution_density")
        ),
        best_link_score=clamp_unit(check_finite(best, f"{subject} best_link_score")),
        solution_probability_mean=clamp_unit(
            check_finite(probability_mean, f"{subject} solution_probability_mean")
        ),
        effective_discussion_volume=volume,
        total_comments=total,
        linked_solution_count=int(evidence.linked_solution_count),
        strong_link_count=int(evidence.strong_link_count),
        solution_candidate_count=int(evidence.solution_candidate_count),
        mean_link_score=clamp_unit(check_finite(mean, f"{subject} mean_link_score")),
        solution_probability_max=clamp_unit(
            check_finite(probability_max, f"{subject} solution_probability_max")
        ),
        link_count=len(link_scores),
        link_strength_source=link_source,
        probability_source=probability_source,
        density_source=density_source,
    )


def derive_evidence_list(
    evidence: Sequence[IssueResolutionEvidence],
) -> List[DerivedEvidence]:
    """Derive for a whole sequence, preserving order."""
    return [derive_evidence(record) for record in evidence]


def strong_link_count_from_scores(
    link_scores: Optional[List[float]], threshold: float
) -> int:
    """
    Count links at or above ``threshold``.

    Provided so a caller with only per-link scores can produce
    ``strong_link_count`` without duplicating the comparison.  The linker stage
    owns the definition of "strong"; this is a convenience that does not redefine
    it, and the threshold used is echoed into the result.
    """
    if not link_scores:
        return 0
    validated = stable_sorted_scores(link_scores, "link_scores")
    return sum(1 for score in validated if score >= threshold)
