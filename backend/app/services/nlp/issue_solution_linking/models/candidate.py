"""
Candidate generation: which (issue, issue-label, solution) triples are even
considered for scoring.

Why this module exists
----------------------
The naive implementation is a cross product: every issue label against every
solution comment.  It is O(issues x solutions), it is wrong, and it is wrong in a
way that *looks* correct - every link comes back with a plausible score, so the
output fills up and nothing crashes.  The score threshold does not save you: a
low-similarity pair and a high-similarity pair both produce numbers, and the
threshold becomes the only thing standing between a real edge and a manufactured
one.  Most real corpora contain far more issue-bearing comments than solutions,
so the cross product is mostly noise by construction.

The rule this module implements instead
---------------------------------------
A solution is a candidate for an issue when both of these hold:

1. **The solution is worth considering at all.**  ``is_solution`` is True, or
   ``solution_probability >= candidate_solution_threshold``.  A comment that is
   neither was never a solution, and scoring it against an issue would let a
   bare "same bro" acquire a link by lexical accident.
2. **It is in the same discussion.**  Same ``post_id`` by default.  A solution to
   a fees complaint is nearly always in the fees thread; a comment that happens
   to share vocabulary is a hypothesis, and V1 refuses to make it a finding.

Both rules are configuration, not policy baked into code.  Cross-thread matching
is a real requirement when the community's answers are spread across posts, and
it is available via ``allow_cross_post`` - where the thread term is reduced to
``cross_post_thread_relevance`` so the score has to be carried by the text.

Ordering and determinism
------------------------
Candidates are generated in a total order that does not depend on dict iteration,
set iteration or input order: issues in input order (the caller controls that),
labels sorted, solutions in input order.  Every rejection is counted by reason so
a run that produced nothing can say *why* - "0 links" and "0 links because no
solution cleared the candidate threshold" are very different situations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence,
    LinkModelConfig,
    SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.models.labels import normalise_labels


#: Rejection reasons, in the order they are counted.  Exposed as constants so a
#: consumer can switch on a reason instead of matching on an English sentence.
REASON_SAME_POST = "same_post"
REASON_CROSS_POST = "cross_post"
REASON_PARENT_CHILD = "parent_child"
REASON_UNRESOLVED_THREAD = "unresolved_thread"
REASON_CROSS_THREAD_DISABLED = "cross_thread_disabled"
REASON_BELOW_CANDIDATE_THRESHOLD = "below_candidate_threshold"
REASON_ISSUE_HAS_NO_LABEL = "issue_has_no_label"
REASON_CAPPED = "capped"

#: Thread outcomes that block a pair outright when cross-post matching is off.
#: Both mean the linker could not establish that the two comments share a
#: discussion, so scoring them would be the cross product wearing a different
#: hat.  They are kept as separate reasons because "provably a different thread"
#: and "we cannot tell" are different problems with different fixes - one is data,
#: the other is missing ids.  With ``allow_cross_post=True`` an unresolvable pair
#: is admitted at the cross-post tier instead: refusing it would make the option
#: useless for exactly the rows that need it most, the ones missing ids.
_BLOCKED_THREAD_REASONS = frozenset({REASON_CROSS_THREAD_DISABLED, REASON_UNRESOLVED_THREAD})


class CandidateDecision(BaseModel):
    """
    Whether one (issue, label, solution) triple is scored at all, and why.

    ``accepted=False`` always carries a reason.  A rejection with no reason is
    indistinguishable from a bug, and this stage's whole purpose is to be able to
    explain its own silence.
    """

    issue_document_id: str
    issue_label: str
    solution_document_id: str
    accepted: bool = False
    reason: str = ""
    #: True when both documents carry a ``post_id`` and they match.
    same_post: bool = False
    #: True when the solution is a direct reply to the issue's comment.
    parent_child: bool = False
    #: Thread relevance in [0, 1], the value fed into the score.
    thread_relevance: float = 0.0
    #: Why the solution was a candidate at all: "is_solution" or
    #: "probability_threshold".
    candidacy_basis: str = ""

    @property
    def relation(self) -> str:
        """``parent_child`` / ``same_post`` / ``cross_post`` / ``none``."""
        if not self.accepted:
            return "none"
        if self.parent_child:
            return REASON_PARENT_CHILD
        if self.same_post:
            return REASON_SAME_POST
        return REASON_CROSS_POST


def normalise_document_ids(values: Sequence[str]) -> List[str]:
    """Sorted, de-duplicated, blank-free document ids - for deterministic order."""
    seen: set = set()
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return sorted(result)


def normalised_labels(issue: IssueEvidence) -> List[str]:
    """
    The issue's labels, cleaned and in a stable order.

    Delegates to the shared :func:`~.labels.normalise_labels` so candidate
    ordering and the label metadata reported on a
    :class:`~.models.result.LinkingResult` can never disagree about what a
    document's labels are.
    """
    return normalise_labels(issue.issue_labels)


def is_solution_candidate(
    solution: SolutionEvidence,
    candidate_solution_threshold: float,
) -> Tuple[bool, str]:
    """
    Rule 1: is this comment worth scoring against anything?

    Returns ``(accepted, basis)`` where basis is ``"is_solution"`` when the
    upstream boolean already asserted it, ``"probability_threshold"`` when only
    the probability cleared the bar, and ``""`` when it did not.

    The OR is deliberate.  The upstream boolean is a decision somebody already
    made against a reviewed threshold, and re-thresholding it here would quietly
    overrule a previous stage; the probability is offered for callers who want a
    stricter bar, not as a way to veto a stated ``True``.
    """
    if solution.is_solution:
        return True, "is_solution"
    if float(solution.solution_probability or 0.0) >= candidate_solution_threshold:
        return True, "probability_threshold"
    return False, ""


def thread_signals(
    issue: IssueEvidence,
    solution: SolutionEvidence,
    config: LinkModelConfig,
) -> Tuple[bool, bool, float, str]:
    """
    Rule 2: where do these two comments sit relative to each other?

    Returns ``(same_post, parent_child, thread_relevance, reason)``.

    ``thread_relevance`` is 1.0 for a direct reply, 1.0 for a same-post pair,
    and ``cross_post_thread_relevance`` (0.0 by default) when the pair is
    cross-post.  An unresolvable thread - either side missing a ``post_id`` - is
    never given a fabricated value: it either blocks the pair (the default) or is
    admitted at the cross-post level if cross-post matching is enabled.
    """
    issue_post = str(issue.post_id).strip() if issue.post_id else ""
    solution_post = str(solution.post_id).strip() if solution.post_id else ""
    resolvable = bool(issue_post and solution_post)

    parent_child = bool(
        config.use_parent_context
        and solution.parent_comment_id
        and issue.comment_id
        and str(solution.parent_comment_id).strip() == str(issue.comment_id).strip()
    )

    if not resolvable:
        # Cannot prove same-thread, so it is treated as the weakest case rather
        # than optimistically promoted.
        if config.allow_cross_post:
            return False, parent_child, config.cross_post_thread_relevance, REASON_UNRESOLVED_THREAD
        return False, parent_child, 0.0, REASON_UNRESOLVED_THREAD

    if issue_post == solution_post:
        relevance = 1.0
        if parent_child:
            relevance = max(relevance, config.parent_child_thread_relevance)
        return True, parent_child, relevance, (
            REASON_PARENT_CHILD if parent_child else REASON_SAME_POST
        )

    if config.allow_cross_post:
        return False, parent_child, config.cross_post_thread_relevance, REASON_CROSS_POST
    return False, parent_child, 0.0, REASON_CROSS_THREAD_DISABLED


def generate_candidates(
    issues: Sequence[IssueEvidence],
    solutions: Sequence[SolutionEvidence],
    config: LinkModelConfig,
) -> Tuple[List[CandidateDecision], Dict[str, int]]:
    """
    Build the accepted (issue, label, solution) candidates, plus rejection counts.

    Every accepted candidate is a *scoring* candidate; whether it becomes a
    ``linked=True`` edge is decided later, by the score.  Rejections are counted
    by reason in :data:`REASON_SAME_POST`'s neighbourhood, so an empty result can
    be explained without re-running with a debugger attached.
    """
    accepted: List[CandidateDecision] = []
    rejected: Dict[str, int] = {}

    def count(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    # Candidate gating happens once per solution, not per triple, so the
    # probability threshold is not re-applied N times per issue.
    eligible: List[Tuple[SolutionEvidence, str]] = []
    for solution in solutions:
        ok, basis = is_solution_candidate(
            solution, config.candidate_solution_threshold
        )
        if ok:
            eligible.append((solution, basis))
        else:
            count(REASON_BELOW_CANDIDATE_THRESHOLD)

    if not eligible or not issues:
        return accepted, rejected

    for issue in issues:
        labels = normalised_labels(issue)
        if not labels:
            count(REASON_ISSUE_HAS_NO_LABEL)
            continue

        per_label: Dict[str, List[CandidateDecision]] = {}
        for label in labels:
            scored: List[CandidateDecision] = []
            for solution, basis in eligible:
                same_post, parent_child, relevance, reason = thread_signals(
                    issue, solution, config
                )
                if reason in _BLOCKED_THREAD_REASONS and not (
                    reason == REASON_UNRESOLVED_THREAD and config.allow_cross_post
                ):
                    count(reason)
                    continue
                scored.append(
                    CandidateDecision(
                        issue_document_id=issue.document_id,
                        issue_label=label,
                        solution_document_id=solution.document_id,
                        accepted=True,
                        reason=reason,
                        same_post=same_post,
                        parent_child=parent_child,
                        thread_relevance=relevance,
                        candidacy_basis=basis,
                    )
                )
            if config.max_candidates_per_issue is not None:
                # Deterministic cap: sort by document id, then keep the first N.
                # Sorting by document id rather than by score keeps the cap
                # independent of scoring, which would otherwise make which
                # candidates get scored depend on the cap itself.
                scored.sort(key=lambda d: d.solution_document_id)
                dropped = len(scored) - config.max_candidates_per_issue
                if dropped > 0:
                    rejected[REASON_CAPPED] = rejected.get(REASON_CAPPED, 0) + dropped
                    scored = scored[: config.max_candidates_per_issue]
            per_label[label] = scored
            accepted.extend(scored)

    return accepted, rejected
