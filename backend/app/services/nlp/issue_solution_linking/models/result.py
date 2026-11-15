"""
Result structures for Issue-Solution Linking.

The design commitment this module enforces
------------------------------------------
**Every score is preserved, whether or not the link was asserted.**  A
``LinkPrediction`` with ``linked=False`` is not a discarded candidate - it is the
evidence for why no edge was published, and it is the only way a reviewer can
tell a threshold decision apart from a stage that never looked.  So
``similarity_score``, ``solution_probability``, ``thread_relevance``,
``response_role_score``, ``link_score`` and ``confidence`` are all present on
every link, and ``linked`` is a boolean *derived* from ``link_score`` at the
configured threshold.

Confidence is a boundary distance, not a probability
----------------------------------------------------
``confidence`` answers "how far from the decision boundary is this score?",
normalised so the boundary itself scores 0.0 and either extreme scores 1.0.  It
is emphatically **not** a probability that the link is correct: nothing here has
been calibrated against annotated links, because no annotated link dataset
exists yet.  It is named ``confidence`` because that is what a consumer asks
for, and it is documented here, in ``config_dict()`` and in the docs so nobody
can mistake it for a posterior.  Treating an uncalibrated number as a
probability is how a knowledge base ends up asserting things at 0.9.

Signals
-------
``signals`` is the audit trail: the named components that produced the score,
with the decision facts attached (same post? direct reply? what role?).  It is a
flat mapping of strings and booleans so it serialises without custom encoders,
and every consumer of a link can print it and see the reasoning.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.services.nlp.issue_solution_linking.models.candidate import (
    REASON_CROSS_POST,
    REASON_PARENT_CHILD,
    REASON_SAME_POST,
)


class LinkPrediction(BaseModel):
    """
    One scored (issue, issue-label, solution) triple.

    Multi-label issues produce one of these per label, against the same solution
    text.  Two labels on one issue are two *separate* links, never one merged
    verdict, because "this comment addresses the fees issue but says nothing
    about ROI" is exactly the distinction a knowledge base needs and collapsing
    the labels destroys it.
    """

    #: The document that expressed the issue.
    issue_document_id: str
    #: Which of that document's predicted labels this link is about.
    issue_label: str
    #: The candidate solution comment.
    solution_document_id: str

    # --- the three inputs to the decision --------------------------------
    #: Cosine similarity between the issue text and the solution text, in [0, 1].
    #: 0.0 when either side has no usable terms - never NaN.
    similarity_score: float = 0.0
    #: The upstream solution detector's probability, echoed unchanged.
    solution_probability: float = 0.0
    #: Thread placement strength in [0, 1]: direct reply or same post -> 1.0,
    #: cross post -> ``cross_post_thread_relevance``.
    thread_relevance: float = 0.0
    #: Response-role evidence weight, from ``config.role_scores``.
    response_role_score: float = 0.5

    # --- the decision ------------------------------------------------------
    #: The four terms, before weighting.  Reported so a score can be checked by
    #: hand against the documented formula.
    score_components: Dict[str, float] = Field(default_factory=dict)
    #: The weighted blend, in [0, 1] when ``normalise_weights`` is on.
    link_score: float = 0.0
    #: ``link_score >= link_threshold``.  The boolean a consumer acts on.
    linked: bool = False
    #: The threshold in force for this link.
    link_threshold: float = 0.5
    #: Distance from the decision boundary, in [0, 1].  A heuristic, NOT a
    #: probability - see the module docstring.
    confidence: float = 0.0

    # --- provenance and context -------------------------------------------
    model_name: str = ""
    model_version: str = ""
    #: Audit trail: same post, direct reply, role, candidacy basis, zero-vector
    #: flag, issue/solution text used.
    signals: Dict[str, Any] = Field(default_factory=dict)

    # --- echoed source metadata (never modified by the linker) -----------
    issue_post_id: Optional[str] = None
    issue_comment_id: Optional[str] = None
    issue_text: str = ""
    issue_original_text: Optional[str] = None
    solution_post_id: Optional[str] = None
    solution_comment_id: Optional[str] = None
    solution_text: str = ""
    solution_original_text: Optional[str] = None
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False

    @property
    def relation(self) -> str:
        """
        How the two comments are placed relative to each other.

        Read from the recorded ``thread_reason`` rather than inferred from
        ``thread_relevance``: a cross-post pair carries a thread relevance of 0.0
        by design, so inferring from the value would report it as
        ``unresolved_thread`` - two different situations (we know they are in
        different threads vs. we could not tell) that both deserve to be
        distinguishable in the output.
        """
        reason = str(self.signals.get("thread_reason") or "")
        if reason:
            return reason
        if bool(self.signals.get("parent_child")):
            return REASON_PARENT_CHILD
        if bool(self.signals.get("same_post")):
            return REASON_SAME_POST
        return "unresolved_thread"

    @property
    def is_strong_candidate(self) -> bool:
        """
        A link that clears the threshold *and* has real semantic support.

        Reported separately because a link can be asserted on thread placement
        and detector confidence alone, which is a weaker claim than "these two
        texts are about the same thing".  Consumers that care should ask for
        this rather than reading ``linked`` as the whole story.
        """
        return self.linked and self.similarity_score > 0.0


class LinkingResult(BaseModel):
    """
    Structured output of one linking run.

    ``links`` holds every scored candidate; ``n_linked`` counts only those with
    ``linked=True``.  Nothing is dropped between the two, so a threshold sweep
    needs no re-run: re-derive the boolean with
    :meth:`links_at_threshold` and compare.
    """

    model_name: str
    model_version: str

    link_threshold: float
    #: Every scored candidate, in a total deterministic order.
    links: List[LinkPrediction] = Field(default_factory=list)

    n_issues: int = 0
    n_solutions: int = 0
    #: Solutions that cleared the candidate rules at all.
    n_solution_candidates: int = 0
    #: Issue *labels* considered (one document may contribute several).
    n_issue_labels: int = 0
    #: Scored candidates, i.e. ``len(links)``.
    n_scored_links: int = 0
    #: Candidates asserted as links.
    n_linked: int = 0
    #: Distinct issues with at least one asserted link.
    n_linked_issues: int = 0
    #: Distinct solutions used by at least one asserted link.
    n_linked_solutions: int = 0

    #: Distinct TF-IDF terms fitted for this run.  0 means no usable vocabulary
    #: existed, in which case every similarity is 0.0 and the scores rest entirely
    #: on the non-semantic terms.
    n_features: int = 0
    #: Representations with no usable terms (empty text, or all placeholders).
    n_zero_vector_issues: int = 0
    n_zero_vector_solutions: int = 0

    #: Candidate rejections by reason, so an empty result can be explained.
    candidate_rejections: Dict[str, int] = Field(default_factory=dict)
    #: Issue labels per issue document, canonical order.
    issue_labels_by_document: Dict[str, List[str]] = Field(default_factory=dict)

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    linked_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    # -- lookups -----------------------------------------------------------

    def links_for_issue(self, issue_document_id: str) -> List[LinkPrediction]:
        """Every scored link for one issue document, in run order."""
        return [link for link in self.links if link.issue_document_id == issue_document_id]

    def links_for_solution(self, solution_document_id: str) -> List[LinkPrediction]:
        """Every scored link using one solution, in run order."""
        return [
            link for link in self.links if link.solution_document_id == solution_document_id
        ]

    def linked_pairs(self) -> List[Dict[str, str]]:
        """
        The asserted edges, as plain dicts, in run order.

        The shape a downstream consumer or a future linker will want:
        ``issue_document_id``, ``issue_label``, ``solution_document_id``.
        """
        return [
            {
                "issue_document_id": link.issue_document_id,
                "issue_label": link.issue_label,
                "solution_document_id": link.solution_document_id,
            }
            for link in self.links
            if link.linked
        ]

    def labels_for(self, issue_document_id: str) -> List[str]:
        """The asserted labels for one issue document, canonical order."""
        return sorted(
            {
                link.issue_label
                for link in self.links
                if link.issue_document_id == issue_document_id and link.linked
            }
        )

    def solutions_for(self, issue_document_id: str, issue_label: Optional[str] = None) -> List[str]:
        """
        Solution document ids asserted against one issue (optionally one label).

        This is the "one issue -> many solutions" and "one solution -> many
        issues" view of the graph, which is the deliverable the assignment
        describes.
        """
        return [
            link.solution_document_id
            for link in self.links
            if link.linked
            and link.issue_document_id == issue_document_id
            and (issue_label is None or link.issue_label == issue_label)
        ]

    def issues_for(self, solution_document_id: str) -> List[str]:
        """Issue document ids one solution was asserted to address, canonical order."""
        return sorted(
            {
                link.issue_document_id
                for link in self.links
                if link.linked and link.solution_document_id == solution_document_id
            }
        )

    # -- threshold re-derivation ------------------------------------------

    def links_at_threshold(self, threshold: float) -> List[LinkPrediction]:
        """
        Re-derive the boolean at a different threshold, without re-running.

        Lets a consumer explore the precision/recall trade on scores it already
        has.  The returned objects are copies carrying the new ``link_threshold``
        and ``linked``; every other field, including ``link_score`` and
        ``confidence``, is unchanged.
        """
        chosen = self._validate_threshold(threshold)
        return [
            link.model_copy(
                update={
                    "link_threshold": chosen,
                    "linked": link.link_score >= chosen,
                }
            )
            for link in self.links
        ]

    @staticmethod
    def _validate_threshold(threshold: Any) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be a number, got {type(threshold).__name__}"
            )
        value = float(threshold)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {value}")
        return value

    # -- summaries ---------------------------------------------------------

    def format_summary(self) -> str:
        """Human-readable summary that never overstates what was found."""
        lines = [
            "Issue-Solution Linking",
            f"  issues              : {self.n_issues} document(s), "
            f"{self.n_issue_labels} label(s)",
            f"  solutions           : {self.n_solutions} document(s), "
            f"{self.n_solution_candidates} candidate(s)",
            f"  scored candidates   : {self.n_scored_links}",
            f"  asserted links      : {self.n_linked}",
            f"  issues with a link  : {self.n_linked_issues}",
            f"  solutions used      : {self.n_linked_solutions}",
            f"  link_threshold      : {self.link_threshold}",
        ]
        if self.candidate_rejections:
            detail = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(self.candidate_rejections.items())
            )
            lines.append(f"  candidates rejected : {detail}")
        if not self.links:
            lines.append("  no candidate was scored; see candidate_rejections")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")
            return "\n".join(lines)

        lines.append("")
        lines.append("  asserted links (score-ordered):")
        for link in sorted(
            (link for link in self.links if link.linked),
            key=lambda link: (-link.link_score, link.issue_document_id),
        ):
            lines.append(
                f"    {link.issue_label:<24} {link.issue_document_id} <- "
                f"{link.solution_document_id}  score={link.link_score:.4f} "
                f"sim={link.similarity_score:.4f} p={link.solution_probability:.4f} "
                f"[{link.relation}]"
            )
        below = [link for link in self.links if not link.linked]
        if below:
            lines.append("")
            lines.append(
                f"  {len(below)} candidate(s) scored but were not asserted "
                "(scores preserved, not discarded):"
            )
            for link in below[:5]:
                lines.append(
                    f"    {link.issue_label:<24} {link.issue_document_id} <- "
                    f"{link.solution_document_id}  score={link.link_score:.4f} "
                    f"sim={link.similarity_score:.4f}"
                )
            if len(below) > 5:
                lines.append(f"    ... and {len(below) - 5} more")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


class LinkingRunReport(BaseModel):
    """
    What one linking run did, independent of what it concluded.

    Separated from :class:`LinkingResult` so the provenance of a run survives even
    when the result is empty - "0 links, 0 candidates because no solution cleared
    the threshold" is a reportable fact, and folding it into a result nobody
    looks at because it is empty is how a broken pipeline goes unnoticed.
    """

    model_name: str
    model_version: str
    link_threshold: float

    n_issue_documents: int = 0
    n_issue_labels: int = 0
    n_solution_documents: int = 0
    n_solution_candidates: int = 0
    n_candidates_scored: int = 0
    n_links_asserted: int = 0

    #: Distinct TF-IDF terms fitted for this run.
    n_features: int = 0
    #: Issue texts that produced no usable terms, so their vectors are zero.
    n_zero_vector_issues: int = 0
    #: Solution texts that produced no usable terms, so their vectors are zero.
    n_zero_vector_solutions: int = 0

    candidate_rejections: Dict[str, int] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    linked_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def produced_links(self) -> bool:
        return self.n_links_asserted > 0

    @property
    def is_degenerate(self) -> bool:
        """
        True when the run had input but produced nothing to score.

        The signature of a misconfiguration - every candidate rejected, or no
        usable vocabulary anywhere - as opposed to a corpus that genuinely has
        no solutions.
        """
        if self.n_issue_documents == 0 or self.n_solution_documents == 0:
            return False
        return self.n_candidates_scored == 0
