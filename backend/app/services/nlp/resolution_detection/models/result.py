"""
Result structures for Resolution Detection.

Two design commitments
----------------------
**Every result reports the components that produced it.**  A status of
``partially_resolved`` is not a self-explanatory claim, so the score, the four
weighted terms, the weights themselves and the thresholds all travel with it.  A
reviewer can recompute the verdict by hand, and a disagreement can be located in
a specific term rather than in a general sense of "the model seems off".

**Thin evidence is reported, not hidden.**  A verdict from a two-comment
discussion carries ``evidence_sufficient=False`` and a warning, and its
confidence is scaled down.  The *status* is still one of the three required
states - inventing a fourth would turn a question about our confidence into an
answer about the issue - but nothing about the weakness is concealed.

Confidence is a boundary distance, not a probability
----------------------------------------------------
``confidence`` answers "how far from the nearest status boundary is this score?",
scaled so a boundary scores 0.0 and a score 0.5 or more away from every boundary
scores 1.0.  It is emphatically **not** a probability that the status is correct:
nothing here has been calibrated against annotated resolution outcomes, because
none exist.  Presenting an uncalibrated number as a probability is how a triage
queue ends up treating a 0.9 as a settled fact.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.services.nlp.resolution_detection.base import (
    IssueResolutionEvidence,
    ResolutionStatus,
    TemporalEvidence,
)


class ResolutionResult(BaseModel):
    """
    One issue (or one issue-label pair) and its assessed resolution.

    ``issue_id`` and ``issue_label`` together identify what was judged.  For a
    multi-label issue document there is one result per label, and the two can
    legitimately disagree - "fees" resolved while "roi" is still open is a normal
    and important outcome, and a single collapsed verdict per document would
    hide it.
    """

    # -- what was judged -------------------------------------------------------
    issue_id: str
    issue_label: str
    post_id: Optional[str] = None

    # -- the verdict -----------------------------------------------------------
    status: ResolutionStatus
    resolution_score: float
    #: Heuristic boundary distance in [0, 1].  NOT a probability.
    confidence: float

    # -- the evidence behind it, preserved verbatim ---------------------------
    #: The full input record, echoed so a result is self-contained and a reader
    #: never has to join back to the pipeline to see what it was judged on.
    evidence: IssueResolutionEvidence

    # -- how the score was built ----------------------------------------------
    #: The four weighted terms, before weighting.
    score_components: Dict[str, float] = Field(default_factory=dict)
    #: The normalised weights actually used.
    weights: Dict[str, float] = Field(default_factory=dict)
    #: The thresholds in force, so a verdict can be re-decided without the config.
    thresholds: Dict[str, float] = Field(default_factory=dict)
    #: Counts and provenance: discussion volume, link counts, and whether the
    #: strength and probability terms were supplied or derived.
    signals: Dict[str, Any] = Field(default_factory=dict)

    #: Derived densities, repeated at the top level because they are the two
    #: numbers most often quoted from a result.
    solution_density: float = 0.0
    strong_solution_density: float = 0.0
    best_link_score: float = 0.0
    solution_probability_mean: float = 0.0

    linked_solution_count: int = 0
    strong_link_count: int = 0
    solution_candidate_count: int = 0
    effective_discussion_volume: int = 0

    #: False when the discussion was thinner than
    #: ``minimum_discussion_volume``.  The status above is still one of the three
    #: required states; this flag is what says how much to trust it.
    evidence_sufficient: bool = True
    warnings: List[str] = Field(default_factory=list)

    #: Preserved when supplied, and never a score input in V1.
    temporal: Optional[TemporalEvidence] = None

    # -- provenance ------------------------------------------------------------
    model_name: str = ""
    model_version: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    linked_at: Optional[str] = None

    def pair_key(self) -> str:
        """Stable identity of the (issue, label) pair judged here."""
        return f"{self.issue_id}|{self.issue_label}"

    @property
    def is_resolved(self) -> bool:
        return self.status == ResolutionStatus.RESOLVED

    @property
    def is_partially_resolved(self) -> bool:
        return self.status == ResolutionStatus.PARTIALLY_RESOLVED

    @property
    def is_unresolved(self) -> bool:
        return self.status == ResolutionStatus.UNRESOLVED

    @property
    def recompute_score(self) -> float:
        """
        The weighted sum, recomputed from the reported components.

        Provided so a test - or a reviewer - can confirm that ``resolution_score``
        is the documented equation applied to the reported terms, using the
        reported weights, without trusting the resolver's arithmetic.
        """
        return sum(
            self.weights.get(name, 0.0) * value
            for name, value in self.score_components.items()
        )

    def redecide(self, unresolved_upper: float, resolved_lower: float) -> ResolutionStatus:
        """
        Re-apply the threshold function at different cut-points.

        Returns the status this score *would* receive; the result object is not
        modified.  Lets a consumer explore the trade-off without a re-run, which
        is free because the score does not depend on the thresholds.
        """
        if self.resolution_score < unresolved_upper:
            return ResolutionStatus.UNRESOLVED
        if self.resolution_score < resolved_lower:
            return ResolutionStatus.PARTIALLY_RESOLVED
        return ResolutionStatus.RESOLVED

    def format_summary(self) -> str:
        """One line, plus warnings, in a form a person can read in a triage list."""
        sufficient = "ok" if self.evidence_sufficient else "THIN"
        line = (
            f"{self.issue_id}[{self.issue_label}] "
            f"{self.status.value:<19} score={self.resolution_score:.4f} "
            f"conf={self.confidence:.3f} "
            f"linked={self.linked_solution_count}/{self.effective_discussion_volume} "
            f"density={self.solution_density:.3f} best={self.best_link_score:.3f} "
            f"evidence={sufficient}"
        )
        for warning in self.warnings:
            line += f"\n    ! {warning}"
        return line


class ResolutionRunReport(BaseModel):
    """
    What one detection run did, independent of what it concluded.

    Kept separate from the results for the same reason as in every other stage: a
    run's provenance has to survive an empty result.  "0 issues assessed, all of
    them on two-comment threads" is a reportable fact about the pipeline, and
    folding it into an empty result nobody opens is how a degenerate run goes
    unnoticed.
    """

    model_name: str
    model_version: str

    n_evidence_records: int = 0
    #: Distinct issue documents, which is fewer than the record count whenever an
    #: issue carries several labels.
    n_issue_documents: int = 0
    n_issue_labels: int = 0

    n_resolved: int = 0
    n_partially_resolved: int = 0
    n_unresolved: int = 0
    n_evidence_sufficient: int = 0
    n_evidence_insufficient: int = 0

    #: (issue, label) pairs that appeared more than once; first occurrence wins.
    duplicate_pair_keys: List[str] = Field(default_factory=list)
    #: Records whose ``response_counts`` histogram disagrees with the scalar role
    #: counts.  A warning, not an error: both views are legitimate and the
    #: histogram is the one a caller most often gets wrong.
    response_count_mismatches: List[str] = Field(default_factory=list)
    #: Records with temporal evidence supplied, and with a follow-up resolution
    #: signal - reported because the signal is recorded but unused in V1.
    n_with_temporal_evidence: int = 0
    n_with_follow_up_resolution_signal: int = 0

    thresholds: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    linked_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def status_counts(self) -> Dict[str, int]:
        return {
            ResolutionStatus.RESOLVED.value: self.n_resolved,
            ResolutionStatus.PARTIALLY_RESOLVED.value: self.n_partially_resolved,
            ResolutionStatus.UNRESOLVED.value: self.n_unresolved,
        }

    @property
    def produced_results(self) -> bool:
        return self.n_evidence_records > 0

    @property
    def is_degenerate(self) -> bool:
        """
        True when there was input but no verdict in the run was worth acting on.

        Two shapes, both pipeline problems rather than findings about the corpus:
        every record was a duplicate of one already assessed, so nothing new was
        judged; or nothing cleared the discussion-volume guard, so every verdict
        is a non-answer wearing a status.  A minimum volume set above the
        corpus's typical thread size produces the second, and it looks exactly
        like "all these issues are unresolved" in a status histogram.
        """
        if self.n_evidence_records == 0:
            return False
        return self.n_issue_labels == 0 or self.n_evidence_sufficient == 0


class ResolutionBatchResult(BaseModel):
    """
    Structured output of one detection run over many issues.

    Holds every result in a stable order plus the run-level provenance, and
    exposes the queries a consumer actually asks: which issues are still open,
    what happened to one issue's several labels, and how many verdicts rest on
    thin evidence.
    """

    model_name: str
    model_version: str

    results: List[ResolutionResult] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    linked_at: Optional[str] = None
    report: Optional[ResolutionRunReport] = None

    # -- lookups ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> ResolutionResult:
        return self.results[index]

    def by_pair_key(self) -> Dict[str, ResolutionResult]:
        """Every result keyed by ``issue_id|label``."""
        return {result.pair_key(): result for result in self.results}

    def results_for_issue(self, issue_id: str) -> List[ResolutionResult]:
        """
        Every label assessed for one issue document, in stable order.

        This is the multi-label view: one document, several independently judged
        questions, which may resolve differently.
        """
        return [result for result in self.results if result.issue_id == issue_id]

    def statuses_for_issue(self, issue_id: str) -> Dict[str, str]:
        """``{issue_label: status}`` for one document."""
        return {
            result.issue_label: result.status.value
            for result in self.results_for_issue(issue_id)
        }

    def with_status(self, status: ResolutionStatus) -> List[ResolutionResult]:
        """All results carrying one status, in stable order."""
        wanted = ResolutionStatus(status) if not isinstance(status, ResolutionStatus) else status
        return [result for result in self.results if result.status == wanted]

    def unresolved(self) -> List[ResolutionResult]:
        """
        The answer to the assignment's question: "which issues remain unresolved?"

        Both ``unresolved`` and ``partially_resolved`` count as *not resolved*,
        and this method returns the strict ``unresolved`` set;
        :meth:`not_resolved` returns the union.  A triage list that merged them
        would understate the work left to do.
        """
        return self.with_status(ResolutionStatus.UNRESOLVED)

    def not_resolved(self) -> List[ResolutionResult]:
        """Everything that is not fully ``resolved`` - the open-issue work list."""
        return [
            result
            for result in self.results
            if result.status != ResolutionStatus.RESOLVED
        ]

    def thin_evidence(self) -> List[ResolutionResult]:
        """Results whose discussion was too small to support the verdict."""
        return [result for result in self.results if not result.evidence_sufficient]

    def status_counts(self) -> Dict[str, int]:
        counts = {status.value: 0 for status in ResolutionStatus}
        for result in self.results:
            counts[result.status.value] += 1
        return counts

    def format_summary(self) -> str:
        """A readable roll-up, with the thin-evidence cases called out."""
        counts = self.status_counts()
        lines = [
            "Resolution Detection",
            f"  issues assessed : {self.report.n_issue_documents if self.report else 0} "
            f"document(s), {len(self)} label-level verdict(s)",
            f"  resolved        : {counts[ResolutionStatus.RESOLVED.value]}",
            f"  partially resolv.: {counts[ResolutionStatus.PARTIALLY_RESOLVED.value]}",
            f"  unresolved      : {counts[ResolutionStatus.UNRESOLVED.value]}",
            f"  thresholds      : unresolved < {self.thresholds.get('unresolved_upper', 0):.2f}"
            f" <= partially_resolved < {self.thresholds.get('resolved_lower', 0):.2f}"
            f" <= resolved",
        ]
        thin = self.thin_evidence()
        if thin:
            lines.append(
                f"  thin evidence   : {len(thin)} verdict(s) rest on discussions below "
                "the minimum volume; see the per-result warnings"
            )
        if self.report and self.report.duplicate_pair_keys:
            lines.append(
                f"  duplicates      : {len(self.report.duplicate_pair_keys)} repeated "
                "(issue, label) pair(s); first occurrence kept"
            )
        for warning in (self.report.warnings if self.report else []):
            lines.append(f"  ! {warning}")
        if not self.results:
            lines.append("  no issue was assessed; see the run report")
        return "\n".join(lines)
