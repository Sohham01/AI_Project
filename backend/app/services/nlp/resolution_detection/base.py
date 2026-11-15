"""
Base interfaces, configuration and evidence models for Resolution Detection.

What this stage does
--------------------
Issue identification answers "what problem is being expressed?".  Solution
detection answers "does this comment offer a fix?".  Issue-solution linking
answers "does this fix address this problem?".  This stage answers the fourth
question, the only one that is about a discussion rather than a comment or a
pair:

    considering the whole discussion, how resolved does this issue appear?

The distinction is the whole point of the stage, and it is easy to lose:

    one solution present  !=  issue resolved

A single confident "just restart the router" in a forty-comment thread where
nobody else agreed is not a resolved issue.  A thread with four different
proposed fixes and no follow-up is not a resolved issue either.  Resolution is a
property of the *conversation*, and no per-comment classifier can see it.

What this stage deliberately does NOT answer
--------------------------------------------
- **What should the reader do?** No answer generation, no generated solutions.
- **Should the issue be closed?** No workflow or state transition.  This stage
  reports an assessment, and a downstream system may or may not act on it.
- **Is the author's sentiment positive?** Sentiment and emotion are reported
  elsewhere and are NOT inputs here.  See the note on sentiment in
  :mod:`evidence_based_resolver`; a happy poster and a fixed problem are
  different facts, and conflating them is a common and serious error.

Independence
------------
This package imports nothing from any other NLP stage.  :class:`IssueResolution
Evidence` is a neutral container of counts and scores: whoever assembles the
evidence - a pipeline, a script, a human - decides how, and this stage never
asks.  Response roles arrive as a plain ``Dict[str, int]`` histogram and a few
integer counts, so a renamed or replaced upstream classifier is a wiring change
rather than a breaking import.

There is no ``train()``.  V1 is a deterministic function of the evidence and the
configuration, not a fitted model, which is the point: it is auditable, and every
number it produces can be recomputed by hand from the reported components.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class ResolutionStatus(str, Enum):
    """
    The three resolution states this stage is required to distinguish.

    Exactly three, deliberately.  An earlier temptation is to add
    ``insufficient_evidence`` as a fourth state for threads too small to judge;
    that is rejected here because it answers a question about *our confidence*
    rather than about the *issue*, and a consumer asking "which issues remain
    unresolved?" needs a partition of the issues, not a bucket of non-answers.

    Insufficient evidence is therefore reported as :attr:`evidence_sufficient`
    plus a warning on the result, leaving the status in the required three.  A
    thin thread and a genuinely unresolved thread can both be ``unresolved``;
    ``evidence_sufficient=False`` is what distinguishes them, and it is on every
    result precisely so a consumer can filter on it.
    """

    #: Enough relevant, well-linked solution evidence to call the issue settled.
    RESOLVED = "resolved"
    #: Some real progress, but the evidence does not reach the resolved bar.
    PARTIALLY_RESOLVED = "partially_resolved"
    #: The issue has no meaningful solution evidence attached to it.
    UNRESOLVED = "unresolved"

    def __str__(self) -> str:
        return self.value


#: Convenience constants, so callers can compare against plain strings if they
#: prefer without importing the enum.
STATUS_RESOLVED = ResolutionStatus.RESOLVED
STATUS_PARTIALLY_RESOLVED = ResolutionStatus.PARTIALLY_RESOLVED
STATUS_UNRESOLVED = ResolutionStatus.UNRESOLVED

#: Canonical order, weakest evidence first.  Used for evaluation and reporting so
#: "worse to better" is a defined sequence rather than alphabetical accident.
STATUS_ORDER: Tuple[ResolutionStatus, ...] = (
    ResolutionStatus.UNRESOLVED,
    ResolutionStatus.PARTIALLY_RESOLVED,
    ResolutionStatus.RESOLVED,
)


def parse_status(value: Any) -> ResolutionStatus:
    """
    Parse a status string into a :class:`ResolutionStatus`.

    Accepts the three canonical values case-insensitively, plus the obvious
    spellings (``partial``, ``partially-resolved``).  Refuses anything else: an
    unrecognised gold label silently defaulting to ``unresolved`` would quietly
    inflate a resolution rate.
    """
    if isinstance(value, ResolutionStatus):
        return value
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resolved": ResolutionStatus.RESOLVED,
        "partially_resolved": ResolutionStatus.PARTIALLY_RESOLVED,
        "partial": ResolutionStatus.PARTIALLY_RESOLVED,
        "partially": ResolutionStatus.PARTIALLY_RESOLVED,
        "unresolved": ResolutionStatus.UNRESOLVED,
        # Hyphens and spaces are normalised to underscores above, so the
        # hyphenated and spaced spellings arrive here too.
        "un_resolved": ResolutionStatus.UNRESOLVED,
        "not_resolved": ResolutionStatus.UNRESOLVED,
        "open": ResolutionStatus.UNRESOLVED,
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(
        f"unrecognised resolution status {value!r}. Expected one of: "
        "resolved, partially_resolved, unresolved."
    )


# ---------------------------------------------------------------------------
# Temporal evidence
# ---------------------------------------------------------------------------


class TemporalEvidence(BaseModel):
    """
    Optional, order-related evidence.  Never required, never decisive.

    What a solution came *after* the issue is a precondition for it being an
    answer: a "fix" posted before the complaint existed cannot address it.  That
    is a sanity check, not a proof.  Chronological order carries no information
    about whether the fix *worked*, and this stage never claims it does.

    ``follow_up_count`` and ``follow_up_resolution_signal`` are the fields that
    touch effectiveness, and even then weakly - a follow-up comment count is a
    proxy for engagement, not an outcome.  In V1 all three are echoed and
    reported and none of them enters :attr:`resolution_score`, because no
    measurement justifies a weight for them.  The weights for temporal evidence
    become a question for a model that has annotated outcomes to fit, and until
    then the honest thing is to preserve the data and say it was not used.
    """

    #: Did at least one linked solution appear after the issue was raised?
    solution_after_issue: bool = False
    #: Comments by the original poster after the issue was raised.
    follow_up_count: int = 0
    #: Did any follow-up carry a resolution signal ("that worked", "fixed it")?
    follow_up_resolution_signal: bool = False

    @property
    def has_any(self) -> bool:
        """True when any temporal field was actually populated."""
        return bool(
            self.solution_after_issue
            or self.follow_up_count
            or self.follow_up_resolution_signal
        )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class IssueResolutionEvidence(BaseModel):
    """
    One issue's resolution evidence: counts and scores about a single issue,
    within a single discussion.

    Neutral by construction
    ------------------------
    This model deliberately has no reference to any other stage's types.  It
    holds numbers, and whoever produced them - a linker, a counter, a script, a
    human reading the thread - is not this stage's business.  That is what makes
    the stage independently testable and independently replaceable.

    Multi-issue documents
    ---------------------
    An issue document may carry several labels (``["fees", "roi"]``).  **One
    instance of this model is one (issue, label) pair**, and resolution is
    evaluated per instance.  A "fees" answer and an "roi" answer are different
    evidence for different questions, and averaging them into one verdict for the
    document is how a solved fees problem gets recorded as a solved ROI problem.

    The counts
    ----------
    ``total_comments`` is the denominator of every density, and it is the single
    most important field for interpreting the result: 1 comment with 1 linked
    solution has density 1.0 and tells you almost nothing.  See
    :attr:`ResolutionModelConfig.minimum_discussion_volume`.

    Counts describing responses (``agreement_count`` and friends) are **not** used
    in V1 scoring, for the same reason temporal evidence is not: no measurement
    justifies a weight.  They are preserved because they are the obvious inputs to
    a future learned model, and dropping them now would mean re-collecting them
    later.  Their presence here is a statement about V2, not a claim about V1.
    """

    # -- identity ---------------------------------------------------------------
    issue_id: str
    issue_label: str
    post_id: Optional[str] = None

    # -- discussion volume ------------------------------------------------------
    #: Comments in the discussion.  The denominator of every density.
    total_comments: int = 0
    #: Defaults to ``total_comments`` when not supplied.
    discussion_volume: Optional[int] = None
    #: Comments the solution stage assessed at all, whether or not they were
    #: linked.  A count of *candidates*; see ``linked_solution_count``.
    solution_candidate_count: int = 0

    # -- solution coverage (the only evidence that contributes to the score) ----
    #: Solutions linked to **this issue and this label**, not merely present in
    #: the thread.  An unrelated solution in the same discussion must not raise
    #: this number, and counting it here is the single easiest way to make a
    #: linker mistake look like a resolution.
    linked_solution_count: int = 0
    #: Of those, how many the linker considered strong.  Supplied by the caller
    #: (or derived from ``link_scores``).
    strong_link_count: int = 0
    #: Supplied density, for callers who computed it.  Validated against
    #: [0, 1] and, when given, cross-checked against the derived value.
    solution_density: Optional[float] = None

    # -- link strength ----------------------------------------------------------
    #: Highest-scoring link for this issue, or 0.0 when there are none.
    best_link_score: Optional[float] = None
    #: Mean of the links for this issue, or 0.0 when there are none.
    mean_link_score: Optional[float] = None
    #: Per-link scores.  When present they take precedence over the summary
    #: fields above, so the caller does not have to summarise them twice.
    link_scores: Optional[List[float]] = None

    # -- detector confidence ----------------------------------------------------
    #: Mean solution probability over the linked solutions, or 0.0.
    solution_probability_mean: Optional[float] = None
    #: Highest solution probability among them, or 0.0.
    solution_probability_max: Optional[float] = None
    solution_probabilities: Optional[List[float]] = None

    # -- response roles ---------------------------------------------------------
    #: Raw histogram, e.g. ``{"solution": 2, "opinion": 4}``.  Never branched on;
    #: not a score input in V1.
    response_counts: Dict[str, int] = Field(default_factory=dict)
    agreement_count: int = 0
    experience_count: int = 0
    opinion_count: int = 0
    advice_count: int = 0
    solution_count: int = 0
    counterargument_count: int = 0

    # -- optional temporal ------------------------------------------------------
    temporal: Optional[TemporalEvidence] = None

    # -- helpers ----------------------------------------------------------------

    @property
    def effective_discussion_volume(self) -> int:
        """``discussion_volume`` when given, otherwise ``total_comments``."""
        if self.discussion_volume is not None:
            return int(self.discussion_volume)
        return int(self.total_comments)

    @property
    def has_linked_solutions(self) -> bool:
        return self.linked_solution_count > 0

    def pair_key(self) -> str:
        """Stable identity of this (issue, label) pair."""
        return f"{self.issue_id}|{self.issue_label}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ResolutionModelConfig(BaseModel):
    """
    Configuration for the evidence-based resolution scorer.

    Every number is here and nothing is buried in the implementation, because a
    resolution verdict is a claim a reader will act on ("this issue is still
    open, go answer it") and that claim has to be reproducible from its
    configuration.

    The score
    ---------
    Four weighted terms, each in [0, 1], with the weights normalised to sum to 1::

        resolution_score = w_density * solution_density
                        + w_strong   * strong_solution_density
                        + w_strength * best_link_score
                        + w_conf     * solution_probability_mean

    where::

        solution_density        = linked_solution_count / max(total_comments, 1)
        strong_solution_density = strong_link_count    / max(total_comments, 1)

    The weights are a **judgement call, not a measurement.**  They were not
    tuned, because tuning them requires annotated resolution outcomes and none
    exist.  Density and strong density are weighted above link strength on
    purpose: *how many* relevant solutions a discussion attracted is harder to
    fake by accident than *how lexically similar* any single one is, and a
    lexical term alone would let one word-overlap carry a verdict.

    Thresholds
    ----------
    Two effective cut-points, three named thresholds::

        score < unresolved_upper              -> unresolved
        unresolved_upper <= score < resolved_lower -> partially_resolved
        score >= resolved_lower               -> resolved

    ``partial_upper`` is validated to lie within ``[unresolved_upper,
    resolved_lower]`` and is reported in the configuration as the nominal
    partial/resolved boundary.  Three names for two cut-points is inherent to
    naming all three bands: the partially-resolved band needs a lower and an
    upper edge, and the upper edge is the same number as the resolved band's
    lower edge.  A configuration that sets them inconsistently is rejected rather
    than silently resolved.
    """

    # --- score weights ----------------------------------------------------------
    solution_density_weight: float = 0.30
    strong_link_weight: float = 0.20
    link_strength_weight: float = 0.25
    solution_confidence_weight: float = 0.25
    #: Divide the weights by their sum so the score stays on 0-1.  Turning it off
    #: makes the score no longer comparable to the thresholds.
    normalise_weights: bool = True

    # --- thresholds -------------------------------------------------------------
    #: Below this score the issue is ``unresolved``.
    unresolved_upper: float = 0.30
    #: Nominal upper edge of the partially-resolved band.  Must satisfy
    #: ``unresolved_upper <= partial_upper <= resolved_lower``.
    partial_upper: float = 0.55
    #: At or above this score the issue is ``resolved``.
    resolved_lower: float = 0.55

    # --- discussion volume ------------------------------------------------------
    #: Minimum ``effective_discussion_volume`` for a verdict to be treated as
    #: supported by evidence.  Below it the status is still reported as one of the
    #: three required states, but ``evidence_sufficient`` is False, a warning is
    #: attached, and confidence is scaled down (see
    #: ``insufficient_evidence_confidence_scale``).
    minimum_discussion_volume: int = 3

    # --- confidence -------------------------------------------------------------
    #: Multiplier applied to confidence when ``evidence_sufficient`` is False.
    #: Only the *confidence* is reduced, never the status: a thin thread is not
    #: thereby declared unresolved, it is declared less certainly judged.  Silently
    #: capping the score instead would manufacture a verdict out of the guard.
    insufficient_evidence_confidence_scale: float = 0.5

    # --- linking thresholds passed through for reporting -------------------------
    #: The linker's own notion of a "strong" link, echoed into the result so a
    #: reader can see the criterion behind ``strong_link_count`` without opening
    #: the linker.  Not used to reclassify anything here.
    strong_link_threshold: float = 0.6

    # --- reporting ---------------------------------------------------------------
    #: Opt-in only: a wall-clock field would break reproducibility.
    include_timestamp: bool = False

    def weight_map(self) -> Dict[str, float]:
        """The four weights, keyed by the evidence term they multiply."""
        return {
            "solution_density": float(self.solution_density_weight),
            "strong_solution_density": float(self.strong_link_weight),
            "best_link_score": float(self.link_strength_weight),
            "solution_probability_mean": float(self.solution_confidence_weight),
        }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ResolutionDetection(ABC):
    """
    Abstract interface for Resolution Detection.

    Any implementation must:

    - accept :class:`IssueResolutionEvidence` and mutate neither the records nor
      the caller's list;
    - produce a status from exactly :data:`STATUS_ORDER` - never a fourth state,
      never ``None``, and never an exception for a thin thread;
    - report ``evidence_sufficient`` and a warning whenever the discussion was
      too thin to support the verdict;
    - report ``resolution_score`` and ``confidence`` as the deterministic
      quantities they are, with no claim of calibration;
    - carry ``model_name`` and ``model_version`` on every result.

    V1 is not a fitted model, so there is no ``train()``.  A future learned
    scorer would add one behind this same interface, and callers would not
    change.
    """

    @abstractmethod
    def detect(
        self,
        evidence: Sequence[IssueResolutionEvidence],
    ) -> Any:
        """Assess resolution for each issue; return a structured result."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"evidence_based_resolution"``."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""
        ...

    @property
    @abstractmethod
    def config(self) -> ResolutionModelConfig:
        ...

    @property
    @abstractmethod
    def thresholds(self) -> Dict[str, float]:
        """The threshold values in force, for echoing onto every result."""
        ...

    @property
    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob that affects the output."""
        ...
