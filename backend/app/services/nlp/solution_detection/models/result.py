"""
Result structures for solution detection.

The central design point is that a prediction carries **three separate facts**,
and they are not allowed to blur into each other:

1. ``solution_probability`` - what the model actually computed
2. ``is_solution`` - the boolean a consumer acts on, derived from the threshold
3. ``is_candidate`` / ``candidate_assessment`` - whether the comment was even
   worth testing, derived from an upstream ``response_type``

(3) is a triage signal, not evidence.  A comment can be a candidate and get
``is_solution=False`` - that is the ``advice`` case, and it is expected rather
than a bug.  Reporting the disagreement explicitly, as
``n_candidates_not_solutions``, is more useful than hiding it: it is a direct
measure of how much the two stages' notions of "helpful" diverge, which is
exactly what an annotation guideline needs to know.

Why the probability is reported alongside the boolean
-----------------------------------------------------
``is_solution`` is a hard claim that a downstream linker will turn into an
Issue-Solution edge.  A hard claim made at 0.51 and the same claim made at 0.99
look identical downstream but do not deserve equal trust, so the probability and
the threshold in force travel with every prediction.  :meth:`reclassify_at`
re-derives the boolean at a different threshold without re-running inference, so
a consumer can trade precision for recall on data it already has.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.services.nlp.solution_detection.models.candidate import (
    CandidateAssessment,
    classify_response_candidate,
)
from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    TARGET_LABELS,
)


class SolutionPrediction(BaseModel):
    """
    Binary solution prediction for one comment.

    ``solution_probability + not_solution_probability == 1`` (within 1e-6 of
    rounding), because the target is binary and the two outcomes are exhaustive.

    ``is_solution`` is ``solution_probability >= threshold``.  When the comment
    falls inside the opt-in abstention band (``threshold <= p < abstain_threshold``)
    it is reported ``is_solution=False, abstained=True``: the boundary leans yes,
    but not confidently enough to assert a link.
    """

    document_id: str

    #: The boolean a downstream consumer acts on.  Never inferred from
    #: ``response_type``.
    is_solution: bool = False
    #: P(y = solution | x) = sigmoid(w.x + b).
    solution_probability: float = 0.0
    #: P(y = not_solution | x) = 1 - solution_probability.
    not_solution_probability: float = 0.0
    #: Decision boundary applied to ``solution_probability``.
    threshold: float = 0.5
    #: True when the probability was inside the abstention band.
    abstained: bool = False

    # Response context: echoed, and used only for candidate triage
    response_type: Optional[str] = None
    response_probabilities: Dict[str, float] = Field(default_factory=dict)
    #: True when the upstream response role marks this as worth testing.
    is_candidate: bool = False

    # Echoed metadata for traceability (never modified by the model)
    text: str = ""
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_keywords: List[str] = Field(default_factory=list)
    issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)
    parent_comment_text: Optional[str] = None

    # -- Candidate ----------------------------------------------------

    def candidate_assessment(self) -> CandidateAssessment:
        """
        Full candidate assessment for this comment's response role.

        Returns a record whose ``is_solution`` is the **detector's verdict**
        attached via :meth:`CandidateAssessment.with_prediction`, never the role.
        """
        return classify_response_candidate(self.response_type).with_prediction(
            self.is_solution
        )

    @property
    def candidate_label(self) -> str:
        """"solution", "advice" or "other" - the role's candidate bucket."""
        if self.is_candidate:
            return self.response_type or "unknown_candidate"
        return "other"

    # -- Threshold re-derivation --------------------------------------

    def reclassify_at(
        self,
        threshold: float,
        abstain_threshold: Optional[float] = None,
    ) -> bool:
        """
        Re-derive ``is_solution`` at a different threshold, without re-inference.

        Lets a consumer explore the precision/recall trade-off on probabilities
        it already has, instead of re-running the model for every cutoff.
        """
        is_solution = self.solution_probability >= threshold
        if is_solution and abstain_threshold is not None:
            if self.solution_probability < abstain_threshold:
                return False
        return is_solution

    @property
    def is_weak_positive(self) -> bool:
        """Positive boundary, but inside the abstention band."""
        return self.abstained

    @property
    def predicted_label(self) -> str:
        """Canonical label name for the emitted boolean."""
        return LABEL_SOLUTION if self.is_solution else LABEL_NOT_SOLUTION


class SolutionDetectionResult(BaseModel):
    """Structured output of one solution-detection run."""

    model_name: str
    model_version: str

    #: Decision boundary applied to P(solution) on this run.
    threshold: float
    #: Opt-in abstention band in force, or ``None`` when abstention is disabled.
    abstain_threshold: Optional[float] = None
    predictions: List[SolutionPrediction] = Field(default_factory=list)

    n_documents: int = 0
    #: Comments with ``is_solution=True``.
    n_solutions: int = 0
    #: Comments with ``is_solution=False`` (abstentions included).
    n_not_solutions: int = 0
    #: Comments inside the abstention band: positive-leaning, not asserted.
    n_abstained: int = 0
    #: Comments where a response role marked the comment a candidate.
    n_candidates: int = 0
    #: Candidates the detector rejected - the ``advice`` that was not a
    #: solution.  The headline number for "advice != solution".
    n_candidates_not_solutions: int = 0
    #: The reverse disagreement: a role was supplied, and it is not a candidate
    #: role, yet the detector flagged a solution.  Usually means the response
    #: role is mislabelled, or the comment is solution-shaped without being
    #: advice-shaped.  Documents with **no** response role are unassessed rather
    #: than contradicted, and are not counted here.
    n_solutions_not_candidates: int = 0
    #: Comments for which optional thread context was supplied (not read by V1).
    n_documents_with_context: int = 0
    #: Comments for which a response role was supplied.
    n_documents_with_response_context: int = 0

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def prediction_for(self, document_id: str) -> Optional[SolutionPrediction]:
        """First prediction for ``document_id``, or None."""
        for prediction in self.predictions:
            if prediction.document_id == document_id:
                return prediction
        return None

    def solution_documents(self) -> List[SolutionPrediction]:
        """Predictions asserted as solutions."""
        return [p for p in self.predictions if p.is_solution]

    def candidate_documents(self) -> List[SolutionPrediction]:
        """Predictions whose response role made them candidates."""
        return [p for p in self.predictions if p.is_candidate]

    def candidates_rejected(self) -> List[SolutionPrediction]:
        """Candidates the detector said were not solutions."""
        return [
            p for p in self.predictions if p.is_candidate and not p.is_solution
        ]

    def average_probability(self) -> float:
        """Mean P(solution) across the run; 0.0 for an empty run."""
        if not self.predictions:
            return 0.0
        total = sum(p.solution_probability for p in self.predictions)
        return round(total / len(self.predictions), 6)

    def solution_rate(self) -> float:
        """Share of comments asserted as solutions; 0.0 for an empty run."""
        if not self.predictions:
            return 0.0
        return round(self.n_solutions / len(self.predictions), 6)

    def label_frequency(self) -> Dict[str, int]:
        """How many comments got each canonical label (deterministic order)."""
        counts = {LABEL_NOT_SOLUTION: self.n_not_solutions, LABEL_SOLUTION: self.n_solutions}
        return {name: counts[name] for name in TARGET_LABELS}


class SolutionTrainingReport(BaseModel):
    """
    What was actually learned by ``train()``.

    Reported so a reviewer can see the target distribution, which
    ``solution_type`` categories exist in the data (and that none were learned),
    and which data-adequacy checks passed - instead of assuming the model is
    trustworthy because it trained without crashing.
    """

    model_name: str
    model_version: str

    n_training_documents: int
    #: Always 2 when training succeeded: the target is binary.
    n_classes: int = 2
    #: Canonical target order, matching the confusion-matrix axis order.
    labels: List[str] = Field(default_factory=list)
    #: Annotations per target, canonical order.
    examples_per_class: Dict[str, int] = Field(default_factory=dict)

    #: Share of the corpus holding the more frequent target.
    majority_class_share: float = 0.0
    #: Which target is the majority, named so the warning is actionable.
    majority_class: Optional[str] = None

    #: Free-form ``solution_type`` values seen in the annotations.  Reported for
    #: coverage only - never a target and never a feature in V1.
    observed_solution_types: List[str] = Field(default_factory=list)
    solution_type_counts: Dict[str, int] = Field(default_factory=dict)

    #: Distinct annotated terms the model can key on.
    n_features: int = 0

    threshold: float = 0.5
    abstain_threshold: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A model that passed the data-adequacy guards."""
        return self.n_classes == 2 and self.n_training_documents > 0

    @property
    def n_positive(self) -> int:
        return self.examples_per_class.get(LABEL_SOLUTION, 0)

    @property
    def n_negative(self) -> int:
        return self.examples_per_class.get(LABEL_NOT_SOLUTION, 0)

    @property
    def is_imbalanced(self) -> bool:
        """True when one target holds more than the configured share."""
        return self.majority_class_share > 0.5

    @property
    def learns_solution_types(self) -> bool:
        """
        Always ``False`` in V1.

        Stated as a property so a test can assert it and a reviewer can read it:
        ``solution_type`` is annotation data, not a target.
        """
        return False
