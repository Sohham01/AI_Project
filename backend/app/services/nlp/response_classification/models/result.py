"""
Result structures for response classification.

Everything a downstream consumer needs in order to act on, audit or re-score a
prediction:

- the predicted response type plus the **full** probability vector over the
  classes the model can emit, so a consumer can lower or raise the confidence
  floor later without re-running inference
- per-comment confidence and an explicit abstention flag
- model name / version and the threshold + configuration in force
- the response types the model was never trained on, and the ones the caller
  asked about that it cannot emit (taxonomy / data drift)

On the probability vector: it covers ``known_response_types``, which is the set
the model can actually emit.  When a response type has no training examples the
model cannot assign it a probability, so it is reported in
``missing_response_types`` rather than being given a fabricated ``0.0``.  A
consumer that needs all six must check that field.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResponsePrediction(BaseModel):
    """
    Single-label response prediction for one comment.

    ``response_probabilities`` contains EVERY class this model can emit, and the
    values sum to 1 because the six roles are mutually exclusive.  The
    prediction is the arg-max, or ``None`` when its probability falls below the
    confidence floor in force - the model abstains rather than guessing a
    discussion role, because a wrong role is worse than an absent one downstream
    (an "advice" comment mistaken for a "solution" would be treated as evidence
    that the problem was answered).

    Values are rounded to 6 decimal places, so a sum may land within 1e-6 of
    1.0 rather than exactly on it.
    """

    document_id: str
    response_type: Optional[str] = None
    #: Winning class probability, or the top class' probability when abstained.
    confidence: float = 0.0
    response_probabilities: Dict[str, float] = Field(default_factory=dict)
    #: True when the winning class did not clear the confidence floor.
    abstained: bool = False

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

    @property
    def has_response_type(self) -> bool:
        return self.response_type is not None

    def top_response_type(self) -> Optional[str]:
        """
        Highest-probability class regardless of threshold (or None).

        This is the arg-max even when the prediction abstained, so a consumer can
        see what the model leaned towards and why it declined to commit.
        """
        if not self.response_probabilities:
            return None
        return max(
            sorted(self.response_probabilities.items()),
            key=lambda kv: kv[1],
        )[0]

    def response_type_at(self, threshold: float) -> Optional[str]:
        """Re-derive the class at a different confidence floor, no re-inference."""
        top = self.top_response_type()
        if top is None:
            return None
        return top if self.response_probabilities[top] > threshold else None

    def probability_of(self, response_type: str) -> Optional[float]:
        """Probability assigned to one class, or None if the model never learned it."""
        return self.response_probabilities.get(response_type)


class ResponseClassificationResult(BaseModel):
    """Structured output of one response-classification run."""

    model_name: str
    model_version: str

    #: Confidence floor applied to the winning class, not a class boundary.
    threshold: float
    predictions: List[ResponsePrediction] = Field(default_factory=list)

    n_documents: int = 0
    #: The response types this model can emit, in canonical V1 order.
    known_response_types: List[str] = Field(default_factory=list)
    #: V1 response types this model has no training data for, so it can never
    #: emit them.  Reported rather than hidden, so a downstream consumer does not
    #: treat their absence as a negative prediction.
    missing_response_types: List[str] = Field(default_factory=list)
    #: Response types the model knows but never predicted on this run.
    unpredicted_response_types: List[str] = Field(default_factory=list)
    #: Comments where the winning class was below the confidence floor.
    n_abstained: int = 0
    #: Comments for which optional thread context was supplied.  Carried for
    #: traceability only: the V1 classifier does not read it.
    n_documents_with_context: int = 0

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def prediction_for(self, document_id: str) -> Optional[ResponsePrediction]:
        """First prediction for ``document_id``, or None."""
        for prediction in self.predictions:
            if prediction.document_id == document_id:
                return prediction
        return None

    def response_type_frequency(self) -> Dict[str, int]:
        """How many comments were assigned each type (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for prediction in self.predictions:
            if prediction.response_type is None:
                continue
            counts[prediction.response_type] = (
                counts.get(prediction.response_type, 0) + 1
            )
        return dict(sorted(counts.items()))

    def average_confidence(self) -> float:
        """Mean top-class probability across the run; 0.0 for an empty run."""
        if not self.predictions:
            return 0.0
        total = sum(p.confidence for p in self.predictions)
        return round(total / len(self.predictions), 6)


class ResponseTrainingReport(BaseModel):
    """
    What was actually learned by ``train()``.

    Reported so a reviewer can see the class set, the evidence per class, which
    of the six V1 types are still missing, and the data-adequacy checks that
    passed - instead of assuming the model is trustworthy because it trained
    without crashing.
    """

    model_name: str
    model_version: str

    n_training_documents: int
    #: Number of response types the model can emit (a subset of the six).
    n_classes: int
    #: The classes learned, in canonical V1 order.
    labels: List[str] = Field(default_factory=list)
    #: All six V1 response types, for comparison with ``labels``.
    expected_labels: List[str] = Field(default_factory=list)
    #: V1 response types absent from the training data.
    missing_response_types: List[str] = Field(default_factory=list)
    #: Annotated documents available per class.
    examples_per_class: Dict[str, int] = Field(default_factory=dict)
    #: Share of the corpus held by the most frequent class.
    majority_class_share: float = 0.0
    #: Most frequent class, named so the imbalance warning is actionable.
    majority_class: Optional[str] = None
    #: Distinct annotated terms the model can key on.
    n_features: int = 0
    #: Number of classes per comment (always 1 - stated for clarity).
    classes_per_document: int = 1

    threshold: float = 0.5
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A model that passed the data-adequacy guards."""
        return self.n_classes >= 2 and self.n_training_documents > 0

    @property
    def covers_full_taxonomy(self) -> bool:
        """True when the model can emit all six V1 response types."""
        return not self.missing_response_types
