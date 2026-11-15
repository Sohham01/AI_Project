"""
Result structures for issue identification.

Everything a downstream consumer needs in order to act on, audit or re-score a
prediction:

- per-document predicted labels and the *full* per-label probability vector
  (so a consumer can re-threshold later without re-running the model)
- an overall confidence per document
- model name / version and the threshold + configuration in force
- labels the model could not possibly emit (taxonomy drift detection)
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IssuePrediction(BaseModel):
    """
    Multi-label prediction for one document.

    ``issue_probabilities`` contains EVERY label the model knows, not just the
    ones above threshold, so downstream code can raise or lower the decision
    boundary without re-running inference.  ``predicted_issue_labels``
    contains the labels whose probability was strictly greater than the
    threshold in force.
    """

    document_id: str
    predicted_issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0

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

    @property
    def has_issue(self) -> bool:
        return len(self.predicted_issue_labels) > 0

    @property
    def is_multi_label(self) -> bool:
        return len(self.predicted_issue_labels) > 1

    def top_label(self) -> Optional[str]:
        """Highest-probability label regardless of threshold (or None)."""
        if not self.issue_probabilities:
            return None
        return max(
            sorted(self.issue_probabilities.items()),
            key=lambda kv: kv[1],
        )[0]

    def labels_above(self, threshold: float) -> List[str]:
        """Re-derive the predicted set at a different threshold, no re-inference."""
        return sorted(
            label
            for label, probability in self.issue_probabilities.items()
            if probability > threshold
        )


class IssueIdentificationResult(BaseModel):
    """Structured output of one prediction run."""

    model_name: str
    model_version: str

    threshold: float
    predictions: List[IssuePrediction] = Field(default_factory=list)

    n_documents: int = 0
    #: The label set the model can emit (learned from annotation data).
    known_labels: List[str] = Field(default_factory=list)
    #: Labels supplied by the caller that this model was never trained on.
    #: They are never predicted; they are reported so taxonomy drift between
    #: annotation rounds is visible instead of silent.
    unseen_labels: List[str] = Field(default_factory=list)
    #: Documents for which no label cleared the threshold.
    n_documents_without_issue: int = 0
    #: Documents with more than one predicted label.
    n_multi_label_documents: int = 0

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def prediction_for(self, document_id: str) -> Optional[IssuePrediction]:
        """First prediction for ``document_id``, or None."""
        for prediction in self.predictions:
            if prediction.document_id == document_id:
                return prediction
        return None

    def label_frequency(self) -> Dict[str, int]:
        """How many documents were assigned each label (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for prediction in self.predictions:
            for label in prediction.predicted_issue_labels:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))


class IssueTrainingReport(BaseModel):
    """
    What was actually learned by ``train()``.

    Reported so a reviewer can see the label set, the evidence per label, and
    the data-adequacy checks that passed — instead of assuming the model is
    trustworthy because it trained without crashing.
    """

    model_name: str
    model_version: str

    n_training_documents: int
    n_labels: int
    labels: List[str] = Field(default_factory=list)
    #: Annotated positive examples available per label.
    positive_examples_per_label: Dict[str, int] = Field(default_factory=dict)
    #: Distinct annotated terms the model can key on.
    n_features: int = 0
    #: Documents annotated with no issue at all.
    n_documents_without_labels: int = 0
    #: Documents carrying more than one issue.
    n_multi_label_documents: int = 0

    threshold: float = 0.5
    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A model that passed the data-adequacy guards."""
        return self.n_labels >= 2 and self.n_training_documents > 0
