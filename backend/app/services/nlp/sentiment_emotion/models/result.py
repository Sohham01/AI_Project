"""
Result structures for sentiment and emotion analysis.

Everything a downstream consumer needs in order to act on, audit or re-score a
prediction:

- the predicted class / label set plus the **full** probability vector, so a
  consumer can re-threshold later without re-running inference
- per-document confidence
- model name / version and the threshold + configuration in force
- classes or labels the model could not possibly emit (taxonomy drift)

The two dimensions keep separate probability vectors, and that separation is
the point: a consumer receives ``sentiment_probabilities`` and
``emotion_probabilities`` as independent evidence, and is expected to treat
them as such rather than deriving one from the other.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SentimentPrediction(BaseModel):
    """
    Single-label sentiment prediction for one document.

    ``sentiment_probabilities`` contains EVERY class the model knows, and the
    values sum to 1 because sentiment is mutually exclusive.  ``sentiment`` is
    the winning class, or ``None`` when its probability is below the confidence
    threshold in force — the model abstains instead of guessing, rather than
    substituting a fallback class that the annotation data never taught it.
    """

    document_id: str
    sentiment: Optional[str] = None
    sentiment_probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    #: True when no class cleared the confidence floor.
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

    @property
    def has_sentiment(self) -> bool:
        return self.sentiment is not None

    def top_sentiment(self) -> Optional[str]:
        """Highest-probability class regardless of threshold (or None)."""
        if not self.sentiment_probabilities:
            return None
        return max(
            sorted(self.sentiment_probabilities.items()),
            key=lambda kv: kv[1],
        )[0]

    def sentiment_at(self, threshold: float) -> Optional[str]:
        """Re-derive the class at a different confidence floor, no re-inference."""
        top = self.top_sentiment()
        if top is None:
            return None
        return top if self.sentiment_probabilities[top] > threshold else None


class EmotionPrediction(BaseModel):
    """
    Multi-label emotion prediction for one document.

    ``emotion_probabilities`` contains every emotion the model knows.  These are
    **independent** One-vs-Rest sigmoid outputs: they do not sum to 1, and
    several can exceed the threshold at once.  ``emotions`` holds those that
    did.

    A document whose emotions list is empty means "no emotion above the
    threshold", which is a valid outcome and not a prediction failure.
    """

    document_id: str
    emotions: List[str] = Field(default_factory=list)
    emotion_probabilities: Dict[str, float] = Field(default_factory=dict)
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
    topic_keywords: List[str] = Field(default_factory=list)
    issue_labels: List[str] = Field(default_factory=list)
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)

    @property
    def has_emotion(self) -> bool:
        return len(self.emotions) > 0

    @property
    def is_multi_label(self) -> bool:
        return len(self.emotions) > 1

    def top_emotion(self) -> Optional[str]:
        """Highest-probability emotion regardless of threshold (or None)."""
        if not self.emotion_probabilities:
            return None
        return max(
            sorted(self.emotion_probabilities.items()),
            key=lambda kv: kv[1],
        )[0]

    def emotions_above(self, threshold: float) -> List[str]:
        """Re-derive the label set at a different threshold, no re-inference."""
        return sorted(
            emotion
            for emotion, probability in self.emotion_probabilities.items()
            if probability > threshold
        )


class SentimentAnalysisResult(BaseModel):
    """Structured output of one sentiment prediction run."""

    model_name: str
    model_version: str

    #: Confidence floor applied to the winning class, not a class boundary.
    threshold: float
    predictions: List[SentimentPrediction] = Field(default_factory=list)

    n_documents: int = 0
    #: The class set the model can emit (learned from annotation data).
    known_sentiments: List[str] = Field(default_factory=list)
    #: Classes supplied by the caller that this model never trained on.
    unseen_sentiments: List[str] = Field(default_factory=list)
    #: Documents where the winning class was below the confidence floor.
    n_abstained: int = 0

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def prediction_for(self, document_id: str) -> Optional[SentimentPrediction]:
        """First prediction for ``document_id``, or None."""
        for prediction in self.predictions:
            if prediction.document_id == document_id:
                return prediction
        return None

    def sentiment_frequency(self) -> Dict[str, int]:
        """How many documents were assigned each class (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for prediction in self.predictions:
            if prediction.sentiment is None:
                continue
            counts[prediction.sentiment] = counts.get(prediction.sentiment, 0) + 1
        return dict(sorted(counts.items()))

    def average_confidence(self) -> float:
        """Mean top-class probability across the run; 0.0 for an empty run."""
        if not self.predictions:
            return 0.0
        total = sum(p.confidence for p in self.predictions)
        return round(total / len(self.predictions), 6)


class EmotionAnalysisResult(BaseModel):
    """Structured output of one emotion prediction run."""

    model_name: str
    model_version: str

    #: Per-label decision boundary on each independent sigmoid.
    threshold: float
    predictions: List[EmotionPrediction] = Field(default_factory=list)

    n_documents: int = 0
    #: The label set the model can emit (learned from annotation data).
    known_emotions: List[str] = Field(default_factory=list)
    #: Labels supplied by the caller that this model never trained on.  They are
    #: never predicted; they are reported so taxonomy drift between annotation
    #: rounds is visible instead of silent.
    unseen_labels: List[str] = Field(default_factory=list)
    #: Documents for which no emotion cleared the threshold.
    n_documents_without_emotion: int = 0
    #: Documents carrying more than one emotion.
    n_multi_label_documents: int = 0

    config: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def prediction_for(self, document_id: str) -> Optional[EmotionPrediction]:
        """First prediction for ``document_id``, or None."""
        for prediction in self.predictions:
            if prediction.document_id == document_id:
                return prediction
        return None

    def emotion_frequency(self) -> Dict[str, int]:
        """How many documents were assigned each emotion (deterministically ordered)."""
        counts: Dict[str, int] = {}
        for prediction in self.predictions:
            for emotion in prediction.emotions:
                counts[emotion] = counts.get(emotion, 0) + 1
        return dict(sorted(counts.items()))

    def average_confidence(self) -> float:
        """Mean top-emotion probability across the run; 0.0 for an empty run."""
        if not self.predictions:
            return 0.0
        total = sum(p.confidence for p in self.predictions)
        return round(total / len(self.predictions), 6)


class SentimentTrainingReport(BaseModel):
    """
    What was actually learned by the sentiment model's ``train()``.

    Reported so a reviewer can see the class set, the evidence per class, and
    the data-adequacy checks that passed — instead of assuming the model is
    trustworthy because it trained without crashing.
    """

    model_name: str
    model_version: str

    n_training_documents: int
    n_classes: int
    labels: List[str] = Field(default_factory=list)
    #: Annotated documents available per class.
    examples_per_class: Dict[str, int] = Field(default_factory=dict)
    #: Share of the corpus held by the most frequent class.
    majority_class_share: float = 0.0
    #: Distinct annotated terms the model can key on.
    n_features: int = 0
    #: Number of mutually exclusive classes (always 1 here, stated for clarity).
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


class EmotionTrainingReport(BaseModel):
    """
    What was actually learned by the emotion model's ``train()``.

    Multi-label evidence is reported per label, plus the counts that make
    multi-label corpora unusual — documents with no emotion and documents with
    several — so a reviewer can see whether the corpus actually exercises the
    multi-label path.
    """

    model_name: str
    model_version: str

    n_training_documents: int
    n_labels: int
    labels: List[str] = Field(default_factory=list)
    #: Annotated positive examples available per emotion.
    positive_examples_per_label: Dict[str, int] = Field(default_factory=list)
    #: Distinct annotated terms the model can key on.
    n_features: int = 0
    #: Documents annotated with no emotion at all.
    n_documents_without_emotions: int = 0
    #: Documents carrying more than one emotion.
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
