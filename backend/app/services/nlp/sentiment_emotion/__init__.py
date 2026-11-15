"""
Sentiment and emotion analysis stage.

Sentiment and emotion are different dimensions, and this package models them
as two separate pipelines that happen to share an input model:

    Sentiment  : the overall polarity of the text -> exactly one class
                 (conventionally positive / negative / neutral)
    Emotion    : the emotional state expressed    -> a *set* of labels, and
                 only where annotation data supports it

    "placement season is horrible"
        sentiment = negative
        emotion   = frustration
    "I was nervous but it worked out well"
        sentiment = positive
        emotion   = anxiety          <- an emotion is NOT implied by the polarity

Why they are not one model:
- Sentiment is mutually exclusive, so its probabilities are a normalised
  distribution and the prediction is an arg-max.  Emotion is multi-label, so it
  uses independent One-vs-Rest sigmoids that do not sum to 1 and can fire several
  at once.  One shared head would force one of those two truths to be wrong.
- **Emotion is never inferred from sentiment.**  Each dimension has its own
  annotation type, ABC, classifier and training run, so nothing can leak
  across.  A negative document legitimately carries no emotion at all, or an
  emotion sentiment would never have suggested.
- Emotion annotation is expensive and subjective, so it is the dimension most
  likely to launch with partial coverage; shipping them separately means a
  project can add emotion without disturbing a working sentiment model.

Both label sets are annotation-driven.  No module here ships an opinion
lexicon, a sentiment word list or a "final" emotion taxonomy: whatever the
annotators recorded is the class set, learned at ``train()`` time, and nothing
in this package can emit a label that annotation data never contained.

Input is already-preprocessed text; this stage neither imports nor re-runs the
preprocessing pipeline, and never mutates raw text, ``normalized_text``, topic
output or issue predictions.

Versioned model identifiers:
    sentiment_model = "tfidf_logistic"              sentiment_model_version = "v1"
    emotion_model   = "tfidf_logistic_multilabel"   emotion_model_version   = "v1"

See docs/sentiment_emotion.md for the annotation CSV formats, the mathematics,
the evaluation caveats, the multilingual limitations, and the roadmap to a
neural classifier.
"""

from app.services.nlp.sentiment_emotion.base import (
    BaseTextFeatureConfig,
    EmotionAnalysis,
    EmotionModelConfig,
    SentimentAnalysis,
    SentimentEmotionDocument,
    SentimentModelConfig,
)
from app.services.nlp.sentiment_emotion.models.annotation import (
    DEFAULT_EMOTION_CSV,
    DEFAULT_SENTIMENT_CSV,
    EMOTION_CSV_COLUMNS,
    MULTI_VALUE_SEPARATOR,
    SENTIMENT_CSV_COLUMNS,
    EmotionAnnotation,
    EmotionCorpus,
    SentimentAnnotation,
    SentimentCorpus,
)
from app.services.nlp.sentiment_emotion.models.evaluation import (
    MIN_CLASSES_FOR_MEANINGFUL_METRICS,
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1,
    EmotionEvaluationReport,
    SentimentEvaluationReport,
    evaluate_emotions,
    evaluate_sentiment,
)
from app.services.nlp.sentiment_emotion.models.result import (
    EmotionAnalysisResult,
    EmotionPrediction,
    EmotionTrainingReport,
    SentimentAnalysisResult,
    SentimentPrediction,
    SentimentTrainingReport,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
    TfidfLogisticEmotionClassifier,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment import (
    TfidfLogisticSentimentClassifier,
)

__all__ = [
    # interfaces + input/config
    "SentimentAnalysis",
    "EmotionAnalysis",
    "SentimentEmotionDocument",
    "BaseTextFeatureConfig",
    "SentimentModelConfig",
    "EmotionModelConfig",
    # annotation
    "SentimentAnnotation",
    "SentimentCorpus",
    "EmotionAnnotation",
    "EmotionCorpus",
    "SENTIMENT_CSV_COLUMNS",
    "EMOTION_CSV_COLUMNS",
    "DEFAULT_SENTIMENT_CSV",
    "DEFAULT_EMOTION_CSV",
    "MULTI_VALUE_SEPARATOR",
    # results
    "SentimentPrediction",
    "SentimentAnalysisResult",
    "SentimentTrainingReport",
    "EmotionPrediction",
    "EmotionAnalysisResult",
    "EmotionTrainingReport",
    # evaluation
    "SentimentEvaluationReport",
    "EmotionEvaluationReport",
    "evaluate_sentiment",
    "evaluate_emotions",
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_CLASSES_FOR_MEANINGFUL_METRICS",
    "MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1",
    # implementations
    "TfidfLogisticSentimentClassifier",
    "TfidfLogisticEmotionClassifier",
]
