"""
Result, annotation and evaluation structures for the sentiment/emotion stage.
"""

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

__all__ = [
    "MULTI_VALUE_SEPARATOR",
    "SENTIMENT_CSV_COLUMNS",
    "EMOTION_CSV_COLUMNS",
    "DEFAULT_SENTIMENT_CSV",
    "DEFAULT_EMOTION_CSV",
    "SentimentAnnotation",
    "SentimentCorpus",
    "EmotionAnnotation",
    "EmotionCorpus",
    "SentimentPrediction",
    "SentimentAnalysisResult",
    "SentimentTrainingReport",
    "EmotionPrediction",
    "EmotionAnalysisResult",
    "EmotionTrainingReport",
    "SentimentEvaluationReport",
    "EmotionEvaluationReport",
    "evaluate_sentiment",
    "evaluate_emotions",
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_CLASSES_FOR_MEANINGFUL_METRICS",
    "MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1",
]
