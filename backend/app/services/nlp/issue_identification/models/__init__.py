"""
Result and annotation structures for the issue identification stage.
"""

from app.services.nlp.issue_identification.models.annotation import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    MULTI_VALUE_SEPARATOR,
    AnnotationCorpus,
    IssueAnnotation,
)
from app.services.nlp.issue_identification.models.evaluation import (
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    MIN_LABELS_FOR_MEANINGFUL_METRICS,
    MIN_POSITIVES_FOR_STABLE_PER_LABEL_F1,
    IssueEvaluationReport,
    evaluate_predictions,
)
from app.services.nlp.issue_identification.models.result import (
    IssueIdentificationResult,
    IssuePrediction,
    IssueTrainingReport,
)

__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_ANNOTATION_CSV",
    "MULTI_VALUE_SEPARATOR",
    "IssueAnnotation",
    "AnnotationCorpus",
    "IssuePrediction",
    "IssueIdentificationResult",
    "IssueTrainingReport",
    "IssueEvaluationReport",
    "evaluate_predictions",
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_LABELS_FOR_MEANINGFUL_METRICS",
    "MIN_POSITIVES_FOR_STABLE_PER_LABEL_F1",
]
