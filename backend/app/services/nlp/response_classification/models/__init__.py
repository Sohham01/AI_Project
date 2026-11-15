"""
Results and evaluation models for response classification.
"""

from app.services.nlp.response_classification.models.annotation import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    ResponseAnnotation,
    ResponseCorpus,
    is_valid_class,
)
from app.services.nlp.response_classification.models.evaluation import (
    MIN_CLASSES_FOR_MEANINGFUL_METRICS,
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1,
    ResponseEvaluationReport,
    evaluate_response_classification,
)
from app.services.nlp.response_classification.models.labels import (
    EXPECTED_NUMBER_OF_RESPONSE_TYPES,
    RESPONSE_TYPE_CODES,
    RESPONSE_TYPE_DESCRIPTIONS,
    RESPONSE_TYPES,
    is_valid_response_type,
    response_type_order,
    unknown_response_types,
    validate_response_type,
)
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponsePrediction,
    ResponseTrainingReport,
)

__all__ = [
    # annotation
    "ResponseAnnotation",
    "ResponseCorpus",
    "CSV_COLUMNS",
    "DEFAULT_ANNOTATION_CSV",
    "is_valid_class",
    # labels
    "RESPONSE_TYPES",
    "RESPONSE_TYPE_CODES",
    "RESPONSE_TYPE_DESCRIPTIONS",
    "EXPECTED_NUMBER_OF_RESPONSE_TYPES",
    "is_valid_response_type",
    "validate_response_type",
    "unknown_response_types",
    "response_type_order",
    # results
    "ResponsePrediction",
    "ResponseClassificationResult",
    "ResponseTrainingReport",
    # evaluation
    "ResponseEvaluationReport",
    "evaluate_response_classification",
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_CLASSES_FOR_MEANINGFUL_METRICS",
    "MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1",
]
