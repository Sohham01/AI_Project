"""
Results, evaluation and candidate models for solution detection.
"""

from app.services.nlp.solution_detection.models.annotation import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    SolutionAnnotation,
    SolutionCorpus,
)
from app.services.nlp.solution_detection.models.candidate import (
    DEFAULT_CANDIDATE_RESPONSE_TYPES,
    RESPONSE_TYPE_ADVICE,
    RESPONSE_TYPE_SOLUTION,
    CandidateAssessment,
    classify_response_candidate,
    is_advice_response_type,
    is_solution_candidate,
    is_solution_response_type,
)
from app.services.nlp.solution_detection.models.evaluation import (
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS,
    MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1,
    SolutionEvaluationReport,
    evaluate_solution_detection,
)
from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    NEGATIVE_CLASS_INDEX,
    POSITIVE_CLASS_INDEX,
    TARGET_LABELS,
    distinct_solution_types,
    label_to_bool,
    normalise_solution_type,
    parse_bool_cell,
    solution_type_counts,
    target_index,
    target_label,
    validate_solution_type,
)
from app.services.nlp.solution_detection.models.result import (
    SolutionDetectionResult,
    SolutionPrediction,
    SolutionTrainingReport,
)

__all__ = [
    # annotation
    "SolutionAnnotation",
    "SolutionCorpus",
    "CSV_COLUMNS",
    "DEFAULT_ANNOTATION_CSV",
    # labels / target
    "LABEL_NOT_SOLUTION",
    "LABEL_SOLUTION",
    "TARGET_LABELS",
    "NEGATIVE_CLASS_INDEX",
    "POSITIVE_CLASS_INDEX",
    "target_label",
    "target_index",
    "label_to_bool",
    "parse_bool_cell",
    "normalise_solution_type",
    "validate_solution_type",
    "distinct_solution_types",
    "solution_type_counts",
    # candidate logic
    "CandidateAssessment",
    "classify_response_candidate",
    "is_solution_candidate",
    "is_solution_response_type",
    "is_advice_response_type",
    "RESPONSE_TYPE_SOLUTION",
    "RESPONSE_TYPE_ADVICE",
    "DEFAULT_CANDIDATE_RESPONSE_TYPES",
    # results
    "SolutionPrediction",
    "SolutionDetectionResult",
    "SolutionTrainingReport",
    # evaluation
    "SolutionEvaluationReport",
    "evaluate_solution_detection",
    "MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS",
    "MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS",
    "MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1",
]
