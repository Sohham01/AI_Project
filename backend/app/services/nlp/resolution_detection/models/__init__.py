"""
Models for Resolution Detection: derived evidence, results, annotations,
evaluation.

Split into modules because they answer different questions:

- :mod:`.labels`       - how a value is parsed and checked
- :mod:`.evidence`     - what the four score terms are, and whether the record is coherent
- :mod:`.result`       - what a verdict reports, and what a run reports
- :mod:`.annotation`   - what a human adjudication looks like (schema only; no data ships)
- :mod:`.evaluation`   - how a run is scored against adjudications (refuses without them)
"""

from app.services.nlp.resolution_detection.models.annotation import (
    DEFAULT_RESOLUTION_CSV,
    RESOLUTION_CSV_COLUMNS,
    ResolutionAnnotation,
    ResolutionCorpus,
)
from app.services.nlp.resolution_detection.models.evaluation import (
    PerStatusMetrics,
    ResolutionEvaluation,
    evaluate_resolution,
    evaluate_results_list,
)
from app.services.nlp.resolution_detection.models.evidence import (
    DerivedEvidence,
    derive_evidence,
    derive_evidence_list,
    strong_link_count_from_scores,
    validate_evidence,
)
from app.services.nlp.resolution_detection.models.result import (
    ResolutionBatchResult,
    ResolutionResult,
    ResolutionRunReport,
)

__all__ = [
    "DerivedEvidence",
    "derive_evidence",
    "derive_evidence_list",
    "validate_evidence",
    "strong_link_count_from_scores",
    "ResolutionResult",
    "ResolutionBatchResult",
    "ResolutionRunReport",
    "ResolutionAnnotation",
    "ResolutionCorpus",
    "RESOLUTION_CSV_COLUMNS",
    "DEFAULT_RESOLUTION_CSV",
    "ResolutionEvaluation",
    "PerStatusMetrics",
    "evaluate_resolution",
    "evaluate_results_list",
]
