"""Phase 4 & 5 — Contextual Issue–Solution Mapping & Explainable Unresolved-Issue Detection."""

from ml.linking.contextual_mapper import (
    ISSUE_DESCRIPTIONS,
    ContextualIssueSolutionMapper,
    compute_negation_and_non_actionable_penalty,
    evaluate_issue_solution_ranking,
)
from ml.linking.resolution_engine import (
    RESOLUTION_STATES,
    VERDICT_TIERS,
    ExplainableResolutionEngine,
)

__all__ = [
    "ISSUE_DESCRIPTIONS",
    "ContextualIssueSolutionMapper",
    "compute_negation_and_non_actionable_penalty",
    "evaluate_issue_solution_ranking",
    "RESOLUTION_STATES",
    "VERDICT_TIERS",
    "ExplainableResolutionEngine",
]
