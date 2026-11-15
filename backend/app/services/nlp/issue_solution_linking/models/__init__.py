"""
Data models and helpers for Issue-Solution Linking.

Split by role, so a consumer imports what it needs and nothing else:

- :mod:`labels` - strict cell parsing and canonical label ordering.
- :mod:`candidate` - which (issue, label, solution) triples are scored at all.
- :mod:`result` - the scored links, and the provenance of a run.
- :mod:`annotation` - the schema a future human-annotated link dataset takes.
  No such dataset ships with this project.
- :mod:`evaluation` - metrics, which refuse to be computed without an explicit
  annotated corpus.
"""

from app.services.nlp.issue_solution_linking.models.annotation import (
    DEFAULT_LINK_CSV,
    LINK_CSV_COLUMNS,
    LinkAnnotation,
    LinkCorpus,
    parse_linked_cell,
)
from app.services.nlp.issue_solution_linking.models.candidate import (
    REASON_BELOW_CANDIDATE_THRESHOLD,
    REASON_CAPPED,
    REASON_CROSS_POST,
    REASON_CROSS_THREAD_DISABLED,
    REASON_ISSUE_HAS_NO_LABEL,
    REASON_PARENT_CHILD,
    REASON_SAME_POST,
    REASON_UNRESOLVED_THREAD,
    CandidateDecision,
    generate_candidates,
    is_solution_candidate,
    normalised_labels,
    thread_signals,
)
from app.services.nlp.issue_solution_linking.models.evaluation import (
    LinkingEvaluation,
    evaluate_linking,
    threshold_sweep,
)
from app.services.nlp.issue_solution_linking.models.labels import (
    normalise_labels,
    normalise_probabilities,
    parse_bool_cell,
    parse_float_cell,
    strip_placeholder_tokens,
)
from app.services.nlp.issue_solution_linking.models.result import (
    LinkPrediction,
    LinkingResult,
    LinkingRunReport,
)

__all__ = [
    # annotation schema (no data ships with it)
    "DEFAULT_LINK_CSV",
    "LINK_CSV_COLUMNS",
    "LinkAnnotation",
    "LinkCorpus",
    "parse_linked_cell",
    # candidate generation
    "CandidateDecision",
    "REASON_BELOW_CANDIDATE_THRESHOLD",
    "REASON_CAPPED",
    "REASON_CROSS_POST",
    "REASON_CROSS_THREAD_DISABLED",
    "REASON_ISSUE_HAS_NO_LABEL",
    "REASON_PARENT_CHILD",
    "REASON_SAME_POST",
    "REASON_UNRESOLVED_THREAD",
    "generate_candidates",
    "is_solution_candidate",
    "normalised_labels",
    "thread_signals",
    # results
    "LinkPrediction",
    "LinkingResult",
    "LinkingRunReport",
    # evaluation (requires an explicit annotated corpus)
    "LinkingEvaluation",
    "evaluate_linking",
    "threshold_sweep",
    # parsing and label helpers
    "normalise_labels",
    "normalise_probabilities",
    "parse_bool_cell",
    "parse_float_cell",
    "strip_placeholder_tokens",
]
