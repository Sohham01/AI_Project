"""
Issue-Solution Linking
======================

The cross-comment stage of the pipeline.  Issue identification and solution
detection both answer a question about a *single* comment; this one answers
"which detected solution addresses which identified issue?", and it is the only
stage that puts two documents side by side.

What it outputs
---------------
A :class:`LinkingResult` of :class:`LinkPrediction` records - one per
(issue, issue-label, solution) candidate, carrying every score behind the
decision and the model name and version whether or not the link was asserted.  A
reader can therefore always tell "the threshold said no" apart from "this stage
never looked".

What it refuses to do
---------------------
- It does not decide whether an issue was *resolved*.  A link says a comment
  addresses an issue, not that anyone used it or that the problem went away.
- It does not generate solutions or answer questions.
- It does not link every solution to every issue.  Candidates are thread-scoped
  by default, and every rejected candidate is counted by reason.
- It does not quote precision or recall.  No human-annotated link dataset exists,
  so :func:`evaluate_linking` requires one to be supplied explicitly and refuses
  to produce a number without it.

Stage boundaries
----------------
This package imports nothing from any other NLP stage.  ``IssueEvidence`` and
``SolutionEvidence`` are neutral containers describing what this stage needs, so
an upstream stage can be replaced or renamed without touching a single import
here.  Response role, solution probability and issue probabilities all arrive as
plain data.

Quick start
-----------
::

    from app.services.nlp.issue_solution_linking import (
        IssueEvidence,
        SolutionEvidence,
        TfidfCosineIssueSolutionLinker,
    )

    linker = TfidfCosineIssueSolutionLinker(link_threshold=0.5)
    result = linker.link(issues, solutions)
    print(result.format_summary())
    for edge in result.linked_pairs():
        ...  # issue_document_id, issue_label, solution_document_id

See ``docs/issue_solution_linking.md`` for the scoring mathematics, the
configuration reference and the known limitations.
"""

from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence,
    IssueSolutionLinking,
    LinkModelConfig,
    SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.models import (
    DEFAULT_LINK_CSV,
    LINK_CSV_COLUMNS,
    REASON_BELOW_CANDIDATE_THRESHOLD,
    REASON_CAPPED,
    REASON_CROSS_POST,
    REASON_CROSS_THREAD_DISABLED,
    REASON_ISSUE_HAS_NO_LABEL,
    REASON_PARENT_CHILD,
    REASON_SAME_POST,
    REASON_UNRESOLVED_THREAD,
    CandidateDecision,
    LinkAnnotation,
    LinkCorpus,
    LinkPrediction,
    LinkingEvaluation,
    LinkingResult,
    LinkingRunReport,
    evaluate_linking,
    generate_candidates,
    is_solution_candidate,
    normalised_labels,
    parse_linked_cell,
    thread_signals,
    threshold_sweep,
)
from app.services.nlp.issue_solution_linking.tfidf_cosine_linker import (
    LINK_MODEL_NAME,
    LINK_MODEL_VERSION,
    TfidfCosineIssueSolutionLinker,
)

__all__ = [
    # contract
    "IssueEvidence",
    "SolutionEvidence",
    "LinkModelConfig",
    "IssueSolutionLinking",
    # implementation
    "TfidfCosineIssueSolutionLinker",
    "LINK_MODEL_NAME",
    "LINK_MODEL_VERSION",
    # candidate generation
    "CandidateDecision",
    "generate_candidates",
    "is_solution_candidate",
    "thread_signals",
    "normalised_labels",
    "REASON_BELOW_CANDIDATE_THRESHOLD",
    "REASON_CAPPED",
    "REASON_CROSS_POST",
    "REASON_CROSS_THREAD_DISABLED",
    "REASON_ISSUE_HAS_NO_LABEL",
    "REASON_PARENT_CHILD",
    "REASON_SAME_POST",
    "REASON_UNRESOLVED_THREAD",
    # results
    "LinkPrediction",
    "LinkingResult",
    "LinkingRunReport",
    # annotation schema - defined, deliberately unshipped
    "LinkAnnotation",
    "LinkCorpus",
    "LINK_CSV_COLUMNS",
    "DEFAULT_LINK_CSV",
    "parse_linked_cell",
    # evaluation - requires an explicit annotated corpus
    "LinkingEvaluation",
    "evaluate_linking",
    "threshold_sweep",
]
