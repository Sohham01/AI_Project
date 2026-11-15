"""
Resolution Detection: how resolved does an issue appear, from the discussion?

The fourth question in the chain.  Issue identification asks what problem is being
expressed, solution detection whether a comment offers a fix, issue-solution
linking whether that fix addresses that problem, and this stage asks the only
question that is about a whole conversation:

    considering the whole discussion, how resolved does this issue appear?

``one solution present != issue resolved`` is the distinction the stage exists to
keep, and the implementation holds the line on it: only evidence linked to a
specific ``(issue, label)`` pair reaches the score.

Neutral evidence
----------------
:class:`IssueResolutionEvidence` is a plain container of counts and scores.  This
package imports **nothing** from any other NLP stage, so it can be built, tested
and replaced on its own.  Whoever assembles the evidence decides how it was
produced; this stage does not ask and does not need to know.

Three statuses, never four
--------------------------
``resolved``, ``partially_resolved``, ``unresolved``.  A fourth
``insufficient_evidence`` state was considered and rejected: it describes our
confidence, not the issue, and a consumer asking "which issues remain
unresolved?" needs a partition of the issues.  A thin discussion is reported as
``evidence_sufficient=False`` with a warning and reduced confidence while the
status stays inside the required three.

No accuracy numbers
-------------------
Resolution is a judgement about a discussion, so unlike the earlier stages it
cannot be measured by unit tests.  **No annotated resolution dataset exists in
this project and none has been fabricated.**  The annotation schema and the
evaluator ship; calling
:func:`~.models.evaluation.evaluate_resolution` without an explicit corpus raises
rather than inventing a metric.  The synthetic cases in the test suite and the
demo are plumbing checks, labelled as such, and are never a measurement.

Not inputs, on purpose
----------------------
Sentiment, emotion, comment volume as engagement, and unlinked solution
candidates.  A happy poster and a fixed problem are different facts, and a
positive-sentiment bonus would make the stage reward issues people are happy
about.  See :mod:`evidence_based_resolver` for the argument.

Quick start
-----------
::

    from app.services.nlp.resolution_detection import (
        EvidenceBasedResolver,
        IssueResolutionEvidence,
        ResolutionStatus,
    )

    resolver = EvidenceBasedResolver()
    result = resolver.detect(
        IssueResolutionEvidence(
            issue_id="post-42",
            issue_label="fees",
            total_comments=8,
            solution_candidate_count=2,
            linked_solution_count=2,
            strong_link_count=1,
            link_scores=[0.71, 0.34],
            solution_probabilities=[0.88, 0.61],
        )
    )
    result.status is ResolutionStatus.RESOLVED
"""

from app.services.nlp.resolution_detection.base import (
    STATUS_ORDER,
    STATUS_PARTIALLY_RESOLVED,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
    IssueResolutionEvidence,
    ResolutionDetection,
    ResolutionModelConfig,
    ResolutionStatus,
    TemporalEvidence,
    parse_status,
)
from app.services.nlp.resolution_detection.evidence_based_resolver import (
    MODEL_NAME,
    MODEL_VERSION,
    EvidenceBasedResolver,
)
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
    # statuses
    "ResolutionStatus",
    "STATUS_ORDER",
    "STATUS_RESOLVED",
    "STATUS_PARTIALLY_RESOLVED",
    "STATUS_UNRESOLVED",
    "parse_status",
    # evidence
    "IssueResolutionEvidence",
    "TemporalEvidence",
    "DerivedEvidence",
    "derive_evidence",
    "derive_evidence_list",
    "validate_evidence",
    "strong_link_count_from_scores",
    # configuration and interface
    "ResolutionModelConfig",
    "ResolutionDetection",
    "EvidenceBasedResolver",
    "MODEL_NAME",
    "MODEL_VERSION",
    # results
    "ResolutionResult",
    "ResolutionBatchResult",
    "ResolutionRunReport",
    # annotation and evaluation
    "ResolutionAnnotation",
    "ResolutionCorpus",
    "RESOLUTION_CSV_COLUMNS",
    "DEFAULT_RESOLUTION_CSV",
    "ResolutionEvaluation",
    "PerStatusMetrics",
    "evaluate_resolution",
    "evaluate_results_list",
]
