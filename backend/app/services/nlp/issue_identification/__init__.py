"""
Issue identification stage.

Topic discovery answers "what is this discussion generally about?".
Issue identification answers "what specific problem, concern, complaint or
difficulty is being expressed?".

    "IIM life is amazing but the workload is killing me"
      topic = campus life / academics
      issue = workload

Why multi-label, and why it is separate from topic discovery:
- One text often expresses several problems at once ("fees are too high AND
  the ROI is questionable"), so softmax's mutually-exclusive assumption is
  wrong.  The baseline trains one independent binary classifier per label and
  thresholds the sigmoids.
- Topics are discovered unsupervised and drift whenever the corpus changes.
  Issue labels are a human-defined taxonomy that must stay stable, so they are
  learned from manual annotation (IssueAnnotation) rather than mined.
- The label set is never hard-coded: whatever the annotators used is the label
  set, and the taxonomy is expected to evolve between annotation rounds.

Input is already-preprocessed text; this stage neither imports nor re-runs the
preprocessing pipeline, and never mutates raw text or earlier stage output.

See docs/issue_identification.md for the annotation CSV format, the
mathematics, the evaluation caveats, and the roadmap to a neural classifier.
"""

from app.services.nlp.issue_identification.base import (
    IssueDocument,
    IssueIdentification,
    IssueModelConfig,
)
from app.services.nlp.issue_identification.models.annotation import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    MULTI_VALUE_SEPARATOR,
    AnnotationCorpus,
    IssueAnnotation,
)
from app.services.nlp.issue_identification.models.evaluation import (
    IssueEvaluationReport,
    evaluate_predictions,
)
from app.services.nlp.issue_identification.models.result import (
    IssueIdentificationResult,
    IssuePrediction,
    IssueTrainingReport,
)
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)

__all__ = [
    # interface + input/config
    "IssueIdentification",
    "IssueDocument",
    "IssueModelConfig",
    # annotation
    "IssueAnnotation",
    "AnnotationCorpus",
    "CSV_COLUMNS",
    "DEFAULT_ANNOTATION_CSV",
    "MULTI_VALUE_SEPARATOR",
    # results
    "IssuePrediction",
    "IssueIdentificationResult",
    "IssueTrainingReport",
    # evaluation
    "IssueEvaluationReport",
    "evaluate_predictions",
    # implementations
    "TfidfLogisticIssueClassifier",
]
