"""
Solution detection stage.

Solution detection asks one question: **does this comment provide an actionable
solution/recommendation?**

    "apply for the scholarship and submit the required documents" -> True
    "try applying to the scholarship before the deadline"        -> True
    "you should contact the placement office"                     -> True (advice)
    "same bro"                                                    -> False
    "I faced the same issue last year"                            -> False
    "I think fees are too high"                                   -> False
    "actually fees aren't that high compared with other colleges"  -> False

How it differs from response classification
-------------------------------------------
The completed response-classification stage answers a *different* question -
"what role does this comment play?" - with one of six roles.  The two are
adjacent, and the overlap is exactly two roles::

    response_type        is it a solution candidate?   is_solution
    -------------------  ---------------------------   -----------
    advice               yes                          from the detector
    solution             yes                          from the detector
    agreement            no                           False
    personal_experience  no                           False
    opinion              no                           False
    counterargument      no                           False

**A candidate is not a solution.**  "You should probably try emailing them, not
sure" is advice, so it is a candidate, and it quite possibly contains nothing
actionable.  ``is_solution`` therefore comes only from this stage's trained
detector - never from the response role.  :mod:`.models.candidate` enforces that
structurally: a :class:`CandidateAssessment` carries ``is_solution=None`` by
construction, so "advice implies solution" cannot even be written down without
running the detector.

What is NOT in this package
---------------------------
- **Issue-Solution Linking.**  This stage answers a per-comment question and
  stops.  Deciding *which issue* a solution answers needs the thread structure
  and is a separate, later stage.  ``post_id`` and ``issue_labels`` are carried
  and echoed, never used as features.
- **Resolution Detection.**  A comment can propose a solution that nobody used.
  Whether a problem was actually fixed is a different question.
- **A solution taxonomy.**  ``solution_type`` is free-form annotation data,
  reported for coverage and never learned in V1.  There is no category list in
  this package, on purpose: a fixed list would encode one community's vocabulary
  into a general model.

Design summary
--------------
- **Sigmoid, because the target is binary.**  Two-class softmax is
  algebraically identical to a single sigmoid, so ``predict_proba(X)[:, 1]`` *is*
  ``sigma(w.x + b)``.  The documented mathematics is exact, not approximate.
- **The threshold is a decision boundary**, not a confidence floor: it is what
  turns P(solution) into the boolean a linker consumes.
- **Abstention is opt-in** and sits *above* the boundary
  (``abstain_threshold``), covering the weak-positive band where the model is
  positive-leaning but not confident enough to assert a link.
- **Context is carried but unused.**  ``parent_comment_text``, ``topic_*`` and
  ``issue_*`` are optional on the input model so a context-aware detector needs no
  interface change.  V1 reads comment text alone, and a test enforces it.
- **Response context is triage only.**  ``response_type`` marks a comment as
  worth checking; it never influences ``is_solution``.
- **Input is already-preprocessed text.**  This stage neither imports nor re-runs
  preprocessing, and mutates nothing it is given.

Versioned model identifiers:
    solution_model = "tfidf_logistic_binary"   solution_model_version = "v1"

See docs/solution_detection.md for the annotation CSV format, the mathematics,
the advice-vs-solution discussion, the evaluation caveats, the multilingual
limitations, and the roadmap to solution extraction and Issue-Solution Linking.
"""

from app.services.nlp.solution_detection.base import (
    SolutionDetection,
    SolutionDocument,
    SolutionModelConfig,
)
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
from app.services.nlp.solution_detection.tfidf_logistic_binary import (
    TfidfLogisticSolutionClassifier,
    library_versions,
)

__all__ = [
    # interfaces + input/config
    "SolutionDetection",
    "SolutionDocument",
    "SolutionModelConfig",
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
    # implementations
    "TfidfLogisticSolutionClassifier",
    "library_versions",
]
