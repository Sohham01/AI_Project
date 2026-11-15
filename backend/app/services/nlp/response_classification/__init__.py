"""
Response classification stage.

Response classification asks one question: **what role does this comment play in
the discussion?**

    "same bro"                               -> agreement
    "I faced this last year"                 -> personal_experience
    "I think this college is better"         -> opinion
    "you should contact the placement office" -> advice
    "apply for the scholarship"              -> solution
    "actually the fees aren't that high"     -> counterargument

It is a different question from every stage before it:

    language / script / code-mix : what language is this written in?
    preprocessing                : what does a clean version of it look like?
    topic discovery              : what is this discussion about?
    issue identification         : what problem is being complained about?
    sentiment / emotion          : how does the writer feel about it?
    response classification      : what is the writer DOING here?

In particular it is **not** sentiment: the same role can be expressed happily or
angrily, and "same bro" carries almost no polarity at all.  It is also **not**
issue identification: an issue says what is wrong, a response role says what the
commenter is doing about it - or not doing anything about it.

Why it follows issue identification: a role is only meaningful relative to
something being discussed.  Agreement and counterargument are dialogic - they
respond to a claim - so the discussion's topic and its issues are the substrate.
Both travel on the input document as optional metadata and are echoed back
untouched.

Design summary
--------------
- **The six labels are explicit.**  They are defined by the project assignment,
  not discovered from data, so they are written down once in
  :mod:`.models.labels` and an annotation carrying anything else is refused.  No
  V1 class may be added, removed or renamed without bumping the model version.
- **Softmax, not sigmoid.**  A comment plays one primary role, so the six
  probabilities form a distribution summing to 1 and the prediction is the
  arg-max.  A One-vs-Rest sigmoid ensemble would answer a different question.
- **Context is carried but unused.**  ``parent_comment_text``, ``topic_id`` and
  ``issue_labels`` are optional on the input model so a future context-aware
  classifier can be added without changing this interface.  V1 reads comment
  text alone, and a test enforces that rather than merely documenting it.
- **Input is already-preprocessed text.**  This stage neither imports nor re-runs
  the preprocessing pipeline, and never mutates raw text, ``normalized_text``,
  topic output, issue predictions or sentiment/emotion results.

Versioned model identifiers:
    response_model = "tfidf_logistic_multiclass"   response_model_version = "v1"

See docs/response_classification.md for the annotation CSV format, the
mathematics, the class-imbalance discussion, the evaluation caveats, the
multilingual limitations, and the roadmap to a context-aware neural classifier.
"""

from app.services.nlp.response_classification.base import (
    ResponseClassification,
    ResponseDocument,
    ResponseModelConfig,
)
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
from app.services.nlp.response_classification.tfidf_logistic_multiclass import (
    TfidfLogisticResponseClassifier,
    library_versions,
)

__all__ = [
    # interfaces + input/config
    "ResponseClassification",
    "ResponseDocument",
    "ResponseModelConfig",
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
    # implementations
    "TfidfLogisticResponseClassifier",
    "library_versions",
]
