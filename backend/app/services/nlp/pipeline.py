"""Offline downstream orchestration for already-normalized discussion records.

This module is deliberately an adapter, not a second NLP implementation.  It
owns the handoff between the deterministic language/preprocessing stages and
the separately-trained analysis stages.  A stage is either injected, explicitly
reported as unavailable, or reported as failed; the orchestrator never creates
placeholder predictions for an unready model.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, Field, StrictStr, ValidationError, field_validator

from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.language_analysis_service import FullLanguageAnalysis, LanguageAnalysisService
from app.services.nlp.preprocessing.base import PreprocessingResult
from app.services.nlp.preprocessing.multilingual_preprocessor import (
    MultilingualTextPreprocessor,
)
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.sentiment_emotion.base import SentimentEmotionDocument
from app.services.nlp.issue_identification.base import IssueDocument
from app.services.nlp.response_classification.base import ResponseDocument
from app.services.nlp.solution_detection.base import SolutionDocument
from app.services.nlp.issue_identification.models.result import IssueIdentificationResult
from app.services.nlp.sentiment_emotion.models.result import EmotionAnalysisResult, SentimentAnalysisResult
from app.services.nlp.response_classification.models.result import ResponseClassificationResult
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.solution_detection.models.result import SolutionDetectionResult
from app.services.nlp.thread_evidence import (
    ASSESSMENT_FIELDS, DiscussionCompleteness, ThreadEvidenceAdapter, ThreadEvidenceResult,
    _dump_prediction, _require_assessment_fields,
)


PIPELINE_STAGE_ORDER: Tuple[str, ...] = (
    "language",
    "preprocessing",
    "topic",
    "issue",
    "sentiment",
    "emotion",
    "response",
    "solution",
    "linking",
    "resolution",
)

RESULT_SCHEMAS = {
    "issue": IssueIdentificationResult,
    "sentiment": SentimentAnalysisResult,
    "emotion": EmotionAnalysisResult,
    "response": ResponseClassificationResult,
    "solution": SolutionDetectionResult,
}
# Matches ingestion.collection's approved direct communities, without importing
# collection/provider infrastructure into this offline adapter.
APPROVED_DIRECT_SUBREDDITS = frozenset({"iimsambalpur", "iimsambalpurug"})


class NormalizedRecord(BaseModel):
    """The minimum trusted contract consumed by the downstream pipeline.

    ``source_mode`` is required on purpose.  Direct institutional evidence and
    contextual evidence must not be reconstructed from a subreddit name later.
    ``text`` is the source text supplied to language detection and preprocessing;
    ``original_text`` may be supplied when the caller already has a separate
    traceability copy.
    """

    document_id: StrictStr
    text: StrictStr
    created_at: AwareDatetime
    source_mode: Literal["direct", "context"]
    source_subreddit: StrictStr
    post_id: Optional[StrictStr] = None
    comment_id: Optional[StrictStr] = None
    parent_comment_id: Optional[StrictStr] = None
    parent_reddit_id: Optional[StrictStr] = None
    parent_comment_text: Optional[StrictStr] = None
    url: Optional[StrictStr] = None
    original_text: Optional[StrictStr] = None

    @field_validator("document_id", "text", "source_subreddit", "post_id", "comment_id",
                     "parent_comment_id", "parent_reddit_id", "url")
    @classmethod
    def nonblank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("nonblank_string_required")
        return value

    @field_validator("document_id", "post_id", "comment_id", "parent_comment_id", "parent_reddit_id",
                     "source_subreddit")
    @classmethod
    def unambiguous_identifier(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and any(char.isspace() for char in value):
            raise ValueError("identifier_whitespace")
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def explicit_datetime(cls, value: Any) -> Any:
        if not isinstance(value, (datetime, str)):
            raise ValueError("timezone_aware_datetime_required")
        if isinstance(value, str):
            # Pydantic also accepts numeric strings as epoch seconds. The
            # normalized contract requires an explicit ISO timestamp instead.
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("url")
    @classmethod
    def absolute_url(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.hostname or any(c.isspace() for c in value):
                raise ValueError("absolute_http_url_required")
        return value


class PipelineRecord(BaseModel):
    """One accepted record plus all non-fabricated downstream metadata."""

    document_id: str
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    parent_reddit_id: Optional[str] = None
    parent_comment_text: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime
    source_mode: str
    source_subreddit: str
    original_text: Optional[str] = None
    text: str
    normalized_text: Optional[str] = None
    language_analysis: Optional[Dict[str, Any]] = None
    stage_outputs: Dict[str, Any] = Field(default_factory=dict)
    stage_errors: Dict[str, str] = Field(default_factory=dict)


class InvalidRecord(BaseModel):
    """Sanitized validation information; raw payloads are never echoed."""

    index: int
    document_id: Optional[str] = None
    fields: List[str] = Field(default_factory=list)
    reason: str


class StageReport(BaseModel):
    name: str
    status: str
    input_records: int = 0
    output_records: int = 0
    error_type: Optional[str] = None
    detail: Optional[str] = None


class PipelineResult(BaseModel):
    status: str
    records: List[PipelineRecord] = Field(default_factory=list)
    stages: List[StageReport] = Field(default_factory=list)
    invalid_records: List[InvalidRecord] = Field(default_factory=list)
    duplicate_document_ids: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    batch_outputs: Dict[str, Any] = Field(default_factory=dict)
    thread_evidence: Optional[ThreadEvidenceResult] = None


def _dump(value: Any) -> Any:
    """Serialize a Pydantic result without requiring one Pydantic major version."""
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            # Retain tuples (e.g. LanguageResult.all_detected) for strict
            # schema validation. PipelineResult handles JSON serialization.
            return value.model_dump(mode="python")
        return value.dict()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _dump(item) for key, item in vars(value).items()}
    return value


def _model_input(record: PipelineRecord, kind: str) -> BaseModel:
    common = {
        "document_id": record.document_id,
        "text": record.normalized_text or "",
        "post_id": record.post_id,
        "comment_id": record.comment_id,
        "original_text": record.original_text,
    }
    language = record.language_analysis or {}
    common.update(
        language=language.get("language", "unknown"),
        script=language.get("script", "unknown"),
        is_code_mixed=language.get("is_code_mixed", False),
        detected_languages=language.get("detected_languages", []),
    )
    issue = record.stage_outputs.get("issue", {})
    if kind in {"sentiment", "emotion", "response", "solution"}:
        common.update(
            issue_labels=issue.get("predicted_issue_labels", []),
            issue_probabilities=issue.get("issue_probabilities", {}),
        )
    if kind in {"sentiment", "emotion"}:
        return SentimentEmotionDocument(**common)
    if kind == "issue":
        return IssueDocument(**common)
    if kind == "response":
        return ResponseDocument(**common, parent_comment_text=record.parent_comment_text)
    if kind == "solution":
        response = record.stage_outputs.get("response", {})
        return SolutionDocument(
            **common,
            parent_comment_text=record.parent_comment_text,
            response_type=response.get("response_type"),
            response_probabilities=response.get("response_probabilities", {}),
        )
    raise ValueError(f"unsupported per-record stage {kind!r}")


def _identity_errors(payload: Mapping[str, Any]) -> set[str]:
    """Cross-field checks also run when individual field validation fails.

    document_id is an independent corpus key (exporters may prefix Reddit IDs).
    A post needs post_id; a comment needs both its own ID and its thread ID.
    """
    errors: set[str] = set()
    post, comment = payload.get("post_id"), payload.get("comment_id")
    if post is None:
        errors.add("post_id")
    for field in ("parent_comment_id", "parent_reddit_id", "parent_comment_text"):
        if payload.get(field) is not None and comment is None:
            errors.update({field, "comment_id"})
    if isinstance(comment, str) and payload.get("parent_comment_id") == comment:
        errors.add("parent_comment_id")
    if isinstance(comment, str) and payload.get("parent_reddit_id") in (comment, f"t1_{comment}"):
        errors.add("parent_reddit_id")
    subreddit = payload.get("source_subreddit")
    if payload.get("source_mode") == "direct" and (
        not isinstance(subreddit, str) or subreddit.casefold() not in APPROVED_DIRECT_SUBREDDITS
    ):
        errors.update({"source_mode", "source_subreddit"})
    return errors


def _validate_response(result: ResponseClassificationResult) -> None:
    """Validate taxonomy/order without inventing probabilities for absent classes."""
    known = result.known_response_types
    if not known or known != [label for label in RESPONSE_TYPES if label in known]:
        raise ValueError("invalid_response_label_order")
    if result.missing_response_types != [label for label in RESPONSE_TYPES if label not in known]:
        raise ValueError("invalid_missing_response_types")
    unpredicted = result.unpredicted_response_types
    if unpredicted != [label for label in known if label in unpredicted]:
        raise ValueError("invalid_unpredicted_response_types")
    for prediction in result.predictions:
        probabilities = prediction.response_probabilities
        if set(probabilities) != set(known) or any(not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("invalid_response_probabilities")
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("invalid_response_distribution")
        if prediction.response_type is not None and prediction.response_type not in known:
            raise ValueError("invalid_response_type")
        # Preserve the exact supplied values, only ordering their keys.
        prediction.response_probabilities = {label: probabilities[label] for label in known}


def _validate_echoes(result: BaseModel, documents: Sequence[BaseModel]) -> None:
    """Reject a model handoff that contradicts its input identity or source text."""
    by_id = {document.document_id: document for document in documents}
    echoed = (
        "post_id", "comment_id", "text", "original_text", "language", "script",
        "is_code_mixed", "detected_languages", "parent_comment_text",
    )
    for prediction in result.predictions:
        document = by_id[prediction.document_id]
        for field in echoed:
            if hasattr(prediction, field) and hasattr(document, field):
                if getattr(prediction, field) != getattr(document, field):
                    raise ValueError("prediction_source_metadata_mismatch")


def _validate_raw_echoes(raw: Any, documents: Sequence[BaseModel], records: Sequence[PipelineRecord]) -> None:
    """Check optional source/context echoes before neutral schemas drop extras."""
    by_id = {document.document_id: document.model_dump() for document in documents}
    source_by_id = {record.document_id: record for record in records}
    if not isinstance(raw, Mapping):
        return  # The result schema reports malformed envelopes.
    for prediction in raw.get("predictions", []):
        if not isinstance(prediction, Mapping):
            continue
        document_id = prediction.get("document_id")
        if not isinstance(document_id, str) or document_id not in by_id:
            continue  # Schema and exact ID-coverage validation handle these.
        expected = dict(by_id[document_id])
        source = source_by_id[document_id]
        for field in ("source_mode", "source_subreddit", "created_at", "url",
                      "parent_comment_id", "parent_reddit_id", "parent_comment_text", "normalized_text"):
            expected[field] = getattr(source, field)
        for field, value in expected.items():
            if field in prediction and prediction[field] != value:
                raise ValueError("prediction_source_metadata_mismatch")


class DownstreamPipeline:
    """Run the safe portion of the downstream pipeline over in-memory records.

    Supported models must expose ``predict`` and ``is_trained is True``.
    Topic discovery remains unavailable because it fits a corpus model. Injected
    no-fit linking/resolution interfaces consume validated thread evidence.
    """

    def __init__(
        self,
        *,
        language_service: Optional[LanguageAnalysisService] = None,
        preprocessor: Optional[MultilingualTextPreprocessor] = None,
        topic_model: Any = None,
        issue_model: Any = None,
        sentiment_model: Any = None,
        emotion_model: Any = None,
        response_model: Any = None,
        solution_model: Any = None,
        linking_model: Any = None,
        resolution_model: Any = None,
    ) -> None:
        self.language_service = language_service if language_service is not None else LanguageAnalysisService(
            language_detector=SimpleLanguageDetector(),
            script_detector=SimpleScriptDetector(),
            code_mix_detector=SimpleCodeMixDetector(),
        )
        self.preprocessor = preprocessor if preprocessor is not None else MultilingualTextPreprocessor()
        self.models = {
            "issue": issue_model,
            "sentiment": sentiment_model,
            "emotion": emotion_model,
            "response": response_model,
            "solution": solution_model,
        }
        self.thread_adapter = ThreadEvidenceAdapter(
            linking_model=linking_model, resolution_model=resolution_model,
        )

    def run(
        self, records: Sequence[Mapping[str, Any]], *,
        discussion_complete: DiscussionCompleteness = False,
    ) -> PipelineResult:
        accepted: List[PipelineRecord] = []
        invalid: List[InvalidRecord] = []
        duplicate_ids: List[str] = []
        seen_ids: set[str] = set()

        for index, payload in enumerate(records or []):
            fields = _identity_errors(payload) if isinstance(payload, Mapping) else {"record"}
            try:
                record = NormalizedRecord(**payload)
            except (ValidationError, TypeError, ValueError) as exc:
                if isinstance(exc, ValidationError):
                    fields.update(
                        {
                            str(error.get("loc", ["unknown"])[0])
                            for error in exc.errors()
                        }
                    )
                else:
                    fields.add("record")
            if fields:
                invalid.append(
                    InvalidRecord(
                        index=index,
                        fields=sorted(fields),
                        reason="record_validation_failed",
                    )
                )
                continue

            if record.document_id in seen_ids:
                if record.document_id not in duplicate_ids:
                    duplicate_ids.append(record.document_id)
                continue
            seen_ids.add(record.document_id)
            accepted.append(
                PipelineRecord(
                    document_id=record.document_id,
                    post_id=record.post_id,
                    comment_id=record.comment_id,
                    parent_comment_id=record.parent_comment_id,
                    parent_reddit_id=record.parent_reddit_id,
                    parent_comment_text=record.parent_comment_text,
                    url=record.url,
                    created_at=record.created_at,
                    source_mode=record.source_mode,
                    source_subreddit=record.source_subreddit,
                    original_text=record.original_text if record.original_text is not None else record.text,
                    text=record.text,
                )
            )

        stages: List[StageReport] = []
        batch_outputs: Dict[str, Any] = {}
        self._run_language(accepted, stages)
        self._run_preprocessing(accepted, stages)
        for stage_name in PIPELINE_STAGE_ORDER[2:-2]:
            self._run_model_stage(stage_name, accepted, stages, batch_outputs)

        thread_evidence = self.process_thread_evidence(
            PipelineResult(status="pending", records=accepted, batch_outputs=batch_outputs,
                           invalid_records=invalid, duplicate_document_ids=duplicate_ids),
            discussion_complete=discussion_complete,
        )
        for name in ("linking", "resolution"):
            operations = [getattr(thread, name) for thread in thread_evidence.threads]
            statuses = {operation.status for operation in operations}
            coverage = thread_evidence.coverage
            if not accepted:
                stage_status, detail = "skipped", "no_valid_records"
            elif statuses == {"unavailable"}:
                stage_status = "unavailable"
                details = {operation.detail for operation in operations}
                detail = next(iter(details)) if len(details) == 1 else "see_thread_reports"
            elif "failed" in statuses:
                stage_status = "failed" if statuses == {"failed"} else "partial_failure"
                detail = "see_thread_reports"
            elif "unavailable" in statuses:
                stage_status, detail = "partial_failure", "see_thread_reports"
            elif name == "resolution" and coverage.resolution_excluded:
                stage_status = "partial_failure" if coverage.resolution_outputs else "skipped"
                detail = "see_thread_reports"
            else:
                stage_status = "completed" if "completed" in statuses else "skipped"
                detail = "see_thread_reports"
            stages.append(StageReport(
                name=name, status=stage_status, detail=detail,
                input_records=(coverage.issues_linking_eligible if name == "linking" else coverage.resolution_eligible),
                output_records=(coverage.links_supported if name == "linking" else coverage.resolution_outputs),
            ))

        if not accepted:
            status = "invalid_input"
        elif invalid or duplicate_ids or any(
            stage.status in {"failed", "partial_failure", "unavailable"} for stage in stages
        ):
            status = "partial_failure" if accepted else "invalid_input"
        else:
            status = "completed"
        warnings: List[str] = []
        if duplicate_ids:
            warnings.append("duplicate_document_ids: first occurrence retained")
        if not accepted:
            warnings.append("no_valid_records: no downstream stages produced predictions")
        return PipelineResult(
            status=status,
            records=accepted,
            stages=stages,
            invalid_records=invalid,
            duplicate_document_ids=duplicate_ids,
            warnings=warnings,
            batch_outputs=batch_outputs,
            thread_evidence=thread_evidence,
        )

    def process_thread_evidence(
        self, result: PipelineResult, *, discussion_complete: DiscussionCompleteness = False,
    ) -> ThreadEvidenceResult:
        """Link/resolve existing outputs without loading or running classifiers."""
        return self.thread_adapter.process(result, discussion_complete=discussion_complete)

    def _run_language(self, records: List[PipelineRecord], stages: List[StageReport]) -> None:
        if not records:
            stages.append(StageReport(name="language", status="skipped", detail="no_valid_records"))
            return
        failures = 0
        for record in records:
            try:
                result = FullLanguageAnalysis.model_validate(
                    _dump(self.language_service.analyze(record.text)), strict=True
                )
                record.language_analysis = _dump(result)
                record.stage_outputs["language"] = record.language_analysis
            except Exception as exc:  # noqa: BLE001 - stage boundary is sanitized
                failures += 1
                record.stage_errors["language"] = type(exc).__name__
        stages.append(
            StageReport(
                name="language",
                status=("failed" if failures == len(records) else "partial_failure") if failures else "completed",
                input_records=len(records),
                output_records=len(records) - failures,
                error_type="stage_error" if failures else None,
            )
        )

    def _run_preprocessing(self, records: List[PipelineRecord], stages: List[StageReport]) -> None:
        if not records:
            stages.append(StageReport(name="preprocessing", status="skipped", detail="no_valid_records"))
            return
        failures = 0
        for record in records:
            if record.language_analysis is None:
                failures += 1
                record.stage_errors["preprocessing"] = "language_not_ready"
                continue
            try:
                language = record.language_analysis or {}
                result = self.preprocessor.preprocess(
                    record.text,
                    language=language.get("language"),
                    script=language.get("script"),
                    is_code_mixed=bool(language.get("is_code_mixed", False)),
                    detected_languages=language.get("detected_languages", []),
                )
                dumped = _dump(PreprocessingResult.model_validate(_dump(result), strict=True))
                record.stage_outputs["preprocessing"] = dumped
                if not dumped["normalized_text"].strip():
                    failures += 1
                    record.stage_errors["preprocessing"] = "empty_normalized_text"
                    continue
                record.normalized_text = dumped["normalized_text"]
            except Exception as exc:  # noqa: BLE001 - stage boundary is sanitized
                failures += 1
                record.stage_errors["preprocessing"] = type(exc).__name__
        stages.append(
            StageReport(
                name="preprocessing",
                status=("failed" if failures == len(records) else "partial_failure") if failures else "completed",
                input_records=len(records),
                output_records=len(records) - failures,
                error_type="stage_error" if failures else None,
            )
        )

    def _run_model_stage(
        self,
        name: str,
        records: List[PipelineRecord],
        stages: List[StageReport],
        batch_outputs: Dict[str, Any],
    ) -> None:
        if not records:
            stages.append(StageReport(name=name, status="skipped", detail="no_valid_records"))
            return
        if name == "topic":
            stages.append(StageReport(
                name=name, status="unavailable", input_records=len(records),
                detail="topic_discovery_requires_fitting",
            ))
            return
        applicable = [record for record in records
                      if name not in {"response", "solution"} or record.comment_id is not None]
        if not applicable:
            stages.append(StageReport(name=name, status="skipped", detail="no_comment_records"))
            return
        eligible = [record for record in applicable if record.normalized_text is not None
                     and record.normalized_text.strip() and "preprocessing" not in record.stage_errors]
        for record in applicable:
            if record.normalized_text is None or "preprocessing" in record.stage_errors:
                record.stage_errors[name] = "preprocessing_not_ready"
        if not eligible:
            stages.append(StageReport(name=name, status="skipped", input_records=len(applicable),
                                      detail="no_preprocessed_records"))
            return
        model = self.models.get(name)
        if model is None:
            stages.append(StageReport(name=name, status="unavailable", input_records=len(eligible),
                                      detail="not_configured"))
            return
        phase = "readiness_check_failed"
        try:
            if getattr(model, "is_trained", None) is not True:
                stages.append(StageReport(name=name, status="unavailable", input_records=len(eligible),
                                          detail="model_not_ready"))
                return
            if getattr(getattr(model, "config", None), "use_topic_features", False) is True:
                stages.append(StageReport(name=name, status="unavailable", input_records=len(eligible),
                                          detail="topic_context_not_available"))
                return
            phase = "model_input_invalid"
            documents = [_model_input(record, name) for record in eligible]
            phase = "prediction_failed"
            raw_result = model.predict(deepcopy(documents))
            phase = "invalid_stage_result"
            # Bound each raw prediction before recursive serialization or schema
            # defaults can discard its extras. Keep the existing per-prediction
            # limits independent of the number of records in a model batch.
            raw_predictions = (raw_result.get("predictions", []) if isinstance(raw_result, Mapping)
                               else getattr(raw_result, "predictions", []))
            if isinstance(raw_predictions, (list, tuple)):
                for prediction in raw_predictions:
                    _dump_prediction(prediction)
            payload = _dump(raw_result)
            _validate_raw_echoes(payload, documents, eligible)
            result = RESULT_SCHEMAS[name].model_validate(payload, strict=True)
            if name in ASSESSMENT_FIELDS and isinstance(payload, Mapping):
                for prediction in payload.get("predictions", []):
                    _require_assessment_fields(name, prediction)
            items = result.predictions
            expected = {record.document_id for record in eligible}
            predicted_ids = [item.document_id for item in items]
            if len(predicted_ids) != len(expected) or set(predicted_ids) != expected:
                raise ValueError("prediction_id_coverage_mismatch")
            if result.n_documents != len(expected):
                raise ValueError("prediction_document_count_mismatch")
            _validate_echoes(result, documents)
            if name == "response":
                _validate_response(result)
            dumped = _dump(result)
            predictions = {item["document_id"]: item for item in dumped["predictions"]}
            for record in eligible:
                record.stage_outputs[name] = predictions[record.document_id]
            batch_outputs[name] = dumped
            stages.append(
                StageReport(
                    name=name,
                    status="completed" if len(eligible) == len(applicable) else "partial_failure",
                    input_records=len(applicable),
                    output_records=len(items),
                    detail="some_records_not_preprocessed" if len(eligible) != len(applicable) else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - stage boundary is sanitized
            for record in eligible:
                record.stage_errors[name] = type(exc).__name__
            stages.append(
                StageReport(
                    name=name,
                    status="failed",
                    input_records=len(applicable),
                    output_records=0,
                    error_type="stage_error",
                    detail=phase,
                )
            )


__all__ = [
    "DownstreamPipeline",
    "InvalidRecord",
    "NormalizedRecord",
    "PIPELINE_STAGE_ORDER",
    "PipelineRecord",
    "PipelineResult",
    "StageReport",
]
