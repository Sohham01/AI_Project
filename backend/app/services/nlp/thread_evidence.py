"""In-memory, no-fit handoff from existing predictions to thread evidence.

Source records remain authoritative. Model results are checked, never used to
repair source metadata, and an unavailable assessment never becomes a verdict.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Field

from app.services.nlp.issue_identification.models.result import IssuePrediction
from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence, IssueSolutionLinking, SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.models.result import LinkPrediction, LinkingResult
from app.services.nlp.issue_solution_linking.tfidf_cosine_linker import TfidfCosineIssueSolutionLinker
from app.services.nlp.resolution_detection.base import (
    IssueResolutionEvidence, ResolutionDetection, TemporalEvidence,
)
from app.services.nlp.resolution_detection.models.evidence import derive_evidence
from app.services.nlp.resolution_detection.models.result import ResolutionBatchResult
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.response_classification.models.result import ResponsePrediction
from app.services.nlp.solution_detection.models.result import SolutionPrediction

if TYPE_CHECKING:
    from app.services.nlp.pipeline import PipelineRecord, PipelineResult

ThreadKey = Tuple[str, str, str]
DiscussionCompleteness = Union[bool, Mapping[ThreadKey, bool]]
ASSESSMENT_FIELDS = {
    "issue": ("predicted_issue_labels", "issue_probabilities", "confidence"),
    "response": ("response_type", "response_probabilities", "confidence", "abstained"),
    "solution": ("is_solution", "solution_probability", "not_solution_probability", "threshold", "abstained"),
}
MAX_PREDICTION_DEPTH = 32
MAX_PREDICTION_ITEMS = 10000


def _dump_prediction(value: Any) -> Any:
    """Copy a prediction within explicit bounds; never stringify unknown objects.

    Cycles and unsupported values are rejection conditions, including in extra
    fields. Shared, acyclic containers are allowed. No source values are repaired.
    """
    active = set()
    remaining = MAX_PREDICTION_ITEMS

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if depth > MAX_PREDICTION_DEPTH or remaining < 0:
            raise ValueError("prediction_payload_limit")
        if item is None or isinstance(item, (str, bool, int, float, datetime)):
            return item
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic_prediction_payload")
        active.add(identity)
        try:
            if isinstance(item, BaseModel):
                return visit(item.model_dump(mode="python"), depth + 1)
            if isinstance(item, dict):
                if len(item) > remaining or any(not isinstance(key, str) for key in item):
                    raise ValueError("invalid_prediction_mapping")
                return {key: visit(value, depth + 1) for key, value in item.items()}
            if isinstance(item, (list, tuple)):
                if len(item) > remaining:
                    raise ValueError("prediction_payload_limit")
                values = [visit(value, depth + 1) for value in item]
                return tuple(values) if isinstance(item, tuple) else values
            raise ValueError("unsupported_prediction_value")
        finally:
            active.remove(identity)

    return visit(value, 0)


def _require_assessment_fields(name: str, payload: Mapping[str, Any]) -> None:
    # Neutral schemas have defaults for convenience; absent decisions are not
    # evidence that a classifier processed a record and returned a negative.
    if not isinstance(payload, Mapping) or any(field not in payload for field in ASSESSMENT_FIELDS[name]):
        raise ValueError("missing_explicit_assessment_fields")


class EvidenceCoverage(BaseModel):
    """Counts are adapter observations, not copied model summary counters.

    Attempts/failures count calls; pairs_attempted/pairs_failed count label-level
    records. Exclusion reasons can overlap, while resolution_excluded is unique.
    """

    records_received: int = 0
    records_validated: int = 0
    records_rejected: int = 0
    upstream_invalid_records: int = 0
    upstream_duplicate_ids: int = 0
    threads_received: int = 0
    threads_validated: int = 0
    threads_complete: int = 0
    threads_partial: int = 0
    issue_predictions_received: int = 0
    issue_predictions_validated: int = 0
    issues_received: int = 0
    issues_validated: int = 0
    issue_labels_received: int = 0
    issue_labels_validated: int = 0
    issues_linking_eligible: int = 0
    threads_linking_eligible: int = 0
    solution_predictions_received: int = 0
    solution_predictions_validated: int = 0
    solutions_received: int = 0
    solutions_validated: int = 0
    response_predictions_received: int = 0
    response_predictions_validated: int = 0
    link_attempts: int = 0
    link_failures: int = 0
    links_returned: int = 0
    links_validated: int = 0
    links_supported: int = 0
    link_rejections: int = 0
    resolution_pairs_received: int = 0
    resolution_eligible: int = 0
    resolution_excluded: int = 0
    resolution_attempts: int = 0
    resolution_pairs_attempted: int = 0
    resolution_outputs: int = 0
    resolution_failures: int = 0
    resolution_pairs_failed: int = 0
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)
    resolution_excluded_reasons: Dict[str, int] = Field(default_factory=dict)


class EvidenceOperation(BaseModel):
    status: str = "skipped"
    detail: Optional[str] = None
    error_type: Optional[str] = None


class IssueLabelCoverage(BaseModel):
    issue_id: str
    issue_label: str
    total_comments: int
    solution_predictions_validated: int
    linked_solution_ids: List[str] = Field(default_factory=list)
    strong_solution_ids: List[str] = Field(default_factory=list)
    resolution_eligible: bool = False
    excluded_reasons: List[str] = Field(default_factory=list)


class ThreadEvidenceReport(BaseModel):
    source_mode: str
    source_subreddit: str  # casefolded grouping key; source spelling stays below
    post_id: str
    status: str = "partial_failure"
    discussion_complete: bool = False
    prediction_processing_complete: bool = False
    coverage_complete: bool = False
    # Original URLs, times, provenance, text, and parent identities live here;
    # neutral stage schemas intentionally have no fields for most of them.
    records: List[Dict[str, Any]] = Field(default_factory=list)
    coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)
    validation_errors: List[str] = Field(default_factory=list)
    prediction_rejections: Dict[str, str] = Field(default_factory=dict)
    response_role_histogram: Dict[str, int] = Field(
        default_factory=lambda: {role: 0 for role in RESPONSE_TYPES}
    )
    issue_labels: List[IssueLabelCoverage] = Field(default_factory=list)
    linking: EvidenceOperation = Field(default_factory=EvidenceOperation)
    resolution: EvidenceOperation = Field(default_factory=EvidenceOperation)
    linking_result: Optional[LinkingResult] = None
    supported_links: List[LinkPrediction] = Field(default_factory=list)
    resolution_evidence: List[IssueResolutionEvidence] = Field(default_factory=list)
    resolution_result: Optional[ResolutionBatchResult] = None

    def thread_key(self) -> ThreadKey:
        return self.source_mode, self.source_subreddit, self.post_id


class ThreadEvidenceResult(BaseModel):
    status: str = "skipped"
    coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)
    threads: List[ThreadEvidenceReport] = Field(default_factory=list)
    # Sanitized indexes only: invalid source records cannot supply trusted IDs.
    invalid_record_indexes: List[int] = Field(default_factory=list)


def _unit(value: float) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid_unit_score")


def _count_reason(mapping: Dict[str, int], reason: str, count: int = 1) -> None:
    mapping[reason] = mapping.get(reason, 0) + count


def _key(record: PipelineRecord) -> ThreadKey:
    return record.source_mode, record.source_subreddit.casefold(), record.post_id


def _positive(prediction: SolutionPrediction) -> bool:
    return prediction.is_solution and not prediction.abstained and prediction.solution_probability > 0


class ThreadEvidenceAdapter:
    """Consume a PipelineResult without preprocessing or classifier inference.

    Inject implementations of the existing link/detect interfaces. The built-in
    TF-IDF linker (including subclasses) is blocked because link() fits a corpus.
    Other injected implementations are the caller's explicit no-fit dependency.
    Completeness is an explicit assertion about supplied discussion coverage,
    never inferred from record counts or classifier success.
    """

    def __init__(
        self, *, linking_model: Optional[IssueSolutionLinking] = None,
        resolution_model: Optional[ResolutionDetection] = None,
    ) -> None:
        self.linker = linking_model
        self.resolver = resolution_model

    def process(
        self, result: PipelineResult, *, discussion_complete: DiscussionCompleteness = False,
    ) -> ThreadEvidenceResult:
        # Lazy import avoids a pipeline/model cycle and reuses its source contract.
        from app.services.nlp.pipeline import NormalizedRecord, PipelineRecord, _identity_errors

        output = ThreadEvidenceResult()
        groups: Dict[ThreadKey, List[PipelineRecord]] = defaultdict(list)
        for index, record in enumerate(result.records):
            try:
                # Validate source metadata independently of opaque stage payloads.
                # Do not dump/deepcopy predictions at the source boundary: a bad
                # payload is a local prediction rejection, not lost source data.
                payload = record.model_dump(exclude={"stage_outputs"})
                record = PipelineRecord.model_validate(
                    {**payload, "stage_outputs": record.stage_outputs}, strict=True,
                )
                if _identity_errors(payload):
                    raise ValueError("invalid_source_identity")
                NormalizedRecord.model_validate(payload)
                groups[_key(record)].append(record)
            except Exception:  # source boundary: never echo invalid metadata
                output.invalid_record_indexes.append(index)

        # Validate references against the entire input, before partitioned calls.
        documents: Dict[str, List[ThreadKey]] = defaultdict(list)
        comments: Dict[str, List[ThreadKey]] = defaultdict(list)
        for key, records in groups.items():
            for record in records:
                documents[record.document_id].append(key)
                if record.comment_id is not None:
                    comments[record.comment_id].append(key)

        batch_errors = self._batch_errors(
            [row for records in groups.values() for row in records], result.batch_outputs,
        )
        for key in sorted(groups):
            records = sorted(groups[key], key=lambda row: row.document_id)
            complete = (discussion_complete is True if not isinstance(discussion_complete, Mapping)
                        else discussion_complete.get(key) is True)
            thread = ThreadEvidenceReport(
                source_mode=key[0], source_subreddit=key[1], post_id=key[2],
                discussion_complete=complete,
                records=[record.model_dump(exclude={"stage_outputs", "stage_errors"}) for record in records],
                validation_errors=self._reference_errors(records, documents, comments),
            )
            self._process_thread(
                thread, records, batch_errors,
                source_coverage_valid=not (output.invalid_record_indexes or result.invalid_records or result.duplicate_document_ids),
            )
            output.threads.append(thread)

        for thread in output.threads:
            for field in EvidenceCoverage.model_fields:
                value = getattr(thread.coverage, field)
                if isinstance(value, dict):
                    for reason, count in value.items():
                        _count_reason(getattr(output.coverage, field), reason, count)
                else:
                    setattr(output.coverage, field, getattr(output.coverage, field) + value)
        output.coverage.records_received += len(output.invalid_record_indexes)
        output.coverage.records_rejected += len(output.invalid_record_indexes)
        output.coverage.upstream_invalid_records = len(result.invalid_records)
        output.coverage.upstream_duplicate_ids = len(result.duplicate_document_ids)
        output.coverage.rejection_reasons = dict(sorted(output.coverage.rejection_reasons.items()))
        output.coverage.resolution_excluded_reasons = dict(sorted(output.coverage.resolution_excluded_reasons.items()))
        output.status = ("partial_failure" if output.invalid_record_indexes or result.invalid_records or result.duplicate_document_ids or any(
            not thread.coverage_complete for thread in output.threads
        ) else "completed") if result.records or result.invalid_records or result.duplicate_document_ids else "skipped"
        return output

    @staticmethod
    def _batch_errors(records, batch_outputs) -> Dict[str, str]:
        """If batch metadata is supplied it must agree with per-record outputs.

        Missing batches are allowed: per-record prediction envelopes are a public
        input too. A foreign/duplicate batch ID invalidates that stage's handoff.
        Local echo failures remain local to their own thread below.
        """
        from app.services.nlp.pipeline import RESULT_SCHEMAS, _dump, _validate_response

        errors: Dict[str, str] = {}
        by_id = {record.document_id: record for record in records}
        for name in ("issue", "response", "solution"):
            if name not in batch_outputs:
                continue
            try:
                batch = RESULT_SCHEMAS[name].model_validate(_dump(batch_outputs[name]), strict=True)
                ids = [item.document_id for item in batch.predictions]
                expected = {row.document_id for row in records if name in row.stage_outputs}
                if len(set(ids)) != len(ids) or set(ids) != expected or batch.n_documents != len(ids):
                    raise ValueError("batch_prediction_coverage_mismatch")
                _unit(batch.threshold)
                if not batch.model_name.strip() or not batch.model_version.strip():
                    raise ValueError("missing_model_identity")
                if name == "response":
                    _validate_response(batch)
                for prediction in batch.predictions:
                    if prediction.document_id not in by_id:
                        raise ValueError("foreign_prediction_id")
                    try:
                        raw = _dump_prediction(by_id[prediction.document_id].stage_outputs[name])
                        if prediction.model_dump() != raw:
                            raise ValueError("batch_record_prediction_mismatch")
                    except Exception:
                        # A mismatch is attached to this record rather than making
                        # an unrelated thread lose its otherwise valid predictions.
                        errors[f"{name}:{prediction.document_id}"] = "batch_record_prediction_mismatch"
            except Exception:
                errors[name] = "invalid_batch_metadata"
        return errors

    @staticmethod
    def _reference_errors(records, documents, comments) -> List[str]:
        errors = set()
        key = _key(records[0])
        identities = [(row.post_id, row.comment_id) for row in records]
        if len(set(identities)) != len(identities):
            errors.add("duplicate_source_identity")
        by_comment = {row.comment_id: row for row in records if row.comment_id is not None}
        post = next((row for row in records if row.comment_id is None), None)
        parents = {}
        for row in records:
            if len(documents[row.document_id]) != 1:
                errors.add("duplicate_document_id")
            parent = row.parent_comment_id
            reddit = row.parent_reddit_id
            if reddit is not None:
                if reddit.startswith("t1_") and reddit[3:]:
                    if parent is not None and parent != reddit[3:]:
                        errors.add("conflicting_parent_identity")
                    parent = reddit[3:]
                elif reddit.startswith("t3_") and reddit[3:]:
                    if reddit[3:] != row.post_id or parent is not None:
                        errors.add("conflicting_parent_identity")
                else:
                    errors.add("invalid_parent_reddit_id")
            if parent:
                parents[row.comment_id] = parent
                if parent == row.comment_id:
                    errors.add("self_parent")
                if parent in comments and key not in comments[parent]:
                    errors.add("cross_thread_parent")
                if parent in by_comment:
                    source_parent = by_comment[parent]
                    if source_parent.created_at > row.created_at:
                        errors.add("parent_after_child")
                    if (row.parent_comment_text is not None
                            and row.parent_comment_text not in (source_parent.text, source_parent.original_text)):
                        errors.add("conflicting_parent_text")
            if row.comment_id is not None and post and row.created_at < post.created_at:
                errors.add("comment_before_post")
        # Equal timestamps do not rule out a cycle, so check identity separately.
        for comment in parents:
            visited = set()
            while comment in parents:
                if comment in visited:
                    errors.add("parent_cycle")
                    break
                visited.add(comment)
                comment = parents[comment]
        return sorted(errors)

    @staticmethod
    def _prediction(record, name, batch_errors, raw):
        from app.services.nlp.pipeline import _model_input

        if name in batch_errors or f"{name}:{record.document_id}" in batch_errors:
            raise ValueError("invalid_batch_metadata")
        if name in record.stage_errors or "preprocessing" in record.stage_errors:
            raise ValueError("upstream_stage_failed")
        if not isinstance(record.normalized_text, str) or not record.normalized_text.strip():
            raise ValueError("missing_normalized_text")
        _require_assessment_fields(name, raw)
        # Do not let permissive defaults on neutral stage schemas mask omissions.
        expected = _model_input(record, name).model_dump()
        echoed = ("document_id", "post_id", "comment_id", "text", "original_text",
                  "language", "script", "is_code_mixed", "detected_languages")
        if name != "issue":
            echoed += ("parent_comment_text", "issue_labels", "issue_probabilities")
        if name == "solution":
            echoed += ("response_type", "response_probabilities")
        for field in echoed:
            if field not in raw or raw[field] != expected[field]:
                raise ValueError("prediction_source_metadata_mismatch")
        for field in ("source_mode", "source_subreddit", "created_at", "url",
                      "parent_comment_id", "parent_reddit_id"):
            if field in raw and raw[field] != getattr(record, field):
                raise ValueError("prediction_source_metadata_mismatch")
        if raw.get("topic_id") is not None or raw.get("topic_label") is not None or raw.get("topic_keywords"):
            raise ValueError("topic_context_not_available")
        schema = {"issue": IssuePrediction, "response": ResponsePrediction, "solution": SolutionPrediction}[name]
        prediction = schema.model_validate(raw, strict=True)
        if name == "issue":
            labels = prediction.predicted_issue_labels
            if len(set(labels)) != len(labels) or any(not label.strip() or label != label.strip() for label in labels):
                raise ValueError("invalid_issue_labels")
            for probability in prediction.issue_probabilities.values():
                _unit(probability)
            if not set(labels) <= set(prediction.issue_probabilities):
                raise ValueError("missing_issue_probability")
            if any(prediction.issue_probabilities[label] <= 0 for label in labels):
                raise ValueError("unsupported_issue_label")
            _unit(prediction.confidence)
        elif name == "solution":
            for probability in (prediction.solution_probability, prediction.not_solution_probability, prediction.threshold):
                _unit(probability)
            if abs(prediction.solution_probability + prediction.not_solution_probability - 1) > 1e-5:
                raise ValueError("invalid_solution_distribution")
            if prediction.is_solution != (prediction.solution_probability >= prediction.threshold and not prediction.abstained):
                raise ValueError("invalid_solution_decision")
            if prediction.abstained and prediction.solution_probability < prediction.threshold:
                raise ValueError("invalid_solution_abstention")
        else:
            probabilities = prediction.response_probabilities
            if not probabilities or not set(probabilities) <= set(RESPONSE_TYPES):
                raise ValueError("invalid_response_taxonomy")
            for probability in probabilities.values():
                _unit(probability)
            _unit(prediction.confidence)
            if abs(sum(probabilities.values()) - 1) > 1e-5:
                raise ValueError("invalid_response_distribution")
            if prediction.response_type is not None and prediction.response_type not in probabilities:
                raise ValueError("invalid_response_type")
            if prediction.abstained != (prediction.response_type is None):
                raise ValueError("invalid_response_abstention")
            if abs(prediction.confidence - max(probabilities.values())) > 1e-5:
                raise ValueError("invalid_response_confidence")
            if prediction.response_type is not None and probabilities[prediction.response_type] != max(probabilities.values()):
                raise ValueError("invalid_response_decision")
            prediction.response_probabilities = {role: probabilities[role] for role in RESPONSE_TYPES if role in probabilities}
        return prediction

    def _process_thread(self, thread, records, batch_errors, *, source_coverage_valid) -> None:
        coverage = thread.coverage
        coverage.records_received = len(records)
        coverage.threads_received = 1
        coverage.records_validated = 0 if thread.validation_errors else len(records)
        coverage.records_rejected = len(records) - coverage.records_validated
        coverage.threads_validated = int(not thread.validation_errors)
        if thread.validation_errors:
            for reason in thread.validation_errors:
                _count_reason(coverage.rejection_reasons, reason)

        predictions = {name: {} for name in ("issue", "response", "solution")}
        raw_pairs = set()
        raw_positive_solutions = set()
        for row in records:
            for name in predictions:
                if name not in row.stage_outputs:
                    continue
                received = f"{name}_predictions_received"
                setattr(coverage, received, getattr(coverage, received) + 1)
                try:
                    # Count explicit, shallow observations even when another
                    # payload field cannot be serialized. These are not evidence.
                    raw = row.stage_outputs[name]
                    observed = vars(raw) if isinstance(raw, BaseModel) else raw
                    if name == "issue" and isinstance(observed, dict):
                        labels = observed.get("predicted_issue_labels", [])
                        if isinstance(labels, list) and len(labels) <= MAX_PREDICTION_ITEMS:
                            raw_pairs.update((row.document_id, label) for label in labels
                                             if isinstance(label, str) and label.strip())
                    if name == "solution" and isinstance(observed, dict) and observed.get("is_solution") is True:
                        raw_positive_solutions.add(row.document_id)
                    raw = _dump_prediction(raw)
                    if thread.validation_errors:
                        raise ValueError("invalid_thread_metadata")
                    if name != "issue" and row.comment_id is None:
                        raise ValueError("comment_only_prediction")
                    prediction = self._prediction(row, name, batch_errors, raw)
                    predictions[name][row.document_id] = prediction
                    validated = f"{name}_predictions_validated"
                    setattr(coverage, validated, getattr(coverage, validated) + 1)
                except Exception:
                    thread.prediction_rejections[f"{name}:{row.document_id}"] = "invalid_prediction"
                    _count_reason(coverage.rejection_reasons, f"invalid_{name}_prediction")

        coverage.issues_received = len({doc_id for doc_id, _ in raw_pairs})
        coverage.issue_labels_received = len(raw_pairs)
        coverage.solutions_received = len(raw_positive_solutions)
        comments = [row for row in records if row.comment_id is not None]
        by_id = {row.document_id: row for row in records}
        for prediction in predictions["response"].values():
            if prediction.response_type is not None:
                thread.response_role_histogram[prediction.response_type] += 1
        thread.prediction_processing_complete = (
            source_coverage_valid and not thread.validation_errors
            and len(predictions["issue"]) == len(records)
            and len(predictions["solution"]) == len(comments)
        )
        issues = []
        for doc_id, prediction in sorted(predictions["issue"].items()):
            if prediction.predicted_issue_labels:
                row = by_id[doc_id]
                issues.append(IssueEvidence(
                    **self._common(row, prediction), issue_labels=sorted(prediction.predicted_issue_labels),
                    issue_probabilities=prediction.issue_probabilities,
                ))
        solutions = []
        for doc_id, prediction in sorted(predictions["solution"].items()):
            if _positive(prediction):
                row = by_id[doc_id]
                # Response context is used only when independently validated.
                response = predictions["response"].get(doc_id)
                solutions.append(SolutionEvidence(
                    **self._common(row, prediction), is_solution=True,
                    solution_probability=prediction.solution_probability,
                    response_type=response.response_type if response else None,
                    response_probabilities=response.response_probabilities if response else {},
                    parent_comment_id=row.parent_comment_id or (
                        row.parent_reddit_id[3:] if row.parent_reddit_id and row.parent_reddit_id.startswith("t1_") else None
                    ), parent_comment_text=row.parent_comment_text,
                ))
        coverage.issues_validated = len(issues)
        coverage.issue_labels_validated = sum(len(issue.issue_labels) for issue in issues)
        coverage.solutions_validated = len(solutions)
        coverage.issues_linking_eligible = len(issues) if solutions else 0
        coverage.threads_linking_eligible = int(bool(issues and solutions))
        self._link(thread, issues, solutions, by_id)
        self._resolve(thread, issues, raw_pairs, comments, predictions)
        thread.coverage_complete = (
            thread.discussion_complete and thread.prediction_processing_complete
            and not thread.prediction_rejections
            and coverage.resolution_outputs == coverage.resolution_pairs_received
            and thread.linking.status not in {"failed", "unavailable"}
            and thread.resolution.status not in {"failed", "unavailable"}
        )
        thread.status = "completed" if thread.coverage_complete else "partial_failure"
        coverage.threads_complete = int(thread.coverage_complete)
        coverage.threads_partial = int(not thread.coverage_complete)
        coverage.rejection_reasons = dict(sorted(coverage.rejection_reasons.items()))
        coverage.resolution_excluded_reasons = dict(sorted(coverage.resolution_excluded_reasons.items()))

    @staticmethod
    def _common(row, prediction):
        return dict(
            document_id=row.document_id, post_id=row.post_id, comment_id=row.comment_id,
            original_text=row.original_text, normalized_text=row.normalized_text,
            language=prediction.language, script=prediction.script,
            is_code_mixed=prediction.is_code_mixed, detected_languages=prediction.detected_languages,
        )

    def _link(self, thread, issues, solutions, by_id) -> None:
        coverage = thread.coverage
        if self.linker is None:
            thread.linking = EvidenceOperation(status="unavailable", detail="not_configured")
            return
        if isinstance(self.linker, TfidfCosineIssueSolutionLinker):
            thread.linking = EvidenceOperation(status="unavailable", detail="linker_requires_fitting")
            return
        if not isinstance(self.linker, IssueSolutionLinking):
            thread.linking = EvidenceOperation(status="unavailable", detail="invalid_linker_interface")
            return
        if not issues or not solutions:
            thread.linking.detail = "no_eligible_evidence"
            return
        coverage.link_attempts = 1
        phase = "linking_failed"
        try:
            # Inputs are disposable copies; mutated model arguments never replace
            # the authoritative evidence used for result validation.
            raw = self.linker.link(deepcopy(issues), deepcopy(solutions))
            phase = "invalid_linking_result"
            from app.services.nlp.pipeline import _dump
            payload = _dump(raw)
            coverage.links_returned = len(payload.get("links", [])) if isinstance(payload, dict) else 0
            result = LinkingResult.model_validate(payload, strict=True)
            _unit(result.link_threshold)
            if not result.model_name.strip() or not result.model_version.strip():
                raise ValueError("missing_model_identity")
            issue_map = {item.document_id: item for item in issues}
            solution_map = {item.document_id: item for item in solutions}
            seen = set()
            for link in result.links:
                pair = (link.issue_document_id, link.issue_label, link.solution_document_id)
                if pair in seen:
                    raise ValueError("duplicate_link")
                seen.add(pair)
                issue = issue_map[link.issue_document_id]
                solution = solution_map[link.solution_document_id]
                if link.issue_label not in issue.issue_labels:
                    raise ValueError("foreign_issue_label")
                for prefix, evidence in (("issue", issue), ("solution", solution)):
                    for suffix, source in (("post_id", "post_id"), ("comment_id", "comment_id"),
                                           ("text", "normalized_text"), ("original_text", "original_text")):
                        if getattr(link, f"{prefix}_{suffix}") != getattr(evidence, source):
                            raise ValueError("link_source_metadata_mismatch")
                if (link.model_name, link.model_version, link.link_threshold) != (
                    result.model_name, result.model_version, result.link_threshold
                ):
                    raise ValueError("link_model_metadata_mismatch")
                if (link.language, link.script, link.is_code_mixed) != (
                    solution.language, solution.script, issue.is_code_mixed or solution.is_code_mixed
                ):
                    raise ValueError("link_language_metadata_mismatch")
                for field in ("similarity_score", "solution_probability", "thread_relevance",
                              "response_role_score", "link_score", "confidence", "link_threshold"):
                    _unit(getattr(link, field))
                for score in link.score_components.values():
                    _unit(score)
                for component, field in (("semantic_similarity", "similarity_score"),
                                         ("solution_probability", "solution_probability"),
                                         ("thread_relevance", "thread_relevance"),
                                         ("response_role", "response_role_score")):
                    if component in link.score_components and link.score_components[component] != getattr(link, field):
                        raise ValueError("link_component_mismatch")
                if abs(link.solution_probability - solution.solution_probability) > 1e-6:
                    raise ValueError("link_probability_mismatch")
                if link.linked != (link.link_score >= link.link_threshold):
                    raise ValueError("invalid_link_decision")
                # Explicit thread claims must be consistent even though they do
                # not determine membership (the source key does).
                parent_child = solution.parent_comment_id == issue.comment_id and issue.comment_id is not None
                for field, expected in (("same_post", True), ("parent_child", parent_child)):
                    if field in link.signals and link.signals[field] is not expected:
                        raise ValueError("link_thread_signal_mismatch")
                if link.signals.get("thread_reason", "same_post") not in {"same_post", "parent_child"}:
                    raise ValueError("cross_thread_link_signal")
                if link.signals.get("thread_reason") == "parent_child" and not parent_child:
                    raise ValueError("link_parent_signal_mismatch")
                if "response_type" in link.signals and link.signals["response_type"] != (solution.response_type or ""):
                    raise ValueError("link_response_metadata_mismatch")
            labels = {issue.document_id: sorted(issue.issue_labels) for issue in issues}
            asserted = [link for link in result.links if link.linked]
            counts = dict(
                n_issues=len(issues), n_solutions=len(solutions),
                n_solution_candidates=len({link.solution_document_id for link in result.links}),
                n_issue_labels=sum(len(value) for value in labels.values()), n_scored_links=len(result.links),
                n_linked=len(asserted), n_linked_issues=len({link.issue_document_id for link in asserted}),
                n_linked_solutions=len({link.solution_document_id for link in asserted}),
            )
            if result.issue_labels_by_document != labels or any(getattr(result, field) != count for field, count in counts.items()):
                raise ValueError("link_result_count_mismatch")
            if (result.model_name, result.model_version) != (self.linker.name, self.linker.version):
                raise ValueError("link_model_identity_mismatch")
            # Coverage still comes from actual inputs/operations, independently
            # of model summary fields that were checked above.
            result.links.sort(key=lambda link: (link.issue_document_id, link.issue_label, link.solution_document_id))
            thread.linking_result = result
            coverage.links_validated = len(result.links)
            for link in result.links:
                reason = None
                if link.issue_document_id == link.solution_document_id:
                    reason = "self_link"
                elif by_id[link.solution_document_id].created_at < by_id[link.issue_document_id].created_at:
                    reason = "solution_before_issue"
                elif not link.linked or link.link_score <= 0:
                    reason = "not_positive_link"
                elif link.similarity_score <= 0 or link.signals.get("issue_zero_vector") or link.signals.get("solution_zero_vector"):
                    reason = "no_semantic_support"
                if reason:
                    coverage.link_rejections += 1
                    _count_reason(coverage.rejection_reasons, reason)
                else:
                    thread.supported_links.append(link)
            coverage.links_supported = len(thread.supported_links)
            thread.linking.status = "completed"
        except Exception as exc:
            coverage.link_failures = 1
            coverage.link_rejections = coverage.links_returned
            if coverage.links_returned:
                _count_reason(coverage.rejection_reasons, "invalid_linking_result", coverage.links_returned)
            thread.linking = EvidenceOperation(status="failed", detail=phase, error_type=type(exc).__name__)

    def _resolve(self, thread, issues, raw_pairs, comments, predictions) -> None:
        coverage = thread.coverage
        valid_pairs = {(issue.document_id, label) for issue in issues for label in issue.issue_labels}
        coverage.resolution_pairs_received = len(raw_pairs)
        threshold = 0.6
        minimum_volume = 3
        if self.resolver is not None and isinstance(self.resolver, ResolutionDetection):
            try:
                config = self.resolver.config
                threshold = config.strong_link_threshold
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                    raise ValueError("invalid_resolver_config")
                _unit(threshold)
                minimum_volume = config.minimum_discussion_volume
                if type(minimum_volume) is not int or minimum_volume < 0:
                    raise ValueError("invalid_resolver_config")
            except Exception:
                threshold = None
                minimum_volume = None
        evidence = []
        for doc_id, label in sorted(raw_pairs):
            links = [link for link in thread.supported_links
                     if (link.issue_document_id, link.issue_label) == (doc_id, label)]
            reasons = []
            if (doc_id, label) not in valid_pairs:
                reasons.append("invalid_issue_prediction")
            if thread.validation_errors:
                reasons.append("invalid_thread_metadata")
            if not thread.discussion_complete:
                reasons.append("discussion_completeness_not_asserted")
            if not thread.prediction_processing_complete:
                reasons.append("incomplete_issue_solution_processing")
            if thread.linking.status != "completed":
                reasons.append("linking_not_completed")
            if not links:
                reasons.append("no_supported_positive_links")
            if threshold is None:
                reasons.append("invalid_resolver_config")
            strong = [link.solution_document_id for link in links
                       if threshold is not None and link.link_score >= threshold]
            if minimum_volume is not None and len(comments) < minimum_volume:
                reasons.append("insufficient_discussion_volume")
            if threshold is not None and links and not strong:
                reasons.append("no_supported_strong_links")
            pair = IssueLabelCoverage(
                issue_id=doc_id, issue_label=label, total_comments=len(comments),
                solution_predictions_validated=len(predictions["solution"]),
                linked_solution_ids=sorted({link.solution_document_id for link in links}),
                strong_solution_ids=sorted(set(strong)), excluded_reasons=reasons,
                resolution_eligible=not reasons,
            )
            thread.issue_labels.append(pair)
            if reasons:
                coverage.resolution_excluded += 1
                for reason in reasons:
                    _count_reason(coverage.resolution_excluded_reasons, reason)
                continue
            item = IssueResolutionEvidence(
                issue_id=doc_id, issue_label=label, post_id=thread.post_id,
                total_comments=len(comments), discussion_volume=len(comments),
                solution_candidate_count=len(predictions["solution"]),
                linked_solution_count=len(pair.linked_solution_ids), strong_link_count=len(pair.strong_solution_ids),
                link_scores=[link.link_score for link in links],
                solution_probabilities=[predictions["solution"][link.solution_document_id].solution_probability for link in links],
                response_counts=dict(thread.response_role_histogram),
                agreement_count=thread.response_role_histogram["agreement"],
                experience_count=thread.response_role_histogram["personal_experience"],
                opinion_count=thread.response_role_histogram["opinion"],
                advice_count=thread.response_role_histogram["advice"],
                solution_count=thread.response_role_histogram["solution"],
                counterargument_count=thread.response_role_histogram["counterargument"],
                temporal=TemporalEvidence(solution_after_issue=True),
            )
            derive_evidence(item)  # actual downstream validation, before detect()
            evidence.append(item)
        coverage.resolution_eligible = len(evidence)
        thread.resolution_evidence = evidence
        if self.resolver is None:
            thread.resolution = EvidenceOperation(status="unavailable", detail="not_configured")
            return
        if not isinstance(self.resolver, ResolutionDetection):
            thread.resolution = EvidenceOperation(status="unavailable", detail="invalid_resolver_interface")
            return
        if not evidence:
            thread.resolution.detail = "no_eligible_evidence"
            return
        coverage.resolution_attempts = 1
        coverage.resolution_pairs_attempted = len(evidence)
        phase = "resolution_failed"
        try:
            from app.services.nlp.pipeline import _dump
            raw = self.resolver.detect(deepcopy(evidence))
            phase = "invalid_resolution_result"
            result = ResolutionBatchResult.model_validate(_dump(raw), strict=True)
            expected = {(item.issue_id, item.issue_label): item for item in evidence}
            ids = [(item.issue_id, item.issue_label) for item in result.results]
            if len(ids) != len(expected) or set(ids) != set(expected):
                raise ValueError("resolution_id_coverage_mismatch")
            if not result.model_name.strip() or not result.model_version.strip():
                raise ValueError("missing_model_identity")
            for item in result.results:
                source = expected[(item.issue_id, item.issue_label)]
                if item.post_id != thread.post_id or item.evidence.model_dump() != source.model_dump():
                    raise ValueError("resolution_source_metadata_mismatch")
                if (item.model_name, item.model_version) != (result.model_name, result.model_version):
                    raise ValueError("resolution_model_metadata_mismatch")
                derived = derive_evidence(source)
                if item.evidence_sufficient is not (derived.effective_discussion_volume >= minimum_volume):
                    raise ValueError("resolution_evidence_sufficiency_mismatch")
                for field in ("resolution_score", "confidence"):
                    _unit(getattr(item, field))
                for field in ("solution_density", "strong_solution_density", "best_link_score",
                              "solution_probability_mean", "linked_solution_count", "strong_link_count",
                              "solution_candidate_count", "effective_discussion_volume"):
                    if getattr(item, field) != getattr(derived, field):
                        raise ValueError("resolution_derived_evidence_mismatch")
                if item.temporal != source.temporal:
                    raise ValueError("resolution_temporal_mismatch")
                if item.score_components != derived.as_components():
                    raise ValueError("resolution_components_mismatch")
            if (result.model_name, result.model_version) != (self.resolver.name, self.resolver.version):
                raise ValueError("resolution_model_identity_mismatch")
            result.results.sort(key=lambda item: (item.issue_id, item.issue_label))
            thread.resolution_result = result
            coverage.resolution_outputs = len(result.results)
            thread.resolution.status = "completed"
        except Exception as exc:
            coverage.resolution_failures = 1
            coverage.resolution_pairs_failed = len(evidence)
            thread.resolution = EvidenceOperation(status="failed", detail=phase, error_type=type(exc).__name__)


__all__ = [
    "DiscussionCompleteness", "EvidenceCoverage", "IssueLabelCoverage", "ThreadEvidenceAdapter",
    "ThreadEvidenceReport", "ThreadEvidenceResult", "ThreadKey",
]
