"""Offline handoff contracts using typed no-fit link/detect interface doubles.

No corpus-fitting linker, classifier training, artifact, network, or DB calls.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Optional, Sequence
import unittest
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
for directory in (BACKEND, ROOT):
    directory = str(directory)
    if directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)

from app.services.nlp.issue_identification.models.result import IssueIdentificationResult, IssuePrediction
from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence, IssueSolutionLinking, LinkModelConfig, SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.models.result import LinkPrediction, LinkingResult
from app.services.nlp.issue_solution_linking.tfidf_cosine_linker import TfidfCosineIssueSolutionLinker
from app.services.nlp.pipeline import DownstreamPipeline, InvalidRecord, PipelineRecord, PipelineResult
from app.services.nlp.resolution_detection.base import (
    IssueResolutionEvidence, ResolutionDetection, ResolutionModelConfig, ResolutionStatus,
)
from app.services.nlp.resolution_detection.models.evidence import derive_evidence
from app.services.nlp.resolution_detection.models.result import ResolutionBatchResult, ResolutionResult
from app.services.nlp.resolution_detection.evidence_based_resolver import EvidenceBasedResolver
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.response_classification.models.result import ResponsePrediction
from app.services.nlp.solution_detection.models.result import SolutionPrediction
from app.services.nlp.thread_evidence import ThreadEvidenceAdapter
from backend.test_nlp_pipeline import (
    RecordingLanguage, RecordingModel, RecordingPreprocessor, language_analysis, record, stage_result,
)

PRIVATE = "private-model-error-must-not-leak"
TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def discussion(post="p1", *, mode="direct", subreddit="IIMSambalpur", labels=("fees", "roi")):
    """A post plus three unique comments; two positive solution assessments."""
    rows = []
    for number, kind in enumerate(("post", "s1", "s2", "other")):
        doc_id = f"{post}:{kind}"
        comment = None if kind == "post" else f"{post}-{kind}"
        text = f"{kind} hostel fees and return on investment"
        row = PipelineRecord(
            document_id=doc_id, post_id=post, comment_id=comment,
            parent_reddit_id=f"t3_{post}" if comment else None,
            text=text, original_text=f"Original {text}", normalized_text=f"normalized {text}",
            created_at=TIME + timedelta(hours=number), source_mode=mode, source_subreddit=subreddit,
            url=f"https://www.reddit.com/r/{subreddit}/comments/{post}/thread/{comment or ''}",
            language_analysis=language_analysis().model_dump(),
        )
        echo = dict(
            document_id=doc_id, post_id=post, comment_id=comment,
            text=row.normalized_text, original_text=row.original_text,
            language="hi", script="latin", is_code_mixed=True, detected_languages=["hi", "en"],
        )
        own_labels = list(labels) if kind == "post" else []
        probabilities = {label: 0.9 if own_labels else 0.1 for label in labels}
        row.stage_outputs["issue"] = IssuePrediction(
            **echo, predicted_issue_labels=own_labels, issue_probabilities=probabilities, confidence=0.9,
        ).model_dump()
        if comment:
            role = "solution" if kind == "s1" else "advice" if kind == "s2" else "personal_experience"
            context = dict(issue_labels=own_labels, issue_probabilities=probabilities, parent_comment_text=None)
            response = ResponsePrediction(
                **echo, **context, response_type=role, response_probabilities={role: 1.0}, confidence=1.0,
            )
            row.stage_outputs["response"] = response.model_dump()
            positive = kind in {"s1", "s2"}
            row.stage_outputs["solution"] = SolutionPrediction(
                **echo, **context, response_type=role, response_probabilities={role: 1.0},
                is_solution=positive, solution_probability=0.9 if positive else 0.1,
                not_solution_probability=0.1 if positive else 0.9,
                is_candidate=positive,
            ).model_dump()
        rows.append(row)
    return rows


def pipeline_result(*threads):
    return PipelineResult(status="completed", records=[row for rows in threads for row in rows])


class NoFitLinker(IssueSolutionLinking):
    """Existing interface, actual typed results; no learned/fitted implementation."""

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.inputs = []
        self._config = LinkModelConfig()
        self.train = Mock(side_effect=AssertionError("training forbidden"))
        self.fit = Mock(side_effect=AssertionError("fitting forbidden"))
        self.predict = Mock(side_effect=AssertionError("classifier inference forbidden"))

    name = property(lambda self: "offline_no_fit_linker")
    version = property(lambda self: "test-1")
    config = property(lambda self: self._config)
    config_dict = property(lambda self: self._config.model_dump())
    link_threshold = property(lambda self: self._config.link_threshold)

    def link(
        self, issues: Sequence[IssueEvidence], solutions: Sequence[SolutionEvidence],
        link_threshold: Optional[float] = None,
    ) -> LinkingResult:
        self.inputs.append((deepcopy(issues), deepcopy(solutions)))
        links = []
        for issue in issues:
            for label in issue.issue_labels:
                for solution in solutions:
                    links.append(LinkPrediction(
                        issue_document_id=issue.document_id, issue_label=label,
                        solution_document_id=solution.document_id,
                        similarity_score=0.7, solution_probability=solution.solution_probability,
                        thread_relevance=1.0, response_role_score=0.8, link_score=0.8,
                        linked=True, link_threshold=0.5, confidence=0.6,
                        model_name=self.name, model_version=self.version,
                        issue_post_id=issue.post_id, issue_comment_id=issue.comment_id,
                        issue_text=issue.normalized_text, issue_original_text=issue.original_text,
                        solution_post_id=solution.post_id, solution_comment_id=solution.comment_id,
                        solution_text=solution.normalized_text, solution_original_text=solution.original_text,
                        language=solution.language, script=solution.script,
                        is_code_mixed=issue.is_code_mixed or solution.is_code_mixed,
                        signals={"same_post": True, "thread_reason": "same_post"},
                    ))
        result = LinkingResult(
            model_name=self.name, model_version=self.version, link_threshold=0.5, links=links,
            n_issues=len(issues), n_solutions=len(solutions), n_solution_candidates=len(solutions),
            n_issue_labels=sum(len(issue.issue_labels) for issue in issues), n_scored_links=len(links),
            n_linked=len(links), n_linked_issues=len(issues), n_linked_solutions=len(solutions),
            issue_labels_by_document={issue.document_id: sorted(issue.issue_labels) for issue in issues},
        )
        return self.behavior(result, issues, solutions) if self.behavior else result


class RecordingResolver(ResolutionDetection):
    """Typed mock with mechanically coherent echoes, not a quality assertion."""

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.inputs = []
        self._config = ResolutionModelConfig()
        self.train = Mock(side_effect=AssertionError("training forbidden"))
        self.fit = Mock(side_effect=AssertionError("fitting forbidden"))
        self.predict = Mock(side_effect=AssertionError("classifier inference forbidden"))

    name = property(lambda self: "offline_resolver")
    version = property(lambda self: "test-1")
    config = property(lambda self: self._config)
    config_dict = property(lambda self: self._config.model_dump())
    thresholds = property(lambda self: {"unresolved_upper": 0.3, "resolved_lower": 0.55})

    def detect(self, evidence: Sequence[IssueResolutionEvidence]) -> ResolutionBatchResult:
        self.inputs.append(deepcopy(evidence))
        results = []
        for item in evidence:
            derived = derive_evidence(item)
            results.append(ResolutionResult(
                issue_id=item.issue_id, issue_label=item.issue_label, post_id=item.post_id,
                status=ResolutionStatus.PARTIALLY_RESOLVED, resolution_score=0.5, confidence=0.1,
                evidence=item.model_copy(deep=True), score_components=derived.as_components(),
                solution_density=derived.solution_density, strong_solution_density=derived.strong_solution_density,
                best_link_score=derived.best_link_score, solution_probability_mean=derived.solution_probability_mean,
                linked_solution_count=derived.linked_solution_count, strong_link_count=derived.strong_link_count,
                solution_candidate_count=derived.solution_candidate_count,
                effective_discussion_volume=derived.effective_discussion_volume,
                temporal=item.temporal, model_name=self.name, model_version=self.version,
            ))
        batch = ResolutionBatchResult(model_name=self.name, model_version=self.version, results=results)
        return self.behavior(batch, evidence) if self.behavior else batch


class TestThreadEvidence(unittest.TestCase):
    def setUp(self):
        self.linker = NoFitLinker()
        self.resolver = RecordingResolver()
        self.adapter = ThreadEvidenceAdapter(linking_model=self.linker, resolution_model=self.resolver)

    def tearDown(self):
        for model in (self.linker, self.resolver):
            for method in (model.fit, model.train, model.predict):
                method.assert_not_called()

    def test_valid_handoff_uses_real_types_unique_per_label_counts_and_source_envelope(self):
        source = pipeline_result(discussion())
        before = source.model_dump()
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(source.model_dump(), before)
        thread = result.threads[0]
        self.assertTrue(thread.prediction_processing_complete)
        self.assertTrue(thread.coverage_complete)
        self.assertEqual(list(thread.response_role_histogram), list(RESPONSE_TYPES))
        self.assertEqual(thread.response_role_histogram["personal_experience"], 1)
        self.assertEqual(thread.response_role_histogram["solution"], 1)
        self.assertEqual(thread.response_role_histogram["advice"], 1)
        by_id = {row.document_id: row for row in source.records}
        for envelope in thread.records:
            row = by_id[envelope["document_id"]]
            for field in ("source_mode", "source_subreddit", "url", "created_at", "parent_reddit_id", "original_text"):
                self.assertEqual(envelope[field], getattr(row, field))
            self.assertNotIn("stage_outputs", envelope)
        for item in self.resolver.inputs[0]:
            self.assertIsInstance(item, IssueResolutionEvidence)
            self.assertEqual(item.total_comments, 3)
            self.assertEqual(item.discussion_volume, 3)
            self.assertEqual(item.solution_candidate_count, 3)  # assessed, including a negative
            self.assertEqual(item.linked_solution_count, 2)
            self.assertEqual(item.strong_link_count, 2)
            self.assertEqual(item.experience_count, 1)
            self.assertEqual(item.link_scores, [0.8, 0.8])
            self.assertTrue(item.temporal.solution_after_issue)
        for issues, solutions in self.linker.inputs:
            self.assertTrue(all(isinstance(item, IssueEvidence) for item in issues))
            self.assertTrue(all(isinstance(item, SolutionEvidence) for item in solutions))
            self.assertTrue(all(item.comment_id is not None for item in solutions))
        coverage = result.coverage
        self.assertEqual((coverage.records_received, coverage.threads_received, coverage.threads_complete), (4, 1, 1))
        self.assertEqual((coverage.issues_received, coverage.issues_validated, coverage.issues_linking_eligible), (1, 1, 1))
        self.assertEqual((coverage.solution_predictions_received, coverage.solution_predictions_validated), (3, 3))
        self.assertEqual((coverage.solutions_received, coverage.solutions_validated), (2, 2))
        self.assertEqual((coverage.link_attempts, coverage.links_returned, coverage.links_supported), (1, 4, 4))
        self.assertEqual((coverage.resolution_pairs_received, coverage.resolution_eligible, coverage.resolution_excluded), (2, 2, 0))
        self.assertEqual((coverage.resolution_attempts, coverage.resolution_pairs_attempted, coverage.resolution_outputs), (1, 2, 2))

    def test_unknown_completeness_is_not_an_unresolved_verdict(self):
        result = self.adapter.process(pipeline_result(discussion()))
        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(len(self.linker.inputs), 1)
        self.assertEqual(self.resolver.inputs, [])
        self.assertEqual(result.coverage.resolution_outputs, 0)
        self.assertEqual(result.coverage.resolution_excluded_reasons, {"discussion_completeness_not_asserted": 2})
        self.assertEqual(result.coverage.threads_partial, 1)

    def test_complete_flag_requires_literal_true_and_exact_provenance_key(self):
        for flag in (None, 1, "True", {("direct", "IIMSambalpur", "p1"): True},
                     {("context", "iimsambalpur", "p1"): True}):
            with self.subTest(flag=flag):
                result = self.adapter.process(pipeline_result(discussion()), discussion_complete=flag)
                self.assertFalse(result.threads[0].discussion_complete)
                self.assertEqual(result.coverage.resolution_attempts, 0)
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete={
            ("direct", "iimsambalpur", "p1"): True,
        })
        self.assertTrue(result.threads[0].coverage_complete)

    def test_missing_negative_predictions_block_resolution_for_the_whole_thread(self):
        for stage in ("issue", "solution"):
            with self.subTest(stage=stage):
                rows = discussion()
                rows[-1].stage_outputs.pop(stage)
                result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                self.assertFalse(result.threads[0].prediction_processing_complete)
                self.assertEqual(result.coverage.resolution_eligible, 0)
                self.assertEqual(result.coverage.resolution_outputs, 0)
                self.assertEqual(result.coverage.resolution_excluded_reasons["incomplete_issue_solution_processing"], 2)
        self.assertEqual(self.resolver.inputs, [])

    def test_no_solutions_and_no_asserted_links_never_invoke_resolver(self):
        rows = discussion()
        for row in rows[1:]:
            row.stage_outputs["solution"].update(is_solution=False, solution_probability=0.1, not_solution_probability=0.9)
        result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
        self.assertEqual(self.linker.inputs, [])
        self.assertEqual(result.coverage.resolution_outputs, 0)
        self.assertEqual(result.coverage.solutions_validated, 0)
        self.assertTrue(result.threads[0].prediction_processing_complete)

        def below_threshold(batch, *_):
            for link in batch.links:
                link.linked, link.link_score = False, 0.2
            batch.n_linked = batch.n_linked_issues = batch.n_linked_solutions = 0
            return batch

        self.linker.behavior = below_threshold
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.coverage.link_rejections, 4)
        self.assertEqual(result.coverage.rejection_reasons, {"not_positive_link": 4})
        self.assertEqual(self.resolver.inputs, [])

        def no_candidates(batch, *_):
            batch.links = []
            batch.n_solution_candidates = batch.n_scored_links = batch.n_linked = 0
            batch.n_linked_issues = batch.n_linked_solutions = 0
            return batch
        self.linker.behavior = no_candidates
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.threads[0].linking.status, "completed")
        self.assertEqual(result.coverage.links_returned, 0)
        self.assertEqual(result.coverage.resolution_attempts, 0)
        self.assertEqual(self.resolver.inputs, [])

    def test_labels_without_links_are_excluded_while_supported_label_is_assessed(self):
        def fees_only(batch, *_):
            batch.links = [link for link in batch.links if link.issue_label == "fees"]
            batch.n_scored_links = batch.n_linked = len(batch.links)
            return batch
        self.linker.behavior = fees_only
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.coverage.resolution_eligible, 1)
        self.assertEqual(result.coverage.resolution_excluded, 1)
        self.assertEqual(result.coverage.resolution_outputs, 1)
        self.assertEqual([item.issue_label for item in self.resolver.inputs[0]], ["fees"])
        self.assertEqual(result.threads[0].issue_labels[1].excluded_reasons, ["no_supported_positive_links"])
        self.assertFalse(result.threads[0].coverage_complete)

    def test_builtin_corpus_fitting_linker_is_blocked_before_link(self):
        model = TfidfCosineIssueSolutionLinker()
        with patch.object(model, "link", side_effect=AssertionError("must not fit")) as link:
            result = ThreadEvidenceAdapter(linking_model=model, resolution_model=self.resolver).process(
                pipeline_result(discussion()), discussion_complete=True,
            )
        link.assert_not_called()
        self.assertEqual(result.threads[0].linking.detail, "linker_requires_fitting")
        self.assertEqual(result.coverage.link_attempts, 0)
        self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_absent_models_are_explicit_and_never_implicitly_loaded(self):
        result = ThreadEvidenceAdapter().process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.threads[0].linking.detail, "not_configured")
        self.assertEqual(result.threads[0].resolution.detail, "not_configured")
        self.assertEqual(result.coverage.link_attempts, 0)
        self.assertEqual(result.coverage.resolution_outputs, 0)

    def test_grouping_uses_mode_casefolded_subreddit_and_post(self):
        first = discussion("shared")
        first[1].source_subreddit = "iImSaMbAlPuR"
        other_mode = discussion("shared", mode="context")
        other_community = discussion("shared", mode="context", subreddit="MBA")
        for index, rows in enumerate((first, other_mode, other_community)):
            for row in rows:
                row.document_id = f"source{index}:{row.document_id}"
                for prediction in row.stage_outputs.values():
                    prediction["document_id"] = row.document_id
        result = self.adapter.process(pipeline_result(other_community, first, other_mode), discussion_complete=True)
        self.assertEqual([thread.thread_key() for thread in result.threads], [
            ("context", "iimsambalpur", "shared"), ("context", "mba", "shared"),
            ("direct", "iimsambalpur", "shared"),
        ])
        self.assertEqual(result.coverage.threads_complete, 3)
        for issues, solutions in self.linker.inputs:
            self.assertEqual(len({item.document_id.split(":")[0] for item in [*issues, *solutions]}), 1)

    def test_cross_thread_parent_is_rejected_but_other_thread_continues(self):
        bad, good = discussion("bad"), discussion("good")
        bad[1].parent_comment_id, bad[1].parent_reddit_id = "good-other", "t1_good-other"
        result = self.adapter.process(pipeline_result(bad, good), discussion_complete=True)
        self.assertIn("cross_thread_parent", result.threads[0].validation_errors)
        self.assertEqual(result.coverage.records_rejected, 4)
        self.assertEqual(result.coverage.link_attempts, 1)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertTrue(result.threads[1].coverage_complete)

    def test_cyclic_prediction_is_rejected_without_aborting_healthy_thread(self):
        bad, good = discussion("bad"), discussion("good")
        raw = bad[0].stage_outputs["issue"]
        raw["unexpected"] = raw
        raw["error"] = PRIVATE
        result = self.adapter.process(pipeline_result(bad, good), discussion_complete=True)
        rejected, healthy = result.threads
        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(result.invalid_record_indexes, [])
        self.assertEqual((result.coverage.records_received, result.coverage.records_validated), (8, 8))
        self.assertEqual((result.coverage.issue_predictions_received, result.coverage.issue_predictions_validated), (8, 7))
        self.assertEqual((rejected.coverage.issues_received, rejected.coverage.issue_labels_received), (1, 2))
        self.assertEqual(rejected.prediction_rejections, {"issue:bad:post": "invalid_prediction"})
        self.assertFalse(rejected.prediction_processing_complete)
        self.assertFalse(rejected.coverage_complete)
        self.assertEqual(rejected.coverage.resolution_outputs, 0)
        self.assertIsNone(rejected.resolution_result)
        self.assertEqual(rejected.coverage.resolution_excluded, 2)
        self.assertTrue(healthy.coverage_complete)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertEqual(len(self.linker.inputs), 1)
        self.assertEqual({item.post_id for item in self.linker.inputs[0][0]}, {"good"})
        self.assertEqual({item.post_id for item in self.resolver.inputs[0]}, {"good"})
        self.assertIs(raw["unexpected"], raw)  # no in-place source repair
        self.assertEqual(raw["error"], PRIVATE)
        envelope = next(row for row in rejected.records if row["document_id"] == "bad:post")
        self.assertEqual(envelope["original_text"], bad[0].original_text)
        self.assertNotIn(PRIVATE, result.model_dump_json())
        self.assertNotIn("unexpected", result.model_dump_json())

    def test_malformed_and_bounded_payload_rejections_stay_local(self):
        class CannotCopy:
            def __deepcopy__(self, memo):
                raise RuntimeError(PRIVATE)

        class CannotDump(IssuePrediction):
            def model_dump(self, **kwargs):
                raise RuntimeError(PRIVATE)

        deep = PRIVATE
        for _ in range(100):
            deep = {"nested": deep}
        cycle = []
        cycle.append(cycle)
        for payload in (CannotCopy(), deep, [PRIVATE] * 20000, cycle, {1: PRIVATE}):
            with self.subTest(payload_type=type(payload).__name__):
                bad = discussion("bad")
                bad[0].stage_outputs["issue"]["unexpected"] = payload
                result = self.adapter.process(pipeline_result(bad, discussion("good")), discussion_complete=True)
                self.assertEqual(result.invalid_record_indexes, [])
                self.assertEqual(result.threads[0].prediction_rejections, {"issue:bad:post": "invalid_prediction"})
                self.assertEqual((result.coverage.issue_predictions_received, result.coverage.issue_predictions_validated), (8, 7))
                self.assertEqual(result.threads[0].coverage.resolution_outputs, 0)
                self.assertTrue(result.threads[1].coverage_complete)
                self.assertEqual(result.coverage.resolution_outputs, 2)
                self.assertNotIn(PRIVATE, result.model_dump_json())

        for payload in (None, [PRIVATE], CannotDump(**discussion("bad")[0].stage_outputs["issue"])):
            with self.subTest(root_type=type(payload).__name__):
                bad = discussion("bad")
                bad[0].stage_outputs["issue"] = payload
                result = self.adapter.process(pipeline_result(bad, discussion("good")), discussion_complete=True)
                self.assertEqual(result.invalid_record_indexes, [])
                self.assertEqual(result.threads[0].prediction_rejections, {"issue:bad:post": "invalid_prediction"})
                self.assertEqual(result.threads[0].coverage.issues_validated, 0)
                self.assertIsNone(result.threads[0].resolution_result)
                self.assertTrue(result.threads[1].coverage_complete)
                self.assertEqual(result.coverage.resolution_outputs, 2)
                self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_cyclic_response_and_solution_envelopes_are_counted_and_rejected(self):
        for stage in ("response", "solution"):
            with self.subTest(stage=stage):
                bad = discussion("bad")
                raw = bad[1].stage_outputs[stage]
                raw["unexpected"] = raw
                raw["error"] = PRIVATE
                result = self.adapter.process(pipeline_result(bad, discussion("good")), discussion_complete=True)
                self.assertEqual(result.threads[0].prediction_rejections, {f"{stage}:bad:s1": "invalid_prediction"})
                self.assertEqual(getattr(result.coverage, f"{stage}_predictions_received"), 6)
                self.assertEqual(getattr(result.coverage, f"{stage}_predictions_validated"), 5)
                if stage == "solution":
                    self.assertEqual(result.threads[0].coverage.solutions_received, 2)
                    self.assertEqual(result.threads[0].coverage.solutions_validated, 1)
                    self.assertIsNone(result.threads[0].resolution_result)
                self.assertTrue(result.threads[1].coverage_complete)
                self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_unserializable_record_echo_does_not_invalidate_healthy_batch(self):
        source = pipeline_result(discussion("bad"), discussion("good"))
        source.batch_outputs["issue"] = IssueIdentificationResult(
            model_name="offline", model_version="test-1", threshold=0.5,
            n_documents=len(source.records), known_labels=["fees", "roi"],
            predictions=[IssuePrediction(**row.stage_outputs["issue"]) for row in source.records],
        ).model_dump()
        raw = source.records[0].stage_outputs["issue"]
        raw["unexpected"] = raw
        raw["error"] = PRIVATE
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.threads[0].prediction_rejections, {"issue:bad:post": "invalid_prediction"})
        self.assertEqual(result.coverage.issue_predictions_validated, 7)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertTrue(result.threads[1].coverage_complete)
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_shared_acyclic_payload_and_fixture_dump_deepcopy_remain_sound(self):
        source = pipeline_result(discussion())
        shared = ["benign-extra"]
        source.records[0].stage_outputs["issue"]["unexpected"] = {"first": shared, "second": shared}
        before = source.model_dump()
        copied = deepcopy(source)
        self.assertEqual(copied.model_dump(), before)
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertEqual(source.model_dump(), before)
        self.assertEqual(copied.model_dump(), before)
        self.assertIs(source.records[0].stage_outputs["issue"]["unexpected"]["first"], shared)

    def test_conflicting_parent_metadata_and_timestamps_are_rejected(self):
        cases = [
            ({"parent_comment_id": "p1-other", "parent_reddit_id": "t1_other"}, "conflicting_parent_identity"),
            ({"parent_reddit_id": "t3_elsewhere"}, "conflicting_parent_identity"),
            ({"parent_reddit_id": "bare_parent"}, "invalid_parent_reddit_id"),
            ({"parent_comment_id": "p1-other", "parent_reddit_id": "t1_p1-other"}, "parent_after_child"),
            ({"created_at": TIME - timedelta(hours=1)}, "comment_before_post"),
        ]
        for changes, reason in cases:
            with self.subTest(reason=reason):
                rows = discussion()
                for field, value in changes.items():
                    setattr(rows[1], field, value)
                result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                self.assertIn(reason, result.threads[0].validation_errors)
                self.assertEqual(result.coverage.link_attempts, 0)
                self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_equal_time_parent_cycle_is_rejected(self):
        rows = discussion()
        rows[1].parent_comment_id, rows[1].parent_reddit_id = "p1-s2", "t1_p1-s2"
        rows[2].parent_comment_id, rows[2].parent_reddit_id = "p1-s1", "t1_p1-s1"
        rows[2].created_at = rows[1].created_at
        result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
        self.assertIn("parent_cycle", result.threads[0].validation_errors)
        self.assertEqual(result.coverage.link_attempts, 0)

    def test_source_and_prediction_metadata_mismatch_is_rejected(self):
        for stage in ("issue", "response", "solution"):
            for field, value in (("post_id", "foreign"), ("document_id", "foreign"),
                                 ("text", PRIVATE), ("original_text", PRIVATE),
                                 ("created_at", TIME - timedelta(days=1)), ("source_mode", "context")):
                with self.subTest(stage=stage, field=field):
                    rows = discussion()
                    row = rows[0] if stage == "issue" else rows[1]
                    row.stage_outputs[stage][field] = value
                    result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                    self.assertIn(f"{stage}:{row.document_id}", result.threads[0].prediction_rejections)
                    self.assertNotIn(PRIVATE, result.model_dump_json())
                    if stage != "response":
                        self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_invalid_distribution_labels_and_post_solution_are_not_validated(self):
        cases = [
            ("issue", "predicted_issue_labels", ["fees", "fees"]),
            ("issue", "issue_probabilities", {"fees": float("nan"), "roi": 0.9}),
            ("solution", "solution_probability", float("inf")),
            ("solution", "not_solution_probability", 0.9),
            ("response", "response_type", "new-role"),
        ]
        for stage, field, value in cases:
            with self.subTest(stage=stage, field=field):
                rows = discussion()
                row = rows[0] if stage == "issue" else rows[1]
                row.stage_outputs[stage][field] = value
                result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                self.assertIn(f"{stage}:{row.document_id}", result.threads[0].prediction_rejections)
        rows = discussion()
        rows[0].stage_outputs["solution"] = deepcopy(rows[1].stage_outputs["solution"])
        result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
        self.assertIn("solution:p1:post", result.threads[0].prediction_rejections)

    def test_missing_decision_fields_cannot_default_to_processed_negatives(self):
        for name, fields in (
            ("issue", ("predicted_issue_labels", "issue_probabilities")),
            ("solution", ("is_solution", "solution_probability", "abstained")),
        ):
            for field in fields:
                with self.subTest(stage=name, field=field):
                    rows = discussion()
                    # The absent decision is on a negative record, so other valid
                    # positive links still exist and cannot hide the coverage gap.
                    rows[-1].stage_outputs[name].pop(field)
                    result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                    self.assertFalse(result.threads[0].prediction_processing_complete)
                    self.assertEqual(result.coverage.resolution_attempts, 0)

        def omitted_decision(documents):
            payload = stage_result("solution", documents).model_dump()
            payload["predictions"][0].pop("is_solution")
            return payload

        calls = []
        model = RecordingModel("solution", calls, behavior=omitted_decision)
        result = DownstreamPipeline(
            language_service=RecordingLanguage(calls), preprocessor=RecordingPreprocessor(calls),
            solution_model=model,
        ).run([record()])
        self.assertNotIn("solution", result.batch_outputs)
        self.assertEqual(result.records[0].stage_errors["solution"], "ValueError")
        model.train.assert_not_called()
        model.fit.assert_not_called()

    def test_returned_link_foreign_ids_labels_metadata_and_counts_fail_atomically(self):
        cases = [("solution_document_id", "other:s1"), ("issue_document_id", "other:post"),
                 ("issue_label", "not-predicted"), ("issue_post_id", "foreign"),
                 ("solution_original_text", PRIVATE), ("solution_probability", 0.8),
                 ("model_name", "foreign-model"), ("similarity_score", float("nan"))]
        for field, value in cases:
            with self.subTest(field=field):
                def change(batch, *_, field=field, value=value):
                    setattr(batch.links[0], field, value)
                    return batch
                self.linker.behavior = change
                result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
                self.assertEqual(result.coverage.link_failures, 1)
                self.assertEqual(result.coverage.links_supported, 0)
                self.assertIsNone(result.threads[0].linking_result)
                self.assertEqual(result.coverage.resolution_attempts, 0)
                self.assertNotIn(PRIVATE, result.model_dump_json())
        self.linker.behavior = lambda batch, *_: batch.model_copy(update={"n_issues": 9000})
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.coverage.issues_received, 1)
        self.assertEqual(result.coverage.link_failures, 1)

    def test_duplicate_returned_links_never_inflate_unique_solution_counts(self):
        def duplicate(batch, *_):
            batch.links.append(batch.links[0].model_copy())
            return batch
        self.linker.behavior = duplicate
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.coverage.links_returned, 5)
        self.assertEqual(result.coverage.link_rejections, 5)
        self.assertEqual(result.coverage.links_supported, 0)
        self.assertEqual(result.coverage.resolution_outputs, 0)

    def test_zero_semantic_support_cannot_be_resolved(self):
        def unsupported(batch, *_):
            for link in batch.links:
                link.similarity_score = 0.0
            return batch
        self.linker.behavior = unsupported
        result = self.adapter.process(pipeline_result(discussion()), discussion_complete=True)
        self.assertEqual(result.coverage.links_validated, 4)
        self.assertEqual(result.coverage.links_supported, 0)
        self.assertEqual(result.coverage.rejection_reasons, {"no_semantic_support": 4})
        self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_solution_before_issue_is_rejected_with_actual_times(self):
        rows = discussion()
        # A comment issue raised after the first solution; it is not the parent
        # of that earlier comment, so only the pair's chronology fails.
        rows[0].stage_outputs["issue"]["predicted_issue_labels"] = []
        rows[-1].stage_outputs["issue"]["predicted_issue_labels"] = ["fees", "roi"]
        rows[-1].stage_outputs["issue"]["issue_probabilities"] = {"fees": 0.9, "roi": 0.9}
        for name in ("response", "solution"):
            rows[-1].stage_outputs[name]["issue_labels"] = ["fees", "roi"]
            rows[-1].stage_outputs[name]["issue_probabilities"] = {"fees": 0.9, "roi": 0.9}
        result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
        self.assertEqual(result.coverage.rejection_reasons, {"solution_before_issue": 4})
        self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_linking_failure_is_isolated_per_thread_and_not_leaked(self):
        def fail_one(batch, issues, _):
            if issues[0].post_id == "bad":
                raise RuntimeError(PRIVATE)
            return batch
        self.linker.behavior = fail_one
        result = self.adapter.process(pipeline_result(discussion("bad"), discussion("good")), discussion_complete=True)
        self.assertEqual((result.coverage.link_attempts, result.coverage.link_failures), (2, 1))
        self.assertEqual((result.coverage.resolution_attempts, result.coverage.resolution_outputs), (1, 2))
        self.assertEqual((result.coverage.threads_partial, result.coverage.threads_complete), (1, 1))
        self.assertEqual(result.threads[0].linking.error_type, "RuntimeError")
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_resolution_failure_and_foreign_or_modified_results_are_isolated(self):
        for failure in ("exception", "foreign", "evidence", "derived", "missing", "extra"):
            with self.subTest(failure=failure):
                def fail_one(batch, evidence):
                    if evidence[0].post_id == "bad":
                        if failure == "exception":
                            raise RuntimeError(PRIVATE)
                        if failure == "foreign":
                            batch.results[0].issue_id = "good:post"
                        elif failure == "evidence":
                            batch.results[0].evidence.total_comments = 99
                        elif failure == "derived":
                            batch.results[0].linked_solution_count = 99
                        elif failure == "missing":
                            batch.results.pop()
                        elif failure == "extra":
                            batch.results.append(batch.results[0].model_copy(deep=True))
                    return batch
                self.resolver.behavior = fail_one
                result = self.adapter.process(pipeline_result(discussion("bad"), discussion("good")), discussion_complete=True)
                self.assertEqual((result.coverage.resolution_attempts, result.coverage.resolution_failures), (2, 1))
                self.assertEqual((result.coverage.resolution_pairs_attempted, result.coverage.resolution_pairs_failed), (4, 2))
                self.assertEqual(result.coverage.resolution_outputs, 2)
                self.assertIsNone(result.threads[0].resolution_result)
                self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_invalid_resolver_config_excludes_without_aborting_other_threads(self):
        # A malformed injected configuration must not escape the thread boundary.
        self.resolver._config.strong_link_threshold = "not-a-score"
        result = self.adapter.process(pipeline_result(discussion("a"), discussion("b")), discussion_complete=True)
        self.assertEqual(result.coverage.link_attempts, 2)
        self.assertEqual(result.coverage.resolution_attempts, 0)
        self.assertEqual(result.coverage.resolution_excluded_reasons, {"invalid_resolver_config": 4})
        self.assertEqual(len(result.threads), 2)

    def test_insufficient_volume_is_excluded_before_real_resolver_and_healthy_thread_continues(self):
        resolver = EvidenceBasedResolver()
        source = pipeline_result(discussion("bad")[:3], discussion("good"))
        before = source.model_dump()
        with patch.object(resolver, "detect", wraps=resolver.detect) as detect:
            result = ThreadEvidenceAdapter(linking_model=self.linker, resolution_model=resolver).process(
                source, discussion_complete=True,
            )
        thin, healthy = result.threads
        self.assertFalse(thin.coverage_complete)
        self.assertTrue(thin.prediction_processing_complete)
        self.assertIsNone(thin.resolution_result)
        self.assertEqual(thin.coverage.resolution_attempts, 0)
        self.assertEqual(thin.coverage.resolution_excluded_reasons, {"insufficient_discussion_volume": 2})
        self.assertTrue(all(pair.excluded_reasons == ["insufficient_discussion_volume"] for pair in thin.issue_labels))
        self.assertTrue(healthy.coverage_complete)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        detect.assert_called_once()
        self.assertEqual({item.post_id for item in detect.call_args.args[0]}, {"good"})
        self.assertTrue(all(item.evidence_sufficient for item in healthy.resolution_result.results))
        self.assertEqual(source.model_dump(), before)

    def test_weak_only_label_is_excluded_while_strong_label_and_healthy_thread_continue(self):
        def weak_roi(batch, issues, _):
            if issues[0].post_id == "bad":
                for link in batch.links:
                    if link.issue_label == "roi":
                        link.link_score = 0.55  # positive at linker 0.5, weak at resolver 0.6
            return batch

        self.linker.behavior = weak_roi
        source = pipeline_result(discussion("bad"), discussion("good"))
        before = source.model_dump()
        result = self.adapter.process(source, discussion_complete=True)
        mixed, healthy = result.threads
        self.assertEqual(mixed.coverage.links_supported, 4)
        self.assertEqual(mixed.coverage.resolution_outputs, 1)
        self.assertEqual(mixed.issue_labels[1].excluded_reasons, ["no_supported_strong_links"])
        self.assertEqual(mixed.issue_labels[1].linked_solution_ids, ["bad:s1", "bad:s2"])
        self.assertEqual(mixed.issue_labels[1].strong_solution_ids, [])
        self.assertEqual([item.issue_label for item in mixed.resolution_result.results], ["fees"])
        self.assertTrue(healthy.coverage_complete)
        self.assertEqual(result.coverage.resolution_outputs, 3)
        self.assertEqual(source.model_dump(), before)

    def test_malformed_discussion_minimum_is_sanitized_without_losing_linked_threads(self):
        for minimum in (PRIVATE, -1, True, 1.5, float("nan"), None):
            with self.subTest(minimum=minimum):
                self.resolver._config.minimum_discussion_volume = minimum
                result = self.adapter.process(pipeline_result(discussion("a"), discussion("b")), discussion_complete=True)
                self.assertEqual(result.coverage.link_attempts, 2)
                self.assertEqual(result.coverage.links_supported, 8)
                self.assertEqual(result.coverage.resolution_attempts, 0)
                self.assertEqual(result.coverage.resolution_excluded_reasons, {"invalid_resolver_config": 4})
                self.assertEqual(self.resolver.inputs, [])
                self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_false_evidence_sufficiency_is_rejected_against_source_volume_per_thread(self):
        def false_sufficiency(batch, evidence):
            if evidence[0].post_id == "bad":
                batch.results[0].evidence_sufficient = False
                batch.results[0].warnings = [PRIVATE]
            return batch

        self.resolver.behavior = false_sufficiency
        result = self.adapter.process(pipeline_result(discussion("bad"), discussion("good")), discussion_complete=True)
        self.assertEqual(result.coverage.resolution_failures, 1)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertIsNone(result.threads[0].resolution_result)
        self.assertEqual(result.threads[0].resolution.detail, "invalid_resolution_result")
        self.assertTrue(result.threads[1].coverage_complete)
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_real_resolver_handoff_is_source_derived_and_deterministic_at_configured_boundaries(self):
        self.linker.behavior = lambda batch, *_: batch
        resolver = EvidenceBasedResolver(ResolutionModelConfig(strong_link_threshold=0.8))
        adapter = ThreadEvidenceAdapter(linking_model=self.linker, resolution_model=resolver)
        source = pipeline_result(discussion())
        before = source.model_dump()
        first = adapter.process(source, discussion_complete=True)
        source.records.reverse()
        second = adapter.process(source, discussion_complete=True)
        source.records.reverse()
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(source.model_dump(), before)
        thread = first.threads[0]
        self.assertTrue(thread.coverage_complete)
        self.assertEqual(thread.coverage.resolution_outputs, 2)
        for item in thread.resolution_result.results:
            self.assertEqual((item.effective_discussion_volume, item.strong_link_count), (3, 2))
            self.assertTrue(item.evidence_sufficient)
            self.assertEqual(item.model_dump(), resolver.detect_one(item.evidence).model_dump())

    def test_foreign_batch_prediction_ids_and_local_batch_mismatch_are_rejected(self):
        source = pipeline_result(discussion("bad"), discussion("good"))
        predictions = [IssuePrediction(**row.stage_outputs["issue"]) for row in source.records]
        batch = IssueIdentificationResult(
            model_name="offline", model_version="test-1", threshold=0.5,
            n_documents=len(predictions), predictions=predictions, known_labels=["fees", "roi"],
        )
        source.batch_outputs["issue"] = batch.model_dump()
        # Local echo disagreement isolates the bad thread.
        source.batch_outputs["issue"]["predictions"][0]["original_text"] = PRIVATE
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.coverage.resolution_outputs, 2)
        self.assertTrue(result.threads[1].coverage_complete)
        self.assertNotIn(PRIVATE, result.model_dump_json())
        source.batch_outputs["issue"]["predictions"][0]["document_id"] = "foreign"
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.coverage.issue_predictions_validated, 0)
        self.assertEqual(result.coverage.resolution_outputs, 0)

    def test_invalid_source_and_upstream_input_losses_gate_completeness(self):
        for field, value in (("created_at", datetime(2026, 1, 1)), ("url", "relative-url"),
                             ("source_mode", "guessed"), ("post_id", None)):
            with self.subTest(field=field):
                source = pipeline_result(discussion("bad"), discussion("good"))
                setattr(source.records[0], field, value)
                result = self.adapter.process(source, discussion_complete=True)
                self.assertEqual(result.invalid_record_indexes, [0])
                self.assertEqual(result.coverage.records_received, 8)
                self.assertEqual(result.coverage.records_rejected, 1)
                self.assertEqual(result.coverage.resolution_attempts, 0)
        source = pipeline_result(discussion())
        source.invalid_records = [InvalidRecord(index=5, fields=["text"], reason="record_validation_failed")]
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.coverage.upstream_invalid_records, 1)
        self.assertEqual(result.coverage.resolution_attempts, 0)
        source.invalid_records = []
        source.duplicate_document_ids = ["p1:s1"]
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.coverage.upstream_duplicate_ids, 1)
        self.assertEqual(result.coverage.resolution_attempts, 0)

    def test_duplicate_documents_and_source_identities_cannot_be_complete(self):
        for duplicate_document in (True, False):
            with self.subTest(duplicate_document=duplicate_document):
                rows = discussion()
                extra = rows[1].model_copy(deep=True)
                if not duplicate_document:
                    extra.document_id = "different-corpus-key"
                rows.append(extra)
                result = self.adapter.process(pipeline_result(rows), discussion_complete=True)
                self.assertIn("duplicate_source_identity", result.threads[0].validation_errors)
                self.assertEqual(result.coverage.link_attempts, 0)
                self.assertEqual(result.coverage.solutions_received, 2 if duplicate_document else 3)

    def test_empty_and_input_order_are_deterministic(self):
        empty = self.adapter.process(PipelineResult(status="invalid_input"))
        self.assertEqual(empty.status, "skipped")
        self.assertEqual(empty.coverage.records_received, 0)
        self.assertEqual(empty.coverage.threads_received, 0)
        self.assertEqual(empty.threads, [])
        self.assertEqual(self.linker.inputs, [])
        self.assertEqual(self.resolver.inputs, [])
        source = pipeline_result(discussion("b"), discussion("a"))
        first = self.adapter.process(source, discussion_complete=True)
        source.records.reverse()
        second = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(first.model_dump(), second.model_dump())

    def test_mutating_model_arguments_cannot_change_source_or_saved_evidence(self):
        source = pipeline_result(discussion())
        before = source.model_dump()

        def mutate_link_input(batch, issues, solutions):
            issues[0].post_id = PRIVATE
            solutions[0].normalized_text = PRIVATE
            return batch

        def mutate_resolution_input(batch, evidence):
            evidence[0].post_id = PRIVATE
            evidence[0].total_comments = 1000
            return batch

        self.linker.behavior = mutate_link_input
        self.resolver.behavior = mutate_resolution_input
        result = self.adapter.process(source, discussion_complete=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(source.model_dump(), before)
        self.assertEqual(result.threads[0].resolution_evidence[0].total_comments, 3)
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_run_integration_and_reprocessing_do_not_add_classifier_inference(self):
        # This one-comment integration intentionally exercises a permitted custom
        # policy; the default minimum of three is covered with real threads above.
        self.resolver._config.minimum_discussion_volume = 1
        calls = []
        issue = RecordingModel("issue", calls)
        response = RecordingModel("response", calls)

        def issue_result(documents):
            result = stage_result("issue", documents)
            for prediction in result.predictions:
                if prediction.comment_id:
                    prediction.predicted_issue_labels = []
            return result
        issue.behavior = issue_result

        def positive_result(documents):
            result = stage_result("solution", documents)
            for prediction in result.predictions:
                prediction.is_solution = True
                prediction.solution_probability = 0.9
                prediction.not_solution_probability = 0.1
            result.n_solutions, result.n_not_solutions = len(documents), 0
            return result
        solution = RecordingModel("solution", calls, behavior=positive_result)
        pipeline = DownstreamPipeline(
            language_service=RecordingLanguage(calls), preprocessor=RecordingPreprocessor(calls),
            issue_model=issue, response_model=response, solution_model=solution,
            linking_model=self.linker, resolution_model=self.resolver,
        )
        post = record(document_id="post", comment_id=None, parent_comment_id=None,
                      parent_reddit_id=None, parent_comment_text=None)
        comment = record(parent_comment_id=None, parent_reddit_id="t3_p-1", parent_comment_text=None)
        result = pipeline.run([post, comment], discussion_complete=True)
        self.assertEqual(result.thread_evidence.coverage.resolution_outputs, 1)
        self.assertEqual(result.thread_evidence.status, "completed")
        before = list(calls)
        repeat = pipeline.process_thread_evidence(result, discussion_complete=True)
        self.assertEqual(calls, before)
        self.assertEqual(repeat.model_dump(), result.thread_evidence.model_dump())
        for model in (issue, response, solution):
            model.predict.assert_called_once()
            model.train.assert_not_called()
            model.fit.assert_not_called()
        stages = {stage.name: stage for stage in result.stages}
        self.assertEqual(stages["linking"].status, "completed")
        self.assertEqual(stages["resolution"].status, "completed")
        self.assertEqual(stages["topic"].detail, "topic_discovery_requires_fitting")


if __name__ == "__main__":
    unittest.main()
