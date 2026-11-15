"""Offline contracts for the orchestrator; no classifiers are trained or fitted.

Only the language, preprocessing, and prediction boundaries are doubles.  Their
successful return values use the real stage schemas, so orchestration exercises
the same strict validation and context adapters as production.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

# Both ``python backend/test_nlp_pipeline.py`` and root-level discovery work.
# Put the root first: backend/ml is an artifact directory, not the ml package.
BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
for directory in (BACKEND, ROOT):
    directory = str(directory)
    if directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)

from app.services.nlp.code_mix_detector import CodeMixResult
from app.services.nlp.issue_identification.base import IssueDocument
from app.services.nlp.issue_identification.models.result import (
    IssueIdentificationResult,
    IssuePrediction,
)
from app.services.nlp.language import LanguageResult
from app.services.nlp.language_analysis_service import FullLanguageAnalysis
from app.services.nlp.pipeline import DownstreamPipeline, PIPELINE_STAGE_ORDER
from app.services.nlp.preprocessing.base import PreprocessingResult
from app.services.nlp.response_classification.base import ResponseDocument
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.response_classification.models.result import (
    ResponseClassificationResult,
    ResponsePrediction,
)
from app.services.nlp.script_detector import ScriptResult
from app.services.nlp.sentiment_emotion.base import SentimentEmotionDocument
from app.services.nlp.sentiment_emotion.models.result import (
    EmotionAnalysisResult,
    EmotionPrediction,
    SentimentAnalysisResult,
    SentimentPrediction,
)
from app.services.nlp.solution_detection.base import SolutionDocument
from app.services.nlp.solution_detection.models.result import (
    SolutionDetectionResult,
    SolutionPrediction,
)
from ml.configs.config import TaskType


SUPPORTED = ("issue", "sentiment", "emotion", "response", "solution")
STAGE_ORDER = (
    "language", "preprocessing", "topic", *SUPPORTED, "linking", "resolution",
)
RESPONSE_LABELS = (
    "agreement", "personal_experience", "opinion", "advice", "solution",
    "counterargument",
)
PRIVATE = "private-payload-must-not-appear"


def record(**overrides):
    value = {
        "document_id": "c-1",
        "text": "placement ka kya scene hai?",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "source_mode": "direct",
        "source_subreddit": "IIMSambalpur",
        "post_id": "p-1",
        "comment_id": "c-1",
        "parent_comment_id": "parent-1",
        "parent_reddit_id": "t1_parent-1",
        "parent_comment_text": "How do I contact the placement office?",
        "url": "https://www.reddit.com/r/IIMSambalpur/comments/p-1/thread/c-1/",
        "original_text": "Placement ka kya scene hai?! <original>",
    }
    value.update(overrides)
    return value


def language_analysis():
    metadata = {"detector_name": "offline-double", "detector_version": "test-1"}
    return FullLanguageAnalysis(
        language="hi", language_confidence=0.8,
        script="latin", script_confidence=1.0,
        is_code_mixed=True, detected_languages=["hi", "en"],
        **metadata,
        language_detail=LanguageResult(
            language="hi", confidence=0.8,
            all_detected=[("hi", 0.8), ("en", 0.2)], **metadata,
        ),
        script_detail=ScriptResult(
            script="latin", confidence=1.0, script_counts={"latin": 25}, **metadata,
        ),
        code_mix_detail=CodeMixResult(
            is_code_mixed=True, confidence=0.9, detected_languages=["hi", "en"],
            evidence={"hi": 0.7, "en": 0.3}, **metadata,
        ),
    )


class RecordingLanguage:
    def __init__(self, calls, behavior=None):
        self.calls = calls
        self.behavior = behavior
        self.texts = []

    def analyze(self, text):
        self.calls.append("language")
        self.texts.append(text)
        return self.behavior(text) if self.behavior else language_analysis()


class RecordingPreprocessor:
    def __init__(self, calls, behavior=None):
        self.calls = calls
        self.behavior = behavior
        self.inputs = []

    def preprocess(self, text, **kwargs):
        self.calls.append("preprocessing")
        self.inputs.append((text, kwargs))
        if self.behavior:
            return self.behavior(text)
        return preprocessing_result(text)


def preprocessing_result(text, normalized_text=None):
    return PreprocessingResult(
        original_text=text,
        normalized_text=f"normalized::{text}" if normalized_text is None else normalized_text,
        preprocessing_version="offline-test-1",
        stages_applied=["offline-normalization"],
    )


def stage_result(name, documents):
    """Return actual Result/Prediction models with distinct per-document context."""
    common = dict(model_name=f"offline-{name}", model_version="test-1", threshold=0.5,
                  n_documents=len(documents), config={"offline": True},
                  warnings=["synthetic fixture"])
    predictions = []
    for document in documents:
        echo = document.model_dump()
        if name == "issue":
            prediction = IssuePrediction(
                **echo, predicted_issue_labels=[f"concern-{document.document_id}"],
                issue_probabilities={f"concern-{document.document_id}": 0.85}, confidence=0.85,
            )
        elif name == "sentiment":
            prediction = SentimentPrediction(
                **echo, sentiment="negative", sentiment_probabilities={"negative": 0.8, "neutral": 0.2},
                confidence=0.8,
            )
        elif name == "emotion":
            prediction = EmotionPrediction(
                **echo, emotions=["concern"], emotion_probabilities={"concern": 0.75}, confidence=0.75,
            )
        elif name == "response":
            advice = 0.3 if document.document_id.endswith("2") else 0.65
            prediction = ResponsePrediction(
                **echo, response_type="advice" if advice > 0.5 else "solution",
                # Deliberately noncanonical key order: the adapter must reorder
                # without changing values or filling absent classes with zeros.
                response_probabilities={"solution": 1.0 - advice, "advice": advice},
                confidence=max(advice, 1.0 - advice),
            )
        elif name == "solution":
            prediction = SolutionPrediction(
                **echo, is_solution=False, solution_probability=0.2,
                not_solution_probability=0.8,
                is_candidate=document.response_type in {"advice", "solution"},
            )
        else:
            raise AssertionError(f"unsupported fixture stage: {name}")
        predictions.append(prediction)
    common["predictions"] = list(reversed(predictions))  # Join by ID, not position.
    if name == "issue":
        return IssueIdentificationResult(
            **common, known_labels=[f"concern-{doc.document_id}" for doc in documents],
        )
    if name == "sentiment":
        return SentimentAnalysisResult(**common, known_sentiments=["negative", "neutral"])
    if name == "emotion":
        return EmotionAnalysisResult(**common, known_emotions=["concern"])
    if name == "response":
        predicted = {prediction.response_type for prediction in predictions}
        return ResponseClassificationResult(
            **common, known_response_types=["advice", "solution"],
            missing_response_types=[label for label in RESPONSE_LABELS if label not in {"advice", "solution"}],
            unpredicted_response_types=[label for label in ("advice", "solution") if label not in predicted],
        )
    return SolutionDetectionResult(**common, n_not_solutions=len(documents))


class RecordingModel:
    def __init__(self, name, calls, *, trained=True, behavior=None):
        self.name = name
        self.calls = calls
        self.is_trained = trained
        self.behavior = behavior
        self.batches = []
        self.predict = Mock(side_effect=self._predict)
        self.train = Mock(side_effect=AssertionError("training is forbidden"))
        self.fit = Mock(side_effect=AssertionError("fitting is forbidden"))

    def _predict(self, documents):
        self.calls.append(self.name)
        self.batches.append(documents)
        return self.behavior(documents) if self.behavior else stage_result(self.name, documents)


class ReadinessFailure:
    def __init__(self):
        self.checks = 0
        self.predict = Mock(side_effect=AssertionError("prediction is forbidden"))
        self.train = Mock(side_effect=AssertionError("training is forbidden"))
        self.fit = Mock(side_effect=AssertionError("fitting is forbidden"))

    @property
    def is_trained(self):
        self.checks += 1
        raise RuntimeError(PRIVATE)


def raise_private(*args):
    raise RuntimeError(PRIVATE)


def reports(result):
    return {stage.name: stage for stage in result.stages}


class TestDownstreamPipeline(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.language = RecordingLanguage(self.calls)
        self.preprocessor = RecordingPreprocessor(self.calls)
        self.models = {name: RecordingModel(name, self.calls) for name in SUPPORTED}

    def pipeline(self, *, configured=True, **overrides):
        kwargs = dict(language_service=self.language, preprocessor=self.preprocessor)
        if configured:
            kwargs.update({f"{name}_model": model for name, model in self.models.items()})
        kwargs.update(overrides)
        return DownstreamPipeline(**kwargs)

    def assert_stage_failed(self, result, name, *, error="ValueError", detail="invalid_stage_result"):
        stage = reports(result)[name]
        self.assertEqual(result.status, "partial_failure")
        self.assertEqual((stage.status, stage.output_records, stage.error_type, stage.detail),
                         ("failed", 0, "stage_error", detail))
        self.assertNotIn(name, result.batch_outputs)
        for accepted in result.records:
            self.assertNotIn(name, accepted.stage_outputs)
            self.assertEqual(accepted.stage_errors[name], error)
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_valid_input_preserves_provenance_parent_identity_and_original(self):
        payloads = [
            record(document_id="corpus:c-1"),
            record(document_id="context:c-2", comment_id="c-2", source_mode="context",
                   source_subreddit="MBA", created_at="2026-01-02T10:30:00+05:30",
                   parent_comment_id=None, parent_reddit_id="t3_p-1"),
            # Context remains context even for an approved direct community.
            record(document_id="context:c-3", comment_id="c-3", source_mode="context",
                   source_subreddit="IIMSambalpur"),
        ]
        before = deepcopy(payloads)
        result = self.pipeline().run(payloads)

        self.assertEqual(result.status, "partial_failure")  # Unsupported integrations are explicit.
        self.assertEqual(result.invalid_records, [])
        self.assertEqual(result.duplicate_document_ids, [])
        self.assertEqual(payloads, before)
        for payload, accepted in zip(payloads, result.records):
            with self.subTest(document_id=payload["document_id"]):
                for field in ("document_id", "post_id", "comment_id", "parent_comment_id",
                              "parent_reddit_id", "parent_comment_text", "url", "source_mode",
                              "source_subreddit", "original_text", "text"):
                    self.assertEqual(getattr(accepted, field), payload[field])
                self.assertEqual(accepted.normalized_text, f"normalized::{payload['text']}")
                self.assertEqual(accepted.stage_errors, {})
                self.assertEqual(accepted.language_analysis, language_analysis().model_dump())
                preprocessed = accepted.stage_outputs["preprocessing"]
                self.assertEqual(preprocessed["original_text"], payload["text"])
                self.assertEqual(preprocessed["stages_applied"], ["offline-normalization"])
        self.assertEqual(result.records[0].created_at, payloads[0]["created_at"])
        self.assertEqual(result.records[1].created_at.isoformat(), "2026-01-02T10:30:00+05:30")
        self.assertEqual(result.records[1].created_at.utcoffset(), timedelta(hours=5, minutes=30))

    def test_post_identity_and_original_text_fallback(self):
        payload = record(document_id="export:post:p-1", comment_id=None, parent_comment_id=None,
                         parent_reddit_id=None, parent_comment_text=None, original_text=None,
                         source_subreddit="iImSaMbAlPuRuG", created_at="2026-01-01T00:00:00Z")
        result = self.pipeline().run([payload])

        self.assertEqual(result.invalid_records, [])
        accepted = result.records[0]
        self.assertIsNone(accepted.comment_id)
        self.assertEqual(accepted.document_id, "export:post:p-1")
        self.assertEqual(accepted.original_text, payload["text"])
        self.assertEqual(accepted.created_at, datetime(2026, 1, 1, tzinfo=timezone.utc))
        for name in ("issue", "sentiment", "emotion"):
            model = self.models[name]
            self.assertEqual(model.batches[0][0].original_text, payload["text"])
        for name in ("response", "solution"):
            self.models[name].predict.assert_not_called()
            self.assertEqual(reports(result)[name].detail, "no_comment_records")
            self.assertNotIn(name, accepted.stage_outputs)

    def test_missing_required_fields_are_aggregated(self):
        result = self.pipeline().run([{}])

        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.records, [])
        self.assertEqual(result.invalid_records[0].fields, sorted([
            "document_id", "text", "created_at", "source_mode", "source_subreddit", "post_id",
        ]))
        self.assertEqual(self.calls, [])

    def test_malformed_fields_and_identity_errors_are_aggregated_and_sanitized(self):
        payload = record(
            document_id=PRIVATE, text=123, created_at=datetime(2026, 1, 1),
            source_subreddit="MBA", post_id=None, comment_id=None,
            parent_comment_id=PRIVATE, parent_reddit_id=f"t1_{PRIVATE}",
            parent_comment_text=PRIVATE, url=f"/{PRIVATE}", original_text=PRIVATE,
        )
        result = self.pipeline().run([payload])

        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.records, [])
        self.assertEqual(result.invalid_records[0].model_dump(), {
            "index": 0, "document_id": None,
            "reason": "record_validation_failed",
            "fields": sorted([
                "text", "created_at", "source_mode", "source_subreddit", "post_id", "comment_id",
                "parent_comment_id", "parent_reddit_id", "parent_comment_text", "url",
            ]),
        })
        self.assertNotIn(PRIVATE, result.model_dump_json())
        self.assertEqual(self.calls, [])
        self.assertTrue(all(stage.status == "skipped" for stage in result.stages))

    def test_required_strings_and_source_mode_are_not_coerced_or_blank(self):
        cases = [
            ("document_id", ""), ("document_id", 7), ("text", " \n\t"), ("text", 12),
            ("source_subreddit", ""), ("source_subreddit", 12),
            ("source_mode", "other"), ("source_mode", None),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                changes = {"source_mode": "context", field: value}
                payload = record(**changes)
                result = self.pipeline().run([payload])
                self.assertEqual(result.status, "invalid_input")
                self.assertEqual(result.invalid_records[0].fields, [field])
        self.assertEqual(self.calls, [])

    def test_naive_and_epoch_timestamps_are_rejected(self):
        for timestamp in (datetime(2026, 1, 1), "2026-01-01T12:00:00", "2026-01-01",
                          1767225600, 1767225600.0, "1767225600", "1767225600000",
                          None, True, "not-a-timestamp"):
            with self.subTest(timestamp=timestamp):
                result = self.pipeline().run([record(created_at=timestamp)])
                self.assertEqual(result.status, "invalid_input")
                self.assertEqual(result.invalid_records[0].fields, ["created_at"])
        self.assertEqual(self.calls, [])

    def test_invalid_urls_are_rejected_without_echoing_them(self):
        for url in (f"/{PRIVATE}", f"ftp://example.com/{PRIVATE}", "https://",
                    f"https://example.com/{PRIVATE} space", "", 99):
            with self.subTest(url=url):
                result = self.pipeline().run([record(url=url)])
                self.assertEqual(result.invalid_records[0].fields, ["url"])
                self.assertNotIn(PRIVATE, result.model_dump_json())
        self.assertEqual(self.calls, [])

    def test_identity_validation_rejects_orphan_self_parent_and_ambiguous_ids(self):
        cases = [
            ({"post_id": None}, {"post_id"}),
            ({"parent_comment_id": "c-1"}, {"parent_comment_id"}),
            ({"parent_reddit_id": "c-1"}, {"parent_reddit_id"}),
            ({"parent_reddit_id": "t1_c-1"}, {"parent_reddit_id"}),
            ({"comment_id": None}, {"comment_id", "parent_comment_id", "parent_reddit_id", "parent_comment_text"}),
            ({"document_id": " corpus:c-1"}, {"document_id"}),
            ({"post_id": "p 1"}, {"post_id"}),
            ({"comment_id": 1}, {"comment_id"}),
            ({"parent_comment_id": " "}, {"parent_comment_id"}),
            ({"parent_reddit_id": "t1_ parent"}, {"parent_reddit_id"}),
            ({"source_mode": "direct", "source_subreddit": "MBA"}, {"source_mode", "source_subreddit"}),
        ]
        for changes, fields in cases:
            with self.subTest(changes=changes):
                result = self.pipeline().run([record(**changes)])
                self.assertEqual(result.status, "invalid_input")
                self.assertEqual(result.invalid_records[0].fields, sorted(fields))
        self.assertEqual(self.calls, [])

    def test_non_mapping_records_are_sanitized(self):
        result = self.pipeline().run([None, PRIVATE, 123, [PRIVATE]])

        self.assertEqual(result.status, "invalid_input")
        self.assertEqual([invalid.index for invalid in result.invalid_records], [0, 1, 2, 3])
        for invalid in result.invalid_records:
            self.assertEqual(invalid.fields, ["record"])
            self.assertEqual(invalid.reason, "record_validation_failed")
            self.assertIsNone(invalid.document_id)
        self.assertNotIn(PRIVATE, result.model_dump_json())
        self.assertEqual(self.calls, [])

    def test_mixed_valid_and_invalid_records_only_infer_valid_records(self):
        result = self.pipeline().run([record(document_id=PRIVATE, text=""), record()])

        self.assertEqual(result.status, "partial_failure")
        self.assertEqual([accepted.document_id for accepted in result.records], ["c-1"])
        self.assertEqual(result.invalid_records[0].index, 0)
        self.assertNotIn(PRIVATE, result.model_dump_json())
        for model in self.models.values():
            self.assertEqual([doc.document_id for doc in model.batches[0]], ["c-1"])

    def test_duplicates_keep_first_valid_record_and_report_id_once(self):
        payloads = [record(text="first"), record(text="second"), record(text="third"),
                    record(document_id="c-2", comment_id="c-2", text="independent")]
        result = self.pipeline().run(payloads)

        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(result.duplicate_document_ids, ["c-1"])
        self.assertEqual([accepted.text for accepted in result.records], ["first", "independent"])
        self.assertEqual(result.invalid_records, [])
        self.assertEqual(result.warnings, ["duplicate_document_ids: first occurrence retained"])
        self.assertEqual(self.language.texts, ["first", "independent"])
        for model in self.models.values():
            model.predict.assert_called_once()
            self.assertEqual([doc.document_id for doc in model.batches[0]], ["c-1", "c-2"])

    def test_invalid_first_occurrence_does_not_reserve_document_id(self):
        result = self.pipeline().run([record(text=""), record(text="valid replacement")])

        self.assertEqual(result.duplicate_document_ids, [])
        self.assertEqual(result.records[0].text, "valid replacement")
        self.assertEqual(result.invalid_records[0].index, 0)
        self.assertEqual(self.language.texts, ["valid replacement"])

    def test_empty_input_skips_every_stage_without_invocation(self):
        result = self.pipeline().run([])

        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.records, [])
        self.assertEqual(result.invalid_records, [])
        self.assertEqual(result.duplicate_document_ids, [])
        self.assertEqual(result.batch_outputs, {})
        self.assertEqual(result.warnings, ["no_valid_records: no downstream stages produced predictions"])
        self.assertEqual([stage.name for stage in result.stages], list(STAGE_ORDER))
        for stage in result.stages:
            self.assertEqual((stage.status, stage.detail, stage.input_records, stage.output_records),
                             ("skipped", "no_valid_records", 0, 0))
        self.assertEqual(self.calls, [])

    def test_supported_order_typed_inputs_and_context_transfers(self):
        payloads = [record(), record(document_id="c-2", comment_id="c-2", text="second comment",
                                     original_text="SECOND original", parent_comment_text="Second parent")]
        result = self.pipeline().run(payloads)

        self.assertEqual(tuple(PIPELINE_STAGE_ORDER), STAGE_ORDER)
        self.assertEqual([stage.name for stage in result.stages], list(STAGE_ORDER))
        self.assertEqual(self.calls, ["language", "language", "preprocessing", "preprocessing", *SUPPORTED])
        self.assertEqual(self.language.texts, [payload["text"] for payload in payloads])
        self.assertEqual(self.preprocessor.inputs, [
            (payload["text"], {"language": "hi", "script": "latin", "is_code_mixed": True,
                               "detected_languages": ["hi", "en"]}) for payload in payloads
        ])
        document_types = {"issue": IssueDocument, "sentiment": SentimentEmotionDocument,
                          "emotion": SentimentEmotionDocument, "response": ResponseDocument,
                          "solution": SolutionDocument}
        for name, model in self.models.items():
            with self.subTest(stage=name):
                model.predict.assert_called_once()
                model.train.assert_not_called()
                model.fit.assert_not_called()
                stage = reports(result)[name]
                self.assertEqual((stage.status, stage.input_records, stage.output_records), ("completed", 2, 2))
                for payload, document, accepted in zip(payloads, model.batches[0], result.records):
                    self.assertIsInstance(document, document_types[name])
                    self.assertEqual(document.document_id, payload["document_id"])
                    self.assertEqual(document.text, accepted.normalized_text)
                    self.assertEqual(document.original_text, payload["original_text"])
                    self.assertEqual((document.post_id, document.comment_id), (payload["post_id"], payload["comment_id"]))
                    self.assertEqual((document.language, document.script, document.is_code_mixed,
                                      document.detected_languages), ("hi", "latin", True, ["hi", "en"]))
                    self.assertIsNone(document.topic_id)
                    if name != "issue":
                        issue = accepted.stage_outputs["issue"]
                        self.assertEqual(document.issue_labels, [f"concern-{document.document_id}"])
                        self.assertEqual(document.issue_labels, issue["predicted_issue_labels"])
                        self.assertEqual(document.issue_probabilities, issue["issue_probabilities"])
                    if name in {"response", "solution"}:
                        self.assertEqual(document.parent_comment_text, payload["parent_comment_text"])
                    if name == "solution":
                        response = accepted.stage_outputs["response"]
                        self.assertEqual(document.response_type, response["response_type"])
                        self.assertEqual(document.response_probabilities, response["response_probabilities"])
                        self.assertEqual(list(document.response_probabilities), ["advice", "solution"])
        # The intentionally unsupported stages make overall completion partial.
        self.assertEqual(result.status, "partial_failure")

    def test_batch_results_are_stored_once_and_records_hold_only_their_prediction(self):
        result = self.pipeline().run([record(), record(document_id="c-2", comment_id="c-2")])

        self.assertEqual(set(result.batch_outputs), set(SUPPORTED))
        for name in SUPPORTED:
            batch = result.batch_outputs[name]
            self.assertEqual(batch["n_documents"], 2)
            self.assertEqual(batch["model_name"], f"offline-{name}")
            self.assertEqual(len(batch["predictions"]), 2)
            by_id = {prediction["document_id"]: prediction for prediction in batch["predictions"]}
            self.assertEqual(set(by_id), {"c-1", "c-2"})
            for accepted in result.records:
                prediction = accepted.stage_outputs[name]
                self.assertEqual(prediction, by_id[accepted.document_id])
                for batch_field in ("predictions", "n_documents", "model_name", "model_version", "config", "warnings"):
                    self.assertNotIn(batch_field, prediction)
        for accepted in result.records:
            self.assertEqual(set(accepted.stage_outputs), {"language", "preprocessing", *SUPPORTED})
        # This also verifies nested typed language tuples are JSON serializable.
        self.assertEqual(result.model_dump_json().count('"model_name"'), len(SUPPORTED))

    def test_unconfigured_models_are_explicitly_unavailable(self):
        result = self.pipeline(configured=False).run([record()])

        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(self.calls, ["language", "preprocessing"])
        self.assertEqual(result.batch_outputs, {})
        self.assertEqual(set(result.records[0].stage_outputs), {"language", "preprocessing"})
        for name in SUPPORTED:
            stage = reports(result)[name]
            self.assertEqual((stage.status, stage.detail, stage.input_records, stage.output_records),
                             ("unavailable", "not_configured", 1, 0))

    def test_topic_and_noninterface_evidence_models_are_never_invoked(self):
        unsupported = {name: ReadinessFailure() for name in ("topic", "linking", "resolution")}
        result = self.pipeline(**{f"{name}_model": model for name, model in unsupported.items()}).run([record()])

        for name, model in unsupported.items():
            with self.subTest(stage=name):
                self.assertEqual(model.checks, 0)
                model.predict.assert_not_called()
                model.train.assert_not_called()
                model.fit.assert_not_called()
                stage = reports(result)[name]
                expected_detail = {"topic": "topic_discovery_requires_fitting",
                                   "linking": "invalid_linker_interface",
                                   "resolution": "invalid_resolver_interface"}[name]
                self.assertEqual((stage.status, stage.detail, stage.input_records, stage.output_records),
                                 ("unavailable", expected_detail, 1 if name == "topic" else 0, 0))
                self.assertNotIn(name, result.records[0].stage_outputs)
                self.assertNotIn(name, result.batch_outputs)
        self.assertEqual(self.calls, ["language", "preprocessing", *SUPPORTED])

    def test_only_literal_trained_true_is_ready(self):
        for name in SUPPORTED:
            for trained in (False, None, 1, "True"):
                with self.subTest(stage=name, trained=trained):
                    model = RecordingModel(name, self.calls, trained=trained)
                    result = self.pipeline(**{f"{name}_model": model}).run([record()])
                    stage = reports(result)[name]
                    self.assertEqual((stage.status, stage.detail, stage.output_records),
                                     ("unavailable", "model_not_ready", 0))
                    model.predict.assert_not_called()
                    model.train.assert_not_called()
                    model.fit.assert_not_called()
                    self.assertNotIn(name, result.batch_outputs)
                    self.assertNotIn(name, result.records[0].stage_outputs)
                    for independent in set(SUPPORTED) - {name}:
                        self.assertEqual(reports(result)[independent].status, "completed")

    def test_readiness_property_exceptions_are_sanitized_and_isolated(self):
        for name in SUPPORTED:
            with self.subTest(stage=name):
                model = ReadinessFailure()
                result = self.pipeline(**{f"{name}_model": model}).run([record()])
                self.assertEqual(model.checks, 1)
                model.predict.assert_not_called()
                model.train.assert_not_called()
                model.fit.assert_not_called()
                self.assert_stage_failed(result, name, error="RuntimeError", detail="readiness_check_failed")
                for independent in set(SUPPORTED) - {name}:
                    self.assertEqual(reports(result)[independent].status, "completed")

    def test_preprocessing_failure_excludes_only_failed_record_from_every_model(self):
        def preprocess(text):
            if text == "bad text":
                raise RuntimeError(PRIVATE)
            return preprocessing_result(text)

        self.preprocessor.behavior = preprocess
        result = self.pipeline().run([record(text="bad text"), record(document_id="c-2", comment_id="c-2")])

        failed, healthy = result.records
        self.assertEqual(result.status, "partial_failure")
        self.assertIsNone(failed.normalized_text)
        self.assertEqual(failed.stage_errors["preprocessing"], "RuntimeError")
        self.assertEqual(healthy.stage_errors, {})
        stage = reports(result)["preprocessing"]
        self.assertEqual((stage.status, stage.input_records, stage.output_records), ("partial_failure", 2, 1))
        for name, model in self.models.items():
            self.assertEqual([doc.document_id for doc in model.batches[0]], ["c-2"])
            self.assertEqual(failed.stage_errors[name], "preprocessing_not_ready")
            self.assertNotIn(name, failed.stage_outputs)
            self.assertIn(name, healthy.stage_outputs)
            stage = reports(result)[name]
            self.assertEqual((stage.status, stage.input_records, stage.output_records, stage.detail),
                             ("partial_failure", 2, 1, "some_records_not_preprocessed"))
            self.assertEqual(result.batch_outputs[name]["n_documents"], 1)
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_all_preprocessing_failures_skip_inference(self):
        self.preprocessor.behavior = raise_private
        result = self.pipeline().run([record()])

        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(reports(result)["preprocessing"].status, "failed")
        self.assertEqual(self.calls, ["language", "preprocessing"])
        self.assertEqual(result.batch_outputs, {})
        for name, model in self.models.items():
            model.predict.assert_not_called()
            stage = reports(result)[name]
            self.assertEqual((stage.status, stage.detail, stage.output_records),
                             ("skipped", "no_preprocessed_records", 0))
            self.assertEqual(result.records[0].stage_errors[name], "preprocessing_not_ready")
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_empty_normalized_text_is_a_preprocessing_failure(self):
        self.preprocessor.behavior = lambda text: preprocessing_result(text, " \t\n")
        result = self.pipeline().run([record()])

        self.assertIsNone(result.records[0].normalized_text)
        self.assertEqual(result.records[0].stage_errors["preprocessing"], "empty_normalized_text")
        self.assertEqual(reports(result)["preprocessing"].output_records, 0)
        self.assertEqual(result.batch_outputs, {})
        for model in self.models.values():
            model.predict.assert_not_called()

    def test_language_failure_blocks_that_records_preprocessing_and_inference(self):
        self.language.behavior = lambda text: raise_private() if text == "bad language" else language_analysis()
        result = self.pipeline().run([record(text="bad language"), record(document_id="c-2", comment_id="c-2")])

        failed, healthy = result.records
        self.assertIsNone(failed.language_analysis)
        self.assertEqual(failed.stage_errors["language"], "RuntimeError")
        self.assertEqual(failed.stage_errors["preprocessing"], "language_not_ready")
        self.assertEqual(healthy.stage_errors, {})
        self.assertEqual([text for text, kwargs in self.preprocessor.inputs], [healthy.text])
        self.assertEqual(reports(result)["language"].status, "partial_failure")
        for name, model in self.models.items():
            self.assertEqual([doc.document_id for doc in model.batches[0]], ["c-2"])
            self.assertEqual(failed.stage_errors[name], "preprocessing_not_ready")
        self.assertNotIn(PRIVATE, result.model_dump_json())

    def test_malformed_language_and_preprocessing_results_do_not_enable_inference(self):
        for boundary in ("language", "preprocessing"):
            with self.subTest(boundary=boundary):
                language = RecordingLanguage(self.calls)
                preprocessor = RecordingPreprocessor(self.calls)
                if boundary == "language":
                    language.behavior = lambda text: {"language": PRIVATE}
                else:
                    preprocessor.behavior = lambda text: {
                        "original_text": PRIVATE, "normalized_text": 99,
                        "preprocessing_version": "test", "stages_applied": [],
                    }
                result = self.pipeline(language_service=language, preprocessor=preprocessor).run([record()])
                self.assertEqual(result.records[0].stage_errors[boundary], "ValidationError")
                self.assertIsNone(result.records[0].normalized_text)
                self.assertEqual(result.batch_outputs, {})
                self.assertNotIn(PRIVATE, result.model_dump_json())
                for model in self.models.values():
                    model.predict.assert_not_called()

    def test_stage_prediction_failure_continues_all_independent_models(self):
        for name in SUPPORTED:
            with self.subTest(stage=name):
                failed_model = RecordingModel(name, self.calls, behavior=raise_private)
                result = self.pipeline(**{f"{name}_model": failed_model}).run([record()])
                self.assert_stage_failed(result, name, error="RuntimeError", detail="prediction_failed")
                failed_model.predict.assert_called_once()
                for independent in set(SUPPORTED) - {name}:
                    self.assertEqual(reports(result)[independent].status, "completed")
                    self.assertIn(independent, result.batch_outputs)
                if name == "issue":
                    for downstream in ("sentiment", "emotion", "response", "solution"):
                        document = self.models[downstream].batches[-1][0]
                        self.assertEqual(document.issue_labels, [])
                        self.assertEqual(document.issue_probabilities, {})
                if name == "response":
                    document = self.models["solution"].batches[-1][0]
                    self.assertIsNone(document.response_type)
                    self.assertEqual(document.response_probabilities, {})

    def test_missing_duplicate_foreign_or_extra_result_ids_never_publish_success(self):
        id_cases = {
            "missing": ["c-1"],
            "duplicate": ["c-1", "c-1"],
            "foreign": ["c-1", "not-in-the-batch"],
            "extra": ["c-1", "c-2", "not-in-the-batch"],
        }
        for name in SUPPORTED:
            for case, ids in id_cases.items():
                with self.subTest(stage=name, case=case):
                    def malformed_ids(documents, stage=name, predicted_ids=ids):
                        result = stage_result(stage, documents)
                        template = result.predictions[0]
                        result.predictions = [template.model_copy(update={"document_id": doc_id})
                                              for doc_id in predicted_ids]
                        return result

                    model = RecordingModel(name, self.calls, behavior=malformed_ids)
                    result = self.pipeline(**{f"{name}_model": model}).run([
                        record(), record(document_id="c-2", comment_id="c-2"),
                    ])
                    self.assert_stage_failed(result, name)
                    model.predict.assert_called_once()
                    for independent in set(SUPPORTED) - {name}:
                        self.assertEqual(reports(result)[independent].status, "completed")

    def test_incorrect_result_document_count_is_rejected(self):
        for name in SUPPORTED:
            with self.subTest(stage=name):
                def wrong_count(documents, stage=name):
                    result = stage_result(stage, documents)
                    result.n_documents = 99
                    return result

                model = RecordingModel(name, self.calls, behavior=wrong_count)
                result = self.pipeline(**{f"{name}_model": model}).run([record()])
                self.assert_stage_failed(result, name)

    def test_malformed_stage_result_schema_is_not_treated_as_predictions(self):
        for name in SUPPORTED:
            with self.subTest(stage=name):
                model = RecordingModel(name, self.calls, behavior=lambda documents: {
                    "predictions": [{"document_id": "c-1"}], "private_input": PRIVATE,
                })
                result = self.pipeline(**{f"{name}_model": model}).run([record()])
                self.assert_stage_failed(result, name, error="ValidationError")

    def test_response_full_taxonomy_probability_order_and_values_are_preserved(self):
        probabilities = dict(zip(RESPONSE_LABELS, (0.05, 0.1, 0.15, 0.4, 0.2, 0.1)))

        def full_response(documents):
            result = stage_result("response", documents)
            result.threshold = 0.3
            result.known_response_types = list(RESPONSE_LABELS)
            result.missing_response_types = []
            result.unpredicted_response_types = [label for label in RESPONSE_LABELS if label != "advice"]
            for prediction in result.predictions:
                prediction.response_probabilities = dict(reversed(list(probabilities.items())))
                prediction.response_type = "advice"
                prediction.confidence = 0.4
            return result

        self.models["response"].behavior = full_response
        result = self.pipeline().run([record()])

        self.assertEqual(tuple(RESPONSE_TYPES), RESPONSE_LABELS)
        self.assertEqual(reports(result)["response"].status, "completed")
        batch = result.batch_outputs["response"]
        self.assertEqual(batch["known_response_types"], list(RESPONSE_LABELS))
        self.assertEqual(batch["missing_response_types"], [])
        self.assertEqual(batch["unpredicted_response_types"], [label for label in RESPONSE_LABELS if label != "advice"])
        for output in (batch["predictions"][0], result.records[0].stage_outputs["response"]):
            self.assertEqual(list(output["response_probabilities"]), list(RESPONSE_LABELS))
            self.assertEqual(output["response_probabilities"], probabilities)
        self.assertEqual(list(self.models["solution"].batches[0][0].response_probabilities), list(RESPONSE_LABELS))

    def test_response_partial_taxonomy_and_abstention_do_not_fabricate_missing_classes(self):
        def abstained(documents):
            result = stage_result("response", documents)
            result.threshold = 0.9
            result.predictions[0].response_type = None
            result.predictions[0].abstained = True
            result.unpredicted_response_types = ["advice", "solution"]
            result.n_abstained = 1
            return result

        self.models["response"].behavior = abstained
        result = self.pipeline().run([record()])

        self.assertEqual(reports(result)["response"].status, "completed")
        batch = result.batch_outputs["response"]
        self.assertEqual(batch["known_response_types"], ["advice", "solution"])
        self.assertEqual(batch["missing_response_types"], [label for label in RESPONSE_LABELS if label not in {"advice", "solution"}])
        prediction = result.records[0].stage_outputs["response"]
        self.assertIsNone(prediction["response_type"])
        self.assertTrue(prediction["abstained"])
        self.assertEqual(list(prediction["response_probabilities"]), ["advice", "solution"])
        self.assertEqual(prediction["response_probabilities"], {"advice": 0.65, "solution": 0.35})
        self.assertIsNone(self.models["solution"].batches[0][0].response_type)

    def test_response_noncanonical_taxonomy_metadata_is_rejected(self):
        cases = [
            {"known_response_types": ["solution", "advice"]},
            {"known_response_types": ["advice", "advice", "solution"]},
            {"known_response_types": []},
            {"known_response_types": ["advice", "unknown-role"]},
            {"missing_response_types": []},
            {"missing_response_types": ["counterargument", "opinion", "personal_experience", "agreement"]},
            {"unpredicted_response_types": ["solution", "advice"]},
            {"unpredicted_response_types": ["advice", "advice"]},
            {"unpredicted_response_types": ["agreement"]},
        ]
        for updates in cases:
            with self.subTest(updates=updates):
                def invalid_metadata(documents, changes=updates):
                    result = stage_result("response", documents)
                    for field, value in changes.items():
                        setattr(result, field, value)
                    return result

                model = RecordingModel("response", self.calls, behavior=invalid_metadata)
                result = self.pipeline(response_model=model).run([record()])
                self.assert_stage_failed(result, "response")
                self.assertEqual(reports(result)["solution"].status, "completed")
                self.assertIsNone(self.models["solution"].batches[-1][0].response_type)

    def test_response_invalid_probability_vectors_or_unknown_prediction_are_rejected(self):
        cases = [
            ({"advice": 1.0}, "advice"),
            ({"advice": 0.6, "solution": 0.3, "opinion": 0.1}, "advice"),
            ({"advice": -0.2, "solution": 1.2}, "advice"),
            ({"advice": 0.3, "solution": 0.3}, "advice"),
            ({"advice": float("nan"), "solution": 0.5}, "advice"),
            ({"advice": float("inf"), "solution": 0.5}, "advice"),
            ({"advice": 0.65, "solution": 0.35}, "unknown-role"),
        ]
        for probabilities, response_type in cases:
            with self.subTest(probabilities=probabilities, response_type=response_type):
                def invalid_prediction(documents, values=probabilities, label=response_type):
                    result = stage_result("response", documents)
                    result.predictions[0].response_probabilities = values
                    result.predictions[0].response_type = label
                    return result

                model = RecordingModel("response", self.calls, behavior=invalid_prediction)
                result = self.pipeline(response_model=model).run([record()])
                self.assert_stage_failed(result, "response")

    def test_training_task_type_remains_the_five_existing_tasks(self):
        self.assertEqual([(task.name, task.value) for task in TaskType], [
            ("SENTIMENT", "sentiment"), ("EMOTION", "emotion"), ("ISSUE", "issue"),
            ("RESPONSE", "response"), ("SOLUTION", "solution"),
        ])
        self.assertEqual(set(TaskType.__members__), {"SENTIMENT", "EMOTION", "ISSUE", "RESPONSE", "SOLUTION"})

    def test_mixed_post_and_comments_only_classify_comment_roles_and_solutions(self):
        post = record(document_id="post:p-1", comment_id=None, parent_comment_id=None,
                      parent_reddit_id=None, parent_comment_text=None)
        result = self.pipeline().run([post, record()])
        for name in ("response", "solution"):
            self.assertEqual([doc.document_id for doc in self.models[name].batches[0]], ["c-1"])
            stage = reports(result)[name]
            self.assertEqual((stage.status, stage.input_records, stage.output_records), ("completed", 1, 1))
            self.assertNotIn(name, result.records[0].stage_outputs)
        self.assertEqual(reports(result)["issue"].output_records, 2)

    def test_mismatched_prediction_metadata_fails_the_handoff(self):
        for name in SUPPORTED:
            for field, value in (("post_id", "different-thread"), ("comment_id", "different-comment"),
                                 ("text", "rewritten text"), ("original_text", PRIVATE)):
                with self.subTest(stage=name, field=field):
                    def mismatched(documents, stage=name, key=field, replacement=value):
                        result = stage_result(stage, documents)
                        setattr(result.predictions[0], key, replacement)
                        return result

                    model = RecordingModel(name, self.calls, behavior=mismatched)
                    result = self.pipeline(**{f"{name}_model": model}).run([record()])
                    self.assert_stage_failed(result, name)
                    self.assertEqual(result.records[0].post_id, "p-1")
                    self.assertEqual(result.records[0].comment_id, "c-1")

    def test_model_requiring_unavailable_topic_features_does_not_infer(self):
        from types import SimpleNamespace

        for name in ("issue", "sentiment", "emotion"):
            with self.subTest(stage=name):
                model = RecordingModel(name, self.calls)
                model.config = SimpleNamespace(use_topic_features=True)
                result = self.pipeline(**{f"{name}_model": model}).run([record()])
                model.predict.assert_not_called()
                self.assertEqual(reports(result)[name].status, "unavailable")
                self.assertEqual(reports(result)[name].detail, "topic_context_not_available")
                self.assertNotIn(name, result.batch_outputs)

    def test_mutated_inference_arguments_cannot_authorize_rewritten_sentiment(self):
        def mutate_then_echo(documents):
            documents[0].text = PRIVATE
            documents[0].issue_labels.append("invented-context")
            documents[0].issue_probabilities["invented-context"] = 1.0
            return stage_result("sentiment", documents)

        self.models["sentiment"].behavior = mutate_then_echo
        payloads = [record(), record(document_id="c-2", comment_id="c-2")]
        before = deepcopy(payloads)
        result = self.pipeline().run(payloads)
        self.assert_stage_failed(result, "sentiment")
        self.assertEqual(payloads, before)
        for accepted in result.records:
            self.assertEqual(accepted.normalized_text, f"normalized::{accepted.text}")
            self.assertNotIn("invented-context", accepted.stage_outputs["issue"]["predicted_issue_labels"])
        for independent in set(SUPPORTED) - {"sentiment"}:
            self.assertEqual(reports(result)[independent].status, "completed")
        self.assertNotIn("invented-context", result.model_dump_json())

    def test_raw_source_and_known_context_echoes_are_checked_before_schema_erasure(self):
        cases = [
            ("issue", "source_mode", "context"),
            ("issue", "source_subreddit", "MBA"),
            ("issue", "created_at", datetime(2025, 1, 1, tzinfo=timezone.utc)),
            ("issue", "url", "https://example.com/foreign"),
            ("issue", "parent_comment_id", "foreign"),
            ("issue", "parent_reddit_id", "t1_foreign"),
            ("issue", "parent_comment_text", PRIVATE),
            ("issue", "topic_id", 1),
            ("sentiment", "issue_labels", ["invented-context"]),
            ("emotion", "issue_probabilities", {"invented-context": 1.0}),
            ("solution", "response_type", "opinion"),
            ("solution", "response_probabilities", {"solution": 1.0}),
        ]
        for name, field, value in cases:
            with self.subTest(stage=name, field=field):
                def contradictory(documents, stage=name, key=field, replacement=value):
                    payload = stage_result(stage, documents).model_dump()
                    payload["predictions"][0][key] = replacement
                    return payload

                model = RecordingModel(name, self.calls, behavior=contradictory)
                payloads = [record(), record(document_id="c-2", comment_id="c-2")]
                before = deepcopy(payloads)
                result = self.pipeline(**{f"{name}_model": model}).run(payloads)
                self.assert_stage_failed(result, name)
                self.assertEqual(payloads, before)
                for independent in set(SUPPORTED) - {name}:
                    self.assertEqual(reports(result)[independent].status, "completed")

    def test_run_rejects_bounded_and_cyclic_extra_prediction_data_atomically(self):
        cycle = []
        cycle.append(cycle)
        deep = PRIVATE
        for _ in range(40):
            deep = {"nested": deep}
        for extra in (cycle, [PRIVATE] * 10001, deep):
            with self.subTest(extra_type=type(extra).__name__):
                def oversized(documents):
                    payload = stage_result("issue", documents).model_dump()
                    payload["predictions"][0]["unexpected"] = extra
                    return payload

                model = RecordingModel("issue", self.calls, behavior=oversized)
                result = self.pipeline(issue_model=model).run([
                    record(), record(document_id="c-2", comment_id="c-2"),
                ])
                self.assert_stage_failed(result, "issue")
                for independent in set(SUPPORTED) - {"issue"}:
                    self.assertEqual(reports(result)[independent].status, "completed")

    def test_real_language_and_preprocessing_are_offline_and_preserve_multilingual_text(self):
        texts = ("Fees are not affordable at IIM Sambalpur.", "पढ़ाई अच्छी है",
                 "পড়াশোনা ভালো", "ପାଠପଢ଼ା ଭଲ", "bhai placement ka kya scene hai?")
        for text in texts:
            with self.subTest(text=text):
                result = DownstreamPipeline().run([record(text=text, original_text=text)])
                self.assertEqual(reports(result)["language"].status, "completed")
                self.assertEqual(reports(result)["preprocessing"].status, "completed")
                self.assertEqual(result.records[0].original_text, text)
                self.assertTrue(result.records[0].normalized_text)
                self.assertEqual(result.batch_outputs, {})
                self.assertEqual(reports(result)["response"].status, "unavailable")


if __name__ == "__main__":
    unittest.main()
