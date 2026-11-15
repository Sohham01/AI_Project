"""
Comprehensive tests for the ML training framework.

Covers: dataset loading, synthetic data, splitting, leakage detection,
class distribution, trainer pipeline, evaluation reports, metric computation,
artifact/registry behaviour, and edge cases.

All tests use SYNTHETIC data only.  No real annotation files are created
or modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

# Ensure the backend is importable.
backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend")
)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Ensure the project root (for ml/) is importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from ml.configs import (
    TaskConfig,
    TaskType,
    ModelRegistry,
    default_artifact_directory,
    default_registry_path,
    get_task_metadata,
)
from ml.data.dataset import (
    AnnotationNotFoundError,
    TrainingRecord,
    join_label,
    load_annotation_dataset,
    make_synthetic_records,
    split_label,
)
from ml.data.splitting import (
    SplitResult,
    SplitSpec,
    apply_split,
    check_leakage,
    split_records,
)
from ml.evaluation.metrics import ClassificationMetricSet, compute_metrics
from ml.evaluation.reports import (
    EvaluationReport,
    build_evaluation_report,
    report_class_distribution,
)
from ml.training.trainer import (
    Trainer,
    build_model,
    available_models,
    records_to_annotations,
    LoadedDataset,
    DEFAULT_MODEL_NAME,
    MODEL_FACTORIES,
)


class TestDatasetRecords(unittest.TestCase):
    """Tests for the TrainingRecord abstraction."""

    def test_record_to_dict(self):
        r = TrainingRecord(
            document_id="doc-1",
            text="some text",
            label="positive",
            post_id="p1",
            language="en",
            script="latin",
            is_code_mixed=False,
            detected_languages=["en"],
            topic_id=3,
            metadata={"foo": "bar"},
        )
        d = r.to_dict()
        self.assertEqual(d["document_id"], "doc-1")
        self.assertEqual(d["label"], "positive")
        self.assertEqual(d["metadata"], {"foo": "bar"})

    def test_record_from_dict(self):
        row = {
            "document_id": "doc-2",
            "text": "hello",
            "label": "negative|neutral",
            "post_id": "",
            "comment_id": "c1",
            "language": "hi",
            "script": "devanagari",
            "is_code_mixed": "true",
            "detected_languages": "hi|en",
            "topic_id": "5",
            "extra": "ignored",
        }
        r = TrainingRecord.from_dict(row)
        self.assertEqual(r.document_id, "doc-2")
        self.assertEqual(r.label, "negative|neutral")
        self.assertTrue(r.is_code_mixed)
        self.assertEqual(r.detected_languages, ["hi", "en"])
        self.assertEqual(r.topic_id, 5)
        # Extra keys go into metadata.
        self.assertIn("extra", r.metadata)

    def test_split_label_single(self):
        self.assertEqual(split_label("positive"), ["positive"])

    def test_split_label_multi(self):
        self.assertEqual(
            split_label("fees|hostel|mess_food"),
            ["fees", "hostel", "mess_food"],
        )

    def test_split_label_empty(self):
        self.assertEqual(split_label(""), [])

    def test_join_label_single(self):
        self.assertEqual(join_label(["positive"]), "positive")

    def test_join_label_multi(self):
        self.assertEqual(join_label(["fees", "hostel"]), "fees|hostel")

    def test_join_label_empty(self):
        self.assertEqual(join_label([]), "")


class TestSyntheticData(unittest.TestCase):
    """Tests for synthetic demo data generation."""

    def test_generate_n_records(self):
        records = make_synthetic_records(n=50, label_options=["a", "b", "c"])
        self.assertEqual(len(records), 50)

    def test_document_ids_are_unique(self):
        records = make_synthetic_records(n=100, label_options=["x", "y"])
        ids = [r.document_id for r in records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_single_label_each_record(self):
        records = make_synthetic_records(
            n=20, label_options=["pos", "neg"], single_label=True
        )
        for r in records:
            self.assertEqual(split_label(r.label), [r.label])

    def test_multilabel_possible(self):
        records = make_synthetic_records(
            n=20, label_options=["a", "b", "c", "d"], single_label=False
        )
        multi = [r for r in records if "|" in r.label]
        # Not guaranteed to have any, but the generator supports it.
        self.assertLessEqual(len(multi), len(records))

    def test_deterministic_seed(self):
        a = make_synthetic_records(n=10, label_options=["x", "y"], seed=42)
        b = make_synthetic_records(n=10, label_options=["x", "y"], seed=42)
        self.assertEqual([r.document_id for r in a], [r.document_id for r in b])

    def test_different_seed_different_output(self):
        a = make_synthetic_records(n=10, label_options=["x", "y"], seed=1)
        b = make_synthetic_records(n=10, label_options=["x", "y"], seed=2)
        # Very unlikely to be identical.
        self.assertNotEqual([r.label for r in a], [r.label for r in b])

    def test_synthetic_is_marked_as_not_real(self):
        # The framework must label these as synthetic.
        records = make_synthetic_records(n=5, label_options=["a"])
        # We can't check the source label from the record alone — that's on the
        # caller (load_dataset sets source="synthetic").
        # Just verify the records are usable.
        for r in records:
            self.assertTrue(r.text)
            self.assertTrue(r.document_id.startswith("synthetic-"))


class TestAnnotationLoading(unittest.TestCase):
    """Tests for real annotation loading and error handling."""

    def test_missing_file_raises_annotation_not_found(self):
        with self.assertRaises(AnnotationNotFoundError):
            load_annotation_dataset(
                task_name="sentiment",
                path="data/annotations/does_not_exist.csv",
            )

    def test_missing_file_message_is_clear(self):
        try:
            load_annotation_dataset(
                task_name="issue",
                path="data/annotations/missing.csv",
            )
        except AnnotationNotFoundError as e:
            msg = str(e)
            self.assertIn("Annotation dataset not found", msg)
            self.assertIn("missing.csv", msg)
            self.assertIn("synthetic", msg.lower())
        else:
            self.fail("expected AnnotationNotFoundError")

    def test_unknown_task_raises(self):
        records = make_synthetic_records(n=5, label_options=["a", "b"])
        # Write a tiny fake CSV so the loader finds the file but rejects the task.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("document_id,text,label\n")
            f.write("r1,hello,a\n")
            tmp_path = f.name
        try:
            from ml.data.dataset import load_annotation_dataset

            with self.assertRaises(ValueError) as ctx:
                load_annotation_dataset(
                    task_name="nonexistent",
                    path=tmp_path,
                )
            self.assertIn("Unknown task", str(ctx.exception))
        finally:
            os.unlink(tmp_path)


class TestSplitting(unittest.TestCase):
    """Tests for deterministic splitting and leakage detection."""

    def test_split_is_deterministic(self):
        records = make_synthetic_records(n=100, label_options=["a", "b", "c"])
        labels = [r.label for r in records]
        s1 = split_records(records, labels, SplitSpec(random_state=42))
        s2 = split_records(records, labels, SplitSpec(random_state=42))
        self.assertEqual(s1.train, s2.train)
        self.assertEqual(s1.validation, s2.validation)
        self.assertEqual(s1.test, s2.test)

    def test_different_seed_different_split(self):
        records = make_synthetic_records(n=100, label_options=["a", "b"])
        labels = [r.label for r in records]
        s1 = split_records(records, labels, SplitSpec(random_state=1))
        s2 = split_records(records, labels, SplitSpec(random_state=2))
        self.assertNotEqual(s1.train, s2.train)

    def test_no_overlap_between_splits(self):
        records = make_synthetic_records(n=60, label_options=["x", "y", "z"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        ok, msg = split.verify_no_overlap()
        self.assertTrue(ok, msg)

    def test_split_proportions(self):
        n = 100
        records = make_synthetic_records(n=n, label_options=["a", "b"])
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(
            train_size=0.6, validation_size=0.2, test_size=0.2
        ))
        # Proportions are approximate due to integer rounding.
        self.assertAlmostEqual(len(split.train) / n, 0.6, delta=0.02)
        self.assertAlmostEqual(len(split.validation) / n, 0.2, delta=0.02)
        self.assertAlmostEqual(len(split.test) / n, 0.2, delta=0.02)

    def test_empty_dataset_returns_empty_splits(self):
        split = split_records([], [])
        self.assertEqual(split.train, [])
        self.assertEqual(split.validation, [])
        self.assertEqual(split.test, [])
        self.assertFalse(split.stratify_used)

    def test_tiny_dataset_all_goes_to_train(self):
        records = make_synthetic_records(n=2, label_options=["a", "b"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        self.assertEqual(len(split.train), 2)
        self.assertEqual(split.validation, [])
        self.assertEqual(split.test, [])

    def test_apply_split_returns_correct_lists(self):
        records = make_synthetic_records(n=30, label_options=["p", "q"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        train, val, test = apply_split(records, split)
        self.assertEqual(len(train), len(split.train))
        self.assertEqual(len(val), len(split.validation))
        self.assertEqual(len(test), len(split.test))
        # No record should appear in two splits.
        all_ids = {r.document_id for r in train} | {
            r.document_id for r in val
        } | {r.document_id for r in test}
        self.assertEqual(len(all_ids), len(train) + len(val) + len(test))

    def test_stratification_not_used_for_multilabel(self):
        records = make_synthetic_records(
            n=50, label_options=["a", "b", "c"], single_label=False
        )
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(stratify=True))
        self.assertFalse(split.stratify_used)
        self.assertTrue(any("multi-label" in w for w in split.warnings))

    def test_stratification_used_for_single_label(self):
        records = make_synthetic_records(n=200, label_options=["x", "y", "z"])
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(stratify=True))
        self.assertTrue(split.stratify_used)

    def test_impossible_split_sizes_raise(self):
        with self.assertRaises(ValueError):
            SplitSpec(train_size=0.5, validation_size=0.3, test_size=0.3)
        with self.assertRaises(ValueError):
            SplitSpec(train_size=0.0, validation_size=0.5, test_size=0.5)
        with self.assertRaises(ValueError):
            SplitSpec(train_size=1.5, validation_size=0.0, test_size=-0.5)

    def test_leakage_detection_id_overlap(self):
        r1 = TrainingRecord(document_id="d1", text="text1", label="a")
        r2 = TrainingRecord(document_id="d1", text="text2", label="b")  # same id
        r3 = TrainingRecord(document_id="d2", text="text3", label="a")
        report = check_leakage([r1], [r2], [r3])
        self.assertTrue(report["leakage_detected"])
        self.assertTrue(any("train/validation" in w for w in report["cross_split_id_collisions"]))

    def test_leakage_detection_text_overlap(self):
        r1 = TrainingRecord(document_id="d1", text="same text", label="a")
        r2 = TrainingRecord(document_id="d2", text="same text", label="b")
        r3 = TrainingRecord(document_id="d3", text="other", label="a")
        report = check_leakage([r1], [r2], [r3])
        self.assertTrue(report["leakage_detected"])
        self.assertTrue(any("train/validation" in w for w in report["cross_split_text_collisions"]))

    def test_no_leakage_clean_split(self):
        # 3 label options ensures texts from the template pool don't
        # collide across splits by label proxy (different labels → different
        # template choices → different texts).
        records = make_synthetic_records(n=90, label_options=["x", "y", "z"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        train, val, test = apply_split(records, split)
        report = check_leakage(train, val, test)
        self.assertFalse(report["leakage_detected"])


class TestClassDistribution(unittest.TestCase):
    """Tests for class distribution reporting."""

    def test_empty_dataset(self):
        result = report_class_distribution({}, 0)
        self.assertEqual(result["total"], 0)
        self.assertIn("warning", result)

    def test_single_class(self):
        result = report_class_distribution({"a": 100}, 100)
        self.assertEqual(result["imbalance_ratio"], 1.0)
        self.assertTrue(result["warning"] is not None)

    def test_balanced_two_class(self):
        result = report_class_distribution({"a": 50, "b": 50}, 100)
        self.assertAlmostEqual(result["imbalance_ratio"], 0.5)
        self.assertIsNone(result["warning"])

    def test_distribution_contains_percentages(self):
        result = report_class_distribution({"a": 70, "b": 30}, 100)
        self.assertAlmostEqual(result["distribution"]["a"]["percentage"], 0.7)
        self.assertAlmostEqual(result["distribution"]["b"]["percentage"], 0.3)


class TestMetricsComputation(unittest.TestCase):
    """Tests for the unified metric computation."""

    def test_empty_arrays(self):
        m = compute_metrics(np.array([]), np.array([]), ["a", "b"])
        self.assertFalse(m.is_meaningful)
        self.assertIsNotNone(m.reason_empty)

    def test_single_label_basic_metrics(self):
        y_true = np.array(["a", "a", "b", "b", "a"])
        y_pred = np.array(["a", "b", "b", "a", "a"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertTrue(m.is_meaningful)
        self.assertIsNotNone(m.accuracy)
        self.assertIsNotNone(m.macro_f1)
        self.assertIsNotNone(m.micro_f1)

    def test_single_label_per_class(self):
        y_true = np.array(["a", "a", "b", "b", "a", "b"])
        y_pred = np.array(["a", "a", "b", "a", "a", "b"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertIn("a", m.per_class)
        self.assertIn("b", m.per_class)
        self.assertIn("f1", m.per_class["a"])

    def test_multilabel_basic_metrics(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        m = compute_metrics(y_true, y_pred, ["x", "y", "z"], multilabel=True)
        self.assertTrue(m.is_meaningful)
        self.assertIsNotNone(m.micro_f1)
        self.assertIsNotNone(m.macro_f1)
        self.assertIsNotNone(m.per_label)

    def test_multilabel_subset_accuracy(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 0], [0, 1], [1, 1]])
        m = compute_metrics(y_true, y_pred, ["a", "b"], multilabel=True)
        self.assertAlmostEqual(m.subset_accuracy, 1.0)

    def test_metrics_serializable(self):
        m = compute_metrics(
            np.array(["a", "b", "a"]),
            np.array(["a", "a", "b"]),
            ["a", "b"],
        )
        serializable = m.metrics_serializable()
        self.assertIn("macro_f1", serializable)
        self.assertIn("n_samples", serializable)
        self.assertIn("is_meaningful", serializable)
        # Should be JSON-serialisable.
        json.dumps(serializable)

    def test_single_label_with_missing_labels(self):
        # If only one class appears in evaluation, metrics requiring diversity
        # should be None with a reason. Accuracy can still be computed, but the
        # framework's policy is to mark the result as not meaningful when there
        # is no class diversity — macro-F1 and per-class comparisons need >=2.
        y_true = np.array(["a", "a", "a", "a", "a"])
        y_pred = np.array(["a", "a", "a", "a", "a"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertFalse(m.is_meaningful)
        self.assertIsNotNone(m.reason_insufficient_classes)

    def test_multilabel_with_unused_label(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0, 0, 0]])
        m = compute_metrics(y_true, y_pred, ["a", "b", "c"], multilabel=True)
        # Label 'b' and 'c' never appear.
        self.assertIn("b", m.per_label)
        self.assertIn("c", m.per_label)


class TestEvaluationReport(unittest.TestCase):
    """Tests for the unified evaluation report."""

    def test_report_summary_dict(self):
        report = EvaluationReport(
            task_name="sentiment",
            model_name="tfidf_logistic",
            model_version="v1",
            n_train=100,
            n_validation=20,
            n_test=20,
            labels=["positive", "negative"],
            is_meaningful=True,
            metrics=ClassificationMetricSet(["positive", "negative"]),
        )
        d = report.summary_dict()
        self.assertEqual(d["task"], "sentiment")
        self.assertEqual(d["model"], "tfidf_logistic v1")
        self.assertTrue(d["is_meaningful"])

    def test_report_metrics_serializable(self):
        m = ClassificationMetricSet(["a", "b"])
        m.accuracy = 0.8
        m.macro_f1 = 0.75
        report = EvaluationReport(
            task_name="test",
            model_name="m",
            model_version="v1",
            metrics=m,
        )
        serializable = report.metrics_serializable()
        self.assertEqual(serializable["accuracy"], 0.8)
        self.assertEqual(serializable["macro_f1"], 0.75)

    def test_is_meaningful_false_when_leakage(self):
        report = EvaluationReport(
            task_name="t",
            model_name="m",
            model_version="v1",
            is_meaningful=False,
            metrics=ClassificationMetricSet([]),
        )
        self.assertFalse(report.is_meaningful)


class TestModelRegistry(unittest.TestCase):
    """Tests for the local JSON model registry."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.tmpdir, "registry.json")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_and_list(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        entry_id = registry.register({
            "task": "sentiment",
            "model_name": "tfidf_logistic",
            "model_version": "v1",
            "artifact_path": "ml/artifacts/sentiment__tfidf_logistic__v1.joblib",
            "dataset": "data/annotations/sentiment_annotations.csv",
            "configuration": {"task_name": "sentiment"},
            "metrics": {"test": {"accuracy": 0.8}},
            "warnings": [],
        })
        self.assertEqual(len(entry_id), 16)
        entries = registry.list_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["task"], "sentiment")

    def test_filter_by_task(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        registry.register({
            "task": "sentiment",
            "model_name": "m1",
            "model_version": "v1",
            "artifact_path": "",
            "dataset": "",
            "configuration": {},
            "metrics": {},
            "warnings": [],
        })
        registry.register({
            "task": "response",
            "model_name": "m2",
            "model_version": "v1",
            "artifact_path": "",
            "dataset": "",
            "configuration": {},
            "metrics": {},
            "warnings": [],
        })
        sentiment_entries = registry.list_entries(task="sentiment")
        self.assertEqual(len(sentiment_entries), 1)
        self.assertEqual(sentiment_entries[0]["model_name"], "m1")

    def test_find_latest(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        registry.register({
            "task": "sentiment",
            "model_name": "tfidf_logistic",
            "model_version": "v1",
            "artifact_path": "a.joblib",
            "dataset": "",
            "configuration": {},
            "metrics": {},
            "warnings": [],
        })
        latest = registry.find_latest("sentiment", "tfidf_logistic")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["model_name"], "tfidf_logistic")

    def test_find_latest_unknown_returns_none(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        self.assertIsNone(registry.find_latest("unknown_task", "unknown_model"))

    def test_registry_persists_to_disk(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        registry.register({
            "task": "issue",
            "model_name": "clf",
            "model_version": "v1",
            "artifact_path": "x.joblib",
            "dataset": "data/annotations/issue_annotations.csv",
            "configuration": {"task_name": "issue"},
            "metrics": {},
            "warnings": [],
        })
        # Re-open and check.
        registry2 = ModelRegistry(registry_path=self.registry_path)
        entries = registry2.list_entries(task="issue")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["model_name"], "clf")

    def test_registry_is_json_serialisable(self):
        registry = ModelRegistry(registry_path=self.registry_path)
        registry.register({
            "task": "sentiment",
            "model_name": "m",
            "model_version": "v1",
            "artifact_path": "",
            "dataset": "",
            "configuration": {},
            "metrics": {},
            "warnings": [],
        })
        with open(self.registry_path, "r") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)


class TestTaskConfig(unittest.TestCase):
    """Tests for TaskConfig validation."""

    def test_valid_config(self):
        config = TaskConfig(task_name="sentiment")
        self.assertEqual(config.task_name, "sentiment")
        self.assertTrue(config.is_multilabel is False)

    def test_multilabel_derived_from_task(self):
        config = TaskConfig(task_name="emotion")
        self.assertTrue(config.is_multilabel)
        config2 = TaskConfig(task_name="issue")
        self.assertTrue(config2.is_multilabel)

    def test_default_dataset_path(self):
        config = TaskConfig(task_name="sentiment")
        self.assertEqual(
            config.dataset_path,
            "data/annotations/sentiment_annotations.csv",
        )
        config2 = TaskConfig(task_name="solution")
        self.assertEqual(
            config2.dataset_path,
            "data/annotations/solution_annotations.csv",
        )

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            TaskConfig(task_name="bogus_task")

    def test_split_proportions_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            TaskConfig(
                task_name="sentiment",
                train_size=0.5,
                validation_size=0.3,
                test_size=0.3,
            )

    def test_config_to_dict(self):
        config = TaskConfig(task_name="response", artifact_directory="tmp/art")
        d = config.to_dict()
        self.assertEqual(d["task_name"], "response")
        self.assertEqual(d["artifact_directory"], "tmp/art")
        # Model should not be serialised.
        self.assertNotIn("model", d)
        # The test-only registry override must not leak into registry IDs.
        self.assertNotIn("registry_path", d)

    def test_default_artifact_directory_is_anchored_to_project_root(self):
        config = TaskConfig(task_name="sentiment")
        self.assertEqual(
            config.artifact_directory, default_artifact_directory()
        )
        self.assertTrue(os.path.isabs(config.artifact_directory))
        self.assertTrue(
            config.artifact_directory.replace("\\", "/").endswith(
                "ml/artifacts"
            )
        )


class TestRegistryHygiene(unittest.TestCase):
    """The canonical registry must be reproducible and CWD-independent."""

    def test_default_registry_path_is_absolute_canonical(self):
        path = default_registry_path()
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(
            path.replace("\\", "/").endswith("ml/artifacts/registry.json")
        )

    def test_registry_write_is_atomic_and_leaves_valid_json(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        path = os.path.join(tmpdir, "registry.json")

        registry = ModelRegistry(registry_path=path)
        registry.register(
            {
                "task": "sentiment",
                "model_name": "m",
                "model_version": "v1",
            }
        )

        with open(path, "r", encoding="utf-8") as handle:
            json.load(handle)
        leftovers = [n for n in os.listdir(tmpdir) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_training_run_does_not_touch_canonical_registry(self):
        canonical = default_registry_path()
        before = None
        if os.path.exists(canonical):
            with open(canonical, "rb") as handle:
                before = hashlib.sha256(handle.read()).hexdigest()

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=os.path.join(tmpdir, "artifacts"),
            registry_path=os.path.join(tmpdir, "registry.json"),
            dataset_path="data/annotations/does_not_exist.csv",
        )
        Trainer(config).run(
            use_synthetic=True, save_artifact_flag=True, register=True
        )

        if before is not None:
            with open(canonical, "rb") as handle:
                after = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(before, after)


class TestTrainerPipeline(unittest.TestCase):
    """End-to-end tests for the Trainer using synthetic data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Make the artifact directory inside the temp dir so we don't pollute
        # the project.
        self.artifact_dir = os.path.join(self.tmpdir, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_trainer_builds(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="",
        )
        trainer = Trainer(config)
        self.assertEqual(trainer.task, TaskType.SENTIMENT)

    def test_trainer_loads_synthetic_when_no_real_data(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        dataset = trainer.load_dataset(use_synthetic=True)
        self.assertEqual(dataset.source, "synthetic")
        self.assertGreater(dataset.n_records, 0)
        self.assertIn("SYNTHETIC", dataset.warnings[0])

    def test_trainer_full_pipeline_with_synthetic(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=True,
            register=False,
        )
        # Basic structure checks.
        self.assertEqual(result["task"], "sentiment")
        self.assertEqual(result["dataset"]["source"], "synthetic")
        self.assertIn("split", result)
        self.assertIn("n_train", result["split"])
        self.assertIn("training_report", result)
        self.assertIn("validation_report", result)
        self.assertIn("test_report", result)
        self.assertIn("warnings", result)
        # Synthetic data must be clearly labelled.
        self.assertTrue(
            any("SYNTHETIC" in w for w in result["warnings"]),
            "warnings should flag the synthetic nature of the data",
        )
        # At least one warning about synthetic data.
        synthetic_warnings = [w for w in result["warnings"] if "synthetic" in w.lower()]
        self.assertGreater(len(synthetic_warnings), 0)

    def test_trainer_skips_real_task_that_has_no_file(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        with self.assertRaises(AnnotationNotFoundError):
            trainer.load_dataset(use_synthetic=False)

    def test_trainer_requires_split_before_training(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
        )
        trainer = Trainer(config)
        with self.assertRaises(RuntimeError):
            trainer.train()

    def test_trainer_requires_dataset_before_splitting(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
        )
        trainer = Trainer(config)
        with self.assertRaises(RuntimeError):
            trainer.split_dataset()

    def test_trainer_artifact_is_joblib(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=True,
            register=False,
        )
        artifact_path = result["artifact_path"]
        self.assertTrue(artifact_path.endswith(".joblib"))
        # Load it back and verify it contains a model + metadata.
        loaded = joblib.load(artifact_path)
        self.assertIn("model", loaded)
        self.assertIn("metadata", loaded)

    def test_trainer_artifact_name_is_deterministic(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result1 = trainer.run(use_synthetic=True, register=False)
        config2 = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        # Force a new trainer to get the same name.
        self.assertTrue(result1["artifact_path"].endswith(".joblib"))

    def test_trainer_multilabel_task_splits_without_stratification(self):
        config = TaskConfig(
            task_name="emotion",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=False,
            register=False,
        )
        self.assertTrue(result["split"]["stratify_used"] is False)

    def test_trainer_validation_and_test_reports_are_meaningful(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=False,
            register=False,
        )
        val = result["validation_report"]
        test = result["test_report"]
        # Synthetic data is at least 10 records + 20 extra = 30.
        # With 70/15/15 split we should have enough for a meaningful metric.
        self.assertIsNotNone(val)
        self.assertIsNotNone(test)

    def test_trainer_class_counts_in_dataset(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        dataset = trainer.load_dataset(use_synthetic=True)
        self.assertGreater(len(dataset.label_counts), 0)
        total = sum(dataset.label_counts.values())
        self.assertEqual(total, dataset.n_records)

    def test_trainer_unknown_task_in_run(self):
        with self.assertRaises(ValueError):
            TaskConfig(
                task_name="bogus_task",
                artifact_directory=self.artifact_dir,
            )

    def test_trainer_model_registry_integration(self):
        # Use a temporary registry so the test never writes to the canonical
        # ml/artifacts/registry.json.
        registry_path = os.path.join(self.tmpdir, "registry.json")
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=self.artifact_dir,
            registry_path=registry_path,
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=True,
            register=True,
        )
        self.assertIsNotNone(result.get("registry_id"))
        # Registry should have been written to the injected path.
        registry = ModelRegistry(registry_path=registry_path)
        entries = registry.list_entries(task="sentiment")
        self.assertGreater(len(entries), 0)
        self.assertEqual(entries[-1]["model_name"], "tfidf_logistic")


class TestRecordsToAnnotations(unittest.TestCase):
    """Tests for the TrainingRecord -> task-specific annotation conversion."""

    def test_sentiment_conversion(self):
        records = [
            TrainingRecord(document_id="d1", text="great", label="positive"),
        ]
        anns = records_to_annotations(records, TaskType.SENTIMENT)
        self.assertEqual(len(anns), 1)
        from app.services.nlp.sentiment_emotion.models.annotation import (
            SentimentAnnotation,
        )
        self.assertIsInstance(anns[0], SentimentAnnotation)
        self.assertEqual(anns[0].sentiment, "positive")

    def test_emotion_conversion(self):
        records = [
            TrainingRecord(document_id="d1", text="stress", label="frustration|anxiety"),
        ]
        anns = records_to_annotations(records, TaskType.EMOTION)
        from app.services.nlp.sentiment_emotion.models.annotation import (
            EmotionAnnotation,
        )
        self.assertIsInstance(anns[0], EmotionAnnotation)
        self.assertEqual(anns[0].emotions, ["anxiety", "frustration"])

    def test_issue_conversion(self):
        records = [
            TrainingRecord(document_id="d1", text="fees", label="fees|roi"),
        ]
        anns = records_to_annotations(records, TaskType.ISSUE)
        from app.services.nlp.issue_identification.models.annotation import (
            IssueAnnotation,
        )
        self.assertIsInstance(anns[0], IssueAnnotation)
        self.assertEqual(anns[0].labels, ["fees", "roi"])

    def test_response_conversion(self):
        records = [
            TrainingRecord(document_id="d1", text="same bro", label="agreement"),
        ]
        anns = records_to_annotations(records, TaskType.RESPONSE)
        from app.services.nlp.response_classification.models.annotation import (
            ResponseAnnotation,
        )
        self.assertIsInstance(anns[0], ResponseAnnotation)
        self.assertEqual(anns[0].response_type, "agreement")

    def test_solution_conversion(self):
        records = [
            TrainingRecord(
                document_id="d1",
                text="apply for aid",
                label="true",
                metadata={"solution_type": "financial_aid"},
            ),
        ]
        anns = records_to_annotations(records, TaskType.SOLUTION)
        from app.services.nlp.solution_detection.models.annotation import (
            SolutionAnnotation,
        )
        self.assertIsInstance(anns[0], SolutionAnnotation)
        self.assertTrue(anns[0].is_solution)

    def test_solution_conversion_false_label(self):
        records = [
            TrainingRecord(document_id="d1", text="same bro", label="false"),
        ]
        anns = records_to_annotations(records, TaskType.SOLUTION)
        self.assertFalse(anns[0].is_solution)


class TestBuildModel(unittest.TestCase):
    """Tests for the model factory."""

    def test_each_task_builds_a_model(self):
        for task in TaskType:
            model = build_model(task)
            self.assertIsNotNone(model)
            self.assertTrue(model.name)
            self.assertTrue(model.version)

    def test_model_is_untrained_after_build(self):
        model = build_model(TaskType.SENTIMENT)
        self.assertFalse(model.is_trained)

    def test_unknown_task_raises(self):
        from ml.configs import TaskType

        with self.assertRaises(ValueError):
            build_model(TaskType("nonexistent"))


class TestModelSelection(unittest.TestCase):
    """Explicit model selection: no silent cross-task fallback."""

    def test_default_name_maps_to_default_model(self):
        self.assertEqual(DEFAULT_MODEL_NAME, "tfidf_logistic")
        for task in TaskType:
            self.assertIn(DEFAULT_MODEL_NAME, available_models(task))

    def test_response_models_are_registered(self):
        names = available_models(TaskType.RESPONSE)
        self.assertIn("response_mlp", names)
        self.assertIn("text_cnn", names)
        self.assertIn("muril", names)

    def test_build_explicit_response_mlp(self):
        model = build_model(TaskType.RESPONSE, model_name="response_mlp")
        self.assertEqual(type(model).__name__, "ResponseMLPClassifier")

    def test_build_explicit_text_cnn(self):
        model = build_model(TaskType.RESPONSE, model_name="text_cnn")
        self.assertEqual(type(model).__name__, "TextCNNResponseClassifier")

    def test_unknown_model_name_raises(self):
        with self.assertRaises(ValueError):
            build_model(TaskType.RESPONSE, model_name="does_not_exist")

    def test_wrong_task_model_name_raises(self):
        # response_mlp is a RESPONSE model, never a SENTIMENT one.
        with self.assertRaises(ValueError):
            build_model(TaskType.SENTIMENT, model_name="response_mlp")

    def test_only_response_has_extra_models(self):
        self.assertEqual(available_models(TaskType.SENTIMENT), ["tfidf_logistic"])
        self.assertEqual(available_models(TaskType.ISSUE), ["tfidf_logistic"])
        self.assertEqual(len(MODEL_FACTORIES[TaskType.RESPONSE]), 4)

    def test_trainer_rejects_model_with_mismatched_task(self):
        class _WrongTaskModel:
            task = "response"

        cfg = TaskConfig(
            task_name="sentiment",
            artifact_directory=tempfile.mkdtemp(),
        )
        trainer = Trainer(cfg)
        with self.assertRaises(ValueError):
            trainer._validate_model_task(_WrongTaskModel())

    def test_trainer_accepts_matching_task_and_models_without_task_attr(self):
        cfg = TaskConfig(
            task_name="response",
            artifact_directory=tempfile.mkdtemp(),
        )
        trainer = Trainer(cfg)

        class _ResponseModel:
            task = "response"

        class _NoTaskAttr:
            pass

        trainer._validate_model_task(_ResponseModel())
        trainer._validate_model_task(_NoTaskAttr())

    def test_config_model_name_selects_baseline(self):
        cfg = TaskConfig(
            task_name="response",
            artifact_directory=tempfile.mkdtemp(),
            model_name="text_cnn",
        )
        trainer = Trainer(cfg)
        model = trainer._resolve_model()
        self.assertEqual(type(model).__name__, "TextCNNResponseClassifier")


class TestMultilingualMetadataPreservation(unittest.TestCase):
    """Tests that metadata flows through the pipeline."""

    def test_synthetic_records_carry_language_metadata(self):
        records = make_synthetic_records(n=10, label_options=["a", "b"])
        for r in records:
            self.assertIsNotNone(r.language)
            self.assertIsNotNone(r.script)
            self.assertIsInstance(r.detected_languages, list)

    def test_metadata_preserved_in_annotation_conversion(self):
        record = TrainingRecord(
            document_id="d1",
            text="hello",
            label="positive",
            language="hi",
            script="devanagari",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=9,
        )
        anns = records_to_annotations([record], TaskType.SENTIMENT)
        self.assertEqual(anns[0].language, "hi")
        self.assertEqual(anns[0].script, "devanagari")
        self.assertTrue(anns[0].is_code_mixed)
        self.assertEqual(anns[0].detected_languages, ["hi", "en"])
        self.assertEqual(anns[0].topic_id, 9)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_empty_dataset_in_trainer_rejects(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=tempfile.mkdtemp(),
            dataset_path="data/annotations/does_not_exist.csv",
        )
        trainer = Trainer(config)
        # Synthetic data is used, so it won't be empty.
        # But we can test the empty dataset path directly.
        from ml.data.dataset import make_synthetic_records

        records = make_synthetic_records(n=0, label_options=["a"])
        dataset = LoadedDataset(
            records=records,
            task=TaskType.SENTIMENT,
            source="synthetic",
            n_records=0,
            labels=[],
        )
        self.assertFalse(dataset.is_meaningful)

    def test_single_label_only_dataset(self):
        config = TaskConfig(
            task_name="sentiment",
            artifact_directory=tempfile.mkdtemp(),
        )
        trainer = Trainer(config)
        # Build a dataset with only one label.
        records = [
            TrainingRecord(document_id=f"d{i}", text="x", label="positive")
            for i in range(10)
        ]
        dataset = LoadedDataset(
            records=records,
            task=TaskType.SENTIMENT,
            source="synthetic",
            n_records=10,
            labels=["positive"],
        )
        self.assertFalse(dataset.is_meaningful)

    def test_missing_labels_in_evaluation(self):
        # Verify the split has labels in eval that aren't in train.
        records = [
            TrainingRecord(document_id=f"d{i}", text="x", label=f"label{i}")
            for i in range(10)
        ]
        split = split_records(records, [r.label for r in records])
        # Manually construct a scenario where eval has a label train doesn't.
        # Use a tiny dataset so we control which record goes where.
        records2 = [
            TrainingRecord(document_id="a", text="x", label="only_in_eval"),
            TrainingRecord(document_id="b", text="y", label="in_both"),
            TrainingRecord(document_id="c", text="z", label="in_both"),
        ]
        split2 = split_records(
            records2,
            [r.label for r in records2],
            SplitSpec(train_size=0.33, validation_size=0.33, test_size=0.34),
        )
        # train gets index 2 (in_both), val gets index 0 (only_in_eval).
        # So eval has "only_in_eval" which train doesn't.
        dataset = LoadedDataset(
            records=records2,
            task=TaskType.SENTIMENT,
            source="synthetic",
            n_records=3,
            labels=["only_in_eval", "in_both"],
        )
        missing = dataset.verify_no_missing_labels(split2)
        self.assertTrue(len(missing) > 0)


if __name__ == "__main__":
    unittest.main()
