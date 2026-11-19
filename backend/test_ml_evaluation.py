"""
Tests for the ML evaluation framework.

Covers metric computation, report building, class distribution reporting,
and serialization.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from ml.evaluation.metrics import ClassificationMetricSet, compute_metrics
from ml.evaluation.reports import (
    EvaluationReport,
    build_evaluation_report,
    report_class_distribution,
)


class TestMetricComputation(unittest.TestCase):
    """Tests for compute_metrics."""

    def test_single_label_accuracy(self):
        y_true = np.array(["a", "a", "b", "b", "a"])
        y_pred = np.array(["a", "a", "b", "b", "b"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertAlmostEqual(m.accuracy, 0.8)

    def test_single_label_macro_f1(self):
        y_true = np.array(["a", "a", "b", "b"])
        y_pred = np.array(["a", "b", "b", "a"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertIsNotNone(m.macro_f1)

    def test_single_label_per_class(self):
        y_true = np.array(["a", "a", "b", "b", "a"])
        y_pred = np.array(["a", "a", "b", "a", "b"])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        self.assertIn("a", m.per_class)
        self.assertIn("b", m.per_class)

    def test_multilabel_micro_f1(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
        m = compute_metrics(y_true, y_pred, ["x", "y", "z"], multilabel=True)
        self.assertIsNotNone(m.micro_f1)
        self.assertIsNotNone(m.macro_f1)

    def test_multilabel_per_label(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 0], [0, 0], [0, 1]])
        m = compute_metrics(y_true, y_pred, ["a", "b"], multilabel=True)
        self.assertIn("a", m.per_label)
        self.assertIn("b", m.per_label)

    def test_empty_arrays(self):
        m = compute_metrics(np.array([]), np.array([]), ["a", "b"])
        self.assertFalse(m.is_meaningful)

    def test_metrics_serializable_json(self):
        m = compute_metrics(
            np.array(["a", "b", "a"]),
            np.array(["a", "a", "b"]),
            ["a", "b"],
        )
        serializable = m.metrics_serializable()
        json.dumps(serializable)  # should not raise
        self.assertIn("macro_f1", serializable)


class TestEvaluationReport(unittest.TestCase):
    """Tests for EvaluationReport and build_evaluation_report."""

    def test_build_report_with_trained_model(self):
        report = build_evaluation_report(
            task_name="sentiment",
            model_name="tfidf_logistic",
            model_version="v1",
            n_train=100,
            n_validation=20,
            n_test=20,
            labels=["positive", "negative"],
            raw_report={"accuracy": 0.8, "macro_f1": 0.75, "warnings": []},
            is_multilabel=False,
            model_is_trained=True,
            leakage_detected=False,
        )
        self.assertTrue(report.is_meaningful)
        self.assertIn("positive", report.labels)
        self.assertEqual(report.model_name, "tfidf_logistic")

    def test_build_report_with_leakage(self):
        report = build_evaluation_report(
            task_name="sentiment",
            model_name="m",
            model_version="v1",
            n_train=50,
            n_validation=10,
            n_test=10,
            labels=["a", "b"],
            raw_report={},
            is_multilabel=False,
            model_is_trained=True,
            leakage_detected=True,
        )
        self.assertFalse(report.is_meaningful)
        self.assertTrue(any("leakage" in w.lower() for w in report.warnings))

    def test_build_report_untrained_model(self):
        report = build_evaluation_report(
            task_name="t",
            model_name="m",
            model_version="v1",
            n_train=0,
            n_validation=0,
            n_test=0,
            labels=[],
            raw_report=None,
            model_is_trained=False,
        )
        self.assertFalse(report.is_meaningful)
        self.assertTrue(any("not trained" in w for w in report.warnings))

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


class TestClassDistribution(unittest.TestCase):
    """Tests for report_class_distribution."""

    def test_balanced_classes(self):
        result = report_class_distribution({"a": 50, "b": 50}, 100)
        self.assertAlmostEqual(result["imbalance_ratio"], 0.5)
        self.assertIsNone(result["warning"])

    def test_imbalanced_classes(self):
        result = report_class_distribution({"a": 90, "b": 10}, 100)
        self.assertAlmostEqual(result["imbalance_ratio"], 0.9)
        self.assertIsNotNone(result["warning"])
        self.assertIn("imbalance", result["warning"].lower())

    def test_empty(self):
        result = report_class_distribution({}, 0)
        self.assertEqual(result["total"], 0)
        self.assertIn("warning", result)


class TestMultilabelAndNullEvaluationIntegrity(unittest.TestCase):
    """Tests for multi-label metric extraction and null/missing/empty value preservation."""

    def test_multilabel_report_extracts_subset_accuracy_hamming_loss_and_per_label(self):
        from app.services.nlp.sentiment_emotion.models.evaluation import evaluate_emotions

        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ])
        y_pred = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],  # 1 false positive on label c
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ])
        labels = ["hope", "curiosity", "gratitude"]
        raw = evaluate_emotions(y_true, y_pred, labels)
        report = build_evaluation_report(
            task_name="emotion",
            model_name="tfidf_logistic_multilabel",
            model_version="v1",
            n_train=30,
            n_validation=10,
            n_test=10,
            labels=labels,
            raw_report=raw,
            is_multilabel=True,
        )
        self.assertTrue(report.is_meaningful)
        m = report.metrics
        self.assertIsNone(m.accuracy)
        self.assertIsNone(m.confusion_matrix)
        self.assertEqual(m.per_class, {})
        self.assertAlmostEqual(m.subset_accuracy, 0.9)
        self.assertAlmostEqual(m.hamming_loss, 1.0 / 30.0, places=5)
        self.assertIsNotNone(m.macro_f1)
        self.assertIsNotNone(m.micro_f1)
        self.assertEqual(set(m.per_label.keys()), set(labels))
        for lbl in labels:
            self.assertIn("precision", m.per_label[lbl])
            self.assertIn("recall", m.per_label[lbl])
            self.assertIn("f1", m.per_label[lbl])
            self.assertIn("support", m.per_label[lbl])

    def test_singlelabel_report_never_populates_multilabel_metrics(self):
        from app.services.nlp.solution_detection.models.evaluation import evaluate_solution_detection

        y_true = [True, False] * 6
        y_pred = [True, False] * 6
        raw = evaluate_solution_detection(y_true, y_pred, ["not_solution", "solution"])
        report = build_evaluation_report(
            task_name="solution",
            model_name="tfidf_logistic_binary",
            model_version="v1",
            n_train=30,
            n_validation=12,
            n_test=12,
            labels=["not_solution", "solution"],
            raw_report=raw,
            is_multilabel=False,
        )
        self.assertTrue(report.is_meaningful)
        m = report.metrics
        self.assertAlmostEqual(m.accuracy, 1.0)
        self.assertAlmostEqual(m.macro_f1, 1.0)
        self.assertIsNone(m.subset_accuracy)
        self.assertIsNone(m.hamming_loss)
        self.assertIsNone(m.hamming_accuracy)
        self.assertEqual(m.per_label, {})
        self.assertEqual(m.confusion_matrix, [[6, 0], [0, 6]])
        self.assertEqual(m.per_class["not_solution"]["support"], 6)
        self.assertEqual(m.per_class["solution"]["support"], 6)

    def test_withheld_raw_report_keeps_metrics_null_not_zero(self):
        from app.services.nlp.solution_detection.models.evaluation import evaluate_solution_detection

        # Only one class present in evaluation -> withheld (is_meaningful=False)
        raw = evaluate_solution_detection([False] * 12, [False] * 12, ["not_solution", "solution"])
        self.assertFalse(raw.is_meaningful)
        report = build_evaluation_report(
            task_name="solution",
            model_name="tfidf_logistic_binary",
            model_version="v1",
            n_train=30,
            n_validation=12,
            n_test=12,
            labels=["not_solution", "solution"],
            raw_report=raw,
            is_multilabel=False,
        )
        self.assertFalse(report.is_meaningful)
        self.assertIsNone(report.metrics.accuracy)
        self.assertIsNone(report.metrics.macro_f1)
        self.assertIsNone(report.metrics.confusion_matrix)
        self.assertIsNotNone(report.metrics.reason_insufficient_classes)

    def test_null_missing_and_empty_inputs_preserve_none(self):
        # 1. None raw_report
        r_none = build_evaluation_report(
            task_name="emotion",
            model_name="m",
            model_version="v1",
            n_train=20,
            n_validation=10,
            n_test=10,
            labels=["a", "b"],
            raw_report=None,
            is_multilabel=True,
        )
        self.assertFalse(r_none.is_meaningful)
        self.assertIsNone(r_none.metrics.subset_accuracy)
        self.assertIsNone(r_none.metrics.hamming_loss)
        self.assertIsNone(r_none.metrics.macro_f1)

        # 2. Empty class in single-label compute_metrics
        y_true = np.array(["a", "a", "b", "b", "a", "b"])
        y_pred = np.array(["a", "a", "b", "b", "a", "b"])
        m_single = compute_metrics(y_true, y_pred, ["a", "b", "c"], multilabel=False)
        self.assertEqual(m_single.per_class["c"]["support"], 0)
        self.assertIsNone(m_single.per_class["c"]["f1"])
        self.assertIsNone(m_single.per_class["c"]["precision"])
        self.assertIsNone(m_single.per_class["c"]["recall"])
        serial = m_single.metrics_serializable()
        self.assertIsNone(serial["per_class"]["c"]["f1"])

        # 3. Missing (None) prediction in single-label compute_metrics counts as a miss
        y_true_miss = np.array(["a", "a", "b", "b", "a"], dtype=object)
        y_pred_miss = np.array(["a", None, "b", "b", "a"], dtype=object)
        m_miss = compute_metrics(y_true_miss, y_pred_miss, ["a", "b"], multilabel=False)
        self.assertAlmostEqual(m_miss.accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
