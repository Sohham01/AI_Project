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


if __name__ == "__main__":
    unittest.main()
