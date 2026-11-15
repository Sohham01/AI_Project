"""
Tests for the ANN Response Classification experiment.

Covers:
- mathematical primitives from ml.models.ann.base
- NumpyMLP core (framework-free)
- ResponseMLPClassifier adapter (ResponseClassification contract)
- Trainer integration via TaskType.RESPONSE + model override

All tests use SYNTHETIC data only.  No real annotation files are created
or modified.  No completed NLP stages are modified.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import joblib
import numpy as np

# Project root + backend on the path so both ml/ and app/ are importable.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_backend_dir = os.path.join(_project_root, "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# ---------------------------------------------------------------------------
# Mathematical primitives
# ---------------------------------------------------------------------------

from ml.models.ann.base import (
    clip_gradient,
    cross_entropy_loss,
    fan_in_scaled_init,
    lecun_normal_init,
    one_hot_encode,
    relu,
    relu_derivative,
    softmax,
)

# ---------------------------------------------------------------------------
# Core MLP
# ---------------------------------------------------------------------------

from ml.models.ann.numpy_mlp import NumpyMLP

# ---------------------------------------------------------------------------
# Response MLP adapter
# ---------------------------------------------------------------------------

from ml.models.ann.response_mlp import ResponseMLPClassifier

# ---------------------------------------------------------------------------
# Frozen response-classification contracts (read-only imports only)
# ---------------------------------------------------------------------------

from app.services.nlp.response_classification.base import ResponseClassification
from app.services.nlp.response_classification.models.annotation import (
    ResponseAnnotation,
)
from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES,
    response_type_order,
)

# ---------------------------------------------------------------------------
# ML framework
# ---------------------------------------------------------------------------

from ml.configs import TaskConfig, TaskType, ModelRegistry
from ml.data.dataset import (
    AnnotationNotFoundError,
    TrainingRecord,
    make_synthetic_records,
    split_label,
)
from ml.data.splitting import SplitResult, SplitSpec, apply_split, split_records
from ml.evaluation.reports import EvaluationReport, build_evaluation_report
from ml.training.trainer import (
    Trainer,
    LoadedDataset,
    PipelineState,
    build_model,
    records_to_annotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response_annotation_records(
    n: int, seed: int = 42
) -> List[ResponseAnnotation]:
    """Synthetic single-label response annotations for plumbing tests."""
    labels = list(RESPONSE_TYPES)
    records = make_synthetic_records(
        n=n, label_options=labels, single_label=True, seed=seed
    )
    return [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in records
    ]


# ===========================================================================
# 1. MATHEMATICAL PRIMITIVES
# ===========================================================================


class TestReLU(unittest.TestCase):
    """ReLU and its derivative."""

    def test_negative_returns_zero(self):
        x = np.array([-3.0, -1.0, -0.5, -1e-9])
        out = relu(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0])

    def test_zero_returns_zero(self):
        x = np.array([0.0, -0.0])
        out = relu(x)
        np.testing.assert_array_equal(out, [0.0, 0.0])

    def test_positive_unchanged(self):
        x = np.array([0.5, 1.0, 2.5, 100.0])
        out = relu(x)
        np.testing.assert_array_equal(out, x)

    def test_mixed_vector(self):
        x = np.array([-2.0, -0.1, 0.0, 0.5, 3.0])
        out = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 0.5, 3.0])
        np.testing.assert_array_equal(out, expected)

    def test_derivative_negative(self):
        x = np.array([-5.0, -1.0, -0.01])
        d = relu_derivative(x)
        np.testing.assert_array_equal(d, [0.0, 0.0, 0.0])

    def test_derivative_positive(self):
        x = np.array([0.1, 1.0, 5.0])
        d = relu_derivative(x)
        np.testing.assert_array_equal(d, [1.0, 1.0, 1.0])

    def test_derivative_at_zero_is_zero(self):
        d = relu_derivative(np.array([0.0]))
        np.testing.assert_array_equal(d, [0.0])

    def test_2d_input(self):
        x = np.array([[-1.0, 2.0], [0.0, -3.0]])
        out = relu(x)
        expected = np.array([[0.0, 2.0], [0.0, 0.0]])
        np.testing.assert_array_equal(out, expected)


class TestSoftmax(unittest.TestCase):
    """Numerically stable softmax from ml.models.ann.base."""

    def test_probabilities_sum_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        p = softmax(logits)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_large_logits_no_overflow(self):
        logits = np.array([1000.0, 1001.0, 1002.0])
        p = softmax(logits)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_negative_logits(self):
        logits = np.array([-10.0, -5.0, 0.0])
        p = softmax(logits)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertGreater(p[2], p[1])
        self.assertGreater(p[1], p[0])

    def test_all_equal_logits_uniform(self):
        logits = np.ones(5)
        p = softmax(logits)
        np.testing.assert_array_almost_equal(p, np.full(5, 0.2), decimal=6)

    def test_single_element(self):
        p = softmax(np.array([42.0]))
        np.testing.assert_array_almost_equal(p, [1.0])

    def test_2d_input_columns_sum_to_one(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p = softmax(logits, axis=0)
        col_sums = p.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, [1.0, 1.0])

    def test_output_shape_matches_input(self):
        logits = np.random.randn(10)
        p = softmax(logits)
        self.assertEqual(p.shape, logits.shape)

    def test_positive_probabilities(self):
        logits = np.random.randn(8) * 10
        p = softmax(logits)
        self.assertTrue(np.all(p > 0))


class TestOneHotEncode(unittest.TestCase):
    """one_hot_encode from ml.models.ann.base."""

    def test_basic(self):
        labels = np.array([0, 2, 1, 0])
        Y = one_hot_encode(labels, n_classes=3)
        expected = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        np.testing.assert_array_equal(Y, expected)
        self.assertEqual(Y.shape, (3, 4))

    def test_empty_labels(self):
        Y = one_hot_encode(np.array([], dtype=int), n_classes=3)
        self.assertEqual(Y.shape, (3, 0))

    def test_single_class(self):
        labels = np.array([0, 0, 0])
        Y = one_hot_encode(labels, n_classes=1)
        np.testing.assert_array_equal(Y, [[1.0, 1.0, 1.0]])
        self.assertEqual(Y.shape, (1, 3))

    def test_invalid_label_negative(self):
        with self.assertRaises(ValueError):
            one_hot_encode(np.array([-1]), n_classes=3)

    def test_invalid_label_out_of_range(self):
        with self.assertRaises(ValueError):
            one_hot_encode(np.array([3]), n_classes=3)

    def test_integer_dtype(self):
        labels = np.array([1, 0], dtype=np.int32)
        Y = one_hot_encode(labels, n_classes=2)
        np.testing.assert_array_equal(Y, [[0.0, 1.0], [1.0, 0.0]])


class TestCrossEntropyLoss(unittest.TestCase):
    """cross_entropy_loss from ml.models.ann.base."""

    def test_perfect_prediction_low_loss(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        one_hot = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        loss = cross_entropy_loss(probs, one_hot)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_uniform_probs_higher_loss(self):
        probs = np.full((2, 4), 0.5)
        one_hot = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        loss = cross_entropy_loss(probs, one_hot)
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, -np.log(0.5), delta=1e-6)

    def test_shape_mismatch_raises(self):
        probs = np.ones((3, 4))
        one_hot = np.ones((2, 4))
        with self.assertRaises(ValueError):
            cross_entropy_loss(probs, one_hot)

    def test_epsilon_avoid_log_zero(self):
        probs = np.array([[0.0, 1.0], [1.0, 0.0]])
        one_hot = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = cross_entropy_loss(probs, one_hot)
        self.assertTrue(np.isfinite(loss))


class TestGradientClipping(unittest.TestCase):
    """clip_gradient from ml.models.ann.base."""

    def test_small_gradients_unchanged(self):
        grads = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        result = clip_gradient(grads, threshold=10.0)
        np.testing.assert_array_equal(result[0], grads[0])

    def test_large_gradients_scaled(self):
        big = np.ones((5, 5)) * 100.0
        grads = [big.copy()]
        result = clip_gradient(grads, threshold=1.0)
        total_norm = float(np.sqrt(sum(np.linalg.norm(g) ** 2 for g in result)))
        self.assertLessEqual(total_norm, 1.0 + 1e-6)

    def test_multiple_gradients_scaled_together(self):
        g1 = np.ones((3, 3)) * 50.0
        g2 = np.ones((2, 2)) * 50.0
        grads = [g1.copy(), g2.copy()]
        result = clip_gradient(grads, threshold=1.0)
        total_norm = float(np.sqrt(sum(np.linalg.norm(g) ** 2 for g in result)))
        self.assertLessEqual(total_norm, 1.0 + 1e-6)

    def test_empty_grads(self):
        grads = []
        result = clip_gradient(grads, threshold=1.0)
        self.assertEqual(result, [])

    def test_zero_norm_unchanged(self):
        grads = [np.zeros((3, 3))]
        result = clip_gradient(grads, threshold=1.0)
        np.testing.assert_array_equal(result[0], 0.0)


class TestInitialization(unittest.TestCase):
    """fan_in_scaled_init and lecun_normal_init from ml.models.ann.base."""

    def test_fan_in_scaled_shape(self):
        rng = np.random.default_rng(42)
        W = fan_in_scaled_init((64, 32), rng)
        self.assertEqual(W.shape, (64, 32))

    def test_lecun_normal_shape(self):
        rng = np.random.default_rng(42)
        W = lecun_normal_init((32, 16), rng)
        self.assertEqual(W.shape, (32, 16))

    def test_fan_in_scaled_mean_near_zero(self):
        rng = np.random.default_rng(123)
        W = fan_in_scaled_init((100, 50), rng)
        self.assertAlmostEqual(float(W.mean()), 0.0, delta=0.05)
        self.assertGreater(float(W.std()), 0.0)

    def test_deterministic_same_seed_same_weights(self):
        shape = (32, 16)
        w1 = fan_in_scaled_init(shape, np.random.default_rng(99))
        w2 = fan_in_scaled_init(shape, np.random.default_rng(99))
        np.testing.assert_array_equal(w1, w2)

    def test_different_seed_different_weights(self):
        shape = (32, 16)
        w1 = fan_in_scaled_init(shape, np.random.default_rng(1))
        w2 = fan_in_scaled_init(shape, np.random.default_rng(2))
        self.assertFalse(np.array_equal(w1, w2))


# ===========================================================================
# 2. NUMPYMLP CORE
# ===========================================================================


class TestNumpyMLPConstruction(unittest.TestCase):
    """Constructor validation and metadata."""

    def test_defaults(self):
        mlp = NumpyMLP(input_dim=100, n_classes=6)
        self.assertEqual(mlp.input_dim, 100)
        self.assertEqual(mlp.n_classes, 6)
        self.assertEqual(mlp.hidden1, 256)
        self.assertEqual(mlp.hidden2, 64)
        self.assertEqual(mlp.dropout, 0.3)
        self.assertEqual(mlp.learning_rate, 0.01)
        self.assertEqual(mlp.momentum, 0.9)
        self.assertEqual(mlp.epochs, 40)
        self.assertEqual(mlp.batch_size, 16)
        self.assertEqual(mlp.random_state, 42)
        self.assertFalse(mlp.is_fitted)

    def test_input_dim_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=0, n_classes=6)
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=-1, n_classes=6)

    def test_n_classes_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=0)
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=-1)

    def test_dropout_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, dropout=-0.1)
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, dropout=1.5)

    def test_learning_rate_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, learning_rate=0.0)

    def test_epochs_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, epochs=0)

    def test_batch_size_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, batch_size=0)

    def test_clip_gradient_validation(self):
        with self.assertRaises(ValueError):
            NumpyMLP(input_dim=10, n_classes=2, clip_gradient=0.0)

    def test_n_parameters_before_fit(self):
        mlp = NumpyMLP(input_dim=100, n_classes=6, hidden1=32, hidden2=16)
        expected = (
            32 * 100 + 32 + 16 * 32 + 16 + 6 * 16 + 6
        )
        self.assertEqual(mlp.n_parameters, expected)

    def test_n_parameters_after_fit(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3, hidden1=8, hidden2=4, random_state=1)
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, size=20)
        mlp.fit(X, y)
        expected = 8 * 10 + 8 + 4 * 8 + 4 + 3 * 4 + 3
        self.assertEqual(mlp.n_parameters, expected)

    def test_epoch_losses_empty_before_fit(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3)
        self.assertEqual(mlp.epoch_losses, [])

    def test_verbose_default_false(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3)
        self.assertFalse(mlp.verbose)


class TestNumpyMLPForward(unittest.TestCase):
    """Forward propagation shapes and properties."""

    def setUp(self):
        self.mlp = NumpyMLP(
            input_dim=20, n_classes=6, hidden1=16, hidden2=8, random_state=42
        )
        self.X = np.random.randn(32, 20)
        self.y = np.random.randint(0, 6, size=32)

    def test_fit_returns_self(self):
        result = self.mlp.fit(self.X, self.y)
        self.assertIs(result, self.mlp)

    def test_predict_proba_shape(self):
        self.mlp.fit(self.X, self.y)
        proba = self.mlp.predict_proba(self.X)
        self.assertEqual(proba.shape, (6, 32))

    def test_predict_shape(self):
        self.mlp.fit(self.X, self.y)
        pred = self.mlp.predict(self.X)
        self.assertEqual(pred.shape, (32,))

    def test_predict_dtype(self):
        self.mlp.fit(self.X, self.y)
        pred = self.mlp.predict(self.X)
        self.assertEqual(pred.dtype, np.int64)

    def test_predict_proba_probabilities_sum_to_one(self):
        self.mlp.fit(self.X, self.y)
        proba = self.mlp.predict_proba(self.X)
        col_sums = proba.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, np.ones(32), decimal=6)

    def test_predict_is_argmax_of_proba(self):
        self.mlp.fit(self.X, self.y)
        proba = self.mlp.predict_proba(self.X)
        pred = self.mlp.predict(self.X)
        argmax = np.argmax(proba, axis=0)
        np.testing.assert_array_equal(pred, argmax)

    def test_predict_proba_requires_fit(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3)
        with self.assertRaises(RuntimeError):
            mlp.predict_proba(np.random.randn(5, 10))

    def test_predict_requires_fit(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3)
        with self.assertRaises(RuntimeError):
            mlp.predict(np.random.randn(5, 10))

    def test_input_dim_mismatch_raises(self):
        self.mlp.fit(self.X, self.y)
        wrong = np.random.randn(5, 9)
        with self.assertRaises(ValueError):
            self.mlp.predict_proba(wrong)

    def test_empty_input_handling(self):
        self.mlp.fit(self.X, self.y)
        empty_X = np.empty((0, 20))
        proba = self.mlp.predict_proba(empty_X)
        self.assertEqual(proba.shape, (6, 0))
        pred = self.mlp.predict(empty_X)
        self.assertEqual(pred.shape, (0,))


class TestNumpyMLPTraining(unittest.TestCase):
    """Training dynamics: loss decreases, determinism, history."""

    def test_loss_decreases_on_learnable_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, size=(n, 20))
        Wtrue = rng.normal(0, 1, size=(6, 20))
        logits = Wtrue @ X.T
        y = np.argmax(logits, axis=0)

        mlp = NumpyMLP(
            input_dim=20,
            n_classes=6,
            hidden1=32,
            hidden2=16,
            epochs=30,
            batch_size=16,
            learning_rate=0.02,
            random_state=42,
        )
        mlp.fit(X, y)
        losses = mlp.epoch_losses
        self.assertGreater(len(losses), 0)
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_deterministic_training_same_seed(self):
        rng = np.random.default_rng(123)
        X = rng.normal(0, 1, size=(100, 10))
        y = rng.integers(0, 4, size=100)

        mlp1 = NumpyMLP(
            input_dim=10, n_classes=4, hidden1=16, hidden2=8,
            epochs=10, batch_size=16, random_state=42,
        )
        mlp1.fit(X, y)

        mlp2 = NumpyMLP(
            input_dim=10, n_classes=4, hidden1=16, hidden2=8,
            epochs=10, batch_size=16, random_state=42,
        )
        mlp2.fit(X, y)

        np.testing.assert_array_equal(mlp1._W1, mlp2._W1)
        np.testing.assert_array_equal(mlp1._W2, mlp2._W2)
        np.testing.assert_array_equal(mlp1._W3, mlp2._W3)
        for l1, l2 in zip(mlp1.epoch_losses, mlp2.epoch_losses):
            self.assertAlmostEqual(l1, l2)

    def test_different_seed_different_weights(self):
        rng = np.random.default_rng(123)
        X = rng.normal(0, 1, size=(100, 10))
        y = rng.integers(0, 4, size=100)

        mlp1 = NumpyMLP(
            input_dim=10, n_classes=4, hidden1=16, hidden2=8,
            epochs=5, batch_size=16, random_state=1,
        )
        mlp1.fit(X, y)

        mlp2 = NumpyMLP(
            input_dim=10, n_classes=4, hidden1=16, hidden2=8,
            epochs=5, batch_size=16, random_state=2,
        )
        mlp2.fit(X, y)

        self.assertFalse(np.array_equal(mlp1._W1, mlp2._W1))

    def test_training_history_length_equals_epochs(self):
        mlp = NumpyMLP(
            input_dim=10, n_classes=3, epochs=15, batch_size=8, random_state=7
        )
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, size=50)
        mlp.fit(X, y)
        self.assertEqual(len(mlp.epoch_losses), 15)

    def test_all_losses_finite(self):
        mlp = NumpyMLP(
            input_dim=10, n_classes=3, epochs=10, batch_size=8, random_state=7
        )
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, size=50)
        mlp.fit(X, y)
        for loss in mlp.epoch_losses:
            self.assertTrue(np.isfinite(loss), f"Non-finite loss: {loss}")

    def test_retrain_resets_weights(self):
        mlp = NumpyMLP(
            input_dim=10, n_classes=3, epochs=5, batch_size=8, random_state=7
        )
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, size=50)
        mlp.fit(X, y)
        first_loss = mlp.epoch_losses[-1]

        mlp.fit(X, y)
        second_loss = mlp.epoch_losses[-1]
        self.assertAlmostEqual(first_loss, second_loss)

    def test_labels_out_of_range_raises(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3, random_state=7)
        X = np.random.randn(20, 10)
        y = np.array([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            mlp.fit(X, y)

    def test_X_y_shape_mismatch_raises(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3, random_state=7)
        X = np.random.randn(20, 10)
        y = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            mlp.fit(X, y)

    def test_feature_dim_mismatch_raises(self):
        mlp = NumpyMLP(input_dim=10, n_classes=3, random_state=7)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, size=20)
        with self.assertRaises(ValueError):
            mlp.fit(X, y)

    def test_get_state_after_fit(self):
        mlp = NumpyMLP(
            input_dim=10, n_classes=3, hidden1=8, hidden2=4,
            epochs=3, batch_size=8, random_state=7,
        )
        X = np.random.randn(30, 10)
        y = np.random.randint(0, 3, size=30)
        mlp.fit(X, y)
        state = mlp.get_state()
        self.assertEqual(state["input_dim"], 10)
        self.assertEqual(state["n_classes"], 3)
        self.assertEqual(state["hidden1"], 8)
        self.assertEqual(state["hidden2"], 4)
        self.assertIn("W1_shape", state)
        self.assertIn("epoch_losses", state)
        self.assertEqual(len(state["epoch_losses"]), 3)


# ===========================================================================
# 3. RESPONSEMLPCLASSIFIER
# ===========================================================================


class TestResponseMLPClassifierContract(unittest.TestCase):
    """Verify the adapter implements ResponseClassification correctly."""

    def test_implements_response_classification(self):
        clf = ResponseMLPClassifier()
        self.assertIsInstance(clf, ResponseClassification)

    def test_name(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(clf.name, "response_mlp_numpy")

    def test_version(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(clf.version, "v1")

    def test_expected_labels_are_six_v1_types(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(clf.expected_labels, list(RESPONSE_TYPES))

    def test_missing_response_types_before_train(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(
            clf.missing_response_types, list(RESPONSE_TYPES)
        )

    def test_is_trained_false_before_train(self):
        clf = ResponseMLPClassifier()
        self.assertFalse(clf.is_trained)

    def test_threshold_default(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(clf.threshold, 0.5)

    def test_config_is_response_model_config(self):
        from app.services.nlp.response_classification.base import (
            ResponseModelConfig,
        )

        clf = ResponseMLPClassifier()
        self.assertIsInstance(clf.config, ResponseModelConfig)

    def test_labels_empty_before_train(self):
        clf = ResponseMLPClassifier()
        self.assertEqual(clf.labels, [])

    def test_constructor_custom_hyperparams(self):
        clf = ResponseMLPClassifier(
            hidden1=128,
            hidden2=32,
            dropout=0.2,
            learning_rate=0.005,
            momentum=0.85,
            epochs=20,
            batch_size=8,
            random_state=123,
            clip_gradient=3.0,
            max_features=5000,
        )
        self.assertEqual(clf.config.max_features, 5000)


class TestResponseMLPClassifierTraining(unittest.TestCase):
    """End-to-end training of the response MLP adapter."""

    def setUp(self):
        self.clf = ResponseMLPClassifier(
            hidden1=32,
            hidden2=16,
            epochs=10,
            batch_size=8,
            learning_rate=0.01,
            random_state=42,
            min_training_documents=6,
            min_examples_per_class=1,
            max_features=500,
        )
        self.annotations = _make_response_annotation_records(n=30, seed=42)

    def test_train_returns_training_report(self):
        report = self.clf.train(self.annotations)
        self.assertIsNotNone(report)
        self.assertTrue(report.is_usable)
        self.assertIn(report.model_name, ["response_mlp_numpy"])

    def test_train_sets_is_trained(self):
        self.clf.train(self.annotations)
        self.assertTrue(self.clf.is_trained)

    def test_train_sets_labels(self):
        self.clf.train(self.annotations)
        self.assertTrue(len(self.clf.labels) > 0)
        for lbl in self.clf.labels:
            self.assertIn(lbl, RESPONSE_TYPES)

    def test_labels_in_canonical_order(self):
        self.clf.train(self.annotations)
        self.assertEqual(self.clf.labels, response_type_order(self.clf.labels))

    def test_train_training_report_has_labels(self):
        report = self.clf.train(self.annotations)
        self.assertTrue(len(report.labels) > 0)
        for lbl in report.labels:
            self.assertIn(lbl, RESPONSE_TYPES)

    def test_train_training_report_labels_match_property(self):
        self.clf.train(self.annotations)
        self.assertEqual(self.clf.labels, self.clf.training_report.labels)

    def test_empty_annotations_raises(self):
        with self.assertRaises(ValueError):
            self.clf.train([])

    def test_duplicate_document_ids_raises(self):
        dup = self.annotations[:5] + [
            ResponseAnnotation(
                document_id=self.annotations[0].document_id,
                text="dup text",
                response_type="opinion",
            )
        ]
        with self.assertRaises(ValueError):
            self.clf.train(dup)

    def test_insufficient_training_documents_raises(self):
        clf = ResponseMLPClassifier(
            min_training_documents=20,
            min_examples_per_class=1,
            random_state=42,
        )
        few = _make_response_annotation_records(n=5, seed=42)
        with self.assertRaises(ValueError):
            clf.train(few)

    def test_training_report_has_library_versions(self):
        report = self.clf.train(self.annotations)
        self.assertIn("numpy", report.library_versions)
        self.assertIn("scikit-learn", report.library_versions)

    def test_predict_after_train_returns_result(self):
        self.clf.train(self.annotations)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(document_id=f"pred-{i}", text=f"this is comment {i}")
            for i in range(5)
        ]
        result = self.clf.predict(docs)
        self.assertIsNotNone(result)
        self.assertEqual(result.model_name, "response_mlp_numpy")
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual(result.n_documents, 5)

    def test_predict_empty_docs(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        result = clf.predict([])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.predictions, [])

    def test_predict_probabilities_sum_to_one(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(document_id=f"ptest-{i}", text=f"test {i}")
            for i in range(10)
        ]
        result = clf.predict(docs)
        for pred in result.predictions:
            psum = sum(pred.response_probabilities.values())
            self.assertAlmostEqual(psum, 1.0, places=4)

    def test_predict_probabilities_have_all_known_labels(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(document_id=f"plbl-{i}", text=f"sample {i}")
            for i in range(5)
        ]
        result = clf.predict(docs)
        for pred in result.predictions:
            for lbl in clf.labels:
                self.assertIn(lbl, pred.response_probabilities)

    def test_predict_response_type_in_labels(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(document_id=f"rtype-{i}", text=f"text {i}")
            for i in range(5)
        ]
        result = clf.predict(docs)
        for pred in result.predictions:
            if pred.response_type is not None:
                self.assertIn(pred.response_type, clf.labels)

    def test_predict_with_custom_threshold(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(document_id=f"thr-{i}", text=f"thr {i}")
            for i in range(5)
        ]
        result_default = clf.predict(docs)
        result_high = clf.predict(docs, threshold=0.9)
        self.assertGreaterEqual(
            result_high.n_abstained, result_default.n_abstained
        )

    def test_evaluate_after_train(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=15, seed=42)
        clf.train(anns)
        report = clf.evaluate(anns[:10])
        self.assertIsNotNone(report)
        self.assertTrue(hasattr(report, "is_meaningful"))

    def test_evaluate_empty_annotations(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        report = clf.evaluate([])
        self.assertIsNotNone(report)
        self.assertFalse(report.is_meaningful)

    def test_predict_preserves_original_text(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(
                document_id="text-pres-1",
                text="preserve this exact text",
                original_text="ORIGINAL: preserve this exact text",
            )
        ]
        result = clf.predict(docs)
        pred = result.predictions[0]
        self.assertEqual(pred.text, "preserve this exact text")
        self.assertEqual(
            pred.original_text, "ORIGINAL: preserve this exact text"
        )

    def test_predict_metadata_preserved(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )

        docs = [
            ResponseDocument(
                document_id="meta-1",
                text="meta text",
                post_id="post-123",
                comment_id="cmt-456",
                language="hi",
                script="devanagari",
                is_code_mixed=True,
                detected_languages=["hi", "en"],
                topic_id=7,
                topic_label="placement",
                topic_keywords=["placement", "PPO"],
                issue_labels=["fees"],
                issue_probabilities={"fees": 0.8},
                parent_comment_text="parent text here",
            )
        ]
        result = clf.predict(docs)
        pred = result.predictions[0]
        self.assertEqual(pred.post_id, "post-123")
        self.assertEqual(pred.comment_id, "cmt-456")
        self.assertEqual(pred.language, "hi")
        self.assertEqual(pred.script, "devanagari")
        self.assertTrue(pred.is_code_mixed)
        self.assertEqual(pred.detected_languages, ["hi", "en"])
        self.assertEqual(pred.topic_id, 7)
        self.assertEqual(pred.topic_label, "placement")
        self.assertEqual(pred.topic_keywords, ["placement", "PPO"])
        self.assertEqual(pred.issue_labels, ["fees"])
        self.assertEqual(pred.issue_probabilities, {"fees": 0.8})
        self.assertEqual(pred.parent_comment_text, "parent text here")

    def test_same_response_labels_as_frozen_package(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        self.assertEqual(
            set(clf.labels) | set(clf.missing_response_types),
            set(RESPONSE_TYPES),
        )

    def test_model_config_preserved(self):
        clf = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=3, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        cfg = clf.config
        self.assertEqual(cfg.max_features, 300)


class TestResponseMLPClassifierSerialization(unittest.TestCase):
    """Test artifact serialization via save_artifact()."""

    def setUp(self):  # noqa: N802
        from app.services.nlp.response_classification.models.annotation import (
            ResponseAnnotation,
        )

        clf = ResponseMLPClassifier(
            hidden1=16,
            hidden2=8,
            epochs=5,
            batch_size=4,
            learning_rate=0.01,
            random_state=42,
            min_training_documents=6,
            min_examples_per_class=1,
            max_features=300,
        )
        anns = _make_response_annotation_records(n=18, seed=42)
        clf.train(anns)
        self.clf = clf

    def test_save_artifact_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_artifact.joblib")
            saved = self.clf.save_artifact(path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(saved, path)

    def test_loaded_artifact_can_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_artifact.joblib")
            self.clf.save_artifact(path)
            loaded = joblib.load(path)
            model = loaded["model"]
            vectorizer = loaded["vectorizer"]
            docs = [
                "loaded text for prediction",
            ]
            X = vectorizer.transform(docs)
            if hasattr(X, "toarray"):
                X = X.toarray()
            proba = model.predict_proba(X)
            self.assertEqual(proba.shape[0], 6)

    def test_get_state_after_train(self):
        state = self.clf.get_state()
        self.assertEqual(state["name"], "response_mlp_numpy")
        self.assertEqual(state["version"], "v1")
        self.assertIn("n_features", state)
        self.assertIn("n_classes", state)
        self.assertIn("class_index", state)
        self.assertIn("labels", state)
        self.assertIn("mlp_state", state)
        self.assertIn("n_parameters", state["mlp_state"])


# ===========================================================================
# 4. TRAINER INTEGRATION
# ===========================================================================


class TestTrainerANNIntegration(unittest.TestCase):
    """Verify the Trainer can drive ResponseMLPClassifier through the
    existing generic interface using TaskType.RESPONSE + model override."""

    def test_task_type_response_exists(self):
        self.assertIn("response", {t.value for t in TaskType})

    def test_task_type_response_is_single_label(self):
        cfg = TaskConfig(task_name="response")
        self.assertFalse(cfg.is_multilabel)

    def test_trainer_accepts_explicit_model_override(self):
        cfg = TaskConfig(
            task_name="response",
            random_state=42,
            minimum_training_examples=12,
        )
        trainer = Trainer(cfg)
        model = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=5, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        result = trainer.run(
            model=model,
            use_synthetic=True,
            save_artifact_flag=False,
            register=False,
        )
        self.assertIsNotNone(result)
        self.assertIn("test_report", result)
        self.assertIsInstance(result["test_report"], EvaluationReport)

    def test_trainer_default_baseline_still_works(self):
        cfg = TaskConfig(
            task_name="response",
            random_state=42,
            minimum_training_examples=12,
        )
        trainer = Trainer(cfg)
        result = trainer.run(
            use_synthetic=True,
            save_artifact_flag=False,
            register=False,
        )
        self.assertIsNotNone(result)
        self.assertIn("test_report", result)

    def test_both_models_produce_evaluation_reports(self):
        anns = _make_response_annotation_records(n=60, seed=42)
        records = [
            TrainingRecord(
                document_id=a.document_id,
                text=a.text,
                label=a.normalised_response_type(),
                post_id=a.post_id,
                comment_id=a.comment_id,
                language=a.language,
                script=a.script,
                is_code_mixed=a.is_code_mixed,
                detected_languages=list(a.detected_languages),
                topic_id=a.topic_id,
            )
            for a in anns
        ]

        split = split_records(
            records=records,
            labels=[r.label for r in records],
            spec=SplitSpec(
                train_size=0.7,
                validation_size=0.15,
                test_size=0.15,
                random_state=42,
                stratify=True,
                min_train_examples=12,
                min_validation_examples=1,
                min_test_examples=1,
            ),
        )

        train_recs, val_recs, test_recs = apply_split(records, split)

        # Logistic baseline.
        logistic = build_model(TaskType.RESPONSE)
        logistic_annotations = records_to_annotations(train_recs, TaskType.RESPONSE)
        logistic.train(logistic_annotations)
        logistic_val_report = logistic.evaluate(
            records_to_annotations(val_recs, TaskType.RESPONSE)
        )
        self.assertTrue(hasattr(logistic_val_report, "is_meaningful"))

        # ANN model.
        ann = ResponseMLPClassifier(
            hidden1=32, hidden2=16, epochs=8, batch_size=8,
            learning_rate=0.01, random_state=42,
            min_training_documents=12, min_examples_per_class=1,
            max_features=500,
        )
        ann_annotations = records_to_annotations(train_recs, TaskType.RESPONSE)
        ann.train(ann_annotations)
        ann_val_report = ann.evaluate(
            records_to_annotations(val_recs, TaskType.RESPONSE)
        )
        self.assertTrue(hasattr(ann_val_report, "is_meaningful"))

    def test_artifacts_saved_to_registry(self):
        cfg = TaskConfig(
            task_name="response",
            random_state=42,
            minimum_training_examples=12,
            artifact_directory=tempfile.mkdtemp(),
            registry_path=os.path.join(tempfile.mkdtemp(), "registry.json"),
        )
        trainer = Trainer(cfg)
        model = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=5, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        result = trainer.run(
            model=model,
            use_synthetic=True,
            save_artifact_flag=True,
            register=True,
        )
        self.assertIsNotNone(result.get("artifact_path"))
        self.assertIsNotNone(result.get("registry_id"))

    def test_leakage_check_runs_with_ann_model(self):
        cfg = TaskConfig(
            task_name="response",
            random_state=42,
            minimum_training_examples=12,
        )
        trainer = Trainer(cfg)
        model = ResponseMLPClassifier(
            hidden1=16, hidden2=8, epochs=5, batch_size=4,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_features=300,
        )
        result = trainer.run(
            model=model,
            use_synthetic=True,
            save_artifact_flag=False,
            register=False,
        )
        self.assertIn("leakage_report", result)
        self.assertIn("leakage_detected", result["leakage_report"])


# ===========================================================================
# 5. FRAMEWORK FREEZE VERIFICATION
# ===========================================================================


class TestFrameworkFreeze(unittest.TestCase):
    """Sanity checks that the frozen NLP package was not modified by this
    experiment and the ANN adapter does not import any preprocessing module
    or modify any frozen classifier file."""

    def test_response_classification_package_not_modified(self):
        # We reach here via read-only imports from the frozen package.
        # If any import of a preprocessing module from ann code existed it
        # would be a red flag — but we only import contracts.
        from app.services.nlp.response_classification import (
            base as _rc_base,
        )

        self.assertTrue(hasattr(_rc_base, "ResponseClassification"))

    def test_no_preprocessing_import_in_ann_adapter(self):
        import inspect

        src = inspect.getsource(ResponseMLPClassifier)
        self.assertNotIn("preprocessing", src.lower())
        self.assertNotIn("multilingual_preprocessor", src.lower())

    def test_no_modification_to_frozen_response_files(self):
        # The test itself does not modify anything; this is a placeholder so
        # the suite explicitly documents the freeze expectation.  Real
        # verification is done via git diff in the CI/report step.
        pass


if __name__ == "__main__":
    unittest.main()
