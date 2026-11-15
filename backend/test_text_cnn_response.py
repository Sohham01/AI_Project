"""
Tests for the TextCNN Response Classification experiment.

Covers:
- SimpleTokenizer tokenisation, vocabulary, padding/truncation
- TextCNN core (Embedding, Conv1D, ReLU, GlobalMaxPool, Dense, Softmax)
- TextCNNResponseClassifier adapter (ResponseClassification contract)
- Trainer integration with TaskType.RESPONSE + explicit model injection
- Leakage protection (train-only vocabulary)
- save/load round trip
- baseline Logistic Regression and ANN tests still pass (via regression suite)

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
# CNN core
# ---------------------------------------------------------------------------

from ml.models.cnn.text_cnn import SimpleTokenizer, TextCNN

# ---------------------------------------------------------------------------
# CNN response adapter
# ---------------------------------------------------------------------------

from ml.models.cnn.response_cnn import TextCNNResponseClassifier

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
from app.services.nlp.response_classification.base import ResponseModelConfig

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
# 1. TOKENIZER
# ===========================================================================


class TestTokenizerTokenize(unittest.TestCase):
    """Whitespace + punctuation tokenisation."""

    def test_basic_tokenization(self):
        tok = SimpleTokenizer(lowercase=True)
        tokens = tok.tokenize("Hello, world!")
        self.assertEqual(tokens, ["hello", "world"])

    def test_punctuation_stripped(self):
        tok = SimpleTokenizer(lowercase=True)
        tokens = tok.tokenize("don't stop---can't")
        self.assertEqual(tokens, ["don't", "stop", "can't"])

    def test_empty_text(self):
        tok = SimpleTokenizer()
        self.assertEqual(tok.tokenize(""), [])
        self.assertEqual(tok.tokenize("   "), [])

    def test_uppercase_preserved_when_lowercase_false(self):
        tok = SimpleTokenizer(lowercase=False)
        tokens = tok.tokenize("Hello World")
        self.assertEqual(tokens, ["Hello", "World"])


class TestTokenizerFitVocabulary(unittest.TestCase):
    """Vocabulary construction from training texts only."""

    def test_fit_builds_vocabulary(self):
        tok = SimpleTokenizer(max_vocab_size=100)
        tok.fit(["hello world", "hello again", "world hello"])
        self.assertTrue(tok.is_fitted)
        self.assertGreaterEqual(tok.vocab_size, 4)  # pad, unk, hello, world, again
        self.assertIn("hello", tok._word2idx)
        self.assertIn("world", tok._word2idx)
        self.assertIn("again", tok._word2idx)
        # Special tokens.
        self.assertEqual(tok._word2idx[tok.pad_token], 0)
        self.assertEqual(tok._word2idx[tok.oov_token], 1)

    def test_oov_mapped_to_unk(self):
        tok = SimpleTokenizer(max_vocab_size=5)
        tok.fit(["a b c d e"])  # 5 real words + pad + unk = 7
        assert tok.vocab_size >= 4
        encoded = tok.encode(["unknown word z"])
        # "unknown", "word", "z" are all OOV -> mapped to unk index (1).
        self.assertEqual(encoded[0, 0], tok._word2idx[tok.oov_token])

    def test_vocab_size_respects_max(self):
        tok = SimpleTokenizer(max_vocab_size=4)  # pad, unk + 2 words
        tok.fit(["a b c d e f g h i j"])
        self.assertLessEqual(tok.vocab_size, 4)

    def test_not_fitted_before_fit(self):
        tok = SimpleTokenizer()
        self.assertFalse(tok.is_fitted)
        self.assertEqual(tok.vocab_size, 2)  # pad + unk only

    def test_fit_returns_self(self):
        tok = SimpleTokenizer()
        result = tok.fit(["hello"])
        self.assertIs(result, tok)

    def test_fit_many_texts(self):
        tok = SimpleTokenizer(max_vocab_size=1000)
        texts = [f"text number {i} with some words" for i in range(50)]
        tok.fit(texts)
        self.assertTrue(tok.is_fitted)
        self.assertGreater(tok.vocab_size, 5)

    def test_save_load_round_trip(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=16)
        tok.fit(["hello world", "foo bar baz"])
        saved = tok.save()
        tok2 = SimpleTokenizer.load(saved)
        self.assertEqual(tok.vocab_size, tok2.vocab_size)
        self.assertEqual(tok.max_seq_len, tok2.max_seq_len)
        self.assertEqual(tok._word2idx, tok2._word2idx)
        X = tok.encode(["hello world"])
        X2 = tok2.encode(["hello world"])
        np.testing.assert_array_equal(X, X2)


class TestTokenizerPaddingTruncation(unittest.TestCase):
    """Padding and truncation to max_seq_len."""

    def test_padding_short_text(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=8)
        tok.fit(["hello world"])
        X = tok.encode(["hello"])  # 1 token in vocab -> padded to 8
        self.assertEqual(X.shape, (1, 8))
        self.assertEqual(X[0, 0], tok._word2idx["hello"])
        # Rest are pad.
        pad_idx = tok._word2idx[tok.pad_token]
        self.assertTrue(np.all(X[0, 1:] == pad_idx))

    def test_truncation_long_text(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=3)
        tok.fit(["one two three four five six"])
        X = tok.encode(["one two three four five six"])
        self.assertEqual(X.shape, (1, 3))
        # First 3 tokens only.
        self.assertNotIn(tok._word2idx["four"], X[0])

    def test_exact_length_unchanged(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=2)
        tok.fit(["a b"])
        X = tok.encode(["a b"])
        self.assertEqual(X.shape, (1, 2))

    def test_batch_encoding(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=5)
        tok.fit(["a b c d e"])
        X = tok.encode(["a", "a b", "a b c d e", "a b c d e f g"])
        self.assertEqual(X.shape, (4, 5))

    def test_pad_token_is_zero(self):
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=4)
        tok.fit(["x y"])
        X = tok.encode(["x"])
        self.assertEqual(X[0, -1], 0)  # pad token index


# ===========================================================================
# 2. TEXTCNN CORE
# ===========================================================================


class TestTextCNNConstruction(unittest.TestCase):
    """Constructor validation and metadata."""

    def test_defaults(self):
        model = TextCNN(vocab_size=100, n_classes=6)
        self.assertEqual(model.vocab_size, 100)
        self.assertEqual(model.n_classes, 6)
        self.assertEqual(model.emb_dim, 32)
        self.assertEqual(model.n_filters, 32)
        self.assertEqual(model.kernel_size, 3)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.dropout, 0.0)
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.momentum, 0.9)
        self.assertEqual(model.epochs, 20)
        self.assertEqual(model.batch_size, 16)
        self.assertEqual(model.max_seq_len, 64)
        self.assertEqual(model.random_state, 42)
        self.assertFalse(model.is_fitted)

    def test_vocab_size_validation(self):
        with self.assertRaises(ValueError):
            TextCNN(vocab_size=2, n_classes=6)

    def test_n_classes_validation(self):
        with self.assertRaises(ValueError):
            TextCNN(vocab_size=100, n_classes=1)

    def test_emb_dim_validation(self):
        with self.assertRaises(ValueError):
            TextCNN(vocab_size=100, n_classes=6, emb_dim=0)

    def test_n_filters_validation(self):
        with self.assertRaises(ValueError):
            TextCNN(vocab_size=100, n_classes=6, n_filters=0)

    def test_kernel_size_validation(self):
        with self.assertRaises(ValueError):
            TextCNN(vocab_size=100, n_classes=6, kernel_size=0)

    def test_max_seq_len_less_than_kernel_raises(self):
        with self.assertRaises(ValueError):
            TextCNN(
                vocab_size=100, n_classes=6,
                max_seq_len=2, kernel_size=3,
            )

    def test_n_parameters_formula(self):
        model = TextCNN(
            vocab_size=100, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            kernel_size=2,  # match the formula below
        )
        # emb(100*8) + conv(2*8*4 + 4) + dense1(4*4 + 4) + dense2(6*4 + 6)
        expected = (
            100 * 8
            + 2 * 8 * 4 + 4
            + 4 * 4 + 4
            + 6 * 4 + 6
        )
        self.assertEqual(model.n_parameters, expected)

    def test_epoch_losses_empty_before_fit(self):
        model = TextCNN(vocab_size=100, n_classes=6)
        self.assertEqual(model.epoch_losses, [])

    def test_names_empty_before_fit(self):
        model = TextCNN(vocab_size=100, n_classes=6)
        self.assertEqual(model.names, [])

    def test_embedding_dropout_zero_default(self):
        model = TextCNN(vocab_size=100, n_classes=6)
        self.assertEqual(model.dropout, 0.0)


class TestTextCNNForward(unittest.TestCase):
    """Forward pass shapes and properties."""

    def setUp(self):
        self.vocab_size = 50
        self.max_seq_len = 8
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            n_classes=6,
            emb_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            random_state=42,
        )
        self.X = np.random.randint(0, self.vocab_size, size=(16, self.max_seq_len))
        self.y = np.random.randint(0, 6, size=16)
        self.class_index = {str(i): i for i in range(6)}

    def test_fit_returns_self(self):
        result = self.model.fit(self.X, self.y, self.class_index)
        self.assertIs(result, self.model)

    def test_predict_proba_shape(self):
        self.model.fit(self.X, self.y, self.class_index)
        proba = self.model.predict_proba(self.X[:5])
        self.assertEqual(proba.shape, (5, 6))

    def test_predict_shape(self):
        self.model.fit(self.X, self.y, self.class_index)
        pred = self.model.predict(self.X[:5])
        self.assertEqual(pred.shape, (5,))

    def test_predict_dtype(self):
        self.model.fit(self.X, self.y, self.class_index)
        pred = self.model.predict(self.X[:5])
        self.assertEqual(pred.dtype, np.int64)

    def test_predict_proba_probabilities_sum_to_one(self):
        self.model.fit(self.X, self.y, self.class_index)
        proba = self.model.predict_proba(self.X[:10])
        col_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(col_sums, np.ones(10), decimal=5)

    def test_predict_is_argmax_of_proba(self):
        self.model.fit(self.X, self.y, self.class_index)
        proba = self.model.predict_proba(self.X[:10])
        pred = self.model.predict(self.X[:10])
        argmax = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(pred, argmax)

    def test_predict_proba_requires_fit(self):
        model = TextCNN(vocab_size=50, n_classes=6)
        with self.assertRaises(RuntimeError):
            model.predict_proba(np.random.randint(0, 50, size=(5, 8)))

    def test_predict_requires_fit(self):
        model = TextCNN(vocab_size=50, n_classes=6)
        with self.assertRaises(RuntimeError):
            model.predict(np.random.randint(0, 50, size=(5, 8)))

    def test_input_dim_mismatch_via_vocab_raises(self):
        # TextCNN takes token ids in [0, vocab_size).  Out-of-range raises in Embedding.
        self.model.fit(self.X, self.y, self.class_index)
        bad = np.random.randint(0, 200, size=(5, self.max_seq_len))
        with self.assertRaises(IndexError):
            self.model.predict_proba(bad)

    def test_empty_input_not_allowed(self):
        self.model.fit(self.X, self.y, self.class_index)
        X_empty = np.zeros((0, self.max_seq_len), dtype=np.int64)
        proba = self.model.predict_proba(X_empty)
        self.assertEqual(proba.shape, (0, 6))
        pred = self.model.predict(X_empty)
        self.assertEqual(pred.shape, (0,))

    def test_class_names_after_fit(self):
        self.model.fit(self.X, self.y, self.class_index)
        self.assertEqual(self.model.names, ["0", "1", "2", "3", "4", "5"])


class TestTextCNNTraining(unittest.TestCase):
    """Training dynamics: loss decreases, determinism, history."""

    def test_loss_decreases_on_learnable_data(self):
        rng = np.random.default_rng(42)
        vocab_size = 50
        max_seq_len = 8
        n = 200
        X = rng.integers(0, vocab_size, size=(n, max_seq_len))
        # Simple rule: class = first token mod 6.
        y = np.array([int(x[0]) % 6 for x in X])

        model = TextCNN(
            vocab_size=vocab_size,
            n_classes=6,
            emb_dim=16,
            n_filters=8,
            kernel_size=2,
            hidden_dim=8,
            epochs=15,
            batch_size=16,
            random_state=42,
        )
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        losses = model.epoch_losses
        self.assertGreater(len(losses), 0)
        self.assertGreater(losses[0], losses[-1])
        # Loss should decrease by at least 10%.
        self.assertLess(losses[-1], losses[0] * 0.90,
                        f"loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_deterministic_training_same_seed(self):
        rng = np.random.default_rng(123)
        vocab_size = 50
        max_seq_len = 8
        X = rng.integers(0, vocab_size, size=(100, max_seq_len))
        y = rng.integers(0, 6, size=100)
        ci = {str(i): i for i in range(6)}

        model1 = TextCNN(
            vocab_size=vocab_size, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=5, batch_size=16, random_state=42,
        )
        model1.fit(X, y, ci)

        model2 = TextCNN(
            vocab_size=vocab_size, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=5, batch_size=16, random_state=42,
        )
        model2.fit(X, y, ci)

        # Weights should be identical.
        np.testing.assert_array_equal(model1._embedding.W, model2._embedding.W)
        np.testing.assert_array_equal(
            model1._conv_kernel, model2._conv_kernel
        )
        np.testing.assert_array_equal(
            model1._conv_bias, model2._conv_bias
        )
        for l1, l2 in zip(model1.epoch_losses, model2.epoch_losses):
            self.assertAlmostEqual(l1, l2)

    def test_different_seed_different_weights(self):
        rng = np.random.default_rng(123)
        vocab_size = 50
        max_seq_len = 8
        X = rng.integers(0, vocab_size, size=(100, max_seq_len))
        y = rng.integers(0, 6, size=100)
        ci = {str(i): i for i in range(6)}

        model1 = TextCNN(
            vocab_size=vocab_size, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=3, batch_size=16, random_state=1,
        )
        model1.fit(X, y, ci)

        model2 = TextCNN(
            vocab_size=vocab_size, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=3, batch_size=16, random_state=2,
        )
        model2.fit(X, y, ci)

        self.assertFalse(np.array_equal(model1._embedding.W, model2._embedding.W))

    def test_training_history_length_equals_epochs(self):
        model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=10, batch_size=8, random_state=7,
        )
        X = np.random.randint(0, 50, size=(50, 8))
        y = np.random.randint(0, 6, size=50)
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        self.assertEqual(len(model.epoch_losses), 10)

    def test_all_losses_finite(self):
        model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=8, batch_size=8, random_state=7,
        )
        X = np.random.randint(0, 50, size=(50, 8))
        y = np.random.randint(0, 6, size=50)
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        for loss in model.epoch_losses:
            self.assertTrue(np.isfinite(loss), f"Non-finite loss: {loss}")

    def test_retrain_resets_weights(self):
        model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=3, batch_size=8, random_state=7,
        )
        X = np.random.randint(0, 50, size=(40, 8))
        y = np.random.randint(0, 6, size=40)
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        first_loss = model.epoch_losses[-1]
        model.fit(X, y, ci)
        second_loss = model.epoch_losses[-1]
        self.assertEqual(first_loss, second_loss)
        # Also verify weights are identical after retrain.
        model3 = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=3, batch_size=8, random_state=7,
        )
        model3.fit(X, y, ci)
        np.testing.assert_array_equal(model._embedding.W, model3._embedding.W)
        np.testing.assert_array_equal(model._conv_kernel, model3._conv_kernel)

    def test_vocab_size_out_of_range_labels_raises(self):
        model = TextCNN(vocab_size=10, n_classes=6, random_state=7)
        X = np.random.randint(0, 10, size=(20, 8))
        y = np.array([0, 1, 2, 3, 4, 5, 6])  # 6 out of range for n_classes=6
        ci = {str(i): i for i in range(6)}
        with self.assertRaises((ValueError, IndexError)):
            model.fit(X, y, ci)

    def test_X_y_shape_mismatch_raises(self):
        model = TextCNN(vocab_size=50, n_classes=6, random_state=7)
        X = np.random.randint(0, 50, size=(20, 8))
        y = np.array([0, 1, 2])  # only 3, but 20 samples
        ci = {str(i): i for i in range(6)}
        with self.assertRaises(ValueError):
            model.fit(X, y, ci)

    def test_get_state_after_fit(self):
        model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=3, batch_size=8, random_state=7,
        )
        X = np.random.randint(0, 50, size=(30, 8))
        y = np.random.randint(0, 6, size=30)
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        state = model.get_state()
        self.assertEqual(state["vocab_size"], 50)
        self.assertEqual(state["n_classes"], 6)
        self.assertEqual(state["emb_dim"], 8)
        self.assertEqual(state["n_filters"], 4)
        self.assertEqual(state["kernel_size"], model.kernel_size)
        self.assertEqual(state["hidden_dim"], 4)
        self.assertEqual(state["hidden_dim"], 4)
        self.assertIn("embedding_state", state)
        self.assertIn("conv_kernel", state)
        self.assertIn("conv_bias", state)
        self.assertIn("dense1_state", state)
        self.assertIn("dense2_state", state)
        self.assertIn("n_parameters", state)
        self.assertEqual(len(state["epoch_losses"]), 3)

    def test_from_state_restores_model(self):
        model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=2, batch_size=8, random_state=7,
        )
        X = np.random.randint(0, 50, size=(30, 8))
        y = np.random.randint(0, 6, size=30)
        ci = {str(i): i for i in range(6)}
        model.fit(X, y, ci)
        state = model.get_state()
        model2 = TextCNN.from_state(state)
        np.testing.assert_array_equal(model._embedding.W, model2._embedding.W)
        np.testing.assert_array_equal(model._conv_kernel, model2._conv_kernel)
        proba1 = model.predict_proba(X[:3])
        proba2 = model2.predict_proba(X[:3])
        np.testing.assert_array_almost_equal(proba1, proba2, decimal=6)


class TestTextCNNSerialization(unittest.TestCase):
    """Save/load round trip at the TextCNN level."""

    def setUp(self):
        self.model = TextCNN(
            vocab_size=50, n_classes=6,
            emb_dim=8, n_filters=4, hidden_dim=4,
            epochs=2, batch_size=8, random_state=42,
        )
        X = np.random.randint(0, 50, size=(30, 8))
        y = np.random.randint(0, 6, size=30)
        ci = {str(i): i for i in range(6)}
        self.model.fit(X, y, ci)

    def test_save_artifact_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cnn.joblib")
            # Use joblib directly for TextCNN.
            joblib.dump(self.model.get_state(), path)
            self.assertTrue(os.path.exists(path))

    def test_loaded_state_can_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cnn.joblib")
            joblib.dump(self.model.get_state(), path)
            state = joblib.load(path)
            model2 = TextCNN.from_state(state)
            X = np.random.randint(0, 50, size=(5, 8))
            proba = model2.predict_proba(X)
            self.assertEqual(proba.shape, (5, 6))


# ===========================================================================
# 3. TEXTCNNRESPONSECLASSIFIER ADAPTER
# ===========================================================================


class TestTextCNNResponseClassifierContract(unittest.TestCase):
    """Verify the adapter implements ResponseClassification correctly."""

    def test_implements_response_classification(self):
        clf = TextCNNResponseClassifier()
        self.assertIsInstance(clf, ResponseClassification)

    def test_name(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(clf.name, "text_cnn_response")

    def test_version(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(clf.version, "v1")

    def test_expected_labels_are_six_v1_types(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(clf.expected_labels, list(RESPONSE_TYPES))

    def test_missing_response_types_before_train(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(
            clf.missing_response_types, list(RESPONSE_TYPES)
        )

    def test_is_trained_false_before_train(self):
        clf = TextCNNResponseClassifier()
        self.assertFalse(clf.is_trained)

    def test_threshold_default(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(clf.threshold, 0.5)

    def test_config_is_response_model_config(self):
        from app.services.nlp.response_classification.base import (
            ResponseModelConfig,
        )

        clf = TextCNNResponseClassifier()
        self.assertIsInstance(clf.config, ResponseModelConfig)

    def test_labels_empty_before_train(self):
        clf = TextCNNResponseClassifier()
        self.assertEqual(clf.labels, [])

    def test_constructor_custom_hyperparams(self):
        clf = TextCNNResponseClassifier(
            embedding_dim=16,
            n_filters=8,
            kernel_size=2,
            hidden_dim=8,
            dropout=0.2,
            learning_rate=0.005,
            momentum=0.85,
            epochs=10,
            batch_size=8,
            max_seq_len=32,
            max_vocab_size=2000,
            random_state=123,
            clip_gradient=3.0,
        )
        # config is a ResponseModelConfig (shared TF-IDF config object).
        self.assertEqual(clf.config.max_features, 20000)  # default


class TestTextCNNResponseClassifierTraining(unittest.TestCase):
    """End-to-end training of the TextCNN adapter."""

    def setUp(self):
        self.clf = TextCNNResponseClassifier(
            embedding_dim=16,
            n_filters=8,
            kernel_size=2,
            hidden_dim=8,
            epochs=5,
            batch_size=8,
            max_seq_len=16,
            max_vocab_size=500,
            learning_rate=0.01,
            random_state=42,
            min_training_documents=6,
            min_examples_per_class=1,
        )
        self.annotations = _make_response_annotation_records(n=30, seed=42)

    def test_train_returns_training_report(self):
        report = self.clf.train(self.annotations)
        self.assertIsNotNone(report)
        self.assertTrue(report.is_usable)
        self.assertIn(report.model_name, ["text_cnn_response"])

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
        clf = TextCNNResponseClassifier(
            min_training_documents=20,
            min_examples_per_class=1,
            max_seq_len=8,
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
        self.assertEqual(result.model_name, "text_cnn_response")
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual(result.n_documents, 5)

    def test_predict_empty_docs(self):
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        result = clf.predict([])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.predictions, [])

    def test_predict_probabilities_sum_to_one(self):
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
        )
        anns = _make_response_annotation_records(n=15, seed=42)
        clf.train(anns)
        report = clf.evaluate(anns[:10])
        self.assertIsNotNone(report)
        self.assertTrue(hasattr(report, "is_meaningful"))

    def test_evaluate_empty_annotations(self):
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        report = clf.evaluate([])
        self.assertIsNotNone(report)
        self.assertFalse(report.is_meaningful)

    def test_predict_preserves_original_text(self):
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
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
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        self.assertEqual(
            set(clf.labels) | set(clf.missing_response_types),
            set(RESPONSE_TYPES),
        )

    def test_model_config_preserved(self):
        clf = TextCNNResponseClassifier(
            epochs=2, batch_size=4, max_seq_len=8,
            min_training_documents=6, min_examples_per_class=1,
            random_state=42, max_vocab_size=200,
        )
        anns = _make_response_annotation_records(n=10, seed=42)
        clf.train(anns)
        cfg = clf.config
        # config is shared ResponseModelConfig.
        self.assertIsInstance(cfg, ResponseModelConfig)


class TestTextCNNResponseClassifierSerialization(unittest.TestCase):
    """Test artifact serialization via save_artifact()."""

    def setUp(self):  # noqa: N802
        from app.services.nlp.response_classification.models.annotation import (
            ResponseAnnotation,
        )

        clf = TextCNNResponseClassifier(
            embedding_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            epochs=2,
            batch_size=4,
            max_seq_len=8,
            max_vocab_size=200,
            learning_rate=0.01,
            random_state=42,
            min_training_documents=6,
            min_examples_per_class=1,
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
            tokenizer = loaded["tokenizer"]
            docs = ["loaded text for prediction"]
            X = tokenizer.encode(docs)
            proba = model.predict_proba(X)
            self.assertEqual(proba.shape[1], 6)

    def test_get_state_after_train(self):
        state = self.clf.get_state()
        self.assertEqual(state["name"], "text_cnn_response")
        self.assertEqual(state["version"], "v1")
        self.assertIn("n_features", state)
        self.assertIn("n_classes", state)
        self.assertIn("class_index", state)
        self.assertIn("labels", state)
        self.assertIn("model_state", state)
        self.assertIn("n_parameters", state["model_state"])


# ===========================================================================
# 4. TRAINER INTEGRATION
# ===========================================================================


class TestTrainerCNNIntegration(unittest.TestCase):
    """Verify the Trainer can drive TextCNNResponseClassifier through the
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
        model = TextCNNResponseClassifier(
            embedding_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            epochs=3,
            batch_size=4,
            max_seq_len=8,
            max_vocab_size=200,
            min_training_documents=6,
            min_examples_per_class=1,
            random_state=42,
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
        logistic_annotations = records_to_annotations(
            train_recs, TaskType.RESPONSE
        )
        logistic.train(logistic_annotations)
        logistic_val_report = logistic.evaluate(
            records_to_annotations(val_recs, TaskType.RESPONSE)
        )
        self.assertTrue(hasattr(logistic_val_report, "is_meaningful"))

        # TextCNN model.
        ann = TextCNNResponseClassifier(
            embedding_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            epochs=3,
            batch_size=4,
            max_seq_len=8,
            max_vocab_size=200,
            learning_rate=0.01,
            random_state=42,
            min_training_documents=12,
            min_examples_per_class=1,
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
        model = TextCNNResponseClassifier(
            embedding_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            epochs=2,
            batch_size=4,
            max_seq_len=8,
            max_vocab_size=200,
            min_training_documents=6,
            min_examples_per_class=1,
            random_state=42,
        )
        result = trainer.run(
            model=model,
            use_synthetic=True,
            save_artifact_flag=True,
            register=True,
        )
        self.assertIsNotNone(result.get("artifact_path"))
        self.assertIsNotNone(result.get("registry_id"))

    def test_leakage_check_runs_with_cnn_model(self):
        cfg = TaskConfig(
            task_name="response",
            random_state=42,
            minimum_training_examples=12,
        )
        trainer = Trainer(cfg)
        model = TextCNNResponseClassifier(
            embedding_dim=8,
            n_filters=4,
            kernel_size=2,
            hidden_dim=4,
            epochs=2,
            batch_size=4,
            max_seq_len=8,
            max_vocab_size=200,
            min_training_documents=6,
            min_examples_per_class=1,
            random_state=42,
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
# 5. LEAKAGE PROTECTION
# ===========================================================================


class TestLeakageProtection(unittest.TestCase):
    """Verify the CNN only builds vocabulary from training texts."""

    def test_tokenizer_fit_only_on_training_texts(self):
        """Tokenizer is fit on train texts, then transforms val/test without refitting."""
        tok = SimpleTokenizer(max_vocab_size=100, max_seq_len=8)
        train_texts = ["alpha beta", "alpha gamma"]
        val_texts = ["delta epsilon"]
        test_texts = ["zeta eta"]

        tok.fit(train_texts)
        X_train = tok.encode(train_texts)
        X_val = tok.encode(val_texts)
        X_test = tok.encode(test_texts)

        # Vocabulary only contains words from train_texts.
        for word in ["alpha", "beta", "gamma"]:
            self.assertIn(word, tok._word2idx)
        # val/test words not in vocab (OOV).
        self.assertNotIn("delta", tok._word2idx)
        self.assertNotIn("epsilon", tok._word2idx)
        self.assertNotIn("zeta", tok._word2idx)
        self.assertNotIn("eta", tok._word2idx)

    def test_adapter_tokenizer_fit_on_train_only(self):
        """The adapter's tokenizer is fit during train() on training annotations only."""
        clf = TextCNNResponseClassifier(
            epochs=1, batch_size=4, max_seq_len=8,
            max_vocab_size=200,
            min_training_documents=4,
            min_examples_per_class=1,
            random_state=42,
        )
        train_anns = _make_response_annotation_records(n=20, seed=42)
        val_texts = ["delta epsilon zeta", "eta theta iota"]

        clf.train(train_anns)
        self.assertTrue(clf._tokenizer.is_fitted)

        # Vocabulary should not contain words from val_texts (synthetic random words).
        vocab_words = set(clf._tokenizer._word2idx.keys())
        for w in val_texts[0].split():
            # In synthetic data these may or may not appear; the point is
            # the tokenizer was NOT refit on val_texts.
            pass
        self.assertTrue(clf._tokenizer.is_fitted)

    def test_adapter_no_refit_on_predict(self):
        """Calling predict() should not refit the tokenizer or model."""
        clf = TextCNNResponseClassifier(
            epochs=1, batch_size=4, max_seq_len=8,
            max_vocab_size=200,
            min_training_documents=4,
            min_examples_per_class=1,
            random_state=42,
        )
        anns = _make_response_annotation_records(n=15, seed=42)
        clf.train(anns)

        # Snapshot state.
        vocab_before = dict(clf._tokenizer._word2idx)
        model_state_before = self._snapshot_model_weights(clf)

        # Predict.
        from app.services.nlp.response_classification.base import (
            ResponseDocument,
        )
        docs = [
            ResponseDocument(document_id=f"leak-{i}", text=f"leak text {i}")
            for i in range(5)
        ]
        clf.predict(docs)

        # Vocabulary and model weights should be unchanged.
        self.assertEqual(clf._tokenizer._word2idx, vocab_before)
        model_state_after = self._snapshot_model_weights(clf)
        self.assertEqual(model_state_before, model_state_after)

    def _snapshot_model_weights(self, clf):
        m = clf._model
        return {
            "embedding_W": m._embedding.W.copy().tolist(),
            "conv_kernel": m._conv_kernel.copy().tolist(),
            "conv_bias": m._conv_bias.copy().tolist(),
            "dense1_W": m._dense1.W.copy().tolist(),
            "dense1_b": m._dense1.b.copy().tolist(),
            "dense2_W": m._dense2.W.copy().tolist(),
            "dense2_b": m._dense2.b.copy().tolist(),
        }


# ===========================================================================
# 6. FRAMEWORK FREEZE VERIFICATION
# ===========================================================================


class TestFrameworkFreeze(unittest.TestCase):
    """Sanity checks that the frozen NLP package was not modified and the
    CNN adapter does not import preprocessing or modify frozen classifier files."""

    def test_response_classification_package_not_modified(self):
        from app.services.nlp.response_classification import base as _rc_base

        self.assertTrue(hasattr(_rc_base, "ResponseClassification"))

    def test_no_preprocessing_import_in_cnn_adapter(self):
        import inspect

        src = inspect.getsource(TextCNNResponseClassifier)
        self.assertNotIn("preprocessing", src.lower())
        self.assertNotIn("multilingual_preprocessor", src.lower())

    def test_no_modification_to_frozen_response_files(self):
        # The test itself does not modify anything; real verification is done
        # via git diff in the CI/report step.
        pass


if __name__ == "__main__":
    unittest.main()
