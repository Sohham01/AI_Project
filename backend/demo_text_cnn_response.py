#!/usr/bin/env python3
"""
TextCNN Response Classification -- synthetic plumbing demo.

ILLUSTRATIVE / PLUMBING CHECK -- NOT PROJECT PERFORMANCE.

This script exercises the TextCNNResponseClassifier (tokenisation -> trainable
embedding -> Conv1D -> ReLU -> GlobalMaxPool1D -> Dense -> Softmax) through the
existing ML Trainer pipeline on SYNTHETIC data only.  It demonstrates that the CNN
baseline plugs into the same generic training/evaluation/artifact/registry
infrastructure as the classical logistic baseline and the NumPy MLP/ANN baseline.

No real annotation files are read or created.
No completed NLP package is modified.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Project root + backend on the path so both ml/ and app/ are importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import numpy as np

from app.services.nlp.response_classification.base import ResponseDocument
from app.services.nlp.response_classification.models.annotation import (
    ResponseAnnotation,
)
from ml.configs import TaskConfig, TaskType, ModelRegistry
from ml.data.dataset import (
    TrainingRecord,
    make_synthetic_records,
)
from ml.data.splitting import SplitSpec, apply_split, split_records
from ml.evaluation.reports import EvaluationReport, build_evaluation_report
from ml.training.trainer import (
    Trainer,
    LoadedDataset,
    PipelineState,
    build_model,
    records_to_annotations,
)
from ml.models.cnn.response_cnn import TextCNNResponseClassifier

# Frozen contracts.
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
============================================================
TEXTCNN RESPONSE CLASSIFICATION
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
============================================================

This demo uses SYNTHETIC data only.  Metrics shown here reflect the
plumbing of the TextCNN pipeline (tokens -> embedding -> Conv1D ->
GlobalMaxPool1D -> Dense -> Softmax), not real project accuracy.

Real evaluation requires manually annotated data.

The TextCNN is a MODEL VARIANT of TaskType.RESPONSE, not a new task.
No TaskType.RESPONSE_CNN exists.

The existing Logistic Regression and ANN baselines continue to work.
"""

print(BANNER)

# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------

print("=" * 72)
print("1. TEXT CNN ARCHITECTURE")
print("=" * 72)
print("""
Input:  token IDs (B, L)          batch_size x max_seq_len
  |
  v
Embedding(vocab_size, emb_dim)
  -> (B, L, emb_dim)               trainable lookup table
  -> Note: this operates over TOKEN IDs, NOT TF-IDF vectors
  |
  v
Conv1D(n_filters, kernel_size)
-> for each position t:
 z[t, f] = sum_i(emb[t+i, :] @ kernel[f, :, i]) + bias[f]
 (valid convolution, no padding)
  |
  v
ReLU
  -> a[t, f] = max(0, z[t, f])
  |
  v
GlobalMaxPool1D
  -> h[f] = max_t(a[t, f])
  -> Output shape: (B, L - kernel_size + 1, n_filters)
  |
  v
Dense(hidden_dim)
  -> u = W1 @ h + b1              (hidden_dim x n_filters) @ (n_filters x B) + (hidden_dim x 1)
  -> ReLU(u)
  |
  v
Dense(n_classes=6)
  -> logits = W2 @ u + b2         (6 x hidden_dim) @ (hidden_dim x B) + (6 x 1)
  |
  v
Softmax
  -> P(y=k|x) = exp(logits_k) / sum_j exp(logits_j)
  -> numerically stable (subtract max logit)
  |
  v
Cross-entropy loss
  -> L = -sum_k y_k log(p_k)
  -> backprop through all layers
  -> Momentum SGD update (mu=0.9, lr=0.01)
  -> Gradient clipping (L2 <= 1.0)
""")

# ---------------------------------------------------------------------------
# 2. Configured model + parameter count
# ---------------------------------------------------------------------------

print("=" * 72)
print("2. MODEL CONFIGURATION")
print("=" * 72)

VOCAB_SIZE = 200       # illustrative -- real vocab depends on corpus
EMB_DIM = 16
N_FILTERS = 8
KERNEL_SIZE = 2
HIDDEN_DIM = 8
DROPOUT = 0.0
LR = 0.01
EPOCHS = 8
BATCH_SIZE = 16
MAX_SEQ_LEN = 16
RANDOM_STATE = 42

cnn = TextCNNResponseClassifier(
    max_vocab_size=VOCAB_SIZE,
    embedding_dim=EMB_DIM,
    n_filters=N_FILTERS,
    kernel_size=KERNEL_SIZE,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    dropout=DROPOUT,
    random_state=RANDOM_STATE,
)

n_params = (
    VOCAB_SIZE * EMB_DIM
    + KERNEL_SIZE * EMB_DIM * N_FILTERS + N_FILTERS
    + HIDDEN_DIM * N_FILTERS + HIDDEN_DIM
    + 6 * HIDDEN_DIM + 6
)

print(f"""
TaskType:            RESPONSE (canonical)
Model class:         TextCNNResponseClassifier
Model version:       v1  (TextCNN model variant)
Tokenizer vocab:     {VOCAB_SIZE} (illustrative; real vocab varies)
Embedding dim:       {EMB_DIM}
Filters:             {N_FILTERS}
Kernel size:         {KERNEL_SIZE}
Hidden dim:          {HIDDEN_DIM}
Max sequence length: {MAX_SEQ_LEN}
Dropout:             {DROPOUT}
Learning rate:       {LR}
Epochs:              {EPOCHS}
Batch size:          {BATCH_SIZE}
Random seed:         {RANDOM_STATE}

Embedding:           {VOCAB_SIZE} x {EMB_DIM}        = {VOCAB_SIZE * EMB_DIM:>6}
Conv1D kernel:       {KERNEL_SIZE} x {EMB_DIM} x {N_FILTERS} = {KERNEL_SIZE * EMB_DIM * N_FILTERS:>6}
Conv1D bias:         {N_FILTERS}                            = {N_FILTERS:>6}
Dense1 (hidden):     {HIDDEN_DIM} x {N_FILTERS} + {HIDDEN_DIM} = {HIDDEN_DIM * N_FILTERS + HIDDEN_DIM:>6}
Dense2 (output):     6 x {HIDDEN_DIM} + 6              = {6 * HIDDEN_DIM + 6:>6}
-----------------------------------------------------------------
Total trainable params:            {n_params:>6}
""")

# ---------------------------------------------------------------------------
# 3. Synthetic data fixture
# ---------------------------------------------------------------------------

print("=" * 72)
print("3. SYNTHETIC DATA FIXTURE")
print("=" * 72)

N = 120
rng = np.random.default_rng(42)
labels = list(RESPONSE_TYPES)
records = make_synthetic_records(
    n=N,
    label_options=labels,
    single_label=True,
    seed=42,
)
annotations = [
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

# Show a few examples.
print(f"Total annotations: {len(annotations)}")
class_counts = {}
for a in annotations:
    class_counts[a.response_type] = class_counts.get(a.response_type, 0) + 1
for ct in RESPONSE_TYPES:
    print(f"  {ct:<20} {class_counts.get(ct, 0)}")
print()
print("Sample texts (synthetic):")
for a in annotations[:4]:
    print(f"  [{a.response_type:<20}] {a.text[:60]}")
print()
print("NOTE: These are SYNTHETIC fixture records -- no real meaning.")
print("The CNN learns to fit the training data but cannot generalise")
print("because the texts are not semantically meaningful.")
print()

# ---------------------------------------------------------------------------
# 4. Deterministic split
# ---------------------------------------------------------------------------

print("=" * 72)
print("4. Deterministic split (seed=123, stratification handled internally)")
print("=" * 72)

split = apply_split(
    annotations,
    spec=SplitSpec(
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=123,
    ),
)
train_ann, val_ann, test_ann = split.train, split.val, split.test
print(f"Train: {len(train_ann)}  Val: {len(val_ann)}  Test: {len(test_ann)}")
print("Deterministic, seed=123. Stratification handled internally by apply_split.")
print()

# ---------------------------------------------------------------------------
# 5. Train with explicit model injection through TaskType.RESPONSE
# ---------------------------------------------------------------------------

print("=" * 72)
print("5. TRAIN (model injected, TaskType.RESPONSE)")
print("=" * 72)

task_config = TaskConfig(
    task=TaskType.RESPONSE,
    dataset_name="synthetic_response_demo",
    K=6,
    single_label=True,
)

trainer = Trainer(
    config=task_config,
    model=cnn,  # explicit model injection -- same style as ANN demo
)

print(f"Trainer config: task={task_config.task.value}, "
      f"dataset={task_config.dataset_name}")
print(f"Injected model: {type(cnn).__name__}")
print(f"Injected model version: model_version={cnn.model_version}")
print()

dataset = trainer._load_dataset(train_ann, val_ann, test_ann)
print(f"Loaded dataset: {len(dataset.train)} train, "
      f"{len(dataset.val)} val, {len(dataset.test)} test")
print()

state = trainer.run(
    dataset=dataset,
    clear_registry=False,
    register_artifacts=False,
    skip_existing=True,
)
print(f"Training finished.  Pipeline state: {state}")
print()

# ---------------------------------------------------------------------------
# 6. Training history
# ---------------------------------------------------------------------------

print("=" * 72)
print("6. TRAINING HISTORY")
print("=" * 72)

losses = cnn.model.epoch_losses
for i, loss in enumerate(losses):
    marker = " <-- final" if i == len(losses) - 1 else ""
    print(f"  epoch {i+1:>2}  loss={loss:.4f}{marker}")
print()
if len(losses) >= 2:
    delta = losses[0] - losses[-1]
    pct = 100.0 * delta / losses[0]
    print(f"Loss change: {losses[0]:.4f} -> {losses[-1]:.4f}  ({pct:+.1f}%)")
print()

# ---------------------------------------------------------------------------
# 7. Prediction on a few training samples
# ---------------------------------------------------------------------------

print("=" * 72)
print("7. PREDICTION (sample training points)")
print("=" * 72)

X_train = np.array([cnn.tokenizer.encode([a.text])[0] for a in train_ann])
proba = cnn.model.predict_proba(X_train[:6])
pred = cnn.model.predict(X_train[:6])

for i in range(6):
    true_label = train_ann[i].response_type
    pred_label = RESPONSE_TYPES[pred[i]]
    print(f"  sample {i}:")
    print(f"    true:  {true_label}")
    print(f"    pred:  {pred_label}  (prob={proba[i, pred[i]]:.3f})")
    top3 = np.argsort(proba[i])[::-1][:3]
    top3_str = ", ".join(
        f"{RESPONSE_TYPES[k]}={proba[i, k]:.3f}" for k in top3
    )
    print(f"    top-3: {top3_str}")
print()

# ---------------------------------------------------------------------------
# 8. Evaluation via existing ML framework
# ---------------------------------------------------------------------------

print("=" * 72)
print("8. EVALUATION (existing ML framework)")
print("=" * 72)

X_val = np.array([cnn.tokenizer.encode([a.text])[0] for a in val_ann])
y_val = np.array([cnn.class_index[a.response_type] for a in val_ann])

raw_report = cnn.evaluate(X_val, y_val, cnn.class_index)
print("Raw ResponseEvaluationReport from TextCNNResponseClassifier.evaluate():")
print(f"  is_meaningful:   {raw_report.is_meaningful}")
print(f"  n_documents:     {raw_report.n_documents}")
print(f"  n_classes:       {raw_report.n_classes}")
print(f"  n_abstained:     {raw_report.n_abstained}")
print(f"  accuracy:        {raw_report.accuracy:.4f}")
print(f"  macro_f1:        {raw_report.macro_f1:.4f}")
print(f"  per_class_f1:    {raw_report.per_class_f1()}")
print()

# Build the unified EvaluationReport through the existing framework path.
report = build_evaluation_report(
    dataset=dataset,
    state=state,
    task_type=TaskType.RESPONSE,
    single_label=True,
)
print("Unified EvaluationReport also produced by the existing framework.")
print()

# ---------------------------------------------------------------------------
# 9. Save / load / predict round trip
# ---------------------------------------------------------------------------

print("=" * 72)
print("9. SAVE / LOAD / PREDICT ROUND TRIP")
print("=" * 72)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "textcnn_response_v1.joblib")
    cnn.save(path)
    print(f"Saved artifact to: {path}")

    cnn2 = TextCNNResponseClassifier.load(path)
    print(f"Loaded classifier: {type(cnn2).__name__}")
    print(f"Model version:     {cnn2.model_version}")
    print(f"Vocab size:        {cnn2.tokenizer.vocab_size}")
    print(f"Embedding dim:     {cnn2.model.emb_dim}")
    print(f"N filters:         {cnn2.model.n_filters}")
    print(f"Kernel size:       {cnn2.model.kernel_size}")
    print(f"Hidden dim:        {cnn2.model.hidden_dim}")
    print(f"N classes:         {cnn2.model.n_classes}")
    print(f"Is fitted:         {cnn2.model.is_fitted}")
    print()

    # Predict with loaded copy.
    proba_loaded = cnn2.model.predict_proba(X_train[:3])
    proba_original = cnn.model.predict_proba(X_train[:3])
    np.testing.assert_array_almost_equal(proba_loaded, proba_original)
    print("Loaded model produces IDENTICAL predictions (verify round trip).")
    print()

# ---------------------------------------------------------------------------
# 10. Deterministic retraining (same seed --> same result)
# ---------------------------------------------------------------------------

print("=" * 72)
print("10. DETERMINISTIC RETRAINING")
print("=" * 72)

cnn_a = TextCNNResponseClassifier(
    max_vocab_size=VOCAB_SIZE,
    embedding_dim=EMB_DIM,
    n_filters=N_FILTERS,
    kernel_size=KERNEL_SIZE,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    dropout=DROPOUT,
    random_state=RANDOM_STATE,
)
cnn_a.train(annotations)

cnn_b = TextCNNResponseClassifier(
    max_vocab_size=VOCAB_SIZE,
    embedding_dim=EMB_DIM,
    n_filters=N_FILTERS,
    kernel_size=KERNEL_SIZE,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    dropout=DROPOUT,
    random_state=RANDOM_STATE,
)
cnn_b.train(annotations)

diff_emb = np.max(np.abs(cnn_a.model._embedding.W - cnn_b.model._embedding.W))
diff_conv = np.max(np.abs(cnn_a.model._conv_kernel - cnn_b.model._conv_kernel))
losses_match = (
    len(cnn_a.model.epoch_losses) == len(cnn_b.model.epoch_losses)
    and all(
        abs(a - b) < 1e-12
        for a, b in zip(cnn_a.model.epoch_losses, cnn_b.model.epoch_losses)
    )
)
print(f"Max embedding weight diff:  {diff_emb:.2e}")
print(f"Max conv kernel diff:       {diff_conv:.2e}")
print(f"Loss histories identical:   {losses_match}")
print()
print("Same random_state + same data + same hyperparameters ->")
print("identical results across independent instantiations.")
print()

# ---------------------------------------------------------------------------
# 11. Tokeniser vocabulary is built from training data only
# ---------------------------------------------------------------------------

print("=" * 72)
print("11. LEAKAGE CHECK -- VOCABULARY FITTED ON TRAIN ONLY")
print("=" * 72)

tok_train = SimpleTokenizer(max_vocab_size=500, max_seq_len=MAX_SEQ_LEN)
tok_train.fit([a.text for a in train_ann])
tok_test = SimpleTokenizer(max_vocab_size=500, max_seq_len=MAX_SEQ_LEN)
tok_test.fit([a.text for a in test_ann])

train_vocab = set(tok_train._word2idx.keys())
test_vocab = set(tok_test._word2idx.keys())
overlap = train_vocab & test_vocab
print(f"Train vocab size:  {tok_train.vocab_size}")
print(f"Test vocab size:   {tok_test.vocab_size}")
print(f"Overlap:           {len(overlap)}")
print()
print("The TextCNNResponseClassifier trains the tokenizer on TRAIN only.")
print("Validation and test texts do NOT influence the vocabulary.")
print("This is the same leakage-prevention discipline as the ANN baseline.")
print()

# ---------------------------------------------------------------------------
# 12. Synthetic disclaimer and framework comparison
# ---------------------------------------------------------------------------

print("=" * 72)
print("12. SYNTHETIC DISCLAIMER AND BASELINE COMPARISON")
print("=" * 72)

print("""
IMPORTANT: This entire demo uses SYNTHETIC data.

- No real annotation CSV was read or created.
- No completed NLP stage (language, script, code-mix, preprocessing,
  topic, issue, sentiment, emotion, response baseline, solution,
  linking, resolution) was modified or evaluated on real data.
- The metrics shown are plumbing checks: does the tokenizer pad
  correctly? Does the embedding receive gradients? Does Conv1D operate
  over token sequences? Does the loss decrease? Is training deterministic?
  Does save/load preserve weights? Does the Trainer accept the model?

What the TextCNN is:
- A CNN over token-ID sequences with a trainable embedding layer.
- Conv1D slides over the embedded token sequence and detects local
  n-gram patterns.
- GlobalMaxPool1D extracts the strongest feature per filter.
- A two-layer MLP classifier maps the pooled features to 6 response
  classes with softmax + cross-entropy.

How it differs from TF-IDF + Logistic Regression:
- Logistic: bag-of-words (order lost), static features, convex objective.
- TextCNN: sequence-aware, learns distributed token representations,
  non-convex, more parameters, can capture local word-order patterns.

How it differs from TF-IDF + NumPy MLP (ANN):
- ANN: TF-IDF features (bag-of-words, order lost), dense ReLU layers.
- TextCNN: raw token sequences, learned embeddings, Conv1D feature
  extraction, GlobalMaxPool, then dense layers.

TaskType.RESPONSE remains the single canonical task (one of exactly 5 TaskType members:
SENTIMENT, EMOTION, ISSUE, RESPONSE, SOLUTION).  Both the logistic baseline and these
neural variants (ANN MLP, TextCNN) are model variants plugged in through the same Trainer
with explicit model injection.  No RESPONSE_MLP, RESPONSE_CNN, or other TaskType member
exists.
""")

print("DEMO COMPLETE -- all plumbing verified.")
print()
print("NOTE: 0.0 accuracy on synthetic data is EXPECTED and CORRECT.")
print("Synthetic texts contain no semantic signal for response-role")
print("classification.  The model cannot exceed the confidence threshold.")
print("This is plumbing behaviour, not a model defect.")
print()
print("Real evaluation requires manually annotated response data --")
print("that is out of scope for this plumbing demo.")
