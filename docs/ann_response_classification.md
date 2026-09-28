# ANN Response Classification — TF-IDF → NumPy MLP

## Status

**ILLUSTRATIVE / PLUMBING CHECK — NOT PROJECT PERFORMANCE.**

The ANN described here is a classical neural baseline plugged into the existing
ML training framework.  It is exercised on synthetic data only.  Real evaluation
requires manually annotated response-classification data.

## Why ANN after the classical baseline

The project already has a working classical baseline for response classification:

- **TF-IDF → TfidfLogisticResponseClassifier** (`multinomial` logistic regression)

That baseline is the primary model for V1.  The ANN is a controlled experiment
to see whether a small feedforward network can learn the same six response
roles from the same TF-IDF input, using the same split, the same evaluator, the
same artifact and registry code.

The ANN does **not** replace the logistic baseline.  It runs alongside it as an
alternative model for the same task.

Future model variants (CNN, multilingual transformer) will plug into the same
interface, so the experiment track is additive, not destructive.

## Task taxonomy

The ANN is a **model**, not a new **task**.

- **Task:** `TaskType.RESPONSE` — single-label response-role classification, six
  V1 types.
- **Model variants:**
  - `TfidfLogisticResponseClassifier` (classical baseline)
  - `ResponseMLPClassifier` (ANN baseline, this document)
  - future: text CNN, multilingual transformer

Training uses the same task type and the same annotation contract; the model is
supplied as an explicit override to the Trainer.

```python
from ml.configs import TaskConfig, TaskType
from ml.training.trainer import Trainer
from ml.models.ann.response_mlp import ResponseMLPClassifier

config = TaskConfig(task_name="response", random_state=42, minimum_training_examples=12)
trainer = Trainer(config)
model = ResponseMLPClassifier(
    hidden1=256, hidden2=64, epochs=40, batch_size=16,
    learning_rate=0.01, random_state=42, max_features=5000,
)
result = trainer.run(model=model, use_synthetic=True)
```

## Architecture

```
TF-IDF (fit on training text only)
  |
  v
Dense(H1 = 256)
  -> z = W1 @ x + b1
  -> ReLU(z) = max(0, z)
  -> Dropout (rate = 0.3, inverted during training)
  |
  v
Dense(H2 = 64)
  -> z = W2 @ h1 + b2
  -> ReLU(z) = max(0, z)
  |
  v
Dense(K = 6)
  -> logits = W3 @ h2 + b3
  -> Softmax:  P(y = k | x) = exp(z_k) / Σ_j exp(z_j)
  -> Cross-entropy loss: L = -Σ_k y_k log(p_k)
  -> Backpropagation + momentum SGD + gradient clipping
```

Default hyper-parameters:

| Parameter        | Default |
|------------------|---------|
| hidden1          | 256     |
| hidden2          | 64      |
| output classes   | 6       |
| activation       | ReLU    |
| dropout rate     | 0.3     |
| optimizer        | momentum SGD (mu = 0.9) |
| learning rate    | 0.01    |
| epochs           | 40      |
| batch size       | 16      |
| gradient clip    | L2 <= 5.0 |
| random seed      | 42      |

The six output classes are the canonical V1 response types:

- agreement
- personal_experience
- opinion
- advice
- solution
- counterargument (empty in synthetic tests until real data covers it)

## Dense layer

A dense (fully connected) layer maps an input vector `x` to an output vector `z`:

```
z = Wx + b
```

- `W` is a weight matrix of shape `(out_dim, in_dim)`.
- `b` is a bias vector of shape `(out_dim,)`.
- For a batch of `B` examples the same equation applies column-wise: `z = W @ X + b`.

In this model the three dense layers are:

1. `W1 (256 x D)`, `b1 (256)` — input to first hidden layer.
2. `W2 (64 x 256)`, `b2 (64)` — first hidden to second hidden layer.
3. `W3 (6 x 64)`, `b3 (6)` — second hidden to output logits.

The total number of trainable parameters is:

```
W1 + b1 + W2 + b2 + W3 + b3
= 256*D + 256 + 64*256 + 64 + 6*64 + 6
= 256*D + 16384 + 384 + 6 + 256
```

where `D` is the number of TF-IDF features.

## ReLU

The Rectified Linear Unit is applied after each hidden dense layer:

```
ReLU(z) = max(0, z)
```

Its derivative (used during back-propagation) is:

```
d/dx ReLU(x) = 1  if x > 0
               0  otherwise
```

ReLU is simple, piecewise linear, and avoids saturation for positive inputs.  It
is the default hidden activation for this baseline.

## Dropout

During training, dropout randomly zeroes a fraction of the hidden units and
scales the survivors so the expected activation magnitude is preserved.  The
implementation uses **inverted dropout**: at training time each unit is kept with
probability `keep = 1 - rate` and scaled by `1 / keep`; at inference time no
units are dropped and no scaling is applied.

Dropout here is:

- applied to the first hidden layer only (`H1`), not the second;
- disabled during inference (`predict`, `evaluate`);
- deterministic given the random seed, so repeated training with the same seed
  produces the same mask sequence.

## Softmax

The output logits are converted to a probability distribution with the numerically
stable softmax:

```
P(y = k | x) = exp(z_k) / Σ_j exp(z_j)
```

To avoid overflow for large logits, the implementation subtracts the per-sample
maximum before exponentiating:

```
z' = z - max(z)
P_k = exp(z'_k) / Σ_j exp(z'_j)
```

This preserves the output because softmax is invariant under a constant shift of
all logits for the same sample.

## Cross-entropy loss

For single-label multiclass classification with one-hot targets the loss is the
average cross-entropy over the batch:

```
L = -(1/B) * Σ_n Σ_k y_{nk} * log(p_{nk})
```

- `y_{nk}` is 1 if example `n` belongs to class `k`, else 0.
- `p_{nk}` is the predicted probability for class `k` on example `n`.
- `B` is the batch size.

A small epsilon (`1e-9`) is clamped to the probabilities to avoid `log(0)`.

For a single example with the correct class `c`:

```
L = -log(P(y = c | x))
```

This is the standard multi-class cross-entropy used by both the logistic baseline
and the ANN.

## Forward propagation

For a batch of `B` examples with TF-IDF input `X` of shape `(D, B)`:

1. **First hidden layer:**
   ```
   Z1 = W1 @ X + b1            # (H1, B)
   H1 = ReLU(Z1)               # (H1, B)
   H1d = H1 * mask             # inverted dropout
   ```

2. **Second hidden layer:**
   ```
   Z2 = W2 @ H1d + b2          # (H2, B)
   H2 = ReLU(Z2)               # (H2, B)
   ```

3. **Output layer:**
   ```
   logits = W3 @ H2 + b3       # (K, B)
   probs  = softmax(logits)     # (K, B), each column sums to 1
   ```

The loss is then computed from `probs` and the one-hot labels `Y` of shape
`(K, B)`.

## Back-propagation

The gradients are computed by the chain rule starting from the loss.

1. **Output layer gradient:**
   The derivative of cross-entropy with softmax simplifies to:
   ```
   d_logits = (probs - Y) / B     # (K, B)
   ```

2. **Third weight/bias:**
   ```
   dW3 = d_logits @ H2.T          # (K, H2)
   db3 = Σ_cols(d_logits)         # (K, 1)
   ```

3. **Second hidden layer gradient:**
   ```
   d_H2 = W3.T @ d_logits         # (H2, B)
   d_Z2 = d_H2 * (Z2 > 0)        # ReLU derivative
   dW2 = d_Z2 @ H1d.T             # (H2, H1)
   db2 = Σ_cols(d_Z2)             # (H2, 1)
   ```

4. **First hidden layer gradient (through dropout):**
   ```
   d_H1d = W2.T @ d_Z2           # (H1, B)
   d_H1  = d_H1d * mask           # undo dropout scale
   d_Z1  = d_H1 * (Z1 > 0)       # ReLU derivative
   dW1  = d_Z1 @ X.T             # (H1, D)
   db1  = Σ_cols(d_Z1)           # (H1, 1)
   ```

All gradient arrays have the same shape as their corresponding parameters.

## Gradient clipping

Before the parameter update, each gradient array is checked against a global L2
norm threshold.  If the combined norm of all gradients exceeds the threshold, all
gradients are scaled by `threshold / total_norm`.

This prevents a single large gradient from dominating the update and is a simple
form of stabilisation for SGD on TF-IDF features.

## Gradient descent / parameter update

Parameters are updated with momentum SGD:

```
v = mu * v - lr * g
p = p + v
```

- `mu` is the momentum coefficient (default 0.9).
- `lr` is the learning rate (default 0.01).
- `g` is the clipped gradient.
- `v` is the velocity (one per parameter, maintained across batches).

Momentum smooths the trajectory and often converges faster than plain SGD on
this kind of data.  The weights are initialised once per training run and the
velocity is reset at the start of each `fit()` call.

## Initialisation

Weight matrices are initialised with a fan-in-scaled (He-style) normal
distribution:

```
W ~ N(0, sqrt(scale / fan_in))
```

- `fan_in` is the number of input units to the layer.
- `scale` defaults to 2.0 (He initialisation, suitable for ReLU networks).
- A separate `numpy.random.Generator` seeded by the user-provided `random_state`
  is used so the initialisation is deterministic.

Biases are initialised to zero.

Deterministic initialisation means two models constructed with the same
architecture and the same `random_state` will start with identical weights and,
given identical data, will follow identical trajectories.

## Reproducibility

Reproducibility is enforced at several levels:

- **Random seed:** `random_state` is passed to the numpy generator and to the
  dropout mask cache.
- **Deterministic dropout masks:** masks are cached by batch size so the same
  mask is reused within an epoch and across epochs of the same batch size.
- **Same data, same seed → same result:** repeated training runs with identical
  architecture, data, and seed produce identical final weights and predictions.
- **Synthetic data:** the demo and tests use `make_synthetic_records(seed=42)` so
  the dataset itself is deterministic.

## TF-IDF input

The model does **not** operate on raw text.  It receives TF-IDF features from a
`sklearn.feature_extraction.text.TfidfVectorizer` that is owned and fitted by the
`ResponseMLPClassifier` adapter.

The adapter:

- builds a `TfidfVectorizer` with the same kinds of parameters the logistic
  baseline uses (max_features, ngram_range, sublinear_tf, min_df, max_df, etc.);
- fits the vectorizer on **training texts only**;
- transforms validation and test texts through the **fitted** vectorizer;
- saves the fitted vectorizer inside the joblib artifact so a loaded model can
  score new comments.

This is the same leakage-prevention discipline as the logistic baseline.

## Data leakage prevention

The training pipeline is:

1. load dataset
2. split into train / validation / test (deterministic, stratified for single-label
   tasks)
3. **fit TF-IDF on training texts only**
4. transform validation and test texts with the fitted vectorizer
5. train the MLP on the training TF-IDF matrix
6. evaluate on validation and test TF-IDF matrices

At no point are validation or test texts seen by the vectorizer's `fit()` or by
the MLP's `fit()`.  The existing `split_records` function and leakage checker
verify that no record appears in more than one split.

## Train / validation / test split

The split is deterministic with a fixed seed (default 42) and stratified for
single-label tasks.  The default proportions are:

- train: 70%
- validation: 15%
- test: 15%

The Trainer enforces minimum sizes (`minimum_training_examples`, plus the split
layer's own `min_train_examples`, `min_validation_examples`, `min_test_examples`)
so training does not proceed on degenerate splits.

## Why this is a baseline neural experiment

This is a **baseline** because:

- the input representation is still TF-IDF (bag-of-ngrams), not learned
  embeddings or contextualised representations;
- the architecture is a small three-layer MLP, not a deep or wide network;
- the training is on CPU with NumPy only;
- the task is the same six response roles the logistic baseline already handles.

The ANN is a stepping stone toward more expressive models (CNN, multilingual
transformer) that will use the same training/evaluation/artifact/registry
infrastructure.

## Limitations of TF-IDF + MLP

- **Bag-of-ngrams input:** TF-IDF loses word order and long-range context beyond
  the ngram window.  A comment like "I agree, but only if..." and "I disagree,
  unless..." may look similar after TF-IDF if the shared tokens dominate.
- **No contextualised representations:** each token is treated independently of
  its surrounding sentence structure.
- **Small dataset sensitivity:** with limited training examples the MLP can overfit
  or fail to cover all six classes, especially rare ones like `counterargument`.
- **Multilingual data:** TF-IDF on multilingual text without language-specific
  preprocessing treats each script as a separate feature space, which can dilute
  signal when the same issue is discussed in Hindi, Bengali, Odia, and English
  within the same community.
- **No sequence structure:** the MLP has no notion of the comment being a sequence
  or of the parent comment context; it sees only the flattened TF-IDF vector.

These limitations are the motivation for the CNN and multilingual transformer
stages that follow.

## CPU-only NumPy implementation

This experiment is intentionally **pure NumPy**:

- forward pass: matrix multiplications and element-wise ops
- loss: cross-entropy
- backward pass: explicit gradient formulas
- update: momentum SGD with clipping
- no autograd, no GPU, no deep-learning framework

This keeps the experiment inspectable and avoids adding TensorFlow or PyTorch as
a dependency.  If a future stage adds a framework, the existing `TrainableModel`
interface is the plug-in point, not this file.

GPU support is not required for this baseline and is not assumed.

## Why ANN does not automatically outperform logistic regression

A neural network is not automatically better than logistic regression on this task
because:

- both models can use the **same** TF-IDF input;
- logistic regression is already a linear model in the TF-IDF space, and for many
  text-classification tasks a linear model on TF-IDF is hard to beat without
  richer features;
- the MLP adds non-linearity and capacity, but capacity without richer features or
  more data can overfit or simply match the linear baseline;
- the six response roles may be separable enough in TF-IDF space that a linear
  decision boundary is adequate;
- the ANN has more hyper-parameters (hidden sizes, dropout, learning rate, momentum,
  epochs, batch size, gradient clipping) to tune, and a poor choice can underperform
  the logistic baseline.

Any claim that one model is better than the other must be based on evaluation on a
**real annotated dataset**, using the same split, not on synthetic data.

## Future CNN

The next neural experiment (not part of this document) is a 1D CNN on the same
TF-IDF or token-sequence input.  The CNN would add:

- convolutional filters over ngram positions;
- pooling to capture the most salient ngrams;
- a denser classification head.

The CNN is not implemented here and has no equations documented here beyond the
fact that it is a future stage.

## Future multilingual transformer

After the CNN, the project may experiment with a multilingual transformer
(embeddings or fine-tuned) to move beyond TF-IDF.  That stage would:

- replace the TF-IDF input with contextualised token representations;
- handle multilingual comments without separate feature spaces per script;
- require substantially more compute and data;
- plug into the same Trainer/evaluator/artifact/registry infrastructure.

No transformer architecture is documented or implemented here.

## Current code location

- MLP core: `ml/models/ann/numpy_mlp.py`
- Math helpers: `ml/models/ann/base.py`
- Response adapter: `ml/models/ann/response_mlp.py`
- Tests: `backend/test_ann_response.py`
- Demo: `backend/demo_ann_response.py`
- Classical baseline (unchanged): `app/services/nlp/response_classification/`

## Relationship to the classical baseline

The classical baseline remains the primary response-classification model:

- task: `TaskType.RESPONSE`
- default model: `TfidfLogisticResponseClassifier`
- module: `app/services/nlp/response_classification/tfidf_logistic_multiclass.py`

The ANN is an alternative model for the same task, supplied as an explicit
`model=` override to the Trainer.  No task-type, annotation-schema, or split
changes are required to run it.

Future comparison will report:

- accuracy
- macro F1
- micro F1
- weighted F1
- precision
- recall
- per-class F1
- confusion matrix

on the same validation/test split, using real annotated data, with the explicit
caveat that results are data-dependent and not implied by synthetic demos.
