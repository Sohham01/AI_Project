"""
TextCNN Response Classification — TF-IDF + NumPy Logistic vs Token-Sequence Conv1D

## Status

EXPERIMENTAL — a model variant of ``TaskType.RESPONSE``, not a new
task type.  The six V1 response classes are fixed by the assignment:
``agreement``, ``personal_experience``, ``opinion``, ``advice``,
``solution``, ``counterargument``.

## Where it lives

- Model core: ``ml/models/cnn/text_cnn.py``
- Response adapter: ``ml/models/cnn/response_cnn.py``
- Tests: ``backend/test_text_cnn_response.py``
- Demo: ``backend/demo_text_cnn_response.py``

## Why TextCNN

The existing response classifiers all start from TF-IDF feature vectors:

- **Logistic Regression baseline**: TF-IDF -> multiclass logistic.  Fast,
  interpretable, strong on local function-word phrases ("same bro", "i agree
  with you", "actually though").
- **NumPy MLP (ANN)**: TF-IDF -> Dense -> ReLU -> ... -> Dense -> Softmax.  A
  deeper feed-forward baseline over the same bag-of-ngrams features.

All three of these treat each comment as an unordered bag of ngrams.  That works
surprisingly well for response-role clues that live in short phrases, but it
cannot exploit word order or short-range position at all.  A comment whose
entire role signal is in the first four words ("actually, I think") gets the
same representation regardless of where those words appear.

A text CNN over token IDs is a natural next experiment because:

- it operates directly on **token sequences**, so position and local order are
  first-class inputs rather than lost in the bag-of-ngrams abstraction;
- a 1-D convolution over the embedded sequence is the standard "first neural
  baseline" for short-text classification, and it is simple enough to implement
  in a few hundred lines of NumPy;
- it gives us a clean comparison point: TF-IDF (bag) vs TF-IDF+MLP (deeper bag)
  vs token-sequence Conv1D (ordered, local ngrams learned from data).

This is explicitly an **experiment**.  Nothing here claims to beat the
TF-IDF baselines on project data.

## Architecture

The model is a single configurable Conv1D classifier::

    tokens (B, L)                  # token ids, B = batch, L = max_seq_len
      -> Embedding(V, D)           # (V, D) lookup, V = vocab_size, D = emb_dim
      -> (B, L, D)                 # embedded sequence
      -> Conv1D(F, K)              # F filters, kernel size K, VALID padding, stride 1
      -> (B, L-K+1, F)             # feature maps
      -> ReLU
      -> GlobalMaxPool1D
      -> (B, F)                    # one scalar per filter
      -> Dense(C)                  # C = n_classes (6 for V1 response types)
      -> (B, C)                    # logits
      -> Softmax
      -> (B, C)                    # probabilities

All dimensions are configurable via the constructor:

- ``vocab_size`` — from the fitted tokenizer.
- ``n_classes`` — 6 for V1.
- ``emb_dim`` — token embedding size (default 32).
- ``n_filters`` — number of Conv1D filters (default 32).
- ``kernel_size`` — Conv1D window in tokens (default 3).
- ``hidden_dim`` — Dense hidden layer size (default 32).
- ``dropout`` — dropout rate applied after Conv1D ReLU and after hidden ReLU
  (default 0.0).
- ``learning_rate``, ``momentum`` — SGD hyperparameters.
- ``epochs``, ``batch_size`` — training schedule.
- ``max_seq_len`` — maximum token sequence length; shorter sequences are
  right-padded with the pad token id, longer ones are truncated.
- ``random_state`` — seeds every RNG source for deterministic behaviour.
- ``clip_gradient`` — global gradient clipping threshold (L2 norm).

The model owns its tokenizer.  The adapter (``TextCNNResponseClassifier``)
builds the ``TextCNN`` internally and also exposes a ``config`` property that
returns a ``ResponseModelConfig`` for the existing ``ResponseClassification``
contract.

## Input representation

The input is a **sequence of token ids**, not a TF-IDF vector.

1. A ``SimpleTokenizer`` (owned by the model) is fitted on **training texts
   only**.  Vocabulary construction uses training data exclusively — validation
   and test texts never influence it.
2. Tokenization is whitespace + punctuation split with lowercasing.
3. The vocabulary has two special tokens always present: ``<pad>`` (index 0)
   and ``<unk>`` (index 1).  OOV tokens at inference time map to ``<unk>``.
4. Each text is encoded to a fixed-length integer array of shape
   ``(max_seq_len,)``, right-padded with the pad token id.
5. A batch of texts is a 2-D array of shape ``(N, max_seq_len)`` with dtype
   ``np.int64``.

The embedding layer then looks up each token id in a learned
``(vocab_size, emb_dim)`` matrix.  This is the first learnable component and it
is trained jointly with the rest of the network.

This is **not** a TF-IDF vector feed into a Conv1D.  The Conv1D convolves over
the embedded sequence, not over a sparse feature vector.

## Mathematical operations

### Embedding lookup

Given token ids ``x_int`` of shape ``(...,)`` and embedding matrix
``W_emb`` of shape ``(V, D)``:

.. math::

    x_{emb} = W_{emb}[x_{int}]

The result has shape ``(..., D)`` and is differentiable w.r.t. ``W_emb``.

### Conv1D, valid padding, stride 1

For one sample with embedded sequence ``x`` of shape ``(L, D)``, kernel
``K`` of shape ``(k, D, F)``, and bias ``b`` of shape ``(F,)``:

.. math::

    out[t, f] = \sum_{i=0}^{k-1} \sum_{c=0}^{D-1} K[i, c, f] \cdot x[t+i, c] + b[f]

for ``t`` in ``[0, L - k]`` and ``f`` in ``[0, F-1]``.

The implementation in ``conv1d_valid`` is explicit and easy to read: the kernel
is reshaped from ``(k, D, F)`` to ``(k*D, F)``, each window of length ``k`` is
flattened to a vector of length ``k*D``, and the output is the matrix product
``window @ kernel_reshaped + bias``.

The backward pass (``conv1d_valid_backward``) is the standard convolution
gradient: the gradient w.r.t. the input is a transposed convolution (full
padding), the gradient w.r.t. the kernel is the cross-correlation of the input
with the output gradient, and the gradient w.r.t. the bias is the sum over time
and batch.

### ReLU

.. math::

    a = \max(0, conv\_out)

The backward pass is the indicator ``(conv_out > 0)``.

### GlobalMaxPool1D

.. math::

    pooled[b, f] = \max_t a[b, t, f]

The output is shape ``(B, F)`` — one scalar per filter per sample.  The
backward pass routes the gradient only to the time index that achieved the max
for each filter (using ``np.put_along_axis``).

This is a form of position-independent feature detection: each filter learns to
fire on some local pattern, and the pooling picks whether that pattern appeared
anywhere in the sequence.  This is what makes the model somewhat position-robust
while still being sensitive to local word order.

### Dense (linear)

Two dense layers:

.. math::

    h_{pre} = pooled \cdot W_1^T + b_1, \quad h = \max(0, h_{pre})

    logits = h \cdot W_2^T + b_2

where ``W_1`` is ``(hidden_dim, n_filters)``, ``b_1`` is ``(hidden_dim,)``,
``W_2`` is ``(n_classes, hidden_dim)``, and ``b_2`` is ``(n_classes,)``.

### Softmax (numerically stable)

.. math::

    logits_s = logits - \max(logits, axis=1, keepdims=True)

    probs = \frac{\exp(logits_s)}{\sum_c \exp(logits_s[:, c])}

The max-subtraction is the standard stability trick: it prevents overflow in the
exponential without changing the output probabilities.

### Cross-entropy loss

For one-hot targets ``y`` (shape ``(B, n_classes)``) and predicted probabilities
``probs``:

.. math::

    L = -\frac{1}{B} \sum_{b=1}^{B} \sum_{c=1}^{C} y_{b,c} \cdot \log(probs_{b,c} + \epsilon)

with ``epsilon = 1e-9`` for numerical safety.

### Backpropagation

The backward pass is chain-rule through the full architecture.  Key steps:

1. **Softmax + cross-entropy** gradient: ``d_logits = (probs - y_onehot) / B``.
   This is the standard combined gradient.
2. **Dense2 backward**: ``d_h = W_2 @ d_logits.T`` (accumulated), then update
   ``W_2``, ``b_2`` with momentum SGD.
3. **ReLU backward**: ``d_h_pre = d_h * (h_pre > 0)``.
4. **Dense1 backward**: ``d_pooled = W_1 @ d_h_pre.T``, then update ``W_1``,
   ``b_1``.
5. **GlobalMaxPool backward**: route ``d_pooled`` back to the argmax positions in
   the conv output, producing ``d_conv_out`` with the same shape as the conv
   output but zero everywhere except at the max positions.
6. **Conv ReLU backward**: ``d_conv_pre_relu = d_conv_out * (conv_pre_relu > 0)``.
7. **Conv1D backward** (per sample via ``conv1d_valid_backward``): produces
   ``d_emb`` (gradient w.r.t. the embedded input), ``d_kernel``, ``d_bias``.
8. **Embedding update**: the per-sample ``d_emb`` contributions are accumulated
   across the batch, clipped if needed, and used to update ``W_emb`` with plain
   (non-momentum) SGD.
9. **Conv update**: ``d_kernel`` and ``d_bias`` are accumulated across the batch,
   clipped, and updated with momentum SGD.

### Optimisation

Conv layers and Dense layers use **momentum SGD**:

.. math::

    v \leftarrow \mu \cdot v - \eta \cdot g

    \theta \leftarrow \theta + v

where ``mu`` is the momentum, ``eta`` the learning rate, and ``g`` the gradient.
Velocity buffers are zeroed at the start of each ``fit()`` call so repeated
training runs from the same seed produce identical trajectories.

The embedding layer uses plain SGD (no momentum), matching the common practice of
treating the embedding as a lookup table updated directly by the accumulated
gradient.

Gradient clipping is applied globally: if the L2 norm of the conv gradients
exceeds ``clip_gradient``, all conv gradients are scaled down proportionally.
If the embedding gradient norm exceeds the threshold, it too is scaled.

## Training objective

The training objective is **multiclass cross-entropy** over the six response
classes.  Because the six roles are mutually exclusive (a comment plays one
primary role in its discussion), a single softmax distribution is the correct
output, and the loss penalises the probability mass assigned to the wrong class.

The model is trained to minimise the average cross-entropy over the training
set.  No class weighting is applied in the V1 implementation; the TF-IDF
logistic baseline uses ``class_weight="balanced"`` but the CNN leaves balancing
as a possible follow-up experiment.

## Hyperparameters

Defaults (used by the adapter when no explicit values are given)::

    embedding_dim  = 32
    n_filters      = 32
    kernel_size    = 3
    hidden_dim     = 32
    dropout        = 0.0
    learning_rate  = 0.01
    momentum       = 0.9
    epochs         = 20
    batch_size     = 16
    max_seq_len    = 64
    random_state   = 42
    clip_gradient  = 5.0

Small/CPU-friendly defaults used in the test fixture::

    embedding_dim  = 8
    n_filters      = 4
    kernel_size    = 2
    hidden_dim     = 4
    epochs         = 3..6
    batch_size     = 8..16

These small values are purely for fast synthetic tests.  They are not
recommendations for project data.

## How it integrates into the existing Trainer

The existing ``Trainer`` in ``ml/training/trainer.py`` is generic: it accepts
any model that implements the ``TrainableModel`` protocol (or the richer
``ResponseClassification`` interface for response tasks).  The trainer does not
inspect model internals; it calls ``train()``, ``predict()``, ``evaluate()`` and
reads ``name``, ``version``, ``is_trained``, ``threshold``, ``labels``.

Integration points for the TextCNN:

1. **Model injection**: The CNN is passed to the trainer explicitly via
   ``model=TextCNNResponseClassifier(...)``.  It does **not** go through
   ``build_model()`` — that factory only returns the classical TF-IDF baselines.
2. **TaskType**: The task remains ``TaskType.RESPONSE``.  There is no
   ``TaskType.RESPONSE_CNN``.  The CNN is a model variant, not a new task.
3. **Annotation conversion**: The existing ``records_to_annotations`` function
   already converts ``TrainingRecord`` objects into ``ResponseAnnotation``
   objects for ``TaskType.RESPONSE``, so the trainer can feed the CNN without
   any new conversion logic.
4. **Evaluation**: The existing ``EvaluationReport`` and metric computation paths
   work on the structured result objects returned by the model, so the CNN
   inherits the same evaluation pipeline as the baselines.
5. **Artifacts**: The model's ``save_artifact`` / ``load_artifact`` methods use
   the same joblib-based artifact convention as the rest of the framework, and
   the trainer's save step calls the model's own save method.

No changes to the trainer, the splitting logic, the evaluation framework, or the
existing baselines are required.

## Limitations

1. **Pure NumPy, no GPU**.  The implementation is intentionally CPU-only and
   uses only NumPy.  Conv1D is implemented as explicit loops over the batch and
   time dimensions.  This is fine for small models and small batches, but it is
   not scalable to large vocabularies or long documents.

2. **Small architectures only**.  The default hyperparameters (32-dim embedding,
   32 filters, 32-dim hidden) are chosen so the model fits comfortably in CPU
   memory and trains quickly on reasonable batch sizes.  Scaling up the filter
   count or embedding dimension will increase training time linearly.

3. **No pre-trained embeddings**.  The embedding matrix is learned from scratch
   on the training data.  For low-resource languages or rare words, this means
   the model has no external signal.  A future experiment could initialise the
   embedding from a pre-trained multilingual source, but that is out of scope for
   V1.

4. **Vocabulary is training-only**.  This is correct for preventing leakage, but
   it means any token that appears only in validation or test data is mapped to
   ``<unk>``.  If the test set contains a large fraction of words unseen in
   training, the embedding cannot contribute useful signal for those tokens.

5. **Vanilla Conv1D, no architectural refinements**.  There is no dilated
   convolution, no multiple kernel sizes concatenated (the "TextCNN" style from
   the Kim 2014 paper), no attention, no residual connections.  This is a single
   Conv1D layer followed by pooling, deliberately simple.

6. **Synthetic data only for plumbing tests**.  All tests and the demo use
   synthetic data.  The model's real performance on annotated project data is
   unknown and must be measured, not assumed.

7. **No sequence length adaptation**.  ``max_seq_len`` is fixed at construction
   time.  Very long comments are truncated; very short ones are padded.  There is
   no dynamic batching by length.

8. **Deterministic only with the same seed and same code**.  Like all the ML
   models in this project, the CNN is deterministic given a fixed
   ``random_state`` and fixed data, but it is not deterministic across NumPy
   versions or platforms in general.

## How it differs from TF-IDF + ANN (NumPy MLP)

Both the existing ANN and the new CNN are NumPy-based and framework-free.  The
difference is in the **input representation** and therefore in what patterns
each can learn:

- **TF-IDF + MLP (ANN)**: input is a sparse TF-IDF vector of shape
  ``(n_features,)`` where ``n_features`` is the size of the TF-IDF vocabulary
  (typically thousands).  The MLP is ``Dense(n_features -> hidden1) -> ReLU ->
  ... -> Dense(hiddenN -> 6) -> Softmax``.  It can learn arbitrary
  combinations of TF-IDF features but cannot exploit word order, because the TF-IDF
  vector has already destroyed position information.

- **TextCNN**: input is a sequence of token ids of shape ``(max_seq_len,)``.
  The embedding layer maps each token to a dense vector, and the Conv1D operates
  over the resulting ``(max_seq_len, emb_dim)`` tensor, learning local patterns
  across adjacent tokens.  Word order and local ngrams are directly available to
  the model.

In practice this means:

- The TF-IDF + MLP can learn that the *presence* of certain ngrams is predictive
  of a role, but not their position or relative order (beyond what the ngrams
  themselves encode).
- The TextCNN can learn that a pattern like `"actually" + "I" + "think"` in that
  order is predictive, even if those words appear at different positions in
  different comments, because the Conv1D slides over the sequence and the
  pooling picks up the presence of the pattern regardless of position.

Neither is universally better.  For response roles, many of the strongest cues
are short fixed phrases ("same bro", "true that", "i agree"), which TF-IDF with
ngrams already captures well.  The CNN's advantage would show up if role signals
depend more on position-dependent phrasing or short-range syntactic structure than
on individual ngrams.

## Equations reference

For the record, the full forward pass for one batch is:

1. ``emb = W_emb[x_int]``                            # (B, L, D)
2. ``conv[t] = sum_i sum_c K[i,c,f] * emb[t+i,c] + b[f]``  # (B, L-K+1, F)
3. ``a = relu(conv)``
4. ``pooled[b, f] = max_t a[b, t, f]``              # (B, F)
5. ``h_pre = pooled @ W1.T + b1``                  # (B, H)
6. ``h = relu(h_pre)``
7. ``logits = h @ W2.T + b2``                      # (B, C)
8. ``probs = softmax(logits)``

where the softmax is the numerically stable version.

The loss (averaged over the batch) is:

.. math::

    L = -\frac{1}{B}\sum_b \log probs[b, y_b]

where ``y_b`` is the true class index for sample ``b``.

## V1 scope

This is an **experimental model variant** for ``TaskType.RESPONSE``.  It is not
part of the completed NLP pipeline stages, which remain frozen.  The CNN exists
to give the project a sequence-based neural baseline alongside the bag-of-ngrams
baselines, so the team can compare what each representation learns on the same
six-class response classification problem.

No claims about real-world performance are made.  All tests use synthetic data.
The demo clearly separates synthetic plumbing verification from any real
evaluation.
"""
