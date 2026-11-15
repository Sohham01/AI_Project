"""
Text CNN core — pure NumPy implementation of a sequence-level 1-D convolutional
text classifier.

Architecture (configurable)::

    tokens (B, L)              # token ids, B = batch, L = max_seq_len
      -> Embedding(V, D)       # (V, D) lookup, V = vocab size, D = emb dim
      -> (B, L, D)             # embedded sequence
      -> Conv1D(F, K)          # F filters, kernel size K, valid padding
      -> (B, L-K+1, F)         # feature maps
      -> ReLU
      -> GlobalMaxPool1D
      -> (B, F)                # one scalar per filter
      -> Dense(C)              # C = n_classes
      -> (B, C)                # logits
      -> Softmax

Deterministic: ``random_state`` seeds every RNG source.  CPU-only, no framework.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tokenizer / vocabulary
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Whitespace + punctuation tokeniser with a trainable vocabulary.

    Vocabulary is built from TRAINING texts only.  OOV tokens are mapped to
    ``<unk>``; no validation/test text ever touches ``fit()``.
    """

    PASTE_TOKEN_RE = __import__("re").compile(r"[^\w']+")

    def __init__(
        self,
        max_vocab_size: int = 10000,
        max_seq_len: int = 64,
        lowercase: bool = True,
        oov_token: str = "<unk>",
        pad_token: str = "<pad>",
    ) -> None:
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.lowercase = lowercase
        self.oov_token = oov_token
        self.pad_token = pad_token
        # Special tokens are always present, even before fit().
        self._word2idx: Dict[str, int] = {pad_token: 0, oov_token: 1}
        self._idx2word: List[str] = [pad_token, oov_token]
        self._vocab_fitted = False

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        text = text.lower() if self.lowercase else text
        text = self.PASTE_TOKEN_RE.sub(" ", text)
        tokens = text.split()
        return tokens

    # ------------------------------------------------------------------
    # Vocabulary (train only)
    # ------------------------------------------------------------------

    def fit(self, texts: Sequence[str]) -> "SimpleTokenizer":
        counter: Dict[str, int] = {}
        for text in texts:
            for tok in self.tokenize(text):
                counter[tok] = counter.get(tok, 0) + 1
        # Special tokens first.
        self._word2idx = {self.pad_token: 0, self.oov_token: 1}
        self._idx2word = [self.pad_token, self.oov_token]
        # Most frequent words up to max_vocab_size.
        for word, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[
            : self.max_vocab_size - 2
        ]:
            if word not in self._word2idx:
                self._word2idx[word] = len(self._idx2word)
                self._idx2word.append(word)
        self._vocab_fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self._idx2word) if self._vocab_fitted else 2  # pad + unk

    @property
    def is_fitted(self) -> bool:
        return self._vocab_fitted

    def save(self) -> Dict[str, Any]:
        return {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_len": self.max_seq_len,
            "lowercase": self.lowercase,
            "oov_token": self.oov_token,
            "pad_token": self.pad_token,
            "word2idx": dict(self._word2idx),
            "idx2word": list(self._idx2word),
            "vocab_fitted": self._vocab_fitted,
        }

    @classmethod
    def load(cls, saved: Dict[str, Any]) -> "SimpleTokenizer":
        tok = cls(
            max_vocab_size=saved["max_vocab_size"],
            max_seq_len=saved["max_seq_len"],
            lowercase=saved["lowercase"],
            oov_token=saved["oov_token"],
            pad_token=saved["pad_token"],
        )
        tok._word2idx = dict(saved["word2idx"])
        tok._idx2word = list(saved["idx2word"])
        tok._vocab_fitted = saved["vocab_fitted"]
        return tok

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return integer token matrix, shape (B, L), padded/truncated."""
        encoded: List[List[int]] = []
        for text in texts:
            tokens = self.tokenize(text)
            ids = [self._word2idx.get(t, self._word2idx[self.oov_token]) for t in tokens]
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
            while len(ids) < self.max_seq_len:
                ids.append(self._word2idx[self.pad_token])
            encoded.append(ids)
        return np.array(encoded, dtype=np.int64)

    def decode(self, ids: np.ndarray) -> List[str]:
        return [self._idx2word[i] for i in ids.tolist()]


# ---------------------------------------------------------------------------
# Conv1D (numpy, valid padding, stride 1)
# ---------------------------------------------------------------------------


def conv1d_valid(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """1-D convolution, VALID padding, stride 1.

    Parameters
    ----------
    x : (L, C_in) input sequence (one sample).
    kernel : (K, C_in, C_out) convolution kernel.
    bias : (C_out,) bias vector.

    Returns
    -------
    out : (L - K + 1, C_out)
    """
    L, C_in = x.shape
    K, C_in_k, C_out = kernel.shape
    assert C_in == C_in_k, f"{C_in} != {C_in_k}"
    out_len = L - K + 1
    if out_len < 1:
        raise ValueError(f"input length {L} < kernel size {K}")
    out = np.empty((out_len, C_out), dtype=np.float64)
    for t in range(out_len):
        window = x[t : t + K].reshape(-1)  # (K*C_in,)
        out[t] = window @ kernel.reshape(-1, C_out) + bias
    return out


def conv1d_valid_backward(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    out_grad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop through conv1d_valid.  Returns (dx, dkernel, dbias)."""
    L, C_in = x.shape
    K, C_in_k, C_out = kernel.shape
    out_len = L - K + 1
    dx = np.zeros_like(x)
    dkernel = np.zeros_like(kernel)
    dbias = np.sum(out_grad, axis=0)
    for t in range(out_len):
        window = x[t : t + K].reshape(-1)
        dkernel += np.outer(window, out_grad[t]).reshape(K, C_in, C_out)
        dx[t : t + K] += np.tensordot(out_grad[t], kernel, axes=([0], [2]))
    return dx, dkernel, dbias


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------


class Embedding:
    """Trainable lookup table: (V, D)."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        rng: np.random.Generator,
    ) -> None:
        scale = math.sqrt(2.0 / vocab_size)
        self.W = rng.normal(0.0, scale, size=(vocab_size, emb_dim)).astype(np.float64)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

    @property
    def n_parameters(self) -> int:
        return self.W.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (...,) int indices -> (..., D) embeddings."""
        return self.W[x]

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """x: (B, L) int indices, grad: (B, L, D).  Returns dW."""
        dW = np.zeros_like(self.W)
        np.add.at(dW, x, grad)
        return dW

    def get_state(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "W": self.W.tolist(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.W = np.array(state["W"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Global Max Pool 1D
# ---------------------------------------------------------------------------


def global_max_pool1d(x: np.ndarray) -> np.ndarray:
    """x: (..., S, F) -> (..., F).  Max over the sequence axis (S)."""
    axis = x.ndim - 2
    out = np.max(x, axis=axis)
    return out


def global_max_pool1d_backward(
    x: np.ndarray,
    grad: np.ndarray,
) -> np.ndarray:
    """Backward through global max pool.  grad: (..., F)."""
    axis = x.ndim - 2
    # Scatter each gradient value to the position of the max along the sequence
    # axis (the only input that influenced the pooled output).
    max_idx = np.argmax(x, axis=axis, keepdims=True)
    out = np.zeros_like(x)
    np.put_along_axis(out, max_idx, grad[..., None, :], axis=axis)
    return out


# ---------------------------------------------------------------------------
# Dense (linear) layer
# ---------------------------------------------------------------------------


class Dense:
    """Fully-connected: z = x @ W.T + b."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rng: np.random.Generator,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        scale = math.sqrt(2.0 / in_dim)
        self.W = rng.normal(0.0, scale, size=(out_dim, in_dim)).astype(np.float64)
        self.b = np.zeros((out_dim,), dtype=np.float64)
        self.lr = learning_rate
        self.mu = momentum
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def n_parameters(self) -> int:
        return self.W.size + self.b.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in_dim) -> (..., out_dim)."""
        self._cache_x = x
        return x @ self.W.T + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """grad: (..., out_dim).  Returns dx with same shape as cached x."""
        x = self._cache_x
        # dW: sum over batch dims of outer(grad_last, x_last).
        grad_last = grad.reshape(-1, self.out_dim)       # (N, out_dim)
        x_reshaped = x.reshape(-1, self.in_dim)          # (N, in_dim)
        dW = grad_last.T @ x_reshaped                    # (out_dim, in_dim)
        db = np.sum(grad_last, axis=0)                   # (out_dim,)
        dx = grad @ self.W                               # (..., in_dim)
        # Momentum update.
        self.vW = self.mu * self.vW - self.lr * dW
        self.vb = self.mu * self.vb - self.lr * db
        self.W += self.vW
        self.b += self.vb
        return dx

    def clip_gradients(self, threshold: float) -> None:
        norm = math.sqrt(np.sum(self.vW ** 2) + np.sum(self.vb ** 2))
        if norm > threshold:
            scale = threshold / norm
            self.vW *= scale
            self.vb *= scale

    def get_state(self) -> Dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "lr": self.lr,
            "mu": self.mu,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.W = np.array(state["W"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)


# ---------------------------------------------------------------------------
# Text CNN model
# ---------------------------------------------------------------------------


class TextCNN:
    """Sequence-level text CNN: Embedding -> Conv1D -> ReLU -> GlobalMaxPool -> Dense -> Softmax.

    Deterministic, CPU-only, pure NumPy.  Supports save/load for inference.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        emb_dim: int = 32,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 20,
        batch_size: int = 16,
        max_seq_len: int = 64,
        random_state: int = 42,
        clip_gradient: float = 5.0,
        verbose: bool = False,
    ) -> None:
        if vocab_size < 3:
            raise ValueError(f"vocab_size must be >= 3, got {vocab_size}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if emb_dim < 1:
            raise ValueError(f"emb_dim must be >= 1, got {emb_dim}")
        if n_filters < 1:
            raise ValueError(f"n_filters must be >= 1, got {n_filters}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if max_seq_len < kernel_size:
            raise ValueError(
                f"max_seq_len={max_seq_len} < kernel_size={kernel_size}"
            )

        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.random_state = random_state
        self.clip_gradient = clip_gradient
        self.verbose = verbose

        rng = np.random.default_rng(random_state)
        self._embedding = Embedding(vocab_size, emb_dim, rng)
        self._conv_kernel = rng.normal(
            0.0,
            math.sqrt(2.0 / emb_dim),
            size=(kernel_size, emb_dim, n_filters),
        ).astype(np.float64)
        self._conv_bias = np.zeros((n_filters,), dtype=np.float64)
        self._dense1 = Dense(n_filters, hidden_dim, rng, learning_rate, momentum)
        self._dense2 = Dense(hidden_dim, n_classes, rng, learning_rate, momentum)

        self._vW_conv = np.zeros_like(self._conv_kernel)
        self._vb_conv = np.zeros_like(self._conv_bias)

        self._is_fitted = False
        self._epoch_losses: List[float] = []
        # Training-mode flags read by forward(); initialised here so forward()
        # is safe to call before fit() (previously only set inside fit()).
        self._training = False
        self._epoch = 0
        # Label mapping.
        self._class_index: Dict[str, int] = {}
        self._index_class: List[str] = []
        self._n_features = 0
        # Snapshot initial weights so fit() can reset on each call (deterministic retrain).
        self._init_embedding_W = self._embedding.W.copy()
        self._init_conv_kernel = self._conv_kernel.copy()
        self._init_conv_bias = self._conv_bias.copy()
        self._init_dense1_W = self._dense1.W.copy()
        self._init_dense1_b = self._dense1.b.copy()
        self._init_dense2_W = self._dense2.W.copy()
        self._init_dense2_b = self._dense2.b.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def names(self) -> List[str]:
        return list(self._index_class)

    @property
    def n_parameters(self) -> int:
        return (
            self._embedding.n_parameters
            + self._conv_kernel.size
            + self._conv_bias.size
            + self._dense1.n_parameters
            + self._dense2.n_parameters
        )

    @property
    def epoch_losses(self) -> List[float]:
        return list(self._epoch_losses)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, L) token ids -> (B, C) probabilities."""
        # Embedding: (B, L) -> (B, L, D)
        emb = self._embedding.forward(x)
        B, L, D = emb.shape

        # Conv1D: (B, L, D) -> (B, L-K+1, F)
        K = self.kernel_size
        F = self.n_filters
        out_len = L - K + 1
        conv_out = np.empty((B, out_len, F), dtype=np.float64)
        for b in range(B):
            conv_out[b] = conv1d_valid(emb[b], self._conv_kernel, self._conv_bias)
        # ReLU.
        conv_out = np.maximum(0.0, conv_out)
        # Dropout (training).
        if self.dropout > 0.0 and self._training:
            rng = np.random.default_rng(self.random_state + self._epoch)
            mask = (rng.random(conv_out.shape) < (1.0 - self.dropout)).astype(np.float64)
            mask /= (1.0 - self.dropout)
            conv_out = conv_out * mask
        # Global max pool: (B, out_len, F) -> (B, F)
        pool = global_max_pool1d(conv_out)
        # Dense1: (B, F) -> (B, H)
        h = self._dense1.forward(pool)
        h = np.maximum(0.0, h)
        # Dense2: (B, H) -> (B, C)
        logits = self._dense2.forward(h)
        # Softmax.
        logits_s = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_s)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_index: Dict[str, int],
    ) -> "TextCNN":
        """Train the CNN.

        X: (N, L) token ids.
        y: (N,) integer class labels.
        class_index: label -> int mapping.
        """
        self._training = True
        N, L = X.shape
        if y.shape[0] != N:
            raise ValueError(
                f"X has {N} samples but y has {y.shape[0]}; "
                f"expected same number of samples"
            )
        # Validate labels before any random state use.
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(
                f"labels must be in [0, {self.n_classes}), got range "
                f"[{y.min()}, {y.max()}]"
            )
        # Reset model weights to initial state so repeated fit() calls are deterministic.
        self._embedding.W[...] = self._init_embedding_W
        self._conv_kernel[...] = self._init_conv_kernel
        self._conv_bias[...] = self._init_conv_bias
        self._dense1.W[...] = self._init_dense1_W
        self._dense1.b[...] = self._init_dense1_b
        self._dense2.W[...] = self._init_dense2_W
        self._dense2.b[...] = self._init_dense2_b
        self._vW_conv[...] = 0.0
        self._vb_conv[...] = 0.0
        self._dense1.vW[...] = 0.0
        self._dense1.vb[...] = 0.0
        self._dense2.vW[...] = 0.0
        self._dense2.vb[...] = 0.0
        self._dense1._cache_x = None
        self._dense2._cache_x = None
        self._iteration = 0
        # Build ordered class-name list from class_index.
        self._index_class = [name for name, _ in sorted(class_index.items(), key=lambda kv: kv[1])]
        self._class_index = dict(class_index)
        # Reset RNG so repeated fit() calls are deterministic.
        rng = np.random.default_rng(self.random_state)
        self._epoch = 0  # reset so loss history is clean on retrain

        K = self.kernel_size
        F = self.n_filters
        H = self.hidden_dim
        C = self.n_classes
        lr = self.learning_rate
        mu = self.momentum
        epochs = self.epochs
        bs = self.batch_size

        for epoch in range(epochs):
            self._epoch = epoch
            perm = rng.permutation(N)
            Xs = X[perm]
            ys = y[perm]
            epoch_loss = 0.0
            n_batches = max(1, (N + bs - 1) // bs)

            for b in range(n_batches):
                start = b * bs
                end = min(N, start + bs)
                Xn = Xs[start:end]
                yn = ys[start:end]
                B = end - start

                # ---- forward (with dropout) ----
                emb = self._embedding.forward(Xn)
                conv_out = np.empty((B, L - K + 1, F), dtype=np.float64)
                for i in range(B):
                    conv_out[i] = conv1d_valid(
                        emb[i], self._conv_kernel, self._conv_bias
                    )
                conv_pre_relu = conv_out.copy()
                conv_out = np.maximum(0.0, conv_out)

                # Dropout.
                if self.dropout > 0.0:
                    mask = (
                        rng.random(conv_out.shape) < (1.0 - self.dropout)
                    ).astype(np.float64)
                    mask /= (1.0 - self.dropout)
                    conv_out_dropped = conv_out * mask
                else:
                    conv_out_dropped = conv_out

                pool = global_max_pool1d(conv_out_dropped)
                h_pre = self._dense1.forward(pool)
                h = np.maximum(0.0, h_pre)
                logits = self._dense2.forward(h)
                logits_s = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits_s)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                # ---- loss ----
                eps = 1e-9
                Y_onehot = np.zeros((B, C), dtype=np.float64)
                Y_onehot[np.arange(B), yn] = 1.0
                loss = -np.sum(Y_onehot * np.log(probs + eps)) / B
                epoch_loss += loss * B

                # ---- backward ----
                d_logits = (probs - Y_onehot) / B  # (B, C)

                # Dense2 backward.
                d_h = self._dense2.backward(d_logits)  # (B, H)
                d_h_pre = d_h * (h_pre > 0)

                # Dense1 backward.
                d_pool = self._dense1.backward(d_h_pre)  # (B, F)

                # Global max pool backward.
                d_pool_expanded = np.zeros_like(conv_out_dropped)
                for i in range(B):
                    max_idx = np.argmax(conv_out_dropped[i], axis=0)
                    for f in range(F):
                        d_pool_expanded[i, max_idx[f], f] = d_pool[i, f]

                # Undo dropout.
                if self.dropout > 0.0:
                    d_conv_out = d_pool_expanded * mask
                else:
                    d_conv_out = d_pool_expanded

                # ReLU backward.
                d_conv_pre_relu = d_conv_out * (conv_pre_relu > 0)

                # Conv1D backward (per sample): d_emb, d_kernel, d_bias.
                d_kernel_total = np.zeros_like(self._conv_kernel)
                d_bias_total = np.zeros_like(self._conv_bias)
                d_emb = np.zeros_like(emb)
                for i in range(B):
                    dx_i, dk, db = conv1d_valid_backward(
                        emb[i],
                        self._conv_kernel,
                        self._conv_bias,
                        d_conv_pre_relu[i],
                    )
                    d_emb[i] = dx_i
                    d_kernel_total += dk
                    d_bias_total += db

                # Clip conv gradients.
                conv_norm = math.sqrt(
                    np.sum(d_kernel_total ** 2) + np.sum(d_bias_total ** 2)
                )
                if conv_norm > self.clip_gradient:
                    scale = self.clip_gradient / conv_norm
                    d_kernel_total *= scale
                    d_bias_total *= scale

                # Update conv.
                self._vW_conv = mu * self._vW_conv - lr * d_kernel_total
                self._vb_conv = mu * self._vb_conv - lr * d_bias_total
                self._conv_kernel += self._vW_conv
                self._conv_bias += self._vb_conv

                # Update embedding.
                dW_emb = self._embedding.backward(Xn, d_emb)
                emb_norm = np.sqrt(np.sum(dW_emb ** 2))
                if emb_norm > self.clip_gradient:
                    dW_emb *= self.clip_gradient / emb_norm
                self._embedding.W -= lr * dW_emb

            avg_loss = epoch_loss / N
            self._epoch_losses.append(float(avg_loss))
            if self.verbose and (
                epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1
            ):
                print(f"  epoch {epoch + 1:>3}/{epochs}  loss={avg_loss:.6f}")

        self._is_fitted = True
        self._training = False
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, L) token ids -> (N, C) probabilities."""
        self._check_fitted()
        self._training = False
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, L) -> (N,) class indices."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int64)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-12, None))

    # ------------------------------------------------------------------
    # State / metadata
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("model not fitted; call fit() first")
        return {
            "vocab_size": self.vocab_size,
            "n_classes": self.n_classes,
            "emb_dim": self.emb_dim,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "random_state": self.random_state,
            "clip_gradient": self.clip_gradient,
            "class_index": dict(self._class_index),
            "index_class": list(self._index_class),
            "embedding_state": self._embedding.get_state(),
            "conv_kernel": self._conv_kernel.tolist(),
            "conv_bias": self._conv_bias.tolist(),
            "dense1_state": self._dense1.get_state(),
            "dense2_state": self._dense2.get_state(),
            "n_parameters": int(self.n_parameters),
            "epoch_losses": [round(l, 8) for l in self._epoch_losses],
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TextCNN":
        vocab_size = state["vocab_size"]
        n_classes = state["n_classes"]
        model = cls(
            vocab_size=vocab_size,
            n_classes=n_classes,
            emb_dim=state["emb_dim"],
            n_filters=state["n_filters"],
            kernel_size=state["kernel_size"],
            hidden_dim=state["hidden_dim"],
            dropout=state["dropout"],
            learning_rate=state["learning_rate"],
            momentum=state["momentum"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
            max_seq_len=state["max_seq_len"],
            random_state=state["random_state"],
            clip_gradient=state["clip_gradient"],
            verbose=False,
        )
        model._embedding.set_state(state["embedding_state"])
        model._conv_kernel = np.array(
            state["conv_kernel"], dtype=np.float64
        )
        model._conv_bias = np.array(
            state["conv_bias"], dtype=np.float64
        )
        model._dense1.set_state(state["dense1_state"])
        model._dense2.set_state(state["dense2_state"])
        model._class_index = dict(state["class_index"])
        model._index_class = list(state["index_class"])
        model._n_features = state.get("max_seq_len", 64)
        model._is_fitted = True
        model._epoch_losses = state.get("epoch_losses", [])
        return model

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("model not fitted; call fit() first")
