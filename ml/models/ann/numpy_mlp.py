"""
Reusable NumPy MLP core — forward, loss, backward, update.

Framework-free: only depends on numpy.  Task-specific wrappers (e.g. the
response classification adapter in ``response_mlp.py``) add featurisation,
label mapping, and the predict/evaluate contract that the ML framework expects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class NumpyMLP:
    """
    Fully-connected MLP in pure NumPy with ReLU hidden activations and
    numerically stable softmax on the output layer.

    Architecture (configurable)::

        input
          -> Dense(hidden1)
          -> ReLU
          -> Dropout(dropout)        # training only
          -> Dense(hidden2)
          -> ReLU
          -> Dense(n_classes)
          -> Softmax

    Designed for single-label multiclass classification.  The final layer
    produces raw logits; softmax is applied in ``predict_proba`` and during
    loss computation.

    Deterministic: ``random_state`` seeds every RNG source (weight init,
    dropout mask, batch shuffling).  Two runs with the same seed and data
    produce identical weights and predictions.

    No deep-learning framework is used.  This keeps the project's dependency
    footprint small and lets the ML framework run a non-sklearn model through
    the same train / evaluate / artifact pipeline as the classical baselines.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 40,
        batch_size: int = 16,
        random_state: int = 42,
        clip_gradient: float = 5.0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            input_dim: feature dimension after featurisation (e.g. TF-IDF vocab size).
            n_classes: number of output classes (6 for V1 response types).
            hidden1: first hidden layer width.
            hidden2: second hidden layer width.
            dropout: inverted-dropout keep rate; 0 means no dropout.
            learning_rate: step size for stochastic gradient descent.
            momentum: SGD momentum coefficient.
            epochs: full passes over the training set.
            batch_size: mini-batch size.
            random_state: seed for every RNG source.
            clip_gradient: global gradient norm clip threshold.
            verbose: print per-epoch loss when True.
        """
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if clip_gradient <= 0:
            raise ValueError(f"clip_gradient must be positive, got {clip_gradient}")

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.clip_gradient = clip_gradient
        self.verbose = verbose

        # Weight storage: None until fit().
        self._W1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._W2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None
        self._W3: Optional[np.ndarray] = None
        self._b3: Optional[np.ndarray] = None

        self._rng: Optional[np.random.Generator] = None
        self._dropout_mask_cache: Optional[_DropoutMask] = None

        # Training log.
        self._epoch_losses: List[float] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyMLP":
        """
        Train the MLP on (X, y).

        Args:
            X: training features, shape (n_samples, input_dim), float64.
            y: training labels, shape (n_samples,), integer class indices in
                [0, n_classes).

        Returns:
            self, for chaining.
        """
        if self._is_fitted:
            self._reset_weights()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}"
            )
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but input_dim={self.input_dim}"
            )
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(
                f"labels must be in [0, {self.n_classes}), got range "
                f"[{y.min()}, {y.max()}]"
            )

        self._rng = np.random.default_rng(self.random_state)
        self._dropout_mask_cache = _DropoutMask(self._rng, self.dropout)

        n_samples, _ = X.shape
        n_classes = self.n_classes

        # ---- weight initialisaton (fan-in scaled, deterministic) ----
        self._W1 = self._rng.normal(
            0.0, np.sqrt(2.0 / self.input_dim), size=(self.hidden1, self.input_dim)
        )
        self._b1 = np.zeros((self.hidden1, 1))
        self._W2 = self._rng.normal(
            0.0,
            np.sqrt(2.0 / self.hidden1),
            size=(self.hidden2, self.hidden1),
        )
        self._b2 = np.zeros((self.hidden2, 1))
        self._W3 = self._rng.normal(
            0.0,
            np.sqrt(2.0 / self.hidden2),
            size=(n_classes, self.hidden2),
        )
        self._b3 = np.zeros((n_classes, 1))

        # Momentum buffers.
        vW1 = np.zeros_like(self._W1)
        vb1 = np.zeros_like(self._b1)
        vW2 = np.zeros_like(self._W2)
        vb2 = np.zeros_like(self._b2)
        vW3 = np.zeros_like(self._W3)
        vb3 = np.zeros_like(self._b3)

        # One-hot labels: (K, N).
        Y = np.zeros((n_classes, n_samples), dtype=np.float64)
        for n in range(n_samples):
            Y[y[n], n] = 1.0

        clip = self.clip_gradient
        bs = self.batch_size
        lr = self.learning_rate
        mu = self.momentum
        epochs = self.epochs

        for epoch in range(epochs):
            # Shuffle.
            perm = self._rng.permutation(n_samples)
            Xs = X[perm].T  # (D, N)
            Ys = Y[:, perm]  # (K, N)

            epoch_loss = 0.0
            n_batches = max(1, (n_samples + bs - 1) // bs)

            for b in range(n_batches):
                start = b * bs
                end = min(n_samples, start + bs)
                Xn = Xs[:, start:end]   # (D, B)
                Yn = Ys[:, start:end]   # (K, B)
                B = end - start

                # ---- forward ----
                Z1 = self._W1 @ Xn + self._b1                 # (H1, B)
                H1 = np.maximum(0.0, Z1)                      # ReLU
                mask = self._dropout_mask_cache.get(B, self.hidden1)
                H1d = H1 * mask                                # inverted dropout

                Z2 = self._W2 @ H1d + self._b2                # (H2, B)
                H2 = np.maximum(0.0, Z2)                      # ReLU

                logits = self._W3 @ H2 + self._b3             # (K, B)

                # Numerically stable softmax.
                logits_s = logits - np.max(logits, axis=0, keepdims=True)
                exp_logits = np.exp(logits_s)
                probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # (K, B)

                # ---- loss (cross-entropy, averaged over batch) ----
                eps = 1e-9
                loss = -np.sum(Yn * np.log(probs + eps)) / B
                epoch_loss += loss * B

                # ---- backward ----
                d_logits = (probs - Yn) / B                   # (K, B)

                dW3 = d_logits @ H2.T                         # (K, H2)
                db3 = np.sum(d_logits, axis=1, keepdims=True)

                d_H2 = self._W3.T @ d_logits                 # (H2, B)
                d_Z2 = d_H2 * (Z2 > 0)                       # ReLU deriv
                dW2 = d_Z2 @ H1d.T                           # (H2, H1)
                db2 = np.sum(d_Z2, axis=1, keepdims=True)

                d_H1d = self._W2.T @ d_Z2                    # (H1, B)
                d_H1 = d_H1d * mask                          # undo dropout scale
                d_Z1 = d_H1 * (Z1 > 0)                       # ReLU deriv
                dW1 = d_Z1 @ Xn.T                            # (H1, D)
                db1 = np.sum(d_Z1, axis=1, keepdims=True)

                # Gradient clipping.
                for g in (dW1, db1, dW2, db2, dW3, db3):
                    gn = np.linalg.norm(g)
                    if gn > clip:
                        g *= clip / gn

                # Momentum + update.
                vW1 = mu * vW1 - lr * dW1
                vb1 = mu * vb1 - lr * db1
                vW2 = mu * vW2 - lr * dW2
                vb2 = mu * vb2 - lr * db2
                vW3 = mu * vW3 - lr * dW3
                vb3 = mu * vb3 - lr * db3

                self._W1 += vW1
                self._b1 += vb1
                self._W2 += vW2
                self._b2 += vb2
                self._W3 += vW3
                self._b3 += vb3

            avg_loss = epoch_loss / n_samples
            self._epoch_losses.append(float(avg_loss))
            if self.verbose and (
                epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1
            ):
                print(f"  epoch {epoch + 1:>3}/{epochs}  loss={avg_loss:.6f}")

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix, shape (n_classes, n_samples).

        Numerically stable softmax is applied to the final logits.
        Dropout is disabled during inference.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects {self.input_dim}"
            )
        Z1 = self._W1 @ X.T + self._b1
        H1 = np.maximum(0.0, Z1)
        Z2 = self._W2 @ H1 + self._b2
        H2 = np.maximum(0.0, Z2)
        logits = self._W3 @ H2 + self._b3
        logits_s = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits_s)
        return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard class predictions, shape (n_samples,), integer indices.

        Arg-max over the probability vector; ties resolve to the lower-index
        class (deterministic).
        """
        proba = self.predict_proba(X)  # (K, N)
        return np.argmax(proba, axis=0).astype(np.int64)

    # ------------------------------------------------------------------
    # State / metadata
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        """Total trainable scalar parameters."""
        if not self._is_fitted:
            # Compute from architecture; weights not yet materialised.
            return (
                self.hidden1 * self.input_dim
                + self.hidden1
                + self.hidden2 * self.hidden1
                + self.hidden2
                + self.n_classes * self.hidden2
                + self.n_classes
            )
        return (
            self._W1.size
            + self._b1.size
            + self._W2.size
            + self._b2.size
            + self._W3.size
            + self._b3.size
        )

    @property
    def epoch_losses(self) -> List[float]:
        """Training loss per epoch, empty before fit()."""
        return list(self._epoch_losses)

    def get_state(self) -> Dict[str, Any]:
        """Serializable snapshot of weights + metadata for artifact storage."""
        if not self._is_fitted:
            raise RuntimeError("model not fitted; call fit() first")
        return {
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "clip_gradient": self.clip_gradient,
            "verbose": self.verbose,
            "W1_shape": list(self._W1.shape),
            "W2_shape": list(self._W2.shape),
            "W3_shape": list(self._W3.shape),
            "n_parameters": int(self.n_parameters),
            "epoch_losses": [round(l, 8) for l in self._epoch_losses],
        }

    def _reset_weights(self) -> None:
        self._W1 = self._b1 = self._W2 = self._b2 = self._W3 = self._b3 = None
        self._rng = None
        self._dropout_mask_cache = None
        self._epoch_losses = []
        self._is_fitted = False

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._W1 is None:
            raise RuntimeError("model not fitted; call fit() first")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DropoutMask:
    """Cache a dropout mask for a given batch size so masks are deterministic."""

    def __init__(self, rng: np.random.Generator, rate: float) -> None:
        self._rng = rng
        self._rate = rate
        self._mask: Optional[np.ndarray] = None
        self._last_n: int = -1

    def get(self, n: int, hidden_dim: int) -> np.ndarray:
        if self._mask is None or self._last_n != n:
            keep = 1.0 - self._rate
            self._mask = (
                (self._rng.random((hidden_dim, n)) < keep).astype(np.float64) / keep
            )
            self._last_n = n
        return self._mask
