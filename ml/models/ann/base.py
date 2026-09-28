"""
Base abstractions for ANN models in the ML experimentation layer.

These are shared helpers that task-specific ANN adapters (e.g. the response
MLP) can reuse without depending on the frozen NLP classifier implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Numerically stable softmax.

    Subtracts the per-column (or per-row) max before exponentiating so that
    large logits do not overflow.  Returns a probability array with the same
    shape as ``logits`` where each slice along ``axis`` sums to 1.

    Args:
        logits: raw scores, any shape.
        axis: axis along which probabilities sum to 1 (default 0, i.e.
            each column is a distribution).

    Returns:
        Probability array of the same shape as ``logits``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU derivative: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float64)


def cross_entropy_loss(
    probs: np.ndarray,
    one_hot: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """
    Average cross-entropy loss for single-label multiclass with softmax.

    Args:
        probs: probability matrix, shape (K, N), each column sums to 1.
        one_hot: one-hot label matrix, same shape as probs.
        epsilon: clamp to avoid log(0).

    Returns:
        Scalar loss, averaged over N examples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    one_hot = np.asarray(one_hot, dtype=np.float64)
    if probs.shape != one_hot.shape:
        raise ValueError(
            f"probs shape {probs.shape} != one_hot shape {one_hot.shape}"
        )
    clamped = np.clip(probs, epsilon, 1.0)
    return float(-np.sum(one_hot * np.log(clamped)) / probs.shape[1])


def one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot matrix, shape (n_classes, N).

    Args:
        labels: 1-D integer array of class indices in [0, n_classes).
        n_classes: number of classes.

    Returns:
        One-hot matrix (n_classes, N).
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if labels.size == 0:
        return np.zeros((n_classes, 0), dtype=np.float64)
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(
            f"labels must be in [0, {n_classes}), got range [{labels.min()}, {labels.max()}]"
        )
    Y = np.zeros((n_classes, labels.size), dtype=np.float64)
    for n, k in enumerate(labels):
        Y[k, n] = 1.0
    return Y


def clip_gradient(grads: Sequence[np.ndarray], threshold: float) -> List[np.ndarray]:
    """
    Clip a list of gradient arrays to a global L2 norm threshold.

    Scales all gradients by ``threshold / norm`` when the combined norm
    exceeds ``threshold``, otherwise leaves them unchanged.

    Args:
        grads: gradient arrays.
        threshold: maximum L2 norm.

    Returns:
        List of gradient arrays, scaled if needed.  The caller owns the
        returned arrays (this function modifies in place when scaling).
    """
    total_sq = sum(float(np.linalg.norm(g) ** 2) for g in grads if g.size > 0)
    total_norm = float(np.sqrt(total_sq))
    if total_norm > threshold and total_norm > 0:
        scale = threshold / total_norm
        for g in grads:
            if g.size > 0:
                g *= scale
    return list(grads)


def lecun_normal_init(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """
    LeCun normal initialisation: N(0, sqrt(1 / fan_in)).

    Args:
        shape: weight shape (out_dim, in_dim).
        rng: numpy random generator.

    Returns:
        Initialised weight array.
    """
    fan_in = shape[1] if len(shape) >= 2 else max(shape[0], 1)
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=shape)


def fan_in_scaled_init(
    shape: Tuple[int, ...],
    rng: np.random.Generator,
    scale: float = 2.0,
) -> np.ndarray:
    """
    He / fan-in-scaled initialisation: N(0, sqrt(scale / fan_in)).

    Used for ReLU networks.  The default ``scale=2.0`` is the classic He
    initialisation.
    """
    fan_in = shape[1] if len(shape) >= 2 else max(shape[0], 1)
    std = np.sqrt(scale / fan_in)
    return rng.normal(0.0, std, size=shape)
