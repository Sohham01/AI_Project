"""Apples-to-apples baseline architectures (Logistic Regression, ANN/MLP, 1D TextCNN)
for head-to-head comparison against MuRIL-DAPT v3 Transformer on identical splits
and unseen 54-probe stress benchmarks.

All three baselines support single-label (binary or multiclass) and multi-label
classification with validation-only threshold & temperature calibration, plus
parameter counts, training time, and inference latency telemetry.
"""

from __future__ import annotations

from collections import Counter
import gc
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion


def _macro_f1_single(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if int((y_true == c).sum()) == 0 and (tp + fp) == 0:
            continue
        denom = 2 * tp + fp + fn
        scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _macro_f1_multilabel(
    y_true: np.ndarray, probs: np.ndarray, thresholds: Sequence[float]
) -> Tuple[float, Dict[int, float]]:
    per_col: Dict[int, float] = {}
    for col, tau in enumerate(thresholds):
        truth = y_true[:, col]
        pred = (probs[:, col] > tau).astype(int)
        tp = int(((truth == 1) & (pred == 1)).sum())
        fp = int(((truth == 0) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        if int(truth.sum()) == 0 and (tp + fp) == 0:
            continue
        denom = 2 * tp + fp + fn
        per_col[col] = (2.0 * tp / denom) if denom > 0 else 0.0
    return (float(np.mean(list(per_col.values()))) if per_col else 0.0), per_col


def _calibrate_multilabel_thresholds(
    y_true: np.ndarray, probs: np.ndarray, labels: Sequence[str], default_threshold: float = 0.5
) -> Dict[str, float]:
    grid = [round(0.15 + 0.025 * i, 3) for i in range(29)]
    tuned: Dict[str, float] = {}
    for col, label in enumerate(labels):
        truth = y_true[:, col]
        if int(truth.sum()) == 0:
            tuned[label] = default_threshold
            continue
        best_tau, best_f1 = default_threshold, -1.0
        col_p = probs[:, col]
        for tau in grid:
            pred = (col_p > tau).astype(int)
            tp = int(((truth == 1) & (pred == 1)).sum())
            fp = int(((truth == 0) & (pred == 1)).sum())
            fn = int(((truth == 1) & (pred == 0)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            if (f1 > best_f1 + 1e-12) or (
                abs(f1 - best_f1) <= 1e-12 and abs(tau - default_threshold) < abs(best_tau - default_threshold)
            ):
                best_f1, best_tau = f1, tau
        tuned[label] = best_tau
    return tuned

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.maximum(exps.sum(axis=1, keepdims=True), 1e-12)


def _calibrate_temperature_np(
    logits: np.ndarray,
    y_idx: np.ndarray,
    multilabel: bool,
) -> float:
    """Validation-only scalar temperature T >= 1.0 minimizing NLL / BCE (non-sharpening)."""
    grid = [1.0, 1.15, 1.35, 1.6, 2.0, 2.5, 3.0]
    best_t = 1.0
    best_loss = float("inf")
    eps = 1e-7
    for t in grid:
        scaled = logits / t
        if multilabel:
            p = np.clip(_sigmoid_np(scaled), eps, 1.0 - eps)
            loss = -float(np.mean(y_idx * np.log(p) + (1.0 - y_idx) * np.log(1.0 - p)))
        else:
            p = np.clip(_softmax_np(scaled), eps, 1.0 - eps)
            loss = -float(np.mean(np.log(p[np.arange(len(y_idx)), y_idx])))
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t


def _encode_targets(
    labels_list: Sequence[Any],
    class_names: Sequence[str],
    multilabel: bool,
) -> np.ndarray:
    c2i = {c: i for i, c in enumerate(class_names)}
    if multilabel:
        mat = np.zeros((len(labels_list), len(class_names)), dtype=np.float32)
        for row_i, ys in enumerate(labels_list):
            for lbl in ys:
                if lbl in c2i:
                    mat[row_i, c2i[lbl]] = 1.0
        return mat
    return np.array([c2i[str(y)] for y in labels_list], dtype=np.int64)


def _make_word_char_vectorizer(max_word: int = 12000, max_char: int = 18000) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_word,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_char,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 1. Logistic Regression Baseline (Word+Char TF-IDF + L2 LogReg)
# ---------------------------------------------------------------------------
class LogisticRegressionBaseline:
    """Linear Word+Char TF-IDF + L2 Logistic Regression benchmark."""

    def __init__(self, *, multilabel: bool = False, random_state: int = 42, c_value: float = 2.0) -> None:
        self.multilabel = multilabel
        self.random_state = random_state
        self.c_value = c_value
        self.vectorizer = _make_word_char_vectorizer()
        self.labels_: List[str] = []
        self.model_: Any = None
        self.thresholds_: np.ndarray | None = None
        self.train_seconds_: float = 0.0
        self.param_count_: int = 0

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
    ) -> "LogisticRegressionBaseline":
        t0 = time.perf_counter()
        self.labels_ = list(class_names)
        X_tr = self.vectorizer.fit_transform(train_texts)
        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)

        base_lr = LogisticRegression(
            C=self.c_value,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        if self.multilabel:
            self.model_ = OneVsRestClassifier(base_lr)
            self.model_.fit(X_tr, y_tr)
            self.param_count_ = sum(
                int(est.coef_.size + est.intercept_.size) for est in self.model_.estimators_
            )
            val_probs = self.predict_proba(val_texts)
            y_va = _encode_targets(val_targets, self.labels_, True)
            tuned = _calibrate_multilabel_thresholds(y_va, val_probs, self.labels_)
            self.thresholds_ = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)
        else:
            self.model_ = base_lr
            self.model_.fit(X_tr, y_tr)
            self.param_count_ = int(self.model_.coef_.size + self.model_.intercept_.size)

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        probs = self.model_.predict_proba(X)
        return np.asarray(probs, dtype=np.float64)

    def predict(self, texts: Sequence[str]) -> List[Any]:
        probs = self.predict_proba(texts)
        if self.multilabel:
            thr = self.thresholds_ if self.thresholds_ is not None else np.full(len(self.labels_), 0.5)
            out: List[List[str]] = []
            for row in probs:
                picked = [self.labels_[i] for i, p in enumerate(row) if p >= thr[i]]
                if not picked:
                    picked = [self.labels_[int(np.argmax(row))]]
                out.append(sorted(picked))
            return out
        return [self.labels_[int(np.argmax(row))] for row in probs]


# ---------------------------------------------------------------------------
# 2. ANN (Multi-Layer Perceptron over Dense TF-IDF) — PyTorch GPU/CPU
# ---------------------------------------------------------------------------
@dataclass
class NeuralBaselineConfig:
    epochs: int = 18
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    dropout: float = 0.35
    random_state: int = 42
    device: str = "auto"


class ANNMLPClassifier:
    """2-hidden-layer Feed-Forward Neural Network (ANN/MLP) over Word+Char TF-IDF."""

    def __init__(
        self,
        *,
        multilabel: bool = False,
        max_features: int = 3072,
        hidden_dims: Tuple[int, int] = (256, 128),
        config: NeuralBaselineConfig | None = None,
    ) -> None:
        self.multilabel = multilabel
        self.max_features = max_features
        self.hidden_dims = hidden_dims
        self.config = config or NeuralBaselineConfig()
        self.vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features // 2, min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=max_features // 2, min_df=2, sublinear_tf=True)),
            ]
        )
        self.labels_: List[str] = []
        self.model_: Any = None
        self.device_: Any = None
        self.temperature_: float = 1.0
        self.thresholds_: np.ndarray | None = None
        self.train_seconds_: float = 0.0
        self.param_count_: int = 0
        self.best_epoch_: int = 0
        self.history_: List[Dict[str, float]] = []

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
    ) -> "ANNMLPClassifier":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        t0 = time.perf_counter()
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

        self.device_ = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device == "auto"
            else torch.device(self.config.device)
        )
        self.labels_ = list(class_names)
        n_classes = len(self.labels_)

        X_tr = self.vectorizer.fit_transform(train_texts).astype(np.float32).toarray()
        X_va = self.vectorizer.transform(val_texts).astype(np.float32).toarray()
        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        in_dim = X_tr.shape[1]
        h1, h2 = self.hidden_dims
        model = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(max(0.15, self.config.dropout - 0.10)),
            nn.Linear(h2, n_classes),
        ).to(self.device_)
        self.param_count_ = sum(p.numel() for p in model.parameters())

        if self.multilabel:
            pos = y_tr.sum(axis=0)
            neg = len(y_tr) - pos
            pos_weight = torch.tensor(
                np.clip(neg / np.maximum(pos, 1.0), 1.0, 8.0), dtype=torch.float32, device=self.device_
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        else:
            counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
            weights = len(y_tr) / np.maximum(counts * n_classes, 1.0)
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(np.clip(weights, 0.5, 6.0), dtype=torch.float32, device=self.device_)
            )
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))

        gen = torch.Generator().manual_seed(self.config.random_state)
        loader = DataLoader(ds_tr, batch_size=self.config.batch_size, shuffle=True, generator=gen)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.config.epochs, 1))

        best_score = -1.0
        best_state = None
        X_va_t = torch.from_numpy(X_va).to(self.device_)

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            tr_loss = 0.0
            n_b = 0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += float(loss.detach().item())
                n_b += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                va_logits = model(X_va_t).cpu().numpy()
            if self.multilabel:
                va_probs = _sigmoid_np(va_logits)
                score, _ = _macro_f1_multilabel(y_va.astype(int), va_probs, [0.5] * n_classes)
            else:
                va_pred = np.argmax(va_logits, axis=1)
                score = _macro_f1_single(y_va, va_pred, n_classes)

            self.history_.append(
                {"epoch": epoch, "train_loss": round(tr_loss / max(n_b, 1), 4), "val_macro_f1": round(score, 4)}
            )
            if score > best_score:
                best_score = score
                self.best_epoch_ = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(self.device_) for k, v in best_state.items()})
        self.model_ = model

        # Validation-only temperature & threshold calibration
        va_logits = self._predict_logits(val_texts)
        self.temperature_ = _calibrate_temperature_np(va_logits, y_va, self.multilabel)
        if self.multilabel:
            va_probs = _sigmoid_np(va_logits / self.temperature_)
            tuned = _calibrate_multilabel_thresholds(y_va, va_probs, self.labels_)
            self.thresholds_ = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def _predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        X = self.vectorizer.transform(texts).astype(np.float32).toarray()
        self.model_.eval()
        out_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i : i + 256]).to(self.device_)
                out_chunks.append(self.model_(xb).cpu().numpy())
        return np.vstack(out_chunks) if out_chunks else np.zeros((0, len(self.labels_)), dtype=np.float32)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        logits = self._predict_logits(texts) / max(self.temperature_, 0.1)
        return _sigmoid_np(logits) if self.multilabel else _softmax_np(logits)

    def predict(self, texts: Sequence[str]) -> List[Any]:
        probs = self.predict_proba(texts)
        if self.multilabel:
            thr = self.thresholds_ if self.thresholds_ is not None else np.full(len(self.labels_), 0.5)
            out: List[List[str]] = []
            for row in probs:
                picked = [self.labels_[i] for i, p in enumerate(row) if p >= thr[i]]
                if not picked:
                    picked = [self.labels_[int(np.argmax(row))]]
                out.append(sorted(picked))
            return out
        return [self.labels_[int(np.argmax(row))] for row in probs]

    def cleanup(self) -> None:
        import torch

        self.model_ = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. 1D TextCNN Classifier (Kim 2014 Multi-Kernel Convolutional Network)
# ---------------------------------------------------------------------------
class TextCNNClassifier:
    """Multi-kernel 1D Convolutional Neural Network (k=2,3,4,5) over token embeddings."""

    def __init__(
        self,
        *,
        multilabel: bool = False,
        max_vocab: int = 12000,
        max_len: int = 128,
        embed_dim: int = 128,
        num_filters: int = 96,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        config: NeuralBaselineConfig | None = None,
    ) -> None:
        self.multilabel = multilabel
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.config = config or NeuralBaselineConfig(epochs=20, learning_rate=2e-3, dropout=0.40)
        self.vocab_: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.labels_: List[str] = []
        self.model_: Any = None
        self.device_: Any = None
        self.temperature_: float = 1.0
        self.thresholds_: np.ndarray | None = None
        self.train_seconds_: float = 0.0
        self.param_count_: int = 0
        self.best_epoch_: int = 0
        self.history_: List[Dict[str, float]] = []

    def _build_vocab(self, texts: Sequence[str]) -> None:
        counts: Counter = Counter()
        for t in texts:
            counts.update(_TOKEN_RE.findall(str(t).lower()))
        self.vocab_ = {"<PAD>": 0, "<UNK>": 1}
        for tok, freq in counts.most_common(self.max_vocab - 2):
            if freq >= 2:
                self.vocab_[tok] = len(self.vocab_)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.max_len), dtype=np.int64)
        unk = self.vocab_["<UNK>"]
        for i, t in enumerate(texts):
            toks = _TOKEN_RE.findall(str(t).lower())[: self.max_len]
            for j, tok in enumerate(toks):
                arr[i, j] = self.vocab_.get(tok, unk)
        return arr

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
    ) -> "TextCNNClassifier":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        class _TextCNNNet(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                embed_dim: int,
                num_filters: int,
                kernel_sizes: Tuple[int, ...],
                num_classes: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.emb_drop = nn.Dropout(0.25)
                self.convs = nn.ModuleList(
                    [nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
                )
                feat_dim = num_filters * len(kernel_sizes)
                self.norm = nn.LayerNorm(feat_dim)
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(feat_dim, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                emb = self.emb_drop(self.embedding(x)).transpose(1, 2)  # (B, E, L)
                pooled = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
                cat = self.drop(self.norm(torch.cat(pooled, dim=1)))
                return self.fc(cat)

        t0 = time.perf_counter()
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

        self.device_ = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device == "auto"
            else torch.device(self.config.device)
        )
        self.labels_ = list(class_names)
        n_classes = len(self.labels_)

        self._build_vocab(train_texts)
        X_tr = self._encode_texts(train_texts)
        X_va = self._encode_texts(val_texts)
        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        model = _TextCNNNet(
            vocab_size=len(self.vocab_),
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            num_classes=n_classes,
            dropout=self.config.dropout,
        ).to(self.device_)
        self.param_count_ = sum(p.numel() for p in model.parameters())

        if self.multilabel:
            pos = y_tr.sum(axis=0)
            neg = len(y_tr) - pos
            pos_weight = torch.tensor(
                np.clip(neg / np.maximum(pos, 1.0), 1.0, 8.0), dtype=torch.float32, device=self.device_
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        else:
            counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
            weights = len(y_tr) / np.maximum(counts * n_classes, 1.0)
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(np.clip(weights, 0.5, 6.0), dtype=torch.float32, device=self.device_)
            )
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))

        gen = torch.Generator().manual_seed(self.config.random_state)
        loader = DataLoader(ds_tr, batch_size=self.config.batch_size, shuffle=True, generator=gen)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.config.epochs, 1))

        best_score = -1.0
        best_state = None
        X_va_t = torch.from_numpy(X_va).to(self.device_)

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            tr_loss = 0.0
            n_b = 0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += float(loss.detach().item())
                n_b += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                va_logits = model(X_va_t).cpu().numpy()
            if self.multilabel:
                va_probs = _sigmoid_np(va_logits)
                score, _ = _macro_f1_multilabel(y_va.astype(int), va_probs, [0.5] * n_classes)
            else:
                va_pred = np.argmax(va_logits, axis=1)
                score = _macro_f1_single(y_va, va_pred, n_classes)

            self.history_.append(
                {"epoch": epoch, "train_loss": round(tr_loss / max(n_b, 1), 4), "val_macro_f1": round(score, 4)}
            )
            if score > best_score:
                best_score = score
                self.best_epoch_ = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(self.device_) for k, v in best_state.items()})
        self.model_ = model

        va_logits = self._predict_logits(val_texts)
        self.temperature_ = _calibrate_temperature_np(va_logits, y_va, self.multilabel)
        if self.multilabel:
            va_probs = _sigmoid_np(va_logits / self.temperature_)
            tuned = _calibrate_multilabel_thresholds(y_va, va_probs, self.labels_)
            self.thresholds_ = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def _predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        X = self._encode_texts(texts)
        self.model_.eval()
        out_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i : i + 256]).to(self.device_)
                out_chunks.append(self.model_(xb).cpu().numpy())
        return np.vstack(out_chunks) if out_chunks else np.zeros((0, len(self.labels_)), dtype=np.float32)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        logits = self._predict_logits(texts) / max(self.temperature_, 0.1)
        return _sigmoid_np(logits) if self.multilabel else _softmax_np(logits)

    def predict(self, texts: Sequence[str]) -> List[Any]:
        probs = self.predict_proba(texts)
        if self.multilabel:
            thr = self.thresholds_ if self.thresholds_ is not None else np.full(len(self.labels_), 0.5)
            out: List[List[str]] = []
            for row in probs:
                picked = [self.labels_[i] for i, p in enumerate(row) if p >= thr[i]]
                if not picked:
                    picked = [self.labels_[int(np.argmax(row))]]
                out.append(sorted(picked))
            return out
        return [self.labels_[int(np.argmax(row))] for row in probs]

    def cleanup(self) -> None:
        import torch

        self.model_ = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
