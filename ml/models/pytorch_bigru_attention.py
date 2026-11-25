"""Experimental PyTorch Bidirectional GRU + Additive Self-Attention Classifier (V6.4).

Fills the recurrent sequence-modeling slot alongside RCI's TF-IDF LogReg, MLP ANN,
1D TextCNN, XGBoost, Keras Multi-Task, and Transformer ensembles.

Supports:
  - Single-label binary & multiclass (`CrossEntropyLoss` with inverse-frequency weights
    and validation temperature scaling T >= 1.0)
  - Multi-label (`BCEWithLogitsLoss` with capped `pos_weight` and validation threshold tuning)
  - Inspectable token attention weights (`get_attention_weights`) with strict `<PAD>` masking
  - Deterministic disk persistence (`save` / `load`) and graceful error when `torch` is absent
"""

from __future__ import annotations

import gc
import importlib.util
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np

from ml.models.neural_baselines import (
    _calibrate_multilabel_thresholds,
    _calibrate_temperature_np,
    _encode_targets,
    _macro_f1_multilabel,
    _macro_f1_single,
    _sigmoid_np,
    _softmax_np,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
_TOKEN_RE = re.compile(r"(?u)[A-Za-z0-9\u0900-\u0d7f']+")


def require_torch() -> Any:
    """Import torch cleanly or raise an actionable optional-dependency error."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is an optional ML dependency and is not installed in the slim base runtime.\n"
            "Install optional ML extras with:\n  pip install -r backend/requirements-ml.txt"
        ) from exc


@dataclass
class BiGRUAttentionConfig:
    """Configuration for experimental PyTorch BiGRU + Additive Attention classifier."""

    max_vocab: int = 12000
    max_len: int = 128
    embed_dim: int = 128
    hidden_dim: int = 96
    attn_dim: int = 64
    dropout: float = 0.35
    epochs: int = 24
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    max_pos_weight: float = 6.0
    random_state: int = 42
    device: str = "auto"


def _build_bigru_net(
    *,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    attn_dim: int,
    num_classes: int,
    dropout: float,
) -> Any:
    import torch
    import torch.nn as nn

    class _BiGRUAttentionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.emb_drop = nn.Dropout(0.25)
            self.gru = nn.GRU(
                embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True
            )
            gru_out = hidden_dim * 2
            self.attn_proj = nn.Linear(gru_out, attn_dim)
            self.attn_score = nn.Linear(attn_dim, 1)
            self.norm = nn.LayerNorm(gru_out * 2)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(gru_out * 2, num_classes)

        def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mask = x != 0  # (B, L)
            emb = self.emb_drop(self.embedding(x))
            seq_out, _ = self.gru(emb)  # (B, L, 2H)

            raw_scores = self.attn_score(torch.tanh(self.attn_proj(seq_out))).squeeze(-1)  # (B, L)
            raw_scores = raw_scores.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(raw_scores, dim=1) * mask.float()
            denom = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            attn_weights = attn_weights / denom

            attn_ctx = torch.bmm(attn_weights.unsqueeze(1), seq_out).squeeze(1)  # (B, 2H)
            masked_seq = seq_out.masked_fill(~mask.unsqueeze(-1), -1e9)
            max_pool = masked_seq.max(dim=1).values
            max_pool = torch.where(mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(attn_ctx))

            feat = self.drop(self.norm(torch.cat([attn_ctx, max_pool], dim=1)))
            logits = self.fc(feat)
            return logits, attn_weights

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits, _ = self.forward_with_attention(x)
            return logits

    return _BiGRUAttentionNet()


class BiGRUAttentionClassifier:
    """Bidirectional GRU + Masked Additive Self-Attention text classifier."""

    def __init__(
        self,
        *,
        task_name: str = "response",
        multilabel: bool = False,
        config: BiGRUAttentionConfig | None = None,
    ) -> None:
        self.task_name = task_name
        self.multilabel = multilabel
        self.config = config or BiGRUAttentionConfig()
        self.vocab_: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.labels_: List[str] = []
        self.model_: Any = None
        self.device_: Any = None
        self.temperature_: float = 1.0
        self.thresholds_: np.ndarray | None = None
        self.binary_threshold_: float = 0.5
        self.param_count_: int = 0
        self.train_seconds_: float = 0.0
        self.best_epoch_: int = 0
        self.history_: List[Dict[str, float]] = []

    def _build_vocab(self, texts: Sequence[str]) -> None:
        counts: Counter = Counter()
        for t in texts:
            counts.update(_TOKEN_RE.findall(str(t).lower()))
        self.vocab_ = {"<PAD>": 0, "<UNK>": 1}
        min_freq = 1 if len(texts) < 100 else 2
        for tok, freq in counts.most_common(self.config.max_vocab - 2):
            if freq >= min_freq:
                self.vocab_[tok] = len(self.vocab_)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.config.max_len), dtype=np.int64)
        unk = self.vocab_["<UNK>"]
        for i, t in enumerate(texts):
            toks = _TOKEN_RE.findall(str(t).lower())[: self.config.max_len]
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
    ) -> "BiGRUAttentionClassifier":
        torch = require_torch()
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

        self._build_vocab(train_texts)
        X_tr = self._encode_texts(train_texts)
        X_va = self._encode_texts(val_texts)
        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        model = _build_bigru_net(
            vocab_size=len(self.vocab_),
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            attn_dim=self.config.attn_dim,
            num_classes=n_classes,
            dropout=self.config.dropout,
        ).to(self.device_)
        self.param_count_ = sum(p.numel() for p in model.parameters())

        if self.multilabel:
            pos = y_tr.sum(axis=0)
            neg = len(y_tr) - pos
            pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, self.config.max_pos_weight)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pw, dtype=torch.float32, device=self.device_)
            )
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
            if score >= best_score:
                best_score = score
                self.best_epoch_ = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(self.device_) for k, v in best_state.items()})
        self.model_ = model

        # Validation-only temperature + threshold calibration
        va_logits = self._predict_logits(val_texts)
        self.temperature_ = _calibrate_temperature_np(va_logits, y_va, self.multilabel)
        if self.multilabel:
            va_probs = _sigmoid_np(va_logits / self.temperature_)
            tuned = _calibrate_multilabel_thresholds(y_va, va_probs, self.labels_)
            self.thresholds_ = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)
        elif n_classes == 2:
            va_probs = _softmax_np(va_logits / self.temperature_)[:, 1]
            best_tau, best_f1 = 0.5, -1.0
            for tau in [round(0.25 + 0.025 * i, 3) for i in range(21)]:
                pred = (va_probs >= tau).astype(int)
                f1 = _macro_f1_single(y_va, pred, 2)
                if f1 > best_f1:
                    best_f1, best_tau = f1, tau
            self.binary_threshold_ = best_tau

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def _predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        torch = require_torch()
        X = self._encode_texts(texts)
        self.model_.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i : i + 256]).to(self.device_)
                chunks.append(self.model_(xb).cpu().numpy())
        return np.vstack(chunks) if chunks else np.zeros((0, len(self.labels_)), dtype=np.float32)

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
        if len(self.labels_) == 2:
            return [self.labels_[1 if row[1] >= self.binary_threshold_ else 0] for row in probs]
        return [self.labels_[int(np.argmax(row))] for row in probs]

    def get_attention_weights(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(token_ids, attention_weights)` with strict zero mass on `<PAD>` tokens."""
        torch = require_torch()
        X = self._encode_texts(texts)
        self.model_.eval()
        with torch.no_grad():
            _, attn = self.model_.forward_with_attention(torch.from_numpy(X).to(self.device_))
        return X, attn.cpu().numpy()

    def save(self, directory: Path | str) -> int:
        torch = require_torch()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), d / "weights.pt")
        joblib.dump(self.vocab_, d / "vocab.joblib", compress=3)
        meta = {
            "task_name": self.task_name,
            "multilabel": self.multilabel,
            "labels": self.labels_,
            "config": asdict(self.config),
            "vocab_size": len(self.vocab_),
            "temperature": self.temperature_,
            "binary_threshold": self.binary_threshold_,
            "thresholds": self.thresholds_.tolist() if self.thresholds_ is not None else None,
            "param_count": self.param_count_,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())

    @classmethod
    def load(cls, directory: Path | str) -> "BiGRUAttentionClassifier":
        torch = require_torch()
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cfg = BiGRUAttentionConfig(**meta["config"])
        inst = cls(task_name=meta["task_name"], multilabel=meta["multilabel"], config=cfg)
        inst.labels_ = list(meta["labels"])
        inst.vocab_ = joblib.load(d / "vocab.joblib")
        inst.temperature_ = float(meta["temperature"])
        inst.binary_threshold_ = float(meta["binary_threshold"])
        inst.thresholds_ = np.array(meta["thresholds"], dtype=np.float64) if meta["thresholds"] else None
        inst.param_count_ = int(meta["param_count"])
        inst.device_ = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
        model = _build_bigru_net(
            vocab_size=len(inst.vocab_),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            attn_dim=cfg.attn_dim,
            num_classes=len(inst.labels_),
            dropout=cfg.dropout,
        ).to(inst.device_)
        state = torch.load(d / "weights.pt", map_location=inst.device_, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        inst.model_ = model
        return inst

    def cleanup(self) -> None:
        torch = require_torch()
        self.model_ = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
