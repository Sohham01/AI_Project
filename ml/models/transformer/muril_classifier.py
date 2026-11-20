"""Upgraded GPU-accelerated MuRIL sequence classifier for all five RCI NLP tasks.

Supports:
  - Single-label multiclass: ``sentiment``, ``response`` (canonical 6-class order)
  - Single-label binary:     ``solution`` (canonical ``["not_solution", "solution"]``)
  - Multi-label:             ``emotion``, ``issue`` (``BCEWithLogitsLoss`` +
                             per-label ``pos_weight`` + validation-only threshold
                             calibration)

Engineering upgrades over the naive v1 training loop:
  1. Automatic local HuggingFace snapshot resolution (checks ``HF_HOME`` and
     ``D:/AI_Project_Transformer_Temp/hf_home``) for fast, offline-reproducible runs.
  2. Automatic Mixed Precision (``bfloat16``/``float16``) on CUDA GPUs.
  3. Decoupled AdamW weight decay (zero decay on biases & LayerNorm weights),
     linear warmup (10%), cosine learning-rate decay, and gradient clipping.
  4. Validation-driven best-checkpoint restoration so the model keeps the epoch
     with the strongest held-out validation Macro F1 rather than the final epoch.
  5. Strictly validation-only per-label threshold calibration for multi-label tasks.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.solution_detection.models.labels import TARGET_LABELS


DEFAULT_LOCAL_HF_CACHE = Path("D:/AI_Project_Transformer_Temp/hf_home")


def resolve_local_muril_path(model_id: str = "google/muril-base-cased") -> Tuple[str, bool]:
    """Resolve a locally cached MuRIL snapshot if available.

    Returns ``(path_or_id, local_files_only)``.
    """
    explicit_path = Path(model_id)
    if explicit_path.is_dir() and (explicit_path / "config.json").is_file():
        return str(explicit_path), True

    slug = "models--" + model_id.replace("/", "--")
    candidate_roots = []
    if os.environ.get("HF_HOME"):
        candidate_roots.append(Path(os.environ["HF_HOME"]) / "hub" / slug)
    candidate_roots.append(DEFAULT_LOCAL_HF_CACHE / "hub" / slug)
    candidate_roots.append(Path.home() / ".cache" / "huggingface" / "hub" / slug)

    for repo_dir in candidate_roots:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snap in sorted(snapshots_dir.iterdir(), reverse=True):
            has_weights = (
                (snap / "pytorch_model.bin").is_file()
                or (snap / "model.safetensors").is_file()
            )
            if (snap / "config.json").is_file() and (snap / "vocab.txt").is_file() and has_weights:
                return str(snap), True

    return model_id, False


@dataclass
class MuRILTaskConfig:
    """Hyperparameters for fine-tuning ``google/muril-base-cased``."""

    task: str  # sentiment | emotion | issue | response | solution
    model_id: str = "google/muril-base-cased"
    max_length: int = 160
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 6
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    threshold: float = 0.5
    use_class_weights: bool = True
    max_pos_weight: float = 8.0
    use_amp: bool = True
    device: str = "auto"
    random_state: int = 20261116
    verbose: bool = True


class _EncodedTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], targets: Sequence[Any]) -> None:
        self.texts = list(texts)
        self.targets = list(targets)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        return self.texts[idx], self.targets[idx]


def _macro_f1_single(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())
        if support == 0 and (tp + fp) == 0:
            continue
        denom = 2 * tp + fp + fn
        scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _macro_f1_multilabel(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Sequence[float],
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
    macro = float(np.mean(list(per_col.values()))) if per_col else 0.0
    return macro, per_col


def _calibrate_multilabel_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: Sequence[str],
    default_threshold: float = 0.5,
) -> Dict[str, float]:
    grid = [round(0.15 + 0.05 * i, 2) for i in range(15)]
    tuned: Dict[str, float] = {}
    for col, label in enumerate(labels):
        truth = y_true[:, col]
        if int(truth.sum()) == 0:
            tuned[label] = default_threshold
            continue
        best_tau = default_threshold
        best_f1 = -1.0
        col_probs = probs[:, col]
        for tau in grid:
            pred = (col_probs > tau).astype(int)
            tp = int(((truth == 1) & (pred == 1)).sum())
            fp = int(((truth == 0) & (pred == 1)).sum())
            fn = int(((truth == 1) & (pred == 0)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            if (f1 > best_f1 + 1e-12) or (
                abs(f1 - best_f1) <= 1e-12
                and abs(tau - default_threshold) < abs(best_tau - default_threshold)
            ):
                best_f1 = f1
                best_tau = tau
        tuned[label] = best_tau
    return tuned


class MuRILTaskClassifier:
    """Unified fine-tuned MuRIL encoder supporting single-label, binary, and multi-label tasks."""

    _NAME = "muril_transformer_v2"
    _VERSION = "v2"

    def __init__(self, config: MuRILTaskConfig) -> None:
        if config.task not in {"sentiment", "emotion", "issue", "response", "solution"}:
            raise ValueError(f"Unsupported task {config.task!r}")
        self._config = config
        self._multilabel = config.task in {"emotion", "issue"}
        self._device: Optional[torch.device] = None
        self._tokenizer = None
        self._model = None
        self._labels: List[str] = []
        self._per_label_thresholds: Dict[str, float] = {}
        self._trained = False
        self._history: List[Dict[str, Any]] = []
        self._best_epoch: Optional[int] = None
        self._best_val_macro_f1: Optional[float] = None
        self._resolved_model_path: Optional[str] = None

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def task(self) -> str:
        return self._config.task

    @property
    def is_multilabel(self) -> bool:
        return self._multilabel

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def per_label_thresholds(self) -> Dict[str, float]:
        return dict(self._per_label_thresholds)

    @property
    def is_trained(self) -> bool:
        return self._trained and self._model is not None

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def _resolve_device(self) -> torch.device:
        req = self._config.device.strip().lower()
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(req)

    def _canonical_labels(self, raw_targets: Sequence[Any]) -> List[str]:
        if self._config.task == "response":
            return list(RESPONSE_TYPES)
        if self._config.task == "solution":
            return list(TARGET_LABELS)
        if self._multilabel:
            return sorted({str(lbl) for row in raw_targets for lbl in row})
        return sorted({str(lbl) for lbl in raw_targets})

    def _encode_targets(self, raw_targets: Sequence[Any]) -> np.ndarray:
        label_to_idx = {lbl: idx for idx, lbl in enumerate(self._labels)}
        if self._multilabel:
            matrix = np.zeros((len(raw_targets), len(self._labels)), dtype=np.float32)
            for row_idx, row_labels in enumerate(raw_targets):
                for lbl in row_labels:
                    if lbl in label_to_idx:
                        matrix[row_idx, label_to_idx[lbl]] = 1.0
            return matrix
        return np.array([label_to_idx[str(lbl)] for lbl in raw_targets], dtype=np.int64)

    def _init_model(self) -> None:
        self._device = self._resolve_device()
        model_path, local_only = resolve_local_muril_path(self._config.model_id)
        self._resolved_model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_only
        )
        label2id = {lbl: i for i, lbl in enumerate(self._labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
        hf_cfg.num_labels = len(self._labels)
        hf_cfg.label2id = label2id
        hf_cfg.id2label = id2label
        hf_cfg.problem_type = (
            "multi_label_classification"
            if self._multilabel
            else "single_label_classification"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=hf_cfg,
            local_files_only=local_only,
        ).to(self._device)

    def _use_cuda_amp(self) -> Tuple[bool, torch.dtype]:
        if not self._config.use_amp or self._device is None or self._device.type != "cuda":
            return False, torch.float32
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Fine-tune MuRIL on ``train_texts`` with validation-selected checkpointing."""
        if not train_texts or len(train_texts) != len(train_targets):
            raise ValueError("train_texts and train_targets must be non-empty and aligned")

        self._labels = self._canonical_labels(train_targets)
        self._per_label_thresholds = {lbl: self._config.threshold for lbl in self._labels}
        self._history = []
        self._best_epoch = None
        self._best_val_macro_f1 = None

        torch.manual_seed(self._config.random_state)
        np.random.seed(self._config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.random_state)

        self._init_model()
        assert self._model is not None and self._tokenizer is not None and self._device is not None

        y_train = self._encode_targets(train_targets)
        y_val = (
            self._encode_targets(val_targets)
            if (val_texts is not None and val_targets is not None and len(val_texts) > 0)
            else None
        )

        dataset = _EncodedTextDataset(train_texts, y_train)
        generator = torch.Generator().manual_seed(self._config.random_state)

        def collate_fn(items):
            batch_texts, batch_targets = zip(*items)
            encoded = self._tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            if self._multilabel:
                t_tensor = torch.tensor(np.asarray(batch_targets), dtype=torch.float32)
            else:
                t_tensor = torch.tensor(np.asarray(batch_targets), dtype=torch.long)
            return encoded, t_tensor

        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_fn,
        )

        if self._multilabel:
            pos_weight = None
            if self._config.use_class_weights:
                pos_counts = np.maximum(y_train.sum(axis=0), 1.0)
                neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
                weights = np.minimum(neg_counts / pos_counts, self._config.max_pos_weight)
                pos_weight = torch.tensor(weights, dtype=torch.float32, device=self._device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            class_weights = None
            if self._config.use_class_weights:
                counts = Counter(int(v) for v in y_train)
                n_total = len(y_train)
                n_cls = len(self._labels)
                weights = [
                    n_total / (n_cls * max(counts.get(i, 1), 1))
                    for i in range(n_cls)
                ]
                class_weights = torch.tensor(weights, dtype=torch.float32, device=self._device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Decoupled weight decay: exclude bias and LayerNorm parameters
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        param_groups = [
            {
                "params": [
                    p for n, p in self._model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self._config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self._model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self._config.learning_rate)

        total_steps = max(len(loader) * self._config.epochs, 1)
        warmup_steps = max(int(total_steps * self._config.warmup_ratio), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        amp_enabled, amp_dtype = self._use_cuda_amp()
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_thresholds: Dict[str, float] = dict(self._per_label_thresholds)
        best_val_score = -1.0
        best_val_loss = float("inf")

        for epoch in range(1, self._config.epochs + 1):
            self._model.train()
            total_loss = 0.0
            seen = 0
            for encoded, target_tensor in loader:
                encoded = {k: v.to(self._device, non_blocking=True) for k, v in encoded.items()}
                target_tensor = target_tensor.to(self._device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=self._device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = self._model(**encoded).logits
                    loss = criterion(logits, target_tensor)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.max_grad_norm
                    )
                    optimizer.step()
                scheduler.step()

                batch_n = int(target_tensor.shape[0])
                total_loss += float(loss.detach().item()) * batch_n
                seen += batch_n

            avg_loss = total_loss / max(seen, 1)
            epoch_info: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(avg_loss, 6),
            }

            if val_texts is not None and y_val is not None:
                val_probs = self.predict_proba(val_texts)
                clipped = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
                if self._multilabel:
                    val_loss = float(
                        -np.mean(
                            y_val * np.log(clipped) + (1.0 - y_val) * np.log(1.0 - clipped)
                        )
                    )
                    tuned = _calibrate_multilabel_thresholds(
                        y_val, val_probs, self._labels, self._config.threshold
                    )
                    val_macro, _ = _macro_f1_multilabel(
                        y_val, val_probs, [tuned[lbl] for lbl in self._labels]
                    )
                else:
                    val_loss = float(
                        -np.mean(np.log(clipped[np.arange(len(y_val)), y_val]))
                    )
                    val_pred = np.argmax(val_probs, axis=1)
                    val_macro = _macro_f1_single(y_val, val_pred, len(self._labels))
                    tuned = dict(self._per_label_thresholds)

                epoch_info["val_macro_f1"] = round(val_macro, 6)
                epoch_info["val_loss"] = round(val_loss, 6)
                is_better = (val_macro > best_val_score + 1e-6) or (
                    abs(val_macro - best_val_score) <= 1e-6 and val_loss < best_val_loss - 1e-6
                )
                if is_better:
                    best_val_score = val_macro
                    best_val_loss = val_loss
                    self._best_epoch = epoch
                    self._best_val_macro_f1 = round(val_macro, 6)
                    best_thresholds = dict(tuned)
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                if self._config.verbose:
                    print(
                        f"  [MuRIL:{self._config.task}] epoch {epoch}/{self._config.epochs} "
                        f"loss={avg_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_macro:.4f}"
                    )
            elif self._config.verbose:
                print(
                    f"  [MuRIL:{self._config.task}] epoch {epoch}/{self._config.epochs} "
                    f"loss={avg_loss:.4f}"
                )

            self._history.append(epoch_info)

        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )
            self._per_label_thresholds = best_thresholds

        self._model.eval()
        self._trained = True

        return {
            "task": self._config.task,
            "model_name": self._NAME,
            "model_version": self._VERSION,
            "device": str(self._device),
            "amp_enabled": amp_enabled,
            "amp_dtype": str(amp_dtype),
            "labels": list(self._labels),
            "per_label_thresholds": dict(self._per_label_thresholds),
            "best_epoch": self._best_epoch,
            "best_val_macro_f1": self._best_val_macro_f1,
            "history": list(self._history),
        }

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._device is None:
            raise RuntimeError("Model has not been initialized or trained")
        if not texts:
            return np.zeros((0, len(self._labels)), dtype=np.float32)

        self._model.eval()
        amp_enabled, amp_dtype = self._use_cuda_amp()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._config.eval_batch_size):
                batch = list(texts[start : start + self._config.eval_batch_size])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device, non_blocking=True) for k, v in encoded.items()}
                with torch.amp.autocast(
                    device_type=self._device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = self._model(**encoded).logits
                if self._multilabel:
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                chunks.append(probs)
        return np.vstack(chunks)

    def predict_labels(
        self,
        texts: Sequence[str],
        threshold: Optional[float] = None,
    ) -> Tuple[List[Any], np.ndarray]:
        """Return ``(predicted_labels, probability_matrix)`` using calibrated thresholds."""
        probs = self.predict_proba(texts)
        if self._multilabel:
            outputs: List[List[str]] = []
            for row in probs:
                active = [
                    lbl
                    for idx, lbl in enumerate(self._labels)
                    if float(row[idx]) > (
                        threshold
                        if threshold is not None
                        else self._per_label_thresholds.get(lbl, self._config.threshold)
                    )
                ]
                outputs.append(sorted(active))
            return outputs, probs

        indices = np.argmax(probs, axis=1)
        return [self._labels[int(i)] for i in indices], probs

    def save(self, directory: str | Path) -> Path:
        if not self.is_trained or self._model is None or self._tokenizer is None:
            raise RuntimeError("Cannot save an untrained MuRILTaskClassifier")
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out_dir)
        self._tokenizer.save_pretrained(out_dir)
        meta = {
            "task": self._config.task,
            "model_name": self._NAME,
            "model_version": self._VERSION,
            "multilabel": self._multilabel,
            "labels": self._labels,
            "per_label_thresholds": self._per_label_thresholds,
            "max_length": self._config.max_length,
            "best_epoch": self._best_epoch,
            "best_val_macro_f1": self._best_val_macro_f1,
            "history": self._history,
        }
        (out_dir / "rci_metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(cls, directory: str | Path, device: str = "auto") -> "MuRILTaskClassifier":
        in_dir = Path(directory)
        meta = json.loads((in_dir / "rci_metadata.json").read_text(encoding="utf-8"))
        cfg = MuRILTaskConfig(
            task=meta["task"],
            model_id=str(in_dir),
            max_length=int(meta.get("max_length", 160)),
            device=device,
        )
        inst = cls(cfg)
        inst._labels = list(meta["labels"])
        inst._per_label_thresholds = {
            str(k): float(v) for k, v in meta.get("per_label_thresholds", {}).items()
        }
        inst._best_epoch = meta.get("best_epoch")
        inst._best_val_macro_f1 = meta.get("best_val_macro_f1")
        inst._history = list(meta.get("history", []))
        inst._device = inst._resolve_device()
        inst._tokenizer = AutoTokenizer.from_pretrained(str(in_dir), local_files_only=True)
        inst._model = AutoModelForSequenceClassification.from_pretrained(
            str(in_dir), local_files_only=True
        ).to(inst._device)
        inst._model.eval()
        inst._trained = True
        return inst
