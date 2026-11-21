"""Upgraded GPU-accelerated MuRIL sequence classifier (v3) for all five RCI NLP tasks.

Supports:
  - Single-label multiclass: ``sentiment``, ``response`` (canonical 6-class order)
  - Single-label binary:     ``solution`` (canonical ``["not_solution", "solution"]``)
  - Multi-label:             ``emotion``, ``issue`` (``BCEWithLogitsLoss`` +
                             per-label ``pos_weight`` + validation-only threshold
                             calibration)

Engineering upgrades (v3):
  1. Automatic local HuggingFace snapshot OR Domain-Adaptive Pretrained (DAPT)
     checkpoint resolution for fast, offline-reproducible runs.
  2. Attention-masked [CLS] + Mean + Max pooling with LayerNorm and 5-branch
     Multi-Sample Dropout (Inoue, 2019) for robust representation of short,
     code-mixed, and typo-noisy Reddit replies.
  3. Layer-wise Learning Rate Decay (LLRD) + decoupled AdamW weight decay +
     linear warmup + cosine LR decay + gradient clipping + single-label label
     smoothing (prevents overfitting on clean training templates).
  4. Automatic Mixed Precision (``bfloat16``/``float16``) on CUDA GPUs.
  5. Validation-driven best-checkpoint restoration, validation-only scalar
     temperature scaling (NLL calibration), and finer validation-only per-label
     threshold calibration for multi-label tasks.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES
from app.services.nlp.solution_detection.models.labels import TARGET_LABELS


DEFAULT_LOCAL_HF_CACHE = (
    Path(
        os.environ.get(
            "RCI_MODEL_CACHE_DIR",
            str(Path(__file__).resolve().parents[3] / ".cache" / "models"),
        )
    )
    / "hf_home"
)


def resolve_local_muril_path(model_id: str = "google/muril-base-cased") -> Tuple[str, bool]:
    """Resolve a locally cached MuRIL snapshot or local DAPT directory if available.

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
    """Hyperparameters for fine-tuning MuRIL (base or domain-adapted DAPT)."""

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
    pooling: str = "cls_mean_max"  # "cls_mean_max" | "cls"
    multi_sample_dropout: int = 5
    dropout_rate: float = 0.2
    llrd: float = 0.90
    head_lr_multiplier: float = 3.0
    label_smoothing: float = 0.05
    calibrate_temperature: bool = True
    use_amp: bool = True
    device: str = "auto"
    random_state: int = 20261116
    verbose: bool = True


class _PooledMultiSampleModel(nn.Module):
    """BERT/MuRIL encoder with [CLS]+Mean+Max attention pooling and Multi-Sample Dropout."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: int,
        *,
        multi_sample_dropout: int = 5,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooled_dim = hidden_size * 3
        self.norm = nn.LayerNorm(self.pooled_dim)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(max(1, multi_sample_dropout))]
        )
        self.classifier = nn.Linear(self.pooled_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            enc_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**enc_kwargs)
        hidden = outputs.last_hidden_state  # [B, L, H]

        cls_vec = hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_vec = (hidden * mask).sum(dim=1) / denom
        masked_hidden = hidden.masked_fill(mask == 0, -1e4)
        max_vec = masked_hidden.max(dim=1).values

        pooled = self.norm(torch.cat([cls_vec, mean_vec, max_vec], dim=-1))
        logits = torch.stack(
            [self.classifier(dp(pooled)) for dp in self.dropouts], dim=0
        ).mean(dim=0)
        return logits


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
    # Finer 33-point grid strictly over held-out validation probabilities
    grid = [round(0.10 + 0.025 * i, 3) for i in range(33)]
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


def _logits_to_probs(logits: np.ndarray, multilabel: bool, temperature: float = 1.0) -> np.ndarray:
    scaled = logits.astype(np.float64) / max(float(temperature), 1e-3)
    if multilabel:
        clipped = np.clip(scaled, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return (exps / np.sum(exps, axis=1, keepdims=True)).astype(np.float32)


def _validation_nll(logits: np.ndarray, y_val: np.ndarray, multilabel: bool, temperature: float) -> float:
    probs = _logits_to_probs(logits, multilabel=multilabel, temperature=temperature)
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    if multilabel:
        return float(-np.mean(y_val * np.log(clipped) + (1.0 - y_val) * np.log(1.0 - clipped)))
    return float(-np.mean(np.log(clipped[np.arange(len(y_val)), y_val])))


def _calibrate_temperature(logits: np.ndarray, y_val: np.ndarray, multilabel: bool) -> float:
    """Find scalar temperature T >= 1.0 minimizing NLL strictly on held-out validation logits.

    Clamping T >= 1.0 prevents calibration from artificially sharpening overconfident logits.
    """
    if len(logits) == 0:
        return 1.0
    coarse = [round(1.0 + 0.1 * i, 2) for i in range(31)]  # 1.0 .. 4.0
    best_t = 1.0
    best_nll = _validation_nll(logits, y_val, multilabel, 1.0)
    for t in coarse:
        nll = _validation_nll(logits, y_val, multilabel, t)
        if nll < best_nll - 1e-6:
            best_nll = nll
            best_t = t
    # Local refinement around best_t, strictly clamped to >= 1.0
    fine = [round(max(1.0, best_t - 0.10 + 0.02 * i), 3) for i in range(11)]
    for t in fine:
        nll = _validation_nll(logits, y_val, multilabel, t)
        if nll < best_nll - 1e-6:
            best_nll = nll
            best_t = t
    return float(max(1.0, best_t))


_LAYER_RE = re.compile(r"\.layer\.(\d+)\.")


def _build_llrd_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    head_lr_multiplier: float,
    llrd: float,
    weight_decay: float,
    num_hidden_layers: int = 12,
) -> List[Dict[str, Any]]:
    """Create AdamW parameter groups with Layer-wise Learning Rate Decay (LLRD)."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        if llrd >= 0.999:
            lr = base_lr * head_lr_multiplier if name.startswith(("classifier", "norm")) else base_lr
        else:
            if name.startswith(("classifier", "norm")):
                lr = base_lr * head_lr_multiplier
            elif "embeddings" in name:
                lr = base_lr * (llrd ** (num_hidden_layers + 1))
            else:
                m = _LAYER_RE.search(name)
                if m:
                    layer_idx = int(m.group(1))
                    lr = base_lr * (llrd ** (num_hidden_layers - layer_idx))
                else:
                    lr = base_lr
        key = (round(lr, 10), wd)
        buckets.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in buckets.items()
    ]


class MuRILTaskClassifier:
    """Unified fine-tuned MuRIL encoder supporting single-label, binary, and multi-label tasks."""

    _NAME = "muril_transformer_v3"
    _VERSION = "v3"

    def __init__(self, config: MuRILTaskConfig) -> None:
        if config.task not in {"sentiment", "emotion", "issue", "response", "solution"}:
            raise ValueError(f"Unsupported task {config.task!r}")
        if config.pooling not in {"cls_mean_max", "cls"}:
            raise ValueError(f"Unsupported pooling {config.pooling!r}")
        self._config = config
        self._multilabel = config.task in {"emotion", "issue"}
        self._device: Optional[torch.device] = None
        self._tokenizer = None
        self._model: Optional[nn.Module] = None
        self._num_hidden_layers: int = 12
        self._labels: List[str] = []
        self._per_label_thresholds: Dict[str, float] = {}
        self._temperature: float = 1.0
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
    def temperature(self) -> float:
        return float(self._temperature)

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
        hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
        self._num_hidden_layers = int(getattr(hf_cfg, "num_hidden_layers", 12))

        if self._config.pooling == "cls_mean_max":
            encoder = AutoModel.from_pretrained(
                model_path,
                config=hf_cfg,
                local_files_only=local_only,
            )
            self._model = _PooledMultiSampleModel(
                encoder=encoder,
                hidden_size=int(hf_cfg.hidden_size),
                num_labels=len(self._labels),
                multi_sample_dropout=self._config.multi_sample_dropout,
                dropout_rate=self._config.dropout_rate,
            ).to(self._device)
        else:
            label2id = {lbl: i for i, lbl in enumerate(self._labels)}
            id2label = {i: lbl for lbl, i in label2id.items()}
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

    def _forward_logits(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self._model is not None
        if isinstance(self._model, _PooledMultiSampleModel):
            return self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
        return self._model(**encoded).logits

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
        self._temperature = 1.0
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
            criterion = torch.nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=float(max(0.0, self._config.label_smoothing)),
            )

        param_groups = _build_llrd_param_groups(
            self._model,
            base_lr=self._config.learning_rate,
            head_lr_multiplier=self._config.head_lr_multiplier,
            llrd=self._config.llrd,
            weight_decay=self._config.weight_decay,
            num_hidden_layers=self._num_hidden_layers,
        )
        optimizer = torch.optim.AdamW(param_groups)

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
        best_val_logits: Optional[np.ndarray] = None

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
                    logits = self._forward_logits(encoded)
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
                val_logits = self.predict_logits(val_texts)
                val_probs = _logits_to_probs(val_logits, multilabel=self._multilabel, temperature=1.0)
                val_loss = _validation_nll(val_logits, y_val, self._multilabel, temperature=1.0)
                if self._multilabel:
                    # Evaluate multi-label checkpoint quality using fixed 0.5 threshold & validation BCE loss
                    # (never select epochs by per-epoch threshold overfitting on small validation splits)
                    val_macro, _ = _macro_f1_multilabel(
                        y_val, val_probs, [self._config.threshold for _ in self._labels]
                    )
                    tuned = dict(self._per_label_thresholds)
                else:
                    val_pred = np.argmax(val_probs, axis=1)
                    val_macro = _macro_f1_single(y_val, val_pred, len(self._labels))
                    tuned = dict(self._per_label_thresholds)

                epoch_info["val_macro_f1"] = round(val_macro, 6)
                epoch_info["val_loss"] = round(val_loss, 6)
                if self._multilabel:
                    is_better = (val_loss < best_val_loss - 1e-4) or (
                        abs(val_loss - best_val_loss) <= 1e-4 and val_macro > best_val_score + 1e-6
                    )
                else:
                    is_better = (val_macro > best_val_score + 1e-6) or (
                        abs(val_macro - best_val_score) <= 1e-6 and val_loss < best_val_loss - 1e-6
                    )
                if is_better:
                    best_val_score = val_macro
                    best_val_loss = val_loss
                    self._best_epoch = epoch
                    self._best_val_macro_f1 = round(val_macro, 6)
                    best_thresholds = dict(tuned)
                    best_val_logits = val_logits
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                if self._config.verbose:
                    print(
                        f"  [MuRIL:{self._config.task}] epoch {epoch}/{self._config.epochs} "
                        f"loss={avg_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_macro:.4f}",
                        flush=True,
                    )
            elif self._config.verbose:
                print(
                    f"  [MuRIL:{self._config.task}] epoch {epoch}/{self._config.epochs} "
                    f"loss={avg_loss:.4f}",
                    flush=True,
                )

            self._history.append(epoch_info)

        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )
            self._per_label_thresholds = best_thresholds

        # Validation-only scalar temperature scaling + recalibration of multi-label thresholds
        if (
            self._config.calibrate_temperature
            and best_val_logits is not None
            and y_val is not None
        ):
            self._temperature = _calibrate_temperature(best_val_logits, y_val, self._multilabel)
            if self._multilabel:
                cal_probs = _logits_to_probs(
                    best_val_logits, multilabel=True, temperature=self._temperature
                )
                self._per_label_thresholds = _calibrate_multilabel_thresholds(
                    y_val, cal_probs, self._labels, self._config.threshold
                )

        self._model.eval()
        self._trained = True

        return {
            "task": self._config.task,
            "model_name": self._NAME,
            "model_version": self._VERSION,
            "pooling": self._config.pooling,
            "temperature": round(self._temperature, 4),
            "device": str(self._device),
            "amp_enabled": amp_enabled,
            "amp_dtype": str(amp_dtype),
            "labels": list(self._labels),
            "per_label_thresholds": dict(self._per_label_thresholds),
            "best_epoch": self._best_epoch,
            "best_val_macro_f1": self._best_val_macro_f1,
            "history": list(self._history),
        }

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
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
                    logits = self._forward_logits(encoded)
                chunks.append(logits.float().cpu().numpy())
        return np.vstack(chunks)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        logits = self.predict_logits(texts)
        if len(logits) == 0:
            return np.zeros((0, len(self._labels)), dtype=np.float32)
        return _logits_to_probs(logits, multilabel=self._multilabel, temperature=self._temperature)

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
        if isinstance(self._model, _PooledMultiSampleModel):
            enc_dir = out_dir / "encoder"
            enc_dir.mkdir(parents=True, exist_ok=True)
            self._model.encoder.save_pretrained(enc_dir)
            self._tokenizer.save_pretrained(out_dir)
            torch.save(
                {
                    "norm": self._model.norm.state_dict(),
                    "classifier": self._model.classifier.state_dict(),
                    "hidden_size": self._model.pooled_dim // 3,
                    "num_labels": len(self._labels),
                    "multi_sample_dropout": len(self._model.dropouts),
                    "dropout_rate": self._config.dropout_rate,
                },
                out_dir / "pooled_head.pt",
            )
            arch_kind = "cls_mean_max"
        else:
            self._model.save_pretrained(out_dir)
            self._tokenizer.save_pretrained(out_dir)
            arch_kind = "cls"

        meta = {
            "task": self._config.task,
            "model_name": self._NAME,
            "model_version": self._VERSION,
            "pooling": arch_kind,
            "temperature": float(self._temperature),
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
        pooling = str(meta.get("pooling", "cls"))
        cfg = MuRILTaskConfig(
            task=meta["task"],
            model_id=str(in_dir),
            max_length=int(meta.get("max_length", 160)),
            pooling=pooling,
            device=device,
        )
        inst = cls(cfg)
        inst._labels = list(meta["labels"])
        inst._per_label_thresholds = {
            str(k): float(v) for k, v in meta.get("per_label_thresholds", {}).items()
        }
        inst._temperature = float(meta.get("temperature", 1.0))
        inst._best_epoch = meta.get("best_epoch")
        inst._best_val_macro_f1 = meta.get("best_val_macro_f1")
        inst._history = list(meta.get("history", []))
        inst._device = inst._resolve_device()
        inst._tokenizer = AutoTokenizer.from_pretrained(str(in_dir), local_files_only=True)

        if pooling == "cls_mean_max" and (in_dir / "pooled_head.pt").is_file():
            head_ckpt = torch.load(in_dir / "pooled_head.pt", map_location="cpu", weights_only=True)
            encoder = AutoModel.from_pretrained(str(in_dir / "encoder"), local_files_only=True)
            model = _PooledMultiSampleModel(
                encoder=encoder,
                hidden_size=int(head_ckpt["hidden_size"]),
                num_labels=int(head_ckpt["num_labels"]),
                multi_sample_dropout=int(head_ckpt.get("multi_sample_dropout", 5)),
                dropout_rate=float(head_ckpt.get("dropout_rate", 0.2)),
            )
            model.norm.load_state_dict(head_ckpt["norm"])
            model.classifier.load_state_dict(head_ckpt["classifier"])
            inst._model = model.to(inst._device)
        else:
            inst._model = AutoModelForSequenceClassification.from_pretrained(
                str(in_dir), local_files_only=True
            ).to(inst._device)

        inst._model.eval()
        inst._trained = True
        return inst
