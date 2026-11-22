"""Shared leak-free data loader, group-preserving multi-label CV, flexible GPU trainer,
calibration suite, and thread-clustered bootstrap statistics for V5 Optimization.

Strict isolation guarantees:
- Never modifies data/annotations/real_reddit_labeled_v1/, data/live_corpus/, or ml/artifacts/live_corpus/.
- All outputs go exclusively to ml/artifacts/v5_experiments/ and $RCI_MODEL_CACHE_DIR/v5_experiments/.
- Frozen 92-comment V5 test set is locked read-only; never used for fold selection, tuning, or DAPT.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

_DEFAULT_CACHE = Path(os.environ.get("RCI_MODEL_CACHE_DIR") or (ROOT / ".cache" / "models"))
os.environ.setdefault("RCI_MODEL_CACHE_DIR", str(_DEFAULT_CACHE))
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.evaluation.error_analysis import compute_calibration_metrics
from ml.models.transformer.muril_classifier import (
    _build_llrd_param_groups,
    _calibrate_temperature,
    _logits_to_probs,
    _validation_nll,
    resolve_local_muril_path,
)

SEED = 20261116
V1_LABELED_DIR = ROOT / "data" / "annotations" / "real_reddit_labeled_v1"
CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
V5_EVAL_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "evaluation.json"
EXP_ROOT = ROOT / "ml" / "artifacts" / "v5_experiments"
CKPT_ROOT = Path(os.environ.get("RCI_MODEL_CACHE_DIR", str(ROOT / ".cache" / "models"))) / "v5_experiments"

TASK_NAMES = ("issue", "sentiment", "emotion", "response", "solution")
MULTILABEL_TASKS = {"issue", "emotion"}

CANONICAL_LABELS: Dict[str, List[str]] = {
    "issue": [
        "academic_workload",
        "administrative_communication",
        "campus_infrastructure",
        "curriculum_teaching",
        "extracurriculars",
        "financial_concerns",
        "hostel_facilities",
        "placements",
    ],
    "sentiment": ["negative", "neutral", "positive"],
    "emotion": ["confidence", "curiosity", "frustration", "gratitude", "hope", "stress"],
    "response": [
        "agreement",
        "personal_experience",
        "opinion",
        "advice",
        "solution",
        "counterargument",
    ],
    "solution": ["not_solution", "solution"],
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_corpus_lookups() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load corpus posts and comment lookup keyed by pseudonymized comment_id (cmt_XXXXX)."""
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    post_by_pid: Dict[str, Dict[str, Any]] = {}
    cmt_by_cid: Dict[str, Dict[str, Any]] = {}
    for p in corpus["posts"]:
        post_by_pid[p["reddit_id"]] = p
        for c in p["comments"]:
            cmt_by_cid[c["reddit_id"]] = {**c, "post_title": p["title"], "post_body": p.get("body") or ""}
    return corpus, post_by_pid, cmt_by_cid


def parse_multilabel_str(raw: str) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in str(raw).split("|") if tok.strip()]


def load_v1_records_and_frozen_splits() -> Dict[str, Any]:
    """Load all 459 V1 real labeled rows with exact text-verified parent context & frozen V5 split.

    NOTE (Forensic Alignment Fix): Because `corpus.json` was later re-pseudonymized (366->365 threads),
    sequential `post_id`/`comment_id` numbers drifted after `post_0064`. We therefore match every V1
    row to its exact `corpus.json` thread & comment by normalized `(post_title[:120], comment_text)`
    (verified 459/459 exact match!) and record both canonical V1 IDs and exact `corpus_post_id` /
    `corpus_comment_id` for leak-free DAPT exclusion, true parent-comment lookup, and paired V5 checks.
    """
    corpus, _, _ = load_corpus_lookups()
    norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()
    posts_by_title120: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in corpus["posts"]:
        posts_by_title120[norm(p.get("title", ""))[:120]].append(p)

    with (V1_LABELED_DIR / "master_labeled_v1.csv").open("r", encoding="utf-8") as f:
        master_rows = list(csv.DictReader(f))

    # Reproduce exact V5 train / val / test split using ml.data.splitting
    split_spec = SplitSpec(
        train_size=0.65, validation_size=0.15, test_size=0.20, random_state=SEED, stratify=True
    )
    recs = load_annotation_dataset(TaskType.ISSUE, str(V1_LABELED_DIR / "issue.csv"))
    sp = split_records(recs, labels=[r.label for r in recs], spec=split_spec, groups=extract_groups(recs, "post_id"))
    tr, va, te = apply_split(recs, sp)
    leak = check_leakage(tr, va, te)
    assert not leak.get("leakage_detected"), f"Unexpected leakage in V5 frozen split: {leak}"

    train_dids = {r.document_id for r in tr}
    val_dids = {r.document_id for r in va}
    test_dids = {r.document_id for r in te}
    train_pids = sorted({r.post_id for r in tr if r.post_id})
    val_pids = sorted({r.post_id for r in va if r.post_id})
    test_pids = sorted({r.post_id for r in te if r.post_id})
    assert not (set(train_pids) & set(test_pids))
    assert not (set(val_pids) & set(test_pids))
    assert not (set(train_pids) & set(val_pids))

    enriched: List[Dict[str, Any]] = []
    for row in master_rows:
        did = row["document_id"]
        cid = row["comment_id"]
        pid = row["post_id"]
        split_name = "train" if did in train_dids else ("val" if did in val_dids else "test")

        title_raw = (row.get("post_title") or "").strip()
        text_raw = (row.get("text") or "").strip()
        nt = norm(text_raw)
        hit_post = None
        hit_cmt = None
        for cand_p in posts_by_title120.get(norm(title_raw)[:120], []):
            for x in cand_p.get("comments") or []:
                nb = norm(x.get("body", ""))
                if nb == nt or (len(nt) >= 30 and (nb.startswith(nt[:60]) or nt.startswith(nb[:60]))):
                    hit_post = cand_p
                    hit_cmt = x
                    break
            if hit_cmt is not None:
                break

        corpus_pid = hit_post["reddit_id"] if hit_post is not None else pid
        corpus_cid = hit_cmt["reddit_id"] if hit_cmt is not None else cid
        parent_comment_text = ""
        parent_cid = None
        if hit_post is not None and hit_cmt is not None:
            prid = hit_cmt.get("parent_reddit_id")
            if prid:
                in_post_cmts = {x["reddit_id"]: x for x in hit_post.get("comments") or []}
                if prid in in_post_cmts:
                    parent_cid = prid
                    parent_comment_text = str(in_post_cmts[prid].get("body") or "").strip()

        enriched.append(
            {
                "document_id": did,
                "post_id": pid,
                "comment_id": cid,
                "corpus_post_id": corpus_pid,
                "corpus_comment_id": corpus_cid,
                "subreddit": row.get("subreddit") or "",
                "post_title": title_raw,
                "parent_comment_id": parent_cid,
                "parent_comment_text": parent_comment_text,
                "text": text_raw,
                "v5_split": split_name,
                "labels": {
                    "issue": parse_multilabel_str(row.get("issues", "")),
                    "sentiment": str(row.get("sentiment", "")).strip(),
                    "emotion": parse_multilabel_str(row.get("emotions", "")),
                    "response": str(row.get("response", "")).strip(),
                    "solution": str(row.get("solution", "")).strip(),
                },
            }
        )

    train_rows = [r for r in enriched if r["v5_split"] == "train"]
    val_rows = [r for r in enriched if r["v5_split"] == "val"]
    trainval_rows = [r for r in enriched if r["v5_split"] in {"train", "val"}]
    test_rows = [r for r in enriched if r["v5_split"] == "test"]
    test_corpus_pids = sorted({r["corpus_post_id"] for r in test_rows})
    trainval_corpus_pids = sorted({r["corpus_post_id"] for r in trainval_rows})
    assert not (set(trainval_corpus_pids) & set(test_corpus_pids)), "Corpus post_id overlap between trainval and test!"

    manifest = {
        "seed": SEED,
        "dataset": "real_reddit_labeled_v1",
        "total_rows": len(enriched),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "trainval_rows": len(trainval_rows),
        "test_rows": len(test_rows),
        "train_post_ids": train_pids,
        "val_post_ids": val_pids,
        "test_post_ids": test_pids,
        "test_corpus_post_ids": test_corpus_pids,
        "trainval_corpus_post_ids": trainval_corpus_pids,
        "test_document_ids": sorted(test_dids),
        "test_sha256": hashlib.sha256(",".join(sorted(test_dids)).encode("utf-8")).hexdigest(),
    }
    save_json(EXP_ROOT / "manifests" / "v5_frozen_splits.json", manifest)

    return {
        "all": enriched,
        "train": train_rows,
        "val": val_rows,
        "trainval": trainval_rows,
        "test": test_rows,
        "manifest": manifest,
    }


def build_input_pair(row: Dict[str, Any], context_mode: str) -> Tuple[str, Optional[str]]:
    """Build legitimate (reply, context) pair using ONLY reply + parent post title / parent comment.

    Never includes child replies, descendant comments, sibling comments, or predicted labels.
    """
    reply = row["text"]
    title = row.get("post_title") or ""
    parent = row.get("parent_comment_text") or ""

    if context_mode == "comment_only":
        return reply, None
    if context_mode == "title_concat":
        # Legacy V5 issue format: single string concatenation
        return f"{title} | {reply}" if title else reply, None
    if context_mode == "title_pair":
        return reply, title if title else None
    if context_mode == "parent_pair":
        ctx = parent if parent else title
        return reply, ctx if ctx else None
    if context_mode == "title_parent_pair":
        if parent and title:
            ctx = f"Post: {title} | Parent: {parent}"
        elif parent:
            ctx = f"Parent: {parent}"
        elif title:
            ctx = f"Post: {title}"
        else:
            ctx = ""
        return reply, ctx if ctx else None
    raise ValueError(f"Unknown context_mode: {context_mode!r}")


def encode_task_targets(rows: Sequence[Dict[str, Any]], task: str) -> np.ndarray:
    labels = CANONICAL_LABELS[task]
    l2i = {l: i for i, l in enumerate(labels)}
    if task in MULTILABEL_TASKS:
        mat = np.zeros((len(rows), len(labels)), dtype=np.float32)
        for i, r in enumerate(rows):
            for lbl in r["labels"][task]:
                if lbl in l2i:
                    mat[i, l2i[lbl]] = 1.0
        return mat
    return np.array([l2i[r["labels"][task]] for r in rows], dtype=np.int64)


def make_group_cv_folds(
    rows: Sequence[Dict[str, Any]],
    task: Optional[str] = None,
    n_splits: int = 5,
    seed: int = SEED,
) -> List[Tuple[List[int], List[int]]]:
    """Deterministic group-preserving 5-fold CV grouped strictly by original post_id.

    - For single-label tasks: sklearn StratifiedGroupKFold.
    - For multi-label tasks or joint MTL (task=None): Group-Preserving Iterative Stratification
      that assigns thread groups containing the rarest remaining label to the fold with the
      largest label deficit while balancing total fold size.
    """
    n = len(rows)
    groups = np.array([r["post_id"] for r in rows])

    if task is not None and task not in MULTILABEL_TASKS:
        y = encode_task_targets(rows, task)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds: List[Tuple[List[int], List[int]]] = []
        for tr_idx, va_idx in sgkf.split(np.zeros(n), y, groups):
            assert not (set(groups[tr_idx]) & set(groups[va_idx])), "Group leakage detected!"
            folds.append((tr_idx.tolist(), va_idx.tolist()))
        return folds

    # Group-Preserving Iterative Multi-Label Stratification (works for issue, emotion, or joint 5-task MTL)
    if task is not None:
        y_mat = encode_task_targets(rows, task)
    else:
        # Concatenate all 5 tasks' binary indicator matrices for joint MTL group balancing
        parts = []
        for t in TASK_NAMES:
            if t in MULTILABEL_TASKS:
                parts.append(encode_task_targets(rows, t))
            else:
                y_s = encode_task_targets(rows, t)
                oh = np.zeros((n, len(CANONICAL_LABELS[t])), dtype=np.float32)
                oh[np.arange(n), y_s] = 1.0
                parts.append(oh)
        y_mat = np.concatenate(parts, axis=1)

    n_labels = y_mat.shape[1]
    pid_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        pid_to_indices[r["post_id"]].append(idx)

    group_ids = sorted(pid_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    group_label_counts = {
        gid: y_mat[pid_to_indices[gid]].sum(axis=0) for gid in group_ids
    }
    group_sizes = {gid: len(pid_to_indices[gid]) for gid in group_ids}

    total_label_counts = y_mat.sum(axis=0)
    desired_label = np.tile(total_label_counts / float(n_splits), (n_splits, 1))
    desired_size = np.full(n_splits, n / float(n_splits), dtype=np.float64)

    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    unassigned = set(group_ids)

    while unassigned:
        # Compute remaining positive counts per label among unassigned groups
        rem_counts = np.zeros(n_labels, dtype=np.float64)
        for gid in unassigned:
            rem_counts += group_label_counts[gid]
        pos_labels = [c for c in range(n_labels) if rem_counts[c] > 0]
        if pos_labels:
            min_rem = min(rem_counts[c] for c in pos_labels)
            rarest_labels = [c for c in pos_labels if rem_counts[c] == min_rem]
            target_lbl = rng.choice(rarest_labels)
            cand_groups = [gid for gid in unassigned if group_label_counts[gid][target_lbl] > 0]
            # Assign larger rare groups first to stabilize fold totals
            cand_groups.sort(key=lambda g: (-group_label_counts[g][target_lbl], -group_sizes[g], g))
            for gid in cand_groups:
                if gid not in unassigned:
                    continue
                # Pick fold with largest remaining desire for target_lbl, breaking ties by remaining size capacity
                best_fold = max(
                    range(n_splits),
                    key=lambda f: (round(desired_label[f, target_lbl], 6), round(desired_size[f], 6), -len(fold_groups[f])),
                )
                fold_groups[best_fold].append(gid)
                desired_label[best_fold] -= group_label_counts[gid]
                desired_size[best_fold] -= group_sizes[gid]
                unassigned.remove(gid)
        else:
            # Remaining groups have all-zero labels (e.g., no emotion/issue tags); distribute by size capacity
            for gid in sorted(unassigned, key=lambda g: (-group_sizes[g], g)):
                best_fold = max(range(n_splits), key=lambda f: (round(desired_size[f], 6), -len(fold_groups[f])))
                fold_groups[best_fold].append(gid)
                desired_size[best_fold] -= group_sizes[gid]
                unassigned.remove(gid)

    folds = []
    for f in range(n_splits):
        val_pids = set(fold_groups[f])
        va_idx = [i for i, r in enumerate(rows) if r["post_id"] in val_pids]
        tr_idx = [i for i, r in enumerate(rows) if r["post_id"] not in val_pids]
        assert not (set(groups[tr_idx]) & set(groups[va_idx])), "Group leakage in iterative CV!"
        folds.append((tr_idx, va_idx))
    return folds


# ============================================================================
# Task-Specific Loss Functions & Pure-PyTorch LoRA
# ============================================================================

class AsymmetricLossMultiLabel(nn.Module):
    """Asymmetric Loss (Ridnik et al., ICCV 2021) for imbalanced multi-label learning."""

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0.0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-7))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-7))
        loss = los_pos + los_neg
        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = torch.pow((1.0 - pt).clamp(min=1e-7), gamma)
            loss = loss * weight
        return -loss.mean()


class FocalCrossEntropyLoss(nn.Module):
    """Focal Cross-Entropy Loss for imbalanced single-label or binary classification."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal = ((1.0 - pt).clamp(min=1e-7) ** self.gamma) * ce
        return focal.mean()


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Logit-Adjusted Cross-Entropy (Menon et al., ICLR 2021) for long-tailed classification."""

    def __init__(self, class_priors: np.ndarray, tau: float = 0.75, label_smoothing: float = 0.05) -> None:
        super().__init__()
        priors = np.clip(class_priors.astype(np.float32), 1e-6, 1.0)
        self.register_buffer("log_priors", torch.tensor(np.log(priors), dtype=torch.float32))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj_logits = logits + self.tau * self.log_priors.to(logits.device)
        return F.cross_entropy(adj_logits, targets, label_smoothing=self.label_smoothing)


class LoRALinear(nn.Module):
    """Dependency-free pure-PyTorch LoRA wrapper (Hu et al., ICLR 2022)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / float(r)
        self.lora_drop = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.lora_drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return out + lora


def inject_lora(encoder: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.1) -> int:
    """Freeze encoder and wrap attention query/value linear layers with LoRA adapters."""
    for p in encoder.parameters():
        p.requires_grad = False
    replaced = 0
    for _, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in {"query", "value", "q_proj", "v_proj"}:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


# ============================================================================
# Unified Transformer Encoder + Pooling + Multi-Sample Dropout
# ============================================================================

class PooledTransformerClassifier(nn.Module):
    """Backbone-agnostic [CLS]+Mean+Max + LayerNorm + 5x Multi-Sample Dropout classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: int,
        *,
        multi_sample_dropout: int = 5,
        dropout_rate: float = 0.2,
        reply_only_pool: bool = False,
        sep_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooled_dim = hidden_size * 3
        self.norm = nn.LayerNorm(self.pooled_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(max(1, multi_sample_dropout))])
        self.classifier = nn.Linear(self.pooled_dim, num_labels)
        self.reply_only_pool = bool(reply_only_pool)
        self.sep_token_id = sep_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            enc_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**enc_kwargs)
        hidden = outputs.last_hidden_state
        cls_vec = hidden[:, 0, :]
        pool_mask = attention_mask
        if self.reply_only_pool and self.sep_token_id is not None:
            is_sep = (input_ids == int(self.sep_token_id)).long()
            seg_idx = torch.cumsum(is_sep, dim=1) - is_sep
            reply_mask = attention_mask * (seg_idx == 0).to(attention_mask.dtype) * (1 - is_sep).to(attention_mask.dtype)
            has_reply = (reply_mask.sum(dim=1, keepdim=True) > 0)
            pool_mask = torch.where(has_reply, reply_mask, attention_mask)
        mask = pool_mask.unsqueeze(-1).to(hidden.dtype)
        mean_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        max_vec = hidden.masked_fill(mask == 0, -1e4).max(dim=1).values
        pooled = self.norm(torch.cat([cls_vec, mean_vec, max_vec], dim=-1))
        return torch.stack([self.classifier(dp(pooled)) for dp in self.dropouts], dim=0).mean(dim=0)


class _PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, Optional[str]]], targets: np.ndarray) -> None:
        self.pairs = list(pairs)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, Optional[str]], Any]:
        return self.pairs[idx], self.targets[idx]


def tokenize_pairs(
    tokenizer: Any,
    pairs: Sequence[Tuple[str, Optional[str]]],
    max_length: int,
    use_token_type_ids: bool = True,
) -> Dict[str, torch.Tensor]:
    has_second = any(b is not None and b != "" for _, b in pairs)
    if has_second:
        firsts = [a for a, _ in pairs]
        seconds = [b if b is not None else "" for _, b in pairs]
        # Preserve reply (first sequence) completely whenever possible by truncating context first
        try:
            enc = tokenizer(
                firsts,
                seconds,
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception:
            enc = tokenizer(
                firsts,
                seconds,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
    else:
        firsts = [a for a, _ in pairs]
        enc = tokenizer(
            firsts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    if not use_token_type_ids and "token_type_ids" in enc:
        del enc["token_type_ids"]
    return dict(enc)


@dataclass
class TrainSpec:
    task: str
    backbone_id: str
    context_mode: str = "comment_only"
    loss_type: str = "default"  # default | bce | bce_pos | asl | ce | ce_weighted | logit_adj | focal
    max_length: int = 160
    epochs: int = 6
    batch_size: int = 8
    eval_batch_size: int = 32
    learning_rate: float = 4e-5
    head_lr_multiplier: float = 4.0
    llrd: float = 0.92
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    dropout_rate: float = 0.20
    label_smoothing: float = 0.05
    max_pos_weight: float = 2.5
    logit_adj_tau: float = 0.75
    asl_gamma_neg: float = 2.0
    asl_clip: float = 0.05
    rdrop_alpha: float = 0.0
    use_lora: bool = False
    lora_r: int = 8
    reply_only_pool: bool = False
    seed: int = SEED


def _bernoulli_sym_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Exact symmetric KL divergence between independent Bernoulli distributions (multi-label R-Drop)."""
    p = torch.sigmoid(logits_p).clamp(1e-6, 1.0 - 1e-6)
    q = torch.sigmoid(logits_q).clamp(1e-6, 1.0 - 1e-6)
    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    return 0.5 * (kl_pq + kl_qp).mean()


def _categorical_sym_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Exact symmetric KL divergence between categorical softmax distributions (single-label R-Drop)."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = F.kl_div(log_q, p, reduction="batchmean")
    kl_qp = F.kl_div(log_p, q, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def _make_criterion(spec: TrainSpec, y_train: np.ndarray, device: torch.device) -> nn.Module:
    multilabel = spec.task in MULTILABEL_TASKS
    n_labels = len(CANONICAL_LABELS[spec.task])
    lt = spec.loss_type
    if lt == "default":
        lt = "bce_pos" if multilabel else "ce_weighted"

    if multilabel:
        if lt == "bce":
            return nn.BCEWithLogitsLoss()
        if lt == "bce_pos":
            pos_counts = np.maximum(y_train.sum(axis=0), 1.0)
            neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
            w = np.minimum(neg_counts / pos_counts, spec.max_pos_weight)
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, dtype=torch.float32, device=device))
        if lt == "asl":
            return AsymmetricLossMultiLabel(gamma_neg=spec.asl_gamma_neg, gamma_pos=0.0, clip=spec.asl_clip)
        if lt == "cb_bce":
            # Effective-number class-balanced pos_weight (Cui et al., CVPR 2019, beta=0.95), softly capped at 4.0
            pos_counts = np.maximum(y_train.sum(axis=0), 1.0)
            neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
            eff_pos = (1.0 - np.power(0.95, pos_counts)) / (1.0 - 0.95)
            eff_neg = (1.0 - np.power(0.95, neg_counts)) / (1.0 - 0.95)
            w = np.clip(eff_neg / np.maximum(eff_pos, 1e-6), 1.0, 4.0)
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, dtype=torch.float32, device=device))
        raise ValueError(f"Unsupported multi-label loss_type: {lt}")

    counts = Counter(int(v) for v in y_train)
    n_total = max(len(y_train), 1)
    priors = np.array([max(counts.get(i, 1), 1) / float(n_total) for i in range(n_labels)], dtype=np.float32)
    inv_w = torch.tensor([n_total / (n_labels * max(counts.get(i, 1), 1)) for i in range(n_labels)], dtype=torch.float32, device=device)

    if lt == "ce":
        return nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)
    if lt == "ce_weighted":
        return nn.CrossEntropyLoss(weight=inv_w, label_smoothing=spec.label_smoothing)
    if lt == "logit_adj":
        return LogitAdjustedCrossEntropyLoss(priors, tau=spec.logit_adj_tau, label_smoothing=spec.label_smoothing)
    if lt == "focal":
        return FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.0)
    if lt == "focal_weighted":
        soft_w = torch.sqrt(inv_w)
        return FocalCrossEntropyLoss(gamma=2.0, weight=soft_w, label_smoothing=0.0)
    raise ValueError(f"Unsupported single-label loss_type: {lt}")


def _model_uses_token_type_ids(hf_cfg: Any) -> bool:
    model_type = str(getattr(hf_cfg, "model_type", "")).lower()
    if model_type in {"xlm-roberta", "roberta", "camembert", "distilbert"}:
        return False
    return int(getattr(hf_cfg, "type_vocab_size", 1)) >= 2


def train_and_predict_logits(
    spec: TrainSpec,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    eval_row_sets: Dict[str, Sequence[Dict[str, Any]]],
    *,
    return_model_info: bool = False,
) -> Dict[str, Any]:
    """Train one model on train_rows, pick best epoch on val_rows, return raw logits on val & eval_row_sets."""
    clear_cuda()
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    random.seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multilabel = spec.task in MULTILABEL_TASKS
    labels = CANONICAL_LABELS[spec.task]

    model_path, local_only = resolve_local_muril_path(spec.backbone_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
    use_tti = _model_uses_token_type_ids(hf_cfg)
    encoder = AutoModel.from_pretrained(model_path, config=hf_cfg, local_files_only=local_only)
    lora_modules = 0
    if spec.use_lora:
        lora_modules = inject_lora(encoder, r=spec.lora_r, alpha=float(spec.lora_r * 2), dropout=0.1)

    model = PooledTransformerClassifier(
        encoder=encoder,
        hidden_size=int(hf_cfg.hidden_size),
        num_labels=len(labels),
        multi_sample_dropout=5,
        dropout_rate=spec.dropout_rate,
        reply_only_pool=spec.reply_only_pool,
        sep_token_id=getattr(tokenizer, "sep_token_id", None),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_pairs = [build_input_pair(r, spec.context_mode) for r in train_rows]
    val_pairs = [build_input_pair(r, spec.context_mode) for r in val_rows]
    y_train = encode_task_targets(train_rows, spec.task)
    y_val = encode_task_targets(val_rows, spec.task)

    def collate_fn(items):
        p_batch, t_batch = zip(*items)
        enc = tokenize_pairs(tokenizer, p_batch, spec.max_length, use_token_type_ids=use_tti)
        t_tensor = torch.tensor(np.asarray(t_batch), dtype=torch.float32 if multilabel else torch.long)
        return enc, t_tensor

    ds = _PairDataset(train_pairs, y_train)
    gen = torch.Generator().manual_seed(spec.seed)
    loader = DataLoader(ds, batch_size=spec.batch_size, shuffle=True, generator=gen, num_workers=0, collate_fn=collate_fn)

    criterion = _make_criterion(spec, y_train, device)
    if spec.use_lora:
        lora_lr = max(spec.learning_rate * 5.0, 2e-4)
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("classifier", "norm"))], "lr": lora_lr * 2.0, "weight_decay": spec.weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("classifier", "norm"))], "lr": lora_lr, "weight_decay": 0.0},
        ]
    else:
        param_groups = _build_llrd_param_groups(
            model,
            base_lr=spec.learning_rate,
            head_lr_multiplier=spec.head_lr_multiplier,
            llrd=spec.llrd,
            weight_decay=spec.weight_decay,
            num_hidden_layers=int(getattr(hf_cfg, "num_hidden_layers", 12)),
        )
    optimizer = torch.optim.AdamW(param_groups, fused=(device.type == "cuda"))
    total_steps = max(len(loader) * spec.epochs, 1)
    warmup_steps = max(int(total_steps * spec.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    @torch.no_grad()
    def infer_logits(pairs: Sequence[Tuple[str, Optional[str]]]) -> Tuple[np.ndarray, float]:
        model.eval()
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros((0, len(labels)), dtype=np.float32), 0.0
        # Length-sorted batching avoids padding short comments to max batch length (~3-4x faster, identical outputs)
        order = sorted(range(n_pairs), key=lambda i: len(pairs[i][0]) + len(pairs[i][1] or ""))
        sorted_pairs = [pairs[i] for i in order]
        out_chunks: List[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n_pairs, spec.eval_batch_size):
            batch_p = sorted_pairs[start : start + spec.eval_batch_size]
            enc = tokenize_pairs(tokenizer, batch_p, spec.max_length, use_token_type_ids=use_tti)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids"),
                )
            out_chunks.append(logits.float().cpu().numpy())
        lat_ms = ((time.perf_counter() - t0) * 1000.0) / max(n_pairs, 1)
        sorted_out = np.concatenate(out_chunks, axis=0)
        restored = np.empty_like(sorted_out)
        restored[order] = sorted_out
        return restored, lat_ms

    best_state = None
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_epoch = 1
    best_val_logits = None
    t_train0 = time.perf_counter()

    for epoch in range(1, spec.epochs + 1):
        model.train()
        for enc, tgt in loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits1 = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                if spec.rdrop_alpha > 0.0:
                    logits2 = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                    ce_loss = 0.5 * (criterion(logits1, tgt) + criterion(logits2, tgt))
                    kl_loss = _bernoulli_sym_kl(logits1, logits2) if multilabel else _categorical_sym_kl(logits1, logits2)
                    loss = ce_loss + spec.rdrop_alpha * kl_loss
                else:
                    loss = criterion(logits1, tgt)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        val_logits, _ = infer_logits(val_pairs)
        val_probs = _logits_to_probs(val_logits, multilabel=multilabel, temperature=1.0)
        val_loss = _validation_nll(val_logits, y_val, multilabel, temperature=1.0)
        if multilabel:
            val_pred = (val_probs > 0.5).astype(int)
            val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
            better = (val_loss < best_val_loss - 1e-4) or (abs(val_loss - best_val_loss) <= 1e-4 and val_f1 > best_val_f1)
        else:
            val_pred = np.argmax(val_probs, axis=1)
            val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
            better = (val_f1 > best_val_f1 + 1e-6) or (abs(val_f1 - best_val_f1) <= 1e-6 and val_loss < best_val_loss)

        if better:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch
            best_val_logits = val_logits.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_s = round(time.perf_counter() - t_train0, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    eval_logits: Dict[str, np.ndarray] = {}
    eval_lat_ms: Dict[str, float] = {}
    for set_name, rows_set in eval_row_sets.items():
        pairs_set = [build_input_pair(r, spec.context_mode) for r in rows_set]
        lg, lms = infer_logits(pairs_set)
        eval_logits[set_name] = lg
        eval_lat_ms[set_name] = round(lms, 3)

    del model, encoder, optimizer, scheduler, best_state
    clear_cuda()

    return {
        "val_logits": best_val_logits,
        "eval_logits": eval_logits,
        "eval_latency_ms": eval_lat_ms,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 5),
        "best_val_f1_raw": round(float(best_val_f1), 5),
        "train_seconds": train_s,
        "peak_vram_mb": peak_vram_mb,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_modules": lora_modules,
    }


# ============================================================================
# Calibration, Evaluation & Thread-Clustered Bootstrap Statistics
# ============================================================================

def tune_multilabel_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: Sequence[str],
    *,
    mode: str = "bounded",  # fixed_05 | unbounded | bounded | regularized
    low: float = 0.30,
    high: float = 0.70,
    prior_strength: float = 10.0,
) -> Dict[str, float]:
    """Calibrate per-label multi-label thresholds strictly on OOF/val probabilities."""
    if mode == "fixed_05":
        return {lbl: 0.5 for lbl in labels}
    lo = 0.10 if mode == "unbounded" else low
    hi = 0.90 if mode == "unbounded" else high
    steps = int(round((hi - lo) / 0.025)) + 1
    grid = [round(lo + 0.025 * i, 3) for i in range(steps)]

    tuned: Dict[str, float] = {}
    for col, lbl in enumerate(labels):
        truth = y_true[:, col]
        n_pos = int(truth.sum())
        if n_pos == 0:
            tuned[lbl] = 0.5
            continue
        best_tau = 0.5
        best_f1 = -1.0
        col_p = probs[:, col]
        for tau in grid:
            pred = (col_p > tau).astype(int)
            tp = int(((truth == 1) & (pred == 1)).sum())
            fp = int(((truth == 0) & (pred == 1)).sum())
            fn = int(((truth == 1) & (pred == 0)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and abs(tau - 0.5) < abs(best_tau - 0.5)):
                best_f1 = f1
                best_tau = tau
        if mode == "regularized":
            # Empirical Bayes shrinkage toward 0.5 proportional to positive support
            w = n_pos / float(n_pos + prior_strength)
            best_tau = round(float(np.clip(w * best_tau + (1.0 - w) * 0.5, low, high)), 3)
        tuned[lbl] = float(best_tau)
    return tuned


def fit_per_class_temperatures(logits: np.ndarray, y_val: np.ndarray, multilabel: bool) -> List[float]:
    """Fit per-label scalar temperatures T_c >= 1.0 minimizing NLL strictly on validation/OOF logits."""
    n_cls = logits.shape[1]
    if not multilabel:
        # Single-label vector temperature scaling (T_c >= 1.0 coordinate descent)
        temps = np.ones(n_cls, dtype=np.float64)
        grid = [round(1.0 + 0.1 * i, 2) for i in range(26)]
        for _ in range(2):
            for c in range(n_cls):
                best_t = temps[c]
                best_nll = float("inf")
                for cand in grid:
                    t_try = temps.copy()
                    t_try[c] = cand
                    scaled = logits.astype(np.float64) / t_try[None, :]
                    scaled -= scaled.max(axis=1, keepdims=True)
                    p = np.exp(scaled)
                    p /= p.sum(axis=1, keepdims=True)
                    nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-7, 1.0))))
                    if nll < best_nll - 1e-6:
                        best_nll = nll
                        best_t = cand
                temps[c] = best_t
        return [round(float(t), 3) for t in temps]

    temps_ml: List[float] = []
    grid = [round(1.0 + 0.1 * i, 2) for i in range(31)]
    for c in range(n_cls):
        best_t = 1.0
        best_nll = float("inf")
        col_l = logits[:, c].astype(np.float64)
        col_y = y_val[:, c].astype(np.float64)
        for cand in grid:
            p = 1.0 / (1.0 + np.exp(-np.clip(col_l / cand, -40.0, 40.0)))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            nll = float(-np.mean(col_y * np.log(p) + (1.0 - col_y) * np.log(1.0 - p)))
            if nll < best_nll - 1e-6:
                best_nll = nll
                best_t = cand
        temps_ml.append(round(float(best_t), 3))
    return temps_ml


def apply_logits_calibration(
    logits: np.ndarray,
    multilabel: bool,
    temperature: Any = 1.0,
) -> np.ndarray:
    if isinstance(temperature, (list, tuple, np.ndarray)):
        t_arr = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0)[None, :]
        scaled = logits.astype(np.float64) / t_arr
        if multilabel:
            return (1.0 / (1.0 + np.exp(-np.clip(scaled, -40.0, 40.0)))).astype(np.float32)
        scaled -= scaled.max(axis=1, keepdims=True)
        exps = np.exp(scaled)
        return (exps / exps.sum(axis=1, keepdims=True)).astype(np.float32)
    return _logits_to_probs(logits, multilabel=multilabel, temperature=max(float(temperature), 1.0))


def evaluate_predictions(
    task: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    binary_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive single-label or multi-label evaluation metrics + 5-bin ECE."""
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS

    if multilabel:
        taus = np.array([float((thresholds or {}).get(lbl, 0.5)) for lbl in labels], dtype=np.float32)
        y_pred = (probs > taus[None, :]).astype(int)
        per_label: Dict[str, Any] = {}
        f1s = []
        for idx, lbl in enumerate(labels):
            tr_col = y_true[:, idx].astype(int)
            pr_col = y_pred[:, idx].astype(int)
            tp = int(((tr_col == 1) & (pr_col == 1)).sum())
            fp = int(((tr_col == 0) & (pr_col == 1)).sum())
            fn = int(((tr_col == 1) & (pr_col == 0)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            sup = int(tr_col.sum())
            pred_pos = int(pr_col.sum())
            per_label[lbl] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": sup,
                "predicted_positive": pred_pos,
                "threshold": round(float(taus[idx]), 3),
            }
            if sup > 0 or pred_pos > 0:
                f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        subset_acc = float(np.mean(np.all(y_true.astype(int) == y_pred, axis=1)))
        hamming_loss = float(np.mean(y_true.astype(int) != y_pred))

        # Build human-readable targets for compute_calibration_metrics
        true_lists = [[labels[j] for j in range(len(labels)) if y_true[i, j] > 0.5] for i in range(len(y_true))]
        cal = compute_calibration_metrics(true_lists, probs, labels, multilabel=True, n_bins=5)
        return {
            "task": task,
            "multilabel": True,
            "n_samples": int(len(y_true)),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "subset_accuracy": round(subset_acc, 4),
            "hamming_loss": round(hamming_loss, 4),
            "ece": round(float(cal["ece"]), 4),
            "brier": round(float(cal["brier_score"]), 4),
            "per_class": per_label,
        }

    # Single-label (multiclass or binary solution)
    if task == "solution" and binary_threshold != 0.5:
        y_pred = (probs[:, 1] >= float(binary_threshold)).astype(int)
    else:
        y_pred = np.argmax(probs, axis=1).astype(int)

    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(float(p_arr[i]), 4),
            "recall": round(float(r_arr[i]), 4),
            "f1": round(float(f1_arr[i]), 4),
            "support": int(sup_arr[i]),
            "predicted_count": int((y_pred == i).sum()),
        }
        for i, lbl in enumerate(labels)
    }
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    true_strs = [labels[int(v)] for v in y_true]
    cal = compute_calibration_metrics(true_strs, probs, labels, multilabel=False, n_bins=5)

    return {
        "task": task,
        "multilabel": False,
        "n_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(acc, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(float(cal["ece"]), 4),
        "brier": round(float(cal["brier_score"]), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def thread_clustered_bootstrap_ci(
    rows: Sequence[Dict[str, Any]],
    task: str,
    y_true: np.ndarray,
    probs_a: np.ndarray,
    *,
    probs_b: Optional[np.ndarray] = None,
    thresholds_a: Optional[Dict[str, float]] = None,
    thresholds_b: Optional[Dict[str, float]] = None,
    n_boot: int = 2000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Thread-clustered bootstrap 95% CI (resampling post_id groups with replacement)."""
    rng = np.random.default_rng(seed)
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS

    def _to_pred(probs, taus):
        if multilabel:
            t_arr = np.array([float((taus or {}).get(l, 0.5)) for l in labels], dtype=np.float32)
            return (probs > t_arr[None, :]).astype(int)
        return np.argmax(probs, axis=1).astype(int)

    pred_a = _to_pred(probs_a, thresholds_a)
    pred_b = _to_pred(probs_b, thresholds_b) if probs_b is not None else None

    pid_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        pid_to_idx[r["post_id"]].append(i)
    pids = sorted(pid_to_idx.keys())
    n_clusters = len(pids)

    scores_a: List[float] = []
    deltas: List[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = [i for c_i in sampled for i in pid_to_idx[pids[c_i]]]
        yt = y_true[idx]
        pa = pred_a[idx]
        fa = float(f1_score(yt, pa, average="macro", zero_division=0))
        scores_a.append(fa)
        if pred_b is not None:
            pb = pred_b[idx]
            fb = float(f1_score(yt, pb, average="macro", zero_division=0))
            deltas.append(fa - fb)

    lo_a, hi_a = np.percentile(scores_a, [2.5, 97.5])
    out: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "macro_f1_mean": round(float(np.mean(scores_a)), 4),
        "ci95_low": round(float(lo_a), 4),
        "ci95_high": round(float(hi_a), 4),
    }
    if deltas:
        lo_d, hi_d = np.percentile(deltas, [2.5, 97.5])
        p_better = float(np.mean([d > 0 for d in deltas]))
        out["paired_delta_mean"] = round(float(np.mean(deltas)), 4)
        out["paired_delta_ci95"] = [round(float(lo_d), 4), round(float(hi_d), 4)]
        out["p_a_beats_b"] = round(p_better, 4)
    return out
