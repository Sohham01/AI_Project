"""V6.6 Shared Transformer Architecture, Universal WWM Masking & Nested OOF Stacking.

Implements three targeted upgrades over V6.1 without changing frozen splits or
production artifacts:
  1. `resolve_local_backbone_path`: resolves both WordPiece (`MuRIL`) and
     SentencePiece (`XLM-RoBERTa`) HuggingFace snapshots & local DAPT checkpoints
     strictly offline from `$HF_HOME` / `$RCI_MODEL_CACHE_DIR`.
  2. `mask_tokens_wwm_universal`: atomic Whole-Word Masking for BOTH WordPiece
     (`##` continuation tokens) and SentencePiece (`\\u2581` word boundaries).
  3. `GatedReplyContextClassifier`: full cross-attention across `[CLS] reply [SEP] parent [SEP]`
     with reply-masked `[Mean + Max]` pooling and a zero-initialized bounded
     tanh contrastive gate (`g = 0.30 * tanh(alpha)`) preventing parent context bleed
     while letting gradient flow when parent disambiguation helps (`counterargument`, `solution`).
  4. `fit_predict_nested_oof_stacker`: strictly outer-fold-isolated simplex
     ensembling (`w >= 0, sum(w) == 1`) with bounded OOF thresholds `[0.30, 0.70]`.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from torch import nn

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    MULTILABEL_TASKS,
    evaluate_predictions,
    tune_multilabel_thresholds,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_HF_CACHE = (
    Path(os.environ.get("RCI_MODEL_CACHE_DIR", str(ROOT / ".cache" / "models")))
    / "hf_home"
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("vocab.txt", "tokenizer.json", "sentencepiece.bpe.model", "tokenizer_config.json")


def _has_model_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    has_weights = any((path / wf).is_file() for wf in WEIGHT_FILES)
    has_tok = any((path / tf).is_file() for tf in TOKENIZER_FILES)
    return has_weights and has_tok


def resolve_local_backbone_path(model_id: str) -> Tuple[str, bool]:
    """Resolve an explicit local checkpoint directory or cached HF snapshot offline."""
    explicit = Path(model_id)
    if _has_model_files(explicit):
        return str(explicit), True

    cache_root = Path(os.environ.get("RCI_MODEL_CACHE_DIR", str(ROOT / ".cache" / "models")))
    for sub in ("v6_6_experiments", "v5_experiments"):
        cand = cache_root / sub / model_id
        if _has_model_files(cand):
            return str(cand), True

    slug = "models--" + model_id.replace("/", "--")
    candidate_roots: List[Path] = []
    if os.environ.get("HF_HOME"):
        candidate_roots.append(Path(os.environ["HF_HOME"]) / "hub" / slug)
    candidate_roots.append(DEFAULT_LOCAL_HF_CACHE / "hub" / slug)
    candidate_roots.append(Path.home() / ".cache" / "huggingface" / "hub" / slug)

    for repo_dir in candidate_roots:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snap in sorted(snapshots_dir.iterdir(), reverse=True):
            if _has_model_files(snap):
                return str(snap), True

    return model_id, False


def _is_punct_token(tok: str) -> bool:
    stripped = tok.lstrip("\u2581").lstrip("#")
    return bool(stripped) and all(not ch.isalnum() for ch in stripped)


def group_subword_positions(
    tokens: Sequence[Optional[str]],
    ids_row: Sequence[int],
    attn_row: Sequence[int],
    special_ids: set[int],
    *,
    is_sentencepiece: bool,
) -> Tuple[List[List[int]], int]:
    """Group WordPiece ('##') or SentencePiece ('\\u2581') subwords into atomic whole-word spans."""
    word_groups: List[List[int]] = []
    valid_tokens = 0
    for pos, (tid, att, tok) in enumerate(zip(ids_row, attn_row, tokens)):
        if att == 0 or tid in special_ids or tok is None or tok == "":
            continue
        valid_tokens += 1
        if is_sentencepiece:
            starts_new = tok.startswith("\u2581") or _is_punct_token(tok)
            prev_is_punct = bool(word_groups) and _is_punct_token(tokens[word_groups[-1][-1]] or "")
            if not starts_new and word_groups and not prev_is_punct:
                word_groups[-1].append(pos)
            else:
                word_groups.append([pos])
        else:
            if tok.startswith("##") and word_groups:
                word_groups[-1].append(pos)
            else:
                word_groups.append([pos])
    return word_groups, valid_tokens


def _tokenizer_is_sentencepiece(tokenizer: Any, sample_tokens: Sequence[Optional[str]]) -> bool:
    cls_name = tokenizer.__class__.__name__.lower()
    if any(k in cls_name for k in ("roberta", "camembert", "xlm", "albert", "deberta", "spiece")):
        return True
    if any(isinstance(t, str) and t.startswith("\u2581") for t in sample_tokens):
        return True
    return False


def mask_tokens_wwm_universal(
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    tokenizer: Any,
    mlm_probability: float,
    rng: random.Random,
    generator: "torch.Generator",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Atomic Whole-Word Masking (80/10/10) supporting both WordPiece and SentencePiece."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for mask_tokens_wwm_universal")

    special_ids: set[int] = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_ids.add(int(val))

    bsz, seq_len = input_ids.shape
    masked_indices = torch.zeros((bsz, seq_len), dtype=torch.bool)

    for b in range(bsz):
        ids_row = input_ids[b].tolist()
        attn_row = attention_mask[b].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids_row)
        is_sp = _tokenizer_is_sentencepiece(tokenizer, toks)
        word_groups, valid_tokens = group_subword_positions(
            toks, ids_row, attn_row, special_ids, is_sentencepiece=is_sp
        )
        if not word_groups or valid_tokens == 0:
            continue

        target_masked = max(1, int(round(valid_tokens * mlm_probability)))
        rng.shuffle(word_groups)
        covered = 0
        for grp in word_groups:
            if covered >= target_masked:
                break
            if covered > 0 and covered + len(grp) > target_masked + 2:
                continue
            for pos in grp:
                masked_indices[b, pos] = True
            covered += len(grp)

    labels = input_ids.clone()
    labels[~masked_indices] = -100

    vocab_n = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    replace_rand = torch.rand(labels.shape, generator=generator)
    masked_input = input_ids.clone()
    masked_input[(replace_rand < 0.8) & masked_indices] = tokenizer.mask_token_id
    indices_random = (replace_rand >= 0.8) & (replace_rand < 0.9) & masked_indices
    if indices_random.any():
        rand_words = torch.randint(max(vocab_n, 10), labels.shape, dtype=torch.long, generator=generator)
        masked_input[indices_random] = rand_words[indices_random]
    return masked_input, labels


if HAS_TORCH:

    class GatedReplyContextClassifier(nn.Module):
        """Transformer sequence classifier with optional reply-masked contrastive context gating.

        When `pooling_mode == "gated_reply"` and a sequence pair `<reply> [SEP] <context>` is
        encoded, full Transformer self-attention cross-attends across reply and parent context,
        while `[Mean, Max]` pooling is isolated to Segment 0 (reply) tokens with a bounded
        zero-initialized gate (`0.30 * tanh(alpha)`) controlling contrastive parent fusion.
        """

        def __init__(
            self,
            encoder: nn.Module,
            hidden_size: int,
            num_labels: int,
            *,
            pooling_mode: str = "standard",  # standard | gated_reply
            sep_token_id: Optional[int] = None,
            multi_sample_dropout: int = 5,
            dropout_rate: float = 0.20,
        ) -> None:
            super().__init__()
            if pooling_mode not in {"standard", "gated_reply"}:
                raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}")
            self.encoder = encoder
            self.hidden_size = int(hidden_size)
            self.pooling_mode = pooling_mode
            self.sep_token_id = sep_token_id
            self.pooled_dim = self.hidden_size * 3
            self.norm = nn.LayerNorm(self.pooled_dim)
            self.dropouts = nn.ModuleList(
                [nn.Dropout(dropout_rate) for _ in range(max(1, multi_sample_dropout))]
            )
            self.classifier = nn.Linear(self.pooled_dim, num_labels)
            # Zero-initialized gate starts identical to pure reply-only pooling (zero bleed).
            self.gate_alpha = nn.Parameter(torch.zeros(self.hidden_size))

        def extract_span_masks(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute `(reply_mask, ctx_mask)` cleanly for `[CLS] A [SEP] B [SEP]` and `<s> A </s> </s> B </s>`."""
            if self.sep_token_id is None:
                return attention_mask.to(torch.float32), torch.zeros_like(attention_mask, dtype=torch.float32)

            is_sep = (input_ids == int(self.sep_token_id)).long()
            seg_idx = torch.cumsum(is_sep, dim=1) - is_sep
            seq_len = input_ids.shape[1]
            pos_gt0 = (torch.arange(seq_len, device=input_ids.device).unsqueeze(0) > 0).to(attention_mask.dtype)
            non_sep = (1 - is_sep).to(attention_mask.dtype)

            reply_mask = attention_mask * (seg_idx == 0).to(attention_mask.dtype) * non_sep * pos_gt0
            has_reply = reply_mask.sum(dim=1, keepdim=True) > 0
            reply_mask = torch.where(has_reply, reply_mask, attention_mask.to(reply_mask.dtype))
            ctx_mask = attention_mask * (seg_idx >= 1).to(attention_mask.dtype) * non_sep
            return reply_mask, ctx_mask

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            enc_kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                enc_kwargs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**enc_kwargs).last_hidden_state
            cls_vec = hidden[:, 0, :]

            if self.pooling_mode == "standard":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                mean_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                max_vec = hidden.masked_fill(mask == 0, -1e4).max(dim=1).values
                pooled = torch.cat([cls_vec, mean_vec, max_vec], dim=-1)
            else:
                reply_mask, ctx_mask = self.extract_span_masks(input_ids, attention_mask)
                r_mask = reply_mask.unsqueeze(-1).to(hidden.dtype)
                r_mean = (hidden * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1e-6)
                r_max = hidden.masked_fill(r_mask == 0, -1e4).max(dim=1).values

                c_sum = ctx_mask.sum(dim=1, keepdim=True).to(hidden.dtype)
                has_ctx = (c_sum > 0).to(hidden.dtype)
                c_mask = ctx_mask.unsqueeze(-1).to(hidden.dtype)
                c_mean = (hidden * c_mask).sum(dim=1) / c_sum.clamp(min=1e-6)

                g = (0.30 * torch.tanh(self.gate_alpha)).to(hidden.dtype).unsqueeze(0)
                cls_fused = cls_vec + has_ctx * (g * c_mean)
                r_fused = r_mean + has_ctx * (g * (r_mean - c_mean))
                pooled = torch.cat([cls_fused, r_fused, r_max], dim=-1)

            normed = self.norm(pooled)
            return torch.stack([self.classifier(dp(normed)) for dp in self.dropouts], dim=0).mean(dim=0)


def _generate_simplex_weights(n_models: int, units: int = 10) -> np.ndarray:
    """Enumerate all probability simplex weight vectors with step `1/units`."""
    combos: List[List[int]] = []
    # Compositions of `units` into `n_models` non-negative integers via bars-and-stars
    for dividers in itertools.combinations(range(units + n_models - 1), n_models - 1):
        prev = -1
        parts: List[int] = []
        for d in dividers:
            parts.append(d - prev - 1)
            prev = d
        parts.append((units + n_models - 1) - prev - 1)
        combos.append(parts)
    arr = np.asarray(combos, dtype=np.float64) / float(units)
    return arr


def _macro_f1_quick(y_true: np.ndarray, probs: np.ndarray, multilabel: bool, thresh: Optional[np.ndarray] = None) -> float:
    if multilabel:
        t = 0.5 if thresh is None else thresh[None, :]
        pred = (probs > t).astype(int)
        f1s: List[float] = []
        for c in range(y_true.shape[1]):
            tr = y_true[:, c]
            pr = pred[:, c]
            sup = int(tr.sum())
            ppos = int(pr.sum())
            if sup == 0 and ppos == 0:
                continue
            tp = int(((tr == 1) & (pr == 1)).sum())
            fp = int(((tr == 0) & (pr == 1)).sum())
            fn = int(((tr == 1) & (pr == 0)).sum())
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0

    pred = np.argmax(probs, axis=1)
    n_cls = probs.shape[1]
    f1s = []
    for c in range(n_cls):
        tr = (y_true == c)
        pr = (pred == c)
        tp = int((tr & pr).sum())
        fp = int((~tr & pr).sum())
        fn = int((tr & ~pr).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _quick_nll(y_true: np.ndarray, probs: np.ndarray, multilabel: bool) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    if multilabel:
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


def fit_predict_nested_oof_stacker(
    *,
    task: str,
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    y_true: np.ndarray,
    oof_prob_candidates: Sequence[np.ndarray],
    test_prob_candidates: Sequence[np.ndarray],
    labels: Sequence[str],
    candidate_names: Sequence[str],
    units: int = 10,
) -> Dict[str, Any]:
    """Strictly outer-fold-isolated simplex stacking over calibrated probability matrices.

    For each outer fold `(tr_idx, va_idx)`:
      - Searches simplex weights `w` (`w >= 0, sum(w) == 1`) + bounded thresholds
        exclusively on `tr_idx`.
      - Writes blended predictions strictly into `nested_oof_probs[va_idx]`.
      - Averages outer fold weights & thresholds before scoring `stacked_test_probs`.
    """
    n_models = len(oof_prob_candidates)
    assert n_models == len(test_prob_candidates) == len(candidate_names) >= 2
    n_samples, n_classes = oof_prob_candidates[0].shape
    multilabel = task in MULTILABEL_TASKS

    oof_stack = np.stack([np.asarray(p, dtype=np.float64) for p in oof_prob_candidates], axis=0)  # (M, N, C)
    test_stack = np.stack([np.asarray(p, dtype=np.float64) for p in test_prob_candidates], axis=0)  # (M, N_te, C)
    weight_grid = _generate_simplex_weights(n_models, units=units)

    nested_oof_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    fold_weights: List[np.ndarray] = []
    fold_taus: List[Dict[str, float]] = []

    for tr_idx, va_idx in folds:
        tr_idx_arr = np.asarray(tr_idx, dtype=int)
        va_idx_arr = np.asarray(va_idx, dtype=int)
        assert not (set(tr_idx_arr.tolist()) & set(va_idx_arr.tolist())), "Outer fold leakage!"

        y_tr = y_true[tr_idx_arr]
        oof_tr = oof_stack[:, tr_idx_arr, :]  # (M, N_tr, C)
        oof_va = oof_stack[:, va_idx_arr, :]  # (M, N_va, C)

        # Rank all simplex weights on inner training rows (with tiny NLL & diversity tiebreak)
        # Rank all simplex weights on inner training rows (with tiny NLL & diversity tiebreak)
        scored: List[Tuple[float, np.ndarray]] = []
        for w in weight_grid:
            p_tr = np.tensordot(w, oof_tr, axes=(0, 0))
            f1 = _macro_f1_quick(y_tr, p_tr, multilabel=multilabel)
            nll = _quick_nll(y_tr, p_tr, multilabel=multilabel)
            diversity_bonus = -1.5e-3 * float(np.sum(w ** 2))
            scored.append((f1 - 2.0e-3 * nll + diversity_bonus, w))

        scored.sort(key=lambda x: -x[0])

        if not multilabel:
            best_w = scored[0][1]
            fold_weights.append(best_w)
            nested_oof_probs[va_idx_arr] = np.tensordot(best_w, oof_va, axes=(0, 0))
        else:
            # Fine-tune bounded [0.12, 0.65] thresholds on calibrated posteriors strictly on tr_idx
            top_candidates: List[np.ndarray] = [s[1] for s in scored[:12]]
            for m in range(n_models):
                corner = np.zeros(n_models, dtype=np.float64)
                corner[m] = 1.0
                top_candidates.append(corner)

            best_score = -1e9
            best_w = top_candidates[0]
            best_taus: Dict[str, float] = {lbl: 0.5 for lbl in labels}
            for w in top_candidates:
                p_tr = np.tensordot(w, oof_tr, axes=(0, 0))
                taus = tune_multilabel_thresholds(y_tr, p_tr, labels, mode="bounded", low=0.12, high=0.65)
                tau_arr = np.asarray([taus[l] for l in labels], dtype=np.float64)
                f1_t = _macro_f1_quick(y_tr, p_tr, multilabel=True, thresh=tau_arr)
                nll = _quick_nll(y_tr, p_tr, multilabel=True)
                score = f1_t - 2.0e-3 * nll - 1.5e-3 * float(np.sum(w ** 2))
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_taus = taus
            fold_weights.append(best_w)
            fold_taus.append(best_taus)
            nested_oof_probs[va_idx_arr] = np.tensordot(best_w, oof_va, axes=(0, 0))

    mean_w = np.mean(np.stack(fold_weights, axis=0), axis=0)
    mean_w = np.maximum(mean_w, 0.0)
    mean_w /= max(float(mean_w.sum()), 1e-12)

    mean_taus: Optional[Dict[str, float]] = None
    if multilabel and fold_taus:
        mean_taus = {
            lbl: round(float(np.mean([ft[lbl] for ft in fold_taus])), 3)
            for lbl in labels
        }

    stacked_test_probs = np.tensordot(mean_w, test_stack, axes=(0, 0))
    oof_eval = evaluate_predictions(task, y_true, nested_oof_probs, thresholds=mean_taus)

    return {
        "task": task,
        "candidate_names": list(candidate_names),
        "mean_weights": {name: round(float(mean_w[i]), 4) for i, name in enumerate(candidate_names)},
        "per_fold_weights": [
            {name: round(float(fw[i]), 3) for i, name in enumerate(candidate_names)}
            for fw in fold_weights
        ],
        "thresholds": mean_taus,
        "nested_oof_probs": nested_oof_probs.astype(np.float32),
        "stacked_test_probs": stacked_test_probs.astype(np.float32),
        "oof_eval": oof_eval,
    }
