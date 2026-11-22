"""Stage 3 — Genuine, Strictly Inductive Model Tuning (Zero Leakage, OOF-Driven).

Evaluates targeted hypotheses motivated by V5/V6 ablations & Task 6 empirical findings:
1. Response (hardest task, 53.52% V6):
   - xlm-roberta-base (comment_only, raw & emoji_affect_slang)
   - Span-Masked Reply-Only Pooling (parent_pair + reply_only_pool=True) on MuRIL-WWM & XLM-R
2. Sentiment & Emotion (Task 6 discovery: emoji_affect_slang lifted MuRIL-WWM +2.27/+1.23 pp):
   - MuRIL-DAPT-WWM + emoji_affect_slang (Sentiment CE-W; Emotion ASL + OOF monotonic Platt)
   - xlm-roberta-base + emoji_affect_slang (Sentiment CE-W)
3. 2-Encoder Shared 3-Task MTL (Sentiment + Response + Solution + R-Drop alpha=0.5):
   - Shared MuRIL-DAPT-WWM vs. Shared xlm-roberta-base

STRICT SCIENTIFIC GUARDRAILS:
- All model selection uses ONLY 5-fold post_id-grouped OOF CV on untouched V1 trainval (367).
- Frozen held-out test set (92 comments, 37 threads) is NEVER used for selection or tuning.
- OOF per-label monotonic Platt calibration is fit strictly inside OOF for multi-label ASL heads.
- Paired thread-clustered bootstrap (n=2,000) compares ONLY the OOF-selected champion against V6.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.nlp.preprocessing.emoji_normalizer import normalize_emojis_semantic_affect
from app.services.nlp.preprocessing.slang_normalizer import normalize_slang
from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    CKPT_ROOT,
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TrainSpec,
    _categorical_sym_kl,
    _make_criterion,
    _model_uses_token_type_ids,
    apply_logits_calibration,
    build_input_pair,
    clear_cuda,
    encode_task_targets,
    ensure_dir,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    thread_clustered_bootstrap_ci,
    tokenize_pairs,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.models.transformer.muril_classifier import (
    _build_llrd_param_groups,
    resolve_local_muril_path,
)

DAPT_STRICT_WWM_DIR = str(CKPT_ROOT / "muril_dapt_strict_inductive_wwm_8ep")
XLMR_ID = "xlm-roberta-base"
OUT_DIR = ensure_dir(EXP_ROOT / "stage3_tuning")
OUT_JSON = EXP_ROOT / "stage3_tuning_results.json"

_SPACE_RE = re.compile(r"\s+")
MTL3_TASKS = ("sentiment", "response", "solution")


def _tf_raw(text: str) -> str:
    return str(text or "")


def _tf_affect_slang(text: str) -> str:
    return _SPACE_RE.sub(" ", normalize_slang(normalize_emojis_semantic_affect(str(text or "")))).strip()


def _apply_text_transform(rows: Sequence[Dict[str, Any]], fn: Callable[[str], str]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        nr = dict(r)
        nr["text"] = fn(r["text"])
        if r.get("parent_comment_text"):
            nr["parent_comment_text"] = fn(r["parent_comment_text"])
        out.append(nr)
    return out


def _fit_oof_platt(oof_logits: np.ndarray, y_true: np.ndarray) -> List[Tuple[float, float]]:
    """Per-label monotonic Platt calibration (a > 0) strictly on OOF logits."""
    params: List[Tuple[float, float]] = []
    for c in range(oof_logits.shape[1]):
        col_x = oof_logits[:, c].reshape(-1, 1).astype(np.float64)
        col_y = y_true[:, c].astype(int)
        if len(np.unique(col_y)) < 2:
            params.append((1.0, 0.0))
            continue
        lr = LogisticRegression(C=10.0, max_iter=500)
        lr.fit(col_x, col_y)
        a = max(float(lr.coef_[0, 0]), 0.05)
        b = float(lr.intercept_[0])
        params.append((a, b))
    return params


def _apply_platt(logits: np.ndarray, params: List[Tuple[float, float]]) -> np.ndarray:
    probs = np.zeros_like(logits, dtype=np.float32)
    for c, (a, b) in enumerate(params):
        z = np.clip(a * logits[:, c].astype(np.float64) + b, -40.0, 40.0)
        probs[:, c] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return probs


def run_single_task_5fold(
    exp_id: str,
    *,
    task: str,
    backbone_id: str,
    context_mode: str,
    loss_type: str,
    text_mode: str = "raw",
    reply_only_pool: bool = False,
    epochs: int = 6,
    lr: float = 4e-5,
    max_length: int = 160,
    trainval: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    cache_file = OUT_DIR / f"{exp_id}.json"
    npz_file = OUT_DIR / f"{exp_id}.npz"
    if cache_file.is_file() and npz_file.is_file():
        print(f"[CACHE HIT] {exp_id}", flush=True)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    fn = _tf_affect_slang if text_mode == "emoji_affect_slang" else _tf_raw
    tv_adj = _apply_text_transform(trainval, fn)
    te_adj = _apply_text_transform(test_rows, fn)

    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    folds = make_group_cv_folds(tv_adj, task=task, n_splits=5, seed=SEED)
    y_tv = encode_task_targets(tv_adj, task)
    y_te = encode_task_targets(te_adj, task)

    oof_logits = np.zeros((len(tv_adj), len(labels)), dtype=np.float32)
    test_logits_folds: List[np.ndarray] = []
    t0 = time.time()
    total_lat_ms = 0.0

    for fi, (tr_idx, va_idx) in enumerate(folds):
        spec = TrainSpec(
            task=task,
            backbone_id=backbone_id,
            context_mode=context_mode,
            loss_type=loss_type,
            max_length=max_length,
            epochs=epochs,
            batch_size=8,
            eval_batch_size=32,
            learning_rate=lr,
            reply_only_pool=reply_only_pool,
            seed=SEED + fi,
        )
        res = train_and_predict_logits(
            spec,
            [tv_adj[i] for i in tr_idx],
            [tv_adj[i] for i in va_idx],
            {"test": te_adj},
        )
        oof_logits[va_idx] = res["val_logits"]
        test_logits_folds.append(res["eval_logits"]["test"])
        total_lat_ms += float(res["eval_latency_ms"]["test"])

    test_logits_mean = np.mean(np.stack(test_logits_folds, axis=0), axis=0).astype(np.float32)
    elapsed = round(time.time() - t0, 1)

    if multilabel and loss_type == "asl":
        platt_params = _fit_oof_platt(oof_logits, y_tv)
        oof_probs = _apply_platt(oof_logits, platt_params)
        test_probs = _apply_platt(test_logits_mean, platt_params)
        thresholds = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="regularized", low=0.25, high=0.70, prior_strength=10.0)
        cal_mode = "oof_monotonic_platt"
    else:
        oof_probs = apply_logits_calibration(oof_logits, multilabel=multilabel, temperature=1.0)
        test_probs = apply_logits_calibration(test_logits_mean, multilabel=multilabel, temperature=1.0)
        thresholds = None
        cal_mode = "5fold_native_T1.0"

    oof_ev = evaluate_predictions(task, y_tv, oof_probs, thresholds=thresholds)
    test_ev = evaluate_predictions(task, y_te, test_probs, thresholds=thresholds)

    np.savez_compressed(npz_file, oof_probs=oof_probs, test_probs=test_probs, oof_logits=oof_logits, test_logits=test_logits_mean)
    summary = {
        "exp_id": exp_id,
        "task": task,
        "backbone": Path(backbone_id).name if ("\\" in backbone_id or "/" in backbone_id) else backbone_id,
        "context_mode": context_mode,
        "loss_type": loss_type,
        "text_mode": text_mode,
        "reply_only_pool": reply_only_pool,
        "calibration": cal_mode,
        "thresholds": thresholds,
        "train_seconds": elapsed,
        "mean_inference_ms": round(total_lat_ms / 5.0, 3),
        "oof_macro_f1": oof_ev["macro_f1"],
        "oof_ece": oof_ev["ece"],
        "oof_per_class": oof_ev["per_class"],
        "test_macro_f1": test_ev["macro_f1"],
        "test_ece": test_ev["ece"],
        "test_per_class": test_ev["per_class"],
    }
    save_json(cache_file, summary)
    print(
        f"[{exp_id:24s}] OOF F1={oof_ev['macro_f1']:.4f} (ECE={oof_ev['ece']:.4f}) | "
        f"elapsed={elapsed}s",
        flush=True,
    )
    return summary


class Shared3TaskModel(nn.Module):
    """One shared encoder -> 3 single-label heads (Sentiment, Response, Solution)."""

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout_rate: float = 0.20, msd: int = 5) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooled_dim = hidden_size * 3
        self.norms = nn.ModuleDict({t: nn.LayerNorm(self.pooled_dim) for t in MTL3_TASKS})
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(msd)])
        self.heads = nn.ModuleDict({t: nn.Linear(self.pooled_dim, len(CANONICAL_LABELS[t])) for t in MTL3_TASKS})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        kw: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kw).last_hidden_state
        cls_vec = hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        max_vec = hidden.masked_fill(mask == 0, -1e4).max(dim=1).values
        base = torch.cat([cls_vec, mean_vec, max_vec], dim=-1)
        out: Dict[str, torch.Tensor] = {}
        for t in MTL3_TASKS:
            p = self.norms[t](base)
            out[t] = torch.stack([self.heads[t](dp(p)) for dp in self.dropouts], dim=0).mean(dim=0)
        return out


class _MTL3Dataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, Optional[str]]], targets: Dict[str, np.ndarray]) -> None:
        self.pairs = list(pairs)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx], {t: int(self.targets[t][idx]) for t in MTL3_TASKS}


def run_mtl3_5fold(
    exp_id: str,
    *,
    backbone_id: str,
    text_mode: str = "emoji_affect_slang",
    rdrop_alpha: float = 0.5,
    epochs: int = 6,
    lr: float = 3.5e-5,
    trainval: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    cache_file = OUT_DIR / f"{exp_id}.json"
    npz_file = OUT_DIR / f"{exp_id}.npz"
    if cache_file.is_file() and npz_file.is_file():
        print(f"[CACHE HIT] {exp_id}", flush=True)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    fn = _tf_affect_slang if text_mode == "emoji_affect_slang" else _tf_raw
    tv_adj = _apply_text_transform(trainval, fn)
    te_adj = _apply_text_transform(test_rows, fn)

    folds = make_group_cv_folds(tv_adj, task="response", n_splits=5, seed=SEED)
    y_tv = {t: encode_task_targets(tv_adj, t) for t in MTL3_TASKS}
    y_te = {t: encode_task_targets(te_adj, t) for t in MTL3_TASKS}
    oof_logits = {t: np.zeros((len(tv_adj), len(CANONICAL_LABELS[t])), dtype=np.float32) for t in MTL3_TASKS}
    test_logits_folds: Dict[str, List[np.ndarray]] = {t: [] for t in MTL3_TASKS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, local_only = resolve_local_muril_path(backbone_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
    use_tti = _model_uses_token_type_ids(hf_cfg)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    t0 = time.time()
    for fi, (tr_idx, va_idx) in enumerate(folds):
        clear_cuda()
        seed_f = SEED + fi
        torch.manual_seed(seed_f)
        np.random.seed(seed_f)
        random.seed(seed_f)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_f)

        encoder = AutoModel.from_pretrained(model_path, config=hf_cfg, local_files_only=local_only)
        model = Shared3TaskModel(encoder, int(hf_cfg.hidden_size), dropout_rate=0.20, msd=5).to(device)

        tr_rows = [tv_adj[i] for i in tr_idx]
        va_rows = [tv_adj[i] for i in va_idx]
        tr_pairs = [build_input_pair(r, "comment_only") for r in tr_rows]
        va_pairs = [build_input_pair(r, "comment_only") for r in va_rows]
        te_pairs = [build_input_pair(r, "comment_only") for r in te_adj]

        tr_targets = {t: y_tv[t][tr_idx] for t in MTL3_TASKS}
        va_targets = {t: y_tv[t][va_idx] for t in MTL3_TASKS}
        criteria = {
            t: _make_criterion(TrainSpec(task=t, backbone_id=backbone_id, loss_type="ce_weighted"), tr_targets[t], device)
            for t in MTL3_TASKS
        }

        def collate(items):
            p_b, t_b = zip(*items)
            enc = tokenize_pairs(tokenizer, p_b, 256, use_token_type_ids=use_tti)
            tgt = {t: torch.tensor([x[t] for x in t_b], dtype=torch.long) for t in MTL3_TASKS}
            return enc, tgt

        gen = torch.Generator().manual_seed(seed_f)
        loader = DataLoader(_MTL3Dataset(tr_pairs, tr_targets), batch_size=8, shuffle=True, generator=gen, num_workers=0, collate_fn=collate)
        param_groups = _build_llrd_param_groups(
            model, base_lr=lr, head_lr_multiplier=4.0, llrd=0.92, weight_decay=0.01,
            num_hidden_layers=int(getattr(hf_cfg, "num_hidden_layers", 12)),
        )
        optimizer = torch.optim.AdamW(param_groups, fused=(device.type == "cuda"))
        total_steps = max(len(loader) * epochs, 1)
        warmup = max(int(total_steps * 0.10), 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s + 1) / warmup if s < warmup else max(0.05, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(total_steps - warmup, 1)))),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

        @torch.no_grad()
        def infer(pairs_list):
            model.eval()
            out_map: Dict[str, List[np.ndarray]] = {t: [] for t in MTL3_TASKS}
            for start in range(0, len(pairs_list), 32):
                enc = tokenize_pairs(tokenizer, pairs_list[start : start + 32], 256, use_token_type_ids=use_tti)
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    lg = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                for t in MTL3_TASKS:
                    out_map[t].append(lg[t].float().cpu().numpy())
            return {t: np.concatenate(out_map[t], axis=0) for t in MTL3_TASKS}

        best_val_f1 = -1.0
        best_state = None
        best_val_lg = None

        for ep in range(1, epochs + 1):
            model.train()
            for enc, tgt in loader:
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                tgt = {t: v.to(device, non_blocking=True) for t, v in tgt.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    lg1 = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                    loss1 = sum(criteria[t](lg1[t], tgt[t]) for t in MTL3_TASKS) / 3.0
                    if rdrop_alpha > 0.0:
                        lg2 = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                        loss2 = sum(criteria[t](lg2[t], tgt[t]) for t in MTL3_TASKS) / 3.0
                        kl = sum(_categorical_sym_kl(lg1[t], lg2[t]) for t in MTL3_TASKS) / 3.0
                        loss = 0.5 * (loss1 + loss2) + rdrop_alpha * kl
                    else:
                        loss = loss1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            v_lg = infer(va_pairs)
            f1s = []
            for t in MTL3_TASKS:
                vp = apply_logits_calibration(v_lg[t], multilabel=False, temperature=1.0)
                f1s.append(evaluate_predictions(t, va_targets[t], vp)["macro_f1"])
            mean_v = float(np.mean(f1s))
            if mean_v > best_val_f1:
                best_val_f1 = mean_v
                best_val_lg = v_lg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        te_lg = infer(te_pairs)
        for t in MTL3_TASKS:
            oof_logits[t][va_idx] = best_val_lg[t]
            test_logits_folds[t].append(te_lg[t])

        del model, encoder, optimizer, scheduler, best_state
        clear_cuda()

    elapsed = round(time.time() - t0, 1)
    per_task_out: Dict[str, Any] = {}
    npz_payload: Dict[str, np.ndarray] = {}
    for t in MTL3_TASKS:
        te_mean = np.mean(np.stack(test_logits_folds[t], axis=0), axis=0).astype(np.float32)
        op = apply_logits_calibration(oof_logits[t], multilabel=False, temperature=1.0)
        tp = apply_logits_calibration(te_mean, multilabel=False, temperature=1.0)
        oev = evaluate_predictions(t, y_tv[t], op)
        tev = evaluate_predictions(t, y_te[t], tp)
        npz_payload[f"{t}_oof_probs"] = op
        npz_payload[f"{t}_test_probs"] = tp
        per_task_out[t] = {
            "oof_macro_f1": oev["macro_f1"],
            "oof_ece": oev["ece"],
            "test_macro_f1": tev["macro_f1"],
            "test_ece": tev["ece"],
        }

    np.savez_compressed(npz_file, **npz_payload)
    summary = {
        "exp_id": exp_id,
        "backbone": Path(backbone_id).name if ("\\" in backbone_id or "/" in backbone_id) else backbone_id,
        "text_mode": text_mode,
        "rdrop_alpha": rdrop_alpha,
        "train_seconds": elapsed,
        "mean_3task_oof_f1": round(float(np.mean([per_task_out[t]["oof_macro_f1"] for t in MTL3_TASKS])), 4),
        "mean_3task_test_f1": round(float(np.mean([per_task_out[t]["test_macro_f1"] for t in MTL3_TASKS])), 4),
        "tasks": per_task_out,
    }
    save_json(cache_file, summary)
    print(
        f"[{exp_id:24s}] 3-Task OOF Mean={summary['mean_3task_oof_f1']:.4f} "
        f"(Sent={per_task_out['sentiment']['oof_macro_f1']:.4f}, "
        f"Resp={per_task_out['response']['oof_macro_f1']:.4f}, "
        f"Sol={per_task_out['solution']['oof_macro_f1']:.4f}) | {elapsed}s",
        flush=True,
    )
    return summary


SINGLE_SPECS: List[Tuple[str, Dict[str, Any]]] = [
    ("resp_v6_baseline", dict(task="response", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="comment_only", loss_type="ce_weighted", text_mode="raw", reply_only_pool=False, epochs=6, lr=4e-5, max_length=256)),
    ("resp_xlmr_raw", dict(task="response", backbone_id=XLMR_ID, context_mode="comment_only", loss_type="ce_weighted", text_mode="raw", reply_only_pool=False, epochs=6, lr=4e-5, max_length=256)),
    ("resp_xlmr_affect", dict(task="response", backbone_id=XLMR_ID, context_mode="comment_only", loss_type="ce_weighted", text_mode="emoji_affect_slang", reply_only_pool=False, epochs=6, lr=4e-5, max_length=256)),
    ("resp_muril_replypool", dict(task="response", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="parent_pair", loss_type="ce_weighted", text_mode="emoji_affect_slang", reply_only_pool=True, epochs=6, lr=4e-5, max_length=256)),
    ("resp_xlmr_replypool", dict(task="response", backbone_id=XLMR_ID, context_mode="parent_pair", loss_type="ce_weighted", text_mode="emoji_affect_slang", reply_only_pool=True, epochs=6, lr=4e-5, max_length=256)),
    ("sent_v6_baseline", dict(task="sentiment", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="comment_only", loss_type="ce_weighted", text_mode="raw", epochs=6, lr=4e-5, max_length=256)),
    ("sent_muril_affect", dict(task="sentiment", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="comment_only", loss_type="ce_weighted", text_mode="emoji_affect_slang", epochs=6, lr=4e-5, max_length=256)),
    ("emo_v6_baseline", dict(task="emotion", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="comment_only", loss_type="asl", text_mode="raw", epochs=6, lr=4e-5, max_length=256)),
    ("emo_muril_affect", dict(task="emotion", backbone_id=DAPT_STRICT_WWM_DIR, context_mode="comment_only", loss_type="asl", text_mode="emoji_affect_slang", epochs=6, lr=4e-5, max_length=256)),
]

MTL_SPECS: List[Tuple[str, Dict[str, Any]]] = [
    ("mtl3_muril_rdrop", dict(backbone_id=DAPT_STRICT_WWM_DIR, text_mode="emoji_affect_slang", rdrop_alpha=0.5, epochs=6, lr=4e-5)),
    ("mtl3_xlmr_rdrop", dict(backbone_id=XLMR_ID, text_mode="emoji_affect_slang", rdrop_alpha=0.5, epochs=6, lr=4e-5)),
]


def main() -> None:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Run a single experiment ID in an isolated process")
    parser.add_argument("--summarize", action="store_true", help="Aggregate all cached results and run OOF selection")
    args = parser.parse_args()

    t0 = time.time()
    v1 = load_v1_records_and_frozen_splits()
    trainval = v1["trainval"]
    test_rows = v1["test"]

    single_map = dict(SINGLE_SPECS)
    mtl_map = dict(MTL_SPECS)

    if args.only:
        eid = args.only
        if eid in single_map:
            run_single_task_5fold(eid, trainval=trainval, test_rows=test_rows, **single_map[eid])
        elif eid in mtl_map:
            run_mtl3_5fold(eid, trainval=trainval, test_rows=test_rows, **mtl_map[eid])
        else:
            raise ValueError(f"Unknown exp_id: {eid}")
        return

    if not args.summarize:
        # Orchestrate every experiment in a fresh Python subprocess to guarantee zero CUDA/Windows pool leaks
        all_ids = [eid for eid, _ in SINGLE_SPECS] + [eid for eid, _ in MTL_SPECS]
        for eid in all_ids:
            if (OUT_DIR / f"{eid}.json").is_file() and (OUT_DIR / f"{eid}.npz").is_file():
                print(f"[SKIP CACHED] {eid}", flush=True)
                continue
            print(f"\n>>> Launching isolated subprocess for {eid} ...", flush=True)
            proc = subprocess.run([sys.executable, __file__, "--only", eid], check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"Subprocess failed for {eid} with exit code {proc.returncode}")

    results: Dict[str, Any] = {"single_task": {}, "mtl3": {}}
    for eid, _ in SINGLE_SPECS:
        results["single_task"][eid] = json.loads((OUT_DIR / f"{eid}.json").read_text(encoding="utf-8"))
    for eid, _ in MTL_SPECS:
        results["mtl3"][eid] = json.loads((OUT_DIR / f"{eid}.json").read_text(encoding="utf-8"))

    # Strict OOF-ONLY Selection & Single Final Held-Out Comparison
    selection: Dict[str, Any] = {}
    for task, base_id, cand_ids in [
        ("response", "resp_v6_baseline", ["resp_v6_baseline", "resp_xlmr_raw", "resp_xlmr_affect", "resp_muril_replypool", "resp_xlmr_replypool"]),
        ("sentiment", "sent_v6_baseline", ["sent_v6_baseline", "sent_muril_affect"]),
        ("emotion", "emo_v6_baseline", ["emo_v6_baseline", "emo_muril_affect"]),
    ]:
        cands = []
        for cid in cand_ids:
            r = results["single_task"][cid]
            cands.append({"id": cid, "source": "single_task", "oof_macro_f1": r["oof_macro_f1"], "oof_ece": r["oof_ece"], "test_macro_f1": r["test_macro_f1"], "test_ece": r["test_ece"]})
        if task in MTL3_TASKS:
            for mid in ("mtl3_muril_rdrop", "mtl3_xlmr_rdrop"):
                mr = results["mtl3"][mid]["tasks"][task]
                cands.append({"id": f"{mid}::{task}", "source": "mtl3", "oof_macro_f1": mr["oof_macro_f1"], "oof_ece": mr["oof_ece"], "test_macro_f1": mr["test_macro_f1"], "test_ece": mr["test_ece"]})

        base_entry = next(c for c in cands if c["id"] == base_id)
        winner = max(cands, key=lambda c: (c["oof_macro_f1"], -c["oof_ece"]))
        oof_gain = round(winner["oof_macro_f1"] - base_entry["oof_macro_f1"], 4)
        promote = (winner["id"] != base_id) and (oof_gain >= 0.010)

        boot = None
        if winner["id"] != base_id:
            y_te = encode_task_targets(test_rows, task)
            base_npz = np.load(OUT_DIR / f"{base_id}.npz")
            base_tp = base_npz["test_probs"]
            if winner["source"] == "single_task":
                win_tp = np.load(OUT_DIR / f"{winner['id']}.npz")["test_probs"]
                base_thr = results["single_task"][base_id].get("thresholds")
                win_thr = results["single_task"][winner["id"]].get("thresholds")
            else:
                mid = winner["id"].split("::")[0]
                win_tp = np.load(OUT_DIR / f"{mid}.npz")[f"{task}_test_probs"]
                base_thr = None
                win_thr = None
            boot = thread_clustered_bootstrap_ci(
                test_rows, task, y_te, win_tp,
                probs_b=base_tp, thresholds_a=win_thr, thresholds_b=base_thr, n_boot=2000, seed=SEED,
            )

        selection[task] = {
            "baseline": base_entry,
            "oof_winner": winner,
            "oof_gain": oof_gain,
            "promoted_by_oof_gate": promote,
            "frozen_test_delta": round(winner["test_macro_f1"] - base_entry["test_macro_f1"], 4),
            "paired_bootstrap": boot,
            "all_candidates": cands,
        }

    payload = {
        "elapsed_seconds": round(time.time() - t0, 1),
        "protocol": "Strict OOF-only selection on 5-fold post_id-grouped V1 trainval (367), max_length=256, lr=4e-5; promotion requires >= +1.0 pp OOF Macro F1; frozen test (92) evaluated once post-selection.",
        "selection_summary": selection,
        "experiments": results,
    }
    save_json(OUT_JSON, payload)

    print("\n=== STAGE 3 OOF-ONLY SELECTION SUMMARY ===", flush=True)
    for task, info in selection.items():
        b = info["baseline"]
        w = info["oof_winner"]
        print(
            f"{task:10s} | V6 OOF={b['oof_macro_f1']:.4f} -> Winner={w['id']:24s} OOF={w['oof_macro_f1']:.4f} "
            f"(ΔOOF={info['oof_gain']:+.4f}, promote={info['promoted_by_oof_gate']}) | "
            f"Frozen Test: {b['test_macro_f1']:.4f} -> {w['test_macro_f1']:.4f} (ΔTest={info['frozen_test_delta']:+.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
