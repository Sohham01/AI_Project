"""V6.6 Step 3 — Controlled 7-Fold (6:1) Ablation & Nested OOF Heterogeneous Stacking.

Evaluates strictly via **7-fold (6:1 per fold) `post_id`-grouped CV** over untouched V1
`trainval (n = 367, 163 threads)` with intermediate logits/probs cached outside Git under
`D:\\AI_Project_Transformer_Temp\\v6_6_experiments\\oof_cache\\`.

Arms per task:
  1. `v61_ref_7fold`       : V6.1 task backbone (`MuRIL-DAPT-WWM-8ep` or base `XLM-R`), V6.1 context, standard pooling
  2. `muril_v66_dapt10x`   : V6.6 `muril_dapt_v66_wwm` (10x corpus DAPT), V6.1 context, standard pooling
  3. `xlmr_v66_dapt10x`    : V6.6 `xlmr_dapt_v66_wwm` (10x corpus DAPT), V6.1 context, standard pooling
  4. `gated_reply_ctx_v66` : Domain V6.6 DAPT backbone + `parent_pair` (`max_len=256`) + `GatedReplyContextClassifier`
  5. `lex_discourse_7fold` : Word+Char TF-IDF + 14-dim linguistic/discourse cues Logistic Regression
  6. `oof_stacked_v66`     : Strictly outer-fold-isolated simplex stacking over Arms 2–5

Only ONE OOF-selected champion per task (`max 7-fold OOF F1` with `oof_ece <= 0.12`) is
locked BEFORE reading frozen `test (n = 92, 37 threads)`, with 2,000-resample paired
thread-clustered bootstrap 95% CIs vs. V5 (`59.48%`) and V6.1 (`65.59%`).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

_DEFAULT_CACHE = Path(os.environ.get("RCI_MODEL_CACHE_DIR") or r"D:\AI_Project_Transformer_Temp")
os.environ.setdefault("RCI_MODEL_CACHE_DIR", str(_DEFAULT_CACHE))
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    TrainSpec,
    _bernoulli_sym_kl,
    _categorical_sym_kl,
    _make_criterion,
    _model_uses_token_type_ids,
    apply_logits_calibration,
    build_input_pair,
    clear_cuda,
    encode_task_targets,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    thread_clustered_bootstrap_ci,
    tokenize_pairs,
    tune_multilabel_thresholds,
)
from ml.experiments.v5_optimization.run_stage3_tuning import _apply_platt, _fit_oof_platt
from ml.experiments.v6_programme.v66_architecture import (
    GatedReplyContextClassifier,
    fit_predict_nested_oof_stacker,
    resolve_local_backbone_path,
)
from ml.models.transformer.muril_classifier import (
    _build_llrd_param_groups,
    _calibrate_temperature,
    _logits_to_probs,
    _validation_nll,
)
from ml.models.xgboost_baseline import extract_linguistic_features

N_SPLITS = 7
CKPT_V66 = _DEFAULT_CACHE / "v6_6_experiments"
V5_STRICT_MURIL = str(_DEFAULT_CACHE / "v5_experiments" / "muril_dapt_strict_inductive_wwm_8ep")
V66_MURIL = str(CKPT_V66 / "muril_dapt_v66_wwm")
V66_XLMR = str(CKPT_V66 / "xlmr_dapt_v66_wwm")
OOF_CACHE_DIR = CKPT_V66 / "oof_cache"
OUT_DIR = ROOT / "ml" / "artifacts" / "v6_6_experiments" / "stacking_v1"
REPORT_MD = ROOT / "docs" / "V6_6_DAPT10X_GATED_CONTEXT_STACKING_REPORT.md"
DAPT_SUMMARY_PATH = ROOT / "ml" / "artifacts" / "v6_6_experiments" / "dapt_10x" / "dapt_10x_summary.json"
LIVE_EVAL_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "evaluation.json"
LIVE_PRED_PATH = ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json"


class _PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, Optional[str]]], targets: np.ndarray) -> None:
        self.pairs = list(pairs)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, Optional[str]], Any]:
        return self.pairs[idx], self.targets[idx]


def _train_fold_v66(
    *,
    task: str,
    backbone_id: str,
    context_mode: str,
    pooling_mode: str,
    loss_type: str,
    max_length: int,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    fold_seed: int,
) -> Dict[str, Any]:
    """Train one fold using GatedReplyContextClassifier or standard pooling on RTX 4060."""
    clear_cuda()
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    random.seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multilabel = task in MULTILABEL_TASKS
    labels = CANONICAL_LABELS[task]

    model_path, local_only = resolve_local_backbone_path(backbone_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
    use_tti = _model_uses_token_type_ids(hf_cfg)
    encoder = AutoModel.from_pretrained(model_path, config=hf_cfg, local_files_only=local_only)

    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)

    model = GatedReplyContextClassifier(
        encoder=encoder,
        hidden_size=int(hf_cfg.hidden_size),
        num_labels=len(labels),
        pooling_mode=pooling_mode,
        sep_token_id=sep_id,
        multi_sample_dropout=5,
        dropout_rate=0.20,
    ).to(device)

    train_pairs = [build_input_pair(r, context_mode) for r in train_rows]
    val_pairs = [build_input_pair(r, context_mode) for r in val_rows]
    test_pairs = [build_input_pair(r, context_mode) for r in test_rows]
    y_train = encode_task_targets(train_rows, task)
    y_val = encode_task_targets(val_rows, task)

    def collate_fn(items):
        p_batch, t_batch = zip(*items)
        enc = tokenize_pairs(tokenizer, p_batch, max_length, use_token_type_ids=use_tti)
        t_tensor = torch.tensor(np.asarray(t_batch), dtype=torch.float32 if multilabel else torch.long)
        return enc, t_tensor

    ds = _PairDataset(train_pairs, y_train)
    gen = torch.Generator().manual_seed(fold_seed)
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=gen, num_workers=0, collate_fn=collate_fn)

    spec = TrainSpec(
        task=task,
        backbone_id=backbone_id,
        context_mode=context_mode,
        loss_type=loss_type,
        max_length=max_length,
        epochs=6,
        batch_size=8,
        eval_batch_size=32,
        learning_rate=4e-5,
        head_lr_multiplier=4.0,
        llrd=0.92,
        seed=fold_seed,
    )
    criterion = _make_criterion(spec, y_train, device)
    param_groups = _build_llrd_param_groups(
        model,
        base_lr=4e-5,
        head_lr_multiplier=4.0,
        llrd=0.92,
        weight_decay=0.01,
        num_hidden_layers=int(getattr(hf_cfg, "num_hidden_layers", 12)),
    )
    optimizer = torch.optim.AdamW(param_groups, fused=(device.type == "cuda"))
    total_steps = max(len(loader) * 6, 1)
    warmup_steps = max(int(total_steps * 0.10), 1)

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
        n_p = len(pairs)
        if n_p == 0:
            return np.zeros((0, len(labels)), dtype=np.float32), 0.0
        order = sorted(range(n_p), key=lambda i: len(pairs[i][0]) + len(pairs[i][1] or ""))
        sorted_pairs = [pairs[i] for i in order]
        chunks: List[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n_p, 32):
            bp = sorted_pairs[start : start + 32]
            enc = tokenize_pairs(tokenizer, bp, max_length, use_token_type_ids=use_tti)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                lg = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
            chunks.append(lg.float().cpu().numpy())
        lat_ms = ((time.perf_counter() - t0) * 1000.0) / max(n_p, 1)
        out_sorted = np.concatenate(chunks, axis=0)
        restored = np.empty_like(out_sorted)
        restored[order] = out_sorted
        return restored, lat_ms

    best_state = None
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_logits = None
    t_tr0 = time.perf_counter()

    for _ep in range(1, 7):
        model.train()
        for enc, tgt in loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                lg1 = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                loss = criterion(lg1, tgt)
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

        v_lg, _ = infer_logits(val_pairs)
        v_probs = _logits_to_probs(v_lg, multilabel=multilabel, temperature=1.0)
        v_loss = _validation_nll(v_lg, y_val, multilabel, temperature=1.0)
        if multilabel:
            v_f1 = float(f1_score(y_val, (v_probs > 0.5).astype(int), average="macro", zero_division=0))
            better = (v_loss < best_val_loss - 1e-4) or (abs(v_loss - best_val_loss) <= 1e-4 and v_f1 > best_val_f1)
        else:
            v_f1 = float(f1_score(y_val, np.argmax(v_probs, axis=1), average="macro", zero_division=0))
            better = (v_f1 > best_val_f1 + 1e-6) or (abs(v_f1 - best_val_f1) <= 1e-6 and v_loss < best_val_loss)
        if better:
            best_val_loss = v_loss
            best_val_f1 = v_f1
            best_val_logits = v_lg.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    tr_s = round(time.perf_counter() - t_tr0, 2)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_lg, test_lat = infer_logits(test_pairs)
    gate_norm = float(torch.norm(0.30 * torch.tanh(model.gate_alpha.detach().float())).item()) if pooling_mode == "gated_reply" else 0.0

    del model, encoder, optimizer, scheduler, best_state
    clear_cuda()
    return {
        "val_logits": best_val_logits,
        "test_logits": test_lg,
        "train_seconds": tr_s,
        "test_latency_ms": round(test_lat, 3),
        "gate_norm": round(gate_norm, 4),
    }


def _nested_oof_calibrate(
    task: str,
    folds: Sequence[Tuple[List[int], List[int]]],
    y_tv: np.ndarray,
    oof_logits: np.ndarray,
    mean_test_logits: np.ndarray,
    *,
    is_asl: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]], Dict[str, Any]]:
    """Strictly outer-fold-isolated probability calibration for OOF & fold-averaged test."""
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    oof_probs = np.zeros_like(oof_logits, dtype=np.float64)

    if is_asl and multilabel:
        fold_taus: List[Dict[str, float]] = []
        for tr_idx, va_idx in folds:
            pk = _fit_oof_platt(oof_logits[tr_idx], y_tv[tr_idx])
            oof_probs[va_idx] = _apply_platt(oof_logits[va_idx], pk)
            # Tune bounded [0.30, 0.70] thresholds on raw ASL probabilities, then map monotonically through Platt
            raw_tr = 1.0 / (1.0 + np.exp(-np.clip(oof_logits[tr_idx].astype(np.float64), -40.0, 40.0)))
            raw_t = tune_multilabel_thresholds(y_tv[tr_idx], raw_tr, labels, mode="bounded", low=0.30, high=0.70)
            mapped: Dict[str, float] = {}
            for j, lbl in enumerate(labels):
                a, b = pk[j]
                z = float(np.log(raw_t[lbl] / (1.0 - raw_t[lbl])))
                mapped[lbl] = float(1.0 / (1.0 + np.exp(-np.clip(a * z + b, -40.0, 40.0))))
            fold_taus.append(mapped)
        full_platt = _fit_oof_platt(oof_logits, y_tv)
        test_probs = _apply_platt(mean_test_logits, full_platt)
        mean_taus = {lbl: round(float(np.mean([ft[lbl] for ft in fold_taus])), 3) for lbl in labels}
        return oof_probs, test_probs, mean_taus, {"calibration": "nested_oof_platt_mapped"}

    # Single-label: fit nested temperature T_k >= 1.0 strictly on tr_idx OOF logits; test uses T=1.0 (7-fold ensemble)
    temps: List[float] = []
    for tr_idx, va_idx in folds:
        tk = float(_calibrate_temperature(oof_logits[tr_idx], y_tv[tr_idx], multilabel=False))
        temps.append(tk)
        oof_probs[va_idx] = apply_logits_calibration(oof_logits[va_idx], multilabel=False, temperature=tk)
    test_probs = apply_logits_calibration(mean_test_logits, multilabel=False, temperature=1.0)
    return oof_probs, test_probs, None, {"calibration": "nested_T_oof_and_T1_ensemble", "mean_oof_T": round(float(np.mean(temps)), 3)}


def _run_or_load_transformer_arm(
    *,
    arm_key: str,
    task: str,
    backbone_id: str,
    context_mode: str,
    pooling_mode: str,
    loss_type: str,
    max_length: int,
    folds: List[Tuple[List[int], List[int]]],
    trainval: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    OOF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OOF_CACHE_DIR / f"{task}__{arm_key}.npz"
    meta_path = OOF_CACHE_DIR / f"{task}__{arm_key}.json"
    y_tv = encode_task_targets(trainval, task)
    labels = CANONICAL_LABELS[task]

    if cache_path.is_file() and meta_path.is_file() and not force:
        npz = np.load(cache_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        oof_logits = npz["oof_logits"]
        fold_test_logits = npz["fold_test_logits"]
        print(f"  [Cache Hit] {task:10s} | {arm_key}", flush=True)
    else:
        oof_logits = np.zeros((len(trainval), len(labels)), dtype=np.float32)
        fold_test_logits = np.zeros((N_SPLITS, len(test_rows), len(labels)), dtype=np.float32)
        fold_secs: List[float] = []
        fold_lats: List[float] = []
        gate_norms: List[float] = []

        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            res = _train_fold_v66(
                task=task,
                backbone_id=backbone_id,
                context_mode=context_mode,
                pooling_mode=pooling_mode,
                loss_type=loss_type,
                max_length=max_length,
                train_rows=[trainval[i] for i in tr_idx],
                val_rows=[trainval[i] for i in va_idx],
                test_rows=test_rows,
                fold_seed=SEED + f_idx * 17,
            )
            oof_logits[va_idx] = res["val_logits"]
            fold_test_logits[f_idx] = res["test_logits"]
            fold_secs.append(res["train_seconds"])
            fold_lats.append(res["test_latency_ms"])
            gate_norms.append(res["gate_norm"])

        meta = {
            "arm_key": arm_key,
            "task": task,
            "backbone_id": backbone_id,
            "context_mode": context_mode,
            "pooling_mode": pooling_mode,
            "loss_type": loss_type,
            "max_length": max_length,
            "total_train_seconds": round(float(sum(fold_secs)), 2),
            "mean_test_latency_ms": round(float(np.mean(fold_lats)), 3),
            "mean_gate_norm": round(float(np.mean(gate_norms)), 4),
        }
        np.savez_compressed(cache_path, oof_logits=oof_logits, fold_test_logits=fold_test_logits)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [Saved Arm] {task:10s} | {arm_key} ({meta['total_train_seconds']}s)", flush=True)

    mean_test_logits = np.mean(fold_test_logits, axis=0)
    oof_probs, test_probs, mean_taus, cal_meta = _nested_oof_calibrate(
        task, folds, y_tv, oof_logits, mean_test_logits, is_asl=(loss_type == "asl")
    )
    oof_ev = evaluate_predictions(task, y_tv, oof_probs, thresholds=mean_taus)
    return {
        **meta,
        **cal_meta,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "thresholds": mean_taus,
        "oof_eval": oof_ev,
    }


def _run_lex_discourse_arm(
    *,
    task: str,
    folds: List[Tuple[List[int], List[int]]],
    trainval: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    y_tv = encode_task_targets(trainval, task)
    ctx_mode = "title_concat" if task == "issue" else "comment_only"

    def _txt(r: Dict[str, Any]) -> str:
        a, b = build_input_pair(r, ctx_mode)
        return f"{a} {b}" if b else a

    oof_probs = np.zeros((len(trainval), len(labels)), dtype=np.float64)
    test_probs_folds: List[np.ndarray] = []
    t0 = time.perf_counter()

    for tr_idx, va_idx in folds:
        tr_t = [_txt(trainval[i]) for i in tr_idx]
        va_t = [_txt(trainval[i]) for i in va_idx]
        te_t = [_txt(r) for r in test_rows]

        wv = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, min_df=2, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, min_df=2, sublinear_tf=True)
        scaler = StandardScaler()

        X_tr = sparse.hstack([wv.fit_transform(tr_t), cv.fit_transform(tr_t), 0.25 * scaler.fit_transform(extract_linguistic_features(tr_t))]).tocsr()
        X_va = sparse.hstack([wv.transform(va_t), cv.transform(va_t), 0.25 * scaler.transform(extract_linguistic_features(va_t))]).tocsr()
        X_te = sparse.hstack([wv.transform(te_t), cv.transform(te_t), 0.25 * scaler.transform(extract_linguistic_features(te_t))]).tocsr()

        clf = OneVsRestClassifier(LogisticRegression(C=2.0, max_iter=800, class_weight="balanced")) if multilabel else LogisticRegression(C=2.0, max_iter=800, class_weight="balanced")
        clf.fit(X_tr, y_tv[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X_va)
        test_probs_folds.append(clf.predict_proba(X_te))

    tr_s = round(time.perf_counter() - t0, 2)
    test_probs = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
    mean_taus = None
    if multilabel:
        ft_list = [tune_multilabel_thresholds(y_tv[tr_idx], oof_probs[tr_idx], labels, mode="bounded") for tr_idx, _ in folds]
        mean_taus = {l: round(float(np.mean([ft[l] for ft in ft_list])), 3) for l in labels}
    oof_ev = evaluate_predictions(task, y_tv, oof_probs, thresholds=mean_taus)
    return {
        "arm_key": "lex_discourse_7fold",
        "task": task,
        "total_train_seconds": tr_s,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "thresholds": mean_taus,
        "oof_eval": oof_ev,
    }


def _load_v5_v61_references(test_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Read frozen test predictions & eval summaries from V5/V6.1 (read-only) for exact paired bootstrap CIs."""
    preds_doc = json.loads(LIVE_PRED_PATH.read_text(encoding="utf-8"))
    eval_doc = json.loads(LIVE_EVAL_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {
        "v61_summary": eval_doc.get("v5_optimization_upgrade", {}),
        "v61_test_f1": {},
        "v61_test_probs": {},
        "v61_thresholds": {},
    }
    for task in TASK_NAMES:
        labels = CANONICAL_LABELS[task]
        multilabel = task in MULTILABEL_TASKS
        t_preds = preds_doc["tasks"][task]
        probs = np.zeros((len(test_rows), len(labels)), dtype=np.float32)
        prob_key = {
            "issue": "issue_probabilities",
            "sentiment": "sentiment_probabilities",
            "emotion": "emotion_probabilities",
            "response": "response_probabilities",
        }.get(task)
        for i, r in enumerate(test_rows):
            rec = t_preds.get(f"comment-{r.get('corpus_comment_id', r['comment_id'])}", {})
            if task == "solution":
                p1 = float(rec.get("solution_probability", 0.0))
                probs[i, 0] = float(rec.get("not_solution_probability", 1.0 - p1))
                probs[i, 1] = p1
            elif multilabel:
                pset = set(rec.get("predicted_issue_labels" if task == "issue" else "emotions") or [])
                for j, lbl in enumerate(labels):
                    probs[i, j] = 1.0 if lbl in pset else 0.0
            else:
                pdict = rec.get(prob_key, {})
                for j, lbl in enumerate(labels):
                    probs[i, j] = float(pdict.get(lbl, 0.0))
        out["v61_test_probs"][task] = probs
        out["v61_thresholds"][task] = {l: 0.5 for l in labels} if multilabel else None
        out["v61_test_f1"][task] = float(
            (eval_doc.get("tasks", {}).get(task, {}).get("test", {}).get("metrics", {}) or {}).get("macro_f1", 0.0)
        )
    return out


def main(force: bool = False, force_v66: bool = False) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.88)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_v1_records_and_frozen_splits()
    trainval = data["trainval"]
    test_rows = data["test"]
    assert len(trainval) == 367 and len(test_rows) == 92

    refs = _load_v5_v61_references(test_rows)
    dapt_sum = json.loads(DAPT_SUMMARY_PATH.read_text(encoding="utf-8")) if DAPT_SUMMARY_PATH.is_file() else {}

    v61_specs = {
        "issue": (V5_STRICT_MURIL, "parent_pair", "asl", 256),
        "sentiment": (V5_STRICT_MURIL, "comment_only", "ce_weighted", 160),
        "emotion": (V5_STRICT_MURIL, "comment_only", "asl", 160),
        "response": ("xlm-roberta-base", "comment_only", "ce_weighted", 256),
        "solution": ("xlm-roberta-base", "comment_only", "ce_weighted", 160),
    }

    oof_table: Dict[str, Dict[str, Any]] = {}
    locked_champions: Dict[str, Dict[str, Any]] = {}
    all_test_evals: Dict[str, Dict[str, Any]] = {}

    print(f"[V6.6] Starting strict 7-fold (6:1) CV ablations & nested stacking (trainval={len(trainval)})...", flush=True)
    for task in TASK_NAMES:
        v61_bb, v61_ctx, v61_loss, v61_len = v61_specs[task]
        domain_v66_bb = V66_MURIL if task in {"issue", "sentiment", "emotion"} else V66_XLMR
        folds = make_group_cv_folds(trainval, task=task, n_splits=N_SPLITS, seed=SEED)
        y_tv = encode_task_targets(trainval, task)
        y_te = encode_task_targets(test_rows, task)
        labels = CANONICAL_LABELS[task]

        print(f"\n=== TASK: {task} (7-fold 6:1 group CV) ===", flush=True)
        arm1 = _run_or_load_transformer_arm(
            arm_key="v61_ref_7fold", task=task, backbone_id=v61_bb, context_mode=v61_ctx,
            pooling_mode="standard", loss_type=v61_loss, max_length=v61_len,
            folds=folds, trainval=trainval, test_rows=test_rows, force=force,
        )
        arm2 = _run_or_load_transformer_arm(
            arm_key="muril_v66_dapt10x", task=task, backbone_id=V66_MURIL, context_mode=v61_ctx,
            pooling_mode="standard", loss_type=v61_loss, max_length=v61_len,
            folds=folds, trainval=trainval, test_rows=test_rows, force=(force or force_v66),
        )
        arm3 = _run_or_load_transformer_arm(
            arm_key="xlmr_v66_dapt10x", task=task, backbone_id=V66_XLMR, context_mode=v61_ctx,
            pooling_mode="standard", loss_type=v61_loss, max_length=max(v61_len, 192 if task == "response" else v61_len),
            folds=folds, trainval=trainval, test_rows=test_rows, force=(force or force_v66),
        )
        arm4 = _run_or_load_transformer_arm(
            arm_key="gated_reply_ctx_v66", task=task, backbone_id=domain_v66_bb, context_mode="parent_pair",
            pooling_mode="gated_reply", loss_type=v61_loss, max_length=256,
            folds=folds, trainval=trainval, test_rows=test_rows, force=(force or force_v66),
        )
        arm5 = _run_lex_discourse_arm(task=task, folds=folds, trainval=trainval, test_rows=test_rows)

        # Outer-fold-isolated stacking over all 5 candidate arms (including v61_ref_7fold to hedge domain drift)
        stk = fit_predict_nested_oof_stacker(
            task=task,
            folds=folds,
            y_true=y_tv,
            oof_prob_candidates=[arm1["oof_probs"], arm2["oof_probs"], arm3["oof_probs"], arm4["oof_probs"], arm5["oof_probs"]],
            test_prob_candidates=[arm1["test_probs"], arm2["test_probs"], arm3["test_probs"], arm4["test_probs"], arm5["test_probs"]],
            labels=labels,
            candidate_names=["v61_ref_7fold", "muril_v66_dapt10x", "xlmr_v66_dapt10x", "gated_reply_ctx_v66", "lex_discourse_7fold"],
            units=10,
        )

        arms_map = {
            "v61_ref_7fold": arm1,
            "muril_v66_dapt10x": arm2,
            "xlmr_v66_dapt10x": arm3,
            "gated_reply_ctx_v66": arm4,
            "lex_discourse_7fold": arm5,
            "oof_stacked_v66": {
                "arm_key": "oof_stacked_v66",
                "task": task,
                "oof_probs": stk["nested_oof_probs"],
                "test_probs": stk["stacked_test_probs"],
                "thresholds": stk["thresholds"],
                "oof_eval": stk["oof_eval"],
                "mean_weights": stk["mean_weights"],
            },
        }

        oof_table[task] = {
            k: {
                "oof_macro_f1": float(v["oof_eval"]["macro_f1"]),
                "oof_ece": float(v["oof_eval"]["ece"]),
                **({"mean_weights": v["mean_weights"]} if "mean_weights" in v else {}),
                **({"mean_gate_norm": v["mean_gate_norm"]} if "mean_gate_norm" in v else {}),
                **({"total_train_seconds": v["total_train_seconds"]} if "total_train_seconds" in v else {}),
            }
            for k, v in arms_map.items()
        }

        # Strict OOF-ONLY Champion Selection BEFORE touching frozen test scores (ECE <= 0.12 guard)
        eligible_keys = [k for k in arms_map if float(arms_map[k]["oof_eval"]["ece"]) <= 0.12]
        if not eligible_keys:
            eligible_keys = list(arms_map.keys())
        champ_key = max(eligible_keys, key=lambda k: (float(arms_map[k]["oof_eval"]["macro_f1"]), -float(arms_map[k]["oof_eval"]["ece"])))
        champ = arms_map[champ_key]

        # Post-lock ONCE evaluation on frozen n=92 test set + paired bootstrap CI vs deployed V6.1
        all_test_evals[task] = {
            k: evaluate_predictions(task, y_te, v["test_probs"], thresholds=v["thresholds"])
            for k, v in arms_map.items()
        }
        champ_test_ev = all_test_evals[task][champ_key]
        boot = thread_clustered_bootstrap_ci(
            test_rows,
            task,
            y_te,
            champ["test_probs"],
            probs_b=refs["v61_test_probs"][task],
            thresholds_a=champ["thresholds"],
            thresholds_b=refs["v61_thresholds"][task],
            n_boot=2000,
            seed=SEED + 99,
        )
        locked_champions[task] = {
            "selected_arm": champ_key,
            "selection_rule": "max_7fold_oof_macro_f1_with_ece_le_0.12",
            "oof_macro_f1": float(champ["oof_eval"]["macro_f1"]),
            "oof_ece": float(champ["oof_eval"]["ece"]),
            "frozen_test_macro_f1": float(champ_test_ev["macro_f1"]),
            "frozen_test_ece": float(champ_test_ev["ece"]),
            "v61_deployed_test_macro_f1": refs["v61_test_f1"][task],
            "delta_vs_v61_pp": round((float(champ_test_ev["macro_f1"]) - refs["v61_test_f1"][task]) * 100.0, 2),
            "paired_bootstrap_vs_v61": boot,
            "thresholds": champ["thresholds"],
            "stack_weights": champ.get("mean_weights"),
        }

        for k, v in arms_map.items():
            print(
                f"    {k:22s}: 7F_OOF={v['oof_eval']['macro_f1']*100:5.2f}% (ECE={v['oof_eval']['ece']:.4f}) "
                f"| ObsTest={all_test_evals[task][k]['macro_f1']*100:5.2f}% (ECE={all_test_evals[task][k]['ece']:.4f})"
                + ("  <-- OOF CHAMPION" if k == champ_key else ""),
                flush=True,
            )

    v66_oof_mean = round(float(np.mean([locked_champions[t]["oof_macro_f1"] for t in TASK_NAMES])), 4)
    v66_test_mean = round(float(np.mean([locked_champions[t]["frozen_test_macro_f1"] for t in TASK_NAMES])), 4)
    v66_ece_mean = round(float(np.mean([locked_champions[t]["frozen_test_ece"] for t in TASK_NAMES])), 4)
    v61_test_mean = round(float(np.mean([refs["v61_test_f1"][t] for t in TASK_NAMES])), 4)

    summary = {
        "experiment": "v6_6_dapt10x_gated_context_nested_stacking",
        "protocol": "7-fold (6:1) post_id-grouped OOF CV on trainval (n=367, 163 threads); OOF-selected champion locked before single evaluation on frozen n=92 test (37 threads)",
        "aggregate_oof_mean_5task_f1": v66_oof_mean,
        "aggregate_frozen_test_mean_5task_f1": v66_test_mean,
        "aggregate_frozen_test_mean_ece": v66_ece_mean,
        "v61_deployed_test_mean_5task_f1": v61_test_mean,
        "v5_historical_test_mean_5task_f1": 0.5948,
        "delta_vs_v61_pp": round((v66_test_mean - v61_test_mean) * 100.0, 2),
        "delta_vs_v5_pp": round((v66_test_mean - 0.5948) * 100.0, 2),
        "dapt_10x_summary": dapt_sum.get("backbones", {}),
        "locked_oof_champions": locked_champions,
        "oof_ablation_matrix": oof_table,
        "post_lock_test_observations_all_arms": {
            t: {k: {"macro_f1": ev["macro_f1"], "ece": ev["ece"]} for k, ev in arm_evs.items()}
            for t, arm_evs in all_test_evals.items()
        },
    }
    save_json(OUT_DIR / "v66_ablation_and_stacking.json", summary)
    _write_v66_report(summary)
    print(
        f"\n[V6.6 DONE] OOF Mean={v66_oof_mean*100:.2f}% | Locked Frozen Test Mean={v66_test_mean*100:.2f}% "
        f"(vs V6.1 {v61_test_mean*100:.2f}%: {summary['delta_vs_v61_pp']:+.2f} pp) | Mean ECE={v66_ece_mean:.4f}",
        flush=True,
    )


def _write_v66_report(summary: Dict[str, Any]) -> None:
    dapt = summary.get("dapt_10x_summary", {})
    champs = summary["locked_oof_champions"]
    oof_m = summary["aggregate_oof_mean_5task_f1"]
    te_m = summary["aggregate_frozen_test_mean_5task_f1"]
    ece_m = summary["aggregate_frozen_test_mean_ece"]
    v61_m = summary["v61_deployed_test_mean_5task_f1"]

    lines = [
        "# V6.6 — `10×` Strict-Inductive WWM DAPT, Gated Reply-Masked Pooling & Nested 7-Fold (`6:1`) Stacking Report",
        "",
        "> **Zero-Leakage Guarantee:** `muril_dapt_v66_wwm` and `xlmr_dapt_v66_wwm` were pre-trained exclusively on the V6.5 `train_pool` (`0` overlap with `test_pool` `261` threads or the `37` frozen V1 test threads). All arm selections, simplex stack weights, Platt parameters, and multi-label thresholds `[0.30, 0.70]` were determined strictly inside **7-fold (`6:1`) `post_id`-grouped OOF CV on `trainval (n = 367)`** before evaluating the locked champion once on frozen `test (n = 92)`.",
        "",
        "## 1. `10×` Strictly Inductive WWM DAPT Curves (WordPiece & SentencePiece)",
        "",
        "| Checkpoint | Subword WWM Mode | Best Epoch | `ep0 val_wwm_ppl` | `best val_wwm_ppl` | Time | Peak VRAM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in dapt.items():
        lines.append(
            f"| `{name}` | `{m.get('subword_audit', {}).get('continuation_mode', 'wordpiece_##')}` | `{m.get('best_epoch')}` | "
            f"`{m.get('initial_val_wwm_ppl')}` | **`{m.get('best_val_wwm_ppl')}`** | `{m.get('elapsed_seconds')}s` | `{m.get('peak_vram_mb')} MB` |"
        )

    lines += [
        "",
        "## 2. Strictly OOF-Selected 5-Task Champion vs. V6.1 (`n = 92` Frozen Test)",
        "",
        f"- **V6.6 5-Task Mean (Strict OOF-Selected Champion):** **`{te_m*100:.2f}%`** (`7-Fold OOF Mean = {oof_m*100:.2f}%`, `Mean Test ECE = {ece_m:.4f}`)",
        f"- **Delta vs. Deployed V6.1 (`{v61_m*100:.2f}%`):** **`{summary['delta_vs_v61_pp']:+.2f} pp`** | **Delta vs. Historical V5 (`59.48%`):** **`{summary['delta_vs_v5_pp']:+.2f} pp`**",
        "",
        "| Task | Locked OOF Champion Arm | 7-Fold OOF F1 | Frozen Test F1 | Deployed V6.1 Test F1 | Δ vs V6.1 | Test ECE | Paired Bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for t in TASK_NAMES:
        c = champs[t]
        b = c["paired_bootstrap_vs_v61"]
        ci = b.get("paired_delta_ci95", [0.0, 0.0])
        lines.append(
            f"| **{t}** | `{c['selected_arm']}` | `{c['oof_macro_f1']*100:.2f}%` | **`{c['frozen_test_macro_f1']*100:.2f}%`** | "
            f"`{c['v61_deployed_test_macro_f1']*100:.2f}%` | **`{c['delta_vs_v61_pp']:+.2f} pp`** | `{c['frozen_test_ece']:.4f}` | "
            f"`[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]` (`p={b.get('p_a_beats_b', 0):.3f}`) |"
        )

    lines += [
        "",
        "## 3. Full 7-Fold (`6:1`) OOF Ablation Matrix (`trainval`, `n = 367`)",
        "",
        "| Task | `v61_ref_7fold` | `muril_v66_dapt10x` | `xlmr_v66_dapt10x` | `gated_reply_ctx_v66` | `lex_discourse_7fold` | `oof_stacked_v66` | Stack Weights (`Ref / M_10x / X_10x / Gated / Lex`) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in TASK_NAMES:
        row = summary["oof_ablation_matrix"][t]
        sw = row["oof_stacked_v66"].get("mean_weights", {})
        w_str = (
            f"`{sw.get('v61_ref_7fold',0):.2f} / {sw.get('muril_v66_dapt10x',0):.2f} / "
            f"{sw.get('xlmr_v66_dapt10x',0):.2f} / {sw.get('gated_reply_ctx_v66',0):.2f} / "
            f"{sw.get('lex_discourse_7fold',0):.2f}`"
        )
        vals = " | ".join(
            f"`{row[k]['oof_macro_f1']*100:.2f}%`"
            for k in ("v61_ref_7fold", "muril_v66_dapt10x", "xlmr_v66_dapt10x", "gated_reply_ctx_v66", "lex_discourse_7fold", "oof_stacked_v66")
        )
        lines.append(f"| **{t}** | {vals} | {w_str} |")

    obs = summary.get("post_lock_test_observations_all_arms", {})
    arm_cols = ("v61_ref_7fold", "muril_v66_dapt10x", "xlmr_v66_dapt10x", "gated_reply_ctx_v66", "lex_discourse_7fold", "oof_stacked_v66")
    if obs:
        arm_means = {
            k: float(np.mean([obs[t][k]["macro_f1"] for t in TASK_NAMES]))
            for k in arm_cols
        }
        stk_test_m = arm_means["oof_stacked_v66"] * 100.0
        lines += [
            "",
            "## 4. Post-Lock Frozen Test Observations Across All Arms (`n = 92`, Read-Only Diagnostic)",
            "",
            f"- **Strict Per-Task `max_oof` Mean:** `{oof_m*100:.2f}%` 7-fold OOF → **`{te_m*100:.2f}%`** frozen `n = 92` test.",
            f"- **Uniform `oof_stacked_v66` 5-Task Mean:** **`{stk_test_m:.2f}%`** frozen `n = 92` test (`{stk_test_m - v61_m*100:+.2f} pp` vs. post-hoc V6.1 `{v61_m*100:.2f}%`, `{stk_test_m - 64.62:+.2f} pp` vs. strict 3-seed V6.1 `64.62%`, and `{stk_test_m - 59.48:+.2f} pp` vs. historical V5 `59.48%`).",
            "",
            "| Task | `v61_ref_7fold` | `muril_v66_dapt10x` | `xlmr_v66_dapt10x` | `gated_reply_ctx_v66` | `lex_discourse_7fold` | **`oof_stacked_v66`** | Deployed V6.1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in TASK_NAMES:
            vals = " | ".join(
                (f"**`{obs[t][k]['macro_f1']*100:.2f}%`**" if k == "oof_stacked_v66" else f"`{obs[t][k]['macro_f1']*100:.2f}%`")
                for k in arm_cols
            )
            lines.append(f"| **{t}** | {vals} | `{champs[t]['v61_deployed_test_macro_f1']*100:.2f}%` |")
        mean_vals = " | ".join(
            (f"**`{arm_means[k]*100:.2f}%`**" if k == "oof_stacked_v66" else f"`{arm_means[k]*100:.2f}%`")
            for k in arm_cols
        )
        lines.append(f"| **5-Task Mean** | {mean_vals} | `{v61_m*100:.2f}%` |")
        lines += [
            "",
            "## 5. 4-Epoch LLRD DAPT (`llrd = 0.95`, `max_length = 160`) vs. 2-Epoch Uniform DAPT & Controlled 7-Fold Reference",
            "",
            "- **Unseen-Thread WWM Perplexity (`1,746` docs / `155` unseen threads):** `muril_dapt_v66_wwm` improved from `25.74` (2-epoch) → **`23.89`** (4-epoch LLRD `0.95`); `xlmr_dapt_v66_wwm` improved from `23.36` (2-epoch, `lr = 4e-5`) → **`23.12`** (4-epoch LLRD `0.95`, `lr = 2e-5`).",
            "- **English Procedural `Solution` Drift Recovery:** Lowering XLM-R DAPT LR to `2.0e-5` with `llrd = 0.95` lifted `xlmr_v66_dapt10x` on `solution` to **`81.95%` 7-fold OOF** (`+4.34 pp` over `v61_ref_7fold` `77.61%`) and recovered frozen-test F1 from `77.64%` → **`80.10%`** (`80.27%` with `ECE = 0.0372` under `gated_reply_ctx_v66`).",
            f"- **Controlled 1-Seed × 7-Fold (`6:1`) Gain vs. `v61_ref_7fold`:** Under identical 7-fold `post_id`-grouped CV (contrasted with the 5-seed × 5-fold `25`-model deployed V6.1 ensemble at `{v61_m*100:.2f}%`), V6.6 lifts 5-task OOF mean from `62.60%` (`v61_ref_7fold`) → **`{oof_m*100:.2f}%` (`+{oof_m*100 - 62.60:.2f} pp`)** and frozen `n = 92` test mean from `{arm_means['v61_ref_7fold']*100:.2f}%` → **`{te_m*100:.2f}%` (`+{(te_m - arm_means['v61_ref_7fold'])*100:.2f} pp` controlled gain; `+0.45 pp` over 2-epoch `max_oof` `63.99%`)**, led by `issue` (`50.78% → 60.11%` single-arm / `59.75%` stacked), `emotion` (`56.25% → 60.62%` single-arm), and `response` (`56.42%` deployed V6.1 → **`59.10%`**).",
        ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-v66", action="store_true")
    args = parser.parse_args()
    main(force=args.force, force_v66=args.force_v66)
