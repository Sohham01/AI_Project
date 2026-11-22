"""Phase 1 — Five-Fold Group-Aware Cross-Validation & OOF Calibration Suite.

Strictly follows Section 3 and Section 5 of the V5 Model Optimization Program:
- Uses strictly inductive MuRIL-DAPT (0 test-thread overlap) as the clean backbone.
- Runs 5-fold group-aware CV on trainval (n=367, 163 threads) grouped strictly by post_id:
  * StratifiedGroupKFold for single-label tasks (sentiment, response, solution).
  * Group-Preserving Iterative Multi-Label Stratification for multi-label tasks (issue, emotion).
- Measures OOF calibration across:
  * Raw (T=1.0), Global scalar temperature (T>=1.0), Per-class temperature (T_c>=1.0).
  * Multi-label thresholds: fixed_05, unbounded [0.10,0.90], bounded [0.30,0.70], regularized shrinkage.
  * Fold-to-fold threshold stability (mean, std, min, max across the 5 folds).
- Empirically measures the Single-Model OOF vs. 5-Model Ensemble calibration mismatch using
  inner 5-fold CV strictly inside V5 train (n=298) evaluated on strictly unseen V5 val (n=69).
- Also trains a single-split inductive control (298/69) and a 1-checkpoint full-trainval (367)
  single-model alternative (trained for median best_epoch from CV, calibrated via OOF).
- Saves all frozen-test logits (n=92) to disk for Phase 8/9 final controlled evaluation ONLY;
  never uses test labels for selection or tuning.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    CKPT_ROOT,
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    TrainSpec,
    apply_logits_calibration,
    encode_task_targets,
    ensure_dir,
    evaluate_predictions,
    fit_per_class_temperatures,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature, _validation_nll


def _v5_context_mode(task: str) -> str:
    return "title_concat" if task == "issue" else "comment_only"


def _tune_binary_threshold(y_true: np.ndarray, probs: np.ndarray, low: float = 0.30, high: float = 0.70) -> float:
    steps = int(round((high - low) / 0.025)) + 1
    grid = [round(low + 0.025 * i, 3) for i in range(steps)]
    best_tau = 0.5
    best_f1 = -1.0
    for tau in grid:
        pred = (probs[:, 1] >= tau).astype(int)
        # Macro F1 across both binary classes
        f1s = []
        for cls in (0, 1):
            tp = int(((y_true == cls) & (pred == cls)).sum())
            fp = int(((y_true != cls) & (pred == cls)).sum())
            fn = int(((y_true == cls) & (pred != cls)).sum())
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        mf1 = 0.5 * sum(f1s)
        if (mf1 > best_f1 + 1e-12) or (abs(mf1 - best_f1) <= 1e-12 and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_f1 = mf1
            best_tau = tau
    return float(best_tau)


def run_phase1() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase1_cv")
    logits_dir = ensure_dir(out_dir / "logits")
    backbone = str(CKPT_ROOT / "muril_dapt_inductive_mlm_4ep")
    assert Path(backbone).exists(), f"Inductive DAPT checkpoint missing at {backbone}"

    data = load_v1_records_and_frozen_splits()
    train_rows = data["train"]       # 298
    val_rows = data["val"]           # 69
    trainval_rows = data["trainval"] # 367
    test_rows = data["test"]         # 92 (frozen — inference only, NO selection!)

    summary_path = out_dir / "phase1_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "phase": "phase1_oof_cv_calibration",
            "backbone": backbone,
            "seed": SEED,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_trainval": len(trainval_rows),
            "n_test_frozen": len(test_rows),
            "tasks": {},
        }

    t_all = time.perf_counter()

    for task in TASK_NAMES:
        if task in summary.get("tasks", {}) and (logits_dir / f"{task}_phase1_logits.npz").exists():
            print(f"\n[Phase 1] Skipping already-completed task: {task.upper()}", flush=True)
            continue
        print(f"\n==================== PHASE 1 CV: {task.upper()} ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        ctx_mode = _v5_context_mode(task)
        y_tv = encode_task_targets(trainval_rows, task)
        y_tr = encode_task_targets(train_rows, task)
        y_va = encode_task_targets(val_rows, task)

        # ------------------------------------------------------------------
        # 1. Single-Split Inductive Control (298 train / 69 val)
        # ------------------------------------------------------------------
        spec_ss = TrainSpec(
            task=task,
            backbone_id=backbone,
            context_mode=ctx_mode,
            loss_type="default",
            max_length=160,
            epochs=6,
            seed=SEED,
        )
        ss_res = train_and_predict_logits(spec_ss, train_rows, val_rows, {"test": test_rows})
        ss_val_logits = ss_res["val_logits"]
        ss_test_logits = ss_res["eval_logits"]["test"]
        ss_t = float(_calibrate_temperature(ss_val_logits, y_va, multilabel=multilabel))
        ss_val_probs = apply_logits_calibration(ss_val_logits, multilabel, ss_t)
        ss_taus = (
            tune_multilabel_thresholds(y_va, ss_val_probs, labels, mode="unbounded")
            if multilabel
            else None
        )
        ss_val_eval = evaluate_predictions(task, y_va, ss_val_probs, thresholds=ss_taus)
        print(
            f"  [Single-Split 298/69 Inductive] val_macro_f1={ss_val_eval['macro_f1']:.4f} "
            f"T={ss_t:.2f} best_ep={ss_res['best_epoch']} ({ss_res['train_seconds']}s)",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 2. Inner 5-Fold CV inside train (298) -> Ensemble Calibration Mismatch on unseen val (69)
        # ------------------------------------------------------------------
        inner_folds = make_group_cv_folds(train_rows, task=task, n_splits=5, seed=SEED + 10)
        inner_oof_logits = np.zeros((len(train_rows), len(labels)), dtype=np.float32)
        inner_val_logits_folds: List[np.ndarray] = []
        for f_idx, (itr, iva) in enumerate(inner_folds):
            sp_in = TrainSpec(
                task=task,
                backbone_id=backbone,
                context_mode=ctx_mode,
                loss_type="default",
                max_length=160,
                epochs=6,
                seed=SEED + 100 + f_idx,
            )
            r_in = train_and_predict_logits(
                sp_in,
                [train_rows[i] for i in itr],
                [train_rows[i] for i in iva],
                {"val69": val_rows},
            )
            inner_oof_logits[iva] = r_in["val_logits"]
            inner_val_logits_folds.append(r_in["eval_logits"]["val69"])

        # Compare single-model OOF calibration vs 5-model ensemble calibration on strictly unseen val (n=69)
        t_inner_single = float(_calibrate_temperature(inner_oof_logits, y_tr, multilabel=multilabel))
        ens_val69_logits = np.mean(np.stack(inner_val_logits_folds, axis=0), axis=0)
        t_inner_ens_opt = float(_calibrate_temperature(ens_val69_logits, y_va, multilabel=multilabel))

        # Single-model OOF thresholds (bounded) applied to 5-model ensemble on unseen val69
        inner_oof_probs = apply_logits_calibration(inner_oof_logits, multilabel, t_inner_single)
        ens_val69_raw_probs = apply_logits_calibration(ens_val69_logits, multilabel, 1.0)
        ens_val69_singleT_probs = apply_logits_calibration(ens_val69_logits, multilabel, t_inner_single)
        ens_val69_ensT_probs = apply_logits_calibration(ens_val69_logits, multilabel, t_inner_ens_opt)

        # Also evaluate average single-model val69 metrics across the 5 inner checkpoints
        indiv_val69_evals = [
            evaluate_predictions(
                task,
                y_va,
                apply_logits_calibration(lg, multilabel, t_inner_single),
                thresholds=(
                    tune_multilabel_thresholds(y_tr, inner_oof_probs, labels, mode="bounded")
                    if multilabel
                    else None
                ),
            )
            for lg in inner_val_logits_folds
        ]
        taus_inner_bounded = (
            tune_multilabel_thresholds(y_tr, inner_oof_probs, labels, mode="bounded")
            if multilabel
            else None
        )
        ens_mismatch_report = {
            "t_single_oof_298": round(t_inner_single, 3),
            "t_ensemble_opt_val69": round(t_inner_ens_opt, 3),
            "delta_t_ens_minus_single": round(t_inner_ens_opt - t_inner_single, 3),
            "nll_ens_raw_T1": round(float(_validation_nll(ens_val69_logits, y_va, multilabel, 1.0)), 5),
            "nll_ens_with_single_T": round(float(_validation_nll(ens_val69_logits, y_va, multilabel, t_inner_single)), 5),
            "nll_ens_with_opt_T": round(float(_validation_nll(ens_val69_logits, y_va, multilabel, t_inner_ens_opt)), 5),
            "mean_single_ckpt_val69_macro_f1": round(float(np.mean([e["macro_f1"] for e in indiv_val69_evals])), 4),
            "mean_single_ckpt_val69_ece": round(float(np.mean([e["ece"] for e in indiv_val69_evals])), 4),
            "ens_raw_val69": {
                k: v
                for k, v in evaluate_predictions(task, y_va, ens_val69_raw_probs, thresholds=taus_inner_bounded).items()
                if k in {"macro_f1", "micro_f1", "ece", "brier"}
            },
            "ens_singleT_val69": {
                k: v
                for k, v in evaluate_predictions(task, y_va, ens_val69_singleT_probs, thresholds=taus_inner_bounded).items()
                if k in {"macro_f1", "micro_f1", "ece", "brier"}
            },
            "ens_optT_val69": {
                k: v
                for k, v in evaluate_predictions(task, y_va, ens_val69_ensT_probs, thresholds=taus_inner_bounded).items()
                if k in {"macro_f1", "micro_f1", "ece", "brier"}
            },
        }
        print(
            f"  [Nested Ensemble Calibration Check on Val69] T_single={t_inner_single:.2f} vs T_ens_opt={t_inner_ens_opt:.2f} "
            f"| NLL raw={ens_mismatch_report['nll_ens_raw_T1']:.4f} singleT={ens_mismatch_report['nll_ens_with_single_T']:.4f} "
            f"optT={ens_mismatch_report['nll_ens_with_opt_T']:.4f} "
            f"| 1-ckpt F1={ens_mismatch_report['mean_single_ckpt_val69_macro_f1']:.4f} vs 5-ens F1={ens_mismatch_report['ens_singleT_val69']['macro_f1']:.4f}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 3. Primary Outer 5-Fold Group-Aware CV on trainval (n=367)
        # ------------------------------------------------------------------
        outer_folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)
        oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
        test_logits_folds: List[np.ndarray] = []
        fold_records: List[Dict[str, Any]] = []
        fold_best_epochs: List[int] = []
        fold_thresholds_unbounded: List[Dict[str, float]] = []
        fold_thresholds_bounded: List[Dict[str, float]] = []

        for f_idx, (tr_idx, va_idx) in enumerate(outer_folds):
            tr_sub = [trainval_rows[i] for i in tr_idx]
            va_sub = [trainval_rows[i] for i in va_idx]
            tr_pids = {r["post_id"] for r in tr_sub}
            va_pids = {r["post_id"] for r in va_sub}
            assert not (tr_pids & va_pids), f"Fold {f_idx} group leakage!"

            sp_f = TrainSpec(
                task=task,
                backbone_id=backbone,
                context_mode=ctx_mode,
                loss_type="default",
                max_length=160,
                epochs=6,
                seed=SEED + f_idx,
            )
            r_f = train_and_predict_logits(sp_f, tr_sub, va_sub, {"test": test_rows})
            va_lg = r_f["val_logits"]
            oof_logits[va_idx] = va_lg
            test_logits_folds.append(r_f["eval_logits"]["test"])
            fold_best_epochs.append(int(r_f["best_epoch"]))

            y_va_f = y_tv[va_idx]
            t_f = float(_calibrate_temperature(va_lg, y_va_f, multilabel=multilabel))
            p_f = apply_logits_calibration(va_lg, multilabel, t_f)
            if multilabel:
                tu = tune_multilabel_thresholds(y_va_f, p_f, labels, mode="unbounded")
                tb = tune_multilabel_thresholds(y_va_f, p_f, labels, mode="bounded")
                fold_thresholds_unbounded.append(tu)
                fold_thresholds_bounded.append(tb)
                ev_f = evaluate_predictions(task, y_va_f, p_f, thresholds={l: 0.5 for l in labels})
            else:
                ev_f = evaluate_predictions(task, y_va_f, p_f)

            fold_records.append(
                {
                    "fold": f_idx,
                    "n_train": len(tr_sub),
                    "n_val": len(va_sub),
                    "n_train_threads": len(tr_pids),
                    "n_val_threads": len(va_pids),
                    "leakage_threads": 0,
                    "best_epoch": int(r_f["best_epoch"]),
                    "val_loss": r_f["best_val_loss"],
                    "val_macro_f1_default": ev_f["macro_f1"],
                    "val_ece": ev_f["ece"],
                    "fold_temperature": round(t_f, 3),
                    "train_seconds": r_f["train_seconds"],
                    "peak_vram_mb": r_f["peak_vram_mb"],
                }
            )
            print(
                f"  [Outer Fold {f_idx}] n_val={len(va_sub)} ({len(va_pids)} threads) "
                f"macro_f1@0.5={ev_f['macro_f1']:.4f} ece={ev_f['ece']:.4f} "
                f"best_ep={r_f['best_epoch']} vram={r_f['peak_vram_mb']}MB ({r_f['train_seconds']}s)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # 4. Comprehensive OOF Calibration Analysis on all 367 samples
        # ------------------------------------------------------------------
        t_global = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
        t_per_class = fit_per_class_temperatures(oof_logits, y_tv, multilabel=multilabel)

        oof_probs_raw = apply_logits_calibration(oof_logits, multilabel, 1.0)
        oof_probs_global = apply_logits_calibration(oof_logits, multilabel, t_global)
        oof_probs_percls = apply_logits_calibration(oof_logits, multilabel, t_per_class)

        cal_comparison: Dict[str, Any] = {
            "raw_T1": {
                "temperature": 1.0,
                "nll": round(float(_validation_nll(oof_logits, y_tv, multilabel, 1.0)), 5),
                "eval_default": evaluate_predictions(task, y_tv, oof_probs_raw),
            },
            "global_T": {
                "temperature": round(t_global, 3),
                "nll": round(float(_validation_nll(oof_logits, y_tv, multilabel, t_global)), 5),
                "eval_default": evaluate_predictions(task, y_tv, oof_probs_global),
            },
            "per_class_T": {
                "temperatures": {lbl: t_per_class[i] for i, lbl in enumerate(labels)},
                "eval_default": evaluate_predictions(task, y_tv, oof_probs_percls),
            },
        }

        threshold_comparison: Dict[str, Any] = {}
        threshold_stability: Dict[str, Any] = {}
        if multilabel:
            for mode_name in ("fixed_05", "unbounded", "bounded", "regularized"):
                taus = tune_multilabel_thresholds(y_tv, oof_probs_global, labels, mode=mode_name)
                ev_m = evaluate_predictions(task, y_tv, oof_probs_global, thresholds=taus)
                # Also measure nested out-of-fold threshold generalization (fit threshold on 4 folds, evaluate on 5th!)
                nested_preds = np.zeros_like(y_tv, dtype=int)
                for f_idx, (tr_idx, va_idx) in enumerate(outer_folds):
                    taus_4 = tune_multilabel_thresholds(y_tv[tr_idx], oof_probs_global[tr_idx], labels, mode=mode_name)
                    t_arr = np.array([taus_4[l] for l in labels], dtype=np.float32)
                    nested_preds[va_idx] = (oof_probs_global[va_idx] > t_arr[None, :]).astype(int)
                nested_f1s = []
                nested_per_lbl = {}
                for col, lbl in enumerate(labels):
                    tp = int(((y_tv[:, col] == 1) & (nested_preds[:, col] == 1)).sum())
                    fp = int(((y_tv[:, col] == 0) & (nested_preds[:, col] == 1)).sum())
                    fn = int(((y_tv[:, col] == 1) & (nested_preds[:, col] == 0)).sum())
                    p_v = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r_v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f_v = (2 * p_v * r_v / (p_v + r_v)) if (p_v + r_v) > 0 else 0.0
                    nested_per_lbl[lbl] = {"precision": round(p_v, 4), "recall": round(r_v, 4), "f1": round(f_v, 4)}
                    nested_f1s.append(f_v)
                threshold_comparison[mode_name] = {
                    "thresholds": taus,
                    "oof_insample_macro_f1": ev_m["macro_f1"],
                    "oof_nested_macro_f1": round(float(np.mean(nested_f1s)), 4),
                    "oof_nested_per_class": nested_per_lbl,
                    "oof_insample_eval": ev_m,
                }

            for lbl in labels:
                u_vals = [d[lbl] for d in fold_thresholds_unbounded]
                b_vals = [d[lbl] for d in fold_thresholds_bounded]
                threshold_stability[lbl] = {
                    "unbounded_mean": round(float(np.mean(u_vals)), 3),
                    "unbounded_std": round(float(np.std(u_vals)), 3),
                    "unbounded_range": [min(u_vals), max(u_vals)],
                    "bounded_mean": round(float(np.mean(b_vals)), 3),
                    "bounded_std": round(float(np.std(b_vals)), 3),
                    "bounded_range": [min(b_vals), max(b_vals)],
                }
        elif task == "solution":
            # Compare default 0.50 vs nested OOF threshold on binary solution
            tau_sol = _tune_binary_threshold(y_tv, oof_probs_global, 0.35, 0.70)
            nested_pred = np.zeros(len(y_tv), dtype=int)
            fold_taus_sol = []
            for f_idx, (tr_idx, va_idx) in enumerate(outer_folds):
                t4 = _tune_binary_threshold(y_tv[tr_idx], oof_probs_global[tr_idx], 0.35, 0.70)
                fold_taus_sol.append(t4)
                nested_pred[va_idx] = (oof_probs_global[va_idx, 1] >= t4).astype(int)
            from sklearn.metrics import f1_score as _f1
            threshold_comparison["fixed_05"] = {
                "threshold": 0.5,
                "oof_macro_f1": evaluate_predictions(task, y_tv, oof_probs_global, binary_threshold=0.5)["macro_f1"],
            }
            threshold_comparison["bounded_035_070"] = {
                "threshold": tau_sol,
                "oof_insample_macro_f1": evaluate_predictions(task, y_tv, oof_probs_global, binary_threshold=tau_sol)["macro_f1"],
                "oof_nested_macro_f1": round(float(_f1(y_tv, nested_pred, average="macro", zero_division=0)), 4),
                "fold_taus": fold_taus_sol,
            }

        # ------------------------------------------------------------------
        # 5. Final Single-Model Alternative (1 checkpoint trained on 100% of trainval=367)
        #    Trained for median CV best_epoch; calibrated with OOF temperature/thresholds.
        # ------------------------------------------------------------------
        med_epoch = max(2, int(round(statistics.median(fold_best_epochs))))
        # Use a dummy tiny slice only to satisfy val interface while running exact med_epoch
        # Wait: let's train exact med_epoch cleanly without early-stopping leakage by passing epochs=med_epoch
        # and selecting the final epoch? Better: train_and_predict_logits selects best epoch on val;
        # here we can evaluate final epoch by training directly on all 367 rows for exact med_epoch!
        # Let's reuse a clean 1-call helper by passing trainval as train and val_rows=trainval[:8] only when
        # we want best_epoch? NO — never select on train! Instead let's train exact med_epoch via a tiny
        # helper right here so single-model 367 is 100% honest fixed-epoch training.
        sm_test_logits, sm_train_s, sm_vram = _train_fixed_epochs_all_trainval(
            TrainSpec(
                task=task,
                backbone_id=backbone,
                context_mode=ctx_mode,
                loss_type="default",
                max_length=160,
                epochs=med_epoch,
                seed=SEED + 500,
            ),
            trainval_rows,
            test_rows,
        )
        print(
            f"  [Single-Model Full-367 Alternative] fixed_epochs={med_epoch} "
            f"vram={sm_vram}MB ({sm_train_s}s)",
            flush=True,
        )

        # Save all raw logits for reproducible Phase 8/9 final evaluation (never tuned on test!)
        ens_test_logits = np.mean(np.stack(test_logits_folds, axis=0), axis=0)
        np.savez_compressed(
            logits_dir / f"{task}_phase1_logits.npz",
            oof_logits=oof_logits,
            fold_test_logits=np.stack(test_logits_folds, axis=0),
            ens_test_logits=ens_test_logits,
            single_split_val_logits=ss_val_logits,
            single_split_test_logits=ss_test_logits,
            full367_single_test_logits=sm_test_logits,
            inner_oof_298_logits=inner_oof_logits,
            inner_ens_val69_logits=ens_val69_logits,
        )

        summary["tasks"][task] = {
            "context_mode": ctx_mode,
            "labels": labels,
            "fold_best_epochs": fold_best_epochs,
            "median_best_epoch": med_epoch,
            "single_split_inductive_val69": {
                "temperature": round(ss_t, 3),
                "thresholds": ss_taus,
                "best_epoch": ss_res["best_epoch"],
                "val_eval": ss_val_eval,
                "train_seconds": ss_res["train_seconds"],
            },
            "nested_ensemble_calibration_check": ens_mismatch_report,
            "outer_folds": fold_records,
            "oof_calibration_comparison": cal_comparison,
            "oof_threshold_comparison": threshold_comparison,
            "oof_fold_threshold_stability": threshold_stability,
            "single_model_full367_meta": {
                "fixed_epochs": med_epoch,
                "train_seconds": sm_train_s,
                "peak_vram_mb": sm_vram,
            },
        }
        save_json(out_dir / "phase1_summary.json", summary)

    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "phase1_summary.json", summary)
    print(f"\nPhase 1 complete in {summary['total_elapsed_seconds']}s -> {out_dir / 'phase1_summary.json'}", flush=True)
    return summary


def _train_fixed_epochs_all_trainval(
    spec: TrainSpec,
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
):
    """Train a single model on 100% of trainval (n=367) for an exact CV-selected epoch budget."""
    import math as _math
    import random as _random
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from ml.experiments.v5_optimization.common import (
        PooledTransformerClassifier,
        _PairDataset,
        _make_criterion,
        _model_uses_token_type_ids,
        build_input_pair,
        clear_cuda,
        tokenize_pairs,
    )
    from ml.models.transformer.muril_classifier import _build_llrd_param_groups, resolve_local_muril_path

    clear_cuda()
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    _random.seed(spec.seed)
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
    model = PooledTransformerClassifier(
        encoder=encoder,
        hidden_size=int(hf_cfg.hidden_size),
        num_labels=len(labels),
        multi_sample_dropout=5,
        dropout_rate=spec.dropout_rate,
    ).to(device)

    train_pairs = [build_input_pair(r, spec.context_mode) for r in train_rows]
    y_train = encode_task_targets(train_rows, spec.task)

    def collate_fn(items):
        p_batch, t_batch = zip(*items)
        enc = tokenize_pairs(tokenizer, p_batch, spec.max_length, use_token_type_ids=use_tti)
        t_tensor = torch.tensor(np.asarray(t_batch), dtype=torch.float32 if multilabel else torch.long)
        return enc, t_tensor

    ds = _PairDataset(train_pairs, y_train)
    gen = torch.Generator().manual_seed(spec.seed)
    loader = DataLoader(ds, batch_size=spec.batch_size, shuffle=True, generator=gen, num_workers=0, collate_fn=collate_fn)
    criterion = _make_criterion(spec, y_train, device)
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
        return max(0.05, 0.5 * (1.0 + _math.cos(_math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    t0 = time.perf_counter()
    for _ in range(spec.epochs):
        model.train()
        for enc, tgt in loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = criterion(model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")), tgt)
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

    train_s = round(time.perf_counter() - t0, 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    model.eval()
    test_pairs = [build_input_pair(r, spec.context_mode) for r in test_rows]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(test_pairs), spec.eval_batch_size):
            enc = tokenize_pairs(tokenizer, test_pairs[start : start + spec.eval_batch_size], spec.max_length, use_token_type_ids=use_tti)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                lg = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
            chunks.append(lg.float().cpu().numpy())
    test_logits = np.concatenate(chunks, axis=0)

    del model, encoder, optimizer, scheduler
    clear_cuda()
    return test_logits, train_s, peak_vram


if __name__ == "__main__":
    run_phase1()
