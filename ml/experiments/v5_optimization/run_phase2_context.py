"""Phase 2 — Native Segment-Pair Parent Context Ablation (Section 6).

Compares 4 strictly legitimate context variants (ZERO child/descendant/sibling leakage)
using native MuRIL tokenizer segment pairs `[CLS] reply [SEP] context [SEP]` with
`max_length=256`, `truncation="only_second"`, and native `token_type_ids`:
  1. comment_only
  2. title_pair
  3. parent_pair (parent comment if available, else post title)
  4. title_parent_pair (Post: {title} | Parent: {parent})

Evaluates all 5 tasks independently via identical 5-fold group-aware CV (n=367).
Tracks priority targets: counterargument, solution vs advice, neutral vs negative,
short comments (<=8 words), comments with a real parent comment, truncation rates,
calibration (ECE/Brier), latency, and peak VRAM.
Selects winning context mode per task strictly from OOF CV (never test set!).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

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
    build_input_pair,
    encode_task_targets,
    ensure_dir,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature, resolve_local_muril_path

CONTEXT_VARIANTS = ("comment_only", "title_pair", "parent_pair", "title_parent_pair")


def measure_truncation_stats(tokenizer: Any, rows: List[Dict[str, Any]], context_mode: str, max_length: int) -> Dict[str, Any]:
    total_trunc = 0
    reply_trunc = 0
    ctx_trunc = 0
    total_lens = []
    for r in rows:
        a, b = build_input_pair(r, context_mode)
        len_a = len(tokenizer.encode(a, add_special_tokens=False))
        len_b = len(tokenizer.encode(b, add_special_tokens=False)) if b else 0
        overhead = 3 if b else 2
        tot = len_a + len_b + overhead
        total_lens.append(tot)
        if tot > max_length:
            total_trunc += 1
            if len_a + overhead > max_length:
                reply_trunc += 1
            if len_b > 0:
                ctx_trunc += 1
    n = max(len(rows), 1)
    return {
        "max_length": max_length,
        "median_tokens": int(np.median(total_lens)),
        "p95_tokens": int(np.percentile(total_lens, 95)),
        "max_tokens": int(np.max(total_lens)),
        "total_truncated_pct": round(100.0 * total_trunc / n, 2),
        "reply_truncated_pct": round(100.0 * reply_trunc / n, 2),
        "context_truncated_pct": round(100.0 * ctx_trunc / n, 2),
    }


def _slice_macro_f1(task: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    if task in MULTILABEL_TASKS:
        f1s = []
        for c in range(yt.shape[1]):
            if yt[:, c].sum() > 0 or yp[:, c].sum() > 0:
                tp = int(((yt[:, c] == 1) & (yp[:, c] == 1)).sum())
                fp = int(((yt[:, c] == 0) & (yp[:, c] == 1)).sum())
                fn = int(((yt[:, c] == 1) & (yp[:, c] == 0)).sum())
                d = 2 * tp + fp + fn
                f1s.append((2.0 * tp / d) if d > 0 else 0.0)
        return round(float(np.mean(f1s)) if f1s else 0.0, 4)
    return round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)


def run_phase2() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase2_context")
    logits_dir = ensure_dir(out_dir / "logits")
    backbone = str(CKPT_ROOT / "muril_dapt_inductive_mlm_4ep")
    assert Path(backbone).exists(), f"Inductive DAPT checkpoint missing at {backbone}"

    model_path, local_only = resolve_local_muril_path(backbone)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)

    data = load_v1_records_and_frozen_splits()
    trainval_rows = data["trainval"]  # 367
    test_rows = data["test"]          # 92 (saved untouched for Phase 8/9 only)

    short_mask = np.array([len(r["text"].split()) <= 8 for r in trainval_rows], dtype=bool)
    has_parent_mask = np.array([bool(r["parent_comment_text"]) for r in trainval_rows], dtype=bool)

    trunc_table = {
        mode: measure_truncation_stats(tokenizer, trainval_rows, mode, max_length=256)
        for mode in CONTEXT_VARIANTS
    }

    summary_path = out_dir / "phase2_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "phase": "phase2_context_ablation",
            "backbone": backbone,
            "max_length": 256,
            "truncation_policy": "only_second (preserve 100% of reply whenever possible)",
            "n_trainval": len(trainval_rows),
            "n_short_comments_le8": int(short_mask.sum()),
            "n_with_parent_comment": int(has_parent_mask.sum()),
            "truncation_stats": trunc_table,
            "tasks": {},
            "selected_context_by_task": {},
        }

    t_all = time.perf_counter()

    for task in TASK_NAMES:
        if task in summary.get("tasks", {}) and task in summary.get("selected_context_by_task", {}):
            print(f"\n[Phase 2] Skipping already-completed task: {task.upper()}", flush=True)
            continue
        print(f"\n==================== PHASE 2 CONTEXT: {task.upper()} ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)

        task_variants: Dict[str, Any] = {}
        for mode in CONTEXT_VARIANTS:
            oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
            test_logits_folds: List[np.ndarray] = []
            latencies: List[float] = []
            vrams: List[float] = []
            train_secs: List[float] = []

            for f_idx, (tr_idx, va_idx) in enumerate(folds):
                spec = TrainSpec(
                    task=task,
                    backbone_id=backbone,
                    context_mode=mode,
                    loss_type="default",
                    max_length=256,
                    epochs=6,
                    seed=SEED + f_idx,
                )
                r_f = train_and_predict_logits(
                    spec,
                    [trainval_rows[i] for i in tr_idx],
                    [trainval_rows[i] for i in va_idx],
                    {"test": test_rows},
                )
                oof_logits[va_idx] = r_f["val_logits"]
                test_logits_folds.append(r_f["eval_logits"]["test"])
                latencies.append(float(r_f["eval_latency_ms"]["test"]))
                vrams.append(float(r_f["peak_vram_mb"]))
                train_secs.append(float(r_f["train_seconds"]))

            t_global = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
            oof_probs = apply_logits_calibration(oof_logits, multilabel, t_global)

            if multilabel:
                # Honest nested OOF thresholding (bounded [0.30, 0.70]) AND fixed 0.5
                nested_pred = np.zeros_like(y_tv, dtype=int)
                for tr_idx, va_idx in folds:
                    taus_4 = tune_multilabel_thresholds(y_tv[tr_idx], oof_probs[tr_idx], labels, mode="bounded")
                    t_arr = np.array([taus_4[l] for l in labels], dtype=np.float32)
                    nested_pred[va_idx] = (oof_probs[va_idx] > t_arr[None, :]).astype(int)
                full_taus_bounded = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="bounded")
                ev_fixed = evaluate_predictions(task, y_tv, oof_probs, thresholds={l: 0.5 for l in labels})
                ev_bounded_insample = evaluate_predictions(task, y_tv, oof_probs, thresholds=full_taus_bounded)
                nested_f1s = []
                nested_per_class = {}
                for c, lbl in enumerate(labels):
                    tp = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 1)).sum())
                    fp = int(((y_tv[:, c] == 0) & (nested_pred[:, c] == 1)).sum())
                    fn = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 0)).sum())
                    p_v = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r_v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f_v = (2 * p_v * r_v / (p_v + r_v)) if (p_v + r_v) > 0 else 0.0
                    nested_per_class[lbl] = {"precision": round(p_v, 4), "recall": round(r_v, 4), "f1": round(f_v, 4)}
                    nested_f1s.append(f_v)
                primary_oof_f1 = round(float(np.mean(nested_f1s)), 4)
                y_pred_for_slices = nested_pred
                eval_obj = ev_fixed
            else:
                eval_obj = evaluate_predictions(task, y_tv, oof_probs)
                primary_oof_f1 = eval_obj["macro_f1"]
                y_pred_for_slices = np.argmax(oof_probs, axis=1)
                full_taus_bounded = None
                nested_per_class = eval_obj["per_class"]

            # Priority diagnostics requested in Section 6
            priority_diag: Dict[str, Any] = {
                "short_comments_le8_macro_f1": _slice_macro_f1(task, y_tv, y_pred_for_slices, short_mask),
                "has_parent_macro_f1": _slice_macro_f1(task, y_tv, y_pred_for_slices, has_parent_mask),
                "top_level_macro_f1": _slice_macro_f1(task, y_tv, y_pred_for_slices, ~has_parent_mask),
            }
            if task == "response":
                cm = np.asarray(eval_obj["confusion_matrix"])
                i_ctr = labels.index("counterargument")
                i_op = labels.index("opinion")
                i_sol = labels.index("solution")
                i_adv = labels.index("advice")
                priority_diag["counterargument_f1"] = eval_obj["per_class"]["counterargument"]["f1"]
                priority_diag["solution_f1"] = eval_obj["per_class"]["solution"]["f1"]
                priority_diag["advice_f1"] = eval_obj["per_class"]["advice"]["f1"]
                priority_diag["counterargument_to_opinion_errors"] = int(cm[i_ctr, i_op])
                priority_diag["solution_to_advice_errors"] = int(cm[i_sol, i_adv])
            elif task == "sentiment":
                cm = np.asarray(eval_obj["confusion_matrix"])
                i_neg = labels.index("negative")
                i_neu = labels.index("neutral")
                priority_diag["neg_neu_mutual_errors"] = int(cm[i_neg, i_neu] + cm[i_neu, i_neg])
                priority_diag["negative_f1"] = eval_obj["per_class"]["negative"]["f1"]
                priority_diag["neutral_f1"] = eval_obj["per_class"]["neutral"]["f1"]
            elif task == "solution":
                cm = np.asarray(eval_obj["confusion_matrix"])
                priority_diag["solution_pos_precision"] = eval_obj["per_class"]["solution"]["precision"]
                priority_diag["solution_pos_recall"] = eval_obj["per_class"]["solution"]["recall"]
                priority_diag["solution_fp_count"] = int(cm[0, 1])

            ens_test_logits = np.mean(np.stack(test_logits_folds, axis=0), axis=0)
            np.savez_compressed(
                logits_dir / f"{task}_{mode}_logits.npz",
                oof_logits=oof_logits,
                fold_test_logits=np.stack(test_logits_folds, axis=0),
                ens_test_logits=ens_test_logits,
            )

            task_variants[mode] = {
                "context_mode": mode,
                "primary_oof_macro_f1": primary_oof_f1,
                "oof_macro_f1_fixed05": eval_obj["macro_f1"],
                "oof_ece": eval_obj["ece"],
                "oof_brier": eval_obj["brier"],
                "temperature": round(t_global, 3),
                "bounded_thresholds": full_taus_bounded,
                "per_class": nested_per_class,
                "priority_diagnostics": priority_diag,
                "mean_latency_ms_per_sample": round(float(np.mean(latencies)), 3),
                "peak_vram_mb": round(float(np.max(vrams)), 1),
                "total_5fold_train_seconds": round(float(np.sum(train_secs)), 2),
            }
            print(
                f"  [{task} | {mode:17s}] OOF_F1={primary_oof_f1:.4f} (fixed05={eval_obj['macro_f1']:.4f}) "
                f"ECE={eval_obj['ece']:.4f} short={priority_diag['short_comments_le8_macro_f1']:.4f} "
                f"lat={task_variants[mode]['mean_latency_ms_per_sample']:.2f}ms vram={task_variants[mode]['peak_vram_mb']}MB",
                flush=True,
            )

        # Select winning context strictly on OOF CV (simplicity tie-break: +0.005 margin required over comment_only)
        base_f1 = task_variants["comment_only"]["primary_oof_macro_f1"]
        best_mode = max(CONTEXT_VARIANTS, key=lambda m: task_variants[m]["primary_oof_macro_f1"])
        if best_mode != "comment_only" and task_variants[best_mode]["primary_oof_macro_f1"] < base_f1 + 0.005:
            selected_mode = "comment_only"
            selection_note = f"kept simpler comment_only ({base_f1:.4f}) since {best_mode} gain < +0.005"
        else:
            selected_mode = best_mode
            selection_note = f"selected {best_mode} (+{task_variants[best_mode]['primary_oof_macro_f1'] - base_f1:.4f} OOF F1 vs comment_only)"

        summary["tasks"][task] = task_variants
        summary["selected_context_by_task"][task] = {
            "selected_mode": selected_mode,
            "best_raw_mode": best_mode,
            "note": selection_note,
        }
        print(f"  >>> {task.upper()} OOF SELECTION: {selected_mode} ({selection_note})", flush=True)
        save_json(out_dir / "phase2_summary.json", summary)

    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "phase2_summary.json", summary)
    print(f"\nPhase 2 complete in {summary['total_elapsed_seconds']}s -> {out_dir / 'phase2_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    run_phase2()
