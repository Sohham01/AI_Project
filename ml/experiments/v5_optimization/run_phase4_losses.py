"""Phase 4 — Task-Specific Loss Function Ablation (Section 8).

Runs on each task's Phase 2 OOF-selected context mode via identical 5-fold group-aware CV (n=367):
- Multi-label (issue, emotion): bce | bce_pos (V5 default) | cb_bce | asl
- Single-label (response, sentiment): ce | ce_weighted (V5 default) | logit_adj (tau=0.75) | focal_weighted
- Binary (solution): ce_weighted (V5 default) | ce | focal | focal_weighted
  + hard-negative error analysis breaking down false positives by true response category and clause count.

Recalibrates probabilities and nested OOF thresholds strictly on OOF (n=367).
Selects winning loss per task strictly on OOF CV; saves frozen-test logits untouched for Phase 8/9.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
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
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature

LOSS_GRID: Dict[str, Tuple[str, ...]] = {
    "issue": ("bce_pos", "bce", "cb_bce", "asl"),
    "emotion": ("bce_pos", "bce", "cb_bce", "asl"),
    "response": ("ce_weighted", "ce", "logit_adj", "focal_weighted"),
    "sentiment": ("ce_weighted", "ce", "logit_adj", "focal_weighted"),
    "solution": ("ce_weighted", "ce", "focal", "focal_weighted"),
}


def _count_clauses(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?\n;]+|\b(?:and|but|also|however|if|or|then)\b", text, flags=re.I) if len(p.strip()) >= 6]
    return max(1, len(parts))


def _solution_hard_negative_analysis(rows: List[Dict[str, Any]], y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    fp_idx = [i for i in range(len(rows)) if int(y_true[i]) == 0 and int(pred[i]) == 1]
    fn_idx = [i for i in range(len(rows)) if int(y_true[i]) == 1 and int(pred[i]) == 0]
    tp_idx = [i for i in range(len(rows)) if int(y_true[i]) == 1 and int(pred[i]) == 1]

    fp_by_response = Counter(rows[i]["labels"]["response"] for i in fp_idx)
    fp_clauses = [_count_clauses(rows[i]["text"]) for i in fp_idx]
    tp_clauses = [_count_clauses(rows[i]["text"]) for i in tp_idx]
    multiclause_fps = sum(1 for c in fp_clauses if c >= 3)

    top_hard_fps = sorted(fp_idx, key=lambda i: float(probs[i, 1]), reverse=True)[:10]
    examples = [
        {
            "document_id": rows[i]["document_id"],
            "true_response": rows[i]["labels"]["response"],
            "p_solution": round(float(probs[i, 1]), 4),
            "clause_count": _count_clauses(rows[i]["text"]),
            "word_count": len(rows[i]["text"].split()),
            "snippet": rows[i]["text"][:160],
        }
        for i in top_hard_fps
    ]
    return {
        "n_fp": len(fp_idx),
        "n_fn": len(fn_idx),
        "n_tp": len(tp_idx),
        "fp_by_true_response": dict(fp_by_response),
        "fp_multiclause_ge3_count": multiclause_fps,
        "fp_mean_clauses": round(float(np.mean(fp_clauses)) if fp_clauses else 0.0, 2),
        "tp_mean_clauses": round(float(np.mean(tp_clauses)) if tp_clauses else 0.0, 2),
        "top_hard_negative_fps": examples,
    }


def run_phase4() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase4_losses")
    logits_dir = ensure_dir(out_dir / "logits")
    backbone = str(CKPT_ROOT / "muril_dapt_inductive_mlm_4ep")
    assert Path(backbone).exists(), f"Inductive DAPT checkpoint missing at {backbone}"

    p2_path = EXP_ROOT / "phase2_context" / "phase2_summary.json"
    assert p2_path.exists(), f"Run Phase 2 first ({p2_path} missing)"
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))

    data = load_v1_records_and_frozen_splits()
    trainval_rows = data["trainval"]
    test_rows = data["test"]

    summary_path = out_dir / "phase4_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "phase": "phase4_loss_function_ablation",
            "backbone": backbone,
            "max_length": 256,
            "context_from_phase2": {t: p2["selected_context_by_task"][t]["selected_mode"] for t in TASK_NAMES},
            "tasks": {},
            "selected_loss_by_task": {},
        }

    t_all = time.perf_counter()

    for task in TASK_NAMES:
        if task in summary.get("tasks", {}) and task in summary.get("selected_loss_by_task", {}):
            print(f"\n[Phase 4] Skipping already-completed task: {task.upper()}", flush=True)
            continue
        ctx_mode = p2["selected_context_by_task"][task]["selected_mode"]
        print(f"\n==================== PHASE 4 LOSSES: {task.upper()} (ctx={ctx_mode}) ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)

        task_losses: Dict[str, Any] = {}
        default_loss = LOSS_GRID[task][0]

        for loss_name in LOSS_GRID[task]:
            p2_reuse = EXP_ROOT / "phase2_context" / "logits" / f"{task}_{ctx_mode}_logits.npz"
            if loss_name == default_loss and p2_reuse.exists():
                cached = np.load(p2_reuse)
                oof_logits = cached["oof_logits"].astype(np.float32)
                test_logits_folds = [cached["fold_test_logits"][k] for k in range(len(folds))]
                train_secs = [float(p2["tasks"][task][ctx_mode]["total_5fold_train_seconds"]) / len(folds)] * len(folds)
                print(f"  [{task} | {loss_name:14s}] Reused exact Phase 2 logits ({p2_reuse.name})", flush=True)
            else:
                oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
                test_logits_folds = []
                train_secs = []

                for f_idx, (tr_idx, va_idx) in enumerate(folds):
                    spec = TrainSpec(
                        task=task,
                        backbone_id=backbone,
                        context_mode=ctx_mode,
                        loss_type=loss_name,
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
                    train_secs.append(float(r_f["train_seconds"]))

            t_global = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
            oof_probs_raw = apply_logits_calibration(oof_logits, multilabel, 1.0)
            oof_probs = apply_logits_calibration(oof_logits, multilabel, t_global)

            if multilabel:
                # Nested OOF bounded thresholding for fair comparison across losses (critical for ASL!)
                nested_pred = np.zeros_like(y_tv, dtype=int)
                for tr_idx, va_idx in folds:
                    taus_4 = tune_multilabel_thresholds(y_tv[tr_idx], oof_probs[tr_idx], labels, mode="bounded")
                    t_arr = np.array([taus_4[l] for l in labels], dtype=np.float32)
                    nested_pred[va_idx] = (oof_probs[va_idx] > t_arr[None, :]).astype(int)
                taus_bounded = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="bounded")
                ev_raw = evaluate_predictions(task, y_tv, oof_probs_raw)
                ev_cal = evaluate_predictions(task, y_tv, oof_probs)
                nested_f1s = []
                nested_per_class = {}
                for c, lbl in enumerate(labels):
                    tp = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 1)).sum())
                    fp = int(((y_tv[:, c] == 0) & (nested_pred[:, c] == 1)).sum())
                    fn = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 0)).sum())
                    p_v = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r_v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f_v = (2 * p_v * r_v / (p_v + r_v)) if (p_v + r_v) > 0 else 0.0
                    nested_per_class[lbl] = {
                        "precision": round(p_v, 4),
                        "recall": round(r_v, 4),
                        "f1": round(f_v, 4),
                        "support": int(y_tv[:, c].sum()),
                        "predicted_positive": int(nested_pred[:, c].sum()),
                    }
                    nested_f1s.append(f_v)
                primary_oof_f1 = round(float(np.mean(nested_f1s)), 4)
            else:
                ev_raw = evaluate_predictions(task, y_tv, oof_probs_raw)
                ev_cal = evaluate_predictions(task, y_tv, oof_probs)
                primary_oof_f1 = ev_cal["macro_f1"]
                taus_bounded = None
                nested_per_class = ev_cal["per_class"]

            extra_diag: Dict[str, Any] = {}
            if task == "response":
                cm = np.asarray(ev_cal["confusion_matrix"])
                extra_diag = {
                    "counterargument_f1": ev_cal["per_class"]["counterargument"]["f1"],
                    "solution_f1": ev_cal["per_class"]["solution"]["f1"],
                    "advice_f1": ev_cal["per_class"]["advice"]["f1"],
                    "counterargument_to_opinion_errors": int(cm[labels.index("counterargument"), labels.index("opinion")]),
                    "solution_to_advice_errors": int(cm[labels.index("solution"), labels.index("advice")]),
                }
            elif task == "sentiment":
                cm = np.asarray(ev_cal["confusion_matrix"])
                extra_diag = {
                    "neg_neu_mutual_errors": int(
                        cm[labels.index("negative"), labels.index("neutral")]
                        + cm[labels.index("neutral"), labels.index("negative")]
                    ),
                }
            elif task == "solution":
                extra_diag = _solution_hard_negative_analysis(trainval_rows, y_tv, oof_probs)

            ens_test_logits = np.mean(np.stack(test_logits_folds, axis=0), axis=0)
            np.savez_compressed(
                logits_dir / f"{task}_{loss_name}_logits.npz",
                oof_logits=oof_logits,
                fold_test_logits=np.stack(test_logits_folds, axis=0),
                ens_test_logits=ens_test_logits,
            )

            task_losses[loss_name] = {
                "loss_type": loss_name,
                "primary_oof_macro_f1": primary_oof_f1,
                "oof_macro_f1_fixed05": ev_cal["macro_f1"],
                "raw_ece": ev_raw["ece"],
                "calibrated_ece": ev_cal["ece"],
                "calibrated_brier": ev_cal["brier"],
                "temperature": round(t_global, 3),
                "bounded_thresholds": taus_bounded,
                "per_class": nested_per_class,
                "extra_diagnostics": extra_diag,
                "total_5fold_train_seconds": round(float(np.sum(train_secs)), 2),
            }
            print(
                f"  [{task} | {loss_name:14s}] OOF_F1={primary_oof_f1:.4f} (fixed05={ev_cal['macro_f1']:.4f}) "
                f"raw_ECE={ev_raw['ece']:.4f} cal_ECE={ev_cal['ece']:.4f} T={t_global:.2f}",
                flush=True,
            )

        # Select winning loss strictly on OOF CV (require +0.004 margin or substantially better minority/ECE)
        base_f1 = task_losses[default_loss]["primary_oof_macro_f1"]
        best_loss = max(LOSS_GRID[task], key=lambda l: task_losses[l]["primary_oof_macro_f1"])
        if best_loss != default_loss and task_losses[best_loss]["primary_oof_macro_f1"] < base_f1 + 0.004:
            selected_loss = default_loss
            note = f"kept baseline {default_loss} ({base_f1:.4f}); {best_loss} delta < +0.004"
        else:
            selected_loss = best_loss
            note = f"selected {best_loss} ({task_losses[best_loss]['primary_oof_macro_f1']:.4f} vs {default_loss} {base_f1:.4f})"

        summary["tasks"][task] = task_losses
        summary["selected_loss_by_task"][task] = {
            "selected_loss": selected_loss,
            "best_raw_loss": best_loss,
            "note": note,
        }
        print(f"  >>> {task.upper()} LOSS SELECTION: {selected_loss} ({note})", flush=True)
        save_json(out_dir / "phase4_summary.json", summary)

    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "phase4_summary.json", summary)
    print(f"\nPhase 4 complete in {summary['total_elapsed_seconds']}s -> {out_dir / 'phase4_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    run_phase4()
