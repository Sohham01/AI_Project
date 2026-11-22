"""Phase 5 — DAPT Whole-Word Masking, Extended Schedule, Alternative Backbones & LoRA (Section 9).

Executes cleanly after Phases 1, 2, 4, 3:
  A. Trains strictly inductive WWM DAPT (4 epochs & up to 8 epochs with early stopping)
     excluding all 37 frozen test threads and grouping DAPT validation by post_id.
  B. Compares downstream 5-fold OOF CV across all 5 tasks for:
     1. muril_base (google/muril-base-cased, no DAPT)
     2. muril_dapt_inductive_mlm_4ep (standard subword MLM, 4ep)
     3. muril_dapt_inductive_wwm_4ep (WordPiece WWM, 4ep)
     4. muril_dapt_inductive_wwm_8ep (WordPiece WWM, up to 8ep early-stopped)
  C. Compares alternative backbones on identical splits/context/loss/budget:
     5. ai4bharat/IndicBERTv2-MLM-only
     6. xlm-roberta-base
  D. Compares Pure-PyTorch LoRA (r=8) vs full LLRD fine-tuning on best inductive MuRIL-DAPT
     (measuring Macro F1, VRAM, training/inference speed, trainable params, and 3-seed stability).
"""

from __future__ import annotations

import json
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
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.experiments.v5_optimization.run_dapt_inductive import (
    build_strictly_inductive_dapt_corpus,
    train_inductive_dapt_variant,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature


def _eval_5fold_backbone(
    tag: str,
    backbone_id: str,
    trainval_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    ctx_by_task: Dict[str, str],
    loss_by_task: Dict[str, str],
    logits_dir: Path,
    *,
    use_lora: bool = False,
    lora_r: int = 8,
    seed_offset: int = 0,
    tasks_subset: tuple[str, ...] = TASK_NAMES,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tag": tag, "backbone_id": backbone_id, "use_lora": use_lora, "tasks": {}}
    f1s = []
    for task in tasks_subset:
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)

        oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
        test_logits_folds: List[np.ndarray] = []
        vrams, secs, lats = [], [], []
        tot_p, tr_p = 0, 0

        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            spec = TrainSpec(
                task=task,
                backbone_id=backbone_id,
                context_mode=ctx_by_task[task],
                loss_type=loss_by_task[task],
                max_length=256,
                epochs=8 if use_lora else 6,
                batch_size=8,
                learning_rate=4e-5,
                use_lora=use_lora,
                lora_r=lora_r,
                seed=SEED + seed_offset + f_idx,
            )
            r_f = train_and_predict_logits(
                spec,
                [trainval_rows[i] for i in tr_idx],
                [trainval_rows[i] for i in va_idx],
                {"test": test_rows},
            )
            oof_logits[va_idx] = r_f["val_logits"]
            test_logits_folds.append(r_f["eval_logits"]["test"])
            vrams.append(r_f["peak_vram_mb"])
            secs.append(r_f["train_seconds"])
            lats.append(r_f["eval_latency_ms"]["test"])
            tot_p = r_f["total_params"]
            tr_p = r_f["trainable_params"]

        t_g = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
        probs = apply_logits_calibration(oof_logits, multilabel, t_g)
        if multilabel:
            nested_pred = np.zeros_like(y_tv, dtype=int)
            for tr_idx, va_idx in folds:
                taus_4 = tune_multilabel_thresholds(y_tv[tr_idx], probs[tr_idx], labels, mode="bounded")
                t_arr = np.array([taus_4[l] for l in labels], dtype=np.float32)
                nested_pred[va_idx] = (probs[va_idx] > t_arr[None, :]).astype(int)
            f1_cols = []
            per_cls = {}
            for c, lbl in enumerate(labels):
                tp = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 1)).sum())
                fp = int(((y_tv[:, c] == 0) & (nested_pred[:, c] == 1)).sum())
                fn = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 0)).sum())
                pv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rv = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fv = (2 * pv * rv / (pv + rv)) if (pv + rv) > 0 else 0.0
                per_cls[lbl] = {"precision": round(pv, 4), "recall": round(rv, 4), "f1": round(fv, 4)}
                f1_cols.append(fv)
            ev05 = evaluate_predictions(task, y_tv, probs)
            primary_f1 = round(float(np.mean(f1_cols)), 4)
            taus = tune_multilabel_thresholds(y_tv, probs, labels, mode="bounded")
        else:
            ev05 = evaluate_predictions(task, y_tv, probs)
            primary_f1 = ev05["macro_f1"]
            per_cls = ev05["per_class"]
            taus = None

        f1s.append(primary_f1)
        if seed_offset == 0:
            np.savez_compressed(
                logits_dir / f"{tag}_{task}_logits.npz",
                oof_logits=oof_logits,
                fold_test_logits=np.stack(test_logits_folds, axis=0),
                ens_test_logits=np.mean(np.stack(test_logits_folds, axis=0), axis=0),
            )
        out["tasks"][task] = {
            "primary_oof_macro_f1": primary_f1,
            "oof_macro_f1_fixed05": ev05["macro_f1"],
            "oof_ece": ev05["ece"],
            "oof_brier": ev05["brier"],
            "temperature": round(t_g, 3),
            "bounded_thresholds": taus,
            "per_class": per_cls,
            "total_params": tot_p,
            "trainable_params": tr_p,
            "trainable_pct": round(100.0 * tr_p / max(tot_p, 1), 3),
            "peak_vram_mb": round(float(np.max(vrams)), 1),
            "total_5fold_train_seconds": round(float(np.sum(secs)), 2),
            "mean_latency_ms": round(float(np.mean(lats)), 3),
        }
        print(
            f"  [{tag:28s} | {task:9s}] OOF_F1={primary_f1:.4f} ECE={ev05['ece']:.4f} "
            f"vram={out['tasks'][task]['peak_vram_mb']}MB time={out['tasks'][task]['total_5fold_train_seconds']}s",
            flush=True,
        )
    out["mean_oof_macro_f1"] = round(float(np.mean(f1s)), 4)
    return out


def run_phase5() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase5_dapt_backbones")
    logits_dir = ensure_dir(out_dir / "logits")

    p2 = json.loads((EXP_ROOT / "phase2_context" / "phase2_summary.json").read_text(encoding="utf-8"))
    p4 = json.loads((EXP_ROOT / "phase4_losses" / "phase4_summary.json").read_text(encoding="utf-8"))
    ctx_by_task = {t: p2["selected_context_by_task"][t]["selected_mode"] for t in TASK_NAMES}
    loss_by_task = {t: p4["selected_loss_by_task"][t]["selected_loss"] for t in TASK_NAMES}

    # Step 1: Ensure strictly inductive WWM 4ep and WWM 8ep checkpoints exist
    dapt_data = build_strictly_inductive_dapt_corpus()
    dapt_summaries: Dict[str, Any] = {}
    for vname, kw in [
        ("muril_dapt_inductive_mlm_4ep", dict(masking_mode="standard", epochs=4, early_stop_patience=0)),
        ("muril_dapt_inductive_wwm_4ep", dict(masking_mode="wwm", epochs=4, early_stop_patience=0)),
        ("muril_dapt_inductive_wwm_8ep", dict(masking_mode="wwm", epochs=8, early_stop_patience=2)),
    ]:
        ckpt_dir = CKPT_ROOT / vname
        meta_path = EXP_ROOT / "dapt" / f"{vname}_summary.json"
        if (ckpt_dir / "config.json").exists() and meta_path.exists():
            print(f"[Phase 5] Reusing existing strictly inductive DAPT checkpoint: {vname}", flush=True)
            dapt_summaries[vname] = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            print(f"[Phase 5] Training strictly inductive DAPT checkpoint: {vname}", flush=True)
            dapt_summaries[vname] = train_inductive_dapt_variant(
                vname, dapt_data["train_texts"], dapt_data["val_texts"], **kw
            )

    data = load_v1_records_and_frozen_splits()
    trainval_rows = data["trainval"]
    test_rows = data["test"]

    summary_path = out_dir / "phase5_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dapt_training_summaries"] = dapt_summaries
    else:
        summary = {
            "phase": "phase5_dapt_backbones_lora",
            "context_by_task": ctx_by_task,
            "loss_by_task": loss_by_task,
            "dapt_training_summaries": dapt_summaries,
            "backbones": {},
            "lora_3seed_variability": {},
        }

    t_all = time.perf_counter()

    candidates = [
        ("muril_base", "google/muril-base-cased", False),
        ("muril_dapt_inductive_mlm_4ep", str(CKPT_ROOT / "muril_dapt_inductive_mlm_4ep"), False),
        ("muril_dapt_inductive_wwm_4ep", str(CKPT_ROOT / "muril_dapt_inductive_wwm_4ep"), False),
        ("muril_dapt_inductive_wwm_8ep", str(CKPT_ROOT / "muril_dapt_inductive_wwm_8ep"), False),
        ("indicbert_v2", "ai4bharat/IndicBERTv2-MLM-only", False),
        ("xlm_roberta_base", "xlm-roberta-base", False),
    ]

    for tag, model_id, use_lora in candidates:
        if tag in summary.get("backbones", {}) and "mean_oof_macro_f1" in summary["backbones"][tag]:
            print(f"\n[Phase 5] Skipping already-completed backbone: {tag.upper()}", flush=True)
            continue
        print(f"\n==================== PHASE 5 BACKBONE: {tag.upper()} ====================", flush=True)
        # Reuse exact Phase 4 winning-loss logits for muril_dapt_inductive_mlm_4ep (identical folds, ctx, loss, seed)
        if tag == "muril_dapt_inductive_mlm_4ep":
            reused_ok = all(
                (EXP_ROOT / "phase4_losses" / "logits" / f"{t}_{loss_by_task[t]}_logits.npz").exists()
                for t in TASK_NAMES
            )
            if reused_ok:
                reused_out: Dict[str, Any] = {"tag": tag, "backbone_id": model_id, "use_lora": False, "tasks": {}}
                f1s_r = []
                for t in TASK_NAMES:
                    src_npz = EXP_ROOT / "phase4_losses" / "logits" / f"{t}_{loss_by_task[t]}_logits.npz"
                    dst_npz = logits_dir / f"{tag}_{t}_logits.npz"
                    dst_npz.write_bytes(src_npz.read_bytes())
                    m4 = p4["tasks"][t][loss_by_task[t]]
                    reused_out["tasks"][t] = {
                        "primary_oof_macro_f1": m4["primary_oof_macro_f1"],
                        "oof_macro_f1_fixed05": m4["oof_macro_f1_fixed05"],
                        "oof_ece": m4["calibrated_ece"],
                        "oof_brier": m4["calibrated_brier"],
                        "temperature": m4["temperature"],
                        "bounded_thresholds": m4["bounded_thresholds"],
                        "per_class": m4["per_class"],
                        "total_params": 237563912,
                        "trainable_params": 237563912,
                        "trainable_pct": 100.0,
                        "peak_vram_mb": p2["tasks"][t][ctx_by_task[t]]["peak_vram_mb"],
                        "total_5fold_train_seconds": m4["total_5fold_train_seconds"],
                        "mean_latency_ms": p2["tasks"][t][ctx_by_task[t]]["mean_latency_ms_per_sample"],
                    }
                    f1s_r.append(m4["primary_oof_macro_f1"])
                reused_out["mean_oof_macro_f1"] = round(float(np.mean(f1s_r)), 4)
                summary["backbones"][tag] = reused_out
                print(f"  >>> [{tag}] Reused exact Phase 4 logits: 5-TASK MEAN OOF F1 = {reused_out['mean_oof_macro_f1']:.4f}", flush=True)
                save_json(out_dir / "phase5_summary.json", summary)
                continue
        try:
            res = _eval_5fold_backbone(
                tag,
                model_id,
                trainval_rows,
                test_rows,
                ctx_by_task,
                loss_by_task,
                logits_dir,
                use_lora=use_lora,
            )
            summary["backbones"][tag] = res
            print(f"  >>> [{tag}] 5-TASK MEAN OOF F1 = {res['mean_oof_macro_f1']:.4f}", flush=True)
        except Exception as exc:
            print(f"  [WARN] Backbone {tag} failed: {type(exc).__name__}: {exc}", flush=True)
            summary["backbones"][tag] = {"error": f"{type(exc).__name__}: {exc}"}
        save_json(out_dir / "phase5_summary.json", summary)

    # Pick best inductive MuRIL-DAPT checkpoint strictly on 5-task mean OOF F1 for LoRA comparison
    dapt_tags = ["muril_dapt_inductive_mlm_4ep", "muril_dapt_inductive_wwm_4ep", "muril_dapt_inductive_wwm_8ep"]
    best_dapt_tag = max(
        [t for t in dapt_tags if "mean_oof_macro_f1" in summary["backbones"].get(t, {})],
        key=lambda t: summary["backbones"][t]["mean_oof_macro_f1"],
    )
    best_dapt_path = str(CKPT_ROOT / best_dapt_tag)
    summary["best_inductive_dapt_tag"] = best_dapt_tag

    print(f"\n==================== PHASE 5D: LORA (r=8) ON {best_dapt_tag} ====================", flush=True)
    summary["backbones"]["muril_dapt_lora_r8"] = _eval_5fold_backbone(
        "muril_dapt_lora_r8",
        best_dapt_path,
        trainval_rows,
        test_rows,
        ctx_by_task,
        loss_by_task,
        logits_dir,
        use_lora=True,
        lora_r=8,
    )

    # 3-seed run-to-run variability comparison (Full LLRD vs LoRA r=8) on sentiment & response
    print("\n==================== PHASE 5D: 3-SEED STABILITY (FULL vs LORA) ====================", flush=True)
    for mode_label, is_lora in [("full_llrd", False), ("lora_r8", True)]:
        seed_runs: Dict[str, List[float]] = {"sentiment": [], "response": []}
        for s_off in (0, 1000, 2000):
            if s_off == 0:
                ref = summary["backbones"]["muril_dapt_lora_r8" if is_lora else best_dapt_tag]
                for tk in ("sentiment", "response"):
                    seed_runs[tk].append(ref["tasks"][tk]["primary_oof_macro_f1"])
            else:
                r_s = _eval_5fold_backbone(
                    f"{mode_label}_s{s_off}",
                    best_dapt_path,
                    trainval_rows,
                    test_rows,
                    ctx_by_task,
                    loss_by_task,
                    logits_dir,
                    use_lora=is_lora,
                    lora_r=8,
                    seed_offset=s_off,
                    tasks_subset=("sentiment", "response"),
                )
                for tk in ("sentiment", "response"):
                    seed_runs[tk].append(r_s["tasks"][tk]["primary_oof_macro_f1"])
        summary["lora_3seed_variability"][mode_label] = {
            tk: {
                "seed_scores": vals,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
            for tk, vals in seed_runs.items()
        }

    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "phase5_summary.json", summary)
    print(f"\nPhase 5 complete in {summary['total_elapsed_seconds']}s -> {out_dir / 'phase5_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    run_phase5()
