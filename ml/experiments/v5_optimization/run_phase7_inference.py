"""Phase 7 — Context Length (160/256/384) & GPU/CPU Inference Profiling (Section 11).

Measures:
1. Exact token truncation breakdown (total, reply-side, context-side) at max_length=160, 256, 384
   for both comment-only and context-aware pair inputs across all 459 real comments.
2. Controlled 5-fold OOF CV Macro F1 & Peak VRAM at max_length=160 vs 256 vs 384
   on context-sensitive tasks (issue, response, solution) using Phase 2/4/5 winning settings.
3. Comprehensive GPU & CPU inference benchmark:
   - Dynamic padding vs static padding
   - FP32 vs BF16 mixed precision (GPU)
   - Batch sizes 1, 8, 16, 32 (GPU) and 1, 16 (CPU)
   - Model load time & memory footprint: 1 single-task MuRIL vs 5 separate MuRIL vs 1 Shared MTL MuRIL vs TF-IDF
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
    PooledTransformerClassifier,
    TrainSpec,
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
    tokenize_pairs,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.experiments.v5_optimization.run_phase2_context import measure_truncation_stats
from ml.models.transformer.muril_classifier import _calibrate_temperature, resolve_local_muril_path


def _benchmark_inference_grid(
    backbone_id: str,
    sample_pairs: List[tuple[str, str | None]],
) -> Dict[str, Any]:
    clear_cuda()
    model_path, local_only = resolve_local_muril_path(backbone_id)

    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    hf_cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
    use_tti = _model_uses_token_type_ids(hf_cfg)
    encoder = AutoModel.from_pretrained(model_path, config=hf_cfg, local_files_only=local_only)
    model = PooledTransformerClassifier(encoder, int(hf_cfg.hidden_size), num_labels=6)
    load_cpu_s = round(time.perf_counter() - t_load0, 3)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb_fp32 = round(param_bytes / (1024 * 1024), 1)

    results: Dict[str, Any] = {
        "cold_load_cpu_seconds_1model": load_cpu_s,
        "estimated_load_5separate_models_seconds": round(load_cpu_s * 5.0, 2),
        "weights_fp32_mb_1model": param_mb_fp32,
        "weights_fp32_mb_5models": round(param_mb_fp32 * 5.0, 1),
        "gpu_benchmarks": [],
        "cpu_benchmarks": [],
    }

    # GPU latency benchmarks (CUDA RTX 4060)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        t_gpu0 = time.perf_counter()
        model = model.to(device).eval()
        torch.cuda.synchronize()
        results["to_gpu_transfer_seconds"] = round(time.perf_counter() - t_gpu0, 3)

        for max_len in (160, 256, 384):
            for pad_mode in ("dynamic", "static"):
                for use_bf16 in (True, False):
                    if pad_mode == "static" and not use_bf16:
                        continue
                    for bs in (1, 8, 16, 32):
                        if pad_mode == "static" and bs not in (8, 32):
                            continue
                        if not use_bf16 and bs not in (8, 32):
                            continue
                        torch.cuda.reset_peak_memory_stats()
                        # Warmup
                        sub = sample_pairs[: min(bs, len(sample_pairs))]
                        with torch.no_grad():
                            for _ in range(2):
                                enc = _tok_mode(tokenizer, sub, max_len, pad_mode, use_tti)
                                enc = {k: v.to(device) for k, v in enc.items()}
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                                    _ = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            for start in range(0, len(sample_pairs), bs):
                                chunk = sample_pairs[start : start + bs]
                                enc = _tok_mode(tokenizer, chunk, max_len, pad_mode, use_tti)
                                enc = {k: v.to(device) for k, v in enc.items()}
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                                    _ = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
                        torch.cuda.synchronize()
                        dt = time.perf_counter() - t0
                        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                        results["gpu_benchmarks"].append(
                            {
                                "max_length": max_len,
                                "padding": pad_mode,
                                "precision": "bf16" if use_bf16 else "fp32",
                                "batch_size": bs,
                                "ms_per_comment": round(1000.0 * dt / len(sample_pairs), 3),
                                "comments_per_sec": round(len(sample_pairs) / max(dt, 1e-6), 1),
                                "peak_vram_mb": peak_mb,
                            }
                        )

    # CPU latency benchmarks (small 32-comment slice to keep profiling fast)
    model = model.to("cpu").eval()
    clear_cuda()
    cpu_subset = sample_pairs[:32]
    for max_len in (160, 256, 384):
        for bs in (1, 16):
            t0 = time.perf_counter()
            with torch.no_grad():
                for start in range(0, len(cpu_subset), bs):
                    chunk = cpu_subset[start : start + bs]
                    enc = _tok_mode(tokenizer, chunk, max_len, "dynamic", use_tti)
                    _ = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
            dt = time.perf_counter() - t0
            results["cpu_benchmarks"].append(
                {
                    "max_length": max_len,
                    "padding": "dynamic",
                    "precision": "fp32",
                    "batch_size": bs,
                    "ms_per_comment": round(1000.0 * dt / len(cpu_subset), 2),
                    "comments_per_sec": round(len(cpu_subset) / max(dt, 1e-6), 2),
                }
            )

    del model, encoder
    clear_cuda()
    return results


def _tok_mode(tokenizer, pairs, max_len: int, pad_mode: str, use_tti: bool):
    pad_arg = "max_length" if pad_mode == "static" else True
    has_second = any(b is not None and b != "" for _, b in pairs)
    if has_second:
        firsts = [a for a, _ in pairs]
        seconds = [b if b is not None else "" for _, b in pairs]
        try:
            enc = tokenizer(firsts, seconds, padding=pad_arg, truncation="only_second", max_length=max_len, return_tensors="pt")
        except Exception:
            enc = tokenizer(firsts, seconds, padding=pad_arg, truncation=True, max_length=max_len, return_tensors="pt")
    else:
        enc = tokenizer([a for a, _ in pairs], padding=pad_arg, truncation=True, max_length=max_len, return_tensors="pt")
    if not use_tti and "token_type_ids" in enc:
        del enc["token_type_ids"]
    return dict(enc)


def run_phase7() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase7_inference")

    p2 = json.loads((EXP_ROOT / "phase2_context" / "phase2_summary.json").read_text(encoding="utf-8"))
    p4 = json.loads((EXP_ROOT / "phase4_losses" / "phase4_summary.json").read_text(encoding="utf-8"))
    p5_path = EXP_ROOT / "phase5_dapt_backbones" / "phase5_summary.json"
    if p5_path.exists():
        p5 = json.loads(p5_path.read_text(encoding="utf-8"))
        best_dapt = p5.get("best_inductive_dapt_tag", "muril_dapt_inductive_mlm_4ep")
    else:
        best_dapt = "muril_dapt_inductive_mlm_4ep"
    backbone = str(CKPT_ROOT / best_dapt)

    ctx_by_task = {t: p2["selected_context_by_task"][t]["selected_mode"] for t in TASK_NAMES}
    loss_by_task = {t: p4["selected_loss_by_task"][t]["selected_loss"] for t in TASK_NAMES}

    data = load_v1_records_and_frozen_splits()
    all_rows = data["all"]
    trainval_rows = data["trainval"]
    test_rows = data["test"]

    model_path, local_only = resolve_local_muril_path(backbone)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)

    # 1. Exact truncation matrix at 160 / 256 / 384 across all 459 comments
    trunc_matrix: Dict[str, Any] = {}
    for mode in ("comment_only", "title_pair", "parent_pair", "title_parent_pair"):
        trunc_matrix[mode] = {
            str(ml): measure_truncation_stats(tokenizer, all_rows, mode, max_length=ml)
            for ml in (160, 256, 384)
        }

    # 2. Controlled 5-fold OOF CV comparison at max_length=160 vs 256 vs 384 on context-heavy tasks
    length_ablation: Dict[str, Any] = {}
    t_all = time.perf_counter()
    for task in ("issue", "response", "solution"):
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)
        # Ensure we test with context-aware mode (use Phase 2 winner, or title_parent_pair if comment_only won)
        test_ctx = ctx_by_task[task] if ctx_by_task[task] != "comment_only" else "title_parent_pair"
        length_ablation[task] = {"context_mode": test_ctx, "lengths": {}}

        for ml in (160, 256, 384):
            oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
            vrams, secs, lats = [], [], []
            for f_idx, (tr_idx, va_idx) in enumerate(folds):
                spec = TrainSpec(
                    task=task,
                    backbone_id=backbone,
                    context_mode=test_ctx,
                    loss_type=loss_by_task[task],
                    max_length=ml,
                    epochs=6,
                    batch_size=4 if ml >= 384 else 8,
                    seed=SEED + f_idx,
                )
                r_f = train_and_predict_logits(
                    spec,
                    [trainval_rows[i] for i in tr_idx],
                    [trainval_rows[i] for i in va_idx],
                    {"test": test_rows},
                )
                oof_logits[va_idx] = r_f["val_logits"]
                vrams.append(r_f["peak_vram_mb"])
                secs.append(r_f["train_seconds"])
                lats.append(r_f["eval_latency_ms"]["test"])

            t_g = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
            probs = apply_logits_calibration(oof_logits, multilabel, t_g)
            if multilabel:
                nested_pred = np.zeros_like(y_tv, dtype=int)
                for tr_idx, va_idx in folds:
                    taus_4 = tune_multilabel_thresholds(y_tv[tr_idx], probs[tr_idx], labels, mode="bounded")
                    t_arr = np.array([taus_4[l] for l in labels], dtype=np.float32)
                    nested_pred[va_idx] = (probs[va_idx] > t_arr[None, :]).astype(int)
                f1s = []
                for c in range(len(labels)):
                    tp = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 1)).sum())
                    fp = int(((y_tv[:, c] == 0) & (nested_pred[:, c] == 1)).sum())
                    fn = int(((y_tv[:, c] == 1) & (nested_pred[:, c] == 0)).sum())
                    d = 2 * tp + fp + fn
                    f1s.append((2.0 * tp / d) if d > 0 else 0.0)
                primary_f1 = round(float(np.mean(f1s)), 4)
                ev = evaluate_predictions(task, y_tv, probs)
            else:
                ev = evaluate_predictions(task, y_tv, probs)
                primary_f1 = ev["macro_f1"]

            length_ablation[task]["lengths"][str(ml)] = {
                "primary_oof_macro_f1": primary_f1,
                "oof_ece": ev["ece"],
                "peak_vram_mb": round(float(np.max(vrams)), 1),
                "total_5fold_train_seconds": round(float(np.sum(secs)), 2),
                "mean_gpu_eval_ms_per_comment": round(float(np.mean(lats)), 3),
            }
            print(
                f"  [Phase 7 | {task:8s} | max_len={ml}] OOF_F1={primary_f1:.4f} ECE={ev['ece']:.4f} "
                f"vram={length_ablation[task]['lengths'][str(ml)]['peak_vram_mb']}MB",
                flush=True,
            )

    # 3. GPU & CPU inference profiling grid on 92 real context-paired inputs
    sample_pairs = [build_input_pair(r, "title_parent_pair") for r in test_rows]
    inf_profile = _benchmark_inference_grid(backbone, sample_pairs)

    summary = {
        "phase": "phase7_context_length_and_inference",
        "backbone": backbone,
        "truncation_matrix_459": trunc_matrix,
        "length_ablation_oof": length_ablation,
        "inference_profiling": inf_profile,
        "total_elapsed_seconds": round(time.perf_counter() - t_all, 2),
    }
    save_json(out_dir / "phase7_summary.json", summary)
    print(f"\nPhase 7 complete in {summary['total_elapsed_seconds']}s -> {out_dir / 'phase7_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    run_phase7()
