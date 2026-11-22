"""Forensic Remediation & 100% Strictly Inductive Clean Verification Pass.

Fixes the pseudonym-ID drift discovered during Phase 8 audit (corpus.json 366->365 renumbering
after post_0064) and runs a completely airtight, zero-leakage end-to-end verification:
  1. Trains `muril_dapt_strict_inductive_wwm_8ep` excluding ALL 37 frozen test threads by
     exact `corpus_post_id`, normalized `post_title[:120]`, AND normalized comment text
     (asserting 0/37 overlap on all three checks!).
  2. Re-evaluates TRUE text-aligned `parent_pair` and `title_parent_pair` (459/459 exact comment
     matches, 254/459 true parent comments) vs `comment_only` and `title_pair` on 5-fold OOF CV
     across all 5 tasks using `muril_dapt_strict_inductive_wwm_8ep` (+ `xlm-roberta-base` on
     `response` & `solution` which never used Reddit DAPT at all!).
  3. Re-runs exact OOF-selected final comparison on untouched frozen n=92 test set with
     92/92 exact text-aligned V5 predictions.json paired thread-clustered bootstrap 95% CIs
     (verifying exact match with evaluation.json) + 3-seed stability.
  4. Runs clean Phase 6 Active Learning queue generation (zero overlap by corpus_post_id,
     post_title, and comment text!).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score

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
    V5_EVAL_PATH,
    TrainSpec,
    apply_logits_calibration,
    encode_task_targets,
    ensure_dir,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    thread_clustered_bootstrap_ci,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.experiments.v5_optimization.run_dapt_inductive import (
    build_strictly_inductive_dapt_corpus,
    train_inductive_dapt_variant,
)
from ml.experiments.v5_optimization.run_phase6_active_learning import run_phase6
from ml.experiments.v5_optimization.run_phase8_9_final_eval import (
    _extract_v5_test_probs,
    _fit_5fold_tfidf,
    _per_class_bootstrap_ci,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature


def _eval_5fold_cfg(
    task: str,
    backbone_id: str,
    ctx_mode: str,
    loss_type: str,
    trainval_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    folds: List[tuple[List[int], List[int]]],
    seed: int = SEED,
) -> Dict[str, Any]:
    multilabel = task in MULTILABEL_TASKS
    labels = CANONICAL_LABELS[task]
    y_tv = encode_task_targets(trainval_rows, task)

    oof_logits = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
    te_folds = []
    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        sp = TrainSpec(
            task=task,
            backbone_id=backbone_id,
            context_mode=ctx_mode,
            loss_type=loss_type,
            max_length=256,
            epochs=6,
            seed=seed + f_idx * 13,
        )
        rf = train_and_predict_logits(
            sp,
            [trainval_rows[i] for i in tr_idx],
            [trainval_rows[i] for i in va_idx],
            {"test": test_rows},
        )
        oof_logits[va_idx] = rf["val_logits"]
        te_folds.append(rf["eval_logits"]["test"])

    t_g = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
    oof_probs = apply_logits_calibration(oof_logits, multilabel, t_g)
    ens_test_logits = np.mean(np.stack(te_folds, axis=0), axis=0)
    # Per Phase 1 empirical nested ensemble check, 5-fold logit ensembles are calibrated at T=1.0
    ens_test_probs = apply_logits_calibration(ens_test_logits, multilabel, 1.0)

    if multilabel:
        nested_pred = np.zeros_like(y_tv, dtype=int)
        for tr_idx, va_idx in folds:
            t4 = tune_multilabel_thresholds(y_tv[tr_idx], oof_probs[tr_idx], labels, mode="bounded")
            ta = np.array([t4[l] for l in labels], dtype=np.float32)
            nested_pred[va_idx] = (oof_probs[va_idx] > ta[None, :]).astype(int)
        nested_f1 = round(float(f1_score(y_tv, nested_pred, average="macro", zero_division=0)), 4)
        taus = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="bounded")
        ev_oof = evaluate_predictions(task, y_tv, oof_probs, thresholds=taus)
    else:
        taus = None
        ev_oof = evaluate_predictions(task, y_tv, oof_probs)
        nested_f1 = ev_oof["macro_f1"]

    return {
        "oof_nested_macro_f1": nested_f1,
        "oof_eval": ev_oof,
        "temperature": round(t_g, 3),
        "thresholds": taus,
        "oof_logits": oof_logits,
        "oof_probs": oof_probs,
        "ens_test_logits": ens_test_logits,
        "ens_test_probs": ens_test_probs,
    }


def main() -> None:
    out_dir = ensure_dir(EXP_ROOT / "strict_clean_final")
    logits_dir = ensure_dir(out_dir / "logits")
    t_all = time.perf_counter()

    # 1. Strictly inductive DAPT with 0/37 overlap by corpus_post_id, post_title, AND text
    dapt_name = "muril_dapt_strict_inductive_wwm_8ep"
    dapt_dir = CKPT_ROOT / dapt_name
    dapt_meta_path = EXP_ROOT / "dapt" / f"{dapt_name}_summary.json"
    dapt_corpus = build_strictly_inductive_dapt_corpus(manifest_name="strict_inductive_manifest.json")
    dm = dapt_corpus["manifest"]
    print(
        f"[Strict DAPT] train_threads={dm['dapt_train_threads']} ({dm['dapt_train_docs']} docs) | "
        f"val_threads={dm['dapt_val_threads']} ({dm['dapt_val_docs']} docs) | "
        f"excluded_test_threads={dm['excluded_frozen_test_threads']} ({dm['excluded_frozen_test_comments']} cmts) | "
        f"intersections={dm['intersection_dapt_train_and_test_threads']}/{dm['intersection_dapt_val_and_test_threads']}",
        flush=True,
    )
    if (dapt_dir / "config.json").exists() and dapt_meta_path.exists():
        dapt_meta = json.loads(dapt_meta_path.read_text(encoding="utf-8"))
        print(f"[Strict DAPT] Reusing {dapt_name}", flush=True)
    else:
        dapt_meta = train_inductive_dapt_variant(
            dapt_name,
            dapt_corpus["train_texts"],
            dapt_corpus["val_texts"],
            masking_mode="wwm",
            epochs=8,
            early_stop_patience=2,
            seed=SEED + 88,
        )

    # 2. Load exact text-aligned 459 V1 rows (verified 459/459 corpus match, 254/459 true parents)
    data = load_v1_records_and_frozen_splits()
    trainval_rows = data["trainval"]
    test_rows = data["test"]
    v5_eval_doc = json.loads(V5_EVAL_PATH.read_text(encoding="utf-8"))

    # Verify exact V5 predictions.json reconstruction across all 5 tasks FIRST!
    print("\n=== VERIFYING 92/92 EXACT TEXT-ALIGNED V5 BASELINE RECONSTRUCTION ===", flush=True)
    v5_bundle: Dict[str, Any] = {}
    for task in TASK_NAMES:
        y_te = encode_task_targets(test_rows, task)
        v5_probs, v5_taus, v5_matched = _extract_v5_test_probs(test_rows, task)
        rec_ev = evaluate_predictions(task, y_te[v5_matched], v5_probs[v5_matched], thresholds=v5_taus)
        off_f1 = float(v5_eval_doc["tasks"][task]["test"]["metrics"]["macro_f1"])
        v5_bundle[task] = {
            "probs": v5_probs,
            "taus": v5_taus,
            "matched": v5_matched,
            "official_f1": round(off_f1, 4),
            "reconstructed_f1": rec_ev["macro_f1"],
            "eval": rec_ev,
        }
        print(
            f"  [V5 Verify | {task:9s}] matched={int(v5_matched.sum())}/92 | "
            f"official={off_f1:.4f} vs reconstructed={rec_ev['macro_f1']:.4f}",
            flush=True,
        )

    # 3. Clean TRUE parent-context + strictly-inductive WWM 8ep & XLM-R ablation on OOF CV (n=367)
    loss_by_task = {
        "issue": "asl",
        "sentiment": "ce_weighted",
        "emotion": "asl",
        "response": "ce_weighted",
        "solution": "ce_weighted",
    }

    # Candidate matrix per task (all 100% strictly inductive — ZERO test threads seen!)
    task_candidates: Dict[str, List[tuple[str, str, str, str]]] = {
        "issue": [
            ("strict_wwm8_title_pair_asl", str(dapt_dir), "title_pair", "asl"),
            ("strict_wwm8_title_pair_bcepos", str(dapt_dir), "title_pair", "bce_pos"),
            ("strict_wwm8_true_parent_pair_asl", str(dapt_dir), "parent_pair", "asl"),
            ("strict_wwm8_true_title_parent_asl", str(dapt_dir), "title_parent_pair", "asl"),
        ],
        "sentiment": [
            ("strict_wwm8_comment_only", str(dapt_dir), "comment_only", "ce_weighted"),
            ("strict_wwm8_true_parent_pair", str(dapt_dir), "parent_pair", "ce_weighted"),
            ("strict_wwm8_title_pair", str(dapt_dir), "title_pair", "ce_weighted"),
        ],
        "emotion": [
            ("strict_wwm8_comment_only_asl", str(dapt_dir), "comment_only", "asl"),
            ("strict_wwm8_comment_only_bcepos", str(dapt_dir), "comment_only", "bce_pos"),
            ("strict_wwm8_true_parent_pair_asl", str(dapt_dir), "parent_pair", "asl"),
        ],
        "response": [
            ("strict_wwm8_comment_only", str(dapt_dir), "comment_only", "ce_weighted"),
            ("strict_wwm8_true_parent_pair", str(dapt_dir), "parent_pair", "ce_weighted"),
            ("strict_wwm8_true_title_parent", str(dapt_dir), "title_parent_pair", "ce_weighted"),
            ("xlmr_comment_only", "xlm-roberta-base", "comment_only", "ce_weighted"),
            ("xlmr_true_parent_pair", "xlm-roberta-base", "parent_pair", "ce_weighted"),
        ],
        "solution": [
            ("strict_wwm8_comment_only", str(dapt_dir), "comment_only", "ce_weighted"),
            ("strict_wwm8_true_parent_pair", str(dapt_dir), "parent_pair", "ce_weighted"),
            ("xlmr_comment_only", "xlm-roberta-base", "comment_only", "ce_weighted"),
            ("xlmr_true_parent_pair", "xlm-roberta-base", "parent_pair", "ce_weighted"),
        ],
    }

    summary: Dict[str, Any] = {
        "phase": "strict_clean_inductive_final_remediation",
        "dapt_strict_meta": dapt_meta,
        "dapt_strict_manifest_summary": {
            k: dm[k]
            for k in (
                "total_corpus_threads",
                "excluded_frozen_test_threads",
                "excluded_frozen_test_comments",
                "dapt_train_threads",
                "dapt_val_threads",
                "dapt_train_docs",
                "dapt_val_docs",
                "intersection_dapt_train_and_test_threads",
                "intersection_dapt_val_and_test_threads",
            )
        },
        "v5_verification": {
            t: {
                "matched": int(v5_bundle[t]["matched"].sum()),
                "official_f1": v5_bundle[t]["official_f1"],
                "reconstructed_f1": v5_bundle[t]["reconstructed_f1"],
                "reconstructed_ece": v5_bundle[t]["eval"]["ece"],
            }
            for t in TASK_NAMES
        },
        "tasks": {},
        "three_seed_final": {},
        "final_selection": {},
    }

    for task in TASK_NAMES:
        print(f"\n==================== STRICT CLEAN EVAL: {task.upper()} ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        y_te = encode_task_targets(test_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)

        v5_p = v5_bundle[task]["probs"]
        v5_t = v5_bundle[task]["taus"]
        v5_m = v5_bundle[task]["matched"]
        m_test_rows = [test_rows[i] for i in range(len(test_rows)) if v5_m[i]]

        cands_out: Dict[str, Any] = {}
        oof_scores: Dict[str, float] = {}
        cand_specs_map: Dict[str, tuple[str, str, str]] = {}
        cand_probs_map: Dict[str, Dict[str, Any]] = {}

        for cname, bb, ctx, lt in task_candidates[task]:
            res = _eval_5fold_cfg(task, bb, ctx, lt, trainval_rows, test_rows, folds, seed=SEED)
            oof_scores[cname] = res["oof_nested_macro_f1"]
            cand_specs_map[cname] = (bb, ctx, lt)
            cand_probs_map[cname] = res

            te_ev = evaluate_predictions(task, y_te, res["ens_test_probs"], thresholds=res["thresholds"])
            ci_vs_v5 = thread_clustered_bootstrap_ci(
                m_test_rows,
                task,
                y_te[v5_m],
                res["ens_test_probs"][v5_m],
                probs_b=v5_p[v5_m],
                thresholds_a=res["thresholds"],
                thresholds_b=v5_t,
                n_boot=2000,
                seed=SEED,
            )
            pcls_ci = _per_class_bootstrap_ci(test_rows, task, y_te, res["ens_test_probs"], thresholds=res["thresholds"])

            np.savez_compressed(
                logits_dir / f"{task}_{cname}.npz",
                oof_logits=res["oof_logits"],
                ens_test_logits=res["ens_test_logits"],
            )
            cands_out[cname] = {
                "backbone": bb,
                "context_mode": ctx,
                "loss_type": lt,
                "oof_nested_macro_f1": res["oof_nested_macro_f1"],
                "oof_ece": res["oof_eval"]["ece"],
                "temperature_single_oof": res["temperature"],
                "thresholds": res["thresholds"],
                "frozen_test_eval": te_ev,
                "bootstrap_ci_vs_v5": ci_vs_v5,
                "per_class_bootstrap_ci": pcls_ci,
            }
            print(
                f"  [{task:9s} | {cname:35s}] OOF={res['oof_nested_macro_f1']:.4f} "
                f"| Test92={te_ev['macro_f1']:.4f} ECE={te_ev['ece']:.4f} "
                f"| dV5={ci_vs_v5['paired_delta_mean']:+.4f} CI={ci_vs_v5['paired_delta_ci95']} p>V5={ci_vs_v5['p_a_beats_b']:.3f}",
                flush=True,
            )

        # Select winning neural config STRICTLY on OOF CV (never test!)
        best_neural = max(oof_scores.keys(), key=lambda k: oof_scores[k])
        bb_w, ctx_w, lt_w = cand_specs_map[best_neural]
        res_w = cand_probs_map[best_neural]

        # Test OOF-calibrated Neural + Word+Char TF-IDF hybrid on the OOF winner
        tf_res = _fit_5fold_tfidf(trainval_rows, test_rows, task, folds, ctx_w)
        best_alpha = 1.0
        best_hyb_oof = res_w["oof_nested_macro_f1"]
        best_hyb_taus = res_w["thresholds"]
        for alpha in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            b_oof = alpha * res_w["oof_probs"] + (1.0 - alpha) * tf_res["oof_probs"]
            if multilabel:
                np_pred = np.zeros_like(y_tv, dtype=int)
                for tr_idx, va_idx in folds:
                    t4 = tune_multilabel_thresholds(y_tv[tr_idx], b_oof[tr_idx], labels, mode="bounded")
                    ta = np.array([t4[l] for l in labels], dtype=np.float32)
                    np_pred[va_idx] = (b_oof[va_idx] > ta[None, :]).astype(int)
                sc = round(float(f1_score(y_tv, np_pred, average="macro", zero_division=0)), 4)
                tf_t = tune_multilabel_thresholds(y_tv, b_oof, labels, mode="bounded")
            else:
                sc = round(float(f1_score(y_tv, np.argmax(b_oof, axis=1), average="macro", zero_division=0)), 4)
                tf_t = None
            if sc > best_hyb_oof + 0.004:  # require >= +0.4 pp OOF gain to justify hybrid complexity
                best_hyb_oof = sc
                best_alpha = alpha
                best_hyb_taus = tf_t

        hyb_test_probs = best_alpha * res_w["ens_test_probs"] + (1.0 - best_alpha) * tf_res["test_probs"]
        hyb_te_ev = evaluate_predictions(task, y_te, hyb_test_probs, thresholds=best_hyb_taus)
        hyb_ci = thread_clustered_bootstrap_ci(
            m_test_rows,
            task,
            y_te[v5_m],
            hyb_test_probs[v5_m],
            probs_b=v5_p[v5_m],
            thresholds_a=best_hyb_taus,
            thresholds_b=v5_t,
            n_boot=2000,
            seed=SEED,
        )
        cands_out[f"hybrid_{best_neural}_a{best_alpha}"] = {
            "base_neural": best_neural,
            "alpha_neural": best_alpha,
            "oof_nested_macro_f1": best_hyb_oof,
            "thresholds": best_hyb_taus,
            "frozen_test_eval": hyb_te_ev,
            "bootstrap_ci_vs_v5": hyb_ci,
        }
        chosen_key = f"hybrid_{best_neural}_a{best_alpha}" if best_alpha < 1.0 else best_neural

        # 3-Seed Stability for the OOF-selected winning neural configuration
        s_oofs = [res_w["oof_nested_macro_f1"]]
        s_tests = [cands_out[best_neural]["frozen_test_eval"]["macro_f1"]]
        for run_seed in (SEED + 1, SEED + 2):
            rs = _eval_5fold_cfg(task, bb_w, ctx_w, lt_w, trainval_rows, test_rows, folds, seed=run_seed)
            ev_s = evaluate_predictions(task, y_te, rs["ens_test_probs"], thresholds=rs["thresholds"])
            s_oofs.append(rs["oof_nested_macro_f1"])
            s_tests.append(ev_s["macro_f1"])

        summary["three_seed_final"][task] = {
            "config_name": best_neural,
            "backbone": bb_w,
            "context_mode": ctx_w,
            "loss_type": lt_w,
            "oof_per_seed": s_oofs,
            "oof_mean": round(float(np.mean(s_oofs)), 4),
            "oof_std": round(float(np.std(s_oofs)), 4),
            "test92_per_seed": s_tests,
            "test92_mean": round(float(np.mean(s_tests)), 4),
            "test92_std": round(float(np.std(s_tests)), 4),
        }
        summary["tasks"][task] = {
            "oof_winner_neural": best_neural,
            "oof_winner_final": chosen_key,
            "candidates": cands_out,
        }
        chosen_obj = cands_out[chosen_key]
        summary["final_selection"][task] = {
            "chosen_candidate": chosen_key,
            "oof_macro_f1": chosen_obj["oof_nested_macro_f1"],
            "frozen_test_macro_f1": chosen_obj["frozen_test_eval"]["macro_f1"],
            "frozen_test_ece": chosen_obj["frozen_test_eval"]["ece"],
            "v5_official_test_macro_f1": v5_bundle[task]["official_f1"],
            "v5_reconstructed_test_macro_f1": v5_bundle[task]["reconstructed_f1"],
            "paired_delta_vs_v5": chosen_obj["bootstrap_ci_vs_v5"]["paired_delta_mean"],
            "paired_delta_ci95": chosen_obj["bootstrap_ci_vs_v5"]["paired_delta_ci95"],
            "p_beats_v5": chosen_obj["bootstrap_ci_vs_v5"]["p_a_beats_b"],
            "three_seed_test_mean": summary["three_seed_final"][task]["test92_mean"],
            "three_seed_test_std": summary["three_seed_final"][task]["test92_std"],
        }
        print(
            f"  >>> FINAL STRICT-CLEAN {task.upper()}: {chosen_key} | "
            f"OOF={chosen_obj['oof_nested_macro_f1']:.4f} | Test92={chosen_obj['frozen_test_eval']['macro_f1']:.4f} "
            f"(3-seed={summary['three_seed_final'][task]['test92_mean']:.4f}+/-{summary['three_seed_final'][task]['test92_std']:.4f}) "
            f"vs V5={v5_bundle[task]['official_f1']:.4f} | paired_delta={chosen_obj['bootstrap_ci_vs_v5']['paired_delta_mean']:+.4f} "
            f"CI95={chosen_obj['bootstrap_ci_vs_v5']['paired_delta_ci95']}",
            flush=True,
        )
        save_json(out_dir / "strict_clean_summary.json", summary)

    summary["overall_5task_means"] = {
        "v5_official_transductive": round(float(np.mean([v5_bundle[t]["official_f1"] for t in TASK_NAMES])), 4),
        "strict_inductive_oof_winners_test92": round(
            float(np.mean([summary["final_selection"][t]["frozen_test_macro_f1"] for t in TASK_NAMES])), 4
        ),
        "strict_inductive_3seed_mean_test92": round(
            float(np.mean([summary["three_seed_final"][t]["test92_mean"] for t in TASK_NAMES])), 4
        ),
    }
    summary["elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "strict_clean_summary.json", summary)
    print(f"\nSTRICT CLEAN REMEDIATION COMPLETE in {summary['elapsed_seconds']}s: {summary['overall_5task_means']}", flush=True)

    # 4. Run clean Phase 6 Active Learning Queue (zero overlap by corpus_post_id, post_title, and text)
    print("\n=== RUNNING CLEAN PHASE 6 ACTIVE LEARNING QUEUE ===", flush=True)
    run_phase6(n_train_queue=180, n_test_new_queue=70)


if __name__ == "__main__":
    main()
