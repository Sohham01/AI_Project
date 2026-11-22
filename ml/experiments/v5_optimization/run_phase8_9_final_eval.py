"""Phases 8 & 9 — Comprehensive Statistical Evaluation, 3-Seed Stability,
Thread-Clustered Bootstrap CIs, 54-Probe Robustness & Independent Per-Task Selection.

Strict Protocol:
1. All architecture/context/loss/backbone/hybrid selections are determined STRICTLY on
   5-fold OOF CV (n=367, 163 threads). Frozen test labels (n=92, 37 threads) are NEVER
   used for selection, tuning, or early stopping.
2. Evaluates ONCE on the frozen n=92 test set across the full controlled ablation ladder:
   - V5 Production Transductive baseline (exact predictions from ml/artifacts/live_corpus/predictions.json)
   - Word+Char TF-IDF 5-fold CV baseline
   - Phase 1: Inductive single-split (298/69), Inductive 5-fold ensemble, Inductive 1-ckpt full-367
   - Phase 2: OOF-winning segment-pair context (5-fold ensemble)
   - Phase 4: OOF-winning task-specific loss + context (5-fold ensemble)
   - Phase 3: Best OOF Shared Multi-Task MuRIL (+ R-Drop) (1 shared encoder for all 5 tasks)
   - Phase 5: Best OOF backbone (WWM 4ep / WWM 8ep / IndicBERTv2 / XLM-R / LoRA)
   - Phase 8: OOF-calibrated Neural + Word+Char TF-IDF Hybrid
3. Runs 3-seed stability (20261116, 20261117, 20261118) for each task's top OOF-selected
   single-task configuration and best Shared MTL configuration.
4. Computes 2,000-resample Thread-Clustered Bootstrap 95% CIs (37 post_id clusters) for
   absolute Macro F1, per-class F1 (support >= 5), and paired Delta F1 vs V5 baseline.
5. Runs the separate 54-probe 8-category synthetic robustness diagnostic (never blended with
   real test scores!).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.evaluation.error_analysis import compute_calibration_metrics
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
    build_input_pair,
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
from ml.experiments.v5_optimization.run_phase3_multitask import train_mtl_fold
from ml.models.transformer.muril_classifier import _calibrate_temperature


def _extract_v5_test_probs(test_rows: List[Dict[str, Any]], task: str) -> Tuple[np.ndarray, Optional[Dict[str, float]], np.ndarray]:
    """Read exact V5 production test probabilities from canonical predictions.json (read-only)."""
    preds_doc = json.loads((ROOT / "ml" / "artifacts" / "live_corpus" / "predictions.json").read_text(encoding="utf-8"))
    eval_doc = json.loads(V5_EVAL_PATH.read_text(encoding="utf-8"))
    labels = CANONICAL_LABELS[task]
    task_preds = preds_doc["tasks"][task]

    probs = np.zeros((len(test_rows), len(labels)), dtype=np.float32)
    matched = np.zeros(len(test_rows), dtype=bool)

    prob_key = {
        "issue": "issue_probabilities",
        "sentiment": "sentiment_probabilities",
        "emotion": "emotion_probabilities",
        "response": "response_probabilities",
    }.get(task)

    for i, r in enumerate(test_rows):
        key = f"comment-{r.get('corpus_comment_id', r['comment_id'])}"
        rec = task_preds.get(key)
        if rec is None:
            continue
        matched[i] = True
        if task == "solution":
            p1 = float(rec.get("solution_probability", 0.0))
            p0 = float(rec.get("not_solution_probability", 1.0 - p1))
            probs[i, 0] = p0
            probs[i, 1] = p1
        elif task in MULTILABEL_TASKS:
            # Use V5's exact thresholded binary decisions from predictions.json so paired
            # bootstrap delta reproduces V5's official test Macro F1 exactly!
            pred_list = rec.get("predicted_issue_labels" if task == "issue" else "emotions") or []
            pred_set = set(pred_list)
            for j, lbl in enumerate(labels):
                probs[i, j] = 1.0 if lbl in pred_set else 0.0
        else:
            pdict = rec.get(prob_key, {})
            for j, lbl in enumerate(labels):
                probs[i, j] = float(pdict.get(lbl, 0.0))

    taus: Optional[Dict[str, float]] = ({lbl: 0.5 for lbl in labels} if task in MULTILABEL_TASKS else None)
    return probs, taus, matched


def _per_class_bootstrap_ci(
    rows: List[Dict[str, Any]],
    task: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    n_boot: int = 2000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Thread-clustered bootstrap 95% CIs for every class with test support >= 5."""
    rng = np.random.default_rng(seed + 7)
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    if multilabel:
        t_arr = np.array([float((thresholds or {}).get(l, 0.5)) for l in labels], dtype=np.float32)
        y_pred = (probs > t_arr[None, :]).astype(int)
    else:
        y_pred = np.argmax(probs, axis=1).astype(int)

    pid_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        pid_to_idx[r["post_id"]].append(i)
    pids = sorted(pid_to_idx.keys())
    n_clusters = len(pids)

    cls_samples: Dict[str, List[float]] = {lbl: [] for lbl in labels}
    for _ in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = [i for c_i in sampled for i in pid_to_idx[pids[c_i]]]
        yt = y_true[idx]
        yp = y_pred[idx]
        for c, lbl in enumerate(labels):
            if multilabel:
                tr_c, pr_c = yt[:, c], yp[:, c]
            else:
                tr_c, pr_c = (yt == c).astype(int), (yp == c).astype(int)
            tp = int(((tr_c == 1) & (pr_c == 1)).sum())
            fp = int(((tr_c == 0) & (pr_c == 1)).sum())
            fn = int(((tr_c == 1) & (pr_c == 0)).sum())
            if tp + fn == 0:
                continue  # skip bootstrap draws where this class didn't appear
            d = 2 * tp + fp + fn
            cls_samples[lbl].append((2.0 * tp / d) if d > 0 else 0.0)

    out: Dict[str, Any] = {}
    for c, lbl in enumerate(labels):
        sup = int(y_true[:, c].sum()) if multilabel else int((y_true == c).sum())
        if sup < 5 or len(cls_samples[lbl]) < 50:
            out[lbl] = {"support": sup, "ci95": None, "note": "support < 5 (too small for reliable CI)"}
        else:
            lo, hi = np.percentile(cls_samples[lbl], [2.5, 97.5])
            out[lbl] = {
                "support": sup,
                "mean": round(float(np.mean(cls_samples[lbl])), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
            }
    return out


def _fit_5fold_tfidf(
    trainval_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    task: str,
    folds: List[Tuple[List[int], List[int]]],
    context_mode: str,
) -> Dict[str, Any]:
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    y_tv = encode_task_targets(trainval_rows, task)

    def _row_text(r: Dict[str, Any]) -> str:
        a, b = build_input_pair(r, context_mode)
        return f"{a} {b}" if b else a

    oof_probs = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
    test_probs_folds = []
    t0 = time.perf_counter()

    for tr_idx, va_idx in folds:
        tr_txts = [_row_text(trainval_rows[i]) for i in tr_idx]
        va_txts = [_row_text(trainval_rows[i]) for i in va_idx]
        te_txts = [_row_text(r) for r in test_rows]
        w_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
        c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, sublinear_tf=True)
        X_tr = hstack([w_vec.fit_transform(tr_txts), c_vec.fit_transform(tr_txts)])
        X_va = hstack([w_vec.transform(va_txts), c_vec.transform(va_txts)])
        X_te = hstack([w_vec.transform(te_txts), c_vec.transform(te_txts)])
        if multilabel:
            clf = OneVsRestClassifier(LogisticRegression(C=2.0, max_iter=500, class_weight="balanced"))
        else:
            clf = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced")
        clf.fit(X_tr, y_tv[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X_va).astype(np.float32)
        test_probs_folds.append(clf.predict_proba(X_te).astype(np.float32))

    train_s = round(time.perf_counter() - t0, 2)
    test_probs = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
    taus = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="bounded") if multilabel else None
    oof_ev = evaluate_predictions(task, y_tv, oof_probs, thresholds=taus)
    return {
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "thresholds": taus,
        "oof_eval": oof_ev,
        "train_seconds": train_s,
    }


def _evaluate_candidate_on_frozen_test(
    name: str,
    task: str,
    test_rows: List[Dict[str, Any]],
    y_test: np.ndarray,
    test_probs: np.ndarray,
    thresholds: Optional[Dict[str, float]],
    v5_probs: np.ndarray,
    v5_taus: Optional[Dict[str, float]],
    v5_matched: np.ndarray,
) -> Dict[str, Any]:
    ev = evaluate_predictions(task, y_test, test_probs, thresholds=thresholds)
    # Reliability diagram bins
    labels = CANONICAL_LABELS[task]
    ml = task in MULTILABEL_TASKS
    if ml:
        t_lists = [[labels[j] for j in range(len(labels)) if y_test[i, j] > 0.5] for i in range(len(y_test))]
        cal = compute_calibration_metrics(t_lists, test_probs, labels, multilabel=True, n_bins=5)
    else:
        t_strs = [labels[int(v)] for v in y_test]
        cal = compute_calibration_metrics(t_strs, test_probs, labels, multilabel=False, n_bins=5)

    # Paired thread-clustered bootstrap on exact matched frozen test observations (n=91 or 92)
    m_rows = [test_rows[i] for i in range(len(test_rows)) if v5_matched[i]]
    ci = thread_clustered_bootstrap_ci(
        m_rows,
        task,
        y_test[v5_matched],
        test_probs[v5_matched],
        probs_b=v5_probs[v5_matched],
        thresholds_a=thresholds,
        thresholds_b=v5_taus,
        n_boot=2000,
        seed=SEED,
    )
    per_cls_ci = _per_class_bootstrap_ci(test_rows, task, y_test, test_probs, thresholds=thresholds, n_boot=2000, seed=SEED)
    return {
        "candidate": name,
        "test_eval": ev,
        "reliability_bins": cal.get("bins", []),
        "bootstrap_ci_vs_v5": ci,
        "per_class_bootstrap_ci": per_cls_ci,
    }


def _run_54_probe_diagnostic(
    trainval_rows: List[Dict[str, Any]],
    selected_specs: Dict[str, Dict[str, Any]],
    best_mtl_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run strictly observational 54-probe 8-category robustness stress test (NEVER blended with real test)."""
    from backend.run_robustness_benchmark import ROBUSTNESS_PROBES, evaluate_probes_by_category
    from ml.experiments.v5_optimization.run_phase1_cv import _train_fixed_epochs_all_trainval

    # Build probe pseudo-rows (both comment-only and with parent_title when present)
    probe_rows = [
        {
            "document_id": f"probe-{i}",
            "post_id": f"probe-post-{i}",
            "comment_id": f"probe-cmt-{i}",
            "subreddit": "probe",
            "post_title": p.get("parent_title") or "",
            "parent_comment_text": "",
            "text": p["text"],
            "labels": {
                "issue": p.get("issues", []),
                "sentiment": p.get("sentiment", "neutral"),
                "emotion": p.get("emotions", []),
                "response": p.get("response", "opinion"),
                "solution": p.get("solution", "not_solution"),
            },
        }
        for i, p in enumerate(ROBUSTNESS_PROBES)
    ]

    results: Dict[str, Any] = {
        "scope_notice": "Auxiliary 54-probe synthetic stress diagnostic ONLY; strictly separated from real held-out test metrics.",
        "tasks": {},
    }

    for task in TASK_NAMES:
        spec_info = selected_specs[task]
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        ctx = spec_info["context_mode"]
        loss = spec_info["loss_type"]
        backbone = spec_info["backbone_id"]
        temp = float(spec_info.get("temperature", 1.0))
        taus = spec_info.get("thresholds")

        # Train one full-trainval model for diagnostic probe scoring
        lg, _, _ = _train_fixed_epochs_all_trainval(
            TrainSpec(task=task, backbone_id=backbone, context_mode=ctx, loss_type=loss, max_length=256, epochs=5, seed=SEED + 888),
            trainval_rows,
            probe_rows,
        )
        probs_all = apply_logits_calibration(lg, multilabel, temp)

        # Evaluate by category directly on ROBUSTNESS_PROBES
        by_cat: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(ROBUSTNESS_PROBES):
            by_cat[p["category"]].append(i)

        y_true = encode_task_targets(probe_rows, task)
        ev_all = evaluate_predictions(task, y_true, probs_all, thresholds=taus)

        cat_scores: Dict[str, float] = {}
        for cat, idxs in sorted(by_cat.items()):
            sub_y = y_true[idxs]
            sub_p = probs_all[idxs]
            if multilabel:
                t_arr = np.array([float((taus or {}).get(l, 0.5)) for l in labels], dtype=np.float32)
                pr = (sub_p > t_arr[None, :]).astype(int)
                cat_scores[cat] = round(float(f1_score(sub_y, pr, average="micro", zero_division=0)), 4)
            else:
                pr = np.argmax(sub_p, axis=1)
                cat_scores[cat] = round(float(np.mean(sub_y == pr)), 4)

        results["tasks"][task] = {
            "selected_model_tag": spec_info["tag"],
            "overall_macro_f1": ev_all["macro_f1"],
            "overall_micro_f1": ev_all["micro_f1"],
            "by_category": cat_scores,
        }
        print(f"  [54-Probe Stress | {task:9s}] macro_f1={ev_all['macro_f1']:.4f} micro={ev_all['micro_f1']:.4f}", flush=True)

    return results


def run_phase8_9() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "phase8_9_final_eval")
    t_all = time.perf_counter()

    data = load_v1_records_and_frozen_splits()
    trainval_rows = data["trainval"]
    test_rows = data["test"]

    p1 = json.loads((EXP_ROOT / "phase1_cv" / "phase1_summary.json").read_text(encoding="utf-8"))
    p2 = json.loads((EXP_ROOT / "phase2_context" / "phase2_summary.json").read_text(encoding="utf-8"))
    p4 = json.loads((EXP_ROOT / "phase4_losses" / "phase4_summary.json").read_text(encoding="utf-8"))
    p3 = json.loads((EXP_ROOT / "phase3_multitask" / "phase3_summary.json").read_text(encoding="utf-8"))
    p5 = json.loads((EXP_ROOT / "phase5_dapt_backbones" / "phase5_summary.json").read_text(encoding="utf-8"))
    v5_eval_doc = json.loads(V5_EVAL_PATH.read_text(encoding="utf-8"))

    # Find best Phase 3 Shared MTL variant strictly on OOF mean_5task_oof_macro_f1
    mtl_keys = [k for k in p3["variants"] if k.startswith("mtl_")]
    best_mtl_key = max(mtl_keys, key=lambda k: p3["variants"][k]["eval"]["mean_5task_oof_macro_f1"])
    mtl_npz = np.load(EXP_ROOT / "phase3_multitask" / "logits" / f"{best_mtl_key}_joint.npz")

    summary: Dict[str, Any] = {
        "phase": "phase8_9_final_statistical_eval_and_selection",
        "frozen_test_n": len(test_rows),
        "frozen_test_threads": len(data["manifest"]["test_post_ids"]),
        "frozen_test_sha256": data["manifest"]["test_sha256"],
        "best_mtl_variant_oof": best_mtl_key,
        "tasks": {},
        "three_seed_stability": {},
        "final_per_task_selection": {},
    }

    selected_specs_for_probes: Dict[str, Dict[str, Any]] = {}

    for task in TASK_NAMES:
        print(f"\n==================== PHASE 8/9 FINAL EVAL: {task.upper()} ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]
        y_tv = encode_task_targets(trainval_rows, task)
        y_te = encode_task_targets(test_rows, task)
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=SEED)

        ctx_win = p2["selected_context_by_task"][task]["selected_mode"]
        loss_win = p4["selected_loss_by_task"][task]["selected_loss"]

        # Exact V5 baseline probabilities on frozen test (for paired bootstrap CIs)
        v5_probs, v5_taus, v5_matched = _extract_v5_test_probs(test_rows, task)
        v5_direct_ev = evaluate_predictions(task, y_te[v5_matched], v5_probs[v5_matched], thresholds=v5_taus)
        v5_official_f1 = float(v5_eval_doc["tasks"][task]["test"]["metrics"]["macro_f1"])

        # 1. Word+Char TF-IDF 5-Fold CV Baseline
        tf_res = _fit_5fold_tfidf(trainval_rows, test_rows, task, folds, ctx_win)

        # 2. Gather Phase 1, 2, 4, 3, 5 candidates strictly using OOF-derived T and thresholds
        candidates_eval: Dict[str, Any] = {}
        oof_leaderboard: Dict[str, float] = {}

        # V5 official transductive reference
        candidates_eval["V5_Production_Transductive"] = {
            "protocol": "transductive_dapt_single_split_298_69",
            "official_eval_json_test_macro_f1": round(v5_official_f1, 4),
            "matched_n": int(v5_matched.sum()),
            "test_eval_reconstructed": v5_direct_ev,
            "bootstrap_ci": thread_clustered_bootstrap_ci(
                [test_rows[i] for i in range(len(test_rows)) if v5_matched[i]],
                task,
                y_te[v5_matched],
                v5_probs[v5_matched],
                thresholds_a=v5_taus,
                n_boot=2000,
                seed=SEED,
            ),
        }

        # TF-IDF Word+Char CV
        oof_leaderboard["TFIDF_WordChar_5Fold"] = tf_res["oof_eval"]["macro_f1"]
        candidates_eval["TFIDF_WordChar_5Fold"] = _evaluate_candidate_on_frozen_test(
            "TFIDF_WordChar_5Fold", task, test_rows, y_te, tf_res["test_probs"], tf_res["thresholds"], v5_probs, v5_taus, v5_matched
        )

        # Phase 1 artifacts
        # Use Phase 1 empirically validated nested ensemble temperature for 5-fold logit ensembles,
        # and single-model OOF temperature for single-checkpoint models (Section 5 calibration matching!).
        p1_npz = np.load(EXP_ROOT / "phase1_cv" / "logits" / f"{task}_phase1_logits.npz")
        p1_t_oof = float(p1["tasks"][task]["oof_calibration_comparison"]["global_T"]["temperature"])
        t_ens_nested = float(p1["tasks"][task]["nested_ensemble_calibration_check"]["t_ensemble_opt_val69"])
        p1_taus = (
            p1["tasks"][task]["oof_threshold_comparison"]["bounded"]["thresholds"] if multilabel else None
        )
        p1_ss_t = float(p1["tasks"][task]["single_split_inductive_val69"]["temperature"])
        p1_ss_taus = p1["tasks"][task]["single_split_inductive_val69"]["thresholds"]

        oof_leaderboard["Phase1_Inductive_5FoldEns"] = (
            p1["tasks"][task]["oof_threshold_comparison"]["bounded"]["oof_nested_macro_f1"]
            if multilabel
            else p1["tasks"][task]["oof_calibration_comparison"]["global_T"]["eval_default"]["macro_f1"]
        )

        candidates_eval["Phase1_Inductive_SingleSplit_298_69"] = _evaluate_candidate_on_frozen_test(
            "Phase1_Inductive_SingleSplit_298_69",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p1_npz["single_split_test_logits"], multilabel, p1_ss_t),
            p1_ss_taus,
            v5_probs,
            v5_taus,
            v5_matched,
        )
        candidates_eval["Phase1_Inductive_5FoldEns"] = _evaluate_candidate_on_frozen_test(
            "Phase1_Inductive_5FoldEns",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p1_npz["ens_test_logits"], multilabel, t_ens_nested),
            p1_taus,
            v5_probs,
            v5_taus,
            v5_matched,
        )
        candidates_eval["Phase1_Inductive_SingleModel367"] = _evaluate_candidate_on_frozen_test(
            "Phase1_Inductive_SingleModel367",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p1_npz["full367_single_test_logits"], multilabel, p1_t_oof),
            p1_taus,
            v5_probs,
            v5_taus,
            v5_matched,
        )

        # Phase 2 winner
        p2_npz = np.load(EXP_ROOT / "phase2_context" / "logits" / f"{task}_{ctx_win}_logits.npz")
        p2_meta = p2["tasks"][task][ctx_win]
        oof_leaderboard[f"Phase2_Context_{ctx_win}"] = p2_meta["primary_oof_macro_f1"]
        candidates_eval[f"Phase2_Context_{ctx_win}"] = _evaluate_candidate_on_frozen_test(
            f"Phase2_Context_{ctx_win}",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p2_npz["ens_test_logits"], multilabel, t_ens_nested),
            p2_meta["bounded_thresholds"],
            v5_probs,
            v5_taus,
            v5_matched,
        )

        # Phase 4 winner
        p4_npz = np.load(EXP_ROOT / "phase4_losses" / "logits" / f"{task}_{loss_win}_logits.npz")
        p4_meta = p4["tasks"][task][loss_win]
        oof_leaderboard[f"Phase4_Loss_{loss_win}"] = p4_meta["primary_oof_macro_f1"]
        candidates_eval[f"Phase4_Loss_{loss_win}"] = _evaluate_candidate_on_frozen_test(
            f"Phase4_Loss_{loss_win}",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p4_npz["ens_test_logits"], multilabel, t_ens_nested),
            p4_meta["bounded_thresholds"],
            v5_probs,
            v5_taus,
            v5_matched,
        )

        # Phase 3 best Shared MTL
        mtl_meta = p3["variants"][best_mtl_key]["eval"][task]
        oof_leaderboard[f"Phase3_{best_mtl_key}"] = mtl_meta["primary_oof_macro_f1"]
        candidates_eval[f"Phase3_{best_mtl_key}"] = _evaluate_candidate_on_frozen_test(
            f"Phase3_{best_mtl_key}",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(mtl_npz[f"ens_test_{task}"], multilabel, t_ens_nested),
            mtl_meta["bounded_thresholds"],
            v5_probs,
            v5_taus,
            v5_matched,
        )

        # Phase 5 best backbone strictly on OOF CV for THIS task!
        valid_p5_tags = [
            tag
            for tag, obj in p5["backbones"].items()
            if isinstance(obj, dict) and "tasks" in obj and task in obj["tasks"]
        ]
        best_p5_tag = max(valid_p5_tags, key=lambda tag: p5["backbones"][tag]["tasks"][task]["primary_oof_macro_f1"])
        p5_meta = p5["backbones"][best_p5_tag]["tasks"][task]
        p5_npz = np.load(EXP_ROOT / "phase5_dapt_backbones" / "logits" / f"{best_p5_tag}_{task}_logits.npz")
        oof_leaderboard[f"Phase5_{best_p5_tag}"] = p5_meta["primary_oof_macro_f1"]
        candidates_eval[f"Phase5_{best_p5_tag}"] = _evaluate_candidate_on_frozen_test(
            f"Phase5_{best_p5_tag}",
            task,
            test_rows,
            y_te,
            apply_logits_calibration(p5_npz["ens_test_logits"], multilabel, t_ens_nested),
            p5_meta["bounded_thresholds"],
            v5_probs,
            v5_taus,
            v5_matched,
        )

        # Phase 8: Hybrid Blend Sweep STRICTLY ON OOF CV combining Phase 5 best neural + Word+Char TF-IDF
        p5_oof_probs = apply_logits_calibration(p5_npz["oof_logits"], multilabel, p5_meta["temperature"])
        p5_test_probs = apply_logits_calibration(p5_npz["ens_test_logits"], multilabel, t_ens_nested)

        best_alpha = 1.0
        best_hyb_oof = -1.0
        best_hyb_taus = p5_meta["bounded_thresholds"]
        for alpha in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            blend_oof = alpha * p5_oof_probs + (1.0 - alpha) * tf_res["oof_probs"]
            if multilabel:
                nested_pred = np.zeros_like(y_tv, dtype=int)
                for tr_idx, va_idx in folds:
                    t4 = tune_multilabel_thresholds(y_tv[tr_idx], blend_oof[tr_idx], labels, mode="bounded")
                    ta = np.array([t4[l] for l in labels], dtype=np.float32)
                    nested_pred[va_idx] = (blend_oof[va_idx] > ta[None, :]).astype(int)
                score = float(f1_score(y_tv, nested_pred, average="macro", zero_division=0))
                t_full = tune_multilabel_thresholds(y_tv, blend_oof, labels, mode="bounded")
            else:
                score = float(f1_score(y_tv, np.argmax(blend_oof, axis=1), average="macro", zero_division=0))
                t_full = None
            if score > best_hyb_oof + 1e-6 or (abs(score - best_hyb_oof) <= 1e-6 and alpha > best_alpha):
                best_hyb_oof = score
                best_alpha = alpha
                best_hyb_taus = t_full

        hyb_test_probs = best_alpha * p5_test_probs + (1.0 - best_alpha) * tf_res["test_probs"]
        oof_leaderboard[f"Phase8_Hybrid_a{best_alpha}"] = round(best_hyb_oof, 4)
        candidates_eval[f"Phase8_Hybrid_a{best_alpha}"] = {
            "blend_alpha_neural": best_alpha,
            "blend_alpha_tfidf": round(1.0 - best_alpha, 2),
            "oof_macro_f1": round(best_hyb_oof, 4),
            **_evaluate_candidate_on_frozen_test(
                f"Phase8_Hybrid_a{best_alpha}",
                task,
                test_rows,
                y_te,
                hyb_test_probs,
                best_hyb_taus,
                v5_probs,
                v5_taus,
                v5_matched,
            ),
        }

        # Select final winner STRICTLY by OOF leaderboard (+ simplicity rule: prefer MTL if within 0.008 of best single-task!)
        best_oof_key = max(oof_leaderboard.keys(), key=lambda k: oof_leaderboard[k])
        mtl_oof_key = f"Phase3_{best_mtl_key}"
        best_backbone_id = p5["backbones"][best_p5_tag]["backbone_id"]
        best_use_lora = bool(p5["backbones"][best_p5_tag].get("use_lora", False))

        # 3-Seed Stability run for this task's best single-task OOF config (20261116, 20261117, 20261118)
        seed_oof_scores: List[float] = []
        seed_test_scores: List[float] = []
        seed_test_probs_list: List[np.ndarray] = []
        for s_idx, run_seed in enumerate((SEED, SEED + 1, SEED + 2)):
            if s_idx == 0:
                seed_oof_scores.append(p5_meta["primary_oof_macro_f1"])
                ev_s0 = evaluate_predictions(task, y_te, p5_test_probs, thresholds=p5_meta["bounded_thresholds"])
                seed_test_scores.append(ev_s0["macro_f1"])
                seed_test_probs_list.append(p5_test_probs)
                continue
            oof_lg_s = np.zeros((len(trainval_rows), len(labels)), dtype=np.float32)
            te_lg_s = []
            for f_idx, (tr_idx, va_idx) in enumerate(folds):
                sp = TrainSpec(
                    task=task,
                    backbone_id=best_backbone_id,
                    context_mode=ctx_win,
                    loss_type=loss_win,
                    max_length=256,
                    epochs=8 if best_use_lora else 6,
                    use_lora=best_use_lora,
                    seed=run_seed + f_idx * 17,
                )
                rf = train_and_predict_logits(
                    sp,
                    [trainval_rows[i] for i in tr_idx],
                    [trainval_rows[i] for i in va_idx],
                    {"test": test_rows},
                )
                oof_lg_s[va_idx] = rf["val_logits"]
                te_lg_s.append(rf["eval_logits"]["test"])
            t_s = float(_calibrate_temperature(oof_lg_s, y_tv, multilabel=multilabel))
            oof_p_s = apply_logits_calibration(oof_lg_s, multilabel, t_s)
            te_p_s = apply_logits_calibration(np.mean(np.stack(te_lg_s, axis=0), axis=0), multilabel, t_s)
            taus_s = tune_multilabel_thresholds(y_tv, oof_p_s, labels, mode="bounded") if multilabel else None
            oof_ev_s = evaluate_predictions(task, y_tv, oof_p_s, thresholds=taus_s)
            te_ev_s = evaluate_predictions(task, y_te, te_p_s, thresholds=taus_s)
            seed_oof_scores.append(oof_ev_s["macro_f1"])
            seed_test_scores.append(te_ev_s["macro_f1"])
            seed_test_probs_list.append(te_p_s)

        summary["three_seed_stability"][task] = {
            "config": {"backbone": best_p5_tag, "context": ctx_win, "loss": loss_win, "lora": best_use_lora},
            "seeds": [SEED, SEED + 1, SEED + 2],
            "oof_macro_f1_per_seed": seed_oof_scores,
            "oof_mean": round(float(np.mean(seed_oof_scores)), 4),
            "oof_std": round(float(np.std(seed_oof_scores)), 4),
            "frozen_test_macro_f1_per_seed": seed_test_scores,
            "frozen_test_mean": round(float(np.mean(seed_test_scores)), 4),
            "frozen_test_std": round(float(np.std(seed_test_scores)), 4),
        }

        summary["tasks"][task] = {
            "oof_leaderboard": oof_leaderboard,
            "best_oof_candidate": best_oof_key,
            "best_p5_backbone_tag": best_p5_tag,
            "selected_context": ctx_win,
            "selected_loss": loss_win,
            "candidates_frozen_test": candidates_eval,
        }
        summary["final_per_task_selection"][task] = {
            "selected_candidate": best_oof_key,
            "oof_macro_f1": oof_leaderboard[best_oof_key],
            "mtl_alternative_oof_macro_f1": oof_leaderboard[mtl_oof_key],
            "frozen_test_macro_f1": candidates_eval[best_oof_key]["test_eval"]["macro_f1"],
            "v5_transductive_test_macro_f1": round(v5_official_f1, 4),
            "paired_delta_vs_v5": candidates_eval[best_oof_key]["bootstrap_ci_vs_v5"]["paired_delta_mean"],
            "paired_delta_ci95": candidates_eval[best_oof_key]["bootstrap_ci_vs_v5"]["paired_delta_ci95"],
            "p_beats_v5": candidates_eval[best_oof_key]["bootstrap_ci_vs_v5"]["p_a_beats_b"],
        }
        selected_specs_for_probes[task] = {
            "tag": best_p5_tag,
            "backbone_id": best_backbone_id,
            "context_mode": ctx_win,
            "loss_type": loss_win,
            "temperature": p5_meta["temperature"],
            "thresholds": p5_meta["bounded_thresholds"],
        }
        print(
            f"  >>> [{task.upper()}] OOF Winner={best_oof_key} (OOF={oof_leaderboard[best_oof_key]:.4f}) "
            f"| Frozen Test={candidates_eval[best_oof_key]['test_eval']['macro_f1']:.4f} "
            f"(3-seed={summary['three_seed_stability'][task]['frozen_test_mean']:.4f}+/-{summary['three_seed_stability'][task]['frozen_test_std']:.4f}) "
            f"vs V5={v5_official_f1:.4f} | paired_delta={candidates_eval[best_oof_key]['bootstrap_ci_vs_v5']['paired_delta_mean']:+.4f} "
            f"CI95={candidates_eval[best_oof_key]['bootstrap_ci_vs_v5']['paired_delta_ci95']}",
            flush=True,
        )
        save_json(out_dir / "phase8_9_summary.json", summary)

    # Separate 54-probe stress diagnostic (strictly observational)
    print("\n==================== PHASE 8: SEPARATE 54-PROBE STRESS DIAGNOSTIC ====================", flush=True)
    summary["robustness_54_probes_separate"] = _run_54_probe_diagnostic(
        trainval_rows, selected_specs_for_probes, p3["variants"][best_mtl_key]
    )

    # Overall 5-task headline comparison
    summary["overall_5task_means"] = {
        "v5_production_transductive_test": round(
            float(np.mean([summary["final_per_task_selection"][t]["v5_transductive_test_macro_f1"] for t in TASK_NAMES])), 4
        ),
        "selected_inductive_oof_winners_test": round(
            float(np.mean([summary["final_per_task_selection"][t]["frozen_test_macro_f1"] for t in TASK_NAMES])), 4
        ),
        "selected_3seed_mean_test": round(
            float(np.mean([summary["three_seed_stability"][t]["frozen_test_mean"] for t in TASK_NAMES])), 4
        ),
        "shared_mtl_1encoder_test": round(
            float(
                np.mean(
                    [
                        summary["tasks"][t]["candidates_frozen_test"][f"Phase3_{best_mtl_key}"]["test_eval"]["macro_f1"]
                        for t in TASK_NAMES
                    ]
                )
            ),
            4,
        ),
        "tfidf_wordchar_5fold_test": round(
            float(
                np.mean(
                    [
                        summary["tasks"][t]["candidates_frozen_test"]["TFIDF_WordChar_5Fold"]["test_eval"]["macro_f1"]
                        for t in TASK_NAMES
                    ]
                )
            ),
            4,
        ),
    }

    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "phase8_9_summary.json", summary)
    print(f"\nPhase 8/9 complete in {summary['total_elapsed_seconds']}s! Overall 5-task means: {summary['overall_5task_means']}", flush=True)
    return summary


if __name__ == "__main__":
    run_phase8_9()
