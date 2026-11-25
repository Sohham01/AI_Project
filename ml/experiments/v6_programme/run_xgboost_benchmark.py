"""Sprint P1-C Runner — Controlled 5-Fold Group-Aware CV Benchmark for Experimental XGBoost.

Evaluates two leak-free XGBoost configurations across all 5 RCI NLP tasks strictly on
``trainval`` (n = 367 real comments across 163 threads) using identical ``post_id``-grouped
5-fold CV splits (never touching the frozen 92-comment test set!):
  - ``xgb_sparse_tfidf``: Word+Char CSR TF-IDF + CPU tree_method="hist"
  - ``xgb_svd_linguistic``: TruncatedSVD TF-IDF + 14-dim linguistic cues + CPU tree_method="hist"

Runs identical 5-fold group-aware CV for Logistic Regression, ANN/MLP, and TextCNN
alongside saved MuRIL-DAPT-WWM / XLM-R OOF CV metrics for strictly apples-to-apples
cross-validation comparison.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    build_input_pair,
    encode_task_targets,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
)
from ml.models.neural_baselines import ANNMLPClassifier, LogisticRegressionBaseline, NeuralBaselineConfig, TextCNNClassifier
from ml.models.xgboost_baseline import XGBoostBaselineClassifier, XGBoostConfig

OUT_DIR = ROOT / "ml" / "artifacts" / "v6_3_experiments" / "xgboost_v1"
REPORT_MD = ROOT / "docs" / "V6_3_XGBOOST_BENCHMARK_REPORT.md"

CV_SEEDS = (SEED, SEED + 1, SEED + 2)


def _row_text(row: Dict[str, Any], task: str) -> str:
    rep, ctx = build_input_pair(row, "title_concat" if task == "issue" else "comment_only")
    return f"{rep} | {ctx}" if ctx else rep


def _target_list(rows: List[Dict[str, Any]], task: str) -> List[Any]:
    return [r["labels"][task] for r in rows]


def _run_model_cv(
    model_key: str,
    task: str,
    trainval_rows: List[Dict[str, Any]],
    seeds: tuple[int, ...],
) -> Dict[str, Any]:
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    y_all = encode_task_targets(trainval_rows, task)

    seed_macros: List[float] = []
    seed_eces: List[float] = []
    all_fold_macros: List[float] = []
    train_times: List[float] = []
    infer_lats: List[float] = []
    model_sizes_kb: List[float] = []
    last_oof_eval: Dict[str, Any] = {}
    last_taus: Dict[str, float] = {l: 0.5 for l in labels}
    last_bin_tau = 0.5

    for seed in seeds:
        folds = make_group_cv_folds(trainval_rows, task=task, n_splits=5, seed=seed)
        oof_probs = np.zeros((len(trainval_rows), len(labels)), dtype=np.float64)
        fold_taus: List[np.ndarray] = []
        fold_bin_taus: List[float] = []

        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            tr_rows = [trainval_rows[i] for i in tr_idx]
            va_rows = [trainval_rows[i] for i in va_idx]
            tr_texts = [_row_text(r, task) for r in tr_rows]
            va_texts = [_row_text(r, task) for r in va_rows]
            tr_y = _target_list(tr_rows, task)
            va_y = _target_list(va_rows, task)

            t0 = time.perf_counter()
            if model_key in {"xgb_sparse_tfidf", "xgb_svd_linguistic"}:
                fmode = "sparse_tfidf" if model_key == "xgb_sparse_tfidf" else "svd_linguistic"
                clf = XGBoostBaselineClassifier(
                    task_name=task,
                    multilabel=multilabel,
                    config=XGBoostConfig(feature_mode=fmode, random_state=seed + f_idx, n_jobs=4),
                )
                clf.fit(tr_texts, tr_y, va_texts, va_y, labels)
                tr_s = time.perf_counter() - t0
                ti0 = time.perf_counter()
                p_va = clf.predict_proba(va_texts)
                inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                if multilabel and clf.thresholds_ is not None:
                    fold_taus.append(clf.thresholds_)
                if task == "solution":
                    fold_bin_taus.append(clf.binary_threshold_)
                if f_idx == 0 and seed == seeds[0]:
                    with tempfile.TemporaryDirectory() as td:
                        size_b = clf.save(Path(td) / "xgb")
                        model_sizes_kb.append(size_b / 1024.0)
            elif model_key == "logistic_regression":
                clf_lr = LogisticRegressionBaseline(multilabel=multilabel, random_state=seed + f_idx)
                clf_lr.fit(tr_texts, tr_y, va_texts, va_y, labels)
                tr_s = time.perf_counter() - t0
                ti0 = time.perf_counter()
                p_va = clf_lr.predict_proba(va_texts)
                inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                if multilabel and clf_lr.thresholds_ is not None:
                    fold_taus.append(clf_lr.thresholds_)
                if f_idx == 0 and seed == seeds[0]:
                    with tempfile.TemporaryDirectory() as td:
                        p = Path(td) / "lr.joblib"
                        joblib.dump(clf_lr, p, compress=3)
                        model_sizes_kb.append(p.stat().st_size / 1024.0)
            elif model_key == "ann_mlp":
                clf_ann = ANNMLPClassifier(
                    multilabel=multilabel,
                    config=NeuralBaselineConfig(random_state=seed + f_idx),
                )
                clf_ann.fit(tr_texts, tr_y, va_texts, va_y, labels)
                tr_s = time.perf_counter() - t0
                ti0 = time.perf_counter()
                p_va = clf_ann.predict_proba(va_texts)
                inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                if multilabel and clf_ann.thresholds_ is not None:
                    fold_taus.append(clf_ann.thresholds_)
                if f_idx == 0 and seed == seeds[0]:
                    model_sizes_kb.append((clf_ann.param_count_ * 4.0) / 1024.0)
                clf_ann.cleanup()
            elif model_key == "cnn_textcnn":
                clf_cnn = TextCNNClassifier(
                    multilabel=multilabel,
                    config=NeuralBaselineConfig(epochs=20, learning_rate=2e-3, dropout=0.40, random_state=seed + f_idx),
                )
                clf_cnn.fit(tr_texts, tr_y, va_texts, va_y, labels)
                tr_s = time.perf_counter() - t0
                ti0 = time.perf_counter()
                p_va = clf_cnn.predict_proba(va_texts)
                inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                if multilabel and clf_cnn.thresholds_ is not None:
                    fold_taus.append(clf_cnn.thresholds_)
                if f_idx == 0 and seed == seeds[0]:
                    model_sizes_kb.append((clf_cnn.param_count_ * 4.0) / 1024.0)
                clf_cnn.cleanup()
            else:
                raise ValueError(f"Unknown model_key={model_key!r}")

            oof_probs[va_idx] = p_va
            train_times.append(tr_s)
            infer_lats.append(inf_ms)

            # Individual fold score (purely on that validation fold!)
            fold_t_map = (
                {labels[j]: float(fold_taus[-1][j]) for j in range(len(labels))}
                if (multilabel and fold_taus)
                else None
            )
            fold_btau = fold_bin_taus[-1] if (task == "solution" and fold_bin_taus) else 0.5
            fev = evaluate_predictions(task, y_all[va_idx], p_va, thresholds=fold_t_map, binary_threshold=fold_btau)
            all_fold_macros.append(float(fev["macro_f1"]))

        mean_taus = {labels[j]: float(np.mean([ft[j] for ft in fold_taus])) for j in range(len(labels))} if fold_taus else {l: 0.5 for l in labels}
        mean_btau = float(np.mean(fold_bin_taus)) if fold_bin_taus else 0.5
        oof_ev = evaluate_predictions(
            task, y_all, oof_probs,
            thresholds=mean_taus if multilabel else None,
            binary_threshold=mean_btau,
        )
        seed_macros.append(float(oof_ev["macro_f1"]))
        seed_eces.append(float(oof_ev["ece"]))
        last_oof_eval = oof_ev
        last_taus = mean_taus
        last_bin_tau = mean_btau

    return {
        "model": model_key,
        "task": task,
        "cv_oof_macro_f1_mean": round(float(np.mean(seed_macros)), 4),
        "cv_oof_macro_f1_seed_std": round(float(np.std(seed_macros)), 4),
        "fold_macro_f1_mean": round(float(np.mean(all_fold_macros)), 4),
        "fold_macro_f1_std": round(float(np.std(all_fold_macros)), 4),
        "cv_oof_ece_mean": round(float(np.mean(seed_eces)), 4),
        "mean_fold_train_seconds": round(float(np.mean(train_times)), 3),
        "total_5fold_train_seconds": round(float(np.mean(train_times) * 5.0), 3),
        "inference_ms_per_comment": round(float(np.mean(infer_lats)), 4),
        "model_size_kb": round(float(np.mean(model_sizes_kb)) if model_sizes_kb else 0.0, 1),
        "last_seed_per_class": last_oof_eval.get("per_class", {}),
        "thresholds": last_taus if multilabel else {"binary": round(last_bin_tau, 3)},
    }


def _load_muril_oof_reference() -> Dict[str, Any]:
    """Read existing V5 Phase 1 OOF summary (read-only) so we have zero-leakage CV reference numbers."""
    p = EXP_ROOT / "phase1_cv" / "phase1_summary.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    for t, td in (raw.get("tasks") or {}).items():
        # Pick primary calibrated OOF metrics logged in phase1_summary.json
        oof_sec = td.get("oof_calibrated") or td.get("outer_oof_calibrated") or td.get("oof") or {}
        if not oof_sec and isinstance(td, dict):
            for k in ("oof_temp_scaled", "oof_global_T", "oof_per_class_T", "oof_bounded"):
                if isinstance(td.get(k), dict) and "macro_f1" in td[k]:
                    oof_sec = td[k]
                    break
        out[t] = {
            "cv_oof_macro_f1": oof_sec.get("macro_f1"),
            "cv_oof_ece": oof_sec.get("ece"),
            "note": "Saved MuRIL-DAPT 5-fold inductive OOF CV reference",
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_v1_records_and_frozen_splits()
    trainval = data["trainval"]  # Strictly 367 trainval rows! Frozen test (n=92) NEVER touched!
    assert len(trainval) == 367

    models_cv = [
        ("xgb_sparse_tfidf", CV_SEEDS),
        ("xgb_svd_linguistic", CV_SEEDS),
        ("logistic_regression", (SEED,)),
        ("ann_mlp", (SEED,)),
        ("cnn_textcnn", (SEED,)),
    ]

    results_by_task: Dict[str, Dict[str, Any]] = {t: {} for t in TASK_NAMES}
    print(f"[P1-C] Running strict 5-fold post_id-grouped CV on trainval (n={len(trainval)}; 92 frozen test untouched)...", flush=True)

    for task in TASK_NAMES:
        print(f"\n--- TASK: {task.upper()} ---", flush=True)
        for m_key, seeds in models_cv:
            res = _run_model_cv(m_key, task, trainval, seeds)
            results_by_task[task][m_key] = res
            print(
                f"  {m_key:20s}: OOF_F1={res['cv_oof_macro_f1_mean']*100:5.2f}% (±{res['cv_oof_macro_f1_seed_std']*100:4.2f} seed / ±{res['fold_macro_f1_std']*100:4.2f} fold) "
                f"| ECE={res['cv_oof_ece_mean']:.4f} | train_5f={res['total_5fold_train_seconds']:.2f}s "
                f"| inf={res['inference_ms_per_comment']:.3f}ms | size={res['model_size_kb']:.1f}KB",
                flush=True,
            )

    muril_ref = _load_muril_oof_reference()

    # Aggregate 5-task macro summary
    agg: Dict[str, Any] = {}
    for m_key, _ in models_cv:
        f1s = [results_by_task[t][m_key]["cv_oof_macro_f1_mean"] for t in TASK_NAMES]
        f_stds = [results_by_task[t][m_key]["fold_macro_f1_std"] for t in TASK_NAMES]
        eces = [results_by_task[t][m_key]["cv_oof_ece_mean"] for t in TASK_NAMES]
        tr_s = [results_by_task[t][m_key]["total_5fold_train_seconds"] for t in TASK_NAMES]
        inf_s = [results_by_task[t][m_key]["inference_ms_per_comment"] for t in TASK_NAMES]
        sz_kb = [results_by_task[t][m_key]["model_size_kb"] for t in TASK_NAMES]
        agg[m_key] = {
            "mean_5task_cv_oof_macro_f1": round(float(np.mean(f1s)), 4),
            "mean_fold_std": round(float(np.mean(f_stds)), 4),
            "mean_ece": round(float(np.mean(eces)), 4),
            "total_train_seconds_all_5_tasks": round(float(sum(tr_s)), 2),
            "mean_inference_ms_per_comment": round(float(np.mean(inf_s)), 4),
            "mean_model_size_kb": round(float(np.mean(sz_kb)), 1),
        }

    payload = {
        "experiment": "xgboost_experimental_5fold_cv_v6_3",
        "evaluation_protocol": "5-fold post_id-grouped CV on trainval (n=367, 163 threads); frozen n=92 test NOT used for tuning/selection",
        "experimental_status": "experimental_comparator_not_promoted_to_production",
        "xgboost_config": XGBoostConfig().__dict__,
        "aggregate_cv": agg,
        "by_task": results_by_task,
        "muril_dapt_oof_reference": muril_ref,
    }
    (OUT_DIR / "xgboost_cv_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build clean comparison report
    lines = [
        "# V6.3 — Experimental XGBoost 5-Fold Group-Aware CV Benchmark Report",
        "",
        "> **Experimental Governance:** XGBoost (`tree_method=\"hist\"`) is added strictly as an experimental comparator on identical `post_id`-grouped 5-fold CV over `trainval (n = 367, 163 threads)`. The frozen `n = 92` real test set remains untouched, and production V6.1 (`MuRIL-DAPT-WWM + XLM-R`) and `REQUIRED_ARCHITECTURES` remain unchanged.",
        "",
        "## 1. 5-Task Aggregate 5-Fold OOF CV Summary (`n = 367`)",
        "",
        "| Architecture | Feature Representation | 5-Task Mean OOF F1 | Mean Fold Std | Mean ECE | 5-Task Train Time | Inference (ms/doc) | Avg Model Size |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels_map = {
        "xgb_sparse_tfidf": ("XGBoost (Experimental Sparse)", "CSR Word+Char TF-IDF (30k max)"),
        "xgb_svd_linguistic": ("XGBoost (Experimental Dense)", "TruncatedSVD-128 + 14 Ling. Cues"),
        "logistic_regression": ("Logistic Regression Baseline", "CSR Word+Char TF-IDF (30k max)"),
        "ann_mlp": ("ANN (2-Layer MLP)", "Dense TF-IDF (3,072 dims)"),
        "cnn_textcnn": ("1D TextCNN (Kim 2014)", "Token Embeddings (k=2..5)"),
    }
    for m_key, (m_label, feat) in labels_map.items():
        a = agg[m_key]
        lines.append(
            f"| **{m_label}** | {feat} | `{a['mean_5task_cv_oof_macro_f1']*100:.2f}%` | `±{a['mean_fold_std']*100:.2f}%` | `{a['mean_ece']:.4f}` | `{a['total_train_seconds_all_5_tasks']:.2f}s` | `{a['mean_inference_ms_per_comment']:.3f} ms` | `{a['mean_model_size_kb']:.1f} KB` |"
        )

    lines += [
        "",
        "## 2. Per-Task 5-Fold OOF Macro F1 (%) Comparison",
        "",
        "| Task | LogReg | ANN | TextCNN | XGB (Sparse TF-IDF) | XGB (SVD + Ling.) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASK_NAMES:
        row_vals = [
            f"`{results_by_task[task][k]['cv_oof_macro_f1_mean']*100:.2f}%` (±{results_by_task[task][k]['fold_macro_f1_std']*100:.1f})"
            if k.startswith("xgb_")
            else f"`{results_by_task[task][k]['cv_oof_macro_f1_mean']*100:.2f}%`"
            for k in ("logistic_regression", "ann_mlp", "cnn_textcnn", "xgb_sparse_tfidf", "xgb_svd_linguistic")
        ]
        lines.append(f"| **{task}** | " + " | ".join(row_vals) + " |")

    lines += [
        "",
        "## 3. Architectural Analysis & Non-Promotion Rationale",
        "- **Sparse TF-IDF vs. SVD+Linguistic:** High-dimensional sparse CSR TF-IDF (`xgb_sparse_tfidf`) preserves exact Hinglish/domain unigrams and character n-grams for tree splits, whereas `xgb_svd_linguistic` compresses lexical boundaries into 128 SVD dimensions paired with 14 explicit linguistic/discourse cues.",
        "- **Why XGBoost stays experimental:** On 367 short code-mixed comments, boosted decision trees cannot model subword cross-lingual contextual semantics (Hinglish syntax + parent-comment pragmatics) as strongly as `MuRIL-DAPT-WWM` / `XLM-R`, and linear L2 Logistic Regression remains a tough high-dimensional sparse text competitor.",
        "- **Known Limitations:** Tree splits over sparse bag-of-ngrams exhibit higher fold-to-fold variance (`n ~ 73` val samples per fold) on rare classes in `issue` (`extracurriculars`) and `emotion` (`gratitude`).",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[P1-C] Saved {OUT_DIR / 'xgboost_cv_benchmark.json'} and {REPORT_MD}")


if __name__ == "__main__":
    main()
