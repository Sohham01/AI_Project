"""V6.5 Benchmark Runner — Controlled 7-Fold (6:1 Train:Val Ratio) Group-Aware CV.

Evaluates all 7 non-Transformer / Deep / Boosted model families under a strict
**6:1 per-fold train-to-validation ratio** (`n_splits = 7` `post_id`-grouped folds ->
6/7 = 85.71% train vs 1/7 = 14.29% validation) on `trainval` (n = 367 real comments
across 163 threads) across 3 seeds (`SEED, SEED+1, SEED+2`), NEVER touching or tuning on
the frozen 92-comment / 37-thread V1 test split:
  - `logistic_regression` (scikit-learn CSR Word+Char TF-IDF)
  - `xgb_sparse_tfidf` (XGBoost CSR Word+Char TF-IDF, CPU hist)
  - `xgb_svd_linguistic` (XGBoost SVD-128 + 14 linguistic cues, CPU hist)
  - `ann_mlp` (PyTorch 2-Layer Dense TF-IDF MLP)
  - `cnn_textcnn` (PyTorch 1D TextCNN)
  - `pytorch_bigru_attention` (PyTorch BiGRU + Masked Additive Attention + MaxPool)
  - `keras_multitask_torch` (Keras 3 Functional 5-Head Multi-Task, `KERAS_BACKEND="torch"`)

Also compares head-to-head against V6.4 5-fold (4:1 per-fold) OOF CV baselines saved in
`ml/artifacts/v6_4_experiments/pytorch_keras_v1/pytorch_keras_cv_benchmark.json`,
writing outputs strictly to `ml/artifacts/v6_5_experiments/cv_6_1_benchmark/` and
`docs/V6_5_6_1_CORPUS_AND_BENCHMARK_REPORT.md`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    build_input_pair,
    encode_task_targets,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
)
from ml.models.keras_multitask import KerasMultiTaskClassifier, KerasMultiTaskConfig
from ml.models.neural_baselines import (
    ANNMLPClassifier,
    LogisticRegressionBaseline,
    NeuralBaselineConfig,
    TextCNNClassifier,
)
from ml.models.pytorch_bigru_attention import BiGRUAttentionClassifier, BiGRUAttentionConfig
from ml.models.xgboost_baseline import XGBoostBaselineClassifier, XGBoostConfig

OUT_DIR = ROOT / "ml" / "artifacts" / "v6_5_experiments" / "cv_6_1_benchmark"
PRIOR_5F_PATH = ROOT / "ml" / "artifacts" / "v6_4_experiments" / "pytorch_keras_v1" / "pytorch_keras_cv_benchmark.json"
CORPUS_SPLIT_PATH = ROOT / "data" / "corpus_v6_5" / "corpus_split_6_1_manifest.json"
CORPUS_V65_MANIFEST = ROOT / "data" / "corpus_v6_5" / "combined_corpus_v6_5_manifest.json"
BLIND_185_MANIFEST = ROOT / "data" / "annotations" / "v6_5_package" / "blind_185_to_6_1" / "blind_185_manifest.json"
REPORT_MD = ROOT / "docs" / "V6_5_6_1_CORPUS_AND_BENCHMARK_REPORT.md"

N_SPLITS = 7  # 6 train folds : 1 val fold == exact 6:1 train/val fold ratio
CV_SEEDS = (SEED, SEED + 1, SEED + 2)

SINGLE_TASK_FAMILIES = (
    "logistic_regression",
    "xgb_sparse_tfidf",
    "xgb_svd_linguistic",
    "ann_mlp",
    "cnn_textcnn",
    "pytorch_bigru_attention",
)


def _row_text(row: Dict[str, Any], task: str) -> str:
    rep, ctx = build_input_pair(row, "title_concat" if task == "issue" else "comment_only")
    return f"{rep} | {ctx}" if ctx else rep


def _run_single_task_7fold_cv(
    model_key: str,
    trainval: List[Dict[str, Any]],
    seeds: tuple[int, ...],
) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for task in TASK_NAMES:
        labels = CANONICAL_LABELS[task]
        multilabel = task in MULTILABEL_TASKS
        y_all = encode_task_targets(trainval, task)

        seed_macros: List[float] = []
        seed_eces: List[float] = []
        fold_macros: List[float] = []
        train_times: List[float] = []
        infer_lats: List[float] = []
        model_sizes_kb: List[float] = []
        fold_ratios: List[float] = []
        last_eval: Dict[str, Any] = {}
        last_taus: Dict[str, float] = {l: 0.5 for l in labels}
        last_btau = 0.5
        param_cnt = 0

        for seed in seeds:
            folds = make_group_cv_folds(trainval, task=task, n_splits=N_SPLITS, seed=seed)
            oof_probs = np.zeros((len(trainval), len(labels)), dtype=np.float64)
            fold_taus: List[np.ndarray] = []
            fold_btaus: List[float] = []

            for f_idx, (tr_idx, va_idx) in enumerate(folds):
                fold_ratios.append(len(tr_idx) / max(len(va_idx), 1))
                tr_rows = [trainval[i] for i in tr_idx]
                va_rows = [trainval[i] for i in va_idx]
                tr_texts = [_row_text(r, task) for r in tr_rows]
                va_texts = [_row_text(r, task) for r in va_rows]
                tr_y = [r["labels"][task] for r in tr_rows]
                va_y = [r["labels"][task] for r in va_rows]

                t0 = time.perf_counter()
                if model_key in {"xgb_sparse_tfidf", "xgb_svd_linguistic"}:
                    fmode = "sparse_tfidf" if model_key == "xgb_sparse_tfidf" else "svd_linguistic"
                    clf: Any = XGBoostBaselineClassifier(
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
                        fold_btaus.append(clf.binary_threshold_)
                    if f_idx == 0 and seed == seeds[0]:
                        with tempfile.TemporaryDirectory() as td:
                            model_sizes_kb.append(clf.save(Path(td) / "xgb") / 1024.0)
                elif model_key == "logistic_regression":
                    clf = LogisticRegressionBaseline(multilabel=multilabel, random_state=seed + f_idx)
                    clf.fit(tr_texts, tr_y, va_texts, va_y, labels)
                    tr_s = time.perf_counter() - t0
                    ti0 = time.perf_counter()
                    p_va = clf.predict_proba(va_texts)
                    inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                    if multilabel and clf.thresholds_ is not None:
                        fold_taus.append(clf.thresholds_)
                    if f_idx == 0 and seed == seeds[0]:
                        with tempfile.TemporaryDirectory() as td:
                            p = Path(td) / "lr.joblib"
                            joblib.dump(clf, p, compress=3)
                            model_sizes_kb.append(p.stat().st_size / 1024.0)
                elif model_key == "ann_mlp":
                    clf = ANNMLPClassifier(
                        multilabel=multilabel,
                        config=NeuralBaselineConfig(random_state=seed + f_idx),
                    )
                    clf.fit(tr_texts, tr_y, va_texts, va_y, labels)
                    tr_s = time.perf_counter() - t0
                    ti0 = time.perf_counter()
                    p_va = clf.predict_proba(va_texts)
                    inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                    if multilabel and clf.thresholds_ is not None:
                        fold_taus.append(clf.thresholds_)
                    if f_idx == 0 and seed == seeds[0]:
                        param_cnt = clf.param_count_
                        model_sizes_kb.append((clf.param_count_ * 4.0) / 1024.0)
                    clf.cleanup()
                elif model_key == "cnn_textcnn":
                    clf = TextCNNClassifier(
                        multilabel=multilabel,
                        config=NeuralBaselineConfig(epochs=20, learning_rate=2e-3, dropout=0.40, random_state=seed + f_idx),
                    )
                    clf.fit(tr_texts, tr_y, va_texts, va_y, labels)
                    tr_s = time.perf_counter() - t0
                    ti0 = time.perf_counter()
                    p_va = clf.predict_proba(va_texts)
                    inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                    if multilabel and clf.thresholds_ is not None:
                        fold_taus.append(clf.thresholds_)
                    if f_idx == 0 and seed == seeds[0]:
                        param_cnt = clf.param_count_
                        model_sizes_kb.append((clf.param_count_ * 4.0) / 1024.0)
                    clf.cleanup()
                elif model_key == "pytorch_bigru_attention":
                    clf = BiGRUAttentionClassifier(
                        task_name=task,
                        multilabel=multilabel,
                        config=BiGRUAttentionConfig(random_state=seed + f_idx),
                    )
                    clf.fit(tr_texts, tr_y, va_texts, va_y, labels)
                    tr_s = time.perf_counter() - t0
                    ti0 = time.perf_counter()
                    p_va = clf.predict_proba(va_texts)
                    inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)
                    if multilabel and clf.thresholds_ is not None:
                        fold_taus.append(clf.thresholds_)
                    if task == "solution":
                        fold_btaus.append(clf.binary_threshold_)
                    if f_idx == 0 and seed == seeds[0]:
                        param_cnt = clf.param_count_
                        with tempfile.TemporaryDirectory() as td:
                            model_sizes_kb.append(clf.save(Path(td) / "bigru") / 1024.0)
                    clf.cleanup()
                else:
                    raise ValueError(f"Unknown model_key={model_key!r}")

                fold_t_map = (
                    {labels[j]: float(fold_taus[-1][j]) for j in range(len(labels))}
                    if (multilabel and fold_taus)
                    else None
                )
                fold_bt = fold_btaus[-1] if (task == "solution" and fold_btaus) else 0.5
                fev = evaluate_predictions(task, y_all[va_idx], p_va, thresholds=fold_t_map, binary_threshold=fold_bt)
                fold_macros.append(float(fev["macro_f1"]))
                oof_probs[va_idx] = p_va
                train_times.append(tr_s)
                infer_lats.append(inf_ms)

            mean_taus = (
                {labels[j]: float(np.mean([ft[j] for ft in fold_taus])) for j in range(len(labels))}
                if fold_taus
                else {l: 0.5 for l in labels}
            )
            mean_btau = float(np.mean(fold_btaus)) if fold_btaus else 0.5
            oof_ev = evaluate_predictions(
                task,
                y_all,
                oof_probs,
                thresholds=mean_taus if multilabel else None,
                binary_threshold=mean_btau,
            )
            seed_macros.append(float(oof_ev["macro_f1"]))
            seed_eces.append(float(oof_ev["ece"]))
            last_eval = oof_ev
            last_taus = mean_taus
            last_btau = mean_btau

        by_task[task] = {
            "model": model_key,
            "task": task,
            "n_splits": N_SPLITS,
            "mean_train_to_val_ratio": round(float(np.mean(fold_ratios)), 3),
            "cv_oof_macro_f1_mean": round(float(np.mean(seed_macros)), 4),
            "cv_oof_macro_f1_seed_std": round(float(np.std(seed_macros)), 4),
            "fold_macro_f1_mean": round(float(np.mean(fold_macros)), 4),
            "fold_macro_f1_std": round(float(np.std(fold_macros)), 4),
            "cv_oof_ece_mean": round(float(np.mean(seed_eces)), 4),
            "mean_fold_train_seconds": round(float(np.mean(train_times)), 3),
            "total_7fold_train_seconds": round(float(np.mean(train_times) * N_SPLITS), 3),
            "inference_ms_per_comment": round(float(np.mean(infer_lats)), 4),
            "model_size_kb": round(float(np.mean(model_sizes_kb)) if model_sizes_kb else 0.0, 1),
            "param_count": param_cnt,
            "last_seed_per_class": last_eval.get("per_class", {}),
            "thresholds": last_taus if multilabel else {"binary": round(last_btau, 3)},
        }
        print(
            f"  [{model_key:23s}] {task:10s}: 7F_OOF_F1={by_task[task]['cv_oof_macro_f1_mean']*100:5.2f}% "
            f"(±{by_task[task]['cv_oof_macro_f1_seed_std']*100:4.2f} seed) | ECE={by_task[task]['cv_oof_ece_mean']:.4f}",
            flush=True,
        )
    return by_task


def _run_keras_multitask_7fold_cv(trainval: List[Dict[str, Any]], seeds: tuple[int, ...]) -> Dict[str, Dict[str, Any]]:
    y_all_by_task = {t: encode_task_targets(trainval, t) for t in TASK_NAMES}
    seed_macros: Dict[str, List[float]] = {t: [] for t in TASK_NAMES}
    seed_eces: Dict[str, List[float]] = {t: [] for t in TASK_NAMES}
    fold_macros: Dict[str, List[float]] = {t: [] for t in TASK_NAMES}
    fold_train_secs: List[float] = []
    fold_infer_ms: List[float] = []
    model_sizes_kb: List[float] = []
    fold_ratios: List[float] = []
    last_eval: Dict[str, Any] = {}
    last_taus: Dict[str, Any] = {}
    param_cnt = 0

    for seed in seeds:
        folds = make_group_cv_folds(trainval, task="response", n_splits=N_SPLITS, seed=seed)
        oof_probs = {t: np.zeros((len(trainval), len(CANONICAL_LABELS[t])), dtype=np.float64) for t in TASK_NAMES}
        fold_taus: Dict[str, List[np.ndarray]] = {"issue": [], "emotion": []}
        fold_btaus: List[float] = []

        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            fold_ratios.append(len(tr_idx) / max(len(va_idx), 1))
            tr_rows = [trainval[i] for i in tr_idx]
            va_rows = [trainval[i] for i in va_idx]
            tr_texts = [_row_text(r, "issue") for r in tr_rows]
            va_texts = [_row_text(r, "issue") for r in va_rows]
            tr_y_map = {t: [r["labels"][t] for r in tr_rows] for t in TASK_NAMES}
            va_y_map = {t: [r["labels"][t] for r in va_rows] for t in TASK_NAMES}

            t0 = time.perf_counter()
            clf = KerasMultiTaskClassifier(config=KerasMultiTaskConfig(random_state=seed + f_idx))
            clf.fit(tr_texts, tr_y_map, va_texts, va_y_map, CANONICAL_LABELS)
            tr_s = time.perf_counter() - t0
            ti0 = time.perf_counter()
            p_va_map = clf.predict_proba(va_texts)
            inf_ms = ((time.perf_counter() - ti0) * 1000.0) / max(len(va_texts), 1)

            fold_train_secs.append(tr_s)
            fold_infer_ms.append(inf_ms)
            if f_idx == 0 and seed == seeds[0]:
                param_cnt = clf.param_count_
                with tempfile.TemporaryDirectory() as td:
                    model_sizes_kb.append(clf.save(Path(td) / "keras_mt") / 1024.0)

            fold_taus["issue"].append(clf.thresholds_["issue"])
            fold_taus["emotion"].append(clf.thresholds_["emotion"])
            fold_btaus.append(float(clf.thresholds_["solution"]))

            for t in TASK_NAMES:
                oof_probs[t][va_idx] = p_va_map[t]
                labels = CANONICAL_LABELS[t]
                t_map = (
                    {labels[j]: float(fold_taus[t][-1][j]) for j in range(len(labels))}
                    if t in MULTILABEL_TASKS
                    else None
                )
                bt = fold_btaus[-1] if t == "solution" else 0.5
                fev = evaluate_predictions(t, y_all_by_task[t][va_idx], p_va_map[t], thresholds=t_map, binary_threshold=bt)
                fold_macros[t].append(float(fev["macro_f1"]))

        for t in TASK_NAMES:
            labels = CANONICAL_LABELS[t]
            multilabel = t in MULTILABEL_TASKS
            mean_taus = (
                {labels[j]: float(np.mean([ft[j] for ft in fold_taus[t]])) for j in range(len(labels))}
                if multilabel
                else {l: 0.5 for l in labels}
            )
            mean_btau = float(np.mean(fold_btaus)) if t == "solution" else 0.5
            oof_ev = evaluate_predictions(
                t,
                y_all_by_task[t],
                oof_probs[t],
                thresholds=mean_taus if multilabel else None,
                binary_threshold=mean_btau,
            )
            seed_macros[t].append(float(oof_ev["macro_f1"]))
            seed_eces[t].append(float(oof_ev["ece"]))
            last_eval[t] = oof_ev
            last_taus[t] = mean_taus if multilabel else {"binary": round(mean_btau, 3)}

    by_task: Dict[str, Dict[str, Any]] = {}
    mean_tr_7f = round(float(np.mean(fold_train_secs) * N_SPLITS), 3)
    mean_inf = round(float(np.mean(fold_infer_ms)), 4)
    sz_kb = round(float(np.mean(model_sizes_kb)) if model_sizes_kb else 0.0, 1)
    mean_ratio = round(float(np.mean(fold_ratios)), 3)

    for t in TASK_NAMES:
        by_task[t] = {
            "model": "keras_multitask_torch",
            "task": t,
            "n_splits": N_SPLITS,
            "mean_train_to_val_ratio": mean_ratio,
            "cv_oof_macro_f1_mean": round(float(np.mean(seed_macros[t])), 4),
            "cv_oof_macro_f1_seed_std": round(float(np.std(seed_macros[t])), 4),
            "fold_macro_f1_mean": round(float(np.mean(fold_macros[t])), 4),
            "fold_macro_f1_std": round(float(np.std(fold_macros[t])), 4),
            "cv_oof_ece_mean": round(float(np.mean(seed_eces[t])), 4),
            "mean_fold_train_seconds": round(float(np.mean(fold_train_secs)), 3),
            "total_7fold_train_seconds": mean_tr_7f,
            "inference_ms_per_comment": mean_inf,
            "model_size_kb": sz_kb,
            "param_count": param_cnt,
            "last_seed_per_class": last_eval[t].get("per_class", {}),
            "thresholds": last_taus[t],
        }
        print(
            f"  [keras_multitask_torch  ] {t:10s}: 7F_OOF_F1={by_task[t]['cv_oof_macro_f1_mean']*100:5.2f}% "
            f"(±{by_task[t]['cv_oof_macro_f1_seed_std']*100:4.2f} seed) | ECE={by_task[t]['cv_oof_ece_mean']:.4f}",
            flush=True,
        )
    return by_task


def _agg(task_map: Dict[str, Dict[str, Any]], *, joint_train: bool = False) -> Dict[str, Any]:
    return {
        "n_splits": N_SPLITS,
        "mean_fold_train_to_val_ratio": task_map["issue"]["mean_train_to_val_ratio"],
        "mean_5task_cv_oof_macro_f1": round(float(np.mean([task_map[t]["cv_oof_macro_f1_mean"] for t in TASK_NAMES])), 4),
        "mean_fold_std": round(float(np.mean([task_map[t]["fold_macro_f1_std"] for t in TASK_NAMES])), 4),
        "mean_ece": round(float(np.mean([task_map[t]["cv_oof_ece_mean"] for t in TASK_NAMES])), 4),
        "total_train_seconds_all_5_tasks": round(
            float(task_map["issue"]["total_7fold_train_seconds"])
            if joint_train
            else float(sum(task_map[t]["total_7fold_train_seconds"] for t in TASK_NAMES)),
            2,
        ),
        "mean_inference_ms_per_comment": round(float(np.mean([task_map[t]["inference_ms_per_comment"] for t in TASK_NAMES])), 4),
        "mean_model_size_kb": round(float(np.mean([task_map[t]["model_size_kb"] for t in TASK_NAMES])), 1),
        "param_count": int(task_map["issue"].get("param_count", 0)),
    }


def write_v65_report(payload: Dict[str, Any]) -> None:
    """Write unified V6.5 Corpus Expansion, 6:1 Split, and 7-Fold CV Benchmark Report."""
    exp = json.loads(CORPUS_V65_MANIFEST.read_text(encoding="utf-8")) if CORPUS_V65_MANIFEST.is_file() else {}
    spl = json.loads(CORPUS_SPLIT_PATH.read_text(encoding="utf-8")) if CORPUS_SPLIT_PATH.is_file() else {}
    b185 = json.loads(BLIND_185_MANIFEST.read_text(encoding="utf-8")) if BLIND_185_MANIFEST.is_file() else {}
    prior_5f = json.loads(PRIOR_5F_PATH.read_text(encoding="utf-8")) if PRIOR_5F_PATH.is_file() else {}

    bva = exp.get("before_vs_after", {})
    tr_p = spl.get("train_pool", {})
    te_p = spl.get("test_pool", {})
    agg7 = payload["aggregate_cv_7fold_6_to_1"]
    agg5 = prior_5f.get("aggregate_cv", {})

    lines = [
        "# V6.5 — Real Corpus Expansion, 6:1 Train/Test Protocol & 7-Fold (`6:1`) Benchmark Report",
        "",
        "> **Protocol Integrity Guard:** `data/annotations/real_reddit_labeled_v1/` (`367` trainval / `92` frozen test across `37` unseen threads), `data/live_corpus/corpus.json` (`v6.1`), `data/corpus_v6_3/new_unannotated_corpus_v6_3.json` (`v6.3`), and `data/corpus_v6_4/new_iim_corpus_v6_4.json` (`v6.4`) remain **100% immutable**. All V6.5 threads are strictly unannotated real discussions (`annotation_status = \"unannotated\"`), with zero frozen-test leakage and zero fabricated human labels.",
        "",
        "## 1. Real Corpus Growth (`v6.1.0` → `v6.3.0` → `v6.4.0` → `v6.5.0`)",
        "",
        "| Cohort | Threads | Human Comments | Notes |",
        "|---|---:|---:|---|",
        f"| **Baseline (`v6.1.0`)** | `{bva.get('baseline_v6_1', {}).get('threads', 365):,}` | `{bva.get('baseline_v6_1', {}).get('comments', 3557):,}` | Production corpus + `459` V1 labeled comments |",
        f"| **V6.3 Expansion (`v6.3.0`)** | `+{bva.get('v6_3_expansion', {}).get('threads', 451):,}` | `+{bva.get('v6_3_expansion', {}).get('comments', 4013):,}` | Broad MBA / regional higher-ed discussions |",
        f"| **V6.4 IIM Expansion (`v6.4.0`)** | `+{bva.get('v6_4_iim_expansion', {}).get('threads', 526):,}` | `+{bva.get('v6_4_iim_expansion', {}).get('comments', 9108):,}` | Dedicated IIM + IPMAT + IIMCalcutta communities |",
        f"| **V6.5 Campus/Pain-Point Expansion (`v6.5.0`)** | **`+{bva.get('v6_5_expansion', {}).get('threads', 0):,}`** | **`+{bva.get('v6_5_expansion', {}).get('comments', 0):,}`** | BLACKI/CAP/New-IIM waitlist, fees, hostel, placements |",
        f"| **Combined V6.5 Real Research Corpus** | **`{bva.get('combined_v6_5_total', {}).get('threads', 0):,}`** | **`{bva.get('combined_v6_5_total', {}).get('comments', 0):,}`** | `{exp.get('leakage_audit', {}).get('threads_quarantined_from_dapt_train', 0)}` leaked/quarantined threads |",
        "",
        "## 2. Two-Tier 6:1 Train/Test Protocol",
        "",
        "### 2.1 Combined Corpus Thread-Grouped `6:1` Split (`corpus_split_6_1_manifest.json`)",
        "",
        "| Partition | Threads | Thread Share | Comments | Comment Share | Pinned Guards |",
        "|---|---:|---:|---:|---:|---|",
        f"| **`train_pool`** | **`{tr_p.get('threads', 0):,}`** | `{tr_p.get('share_threads_pct', 0):.2f}%` | `{tr_p.get('comments', 0):,}` | `{tr_p.get('share_comments_pct', 0):.2f}%` | All `163` V1 trainval threads pinned |",
        f"| **`test_pool`** | **`{te_p.get('threads', 0):,}`** | `{te_p.get('share_threads_pct', 0):.2f}%` | `{te_p.get('comments', 0):,}` | `{te_p.get('share_comments_pct', 0):.2f}%` | All `37` frozen V1 test threads pinned (`0` overlap) |",
        f"| **Observed Ratio** | **`{spl.get('observed_ratios', {}).get('thread_train_to_test', 0):.4f} : 1`** | — | **`{spl.get('observed_ratios', {}).get('comment_train_to_test', 0):.4f} : 1`** | — | Target: `6.0000 : 1` (`85.71% / 14.29%`) |",
        "",
        "### 2.2 Supervised `+185` Blind Human Annotation Target (`552 : 92 = 6.00 : 1`)",
        "",
        f"- **Why `+185`:** Existing V1 labeled data stands at `367` trainval : `92` frozen test (`3.99 : 1`). Adding **`{b185.get('n_new_items', 185)}` real comments** (`{b185.get('unique_threads', 0)}` diverse `train_pool` threads; `0` frozen-test/test-pool overlap) raises supervised training data to `367 + 185 = 552` against the untouched `92` frozen test comments (**`{b185.get('resulting_train_to_frozen_test_ratio', 6.0):.2f} : 1`**).",
        "- **Blind Files (`pending_human_annotation`):** [`blind_185_v6_5_annotator_1.xlsx`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_v6_5_annotator_1.xlsx), [`blind_185_v6_5_annotator_2.xlsx`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_v6_5_annotator_2.xlsx), and [`blind_185_adjudication_template.csv`](../data/annotations/v6_5_package/blind_185_to_6_1/blind_185_adjudication_template.csv).",
        "",
        "## 3. 7-Fold (`6:1` Per-Fold) vs. 5-Fold (`4:1`) OOF Benchmark (`trainval`, `n = 367`)",
        "",
        "Switching from 5-fold (`4:1`, `80%` train / `20%` val) to **7-fold `post_id`-grouped CV (`6:1`, `85.7%` train / `14.3%` val)** increases per-fold training exposure without touching the frozen `n = 92` test set (`3` seeds).",
        "",
        "| Model Family | 7-Fold (`6:1`) Mean OOF F1 | 5-Fold (`4:1`) Mean OOF F1 | Δ (7F vs 5F) | 7-Fold ECE | 7-Fold Train Time | Latency / doc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    row_labels = {
        "ann_mlp": "**ANN (2-Layer MLP)**",
        "pytorch_bigru_attention": "**PyTorch BiGRU + Additive Attention**",
        "logistic_regression": "**Logistic Regression (Word+Char TF-IDF)**",
        "xgb_svd_linguistic": "**XGBoost (Dense SVD+Ling)**",
        "xgb_sparse_tfidf": "**XGBoost (Sparse TF-IDF)**",
        "keras_multitask_torch": "**Keras 3 Multi-Task (`KERAS_BACKEND='torch'`)**",
        "cnn_textcnn": "**1D TextCNN (Kim 2014)**",
    }
    for k in sorted(agg7.keys(), key=lambda x: -agg7[x]["mean_5task_cv_oof_macro_f1"]):
        a7 = agg7[k]
        a5 = agg5.get(k, {})
        f5 = a5.get("mean_5task_cv_oof_macro_f1")
        f5_str = f"`{f5*100:.2f}%`" if f5 is not None else "—"
        delta_str = f"`{(a7['mean_5task_cv_oof_macro_f1'] - f5)*100:+.2f} pp`" if f5 is not None else "—"
        lines.append(
            f"| {row_labels.get(k, k)} | **`{a7['mean_5task_cv_oof_macro_f1']*100:.2f}%`** | {f5_str} | {delta_str} | "
            f"`{a7['mean_ece']:.4f}` | `{a7['total_train_seconds_all_5_tasks']:.2f}s` | `{a7['mean_inference_ms_per_comment']:.3f} ms` |"
        )

    lines += [
        "",
        "### Per-Task 7-Fold (`6:1`) OOF Macro F1 (%)",
        "",
        "| Task | ANN (MLP) | PyTorch BiGRU-Attn | LogReg | XGB (SVD+Ling) | XGB (Sparse) | Keras 3 MT (Torch) | TextCNN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    bt7 = payload["by_task_7fold"]
    cols = ["ann_mlp", "pytorch_bigru_attention", "logistic_regression", "xgb_svd_linguistic", "xgb_sparse_tfidf", "keras_multitask_torch", "cnn_textcnn"]
    for t in TASK_NAMES:
        vals = " | ".join(f"`{bt7[c][t]['cv_oof_macro_f1_mean']*100:.2f}%`" for c in cols)
        lines.append(f"| **{t}** | {vals} |")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_v1_records_and_frozen_splits()
    trainval = data["trainval"]
    assert len(trainval) == 367

    print(
        f"[V6.5] Running 7-fold (6:1 per-fold) post_id-grouped CV across 7 model families "
        f"(trainval={len(trainval)}, 3 seeds; 92 frozen test untouched)...",
        flush=True,
    )
    by_model_tasks: Dict[str, Dict[str, Dict[str, Any]]] = {}
    agg: Dict[str, Dict[str, Any]] = {}

    for mkey in SINGLE_TASK_FAMILIES:
        print(f"-> Running {mkey}...", flush=True)
        res = _run_single_task_7fold_cv(mkey, trainval, CV_SEEDS)
        by_model_tasks[mkey] = res
        agg[mkey] = _agg(res, joint_train=False)

    print("-> Running keras_multitask_torch...", flush=True)
    keras_res = _run_keras_multitask_7fold_cv(trainval, CV_SEEDS)
    by_model_tasks["keras_multitask_torch"] = keras_res
    agg["keras_multitask_torch"] = _agg(keras_res, joint_train=True)

    payload = {
        "experiment": "v6_5_7fold_6_to_1_cv_benchmark",
        "keras_backend": "torch",
        "n_splits": N_SPLITS,
        "evaluation_protocol": (
            "7-fold post_id-grouped CV (6:1 train:val fold ratio = 85.71% / 14.29%) "
            "on V1 trainval (n=367, 163 threads) across 3 seeds; frozen n=92 test NOT touched"
        ),
        "experimental_status": "isolated_6_to_1_cv_benchmark",
        "aggregate_cv_7fold_6_to_1": agg,
        "by_task_7fold": by_model_tasks,
    }
    (OUT_DIR / "v65_6_1_cv_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_v65_report(payload)
    print(f"[V6.5] Saved {OUT_DIR / 'v65_6_1_cv_benchmark.json'} and {REPORT_MD}")


if __name__ == "__main__":
    main()
