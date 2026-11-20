"""Head-to-head leakage-free benchmark comparing:
  1. Baseline V1 (Word TF-IDF + Logistic Regression, fixed threshold 0.5)
  2. Improved Classical V2 (Word + Char-WB TF-IDF FeatureUnion + Balanced Weights
     + Validation-Only Per-Label Threshold Calibration)
  3. Fine-Tuned MuRIL Transformer (google/muril-base-cased, CUDA AMP, warmup/cosine,
     validation checkpointing, validation threshold calibration)
  4. Thread-Context Ablation for Issue Identification (comment + parent post title,
     strictly post_id-grouped with zero train/test thread overlap)

Evaluated on BOTH:
  - Clean held-out test split (n = 192, 24 unseen threads)
  - Perturbed noisy Reddit-style test split (typos, vowel drops, informal slang,
    mixed punctuation) to test morphological / code-mixed robustness honestly.

Outputs are isolated in ml/artifacts/model_upgrade_benchmark/ and never modify
ml/artifacts/demo_synthetic/, ml/artifacts/demo_synthetic_v2/, or ml/artifacts/registry.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for candidate in (str(ROOT), str(BACKEND)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.models.transformer.muril_classifier import MuRILTaskClassifier, MuRILTaskConfig
from ml.training.trainer import records_to_annotations
from app.services.nlp.issue_identification.base import IssueModelConfig
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)
from app.services.nlp.response_classification.base import ResponseModelConfig
from app.services.nlp.response_classification.tfidf_logistic_multiclass import (
    TfidfLogisticResponseClassifier,
)
from app.services.nlp.sentiment_emotion.base import (
    EmotionModelConfig,
    SentimentModelConfig,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
    TfidfLogisticEmotionClassifier,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment import (
    TfidfLogisticSentimentClassifier,
)
from app.services.nlp.solution_detection.base import SolutionModelConfig
from app.services.nlp.solution_detection.tfidf_logistic_binary import (
    TfidfLogisticSolutionClassifier,
)

DATA_DIR = ROOT / "data" / "demo_synthetic"
OUT_DIR = ROOT / "ml" / "artifacts" / "model_upgrade_benchmark"
SEED = 20261116
TASKS = ("issue", "emotion", "sentiment", "response", "solution")


def load_post_titles() -> Dict[str, str]:
    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    return {p["reddit_id"]: p["title"] for p in corpus.get("posts", [])}


def perturb_reddit_noise(text: str, rng: random.Random) -> str:
    """Apply deterministic Reddit-style typos, vowel drops, and informal noise.

    Simulates organic mobile typing / Hinglish spelling drift without changing
    the underlying semantic intent of the comment.
    """
    tokens = text.split()
    noisy: List[str] = []
    vowels = set("aeiouAEIOU")
    for tok in tokens:
        if len(tok) >= 5 and tok.isascii() and tok.isalpha() and rng.random() < 0.45:
            op = rng.choice(("drop_vowel", "swap", "double_char"))
            if op == "drop_vowel":
                inner = [i for i in range(1, len(tok) - 1) if tok[i] in vowels]
                if inner:
                    idx = rng.choice(inner)
                    tok = tok[:idx] + tok[idx + 1 :]
            elif op == "swap" and len(tok) >= 5:
                idx = rng.randint(1, len(tok) - 3)
                tok = tok[:idx] + tok[idx + 1] + tok[idx] + tok[idx + 2 :]
            elif op == "double_char":
                idx = rng.randint(1, len(tok) - 2)
                tok = tok[:idx] + tok[idx] + tok[idx:]
        noisy.append(tok)
    prefix = rng.choice(("tbh ", "bro ", "honestly ", "yaar ", ""))
    suffix = rng.choice((" pls", " tbh...", "!!", " fr", ""))
    return (prefix + " ".join(noisy) + suffix).strip()


def score_multilabel(
    gold: Sequence[Sequence[str]],
    preds: Sequence[Sequence[str]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    l_count = len(labels)
    exact = 0
    hamming_errors = 0
    total_tp = total_fp = total_fn = 0
    per_label: Dict[str, Dict[str, Any]] = {}
    f1s: List[float] = []

    gold_sets = [set(g) for g in gold]
    pred_sets = [set(p) for p in preds]

    for g_set, p_set in zip(gold_sets, pred_sets):
        if g_set == p_set:
            exact += 1
        for lbl in labels:
            if (lbl in g_set) != (lbl in p_set):
                hamming_errors += 1

    for lbl in labels:
        tp = sum((lbl in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fp = sum((lbl not in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fn = sum((lbl in g) and (lbl not in p) for g, p in zip(gold_sets, pred_sets))
        support = sum(lbl in g for g in gold_sets)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_label[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_denom) if micro_denom > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    subset_acc = exact / n if n > 0 else 0.0
    hamming_loss = hamming_errors / (n * l_count) if (n * l_count) > 0 else 0.0
    return {
        "n_samples": n,
        "multilabel": True,
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "subset_accuracy": round(subset_acc, 6),
        "hamming_loss": round(hamming_loss, 6),
        "per_label": per_label,
    }


def score_singlelabel(
    gold: Sequence[str],
    preds: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, preds))
    f1s: List[float] = []
    per_class: Dict[str, Dict[str, Any]] = {}
    for lbl in labels:
        tp = sum((g == lbl) and (p == lbl) for g, p in zip(gold, preds))
        fp = sum((g != lbl) and (p == lbl) for g, p in zip(gold, preds))
        fn = sum((g == lbl) and (p != lbl) for g, p in zip(gold, preds))
        support = sum(g == lbl for g in gold)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_class[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }
    return {
        "n_samples": n,
        "multilabel": False,
        "accuracy": round(correct / n, 6) if n > 0 else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 6) if f1s else 0.0,
        "per_class": per_class,
    }


def parse_record_targets(task_name: str, rows: Sequence[Any]) -> List[Any]:
    if task_name in {"emotion", "issue"}:
        return [
            sorted(
                part.strip().lower()
                for part in str(r.label or "").replace(",", "|").split("|")
                if part.strip()
            )
            for r in rows
        ]
    return [str(r.label).strip().lower() for r in rows]


def _with_text(rows: Sequence[Any], new_texts: Sequence[str]) -> List[Any]:
    from copy import copy

    out = []
    for row, txt in zip(rows, new_texts):
        cloned = copy(row)
        cloned.text = txt
        out.append(cloned)
    return out


def run_classical(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    noisy_test_rows: Sequence[Any],
    upgraded: bool,
) -> Dict[str, Any]:
    task = TaskType(task_name)
    t0 = time.perf_counter()
    if task_name == "issue":
        cfg = (
            IssueModelConfig(
                use_char_ngrams=True,
                class_weight="balanced",
                C=4.0,
            )
            if upgraded
            else IssueModelConfig()
        )
        model = TfidfLogisticIssueClassifier(config=cfg)
    elif task_name == "emotion":
        cfg = (
            EmotionModelConfig(
                use_char_ngrams=True,
                class_weight="balanced",
                C=4.0,
            )
            if upgraded
            else EmotionModelConfig()
        )
        model = TfidfLogisticEmotionClassifier(config=cfg)
    elif task_name == "sentiment":
        cfg = (
            SentimentModelConfig(
                use_char_ngrams=True,
                class_weight="balanced",
                C=4.0,
            )
            if upgraded
            else SentimentModelConfig()
        )
        model = TfidfLogisticSentimentClassifier(config=cfg)
    elif task_name == "response":
        cfg = (
            ResponseModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
            if upgraded
            else ResponseModelConfig()
        )
        model = TfidfLogisticResponseClassifier(config=cfg)
    elif task_name == "solution":
        cfg = (
            SolutionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0)
            if upgraded
            else SolutionModelConfig()
        )
        model = TfidfLogisticSolutionClassifier(config=cfg)
    else:
        raise ValueError(task_name)

    train_ann = records_to_annotations(train_rows, task)
    val_ann = records_to_annotations(val_rows, task)
    test_ann = records_to_annotations(test_rows, task)
    noisy_ann = records_to_annotations(noisy_test_rows, task)

    model.train(train_ann)
    calibrated_thresholds = None
    if upgraded and hasattr(model, "calibrate_thresholds"):
        calibrated_thresholds = model.calibrate_thresholds(val_ann)

    train_seconds = round(time.perf_counter() - t0, 3)

    def _predict_rows(anns):
        if task_name == "issue":
            docs = [a.to_issue_document() for a in anns]
            res = model.predict(docs)
            return [p.predicted_issue_labels for p in res.predictions]
        if task_name == "emotion":
            docs = [a.to_document() for a in anns]
            res = model.predict(docs)
            return [p.emotions for p in res.predictions]
        if task_name == "sentiment":
            docs = [a.to_document() for a in anns]
            res = model.predict(docs)
            return [p.sentiment or "" for p in res.predictions]
        if task_name == "response":
            docs = [a.to_response_document() for a in anns]
            res = model.predict(docs)
            return [p.response_type or "" for p in res.predictions]
        docs = [a.to_solution_document() for a in anns]
        res = model.predict(docs)
        return ["solution" if p.is_solution else "not_solution" for p in res.predictions]

    gold_test = parse_record_targets(task_name, test_rows)
    clean_preds = _predict_rows(test_ann)
    noisy_preds = _predict_rows(noisy_ann)

    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    return {
        "train_seconds": train_seconds,
        "labels": list(model.labels),
        "calibrated_thresholds": calibrated_thresholds,
        "clean_test": scorer(gold_test, clean_preds, model.labels),
        "noisy_test": scorer(gold_test, noisy_preds, model.labels),
    }


def run_muril(
    task_name: str,
    train_texts: Sequence[str],
    train_targets: Sequence[Any],
    val_texts: Sequence[str],
    val_targets: Sequence[Any],
    test_texts: Sequence[str],
    noisy_test_texts: Sequence[str],
    test_targets: Sequence[Any],
    epochs: int,
    save_dir: Path | None = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = MuRILTaskConfig(
        task=task_name,
        epochs=epochs,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        max_length=160,
        random_state=SEED,
        verbose=True,
    )
    clf = MuRILTaskClassifier(cfg)
    fit_summary = clf.fit(train_texts, train_targets, val_texts, val_targets)
    train_seconds = round(time.perf_counter() - t0, 3)

    clean_preds, _ = clf.predict_labels(test_texts)
    noisy_preds, _ = clf.predict_labels(noisy_test_texts)

    if save_dir is not None:
        clf.save(save_dir)

    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    return {
        "train_seconds": train_seconds,
        "device": fit_summary["device"],
        "amp_dtype": fit_summary["amp_dtype"],
        "best_epoch": fit_summary["best_epoch"],
        "best_val_macro_f1": fit_summary["best_val_macro_f1"],
        "calibrated_thresholds": fit_summary["per_label_thresholds"] if clf.is_multilabel else None,
        "labels": list(clf.labels),
        "clean_test": scorer(test_targets, clean_preds, clf.labels),
        "noisy_test": scorer(test_targets, noisy_preds, clf.labels),
    }


def main(run_muril_models: bool = True, muril_epochs: int = 6, save_muril_issue: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    post_titles = load_post_titles()
    report: Dict[str, Any] = {
        "benchmark_scope": "Thread-grouped post_id splits (896 train / 192 val / 192 test, seed=20261116) on synthetic demo corpus + noisy Reddit perturbation stress test.",
        "seed": SEED,
        "tasks": {},
    }

    for task_name in TASKS:
        print(f"\n==================== TASK: {task_name.upper()} ====================")
        records = load_annotation_dataset(TaskType(task_name), str(DATA_DIR / f"{task_name}.csv"))
        spec = SplitSpec(
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            random_state=SEED,
            stratify=True,
        )
        groups = extract_groups(records, "post_id")
        split = split_records(
            records,
            labels=[r.label for r in records],
            spec=spec,
            groups=groups,
        )
        train_rows, val_rows, test_rows = apply_split(records, split)
        leak = check_leakage(train_rows, val_rows, test_rows)
        assert not leak.get("leakage_detected"), f"Leakage detected in {task_name}"

        rng = random.Random(SEED + TASKS.index(task_name))
        noisy_texts = [perturb_reddit_noise(r.text, rng) for r in test_rows]
        noisy_test_rows = _with_text(test_rows, noisy_texts)

        v1_res = run_classical(task_name, train_rows, val_rows, test_rows, noisy_test_rows, upgraded=False)
        v2_res = run_classical(task_name, train_rows, val_rows, test_rows, noisy_test_rows, upgraded=True)

        task_entry: Dict[str, Any] = {
            "leakage_detected": False,
            "baseline_v1_tfidf": v1_res,
            "upgraded_v2_word_char_calibrated": v2_res,
        }

        print(
            f"  [V1 Baseline]  clean_macro_f1={v1_res['clean_test']['macro_f1']:.4f} "
            f"| noisy_macro_f1={v1_res['noisy_test']['macro_f1']:.4f}"
        )
        print(
            f"  [V2 Word+Char] clean_macro_f1={v2_res['clean_test']['macro_f1']:.4f} "
            f"| noisy_macro_f1={v2_res['noisy_test']['macro_f1']:.4f}"
        )

        # Thread-title context ablation for Issue Identification (zero post_id leakage!)
        if task_name == "issue":
            def _attach_title(rows_seq):
                return _with_text(
                    rows_seq,
                    [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in rows_seq],
                )

            ctx_train = _attach_title(train_rows)
            ctx_val = _attach_title(val_rows)
            ctx_test = _attach_title(test_rows)
            ctx_noisy_rng = random.Random(SEED + 999)
            ctx_noisy_test = _with_text(
                test_rows,
                [
                    f"{perturb_reddit_noise(post_titles.get(r.post_id or '', ''), ctx_noisy_rng)} | {nt}"
                    for r, nt in zip(test_rows, noisy_texts)
                ],
            )
            v2_ctx_res = run_classical(
                task_name, ctx_train, ctx_val, ctx_test, ctx_noisy_test, upgraded=True
            )
            task_entry["upgraded_v2_with_post_title_context"] = v2_ctx_res
            print(
                f"  [V2 + Context] clean_macro_f1={v2_ctx_res['clean_test']['macro_f1']:.4f} "
                f"(subset={v2_ctx_res['clean_test']['subset_accuracy']:.4f}) "
                f"| noisy_macro_f1={v2_ctx_res['noisy_test']['macro_f1']:.4f}"
            )

        if run_muril_models:
            tr_t = [r.text for r in train_rows]
            va_t = [r.text for r in val_rows]
            te_t = [r.text for r in test_rows]
            tr_y = parse_record_targets(task_name, train_rows)
            va_y = parse_record_targets(task_name, val_rows)
            te_y = parse_record_targets(task_name, test_rows)

            muril_res = run_muril(
                task_name,
                tr_t,
                tr_y,
                va_t,
                va_y,
                te_t,
                noisy_texts,
                te_y,
                epochs=muril_epochs,
                save_dir=(OUT_DIR / f"muril_{task_name}" if (save_muril_issue and task_name == "issue") else None),
            )
            task_entry["muril_v2_gpu"] = muril_res
            print(
                f"  [MuRIL v2 GPU] clean_macro_f1={muril_res['clean_test']['macro_f1']:.4f} "
                f"| noisy_macro_f1={muril_res['noisy_test']['macro_f1']:.4f} "
                f"(best_ep={muril_res['best_epoch']}, {muril_res['train_seconds']}s)"
            )

            if task_name == "issue":
                tr_ctx = [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in train_rows]
                va_ctx = [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in val_rows]
                te_ctx = [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in test_rows]
                ctx_noisy_rng2 = random.Random(SEED + 999)
                te_ctx_noisy = [
                    f"{perturb_reddit_noise(post_titles.get(r.post_id or '', ''), ctx_noisy_rng2)} | {nt}"
                    for r, nt in zip(test_rows, noisy_texts)
                ]
                muril_ctx_res = run_muril(
                    task_name,
                    tr_ctx,
                    tr_y,
                    va_ctx,
                    va_y,
                    te_ctx,
                    te_ctx_noisy,
                    te_y,
                    epochs=muril_epochs,
                    save_dir=None,
                )
                task_entry["muril_v2_gpu_with_post_title_context"] = muril_ctx_res
                print(
                    f"  [MuRIL + Ctx]  clean_macro_f1={muril_ctx_res['clean_test']['macro_f1']:.4f} "
                    f"(subset={muril_ctx_res['clean_test']['subset_accuracy']:.4f}) "
                    f"| noisy_macro_f1={muril_ctx_res['noisy_test']['macro_f1']:.4f}"
                )

        report["tasks"][task_name] = task_entry

    out_path = OUT_DIR / "benchmark_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved full benchmark report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-muril", action="store_true", help="Skip GPU MuRIL training")
    parser.add_argument("--epochs", type=int, default=6, help="MuRIL fine-tuning epochs (default 6)")
    args = parser.parse_args()
    main(run_muril_models=not args.no_muril, muril_epochs=args.epochs)
