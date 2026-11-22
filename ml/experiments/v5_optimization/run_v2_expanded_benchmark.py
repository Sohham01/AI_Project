"""Item #18 / Track B — V2 Expanded Real-Data Training & Testing Benchmark.

Strict Integrity & Separation:
- NEVER modifies data/annotations/real_reddit_labeled_v1/ or ml/artifacts/live_corpus/.
- Reads isolated data/annotations/real_reddit_labeled_v2/master_labeled_v2.csv (707 real rows:
  545 trainval_v2 + 92 untouched test_v5_frozen + 70 brand-new unseen-thread test_v2_new = 162 test).
- Groups strictly by exact `corpus_post_id` and asserts bit-exact preservation of frozen V5 test
  SHA-256 plus ZERO corpus_post_id overlap across trainval_v2, test_v5_frozen, and test_v2_new.
- Uses each task's 100% strictly inductive Track A OOF-winning architecture from strict_clean_summary.json
  and evaluates on BOTH:
    (1) Untouched test_v5_frozen (n=92, 37 threads) — exact paired bootstrap delta vs V1-trained same arch!
    (2) Expanded test_v2_expanded (n=162, 69 threads) — larger unseen-thread test set & tighter CIs!
"""

from __future__ import annotations

import csv
import hashlib
import json
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
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    TrainSpec,
    apply_logits_calibration,
    encode_task_targets,
    ensure_dir,
    evaluate_predictions,
    load_corpus_lookups,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    parse_multilabel_str,
    save_json,
    thread_clustered_bootstrap_ci,
    train_and_predict_logits,
    tune_multilabel_thresholds,
)
from ml.models.transformer.muril_classifier import _calibrate_temperature

V2_DIR = ROOT / "data" / "annotations" / "real_reddit_labeled_v2"


def load_v2_dataset() -> Dict[str, Any]:
    master_v2 = V2_DIR / "master_labeled_v2.csv"
    assert master_v2.exists(), f"Missing {master_v2} — run build_v2_curated_dataset.py first!"

    v1 = load_v1_records_and_frozen_splits()
    frozen_dids = set(v1["manifest"]["test_document_ids"])
    frozen_cpids = set(v1["manifest"]["test_corpus_post_ids"])
    _, post_by_pid, _ = load_corpus_lookups()

    rows: List[Dict[str, Any]] = []
    with master_v2.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cpid = row["corpus_post_id"]
            ccid = row["corpus_comment_id"]
            parent_cid = None
            parent_txt = ""
            p_obj = post_by_pid.get(cpid)
            if p_obj is not None:
                in_post = {x["reddit_id"]: x for x in p_obj.get("comments") or []}
                c_obj = in_post.get(ccid)
                if c_obj is not None:
                    prid = c_obj.get("parent_reddit_id")
                    if prid and prid in in_post:
                        parent_cid = prid
                        parent_txt = str(in_post[prid].get("body") or "").strip()

            rows.append(
                {
                    "document_id": row["document_id"],
                    # Group strictly by verified canonical corpus_post_id!
                    "post_id": cpid,
                    "corpus_post_id": cpid,
                    "comment_id": ccid,
                    "subreddit": row.get("subreddit") or "",
                    "post_title": (row.get("post_title") or "").strip(),
                    "parent_comment_id": parent_cid,
                    "parent_comment_text": parent_txt,
                    "text": (row.get("text") or "").strip(),
                    "v2_split": row["v2_split"].strip(),
                    "annotation_source": row.get("annotation_source") or "",
                    "labels": {
                        "issue": parse_multilabel_str(row.get("issues", "")),
                        "sentiment": str(row.get("sentiment", "")).strip(),
                        "emotion": parse_multilabel_str(row.get("emotions", "")),
                        "response": str(row.get("response", "")).strip(),
                        "solution": str(row.get("solution", "")).strip(),
                    },
                }
            )

    trainval_v2 = [r for r in rows if r["v2_split"] in {"trainval_v1", "trainval_v2_new"}]
    test_v5_frozen = [r for r in rows if r["v2_split"] == "test_v5_frozen"]
    test_v2_new = [r for r in rows if r["v2_split"] == "test_v2_new"]

    # Align frozen test order identically to V1 test order
    v1_test_order = [r["document_id"] for r in v1["test"]]
    by_did = {r["document_id"]: r for r in test_v5_frozen}
    test_v5_frozen = [by_did[d] for d in v1_test_order]
    test_v2_expanded = test_v5_frozen + test_v2_new

    assert {r["document_id"] for r in test_v5_frozen} == frozen_dids
    sha_check = hashlib.sha256(",".join(sorted(r["document_id"] for r in test_v5_frozen)).encode("utf-8")).hexdigest()
    assert sha_check == v1["manifest"]["test_sha256"]

    tr_pids = {r["corpus_post_id"] for r in trainval_v2}
    te_f_pids = {r["corpus_post_id"] for r in test_v5_frozen}
    te_n_pids = {r["corpus_post_id"] for r in test_v2_new}
    assert te_f_pids == frozen_cpids
    assert not (tr_pids & te_f_pids), "LEAKAGE: trainval_v2 overlaps test_v5_frozen!"
    assert not (tr_pids & te_n_pids), "LEAKAGE: trainval_v2 overlaps test_v2_new!"
    assert not (te_f_pids & te_n_pids), "LEAKAGE: test_v5_frozen overlaps test_v2_new!"

    return {
        "all": rows,
        "trainval_v2": trainval_v2,
        "test_v5_frozen": test_v5_frozen,
        "test_v2_new": test_v2_new,
        "test_v2_expanded": test_v2_expanded,
        "stats": {
            "total_rows": len(rows),
            "total_threads": len({r["corpus_post_id"] for r in rows}),
            "trainval_v2_rows": len(trainval_v2),
            "trainval_v2_threads": len(tr_pids),
            "test_v5_frozen_rows": len(test_v5_frozen),
            "test_v5_frozen_threads": len(te_f_pids),
            "test_v2_new_rows": len(test_v2_new),
            "test_v2_new_threads": len(te_n_pids),
            "test_v2_expanded_rows": len(test_v2_expanded),
            "test_v2_expanded_threads": len(te_f_pids | te_n_pids),
            "annotation_sources": dict(Counter(r["annotation_source"] for r in rows)),
        },
    }


def run_v2_benchmark() -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "v2_expanded_data")
    logits_dir = ensure_dir(out_dir / "logits")
    t_all = time.perf_counter()

    ds = load_v2_dataset()
    trainval_v2 = ds["trainval_v2"]
    test_frozen = ds["test_v5_frozen"]
    test_expanded = ds["test_v2_expanded"]

    sc = json.loads((EXP_ROOT / "strict_clean_final" / "strict_clean_summary.json").read_text(encoding="utf-8"))

    summary_path = out_dir / "v2_benchmark_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "phase": "track_b_v2_expanded_real_data_benchmark",
            "provenance_disclosure": (
                "Track B adds 248 genuine anonymized Reddit comments from data/live_corpus/corpus.json "
                "(178 active-learning trainval + 70 unseen-thread test; 2 bot/noise rows abstained) with "
                "rubric-curated annotations pending human IAA verification via iaa_double_annotation_50.csv. "
                "Reported completely separately from Track A (untouched V1 459-comment benchmark)."
            ),
            "dataset_stats": ds["stats"],
            "tasks": {},
        }

    for task in TASK_NAMES:
        if task in summary.get("tasks", {}) and (logits_dir / f"v2_{task}_logits.npz").exists():
            print(f"\n[Track B V2] Skipping already-completed task: {task.upper()}", flush=True)
            continue
        print(f"\n==================== TRACK B (V2 EXPANDED DATA): {task.upper()} ====================", flush=True)
        multilabel = task in MULTILABEL_TASKS
        labels = CANONICAL_LABELS[task]

        # Use the exact Track A OOF-winning single-task neural config for pure apples-to-apples data-lift test
        win_neural = sc["tasks"][task]["oof_winner_neural"]
        win_obj = sc["tasks"][task]["candidates"][win_neural]
        backbone_id = win_obj["backbone"]
        ctx_win = win_obj["context_mode"]
        loss_win = win_obj["loss_type"]

        # Load Track A V1-trained exact logits on frozen n=92 for paired data-expansion delta!
        v1_npz = np.load(EXP_ROOT / "strict_clean_final" / "logits" / f"{task}_{win_neural}.npz")
        v1_frozen_probs = apply_logits_calibration(v1_npz["ens_test_logits"], multilabel, 1.0)
        v1_frozen_taus = win_obj["thresholds"]

        y_tv = encode_task_targets(trainval_v2, task)
        y_te_f = encode_task_targets(test_frozen, task)
        y_te_e = encode_task_targets(test_expanded, task)
        folds = make_group_cv_folds(trainval_v2, task=task, n_splits=5, seed=SEED)

        oof_logits = np.zeros((len(trainval_v2), len(labels)), dtype=np.float32)
        te_f_folds, te_e_folds = [], []
        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            sp = TrainSpec(
                task=task,
                backbone_id=backbone_id,
                context_mode=ctx_win,
                loss_type=loss_win,
                max_length=256,
                epochs=6,
                seed=SEED + 2000 + f_idx * 13,
            )
            rf = train_and_predict_logits(
                sp,
                [trainval_v2[i] for i in tr_idx],
                [trainval_v2[i] for i in va_idx],
                {"test_frozen": test_frozen, "test_expanded": test_expanded},
            )
            oof_logits[va_idx] = rf["val_logits"]
            te_f_folds.append(rf["eval_logits"]["test_frozen"])
            te_e_folds.append(rf["eval_logits"]["test_expanded"])

        t_g = float(_calibrate_temperature(oof_logits, y_tv, multilabel=multilabel))
        oof_probs = apply_logits_calibration(oof_logits, multilabel, t_g)
        taus = tune_multilabel_thresholds(y_tv, oof_probs, labels, mode="bounded") if multilabel else None

        ens_f_logits = np.mean(np.stack(te_f_folds, axis=0), axis=0)
        ens_e_logits = np.mean(np.stack(te_e_folds, axis=0), axis=0)
        # Per Phase 1 ensemble calibration finding, 5-fold logit ensembles use T=1.0
        probs_f = apply_logits_calibration(ens_f_logits, multilabel, 1.0)
        probs_e = apply_logits_calibration(ens_e_logits, multilabel, 1.0)

        ev_oof = evaluate_predictions(task, y_tv, oof_probs, thresholds=taus)
        ev_f = evaluate_predictions(task, y_te_f, probs_f, thresholds=taus)
        ev_e = evaluate_predictions(task, y_te_e, probs_e, thresholds=taus)

        ci_frozen_vs_v1 = thread_clustered_bootstrap_ci(
            test_frozen,
            task,
            y_te_f,
            probs_f,
            probs_b=v1_frozen_probs,
            thresholds_a=taus,
            thresholds_b=v1_frozen_taus,
            n_boot=2000,
            seed=SEED,
        )
        ci_expanded = thread_clustered_bootstrap_ci(
            test_expanded,
            task,
            y_te_e,
            probs_e,
            thresholds_a=taus,
            n_boot=2000,
            seed=SEED,
        )

        np.savez_compressed(
            logits_dir / f"v2_{task}_logits.npz",
            oof_logits=oof_logits,
            ens_frozen_logits=ens_f_logits,
            ens_expanded_logits=ens_e_logits,
        )

        summary["tasks"][task] = {
            "config_from_track_a": {
                "name": win_neural,
                "backbone": backbone_id,
                "context": ctx_win,
                "loss": loss_win,
            },
            "v1_same_arch_frozen92_f1": win_obj["frozen_test_eval"]["macro_f1"],
            "v1_same_arch_frozen92_ci_width": round(
                win_obj["bootstrap_ci_vs_v5"]["ci95_high"] - win_obj["bootstrap_ci_vs_v5"]["ci95_low"], 4
            ),
            "temperature_single_oof": round(t_g, 3),
            "bounded_thresholds": taus,
            "v2_oof_eval": ev_oof,
            "frozen_test_n92_eval": ev_f,
            "frozen_test_n92_ci_vs_v1_same_arch": ci_frozen_vs_v1,
            "expanded_test_eval": ev_e,
            "expanded_test_ci": ci_expanded,
            "expanded_test_ci_width": round(ci_expanded["ci95_high"] - ci_expanded["ci95_low"], 4),
        }
        print(
            f"  [V2 {task.upper():9s}] OOF(545)={ev_oof['macro_f1']:.4f} | "
            f"Frozen92={ev_f['macro_f1']:.4f} vs V1={win_obj['frozen_test_eval']['macro_f1']:.4f} "
            f"(delta={ci_frozen_vs_v1['paired_delta_mean']:+.4f} CI={ci_frozen_vs_v1['paired_delta_ci95']}) | "
            f"Expanded(162)={ev_e['macro_f1']:.4f} CI=[{ci_expanded['ci95_low']:.4f},{ci_expanded['ci95_high']:.4f}]",
            flush=True,
        )
        save_json(out_dir / "v2_benchmark_summary.json", summary)

    summary["overall_means"] = {
        "v1_track_a_same_arch_frozen92": round(
            float(np.mean([summary["tasks"][t]["v1_same_arch_frozen92_f1"] for t in TASK_NAMES])), 4
        ),
        "v2_oof_545_mean": round(float(np.mean([summary["tasks"][t]["v2_oof_eval"]["macro_f1"] for t in TASK_NAMES])), 4),
        "v2_on_frozen_n92_mean": round(
            float(np.mean([summary["tasks"][t]["frozen_test_n92_eval"]["macro_f1"] for t in TASK_NAMES])), 4
        ),
        "v2_on_expanded_n162_mean": round(
            float(np.mean([summary["tasks"][t]["expanded_test_eval"]["macro_f1"] for t in TASK_NAMES])), 4
        ),
    }
    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
    save_json(out_dir / "v2_benchmark_summary.json", summary)
    print(f"\nTrack B V2 Expanded Benchmark complete in {summary['total_elapsed_seconds']}s: {summary['overall_means']}", flush=True)
    return summary


if __name__ == "__main__":
    run_v2_benchmark()
