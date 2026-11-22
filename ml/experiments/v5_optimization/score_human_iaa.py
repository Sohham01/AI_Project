"""Exact Inter-Annotator Agreement (IAA) Scorer, Validation Gate & Disagreement Adjudicator.

Computes exact statistical IAA across all 5 canonical RCI NLP tasks:
- Single-label (`sentiment`, `response`, `solution`):
    * Cohen's kappa (exact)
    * Krippendorff's alpha (nominal, exact)
    * Raw percent agreement & per-class confusion matrix
- Multi-label (`issue`, `emotion`):
    * Per-label binary Cohen's kappa & binary Krippendorff's alpha
    * Micro-averaged & Macro-averaged binary Krippendorff's alpha
    * MASI-weighted Krippendorff's alpha (Passonneau 2006 set-distance metric)
    * Exact-set match % & mean Jaccard similarity

Validation Gate Thresholds (from docs/V5_ACTIVE_LEARNING_PLAN.md):
- sentiment & solution: Cohen's kappa >= 0.70
- response (6-class):   Cohen's kappa >= 0.65
- issue & emotion:      Krippendorff's alpha (MASI or micro binary) >= 0.65

Usage:
  # Score completed Human Studio CSV vs Pass-1 V2 rubric labels:
  python ml/experiments/v5_optimization/score_human_iaa.py

  # Score two independent human annotators (Annotator A vs Annotator B):
  python ml/experiments/v5_optimization/score_human_iaa.py --annotator-a <csv_a> --annotator-b <csv_b>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    EXP_ROOT,
    ensure_dir,
    parse_multilabel_str,
    save_json,
)

AL_DIR = EXP_ROOT / "active_learning"
DEFAULT_HUMAN_CSV = AL_DIR / "iaa_50_human_annotator_studio.csv"
DEFAULT_V2_MASTER = ROOT / "data" / "annotations" / "real_reddit_labeled_v2" / "master_labeled_v2.csv"
DEFAULT_V2_ABSTAIN = ROOT / "data" / "annotations" / "real_reddit_labeled_v2" / "adjudication_queue.csv"
DISAGREEMENT_CSV = AL_DIR / "iaa_disagreements_adjudication.csv"
RESULTS_JSON = AL_DIR / "iaa_results.json"
REPORT_MD = ROOT / "docs" / "V5_HUMAN_IAA_REPORT.md"


def krippendorff_alpha_nominal(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Exact Krippendorff's alpha for 2 coders on nominal categories (0/1 or class labels)."""
    n = len(a)
    if n == 0:
        return 0.0
    do = sum(1.0 for x, y in zip(a, b) if x != y) / float(n)
    counts = Counter(list(a) + list(b))
    total = 2.0 * n
    # Expected disagreement De = 1 - sum_c n_c*(n_c - 1) / (2n * (2n - 1))
    if total <= 1:
        return 1.0 if do == 0.0 else 0.0
    same_chance = sum(c * (c - 1.0) for c in counts.values()) / (total * (total - 1.0))
    de = 1.0 - same_chance
    if de <= 1e-12:
        return 1.0 if do == 0.0 else 0.0
    return float(1.0 - (do / de))


def masi_distance(set_a: Set[str], set_b: Set[str]) -> float:
    """Passonneau (2006) MASI distance in [0, 1] between two label sets (0 = identical, 1 = disjoint)."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    jaccard = inter / float(union)
    if set_a == set_b:
        m = 1.0
    elif set_a.issubset(set_b) or set_b.issubset(set_a):
        m = 2.0 / 3.0
    elif inter > 0:
        m = 1.0 / 3.0
    else:
        m = 0.0
    return float(1.0 - m * jaccard)


def krippendorff_alpha_masi(sets_a: Sequence[Set[str]], sets_b: Sequence[Set[str]]) -> float:
    """Exact Krippendorff's alpha for 2 multi-label coders using MASI set distance."""
    n = len(sets_a)
    if n == 0:
        return 0.0
    do = sum(masi_distance(sa, sb) for sa, sb in zip(sets_a, sets_b)) / float(n)
    pool = list(sets_a) + list(sets_b)
    m = len(pool)
    if m <= 1:
        return 1.0 if do == 0.0 else 0.0
    pair_sum = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pair_sum += masi_distance(pool[i], pool[j])
    de = pair_sum / (m * (m - 1) / 2.0)
    if de <= 1e-12:
        return 1.0 if do == 0.0 else 0.0
    return float(1.0 - (do / de))


def safe_cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) == 0:
        return 0.0
    if len(set(a) | set(b)) <= 1:
        return 1.0 if list(a) == list(b) else 0.0
    val = float(cohen_kappa_score(a, b))
    return 0.0 if np.isnan(val) else val


def _load_studio_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("document_id") or "").strip()
            status = (r.get("review_status") or "").strip().lower() or "unreviewed"
            out[did] = {
                "document_id": did,
                "post_id": r.get("post_id", ""),
                "post_title": r.get("post_title", ""),
                "parent_text": r.get("parent_text") or r.get("parent_comment_text") or "",
                "text": r.get("original_text") or r.get("text") or "",
                "review_status": status,
                "abstain_reason": (r.get("abstain_reason") or "").strip().lower(),
                "annotator_id": (r.get("annotator_id") or "").strip(),
                "annotator_notes": (r.get("annotator_notes") or "").strip(),
                "sentiment": (r.get("sentiment") or "").strip().lower(),
                "emotions": set(parse_multilabel_str((r.get("emotions") or "").lower())),
                "issues": set(parse_multilabel_str((r.get("issues") or "").lower())),
                "response": (r.get("response_type") or r.get("response") or "").strip().lower(),
                "solution": (
                    "solution"
                    if (r.get("is_solution") or r.get("solution") or "").strip().lower() == "true"
                    else (
                        "not_solution"
                        if (r.get("is_solution") or r.get("solution") or "").strip().lower() == "false"
                        else (r.get("is_solution") or r.get("solution") or "").strip().lower()
                    )
                ),
            }
    return out


def _load_v2_pass1_reference(doc_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if DEFAULT_V2_MASTER.exists():
        with DEFAULT_V2_MASTER.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                did = (r.get("document_id") or "").strip()
                if did not in doc_ids:
                    continue
                out[did] = {
                    "document_id": did,
                    "post_id": r.get("post_id", ""),
                    "post_title": r.get("post_title", ""),
                    "parent_text": "",
                    "text": r.get("text", ""),
                    "review_status": "labeled",
                    "abstain_reason": "",
                    "annotator_id": "v2_rubric_pass1",
                    "sentiment": (r.get("sentiment") or "").strip().lower(),
                    "emotions": set(parse_multilabel_str((r.get("emotions") or "").lower())),
                    "issues": set(parse_multilabel_str((r.get("issues") or "").lower())),
                    "response": (r.get("response") or "").strip().lower(),
                    "solution": (r.get("solution") or "").strip().lower(),
                }
    if DEFAULT_V2_ABSTAIN.exists():
        with DEFAULT_V2_ABSTAIN.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                did = (r.get("document_id") or "").strip()
                if did not in doc_ids:
                    continue
                out[did] = {
                    "document_id": did,
                    "post_id": r.get("post_id", ""),
                    "post_title": "",
                    "parent_text": "",
                    "text": r.get("text", ""),
                    "review_status": (r.get("review_status") or "abstain").strip().lower(),
                    "abstain_reason": (r.get("abstain_reason") or "").strip().lower(),
                    "annotator_id": "v2_rubric_pass1",
                    "sentiment": "",
                    "emotions": set(),
                    "issues": set(),
                    "response": "",
                    "solution": "",
                }
    return out


def compute_iaa_report(
    annotator_a: Dict[str, Dict[str, Any]],
    annotator_b: Dict[str, Dict[str, Any]],
    *,
    label_a: str = "Human Annotator",
    label_b: str = "Pass-1 V2 Rubric",
) -> Dict[str, Any]:
    shared_ids = sorted(set(annotator_a.keys()) & set(annotator_b.keys()))
    reviewed_ids = [
        d for d in shared_ids if annotator_a[d]["review_status"] != "unreviewed" and annotator_b[d]["review_status"] != "unreviewed"
    ]
    both_labeled = [
        d for d in reviewed_ids if annotator_a[d]["review_status"] == "labeled" and annotator_b[d]["review_status"] == "labeled"
    ]

    status_agree = (
        round(
            100.0 * sum(1 for d in reviewed_ids if annotator_a[d]["review_status"] == annotator_b[d]["review_status"]) / len(reviewed_ids),
            2,
        )
        if reviewed_ids
        else 0.0
    )
    binary_status_agree = (
        round(
            100.0
            * sum(
                1
                for d in reviewed_ids
                if (annotator_a[d]["review_status"] == "labeled") == (annotator_b[d]["review_status"] == "labeled")
            )
            / len(reviewed_ids),
            2,
        )
        if reviewed_ids
        else 0.0
    )

    metrics: Dict[str, Any] = {
        "annotator_a": label_a,
        "annotator_b": label_b,
        "total_pack_items": len(shared_ids),
        "reviewed_by_both": len(reviewed_ids),
        "both_labeled_count": len(both_labeled),
        "review_status_agreement_pct": status_agree,
        "labeled_vs_nonlabeled_agreement_pct": binary_status_agree,
        "tasks": {},
        "gate_checks": {},
        "disagreements": [],
    }

    if len(both_labeled) == 0:
        metrics["status"] = "PENDING_HUMAN_ANNOTATIONS"
        return metrics

    metrics["status"] = "COMPLETE" if len(reviewed_ids) >= len(shared_ids) else "PARTIAL_PROGRESS"

    # Single-label tasks
    for task, gate_min in (("sentiment", 0.70), ("response", 0.65), ("solution", 0.70)):
        va = [annotator_a[d][task] for d in both_labeled]
        vb = [annotator_b[d][task] for d in both_labeled]
        labels = CANONICAL_LABELS[task]
        l2i = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for xa, xb in zip(va, vb):
            if xa in l2i and xb in l2i:
                cm[l2i[xa], l2i[xb]] += 1
        kappa = round(safe_cohen_kappa(va, vb), 4)
        alpha = round(krippendorff_alpha_nominal(va, vb), 4)
        pct = round(100.0 * sum(1 for xa, xb in zip(va, vb) if xa == xb) / len(both_labeled), 2)
        metrics["tasks"][task] = {
            "paradigm": "single_label",
            "n_compared": len(both_labeled),
            "raw_agreement_pct": pct,
            "cohen_kappa": kappa,
            "krippendorff_alpha_nominal": alpha,
            "gate_threshold": gate_min,
            "gate_passed": bool(kappa >= gate_min),
            "confusion_matrix_labels": labels,
            "confusion_matrix": cm.tolist(),
        }
        metrics["gate_checks"][task] = bool(kappa >= gate_min)

    # Multi-label tasks
    for task, field_key, gate_min in (("issue", "issues", 0.65), ("emotion", "emotions", 0.65)):
        labels = CANONICAL_LABELS[task]
        sa_list = [annotator_a[d][field_key] for d in both_labeled]
        sb_list = [annotator_b[d][field_key] for d in both_labeled]

        masi_alpha = round(krippendorff_alpha_masi(sa_list, sb_list), 4)
        exact_pct = round(100.0 * sum(1 for sa, sb in zip(sa_list, sb_list) if sa == sb) / len(both_labeled), 2)
        jaccards = [
            1.0 if (not sa and not sb) else len(sa & sb) / float(len(sa | sb))
            for sa, sb in zip(sa_list, sb_list)
        ]
        mean_jaccard = round(float(np.mean(jaccards)), 4)

        per_label = {}
        flat_a, flat_b = [], []
        active_alphas = []
        for lbl in labels:
            ba = [1 if lbl in sa else 0 for sa in sa_list]
            bb = [1 if lbl in sb else 0 for sb in sb_list]
            flat_a.extend(ba)
            flat_b.extend(bb)
            lk = round(safe_cohen_kappa(ba, bb), 4)
            la = round(krippendorff_alpha_nominal(ba, bb), 4)
            per_label[lbl] = {
                "count_a": int(sum(ba)),
                "count_b": int(sum(bb)),
                "cohen_kappa": lk,
                "krippendorff_alpha": la,
            }
            if sum(ba) + sum(bb) > 0:
                active_alphas.append(la)

        micro_alpha = round(krippendorff_alpha_nominal(flat_a, flat_b), 4)
        macro_alpha = round(float(np.mean(active_alphas)) if active_alphas else 1.0, 4)
        gate_pass = bool(max(masi_alpha, micro_alpha) >= gate_min)
        metrics["tasks"][task] = {
            "paradigm": "multi_label",
            "n_compared": len(both_labeled),
            "exact_set_agreement_pct": exact_pct,
            "mean_jaccard": mean_jaccard,
            "krippendorff_alpha_masi": masi_alpha,
            "krippendorff_alpha_micro_binary": micro_alpha,
            "krippendorff_alpha_macro_binary": macro_alpha,
            "gate_threshold": gate_min,
            "gate_passed": gate_pass,
            "per_label": per_label,
        }
        metrics["gate_checks"][task] = gate_pass

    metrics["all_gates_passed"] = bool(all(metrics["gate_checks"].values()) and len(reviewed_ids) >= len(shared_ids))

    # Build side-by-side disagreement list for adjudication
    for d in reviewed_ids:
        ra, rb = annotator_a[d], annotator_b[d]
        diffs = []
        if ra["review_status"] != rb["review_status"]:
            diffs.append("review_status")
        if ra["review_status"] == "labeled" and rb["review_status"] == "labeled":
            for k in ("sentiment", "response", "solution"):
                if ra[k] != rb[k]:
                    diffs.append(k)
            if ra["issues"] != rb["issues"]:
                diffs.append("issue")
            if ra["emotions"] != rb["emotions"]:
                diffs.append("emotion")
        if diffs:
            metrics["disagreements"].append(
                {
                    "document_id": d,
                    "post_id": ra["post_id"] or rb["post_id"],
                    "disagreed_tasks": "|".join(diffs),
                    "post_title": ra["post_title"] or rb["post_title"],
                    "parent_text": ra["parent_text"] or rb["parent_text"],
                    "text": ra["text"] or rb["text"],
                    "a_status": ra["review_status"],
                    "b_status": rb["review_status"],
                    "a_sentiment": ra["sentiment"],
                    "b_sentiment": rb["sentiment"],
                    "a_emotions": "|".join(sorted(ra["emotions"])),
                    "b_emotions": "|".join(sorted(rb["emotions"])),
                    "a_issues": "|".join(sorted(ra["issues"])),
                    "b_issues": "|".join(sorted(rb["issues"])),
                    "a_response": ra["response"],
                    "b_response": rb["response"],
                    "a_solution": ra["solution"],
                    "b_solution": rb["solution"],
                    "adjudicated_status": "",
                    "adjudicated_sentiment": "",
                    "adjudicated_emotions": "",
                    "adjudicated_issues": "",
                    "adjudicated_response": "",
                    "adjudicated_solution": "",
                    "adjudication_notes": "",
                }
            )
    return metrics


def write_iaa_outputs(report: Dict[str, Any]) -> None:
    save_json(RESULTS_JSON, report)
    if report.get("disagreements"):
        with DISAGREEMENT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report["disagreements"][0].keys()))
            w.writeheader()
            w.writerows(report["disagreements"])

    lines = [
        "# Phase 6 — Human Inter-Annotator Agreement (IAA) & Adjudication Report",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Coders Compared:** `{report['annotator_a']}` vs. `{report['annotator_b']}`  ",
        f"**Coverage:** `{report['reviewed_by_both']} / {report['total_pack_items']}` items reviewed by both (`{report['both_labeled_count']}` both labeled; review-status agreement `{report['review_status_agreement_pct']}%`)  ",
        f"**Validation Gate Passed:** `{'YES' if report.get('all_gates_passed') else 'NO (pending review or adjudication)'}`",
        "",
    ]
    if report["both_labeled_count"] > 0:
        lines += [
            "## Task-Level IAA Summary",
            "",
            "| Task | Paradigm | $n$ | Raw / Exact % | Cohen's $\\kappa$ | Krippendorff's $\\alpha$ | Gate | Pass? |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for t in ("sentiment", "response", "solution"):
            o = report["tasks"][t]
            lines.append(
                f"| **{t}** | Single-label | {o['n_compared']} | {o['raw_agreement_pct']}% | `{o['cohen_kappa']:.4f}` | `{o['krippendorff_alpha_nominal']:.4f}` | $\\ge {o['gate_threshold']}$ | {'PASS' if o['gate_passed'] else 'FAIL'} |"
            )
        for t in ("issue", "emotion"):
            o = report["tasks"][t]
            lines.append(
                f"| **{t}** | Multi-label | {o['n_compared']} | {o['exact_set_agreement_pct']}% (Jaccard `{o['mean_jaccard']}`) | *(per-label)* | MASI `{o['krippendorff_alpha_masi']:.4f}` / Micro `{o['krippendorff_alpha_micro_binary']:.4f}` | $\\ge {o['gate_threshold']}$ | {'PASS' if o['gate_passed'] else 'FAIL'} |"
            )
        lines += [
            "",
            f"Total items with at least one task disagreement: **`{len(report['disagreements'])}`** (exported to `ml/artifacts/v5_experiments/active_learning/iaa_disagreements_adjudication.csv`).",
        ]
    else:
        lines.append("No completed human annotations found yet in the blind review CSV. Complete labeling in the local Annotation Studio (`http://127.0.0.1:8765/annotate`) and re-run `score_human_iaa.py`.")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_adjudication_to_v2(report: Dict[str, Any], adjudication_csv: Path = DISAGREEMENT_CSV) -> Dict[str, Any]:
    """Apply completed human IAA & adjudicated disagreement decisions to isolated V2 dataset only.

    Strictly refuses to run until all 50 pack items have genuine human annotations,
    never modifies `test_v5_frozen` rows, and never touches `real_reddit_labeled_v1/`.
    """
    if report.get("status") != "COMPLETE":
        raise RuntimeError(
            f"Cannot apply adjudication while IAA status is '{report.get('status')}' "
            f"({report.get('reviewed_by_both')}/{report.get('total_pack_items')} reviewed)."
        )

    adjudications: Dict[str, Dict[str, str]] = {}
    if adjudication_csv.exists():
        with adjudication_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                did = (r.get("document_id") or "").strip()
                adj_status = (r.get("adjudicated_status") or "").strip().lower()
                if not adj_status:
                    continue
                if adj_status == "labeled":
                    sent = (r.get("adjudicated_sentiment") or "").strip().lower()
                    resp = (r.get("adjudicated_response") or "").strip().lower()
                    sol = (r.get("adjudicated_solution") or "").strip().lower()
                    if sol == "true":
                        sol = "solution"
                    elif sol == "false":
                        sol = "not_solution"
                    emos = sorted(set(parse_multilabel_str((r.get("adjudicated_emotions") or "").lower())))
                    iss = sorted(set(parse_multilabel_str((r.get("adjudicated_issues") or "").lower())))
                    if sent not in CANONICAL_LABELS["sentiment"]:
                        raise ValueError(f"{did}: invalid adjudicated_sentiment '{sent}'")
                    if resp not in CANONICAL_LABELS["response"]:
                        raise ValueError(f"{did}: invalid adjudicated_response '{resp}'")
                    if sol not in CANONICAL_LABELS["solution"]:
                        raise ValueError(f"{did}: invalid adjudicated_solution '{sol}'")
                    for e in emos:
                        if e not in CANONICAL_LABELS["emotion"]:
                            raise ValueError(f"{did}: invalid adjudicated_emotion '{e}'")
                    for i_lbl in iss:
                        if i_lbl not in CANONICAL_LABELS["issue"]:
                            raise ValueError(f"{did}: invalid adjudicated_issue '{i_lbl}'")
                    adjudications[did] = {
                        "status": "labeled",
                        "sentiment": sent,
                        "emotions": "|".join(emos),
                        "issues": "|".join(iss),
                        "response": resp,
                        "solution": sol,
                    }
                elif adj_status in {"abstain", "excluded", "uncertain"}:
                    adjudications[did] = {"status": adj_status}
                else:
                    raise ValueError(f"{did}: invalid adjudicated_status '{adj_status}'")

    v2_dir = DEFAULT_V2_MASTER.parent
    with DEFAULT_V2_MASTER.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    updated_rows = []
    abstained_moved = []
    n_updated = 0
    for r in rows:
        did = (r.get("document_id") or "").strip()
        if did in adjudications:
            if r.get("v2_split") == "test_v5_frozen":
                raise RuntimeError(f"Refusing to modify frozen V5 test row: {did}")
            dec = adjudications[did]
            if dec["status"] != "labeled":
                abstained_moved.append(did)
                continue
            r["sentiment"] = dec["sentiment"]
            r["emotions"] = dec["emotions"]
            r["issues"] = dec["issues"]
            r["response"] = dec["response"]
            r["solution"] = dec["solution"]
            r["annotation_source"] = "human_iaa_adjudicated_v2"
            n_updated += 1
        updated_rows.append(r)

    with DEFAULT_V2_MASTER.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(updated_rows)

    for task, col, hdr in (
        ("sentiment", "sentiment", "label"),
        ("response", "response", "label"),
        ("solution", "solution", "label"),
        ("emotion", "emotions", "labels"),
        ("issue", "issues", "labels"),
    ):
        with (v2_dir / f"{task}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["document_id", "post_id", "comment_id", "text", hdr])
            for r in updated_rows:
                w.writerow([r["document_id"], r["post_id"], r["comment_id"], r["text"], r[col]])

    summary = {
        "human_iaa_status": report["status"],
        "all_gates_passed": report.get("all_gates_passed", False),
        "n_adjudicated_updated": n_updated,
        "n_moved_to_abstain": len(abstained_moved),
        "abstained_ids": abstained_moved,
        "total_v2_rows_after_adjudication": len(updated_rows),
        "task_metrics": report.get("tasks", {}),
    }
    save_json(v2_dir / "human_iaa_verification.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", default=str(DEFAULT_HUMAN_CSV), help="Primary Human Studio CSV")
    parser.add_argument("--annotator-b", default="", help="Optional second Human Studio CSV (defaults to V2 Pass-1 rubric)")
    parser.add_argument("--no-write", action="store_true", help="Print JSON summary only without writing report files")
    parser.add_argument(
        "--apply-adjudication",
        action="store_true",
        help="After completing 50/50 review & adjudicating disagreements, apply verified updates to isolated V2 CSVs",
    )
    args = parser.parse_args()

    path_a = Path(args.annotator_a)
    assert path_a.exists(), f"Missing Annotator A CSV: {path_a}"
    cod_a = _load_studio_csv(path_a)

    if args.annotator_b:
        path_b = Path(args.annotator_b)
        assert path_b.exists(), f"Missing Annotator B CSV: {path_b}"
        cod_b = _load_studio_csv(path_b)
        rep = compute_iaa_report(cod_a, cod_b, label_a=path_a.name, label_b=path_b.name)
    else:
        cod_b = _load_v2_pass1_reference(set(cod_a.keys()))
        rep = compute_iaa_report(cod_a, cod_b, label_a="Human Reviewer (Blind Studio)", label_b="V2 Rubric Pass-1")

    if not args.no_write:
        write_iaa_outputs(rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "disagreements"}, indent=2))
    if not args.no_write:
        print(f"Disagreements exported: {len(rep.get('disagreements', []))} -> {DISAGREEMENT_CSV}")
        print(f"Report written -> {REPORT_MD}")

    if args.apply_adjudication:
        adj_res = apply_adjudication_to_v2(rep)
        print("Applied adjudication to V2 dataset:")
        print(json.dumps(adj_res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
