"""Structured Error Analysis & Calibration Reliability Module (pure Python + NumPy).

Computes per-task confusion diagnostics, top confused class pairs with root-cause
interpretations, Expected Calibration Error (ECE) reliability curves, confidence-tier
accuracy profiles, and cross-architecture probe recovery / residual failure reports
comparing Logistic Regression, ANN (MLP), CNN (1D TextCNN), and MuRIL-DAPT v3 Transformer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.55

_CONFUSION_DIAGNOSES: Dict[Tuple[str, str], str] = {
    ("advice", "solution"): "Actionable recommendation phrased without concrete step-by-step fix crosses the advice/solution boundary.",
    ("solution", "advice"): "Concrete procedural fix softened with hedging language ('try', 'maybe') looks like general advice.",
    ("opinion", "personal_experience"): "Evaluative stance mentions campus/exam context without explicit first-person event phrasing.",
    ("personal_experience", "opinion"): "First-person anecdote concludes with a strong value judgment, pulling lexical cues toward opinion.",
    ("agreement", "opinion"): "Short endorsement reply lacks explicit lexical agreement markers ('agreed', '+1').",
    ("counterargument", "opinion"): "Polite pushback without explicit negation words resembles neutral opinion.",
    ("neutral", "negative"): "Factual complaint or anxious exam query contains negative-polarity domain terms ('waitlist', 'low score').",
    ("positive", "neutral"): "Understated or Hinglish praise lacks high-weight English sentiment adjectives.",
    ("negative", "neutral"): "Deadpan or sarcastic criticism uses surface-neutral wording.",
    ("not_solution", "solution"): "Descriptive campus/exam process mentions actionable verbs without solving the poster's problem.",
    ("solution", "not_solution"): "Terse or code-mixed fix omits explicit imperative action markers.",
}

_CATEGORY_ERROR_TAXONOMY: Dict[str, Tuple[str, str]] = {
    "code_mixed_language": (
        "Hinglish / code-mix semantics",
        "Romanized Hindi-English tokens fall out of bag-of-words vocabularies; multilingual MuRIL pretraining bridges cross-script meaning.",
    ),
    "paraphrased_comments": (
        "Lexical paraphrase shift",
        "Low surface n-gram overlap defeats linear/ANN/CNN bag-of-words features; contextual self-attention preserves semantic intent.",
    ),
    "informal_language": (
        "Informal Reddit slang & abbreviations",
        "Domain acronyms (WL, CAP, SAP, GEM, LPA) and casual phrasing benefit from Reddit MLM domain adaptation (DAPT).",
    ),
    "parent_post_context": (
        "Parent-post dependency",
        "Short comment relies on the thread title to disambiguate the underlying issue or stance.",
    ),
    "spelling_errors": (
        "Typos & character distortion",
        "Character n-grams and subword tokenization recover distorted spellings where whole-word CNN embeddings map to <UNK>.",
    ),
    "short_comments": (
        "Terse / low-token utterance",
        "Very short replies (2-5 words) carry sparse lexical evidence and naturally sit in lower confidence bands.",
    ),
    "ambiguous_responses": (
        "Boundary / dual-intent phrasing",
        "Reply blends two adjacent classes (e.g. empathetic experience + soft advice); should route to human annotation review.",
    ),
    "unseen_templates": (
        "Unseen syntactic structure",
        "Novel phrasing pattern outside training templates tests compositional generalization beyond memorized phrases.",
    ),
}


def classify_confidence_tier(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def probability_margin(probs: Sequence[float]) -> float:
    if not probs:
        return 0.0
    ordered = sorted((float(p) for p in probs), reverse=True)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    return round(max(0.0, ordered[0] - ordered[1]), 4)


def _is_prediction_correct(true_val: Any, pred_val: Any, multilabel: bool) -> bool:
    if multilabel:
        t_set = set(true_val) if isinstance(true_val, (list, set, tuple)) else ({str(true_val)} if true_val else set())
        p_set = set(pred_val) if isinstance(pred_val, (list, set, tuple)) else ({str(pred_val)} if pred_val else set())
        if not t_set and not p_set:
            return True
        if not t_set or not p_set:
            return False
        # Multi-label sample match: exact set OR >= 0.5 Jaccard overlap
        inter = len(t_set & p_set)
        union = len(t_set | p_set)
        return (inter / max(union, 1)) >= 0.5
    return str(true_val) == str(pred_val)


def compute_calibration_metrics(
    y_true: Sequence[Any],
    prob_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    multilabel: bool = False,
    n_bins: int = 8,
) -> Dict[str, Any]:
    """Compute ECE, MCE, Brier score, reliability bins, and confidence-tier accuracy."""
    probs = np.asarray(prob_matrix, dtype=np.float64)
    n_samples = len(y_true)
    if n_samples == 0 or probs.size == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "mean_confidence": 0.0,
            "empirical_accuracy": 0.0,
            "bins": [],
            "confidence_tiers": {},
        }

    lbl_to_idx = {str(lbl): i for i, lbl in enumerate(labels)}
    if multilabel:
        # Flatten label-level probabilities vs binary ground truth for true probabilistic calibration
        conf_flat: List[float] = []
        corr_flat: List[float] = []
        sample_confs: List[float] = []
        sample_corrs: List[float] = []
        brier_terms: List[float] = []
        for i, true_items in enumerate(y_true):
            t_set = set(true_items) if isinstance(true_items, (list, set, tuple)) else set()
            row = probs[i]
            top_idx = int(np.argmax(row))
            top_p = float(row[top_idx])
            sample_confs.append(top_p)
            sample_corrs.append(1.0 if str(labels[top_idx]) in t_set else 0.0)
            for j, lbl in enumerate(labels):
                p = float(np.clip(row[j], 0.0, 1.0))
                y_bin = 1.0 if str(lbl) in t_set else 0.0
                brier_terms.append((p - y_bin) ** 2)
                # Confidence of binary decision (p if predicted positive else 1-p)
                pred_pos = p >= 0.5
                conf_flat.append(p if pred_pos else (1.0 - p))
                corr_flat.append(1.0 if (pred_pos == (y_bin == 1.0)) else 0.0)
        confs = np.asarray(conf_flat, dtype=np.float64)
        corrs = np.asarray(corr_flat, dtype=np.float64)
        s_confs = np.asarray(sample_confs, dtype=np.float64)
        s_corrs = np.asarray(sample_corrs, dtype=np.float64)
        brier = float(np.mean(brier_terms)) if brier_terms else 0.0
    else:
        y_idx = np.array([lbl_to_idx[str(y)] for y in y_true], dtype=np.int64)
        pred_idx = np.argmax(probs, axis=1)
        confs = probs[np.arange(n_samples), pred_idx]
        corrs = (pred_idx == y_idx).astype(np.float64)
        s_confs = confs
        s_corrs = corrs
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), y_idx] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total_pts = max(len(confs), 1)
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (confs >= lo) & (confs <= hi if b == n_bins - 1 else confs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(
                {
                    "bin_idx": b,
                    "lower": round(lo, 3),
                    "upper": round(hi, 3),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_accuracy": None,
                    "gap": None,
                }
            )
            continue
        m_conf = float(np.mean(confs[mask]))
        m_acc = float(np.mean(corrs[mask]))
        gap = abs(m_conf - m_acc)
        ece += (cnt / total_pts) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin_idx": b,
                "lower": round(lo, 3),
                "upper": round(hi, 3),
                "count": cnt,
                "mean_confidence": round(m_conf, 4),
                "empirical_accuracy": round(m_acc, 4),
                "gap": round(gap, 4),
            }
        )

    tiers: Dict[str, Dict[str, Any]] = {}
    for tier_name, lo, hi in (
        ("high", CONFIDENCE_HIGH_THRESHOLD, 1.01),
        ("medium", CONFIDENCE_MEDIUM_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD),
        ("low", 0.0, CONFIDENCE_MEDIUM_THRESHOLD),
    ):
        mask = (s_confs >= lo) & (s_confs < hi)
        cnt = int(mask.sum())
        tiers[tier_name] = {
            "count": cnt,
            "share": round(cnt / max(len(s_confs), 1), 4),
            "accuracy": round(float(np.mean(s_corrs[mask])), 4) if cnt > 0 else None,
            "error_rate": round(1.0 - float(np.mean(s_corrs[mask])), 4) if cnt > 0 else None,
            "mean_confidence": round(float(np.mean(s_confs[mask])), 4) if cnt > 0 else None,
        }

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "mean_confidence": round(float(np.mean(s_confs)), 4),
        "empirical_accuracy": round(float(np.mean(s_corrs)), 4),
        "bins": bins,
        "confidence_tiers": tiers,
    }


def compute_confusion_and_top_pairs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[str],
    *,
    multilabel: bool = False,
) -> Dict[str, Any]:
    """Compute confusion matrix and ranked top failure pairs / label error diagnostics."""
    lbls = [str(l) for l in labels]
    n = len(y_true)
    if multilabel:
        fp_counts: Counter = Counter()
        fn_counts: Counter = Counter()
        co_err: Counter = Counter()
        sample_errors = 0
        for yt, yp in zip(y_true, y_pred):
            t_set = set(str(x) for x in yt)
            p_set = set(str(x) for x in yp)
            if t_set != p_set:
                sample_errors += 1
            for lbl in p_set - t_set:
                fp_counts[lbl] += 1
            for lbl in t_set - p_set:
                fn_counts[lbl] += 1
            for t_lbl in t_set - p_set:
                for p_lbl in p_set - t_set:
                    co_err[(t_lbl, p_lbl)] += 1

        top_pairs: List[Dict[str, Any]] = []
        for (t_lbl, p_lbl), cnt in co_err.most_common(6):
            diag = _CONFUSION_DIAGNOSES.get(
                (t_lbl, p_lbl),
                f"Multi-label co-occurrence confusion between '{t_lbl}' (missed) and '{p_lbl}' (over-predicted).",
            )
            top_pairs.append(
                {
                    "true_label": t_lbl,
                    "predicted_label": p_lbl,
                    "count": int(cnt),
                    "share_of_errors": round(cnt / max(sample_errors, 1), 4),
                    "diagnosis": diag,
                }
            )
        per_label_errors = {
            lbl: {"false_positives": int(fp_counts[lbl]), "false_negatives": int(fn_counts[lbl])}
            for lbl in lbls
        }
        return {
            "multilabel": True,
            "labels": lbls,
            "matrix": None,
            "n_samples": n,
            "error_count": sample_errors,
            "error_rate": round(sample_errors / max(n, 1), 4),
            "top_confusions": top_pairs,
            "per_label_errors": per_label_errors,
        }

    idx = {l: i for i, l in enumerate(lbls)}
    mat = [[0 for _ in lbls] for _ in lbls]
    err_pairs: Counter = Counter()
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        ts, ps = str(yt), str(yp)
        if ts in idx and ps in idx:
            mat[idx[ts]][idx[ps]] += 1
        if ts == ps:
            correct += 1
        else:
            err_pairs[(ts, ps)] += 1

    err_count = n - correct
    top_pairs = []
    for (t_lbl, p_lbl), cnt in err_pairs.most_common(6):
        diag = _CONFUSION_DIAGNOSES.get(
            (t_lbl, p_lbl),
            f"Samples truly belonging to '{t_lbl}' were misrouted to '{p_lbl}' due to overlapping surface wording.",
        )
        top_pairs.append(
            {
                "true_label": t_lbl,
                "predicted_label": p_lbl,
                "count": int(cnt),
                "share_of_errors": round(cnt / max(err_count, 1), 4),
                "diagnosis": diag,
            }
        )

    return {
        "multilabel": False,
        "labels": lbls,
        "matrix": mat,
        "n_samples": n,
        "correct": correct,
        "error_count": err_count,
        "error_rate": round(err_count / max(n, 1), 4),
        "top_confusions": top_pairs,
    }


def analyze_architecture_recoveries_and_errors(
    *,
    task_name: str,
    probe_texts: Sequence[str],
    probe_categories: Sequence[str],
    y_true: Sequence[Any],
    model_preds_by_arch: Dict[str, Sequence[Any]],
    model_probs_by_arch: Dict[str, np.ndarray],
    labels: Sequence[str],
    multilabel: bool = False,
    target_arch: str = "transformer_muril_dapt_v3",
    baseline_archs: Sequence[str] = ("logistic_regression", "ann_mlp", "cnn_textcnn"),
    max_examples: int = 6,
) -> Dict[str, Any]:
    """Compare all 4 architectures on unseen probes to expose WHY Transformer wins & residual failure modes."""
    n = len(probe_texts)
    per_cat_totals: Counter = Counter(probe_categories)
    per_cat_errors: Dict[str, Counter] = {arch: Counter() for arch in model_preds_by_arch}
    agreement = {
        "all_correct": 0,
        "transformer_only_win": 0,
        "transformer_beats_majority_baselines": 0,
        "baseline_beats_transformer": 0,
        "all_failed": 0,
    }

    wins: List[Dict[str, Any]] = []
    residuals: List[Dict[str, Any]] = []
    failure_taxonomy_counts: Counter = Counter()

    for i in range(n):
        cat = str(probe_categories[i])
        true_v = y_true[i]
        arch_ok: Dict[str, bool] = {}
        arch_pred_fmt: Dict[str, Any] = {}
        arch_conf: Dict[str, float] = {}

        for arch, preds in model_preds_by_arch.items():
            pv = preds[i]
            ok = _is_prediction_correct(true_v, pv, multilabel)
            arch_ok[arch] = ok
            if not ok:
                per_cat_errors[arch][cat] += 1
            arch_pred_fmt[arch] = list(pv) if multilabel else str(pv)
            probs_row = model_probs_by_arch[arch][i]
            arch_conf[arch] = round(float(np.max(probs_row)) if len(probs_row) else 0.0, 4)

        t_ok = arch_ok.get(target_arch, False)
        present_baselines = [b for b in baseline_archs if b in arch_ok]
        b_oks = [arch_ok[b] for b in present_baselines]
        b_wrong_count = sum(1 for ok in b_oks if not ok)

        if t_ok and all(b_oks):
            agreement["all_correct"] += 1
        elif t_ok and b_wrong_count == len(present_baselines) and present_baselines:
            agreement["transformer_only_win"] += 1
            agreement["transformer_beats_majority_baselines"] += 1
        elif t_ok and b_wrong_count >= max(1, (len(present_baselines) + 1) // 2):
            agreement["transformer_beats_majority_baselines"] += 1
        elif (not t_ok) and any(b_oks):
            agreement["baseline_beats_transformer"] += 1
        elif (not t_ok) and not any(b_oks):
            agreement["all_failed"] += 1

        tax_label, tax_reason = _CATEGORY_ERROR_TAXONOMY.get(
            cat, ("Domain shift / wording ambiguity", "Unseen phrasing outside training distribution.")
        )
        fmt_true = sorted(list(true_v)) if multilabel else str(true_v)

        if t_ok and b_wrong_count >= 1:
            wins.append(
                {
                    "probe_index": i,
                    "category": cat,
                    "text": str(probe_texts[i])[:220],
                    "true_label": fmt_true,
                    "transformer_prediction": arch_pred_fmt[target_arch],
                    "transformer_confidence": arch_conf[target_arch],
                    "baselines_failed": b_wrong_count,
                    "baseline_predictions": {b: arch_pred_fmt[b] for b in present_baselines},
                    "why_transformer_won": tax_reason,
                }
            )
        elif not t_ok:
            failure_taxonomy_counts[tax_label] += 1
            residuals.append(
                {
                    "probe_index": i,
                    "category": cat,
                    "text": str(probe_texts[i])[:220],
                    "true_label": fmt_true,
                    "transformer_prediction": arch_pred_fmt[target_arch],
                    "transformer_confidence": arch_conf[target_arch],
                    "confidence_tier": classify_confidence_tier(arch_conf[target_arch]),
                    "baseline_predictions": {b: arch_pred_fmt[b] for b in present_baselines},
                    "failure_mode": tax_label,
                    "root_cause": tax_reason,
                }
            )

    # Sort wins so cases where ALL 3 baselines failed appear first, then highest transformer confidence
    wins.sort(key=lambda x: (-x["baselines_failed"], -x["transformer_confidence"]))
    # Sort residual errors so high-confidence errors (most important to audit!) appear first
    residuals.sort(key=lambda x: -x["transformer_confidence"])

    per_category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, total in sorted(per_cat_totals.items()):
        per_category_summary[cat] = {
            "n_probes": int(total),
            "error_rates": {
                arch: round(per_cat_errors[arch][cat] / max(total, 1), 4)
                for arch in model_preds_by_arch
            },
            "accuracies": {
                arch: round(1.0 - (per_cat_errors[arch][cat] / max(total, 1)), 4)
                for arch in model_preds_by_arch
            },
        }

    return {
        "task": task_name,
        "n_probes": n,
        "agreement_summary": agreement,
        "per_category_breakdown": per_category_summary,
        "failure_taxonomy": [
            {"mode": mode, "count": int(cnt)} for mode, cnt in failure_taxonomy_counts.most_common()
        ],
        "transformer_wins": wins[:max_examples],
        "residual_errors": residuals[:max_examples],
    }
