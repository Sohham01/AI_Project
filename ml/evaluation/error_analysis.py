"""Data-Driven Error Analysis & Calibration Reliability Module (pure Python + NumPy).

Computes per-task confusion diagnostics, empirically derived summaries of actual
misclassified samples (tokens, lengths, OOV ratios, confidence margins, snippets —
ZERO canned lookup descriptions), Expected Calibration Error (ECE) reliability
curves using well-populated 5-bin calibration intervals, confidence-tier accuracy,
and cross-architecture recovery / residual failure reports comparing Logistic
Regression, ANN (MLP), CNN (1D TextCNN), and MuRIL-DAPT v3 Transformer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.55

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was",
    "have", "has", "had", "not", "but", "from", "they", "their", "them", "will",
    "would", "could", "should", "what", "when", "where", "which", "about", "there",
    "here", "just", "like", "more", "some", "very", "only", "also", "into", "than",
    "then", "been", "being", "can", "all", "any", "out", "get", "got", "one", "two",
    "how", "why", "who", "its", "it's", "don't", "dont", "did", "does", "i'm", "im",
    "iim", "sambalpur", "reddit", "link", "https", "http", "www", "com",
})
_HINGLISH_MARKERS = frozenset({
    "hai", "hain", "nahi", "nhi", "kya", "kyu", "kyun", "kaise", "bhai", "bhaiya",
    "yaar", "toh", "bhi", "mein", "mera", "tera", "aur", "lekin", "agar", "accha",
    "achha", "theek", "karo", "karna", "karke", "gaya", "gayi", "gaye", "dekho",
    "bola", "mila", "mile", "chahiye", "paisa", "paise", "sahi", "bahut", "bohot",
    "zyada", "abhi", "pehle", "phir", "kuch", "koi", "kaun", "wala", "wali", "wale",
    "haan", "shukriya", "hoga", "raha", "rahi", "rahe", "hota", "jao", "rha",
    "matlab", "shayad", "bilkul", "thoda", "batao", "pata", "aayi", "aayegi",
    "aaegi", "baat", "yaha", "waha", "kitna", "kab", "apna", "khud", "lagta", "fir",
})


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


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(str(text or ""))]


def _empirical_text_traits(
    text: str,
    train_vocab: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    toks = _tokenize(text)
    wc = len(toks)
    hinglish_hits = [t for t in toks if t in _HINGLISH_MARKERS]
    oov_ratio = None
    if train_vocab and wc > 0:
        oov = sum(1 for t in toks if t not in train_vocab)
        oov_ratio = round(oov / wc, 3)
    return {
        "word_count": wc,
        "has_question": "?" in str(text or ""),
        "hinglish_tokens": sorted(set(hinglish_hits))[:4],
        "is_short": wc <= 8,
        "is_long": wc >= 35,
        "oov_ratio": oov_ratio,
    }


def _describe_pair_from_samples(
    true_lbl: str,
    pred_lbl: str,
    sample_texts: Sequence[str],
    sample_confs: Sequence[float],
) -> str:
    """Build a strictly empirical summary directly from the actual misclassified samples."""
    n = len(sample_texts)
    if n == 0:
        return f"Observed {true_lbl} -> {pred_lbl} confusion (sample texts unavailable)."

    traits = [_empirical_text_traits(t) for t in sample_texts]
    med_wc = int(round(float(np.median([tr["word_count"] for tr in traits]))))
    q_cnt = sum(1 for tr in traits if tr["has_question"])
    hm_cnt = sum(1 for tr in traits if tr["hinglish_tokens"])
    short_cnt = sum(1 for tr in traits if tr["is_short"])

    tok_counts: Counter = Counter()
    for t in sample_texts:
        for tok in set(_tokenize(t)):
            if len(tok) >= 3 and tok not in _STOPWORDS and not tok.isdigit():
                tok_counts[tok] += 1
    top_toks = [w for w, c in tok_counts.most_common(4) if c >= (2 if n >= 3 else 1)]

    parts: List[str] = [f"n={n} misclassified samples (median {med_wc} words"]
    if sample_confs:
        parts[0] += f", mean conf {float(np.mean(sample_confs)) * 100:.0f}%)"
    else:
        parts[0] += ")"

    flags: List[str] = []
    if short_cnt:
        flags.append(f"{short_cnt}/{n} terse (<=8 words)")
    if q_cnt:
        flags.append(f"{q_cnt}/{n} contain questions")
    if hm_cnt:
        flags.append(f"{hm_cnt}/{n} include Hinglish tokens")
    if flags:
        parts.append("; ".join(flags))
    if top_toks:
        parts.append(f"recurring terms: {', '.join(top_toks)}")

    excerpt = " ".join(str(sample_texts[0]).split())[:110]
    parts.append(f'e.g. "{excerpt}"')
    return " · ".join(parts)


def _empirical_failure_mode(
    traits: Dict[str, Any],
    conf: float,
    margin: float,
) -> tuple[str, str]:
    """Derive failure mode label and data-grounded reason strictly from sample statistics."""
    reasons: List[str] = [f"{traits['word_count']} words", f"conf {conf * 100:.0f}% (margin {margin * 100:.0f} pp)"]
    if traits.get("oov_ratio") is not None:
        reasons.append(f"{traits['oov_ratio'] * 100:.0f}% unseen vocab")
    if traits["hinglish_tokens"]:
        reasons.append(f"Hinglish markers: {', '.join(traits['hinglish_tokens'][:3])}")
    if traits["has_question"]:
        reasons.append("interrogative phrasing ('?')")

    if traits["hinglish_tokens"]:
        mode = "Code-mixed / Hinglish phrasing"
    elif traits["is_short"]:
        mode = "Short utterance (<=8 words)"
    elif (traits.get("oov_ratio") or 0.0) >= 0.45:
        mode = "High unseen-vocabulary shift"
    elif margin < 0.15:
        mode = "Close decision boundary (low margin)"
    elif traits["is_long"]:
        mode = "Multi-clause / mixed-intent comment"
    else:
        mode = "Semantic / pragmatic ambiguity"
    return mode, "Observed sample traits: " + "; ".join(reasons) + "."


def _is_prediction_correct(true_val: Any, pred_val: Any, multilabel: bool) -> bool:
    if multilabel:
        t_set = set(true_val) if isinstance(true_val, (list, set, tuple)) else ({str(true_val)} if true_val else set())
        p_set = set(pred_val) if isinstance(pred_val, (list, set, tuple)) else ({str(pred_val)} if pred_val else set())
        if not t_set and not p_set:
            return True
        if not t_set or not p_set:
            return False
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
    n_bins: int = 5,
    source_split: str = "heldout_real_test",
) -> Dict[str, Any]:
    """Compute ECE, MCE, Brier score, reliability bins, and confidence-tier accuracy."""
    probs = np.asarray(prob_matrix, dtype=np.float64)
    n_samples = len(y_true)
    if n_samples == 0 or probs.size == 0:
        return {
            "n_samples": 0,
            "source_split": source_split,
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "mean_confidence": 0.0,
            "empirical_accuracy": 0.0,
            "bins": [],
            "confidence_tiers": {},
        }

    lbl_idx = {str(l): i for i, l in enumerate(labels)}
    n_classes = len(labels)

    if multilabel:
        y_bin = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, row in enumerate(y_true):
            for l in row:
                if str(l) in lbl_idx:
                    y_bin[i, lbl_idx[str(l)]] = 1.0
        brier = float(np.mean((probs - y_bin) ** 2))
        # Textbook multi-label calibration: predicted positive probability P(y=1) vs empirical positive frequency y
        confs = probs.reshape(-1)
        accs = y_bin.reshape(-1)
        sample_confs = np.max(probs, axis=1)
        sample_accs = np.array(
            [
                1.0
                if _is_prediction_correct(
                    y_true[i],
                    [labels[j] for j in range(n_classes) if probs[i, j] >= 0.5],
                    multilabel=True,
                )
                else 0.0
                for i in range(n_samples)
            ],
            dtype=np.float64,
        )
    else:
        y_idx = np.array([lbl_idx.get(str(v), 0) for v in y_true], dtype=np.int64)
        y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        y_onehot[np.arange(n_samples), y_idx] = 1.0
        brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
        pred_idx = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        accs = (pred_idx == y_idx).astype(np.float64)
        sample_confs = confs
        sample_accs = accs

    # Equal-width reliability bins (default 5 bins of width 0.20 so each bin has statistical support)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total_pts = max(len(confs), 1)
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (confs >= lo) & (confs <= hi if b == n_bins - 1 else confs < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            avg_c = float(confs[mask].mean())
            avg_a = float(accs[mask].mean())
            gap = abs(avg_a - avg_c)
            ece += (cnt / total_pts) * gap
            mce = max(mce, gap)
        else:
            avg_c, avg_a, gap = None, None, None
        bins_out.append(
            {
                "bin": b,
                "lower": round(lo, 3),
                "upper": round(hi, 3),
                "count": cnt,
                "mean_confidence": round(avg_c, 4) if avg_c is not None else None,
                "empirical_accuracy": round(avg_a, 4) if avg_a is not None else None,
                "gap": round(gap, 4) if gap is not None else None,
            }
        )

    tier_buckets: Dict[str, List[Tuple[float, float]]] = {"high": [], "medium": [], "low": []}
    for sc, sa in zip(sample_confs, sample_accs):
        tier_buckets[classify_confidence_tier(float(sc))].append((float(sc), float(sa)))

    tiers_out: Dict[str, Dict[str, Any]] = {}
    for tier in ("high", "medium", "low"):
        pts = tier_buckets[tier]
        cnt = len(pts)
        acc_val = float(np.mean([p[1] for p in pts])) if cnt else None
        conf_val = float(np.mean([p[0] for p in pts])) if cnt else None
        tiers_out[tier] = {
            "count": cnt,
            "share": round(cnt / max(n_samples, 1), 4),
            "accuracy": round(acc_val, 4) if acc_val is not None else None,
            "error_rate": round(1.0 - acc_val, 4) if acc_val is not None else None,
            "mean_confidence": round(conf_val, 4) if conf_val is not None else None,
        }

    return {
        "n_samples": n_samples,
        "source_split": source_split,
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier), 4),
        "mean_confidence": round(float(np.mean(sample_confs)), 4),
        "empirical_accuracy": round(float(np.mean(sample_accs)), 4),
        "bins": bins_out,
        "confidence_tiers": tiers_out,
    }


def compute_confusion_and_top_pairs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[str],
    *,
    multilabel: bool = False,
    texts: Optional[Sequence[str]] = None,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute confusion matrix and empirically derived top failure pair summaries."""
    lbls = [str(l) for l in labels]
    n = len(y_true)
    if multilabel:
        fp_counts: Counter = Counter()
        fn_counts: Counter = Counter()
        co_err: Counter = Counter()
        pair_texts: Dict[tuple[str, str], List[str]] = defaultdict(list)
        pair_confs: Dict[tuple[str, str], List[float]] = defaultdict(list)
        sample_errors = 0
        for idx_i, (yt, yp) in enumerate(zip(y_true, y_pred)):
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
                    if texts is not None and idx_i < len(texts):
                        pair_texts[(t_lbl, p_lbl)].append(str(texts[idx_i]))
                    if probs is not None and idx_i < len(probs):
                        pair_confs[(t_lbl, p_lbl)].append(float(np.max(probs[idx_i])))

        top_pairs: List[Dict[str, Any]] = []
        for (t_lbl, p_lbl), cnt in co_err.most_common(6):
            diag = _describe_pair_from_samples(
                t_lbl, p_lbl, pair_texts[(t_lbl, p_lbl)], pair_confs[(t_lbl, p_lbl)]
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
    pair_texts_sl: Dict[tuple[str, str], List[str]] = defaultdict(list)
    pair_confs_sl: Dict[tuple[str, str], List[float]] = defaultdict(list)
    correct = 0
    for idx_i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        ts, ps = str(yt), str(yp)
        if ts in idx and ps in idx:
            mat[idx[ts]][idx[ps]] += 1
        if ts == ps:
            correct += 1
        else:
            err_pairs[(ts, ps)] += 1
            if texts is not None and idx_i < len(texts):
                pair_texts_sl[(ts, ps)].append(str(texts[idx_i]))
            if probs is not None and idx_i < len(probs):
                pair_confs_sl[(ts, ps)].append(float(np.max(probs[idx_i])))

    err_count = n - correct
    top_pairs = []
    for (t_lbl, p_lbl), cnt in err_pairs.most_common(6):
        diag = _describe_pair_from_samples(
            t_lbl, p_lbl, pair_texts_sl[(t_lbl, p_lbl)], pair_confs_sl[(t_lbl, p_lbl)]
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
    train_texts: Optional[Sequence[str]] = None,
    max_examples: int = 6,
) -> Dict[str, Any]:
    """Compare all 4 architectures on evaluation items with purely data-driven explanations."""
    n = len(probe_texts)
    train_vocab: Optional[Set[str]] = None
    if train_texts:
        train_vocab = {tok for txt in train_texts for tok in _tokenize(txt)}

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
        arch_margin: Dict[str, float] = {}

        for arch, preds in model_preds_by_arch.items():
            pv = preds[i]
            ok = _is_prediction_correct(true_v, pv, multilabel)
            arch_ok[arch] = ok
            if not ok:
                per_cat_errors[arch][cat] += 1
            arch_pred_fmt[arch] = list(pv) if multilabel else str(pv)
            probs_row = model_probs_by_arch[arch][i]
            arch_conf[arch] = round(float(np.max(probs_row)) if len(probs_row) else 0.0, 4)
            arch_margin[arch] = probability_margin(list(probs_row)) if len(probs_row) else 0.0

        t_ok = arch_ok.get(target_arch, False)
        present_baselines = [b for b in baseline_archs if b in arch_ok]
        b_oks = [arch_ok[b] for b in present_baselines]
        b_wrong = [b for b in present_baselines if not arch_ok[b]]
        b_wrong_count = len(b_wrong)

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

        traits = _empirical_text_traits(str(probe_texts[i]), train_vocab=train_vocab)
        mode_lbl, empirical_reason = _empirical_failure_mode(
            traits, arch_conf.get(target_arch, 0.0), arch_margin.get(target_arch, 0.0)
        )
        fmt_true = sorted(list(true_v)) if multilabel else str(true_v)

        if t_ok and b_wrong_count >= 1:
            failed_str = ", ".join(
                f"{b.split('_')[0]}->{arch_pred_fmt[b]}" for b in b_wrong
            )
            oov_note = (
                f", {traits['oov_ratio'] * 100:.0f}% unseen vocab"
                if traits.get("oov_ratio") is not None
                else ""
            )
            hl_note = (
                f", Hinglish: {', '.join(traits['hinglish_tokens'][:3])}"
                if traits["hinglish_tokens"]
                else ""
            )
            win_reason = (
                f"Transformer predicted {arch_pred_fmt[target_arch]} at {arch_conf[target_arch] * 100:.0f}% conf "
                f"({traits['word_count']} words{oov_note}{hl_note}), whereas {b_wrong_count}/{len(present_baselines)} "
                f"baselines missed ({failed_str})."
            )
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
                    "why_transformer_won": win_reason,
                }
            )
        elif not t_ok:
            failure_taxonomy_counts[mode_lbl] += 1
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
                    "failure_mode": mode_lbl,
                    "root_cause": empirical_reason,
                }
            )

    wins.sort(key=lambda x: (-x["baselines_failed"], -x["transformer_confidence"]))
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
