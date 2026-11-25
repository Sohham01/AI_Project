"""Sprint P0-C — Multi-Annotator Reliability, Blind Review, Disagreement Tracking & Adjudication Framework.

Supports:
  - K >= 2 independent annotators across multiple versioned annotation rounds
  - Strict blind-sheet verification (no peer labels, no model predictions)
  - Single-label reliability: raw agreement %, pairwise Cohen's kappa, Fleiss' kappa,
    and general multi-coder Krippendorff's alpha (nominal, handles missingness)
  - Multi-label reliability (emotion & issue): per-class binary agreement, per-class
    kappa/alpha with low-support & unreliable-class warnings, micro/macro binary alpha,
    exact-set match, mean Jaccard, and MASI-weighted Krippendorff's alpha
  - Non-destructive preservation of all original annotator judgments + adjudicated labels
    tagged with taxonomy and guideline versions.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score

TAXONOMY_VERSION = "rci_taxonomy_v2.0"
GUIDELINE_VERSION = "ANNOTATION_GUIDELINES_v2.0"

CANONICAL_RESPONSE_ORDER: Tuple[str, ...] = (
    "agreement",
    "personal_experience",
    "opinion",
    "advice",
    "solution",
    "counterargument",
)

SINGLE_LABEL_TASKS: Dict[str, Tuple[str, ...]] = {
    "sentiment": ("negative", "neutral", "positive"),
    "response": CANONICAL_RESPONSE_ORDER,
    "solution": ("not_solution", "solution"),
}

MULTI_LABEL_TASKS: Dict[str, Tuple[str, ...]] = {
    "issue": (
        "academic_workload",
        "administrative_communication",
        "campus_infrastructure",
        "curriculum_teaching",
        "extracurriculars",
        "financial_concerns",
        "hostel_facilities",
        "placements",
    ),
    "emotion": (
        "confidence",
        "curiosity",
        "frustration",
        "gratitude",
        "hope",
        "stress",
    ),
}

DEFAULT_IAA_THRESHOLDS: Dict[str, float] = {
    "sentiment": 0.70,
    "solution": 0.70,
    "response": 0.65,
    "issue": 0.65,
    "emotion": 0.65,
}


@dataclass
class AnnotationSubmission:
    """Single annotator judgment on one document preserving provenance and guideline version."""

    document_id: str
    annotator_id: str
    round_id: str = "round_1"
    review_status: str = "labeled"  # labeled | uncertain | abstain | excluded | unreviewed
    abstain_reason: str = ""
    sentiment: str = ""
    emotions: Set[str] = field(default_factory=set)
    issues: Set[str] = field(default_factory=set)
    response: str = ""
    solution: str = ""
    submitted_at: str = ""
    taxonomy_version: str = TAXONOMY_VERSION
    guideline_version: str = GUIDELINE_VERSION
    notes: str = ""


def verify_blind_sheet(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Verify a candidate annotation sheet is strictly blind (zero peer or model labels)."""
    errors: List[str] = []
    if not rows:
        return False, ["empty_candidate_sheet"]

    forbidden_cols = {
        k
        for row in rows
        for k in row.keys()
        if any(
            k.lower().startswith(pfx)
            for pfx in ("peer_", "annotator_a_", "annotator_b_", "v1_", "v2_", "pred_", "model_", "gold_")
        )
    }
    if forbidden_cols:
        errors.append(f"non_blind_columns_present:{sorted(forbidden_cols)}")

    label_cols = ("sentiment", "emotions", "issues", "response_type", "response", "is_solution", "solution", "review_status")
    for idx, row in enumerate(rows, start=1):
        for col in label_cols:
            if col in row and str(row.get(col) or "").strip() not in {"", "unreviewed"}:
                errors.append(f"row_{idx}_non_blank_label:{col}")
                if len(errors) >= 10:
                    return False, errors
    return len(errors) == 0, errors


def safe_cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    if len(set(a) | set(b)) <= 1:
        return 1.0 if list(a) == list(b) else 0.0
    val = float(cohen_kappa_score(list(a), list(b)))
    return 0.0 if np.isnan(val) else val


def fleiss_kappa(ratings_by_unit: Sequence[Sequence[Any]], categories: Sequence[Any]) -> float:
    """Exact Fleiss (1971) multi-rater kappa for balanced ratings across K >= 2 coders."""
    if not ratings_by_unit or not categories:
        return 0.0
    cats = list(categories)
    c_idx = {c: i for i, c in enumerate(cats)}
    valid_units = [u for u in ratings_by_unit if all(v in c_idx for v in u) and len(u) >= 2]
    if not valid_units:
        return 0.0

    k_raters = len(valid_units[0])
    if any(len(u) != k_raters for u in valid_units):
        # Fleiss requires equal rater counts per item; caller uses Krippendorff's alpha for unbalanced missingness
        min_k = min(len(u) for u in valid_units)
        valid_units = [u[:min_k] for u in valid_units]
        k_raters = min_k

    n_units = len(valid_units)
    mat = np.zeros((n_units, len(cats)), dtype=np.float64)
    for row_i, unit in enumerate(valid_units):
        for val in unit:
            mat[row_i, c_idx[val]] += 1.0

    p_j = mat.sum(axis=0) / (n_units * k_raters)
    p_e = float(np.sum(p_j ** 2))
    p_i = (np.sum(mat ** 2, axis=1) - k_raters) / (k_raters * (k_raters - 1.0))
    p_bar = float(np.mean(p_i))
    denom = 1.0 - p_e
    if denom <= 1e-12:
        return 1.0 if abs(p_bar - 1.0) <= 1e-12 else 0.0
    return float((p_bar - p_e) / denom)


def krippendorff_alpha_nominal(ratings_by_unit: Sequence[Sequence[Optional[Any]]]) -> float:
    """General coincidence-matrix Krippendorff's alpha for K >= 2 coders (supports missing None values)."""
    coincidences: Dict[Tuple[Any, Any], float] = defaultdict(float)
    marginals: Counter = Counter()
    total_pairable = 0.0

    for unit in ratings_by_unit:
        vals = [v for v in unit if v is not None and v != ""]
        m_u = len(vals)
        if m_u < 2:
            continue
        weight = 1.0 / float(m_u - 1)
        for i in range(m_u):
            marginals[vals[i]] += 1.0
            total_pairable += 1.0
            for j in range(m_u):
                if i == j:
                    continue
                coincidences[(vals[i], vals[j])] += weight

    if total_pairable <= 1.0:
        return 0.0

    do = sum(w for (c, k), w in coincidences.items() if c != k) / total_pairable
    de = 1.0 - (sum(nc * (nc - 1.0) for nc in marginals.values()) / (total_pairable * (total_pairable - 1.0)))
    if de <= 1e-12:
        return 1.0 if do <= 1e-12 else 0.0
    return float(1.0 - (do / de))


def masi_distance(set_a: Set[str], set_b: Set[str]) -> float:
    """Passonneau (2006) MASI set-distance in [0, 1]."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    jacc = inter / float(union)
    if set_a == set_b:
        m = 1.0
    elif set_a.issubset(set_b) or set_b.issubset(set_a):
        m = 2.0 / 3.0
    elif inter > 0:
        m = 1.0 / 3.0
    else:
        m = 0.0
    return float(1.0 - m * jacc)


def krippendorff_alpha_masi(sets_by_unit: Sequence[Sequence[Optional[Set[str]]]]) -> float:
    """Multi-coder Krippendorff's alpha using MASI distance for K >= 2 coders."""
    do_sum = 0.0
    pool: List[Set[str]] = []
    pairable_units = 0

    for unit in sets_by_unit:
        vals = [v for v in unit if v is not None]
        m_u = len(vals)
        if m_u < 2:
            continue
        pairable_units += 1
        unit_do = 0.0
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    unit_do += masi_distance(vals[i], vals[j])
        do_sum += unit_do / float(m_u * (m_u - 1))
        pool.extend(vals)

    n = len(pool)
    if n < 2 or pairable_units == 0:
        return 0.0
    do = do_sum / float(pairable_units)
    de_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            de_sum += masi_distance(pool[i], pool[j])
    de = de_sum / float(n * (n - 1) / 2.0)
    if de <= 1e-12:
        return 1.0 if do <= 1e-12 else 0.0
    return float(1.0 - (do / de))


def compute_single_label_iaa(
    submissions_by_annotator: Dict[str, Dict[str, AnnotationSubmission]],
    task_name: str,
    valid_labels: Sequence[str],
    *,
    gate_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute comprehensive single-label IAA across K >= 2 annotators."""
    coders = sorted(submissions_by_annotator.keys())
    if len(coders) < 2:
        return {"task": task_name, "status": "INSUFFICIENT_ANNOTATORS", "n_annotators": len(coders)}

    all_doc_ids = sorted({did for m in submissions_by_annotator.values() for did in m})
    ratings_by_unit: List[List[Optional[str]]] = []
    complete_units: List[List[str]] = []
    complete_doc_ids: List[str] = []

    for did in all_doc_ids:
        row_vals: List[Optional[str]] = []
        for c in coders:
            sub = submissions_by_annotator[c].get(did)
            if sub and sub.review_status == "labeled":
                val = getattr(sub, task_name, "")
                row_vals.append(val if val in valid_labels else None)
            else:
                row_vals.append(None)
        if sum(1 for v in row_vals if v is not None) >= 2:
            ratings_by_unit.append(row_vals)
        if all(v is not None for v in row_vals):
            complete_units.append([str(v) for v in row_vals])
            complete_doc_ids.append(did)

    if not ratings_by_unit:
        return {
            "task": task_name,
            "paradigm": "single_label",
            "status": "PENDING_HUMAN_ANNOTATIONS",
            "n_annotators": len(coders),
            "n_compared": 0,
        }

    # Exact all-coder raw agreement + pairwise stats
    raw_all_pct = (
        round(100.0 * sum(1 for u in complete_units if len(set(u)) == 1) / len(complete_units), 2)
        if complete_units
        else 0.0
    )

    pairwise_kappa: Dict[str, float] = {}
    pairwise_raw: Dict[str, float] = {}
    for c1, c2 in itertools.combinations(coders, 2):
        idx1, idx2 = coders.index(c1), coders.index(c2)
        pairs = [(u[idx1], u[idx2]) for u in ratings_by_unit if u[idx1] is not None and u[idx2] is not None]
        if pairs:
            va = [p[0] for p in pairs]
            vb = [p[1] for p in pairs]
            key = f"{c1}__vs__{c2}"
            pairwise_kappa[key] = round(safe_cohen_kappa(va, vb), 4)
            pairwise_raw[key] = round(100.0 * sum(1 for x, y in pairs if x == y) / len(pairs), 2)

    mean_kappa = round(float(np.mean(list(pairwise_kappa.values()))) if pairwise_kappa else 0.0, 4)
    mean_pair_raw = round(float(np.mean(list(pairwise_raw.values()))) if pairwise_raw else 0.0, 2)
    f_kappa = round(fleiss_kappa(complete_units, valid_labels), 4) if complete_units else 0.0
    k_alpha = round(krippendorff_alpha_nominal(ratings_by_unit), 4)

    # Per-class one-vs-rest summary + low support check
    per_class: Dict[str, Any] = {}
    unreliable_classes: List[str] = []
    for lbl in valid_labels:
        bin_units = [[(1 if v == lbl else 0) if v is not None else None for v in u] for u in ratings_by_unit]
        bin_alpha = round(krippendorff_alpha_nominal(bin_units), 4)
        bin_complete = [[(1 if v == lbl else 0) for v in u] for u in complete_units]
        bin_fleiss = round(fleiss_kappa(bin_complete, [0, 1]), 4) if bin_complete else 0.0
        counts_per_coder = {
            c: sum(1 for u in ratings_by_unit if u[idx] == lbl) for idx, c in enumerate(coders)
        }
        tot_pos = sum(counts_per_coder.values())
        if tot_pos > 0 and bin_alpha < 0.65:
            unreliable_classes.append(lbl)
        per_class[lbl] = {
            "counts_by_annotator": counts_per_coder,
            "total_assignments": tot_pos,
            "binary_fleiss_kappa": bin_fleiss,
            "binary_krippendorff_alpha": bin_alpha,
            "low_support_warning": tot_pos < 5,
        }

    # First-pair confusion matrix when >= 2 coders
    l2i = {l: i for i, l in enumerate(valid_labels)}
    cm_agg = np.zeros((len(valid_labels), len(valid_labels)), dtype=int)
    for u in ratings_by_unit:
        for i in range(len(coders)):
            for j in range(i + 1, len(coders)):
                if u[i] is not None and u[j] is not None:
                    cm_agg[l2i[str(u[i])], l2i[str(u[j])]] += 1

    thresh = gate_threshold if gate_threshold is not None else DEFAULT_IAA_THRESHOLDS.get(task_name, 0.65)
    primary_stat = mean_kappa if len(coders) == 2 else f_kappa
    return {
        "task": task_name,
        "paradigm": "single_label",
        "status": "COMPLETE" if len(complete_units) >= len(all_doc_ids) else "PARTIAL",
        "n_annotators": len(coders),
        "n_units_with_2plus_ratings": len(ratings_by_unit),
        "n_complete_all_coders": len(complete_units),
        "raw_exact_agreement_pct": raw_all_pct,
        "mean_pairwise_agreement_pct": mean_pair_raw,
        "cohen_kappa": mean_kappa,
        "pairwise_cohen_kappa": pairwise_kappa,
        "fleiss_kappa": f_kappa,
        "krippendorff_alpha_nominal": k_alpha,
        "gate_threshold": thresh,
        "gate_passed": bool(primary_stat >= thresh),
        "unreliable_classes": unreliable_classes,
        "per_class": per_class,
        "confusion_matrix_labels": list(valid_labels),
        "pairwise_confusion_matrix": cm_agg.tolist(),
    }


def compute_multilabel_iaa(
    submissions_by_annotator: Dict[str, Dict[str, AnnotationSubmission]],
    task_name: str,
    valid_labels: Sequence[str],
    *,
    attr_name: Optional[str] = None,
    gate_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute per-class + aggregate multi-label IAA without hiding weak classes in one average."""
    coders = sorted(submissions_by_annotator.keys())
    if len(coders) < 2:
        return {"task": task_name, "status": "INSUFFICIENT_ANNOTATORS", "n_annotators": len(coders)}

    field_name = attr_name or ("issues" if task_name == "issue" else "emotions")
    all_doc_ids = sorted({did for m in submissions_by_annotator.values() for did in m})

    sets_by_unit: List[List[Optional[Set[str]]]] = []
    complete_sets: List[List[Set[str]]] = []
    for did in all_doc_ids:
        row: List[Optional[Set[str]]] = []
        for c in coders:
            sub = submissions_by_annotator[c].get(did)
            if sub and sub.review_status == "labeled":
                raw_set = getattr(sub, field_name, set())
                row.append({x for x in raw_set if x in valid_labels})
            else:
                row.append(None)
        if sum(1 for v in row if v is not None) >= 2:
            sets_by_unit.append(row)
        if all(v is not None for v in row):
            complete_sets.append([set(v) for v in row if v is not None])

    if not sets_by_unit:
        return {
            "task": task_name,
            "paradigm": "multi_label",
            "status": "PENDING_HUMAN_ANNOTATIONS",
            "n_annotators": len(coders),
            "n_compared": 0,
        }

    # Exact set & pairwise Jaccard
    exact_set_pct = (
        round(100.0 * sum(1 for u in complete_sets if all(s == u[0] for s in u[1:])) / len(complete_sets), 2)
        if complete_sets
        else 0.0
    )
    pair_jaccards: List[float] = []
    for u in sets_by_unit:
        present = [s for s in u if s is not None]
        for sa, sb in itertools.combinations(present, 2):
            pair_jaccards.append(1.0 if (not sa and not sb) else len(sa & sb) / float(len(sa | sb)))

    per_label: Dict[str, Any] = {}
    flat_units: List[List[Optional[int]]] = []
    active_alphas: List[float] = []
    unreliable_classes: List[str] = []
    low_support_classes: List[str] = []

    for lbl in valid_labels:
        bin_units: List[List[Optional[int]]] = [
            [(1 if lbl in s else 0) if s is not None else None for s in u]
            for u in sets_by_unit
        ]
        flat_units.extend(bin_units)
        bin_complete = [[(1 if lbl in s else 0) for s in u] for u in complete_sets]

        pair_raw_hits = 0
        pair_total = 0
        both_pos = 0
        first_only = 0
        second_only = 0
        neither = 0
        pair_kappas: List[float] = []

        for c1_i, c2_i in itertools.combinations(range(len(coders)), 2):
            va = [u[c1_i] for u in bin_units if u[c1_i] is not None and u[c2_i] is not None]
            vb = [u[c2_i] for u in bin_units if u[c1_i] is not None and u[c2_i] is not None]
            if va:
                pair_kappas.append(safe_cohen_kappa(va, vb))
                for x, y in zip(va, vb):
                    pair_total += 1
                    if x == y:
                        pair_raw_hits += 1
                    if x == 1 and y == 1:
                        both_pos += 1
                    elif x == 1 and y == 0:
                        first_only += 1
                    elif x == 0 and y == 1:
                        second_only += 1
                    else:
                        neither += 1

        alpha_lbl = round(krippendorff_alpha_nominal(bin_units), 4)
        fleiss_lbl = round(fleiss_kappa(bin_complete, [0, 1]), 4) if bin_complete else 0.0
        mean_k_lbl = round(float(np.mean(pair_kappas)) if pair_kappas else 0.0, 4)
        raw_lbl_pct = round(100.0 * pair_raw_hits / pair_total, 2) if pair_total else 0.0
        counts_by_coder = {
            coders[idx]: sum(1 for u in bin_units if u[idx] == 1) for idx in range(len(coders))
        }
        tot_pos = sum(counts_by_coder.values())
        low_sup = tot_pos < 5
        if low_sup:
            low_support_classes.append(lbl)
        if tot_pos > 0:
            active_alphas.append(alpha_lbl)
            if alpha_lbl < 0.65:
                unreliable_classes.append(lbl)

        per_label[lbl] = {
            "counts_by_annotator": counts_by_coder,
            "total_positives": tot_pos,
            "raw_binary_agreement_pct": raw_lbl_pct,
            "cohen_kappa": mean_k_lbl,
            "fleiss_kappa": fleiss_lbl,
            "krippendorff_alpha": alpha_lbl,
            "pairwise_binary_confusion": {
                "both_positive": both_pos,
                "coder1_only": first_only,
                "coder2_only": second_only,
                "both_negative": neither,
            },
            "low_support_warning": low_sup,
            "unreliable_warning": bool(tot_pos > 0 and alpha_lbl < 0.65),
        }

    masi_alpha = round(krippendorff_alpha_masi(sets_by_unit), 4)
    micro_alpha = round(krippendorff_alpha_nominal(flat_units), 4)
    macro_alpha = round(float(np.mean(active_alphas)) if active_alphas else 0.0, 4)
    thresh = gate_threshold if gate_threshold is not None else DEFAULT_IAA_THRESHOLDS.get(task_name, 0.65)
    gate_pass = bool(max(masi_alpha, micro_alpha) >= thresh and not unreliable_classes)

    return {
        "task": task_name,
        "paradigm": "multi_label",
        "status": "COMPLETE" if len(complete_sets) >= len(all_doc_ids) else "PARTIAL",
        "n_annotators": len(coders),
        "n_units_with_2plus_ratings": len(sets_by_unit),
        "n_complete_all_coders": len(complete_sets),
        "exact_set_agreement_pct": exact_set_pct,
        "mean_pairwise_jaccard": round(float(np.mean(pair_jaccards)) if pair_jaccards else 0.0, 4),
        "krippendorff_alpha_masi": masi_alpha,
        "krippendorff_alpha_micro_binary": micro_alpha,
        "krippendorff_alpha_macro_binary": macro_alpha,
        "gate_threshold": thresh,
        "gate_passed": gate_pass,
        "unreliable_classes": unreliable_classes,
        "low_support_classes": low_support_classes,
        "reliability_note": (
            "Multi-label reliability requires inspecting per_label scores alongside aggregate alpha; "
            f"unreliable classes (<0.65 alpha): {unreliable_classes or 'none'}."
        ),
        "per_label": per_label,
    }


def compute_full_multiannotator_iaa(
    submissions_by_annotator: Dict[str, Dict[str, AnnotationSubmission]],
    *,
    round_id: str = "round_1",
    taxonomy_version: str = TAXONOMY_VERSION,
    guideline_version: str = GUIDELINE_VERSION,
) -> Dict[str, Any]:
    """Compute complete 5-task reliability report + disagreement ledger across K >= 2 annotators."""
    coders = sorted(submissions_by_annotator.keys())
    all_ids = sorted({did for m in submissions_by_annotator.values() for did in m})

    if len(coders) < 2:
        return {
            "status": "SINGLE_ANNOTATOR_ONLY_CANNOT_COMPUTE_IAA",
            "round_id": round_id,
            "taxonomy_version": taxonomy_version,
            "guideline_version": guideline_version,
            "annotators": coders,
            "n_documents": len(all_ids),
            "tasks": {},
            "disagreements": [],
            "pending_metrics": ["cohen_kappa", "fleiss_kappa", "krippendorff_alpha", "per_label_multilabel_iaa"],
        }

    tasks_out: Dict[str, Any] = {}
    for t_name, labels in SINGLE_LABEL_TASKS.items():
        tasks_out[t_name] = compute_single_label_iaa(submissions_by_annotator, t_name, labels)
    for t_name, labels in MULTI_LABEL_TASKS.items():
        tasks_out[t_name] = compute_multilabel_iaa(submissions_by_annotator, t_name, labels)

    disagreements: List[Dict[str, Any]] = []
    for did in all_ids:
        subs = {c: submissions_by_annotator[c][did] for c in coders if did in submissions_by_annotator[c]}
        if len(subs) < 2:
            continue
        statuses = {s.review_status for s in subs.values()}
        diff_tasks: List[str] = []
        if len(statuses) > 1:
            diff_tasks.append("review_status")
        if statuses == {"labeled"}:
            for t in ("sentiment", "response", "solution"):
                if len({getattr(s, t) for s in subs.values()}) > 1:
                    diff_tasks.append(t)
            for t, attr in (("issue", "issues"), ("emotion", "emotions")):
                first = getattr(next(iter(subs.values())), attr)
                if any(getattr(s, attr) != first for s in subs.values()):
                    diff_tasks.append(t)
        if diff_tasks:
            disagreements.append(
                {
                    "document_id": did,
                    "round_id": round_id,
                    "disagreed_tasks": "|".join(diff_tasks),
                    "annotator_judgments": {
                        c: {
                            "review_status": s.review_status,
                            "abstain_reason": s.abstain_reason,
                            "sentiment": s.sentiment,
                            "emotions": "|".join(sorted(s.emotions)),
                            "issues": "|".join(sorted(s.issues)),
                            "response": s.response,
                            "solution": s.solution,
                            "notes": s.notes,
                        }
                        for c, s in subs.items()
                    },
                }
            )

    any_compared = any(int(t.get("n_units_with_2plus_ratings", t.get("n_compared", 0))) > 0 for t in tasks_out.values())
    all_gates = bool(any_compared and all(t.get("gate_passed") for t in tasks_out.values()))

    return {
        "status": "COMPLETE" if any_compared else "PENDING_HUMAN_ANNOTATIONS",
        "round_id": round_id,
        "taxonomy_version": taxonomy_version,
        "guideline_version": guideline_version,
        "annotators": coders,
        "n_documents": len(all_ids),
        "n_disagreements": len(disagreements),
        "all_gates_passed": all_gates,
        "tasks": tasks_out,
        "disagreements": disagreements,
    }


def merge_independent_annotations_and_adjudicate(
    submissions_by_annotator: Dict[str, Dict[str, AnnotationSubmission]],
    adjudications_by_doc: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    round_id: str = "round_1",
) -> List[Dict[str, Any]]:
    """Preserve every original annotator's decision alongside final adjudicated labels (never overwrites history)."""
    coders = sorted(submissions_by_annotator.keys())
    all_ids = sorted({did for m in submissions_by_annotator.values() for did in m})
    adj_map = adjudications_by_doc or {}
    records: List[Dict[str, Any]] = []

    for did in all_ids:
        originals: Dict[str, Any] = {}
        statuses = set()
        for c in coders:
            if did in submissions_by_annotator[c]:
                s = submissions_by_annotator[c][did]
                statuses.add(s.review_status)
                d = asdict(s)
                d["emotions"] = sorted(s.emotions)
                d["issues"] = sorted(s.issues)
                originals[c] = d

        adj = adj_map.get(did)
        final_label: Optional[Dict[str, Any]] = None
        resolution_method = "pending_adjudication"

        if adj:
            final_label = {
                "sentiment": adj.get("sentiment", ""),
                "emotions": sorted(adj.get("emotions") or []),
                "issues": sorted(adj.get("issues") or []),
                "response": adj.get("response", ""),
                "solution": adj.get("solution", ""),
                "review_status": adj.get("review_status", "labeled"),
                "adjudicator_id": adj.get("adjudicator_id", "adjudicator_1"),
                "adjudication_notes": adj.get("notes", ""),
            }
            resolution_method = "human_adjudicated"
        elif len(originals) >= 2 and statuses == {"labeled"}:
            vals = list(originals.values())
            unanimous = all(
                vals[i][k] == vals[0][k]
                for i in range(1, len(vals))
                for k in ("sentiment", "emotions", "issues", "response", "solution")
            )
            if unanimous:
                final_label = {
                    "sentiment": vals[0]["sentiment"],
                    "emotions": list(vals[0]["emotions"]),
                    "issues": list(vals[0]["issues"]),
                    "response": vals[0]["response"],
                    "solution": vals[0]["solution"],
                    "review_status": "labeled",
                    "adjudicator_id": "unanimous_agreement",
                    "adjudication_notes": "",
                }
                resolution_method = "unanimous_multi_annotator"

        records.append(
            {
                "document_id": did,
                "round_id": round_id,
                "taxonomy_version": TAXONOMY_VERSION,
                "guideline_version": GUIDELINE_VERSION,
                "original_annotations": originals,
                "resolution_method": resolution_method,
                "final_adjudicated_label": final_label,
            }
        )
    return records
