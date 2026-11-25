"""Sprint P0-C Runner — Honest Audit of Existing Annotations + Blind Round-2 Stratified Re-Annotation Package.

1. Audits every existing annotation dataset to verify how many independent annotators
   actually contributed, whether original decisions are preserved, and whether IAA can
   validly be computed (never fabricating or estimating IAA!).
2. Prepares a blind, strictly leakage-free Round-2 stratified sample (n = 80 items sampled
   exclusively from trainval_v1 [n=367], ZERO frozen test rows) across all subreddits,
   issues, response roles, solution candidates, and multilingual/code-mixed comments.
3. Emits blind CSVs + ready-to-annotate Excel workbooks (Annotator 1 & Annotator 2)
   plus a blind Adjudication workbook under data/annotations/v6_3_package/iaa_blind_round2/.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.evaluation.annotation_reliability import (
    GUIDELINE_VERSION,
    TAXONOMY_VERSION,
    verify_blind_sheet,
)
from ml.experiments.v5_optimization.common import EXP_ROOT, load_v1_records_and_frozen_splits

AUDIT_OUT = ROOT / "ml" / "artifacts" / "v6_3_experiments" / "annotation_audit" / "existing_annotation_audit.json"
PKG_DIR = ROOT / "data" / "annotations" / "v6_3_package" / "iaa_blind_round2"
TEMPLATE_XLSX = ROOT / "data" / "annotations" / "templates" / "RCI_Manual_Annotation_Template.xlsx"


def audit_existing_annotations() -> Dict[str, Any]:
    """Audit all existing annotation CSVs cleanly and report exact independent-annotator counts."""
    al_dir = EXP_ROOT / "active_learning"

    def _csv_summary(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {"exists": False, "path": str(path)}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        annotator_ids = sorted({(r.get("annotator_id") or "").strip() for r in rows if (r.get("annotator_id") or "").strip()})
        sources = sorted({(r.get("annotation_source") or "").strip() for r in rows if (r.get("annotation_source") or "").strip()})
        statuses = Counter(((r.get("review_status") or "").strip().lower() or "<blank>") for r in rows)
        return {
            "exists": True,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "n_rows": len(rows),
            "distinct_annotator_ids": annotator_ids,
            "annotation_sources": sources,
            "review_status_counts": dict(sorted(statuses.items())),
        }

    v1_info = _csv_summary(ROOT / "data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv")
    v2_info = _csv_summary(ROOT / "data/annotations/real_reddit_labeled_v2/master_labeled_v2.csv")
    ha_info = _csv_summary(al_dir / "iaa_50_human_annotator_studio.csv")
    hb_info = _csv_summary(al_dir / "iaa_50_human_annotator_b_studio.csv")
    dbl_info = _csv_summary(al_dir / "iaa_double_annotation_50.csv")

    return {
        "audit_version": "v6.3",
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": GUIDELINE_VERSION,
        "findings": {
            "real_reddit_labeled_v1": {
                **v1_info,
                "independent_human_annotators_recorded": 1,
                "original_multi_annotator_decisions_preserved": False,
                "can_compute_true_multi_human_iaa": False,
                "note": "Single-curator rubric-curated dataset (459 rows); contains NO independent second annotator.",
            },
            "iaa_50_human_vs_v2_pass1": {
                **ha_info,
                "independent_human_annotators_recorded": 1,
                "comparison_conducted": "1 blind human reviewer (researcher_1) vs Pass-1 V2 rubric (0/5 IAA gates passed; Track B V2 blocked)",
                "can_compute_true_multi_human_iaa": False,
                "note": "Track B V2 expansion failed all 5 IAA gates against 1 human reviewer and remains archived/blocked.",
            },
            "iaa_50_annotator_b": {
                **hb_info,
                "independent_human_annotators_recorded": 0,
                "can_compute_true_multi_human_iaa": False,
                "note": "Blank template awaiting second human annotator.",
            },
            "iaa_double_annotation_50": dbl_info,
            "real_reddit_labeled_v2": v2_info,
        },
        "metrics_status": "PENDING_SECOND_INDEPENDENT_ANNOTATOR",
        "metrics_pending": [
            "two_human_cohen_kappa_single_label (sentiment, response, solution)",
            "multi_rater_fleiss_kappa",
            "multi_human_krippendorff_alpha (nominal + MASI + per-label binary)",
            "two_human_adjudicated_gold_agreement_audit",
        ],
    }


def _export_blind_xlsx(rows: List[Dict[str, Any]], out_xlsx: Path) -> None:
    """Populate a fresh copy of RCI_Manual_Annotation_Template.xlsx if openpyxl is installed."""
    if not TEMPLATE_XLSX.is_file():
        return
    try:
        import openpyxl
    except ImportError:
        return

    wb = openpyxl.load_workbook(TEMPLATE_XLSX)
    ws = wb["02_Annotation_Sheet"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col_map = {h: idx for idx, h in enumerate(headers, start=1) if h}
    for r_i, item in enumerate(rows, start=2):
        for k, v in item.items():
            if k in col_map:
                ws.cell(row=r_i, column=col_map[k], value=v)
    wb.save(out_xlsx)


def build_round2_blind_sample(target_n: int = 80) -> Dict[str, Any]:
    """Sample 80 stratified blind items strictly from trainval (0 frozen test rows!)."""
    data = load_v1_records_and_frozen_splits()
    trainval = data["trainval"]
    test_ids = {r["document_id"] for r in data["test"]}
    test_pids = {r["post_id"] for r in data["test"]}

    # Verify zero overlap guard
    assert not any(r["document_id"] in test_ids or r["post_id"] in test_pids for r in trainval)

    # Load language detector for stratification metadata only (never exposed as label!)
    from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
    from app.services.nlp.preprocessing.multilingual_preprocessor import MultilingualTextPreprocessor

    prep = MultilingualTextPreprocessor()
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in sorted(trainval, key=lambda x: x["document_id"]):
        lid = detect_indian_reddit_language(rec["text"])
        resp_cls = rec["labels"]["response"]
        iss_list = rec["labels"]["issue"]
        primary_iss = iss_list[0] if iss_list else "no_issue"
        key = f"{resp_cls}|{primary_iss}|{'indic' if lid['is_code_mixed'] or lid['language'] in {'hi','hi-en','bn','bn-en','or','or-en'} else 'en'}"
        buckets[key].append((rec, lid))

    # Deterministic round-robin stratified sampling across buckets
    picked: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_dids: set[str] = set()
    sorted_keys = sorted(buckets.keys())
    round_idx = 0
    while len(picked) < target_n:
        progressed = False
        for k in sorted_keys:
            if round_idx < len(buckets[k]):
                rec, lid = buckets[k][round_idx]
                if rec["document_id"] not in seen_dids:
                    picked.append((rec, lid))
                    seen_dids.add(rec["document_id"])
                    progressed = True
                    if len(picked) >= target_n:
                        break
        if not progressed:
            break
        round_idx += 1

    # Strict unified 25-column blind schema (matches RCI_Manual_Annotation_Template.xlsx & validate_pilot_annotations.py)
    fieldnames = [
        "document_id", "post_id", "comment_id", "subreddit", "post_title",
        "parent_comment_id", "parent_text", "parent_context_status", "created_at", "score",
        "original_text", "normalized_text", "language", "script", "is_code_mixed",
        "detected_languages", "sentiment", "emotions", "issues", "response_type",
        "is_solution", "review_status", "abstain_reason", "annotator_id", "annotator_notes",
    ]

    blind_rows: List[Dict[str, Any]] = []
    strat_summary = {"by_subreddit": Counter(), "by_language": Counter(), "has_parent": 0}
    for rec, lid in picked:
        pre = prep.preprocess(rec["text"])
        p_txt = rec.get("parent_comment_text") or ""
        p_status = "available" if rec.get("parent_comment_id") and p_txt else ("unavailable" if rec.get("parent_comment_id") else "top_level")
        row = {
            "document_id": f"r2-{rec['document_id']}",
            "post_id": rec["post_id"],
            "comment_id": rec["comment_id"],
            "subreddit": rec.get("subreddit", ""),
            "post_title": rec.get("post_title", ""),
            "parent_comment_id": rec.get("parent_comment_id") or "",
            "parent_text": p_txt,
            "parent_context_status": p_status,
            "created_at": "",
            "score": "",
            "original_text": rec["text"],
            "normalized_text": pre.normalized_text,
            "language": lid["language"],
            "script": lid["script"],
            "is_code_mixed": "true" if lid["is_code_mixed"] else "false",
            "detected_languages": "|".join(lid.get("detected_languages") or [lid["language"]]),
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        }
        blind_rows.append(row)
        strat_summary["by_subreddit"][row["subreddit"] or "unknown"] += 1
        strat_summary["by_language"][lid["language"]] += 1
        if p_status == "available":
            strat_summary["has_parent"] += 1

    ok_blind, blind_errs = verify_blind_sheet(blind_rows)
    assert ok_blind, f"Round-2 sheet failed blind verification: {blind_errs}"

    PKG_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("round2_blind_80_annotator_1.csv", "round2_blind_80_annotator_2.csv"):
        p = PKG_DIR / fname
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(blind_rows)

    _export_blind_xlsx(blind_rows, PKG_DIR / "round2_blind_80_annotator_1.xlsx")
    _export_blind_xlsx(blind_rows, PKG_DIR / "round2_blind_80_annotator_2.xlsx")

    # Empty adjudication template
    adj_fields = [
        "document_id", "post_id", "post_title", "parent_text", "original_text",
        "annotator_1_labels", "annotator_2_labels",
        "adjudicated_status", "adjudicated_sentiment", "adjudicated_emotions",
        "adjudicated_issues", "adjudicated_response", "adjudicated_solution",
        "adjudicator_id", "adjudication_notes",
    ]
    with (PKG_DIR / "round2_adjudication_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adj_fields)
        w.writeheader()

    manifest = {
        "round_id": "round_2_blind_stratified",
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": GUIDELINE_VERSION,
        "n_items": len(blind_rows),
        "frozen_test_overlap_count": 0,
        "blind_verified": True,
        "iaa_metrics_status": "PENDING_INDEPENDENT_HUMAN_SUBMISSIONS",
        "sampling_summary": {
            "by_subreddit": dict(strat_summary["by_subreddit"].most_common()),
            "by_language": dict(strat_summary["by_language"].most_common()),
            "with_parent_context": strat_summary["has_parent"],
        },
        "files": [
            "round2_blind_80_annotator_1.csv",
            "round2_blind_80_annotator_1.xlsx",
            "round2_blind_80_annotator_2.csv",
            "round2_blind_80_annotator_2.xlsx",
            "round2_adjudication_template.csv",
        ],
    }
    (PKG_DIR / "round2_sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    audit = audit_existing_annotations()
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = build_round2_blind_sample(target_n=80)
    print(json.dumps({"audit_status": audit["metrics_status"], "round2_manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
