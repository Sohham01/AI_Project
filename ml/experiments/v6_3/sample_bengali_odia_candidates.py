"""Sprint P1-B — Improve Bengali & Odia Coverage Across Expanded Real Corpus (v2 Candidate Sampler).

Preserves data/annotations/multilingual_eval_real_v1.csv (n=84) and
data/annotations/multilingual_eval_synthetic_probes_v1.csv (n=24) untouched.

Samples authentic Bengali-script, Odia-script, Romanized Bengali (Banglish),
Romanized Odia, and mixed-language real Reddit discussions from both the baseline
365-thread corpus and the expanded V6.3 unannotated corpus into an annotation-ready
CSV / JSONL / XLSX review package with explicit detector_hypothesis warnings (never
auto-assigning gold ground truth!).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from ml.experiments.v6_programme.eval_multilingual_lid import _evaluate_rows

_BENGALI_SCRIPT_RE = re.compile(r"[\u0980-\u09FF]")
_ODIA_SCRIPT_RE = re.compile(r"[\u0B00-\u0B7F]")
_DEVANAGARI_SCRIPT_RE = re.compile(r"[\u0900-\u097F]")

BASELINE_CORPUS = ROOT / "data" / "live_corpus" / "corpus.json"
V63_EXPANSION = ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"
V1_REAL_EVAL = ROOT / "data" / "annotations" / "multilingual_eval_real_v1.csv"
V1_SYNTH_EVAL = ROOT / "data" / "annotations" / "multilingual_eval_synthetic_probes_v1.csv"

OUT_PKG_DIR = ROOT / "data" / "annotations" / "v6_3_package" / "multilingual_bn_or_v2"
OUT_REPORT_PATH = ROOT / "ml" / "artifacts" / "v6_3_experiments" / "multilingual_bn_or" / "coverage_and_eval_report.json"

# High-precision Banglish & Romanized Odia candidate recall patterns (used for candidate sampling ONLY, NEVER gold truth!)
_BANGLISH_RE = re.compile(
    r"\b(amar|tumi|tomar|ami|amra|apni|kintu|korbo|korte|korchi|hocche|hobe|hoyeche|bhalo|"
    r"onek|beshi|kemon|keno|ki\s+bhabe|ache|nei|parchi|parbo|bolchi|jodi|taka|lagche|bharti|"
    r"theke|diye|niye|ekhane|okhane|shob|jaiga|bangla|kolkata\s+theke)\b",
    re.IGNORECASE,
)
_ROMAN_ODIA_RE = re.compile(
    r"\b(mu|tume|tumara|mora|ama|aaji|kalia|kemiti|kan|kana|katha|bujhili|rahucha|kouthi|"
    r"achi|achhi|nhi\s+achi|kichi|bhala|besi|sabu|pasuchu|karibi|karucha|heuchi|heba|"
    r"jaichi|asichi|tanka|bhubaneswar\s+ru|sambalpur\s+ra|odia)\b",
    re.IGNORECASE,
)


def _classify_candidate(text: str, lid: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (stratum, romanized_or_native, sampling_reason) or ('', '', '')."""
    has_bn_script = bool(_BENGALI_SCRIPT_RE.search(text))
    has_or_script = bool(_ODIA_SCRIPT_RE.search(text))
    has_dev_script = bool(_DEVANAGARI_SCRIPT_RE.search(text))
    has_latin = bool(re.search(r"[A-Za-z]{2,}", text))

    if has_bn_script:
        r_or_n = "mixed_script" if has_latin else "native_script"
        return (
            "bengali_native_script",
            r_or_n,
            "contains_bengali_unicode_block_0980_09FF",
        )
    if has_or_script:
        r_or_n = "mixed_script" if has_latin else "native_script"
        return (
            "odia_native_script",
            r_or_n,
            "contains_odia_unicode_block_0B00_0B7F",
        )

    lang = str(lid.get("language") or "")
    bn_hits = _BANGLISH_RE.findall(text)
    or_hits = _ROMAN_ODIA_RE.findall(text)

    if lang in {"bn", "bn-en"} or len(bn_hits) >= 2:
        return (
            "romanized_bengali",
            "romanized",
            f"lid={lang or 'none'}; banglish_cues={sorted(set(x.lower() for x in bn_hits))[:4]}",
        )
    if lang in {"or", "or-en"} or len(or_hits) >= 2:
        return (
            "romanized_odia",
            "romanized",
            f"lid={lang or 'none'}; roman_odia_cues={sorted(set(x.lower() for x in or_hits))[:4]}",
        )
    if lid.get("has_multi_language_spans") or (has_dev_script and has_latin) or lang == "hi-en":
        return (
            "mixed_language_control",
            "mixed_script" if has_dev_script else "romanized",
            f"lid={lang}; multi_span={bool(lid.get('has_multi_language_spans'))}",
        )
    return "", "", ""


def _load_v1_eval_stats(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "post_id": r.get("post_id", ""),
                "text": r["text"],
                "gold_language": r["gold_language"],
                "gold_script": r["gold_script"],
                "gold_is_code_mixed": str(r["gold_is_code_mixed"]).lower() == "true",
                "gold_multi_span": str(r["gold_multi_span"]).lower() == "true",
            })
    return rows


def run_sampling_and_coverage_audit() -> Dict[str, Any]:
    OUT_PKG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Measure baseline v1 evaluation coverage (honestly separating real vs synthetic!)
    v1_real = _load_v1_eval_stats(V1_REAL_EVAL)
    v1_synth = _load_v1_eval_stats(V1_SYNTH_EVAL)
    v1_real_lang_counts = Counter(r["gold_language"] for r in v1_real)
    v1_real_script_counts = Counter(r["gold_script"] for r in v1_real)
    v1_real_eval = _evaluate_rows(v1_real)
    v1_synth_eval = _evaluate_rows(v1_synth)

    v1_ids = {r["id"] for r in v1_real}

    # 2. Scan baseline + V6.3 expanded corpus for real candidates across the 5 target strata
    corpus_sources: List[Tuple[str, Path]] = [("v6_1_baseline", BASELINE_CORPUS)]
    if V63_EXPANSION.is_file():
        corpus_sources.append(("v6_3_expansion", V63_EXPANSION))

    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    corpus_scan_totals = Counter()

    for partition, path in corpus_sources:
        data = json.loads(path.read_text(encoding="utf-8"))
        for post in data.get("posts", []):
            pid = str(post.get("reddit_id") or "")
            sub = str(post.get("subreddit") or "")
            title = str(post.get("title") or "")
            for c in post.get("comments", []):
                cid = str(c.get("reddit_id") or "")
                body = str(c.get("body") or "").strip()
                if not body or len(body) < 8:
                    continue
                if cid in v1_ids:
                    continue
                corpus_scan_totals[partition] += 1
                lid = detect_indian_reddit_language(body)
                stratum, r_or_n, reason = _classify_candidate(body, lid)
                if not stratum:
                    continue
                by_stratum[stratum].append({
                    "example_id": f"mlv2-{cid}",
                    "corpus_partition": partition,
                    "source_community": sub,
                    "language_candidate_detector_hypothesis": lid["language"],
                    "script_category": lid["script"],
                    "romanized_or_native": r_or_n,
                    "sampling_stratum": stratum,
                    "thread_group_id": pid,
                    "post_title": title[:160],
                    "text": body,
                    "sampling_reason": reason,
                    "annotation_status": "pending_human_review",
                    "gold_language": "",
                    "gold_script": "",
                    "gold_is_code_mixed": "",
                    "gold_multi_span": "",
                    "review_status": "",
                    "reviewer_notes": "",
                })

    # Balanced selection: keep ALL authentic Bengali/Odia script + Banglish + Roman-Odia candidates (up to 45 each),
    # plus stratified mixed-language control up to 35
    caps = {
        "bengali_native_script": 45,
        "odia_native_script": 45,
        "romanized_bengali": 45,
        "romanized_odia": 45,
        "mixed_language_control": 35,
    }
    selected_rows: List[Dict[str, Any]] = []
    available_counts = {k: len(v) for k, v in sorted(by_stratum.items())}
    for stratum, cap in caps.items():
        items = sorted(
            by_stratum.get(stratum, []),
            key=lambda x: (0 if x["corpus_partition"] == "v6_3_expansion" else 1, -min(len(x["text"]), 300), x["example_id"]),
        )
        # Keep thread diversity max 4 per thread
        per_thread: Counter = Counter()
        for it in items:
            if per_thread[it["thread_group_id"]] >= 4:
                continue
            per_thread[it["thread_group_id"]] += 1
            selected_rows.append(it)
            if sum(1 for r in selected_rows if r["sampling_stratum"] == stratum) >= cap:
                break

    selected_counts = Counter(r["sampling_stratum"] for r in selected_rows)

    csv_path = OUT_PKG_DIR / "multilingual_eval_real_v2_candidates.csv"
    jsonl_path = OUT_PKG_DIR / "multilingual_eval_real_v2_candidates.jsonl"
    if selected_rows:
        fieldnames = list(selected_rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(selected_rows)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in selected_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Also emit Excel workbook when openpyxl is available
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Font, PatternFill

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Multilingual_BN_OR_V2"
            ws.append(fieldnames)
            for c_i in range(1, len(fieldnames) + 1):
                cell = ws.cell(row=1, column=c_i)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1E3A5F" if c_i <= 12 else "9A3412")
            for row in selected_rows:
                ws.append([row[k] for k in fieldnames])
            ws.column_dimensions["I"].width = 30
            ws.column_dimensions["J"].width = 48
            ws.column_dimensions["K"].width = 34
            ws.freeze_panes = "A2"
            for r in range(2, len(selected_rows) + 2):
                ws.cell(row=r, column=10).alignment = Alignment(wrap_text=True, vertical="top")
            wb.save(OUT_PKG_DIR / "multilingual_eval_real_v2_candidates.xlsx")
        except ImportError:
            pass

    report = {
        "eval_version": "multilingual_bn_or_v6.3",
        "v1_real_eval_n": len(v1_real),
        "v1_real_language_counts": dict(v1_real_lang_counts),
        "v1_real_script_counts": dict(v1_real_script_counts),
        "v1_real_baseline_metrics": {
            "accuracy": v1_real_eval["language_metrics"]["accuracy"],
            "macro_f1": v1_real_eval["language_metrics"]["macro_f1"],
            "per_class": v1_real_eval["language_metrics"]["per_class"],
            "script_accuracy": v1_real_eval["script_metrics"]["accuracy"],
        },
        "v1_synthetic_probes_n": len(v1_synth),
        "v1_synthetic_note": "Strictly separated synthetic diagnostic probes — NOT authentic Reddit evaluation.",
        "v1_synthetic_metrics": {
            "accuracy": v1_synth_eval["language_metrics"]["accuracy"],
            "macro_f1": v1_synth_eval["language_metrics"]["macro_f1"],
            "per_class": v1_synth_eval["language_metrics"]["per_class"],
        },
        "expanded_corpus_candidate_scan": {
            "comments_scanned_by_partition": dict(corpus_scan_totals),
            "raw_candidates_found_by_stratum": available_counts,
            "selected_for_v2_review_by_stratum": dict(selected_counts),
            "total_v2_candidates": len(selected_rows),
            "v2_annotation_status": "PENDING_HUMAN_REVIEW_DO_NOT_CLAIM_BN_OR_REPRESENTATIVE_PERFORMANCE_YET",
            "files": [
                "data/annotations/v6_3_package/multilingual_bn_or_v2/multilingual_eval_real_v2_candidates.csv",
                "data/annotations/v6_3_package/multilingual_bn_or_v2/multilingual_eval_real_v2_candidates.jsonl",
            ],
        },
    }
    OUT_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    rep = run_sampling_and_coverage_audit()
    print(json.dumps(rep["expanded_corpus_candidate_scan"], indent=2))


if __name__ == "__main__":
    main()
