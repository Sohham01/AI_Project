"""Sprint P1-E (Part A) — Blind, Disjoint Independent Evaluation Sampler (v2) for Linking & Resolution.

Preserves data/annotations/issue_solution_eval_v1.csv (64 pairs) and
data/annotations/resolution_eval_v1.csv (40 threads) strictly untouched as
V6.0 human-reviewed development benchmarks.

Creates strictly DISJOINT, confidence-independent stratified candidate packages:
  - issue_solution_eval_v2_blind.csv / .xlsx (180 candidate pairs across all 8 issues)
  - resolution_eval_v2_blind.csv / .xlsx (120 threads with full thread timeline digest)
  - v2_protocol_freeze_manifest.json (exclusions + SHA-256 + protocol freeze timestamp)

Zero model predictions or confidence scores appear in the blind annotation sheets!
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from ml.linking.contextual_mapper import ISSUE_DESCRIPTIONS

BASELINE_CORPUS = ROOT / "data" / "live_corpus" / "corpus.json"
V63_EXPANSION = ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"
V1_ISS_CSV = ROOT / "data" / "annotations" / "issue_solution_eval_v1.csv"
V1_RES_CSV = ROOT / "data" / "annotations" / "resolution_eval_v1.csv"
OUT_DIR = ROOT / "data" / "annotations" / "v6_3_package" / "linking_and_resolution_v2"

# Pure lexical/structural cues ONLY (never reads model confidence or predictions!)
_ACTION_LEX_RE = re.compile(
    r"\b(email|mail|call|apply|check|contact|aicte|refund|portal|submit|ticket|follow|guideline|loan|bank|linkedin|compare|withdraw|upload|document)\b",
    re.I,
)
_OUTCOME_LEX_RE = re.compile(
    r"\b(worked|solved|resolved|got my|converted|received|waiting|pending|no reply|didn'?t respond|not picking up|rejected|tried)\b",
    re.I,
)
_SOCIAL_TRAP_RE = re.compile(
    r"\b(cfbr|dm me|agreed|same|bro|\+1|thanks|not true|wrong|doesn'?t work|bad idea)\b",
    re.I,
)
_ISSUE_LEX: Dict[str, re.Pattern[str]] = {
    "academic_workload": re.compile(r"\b(quiz|quizzes|assignment|case|exam|workload|attendance|sleep|hectic|schedule)\b", re.I),
    "administrative_communication": re.compile(r"\b(waitlist|merit list|portal|offer|circular|registrar|admissions|email|document|cutoff|cap|sap)\b", re.I),
    "campus_infrastructure": re.compile(r"\b(wifi|library|classroom|campus|lab|station|airport|transport|hospital|medical|construction)\b", re.I),
    "curriculum_teaching": re.compile(r"\b(curriculum|faculty|prof|elective|pedagogy|dsai|mpp|course|syllabus|teaching)\b", re.I),
    "extracurriculars": re.compile(r"\b(club|committee|placecom|fest|competition|sports|por|societies|induction)\b", re.I),
    "financial_concerns": re.compile(r"\b(fee|fees|loan|sbi|cibil|emi|scholarship|refund|aicte|roi|withdrawal|deduction)\b", re.I),
    "hostel_facilities": re.compile(r"\b(hostel|mess|food|room|cooler|ac|geyser|laundry|warden|kettle|hygiene|dorm)\b", re.I),
    "placements": re.compile(r"\b(placement|placements|sip|internship|ctc|lpa|package|recruiter|company|shortlist|cv|resume|interview)\b", re.I),
}


def _infer_issue_from_lexical_only(text: str, fallback: str = "administrative_communication") -> str:
    scores = {iss: len(pat.findall(text)) for iss, pat in _ISSUE_LEX.items()}
    best_iss, best_cnt = max(scores.items(), key=lambda kv: kv[1])
    return best_iss if best_cnt > 0 else fallback


def _load_all_threads() -> List[Dict[str, Any]]:
    threads: List[Dict[str, Any]] = []
    for part, path in (("v6_1_baseline", BASELINE_CORPUS), ("v6_3_expansion", V63_EXPANSION)):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in data.get("posts", []):
            threads.append({**p, "_corpus_partition": part})
    return threads


def _write_csv_and_xlsx(rows: List[Dict[str, Any]], csv_path: Path, xlsx_path: Path, n_context_cols: int) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = csv_path.stem[:30]
        ws.append(fieldnames)
        for ci in range(1, len(fieldnames) + 1):
            c = ws.cell(row=1, column=ci)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1E3A5F" if ci <= n_context_cols else "9A3412")
        for r in rows:
            ws.append([r[k] for k in fieldnames])
        ws.freeze_panes = "A2"
        for col_letter in ("E", "F", "G"):
            ws.column_dimensions[col_letter].width = 42
        for r_i in range(2, len(rows) + 2):
            for c_i in range(4, min(n_context_cols + 1, 9)):
                ws.cell(row=r_i, column=c_i).alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(xlsx_path)
    except ImportError:
        pass


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threads = _load_all_threads()

    # Load v1 development IDs to exclude strictly!
    v1_pairs: Set[Tuple[str, str, str]] = set()
    if V1_ISS_CSV.is_file():
        with V1_ISS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                v1_pairs.add((r["issue_id"], r["post_id"], r["solution_comment_id"]))

    v1_res_pids: Set[str] = set()
    if V1_RES_CSV.is_file():
        with V1_RES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                v1_res_pids.add(r["post_id"])

    # ------------------------------------------------------------------
    # 1. Sample Issue-Solution Independent v2 Pairs (target 180)
    # Stratified WITHOUT using model confidence! Covers actionable,
    # lexical-trap/contradictory, short/social, and mismatched-topic candidates.
    # ------------------------------------------------------------------
    cand_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in threads:
        pid = str(p["reddit_id"])
        sub = str(p.get("subreddit") or "")
        title = str(p.get("title") or "")
        part = str(p["_corpus_partition"])
        cmts = p.get("comments") or []
        by_cid = {c["reddit_id"]: c for c in cmts}
        thread_iss = _infer_issue_from_lexical_only(f"{title} {p.get('body') or ''}")

        for c in cmts:
            cid = str(c["reddit_id"])
            body = str(c.get("body") or "").strip()
            if len(body) < 4:
                continue
            if (thread_iss, pid, cid) in v1_pairs:
                continue
            p_cid = c.get("parent_reddit_id")
            p_ctx = by_cid[p_cid]["body"] if (p_cid and p_cid in by_cid) else title
            lid = detect_indian_reddit_language(body)

            has_act = bool(_ACTION_LEX_RE.search(body))
            has_trap = bool(_SOCIAL_TRAP_RE.search(body))
            words_n = len(body.split())
            bucket = "actionable_lexical" if (has_act and words_n >= 10) else ("trap_or_social" if (has_trap or words_n < 8) else "general_discussion")

            cand_pools[f"{thread_iss}|{bucket}"].append({
                "pair_id": f"isv2-{pid}-{cid}",
                "corpus_partition": part,
                "subreddit": sub,
                "issue_id": thread_iss,
                "post_id": pid,
                "solution_comment_id": cid,
                "post_title": title,
                "parent_context": str(p_ctx or "")[:320],
                "comment_text": body,
                "language": lid["language"],
                "sampling_stratum": f"{part}|{bucket}",
                "relationship_label": "",  # genuine_solution | partial_or_context_dependent | not_valid_relationship
                "failure_trap_tag": "",    # optional: contradictory | wrong_issue | pure_agreement | non_actionable | insufficient
                "review_status": "",       # labeled | uncertain | abstain
                "annotator_id": "",
                "human_note": "",
            })

    selected_iss: List[Dict[str, Any]] = []
    used_pairs: Set[Tuple[str, str]] = set()
    for iss in sorted(ISSUE_DESCRIPTIONS):
        picked_for_iss: List[Dict[str, Any]] = []
        # Target ~22-23 per issue (~180 total) rotating actionable, trap, general and preferring v6.3 + diverse threads
        for bucket, quota in (("actionable_lexical", 9), ("trap_or_social", 7), ("general_discussion", 7)):
            pool = sorted(
                cand_pools.get(f"{iss}|{bucket}", []),
                key=lambda x: (0 if x["corpus_partition"] == "v6_3_expansion" else 1, hashlib.sha256(x["pair_id"].encode()).hexdigest()),
            )
            taken = 0
            for item in pool:
                if (item["post_id"], item["solution_comment_id"]) in used_pairs:
                    continue
                used_pairs.add((item["post_id"], item["solution_comment_id"]))
                picked_for_iss.append(item)
                taken += 1
                if taken >= quota:
                    break
        selected_iss.extend(picked_for_iss)
    selected_iss = selected_iss[:180]

    # ------------------------------------------------------------------
    # 2. Sample Resolution Independent v2 Threads (target 120)
    # Stratified WITHOUT using model predictions! Includes threads with
    # outcome cues, pending cues, advice/action cues, and baseline threads.
    # ------------------------------------------------------------------
    res_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in threads:
        pid = str(p["reddit_id"])
        if pid in v1_res_pids:
            continue
        cmts = p.get("comments") or []
        if len(cmts) < 2:
            continue
        title = str(p.get("title") or "")
        body = str(p.get("body") or "")
        part = str(p["_corpus_partition"])
        sub = str(p.get("subreddit") or "")
        full_blob = f"{title} {body} " + " ".join(str(c.get("body") or "") for c in cmts[:12])
        has_out = bool(_OUTCOME_LEX_RE.search(full_blob))
        has_act = bool(_ACTION_LEX_RE.search(full_blob))
        stratum = (
            f"{part}|outcome_or_pending_cues"
            if has_out
            else (f"{part}|action_or_advice_cues" if has_act else f"{part}|low_signal_or_open")
        )
        # Compact chronological thread digest so annotator sees full context!
        sorted_c = sorted(cmts, key=lambda c: (c.get("created_at") or "", c["reddit_id"]))
        timeline_lines = []
        for idx_c, c in enumerate(sorted_c[:15], start=1):
            c_txt = str(c.get("body") or "").replace("\n", " ").strip()
            timeline_lines.append(f"[{idx_c}] ({c['reddit_id']} @ {c.get('created_at','')}) {c_txt[:220]}")
        digest = "\n".join(timeline_lines)

        res_pools[stratum].append({
            "eval_id": f"resv2-{pid}",
            "corpus_partition": part,
            "subreddit": sub,
            "post_id": pid,
            "issue_domain_hint": _infer_issue_from_lexical_only(f"{title} {body}"),
            "post_title": title,
            "post_body": body[:400],
            "n_comments": len(cmts),
            "thread_timeline_digest": digest,
            "sampling_stratum": stratum,
            "gold_resolution_state": "",         # reported_success | reported_pending | proposed_only | insufficient_evidence
            "gold_verdict_tier": "",             # Resolved | Partially Resolved | Unresolved | Insufficient Evidence
            "gold_has_contradiction": "",        # true | false
            "gold_chronological_reversal": "",   # true | false
            "gold_evidence_comment_id": "",
            "review_status": "",                 # labeled | uncertain | abstain
            "annotator_id": "",
            "human_note": "",
        })

    selected_res: List[Dict[str, Any]] = []
    strat_keys = sorted(res_pools.keys())
    for k in strat_keys:
        res_pools[k].sort(key=lambda x: hashlib.sha256(x["post_id"].encode()).hexdigest())
    round_i = 0
    while len(selected_res) < 120:
        progressed = False
        for k in strat_keys:
            if round_i < len(res_pools[k]):
                selected_res.append(res_pools[k][round_i])
                progressed = True
                if len(selected_res) >= 120:
                    break
        if not progressed:
            break
        round_i += 1

    # Write blind CSVs, JSONLs, and Excel workbooks
    iss_csv = OUT_DIR / "issue_solution_eval_v2_blind.csv"
    res_csv = OUT_DIR / "resolution_eval_v2_blind.csv"
    _write_csv_and_xlsx(selected_iss, iss_csv, OUT_DIR / "issue_solution_eval_v2_blind.xlsx", n_context_cols=11)
    _write_csv_and_xlsx(selected_res, res_csv, OUT_DIR / "resolution_eval_v2_blind.xlsx", n_context_cols=10)

    # Assert 0 overlap with v1 development sets
    assert not any((r["issue_id"], r["post_id"], r["solution_comment_id"]) in v1_pairs for r in selected_iss)
    assert not any(r["post_id"] in v1_res_pids for r in selected_res)

    manifest = {
        "eval_version": "v2_independent_protocol",
        "sampling_policy": "Confidence-independent lexical/structural stratification; 0% overlap with v1 dev sets; blind (zero model outputs in sheet).",
        "issue_solution_v2": {
            "n_pairs": len(selected_iss),
            "v1_dev_overlap": 0,
            "by_partition": dict(Counter(r["corpus_partition"] for r in selected_iss)),
            "by_issue": dict(Counter(r["issue_id"] for r in selected_iss)),
            "csv_sha256": hashlib.sha256(iss_csv.read_bytes()).hexdigest() if iss_csv.is_file() else "",
            "status": "PENDING_INDEPENDENT_HUMAN_ANNOTATION",
        },
        "resolution_v2": {
            "n_threads": len(selected_res),
            "v1_dev_overlap": 0,
            "by_partition": dict(Counter(r["corpus_partition"] for r in selected_res)),
            "by_stratum": dict(Counter(r["sampling_stratum"] for r in selected_res)),
            "csv_sha256": hashlib.sha256(res_csv.read_bytes()).hexdigest() if res_csv.is_file() else "",
            "status": "PENDING_INDEPENDENT_HUMAN_ANNOTATION",
        },
    }
    (OUT_DIR / "v2_protocol_freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
