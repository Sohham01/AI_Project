"""Prepare strictly blind, Annotation-Studio-ready CSVs for Human IAA review (50 items).

Guarantees:
- Never copies model predictions or Pass-1 V2 rubric labels into the human review CSV.
- Every label cell (sentiment, emotions, issues, response_type, is_solution) is 100% blank.
- Compatible directly with `backend/run_annotation_studio.py --csv <path> --port 8765`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import EXP_ROOT, load_corpus_lookups

AL_DIR = EXP_ROOT / "active_learning"
SRC_IAA_CSV = AL_DIR / "iaa_double_annotation_50.csv"
STUDIO_HUMAN_A_CSV = AL_DIR / "iaa_50_human_annotator_studio.csv"
STUDIO_HUMAN_B_CSV = AL_DIR / "iaa_50_human_annotator_b_studio.csv"

STUDIO_FIELDNAMES = [
    "document_id",
    "post_id",
    "comment_id",
    "subreddit",
    "post_title",
    "parent_comment_id",
    "parent_text",
    "parent_context_status",
    "created_at",
    "score",
    "original_text",
    "normalized_text",
    "language",
    "script",
    "is_code_mixed",
    "detected_languages",
    "sentiment",
    "emotions",
    "issues",
    "response_type",
    "is_solution",
    "review_status",
    "abstain_reason",
    "annotator_id",
    "annotator_notes",
]


def prepare_studio_csvs(overwrite: bool = False) -> dict:
    assert SRC_IAA_CSV.exists(), f"Missing {SRC_IAA_CSV} — run Phase 6 active learning first."
    _, post_by_pid, _ = load_corpus_lookups()

    src_rows = list(csv.DictReader(SRC_IAA_CSV.open("r", encoding="utf-8")))
    studio_rows = []
    for r in src_rows:
        pid = r["post_id"]
        cid = r["comment_id"]
        p_obj = post_by_pid.get(pid, {})
        c_obj = next((c for c in p_obj.get("comments") or [] if c.get("reddit_id") == cid), {})
        prid = c_obj.get("parent_reddit_id") or ""
        parent_txt = r.get("parent_comment_text") or ""
        raw_txt = r.get("text") or ""

        studio_rows.append(
            {
                "document_id": r["document_id"],
                "post_id": pid,
                "comment_id": cid,
                "subreddit": r.get("subreddit") or "",
                "post_title": r.get("post_title") or "",
                "parent_comment_id": prid if parent_txt else "",
                "parent_text": parent_txt,
                "parent_context_status": "available" if parent_txt else "top_level",
                "created_at": str(c_obj.get("created_utc") or ""),
                "score": str(c_obj.get("score") if c_obj.get("score") is not None else ""),
                "original_text": raw_txt,
                "normalized_text": " ".join(raw_txt.split()),
                "language": str(c_obj.get("language") or "unknown"),
                "script": str(c_obj.get("script") or "Latin"),
                "is_code_mixed": str(bool(c_obj.get("is_code_mixed", False))).lower(),
                "detected_languages": str(c_obj.get("detected_languages") or ""),
                # Strictly blank blind human annotation cells
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
        )

    written = []
    for target in (STUDIO_HUMAN_A_CSV, STUDIO_HUMAN_B_CSV):
        if target.exists() and not overwrite:
            print(f"[IAA Studio Prep] Preserving existing file (use --overwrite to reset): {target}")
            continue
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STUDIO_FIELDNAMES)
            w.writeheader()
            w.writerows(studio_rows)
        written.append(str(target))
        print(f"[IAA Studio Prep] Wrote {len(studio_rows)} strictly blank blind items -> {target}")

    return {"n_items": len(studio_rows), "written": written}


if __name__ == "__main__":
    overwrite_flag = "--overwrite" in sys.argv
    prepare_studio_csvs(overwrite=overwrite_flag)
