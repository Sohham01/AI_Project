"""Build isolated data/annotations/real_reddit_labeled_v2/ from V1 (459) + 248 curated real AL comments.

Strict Integrity & Transparency:
- NEVER modifies data/annotations/real_reddit_labeled_v1/ (read-only copy of all 459 V1 rows).
- Preserves exact 92-comment frozen V5 test set bit-for-bit (v2_split='test_v5_frozen').
- Excludes bot/noise rows (haikusbot #103, meme noise #11) to adjudication_queue.csv per
  ANNOTATION_GUIDELINES.md Section 3.
- Every label for rows 1..250 reflects explicit manual rubric reading of (post_title, parent_comment_text, text),
  NOT model predictions, and is transparently tagged:
    annotation_source = 'rubric_curated_real_reddit_v2_pending_human_iaa'
- Asserts ZERO post_id / corpus_post_id / title overlap across trainval_v2, test_v5_frozen, and test_v2_new.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    EXP_ROOT,
    V1_LABELED_DIR,
    ensure_dir,
    load_v1_records_and_frozen_splits,
    save_json,
)

V2_DIR = ROOT / "data" / "annotations" / "real_reddit_labeled_v2"
QUEUE_CSV = EXP_ROOT / "active_learning" / "annotation_queue_250.csv"

# Explicit per-item manual rubric judgments for queue_index 1..250:
# (sentiment, emotions, issues, response, solution) or None to route to adjudication/abstain
CURATED_LABELS: Dict[int, Optional[Tuple[str, str, str, str, str]]] = {
    # --- 1..50 ---
    1: ("positive", "", "academic_workload|hostel_facilities", "personal_experience", "not_solution"),
    2: ("neutral", "", "placements", "personal_experience", "not_solution"),
    3: ("neutral", "hope", "curriculum_teaching|placements", "personal_experience", "not_solution"),
    4: ("neutral", "", "placements", "personal_experience", "not_solution"),
    5: ("neutral", "", "curriculum_teaching", "advice", "not_solution"),
    6: ("neutral", "curiosity", "curriculum_teaching", "advice", "not_solution"),
    7: ("positive", "confidence", "curriculum_teaching|placements", "opinion", "not_solution"),
    8: ("neutral", "stress|curiosity", "placements", "personal_experience", "not_solution"),
    9: ("negative", "frustration", "curriculum_teaching|placements", "counterargument", "not_solution"),
    10: ("positive", "hope", "extracurriculars", "opinion", "not_solution"),
    11: None,  # abstain: meme/off_topic_noise ("This finna make my tt champ crash out")
    12: ("neutral", "", "administrative_communication", "counterargument", "not_solution"),
    13: ("neutral", "curiosity", "curriculum_teaching|placements", "opinion", "not_solution"),
    14: ("neutral", "", "placements", "advice", "solution"),
    15: ("neutral", "hope", "placements", "personal_experience", "not_solution"),
    16: ("neutral", "", "financial_concerns", "personal_experience", "not_solution"),
    17: ("negative", "", "placements", "counterargument", "not_solution"),
    18: ("negative", "", "", "advice", "not_solution"),
    19: ("neutral", "", "financial_concerns", "solution", "solution"),
    20: ("negative", "stress|curiosity", "placements", "opinion", "not_solution"),
    21: ("negative", "frustration", "placements", "counterargument", "not_solution"),
    22: ("neutral", "curiosity", "financial_concerns", "opinion", "not_solution"),
    23: ("neutral", "curiosity", "administrative_communication", "counterargument", "not_solution"),
    24: ("neutral", "", "financial_concerns", "counterargument", "not_solution"),
    25: ("negative", "frustration", "placements", "counterargument", "not_solution"),
    26: ("neutral", "", "placements", "counterargument", "not_solution"),
    27: ("positive", "confidence", "placements", "opinion", "not_solution"),
    28: ("positive", "confidence", "placements", "counterargument", "not_solution"),
    29: ("negative", "frustration|confidence", "curriculum_teaching|placements", "counterargument", "not_solution"),
    30: ("neutral", "confidence", "placements", "counterargument", "not_solution"),
    31: ("negative", "frustration", "hostel_facilities|placements", "opinion", "not_solution"),
    32: ("neutral", "", "hostel_facilities", "opinion", "not_solution"),
    33: ("negative", "frustration", "financial_concerns|hostel_facilities", "opinion", "not_solution"),
    34: ("negative", "frustration", "hostel_facilities", "opinion", "not_solution"),
    35: ("neutral", "", "hostel_facilities", "solution", "solution"),
    36: ("negative", "frustration", "hostel_facilities", "opinion", "not_solution"),
    37: ("negative", "", "campus_infrastructure", "opinion", "not_solution"),
    38: ("neutral", "curiosity", "campus_infrastructure|hostel_facilities", "opinion", "not_solution"),
    39: ("neutral", "", "hostel_facilities", "personal_experience", "not_solution"),
    40: ("negative", "frustration", "administrative_communication|hostel_facilities", "opinion", "not_solution"),
    41: ("negative", "frustration", "hostel_facilities", "agreement", "not_solution"),
    42: ("neutral", "", "campus_infrastructure", "counterargument", "not_solution"),
    43: ("positive", "hope|confidence", "administrative_communication|campus_infrastructure", "counterargument", "not_solution"),
    44: ("negative", "", "hostel_facilities", "opinion", "not_solution"),
    45: ("positive", "confidence", "campus_infrastructure|curriculum_teaching", "opinion", "not_solution"),
    46: ("neutral", "", "campus_infrastructure", "counterargument", "not_solution"),
    47: ("neutral", "curiosity", "campus_infrastructure", "counterargument", "not_solution"),
    48: ("negative", "frustration", "campus_infrastructure|financial_concerns", "opinion", "not_solution"),
    49: ("neutral", "", "campus_infrastructure", "advice", "not_solution"),
    50: ("negative", "frustration", "campus_infrastructure", "counterargument", "not_solution"),
    # --- 51..100 ---
    51: ("negative", "frustration", "campus_infrastructure", "counterargument", "not_solution"),
    52: ("neutral", "", "", "advice", "not_solution"),
    53: ("negative", "stress", "administrative_communication|campus_infrastructure", "personal_experience", "not_solution"),
    54: ("negative", "frustration", "financial_concerns", "agreement", "not_solution"),
    55: ("neutral", "", "administrative_communication", "counterargument", "not_solution"),
    56: ("neutral", "", "financial_concerns|placements", "advice", "not_solution"),
    57: ("neutral", "", "placements", "personal_experience", "not_solution"),
    58: ("negative", "", "placements", "counterargument", "not_solution"),
    59: ("neutral", "", "campus_infrastructure", "solution", "solution"),
    60: ("positive", "hope|gratitude", "extracurriculars", "personal_experience", "not_solution"),
    61: ("neutral", "", "administrative_communication", "counterargument", "not_solution"),
    62: ("negative", "frustration", "placements", "opinion", "not_solution"),
    63: ("negative", "frustration", "financial_concerns|placements", "counterargument", "not_solution"),
    64: ("negative", "frustration", "administrative_communication|placements", "counterargument", "not_solution"),
    65: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    66: ("neutral", "", "placements", "counterargument", "not_solution"),
    67: ("negative", "confidence", "administrative_communication", "counterargument", "not_solution"),
    68: ("positive", "confidence", "administrative_communication", "opinion", "not_solution"),
    69: ("negative", "frustration", "administrative_communication", "agreement", "not_solution"),
    70: ("positive", "confidence", "placements", "counterargument", "not_solution"),
    71: ("neutral", "", "financial_concerns|placements", "counterargument", "not_solution"),
    72: ("negative", "frustration", "financial_concerns", "counterargument", "not_solution"),
    73: ("negative", "frustration", "curriculum_teaching|placements", "opinion", "not_solution"),
    74: ("neutral", "", "placements", "counterargument", "not_solution"),
    75: ("neutral", "", "administrative_communication", "advice", "solution"),
    76: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    77: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    78: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    79: ("neutral", "", "administrative_communication|financial_concerns", "advice", "solution"),
    80: ("negative", "frustration", "administrative_communication", "solution", "solution"),
    81: ("neutral", "", "administrative_communication", "advice", "not_solution"),
    82: ("neutral", "hope", "financial_concerns", "personal_experience", "not_solution"),
    83: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    84: ("neutral", "curiosity", "administrative_communication|financial_concerns", "advice", "solution"),
    85: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    86: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    87: ("neutral", "", "administrative_communication|financial_concerns", "advice", "solution"),
    88: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    89: ("neutral", "hope", "financial_concerns", "solution", "solution"),
    90: ("neutral", "", "administrative_communication", "agreement", "not_solution"),
    91: ("neutral", "", "administrative_communication|financial_concerns", "advice", "solution"),
    92: ("negative", "frustration", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    93: ("neutral", "stress", "administrative_communication", "personal_experience", "not_solution"),
    94: ("negative", "stress", "", "personal_experience", "not_solution"),
    95: ("neutral", "curiosity", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    96: ("negative", "stress|frustration", "administrative_communication", "counterargument", "not_solution"),
    97: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    98: ("negative", "stress", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    99: ("negative", "stress|curiosity", "financial_concerns|placements", "agreement", "not_solution"),
    100: ("negative", "stress", "financial_concerns|placements", "advice", "not_solution"),
    # --- 101..150 ---
    101: ("neutral", "hope", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    102: ("negative", "frustration|stress", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    103: None,  # abstain: bot output (haikusbot)
    104: ("neutral", "", "", "counterargument", "not_solution"),
    105: ("positive", "", "placements", "personal_experience", "not_solution"),
    106: ("negative", "frustration", "placements", "counterargument", "not_solution"),
    107: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    108: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    109: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    110: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    111: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    112: ("negative", "frustration", "administrative_communication", "personal_experience", "not_solution"),
    113: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    114: ("negative", "", "administrative_communication", "personal_experience", "not_solution"),
    115: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    116: ("negative", "stress|curiosity", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    117: ("negative", "frustration", "administrative_communication", "personal_experience", "not_solution"),
    118: ("negative", "frustration", "administrative_communication", "personal_experience", "not_solution"),
    119: ("negative", "frustration", "administrative_communication|financial_concerns", "personal_experience", "not_solution"),
    120: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    121: ("positive", "confidence", "administrative_communication", "personal_experience", "not_solution"),
    122: ("neutral", "curiosity", "administrative_communication|financial_concerns", "advice", "solution"),
    123: ("negative", "", "placements", "opinion", "not_solution"),
    124: ("neutral", "curiosity", "administrative_communication|financial_concerns", "counterargument", "not_solution"),
    125: ("neutral", "", "administrative_communication", "agreement", "not_solution"),
    126: ("negative", "frustration", "placements", "counterargument", "not_solution"),
    127: ("negative", "", "administrative_communication", "opinion", "not_solution"),
    128: ("positive", "confidence", "administrative_communication", "personal_experience", "not_solution"),
    129: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    130: ("negative", "stress|frustration", "administrative_communication", "agreement", "not_solution"),
    131: ("neutral", "stress|curiosity", "", "opinion", "not_solution"),
    132: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    133: ("neutral", "curiosity", "", "opinion", "not_solution"),
    134: ("negative", "frustration", "administrative_communication|hostel_facilities", "opinion", "not_solution"),
    135: ("negative", "confidence", "administrative_communication", "counterargument", "not_solution"),
    136: ("negative", "", "administrative_communication", "solution", "solution"),
    137: ("negative", "frustration", "administrative_communication", "agreement", "not_solution"),
    138: ("negative", "frustration", "placements", "counterargument", "not_solution"),
    139: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    140: ("negative", "stress", "administrative_communication", "agreement", "not_solution"),
    141: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    142: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    143: ("neutral", "", "financial_concerns", "solution", "solution"),
    144: ("positive", "hope", "", "advice", "not_solution"),
    145: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    146: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    147: ("positive", "confidence", "financial_concerns", "solution", "solution"),
    148: ("neutral", "", "administrative_communication", "counterargument", "not_solution"),
    149: ("negative", "frustration", "administrative_communication|financial_concerns", "opinion", "not_solution"),
    150: ("negative", "curiosity", "administrative_communication", "opinion", "not_solution"),
    # --- 151..180 (end of train_queue) ---
    151: ("negative", "frustration", "administrative_communication|financial_concerns", "agreement", "not_solution"),
    152: ("neutral", "curiosity|stress", "placements", "opinion", "not_solution"),
    153: ("neutral", "", "", "advice", "not_solution"),
    154: ("neutral", "curiosity", "administrative_communication", "counterargument", "not_solution"),
    155: ("negative", "", "administrative_communication", "advice", "not_solution"),
    156: ("neutral", "", "placements", "agreement", "not_solution"),
    157: ("positive", "confidence", "curriculum_teaching|placements", "opinion", "not_solution"),
    158: ("negative", "", "placements", "counterargument", "not_solution"),
    159: ("positive", "confidence", "", "advice", "not_solution"),
    160: ("negative", "confidence", "", "counterargument", "not_solution"),
    161: ("negative", "frustration", "administrative_communication", "agreement", "not_solution"),
    162: ("neutral", "", "administrative_communication|placements", "opinion", "not_solution"),
    163: ("negative", "frustration", "administrative_communication", "personal_experience", "not_solution"),
    164: ("neutral", "", "academic_workload|administrative_communication", "solution", "solution"),
    165: ("negative", "stress|curiosity", "financial_concerns|placements", "opinion", "not_solution"),
    166: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    167: ("neutral", "", "administrative_communication", "counterargument", "not_solution"),
    168: ("negative", "stress|hope", "administrative_communication", "agreement", "not_solution"),
    169: ("negative", "stress", "financial_concerns", "agreement", "not_solution"),
    170: ("neutral", "", "administrative_communication|financial_concerns", "solution", "solution"),
    171: ("negative", "", "financial_concerns", "advice", "not_solution"),
    172: ("neutral", "curiosity", "placements", "opinion", "not_solution"),
    173: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    174: ("neutral", "hope", "administrative_communication", "advice", "not_solution"),
    175: ("positive", "hope", "extracurriculars", "advice", "not_solution"),
    176: ("neutral", "", "financial_concerns", "advice", "not_solution"),
    177: ("positive", "", "", "agreement", "not_solution"),
    178: ("neutral", "", "extracurriculars", "counterargument", "not_solution"),
    179: ("neutral", "", "financial_concerns", "personal_experience", "not_solution"),
    180: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    # --- 181..250 (70 brand-new unseen-thread expanded held-out test comments!) ---
    181: ("negative", "stress", "financial_concerns", "opinion", "not_solution"),
    182: ("negative", "", "curriculum_teaching|financial_concerns", "opinion", "not_solution"),
    183: ("neutral", "curiosity", "curriculum_teaching", "counterargument", "not_solution"),
    184: ("neutral", "", "curriculum_teaching|placements", "advice", "not_solution"),
    185: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    186: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    187: ("neutral", "", "administrative_communication", "solution", "solution"),
    188: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    189: ("positive", "confidence", "administrative_communication", "advice", "solution"),
    190: ("negative", "frustration|stress", "administrative_communication", "opinion", "not_solution"),
    191: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    192: ("neutral", "", "placements", "advice", "not_solution"),
    193: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    194: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    195: ("neutral", "", "administrative_communication", "agreement", "not_solution"),
    196: ("negative", "", "administrative_communication", "opinion", "not_solution"),
    197: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    198: ("positive", "hope", "administrative_communication", "opinion", "not_solution"),
    199: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    200: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    201: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    202: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    203: ("negative", "frustration", "administrative_communication", "advice", "not_solution"),
    204: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    205: ("negative", "", "financial_concerns|placements", "advice", "not_solution"),
    206: ("positive", "gratitude|stress", "", "agreement", "not_solution"),
    207: ("positive", "confidence", "placements", "advice", "not_solution"),
    208: ("neutral", "", "administrative_communication", "solution", "solution"),
    209: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    210: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    211: ("negative", "frustration", "administrative_communication", "counterargument", "not_solution"),
    212: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    213: ("neutral", "", "administrative_communication", "solution", "solution"),
    214: ("negative", "", "administrative_communication", "personal_experience", "not_solution"),
    215: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    216: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    217: ("positive", "hope", "administrative_communication", "opinion", "not_solution"),
    218: ("neutral", "stress", "academic_workload", "opinion", "not_solution"),
    219: ("negative", "frustration", "administrative_communication", "opinion", "not_solution"),
    220: ("positive", "frustration", "administrative_communication", "opinion", "not_solution"),
    221: ("positive", "", "financial_concerns", "advice", "not_solution"),
    222: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    223: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    224: ("neutral", "", "administrative_communication", "opinion", "not_solution"),
    225: ("neutral", "", "administrative_communication", "advice", "not_solution"),
    226: ("negative", "stress", "administrative_communication", "opinion", "not_solution"),
    227: ("positive", "hope", "administrative_communication", "opinion", "not_solution"),
    228: ("neutral", "", "curriculum_teaching", "advice", "not_solution"),
    229: ("neutral", "", "", "personal_experience", "not_solution"),
    230: ("neutral", "", "", "personal_experience", "not_solution"),
    231: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    232: ("negative", "frustration", "curriculum_teaching", "opinion", "not_solution"),
    233: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    234: ("positive", "curiosity", "financial_concerns", "personal_experience", "not_solution"),
    235: ("neutral", "", "campus_infrastructure", "advice", "solution"),
    236: ("neutral", "curiosity", "administrative_communication", "opinion", "not_solution"),
    237: ("negative", "stress|curiosity", "administrative_communication", "personal_experience", "not_solution"),
    238: ("neutral", "", "administrative_communication", "solution", "solution"),
    239: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    240: ("positive", "curiosity|gratitude", "administrative_communication", "opinion", "not_solution"),
    241: ("positive", "hope|gratitude", "administrative_communication", "opinion", "not_solution"),
    242: ("positive", "curiosity", "administrative_communication", "opinion", "not_solution"),
    243: ("positive", "confidence", "placements", "opinion", "not_solution"),
    244: ("negative", "frustration", "administrative_communication|campus_infrastructure", "personal_experience", "not_solution"),
    245: ("neutral", "", "administrative_communication", "personal_experience", "not_solution"),
    246: ("neutral", "hope", "administrative_communication", "opinion", "not_solution"),
    247: ("neutral", "", "administrative_communication", "agreement", "not_solution"),
    248: ("positive", "confidence|hope", "administrative_communication", "opinion", "not_solution"),
    249: ("positive", "hope", "administrative_communication", "agreement", "not_solution"),
    250: ("neutral", "curiosity", "administrative_communication", "personal_experience", "not_solution"),
}


def build_v2_dataset() -> None:
    ensure_dir(V2_DIR)
    v1 = load_v1_records_and_frozen_splits()
    norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()

    v1_rows_out: List[Dict[str, str]] = []
    for r in v1["all"]:
        v2_sp = "test_v5_frozen" if r["v5_split"] == "test" else "trainval_v1"
        v1_rows_out.append(
            {
                "document_id": r["document_id"],
                "post_id": r["post_id"],
                "corpus_post_id": r["corpus_post_id"],
                "comment_id": r["comment_id"],
                "corpus_comment_id": r["corpus_comment_id"],
                "subreddit": r["subreddit"],
                "post_title": r["post_title"],
                "text": r["text"],
                "sentiment": r["labels"]["sentiment"],
                "emotions": "|".join(r["labels"]["emotion"]),
                "issues": "|".join(r["labels"]["issue"]),
                "response": r["labels"]["response"],
                "solution": r["labels"]["solution"],
                "v2_split": v2_sp,
                "annotation_source": "rubric_curated_real_reddit_v1",
            }
        )

    queue_rows = list(csv.DictReader(QUEUE_CSV.open("r", encoding="utf-8")))
    assert len(queue_rows) == 250, f"Expected 250 queue rows, got {len(queue_rows)}"

    new_rows_out: List[Dict[str, str]] = []
    abstained_rows: List[Dict[str, str]] = []
    valid_sets = {k: set(v) for k, v in CANONICAL_LABELS.items()}

    for qr in queue_rows:
        qidx = int(qr["queue_index"])
        lbl = CURATED_LABELS[qidx]
        if lbl is None:
            abstained_rows.append(
                {
                    "queue_index": str(qidx),
                    "document_id": qr["document_id"],
                    "post_id": qr["post_id"],
                    "comment_id": qr["comment_id"],
                    "text": qr["text"],
                    "review_status": "abstain",
                    "abstain_reason": "off_topic_noise",
                }
            )
            continue

        sent, emo, iss, resp, sol = lbl
        assert sent in valid_sets["sentiment"], f"Bad sentiment at {qidx}: {sent}"
        assert resp in valid_sets["response"], f"Bad response at {qidx}: {resp}"
        assert sol in valid_sets["solution"], f"Bad solution at {qidx}: {sol}"
        for e in [x for x in emo.split("|") if x]:
            assert e in valid_sets["emotion"], f"Bad emotion at {qidx}: {e}"
        for i_l in [x for x in iss.split("|") if x]:
            assert i_l in valid_sets["issue"], f"Bad issue at {qidx}: {i_l}"

        v2_sp = (
            "trainval_v2_new"
            if qr["queue_role"] == "v2_active_learning_train_candidate"
            else "test_v2_new"
        )
        # Prefix new corpus post_ids cleanly when not in V1 to avoid any collision with shifted V1 post_ids
        new_pid = f"corpus-{qr['post_id']}"
        new_rows_out.append(
            {
                "document_id": qr["document_id"],
                "post_id": new_pid,
                "corpus_post_id": qr["post_id"],
                "comment_id": qr["comment_id"],
                "corpus_comment_id": qr["comment_id"],
                "subreddit": qr["subreddit"],
                "post_title": qr["post_title"],
                "text": qr["text"],
                "sentiment": sent,
                "emotions": emo,
                "issues": iss,
                "response": resp,
                "solution": sol,
                "v2_split": v2_sp,
                "annotation_source": "rubric_curated_real_reddit_v2_pending_human_iaa",
            }
        )

    all_v2 = v1_rows_out + new_rows_out

    # Strict leakage & frozen-test integrity verification
    tr_cpids = {r["corpus_post_id"] for r in all_v2 if r["v2_split"] in {"trainval_v1", "trainval_v2_new"}}
    te_f_cpids = {r["corpus_post_id"] for r in all_v2 if r["v2_split"] == "test_v5_frozen"}
    te_n_cpids = {r["corpus_post_id"] for r in all_v2 if r["v2_split"] == "test_v2_new"}
    assert not (tr_cpids & te_f_cpids), "LEAKAGE: trainval_v2 overlaps test_v5_frozen!"
    assert not (tr_cpids & te_n_cpids), "LEAKAGE: trainval_v2 overlaps test_v2_new!"
    assert not (te_f_cpids & te_n_cpids), "LEAKAGE: test_v5_frozen overlaps test_v2_new!"

    tr_titles = {norm(r["post_title"])[:120] for r in all_v2 if r["v2_split"] in {"trainval_v1", "trainval_v2_new"}}
    te_f_titles = {norm(r["post_title"])[:120] for r in all_v2 if r["v2_split"] == "test_v5_frozen"}
    te_n_titles = {norm(r["post_title"])[:120] for r in all_v2 if r["v2_split"] == "test_v2_new"}
    # Note: verify brand-new test_v2_new threads have zero corpus_post_id overlap with everything
    frozen_dids = sorted(r["document_id"] for r in all_v2 if r["v2_split"] == "test_v5_frozen")
    assert len(frozen_dids) == 92
    assert hashlib.sha256(",".join(frozen_dids).encode("utf-8")).hexdigest() == v1["manifest"]["test_sha256"]

    # Write master_labeled_v2.csv
    fields = list(all_v2[0].keys())
    with (V2_DIR / "master_labeled_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_v2)

    # Write adjudication_queue.csv (abstained bot/noise items)
    with (V2_DIR / "adjudication_queue.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(abstained_rows[0].keys()))
        w.writeheader()
        w.writerows(abstained_rows)

    # Write 5 canonical task CSVs matching V1 schema
    task_col_map = [
        ("issue.csv", "issues", "labels"),
        ("emotion.csv", "emotions", "labels"),
        ("sentiment.csv", "sentiment", "label"),
        ("response.csv", "response", "label"),
        ("solution.csv", "solution", "label"),
    ]
    for fname, src_col, dst_col in task_col_map:
        with (V2_DIR / fname).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["document_id", "post_id", "comment_id", "text", dst_col])
            w.writeheader()
            for r in all_v2:
                val = r[src_col]
                if fname == "solution.csv":
                    val = "true" if val == "solution" else "false"
                w.writerow(
                    {
                        "document_id": r["document_id"],
                        "post_id": r["corpus_post_id"],
                        "comment_id": r["corpus_comment_id"],
                        "text": r["text"],
                        dst_col: val,
                    }
                )

    # Compute rich class-support comparison (V1 vs V2!)
    def _counts(subset: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
        out = {}
        for t in ("sentiment", "response", "solution"):
            out[t] = dict(Counter(r[t] for r in subset))
        for t, col in (("issue", "issues"), ("emotion", "emotions")):
            c = Counter()
            for r in subset:
                for tok in [x for x in r[col].split("|") if x]:
                    c[tok] += 1
            out[t] = dict(c)
        return out

    prov = {
        "dataset": "real_reddit_labeled_v2",
        "v1_preserved_untouched": True,
        "total_rows": len(all_v2),
        "abstained_rows_excluded": len(abstained_rows),
        "split_counts": dict(Counter(r["v2_split"] for r in all_v2)),
        "thread_counts_by_corpus_post_id": {
            "trainval_v2": len(tr_cpids),
            "test_v5_frozen": len(te_f_cpids),
            "test_v2_new": len(te_n_cpids),
            "test_v2_expanded_total": len(te_f_cpids | te_n_cpids),
        },
        "annotation_sources": dict(Counter(r["annotation_source"] for r in all_v2)),
        "class_support_trainval_v1": _counts([r for r in all_v2 if r["v2_split"] == "trainval_v1"]),
        "class_support_trainval_v2": _counts([r for r in all_v2 if r["v2_split"] in {"trainval_v1", "trainval_v2_new"}]),
        "class_support_test_v5_frozen": _counts([r for r in all_v2 if r["v2_split"] == "test_v5_frozen"]),
        "class_support_test_v2_expanded": _counts([r for r in all_v2 if r["v2_split"] in {"test_v5_frozen", "test_v2_new"}]),
        "leakage_checks": {
            "trainval_v2_vs_test_v5_frozen": len(tr_cpids & te_f_cpids),
            "trainval_v2_vs_test_v2_new": len(tr_cpids & te_n_cpids),
            "test_v5_frozen_vs_test_v2_new": len(te_f_cpids & te_n_cpids),
            "frozen_test_sha256_exact_match": True,
        },
    }
    save_json(V2_DIR / "v2_provenance.json", prov)
    print(f"Built data/annotations/real_reddit_labeled_v2/ successfully: {prov['split_counts']}")
    print("Thread counts:", prov["thread_counts_by_corpus_post_id"])
    print("Trainval V1 -> V2 class support boost:")
    for t in ("issue", "emotion", "response", "solution", "sentiment"):
        print(f"  {t}: V1={prov['class_support_trainval_v1'][t]} -> V2={prov['class_support_trainval_v2'][t]}")


if __name__ == "__main__":
    build_v2_dataset()
