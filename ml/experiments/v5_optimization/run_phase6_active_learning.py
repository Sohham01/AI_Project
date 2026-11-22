"""Phase 6 — Active Learning Annotation Queue & IAA Package Generator (Section 10 & Item #18).

Strict integrity rules:
- Samples strictly from existing anonymized real comments in data/live_corpus/corpus.json
  (ZERO external Reddit/Arctic Shift scraping).
- Completely excludes ALL 37 frozen V5 held-out test threads AND all 459 already-labeled V1 comments.
- Combines real acquisition criteria:
  1. MuRIL prediction uncertainty (entropy / low max probability)
  2. Smallest margin between competing classes (response, sentiment)
  3. MuRIL vs Word+Char TF-IDF disagreement
  4. Rare-class lexical + semantic candidates (academic_workload, curriculum_teaching,
     hostel_facilities, campus_infrastructure, counterargument, solution, stress, frustration)
  5. Context-dependent replies (real parent comment present)
  6. Thread-diversity cap (<= 4 comments per post_id) so no single thread dominates.
- Reserves a strictly disjoint block of completely unlabeled threads (~30-35 new unseen threads,
  ~70 comments) for expanded held-out real testing (Item #18), completely separate from the
  ~180 active-learning training candidates!
- Outputs BLANK human-annotation CSVs (NEVER fills label columns with model predictions)
  plus a 50-comment IAA double-annotation QA pack.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    CKPT_ROOT,
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TASK_NAMES,
    TrainSpec,
    apply_logits_calibration,
    encode_task_targets,
    ensure_dir,
    load_corpus_lookups,
    load_v1_records_and_frozen_splits,
    save_json,
)
from ml.experiments.v5_optimization.run_phase1_cv import _train_fixed_epochs_all_trainval

RARE_PATTERNS: Dict[str, re.Pattern] = {
    "academic_workload": re.compile(
        r"\b(quiz|quizzes|assignment|assignments|deadline|workload|attendance|rigor|rigorous|sleep|exam|exams|midterm|endterm|term\s*[123456]|cgpa|gpa|grade|grades|grading|pressure|hectic|backlog|credits)\b",
        re.I,
    ),
    "curriculum_teaching": re.compile(
        r"\b(prof|profs|professor|faculty|teaching|curriculum|course|courses|elective|electives|pedagogy|case\s*study|cases|syllabus|lecture|lectures|visiting\s*faculty|subject|subjects|specialization)\b",
        re.I,
    ),
    "hostel_facilities": re.compile(
        r"\b(hostel|hostels|mess|room|rooms|roommate|ac|cooler|water|laundry|warden|food|canteen|washroom|bathroom|single\s*room|twin\s*sharing|hygiene)\b",
        re.I,
    ),
    "campus_infrastructure": re.compile(
        r"\b(campus|wifi|wi-fi|library|classroom|lab|labs|shuttle|transport|airport|station|construction|gym|sports|medical|hospital|building|infrastructure|city|location)\b",
        re.I,
    ),
    "counterargument": re.compile(
        r"\b(actually|disagree|not\s+true|wrong|incorrect|myth|false|but\s+that|on\s+the\s+contrary|nope|nah|nahi\s+bhai|galat|misleading|not\s+really|contrary|fact\s+check|however\s+you)\b",
        re.I,
    ),
    "solution": re.compile(
        r"\b(mail|email|portal|contact|apply|submit|attached|document|refund|guideline|aicte|ugc|step|steps|procedure|workaround|fix|solve|link|form|scholarship|bank\s+loan|sbi|hdfc|check\s+the)\b",
        re.I,
    ),
    "stress": re.compile(
        r"\b(stress|stressed|anxious|anxiety|worried|worry|panic|scared|tense|overwhelmed|burnout|depressed|tension|darr|confused|helpless|nervous|freaking\s+out)\b",
        re.I,
    ),
    "frustration": re.compile(
        r"\b(frustrat|annoy|ridiculous|pathetic|worst|horrible|terrible|scam|bakwas|bekar|useless|waste|tired\s+of|fed\s+up|disappoint|shame|toxic|unfair|sucks|suck|bullshit)\b",
        re.I,
    ),
}


def _fit_predict_tfidf(train_rows: List[Dict[str, Any]], pool_rows: List[Dict[str, Any]], task: str) -> np.ndarray:
    tr_texts = [f"{r['post_title']} {r['text']}" if task == "issue" else r["text"] for r in train_rows]
    pl_texts = [f"{r['post_title']} {r['text']}" if task == "issue" else r["text"] for r in pool_rows]
    w_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, sublinear_tf=True)
    X_tr = hstack([w_vec.fit_transform(tr_texts), c_vec.fit_transform(tr_texts)])
    X_pl = hstack([w_vec.transform(pl_texts), c_vec.transform(pl_texts)])
    y_tr = encode_task_targets(train_rows, task)
    if task in MULTILABEL_TASKS:
        clf = OneVsRestClassifier(LogisticRegression(C=2.0, max_iter=500, class_weight="balanced"))
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_pl).astype(np.float32)
    clf = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_pl).astype(np.float32)


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-7, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])


def _margin(probs: np.ndarray) -> np.ndarray:
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def run_phase6(n_train_queue: int = 180, n_test_new_queue: int = 70) -> Dict[str, Any]:
    out_dir = ensure_dir(EXP_ROOT / "active_learning")
    t0 = time.perf_counter()

    v1 = load_v1_records_and_frozen_splits()
    trainval_v1 = v1["trainval"]
    norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()
    test_pids = set(v1["manifest"]["test_corpus_post_ids"])
    test_titles_120 = {norm(r["post_title"])[:120] for r in v1["test"] if r.get("post_title")}
    v1_trainval_pids = set(v1["manifest"]["trainval_corpus_post_ids"])
    v1_cids = {r["corpus_comment_id"] for r in v1["all"]}
    v1_norm_texts = {norm(r["text"]) for r in v1["all"]}

    corpus, _, cmt_by_cid = load_corpus_lookups()

    pool: List[Dict[str, Any]] = []
    for p in corpus["posts"]:
        pid = p["reddit_id"]
        title = (p.get("title") or "").strip()
        if pid in test_pids or norm(title)[:120] in test_titles_120:
            continue  # STRICT: never touch frozen V5 test threads (checked by corpus_post_id AND title)!
        sub = p.get("subreddit") or ""
        in_post_cmts = {x["reddit_id"]: x for x in p.get("comments") or []}
        for c in p.get("comments") or []:
            cid = c["reddit_id"]
            txt = " ".join(str(c.get("body") or "").split())
            if cid in v1_cids or norm(txt) in v1_norm_texts:
                continue
            if len(txt.split()) < 6:
                continue
            if txt.lower() in {"[deleted]", "[removed]"} or txt.lower().startswith("http"):
                continue
            prid = c.get("parent_reddit_id")
            parent_txt = str(in_post_cmts[prid].get("body") or "").strip() if (prid and prid in in_post_cmts) else ""
            pool.append(
                {
                    "document_id": f"doc-v2-{cid}",
                    "post_id": pid,
                    "comment_id": cid,
                    "subreddit": sub,
                    "post_title": title,
                    "parent_comment_id": prid if parent_txt else "",
                    "parent_comment_text": parent_txt,
                    "parent_context_status": "available" if parent_txt else "top_level",
                    "text": txt,
                    "is_completely_unlabeled_thread": pid not in v1_trainval_pids,
                }
            )

    pool_pids = {r["post_id"] for r in pool}
    assert not (pool_pids & test_pids), "CRITICAL LEAKAGE: test post_id in active learning pool!"
    assert not ({r["comment_id"] for r in pool} & v1_cids), "Already labeled V1 comment found in pool!"

    # Select best available inductive DAPT backbone & Phase 2/4 settings
    p2_path = EXP_ROOT / "phase2_context" / "phase2_summary.json"
    p4_path = EXP_ROOT / "phase4_losses" / "phase4_summary.json"
    p5_path = EXP_ROOT / "phase5_dapt_backbones" / "phase5_summary.json"
    p2 = json.loads(p2_path.read_text(encoding="utf-8")) if p2_path.exists() else {}
    p4 = json.loads(p4_path.read_text(encoding="utf-8")) if p4_path.exists() else {}
    p5 = json.loads(p5_path.read_text(encoding="utf-8")) if p5_path.exists() else {}

    strict_dapt = CKPT_ROOT / "muril_dapt_strict_inductive_wwm_8ep"
    if (strict_dapt / "config.json").exists():
        backbone = str(strict_dapt)
    else:
        best_dapt = p5.get("best_inductive_dapt_tag", "muril_dapt_inductive_mlm_4ep")
        backbone = str(CKPT_ROOT / best_dapt)

    print(f"[Phase 6] Scoring {len(pool)} eligible unlabeled real comments ({len(pool_pids)} non-test threads)...", flush=True)

    # Fit MuRIL (on trainval_v1 ONLY) & TF-IDF (on trainval_v1 ONLY) for all 5 tasks to compute acquisition scores
    muril_probs: Dict[str, np.ndarray] = {}
    tfidf_probs: Dict[str, np.ndarray] = {}
    for task in TASK_NAMES:
        ctx = p2.get("selected_context_by_task", {}).get(task, {}).get("selected_mode", "comment_only")
        loss = p4.get("selected_loss_by_task", {}).get(task, {}).get("selected_loss", "default")
        lg, _, _ = _train_fixed_epochs_all_trainval(
            TrainSpec(task=task, backbone_id=backbone, context_mode=ctx, loss_type=loss, max_length=256, epochs=4, seed=SEED + 600),
            trainval_v1,
            pool,
        )
        muril_probs[task] = apply_logits_calibration(lg, task in MULTILABEL_TASKS, 1.0)
        tfidf_probs[task] = _fit_predict_tfidf(trainval_v1, pool, task)

    # Compute rich acquisition scores per candidate
    n_pool = len(pool)
    ent_resp = _entropy(muril_probs["response"])
    ent_sent = _entropy(muril_probs["sentiment"])
    margin_resp = _margin(muril_probs["response"])
    margin_sent = _margin(muril_probs["sentiment"])

    disagree_resp = (np.argmax(muril_probs["response"], axis=1) != np.argmax(tfidf_probs["response"], axis=1)).astype(float)
    disagree_sent = (np.argmax(muril_probs["sentiment"], axis=1) != np.argmax(tfidf_probs["sentiment"], axis=1)).astype(float)
    disagree_sol = (np.argmax(muril_probs["solution"], axis=1) != np.argmax(tfidf_probs["solution"], axis=1)).astype(float)
    disagree_score = (disagree_resp + disagree_sent + disagree_sol) / 3.0

    # Rare-class model + lexical relevance score
    idx_aw = CANONICAL_LABELS["issue"].index("academic_workload")
    idx_ct = CANONICAL_LABELS["issue"].index("curriculum_teaching")
    idx_hf = CANONICAL_LABELS["issue"].index("hostel_facilities")
    idx_ci = CANONICAL_LABELS["issue"].index("campus_infrastructure")
    idx_ctr = CANONICAL_LABELS["response"].index("counterargument")
    idx_sol_r = CANONICAL_LABELS["response"].index("solution")
    idx_str = CANONICAL_LABELS["emotion"].index("stress")
    idx_fru = CANONICAL_LABELS["emotion"].index("frustration")

    for i, r in enumerate(pool):
        combo = f"{r['post_title']} {r['text']}"
        matched_rare = [k for k, pat in RARE_PATTERNS.items() if pat.search(combo)]
        # Soft probability signals for target rare classes
        prob_signals = {
            "academic_workload": float(muril_probs["issue"][i, idx_aw]),
            "curriculum_teaching": float(muril_probs["issue"][i, idx_ct]),
            "hostel_facilities": float(muril_probs["issue"][i, idx_hf]),
            "campus_infrastructure": float(muril_probs["issue"][i, idx_ci]),
            "counterargument": float(muril_probs["response"][i, idx_ctr]),
            "solution": float(max(muril_probs["response"][i, idx_sol_r], muril_probs["solution"][i, 1])),
            "stress": float(muril_probs["emotion"][i, idx_str]),
            "frustration": float(muril_probs["emotion"][i, idx_fru]),
        }
        top_rare = max(prob_signals.values())
        unc = 0.5 * (float(ent_resp[i]) + float(ent_sent[i]))
        inv_margin = 0.5 * ((1.0 - float(margin_resp[i])) + (1.0 - float(margin_sent[i])))
        has_parent_bonus = 0.12 if r["parent_comment_text"] else 0.0
        rare_lex_bonus = min(0.30, 0.12 * len(matched_rare))
        total_acq = 0.30 * unc + 0.20 * inv_margin + 0.20 * float(disagree_score[i]) + 0.25 * top_rare + rare_lex_bonus + has_parent_bonus

        reasons = []
        if unc >= 0.72:
            reasons.append("high_entropy")
        if inv_margin >= 0.65:
            reasons.append("small_margin")
        if disagree_score[i] >= 0.66:
            reasons.append("muril_tfidf_disagreement")
        if matched_rare:
            reasons.append("rare_kw:" + "+".join(matched_rare[:3]))
        if r["parent_comment_text"]:
            reasons.append("parent_context_reply")

        r["_acq_score"] = round(float(total_acq), 4)
        r["_uncertainty"] = round(float(unc), 4)
        r["_min_margin"] = round(float(min(margin_resp[i], margin_sent[i])), 4)
        r["_disagreement"] = round(float(disagree_score[i]), 4)
        r["_rare_targets"] = matched_rare
        r["_prob_signals"] = {k: round(v, 3) for k, v in prob_signals.items()}
        r["_reasons"] = "|".join(reasons) if reasons else "diversity"

    # STEP A: Carve out a strictly disjoint set of COMPLETELY UNLABELED threads for expanded real test (Item #18)
    # Sample ~35 unseen threads stratified across subreddits (~70 comments total) with NO selection bias toward model errors!
    # (Held-out test set must be a representative unseen-thread sample, NOT active-learning hard-mined!)
    unlabeled_threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        if r["is_completely_unlabeled_thread"]:
            unlabeled_threads[r["post_id"]].append(r)

    rng = np.random.default_rng(SEED + 618)
    ul_pids = sorted(unlabeled_threads.keys())
    rng.shuffle(ul_pids)

    test_new_rows: List[Dict[str, Any]] = []
    test_new_pids: set[str] = set()
    for pid in ul_pids:
        if len(test_new_rows) >= n_test_new_queue:
            break
        cmts = unlabeled_threads[pid]
        # Take up to 3 representative comments from this unseen thread
        take = cmts[: min(3, len(cmts), n_test_new_queue - len(test_new_rows))]
        if take:
            test_new_pids.add(pid)
            for item in take:
                test_new_rows.append({**item, "queue_role": "v2_expanded_test_candidate"})

    # STEP B: Select n_train_queue active-learning training candidates STRICTLY excluding test_pids AND test_new_pids!
    train_pool = [r for r in pool if r["post_id"] not in test_new_pids]
    assert not ({r["post_id"] for r in train_pool} & (test_pids | test_new_pids))

    selected_train: List[Dict[str, Any]] = []
    selected_cids: set[str] = set()
    thread_counts: Counter = Counter()
    max_per_thread = 4

    # First ensure guaranteed quota for each priority rare target (16 each where available)
    priority_targets = [
        "academic_workload",
        "curriculum_teaching",
        "hostel_facilities",
        "campus_infrastructure",
        "counterargument",
        "solution",
        "stress",
        "frustration",
    ]
    for tgt in priority_targets:
        cands = [
            r
            for r in train_pool
            if r["comment_id"] not in selected_cids
            and (tgt in r["_rare_targets"] or r["_prob_signals"][tgt] >= 0.30)
        ]
        cands.sort(key=lambda r: (r["_prob_signals"][tgt] + 0.5 * r["_acq_score"]), reverse=True)
        added = 0
        for r in cands:
            if added >= 15 or len(selected_train) >= n_train_queue:
                break
            if thread_counts[r["post_id"]] >= max_per_thread:
                continue
            selected_cids.add(r["comment_id"])
            thread_counts[r["post_id"]] += 1
            selected_train.append({**r, "queue_role": "v2_active_learning_train_candidate", "primary_bucket": tgt})
            added += 1

    # Fill remaining slots by highest combined active-learning score with thread-diversity cap
    remaining = [r for r in train_pool if r["comment_id"] not in selected_cids]
    remaining.sort(key=lambda r: r["_acq_score"], reverse=True)
    for r in remaining:
        if len(selected_train) >= n_train_queue:
            break
        if thread_counts[r["post_id"]] >= max_per_thread:
            continue
        selected_cids.add(r["comment_id"])
        thread_counts[r["post_id"]] += 1
        selected_train.append({**r, "queue_role": "v2_active_learning_train_candidate", "primary_bucket": "uncertainty_disagreement"})

    full_queue = selected_train + [{**r, "primary_bucket": "representative_unseen_test_thread"} for r in test_new_rows]
    # Final leakage assertions
    tr_q_pids = {r["post_id"] for r in selected_train}
    te_q_pids = {r["post_id"] for r in test_new_rows}
    assert not (tr_q_pids & test_pids), "Train queue overlaps frozen V5 test!"
    assert not (te_q_pids & test_pids), "New test queue overlaps frozen V5 test!"
    assert not (tr_q_pids & te_q_pids), "Train queue overlaps new test queue!"
    assert not (te_q_pids & v1_trainval_pids), "New test queue overlaps V1 trainval threads!"

    # Write BLANK human-annotation CSV (strictly NO model predictions in label columns!)
    fieldnames = [
        "queue_index",
        "queue_role",
        "primary_bucket",
        "document_id",
        "post_id",
        "comment_id",
        "subreddit",
        "parent_context_status",
        "post_title",
        "parent_comment_text",
        "text",
        "acquisition_score",
        "uncertainty",
        "min_margin",
        "muril_tfidf_disagreement",
        "selection_reasons",
        # Blank human annotation columns matching ANNOTATION_GUIDELINES.md:
        "sentiment",
        "emotions",
        "issues",
        "response_type",
        "is_solution",
        "review_status",
        "abstain_reason",
        "annotator_notes",
    ]
    queue_csv = out_dir / "annotation_queue_250.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, r in enumerate(full_queue, start=1):
            w.writerow(
                {
                    "queue_index": idx,
                    "queue_role": r["queue_role"],
                    "primary_bucket": r["primary_bucket"],
                    "document_id": r["document_id"],
                    "post_id": r["post_id"],
                    "comment_id": r["comment_id"],
                    "subreddit": r["subreddit"],
                    "parent_context_status": r["parent_context_status"],
                    "post_title": r["post_title"],
                    "parent_comment_text": r["parent_comment_text"],
                    "text": r["text"],
                    "acquisition_score": r["_acq_score"],
                    "uncertainty": r["_uncertainty"],
                    "min_margin": r["_min_margin"],
                    "muril_tfidf_disagreement": r["_disagreement"],
                    "selection_reasons": r["_reasons"],
                    "sentiment": "",
                    "emotions": "",
                    "issues": "",
                    "response_type": "",
                    "is_solution": "",
                    "review_status": "",
                    "abstain_reason": "",
                    "annotator_notes": "",
                }
            )

    # Export 50-item stratified IAA double-annotation QA pack (35 from train queue + 15 from new test queue)
    iaa_items = selected_train[:35] + test_new_rows[:15]
    iaa_csv = out_dir / "iaa_double_annotation_50.csv"
    with iaa_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, r in enumerate(iaa_items, start=1):
            w.writerow(
                {
                    "queue_index": idx,
                    "queue_role": r["queue_role"],
                    "primary_bucket": r.get("primary_bucket", "iaa"),
                    "document_id": r["document_id"],
                    "post_id": r["post_id"],
                    "comment_id": r["comment_id"],
                    "subreddit": r["subreddit"],
                    "parent_context_status": r["parent_context_status"],
                    "post_title": r["post_title"],
                    "parent_comment_text": r["parent_comment_text"],
                    "text": r["text"],
                    "acquisition_score": r["_acq_score"],
                    "uncertainty": r["_uncertainty"],
                    "min_margin": r["_min_margin"],
                    "muril_tfidf_disagreement": r["_disagreement"],
                    "selection_reasons": r["_reasons"],
                    "sentiment": "",
                    "emotions": "",
                    "issues": "",
                    "response_type": "",
                    "is_solution": "",
                    "review_status": "",
                    "abstain_reason": "",
                    "annotator_notes": "",
                }
            )

    summary = {
        "phase": "phase6_active_learning_queue",
        "backbone_used_for_acquisition": backbone,
        "eligible_unlabeled_pool_comments": n_pool,
        "eligible_unlabeled_pool_threads": len(pool_pids),
        "queue_total_comments": len(full_queue),
        "train_queue_comments": len(selected_train),
        "train_queue_threads": len(tr_q_pids),
        "expanded_test_queue_comments": len(test_new_rows),
        "expanded_test_queue_threads": len(te_q_pids),
        "iaa_double_annotation_pack_size": len(iaa_items),
        "bucket_distribution": dict(Counter(r["primary_bucket"] for r in full_queue)),
        "subreddit_distribution_train_queue": dict(Counter(r["subreddit"] for r in selected_train)),
        "subreddit_distribution_test_new_queue": dict(Counter(r["subreddit"] for r in test_new_rows)),
        "parent_context_present_pct": round(100.0 * sum(1 for r in full_queue if r["parent_comment_text"]) / max(len(full_queue), 1), 1),
        "leakage_checks": {
            "overlap_train_queue_and_frozen_test_threads": len(tr_q_pids & test_pids),
            "overlap_new_test_queue_and_frozen_test_threads": len(te_q_pids & test_pids),
            "overlap_train_queue_and_new_test_queue_threads": len(tr_q_pids & te_q_pids),
            "overlap_new_test_queue_and_v1_trainval_threads": len(te_q_pids & v1_trainval_pids),
        },
        "files": {
            "annotation_queue_csv": str(queue_csv),
            "iaa_double_annotation_csv": str(iaa_csv),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    save_json(out_dir / "active_learning_summary.json", summary)
    print(f"[Phase 6] Active learning queue ready: {len(full_queue)} items -> {queue_csv} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


if __name__ == "__main__":
    run_phase6()
