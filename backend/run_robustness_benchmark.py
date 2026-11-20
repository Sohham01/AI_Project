"""Comprehensive model resource + 8-category synthetic robustness benchmark.

Compares on identical leakage-free thread-grouped splits (seed=20261116):
  1. Baseline V1: Original Word (1,2) TF-IDF + Logistic Regression
  2. Upgraded V2: Word (1,2) + Char-WB (3,5) TF-IDF + Balanced Weights + Val Calibration
  3. MuRIL v2 GPU: Fine-tuned ``google/muril-base-cased`` (CUDA bf16 AMP,
     val_macro_f1 + val_loss checkpointing, validation-only threshold calibration)

Reports:
  - Held-out test metrics: Macro F1, Micro F1, per-class/per-label Precision/Recall/Support,
    Confusion Matrices (single-label), Subset Accuracy & Hamming Loss (multi-label)
  - Resource profile: training time (s), test inference latency (ms total & ms/comment),
    peak CPU memory delta (MB via tracemalloc), peak GPU VRAM (MB via torch.cuda),
    and serialized artifact size on disk (KB)
  - Separate 8-category synthetic robustness stress test:
      1. unseen_templates
      2. paraphrased_comments
      3. spelling_errors
      4. informal_language
      5. short_comments
      6. ambiguous_responses
      7. parent_post_context (evaluated BOTH comment-only AND with parent post title)
      8. code_mixed_language

SCIENTIFIC BOUNDARY:
  All stress tests here are synthetic probes designed to expose generalization
  limits of synthetic-trained baselines. They are NOT real Reddit human-annotated
  evaluation results and are stored in isolated ``ml/artifacts/robustness_benchmark/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for candidate in (str(ROOT), str(BACKEND)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from ml.configs import TaskType
from ml.data.dataset import load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, check_leakage, extract_groups, split_records
from ml.models.transformer.muril_classifier import MuRILTaskClassifier, MuRILTaskConfig
from ml.training.trainer import records_to_annotations
from app.services.nlp.issue_identification.base import IssueModelConfig
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)
from app.services.nlp.response_classification.base import ResponseModelConfig
from app.services.nlp.response_classification.tfidf_logistic_multiclass import (
    TfidfLogisticResponseClassifier,
)
from app.services.nlp.sentiment_emotion.base import (
    EmotionModelConfig,
    SentimentModelConfig,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
    TfidfLogisticEmotionClassifier,
)
from app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment import (
    TfidfLogisticSentimentClassifier,
)
from app.services.nlp.solution_detection.base import SolutionModelConfig
from app.services.nlp.solution_detection.tfidf_logistic_binary import (
    TfidfLogisticSolutionClassifier,
)

DATA_DIR = ROOT / "data" / "demo_synthetic"
OUT_DIR = ROOT / "ml" / "artifacts" / "robustness_benchmark"
SEED = 20261116
TASKS = ("sentiment", "emotion", "issue", "response", "solution")


# ---------------------------------------------------------------------------
# 8-category unseen synthetic robustness probe set (NEVER seen during training)
# ---------------------------------------------------------------------------
# Each item carries gold labels across all 5 tasks + optional parent_title so
# we can evaluate every task honestly under distribution shift.
ROBUSTNESS_PROBES: List[Dict[str, Any]] = [
    # --- 1. unseen_templates (12 items) ---
    {
        "id": "ut_01", "category": "unseen_templates",
        "parent_title": "Placement brochure figures vs actual CTC breakdown",
        "text": "Every season the brochure highlights average CTC while ignoring how many offers are variable-heavy sales profiles.",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["placements"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "ut_02", "category": "unseen_templates",
        "parent_title": "Hostel Block B water cooler and laundry queue",
        "text": "Logging a joint floor complaint on the facility maintenance portal with room numbers got our cooler fixed in two days.",
        "sentiment": "positive", "emotions": ["gratitude", "confidence"], "issues": ["hostel_facilities"],
        "response": "personal_experience", "solution": "solution",
    },
    {
        "id": "ut_03", "category": "unseen_templates",
        "parent_title": "Term 2 quiz schedule clashing with live projects",
        "text": "Students should ask the class representative to consolidate all quiz dates into one calendar and request a 48-hour buffer.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["academic_workload"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "ut_04", "category": "unseen_templates",
        "parent_title": "SBI Scholar Loan fee installment delay",
        "text": "Attach your bank sanction letter and mail the accounts desk directly; they grant a 15-day extension without late penalty.",
        "sentiment": "positive", "emotions": ["hope", "gratitude"], "issues": ["financial_concerns"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "ut_05", "category": "unseen_templates",
        "parent_title": "Is the case pedagogy too theoretical in Term 1?",
        "text": "Not entirely true—our finance and operations electives brought industry practitioners who walked through live balance sheets.",
        "sentiment": "neutral", "emotions": ["confidence"], "issues": ["curriculum_teaching"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "ut_06", "category": "unseen_templates",
        "parent_title": "Campus Wi-Fi outage near library reading room",
        "text": "Completely second this; library bandwidth collapses right before midnight submission windows.",
        "sentiment": "negative", "emotions": ["frustration", "stress"], "issues": ["campus_infrastructure"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "ut_07", "category": "unseen_templates",
        "parent_title": "Club induction overlapping with mid-term prep",
        "text": "Do the consulting and analytics clubs publish a semester activity calendar before induction interviews?",
        "sentiment": "neutral", "emotions": ["curiosity"], "issues": ["extracurriculars"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "ut_08", "category": "unseen_templates",
        "parent_title": "Last-minute circular about attendance policy",
        "text": "Sending policy revisions twelve hours before deadlines creates avoidable panic across the entire batch.",
        "sentiment": "negative", "emotions": ["stress", "frustration"], "issues": ["administrative_communication"],
        "response": "opinion", "solution": "not_solution",
    },

    # --- 2. paraphrased_comments (8 items) ---
    {
        "id": "pc_01", "category": "paraphrased_comments",
        "parent_title": "Summer internship preparation timeline",
        "text": "I share the exact same impression about summer interview prep needing structured mock sessions early.",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["placements"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "pc_02", "category": "paraphrased_comments",
        "parent_title": "Fee refund and hostel deposit clearance",
        "text": "When I cleared my hostel caution deposit last term, keeping stamped receipts sped up the finance approval.",
        "sentiment": "positive", "emotions": ["confidence", "gratitude"], "issues": ["financial_concerns"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "pc_03", "category": "paraphrased_comments",
        "parent_title": "Overlapping elective case submissions",
        "text": "Consider drafting a polite joint email to the program office proposing staggered submission deadlines.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["academic_workload"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "pc_04", "category": "paraphrased_comments",
        "parent_title": "Hostel mess hygiene complaints",
        "text": "The cleanest remedy is rotating student mess auditors weekly and posting inspection photos on the notice board.",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["hostel_facilities"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "pc_05", "category": "paraphrased_comments",
        "parent_title": "Are visiting faculty sessions worth the tuition?",
        "text": "That overlooks how several visiting professors bring current corporate datasets that permanent syllabi lack.",
        "sentiment": "neutral", "emotions": ["confidence"], "issues": ["curriculum_teaching"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "pc_06", "category": "paraphrased_comments",
        "parent_title": "Shuttle bus frequency to Sambalpur station",
        "text": "The evening shuttle gap leaves students stranded after late lab hours, which feels exhausting.",
        "sentiment": "negative", "emotions": ["stress", "frustration"], "issues": ["campus_infrastructure"],
        "response": "opinion", "solution": "not_solution",
    },

    # --- 3. spelling_errors (8 items) ---
    {
        "id": "se_01", "category": "spelling_errors",
        "parent_title": "Placement shortlist updates",
        "text": "plcmnt comitee shuld share shrtlist crteria erlier, evryone is stresd out waiting",
        "sentiment": "negative", "emotions": ["stress", "frustration"], "issues": ["placements"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "se_02", "category": "spelling_errors",
        "parent_title": "Hostel room maintenance",
        "text": "try emaling the hostl wardn with picturs of the leakge, that workd quick for us",
        "sentiment": "positive", "emotions": ["hope", "gratitude"], "issues": ["hostel_facilities"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "se_03", "category": "spelling_errors",
        "parent_title": "Tuition fees and education loan",
        "text": "u can aply for the merit scholrshp portal befor 15th to cut down tution burdn",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["financial_concerns"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "se_04", "category": "spelling_errors",
        "parent_title": "Term 1 assignment load",
        "text": "totlly agre bro, 3 case submisns in one wknd is way too much worklod",
        "sentiment": "negative", "emotions": ["stress"], "issues": ["academic_workload"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "se_05", "category": "spelling_errors",
        "parent_title": "Campus wifi and lab access",
        "text": "actully wifi in the new acdemic blck has been much bettr since last mnth",
        "sentiment": "positive", "emotions": ["confidence"], "issues": ["campus_infrastructure"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "se_06", "category": "spelling_errors",
        "parent_title": "Admin circular clarity",
        "text": "i facd the sam confusn with regstratn forms last semstr until offce replyd",
        "sentiment": "neutral", "emotions": ["frustration"], "issues": ["administrative_communication"],
        "response": "personal_experience", "solution": "not_solution",
    },

    # --- 4. informal_language (8 items) ---
    {
        "id": "il_01", "category": "informal_language",
        "parent_title": "Mess dinner menu review",
        "text": "ngl mess paneer was actually fire tonight, huge W compared to last month tbh!!",
        "sentiment": "positive", "emotions": ["gratitude"], "issues": ["hostel_facilities"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "il_02", "category": "informal_language",
        "parent_title": "SIP resume review grind",
        "text": "bro just ping two seniors on linkedin for CV edits, legit saved my shortlist fr",
        "sentiment": "positive", "emotions": ["gratitude", "confidence"], "issues": ["placements"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "il_03", "category": "informal_language",
        "parent_title": "Quiz + case comp clash",
        "text": "lmao lowkey dying with back to back quizzes and club decks this week",
        "sentiment": "negative", "emotions": ["stress", "frustration"], "issues": ["academic_workload", "extracurriculars"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "il_04", "category": "informal_language",
        "parent_title": "Is 22L loan worth it for baby IIM?",
        "text": "nah fam gotta disagree there, if you already earn 16 LPA taking a 22L loan makes zero sense imo",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["financial_concerns"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "il_05", "category": "informal_language",
        "parent_title": "Library AC and power sockets",
        "text": "facts bro!! half the sockets on floor 2 don't even work",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["campus_infrastructure"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "il_06", "category": "informal_language",
        "parent_title": "How to get bonafide certificate fast?",
        "text": "drop by academic office right at 10am with your ID card, they stamp it same day ez",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["administrative_communication"],
        "response": "solution", "solution": "solution",
    },

    # --- 5. short_comments (8 items) ---
    {
        "id": "sc_01", "category": "short_comments",
        "parent_title": "Placement committee transparency",
        "text": "Same here, fully agree.",
        "sentiment": "neutral", "emotions": [], "issues": [],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "sc_02", "category": "short_comments",
        "parent_title": "Are fees too high?",
        "text": "Hard disagree on that.",
        "sentiment": "negative", "emotions": ["frustration"], "issues": [],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "sc_03", "category": "short_comments",
        "parent_title": "Hostel Wi-Fi keeps dropping",
        "text": "Raise an IT helpdesk ticket.",
        "sentiment": "neutral", "emotions": [], "issues": ["campus_infrastructure"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "sc_04", "category": "short_comments",
        "parent_title": "Elective bidding tips",
        "text": "Ask second-year seniors first.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["curriculum_teaching"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "sc_05", "category": "short_comments",
        "parent_title": "Mess food quality",
        "text": "Mess food is awful.",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["hostel_facilities"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "sc_06", "category": "short_comments",
        "parent_title": "Scholarship disbursement timeline",
        "text": "Faced this last term.",
        "sentiment": "neutral", "emotions": [], "issues": [],
        "response": "personal_experience", "solution": "not_solution",
    },

    # --- 6. ambiguous_responses (6 items) ---
    {
        "id": "ar_01", "category": "ambiguous_responses",
        "parent_title": "Should I take IIM Sambalpur or repeat CAT?",
        "text": "Maybe check alumni LinkedIn profiles first before locking the seat, though honestly everyone's risk appetite differs.",
        "sentiment": "neutral", "emotions": ["curiosity"], "issues": ["placements"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "ar_02", "category": "ambiguous_responses",
        "parent_title": "Term 1 workload burnout",
        "text": "It was brutal for me initially, yet looking back the pace actually prepared me well for summer interviews.",
        "sentiment": "positive", "emotions": ["confidence", "stress"], "issues": ["academic_workload"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "ar_03", "category": "ambiguous_responses",
        "parent_title": "Admin response speed on transcripts",
        "text": "I wouldn't say admin is hopeless, just understaffed during convocation week; emailing twice usually works.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["administrative_communication"],
        "response": "counterargument", "solution": "solution",
    },
    {
        "id": "ar_04", "category": "ambiguous_responses",
        "parent_title": "Club budgets and fest sponsorship",
        "text": "Publishing a simple expense sheet after every fest would probably reduce half the rumors.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["extracurriculars"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "ar_05", "category": "ambiguous_responses",
        "parent_title": "Campus infrastructure construction noise",
        "text": "Sure the construction dust is annoying right now, still glad the new auditorium is finally coming up.",
        "sentiment": "positive", "emotions": ["frustration", "hope"], "issues": ["campus_infrastructure"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "ar_06", "category": "ambiguous_responses",
        "parent_title": "Education loan moratorium interest",
        "text": "True, interest compounds fast, though paying simple interest monthly during the two years cuts the final EMI.",
        "sentiment": "neutral", "emotions": ["confidence"], "issues": ["financial_concerns"],
        "response": "solution", "solution": "solution",
    },

    # --- 7. parent_post_context (8 elliptic comments whose issue lives ONLY in parent_title) ---
    {
        "id": "ppc_01", "category": "parent_post_context",
        "parent_title": "Hostel Block C geyser and water supply breakdown",
        "text": "Happened on our floor all week too; keep escalating until they send a technician.",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["hostel_facilities"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "ppc_02", "category": "parent_post_context",
        "parent_title": "Summer placements shortlist criteria and recruiter mix",
        "text": "Same doubt here, nobody has shared clear numbers with the incoming batch yet.",
        "sentiment": "negative", "emotions": ["curiosity", "stress"], "issues": ["placements"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "ppc_03", "category": "parent_post_context",
        "parent_title": "Tuition fee installment deadline and bank loan disbursement",
        "text": "Submit the bank acknowledgment slip before Friday and they will waive the penalty.",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["financial_concerns"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "ppc_04", "category": "parent_post_context",
        "parent_title": "Term 2 exam timetable and case study overload",
        "text": "Honestly the pace feels unsustainable when three deadlines land on the exact same night.",
        "sentiment": "negative", "emotions": ["stress", "frustration"], "issues": ["academic_workload"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "ppc_05", "category": "parent_post_context",
        "parent_title": "Elective curriculum quality and visiting faculty feedback",
        "text": "Actually the second-half sessions were super practical once the practitioner joined.",
        "sentiment": "positive", "emotions": ["gratitude", "confidence"], "issues": ["curriculum_teaching"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "ppc_06", "category": "parent_post_context",
        "parent_title": "Campus Wi-Fi routers and library lab computers",
        "text": "Try raising a batch-level complaint on the portal so IT prioritizes our wing.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["campus_infrastructure"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "ppc_07", "category": "parent_post_context",
        "parent_title": "Cultural club induction and student committee workload",
        "text": "I joined two committees last year and had to drop one because late-night meetings clashed.",
        "sentiment": "neutral", "emotions": ["stress"], "issues": ["extracurriculars"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "ppc_08", "category": "parent_post_context",
        "parent_title": "Administrative office circular delays for bonafide letters",
        "text": "Better to walk in during morning hours with a printed copy instead of waiting on email.",
        "sentiment": "neutral", "emotions": ["hope"], "issues": ["administrative_communication"],
        "response": "advice", "solution": "solution",
    },

    # --- 8. code_mixed_language (8 unseen Hinglish + Devanagari + campus multilingual) ---
    {
        "id": "cm_01", "category": "code_mixed_language",
        "parent_title": "Placement committee updates in r/MBAIndia",
        "text": "Bhai placement committee se clear timeline maango, warna sab juniors confused rahenge.",
        "sentiment": "negative", "emotions": ["frustration", "stress"], "issues": ["placements"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "cm_02", "category": "code_mixed_language",
        "parent_title": "Hostel mess food complaint",
        "text": "Hostel mess ka khana bahut kharab ho gaya hai yaar, warden ko photos ke saath mail bhejna chahiye.",
        "sentiment": "negative", "emotions": ["frustration"], "issues": ["hostel_facilities"],
        "response": "advice", "solution": "solution",
    },
    {
        "id": "cm_03", "category": "code_mixed_language",
        "parent_title": "Education loan aur fee structure",
        "text": "Fee structure थोड़ा heavy hai par scholarship portal pe apply karne se relief mil sakta hai.",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["financial_concerns"],
        "response": "solution", "solution": "solution",
    },
    {
        "id": "cm_04", "category": "code_mixed_language",
        "parent_title": "Term 1 workload experience",
        "text": "Bilkul sahi bola bhai, pehle term mein sleep schedule poora bigad jata hai.",
        "sentiment": "negative", "emotions": ["stress"], "issues": ["academic_workload"],
        "response": "agreement", "solution": "not_solution",
    },
    {
        "id": "cm_05", "category": "code_mixed_language",
        "parent_title": "Campus Wi-Fi and classroom projector issues",
        "text": "नहीं यार, नए academic block में Wi-Fi speed पिछले महीने से काफी improve हुई है।",
        "sentiment": "positive", "emotions": ["confidence", "gratitude"], "issues": ["campus_infrastructure"],
        "response": "counterargument", "solution": "not_solution",
    },
    {
        "id": "cm_06", "category": "code_mixed_language",
        "parent_title": "Admin circular confusion",
        "text": "Mera bhi same experience tha last sem mein, admin office jaake personally clarify karwana pada.",
        "sentiment": "neutral", "emotions": ["frustration"], "issues": ["administrative_communication"],
        "response": "personal_experience", "solution": "not_solution",
    },
    {
        "id": "cm_07", "category": "code_mixed_language",
        "parent_title": "Faculty teaching quality discussion",
        "text": "Analytics aur strategy ke profs kaafi supportive hain, class discussion genuinely helpful lagta hai.",
        "sentiment": "positive", "emotions": ["gratitude", "confidence"], "issues": ["curriculum_teaching"],
        "response": "opinion", "solution": "not_solution",
    },
    {
        "id": "cm_08", "category": "code_mixed_language",
        "parent_title": "Club events aur case competitions",
        "text": "Case club join karne ka best tareeka hai ki orientation mein hi senior mentors se baat kar lo.",
        "sentiment": "positive", "emotions": ["hope"], "issues": ["extracurriculars"],
        "response": "solution", "solution": "solution",
    },
]


def load_post_titles() -> Dict[str, str]:
    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    return {p["reddit_id"]: p["title"] for p in corpus.get("posts", [])}


def score_multilabel(
    gold: Sequence[Sequence[str]],
    preds: Sequence[Sequence[str]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    l_count = len(labels)
    exact = 0
    hamming_errors = 0
    total_tp = total_fp = total_fn = 0
    per_label: Dict[str, Dict[str, Any]] = {}
    f1s: List[float] = []

    gold_sets = [set(g) for g in gold]
    pred_sets = [set(p) for p in preds]

    for g_set, p_set in zip(gold_sets, pred_sets):
        if g_set == p_set:
            exact += 1
        for lbl in labels:
            if (lbl in g_set) != (lbl in p_set):
                hamming_errors += 1

    for lbl in labels:
        tp = sum((lbl in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fp = sum((lbl not in g) and (lbl in p) for g, p in zip(gold_sets, pred_sets))
        fn = sum((lbl in g) and (lbl not in p) for g, p in zip(gold_sets, pred_sets))
        support = sum(lbl in g for g in gold_sets)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_label[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_denom) if micro_denom > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {
        "n_samples": n,
        "multilabel": True,
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "subset_accuracy": round(exact / n, 6) if n > 0 else 0.0,
        "hamming_loss": round(hamming_errors / (n * l_count), 6) if (n * l_count) > 0 else 0.0,
        "per_label": per_label,
    }


def score_singlelabel(
    gold: Sequence[str],
    preds: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, preds))
    index = {lbl: i for i, lbl in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for g, p in zip(gold, preds):
        if g in index and p in index:
            cm[index[g]][index[p]] += 1

    f1s: List[float] = []
    per_class: Dict[str, Dict[str, Any]] = {}
    for lbl in labels:
        tp = sum((g == lbl) and (p == lbl) for g, p in zip(gold, preds))
        fp = sum((g != lbl) and (p == lbl) for g, p in zip(gold, preds))
        fn = sum((g == lbl) and (p != lbl) for g, p in zip(gold, preds))
        support = sum(g == lbl for g in gold)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if support > 0 else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_class[lbl] = {
            "precision": round(prec, 6) if prec is not None else None,
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }
    acc = round(correct / n, 6) if n > 0 else 0.0
    return {
        "n_samples": n,
        "multilabel": False,
        "accuracy": acc,
        "micro_f1": acc,
        "macro_f1": round(float(np.mean(f1s)), 6) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": cm,
    }


def _parse_targets(task_name: str, rows: Sequence[Any]) -> List[Any]:
    if task_name in {"emotion", "issue"}:
        return [
            sorted(
                part.strip().lower()
                for part in str(r.label or "").replace(",", "|").split("|")
                if part.strip()
            )
            for r in rows
        ]
    return [str(r.label).strip().lower() for r in rows]


def _probe_targets(task_name: str, items: Sequence[Dict[str, Any]]) -> List[Any]:
    key = {"sentiment": "sentiment", "emotion": "emotions", "issue": "issues", "response": "response", "solution": "solution"}[task_name]
    return [item[key] for item in items]


def _with_text(rows: Sequence[Any], texts: Sequence[str]) -> List[Any]:
    from copy import copy

    out = []
    for r, t in zip(rows, texts):
        c = copy(r)
        c.text = t
        out.append(c)
    return out


def _build_classical_docs(task_name: str, texts: Sequence[str]) -> List[Any]:
    from app.services.nlp.issue_identification.base import IssueDocument
    from app.services.nlp.response_classification.base import ResponseDocument
    from app.services.nlp.sentiment_emotion.base import SentimentEmotionDocument
    from app.services.nlp.solution_detection.base import SolutionDocument

    if task_name == "issue":
        return [IssueDocument(document_id=f"p{i}", text=t) for i, t in enumerate(texts)]
    if task_name in {"sentiment", "emotion"}:
        return [SentimentEmotionDocument(document_id=f"p{i}", text=t) for i, t in enumerate(texts)]
    if task_name == "response":
        return [ResponseDocument(document_id=f"p{i}", text=t) for i, t in enumerate(texts)]
    return [SolutionDocument(document_id=f"p{i}", text=t) for i, t in enumerate(texts)]


def _predict_classical(model: Any, task_name: str, texts: Sequence[str]) -> Tuple[List[Any], float]:
    docs = _build_classical_docs(task_name, texts)
    t0 = time.perf_counter()
    res = model.predict(docs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if task_name == "issue":
        return [p.predicted_issue_labels for p in res.predictions], elapsed_ms
    if task_name == "emotion":
        return [p.emotions for p in res.predictions], elapsed_ms
    if task_name == "sentiment":
        return [p.sentiment or "" for p in res.predictions], elapsed_ms
    if task_name == "response":
        return [p.response_type or "" for p in res.predictions], elapsed_ms
    return ["solution" if p.is_solution else "not_solution" for p in res.predictions], elapsed_ms


def _artifact_size_kb_joblib(model: Any) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.joblib"
        joblib.dump(model, p)
        return round(p.stat().st_size / 1024.0, 2)


def evaluate_probes_by_category(
    task_name: str,
    labels: Sequence[str],
    predict_fn,
    *,
    include_parent_title: bool = False,
) -> Dict[str, Any]:
    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    all_texts = [
        f"{item['parent_title']} | {item['text']}" if include_parent_title else item["text"]
        for item in ROBUSTNESS_PROBES
    ]
    all_gold = _probe_targets(task_name, ROBUSTNESS_PROBES)
    all_preds, total_ms = predict_fn(all_texts)

    by_cat: Dict[str, Any] = {}
    categories = list(dict.fromkeys(item["category"] for item in ROBUSTNESS_PROBES))
    for cat in categories:
        idxs = [i for i, item in enumerate(ROBUSTNESS_PROBES) if item["category"] == cat]
        cat_gold = [all_gold[i] for i in idxs]
        cat_preds = [all_preds[i] for i in idxs]
        cat_score = scorer(cat_gold, cat_preds, labels)
        by_cat[cat] = {
            "n": len(idxs),
            "macro_f1": cat_score["macro_f1"],
            "micro_f1": cat_score["micro_f1"],
            **(
                {"subset_accuracy": cat_score["subset_accuracy"], "hamming_loss": cat_score["hamming_loss"]}
                if task_name in {"emotion", "issue"}
                else {"accuracy": cat_score["accuracy"]}
            ),
        }

    overall = scorer(all_gold, all_preds, labels)
    return {
        "n_probes": len(ROBUSTNESS_PROBES),
        "include_parent_title": include_parent_title,
        "inference_ms": round(total_ms, 2),
        "overall_macro_f1": overall["macro_f1"],
        "overall_micro_f1": overall["micro_f1"],
        **(
            {"overall_subset_accuracy": overall["subset_accuracy"], "overall_hamming_loss": overall["hamming_loss"]}
            if task_name in {"emotion", "issue"}
            else {"overall_accuracy": overall["accuracy"]}
        ),
        "by_category": by_cat,
    }


def benchmark_classical(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    *,
    upgraded: bool,
    with_parent_title: bool = False,
    post_titles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    task = TaskType(task_name)
    if with_parent_title and post_titles is not None:
        train_rows = _with_text(train_rows, [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in train_rows])
        val_rows = _with_text(val_rows, [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in val_rows])
        test_rows = _with_text(test_rows, [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in test_rows])

    if task_name == "issue":
        cfg = IssueModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else IssueModelConfig()
        model = TfidfLogisticIssueClassifier(config=cfg)
    elif task_name == "emotion":
        cfg = EmotionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else EmotionModelConfig()
        model = TfidfLogisticEmotionClassifier(config=cfg)
    elif task_name == "sentiment":
        cfg = SentimentModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else SentimentModelConfig()
        model = TfidfLogisticSentimentClassifier(config=cfg)
    elif task_name == "response":
        cfg = ResponseModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else ResponseModelConfig()
        model = TfidfLogisticResponseClassifier(config=cfg)
    else:
        cfg = SolutionModelConfig(use_char_ngrams=True, class_weight="balanced", C=4.0) if upgraded else SolutionModelConfig()
        model = TfidfLogisticSolutionClassifier(config=cfg)

    train_ann = records_to_annotations(train_rows, task)
    val_ann = records_to_annotations(val_rows, task)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model.train(train_ann)
    if upgraded and hasattr(model, "calibrate_thresholds"):
        model.calibrate_thresholds(val_ann)
    train_s = round(time.perf_counter() - t0, 4)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    test_texts = [r.text for r in test_rows]
    test_gold = _parse_targets(task_name, test_rows)
    test_preds, infer_ms = _predict_classical(model, task_name, test_texts)
    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    heldout = scorer(test_gold, test_preds, model.labels)

    probe_eval = evaluate_probes_by_category(
        task_name,
        model.labels,
        lambda txts: _predict_classical(model, task_name, txts),
        include_parent_title=with_parent_title,
    )

    return {
        "resources": {
            "train_seconds": train_s,
            "test_inference_total_ms": round(infer_ms, 2),
            "test_inference_ms_per_comment": round(infer_ms / max(len(test_texts), 1), 4),
            "peak_cpu_tracemalloc_mb": round(peak_bytes / (1024 * 1024), 2),
            "peak_gpu_vram_mb": 0.0,
            "artifact_size_kb": _artifact_size_kb_joblib(model),
        },
        "heldout_test": heldout,
        "robustness_probes": probe_eval,
    }


def benchmark_muril(
    task_name: str,
    train_rows: Sequence[Any],
    val_rows: Sequence[Any],
    test_rows: Sequence[Any],
    *,
    epochs: int,
    with_parent_title: bool = False,
    post_titles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    def _texts(rows_seq):
        if with_parent_title and post_titles is not None:
            return [f"{post_titles.get(r.post_id or '', '')} | {r.text}" for r in rows_seq]
        return [r.text for r in rows_seq]

    tr_t, va_t, te_t = _texts(train_rows), _texts(val_rows), _texts(test_rows)
    tr_y = _parse_targets(task_name, train_rows)
    va_y = _parse_targets(task_name, val_rows)
    te_y = _parse_targets(task_name, test_rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    t0 = time.perf_counter()
    cfg = MuRILTaskConfig(
        task=task_name,
        epochs=epochs,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        max_length=160,
        random_state=SEED,
        verbose=True,
    )
    clf = MuRILTaskClassifier(cfg)
    fit_summary = clf.fit(tr_t, tr_y, va_t, va_y)
    train_s = round(time.perf_counter() - t0, 3)
    _, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ti0 = time.perf_counter()
    test_preds, _ = clf.predict_labels(te_t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - ti0) * 1000.0

    scorer = score_multilabel if task_name in {"emotion", "issue"} else score_singlelabel
    heldout = scorer(te_y, test_preds, clf.labels)

    def _pred_fn(txts: Sequence[str]):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s0 = time.perf_counter()
        p, _ = clf.predict_labels(txts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return p, (time.perf_counter() - s0) * 1000.0

    probe_eval = evaluate_probes_by_category(
        task_name, clf.labels, _pred_fn, include_parent_title=with_parent_title
    )

    # Approximate serialized checkpoint size (fp32 weights ~906 MB for MuRIL base)
    param_bytes = sum(p.numel() * p.element_size() for p in clf._model.parameters())
    artifact_kb = round(param_bytes / 1024.0, 1)

    # Free VRAM between tasks
    del clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_epoch": fit_summary["best_epoch"],
        "best_val_macro_f1": fit_summary["best_val_macro_f1"],
        "resources": {
            "device": fit_summary["device"],
            "amp_dtype": fit_summary["amp_dtype"],
            "train_seconds": train_s,
            "test_inference_total_ms": round(infer_ms, 2),
            "test_inference_ms_per_comment": round(infer_ms / max(len(te_t), 1), 4),
            "peak_cpu_tracemalloc_mb": round(peak_cpu / (1024 * 1024), 2),
            "peak_gpu_vram_mb": peak_vram_mb,
            "artifact_size_kb": artifact_kb,
        },
        "heldout_test": heldout,
        "robustness_probes": probe_eval,
    }


def main(epochs: int = 6, run_muril: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    post_titles = load_post_titles()

    report: Dict[str, Any] = {
        "disclaimer": "SYNTHETIC BENCHMARK AND STRESS-TEST SUITE ONLY. Not real Reddit human-annotated evaluation.",
        "seed": SEED,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "probe_counts_by_category": dict(Counter(item["category"] for item in ROBUSTNESS_PROBES)),
        "total_probes": len(ROBUSTNESS_PROBES),
        "tasks": {},
    }

    for task_name in TASKS:
        print(f"\n==================== ROBUSTNESS BENCHMARK: {task_name.upper()} ====================")
        records = load_annotation_dataset(TaskType(task_name), str(DATA_DIR / f"{task_name}.csv"))
        spec = SplitSpec(
            train_size=0.70, validation_size=0.15, test_size=0.15,
            random_state=SEED, stratify=True,
        )
        groups = extract_groups(records, "post_id")
        split = split_records(records, labels=[r.label for r in records], spec=spec, groups=groups)
        train_rows, val_rows, test_rows = apply_split(records, split)
        leak = check_leakage(train_rows, val_rows, test_rows)
        assert not leak.get("leakage_detected"), f"Leakage detected in {task_name}"

        v1 = benchmark_classical(task_name, train_rows, val_rows, test_rows, upgraded=False)
        v2 = benchmark_classical(task_name, train_rows, val_rows, test_rows, upgraded=True)

        task_out: Dict[str, Any] = {
            "v1_tfidf_baseline": v1,
            "v2_word_char_calibrated": v2,
        }
        print(
            f"  [V1]    heldout={v1['heldout_test']['macro_f1']:.4f} | "
            f"8-cat probes={v1['robustness_probes']['overall_macro_f1']:.4f} | "
            f"infer={v1['resources']['test_inference_ms_per_comment']:.3f} ms/doc"
        )
        print(
            f"  [V2]    heldout={v2['heldout_test']['macro_f1']:.4f} | "
            f"8-cat probes={v2['robustness_probes']['overall_macro_f1']:.4f} | "
            f"infer={v2['resources']['test_inference_ms_per_comment']:.3f} ms/doc"
        )

        if task_name == "issue":
            v2_ctx = benchmark_classical(
                task_name, train_rows, val_rows, test_rows,
                upgraded=True, with_parent_title=True, post_titles=post_titles,
            )
            task_out["v2_word_char_with_parent_title"] = v2_ctx
            print(
                f"  [V2+PT] heldout={v2_ctx['heldout_test']['macro_f1']:.4f} | "
                f"8-cat probes={v2_ctx['robustness_probes']['overall_macro_f1']:.4f} | "
                f"parent_ctx_cat={v2_ctx['robustness_probes']['by_category']['parent_post_context']['macro_f1']:.4f}"
            )

        if run_muril:
            mu = benchmark_muril(task_name, train_rows, val_rows, test_rows, epochs=epochs)
            task_out["muril_v2_gpu"] = mu
            print(
                f"  [MuRIL] heldout={mu['heldout_test']['macro_f1']:.4f} | "
                f"8-cat probes={mu['robustness_probes']['overall_macro_f1']:.4f} | "
                f"best_ep={mu['best_epoch']} | VRAM={mu['resources']['peak_gpu_vram_mb']} MB | "
                f"infer={mu['resources']['test_inference_ms_per_comment']:.3f} ms/doc"
            )
            if task_name == "issue":
                mu_ctx = benchmark_muril(
                    task_name, train_rows, val_rows, test_rows,
                    epochs=epochs, with_parent_title=True, post_titles=post_titles,
                )
                task_out["muril_v2_gpu_with_parent_title"] = mu_ctx
                print(
                    f"  [Mu+PT] heldout={mu_ctx['heldout_test']['macro_f1']:.4f} | "
                    f"8-cat probes={mu_ctx['robustness_probes']['overall_macro_f1']:.4f} | "
                    f"parent_ctx_cat={mu_ctx['robustness_probes']['by_category']['parent_post_context']['macro_f1']:.4f}"
                )

        report["tasks"][task_name] = task_out

    out_path = OUT_DIR / "robustness_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved robustness & resource benchmark to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--no-muril", action="store_true")
    args = parser.parse_args()
    main(epochs=args.epochs, run_muril=not args.no_muril)
