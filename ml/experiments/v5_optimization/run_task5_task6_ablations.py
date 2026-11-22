"""Task 5 (Existing-LID Limitation Study) & Task 6 (Preprocessing Ablation) Runner.

Strictly inductive & zero test leakage:
- Task 5 evaluates `langdetect` vs. `IndianRedditLID` across all 3,557 anonymized
  live corpus comments + 12 canonical multilingual/code-mixed probes.
- Task 6 evaluates 5 preprocessing strategies strictly on V1 5-fold post_id-grouped
  OOF CV (`trainval n = 367`), NEVER touching the frozen 92-comment test set for
  decisions. Tests all 5 tasks with Word+Char TF-IDF Logistic Regression and tests
  Sentiment & Emotion with strict-inductive `MuRIL-DAPT-WWM` 5-fold CV.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.nlp.indian_reddit_lid import detect_indian_reddit_language
from app.services.nlp.preprocessing.emoji_normalizer import (
    _EMOJI_RE,
    normalize_emojis,
    normalize_emojis_semantic_affect,
    strip_emojis,
)
from app.services.nlp.preprocessing.slang_normalizer import normalize_slang
from ml.experiments.v5_optimization.common import (
    CANONICAL_LABELS,
    CKPT_ROOT,
    CORPUS_PATH,
    EXP_ROOT,
    MULTILABEL_TASKS,
    SEED,
    TrainSpec,
    apply_logits_calibration,
    encode_task_targets,
    evaluate_predictions,
    load_v1_records_and_frozen_splits,
    make_group_cv_folds,
    save_json,
    train_and_predict_logits,
)

DetectorFactory.seed = 42

OUT_PATH = EXP_ROOT / "task5_task6_ablations.json"
DAPT_STRICT_WWM_DIR = CKPT_ROOT / "muril_dapt_strict_inductive_wwm_8ep"

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_STEM_SUFFIXES = (
    "ational", "tional", "ization", "fulness", "ousness", "iveness",
    "ement", "ment", "ness", "able", "ible", "ance", "ence", "ally",
    "ing", "ied", "ies", "ed", "er", "es", "ly", "s",
)


def _simple_stem(word: str) -> str:
    if len(word) <= 4 or not word.isascii():
        return word
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess_raw(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def preprocess_conventional_nlp(text: str) -> str:
    """Conventional pipeline: lowercase + remove emojis/punctuation + stopwords + stemming."""
    t = strip_emojis(str(text or "")).lower()
    t = _PUNCT_RE.sub(" ", t)
    toks = [
        _simple_stem(w)
        for w in t.split()
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    ]
    return " ".join(toks) if toks else "empty"


def preprocess_emoji_removed(text: str) -> str:
    return _SPACE_RE.sub(" ", strip_emojis(str(text or ""))).strip()


def preprocess_emoji_token_slang(text: str) -> str:
    return _SPACE_RE.sub(" ", normalize_slang(normalize_emojis(str(text or "")))).strip()


def preprocess_emoji_affect_slang(text: str) -> str:
    return _SPACE_RE.sub(" ", normalize_slang(normalize_emojis_semantic_affect(str(text or "")))).strip()


STRATEGIES: Dict[str, Callable[[str], str]] = {
    "raw": preprocess_raw,
    "conventional_nlp": preprocess_conventional_nlp,
    "emoji_removed": preprocess_emoji_removed,
    "emoji_token_slang": preprocess_emoji_token_slang,
    "emoji_affect_slang": preprocess_emoji_affect_slang,
}


def run_task5_lid_study() -> Dict[str, Any]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    comments = [c for p in corpus["posts"] for c in p["comments"]]

    valid_indian_langs = {"en", "hi", "bn", "or", "ta", "te", "kn", "ml", "gu", "pa", "mr", "ur", "ne"}
    ld_all = Counter()
    our_all = Counter()
    ld_on_indic = Counter()
    ld_bogus_total = 0
    ld_bogus_on_indic = 0
    indic_total = 0
    examples: List[Dict[str, Any]] = []

    for c in comments:
        body = str(c.get("body") or "")
        ours = detect_indian_reddit_language(body)
        our_lang = ours["language"]
        our_all[our_lang] += 1
        try:
            ld_lang = detect(body) if len(body.strip()) >= 3 else "err"
        except Exception:
            ld_lang = "err"
        ld_all[ld_lang] += 1
        is_bogus = ld_lang not in valid_indian_langs and ld_lang != "err"
        if is_bogus:
            ld_bogus_total += 1
        if our_lang in {"hi-en", "hi", "bn-en", "bn", "or-en", "or"}:
            indic_total += 1
            ld_on_indic[ld_lang] += 1
            if is_bogus:
                ld_bogus_on_indic += 1
                if len(examples) < 10 and 12 <= len(body) <= 180:
                    examples.append({
                        "comment_id": c["document_id"],
                        "text": body[:160],
                        "langdetect_pred": ld_lang,
                        "indian_reddit_lid_pred": our_lang,
                        "script": ours["script"],
                        "is_code_mixed": ours["is_code_mixed"],
                        "matched_markers": ours.get("matched_markers", [])[:6],
                    })

    probes = [
        ("Bhai IIM ABC ka summer placement scene kaisa hai 😭", "hi-en", "latin"),
        ("ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro", "or-en", "mixed"),
        ("IIM e giye ki actually ROI pawa jay?", "bn-en", "latin"),
        ("Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl.", "hi-en", "latin"),
        ("IIM jaana worth it hai kya?", "hi-en", "latin"),
        ("यह कॉलेज का प्लेसमेंट कैसा है?", "hi", "devanagari"),
        ("এই কলেজের হোস্টেল কেমন?", "bn", "bengali"),
        ("ଆଜିର କ୍ଲାସ୍ ବହୁତ ଭଲ ଥିଲା", "or", "odia"),
        ("Aaji class kemiti thila bhai, bahut hectic hela", "or-en", "latin"),
        ("Kolkata theke IIM Sambalpur jabo kemon hobe placement?", "bn-en", "latin"),
        ("Try applying for the need-based financial aid scholarship.", "en", "latin"),
        ("Fr bro got rejected at 99.3%ile 🤡", "en", "latin"),
    ]
    probe_rows = []
    our_probe_correct = 0
    ld_probe_correct = 0
    for text, gold_lang, gold_script in probes:
        ours = detect_indian_reddit_language(text)
        try:
            ld_p = detect(text)
        except Exception:
            ld_p = "err"
        our_ok = (ours["language"] == gold_lang and ours["script"] == gold_script)
        ld_ok = (ld_p == gold_lang)
        our_probe_correct += int(our_ok)
        ld_probe_correct += int(ld_ok)
        probe_rows.append({
            "text": text,
            "gold_language": gold_lang,
            "gold_script": gold_script,
            "langdetect_pred": ld_p,
            "indian_reddit_lid_pred": ours["language"],
            "indian_reddit_script": ours["script"],
            "indian_reddit_code_mixed": ours["is_code_mixed"],
            "indian_reddit_correct": our_ok,
            "langdetect_correct": ld_ok,
        })

    return {
        "corpus_size": len(comments),
        "indian_reddit_lid_distribution": dict(our_all.most_common()),
        "langdetect_top_distribution": dict(ld_all.most_common(15)),
        "langdetect_bogus_non_indian_total": ld_bogus_total,
        "langdetect_bogus_non_indian_rate": round(ld_bogus_total / max(len(comments), 1), 4),
        "code_mixed_or_roman_indic_count": indic_total,
        "langdetect_on_indic_distribution": dict(ld_on_indic.most_common(12)),
        "langdetect_bogus_on_indic_count": ld_bogus_on_indic,
        "langdetect_bogus_on_indic_rate": round(ld_bogus_on_indic / max(indic_total, 1), 4),
        "langdetect_missed_as_english_on_indic": ld_on_indic.get("en", 0),
        "langdetect_failure_rate_on_indic": round(
            (indic_total - ld_on_indic.get("hi", 0) - ld_on_indic.get("bn", 0)) / max(indic_total, 1), 4
        ),
        "real_misclassification_examples": examples,
        "controlled_probe_accuracy": {
            "n_probes": len(probes),
            "indian_reddit_lid_accuracy": round(our_probe_correct / len(probes), 4),
            "langdetect_accuracy": round(ld_probe_correct / len(probes), 4),
            "probes": probe_rows,
        },
    }


def _eval_tfidf_5fold(rows: List[Dict[str, Any]], task: str, transform: Callable[[str], str]) -> Dict[str, Any]:
    labels = CANONICAL_LABELS[task]
    folds = make_group_cv_folds(rows, task=task, n_splits=5, seed=SEED)
    texts = [transform(r["text"]) for r in rows]
    has_emoji = np.array([bool(_EMOJI_RE.search(str(r["text"]))) for r in rows])
    y = encode_task_targets(rows, task)

    if task in MULTILABEL_TASKS:
        oof_probs = np.zeros((len(rows), len(labels)), dtype=float)
        for tr_idx, va_idx in folds:
            vec = FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)),
                ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=12000, sublinear_tf=True)),
            ])
            X_tr = vec.fit_transform([texts[i] for i in tr_idx])
            X_va = vec.transform([texts[i] for i in va_idx])
            clf = OneVsRestClassifier(LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42))
            clf.fit(X_tr, y[tr_idx].astype(int))
            oof_probs[va_idx] = clf.predict_proba(X_va)
        preds = (oof_probs >= 0.50).astype(int)
        macro_f1 = float(f1_score(y.astype(int), preds, average="macro", zero_division=0))
        em_f1 = float(f1_score(y[has_emoji].astype(int), preds[has_emoji], average="micro", zero_division=0)) if has_emoji.sum() else 0.0
        return {"oof_macro_f1": round(macro_f1, 4), "emoji_subset_micro_f1": round(em_f1, 4), "n_emoji_rows": int(has_emoji.sum())}
    else:
        oof_preds = np.zeros(len(rows), dtype=int)
        for tr_idx, va_idx in folds:
            vec = FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)),
                ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=12000, sublinear_tf=True)),
            ])
            X_tr = vec.fit_transform([texts[i] for i in tr_idx])
            X_va = vec.transform([texts[i] for i in va_idx])
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr, y[tr_idx])
            oof_preds[va_idx] = clf.predict(X_va)
        macro_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
        em_acc = float(np.mean(y[has_emoji] == oof_preds[has_emoji])) if has_emoji.sum() else 0.0
        return {"oof_macro_f1": round(macro_f1, 4), "emoji_subset_acc": round(em_acc, 4), "n_emoji_rows": int(has_emoji.sum())}


def _eval_muril_5fold(rows: List[Dict[str, Any]], task: str, loss_type: str, transform: Callable[[str], str]) -> Dict[str, Any]:
    adj_rows = [{**r, "text": transform(r["text"])} for r in rows]
    labels = CANONICAL_LABELS[task]
    multilabel = task in MULTILABEL_TASKS
    folds = make_group_cv_folds(adj_rows, task=task, n_splits=5, seed=SEED)
    y_true = encode_task_targets(adj_rows, task)
    oof_logits = np.zeros((len(adj_rows), len(labels)), dtype=np.float32)

    t0 = time.time()
    for fi, (tr_idx, va_idx) in enumerate(folds):
        spec = TrainSpec(
            task=task,
            backbone_id=str(DAPT_STRICT_WWM_DIR),
            context_mode="comment_only",
            loss_type=loss_type,
            epochs=5,
            batch_size=16,
            eval_batch_size=32,
            learning_rate=4e-5,
            seed=SEED + fi,
        )
        res = train_and_predict_logits(
            spec,
            [adj_rows[i] for i in tr_idx],
            [adj_rows[i] for i in va_idx],
            {},
        )
        oof_logits[va_idx] = res["val_logits"]

    probs = apply_logits_calibration(oof_logits, multilabel=multilabel, temperature=1.0)
    ev = evaluate_predictions(task, y_true, probs)
    return {
        "oof_macro_f1": round(float(ev["macro_f1"]), 4),
        "oof_ece": round(float(ev["ece"]), 4),
        "train_seconds": round(time.time() - t0, 1),
    }


def run_task6_preprocessing_ablation() -> Dict[str, Any]:
    v1 = load_v1_records_and_frozen_splits()
    trainval = v1["trainval"]
    n_emoji = sum(1 for r in trainval if _EMOJI_RE.search(str(r["text"])))

    tfidf_results: Dict[str, Any] = {}
    for strat_name, fn in STRATEGIES.items():
        per_task = {}
        for task in ("issue", "sentiment", "emotion", "response", "solution"):
            per_task[task] = _eval_tfidf_5fold(trainval, task, fn)
        mean_f1 = round(float(np.mean([v["oof_macro_f1"] for v in per_task.values()])), 4)
        tfidf_results[strat_name] = {"mean_5task_oof_f1": mean_f1, "tasks": per_task}
        print(f"[Task 6 TF-IDF 5-Fold] {strat_name:20s} -> mean 5-task OOF F1 = {mean_f1:.4f}", flush=True)

    muril_results: Dict[str, Any] = {}
    for strat_name in ("raw", "conventional_nlp", "emoji_removed", "emoji_token_slang", "emoji_affect_slang"):
        fn = STRATEGIES[strat_name]
        strat_out: Dict[str, Any] = {}
        for task, loss_type in (("sentiment", "ce_weighted"), ("emotion", "asl")):
            res = _eval_muril_5fold(trainval, task, loss_type, fn)
            strat_out[task] = res
            print(
                f"[Task 6 MuRIL-WWM 5-Fold] {strat_name:20s} | {task:9s} -> OOF F1 = {res['oof_macro_f1']:.4f} ({res['train_seconds']}s)",
                flush=True,
            )
        strat_out["mean_affect_oof_f1"] = round(
            float(np.mean([strat_out["sentiment"]["oof_macro_f1"], strat_out["emotion"]["oof_macro_f1"]])),
            4,
        )
        muril_results[strat_name] = strat_out

    return {
        "trainval_size": len(trainval),
        "trainval_emoji_rows": n_emoji,
        "protocol": "5-fold post_id-grouped OOF CV on untouched V1 trainval (367); frozen test (92) never touched",
        "tfidf_5task_ablation": tfidf_results,
        "muril_wwm_affect_ablation": muril_results,
    }


def main() -> None:
    t0 = time.time()
    print("=== Running Task 5 Existing-LID Limitation Study ===", flush=True)
    task5 = run_task5_lid_study()
    print(
        f"Task 5 done: langdetect bogus non-Indian rate={task5['langdetect_bogus_non_indian_rate']:.1%} "
        f"(overall), {task5['langdetect_bogus_on_indic_rate']:.1%} (on code-mixed/Indic), "
        f"probe accuracy: IndianRedditLID={task5['controlled_probe_accuracy']['indian_reddit_lid_accuracy']:.1%} "
        f"vs langdetect={task5['controlled_probe_accuracy']['langdetect_accuracy']:.1%}",
        flush=True,
    )
    print("\n=== Running Task 6 Preprocessing Ablation (5-Fold OOF CV) ===", flush=True)
    task6 = run_task6_preprocessing_ablation()
    payload = {
        "generated_at_epoch": round(time.time(), 1),
        "elapsed_seconds": round(time.time() - t0, 1),
        "task5_lid_study": task5,
        "task6_preprocessing_ablation": task6,
    }
    save_json(OUT_PATH, payload)
    print(f"\nSaved Task 5 & Task 6 ablation report JSON -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
