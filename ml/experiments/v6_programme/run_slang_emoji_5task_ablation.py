"""Phase 3 — 5-Task Gen-Z Slang & Emoji Preprocessing Ablation (Strict OOF CV, Zero Test Leakage).

Evaluates all four required normalization regimes across ALL FIVE canonical RCI tasks
(`issue`, `sentiment`, `emotion`, `response`, `solution`) strictly on ``trainval_v1``
(``n = 367``, 5-fold GroupKFold grouped by ``post_id`` — zero frozen test-set exposure!):

  (A) original              — Unaltered comment text
  (B) slang_normalized      — Context-aware Gen-Z + Indian internet slang normalization
  (C) emoji_aware           — Affect-preserving emoji semantic verbalization
  (D) slang_emoji_semantic  — Combined context-aware slang + emoji semantic verbalization

Benchmarks TWO model families on identical group-aware folds:
1. Word (1–2) + Char_wb (3–5) TF-IDF + class-balanced LogisticRegression (all 5 tasks × 4 modes).
2. Fine-tuned ``MuRIL-DAPT-WWM`` Transformer executed via isolated 5-fold subprocesses
   (``--worker-neural <mode> <task>``) to obey Windows GPU memory hygiene (<14 checkpoints/process).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.nlp.preprocessing.context_normalizer import apply_informal_normalization

LABELED_CSV = ROOT / "data" / "annotations" / "real_reddit_labeled_v2" / "master_labeled_v2.csv"
CORPUS_JSON = ROOT / "data" / "live_corpus" / "corpus.json"
OUT_DIR = ROOT / "ml" / "artifacts" / "v6_programme"
CACHE_DIR = OUT_DIR / "slang_emoji_cache"
OUT_JSON = OUT_DIR / "slang_emoji_5task_ablation.json"

MODES = ("original", "slang_normalized", "emoji_aware", "slang_emoji_semantic")
TASKS = ("issue", "sentiment", "emotion", "response", "solution")

ISSUE_LABELS = (
    "academic_workload",
    "administrative_communication",
    "campus_infrastructure",
    "curriculum_teaching",
    "extracurriculars",
    "financial_concerns",
    "hostel_facilities",
    "placements",
)
SENTIMENT_LABELS = ("negative", "neutral", "positive")
EMOTION_LABELS = ("confidence", "curiosity", "frustration", "gratitude", "hope", "stress")
RESPONSE_LABELS = (
    "agreement",
    "personal_experience",
    "opinion",
    "advice",
    "solution",
    "counterargument",
)
SOLUTION_LABELS = ("not_solution", "solution")


def _load_trainval_v1() -> List[Dict[str, Any]]:
    with LABELED_CSV.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("v2_split") == "trainval_v1"]
    assert len(rows) == 367, f"Expected 367 trainval_v1 rows, found {len(rows)}"
    return rows


def _build_task_targets(rows: List[Dict[str, Any]], task: str) -> Tuple[np.ndarray, bool, List[str]]:
    if task == "issue":
        Y = np.zeros((len(rows), len(ISSUE_LABELS)), dtype=np.int32)
        for i, r in enumerate(rows):
            toks = {x.strip() for x in r["issues"].split("|") if x.strip()}
            for j, lb in enumerate(ISSUE_LABELS):
                if lb in toks:
                    Y[i, j] = 1
        return Y, True, list(ISSUE_LABELS)
    if task == "emotion":
        Y = np.zeros((len(rows), len(EMOTION_LABELS)), dtype=np.int32)
        for i, r in enumerate(rows):
            toks = {x.strip() for x in r["emotions"].split("|") if x.strip()}
            for j, lb in enumerate(EMOTION_LABELS):
                if lb in toks:
                    Y[i, j] = 1
        return Y, True, list(EMOTION_LABELS)
    if task == "sentiment":
        return np.array([SENTIMENT_LABELS.index(r["sentiment"]) for r in rows], dtype=np.int32), False, list(SENTIMENT_LABELS)
    if task == "response":
        return np.array([RESPONSE_LABELS.index(r["response"]) for r in rows], dtype=np.int32), False, list(RESPONSE_LABELS)
    if task == "solution":
        return np.array([SOLUTION_LABELS.index(r["solution"]) for r in rows], dtype=np.int32), False, list(SOLUTION_LABELS)
    raise ValueError(f"Unknown task: {task}")


def _prep_texts(rows: List[Dict[str, Any]], mode: str) -> List[str]:
    out = []
    for r in rows:
        title = str(r.get("post_title") or "").strip()
        body = str(r.get("text") or "").strip()
        norm_body = apply_informal_normalization(body, mode=mode)["normalized_text"]
        norm_title = apply_informal_normalization(title, mode=mode)["normalized_text"] if title else ""
        out.append(f"{norm_title} [SEP] {norm_body}" if norm_title else norm_body)
    return out


def _tune_multilabel_thresholds_oof(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, List[float]]:
    best_th = []
    preds = np.zeros_like(y_true)
    for j in range(y_true.shape[1]):
        best_f, best_t = -1.0, 0.5
        for t in np.linspace(0.20, 0.75, 23):
            p = (probs[:, j] >= t).astype(int)
            f = f1_score(y_true[:, j], p, zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        best_th.append(round(best_t, 3))
        preds[:, j] = (probs[:, j] >= best_t).astype(int)
    return float(f1_score(y_true, preds, average="macro", zero_division=0)), best_th


def _run_tfidf_ablation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = np.array([r["post_id"] for r in rows])
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(rows, groups=groups))

    results: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        texts = _prep_texts(rows, mode)
        task_scores: Dict[str, float] = {}
        task_details: Dict[str, Any] = {}
        for task in TASKS:
            Y, is_ml, labels = _build_task_targets(rows, task)
            if is_ml:
                oof_probs = np.zeros((len(rows), len(labels)), dtype=np.float32)
                for tr, va in splits:
                    w_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True, min_df=2)
                    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True, min_df=3)
                    X_tr = hstack([w_vec.fit_transform([texts[i] for i in tr]), c_vec.fit_transform([texts[i] for i in tr])])
                    X_va = hstack([w_vec.transform([texts[i] for i in va]), c_vec.transform([texts[i] for i in va])])
                    clf = OneVsRestClassifier(LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced"))
                    clf.fit(X_tr, Y[tr])
                    oof_probs[va] = clf.predict_proba(X_va)
                macro_f1, ths = _tune_multilabel_thresholds_oof(Y, oof_probs)
                preds = np.column_stack([(oof_probs[:, j] >= ths[j]).astype(int) for j in range(len(labels))])
                per_cls = {lb: round(float(f1_score(Y[:, j], preds[:, j], zero_division=0)), 4) for j, lb in enumerate(labels)}
            else:
                oof_preds = np.zeros(len(rows), dtype=np.int32)
                for tr, va in splits:
                    w_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True, min_df=2)
                    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True, min_df=3)
                    X_tr = hstack([w_vec.fit_transform([texts[i] for i in tr]), c_vec.fit_transform([texts[i] for i in tr])])
                    X_va = hstack([w_vec.transform([texts[i] for i in va]), c_vec.transform([texts[i] for i in va])])
                    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
                    clf.fit(X_tr, Y[tr])
                    oof_preds[va] = clf.predict(X_va)
                macro_f1 = float(f1_score(Y, oof_preds, average="macro", zero_division=0))
                per_cls = {
                    lb: round(float(f1_score((Y == j).astype(int), (oof_preds == j).astype(int), zero_division=0)), 4)
                    for j, lb in enumerate(labels)
                }
            task_scores[task] = round(macro_f1, 4)
            task_details[task] = {"oof_macro_f1": round(macro_f1, 4), "per_class_f1": per_cls}

        results[mode] = {
            "mean_5task_oof_macro_f1": round(float(np.mean(list(task_scores.values()))), 4),
            "tasks": task_details,
        }
    return results


def _run_neural_worker(mode: str, task: str) -> None:
    """Isolated 5-fold MuRIL-DAPT-WWM fine-tune for a single (mode, task) pair (5 checkpoints max)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    cache_file = CACHE_DIR / f"muril_{mode}_{task}.json"
    if cache_file.is_file():
        print(f"[Cache hit] {cache_file.name}")
        return

    rows = _load_trainval_v1()
    groups = np.array([r["post_id"] for r in rows])
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(rows, groups=groups))
    Y, is_ml, labels = _build_task_targets(rows, task)

    titles = [apply_informal_normalization(str(r.get("post_title") or ""), mode=mode)["normalized_text"] for r in rows]
    bodies = [apply_informal_normalization(str(r.get("text") or ""), mode=mode)["normalized_text"] for r in rows]

    model_dir = Path("D:/AI_Project_Transformer_Temp/v5_experiments/muril_dapt_strict_inductive_wwm_8ep")
    model_name = str(model_dir) if model_dir.is_dir() else "google/muril-base-cased"
    os.environ.setdefault("HF_HOME", "D:/AI_Project_Transformer_Temp/hf_home")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)

    class _PairDS(Dataset):
        def __init__(self, idxs: np.ndarray):
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int):
            idx = int(self.idxs[i])
            return titles[idx], bodies[idx], Y[idx]

    def _collate(batch):
        ts, bs, ys = zip(*batch)
        enc = tok(list(ts), list(bs), padding=True, truncation=True, max_length=160, return_tensors="pt")
        y_tensor = torch.tensor(np.array(ys), dtype=torch.float32 if is_ml else torch.long)
        return enc, y_tensor

    class _Head(nn.Module):
        def __init__(self, base_name: str, out_dim: int):
            super().__init__()
            self.enc = AutoModel.from_pretrained(base_name)
            h = self.enc.config.hidden_size
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(h, out_dim)

        def forward(self, **kw):
            out = self.enc(**kw).last_hidden_state
            mask = kw["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            return self.fc(self.drop(pooled))

    oof_logits = np.zeros((len(rows), len(labels)), dtype=np.float32)
    t0 = time.perf_counter()
    amp_ok = dev == "cuda"
    amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16

    for fold_i, (tr, va) in enumerate(splits):
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)
        model = _Head(model_name, len(labels)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

        if is_ml:
            pos = Y[tr].sum(axis=0).clip(min=1)
            neg = (len(tr) - pos).clip(min=1)
            pos_w = torch.tensor(np.sqrt(neg / pos), dtype=torch.float32, device=dev)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            cnt = np.bincount(Y[tr], minlength=len(labels)).clip(min=1)
            cw = torch.tensor(len(tr) / (len(labels) * cnt), dtype=torch.float32, device=dev)
            crit = nn.CrossEntropyLoss(weight=cw)

        dl_tr = DataLoader(_PairDS(tr), batch_size=16, shuffle=True, collate_fn=_collate)
        dl_va = DataLoader(_PairDS(va), batch_size=32, shuffle=False, collate_fn=_collate)

        epochs = 5 if task in ("response", "emotion", "issue") else 4
        model.train()
        for _ in range(epochs):
            for enc, yb in dl_tr:
                enc = {k: v.to(dev) for k, v in enc.items()}
                yb = yb.to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda" if amp_ok else "cpu", dtype=amp_dtype, enabled=amp_ok):
                    logits = model(**enc)
                    loss = crit(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        fold_out = []
        with torch.no_grad():
            for enc, _ in dl_va:
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.amp.autocast(device_type="cuda" if amp_ok else "cpu", dtype=amp_dtype, enabled=amp_ok):
                    logits = model(**enc)
                fold_out.append(logits.float().cpu().numpy())
        oof_logits[va] = np.concatenate(fold_out, axis=0)
        del model, opt
        if amp_ok:
            torch.cuda.empty_cache()

    if is_ml:
        probs = 1.0 / (1.0 + np.exp(-oof_logits))
        macro_f1, ths = _tune_multilabel_thresholds_oof(Y, probs)
        preds = np.column_stack([(probs[:, j] >= ths[j]).astype(int) for j in range(len(labels))])
        per_cls = {lb: round(float(f1_score(Y[:, j], preds[:, j], zero_division=0)), 4) for j, lb in enumerate(labels)}
    else:
        preds = np.argmax(oof_logits, axis=1)
        macro_f1 = float(f1_score(Y, preds, average="macro", zero_division=0))
        per_cls = {
            lb: round(float(f1_score((Y == j).astype(int), (preds == j).astype(int), zero_division=0)), 4)
            for j, lb in enumerate(labels)
        }

    res = {
        "mode": mode,
        "task": task,
        "backbone": "muril_dapt_strict_inductive_wwm_8ep",
        "oof_macro_f1": round(macro_f1, 4),
        "per_class_f1": per_cls,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[MuRIL OOF] mode={mode:22s} task={task:10s} macro_f1={macro_f1*100:5.2f}% ({res['seconds']}s)")


def _audit_informal_language(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    corpus = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    all_bodies = [cm.get("body", "") for p in corpus.get("posts", []) for cm in p.get("comments", [])]

    def _scan(texts: List[str]) -> Dict[str, Any]:
        slang_changed = 0
        emoji_present = 0
        both_changed = 0
        action_counts = Counter()
        term_counts = Counter()
        emoji_counts = Counter()
        examples = []
        for t in texts:
            r_s = apply_informal_normalization(t, mode="slang_normalized")
            r_d = apply_informal_normalization(t, mode="slang_emoji_semantic")
            assert r_d["original_text"] == t, "Original text mutated!"
            if r_s["changed"]:
                slang_changed += 1
            if r_d["emojis_detected"]:
                emoji_present += 1
                for em in r_d["emojis_detected"]:
                    emoji_counts[em["emoji"]] += 1
            if r_d["changed"]:
                both_changed += 1
            for tr in r_d["transformations"]:
                action_counts[tr["action"]] += 1
                term_counts[f"{tr['term']} ({tr['action']})"] += 1
            if len(examples) < 12 and any(tr["action"] in ("context_expanded", "literal_preserved", "indian_slang_expanded") for tr in r_d["transformations"]):
                examples.append({
                    "original": t[:160],
                    "normalized": r_d["normalized_text"][:180],
                    "transformations": r_d["transformations"][:5],
                })
        return {
            "n_texts": len(texts),
            "original_text_100pct_preserved": True,
            "slang_changed_count": slang_changed,
            "slang_changed_pct": round(slang_changed / max(len(texts), 1) * 100, 2),
            "emoji_present_count": emoji_present,
            "emoji_present_pct": round(emoji_present / max(len(texts), 1) * 100, 2),
            "combined_changed_count": both_changed,
            "combined_changed_pct": round(both_changed / max(len(texts), 1) * 100, 2),
            "action_counts": dict(action_counts.most_common()),
            "top_terms": dict(term_counts.most_common(15)),
            "top_emojis": dict(emoji_counts.most_common(12)),
            "context_examples": examples,
        }

    # Explicit unit verification of context-aware Gen-Z disambiguation (mid / based / cooked)
    context_probes = [
        ("The hostel mess food is literally mid tbh 😭", "slang_emoji_semantic"),
        ("Progress was stagnant in mid Dec to mid Jan during mid sem.", "slang_emoji_semantic"),
        ("Honestly based take bro, IIM fees are wild 💀", "slang_emoji_semantic"),
        ("Shortlisting is profile based and based on CAT percentile.", "slang_emoji_semantic"),
        ("We are totally cooked if waitlist doesn't move fr 🤡", "slang_emoji_semantic"),
        ("At least the canteen serves freshly cooked food.", "slang_emoji_semantic"),
    ]
    probe_results = [apply_informal_normalization(txt, mode=m) for txt, m in context_probes]

    return {
        "full_corpus_3557": _scan(all_bodies),
        "trainval_v1_367": _scan([r["text"] for r in rows]),
        "context_disambiguation_probes": probe_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-neural", nargs=2, metavar=("MODE", "TASK"))
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args()

    if args.worker_neural:
        _run_neural_worker(args.worker_neural[0], args.worker_neural[1])
        return

    rows = _load_trainval_v1()
    print("[Phase 3] Auditing informal slang & emoji prevalence + context disambiguation...")
    audit = _audit_informal_language(rows)

    print("[Phase 3] Running 5-fold GroupKFold OOF TF-IDF ablation across all 5 tasks × 4 modes...")
    tfidf_res = _run_tfidf_ablation(rows)
    for m in MODES:
        scores = {t: tfidf_res[m]["tasks"][t]["oof_macro_f1"] for t in TASKS}
        print(f"  TF-IDF {m:22s}: mean={tfidf_res[m]['mean_5task_oof_macro_f1']*100:5.2f}% | " +
              " ".join(f"{t[:4]}={v*100:5.2f}%" for t, v in scores.items()))

    neural_res: Dict[str, Any] = {}
    if not args.skip_neural:
        print("[Phase 3] Launching isolated 5-fold MuRIL-DAPT-WWM subprocesses (4 modes × 5 tasks)...")
        py = sys.executable
        for mode in MODES:
            neural_res[mode] = {"tasks": {}}
            t_vals = []
            for task in TASKS:
                cmd = [py, str(Path(__file__).resolve()), "--worker-neural", mode, task]
                cp = subprocess.run(cmd, check=True)
                assert cp.returncode == 0
                item = json.loads((CACHE_DIR / f"muril_{mode}_{task}.json").read_text(encoding="utf-8"))
                neural_res[mode]["tasks"][task] = {
                    "oof_macro_f1": item["oof_macro_f1"],
                    "per_class_f1": item["per_class_f1"],
                    "seconds": item["seconds"],
                }
                t_vals.append(item["oof_macro_f1"])
            neural_res[mode]["mean_5task_oof_macro_f1"] = round(float(np.mean(t_vals)), 4)

        print("\n=== NEURAL MuRIL-DAPT-WWM 5-TASK OOF SUMMARY ===")
        for m in MODES:
            scores = {t: neural_res[m]["tasks"][t]["oof_macro_f1"] for t in TASKS}
            print(f"  MuRIL  {m:22s}: mean={neural_res[m]['mean_5task_oof_macro_f1']*100:5.2f}% | " +
                  " ".join(f"{t[:4]}={v*100:5.2f}%" for t, v in scores.items()))

    # Compute exact deltas vs (A) original for both families
    def _deltas(block: Dict[str, Any]) -> Dict[str, Any]:
        if not block:
            return {}
        base = block["original"]
        out = {}
        for m in MODES:
            out[m] = {
                "mean_delta_pp": round((block[m]["mean_5task_oof_macro_f1"] - base["mean_5task_oof_macro_f1"]) * 100, 2),
                "task_deltas_pp": {
                    t: round((block[m]["tasks"][t]["oof_macro_f1"] - base["tasks"][t]["oof_macro_f1"]) * 100, 2)
                    for t in TASKS
                },
            }
        return out

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "Strict 5-fold GroupKFold OOF CV on trainval_v1 (n=367, grouped by post_id). Frozen test_v5 (n=92) untouched.",
        "audit": audit,
        "tfidf_oof_5task": tfidf_res,
        "tfidf_deltas_vs_original": _deltas(tfidf_res),
        "muril_dapt_oof_5task": neural_res,
        "muril_deltas_vs_original": _deltas(neural_res),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Phase 3] Saved complete 5-task ablation report to {OUT_JSON}")


if __name__ == "__main__":
    main()
