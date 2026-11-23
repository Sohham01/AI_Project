"""Phase 2 Multilingual Language, Script, Span & Romanized Normalization Evaluation.

Strictly separates:
1. Manually reviewed REAL Reddit comment evaluation set (n = 84 real comments
   from ``data/live_corpus/corpus.json``), saved to
   ``data/annotations/multilingual_eval_real_v1.csv``.
2. Controlled SYNTHETIC multilingual diagnostic suite (n = 24 covering Hindi
   Devanagari, Bengali script, Odia script, Banglish, Roman-Odia, mixed scripts,
   and multi-sentence code-switching), saved to
   ``data/annotations/multilingual_eval_synthetic_probes_v1.csv``.

Computes per-language Precision/Recall/F1, script classification accuracy,
code-mixed & multi-span detection metrics, full confusion matrices, confidence &
abstention curves, and non-destructive Romanized normalization audit across the
full 3,557-comment real corpus.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.nlp.indian_reddit_lid import (
    detect_indian_reddit_language,
    normalize_romanized_indic,
)

CORPUS_PATH = ROOT / "data" / "live_corpus" / "corpus.json"
REAL_EVAL_CSV = ROOT / "data" / "annotations" / "multilingual_eval_real_v1.csv"
SYNTH_EVAL_CSV = ROOT / "data" / "annotations" / "multilingual_eval_synthetic_probes_v1.csv"
OUT_JSON = ROOT / "ml" / "artifacts" / "v6_programme" / "multilingual_lid_eval.json"

# Manually reviewed linguistic annotations on 84 real corpus comments
# Format: (comment_id, gold_language, gold_script, gold_is_code_mixed, gold_multi_span, linguistic_note)
REAL_GOLD_ANNOTATIONS: List[Tuple[str, str, str, bool, bool, str]] = [
    ("cmt_00009", "hi-en", "latin", True, False, "Romanized Hindi syntax + English 'wait list' + emoji"),
    ("cmt_00018", "und", "unknown", False, False, "Numeric academic profile only (94/93.2/9.6)"),
    ("cmt_00026", "und", "unknown", False, False, "Numeric waitlist number only (3005)"),
    ("cmt_00040", "hi-en", "latin", True, False, "Romanized Hindi + English 'mostly'"),
    ("cmt_00050", "hi-en", "latin", True, False, "English matrix + Hinglish phrase 'iss baar'"),
    ("cmt_00054", "hi-en", "latin", True, False, "Romanized Hindi 'Hogya bhai' + English 'convert'"),
    ("cmt_00058", "hi-en", "latin", True, False, "English 'DM me ... sectional' + Hindi 'reh gya usme'"),
    ("cmt_00060", "hi", "latin", False, False, "Pure Romanized Hindi idiom"),
    ("cmt_00071", "hi-en", "latin", True, True, "Multi-span switch: Sent 1 English, Sent 2 Romanized Hindi"),
    ("cmt_01028", "hi-en", "latin", True, True, "Multi-span switch: 4 English sentences + final Hinglish sentence"),
    ("cmt_01610", "hi-en", "mixed", True, False, "Mixed Devanagari ('गोलगप्पे') + Romanized Hindi + English"),
    ("cmt_01936", "or-en", "latin", True, True, "Genuine Romanized Odia ('Ae sabu kana pasuchu') + English/Hindi"),
    ("cmt_03434", "en", "latin", False, False, "English MBA cutoff query"),
    ("cmt_00565", "en", "latin", False, False, "Colloquial English reaction"),
    ("cmt_00151", "en", "latin", False, False, "English factual clarification"),
    ("cmt_01401", "en", "latin", False, False, "English waitlist discussion"),
    ("cmt_01258", "en", "latin", False, False, "Short English question"),
    ("cmt_01146", "en", "latin", False, False, "Short English MBA question"),
    ("cmt_00715", "en", "latin", False, False, "English placement query"),
    ("cmt_00524", "en", "latin", False, False, "Multi-line English comparison"),
    ("cmt_02911", "en", "latin", False, False, "Short English advice"),
    ("cmt_00445", "en", "latin", False, False, "English query with spelling typo ('perecentitle')"),
    ("cmt_03163", "en", "latin", False, False, "English personal reflection"),
    ("cmt_02184", "en", "latin", False, False, "English deadline instruction"),
    ("cmt_00184", "en", "latin", False, False, "English program comparison"),
    ("cmt_00176", "en", "latin", False, False, "English portal status report"),
    ("cmt_00479", "en", "latin", False, False, "English with emoji"),
    ("cmt_01125", "en", "latin", False, False, "Short Indian-MBA English abbreviation query"),
    ("cmt_01192", "en", "latin", False, False, "Sarcastic English comment with emoji"),
    ("cmt_02674", "en", "latin", False, False, "English website info comment"),
    ("cmt_03230", "en", "latin", False, False, "English waitlist movement update"),
    ("cmt_00161", "en", "latin", False, False, "English median package query"),
    ("cmt_03002", "en", "latin", False, False, "Indian English counselling reply"),
    ("cmt_01016", "en", "latin", False, False, "Short English exclamation"),
    ("cmt_03132", "hi-en", "latin", True, False, "Romanized Hindi + English 'phone' + emoji"),
    ("cmt_02537", "hi-en", "latin", True, False, "Two-sentence Hinglish waitlist query"),
    ("cmt_01569", "hi-en", "latin", True, False, "Short English 'Thanks' + Hindi 'bhai'"),
    ("cmt_02657", "hi-en", "latin", True, False, "Gen-Z 'Ig' + Hinglish + English 'portal update till now'"),
    ("cmt_03368", "hi-en", "latin", True, False, "Slang + Hinglish + English 'classes start admission'"),
    ("cmt_01902", "hi-en", "latin", True, False, "English slang 'wdym' + Hinglish 'matlab aa rha aek aur'"),
    ("cmt_00052", "hi-en", "latin", True, False, "Hinglish 'Bhai EWS ka toh' + English"),
    ("cmt_00952", "hi-en", "latin", True, False, "Hinglish 'Kya pata' + English clause"),
    ("cmt_02549", "hi-en", "latin", True, True, "Multi-span switch: Sent 1 Hinglish, Sent 2 English"),
    ("cmt_02234", "hi-en", "latin", True, False, "English 'Just kidding' + Hindi 'Bhai'"),
    ("cmt_00934", "en", "latin", False, False, "English college ranking containing proper noun 'Bodh gaya'"),
    ("cmt_01562", "hi-en", "latin", True, False, "Hinglish with English 'interview'"),
    ("cmt_02218", "hi-en", "latin", True, False, "Hindi vocative 'Bhai' + English sentence"),
    ("cmt_00665", "hi-en", "latin", True, True, "Multi-span switch: Sent 1 Hinglish, Sent 2 English"),
    ("cmt_00602", "hi-en", "latin", True, False, "English phrase + Hindi 'h bhai ki'"),
    ("cmt_02397", "hi-en", "latin", True, False, "Hinglish 'Koi na yaawr' + English slang"),
    ("cmt_00606", "hi-en", "latin", True, False, "Hinglish + English 'Fake news' / 'course'"),
    ("cmt_02320", "hi-en", "latin", True, False, "Hinglish clause + English clause"),
    ("cmt_02240", "hi-en", "latin", True, False, "Romanized Hindi + English 'release'"),
    ("cmt_03435", "hi-en", "latin", True, False, "Hinglish 'Koi na' + English advice"),
    ("cmt_01487", "hi", "latin", False, False, "Pure Romanized Hindi ('Thik hai bhai')"),
    ("cmt_03190", "hi", "latin", False, False, "Pure Romanized Hindi ('Ye hai kya')"),
    ("cmt_00082", "hi-en", "latin", True, False, "Romanized Hindi + English acronym 'iims'"),
    ("cmt_03052", "hi", "latin", False, False, "Pure Romanized Hindi ('Bhai Tera hua kya')"),
    ("cmt_02207", "hi", "latin", False, False, "Pure Romanized Hindi phonetic contraction ('Mtlb?')"),
    ("cmt_02555", "hi-en", "latin", True, False, "Romanized Hindi + English acronym 'ews'"),
    ("cmt_00411", "hi-en", "latin", True, False, "Romanized Hindi + English 'placements'"),
    ("cmt_01933", "hi-en", "latin", True, False, "Hinglish + English 'Try', 'time'"),
    ("cmt_00154", "hi-en", "latin", True, False, "English 'Package' + Romanized Hindi 'kitna lga??'"),
    ("cmt_02926", "hi-en", "latin", True, False, "English 'Dm' + Romanized Hindi 'kro uska bhi hai'"),
    ("cmt_01898", "en", "latin", False, False, "Single English word ('Tomorrow')"),
    ("cmt_03555", "en", "latin", False, False, "Single English abbreviation ('Obv')"),
    ("cmt_02274", "en", "latin", False, False, "Single English word ('Welcome!!')"),
    ("cmt_03350", "und", "latin", False, False, "Proper noun festival name ('Durga Puja?') — language-ambiguous"),
    ("cmt_01201", "und", "latin", False, False, "Reddit moderation placeholder ('[ Removed by Reddit ]')"),
    ("cmt_00468", "en", "latin", False, False, "Single English word ('Confusing')"),
    ("cmt_00327", "en", "latin", False, False, "Single English word ('Sarcasm')"),
    ("cmt_01420", "en", "latin", False, False, "Two-word English phrase ('genuinely asking')"),
    ("cmt_01843", "und", "unknown", False, False, "Emoji-only comment ('🤣🤣🤣')"),
    ("cmt_00480", "en", "latin", False, False, "Elongated English word + emoji ('Whyy 😭?')"),
    ("cmt_01646", "en", "latin", False, False, "Informal English with 'Bro'"),
    ("cmt_01089", "en", "latin", False, False, "Long English ROI discussion"),
    ("cmt_02293", "en", "latin", False, False, "English with 'bro'"),
    ("cmt_01967", "en", "latin", False, False, "English with 'bro'"),
    ("cmt_02646", "hi-en", "latin", True, True, "Multi-span: 'Ha' (Hindi yes) + English update, Sent 2 English"),
    ("cmt_03445", "en", "latin", False, False, "English humorous commentary with 'bro'"),
    ("cmt_02252", "en", "latin", False, False, "English DM request with 'Bro'"),
    ("cmt_01382", "en", "latin", False, False, "Short English reply ('No bro')"),
    ("cmt_02256", "en", "latin", False, False, "Short English reply ('No idea bro')"),
    ("cmt_02231", "en", "latin", False, False, "English JEE rank question with 'bro'"),
]

# Strictly separate controlled synthetic multilingual probe suite (NEVER mixed with real eval!)
SYNTHETIC_PROBES: List[Tuple[str, str, str, str, bool, bool]] = [
    ("syn_01", "Is the placement record at IIM Sambalpur improving this year?", "en", "latin", False, False),
    ("syn_02", "हॉस्टल और मेस का खाना बहुत खराब है।", "hi", "devanagari", False, False),
    ("syn_03", "क्या IIM Sambalpur अच्छा है placement के लिए?", "hi-en", "mixed", True, False),
    ("syn_04", "Bhai hostel fees bohot high hai aur mess food literally mid hai.", "hi-en", "latin", True, False),
    ("syn_05", "Thik hai bhai, abhi kuch nahi hoga.", "hi", "latin", False, False),
    ("syn_06", "IIM placement season is stressful. Bhai kuch samajh nahi aa raha.", "hi-en", "latin", True, True),
    ("syn_07", "এই কলেজের হোস্টেল আর মেসের খাবার কেমন?", "bn", "bengali", False, False),
    ("syn_08", "আমি IIM Sambalpur এর placement report দেখলাম।", "bn-en", "mixed", True, False),
    ("syn_09", "IIM e giye ki actually ROI pawa jay? fees onek beshi.", "bn-en", "latin", True, False),
    ("syn_10", "Ami bujhte parchi na keno eto taka lagche.", "bn", "latin", False, False),
    ("syn_11", "The curriculum looks intense. Kintu amar mone hoy placement bhalo hobe.", "bn-en", "latin", True, True),
    ("syn_12", "হোস্টেল ভালো না। But the faculty support is decent.", "bn-en", "mixed", True, True),
    ("syn_13", "ହଷ୍ଟେଲ ରୁମ୍ ଆଉ ମେସ୍ ଖାଦ୍ୟ ବହୁତ ଖରାପ ଅଛି।", "or", "odia", False, False),
    ("syn_14", "ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro।", "or-en", "mixed", True, False),
    ("syn_15", "IIM Sambalpur ra placement package kemiti achi?", "or-en", "latin", True, False),
    ("syn_16", "Aaji class bahut hectic thila, katha bujhili na.", "or-en", "latin", True, False),
    ("syn_17", "Tume kouthi rahucha ebe? Kichi katha bujhili.", "or", "latin", False, False),
    ("syn_18", "The campus is remote. Kintu hostel ra condition bhala achi.", "or-en", "latin", True, True),
    ("syn_19", "95.4 / 92 / 8.7", "und", "unknown", False, False),
    ("syn_20", "😭😭😭", "und", "unknown", False, False),
    ("syn_21", "Waitlist moved by 45 today. बहुत अच्छी खबर है!", "hi-en", "mixed", True, True),
    ("syn_22", "Don't take a huge loan without checking median CTC. Paisa bachana zaroori hai.", "hi-en", "latin", True, True),
    ("syn_23", "Kolkata theke Sambalpur train ache ki? Direct route helps.", "bn-en", "latin", True, True),
    ("syn_24", "Bhubaneswar ru direct train achi. Travel time is around 5 hours.", "or-en", "latin", True, True),
]


def _prf_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: Dict[str, Dict[str, Any]] = {}
    f1s: List[float] = []
    cm: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(y_true, y_pred):
        cm[g][p] += 1

    for lb in labels:
        tp = cm[lb][lb]
        fp = sum(cm[g][lb] for g in labels if g != lb)
        fn = sum(cm[lb][p] for p in labels if p != lb)
        support = sum(cm[lb].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[lb] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1s.append(f1)

    acc = sum(1 for g, p in zip(y_true, y_pred) if g == p) / max(len(y_true), 1)
    return {
        "n_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _binary_prf(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for g, p in zip(y_true, y_pred) if g and p)
    fp = sum(1 for g, p in zip(y_true, y_pred) if not g and p)
    fn = sum(1 for g, p in zip(y_true, y_pred) if g and not p)
    tn = sum(1 for g, p in zip(y_true, y_pred) if not g and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_lang_true, y_lang_pred = [], []
    y_script_true, y_script_pred = [], []
    y_cm_true, y_cm_pred = [], []
    y_ms_true, y_ms_pred = [], []
    errors: List[Dict[str, Any]] = []
    abstain_stats = {"total_abstained": 0, "abstained_on_true_und": 0, "abstained_on_labeled": 0, "non_abstained_correct": 0, "non_abstained_total": 0}
    conf_correct: List[float] = []
    conf_wrong: List[float] = []

    for r in rows:
        pred = detect_indian_reddit_language(r["text"])
        gl = r["gold_language"]
        pl = pred["language"]
        gs = r["gold_script"]
        ps = pred["script"]
        gcm = bool(r["gold_is_code_mixed"])
        pcm = bool(pred["is_code_mixed"])
        gms = bool(r["gold_multi_span"])
        pms = bool(pred["has_multi_language_spans"])

        y_lang_true.append(gl)
        y_lang_pred.append(pl)
        y_script_true.append(gs)
        y_script_pred.append(ps)
        y_cm_true.append(gcm)
        y_cm_pred.append(pcm)
        y_ms_true.append(gms)
        y_ms_pred.append(pms)

        if pred["abstain"]:
            abstain_stats["total_abstained"] += 1
            if gl == "und":
                abstain_stats["abstained_on_true_und"] += 1
            else:
                abstain_stats["abstained_on_labeled"] += 1
        else:
            abstain_stats["non_abstained_total"] += 1
            if pl == gl:
                abstain_stats["non_abstained_correct"] += 1

        if pl == gl:
            conf_correct.append(float(pred["confidence"]))
        else:
            conf_wrong.append(float(pred["confidence"]))
            errors.append({
                "id": r["id"],
                "text": r["text"][:180],
                "gold_language": gl,
                "pred_language": pl,
                "gold_script": gs,
                "pred_script": ps,
                "pred_confidence": pred["confidence"],
                "pred_abstain": pred["abstain"],
                "matched_markers": pred["matched_markers"],
                "note": r.get("note", ""),
            })

    non_abs_tot = max(abstain_stats["non_abstained_total"], 1)
    abstain_stats["coverage"] = round(abstain_stats["non_abstained_total"] / max(len(rows), 1), 4)
    abstain_stats["selective_accuracy_when_confident"] = round(abstain_stats["non_abstained_correct"] / non_abs_tot, 4)
    abstain_stats["mean_confidence_correct"] = round(sum(conf_correct) / max(len(conf_correct), 1), 4)
    abstain_stats["mean_confidence_errors"] = round(sum(conf_wrong) / max(len(conf_wrong), 1), 4)

    return {
        "language_metrics": _prf_report(y_lang_true, y_lang_pred),
        "script_metrics": _prf_report(y_script_true, y_script_pred),
        "code_mixed_detection": _binary_prf(y_cm_true, y_cm_pred),
        "multi_language_span_detection": _binary_prf(y_ms_true, y_ms_pred),
        "confidence_and_abstention": abstain_stats,
        "errors": errors,
    }


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    by_cid: Dict[str, Tuple[str, str]] = {}
    all_bodies: List[str] = []
    for p in corpus.get("posts", []):
        for cm in p.get("comments", []):
            cid = cm["reddit_id"]
            body = cm.get("body", "")
            by_cid[cid] = (p["reddit_id"], body)
            all_bodies.append(body)

    # 1. Write & evaluate REAL manually reviewed dataset
    real_rows: List[Dict[str, Any]] = []
    for cid, gl, gs, gcm, gms, note in REAL_GOLD_ANNOTATIONS:
        pid, body = by_cid[cid]
        real_rows.append({
            "id": cid,
            "post_id": pid,
            "text": body,
            "gold_language": gl,
            "gold_script": gs,
            "gold_is_code_mixed": gcm,
            "gold_multi_span": gms,
            "eval_scope": "real_reddit_corpus_manual_review_v1",
            "note": note,
        })

    REAL_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REAL_EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()))
        w.writeheader()
        w.writerows(real_rows)

    real_eval = _evaluate_rows(real_rows)

    # 2. Write & evaluate strictly separated SYNTHETIC multilingual probe suite
    synth_rows: List[Dict[str, Any]] = []
    for sid, txt, gl, gs, gcm, gms in SYNTHETIC_PROBES:
        synth_rows.append({
            "id": sid,
            "text": txt,
            "gold_language": gl,
            "gold_script": gs,
            "gold_is_code_mixed": gcm,
            "gold_multi_span": gms,
            "eval_scope": "controlled_synthetic_probe_NOT_real_world_proof",
            "note": "Synthetic cross-script & Romanized diagnostic probe",
        })
    with SYNTH_EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(synth_rows[0].keys()))
        w.writeheader()
        w.writerows(synth_rows)

    synth_eval = _evaluate_rows(synth_rows)

    # 3. Audit non-destructive Romanized normalization & span detection across all 3,557 real comments
    norm_changed = 0
    total_transforms = 0
    cat_counts = Counter()
    rule_counts = Counter()
    sample_norms = []
    multi_span_docs = 0
    abstain_docs = 0
    doc_lang_counts = Counter()
    doc_script_counts = Counter()

    for body in all_bodies:
        nres = normalize_romanized_indic(body)
        assert nres["original_text"] == body, "Original text was mutated!"
        if nres["changed"]:
            norm_changed += 1
            total_transforms += len(nres["transformations"])
            for tr in nres["transformations"]:
                cat_counts[tr["category"]] += 1
                rule_counts[f"{tr['original'].lower()}->{tr['normalized']}"] += 1
            if len(sample_norms) < 12:
                sample_norms.append({
                    "original": body[:160],
                    "normalized": nres["normalized_text"][:160],
                    "transformations": nres["transformations"][:5],
                })
        lres = detect_indian_reddit_language(body)
        doc_lang_counts[lres["language"]] += 1
        doc_script_counts[lres["script"]] += 1
        if lres["has_multi_language_spans"]:
            multi_span_docs += 1
        if lres["abstain"]:
            abstain_docs += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "real_corpus_eval": {
            "dataset_path": str(REAL_EVAL_CSV.relative_to(ROOT)),
            "scope": "real_reddit_comments_manually_reviewed_v1",
            **real_eval,
        },
        "synthetic_probe_eval": {
            "dataset_path": str(SYNTH_EVAL_CSV.relative_to(ROOT)),
            "scope": "controlled_synthetic_multilingual_diagnostic_separate_from_real_corpus",
            **synth_eval,
        },
        "full_corpus_audit": {
            "n_comments": len(all_bodies),
            "original_text_100pct_preserved": True,
            "language_distribution": dict(doc_lang_counts.most_common()),
            "script_distribution": dict(doc_script_counts.most_common()),
            "multi_language_span_comments": multi_span_docs,
            "abstained_comments": abstain_docs,
            "romanized_normalization_comments_touched": norm_changed,
            "romanized_normalization_total_edits": total_transforms,
            "transformation_categories": dict(cat_counts.most_common()),
            "top_variant_mappings": dict(rule_counts.most_common(15)),
            "sample_normalizations": sample_norms,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== PHASE 2 REAL CORPUS EVAL (n=84) ===")
    print("  Lang acc:", real_eval["language_metrics"]["accuracy"], "macro F1:", real_eval["language_metrics"]["macro_f1"])
    print("  Per-class:", json.dumps(real_eval["language_metrics"]["per_class"]))
    print("  Script acc:", real_eval["script_metrics"]["accuracy"])
    print("  Code-mix F1:", real_eval["code_mixed_detection"]["f1"])
    print("  Multi-span F1:", real_eval["multi_language_span_detection"]["f1"])
    print("  Abstention:", json.dumps(real_eval["confidence_and_abstention"]))
    print(f"  Real errors ({len(real_eval['errors'])}):")
    for e in real_eval["errors"]:
        print(f"    {e['id']}: gold={e['gold_language']} pred={e['pred_language']} conf={e['pred_confidence']} markers={e['matched_markers']} | {e['text'][:90]}")

    print("\n=== PHASE 2 SYNTHETIC PROBE EVAL (n=24, SEPARATE) ===")
    print("  Lang acc:", synth_eval["language_metrics"]["accuracy"], "macro F1:", synth_eval["language_metrics"]["macro_f1"])
    print("  Script acc:", synth_eval["script_metrics"]["accuracy"])
    print("  Multi-span F1:", synth_eval["multi_language_span_detection"]["f1"])
    if synth_eval["errors"]:
        print(f"  Synth errors ({len(synth_eval['errors'])}):")
        for e in synth_eval["errors"]:
            print(f"    {e['id']}: gold={e['gold_language']} pred={e['pred_language']} | {e['text'][:90]}")

    print("\n=== FULL 3,557-COMMENT CORPUS AUDIT ===")
    print("  Languages:", report["full_corpus_audit"]["language_distribution"])
    print("  Multi-lang span docs:", multi_span_docs, "| Abstained:", abstain_docs)
    print("  Romanized normalizations:", norm_changed, "comments,", total_transforms, "edits")


if __name__ == "__main__":
    main()
