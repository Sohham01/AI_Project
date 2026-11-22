"""Write the blind Full-Arctic-Shift-Context Pass-2 Audit CSV (50 items) and score against Pass-1 V2.

PROVENANCE & SCIENTIFIC INTEGRITY NOTE:
- Every one of the 50 items was annotated BLIND to Pass-1 V2 labels after inspecting full local
  corpus thread trees AND live Arctic Shift API community responses (27/27 posts matched live;
  +136 extra comments fetched; all PII redacted).
- This file is explicitly tagged `annotator_id = "ai_arctic_shift_full_context_pass2"` and saved
  to `iaa_50_ai_arctic_shift_context_audit.csv` — NEVER falsely labeled as independent human IAA.
- `iaa_50_human_annotator_studio.csv` remains strictly blank for true human blind review.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import CANONICAL_LABELS, EXP_ROOT, save_json
from ml.experiments.v5_optimization.score_human_iaa import (
    _load_studio_csv,
    _load_v2_pass1_reference,
    compute_iaa_report,
)

AL_DIR = EXP_ROOT / "active_learning"
TEMPLATE_STUDIO_CSV = AL_DIR / "iaa_50_human_annotator_studio.csv"
OUT_AUDIT_CSV = AL_DIR / "iaa_50_ai_arctic_shift_context_audit.csv"
OUT_AUDIT_JSON = AL_DIR / "iaa_ai_arctic_shift_audit_results.json"
OUT_DISAGREE_CSV = AL_DIR / "iaa_ai_arctic_shift_disagreements.csv"
OUT_AUDIT_REPORT_MD = ROOT / "docs" / "V5_ARCTIC_SHIFT_CONTEXT_AUDIT_REPORT.md"

# 50 blind Pass-2 labels adjudicated from full local + live Arctic Shift thread trees & community replies.
# Format: doc_id -> (review_status, abstain_reason, sentiment, emotions, issues, response_type, is_solution, notes)
PASS2_BLIND_LABELS = {
    "doc-v2-cmt_01739": ("labeled", "", "positive", "", "academic_workload|hostel_facilities", "personal_experience", "not_solution", "AMA firsthand report on 1st-sem workload and double-sharing rooms; reply asks follow-up"),
    "doc-v2-cmt_01807": ("labeled", "", "neutral", "", "placements", "opinion", "not_solution", "Factual CSD internship rule (sem breaks, offline); reply asks follow-up on cities/expenses"),
    "doc-v2-cmt_01744": ("labeled", "", "positive", "hope", "curriculum_teaching|placements", "personal_experience", "not_solution", "Reports NEP 2020 3rd-yr exit policy + batchmates' plans + constructive placement outlook"),
    "doc-v2-cmt_01759": ("labeled", "", "neutral", "", "placements", "personal_experience", "not_solution", "Reports current batch social internship NGO tie-ups; reply clarifies wanting DSAI info"),
    "doc-v2-cmt_01621": ("labeled", "", "neutral", "", "curriculum_teaching", "advice", "not_solution", "Compares IITM BS vs IIM Sbp BS domains and advises choosing better option"),
    "doc-v2-cmt_03211": ("labeled", "", "neutral", "", "curriculum_teaching|placements", "advice", "not_solution", "Multi-clause conditional decision advice; community reply challenges no-placement claim"),
    "doc-v2-cmt_00822": ("labeled", "", "positive", "confidence", "curriculum_teaching|placements", "opinion", "not_solution", "Authoritative course/flagship ranking praising IMT marketing placements; 2 appreciative replies"),
    "doc-v2-cmt_01043": ("labeled", "", "neutral", "curiosity|stress", "placements", "personal_experience", "not_solution", "28yo 5yoe shares GRE/1yr-vs-2yr MBA pivot dilemma; alum replies confirming pivot challenge"),
    "doc-v2-cmt_01831": ("labeled", "", "negative", "frustration", "curriculum_teaching|placements", "counterargument", "not_solution", "Pushes back on unrealistic 1st-yr finance internship expectation; notes curriculum change; reply concedes"),
    "doc-v2-cmt_02320": ("labeled", "", "positive", "", "extracurriculars", "opinion", "not_solution", "Fresher asking to be added to student-run freshers showcase website"),
    "doc-v2-cmt_01823": ("abstain", "off_topic_noise", "", "", "", "", "", "Inside-joke meme banter ('tt champ crash out') after joking CSDC apology chain"),
    "doc-v2-cmt_00212": ("labeled", "", "neutral", "", "", "counterargument", "not_solution", "Corrects parent's 5-year E-MBA misconception with exact 3/4/5-year eligibility facts"),
    "doc-v2-cmt_01824": ("labeled", "", "neutral", "curiosity", "curriculum_teaching|placements", "opinion", "not_solution", "Evaluates BBA-like curriculum + eco/policy depth and asks internship parity vs Symbiosis"),
    "doc-v2-cmt_00900": ("labeled", "", "neutral", "", "placements", "advice", "solution", "Candid warning + concrete actionable PI prep steps for gap years; parent replied 'Thanks, mind if i Dm?'"),
    "doc-v2-cmt_01811": ("labeled", "", "neutral", "hope", "placements", "personal_experience", "not_solution", "Current student reports batch operations/NGO internships and expected corporate roles"),
    "doc-v2-cmt_01860": ("labeled", "", "neutral", "", "", "agreement", "not_solution", "Confirms sending DM after parent said 'Dm me'"),
    "doc-v2-cmt_02480": ("labeled", "", "negative", "", "placements", "counterargument", "not_solution", "Directly refutes parent's '100% internship' reassurance on placement worth"),
    "doc-v2-cmt_01878": ("labeled", "", "negative", "", "", "advice", "not_solution", "Advises targeting private colleges as better than IIM Sambalpur after low WL chances"),
    "doc-v2-cmt_01855": ("labeled", "", "neutral", "", "financial_concerns", "opinion", "not_solution", "Bare factual confirmation of SC/ST/OBC scholarship; 3 replies still ask for details/confirmation"),
    "doc-v2-cmt_00304": ("labeled", "", "neutral", "curiosity|stress", "", "counterargument", "not_solution", "Pushes back on PGPX fit due to 7-8yr avg workex and asks about CAT drop; reply counters with 4yr eligibility"),
    "doc-v2-cmt_00917": ("labeled", "", "negative", "frustration", "placements", "counterargument", "not_solution", "OP sharply refutes commenter defending baby IIMs on OP's placement critique post"),
    "doc-v2-cmt_01859": ("labeled", "", "neutral", "", "", "opinion", "not_solution", "Asks commenter to initiate Reddit chat after 'Dm me'"),
    "doc-v2-cmt_00211": ("labeled", "", "neutral", "curiosity", "", "counterargument", "not_solution", "Politely challenges/qualifies parent's 3-yr E-MBA claim citing 5-yr requirement; parent clarifies"),
    "doc-v2-cmt_01977": ("labeled", "", "neutral", "", "financial_concerns", "opinion", "not_solution", "Factual clarification that need-based scholarship is only for MBA students"),
    "doc-v2-cmt_02016": ("labeled", "", "negative", "frustration", "", "counterargument", "not_solution", "Sarcastic callout challenging credibility of brand-new account warning students off BS DSAI"),
    "doc-v2-cmt_00843": ("labeled", "", "negative", "", "placements", "counterargument", "not_solution", "Pushes back that students must also demonstrate skills for placements; community reply pushes back on PlaceCom"),
    "doc-v2-cmt_00690": ("labeled", "", "positive", "confidence", "placements", "opinion", "not_solution", "Praises top-10 B-school alumni network, brand tag, and career opportunities; parent agrees"),
    "doc-v2-cmt_03501": ("labeled", "", "neutral", "", "placements", "counterargument", "not_solution", "Explicitly refutes parent ('Not really bro') citing freshers getting strong SIPs at IIM M"),
    "doc-v2-cmt_02017": ("labeled", "", "negative", "frustration|confidence", "curriculum_teaching|financial_concerns|placements", "counterargument", "not_solution", "Heated point-by-point refutation of OP's BS DSAI bashing & loan math; accuses OP of waitlist manipulation"),
    "doc-v2-cmt_00721": ("labeled", "", "neutral", "confidence", "placements", "counterargument", "not_solution", "Data-driven refutation of IIM I/K ranking vs FMS/SPJIMR/XL citing 5-10yr outcomes; opponent replies respectfully"),
    "doc-v2-cmt_03532": ("labeled", "", "negative", "", "hostel_facilities|placements", "opinion", "not_solution", "Sarcastic roast mocking both placements and mess food on rat-in-mess post"),
    "doc-v2-cmt_03530": ("labeled", "", "neutral", "", "", "opinion", "not_solution", "Pure wordplay pun ('cat would eat him') continuing parent's rat-CAT-score joke"),
    "doc-v2-cmt_03524": ("labeled", "", "negative", "frustration", "financial_concerns|hostel_facilities", "opinion", "not_solution", "Angry complaint questioning fees when basic mess hygiene is not maintained"),
    "doc-v2-cmt_03531": ("labeled", "", "negative", "", "hostel_facilities", "opinion", "not_solution", "Visceral disgust reaction ('would vomit seeing this') to rat in mess post"),
    "doc-v2-cmt_02059": ("labeled", "", "neutral", "", "hostel_facilities", "solution", "solution", "Concrete actionable dorm cooking workaround (keep electric kettles); OP moves directly to next question"),
    "doc-v2-cmt_02196": ("labeled", "", "negative", "", "financial_concerns", "opinion", "not_solution", "Critiques massive 30L/42L education loans as a long-run 'honey trap'"),
    "doc-v2-cmt_00413": ("labeled", "", "neutral", "confidence", "curriculum_teaching|financial_concerns", "advice", "solution", "Recommends specific Rs. 600 Udemy resource for learning DS in depth; OP pushes back on needing undergrad degree"),
    "doc-v2-cmt_00414": ("labeled", "", "neutral", "curiosity", "curriculum_teaching", "counterargument", "not_solution", "OP pushes back on Udemy suggestion ('What about undergrad? I don't want to be 12th pass lol'); parent concedes"),
    "doc-v2-cmt_00415": ("labeled", "", "neutral", "", "curriculum_teaching|placements", "advice", "not_solution", "Clarifies prior Udemy tip, suggests MBA option, notes friend's Amul placement while admitting AI/ML specifics unknown"),
    "doc-v2-cmt_00009": ("labeled", "", "negative", "", "", "opinion", "not_solution", "Mocking quip disparaging IIM Sambalpur waitlisting"),
    "doc-v2-cmt_02654": ("labeled", "", "negative", "frustration", "administrative_communication", "opinion", "not_solution", "Complains waitlist numbers are unreleased and urges IIM Sambalpur to release 3rd list"),
    "doc-v2-cmt_02656": ("labeled", "", "neutral", "", "administrative_communication", "solution", "solution", "Direct actionable pointer ('see the mail IIM mumbai had sent') answering 'Which list btw?'"),
    "doc-v2-cmt_02657": ("labeled", "", "neutral", "", "administrative_communication", "opinion", "not_solution", "Observes admission portal has not been updated yet"),
    "doc-v2-cmt_01131": ("labeled", "", "positive", "hope", "", "advice", "not_solution", "Encouraging college-selection advice (XIMB, IIT R, SPJIMR + 36m workex note) for Big4 consultant profile"),
    "doc-v2-cmt_03062": ("labeled", "", "negative", "frustration", "administrative_communication", "personal_experience", "not_solution", "Reports not receiving email + confusing portal/email discrepancy on 3rd merit list"),
    "doc-v2-cmt_03063": ("labeled", "", "neutral", "curiosity", "administrative_communication", "opinion", "not_solution", "Asks community whether anyone received 3rd merit list DSAI offer/WL movement"),
    "doc-v2-cmt_00551": ("labeled", "", "positive", "hope|confidence", "", "advice", "solution", "Detailed encouraging career & CAT action plan (core/WFH job instead of drop + 3 hrs/day study)"),
    "doc-v2-cmt_02617": ("labeled", "", "neutral", "", "administrative_communication", "opinion", "not_solution", "Speculative political explanation for why 3rd merit list is delayed"),
    "doc-v2-cmt_02618": ("labeled", "", "neutral", "", "administrative_communication", "opinion", "not_solution", "Speculates all seats may be filled explaining 3rd merit list delay; reply pushes back 'No way lol'"),
    "doc-v2-cmt_02619": ("labeled", "", "neutral", "", "administrative_communication", "opinion", "not_solution", "Echoes speculation that all seats appear filled on delayed 3rd list thread"),
}


def validate_entry(doc_id: str, entry: tuple) -> None:
    status, reason, sent, emos, issues, resp, sol, _ = entry
    assert status in {"labeled", "abstain", "uncertain", "excluded"}, f"{doc_id}: bad status {status}"
    if status != "labeled":
        assert not any([sent, emos, issues, resp, sol]), f"{doc_id}: non-labeled has labels"
        assert reason, f"{doc_id}: missing abstain_reason"
        return
    assert sent in CANONICAL_LABELS["sentiment"], f"{doc_id}: bad sentiment {sent}"
    assert resp in CANONICAL_LABELS["response"], f"{doc_id}: bad response {resp}"
    assert sol in CANONICAL_LABELS["solution"], f"{doc_id}: bad solution {sol}"
    for e in [x for x in emos.split("|") if x]:
        assert e in CANONICAL_LABELS["emotion"], f"{doc_id}: bad emotion {e}"
    for i_lbl in [x for x in issues.split("|") if x]:
        assert i_lbl in CANONICAL_LABELS["issue"], f"{doc_id}: bad issue {i_lbl}"


def main() -> int:
    with TEMPLATE_STUDIO_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    assert len(rows) == 50, f"Expected 50 rows, got {len(rows)}"
    assert set(PASS2_BLIND_LABELS.keys()) == {r["document_id"] for r in rows}, "Exact 50 document_id mismatch!"

    for r in rows:
        did = r["document_id"]
        entry = PASS2_BLIND_LABELS[did]
        validate_entry(did, entry)
        status, reason, sent, emos, issues, resp, sol, notes = entry
        r["review_status"] = status
        r["abstain_reason"] = reason
        r["sentiment"] = sent
        r["emotions"] = "|".join(sorted(x for x in emos.split("|") if x))
        r["issues"] = "|".join(sorted(x for x in issues.split("|") if x))
        r["response_type"] = resp
        r["is_solution"] = sol
        r["annotator_id"] = "ai_arctic_shift_full_context_pass2"
        r["annotator_notes"] = notes

    with OUT_AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote 50 blind full-Arctic-Shift-context Pass-2 labels -> {OUT_AUDIT_CSV}")

    # Now score Pass-2 (Full Arctic Shift Thread + Community Replies) vs Pass-1 (Narrow Title+Parent V2 Rubric)
    cod_p2 = _load_studio_csv(OUT_AUDIT_CSV)
    cod_p1 = _load_v2_pass1_reference(set(cod_p2.keys()))
    rep = compute_iaa_report(
        cod_p2,
        cod_p1,
        label_a="AI Pass-2 (Full Arctic Shift Thread + Community Replies, Blind)",
        label_b="V2 Rubric Pass-1 (Narrow Title+Parent Context)",
    )
    rep["is_human_iaa"] = False
    rep["human_iaa_gate_unlocked"] = False
    rep["provenance_notice"] = (
        "AI-assisted full-thread-context consistency audit using live Arctic Shift community replies "
        "(27/27 posts matched live; +136 extra comments fetched). NOT independent human Inter-Annotator "
        "Agreement; genuine human review/sign-off is still required before unlocking the Human IAA gate."
    )
    save_json(OUT_AUDIT_JSON, rep)

    if rep.get("disagreements"):
        with OUT_DISAGREE_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep["disagreements"][0].keys()))
            w.writeheader()
            w.writerows(rep["disagreements"])

    # Write transparent Markdown audit report
    lines = [
        "# Phase 6 — Full-Arctic-Shift-Context Blind Pass-2 Audit Report (NOT Human IAA)",
        "",
        "> [!WARNING]",
        "> **Provenance & Scientific Integrity Boundary:** This report compares **Blind AI Pass-2** (annotated with full local + live Arctic Shift post bodies, ancestor chains, and community direct replies across all 27 threads / +136 extra live comments) against **V2 Rubric Pass-1** (narrow `post_title + parent_text` context). It is an **AI consistency & context-sensitivity audit**, **NOT** independent human Inter-Annotator Agreement. `human_iaa_verified` remains `false` until a human reviews/adjudicates the pack.",
        "",
        f"**Coverage:** `{rep['reviewed_by_both']} / {rep['total_pack_items']}` items (`{rep['both_labeled_count']}` both labeled; `1` both abstained as `off_topic_noise`; review-status agreement `{rep['review_status_agreement_pct']}%`)  ",
        f"**Live Arctic Shift Thread Lookup:** `27 / 27` posts matched (`+136` extra live comments fetched; zero author names stored; PII redacted)  ",
        "**Human IAA Gate Unlocked:** `NO` *(requires human sign-off; `ml/artifacts/v5_experiments/active_learning/iaa_50_human_annotator_studio.csv` preserved strictly blank)*",
        "",
        "## Task-Level Agreement: Full-Arctic-Shift Pass-2 vs. Narrow Pass-1",
        "",
        "| Task | Paradigm | $n$ | Raw / Exact % | Cohen's $\\kappa$ | Krippendorff's $\\alpha$ | Reference Threshold | Pass-2 vs Pass-1 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for t in ("sentiment", "response", "solution"):
        o = rep["tasks"][t]
        lines.append(
            f"| **{t}** | Single-label | {o['n_compared']} | {o['raw_agreement_pct']}% | `{o['cohen_kappa']:.4f}` | `{o['krippendorff_alpha_nominal']:.4f}` | $\\ge {o['gate_threshold']}$ | {'ABOVE' if o['gate_passed'] else 'BELOW — inspect'} |"
        )
    for t in ("issue", "emotion"):
        o = rep["tasks"][t]
        lines.append(
            f"| **{t}** | Multi-label | {o['n_compared']} | {o['exact_set_agreement_pct']}% (Jaccard `{o['mean_jaccard']}`) | *(per-label)* | MASI `{o['krippendorff_alpha_masi']:.4f}` / Micro `{o['krippendorff_alpha_micro_binary']:.4f}` | $\\ge {o['gate_threshold']}$ | {'ABOVE' if o['gate_passed'] else 'BELOW — inspect'} |"
        )

    lines += [
        "",
        "## Per-Label Multi-Label Breakdown",
        "",
        "| Task | Label | Pass-2 Count | Pass-1 Count | Cohen's $\\kappa$ | Krippendorff's $\\alpha$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t in ("issue", "emotion"):
        for lbl, stats in rep["tasks"][t]["per_label"].items():
            lines.append(
                f"| `{t}` | `{lbl}` | {stats['count_a']} | {stats['count_b']} | `{stats['cohen_kappa']:.4f}` | `{stats['krippendorff_alpha']:.4f}` |"
            )

    lines += [
        "",
        f"## Disagreements Flagged for Human Adjudication (`{len(rep['disagreements'])}` / 50 items)",
        "",
        "Exported side-by-side with `adjudicated_*` columns to `ml/artifacts/v5_experiments/active_learning/iaa_ai_arctic_shift_disagreements.csv`.",
        "",
        "| Document ID | Disagreed Tasks | Pass-2 (Arctic Shift Context) | Pass-1 (Narrow Context) | Comment Excerpt |",
        "|---|---|---|---|---|",
    ]
    for d in rep["disagreements"]:
        p2_s = f"sent=`{d['a_sentiment']}`, emo=`{d['a_emotions']}`, iss=`{d['a_issues']}`, resp=`{d['a_response']}`, sol=`{d['a_solution']}`"
        p1_s = f"sent=`{d['b_sentiment']}`, emo=`{d['b_emotions']}`, iss=`{d['b_issues']}`, resp=`{d['b_response']}`, sol=`{d['b_solution']}`"
        excerpt = d["text"].replace("\n", " ").replace("|", "/")[:110]
        lines.append(f"| `{d['document_id']}` | `{d['disagreed_tasks']}` | {p2_s} | {p1_s} | {excerpt} |")

    OUT_AUDIT_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in rep.items() if k != "disagreements"}, indent=2))
    print(f"\nTotal items with >=1 task disagreement: {len(rep['disagreements'])} / 50")
    print(f"Disagreements CSV -> {OUT_DISAGREE_CSV}")
    print(f"Audit Report MD   -> {OUT_AUDIT_REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
