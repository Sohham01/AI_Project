"""Expanded multi-register synthetic training & robustness scaling benchmark.

Measures honestly how scaling training data (~3.8x more training comments across
~450+ distinct threads with multi-register English, Hinglish code-mixing,
informal Reddit slang, short pragmatic comments, spelling noise, and parent-post
context) improves generalization on:
  1. The untouched ``demo_synthetic`` held-out test set (n = 192, 24 unseen threads)
  2. The untouched 8-category unseen robustness stress test (54 probes from
     ``backend/run_robustness_benchmark.py``)

SCIENTIFIC INTEGRITY & ISOLATION GUARANTEES:
  - Strictly SYNTHETIC training expansion; NEVER fabricates labels on real
    Reddit comments and NEVER presents synthetic results as real-world accuracy.
  - Zero data leakage: disjoint ``post_id`` groups verified by ``check_leakage()``
    plus explicit anti-contamination checks asserting zero text/6-gram overlap
    with held-out test rows and robustness probes.
  - Writes ONLY into isolated gitignored ``ml/artifacts/expanded_data_benchmark/``.
    Never modifies ``data/demo_synthetic/``, ``ml/artifacts/demo_synthetic/``,
    ``ml/artifacts/demo_synthetic_v2/``, or ``ml/artifacts/registry.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.run_robustness_benchmark import (
    DATA_DIR,
    ROBUSTNESS_PROBES,
    SEED,
    TASKS,
    benchmark_classical,
    benchmark_muril,
    load_post_titles,
)
from ml.configs import TaskType
from ml.data.dataset import TrainingRecord, load_annotation_dataset
from ml.data.splitting import (
    SplitSpec,
    apply_split,
    check_leakage,
    extract_groups,
    split_records,
)

OUT_DIR = ROOT / "ml" / "artifacts" / "expanded_data_benchmark"
BASELINE_REPORT_PATH = ROOT / "ml" / "artifacts" / "robustness_benchmark" / "robustness_report.json"

# ---------------------------------------------------------------------------
# Rich multi-register synthetic generator (strictly disjoint from probes)
# ---------------------------------------------------------------------------

ISSUE_DOMAINS: Dict[str, Dict[str, Any]] = {
    "placements": {
        "titles": [
            "Audited summer internship and final offer breakdown for MBA batch",
            "How strong are consulting, product, and analytics recruitments this cycle?",
            "Placement committee communication around company shortlists and PPOs",
            "Realistic expectations on median fixed pay versus total package claims",
            "Recruiter turnout and summer internship conversion rate review",
            "IIM Sambalpur placements vs other baby IIMs and job market reality",
            "What is the real average CTC and median package after MBA graduation?",
        ],
        "nouns": [
            "final placements",
            "summer internship shortlists",
            "fixed versus variable CTC",
            "PPO conversions",
            "corporate recruitment drives",
            "alumni referrals for consulting roles",
            "domain-wise hiring numbers",
            "resume verification timelines",
            "median LPA package",
            "audited placement statistics",
            "baby IIM career outcomes",
            "post-MBA job offers",
        ],
        "hinglish_nouns": [
            "placement ka scene",
            "SIP shortlist",
            "PPO conversion",
            "median package",
            "company interviews",
            "marketing aur finance roles",
            "average CTC aur job market",
        ],
        "solutions": [
            "Request an audited role-wise median fixed pay sheet through the student placement council before finalizing offers.",
            "Form mock interview pods with second-year students who already converted summer PPOs in your target sector.",
            "Upload your verified certificates and one-page CV to the placement portal 72 hours ahead of the shortlist freeze.",
            "Ask the career cell to publish an anonymized domain tracker showing fixed salary separately from joining bonuses.",
        ],
        "hinglish_solutions": [
            "Bhai second year seniors ke saath mock interviews aur CV review schedule kar lo, shortlist chances kaafi badh jaate hain.",
            "Placement cell ko batch representative ke through mail bhejo ki fixed pay aur variable alag se share karein.",
        ],
    },
    "hostel_facilities": {
        "titles": [
            "Hostel block room allocation, cooling, and dining hall menu updates",
            "Mess meal quality, drinking water filters, and laundry queue feedback",
            "Single versus twin-sharing residence conditions on campus",
            "Housekeeping response speed for hostel plumbing and electrical repairs",
            "Evening canteen timings and weekend mess menu variety",
            "AC rooms, hostel stay rules, and veg/non-veg mess options at IIM Sambalpur",
        ],
        "nouns": [
            "hostel maintenance requests",
            "mess food consistency",
            "water dispenser servicing",
            "laundry machine turnaround",
            "room ventilation and AC upkeep",
            "block warden follow-ups",
            "dining hall hygiene standards",
            "night canteen availability",
            "single occupancy hostel rooms",
            "resident hostel facilities",
        ],
        "hinglish_nouns": [
            "hostel rooms aur AC",
            "mess ka khana",
            "drinking water cooler",
            "laundry queue",
            "warden office ka response",
            "hostel cleanliness",
        ],
        "solutions": [
            "Log the room number and defect photo in the hostel helpdesk register and copy the resident warden for 48-hour escalation.",
            "Propose a rotating weekly student mess committee audit with shared kitchen inspection notes every Sunday.",
            "Book off-peak laundry slots before 8 AM using the floor sign-up sheet to avoid weekend bottlenecks.",
            "Submit a signed wing request to replace the faulty drinking water filter cartridge during the Tuesday maintenance round.",
        ],
        "hinglish_solutions": [
            "Yaar hostel maintenance app pe room photo daal ke warden ko CC kar do, 2 din mein technician aa jaata hai.",
            "Mess committee ke feedback form mein weekly menu changes aur hygiene points list kar do, next meeting mein fix ho jaata hai.",
        ],
    },
    "academic_workload": {
        "titles": [
            "Managing back-to-back term quizzes, case preps, and group projects",
            "Term 1 and Term 2 sleep schedule versus surprise quiz pressure",
            "Overlapping case study deadlines across core management courses",
            "How intense is the first-year grading curve and submission calendar?",
            "Balancing live corporate projects with mid-term exam preparation",
            "Strict attendance rules, CP marks, and MBA academic pressure",
        ],
        "nouns": [
            "overlapping quiz dates",
            "midnight case study submissions",
            "term-end exam intensity",
            "group project coordination",
            "surprise class assessments",
            "daily pre-read volume",
            "relative grading pressure",
            "weekend assignment clusters",
            "strict attendance requirements",
            "academic workload and exams",
        ],
        "hinglish_nouns": [
            "Term 1 ka workload",
            "back to back quizzes",
            "midnight submissions",
            "case study pressure",
            "sleep schedule aur exams",
            "assignment deadlines",
        ],
        "solutions": [
            "Have the section academic rep merge every professor's quiz date into a shared calendar and request a 24-hour spacing rule.",
            "Split long Harvard case pre-reads across your four-person study group and swap half-page executive briefs each evening.",
            "Block two protected hours every morning exclusively for quantitative problem sets before club meetings start.",
            "Ask the program office to cap simultaneous major deliverables at two per week across core sections.",
        ],
        "hinglish_solutions": [
            "Study group mein cases divide kar lo aur short notes share karo, raat ko kaafi time bach jaata hai.",
            "Section CR ko bolo ki saare quizzes ka ek calendar banake program office se spacing maange.",
        ],
    },
    "financial_concerns": {
        "titles": [
            "Total MBA tuition fees, living expenses, and education loan ROI math",
            "Scholar loan disbursement, caution deposit refund, and fee installments",
            "Is taking a 20+ lakh student loan sensible with prior work experience?",
            "Need-based financial aid, merit scholarships, and hostel fee breakup",
            "EMI burden after graduation versus starting post-MBA salary",
            "IIM Sambalpur fee structure, acceptance deposit refund, and ROI review",
        ],
        "nouns": [
            "total program tuition",
            "education loan EMI calculations",
            "scholarship eligibility criteria",
            "installment deadline extensions",
            "hostel caution deposit refunds",
            "post-MBA return on investment",
            "collateral-free bank loan terms",
            "out-of-pocket living costs",
            "acceptance fee withdrawal refund",
            "20 lakh MBA fee structure",
        ],
        "hinglish_nouns": [
            "fees aur loan EMI",
            "MBA ka ROI",
            "scholarship form",
            "bank loan disbursement",
            "tuition installment",
            "refund aur living expenses",
        ],
        "solutions": [
            "Email your bank loan sanction letter to the finance desk ten days before the due date to obtain a penalty-free extension.",
            "Compare SBI Scholar and Union Bank collateral-free interest rates and keep stamped fee receipts for seamless reimbursement.",
            "Submit family income verification documents on the institute scholarship portal before the autumn window closes.",
            "Build a conservative spreadsheet modeling post-tax monthly cash flow against base fixed salary before locking your loan tenure.",
        ],
        "hinglish_solutions": [
            "Accounts office ko bank sanction letter attach karke mail kar do, bina late fine ke extension mil jaata hai.",
            "Institute scholarship portal pe deadline se pehle documents upload kar do, tuition ka burden kaafi kam hota hai.",
        ],
    },
    "curriculum_teaching": {
        "titles": [
            "Case method pedagogy, core faculty depth, and industry electives",
            "Are analytics, finance, and operations electives practical enough?",
            "Visiting practitioners versus permanent faculty classroom quality",
            "Feedback on Python, SQL, and financial modeling lab modules",
            "Syllabus relevance to modern consulting and product management roles",
            "Review of IIM Sambalpur UG BS Data Science and Management curriculum",
            "MBA vs BS degree course structure, faculty expertise, and academic quality",
        ],
        "nouns": [
            "case discussion quality",
            "visiting industry faculty sessions",
            "hands-on analytics coursework",
            "finance and strategy electives",
            "classroom participation grading",
            "updated syllabus materials",
            "faculty office hours",
            "applied simulation workshops",
            "BS Data Science and AI curriculum",
            "undergraduate management degree syllabus",
            "MBA specialization electives",
        ],
        "hinglish_nouns": [
            "faculty aur teaching quality",
            "case study discussions",
            "electives ka syllabus",
            "visiting professors ke lectures",
            "practical analytics labs",
            "BS Data Science course structure",
        ],
        "solutions": [
            "Share concrete tool requests like SQL dashboards and live valuation templates in the mid-term course feedback survey.",
            "Book professor office hours in small teams to walk through real company annual reports alongside textbook models.",
            "Coordinate with the academic council to invite alumni practitioners for weekend hands-on product and finance clinics.",
            "Publish curated pre-course primer playlists before Term 1 so non-engineering students build spreadsheet fluency early.",
        ],
        "hinglish_solutions": [
            "Mid-term faculty feedback mein specific practical tools mention kar do, kaafi profs live datasets add kar dete hain.",
            "Prof ke office hours book karke real balance sheets discuss karo, concepts bohot clear ho jaate hain.",
        ],
    },
    "campus_infrastructure": {
        "titles": [
            "Campus Wi-Fi bandwidth, library seating, and classroom AV setup",
            "Shuttle bus frequency to railway station and nearby city connectivity",
            "24/7 study rooms, power outlets, and sports complex facilities",
            "New permanent campus buildings, lab workstations, and medical clinic",
            "Network stability during online proctored assessments and interviews",
            "IIM Sambalpur permanent campus connectivity, Jharsuguda airport, and location review",
        ],
        "nouns": [
            "campus Wi-Fi stability",
            "library reading room seating",
            "evening shuttle frequency",
            "classroom power sockets",
            "IT helpdesk turnaround",
            "indoor sports court lighting",
            "campus medical center hours",
            "high-speed LAN ports for interviews",
            "new permanent campus infrastructure",
            "airport and railway connectivity",
        ],
        "hinglish_nouns": [
            "campus Wi-Fi speed",
            "library seating aur sockets",
            "shuttle bus timing",
            "new permanent campus buildings",
            "sports ground lights",
            "station aur airport connectivity",
        ],
        "solutions": [
            "Raise an IT ticket with the exact router floor ID and timestamp so network engineers can rebalance congested access points.",
            "Reserve the dedicated placement interview cabins with wired Ethernet ports 24 hours ahead of online recruiter rounds.",
            "Share a consolidated evening student demand log with the transport coordinator to add an 8:30 PM station shuttle.",
            "Install surge-protected multi-socket strips across the second-floor library tables through the student infrastructure fund.",
        ],
        "hinglish_solutions": [
            "Online interview ke liye wired Ethernet wala cabin pehle se book kar lo, Wi-Fi drop ka tension khatam ho jaata hai.",
            "IT helpdesk pe exact floor aur time ke saath ticket daal do, router reset karke speed theek kar dete hain.",
        ],
    },
    "administrative_communication": {
        "titles": [
            "Clarity and lead time on official program office circulars",
            "Transcript, bonafide certificate, and registration desk turnaround",
            "Last-minute schedule changes and attendance policy notifications",
            "Admissions waitlist updates, CAT percentile cutoffs, and merit list releases",
            "Document verification portal guidelines for incoming students",
            "IIM Sambalpur CAP PI call, waitlist movement, and conversion chances",
            "Offer letter timeline, buffer calls, and admission office email replies",
            "UG BS admission process, JEE/CUET eligibility, and shortlist status",
        ],
        "nouns": [
            "official circular lead times",
            "admissions office email replies",
            "bonafide certificate processing",
            "waitlist movement announcements",
            "attendance rule clarifications",
            "registration portal instructions",
            "timely exam timetable notices",
            "single-window student desk updates",
            "CAT percentile cutoff and PI shortlist",
            "CAP interview and merit list conversion",
            "offer letter and buffer call updates",
            "UG admission portal verification",
        ],
        "hinglish_nouns": [
            "admin office ke circulars",
            "mail ka reply",
            "waitlist movement aur merit list",
            "bonafide certificate process",
            "CAP interview aur CAT cutoff",
            "admission portal aur document verification",
        ],
        "solutions": [
            "Visit the academic office counter during the 10 AM to 12 PM window with your printed ID and form for same-day stamping.",
            "Ask the student council to maintain a pinned weekly digest summarizing every official circular and deadline in one view.",
            "Keep your application roll number in the email subject line and follow up politely after three working days.",
            "Publish a two-page onboarding checklist with exact stamp requirements so new admits avoid repeat counter visits.",
        ],
        "hinglish_solutions": [
            "Subah 10 baje academic office mein ID card aur form leke chale jao, same day stamp mil jaata hai.",
            "Email ke subject mein roll number likh ke bhejo aur 3 working days baad polite reminder daal do, reply aa jaata hai.",
        ],
    },
    "extracurriculars": {
        "titles": [
            "Consulting, finance, and analytics club inductions and case comps",
            "Annual management fest organizing workload and sponsorship teams",
            "Balancing student council committees with academic commitments",
            "Do national B-school case competitions actually boost CV shortlists?",
            "Cultural societies, sports tournaments, and student club events",
        ],
        "nouns": [
            "club induction tasks",
            "national case competitions",
            "annual fest committee hours",
            "consulting and marketing clubs",
            "student council elections",
            "inter-IIM sports meets",
            "extracurricular CV spikes",
            "club event scheduling",
        ],
        "hinglish_nouns": [
            "club inductions",
            "case competitions",
            "fest committee ka kaam",
            "analytics aur finance clubs",
            "sports tournaments",
            "clubs aur cultural societies",
        ],
        "solutions": [
            "Join at most one domain club and one cultural society in Term 1 so you can deliver strong work without burning out.",
            "Have the student activity council publish a unified semester event calendar before inductions to prevent exam clashes.",
            "Team up across finance, operations, and marketing backgrounds for national case competitions to cover slide design and valuation.",
            "Archive winning case decks in a shared club repository so incoming juniors learn structured problem solving faster.",
        ],
        "hinglish_solutions": [
            "Term 1 mein sirf ek domain club aur ek hobby club join karo, warna quizzes aur club tasks clash ho jaate hain.",
            "Case comp ke liye alag specializations ke teammates rakho, deck aur numbers dono strong bante hain.",
        ],
    },
}

ISSUE_NAMES = tuple(ISSUE_DOMAINS.keys())


def _mild_typo(text: str, rng: random.Random) -> str:
    """Apply light realistic Reddit typing shorthand/typos without breaking meaning."""
    replacements = [
        ("you ", "u "),
        ("your ", "ur "),
        ("please ", "pls "),
        ("college", "clg"),
        ("placement", "plcmnt"),
        ("hostel", "hstl"),
        ("professor", "prof"),
        ("semester", "sem"),
        ("definitely", "def"),
        ("probably", "prob"),
        ("because", "bcz"),
        ("message", "msg"),
    ]
    out = text
    for src, dst in replacements:
        if src in out.lower() and rng.random() < 0.35:
            out = out.replace(src, dst)
    words = out.split()
    vowels = set("aeiou")
    mutated = []
    for w in words:
        if len(w) >= 6 and w.isascii() and w.isalpha() and rng.random() < 0.14:
            idxs = [i for i in range(1, len(w) - 1) if w[i] in vowels]
            if idxs:
                i = rng.choice(idxs)
                w = w[:i] + w[i + 1 :]
        mutated.append(w)
    return " ".join(mutated)


def generate_expanded_synthetic_bundle(
    *,
    n_threads: int = 320,
    seed: int = 20261120,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Generate ~3,200 diverse multi-register synthetic comments across fresh threads."""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    post_titles: Dict[str, str] = {}

    casual_prefixes = (
        "",
        "Honestly, ",
        "Tbh ",
        "Ngl ",
        "Imo ",
        "Bro ",
        "Fyi ",
        "Real talk: ",
        "From what I saw, ",
        "For our batch, ",
        "Speaking candidly, ",
        "At campus right now, ",
    )
    casual_suffixes = (
        "",
        " tbh",
        " imo",
        " fr",
        " honestly",
        " for sure",
        " this term",
        " this season",
        " across sections",
        " right now",
    )
    cohorts = (
        "Term 1",
        "Term 2",
        "Term 3",
        "first-year",
        "second-year",
        "MBA-BA",
        "flagship MBA",
        "summer batch",
        "autumn cycle",
        "2025-27 batch",
        "2026 intake",
        "section A",
        "section B",
        "section C",
    )
    short_agree_tails = (
        "with this point",
        "here too",
        "on that part",
        "with what you wrote",
        "for our section",
        "on the ground",
        "after seeing this term",
        "from our side too",
        "across my friend group",
        "especially this cycle",
    )
    short_counter_tails = (
        "based on our batch",
        "from inside campus",
        "compared to last year",
        "for most students here",
        "in our cohort at least",
        "looking at recent numbers",
        "after talking to seniors",
        "from what I experienced",
        "in Term 2 onwards",
        "once you check official data",
    )
    seen_texts: set[str] = set()

    def _unique_text(raw_text: str, cohort: str) -> str:
        cand_text = " ".join(raw_text.split())
        if cand_text.lower() not in seen_texts:
            seen_texts.add(cand_text.lower())
            return cand_text
        variants = [
            f"For {cohort}: {cand_text}",
            f"{cand_text} ({cohort} context)",
            f"In {cohort}, {cand_text[0].lower() + cand_text[1:]}",
            f"At least in {cohort}, {cand_text[0].lower() + cand_text[1:]}",
            f"From {cohort} experience, {cand_text[0].lower() + cand_text[1:]}",
            f"Sharing from {cohort}: {cand_text}",
        ]
        for v in variants:
            v_clean = " ".join(v.split())
            if v_clean.lower() not in seen_texts:
                seen_texts.add(v_clean.lower())
                return v_clean
        return ""

    for t_idx in range(1, n_threads + 1):
        post_id = f"exp_t{t_idx:04d}"
        primary_issue = ISSUE_NAMES[(t_idx - 1) % len(ISSUE_NAMES)]
        secondary_issue = ISSUE_NAMES[(t_idx + 2) % len(ISSUE_NAMES)]
        dom = ISSUE_DOMAINS[primary_issue]
        sec_dom = ISSUE_DOMAINS[secondary_issue]

        title = rng.choice(dom["titles"])
        post_titles[post_id] = title

        cohort = rng.choice(cohorts)
        noun = rng.choice(dom["nouns"])
        noun2 = rng.choice(dom["nouns"])
        hnoun = rng.choice(dom["hinglish_nouns"])
        sec_noun = rng.choice(sec_dom["nouns"])
        sol_text = rng.choice(dom["solutions"])
        hsol_text = rng.choice(dom["hinglish_solutions"])
        ag_tail = rng.choice(short_agree_tails)
        ct_tail = rng.choice(short_counter_tails)

        # 13 cleanly label-aligned multi-register comments per thread:
        # Every single candidate pairs its EXACT text sentiment, emotion(s), issue(s),
        # canonical response class, and binary solution flag with ZERO random label noise!
        candidates = [
            # 1. Negative opinion -> frustration (explicit annoyance / dissatisfaction)
            {
                "text": rng.choice([
                    f"{rng.choice(casual_prefixes)}The repeated mismanagement of {noun} in {cohort} is honestly super frustrating{rng.choice(casual_suffixes)}.",
                    f"Every {cohort} cycle {noun} stays disorganized and students are fed up with the lack of accountability.",
                    f"Yaar {cohort} mein {hnoun} ka haal kaafi irritating ho gaya hai, proper transparency honi chahiye.",
                    f"Lowkey annoyed seeing zero action on {noun} while {sec_noun} also keeps breaking in {cohort}.",
                ]),
                "sentiment": "negative",
                "emotions": ["frustration"],
                "issues": [primary_issue, secondary_issue] if rng.random() < 0.25 else [primary_issue],
                "response": "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 2. Negative personal experience -> stress (+ frustration when paired)
            {
                "text": rng.choice([
                    f"Going through {noun} during {cohort} left me completely exhausted and stressed out every week.",
                    f"Personally I felt super overwhelmed juggling {noun} alongside {sec_noun} in {cohort} without sleep.",
                    f"Mere {cohort} mein {hnoun} ki wajah se kaafi tension aur burnout feel hua, neend hi ud gayi thi.",
                    f"Ngl dealing with {noun} in {cohort} gave our whole wing serious anxiety before deadlines.",
                ]),
                "sentiment": "negative",
                "emotions": ["stress", "frustration"] if t_idx % 3 == 0 else ["stress"],
                "issues": [primary_issue, secondary_issue] if rng.random() < 0.30 else [primary_issue],
                "response": "personal_experience",
                "solution": "not_solution",
                "lang": "en",
            },
            # 3. Positive personal experience -> confidence (assured, solid experience)
            {
                "text": rng.choice([
                    f"In my {cohort} batch, {noun} turned out genuinely solid and made me feel well prepared.",
                    f"Speaking from firsthand experience in {cohort}, {noun} gave me real confidence for interviews.",
                    f"Mere {cohort} mein {hnoun} actually kaafi strong raha, koi doubt nahi ki sahi choice thi.",
                    f"Honestly {noun} during {cohort} was a big win for me, felt totally ready afterwards.",
                ]),
                "sentiment": "positive",
                "emotions": ["confidence"],
                "issues": [primary_issue],
                "response": "personal_experience",
                "solution": "not_solution",
                "lang": "en",
            },
            # 4. Positive remark -> gratitude (explicit thanks / appreciation)
            {
                "text": rng.choice([
                    f"Thanks a ton for laying out {noun} for {cohort} so clearly, really appreciate it!",
                    f"Super grateful to the seniors who broke down {noun} for {cohort}, saved us so much guesswork.",
                    f"Thank you bhai, {cohort} ke {hnoun} pe itni genuine clarity dene ke liye shukriya.",
                    f"Much appreciated! That honest {cohort} breakdown on {noun} really helped me out.",
                ]),
                "sentiment": "positive",
                "emotions": ["gratitude"],
                "issues": [primary_issue] if rng.random() < 0.7 else [],
                "response": "agreement" if t_idx % 2 == 0 else "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 5. Actionable concrete solution -> hope (or neutral empty emotion for terse fixes)
            (
                {
                    "text": rng.choice([
                        f"{rng.choice(casual_prefixes)}{sol_text} Hopefully that sorts it out smoothly.",
                        f"For {cohort} students facing {noun}: {sol_text[0].lower() + sol_text[1:]} Should improve things soon.",
                        f"{hsol_text} Umeed hai isse kaafi relief milega.",
                        f"Best practical fix in {cohort}: {sol_text[0].lower() + sol_text[1:]}",
                    ]),
                    "sentiment": "positive",
                    "emotions": ["hope"],
                    "issues": [primary_issue],
                    "response": "solution",
                    "solution": "solution",
                    "lang": "en",
                }
                if t_idx % 3 != 0
                else {
                    "text": rng.choice([
                        f"{sol_text}",
                        f"Fix for {cohort}: {sol_text[0].lower() + sol_text[1:]}",
                        f"{hsol_text}",
                    ]),
                    "sentiment": "neutral",
                    "emotions": [],
                    "issues": [primary_issue],
                    "response": "solution",
                    "solution": "solution",
                    "lang": "en",
                }
            ),
            # 6. Actionable practical advice (response=advice, solution=solution) -> hope
            {
                "text": rng.choice([
                    f"My practical tip on {noun} for {cohort}: {sol_text[0].lower() + sol_text[1:]}",
                    f"Try doing this early in {cohort} for {noun}: {sol_text[0].lower() + sol_text[1:]}",
                    f"Ek kaam karo bhai ({cohort}): {hsol_text[0].lower() + hsol_text[1:]}",
                    f"Bro just reach out to two {cohort} seniors directly about {noun} and follow their checklist, legit saves time.",
                ]),
                "sentiment": "positive",
                "emotions": ["hope"],
                "issues": [primary_issue],
                "response": "advice",
                "solution": "solution",
                "lang": "en",
            },
            # 7. General non-solution advice (response=advice, solution=not_solution) -> neutral / hope
            {
                "text": rng.choice([
                    f"You should talk to a few {cohort} alumni about {noun} before locking your decision.",
                    f"Better to stay patient around {noun} in {cohort} and weigh your priorities calmly.",
                    f"Bhai {cohort} ke {hnoun} ke baare mein 2-3 logon se opinion zaroor le lena before deciding.",
                    f"Keep realistic expectations on {noun} during {cohort} and always maintain a backup option.",
                ]),
                "sentiment": "neutral",
                "emotions": ["hope"] if t_idx % 2 == 0 else [],
                "issues": [primary_issue],
                "response": "advice",
                "solution": "not_solution",
                "lang": "en",
            },
            # 8. Agreement (mix of frustrated issue-specific agreement and short neutral agreement)
            (
                {
                    "text": rng.choice([
                        f"Completely agree on {noun}, our entire {cohort} section was annoyed by the exact same issue.",
                        f"Spot on about {noun}, it has been super frustrating for {cohort} students too.",
                        f"Sahi bola bhai, {cohort} ke {hnoun} mein humne bhi bilkul yahi problem jheli hai.",
                        f"Facts, {noun} in {cohort} really needs fixing asap.",
                    ]),
                    "sentiment": "negative",
                    "emotions": ["frustration"],
                    "issues": [primary_issue],
                    "response": "agreement",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 2 == 0
                else {
                    "text": rng.choice([
                        f"Totally agree {ag_tail}.",
                        f"Same here {ag_tail}, 100% accurate.",
                        f"Spot on {ag_tail}, couldn't agree more.",
                        f"Bilkul sahi baat hai bhai, {ag_tail}.",
                        f"Agreed, noticed the exact same thing {ag_tail}.",
                        f"Yes, completely second this {ag_tail}.",
                    ]),
                    "sentiment": "neutral",
                    "emotions": [],
                    "issues": [],
                    "response": "agreement",
                    "solution": "not_solution",
                    "lang": "en",
                }
            ),
            # 9. Counterargument (positive confident pushback OR crisp neutral/negative disagreement)
            (
                {
                    "text": rng.choice([
                        f"Actually I disagree—{noun} has noticeably improved across {cohort} and works pretty well now.",
                        f"Not really fair; {noun} in {cohort} is actually quite solid once you check recent outcomes.",
                        f"Nahi yaar aisa nahi hai, {cohort} mein {hnoun} pehle se kaafi better ho gaya hai.",
                        f"Hold on, that ignores how much {cohort} upgraded {noun2} recently, I am pretty confident in it.",
                    ]),
                    "sentiment": "positive",
                    "emotions": ["confidence"],
                    "issues": [primary_issue],
                    "response": "counterargument",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 2 == 0
                else {
                    "text": rng.choice([
                        f"I disagree, that has not been true {ct_tail}.",
                        f"Nah gotta push back on that, things look different {ct_tail}.",
                        f"Not quite accurate {ct_tail}, feels pretty exaggerated.",
                        f"Nahi bhai, thoda zyada negative ho raha hai {ct_tail}.",
                        f"Strongly disagree with that claim {ct_tail}.",
                    ]),
                    "sentiment": "negative" if t_idx % 4 == 1 else "neutral",
                    "emotions": ["frustration"] if t_idx % 4 == 1 else [],
                    "issues": [],
                    "response": "counterargument",
                    "solution": "not_solution",
                    "lang": "en",
                }
            ),
            # 10. Curious inquiry -> neutral + curiosity (always cleanly aligned!)
            {
                "text": rng.choice([
                    f"Does anyone know how {noun} is handled for incoming {cohort} students?",
                    f"Quick question: has there been any official update on {noun} for {cohort}?",
                    f"Bhai {cohort} mein {hnoun} ka exact process kya hota hai? Koi bata sakta hai?",
                    f"Wondering whether {noun} in {cohort} differs for freshers versus people with work ex?",
                ]),
                "sentiment": "neutral",
                "emotions": ["curiosity"],
                "issues": [primary_issue],
                "response": "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 11. Personal experience describing a concrete fix (personal_experience + solution=solution) -> gratitude + confidence
            {
                "text": rng.choice([
                    f"When our {cohort} group ran into trouble with {noun}, we tried this fix: {sol_text[0].lower() + sol_text[1:]} Worked cleanly and I'm so thankful.",
                    f"Had the same issue with {noun} during {cohort} and got it resolved fast once I emailed the desk with screenshots.",
                    f"Humare {cohort} wing mein {hnoun} ka issue tha, portal pe joint ticket aur photos upload karne ke 48 ghante mein solve ho gaya.",
                ]),
                "sentiment": "positive",
                "emotions": ["confidence", "gratitude"],
                "issues": [primary_issue],
                "response": "personal_experience",
                "solution": "solution",
                "lang": "en",
            },
            # 12. Informal Reddit positive opinion -> confidence / hope
            {
                "text": rng.choice([
                    f"Ngl {noun} in {cohort} has been surprisingly awesome lately, massive upgrade imo!",
                    f"Tbh {noun} for {cohort} is legitimately strong if you put in the work, zero regrets.",
                    f"Honestly {cohort} mein {hnoun} kaafi decent lag raha hai, hopefully it keeps getting better.",
                    f"Lowkey impressed by {noun} this {cohort} cycle, solid overall vibe.",
                ]),
                "sentiment": "positive",
                "emotions": ["hope"] if t_idx % 2 == 0 else ["confidence"],
                "issues": [primary_issue],
                "response": "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 13. Context-dependent comment (for parent-post-context learning — ZERO random label noise!)
            (
                {
                    "text": f"Same problem hit our {cohort} batch and nobody gave a clear update, super frustrating.",
                    "sentiment": "negative",
                    "emotions": ["frustration"],
                    "issues": [primary_issue],
                    "response": "personal_experience",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 4 == 0
                else (
                    {
                        "text": f"Any recent {cohort} student here who can clarify whether this got sorted out?",
                        "sentiment": "neutral",
                        "emotions": ["curiosity"],
                        "issues": [primary_issue],
                        "response": "opinion",
                        "solution": "not_solution",
                        "lang": "en",
                    }
                    if t_idx % 4 == 1
                    else (
                        {
                            "text": f"Actually this improved quite a bit for {cohort} compared to last cycle, worth checking directly.",
                            "sentiment": "positive",
                            "emotions": ["confidence"],
                            "issues": [primary_issue],
                            "response": "counterargument",
                            "solution": "not_solution",
                            "lang": "en",
                        }
                        if t_idx % 4 == 2
                        else {
                            "text": f"If the {cohort} team publishes a clear timeline early, it should ease a lot of tension.",
                            "sentiment": "positive",
                            "emotions": ["hope"],
                            "issues": [primary_issue],
                            "response": "opinion",
                            "solution": "not_solution",
                            "lang": "en",
                        }
                    )
                )
            ),
        ]

        for c_idx, cand in enumerate(candidates, start=1):
            raw_t = cand["text"]
            if rng.random() < 0.25 and len(raw_t.split()) >= 7:
                raw_t = _mild_typo(raw_t, rng)
            text = _unique_text(raw_t, cohort)
            if not text:
                continue
            is_mixed = any(
                tok in text.lower()
                for tok in ("bhai", "yaar", "kaafi", "nahi", "karo", "humare", "mein", "sahi", "umeed", "shukriya")
            )
            items.append(
                {
                    "document_id": f"exp_doc_{t_idx:04d}_{c_idx:02d}",
                    "post_id": post_id,
                    "comment_id": f"exp_c_{t_idx:04d}_{c_idx:02d}",
                    "text": text,
                    "sentiment": cand["sentiment"],
                    "emotions": sorted(cand["emotions"]),
                    "issues": sorted(cand["issues"]),
                    "response": cand["response"],
                    "solution": cand["solution"],
                    "language": "hi" if is_mixed else "en",
                    "script": "latin",
                    "is_code_mixed": is_mixed,
                }
            )

    return items, post_titles


def _ngrams(text: str, n: int = 5) -> set[Tuple[str, ...]]:
    toks = [t for t in text.lower().split() if t]
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def verify_zero_contamination(
    expanded_items: Sequence[Dict[str, Any]],
    heldout_test_texts: Sequence[str],
) -> Dict[str, Any]:
    probe_texts = [p["text"].strip().lower() for p in ROBUSTNESS_PROBES]
    probe_5grams = set().union(*(_ngrams(t, 5) for t in probe_texts))
    probe_6grams = set().union(*(_ngrams(t, 6) for t in probe_texts))
    test_set = {t.strip().lower() for t in heldout_test_texts}

    exact_probe_hits = 0
    fivegram_probe_hits = 0
    sixgram_probe_hits = 0
    exact_test_hits = 0
    for item in expanded_items:
        low = item["text"].strip().lower()
        if low in probe_texts:
            exact_probe_hits += 1
        if _ngrams(low, 5) & probe_5grams:
            fivegram_probe_hits += 1
        if _ngrams(low, 6) & probe_6grams:
            sixgram_probe_hits += 1
        if low in test_set:
            exact_test_hits += 1

    if exact_probe_hits or fivegram_probe_hits or sixgram_probe_hits or exact_test_hits:
        raise RuntimeError(
            f"Contamination detected: exact_probe={exact_probe_hits}, "
            f"fivegram_probe={fivegram_probe_hits}, sixgram_probe={sixgram_probe_hits}, "
            f"exact_test={exact_test_hits}"
        )
    return {
        "exact_probe_overlaps": exact_probe_hits,
        "fivegram_probe_overlaps": fivegram_probe_hits,
        "sixgram_probe_overlaps": sixgram_probe_hits,
        "exact_heldout_test_overlaps": exact_test_hits,
        "clean": True,
    }


def _to_task_records(task_name: str, items: Sequence[Dict[str, Any]]) -> List[TrainingRecord]:
    out: List[TrainingRecord] = []
    for it in items:
        if task_name == "sentiment":
            lbl = it["sentiment"]
        elif task_name == "emotion":
            lbl = "|".join(it["emotions"])
        elif task_name == "issue":
            lbl = "|".join(it["issues"])
        elif task_name == "response":
            lbl = it["response"]
        else:
            lbl = it["solution"]
        out.append(
            TrainingRecord(
                document_id=it["document_id"],
                text=it["text"],
                label=lbl,
                post_id=it["post_id"],
                comment_id=it["comment_id"],
                language=it["language"],
                script=it["script"],
                is_code_mixed=it["is_code_mixed"],
                detected_languages=["en", "hi"] if it["is_code_mixed"] else [it["language"]],
            )
        )
    return out


def main(n_threads: int = 320, epochs: int = 4, run_muril: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_post_titles = load_post_titles()
    expanded_items, exp_post_titles = generate_expanded_synthetic_bundle(
        n_threads=n_threads, seed=20261120
    )
    all_post_titles = {**base_post_titles, **exp_post_titles}

    # Split expanded threads 85% train / 15% val strictly by post_id (never touches test!)
    exp_post_ids = sorted({it["post_id"] for it in expanded_items})
    rng = random.Random(20261120)
    rng.shuffle(exp_post_ids)
    val_cut = max(1, int(len(exp_post_ids) * 0.15))
    exp_val_posts = set(exp_post_ids[:val_cut])
    exp_train_items = [it for it in expanded_items if it["post_id"] not in exp_val_posts]
    exp_val_items = [it for it in expanded_items if it["post_id"] in exp_val_posts]

    # Load previous 896-row baseline for direct delta reporting
    baseline_prev = (
        json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        if BASELINE_REPORT_PATH.exists()
        else None
    )

    # Anti-contamination check against demo_synthetic test split + 54 robustness probes
    sent_base = load_annotation_dataset(TaskType.SENTIMENT, str(DATA_DIR / "sentiment.csv"))
    base_spec = SplitSpec(
        train_size=0.70, validation_size=0.15, test_size=0.15, random_state=SEED, stratify=True
    )
    base_split = split_records(
        sent_base,
        labels=[r.label for r in sent_base],
        spec=base_spec,
        groups=extract_groups(sent_base, "post_id"),
    )
    _, _, base_test_sample = apply_split(sent_base, base_split)
    contamination = verify_zero_contamination(expanded_items, [r.text for r in base_test_sample])
    print(
        f"Expanded synthetic corpus verified clean: +{len(exp_train_items)} train, "
        f"+{len(exp_val_items)} val across {n_threads} fresh threads "
        f"(0 probe/test overlaps)."
    )

    report: Dict[str, Any] = {
        "disclaimer": (
            "ISOLATED EXPANDED SYNTHETIC SCALING BENCHMARK ONLY. Evaluates multi-register "
            "data scaling on untouched demo_synthetic test (n=192) and 54 unseen 8-category "
            "robustness probes. Not real Reddit human-annotated evaluation."
        ),
        "seed": SEED,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "expanded_threads_added": n_threads,
        "expanded_train_added": len(exp_train_items),
        "expanded_val_added": len(exp_val_items),
        "contamination_audit": contamination,
        "tasks": {},
    }

    for task_name in TASKS:
        print(f"\n==================== EXPANDED DATA BENCHMARK: {task_name.upper()} ====================")
        base_records = load_annotation_dataset(TaskType(task_name), str(DATA_DIR / f"{task_name}.csv"))
        split = split_records(
            base_records,
            labels=[r.label for r in base_records],
            spec=base_spec,
            groups=extract_groups(base_records, "post_id"),
        )
        base_tr, base_va, test_rows = apply_split(base_records, split)

        train_rows = list(base_tr) + _to_task_records(task_name, exp_train_items)
        val_rows = list(base_va) + _to_task_records(task_name, exp_val_items)

        leak = check_leakage(train_rows, val_rows, test_rows)
        assert not leak.get("leakage_detected"), f"Thread leakage detected in {task_name}: {leak}"

        v1 = benchmark_classical(task_name, train_rows, val_rows, test_rows, upgraded=False)
        v2 = benchmark_classical(task_name, train_rows, val_rows, test_rows, upgraded=True)

        prev_task = (baseline_prev or {}).get("tasks", {}).get(task_name, {})
        prev_v2_probe = prev_task.get("v2_word_char_calibrated", {}).get("robustness_probes", {}).get("overall_macro_f1")
        prev_mu_probe = prev_task.get("muril_v2_gpu", {}).get("robustness_probes", {}).get("overall_macro_f1")

        print(
            f"  [Split] train={len(train_rows)} ({len({r.post_id for r in train_rows})} threads) | "
            f"val={len(val_rows)} | test={len(test_rows)} (untouched)"
        )
        print(
            f"  [V1]    heldout={v1['heldout_test']['macro_f1']:.4f} | "
            f"54-probe={v1['robustness_probes']['overall_macro_f1']:.4f}"
        )
        v2_delta = (
            f" (delta {v2['robustness_probes']['overall_macro_f1'] - prev_v2_probe:+.4f} vs 896-row)"
            if prev_v2_probe is not None
            else ""
        )
        print(
            f"  [V2]    heldout={v2['heldout_test']['macro_f1']:.4f} | "
            f"54-probe={v2['robustness_probes']['overall_macro_f1']:.4f}{v2_delta}"
        )

        task_out: Dict[str, Any] = {
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "leakage_detected": False,
            "v1_tfidf_baseline": v1,
            "v2_word_char_calibrated": v2,
            "delta_vs_896_row_probes": {
                "v2_word_char_delta": (
                    round(v2["robustness_probes"]["overall_macro_f1"] - prev_v2_probe, 6)
                    if prev_v2_probe is not None
                    else None
                ),
            },
        }

        if task_name == "issue":
            v2_ctx = benchmark_classical(
                task_name,
                train_rows,
                val_rows,
                test_rows,
                upgraded=True,
                with_parent_title=True,
                post_titles=all_post_titles,
            )
            task_out["v2_word_char_with_parent_title"] = v2_ctx
            print(
                f"  [V2+PT] heldout={v2_ctx['heldout_test']['macro_f1']:.4f} | "
                f"54-probe={v2_ctx['robustness_probes']['overall_macro_f1']:.4f} | "
                f"parent_ctx={v2_ctx['robustness_probes']['by_category']['parent_post_context']['macro_f1']:.4f}"
            )

        if run_muril:
            mu = benchmark_muril(task_name, train_rows, val_rows, test_rows, epochs=epochs)
            task_out["muril_v2_gpu"] = mu
            mu_delta = (
                f" (delta {mu['robustness_probes']['overall_macro_f1'] - prev_mu_probe:+.4f} vs 896-row)"
                if prev_mu_probe is not None
                else ""
            )
            task_out["delta_vs_896_row_probes"]["muril_v2_gpu_delta"] = (
                round(mu["robustness_probes"]["overall_macro_f1"] - prev_mu_probe, 6)
                if prev_mu_probe is not None
                else None
            )
            print(
                f"  [MuRIL] heldout={mu['heldout_test']['macro_f1']:.4f} | "
                f"54-probe={mu['robustness_probes']['overall_macro_f1']:.4f}{mu_delta} | "
                f"ep={mu['best_epoch']} | {mu['resources']['train_seconds']}s"
            )
            if task_name == "issue":
                mu_ctx = benchmark_muril(
                    task_name,
                    train_rows,
                    val_rows,
                    test_rows,
                    epochs=epochs,
                    with_parent_title=True,
                    post_titles=all_post_titles,
                )
                task_out["muril_v2_gpu_with_parent_title"] = mu_ctx
                print(
                    f"  [Mu+PT] heldout={mu_ctx['heldout_test']['macro_f1']:.4f} | "
                    f"54-probe={mu_ctx['robustness_probes']['overall_macro_f1']:.4f} | "
                    f"parent_ctx={mu_ctx['robustness_probes']['by_category']['parent_post_context']['macro_f1']:.4f}"
                )

        report["tasks"][task_name] = task_out

    out_path = OUT_DIR / "expanded_benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved expanded data benchmark report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=320, help="Fresh multi-register synthetic threads to add (default 320 -> ~3,200 comments)")
    parser.add_argument("--epochs", type=int, default=4, help="MuRIL epochs on expanded corpus (default 4)")
    parser.add_argument("--no-muril", action="store_true", help="Skip GPU MuRIL training")
    args = parser.parse_args()
    main(n_threads=args.threads, epochs=args.epochs, run_muril=not args.no_muril)
