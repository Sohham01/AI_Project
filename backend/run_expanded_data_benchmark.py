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
        ],
        "hinglish_nouns": [
            "placement ka scene",
            "SIP shortlist",
            "PPO conversion",
            "median package",
            "company interviews",
            "marketing aur finance roles",
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
        ],
        "hinglish_nouns": [
            "fees aur loan EMI",
            "MBA ka ROI",
            "scholarship form",
            "bank loan disbursement",
            "tuition installment",
            "living expenses",
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
        ],
        "hinglish_nouns": [
            "faculty aur teaching quality",
            "case study discussions",
            "electives ka syllabus",
            "visiting professors ke lectures",
            "practical analytics labs",
            "class participation",
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
        ],
        "hinglish_nouns": [
            "campus Wi-Fi speed",
            "library seating aur sockets",
            "shuttle bus timing",
            "new campus buildings",
            "sports ground lights",
            "station tak connectivity",
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
            "Admissions waitlist updates and official email responsiveness",
            "Document verification portal guidelines for incoming students",
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
        ],
        "hinglish_nouns": [
            "admin office ke circulars",
            "mail ka reply",
            "waitlist updates",
            "bonafide certificate process",
            "last minute notices",
            "document verification portal",
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
            "Cultural societies, sports tournaments, and peer networking events",
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
            "clubs aur studies ka balance",
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

        # 10 richly varied comments per thread covering all 6 responses, 3 sentiments,
        # all 6 emotions + empty, single/multi/empty issues, solution binary, and 6 registers
        candidates = [
            # 1. Negative opinion + frustration/stress
            {
                "text": rng.choice([
                    f"{rng.choice(casual_prefixes)}The recurring issues with {noun} make {cohort} planning stressful{rng.choice(casual_suffixes)}.",
                    f"Every {cohort} cycle {noun} feels poorly organized and leaves students frustrated.",
                    f"Yaar {cohort} mein {hnoun} kaafi exhausting ho gaya hai, proper transparency honi chahiye.",
                    f"Honestly seeing zero progress on {noun} while {sec_noun} also piles up in {cohort} is draining.",
                ]),
                "sentiment": "negative",
                "emotions": rng.choice([["frustration"], ["stress"], ["frustration", "stress"]]),
                "issues": [primary_issue, secondary_issue] if rng.random() < 0.25 else [primary_issue],
                "response": "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 2. Positive personal experience + gratitude/confidence
            {
                "text": rng.choice([
                    f"In my {cohort} batch, {noun} turned out much smoother than expected once we settled into campus.",
                    f"Speaking from personal experience during {cohort}, {noun} genuinely helped me build confidence.",
                    f"Mere {cohort} mein {hnoun} actually kaafi positive experience raha, seniors ne bohot support kiya.",
                    f"When I navigated {noun} in {cohort}, the peer support made a huge difference and I am really grateful.",
                ]),
                "sentiment": "positive",
                "emotions": rng.choice([["confidence"], ["gratitude"], ["confidence", "gratitude"]]),
                "issues": [primary_issue],
                "response": "personal_experience",
                "solution": "not_solution",
                "lang": "en",
            },
            # 3. Actionable concrete solution (response=solution, is_solution=solution)
            {
                "text": rng.choice([
                    f"{rng.choice(casual_prefixes)}{sol_text}",
                    f"For {cohort} students dealing with {noun}: {sol_text[0].lower() + sol_text[1:]}",
                    f"{hsol_text}",
                    f"Best workaround we saw in {cohort}: {sol_text[0].lower() + sol_text[1:]}",
                ]),
                "sentiment": rng.choice(["positive", "neutral"]),
                "emotions": rng.choice([["hope"], ["confidence"], ["hope", "confidence"]]),
                "issues": [primary_issue],
                "response": "solution",
                "solution": "solution",
                "lang": "en",
            },
            # 4. General advice (some actionable solution, some general non-solution guidance)
            (
                {
                    "text": rng.choice([
                        f"You should speak with two recent {cohort} alumni about {noun} before making your final call.",
                        f"Better to stay flexible on {noun} during {cohort} and keep a backup plan ready.",
                        f"Bhai {cohort} ke {hnoun} ke baare mein seniors se honest feedback le lena, kaafi help milegi.",
                        f"Consider tracking your priorities around {noun} early in {cohort} so nothing catches you off guard.",
                    ]),
                    "sentiment": rng.choice(["neutral", "positive"]),
                    "emotions": rng.choice([["hope"], ["confidence"], []]),
                    "issues": [primary_issue],
                    "response": "advice",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 2 == 0
                else {
                    "text": rng.choice([
                        f"Try writing directly to the {cohort} coordinator about {noun}: {sol_text[0].lower() + sol_text[1:]}",
                        f"My practical tip on {noun} for {cohort}: {sol_text[0].lower() + sol_text[1:]}",
                        f"Ek kaam karo bhai ({cohort}): {hsol_text[0].lower() + hsol_text[1:]}",
                    ]),
                    "sentiment": "positive",
                    "emotions": rng.choice([["hope"], ["hope", "gratitude"]]),
                    "issues": [primary_issue],
                    "response": "advice",
                    "solution": "solution",
                    "lang": "en",
                }
            ),
            # 5. Agreement (mix of issue-explicit and short context-only replies!)
            (
                {
                    "text": rng.choice([
                        f"Completely agree on {noun}, our entire {cohort} section noticed the exact same pattern.",
                        f"Spot on about {noun}, especially when {cohort} deadlines hit together.",
                        f"Sahi bola bhai, {cohort} ke {hnoun} mein humne bhi bilkul yahi face kiya.",
                        f"Seconding this—{noun} in {cohort} definitely needs tighter execution.",
                    ]),
                    "sentiment": rng.choice(["negative", "neutral"]),
                    "emotions": rng.choice([["frustration"], ["stress"], []]),
                    "issues": [primary_issue],
                    "response": "agreement",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 3 != 0
                else {
                    "text": rng.choice([
                        f"Totally agree {ag_tail}.",
                        f"Same here {ag_tail}, 100% true.",
                        f"Spot on {ag_tail}, couldn't agree more.",
                        f"Bilkul sahi baat hai bhai, {ag_tail}.",
                        f"Facts, faced the exact same thing {ag_tail}.",
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
            # 6. Counterargument (mix of issue-specific and short/general!)
            (
                {
                    "text": rng.choice([
                        f"Actually I disagree—{noun} has noticeably improved across {cohort}.",
                        f"Not entirely fair; {noun} works reasonably well in {cohort} when you follow the schedule.",
                        f"Nahi yaar aisa nahi hai, {cohort} mein {hnoun} kaafi better ho gaya hai.",
                        f"Hold on, that overlooks how much {cohort} upgraded {noun2} recently.",
                    ]),
                    "sentiment": rng.choice(["neutral", "positive"]),
                    "emotions": rng.choice([["confidence"], []]),
                    "issues": [primary_issue],
                    "response": "counterargument",
                    "solution": "not_solution",
                    "lang": "en",
                }
                if t_idx % 3 != 0
                else {
                    "text": rng.choice([
                        f"I disagree, that has not been true {ct_tail}.",
                        f"Not really, things look different {ct_tail}.",
                        f"Hard disagree on that take {ct_tail}.",
                        f"Nahi bhai, thoda exaggerate ho raha hai {ct_tail}.",
                        f"That feels overstated {ct_tail}.",
                    ]),
                    "sentiment": rng.choice(["neutral", "negative"]),
                    "emotions": rng.choice([[], ["frustration"]]),
                    "issues": [],
                    "response": "counterargument",
                    "solution": "not_solution",
                    "lang": "en",
                }
            ),
            # 7. Curious question / neutral inquiry (curiosity!)
            {
                "text": rng.choice([
                    f"Does anyone know how {noun} is handled for incoming {cohort} students?",
                    f"Quick question: has there been any official update on {noun} for {cohort}?",
                    f"Bhai {cohort} mein {hnoun} ka exact process kya hota hai? Koi bata sakta hai?",
                    f"Curious whether {noun} in {cohort} differs for freshers versus people with work ex.",
                ]),
                "sentiment": "neutral",
                "emotions": ["curiosity"],
                "issues": [primary_issue],
                "response": "opinion",
                "solution": "not_solution",
                "lang": "en",
            },
            # 8. Personal experience describing a concrete fix (personal_experience + solution=solution!)
            {
                "text": rng.choice([
                    f"When our {cohort} group hit trouble with {noun}, we tried this fix: {sol_text[0].lower() + sol_text[1:]} Worked cleanly.",
                    f"Had the same headache with {noun} during {cohort} and resolved it fast once I emailed the desk with screenshots.",
                    f"Humare {cohort} wing mein {hnoun} ka issue tha, portal pe joint ticket aur photos upload karne ke 48 ghante mein solve ho gaya.",
                ]),
                "sentiment": "positive",
                "emotions": rng.choice([["gratitude", "confidence"], ["gratitude"], ["confidence"]]),
                "issues": [primary_issue],
                "response": "personal_experience",
                "solution": "solution",
                "lang": "en",
            },
            # 9. Short gratitude / hopeful positive remark
            {
                "text": rng.choice([
                    f"Thanks a lot for explaining {noun} for {cohort} so clearly, really reassuring!",
                    f"Super helpful breakdown on {noun} in {cohort}, appreciate you sharing this.",
                    f"Thank you bhai, {cohort} ke {hnoun} pe kaafi clarity mil gayi.",
                    f"Really appreciate the honest {cohort} update, thanks!",
                    f"Thanks for sharing this {cohort} detail, feeling much more hopeful now.",
                ]),
                "sentiment": "positive",
                "emotions": rng.choice([["gratitude"], ["gratitude", "hope"]]),
                "issues": [primary_issue] if rng.random() < 0.6 else [],
                "response": rng.choice(["agreement", "opinion"]),
                "solution": "not_solution",
                "lang": "en",
            },
            # 10. Context-dependent comment (for parent-post-context learning!)
            {
                "text": rng.choice([
                    f"Same issue happened to our {cohort} group and nobody gave a clear timeline.",
                    f"If the {cohort} committee handles this early, half the panic disappears.",
                    f"Any recent {cohort} student who can verify whether this actually got fixed?",
                    f"Honestly this improved a lot for {cohort} compared to before, worth checking directly.",
                ]),
                "sentiment": rng.choice(["negative", "neutral", "positive"]),
                "emotions": rng.choice([["frustration"], ["curiosity"], ["hope"], []]),
                "issues": [primary_issue] if rng.random() < 0.7 else [],
                "response": rng.choice(["personal_experience", "opinion", "counterargument"]),
                "solution": "not_solution",
                "lang": "en",
            },
        ]

        for c_idx, cand in enumerate(candidates, start=1):
            raw_t = cand["text"]
            if rng.random() < 0.22 and len(raw_t.split()) >= 7:
                raw_t = _mild_typo(raw_t, rng)
            text = _unique_text(raw_t, cohort)
            if not text:
                continue
            is_mixed = any(
                tok in text.lower()
                for tok in ("bhai", "yaar", "kaafi", "nahi", "karo", "humare", "mein", "sahi")
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


def _ngrams(text: str, n: int = 6) -> set[Tuple[str, ...]]:
    toks = [t for t in text.lower().split() if t]
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def verify_zero_contamination(
    expanded_items: Sequence[Dict[str, Any]],
    heldout_test_texts: Sequence[str],
) -> Dict[str, Any]:
    probe_texts = [p["text"].strip().lower() for p in ROBUSTNESS_PROBES]
    probe_ngrams = set().union(*(_ngrams(t, 6) for t in probe_texts))
    test_set = {t.strip().lower() for t in heldout_test_texts}

    exact_probe_hits = 0
    ngram_probe_hits = 0
    exact_test_hits = 0
    for item in expanded_items:
        low = item["text"].strip().lower()
        if low in probe_texts:
            exact_probe_hits += 1
        if _ngrams(low, 6) & probe_ngrams:
            ngram_probe_hits += 1
        if low in test_set:
            exact_test_hits += 1

    if exact_probe_hits or ngram_probe_hits or exact_test_hits:
        raise RuntimeError(
            f"Contamination detected: exact_probe={exact_probe_hits}, "
            f"ngram_probe={ngram_probe_hits}, exact_test={exact_test_hits}"
        )
    return {
        "exact_probe_overlaps": exact_probe_hits,
        "sixgram_probe_overlaps": ngram_probe_hits,
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
