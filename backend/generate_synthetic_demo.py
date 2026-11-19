"""Generate a reproducible, synthetic-only Reddit Community Intelligence demo corpus.

No external data or network access is used. The fixed seed reproduces corpus text,
IDs, labels, and split inputs; the generation timestamp is recorded separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.nlp.preprocessing.multilingual_preprocessor import MultilingualTextPreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo_synthetic"
SEED = 20261116
POST_COUNT = 160
COMMENTS_PER_POST = 8

DOMAINS = [
    {"key":"placements","name":"placements and internships","subject":"placement office","issue":"placements","details":["shortlisting timelines","internship role descriptions","the final interview schedule","the PPO decision window","company presentation slots","resume review appointments","the summer internship portal","alumni recruiter sessions","the offer-letter checklist","the placement calendar","the analytics role criteria","the campus interview rooms","the internship preference form","the recruiter Q&A session","the pre-placement talk notices","the offer comparison guide","the consulting case workshop","the product role assessment","the finance internship dates","the placement help desk"]},
    {"key":"workload","name":"academic workload","subject":"course schedule","issue":"academic_workload","details":["two project deadlines landing together","the week with three case submissions","the evening reading load","the midterm preparation calendar","the group assignment handoff","the long case packet","the class participation schedule","the back-to-back quizzes","the weekly reflection notes","the end-term revision window","the statistics problem set","the late lecture finish","the group presentation week","the elective reading list","the missed tutorial slot","the capstone milestone calendar","the analytics lab report","the finance practice set","the cross-course project brief","the make-up class notice"]},
    {"key":"curriculum","name":"curriculum and teaching","subject":"course coordinator","issue":"curriculum_teaching","details":["the case discussion format","the elective registration guide","the classroom feedback form","the course outline revision","the guest lecture schedule","the tutorial group allocation","the grading rubric","the project evaluation criteria","the course material index","the office-hour calendar","the simulation exercise","the reading list update","the class recording notice","the assessment weight breakdown","the lab demonstration slot","the optional workshop plan","the semester learning outcomes","the peer review instructions","the course survey window","the teaching assistant roster"]},
    {"key":"infrastructure","name":"campus infrastructure","subject":"campus facilities team","issue":"campus_infrastructure","details":["the study-room booking page","the evening shuttle timing","the power backup in the seminar hall","the classroom projector queue","the water station near the library","the campus Wi-Fi dead spot","the late-hour reading room access","the seminar room ventilation","the sports ground lights","the accessible ramp entrance","the bicycle stand location","the lecture hall microphones","the computer lab opening hours","the library seat map","the bus-stop shelter","the printer maintenance notice","the campus signboard map","the common-room charging points","the auditorium entry queue","the laboratory access card"]},
    {"key":"hostel","name":"hostel and facilities","subject":"residence office","issue":"hostel_facilities","details":["the laundry slot booking","the hostel maintenance request form","the hot-water schedule","the shared study lounge","the room allocation notice","the mess menu update","the visitor entry procedure","the corridor cleaning roster","the water filter replacement","the quiet-hours reminder","the room repair queue","the common kitchen rules","the hostel Wi-Fi signal","the dining hall timing","the maintenance contact list","the late return procedure","the shared refrigerator space","the room inspection checklist","the laundry machine downtime","the hostel shuttle stop"]},
    {"key":"clubs","name":"clubs and extracurriculars","subject":"student activities office","issue":"extracurriculars","details":["the debate practice calendar","the photography club equipment list","the annual fest volunteer slots","the sports trials notice","the case competition team signup","the music-room booking process","the student council nomination dates","the club budget form","the weekend workshop schedule","the theatre rehearsal venue","the inter-college event travel plan","the society membership renewal","the design sprint briefing","the alumni networking evening","the cultural night rehearsal","the club handover checklist","the entrepreneurship meetup","the outdoor activity consent form","the campus newspaper deadline","the community service roster"]},
    {"key":"administration","name":"administration and communication","subject":"student services desk","issue":"administrative_communication","details":["the scholarship document checklist","the semester registration confirmation","the academic calendar correction","the identity-card update form","the fee receipt download page","the official notice archive","the leave request workflow","the transcript request status","the timetable change alert","the student portal help page","the grievance acknowledgement","the certificate collection window","the exam seating notice","the attendance correction form","the course withdrawal deadline","the hostel allotment email","the payment confirmation process","the lost-card replacement steps","the department contact directory","the campus closure announcement"]},
    {"key":"finance","name":"financial concerns","subject":"finance office","issue":"financial_concerns","details":["the installment payment dates","the education-loan certificate","the scholarship renewal window","the fee breakdown document","the travel reimbursement form","the library fine appeal","the hostel deposit receipt","the student aid application","the invoice correction request","the fee payment portal","the loan interest certificate","the refund processing status","the meal-plan charge summary","the emergency grant criteria","the bank account verification","the late-fee waiver form","the expense claim checklist","the payment schedule notice","the scholarship result update","the fee receipt archive"]},
]

ROLE_ORDER = ["agreement", "personal_experience", "opinion", "advice", "solution", "counterargument"]
ROLE_TEXT = {
    "agreement": [
        "I agree about {detail}; the {subject} can make that information much easier to find.",
        "That matches my view on {detail}. Clear updates from the {subject} would help everyone plan.",
        "Haan, {detail} wali baat sahi hai; {subject} se update clear aaye to kaafi help hoti hai.",
        "मैं सहमत हूँ—{detail} पर साफ सूचना मिले तो अगला कदम समझना आसान होगा।",
    ],
    "personal_experience": [
        "I dealt with {detail} last term. The hardest part was knowing which update from the {subject} was current.",
        "Mere saath bhi {detail} ka scene hua tha; {subject} ko mail karne ke baad status samajh aaya.",
        "पिछले सत्र में मुझे {detail} के लिए इंतज़ार करना पड़ा; फिर कार्यालय से स्थिति स्पष्ट हुई।",
        "When I ran into {detail}, I kept checking old messages until the {subject} sent a newer note.",
    ],
    "opinion": [
        "My view is that {detail} should be explained in one place, not scattered across notices.",
        "I feel {detail} would be less confusing if the {subject} used a consistent update format.",
        "Mere hisaab se {detail} ka process simple hona chahiye; sabko same information milni chahiye.",
        "मेरे विचार से {detail} को स्पष्ट रूप से समझाना चाहिए, ताकि अलग-अलग सूचनाएँ भ्रम न पैदा करें।",
    ],
    "advice": [
        "You could write to the {subject} with the exact question about {detail} and ask for the latest checklist.",
        "Try keeping the {detail} notice handy, then confirm any unclear point with the {subject} before acting.",
        "Aap {detail} ka screenshot rakh lo aur {subject} se specific point confirm kar lo; follow-up easy rahega.",
        "आप {detail} से जुड़ी सूचना सुरक्षित रखें और जरूरत हो तो कार्यालय से लिखित पुष्टि माँगें।",
    ],
    "solution": [
        "The simplest fix is for the {subject} to publish one dated page covering {detail} and its next update.",
        "For {detail}, a shared checklist with a named contact at the {subject} gives everyone the same steps.",
        "Iska practical fix: {subject} ek updated checklist aur contact point ek hi page par publish kare.",
        "समाधान यह है कि {subject} {detail} के लिए तारीख सहित निर्देश और संपर्क एक जगह प्रकाशित करे।",
    ],
    "counterargument": [
        "I see it differently: {detail} is not only a student planning issue; the {subject} also controls how clearly it is communicated.",
        "Not necessarily—{detail} can be manageable when the {subject} shares changes early, so the process itself matters.",
        "Main is baat se poori tarah agree nahi karta; {detail} mein update timing bhi {subject} ke control mein hoti hai.",
        "मैं इस निष्कर्ष से सहमत नहीं हूँ—{detail} में सूचना देने का समय भी महत्वपूर्ण है।",
    ],
}
SENTIMENT_CUES = {
    "positive": ["It feels manageable now.", "That would make the week lighter.", "Kaafi relief milega.", "इससे भरोसा बढ़ेगा।"],
    "neutral": ["I am still checking the details.", "The next update should clarify it.", "Abhi details verify kar raha hoon.", "अगली सूचना से स्थिति स्पष्ट होगी।"],
    "negative": ["The uncertainty has been stressful.", "It has made planning harder.", "Is delay se kaafi tension hui.", "इस देरी से योजना बनाना कठिन हुआ।"],
}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
COMMENT_NOTES = [
    "The next step is what matters here.", "Timing is the detail I would watch.",
    "A clear owner would make the process easier.", "One current update would reduce confusion.",
    "That would help people plan around classes.", "The written instructions should stay easy to find.",
    "A short follow-up would close the loop.", "Keeping the guidance consistent would help.",
]
SUCCESS_OUTCOMES = [
    "I followed the {subject} instructions for {detail}, and the office confirmed my request. That resolved it for me.",
    "After I used the {subject} process for {detail}, my request was accepted and the status changed to confirmed. That fixed my issue.",
    "I tried the {subject} steps for {detail}; the update shows my request is complete, so I can confirm the issue is resolved in my case.",
]
PENDING_OUTCOMES = [
    "I followed the {subject} instructions for {detail}, but my request is still pending. I do not have a confirmed outcome yet.",
    "I used the {subject} process for {detail}; the request has not moved past review, so the issue is not resolved for me yet.",
    "I tried the {subject} steps for {detail}, but I am still waiting for confirmation. I cannot say this worked yet.",
]


def _language(text: str) -> tuple[str, str, bool, list[str]]:
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return "hi", "devanagari", False, ["hi"]
    if any(word in text.lower().split() for word in ("haan,", "mere", "aap", "kaafi", "nahi", "hai.", "rahega.")):
        return "hinglish", "latin", True, ["en", "hi"]
    return "en", "latin", False, ["en"]


def _comment_text(rng: random.Random, role: str, domain: dict, detail: str, sentiment: str, note: str) -> str:
    base = rng.choice(ROLE_TEXT[role]).format(detail=detail, subject=domain["subject"])
    cue = rng.choice(SENTIMENT_CUES[sentiment])
    # Keep language natural: avoid appending a second language to Devanagari text.
    if any("\u0900" <= ch <= "\u097f" for ch in base) and cue.isascii():
        cue = {"positive":"इससे उम्मीद बढ़ेगी।", "neutral":"अभी और जानकारी देखनी होगी।", "negative":"इससे चिंता बनी हुई है।"}[sentiment]
    close = f" {note}" if rng.random() < 0.35 else ""
    return f"{base} {cue}{close}"


def _post_text(rng: random.Random, domain: dict, detail: str, idx: int) -> tuple[str, str]:
    titles = [
        "How are people handling {detail}?",
        "Question about {detail} and the {subject}",
        "Could the process for {detail} be clearer?",
        "Trying to understand the latest information on {detail}",
        "What would make {detail} easier to navigate?",
    ]
    bodies = [
        "I could not find one current explanation of {detail} from the {subject}. Which notice should I use to plan the next step?",
        "The information about {detail} seems to be spread across a few updates. A single dated summary would make the process easier to follow.",
        "I am trying to understand what happens with {detail}. The last update I found may be out of date, so I would appreciate a pointer to the current guidance.",
        "Has anyone dealt with {detail} recently? I want to know what to prepare before contacting the {subject}.",
    ]
    return rng.choice(titles).format(detail=detail, subject=domain["subject"]), rng.choice(bodies).format(detail=detail, subject=domain["subject"], idx=idx)


def build() -> dict:
    rng = random.Random(SEED)
    posts = []
    flat_comments = []
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    post_idx = 0
    for domain in DOMAINS:
        details = list(domain["details"])
        rng.shuffle(details)
        for detail in details:
            post_idx += 1
            post_id = f"syn-post-{post_idx:04d}"
            title, body = _post_text(rng, domain, detail, post_idx)
            created = base_time + timedelta(days=post_idx * 2)
            resolution_state = ("reported_success", "reported_pending", "proposed_only", "insufficient_evidence")[(post_idx - 1) % 4]
            roles = ROLE_ORDER + rng.sample(ROLE_ORDER, 2)
            rng.shuffle(roles)
            sentiments = ["positive", "positive", "positive", "negative", "negative", "neutral", "neutral", "positive"]
            rng.shuffle(sentiments)
            comments = []
            for j, role in enumerate(roles):
                sentiment = sentiments[j]
                cid = f"syn-comment-{post_idx:04d}-{j+1:02d}"
                text = _comment_text(rng, role, domain, detail, sentiment, COMMENT_NOTES[j])
                evidence_type = None
                if role == "personal_experience" and resolution_state == "reported_success":
                    sentiment = "positive"
                    evidence_type = "reported_success"
                    text = rng.choice(SUCCESS_OUTCOMES).format(subject=domain["subject"], detail=detail)
                elif role == "personal_experience" and resolution_state == "reported_pending":
                    sentiment = "negative"
                    evidence_type = "reported_pending"
                    text = rng.choice(PENDING_OUTCOMES).format(subject=domain["subject"], detail=detail)
                lang, script, mixed, detected = _language(text)
                emotion_map = {"positive":["hope", "gratitude"], "neutral":["curiosity"], "negative":["stress", "frustration"]}
                emotions = emotion_map[sentiment].copy()
                if role == "personal_experience" and sentiment == "positive":
                    emotions = ["confidence", "gratitude"]
                # Label derivation is from the generation scenario, never from model predictions.
                is_solution = role in {"advice", "solution"}
                solution_type = domain["issue"] if is_solution else None
                parent_id = None if j < 2 else comments[j-1]["reddit_id"]
                comment = {
                    "reddit_id": cid, "document_id": cid, "body": text,
                    "created_at": (created + timedelta(minutes=(j+1)*7)).isoformat(),
                    "score": 0, "parent_reddit_id": parent_id,
                    "response_type": role, "sentiment": sentiment,
                    "emotions": emotions, "issue_labels": [domain["issue"]],
                    "is_solution": is_solution, "solution_type": solution_type,
                    "resolution_evidence_type": evidence_type,
                    "language": lang, "script": script, "is_code_mixed": mixed,
                    "detected_languages": detected, "synthetic": True,
                }
                comments.append(comment)
                flat_comments.append({**comment, "post_id": post_id, "subreddit": "IIMSambalpur", "issue_category": domain["name"], "issue_key": domain["issue"], "post_title": title, "post_body": body, "parent_comment_text": comments[j-1]["body"] if parent_id else None})
            posts.append({
                "reddit_id": post_id, "title": title, "body": body,
                "subreddit": "IIMSambalpur", "issue_category": domain["name"],
                "issue_key": domain["issue"], "resolution_state": resolution_state,
                "resolution_evidence_comment_id": next((c["reddit_id"] for c in comments if c["resolution_evidence_type"]), None),
                "created_at": created.isoformat(),
                "score": 0, "num_comments": COMMENTS_PER_POST, "url": None,
                "source_mode": "synthetic", "synthetic": True, "comments": comments,
            })
    counts = {
        "sentiment": dict(Counter(c["sentiment"] for c in flat_comments)),
        "response_type": dict(Counter(c["response_type"] for c in flat_comments)),
        "issue_category": dict(Counter(c["issue_key"] for c in flat_comments)),
        "solution": dict(Counter(str(c["is_solution"]).lower() for c in flat_comments)),
    }
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "metadata": {
            "dataset_id": "reddit-community-intelligence-demo",
            "dataset_version": "1.0.0",
            "generation_seed": SEED,
            "generation_timestamp": now,
            "source_provider": "synthetic_generator",
            "source_mode": "synthetic_only",
            "synthetic": True,
            "authorization_status": "synthetic_only",
            "notice": "All posts, comments, IDs, labels, and scenarios are fictional. Not Reddit data, not real student statements, and not evidence about IIM Sambalpur.",
            "post_count": len(posts), "comment_count": len(flat_comments),
            "comments_per_post": COMMENTS_PER_POST,
            "label_counts": counts,
            "label_provenance": "Independent scenario rules in this generator; no model predictions were used to create labels.",
        },
        "posts": posts,
    }


def write_csv(path: Path, rows: list[dict], label_field: str, multi: bool = False) -> None:
    fields = ["document_id", "text", label_field, "post_id", "comment_id", "language", "script", "is_code_mixed", "detected_languages"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        preprocessor = MultilingualTextPreprocessor()
        for row in rows:
            label = row["issue_labels"] if label_field == "labels" else row[label_field]
            normalized = preprocessor.preprocess(
                row["body"], language=row["language"], script=row["script"],
                is_code_mixed=row["is_code_mixed"], detected_languages=row["detected_languages"],
            ).normalized_text
            writer.writerow({
                "document_id": row["document_id"], "text": normalized,
                label_field: "|".join(label) if isinstance(label, list) else str(label).lower() if isinstance(label, bool) else label,
                "post_id": row["post_id"], "comment_id": row["reddit_id"],
                "language": row["language"], "script": row["script"],
                "is_code_mixed": str(row["is_code_mixed"]).lower(),
                "detected_languages": "|".join(row["detected_languages"]),
            })


def main(force: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()) and not force:
        raise SystemExit(f"Refusing to overwrite existing synthetic dataset files in {OUT}. Re-run with --force to regenerate this demo dataset.")
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = build()
    comments = [dict(c, post_id=p["reddit_id"]) for p in dataset["posts"] for c in p["comments"]]
    (OUT / "corpus.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "sentiment.csv", comments, "sentiment")
    write_csv(OUT / "emotion.csv", comments, "emotions")
    write_csv(OUT / "issue.csv", comments, "labels")
    write_csv(OUT / "response.csv", comments, "response_type")
    write_csv(OUT / "solution.csv", comments, "is_solution")
    print(json.dumps(dataset["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite files in data/demo_synthetic")
    main(force=parser.parse_args().force)
