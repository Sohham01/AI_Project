# RCI — Human Annotation Guidelines (v1.0 Real-Data Pilot)

**Project:** Reddit Community Intelligence (RCI)  
**Applies To:** Sanitized blank-label candidate CSVs exported by `backend/import_reddit_export.py` and validated by `backend/validate_pilot_annotations.py`.

> [!IMPORTANT]
> **Ground-Truth Integrity Rule:** Never generate, copy, or accept automated model predictions as ground-truth labels for genuine community comments. Every label in a training CSV must reflect an explicit human judgment made under these guidelines.

---

## 1. Core Annotation Principles

1. **Read context before labeling:** Inspect `post_title`, `parent_text`, and `parent_context_status` (`top_level`, `available`, or `unavailable`) alongside `original_text`.
2. **Preserve text separation:** Read `original_text` (already PII-sanitized) to make your decision; do **not** edit `original_text`, `normalized_text`, `document_id`, `post_id`, or `comment_id`. The validator converts `normalized_text` into each task CSV's `text` column automatically.
3. **Label what the author expressed, not what you assume:** Do not infer emotion automatically from sentiment, do not mark sympathy as a solution, and do not guess an issue domain when `parent_context_status = unavailable` makes the referent unrecoverable.
4. **Never write placeholders into label cells:** Do **not** type `unclear`, `unknown`, `none`, `null`, or `n/a` into `sentiment`, `emotions`, `issues`, `response_type`, or `is_solution`. Use the dedicated `review_status` and `abstain_reason` workflow in Section 3 instead.
5. **Multilingual & code-mixed text (English, Hindi, Hinglish, Devanagari, Bengali, Odia):** Annotate the underlying meaning regardless of script or language mixing. If you cannot read a language or script reliably, set `review_status = abstain` and `abstain_reason = untranslatable` so a fluent reviewer can adjudicate it.

---

## 2. Task Definitions, Taxonomies & Decision Boundaries

### Task 1: Sentiment (Single-Label Multiclass — Required when `review_status = labeled`)

**Column:** `sentiment` | **Format:** exactly one lowercase value (no `|`).

| Label | Definition | Clear Example |
| --- | --- | --- |
| `negative` | Predominantly critical, dissatisfied, anxious, discouraging, or reporting an unresolved grievance. | *"Mess hygiene has gone downhill again and nobody from admin responds."* |
| `neutral` | Primarily factual, procedural, questioning, balanced/mixed without clear polarity, or short agreement with neutral facts. | *"Orientation usually starts in the last week of June; check the offer letter."* |
| `positive` | Predominantly appreciative, encouraging, satisfied, relieved, or reporting a constructive outcome. | *"The finance electives were genuinely practical and faculty stayed back for doubts."* |

**Boundary Rule (Mixed Polarity):** If a comment has mild criticism followed by an explicit positive resolution (*"Term 1 was rough, yet the faculty support made it totally worth it"*), code the **net overall stance** (`positive`). If both poles are genuinely balanced (*"Pros: good peer group; Cons: steep loan"*), code `neutral`.

---

### Task 2: Emotion (Multi-Label — 0, 1, or Multiple Labels)

**Column:** `emotions` | **Format:** pipe-separated (`|`) lowercase labels from the 6 canonical emotions, or **blank** when the comment is purely factual/procedural with no expressed affect.

| Label | Definition | Example Cue |
| --- | --- | --- |
| `confidence` | Self-assurance, conviction, or settled reassurance from experience. | *"Cleared it easily once I focused on mocks; you'll be fine."* |
| `curiosity` | Genuine inquiry, information-seeking, or open wondering. | *"Does the Business Analytics cohort share the same placement pool?"* |
| `frustration` | Annoyance, irritation, exasperation, or resentment toward a recurring obstacle. | *"Sending circulars 12 hours before deadlines every single term is ridiculous."* |
| `gratitude` | Appreciation, thankfulness, or explicit credit for help received. | *"Thanks a ton for sharing the SBI loan contact, really saved us time!"* |
| `hope` | Forward-looking optimism, constructive expectation, or encouragement. | *"Hopefully the new hostel block eases the room crunch next year."* |
| `stress` | Overwhelm, anxiety, burnout, financial worry, or deadline panic. | *"Drowning in 3 quizzes + case submissions + loan EMI anxiety this week."* |

**Boundary Rules:**
- **Empty set (`""`) is valid and important!** Factual statements (*"The campus is 15 km from Sambalpur Junction"*) with `review_status = labeled` should leave `emotions` blank.
- **Emotion $\neq$ Sentiment:** A negative complaint may express `frustration|stress`, only `frustration`, or no emotion at all (`""`). Never auto-fill emotions from sentiment.

---

### Task 3: Issue Domain (Multi-Label — 0, 1, or Multiple Labels)

**Column:** `issues` | **Format:** pipe-separated (`|`) lowercase labels from the 8 canonical domains, or **blank** if the comment discusses no institutional/student issue (e.g., pure greetings or general CAT percentile chatter unrelated to the 8 domains).

| Domain Label | Scope & Typical Topics |
| --- | --- |
| `academic_workload` | Exam clashes, quiz density, assignment/case-study overload, sleep deprivation, rigid attendance pressure. |
| `administrative_communication` | Late circulars, unclear office procedures, slow replies on certificates/transcripts/registrations, policy ambiguity. |
| `campus_infrastructure` | Campus Wi-Fi, library seating/sockets, classrooms, labs, shuttle transport, construction, medical/sports facilities. |
| `curriculum_teaching` | Course quality, case pedagogy, elective relevance, faculty teaching/mentorship, visiting faculty balance. |
| `extracurriculars` | Clubs, student committees, fests, case competitions, induction workload, club transparency/budgets. |
| `financial_concerns` | Tuition fees, education loans, moratorium interest, ROI concerns, scholarships, fee installments/refunds. |
| `hostel_facilities` | Hostel rooms, mess food quality/hygiene, water coolers/geysers, laundry, housekeeping, warden maintenance. |
| `placements` | Summer internships (SIP), final placements, CTC/package transparency, recruiter mix, PlaceCom processes, CV prep. |

**Boundary Rule (Parent-Post Context):** If a comment says *"Same issue on our floor, email the warden with room numbers"* under a post titled *"Hostel Block B water cooler broken"*, label `hostel_facilities` because the thread referent is explicit. If `parent_context_status = unavailable` and the comment says only *"Same here"*, do **not** guess — use `review_status = uncertain` (`abstain_reason = missing_parent`) if you cannot assign the other tasks either, or leave `issues` blank if the comment itself raises no identifiable domain issue.

---

### Task 4: Response Role (Single-Label Multiclass — Exact Canonical Order)

**Column:** `response_type` | **Format:** exactly one lowercase value from the **fixed canonical six-class taxonomy** (never add, rename, or reorder classes):

| Order | Label | Primary Discourse Role | Concrete Example |
| :---: | --- | --- | --- |
| 1 | `agreement` | Endorses, echoes, or seconds what the post or parent commenter said. | *"Same bro, totally agree with every point here."* |
| 2 | `personal_experience` | Reports the writer's own lived/past experience or direct observation. | *"I graduated last year; our finance batch had solid summer conversions."* |
| 3 | `opinion` | States a personal judgment, belief, critique, or question without lived narrative or pushback. | *"IMO taking a 22L loan makes sense only if you want a career pivot."* |
| 4 | `advice` | Recommends a general course of action, strategy, or precaution to another person. | *"You should talk to 3–4 current students on LinkedIn before locking your seat."* |
| 5 | `solution` | Provides a specific, actionable fix, workaround, office procedure, or concrete answer to a problem. | *"Attach your bank sanction letter and email accounts@<EMAIL>; they grant a 15-day fee extension."* |
| 6 | `counterargument` | Pushes back on, corrects, or qualifies a claim made in the post or parent comment. | *"Actually the visiting faculty brought real corporate datasets, so it wasn't purely theoretical."* |

**Critical Boundary Rules:**
- **`advice` vs. `solution`:**
  - `advice` = broader guidance/decision recommendation (*"Prepare well for mocks and compare alumni trajectories"*).
  - `solution` = concrete operational fix or direct answer that resolves a specific stated issue (*"Log a ticket on the IT portal with your MAC address; they whitelist it within 24h"*).
- **`opinion` vs. `counterargument`:**
  - Use `counterargument` **only** when the comment directly refutes, corrects, or pushes back against a claim in the post or parent comment. General negative opinions about the college/mess/fees that agree with a complaint are `opinion` or `agreement`, not `counterargument`.
- **Multi-role comments:** Choose the **primary** communicative intent (priority when genuinely mixed: `solution` > `advice` > `counterargument` > `personal_experience` > `agreement` > `opinion`, unless the secondary clause is merely a one-word opener like *"True, ..."* before a full `counterargument`).

---

### Task 5: Solution Detection (Single-Label Binary — Independent of `response_type`)

**Column:** `is_solution` | **Format:** `not_solution` or `solution` (the validator converts these to `false` / `true` in `solution.csv` and preserves canonical `["not_solution", "solution"]` in `ml/data/dataset.py`).

| Label | Definition |
| --- | --- |
| `solution` | The comment contains a concrete, actionable recommendation, procedural fix, resource, or workaround that helps address an issue discussed in the thread. |
| `not_solution` | The comment contains no actionable fix (e.g., pure complaint, agreement, sympathy, question, anecdote without actionable takeaway, or debate). |

**Boundary Rule:** A `solution` comment proposes a fix — it does **not** prove the issue was actually resolved in the real world (outcome/resolution evidence is modeled separately). Also, `response_type = advice` may be `is_solution = solution` when it gives a concrete actionable step, or `not_solution` when it is vague encouragement (*"Stay positive, things work out"*).

---

## 3. Explicit Uncertain, Abstain & Exclusion Workflow

Whenever an item **cannot** be labeled confidently across the five tasks, **do not guess** and **do not silently coerce it into `neutral`, `opinion`, `not_solution`, or `""`**. Instead:

1. Leave the 5 task label columns **blank**.
2. Set **`review_status`** to one of:
   - `uncertain` — Borderline case, sarcasm, or missing parent context requiring second-reviewer adjudication.
   - `abstain` — Non-substantive item (bot command like `RemindMe!`, single emoji, link-only reply, or script/language you cannot read).
   - `excluded` — Off-topic noise, spam, or residual sensitive/private personal content that should never enter model training.
3. Set **`abstain_reason`** to one of the controlled reason codes:
   - `insufficient_context`
   - `missing_parent`
   - `ambiguous_sarcasm`
   - `mixed_or_borderline`
   - `off_topic_noise`
   - `untranslatable`
   - `pii_risk`
   - `other`
4. Record a brief note in `annotator_notes` explaining why.

When `backend/validate_pilot_annotations.py` runs:
- Rows with `review_status = labeled` are strictly validated and converted into the 5 canonical training CSVs.
- Rows with `review_status ∈ {uncertain, abstain, excluded}` are routed cleanly into `adjudication_queue.csv` and **excluded from model training** until adjudicated by two reviewers.
- Rows with `review_status = ""` or `unreviewed` are left untouched and never silently treated as negative labels.

---

## 4. Worked Annotation Examples

| Sanitized Comment (`original_text`) | `sentiment` | `emotions` | `issues` | `response_type` | `is_solution` | `review_status` | `abstain_reason` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *"Attach your bank sanction letter and mail `<EMAIL>`; accounts gives a 15-day fee extension."* | `positive` | `hope` | `financial_concerns` | `solution` | `solution` | `labeled` | |
| *"Bhai mess food phir kharab ho gaya, subah se sab pareshan hain."* | `negative` | `frustration\|stress` | `hostel_facilities` | `opinion` | `not_solution` | `labeled` | |
| *"Not true actually — our operations visiting prof used live supply chain data."* | `positive` | `confidence` | `curriculum_teaching` | `counterargument` | `not_solution` | `labeled` | |
| *"Campus is 15 km from Sambalpur railway station."* | `neutral` | *(blank)* | *(blank)* | `opinion` | `not_solution` | `labeled` | |
| *"Lol yeah right."* (`parent_context_status = unavailable`) | *(blank)* | *(blank)* | *(blank)* | *(blank)* | *(blank)* | `uncertain` | `missing_parent` |
| *"RemindMe! 3 days"* | *(blank)* | *(blank)* | *(blank)* | *(blank)* | *(blank)* | `abstain` | `off_topic_noise` |

---

## 8. Post-IAA Calibration Addendum (v2.0 — 6 Mandatory Boundary Decision Trees)

Following our 50-item blind Human IAA audit ([V5_HUMAN_IAA_REPORT.md](V5_HUMAN_IAA_REPORT.md)), six recurring boundary discrepancies accounted for >80% of inter-annotator disagreements. Before annotating any new batch (such as the 30-item blind calibration batch `iaa_calibration_30.csv`), **every annotator must apply the six deterministic tie-breakers below**:

1. **`solution` vs. `not_solution` (Actionable Fix Rule):**
   - **Rule:** Tag `solution` **ONLY** if the comment supplies a **concrete, repeatable action, workaround, portal/escalation path, financing/scholarship channel, or step-by-step fix** that helps resolve the issue (e.g., *"Email admissions@… with your sanction letter for a 15-day extension"*).
   - **Tag `not_solution`** for: (a) plain factual AMA answers (*"Yes, OBC quota exists"*), (b) broad general advice without concrete steps (*"Work hard and network"*), or (c) status observations (*"Waitlist moved by 40"*).
2. **`negative` vs. `neutral` Sentiment (Critical Evaluation Rule):**
   - **Rule:** Tag `negative` whenever the author expresses **dissatisfaction, skepticism, warning, complaint, regret, sarcasm, or unmet expectation** — NOT solely profanity or overt anger (e.g., *"Alone internship cannot give you good placement"*, *"Fees feel like a honey trap"*, *"Portal is really confusing"* → **`negative`**).
   - **Reserve `neutral`** for genuinely matter-of-fact statements, neutral data reports, or balanced/unemotional questions (*"Cutoffs were 92 percentile last year"* → **`neutral`**).
3. **`counterargument` vs. `opinion` (Stance Reversal Rule):**
   - **Rule:** Inspect the parent post/comment. If the reply **explicitly or implicitly pushes back against, corrects, challenges, or contradicts** a prior claim/assumption (*"Actually Sambalpur placements aren't as bad as this post claims..."*, *"No, that fee figure is outdated"*), tag **`counterargument`** even without the literal phrase *"I disagree"*.
   - **Reserve `opinion`** for standalone evaluations or recommendations that do **not** contest a prior speaker's claim.
4. **`hostel_facilities` vs. `campus_infrastructure` (Living vs. Academic Facility Split):**
   - **`hostel_facilities`:** Dorm rooms, roommates, mess food, mess hygiene, canteen/kettle/cooking, hostel rules, AC/coolers, wardens, PG/off-campus stays.
   - **`campus_infrastructure`:** Permanent campus construction, classrooms, library, Wi-Fi, labs, location/connectivity (airport, railway station, Jharsuguda/Sambalpur city), weather.
5. **`financial_concerns` vs. `placements` (Cost vs. Finance Career Split):**
   - **`financial_concerns`:** Tuition fees, hostel/mess fees, education loans, EMIs, scholarships, refunds, withdrawal deductions, affordability/ROI cost burden.
   - **`placements` (NOT `financial_concerns`):** Finance-domain job roles, IB/corporate finance SIPs, CTC/LPA packages, recruiting companies, placement committees.
6. **`confidence` Emotion vs. Blank (Emotional Affect vs. Plain Fact):**
   - **Rule:** Leave `emotions` **blank (`""`)** for matter-of-fact procedural replies (*"Check the portal"*, *"Yes, reservation applies"*, *"Program is 4 years"*).
   - **Tag `confidence`** ONLY when the speaker expresses **explicit personal assurance, pride, reassurance, or strong conviction** (*"Don't worry, you will definitely convert at WL 15!"*, *"Super proud of how our batch performed"*).
