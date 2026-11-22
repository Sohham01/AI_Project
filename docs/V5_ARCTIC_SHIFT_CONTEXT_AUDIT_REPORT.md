# Phase 6 — Full-Arctic-Shift-Context Blind Pass-2 Audit Report (NOT Human IAA)

> [!WARNING]
> **Provenance & Scientific Integrity Boundary:** This report compares **Blind AI Pass-2** (annotated with full local + live Arctic Shift post bodies, ancestor chains, and community direct replies across all 27 threads / +136 extra live comments) against **V2 Rubric Pass-1** (narrow `post_title + parent_text` context). It is an **AI consistency & context-sensitivity audit**, **NOT** independent human Inter-Annotator Agreement. `human_iaa_verified` remains `false` until a human reviews/adjudicates the pack.

**Coverage:** `50 / 50` items (`49` both labeled; `1` both abstained as `off_topic_noise`; review-status agreement `100.0%`)  
**Live Arctic Shift Thread Lookup:** `27 / 27` posts matched (`+136` extra live comments fetched; zero author names stored; PII redacted)  
**Human IAA Gate Unlocked:** `NO` *(requires human sign-off; `ml/artifacts/v5_experiments/active_learning/iaa_50_human_annotator_studio.csv` preserved strictly blank prior to human review)*

## Task-Level Agreement: Full-Arctic-Shift Pass-2 vs. Narrow Pass-1

| Task | Paradigm | $n$ | Raw / Exact % | Cohen's $\kappa$ | Krippendorff's $\alpha$ | Reference Threshold | Pass-2 vs Pass-1 |
|---|---|---:|---:|---:|---:|---:|---|
| **sentiment** | Single-label | 49 | 87.76% | `0.7846` | `0.7867` | $\ge 0.7$ | ABOVE |
| **response** | Single-label | 49 | 83.67% | `0.7788` | `0.7809` | $\ge 0.65$ | ABOVE |
| **solution** | Single-label | 49 | 91.84% | `0.5545` | `0.5591` | $\ge 0.7$ | BELOW — inspect |
| **issue** | Multi-label | 49 | 75.51% (Jaccard `0.7789`) | *(per-label)* | MASI `0.7205` / Micro `0.8713` | $\ge 0.65$ | ABOVE |
| **emotion** | Multi-label | 49 | 75.51% (Jaccard `0.7653`) | *(per-label)* | MASI `0.6564` / Micro `0.7242` | $\ge 0.65$ | ABOVE |

## Per-Label Multi-Label Breakdown

| Task | Label | Pass-2 Count | Pass-1 Count | Cohen's $\kappa$ | Krippendorff's $\alpha$ |
|---|---|---:|---:|---:|---:|
| `issue` | `academic_workload` | 1 | 1 | `1.0000` | `1.0000` |
| `issue` | `administrative_communication` | 8 | 12 | `0.7513` | `0.7513` |
| `issue` | `campus_infrastructure` | 0 | 0 | `1.0000` | `1.0000` |
| `issue` | `curriculum_teaching` | 10 | 10 | `1.0000` | `1.0000` |
| `issue` | `extracurriculars` | 1 | 1 | `1.0000` | `1.0000` |
| `issue` | `financial_concerns` | 6 | 7 | `0.7342` | `0.7367` |
| `issue` | `hostel_facilities` | 5 | 6 | `0.8977` | `0.8986` |
| `issue` | `placements` | 19 | 21 | `0.8313` | `0.8328` |
| `emotion` | `confidence` | 6 | 6 | `0.6202` | `0.6240` |
| `emotion` | `curiosity` | 6 | 8 | `0.8339` | `0.8350` |
| `emotion` | `frustration` | 7 | 10 | `0.7879` | `0.7887` |
| `emotion` | `gratitude` | 0 | 0 | `1.0000` | `1.0000` |
| `emotion` | `hope` | 4 | 3 | `0.5392` | `0.5432` |
| `emotion` | `stress` | 2 | 4 | `0.6475` | `0.6486` |

## Disagreements Flagged for Human Adjudication (`25` / 50 items)

Exported side-by-side with `adjudicated_*` columns to `ml/artifacts/v5_experiments/active_learning/iaa_ai_arctic_shift_disagreements.csv`.

| Document ID | Disagreed Tasks | Pass-2 (Arctic Shift Context) | Pass-1 (Narrow Context) | Comment Excerpt |
|---|---|---|---|---|
| `doc-v2-cmt_00009` | `issue|emotion` | sent=`negative`, emo=``, iss=``, resp=`opinion`, sol=`not_solution` | sent=`negative`, emo=`frustration`, iss=`administrative_communication`, resp=`opinion`, sol=`not_solution` | Yarr ab Sambhalpur jaise bhi wait list karne lage hain?🤣 |
| `doc-v2-cmt_00211` | `issue` | sent=`neutral`, emo=`curiosity`, iss=``, resp=`counterargument`, sol=`not_solution` | sent=`neutral`, emo=`curiosity`, iss=`administrative_communication`, resp=`counterargument`, sol=`not_solution` | So what you're saying is that only after 3 years of experience one can apply for E-MBA I think it requires atl |
| `doc-v2-cmt_00212` | `issue` | sent=`neutral`, emo=``, iss=``, resp=`counterargument`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`administrative_communication`, resp=`counterargument`, sol=`not_solution` | The colleges i mentioned and many others require 3 years, A requires 4yrs and B 5yrs. |
| `doc-v2-cmt_00304` | `sentiment|response|issue` | sent=`neutral`, emo=`curiosity|stress`, iss=``, resp=`counterargument`, sol=`not_solution` | sent=`negative`, emo=`curiosity|stress`, iss=`placements`, resp=`opinion`, sol=`not_solution` | I dont think i’ll fit in the pgpx course from IIM A where an average workex is around 7-8 years, for me to be  |
| `doc-v2-cmt_00413` | `sentiment|response|solution|emotion` | sent=`neutral`, emo=`confidence`, iss=`curriculum_teaching|financial_concerns`, resp=`advice`, sol=`solution` | sent=`negative`, emo=``, iss=`curriculum_teaching|financial_concerns`, resp=`opinion`, sol=`not_solution` | If you are interested to learn Data Science in depth, no better than Udemy at Rs. 600 courses. |
| `doc-v2-cmt_00551` | `sentiment|solution|issue|emotion` | sent=`positive`, emo=`confidence|hope`, iss=``, resp=`advice`, sol=`solution` | sent=`neutral`, emo=``, iss=`placements`, resp=`advice`, sol=`not_solution` | If you convert Sambalpur, think about it, but your potential is still high. You can get a much better B-school |
| `doc-v2-cmt_00843` | `sentiment` | sent=`negative`, emo=``, iss=`placements`, resp=`counterargument`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`placements`, resp=`counterargument`, sol=`not_solution` | Bhai kuch skills bhi to btana padega to get good placement |
| `doc-v2-cmt_01131` | `solution|issue|emotion` | sent=`positive`, emo=`hope`, iss=``, resp=`advice`, sol=`not_solution` | sent=`positive`, emo=`confidence`, iss=`administrative_communication`, resp=`advice`, sol=`solution` | XIMB (tier 1.5 ) loves candidates with work experience in consultancy. Expected call 90-93. IITs love engineer |
| `doc-v2-cmt_01744` | `sentiment` | sent=`positive`, emo=`hope`, iss=`curriculum_teaching|placements`, resp=`personal_experience`, sol=`not_solution` | sent=`neutral`, emo=`hope`, iss=`curriculum_teaching|placements`, resp=`personal_experience`, sol=`not_solution` | Yes, you can leave after the 3rd year with a degree according to NEP 2020, and the institute does provide that |
| `doc-v2-cmt_01807` | `response` | sent=`neutral`, emo=``, iss=`placements`, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`placements`, resp=`personal_experience`, sol=`not_solution` | Internships are to be done during the sem breaks when you'll be out of campus. Its mostly offline internships  |
| `doc-v2-cmt_01855` | `response|solution` | sent=`neutral`, emo=``, iss=`financial_concerns`, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`financial_concerns`, resp=`solution`, sol=`solution` | Yes there is for st sc and obc |
| `doc-v2-cmt_01859` | `issue|emotion` | sent=`neutral`, emo=``, iss=``, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=`curiosity`, iss=`financial_concerns`, resp=`opinion`, sol=`not_solution` | I dont know how to do chat can you do a msg please 🙏 |
| `doc-v2-cmt_01860` | `response|issue` | sent=`neutral`, emo=``, iss=``, resp=`agreement`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`financial_concerns`, resp=`personal_experience`, sol=`not_solution` | Done bro please check the msg |
| `doc-v2-cmt_01977` | `response` | sent=`neutral`, emo=``, iss=`financial_concerns`, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`financial_concerns`, resp=`counterargument`, sol=`not_solution` | bhai wha pr jo need bassed scholorship h wo isirf mba walo ke liye h |
| `doc-v2-cmt_02016` | `issue` | sent=`negative`, emo=`frustration`, iss=``, resp=`counterargument`, sol=`not_solution` | sent=`negative`, emo=`frustration`, iss=`placements`, resp=`counterargument`, sol=`not_solution` | So a software engineer makes reddit account this day, only to come in a niche course reddit sub and guide stud |
| `doc-v2-cmt_02017` | `issue` | sent=`negative`, emo=`confidence|frustration`, iss=`curriculum_teaching|financial_concerns|placements`, resp=`counterargument`, sol=`not_solution` | sent=`negative`, emo=`confidence|frustration`, iss=`curriculum_teaching|placements`, resp=`counterargument`, sol=`not_solution` | FUCK THIS GUY , BUILD A COMPANY ,that’s what this course is designed for 50%management and 50% core . Not a si |
| `doc-v2-cmt_02196` | `emotion` | sent=`negative`, emo=``, iss=`financial_concerns`, resp=`opinion`, sol=`not_solution` | sent=`negative`, emo=`stress`, iss=`financial_concerns`, resp=`opinion`, sol=`not_solution` | That's a massive education loan – 30 lakhs for a bachelor's and 42 lakhs for a master's! Education loans can d |
| `doc-v2-cmt_02320` | `emotion` | sent=`positive`, emo=``, iss=`extracurriculars`, resp=`opinion`, sol=`not_solution` | sent=`positive`, emo=`hope`, iss=`extracurriculars`, resp=`opinion`, sol=`not_solution` | Mujhe bhi add krdo , im also a fresher |
| `doc-v2-cmt_02619` | `response` | sent=`neutral`, emo=``, iss=`administrative_communication`, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`administrative_communication`, resp=`agreement`, sol=`not_solution` | Looking like all seats has been filled |
| `doc-v2-cmt_03062` | `response|emotion` | sent=`negative`, emo=`frustration`, iss=`administrative_communication`, resp=`personal_experience`, sol=`not_solution` | sent=`negative`, emo=`frustration|stress`, iss=`administrative_communication`, resp=`opinion`, sol=`not_solution` | No bro, I honestly don’t know what’s going on with them. Some people on Reddit are saying they’ve received the |
| `doc-v2-cmt_03211` | `issue|emotion` | sent=`neutral`, emo=``, iss=`curriculum_teaching|placements`, resp=`advice`, sol=`not_solution` | sent=`neutral`, emo=`curiosity`, iss=`curriculum_teaching`, resp=`advice`, sol=`not_solution` | Bro can you tell me whats your future goals, do you want to pursue bachelors from abroad, if yes then you shou |
| `doc-v2-cmt_03501` | `sentiment|emotion` | sent=`neutral`, emo=``, iss=`placements`, resp=`counterargument`, sol=`not_solution` | sent=`positive`, emo=`confidence`, iss=`placements`, resp=`counterargument`, sol=`not_solution` | Not really bro. A lot of freshers at IIM M have gotten amazing SIPs here. Ig it depends from college to colleg |
| `doc-v2-cmt_03530` | `issue` | sent=`neutral`, emo=``, iss=``, resp=`opinion`, sol=`not_solution` | sent=`neutral`, emo=``, iss=`hostel_facilities`, resp=`opinion`, sol=`not_solution` | he didnt apply for cat as it wld eat him |
| `doc-v2-cmt_03531` | `emotion` | sent=`negative`, emo=``, iss=`hostel_facilities`, resp=`opinion`, sol=`not_solution` | sent=`negative`, emo=`frustration`, iss=`hostel_facilities`, resp=`opinion`, sol=`not_solution` | Mai toh ulti kar dungi dekhte hi 😭 |
| `doc-v2-cmt_03532` | `emotion` | sent=`negative`, emo=``, iss=`hostel_facilities|placements`, resp=`opinion`, sol=`not_solution` | sent=`negative`, emo=`frustration`, iss=`hostel_facilities|placements`, resp=`opinion`, sol=`not_solution` | Bro came for the placements, stayed for the free food since that was better |
