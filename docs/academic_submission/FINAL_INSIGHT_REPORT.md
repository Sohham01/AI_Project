# Deliverable 6 — Final Community Insight Report: IIM Discussions on Reddit (RCI V6.2)

> **Document ID:** `RCI-V6.2-D6` · **Navigation:** [← Deliverable 5b (Pseudocode)](./SYSTEM_PSEUDOCODE.md) · [Submission Index](./README.md)  
> **Epistemic Integrity Standard:** Every analytical finding below explicitly distinguishes **model-derived corpus observations** (`n = 365` threads, `3,557` comments, `verification_status: model_inferred`) from **human-reviewed evaluation benchmarks** (`n = 92` frozen 5-task test, `n = 84` real LID audit, `n = 64` issue–solution pairs, `n = 40` resolution threads, `35` BERTopic cluster reviews). No conclusions are invented where evidence is sparse, and observation-window censoring is stated explicitly.

---

## Q1. What does the IIM Reddit community discuss?

**Corpus Evidence (`365` threads, `3,557` comments across `8` public subreddits):**
- **Subreddit concentration:** `r/IIMSambalpurUG` accounts for `211` threads (`57.8%`), followed by `r/MBAIndia` (`46`, `12.6%`), `r/IIMCATPreparation` (`41`, `11.2%`), `r/CATPrep` (`18`), `r/iim_CAT` (`18`), `r/CATStudyRoom` (`15`), `r/CATpreparation` (`12`), and `r/Indian_Academia` (`4`).
- **Unsupervised Topic Discovery (NMF v1 vs. BERTopic v1 on `3,459` eligible comments):**
  - **Preserved NMF v1 (`10` fixed factors)** captures broad conversational buckets: general MBA/IIM discourse (`Topic 0`, `39.95%`), direct-message requests (`Topic 1`), percentile/waitlist/category queries (`Topic 5`, `10.0%`), Hinglish conversation (`Topic 6`, `9.4%`), BS DSAI vs. MPP program comparisons (`Topic 4`, `2.8%`), and CFBR reach-boosting (`Topic 7`, `1.7%`).
  - **Companion BERTopic v1 (`intfloat/multilingual-e5-small`, `35` natural clusters, `38.2%` explicit `-1` outliers)** uncovers sharp operational themes hidden inside NMF's mega-cluster: **education loans & CIBIL score rejections (`BT-12`, `n=59`)**, **fee refund & registration deadlines (`BT-13`, `n=59`)**, **peer B-school comparisons (`IMI`, `SCMHRD`, `GIM`, `GLIM`, `BT-06`, `n=73`)**, **repeat CAT vs. job workex (`BT-05`, `n=74`)**, **BS DSAI vs. MPP (`BT-07`, `n=68`)**, **JEE/CAT percentile cutoffs (`BT-08`, `n=66`)**, **EWS/OBC category waitlist tracking (`BT-09`/`BT-15`/`BT-20`)**, **faculty & classroom quality anxiety (`BT-16`, `n=49`)**, and **corporate internships & committees (`BT-31`, `n=28`)**.

---

## Q2. What are its major issues?

**Model-Inferred Thread & Comment Distribution across the 8 Canonical Domains:**

| Issue Domain | Primary Threads (`n=365`) | Comments (`n=3,557`) | Share of Comments | Phase 4 Contextual Linked Solutions |
|---|---:|---:|---:|---:|
| **`administrative_communication`** *(Admissions, waitlists, portals, refunds)* | **247** (`67.7%`) | **2,128** | **59.8%** | **297** (`86` clusters) |
| **`placements`** *(SIP, final CTC, recruiters, B-school ROI)* | **72** (`19.7%`) | **1,036** | **29.1%** | **236** (`52` clusters) |
| **`financial_concerns`** *(Tuition fees, education loans, CIBIL, refunds)* | 22 (`6.0%`) | 171 | 4.8% | 134 (`39` clusters) |
| **`extracurriculars`** *(Clubs, committees, student life, gender ratio)* | 9 (`2.5%`) | 113 | 3.2% | 47 (`22` clusters) |
| **`hostel_facilities`** *(Mess hygiene/rats/insects, room rules, kettles)* | 8 (`2.2%`) | 81 | 2.3% | 13 (`10` clusters) |
| **`campus_infrastructure`** *(Location, laptops, VIMSAR hospital access)* | 4 (`1.1%`) | 20 | 0.6% | 33 (`23` clusters) |
| **`curriculum_teaching`** *(BS DSAI/MPP rigor, faculty, Executive MBA)* | 3 (`0.8%`) | 8* | 0.2%* | 150 (`39` clusters)* |
| **`academic_workload`** *(Exam/CAT prep stress, coursework load)* | 0* | 0* | 0.0%* | 41 (`11` clusters)* |

*\*Important Data-Transparency Note:* `curriculum_teaching` and `academic_workload` rarely appear as the *primary thread-level* `issue_key` (because threads are titled around admissions/colleges), yet multi-label comment-level classification identifies `150` curriculum-linked solutions and `41` academic-workload-linked solutions inside broader AMA and college-comparison threads. Together, **admissions/administrative communication (`59.8%`)** and **placements (`29.1%`)** account for **`88.9%`** of all discussion volume.

---

## Q3. What emotions and attitudes accompany them?

**Model-Inferred Sentiment & Multi-Label Emotion Profile (`n = 3,557` comments):**
- **Sentiment (single-label, MuRIL-DAPT-WWM):** `55.3%` **Neutral** (`1,966` — informational queries, cutoff checks, waitlist numbers), `24.2%` **Negative** (`860` — placement skepticism, unresponsive administration, mess hygiene disgust, loan rejection distress), and `20.6%` **Positive** (`731` — admission converts, alumni reassurance, gratitude).
- **Emotions (multi-label, `2,931` total tags across `3,557` comments):**
  1. **`curiosity`** (`1,075` comments, `30.2%`) — dominant affect overall, driven by cutoff/waitlist/criteria questions.
  2. **`frustration`** (`804`, `22.6%`) — concentrated in unanswered administrative inquiries, refund delays, and baby-IIM placement critiques.
  3. **`stress`** (`366`, `10.3%`) — concentrated in loan/EMI fear, waitlist uncertainty, and repeat-CAT dilemmas.
  4. **`hope`** (`274`, `7.7%`), **`gratitude`** (`210`, `5.9%`), and **`confidence`** (`202`, `5.7%`).
- **Critical Sociolinguistic Discovery (Language × Affect):** English comments (`n = 2,898`) are `23.7%` negative with `curiosity` leading (`930` vs. `598` frustration), whereas **Hinglish/Indic comments (`n = 340`) are `45.9%` negative — nearly double English negativity — with `frustration` (`170`, `50.0%`) as the #1 emotion!** Students systematically code-switch into Hinglish when venting about unresponsive offices or perceived scams.

---

## Q4. What experiences do students report?

- **`630` comments (`17.7%` of corpus)** are classified into the canonical **`personal_experience`** role (second only to `opinion` at `1,403` / `39.4%`).
- Recurring firsthand narratives documented across anonymized threads include:
  - **Alumni balanced campus accounts** (e.g., `cmt_00064`, `post_0013`: detailing `13–14 LPA` realistic averages, stagnant Dec–Jan SIP followed by post-Jan final recruiter recovery, edible mess food, and transformative club/case-competition experience).
  - **Alternative B-school outcomes** (`cmt_00066`, `post_0014`: skipping a baby IIM for NIBM Pune finance placements; `cmt_01028`, `post_0104`: cracking BLACKI at age 28 after 4 years of UPSC prep).
  - **Administrative unresponsiveness** (`post_0150`, `post_0151`, `post_0182`, `post_0299`, `post_0319`: repeated reports of admissions phones going unanswered and withdrawal/refund emails receiving zero reply).
  - **Mess hygiene & campus living shocks** (`post_0322`, `post_0362`, `post_0363`: viral complaints about insects/rodents in mess food alongside practical hostel rules like no room cooking except electric kettles in `post_0194`).

---

## Q5. Does the community provide solutions?

**Yes — peer problem-solving is substantial, but must be strictly separated from verified resolution:**
- **Canonical Response Breakdown (`3,557` comments, locked 6-class taxonomy):**
  - `opinion`: `1,403` (`39.4%`)
  - `personal_experience`: `630` (`17.7%`)
  - `agreement`: `515` (`14.5%`)
  - **`solution`: `369` (`10.4%`)**
  - **`advice`: `344` (`9.7%`)**
  - `counterargument`: `296` (`8.3%`)
- **Actionable Density:** `713` replies (`20.0%`) fall into `solution` or `advice` response roles, and our Phase 4 Contextual Multilingual E5 Issue–Solution Mapper links **`951` high-confidence issue–solution edges** (`268` dismissive/non-actionable candidates explicitly penalized) across all 8 domains.
- **Yet Verified Outcomes Remain Scarce:** Under our Phase 5 Explainable Resolution Engine, **`162` threads (`44.4%`) are `proposed_only`** (peers offered concrete advice/solutions with no posted follow-up confirmation), while only **`12` threads (`3.3%`) contain explicit `reported_success`**!

---

## Q6. What solutions or advice are most common?

**Top Recurring Actionable Solution Clusters Discovered by Phase 4 (`282` Semantic Clusters via Cosine Agglomerative Clustering):**
1. **Admissions & Refunds (`297` linked solutions, `86` clusters):** Citing **AICTE 15-day full-refund guidelines** (max ₹1,000 processing deduction before classes start, `cmt_00014`/`cmt_00016`), checking Vercel/portal trackers (`cmt_00455`), benchmarking historical category waitlist movement (`cmt_00023`), and creating admitted-batch WhatsApp/Discord coordination groups (`cmt_01507`).
2. **Placements & Career Strategy (`236` linked solutions, `52` clusters):** Auditing alumni **`"Open to Work"` profiles on LinkedIn** (`cmt_00334`), preferring stronger Tier-2/1.5 peers (`IMI Delhi`, `BITSOM`, `NIBM Pune`, `TAPMI`) over weaker ROI tags (`cmt_00369`/`cmt_00787`), repeating CAT with 1–2 years of work experience (`cmt_00550`/`cmt_01073`), and pursuing live projects/PORs/certifications during MBA (`cmt_00277`).
3. **Financial & Loans (`134` linked solutions, `39` clusters):** Comparing bank/NBFC CIBIL requirements, exploring `25%` merit financial-aid/scholarship programs (`cmt_00809`), and exercising AICTE withdrawal timelines before losing deposits.
4. **Campus & Hostel Workarounds (`46` linked solutions, `33` clusters):** Using electric kettles within hostel room rules (`cmt_02059`), consulting curated campus Discord laptop hardware lists (`cmt_02194`), and utilizing nearby VIMSAR hospital (`2 km`, `cmt_00072`).

---

## Q7. Which issues appear unresolved?

**Phase 5 Explainable Unresolved-Issue Audit (`365` real threads; `0.00%` false resolution claims on `n = 40` human eval):**

| Issue Domain | Verdict | Threads | `reported_success` | `reported_pending` | `proposed_only` | `insufficient_evidence` | Contradictory Threads |
|---|---|---:|---:|---:|---:|---:|---:|
| **`administrative_communication`** | **Unresolved** | 247 | 9 (`3.6%`) | **30** | 90 | 118 | **12** |
| **`hostel_facilities`** | **Unresolved** | 8 | 0 (`0.0%`) | 0 | 5 | 3 | 0 |
| **`placements`** | **Partially Resolved** | 72 | 1 (`1.4%`) | 2 | 51 | 18 | **12** |
| **`financial_concerns`** | **Partially Resolved** | 22 | 1 (`4.5%`) | 2 | 11 | 8 | 1 |
| **`campus_infrastructure`** | **Partially Resolved** | 4 | 0 (`0.0%`) | 0 | 4 | 0 | 0 |
| **`extracurriculars`** | **Insufficient Evidence** | 9 | 1 (`11.1%`) | 0 | 1 | 7 | 1 |
| **`curriculum_teaching`** | **Insufficient Evidence** | 3 | 0 (`0.0%`) | 0 | 0 | 3 | 0 |

- **Most Persistently Unresolved Issue:** **`administrative_communication`** (`247` threads: `30` explicitly `reported_pending`, `90` `proposed_only`, and **`12` threads with explicit contradictory/procedural-failure follow-ups** such as *"bro I mailed them today they didn't respond"*, *"They are not picking up the call"*, *"Ye log phone bhi nhi utha te hai"*).
- **Structural Campus Unresolved Issue:** **`hostel_facilities`** (`0 / 8` confirmed resolutions; viral mess-hygiene reports trigger sarcasm and kettle workarounds without documented institutional fixes).
- **Contested Debate Domain:** **`placements`** has **`12` contradictory threads** (alumni vs. skeptical aspirants contesting median CTC and credibility claims).
- *Epistemic Caveat:* High `proposed_only` rates (`44.4%`) also reflect Reddit's observation-window censoring (students who silently resolve issues offline rarely post updates).

---

## Q8. How do discussions differ by topic, language and IIM?

1. **By Topic (BERTopic v1 vs. NMF v1):** Financial/loan topics (`BT-12`, `BT-13`) and faculty anxiety (`BT-16`) carry concentrated practical/procedural discussions, whereas waitlist clusters (`BT-09`, `BT-15`, `BT-20`, `BT-33`) generate high-volume neutral status queries (`curiosity`) punctuated by frustration spikes when deadlines pass silently.
2. **By Language Register:**
   - **English (`81.1%`, avg `17.4` words):** Higher actionable solution rate (`17.7%`), predominantly neutral (`55.2%`) and curious (`32.2%`), providing structured comparisons and policy references.
   - **Hinglish / Romanized Hindi (`10.1%`):** Shorter conversational reactions, **nearly `2×` higher negative sentiment (`45.9%` vs. `23.6%`)**, `50.0%` frustration rate, and distributed across **21 BERTopic themes** (vs. collapsed 77% into NMF Topic 6).
   - **Terse / Undetermined (`8.8%`):** Waitlist numbers (`"3005"`), academic profiles (`"94/93.2/9.6"`), and emojis (`"🤣🤣🤣"`) — cleanly abstained by Phase 2 LID.
3. **By IIM (`19` distinct IIMs detected via Cross-IIM analyzer):**
   - **IIM Sambalpur** (`217` direct comment mentions, `239` thread mentions): Split evenly between `administrative_communication` (`44.2%`) and `placements` (`43.3%`), `37.8%` negative, highest community solution rate (`28.6%`).
   - **IIM Mumbai (`35`) & IIM Sirmaur (`30`):** Led by `administrative_communication` (`57.1%` / `56.7%`) with `30–31%` negativity and `23%` solution rate.
   - **New/Peer IIMs (`IIM Jammu` `18`, `IIM Visakhapatnam` `16`, `IIM Raipur` `12`, `IIM Udaipur` `12`, `IIM Kashipur` `10`):** Led overwhelmingly by **`placements`** (`44%–100%` of mentions), with **IIM Udaipur** showing the highest positive share (`41.7%`) and **`confidence`** as its dominant emotion, and **IIM Raipur** showing the lowest negativity (`16.7%`). *(Small sample caveats apply for institutes with `< 20` mentions.)*

---

## Q9. How reliable are the AI predictions?

**Strictly Verified Human-Reviewed Reliability Ledger (Zero Fabricated Numbers, Zero Test Peeking):**

| RCI Subsystem | Human-Reviewed Benchmark | Strict Unbiased Headline Metric | Key Reliability Caveat |
|---|---|---|---|
| **5-Task Supervised Classifiers (Strict OOF Selection)** | `n = 92` frozen real test (`37` threads, `0/44` DAPT overlap) | **`64.62%` 3-seed mean Macro F1** (`65.01%` V6.0 pure neural; `64.38%` strict Stage 3) vs. V5 `59.48%` & LogReg `47.86%`. Significant gains on **Issue (`60.83%`, `p=0.992`)** & **Solution (`85.88%`, `p=0.999`)**. | Compact `n=92` test gives `~16–20 pp` CIs on rare classes (`stress`, `counterargument`, `extracurriculars`). Post-hoc `65.59%` V6.1 is explicitly flagged as test-inspected, NOT unbiased. |
| **Multilingual LID & Spans (Phase 2)** | `n = 84` manually reviewed real comments | **`88.10%` accuracy (`85.29%` Macro F1)**; **`95.83%` selective accuracy** when non-abstained; **`100.0%` script accuracy**; **`95.52%` code-mix F1**; **`90.91%` multi-span F1**. | Real Bengali/Odia script recall unmeasured due to `0` native `bn`/`or` script comments in corpus (synthetic probes `100%` reported separately). |
| **Dual Topic Discovery (NMF + BERTopic)** | `n = 3,459` identical docs + `35`-cluster human review (`105` centroid rows) | BERTopic v1: `35` human-reviewed themes (**`4.51/5.0` mean coherence**; `13` substantive, `12` logistics, `9` conversational, `1` URL artifact), Diversity `0.9629`, Balance `0.9509`, `80%` subsample NMI `0.6251`; NMF v1 preserved (`10` factors, `0%` outliers). | UMAP+HDBSCAN `eom` shows seed sensitivity (`seed=202` collapse); hence NMF remains default baseline. |
| **Contextual Issue–Solution Mapping (Phase 4)** | `n = 64` manually graded real pairs (`8` issues) | Pairwise F1 **`86.57%`** (`+6.57 pp` vs. V5 TF-IDF `80.00%`), **`P@3 = 100.0%`**, **`MRR = 1.00`**, FPR **`13.33%`** (`−3.34 pp`). | Slight graded NDCG@3 tradeoff (`0.837` vs `0.874` TF-IDF) between `gold=1` and `gold=2`. |
| **Explainable Resolution Engine (Phase 5)** | `n = 40` manually reviewed real threads | 4-state accuracy **`80.00%`** (`Macro F1 82.52%`), **False Resolution Claim Rate `0.00%` (`0/33`)**, Unresolved F1 **`98.51%`**, Evidence Retrieval **`87.50%`**. | Cannot observe offline resolutions never posted back to Reddit (`proposed_only` censoring). |
| **Human-in-the-Loop Triage** | All `3,557` corpus comments | Mean calibrated confidence surfaces low-confidence ambiguous replies (`<55%`) directly to Annotation Studio for human adjudication. | Model-inferred labels are never treated as human-verified ground truth. |
