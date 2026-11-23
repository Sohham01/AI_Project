# Deliverable 1 — Real-Life Scenario Video Package (Script, Storyboard & Recording Plan)

> **Document ID:** `RCI-V6.2-D1` · **Navigation:** [Submission Index](./README.md) · [Deliverable 2 (Dataset) →](./DELIVERABLE_2_DATASET_PACKAGE.md)  
> **Production Status Disclosure:** This package delivers the complete 3-minute narration script, scene-by-scene storyboard, live screen-recording shot list, and expected analytical outputs. **The actual MP4 video has NOT yet been recorded** and is ready for capture using OBS Studio / Loom against the live local Docker (`http://localhost:8000/app`) or deployed Vercel instance (`https://rci-ten.vercel.app/app/`).

---

## 1. Production Specifications

- **Target Duration:** `3 minutes 15 seconds` (within the 2–4 minute academic target)
- **Resolution & Audio:** `1920×1080` @ 30fps, clean voiceover narration, system zoom `100%` (desktop viewport `1440×900` inside Chrome)
- **Core Narrative Arc:** `Manual Chaos in Multilingual Reddit Threads` → `RCI V6 End-to-End Pipeline (ISSUE → DISCUSSION → RESPONSE → SOLUTION → RESOLUTION EVIDENCE)` → `Honest Evidence-Linked Community Intelligence`

---

## 2. Scene-by-Scene Storyboard & Screen-Recording Plan

| Scene | Time | Visual / Screen Target | Exact On-Screen Action | Expected Analytical Output Shown |
|---|---|---|---|---|
| **Scene 1: The Real-World Problem** | `0:00 – 0:35` | Slide / Split-screen showing raw Reddit thread excerpts (`cmt_00009`, `cmt_01610`, `cmt_01936`, `cmt_0299`) | Highlight English + Hinglish + Devanagari + Roman-Odia + Gen-Z slang (`mid`, `cooked`, `😭`) + nested replies where students ask about waitlists, fee refunds, and placements. | Shows why manual reading across 365 threads / 3,557 comments fails and why generic sentiment dashboards miss *solutions* and *unresolved failures*. |
| **Scene 2: Overview & Evidence-Linked Insight Explorer** | `0:35 – 1:05` | Live RCI Studio `#/overview` | Scroll Tally Strip (`365 threads`, `3,557 comments`, `20.0%` actionable replies, `46 / 365` explicit outcomes: `12 confirmed · 34 pending · 26 contradictory`). Click **"Inspect supporting evidence"** inside Card 1 & Card 2, and highlight **"Export brief (.md)"**. | Displays separated **Observation vs. Interpretation**, clickable `#/thread/<post_id>` badges, quoted comments, NMF/BERTopic tags, and limitations. |
| **Scene 3: Dual Topic Discovery (NMF vs. BERTopic)** | `1:05 – 1:40` | Live RCI Studio `#/topics` | Start on preserved **NMF v1 (10 factors)**. Click **"BERTopic v1 (Multilingual E5-Small)"** switcher. Show `BT-0..BT-13` orbs, filter by **Substantive (13)**, inspect human-reviewed `BT-12` (*Education Loans, CIBIL Score & Fee Financing*, `5/5`) & `BT-13` (*Semester Fee Payment & Refund Deadlines*, `5/5`). Click **"Outliers -1 (1,320)"** and **"NMF vs. BERTopic"** comparison table. | Proves NMF remains intact alongside BERTopic's 35 human-reviewed semantic clusters (`4.51/5.0` mean coherence), explicit `-1` outlier preservation, and 21-topic Hinglish spread. |
| **Scene 4: Multilingual Span LID & Context Slang Lab** | `1:40 – 2:05` | `#/thread/post_0014` & `#/compare?tab=analyzer` | Open a multilingual thread showing parent-reply links and span badges; switch to `#/compare?tab=analyzer` and click **"Context Guard (Mid-Sem vs. Mid)"**. | Demonstrates live span-level language/script chips and context-aware slang disambiguation (`mid-semester` and `based on` preserved literally while `cooked` expands to evaluative distress). |
| **Scene 5: Issue → Solution → Unresolved Evidence Chain** | `2:05 – 2:45` | `#/issues` → `#/resolution` | Click `administrative_communication` in Issues Deep Dive to show Multilingual-E5 semantic solution clusters (`297` matched links). Switch to `#/resolution` (**V6 Explainable Engine** active by default) to inspect the ruled 5-cell verdict strip, **Contradiction Triage Queue** (`26` threads, *"Ye log phone bhi nhi utha te hai"*, `0.0%` false resolution claims), and click **"Jump to evidence →"**. | Demonstrates the project's central innovation: separating proposed peer advice (`162` threads) from verified success (`12` threads) and procedural failures (`34` pending), with one-click jump directly to the highlighted evidence comment. |
| **Scene 6: Strict OOF Benchmarks & Dual-Theme Craft** | `2:45 – 3:15` | `#/models` (+ Daylight mode toggle) | Highlight **Pipeline Integrity (7/7)**, **Strict OOF Evaluation Governance** (`64.62%` 3-seed / `65.01%` V6.0 strict OOF vs. V5 `59.48%`, flagging post-hoc `65.59%`), 4-field Data Provenance Authorizations, and toggle **Daylight mode** briefly. | Closes on rigorous academic trustworthiness, reproducible zero-leakage benchmarks, transparent limitations, and publication-grade SaaS UI craft. |

---

## 3. Full Voiceover Narration Script (~460 Words, ~3 min 15 sec)

> **[0:00 — Scene 1: The Problem]**  
> Every admissions cycle, thousands of IIM aspirants and students turn to Reddit to discuss waitlists, placements, education loans, fee refunds, and campus life. But these discussions are messy: a single thread blends English, Romanized Hinglish, Devanagari, regional languages like Odia, Gen-Z slang, and emojis—nested dozens of replies deep. Reading these threads manually is slow, and ordinary sentiment dashboards only tell you whether students sound upset—not *what problems keep recurring*, *what solutions peers propose*, or *whether those solutions actually worked*.
>
> **[0:35 — Scene 2: Overview & Evidence-Linked Insights]**  
> Meet **Reddit Community Intelligence**, or **RCI**—an end-to-end multilingual NLP research platform built around one core chain: **Issue, to Discussion, to Community Response, to Solution, to Resolution Evidence**. On our Overview Studio, RCI synthesizes 365 anonymized threads and 3,557 comments into evidence-linked cross-model insights. Every insight separates factual corpus observations from cautious interpretations and expands to reveal the exact supporting threads, quoted comments, calibrated confidence scores, and evidence limitations.
>
> **[1:05 — Scene 3: Dual Topic Discovery — NMF & BERTopic]**  
> In the Topic Map, we preserve our deterministic **NMF v1** baseline alongside our new **BERTopic v1** companion powered by `intfloat/multilingual-e5-small`. Switching to BERTopic reveals 35 naturally discovered themes—such as CIBIL education loans, registration refund deadlines, and B-school placement comparisons—while transparently preserving 1,320 outlier comments in topic minus-one and comparing both methods side-by-side on coherence, diversity, and balance.
>
> **[1:40 — Scene 4: Multilingual Spans, Slang & Conversation Context]**  
> Because student comments frequently switch languages mid-comment, RCI separates language from script at the span level—achieving 88.1% real-corpus accuracy and 95.8% selective accuracy when confident—and applies context-aware slang and emoji analysis that expands Gen-Z *"mid"* in *"mess food is mid"* while leaving *"mid-sem"* untouched, always preserving the original text.
>
> **[2:05 — Scene 5: Issue–Solution Mapping & Unresolved Issue Detection]**  
> The heart of RCI connects Issues to Solutions and Resolution Evidence. Our Contextual Multilingual Issue–Solution Mapper combines E5 embeddings, TF-IDF, parent-reply context, and a negation penalty—boosting pairwise F1 by 6.6 percentage points over TF-IDF alone to 86.6%, linking 951 actionable recommendations into 282 semantic solution clusters. Then our 8-signal Explainable Resolution Engine tracks contradictory follow-ups—like students reporting admissions offices not answering calls or emails—achieving 0% false resolution claims on our human-reviewed benchmark and exposing why administrative communication and hostel issues remain largely unresolved despite active peer advice.
>
> **[2:45 — Scene 6: Strict Evaluation & Scientific Integrity]**  
> Finally, our Models & Data vault enforces strict scientific integrity: zero test-thread DAPT leakage, strict out-of-fold selection reporting 64.62% three-seed Macro F1 over our 59.48% V5 baseline—transparently flagging any post-hoc test-inspected diagnostic—plus four-field data provenance authorizations. RCI turns noisy multilingual Reddit threads into auditable, evidence-backed community intelligence.
