# Deliverable 1 — Real-Life Scenario Video Production Script & Storyboard (2–4 Minutes)

> **Target Runtime:** `3 minutes 15 seconds` (strictly within the required **2–4 minute** window; PDF Task 2 & Sec. 19).  
> **Recording Setup:** Screen-record `http://localhost:8000/app/` (or live Vercel Studio `https://rci-ten.vercel.app/app/`) at `1920×1080` (100% browser zoom, Daylight or Dusk theme) with clean voiceover audio.

---

## Shot-by-Shot Storyboard & Timed Script

### Scene 1 — The Real-Life Problem: Analyst Information Overload (`0:00 – 0:50`)
- **Visuals:**
  - `0:00–0:20`: Show a split screen of raw, messy student Reddit threads (scroll rapidly through a spreadsheet/raw feed of `3,557` comments across `8` IIM/CAT subreddits). Highlight live on-screen callouts of the exact multilingual/informal messiness:
    - Hinglish + emoji: *"Bhai IIM ABC ka summer placement scene kaisa hai 😭"*
    - Odia + English script-mix: *"ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro"*
    - Romanized Banglish: *"IIM e giye ki actually ROI pawa jay?"*
    - Gen-Z slang + emoji: *"Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl."*
  - `0:20–0:50`: Show a hypothetical institutional analyst trying to tally concerns manually in Excel — then show `langdetect` failing (`60.6%` of code-mixed Indian comments tagged as Indonesian, Swahili, Somali, or Welsh!) and simple sentiment analysis failing to tell *which issues have solutions* vs. *which issues remain unresolved*.
- **Voiceover (~115 words):**
  > *"Imagine you are an institutional analyst at an IIM trying to understand what students and prospective applicants actually worry about. Conventional annual surveys are slow, formal, and sanitized. Meanwhile, thousands of candid discussions happen every week on Reddit — mixing English, Hindi, Hinglish, Banglish, Odia, emojis, and Gen-Z slang inside the same sentence. Reading 3,557 comments manually is impossible, and off-the-shelf NLP breaks immediately: standard language detectors mislabel over 60% of Indian code-mixed comments as Indonesian, Swahili, or Welsh, and basic sentiment tools can't tell a complaint apart from peer advice, a counterargument, or an actual solution."*

---

### Scene 2 — The Automated Multilingual AI Pipeline (`0:50 – 1:45`)
- **Visuals:**
  - `0:50–1:15`: Open **RCI Studio → Overview (`#/overview`)**. Hover over the interactive **Model Architecture & Inference Network** diagram to highlight the end-to-end flow:
    - *Anonymized Reddit Corpus (`365` threads, `3,557` comments, zero PII)* → *Language/Script & Emoji/Slang Normalization* → *Dual Strictly Inductive 5-Fold Encoders (`MuRIL-DAPT-WWM` + `XLM-RoBERTa-base`, zero test-thread leakage)* → *5 Calibrated NLP Task Heads + Unsupervised NMF + TF-IDF Cosine Issue–Solution Linker*.
  - `1:15–1:45`: Click **Compare & lab (`#/compare`) → Live Multilingual Analyzer tab**. Click two presets live:
    1. *"Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl."* → show instant detection of `hi-en`, `latin`, `💀 → extreme amusement disbelief`, `ngl → not gonna lie`, and live V6 corpus-neighbor predictions.
    2. *"ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro"* → show instant detection of `or-en`, `mixed` Odia+Latin script.
- **Voiceover (~125 words):**
  > *"To solve this, we built Reddit Community Intelligence — an end-to-end multilingual NLP studio that transforms raw, anonymized Reddit threads into structured institutional intelligence. Instead of stopping at sentiment, our pipeline models the full community chain: Issue, Discussion, Community Response, Linked Solution, and Resolution Evidence. First, our Indian-Reddit language and preprocessing layer preserves code-mixed syntax and converts emojis into semantic affect cues — lifting Transformer accuracy where conventional stop-word removal and stemming collapse performance. Next, our strictly inductive domain-adapted MuRIL and XLM-RoBERTa five-fold ensembles score every comment across eight issue domains, polarity, six emotions, six canonical response roles, and actionable solutions — verified with seven automated integrity checks."*

---

### Scene 3 — Discovering Top Issues, Community Solutions & Unresolved Gaps (`1:45 – 2:45`)
- **Visuals:**
  - `1:45–2:05`: Click **Overview (`#/overview`)** → show **Top Issues** & **Issue Friction Priority Matrix** (Admissions/Waitlist `1,995` mentions, Placements `732`, Curriculum `557`, Fees/ROI `406`), then briefly flash **Topic map (`#/topics`)** (10 unsupervised NMF clusters) and **Responses (`#/responses`)** (6 canonical classes in fixed order).
  - `2:05–2:30`: Click **Resolution (`#/resolution`)** → highlight:
    1. **Quantitative Resolution & Unresolved Issues Matrix (Task 11):** Show explicit formulas distinguishing `Partially Resolved` issues (Admissions, Placements, Fees/ROI) from **`Unresolved`** gaps (**Campus Infrastructure** and **Hostel Facilities** — low solution density, `0` verified follow-ups).
    2. **Linked Issue → Most Common Community Solutions (Task 10):** Click *Financial Concerns* and *Placements* to show ranked community solution clusters linked via `TfidfCosineIssueSolutionLinker` (`861` asserted links across `158` threads) — e.g., Education Loans, Scholarships, Alumni Verification, and Alternative B-School / CAT Retake comparisons.
  - `2:30–2:45`: Click **Compare & lab (`#/compare`) → Cross-IIM** and **Languages** tabs → highlight the real register divide: Hinglish comments are nearly **2× more negative (`45.9%` vs. `23.7%`)** while actionable solutions are **4× denser in English (`17.6%` vs. `4.1%`)**.
- **Voiceover (~135 words):**
  > *"In seconds, an analyst sees what would have taken weeks to read. Admissions and waitlist anxiety lead total volume, while Placements and Fees drive the highest negativity. Crucially, our pipeline separates proposed advice from verified outcomes: across 365 threads, our TF-IDF cosine linker connects 861 issue-solution pairs into ranked community themes — from waitlist portal tracking and education loans to alumni LinkedIn verification. Using explicit quantitative criteria, the dashboard flags Campus Infrastructure and Hostel Facilities as unresolved high-friction gaps with sparse community workarounds. Cross-language analytics even reveal a behavioral divide: students switch to Hinglish and emojis when venting frustration — nearly double the negativity of English — while structured peer solutions cluster predominantly in longer English replies."*

---

### Scene 4 — Scientific Rigor & Honest Closing (`2:45 – 3:15`)
- **Visuals:**
  - Click **Models & data (`#/models`)** → show the **4-Architecture 5-Fold Benchmark Dumbbell** (Deployed Neural beating Logistic Regression, ANN, and 1D TextCNN by `+15 to +23 percentage points` on unseen-thread real test comments), the **Strictly Inductive DAPT Perplexity Curve (`88.3 → 25.2`, zero test-thread leakage)**, **Calibration Reliability**, and the **7/7 Pipeline Integrity Banner** + honest **Human IAA Governance Gate**.
- **Voiceover (~65 words):**
  > *"Every number in RCI Studio is grounded in real, pseudonymized public data — zero synthetic shortcuts, zero test-thread leakage, calibrated probabilities, and transparent human agreement audits. With automated multilingual issue discovery, community-solution linking, and unresolved-gap tracking, institutions can listen to students at scale — accurately, ethically, and continuously."*

---

## Pre-Recording Checklist
1. Start local Docker Studio: `docker compose up -d` and open `http://localhost:8000/app/` (confirm green `API ready` pill).
2. Switch to **Daylight mode** (high contrast for YouTube/classroom projectors) or **Dusk dark mode** (clean modern look).
3. Keep browser at `1920×1080` (`100%` zoom), close DevTools/bookmarks bar, and pre-click `#/compare?tab=analyzer` once so the API cache is warm.
4. Keep total duration between **2:50 and 3:30**.
