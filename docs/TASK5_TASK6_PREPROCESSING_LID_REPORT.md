# Tasks 5 & 6 — Multilingual LID Limitation Study & Preprocessing / Emoji Ablation Report

> **Experiment Script:** `ml/experiments/v5_optimization/run_task5_task6_ablations.py`  
> **Machine-Readable Artifact:** `ml/artifacts/v5_experiments/task5_task6_ablations.json` (also bundled into `ml/artifacts/live_corpus/community_intelligence.json` and live in the Studio at `#/compare?tab=language`).  
> **Strict Inductive Guarantee:** All Task 6 model ablations run exclusively on **5-fold `post_id`-grouped OOF CV over untouched V1 `trainval (n = 367)`** — the frozen `n = 92` held-out test set was never touched.

---

## 1. Task 5 — Limitations of Existing Language Identification (`langdetect`) on Indian Student Reddit Data

Problem Statement Task 5 (pp. 7–8) asks us to identify **5 languages** (`English`, `Hindi`, `Bengali`, `Odia`, `Other`) and **5 scripts** (`Roman/Latin`, `Devanagari`, `Bengali`, `Odia`, `Other`), recognize Romanized code-mixed Indian languages (e.g., *"IIM jaana worth it hai kya?"*), and **empirically investigate the limitations of existing LID models**.

### 1.1 Empirical Failure Audit on All `3,557` Anonymized Live Reddit Comments

| Metric | `langdetect 1.0.9` (Off-the-Shelf) | `IndianRedditLID` (`indian_reddit_lid.py`) |
|---|---:|---:|
| **Bogus non-Indian language tags (full `3,557` corpus)** | **`815 / 3,557` (`22.9%`)** *(id, sw, so, et, tl, cy, vi, hu, fi, nl, tr…)* | **`0 / 3,557` (`0.0%`)** |
| **Bogus non-Indian tags on the `340` Code-Mixed / Roman-Indic slice** | **`206 / 340` (`60.6%`)** | **`0 / 340` (`0.0%`)** |
| **Code-Mixed / Roman-Indic comments silently mislabeled as monolingual `en`** | **`134 / 340` (`39.4%`)** | **`0 / 340` (`0.0%`)** |
| **Code-Mixed / Roman-Indic comments recognized as Indic/Code-Mixed** | **`0 / 340` (`0.0%`)** | **`340 / 340` (`100.0%`)** |
| **Controlled 12-Item Multilingual Probe Suite Accuracy** *(all 5 PDF langs & scripts)* | **`33.3%` (`4 / 12`)** | **`100.0%` (`12 / 12`)** |

### 1.2 Why Off-the-Shelf Character N-Gram LID Fails on Indian Code-Mixed Reddit
1. **Romanized script conflation:** `langdetect` assumes Latin script implies a European/Austronesian/African language trained on Wikipedia character trigrams — so Romanized Hindi (`"hai"`, `"bhai"`, `"kya"`, `"nahi"`, `"toh"`, `"karo"`) gets systematically mapped to **Indonesian (`id`: 54 hits)**, **Swahili (`sw`: 39)**, **Somali (`so`: 24)**, **Estonian (`et`: 19)**, **Tagalog (`tl`: 12)**, and **Welsh (`cy`: 11)**.
2. **No code-mix representation:** Single-label LID models cannot emit bilingual tags (`hi-en`, `bn-en`, `or-en`), forcing mixed Hindi-English sentences like *"IIM jaana worth it hai kya?"* into either erroneous `en` or bogus foreign codes.
3. **Short-text instability:** Terse MBA replies (`"98.4%ile GEM"`, `"WL 12"`, `"CFBR"`, `"Congrats!"`) lack sufficient n-gram statistics, triggering random European labels.

### 1.3 Representative Real Misclassifications Caught by Our Audit
- `langdetect` → `id` / `sw` / `so` / `cy` on genuine Hinglish comments containing unambiguous markers (`bhai`, `hai`, `nahi`, `kya`, `toh`, `karna`); `IndianRedditLID` cleanly identifies `hi-en` (`script = latin`, `is_code_mixed = true`).
- On the PDF Page 1 & 7 multilingual probes:
  - *"Bhai IIM ABC ka summer placement scene kaisa hai 😭"* → `hi-en`, `latin`, `code_mixed = True` (`langdetect`: `id` / wrong)
  - *"ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro"* → `or-en`, `mixed`, `code_mixed = True` (`langdetect`: wrong / fails Odia)
  - *"IIM e giye ki actually ROI pawa jay?"* → `bn-en`, `latin`, `code_mixed = True` (`langdetect`: wrong)
  - *"Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl."* → `hi-en`, `latin`, `code_mixed = True`

---

## 2. Task 6 — Controlled Empirical Preprocessing, Gen-Z Slang & Emoji Ablation

Problem Statement Task 6 (pp. 8–9) explicitly asks two empirical questions:
1. **Does conventional preprocessing (`stop-word removal + stemming + punctuation removal`) actually help?**
2. **Should emojis be removed or converted into semantic representations (`😭 → sadness/frustration`, `🔥 → strong approval`, `💀 → extreme amusement/disbelief`)?**

### 2.1 Controlled 5-Fold Group-Aware OOF CV Results (`n = 367`, Zero Test Leakage)

| Preprocessing Strategy | Word+Char TF-IDF *(5-Task Mean OOF F1)* | `MuRIL-DAPT-WWM` Sentiment OOF F1 | `MuRIL-DAPT-WWM` Emotion OOF F1 *(raw 0.50)* | `MuRIL-DAPT-WWM` Affect Mean | Verdict |
|---|---:|---:|---:|---:|---|
| **`raw`** *(whitespace only)* | `45.66%` | `62.41%` | `31.22%` | `46.81%` | Baseline |
| **`conventional_nlp`** *(stop-word + stemming + strip punct/emoji)* | **`44.26%`** *(-1.40 pp)* | **`53.34%`** *(-9.07 pp)* | **`11.60%`** *(-19.62 pp)* | **`32.47%`** *(-14.34 pp)* | **Harmful (Reject)** |
| **`emoji_removed`** *(strip emojis only)* | `45.66%` *(0.00 pp)* | `62.18%` *(-0.23 pp)* | `30.95%` *(-0.27 pp)* | `46.57%` *(-0.24 pp)* | **Hurts affect** |
| **`emoji_token_slang`** *(`<EMOJI_name>` + conservative slang)* | **`46.41%`** *(+0.75 pp)* | `62.48%` *(+0.07 pp)* | `30.55%` *(-0.67 pp)* | `46.52%` *(-0.29 pp)* | Best for TF-IDF |
| **`emoji_affect_slang`** *(semantic affect phrases + Gen-Z slang)* | `46.17%` *(+0.51 pp)* | **`64.68%`** *(**+2.27 pp**)* | **`32.45%`** *(**+1.23 pp**)* | **`48.57%`** *(**+1.76 pp**)* | **Clear Neural Winner** |

### 2.2 Three Evidence-Backed Design Conclusions
1. **Conventional NLP (`stop-word removal + stemming + punctuation removal`) severely degrades performance:** It drops classical TF-IDF by **`-1.40 pp`** and collapses `MuRIL-DAPT-WWM` Affect F1 by **`-14.34 pp`** (`-9.07 pp` Sentiment, `-19.62 pp` Emotion). Removing stop words strips critical negation (`not`, `no`, `never`, `don't`), stripping punctuation destroys interrogative (`?` → `curiosity`) and emphasis (`!!`) signals, and English stemming corrupts Hinglish morphemes.
2. **Emojis should NEVER be stripped — and natural-language semantic affect verbalization beats raw emoji characters and synthetic `<EMOJI_...>` tags on Transformers:** `google/muril-base-cased`'s WordPiece vocabulary maps raw Unicode emojis (`😭`, `💀`, `🤡`) to `[UNK]` and fragments `<EMOJI_loudly_crying_face>` into awkward punctuation subwords. Converting emojis directly into natural-language affect phrases (`😭 → sadness frustration`, `💀 → extreme amusement disbelief`, `🤡 → sarcasm mockery`, `🔥 → strong approval`) maps them straight onto MuRIL's pretrained semantic subspace, lifting **5-fold OOF Sentiment by `+2.27 pp`** and **Emotion by `+1.23 pp`**!
3. **Conservative, Corpus-Verified Gen-Z Slang Handling:** Unambiguous internet abbreviations (`idk`, `ngl`, `fr`, `tbh`, `imo`, `btw`, `cfbr`) are expanded; tone-carrying Indian/peer terms (`bro`, `bhai`, `yaar`, `lol`) and polysemous MBA words (`mid` in *"mid 30s"* / *"mid-tier"*, `based` in *"based on placement network"*, `cooked`) are **intentionally preserved** rather than blindly overwritten (and surfaced transparently in metadata via `detect_slang_terms()`).
