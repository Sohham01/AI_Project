# V6.4 — Real IIM Subreddit Corpus Expansion Report

> **Dataset Integrity & Additive Versioning:** Both `data/live_corpus/corpus.json` (`v6.1.0`: `365` threads, `3,557` comments) and `data/corpus_v6_3/new_unannotated_corpus_v6_3.json` (`v6.3.0`: `451` threads, `4,013` comments) remain **100% immutable**. All **`+526` net-new real IIM threads (`+9,108` comments)** are isolated in [`data/corpus_v6_4/new_iim_corpus_v6_4.json`](../data/corpus_v6_4/new_iim_corpus_v6_4.json) (`corpus_version = "6.4.0"`, `annotation_status = "unannotated"`, `synthetic_only = false`), keeping the frozen `n = 92` test split (`37` unseen threads) untouched with **`0` overlapping threads**.

---

## 1. Corpus Progression (`v6.1.0` → `v6.3.0` → `v6.4.0`)

| Corpus Partition | Unique Threads | Human Comments | Mean Comments / Thread | Frozen Test Leakage |
|---|---:|---:|---:|---:|
| **Baseline (`v6.1.0`, Studio Production)** | `365` | `3,557` | `9.75` | Baseline reference |
| **V6.3 General MBA + Regional Expansion (`v6.3.0`)** | `+451` | `+4,013` | `8.90` | `0` quarantined / `0` leaked |
| **V6.4 Dedicated IIM & Campus Expansion (`v6.4.0`)** | **`+526`** | **`+9,108`** | **`17.32`** *(deeper IIM discussions)* | **`0` quarantined / `0` leaked** |
| **Combined Research Corpus (`v6.4.0` Total)** | **`1,342`** | **`16,678`** | `12.43` | `0%` frozen-test overlap |

Collection script: [`backend/expand_iim_corpus_v64.py`](../backend/expand_iim_corpus_v64.py) · Audit JSON: [`ml/artifacts/v6_4_experiments/iim_expansion/iim_expansion_audit.json`](../ml/artifacts/v6_4_experiments/iim_expansion/iim_expansion_audit.json) · UTC window of V6.4 batch: `2023-09-17T22:15:55Z` to `2026-11-25T09:14:13Z`.

---

## 2. Newly Collected V6.4 IIM Communities & Combined Distribution

### 2.1 New Threads Collected in V6.4 (`+526` Threads, `+9,108` Comments)

| Subreddit | New V6.4 Threads | New V6.4 Comments | Key Topics Covered in Queries |
|---|---:|---:|---|
| `r/CATpreparation` | `240` | `5,934` | `BLACKI`, `New IIM`, `IIM Ahmedabad`, `IIM Calcutta`, `IIM Udaipur`, `IIM Trichy`, `IIM Shillong` |
| `r/IPMATtards` *(new in V6.4)* | `215` | `2,069` | Direct IPM threads (`IIM Indore`, `hostel`, `fees`) across IPM/BBA+MBA IIM programs |
| `r/IIMCalcutta` *(new in V6.4)* | `38` | `489` | Direct campus/admissions threads on `r/IIMCalcutta` |
| `r/MBAIndia` | `30` | `611` | `IIM Mumbai`, `IIM Kozhikode` |
| `r/IIMLucknow` *(new in V6.4)* | `2` | `4` | Direct `r/IIMLucknow` threads |
| `r/iimmumbai` *(new in V6.4)* | `1` | `1` | Direct `r/iimmumbai` thread |

### 2.2 Combined 15-Community Balance (`1,342` Unique Real Threads)

Single-institution skew (`r/IIMSambalpurUG` originally `57.81%` in V6.1 → `25.86%` in V6.3) drops further to **`15.72%`** across `15` complementary communities:

| Subreddit | Combined Threads | Combined Share |
|---|---:|---:|
| `r/CATpreparation` | `290` | `21.61%` |
| `r/MBAIndia` | `225` | `16.77%` |
| `r/IPMATtards` | `215` | `16.02%` |
| `r/IIMSambalpurUG` | `211` | `15.72%` |
| `r/IIMCATPreparation` | `119` | `8.87%` |
| `r/CATPrep` | `63` | `4.69%` |
| `r/Indian_Academia` | `52` | `3.87%` |
| `r/CATStudyRoom` | `47` | `3.50%` |
| `r/IIMCalcutta` | `38` | `2.83%` |
| `r/kolkata` | `24` | `1.79%` |
| `r/Bhubaneswar` | `23` | `1.71%` |
| `r/iim_CAT` | `18` | `1.34%` |
| `r/westbengal` | `14` | `1.04%` |
| `r/IIMLucknow` | `2` | `0.15%` |
| `r/iimmumbai` | `1` | `0.07%` |

---

## 3. Language, Filtering & API Rate-Limit Transparency

- **Provisional `IndianRedditLID` Tags on V6.4 New Comments (`n = 9,108`):** `7,686` `en`, `721` `hi-en` (Hinglish), `607` `und`, `81` `hi`, `11` `bn-en`, `2` `or-en` (`9,040` `latin`, `67` `unknown`, `1` `mixed`).
- **Deterministic Hygiene & Scope Filtering (`iim_expansion_audit.json`):**
  - Comments rejected: `554` duplicate comment bodies across prior/current corpora, `243` bot/AutoMod messages, `43` GIF/bare media embeds, `5` `[deleted]`/`[removed]`, `2` promo/spam messages.
  - Threads rejected: `58` off-topic/non-IIM scope, `6` zero valid comments.
  - PII & URLs: `@email`, `/u/` usernames, phone numbers, raw base36 IDs, and Reddit thread permalinks stripped via salted opaque IDs (`post_v64_*`, `cmt_v64_*`).
- **Rate Limits Respected Cleanly:** Out of `25` search queries executed before reaching the `610`-candidate tree-fetch cap, `6` search plans (`r/IIMIndore`, `r/CATpreparation:Baby IIM` HTTP 422; `r/CATpreparation:IIM Bangalore` + `r/MBAIndia:{BLACKI, Baby IIM, IIM Shillong, IIM Udaipur}` HTTP 429) hit public Arctic Shift rate/timeout limits and were logged honestly in `search_log` rather than bypassing restrictions.
