# V6.3 — Real Dataset Expansion & Community Rebalancing Report

> **Dataset Integrity & Split Boundary:** `data/live_corpus/corpus.json` (`v6.1.0`, `365` threads, `3,557` comments) remains **100% immutable** so all deployed V6.1/V6.2 studio artifacts stay reproducible. All `451` newly collected public Reddit threads (`4,013` comments) are stored in an isolated, versioned file [`data/corpus_v6_3/new_unannotated_corpus_v6_3.json`](../data/corpus_v6_3/new_unannotated_corpus_v6_3.json) (`corpus_version = "6.3.0"`, `annotation_status = "unannotated"`, `synthetic_only = false`), keeping the `92`-example frozen test split (`37` unseen threads) untouched with **0 overlapping threads**.

---

## 1. Before vs. After Corpus Scale (`v6.1.0` → `v6.3.0`)

| Metric | Baseline (`v6.1.0`) | Newly Collected (`v6.3.0` Unannotated) | Combined Research Corpus (`v6.3.0`) | Net Growth |
|---|---:|---:|---:|---:|
| **Unique Pseudonymized Threads** | `365` | `+451` | **`816`** | **`+123.56%`** |
| **Total Human Comments** | `3,557` | `+4,013` | **`7,570`** | **`+112.82%`** |
| **Mean Comments / Thread** | `9.75` | `8.90` | `9.28` | — |
| **Synthetic / Fabricated Rows** | `0` | `0` | `0` | `—` |
| **Frozen Test Thread Overlap (`n=37`)** | N/A | `0` | `0` quarantined | `0%` leakage |
| **UTC Thread Creation Window** | `2023-04-05` to `2026-11-04` | `2025-03-06` to `2026-11-25` | `2023-04-05` to `2026-11-25` | Reaches `2026-11-25T06:53:48Z` |

Collection script: [`backend/expand_real_corpus_v63.py`](../backend/expand_real_corpus_v63.py) (using the read-only keyless Arctic Shift public Reddit archive `/api/posts/search` and `/api/comments/tree`).  
Machine audit artifact: [`ml/artifacts/v6_3_experiments/dataset_expansion/dataset_expansion_audit.json`](../ml/artifacts/v6_3_experiments/dataset_expansion/dataset_expansion_audit.json).

---

## 2. Community / Subreddit Distribution Before & After Expansion

A primary weakness in `v6.1.0` was over-concentration in a single institutional forum (`r/IIMSambalpurUG` = `211 / 365 = 57.81%`). V6.3 prioritized MBA/CAT academic, admissions, financial-aid, and regional eastern-India student forums while intentionally collecting `0` additional `r/IIMSambalpurUG` threads, cutting single-subreddit dominance in half (`57.81% → 25.86%`).

| Subreddit | Baseline Threads (`v6.1`) | Baseline Share | New Threads (`v6.3`) | Combined Threads (`v6.3`) | Combined Comments | Combined Thread Share |
|---|---:|---:|---:|---:|---:|---:|
| `r/IIMSambalpurUG` | 211 | 57.81% | 0 | **211** | 1,989 | **25.86%** *(rebalanced −31.95 pp)* |
| `r/MBAIndia` | 46 | 12.60% | +149 | **195** | 1,936 | 23.90% |
| `r/IIMCATPreparation` | 41 | 11.23% | +78 | **119** | 870 | 14.58% |
| `r/CATPrep` | 18 | 4.93% | +45 | **63** | 431 | 7.72% |
| `r/Indian_Academia` | 4 | 1.10% | +48 | **52** | 490 | 6.37% |
| `r/CATpreparation` | 12 | 3.29% | +38 | **50** | 562 | 6.13% |
| `r/CATStudyRoom` | 15 | 4.11% | +32 | **47** | 255 | 5.76% |
| `r/kolkata` *(new in V6.3)* | 0 | 0.00% | +24 | **24** | 358 | 2.94% |
| `r/Bhubaneswar` *(new in V6.3)* | 0 | 0.00% | +23 | **23** | 282 | 2.82% |
| `r/iim_CAT` | 18 | 4.93% | 0 | **18** | 143 | 2.21% |
| `r/westbengal` *(new in V6.3)* | 0 | 0.00% | +14 | **14** | 254 | 1.72% |
| **Total** | **365** | **100.00%** | **+451** | **816** | **7,570** | **100.00%** |

---

## 3. Language & Script Distribution (Real Collected Text)

All comments were automatically tagged by `detect_indian_reddit_language()` (`IndianRedditLID`) strictly as candidate metadata (never treated as gold evaluation ground truth). Adding eastern regional student/education discussions (`r/kolkata`, `r/westbengal`, `r/Bhubaneswar`) introduced genuine non-synthetic Bengali script, Banglish, and Roman-Odia discussions that were previously absent or nearly absent (`0` Bengali / `1` Roman-Odia in `v6.1.0`):

| Language Tag | New V6.3 Comments (`n = 4,013`) | New Share | Combined Corpus (`n = 7,570`) | Notes |
|---|---:|---:|---:|---|
| `en` (English) | 3,280 | 81.73% | 6,163 | Majority general MBA/university advice |
| `hi-en` (Hinglish) | 312 | 7.77% | 629 | Romanized Hindi–English code-mixing |
| `und` (Abstained / Terse) | 273 | 6.80% | 587 | Short numeric profiles, emojis, low-evidence utterances |
| `bn-en` (Banglish + Bengali mix) | **95** | **2.37%** | 95 | Genuine regional discussions (`0` in baseline `v6.1`) |
| `hi` (Romanized Hindi) | 21 | 0.52% | 63 | High-density Hindi without English anchors |
| `bn` (Bengali) | **18** | **0.45%** | 18 | Native Bengali script and dense Banglish (`0` in baseline) |
| `or-en` (Roman-Odia + English) | **14** | **0.35%** | 15 | Authentic Odisha student threads (`1` in baseline) |

### Script Distribution on Newly Collected V6.3 Comments (`n = 4,013`)
- `latin`: **3,960** (`98.68%`)
- `unknown` (emoji/numeric): **32** (`0.80%`)
- `bengali` (native Unicode block `U+0980–U+09FF`): **13** (`0.32%`)
- `mixed` (e.g., native Bengali + Latin script): **8** (`0.20%`)
- Native Odia script (`U+0B00–U+0B7F`): **0** documented cleanly (even with `r/Bhubaneswar` and `r/Odisha` queries, Odisha public Reddit discussions occur almost exclusively in English and Romanized Odia `or-en`).

---

## 4. Deduplication, Filtering & Collection Limitations

### 4.1 Deterministic Filtering Breakdown
Every raw API thread/comment passed through deterministic hygiene, pseudonymization, and deduplication checks (`dataset_expansion_audit.json`):
- **Bot / AutoModerator rejected comments:** `236`
- **Duplicate normalized comment texts rejected:** `182`
- **Threads rejected for zero valid surviving comments:** `52`
- **GIF / bare media URL comments rejected:** `17`
- **Off-topic scope rejections** (on non-MBA regional queries outside student/academic/career relevance): `9`
- **`[deleted]` / `[removed]` comments rejected:** `6`
- **Spam / promo rejections:** `1`
- **PII redactions triggered & verified clean:** `@email`, `/u/` usernames, phone numbers, raw base36 Reddit IDs (`t1_`, `t3_`), and `reddit.com` thread permalinks stripped via SHA-256 salted opaque IDs (`post_63_...`, `cmt_63_...`).

### 4.2 Documented Collection Limitations (No Bypassing Applied)
During collection on `2026-11-25`, several Arctic Shift query plans encountered legitimate public API rate limits (`HTTP 429`) or gateway query timeouts (`HTTP 422`), notably `r/Odisha` (`3` queries HTTP 429), `r/iim_CAT` (`1` query HTTP 429), `r/ipmat` (`1` query HTTP 429), `r/CATpreparation` (`1` timeout 422, `2` HTTP 429), `r/Indian_Academia` (`2` timeouts 422), `r/kolkata` (`2` HTTP 429), `r/Bhubaneswar` (`1` timeout 422), and `8` thread comment-tree fetches (`HTTP 429`). In accordance with platform limits and RCI governance rules, the collector logged every transient failure in `search_log` rather than circumventing rate limits or scraping restricted endpoints, still exceeding the target range (`816` combined threads vs `700–1,000` target).

---

## 5. Frozen Test Split Isolation & Leakage Audit
`expand_real_corpus_v63.py` loaded all `37` frozen test threads (`526` corpus comments) from `data/annotations/real_reddit_labeled_v1/master_labeled_v1.csv` and `v5_frozen_splits.json`, computing:
- Exact post ID checks against `baseline_ids` (`365`),
- Normalized title & body hash collision checks against all `37` frozen test threads,
- Near-duplicate bigram Jaccard similarity guards against frozen test titles/bodies.

**Result:** `threads_quarantined_from_dapt_train = 0`, `frozen_test_leakage_into_train_pool = 0`, and verified cleanly by [`backend/tests/test_dataset_expansion_and_leakage_v63.py`](../backend/tests/test_dataset_expansion_and_leakage_v63.py).
