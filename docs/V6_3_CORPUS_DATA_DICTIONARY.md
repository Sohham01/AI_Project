# V6.3 — Expanded Corpus Data Dictionary (`data/corpus_v6_3/new_unannotated_corpus_v6_3.json`)

This dictionary documents the schema of the V6.3 real-world Reddit expansion corpus (`corpus_version = "6.3.0"`, `451` newly collected threads, `4,013` comments), stored separately from the immutable `v6.1.0` baseline corpus (`data/live_corpus/corpus.json`, `365` threads).

---

## 1. Top-Level Object

| Field | Type | Description |
|---|---|---|
| `metadata` | `object` | Versioning, governance, provider, split exclusion count, and UTC timestamp bounds (`min`, `max`). |
| `posts` | `list[PostRecord]` | Chronologically sorted list of pseudonymized, deduplicated public Reddit threads (`n = 451`). |

### 1.1 `metadata` Fields

| Field | Example Value | Meaning |
|---|---|---|
| `dataset_id` | `"rci-real-reddit-expansion-v6.3"` | Canonical identifier distinguishing the expansion dataset from `v6.1.0` production corpus. |
| `corpus_version` | `"6.3.0"` | Semantic version of the expansion release. |
| `annotation_status` | `"unannotated"` | Explicit governance flag: threads carry **no** supervised gold labels until Round-2 human annotation finishes. |
| `synthetic_only` | `false` | Confirms `0` synthetic or LLM-generated conversations. |
| `source_provider` | `"arctic_shift_public_archive"` | Read-only public Reddit archive queried ethically without credentials or private endpoints. |
| `baseline_reference` | `"data/live_corpus/corpus.json (365 threads, untouched)"` | Records that the `v6.1.0` corpus was left immutable. |
| `total_new_threads` | `451` | Count of new threads passing dedup, scope, comment, and leakage checks. |
| `total_new_comments` | `4013` | Count of valid cleaned comments across all `451` new threads. |
| `dapt_and_train_excluded_threads` | `0` | Count of threads quarantined due to near-duplicate similarity with the 37 frozen test threads. |
| `date_range_utc` | `{"min": "2025-03-06T05:39:45Z", "max": "2026-11-25T06:53:48Z"}` | Earliest and latest UTC timestamps among newly collected threads. |

---

## 2. `PostRecord` Schema (Each Element in `posts`)

| Field | Type | Nullable | Description |
|---|---|---|---|
| `reddit_id` | `string` | No | Deterministic opaque surrogate thread ID (`post_63_<12-hex>`) derived from salted SHA-256 of raw archive ID. Raw Reddit base36 IDs and URLs are never stored. |
| `subreddit` | `string` | No | Source public subreddit name without `r/` prefix (e.g., `MBAIndia`, `IIMCATPreparation`, `kolkata`, `Bhubaneswar`, `westbengal`). |
| `title` | `string` | No | PII-redacted thread title text. |
| `body` | `string` | No | PII-redacted selftext body (`""` for title-only posts). |
| `created_at` | `string` | No | ISO-8601 UTC submission timestamp (e.g., `"2025-07-07T08:40:06Z"`). |
| `score` | `integer` | No | Public post score observed at collection time. |
| `num_comments` | `integer` | No | Count of retained valid comments stored inside `comments` after bot/duplicate/removed filtering. |
| `annotation_status` | `string` | No | Fixed `"unannotated"` in raw expansion export. |
| `dapt_and_train_excluded` | `boolean` | No | `true` if quarantined because of overlap/similarity with the frozen test split (`false` for all `451` threads). |
| `provenance` | `object` | No | Structured source lineage (`source_provider`, `corpus_version`, `search_query`, `source_id_sha`). |
| `comments` | `list[CommentRecord]` | No | List of cleaned, PII-redacted, non-bot comments belonging to this thread. |

### 2.1 `PostRecord.provenance` Fields

| Field | Type | Description |
|---|---|---|
| `source_provider` | `string` | `"arctic_shift_public_archive"` |
| `corpus_version` | `string` | `"6.3.0-expansion"` |
| `search_query` | `string` | Search context token (e.g. `"hostel"`, `"loan"`, `"college"`) or `"<direct>"` for subreddit direct listing. |
| `source_id_sha` | `string` | 24-hex opaque provenance digest (cannot be reversed to a Reddit URL or username). |

---

## 3. `CommentRecord` Schema (Each Element in `PostRecord.comments`)

| Field | Type | Nullable | Description |
|---|---|---|---|
| `reddit_id` | `string` | No | Deterministic opaque surrogate comment ID (`cmt_63_<14-hex>`). |
| `parent_reddit_id` | `string` | Yes | Opaque surrogate `cmt_63_*` ID of parent comment if replying to a comment, or `null` when replying directly to the post OP (`is_direct_reply_to_op`). |
| `parent_context_status` | `string` | No | `"direct_reply_to_op"`, `"verified_parent_comment"`, or `"orphaned_parent_fallback"` when a parent comment was filtered/removed upstream. |
| `body` | `string` | No | PII-redacted, bot-filtered, non-empty comment text. |
| `created_at` | `string` | No | ISO-8601 UTC timestamp of comment creation (`""` only if missing upstream). |
| `score` | `integer` | No | Public comment vote score at collection time. |
| `language` | `string` | No | Automated **provisional hypothesis** from `IndianRedditLID` (`en`, `hi-en`, `hi`, `bn-en`, `bn`, `or-en`, `und`). **Never** treated as human ground truth. |
| `script` | `string` | No | Detected script class (`latin`, `devanagari`, `bengali`, `odia`, `mixed`, `unknown`). |
| `is_code_mixed` | `boolean` | No | Automated indicator (`true` for `hi-en`, `bn-en`, `or-en`, or multi-script blends). |
| `has_multi_language_spans` | `boolean` | No | `true` when sentence/clause span segmentation detects switching across >= 2 languages within the same comment. |

---

## 4. Privacy & Compliance Guarantees
- **Banned Fields (Verified Absent):** `author`, `author_fullname`, `permalink`, `url`, raw `t1_*` / `t3_*` IDs, `/u/` handles, emails, phone numbers, and raw API headers.
- **Verified By:** [`backend/tests/test_dataset_expansion_and_leakage_v63.py`](../backend/tests/test_dataset_expansion_and_leakage_v63.py) scanning all `451` posts and `4,013` comments directly on disk.
