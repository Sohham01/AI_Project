# Response Classification — Annotation Guide

**Version:** 1.1.0
**Applies to:** the six-class V1 response taxonomy
(`agreement`, `personal_experience`, `opinion`, `advice`, `solution`,
`counterargument`)
**Canonical code:** `app/services/nlp/response_classification/models/labels.py`
**Review cadence:** bump the version below whenever a boundary rule changes.

> This guide is the human decision procedure. The code only *enforces* the
> taxonomy; it never decides an ambiguous case for the annotator.

---

## 1. What you are annotating

You are labelling each Reddit **comment** with its **single primary role** in
the discussion it is replying to. The task is single-label and the six classes
are **mutually exclusive**: pick exactly one. If a comment seems to fit two
classes, choose the one that dominates the *reason the comment was written*,
and record the difficulty in the `notes` column so the guideline can be
sharpened.

The classes are **not** sentiment. "This is terrible" is a *counterargument*
or *opinion*, not a negative-sentiment label.

---

## 2. The six labels

| Code | Label | One-line meaning |
| --- | --- | --- |
| A | `agreement` | Endorses what the parent or another commenter said. |
| B | `personal_experience` | Reports the writer's own past/lived experience. |
| C | `opinion` | States a belief or judgement, without lived evidence. |
| D | `advice` | Recommends an action to somebody else. |
| E | `solution` | Presents the answer or fix outright. |
| F | `counterargument` | Pushes back on the claim being discussed. |

Canonical order and spelling (copy exactly, all lower-case, underscore
separated): `agreement`, `personal_experience`, `opinion`, `advice`,
`solution`, `counterargument`.

---

## 3. Operational decision procedure

Apply these steps in order:

1. Read the comment and parent; use post title/thread as additional context.
2. If text is missing, truncated, unintelligible, or parent context needed for interpretation is unavailable, leave `response_type` blank for review/adjudication. Do not guess `opinion` or infer a response role from sentiment.
3. Does it endorse a prior claim without a substantive personal account? `agreement`. Does it resist/correct/qualify an identifiable prior claim? `counterargument`.
4. Otherwise classify the main contribution: the writer's own lived event = `personal_experience`; belief/judgement = `opinion`; recommendation or actionable fix = `advice`/`solution` (boundary below).
5. For mixed roles, choose the main contribution. If evidence cannot support a consistent choice, leave blank for adjudication; never invent another label.
6. Classify understood meaning, not language. For short/code-mixed text that cannot be understood reliably, leave blank and ask a fluent reviewer.

### 3.1 Advice vs. solution

Both can be imperative; wording alone does not decide. **`advice`** points toward a next step but leaves a key choice, details, or answer unresolved: “contact the placement office”; “try LinkedIn”; “maybe ask seniors”. **`solution`** supplies a concrete, issue-matched answer that can be acted on from the provided details: “Email the registrar your roll number and fee receipt to correct the portal entry”; “Use the scholarship form linked in the pinned post and submit it before 31 March”. “Apply for a scholarship” without a specific process is usually advice. “I emailed them and it worked” is personal experience unless giving the fix is the main contribution. If missing context determines whether an action is a specific fix, leave blank.

### 3.2 Opinion vs. counterargument

**`counterargument`** requires an identifiable prior claim and a reply that rejects, corrects, qualifies, or contests it; proof is not required. “I disagree—the fees are fair” is counterargument when replying to a claim that fees are unfair. **`opinion`** is a judgement not clearly positioned against a specific prior claim (“ROI seems poor to me”). Negative tone alone is not counterargument. Use opinion only when a judgement is actually expressed. Sarcasm is labelled by implied stance only when context makes that stance clear; otherwise leave blank.

### 3.3 Agreement vs. personal_experience

**`agreement`** is endorsement without a meaningful first-person account (“same bro”, “+1”, “exactly”). **`personal_experience`** reports the writer's own lived event as the main contribution (“I faced this last semester and the office fixed it after two days”). When both appear, ask what the reader mainly learns: substantive lived detail = personal experience; incidental anecdote with endorsement as point = agreement.

## 4. Short, neutral, and non-role comments

“same bro”, “this”, “+1”, “exactly” are agreement when they clearly endorse the parent. “ok”, “thanks”, “lol”, emoji-only acknowledgements, and off-topic jokes remain blank unless context clearly shows one of the six roles. Do not force neutral replies into opinion. Do not add “other”, “neutral”, “unclear”, or “not applicable”; blank means unlabelled/requires review, not a seventh class. Mixed-role comments use the dominant contribution; unresolved ambiguity remains blank.

## 5. Context and language

Read the parent comment before labelling. For a top-level comment, use the post title/body context where available. If parent context is absent/truncated, label only if the role is clear without it (for example, an explicit first-person account); otherwise leave blank for adjudication. The CSV has no confidence field: do not encode confidence elsewhere.

For English/Hindi/Bengali/Odia code-mixed and Romanized replies, classify the understood conversational function, not language or script; do not translate into a changed claim. If a short/mixed-language comment cannot be understood reliably, leave blank and request a fluent reviewer.



## 7. What the export gives you

`export_annotation_candidates.py` writes
`data/annotations/response_candidates.csv` with these columns:

`document_id, text, normalized_text, response_type, post_id, comment_id,
parent_comment_id, parent_text, post_title, subreddit, score, created_at,
source_url, source_provider, source_mode, parent_context_status, language,
script, is_code_mixed, detected_languages`

- `text` = the **original** comment, never modified.
- `normalized_text` = the preprocessing pipeline output (what the model sees).
- **`response_type` ships blank.** Fill it with exactly one of the six labels.
- **Never** paste a model prediction into `response_type`; only human
  judgement is ground truth.
- A missing score stays blank; a real score of `0` stays `0`. Invalid supplied
  scores or boolean metadata are rejected rather than silently coerced.
- `source_url`, `source_provider`, `source_mode`, and
  `parent_context_status` preserve review provenance when available. Missing
  context is a warning for human review, not a value to invent.
- `document_id` and `comment_id` must be unique within the candidate batch;
  shared `parent_comment_id` values are legitimate.
- Candidate writing validates before an exclusive create and never overwrites an
  existing destination. Use a bounded, already-authorized static source export;
  do not use the unbounded database helper for a pilot.

### From filled candidates to a training file

```
python preflight.py --csv data/annotations/response_candidates.csv
```

Then convert using the offline CLI (no database or network access):

```powershell
python convert_annotation_candidates.py --input ..\data\annotations\response_candidates.csv --output ..\data\annotations\response_annotations.csv
```

The converter uses `normalized_text` as training `text`, skips blank labels and blank normalized text with counts, reports those exclusions, rejects invalid nonblank labels, duplicate document/comment IDs, invalid metadata, and malformed CSV rows, and refuses to overwrite an existing output unless `--overwrite` is explicitly passed. The converter does not approve a dataset; run it only after human review and manifest approval. Then run local preflight on the **training** CSV:

```powershell
python preflight.py --csv ..\data\annotations\response_annotations.csv
```

Training expects `data/annotations/response_annotations.csv` with columns
`document_id,text,response_type,post_id,comment_id,language,script,
is_code_mixed,detected_languages`.

---

## 8. Quality rules

1. One **primary** role per comment. Exactly one value; never blank in the
   training file.
2. Do not invent classes; the six are fixed for V1.
3. Keep `document_id` unique; a `document_id` with two different labels is a
   contradiction and training will refuse it.
4. When genuinely torn or lacking necessary context, leave the candidate blank for review/adjudication; never use `opinion` as a universal fallback.
5. Inter-annotator agreement matters: run the pre-flight check and watch the
   class distribution; a class with almost no examples cannot be learned.

---

## 9. Version history

| Version | Change |
| --- | --- |
| 1.1.0 | Operational decision procedure, blank-for-adjudication rule, and clearer boundaries for advice/solution, opinion/counterargument, short/neutral replies, context, and code-mixed text. |
| 1.0.0 | Initial six-label guide; advice/solution, agreement/experience and opinion/counterargument boundaries; short-reply and code-mixed rules. |
