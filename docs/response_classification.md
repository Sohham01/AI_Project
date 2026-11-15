# Response Classification

> What **role** does a comment play in a discussion?
> One baseline model: TF-IDF features into a six-way **softmax** logistic regression.

This stage answers a structural question about a thread. It is deliberately
*not* a sentiment stage and *not* an issue-detection stage; see
[What this is not](#what-this-is-not).

- **Code:** `backend/app/services/nlp/response_classification/`
- **Demo:** `backend/demo_response_classification.py`
- **Tests:** `backend/test_response_classification.py`
- **Model identifiers:** `tfidf_logistic_multiclass`, version `v1`

---

## 1. The six response types

The taxonomy is **fixed by the assignment** and is not derived from the data.
The model can never grow a seventh class; an annotation outside this set is a
labelling error and is refused.

| Code | Class | Definition | Illustrative example |
|------|-------|------------|----------------------|
| A | `agreement` | endorses the parent or another commenter's claim | "same bro, exactly" |
| B | `personal_experience` | reports the writer's own past experience | "I faced this exact problem last year" |
| C | `opinion` | states a belief or judgement, without lived evidence | "I think this college is better" |
| D | `advice` | recommends an action **to somebody else** | "you should contact the placement office" |
| E | `solution` | presents the answer or fix outright | "apply for the scholarship on the portal" |
| F | `counterargument` | pushes back on the claim being discussed | "actually the fees are not that high" |

The canonical order above is the ordering used for probability vectors, the
confusion matrix, and every report. It is fixed, so outputs from two runs can be
compared positionally.

### Boundaries that are genuinely hard

These are the pairs annotators argue about, and they are the reason
[Section 8](#8-annotation-guidelines) matters more than the model:

- **`personal_experience` vs `opinion`** — the difference is *evidence*. "I
  think it is good" is `opinion`; "I went through this myself" is
  `personal_experience`. "I have been there and it was fine" is
  `personal_experience`, not `opinion`, because it appeals to something that
  happened.
- **`advice` vs `solution`** — the difference is the *addressee*. "You should
  apply" and "apply" are the same recommendation; one is second person, the
  other imperative. Pick `advice` for explicit "you should", `solution` for a
  bare action.
- **`agreement` vs `counterargument`** — both reference someone else, and both
  are only identifiable **given what is being replied to**. See
  [Section 9](#9-context-is-optional-in-v1).
- **`counterargument` vs `opinion`** — "I think it is bad" is `opinion`; "but
  the fees are not that high" is `counterargument`, because it engages a
  specific opposing claim rather than just adding a view.

---

## 2. What this is not

Response classification is a **fourth** independent axis. Confusing the axes is
the most common integration mistake, so the separations are stated explicitly.

| Axis | Question | Stage |
|------|----------|-------|
| Language / script | Which languages appear? | preprocessing |
| Issue detection | What problem is being discussed? | `issue_identification` |
| Sentiment | What is the emotional tone? | `sentiment_emotion` |
| **Response type** | **What role does this comment play?** | **this stage** |

Concretely:

- **Not sentiment.** "apply for the scholarship" carries no polarity at all, and
  "same bro" can be said happily or sarcastically. Polarity and role are
  independent, and neither is derived from the other.
- **Not issue detection.** A single comment can be `advice` *and* about
  `fees`; a comment can be `opinion` with no issue attached at all. Issue
  labels are carried through as metadata and are not features.
- **Not multi-label.** Each comment gets **exactly one** primary role. A
  comment that agrees *and* shares experience is labelled with the one role its
  author is primarily performing; the secondary aspect is a note in the
  annotation, not a second label. `agreement` and `counterargument` in
  particular are contradictory, so a multi-label output would be incoherent.
- **Not thread reconstruction.** Roles are assigned per comment in isolation.
  Grouping comments into threads, clustering threads, and detecting
  resolution are later stages.

---

## 3. Input contract

`ResponseDocument` consumes **already-preprocessed** text. The model never
re-normalises anything and never imports a preprocessing helper.

```python
ResponseDocument(
    document_id="c-1001",              # required, unique
    text="same bro exactly",           # required; the *normalized* text
    original_text="Same bro, exactly!!",  # preserved, never modified
    post_id="post-1",
    comment_id="c-1001",
    parent_comment_id="c-0998",
    language="en",                     # free-form; 'unknown' is the default
    script="latin",
    is_code_mixed=False,
    detected_languages=["en", "hi"],
    # optional context - carried and echoed, NOT used as features in V1
    parent_comment_text="the fees are far too high for this course",
    topic_id=7,
    topic_label="fees",
    topic_keywords=["fees", "hostel"],
    issue_labels=["fees"],
    issue_probabilities={"fees": 0.82},
)
```

| Field | Required | Notes |
|-------|----------|-------|
| `document_id` | yes | must be unique in a corpus |
| `text` | yes | normalized; non-blank after stripping |
| `response_type` | — | annotation only, not on `ResponseDocument` |
| `original_text` | no | echoed back verbatim for traceability |
| `post_id`, `comment_id` | no | echoed back |
| `language`, `script`, `is_code_mixed`, `detected_languages` | no | carried for reporting; not features |
| `parent_comment_text`, `topic_id`, `topic_label`, `topic_keywords`, `issue_labels`, `issue_probabilities` | no | **ignored as features in V1**, echoed back |

`has_context` reports whether any optional context was supplied. It is
diagnostic only — it does not change the prediction.

`<EMOJI_...>`, `<URL>`, `<USER_MENTION>` and similar placeholders are stripped
from the copy used to build features — they are pure noise for a lexical
model — while the document keeps them. A document whose text is *only* such a
placeholder therefore yields no features; if that happens to every document in
the training set, `train()` refuses with an empty-vocabulary error.

### Isolation

The package imports nothing from `app.services.nlp.*` and nothing from the
database. A test parses its source and asserts this, because a back-reference
to the sentiment or issue stages would silently couple two models that are
supposed to be independently replaceable. Metadata arrives on the input object
and travels out on the output object; it is never used to fit or predict.

---

## 4. Method

### 4.1 TF-IDF

Each document becomes a sparse vector of unigram and bigram counts, reweighted
to emphasise terms that are distinctive in this corpus:

```
tf(t,d)   = 1 + log(count(t, d))
idf(t)    = log( (1 + N) / (1 + df(t)) ) + 1
tfidf(t,d)= tf(t,d) * idf(t)
```

Then L2-normalised so that long comments do not dominate purely by length. In
the demo's 24-comment training set this yields ~207 features — the scale of
anything you can honestly expect to learn from a few dozen examples.

Defaults: `ngram_range=(1, 2)`, `min_df=1`, `max_df=1.0`, `sublinear_tf=True`,
`lowercase=True` (folds case defensively, even though upstream normalisation
already did), `stop_words=None` (see [Section 11](#11-multilingual-and-code-mixed-limitations)),
`token_pattern=r"(?u)\b\w+\b"` — the `(?u)` flag is what keeps Devanagari,
Bengali and Odia words intact instead of being split on non-ASCII boundaries.

### 4.2 Logistic regression with softmax

The six roles are mutually exclusive, so the output is a single six-way
distribution, not six independent scores. For class *k*:

```
z_k = w_k · x + b_k

P(response_type = k | x) = exp(z_k) / Σ_{j=1..6} exp(z_j)
```

Trained by minimising the multiclass cross-entropy (equivalently, L2-regularised
maximum likelihood):

```
minimise over W, b:   -Σ_i log P(y_i | x_i) + λ ‖W‖²₂
```

**Why softmax and not six independent sigmoids.** Six One-vs-Rest binary
classifiers would each answer "is this an agreement?" independently. Their
scores would not sum to 1, and on a comment that reads as agreement *and*
advice both could fire. That is a different question from "which role is this
comment primarily playing?", and it is not the question the assignment asks.
`test_softmax_output_is_a_probability_distribution` pins the behaviour: the six
values sum to 1 within `1e-6` rounding.

**Settings** (`ResponseModelConfig`, exposed in the training report):

| Setting | Default | Why |
|---------|---------|-----|
| `solver` | `lbfgs` | multinomial; deterministic for a given seed |
| `random_state` | `42` | reproducible fits |
| `class_weight` | `"balanced"` | see [Section 7](#7-class-imbalance) |
| `C` | `4.0` | light L2; enough capacity for lexical markers |
| `max_iter` | `1000` | convergence warning becomes an error |
| `max_features` | `20000` | bound for a lexical model |
| `threshold` | `0.5` | confidence floor; below it the model abstains |

### 4.3 Abstention

`predict(comments, threshold=...)` returns both the arg-max and the full
distribution. If the winner's probability is below the floor, the prediction
**abstains**:

```python
prediction.response_type            # None
prediction.abstained                # True
prediction.top_response_type()      # 'counterargument'   <- still available
prediction.confidence               # the raw winner score
prediction.response_probabilities   # all six values, unchanged
```

Abstention is preferable to a confident guess, because a low-probability label
on a *downstream* stage is a bug report waiting to happen: if `advice` and
`counterargument` drive different downstream behaviour, forcing a coin-flip
label manufactures errors that abstention would have flagged. The floor is
configurable, and the full vector is always returned so a consumer can
re-apply a different floor without re-running inference.

`threshold=None` means "use the configured default", not "invalid".

---

## 5. Model interface

```python
from app.services.nlp.response_classification import (
    TfidfLogisticResponseClassifier, ResponseDocument, ResponseAnnotation,
)

classifier = TfidfLogisticResponseClassifier()
report = classifier.train(annotations)          # -> ResponseTrainingReport
result = classifier.predict(documents)          # -> ResponseClassificationResult
report = classifier.evaluate(held_out)          # -> ResponseEvaluationReport
```

Everything is Pydantic and JSON-serialisable:

```python
result.model_dump_json()
```

`ResponsePrediction` carries `response_type`, `confidence`, `abstained`,
`response_probabilities` (a `dict` so the JSON has readable keys), plus
`top_response_type()` and `to_dict()`. The document's `original_text`,
`post_id`, `comment_id`, `parent_comment_text`, `topic_id` and `issue_labels`
are echoed onto every prediction.

`result` also exposes `known_response_types`, `missing_response_types`,
`n_abstained`, `n_documents`, `response_type_frequency()` and
`average_confidence()`.

---

## 6. Refusals

`train()` declines data it cannot learn from, and says why. Each message names
the problem and the way out, rather than fitting a model that looks trained and
generalises to nothing.

| Situation | Behaviour |
|-----------|-----------|
| Empty corpus | refused |
| Fewer than 5 documents | refused |
| Fewer than 2 response types | refused |
| A response type with fewer than 2 examples | refused |
| Label outside the six | refused, listing the valid set |
| Same `document_id` with **conflicting** labels | refused |
| Same `document_id` with the **same** label | warned, deduplicated |
| No usable TF-IDF vocabulary | refused |
| Invalid configuration (e.g. `C <= 0`, `threshold` outside `(0, 1]`, contradictory `min_df`/`max_df`, empty vocabulary params) | refused |
| Training set covering fewer than all six types | **allowed by default**, reported via `missing_response_types`; refused when `require_all_response_types=True` |

The last row is a deliberate default. Requiring all six classes would make
early annotation impossible, and a partially trained model that *declares* which
types it cannot emit is more useful than no model at all. Its probability vector
contains only the types it can emit, and every report states the gap.

A model trained on a subset must also survive being evaluated on a set
containing roles it never saw: those gold labels are reported as
`n_unscoreable` with an explicit warning, and counted as **misses** rather than
dropped from the denominator. Silently discarding them would inflate accuracy
precisely when the model is weakest.

---

## 7. Class imbalance

`class_weight="balanced"` is the **default**, and it is not decoration.

```python
weight_k = N / (n_classes × n_k)
```

Real corpora are heavily skewed towards `agreement` and `opinion`, which are
cheap to collect, while `advice` and `solution` — the roles that actually carry
actionable information — are rare. Unweighted training on such a corpus looks
accurate and is close to useless on the roles that matter.

The demo shows the effect on a 10:3 corpus: with a `counterargument` probe,
balanced weights move ~0.45 of the mass onto the rare class against ~0.20 for
the dominant one.

**Accuracy is still reported**, but never alone, and never without per-class
support. A 0.92 accuracy over a corpus that is 90% `agreement` is a statement
about the corpus, not about the model. `macro-F1` and the per-class table are
there so that can be seen immediately.

`train()` also emits a `class imbalance` warning when one role's share
*exceeds* `imbalance_warning_share` (default `0.8`), naming the majority class
and its share. A share exactly at the threshold is not yet flagged.

---

## 8. Annotation guidelines

**No annotation file ships with this repository.** Real data comes from the human
workflow described here; fabricating labels would make every downstream number
meaningless. The demo's data is explicitly hand-written and is not an annotation
example to copy.

### CSV schema

`data/annotations/response_annotations.csv`, in canonical column order:

```csv
document_id,text,response_type,post_id,comment_id,language,script,is_code_mixed,detected_languages
c-1001,"same bro exactly",agreement,post-1,c-1001,en,latin,false,en
c-1002,"apply for the scholarship",solution,post-1,c-1002,en,latin,false,en
```

Only `document_id`, `text` and `response_type` are required. `detected_languages`
is pipe-separated (`en|hi`). Unknown extra columns are **ignored** by the
loader, so a workflow is free to add its own `notes` or `needs_context` columns
— but it must resolve them before training, because the model will not see them.

Load and write with:

```python
corpus = ResponseCorpus.from_csv("data/annotations/response_annotations.csv")
corpus.validate()                   # taxonomy check, raises on a bad label
corpus.to_csv("path/to/copy.csv")   # never overwrites the source by default
```

`from_csv` accepts a full path or one relative to the repository root, so it
works from either the root or `backend/`. Loading never modifies anything on
disk.

### Rules

1. **Exactly one role per comment.** Pick the role the author is primarily
   performing. Use the `notes` column for a secondary aspect, not a second label.
2. **Annotate the text, not the tone.** Deciding from sentiment is the fastest
   way to produce a sentiment model wearing a response model's name.
3. **`agreement` and `counterargument` need the parent.** If the parent text is
   unavailable, add a `needs_context: true` column (ignored by the loader) and
   exclude those rows from training rather than guessing between two roles that
   are genuinely indistinguishable in isolation.
4. **Annotate in the original script.** Script-mixed and code-switched text
   (`Hinglish`, Bangla in Latin script) is real data and must be kept, labelled
   with its true `language` and `script` and `is_code_mixed: true`. Do not
   transliterate for convenience.
5. **Two annotators, adjudicate disagreements.** Measure agreement per class
   with Cohen's κ. The hard pairs in [Section 1](#1-the-six-response-types) will
   dominate the disagreement, and κ is how you find out whether the label set is
   actually usable.
6. **Split by `post_id`, not by comment.** Threads from the same post share
   vocabulary, so a random comment-level split leaks the post into both sides
   and inflates the score.
7. **One annotation per document.** Conflicts are refused; identical duplicates
   are warned and deduplicated.

### Minimum viable annotation

The guards require ≥5 documents, ≥2 types and ≥2 examples per type. Those are
plumbing floors that catch empty or degenerate input, **not** a target. A
6-class lexical model wants on the order of 100+ examples per class, and
`agreement`/`opinion` will be over-represented in whatever you collect, which is
why balancing is on by default.

---

## 9. Context is optional in V1

`ResponseDocument` accepts `parent_comment_text`, `topic_id`, `topic_label` and
`issue_labels`. **V1 does not use them as features.**

This is a real limitation, not a simplification. "Same bro" is only agreement if
it agrees with something; "actually the fees are not that high" is only a
counterargument relative to a claim it never sees. A comment in isolation is
genuinely ambiguous between `agreement` and `opinion`, and between `opinion` and
`counterargument`.

Pretending otherwise would mean either (a) silently worse predictions, or (b)
building context handling into a stage whose job is to be a replaceable baseline.
So the fields exist on the interface, are echoed onto every prediction for
traceability, and a test asserts that adding context leaves the distribution
**bit-identical**. A future `v2` can consume them with no public signature
change.

---

## 10. Evaluation

```python
report = classifier.evaluate(held_out_annotations, threshold=0.5)
print(report.format_summary())
```

Reports: accuracy, macro/micro/weighted F1, precision and recall, a confusion
matrix, per-class metrics, and the full `classification_report`. The class order
is the canonical one, so matrix positions are stable.

### Metrics are withheld, not fabricated

`is_meaningful` is `False` and every metric is `None` when the evaluation set
is too small to support the number:

| Guard | Threshold | Why |
|-------|-----------|-----|
| `MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS` | ≥ 10 documents | below this a single comment moves accuracy by >10% |
| `MIN_CLASSES_FOR_MEANINGFUL_METRICS` | ≥ 2 distinct response types | macro-F1 and a confusion matrix are undefined for one class |

`reason` explains which guard fired. Abstention and unscoreable counts are
**facts about the data**, so they are reported even on a withheld report — only
the rates are suppressed.

A third rule, `MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1 = 2`, does **not** withhold:
a class seen only once still gets a per-class F1, but the report warns that the
figure is unstable and should not be quoted. Suppressing the whole report
because one class is thin would hide the other five.

### Reading the numbers honestly

- **micro-F1 equals accuracy** on single-label multiclass data. Both are
  reported so the identity is visible rather than assumed.
- **Abstentions count as misses.** Dropping them from the denominator would
  report accuracy over the easy cases only.
- **Off-diagonal cells are the finding.** A confusion matrix shows *which* role
  is being confused with which — that is what an annotation round should fix
  next, and it is invisible in an accuracy figure.
- **Check per-class support before believing a macro-F1.** A macro average over
  one class of support 2 and five of support 500 is not a summary of anything.

---

## 11. Multilingual and code-mixed limitations

Real, and worse than the usual boilerplate:

- The English training fixtures are lexically stereotyped. Real code-switched
  text mixes scripts *within a token* ("applyकरो"), which word-level TF-IDF
  fragments badly; character n-grams would help but are not used here.
- Devanagari, Bengali and Odia are **morphologically inflected**. Stemming is
  not applied, so "korro" / "korchi" / "korbo" are three unrelated features.
  Six to eight examples of a language teaches the model that language exists,
  not how to read it.
- Aggressive upstream normalisation can strip the discourse markers that
  *define* these classes. "lekin" (but) and "however" are load-bearing for
  `counterargument`; a stopword list that removes them removes the class's only
  signal.
- The demo's non-Latin rows are included to prove **nothing crashes**, and they
  do classify — but on data the model was literally trained on. That is a
  memorisation check, not a generalisation result.

A real multilingual evaluation needs per-language held-out sets and
per-language metrics reported separately. Aggregate numbers across scripts hide
a failure in the smaller languages, which is where it will be.

---

## 12. Future work

1. **Context-aware v2.** Concatenate `parent_comment_text` as an extra feature
   block, or run a pair classifier on (parent, comment). Expected to be the
   single largest gain for `agreement` vs `counterargument`.
2. **Linear SVM comparison.** Same features, different loss; often stronger than
   logistic regression on small, high-dimensional text.
3. **Fine-tuning a multilingual encoder** once annotated data is large enough
   to justify it.
4. **Soft labels.** Allow annotators to mark a primary *and* a secondary role and
   train on the distribution, rather than forcing a single hard choice and
   hiding the genuine ambiguity.
5. **Per-language models and metrics**, rather than one model over pooled scripts.
6. **Calibration.** Fit isotonic or temperature scaling on a validation split so
   the 0.5 abstention floor means something, instead of being a raw
   softmax score.
7. **Drift monitoring.** Re-evaluate on a rolling fresh sample; lexical models on
   fast-moving student vocabulary degrade faster than the code suggests.

---

## 13. Tests

```
cd backend
& ".\.venv\Scripts\python.exe" -m pytest test_response_classification.py -q
& ".\.venv\Scripts\python.exe" -m pytest -q          # full regression suite
& ".\.venv\Scripts\python.exe" demo_response_classification.py
```

189 tests cover: the label set and its rejection of invalid labels; annotation
and CSV round-tripping; duplicate/conflicting IDs; every training refusal;
softmax normalisation; determinism across two fits and two `predict()` calls;
metadata propagation; abstention and the confidence floor; class weighting
changing the fitted model; imbalance warnings; confusion-matrix off-diagonals;
withheld metrics; configuration validation; Pydantic coercion; and a source-level
architecture test asserting the package imports nothing from earlier NLP stages.

The demo is a **plumbing check** on synthetic data, not a performance claim.
Its banner says so, and so does this section.
