# Issue Identification

**Stage:** NLP pipeline, after topic discovery
**Baseline model:** TF-IDF → One-vs-Rest Logistic Regression (multi-label)
**Model metadata:** `issue_model = "tfidf_logistic_multilabel"`, `issue_model_version = "v1"`

Answers: *what specific problem, concern, complaint or difficulty is being
expressed?*

> Sentiment/emotion, response classification, solution detection,
> issue→solution linking and resolution are **not** in this stage.

---

## 1. Topic vs. issue

| | Topic discovery | Issue identification |
|---|---|---|
| Question | *What is this discussion generally about?* | *What specific problem is being expressed?* |
| Output | K topics, each with keywords + document mix | A **set** of issue labels per document |
| Labels come from | The model (unsupervised) | **Human annotation** (supervised) |
| Label identity | `"Topic 3"` + keywords | `"fees"`, `"roi"`, `"placement_stress"` |
| Stability | Drifts whenever the corpus changes | Must be a stable, versioned taxonomy |
| Cardinality | Small, corpus-dependent (≈ 6–20) | Grows as annotators find new grievances |

The motivating example from the assignment:

```
"IIM life is amazing but the workload is killing me"
  topic = campus life / academics        <- topic discovery
  issue = workload                       <- issue identification
  emotion = stress / frustration         <- NOT this stage
```

Both are needed, and they are not substitutes. A topic alone cannot tell you
whether anyone is complaining. A label alone cannot tell you how widespread the
subject is. The genuinely useful analytical sentence is the join:

> "The **hostel** topic is dominated by the **hostel** and **mess_food** issues."

---

## 2. Why issue detection follows topic discovery

1. **Two different label spaces with different lifecycles.** Topics are
   re-mined on every corpus refresh; an issue taxonomy has to stay stable so
   that "mess_food in September" is comparable to "mess_food in January".
   Running them as one stage would force one unstable labelling to serve both
   purposes.

2. **Annotation cost is the bottleneck, and topics make it cheaper.** An
   annotator shown a document plus its topic keywords reaches a decision faster
   and more consistently than one shown a bare document. Topic ids are a
   useful navigation aid and optional extra features.

3. **Topic context is a legitimate prior.** A document in the "hostel / mess"
   topic is far more likely to carry a `hostel` or `mess_food` issue. This is
   exposed as an **off-by-default** feature
   (`IssueModelConfig.use_topic_features`) because topic ids are corpus-run
   specific and would not survive a re-fit of the topic model.

4. **Failure modes stay separable.** A degenerate topic model and a
   mis-trained issue classifier are different bugs with different fixes.
   Merging them makes both harder to diagnose.

**Ordering is a soft dependency, not a hard one.** `IssueDocument` accepts
`topic_id` / `topic_label` / `topic_keywords` as *optional*; the classifier
works with text alone. Nothing in the issue stage imports the topic stage.

---

## 3. Why multi-label classification is appropriate

Issue labels are **not** mutually exclusive, and one text frequently expresses
several at once:

```
"fees are way too high and ROI feels questionable"
  -> ["fees", "roi"]                 two independent complaints

"hostel rooms are too small and mess food is bad"
  -> ["hostel", "mess_food"]         two independent complaints

"fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>"
  -> ["fees", "roi", "placement_stress"]   three
```

A model with a softmax head **cannot** express the first example. Softmax
normalises over classes that are assumed to compete for a single mutually
exclusive outcome; here the outcomes genuinely coexist. Forcing a choice would
silently discard half the signal, and the discarded label would look like a
model error rather than an architecture error.

Multi-label is therefore the *correct* problem formulation, not a convenience.
Documents with **zero** issues are also valid (and are trained on as negatives
for every label) — "campus fest was amazing" is useful supervision, not noise.

---

## 4. The mathematics

### 4.1 TF-IDF representation

For document `d` and term `t`:

```
TFIDF(t, d) = TF(t, d) × IDF(t)
```

with

```
TF(t, d)  = 1 + log( count of t in d )                  (sublinear_tf = True)
IDF(t)    = log( (1 + N) / (1 + df(t)) ) + 1            (smooth_idf = True)
```

`N` = number of annotated documents, `df(t)` = number containing `t`. Each
document row is **L2-normalised** so that a long rant does not outweigh a
one-line comment. Stacking the rows gives `X ∈ R^(n_docs × n_terms)`.

Sublinear TF stops a document that repeats "fees" thirty times from becoming a
pure "fees" document; smoothed IDF gives a corpus-wide term a small positive
weight instead of `log(0)`.

### 4.2 Logistic regression (one per label)

For label `k`, a linear score is computed from the TF-IDF vector `x`:

```
z_k = w_k^T x + b_k
```

and squashed by the **sigmoid**:

```
σ(z) = 1 / (1 + e^(-z))
```

giving the probability that label `k` is present:

```
p(y_k = 1 | x) = sigmoid(w_k^T x + b_k)
                 = 1 / (1 + e^(-(w_k^T x + b_k)))
```

Training minimises binary cross-entropy (log loss):

```
L_k = - Σ_d [ y_{d,k} log p_{d,k} + (1 - y_{d,k}) log(1 - p_{d,k}) ]
```

plus L2 regularisation scaled by `1/C` (larger `C` = weaker regularisation).

### 4.3 One-vs-Rest across labels

For `K` labels, `K` **independent** binary classifiers are fitted, one per
label, each on the same feature matrix `X`:

```
for k in 1..K:
    fit w_k, b_k  on  (X, y[:, k])
```

This is `OneVsRestClassifier(LogisticRegression(...))`.

### 4.4 Why this differs from softmax

Softmax over `K` classes:

```
p(y = k | x) = e^{z_k} / Σ_{j=1..K} e^{z_j}
```

has two properties that break issue identification:

1. **The outputs sum to 1.** The model is asserting that exactly one class
   holds. With `["fees", "roi"]` as the truth, at most one of the two can ever
   be predicted — and the true label will be assigned a probability bounded by
   how much mass the others took.
2. **The classes compete.** Raising `p(fees)` *must* lower `p(roi)`. In
   reality they are not alternatives; both should be high.

Sigmoid outputs are independent, so nothing forces a sum-to-1 and several
labels can simultaneously exceed the threshold. This is directly asserted in
the test suite (`test_independent_sigmoids_not_softmax`: the probabilities of a
three-label document sum to well above 1).

**The threshold is the multi-label decision rule:**

```
predicted_labels = { k : p(y_k = 1 | x) > τ }
```

`τ` defaults to `0.5` and is configurable per model and per call. Because the
sigmoids are independent, `|predicted_labels|` can be `0`, `1`, or `K`.

---

## 5. Annotation format

Issue labels are **not defined in code**. The label set is whatever annotators
used, learned at `train()` time. Persist annotations at:

```
data/annotations/issue_annotations.csv
```

### CSV specification

| Column | Required | Type | Notes |
|---|---|---|---|
| `document_id` | **yes** | string | Stable id; the key used to map results back |
| `text` | **yes** | string | **Already preprocessed** (`normalized_text`) |
| `labels` | **yes** | `a\|b\|c` | Pipe-separated; **multi-label**; may be empty |
| `post_id` | no | string | Reddit source row |
| `comment_id` | no | string | Reddit source row |
| `language` | no | string | Defaults to `unknown` |
| `script` | no | string | Defaults to `unknown` |
| `is_code_mixed` | no | bool | `true` / `false` / `1` / `yes` |
| `detected_languages` | no | `hi\|en` | Pipe-separated |
| `topic_id` | no | int | Optional topic-discovery context |

Rules:

- **`labels` is pipe-separated (`|`), not comma-separated**, so a multi-label
  cell never collides with CSV comma separation. A label containing `|` is
  rejected as invalid.
- **Any number of labels is valid**, including zero. `labels` may be empty for
  a document that expresses no issue — this is valuable negative supervision.
- Unknown extra columns are ignored; missing optional columns fall back to the
  defaults above. Missing `document_id`, `text` or `labels` is a hard error.
- `text` must be the **preprocessed** text. This stage never re-runs
  preprocessing, so raw text would degrade feature quality.
- Label strings are free-form and language-neutral: `fees` covers an English,
  a Hinglish and a Bengali complaint alike. Nothing in the schema assumes an
  English-only or IIM-specific taxonomy.

### Example

```csv
document_id,text,labels,language,script,is_code_mixed,detected_languages
ann-001,"the fees are way too high and ROI feels questionable",fees|roi,en,latin,false,en
ann-002,"hostel rooms are too small and mess food is bad",hostel|mess_food,en,latin,false,en
ann-003,"placement season is extremely stressful",placement_stress,en,latin,false,en
ann-004,"mess ka khana kharab hai aur fees bhi zyada hai",mess_food|fees,hi,latin,true,hi|en
ann-005,"campus fest was amazing and the food was great",,en,latin,false,en
```

### Loading

```python
from app.services.nlp.issue_identification import AnnotationCorpus

corpus = AnnotationCorpus.from_csv("data/annotations/issue_annotations.csv")
print(corpus.labels)                # sorted distinct label set
print(corpus.label_counts())        # positives per label
print(corpus.duplicate_document_ids())

corpus.to_csv("data/annotations/issue_annotations_roundtrip.csv")
```

`from_csv` accepts a full path or a path relative to the repository root,
reads UTF-8 (with BOM tolerance), skips fully blank lines, and reports the
offending `file:line` for any malformed row.

---

## 6. Data requirements, and refusing to pretend

`train()` **refuses** to produce a model it cannot honestly justify:

| Condition | Behaviour |
|---|---|
| No annotations | `ValueError` pointing at the CSV path |
| `< min_training_documents` (default 5) | `ValueError` naming the count |
| `< 2` distinct labels | `ValueError` — multi-label needs ≥ 2 labels |
| A label with `< min_positive_examples_per_label` (default 2) positives | `ValueError` naming each offending label and its count |
| A label on **every** document (no negative example) | `ValueError` — the binary sub-problem is degenerate and unfittable |
| All texts blank / no usable TF-IDF features | `ValueError` |

Every error names the offending label or count and suggests the fix, so an
operator can act on it instead of guessing.

**Practical minimums for a real run.** The defaults are a *floor*, not a target:

- **≥ 50–100 annotated documents** per label before trusting a number.
- **Balanced coverage**: rare labels are the ones that break macro-F1. Consider
  `IssueModelConfig(class_weight="balanced")` when classes are skewed.
- **Held-out data**: metrics computed on the training set are not metrics. The
  demo evaluates on records withheld from training purely to make that point.
- **A fixed annotation guideline.** The taxonomy should be agreed *before*
  labelling, not derived afterwards from whatever annotators happened to write.

---

## 7. Evaluation

`IssueEvaluationReport` reports, with multilabel correctness in mind:

| Metric | What it means here |
|---|---|
| **micro-F1** | Pools all (document, label) decisions. The headline number on imbalanced multi-label data, because a rare label counts only as much as its true-positive evidence allows. |
| **macro-F1** | Averages each label equally, so one failing label moves it a lot. Exposes per-label failure; does **not** describe overall quality. |
| micro/macro precision, macro recall | As usual, averaged the corresponding way. |
| **per-label F1 / precision / recall** | Per-label breakdown plus `support` and predicted-count. |
| **subset accuracy** | Exact label-**set** match. Strictest possible number; always ≤ micro-F1 and often near zero on genuinely multi-label data. Reported, never presented as "accuracy". |
| hamming accuracy / loss | Fraction of correct per-label decisions. **Systematically inflated** on sparse multi-label data (an all-zero true and predicted row counts as perfect), so it is labelled as such. |
| classification report | Full `sklearn` per-label report as a dict. |

### Nothing is fabricated

If the evaluation set is too small, metrics are **withheld**, not invented:

- `< 10` documents → `is_meaningful = False`, `reason` states the shortfall,
  and every metric field stays `None`. Per-label `support` is still reported,
  because that is a fact about the data rather than a model claim.
- `< 2` labels → withheld, with the multi-label reason.
- Labels with zero support, labels never predicted, and labels with `< 2`
  positives each raise an explicit warning, because their F1 is convention-zero
  or statistically unstable and must not be quoted as model quality.

> With real Reddit data in early stages, the correct output is usually
> "insufficient data, cannot yet measure" — and that is what this module
> returns. The demo's held-out numbers come from **synthetic** data and are a
> plumbing check, not project performance.

---

## 8. Configuration

`IssueModelConfig` (all validated at construction with actionable errors):

| Group | Parameter | Default | Notes |
|---|---|---|---|
| TF-IDF | `max_features` | `20000` | Vocabulary cap |
| | `min_df` | `1` | Raise to `2`–`3` on real corpora |
| | `max_df` | `1.0` | Lower to drop corpus-wide terms |
| | `ngram_range` | `(1, 2)` | Bigrams capture "mess food", "return on investment" |
| | `stop_words` | `None` | `None` / `"english"` / explicit list only |
| | `lowercase` | `True` | Folds `ROI` → `roi` |
| | `sublinear_tf` | `True` | `1 + log(count)` |
| | `token_pattern` | `(?u)\b\w+\b` | Script-agnostic |
| | `norm` | `"l2"` | Row normalisation |
| | `use_idf`, `smooth_idf` | `True` | Turn off only for ablations |
| Logistic regression | `C` | `4.0` | Inverse regularisation; ↑ = fits harder |
| | `max_iter` | `1000` | |
| | `solver` | `"lbfgs"` | Deterministic |
| | `class_weight` | `None` | `"balanced"` for skewed labels |
| | `random_state` | `42` | Pinned |
| Decision | `threshold` | `0.5` | Overridable per `predict()` call |
| Data guards | `min_training_documents` | `5` | Floor, not a target |
| | `min_positive_examples_per_label` | `2` | |
| Features | `use_topic_features` | `False` | Adds `topicid*` / `topickey*` tokens |
| | `drop_placeholder_tokens` | `True` | Strips `<URL>`/`<MENTION>`/`<EMOJI_...>` |

`IssueModelConfig` deliberately does **not** ship Hindi/Bengali/Odia stop-word
lists; the validator rejects any list other than `None`, `"english"`, or an
explicit user-supplied list. In Romanized code-mixed text those function words
carry the code-switch structure.

### Threshold tuning

The threshold is the recall/precision dial, and it should be tuned on
validation data, not set once and forgotten:

- Lower τ → more labels per document, higher recall, more false positives.
- Higher τ → fewer labels, higher precision, and more documents predicted as
  having **no** issue.
- `result.n_documents_without_issue` is the number to watch: if it is far above
  your expected no-issue rate, τ is too high.

The full probability vector is always returned, so a consumer can re-threshold
later with `prediction.labels_above(τ)` **without re-running inference**.

---

## 9. Multilingual and code-mixed limitations

The corpus is English, Hinglish, Romanized Hindi, Bengali and Odia, frequently
in one sentence. Measures taken:

- **No translation, ever.** A Bengali document stays Bengali; `language`,
  `script`, `is_code_mixed` and `detected_languages` are carried into every
  prediction for downstream filtering.
- **Script-agnostic tokenisation.** `(?u)\b\w+\b` keeps Devanagari, Bengali and
  Odia terms in the vocabulary.
- **No language-specific stop-word removal** (§8).
- **Language-neutral label space.** The same label applies across languages,
  so a Hindi "fees" complaint and an English one are supervised by the same
  weight vector. Nothing creates `hindi_fees` variants.
- **Emoji placeholders are dropped from features by default.** They are
  preprocessing artefacts, issue labels describe problems rather than
  emotions, and they appear in most documents so they add noise, not signal.
  The document's `text` is still returned verbatim.

### Honest limitations

1. **A linear bag-of-words model cannot generalise across languages.** TF-IDF
   shares no features between `fees` and `ফি`, so a Bengali complaint is
   predicted from Bengali tokens alone. If a label's training examples are
   mostly English, native-script performance degrades sharply. This is the
   dominant weakness and the main motivation for §11.
2. **Indic scripts are whitespace-segmented.** Devanagari/Bengali/Odia do not
   separate orthographic syllables with spaces, so `ফি অনেক বেশি` splits into
   grammatical fragments rather than words. A language-aware tokenizer is
   required before native-script labels can be reliable.
3. **Code-switching dilutes the signal.** "mess ka khana kharab hai aur fees
   bhi zyada hai" mixes scripts inside one document, so both the English and
   the Romanized feature views are partial.
4. **Lexical overfitting on a small annotation set.** With tens of examples a
   linear model memorises surface words; `C` and `max_features` then become the
   dominant hyper-parameters. This is why the data guards refuse tiny inputs.
5. **No negation or intensity modelling.** "not worth it" and "worth it" share
   features. Negation-aware features are out of scope for the baseline.
6. **Label imbalance is the norm.** Real complaint data is long-tailed; rare
   labels will show near-zero macro-F1 until they are over-sampled or
   `class_weight="balanced"` is used.

---

## 10. Why this is a baseline

- **Transparent.** One weight vector per label; you can print the top positive
  and negative features of `w_k` and see exactly why a label fired. This is
  invaluable when the taxonomy is still being designed.
- **One dependency, CPU-only, milliseconds per document.** It can be retrained
  on every annotation round and run over the whole corpus in a scheduled job.
- **A real reference point.** A neural model must beat this to justify itself.
  Without a baseline, a 0.82 micro-F1 neural result is uninterpretable.
- **Correct problem formulation.** The multi-label OvR structure is not
  throwaway: it is exactly the head shape a neural model will use
  (`K` independent sigmoid outputs), so the interface, annotation pipeline and
  evaluation code carry over unchanged.
- **Recovers a real baseline quickly.** It becomes genuinely useful long before
  annotation volume is large enough to fine-tune a transformer.

---

## 11. Future neural version (not implemented)

The `IssueIdentification` ABC is the seam. A `TransformerIssueClassifier`
would satisfy the same three methods (`train`, `predict`, plus `name`,
`version`, `labels`, `is_trained`) and return the identical
`IssueIdentificationResult`, so persistence, the API and the dashboard need no
changes.

Planned shape:

1. **Encoder** — a multilingual checkpoint fine-tuned for classification, e.g.
   `xlm-roberta-base` or an Indic-native model (IndicBERT / mDeBERTa-v3).
   Chosen **per language subset** if native-script support stays weak.
2. **Head** — `K` independent sigmoid outputs, i.e. the same multi-label
   structure as the baseline, with `BCEWithLogitsLoss` (binary cross-entropy
   summed over labels, **not** softmax cross-entropy). Positive-class weights
   or focal loss for the long tail.
3. **Threshold calibration** — per-label thresholds tuned on a validation
   split, replacing the single global τ.
4. **Comparison, not replacement** — the TF-IDF baseline stays in the repo as
   the reference to diff against, and as a fallback for languages the
   transformer handles poorly.

Expected gains, honestly scoped: **cross-lingual transfer** (the baseline's
weakest point) and **paraphrase robustness** ("mess food" ≈ "hostel dining").
Expected non-gains: it will not fix a bad taxonomy or thin annotation, and it
costs GPU and a much longer iteration loop.

**Not implemented in this stage**, deliberately: sentiment/emotion
classification, response classification, solution detection, issue→solution
linking, and resolution status.

---

## 12. Persistence

**No database changes were made.** Nothing here writes to
`reddit_posts.body`, `reddit_comments.body`, `normalized_text`, or topic
discovery output, and no tables or columns were added. The annotation CSV and
the `model_dump()`-serialisable results are sufficient to establish the
annotation workflow first.

Once real annotations exist and a schema is agreed, the minimal shape is one
prediction table keyed by `document_id`:
`issue_labels`, `issue_probabilities` (JSON), `confidence`, `model_name`,
`model_version`, `threshold`, `predicted_at`. `document_id` already carries
`post_id` / `comment_id` for the join back to source. That decision is
deliberately deferred until the annotation process has proven itself.

---

## 13. Usage

```python
from app.services.nlp.issue_identification import (
    AnnotationCorpus, IssueDocument, TfidfLogisticIssueClassifier,
)

corpus = AnnotationCorpus.from_csv("data/annotations/issue_annotations.csv")

classifier = TfidfLogisticIssueClassifier(threshold=0.5)
report = classifier.train(corpus.annotations)      # label set learned here
print(report.labels, report.positive_examples_per_label)

result = classifier.predict([
    IssueDocument(document_id="p1", text="fees are way too high and ROI is poor")
])
for prediction in result.predictions:
    print(prediction.predicted_issue_labels, prediction.issue_probabilities)

metrics = classifier.evaluate(corpus.annotations)   # withheld if data is thin
print(metrics.format_summary())
```

Run the demo:

```
python demo_issue_identification.py
```

Run the tests:

```
python -m pytest test_issue_identification.py -q
```

---

## 14. Files

| File | Purpose |
|---|---|
| `backend/app/services/nlp/issue_identification/base.py` | `IssueIdentification` ABC, `IssueDocument`, `IssueModelConfig` |
| `backend/app/services/nlp/issue_identification/models/annotation.py` | `IssueAnnotation`, `AnnotationCorpus`, CSV schema |
| `backend/app/services/nlp/issue_identification/models/result.py` | `IssuePrediction`, `IssueIdentificationResult`, `IssueTrainingReport` |
| `backend/app/services/nlp/issue_identification/models/evaluation.py` | `IssueEvaluationReport`, `evaluate_predictions` |
| `backend/app/services/nlp/issue_identification/tfidf_logistic_classifier.py` | `TfidfLogisticIssueClassifier` — TF-IDF → OvR logistic regression |
| `backend/app/services/nlp/issue_identification/__init__.py` | Public exports |
| `backend/test_issue_identification.py` | 168 tests |
| `backend/demo_issue_identification.py` | Synthetic end-to-end demo |
| `docs/issue_identification.md` | This document |
