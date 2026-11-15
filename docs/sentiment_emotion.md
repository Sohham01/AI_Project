# Sentiment and Emotion Analysis

**Stage:** NLP pipeline, after issue identification
**Baselines:** two independent models - TF-IDF + Logistic Regression (single-label) for sentiment, TF-IDF + One-vs-Rest Logistic Regression (multi-label) for emotion
**Model metadata:** `sentiment_model = "tfidf_logistic"` / `sentiment_model_version = "v1"`, `emotion_model = "tfidf_logistic_multilabel"` / `emotion_model_version = "v1"`

Answers two different questions, in two different models:

- *Sentiment* - what is the overall polarity of the text?
- *Emotion* - which feeling(s) does the writer express?

> Response classification, solution detection, issue-solution linking and
> resolution detection are **not** in this stage.

---

## 1. Sentiment and emotion are not the same dimension

| | Sentiment | Emotion |
|---|---|---|
| Question | *Is this positive, negative or neutral?* | *What is the writer feeling?* |
| Cardinality | **One** mutually exclusive class | A **set** of labels, possibly empty |
| Output law | Probabilities sum to 1 | Independent sigmoids, sum unconstrained |
| Model head | One multiclass logistic regression | One binary logistic regression per label |
| Labels come from | The `SentimentAnnotation` corpus | The `EmotionAnnotation` corpus |
| Can be empty? | No - a class is always the winner, or it abstains | Yes - "no emotion above the threshold" is a valid answer |

The motivating pair from the assignment brief:

```
"placement season is horrible"
  sentiment = negative          <- one class
  emotion   = frustration       <- a label, decided by emotion annotations

"I was nervous but it worked out well"
  sentiment = positive          <- overall polarity
  emotion   = anxiety           <- NOT implied by the polarity
```

The second example is the important one. A text can be **positive and anxious**,
**negative and grateful**, or **negative with no emotion at all**. Sentiment
carries no information about emotion, and a model that infers one from the other
is making up an annotation that nobody made.

### Emotion is never inferred from sentiment

This is a structural guarantee, not a convention:

- separate annotation types (`SentimentAnnotation` vs `EmotionAnnotation`)
- separate configuration models (`SentimentModelConfig` vs `EmotionModelConfig`)
- separate abstract base classes (`SentimentAnalysis` vs `EmotionAnalysis`)
- separate classifiers, separate `train()` calls, separate probability vectors

There is no code path from a sentiment result to an emotion output. If a project
wants "negative implies frustration", that is a modelling decision to be made
with an annotation round, not a default to inherit.

---

## 2. Why this follows issue identification

1. **Issue says what problem; sentiment says how the writer feels about it;**
   emotion says which feeling. A single complaint can carry all three, and
   analytically they are different questions: a topic can be *about* hostel
   mess and be discussed calmly by most posters and angrily by a few.
2. **Sentiment is mutually exclusive by construction.** A text has one overall
   polarity, so a softmax-like head is the right shape for it.
3. **Emotion is not.** Modelling them with one shared head would force one of
   those two truths to be wrong, so they are two models.
4. **Emotion annotation is the expensive, subjective one.** It is the stage most
   likely to be launched with partial label coverage, so keeping it separate
   means a project can ship sentiment and add emotion later without disturbing
   the shipped model.

---

## 3. The mathematics

### 3.1 TF-IDF representation (shared by both)

For a document with tokens $t_1 \dots t_k$:

$$\text{tf}(t, d) = 1 + \log \text{count}(t, d)$$

$$\text{idf}(t) = \ln \frac{1 + N}{1 + \text{df}(t)} + 1$$

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t), \qquad
\hat{v}(d) = \frac{\text{tfidf}(\cdot, d)}{\|\text{tfidf}(\cdot, d)\|_2}$$

`sublinear_tf=True`, `smooth_idf=True`, `norm="l2"` - sklearn's defaults, stated
here so the document matches the code.

### 3.2 Sentiment: one multiclass logistic regression

$$P(y = k \mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j \in S} \exp(w_j^\top x + b_j)}$$

where $S$ is the set of classes **learned from the annotation corpus**. The
denominator runs over the learned classes only, so:

$$\sum_{k \in S} P(y = k \mid x) = 1$$

and with a confidence floor $\tau$:

$$\hat{y} = \begin{cases}
\arg\max_{k \in S} P(y = k \mid x) & \text{if } \max_k P(y = k \mid x) > \tau \\
\text{abstain} \; (y = \texttt{None}) & \text{otherwise}
\end{cases}$$

**The threshold is a confidence floor, not a class boundary.** A text has exactly
one winning class, so a threshold cannot choose between classes; it can only
decide whether the model is confident enough to name one. Below the floor the
model abstains instead of substituting `neutral` - a class the annotation data
never taught it to use as a dumping ground.

### 3.3 Emotion: one binary logistic regression per label

For each label $\ell$ in the learned emotion set $E$, an independent model
$(\mathbf{w}_\ell, b_\ell)$ is fitted:

$$z_\ell = \mathbf{w}_\ell^\top x + b_\ell, \qquad
p_\ell = \sigma(z_\ell) = \frac{1}{1 + e^{-z_\ell}}$$

$$E(x) = \{\, \ell \in E \mid p_\ell > \tau \,\}$$

There is no normalisation across labels, and none should be added. Because the
heads are independent, any subset of $E$ is reachable:

- $\sum_{\ell} p_\ell$ can be below 1 (nothing fires) or above 1 (several fire);
- $p_\ell$ is a calibrated-ish "this label is present" estimate, not a share.

### 3.4 Why the two heads are not interchangeable

A softmax over emotions would force $\sum_\ell p_\ell = 1$, which asserts that
emotions compete. They do not: "anxious **and** grateful" is a legitimate
annotation, and a softmax could never represent it. The one-vs-rest sigmoid has
no such constraint, and the decision threshold moves per label independently.

Conversely, a single sigmoid cannot express sentiment: there is no way to say
"this document is 70% negative and 30% neutral" with one number, and forcing
sentiment through one binary head would make multi-class sentiment
inexpressible.

---

## 4. Annotation format

Both dimensions are annotation-driven. Nothing in the package hard-codes
`positive`/`negative`/`neutral` or any emotion name; the label set is whatever
the annotation corpus contains. Change the taxonomy and the model changes with
it.

### CSV specification

`data/annotations/sentiment_annotations.csv`

| Column | Required | Notes |
|---|---|---|
| `document_id` | yes | unique within the file |
| `text` | yes | **preprocessed** text (the `normalized_text` the pipeline produced) |
| `sentiment` | yes | exactly one class |
| `language` | no | e.g. `en`, `hi`, `bn`, `or` |
| `script` | no | e.g. `latin`, `devanagari`, `bengali`, `oriya` |
| `is_code_mixed` | no | `true` / `false` |
| `detected_languages` | no | `;`-separated |

`data/annotations/emotion_annotations.csv`

| Column | Required | Notes |
|---|---|---|
| `document_id` | yes | unique within the file |
| `text` | yes | **preprocessed** text |
| `emotions` | yes | `;`-separated; **an empty value is valid** and means "no emotion annotated" |
| `language` | no | as above |
| `script` | no | as above |
| `is_code_mixed` | no | as above |
| `detected_languages` | no | `;`-separated |

`MULTI_VALUE_SEPARATOR` is `;`. Neither CSV has been created in this repository:
the corpora used by the demo are in-code synthetic fixtures, and writing a
committed annotation file that nobody actually annotated would be a fabricated
dataset. Populate these only from a real annotation round.

### Example

```csv
document_id,text,sentiment,language,script,is_code_mixed
s01,this placement season is amazing,positive,en,latin,false
s10,"আমাদের প্রার্থীদের উজ্জ্বল ভবিষ্যৎ দেখে সত্যিই ভালো লাগল",positive,bn,bengali,true
s06,maine bahut achha experience paaya yahan,positive,hi,latin,true
```

```csv
document_id,text,emotions,language,script,is_code_mixed
m01,placement pressure hai aur hostel ka stress bhi hai,stress;frustration,hi,latin,true
b01,আমি খুব উদ্বিগ্ন এবং চাপে ভরা,anxiety;stress,bn,bengali,false
z03,the exam was on monday,,en,latin,false
```

### Loading

```python
from app.services.nlp.sentiment_emotion import (
    SentimentCorpus, EmotionCorpus,
    TfidfLogisticSentimentClassifier, TfidfLogisticEmotionClassifier,
)

sentiment_model = TfidfLogisticSentimentClassifier()
sentiment_model.train(SentimentCorpus.from_csv(
    "data/annotations/sentiment_annotations.csv"
).annotations)

emotion_model = TfidfLogisticEmotionClassifier()
emotion_model.train(EmotionCorpus.from_csv(
    "data/annotations/emotion_annotations.csv"
).annotations)
```

---

## 5. Data requirements, and refusing to pretend

`train()` refuses data it cannot honestly learn from, with an actionable error,
rather than fitting something and reporting a number for it:

| Guard | Sentiment | Emotion |
|---|---|---|
| Empty corpus | refuse | refuse |
| Too few documents | `< min_training_documents` (5) | same |
| Too few classes | `< 2` classes | `< 2` labels |
| Too few examples per class | `< min_examples_per_class` (2) | `< min_positive_examples_per_label` (2) |
| Degenerate label | n/a | refuse if a label is on **every** document |
| Unusable vocabulary | refuse if TF-IDF yields no terms | same |
| Blank / whitespace label | refuse | refuse |
| Duplicate `document_id` | reported, not silently dropped | same |
| Wrong record type | `TypeError` naming the expected class | same |

The emotion degenerate-label guard matters most: if every annotated document
carries `stress`, there is no negative example, the binary head for `stress`
cannot be fitted, and a naive implementation would emit a confident constant
`stress=1.0` for every future document.

`train()` also returns a report rather than nothing: the learned label set, the
per-label evidence, the majority-class share, the feature count, the exact
config, and any warnings. A model that trained without crashing is not thereby
trustworthy, and the report is where you check that.

---

## 6. Evaluation

Metrics are **withheld** when the evaluation data cannot support them. The
report says why, and leaves the metric fields `None` instead of filling them
with a number that looks authoritative.

### Sentiment (single-label)

Reported: accuracy, macro/micro F1, macro/micro precision, macro/micro recall,
per-class F1 / precision / recall, a confusion matrix in a fixed class order,
and `sklearn`'s classification report.

- accuracy is valid here, because there is exactly one gold class per document;
- abstentions are counted as **misses**, not dropped - dropping them would
  quietly inflate the score and hide a badly chosen threshold;
- therefore confusion-matrix row sums sit below their support by exactly the
  number of abstentions, and the warning says so;
- a gold class the model never trained on cannot be produced at all, so such
  documents are also counted as misses, excluded from the per-class metrics,
  and reported under `n_unscoreable`;
- per-class F1 below 5 supporting documents is flagged as unstable;
- classes never predicted are listed, because macro-averages are being penalised
  for a reason the reader should know.

### Emotion (multi-label)

Reported: micro and macro F1 / precision / recall, per-label F1 / precision /
recall / support / predicted count, subset accuracy (exact label-set match),
Hamming accuracy, Hamming loss, and `sklearn`'s classification report.

- micro-F1 is the headline for multi-label work; macro-F1 exposes the labels
  that are doing badly, which micro hides;
- subset accuracy is the strictest number and is the one that answers "did we
  get the whole label set right";
- Hamming accuracy looks flattering on sparse label sets and is labelled as
  inflated in the summary;
- an evaluation set with **no** document annotated with zero emotions triggers
  a warning, because subset accuracy is then measured only on documents that
  were guaranteed to have at least one label.

### Nothing is fabricated

- The demo's numbers are labelled `ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT
  PERFORMANCE` in the banner, in the evaluation section, and in the module
  docstring.
- `evaluate()` detects when the evaluation set overlaps the training set and
  warns loudly that the result measures memorisation, not generalisation.
- Thresholds, the confidence floor and the per-label boundary are decision
  policy, so `sentiment_at(t)` and `emotions_above(t)` re-derive a decision at a
  different boundary **without re-running inference**. Sweeping a threshold is
  free, and the probabilities are identical across thresholds.

---

## 7. Configuration

### Shared (`BaseTextFeatureConfig`)

| Setting | Default | Why |
|---|---|---|
| `solver` | `"lbfgs"` | multinomial and OvR logistic regression, deterministic here |
| `random_state` | `42` | reproducibility |
| `max_iter` | `1000` | lbfgs needs headroom on sparse text |
| `C` | `4.0` | regularisation strength, tunable per corpus |
| `min_df` | `1` | keep rare terms; raise for larger corpora |
| `max_df` | `1.0` | no term is dropped for being common |
| `ngram_range` | `(1, 2)` | bigrams matter more here than in topic modelling: "not good" and "not bad" differ only by a bigram |
| `stop_words` | `None` | see below |
| `token_pattern` | `(?u)\b\w+\b` | script-agnostic |
| `lowercase` | `True` | folds case, so `PPO` and `ppo` are one term |
| `threshold` | `0.5` | confidence floor / per-label boundary |
| `drop_placeholder_tokens` | `True` | see below |
| `use_topic_features` | `False` | topic context is context, not signal, by default |

**Stop words.** Only `None`, `"english"`, or an explicit user-supplied list are
accepted; `"hindi"`, `"bengali"` and `"odia"` are rejected with a message saying
so. Two reasons for the `None` default: in Romanized code-mixed text
(`"hostel ka khana bahut ghatiya"`) the function words carry the code-switch
structure, so removing them is a silent quality loss; and a sentiment lexicon is
not a stop-word list, and swapping one for the other is a common mistake.

**Placeholders.** `<URL>`, `<MENTION>` and `<EMOJI_...>` are stripped from a
*copy* of the text used for featurisation, so the model does not learn
`<emoji_loudly_crying_face>` as a feature. The document's own `text` is
unchanged and the placeholders survive into the result for traceability. Set
`drop_placeholder_tokens=False` to keep them.

**Topic features.** With `use_topic_features=True`, topic id and keywords are
injected as prefixed weak tokens (`topicid7`, `topickeyplacement`) that cannot
collide with ordinary vocabulary. Off by default: the topic id is a prediction,
and feeding a prediction back in as a feature compounds error.

---

## 8. Multilingual and code-mixed limitations

The token pattern `(?u)\b\w+\b` matches Devanagari, Bengali and Odia as well as
Latin, so a single model spans scripts without a language-specific branch. That
is a deliberate design choice, and it has real costs:

- **No stop-word removal** means English function words stay in the vocabulary
  and dilute it. Acceptable at this corpus size, worth revisiting with more data.
- **Negation is represented only through bigrams.** `"i am not happy with the
  hostel"` is a genuine failure of this baseline and is recorded as one in
  `test_known_baseline_misses_are_real`, rather than being left to be discovered
  in production.
- **Script-specific morphology is not modelled.** Bengali and Odia inflect
  (`খারাপ` / `খারাপের`), and a word-level model treats those as unrelated terms.
  A character n-gram range would help; it is not enabled by default.
- **Code-switching inside one token is not handled.** `"khana"` and
  `"khanaa"` are different features, and transliterated emotion words carry no
  special treatment.
- **No translation layer.** Romanized Hindi is treated as its own language, so
  it does not benefit from any Hindi-side data.
- **Language is metadata, not a routing decision.** A per-language model or a
  per-language threshold would be the next step, not this baseline.

---

## 9. Why this is a baseline

- It is interpretable: every prediction is a weighted sum over visible terms.
- It trains in seconds on a laptop, so an annotation round can be iterated in
  days rather than weeks.
- It has no hidden state and no external model downloads, so a result can be
  reproduced from the CSV and the config.
- It is honest about its weaknesses, because the guards and warnings are part of
  the model rather than a comment beside it.

It is not a production classifier, and the held-out numbers in the demo
(~30 training sentences) measure the plumbing, not the approach.

---

## 10. Future neural version (not implemented)

- multilingual sentence encoder (e.g. a LaBSE or XLM-R style checkpoint) for
  `x`, replacing TF-IDF, with the same ABCs and the same annotation contracts
- a shared encoder with two heads: a 3-way softmax for sentiment and a per-label
  sigmoid for emotion, so the dimensions stay separate while sharing parameters
- ordinal or 5-point sentiment instead of 3 classes, which is usually what
  annotators actually agree on
- annotator-disagreement modelling, so `emotion_annotations.csv` can record more
  than one annotator and the label becomes a distribution
- threshold calibration per label on a real held-out split
- character n-grams or byte-level features for Indic morphology

None of this is implemented. The ABCs (`SentimentAnalysis`, `EmotionAnalysis`)
are the seam: a neural classifier implements the same three methods and drops in
without a caller changing.

---

## 11. Persistence

Not implemented. This stage holds models in memory. When persistence is added,
the artifact must record the label set, the full config, the library versions
and the training corpus fingerprint, so a model that is later loaded can be
checked against the annotations it claims to have learned. Persisting a
serialised sklearn object without the label set and config is how a stale model
quietly keeps answering with last quarter's taxonomy.

---

## 12. Usage

```python
from app.services.nlp.sentiment_emotion import (
    SentimentEmotionDocument,
    TfidfLogisticSentimentClassifier,
    TfidfLogisticEmotionClassifier,
)

sentiment_model = TfidfLogisticSentimentClassifier()
sentiment_model.train(sentiment_annotations)

emotion_model = TfidfLogisticEmotionClassifier()
emotion_model.train(emotion_annotations)

documents = [
    SentimentEmotionDocument(
        document_id="p1",
        text="placement pressure hai aur hostel ka stress bhi hai",
        original_text="Placement pressure hai aur hostel ka stress bhi hai 😫",
        language="hi", script="latin", is_code_mixed=True,
        detected_languages=["hi", "en"],
        topic_id=1, topic_label="placement",
        issue_labels=["placement_stress", "hostel"],
    )
]

# One class, or an abstention.
sentiment_result = sentiment_model.predict(documents)
prediction = sentiment_result.predictions[0]
prediction.sentiment                     # 'negative' or None
prediction.top_sentiment()               # winning class regardless of the floor
prediction.sentiment_at(0.3)             # re-derive at another floor, no re-inference

# Independent sigmoids; several may fire.
emotion_result = emotion_model.predict(documents, threshold=0.3)
emotion_prediction = emotion_result.predictions[0]
emotion_prediction.emotions               # ['stress', 'frustration']
emotion_prediction.emotion_probabilities  # full independent vector
emotion_prediction.emotions_above(0.5)   # re-derive at another boundary

# Metadata is carried through untouched in both cases.
prediction.issue_labels, prediction.topic_id, prediction.topic_keywords
prediction.text, prediction.original_text, prediction.language
```

---

## 13. Files

```
backend/app/services/nlp/sentiment_emotion/
  base.py                       SentimentEmotionDocument, shared config,
                                SentimentAnalysis / EmotionAnalysis ABCs
  models/
    annotation.py               SentimentAnnotation / EmotionAnnotation, corpora
    result.py                   predictions, results, training reports
    evaluation.py               SentimentEvaluationReport, EmotionEvaluationReport,
                                evaluate_sentiment(), evaluate_emotions()
    _csv.py                     CSV column specs and read/write
  tfidf_logistic_base.py        shared TF-IDF, placeholder handling, validation
  tfidf_logistic_sentiment.py   single-label sentiment model
  tfidf_logistic_emotion.py     One-vs-Rest emotion model
  __init__.py                   public exports

backend/test_sentiment_emotion.py      192 tests
backend/demo_sentiment_emotion.py      nine-section runnable demo
```

`SentimentAnalysis` and `EmotionAnalysis` are the intended extension points.
A neural implementation drops in behind the same three methods, and no caller
needs to change.
