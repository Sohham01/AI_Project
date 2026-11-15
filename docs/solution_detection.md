# Solution Detection

**Stage:** NLP pipeline, per-comment binary classification
**Model:** `solution_model = "tfidf_logistic_binary"`, `solution_model_version = "v1"`
**Package:** `backend/app/services/nlp/solution_detection/`
**Demo:** `backend/demo_solution_detection.py`
**Tests:** `backend/test_solution_detection.py`

---

## 1. What this stage asks

One question, per comment:

> Does this comment provide an actionable solution/recommendation?

| `is_solution` | Meaning | Example |
| --- | --- | --- |
| `True` | Offers a step the reader can take | "apply for the scholarship on the portal" |
| `True` | Offers a recommendation, however hedged | "you should contact the placement office" |
| `False` | No actionable step | "same bro" |
| `False` | Experience, opinion, agreement, rebuttal | "i faced this last year", "actually fees aren't that high" |

The label space is exactly two targets, in a fixed canonical order:

```
not_solution = 0        # no actionable solution
solution     = 1        # provides a solution
```

The order is not cosmetic. It is the integer class index the model is fitted on,
so index 1 always means "provides a solution", and two runs produce
byte-identical confusion-matrix axes.

### What this stage is not

- **Not a taxonomy.** `solution_type` is free-form annotation data (see §4).
- **Not sentiment.** "apply for the scholarship" carries no polarity.
- **Not Issue-Solution Linking.** Deciding *which issue* a detected solution
  answers is a separate, later stage. `post_id` and `issue_labels` are carried
  and echoed, never used as features.
- **Not Resolution Detection.** A comment can propose a solution that nobody
  used. Whether a problem was actually fixed is a different question.
- **Not a context-aware model.** V1 reads comment text alone (§7).

---

## 2. Advice is a candidate, not a solution

This is the distinction the stage exists to keep honest, so it is worth being
precise about.

The completed response-classification stage answers a *different* question — "what
role does this comment play?" — with one of six roles. Two of those roles are
relevant here:

| `response_type` | Is it a solution candidate? | `is_solution` |
| --- | --- | --- |
| `advice` | yes | **from the detector** |
| `solution` | yes | **from the detector** |
| `agreement` | no | `False` |
| `personal_experience` | no | `False` |
| `opinion` | no | `False` |
| `counterargument` | no | `False` |

**A candidate is not a solution.** Both `advice` and `solution` roles make a
comment *worth checking*; neither guarantees it contains anything actionable:

- "you should contact the placement office" → advice **and** a real solution.
- "you should probably do something about this, not sure how" → advice, a
  deontic marker with no step to take, **not** a solution.

The second case is a *guideline* decision, not a model decision. The annotation
guideline used for this stage says a deontic marker alone is not an actionable
recommendation: if the reader cannot act on it without inventing the step, it is
`is_solution=false`. Where the guideline is ambiguous, that ambiguity shows up in
`rationale` disagreements during adjudication — it does not get papered over by
the model.

### How the separation is enforced

`models/candidate.py` holds the candidate logic and is deliberately structural
rather than conventional:

```python
class CandidateAssessment(BaseModel):
    response_type: Optional[str] = None
    is_solution_response: bool = False
    is_advice_response: bool = False
    is_candidate: bool = False
    candidate_types: List[str] = [...]
    reason: str = ""
    is_solution: Optional[bool] = None   # <-- None by construction
```

`is_solution` starts as `None` and is only ever set by
`CandidateAssessment.with_prediction(...)`, which requires a real
`SolutionPrediction`. So "advice implies solution" cannot be written down without
running the detector, and the candidate flags never reach the logistic
regression. A test asserts the vectoriser's input is unaffected by
`response_type`.

The candidate role names are configurable strings
(`DEFAULT_CANDIDATE_RESPONSE_TYPES = ("advice", "solution")`), and the
comparisons are case-insensitive. This package **does not import**
`response_classification`; the role arrives as a plain `Optional[str]`, so
neither stage can break the other by being replaced.

---

## 3. Model: TF-IDF + binary logistic regression

### Why a sigmoid, and why that is exact rather than approximate

scikit-learn's binary `LogisticRegression.predict_proba` applies a softmax over
two columns. For K = 2 that is algebraically identical to a single sigmoid:

```
softmax([z0, z1])[1] = e^z1 / (e^z0 + e^z1)
                     = 1 / (1 + e^(z0 - z1))
                     = sigmoid(z1 - z0)
```

So `predict_proba(X)[:, 1] = σ(w·x + b)` for a single weight vector. No separate
head is needed, the probability is a genuine sigmoid, and a test asserts the
identity numerically rather than describing it.

TF-IDF features are L2-normalised, so the linear score is a function of the
cosine similarity of the comment to the weighted mean solution / non-solution
centroid pair. That is the whole model: a similarity-weighted vote over unigrams
and bigrams, pushed through a logistic link.

### Configuration (all defaults, as shipped)

```python
SolutionModelConfig(
    # --- TF-IDF ---
    max_features=20000,
    min_df=1,
    max_df=1.0,
    ngram_range=(1, 2),
    stop_words=None,
    lowercase=True,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    # --- logistic regression ---
    C=4.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42,
    # --- decision rule ---
    threshold=0.5,
    abstain_threshold=None,
    # --- data-adequacy guards ---
    min_training_documents=10,
    min_examples_per_class=2,
    imbalance_warning_share=0.8,
    # --- preprocessing placeholders ---
    drop_placeholder_tokens=True,
)
```

Why each non-obvious default:

- **`ngram_range=(1, 2)`.** A large share of the actionable signal is
  phrase-shaped: "try to", "you should", "you must", "apply for", "contact the".
  A unigram model reads "should" as a generic word rather than as a deontic
  marker and loses the class.
- **`stop_words=None`.** "you should", "you must" and "try to" are the class's
  most reliable features, and an English stop-word list deletes precisely those.
  No Hindi/Bengali/Odia stop-word lists are supplied either: in Romanized
  code-mixed text those function words carry the code-switch structure, so
  removing them by default would be a silent quality loss. `"english"` is
  accepted if you want it; any other language name is refused.
- **`token_pattern=r"(?u)\b\w+\b"`.** Keeps every script, so a Bengali "আবেদন করতে
  হবে" reaches the same vectoriser as its English counterpart.
- **`class_weight="balanced"`.** Most comments contain no solution, so an
  unweighted model converges on "no", reports high accuracy, and finds nothing —
  the failure that would silently break Issue-Solution Linking downstream.
  Balancing trades a little negative-class recall for positive-class recall. It
  is configurable because a corpus with a genuine majority of solutions is a
  legitimate case.
- **`imbalance_warning_share=0.8`.** A 5:1 skew already lets an
  always-majority baseline dominate a report that quotes only accuracy. The
  check is strictly greater than, and it warns rather than refusing.
- **`drop_placeholder_tokens=True`.** `<URL>`, `<MENTION>` and `<EMOJI_...>` are
  preprocessing artefacts. Threads share URLs and mentions, so those tokens recur
  often enough to become high-order terms with no solution signal. The
  *document* keeps its placeholders — the text handed to the model is a copy.

### Training guards

`train()` refuses, with an actionable message:

| Condition | Why |
| --- | --- |
| empty corpus | nothing to fit |
| fewer than `min_training_documents` (10) | memorise, not generalise |
| only one target class present | a corpus of only solutions cannot teach the difference — there is nothing to contrast against |
| a target with fewer than `min_examples_per_class` (2) | would be predicted arbitrarily |
| blank / unrecognised `is_solution` | a missing annotation must never become a negative example |
| blank `text` | solution detection reads the comment body |
| one `document_id` with two different targets | the corpus contradicts itself; no defensible automatic tie-break |
| invalid configuration (`class_weight`, `threshold`, `norm`, `ngram_range`, `stop_words`, `C`, `min_df > max_df`, …) | the **classifier constructor** raises, naming the knob — the config model itself stays importable without scikit-learn |
| texts that produce no usable TF-IDF features | reported as a vocabulary problem, not a traceback |

It warns (never refuses) about: class imbalance with the majority label and the
score an always-majority baseline would get, positives with no `solution_type`,
negatives that carry one anyway, and duplicate `document_id`s.

---

## 4. Annotations

`SolutionAnnotation` is the hand-off format between human annotators and the
model. `is_solution` is the **only** target.

### CSV schema

Canonical column order (`CSV_COLUMNS`):

| Column | Required | Notes |
| --- | --- | --- |
| `document_id` | **yes** | unique per comment; a conflict is refused |
| `text` | **yes** | already preprocessed (`normalized_text`) |
| `is_solution` | **yes** | `true`/`false` (also `1`/`0`, `yes`/`no`, `t`/`f`, `y`/`n`, `on`/`off`) |
| `solution_type` | no | free-form; any non-blank string |
| `post_id` | no | traceability only |
| `comment_id` | no | traceability only |
| `language` | no | default `unknown` |
| `script` | no | default `unknown` |
| `is_code_mixed` | no | default `false` |
| `detected_languages` | no | pipe-separated, e.g. `hi\|en` |
| `response_type` | no | **neutral** triage context, not a feature |
| `rationale` | no | why the annotator decided as they did |
| `evidence` | no | a quoted span supporting the decision |

Extra columns in a user's file are ignored, with one deliberate exception:
`topic_id` is read if present, for traceability alongside the topic-discovery
stage. It is not part of `CSV_COLUMNS`, so `to_csv()` does not write it back.
Missing optional columns fall back to the defaults above.

The intended location is:

```
data/annotations/solution_annotations.csv
```

**That file is not shipped.** It is produced by the human annotation workflow.
Fabricating labels would make every downstream number fiction, and the test
suite asserts the file is absent. Until it exists, `train()` has nothing real to
learn from and the demo is the only thing that can run — with synthetic data and
a loud disclaimer.

```csv
document_id,text,is_solution,solution_type,post_id,comment_id,language,script,is_code_mixed,detected_languages,response_type,rationale,evidence
ann-001,"apply for the scholarship and submit the required documents",true,financial_aid,p12,c301,hi,latin,true,hi|en,solution,"names the portal and the documents","submit the required documents"
ann-002,"you should contact the placement office",true,placement_preparation,p12,c302,en,latin,false,en,advice,"directs the reader to a specific office","contact the placement office"
ann-003,"same bro",false,,p12,c303,en,latin,false,en,agreement,"no actionable step",
ann-004,"you should probably sort this out somehow",false,,p12,c304,en,latin,false,en,advice,"deontic marker, no concrete step",
ann-005,"i faced the same issue last year",false,,p12,c305,en,latin,false,en,personal_experience,"experience, no recommendation",
```

A blank `is_solution` cell is refused with the line number, **never** read as
`False`. That is the single most damaging failure mode available to this stage:
a missing annotation silently becoming a negative teaches the detector to ignore
solutions.

### `solution_type` is free-form, and never learned

There is deliberately no module-level list of categories. A fixed list would
encode one community's vocabulary into a general model, and every new community
would need a code change and a version bump to express a category its annotators
invented. So `solution_type` is:

- **validated for shape** (a present value must be a non-blank string),
- **normalised** for reporting (whitespace trimmed, blank means absent),
- **reported** in the training report as `observed_solution_types` and
  `solution_type_counts`, sorted for byte-identical output across runs,
- **never** used as a target or a feature in V1.

It is reserved for the future solution-extraction stage (§10). Two annotation
combinations are contradictory and are reported as corpus-level warnings rather
than refused, because both are legitimate states of an annotation in progress:

- `is_solution=true` with no `solution_type` — a real solution whose category is
  undecided. Category coverage is unquotable until it is resolved.
- `is_solution=false` with a `solution_type` — usually an annotator recording
  the comment's *subject* rather than judging it a solution. A signal that the
  guideline needs to address.

---

## 5. Prediction, threshold, abstention

```python
classifier = TfidfLogisticSolutionClassifier()
report    = classifier.train(annotations)          # refuses bad/insufficient data
result    = classifier.predict(comments)          # is_solution + probability
metrics   = classifier.evaluate(held_out)         # None where data is too thin
```

`SolutionDetectionResult` carries, per comment: `is_solution`,
`solution_probability`, `not_solution_probability`, `threshold`, `abstained`,
the candidate flag, and every input field echoed back (`text`, `original_text`,
ids, language metadata, topic/issue metadata, `parent_comment_text`).
`SolutionDetectionResult` adds counts, the full config snapshot, library versions
and warnings.

### The threshold is a decision boundary

```
is_solution = (p >= threshold)
```

Unlike the multiclass stage, where a threshold can only trigger abstention, here
it is what turns a probability into the boolean a linker consumes. **Below the
threshold, `is_solution` is `False`** because that is a real answer the corpus
supports — not because the model is silent. Otherwise `n_abstained` would mean
"the negative class", which is not information.

### Abstention is opt-in, and sits *above* the boundary

`abstain_threshold` covers the band `threshold <= p < abstain_threshold`: the
boundary says yes, but not confidently enough to assert a link downstream. Such
a comment is reported `is_solution=False, abstained=True`.

- Default `None` → never abstain, so evaluation is clean and a non-zero
  `n_abstained` means something.
- `predict(abstain_threshold=...)` overrides per call; omitting the argument
  inherits the configured value, and passing `None` explicitly disables
  abstention for one run without mutating the instance. The two are
  distinguishable via a module-level `INHERIT` sentinel.
- `0` is refused rather than silently treated as disabled — an empty band is far
  more likely a mistake than an intention.

Evaluation counts an abstention as a miss for accuracy and excludes it from the
confusion matrix, so turning the band on cannot inflate the metrics.

---

## 6. Evaluation

### Which number to quote

The positive class is `solution`, and its **F1** is the number that matters.
Everything else is context for it.

- **accuracy** is the most familiar figure and the most misleading one here.
  Most comments contain no solution, so an always-`False` model scores the
  negative share and looks excellent while finding nothing. Reported — because it
  is asked for, and because it is easy to misread — but never alone, and never
  without support counts.
- **precision** is the number a downstream linker should optimise first. A false
  positive asserts that a problem was answered when it was not, and that error
  propagates into the knowledge base. A missed solution is a retrieved comment
  nobody needed; a wrong one is a wrong fact.
- **recall** bounds how much of the actionable content is even reachable. Low
  recall is recoverable; low precision is not, because the links are already
  published.
- **F1** is their harmonic mean and only summarises them when they are close.
  `precision_recall_gap` is reported so the gap is visible.

### Confusion matrix

Axis order is canonical `[not_solution, solution]`:

```
[[true_negative,  false_positive],
 [false_negative, true_positive]]
```

The four counts are also exposed as named fields, because indexing a 2×2 matrix
is exactly the sort of thing that quietly produces a transposed metric. The
off-diagonal cells are the finding: a large `false_positive` cell means the model
asserts solutions where there are none.

### Honesty rules

Nothing is fabricated. `SolutionEvaluationReport` metrics are `Optional[float]`:

| Condition | Result |
| --- | --- |
| fewer than `MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS` (10) documents | `is_meaningful=False`, `reason` names the floor, every metric `None` |
| one target absent from the gold set | withheld — a confident `0.0` would be indistinguishable from a genuinely bad model |
| no scorable rows at all | withheld |
| fewer than `MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS` (2) solutions | **warned**, not withheld — one example makes the positive rates swing, but suppressing the whole report would hide the other class |
| a target with fewer than `MIN_SUPPORT_FOR_STABLE_PER_CLASS_F1` (2) examples | **warned**: that class's own F1 is unstable and should not be quoted alone |

Support counts survive a withheld report, because counts are facts about the
data rather than claims about the model.

Two special cases are counted rather than dropped:

- `y_pred = None` → an **abstention** (or a prediction the evaluator could not
  read as a boolean). Counted as a miss for accuracy, excluded from the matrix.
- `y_true = None` → **unscoreable** (unannotated, or a value that is not a
  boolean). Same treatment.

Values that are neither booleans nor recognised boolean spellings become `None`
rather than being coerced. `bool("maybe")` is `True`, so coercion would score a
mistyped annotation cell as a confident *positive* and inflate every rate. The
warnings state exactly how many rows were set aside, because dropping rows
quietly improves the numbers.

`evaluate()` also warns when the evaluation set overlaps the training set, in
whole or in part — those numbers measure memorisation, not generalisation, and a
perfect `1.0` on a small corpus is how synthetic metrics end up presented as
project performance.

### Evaluation caveats

- **No real annotations exist yet.** Every number produced by
  `demo_solution_detection.py` comes from hand-written synthetic comments. They
  demonstrate the code paths and nothing else.
- **The demo's held-out set is tiny and lexically clean**, so its matrix is close
  to diagonal by construction. A real corpus will not be.
- **A 5:1 or worse imbalance is the expected real-world case**, which is exactly
  why `class_weight="balanced"` is the default and why accuracy is never the
  headline.
- **Metrics are only as good as the guideline.** The advice/solution line in §2
  is a human decision; a model cannot adjudicate it.
- **Precision on a synthetic set says nothing about precision on Reddit.** No
  number in this stage has been measured on real data yet.

---

## 7. Context, and what V1 ignores

`SolutionDocument` accepts optional context so a context-aware detector can be
added later **without changing the public interface**:

| Field | V1 behaviour |
| --- | --- |
| `parent_comment_text` | carried, echoed, **not** read |
| `topic_id`, `topic_label`, `topic_keywords` | carried, echoed, **not** read |
| `issue_labels`, `issue_probabilities` | carried, echoed, **not** read |
| `response_type`, `response_probabilities` | candidate triage **only** — never reaches the model |
| `post_id`, `comment_id`, `original_text` | traceability |
| `language`, `script`, `is_code_mixed`, `detected_languages` | echoed; the stage does not branch on them |

`config_dict()["context"]` states this explicitly, and tests assert the
vectoriser's input text is a function of `text` alone. This is the honest shape
for V1: the features are documented rather than implied, so nobody downstream
assumes a thread-aware model that does not exist.

`original_text` is preserved and never fed to the model — the model sees
`text`, which is expected to be the preprocessing stage's `normalized_text`.

---

## 8. Architecture and independence

```
backend/app/services/nlp/solution_detection/
├── base.py                     SolutionDocument, SolutionModelConfig, SolutionDetection ABC
├── tfidf_logistic_binary.py    TfidfLogisticSolutionClassifier
└── models/
    ├── labels.py               binary targets, boolean parsing, free-form solution_type
    ├── candidate.py            advice/solution candidate logic
    ├── annotation.py           SolutionAnnotation, SolutionCorpus, CSV schema
    ├── result.py               prediction / result / training-report models
    └── evaluation.py           binary metrics, confusion matrix, meaningful-data guards
```

Rules, enforced by tests in `TestArchitectureIsolation`:

- No module outside `solution_detection` is imported, except the Python standard
  library, scikit-learn, NumPy and Pydantic.
- No import of `response_classification`, `preprocessing`, `topic_discovery`,
  `issue_identification`, `sentiment_emotion` or `language_analysis_service`.
- No database, session or ingestion symbols (`sqlalchemy`, `SessionLocal`,
  `get_db`, `app.models`, `app.services.ingest`, `praw`).
- No Issue-Solution Linking or Resolution Detection: no such modules are
  imported, and neither `SolutionPrediction` nor `SolutionDetectionResult`
  carries a linked-issue or resolution field, so a caller cannot read a link out
  of this stage by accident.

`SolutionDetection` is an ABC. Any implementation — a fine-tuned encoder with a
one-unit head, a prompt-based judge — must be trainable from
`SolutionAnnotation` data and must return a `SolutionDetectionResult` carrying
the full probability.

Determinism: TF-IDF is deterministic, `lbfgs` is deterministic, class indices
come from the fixed binary target order rather than from a fitted encoder, and
`random_state` is pinned. Two runs on the same data produce identical
probabilities. Wall-clock timestamps are opt-in (`include_timestamp=False`) for
the same reason.

---

## 9. Reproducing

```bash
cd backend

# demo, on synthetic data, with the disclaimer banner
.\.venv\Scripts\python.exe demo_solution_detection.py

# this stage's suite
.\.venv\Scripts\python.exe -m pytest test_solution_detection.py -q

# whole pipeline regression
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 10. Limitations and roadmap

### Known limitations of V1

- **Negation scope is weak.** "do NOT apply for the scholarship" keeps every
  feature of a solution and flips only the target. There is no negation
  handling.
- **Emoji are lossy.** The preprocessing stage's `<EMOJI_...>` tokens can stand
  for many visually distinct emojis, so tone-carrying emoji contribute nothing.
- **Romanized code-mixed text is out of distribution.** A model trained on
  English leans on `apply` and `should`; "aapko scholarship ke liye apply karna
  chahiye" scores lower than its English counterpart for reasons that have
  nothing to do with whether it is a solution.
- **No cross-lingual transfer.** Bangla- or Devanagari-script text is in the
  vocabulary only if the annotations contain it. None is claimed.
- **Sentence-level only.** A comment split across two comments — a question and
  an answer — is judged on the answer's text alone.
- **No thread or issue context** (§7).
- **The TF-IDF vocabulary is per-fit.** Nothing is persisted, so a refit on a
  different corpus silently changes every probability. There is no saved-model
  artefact and no model registry in this stage.

### Roadmap

1. **Real annotation round.** Populate
   `data/annotations/solution_annotations.csv`, adjudicate `rationale`
   disagreements, and re-run evaluation on a genuinely disjoint held-out split.
   Until then, no performance claim about this stage is meaningful.
2. **Solution extraction.** `solution_type` is already carried for this. Turning
   it into a predicted field — and then into an extracted, structured step
   ("apply" / "on the portal" / "before the deadline") — is the natural next
   stage, and it needs the guideline to settle what counts as a *type*.
3. **Issue-Solution Linking.** Deciding which issue a detected solution answers
   needs the thread structure and the issue labels, both already carried as
   optional metadata. This stage will not do it: a per-comment boolean and a
   cross-comment link are different problems, and conflating them would make
   the linker's errors look like detector errors.
4. **Context-aware V2.** `parent_comment_text` and topic metadata are already on
   the input model, so a context-aware variant needs no interface change. The
   `context` block in `config_dict()` is where the difference would show up.
5. **Calibration.** The threshold is currently a plain probability cut. For a
   linker, calibrated probabilities would let the abstention band mean something
   specific rather than merely being a second threshold.
