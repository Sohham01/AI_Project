# Issue–Solution Linking

> **Status:** implemented and tested. **Not evaluated against human annotations** —
> no link-annotation dataset exists in this project, so no link precision, recall or
> F1 figure is quoted anywhere in this document or in the code. See
> [Evaluation](#evaluation-why-there-are-no-accuracy-numbers).

---

## 1. What this stage does

Issue identification answers *what problem is this comment expressing?*
Solution detection answers *does this comment offer an actionable fix?* Both are
questions about a **single comment**.

This stage answers the cross-comment question:

> **Which detected solutions address which identified issues?**

It is the only stage in the pipeline that puts two documents side by side, and
that is exactly why it is a separate component. The first two are per-comment
classifiers: they can be trained, scored, swapped and versioned independently.
Linking is a **relation over a set of comments**, so its failure modes are
different — it can be wrong by matching the wrong comment, by matching a comment
in the wrong thread, or by matching everything to everything. None of those
errors show up in either classifier's metrics, which is why they need their own
component and their own tests.

### The failure this stage is built to prevent

> *Do not simply link every solution comment to every issue.*

The cross product is the trivial implementation, and it is wrong in a way that
looks fine: every pair produces a plausible score, the output fills up, and the
knowledge base fills with confident nonsense. Two rules prevent it here.

1. **Candidate generation is thread-scoped by default.** A solution is only
   considered for an issue in the same `post_id`. Cross-thread matching is a real
   requirement at corpus scale and a real precision problem, so it is a
   configuration decision (`allow_cross_post=False` by default), not a default.
2. **Every score is a weighted blend, not a maximum.** Semantic similarity is one
   signal among four, so a high similarity cannot assert a link alone and a low
   one cannot veto an otherwise strong match.

Every rejected candidate is counted by reason. "Zero links" and "zero links
because no solution cleared the candidate threshold" are very different
situations, and only one of them means the pipeline is working.

---

## 2. Separation from solution detection

The distinction is worth stating precisely, because the two stages sound
similar and are often confused.

| | Solution detection | Issue–solution linking |
|---|---|---|
| Unit of analysis | one comment | a **pair** of comments |
| Question | does this comment offer a fix? | does this fix address *this* issue? |
| Output | `is_solution`, `solution_probability` | an edge between an issue and a solution |
| Can be wrong by | misclassifying a comment | pairing the right comment with the wrong one |
| Ground truth needed | per-comment labels | **annotated pairs** |
| Measured by | per-comment precision/recall | pair precision/recall — *not yet available* |

Linking **consumes** the solution detector's output; it never re-derives it and
never imports it. `is_solution` and `solution_probability` arrive as plain fields
on `SolutionEvidence`, so the upstream stage can be retrained, replaced or renamed
without touching an import in this package.

This is enforced by test, not by convention: an AST check over every module in
this package fails if anything imports `solution_detection`,
`response_classification`, `preprocessing`, `issue_identification` or
`topic_discovery`, or if anything under `app.services.nlp.` outside
`issue_solution_linking` is imported at all.

### What this stage does not answer

- **Is the issue resolved?** A link says a comment *addresses* an issue. It does
  not say anyone used it, that the problem went away, or that the poster accepted
  it. Resolution detection is a separate stage with different evidence
  requirements, and no `is_resolved` field exists on any result here — enforced by
  test.
- **Is the issue still open?** Needs a link *plus* downstream evidence. Not
  modelled.
- **What should the reader do?** No answer generation, no generated solutions.
  The output is an edge between two comments a human already wrote.

---

## 3. Architecture

```
app/services/nlp/issue_solution_linking/
├── __init__.py                   public API
├── base.py                       evidence models, config, abstract interface
├── tfidf_cosine_linker.py        the implementation
└── models/
    ├── __init__.py
    ├── labels.py                 strict cell parsing, canonical label ordering
    ├── candidate.py              which (issue, label, solution) triples get scored
    ├── result.py                 LinkPrediction, LinkingResult, LinkingRunReport
    ├── annotation.py             the schema a future annotated dataset takes
    └── evaluation.py             metrics — refuses to run without annotations
```

Data flow:

```
IssueEvidence[] ─┐
                 ├─► generate_candidates ─► fit TF-IDF over the batch ─► score
SolutionEvidence ┘        (thread-scoped)      (one vocabulary)          │
                                                                            ▼
                                                          LinkPrediction (per triple)
                                                                            │
                                        ┌───────────────────────────────────┤
                                        ▼                                   ▼
                                  LinkingResult                       LinkingRunReport
                              (what was concluded)               (what the run did)
```

`LinkingResult` and `LinkingRunReport` are separate on purpose. A run's
provenance has to survive even when its result is empty — "0 links, 0 candidates
because nothing cleared the threshold" is a reportable fact, and folding it into
a result nobody opens because it is empty is how a broken pipeline goes unnoticed.

---

## 4. Evidence models

`IssueEvidence` carries: `document_id`, `post_id`, `comment_id`, `original_text`,
`normalized_text`, `issue_labels`, `issue_probabilities`, `topic_id`, `language`,
`script`, `is_code_mixed`, `detected_languages`.

`SolutionEvidence` carries: `document_id`, `post_id`, `comment_id`,
`original_text`, `normalized_text`, `is_solution`, `solution_probability`,
`response_type`, `response_probabilities`, `parent_comment_id`,
`parent_comment_text`, `language`, `script`, `is_code_mixed`,
`detected_languages`.

### The text is the representation; the label is metadata

A label alone is not semantic context. `"fees"` is one token; it shares no
vocabulary with *"the hostel mess charges 9000 a month and it is not worth it"*.
A linker that compares labels to solution text will either miss the real match or
invent one on a shared word.

So the representation is `normalized_text` — the text the issue prediction was
actually made from — and `issue_labels` is carried as metadata describing *which*
issue that text expresses. `include_issue_label_in_text` can fold the label into
the text for someone who has measured that it helps; it is **off** by default,
because a one-token label weighted like a sentence is a link manufactured from a
single shared word. There is a test asserting that text overlap outweighs a
shared label token.

### Multi-label is the normal case

"Fees are way too high and the ROI is questionable" is two issues. **Each label
becomes its own link candidate** against the same text, so the linker can say
*"this comment solves the fees problem"* and, separately, *"this comment says
nothing about ROI"* — a distinction collapsing the labels would make impossible.

One honest limitation: in V1 the two labels **receive the same score**, because
the vector represents the text rather than the label. The split buys
*addressability* (a consumer can hold the fees links and the hostel links
separately, and the evaluation unit is one claim rather than one comment); it does
not buy a label-specific score. That would need a label-aware representation, and
neither option has been measured.

### Inputs are never mutated

`normalized_text` is already preprocessed; this stage owns neither that decision
nor the text. Placeholder stripping (`<URL>`, `<MENTION>`, `<EMOJI_...>`) happens
on a **copy the vectoriser reads**. `original_text` and `normalized_text` come out
of a run byte-identical, which is asserted by test after repeated runs.

The audit trail keeps both versions, so a reviewer can see exactly what was
stripped: `LinkPrediction.issue_text` / `solution_text` echo the upstream text
intact, while `signals["issue_text_used"]` / `solution_text_used` record what
TF-IDF actually consumed.

---

## 5. Candidate generation

A solution becomes a candidate for an issue when **both** hold.

**Rule 1 — is it worth scoring at all?**

```
is_solution == True   OR   solution_probability >= candidate_solution_threshold
```

The `OR` is deliberate. The upstream boolean is a decision someone already made
against a reviewed threshold, and re-thresholding it here would quietly overrule a
previous stage. The probability is offered for callers who want a stricter bar,
not as a way to veto a stated `True`. A comment that is neither was never a
solution, and scoring it would let a bare "same bro" acquire a link by lexical
accident.

**Rule 2 — is it in the same discussion?**

| Situation | `thread_relevance` | Default |
|---|---|---|
| Direct reply (`parent_comment_id` == issue's `comment_id`) | `1.0` | accepted |
| Same `post_id` | `1.0` | accepted |
| Different `post_id` | `cross_post_thread_relevance` (0.0) | **rejected** |
| Either side missing `post_id` | `cross_post_thread_relevance` (0.0) | **rejected** |

"Provably a different thread" and "we cannot tell" are counted under separate
reasons (`cross_thread_disabled` and `unresolved_thread`) because they are
different problems with different fixes — one is data, the other is missing ids.
With `allow_cross_post=True` an unresolvable pair is admitted at the cross-post
tier rather than refused, since refusing it would make the option useless for
exactly the rows that need it most.

An issue with no usable label is skipped entirely (`issue_has_no_label`).

**Candidacy is evaluated once per solution**, not once per triple, so the
probability threshold is not re-applied N times per issue.

### Determinism

Candidates are generated in a total order that does not depend on dict iteration,
set iteration or input order: issues in input order (the caller controls that),
labels sorted, solutions in input order. `max_candidates_per_issue` sorts by
document id before truncating — sorting by score instead would make *which*
candidates get scored depend on the cap, since scoring depends on the cap.

---

## 6. Scoring

### TF-IDF

One vocabulary is fitted per run, over **all** issue and solution texts in the
batch, then L2-normalised. Fitting per run is a deliberate V1 choice: a linker
that only ever sees today's batch has no stale-index problem and no train/serve
skew. The cost is that `similarity_score` values are **only comparable within a
run** — a raw similarity from last week's run means nothing next to today's.

Defaults, all in `LinkModelConfig`:

| Setting | Default | Why |
|---|---|---|
| `ngram_range` | `(1, 1)` | see below |
| `token_pattern` | `(?u)\b\w+\b` | keeps every script |
| `stop_words` | `None` | see below |
| `lowercase` | `True` | |
| `sublinear_tf` | `True` | tames repeated terms |
| `norm` | `"l2"` | required for the cosine shortcut |
| `use_idf` / `smooth_idf` | `True` / `True` | standard |
| `min_df` / `max_df` | `1` / `1.0` | keep rare terms; they are often the specific ones |
| `max_features` | `20000` | bounds memory at corpus scale |

**Why unigrams, unlike the classifier stages.** An issue and the solution that
answers it almost never share a bigram: a fees complaint and *"ask the warden to
lower the mess charges"* overlap on individual content words, while the bigrams
each side invents are all distinctive, so they dominate the vector norm and
*lower* the cosine for exactly the pairs that should match most. Measured on a
five-comment thread, switching `(1, 2)` → `(1, 1)` raised a related pair's
similarity from 0.12 to 0.19.

**Why `stop_words=None`.** No Hindi/Bengali/Odia stop-word lists are applied and
no text is ever translated. In Romanized code-mixed text the function words carry
the code-switch structure, so removing them by default is a silent quality loss.

### Cosine similarity

With L2-normalised vectors the cosine denominator is 1, so the implementation is
a dot product:

$$\mathrm{sim}(i, s) = \frac{v_i \cdot v_s}{\|v_i\| \, \|v_s\|}
\quad\text{which for } \|v_i\| = \|v_s\| = 1 \text{ reduces to } v_i \cdot v_s$$

where $v_i$ and $v_s$ are the TF-IDF vectors of the issue text and the solution
text. TF-IDF components are non-negative, so the result lies in $[0, 1]$.

**Zero vectors.** A text with no usable terms (empty, or only placeholders) has a
zero vector. Its similarity against everything is `0.0` — never `NaN`, never a
crash — and the link is flagged `issue_zero_vector` / `solution_zero_vector` with
a warning explaining that the link rests entirely on thread placement and
detector confidence. Silently treating it as a match, or crashing the run, would
both be worse. An empty vocabulary for the whole batch produces the same
treatment plus a run-level warning that the scores are not trustworthy.

### The blend

$$
\mathrm{link\_score}
= w_{\mathrm{sem}} \cdot \mathrm{sim}
+ w_{\mathrm{prob}} \cdot p(\mathrm{solution})
+ w_{\mathrm{thread}} \cdot \mathrm{thread\_relevance}
+ w_{\mathrm{role}} \cdot \mathrm{role\_score}
$$

With the defaults, which sum to 1:

$$
\mathrm{link\_score} = 0.50\,\mathrm{sim} + 0.30\,p + 0.15\,\mathrm{thread} + 0.05\,\mathrm{role}
$$

and the decision is simply

$$
\mathrm{linked} = \mathrm{link\_score} \geq \mathrm{link\_threshold}, \qquad \mathrm{link\_threshold} = 0.5
$$

| Term | Default | Rationale |
|---|---|---|
| `semantic_similarity` | 0.50 | the most informative single signal |
| `solution_probability` | 0.30 | upstream detector confidence, second |
| `thread_relevance` | 0.15 | a strong prior that must not outvote the text |
| `response_role` | 0.05 | weak evidence by construction |

Weights are divided by their sum when `normalise_weights=True` (the default), so
the score stays on a 0–1 scale and is comparable to `link_threshold` regardless of
how the weights are written.

**These weights are a judgement call, not a measurement.** They were not tuned,
because tuning them requires annotated pairs and none exist. They are documented
here so that whoever eventually has annotations knows exactly which numbers to
treat as arbitrary.

**Why a weighted sum and not a maximum.** `max(sim, probability, thread)` looks
reasonable and is wrong: it lets any single strong signal assert a link on its
own, so one lucky vocabulary overlap outvotes a thread mismatch. A weighted sum
requires signals to *cooperate*.

**Response role.** `role_scores = {"solution": 1.0, "advice": 0.8}`, with
`default_role_score = 0.5` for an unknown or absent role. Neutral rather than
zero, because an unclassified comment is not a proven non-solution and zeroing it
would penalise the pipeline for a gap upstream. The role **never gates
candidacy** and **never promotes `advice` into a solution** — it adjusts evidence
weight only.

### Confidence is not a probability

```
confidence = clip(1 - 2 * |link_score - link_threshold|, 0, 1)
```

`0.0` exactly at the decision boundary, rising to `1.0` at either extreme. It
answers *"how far from the cut is this?"* — which is precisely the question a
reviewer should ask of a borderline link.

It is emphatically **not** a probability that the link is correct. Nothing here
has been calibrated, because there is nothing to calibrate against. It is named
`confidence` because that is what consumers ask for, and it is defined in
`config_dict()`, in the result docstring and here so nobody mistakes it for a
posterior. An uncalibrated number presented as a probability is how a knowledge
base ends up asserting things at 0.9.

### A note on what the default threshold does and does not guarantee

The three non-semantic terms sum to exactly `0.50` at their maxima. So a solution
with `solution_probability = 1.0`, thread relevance `1.0` and role `solution` —
but **zero** lexical overlap — scores exactly `0.50` and is asserted at the default
threshold.

That is intentional: a maximally confident, in-thread solution is real evidence
even without vocabulary overlap, and a lexical gate would discard answers that
rephrase rather than echo. But it means the default threshold is **not** a
semantic gate, and a consumer who needs one should ask for
`LinkPrediction.is_strong_candidate` (`linked` **and** `similarity_score > 0`)
rather than reading `linked` as the whole story.

---

## 7. Results

Every scored candidate is returned, **including those not asserted**. A
`LinkPrediction` with `linked=False` is not a discarded candidate — it is the
evidence for why no edge was published, and it is the only way to tell a
threshold decision apart from a stage that never looked.

`LinkPrediction` fields: `issue_document_id`, `issue_label`,
`solution_document_id`, `similarity_score`, `solution_probability`,
`thread_relevance`, `response_role_score`, `score_components`, `link_score`,
`linked`, `link_threshold`, `confidence`, `model_name`, `model_version`,
`signals`, plus echoed source metadata (`issue_post_id`, `issue_comment_id`,
`issue_text`, `issue_original_text`, and the solution equivalents, `language`,
`script`, `is_code_mixed`).

Model identity is explicit and reported on **every** link, asserted or not:

```
link_model        = "tfidf_cosine_issue_solution"
link_model_version = "v1"
```

Useful lookups on `LinkingResult`:

| Method | Answers |
|---|---|
| `links_for_issue(id)` | every scored link for one issue |
| `links_for_solution(id)` | every scored link using one solution |
| `solutions_for(issue_id, label=None)` | one issue → many solutions |
| `issues_for(solution_id)` | one solution → many issues |
| `labels_for(issue_id)` | which labels were linked |
| `linked_pairs()` | asserted edges as minimal triples |
| `links_at_threshold(t)` | re-decide at `t` **without re-running** |

`links_at_threshold` returns copies with a new `linked` boolean; every other
field, including the scores, is unchanged. A threshold sweep therefore costs no
extra compute, and it does not mutate the result.

---

## 8. Evaluation: why there are no accuracy numbers

Every earlier stage classifies a single comment against per-comment labels, so
`test_*.py` can assert the *mechanics* and a labelled CSV can assert the
*quality*. A linker asserts a **relation**. Its mechanics are testable; its
quality is not, because quality is exactly the question of whether the relation a
human annotated is the relation the model asserted. Only a human can answer that.

**So no link precision, recall or F1 is quoted anywhere in this project.**
`evaluate_linking()` requires a non-empty `LinkCorpus` and raises otherwise:

```
ValueError: cannot evaluate linking without annotated link pairs. This project
ships no link-annotation dataset, so no link precision/recall may be reported.
```

### The schema is ready; the data is not

`LinkAnnotation` is defined in full, and `LinkCorpus` loads it from
`data/annotations/issue_solution_links.csv`:

```
issue_document_id,issue_label,solution_document_id,linked,rationale
```

That file **does not exist and is not fabricated.** A precision number computed
against labels the author invented, or against a fixture the author wrote, is a
number that will be quoted in a report as though it measured the system.

`linked` is a required real boolean and a **blank is refused** — a missing
judgement silently becoming a negative would train nothing and quietly inflate
whatever metric used it. There is no default and no "unknown": an undecided pair
is simply not annotated, and the count of annotated pairs is reported with every
metric.

Triples annotated both `linked=True` and `linked=False` are reported as
annotator disagreements (`on_conflict="report"`, the default) or refused outright
(`on_conflict="raise"`).

### What a real report would have to include

`LinkingEvaluation` computes precision, recall, F1, specificity and accuracy, and
also reports the numbers that make those numbers interpretable:

- `unscored_annotated_pairs` — annotated pairs the linker never produced a
  candidate for. **A metric over a subset is not a metric over the system**: a run
  that scores fewer pairs can look better purely by declining to be judged on
  hard ones.
- `uncovered_positives` — annotated positives never scored, so recall is
  optimistic.
- `conflicting_judgements` — pairs a human marked both ways.
- `note` — a fixed reminder of where the corpus came from.

Every ratio is `Optional[float]`. A corpus with no annotated negatives has no
measurable specificity, and reporting `0.0` there would be a fabricated number;
`None` with an explanation is the honest value.

---

## 9. Limitations

**Lexical similarity only.** TF-IDF matches surface vocabulary. It will not link
"the fee is too high" to "the charge is unreasonable", it cannot match a solution
that is *about* an issue without repeating its words, and it is blind to
negation ("this does **not** fix the fee" scores well on the words `fix` and
`fee`).

**Absolute cosine values are low for short comments.** Real forum comments are
short, and two short comments sharing two content words out of eight terms
legitimately produce a cosine around 0.2. This is why the blend exists and why
`link_score` is not a similarity. It also means the weights — not the raw
similarity — are what carry the decision.

**Cross-lingual pairs do not match.** No translation, no transliteration, no
multilingual embeddings. An English issue and a Bengali solution in the same
thread will score near zero on similarity. The thread and probability terms still
apply, so such a link is possible, but the lexical evidence is absent. Note that
two comments in the *same* script still match well — the Bengali thread in the
demo scores higher than the English ones.

**Code-mixed text is unvalidated.** `is_code_mixed` is echoed and never branched
on. Romanized Hindi, Hinglish and code-switched comments are scored, but nothing
here measures how well, and identifier-dense comments may produce spurious term
matches.

**Fit-per-run has no cross-run comparability.** Cosines are relative to the batch
the vocabulary was fitted on. Re-running on a different batch changes the numbers
even for identical documents.

**The default threshold is unvalidated**, for the same reason as the weights.

**No resolution inference**, by design. See §2.

---

## 10. What a V2 would change

Ordered by expected value, not by ease.

1. **Annotate link pairs.** Everything else is blocked on this. Without it the
   weights, the threshold and the candidate rules cannot be chosen rather than
   guessed.
2. **Dense bi-encoder embeddings** for the similarity term, with the same
   thread-scoped candidate generation and the same blend. This addresses the two
   biggest limitations at once — paraphrase and cross-lingual matching. The
   `IssueSolutionLinking` interface already accommodates it: implement `link()`,
   keep the evidence models, and the consumer code does not change.
3. **Rerank the top-k candidates** with a cross-encoder, scoring the pair
   jointly. Cheaper than full pair enumeration, and it addresses the one thing a
   bi-encoder cannot see: whether the solution answers *this particular*
   question.
4. **Calibrated confidence** — isotonic regression on annotated scores, once
   there is something to calibrate against.
5. **Learned candidate generation** to replace the hand-written thread rules,
   trained on annotated pairs.
6. **Label-aware representations**, so a multi-label issue's two claims can
   receive different scores (§4).

---

## 11. Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `max_features` | `20000` | TF-IDF vocabulary cap |
| `min_df` | `1` | min documents per term (int count or float proportion) |
| `max_df` | `1.0` | max documents per term |
| `ngram_range` | `(1, 1)` | unigrams only; see §6 |
| `stop_words` | `None` | never removed by default |
| `lowercase` | `True` | |
| `sublinear_tf` | `True` | |
| `token_pattern` | `(?u)\b\w+\b` | every script |
| `norm` | `"l2"` | required for the cosine shortcut |
| `use_idf` / `smooth_idf` | `True` / `True` | |
| `semantic_similarity_weight` | `0.50` | |
| `solution_probability_weight` | `0.30` | |
| `thread_relevance_weight` | `0.15` | |
| `response_role_weight` | `0.05` | |
| `normalise_weights` | `True` | divide by the sum so the score stays 0–1 |
| `link_threshold` | `0.5` | |
| `candidate_solution_threshold` | `0.5` | probability that admits a non-`is_solution` comment |
| `require_same_post` | `True` | |
| `allow_cross_post` | `False` | |
| `use_parent_context` | `True` | honour `parent_comment_id` |
| `max_candidates_per_issue` | `None` | cost control, not accuracy |
| `cross_post_thread_relevance` | `0.0` | |
| `parent_child_thread_relevance` | `1.0` | |
| `role_scores` | `{"solution": 1.0, "advice": 0.8}` | |
| `default_role_score` | `0.5` | unknown or absent role |
| `include_issue_label_in_text` | `False` | |
| `drop_placeholder_tokens` | `True` | on a copy only |
| `include_timestamp` | `False` | a clock would break reproducibility |

Invalid configurations are rejected at construction rather than producing
confident nonsense: out-of-range thresholds, negative weights, weights summing to
zero, `min_df` above `max_df`, `min_df`/`max_df` outside their legal domains,
`max_candidates_per_issue` below 1, malformed `ngram_range`, and out-of-range role
scores.

---

## 12. Usage

```python
from app.services.nlp.issue_solution_linking import (
    IssueEvidence,
    SolutionEvidence,
    TfidfCosineIssueSolutionLinker,
)

linker = TfidfCosineIssueSolutionLinker(link_threshold=0.5)
result = linker.link(issues, solutions)

for edge in result.linked_pairs():
    print(edge["issue_document_id"], edge["issue_label"], "<-", edge["solution_document_id"])

# one issue -> many solutions, and back again
result.solutions_for("issue-42")            # every asserted solution for that issue
result.issues_for("solution-99")            # every issue that solution addresses
result.solutions_for("issue-42", "fees")    # restricted to one label

# re-decide without re-running
stricter = result.links_at_threshold(0.7)
```

Run the demo:

```
cd backend
python demo_issue_solution_linking.py
```

The demo uses entirely synthetic comments, labels and probabilities, states that
on every screen, and deliberately prints no accuracy figure.

Run the tests:

```
pytest test_issue_solution_linking.py -q
```

---

## 13. Files

| Path | Contents |
|---|---|
| `backend/app/services/nlp/issue_solution_linking/base.py` | evidence models, `LinkModelConfig`, `IssueSolutionLinking` |
| `backend/app/services/nlp/issue_solution_linking/tfidf_cosine_linker.py` | `TfidfCosineIssueSolutionLinker` |
| `backend/app/services/nlp/issue_solution_linking/models/candidate.py` | candidate generation |
| `backend/app/services/nlp/issue_solution_linking/models/result.py` | `LinkPrediction`, `LinkingResult`, `LinkingRunReport` |
| `backend/app/services/nlp/issue_solution_linking/models/annotation.py` | `LinkAnnotation`, `LinkCorpus` — schema only, no data |
| `backend/app/services/nlp/issue_solution_linking/models/evaluation.py` | `evaluate_linking`, `threshold_sweep` |
| `backend/app/services/nlp/issue_solution_linking/models/labels.py` | strict parsing, canonical label order |
| `backend/test_issue_solution_linking.py` | tests |
| `backend/demo_issue_solution_linking.py` | synthetic end-to-end demo |

No database migration, no configuration change and no modification to any earlier
NLP stage was required or made.
