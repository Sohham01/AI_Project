# Resolution / Unresolved Issue Detection

> **Status:** implemented and tested. **Not evaluated against human annotations** —
> no resolution-annotation dataset exists in this project, so no accuracy, macro F1
> or confusion matrix is quoted anywhere in this document or in the code. See
> [Evaluation](#9-evaluation-why-there-are-no-accuracy-numbers).

---

## 1. What this stage does

Issue identification answers *what problem is being expressed?* Solution detection
answers *does this comment offer a fix?* Linking answers *does that fix address
that problem?*

This stage asks the fourth question, and the only one about a **whole
discussion** rather than a comment or a pair:

> **Considering the whole discussion, how resolved does this issue appear?**

### The failure this stage is built to prevent

> **One solution present ≠ issue resolved.**

This is the whole point of the stage, and it is easy to lose. A single confident
"just restart the router" in a forty-comment thread where nobody else agreed is
**not** a resolved issue — the density term is `1/40`, and no amount of lexical
confidence in that one comment can compensate. A thread with four proposed fixes
and no follow-up is not resolved either.

Resolution is a property of the *conversation*. No per-comment classifier can see
it, which is why it cannot be folded into an earlier stage: linking's metrics are
about pairs and would stay perfect while the conversation went nowhere.

### What this stage does NOT answer

- **What should the reader do?** No answer generation, no generated solutions.
- **Should the issue be closed?** No workflow or state transition. This stage
  reports an assessment; a downstream system may or may not act on it.
- **Is the author positive?** Sentiment and emotion are reported elsewhere and are
  **not inputs here** — see [Sentiment](#sentiment-is-not-an-input).

---

## 2. The three statuses, and why there are only three

| Status | Meaning |
|---|---|
| `resolved` | Enough relevant, well-linked solution evidence to call the issue settled |
| `partially_resolved` | Some real progress, but the evidence does not reach the resolved bar |
| `unresolved` | No meaningful solution evidence attached to this issue |

A fourth status, `insufficient_evidence`, was considered and **rejected**. It
answers a question about *our confidence* rather than about *the issue*, and a
consumer asking "which issues remain unresolved?" needs a **partition of the
issues**, not a bucket of non-answers.

A thin discussion and a genuinely unresolved discussion can both be `unresolved`.
What separates them is `evidence_sufficient`, which is on every result precisely
so a consumer can filter on it:

```python
thin = [r for r in batch.results if not r.evidence_sufficient]
```

---

## 3. The score

```
resolution_score = w_density * solution_density
                + w_strong   * strong_solution_density
                + w_strength * best_link_score
                + w_conf     * solution_probability_mean

solution_density        = linked_solution_count / max(total_comments, 1)
strong_solution_density = strong_link_count    / max(total_comments, 1)
```

Default weights, normalised to sum to 1:

| Term | Weight | What it measures |
|---|---|---|
| `solution_density` | **0.30** | How much of the discussion is a relevant answer |
| `strong_solution_density` | **0.20** | How much of it is a *strong* link |
| `best_link_score` | **0.25** | How well the best answer actually matches |
| `solution_probability_mean` | **0.25** | How confident the solution detector was |

### Why density outweighs lexical strength

The weights are a **judgement call, not a measurement.** They were not tuned,
because tuning requires annotated resolution outcomes and none exist.

The two density terms together carry **0.50** against **0.25** for link strength,
and the reason is that *coverage* is harder to produce by accident than *similar
words*. Give lexical overlap the dominant weight and one reply that happens to
reuse the poster's vocabulary — even proposing the wrong fix — carries the verdict.
Density rewards a discussion that attracted several genuinely relevant answers.

A documented boundary, pinned by a test so it cannot change silently: because
strength and confidence together are worth 0.50, **one solution the linker is very
sure about outranks three it is unsure about**. That is arguably right — a
confidently matched fix is stronger evidence than three speculative matches — but
it is a judgement, not a derivation, and is stated as one.

### Thresholds

Two effective cut-points, three named thresholds:

```
score <  unresolved_upper                -> unresolved
unresolved_upper <= score < resolved_lower -> partially_resolved
score >= resolved_lower                   -> resolved
```

| Threshold | Default | Meaning |
|---|---|---|
| `unresolved_upper` | 0.30 | Below this: `unresolved` |
| `partial_upper` | 0.55 | Nominal upper edge of the partial band |
| `resolved_lower` | 0.55 | At or above this: `resolved` |

Three names for two cut-points is inherent to naming three bands: the partial band
needs a lower and an upper edge, and the upper edge **is** the resolved band's
lower edge. A configuration that sets them inconsistently
(`unresolved_upper > partial_upper > resolved_lower`) is **rejected at
construction**, not silently resolved, because overlapping bands let one score
satisfy two conditions at once.

Thresholds are checked at `__init__`, not lazily, so a bad configuration cannot
produce quietly wrong statuses for an entire run.

### Every number is reproducible

Each result carries its own `score_components`, `weights` and `thresholds`, and
`recompute_score` re-applies the reported equation to the reported terms:

```python
result.resolution_score          # 0.463750000000
result.recompute_score           # 0.463750000000  -> identical
```

A reader can reproduce any verdict by hand without opening the implementation.

---

## 4. Evidence

`IssueResolutionEvidence` is a **neutral container of counts and scores.** It
carries no reference to any other stage's types, so whoever assembles the evidence
— a pipeline, a script, a human — is not this stage's business.

| Field | Scored? | Notes |
|---|---|---|
| `total_comments` | denominator | The single most important field for interpreting the result |
| `discussion_volume` | denominator | Overrides `total_comments` when given |
| `linked_solution_count` | ✅ | Solutions linked to **this issue and this label** |
| `strong_link_count` | ✅ | Subset of the above |
| `best_link_score` | ✅ | Or derived from `link_scores` |
| `solution_probability_mean` | ✅ | Or derived from `solution_probabilities` |
| `solution_candidate_count` | ❌ | **Reporting only** — see below |
| `response_counts`, `agreement_count`, … | ❌ | Recorded for V2; no measurement justifies a weight |
| `temporal` | ❌ | Recorded for V2; see [Temporal evidence](#temporal-evidence-recorded-not-scored) |

### Candidates are not evidence

`solution_candidate_count` is a count of comments assessed as possible solutions
*anywhere in the discussion*. A candidate that was never linked to this issue is
evidence about the **discussion**, not about whether **this issue** got fixed.

Letting candidates raise the score would make any busy thread look resolved: 8
candidates and 0 links still scores exactly `0.0`. There is a test for precisely
this.

### Only the right pair counts

A solution linked to a *different* issue in the same thread must not raise this
number. That is the easiest way to make a linker mistake look like a resolution,
which is why the warning on such a result says so explicitly: *"This is a
statement about the linker's output for this pair, not proof that no fix exists in
the discussion."*

### Validation

An internally inconsistent record is **refused**, with a message naming the fields
involved — because the caller's first question is "which of my numbers is wrong?":

- negative counts
- `linked_solution_count > total_comments` (solutions are comments in the thread)
- `strong_link_count > linked_solution_count`
- `solution_candidate_count < linked_solution_count`
- `discussion_volume < total_comments`
- a supplied density that disagrees with `linked / total`
- a supplied `best_link_score` that disagrees with `max(link_scores)`
- scores outside `[0, 1]`, or `NaN` / `inf` anywhere

The resolver would happily compute a plausible score from an impossible record.
Refusing it is the point: otherwise the impossible record is laundered into a
confident verdict.

---

## 5. Insufficient evidence

When `effective_discussion_volume < minimum_discussion_volume` (default **3**):

- the **status is unchanged** — still one of the required three;
- `evidence_sufficient` is `False`;
- a **warning** is attached explaining why;
- `confidence` is multiplied by `insufficient_evidence_confidence_scale` (0.5).

The score is deliberately **not** capped. Capping it would manufacture a verdict
out of the guard: a thin discussion would be pushed down to `unresolved` by a rule
that has nothing to do with whether the issue was fixed. A thin thread is
**less certainly judged**, not thereby declared unresolved.

A run where *nothing* cleared the guard is flagged in the run report as a likely
misconfiguration — a `minimum_discussion_volume` set above the corpus's typical
thread size — rather than a finding that every issue is unresolved.

---

## 6. Multi-label issues

**One evidence record is one `(issue_id, issue_label)` pair**, and resolution is
evaluated per pair.

A "fees" answer and an "roi" answer are evidence for different questions.
Averaging them into one verdict for the document is how a solved fees problem gets
recorded as a solved ROI problem:

```python
batch.statuses_for_issue("post-1005")
# {'fees': 'partially_resolved', 'roi': 'unresolved'}
```

The same `issue_id` with different labels is **not** a duplicate. An *exact*
`(issue_id, issue_label)` repeat keeps its **first** occurrence and is listed in
the run report.

> **A known limitation:** first-wins is order-dependent. With two contradictory
> records for one pair there is no rule that is both permutation-invariant and picks
> one of them. The choice is therefore disclosed in `report.duplicate_pair_keys`
> rather than hidden, so a reader knows the verdict depended on the order records
> arrived in.

---

## 7. Temporal evidence — recorded, not scored

```python
TemporalEvidence(
    solution_after_issue=True,        # a "fix" posted before the complaint
    follow_up_count=4,                #   cannot address it
    follow_up_resolution_signal=True, # the closest thing to an outcome signal
)
```

A solution appearing *after* the issue is a **precondition** for being an answer,
not proof that it worked. Order alone carries no information about effectiveness.

All three fields are preserved, echoed on the result, and counted in the run
report — and **none of them enters `resolution_score`**, because no measurement
justifies a weight for them. When outcomes are annotated, they become the obvious
V2 input, and the run report says so when any are present.

Note that `follow_up_count` is a proxy for *engagement*, not an outcome. Counting
comments is not the same as observing that something worked.

---

## 8. Sentiment is not an input

A happy poster and a fixed problem are **different facts**.

A thread where the author thanks everyone and describes the problem still
unresolved scores `0.0` here, and would score well on sentiment. That is correct:
nothing in the evidence says anyone confirmed a fix.

Wiring positive sentiment into the score would make this stage quietly reward
issues people are happy about, and the resulting correlation with satisfaction
would be mistaken for evidence of resolution. The same reasoning excludes emotion,
comment count as a proxy for engagement, and unlinked solution candidates.

---

## 9. Evaluation: why there are no accuracy numbers

Resolution is a judgement **about a discussion**. Correctness means "does a human
reading this whole thread agree?", and only a human reading the thread can say.
Unlike the earlier stages, there is nothing to test against.

So:

- **No resolution-annotation dataset exists in this project, and none was
  fabricated.**
- The **schema and loader ship**, ready for an annotation round.
- `evaluate_resolution` **raises** on an empty corpus. The refusal is the feature:
  a caller reaching for a metric with nothing annotated should get an error, not a
  number.
- The synthetic cases in the tests and the demo are **plumbing checks**, labelled
  as such, and are never a measurement.

```python
evaluate_resolution(batch, ResolutionCorpus())
# ValueError: cannot evaluate resolution without adjudicated gold statuses.
# This project ships no resolution-annotation dataset, so no accuracy, F1 or
# confusion matrix may be reported.
```

### Annotation format

One row per annotated issue-label pair:

```csv
issue_id,issue_label,gold_status,rationale
post-1001,fees,resolved,"two linked fixes, poster confirmed"
post-1002,network,unresolved,"single suggestion, no follow-up"
```

`gold_status` is required and must be one of the three statuses — a missing
judgement silently defaulting to `unresolved` would bias every resolution rate
computed over the corpus in one direction, and a resolution rate is exactly the
number most likely to be quoted.

`issue_label` is optional. A multi-label issue needs **one judgement per label**,
since two labels of the same document can resolve differently. An annotation
without a label is matched by `issue_id` alone, which is only sound for a
single-label issue.

### Guards, because these metrics are easy to publish misleadingly

Every number this stage could produce looks like a clean count. Four guards are
applied and reported alongside:

1. **A single-status corpus is flagged.** 100% `unresolved` gold gives accuracy
   1.0 to a resolver that always answers "unresolved" — a perfect score for a
   model that learned nothing.
2. **Undefined metrics stay `None`.** A status with no gold examples has no
   recall. It is **excluded from macro F1**, not counted as `0.0`, which would
   penalise the model for a gap in the annotations.
3. **Unmatched records are reported.** A metric over a subset is not a metric over
   the stage, so uncovered results and unmatched annotations are both listed.
4. **Ambiguous multi-label matches are refused, not guessed.** An unlabelled
   annotation meeting a multi-label result is excluded rather than attributed to
   whichever label sorted first.

`ResolutionEvaluation.format_report()` prints all of these, and always states that
its figures describe only the supplied corpus.

---

## 10. Confidence is a boundary distance, not a probability

```
distance from the nearest status boundary
  -> 0.0 on a boundary
  -> 1.0 at least 0.5 away from every boundary
  -> halved when evidence is insufficient
```

A score exactly on a boundary gets `0.0`: a verdict that a rounding change would
flip is not a confident verdict, whatever the status.

This answers *"how far from the nearest boundary is this score?"* and **nothing
more**. It is emphatically **not** a probability that the status is correct —
nothing here has been calibrated, because there are no annotated outcomes to
calibrate against. It is labelled `conf=` in every summary for that reason.

A consequence worth noticing: a score of `0.5519` is `resolved` with
`conf=0.004`. That is honest. It is *barely* resolved, and reporting it as
confident would be the misleading part.

---

## 11. Determinism

Same evidence plus same configuration gives the same numbers, always. No sampling,
no random seeds, no wall-clock field unless `include_timestamp=True`.

Two runs that disagree mean the input differed, which is diagnosable. A stage whose
output drifts between runs cannot answer "did this get worse?" — the main thing
anyone wants from a resolution metric.

Inputs are never modified: records, the caller's list, the config, and any supplied
score lists all come back unchanged. Each result holds a **deep copy** of its
evidence, so a result cannot quietly rewrite its own history if the caller mutates
the original record afterwards.

---

## 12. API

```python
from app.services.nlp.resolution_detection import (
    EvidenceBasedResolver, IssueResolutionEvidence, ResolutionStatus,
)

resolver = EvidenceBasedResolver()
result = resolver.detect(record)            # one issue or many
batch = resolver.detect_batch(records)      # a ResolutionBatchResult
```

Useful queries on a batch:

| Method | Returns |
|---|---|
| `unresolved()` | Strictly `unresolved` — the stage's headline question |
| `not_resolved()` | `unresolved` + `partially_resolved` — the open-issue work list |
| `with_status(status)` | One status, stable order |
| `results_for_issue(id)` | Every label assessed for one document |
| `statuses_for_issue(id)` | `{label: status}` |
| `thin_evidence()` | Verdicts resting on too little discussion |
| `by_pair_key()` | `{issue_id\|label: result}` |

> `unresolved()` and `not_resolved()` are **not** the same. A triage list that
> merged them would understate the work left to do; a partially-resolved issue is
> still open.

---

## 13. Files and verification

```
backend/app/services/nlp/resolution_detection/
    base.py                        statuses, evidence, config, abstract interface
    evidence_based_resolver.py     the deterministic scorer
    __init__.py                    public API
    models/
        labels.py                  parsing and numeric checks
        evidence.py                derivation and coherence validation
        result.py                  result, batch and run-report models
        annotation.py              annotation schema + CSV loader (no data ships)
        evaluation.py              guarded metrics (refuses without a corpus)
backend/test_resolution_detection.py
backend/demo_resolution_detection.py
```

```
pytest test_resolution_detection.py -q     # 166 passed
python demo_resolution_detection.py        # exit 0
```

The demo prints `ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE` and uses
only synthetic evidence. It reports **no** accuracy or F1 for the stage.
