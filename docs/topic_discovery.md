# Topic Discovery

**Stage:** NLP pipeline, after preprocessing
**Baseline model:** TF-IDF → NMF
**Model metadata:** `topic_model = "nmf"`, `topic_model_version = "v1"`

Automatically discovers the major recurring subjects of discussion in the
Reddit corpus. Topic categories are **not** hard-coded: the model never sees a
list of "expected" IIM topics, and every label is derived from the terms the
model itself learned.

---

## 1. Why topic discovery is separate from issue identification

These are two different questions, and conflating them produces bad analytics.

| | Topic discovery | Issue identification |
|---|---|---|
| Question | *What is being discussed?* | *Is this a problem, and of what kind?* |
| Output | K topics, each with keywords + document mix | Per-document issue label / severity |
| Unsupervised | Yes | Often supervised |
| Cardinality | Small, corpus-dependent (≈ 6–20) | Could be a taxonomy of grievances |
| Stability | Re-run on new data, topics shift | Taxonomy should stay stable |

Concretely, for the IIM Sambalpur Community Intelligence project:

- A post saying *"hostel mess food quality is really bad"* belongs to the
  **hostel/mess** topic. Discovery says *where* the conversation is.
- Whether that same post is a **complaint**, its **sentiment**, and whether it
  is a **recurring institutional grievance** is an *issue* question — a
  downstream stage.

Separating them buys three concrete benefits:

1. **Interpretability.** A discovered topic is a list of terms. A discovered
   issue class is a label whose meaning depends on the classifier. We can audit
   the first directly.
2. **Reusability.** One topic model serves the analytics dashboard, the weekly
   digest, and the issue classifier. The issue layer can consume topic IDs as
   features.
3. **Honest failure modes.** If the corpus is too small to separate topics, the
   topic stage can say so loudly. A topic model silently merging "hostel" and
   "mess" is a visible, diagnosable bug; a mislabelled issue is not.

**Issue identification is explicitly not implemented in this stage.** This
document and this module stop at discovery.

---

## 2. Why NMF was selected as the baseline

The assignment permits LDA, NMF, BERTopic, and Sentence-BERT embeddings. NMF
was chosen for the baseline for five reasons:

1. **Interpretability is free.** Non-negativity is the whole point. Every entry
   of `H` is an "amount of", so a row of `H` reads as *a bag of words for this
   topic*. LDA's topics are probability distributions over a vocabulary that
   must be inspected through word *probabilities*, which is harder to eyeball.
2. **No GPU, no heavyweight model download.** `scikit-learn`'s NMF is CPU-only
   NumPy/SciPy. BERTopic pulls in Sentence-Transformers and a torch install
   (hundreds of MB) for a stage that must run in a scheduled batch job.
3. **Fast enough for iteration on real corpora.** Fitting on tens of thousands
   of Reddit documents takes seconds, so the corpus can be re-clustered every
   night.
4. **Deterministic by construction.** With `init="nndsvda"` (SVD-based
   initialisation) and `random_state` pinned, repeated runs on the same input
   produce byte-identical topics — essential when results are persisted and
   diffed.
5. **Minimal dependency footprint.** One new dependency (`scikit-learn`, which
   already brings `numpy`/`scipy`). Everything else is already in the project.

NMF's known weaknesses (bag-of-words, no synonymy, no cross-lingual alignment)
are exactly the weaknesses that motivate the embedding roadmap in §11.

---

## 3. TF-IDF representation

### 3.1 Formulation

For each document `d` and term `t`:

```
TFIDF(t, d) = TF(t, d) × IDF(t)
```

with

```
TF(t, d)  = ( 1 + log(count of t in d) )        # sublinear_tf = True
IDF(t)    = log( (1 + N) / (1 + df(t)) ) + 1     # smooth_idf = True
```

where `N` is the number of documents and `df(t)` the number of documents
containing `t`.

- **TF is sublinear** (`1 + log count`) so that repeating "placement" thirty
  times in a rant does not make that document a pure "placement" document.
- **IDF is smoothed** so that a term appearing in *every* document gets a small
  positive weight instead of `log(0)`.
- Each row is **L2-normalised** (`norm="l2"`), so document length does not
  dominate the similarity geometry.

Stacking every document into a matrix gives the **document-term matrix**

```
X ∈ R^(n_docs × n_terms)
```

### 3.2 Why TF-IDF, for this corpus

In an IIM subreddit the same function words appear everywhere. IDF automatically
demotes them, so the surviving high-weight terms are the topical ones:
`placement`, `PPO`, `CAT`, `hostel`, `mess`, `internship`, `fees`, `ROI`,
`scholarship`, `coaching`, `interview`. No stop-word list is required to get
there.

---

## 4. NMF — Non-negative Matrix Factorisation

### 4.1 Formulation

NMF approximates the non-negative matrix `X` as a product of two non-negative
matrices:

```
X ≈ W · H
```

| matrix | shape | meaning |
|---|---|---|
| `X` | `(n_docs × n_terms)` | the TF-IDF document-term matrix |
| `W` | `(n_docs × K)` | **document-topic** matrix (loading / membership) |
| `H` | `(K × n_terms)` | **topic-term** matrix (basis / components) |

with `X`, `W`, `H` all element-wise `≥ 0`.

The objective minimised (Frobenius loss, coordinate descent) is

```
min_{W,H ≥ 0}  ½ ‖ X − W H ‖_F²  +  ½ ( α_W ‖W‖₁ + α_H ‖H‖₁ )
```

with `α_W = α_H = 0` by default, i.e. plain Frobenius reconstruction error.

### 4.2 How each part is used

- **Rows of `W` correspond to documents.** Row `i` of `W` says how much of
  each topic document `i` contains. This *is* the topic distribution, and it is
  what `DocumentTopicAssignment.topic_distribution` reports.
- **Columns of `W` correspond to topics.** Column `k` of `W` is the affinity of
  every document to topic `k`; `argmax_k W[i, k]` is document `i`'s dominant
  topic.
- **Rows of `H` represent topics.** Row `k` of `H` is a length-`n_terms`
  weight vector describing topic `k`.
- **The highest-weight terms in `H` become the topic keywords.** The top
  `n_keywords` entries of row `k` (by weight) are the keywords of topic `k`.
  This is the entire labelling mechanism: **labels come from the model, not
  from a predefined category list.**

A document that mixes subjects gets a spread-out row in `W`; a document that
is purely about one subject gets a row with one large value. The
`dominant_topic_score` is that largest value and is therefore a usable
confidence proxy.

### 4.3 Determinism

- `init="nndsvda"` — a deterministic SVD-based initialisation, not random.
- `random_state=42` — pins the stochastic fallback path.
- `solver="cd"`, `beta_loss="frobenius"` — deterministic coordinate descent.
- Keyword ordering uses a total order: weight descending, then term
  ascending. Ties can never reorder between runs.
- `dominant_topic_id` uses `argmax`, which resolves ties to the lowest topic
  id.
- **The discovery timestamp is opt-in** (`include_timestamp=False` by default)
  because a wall-clock field would make otherwise-identical results differ
  between runs. The demo asserts the determinism explicitly.

---

## 5. How topics are discovered (end to end)

1. **Input.** A list of `TopicDocument`s whose `text` is *already* the
   `normalized_text` produced by `MultilingualTextPreprocessor`. This stage
   never re-runs preprocessing, and never mutates raw text or preprocessing
   output — it works on internal copies.
2. **Placeholder stripping.** `<URL>`, `<MENTION>` and `<EMOJI_...>` tokens are
   preprocessing artefacts, not lexical content. They are removed from the
   *vectoriser's copy* so they cannot form an artificial "URL topic" or
   "emoji topic". The document's `text` is echoed back to the caller
   untouched. Disable with `NMFConfig(drop_placeholder_tokens=False)`.
3. **Vectorisation.** `TfidfVectorizer` → sparse `X`.
4. **Factorisation.** `NMF(n_components=n_topics).fit_transform(X)` → `W`; the
   components give `H`.
5. **Keywords.** Top `n_keywords` terms per row of `H`.
6. **Assignments.** Each row of `W` → a `topic_distribution`; `argmax` → the
   dominant topic.
7. **Metadata.** Model name, version, full configuration, library versions and
   any warnings are attached so a run can be reproduced and audited.

---

## 6. Why categories are not hard-coded

Hard-coding an IIM category list (e.g. `["placement", "hostel", "mess",
"fees", "academics"]`) would defeat the assignment's core requirement. The
system must surface what the community *actually* talks about, including
themes nobody anticipated.

The implementation enforces this structurally:

- There is no category list, seed-keyword list, or label table anywhere in the
  module.
- Labels are `"Topic <id>"` — deliberately content-free. The *meaning* of a
  topic is its keyword list, which is model output.
- A test asserts no label contains a domain word such as `"placement"`.

The category a topic corresponds to is a downstream, human/analyst decision
made *from* the keywords — not an input to the model.

---

## 7. Multilingual and code-mixed behaviour

This corpus is Hindi/Bengali/Odia in native script, Romanized Hindi
("Hinglish"), and English, frequently inside a single sentence. The
preprocessing stage guarantees no text is translated. Topic discovery inherits
that guarantee and adds the following care:

- **No translation, ever.** A Devanagari document stays Devanagari. The
  `script`, `language`, `is_code_mixed` and `detected_languages` metadata is
  carried through to the assignment for downstream filtering.
- **Script-agnostic tokenisation.** `token_pattern=r"(?u)\b\w+\b"` keeps every
  script: under `re.UNICODE`, `\w` matches Devanagari, Bengali and Odia as well
  as Latin. A test asserts non-Latin terms reach the vocabulary.
- **No language-specific stop-word removal by default.** `stop_words` defaults
  to `None`. This is a deliberate, enforced decision: the validator *rejects*
  any list other than `None`, `"english"`, or an explicit user-supplied list.
  Hindi/Bengali/Odia stop-word lists are not shipped, because Romanized
  function words (`hai`, `ka`, `kya`, `se`, `ne`) carry the code-switch
  structure and removing them by default would silently rewrite the language
  of the corpus.
- **Domain terms preserved.** `IIM`, `CAT`, `PPO`, `ROI`, `MBA`, `placement`,
  `internship`, `hostel`, `mess` are never stop-words and always survive as
  features. `lowercase=True` folds `IIM` and `iim` to one term; disable with
  `TFIDFConfig(lowercase=False)`.
- **Code-mixed phrases** are captured by the default `ngram_range=(1, 2)`, so
  `kya scene`, `mess food` and `placement PPO` appear as features.

### Honest limitations

1. **Indic scripts are whitespace-segmented.** Devanagari/Bengali/Odia do not
   use spaces between orthographic syllables, so `आईआईएम संबलपुर में` splits into
   grammatical fragments (`अ`, `च`, `छ`). Native-script posts therefore tend to
   cluster together into one script-specific topic rather than aligning with
   their English/Hinglish counterparts. A language-aware tokenizer is required
   to fix this.
2. **No cross-lingual alignment.** TF-IDF shares no features between
   "placement" (Latin) and "प्लेसमेंट" (Devanagari), so the same subject
   appears as two topics. This is the single biggest multilingual weakness of
   the baseline.
3. **Filler topics.** With `stop_words=None`, high-frequency Romanized function
   words can co-occur strongly enough to form a filler topic
   (`ka / kya / scene hai`). The demo demonstrates that passing an explicit,
   clearly-labelled neutral list fixes this — and the default stays
   conservative.
4. **Romanized vs. Latin ambiguity.** The code-mix lexicon in
   `code_mix_detector.py` flags words like `ta`, `na`, `se` ambiguously; topic
   modelling cannot disambiguate them.

---

## 8. Configurable parameters

### `NMFTopicDiscoverer` constructor

| Parameter | Default | Notes |
|---|---|---|
| `n_topics` | `8` | Falls back to `NMFConfig.n_topics`, then to `8` |
| `n_keywords` | `10` | Keywords listed per topic |
| `tfidf_config` | `TFIDFConfig()` | All vectoriser knobs |
| `nmf_config` | `NMFConfig()` | All factorisation knobs |
| `include_timestamp` | `False` | Opt-in, so results stay reproducible |

`discover(documents, n_topics=...)` overrides the topic count per call.

### `TFIDFConfig`

| Parameter | Default | Notes |
|---|---|---|
| `max_features` | `20000` | Vocabulary cap by term frequency |
| `min_df` | `1` | `1` for small corpora; raise to `2`–`3` on real data |
| `max_df` | `1.0` | Lower (e.g. `0.6`) to drop corpus-wide terms |
| `ngram_range` | `(1, 2)` | Unigrams + bigrams keeps code-mixed phrases |
| `stop_words` | `None` | `None` / `"english"` / explicit `list[str]` only |
| `lowercase` | `True` | Case-folds `IIM` → `iim` |
| `sublinear_tf` | `True` | `1 + log(count)` |
| `token_pattern` | `(?u)\b\w+\b` | Script-agnostic |
| `norm` | `"l2"` | Row normalisation |
| `use_idf` / `smooth_idf` | `True` | Turn off only for ablations |

### `NMFConfig`

| Parameter | Default | Notes |
|---|---|---|
| `n_topics` | `8` | Ignored if passed to the constructor |
| `n_keywords` | `10` | |
| `init` | `"nndsvda"` | Deterministic; alternatives `nndsvd`, `random` |
| `solver` | `"cd"` | Coordinate descent (deterministic) |
| `beta_loss` | `"frobenius"` | Or `kullback-leibler` (KL divergence) |
| `random_state` | `42` | Pinned for reproducibility |
| `max_iter` | `300` | Raise if a convergence warning appears |
| `tol` | `1e-4` | |
| `alpha_W` / `alpha_H` | `0.0` | L1 regularisation for sparsity |
| `l1_ratio` | `0.0` | |
| `drop_placeholder_tokens` | `True` | Strip `<URL>`/`<MENTION>`/`<EMOJI_...>` |

Invalid values raise `ValueError` at construction time with an actionable
message (see `test_topic_discovery.py::TestInvalidConfiguration`).

### Choosing `n_topics` in practice

- Must satisfy `n_topics <= n_documents` and `n_topics <= vocabulary_size`;
  otherwise the stage raises `too-small corpus`.
- On a real corpus, sweep `n_topics` and read the keywords. Pick the value
  where topics stop splitting into sub-topics and start merging.
- Very large `K` with `min_df=1` yields near-duplicate topics.

---

## 9. Input / output contract

### Input — `TopicDocument`

```python
TopicDocument(
    document_id="post-042",     # stable id
    text="...",                 # ALREADY preprocessed (normalized_text)
    post_id="post-042",         # or comment_id
    comment_id=None,
    original_text="...",        # raw, never modified
    language="hi", script="latin",
    is_code_mixed=True,
    detected_languages=["hi", "en"],
)
```

### Output — `TopicDiscoveryResult`

| Field | Meaning |
|---|---|
| `topic_model` / `topic_model_version` | `"nmf"` / `"v1"` |
| `n_topics`, `n_documents`, `vocabulary_size` | Run statistics |
| `topics[]` | `topic_id`, `label`, `keywords`, `keyword_weights`, `document_count`, `document_ratio` |
| `documents[]` | One per input document: full metadata, `topic_distribution`, `dominant_topic_id`, `dominant_topic_label`, `dominant_topic_score` |
| `config` | Every knob that affects the output (JSON-serialisable) |
| `library_versions` | scikit-learn / numpy versions |
| `discovered_at` | `None` unless `include_timestamp=True` |
| `warnings` | Duplicates, dropped placeholders, NMF non-convergence |

Helpers: `result.get_document(document_id)` and `result.topic_by_id(topic_id)`.

**Duplicate document ids** are tolerated and reported in `warnings`; every
document still receives its own assignment, so nothing is silently dropped.

**Guarantee:** input documents are never mutated. The discoverer works on
internal copies and returns the caller's `text` and `original_text` verbatim.

---

## 10. Usage

```python
from app.services.nlp.topic_discovery import NMFTopicDiscoverer, TopicDocument

discoverer = NMFTopicDiscoverer(n_topics=8)
result = discoverer.discover(documents)

for topic in sorted(result.topics, key=lambda t: -t.document_count):
    print(topic.label, topic.document_count, topic.keywords)

for doc in result.documents:
    print(doc.document_id, doc.dominant_topic_id, doc.dominant_topic_score)
```

Run the demo:

```
python demo_topic_discovery.py
```

Run the tests:

```
python -m pytest test_topic_discovery.py -q
```

---

## 11. Limitations and the roadmap to embeddings / BERTopic

### Current limitations

- **Bag of words.** No word order, so "not worth it" and "worth it" look alike.
- **No synonymy.** "mess food" and "hostel dining" are unrelated features.
- **No cross-lingual alignment** (§7.2), the most serious issue for this corpus.
- **Topic identity is unstable across runs on different corpora.** `Topic 3`
  in one snapshot may be a different subject in the next. Downstream storage
  must key on `(corpus, n_topics, config_hash, keyword_fingerprint)`, never on
  the bare integer id.
- **`K` is a hyper-parameter, not discovered.** There is no coherence score
  driving selection; `K` is chosen by sweeping and reading keywords.
- **Short posts are weak.** A one-line comment has too few terms for a
  reliable distribution; the score will be near-saturated on whichever topic
  happens to share one term.
- **Emoji semantics are dropped from the features.** The preprocessing stage
  deliberately preserves emoji as tokens for *sentiment*; topic discovery drops
  them as non-topical, so this stage is blind to emoji-only posts.

### Why move to embeddings later

- **Multilingual alignment.** Sentence-BERT / multilingual-e5 / Indic-BERT
  place "placement", "प्लेसमेंट" and "placement ka scene" in the same region of
  the vector space — solving §7.2 outright without a translation step and
  without violating the no-translation rule.
- **Semantics beyond co-occurrence.** Embeddings capture similarity TF-IDF
  cannot, collapsing "mess food" and "hostel dining" into one topic.
- **No vocabulary ceiling.** TF-IDF is limited to observed terms; embeddings
  generalise to unseen vocabulary.
- **Better short-text behaviour**, which matters for comment-level rows.

### Planned approach

`BERTopic(embedding_model="paraphrase-multilingual-MiniLM-L12-v2", ...)`
over the same `TopicDocument` inputs: embed → cluster → c-TF-IDF for topic
labels → assignment. The `TopicDiscovery` ABC is the seam: a `BERTopicDiscoverer`
returns the identical `TopicDiscoveryResult`, so the persistence layer, API and
dashboard do not change. Embedding model choice would be a per-language
decision (Indic-native models for native-script subsets), and the same
`random_state` discipline plus deterministic clustering (HDBSCAN with fixed
seed) keeps runs reproducible.

The NMF baseline stays in place as a fast, dependency-light, fully
interpretable reference to diff any embedding-based result against.

---

## 12. Files

| File | Purpose |
|---|---|
| `backend/app/services/nlp/topic_discovery/base.py` | `TopicDiscovery` ABC, `TopicDocument`, `TFIDFConfig`, `NMFConfig`, result models |
| `backend/app/services/nlp/topic_discovery/nmf_topic_discoverer.py` | `NMFTopicDiscoverer` — TF-IDF → NMF |
| `backend/app/services/nlp/topic_discovery/__init__.py` | Public exports |
| `backend/test_topic_discovery.py` | 103 tests |
| `backend/demo_topic_discovery.py` | 28-document multilingual demo |
| `docs/topic_discovery.md` | This document |
