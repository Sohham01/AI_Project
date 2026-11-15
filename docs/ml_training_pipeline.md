# ML Training Pipeline

**Current stage: classical ML baseline infrastructure.**

ANN and CNN are **NOT** implemented yet.  The purpose of this stage is to
establish a reproducible baseline — load real annotations, split them
deterministically, train the existing classical models, evaluate them
consistently, save artifacts, and record everything in a local registry —
so that future neural experiments have something honest to compare against.

---

## 1. Why the training framework is separate from the NLP implementations

The NLP modules under `backend/app/services/nlp/` are the **source of model
behaviour**.  They own:

- the annotation types (`SentimentAnnotation`, `EmotionAnnotation`, …),
- the corpus loaders (`SentimentCorpus.from_csv`, …),
- the classifier classes (`TfidfLogisticSentimentClassifier`, …),
- and their train / predict / evaluate public APIs.

The ML framework under `ml/` **does not replace any of that**.  It
orchestrates it.  The framework:

1. loads annotation data through the existing corpus loaders,
2. converts records into the task-specific annotation types via an adapter,
3. splits them deterministically,
4. hands the annotations to the existing classifier's `train()` method,
5. evaluates using the classifier's own `evaluate()` output,
6. saves a joblib artifact that contains the trained model plus metadata, and
7. records the run in a local JSON registry.

Nothing in the NLP modules was modified to make the ML framework work.  If a
future model (ANN, CNN, transformer) implements the same public interface —
`train(annotations)`, `predict(documents)`, `evaluate(annotations)` — it can
be plugged into the same trainer without rewriting the framework.

---

## 2. Package structure

```
ml/
    __init__.py              # package root
    data/
        __init__.py
        dataset.py           # TrainingRecord, synthetic data, real annotation loading
        splitting.py         # deterministic split + leakage checks
    training/
        __init__.py
        trainer.py           # Trainer, model factory, records→annotation adapter
    evaluation/
        __init__.py
        metrics.py           # ClassificationMetricSet, compute_metrics
        reports.py           # EvaluationReport, build_evaluation_report, class distribution
    configs/
        __init__.py
        config.py            # TaskConfig, TaskType, ModelRegistry, TASK_METADATA
    artifacts/
        .gitkeep             # joblib artifacts and registry.json land here
```

---

## 3. Dataset abstraction

### TrainingRecord

`ml.data.dataset.TrainingRecord` is the generic transport the splitter and
trainer use.  It carries:

| Field | Type | Meaning |
|---|---|---|
| `document_id` | `str` | Unique document identity (used for leakage checks) |
| `text` | `str` | The text to classify |
| `label` | `str` | Task-specific label; single class for single-label, pipe-joined for multi-label |
| `post_id` | `Optional[str]` | Source post ID (metadata, not used for training) |
| `comment_id` | `Optional[str]` | Source comment ID (metadata) |
| `language` | `str` | Detected language |
| `script` | `str` | Detected script |
| `is_code_mixed` | `bool` | Whether the text mixes scripts/languages |
| `detected_languages` | `List[str]` | All languages detected in the text |
| `topic_id` | `Optional[int]` | Topic assignment from topic discovery |
| `metadata` | `Dict[str, Any]` | Task-specific extras (e.g. `solution_type` for solution detection) |

Not every field is required for every task.  The framework does not require
fields it does not use.

### Label helpers

- `split_label(value: str) -> List[str]` — splits a pipe-joined label string
  into its constituent labels.  Single-label strings pass through unchanged.
- `join_label(labels: Sequence[str]) -> str` — joins labels back into a
  pipe-separated string.

### Real annotation loading

`load_annotation_dataset(task_name, path, base_dir)` delegates to the existing
corpus loaders:

| Task | Loader used |
|---|---|
| sentiment | `SentimentCorpus.from_csv` |
| emotion | `EmotionCorpus.from_csv` |
| issue | `AnnotationCorpus.from_csv` |
| response | `ResponseCorpus.from_csv` |
| solution | `SolutionCorpus.from_csv` |

When the file does not exist the function raises
`AnnotationNotFoundError` with a clear message that includes the path and
suggests synthetic data as a plumbing-check fallback.  It does **not** invent
data.

### Synthetic demo data

`make_synthetic_records(n, label_options, single_label, seed)` generates
SYNTHETIC records explicitly labelled as such.  The text snippets are
plausible multilingual college-placement sentences.  Every record's
`document_id` includes the string `"synthetic-"` so it is identifiable.

**Important:** synthetic data is for plumbing verification only.  Any demo or
test that uses it must clearly label it as synthetic.  The metrics produced
from synthetic data are **not** project performance.

---

## 4. Annotation workflow

Real annotation files live under:

```
data/annotations/
    sentiment_annotations.csv
    emotion_annotations.csv
    issue_annotations.csv
    response_annotations.csv
    solution_annotations.csv
```

Each file follows the CSV schema defined by its task's annotation module.  See
the individual NLP module docs for the exact column specification:

- Sentiment / Emotion: `backend/app/services/nlp/sentiment_emotion/docs/`
- Issue: `backend/app/services/nlp/issue_identification/docs/`
- Response: `backend/app/services/nlp/response_classification/docs/`
- Solution: `backend/app/services/nlp/solution_detection/docs/`

When those files are absent the framework raises
`AnnotationNotFoundError`.  The trainer can fall back to synthetic data with
`use_synthetic=True`, but it always records `source="synthetic"` in the
dataset metadata and includes a warning in the pipeline result.

**No annotation CSV files are fabricated by this framework.**  The project
does not ship with real annotation files; they are created by the human
annotation workflow and placed in `data/annotations/` when ready.

---

## 5. Train / validation / test split

### SplitSpec

`ml.data.splitting.SplitSpec` controls one split operation:

| Field | Default | Meaning |
|---|---|---|
| `train_size` | `0.7` | Fraction for training |
| `validation_size` | `0.15` | Fraction for validation |
| `test_size` | `0.15` | Fraction for test |
| `random_state` | `42` | Seed for deterministic shuffling |
| `stratify` | `True` | Whether to attempt stratification |
| `min_train_examples` | `1` | Minimum training examples required |
| `min_validation_examples` | `1` | Minimum validation examples |
| `min_test_examples` | `1` | Minimum test examples |

The three size fields **must sum to 1.0** (within floating-point tolerance).
The constructor validates this and raises `ValueError` otherwise.

### Deterministic splitting

`split_records(records, labels, spec)` returns a `SplitResult` with three index
lists: `train`, `validation`, `test`.

The split is **fully deterministic**: given the same records, labels, and
`random_state`, the same indices come out every time.  The splitter:

1. sorts records by `(label, document_id)` for stable input ordering,
2. applies a seeded RNG shuffle (or a round-robin stratified interleave when
   stratification is used),
3. slices the ordered indices into three contiguous ranges,
4. returns the three index lists sorted for readability.

### Stratification

Stratification is used for **single-label** classification tasks when the label
distribution supports it.  For **multi-label** tasks the splitter deliberately
does **not** call ordinary `train_test_split(stratify=...)` because a
multi-label binarized matrix would silently drop documents whose label set is
absent from a particular fold.

The splitter reports two booleans:

- `stratify_available` — whether stratification was statistically safe for this
  dataset (enough records, enough classes, no multi-label, minimum class counts
  met).
- `stratify_used` — whether stratification was actually applied (i.e.
  `stratify=True` **and** `stratify_available=True`).

When stratification is requested but not applied, a warning explains why
(too few records, multi-label dataset, too few classes, rare class).

### Minimum size enforcement

After computing proportional sizes, the splitter enforces the minimum
constraints.  If the dataset is too small to satisfy all minimums simultaneously
(e.g. `min_train_examples=50` but only 30 records total), the splitter raises
`ValueError` with a message saying exactly what is wrong and what to adjust.

**A document must never appear in more than one split.**  This is the core
invariant.  The splitter verifies it internally (via `verify_no_overlap`) and
raises `RuntimeError` if the split somehow produced overlapping indices (which
should only happen as an internal bug, not from user data).

### SplitResult

| Field | Meaning |
|---|---|
| `train` | List of indices into the original records |
| `validation` | List of indices |
| `test` | List of indices |
| `random_state` | Seed that produced this split |
| `stratify_used` | Whether stratification was applied |
| `stratify_available` | Whether stratification was safe |
| `warnings` | Any non-fatal notices (duplicate IDs in source data, stratification not applied, etc.) |

`apply_split(records, split)` converts a `SplitResult` into three record lists
`(train, validation, test)`.

---

## 6. Data leakage

### What the framework checks

`check_leakage(train_records, validation_records, test_records)` returns a
report with:

| Field | Meaning |
|---|---|
| `leakage_detected` | `True` if any overlap was found |
| `train_n`, `validation_n`, `test_n` | Sizes of each split |
| `cross_split_id_collisions` | Human-readable list of document-ID overlaps |
| `cross_split_text_collisions` | Human-readable list of exact-text overlaps |
| `warnings` | Consolidated warning messages |

The checker detects:

1. **Duplicate document IDs across splits** — if the same `document_id` appears
   in train and validation, or train and test, or validation and test, that is
   leakage.  The report names the pair and the count.
2. **Exact duplicate texts crossing splits** — even if document IDs differ, the
   same text appearing in two splits is leakage (it means the model can
   memorise the exact string).  The checker fingerprints each text with SHA-256
   and compares the fingerprint sets across splits.

### What the framework does NOT do

- It does **not** silently remove duplicate records.  It reports them and lets
  the caller decide.
- It does **not** drop evaluation rows that the model cannot score.  Those rows
  are counted as misses so the numbers reflect reality (see the NLP module
  docs for how each task handles abstentions and unscoreable gold labels).

### Leakage in the trainer

The `Trainer.run()` pipeline calls `check_leakage` on the three splits before
training.  If leakage is detected, the pipeline:

- records the leakage report in the result,
- adds a warning to the pipeline warnings,
- still proceeds (so the caller can see what happened), but marks evaluation
  reports as `is_meaningful=False` with a leakage warning.

### Verifying no leakage yourself

```python
from ml.data.splitting import split_records, apply_split, check_leakage

records = [...]            # your TrainingRecord list
labels = [r.label for r in records]
split = split_records(records, labels)
train, val, test = apply_split(records, split)
report = check_leakage(train, val, test)
assert not report["leakage_detected"], report["warnings"]
```

---

## 7. Class imbalance

The framework reports class distribution but does **not** automatically alter
class weights.  Each existing classifier already has its own class-weighting
configuration (e.g. `class_weight="balanced"` on the logistic regression).  The
framework leaves that decision to the model and the caller.

### Class distribution reporting

`report_class_distribution(label_counts, total)` returns:

| Field | Meaning |
|---|---|
| `total` | Number of records |
| `distribution` | Per-label `{count, percentage}` |
| `imbalance_ratio` | Majority-class share of the dataset |
| `majority_class` | Label with the most examples |
| `majority_share` | Same as imbalance_ratio |
| `warning` | Non-None when imbalance_ratio > 0.8, with an explanation that accuracy is inflated by skew and macro-F1 / per-class recall should be preferred |

The existing NLP evaluation modules also emit their own imbalance warnings
(e.g. "class X has fewer than 2 examples; its per-class F1 is unstable").

---

## 8. Evaluation

### Unified metric suite

`ml.evaluation.metrics.ClassificationMetricSet` is the single metric container
the framework uses everywhere.  It handles both single-label and multi-label
classification through `compute_metrics(y_true, y_pred, labels, multilabel, ...)`.

#### Single-label metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Fraction of correct predictions (valid for single-label) |
| `macro_f1` | Per-class F1 averaged equally |
| `micro_f1` | Pooled F1 over all decisions |
| `weighted_f1` | Per-class F1 weighted by support |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `per_class` | Dict per label: `{precision, recall, f1, support}` |
| `confusion_matrix` | 2-D matrix over present labels |
| `confusion_matrix_labels` | Label order for the matrix |

#### Multi-label metrics

| Metric | Meaning |
|---|---|
| `micro_f1` | Pooled F1 over all (document, label) decisions — headline number for imbalanced multi-label |
| `macro_f1` | Per-label F1 averaged equally — exposes per-label failure |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `subset_accuracy` | Exact label-set match (always ≤ micro-F1 on genuinely multi-label data) |
| `hamming_loss` | Fraction of misclassified (document, label) pairs |
| `hamming_accuracy` | `1 - hamming_loss` (inflated on sparse data; reported but not as "accuracy") |
| `per_label` | Dict per label: `{precision, recall, f1, support}` |

#### When metrics are withheld

Metrics are **not fabricated**.  If the data is too thin to support them, the
metric set marks `is_meaningful=False` and the relevant fields stay `None` with
a reason:

- `reason_empty` — no valid labels after flattening, or empty input.
- `reason_insufficient_classes` — fewer than 2 classes present (single-label),
  or fewer than 2 labels (multi-label).
- `reason_invalid_shape` — arrays are not 2-D when multi-label was expected.
- `reason_label_mismatch` — label count doesn't match array columns.

The `is_meaningful` property returns `False` when:

- `reason_empty` is set, or
- `reason_insufficient_classes` is set, or
- `n_samples` is less than 5.

Accuracy can still be computed mathematically on a single-class dataset, but
the framework's policy is to mark the result as **not meaningful** when there is
no class diversity, because macro-F1 and per-class comparisons need at least two
classes to be interpretable.  Tests reflect this policy.

### Unified EvaluationReport

`ml.evaluation.reports.EvaluationReport` is the shape every part of the
framework sees, regardless of task:

| Field | Meaning |
|---|---|
| `task_name` | Task (e.g. `"sentiment"`) |
| `model_name` | Model identifier (e.g. `"TfidfLogisticSentimentClassifier"`) |
| `model_version` | Model version string |
| `n_train`, `n_validation`, `n_test` | Split sizes |
| `labels` | Label list the model was trained on |
| `metrics` | `ClassificationMetricSet` |
| `is_meaningful` | Whether the evaluation is trustworthy |
| `data_split_metadata` | Split info including `leakage_detected` |
| `warnings` | All warnings from the model's own evaluation plus the framework |
| `timestamp` | Optional wall-clock timestamp (off by default) |

`build_evaluation_report(...)` translates a task-specific raw report (the output
of the model's `evaluate()` method) into this unified shape.  It inspects the
raw report for known attributes (`accuracy`, `macro_f1`, `per_label`, etc.) and
pulls out what it can.  When the raw report is `None` or has no recognised
attributes, the metric set is empty with an appropriate reason.

### Task-specific evaluation modules (unchanged)

The existing NLP modules already have detailed evaluation logic.  The ML
framework does **not** replace them — it calls the model's `evaluate()` and
translates the result.  The honest-metric conventions (don't drop abstentions,
don't drop unscoreable gold, report support counts, warn on thin classes) are
implemented in the NLP modules and preserved through the translation.

---

## 9. Reproducibility

### What is deterministic

- **Splitting** — `random_state=42` by default; same inputs → same indices.
- **Synthetic data** — `make_synthetic_records` uses a seeded `random.Random`.
- **Artifact naming** — deterministic from `task__model__version`, no timestamp.
- **Registry entry IDs** — SHA-256 content hash of (task, model_name,
  model_version, dataset, configuration), so the same run produces the same ID.

### What is NOT part of deterministic identity

- **Timestamps** — off by default.  `TaskConfig.include_timestamp` must be set
  explicitly to add a wall-clock timestamp to reports and artifacts.  The
  framework never uses the current time as part of a run's identity.
- **Random processes outside the framework's control** — the existing
  classifiers may use internal randomness (e.g. logistic regression solver); the
  framework passes through whatever configuration the model has and does not
  add its own.

### How to reproduce a run

```python
from ml.configs import TaskConfig, TaskType
from ml.training.trainer import Trainer

config = TaskConfig(
    task_name="sentiment",
    dataset_path="data/annotations/sentiment_annotations.csv",
    random_state=42,
    train_size=0.7,
    validation_size=0.15,
    test_size=0.15,
)
trainer = Trainer(config)
result = trainer.run(use_synthetic=False)
```

Given the same annotation file and the same config, this produces the same split,
the same artifact name, and the same registry entry ID every time.

---

## 10. Model artifacts

### What gets saved

`Trainer.save_artifact()` writes a joblib file containing a dict with two keys:

- `"model"` — the trained classifier instance (the actual model object).
- `"metadata"` — a dict with everything needed to identify and reproduce the run:

| Metadata field | Meaning |
|---|---|
| `task` | Task name |
| `model_name` | Model identifier |
| `model_version` | Model version |
| `artifact_path` | Path to this artifact file |
| `dataset` | Path to the annotation file used for training |
| `source` | `"real"` or `"synthetic"` |
| `n_training_records` | Training split size |
| `n_validation_records` | Validation split size |
| `n_test_records` | Test split size |
| `labels` | Label list the model was trained on |
| `configuration` | Full `TaskConfig.to_dict()` |
| `split` | Split metadata: `random_state`, `stratify_used`, `stratify_available`, sizes |
| `trained_at` | Optional timestamp (only when `include_timestamp=True`) |

### Deterministic artifact naming

```
{sanitised_task}__{sanitised_model_name}__{model_version}.joblib
```

Examples:

```
sentiment__TfidfLogisticSentimentClassifier__v1.joblib
issue__TfidfLogisticIssueClassifier__v1.joblib
solution__TfidfLogisticSolutionClassifier__v1.joblib
```

No timestamp is embedded in the filename. Two runs with the same task, model,
and version produce the same filename. The default saver therefore **can
overwrite an existing artifact**; it is not a run-specific immutable output
location. Before any approved training run, the caller must supply a new,
run-specific `artifact_directory` (and a separate registry path) and verify
that the destination is empty. This audit did not write an artifact.

### What is NOT stored

- Raw personal/user information (Reddit usernames, etc.) — the framework only
  stores the `document_id`, `text`, label, and language-analysis metadata that
  the annotation carries.  No Reddit-specific personal metadata is added.
- Unnecessary Reddit metadata — the `TrainingRecord` carries only the fields
  defined above.  The framework does not pull in extra fields from the Reddit
  ingestion models.

### Loading an artifact

```python
import joblib
artifact = joblib.load("ml/artifacts/sentiment__TfidfLogisticSentimentClassifier__v1.joblib")
model = artifact["model"]
metadata = artifact["metadata"]
```

---

## 11. Model registry

### What it is

`ml.configs.ModelRegistry` is a lightweight local JSON registry.  It records
every trained model's metadata in a single JSON file (default:
`ml/artifacts/registry.json`).  No database — just a file that any process can
read.

### What it records per entry

| Field | Meaning |
|---|---|
| `entry_id` | Content-derived SHA-256 hash (16 hex chars) |
| `registered_at_run` | Run signature (content-based, not wall-clock) |
| `task` | Task name |
| `model_name` | Model identifier |
| `model_version` | Model version |
| `artifact_path` | Path to the joblib artifact |
| `dataset` | Path to the training annotation file |
| `configuration` | `TaskConfig.to_dict()` |
| `metrics` | Evaluation metrics (validation + test) |
| `warnings` | All pipeline warnings |

### Operations

- `register(entry)` — append a new entry; returns the content-derived ID.
  Duplicate content produces the same ID (append-only behaviour).
- `list_entries(task=None)` — list all entries, optionally filtered by task.
- `find_latest(task, model_name)` — return the most recently registered entry
  for a task + model name pair.

### What it is NOT

- Not a PostgreSQL registry.  No database integration in this stage.
- Not a model server.  It records metadata; it does not serve models.
- Not versioned by timestamp.  Entry IDs are content-derived, and
  `registered_at_run` is a content-based signature, not a wall-clock time.

---

## 12. Current classical ML baselines

The framework trains the existing classical models through their public APIs.
The models themselves are defined and implemented in the NLP modules and are
**not** duplicated in the ML framework.

| Task | Model | Type | Module |
|---|---|---|---|
| sentiment | `TfidfLogisticSentimentClassifier` | TF-IDF + multiclass Logistic Regression | `app.services.nlp.sentiment_emotion.tfidf_logistic_sentiment` |
| emotion | `TfidfLogisticEmotionClassifier` | TF-IDF + One-vs-Rest Logistic Regression | `app.services.nlp.sentiment_emotion.tfidf_logistic_emotion` |
| issue | `TfidfLogisticIssueClassifier` | TF-IDF + One-vs-Rest Logistic Regression | `app.services.nlp.issue_identification.tfidf_logistic_classifier` |
| response | `TfidfLogisticResponseClassifier` | TF-IDF + multiclass Logistic Regression | `app.services.nlp.response_classification.tfidf_logistic_multiclass` |
| solution | `TfidfLogisticSolutionClassifier` | TF-IDF + binary Logistic Regression | `app.services.nlp.solution_detection.tfidf_logistic_binary` |

### Model mathematics (as implemented, not invented)

#### TF-IDF

The feature representation is term-frequency × inverse-document-frequency:

```
TFIDF(t, d) = TF(t, d) × IDF(t)
```

where `TF(t, d)` is the term frequency of term `t` in document `d`, and
`IDF(t)` is the inverse document frequency of term `t` across the training
corpus.  The existing classifiers use scikit-learn's `TfidfVectorizer` with
their own configuration (n-gram range, min_df, max_df, etc.) — the ML
framework does not redefine or replace this.

#### Multiclass Logistic Regression (sentiment, response)

For K classes, the model learns a weight vector `w_k` and bias `b_k` per class.
The logit for class `k` given feature vector `x` is:

```
z_k = w_k^T x + b_k
```

The predicted probability is the softmax over all classes:

```
P(y = k | x) = exp(z_k) / Σ_j exp(z_j)
```

The classifier predicts the class with the highest probability (subject to a
confidence threshold that can cause abstention).

#### Multi-label One-vs-Rest Logistic Regression (emotion, issue)

Each label `l` gets its own binary classifier.  For label `l`:

```
z_l = w_l^T x + b_l
P(y_l = 1 | x) = σ(z_l) = 1 / (1 + exp(-z_l))
```

A document receives label `l` when `P(y_l = 1 | x)` exceeds the label's
threshold.  Multiple labels can be active for the same document.  This is the
standard One-vs-Rest reduction; it is not a joint multi-label model.

#### Binary Logistic Regression (solution)

Same as the multi-label case but with a single binary classifier:

```
z = w^T x + b
P(is_solution = 1 | x) = σ(z)
```

The document is predicted as a solution when the probability exceeds the
threshold.

### Class weighting

Where the existing classifiers use `class_weight="balanced"` (or similar), the
framework passes through that configuration and reports the resulting class
distribution.  The framework does not add or remove class weighting.

---

## 13. Limitations

### Stage limitations

- **No supervised task annotation data yet.** The repository contains a
  `data/annotations/language_annotations.csv` file, but it has language/script
  review columns rather than labels for sentiment, emotion, issue, response, or
  solution detection. The five supervised annotation CSVs are absent. Until
  reviewed, traceable task annotations are supplied, training is blocked and
  any synthetic metrics remain plumbing-check numbers, not project performance.
- **No ANN or CNN.**  The framework is designed to accept models that implement
  the `TrainableModel` protocol (`train`, `predict`, `evaluate`), but no neural
  models are implemented yet.
- **No cross-validation.**  The framework does a single deterministic
  train/validation/test split.  Cross-validation would be a separate concern.
- **No hyper-parameter search.**  The framework uses the classifiers' default
  configuration.  Hyper-parameter tuning would be a separate concern.
- **No online / incremental training.**  The classifiers are trained from
  scratch on the full training split each time.
- **No model serving.**  The registry records metadata; it does not serve
  predictions.

### Baseline limitations

- TF-IDF + logistic regression is a strong classical baseline but has known
  limitations: it cannot capture word order beyond n-grams, it treats each term
  independently, and it has no notion of semantic similarity between words.
- Multi-label One-vs-Rest reduces the problem to independent binary classifiers,
  which ignores label co-occurrence.
- The baseline is monolingual in the sense that TF-IDF features are term-based,
  not script-aware beyond what the preprocessing provides.

These limitations are precisely why ANN and CNN are planned as future phases:
they can capture sequence structure (RNN/CNN) and semantic relationships
(embedding-based models) that TF-IDF cannot.

---

## 14. Transition to ANN

When the project is ready to experiment with artificial neural networks, the
plan is:

1. **Define an ANN model class** that implements the same `TrainableModel`
   protocol the trainer already expects:
   - `train(annotations)` — build TF-IDF (or embedding) features from the
     annotation texts, train a dense network, store the trained model.
   - `predict(documents)` — featurize and forward-pass.
   - `evaluate(annotations)` — predict + compute metrics.
2. **Register it in the model factory** (or pass it directly to the trainer) so
   the same `Trainer.run()` pipeline works without modification.
3. **Compare against the classical baseline** using the same evaluation reports,
   the same split, and the same metrics — so the comparison is apples-to-apples.

A future ANN architecture might look like:

```
TF-IDF / embedding → Dense → ReLU → Dense → output
```

The evaluation framework does not need to be rewritten for this.  The metrics
(`ClassificationMetricSet`) already handle both single-label and multi-label
output, and the `EvaluationReport` is task-agnostic.

The ML framework does **not** implement any ANN in this stage.

---

## 15. Transition to CNN

After ANN experiments, a CNN-based model could be tried for sequence
classification:

```
Embedding → Conv1D → ReLU → Pooling → Dense → output
```

The same training/evaluation pipeline would apply: featurize the text (this
time with a trainable embedding layer rather than TF-IDF), train the network,
evaluate with the same metric suite.

The ML framework does **not** implement any CNN in this stage.

---

## 16. Future transformer comparison

For multilingual data (the project handles English, Hindi, Bengali, Odia, and
code-mixed text), a transformer-based model (e.g. a multilingual encoder) would
be a natural later comparison point.  The same pipeline pattern would apply:

1. Implement a model class with `train` / `predict` / `evaluate`.
2. Plug it into the existing trainer.
3. Compare against the classical baseline and any ANN/CNN experiments using the
   same splits, metrics, and evaluation reports.

This is future work and is **not** part of the current classical ML baseline
stage.

---

## 17. Running the demo

```bash
D:\AI_Project\backend\.venv\Scripts\python.exe demo_ml_training.py
```

The demo uses **synthetic data only** and prints `ILLUSTRATIVE / PLUMBING
CHECK - NOT PROJECT PERFORMANCE` at the top.  It exercises all five tasks
(sentiment, emotion, issue, response, solution) through the full pipeline:
dataset creation, deterministic splitting, training, validation evaluation, test
evaluation, class distribution, leakage check, artifact creation, and registry
entry.

---

## 18. Running the tests

```bash
# ML-specific tests:
D:\AI_Project\backend\.venv\Scripts\python.exe -m pytest test_ml_splitting.py test_ml_training.py test_ml_evaluation.py -q

# Full regression suite (includes the ML tests):
D:\AI_Project\backend\.venv\Scripts\python.exe -m pytest -q
```

All tests use synthetic data.  No real annotation files are created or modified
by the tests.  Temporary synthetic fixtures are written to `tempfile.mkdtemp()`
and cleaned up in `tearDown`.

---

## 19. Environment notes

### Python interpreter

All verification must use the project's virtual environment:

```
D:\AI_Project\backend\.venv\Scripts\python.exe
```

Do not use the system Python to run or test the ML framework.

### Dependency discrepancies (observed, not modified)

The following discrepancies between `requirements.txt` and the actual installed
environment were observed during this stage.  They are **not** modified as part
of this task:

| Package | requirements.txt | installed |
|---|---|---|
| psycopg2-binary | listed | **not installed** — the project uses `psycopg` (v3) with `postgresql+psycopg://` |
| httpx | listed | **not installed** |
| joblib | **not listed** | installed (used by the ML framework for artifact serialization) |
| fastapi | `0.133.1` | `0.141.1` |
| pydantic | `2.13.4` | `2.13.5` |
| numpy | `2.4.3` | `2.5.3` |
| scikit-learn | `1.9.0` | `1.9.1` |
| pytest | `8.3.3` | `8.3.3` (matches) |

The database driver is `psycopg` (v3), not `psycopg2`.  The `postgresql+psycopg://`
dialect in `backend/app/core/database.py` confirms this.  No database code was
changed in this stage.

### No fabricated annotation files

No files were created under `data/annotations/`.  The framework raises
`AnnotationNotFoundError` when annotation files are absent, and the trainer's
synthetic fallback is explicitly labelled as synthetic in all outputs.

---

## 20. Model artifact readiness audit (2026-10-16)

This section records the first readiness gate for real supervised models. It is
an audit result, not a training result.

### Annotation inventory and provenance

The repository contains exactly one annotation CSV under `data/annotations/`:

| File | Rows | Columns | Use in the five supervised tasks |
|---|---:|---|---|
| `language_annotations.csv` | 15 | `text`, `manual_language`, `manual_script`, `manual_is_code_mixed`, `notes` | Not a sentiment, emotion, issue, response, or solution target file |

The file has no document ID column and no supervised target column. Its presence
does not establish an approved source, reviewer identity, adjudication history,
inter-annotator agreement, or permission to train. Its observed SHA-256 during
this audit was `f3ffde6e43dcef0e5f62a46703bb52bf50743fb8f144e77621e3e413be1d6a35`.
Raw text was not copied into the audit report.

The five configured task paths were checked with the existing loader and all
returned `AnnotationNotFoundError`:

| Task | Expected file | Readiness status |
|---|---|---|
| sentiment | `data/annotations/sentiment_annotations.csv` | **Blocked pending reviewed annotations** |
| emotion | `data/annotations/emotion_annotations.csv` | **Blocked pending reviewed annotations** |
| issue | `data/annotations/issue_annotations.csv` | **Blocked pending reviewed annotations** |
| response | `data/annotations/response_annotations.csv` | **Blocked pending reviewed annotations** |
| solution | `data/annotations/solution_annotations.csv` | **Blocked pending reviewed annotations** |

The annotation schemas were inspected and retained as the required contracts:

- sentiment: one annotator-defined, nonblank class;
- emotion: annotator-defined multi-label values, including a valid empty set;
- issue: annotator-defined multi-label values, including no issue labels;
- response: exactly one of the canonical ordered labels `agreement`,
  `personal_experience`, `opinion`, `advice`, `solution`,
  `counterargument`;
- solution: the boolean `is_solution` target; `solution_type` is free-form
  metadata and is not learned as a target.

No task can be called genuinely human-reviewed or ready for training from the
files currently present. Synthetic fixtures in tests and demos are explicitly
marked synthetic and were not used as evidence of model quality.

### Existing artifacts and registry

`ml/artifacts/` contains `registry.json` and `.gitkeep`, but no project-local
`.joblib`, `.pkl`, `.pickle`, `.safetensors`, `.pt`, `.pth`, or `.onnx` files.
The inspected `backend/ml/` tree also contains no matching model binaries.
No model artifact was deserialized or inference-loaded during this audit.

The registry contains 23 entries: 12 sentiment, 1 issue, and 10 response.
Every entry carries a synthetic-data warning, and every recorded dataset path
was absent from the repository. Four referenced joblib files still existed in
external temporary directories; they were not loaded, are not repository
artifacts, and have no independently verified annotation provenance or review
approval. Registry metadata alone is therefore not evidence of a genuine or
approved model. There are no registry entries establishing emotion or solution
artifacts.

No new artifact, registry entry, annotation file, or evaluation report was
created by this audit.

### Split and evaluation protocol review

The framework provides deterministic `random_state=42` splitting, optional
whole-`post_id` group splitting, duplicate document-ID and exact-text leakage
checks, separate validation/test reports, per-class or per-label metrics, macro
F1, and confusion matrices where the task supports them. The readiness gate
uses 60 total records and 6 records per present class as conservative warnings;
MuRIL has stricter 300-total/30-per-class thresholds. These checks do not prove
annotation genuineness.

The audit reproduced a split defect: three groups of five records could produce
15 training records and empty validation/test folds because whole groups did not
fit the requested proportions. `split_records()` now moves complete groups to
populate requested held-out folds, or raises before training when too few groups
or configured minimums make that impossible. Added regression coverage verifies
the three-group case and the two-group failure case. The corrected reproduction
produced 5 train, 5 validation, and 5 test records with no group overlap.

The framework still does not detect semantic near-duplicates, and duplicate
source records are reported rather than automatically removed. A future approved
run must define the applicable grouping/near-duplicate policy before fitting,
freeze the split before any vectorizer or classifier fit, retain validation for
selection only, and use the test set once for final held-out reporting. The
trainer does not centrally enforce the readiness report before training, so that
gate remains an explicit pre-training approval requirement.

### Verification performed

Only offline, non-training readiness checks were run. The following selections
were launched in separate subprocesses with the project virtual environment;
all exited 0:

| Selection | Result |
|---|---:|
| `backend.test_ml_splitting` | 18 passed |
| `backend.test_ml_evaluation` | 14 passed |
| `backend.test_readiness` | 18 passed |
| Selected dataset/config/annotation adapters from `backend.test_ml_training` | 24 passed |
| `backend.test_annotation_export` | 11 passed |
| `backend.test_convert_annotation_candidates` | 10 passed |
| Selected issue/sentiment/emotion/response/solution annotation-schema classes | 147 passed |
| `backend.test_nlp_pipeline` | 40 passed |
| **Total selected offline executions** | **282 passed** |

The wrapper reported **282 passed executions**, matching the table. No classifier
training, topic fitting, linker fitting, neural execution, model loading,
database access, migration, collection, or live request occurred in this pass.
The direct split/readiness portion of that selection passed **36 tests** after
the group-split correction; those 36 are included in the 282 total above.

Static checks for this milestone must be rerun after any further documentation
or code change. Successful schema and framework tests do not establish model
quality or artifact approval.

### Readiness decision and next gate

All five supervised tasks are **blocked pending reviewed annotations**. Supply
task-specific CSVs with unique traceable document IDs, source/provenance records,
reviewer/adjudication status, current label-schema compliance, and an approved
grouping policy. Then run readiness preflight, review class counts and exclusions,
freeze the split, train only in a new run-specific artifact directory, evaluate
on an untouched held-out test set, load-test through the real inference
interface, and obtain explicit review before any artifact is considered approved.
