# Reddit Community Intelligence (RCI) — Post-Redesign ML Integrity Audit & Release Gate Report

**Project:** `D:\AI_Project`  
**Audit Date:** 2026-11-19  
**Frozen V1 Artifacts (Preserved Unmodified):** `ml/artifacts/demo_synthetic/`  
**Corrected V2 Isolated Artifacts:** `ml/artifacts/demo_synthetic_v2/`  
**Dataset Provenance:** `data/demo_synthetic/` (Seed `20261116`, 160 fictional posts, 1,280 comments, 8 comments/thread, `synthetic_only: true`)  

> [!IMPORTANT]
> **Scientific Scope Guard:** Every model, metric, and prediction audited in this document was trained and evaluated on a **reproducibly generated synthetic corpus**. Near-100% held-out scores on Sentiment, Response, Solution, and Emotion reflect deterministic generator templates and **must never be cited as evidence of real-world Reddit NLP performance**.

---

## 1. Forensic Investigation of the Solution Model Label Mismatch

### 1.1 Confirmed Root Cause (End-to-End 9-Stage Pipeline Trace)

Tracing the binary **Solution** detection task across all nine pipeline stages confirms that **a real label-vocabulary mismatch existed between the generic dataset loader/post-processor (`"True"` / `"False"` / `"true"` / `"false"`) and the classifier's canonical target vocabulary (`"not_solution"` / `"solution"`).**

Crucially, the mismatch occurred in **two distinct places** while the classifier's internal training and internal evaluator operated on Python booleans/indices—explaining why top-level aggregate metrics (`accuracy: 1.0`, `macro_f1: 1.0`) were valid while `per_class` supports and `confusion_matrix` collapsed to `0`:

| Stage | Module & Function | Exact Representation Before Fix | Effect on Pipeline |
| --- | --- | --- | --- |
| **1. Synthetic Data Generation** | `backend/generate_synthetic_demo.py` (`build` `#L172`, `write_csv` `#L238`) | Sets `comment["is_solution"] = role in {"advice", "solution"}` (`bool`). Serializes `is_solution` column in `data/demo_synthetic/solution.csv` as lowercase `"true"` (426 rows) or `"false"` (854 rows). | Valid boolean CSV serialization. |
| **2. Label Extraction** | `ml/data/dataset.py` (`load_annotation_dataset` `#L302-L307`, `_solution_to_record` `#L375-L388`) | `SolutionCorpus.from_csv()` parses `"true"`/`"false"` via `parse_bool_cell()` into `a.is_solution: bool`. However, `_solution_to_record(a)` set `TrainingRecord.label = str(a.is_solution)` → **`"True"` / `"False"`** instead of calling `target_label(a.is_solution)` (`"solution"` / `"not_solution"`). | Contaminated `LoadedDataset.labels` (`["False", "True"]`) and `label_counts` (`{"False": 854, "True": 426}`), diverging from the model's `TARGET_LABELS = ("not_solution", "solution")`. |
| **3. Training Target Construction** | `ml/training/trainer.py` (`records_to_annotations` `#L357-L378`) | Checked `r.label.strip().lower() in {"true", "1", "yes"}` to build `SolutionAnnotation(is_solution=...)`. Inside `TfidfLogisticSolutionClassifier.train()` (`tfidf_logistic_binary.py#L253`), `target_index(bool(r.is_solution))` mapped `False → 0` (`not_solution`) and `True → 1` (`solution`). | Worked when `r.label` was `"True"`/`"False"`, but would silently misclassify canonical `"solution"` as `False` (`"solution" not in {"true", "1", "yes"}`) and silently coerce malformed strings to `False`. |
| **4. Label Normalization** | `app/services/nlp/solution_detection/models/labels.py` (`#L44-L134`) | Defines canonical `TARGET_LABELS = ("not_solution", "solution")`, `target_label(bool)`, `label_to_bool(str)`, and `parse_bool_cell(Any)`. | Canonical normalizers existed in `labels.py` but were **not called** by `ml/data/dataset.py` or `backend/train_demo_baselines.py`. |
| **5. Model Prediction** | `tfidf_logistic_binary.py` (`predict` `#L364-L513`) | Emits `SolutionPrediction(is_solution: bool, solution_probability: float, not_solution_probability: float)`. When serialized via `jsonable_encoder` into `predictions.json`, `prediction["is_solution"]` is a JSON boolean (`true` / `false`). | Predictions use boolean `is_solution`, whereas `trained_model.labels` is `["not_solution", "solution"]`. |
| **6. Evaluation Target Construction (Internal vs. Wrapper)** | `tfidf_logistic_binary.py::evaluate()` (`#L519-L563`) vs. `ml/evaluation/reports.py::_extract_metrics()` (`#L134-L204`) | Inside `trained_model.evaluate()`, `evaluate_solution_detection()` compared boolean `gold` against boolean `predicted` over `TARGET_LABELS = ["not_solution", "solution"]`, producing `accuracy = 1.0`, `macro_f1 = 1.0`, `per_class_support = {"not_solution": 127, "solution": 65}`, and `confusion_matrix = [[127, 0], [0, 65]]`. However, `_extract_metrics()` only copied scalar `accuracy`, `macro_f1`, `weighted_f1`, `precision`, and `recall`, **discarding** `raw_report.per_class_*` and `raw_report.confusion_matrix`. | Forced `backend/train_demo_baselines.py` to re-compute `per_class` and `confusion_matrix` externally via `heldout_class_metrics()`. |
| **7. Class Ordering & Confusion-Matrix Construction** | `backend/train_demo_baselines.py` (`heldout_class_metrics` `#L48-L89`) | `heldout_class_metrics()` received `labels = trainer.state.trained_model.labels` (`["not_solution", "solution"]`), set `gold = label_set(record.label)` (`{"true"}` or `{"false"}`), and set `pred_values = {"true" if pred is True else "false"}`. It then checked `"not_solution" in gold` and `"solution" in gold` (lines 72–75) and `true_label in {"not_solution": 0, "solution": 1}` (line 87). | **Every membership check evaluated to `False`!** For all 192 test samples, `tp = fp = fn = support = 0`, producing `per_class` supports of `0` and `confusion_matrix = [[0, 0], [0, 0]]`. |
| **8. API Model-Report Serialization** | `backend/app/api/demo.py` (`/api/demo/models` `#L405-L411`) | Served `evaluation.json["tasks"]` verbatim. | Propagated the contradictory Solution report (`accuracy: 1.0, macro_f1: 1.0` alongside `support: 0` and `[[0, 0], [0, 0]]`) and `null` multi-label `subset_accuracy`/`hamming_loss` to the frontend. |
| **9. Frontend Metric Presentation** | `backend/app/static/js/views/models.js` (`#L22-L75`, `#L165-L185`) | Detected `totalSupport === 0 && nSamples > 0` on the frozen v1 artifact and displayed an honest diagnostic callout rather than a broken all-zero table; displayed `"Not emitted"` for `null` `subset_accuracy`. | With v2 artifacts loaded, renders the verified `not_solution` (127) / `solution` (65) per-class table, `[[127, 0], [0, 65]]` confusion heatmap, and multi-label `subset_accuracy` + `hamming_loss`. |

### 1.2 Why the Held-Out Synthetic Test Score for Solution Is Genuinely 100% (`[[127, 0], [0, 65]]`)

We verified independently from the raw test split (`192` comments across `24` held-out threads) whether the `1.0` accuracy / macro F1 on the synthetic dataset was an artifact or a genuine property of the synthetic templates:
- In `backend/generate_synthetic_demo.py#L172`, `is_solution` is defined deterministically by `role in {"advice", "solution"}`.
- All `"advice"` and `"solution"` comments in the generator are constructed from 8 distinct deontic/action templates (`"You could write to..."`, `"Try keeping the..."`, `"Aap ... screenshot rakh lo..."`, `"आप ... लिखित पुष्टि माँगें।"`, `"The simplest fix is..."`, `"For ..., a shared checklist..."`, `"Iska practical fix:..."`, `"समाधान यह है कि..."`).
- None of those 8 template stems ever appear in `"agreement"`, `"personal_experience"`, `"opinion"`, or `"counterargument"` templates.
- Because the 8 template stems recur across all 160 threads, a TF-IDF `(1, 2)`-gram logistic regression classifier trained on 112 threads (`896` comments) achieves complete linear separation on the 24 held-out test threads (`127` `not_solution` and `65` `solution` comments, `TP = 65, TN = 127, FP = 0, FN = 0`).
- **Scientific caveat:** This 100% separability is strictly a property of the fixed 24-template synthetic generator (`ROLE_TEXT`) and does not generalize to organic community discussions where advice, opinion, and personal experience share overlapping phrasing.

---

## 2. Multi-Label Evaluation Audit & Root Cause of the Emotion "0% Subset Accuracy" Anomaly

### 2.1 Which Tasks Are Multi-Label vs. Single-Label

| Task | Paradigm | Why | Cardinality in Synthetic Corpus | Valid Metric Family | Invalid / Prohibited Metrics |
| --- | --- | --- | --- | --- | --- |
| **Sentiment** | Single-label multiclass ($K=3$) | Each comment has one mutually exclusive polarity (`negative`, `neutral`, `positive`). | Exactly 1 label/comment | `accuracy`, `macro_f1`, `micro_f1`, `precision`, `recall`, `per_class`, $3\times 3$ `confusion_matrix` | `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `per_label` (must be `null` / `{}`) |
| **Emotion** | **Multi-label** ($L=6$) | A comment expresses multiple co-occurring emotions (`hope`+`gratitude`, `confidence`+`gratitude`, `stress`+`frustration`, or `curiosity`). | 1 to 2 labels/comment (`2,265` tags across `1,280` comments; `1.77` labels/comment) | `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `macro_f1`, `micro_f1`, `precision`, `recall`, `per_label` | Single-label `accuracy`, single-label `confusion_matrix`, `per_class` (must be `null` / `{}`) |
| **Issue** | **Multi-label** ($L=8$) | `TfidfLogisticIssueClassifier` fits 8 independent One-vs-Rest binary relevance heads (`threshold=0.5`), allowing 0, 1, or multiple issues per comment. | 1 gold label/comment in synthetic pilot; 0 or 1 predicted labels/comment on test | `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `macro_f1`, `micro_f1`, `precision`, `recall`, `per_label` | Single-label `accuracy`, single-label `confusion_matrix`, `per_class` (must be `null` / `{}`) |
| **Response** | Single-label multiclass ($K=6$) | Each comment is assigned one primary discourse role from the 6 fixed canonical response classes. | Exactly 1 label/comment | `accuracy`, `macro_f1`, `micro_f1`, `weighted_f1`, `precision`, `recall`, `per_class`, $6\times 6$ `confusion_matrix` | `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `per_label` (must be `null` / `{}`) |
| **Solution** | Single-label binary ($K=2$) | Binary decision (`not_solution` vs. `solution`) via logistic sigmoid threshold (`0.5`). | Exactly 1 label/comment | `accuracy`, `macro_f1`, `weighted_f1`, positive-class `precision` & `recall`, `per_class`, $2\times 2$ `confusion_matrix` | `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `per_label` (must be `null` / `{}`) |

### 2.2 Why Emotion Previously Displayed "0.0% Subset Accuracy" Alongside 98.6% Macro F1

1. **Task Evaluator Computed the True Value:** `evaluate_emotions()` in `backend/app/services/nlp/sentiment_emotion/models/evaluation.py#L561-L565` properly computed:
   - Held-out Validation (`n = 192`): `subset_accuracy = 191 / 192 = 0.994792` (`99.48%`), `hamming_loss = 1 / 1152 = 0.000868` (`0.09%`), `macro_f1 = 0.995726` (`99.57%`), `micro_f1 = 0.998536` (`99.85%`).
   - Held-out Test (`n = 192`): Only 3 test comments had a false-positive `confidence` prediction (`precision = 16/19 = 0.842105, recall = 16/16 = 1.0`), while all other 5 emotion labels had 0 errors. Thus `subset_accuracy = 189 / 192 = 0.984375` (`98.44%`), `hamming_loss = 3 / (192 × 6) = 3 / 1152 = 0.002604` (`0.26%`), `macro_f1 = 0.985714` (`98.57%`), and `micro_f1 = 0.995582` (`99.56%`).
2. **Framework Report Adapter Dropped the Fields:** `ml/evaluation/reports.py::_extract_metrics()` (`#L173-L186`) only extracted `macro_f1` and `micro_f1` for `is_multilabel=True` and never copied `raw_report.subset_accuracy`, `raw_report.hamming_loss`, `raw_report.hamming_accuracy`, `raw_report.macro_precision`, `raw_report.macro_recall`, or `raw_report.per_label_*`. As a result, `evaluation.json` serialized `"subset_accuracy": null` and `"hamming_loss": null`.
3. **Pre-Redesign UI Coerced `null` to `0.0%`:** The pre-redesign UI passed `metrics.accuracy` (`null`) into `Number(null)` (which evaluates to `0` in JavaScript) under the label `"Test subset accuracy"`, fabricating a `0.0%` display.

---

## 3. Before-and-After Evaluation Results (`demo_synthetic` v1 vs. `demo_synthetic_v2`)

All splits use `SEED = 20261116`, `group_by = "post_id"` (`896` train comments [112 threads] / `192` validation comments [24 threads] / `192` test comments [24 threads]), with **zero cross-split thread or text leakage**.

### 3.1 Held-Out Test Split Summary (`n = 192` comments, `24` threads)

| Task | Metric | Frozen V1 (`ml/artifacts/demo_synthetic/evaluation.json`) | Corrected V2 (`ml/artifacts/demo_synthetic_v2/evaluation.json`) | Change & Verification Note |
| --- | --- | --- | --- | --- |
| **Sentiment** (`single-label`, $K=3$) | `accuracy` / `macro_f1` / `micro_f1` | `1.0` / `1.0` / `1.0` | `1.0` / `1.0` / `1.0` | Unchanged; `precision: 1.0`, `recall: 1.0` now also populated from evaluator. |
| | `per_class` supports | `negative: 51, neutral: 46, positive: 95` (sum = 192) | `negative: 51, neutral: 46, positive: 95` (sum = 192) | Verified exact match; validation `per_class` and `confusion_matrix` now also populated. |
| | `confusion_matrix` | `[[51,0,0],[0,46,0],[0,0,95]]` (sum = 192) | `[[51,0,0],[0,46,0],[0,0,95]]` (sum = 192) | Verified exact match. |
| **Emotion** (`multi-label`, $L=6$) | `macro_f1` / `micro_f1` | `0.985714` / `0.995582` | `0.985714` / `0.995582` | Unchanged (`98.6%` macro F1, `99.6%` micro F1). |
| | `subset_accuracy` | `null` (omitted by `_extract_metrics`) | **`0.984375`** (`189 / 192`, `98.4%`) | **Fixed:** Extracted from `EmotionEvaluationReport.subset_accuracy` (Val: `0.994792`). |
| | `hamming_loss` / `hamming_accuracy` | `null` / `null` | **`0.002604`** (`0.26%`) / **`0.997396`** (`99.74%`) | **Fixed:** `3` false-positive `confidence` tags out of `1,152` $(192 \times 6)$ decisions. |
| | `precision` / `recall` (macro) | `null` / `null` | **`0.973684`** / **`1.0`** | **Fixed:** Extracted from `macro_precision` and `macro_recall`. |
| | `accuracy` / `confusion_matrix` | `null` / `null` | `null` / `null` | Preserved `null` (single-label metrics forbidden on multi-label task). |
| **Issue** (`multi-label`, $L=8$) | `macro_f1` / `micro_f1` | `0.585376` / `0.577778` | `0.585376` / `0.577778` | Unchanged (`58.5%` macro F1, `57.8%` micro F1). |
| | `subset_accuracy` | `null` (omitted by `_extract_metrics`) | **`0.40625`** (`78 / 192`, `40.6%`) | **Fixed:** Exact set match on 78 of 192 test comments (Val: `0.390625`). |
| | `hamming_loss` / `hamming_accuracy` | `null` / `null` | **`0.074219`** (`7.42%`) / **`0.925781`** (`92.58%`) | **Fixed:** `114` false negatives and `0` false positives across `1,536` $(192 \times 8)$ decisions. |
| | `precision` / `recall` (macro) | `null` / `null` | **`1.0`** / **`0.453906`** | **Fixed:** Extracted from `macro_precision` and `macro_recall`. |
| | `accuracy` / `confusion_matrix` | `null` / `null` | `null` / `null` | Preserved `null` (single-label metrics forbidden on multi-label task). |
| **Response** (`single-label`, $K=6$) | `accuracy` / `macro_f1` / `micro_f1` / `weighted_f1` | `1.0` / `1.0` / `1.0` / `1.0` | `1.0` / `1.0` / `1.0` / `1.0` | Unchanged; canonical 6-class order locked. |
| | `per_class` supports | `31, 33, 32, 36, 29, 31` (sum = 192) | `31, 33, 32, 36, 29, 31` (sum = 192) | Verified in exact canonical order `agreement` → `counterargument`. |
| | `confusion_matrix` | `diag(31, 33, 32, 36, 29, 31)` (sum = 192) | `diag(31, 33, 32, 36, 29, 31)` (sum = 192) | Verified exact match. |
| **Solution** (`binary`, $K=2$) | `label_counts` (full corpus) | `{"False": 854, "True": 426}` | **`{"not_solution": 854, "solution": 426}`** | **Fixed:** Normalized to canonical `TARGET_LABELS`. |
| | `accuracy` / `macro_f1` / `weighted_f1` | `1.0` / `1.0` / `1.0` | `1.0` / `1.0` / `1.0` | Verified consistent with confusion matrix diagonal. |
| | `per_class["not_solution"]` | `precision: 0.0, recall: 0.0, f1: 0.0, support: 0` | **`precision: 1.0, recall: 1.0, f1: 1.0, support: 127`** | **Fixed:** Normalized gold & predicted labels match canonical `"not_solution"`. |
| | `per_class["solution"]` | `precision: 0.0, recall: 0.0, f1: 0.0, support: 0` | **`precision: 1.0, recall: 1.0, f1: 1.0, support: 65`** | **Fixed:** Normalized gold & predicted labels match canonical `"solution"`. |
| | `confusion_matrix` (`["not_solution", "solution"]`) | `[[0, 0], [0, 0]]` (sum = 0) | **`[[127, 0], [0, 65]]`** (sum = 192) | **Fixed:** Total count = `127 + 65 = 192` test samples (Val: `[[132, 0], [0, 60]]`, sum = 192). |

---

## 4. Mathematical Metric Definitions & Applicability Rules

Let $N$ be the number of evaluated documents (`n_samples = 192`) and $K$ (or $L$) be the number of classes (or labels).

### 4.1 Single-Label Multiclass & Binary Metrics (`sentiment`, `response`, `solution`)
Each document $i \in \{1, \dots, N\}$ has one true class $y_i \in \{1, \dots, K\}$ and one predicted class $\hat{y}_i \in \{1, \dots, K\}$ (or $\text{None}$ if abstained):

- **Accuracy:**
  $$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[y_i = \hat{y}_i]$$
- **Per-Class Precision, Recall, F1, and Support (for class $k$):**
  $$\text{Support}_k = \sum_{i=1}^{N} \mathbb{I}[y_i = k], \quad TP_k = \sum_{i=1}^{N} \mathbb{I}[y_i = k \land \hat{y}_i = k], \quad FP_k = \sum_{i=1}^{N} \mathbb{I}[y_i \neq k \land \hat{y}_i = k], \quad FN_k = \sum_{i=1}^{N} \mathbb{I}[y_i = k \land \hat{y}_i \neq k]$$
  $$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}, \quad F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
  *Empty-Class Guard:* If $\text{Support}_k = 0$ and $TP_k + FP_k = 0$, $\text{Precision}_k$, $\text{Recall}_k$, and $F_{1,k}$ are set to `null` (`None`), **never** `1.0`.
- **Macro F1 & Weighted F1:**
  $$\text{Macro-F1} = \frac{1}{|K_{\text{active}}|} \sum_{k \in K_{\text{active}}} F_{1,k}, \quad \text{Weighted-F1} = \frac{1}{N} \sum_{k=1}^{K} \text{Support}_k \cdot F_{1,k}$$
- **Confusion Matrix $C \in \mathbb{Z}_{\ge 0}^{K \times K}$:**
  $$C_{a, b} = \sum_{i=1}^{N} \mathbb{I}[y_i = a \land \hat{y}_i = b], \quad \sum_{b=1}^{K} C_{a, b} = \text{Support}_a, \quad \sum_{a=1}^{K}\sum_{b=1}^{K} C_{a, b} = N_{\text{scorable}}$$
  For binary `solution`, the canonical axis order is fixed to `["not_solution", "solution"]`, so $C = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix} = \begin{bmatrix} 127 & 0 \\ 0 & 65 \end{bmatrix}$.

### 4.2 Multi-Label Metrics (`emotion`, `issue`)
Each document $i \in \{1, \dots, N\}$ has a binary ground-truth vector $\mathbf{y}_i \in \{0, 1\}^L$ and a binary predicted vector $\hat{\mathbf{y}}_i \in \{0, 1\}^L$:

- **Subset Accuracy (Exact Label-Set Match):**
  $$\text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[\mathbf{y}_i = \hat{\mathbf{y}}_i]$$
  The strictest multi-label metric: a document is counted correct if and only if all $L$ binary label decisions match simultaneously.
- **Hamming Loss & Hamming Accuracy:**
  $$\text{Hamming Loss} = \frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{l=1}^{L} \mathbb{I}[y_{i,l} \neq \hat{y}_{i,l}], \quad \text{Hamming Accuracy} = 1 - \text{Hamming Loss}$$
  Because sparse multi-label matrices are dominated by true negatives ($y_{i,l} = \hat{y}_{i,l} = 0$), Hamming accuracy is naturally high (`92.6%` on Issue even at `40.6%` subset accuracy); therefore **Hamming loss** is reported alongside **Subset accuracy**, **Macro F1**, and **Micro F1**.
- **Micro F1 (Global Pooled Decisions):**
  $$\text{Micro-F1} = \frac{2 \sum_{l=1}^{L} TP_l}{2 \sum_{l=1}^{L} TP_l + \sum_{l=1}^{L} FP_l + \sum_{l=1}^{L} FN_l}$$
- **Macro F1 (Unweighted Mean Across Labels):**
  $$\text{Macro-F1} = \frac{1}{L} \sum_{l=1}^{L} F_{1,l}$$

---

## 5. Comprehensive Audit of All Five Baseline Models

### 5.1 Thread-Grouped Splitting & Leakage Verification Across All Tasks
All five tasks share the exact same deterministic thread-grouped split (`SplitSpec(train_size=0.70, validation_size=0.15, test_size=0.15, random_state=20261116)`, grouped by `post_id`):
- **Training split:** `112` posts (`896` comments)
- **Validation split:** `24` posts (`192` comments)
- **Test split:** `24` posts (`192` comments)
- **Verified Disjoint Thread Sets:** `train_posts ∩ val_posts = ∅`, `train_posts ∩ test_posts = ∅`, `val_posts ∩ test_posts = ∅` (`leakage_detected: false`, `cross_split_id_collisions: []`, `cross_split_text_collisions: []`).

### 5.2 Per-Model Technical Audit

#### Model 1: Sentiment (`TfidfLogisticSentimentClassifier`, `tfidf_logistic v1`)
- **Task Definition:** 3-class single-label polarity classification over preprocessed comment text.
- **Label Vocabulary (Ordered):** `["negative", "neutral", "positive"]` (Full corpus: `negative: 339`, `neutral: 298`, `positive: 643`).
- **Nature:** Single-label multiclass (`multilabel: false`).
- **Supported Metrics (Held-Out Test, `n = 192`):** `accuracy = 1.0`, `macro_f1 = 1.0`, `micro_f1 = 1.0`, `precision = 1.0`, `recall = 1.0`; `per_class` supports `negative: 51, neutral: 46, positive: 95`; `confusion_matrix = [[51, 0, 0], [0, 46, 0], [0, 0, 95]]`.
- **Known Weaknesses:** In `generate_synthetic_demo.py#L74-L78`, every comment appends an explicit sentence from `SENTIMENT_CUES[sentiment]` (4 fixed sentences per polarity). TF-IDF memorizes those 12 cue phrases, yielding 100% synthetic accuracy. Real Reddit comments express sentiment implicitly, sarcastically, or with mixed clauses where bag-of-words TF-IDF will degrade substantially.

#### Model 2: Emotion (`TfidfLogisticEmotionClassifier`, `tfidf_logistic_multilabel v1`)
- **Task Definition:** 6-label multi-label affect detection via One-vs-Rest TF-IDF logistic regression (`threshold = 0.5`).
- **Label Vocabulary (Ordered):** `["confidence", "curiosity", "frustration", "gratitude", "hope", "stress"]` (Full corpus tag counts: `confidence: 112`, `curiosity: 298`, `frustration: 339`, `gratitude: 643`, `hope: 531`, `stress: 339`).
- **Nature:** Multi-label (`multilabel: true`).
- **Supported Metrics (Held-Out Test, `n = 192`):** `macro_f1 = 0.985714`, `micro_f1 = 0.995582`, `subset_accuracy = 0.984375` (`189 / 192`), `hamming_loss = 0.002604` (`3 / 1,152`), `hamming_accuracy = 0.997396`, `precision = 0.973684`, `recall = 1.0`. Per-label test F1: `curiosity: 1.0 (46)`, `frustration: 1.0 (51)`, `gratitude: 1.0 (95)`, `hope: 1.0 (79)`, `stress: 1.0 (51)`, `confidence: 0.914286 (support 16, precision 0.842105, recall 1.0)`.
- **Known Weaknesses:**
  1. In the synthetic generator (`#L167-L170`), emotion labels are deterministically derived from `(sentiment, response_type)`: `positive → [hope, gratitude]` (except `personal_experience + positive → [confidence, gratitude]`), `neutral → [curiosity]`, `negative → [stress, frustration]`. The 3 false positives on `confidence` occur on positive non-`personal_experience` comments that share overlapping tokens with `personal_experience` templates.
  2. As flagged by the evaluator warning, **zero documents in the corpus have an empty emotion set (`[]`)**, so the classifier's ability to abstain on affect-neutral procedural statements is untested.

#### Model 3: Issue (`TfidfLogisticIssueClassifier`, `tfidf_logistic_multilabel v1`)
- **Task Definition:** 8-label multi-label issue domain classification from comment text alone (no post title or thread context used in V1).
- **Label Vocabulary (Ordered):** `["academic_workload", "administrative_communication", "campus_infrastructure", "curriculum_teaching", "extracurriculars", "financial_concerns", "hostel_facilities", "placements"]` (`160` comments per domain in full corpus).
- **Nature:** Multi-label (`multilabel: true`).
- **Supported Metrics (Held-Out Test, `n = 192`):** `macro_f1 = 0.585376` (Val: `0.534049`), `micro_f1 = 0.577778` (Val: `0.561798`), `subset_accuracy = 0.40625` (`78 / 192`; Val: `0.390625`), `hamming_loss = 0.074219` (`114 / 1,536`; Val: `0.076172`), `precision = 1.0`, `recall = 0.453906`.
- **Per-Label Test Breakdown (`n = 192`):**
  - `academic_workload`: Precision `1.000`, Recall `0.025`, F1 `0.048780` (support `40`, 1 TP, 39 FN)
  - `hostel_facilities`: Precision `1.000`, Recall `0.250`, F1 `0.400000` (support `8`, 2 TP, 6 FN)
  - `curriculum_teaching`: Precision `1.000`, Recall `0.375`, F1 `0.545455` (support `16`, 6 TP, 10 FN)
  - `extracurriculars`: Precision `1.000`, Recall `0.4375`, F1 `0.608696` (support `32`, 14 TP, 18 FN)
  - `administrative_communication`: Precision `1.000`, Recall `0.450`, F1 `0.620690` (support `40`, 18 TP, 22 FN)
  - `placements`: Precision `1.000`, Recall `0.59375`, F1 `0.745098` (support `32`, 19 TP, 13 FN)
  - `campus_infrastructure`: Precision `1.000`, Recall `0.750`, F1 `0.857143` (support `16`, 12 TP, 4 FN)
  - `financial_concerns`: Precision `1.000`, Recall `0.750`, F1 `0.857143` (support `8`, 6 TP, 2 FN)
- **Known Weaknesses (First-Principles Explanation of Low Recall / 100% Precision):**
  1. Each synthetic post is assigned a unique `{detail}` phrase (20 distinct `{detail}` strings per domain). Because splitting is **grouped by `post_id`**, all 8 comments sharing a `{detail}` stay in the same split—so **no `{detail}` in the 24 test threads ever appears in the 112 training threads**.
  2. Therefore, on held-out test comments, the text-only classifier can only recognize the domain when the comment template includes the domain's `{subject}` phrase (e.g., `"placement office"`, `"finance office"`, `"course schedule"`). Several `ROLE_TEXT` templates (such as `"My view is that {detail} should be explained in one place..."` and several Hindi templates) omit `{subject}`, leaving zero domain-identifying training tokens in the comment. The One-vs-Rest probabilities fall below `0.5`, emitting `[]` (`0` false positives → `100%` precision, `40.6%` exact-match recall).
  3. On `academic_workload`, `"course schedule"` shares the token `"course"` with `curriculum_teaching` (`"course coordinator"`), further depressing its unigram/bigram margin below the `0.5` threshold (`2.5%` recall). Incorporating thread/post title context or lowering/calibrating per-label thresholds on validation data is required for comment-level issue attribution.

#### Model 4: Response (`TfidfLogisticResponseClassifier`, `tfidf_logistic_multiclass v1`)
- **Task Definition:** 6-class single-label classification of community response discourse roles.
- **Label Vocabulary (Fixed Canonical Order):**
  1. `agreement` (`213` corpus, `31` test)
  2. `personal_experience` (`210` corpus, `33` test)
  3. `opinion` (`213` corpus, `32` test)
  4. `advice` (`221` corpus, `36` test)
  5. `solution` (`205` corpus, `29` test)
  6. `counterargument` (`218` corpus, `31` test)
- **Nature:** Single-label multiclass (`multilabel: false`).
- **Supported Metrics (Held-Out Test, `n = 192`):** `accuracy = 1.0`, `macro_f1 = 1.0`, `micro_f1 = 1.0`, `weighted_f1 = 1.0`, `precision = 1.0`, `recall = 1.0`; `confusion_matrix = diag(31, 33, 32, 36, 29, 31)` (total = `192`).
- **Known Weaknesses:** Each response role is generated from 4 distinct template stems in `ROLE_TEXT`. In real community threads, `advice` vs. `solution` and `opinion` vs. `counterargument` have substantial lexical overlap and require thread-parent context (`parent_comment_text`, currently unused by V1).

#### Model 5: Solution (`TfidfLogisticSolutionClassifier`, `tfidf_logistic_binary v1`)
- **Task Definition:** Binary actionable-recommendation detector (`not_solution` vs. `solution`).
- **Label Vocabulary (Fixed Canonical Order):** `["not_solution", "solution"]` (Full corpus: `not_solution: 854`, `solution: 426`).
- **Nature:** Single-label binary (`multilabel: false`).
- **Supported Metrics (Held-Out Test, `n = 192`):** `accuracy = 1.0`, `macro_f1 = 1.0`, `weighted_f1 = 1.0`, `precision = 1.0`, `recall = 1.0`; `per_class`: `not_solution` (`precision: 1.0, recall: 1.0, f1: 1.0, support: 127`), `solution` (`precision: 1.0, recall: 1.0, f1: 1.0, support: 65`); `confusion_matrix = [[127, 0], [0, 65]]` (total = `192`).
- **Known Weaknesses:** Trained on synthetic labels where `is_solution` is true if and only if `response_type in {"advice", "solution"}`. Furthermore, `is_solution = True` identifies **proposed** advice or solutions, never verified issue resolution (which is tracked separately via `resolution_state`).

---

## 6. Corrections Implemented

1. **Canonical Solution Label Normalization at Dataset Load (`ml/data/dataset.py#L375-L390`):**
   - Updated `_solution_to_record(a)` to map `a.is_solution` via `target_label(bool(a.is_solution))` (`"not_solution"` / `"solution"`), aligning `TrainingRecord.label`, `LoadedDataset.labels`, and `LoadedDataset.label_counts` with `TARGET_LABELS`.
2. **Strict Bidirectional Solution Target Parsing in Trainer (`ml/training/trainer.py#L357-L388`, `#L886`):**
   - Updated `records_to_annotations(..., TaskType.SOLUTION)` to accept both canonical labels (`"solution"`, `"not_solution"`) via `label_to_bool()` and boolean spellings (`"true"`, `"false"`, `"1"`, `"0"`, `"yes"`, `"no"`) via `parse_bool_cell()`, raising a `ValueError` on any blank or unrecognised label rather than silently coercing to `False`.
   - Updated `_default_label_options(TaskType.SOLUTION)` to `["not_solution", "solution"]`.
3. **Complete Multi-Label & Single-Label Metric Extraction (`ml/evaluation/reports.py#L134-L305` & `ml/evaluation/metrics.py#L64-L215`):**
   - Updated `_extract_metrics()` to extract `subset_accuracy`, `hamming_loss`, `hamming_accuracy`, `macro_f1`, `micro_f1`, `precision`, `recall`, and `per_label` (`per_label_f1`, `per_label_precision`, `per_label_recall`, `per_label_support`) for multi-label tasks (`is_multilabel=True`), while keeping single-label `accuracy`, `weighted_f1`, `per_class`, and `confusion_matrix` strictly `None` / `{}`.
   - Updated `_extract_metrics()` to extract `accuracy`, `macro_f1`, `micro_f1`, `weighted_f1`, `precision`, `recall`, `per_class` (`per_class_f1`, `per_class_precision`, `per_class_recall`, `per_class_support`), and `confusion_matrix` (`class_order`) for single-label/binary tasks (`is_multilabel=False`), while keeping multi-label metrics strictly `None` / `{}`.
   - Propagated `raw_report.is_meaningful == False` and `raw_report.reason` into `ClassificationMetricSet` so withheld evaluations remain `None` (`null` in JSON) and `is_meaningful: false`.
   - Added empty-class guards in `compute_metrics()` and `heldout_class_metrics()` so classes with `support == 0` and `0` predictions emit `precision: null, recall: null, f1: null` rather than misleading `0.0` or `1.0` scores.
4. **Consistent Evaluation Post-Processing, Integrity Assertions & Isolated V2 Artifacts (`backend/train_demo_baselines.py`):**
   - Added `normalize_solution_label()` and updated `label_set()` and `heldout_class_metrics()` to normalize both gold and predicted solution values to `TARGET_LABELS = ["not_solution", "solution"]`.
   - Added `verify_task_evaluation_integrity()` asserting split disjointness (`post_id`), vocabulary consistency, confusion-matrix sample sum (`192`), per-class support sum (`192`), positive support for both binary classes (`127` and `65`), multi-label metric completeness, and empty-class safety.
   - Retrained all 5 baselines into the isolated directory `ml/artifacts/demo_synthetic_v2/` while leaving `ml/artifacts/demo_synthetic/` and `ml/artifacts/registry.json` untouched.
5. **Demo API & Model Report Display Wiring (`backend/app/api/demo.py#L20-L27`, `backend/app/static/js/views/models.js`):**
   - Pointed `demo.py::ARTIFACT_DIR` to `ml/artifacts/demo_synthetic_v2` (with automatic fallback to `demo_synthetic`).
   - Updated `models.js` to display emitted `subset_accuracy` and `hamming_loss` on multi-label cards (retaining the `"Not emitted"` badge whenever a metric is `null`) and to display the active artifact directory dynamically.

---

## 7. Regression & Integrity Tests Executed

| Test Suite | Command / Scope | Result | What Was Verified |
| --- | --- | --- | --- |
| **Synthetic Demo & V2 Artifact Integrity** | `pytest backend/tests/test_synthetic_demo_integrity.py` | **6 passed** | Seed reproducibility (`20261116`), canonical 6-class response taxonomy, preservation of frozen v1 `ml/artifacts/demo_synthetic/`, thread-grouped `post_id` split disjointness (`112 / 24 / 24` threads), Solution label normalization & `[[127, 0], [0, 65]]` confusion-matrix sum (`192`), empty-class `null` metric guard, multi-label `subset_accuracy` & `hamming_loss` reporting |
| **Demo FastAPI Contract & Provenance** | `pytest backend/tests/test_demo_api.py` | **4 passed** | All 10 `/api/demo/*` Pydantic schemas validate against v2 artifacts, search/filter/pagination, 422 invalid-filter rejection, explicit resolution evidence rules |
| **ML Evaluation Framework** | `pytest backend/test_ml_evaluation.py` | **18 passed** | Single-label vs. multi-label metric extraction, `subset_accuracy`/`hamming_loss`/`per_label` propagation, withheld/null/missing/empty evaluation handling |
| **ML Splitting & Training Framework** | `pytest backend/test_ml_splitting.py backend/test_ml_training.py` | **138 passed** | Group-aware splitting, leakage detection, canonical Solution record conversion, registry isolation |
| **NLP Task Classifier Suites** | `pytest backend/test_solution_detection.py backend/test_sentiment_emotion.py backend/test_issue_identification.py backend/test_response_classification.py backend/test_app_startup.py` | **776 passed** (21 subtests) | Binary solution detector, sentiment/emotion evaluators, multi-label issue classifier, 6-class response classifier, app startup |
| **Combined Backend & ML Pytest Run** | `942 passed, 21 subtests passed in 11.26s` | **100% Pass** | Zero regressions across all 10 test modules |
| **Live Headless Chrome E2E & Responsive Suite** | `verify_rci.mjs` against `http://127.0.0.1:8000/app/` | **16 / 16 E2E checks, 31 / 31 responsive checks** | All 9 frontend routes load cleanly, `#/models` displays corrected v2 Solution 2×2 confusion matrix (`127` / `65`) and Emotion/Issue `subset_accuracy` & `hamming_loss`, `0` console errors, `0` failed network requests |

---

## 8. Known Limitations & Remaining Work Before Genuine Reddit Data

1. **Synthetic Template Separability:** The pilot corpus (`data/demo_synthetic/`) uses fixed string templates with deterministic cue sentences. Held-out test metrics on this corpus validate **pipeline mathematical integrity and leakage-free execution**, not real-world generalization.
2. **Comment-Only Issue Classification Context:** `TfidfLogisticIssueClassifier` scores comment bodies in isolation without the parent post title or body. When a comment uses an elliptic reply template without repeating the domain noun phrase, One-vs-Rest probabilities fall below `0.5` (`40.6%` test subset accuracy). Concatenating post title context (while keeping `group_by="post_id"`) or calibrating per-label thresholds on validation data is recommended before real-data deployment.
3. **Zero-Emotion Ground Truth Coverage:** Every synthetic comment is assigned 1 or 2 emotions. Real corpora include purely factual or procedural comments with zero emotions (`[]`); human annotation guidelines must include neutral/no-emotion examples.
4. **Multilingual & Indic Tokenization Scope:** The synthetic pilot covers English, Devanagari Hindi, and Hinglish. Before production use on the full research scope, human-annotated data and Unicode grapheme-aware tokenization for Bengali, Odia, Banglish, Romanized Odia, emoji tokens, and Gen-Z slang must be added.

---

## 9. Release Gate Verdict

### **Verdict: PASS (Synthetic Pipeline & Evaluation Integrity Gate)**

- **Internal Mathematical Consistency:** **PASS.** Every task's label vocabulary, dataset class distribution, held-out confusion matrix ($K \times K$, sum = `192`), per-class/per-label support sum (`192`), single-label vs. multi-label metric family separation, and thread-grouped (`post_id`) split isolation (`112 / 24 / 24` threads, zero leakage) are internally consistent and enforced by automated assertions.
- **Artifact & Registry Safety:** **PASS.** Frozen v1 artifacts (`ml/artifacts/demo_synthetic/`) and the canonical model registry (`ml/artifacts/registry.json`) remain untouched; corrected artifacts are isolated in `ml/artifacts/demo_synthetic_v2/`.
- **Epistemic & Scientific Honesty:** **PASS.** Synthetic provenance warnings remain prominent on every route, uncomputed metrics remain `null` (`"Not emitted"`), and synthetic template scores are explicitly bounded from real-world performance claims.
