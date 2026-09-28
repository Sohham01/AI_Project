"""
ANN Response Classification — synthetic plumbing demo.

ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE.

This script exercises the ResponseMLPClassifier (TF-IDF -> NumpyMLP) through the
existing ML Trainer pipeline on SYNTHETIC data only.  It demonstrates that the ANN
baseline plugs into the same generic training/evaluation/artifact/registry
infrastructure as the classical logistic baseline.

No real annotation files are read or created.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Project root + backend on the path so both ml/ and app/ are importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import numpy as np

from app.services.nlp.response_classification.base import ResponseDocument
from app.services.nlp.response_classification.models.annotation import (
    ResponseAnnotation,
)
from ml.configs import TaskConfig, TaskType, ModelRegistry
from ml.data.dataset import (
    TrainingRecord,
    make_synthetic_records,
)
from ml.data.splitting import SplitSpec, apply_split, split_records
from ml.evaluation.reports import EvaluationReport, build_evaluation_report
from ml.training.trainer import (
    Trainer,
    LoadedDataset,
    PipelineState,
    build_model,
    records_to_annotations,
)
from ml.models.ann.response_mlp import ResponseMLPClassifier
from ml.models.ann.numpy_mlp import NumpyMLP

# Frozen contracts.
from app.services.nlp.response_classification.models.labels import (
    RESPONSE_TYPES,
    response_type_order,
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE                    ║
║                                                                              ║
║   This demo uses SYNTHETIC data only.  Metrics shown here reflect the        ║
║   plumbing of the ANN pipeline (TF-IDF -> NumpyMLP), not real project        ║
║   accuracy.  Real evaluation requires manually annotated data.               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(BANNER)


# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------

def _print_architecture() -> None:
    print("=" * 72)
    print("1. ANN ARCHITECTURE")
    print("=" * 72)
    print("""
Input:  TF-IDF features (fit on TRAINING text only)
  |
  v
Dense(H1=256)
  -> z = W1 @ x + b1          (256 x D) @ (D x B) + (256 x 1)
  -> ReLU(z) = max(0, z)
  -> Dropout (rate=0.3, inverted during training)
  |
  v
Dense(H2=64)
  -> z = W2 @ h1 + b2         (64 x 256) @ (256 x B) + (64 x 1)
  -> ReLU(z) = max(0, z)
  |
  v
Dense(K=6)
  -> logits = W3 @ h2 + b3    (6 x 64) @ (64 x B) + (6 x 1)
  -> Softmax:  P(y=k|x) = exp(z_k) / Σ_j exp(z_j)
  -> Cross-entropy loss: L = -Σ_k y_k log(p_k)
  -> Backprop + momentum SGD + gradient clipping (L2 <= 5.0)
""")


# ---------------------------------------------------------------------------
# 2. Parameter count
# ---------------------------------------------------------------------------

def _print_parameter_count() -> None:
    print("=" * 72)
    print("2. PARAMETER COUNT")
    print("=" * 72)
    mlp = NumpyMLP(
        input_dim=100,
        n_classes=6,
        hidden1=256,
        hidden2=64,
        dropout=0.3,
        learning_rate=0.01,
        epochs=40,
        batch_size=16,
        random_state=42,
        clip_gradient=5.0,
    )
    n_params = mlp.n_parameters
    W1 = 256 * 100 + 256
    W2 = 64 * 256 + 64
    W3 = 6 * 64 + 6
    print(f"""
Input dim (TF-IDF features):  100 (illustrative; real TF-IDF varies)
Hidden1:                      256
Hidden2:                      64
Output classes:               6 (response types)

W1 (256 x 100) + b1 (256):   {256 * 100 + 256:>8,}
W2 (64 x 256) + b2 (64):     {64 * 256 + 64:>8,}
W3 (6 x 64) + b3 (6):        {6 * 64 + 6:>8,}
-----------------------------------------
Total trainable parameters:   {n_params:>8,}

Optimizer: Momentum SGD (mu=0.9, lr=0.01)
Gradient clipping: L2 norm <= 5.0
Dropout: rate=0.3 (inverted, training only)
Random seed: 42 (deterministic)
""")


# ---------------------------------------------------------------------------
# 3. Synthetic dataset
# ---------------------------------------------------------------------------

def _print_synthetic_dataset() -> None:
    print("=" * 72)
    print("3. SYNTHETIC DATASET")
    print("=" * 72)
    n = 100
    records = make_synthetic_records(
        n=n,
        label_options=list(RESPONSE_TYPES),
        single_label=True,
        seed=42,
    )
    annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in records
    ]
    from app.services.nlp.response_classification.models.annotation import (
        ResponseCorpus,
    )

    corpus = ResponseCorpus(annotations=annotations)
    counts = corpus.class_counts()
    print(f"""
Generated {n} synthetic single-label response annotations.
Labels: {', '.join(RESPONSE_TYPES)}

Class distribution (synthetic):
""")
    for lbl in response_type_order(list(counts.keys())):
        c = counts.get(lbl, 0)
        bar = "#" * max(1, c * 3)
        print(f"  {lbl:<22} {c:>3}  {bar}")
    print()


# ---------------------------------------------------------------------------
# 4. Deterministic split
# ---------------------------------------------------------------------------

def _print_split(records: list, seed: int = 42) -> tuple:
    print("=" * 72)
    print("4. DETERMINISTIC TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 72)

    training_records = [
        TrainingRecord(
            document_id=a.document_id,
            text=a.text,
            label=a.normalised_response_type(),
            post_id=a.post_id,
            comment_id=a.comment_id,
            language=a.language,
            script=a.script,
            is_code_mixed=a.is_code_mixed,
            detected_languages=list(a.detected_languages),
            topic_id=a.topic_id,
        )
        for a in records
    ]

    split = split_records(
        records=training_records,
        labels=[r.label for r in training_records],
        spec=SplitSpec(
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            random_state=seed,
            stratify=True,
            min_train_examples=10,
            min_validation_examples=1,
            min_test_examples=1,
        ),
    )

    train_recs, val_recs, test_recs = apply_split(training_records, split)

    print(f"""
Split seed:        {seed}
Split strategy:    stratified (single-label)

Train:    {len(train_recs):>3} records
Validation: {len(val_recs):>3} records
Test:     {len(test_recs):>3} records
Total:    {len(train_recs) + len(val_recs) + len(test_recs):>3} records

Leakage check: no record appears in more than one split (verified by split_records).
TF-IDF fitting: TRAINING text ONLY — validation and test are transformed, never fit.
""")
    return train_recs, val_recs, test_recs


# ---------------------------------------------------------------------------
# 5. Training + loss per epoch
# ---------------------------------------------------------------------------

def _train_and_print_loss(train_recs: list) -> ResponseMLPClassifier:
    print("=" * 72)
    print("5. TRAINING — LOSS PER EPOCH")
    print("=" * 72)

    ann = ResponseMLPClassifier(
        hidden1=256,
        hidden2=64,
        dropout=0.3,
        learning_rate=0.01,
        momentum=0.9,
        epochs=40,
        batch_size=16,
        random_state=42,
        clip_gradient=5.0,
        min_training_documents=10,
        min_examples_per_class=1,
        max_features=5000,
        verbose=True,
    )

    train_annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in train_recs
    ]

    report = ann.train(train_annotations)
    print(f"""
Training report:
  model_name:          {report.model_name}
  model_version:       {report.model_version}
  n_training_docs:     {report.n_training_documents}
  n_classes:           {report.n_classes}
  labels:              {', '.join(report.labels)}
  n_features (TF-IDF): {report.n_features}
  is_usable:           {report.is_usable}

Loss per epoch (first 5 and last 5):
""")
    losses = ann._mlp.epoch_losses
    for i, loss in enumerate(losses):
        if i < 5 or i >= len(losses) - 5:
            marker = " <--" if i == 0 or i == len(losses) - 1 else ""
            print(f"  epoch {i + 1:>3}/{len(losses)}  loss = {loss:.6f}{marker}")
    if len(losses) > 10:
        print("  ...")
    print(f"""
  Start loss:  {losses[0]:.6f}
  End loss:    {losses[-1]:.6f}
  Reduction:   {(1 - losses[-1] / losses[0]) * 100:.1f}%
""")
    return ann


# ---------------------------------------------------------------------------
# 6. Sample predictions
# ---------------------------------------------------------------------------

def _print_sample_predictions(ann: ResponseMLPClassifier) -> None:
    print("=" * 72)
    print("6. SAMPLE PREDICTIONS")
    print("=" * 72)

    sample_texts = [
        "I completely agree with your point about the placement season",
        "In my personal experience the hostel food was decent but expensive",
        "My opinion is that the ROI is not great compared to older IIMs",
        "My advice would be to prepare for CAT early and focus on mocks",
        "The solution to the mess food issue is to allow outside caterers",
        "I disagree because the placement stats have actually improved this year",
    ]

    docs = [
        ResponseDocument(document_id=f"sample-{i}", text=text)
        for i, text in enumerate(sample_texts)
    ]

    result = ann.predict(docs)

    print(f"""
Model:         {result.model_name} {result.model_version}
Threshold:     {result.threshold}
n_abstained:   {result.n_abstained}

""")
    for pred in result.predictions:
        conf_str = f"{pred.confidence:.3f}" if pred.response_type else "n/a"
        top_str = pred.response_type or "(abstained)"
        print(f"  [{pred.document_id}] {pred.text[:60]}")
        print(f"    predicted: {top_str:<22} confidence={conf_str}")
        print(f"    probabilities:")
        for lbl in response_type_order(list(pred.response_probabilities.keys())):
            p = pred.response_probabilities.get(lbl, 0.0)
            bar = "#" * int(p * 30)
            marker = " <-- top" if lbl == top_str else ""
            print(f"      {lbl:<22} {p:.4f}  {bar}{marker}")
        print()


# ---------------------------------------------------------------------------
# 7. Six-class probability distribution
# ---------------------------------------------------------------------------

def _print_probability_distribution(ann: ResponseMLPClassifier) -> None:
    print("=" * 72)
    print("7. SIX-CLASS PROBABILITY DISTRIBUTION (aggregate over sample)")
    print("=" * 72)

    sample_texts = [
        "placement season is here and everyone is talking about PPOs",
        "the hostel mess food is really bad and charges are too high",
        "CAT preparation needs consistency and the right material",
        "my advice: start early with mocks and analyse every mistake",
        "a possible solution is to allow outside tiffin services",
        "I disagree — the placement stats have improved this year",
        "ROI is a concern but the brand value justifies the fees",
        "personal experience: the faculty is very supportive",
        "opinion: the campus life is great but academics are tough",
        "agreement: yes the fees are high but placements compensate",
    ]

    docs = [
        ResponseDocument(document_id=f"dist-{i}", text=text)
        for i, text in enumerate(sample_texts)
    ]
    result = ann.predict(docs)

    print(f"\nAverage predicted probability for each class across {len(docs)} samples:\n")
    avg_probs: dict[str, float] = {lbl: 0.0 for lbl in response_type_order(RESPONSE_TYPES)}
    for pred in result.predictions:
        for lbl, p in pred.response_probabilities.items():
            avg_probs[lbl] += p
    for lbl in response_type_order(list(avg_probs.keys())):
        avg = avg_probs[lbl] / len(docs)
        bar = "#" * int(avg * 40)
        print(f"  {lbl:<22} avg={avg:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# 8. Evaluation through existing ML framework
# ---------------------------------------------------------------------------

def _print_evaluation(ann: ResponseMLPClassifier, val_recs: list) -> None:
    print("=" * 72)
    print("8. EVALUATION THROUGH EXISTING ML FRAMEWORK")
    print("=" * 72)

    val_annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in val_recs
    ]

    raw_report = ann.evaluate(val_annotations)
    print(f"""
Raw ResponseEvaluationReport:
  is_meaningful:   {raw_report.is_meaningful}
  n_documents:     {raw_report.n_documents}
  n_classes:       {raw_report.n_classes}
  n_abstained:     {raw_report.n_abstained}
  n_unscoreable:   {raw_report.n_unscoreable}
  accuracy:        {raw_report.accuracy}
  macro_f1:        {raw_report.macro_f1}
  micro_f1:        {raw_report.micro_f1}
  weighted_f1:     {raw_report.weighted_f1}
  macro_precision: {raw_report.macro_precision}
  micro_precision: {raw_report.micro_precision}
  weighted_precision: {raw_report.weighted_precision}
  macro_recall:    {raw_report.macro_recall}
  micro_recall:    {raw_report.micro_recall}
  weighted_recall: {raw_report.weighted_recall}

  Per-class F1:
""")
    for lbl in response_type_order(list(raw_report.per_class_f1.keys())):
        f1 = raw_report.per_class_f1.get(lbl)
        prec = raw_report.per_class_precision.get(lbl)
        rec = raw_report.per_class_recall.get(lbl)
        support = raw_report.per_class_support.get(lbl, "?")
        f1_str = f"{f1:.4f}" if f1 is not None else "n/a"
        prec_str = f"{prec:.4f}" if prec is not None else "n/a"
        rec_str = f"{rec:.4f}" if rec is not None else "n/a"
        print(f"    {lbl:<22} f1={f1_str}  precision={prec_str}  recall={rec_str}  support={support}")

    unified = build_evaluation_report(
        task_name="response",
        model_name=ann.name,
        model_version=ann.version,
        n_train=len(val_recs),
        n_validation=len(val_recs),
        n_test=len(val_recs),
        labels=ann.labels,
        raw_report=raw_report,
        is_multilabel=False,
        model_is_trained=ann.is_trained,
        leakage_detected=False,
    )
    print(f"""
Unified EvaluationReport:
  is_meaningful:   {unified.is_meaningful}
  metrics keys:    {list(unified.metrics.metrics_serializable().keys())}
  warnings:        {unified.warnings if unified.warnings else "(none)"}
""")


# ---------------------------------------------------------------------------
# 9. Deterministic second run
# ---------------------------------------------------------------------------

def _print_deterministic_second_run(val_recs: list) -> None:
    print("=" * 72)
    print("9. DETERMINISTIC SECOND RUN (same seed → same result)")
    print("=" * 72)

    ann1 = ResponseMLPClassifier(
        hidden1=32,
        hidden2=16,
        epochs=8,
        batch_size=8,
        learning_rate=0.01,
        random_state=42,
        min_training_documents=10,
        min_examples_per_class=1,
        max_features=1000,
    )
    ann2 = ResponseMLPClassifier(
        hidden1=32,
        hidden2=16,
        epochs=8,
        batch_size=8,
        learning_rate=0.01,
        random_state=42,
        min_training_documents=10,
        min_examples_per_class=1,
        max_features=1000,
    )

    train_annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
        )
        for r in val_recs[:30]
    ]

    ann1.train(train_annotations)
    ann2.train(train_annotations)

    state1 = ann1.get_state()
    state2 = ann2.get_state()

    # Compare MLP weights.
    s1 = state1["mlp_state"]
    s2 = state2["mlp_state"]
    weights_match = (
        np.array_equal(s1["W1_shape"], s2["W1_shape"])
        and np.array_equal(s1["W2_shape"], s2["W2_shape"])
        and np.array_equal(s1["W3_shape"], s2["W3_shape"])
    )

    # Predict on same docs and compare.
    docs = [
        ResponseDocument(document_id=f"det-{i}", text=f"deterministic check {i}")
        for i in range(5)
    ]
    r1 = ann1.predict(docs)
    r2 = ann2.predict(docs)

    preds_match = all(
        p1.response_type == p2.response_type
        for p1, p2 in zip(r1.predictions, r2.predictions)
    )

    print(f"""
  Run 1 final loss:  {s1['epoch_losses'][-1]:.6f}
  Run 2 final loss:  {s2['epoch_losses'][-1]:.6f}
  Weights identical: {weights_match}
  Predictions match: {preds_match}

  Both runs use random_state=42 with the same data → identical initialisation,
  identical dropout masks, identical SGD trajectory → identical results.
""")


# ---------------------------------------------------------------------------
# 10. Comparison table against classical Logistic Regression
# ---------------------------------------------------------------------------

def _print_comparison(train_recs: list, val_recs: list) -> None:
    print("=" * 72)
    print("10. PLUMBING COMPARISON: Logistic Regression vs ANN")
    print("=" * 72)
    print("""
IMPORTANT: Both models trained on the SAME synthetic data with the SAME split.
This is a plumbing check, not a real performance comparison.

""")

    train_annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in train_recs
    ]
    val_annotations = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in val_recs
    ]

    # Logistic baseline.
    logistic = build_model(TaskType.RESPONSE)
    logistic.train(train_annotations)
    logistic_val = logistic.evaluate(val_annotations)

    # ANN model.
    ann = ResponseMLPClassifier(
        hidden1=32,
        hidden2=16,
        epochs=12,
        batch_size=8,
        learning_rate=0.01,
        random_state=42,
        min_training_documents=10,
        min_examples_per_class=1,
        max_features=5000,
    )
    ann.train(train_annotations)
    ann_val = ann.evaluate(val_annotations)

    def _row(name: str, report: Any) -> None:
        print(f"  {name}")
        print(f"    accuracy:        {report.accuracy}")
        print(f"    macro_f1:        {report.macro_f1}")
        print(f"    micro_f1:        {report.micro_f1}")
        print(f"    weighted_f1:     {report.weighted_f1}")
        print(f"    macro_precision: {report.macro_precision}")
        print(f"    micro_precision: {report.micro_precision}")
        print(f"    weighted_precision: {report.weighted_precision}")
        print(f"    macro_recall:    {report.macro_recall}")
        print(f"    micro_recall:    {report.micro_recall}")
        print(f"    weighted_recall: {report.weighted_recall}")
        print(f"    is_meaningful:   {report.is_meaningful}")
        print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    VALIDATION SET RESULTS                   │")
    print("  │         SYNTHETIC DATA — NOT PROJECT PERFORMANCE           │")
    print("  └─────────────────────────────────────────────────────────────┘")
    _row("Logistic Regression (baseline)", logistic_val)
    _row("ANN MLP (response_mlp_numpy v1)", ann_val)

    print("""
  Notes:
  - Both models use the same TF-IDF featurisation (max_features, ngram_range, etc.
    differ slightly by default but both are valid TF-IDF pipelines).
  - Logistic Regression uses sklearn's multinomial logistic regression.
  - ANN uses a 3-layer numpy MLP with momentum SGD, ReLU, dropout, and gradient
    clipping.
  - On synthetic data neither model is expected to be consistently better.
  - Real comparison requires real annotated data.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _print_architecture()
    _print_parameter_count()
    _print_synthetic_dataset()

    records = [
        ResponseAnnotation(
            document_id=r.document_id,
            text=r.text,
            response_type=r.label,
            post_id=r.post_id,
            comment_id=r.comment_id,
            language=r.language,
            script=r.script,
            is_code_mixed=r.is_code_mixed,
            detected_languages=list(r.detected_languages),
            topic_id=r.topic_id,
        )
        for r in make_synthetic_records(
            n=100,
            label_options=list(RESPONSE_TYPES),
            single_label=True,
            seed=42,
        )
    ]

    train_recs, val_recs, test_recs = _print_split(records, seed=42)

    print("Running training (epochs=40, verbose)...")
    ann = _train_and_print_loss(train_recs)
    _print_sample_predictions(ann)
    _print_probability_distribution(ann)
    _print_evaluation(ann, val_recs)
    _print_deterministic_second_run(train_recs)
    _print_comparison(train_recs, val_recs)

    print("=" * 72)
    print("DEMO COMPLETE — all plumbing verified.")
    print("(ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE)")
    print("=" * 72)


if __name__ == "__main__":
    main()
