#!/usr/bin/env python3
"""
==============================================================================
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
==============================================================================

This script exercises the ML training framework end-to-end on SYNTHETIC data
only.  The metrics printed below are NOT real project performance — they are
generated from fabricated text snippets to verify that the pipeline plumbing
(loading, splitting, training, evaluation, artifact saving, registry) works
correctly.

Real annotation datasets are loaded from:
    data/annotations/sentiment_annotations.csv
    data/annotations/emotion_annotations.csv
    data/annotations/issue_annotations.csv
    data/annotations/response_annotations.csv
    data/annotations/solution_annotations.csv

When those files are absent this demo falls back to synthetic data and
clearly labels it as such.

No files under data/annotations/ are created or modified by this script.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure import paths.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
for p in (project_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.configs import TaskConfig, TaskType
from ml.data.dataset import (
    AnnotationNotFoundError,
    TrainingRecord,
    make_synthetic_records,
    split_label,
)
from ml.data.splitting import SplitResult, SplitSpec, apply_split, check_leakage, split_records
from ml.evaluation.reports import EvaluationReport, build_evaluation_report, report_class_distribution
from ml.training.trainer import Trainer, build_model, records_to_annotations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANNER = """
\033[1;36m
==============================================================================
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
==============================================================================
\033[0m
"""


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("-" * 60)


def main() -> None:
    print(BANNER)

    # We demonstrate with sentiment (single-label) and issue (multi-label).

    for task_type in [TaskType.SENTIMENT, TaskType.ISSUE]:
        _demo_task(task_type)

    _demo_leakage_detection()
    _demo_registry()

    print("\n\033[1;32mPipeline plumbing check complete.\033[0m")
    print("Remember: all metrics above are from SYNTHETIC data and are NOT project performance.")


def _demo_task(task_type: TaskType) -> None:
    section(f"Task: {task_type.value} ({_task_display(task_type)})")

    tmpdir = tempfile.mkdtemp()
    artifact_dir = os.path.join(tmpdir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    config = TaskConfig(
        task_name=task_type.value,
        artifact_directory=artifact_dir,
        dataset_path="data/annotations/does_not_exist.csv",  # force synthetic fallback
    )

    trainer = Trainer(config)

    # 1. Dataset creation (synthetic).
    dataset = trainer.load_dataset(use_synthetic=True)
    print(f"  source      : {dataset.source}")
    print(f"  records     : {dataset.n_records}")
    print(f"  labels      : {dataset.labels}")
    print(f"  label counts: {dataset.label_counts}")

    # 2. Class distribution.
    dist = report_class_distribution(dataset.label_counts, dataset.n_records)
    print(f"  imbalance_ratio: {dist['imbalance_ratio']:.2f}")
    if dist.get("warning"):
        print(f"  imbalance warning: {dist['warning']}")

    # 3. Deterministic splitting.
    split = trainer.split_dataset()
    train_recs, val_recs, test_recs = apply_split(dataset.records, split)
    print(f"  random_state: {split.random_state}")
    print(f"  stratify_used: {split.stratify_used}")
    print(f"  stratify_available: {split.stratify_available}")
    print(f"  train/val/test: {len(split.train)}/{len(split.validation)}/{len(split.test)}")

    # 4. Leakage check.
    leak = check_leakage(train_recs, val_recs, test_recs)
    print(f"  leakage_detected: {leak['leakage_detected']}")
    if leak["cross_split_id_collisions"]:
        print(f"    ID collisions: {leak['cross_split_id_collisions']}")
    if leak["cross_split_text_collisions"]:
        print(f"    text collisions: {leak['cross_split_text_collisions']}")

    # 5. Train one existing baseline classifier.
    training_report = trainer.train()
    model = trainer.state.trained_model
    print(f"  model       : {model.name} {model.version}")
    print(f"  is_trained  : {model.is_trained}")
    print(f"  labels      : {model.labels}")

    # 6. Validation evaluation.
    val_report = trainer.evaluate_validation()
    print(f"  validation meaningful: {val_report.is_meaningful}")
    if val_report.metrics.macro_f1 is not None:
        print(f"  validation macro_f1  : {val_report.metrics.macro_f1:.4f}")
    if val_report.metrics.accuracy is not None:
        print(f"  validation accuracy  : {val_report.metrics.accuracy:.4f}")
    for w in val_report.warnings:
        print(f"    ! {w}")

    # 7. Test evaluation.
    test_report = trainer.evaluate_test()
    print(f"  test meaningful: {test_report.is_meaningful}")
    if test_report.metrics.macro_f1 is not None:
        print(f"  test macro_f1  : {test_report.metrics.macro_f1:.4f}")
    if test_report.metrics.accuracy is not None:
        print(f"  test accuracy  : {test_report.metrics.accuracy:.4f}")
    for w in test_report.warnings:
        print(f"    ! {w}")

    # 8. Artifact creation.
    artifact_path = trainer.save_artifact()
    print(f"  artifact    : {artifact_path}")
    print(f"  artifact exists: {os.path.exists(artifact_path)}")

    # 9. Registry entry.
    trainer._register()
    print(f"  registry_id : {trainer.state.registry_id}")

    # Cleanup.
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def _demo_leakage_detection() -> None:
    section("Leakage detection demo")

    # Clean split: no leakage.
    records = make_synthetic_records(n=60, label_options=["x", "y", "z"])
    labels = [r.label for r in records]
    split = split_records(records, labels)
    train, val, test = apply_split(records, split)
    leak = check_leakage(train, val, test)
    print(f"  clean split — leakage_detected: {leak['leakage_detected']}")

    # Deliberate leakage: same document ID in train and validation.
    r1 = TrainingRecord(document_id="dup-1", text="unique text A", label="a")
    r2 = TrainingRecord(document_id="dup-1", text="unique text B", label="b")  # same ID!
    r3 = TrainingRecord(document_id="uniq-2", text="unique text C", label="a")
    leak2 = check_leakage([r1], [r2], [r3])
    print(f"  deliberate ID leakage — leakage_detected: {leak2['leakage_detected']}")
    print(f"    collisions: {leak2['cross_split_id_collisions']}")


def _demo_registry() -> None:
    section("Model registry demo")

    tmpdir = tempfile.mkdtemp()
    registry_path = os.path.join(tmpdir, "registry.json")

    from ml.configs import ModelRegistry

    registry = ModelRegistry(registry_path=registry_path)

    eid1 = registry.register({
        "task": "sentiment",
        "model_name": "tfidf_logistic",
        "model_version": "v1",
        "artifact_path": "ml/artifacts/sentiment__tfidf_logistic__v1.joblib",
        "dataset": "data/annotations/sentiment_annotations.csv",
        "configuration": {"task_name": "sentiment", "random_state": 42},
        "metrics": {"test": {"macro_f1": 0.75, "accuracy": 0.8}},
        "warnings": [],
    })
    eid2 = registry.register({
        "task": "response",
        "model_name": "tfidf_logistic_multiclass",
        "model_version": "v1",
        "artifact_path": "ml/artifacts/response__tfidf_logistic_multiclass__v1.joblib",
        "dataset": "data/annotations/response_annotations.csv",
        "configuration": {"task_name": "response", "random_state": 42},
        "metrics": {"test": {"macro_f1": 0.62}},
        "warnings": [],
    })

    print(f"  registered sentiment entry: {eid1}")
    print(f"  registered response entry: {eid2}")

    entries = registry.list_entries()
    print(f"  total entries: {len(entries)}")

    sentiment_entries = registry.list_entries(task="sentiment")
    print(f"  sentiment entries: {len(sentiment_entries)}")

    latest = registry.find_latest("sentiment", "tfidf_logistic")
    print(f"  latest sentiment entry model: {latest['model_name']} {latest['model_version']}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def _task_display(task: TaskType) -> str:
    mapping = {
        TaskType.SENTIMENT: "single-label sentiment classification",
        TaskType.EMOTION: "multi-label emotion recognition",
        TaskType.ISSUE: "multi-label issue identification",
        TaskType.RESPONSE: "single-label response role classification",
        TaskType.SOLUTION: "binary solution detection",
    }
    return mapping.get(task, task.value)


if __name__ == "__main__":
    main()
