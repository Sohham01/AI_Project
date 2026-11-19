"""Train existing TF-IDF baseline models on the isolated synthetic demo corpus.

This is an explicit command, never run during application startup. It uses the
project Trainer, group-aware post_id splits, and an isolated prototype registry.
"""
from __future__ import annotations

import argparse
import json
import sys
import csv
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ml.configs import ModelRegistry, TaskConfig, TaskType
from ml.data.splitting import SplitSpec, apply_split, check_leakage
from ml.training.trainer import Trainer, records_to_annotations
from app.services.nlp.solution_detection.models.labels import (
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    TARGET_LABELS,
    label_to_bool,
    parse_bool_cell,
    target_label,
)
from app.services.nlp.topic_discovery.base import TopicDocument
from app.services.nlp.topic_discovery.nmf_topic_discoverer import NMFTopicDiscoverer

DATA = ROOT / "data" / "demo_synthetic"
FROZEN_V1_ARTIFACTS = ROOT / "ml" / "artifacts" / "demo_synthetic"
ARTIFACTS = ROOT / "ml" / "artifacts" / "demo_synthetic_v2"
TASKS = ("sentiment", "emotion", "issue", "response", "solution")
SEED = 20261116


from fastapi.encoders import jsonable_encoder


def jsonable(value: Any) -> Any:
    return jsonable_encoder(value)


def normalize_solution_label(value: Any, document_id: str | None = None) -> str:
    """Normalize any boolean or canonical solution target to 'not_solution' or 'solution'.

    Refuses blank or unrecognised values rather than silently coercing to 'not_solution'.
    """
    if isinstance(value, bool):
        return target_label(value)
    if isinstance(value, str):
        canonical = label_to_bool(value)
        if canonical is not None:
            return target_label(canonical)
    parsed = parse_bool_cell(value, document_id)
    return target_label(parsed)


def label_set(value: Any, task_name: str | None = None) -> set[str]:
    if task_name == "solution":
        if value is None or (isinstance(value, str) and not value.strip()):
            return set()
        if isinstance(value, (list, tuple, set)):
            return {normalize_solution_label(item) for item in value if item is not None and str(item).strip()}
        return {normalize_solution_label(value)}
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(",", "|").split("|")
    return {str(item).strip().lower() for item in values if str(item).strip()}


def heldout_class_metrics(
    task_name: str,
    records: list[Any],
    predictions: dict[str, dict[str, Any]],
    labels: list[str],
) -> tuple[dict[str, Any], list[list[int]] | None]:
    """Compute class-level held-out metrics from actual model outputs and canonical labels."""
    fields = {
        "sentiment": ("sentiment", False),
        "emotion": ("emotions", True),
        "issue": ("predicted_issue_labels", True),
        "response": ("response_type", False),
        "solution": ("is_solution", False),
    }
    field, multilabel = fields[task_name]
    ordered_labels = list(TARGET_LABELS) if task_name == "solution" else list(labels)
    normalized_labels = [str(label).strip().lower() for label in ordered_labels]
    allowed_vocab = set(normalized_labels)

    gold_sets: list[set[str]] = []
    predicted_sets: list[set[str]] = []
    for record in records:
        gold = label_set(record.label, task_name=task_name)
        unknown_gold = gold - allowed_vocab
        if unknown_gold:
            raise ValueError(
                f"Task {task_name!r} gold record {record.document_id!r} contains label(s) "
                f"{sorted(unknown_gold)} outside evaluation vocabulary {ordered_labels}"
            )
        pred = predictions.get(record.document_id, {}).get(field)
        if pred is None:
            pred_values: set[str] = set()
        else:
            pred_values = label_set(pred, task_name=task_name)
        unknown_pred = pred_values - allowed_vocab
        if unknown_pred:
            raise ValueError(
                f"Task {task_name!r} prediction for {record.document_id!r} contains label(s) "
                f"{sorted(unknown_pred)} outside evaluation vocabulary {ordered_labels}"
            )
        gold_sets.append(gold)
        predicted_sets.append(pred_values)

    per_class: dict[str, Any] = {}
    for original, normalized in zip(ordered_labels, normalized_labels):
        tp = sum(normalized in gold and normalized in pred for gold, pred in zip(gold_sets, predicted_sets))
        fp = sum(normalized not in gold and normalized in pred for gold, pred in zip(gold_sets, predicted_sets))
        fn = sum(normalized in gold and normalized not in pred for gold, pred in zip(gold_sets, predicted_sets))
        support = sum(normalized in gold for gold in gold_sets)
        predicted_count = tp + fp
        if support == 0 and predicted_count == 0:
            per_class[original] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
            }
            continue
        precision = round(tp / (tp + fp), 6) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        recall = round(tp / (tp + fn), 6) if (tp + fn) > 0 else (0.0 if predicted_count > 0 else None)
        f1 = (
            round(2 * precision * recall / (precision + recall), 6)
            if (precision is not None and recall is not None and (precision + recall) > 0)
            else (0.0 if (support > 0 or predicted_count > 0) else None)
        )
        per_class[original] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    confusion = None
    if not multilabel:
        index = {label: i for i, label in enumerate(normalized_labels)}
        confusion = [[0 for _ in ordered_labels] for _ in ordered_labels]
        for gold, pred in zip(gold_sets, predicted_sets):
            true_label = next(iter(gold), "")
            predicted_label = next(iter(pred), "")
            if true_label in index and predicted_label in index:
                confusion[index[true_label]][index[predicted_label]] += 1
    return per_class, confusion


def verify_task_evaluation_integrity(
    task_name: str,
    task_report: dict[str, Any],
    train_rows: list[Any],
    validation_rows: list[Any],
    test_rows: list[Any],
    test_predictions: dict[str, dict[str, Any]],
) -> None:
    """Assert scientific and mathematical integrity of a task's evaluation report."""
    labels = list(task_report["labels"])
    if task_name == "solution" and labels != list(TARGET_LABELS):
        raise AssertionError(
            f"Solution task labels must have explicit canonical order {list(TARGET_LABELS)}, got {labels}"
        )

    # 1. Thread-grouped split & zero cross-split leakage
    train_posts = {r.post_id for r in train_rows if r.post_id}
    val_posts = {r.post_id for r in validation_rows if r.post_id}
    test_posts = {r.post_id for r in test_rows if r.post_id}
    if (train_posts & val_posts) or (train_posts & test_posts) or (val_posts & test_posts):
        raise AssertionError(f"Cross-split post_id thread leakage detected in {task_name}")
    if task_report["leakage_report"].get("leakage_detected"):
        raise AssertionError(f"Leakage report flagged leakage for {task_name}")

    # 2. Label vocabulary consistency across dataset, predictions, and evaluation
    allowed_vocab = {str(l).strip().lower() for l in labels}
    dataset_vocab = {str(k).strip().lower() for k in task_report["label_counts"].keys()}
    if not dataset_vocab <= allowed_vocab:
        raise AssertionError(
            f"Task {task_name} dataset label_counts keys {sorted(dataset_vocab)} do not match model labels {labels}"
        )

    field_map = {
        "sentiment": "sentiment",
        "emotion": "emotions",
        "issue": "predicted_issue_labels",
        "response": "response_type",
        "solution": "is_solution",
    }
    pred_field = field_map[task_name]
    observed_gold_test: Counter = Counter()
    for row in test_rows:
        g_set = label_set(row.label, task_name=task_name)
        if not g_set <= allowed_vocab:
            raise AssertionError(f"Test gold label {g_set} outside vocabulary {labels} in {task_name}")
        for g in g_set:
            observed_gold_test[g] += 1
        raw_p = test_predictions.get(row.document_id, {}).get(pred_field)
        if raw_p is not None:
            p_set = label_set(raw_p, task_name=task_name)
            if not p_set <= allowed_vocab:
                raise AssertionError(f"Test prediction {p_set} outside vocabulary {labels} in {task_name}")

    test_metrics = task_report["test"]["metrics"]
    n_test = len(test_rows)
    if test_metrics.get("n_samples") != n_test:
        raise AssertionError(
            f"Task {task_name} n_samples={test_metrics.get('n_samples')} != len(test_rows)={n_test}"
        )

    is_multilabel = task_name in {"emotion", "issue"}
    if bool(test_metrics.get("multilabel")) is not is_multilabel:
        raise AssertionError(f"Task {task_name} multilabel flag mismatch: {test_metrics.get('multilabel')}")

    if is_multilabel:
        # Multi-label integrity checks
        if test_metrics.get("accuracy") is not None:
            raise AssertionError(f"Multi-label task {task_name} must not report single-label accuracy")
        if test_metrics.get("confusion_matrix") is not None:
            raise AssertionError(f"Multi-label task {task_name} must not report single-label confusion_matrix")
        for req_metric in ("subset_accuracy", "hamming_loss", "macro_f1", "micro_f1"):
            val = test_metrics.get(req_metric)
            if val is None or not (0.0 <= float(val) <= 1.0):
                raise AssertionError(f"Multi-label task {task_name} missing valid {req_metric}: {val!r}")
        per_label = test_metrics.get("per_label") or {}
        if set(per_label.keys()) != set(labels):
            raise AssertionError(f"Multi-label task {task_name} per_label keys {list(per_label)} != {labels}")
        for lbl in labels:
            entry = per_label[lbl]
            expected_sup = observed_gold_test[lbl.lower()]
            if entry.get("support") != expected_sup:
                raise AssertionError(
                    f"Task {task_name} label {lbl!r} support={entry.get('support')} != expected={expected_sup}"
                )
            if expected_sup == 0 and (entry.get("f1") == 1.0 or entry.get("recall") == 1.0):
                raise AssertionError(f"Empty label {lbl!r} in {task_name} produced misleading perfect metric: {entry}")
    else:
        # Single-label & binary integrity checks
        for forbidden in ("subset_accuracy", "hamming_loss", "hamming_accuracy"):
            if test_metrics.get(forbidden) is not None:
                raise AssertionError(f"Single-label task {task_name} must not report multi-label {forbidden}")
        for req_metric in ("accuracy", "macro_f1"):
            val = test_metrics.get(req_metric)
            if val is None or not (0.0 <= float(val) <= 1.0):
                raise AssertionError(f"Single-label task {task_name} missing valid {req_metric}: {val!r}")
        per_class = test_metrics.get("per_class") or {}
        if list(per_class.keys()) != labels:
            raise AssertionError(f"Task {task_name} per_class keys {list(per_class)} != canonical {labels}")

        total_support = sum(int(v.get("support", 0)) for v in per_class.values())
        if total_support != n_test:
            raise AssertionError(
                f"Task {task_name} per-class supports sum to {total_support}, expected {n_test}"
            )

        cm = test_metrics.get("confusion_matrix")
        cm_labels = test_metrics.get("confusion_matrix_labels")
        if cm is None or len(cm) != len(labels) or any(len(r) != len(labels) for r in cm):
            raise AssertionError(f"Task {task_name} confusion_matrix has invalid shape: {cm}")
        if cm_labels != labels:
            raise AssertionError(f"Task {task_name} confusion_matrix_labels {cm_labels} != {labels}")

        cm_total = sum(sum(int(c) for c in row) for row in cm)
        if cm_total != n_test:
            raise AssertionError(
                f"Task {task_name} confusion_matrix total count={cm_total} != evaluated sample count={n_test}"
            )

        for idx, lbl in enumerate(labels):
            entry = per_class[lbl]
            expected_sup = observed_gold_test[lbl.lower()]
            row_sum = sum(int(c) for c in cm[idx])
            if entry.get("support") != expected_sup or row_sum != expected_sup:
                raise AssertionError(
                    f"Task {task_name} class {lbl!r} support mismatch: per_class={entry.get('support')}, "
                    f"cm_row_sum={row_sum}, gold_test={expected_sup}"
                )
            if expected_sup > 0 and entry.get("support", 0) <= 0:
                raise AssertionError(f"Class {lbl!r} exists in test data ({expected_sup}) but has 0 support")
            if expected_sup == 0 and (entry.get("f1") == 1.0 or entry.get("recall") == 1.0):
                raise AssertionError(f"Empty class {lbl!r} in {task_name} produced misleading perfect metric: {entry}")


def documents_for(task: TaskType, annotations: list[Any]) -> list[Any]:
    convert = {
        TaskType.SENTIMENT: lambda a: a.to_document(),
        TaskType.EMOTION: lambda a: a.to_document(),
        TaskType.ISSUE: lambda a: a.to_issue_document(),
        TaskType.RESPONSE: lambda a: a.to_response_document(),
        TaskType.SOLUTION: lambda a: a.to_solution_document(),
    }[task]
    return [convert(annotation) for annotation in annotations]


def main(force: bool = False) -> None:
    if not (DATA / "corpus.json").is_file():
        raise SystemExit("Synthetic corpus missing. Run backend/generate_synthetic_demo.py first.")
    if ARTIFACTS.exists() and any(ARTIFACTS.iterdir()) and not force:
        raise SystemExit(f"Refusing to overwrite existing isolated model artifacts in {ARTIFACTS}. Re-run with --force to retrain the synthetic demo baselines.")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    task_reports: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}

    for task_name in TASKS:
        config = TaskConfig(
            task_name=task_name,
            dataset_path=str(DATA / f"{task_name}.csv"),
            artifact_directory=str(ARTIFACTS),
            registry_path=str(ARTIFACTS / "registry.json"),
            group_split_by="post_id",
            random_state=SEED,
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            minimum_training_examples=30,
            include_timestamp=False,
        )
        trainer = Trainer(config)
        dataset = trainer.load_dataset()
        # Existing loader calls any CSV "real". Correct provenance explicitly:
        # these files are produced only by generate_synthetic_demo.py.
        dataset.source = "synthetic"
        split = trainer.split_dataset(spec=SplitSpec(
            train_size=0.70, validation_size=0.15, test_size=0.15,
            random_state=SEED, stratify=True,
        ))
        train_rows, validation_rows, test_rows = apply_split(dataset.records, split)
        trainer.state.leakage_report = check_leakage(train_rows, validation_rows, test_rows)
        if trainer.state.leakage_report.get("leakage_detected"):
            raise RuntimeError(f"Thread split leakage detected for {task_name}")

        trainer.train()
        validation_report = trainer.evaluate_validation()
        test_report = trainer.evaluate_test()
        artifact_path = trainer.save_artifact()
        registry = ModelRegistry(config.registry_path)
        existing = registry.find_latest(task_name, trainer.state.trained_model.name)
        if existing and existing.get("dataset") == config.dataset_path:
            trainer.state.registry_id = existing.get("entry_id")
        else:
            trainer._register()  # Uses the isolated demo_synthetic/registry.json.

        annotations = records_to_annotations(dataset.records, TaskType(task_name))
        inference = trainer.state.trained_model.predict(documents_for(TaskType(task_name), annotations))
        predictions[task_name] = {
            prediction.document_id: jsonable(prediction)
            for prediction in inference.predictions
        }
        test_annotations = records_to_annotations(test_rows, TaskType(task_name))
        test_inference = trainer.state.trained_model.predict(documents_for(TaskType(task_name), test_annotations))
        test_predictions = {
            prediction.document_id: jsonable(prediction)
            for prediction in test_inference.predictions
        }
        class_metrics, confusion = heldout_class_metrics(
            task_name, test_rows, test_predictions, trainer.state.trained_model.labels
        )
        serialized_test = jsonable(test_report)
        metric_payload = serialized_test.setdefault("metrics", {})
        if task_name in {"emotion", "issue"}:
            metric_payload["per_label"] = class_metrics
        else:
            metric_payload["per_class"] = class_metrics
            metric_payload["confusion_matrix"] = confusion
            metric_payload["confusion_matrix_labels"] = trainer.state.trained_model.labels
        task_reports[task_name] = {
            "synthetic_only": True,
            "model_name": trainer.state.trained_model.name,
            "model_version": trainer.state.trained_model.version,
            "labels": trainer.state.trained_model.labels,
            "dataset_records": dataset.n_records,
            "label_counts": dataset.label_counts,
            "split": {
                "seed": split.random_state,
                "group_by": "post_id",
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
                "stratify_used": split.stratify_used,
            },
            "leakage_report": jsonable(trainer.state.leakage_report),
            "validation": jsonable(validation_report),
            "test": serialized_test,
            "artifact_path": str(Path(artifact_path).relative_to(ROOT)).replace("\\", "/"),
            "registry_id": trainer.state.registry_id,
            "warnings": trainer.state.all_warnings(),
            "inference_scope": "in-sample predictions for synthetic demo display; evaluation reports use held-out, thread-grouped splits",
        }
        verify_task_evaluation_integrity(
            task_name,
            task_reports[task_name],
            train_rows,
            validation_rows,
            test_rows,
            test_predictions,
        )
        print(f"trained {task_name}: train={len(split.train)}, validation={len(split.validation)}, test={len(split.test)}, leakage={trainer.state.leakage_report.get('leakage_detected')}")

    topic_documents = []
    with (DATA / "response.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            topic_documents.append(TopicDocument(
                document_id=row["document_id"], text=row["text"],
                post_id=row["post_id"], comment_id=row["comment_id"],
                original_text=row["text"], language=row["language"],
                script=row["script"], is_code_mixed=row["is_code_mixed"].lower() == "true",
                detected_languages=[v for v in row["detected_languages"].split("|") if v],
            ))
    topic_result = NMFTopicDiscoverer(n_topics=8, n_keywords=8).discover(topic_documents)
    (ARTIFACTS / "topic_discovery.json").write_text(
        json.dumps({"synthetic_only": True, "dataset_version": "1.0.0", "result": jsonable(topic_result)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (ARTIFACTS / "evaluation.json").write_text(
        json.dumps({"synthetic_only": True, "dataset_version": "1.0.0", "tasks": task_reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "predictions.json").write_text(
        json.dumps({"synthetic_only": True, "dataset_version": "1.0.0", "tasks": predictions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite isolated artifacts under ml/artifacts/demo_synthetic")
    main(force=parser.parse_args().force)
