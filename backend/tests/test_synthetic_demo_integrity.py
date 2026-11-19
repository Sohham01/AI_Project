from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.api.demo import RESPONSE_TYPES, _data
from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES as CANONICAL_RESPONSE_TYPES
from app.services.nlp.solution_detection.models.labels import TARGET_LABELS
from generate_synthetic_demo import build
from ml.data.dataset import TrainingRecord, load_annotation_dataset
from ml.data.splitting import SplitSpec, apply_split, split_records, extract_groups
from train_demo_baselines import (
    heldout_class_metrics,
    normalize_solution_label,
    verify_task_evaluation_integrity,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "demo_synthetic"
FROZEN_V1_ARTIFACTS = ROOT / "ml" / "artifacts" / "demo_synthetic"
ARTIFACTS_V2 = ROOT / "ml" / "artifacts" / "demo_synthetic_v2"


def test_synthetic_generation_is_seed_reproducible() -> None:
    first = build()
    second = build()
    assert first["posts"] == second["posts"]
    assert first["metadata"]["generation_seed"] == second["metadata"]["generation_seed"] == 20261116


def test_corpus_integrity_and_frozen_response_taxonomy() -> None:
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    metadata = corpus["metadata"]
    posts = corpus["posts"]
    comments = [comment for post in posts for comment in post["comments"]]
    post_ids = [post["reddit_id"] for post in posts]
    comment_ids = [comment["reddit_id"] for comment in comments]
    assert metadata["synthetic"] is True
    assert metadata["authorization_status"] == "synthetic_only"
    assert metadata["post_count"] == len(posts) == 160
    assert metadata["comment_count"] == len(comments) == 1280
    assert len(post_ids) == len(set(post_ids))
    assert len(comment_ids) == len(set(comment_ids))
    assert all(post["synthetic"] and len(post["comments"]) == 8 for post in posts)
    for post in posts:
        local_ids = {comment["reddit_id"] for comment in post["comments"]}
        for comment in post["comments"]:
            parent_id = comment["parent_reddit_id"]
            assert parent_id is None or parent_id in local_ids
            assert comment["synthetic"] is True
    assert tuple(RESPONSE_TYPES) == tuple(CANONICAL_RESPONSE_TYPES)
    observed_roles = {comment["response_type"] for comment in comments}
    assert observed_roles == set(CANONICAL_RESPONSE_TYPES)
    assert sum(metadata["label_counts"]["response_type"].values()) == 1280


def test_frozen_v1_artifacts_remain_untouched() -> None:
    assert FROZEN_V1_ARTIFACTS.is_dir()
    v1_eval = json.loads((FROZEN_V1_ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    assert v1_eval["synthetic_only"] is True
    assert set(v1_eval["tasks"]) == {"sentiment", "emotion", "issue", "response", "solution"}


def test_training_artifacts_are_isolated_and_group_leakage_free() -> None:
    evaluation = json.loads((ARTIFACTS_V2 / "evaluation.json").read_text(encoding="utf-8"))
    predictions = json.loads((ARTIFACTS_V2 / "predictions.json").read_text(encoding="utf-8"))["tasks"]
    assert evaluation["synthetic_only"] is True
    assert set(evaluation["tasks"]) == {"sentiment", "emotion", "issue", "response", "solution"}

    for task_name, task in evaluation["tasks"].items():
        assert task["split"]["group_by"] == "post_id"
        assert (task["split"]["train"], task["split"]["validation"], task["split"]["test"]) == (896, 192, 192)
        assert task["leakage_report"]["leakage_detected"] is False
        assert Path(task["artifact_path"]).as_posix().startswith("ml/artifacts/demo_synthetic_v2/")

        # Reconstruct exact thread-grouped splits from CSV and verify zero post_id overlap
        records = load_annotation_dataset(task_name, str(DATA / f"{task_name}.csv"))
        groups = extract_groups(records, "post_id")
        split = split_records(
            records,
            [r.label for r in records],
            SplitSpec(train_size=0.70, validation_size=0.15, test_size=0.15, random_state=20261116, stratify=True),
            groups=groups,
        )
        train_rows, val_rows, test_rows = apply_split(records, split)
        train_posts = {r.post_id for r in train_rows}
        val_posts = {r.post_id for r in val_rows}
        test_posts = {r.post_id for r in test_rows}
        assert len(train_posts) == 112 and len(val_posts) == 24 and len(test_posts) == 24
        assert not (train_posts & val_posts)
        assert not (train_posts & test_posts)
        assert not (val_posts & test_posts)

        verify_task_evaluation_integrity(
            task_name,
            task,
            train_rows,
            val_rows,
            test_rows,
            predictions[task_name],
        )

    for task in _data()["evaluation"].values():
        assert Path(task["artifact_path"]).as_posix().startswith("ml/artifacts/demo_synthetic_v2/")


def test_solution_label_normalization_and_confusion_matrix_integrity() -> None:
    evaluation = json.loads((ARTIFACTS_V2 / "evaluation.json").read_text(encoding="utf-8"))
    sol = evaluation["tasks"]["solution"]
    assert sol["labels"] == list(TARGET_LABELS) == ["not_solution", "solution"]
    assert sol["label_counts"] == {"not_solution": 854, "solution": 426}

    tm = sol["test"]["metrics"]
    assert tm["n_samples"] == 192
    assert tm["confusion_matrix_labels"] == ["not_solution", "solution"]
    cm = tm["confusion_matrix"]
    assert cm == [[127, 0], [0, 65]]
    assert sum(sum(row) for row in cm) == tm["n_samples"] == 192

    per_class = tm["per_class"]
    assert list(per_class.keys()) == ["not_solution", "solution"]
    assert per_class["not_solution"]["support"] == 127
    assert per_class["solution"]["support"] == 65
    assert per_class["not_solution"]["support"] + per_class["solution"]["support"] == 192
    assert per_class["not_solution"]["support"] > 0 and per_class["solution"]["support"] > 0

    # Normalization contract
    assert normalize_solution_label(True) == "solution"
    assert normalize_solution_label(False) == "not_solution"
    assert normalize_solution_label("true") == "solution"
    assert normalize_solution_label("false") == "not_solution"
    assert normalize_solution_label("solution") == "solution"
    assert normalize_solution_label("not_solution") == "not_solution"
    with pytest.raises(ValueError):
        normalize_solution_label("maybe")
    with pytest.raises(ValueError):
        normalize_solution_label("")


def test_empty_classes_cannot_produce_misleading_perfect_metrics() -> None:
    # Construct test records where only 'not_solution' is present in gold and predictions
    records = [
        TrainingRecord(document_id=f"doc-{i}", text="no solution here", label="not_solution", post_id="p1")
        for i in range(10)
    ]
    preds = {f"doc-{i}": {"is_solution": False} for i in range(10)}
    per_class, cm = heldout_class_metrics("solution", records, preds, ["not_solution", "solution"])
    assert cm == [[10, 0], [0, 0]]
    assert per_class["not_solution"]["support"] == 10
    assert per_class["not_solution"]["f1"] == 1.0
    # Empty class 'solution' must have support=0 and None metrics, never 1.0
    assert per_class["solution"]["support"] == 0
    assert per_class["solution"]["precision"] is None
    assert per_class["solution"]["recall"] is None
    assert per_class["solution"]["f1"] is None


def test_multilabel_tasks_report_subset_accuracy_and_hamming_loss_only() -> None:
    evaluation = json.loads((ARTIFACTS_V2 / "evaluation.json").read_text(encoding="utf-8"))
    for multi_task in ("emotion", "issue"):
        tm = evaluation["tasks"][multi_task]["test"]["metrics"]
        vm = evaluation["tasks"][multi_task]["validation"]["metrics"]
        for m in (tm, vm):
            assert m["multilabel"] is True
            assert m["accuracy"] is None
            assert m["confusion_matrix"] is None
            assert m["per_class"] == {}
            assert m["subset_accuracy"] is not None and 0.0 < m["subset_accuracy"] <= 1.0
            assert m["hamming_loss"] is not None and 0.0 <= m["hamming_loss"] < 1.0
            assert m["macro_f1"] is not None and 0.0 < m["macro_f1"] <= 1.0
            assert m["micro_f1"] is not None and 0.0 < m["micro_f1"] <= 1.0
            assert set(m["per_label"].keys()) == set(evaluation["tasks"][multi_task]["labels"])

    for single_task in ("sentiment", "response", "solution"):
        tm = evaluation["tasks"][single_task]["test"]["metrics"]
        assert tm["multilabel"] is False
        assert tm["accuracy"] is not None
        assert tm["subset_accuracy"] is None
        assert tm["hamming_loss"] is None
        assert tm["hamming_accuracy"] is None
        assert tm["per_label"] == {}
