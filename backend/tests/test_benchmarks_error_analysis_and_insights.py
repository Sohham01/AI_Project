"""Unit & integration tests for 4-architecture baselines, error analysis,
benchmark verification, calibrated confidence, and cross-model insight engine.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest

from app.api import demo
from app.services.nlp.cross_model_insights import (
    build_confidence_summary,
    build_cross_model_insights,
    build_issue_priority_matrix,
    comment_confidence_profile,
    confidence_tier,
)
from ml.evaluation.benchmark_verifier import verify_benchmark_artifact
from ml.evaluation.error_analysis import (
    analyze_architecture_recoveries_and_errors,
    classify_confidence_tier,
    compute_calibration_metrics,
    compute_confusion_and_top_pairs,
    probability_margin,
)
from ml.models.neural_baselines import LogisticRegressionBaseline


def test_confidence_tiers_and_margin() -> None:
    assert classify_confidence_tier(0.92) == "high"
    assert classify_confidence_tier(0.67) == "medium"
    assert classify_confidence_tier(0.41) == "low"
    assert confidence_tier(0.80) == "high"
    assert confidence_tier(0.55) == "medium"
    assert confidence_tier(0.54) == "low"
    assert probability_margin([0.72, 0.20, 0.08]) == pytest.approx(0.52, abs=1e-4)
    assert probability_margin([0.90]) == pytest.approx(0.90, abs=1e-4)


def test_calibration_and_confusion_single_and_multilabel() -> None:
    labels = ["negative", "neutral", "positive"]
    y_true = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    probs = np.array([
        [0.05, 0.10, 0.85],
        [0.80, 0.15, 0.05],
        [0.20, 0.65, 0.15],
        [0.55, 0.25, 0.20],  # wrong: predicted negative
        [0.75, 0.15, 0.10],
        [0.10, 0.82, 0.08],
    ])
    cal = compute_calibration_metrics(y_true, probs, labels, multilabel=False)
    assert 0.0 <= cal["ece"] <= 1.0
    assert 0.0 <= cal["brier_score"] <= 2.0
    assert len(cal["bins"]) == 8
    assert cal["confidence_tiers"]["high"]["count"] == 3

    y_pred = [labels[int(np.argmax(r))] for r in probs]
    cm = compute_confusion_and_top_pairs(y_true, y_pred, labels, multilabel=False)
    assert cm["error_count"] == 1
    assert cm["top_confusions"][0]["true_label"] == "positive"
    assert cm["top_confusions"][0]["predicted_label"] == "negative"

    # Multi-label path
    ml_labels = ["a", "b", "c"]
    ml_true = [["a"], ["b", "c"], ["a", "c"], ["b"]]
    ml_probs = np.array([
        [0.88, 0.12, 0.10],
        [0.10, 0.82, 0.78],
        [0.75, 0.60, 0.20],
        [0.15, 0.80, 0.10],
    ])
    ml_cal = compute_calibration_metrics(ml_true, ml_probs, ml_labels, multilabel=True)
    assert 0.0 <= ml_cal["ece"] <= 1.0
    ml_cm = compute_confusion_and_top_pairs(ml_true, [["a"], ["b", "c"], ["a", "b"], ["b"]], ml_labels, multilabel=True)
    assert ml_cm["multilabel"] is True
    assert ml_cm["error_count"] == 1


def test_architecture_recovery_and_error_analysis() -> None:
    labels = ["negative", "neutral", "positive"]
    texts = ["bhai placements bilkul solid hai", "totally ambiguous reply"]
    cats = ["code_mixed_language", "ambiguous_responses"]
    y_true = ["positive", "neutral"]
    preds = {
        "logistic_regression": ["neutral", "negative"],
        "ann_mlp": ["neutral", "negative"],
        "cnn_textcnn": ["negative", "negative"],
        "transformer_muril_dapt_v3": ["positive", "negative"],
    }
    probs = {
        k: np.array([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]) for k in preds
    }
    rep = analyze_architecture_recoveries_and_errors(
        task_name="sentiment",
        probe_texts=texts,
        probe_categories=cats,
        y_true=y_true,
        model_preds_by_arch=preds,
        model_probs_by_arch=probs,
        labels=labels,
    )
    assert rep["agreement_summary"]["transformer_only_win"] == 1
    assert rep["agreement_summary"]["all_failed"] == 1
    assert len(rep["transformer_wins"]) == 1
    assert rep["transformer_wins"][0]["baselines_failed"] == 3
    assert len(rep["residual_errors"]) == 1


def test_logreg_baseline_runs_without_torch() -> None:
    texts = ["great placement offer", "bad hostel mess food", "fee deadline info"] * 8
    ys = ["positive", "negative", "neutral"] * 8
    lbls = ["negative", "neutral", "positive"]
    clf = LogisticRegressionBaseline(multilabel=False, random_state=7)
    clf.fit(texts, ys, texts[:6], ys[:6], lbls)
    probs = clf.predict_proba(texts[:4])
    assert probs.shape == (4, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert clf.param_count_ > 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch optional in CPU-only CI")
def test_ann_and_cnn_baselines_smoke() -> None:
    from ml.models.neural_baselines import ANNMLPClassifier, NeuralBaselineConfig, TextCNNClassifier

    texts = ["great placement offer", "bad hostel mess food", "fee deadline info"] * 8
    ys = ["positive", "negative", "neutral"] * 8
    lbls = ["negative", "neutral", "positive"]
    cfg = NeuralBaselineConfig(epochs=2, batch_size=8, random_state=7, device="cpu")
    for model_cls in (ANNMLPClassifier, TextCNNClassifier):
        clf = model_cls(multilabel=False, config=cfg)
        clf.fit(texts, ys, texts[:6], ys[:6], lbls)
        probs = clf.predict_proba(texts[:4])
        assert probs.shape == (4, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
        assert clf.param_count_ > 0


def test_benchmark_verifier_catches_leakage_and_passes_valid_live_artifacts() -> None:
    demo._data.cache_clear()
    data = demo._data()
    ev_raw = data["evaluation_raw"]
    if ev_raw.get("architecture_comparison"):
        rep = verify_benchmark_artifact(ev_raw)
        assert rep["verified"] is True, rep

    # Inject synthetic leakage -> verifier must fail zero_split_leakage
    bad = {
        "tasks": {
            "sentiment": {"leakage_report": {"leakage_detected": True}, "split": {"group_by": "random"}}
        },
        "contamination_audit": {"exact_probe_overlaps": 1},
    }
    bad_rep = verify_benchmark_artifact(bad)
    assert bad_rep["verified"] is False
    failed_ids = {c["id"] for c in bad_rep["checks"] if not c["passed"]}
    assert "zero_split_leakage" in failed_ids
    assert "zero_probe_contamination" in failed_ids


def test_live_api_confidence_insights_and_error_analysis_endpoints() -> None:
    demo._data.cache_clear()
    summary = demo.summary()
    assert "confidence_summary" in summary
    assert 0.0 <= summary["confidence_summary"]["mean_confidence"] <= 1.0
    assert summary["issue_priority_matrix"]
    assert summary["cross_model_insights"]["cards"]

    ins = demo.insights()
    demo.InsightsResponse.model_validate(ins)

    ea = demo.error_analysis()
    demo.ErrorAnalysisResponse.model_validate(ea)

    # Confidence filter validation
    high_threads = demo.discussions("", "", "", "", "", "", 1, 12, "high")
    assert all(item["confidence_tier"] == "high" for item in high_threads["items"])
    review_threads = demo.discussions("", "", "", "", "", "", 1, 12, "review")
    assert all(item["low_confidence_comments"] > 0 for item in review_threads["items"])
