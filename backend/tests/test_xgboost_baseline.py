"""Sprint P1-C Unit Tests — XGBoost Baseline (Sparse TF-IDF, SVD+Linguistic, Save/Load, Graceful Fallback)."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


def test_graceful_message_when_xgboost_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.models import xgboost_baseline as xb

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated missing xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as exc_info:
        xb.require_xgboost()
    assert "requirements-ml.txt" in str(exc_info.value)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost optional in base profile; tested in test-ml-optional CI")
@pytest.mark.parametrize("feature_mode", ["sparse_tfidf", "svd_linguistic"])
def test_xgboost_multiclass_binary_multilabel_and_serialization(
    tmp_path: Path, feature_mode: str
) -> None:
    from ml.models.xgboost_baseline import XGBoostBaselineClassifier, XGBoostConfig

    cfg = XGBoostConfig(
        feature_mode=feature_mode,  # type: ignore[arg-type]
        n_estimators=25,
        max_depth=3,
        svd_components=8,
        random_state=42,
        n_jobs=1,
    )

    # 1. Multiclass (6 canonical response classes)
    resp_classes = ["agreement", "personal_experience", "opinion", "advice", "solution", "counterargument"]
    texts_mc = [
        "Same bro agreed totally",
        "In my batch last year placements were decent",
        "I think ROI is moderate for freshers",
        "You should talk to seniors before joining",
        "Email accounts desk with your sanction letter for 15 day extension",
        "Actually that figure is outdated and false",
    ] * 4
    y_mc = resp_classes * 4
    clf_mc = XGBoostBaselineClassifier(task_name="response", multilabel=False, config=cfg)
    clf_mc.fit(texts_mc[:18], y_mc[:18], texts_mc[18:], y_mc[18:], resp_classes)

    feats = clf_mc.transform_features(texts_mc[:4])
    if feature_mode == "sparse_tfidf":
        assert sparse.issparse(feats), "sparse_tfidf must NEVER densify CSR TF-IDF matrix!"
    probs_mc = clf_mc.predict_proba(texts_mc[:4])
    assert probs_mc.shape == (4, 6)
    assert np.all((probs_mc >= 0.0) & (probs_mc <= 1.0))
    assert np.allclose(probs_mc.sum(axis=1), 1.0, atol=1e-5)

    # Save / load round-trip equivalence
    mc_dir = tmp_path / f"xgb_mc_{feature_mode}"
    size_bytes = clf_mc.save(mc_dir)
    assert size_bytes > 0
    reloaded = XGBoostBaselineClassifier.load(mc_dir)
    assert np.allclose(probs_mc, reloaded.predict_proba(texts_mc[:4]), atol=1e-6)
    assert clf_mc.predict(texts_mc[:4]) == reloaded.predict(texts_mc[:4])

    # 2. Binary (solution)
    sol_texts = [
        "Submit AICTE refund form at portal for 15 day refund",
        "Mail admissions office with bank sanction letter",
        "Waiting for waitlist update",
        "Same bro",
    ] * 5
    sol_y = ["solution", "solution", "not_solution", "not_solution"] * 5
    clf_bin = XGBoostBaselineClassifier(task_name="solution", multilabel=False, config=cfg)
    clf_bin.fit(sol_texts[:16], sol_y[:16], sol_texts[16:], sol_y[16:], ["not_solution", "solution"])
    probs_bin = clf_bin.predict_proba(sol_texts[:4])
    assert probs_bin.shape == (4, 2)
    assert np.allclose(probs_bin.sum(axis=1), 1.0, atol=1e-5)

    # 3. Multi-label (emotion: 1 binary classifier per label)
    emo_classes = ["confidence", "curiosity", "frustration", "gratitude", "hope", "stress"]
    emo_texts = [
        "Frustrated and stressed with late circulars",
        "Grateful and hopeful for the new hostel",
        "Curious about internship dates",
        "Confident you will convert easily",
    ] * 5
    emo_y = [
        ["frustration", "stress"],
        ["gratitude", "hope"],
        ["curiosity"],
        ["confidence"],
    ] * 5
    clf_ml = XGBoostBaselineClassifier(task_name="emotion", multilabel=True, config=cfg)
    clf_ml.fit(emo_texts[:16], emo_y[:16], emo_texts[16:], emo_y[16:], emo_classes)
    probs_ml = clf_ml.predict_proba(emo_texts[:4])
    assert probs_ml.shape == (4, 6)
    assert np.all((probs_ml >= 0.0) & (probs_ml <= 1.0))
    preds_ml = clf_ml.predict(emo_texts[:4])
    assert len(preds_ml) == 4 and all(isinstance(x, list) and len(x) >= 1 for x in preds_ml)
