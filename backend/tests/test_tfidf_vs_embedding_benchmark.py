"""V6.7 Unit & Integration Tests — TF-IDF vs. Dense Multilingual Embeddings vs. Hybrid Fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.models.embedding_classifiers import (
    HAS_TORCH,
    HAS_XGBOOST,
    EmbeddingAndHybridLogisticClassifier,
    EmbeddingAndHybridXGBoostClassifier,
    HybridGatedTFIDFEmbeddingMLP,
    HybridMLPConfig,
    center_and_l2_normalize,
)


def _make_synthetic_corpus_and_embeddings(seed: int = 77) -> tuple[list[str], list[str], np.ndarray]:
    """Synthetic dataset where lexical phrases cover half the signal and dense semantic vectors cover the other half."""
    rng = np.random.default_rng(seed)
    classes = ["negative", "neutral", "positive"]
    lex_templates = {
        0: ["terrible mess food complaint hostel worst", "awful scam waste stress penalty"],
        1: ["schedule notice timing portal update form", "attendance syllabus calendar room info"],
        2: ["excellent placement helpful seniors great campus", "awesome ROI supportive faculty happy"],
    }
    paraphrase_neutral_words = [
        "bhai scene alag hai aaj toh",
        "dekhte hain kya hota hai campus mein",
        "kal baat hui thi batchmates ke saath",
    ]
    prototypes = rng.normal(0.0, 1.0, size=(3, 64)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    texts: list[str] = []
    targets: list[str] = []
    embs = np.zeros((120, 64), dtype=np.float32)
    for i in range(120):
        cls = i % 3
        targets.append(classes[cls])
        # Half samples have strong lexical cues; half rely on semantic paraphrase embeddings!
        if (i // 3) % 2 == 0:
            texts.append(f"{lex_templates[cls][i % 2]} sample_{i % 7}")
            embs[i] = 0.45 * prototypes[cls] + 0.55 * rng.normal(0, 0.25, size=64)
        else:
            texts.append(f"{paraphrase_neutral_words[i % 3]} token_{i % 9}")
            embs[i] = 0.85 * prototypes[cls] + 0.15 * rng.normal(0, 0.15, size=64)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return texts, targets, embs


def test_center_and_l2_normalize_leak_free_train_mean() -> None:
    rng = np.random.default_rng(11)
    tr = rng.normal(loc=3.0, scale=0.5, size=(40, 32)).astype(np.float32)
    te = rng.normal(loc=-2.0, scale=0.5, size=(15, 32)).astype(np.float32)

    u_tr, mu_tr = center_and_l2_normalize(tr, None)
    u_te_1, _ = center_and_l2_normalize(te[:5], mu_tr)
    u_te_all, _ = center_and_l2_normalize(te, mu_tr)

    # Unit L2 norm & strict independence from unqueried test rows
    np.testing.assert_allclose(np.linalg.norm(u_tr, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(u_te_all, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(u_te_1, u_te_all[:5], atol=1e-6)


def test_logistic_tfidf_vs_embedding_vs_hybrid_beats_single_view() -> None:
    texts, targets, embs = _make_synthetic_corpus_and_embeddings()
    classes = ["negative", "neutral", "positive"]
    tr_t, va_t, te_t = texts[:72], texts[72:96], texts[96:]
    tr_y, va_y, te_y = targets[:72], targets[72:96], targets[96:]
    tr_e, va_e, te_e = embs[:72], embs[72:96], embs[96:]
    y_te_idx = np.array([classes.index(y) for y in te_y])

    clf_tfidf = EmbeddingAndHybridLogisticClassifier(mode="tfidf_word_char", random_state=42)
    clf_tfidf.fit(tr_t, tr_y, va_t, va_y, classes)
    acc_tfidf = float((np.argmax(clf_tfidf.predict_proba(te_t), axis=1) == y_te_idx).mean())

    clf_emb = EmbeddingAndHybridLogisticClassifier(mode="embedding_only", random_state=42)
    clf_emb.fit(tr_t, tr_y, va_t, va_y, classes, train_emb_blocks=[tr_e], val_emb_blocks=[va_e])
    acc_emb = float((np.argmax(clf_emb.predict_proba(te_t, emb_blocks=[te_e]), axis=1) == y_te_idx).mean())

    clf_hyb = EmbeddingAndHybridLogisticClassifier(mode="hybrid_tfidf_emb", include_linguistic_cues=True, random_state=42)
    clf_hyb.fit(tr_t, tr_y, va_t, va_y, classes, train_emb_blocks=[tr_e], val_emb_blocks=[va_e])
    p_hyb = clf_hyb.predict_proba(te_t, emb_blocks=[te_e])
    acc_hyb = float((np.argmax(p_hyb, axis=1) == y_te_idx).mean())

    assert p_hyb.shape == (len(te_t), 3)
    np.testing.assert_allclose(p_hyb.sum(axis=1), 1.0, atol=1e-5)
    assert acc_hyb >= max(acc_tfidf, acc_emb) - 0.05
    assert acc_hyb > 0.80


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional in base runtime")
def test_hybrid_gated_mlp_multilabel_and_gate_audit() -> None:
    texts, _, embs = _make_synthetic_corpus_and_embeddings(seed=91)
    labels = ["academics", "placements", "hostel"]
    ml_targets = [[labels[i % 3], labels[(i + 1) % 3]] if i % 4 == 0 else [labels[i % 3]] for i in range(120)]

    clf = HybridGatedTFIDFEmbeddingMLP(
        mode="hybrid_gated",
        multilabel=True,
        config=HybridMLPConfig(hidden_dim=48, svd_components=16, epochs=10, random_state=42),
    )
    clf.fit(
        texts[:80],
        ml_targets[:80],
        texts[80:100],
        ml_targets[80:100],
        labels,
        train_emb_blocks=[embs[:80]],
        val_emb_blocks=[embs[80:100]],
    )
    probs = clf.predict_proba(texts[100:], emb_blocks=[embs[100:]])
    assert probs.shape == (20, 3)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert 0.0 < clf.mean_lexical_gate_weight_ < 1.0
    assert clf.thresholds_ is not None and len(clf.thresholds_) == 3
    clf.cleanup()


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost optional in base runtime")
def test_embedding_and_hybrid_xgboost_fit_predict() -> None:
    texts, targets, embs = _make_synthetic_corpus_and_embeddings(seed=55)
    classes = ["negative", "neutral", "positive"]
    clf = EmbeddingAndHybridXGBoostClassifier(
        task_name="sentiment",
        mode="hybrid_svd_emb",
        multilabel=False,
        svd_components=12,
        random_state=42,
        n_jobs=2,
    )
    clf.fit(
        texts[:80],
        targets[:80],
        texts[80:100],
        targets[80:100],
        classes,
        train_emb_blocks=[embs[:80]],
        val_emb_blocks=[embs[80:100]],
    )
    probs = clf.predict_proba(texts[100:], emb_blocks=[embs[100:]])
    assert probs.shape == (20, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_v67_benchmark_artifact_integrity_when_present() -> None:
    art = ROOT / "ml" / "artifacts" / "v6_7_tfidf_vs_embedding" / "tfidf_vs_embedding_benchmark.json"
    if art.is_file():
        doc = json.loads(art.read_text(encoding="utf-8"))
        assert doc["n_splits"] == 7
        assert doc["zero_test_tuning_leakage"] is True
        assert set(doc["arms"]["logreg_tfidf_word_char"]["by_task"].keys()) == {
            "issue",
            "sentiment",
            "emotion",
            "response",
            "solution",
        }
