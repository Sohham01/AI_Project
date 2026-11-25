"""Unit & Integration Tests for V6.4 PyTorch BiGRU-Attention and Keras 3 Multi-Task Models."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_KERAS_TORCH = HAS_TORCH and importlib.util.find_spec("keras") is not None


def test_graceful_error_when_torch_or_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.models import keras_multitask as km
    from ml.models import pytorch_bigru_attention as pb

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"torch", "keras"}:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as exc_t:
        pb.require_torch()
    assert "requirements-ml.txt" in str(exc_t.value)

    with pytest.raises(RuntimeError) as exc_k:
        km.require_keras_torch()
    assert "requirements-ml.txt" in str(exc_k.value)


@pytest.mark.skipif(not HAS_TORCH, reason="torch optional in lightweight base environment")
def test_pytorch_bigru_attention_multiclass_multilabel_attention_and_persistence(tmp_path: Path) -> None:
    from ml.models.pytorch_bigru_attention import BiGRUAttentionClassifier, BiGRUAttentionConfig

    cfg = BiGRUAttentionConfig(embed_dim=32, hidden_dim=24, attn_dim=16, epochs=6, batch_size=8, random_state=42)
    classes = ["agreement", "personal_experience", "opinion", "advice", "solution", "counterargument"]
    texts = [
        "Same bro agreed totally",
        "In my summer internship last year placements were decent",
        "I think ROI is moderate for freshers",
        "You should talk to seniors before paying fees",
        "Email admissions office with sanction letter for refund",
        "Actually that figure is misleading and false",
    ] * 4
    y = classes * 4

    clf = BiGRUAttentionClassifier(task_name="response", multilabel=False, config=cfg)
    clf.fit(texts[:18], y[:18], texts[18:], y[18:], classes)

    probs = clf.predict_proba(texts[:4])
    assert probs.shape == (4, 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    toks, attn = clf.get_attention_weights(texts[:3])
    assert toks.shape == attn.shape
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-4)
    # Strict mask check: <PAD> positions (tok_id == 0) must receive zero attention weight
    assert np.allclose(attn[toks == 0], 0.0, atol=1e-6)

    save_dir = tmp_path / "bigru_resp"
    assert clf.save(save_dir) > 0
    reloaded = BiGRUAttentionClassifier.load(save_dir)
    assert np.allclose(probs, reloaded.predict_proba(texts[:4]), atol=1e-5)
    assert clf.predict(texts[:4]) == reloaded.predict(texts[:4])
    clf.cleanup()
    reloaded.cleanup()


@pytest.mark.skipif(not HAS_KERAS_TORCH, reason="keras+torch optional in lightweight base environment")
def test_keras_multitask_functional_five_heads_and_roundtrip(tmp_path: Path) -> None:
    from ml.models.keras_multitask import KerasMultiTaskClassifier, KerasMultiTaskConfig

    canonical = {
        "issue": ["academic_workload", "administrative_communication", "campus_infrastructure", "curriculum_teaching", "extracurriculars", "financial_concerns", "hostel_facilities", "placements"],
        "sentiment": ["negative", "neutral", "positive"],
        "emotion": ["confidence", "curiosity", "frustration", "gratitude", "hope", "stress"],
        "response": ["agreement", "personal_experience", "opinion", "advice", "solution", "counterargument"],
        "solution": ["not_solution", "solution"],
    }
    sample_texts = [
        "Email admissions for AICTE fee refund deadline asap",
        "Hostel mess food and room wifi condition is frustrating",
        "In my batch placement SIP packages were helpful thanks",
        "You should check waitlist movement before paying deposit",
    ] * 6
    targets = {
        "issue": [["administrative_communication", "financial_concerns"], ["hostel_facilities", "campus_infrastructure"], ["placements"], ["administrative_communication"]] * 6,
        "sentiment": ["negative", "negative", "positive", "neutral"] * 6,
        "emotion": [["stress"], ["frustration"], ["gratitude"], ["curiosity"]] * 6,
        "response": ["solution", "opinion", "personal_experience", "advice"] * 6,
        "solution": ["solution", "not_solution", "not_solution", "not_solution"] * 6,
    }
    tr_idx, va_idx = slice(0, 18), slice(18, 24)
    cfg = KerasMultiTaskConfig(max_word_features=256, max_char_features=256, trunk_width=64, shared_dim=32, epochs=4, batch_size=6, random_state=42)
    clf = KerasMultiTaskClassifier(config=cfg)
    clf.fit(
        sample_texts[tr_idx], {k: v[tr_idx] for k, v in targets.items()},
        sample_texts[va_idx], {k: v[va_idx] for k, v in targets.items()},
        canonical,
    )
    assert clf.backend_name_ == "torch"
    probs = clf.predict_proba(sample_texts[:3])
    assert probs["issue"].shape == (3, 8)
    assert probs["sentiment"].shape == (3, 3) and np.allclose(probs["sentiment"].sum(axis=1), 1.0, atol=1e-5)
    assert probs["emotion"].shape == (3, 6)
    assert probs["response"].shape == (3, 6) and np.allclose(probs["response"].sum(axis=1), 1.0, atol=1e-5)
    assert probs["solution"].shape == (3, 2) and np.allclose(probs["solution"].sum(axis=1), 1.0, atol=1e-5)

    out_dir = tmp_path / "keras_mt"
    assert clf.save(out_dir) > 0
    assert (out_dir / "model.keras").is_file()
    reloaded = KerasMultiTaskClassifier.load(out_dir)
    probs2 = reloaded.predict_proba(sample_texts[:3])
    for task in canonical:
        assert np.allclose(probs[task], probs2[task], atol=1e-5)
