"""Sprint P0-A Regression Tests — Collision-Resistant Embedding Cache Identity."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch optional in lightweight base environment")
def test_embedding_cache_key_sensitive_to_every_position_and_config(tmp_path: Path) -> None:
    from ml.topics.bertopic.embeddings import (
        EMBEDDING_CATALOG,
        compute_embedding_cache_identity,
        compute_embedding_cache_key,
    )

    texts = [f"Reddit discussion doc index {i} about IIM placements and hostel" for i in range(600)]

    base_key = compute_embedding_cache_key("multilingual_e5_small", texts)
    assert len(base_key) == 64

    # Identical inputs -> identical key
    assert compute_embedding_cache_key("multilingual_e5_small", list(texts)) == base_key

    # Changing a text strictly in the middle (index 300, beyond legacy [:200] & [-100:]) MUST change key
    mid_changed = list(texts)
    mid_changed[300] = "Changed middle comment about AICTE refund delay"
    assert compute_embedding_cache_key("multilingual_e5_small", mid_changed) != base_key

    # Changing first or last text MUST change key
    first_changed = list(texts)
    first_changed[0] = "Changed first comment"
    assert compute_embedding_cache_key("multilingual_e5_small", first_changed) != base_key

    last_changed = list(texts)
    last_changed[-1] = "Changed last comment"
    assert compute_embedding_cache_key("multilingual_e5_small", last_changed) != base_key

    # Reordering changes key when order_independent=False, preserves when True
    swapped = list(texts)
    swapped[10], swapped[11] = swapped[11], swapped[10]
    assert compute_embedding_cache_key("multilingual_e5_small", swapped, order_independent=False) != base_key
    ord_key_a = compute_embedding_cache_key("multilingual_e5_small", texts, order_independent=True)
    ord_key_b = compute_embedding_cache_key("multilingual_e5_small", swapped, order_independent=True)
    assert ord_key_a == ord_key_b

    # Config / mode / preprocessing / corpus / revision mutations MUST change key
    assert compute_embedding_cache_key("multilingual_e5_small", texts, is_query=True) != base_key
    assert compute_embedding_cache_key("multilingual_minilm", texts) != base_key
    assert (
        compute_embedding_cache_key("multilingual_e5_small", texts, preprocessing_version="v_custom")
        != base_key
    )
    assert compute_embedding_cache_key("multilingual_e5_small", texts, corpus_version="6.3.0") != base_key
    assert (
        compute_embedding_cache_key("multilingual_e5_small", texts, model_revision="rev-abc123")
        != base_key
    )

    # Local checkpoint file/config fingerprint mutation MUST change key
    local_dir = tmp_path / "local_ckpt"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")
    (local_dir / "model.safetensors").write_bytes(b"weights_v1")

    orig_entry = dict(EMBEDDING_CATALOG["muril_dapt_wwm"])
    try:
        EMBEDDING_CATALOG["muril_dapt_wwm"] = {**orig_entry, "model_name": str(local_dir)}
        ident_1 = compute_embedding_cache_identity("muril_dapt_wwm", texts[:20])
        (local_dir / "config.json").write_text('{"hidden_size": 768, "dropout": 0.2}', encoding="utf-8")
        ident_2 = compute_embedding_cache_identity("muril_dapt_wwm", texts[:20])
        assert ident_1["cache_key"] != ident_2["cache_key"]

        (local_dir / "model.safetensors").write_bytes(b"weights_v2")
        ident_3 = compute_embedding_cache_identity("muril_dapt_wwm", texts[:20])
        assert ident_2["cache_key"] != ident_3["cache_key"]
    finally:
        EMBEDDING_CATALOG["muril_dapt_wwm"] = orig_entry


@pytest.mark.skipif(not HAS_TORCH, reason="torch optional in lightweight base environment")
def test_encode_documents_rejects_legacy_and_mismatched_cache_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.topics.bertopic.embeddings as emb_mod

    class DummySentenceTransformer:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            DummySentenceTransformer.calls += 1
            out = np.ones((len(texts), 384), dtype=np.float32)
            out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", DummySentenceTransformer)

    docs = ["Doc zero", "Doc one", "Doc two"]
    ident = emb_mod.compute_embedding_cache_identity("multilingual_e5_small", docs)
    v2_dir = tmp_path / "v2"
    v2_dir.mkdir(parents=True)

    # Plant a legacy v1 file in cache root and a stale schema/mismatched key in v2/
    np.save(tmp_path / "multilingual_e5_small_legacy.npy", np.zeros((3, 384), dtype=np.float32))
    stem = f"multilingual_e5_small_{ident['cache_key'][:32]}"
    np.save(v2_dir / f"{stem}.npy", np.zeros((3, 384), dtype=np.float32))
    (v2_dir / f"{stem}.meta.json").write_text(
        json.dumps({"cache_schema_version": "v1", "cache_key": "wrong", "n_docs": 3, "dim": 384}),
        encoding="utf-8",
    )

    arr, meta = emb_mod.encode_documents(docs, "multilingual_e5_small", use_cache=True, cache_dir=tmp_path)
    assert DummySentenceTransformer.calls == 1, "Legacy/mismatched cache MUST NOT be reused"
    assert meta["cached"] is False
    assert meta["cache_schema_version"] == emb_mod.EMBEDDING_CACHE_SCHEMA_VERSION
    assert arr.shape == (3, 384)

    # Second call with identical inputs MUST reuse the valid v2 entry without re-encoding
    arr2, meta2 = emb_mod.encode_documents(docs, "multilingual_e5_small", use_cache=True, cache_dir=tmp_path)
    assert DummySentenceTransformer.calls == 1
    assert meta2["cached"] is True
    assert np.allclose(arr, arr2)
