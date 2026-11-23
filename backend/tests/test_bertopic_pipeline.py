"""Phase 1 QA Tests — BERTopic Pipeline, Registry, NMF Preservation & /api/demo/topics Contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_nmf_artifact_preserved_intact_alongside_bertopic() -> None:
    nmf_path = ROOT / "ml" / "artifacts" / "live_corpus" / "topic_discovery.json"
    bt_path = ROOT / "ml" / "artifacts" / "live_corpus" / "bertopic_v1_topics.json"
    reg_path = ROOT / "ml" / "artifacts" / "topic_models" / "registry.json"

    assert nmf_path.is_file(), "Existing NMF topic_discovery.json must never be removed"
    assert bt_path.is_file(), "BERTopic v1 serving artifact must exist separately"
    assert reg_path.is_file(), "Topic model registry must exist"

    nmf = json.loads(nmf_path.read_text(encoding="utf-8"))
    assert nmf["result"]["topic_model"] == "nmf"
    assert nmf["result"]["n_topics"] == 10
    assert nmf["result"]["n_documents"] == 3459

    bt = json.loads(bt_path.read_text(encoding="utf-8"))
    assert bt["model_info"]["algorithm"] == "bertopic"
    assert bt["model_info"]["n_documents"] == 3459
    assert bt["model_info"]["n_topics"] >= 10
    assert bt["outliers"]["count"] > 0, "BERTopic must preserve explicit -1 outlier documents"

    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    ids = {m["model_id"] for m in reg["models"]}
    assert "nmf_v1" in ids
    assert "bertopic_v1" in ids
    assert reg["default_model_id"] == "nmf_v1"


def test_topics_api_backward_compatible_nmf_and_bertopic_switch() -> None:
    client = TestClient(app)

    # Default call remains NMF v1 (backward compatible)
    r_def = client.get("/api/demo/topics")
    assert r_def.status_code == 200
    d_def = r_def.json()
    assert d_def["model"] == "nmf"
    assert len(d_def["items"]) == 10
    assert all(it["namespaced_topic_id"].startswith("nmf:v1:") for it in d_def["items"])
    assert len(d_def["available_models"]) == 2

    # Explicit BERTopic switch returns isolated BERTopic topics, outliers, comparison & metadata
    r_bt = client.get("/api/demo/topics?model=bertopic")
    assert r_bt.status_code == 200
    d_bt = r_bt.json()
    assert d_bt["model"] == "bertopic"
    assert len(d_bt["items"]) >= 10
    assert all(it["namespaced_topic_id"].startswith("bertopic:") for it in d_bt["items"])
    assert d_bt["outliers"] is not None and d_bt["outliers"]["count"] > 0
    assert d_bt["model_info"]["embedding_model"] == "intfloat/multilingual-e5-small"
    assert len(d_bt["comparison"]["table"]) >= 5

    # Invalid model name rejected cleanly
    r_bad = client.get("/api/demo/topics?model=invalid")
    assert r_bad.status_code == 422


def test_bertopic_fit_save_load_reproducibility_and_outliers(tmp_path: Path) -> None:
    """Fast deterministic synthetic-embedding test of BERTopic fit/save/load/transform/outliers."""
    from ml.topics.bertopic.config import BERTopicExperimentConfig
    from ml.topics.bertopic.pipeline import RCIBERTopicPipeline

    rng = np.random.default_rng(123)
    # Create 3 clean dense clusters + 6 distant outlier points in 16-d space
    c0 = rng.normal(loc=np.r_[np.ones(4), np.zeros(12)], scale=0.05, size=(15, 16))
    c1 = rng.normal(loc=np.r_[np.zeros(4), np.ones(4), np.zeros(8)], scale=0.05, size=(15, 16))
    c2 = rng.normal(loc=np.r_[np.zeros(8), np.ones(4), np.zeros(4)], scale=0.05, size=(15, 16))
    outliers = rng.normal(loc=np.r_[np.zeros(12), np.ones(4)], scale=1.8, size=(6, 16))
    embs = np.vstack([c0, c1, c2, outliers]).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-6)

    docs = (
        [f"hostel mess food quality rooms campus water issue {i}" for i in range(15)]
        + [f"placement summer internship package ctc recruiter {i}" for i in range(15)]
        + [f"education loan bank cibil tuition fees refund {i}" for i in range(15)]
        + [f"random miscellaneous offbeat tangent {i}" for i in range(6)]
    )

    cfg = BERTopicExperimentConfig(
        model_id="test_bt_unit",
        embedding_key="multilingual_e5_small",
        embedding_model_name="synthetic-unit-emb",
        umap_n_neighbors=5,
        umap_n_components=3,
        hdbscan_min_cluster_size=6,
        hdbscan_min_samples=3,
        random_state=42,
    )

    p1 = RCIBERTopicPipeline(cfg)
    t1, _, _ = p1.fit(docs, embeddings=embs)

    # Reproducibility check (same seed -> identical topic assignments)
    p2 = RCIBERTopicPipeline(cfg)
    t2, _, _ = p2.fit(docs, embeddings=embs)
    assert t1 == t2, "BERTopic fit must be reproducible under identical random_state"
    assert len({t for t in t1 if t != -1}) >= 2

    # Save & load SafeTensors round-trip + transform assignment
    save_dir = tmp_path / "bt_saved"
    p1.save(save_dir)
    loaded = RCIBERTopicPipeline.load(save_dir)
    assigned, _ = loaded.transform(docs[:3], embeddings=embs[:3])
    assert len(assigned) == 3
    assert all(isinstance(x, int) for x in assigned)
