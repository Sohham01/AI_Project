"""Isolated topic-model registry tracking NMF and BERTopic experiments side by side.

Never modifies or overwrites existing NMF artifacts; registers NMF as a read-only
baseline alongside versioned BERTopic runs under ``ml/artifacts/topic_models/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = ROOT / "ml" / "artifacts" / "topic_models" / "registry.json"


class TopicModelRegistry:
    """Append/upsert JSON registry for comparing NMF and BERTopic runs."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)

    def load(self) -> Dict[str, Any]:
        if not self.registry_path.is_file():
            return {
                "schema_version": "1.0.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "default_model_id": "nmf_v1",
                "models": [],
            }
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def register(
        self,
        *,
        model_id: str,
        algorithm: str,
        embedding_model: Optional[str],
        corpus_version: str,
        preprocessing_config: Dict[str, Any],
        n_documents: int,
        n_topics: int,
        outlier_count: int,
        training_parameters: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
        artifact_paths: Dict[str, str],
        notes: str = "",
        set_default: bool = False,
    ) -> Dict[str, Any]:
        """Upsert a model entry with full provenance and evaluation metadata."""
        data = self.load()
        entry = {
            "model_id": model_id,
            "algorithm": algorithm,
            "embedding_model": embedding_model,
            "corpus_version": corpus_version,
            "preprocessing_config": preprocessing_config,
            "n_documents": int(n_documents),
            "n_topics": int(n_topics),
            "outlier_count": int(outlier_count),
            "outlier_rate": round(int(outlier_count) / max(int(n_documents), 1), 4),
            "training_parameters": training_parameters,
            "evaluation_metrics": evaluation_metrics,
            "artifact_paths": artifact_paths,
            "notes": notes,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        models: List[Dict[str, Any]] = [m for m in data.get("models", []) if m.get("model_id") != model_id]
        models.append(entry)
        data["models"] = models
        if set_default:
            data["default_model_id"] = model_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return entry

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.load().get("models", []):
            if m.get("model_id") == model_id:
                return m
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self.load().get("models", []))
