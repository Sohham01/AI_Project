"""Experimental Keras 3 Multi-Task Neural Classifier on PyTorch Backend (V6.4).

Uses Keras 3 (`KERAS_BACKEND="torch"`) Functional API with a residual shared
representation trunk branching into all 5 RCI NLP task heads simultaneously:
  - ``issue``:     8-label multi-label sigmoid (`binary_crossentropy`)
  - ``sentiment``: 3-class softmax (`sparse_categorical_crossentropy`)
  - ``emotion``:   6-label multi-label sigmoid (`binary_crossentropy`)
  - ``response``:  canonical 6-class softmax (`sparse_categorical_crossentropy`)
  - ``solution``:  binary sigmoid (`binary_crossentropy`)

Runs natively on Python 3.14 + installed PyTorch CUDA/CPU via `KERAS_BACKEND=torch`
without requiring TensorFlow (which has no Python 3.14 wheels).
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from ml.models.neural_baselines import (
    _calibrate_multilabel_thresholds,
    _calibrate_temperature_np,
    _encode_targets,
    _macro_f1_multilabel,
    _macro_f1_single,
    _sigmoid_np,
    _softmax_np,
)
from ml.models.xgboost_baseline import extract_linguistic_features

HAS_KERAS = (
    importlib.util.find_spec("keras") is not None
    and importlib.util.find_spec("torch") is not None
)

TASK_ORDER = ("issue", "sentiment", "emotion", "response", "solution")


def require_keras_torch() -> Tuple[Any, Any]:
    """Configure `KERAS_BACKEND=torch` and import Keras 3 + torch or fail informatively."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import torch
        import keras

        return keras, torch
    except ImportError as exc:
        raise RuntimeError(
            "Keras 3 with PyTorch backend is an optional ML dependency not installed in the base runtime.\n"
            "Install optional ML extras with:\n  pip install -r backend/requirements-ml.txt"
        ) from exc


@dataclass
class KerasMultiTaskConfig:
    """Hyperparameters for the Keras 3 Functional Multi-Task network."""

    max_word_features: int = 1536
    max_char_features: int = 1536
    trunk_width: int = 256
    shared_dim: int = 128
    head_dim: int = 64
    dropout: float = 0.35
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.5e-2
    random_state: int = 42


def _make_multitask_vectorizer(max_word: int, max_char: int) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=max_word, min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=max_char, min_df=2, sublinear_tf=True)),
        ]
    )


class KerasMultiTaskClassifier:
    """5-Head Keras 3 Functional API Multi-Task Classifier (`KERAS_BACKEND='torch'`)."""

    def __init__(self, config: KerasMultiTaskConfig | None = None) -> None:
        self.config = config or KerasMultiTaskConfig()
        self.vectorizer = _make_multitask_vectorizer(
            self.config.max_word_features, self.config.max_char_features
        )
        self.ling_scaler = StandardScaler()
        self.labels_: Dict[str, List[str]] = {}
        self.model_: Any = None
        self.thresholds_: Dict[str, Any] = {}
        self.temperatures_: Dict[str, float] = {t: 1.0 for t in TASK_ORDER}
        self.param_count_: int = 0
        self.train_seconds_: float = 0.0
        self.best_epoch_: int = 0
        self.backend_name_: str = "torch"

    def _featurize(self, texts: Sequence[str], *, fit: bool) -> np.ndarray:
        if fit:
            x_tfidf = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
            x_ling = self.ling_scaler.fit_transform(extract_linguistic_features(texts)).astype(np.float32)
        else:
            x_tfidf = self.vectorizer.transform(texts).astype(np.float32).toarray()
            x_ling = self.ling_scaler.transform(extract_linguistic_features(texts)).astype(np.float32)
        return np.hstack([x_tfidf, x_ling]).astype(np.float32)

    @staticmethod
    def _build_functional_model(keras: Any, in_dim: int, cfg: KerasMultiTaskConfig, labels: Mapping[str, Sequence[str]]) -> Any:
        layers = keras.layers
        inp = keras.Input(shape=(in_dim,), name="tfidf_linguistic_input")
        x = layers.Dense(cfg.trunk_width, activation="gelu", name="trunk_proj")(inp)
        x = layers.LayerNormalization(name="trunk_norm1")(x)
        x = layers.Dropout(cfg.dropout, name="trunk_drop1")(x)

        h = layers.Dense(cfg.trunk_width, activation="gelu", name="res_dense")(x)
        h = layers.LayerNormalization(name="res_norm")(h)
        h = layers.Dropout(max(0.15, cfg.dropout - 0.10), name="res_drop")(h)
        res = layers.Add(name="trunk_residual")([x, h])

        shared = layers.Dense(cfg.shared_dim, activation="gelu", name="shared_bottleneck")(res)
        shared = layers.LayerNormalization(name="shared_norm")(shared)
        shared = layers.Dropout(0.20, name="shared_drop")(shared)

        def _task_tower(name: str) -> Any:
            ht = layers.Dense(cfg.head_dim, activation="gelu", name=f"{name}_proj")(shared)
            ht = layers.LayerNormalization(name=f"{name}_norm")(ht)
            return layers.Dropout(0.15, name=f"{name}_drop")(ht)

        out_issue = layers.Dense(len(labels["issue"]), activation="sigmoid", name="issue")(_task_tower("issue"))
        out_sent = layers.Dense(len(labels["sentiment"]), activation="softmax", name="sentiment")(_task_tower("sentiment"))
        out_emo = layers.Dense(len(labels["emotion"]), activation="sigmoid", name="emotion")(_task_tower("emotion"))
        out_resp = layers.Dense(len(labels["response"]), activation="softmax", name="response")(_task_tower("response"))
        out_sol = layers.Dense(1, activation="sigmoid", name="solution")(_task_tower("solution"))

        return keras.Model(
            inputs=inp,
            outputs={
                "issue": out_issue,
                "sentiment": out_sent,
                "emotion": out_emo,
                "response": out_resp,
                "solution": out_sol,
            },
            name="RCI_Keras3_MultiTask_Classifier",
        )

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets_by_task: Mapping[str, Sequence[Any]],
        val_texts: Sequence[str],
        val_targets_by_task: Mapping[str, Sequence[Any]],
        canonical_labels: Mapping[str, Sequence[str]],
    ) -> "KerasMultiTaskClassifier":
        keras, torch = require_keras_torch()
        keras.utils.set_random_seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

        t0 = time.perf_counter()
        self.backend_name_ = keras.backend.backend()
        self.labels_ = {t: list(canonical_labels[t]) for t in TASK_ORDER}

        X_tr = self._featurize(train_texts, fit=True)
        X_va = self._featurize(val_texts, fit=False)

        def _pack_y(src: Mapping[str, Sequence[Any]]) -> Dict[str, np.ndarray]:
            return {
                "issue": _encode_targets(src["issue"], self.labels_["issue"], True).astype(np.float32),
                "sentiment": _encode_targets(src["sentiment"], self.labels_["sentiment"], False).astype(np.int64),
                "emotion": _encode_targets(src["emotion"], self.labels_["emotion"], True).astype(np.float32),
                "response": _encode_targets(src["response"], self.labels_["response"], False).astype(np.int64),
                "solution": _encode_targets(src["solution"], self.labels_["solution"], False).astype(np.float32).reshape(-1, 1),
            }

        y_tr = _pack_y(train_targets_by_task)
        y_va = _pack_y(val_targets_by_task)

        # Balanced inverse-frequency sample weights for imbalanced multiclass + binary solution heads
        def _cls_weights(y_arr: np.ndarray, n_cls: int) -> np.ndarray:
            flat = y_arr.reshape(-1).astype(np.int64)
            cnt = np.bincount(flat, minlength=n_cls).astype(np.float32)
            w_cls = np.clip(len(flat) / np.maximum(cnt * n_cls, 1.0), 0.5, 4.0)
            return w_cls[flat].astype(np.float32)

        def _multilabel_row_weights(y_mat: np.ndarray) -> np.ndarray:
            pos_counts = np.maximum(y_mat.sum(axis=0), 1.0)
            inv_freq = np.clip(len(y_mat) / (len(pos_counts) * pos_counts), 0.75, 4.0)
            row_w = np.ones(len(y_mat), dtype=np.float32)
            for idx_r, row in enumerate(y_mat):
                active = row > 0.5
                if np.any(active):
                    row_w[idx_r] = float(np.clip(np.max(inv_freq[active]), 1.0, 4.0))
            return row_w

        sample_weights = {
            "issue": _multilabel_row_weights(y_tr["issue"]),
            "sentiment": _cls_weights(y_tr["sentiment"], len(self.labels_["sentiment"])),
            "emotion": _multilabel_row_weights(y_tr["emotion"]),
            "response": _cls_weights(y_tr["response"], len(self.labels_["response"])),
            "solution": _cls_weights(y_tr["solution"], 2),
        }

        model = self._build_functional_model(keras, X_tr.shape[1], self.config, self.labels_)
        self.param_count_ = int(model.count_params())

        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            ),
            loss={
                "issue": "binary_crossentropy",
                "sentiment": "sparse_categorical_crossentropy",
                "emotion": "binary_crossentropy",
                "response": "sparse_categorical_crossentropy",
                "solution": "binary_crossentropy",
            },
            loss_weights={"issue": 1.2, "sentiment": 1.0, "emotion": 1.2, "response": 1.2, "solution": 1.0},
        )

        best_mean_macro = -1.0
        best_weights = None
        for epoch in range(1, self.config.epochs + 1):
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weights,
                batch_size=self.config.batch_size,
                epochs=1,
                verbose=0,
                shuffle=True,
            )
            va_out = model.predict(X_va, batch_size=256, verbose=0)
            f1_issue, _ = _macro_f1_multilabel(y_va["issue"].astype(int), np.asarray(va_out["issue"]), [0.4] * len(self.labels_["issue"]))
            f1_emo, _ = _macro_f1_multilabel(y_va["emotion"].astype(int), np.asarray(va_out["emotion"]), [0.4] * len(self.labels_["emotion"]))
            f1_sent = _macro_f1_single(y_va["sentiment"], np.argmax(va_out["sentiment"], axis=1), len(self.labels_["sentiment"]))
            f1_resp = _macro_f1_single(y_va["response"], np.argmax(va_out["response"], axis=1), len(self.labels_["response"]))
            sol_p = np.asarray(va_out["solution"]).reshape(-1)
            f1_sol = _macro_f1_single(y_va["solution"].reshape(-1).astype(int), (sol_p >= 0.5).astype(int), 2)
            mean_macro = float(np.mean([f1_issue, f1_sent, f1_emo, f1_resp, f1_sol]))
            if mean_macro >= best_mean_macro:
                best_mean_macro = mean_macro
                self.best_epoch_ = epoch
                best_weights = model.get_weights()

        if best_weights is not None:
            model.set_weights(best_weights)
        self.model_ = model

        # Validation-only temperature (T >= 1.0) & threshold calibration
        self.temperatures_ = {t: 1.0 for t in TASK_ORDER}
        raw_va_logits = self._predict_raw_logits(val_texts)
        for t in ("issue", "emotion"):
            self.temperatures_[t] = _calibrate_temperature_np(raw_va_logits[t], y_va[t], multilabel=True)
        for t in ("sentiment", "response"):
            self.temperatures_[t] = _calibrate_temperature_np(raw_va_logits[t], y_va[t], multilabel=False)
        y_sol_va = y_va["solution"].reshape(-1).astype(int)
        self.temperatures_["solution"] = _calibrate_temperature_np(raw_va_logits["solution"], y_sol_va, multilabel=False)

        va_probs = self.predict_proba(val_texts)
        self.thresholds_ = {
            "issue": np.array(
                [
                    _calibrate_multilabel_thresholds(y_va["issue"].astype(int), va_probs["issue"], self.labels_["issue"])[lbl]
                    for lbl in self.labels_["issue"]
                ],
                dtype=np.float64,
            ),
            "emotion": np.array(
                [
                    _calibrate_multilabel_thresholds(y_va["emotion"].astype(int), va_probs["emotion"], self.labels_["emotion"])[lbl]
                    for lbl in self.labels_["emotion"]
                ],
                dtype=np.float64,
            ),
        }
        sol_p1 = va_probs["solution"][:, 1]
        best_tau, best_f1 = 0.5, -1.0
        for tau in [round(0.25 + 0.025 * i, 3) for i in range(21)]:
            f1 = _macro_f1_single(y_sol_va, (sol_p1 >= tau).astype(int), 2)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        self.thresholds_["solution"] = float(best_tau)

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def _predict_raw_logits(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        X = self._featurize(texts, fit=False)
        raw = self.model_.predict(X, batch_size=256, verbose=0)
        eps = 1e-6
        p_sol1 = np.clip(np.asarray(raw["solution"], dtype=np.float64).reshape(-1), eps, 1.0 - eps)
        sol_probs = np.column_stack([1.0 - p_sol1, p_sol1])
        logits: Dict[str, np.ndarray] = {}
        for t in ("issue", "emotion"):
            p = np.clip(np.asarray(raw[t], dtype=np.float64), eps, 1.0 - eps)
            logits[t] = np.log(p / (1.0 - p))
        for t in ("sentiment", "response"):
            p = np.clip(np.asarray(raw[t], dtype=np.float64), eps, 1.0)
            logits[t] = np.log(p)
        logits["solution"] = np.log(sol_probs)
        return logits

    def predict_proba(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        logits = self._predict_raw_logits(texts)
        return {
            "issue": _sigmoid_np(logits["issue"] / max(self.temperatures_.get("issue", 1.0), 1.0)),
            "sentiment": _softmax_np(logits["sentiment"] / max(self.temperatures_.get("sentiment", 1.0), 1.0)),
            "emotion": _sigmoid_np(logits["emotion"] / max(self.temperatures_.get("emotion", 1.0), 1.0)),
            "response": _softmax_np(logits["response"] / max(self.temperatures_.get("response", 1.0), 1.0)),
            "solution": _softmax_np(logits["solution"] / max(self.temperatures_.get("solution", 1.0), 1.0)),
        }

    def predict(self, texts: Sequence[str]) -> Dict[str, List[Any]]:
        probs = self.predict_proba(texts)
        out: Dict[str, List[Any]] = {}
        for t in ("issue", "emotion"):
            thr = self.thresholds_[t]
            preds_ml: List[List[str]] = []
            for row in probs[t]:
                picked = [self.labels_[t][i] for i, p in enumerate(row) if p >= thr[i]]
                if not picked:
                    picked = [self.labels_[t][int(np.argmax(row))]]
                preds_ml.append(sorted(picked))
            out[t] = preds_ml
        for t in ("sentiment", "response"):
            out[t] = [self.labels_[t][int(np.argmax(row))] for row in probs[t]]
        sol_tau = float(self.thresholds_.get("solution", 0.5))
        out["solution"] = [
            self.labels_["solution"][1 if row[1] >= sol_tau else 0] for row in probs["solution"]
        ]
        return out

    def save(self, directory: Path | str) -> int:
        require_keras_torch()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model_.save(d / "model.keras")
        joblib.dump(self.vectorizer, d / "vectorizer.joblib", compress=3)
        joblib.dump(self.ling_scaler, d / "ling_scaler.joblib", compress=3)
        meta = {
            "config": asdict(self.config),
            "labels": self.labels_,
            "backend_name": self.backend_name_,
            "param_count": self.param_count_,
            "temperatures": self.temperatures_,
            "thresholds": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.thresholds_.items()
            },
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())

    @classmethod
    def load(cls, directory: Path | str) -> "KerasMultiTaskClassifier":
        keras, _ = require_keras_torch()
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        inst = cls(config=KerasMultiTaskConfig(**meta["config"]))
        inst.labels_ = {k: list(v) for k, v in meta["labels"].items()}
        inst.backend_name_ = meta["backend_name"]
        inst.param_count_ = int(meta["param_count"])
        inst.temperatures_ = {k: float(v) for k, v in (meta.get("temperatures") or {}).items()} or {t: 1.0 for t in TASK_ORDER}
        inst.vectorizer = joblib.load(d / "vectorizer.joblib")
        inst.ling_scaler = joblib.load(d / "ling_scaler.joblib")
        inst.thresholds_ = {
            k: (np.array(v, dtype=np.float64) if isinstance(v, list) else float(v))
            for k, v in meta["thresholds"].items()
        }
        inst.model_ = keras.saving.load_model(d / "model.keras")
        return inst
