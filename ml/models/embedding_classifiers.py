"""V6.7 Dense Multilingual Embedding & Hybrid TF-IDF + Embedding Classifiers.

Provides controlled, zero-leakage feature representations and classifiers to compare:
  1. Pure Lexical TF-IDF (`word_only` vs. `word_char` subword n-grams)
  2. Pure Dense Multilingual Embeddings (`MiniLM-L12-384`, `Multilingual-E5-Small-384`,
     `MuRIL-DAPT10x-768`, `XLM-R-DAPT10x-768`, and multi-encoder concatenations)
  3. Hybrid Lexical + Semantic Fusion (`TF-IDF ⊕ Centered Unit-Norm Embeddings ⊕ Linguistic Cues`)

Across three distinct model families:
  - Linear: L2-regularized Logistic Regression (`EmbeddingAndHybridLogisticClassifier`)
  - Gradient Boosted Trees: `hist` XGBoost (`EmbeddingAndHybridXGBoostClassifier`)
  - Neural: PyTorch 2-Layer GELU MLP & Dual-Branch Feature-Gated Fusion Network
    (`EmbeddingMLPClassifier`, `HybridGatedTFIDFEmbeddingMLP`)

All vectorizers, `TruncatedSVD` projections, `StandardScaler` statistics, anisotropy
mean-centering vectors, temperatures, and decision thresholds are fit STRICTLY on
training folds (`tr_idx`) with zero test/validation leakage.
"""

from __future__ import annotations

import gc
import importlib.util
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from ml.models.neural_baselines import (
    _calibrate_multilabel_thresholds,
    _calibrate_temperature_np,
    _encode_targets,
    _make_word_char_vectorizer,
    _sigmoid_np,
    _softmax_np,
)
from ml.models.xgboost_baseline import (
    HAS_XGBOOST,
    _balanced_sample_weights,
    _tune_binary_threshold,
    _tune_positive_f1_threshold,
    extract_linguistic_features,
    require_xgboost,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = Path(os.environ.get("RCI_MODEL_CACHE_DIR") or (ROOT / ".cache" / "models"))
if not _DEFAULT_CACHE.exists() and Path("D:/AI_Project_Transformer_Temp").is_dir():
    _DEFAULT_CACHE = Path("D:/AI_Project_Transformer_Temp")


ENCODER_SPECS: Dict[str, Dict[str, Any]] = {
    "minilm_384": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "prefix": "",
        "pooling": "mean",
        "family": "contrastive_sentence",
        "description": "Distilled 50-language paraphrase SentenceTransformer (384-dim)",
    },
    "e5_small_384": {
        "model_id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "prefix": "passage: ",
        "pooling": "mean",
        "family": "contrastive_sentence",
        "description": "Contrastively pretrained 100-language E5 sentence encoder (EN/HI/BN/OR, 384-dim)",
    },
    "muril_dapt10x_768": {
        "model_id": "muril_dapt_v66_wwm",
        "fallback_id": "google/muril-base-cased",
        "dim": 768,
        "prefix": "",
        "pooling": "cls_mean",
        "family": "domain_dapt_mlm",
        "description": "V6.6 4-epoch LLRD 10x Reddit WWM MuRIL contextual encoder (768-dim)",
    },
    "xlmr_dapt10x_768": {
        "model_id": "xlmr_dapt_v66_wwm",
        "fallback_id": "xlm-roberta-base",
        "dim": 768,
        "prefix": "",
        "pooling": "cls_mean",
        "family": "domain_dapt_mlm",
        "description": "V6.6 4-epoch LLRD 10x Reddit WWM XLM-RoBERTa contextual encoder (768-dim)",
    },
}


def center_and_l2_normalize(
    x: np.ndarray,
    mean_vec: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """All-but-the-mean anisotropy centering (fit on train only) + row-wise L2 unit norm."""
    arr = np.asarray(x, dtype=np.float32)
    mu = np.mean(arr, axis=0, keepdims=True) if mean_vec is None else mean_vec
    centered = arr - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.maximum(norms, 1e-8)
    return unit.astype(np.float32), mu.astype(np.float32)


def extract_frozen_embeddings(
    texts: Sequence[str],
    encoder_key: str,
    *,
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    cache_tag: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Encode ``texts`` with a frozen local Transformer/SentenceTransformer checkpoint."""
    if not (HAS_TORCH and HAS_TRANSFORMERS):
        raise RuntimeError("PyTorch and transformers are required for frozen neural embedding extraction.")
    if encoder_key not in ENCODER_SPECS:
        raise ValueError(f"Unknown encoder_key={encoder_key!r}; expected one of {sorted(ENCODER_SPECS)}")

    import hashlib
    import json
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from ml.experiments.v6_programme.v66_architecture import resolve_local_backbone_path

    spec = ENCODER_SPECS[encoder_key]
    clean_texts = [f"{spec['prefix']}{str(t or '').strip()}" for t in texts]

    h = hashlib.sha256()
    h.update(f"{encoder_key}|max_len={max_length}|pooling={spec['pooling']}|n={len(clean_texts)}|".encode("utf-8"))
    for t in clean_texts:
        b = t.encode("utf-8", errors="ignore")
        h.update(f"{len(b)}:".encode("ascii") + b + b"\n")
    digest = h.hexdigest()[:28]

    c_root = cache_dir or (_DEFAULT_CACHE / "v6_7_embeddings_cache")
    c_root.mkdir(parents=True, exist_ok=True)
    tag_prefix = f"{cache_tag}__" if cache_tag else ""
    npy_path = c_root / f"{tag_prefix}{encoder_key}_{digest}.npy"
    meta_path = c_root / f"{tag_prefix}{encoder_key}_{digest}.json"

    if npy_path.is_file() and meta_path.is_file():
        arr = np.load(npy_path)
        if arr.shape == (len(clean_texts), int(spec["dim"])):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["cached"] = True
            return arr.astype(np.float32), meta

    model_path, local_only = resolve_local_backbone_path(spec["model_id"])
    if not local_only and spec.get("fallback_id"):
        model_path, local_only = resolve_local_backbone_path(str(spec["fallback_id"]))

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    enc = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(dev)
    enc.eval()

    amp_enabled = dev == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            inp = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.amp.autocast(device_type="cuda" if amp_enabled else "cpu", dtype=amp_dtype, enabled=amp_enabled):
                hidden = enc(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean_v = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            mean_u = F.normalize(mean_v.float(), p=2, dim=-1)
            if spec["pooling"] == "cls_mean":
                cls_u = F.normalize(hidden[:, 0, :].float(), p=2, dim=-1)
                vec = F.normalize(0.65 * mean_u + 0.35 * cls_u, p=2, dim=-1)
            else:
                vec = mean_u
            chunks.append(vec.cpu().numpy())

    embs = np.concatenate(chunks, axis=0).astype(np.float32)
    elapsed_s = round(time.perf_counter() - t0, 3)

    del enc, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        "encoder_key": encoder_key,
        "resolved_model_path": Path(model_path).name,
        "dim": int(spec["dim"]),
        "pooling": spec["pooling"],
        "family": spec["family"],
        "n_docs": len(clean_texts),
        "encode_seconds": elapsed_s,
        "encode_ms_per_doc": round((elapsed_s * 1000.0) / max(len(clean_texts), 1), 3),
        "cached": False,
    }
    np.save(npy_path, embs)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return embs, meta


def _make_word_only_vectorizer(max_word: int = 12000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_word,
        min_df=2,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# 1. Controlled Linear Classifier: TF-IDF vs. Dense Embedding vs. Hybrid
# ---------------------------------------------------------------------------
class EmbeddingAndHybridLogisticClassifier:
    """Apples-to-apples L2 Logistic Regression over TF-IDF, Dense Embeddings, or Hybrid Fusion."""

    def __init__(
        self,
        *,
        mode: Literal["tfidf_word", "tfidf_word_char", "embedding_only", "hybrid_tfidf_emb"] = "embedding_only",
        multilabel: bool = False,
        c_value: float = 2.5,
        emb_scale: float = 2.2,
        include_linguistic_cues: bool = False,
        random_state: int = 42,
    ) -> None:
        self.mode = mode
        self.multilabel = multilabel
        self.c_value = c_value
        self.emb_scale = emb_scale
        self.include_linguistic_cues = include_linguistic_cues
        self.random_state = random_state

        self.vectorizer_: Any = None
        self.emb_means_: List[np.ndarray] = []
        self.cue_scaler_: Optional[StandardScaler] = None
        self.model_: Any = None
        self.labels_: List[str] = []
        self.thresholds_: Optional[np.ndarray] = None
        self.binary_threshold_: float = 0.5
        self.temperature_: float = 1.0
        self.feature_dim_: int = 0
        self.param_count_: int = 0
        self.train_seconds_: float = 0.0

    def _transform_blocks(
        self,
        texts: Sequence[str],
        emb_blocks: Optional[Sequence[np.ndarray]],
        *,
        fit: bool,
    ) -> Any:
        parts: List[Any] = []
        if self.mode in {"tfidf_word", "tfidf_word_char", "hybrid_tfidf_emb"}:
            if fit:
                self.vectorizer_ = (
                    _make_word_only_vectorizer()
                    if self.mode == "tfidf_word"
                    else _make_word_char_vectorizer()
                )
                x_lex = self.vectorizer_.fit_transform(texts)
            else:
                x_lex = self.vectorizer_.transform(texts)
            parts.append(x_lex)

        if self.mode in {"embedding_only", "hybrid_tfidf_emb"}:
            if not emb_blocks:
                raise ValueError(f"emb_blocks required for mode={self.mode!r}")
            normed_blocks: List[np.ndarray] = []
            if fit:
                self.emb_means_ = []
                for blk in emb_blocks:
                    u, mu = center_and_l2_normalize(blk, None)
                    self.emb_means_.append(mu)
                    normed_blocks.append(u)
            else:
                for blk, mu in zip(emb_blocks, self.emb_means_):
                    u, _ = center_and_l2_normalize(blk, mu)
                    normed_blocks.append(u)
            scale = self.emb_scale / math.sqrt(max(len(normed_blocks), 1))
            x_emb = np.hstack(normed_blocks) * float(scale)
            parts.append(sparse.csr_matrix(x_emb) if any(sparse.issparse(p) for p in parts) else x_emb)

        if self.include_linguistic_cues:
            cues = extract_linguistic_features(texts)
            if fit:
                self.cue_scaler_ = StandardScaler()
                cues_s = 0.25 * self.cue_scaler_.fit_transform(cues)
            else:
                assert self.cue_scaler_ is not None
                cues_s = 0.25 * self.cue_scaler_.transform(cues)
            parts.append(sparse.csr_matrix(cues_s) if any(sparse.issparse(p) for p in parts) else cues_s)

        if len(parts) == 1:
            return parts[0]
        if any(sparse.issparse(p) for p in parts):
            return sparse.hstack([sparse.csr_matrix(p) if not sparse.issparse(p) else p for p in parts]).tocsr()
        return np.hstack(parts).astype(np.float32)

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
        *,
        train_emb_blocks: Optional[Sequence[np.ndarray]] = None,
        val_emb_blocks: Optional[Sequence[np.ndarray]] = None,
    ) -> "EmbeddingAndHybridLogisticClassifier":
        t0 = time.perf_counter()
        self.labels_ = list(class_names)
        X_tr = self._transform_blocks(train_texts, train_emb_blocks, fit=True)
        X_va = self._transform_blocks(val_texts, val_emb_blocks, fit=False)
        self.feature_dim_ = int(X_tr.shape[1])

        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        # Select best regularization C strictly on validation Macro F1 across a compact 3-point grid
        best_c = self.c_value
        best_score = -1.0
        best_model: Any = None
        best_taus: Optional[np.ndarray] = None
        best_btau = 0.5
        best_temp = 1.0

        c_grid = (1.0, 2.5, 5.0) if self.mode in {"embedding_only", "hybrid_tfidf_emb"} else (self.c_value,)
        for c_cand in c_grid:
            base_lr = LogisticRegression(
                C=c_cand,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )
            mdl: Any = OneVsRestClassifier(base_lr) if self.multilabel else base_lr
            mdl.fit(X_tr, y_tr)

            raw_df = mdl.decision_function(X_va)
            if raw_df.ndim == 1:
                logits_va = np.column_stack([-0.5 * raw_df, 0.5 * raw_df])
            else:
                logits_va = np.asarray(raw_df, dtype=np.float64)

            temp = _calibrate_temperature_np(logits_va, y_va, self.multilabel)
            probs_va = _sigmoid_np(logits_va / temp) if self.multilabel else _softmax_np(logits_va / temp)

            if self.multilabel:
                tuned = _calibrate_multilabel_thresholds(y_va, probs_va, self.labels_)
                tau_arr = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)
                pred = (probs_va > tau_arr[None, :]).astype(int)
                f1s = []
                for j in range(len(self.labels_)):
                    tp = int(((y_va[:, j] == 1) & (pred[:, j] == 1)).sum())
                    fp = int(((y_va[:, j] == 0) & (pred[:, j] == 1)).sum())
                    fn = int(((y_va[:, j] == 1) & (pred[:, j] == 0)).sum())
                    denom = 2 * tp + fp + fn
                    if int(y_va[:, j].sum()) > 0 or (tp + fp) > 0:
                        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
                score = float(np.mean(f1s)) if f1s else 0.0
                btau = 0.5
            elif len(self.labels_) == 2:
                btau = _tune_binary_threshold(y_va, probs_va[:, 1], low=0.25, high=0.75)
                pred = (probs_va[:, 1] >= btau).astype(int)
                f1s = []
                for cls in (0, 1):
                    tp = int(((y_va == cls) & (pred == cls)).sum())
                    fp = int(((y_va != cls) & (pred == cls)).sum())
                    fn = int(((y_va == cls) & (pred != cls)).sum())
                    denom = 2 * tp + fp + fn
                    f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
                score = 0.5 * sum(f1s)
                tau_arr = None
            else:
                pred = np.argmax(probs_va, axis=1)
                f1s = []
                for cls in range(len(self.labels_)):
                    tp = int(((y_va == cls) & (pred == cls)).sum())
                    fp = int(((y_va != cls) & (pred == cls)).sum())
                    fn = int(((y_va == cls) & (pred != cls)).sum())
                    if int((y_va == cls).sum()) == 0 and (tp + fp) == 0:
                        continue
                    denom = 2 * tp + fp + fn
                    f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
                score = float(np.mean(f1s)) if f1s else 0.0
                tau_arr = None
                btau = 0.5

            if score > best_score + 1e-9:
                best_score = score
                best_c = c_cand
                best_model = mdl
                best_taus = tau_arr
                best_btau = btau
                best_temp = temp

        self.c_value = best_c
        self.model_ = best_model
        self.thresholds_ = best_taus
        self.binary_threshold_ = best_btau
        self.temperature_ = best_temp
        if self.multilabel:
            self.param_count_ = sum(int(e.coef_.size + e.intercept_.size) for e in self.model_.estimators_)
        else:
            self.param_count_ = int(self.model_.coef_.size + self.model_.intercept_.size)
        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        emb_blocks: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        X = self._transform_blocks(texts, emb_blocks, fit=False)
        raw_df = self.model_.decision_function(X)
        if raw_df.ndim == 1:
            logits = np.column_stack([-0.5 * raw_df, 0.5 * raw_df])
        else:
            logits = np.asarray(raw_df, dtype=np.float64)
        scaled = logits / max(self.temperature_, 1e-6)
        probs = _sigmoid_np(scaled) if self.multilabel else _softmax_np(scaled)
        return np.asarray(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Controlled XGBoost Classifier: Dense Embedding vs. Hybrid SVD-TFIDF + Emb
# ---------------------------------------------------------------------------
class EmbeddingAndHybridXGBoostClassifier:
    """Gradient Boosted Trees (`hist`) over dense embeddings or SVD-TFIDF + Embeddings."""

    def __init__(
        self,
        *,
        task_name: str,
        mode: Literal["embedding_only", "hybrid_svd_emb"] = "embedding_only",
        multilabel: bool = False,
        svd_components: int = 64,
        random_state: int = 42,
        n_jobs: int = 4,
    ) -> None:
        require_xgboost()
        self.task_name = task_name
        self.mode = mode
        self.multilabel = multilabel
        self.svd_components = svd_components
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.vectorizer_: Optional[FeatureUnion] = None
        self.svd_: Optional[TruncatedSVD] = None
        self.svd_scaler_: Optional[StandardScaler] = None
        self.cue_scaler_: Optional[StandardScaler] = None
        self.emb_means_: List[np.ndarray] = []
        self.estimators_: List[Any] = []
        self.constant_probs_: Dict[int, float] = {}
        self.labels_: List[str] = []
        self.thresholds_: Optional[np.ndarray] = None
        self.binary_threshold_: float = 0.5
        self.feature_dim_: int = 0
        self.train_seconds_: float = 0.0

    def _transform(
        self,
        texts: Sequence[str],
        emb_blocks: Sequence[np.ndarray],
        *,
        fit: bool,
    ) -> np.ndarray:
        cols: List[np.ndarray] = []
        if self.mode == "hybrid_svd_emb":
            if fit:
                self.vectorizer_ = _make_word_char_vectorizer()
                x_sp = self.vectorizer_.fit_transform(texts)
                k = max(2, min(self.svd_components, x_sp.shape[0] - 1, x_sp.shape[1] - 1))
                self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
                x_svd = self.svd_.fit_transform(x_sp)
                self.svd_scaler_ = StandardScaler()
                cols.append(self.svd_scaler_.fit_transform(x_svd).astype(np.float32))
            else:
                assert self.vectorizer_ is not None and self.svd_ is not None and self.svd_scaler_ is not None
                cols.append(self.svd_scaler_.transform(self.svd_.transform(self.vectorizer_.transform(texts))).astype(np.float32))

        if fit:
            self.emb_means_ = []
            for blk in emb_blocks:
                u, mu = center_and_l2_normalize(blk, None)
                self.emb_means_.append(mu)
                cols.append(u)
        else:
            for blk, mu in zip(emb_blocks, self.emb_means_):
                u, _ = center_and_l2_normalize(blk, mu)
                cols.append(u)

        cues = extract_linguistic_features(texts)
        if fit:
            self.cue_scaler_ = StandardScaler()
            cols.append(self.cue_scaler_.fit_transform(cues).astype(np.float32))
        else:
            assert self.cue_scaler_ is not None
            cols.append(self.cue_scaler_.transform(cues).astype(np.float32))

        return np.hstack(cols).astype(np.float32)

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
        *,
        train_emb_blocks: Sequence[np.ndarray],
        val_emb_blocks: Sequence[np.ndarray],
    ) -> "EmbeddingAndHybridXGBoostClassifier":
        xgb = require_xgboost()
        t0 = time.perf_counter()
        self.labels_ = list(class_names)
        X_tr = self._transform(train_texts, train_emb_blocks, fit=True)
        X_va = self._transform(val_texts, val_emb_blocks, fit=False)
        self.feature_dim_ = int(X_tr.shape[1])

        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        common = dict(
            tree_method="hist",
            n_estimators=180,
            max_depth=4,
            learning_rate=0.06,
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=2.5,
            early_stopping_rounds=20,
            n_jobs=self.n_jobs,
        )
        self.estimators_ = []
        self.constant_probs_ = {}

        if self.multilabel:
            val_probs = np.zeros((len(val_texts), len(self.labels_)), dtype=np.float32)
            taus = np.full(len(self.labels_), 0.5, dtype=np.float32)
            for col in range(len(self.labels_)):
                yc_tr = y_tr[:, col].astype(np.int32)
                yc_va = y_va[:, col].astype(np.int32)
                pos = int(yc_tr.sum())
                neg = len(yc_tr) - pos
                if pos == 0 or neg == 0:
                    self.estimators_.append(None)
                    cp = float(pos / max(len(yc_tr), 1))
                    self.constant_probs_[col] = cp
                    val_probs[:, col] = cp
                    continue
                spw = min(6.0, math.sqrt(neg / float(max(pos, 1))))
                est = xgb.XGBClassifier(
                    **common,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=spw,
                    random_state=self.random_state + col,
                )
                est.fit(X_tr, yc_tr, eval_set=[(X_va if int(yc_va.sum()) > 0 else X_tr, yc_va if int(yc_va.sum()) > 0 else yc_tr)], verbose=False)
                self.estimators_.append(est)
                val_probs[:, col] = est.predict_proba(X_va)[:, 1]
                taus[col] = _tune_positive_f1_threshold(yc_va, val_probs[:, col])
            self.thresholds_ = taus
        elif len(self.labels_) == 2:
            pos = max(int((y_tr == 1).sum()), 1)
            neg = max(int((y_tr == 0).sum()), 1)
            est = xgb.XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=min(6.0, math.sqrt(neg / float(pos))),
                random_state=self.random_state,
            )
            est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            self.estimators_ = [est]
            vp = est.predict_proba(X_va)[:, 1]
            self.binary_threshold_ = _tune_binary_threshold(y_va, vp)
        else:
            sw = _balanced_sample_weights(y_tr, max_weight=6.0)
            est = xgb.XGBClassifier(
                **common,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(self.labels_),
                random_state=self.random_state,
            )
            est.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
            self.estimators_ = [est]

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        emb_blocks: Sequence[np.ndarray],
    ) -> np.ndarray:
        X = self._transform(texts, emb_blocks, fit=False)
        if self.multilabel:
            out = np.zeros((len(texts), len(self.labels_)), dtype=np.float64)
            for col, est in enumerate(self.estimators_):
                if est is None:
                    out[:, col] = self.constant_probs_.get(col, 0.0)
                else:
                    out[:, col] = est.predict_proba(X)[:, 1]
            return out
        return np.asarray(self.estimators_[0].predict_proba(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. Neural Classifiers: Dense Embedding MLP & Feature-Gated Hybrid MLP
# ---------------------------------------------------------------------------
@dataclass
class HybridMLPConfig:
    hidden_dim: int = 192
    svd_components: int = 128
    dropout: float = 0.35
    learning_rate: float = 1.5e-3
    weight_decay: float = 2e-2
    epochs: int = 28
    batch_size: int = 32
    patience: int = 7
    random_state: int = 42


class HybridGatedTFIDFEmbeddingMLP:
    """PyTorch MLP supporting `embedding_only` or `hybrid_gated` (TF-IDF branch + Embedding branch).

    In `hybrid_gated` mode:
      - Branch 1 encodes fold-fitted `SVD(Word+Char TF-IDF) + 14 linguistic cues` -> `h_lex`
      - Branch 2 encodes fold-centered L2-normalized dense embeddings -> `h_emb`
      - Learned sigmoid feature gate `g = sigmoid(W_g [h_lex || h_emb])` adaptively blends
        lexical vs. semantic representations (`g * h_lex + (1 - g) * h_emb + 0.20 * (h_lex * h_emb)`),
        exposing `mean_lexical_gate_weight_` (`0 = pure embedding`, `1 = pure TF-IDF`) for audit!
    """

    def __init__(
        self,
        *,
        mode: Literal["embedding_only", "hybrid_gated"] = "hybrid_gated",
        multilabel: bool = False,
        config: Optional[HybridMLPConfig] = None,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for HybridGatedTFIDFEmbeddingMLP.")
        self.mode = mode
        self.multilabel = multilabel
        self.config = config or HybridMLPConfig()

        self.vectorizer_: Optional[FeatureUnion] = None
        self.svd_: Optional[TruncatedSVD] = None
        self.lex_scaler_: Optional[StandardScaler] = None
        self.emb_means_: List[np.ndarray] = []
        self.net_: Any = None
        self.device_: Any = None
        self.labels_: List[str] = []
        self.thresholds_: Optional[np.ndarray] = None
        self.binary_threshold_: float = 0.5
        self.temperature_: float = 1.0
        self.param_count_: int = 0
        self.feature_dim_: int = 0
        self.mean_lexical_gate_weight_: float = 0.0
        self.train_seconds_: float = 0.0

    def _prepare_inputs(
        self,
        texts: Sequence[str],
        emb_blocks: Sequence[np.ndarray],
        *,
        fit: bool,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if fit:
            self.emb_means_ = []
            eb_list = []
            for blk in emb_blocks:
                u, mu = center_and_l2_normalize(blk, None)
                self.emb_means_.append(mu)
                eb_list.append(u)
        else:
            eb_list = [center_and_l2_normalize(blk, mu)[0] for blk, mu in zip(emb_blocks, self.emb_means_)]
        x_emb = np.hstack(eb_list).astype(np.float32)

        if self.mode == "embedding_only":
            return None, x_emb

        cues = extract_linguistic_features(texts)
        if fit:
            self.vectorizer_ = _make_word_char_vectorizer()
            x_sp = self.vectorizer_.fit_transform(texts)
            k = max(4, min(self.config.svd_components, x_sp.shape[0] - 1, x_sp.shape[1] - 1))
            self.svd_ = TruncatedSVD(n_components=k, random_state=self.config.random_state)
            x_svd = self.svd_.fit_transform(x_sp)
            raw_lex = np.hstack([x_svd, cues]).astype(np.float32)
            self.lex_scaler_ = StandardScaler()
            x_lex = self.lex_scaler_.fit_transform(raw_lex).astype(np.float32)
        else:
            assert self.vectorizer_ is not None and self.svd_ is not None and self.lex_scaler_ is not None
            x_svd = self.svd_.transform(self.vectorizer_.transform(texts))
            x_lex = self.lex_scaler_.transform(np.hstack([x_svd, cues])).astype(np.float32)
        return x_lex, x_emb

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
        *,
        train_emb_blocks: Sequence[np.ndarray],
        val_emb_blocks: Sequence[np.ndarray],
    ) -> "HybridGatedTFIDFEmbeddingMLP":
        import torch
        from torch import nn

        t0 = time.perf_counter()
        cfg = self.config
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_state)

        self.labels_ = list(class_names)
        n_classes = len(self.labels_)
        x_lex_tr, x_emb_tr = self._prepare_inputs(train_texts, train_emb_blocks, fit=True)
        x_lex_va, x_emb_va = self._prepare_inputs(val_texts, val_emb_blocks, fit=False)
        self.feature_dim_ = int(x_emb_tr.shape[1] + (x_lex_tr.shape[1] if x_lex_tr is not None else 0))

        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = dev
        hdim = cfg.hidden_dim

        if self.mode == "embedding_only":
            class _EmbNet(nn.Module):
                def __init__(self, d_in: int, h: int, d_out: int, p: float) -> None:
                    super().__init__()
                    self.in_proj = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
                    self.res = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
                    self.head = nn.Linear(h, d_out)

                def forward(self, _lex: Optional[torch.Tensor], emb: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    z = self.in_proj(emb)
                    z = z + 0.5 * self.res(z)
                    return self.head(z), None

            net: nn.Module = _EmbNet(x_emb_tr.shape[1], hdim, n_classes, cfg.dropout).to(dev)
        else:
            assert x_lex_tr is not None

            class _GatedHybridNet(nn.Module):
                def __init__(self, d_lex: int, d_emb: int, h: int, d_out: int, p: float) -> None:
                    super().__init__()
                    self.lex_enc = nn.Sequential(nn.Linear(d_lex, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
                    self.emb_enc = nn.Sequential(nn.Linear(d_emb, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
                    self.gate = nn.Linear(h * 2, h)
                    nn.init.zeros_(self.gate.weight)
                    nn.init.constant_(self.gate.bias, -0.4)  # mild initial prior toward semantic branch (~0.40 lex / 0.60 emb)
                    self.interact = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(p))
                    self.norm = nn.LayerNorm(h)
                    self.head = nn.Linear(h, d_out)

                def forward(self, lex: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    h_l = self.lex_enc(lex)
                    h_e = self.emb_enc(emb)
                    g = torch.sigmoid(self.gate(torch.cat([h_l, h_e], dim=-1)))
                    fused = self.norm(g * h_l + (1.0 - g) * h_e + 0.20 * self.interact(h_l * h_e))
                    return self.head(fused), g

            net = _GatedHybridNet(x_lex_tr.shape[1], x_emb_tr.shape[1], hdim, n_classes, cfg.dropout).to(dev)

        self.param_count_ = sum(int(p.numel()) for p in net.parameters() if p.requires_grad)

        if self.multilabel:
            pos = np.maximum(y_tr.sum(axis=0), 1.0)
            neg = np.maximum(len(y_tr) - pos, 1.0)
            pw = torch.tensor(np.clip(np.sqrt(neg / pos), 1.0, 5.0), dtype=torch.float32, device=dev)
            criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
            cw = np.clip(len(y_tr) / np.maximum(n_classes * counts, 1.0), 0.5, 4.5)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), label_smoothing=0.03)

        opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.learning_rate * 0.05)

        t_emb_tr = torch.tensor(x_emb_tr, dtype=torch.float32, device=dev)
        t_lex_tr = torch.tensor(x_lex_tr, dtype=torch.float32, device=dev) if x_lex_tr is not None else None
        t_y_tr = torch.tensor(y_tr, dtype=torch.float32 if self.multilabel else torch.long, device=dev)

        t_emb_va = torch.tensor(x_emb_va, dtype=torch.float32, device=dev)
        t_lex_va = torch.tensor(x_lex_va, dtype=torch.float32, device=dev) if x_lex_va is not None else None

        rng = np.random.default_rng(cfg.random_state)
        best_val_score = -1e9
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        stale = 0

        for _ep in range(cfg.epochs):
            net.train()
            perm = rng.permutation(len(train_texts))
            for s in range(0, len(perm), cfg.batch_size):
                idx = perm[s : s + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                logits, _ = net(t_lex_tr[idx] if t_lex_tr is not None else None, t_emb_tr[idx])
                loss = criterion(logits, t_y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                va_logits, _ = net(t_lex_va, t_emb_va)
                va_l_np = va_logits.cpu().numpy()
            if self.multilabel:
                vp = _sigmoid_np(va_l_np)
                taus_ep = _calibrate_multilabel_thresholds(y_va, vp, self.labels_)
                t_arr = np.array([taus_ep[l] for l in self.labels_])
                pr = (vp > t_arr[None, :]).astype(int)
                f1s = []
                for j in range(n_classes):
                    tp = int(((y_va[:, j] == 1) & (pr[:, j] == 1)).sum())
                    fp = int(((y_va[:, j] == 0) & (pr[:, j] == 1)).sum())
                    fn = int(((y_va[:, j] == 1) & (pr[:, j] == 0)).sum())
                    d = 2 * tp + fp + fn
                    if int(y_va[:, j].sum()) > 0 or (tp + fp) > 0:
                        f1s.append((2.0 * tp / d) if d > 0 else 0.0)
                v_score = float(np.mean(f1s)) if f1s else 0.0
            else:
                vp = _softmax_np(va_l_np)
                pr = np.argmax(vp, axis=1)
                f1s = []
                for c in range(n_classes):
                    tp = int(((y_va == c) & (pr == c)).sum())
                    fp = int(((y_va != c) & (pr == c)).sum())
                    fn = int(((y_va == c) & (pr != c)).sum())
                    if int((y_va == c).sum()) == 0 and (tp + fp) == 0:
                        continue
                    d = 2 * tp + fp + fn
                    f1s.append((2.0 * tp / d) if d > 0 else 0.0)
                v_score = float(np.mean(f1s)) if f1s else 0.0

            if v_score > best_val_score + 1e-4:
                best_val_score = v_score
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        net.eval()
        self.net_ = net

        with torch.no_grad():
            va_logits, va_gate = net(t_lex_va, t_emb_va)
            va_l_np = va_logits.cpu().numpy()
            if va_gate is not None:
                self.mean_lexical_gate_weight_ = round(float(va_gate.mean().item()), 4)

        self.temperature_ = _calibrate_temperature_np(va_l_np, y_va, self.multilabel)
        cal_va = _sigmoid_np(va_l_np / self.temperature_) if self.multilabel else _softmax_np(va_l_np / self.temperature_)
        if self.multilabel:
            tuned = _calibrate_multilabel_thresholds(y_va, cal_va, self.labels_)
            self.thresholds_ = np.array([tuned[lbl] for lbl in self.labels_], dtype=np.float64)
        elif n_classes == 2:
            self.binary_threshold_ = _tune_binary_threshold(y_va, cal_va[:, 1], low=0.25, high=0.75)

        self.train_seconds_ = round(time.perf_counter() - t0, 3)
        return self

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        emb_blocks: Sequence[np.ndarray],
    ) -> np.ndarray:
        import torch

        x_lex, x_emb = self._prepare_inputs(texts, emb_blocks, fit=False)
        self.net_.eval()
        with torch.no_grad():
            t_emb = torch.tensor(x_emb, dtype=torch.float32, device=self.device_)
            t_lex = torch.tensor(x_lex, dtype=torch.float32, device=self.device_) if x_lex is not None else None
            logits, _ = self.net_(t_lex, t_emb)
            l_np = logits.cpu().numpy() / max(self.temperature_, 1e-6)
        probs = _sigmoid_np(l_np) if self.multilabel else _softmax_np(l_np)
        return np.asarray(probs, dtype=np.float64)

    def cleanup(self) -> None:
        self.net_ = None
        gc.collect()
        if HAS_TORCH:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
