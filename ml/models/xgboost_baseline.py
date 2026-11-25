"""Sprint P1-C — Experimental XGBoost Baseline Across All 5 RCI NLP Tasks.

Strictly experimental architecture (never replaces production MuRIL-DAPT-WWM / XLM-R,
never touches REQUIRED_ARCHITECTURES, never modifies frozen test labels).

Supports two leak-free feature representations (fit strictly inside each training fold):
  1. ``sparse_tfidf``: Word (1,2)-grams + Char ``char_wb`` (3,5)-grams CSR sparse matrix
     (never densified!) fed directly to CPU ``tree_method="hist"`` XGBoost.
  2. ``svd_linguistic``: TruncatedSVD-reduced Word+Char TF-IDF + 14-dim standardized
     linguistic/multilingual/discourse cue vector.

Task formulations matching canonical RCI taxonomy:
  - ``issue``: 8-label multi-label (8 independent binary XGBoost estimators + OOF/val threshold tuning)
  - ``solution``: binary (``not_solution`` vs ``solution``, dampened ``scale_pos_weight``, val threshold)
  - ``sentiment``: 3-class multiclass (``multi:softprob``, balanced sample weights)
  - ``response``: canonical 6-class order multiclass (``multi:softprob``, balanced sample weights)
  - ``emotion``: 6-label multi-label (6 independent binary XGBoost estimators + val threshold tuning)
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


def require_xgboost() -> Any:
    """Import xgboost cleanly or raise an actionable optional-dependency error."""
    try:
        import xgboost as xgb

        return xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is an optional experimental dependency and is not installed in the base "
            "runtime. Install optional ML extras with:\n"
            "  pip install -r backend/requirements-ml.txt\n"
            "(Keeps production/Vercel base image slim when offline training is unused.)"
        ) from exc


_HINGLISH_MARKERS_RE = re.compile(
    r"\b(hai|hain|nahi|nhi|kya|kyun|bhai|yaar|toh|bhi|mein|aur|lekin|accha|thik|karo|karna|gaya|paisa|bohot|bahut|abhi|phir|mtlb|rha|hoga)\b",
    re.IGNORECASE,
)
_ACTION_CUE_RE = re.compile(
    r"\b(email|mail|call|apply|check|contact|aicte|refund|portal|submit|ticket|follow|guideline|loan|bank|linkedin|compare)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(not|no|never|don't|didn't|won't|isn't|nhi|nahi|mat)\b", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(r"\b(i|my|me|mine|we|our|mera|maine|mujhe|hum)\b", re.IGNORECASE)
_ADVICE_MODAL_RE = re.compile(r"\b(should|must|recommend|suggest|better to|try to|avoid|prefer|chahiye)\b", re.IGNORECASE)
_COUNTER_RE = re.compile(r"\b(actually|however|not true|disagree|wrong|incorrect|but|false)\b", re.IGNORECASE)


def extract_linguistic_features(texts: Sequence[str]) -> np.ndarray:
    """Extract 14 compact, zero-leakage domain/multilingual/structural cues per text."""
    feats = np.zeros((len(texts), 14), dtype=np.float32)
    for i, raw in enumerate(texts):
        s = str(raw or "")
        n_chars = len(s)
        words = s.split()
        n_words = len(words)
        alpha = sum(1 for ch in s if ch.isalpha())
        digits = sum(1 for ch in s if ch.isdigit())
        upper = sum(1 for ch in s if ch.isupper())
        non_ascii = sum(1 for ch in s if ord(ch) > 127)
        indic_chars = sum(1 for ch in s if 0x0900 <= ord(ch) <= 0x0D7F)
        feats[i, 0] = min(n_chars, 1000) / 1000.0
        feats[i, 1] = min(n_words, 200) / 200.0
        feats[i, 2] = (n_chars / max(n_words, 1)) / 12.0
        feats[i, 3] = s.count("?") / max(n_words, 1)
        feats[i, 4] = s.count("!") / max(n_words, 1)
        feats[i, 5] = digits / max(n_chars, 1)
        feats[i, 6] = upper / max(alpha, 1)
        feats[i, 7] = non_ascii / max(n_chars, 1)
        feats[i, 8] = 1.0 if indic_chars > 0 else 0.0
        feats[i, 9] = min(len(_HINGLISH_MARKERS_RE.findall(s)), 5) / 5.0
        feats[i, 10] = min(len(_ACTION_CUE_RE.findall(s)), 4) / 4.0
        feats[i, 11] = min(len(_NEGATION_RE.findall(s)), 4) / 4.0
        feats[i, 12] = min(len(_FIRST_PERSON_RE.findall(s)), 4) / 4.0
        feats[i, 13] = 1.0 if (_ADVICE_MODAL_RE.search(s) or _COUNTER_RE.search(s)) else 0.0
    return feats


def _encode_targets(
    targets: Sequence[Any], class_names: Sequence[str], multilabel: bool
) -> np.ndarray:
    c2i = {c: i for i, c in enumerate(class_names)}
    if multilabel:
        mat = np.zeros((len(targets), len(class_names)), dtype=np.int32)
        for row_i, ys in enumerate(targets):
            for lbl in ys:
                if lbl in c2i:
                    mat[row_i, c2i[lbl]] = 1
        return mat
    return np.array([c2i[str(y)] for y in targets], dtype=np.int64)


def _balanced_sample_weights(y: np.ndarray, max_weight: float = 6.0) -> np.ndarray:
    counts = np.bincount(y, minlength=int(y.max()) + 1 if len(y) else 1)
    n = max(len(y), 1)
    n_present = max(int((counts > 0).sum()), 1)
    weights = np.ones(len(y), dtype=np.float32)
    for cls, cnt in enumerate(counts):
        if cnt > 0:
            w = min(max_weight, n / float(n_present * cnt))
            weights[y == cls] = float(w)
    return weights


def _tune_binary_threshold(y_true: np.ndarray, probs: np.ndarray, low: float = 0.20, high: float = 0.80) -> float:
    best_tau = 0.5
    best_f1 = -1.0
    for step in range(int(round((high - low) / 0.025)) + 1):
        tau = round(low + 0.025 * step, 3)
        pred = (probs >= tau).astype(int)
        f1s = []
        for cls in (0, 1):
            tp = int(((y_true == cls) & (pred == cls)).sum())
            fp = int(((y_true != cls) & (pred == cls)).sum())
            fn = int(((y_true == cls) & (pred != cls)).sum())
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        mf1 = 0.5 * sum(f1s)
        if (mf1 > best_f1 + 1e-12) or (abs(mf1 - best_f1) <= 1e-12 and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_f1 = mf1
            best_tau = tau
    return float(best_tau)


def _tune_positive_f1_threshold(y_true: np.ndarray, probs: np.ndarray, low: float = 0.20, high: float = 0.75) -> float:
    if int(y_true.sum()) == 0:
        return 0.5
    best_tau = 0.5
    best_f1 = -1.0
    for step in range(int(round((high - low) / 0.025)) + 1):
        tau = round(low + 0.025 * step, 3)
        pred = (probs >= tau).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_f1 = f1
            best_tau = tau
    return float(best_tau)


@dataclass
class XGBoostConfig:
    feature_mode: Literal["sparse_tfidf", "svd_linguistic"] = "sparse_tfidf"
    tree_method: str = "hist"
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.06
    min_child_weight: int = 3
    subsample: float = 0.80
    colsample_bytree: float = 0.70
    reg_alpha: float = 0.50
    reg_lambda: float = 2.00
    early_stopping_rounds: int = 25
    max_word_features: int = 12000
    max_char_features: int = 18000
    svd_components: int = 128
    max_pos_weight: float = 6.0
    random_state: int = 42
    n_jobs: int = 4


class XGBoostBaselineClassifier:
    """Experimental XGBoost classifier supporting binary, multiclass, and multi-label RCI tasks."""

    def __init__(
        self,
        *,
        task_name: str,
        multilabel: bool = False,
        config: Optional[XGBoostConfig] = None,
    ) -> None:
        require_xgboost()
        self.task_name = str(task_name)
        self.multilabel = bool(multilabel)
        self.config = config or XGBoostConfig()
        self.labels_: List[str] = []
        self.vectorizer_: Optional[FeatureUnion] = None
        self.svd_: Optional[TruncatedSVD] = None
        self.ling_scaler_: Optional[StandardScaler] = None
        self.estimators_: List[Any] = []
        self.constant_probs_: Dict[int, float] = {}
        self.thresholds_: Optional[np.ndarray] = None
        self.binary_threshold_: float = 0.5
        self.train_seconds_: float = 0.0
        self.best_iterations_: List[int] = []

    def _build_vectorizer(self) -> FeatureUnion:
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=self.config.max_word_features,
                        min_df=2,
                        sublinear_tf=True,
                        token_pattern=r"(?u)[A-Za-z0-9\u0900-\u0d7f]{2,}",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=self.config.max_char_features,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    def _fit_transform_features(self, train_texts: Sequence[str], val_texts: Sequence[str]) -> Tuple[Any, Any]:
        self.vectorizer_ = self._build_vectorizer()
        X_tr_sp = self.vectorizer_.fit_transform(train_texts).astype(np.float32)
        X_va_sp = self.vectorizer_.transform(val_texts).astype(np.float32)

        if self.config.feature_mode == "sparse_tfidf":
            return X_tr_sp, X_va_sp

        if self.config.feature_mode == "svd_linguistic":
            max_comp = max(2, min(self.config.svd_components, X_tr_sp.shape[0] - 2, X_tr_sp.shape[1] - 1))
            self.svd_ = TruncatedSVD(n_components=max_comp, random_state=self.config.random_state)
            X_tr_svd = self.svd_.fit_transform(X_tr_sp).astype(np.float32)
            X_va_svd = self.svd_.transform(X_va_sp).astype(np.float32)

            self.ling_scaler_ = StandardScaler()
            L_tr = self.ling_scaler_.fit_transform(extract_linguistic_features(train_texts)).astype(np.float32)
            L_va = self.ling_scaler_.transform(extract_linguistic_features(val_texts)).astype(np.float32)
            return np.hstack([X_tr_svd, L_tr]), np.hstack([X_va_svd, L_va])

        raise ValueError(f"Unsupported feature_mode={self.config.feature_mode!r}")

    def transform_features(self, texts: Sequence[str]) -> Any:
        if self.vectorizer_ is None:
            raise RuntimeError("XGBoostBaselineClassifier must be fit or loaded before calling transform_features().")
        X_sp = self.vectorizer_.transform(texts).astype(np.float32)
        if self.config.feature_mode == "sparse_tfidf":
            assert sparse.issparse(X_sp)
            return X_sp
        assert self.svd_ is not None and self.ling_scaler_ is not None
        X_svd = self.svd_.transform(X_sp).astype(np.float32)
        L = self.ling_scaler_.transform(extract_linguistic_features(texts)).astype(np.float32)
        return np.hstack([X_svd, L])

    def _make_xgb(self, **extra: Any) -> Any:
        xgb = require_xgboost()
        params: Dict[str, Any] = {
            "tree_method": self.config.tree_method,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "min_child_weight": self.config.min_child_weight,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "verbosity": 0,
        }
        params.update(extra)
        return xgb.XGBClassifier(**params)

    def fit(
        self,
        train_texts: Sequence[str],
        train_targets: Sequence[Any],
        val_texts: Sequence[str],
        val_targets: Sequence[Any],
        class_names: Sequence[str],
    ) -> "XGBoostBaselineClassifier":
        t0 = time.perf_counter()
        self.labels_ = list(class_names)
        self.estimators_ = []
        self.constant_probs_ = {}
        self.best_iterations_ = []

        X_tr, X_va = self._fit_transform_features(train_texts, val_texts)
        y_tr = _encode_targets(train_targets, self.labels_, self.multilabel)
        y_va = _encode_targets(val_targets, self.labels_, self.multilabel)

        if self.multilabel:
            val_probs = np.zeros((len(val_texts), len(self.labels_)), dtype=np.float64)
            taus = np.full(len(self.labels_), 0.5, dtype=np.float64)
            for col in range(len(self.labels_)):
                yt_c = y_tr[:, col]
                yv_c = y_va[:, col]
                n_pos = int(yt_c.sum())
                n_neg = len(yt_c) - n_pos
                if n_pos == 0 or n_neg == 0:
                    self.estimators_.append(None)
                    self.constant_probs_[col] = float(n_pos / max(len(yt_c), 1))
                    val_probs[:, col] = self.constant_probs_[col]
                    continue

                spw = float(min(self.config.max_pos_weight, max(1.0, n_neg / float(n_pos))))
                has_val_both = len(np.unique(yv_c)) >= 2
                fit_kw: Dict[str, Any] = {"verbose": False}
                if has_val_both:
                    clf = self._make_xgb(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=spw,
                        early_stopping_rounds=self.config.early_stopping_rounds,
                    )
                    clf.fit(X_tr, yt_c, eval_set=[(X_va, yv_c)], **fit_kw)
                    best_it = getattr(clf, "best_iteration", None)
                    if best_it is not None:
                        self.best_iterations_.append(int(best_it))
                else:
                    clf = self._make_xgb(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=spw,
                    )
                    clf.fit(X_tr, yt_c, **fit_kw)
                self.estimators_.append(clf)
                col_p = clf.predict_proba(X_va)[:, 1]
                val_probs[:, col] = col_p
                taus[col] = _tune_positive_f1_threshold(yv_c, col_p)
            self.thresholds_ = taus
        elif len(self.labels_) == 2:
            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            spw = float(min(self.config.max_pos_weight, max(1.0, n_neg / max(n_pos, 1))))
            has_val_both = len(np.unique(y_va)) >= 2
            if has_val_both:
                clf = self._make_xgb(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=spw,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                best_it = getattr(clf, "best_iteration", None)
                if best_it is not None:
                    self.best_iterations_.append(int(best_it))
            else:
                clf = self._make_xgb(objective="binary:logistic", eval_metric="logloss", scale_pos_weight=spw)
                clf.fit(X_tr, y_tr, verbose=False)
            self.estimators_.append(clf)
            va_p = clf.predict_proba(X_va)[:, 1]
            self.binary_threshold_ = _tune_binary_threshold(y_va, va_p)
        else:
            # Multiclass (3-class sentiment or canonical 6-class response)
            sw_tr = _balanced_sample_weights(y_tr, max_weight=self.config.max_pos_weight)
            all_cls_in_val = set(np.unique(y_va)) == set(range(len(self.labels_)))
            all_cls_in_tr = len(np.unique(y_tr)) == len(self.labels_)
            if all_cls_in_tr and all_cls_in_val:
                clf = self._make_xgb(
                    objective="multi:softprob",
                    num_class=len(self.labels_),
                    eval_metric="mlogloss",
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                clf.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
                best_it = getattr(clf, "best_iteration", None)
                if best_it is not None:
                    self.best_iterations_.append(int(best_it))
            else:
                clf = self._make_xgb(
                    objective="multi:softprob",
                    num_class=len(self.labels_),
                    eval_metric="mlogloss",
                )
                clf.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
            self.estimators_.append(clf)

        self.train_seconds_ = round(time.perf_counter() - t0, 4)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.transform_features(texts)
        if self.multilabel:
            probs = np.zeros((len(texts), len(self.labels_)), dtype=np.float64)
            for col in range(len(self.labels_)):
                est = self.estimators_[col]
                if est is None:
                    probs[:, col] = self.constant_probs_.get(col, 0.0)
                else:
                    probs[:, col] = est.predict_proba(X)[:, 1]
            return np.clip(probs, 0.0, 1.0)
        est = self.estimators_[0]
        raw_p = np.asarray(est.predict_proba(X), dtype=np.float64)
        row_sums = np.maximum(raw_p.sum(axis=1, keepdims=True), 1e-12)
        return np.clip(raw_p / row_sums, 0.0, 1.0)

    def predict(self, texts: Sequence[str]) -> List[Any]:
        probs = self.predict_proba(texts)
        if self.multilabel:
            taus = self.thresholds_ if self.thresholds_ is not None else np.full(len(self.labels_), 0.5)
            out: List[List[str]] = []
            for row in probs:
                picked = [self.labels_[i] for i, p in enumerate(row) if p >= taus[i]]
                if not picked:
                    picked = [self.labels_[int(np.argmax(row))]]
                out.append(sorted(picked))
            return out
        if len(self.labels_) == 2:
            return [self.labels_[1] if row[1] >= self.binary_threshold_ else self.labels_[0] for row in probs]
        return [self.labels_[int(np.argmax(row))] for row in probs]

    def save(self, model_dir: Path) -> int:
        """Persist native XGBoost booster JSON(s) + TF-IDF/SVD preprocessor cleanly; returns total bytes."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for idx, est in enumerate(self.estimators_):
            if est is not None:
                est.save_model(str(model_dir / f"booster_{idx}.json"))
        joblib.dump(
            {
                "vectorizer": self.vectorizer_,
                "svd": self.svd_,
                "ling_scaler": self.ling_scaler_,
            },
            model_dir / "features.joblib",
            compress=3,
        )
        meta = {
            "task_name": self.task_name,
            "multilabel": self.multilabel,
            "labels": self.labels_,
            "config": asdict(self.config),
            "constant_probs": {str(k): v for k, v in self.constant_probs_.items()},
            "thresholds": self.thresholds_.tolist() if self.thresholds_ is not None else None,
            "binary_threshold": self.binary_threshold_,
            "n_estimators_saved": len(self.estimators_),
            "train_seconds": self.train_seconds_,
            "best_iterations": self.best_iterations_,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())

    @classmethod
    def load(cls, model_dir: Path) -> "XGBoostBaselineClassifier":
        xgb = require_xgboost()
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        cfg = XGBoostConfig(**meta["config"])
        obj = cls(task_name=meta["task_name"], multilabel=meta["multilabel"], config=cfg)
        obj.labels_ = list(meta["labels"])
        obj.constant_probs_ = {int(k): float(v) for k, v in (meta.get("constant_probs") or {}).items()}
        obj.thresholds_ = np.array(meta["thresholds"], dtype=np.float64) if meta.get("thresholds") is not None else None
        obj.binary_threshold_ = float(meta.get("binary_threshold", 0.5))
        obj.train_seconds_ = float(meta.get("train_seconds", 0.0))
        obj.best_iterations_ = list(meta.get("best_iterations") or [])

        feat_bundle = joblib.load(model_dir / "features.joblib")
        obj.vectorizer_ = feat_bundle["vectorizer"]
        obj.svd_ = feat_bundle.get("svd")
        obj.ling_scaler_ = feat_bundle.get("ling_scaler")

        n_est = int(meta["n_estimators_saved"])
        obj.estimators_ = []
        for idx in range(n_est):
            b_path = model_dir / f"booster_{idx}.json"
            if b_path.is_file():
                clf = xgb.XGBClassifier()
                clf.load_model(str(b_path))
                obj.estimators_.append(clf)
            else:
                obj.estimators_.append(None)
        return obj
