"""Multilingual sentence & document embedding backends for BERTopic and Issue–Solution linking.

Evaluates multiple multilingual embedding families with explicit availability,
licensing, inference speed, and Indian/code-mixed language coverage metadata:
- `intfloat/multilingual-e5-small` (MIT, 384-dim, 100 languages incl. EN/HI/BN/OR; requires 'passage: ' prefix)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 384-dim, 50+ languages)
- `muril_dapt_wwm` (Apache-2.0 local DAPT checkpoint, 768-dim mean-pooled — evaluated honestly as a
  non-contrastive MLM comparator; never assumed to beat sentence-contrastive embeddings!)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = Path(os.environ.get("RCI_MODEL_CACHE_DIR") or (ROOT / ".cache" / "models"))
if not _DEFAULT_CACHE.exists() and Path("D:/AI_Project_Transformer_Temp").is_dir():
    _DEFAULT_CACHE = Path("D:/AI_Project_Transformer_Temp")
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EMBED_CACHE_DIR = _DEFAULT_CACHE / "v6_experiments" / "embeddings"
EMBEDDING_CACHE_SCHEMA_VERSION = "v2.0"
DEFAULT_PREPROCESSING_VERSION = "rci_preproc_v6.3"
EMBEDDING_PIPELINE_VERSION = "rci_bertopic_v6.3"
_HF_MEAN_POOL_MAX_LENGTH = 256

# Single-slot process-local encoder cache avoids reloading the same weights back-to-back
# (e.g. ContextualIssueSolutionMapper encoding queries then passages) and releases RAM cleanly.
_ACTIVE_ENCODER: Dict[str, Any] = {"key": None, "obj": None}

EMBEDDING_CATALOG: Dict[str, Dict[str, Any]] = {
    "multilingual_e5_small": {
        "model_name": "intfloat/multilingual-e5-small",
        "backend": "sentence_transformers",
        "dim": 384,
        "license": "MIT",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "languages": "100 languages (EN, HI, BN, OR, Hinglish/Romanized Indic)",
        "contrastive_pretrained": True,
        "notes": "Contrastively trained multilingual sentence encoder (XLM-R backbone); supports Indian scripts & code-mix.",
    },
    "multilingual_minilm": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "backend": "sentence_transformers",
        "dim": 384,
        "license": "Apache-2.0",
        "passage_prefix": "",
        "query_prefix": "",
        "languages": "50+ languages (EN, HI, BN; Odia not natively in top-50 training mix)",
        "contrastive_pretrained": True,
        "notes": "Distilled multilingual paraphrase encoder; fast GPU/CPU inference.",
    },
    "muril_dapt_wwm": {
        "model_name": str(_DEFAULT_CACHE / "v5_experiments" / "muril_dapt_strict_inductive_wwm_8ep"),
        "backend": "hf_mean_pool",
        "dim": 768,
        "license": "Apache-2.0",
        "passage_prefix": "",
        "query_prefix": "",
        "languages": "17 Indian languages + EN + Romanized Hinglish/Banglish/Odia (strict Reddit WWM DAPT)",
        "contrastive_pretrained": False,
        "notes": "Domain-adapted MLM encoder mean-pooled without contrastive sentence training; included as a rigorous comparator.",
    },
}


def resolve_model_fingerprint(
    spec: Dict[str, Any],
    *,
    model_revision: Optional[str] = None,
) -> str:
    """Return a deterministic model/weight fingerprint for embedding cache invalidation.

    - For local model directories, hashes ``config.json``, ``tokenizer_config.json``, and
      file sizes + first-64KB headers of weight files so weight edits invalidate the cache
      without reading multi-hundred-MB files on every lookup.
    - For HuggingFace Hub IDs, incorporates ``refs/main`` commit hash when present locally
      alongside explicit ``model_revision`` and backend library name/version.
    """
    h = hashlib.sha256()
    model_name = str(spec.get("model_name") or "")
    backend = str(spec.get("backend") or "")
    h.update(f"model={model_name}|backend={backend}|".encode("utf-8"))
    if model_revision:
        h.update(f"explicit_rev={model_revision}|".encode("utf-8"))

    local_dir = Path(model_name)
    if local_dir.is_dir():
        h.update(b"local_dir|")
        for cfg_name in ("config.json", "tokenizer_config.json", "modules.json", "sentence_bert_config.json"):
            cfg_file = local_dir / cfg_name
            if cfg_file.is_file():
                h.update(f"{cfg_name}:".encode("utf-8"))
                h.update(cfg_file.read_bytes())
                h.update(b"|")
        weight_suffixes = {".safetensors", ".bin", ".pt"}
        weight_files = sorted(
            p for p in local_dir.iterdir()
            if p.is_file() and p.suffix.lower() in weight_suffixes
        )
        for wf in weight_files:
            stat = wf.stat()
            h.update(f"{wf.name}:{stat.st_size}:".encode("utf-8"))
            with wf.open("rb") as fh:
                h.update(fh.read(65536))
            h.update(b"|")
    else:
        # Check local HF hub refs/main snapshot revision when available
        hf_home = Path(os.environ.get("HF_HOME") or (_DEFAULT_CACHE / "hf_home"))
        hub_slug = model_name.replace("/", "--")
        ref_file = hf_home / "hub" / f"models--{hub_slug}" / "refs" / "main"
        if ref_file.is_file():
            h.update(f"hf_ref={ref_file.read_text(encoding='utf-8', errors='ignore').strip()}|".encode("utf-8"))

    # Include effective fallback for missing local MuRIL checkpoint
    if backend == "hf_mean_pool" and not local_dir.is_dir():
        h.update(b"fallback=google/muril-base-cased|")

    return h.hexdigest()[:32]


def compute_corpus_digest(
    texts: Sequence[str],
    *,
    order_independent: bool = False,
) -> str:
    """Stream every input text into a collision-resistant SHA-256 digest.

    Uses unambiguous byte-length framing (``<len>:<utf8_bytes>\\n``) so concatenation
    boundaries between documents cannot collide. When ``order_independent=True``,
    encoded document frames are deterministically sorted prior to hashing.
    """
    h = hashlib.sha256()
    h.update(f"n={len(texts)}|order_independent={bool(order_independent)}|".encode("utf-8"))
    if order_independent:
        frames = []
        for t in texts:
            raw = str(t).encode("utf-8", errors="surrogatepass")
            frames.append(f"{len(raw)}:".encode("ascii") + raw + b"\n")
        frames.sort()
        for fr in frames:
            h.update(fr)
    else:
        for t in texts:
            raw = str(t).encode("utf-8", errors="surrogatepass")
            h.update(f"{len(raw)}:".encode("ascii"))
            h.update(raw)
            h.update(b"\n")
    return h.hexdigest()


def compute_embedding_cache_identity(
    embedding_key: str,
    texts: Sequence[str],
    *,
    is_query: bool = False,
    model_revision: Optional[str] = None,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    corpus_version: str = "unspecified",
    pipeline_version: str = EMBEDDING_PIPELINE_VERSION,
    order_independent: bool = False,
) -> Dict[str, Any]:
    """Compute a complete, collision-resistant cache identity covering every input & config field."""
    if embedding_key not in EMBEDDING_CATALOG:
        raise ValueError(f"Unknown embedding_key={embedding_key!r}. Available: {sorted(EMBEDDING_CATALOG)}")

    spec = EMBEDDING_CATALOG[embedding_key]
    mode = "query" if is_query else "passage"
    prefix = str(spec["query_prefix"] if is_query else spec["passage_prefix"])
    corpus_sha256 = compute_corpus_digest(texts, order_independent=order_independent)
    model_fp = resolve_model_fingerprint(spec, model_revision=model_revision)

    identity_payload: Dict[str, Any] = {
        "cache_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "embedding_key": embedding_key,
        "model_name": str(spec["model_name"]),
        "backend": str(spec["backend"]),
        "dim": int(spec["dim"]),
        "model_fingerprint": model_fp,
        "model_revision": str(model_revision or ""),
        "mode": mode,
        "prefix": prefix,
        "normalize_embeddings": True,
        "max_length": _HF_MEAN_POOL_MAX_LENGTH if spec["backend"] == "hf_mean_pool" else "model_default",
        "preprocessing_version": str(preprocessing_version),
        "corpus_version": str(corpus_version),
        "pipeline_version": str(pipeline_version),
        "order_independent": bool(order_independent),
        "n_docs": len(texts),
        "corpus_sha256": corpus_sha256,
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {**identity_payload, "cache_key": cache_key}


def compute_embedding_cache_key(
    embedding_key: str,
    texts: Sequence[str],
    *,
    is_query: bool = False,
    model_revision: Optional[str] = None,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    corpus_version: str = "unspecified",
    pipeline_version: str = EMBEDDING_PIPELINE_VERSION,
    order_independent: bool = False,
) -> str:
    """Convenience wrapper returning the full 64-hex SHA-256 cache key."""
    return str(
        compute_embedding_cache_identity(
            embedding_key,
            texts,
            is_query=is_query,
            model_revision=model_revision,
            preprocessing_version=preprocessing_version,
            corpus_version=corpus_version,
            pipeline_version=pipeline_version,
            order_independent=order_independent,
        )["cache_key"]
    )


def _is_valid_cache_entry(
    cache_path: Path,
    meta_path: Path,
    *,
    expected_identity: Dict[str, Any],
    expected_n_docs: int,
    expected_dim: int,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate strict schema v2 metadata + key equality; never accepts legacy v1 entries."""
    if not (cache_path.is_file() and meta_path.is_file()):
        return False, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if meta.get("cache_schema_version") != EMBEDDING_CACHE_SCHEMA_VERSION:
        return False, None
    if meta.get("cache_key") != expected_identity["cache_key"]:
        return False, None
    if int(meta.get("n_docs", -1)) != expected_n_docs:
        return False, None
    if int(meta.get("dim", -1)) != expected_dim:
        return False, None
    return True, meta


def encode_documents(
    texts: Sequence[str],
    embedding_key: str,
    *,
    is_query: bool = False,
    batch_size: int = 64,
    use_cache: bool = True,
    device: Optional[str] = None,
    model_revision: Optional[str] = None,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    corpus_version: str = "unspecified",
    pipeline_version: str = EMBEDDING_PIPELINE_VERSION,
    order_independent: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute L2-normalized dense embeddings for ``texts`` with v2 deterministic disk caching."""
    if embedding_key not in EMBEDDING_CATALOG:
        raise ValueError(f"Unknown embedding_key={embedding_key!r}. Available: {sorted(EMBEDDING_CATALOG)}")

    spec = EMBEDDING_CATALOG[embedding_key]
    prefix = spec["query_prefix"] if is_query else spec["passage_prefix"]
    normalized_inputs = [str(t or "").strip() for t in texts]
    prefixed_texts = [f"{prefix}{t}" for t in normalized_inputs]

    identity = compute_embedding_cache_identity(
        embedding_key,
        normalized_inputs,
        is_query=is_query,
        model_revision=model_revision,
        preprocessing_version=preprocessing_version,
        corpus_version=corpus_version,
        pipeline_version=pipeline_version,
        order_independent=order_independent,
    )
    cache_key = identity["cache_key"]
    v2_cache_root = (Path(cache_dir) if cache_dir is not None else EMBED_CACHE_DIR) / "v2"
    cache_path = v2_cache_root / f"{embedding_key}_{cache_key[:32]}.npy"
    meta_path = v2_cache_root / f"{embedding_key}_{cache_key[:32]}.meta.json"

    if use_cache:
        ok_meta, cached_meta = _is_valid_cache_entry(
            cache_path,
            meta_path,
            expected_identity=identity,
            expected_n_docs=len(prefixed_texts),
            expected_dim=int(spec["dim"]),
        )
        if ok_meta and cached_meta is not None:
            arr = np.load(cache_path)
            if arr.ndim == 2 and arr.shape == (len(prefixed_texts), int(spec["dim"])):
                meta = {**cached_meta, "cached": True}
                return arr.astype(np.float32), meta

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    if spec["backend"] == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        cache_slot = ("st", spec["model_name"], dev, model_revision or "", id(SentenceTransformer))
        if _ACTIVE_ENCODER["key"] != cache_slot:
            _ACTIVE_ENCODER["key"] = None
            _ACTIVE_ENCODER["obj"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st_kwargs: Dict[str, Any] = {"device": dev}
            if model_revision:
                st_kwargs["revision"] = model_revision
            _ACTIVE_ENCODER["obj"] = SentenceTransformer(spec["model_name"], **st_kwargs)
            _ACTIVE_ENCODER["key"] = cache_slot

        model = _ACTIVE_ENCODER["obj"]
        embs = model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    elif spec["backend"] == "hf_mean_pool":
        from transformers import AutoModel, AutoTokenizer

        model_path = spec["model_name"]
        if not Path(model_path).is_dir():
            model_path = "google/muril-base-cased"
        cache_slot = (model_path, dev, model_revision or "")
        if _ACTIVE_ENCODER["key"] != cache_slot:
            _ACTIVE_ENCODER["key"] = None
            _ACTIVE_ENCODER["obj"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hf_kwargs: Dict[str, Any] = {"revision": model_revision} if model_revision else {}
            tok = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)
            enc = AutoModel.from_pretrained(model_path, **hf_kwargs).to(dev)
            enc.eval()
            _ACTIVE_ENCODER["obj"] = (tok, enc)
            _ACTIVE_ENCODER["key"] = cache_slot

        tok, enc = _ACTIVE_ENCODER["obj"]
        chunks: List[np.ndarray] = []
        amp_enabled = dev == "cuda"
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.no_grad():
            for i in range(0, len(prefixed_texts), batch_size):
                batch = prefixed_texts[i : i + batch_size]
                inp = tok(batch, padding=True, truncation=True, max_length=_HF_MEAN_POOL_MAX_LENGTH, return_tensors="pt")
                inp = {k: v.to(dev) for k, v in inp.items()}
                with torch.amp.autocast(device_type="cuda" if amp_enabled else "cpu", dtype=amp_dtype, enabled=amp_enabled):
                    out = enc(**inp).last_hidden_state
                mask = inp["attention_mask"].unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
                chunks.append(pooled.cpu().numpy())
        embs = np.concatenate(chunks, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported backend: {spec['backend']}")

    elapsed = round(time.perf_counter() - t0, 3)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        **identity,
        "license": spec["license"],
        "device": dev,
        "encode_seconds": elapsed,
        "ms_per_doc": round((elapsed * 1000.0) / max(len(prefixed_texts), 1), 3),
        "cached": False,
    }
    if use_cache:
        v2_cache_root.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return embs, meta


def evaluate_multilingual_embedding_probes(embedding_keys: Sequence[str]) -> Dict[str, Any]:
    """Compare cross-lingual semantic alignment across EN, HI, BN, OR, Hinglish, Banglish, Roman-Odia.

    NOTE: Explicitly labeled as a controlled synthetic cross-lingual diagnostic probe —
    NEVER conflated with real-corpus evaluation! Measures whether semantically equivalent
    student-issue queries across scripts/code-mix align closer to the matching English anchor
    than to non-matching issue anchors (Top-1 Cross-Lingual Retrieval Accuracy & margin).
    """
    anchors = {
        "placements": "How are the summer internships and final placement packages at IIM?",
        "hostel_mess": "Hostel room conditions and mess food quality are terrible on campus.",
        "fees_roi": "MBA tuition fees are very high; is the education loan and ROI actually worth it?",
        "workload": "Academic classes, quizzes, and case-study workload are extremely hectic this term.",
    }
    probes = [
        # (language_group, gold_anchor, text)
        ("hinglish", "placements", "Bhai IIM ABC ka summer placement scene kaisa hai 😭"),
        ("hinglish", "hostel_mess", "Bro hostel life 💀... mess food toh literally mid hai"),
        ("hinglish", "fees_roi", "IIM jaana worth it hai kya? hostel fees bahut high hai"),
        ("hinglish", "workload", "Aaj ka academic workload aur quizzes bahut zyada hectic hai"),
        ("hindi_devanagari", "placements", "इस कॉलेज का प्लेसमेंट और समर इंटर्नशिप कैसा है?"),
        ("hindi_devanagari", "hostel_mess", "हॉस्टल और मेस का खाना बहुत खराब है"),
        ("bengali_script", "hostel_mess", "এই কলেজের হোস্টেল আর মেসের খাবার কেমন?"),
        ("bengali_script", "fees_roi", "আইআইএম-এর ফি অনেক বেশি, লোন নেওয়া কি ঠিক হবে?"),
        ("banglish", "fees_roi", "IIM e giye ki actually ROI pawa jay? fees onek beshi"),
        ("banglish", "placements", "IIM er placement ar internship kemon hoyeche ei bochor?"),
        ("odia_script", "workload", "ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro"),
        ("odia_script", "hostel_mess", "ହଷ୍ଟେଲ ରୁମ୍ ଆଉ ମେସ୍ ଖାଦ୍ୟ ବହୁତ ଖରାପ ଅଛି"),
        ("roman_odia", "workload", "Aaji class bahut hectic thila bhai, assignment load besi"),
        ("roman_odia", "placements", "IIM Sambalpur ra placement package kemiti achi?"),
    ]

    anchor_keys = list(anchors.keys())
    anchor_texts = [anchors[k] for k in anchor_keys]
    probe_texts = [p[2] for p in probes]

    report: Dict[str, Any] = {
        "probe_scope": "controlled_synthetic_multilingual_diagnostic_separate_from_real_corpus",
        "n_probes": len(probes),
        "results": {},
    }
    for ek in embedding_keys:
        a_emb, a_meta = encode_documents(anchor_texts, ek, is_query=False, use_cache=False)
        p_emb, _ = encode_documents(probe_texts, ek, is_query=True, use_cache=False)
        sim = p_emb @ a_emb.T  # cosine similarity (L2-normalized)

        by_lang: Dict[str, List[int]] = {}
        margins: List[float] = []
        correct = 0
        details = []
        for idx, (lg, gold, txt) in enumerate(probes):
            scores = sim[idx]
            pred_idx = int(np.argmax(scores))
            gold_idx = anchor_keys.index(gold)
            ok = int(pred_idx == gold_idx)
            correct += ok
            by_lang.setdefault(lg, []).append(ok)
            gold_sim = float(scores[gold_idx])
            best_other = float(max(scores[j] for j in range(len(anchor_keys)) if j != gold_idx))
            margin = gold_sim - best_other
            margins.append(margin)
            details.append({
                "language_group": lg,
                "gold": gold,
                "predicted": anchor_keys[pred_idx],
                "correct": bool(ok),
                "gold_similarity": round(gold_sim, 4),
                "margin": round(margin, 4),
                "text": txt,
            })

        report["results"][ek] = {
            "model_name": EMBEDDING_CATALOG[ek]["model_name"],
            "license": EMBEDDING_CATALOG[ek]["license"],
            "dim": a_meta["dim"],
            "top1_accuracy": round(correct / len(probes), 4),
            "mean_similarity_margin": round(float(np.mean(margins)), 4),
            "by_language_group": {
                lg: round(float(np.mean(vals)), 4) for lg, vals in sorted(by_lang.items())
            },
            "details": details,
        }
    return report
