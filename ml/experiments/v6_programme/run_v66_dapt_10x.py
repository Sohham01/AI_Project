"""V6.6 Step 2 — 10x Strictly Inductive WWM DAPT on Combined V6.5 `train_pool` (MuRIL + XLM-R).

Trains two domain-adapted Masked Language Models across ALL strictly eligible real
corpus discussions (`data/corpus_v6_5/corpus_split_6_1_manifest.json` -> `train_pool`,
~1,563 threads / ~20,000+ documents) while enforcing TRIPLE zero-leakage exclusion of:
  1. All 261 threads in `corpus_split_6_1_manifest.json["test_pool"]`
  2. All 37 frozen V1 held-out test threads (`v5_frozen_splits.json`)
  3. Any thread/document sharing normalized `title[:120]` or exact comment/post text
     with any excluded test thread.

Checkpoints are written strictly outside the Git repo under:
  `D:\\AI_Project_Transformer_Temp\\v6_6_experiments\\{muril_dapt_v66_wwm,xlmr_dapt_v66_wwm}`
Tracked metadata & perplexity curves go to:
  `ml/artifacts/v6_6_experiments/dapt_10x/`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

_DEFAULT_CACHE = Path(os.environ.get("RCI_MODEL_CACHE_DIR") or r"D:\AI_Project_Transformer_Temp")
os.environ.setdefault("RCI_MODEL_CACHE_DIR", str(_DEFAULT_CACHE))
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ml.experiments.v5_optimization.common import (
    SEED,
    clear_cuda,
    load_v1_records_and_frozen_splits,
    save_json,
)
from ml.experiments.v5_optimization.run_dapt_inductive import (
    _mask_tokens_standard,
    audit_wordpiece_alignment,
)
from ml.experiments.v6_programme.v66_architecture import (
    mask_tokens_wwm_universal,
    resolve_local_backbone_path,
)

CORPUS_PATHS = [
    ("v6_1", ROOT / "data" / "live_corpus" / "corpus.json"),
    ("v6_3", ROOT / "data" / "corpus_v6_3" / "new_unannotated_corpus_v6_3.json"),
    ("v6_4", ROOT / "data" / "corpus_v6_4" / "new_iim_corpus_v6_4.json"),
    ("v6_5", ROOT / "data" / "corpus_v6_5" / "new_corpus_v6_5.json"),
]
SPLIT_6_1_PATH = ROOT / "data" / "corpus_v6_5" / "corpus_split_6_1_manifest.json"
OUT_ART_DIR = ROOT / "ml" / "artifacts" / "v6_6_experiments" / "dapt_10x"
CKPT_V66_DIR = _DEFAULT_CACHE / "v6_6_experiments"

_SPACE_RE = re.compile(r"\s+")


def _norm(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def _norm_lower(text: Any) -> str:
    return _norm(text).lower()


class _TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class _BucketedBatchSampler(Sampler[List[int]]):
    """Deterministic semi-sorted batch sampler reducing padding overhead ~2-3x while keeping epoch randomness."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, window_mult: int = 64) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.window = max(batch_size * window_mult, batch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches: List[List[int]] = []
        for start in range(0, len(idx), self.window):
            chunk = idx[start : start + self.window]
            chunk.sort(key=lambda i: self.lengths[i])
            for b_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b_start : b_start + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def build_v66_strict_dapt_pool(val_thread_fraction: float = 0.10, seed: int = SEED + 66) -> Dict[str, Any]:
    """Build 10x DAPT corpus strictly from V6.5 `train_pool` with triple zero-leakage exclusion."""
    v1 = load_v1_records_and_frozen_splits()
    split_manifest = json.loads(SPLIT_6_1_PATH.read_text(encoding="utf-8"))

    train_pool_pids = set(split_manifest["train_post_ids"])
    test_pool_pids = set(split_manifest["test_post_ids"])
    frozen_v1_pids = set(v1["manifest"]["test_corpus_post_ids"])
    assert not (train_pool_pids & test_pool_pids)
    assert frozen_v1_pids.issubset(test_pool_pids)

    blocked_pids = test_pool_pids | frozen_v1_pids
    blocked_titles_120 = {_norm_lower(r.get("post_title", ""))[:120] for r in v1["test"] if r.get("post_title")}
    blocked_exact_strings: set[str] = set()
    for r in v1["test"]:
        rt = _norm_lower(r.get("text", ""))
        if rt:
            blocked_exact_strings.add(rt)

    all_posts: List[Tuple[str, Dict[str, Any]]] = []
    for cohort, path in CORPUS_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for p in payload.get("posts", []):
            pid = str(p["reddit_id"])
            if pid in blocked_pids or bool(p.get("dapt_and_train_excluded", False)):
                tl = _norm_lower(p.get("title", ""))
                bl = _norm_lower(p.get("body", ""))
                if tl:
                    blocked_titles_120.add(tl[:120])
                    blocked_exact_strings.add(tl)
                if bl:
                    blocked_exact_strings.add(bl)
                for c in p.get("comments") or []:
                    cb = _norm_lower(c.get("body", ""))
                    if cb:
                        blocked_exact_strings.add(cb)
                continue
            all_posts.append((cohort, p))

    eligible_posts: List[Tuple[str, Dict[str, Any]]] = []
    title_excluded_extra = 0
    for cohort, p in all_posts:
        pid = str(p["reddit_id"])
        t120 = _norm_lower(p.get("title", ""))[:120]
        if pid not in train_pool_pids or (t120 and t120 in blocked_titles_120):
            title_excluded_extra += 1
            continue
        eligible_posts.append((cohort, p))

    eligible_pids = sorted({str(p["reddit_id"]) for _, p in eligible_posts})
    assert not (set(eligible_pids) & blocked_pids), "CRITICAL LEAKAGE: test thread in eligible DAPT pool!"

    rng = random.Random(seed)
    shuffled = list(eligible_pids)
    rng.shuffle(shuffled)
    n_val = max(40, int(round(len(shuffled) * val_thread_fraction)))
    dapt_val_pids = set(shuffled[:n_val])
    dapt_train_pids = set(shuffled[n_val:])
    assert not (dapt_train_pids & dapt_val_pids)
    assert not (dapt_train_pids & blocked_pids)
    assert not (dapt_val_pids & blocked_pids)

    def _extract_docs(target_pids: set[str]) -> Tuple[List[str], List[str]]:
        seen_lower: set[str] = set()
        texts: List[str] = []
        doc_ids: List[str] = []
        for _, p in eligible_posts:
            pid = str(p["reddit_id"])
            if pid not in target_pids:
                continue
            t = _norm(p.get("title", ""))
            b = _norm(p.get("body", ""))
            post_txt = f"{t} — {b}".strip(" —") if b else t
            pl = post_txt.lower()
            if len(post_txt) >= 16 and pl not in seen_lower and pl not in blocked_exact_strings:
                seen_lower.add(pl)
                texts.append(post_txt)
                doc_ids.append(f"post-{pid}")
            for c in p.get("comments") or []:
                cb = _norm(c.get("body", ""))
                cl = cb.lower()
                if len(cb) >= 16 and cl not in seen_lower and cl not in blocked_exact_strings:
                    seen_lower.add(cl)
                    texts.append(cb)
                    doc_ids.append(f"comment-{c['reddit_id']}")
        return texts, doc_ids

    train_texts, train_doc_ids = _extract_docs(dapt_train_pids)
    val_texts, val_doc_ids = _extract_docs(dapt_val_pids)

    # Extra guard: remove any cross-split duplicate exact text
    val_lower_set = {t.lower() for t in val_texts}
    dedup_train = [(t, d) for t, d in zip(train_texts, train_doc_ids) if t.lower() not in val_lower_set]
    cross_dups_removed = len(train_texts) - len(dedup_train)
    train_texts = [x[0] for x in dedup_train]
    train_doc_ids = [x[1] for x in dedup_train]

    manifest = {
        "manifest_version": "v6.6_strict_dapt_10x",
        "seed": seed,
        "total_combined_threads": len(train_pool_pids) + len(test_pool_pids),
        "blocked_test_pool_threads": len(blocked_pids),
        "frozen_v1_test_threads": len(frozen_v1_pids),
        "title_excluded_extra_threads": title_excluded_extra,
        "eligible_dapt_threads": len(eligible_pids),
        "dapt_train_threads": len(dapt_train_pids),
        "dapt_val_threads": len(dapt_val_pids),
        "dapt_train_docs": len(train_texts),
        "dapt_val_docs": len(val_texts),
        "cross_split_exact_dups_removed": cross_dups_removed,
        "intersection_train_and_blocked": len(dapt_train_pids & blocked_pids),
        "intersection_val_and_blocked": len(dapt_val_pids & blocked_pids),
        "intersection_train_and_val": len(dapt_train_pids & dapt_val_pids),
        "dapt_train_threads_sha256": hashlib.sha256(",".join(sorted(dapt_train_pids)).encode()).hexdigest(),
        "dapt_val_threads_sha256": hashlib.sha256(",".join(sorted(dapt_val_pids)).encode()).hexdigest(),
        "dapt_train_docs_sha256": hashlib.sha256(",".join(train_doc_ids).encode()).hexdigest(),
        "dapt_val_docs_sha256": hashlib.sha256(",".join(val_doc_ids).encode()).hexdigest(),
    }
    assert manifest["intersection_train_and_blocked"] == 0
    assert manifest["intersection_val_and_blocked"] == 0
    assert manifest["intersection_train_and_val"] == 0

    OUT_ART_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUT_ART_DIR / "v66_strict_dapt_manifest.json", manifest)
    return {"train_texts": train_texts, "val_texts": val_texts, "manifest": manifest}


def _audit_sentencepiece_alignment(tokenizer: Any) -> Dict[str, Any]:
    samples = [
        "bhaiya percentile waitlist IIMSambalpur placement hostel acads prof convert",
        "yaar mocktest cutoff LPA GEM GNEM CAP SAP SIP PPO mess food attendance",
        "bhot stress ho raha hai bhai exams aur assignments dono saath mein hain",
    ]
    multi_piece, total_words = 0, 0
    details = []
    for s in samples:
        toks = tokenizer.tokenize(s)
        groups: List[List[str]] = []
        for t in toks:
            stripped = t.lstrip("\u2581")
            is_punct = bool(stripped) and all(not ch.isalnum() for ch in stripped)
            starts_new = t.startswith("\u2581") or is_punct
            prev_punct = bool(groups) and all(not ch.isalnum() for ch in groups[-1][-1].lstrip("\u2581"))
            if not starts_new and groups and not prev_punct:
                groups[-1].append(t)
            else:
                groups.append([t])
        total_words += len(groups)
        multi_piece += sum(1 for g in groups if len(g) > 1)
        details.append({"text": s, "n_tokens": len(toks), "n_word_units": len(groups), "split_examples": [g for g in groups if len(g) > 1][:10]})
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "continuation_mode": "sentencepiece_\\u2581",
        "multi_subword_fraction": round(multi_piece / max(total_words, 1), 4),
        "samples": details,
    }


def _compute_sparse_mlm_loss(model: Any, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute exact CrossEntropy MLM loss ONLY on `[mask] != -100` positions to cut LM-head VRAM ~8x."""
    labels = batch["labels"]
    mask = labels != -100
    if not mask.any():
        return None
    base = model.bert if hasattr(model, "bert") else model.roberta
    head = model.cls if hasattr(model, "cls") else model.lm_head
    hidden = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    logits = head(hidden[mask].unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits.float(), labels[mask])


_LAYER_IDX_RE = re.compile(r"\.layer\.(\d+)\.")


def _build_dapt_llrd_groups(model: Any, base_lr: float, llrd: float = 0.95, n_layers: int = 12) -> List[Dict[str, Any]]:
    """Layer-wise LR decay during DAPT so lower syntactic layers do not drift on domain text."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias")
    buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else 0.01
        m = _LAYER_IDX_RE.search(name)
        if m:
            layer_i = int(m.group(1))
            lr = base_lr * (llrd ** max(0, n_layers - 1 - layer_i))
        elif "embeddings" in name:
            lr = base_lr * (llrd ** n_layers)
        else:
            # MLM prediction head (`cls` or `lm_head`)
            lr = base_lr
        key = (round(lr, 10), wd)
        buckets.setdefault(key, []).append(param)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]


def train_v66_dapt_backbone(
    *,
    variant_name: str,
    base_model_id: str,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    epochs: int = 4,
    max_length: int = 160,
    batch_size: int = 10,
    grad_accum_steps: int = 4,
    learning_rate: float = 4.5e-5,
    llrd: float = 0.95,
    warmup_ratio: float = 0.10,
    mlm_probability: float = 0.15,
    gradient_checkpointing: bool = False,
    seed: int = SEED + 66,
    force: bool = False,
) -> Dict[str, Any]:
    """Train one 10x WWM DAPT backbone under `D:\\AI_Project_Transformer_Temp\\v6_6_experiments`."""
    out_dir = CKPT_V66_DIR / variant_name
    done_path = out_dir / "dapt_done.json"
    if done_path.is_file() and (out_dir / "model.safetensors").is_file() and not force:
        print(f"[{variant_name}] Reusing completed DAPT checkpoint at {out_dir}", flush=True)
        return json.loads(done_path.read_text(encoding="utf-8"))

    clear_cuda()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, local_only = resolve_local_backbone_path(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_only).to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    subword_audit = (
        _audit_sentencepiece_alignment(tokenizer)
        if "xlm" in base_model_id.lower() or "roberta" in tokenizer.__class__.__name__.lower()
        else audit_wordpiece_alignment(tokenizer)
    )

    train_ds = _TextDataset(train_texts)
    val_ds = _TextDataset(val_texts)
    train_lens = [min(len(t), max_length * 6) for t in train_texts]
    sampler = _BucketedBatchSampler(train_lens, batch_size=batch_size, seed=seed)

    def _make_collate(mode: str, gen: torch.Generator, py_rng: random.Random):
        def collate_fn(batch_texts: List[str]):
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if mode == "wwm":
                masked_ids, labels = mask_tokens_wwm_universal(
                    enc["input_ids"], enc["attention_mask"], tokenizer, mlm_probability, py_rng, gen
                )
            else:
                masked_ids, labels = _mask_tokens_standard(
                    enc["input_ids"], enc["attention_mask"], tokenizer, mlm_probability, gen
                )
            enc["input_ids"] = masked_ids
            enc["labels"] = labels
            return enc

        return collate_fn

    mask_gen = torch.Generator().manual_seed(seed + 99)
    wwm_rng = random.Random(seed + 199)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=_make_collate("wwm", mask_gen, wwm_rng),
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # Pre-sort validation docs by length for ~2.5x faster val evaluation with exact same loss values
    val_sorted = sorted(val_texts, key=len)
    val_ds_sorted = _TextDataset(val_sorted)

    def _eval_val_ppl(eval_mode: str) -> Tuple[float, float]:
        model.eval()
        vgen = torch.Generator().manual_seed(seed + 777)
        vrng = random.Random(seed + 888)
        vloader = DataLoader(
            val_ds_sorted,
            batch_size=max(batch_size, 8),
            shuffle=False,
            num_workers=0,
            collate_fn=_make_collate(eval_mode, vgen, vrng),
        )
        vloss, vb = 0.0, 0
        with torch.no_grad():
            for batch in vloader:
                if int((batch["labels"] != -100).sum().item()) == 0:
                    continue
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss = _compute_sparse_mlm_loss(model, batch)
                if loss is not None and not torch.isnan(loss):
                    vloss += float(loss.detach().item())
                    vb += 1
        mean_l = vloss / max(vb, 1)
        return mean_l, math.exp(min(mean_l, 20.0))

    n_layers = int(getattr(model.config, "num_hidden_layers", 12))
    optimizer = torch.optim.AdamW(
        _build_dapt_llrd_groups(model, base_lr=learning_rate, llrd=llrd, n_layers=n_layers),
        lr=learning_rate,
    )
    accum = max(1, grad_accum_steps)
    total_steps = max(math.ceil(len(train_loader) / accum) * epochs, 1)
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.perf_counter()
    base_std_l, base_std_ppl = _eval_val_ppl("standard")
    base_wwm_l, base_wwm_ppl = _eval_val_ppl("wwm")
    print(
        f"[{variant_name}] base ({base_model_id}) | train={len(train_texts)} val={len(val_texts)} | "
        f"ep0 val_std_ppl={base_std_ppl:.2f} | val_wwm_ppl={base_wwm_ppl:.2f}",
        flush=True,
    )

    history: List[Dict[str, Any]] = [{
        "epoch": 0,
        "train_loss": None,
        "val_std_loss": round(base_std_l, 5),
        "val_std_ppl": round(base_std_ppl, 3),
        "val_wwm_loss": round(base_wwm_l, 5),
        "val_wwm_ppl": round(base_wwm_ppl, 3),
    }]

    best_metric = base_wwm_l
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        tr_loss, tr_b = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader):
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                raw_loss = _compute_sparse_mlm_loss(model, batch)
                if raw_loss is None or torch.isnan(raw_loss):
                    continue
                loss = raw_loss / float(accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += float(raw_loss.detach().item())
            tr_b += 1

            if ((step_idx + 1) % accum == 0) or (step_idx + 1 == len(train_loader)):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step_idx + 1) % 400 == 0:
                print(
                    f"  [{variant_name}] ep {epoch}/{epochs} step {step_idx+1}/{len(train_loader)} "
                    f"({time.perf_counter()-ep_t0:.0f}s) avg_loss={tr_loss/max(tr_b,1):.4f}",
                    flush=True,
                )

        mean_tr = tr_loss / max(tr_b, 1)
        val_std_l, val_std_ppl = _eval_val_ppl("standard")
        val_wwm_l, val_wwm_ppl = _eval_val_ppl("wwm")
        ep_s = round(time.perf_counter() - ep_t0, 1)
        print(
            f"[{variant_name}] ep {epoch}/{epochs} ({ep_s}s) | tr_loss={mean_tr:.4f} | "
            f"val_std_ppl={val_std_ppl:.2f} | val_wwm_ppl={val_wwm_ppl:.2f}",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "epoch_seconds": ep_s,
            "train_loss": round(mean_tr, 5),
            "val_std_loss": round(val_std_l, 5),
            "val_std_ppl": round(val_std_ppl, 3),
            "val_wwm_loss": round(val_wwm_l, 5),
            "val_wwm_ppl": round(val_wwm_ppl, 3),
        })
        if val_wwm_l < best_metric - 1e-4:
            best_metric = val_wwm_l
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed_s = round(time.perf_counter() - t0, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "variant_name": variant_name,
        "base_model_id": base_model_id,
        "checkpoint_dir": str(out_dir),
        "masking_mode": "wwm_universal",
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "max_length": max_length,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "initial_val_std_ppl": history[0]["val_std_ppl"],
        "initial_val_wwm_ppl": history[0]["val_wwm_ppl"],
        "best_val_wwm_ppl": min(h["val_wwm_ppl"] for h in history),
        "final_val_std_ppl": history[-1]["val_std_ppl"],
        "final_val_wwm_ppl": history[-1]["val_wwm_ppl"],
        "subword_audit": subword_audit,
        "history": history,
    }
    save_json(done_path, meta)

    del model, optimizer, scheduler, best_state
    clear_cuda()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--only", choices=("all", "muril", "xlmr"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pool = build_v66_strict_dapt_pool()
    m = pool["manifest"]
    print(
        f"[V6.6 DAPT Pool] eligible_threads={m['eligible_dapt_threads']} "
        f"(train={m['dapt_train_threads']} [{m['dapt_train_docs']} docs], "
        f"val={m['dapt_val_threads']} [{m['dapt_val_docs']} docs], "
        f"blocked={m['blocked_test_pool_threads']}, leakage={m['intersection_train_and_blocked']})",
        flush=True,
    )

    existing_sum = (
        json.loads((OUT_ART_DIR / "dapt_10x_summary.json").read_text(encoding="utf-8"))
        if (OUT_ART_DIR / "dapt_10x_summary.json").is_file()
        else {}
    )
    backbones: Dict[str, Any] = dict(existing_sum.get("backbones") or {})
    for v_key in ("muril_dapt_v66_wwm", "xlmr_dapt_v66_wwm"):
        done_p = CKPT_V66_DIR / v_key / "dapt_done.json"
        if done_p.is_file():
            backbones[v_key] = json.loads(done_p.read_text(encoding="utf-8"))

    if args.only in {"all", "muril"}:
        backbones["muril_dapt_v66_wwm"] = train_v66_dapt_backbone(
            variant_name="muril_dapt_v66_wwm",
            base_model_id="google/muril-base-cased",
            train_texts=pool["train_texts"],
            val_texts=pool["val_texts"],
            epochs=args.epochs,
            learning_rate=4.5e-5,
            llrd=0.95,
            warmup_ratio=0.10,
            seed=SEED + 66,
            force=args.force,
        )
    if args.only in {"all", "xlmr"}:
        backbones["xlmr_dapt_v66_wwm"] = train_v66_dapt_backbone(
            variant_name="xlmr_dapt_v66_wwm",
            base_model_id="xlm-roberta-base",
            train_texts=pool["train_texts"],
            val_texts=pool["val_texts"],
            epochs=args.epochs,
            batch_size=8,
            grad_accum_steps=5,
            learning_rate=2.0e-5,
            llrd=0.95,
            warmup_ratio=0.12,
            gradient_checkpointing=True,
            seed=SEED + 67,
            force=args.force,
        )

    summary = {
        "dapt_protocol": "v6.6_10x_strictly_inductive_wwm_llrd",
        "pool_manifest": m,
        "backbones": backbones,
    }
    save_json(OUT_ART_DIR / "dapt_10x_summary.json", summary)
    print(f"[V6.6] Saved {OUT_ART_DIR / 'dapt_10x_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
