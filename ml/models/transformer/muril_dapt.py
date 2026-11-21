"""Unsupervised Domain-Adaptive Pretraining (DAPT / MLM) for MuRIL on anonymized Reddit text.

Follows Gururangan et al. (ACL 2020, "Don't Stop Pretraining"): continues
Masked Language Modeling (MLM) of ``google/muril-base-cased`` strictly on
unlabeled, anonymized, PII-redacted public Reddit posts and comments so the
encoder adapts to Indian MBA/CAT/IIM Sambalpur vocabulary, Hinglish
code-mixing, acronyms (CAT, CAP, SAP, WL, LPA, GEM, GNEM, PI, WAT), and
casual Reddit phrasing — WITHOUT fabricating any supervised labels.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ml.models.transformer.muril_classifier import resolve_local_muril_path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_dapt_output_dir() -> str:
    base = os.getenv("RCI_MODEL_CACHE_DIR")
    if base:
        return str(Path(base) / "muril_dapt_reddit")
    return str(_REPO_ROOT / ".cache" / "models" / "muril_dapt_reddit")


@dataclass
class DAPTConfig:
    base_model_id: str = "google/muril-base-cased"
    output_dir: str = field(default_factory=default_dapt_output_dir)
    max_length: int = 160
    batch_size: int = 8
    grad_accum_steps: int = 2
    val_fraction: float = 0.12
    epochs: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    mlm_probability: float = 0.15
    max_grad_norm: float = 1.0
    random_state: int = 20261120
    device: str = "auto"
    verbose: bool = True


class _UnlabeledTextDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = [t.strip() for t in texts if t and len(t.strip()) >= 12]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def _mask_tokens_bert(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    mlm_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard BERT 80/10/10 dynamic token masking without masking special tokens."""
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_probability, dtype=torch.float32)

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in (
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ):
        if sid is not None:
            special_mask |= input_ids.eq(sid)
    special_mask |= attention_mask.eq(0)

    prob_matrix.masked_fill_(special_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix, generator=generator).bool()
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    replace_rand = torch.rand(labels.shape, generator=generator)
    indices_replaced = (replace_rand < 0.8) & masked_indices
    masked_input = input_ids.clone()
    masked_input[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    indices_random = (replace_rand >= 0.8) & (replace_rand < 0.9) & masked_indices
    if indices_random.any():
        random_words = torch.randint(
            len(tokenizer), labels.shape, dtype=torch.long, generator=generator
        )
        masked_input[indices_random] = random_words[indices_random]

    # 10% -> keep original token unchanged
    return masked_input, labels


def run_muril_mlm_dapt(
    unlabeled_texts: Sequence[str],
    config: DAPTConfig | None = None,
) -> Dict[str, Any]:
    """Run unsupervised Masked Language Modeling on unlabeled Reddit text and save backbone."""
    cfg = config or DAPTConfig()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )

    # Deduplicate texts cleanly while preserving order
    seen: set[str] = set()
    clean_texts: List[str] = []
    for raw in unlabeled_texts:
        txt = " ".join(str(raw or "").split())
        if len(txt) < 16:
            continue
        key = txt.lower()
        if key not in seen:
            seen.add(key)
            clean_texts.append(txt)

    if len(clean_texts) < 20:
        raise ValueError(f"Need at least 20 unlabeled texts for DAPT, got {len(clean_texts)}")

    # Deterministic disjoint train / held-out validation split for honest perplexity evaluation
    indices = list(range(len(clean_texts)))
    random.Random(cfg.random_state).shuffle(indices)
    n_val = max(10, int(round(len(clean_texts) * float(cfg.val_fraction))))
    n_val = min(n_val, max(len(clean_texts) - 10, 1))
    val_idx = set(indices[:n_val])
    train_texts = [clean_texts[i] for i in range(len(clean_texts)) if i not in val_idx]
    val_texts = [clean_texts[i] for i in range(len(clean_texts)) if i in val_idx]

    model_path, local_only = resolve_local_muril_path(cfg.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_only).to(device)

    train_dataset = _UnlabeledTextDataset(train_texts)
    val_dataset = _UnlabeledTextDataset(val_texts)
    shuffle_gen = torch.Generator().manual_seed(cfg.random_state)
    mask_gen = torch.Generator().manual_seed(cfg.random_state + 99)

    def _make_collate(gen: torch.Generator):
        def collate_fn(batch_texts: List[str]):
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            masked_ids, labels = _mask_tokens_bert(
                enc["input_ids"],
                enc["attention_mask"],
                tokenizer,
                cfg.mlm_probability,
                gen,
            )
            enc["input_ids"] = masked_ids
            enc["labels"] = labels
            return enc
        return collate_fn

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
        collate_fn=_make_collate(mask_gen),
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    def _eval_heldout_mlm() -> tuple[float, float]:
        """Compute deterministic MLM loss & perplexity on held-out unseen Reddit texts."""
        model.eval()
        val_mask_gen = torch.Generator().manual_seed(cfg.random_state + 777)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_make_collate(val_mask_gen),
        )
        v_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if int((batch["labels"] != -100).sum().item()) == 0:
                    continue
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(**batch)
                    loss = out.loss
                if not torch.isnan(loss):
                    v_loss += float(loss.detach().item())
                    v_batches += 1
        mean_v = v_loss / max(v_batches, 1)
        return mean_v, math.exp(min(mean_v, 20.0))

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.learning_rate,
    )
    accum = max(1, int(cfg.grad_accum_steps))
    total_steps = max((math.ceil(len(loader) / accum)) * cfg.epochs, 1)
    warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    t0 = time.perf_counter()
    # Baseline Epoch 0 held-out perplexity BEFORE domain adaptation
    base_val_loss, base_val_ppl = _eval_heldout_mlm()
    if cfg.verbose:
        print(
            f"  [MuRIL-DAPT MLM] epoch 0/{cfg.epochs} (pre-DAPT baseline) "
            f"heldout_val_loss={base_val_loss:.4f} heldout_val_ppl={base_val_ppl:.2f} (n_val={len(val_texts)})",
            flush=True,
        )

    history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "train_mlm_loss": None,
            "train_perplexity": None,
            "mlm_loss": round(base_val_loss, 5),
            "perplexity": round(base_val_ppl, 3),
            "val_mlm_loss": round(base_val_loss, 5),
            "val_perplexity": round(base_val_ppl, 3),
        }
    ]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(loader, start=1):
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(**batch)
                loss = out.loss
            if torch.isnan(loss):
                continue
            scaled_loss = loss / accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss += float(loss.detach().item())
            batches += 1

            if step_idx % accum == 0 or step_idx == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_loss / max(batches, 1)
        train_ppl = math.exp(min(avg_train_loss, 20.0))
        val_loss, val_ppl = _eval_heldout_mlm()
        history.append(
            {
                "epoch": epoch,
                "train_mlm_loss": round(avg_train_loss, 5),
                "train_perplexity": round(train_ppl, 3),
                "mlm_loss": round(val_loss, 5),
                "perplexity": round(val_ppl, 3),
                "val_mlm_loss": round(val_loss, 5),
                "val_perplexity": round(val_ppl, 3),
            }
        )
        if cfg.verbose:
            print(
                f"  [MuRIL-DAPT MLM] epoch {epoch}/{cfg.epochs} "
                f"train_loss={avg_train_loss:.4f} (ppl={train_ppl:.2f}) | "
                f"heldout_val_loss={val_loss:.4f} (val_ppl={val_ppl:.2f})",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "base_model_id": cfg.base_model_id,
        "output_dir": str(out_dir),
        "unlabeled_documents": len(clean_texts),
        "train_documents": len(train_texts),
        "val_documents": len(val_texts),
        "perplexity_split": "heldout_validation",
        "epochs": cfg.epochs,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "initial_perplexity": history[0]["val_perplexity"],
        "final_perplexity": history[-1]["val_perplexity"],
        "initial_val_perplexity": history[0]["val_perplexity"],
        "final_val_perplexity": history[-1]["val_perplexity"],
        "final_train_perplexity": history[-1]["train_perplexity"],
        "history": history,
    }
    (out_dir / "dapt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary
