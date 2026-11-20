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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ml.models.transformer.muril_classifier import resolve_local_muril_path


@dataclass
class DAPTConfig:
    base_model_id: str = "google/muril-base-cased"
    output_dir: str = "D:/AI_Project_Transformer_Temp/muril_dapt_reddit"
    max_length: int = 160
    batch_size: int = 16
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

    model_path, local_only = resolve_local_muril_path(cfg.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_only).to(device)

    dataset = _UnlabeledTextDataset(clean_texts)
    shuffle_gen = torch.Generator().manual_seed(cfg.random_state)
    mask_gen = torch.Generator().manual_seed(cfg.random_state + 99)

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
            mask_gen,
        )
        enc["input_ids"] = masked_ids
        enc["labels"] = labels
        return enc

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
        collate_fn=collate_fn,
    )

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
    total_steps = max(len(loader) * cfg.epochs, 1)
    warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    t0 = time.perf_counter()
    history: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(**batch)
                loss = out.loss
            if torch.isnan(loss):
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().item())
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        ppl = math.exp(min(avg_loss, 20.0))
        history.append({"epoch": epoch, "mlm_loss": round(avg_loss, 5), "perplexity": round(ppl, 3)})
        if cfg.verbose:
            print(
                f"  [MuRIL-DAPT MLM] epoch {epoch}/{cfg.epochs} loss={avg_loss:.4f} perplexity={ppl:.2f}",
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
        "epochs": cfg.epochs,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "initial_perplexity": history[0]["perplexity"] if history else None,
        "final_perplexity": history[-1]["perplexity"] if history else None,
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
