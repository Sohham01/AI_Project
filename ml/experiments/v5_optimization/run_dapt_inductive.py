"""Strictly inductive Domain-Adaptive Pretraining (DAPT) for MuRIL (Section 3C & Phase 5).

Guarantees:
1. Supervised frozen 37 test post_ids (n=92 comments) are loaded FIRST and strictly excluded
   from ALL DAPT training and DAPT validation texts (zero-intersection assertion).
2. DAPT train/validation split is grouped strictly by original post_id (never random text split).
3. Exact thread and document manifests are persisted under ml/artifacts/v5_experiments/dapt/.
4. Supports both standard WordPiece subword masking (mlm) and true WordPiece-aligned
   Whole-Word Masking (wwm) inspecting '##' continuation tokens rather than naive whitespace.
5. Saves checkpoints under $RCI_MODEL_CACHE_DIR/v5_experiments/ without ever touching
   the canonical $RCI_MODEL_CACHE_DIR/muril_dapt_reddit directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v5_optimization.common import (
    CKPT_ROOT,
    EXP_ROOT,
    SEED,
    clear_cuda,
    ensure_dir,
    load_corpus_lookups,
    load_v1_records_and_frozen_splits,
    save_json,
)
from ml.models.transformer.muril_classifier import resolve_local_muril_path


class _TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def audit_wordpiece_alignment(tokenizer: Any) -> Dict[str, Any]:
    """Inspect MuRIL WordPiece tokenization & '##' continuations on Hinglish/campus tokens."""
    samples = [
        "bhaiya percentile waitlist IIMSambalpur placement hostel acads prof convert",
        "yaar mocktest cutoff LPA GEM GNEM CAP SAP SIP PPO mess food attendance",
        "bhot stress ho raha hai bhai exams aur assignments dono saath mein hain",
        "kuch nahi hota bro waitlist move hogi chill karo campus life sahi hai",
        "academic workload curriculum teaching infrastructure extracurriculars",
    ]
    details = []
    multi_piece_words = 0
    total_words = 0
    for s in samples:
        toks = tokenizer.tokenize(s)
        # Reconstruct WordPiece word groups via '##' continuation prefix
        groups: List[List[str]] = []
        for t in toks:
            if t.startswith("##") and groups:
                groups[-1].append(t)
            else:
                groups.append([t])
        total_words += len(groups)
        multi_piece_words += sum(1 for g in groups if len(g) > 1)
        details.append(
            {
                "text": s,
                "n_tokens": len(toks),
                "n_word_units": len(groups),
                "split_examples": [g for g in groups if len(g) > 1][:12],
            }
        )
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": len(tokenizer),
        "continuation_prefix": "##",
        "total_word_units_sampled": total_words,
        "multi_subword_units": multi_piece_words,
        "multi_subword_fraction": round(multi_piece_words / max(total_words, 1), 4),
        "samples": details,
    }


def _mask_tokens_standard(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    mlm_probability: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_probability, dtype=torch.float32)

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id):
        if sid is not None:
            special_mask |= input_ids.eq(sid)
    special_mask |= attention_mask.eq(0)

    prob_matrix.masked_fill_(special_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix, generator=generator).bool()
    labels[~masked_indices] = -100

    replace_rand = torch.rand(labels.shape, generator=generator)
    masked_input = input_ids.clone()
    masked_input[(replace_rand < 0.8) & masked_indices] = tokenizer.mask_token_id
    indices_random = (replace_rand >= 0.8) & (replace_rand < 0.9) & masked_indices
    if indices_random.any():
        rand_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, generator=generator)
        masked_input[indices_random] = rand_words[indices_random]
    return masked_input, labels


def _mask_tokens_wwm(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    mlm_probability: float,
    rng: random.Random,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """True WordPiece Whole-Word Masking using exact '##' continuation grouping.

    Every token starting with '##' is grouped with its preceding root subword so an entire
    Hinglish word, acronym, or domain term is masked atomically (80% [MASK], 10% random, 10% orig).
    """
    special_ids = {
        sid
        for sid in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id)
        if sid is not None
    }
    bsz, seq_len = input_ids.shape
    masked_indices = torch.zeros((bsz, seq_len), dtype=torch.bool)

    for b in range(bsz):
        ids_row = input_ids[b].tolist()
        attn_row = attention_mask[b].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids_row)

        word_groups: List[List[int]] = []
        valid_tokens = 0
        for pos, (tid, att, tok) in enumerate(zip(ids_row, attn_row, toks)):
            if att == 0 or tid in special_ids or tok is None:
                continue
            valid_tokens += 1
            if tok.startswith("##") and word_groups:
                word_groups[-1].append(pos)
            else:
                word_groups.append([pos])

        if not word_groups or valid_tokens == 0:
            continue

        target_masked = max(1, int(round(valid_tokens * mlm_probability)))
        rng.shuffle(word_groups)
        covered = 0
        for grp in word_groups:
            if covered >= target_masked:
                break
            # Avoid blowing past target on a very long single word if we already masked some tokens
            if covered > 0 and covered + len(grp) > target_masked + 2:
                continue
            for pos in grp:
                masked_indices[b, pos] = True
            covered += len(grp)

    labels = input_ids.clone()
    labels[~masked_indices] = -100

    # Apply 80/10/10 at the masked token level (matching BERT WWM official behavior)
    replace_rand = torch.rand(labels.shape, generator=generator)
    masked_input = input_ids.clone()
    masked_input[(replace_rand < 0.8) & masked_indices] = tokenizer.mask_token_id
    indices_random = (replace_rand >= 0.8) & (replace_rand < 0.9) & masked_indices
    if indices_random.any():
        rand_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, generator=generator)
        masked_input[indices_random] = rand_words[indices_random]
    return masked_input, labels


def build_strictly_inductive_dapt_corpus(
    val_thread_fraction: float = 0.12,
    seed: int = SEED,
    manifest_name: str = "strict_inductive_manifest.json",
) -> Dict[str, Any]:
    """Exclude ALL 37 frozen supervised test threads by exact corpus_post_id AND normalized title,
    then group-split the remaining strictly eligible threads by post_id."""
    import re as _re
    norm = lambda s: _re.sub(r"\s+", " ", str(s or "")).strip().lower()

    v1 = load_v1_records_and_frozen_splits()
    test_corpus_pids = set(v1["manifest"]["test_corpus_post_ids"])
    test_titles_120 = {norm(r["post_title"])[:120] for r in v1["test"] if r.get("post_title")}
    train_sup_pids = set(v1["manifest"]["trainval_corpus_post_ids"])
    val_sup_pids: set[str] = set()

    corpus, _, _ = load_corpus_lookups()
    eligible_posts = []
    excluded_posts = []
    for p in corpus["posts"]:
        pid = p["reddit_id"]
        t120 = norm(p.get("title", ""))[:120]
        if pid in test_corpus_pids or t120 in test_titles_120:
            excluded_posts.append(p)
        else:
            eligible_posts.append(p)

    test_pids = {p["reddit_id"] for p in excluded_posts}
    eligible_pids = sorted(p["reddit_id"] for p in eligible_posts)
    excluded_pids = sorted(test_pids)
    assert test_corpus_pids.issubset(test_pids), (
        f"Expected all {len(test_corpus_pids)} frozen test corpus_post_ids excluded!"
    )
    assert not (set(eligible_pids) & test_pids), "CRITICAL LEAKAGE: test post_id found in eligible DAPT pool!"
    assert not ({norm(p.get("title", ""))[:120] for p in eligible_posts} & test_titles_120), (
        "CRITICAL LEAKAGE: frozen test post_title found in eligible DAPT pool!"
    )

    # Deterministic thread-grouped DAPT validation split
    # Prefer sampling DAPT val threads from completely unlabeled + sup-val threads (keeping sup-train untouched or balanced)
    rng = random.Random(seed + 101)
    shuffled_pids = list(eligible_pids)
    rng.shuffle(shuffled_pids)
    n_val_threads = max(20, int(round(len(shuffled_pids) * val_thread_fraction)))
    dapt_val_pids = set(shuffled_pids[:n_val_threads])
    dapt_train_pids = set(shuffled_pids[n_val_threads:])

    assert not (dapt_train_pids & dapt_val_pids), "Overlap between DAPT train and val threads!"
    assert not (dapt_train_pids & test_pids), "Leakage between DAPT train and frozen test threads!"
    assert not (dapt_val_pids & test_pids), "Leakage between DAPT val and frozen test threads!"

    def _extract_docs(target_pids: set[str]) -> Tuple[List[str], List[str]]:
        seen: set[str] = set()
        texts: List[str] = []
        doc_ids: List[str] = []
        for p in eligible_posts:
            pid = p["reddit_id"]
            if pid not in target_pids:
                continue
            title = " ".join(str(p.get("title") or "").split())
            body = " ".join(str(p.get("body") or "").split())
            post_txt = f"{title} — {body}".strip(" —") if body else title
            if len(post_txt) >= 16 and post_txt.lower() not in seen:
                seen.add(post_txt.lower())
                texts.append(post_txt)
                doc_ids.append(f"post-{pid}")
            for c in p.get("comments") or []:
                ctxt = " ".join(str(c.get("body") or "").split())
                if len(ctxt) >= 16 and ctxt.lower() not in seen:
                    seen.add(ctxt.lower())
                    texts.append(ctxt)
                    doc_ids.append(f"comment-{c['reddit_id']}")
        return texts, doc_ids

    train_texts, train_doc_ids = _extract_docs(dapt_train_pids)
    val_texts, val_doc_ids = _extract_docs(dapt_val_pids)

    # Extra defense-in-depth: also exclude any exact normalized string that appears in any excluded test thread
    test_exact_strings: set[str] = set()
    excluded_comment_count = 0
    for p in excluded_posts:
        t = " ".join(str(p.get("title") or "").split()).lower()
        b = " ".join(str(p.get("body") or "").split()).lower()
        if t:
            test_exact_strings.add(t)
        if b:
            test_exact_strings.add(b)
        for c in p.get("comments") or []:
            excluded_comment_count += 1
            ct = " ".join(str(c.get("body") or "").split()).lower()
            if ct:
                test_exact_strings.add(ct)

    train_filtered = [(t, d) for t, d in zip(train_texts, train_doc_ids) if t.lower() not in test_exact_strings]
    val_filtered = [(t, d) for t, d in zip(val_texts, val_doc_ids) if t.lower() not in test_exact_strings]
    exact_dups_removed = (len(train_texts) - len(train_filtered)) + (len(val_texts) - len(val_filtered))
    train_texts, train_doc_ids = ([x[0] for x in train_filtered], [x[1] for x in train_filtered])
    val_texts, val_doc_ids = ([x[0] for x in val_filtered], [x[1] for x in val_filtered])

    manifest = {
        "seed": seed,
        "protocol": "strictly_inductive_thread_grouped",
        "total_corpus_threads": len(corpus["posts"]),
        "excluded_frozen_test_threads": len(excluded_pids),
        "excluded_frozen_test_comments": excluded_comment_count,
        "eligible_dapt_threads": len(eligible_pids),
        "dapt_train_threads": len(dapt_train_pids),
        "dapt_val_threads": len(dapt_val_pids),
        "dapt_train_docs": len(train_texts),
        "dapt_val_docs": len(val_texts),
        "exact_test_string_duplicates_removed": exact_dups_removed,
        "intersection_dapt_train_and_test_threads": len(dapt_train_pids & test_pids),
        "intersection_dapt_val_and_test_threads": len(dapt_val_pids & test_pids),
        "intersection_dapt_train_and_val_threads": len(dapt_train_pids & dapt_val_pids),
        "overlap_dapt_train_with_sup_train_threads": len(dapt_train_pids & train_sup_pids),
        "overlap_dapt_train_with_sup_val_threads": len(dapt_train_pids & val_sup_pids),
        "excluded_test_post_ids": excluded_pids,
        "dapt_train_post_ids": sorted(dapt_train_pids),
        "dapt_val_post_ids": sorted(dapt_val_pids),
        "dapt_train_doc_ids": train_doc_ids,
        "dapt_val_doc_ids": val_doc_ids,
        "dapt_train_sha256": hashlib.sha256(",".join(train_doc_ids).encode("utf-8")).hexdigest(),
        "dapt_val_sha256": hashlib.sha256(",".join(val_doc_ids).encode("utf-8")).hexdigest(),
    }
    assert manifest["intersection_dapt_train_and_test_threads"] == 0
    assert manifest["intersection_dapt_val_and_test_threads"] == 0
    assert manifest["intersection_dapt_train_and_val_threads"] == 0

    save_json(EXP_ROOT / "dapt" / manifest_name, manifest)
    return {
        "train_texts": train_texts,
        "val_texts": val_texts,
        "manifest": manifest,
    }


def train_inductive_dapt_variant(
    variant_name: str,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    *,
    masking_mode: str = "standard",  # standard | wwm
    epochs: int = 4,
    early_stop_patience: int = 2,
    max_length: int = 160,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.15,
    seed: int = SEED + 4,
) -> Dict[str, Any]:
    """Train one strictly inductive MuRIL DAPT variant on RTX 4060 GPU with thread-grouped val PPL."""
    clear_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, local_only = resolve_local_muril_path("google/muril-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_only).to(device)

    wwm_audit = audit_wordpiece_alignment(tokenizer)
    train_ds = _TextDataset(train_texts)
    val_ds = _TextDataset(val_texts)

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
                masked_ids, labels = _mask_tokens_wwm(
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

    shuffle_gen = torch.Generator().manual_seed(seed)
    mask_gen = torch.Generator().manual_seed(seed + 99)
    wwm_rng = random.Random(seed + 199)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
        collate_fn=_make_collate(masking_mode, mask_gen, wwm_rng),
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    def _eval_val_ppl(eval_mode: str) -> Tuple[float, float]:
        model.eval()
        vgen = torch.Generator().manual_seed(seed + 777)
        vrng = random.Random(seed + 888)
        vloader = DataLoader(
            val_ds,
            batch_size=batch_size,
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
                    loss = model(**batch).loss
                if not torch.isnan(loss):
                    vloss += float(loss.detach().item())
                    vb += 1
        mean_l = vloss / max(vb, 1)
        return mean_l, math.exp(min(mean_l, 20.0))

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ],
        lr=learning_rate,
        fused=(device.type == "cuda"),
    )
    accum = max(1, grad_accum_steps)
    total_steps = max(math.ceil(len(train_loader) / accum) * epochs, 1)
    warmup_steps = max(int(total_steps * 0.10), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.perf_counter()
    # Evaluate BOTH standard-MLM and WWM held-out val perplexity on the exact same thread-grouped val set
    base_std_l, base_std_ppl = _eval_val_ppl("standard")
    base_wwm_l, base_wwm_ppl = _eval_val_ppl("wwm")
    print(
        f"[{variant_name}] epoch 0/{epochs} | val_std_ppl={base_std_ppl:.2f} | val_wwm_ppl={base_wwm_ppl:.2f}",
        flush=True,
    )

    history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_std_loss": round(base_std_l, 5),
            "val_std_ppl": round(base_std_ppl, 3),
            "val_wwm_loss": round(base_wwm_l, 5),
            "val_wwm_ppl": round(base_wwm_ppl, 3),
        }
    ]

    best_metric = base_wwm_l if masking_mode == "wwm" else base_std_l
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_b = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader, start=1):
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = model(**batch).loss
            if torch.isnan(loss):
                continue
            scaled = loss / accum
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            tr_loss += float(loss.detach().item())
            tr_b += 1
            if step_idx % accum == 0 or step_idx == len(train_loader):
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

        mean_tr = tr_loss / max(tr_b, 1)
        val_std_l, val_std_ppl = _eval_val_ppl("standard")
        val_wwm_l, val_wwm_ppl = _eval_val_ppl("wwm")
        cur_metric = val_wwm_l if masking_mode == "wwm" else val_std_l
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(mean_tr, 5),
                "train_ppl": round(math.exp(min(mean_tr, 20.0)), 3),
                "val_std_loss": round(val_std_l, 5),
                "val_std_ppl": round(val_std_ppl, 3),
                "val_wwm_loss": round(val_wwm_l, 5),
                "val_wwm_ppl": round(val_wwm_ppl, 3),
            }
        )
        print(
            f"[{variant_name}] epoch {epoch}/{epochs} | train_loss={mean_tr:.4f} "
            f"| val_std_ppl={val_std_ppl:.2f} | val_wwm_ppl={val_wwm_ppl:.2f}",
            flush=True,
        )
        if cur_metric < best_metric - 1e-4:
            best_metric = cur_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(f"[{variant_name}] early stopping at epoch {epoch} (best_epoch={best_epoch})", flush=True)
                break

    elapsed_s = round(time.perf_counter() - t0, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    out_dir = ensure_dir(CKPT_ROOT / variant_name)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "variant_name": variant_name,
        "output_dir": str(out_dir),
        "masking_mode": masking_mode,
        "epochs_configured": epochs,
        "best_epoch": best_epoch,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "elapsed_seconds": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "train_docs": len(train_texts),
        "val_docs": len(val_texts),
        "wordpiece_audit": wwm_audit,
        "history": history,
    }
    save_json(out_dir / "dapt_metadata.json", meta)
    save_json(EXP_ROOT / "dapt" / f"{variant_name}_summary.json", meta)

    del model, optimizer, scheduler, best_state
    clear_cuda()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["muril_dapt_inductive_mlm_4ep"],
        choices=[
            "muril_dapt_inductive_mlm_4ep",
            "muril_dapt_inductive_wwm_4ep",
            "muril_dapt_inductive_wwm_8ep",
        ],
    )
    args = parser.parse_args()

    data = build_strictly_inductive_dapt_corpus()
    m = data["manifest"]
    print(
        f"Inductive DAPT corpus ready: train_threads={m['dapt_train_threads']} ({m['dapt_train_docs']} docs), "
        f"val_threads={m['dapt_val_threads']} ({m['dapt_val_docs']} docs), "
        f"excluded_test_threads={m['excluded_frozen_test_threads']} ({m['excluded_frozen_test_comments']} comments), "
        f"test_overlap={m['intersection_dapt_train_and_test_threads']}/{m['intersection_dapt_val_and_test_threads']}",
        flush=True,
    )

    specs = {
        "muril_dapt_inductive_mlm_4ep": dict(masking_mode="standard", epochs=4, early_stop_patience=0),
        "muril_dapt_inductive_wwm_4ep": dict(masking_mode="wwm", epochs=4, early_stop_patience=0),
        "muril_dapt_inductive_wwm_8ep": dict(masking_mode="wwm", epochs=8, early_stop_patience=2),
    }
    for v in args.variants:
        train_inductive_dapt_variant(v, data["train_texts"], data["val_texts"], **specs[v])


if __name__ == "__main__":
    main()
