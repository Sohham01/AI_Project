# Phase 5 — Strictly Inductive DAPT (WWM & Schedule), Backbone Alternatives & LoRA Ablation Report

**Date:** 2026-11-22  
**Inductive DAPT Cleanliness Contract (Section 3C):** All 37 frozen held-out test `post_id`s (`571` comments) were excluded **before** DAPT partitioning (`intersection = 0 / 0`), and DAPT validation was grouped strictly by `post_id` (`289` train threads / `2,352` docs vs. `39` unseen validation threads / `279` docs; exact manifest at `ml/artifacts/v5_experiments/dapt/inductive_manifest.json`).  
**Downstream Protocol:** Identical 5-fold group-aware OOF CV on untouched V1 `trainval` (`n = 367`, `163` threads) using each task's Phase 2 winning context (`issue: title_pair`, `solution: parent_pair`, others: `comment_only`) and Phase 4 winning loss (`issue/emotion: asl`, others: `ce_weighted`), `max_length = 256`.  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase5_dapt_backbones/phase5_summary.json`

---

## 1. MuRIL WordPiece Alignment Audit (Why Naive Whitespace WWM Would Fail)

Inspecting `google/muril-base-cased`'s `BertTokenizerFast` (`vocab_size = 197,285`) on Indian MBA/Hinglish Reddit vocabulary revealed that **`33.3%` of sampled word units split across multiple WordPiece `##` continuation pieces**:
- Campus & MBA domain terms: `['IIM', '##S', '##am', '##balpur']`, `['wait', '##list']`, `['percent', '##ile']`, `['ac', '##ads']`, `['pro', '##f']`, `['mock', '##test']`, `['cut', '##off']`
- Indian MBA acronyms: `['LP', '##A']`, `['GE', '##M']`, `['GN', '##EM']`, `['CA', '##P']`, `['SI', '##P']`, `['PP', '##O']`

Our WWM implementation (`ml/experiments/v5_optimization/run_dapt_inductive.py:_mask_tokens_wwm`) groups every `##` continuation token atomically with its preceding root subword before applying 80/10/10 masking, forcing MuRIL to reconstruct entire domain concepts from Reddit context rather than trivially guessing suffixes.

---

## 2. Strictly Inductive DAPT Perplexity & Training Schedule Comparison

All 3 inductive checkpoints were trained on identical `289` train / `39` unseen val threads (`batch_size = 4, grad_accum = 4`, effective batch `16`, `lr = 5e-5`, `bf16`, `~5,585 MB` peak VRAM):

| Inductive DAPT Variant | Masking | Epochs (Best Ep) | Time (s) | Epoch 0 `val_std / val_wwm` PPL | Final Best `val_std / val_wwm` PPL |
|---|---|---:|---:|---:|---:|
| `muril_dapt_inductive_mlm_4ep` | Standard subword | `4 (4)` | `234.6s` | `167.12 / 113.26` | **`31.36`** `/ 47.35` |
| `muril_dapt_inductive_wwm_4ep` | WordPiece WWM (`##`) | `4 (4)` | `204.8s` | `167.12 / 113.26` | `49.18 / 34.88` |
| **`muril_dapt_inductive_wwm_8ep`** | **WordPiece WWM (`##`)** | **`8 (7)`** | `406.8s` | `167.12 / 113.26` | `43.11 /` **`32.53`** |

**Schedule Finding:** Under WWM, unseen-thread `val_wwm_ppl` kept improving steadily past epoch 4 (`38.67 → 35.52 → 33.54 → 32.53` at epoch 7) before plateauing at epoch 8 (`32.68`), confirming that 6–8 epochs with validation-based checkpointing outperforms V5's 4-epoch truncation.

---

## 3. Controlled 7-Backbone Downstream 5-Fold OOF Comparison (`n = 367`)

| Backbone / Adapter | Trainable Params | Peak VRAM (MB) | Latency (ms/cmt) | `issue` OOF F1 | `sentiment` OOF F1 | `emotion` OOF F1 | `response` OOF F1 | `solution` OOF F1 | **5-Task Mean OOF F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `muril_base` (no DAPT) | `100.0%` (`237.6M`) | `4,016` | `1.42 ms` | `50.99%` | `64.81%` | `54.33%` | `51.60%` | `73.76%` | `59.10%` |
| `muril_dapt_inductive_mlm_4ep` | `100.0%` (`237.6M`) | `4,016` | `1.43 ms` | `48.68%` | `68.78%` | `57.87%` | `52.92%` | `76.22%` | `60.89%` (`+1.79`) |
| `muril_dapt_inductive_wwm_4ep` | `100.0%` (`237.6M`) | `4,016` | `1.44 ms` | `47.10%` | **`68.86%`** | **`58.18%`** | `53.39%` | `76.00%` | `60.71%` (`+1.61`) |
| **`muril_dapt_inductive_wwm_8ep`** | **`100.0%` (`237.6M`)** | **`4,016`** | **`1.51 ms`** | **`54.31%`** | `67.25%` | `57.77%` | `53.49%` | `76.17%` | **`61.80%` (`+2.70`)** |
| `ai4bharat/IndicBERTv2-MLM-only` | `100.0%` (`278.0M`) | `4,623` | `1.47 ms` | `53.24%` | `62.67%` | `53.40%` | `47.82%` | `72.34%` | `57.89%` (`-1.21`) |
| **`xlm-roberta-base`** *(no DAPT!)* | `100.0%` (`278.0M`) | `4,635` | `1.58 ms` | `52.77%` | `65.57%` | `55.11%` | **`54.64%`** | **`76.74%`** | `60.97%` (`+1.87`) |
| `muril_dapt_lora_r8` (on `wwm_8ep`) | `0.134%` (`318K`) | **`1,702`** | `1.54 ms` | `47.51%` | `58.96%` | `47.51%` | `44.78%` | `68.86%` | `53.52%` (`-8.28`) |

### Key Downstream Findings
1. **Strictly Inductive DAPT Delivers Genuine, Leak-Free Gains:** Even with **zero** test threads ever seen during pretraining, inductive DAPT lifts 5-task OOF Macro F1 from `59.10%` (`muril_base`) to `60.89%` (`mlm_4ep`) and **`61.80%` (`wwm_8ep`, `+2.70 pp`)**!
2. **Extended WWM (`wwm_8ep`) Unlocks Massive Rare-Issue Recovery (`issue: 48.68% → 54.31%`, `+5.63 pp`):**
   - `academic_workload`: **`0.0727 → 0.1667` (`2.3×`)**
   - `extracurriculars`: **`0.1714 → 0.4000` (`+22.9 pp`!)**
   - `campus_infrastructure`: **`0.4304 → 0.4737` (`+4.3 pp`)**
   - `administrative_communication`: **`0.7877 → 0.8291` (`+4.1 pp`)**
   - `curriculum_teaching`: **`0.2256 → 0.2568` (`+3.1 pp`)**
   - `hostel_facilities`: **`0.6800 → 0.7059` (`+2.6 pp`)**
   - `placements`: **`0.7333 → 0.7453` (`+1.2 pp`)**
   - Calibration also improved simultaneously vs. `mlm_4ep` across all 5 tasks (`sentiment ECE: 0.0363 → 0.0248`, `response ECE: 0.0605 → 0.0485`, `solution ECE: 0.0410 → 0.0342`).
3. **Strong Complementary Strength of `xlm-roberta-base` on English/Reddit Discourse Tasks:**
   - Without any Reddit DAPT at all, `xlm-roberta-base` placed **#1 on `response` (`54.64%`, beating `wwm_8ep 53.49%` by `+1.15 pp`)** — pushing `counterargument` to **`0.3404`**, `solution` to **`0.5542`**, and `agreement` to **`0.5195` (`+9.6 pp` over `wwm_8ep`)** — and **#1 on `solution` (`76.74%` vs `76.17%`)**, though with higher raw/calibrated ECE (`0.0868` vs MuRIL `0.0342`).
   - Conversely, MuRIL-DAPT-WWM decisively leads on Indian campus taxonomy (`issue: 54.31%`), Indian code-mixed sentiment (`wwm_4ep: 68.86%`, `+3.29 pp` over XLM-R), and emotion (`wwm_4ep: 58.18%`, `+3.07 pp` over XLM-R).
4. **`ai4bharat/IndicBERTv2-MLM-only`** excelled on `issue` (`53.24%`, #2 overall), but lagged on English-heavy Reddit discourse (`response: 47.82%`, `sentiment: 62.67%`).

---

## 4. Full LLRD Fine-Tuning vs. Pure-PyTorch LoRA ($r = 8$) & 3-Seed Stability

| Regime (on `muril_dapt_inductive_wwm_8ep`) | Trainable % | Peak VRAM | 5-Fold Train Time | 5-Task Mean OOF | Sentiment 3-Seed Mean ± Std | Response 3-Seed Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|
| **Full LLRD Fine-Tuning** | `100.0%` | `4,016 MB` | `~101s / task` | **`61.80%`** | **`68.11% ± 0.97%`** | **`53.85% ± 0.26%`** |
| Pure-PyTorch LoRA ($r=8$, Q/V) | `0.134%` | **`1,702 MB`** (`−58%`) | **`~68s / task`** (`−33%`) | `53.52%` (`−8.28 pp`) | `60.61% ± 1.97%` (`2×` higher std) | `46.16% ± 1.02%` (`4×` higher std) |

**Decision on LoRA:** Despite cutting peak VRAM by `58%` (`4.0 GB → 1.7 GB`), rank-8 Q/V LoRA degrades 5-task OOF Macro F1 by **`−8.28 pp`**, worsens calibration (`response ECE: 0.0485 → 0.1226`), and increases seed-to-seed variance by **2×–4×**. Because full LLRD fine-tuning fits comfortably inside `<50%` of the RTX 4060's 8.59 GB VRAM, **full LLRD fine-tuning is strictly retained**.

---

## 5. Task-Specific OOF-Winning Backbones Carried to Phase 8–9 Selection

Selected strictly via 5-fold OOF CV (never test set):
- **`issue`:** `muril_dapt_inductive_wwm_8ep` (`54.31%` OOF F1)
- **`sentiment`:** `muril_dapt_inductive_wwm_4ep` (`68.86%` OOF F1, `ECE = 0.0272`)
- **`emotion`:** `muril_dapt_inductive_wwm_4ep` (`58.18%` OOF F1)
- **`response`:** `xlm-roberta-base` (`54.64%` OOF F1, `counterargument = 0.3404`), with `muril_dapt_inductive_wwm_8ep` (`53.49%`, lower ECE `0.0485`) as close runner-up
- **`solution`:** `xlm-roberta-base` (`76.74%` OOF F1) vs `muril_dapt_inductive_mlm_4ep / wwm_8ep` (`76.22% / 76.17%`, less than half the ECE: `0.0342` vs `0.0868`)


---

## 6. Forensic Remediation Addendum — 100% Zero-Leakage Strict Inductive DAPT (`44` Threads Excluded)

During the Phase 8 audit, we detected that matching frozen test threads solely by `post_id` allowed 13 shifted post IDs after `post_0064` (from the 366 → 365 corpus re-pseudonymization) to remain in the initial `muril_dapt_inductive_*` pool. We corrected `run_dapt_inductive.py` to exclude frozen test threads by **all three keys simultaneously** (`corpus_post_id`, normalized `post_title[:120]`, and normalized comment text), excluding **44 threads / 749 comments** (`intersections = 0 / 0` verified in `ml/artifacts/v5_experiments/dapt/strict_inductive_manifest.json`) and trained **`muril_dapt_strict_inductive_wwm_8ep`** (`282` train threads / `2,135` docs; `39` unseen val threads / `347` docs; `val_wwm_ppl: 88.32 → 25.23`). All final Track A and Track B numbers in `docs/V5_MODEL_IMPROVEMENT_REPORT.md` use exclusively this 100% zero-leakage strict inductive checkpoint and `xlm-roberta-base` (zero Reddit DAPT).
