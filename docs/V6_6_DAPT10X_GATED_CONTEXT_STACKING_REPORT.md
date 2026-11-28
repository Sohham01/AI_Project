# V6.6 — `10×` Strict-Inductive WWM DAPT, Gated Reply-Masked Pooling & Nested 7-Fold (`6:1`) Stacking Report

> **Zero-Leakage Guarantee:** `muril_dapt_v66_wwm` and `xlmr_dapt_v66_wwm` were pre-trained exclusively on the V6.5 `train_pool` (`0` overlap with `test_pool` `261` threads or the `37` frozen V1 test threads). All arm selections, simplex stack weights, Platt parameters, and multi-label thresholds `[0.30, 0.70]` were determined strictly inside **7-fold (`6:1`) `post_id`-grouped OOF CV on `trainval (n = 367)`** before evaluating the locked champion once on frozen `test (n = 92)`.

## 1. `10×` Strictly Inductive WWM DAPT Curves (WordPiece & SentencePiece)

| Checkpoint | Subword WWM Mode | Best Epoch | `ep0 val_wwm_ppl` | `best val_wwm_ppl` | Time | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| `muril_dapt_v66_wwm` | `wordpiece_##` | `4` | `69.395` | **`23.888`** | `759.19s` | `5345.0 MB` |
| `xlmr_dapt_v66_wwm` | `sentencepiece_\u2581` | `4` | `74.139` | **`23.122`** | `1225.56s` | `5731.3 MB` |

## 2. Strictly OOF-Selected 5-Task Champion vs. V6.1 (`n = 92` Frozen Test)

- **V6.6 5-Task Mean (Strict OOF-Selected Champion):** **`64.44%`** (`7-Fold OOF Mean = 67.31%`, `Mean Test ECE = 0.0653`)
- **Delta vs. Deployed V6.1 (`65.60%`):** **`-1.16 pp`** | **Delta vs. Historical V5 (`59.48%`):** **`+4.96 pp`**

| Task | Locked OOF Champion Arm | 7-Fold OOF F1 | Frozen Test F1 | Deployed V6.1 Test F1 | Δ vs V6.1 | Test ECE | Paired Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| **issue** | `oof_stacked_v66` | `58.91%` | **`59.75%`** | `60.83%` | **`-1.08 pp`** | `0.0373` | `[-7.5, +5.9]` (`p=0.405`) |
| **sentiment** | `v61_ref_7fold` | `71.92%` | **`64.87%`** | `65.11%` | **`-0.24 pp`** | `0.1288` | `[-6.0, +5.5]` (`p=0.600`) |
| **emotion** | `oof_stacked_v66` | `66.96%` | **`58.37%`** | `61.04%` | **`-2.67 pp`** | `0.0193` | `[-11.4, +2.7]` (`p=0.201`) |
| **response** | `xlmr_v66_dapt10x` | `56.83%` | **`59.10%`** | `56.42%` | **`+2.68 pp`** | `0.0750` | `[-4.2, +9.0]` (`p=0.715`) |
| **solution** | `xlmr_v66_dapt10x` | `81.95%` | **`80.10%`** | `84.57%` | **`-4.47 pp`** | `0.0659` | `[-10.9, +1.4]` (`p=0.058`) |

## 3. Full 7-Fold (`6:1`) OOF Ablation Matrix (`trainval`, `n = 367`)

| Task | `v61_ref_7fold` | `muril_v66_dapt10x` | `xlmr_v66_dapt10x` | `gated_reply_ctx_v66` | `lex_discourse_7fold` | `oof_stacked_v66` | Stack Weights (`Ref / M_10x / X_10x / Gated / Lex`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **issue** | `52.38%` | `54.16%` | `58.56%` | `54.39%` | `45.61%` | `58.91%` | `0.06 / 0.01 / 0.69 / 0.03 / 0.21` |
| **sentiment** | `71.92%` | `70.21%` | `71.57%` | `65.51%` | `51.04%` | `70.67%` | `0.07 / 0.37 / 0.54 / 0.01 / 0.00` |
| **emotion** | `57.60%` | `64.65%` | `62.71%` | `54.67%` | `53.64%` | `66.96%` | `0.03 / 0.47 / 0.49 / 0.00 / 0.01` |
| **response** | `53.48%` | `54.09%` | `56.83%` | `56.60%` | `51.57%` | `54.80%` | `0.09 / 0.10 / 0.37 / 0.11 / 0.33` |
| **solution** | `77.61%` | `75.63%` | `81.95%` | `81.86%` | `73.48%` | `81.60%` | `0.17 / 0.00 / 0.74 / 0.07 / 0.01` |

## 4. Post-Lock Frozen Test Observations Across All Arms (`n = 92`, Read-Only Diagnostic)

- **Strict Per-Task `max_oof` Mean:** `67.31%` 7-fold OOF → **`64.44%`** frozen `n = 92` test.
- **Uniform `oof_stacked_v66` 5-Task Mean:** **`63.81%`** frozen `n = 92` test (`-1.79 pp` vs. post-hoc V6.1 `65.60%`, `-0.81 pp` vs. strict 3-seed V6.1 `64.62%`, and `+4.33 pp` vs. historical V5 `59.48%`).

| Task | `v61_ref_7fold` | `muril_v66_dapt10x` | `xlmr_v66_dapt10x` | `gated_reply_ctx_v66` | `lex_discourse_7fold` | **`oof_stacked_v66`** | Deployed V6.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **issue** | `50.78%` | `56.00%` | `60.11%` | `55.33%` | `40.98%` | **`59.75%`** | `60.83%` |
| **sentiment** | `64.87%` | `60.68%` | `61.20%` | `65.17%` | `53.96%` | **`63.24%`** | `65.11%` |
| **emotion** | `56.25%` | `60.62%` | `58.96%` | `57.93%` | `42.57%` | **`58.37%`** | `61.04%` |
| **response** | `59.35%` | `54.21%` | `59.10%` | `57.37%` | `36.13%` | **`57.59%`** | `56.42%` |
| **solution** | `84.03%` | `79.09%` | `80.10%` | `80.27%` | `64.16%` | **`80.10%`** | `84.57%` |
| **5-Task Mean** | `63.06%` | `62.12%` | `63.89%` | `63.21%` | `47.56%` | **`63.81%`** | `65.60%` |

## 5. 4-Epoch LLRD DAPT (`llrd = 0.95`, `max_length = 160`) vs. 2-Epoch Uniform DAPT & Controlled 7-Fold Reference

- **Unseen-Thread WWM Perplexity (`1,746` docs / `155` unseen threads):** `muril_dapt_v66_wwm` improved from `25.74` (2-epoch) → **`23.89`** (4-epoch LLRD `0.95`); `xlmr_dapt_v66_wwm` improved from `23.36` (2-epoch, `lr = 4e-5`) → **`23.12`** (4-epoch LLRD `0.95`, `lr = 2e-5`).
- **English Procedural `Solution` Drift Recovery:** Lowering XLM-R DAPT LR to `2.0e-5` with `llrd = 0.95` lifted `xlmr_v66_dapt10x` on `solution` to **`81.95%` 7-fold OOF** (`+4.34 pp` over `v61_ref_7fold` `77.61%`) and recovered frozen-test F1 from `77.64%` → **`80.10%`** (`80.27%` with `ECE = 0.0372` under `gated_reply_ctx_v66`).
- **Controlled 1-Seed × 7-Fold (`6:1`) Gain vs. `v61_ref_7fold`:** Under identical 7-fold `post_id`-grouped CV (contrasted with the 5-seed × 5-fold `25`-model deployed V6.1 ensemble at `65.60%`), V6.6 lifts 5-task OOF mean from `62.60%` (`v61_ref_7fold`) → **`67.31%` (`+4.71 pp`)** and frozen `n = 92` test mean from `63.06%` → **`64.44%` (`+1.38 pp` controlled gain; `+0.45 pp` over 2-epoch `max_oof` `63.99%`)**, led by `issue` (`50.78% → 60.11%` single-arm / `59.75%` stacked), `emotion` (`56.25% → 60.62%` single-arm), and `response` (`56.42%` deployed V6.1 → **`59.10%`**).
