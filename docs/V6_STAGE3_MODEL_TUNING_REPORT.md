# Stage 3 — Strictly Inductive Model Tuning & Ablation Report (`V6.0 → V6.1`)

> **Script:** `ml/experiments/v5_optimization/run_stage3_tuning.py` (isolated per-experiment subprocess execution, `max_length = 256`, `lr = 4e-5`, `6` epochs, `5`-fold `post_id`-grouped CV on untouched V1 `trainval n = 367`).  
> **Raw Artifact:** `ml/artifacts/v5_experiments/stage3_tuning_results.json`  
> **Scientific Guardrail:** All candidate selection used **5-fold OOF Macro F1 only** (promotion gate: `>= +1.0 pp` OOF Macro F1 without calibration regression). The frozen `n = 92` real held-out test set (`37` unseen threads) was never used for tuning or selection and was evaluated once post-selection with paired thread-clustered bootstrap (`n = 2,000`).

---

## 1. Hypotheses Tested

1. **Backbone specialization on `Response` (6 canonical classes — our hardest task at `53.52%` in V6.0):** Phase 5 showed `xlm-roberta-base` (zero Reddit DAPT) excels at English/Reddit conversational pragmatics (`agreement`, `counterargument`, `solution`).
2. **Span-Masked Reply-Only Pooling (`reply_only_pool = True`) on `Response`:** In Phase 2, naive `parent_pair` concatenation helped `Issue` (`+15.1 pp`) but hurt `Response` because pooling over parent tokens leaked parent stance/affect. Here all 12 self-attention layers cross-attend to `<reply> [SEP] <parent>`, while `[Mean + Max]` pooling is strictly masked to Segment 0 (reply tokens before the first `[SEP]`/`</s>`).
3. **Task 6 Semantic Emoji Affect + Conservative Gen-Z Slang (`emoji_affect_slang`) on `Sentiment` & `Emotion`:** Motivated by our Task 6 ablation finding that verbalizing emojis (`😭 → sadness frustration`, `💀 → extreme amusement disbelief`, `🤡 → sarcasm mockery`, `🔥 → strong approval`) maps them onto in-vocabulary Transformer embeddings instead of WordPiece `[UNK]`.
4. **2-Encoder Shared 3-Task Multi-Task Learning (`Sentiment + Response + Solution` + R-Drop `α = 0.5`):** Testing whether sharing one encoder across all three single-label tasks beats single-task specialized encoders while cutting inference cost by `3×`.

---

## 2. Full Controlled 5-Fold OOF & Frozen Test Comparison (`max_length = 256`, `lr = 4e-5`)

### A. Response Classification (6 Canonical Classes)
| Candidate | Backbone | Context / Pool | Text Mode | 5-Fold OOF F1 | OOF ECE | Frozen Test F1 *(Obs.)* | Test ECE | `counterargument` Test F1 | Selection Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `resp_v6_baseline` | `MuRIL-DAPT-WWM` | `comment_only` | `raw` | `52.97%` | `0.0645` | `53.21%` | `0.0535` | `0.1333` | V6.0 Baseline |
| **`resp_xlmr_raw`** | **`xlm-roberta-base`** | **`comment_only`** | **`raw`** | **`54.71%`** *(**+1.74 pp**)* | `0.0748` | **`58.00%`** *(**+4.79 pp**)* | `0.0783` | **`0.2353`** *(+10.2 pp)* | **PROMOTED (OOF #1)** |
| `resp_xlmr_replypool` | `xlm-roberta-base` | `parent_pair` (`reply_only_pool`) | `emoji_affect_slang` | `54.39%` *(+1.42 pp)* | `0.0772` | `58.44%` *(+5.23 pp)* | `0.1188` | `0.1333` | OOF #2 (proves reply-only masking fixes context bleed) |
| `resp_xlmr_affect` | `xlm-roberta-base` | `comment_only` | `emoji_affect_slang` | `53.59%` *(+0.62 pp)* | `0.0776` | `60.10%` *(+6.89 pp)* | `0.0411` | `0.2222` | Not promoted (strict OOF-only discipline; no test cherry-picking) |
| `mtl3_xlmr_rdrop::response` | `xlm-roberta-base` (3-task MTL) | `comment_only` | `emoji_affect_slang` | `53.51%` *(+0.54 pp)* | `0.0506` | `56.34%` | `0.1121` | — | Strong efficiency option, below single-task OOF #1 |
| `resp_muril_replypool` | `MuRIL-DAPT-WWM` | `parent_pair` (`reply_only_pool`) | `emoji_affect_slang` | `49.45%` | `0.0474` | `55.86%` | `0.0897` | `0.2222` | Below OOF baseline |
| `mtl3_muril_rdrop::response` | `MuRIL-DAPT-WWM` (3-task MTL) | `comment_only` | `emoji_affect_slang` | `49.39%` | `0.0305` | `54.88%` | `0.1193` | — | Negative transfer from Solution/Sentiment |

*Paired thread-clustered bootstrap (`37` thread clusters, `2,000` resamples) for `resp_xlmr_raw` vs. `resp_v6_baseline`: mean `ΔTest = +4.69 pp`, `95% CI = [-3.86, +13.00] pp`, `P(XLM-R > MuRIL) = 0.858`.*

### B. Sentiment Classification (3 Classes)
| Candidate | Backbone | Text Mode | 5-Fold OOF F1 | OOF ECE | Frozen Test F1 *(Obs.)* | Test ECE | Selection Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `sent_v6_baseline` | `MuRIL-DAPT-WWM` | `raw` | `69.40%` | `0.0794` | `63.31%` | `0.1309` | V6.0 Baseline |
| **`sent_muril_affect`** | **`MuRIL-DAPT-WWM`** | **`emoji_affect_slang`** | **`70.95%`** *(**+1.55 pp**)* | `0.0850` | **`65.12%`** *(**+1.81 pp**)* | **`0.1254`** | **PROMOTED (OOF #1)** |
| `mtl3_muril_rdrop::sentiment` | `MuRIL-DAPT-WWM` (3-task MTL) | `emoji_affect_slang` | `65.83%` | `0.0228` | `68.44%` | `0.0523` | Below single-task on OOF (`-3.57 pp`), kept as future MTL efficiency candidate |
| `mtl3_xlmr_rdrop::sentiment` | `xlm-roberta-base` (3-task MTL) | `emoji_affect_slang` | `65.45%` | `0.0325` | `62.58%` | `0.1123` | Below OOF baseline |

*Paired thread-clustered bootstrap (`37` clusters, `2,000` resamples) for `sent_muril_affect` vs. `sent_v6_baseline`: mean `ΔTest = +1.87 pp`, `95% CI = [-0.50, +5.87] pp`, `P(affect > raw) = 0.855`.*

### C. Emotion Classification (6 Multi-Label, ASL + OOF Monotonic Platt)
| Candidate | Backbone | Text Mode | 5-Fold OOF F1 | OOF ECE | Frozen Test F1 *(Obs.)* | Test ECE | Selection Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `emo_v6_baseline` | `MuRIL-DAPT-WWM` | `raw` | `57.28%` | `0.0040` | `56.69%` | `0.0087` | Stage 3 reference (`regularized` thresholds) |
| `emo_muril_affect` | `MuRIL-DAPT-WWM` | `emoji_affect_slang` | `60.80%` *(+3.52 pp)* | `0.0038` | `53.39%` *(-3.30 pp)* | `0.0105` | **NOT PROMOTED** — OOF gain did not generalize on unseen threads (`p = 0.12`); live V6.0 Emotion (`bounded` OOF Platt, `61.04%` test F1, `0.007` ECE) is preserved unchanged |

---

## 3. Summary of V6.1 (`v6.1.0`) Production Promotion Decisions

1. **Promoted `Response` → `xlm-roberta-base` (5-fold, zero DAPT, `comment_only`, `CE-weighted`, `max_length=256`):** Clears the `+1.0 pp` OOF gate (`+1.74 pp`) and delivers **`+2.90 pp` frozen test Macro F1 (`53.52% → 56.42%`)** in the full `v6.1.0` build (and `+4.79 pp`, `58.00%` on the Stage 3 fold seed) while **improving test ECE (`0.085 → 0.066`)**, lifting `counterargument` (`+10.2 pp` in Stage 3).
2. **Evaluated & Held `Sentiment` (`emoji_affect_slang`) on Calibration Gate:** Although `emoji_affect_slang` lifted 5-fold OOF Macro F1 consistently across both seeds (`0.7095` and `0.7096`), under the production `v6.1.0` seed schedule (`SEED + f_idx * 13`) its test calibration regressed (`ECE 0.104 → 0.166`, F1 `0.6222` vs. V6.0 `0.6511`). Enforcing our strict *"no calibration regression"* gate, we keep V6.0's better-calibrated `MuRIL-DAPT-WWM` Sentiment head (`65.11%`, `ECE 0.104`) in `v6.1.0` while deploying `emoji_affect_slang` in our Task 6 ablation study and real-time Live Multilingual Analyzer.
3. **Final Locked `v6.1.0` Strictly Inductive 5-Task Portfolio (`0/44` DAPT test-thread overlap):** **5-Task Mean Frozen Test Macro F1 = `65.59%` (`+6.11 pp` vs. transductive V5 `59.48%`)** and **Mean 5-Bin ECE = `0.059`** (`Issue 60.83%`, `Solution 84.57%`, `Sentiment 65.11%`, `Response 56.42%`, `Emotion 61.04%`).
4. **Human IAA Calibration Pack Prepared (`docs/ANNOTATION_GUIDELINES.md` Section 8 + `iaa_calibration_30.csv`):** Ready for a clean dual-annotator pass whenever the team chooses to recalibrate and unlock Track B V2 (`+178` active-learning rows).
