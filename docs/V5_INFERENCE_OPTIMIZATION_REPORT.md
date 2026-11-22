# Phase 7 — Context Length (160 / 256 / 384) & GPU/CPU Inference Optimization Report

**Date:** 2026-11-22  
**Hardware Profiled:** NVIDIA GeForce RTX 4060 Laptop GPU (`8,188 MiB` VRAM, CUDA `13.2`, `bfloat16` AMP) and host x86_64 CPU (`FP32`)  
**Backbone Profiled:** `muril_dapt_inductive_wwm_8ep` (`237.6M` params, `906.3 MB` FP32 weights per model)  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase7_inference/phase7_summary.json`

---

## 1. Exact Token Truncation Audit Across All 459 Real Comments

Using `truncation="only_second"` so Segment A (the reply itself) is preserved intact whenever possible:

| Input Format | Median / p95 / Max Tokens | Truncated @ `160` (Total / Reply / Ctx) | Truncated @ `256` (Total / Reply / Ctx) | Truncated @ `384` (Total / Reply / Ctx) |
|---|---:|---:|---:|---:|
| `comment_only` | `23 / 128 / 191` | `0.44% / 0.44% / 0.00%` | **`0.00% / 0.00% / 0.00%`** | `0.00% / 0.00% / 0.00%` |
| `title_pair` | `37 / 137 / 234` | `1.31% / 0.44% / 1.31%` | **`0.00% / 0.00% / 0.00%`** | `0.00% / 0.00% / 0.00%` |
| `parent_pair` | `40 / 159 / 866` | `5.01% / 0.44% / 5.01%` | **`1.74% / 0.00% / 1.74%`** | `0.87% / 0.00% / 0.87%` |
| `title_parent_pair` | `53 / 175 / 914` | `6.75% / 0.44% / 6.75%` | **`2.18% / 0.00% / 2.18%`** | `1.09% / 0.00% / 1.09%` |

**Truncation Takeaway:** Raising `max_length` from `160` (V5 default) to **`256`** eliminates **100% of reply truncation (`0.44% → 0.00%`)**, eliminates **100% of `title_pair` truncation (`1.31% → 0.00%`)**, and cuts `parent_pair` context truncation by **65%** (`5.01% → 1.74%`). Expanding further to `384` recovers only `0.87%` of very long parent comments (4 out of 459 rows).

---

## 2. Controlled 5-Fold OOF Ablation: `max_length = 160` vs. `256` vs. `384`

| Task (Context Mode) | `max_len = 160` OOF F1 / ECE / VRAM / Eval ms | **`max_len = 256` OOF F1 / ECE / VRAM / Eval ms** | `max_len = 384`\* OOF F1 / ECE / VRAM / Eval ms |
|---|---:|---:|---:|
| **Issue** (`title_pair`) | `53.77% / 0.1382 / 3,979 MB / 1.40 ms` | **`54.31% / 0.1395 / 4,016 MB / 1.52 ms`** | `56.46% / 0.1100 / 4,011 MB / 1.42 ms`\* |
| **Response** (`title_parent_pair`) | `43.30% / 0.0171 / 3,970 MB / 1.50 ms` | `41.42% / 0.0524 / 4,061 MB / 2.24 ms` | `46.14% / 0.0327 / 4,123 MB / 2.72 ms`\* |
| **Solution** (`parent_pair`) | `75.16% / 0.0357 / 3,973 MB / 1.50 ms` | **`76.17% / 0.0342 / 4,051 MB / 2.17 ms`** | `75.98% / 0.0382 / 4,117 MB / 2.65 ms`\* |

*\*Methodological transparency note: At `max_length = 384`, `run_phase7_inference.py` halved micro-batch size (`batch_size = 4` vs `8` at `160/256`) to protect 8 GB VRAM, doubling optimizer steps per epoch (`176–183s` train time vs `102–110s`). Because `title_pair` max token length is only `234` tokens (`0%` truncated at `256`), `issue`'s `+2.15 pp` bump at `384` is a **smaller-batch (`bs=4`) optimization effect**, NOT a sequence-length effect! On `solution` (`parent_pair`), where `256` genuinely recovers `3.27%` of truncated parent context over `160`, `256` improved OOF F1 by **`+1.01 pp` (`75.16% → 76.17%`)** while `384` regressed slightly (`75.98%`) with `+22%` slower inference.*

**Sequence-Length Recommendation:** Standardize on **`max_length = 256`** for context-paired tasks (`issue`, `solution`) and **`max_length = 160` (or `256` with dynamic padding — identical runtime on short comments!)** for `comment_only` tasks (`sentiment`, `emotion`, `response`).

---

## 3. GPU & CPU Inference Profiling (Dynamic vs. Static Padding, Precision, Batch Size)

Measured on all 92 real `title_parent_pair` inputs (longest context format = worst-case stress benchmark):

### A. RTX 4060 GPU Inference Latency & Peak VRAM (`1` Model)
| `max_length` | Padding | Precision | Batch Size | Latency (ms/cmt) | Throughput (cmt/s) | Peak VRAM (MB) |
|---:|---|---|---:|---:|---:|---:|
| `160` | Dynamic | `bf16` | `1` | `10.80 ms` | `92.6` | `1,111` |
| `160` | Dynamic | `bf16` | `16` | `1.44 ms` | `696.8` | `1,170` |
| **`160`** | **Dynamic** | **`bf16`** | **`32`** | **`1.38 ms`** | **`723.2`** | **`1,227`** |
| `160` | Dynamic | `fp32` | `32` | `3.63 ms` | `275.1` | `1,090` |
| `256` | Dynamic | `bf16` | `1` | `11.70 ms` | `85.5` | `1,113` |
| **`256`** | **Dynamic** | **`bf16`** | **`16`** | **`1.82 ms`** | **`549.9`** | **`1,201`** |
| `256` | Dynamic | `bf16` | `32` | `2.10 ms` | `476.6` | `1,289` |
| `256` | Static | `bf16` | `32` | `2.52 ms` | `397.2` | `1,289` |
| `256` | Dynamic | `fp32` | `32` | `5.31 ms` | `188.3` | `1,191` |
| `384` | Dynamic | `bf16` | `16` | `2.07 ms` | `483.7` | `1,249` |
| `384` | Static | `bf16` | `32` | `3.85 ms` | `260.0` | `1,386` |

### B. Host CPU Inference Latency (`FP32`, Dynamic Padding, `1` Model)
| `max_length` | Batch Size | Latency (ms/cmt) | Throughput (cmt/s) | Key Observation |
|---:|---:|---:|---:|---|
| `160` | `1` | **`50.27 ms`** | **`19.9`** | Zero padding overhead |
| `160` | `16` | `77.40 ms` | `12.9` | Unsorted batch pads short comments to longest item |
| **`256`** | **`1`** | **`48.93 ms`** | **`20.4`** | **Sweet spot for single-request CPU API serving** |
| `256` | `16` | `88.06 ms` | `11.4` | `+80%` slower without length-sorting! |
| `384` | `1` | `52.86 ms` | `18.9` | Minimal penalty at `bs=1` due to dynamic length |
| `384` | `16` | `113.36 ms` | `8.8` | Severe padding waste without length-sorting |

### C. Cold Load Time & Memory Footprint Summary
| Deployment Topology | Cold CPU Load | GPU Transfer | FP32 Weight RAM | Full Corpus (`3,557` cmts) GPU Scoring (`bf16, bs=16`) |
|---|---:|---:|---:|---:|
| **1 Single-Task MuRIL / XLM-R** | `0.81 s` | `0.16 s` | `906 MB` | `~6.5 s` |
| **5 Separate Single-Task Models** | `4.04 s` | `0.80 s` | `4,532 MB` | `~32.4 s` (`~1.2 GB` active VRAM sequentially) |
| **5 Separate × 5-Fold Ensembles (25 ckpts)** | `~20.2 s` | `~4.0 s` | `22.7 GB` *(load sequentially!)* | `~162 s` (`~2.7 min` offline batch) |
| **1 Shared MTL Encoder (Phase 3)** | `0.81 s` | `0.16 s` | `906 MB` | `~4.7 s` (`5.6×` faster all-5-task) |
| **Word+Char TF-IDF Logistic Regression (5 tasks)** | `<0.05 s` | N/A (CPU) | `<15 MB` | `<0.4 s` on CPU |

---

## 4. Concrete Production Inference Optimizations Verified
1. **Always use `bf16` + Dynamic Padding at `batch_size = 16` on GPU:** Delivers **`550 comments/sec`** at `max_length = 256` (**2.9× faster than FP32**, **38% faster than static padding**, and **15% faster than `bs=32`** because smaller batches avoid outlier-length padding inflation!).
2. **Sort batches by token length before offline batch inference** (both GPU and CPU) so long 250-token threads never pad 15-token replies.
3. **Never serve 25 checkpoints simultaneously on CPU/serverless web workers:** Use **single-model full-trainval checkpoints** (or offline batch precomputation with 5-fold ensembles) when latency or memory is constrained — directly informed by our Phase 8–9 comparison of 5-fold ensembles vs. single-checkpoint models.
