# V6.3 — BERTopic Multi-Seed Stability & Expanded-Corpus Sensitivity Report

> **Preservation Guarantee:** `nmf_v1` (`topic_discovery.json`, 10 deterministic factors) and `bertopic_v1` (`bertopic_v1_topics.json`, 35 clusters, `38.2%` `-1` outliers) remain untouched.

## 1. Baseline Corpus (`n = 3,459` Docs) — Multi-Seed Stability Comparison

| Config ID | Seeds | Topic Counts Observed | Collapses (≤ 4 topics) | Mean Outliers `-1` | Mean Pairwise NMI | Mean Pairwise ARI (Non-Outlier) | Mean NPMI | Mean Diversity |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C0_v1_baseline_eom` | 7 | `[35, 31, 2, 30, 33, 34, 32]` | **1/7** | `30.5%` ± 12.5% | `0.5757` | `0.6109` | `-0.0718` | `0.9628` |
| `C1_eom_mcs35` | 7 | `[20, 2, 2, 20, 23, 20, 20]` | **2/7** | `23.1%` ± 14.7% | `0.4335` | `0.4583` | `-0.1205` | `0.9648` |
| `C2_leaf_mcs25` | 7 | `[37, 35, 35, 34, 34, 39, 34]` | **0/7** | `41.2%` ± 2.3% | `0.7697` | `0.9332` | `-0.0983` | `0.9554` |
| `C3_leaf_nn25_md005` | 7 | `[29, 29, 27, 29, 28, 28, 25]` | **0/7** | `52.3%` ± 1.9% | `0.7456` | `0.9140` | `-0.1080` | `0.9514` |
| `C4_eom_nn30_ms12` | 5 | `[27, 18, 24, 28, 26]` | **0/5** | `41.3%` ± 4.9% | `0.7720` | `0.8402` | `-0.1079` | `0.9596` |
| `C5_minilm_leaf_mcs25` | 5 | `[43, 45, 42, 42, 41]` | **0/5** | `35.3%` ± 3.7% | `0.8090` | `0.9467` | `-0.1331` | `0.9793` |

## 2. Expanded V6.3 Corpus (`n = 7152`) — 5-Seed Stability Comparison

| Config ID | Seeds | Topic Counts Observed | Collapses | Mean Outliers `-1` | Mean Pairwise NMI | Mean Pairwise ARI (Non-Outlier) | Mean NPMI |
|---|---:|---|---:|---:|---:|---:|---:|
| `C0_v1_baseline_eom_expanded_v63` | 5 | `[2, 41, 36, 39, 38]` | **1/5** | `33.8%` | `0.4479` | `0.5548` | `-0.1021` |
| `C1_eom_mcs35_expanded_v63` | 5 | `[2, 23, 24, 2, 24]` | **2/5** | `24.1%` | `0.3203` | `0.3746` | `-0.1140` |
| `C2_leaf_mcs25_expanded_v63` | 5 | `[57, 51, 52, 50, 49]` | **0/5** | `59.1%` | `0.6959` | `0.9072` | `-0.1364` |

## 3. Failure Analysis & Recommendation
- **Why `eom` (`C0_v1_baseline_eom`) collapses on certain seeds:** HDBSCAN Excess-of-Mass (`eom`) merges leaf micro-clusters into 2 dense super-basins when UMAP stochastic layout creates a slight bridge across short conversational comments.
- **Most stable tested BERTopic configuration:** `C5_minilm_leaf_mcs25` (`0/5` collapses, mean pairwise NMI `0.8090`, outlier rate `35.3%`).
- **Production Default Recommendation:** Keep **NMF (`nmf_v1`) as the default deterministic topic engine** and retain BERTopic (`intfloat/multilingual-e5-small`) as the interactive exploratory companion rather than replacing NMF.
