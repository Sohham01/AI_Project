# Phase 2 — Native Segment-Pair Parent Context Ablation Report

**Date:** 2026-11-22  
**Backbone:** Strictly Inductive `muril_dapt_inductive_mlm_4ep` (`max_length = 256`, `truncation="only_second"`, native `token_type_ids`)  
**Evaluation Protocol:** Identical 5-fold group-aware OOF CV on `trainval` (`n = 367`, `163` threads; `0` leakage; `n = 92` frozen test untouched for selection). Multi-label scores use honest nested out-of-fold bounded `[0.30, 0.70]` thresholds.  
**Legitimate Context Guarantee:** Strictly uses ONLY the comment itself, its parent post title (`100%` coverage), and/or its immediate parent comment (`55.3%` coverage). **Zero** child replies, descendant comments, sibling comments, or predicted labels are ever included.  
**Artifact Provenance:** `ml/artifacts/v5_experiments/phase2_context/phase2_summary.json`

---

## 1. Truncation Policy & Token Coverage (`max_length = 256`, `truncation="only_second"`)

By placing the current reply in Segment A and context in Segment B (`[CLS] reply [SEP] context [SEP]`) with `truncation="only_second"`, **100.0% of every reply is preserved intact** across all 4 input configurations:

| Input Variant | Format | Median Tokens | p95 Tokens | Max Tokens | Reply Truncated | Context Truncated | Total Truncated |
|---|---|---:|---:|---:|---:|---:|---:|
| `comment_only` | `[CLS] reply [SEP]` | 22 | 130 | 191 | `0.00%` | `0.00%` | `0.00%` |
| `title_pair` | `[CLS] reply [SEP] title [SEP]` | 37 | 139 | 234 | `0.00%` | `0.00%` | `0.00%` |
| `parent_pair` | `[CLS] reply [SEP] parent_or_title [SEP]` | 39 | 166 | 866 | `0.00%` | `1.91%` | `1.91%` |
| `title_parent_pair` | `[CLS] reply [SEP] Post: title \| Parent: parent [SEP]` | 51 | 185 | 914 | `0.00%` | `2.45%` | `2.45%` |

---

## 2. Five-Task Context Ablation Results (5-Fold OOF CV, `n = 367`)

Context does **not** help every task uniformly — our controlled ablation reveals a sharp, task-dependent split between **discourse/topic tasks (`Issue`, `Solution`)** that benefit substantially from legitimate parent context, and **speaker-affect/stance tasks (`Sentiment`, `Emotion`, `Response`)** where pooling over context tokens degrades OOF performance:

| Task | `comment_only` OOF F1 / ECE | `title_pair` OOF F1 / ECE | `parent_pair` OOF F1 / ECE | `title_parent_pair` OOF F1 / ECE | OOF-Selected Winner | Δ vs `comment_only` |
|---|---:|---:|---:|---:|---|---:|
| **Issue** | `41.49% / 0.0498` | **`46.92% / 0.0423`** | `45.69% / 0.0461` | `46.84% / 0.0461` | **`title_pair`** | **`+5.43 pp`** |
| **Sentiment** | **`68.78% / 0.0363`** | `64.69% / 0.0140` | `64.82% / 0.0289` | `62.88% / 0.0254` | **`comment_only`** | `0.00 pp` (ctx hurts `−3.96 to −5.90 pp`) |
| **Emotion** | **`55.94% / 0.0513`** | `44.20% / 0.0720` | `42.67% / 0.0704` | `44.06% / 0.0725` | **`comment_only`** | `0.00 pp` (ctx hurts `−11.74 to −13.27 pp`) |
| **Response** | **`52.92% / 0.0605`** | `45.74% / 0.0642` | `41.42% / 0.0376` | `43.37% / 0.0677` | **`comment_only`** | `0.00 pp` (ctx hurts `−7.18 to −11.50 pp`) |
| **Solution** | `74.61% / 0.0520` | `72.58% / 0.0561` | **`76.22% / 0.0410`** | `75.01% / 0.0654` | **`parent_pair`** | **`+1.61 pp`** |

### Resource Profile Across Context Modes (RTX 4060, `max_length = 256`, `bf16`)
- `comment_only`: `~1.29–1.40 ms/comment`, `~3,984–3,999 MB` peak train VRAM
- `title_pair`: `~1.43–1.45 ms/comment` (`+11%`), `~4,014–4,024 MB` peak train VRAM
- `parent_pair`: `~2.17–2.38 ms/comment` (`+70%`), `~4,045–4,057 MB` peak train VRAM
- `title_parent_pair`: `~2.24–2.42 ms/comment` (`+75%`), `~4,046–4,061 MB` peak train VRAM

---

## 3. Priority-Slice & Mechanistic Error Analysis

### A. Where Context Delivered Genuine Gains
1. **Issue (`title_pair`, `+5.43 pp` overall OOF F1):**
   - **Short comments (`≤ 8` words, `n = 83`):** Macro F1 surged from **`24.79% → 41.97%` (`+17.18 pp`)** because terse replies like *"Same here"* or *"Wait till June"* inherit their domain referent explicitly from the post title.
   - **Rare/medium institutional domains jumped dramatically:** `hostel_facilities` **`0.4444 → 0.7037` (`+25.9 pp`)**, `campus_infrastructure` **`0.3509 → 0.4839` (`+13.3 pp`)**, `administrative_communication` **`0.7732 → 0.8197` (`+4.7 pp`)**, `financial_concerns` **`0.7516 → 0.7953` (`+4.4 pp`)**.
   - Calibration improved simultaneously (`ECE: 0.0498 → 0.0423`) with zero truncation (`0.00%`) and negligible latency cost (`+0.15 ms`).
2. **Solution (`parent_pair`, `+1.61 pp` overall OOF F1):**
   - **Short comments (`≤ 8` words):** Macro F1 jumped **`67.41% → 79.12%` (`+11.71 pp`)**!
   - **Comments with an immediate parent comment (`n = 205`):** Macro F1 rose **`74.47% → 79.23%` (`+4.76 pp`)**.
   - **False-positive reduction (direct hit on V5's core Solution flaw):** False positives dropped by **25%** (`36 → 27`), pushing positive-class `solution` precision from **`55.0% → 60.9%` (`+5.9 pp`)** while cutting ECE by **21%** (`0.0520 → 0.0410`). Seeing the parent comment lets the model distinguish a direct procedural answer to a stated problem from unsolicited general advice.

### B. Why Naive Full-Sequence Pooling Over Context Hurt Sentiment, Emotion & Response
Inspecting the exact per-class errors uncovered the precise architectural root cause:
1. **Context-Token Pooling Contamination in `[CLS] + Mean + Max`:** `PooledTransformerClassifier` averages and max-pools token states over **the entire attention mask (Segment A reply + Segment B context)**.
   - In **Emotion**, almost every Reddit post title is a worried or curious question (*"Waitlisted at 396, any chance?"*). Pooling Segment B tokens into `mean_vec` and `max_vec` caused **`curiosity` F1 to plunge `0.8673 → 0.7016` (`−16.6 pp`)** and **`stress` F1 to collapse `0.2564 → 0.0000`**, because the model could no longer separate the *poster's* curiosity/stress from the *commenter's* emotion.
   - In **Sentiment**, critical/anxious post titles inflated `negative ↔ neutral` mutual confusions from **`56` (`comment_only`) → `74` (`title_pair`) / `75` (`title_parent_pair`)**, dropping `negative` F1 from `0.7000 → 0.5938`.
   - In **Response**, parent context **did** show genuine stance signal — `parent_pair` cut **`counterargument → opinion` confusions nearly in half (`14 → 8`)**, and `title_pair` nudged `counterargument` F1 up (`0.2857 → 0.2963`) — **but** pooling over long parent comments degraded `agreement` (`0.4337 → 0.1795`) and `advice` (`0.6383 → 0.4600`), pulling overall Macro F1 down (`0.5292 → 0.4142`).
2. **Actionable Architectural Follow-Up (Tested in Phase 3):** In Phase 3 (Shared Multi-Task MuRIL), we directly test **Segment-A Reply-Masked Pooling** (allowing full cross-attention from reply tokens to context tokens in all 12 layers, while restricting Mean+Max pooling for `sentiment`, `emotion`, and `response` strictly to Segment-A reply tokens!) alongside pure `comment_only` MTL.

---

## 4. Phase 2 Per-Task Decisions Carried Into Phase 4

Selected strictly via 5-fold OOF CV (never test set):
- **`issue`:** `title_pair` (`+5.43 pp` OOF F1, `0%` truncation)
- **`sentiment`:** `comment_only` (avoids `-3.96 pp` context-affect bleed)
- **`emotion`:** `comment_only` (avoids `-11.74 pp` question/stress bleed)
- **`response`:** `comment_only` (avoids `-7.18 pp` global pooling contamination; Segment-A reply-masked context explored in Phase 3)
- **`solution`:** `parent_pair` (`+1.61 pp` OOF F1, `-25%` false positives, `+11.71 pp` on short comments)


---

## 5. Forensic Remediation Addendum — True Text-Aligned Parent Comments (`459 / 459`)

During the Phase 8 audit, a forensic check discovered that sequential pseudonym IDs (`post_XXXX`, `cmt_XXXXX`) in `data/live_corpus/corpus.json` drifted after `post_0064` relative to `master_labeled_v1.csv` (366 → 365 thread regeneration). While `comment_only` and `title_pair` came directly from `master_labeled_v1.csv` (100% unaffected), the initial Phase 2 `parent_pair` / `title_parent_pair` lookup had joined shifted `comment_id`s.

After fixing `common.py` to match all `459 / 459` rows via exact normalized `(post_title[:120], comment_text)` (**254 / 459 verified true parent comments**) and re-running on the 100% zero-leakage `muril_dapt_strict_inductive_wwm_8ep` (`ml/artifacts/v5_experiments/strict_clean_final/strict_clean_summary.json`):
- **Issue (`true parent_pair` + ASL):** OOF Macro F1 jumped from **`52.69%` (`title_pair`) → `56.00%` (`true parent_pair`, `+3.31 pp`!)**, and reached **`60.83%`** on the untouched frozen `n = 92` test set (`3-seed mean = 59.85% ± 0.99%`, paired bootstrap vs. V5: **`Δ = +15.00 pp`, 95% CI `[+2.83, +26.10]`, `p = 0.992`**)!
- **Sentiment, Emotion, Response & Solution:** Confirmed **`comment_only`** beats `true parent_pair` on all four tasks (`sentiment: 69.22% vs 61.71%`, `emotion: 57.24% vs 42.36%`, `response: 52.08% vs 39.49%` [XLM-R: `49.66% vs 39.63%`], `solution: 74.12% vs 72.07%` [XLM-R: `77.44% vs 72.06%`]). Even with 100% verified true parent comments, pooling Segment-B context tokens into global Mean+Max vectors still introduces affect/role bleed on speaker-level tasks, whereas **topic/issue identification genuinely thrives on true parent-comment context (`+3.31 pp` over `title_pair` alone!)**.
