# Deliverable 5a — Mathematical Foundations of Reddit Community Intelligence (RCI V6.2)

> **Document ID:** `RCI-V6.2-D5a` · **Navigation:** [← Deliverable 4 (Architecture)](./AI_ARCHITECTURE.md) · [Submission Index](./README.md) · [Deliverable 5b (Pseudocode) →](./SYSTEM_PSEUDOCODE.md)  
> Every equation below corresponds directly to an implemented module in this repository — no generic textbook abstractions unrelated to the codebase are included.

---

## 1. Sublinear TF-IDF Weighting (Lexical Baseline & Linker)
**Implementation:** `backend/app/services/nlp/topic_discovery/nmf_topic_discoverer.py`, `issue_solution_linking/tfidf_cosine_linker.py`, `ml/experiments/v6_programme/run_slang_emoji_5task_ablation.py`

For term $t$ in document $d$ across corpus $\mathcal{D}$ of size $N$:
$$\text{tf}_{\text{sub}}(t, d) = \begin{cases} 1 + \ln f_{t,d}, & f_{t,d} > 0 \\ 0, & f_{t,d} = 0 \end{cases}, \qquad \text{idf}(t) = \ln\!\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
$$x_{d,t} = \frac{\text{tf}_{\text{sub}}(t, d)\cdot \text{idf}(t)}{\left\|\mathbf{x}_d\right\|_2}$$

---

## 2. Non-Negative Matrix Factorization (Preserved NMF v1 Baseline)
**Implementation:** `backend/app/services/nlp/topic_discovery/nmf_topic_discoverer.py` (`n_components = 10`, `init = "nndsvda"`)

Given non-negative document-term matrix $V \in \mathbb{R}_{\ge 0}^{N \times M}$, NMF solves:
$$\min_{W \ge 0,\; H \ge 0} \frac{1}{2}\left\|V - W H\right\|_F^2$$
where $W \in \mathbb{R}_{\ge 0}^{N \times K}$ gives document-topic affinities and $H \in \mathbb{R}_{\ge 0}^{K \times M}$ gives topic-term weights. Normalized topic prevalence per document is $\tilde{w}_{d,k} = w_{d,k} / \sum_{j=1}^K w_{d,j}$ and dominant topic assignment is $\hat{z}_d = \arg\max_k w_{d,k}$.

---

## 3. Dense Multilingual Sentence Embeddings & Cosine Similarity
**Implementation:** `ml/topics/bertopic/embeddings.py` (`intfloat/multilingual-e5-small`, `384-dim`, MIT)

For prefixed text $u \in \{\text{"query: "}q,\; \text{"passage: "}p\}$ with last-layer token representations $\mathbf{h}_1, \dots, \mathbf{h}_L \in \mathbb{R}^d$ and attention mask $m_i \in \{0,1\}$:
$$\mathbf{e}(u) = \frac{\bar{\mathbf{h}}(u)}{\left\|\bar{\mathbf{h}}(u)\right\|_2}, \qquad \bar{\mathbf{h}}(u) = \frac{\sum_{i=1}^L m_i \mathbf{h}_i}{\max\!\left(\sum_{i=1}^L m_i,\, \epsilon\right)}$$
Because embeddings are unit L2-normalized, cosine similarity reduces to the inner product:
$$\text{sim}_{\cos}(q, p) = \mathbf{e}(q)^\top \mathbf{e}(p) \in [-1, 1]$$

---

## 4. BERTopic Pipeline: UMAP, HDBSCAN & BM25 Class-Based TF-IDF
**Implementation:** `ml/topics/bertopic/pipeline.py` (`bertopic==0.17.4`), `ml/topics/bertopic/evaluation.py`

### 4.1 UMAP Manifold Projection (`n_neighbors=15, n_components=5, metric="cosine"`)
Constructs a fuzzy topological simplicial set $P_{ij} = p_{j|i} + p_{i|j} - p_{j|i}p_{i|j}$ in $\mathbb{R}^{384}$ and optimizes low-dimensional coordinates $\mathbf{y}_i \in \mathbb{R}^5$ by minimizing fuzzy set cross-entropy:
$$\mathcal{L}_{\text{UMAP}} = \sum_{i \ne j}\left[ P_{ij}\ln\frac{P_{ij}}{Q_{ij}} + (1 - P_{ij})\ln\frac{1 - P_{ij}}{1 - Q_{ij}} \right], \qquad Q_{ij} = \left(1 + a\|\mathbf{y}_i - \mathbf{y}_j\|_2^{2b}\right)^{-1}$$

### 4.2 Density-Based HDBSCAN Clustering (`min_cluster_size=25, min_samples=8, method="eom"`)
Defines core distance $\text{core}_k(\mathbf{y}_i)$ (distance to $k$-th nearest neighbor) and mutual reachability distance:
$$d_{\text{mreach-}k}(\mathbf{y}_i, \mathbf{y}_j) = \max\!\left\{\text{core}_k(\mathbf{y}_i),\; \text{core}_k(\mathbf{y}_j),\; \|\mathbf{y}_i - \mathbf{y}_j\|_2\right\}$$
Builds a minimum spanning tree, condenses clusters with size $\ge m_{\text{clSize}}$, and maximizes excess-of-mass stability $S(C) = \sum_{\mathbf{y} \in C}(\lambda_{\mathbf{y}} - \lambda_{\text{birth}}(C))$, assigning non-dense points explicitly to outlier cluster $-1$.

### 4.3 BM25 Class-Based TF-IDF (c-TF-IDF)
Concatenating all documents assigned to cluster $c \in \{0, \dots, K-1\}$ into a single mega-document with term frequency $t_{w,c}$, frequency across all clusters $f_w = \sum_c t_{w,c}$, and average words per cluster $A = \frac{1}{K}\sum_c \sum_w t_{w,c}$:
$$W_{w,c} = \text{tf}_{w,c}^{\text{norm}} \cdot \ln\!\left(1 + \frac{A - f_w + 0.5}{f_w + 0.5}\right)$$
with sublinear `reduce_frequent_words` dampening $\sqrt{\text{tf}_{w,c}}$ so domain stopwords do not dominate topic representations.

### 4.4 Multi-Metric Topic Evaluation (`evaluation.py`)
- **NPMI Coherence:** $\text{NPMI}(w_i, w_j) = \frac{\ln \frac{P(w_i, w_j) + \epsilon}{P(w_i)P(w_j) + \epsilon}}{-\ln(P(w_i, w_j) + \epsilon)} \in [-1, 1]$ averaged over all $\binom{k}{2}$ pairs in each topic's top-$k$ keywords.
- **Topic Diversity:** $D = \frac{|\bigcup_{c=0}^{K-1} \text{TopK}(c)|}{K \cdot k} \in [0, 1]$.
- **Normalized Size Entropy Balance:** $H_{\text{norm}} = \frac{-\sum_{c=0}^{K-1} p_c \ln p_c}{\ln K} \in [0, 1]$.

---

## 5. Transformer Architecture & Multi-Head Self-Attention
**Implementation:** `ml/experiments/v5_optimization/` (`muril_dapt_strict_inductive_wwm_8ep` & `xlm-roberta-base`)

Each encoder layer applies scaled dot-product attention across $H = 12$ heads ($d_k = 768 / 12 = 64$):
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right)V, \qquad \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$$
For supervised classification, token states are aggregated via `[CLS] + Mean + Max` pooling (`3 × 768 = 2304-dim`) followed by LayerNorm and 5× Multi-Sample Dropout:
$$\mathbf{z} = \frac{1}{5}\sum_{m=1}^{5} W_{\text{head}}\,\text{Dropout}_m\!\left(\text{LayerNorm}\!\left([\mathbf{h}_{\text{CLS}} \,\|\, \mathbf{h}_{\text{mean}} \,\|\, \mathbf{h}_{\text{max}}]\right)\right) + \mathbf{b}$$

---

## 6. Classification Losses
**Implementation:** `ml/experiments/v5_optimization/` & `ml/experiments/v6_programme/run_slang_emoji_5task_ablation.py`

1. **Class-Weighted Cross-Entropy (`CE-W`, used for Sentiment, Response, Solution):**
   $$\mathcal{L}_{\text{CE-W}} = -w_y \ln \frac{\exp(z_y)}{\sum_{c=1}^C \exp(z_c)}, \qquad w_c = \frac{N}{C \cdot N_c}$$
2. **Asymmetric Loss (`ASL`, used for multi-label Issue & Emotion with $\gamma_+ = 0, \gamma_- = 2, m = 0.05$):**
   Let $p = \sigma(z)$ and shifted negative probability $p_m = \max(p - m, 0)$. For each label $j$:
   $$\mathcal{L}_{\text{ASL}}(z_j, y_j) = -y_j \ln(p_j) - (1 - y_j)\, p_{m,j}^{\gamma_-} \ln(1 - p_{m,j})$$
3. **Logit Adjustment (Response calibration ablation, $\tau = 0.75$):**
   $$z_c^{\text{adj}} = z_c + \tau \ln \pi_c, \qquad \pi_c = N_c / N$$
4. **R-Drop Bidirectional KL Regularization (Multi-Task ablation, $\alpha = 0.5$):**
   $$\mathcal{L}_{\text{R-Drop}} = \frac{1}{2}\big(\mathcal{L}_{\text{sup}}^{(1)} + \mathcal{L}_{\text{sup}}^{(2)}\big) + \frac{\alpha}{2}\Big(D_{\text{KL}}(P_1 \| P_2) + D_{\text{KL}}(P_2 \| P_1)\Big)$$

---

## 7. Probability Calibration (Matched to Inference Topology)
**Implementation:** `backend/build_live_corpus_artifacts.py`, `ml/experiments/v5_optimization/`

- **Scalar Temperature Scaling (Single-Checkpoint Single-Label):** Minimizes OOF NLL over $T \ge 1.0$:
  $$\hat{p}_c = \frac{\exp(z_c / T)}{\sum_{j=1}^C \exp(z_j / T)}$$
  For 5-fold logit ensembles $\bar{\mathbf{z}} = \frac{1}{5}\sum_{f=1}^5 \mathbf{z}^{(f)}$, nested OOF verification proved $T_{\text{ens}} = 1.0$ natively optimal.
- **Per-Label Monotonic OOF Platt Calibration (Multi-Label ASL Heads):** Corrects ASL's additive negative-logit shift via constrained logistic regression with $a_j > 0$:
  $$\hat{p}_j = \sigma(a_j \bar{z}_j + b_j)$$
- **5-Bin Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{b=1}^{5} \frac{|B_b|}{n}\left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

---

## 8. Contextual Issue–Solution Similarity & Clustering
**Implementation:** `ml/linking/contextual_mapper.py`

For candidate pair $(I, c)$ inside thread $P$ with parent context $\text{ctx}(c)$:
$$s_{\text{emb}} = \text{clip}\!\left(\frac{\mathbf{e}(q_{I,P,\text{ctx}})^\top \mathbf{e}(c) - 0.68}{0.24},\, 0,\, 1\right), \qquad s_{\text{lex}} = \min\!\big(1,\; 2.2\,\cos_{\text{TF-IDF}}(P \oplus \text{ctx},\, c)\big)$$
$$s_{\text{det}} = 0.60\, p_{\text{sol}}(c) + 0.40\, w_{\text{role}}(\hat{r}_c)$$
$$\text{matching\_score}(I, c) = \text{clip}\!\left(0.38\, s_{\text{emb}} + 0.14\, s_{\text{lex}} + 0.18\, s_{\text{struct}} + 0.30\, s_{\text{det}} - p_{\text{neg}}(c),\; 0,\; 1\right)$$
where $p_{\text{neg}}(c) \in [0, 0.55]$ penalizes negation, dismissals, counterarguments, DM/CFBR redirects, bare URLs, `[Removed]`, and ultra-short `<4`-word replies. Linked pairs ($\text{matching\_score} \ge 0.56 \land p_{\text{neg}} < 0.28$) are grouped per issue via average-linkage Agglomerative Clustering on cosine distance $d_{ab} = 1 - \mathbf{e}(c_a)^\top \mathbf{e}(c_b)$ at threshold $0.14$.

---

## 9. Explainable Unresolved-Issue Resolution Scoring
**Implementation:** `ml/linking/resolution_engine.py`

For thread $T$ with $n_T$ comments, linked solution set $\mathcal{S}_T$, best/mean solution relevance $(r_{\text{best}}, r_{\text{mean}})$, solution density $d_{\text{sol}} = |\mathcal{S}_T|/\max(n_T, 1)$, explicit reported-success indicator $I_{\text{succ}} \in \{0,1\}$, pending indicator $I_{\text{pend}} \in \{0,1\}$, follow-up confirmation count $c_{\text{conf}}$, and contradictory follow-up count $c_{\text{contr}}$:
$$p_{\text{contr}} = \min\!\big(0.45,\; 0.22\, c_{\text{contr}} + 0.12\, I_{\text{pend}}\big)$$
$$S_{\text{raw}}(T) = 0.42\, I_{\text{succ}} + 0.24\, r_{\text{best}} + 0.14\, \min(1,\, 2.5\, d_{\text{sol}}) + 0.12\, \min(1,\, c_{\text{conf}}/2) + 0.08\, r_{\text{mean}} - p_{\text{contr}}$$
$$S_{\text{resolution}}(T) = \begin{cases} \text{clip}(S_{\text{raw}}(T),\, 0,\, 1), & I_{\text{succ}} = 1 \\ \min(\text{clip}(S_{\text{raw}}(T),\, 0,\, 1),\; 0.58), & I_{\text{succ}} = 0 \end{cases}$$
$$\text{confidence}(T) = \text{clip}\!\left(0.30 + 0.40\,\min(1,\, n_T/6) + 0.30\,\min(1,\, |\mathcal{E}_T|/4),\; 0.15,\; 0.96\right)$$
Discussion volume $n_T$ enters **only** the confidence volume guard and can never inflate $S_{\text{resolution}}(T)$.
