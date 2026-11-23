/**
 * View 5: Topic Intelligence Studio (Dual-Method NMF v1 vs. BERTopic v1 Explorer)
 *
 * Strictly preserves NMF v1 as the default deterministic baseline while allowing
 * explicit switching to BERTopic v1 (intfloat/multilingual-e5-small) without ever
 * mixing NMF topic IDs with BERTopic topic IDs. Includes:
 * - URL-synced method switcher (?model=nmf|bertopic&tab=constellation|all|outliers|compare&topic=ID)
 * - Human-reviewed BERTopic cluster labels, coherence ratings (1–5), and discourse-type filters
 * - Interactive Area-Proportional Constellation Map (T0..T9 for NMF, BT-0..BT-13 for BERTopic)
 * - Explicit Outlier (-1) Document Inspector for BERTopic
 * - Side-by-side NMF vs. BERTopic multi-metric comparison table & one-click CSV export
 */
import { api } from "../api.js?v=6.2";
import { horizontalBarChart, topicConstellationSvg } from "../charts.js?v=6.2";
import { badge, callout, deck, emptyState, glossaryTip, tallyStrip } from "../components.js?v=6.2";
import { downloadCsv } from "../export.js?v=6.2";
import { esc, num, percent, pretty, truncate } from "../format.js?v=6.2";

const topicState = {
  model: "nmf",
  viewMode: "constellation", // constellation | all | outliers | compare
  selectedId: 0,
  search: "",
  filterKey: "all",
};

const CLUSTER_TYPE_META = {
  substantive: { label: "Substantive issue", tone: "positive" },
  admissions_logistics: { label: "Admissions logistics", tone: "info" },
  conversational_multilingual: { label: "Multilingual Hinglish", tone: "info" },
  conversational_phatic: { label: "Conversational / phatic", tone: "neutral" },
  media_url_artifact: { label: "URL / media artifact", tone: "warning" },
};

function detectKeywordScript(keywords = []) {
  const joined = keywords.join(" ");
  const hasDev = /[\u0900-\u097F]/.test(joined);
  const hasLatin = /[A-Za-z]/.test(joined);
  if (hasDev && !hasLatin) return "devanagari";
  if (hasDev && hasLatin) return "mixed";
  return "latin";
}

function classifyClusterCharacter(topic, method) {
  const script = detectKeywordScript(topic.keywords || []);
  const kws = (topic.keywords || []).map((k) => k.toLowerCase());
  const algName = method === "bertopic" ? "BERTopic (c-TF-IDF)" : "NMF (TF-IDF)";

  if (method === "bertopic" && topic.cluster_type && CLUSTER_TYPE_META[topic.cluster_type]) {
    const meta = CLUSTER_TYPE_META[topic.cluster_type];
    return {
      script,
      group: topic.cluster_type.startsWith("conversational_") ? "conversational" : topic.cluster_type,
      tag: meta.label,
      tone: meta.tone,
      note: topic.human_review_note || `Dense multilingual E5 + UMAP + HDBSCAN cluster (${algName}).`,
    };
  }

  if (script === "devanagari") {
    return {
      script: "devanagari",
      group: "devanagari",
      tag: "Devanagari tokens",
      tone: "warning",
      note: `Devanagari-script terms extracted by unsupervised ${algName}.`,
    };
  }
  const hasHinglish = kws.some((k) =>
    ["hai", "bhi", "bhai", "kya", "nhi", "nahi", "hoga", "aap", "ek", "mein", "baat", "kar", "chahiye", "toh", "aur", "unhone", "bola"].includes(k)
  );
  if (hasHinglish) {
    return {
      script: "latin",
      group: "latin",
      tag: "Hinglish / code-mixed cluster",
      tone: "info",
      note: `Romanized Hindi-English phrasing discovered by ${algName}.`,
    };
  }
  const hasStopwords = kws.filter((k) => ["the", "that", "for", "would", "my", "i", "it", "is"].includes(k)).length >= 2;
  if (hasStopwords) {
    return {
      script: "latin",
      group: "latin",
      tag: "High-frequency discourse cluster",
      tone: "neutral",
      note: `High-frequency conversational terms retained by ${algName}.`,
    };
  }
  return {
    script: "latin",
    group: "latin",
    tag: method === "bertopic" ? "Semantic neural cluster" : "Thematic NMF cluster",
    tone: method === "bertopic" ? "positive" : "neutral",
    note: method === "bertopic"
      ? "Dense multilingual E5 + UMAP + HDBSCAN topic."
      : "Unsupervised NMF topic derived directly from anonymized community comments.",
  };
}

function renderTopicCard(topic, totalDocs, method) {
  const kws = topic.keywords || [];
  const weights = topic.keyword_weights || [];
  const maxW = Math.max(0.001, ...weights.map((w) => Number(w || 0)));
  const idPrefix = method === "bertopic" ? `BERTopic BT-${topic.topic_id}` : `NMF Topic ${topic.topic_id}`;
  const termHeading = method === "bertopic" ? "Top c-TF-IDF terms (BM25-weighted)" : "Top weighted NMF terms";

  const kwBars = kws.map((kw, idx) => {
    const w = Number(weights[idx] ?? 0);
    const pct = Math.max(4, (w / maxW) * 100);
    return `
      <div class="kw-weight-row">
        <code class="kw-term">${esc(kw)}</code>
        <div class="kw-track"><span class="kw-fill" style="width:${pct.toFixed(1)}%"></span></div>
        <span class="num kw-val">${w ? w.toFixed(3) : "—"}</span>
      </div>
    `;
  }).join("");

  const relatedIssues = (topic.related_supervised_issues || []).map((ri) => `
    <a class="tag tag-neutral" href="#/issues?selected=${encodeURIComponent(ri.issue)}" title="Observational overlap with supervised issue domain">
      ${esc(pretty(ri.issue))} · ${Math.round(Number(ri.share || 0) * 100)}%
    </a>
  `).join("");

  const examples = (topic.representative_examples || []).slice(0, 3).map((ex) => `
    <div class="topic-example-item">
      <p>“${esc(truncate(ex.text, 220))}”</p>
      <div class="topic-example-foot">
        <a href="#/thread/${encodeURIComponent(ex.post_id)}">${esc(ex.post_title || ex.post_id)}</a>
        <span class="num muted">${ex.language ? `${esc(ex.language)} · ` : ""}${method === "bertopic" ? "Cos" : "Score"} ${ex.score !== undefined && ex.score !== null ? Number(ex.score).toFixed(3) : "—"}</span>
      </div>
    </div>
  `).join("");

  const reviewBadge = topic.label_status === "human_reviewed"
    ? badge(`Human-reviewed${topic.human_coherence_rating ? ` · ${topic.human_coherence_rating}/5` : ""}`, "positive")
    : (topic.label_status === "machine_suggested_pending_human_review" ? badge("Machine-suggested", "neutral") : "");

  return `
    <article class="topic-card">
      <header class="topic-card-head">
        <div>
          <div class="tag-row tag-row-mb">
            <span class="tag ${method === "bertopic" ? "tag-positive" : "tag-info"} num">${esc(idPrefix)}</span>
            ${badge(topic.character.tag, topic.character.tone)}
            ${reviewBadge}
          </div>
          ${topic.label && method === "bertopic" ? `<h3 class="text-sm">${esc(topic.label)}</h3>` : ""}
          <p class="muted text-xs">${esc(topic.character.note)}</p>
          ${topic.machine_suggested_label && topic.machine_suggested_label !== topic.label
            ? `<p class="muted text-xs">Original c-TF-IDF label: <code>${esc(topic.machine_suggested_label)}</code></p>`
            : ""}
        </div>
        <div class="topic-card-stat">
          <strong class="num">${num(topic.document_count)}</strong>
          <span class="num">${percent(topic.document_ratio ?? (topic.document_count / Math.max(totalDocs, 1)))}</span>
        </div>
      </header>
      <div>
        <h4 class="subhead-xs">${termHeading}</h4>
        <div class="kw-weight-list">${kwBars}</div>
      </div>
      ${relatedIssues ? `
        <div>
          <h4 class="subhead-xs">Related supervised issues (observational overlap only)</h4>
          <div class="tag-row">${relatedIssues}</div>
        </div>
      ` : ""}
      <div>
        <h4 class="subhead-xs">Representative assigned discussions</h4>
        <div class="preview-comment-list">${examples || `<p class="muted">No examples returned.</p>`}</div>
      </div>
    </article>
  `;
}

function renderOutliersPanel(outliers, totalDocs) {
  if (!outliers || !outliers.count) {
    return callout(
      "NMF assigns every document to one of its 10 dense linear factors (0 explicit outliers). Switch to BERTopic v1 to inspect density-based -1 outlier documents.",
      { tone: "neutral", title: "No explicit outlier bin in NMF:" }
    );
  }
  const langPills = Object.entries(outliers.language_distribution || {})
    .map(([lg, cnt]) => `<span class="tag tag-neutral num">${esc(lg)}: ${num(cnt)}</span>`)
    .join("");

  const sampleCards = (outliers.sample_documents || []).slice(0, 20).map((doc) => `
    <div class="topic-example-item">
      <div class="tag-row">
        <span class="tag tag-warning num">Outlier (-1)</span>
        <span class="tag tag-neutral">${esc(doc.language || "und")} · ${esc(doc.script || "latin")}</span>
        ${doc.issue_category ? `<span class="tag tag-info">${esc(pretty(doc.issue_category))}</span>` : ""}
        <span class="muted num text-xs">${num(doc.word_count)} words</span>
      </div>
      <p>“${esc(truncate(doc.text, 260))}”</p>
      <div class="topic-example-foot">
        <a href="#/thread/${encodeURIComponent(doc.post_id)}">${esc(doc.post_title || doc.post_id)}</a>
        <span class="muted num">${esc(doc.comment_id || "")}</span>
      </div>
    </div>
  `).join("");

  return `
    ${callout(
      `BERTopic explicitly preserves <strong>${num(outliers.count)} outlier comments (${percent(outliers.ratio)} of ${num(totalDocs)} docs)</strong> in topic <code>-1</code> rather than forcing conversational tangents, short banter, or multi-issue narratives into noisy clusters. Median length: <strong>${outliers.median_word_count} words</strong> (${percent(outliers.short_comment_le_5_words_share)} &le; 5 words).`,
      { tone: "warning", title: "Explicit Outlier Preservation (Topic -1):" }
    )}
    <div class="tag-row tag-row-md">${langPills}</div>
    <div class="preview-comment-list">${sampleCards}</div>
  `;
}

function renderComparisonPanel(comparison) {
  const rows = comparison?.table || [];
  if (!rows.length) {
    return `<p class="muted">Comparison benchmark data is not available.</p>`;
  }
  const tbody = rows.map((r) => `
    <tr class="${r.is_winner_bertopic || r.is_preserved_default ? "table-row-highlight" : ""}">
      <td>
        <strong>${esc(r.model_id)}</strong>
        ${r.is_preserved_default ? ` ${badge("Default", "info")}` : ""}
        ${r.is_winner_bertopic ? ` ${badge("Promoted Companion", "positive")}` : ""}
      </td>
      <td>${esc(r.algorithm?.toUpperCase())}</td>
      <td class="text-xs">${esc(r.embedding_model || "—")}</td>
      <td><code>${esc(r.preprocessing_mode || "—")}</code></td>
      <td class="num">${num(r.n_topics)}</td>
      <td class="num">${num(r.outlier_count)} (${(Number(r.outlier_rate || 0) * 100).toFixed(1)}%)</td>
      <td class="num">${r.coherence_npmi !== undefined ? Number(r.coherence_npmi).toFixed(4) : "—"}</td>
      <td class="num">${r.coherence_umass !== undefined ? Number(r.coherence_umass).toFixed(3) : "—"}</td>
      <td class="num">${r.topic_diversity !== undefined ? Number(r.topic_diversity).toFixed(4) : "—"}</td>
      <td class="num">${r.representative_centroid_cosine !== undefined && r.representative_centroid_cosine !== null ? Number(r.representative_centroid_cosine).toFixed(4) : "—"}</td>
      <td class="num">${r.size_balance_entropy !== undefined ? Number(r.size_balance_entropy).toFixed(4) : "—"}</td>
    </tr>
  `).join("");

  const probes = comparison?.multilingual_embedding_probes_synthetic?.results || {};
  const probeRows = Object.entries(probes).map(([k, v]) => `
    <tr>
      <td><strong>${esc(k)}</strong></td>
      <td class="text-xs">${esc(v.license || "—")} · ${num(v.dim)}d</td>
      <td class="num"><strong>${(Number(v.top1_accuracy || 0) * 100).toFixed(1)}%</strong></td>
      <td class="num">${Number(v.mean_similarity_margin || 0) >= 0 ? "+" : ""}${Number(v.mean_similarity_margin || 0).toFixed(4)}</td>
      <td class="text-xs">${Object.entries(v.by_language_group || {}).map(([lg, sc]) => `${lg}: ${Math.round(sc * 100)}%`).join(" · ")}</td>
    </tr>
  `).join("");

  return `
    ${callout(
      "Both NMF v1 and all five BERTopic configurations were evaluated on the <strong>exact same 3,459 eligible corpus comments</strong>. No single metric picks a winner: NMF remains the deterministic 100%-coverage default baseline, while BERTopic v1 (<code>intfloat/multilingual-e5-small</code>) surfaces 35 granular semantic clusters (education loans/CIBIL, refund deadlines, B-school comparisons, internships) with superior size balance (0.951 vs. 0.809) and 4.51/5 mean human coherence.",
      { tone: "info", title: "Fair Side-by-Side Comparison (Identical 3,459 Documents):" }
    )}
    <div class="table-scroll">
      <table class="data-table" aria-label="NMF versus BERTopic multi-metric comparison">
        <thead>
          <tr>
            <th scope="col">Model ID</th>
            <th scope="col">Alg</th>
            <th scope="col">Embedding / Vectorizer</th>
            <th scope="col">Preproc.</th>
            <th scope="col" class="num">Topics</th>
            <th scope="col" class="num">Outliers (-1)</th>
            <th scope="col" class="num">NPMI</th>
            <th scope="col" class="num">UMass</th>
            <th scope="col" class="num">Diversity</th>
            <th scope="col" class="num">Centroid Cos</th>
            <th scope="col" class="num">Balance</th>
          </tr>
        </thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>
    ${probeRows ? `
      <div class="spacer-md"></div>
      <h3 class="subhead-xs">Controlled Synthetic Multilingual Embedding Probe (n = 14 — strictly separate from real corpus)</h3>
      <p class="muted text-xs" style="margin-bottom:var(--space-sm)">Tests whether student-issue queries in Hinglish, Hindi Devanagari, Bengali script, Banglish, Odia script, and Roman-Odia align to the matching English anchor. Notice MuRIL-DAPT mean-pooling suffers anisotropic cone collapse (+0.0017 margin &rarr; 2 degenerate clusters in HDBSCAN).</p>
      <div class="table-scroll">
        <table class="data-table" aria-label="Synthetic multilingual embedding alignment probe results">
          <thead>
            <tr>
              <th scope="col">Encoder</th>
              <th scope="col">License / Dim</th>
              <th scope="col" class="num">Top-1 Acc</th>
              <th scope="col" class="num">Cos Margin</th>
              <th scope="col">Per-Language Breakdown</th>
            </tr>
          </thead>
          <tbody>${probeRows}</tbody>
        </table>
      </div>
    ` : ""}
  `;
}

export async function renderTopics(container, ctx, params = {}) {
  if (params.model === "bertopic" || params.model === "nmf") {
    topicState.model = params.model;
  }
  if (params.tab && ["constellation", "all", "outliers", "compare"].includes(params.tab)) {
    topicState.viewMode = params.tab;
  }
  if (params.topic !== undefined && Number.isFinite(Number(params.topic))) {
    topicState.selectedId = Number(params.topic);
  }
  if (topicState.model === "nmf" && topicState.viewMode === "outliers") {
    topicState.viewMode = "constellation";
  }

  async function loadAndPaint() {
    const [topicRes, summary] = await Promise.all([
      api.topics(topicState.model),
      ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
    ]);
    ctx.summary = summary;

    const method = topicRes.model || topicState.model || "nmf";
    const modelInfo = topicRes.model_info || {};
    const availableModels = topicRes.available_models || [];
    const outliers = topicRes.outliers || null;
    const comparison = topicRes.comparison || {};
    const reviewSummary = modelInfo.human_review_summary || null;

    const rawTopics = (topicRes.items || []).map((t) => ({
      ...t,
      character: classifyClusterCharacter(t, method),
    }));
    const assignedDocs = rawTopics.reduce((s, t) => s + Number(t.document_count || 0), 0);
    const totalDocs = Number(modelInfo.n_documents || assignedDocs || summary.comments || 0);
    const outlierCount = Number(modelInfo.outlier_count ?? outliers?.count ?? 0);
    const largestTopic = [...rawTopics].sort((a, b) => Number(b.document_count) - Number(a.document_count))[0];
    const devTopics = rawTopics.filter((t) => t.character.script === "devanagari");
    const hinglishTopics = rawTopics.filter((t) => t.character.tone === "info");
    const hinglishComments = hinglishTopics.reduce((s, t) => s + Number(t.document_count || 0), 0);

    if (!rawTopics.some((t) => Number(t.topic_id) === Number(topicState.selectedId)) && largestTopic) {
      topicState.selectedId = largestTopic.topic_id;
    }

    function buildFilterPills() {
      if (method === "bertopic") {
        const counts = {
          substantive: rawTopics.filter((t) => t.character.group === "substantive").length,
          admissions_logistics: rawTopics.filter((t) => t.character.group === "admissions_logistics").length,
          conversational: rawTopics.filter((t) => t.character.group === "conversational").length,
          media_url_artifact: rawTopics.filter((t) => t.character.group === "media_url_artifact").length,
        };
        const defs = [
          ["all", `All (${rawTopics.length})`],
          ["substantive", `Substantive (${counts.substantive})`],
          ["admissions_logistics", `Logistics (${counts.admissions_logistics})`],
          ["conversational", `Conversational (${counts.conversational})`],
          ["media_url_artifact", `URL / Media (${counts.media_url_artifact})`],
        ].filter(([k]) => k === "all" || counts[k] > 0);
        return defs.map(([k, lbl]) => `
          <button type="button" class="pill-btn ${topicState.filterKey === k ? "is-active" : ""}" data-topic-filter="${k}">${lbl}</button>
        `).join("");
      }
      if (!devTopics.length) return "";
      return `
        <button type="button" class="pill-btn ${topicState.filterKey === "all" ? "is-active" : ""}" data-topic-filter="all">All (${rawTopics.length})</button>
        <button type="button" class="pill-btn ${topicState.filterKey === "latin" ? "is-active" : ""}" data-topic-filter="latin">Latin / Hinglish (${rawTopics.length - devTopics.length})</button>
        <button type="button" class="pill-btn ${topicState.filterKey === "devanagari" ? "is-active" : ""}" data-topic-filter="devanagari">Devanagari (${devTopics.length})</button>
      `;
    }

    function paint() {
      const q = topicState.search.trim().toLowerCase();
      const filtered = rawTopics.filter((t) => {
        if (topicState.filterKey !== "all") {
          const matchVal = method === "bertopic" ? t.character.group : t.character.script;
          if (matchVal !== topicState.filterKey) return false;
        }
        if (!q) return true;
        const hay = [
          t.label,
          t.machine_suggested_label || "",
          `topic ${t.topic_id}`,
          `bt-${t.topic_id}`,
          ...(t.keywords || []),
          ...(t.representative_examples || []).map((e) => `${e.text} ${e.post_title}`),
        ].join(" ").toLowerCase();
        return hay.includes(q);
      });

      const activeTopic = rawTopics.find((t) => Number(t.topic_id) === Number(topicState.selectedId)) || rawTopics[0];
      const largestPct = largestTopic && totalDocs ? (Number(largestTopic.document_count) / totalDocs) * 100 : 0;
      const outlierPct = totalDocs ? (outlierCount / totalDocs) * 100 : 0;
      const hinglishPct = totalDocs ? (hinglishComments / totalDocs) * 100 : 0;

      const modelSwitcher = `
        <div class="method-bar">
          <div class="tag-row">
            <span class="method-bar-label">Topic model</span>
            <div class="segmented" role="group" aria-label="Select topic discovery method">
              ${availableModels.map((m) => `
                <button type="button" class="seg-btn ${method === m.id ? "is-active" : ""}" data-switch-topic-model="${esc(m.id)}">
                  ${esc(m.label)} (${num(m.n_topics)})
                </button>
              `).join("")}
            </div>
          </div>
          <div class="tag-row">
            ${badge(method === "bertopic" ? "Active: BERTopic v1" : "Active: NMF v1 (Default)", method === "bertopic" ? "positive" : "info")}
            <span class="tag tag-neutral text-xs">Encoder: ${esc(modelInfo.embedding_model || "TF-IDF")}</span>
            <button type="button" class="btn btn-quiet btn-sm" id="export-topics-csv">Export CSV</button>
          </div>
        </div>
      `;

      const tallyRow = tallyStrip([
        {
          label: method === "bertopic" ? "Discovered BERTopic clusters" : "Unsupervised NMF topics",
          valueText: num(rawTopics.length),
          countTo: rawTopics.length,
          unit: method === "bertopic" ? "natural clusters" : "fixed factors",
          accent: true,
          note: method === "bertopic"
            ? (reviewSummary
              ? `Human-reviewed (${reviewSummary.mean_coherence_rating_1_to_5}/5 mean coherence) · ${reviewSummary.cluster_type_breakdown?.substantive || 0} substantive`
              : `HDBSCAN min_cluster_size=${modelInfo.hdbscan?.min_cluster_size || 25}, BM25 c-TF-IDF`)
            : "TF-IDF (1–2 grams, AutoMod boilerplate filtered), 10 weighted terms per factor",
        },
        {
          label: "Assigned comments",
          valueText: num(assignedDocs),
          countTo: assignedDocs,
          note: `Out of ${num(totalDocs)} eligible anonymized community comments`,
        },
        {
          label: method === "bertopic" ? "Explicit outliers (-1)" : "Largest cluster share",
          valueText: method === "bertopic" ? `${outlierPct.toFixed(1)}%` : `${largestPct.toFixed(1)}%`,
          countTo: method === "bertopic" ? outlierPct : largestPct,
          decimals: 1,
          suffix: "%",
          ratioPct: method === "bertopic" ? outlierPct : largestPct,
          color: method === "bertopic" ? "var(--color-warning)" : "var(--color-role-advice)",
          note: method === "bertopic"
            ? `${num(outlierCount)} preserved in topic -1 (never silently forced into clusters)`
            : (largestTopic ? `Topic ${largestTopic.topic_id} (${num(largestTopic.document_count)} comments)` : ""),
        },
        {
          label: method === "bertopic" ? "Topic diversity / balance" : "Hinglish / code-mixed",
          valueText: method === "bertopic"
            ? `${(Number(modelInfo.metrics?.topic_diversity || 0) * 100).toFixed(1)}%`
            : num(hinglishTopics.length),
          countTo: method === "bertopic" ? Number(modelInfo.metrics?.topic_diversity || 0) * 100 : hinglishTopics.length,
          decimals: method === "bertopic" ? 1 : 0,
          suffix: method === "bertopic" ? "%" : "",
          unit: method === "bertopic" ? "" : "topic",
          ratioPct: method === "bertopic" ? Number(modelInfo.metrics?.size_distribution?.normalized_entropy_balance || 0) * 100 : hinglishPct,
          color: "var(--color-accent-2)",
          note: method === "bertopic"
            ? `Entropy balance ${Number(modelInfo.metrics?.size_distribution?.normalized_entropy_balance || 0).toFixed(3)} · 80% subsample NMI ${Number(modelInfo.stability?.subsample_80pct_nmi_all_docs || 0).toFixed(3)}`
            : `${num(hinglishComments)} comments (${hinglishPct.toFixed(1)}%) with Romanized Hindi terms`,
        },
      ], "Topic discovery summary");

      const pillsMarkup = buildFilterPills();
      const modeControls = `
        <div class="view-toggle-bar">
          ${pillsMarkup ? `<div class="filter-pills" role="group" aria-label="Filter topic clusters">${pillsMarkup}</div>` : `<div></div>`}
          <div class="segmented" role="group" aria-label="Topic visualization layout">
            <button type="button" class="seg-btn ${topicState.viewMode === "constellation" ? "is-active" : ""}" data-topic-mode="constellation">Proportional map</button>
            <button type="button" class="seg-btn ${topicState.viewMode === "all" ? "is-active" : ""}" data-topic-mode="all">All ${rawTopics.length} cards</button>
            ${method === "bertopic" ? `<button type="button" class="seg-btn ${topicState.viewMode === "outliers" ? "is-active" : ""}" data-topic-mode="outliers">Outliers -1 (${num(outlierCount)})</button>` : ""}
            <button type="button" class="seg-btn ${topicState.viewMode === "compare" ? "is-active" : ""}" data-topic-mode="compare">NMF vs. BERTopic</button>
          </div>
        </div>
      `;

      const mapTopics = (filtered.length ? filtered : rawTopics).slice(0, 14);
      const activeForMap = mapTopics.some((t) => Number(t.topic_id) === Number(activeTopic?.topic_id))
        ? activeTopic
        : mapTopics[0];

      const constellationSection = activeForMap
        ? `
          ${rawTopics.length > 14 ? `<p class="muted text-xs" style="margin-bottom:var(--space-xs)">Showing top ${mapTopics.length} matching clusters in the proportional map. Switch to <strong>All ${rawTopics.length} cards</strong> to search every discovered topic.</p>` : ""}
          <div class="constellation-layout">
            <div>
              ${topicConstellationSvg(mapTopics, activeForMap.topic_id, { method })}
              <div class="spacer-sm"></div>
              <div class="chart-legend">
                <span class="legend-item"><span class="legend-swatch" style="background:${method === "bertopic" ? "var(--color-positive)" : "var(--color-accent)"}"></span><span>${method === "bertopic" ? "Substantive issue cluster" : "Thematic English cluster"}</span></span>
                <span class="legend-item"><span class="legend-swatch" style="background:var(--color-role-advice)"></span><span>${method === "bertopic" ? "Admissions logistics / Multilingual" : "Hinglish / code-mixed cluster"}</span></span>
              </div>
            </div>
            <div>
              ${renderTopicCard(activeForMap, totalDocs, method)}
            </div>
          </div>
        `
        : "";

      const allCardsHtml = filtered.length
        ? `<div class="topic-card-grid">${filtered.map((t) => renderTopicCard(t, totalDocs, method)).join("")}</div>`
        : emptyState({
            title: `No ${method.toUpperCase()} topics match your filter`,
            description: "Clear the keyword search or switch back to all clusters.",
            actionHtml: `<button type="button" class="btn btn-quiet btn-sm" data-clear-topics>Reset filters</button>`,
          });

      const distRows = [...filtered]
        .sort((a, b) => Number(b.document_count) - Number(a.document_count))
        .slice(0, 18)
        .map((t) => ({
          label: method === "bertopic"
            ? truncate(t.label || `BT-${t.topic_id}`, 42)
            : `Topic ${t.topic_id}: ${(t.keywords || []).slice(0, 3).join(" · ")}`,
          sublabel: t.character.tag,
          count: Number(t.document_count || 0),
          color: t.character.script === "devanagari"
            ? "var(--color-accent-2)"
            : t.character.tone === "info"
              ? "var(--color-role-advice)"
              : t.character.tone === "positive"
                ? "var(--color-positive)"
                : "var(--color-accent)",
        }));

      let mainPanel = constellationSection;
      if (topicState.viewMode === "all") {
        mainPanel = `
          <div class="field" style="max-width:440px;margin-bottom:var(--space-md)">
            <label for="topic-search">Search ${method.toUpperCase()} labels, keywords, or representative comments</label>
            <input id="topic-search" type="search" value="${esc(topicState.search)}" placeholder="Search keywords (loan, cibil, refund, waitlist, dsai, placement, bhai)…">
          </div>
          ${allCardsHtml}
        `;
      } else if (topicState.viewMode === "outliers") {
        mainPanel = renderOutliersPanel(outliers, totalDocs);
      } else if (topicState.viewMode === "compare") {
        mainPanel = renderComparisonPanel(comparison);
      }

      container.innerHTML = `
        ${modelSwitcher}
        ${callout(
          method === "bertopic"
            ? `Inspecting <strong>BERTopic v1</strong> (<code>${esc(modelInfo.embedding_model || "intfloat/multilingual-e5-small")}</code>, license: ${esc(modelInfo.embedding_license || "MIT")}, preprocessing: <code>${esc(modelInfo.preprocessing_mode || "raw_clean")}</code>). All 35 cluster labels have been human-reviewed (mean coherence <strong>${reviewSummary?.mean_coherence_rating_1_to_5 || "4.51"}/5</strong>) while preserving original c-TF-IDF machine labels and never mixing BERTopic IDs with NMF IDs.`
            : `Inspecting <strong>NMF v1 (Baseline Default)</strong> ${glossaryTip("nmf_topics", "NMF topic clusters")} fit on anonymized public Reddit comments (unigrams &amp; bigrams, AutoMod boilerplate removed). Switch to <strong>BERTopic v1</strong> above to inspect dense multilingual clusters, human-reviewed themes, and explicit outliers.`,
          { tone: "info", title: method === "bertopic" ? "BERTopic v1 Semantic Companion:" : "Preserved NMF v1 Baseline:" }
        )}
        ${tallyRow}
        <section class="anim-enter" aria-label="Topic constellation and weighted terms">
          ${modeControls}
          <div class="spacer-md"></div>
          ${mainPanel}
        </section>
        ${topicState.viewMode !== "compare" && distRows.length ? deck(
          `Corpus prevalence across ${filtered.length > 18 ? "top 18 of " : ""}${num(filtered.length)} ${method.toUpperCase()} topics`,
          {
            subtitle: `Assigned comments per cluster (n = ${num(assignedDocs)} assigned${outlierCount ? `, ${num(outlierCount)} preserved in outlier bin -1` : ""})`,
            body: horizontalBarChart(distRows, { total: totalDocs, unit: "comments", showShare: true }),
          }
        ) : ""}
      `;

      container.querySelector("#export-topics-csv")?.addEventListener("click", () => {
        const ok = downloadCsv(
          rawTopics,
          [
            { key: "namespaced_topic_id", label: "Topic ID" },
            { key: "label", label: "Label" },
            { key: "label_status", label: "Label Status" },
            { key: "cluster_type", label: "Cluster Type", get: (r) => r.cluster_type || r.character?.tag || "" },
            { key: "human_coherence_rating", label: "Human Coherence (1-5)" },
            { key: "document_count", label: "Documents" },
            { key: "keywords", label: "Top Keywords", get: (r) => (r.keywords || []).join(", ") },
          ],
          `rci_${method}_topics.csv`
        );
        if (ok) ctx.showToast?.(`Exported ${rawTopics.length} ${method.toUpperCase()} topics to CSV`);
      });

      container.querySelectorAll("[data-switch-topic-model]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const nextModel = btn.dataset.switchTopicModel;
          if (nextModel && nextModel !== topicState.model) {
            topicState.model = nextModel;
            topicState.filterKey = "all";
            if (nextModel === "nmf" && topicState.viewMode === "outliers") {
              topicState.viewMode = "constellation";
            }
            topicState.selectedId = 0;
            ctx.syncQueryParams?.({ model: nextModel, tab: topicState.viewMode, topic: null });
            loadAndPaint();
          }
        });
      });

      container.querySelectorAll("[data-select-topic]").forEach((orb) => {
        const select = () => {
          topicState.selectedId = Number(orb.dataset.selectTopic);
          ctx.syncQueryParams?.({ model: topicState.model, tab: topicState.viewMode, topic: topicState.selectedId });
          paint();
        };
        orb.addEventListener("click", select);
        orb.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); }
        });
      });

      container.querySelectorAll("[data-topic-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
          topicState.viewMode = btn.dataset.topicMode;
          ctx.syncQueryParams?.({ model: topicState.model, tab: topicState.viewMode });
          paint();
        });
      });
      container.querySelectorAll("[data-topic-filter]").forEach((btn) => {
        btn.addEventListener("click", () => { topicState.filterKey = btn.dataset.topicFilter; paint(); });
      });

      const input = container.querySelector("#topic-search");
      if (input) {
        input.addEventListener("input", (e) => {
          topicState.search = e.target.value;
          paint();
          const next = container.querySelector("#topic-search");
          if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
        });
      }
      container.querySelector("[data-clear-topics]")?.addEventListener("click", () => {
        topicState.search = "";
        topicState.filterKey = "all";
        paint();
      });
    }

    paint();
  }

  await loadAndPaint();
}
