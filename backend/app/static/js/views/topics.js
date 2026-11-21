/**
 * View 5: Topic Constellation Studio (Interactive 8-Orb NMF Map + Live Term Inspector)
 * Showcases unsupervised NMF clusters, keyword_weights, corpus share ratios,
 * representative excerpts, and honest disclosure of Devanagari regex tokenization.
 */
import { api } from "../api.js";
import { horizontalBarChart, topicConstellationSvg } from "../charts.js";
import { badge, callout, deck, emptyState, glossaryTip, tallyStrip } from "../components.js";
import { esc, num, percent, share, truncate } from "../format.js";

const topicState = {
  viewMode: "constellation",
  selectedId: 1,
  search: "",
  scriptFilter: "all",
};

function detectKeywordScript(keywords = []) {
  const joined = keywords.join(" ");
  const hasDev = /[\u0900-\u097F]/.test(joined);
  const hasLatin = /[A-Za-z]/.test(joined);
  if (hasDev && !hasLatin) return "devanagari";
  if (hasDev && hasLatin) return "mixed";
  return "latin";
}

function classifyClusterCharacter(topic) {
  const script = detectKeywordScript(topic.keywords || []);
  const kws = (topic.keywords || []).map((k) => k.toLowerCase());
  if (script === "devanagari") {
    return {
      script: "devanagari",
      tag: "Devanagari tokens",
      tone: "warning",
      note: "Devanagari-script terms extracted by unsupervised NMF.",
    };
  }
  const hasHinglish = kws.some((k) => ["hai", "bhi", "bhai", "kya", "nhi", "nahi", "hoga", "aap", "ek", "mein", "baat", "kar", "kar lo", "chahiye", "toh", "aur"].includes(k));
  if (hasHinglish) {
    return {
      script: "latin",
      tag: "Hinglish & code-mixed cluster",
      tone: "info",
      note: "Romanized Hindi-English phrasing and conversational community terms.",
    };
  }
  const hasStopwords = kws.filter((k) => ["the", "that", "for", "would", "my", "i", "it", "is"].includes(k)).length >= 2;
  if (hasStopwords) {
    return {
      script: "latin",
      tag: "High-frequency discourse cluster",
      tone: "neutral",
      note: "High-frequency conversational terms retained across threads.",
    };
  }
  return {
    script: "latin",
    tag: "Thematic discussion cluster",
    tone: "neutral",
    note: "Unsupervised NMF topic derived directly from anonymized community comments.",
  };
}

function renderTopicCard(topic, totalDocs) {
  const kws = topic.keywords || [];
  const weights = topic.keyword_weights || [];
  const maxW = Math.max(0.001, ...weights.map((w) => Number(w || 0)));

  const kwBars = kws.map((kw, idx) => {
    const w = Number(weights[idx] ?? 0);
    const pct = Math.max(4, (w / maxW) * 100);
    return `
      <div class="kw-weight-row">
        <code class="kw-term">${esc(kw)}</code>
        <div class="kw-track"><span class="kw-fill" style="width:${pct.toFixed(1)}%"></span></div>
        <span class="num kw-val">${w ? w.toFixed(3) : "None"}</span>
      </div>
    `;
  }).join("");

  const examples = (topic.representative_examples || []).map((ex) => `
    <div class="topic-example-item">
      <p>“${esc(truncate(ex.text, 220))}”</p>
      <div class="topic-example-foot">
        <a href="#/thread/${encodeURIComponent(ex.post_id)}">${esc(ex.post_title)}</a>
        <span class="num muted">Score ${ex.score !== undefined && ex.score !== null ? Number(ex.score).toFixed(2) : "None"}</span>
      </div>
    </div>
  `).join("");

  return `
    <article class="topic-card">
      <header class="topic-card-head">
        <div>
          <div style="display:flex;flex-wrap:wrap;align-items:center;gap:var(--space-xs);margin-bottom:var(--space-2xs)">
            <span class="tag tag-info num">Topic ${num(topic.topic_id)}</span>
            ${badge(topic.character.tag, topic.character.tone)}
          </div>
          <p class="muted" style="font-size:var(--text-xs)">${esc(topic.character.note)}</p>
        </div>
        <div class="topic-card-stat">
          <strong class="num">${num(topic.document_count)}</strong>
          <span class="num">${percent(topic.document_ratio ?? (topic.document_count / totalDocs))}</span>
        </div>
      </header>
      <div>
        <h3 style="font-size:var(--text-xs);color:var(--color-muted);margin-bottom:var(--space-xs)">Top weighted NMF terms</h3>
        <div class="kw-weight-list">${kwBars}</div>
      </div>
      <div>
        <h3 style="font-size:var(--text-xs);color:var(--color-muted);margin-bottom:var(--space-xs)">Representative assigned comments</h3>
        <div class="preview-comment-list">${examples || `<p class="muted">No examples returned.</p>`}</div>
      </div>
    </article>
  `;
}

export async function renderTopics(container, ctx) {
  const [topicRes, summary] = await Promise.all([
    api.topics(),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
  ]);
  ctx.summary = summary;

  const rawTopics = (topicRes.items || []).map((t) => ({
    ...t,
    character: classifyClusterCharacter(t),
  }));
  const totalDocs = rawTopics.reduce((s, t) => s + Number(t.document_count || 0), 0) || Number(summary.comments || 0);
  const termsPerTopic = rawTopics[0]?.keywords?.length || 10;
  const largestTopic = [...rawTopics].sort((a, b) => Number(b.document_count) - Number(a.document_count))[0];
  const devTopics = rawTopics.filter((t) => t.character.script === "devanagari");
  const devComments = devTopics.reduce((s, t) => s + Number(t.document_count || 0), 0);
  const hinglishTopics = rawTopics.filter((t) => t.character.tone === "info");
  const hinglishComments = hinglishTopics.reduce((s, t) => s + Number(t.document_count || 0), 0);
  if (!rawTopics.some((t) => Number(t.topic_id) === Number(topicState.selectedId)) && largestTopic) {
    topicState.selectedId = largestTopic.topic_id;
  }

  function paint() {
    const q = topicState.search.trim().toLowerCase();
    const filtered = rawTopics.filter((t) => {
      if (topicState.scriptFilter !== "all" && t.character.script !== topicState.scriptFilter) return false;
      if (!q) return true;
      const hay = [t.label, `topic ${t.topic_id}`, ...(t.keywords || []), ...(t.representative_examples || []).map((e) => `${e.text} ${e.post_title}`)].join(" ").toLowerCase();
      return hay.includes(q);
    });
    const activeTopic = rawTopics.find((t) => Number(t.topic_id) === Number(topicState.selectedId)) || rawTopics[0];
    const largestPct = largestTopic && totalDocs ? (Number(largestTopic.document_count) / totalDocs) * 100 : 0;
    const devPct = totalDocs ? (devComments / totalDocs) * 100 : 0;
    const hinglishPct = totalDocs ? (hinglishComments / totalDocs) * 100 : 0;

    const fourthTally = devTopics.length > 0
      ? {
          label: "Devanagari script clusters",
          valueText: num(devTopics.length),
          countTo: devTopics.length,
          unit: devTopics.length === 1 ? "topic" : "topics",
          ratioPct: devPct,
          color: "var(--color-accent-2)",
          note: `${num(devComments)} comments (${devPct.toFixed(1)}%)`,
        }
      : {
          label: "Hinglish / code-mixed",
          valueText: num(hinglishTopics.length),
          countTo: hinglishTopics.length,
          unit: hinglishTopics.length === 1 ? "topic" : "topics",
          ratioPct: hinglishPct,
          color: "var(--color-role-advice)",
          note: `${num(hinglishComments)} comments (${hinglishPct.toFixed(1)}%) with Romanized Hindi terms`,
        };

    const tallyRow = tallyStrip([
      {
        label: "Unsupervised NMF topics",
        valueText: num(rawTopics.length),
        countTo: rawTopics.length,
        unit: "clusters",
        accent: true,
        note: `TF-IDF (1–2 grams, AutoMod boilerplate filtered), ${num(termsPerTopic)} weighted terms per cluster`,
      },
      {
        label: "Assigned comments",
        valueText: num(totalDocs),
        countTo: totalDocs,
        note: "Dominant cluster assignment across anonymized community comments",
      },
      {
        label: "Largest cluster share",
        valueText: `${largestPct.toFixed(1)}%`,
        countTo: largestPct,
        decimals: 1,
        suffix: "%",
        ratioPct: largestPct,
        color: "var(--color-role-advice)",
        note: largestTopic ? `Topic ${largestTopic.topic_id} (${num(largestTopic.document_count)} comments: ${(largestTopic.keywords || []).slice(0, 3).join(", ")})` : "",
      },
      fourthTally,
    ], "NMF topic overview");

    const modeControls = `
      <div class="view-toggle-bar">
        <div class="filter-pills" role="group" aria-label="Filter topics by dominant script">
          <button type="button" class="pill-btn ${topicState.scriptFilter === "all" ? "is-active" : ""}" data-topic-script="all">All (${rawTopics.length})</button>
          <button type="button" class="pill-btn ${topicState.scriptFilter === "latin" ? "is-active" : ""}" data-topic-script="latin">Latin / Hinglish (${rawTopics.length - devTopics.length})</button>
          ${devTopics.length > 0 ? `<button type="button" class="pill-btn ${topicState.scriptFilter === "devanagari" ? "is-active" : ""}" data-topic-script="devanagari">Devanagari (${devTopics.length})</button>` : ""}
        </div>
        <div class="segmented" role="group" aria-label="Topic visualization layout">
          <button type="button" class="seg-btn ${topicState.viewMode === "constellation" ? "is-active" : ""}" data-topic-mode="constellation">Proportional map</button>
          <button type="button" class="seg-btn ${topicState.viewMode === "all" ? "is-active" : ""}" data-topic-mode="all">All ${rawTopics.length} cards</button>
        </div>
      </div>
    `;

    const constellationSection = activeTopic
      ? `
        <div class="constellation-layout">
          <div>
            ${topicConstellationSvg(filtered.length ? filtered : rawTopics, activeTopic.topic_id)}
            <div class="spacer-sm"></div>
            <div class="chart-legend">
              <span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent)"></span><span>Thematic English cluster</span></span>
              <span class="legend-item"><span class="legend-swatch" style="background:var(--color-role-advice)"></span><span>Hinglish / code-mixed cluster</span></span>
              ${devTopics.length > 0 ? `<span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent-2)"></span><span>Devanagari cluster</span></span>` : ""}
            </div>
          </div>
          <div>
            ${renderTopicCard(activeTopic, totalDocs)}
          </div>
        </div>
      `
      : "";

    const allCardsHtml = filtered.length
      ? `<div class="topic-card-grid">${filtered.map((t) => renderTopicCard(t, totalDocs)).join("")}</div>`
      : emptyState({
          title: "No NMF topics match your filter",
          description: "Clear the keyword search or switch back to all clusters.",
          actionHtml: `<button type="button" class="btn btn-quiet btn-sm" data-clear-topics>Reset filters</button>`,
        });

    const distRows = [...rawTopics]
      .sort((a, b) => Number(b.document_count) - Number(a.document_count))
      .map((t) => ({
        label: `Topic ${t.topic_id}: ${(t.keywords || []).slice(0, 3).join(" · ")}`,
        sublabel: t.character.tag,
        count: Number(t.document_count || 0),
        color: t.character.script === "devanagari"
          ? "var(--color-accent-2)"
          : t.character.tone === "info"
            ? "var(--color-role-advice)"
            : "var(--color-accent)",
      }));

    container.innerHTML = `
      ${callout(
        `Unsupervised ${glossaryTip("nmf_topics", "NMF topic clusters")} are fit directly on anonymized public Reddit comments (unigrams &amp; bigrams, AutoMod boilerplate removed). Click any cluster circle to inspect its top weighted terms and representative replies, or compare against supervised domains in <a href="#/issues">Issues studio</a>.`,
        { tone: "info", title: "Unsupervised topic discovery:" }
      )}
      ${tallyRow}
      <section class="anim-enter" aria-label="Topic constellation and weighted terms">
        ${modeControls}
        <div class="spacer-md"></div>
        ${topicState.viewMode === "constellation" ? constellationSection : `
          <div class="field" style="max-width:420px;margin-bottom:var(--space-md)">
            <label for="topic-search">Search topic terms or representative comments</label>
            <input id="topic-search" type="search" value="${esc(topicState.search)}" placeholder="Search keywords (sambalpur, waitlist, placement, cat, fees)…">
          </div>
          ${allCardsHtml}
        `}
      </section>
      ${deck(`Corpus coverage across all ${num(rawTopics.length)} NMF topics`, {
        subtitle: `Number and share of anonymized comments assigned to each dominant unsupervised cluster (n = ${num(totalDocs)})`,
        body: horizontalBarChart(distRows, { total: totalDocs, unit: "comments", showShare: true }),
      })}
    `;

    container.querySelectorAll("[data-select-topic]").forEach((orb) => {
      const select = () => {
        topicState.selectedId = Number(orb.dataset.selectTopic);
        paint();
      };
      orb.addEventListener("click", select);
      orb.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); }
      });
    });

    container.querySelectorAll("[data-topic-mode]").forEach((btn) => {
      btn.addEventListener("click", () => { topicState.viewMode = btn.dataset.topicMode; paint(); });
    });
    container.querySelectorAll("[data-topic-script]").forEach((btn) => {
      btn.addEventListener("click", () => { topicState.scriptFilter = btn.dataset.topicScript; paint(); });
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
      topicState.scriptFilter = "all";
      paint();
    });
  }

  paint();
}
