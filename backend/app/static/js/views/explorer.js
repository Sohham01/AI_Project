/**
 * View 3: Discussion Workbench (Master-Detail Split Studio with 3 Live Preview Tabs)
 * Directly inspired by crisp modern split-pane narrative consoles (Image 2),
 * upgraded in Dusk Smoked Glass: Summary & Evidence | Reply Flow | Model Signals.
 */
import { api } from "../api.js?v=6.0";
import { responseMiniStrip, stackedCompositionBar } from "../charts.js?v=6.0";
import {
  badge,
  confidenceBadge,
  copyIdChip,
  emptyState,
  languageBadge,
  resolutionBadge,
  roleBadge,
  sentimentBadge,
} from "../components.js?v=6.0";
import {
  CANONICAL_RESPONSE_TYPES,
  RESOLUTION_META,
  RESPONSE_META,
  SENTIMENT_ORDER,
  dateText,
  esc,
  issueLabelFromKey,
  num,
  percent,
  pretty,
  shortCommentLabel,
  truncate,
} from "../format.js?v=6.0";
import { icons } from "../icons.js?v=6.0";

const FILTER_KEYS = ["q", "community", "issue", "sentiment", "emotion", "response", "confidence"];

export async function renderExplorer(container, ctx, params = {}) {
  const summary = ctx.summary || await api.summary();
  ctx.summary = summary;

  const filters = {};
  for (const key of FILTER_KEYS) {
    if (params[key] && String(params[key]).trim()) filters[key] = String(params[key]).trim();
  }
  const page = Math.max(1, parseInt(params.page || "1", 10) || 1);
  const pageSize = 12;

  const data = await api.discussions({ ...filters, page, page_size: pageSize });
  const items = data.items || [];
  const totalPages = Math.max(1, Math.ceil((data.total || 0) / data.page_size));
  let selectedId = params.preview && items.some((p) => p.id === params.preview)
    ? params.preview
    : (items[0]?.id || null);
  let activePreviewTab = "evidence";
  let cachedDetail = null;

  const buildHash = (nextFilters, nextPage = 1) => {
    const sp = new URLSearchParams();
    for (const key of FILTER_KEYS) {
      if (nextFilters[key]) sp.set(key, nextFilters[key]);
    }
    if (nextPage > 1) sp.set("page", String(nextPage));
    const qs = sp.toString();
    return qs ? `#/explorer?${qs}` : "#/explorer";
  };

  const activeChips = [];
  if (filters.q) activeChips.push({ key: "q", label: `Query: “${filters.q}”` });
  if (filters.community) activeChips.push({ key: "community", label: `Community: ${filters.community}` });
  if (filters.issue) activeChips.push({ key: "issue", label: `Issue: ${issueLabelFromKey(filters.issue, summary)}` });
  if (filters.sentiment) activeChips.push({ key: "sentiment", label: `Sentiment: ${pretty(filters.sentiment)}` });
  if (filters.emotion) activeChips.push({ key: "emotion", label: `Emotion: ${pretty(filters.emotion)}` });
  if (filters.response) activeChips.push({ key: "response", label: `Response: ${RESPONSE_META[filters.response]?.label || pretty(filters.response)}` });
  if (filters.confidence) activeChips.push({ key: "confidence", label: `Confidence: ${filters.confidence === "review" ? "Needs review" : pretty(filters.confidence)}` });

  const chipsHtml = activeChips.length
    ? `
      <div class="active-filters" aria-label="Active filters">
        <span class="active-filters-label">Active filters:</span>
        ${activeChips.map((chip) => {
          const reduced = { ...filters };
          delete reduced[chip.key];
          return `<a class="filter-chip" href="${esc(buildHash(reduced, 1))}"><span>${esc(chip.label)}</span><span aria-hidden="true">${icons.close}</span></a>`;
        }).join("")}
        <a class="filter-reset-link" href="#/explorer">Reset all</a>
      </div>
    `
    : "";

  const formatSub = (sub) => (sub && !sub.startsWith("r/") ? `r/${sub}` : (sub || "community"));
  const issueOptions = (summary.issue_distribution || [])
    .map((item) => `<option value="${esc(item.key)}" ${filters.issue === item.key ? "selected" : ""}>${esc(pretty(item.label))}</option>`)
    .join("");
  const communityOptions = (summary.community_distribution || [])
    .map((item) => `<option value="${esc(item.label)}" ${filters.community === item.label ? "selected" : ""}>${esc(formatSub(item.label))}</option>`)
    .join("");
  const emotionOptions = (summary.emotion_distribution || [])
    .map((item) => `<option value="${esc(item.label)}" ${filters.emotion === item.label ? "selected" : ""}>${esc(pretty(item.label))}</option>`)
    .join("");
  const responseOptions = CANONICAL_RESPONSE_TYPES
    .map((key, idx) => `<option value="${esc(key)}" ${filters.response === key ? "selected" : ""}>${idx + 1}. ${esc(RESPONSE_META[key].label)}</option>`)
    .join("");

  const filterBar = `
    <form id="explorer-filter-form" class="filter-dock anim-enter" role="search" aria-label="Filter community discussions">
      <div class="filter-grid filter-grid-explorer">
        <div class="field field-span-2">
          <label for="ex-q">Search titles, posts &amp; comments</label>
          <input id="ex-q" name="q" type="search" maxlength="120" value="${esc(filters.q || "")}" placeholder="Placements, waitlist, hostel, fees, CAT…">
        </div>
        <div class="field">
          <label for="ex-issue">Issue domain</label>
          <select id="ex-issue" name="issue">
            <option value="">All issue domains</option>
            ${issueOptions}
          </select>
        </div>
        <div class="field">
          <label for="ex-response">Response class (predicted)</label>
          <select id="ex-response" name="response">
            <option value="">Any canonical response</option>
            ${responseOptions}
          </select>
        </div>
        <div class="field">
          <label for="ex-sentiment">Sentiment (predicted)</label>
          <select id="ex-sentiment" name="sentiment">
            <option value="">Any sentiment</option>
            ${SENTIMENT_ORDER.map((s) => `<option value="${s}" ${filters.sentiment === s ? "selected" : ""}>${pretty(s)}</option>`).join("")}
          </select>
        </div>
        <div class="field">
          <label for="ex-emotion">Emotion (predicted)</label>
          <select id="ex-emotion" name="emotion">
            <option value="">Any emotion</option>
            ${emotionOptions}
          </select>
        </div>
        <div class="field">
          <label for="ex-community">Subreddit</label>
          <select id="ex-community" name="community">
            <option value="">All subreddits</option>
            ${communityOptions}
          </select>
        </div>
        <div class="field">
          <label for="ex-confidence">Prediction confidence</label>
          <select id="ex-confidence" name="confidence">
            <option value="">Any confidence tier</option>
            <option value="high" ${filters.confidence === "high" ? "selected" : ""}>High (&ge;80%)</option>
            <option value="medium" ${filters.confidence === "medium" ? "selected" : ""}>Medium (55–79%)</option>
            <option value="low" ${filters.confidence === "low" ? "selected" : ""}>Low (&lt;55%)</option>
            <option value="review" ${filters.confidence === "review" ? "selected" : ""}>Contains review comments</option>
          </select>
        </div>
      </div>
      <div class="filter-actions">
        <button type="submit" class="btn btn-primary btn-sm">Apply filters</button>
        ${activeChips.length ? `<a class="btn btn-quiet btn-sm" href="#/explorer">Clear filters</a>` : ""}
        <span class="filter-hint">Model filters match threads containing at least one comment with that predicted label.</span>
      </div>
    </form>
  `;

  const startIdx = data.total ? (data.page - 1) * data.page_size + 1 : 0;
  const endIdx = Math.min(data.total, data.page * data.page_size);

  const pagination = data.total > data.page_size
    ? `
      <nav class="pagination-bar" aria-label="Discussion results pagination">
        <div class="pagination-info">Showing <strong class="num">${num(startIdx)} to ${num(endIdx)}</strong> of <strong class="num">${num(data.total)}</strong> matching threads</div>
        <div class="pagination-controls">
          <a class="btn btn-quiet btn-sm ${data.page <= 1 ? "is-disabled" : ""}" href="${data.page > 1 ? esc(buildHash(filters, 1)) : "#"}" aria-disabled="${data.page <= 1 ? "true" : "false"}">First</a>
          <a class="btn btn-quiet btn-sm ${data.page <= 1 ? "is-disabled" : ""}" href="${data.page > 1 ? esc(buildHash(filters, data.page - 1)) : "#"}" aria-disabled="${data.page <= 1 ? "true" : "false"}">Previous</a>
          <span class="num pagination-page">Page <strong>${num(data.page)}</strong> of ${num(totalPages)}</span>
          <a class="btn btn-quiet btn-sm ${!data.has_more ? "is-disabled" : ""}" href="${data.has_more ? esc(buildHash(filters, data.page + 1)) : "#"}" aria-disabled="${!data.has_more ? "true" : "false"}">Next</a>
          <a class="btn btn-quiet btn-sm ${!data.has_more ? "is-disabled" : ""}" href="${data.has_more ? esc(buildHash(filters, totalPages)) : "#"}" aria-disabled="${!data.has_more ? "true" : "false"}">Last</a>
        </div>
      </nav>
    `
    : `<div class="results-bar"><span>Showing all <strong class="num">${num(data.total)}</strong> matching thread${data.total === 1 ? "" : "s"}</span><span>Ordered by post date (newest first)</span></div>`;

  const splitWorkspace = items.length
    ? `
      <section class="deck anim-enter" aria-label="Discussions stream and live split preview">
        <div class="explorer-split">
          <div class="explorer-list-pane" role="list" aria-label="Matching community discussions">
            <div class="pane-head">
              <strong class="pane-title">Narratives (<span class="num">${num(data.total)}</span>)</strong>
              <span class="pane-sub">Click any thread to live-inspect</span>
            </div>
            ${items.map((post) => {
              const isSel = post.id === selectedId;
              return `
                <article class="discussion-card ${isSel ? "is-selected" : ""}" role="listitem" data-preview-id="${esc(post.id)}">
                  <div class="discussion-card-badges">
                    <span class="meta-issue-label">${esc(pretty(post.issue))}</span>
                    <span class="meta-sep" aria-hidden="true">·</span>
                    ${resolutionBadge(post.resolution_state)}
                    ${confidenceBadge(post.mean_confidence, post.confidence_tier)}
                    <span class="num muted" style="font-size:var(--text-xs);margin-left:auto">${dateText(post.created_at)}</span>
                  </div>
                  <h2 class="discussion-card-title">
                    <a href="#/thread/${encodeURIComponent(post.id)}">${esc(post.title)}</a>
                  </h2>
                  <p class="discussion-card-excerpt">${esc(post.body || "Image or title-only community post — open thread to inspect replies.")}</p>
                  <div class="discussion-card-footer">
                    ${responseMiniStrip(post.response_counts)}
                    <a class="row-open-link" href="#/thread/${encodeURIComponent(post.id)}">Open full tree</a>
                  </div>
                </article>
              `;
            }).join("")}
          </div>

          <aside class="explorer-preview-pane" id="explorer-preview-pane" aria-label="Selected thread interactive preview">
            <div class="skeleton-card h-320"></div>
          </aside>
        </div>
      </section>
    `
    : emptyState({
        title: "No discussions match these filters",
        description: "Remove one of the active filters or broaden your search phrase.",
        actionHtml: `<a class="btn btn-primary btn-sm" href="#/explorer">Reset all filters</a>`,
      });

  container.innerHTML = `
    ${filterBar}
    ${chipsHtml}
    <div class="results-bar">
      <span><strong class="num">${num(data.total)}</strong> matching thread${data.total === 1 ? "" : "s"}</span>
      <span class="canonical-key-inline">
        ${CANONICAL_RESPONSE_TYPES.map((k) => `<span class="micro-legend"><span class="legend-swatch" style="background:${RESPONSE_META[k].color}" aria-hidden="true"></span>${RESPONSE_META[k].short}</span>`).join("")}
      </span>
    </div>
    ${splitWorkspace}
    ${pagination}
  `;

  function paintPreviewPane() {
    const pane = container.querySelector("#explorer-preview-pane");
    if (!pane || !cachedDetail) return;

    const post = cachedDetail.post;
    const comments = cachedDetail.comments || [];
    const evidence = post.resolution_evidence_comment_id
      ? comments.find((c) => c.reddit_id === post.resolution_evidence_comment_id)
      : null;
    const actionable = comments.filter((c) => c.is_solution || ["advice", "solution"].includes(c.response_type));
    const respCounts = Object.fromEntries(CANONICAL_RESPONSE_TYPES.map((k) => [k, 0]));
    for (const c of comments) {
      const r = c.predictions?.response?.response_type;
      if (r && respCounts[r] !== undefined) respCounts[r] += 1;
    }
    const respSegments = CANONICAL_RESPONSE_TYPES.map((k) => ({
      key: k,
      label: RESPONSE_META[k].short,
      count: respCounts[k] || 0,
      color: RESPONSE_META[k].color,
    }));

    let tabBody = "";
    if (activePreviewTab === "evidence") {
      tabBody = `
        <div class="aside-section">
          <h3>Predicted 6-class response mix</h3>
          ${stackedCompositionBar(respSegments, { total: comments.length, unit: "comments", ariaLabel: "Preview thread response roles" })}
        </div>

        <div class="aside-section">
          <h3>Resolution evidence status</h3>
          <p style="font-size:var(--text-sm);color:var(--color-ink-2);margin-bottom:var(--space-xs)">${esc(RESOLUTION_META[post.resolution_state]?.description || "")}</p>
          ${evidence
            ? `<blockquote class="preview-quote">“${esc(evidence.body)}”<footer>Explicit follow-up signal in comment <span class="num">${esc(shortCommentLabel(evidence.reddit_id))}</span></footer></blockquote>`
            : `<p class="muted" style="font-size:var(--text-sm)">No explicit outcome follow-up statement detected in this thread.</p>`}
        </div>

        <div class="aside-section">
          <h3>Actionable proposals &amp; advice (<span class="num">${num(actionable.length)}</span>)</h3>
          <div class="preview-comment-list">
            ${actionable.slice(0, 3).map((c) => `
              <div class="preview-comment-item">
                <div class="preview-comment-meta">
                  <span class="num comment-seq">${esc(shortCommentLabel(c.reddit_id))}</span>
                  ${roleBadge(c.predictions?.response?.response_type || c.response_type)}
                  ${sentimentBadge(c.predictions?.sentiment?.sentiment || c.sentiment)}
                  ${languageBadge(c.language, c.script, c.is_code_mixed)}
                </div>
                <p>${esc(truncate(c.body, 200))}</p>
              </div>
            `).join("") || `<p class="muted">No advice or solution replies in this thread.</p>`}
          </div>
        </div>
      `;
    } else if (activePreviewTab === "replies") {
      tabBody = `
        <div class="preview-comment-list">
          ${comments.map((c) => `
            <div class="preview-comment-item">
              <div class="preview-comment-meta">
                <span class="num comment-seq">${esc(shortCommentLabel(c.reddit_id))}</span>
                ${roleBadge(c.predictions?.response?.response_type || c.response_type)}
                ${sentimentBadge(c.predictions?.sentiment?.sentiment || c.sentiment)}
                ${languageBadge(c.language, c.script, c.is_code_mixed)}
              </div>
              <p>${esc(truncate(c.body, 190))}</p>
            </div>
          `).join("")}
        </div>
      `;
    } else {
      tabBody = `
        <div class="preview-comment-list">
          ${comments.map((c) => {
            const pSol = c.predictions?.solution?.is_solution;
            const pSolProb = c.predictions?.solution?.solution_probability;
            const pEm = c.predictions?.emotion?.emotions || [];
            return `
              <div class="preview-comment-item">
                <div class="preview-comment-meta">
                  <span class="num comment-seq">${esc(shortCommentLabel(c.reddit_id))}</span>
                  ${roleBadge(c.predictions?.response?.response_type || c.response_type)}
                  ${pSol ? badge(`Solution: yes (${percent(pSolProb)})`, "positive") : `<span class="meta-inline">Solution: no (<span class="num">${percent(pSolProb)}</span>)</span>`}
                </div>
                <div class="meta-inline">Emotions: ${esc(pEm.length ? pEm.map(pretty).join(", ") : "None")} | Confidence: <span class="num">${percent(c.predictions?.response?.confidence)}</span></div>
              </div>
            `;
          }).join("")}
        </div>
      `;
    }

    pane.innerHTML = `
      <div class="preview-inner">
        <div class="preview-head">
          <div class="preview-head-meta">
            <span class="tag tag-neutral">${esc(pretty(post.issue_category))}</span>
            ${resolutionBadge(post.resolution_state)}
            ${copyIdChip(post.reddit_id)}
          </div>
          <a class="btn btn-primary btn-sm" href="#/thread/${encodeURIComponent(post.reddit_id)}">Open full tree</a>
        </div>

        <h2 class="preview-title">${esc(post.title)}</h2>
        ${post.body ? `<p class="preview-body">${esc(post.body)}</p>` : `<p class="preview-body muted">Title-only or media discussion post.</p>`}
        <p class="preview-dateline">${esc(formatSub(post.subreddit))}, ${dateText(post.created_at)}, <span class="num">${num(comments.length)}</span> replies</p>

        <div class="segmented" role="tablist" aria-label="Preview inspector tabs">
          <button type="button" role="tab" aria-selected="${activePreviewTab === "evidence"}" class="seg-btn ${activePreviewTab === "evidence" ? "is-active" : ""}" data-preview-tab="evidence">Evidence</button>
          <button type="button" role="tab" aria-selected="${activePreviewTab === "replies"}" class="seg-btn ${activePreviewTab === "replies" ? "is-active" : ""}" data-preview-tab="replies">All ${comments.length} replies</button>
          <button type="button" role="tab" aria-selected="${activePreviewTab === "signals"}" class="seg-btn ${activePreviewTab === "signals" ? "is-active" : ""}" data-preview-tab="signals">Model signals</button>
        </div>

        ${tabBody}
      </div>
    `;

    pane.querySelectorAll("[data-preview-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activePreviewTab = btn.dataset.previewTab;
        paintPreviewPane();
      });
    });
  }

  async function loadPreview(postId) {
    const pane = container.querySelector("#explorer-preview-pane");
    if (!pane || !postId) return;
    pane.innerHTML = `<div class="skeleton-card h-320"></div>`;
    try {
      cachedDetail = await api.discussionDetail(postId);
      paintPreviewPane();
    } catch (err) {
      pane.innerHTML = `<p class="muted">Preview could not load: ${esc(err.message)}</p>`;
    }
  }

  container.querySelectorAll(".discussion-card[data-preview-id]").forEach((card) => {
    card.addEventListener("click", (e) => {
      if (e.target.closest(".row-open-link") || e.target.closest("[data-copy-value]")) return;
      if (window.innerWidth > 1023) {
        e.preventDefault();
        const nextId = card.dataset.previewId;
        if (nextId && nextId !== selectedId) {
          selectedId = nextId;
          container.querySelectorAll(".discussion-card").forEach((c) => {
            c.classList.toggle("is-selected", c.dataset.previewId === selectedId);
          });
          loadPreview(selectedId);
        }
      }
    });
  });

  if (selectedId) loadPreview(selectedId);

  const form = container.querySelector("#explorer-filter-form");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const fd = new FormData(form);
      const next = {};
      for (const key of FILTER_KEYS) {
        const val = String(fd.get(key) || "").trim();
        if (val) next[key] = val;
      }
      window.location.hash = buildHash(next, 1);
    });
    form.querySelectorAll("select").forEach((sel) => sel.addEventListener("change", () => form.requestSubmit()));
  }
}
