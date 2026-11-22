/**
 * View 4: Conversation Audit (Visual Parent-Child Thread Tree & Evidence Studio)
 */
import { api } from "../api.js?v=6.0";
import { stackedCompositionBar } from "../charts.js?v=6.0";
import {
  badge,
  confidenceBadge,
  copyIdChip,
  deck,
  glossaryTip,
  languageBadge,
  resolutionBadge,
  roleBadge,
  sentimentBadge,
} from "../components.js?v=6.0";
import {
  CANONICAL_RESPONSE_TYPES,
  LANGUAGE_LABELS,
  RESOLUTION_META,
  RESPONSE_META,
  dateText,
  dateTimeText,
  esc,
  num,
  percent,
  pretty,
  shortCommentLabel,
  truncate,
} from "../format.js?v=6.0";
import { icons } from "../icons.js?v=6.0";

const threadUiState = {
  layout: "tree",
  filter: "all",
  showScenario: false,
};

function computeTreeOrder(comments) {
  const byId = new Map(comments.map((c) => [c.reddit_id, c]));
  const childrenMap = new Map();
  const roots = [];
  for (const c of comments) {
    const pid = c.parent_reddit_id;
    if (pid && byId.has(pid)) {
      if (!childrenMap.has(pid)) childrenMap.set(pid, []);
      childrenMap.get(pid).push(c);
    } else {
      roots.push(c);
    }
  }
  const ordered = [];
  const visited = new Set();
  function walk(node, depth) {
    if (visited.has(node.reddit_id)) return;
    visited.add(node.reddit_id);
    ordered.push({ comment: node, depth, parent: node.parent_reddit_id ? byId.get(node.parent_reddit_id) : null });
    for (const child of childrenMap.get(node.reddit_id) || []) walk(child, depth + 1);
  }
  for (const r of roots) walk(r, 0);
  for (const c of comments) {
    if (!visited.has(c.reddit_id)) walk(c, 0);
  }
  return ordered;
}

export async function renderThread(container, ctx, postId) {
  const data = await api.discussionDetail(postId);
  const post = data.post;
  const isSynthetic = Boolean(post.synthetic);
  const formatSub = (sub) => (sub && !sub.startsWith("r/") ? `r/${sub}` : (sub || "community"));
  const comments = data.comments || [];
  const treeNodes = computeTreeOrder(comments);
  const byId = new Map(comments.map((c) => [c.reddit_id, c]));

  const explicitEvidence = post.resolution_evidence_comment_id ? byId.get(post.resolution_evidence_comment_id) || null : null;
  const solutionComments = comments.filter((c) => c.response_type === "solution");
  const adviceComments = comments.filter((c) => c.response_type === "advice");
  const modelSolutionPositives = comments.filter((c) => c.predictions?.solution?.is_solution === true);

  const langCounts = {};
  for (const c of comments) {
    const l = c.language || "unknown";
    langCounts[l] = (langCounts[l] || 0) + 1;
  }

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

  function matchesFilter(c) {
    if (threadUiState.filter === "actionable") return c.is_solution || c.predictions?.solution?.is_solution || ["advice", "solution"].includes(c.predictions?.response?.response_type);
    if (threadUiState.filter === "evidence") return Boolean(c.resolution_evidence_type);
    if (threadUiState.filter === "multilingual") return c.language !== "en" || c.is_code_mixed;
    if (threadUiState.filter === "review") return Boolean(c.confidence_profile?.needs_review);
    return true;
  }

  function renderProbBars(probObj, orderedKeys = null) {
    if (!probObj || !Object.keys(probObj).length) return `<span class="muted">Not emitted</span>`;
    const keys = orderedKeys
      ? orderedKeys.filter((k) => k in probObj)
      : Object.entries(probObj).sort((a, b) => Number(b[1]) - Number(a[1])).map(([k]) => k);
    return `
      <div class="prob-mini-list">
        ${keys.map((k) => {
          const val = Number(probObj[k] || 0);
          return `
            <div class="prob-mini-row">
              <span class="prob-mini-label">${esc(pretty(k))}</span>
              <div class="prob-mini-track"><span class="prob-mini-fill" style="width:${Math.max(0, Math.min(100, val * 100)).toFixed(1)}%"></span></div>
              <span class="num prob-mini-val">${percent(val)}</span>
            </div>
          `;
        }).join("")}
      </div>
    `;
  }

  function renderCommentCard({ comment, depth, parent }) {
    const pred = comment.predictions || {};
    const pResp = pred.response?.response_type;
    const pSent = pred.sentiment?.sentiment;
    const pEmotions = pred.emotion?.emotions || [];
    const pSol = pred.solution?.is_solution;
    const pSolProb = pred.solution?.solution_probability;
    const pIssues = pred.issue?.predicted_issue_labels || [];
    const cProf = comment.confidence_profile || {};

    const isEvidence = Boolean(comment.resolution_evidence_type);
    const visualDepth = threadUiState.layout === "tree" ? Math.min(depth, 4) : 0;

    const parentRef = parent
      ? `
        <button type="button" class="parent-jump-bar" data-jump-comment="${esc(parent.reddit_id)}" title="Scroll to parent comment ${esc(parent.reddit_id)}">
          <span class="parent-jump-icon" aria-hidden="true">${icons.reply}</span>
          <span class="parent-jump-label">Replying to <strong class="num">${esc(shortCommentLabel(parent.reddit_id))}</strong> (${esc(RESPONSE_META[parent.predictions?.response?.response_type]?.short || "parent")})</span>
          <span class="parent-jump-excerpt">“${esc(truncate(parent.body, 96))}”</span>
        </button>
      `
      : `<div class="root-indicator">Top-level reply to original post</div>`;

    const evidenceBanner = isEvidence
      ? `
        <div class="comment-evidence-banner comment-evidence-${comment.resolution_evidence_type === "reported_success" ? "positive" : "warning"}">
          <strong>Explicit follow-up signal (${esc(RESOLUTION_META[comment.resolution_evidence_type]?.label || pretty(comment.resolution_evidence_type))})</strong>
          <span>First-person outcome language detected in comment text</span>
        </div>
      `
      : "";

    const scenarioComparison = isSynthetic && threadUiState.showScenario
      ? `
        <div class="scenario-compare-strip">
          <span class="scenario-compare-title">Scenario reference labels:</span>
          ${roleBadge(comment.response_type)}
          ${sentimentBadge(comment.sentiment)}
          <span class="meta-inline">Emotions: ${esc((comment.emotions || []).map(pretty).join(", "))}</span>
          ${comment.is_solution ? badge("Actionable", "positive") : `<span class="meta-inline">Non-solution</span>`}
        </div>
      `
      : "";

    return `
      <article id="comment-${esc(comment.reddit_id)}" class="thread-comment ${visualDepth > 0 ? "is-nested" : ""} ${isEvidence ? "is-evidence-comment" : ""}" style="--depth:${visualDepth}">
        ${evidenceBanner}
        <header class="comment-header">
          <div class="comment-id-line">
            <span class="num comment-seq">${esc(shortCommentLabel(comment.reddit_id))}</span>
            ${copyIdChip(comment.reddit_id)}
            <span class="num depth-tag">Depth ${depth}</span>
            ${languageBadge(comment.language, comment.script, comment.is_code_mixed)}
          </div>
          <span class="num comment-time">${dateTimeText(comment.created_at)}</span>
        </header>

        ${parentRef}
        <p class="comment-body" lang="${comment.language === "hi" ? "hi" : "en"}">${esc(comment.body)}</p>

        <div class="comment-predictions-strip" aria-label="Model predictions">
          <span class="strip-caption">Model output:</span>
          ${confidenceBadge(cProf.overall_confidence, cProf.confidence_tier, { needsReview: cProf.needs_review })}
          ${pResp ? roleBadge(pResp) : ""}
          ${pSent ? sentimentBadge(pSent) : ""}
          ${pEmotions.length ? `<span class="meta-inline">Emotions: ${esc(pEmotions.map(pretty).join(", "))}</span>` : ""}
          ${pSol === true
            ? badge(`Solution model: yes (${percent(pSolProb)})`, "positive")
            : pSol === false
              ? `<span class="meta-inline">Solution: no (<span class="num">${percent(pSolProb)}</span>)</span>`
              : ""}
        </div>

        ${scenarioComparison}

        <details class="comment-inspector">
          <summary>Inspect all 5 model probability distributions</summary>
          <div class="inspector-prob-grid">
            <div class="prob-box">
              <h4>Response role <span class="num">${percent(pred.response?.confidence)}</span></h4>
              ${renderProbBars(pred.response?.response_probabilities, CANONICAL_RESPONSE_TYPES)}
            </div>
            <div class="prob-box">
              <h4>Sentiment <span class="num">${percent(pred.sentiment?.confidence)}</span></h4>
              ${renderProbBars(pred.sentiment?.sentiment_probabilities, ["positive", "neutral", "negative"])}
            </div>
            <div class="prob-box">
              <h4>Emotion (multi-label)</h4>
              ${renderProbBars(pred.emotion?.emotion_probabilities)}
            </div>
            <div class="prob-box">
              <h4>Solution &amp; issue predictions</h4>
              <div class="prob-kv"><span>Solution probability</span><strong class="num">${percent(pSolProb)}</strong></div>
              <div class="prob-kv"><span>Predicted issues</span><strong>${pIssues.length ? esc(pIssues.map(pretty).join(", ")) : "None above threshold"}</strong></div>
              ${isSynthetic ? `
                <div class="prob-kv"><span>Reference role</span><code>${esc(comment.response_type)}</code></div>
                <div class="prob-kv"><span>Reference sentiment</span><code>${esc(comment.sentiment)}</code></div>
              ` : `
                <div class="prob-kv"><span>NMF topic</span><code>${esc(pred.response?.topic_label || "Unassigned")}</code></div>
              `}
            </div>
          </div>
        </details>
      </article>
    `;
  }

  function paint() {
    const sourceNodes = threadUiState.layout === "tree"
      ? treeNodes
      : comments.map((c) => ({ comment: c, depth: 0, parent: c.parent_reddit_id ? byId.get(c.parent_reddit_id) : null }));
    const filteredNodes = sourceNodes.filter((n) => matchesFilter(n.comment));

    const actionableCount = comments.filter((c) => c.is_solution || c.predictions?.solution?.is_solution).length;
    const multilingualCount = comments.filter((c) => c.language !== "en" || c.is_code_mixed).length;
    const reviewCount = comments.filter((c) => c.confidence_profile?.needs_review).length;

    const evidenceSidebarBody = `
      <div class="aside-section">
        <h3>Resolution status</h3>
        ${resolutionBadge(post.resolution_state)}
        <p style="margin-top:var(--space-xs);font-size:var(--text-sm);color:var(--color-ink-2)">${esc(RESOLUTION_META[post.resolution_state]?.description || "")}</p>
      </div>

      ${explicitEvidence
        ? `
          <div class="aside-section">
            <h3>Explicit outcome statement</h3>
            <blockquote class="aside-quote">“${esc(explicitEvidence.body)}”</blockquote>
            <button type="button" class="btn btn-quiet btn-sm btn-full" data-jump-comment="${esc(explicitEvidence.reddit_id)}">Jump to evidence ${esc(shortCommentLabel(explicitEvidence.reddit_id))}</button>
          </div>
        `
        : `
          <div class="aside-section">
            <h3>Explicit outcome statement</h3>
            <p class="muted" style="font-size:var(--text-sm)">No explicit outcome follow-up comment was detected in this thread.</p>
          </div>
        `}

      <div class="aside-section">
        <h3>Proposed solutions &amp; advice</h3>
        <dl class="compact-kv">
          <div><dt>Predicted <code>solution</code> replies</dt><dd class="num">${num(solutionComments.length)}</dd></div>
          <div><dt>Predicted <code>advice</code> replies</dt><dd class="num">${num(adviceComments.length)}</dd></div>
          <div><dt>Binary model positive</dt><dd class="num">${num(modelSolutionPositives.length)}</dd></div>
        </dl>
        ${(solutionComments.length || adviceComments.length)
          ? `<div class="aside-jump-list">${[...solutionComments, ...adviceComments].map((c) => `
              <button type="button" class="aside-jump-item" data-jump-comment="${esc(c.reddit_id)}">
                <span class="aside-jump-head"><span class="num">${esc(shortCommentLabel(c.reddit_id))}</span> ${esc(RESPONSE_META[c.response_type]?.label || c.response_type)}</span>
                <span class="aside-jump-text">${esc(truncate(c.body, 88))}</span>
              </button>
            `).join("")}</div>`
          : ""}
      </div>

      <div class="aside-section">
        <h3>Language &amp; script mix</h3>
        <div class="label-chip-wrap">
          ${Object.entries(langCounts).map(([lang, count]) => `<span class="tag tag-neutral">${esc(LANGUAGE_LABELS[lang] || lang)}: <span class="num">${num(count)}</span></span>`).join("")}
        </div>
      </div>

      <div class="aside-section">
        <p class="muted" style="font-size:var(--text-xs)">${glossaryTip("in_sample_vs_heldout", "Model-inferred labels")}. Author usernames are stripped and inline PII is redacted.</p>
      </div>
    `;

    container.innerHTML = `
      <div class="thread-topbar">
        <a class="btn btn-quiet btn-sm" href="#/explorer">Back to discussion workbench</a>
        <div class="thread-topbar-ids">
          ${copyIdChip(post.reddit_id, { label: "post_id" })}
          <a class="btn btn-quiet btn-sm" href="#/issues?selected=${encodeURIComponent(post.issue_key)}">Issue: ${esc(pretty(post.issue_category))}</a>
        </div>
      </div>

      <div class="thread-workbench">
        <div class="thread-primary">
          <article class="deck anim-enter">
            <div class="deck-body">
              <div class="op-card-meta">
                <a class="tag tag-neutral" href="#/explorer?issue=${encodeURIComponent(post.issue_key)}">${esc(pretty(post.issue_category))}</a>
                ${resolutionBadge(post.resolution_state)}
                ${confidenceBadge(post.mean_confidence, post.confidence_tier)}
                <span class="tag tag-neutral">${esc(formatSub(post.subreddit))}</span>
                <span class="num muted" style="font-size:var(--text-xs)">${dateText(post.created_at)}</span>
              </div>
              <h2 class="op-title">${esc(post.title)}</h2>
              ${post.body ? `<p class="op-body">${esc(post.body)}</p>` : ""}
              <div class="op-composition">
                <span class="op-comp-label">Predicted comment roles in this thread (${num(comments.length)} replies):</span>
                ${stackedCompositionBar(respSegments, { total: comments.length, unit: "comments", ariaLabel: "Thread predicted response roles" })}
              </div>
            </div>
          </article>

          <div class="thread-controls-bar">
            <div class="segmented" role="group" aria-label="Comment thread layout">
              <button type="button" class="seg-btn ${threadUiState.layout === "tree" ? "is-active" : ""}" data-thread-layout="tree">Threaded tree</button>
              <button type="button" class="seg-btn ${threadUiState.layout === "flat" ? "is-active" : ""}" data-thread-layout="flat">Chronological</button>
            </div>

            <div class="filter-pills" role="group" aria-label="Filter comments in thread">
              <button type="button" class="pill-btn ${threadUiState.filter === "all" ? "is-active" : ""}" data-thread-filter="all">All (${comments.length})</button>
              <button type="button" class="pill-btn ${threadUiState.filter === "actionable" ? "is-active" : ""}" data-thread-filter="actionable">Advice &amp; solutions (${actionableCount})</button>
              <button type="button" class="pill-btn ${threadUiState.filter === "evidence" ? "is-active" : ""}" data-thread-filter="evidence">Outcome evidence (${explicitEvidence ? 1 : 0})</button>
              <button type="button" class="pill-btn ${threadUiState.filter === "multilingual" ? "is-active" : ""}" data-thread-filter="multilingual">Hindi / Hinglish (${multilingualCount})</button>
              <button type="button" class="pill-btn ${threadUiState.filter === "review" ? "is-active" : ""}" data-thread-filter="review">Review / low conf (${reviewCount})</button>
            </div>

            ${isSynthetic ? `
              <label class="toggle-check">
                <input type="checkbox" id="toggle-scenario-labels" ${threadUiState.showScenario ? "checked" : ""}>
                <span>Compare scenario labels</span>
              </label>
            ` : ""}
          </div>

          <section class="comment-stream" aria-label="Comment reply thread">
            ${filteredNodes.length
              ? filteredNodes.map(renderCommentCard).join("")
              : `<div class="empty-state"><h3>No comments match this filter</h3><div class="empty-state-actions"><button type="button" class="btn btn-quiet btn-sm" data-thread-filter="all">Show all ${comments.length} comments</button></div></div>`}
          </section>
        </div>

        <aside class="thread-sidebar" aria-label="Thread resolution evidence inspector">
          ${deck("Evidence inspector", {
            subtitle: "Model proposals and explicit follow-up signals kept strictly separate",
            body: evidenceSidebarBody,
          })}
        </aside>
      </div>
    `;

    container.querySelectorAll("[data-thread-layout]").forEach((btn) => {
      btn.addEventListener("click", () => { threadUiState.layout = btn.dataset.threadLayout; paint(); });
    });
    container.querySelectorAll("[data-thread-filter]").forEach((btn) => {
      btn.addEventListener("click", () => { threadUiState.filter = btn.dataset.threadFilter; paint(); });
    });
    const scenToggle = container.querySelector("#toggle-scenario-labels");
    if (scenToggle) {
      scenToggle.addEventListener("change", () => { threadUiState.showScenario = scenToggle.checked; paint(); });
    }
    container.querySelectorAll("[data-jump-comment]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.dataset.jumpComment;
        let el = container.querySelector(`#comment-${CSS.escape(targetId)}`);
        if (!el && threadUiState.filter !== "all") {
          threadUiState.filter = "all";
          paint();
          el = container.querySelector(`#comment-${CSS.escape(targetId)}`);
        }
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
          el.classList.remove("flash-highlight");
          void el.offsetWidth;
          el.classList.add("flash-highlight");
        }
      });
    });
  }

  paint();
}
