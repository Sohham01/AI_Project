/**
 * View 2: Issues Intelligence Studio (Visual Pods + Sticky Deep Dive + 9-Col Sortable Table Toggle)
 */
import { api } from "../api.js";
import { stackedCompositionBar } from "../charts.js";
import { callout, copyIdChip, deck, emptyState, resolutionBadge } from "../components.js";
import {
  RESOLUTION_META,
  RESOLUTION_ORDER,
  SENTIMENT_META,
  SENTIMENT_ORDER,
  dateText,
  esc,
  num,
  pretty,
  share,
  truncate,
} from "../format.js";

const viewState = {
  mode: "cards",
  search: "",
  sortBy: "actionable_recommendations",
  sortDir: "desc",
  selectedKey: "",
};

export async function renderIssues(container, ctx, params = {}) {
  const [issuesRes, sentimentRes, resolutionRes] = await Promise.all([
    api.issues(),
    api.sentiment(),
    api.resolution(),
  ]);

  const sentimentByIssue = Object.fromEntries(
    (sentimentRes.by_issue || []).map((row) => [
      row.issue,
      Object.fromEntries((row.distribution || []).map((d) => [d.label, Number(d.count || 0)])),
    ])
  );

  const resolutionByIssue = {};
  for (const row of resolutionRes.items || []) {
    if (!resolutionByIssue[row.issue_key]) resolutionByIssue[row.issue_key] = [];
    resolutionByIssue[row.issue_key].push(row);
  }

  const enriched = (issuesRes.items || []).map((issue) => {
    const sent = sentimentByIssue[issue.key] || { positive: 0, neutral: 0, negative: 0 };
    const sentTotal = sent.positive + sent.neutral + sent.negative;
    const negShare = sentTotal > 0 ? sent.negative / sentTotal : 0;
    const resRows = resolutionByIssue[issue.key] || [];
    const resCounts = Object.fromEntries(RESOLUTION_ORDER.map((k) => [k, 0]));
    for (const r of resRows) {
      if (resCounts[r.state] !== undefined) resCounts[r.state] += 1;
    }
    return { ...issue, sentiment: sent, sentTotal, negShare, resRows, resCounts };
  });

  if (params.selected && enriched.some((i) => i.key === params.selected)) {
    viewState.selectedKey = params.selected;
  } else if (!viewState.selectedKey && enriched.length) {
    viewState.selectedKey = enriched[0].key;
  }

  async function paint() {
    const q = viewState.search.trim().toLowerCase();
    const rows = enriched.filter((i) => !q || i.label.toLowerCase().includes(q) || i.key.toLowerCase().includes(q));
    rows.sort((a, b) => {
      const dir = viewState.sortDir === "asc" ? 1 : -1;
      if (viewState.sortBy === "label") return a.label.localeCompare(b.label) * dir;
      return (Number(a[viewState.sortBy] || 0) - Number(b[viewState.sortBy] || 0)) * dir;
    });
    if (rows.length && !rows.some((r) => r.key === viewState.selectedKey)) {
      viewState.selectedKey = rows[0].key;
    }
    const active = enriched.find((i) => i.key === viewState.selectedKey) || null;

    const toolbar = `
      <div class="view-toggle-bar">
        <div class="field" style="flex:1 1 260px;max-width:380px">
          <label for="issues-filter-input">Filter issue domains</label>
          <input id="issues-filter-input" type="search" value="${esc(viewState.search)}" placeholder="Search by issue name or key (placements, hostel)…">
        </div>
        <div class="segmented" role="group" aria-label="Issues display mode">
          <button type="button" class="seg-btn ${viewState.mode === "cards" ? "is-active" : ""}" data-issue-mode="cards">Visual cards</button>
          <button type="button" class="seg-btn ${viewState.mode === "table" ? "is-active" : ""}" data-issue-mode="table">9-column data table</button>
        </div>
      </div>
    `;

    const cardsMarkup = rows.length
      ? `
        <div class="issue-pod-grid">
          ${rows.map((issue) => {
            const isSel = active && active.key === issue.key;
            const actPct = share(issue.actionable_recommendations, issue.comments, 0);
            const sentTotal = issue.sentTotal || 1;
            const posW = (issue.sentiment.positive / sentTotal) * 100;
            const neuW = (issue.sentiment.neutral / sentTotal) * 100;
            const negW = (issue.sentiment.negative / sentTotal) * 100;
            return `
              <button type="button" class="issue-pod ${isSel ? "is-selected" : ""}" data-select-issue="${esc(issue.key)}">
                <div class="issue-pod-head">
                  <div>
                    <div class="issue-pod-title">${esc(pretty(issue.label))}</div>
                    <div class="issue-pod-sub">${esc(issue.key)}</div>
                  </div>
                  <span class="issue-pod-pct">${actPct}</span>
                </div>
                <svg class="matrix-bar-svg" viewBox="0 0 100 6" preserveAspectRatio="none" aria-hidden="true">
                  <rect x="0" y="0" width="${posW.toFixed(1)}" height="6" fill="var(--color-positive)"/>
                  <rect x="${posW.toFixed(1)}" y="0" width="${neuW.toFixed(1)}" height="6" fill="var(--color-neutral-bar)"/>
                  <rect x="${(posW + neuW).toFixed(1)}" y="0" width="${negW.toFixed(1)}" height="6" fill="var(--color-negative)"/>
                </svg>
                <div class="issue-pod-stats">
                  <span class="num">${num(issue.solutions_proposed)} sol, ${num(issue.advice_comments)} adv</span>
                  <span class="num">${num(issue.reported_successes)} / ${num(issue.posts)} success</span>
                </div>
              </button>
            `;
          }).join("")}
        </div>
      `
      : emptyState({ title: "No issue domains match your filter", description: "Clear the search input to view all 8 synthetic categories." });

    const sortBtn = (col, label, isNum = false) => {
      const activeCol = viewState.sortBy === col;
      const dirLabel = activeCol ? (viewState.sortDir === "asc" ? " (asc)" : " (desc)") : "";
      return `<th scope="col" class="${isNum ? "num" : ""}" aria-sort="${activeCol ? (viewState.sortDir === "asc" ? "ascending" : "descending") : "none"}"><button type="button" class="th-sort ${activeCol ? "is-sorted" : ""}" data-issue-sort="${col}">${esc(label)}${esc(dirLabel)}</button></th>`;
    };

    const tableMarkup = rows.length
      ? `
        <div class="table-scroll">
          <table class="data-table sortable-table" aria-label="Synthetic issue categories with solution and outcome metrics">
            <thead>
              <tr>
                ${sortBtn("label", "Issue category")}
                ${sortBtn("posts", "Posts", true)}
                ${sortBtn("comments", "Comments", true)}
                ${sortBtn("solutions_proposed", "Solutions", true)}
                ${sortBtn("advice_comments", "Advice", true)}
                ${sortBtn("actionable_recommendations", "Actionable", true)}
                ${sortBtn("negShare", "Negative", true)}
                ${sortBtn("reported_successes", "Reported success", true)}
                ${sortBtn("no_confirmed_outcome", "Unconfirmed", true)}
              </tr>
            </thead>
            <tbody>
              ${rows.map((issue) => {
                const isSel = active && active.key === issue.key;
                return `
                  <tr class="selectable-row ${isSel ? "is-selected" : ""}" data-select-issue="${esc(issue.key)}">
                    <td>
                      <button type="button" class="row-select-btn" data-select-issue="${esc(issue.key)}">
                        <span class="table-primary-link">${esc(pretty(issue.label))}</span>
                        <span class="table-subcode">${esc(issue.key)}</span>
                      </button>
                    </td>
                    <td class="num">${num(issue.posts)}</td>
                    <td class="num">${num(issue.comments)}</td>
                    <td class="num">${num(issue.solutions_proposed)}</td>
                    <td class="num">${num(issue.advice_comments)}</td>
                    <td class="num"><strong>${num(issue.actionable_recommendations)}</strong> <span class="muted">(${share(issue.actionable_recommendations, issue.comments, 0)})</span></td>
                    <td class="num">${(issue.negShare * 100).toFixed(1)}%</td>
                    <td class="num">${num(issue.reported_successes)}</td>
                    <td class="num">${num(issue.no_confirmed_outcome)}</td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      `
      : emptyState({ title: "No issue categories match", description: "Clear the search input." });

    let inspectorHtml = "";
    if (active) {
      const sentSegs = SENTIMENT_ORDER.map((k) => ({
        key: k,
        label: SENTIMENT_META[k].label,
        count: active.sentiment[k] || 0,
        color: SENTIMENT_META[k].color,
        href: `#/explorer?issue=${encodeURIComponent(active.key)}&sentiment=${encodeURIComponent(k)}`,
      }));
      const resSegs = RESOLUTION_ORDER.map((k) => ({
        key: k,
        label: RESOLUTION_META[k].label,
        count: active.resCounts[k] || 0,
        color: RESOLUTION_META[k].color,
        href: `#/resolution?issue=${encodeURIComponent(active.key)}&state=${encodeURIComponent(k)}`,
      }));

      const sampleProposals = [];
      for (const r of active.resRows) {
        const list = r.proposals || r.proposed_solutions || [];
        if (list.length) {
          sampleProposals.push({
            postId: r.post_id,
            title: r.title,
            state: r.state,
            text: list[0].text,
          });
        }
        if (sampleProposals.length >= 2) break;
      }

      inspectorHtml = deck(`Deep dive: ${esc(pretty(active.label))}`, {
        id: "issue-inspector",
        className: viewState.mode === "cards" ? "sticky-inspector" : "",
        subtitle: "Aggregate polarity, resolution states, and representative synthetic proposals",
        action: `
          <a class="btn btn-primary btn-sm" href="#/explorer?issue=${encodeURIComponent(active.key)}">Explore ${num(active.posts)} threads</a>
        `,
        body: `
          <div style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:var(--space-md)">
            ${copyIdChip(active.key, { label: "key" })}
            <span class="tag tag-info">${num(active.actionable_recommendations)} actionable (${share(active.actionable_recommendations, active.comments)})</span>
            <span class="tag tag-positive">${num(active.reported_successes)} / ${num(active.posts)} reported success</span>
          </div>

          <div class="aside-section">
            <h3>Predicted sentiment mix</h3>
            ${stackedCompositionBar(sentSegs, { total: active.sentTotal, unit: "comments", ariaLabel: `${pretty(active.label)} sentiment mix` })}
          </div>

          <div class="aside-section">
            <h3>Resolution evidence states</h3>
            ${stackedCompositionBar(resSegs, { total: active.posts, unit: "threads", ariaLabel: `${pretty(active.label)} resolution states` })}
          </div>

          <div class="aside-section">
            <h3>Representative proposed solutions</h3>
            <div class="preview-comment-list">
              ${sampleProposals.map((p) => `
                <div class="preview-comment-item">
                  <p>“${esc(truncate(p.text, 180))}”</p>
                  <div class="topic-example-foot">
                    <a href="#/thread/${encodeURIComponent(p.postId)}">${esc(truncate(p.title, 48))}</a>
                    ${resolutionBadge(p.state)}
                  </div>
                </div>
              `).join("") || `<p class="muted">No proposals found.</p>`}
            </div>
          </div>

          <div class="aside-section">
            <h3>Supporting threads</h3>
            <div id="issue-threads-slot" class="preview-comment-list">
              <div class="muted">Loading matching synthetic discussions…</div>
            </div>
          </div>
        `,
      });
    }

    container.innerHTML = `
      ${callout(
        "Issue categories below are scenario labels from the synthetic generator (20 fictional threads, 160 comments per domain). Sentiment percentages reflect in-sample baseline predictions; resolution counts reflect explicit synthetic outcome reports only.",
        { tone: "info", title: "Data provenance:" }
      )}
      ${viewState.mode === "cards"
        ? `
          <div class="issues-studio-split">
            <section class="issues-canvas-col" aria-label="Scenario issue domains">
              ${toolbar}
              <div class="spacer-md"></div>
              ${cardsMarkup}
            </section>
            ${inspectorHtml}
          </div>
        `
        : `
          ${deck("Scenario issue comparison ledger", {
            subtitle: "Exact /api/demo/issues counts enriched with predicted negative polarity share from /api/demo/sentiment",
            body: `${toolbar}<div class="spacer-md"></div>${tableMarkup}`,
          })}
          ${inspectorHtml}
        `}
    `;

    const input = container.querySelector("#issues-filter-input");
    if (input) {
      input.addEventListener("input", (e) => {
        viewState.search = e.target.value;
        paint();
        const next = container.querySelector("#issues-filter-input");
        if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
      });
    }

    container.querySelectorAll("[data-issue-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        viewState.mode = btn.dataset.issueMode;
        paint();
      });
    });

    container.querySelectorAll("[data-issue-sort]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const col = btn.dataset.issueSort;
        if (viewState.sortBy === col) viewState.sortDir = viewState.sortDir === "desc" ? "asc" : "desc";
        else { viewState.sortBy = col; viewState.sortDir = col === "label" ? "asc" : "desc"; }
        paint();
      });
    });

    container.querySelectorAll("[data-select-issue]").forEach((el) => {
      el.addEventListener("click", () => {
        const key = el.dataset.selectIssue;
        if (key && key !== viewState.selectedKey) {
          viewState.selectedKey = key;
          paint();
        }
      });
    });

    if (active) {
      const slot = container.querySelector("#issue-threads-slot");
      if (slot) {
        try {
          const disc = await api.discussions({ issue: active.key, page: 1, page_size: 4 });
          if (viewState.selectedKey !== active.key) return;
          slot.innerHTML = (disc.items || []).map((t) => `
            <a class="ruled-row thread-mini-row" href="#/thread/${encodeURIComponent(t.id)}">
              <div class="spotlight-row-title">${esc(t.title)}</div>
              <div class="spotlight-row-foot">
                ${resolutionBadge(t.resolution_state)}
                <span class="num">${dateText(t.created_at)}</span>
              </div>
            </a>
          `).join("");
        } catch {
          slot.innerHTML = `<p class="muted">Could not load sample discussions.</p>`;
        }
      }
    }
  }

  await paint();
}
