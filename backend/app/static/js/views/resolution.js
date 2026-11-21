/**
 * View 8: Solutions & Resolution Evidence Auditor
 * Separates proposed solutions/advice from explicit first-person resolution outcomes.
 * Unlocks expandable per-thread scenario proposals, advice comments, and binary model probabilities.
 */
import { api } from "../api.js?v=5.6";
import { stackedCompositionBar, stackedMatrixRows } from "../charts.js?v=5.6";
import { badge, callout, copyIdChip, deck, emptyState, glossaryTip, resolutionBadge } from "../components.js?v=5.6";
import {
  RESOLUTION_META,
  RESOLUTION_ORDER,
  esc,
  issueLabelFromKey,
  num,
  percent,
  pretty,
  share,
  shortCommentLabel,
} from "../format.js?v=5.6";

const resState = {
  stateFilter: "all",
  issueFilter: "all",
  search: "",
  sortBy: "default",
  expandedIds: new Set(),
};

export async function renderResolution(container, ctx, params = {}) {
  const [resData, summary] = await Promise.all([
    api.resolution(),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
  ]);
  ctx.summary = summary;

  if (params.state && RESOLUTION_ORDER.includes(params.state)) resState.stateFilter = params.state;
  else if (params.state === "all") resState.stateFilter = "all";
  if (params.issue) resState.issueFilter = params.issue;

  const items = resData.items || [];
  const totalThreads = items.length;
  const countsByState = Object.fromEntries(RESOLUTION_ORDER.map((k) => [k, 0]));
  let totalProposedSolutions = 0;
  let totalAdviceComments = 0;
  let totalModelPositives = 0;

  const byIssue = {};
  for (const row of items) {
    if (countsByState[row.state] !== undefined) countsByState[row.state] += 1;
    const proposals = row.proposals || row.proposed_solutions || [];
    const advices = row.advice || row.advice_comments || [];
    const modelOut = row.solution_model_output || row.model_solution_predictions || [];
    const modelPos = modelOut.filter((m) => m.is_solution).length;
    totalProposedSolutions += proposals.length;
    totalAdviceComments += advices.length;
    totalModelPositives += modelPos;
    if (!byIssue[row.issue_key]) {
      byIssue[row.issue_key] = {
        key: row.issue_key,
        label: issueLabelFromKey(row.issue_key, summary),
        total: 0,
        counts: Object.fromEntries(RESOLUTION_ORDER.map((k) => [k, 0])),
      };
    }
    byIssue[row.issue_key].total += 1;
    if (byIssue[row.issue_key].counts[row.state] !== undefined) byIssue[row.issue_key].counts[row.state] += 1;
  }

  const issueList = Object.values(byIssue).sort((a, b) => a.label.localeCompare(b.label));

  function paint() {
    const q = resState.search.trim().toLowerCase();
    let filtered = items.filter((row) => {
      if (resState.stateFilter !== "all" && row.state !== resState.stateFilter) return false;
      if (resState.issueFilter !== "all" && row.issue_key !== resState.issueFilter) return false;
      if (!q) return true;
      const hay = [
        row.title, row.post_id, row.issue, row.issue_key, row.state, row.evidence_text || "",
        ...(row.proposals || []).map((s) => s.text),
        ...(row.advice || []).map((a) => a.text),
      ].join(" ").toLowerCase();
      return hay.includes(q);
    });

    if (resState.sortBy === "actionable") {
      filtered = [...filtered].sort((a, b) => (b.actionable_recommendations?.length || 0) - (a.actionable_recommendations?.length || 0));
    } else if (resState.sortBy === "evidence_first") {
      const rank = { reported_success: 0, reported_pending: 1, proposed_only: 2, insufficient_evidence: 3 };
      filtered = [...filtered].sort((a, b) => (rank[a.state] ?? 9) - (rank[b.state] ?? 9));
    }

    const stateCards = `
      <div class="resolution-state-grid" role="group" aria-label="Filter threads by resolution evidence state">
        ${RESOLUTION_ORDER.map((stateKey) => {
          const meta = RESOLUTION_META[stateKey];
          const count = countsByState[stateKey] || 0;
          const active = resState.stateFilter === stateKey;
          return `
            <button type="button" class="res-state-card anim-enter ${active ? "is-active" : ""}" data-res-state="${esc(stateKey)}" aria-pressed="${active}">
              <div class="res-state-top">
                <span class="swatch-inline" style="background:${meta.color}" aria-hidden="true"></span>
                <span class="res-state-label">${esc(meta.label)}</span>
              </div>
              <div class="res-state-nums">
                <strong class="num">${num(count)}</strong>
                <span class="num">${share(count, totalThreads)} of threads</span>
              </div>
              <span class="res-state-desc">${esc(meta.description)}</span>
            </button>
          `;
        }).join("")}
      </div>
    `;

    const funnelSegments = RESOLUTION_ORDER.map((k) => ({
      key: k,
      label: RESOLUTION_META[k].label,
      count: countsByState[k] || 0,
      color: RESOLUTION_META[k].color,
    }));

    const issueMatrixRows = issueList.map((iss) => ({
      key: iss.key,
      label: iss.label,
      total: iss.total,
      actionHref: `#/issues?selected=${encodeURIComponent(iss.key)}`,
      segments: RESOLUTION_ORDER.map((k) => ({
        key: k,
        label: RESOLUTION_META[k].label,
        short: RESOLUTION_META[k].label,
        count: iss.counts[k] || 0,
        color: RESOLUTION_META[k].color,
      })),
    }));

    const tableRows = filtered.map((row) => {
      const expanded = resState.expandedIds.has(row.post_id);
      const proposals = row.proposals || row.proposed_solutions || [];
      const advices = row.advice || row.advice_comments || [];
      const probs = row.solution_model_output || row.model_solution_predictions || [];
      const modelPos = probs.filter((p) => p.is_solution).length;

      const expandRow = expanded
        ? `
          <tr class="expand-detail-row">
            <td colspan="5">
              <div class="res-expand-grid">
                <div class="res-expand-col">
                  <h4>Predicted solution proposals (${proposals.length})</h4>
                  ${proposals.length
                    ? proposals.map((s) => `<div class="res-quote-item"><span class="num res-quote-id">${esc(shortCommentLabel(s.comment_id))} <code>solution</code></span><p>${esc(s.text)}</p></div>`).join("")
                    : `<p class="muted" style="font-size:var(--text-xs)">No predicted solution comments in this thread.</p>`}
                </div>
                <div class="res-expand-col">
                  <h4>Predicted advice comments (${advices.length})</h4>
                  ${advices.length
                    ? advices.map((a) => `<div class="res-quote-item"><span class="num res-quote-id">${esc(shortCommentLabel(a.comment_id))} <code>advice</code></span><p>${esc(a.text)}</p></div>`).join("")
                    : `<p class="muted" style="font-size:var(--text-xs)">No predicted advice comments in this thread.</p>`}
                </div>
                <div class="res-expand-col">
                  <h4>Binary solution model output (${modelPos} / ${probs.length} positive)</h4>
                  <div class="model-prob-chips">
                    ${probs.map((p) => badge(`${shortCommentLabel(p.comment_id)}: ${p.is_solution ? "yes" : "no"} (${percent(p.probability ?? p.solution_probability, 0)})`, p.is_solution ? "positive" : "neutral")).join("")}
                  </div>
                  <div class="spacer-sm"></div>
                  <a class="btn btn-primary btn-sm" href="#/thread/${encodeURIComponent(row.post_id)}">Inspect complete reply tree</a>
                </div>
              </div>
            </td>
          </tr>
        `
        : "";

      return `
        <tr class="evidence-row ${expanded ? "is-selected" : ""}">
          <td>
            <a class="table-primary-link" href="#/thread/${encodeURIComponent(row.post_id)}">${esc(row.title)}</a>
            <div style="margin-top:var(--space-2xs);display:flex;flex-wrap:wrap;gap:var(--space-xs);align-items:center">
              <span class="muted" style="font-size:var(--text-xs)">${esc(pretty(row.issue))}</span>
              ${copyIdChip(row.post_id)}
            </div>
          </td>
          <td>${resolutionBadge(row.state)}</td>
          <td class="evidence-quote-col">
            ${row.evidence_text
              ? `<span class="table-evidence-quote">“${esc(row.evidence_text)}”</span>
                 <span class="table-evidence-sub num">${esc(row.evidence_type || row.state)}, comment ${esc(shortCommentLabel(row.evidence_comment_id))}</span>`
              : `<span class="muted">No explicit outcome statement in thread (unconfirmed).</span>`}
          </td>
          <td class="num">
            <div class="stacked-num-cell">
              <strong>${num(proposals.length)} sol, ${num(advices.length)} adv</strong>
              <span class="muted">${num(modelPos)} model positive</span>
            </div>
          </td>
          <td class="num">
            <button type="button" class="btn btn-quiet btn-sm" data-toggle-expand="${esc(row.post_id)}" aria-expanded="${expanded}">
              ${expanded ? "Hide details" : "Inspect"}
            </button>
          </td>
        </tr>
        ${expandRow}
      `;
    }).join("");

    const auditContent = filtered.length
      ? `
        <div class="table-scroll">
          <table class="data-table evidence-table" aria-label="Thread-level solution proposals and explicit resolution evidence audit">
            <thead>
              <tr>
                <th scope="col">Discussion &amp; issue</th>
                <th scope="col">Evidence state</th>
                <th scope="col">Explicit outcome evidence</th>
                <th scope="col" class="num">Proposals vs. model</th>
                <th scope="col" class="num">Audit</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      `
      : emptyState({
          title: "No threads match these evidence filters",
          description: "Clear the search phrase or switch back to all 4 resolution states.",
          actionHtml: `<button type="button" class="btn btn-primary btn-sm" data-reset-res>Reset evidence filters</button>`,
        });

    container.innerHTML = `
      ${callout(
        `<strong>${glossaryTip("proposal_vs_resolution", "Proposed solution is not demonstrated resolution.")}</strong> Across ${num(totalThreads)} community threads, the corpus holds <strong>${num(totalProposedSolutions)}</strong> predicted solution proposals, <strong>${num(totalAdviceComments)}</strong> advice comments, and <strong>${num(totalModelPositives)}</strong> positive binary solution-model predictions, while <strong>${num(countsByState.reported_success)} threads (${share(countsByState.reported_success, totalThreads)})</strong> contain an explicit first-person resolution statement.`,
        { tone: "info", title: "Research integrity guard:" }
      )}

      ${stateCards}

      <div class="studio-grid-2">
        ${deck("Corpus resolution funnel", {
          subtitle: `Explicit first-person outcome reports compared against unconfirmed proposals (n = ${num(totalThreads)} threads)`,
          body: stackedCompositionBar(funnelSegments, { total: totalThreads, unit: "threads", ariaLabel: `Resolution states across ${totalThreads} threads` }),
        })}
        ${deck("Resolution states by issue domain", {
          subtitle: "Observed distribution of explicit follow-up states across each community issue domain",
          body: stackedMatrixRows(issueMatrixRows, {
            legendItems: RESOLUTION_ORDER.map((k) => ({ label: RESOLUTION_META[k].label, color: RESOLUTION_META[k].color })),
            unit: "threads",
          }),
        })}
      </div>

      ${deck("Thread-level proposal & outcome audit", {
        subtitle: "Expand any thread row to compare predicted solution proposals, advice comments, explicit evidence text, and binary model probabilities",
        action: resState.stateFilter !== "all" || resState.issueFilter !== "all" || resState.search
          ? `<button type="button" class="btn btn-quiet btn-sm" data-reset-res>Clear filters</button>`
          : "",
        body: `
          <div class="filter-dock" style="margin-bottom:var(--space-md)">
            <div class="filter-grid filter-grid-4">
              <div class="field">
                <label for="res-search">Search titles, evidence, or proposals</label>
                <input id="res-search" type="search" value="${esc(resState.search)}" placeholder="Search text or post ID (converted, waitlist, resolved)…">
              </div>
              <div class="field">
                <label for="res-state-sel">Evidence state</label>
                <select id="res-state-sel">
                  <option value="all" ${resState.stateFilter === "all" ? "selected" : ""}>All 4 states (${totalThreads})</option>
                  ${RESOLUTION_ORDER.map((k) => `<option value="${k}" ${resState.stateFilter === k ? "selected" : ""}>${esc(RESOLUTION_META[k].label)} (${countsByState[k]})</option>`).join("")}
                </select>
              </div>
              <div class="field">
                <label for="res-issue-sel">Issue domain</label>
                <select id="res-issue-sel">
                  <option value="all" ${resState.issueFilter === "all" ? "selected" : ""}>All issue domains</option>
                  ${issueList.map((i) => `<option value="${esc(i.key)}" ${resState.issueFilter === i.key ? "selected" : ""}>${esc(i.label)}</option>`).join("")}
                </select>
              </div>
              <div class="field">
                <label for="res-sort-sel">Sort rows by</label>
                <select id="res-sort-sel">
                  <option value="default" ${resState.sortBy === "default" ? "selected" : ""}>Corpus order</option>
                  <option value="evidence_first" ${resState.sortBy === "evidence_first" ? "selected" : ""}>Explicit evidence first</option>
                  <option value="actionable" ${resState.sortBy === "actionable" ? "selected" : ""}>Most actionable proposals</option>
                </select>
              </div>
            </div>
          </div>
          <div class="results-bar" style="margin-bottom:var(--space-sm)">
            <span>Showing <strong class="num">${num(filtered.length)}</strong> of <strong class="num">${num(totalThreads)}</strong> threads</span>
          </div>
          ${auditContent}
        `,
      })}
    `;

    container.querySelectorAll("[data-res-state]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s = btn.dataset.resState;
        resState.stateFilter = resState.stateFilter === s ? "all" : s;
        paint();
      });
    });

    const searchInput = container.querySelector("#res-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        resState.search = e.target.value;
        paint();
        const next = container.querySelector("#res-search");
        if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
      });
    }
    container.querySelector("#res-state-sel")?.addEventListener("change", (e) => { resState.stateFilter = e.target.value; paint(); });
    container.querySelector("#res-issue-sel")?.addEventListener("change", (e) => { resState.issueFilter = e.target.value; paint(); });
    container.querySelector("#res-sort-sel")?.addEventListener("change", (e) => { resState.sortBy = e.target.value; paint(); });
    container.querySelectorAll("[data-toggle-expand]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.toggleExpand;
        if (resState.expandedIds.has(id)) resState.expandedIds.delete(id);
        else resState.expandedIds.add(id);
        paint();
      });
    });
    container.querySelectorAll("[data-reset-res]").forEach((btn) => {
      btn.addEventListener("click", () => {
        resState.stateFilter = "all";
        resState.issueFilter = "all";
        resState.search = "";
        resState.sortBy = "default";
        paint();
      });
    });
  }

  paint();
}
