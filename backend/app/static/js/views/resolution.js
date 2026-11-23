/**
 * View 8: Solutions & Resolution Evidence Studio
 * Unifies the V6 Explainable Resolution Engine (default) with a preserved Legacy V5 Baseline toggle,
 * interactive Contradiction & Evidence Triage Queue, paginated 365-thread audit table with lazy
 * detail expansion, URL state synchronization, and one-click CSV export.
 */
import { api } from "../api.js?v=6.2";
import { stackedCompositionBar, stackedMatrixRows } from "../charts.js?v=6.2";
import { badge, callout, copyIdChip, deck, emptyState, glossaryTip, resolutionBadge } from "../components.js?v=6.2";
import { downloadCsv } from "../export.js?v=6.2";
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
} from "../format.js?v=6.2";

const PAGE_SIZE = 25;

const resState = {
  engine: "v6", // v6 | v5
  triageFilter: "contradictory", // contradictory | success | unresolved | all
  stateFilter: "all",
  issueFilter: "all",
  search: "",
  sortBy: "evidence_first",
  page: 1,
  clusterIssue: "administrative_communication",
  expandedIds: new Set(),
  detailCache: new Map(),
  loadingIds: new Set(),
};

function verdictTone(v) {
  if (v === "Resolved") return "positive";
  if (v === "Partially Resolved") return "warning";
  if (v === "Unresolved") return "negative";
  return "neutral";
}

function renderV6ExplainableSection(resV2, mapV2) {
  if (!resV2 || !resV2.overall_summary) return "";
  const ov = resV2.overall_summary;
  const vd = ov.verdict_distribution || {};
  const sd = ov.state_distribution || {};
  const evRes = resV2.evaluation_benchmark_human_reviewed || {};
  const evMap = mapV2?.evaluation_benchmark_human_reviewed || {};
  const byIss = resV2.by_issue || [];
  const allTrails = resV2.featured_thread_trails || [];

  const counts = {
    contradictory: allTrails.filter((t) => (t.signals?.contradictory_followups_count || 0) > 0).length,
    success: allTrails.filter((t) => t.resolution_state === "reported_success").length,
    unresolved: allTrails.filter((t) => t.verdict === "Unresolved").length,
    all: allTrails.length,
  };

  const filteredTrails = allTrails.filter((t) => {
    if (resState.triageFilter === "contradictory") return (t.signals?.contradictory_followups_count || 0) > 0;
    if (resState.triageFilter === "success") return t.resolution_state === "reported_success";
    if (resState.triageFilter === "unresolved") return t.verdict === "Unresolved";
    return true;
  });

  const statStrip = `
    <div class="stat-strip-mini" role="region" aria-label="Explainable resolution engine verdict totals">
      <div class="stat-mini-cell">
        <span class="stat-mini-label">Resolved</span>
        <strong class="stat-mini-val is-positive num">${num(vd.Resolved || 0)}</strong>
        <span class="stat-mini-sub num">${num(sd.reported_success || 0)} confirmed success</span>
      </div>
      <div class="stat-mini-cell">
        <span class="stat-mini-label">Partially resolved</span>
        <strong class="stat-mini-val is-warning num">${num(vd["Partially Resolved"] || 0)}</strong>
        <span class="stat-mini-sub num">${num(sd.reported_pending || 0)} pending follow-up</span>
      </div>
      <div class="stat-mini-cell">
        <span class="stat-mini-label">Unresolved</span>
        <strong class="stat-mini-val is-negative num">${num(vd.Unresolved || 0)}</strong>
        <span class="stat-mini-sub num">${num(ov.threads_with_contradictory_followups || 0)} contradictory threads</span>
      </div>
      <div class="stat-mini-cell">
        <span class="stat-mini-label">Insufficient evidence</span>
        <strong class="stat-mini-val num">${num(vd["Insufficient Evidence"] || 0)}</strong>
        <span class="stat-mini-sub num">${num(sd.proposed_only || 0)} proposed-only</span>
      </div>
      <div class="stat-mini-cell">
        <span class="stat-mini-label">Audit precision</span>
        <strong class="stat-mini-val is-info num">${percent(evRes.false_resolution_claim_rate, 1)}</strong>
        <span class="stat-mini-sub num">false claims · Unres F1 ${percent(evRes.unresolved_detection?.f1, 1)}</span>
      </div>
    </div>
  `;

  const issRows = byIss.map((r) => `
    <tr>
      <td><a class="table-primary-link" href="#/issues?selected=${encodeURIComponent(r.issue_id)}">${esc(pretty(r.issue_id))}</a></td>
      <td>${badge(r.verdict, verdictTone(r.verdict))}</td>
      <td class="num">${num(r.threads)}</td>
      <td class="num">${num(r.matched_solutions)}</td>
      <td class="num">${num(r.state_distribution?.reported_success || 0)}</td>
      <td class="num">${num(r.state_distribution?.reported_pending || 0)}</td>
      <td class="num">${num(r.state_distribution?.proposed_only || 0)}</td>
      <td class="num"><strong>${num(r.contradictory_threads || 0)}</strong></td>
      <td class="num">${Number(r.mean_resolution_score || 0).toFixed(2)}</td>
    </tr>
  `).join("");

  const trailCards = filteredTrails.slice(0, 14).map((t) => {
    const firstContra = (t.contradictory_evidence || [])[0];
    const firstSuccess = (t.success_evidence || [])[0];
    const jumpId = firstContra?.comment_id || firstSuccess?.comment_id || t.proposed_solution?.comment_id || "";
    const threadHref = jumpId
      ? `#/thread/${encodeURIComponent(t.post_id)}?comment=${encodeURIComponent(jumpId)}`
      : `#/thread/${encodeURIComponent(t.post_id)}`;
    return `
      <article class="surface-well">
        <div class="surface-well-head">
          <a class="surface-well-link" href="${threadHref}">${esc(t.post_title)}</a>
          <div class="tag-row">
            ${badge(t.verdict, verdictTone(t.verdict))}
            ${resolutionBadge(t.resolution_state)}
            <span class="tag tag-neutral num">Score ${Number(t.resolution_score || 0).toFixed(2)} · Conf ${percent(t.confidence, 0)}</span>
          </div>
        </div>
        <div class="muted text-xs">
          <strong>Chain:</strong> Issue <code>${esc(t.issue_id)}</code> &rarr; ${num(t.signals?.solution_oriented_comments || 0)} matched solution(s) (best rel ${Number(t.signals?.best_solution_relevance || 0).toFixed(2)}) &rarr; ${num(t.signals?.contradictory_followups_count || 0)} contradictory signal(s) &rarr; ${num(t.signals?.discussion_span_hours || 0)}h span
        </div>
        ${t.proposed_solution ? `<div class="text-xs"><strong>Top matched solution:</strong> “${esc((t.proposed_solution.text || "").slice(0, 180))}” <span class="muted">(${esc(shortCommentLabel(t.proposed_solution.comment_id))}, score ${Number(t.proposed_solution.matching_score || 0).toFixed(2)})</span></div>` : ""}
        ${firstContra ? `<div class="text-xs text-negative"><strong>Contradictory / failed follow-up (${esc(firstContra.signal)}):</strong> “${esc((firstContra.text || "").slice(0, 180))}”</div>` : ""}
        ${firstSuccess ? `<div class="text-xs text-positive"><strong>Confirmed outcome (${esc(firstSuccess.signal)}):</strong> “${esc((firstSuccess.text || "").slice(0, 180))}”</div>` : ""}
        <div class="surface-well-head text-xs muted">
          <span>${esc(t.explanation)}</span>
          <a class="row-open-link" href="${threadHref}">Jump to evidence &rarr;</a>
        </div>
      </article>
    `;
  }).join("");

  const queuePills = [
    ["contradictory", `Contradictory (${counts.contradictory})`],
    ["success", `Confirmed success (${counts.success})`],
    ["unresolved", `Unresolved (${counts.unresolved})`],
    ["all", `All featured (${counts.all})`],
  ].map(([k, lbl]) => `
    <button type="button" class="pill-btn ${resState.triageFilter === k ? "is-active" : ""}" data-triage-filter="${k}">${lbl}</button>
  `).join("");

  return `
    ${deck("Explainable issue → solution → outcome engine", {
      subtitle: `Contextual Multilingual-E5 Mapper (${num(mapV2?.summary?.total_linked_solutions || 0)} links, Pairwise F1 ${percent(evMap?.v6_contextual_embedding?.pairwise_f1, 1)} vs. TF-IDF ${percent(evMap?.v5_tfidf_baseline?.pairwise_f1, 1)}) + 8-signal Resolution Engine (0.0% false claims on n=${num(evRes.n_threads || 40)} human audit)`,
      action: `<button type="button" class="btn btn-quiet btn-sm" id="export-v6-verdicts-csv">Export verdicts CSV</button>`,
      body: `
        ${statStrip}
        <div class="table-scroll">
          <table class="data-table text-xs" aria-label="Explainable per-issue resolution verdicts">
            <thead>
              <tr>
                <th scope="col">Issue domain</th>
                <th scope="col">Explainable verdict</th>
                <th scope="col" class="num">Threads</th>
                <th scope="col" class="num">E5 Solutions</th>
                <th scope="col" class="num">Confirmed</th>
                <th scope="col" class="num">Pending</th>
                <th scope="col" class="num">Proposed</th>
                <th scope="col" class="num">Contradictory</th>
                <th scope="col" class="num">Mean score</th>
              </tr>
            </thead>
            <tbody>${issRows}</tbody>
          </table>
        </div>
      `,
    })}
    ${deck("Contradiction &amp; outcome evidence triage queue", {
      subtitle: "Auditable end-to-end chains linking issue context, top matched peer solution, and contradictory or confirmed follow-up comments",
      action: `<div class="filter-pills" role="group" aria-label="Filter evidence triage queue">${queuePills}</div>`,
      body: filteredTrails.length
        ? `<div class="grid-stack-xs">${trailCards}</div>`
        : emptyState({ title: "No featured chains in this slice", description: "Select another triage filter above." }),
    })}
  `;
}

function renderLegacyV5Section(matrix, clustersData, linkerSummary, funnelSegments, issueMatrixRows, totalThreads) {
  const rows = matrix?.rows || [];
  const crit = matrix?.criteria_definition || {};
  const statusTone = (s) => (s === "Resolved" ? "positive" : s === "Partially Resolved" ? "warning" : "negative");
  const strengthTone = (s) => (s === "High" ? "positive" : s === "Moderate" ? "info" : "neutral");

  const trs = rows.map((r) => `
    <tr>
      <td><strong>${esc(r.issue_label)}</strong></td>
      <td class="num"><strong>${num(r.comment_mentions)}</strong><div class="muted text-xs">${esc(r.volume_tier)}</div></td>
      <td class="num"><strong>${percent(r.solution_density, 1)}</strong><div class="muted text-xs">${num(r.solutions_detected)} replies</div></td>
      <td class="num"><strong>${num(r.linked_solutions)}</strong><div class="muted text-xs">${percent(r.linked_thread_coverage, 0)} thr</div></td>
      <td class="num"><strong>${num(r.verified_outcomes)}</strong><div class="muted text-xs">${percent(r.verified_outcome_rate, 1)}</div></td>
      <td>${badge(r.resolution_strength, strengthTone(r.resolution_strength))}</td>
      <td>${badge(r.status, statusTone(r.status))}</td>
    </tr>
  `).join("");

  const rulesHtml = (crit.status_rules || []).map((rule) => `
    <li><strong>${esc(rule.status)}:</strong> <code>${esc(rule.formula)}</code> — ${esc(rule.interpretation)}</li>
  `).join("");

  const byIssue = clustersData?.by_issue || {};
  const issueKeys = Object.keys(byIssue);
  if (issueKeys.length && !byIssue[resState.clusterIssue]) resState.clusterIssue = issueKeys[0];
  const active = byIssue[resState.clusterIssue];
  const clusters = active?.clusters || [];
  const lSum = linkerSummary || clustersData?.linker_summary || {};

  const pills = issueKeys.map((k) => {
    const info = byIssue[k];
    const isAct = k === resState.clusterIssue;
    return `<button type="button" class="btn ${isAct ? "btn-primary" : "btn-quiet"} btn-sm" data-cluster-issue="${esc(k)}">${esc(info.issue_label)} <span class="num">(${num(info.linked_solution_count ?? info.actionable_comment_count ?? 0)})</span></button>`;
  }).join("");

  const cards = clusters.length
    ? clusters.map((c, idx) => `
        <div class="surface-well">
          <div class="surface-well-head">
            <strong>${idx + 1}. ${esc(c.theme)}</strong>
            <span class="num text-xs"><strong>${num(c.count)} linked (${percent(c.share, 0)})</strong>${c.mean_link_score ? ` · blend ${Number(c.mean_link_score).toFixed(2)}` : ""}</span>
          </div>
          ${(c.top_terms || []).length ? `<div class="muted text-xs">Keywords: ${c.top_terms.map((t) => `<code>${esc(t)}</code>`).join(" ")}</div>` : ""}
          ${(c.examples || []).slice(0, 1).map((ex) => `
            <div class="text-xs">
              “${esc(ex.text)}”
              <span class="muted"> — <a href="#/thread/${encodeURIComponent(ex.post_id)}">${esc(shortCommentLabel(ex.comment_id))}</a>${ex.cosine_similarity !== undefined ? ` (cos ${Number(ex.cosine_similarity).toFixed(2)})` : ""}</span>
            </div>
          `).join("")}
        </div>
      `).join("")
    : `<p class="muted text-xs">No cosine-linked solution clusters found for this issue.</p>`;

  return `
    ${callout(
      "Comparing against the preserved V5 lexical/TF-IDF baseline (17 heuristic reported successes before filtering advisor credentials & question guards). Switch back to <strong>V6 Explainable Engine</strong> for audited zero-false-claim verdicts.",
      { tone: "warning", title: "Legacy V5 Baseline Mode:" }
    )}
    <div class="studio-grid-2">
      ${deck("Legacy V5 resolution funnel", {
        subtitle: `Unfiltered regex follow-up labels across n = ${num(totalThreads)} threads`,
        body: stackedCompositionBar(funnelSegments, { total: totalThreads, unit: "threads", ariaLabel: `Legacy resolution states across ${totalThreads} threads` }),
      })}
      ${deck("Legacy states by issue domain", {
        subtitle: "Historical V5 follow-up distribution per issue category",
        body: stackedMatrixRows(issueMatrixRows, {
          legendItems: RESOLUTION_ORDER.map((k) => ({ label: RESOLUTION_META[k].label, color: RESOLUTION_META[k].color })),
          unit: "threads",
        }),
      })}
    </div>
    ${rows.length ? deck("Legacy quantitative resolution matrix (TF-IDF baseline)", {
      subtitle: "Baseline status derived from XLM-R solution density, TF-IDF cosine-linked coverage, and V5 outcome flags",
      body: `
        <div class="table-scroll">
          <table class="data-table text-xs" aria-label="Legacy quantitative per-issue resolution status matrix">
            <thead>
              <tr>
                <th scope="col">Issue domain</th>
                <th scope="col" class="num">Volume</th>
                <th scope="col" class="num">Solutions</th>
                <th scope="col" class="num">Linked</th>
                <th scope="col" class="num">Verified</th>
                <th scope="col">Strength</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>${trs}</tbody>
          </table>
        </div>
        ${rulesHtml ? `
          <details class="evidence-details">
            <summary>Inspect quantitative status criteria formulas</summary>
            <ul class="muted text-xs">${rulesHtml}</ul>
          </details>
        ` : ""}
      `,
    }) : ""}
    ${issueKeys.length ? deck("TF-IDF lexical solution themes (baseline comparator)", {
      subtitle: lSum.total_asserted_links
        ? `Same-post TF-IDF cosine linker (threshold ${lSum.link_threshold ?? 0.45}): ${num(lSum.total_asserted_links)} asserted links across ${num(lSum.threads_with_linked_solutions)} threads`
        : "Ranked baseline community solution themes per issue domain",
      body: `
        <div class="tag-row tag-row-md">${pills}</div>
        <div class="grid-stack-sm">${cards}</div>
      `,
    }) : ""}
  `;
}

export async function renderResolution(container, ctx, params = {}) {
  const [resData, summary] = await Promise.all([
    api.resolution({ compact: true }),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
  ]);
  ctx.summary = summary;

  if (params.engine === "v5" || params.engine === "v6") resState.engine = params.engine;
  if (params.state && RESOLUTION_ORDER.includes(params.state)) resState.stateFilter = params.state;
  else if (params.state === "all") resState.stateFilter = "all";
  if (params.issue) resState.issueFilter = params.issue;

  const items = resData.items || [];
  const totalThreads = items.length;
  const v2Overall = resData.explainable_resolution_v2?.overall_summary || {};
  const v2States = v2Overall.state_distribution || {};
  const hasV2 = Number(v2Overall.total_threads || 0) > 0;

  const legacyCountsByState = Object.fromEntries(RESOLUTION_ORDER.map((k) => [k, 0]));
  let totalProposedSolutions = 0;
  let totalAdviceComments = 0;
  let totalModelPositives = 0;

  const byIssue = {};
  for (const row of items) {
    if (legacyCountsByState[row.state] !== undefined) legacyCountsByState[row.state] += 1;
    totalProposedSolutions += Number(row.proposal_count ?? (row.proposals?.length || 0));
    totalAdviceComments += Number(row.advice_count ?? (row.advice?.length || 0));
    totalModelPositives += Number(row.model_positive_count ?? 0);
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

  async function ensureThreadDetail(postId) {
    if (resState.detailCache.has(postId) || resState.loadingIds.has(postId)) return;
    resState.loadingIds.add(postId);
    paint();
    try {
      const detail = await api.discussionDetail(postId);
      resState.detailCache.set(postId, detail);
    } catch (err) {
      resState.detailCache.set(postId, { error: err?.message || "Failed to load thread details." });
    } finally {
      resState.loadingIds.delete(postId);
      paint();
    }
  }

  function renderExpandedBody(row) {
    if (resState.loadingIds.has(row.post_id)) {
      return `<div class="muted text-xs">Loading proposal, advice, and model probability breakdown…</div>`;
    }
    const detail = resState.detailCache.get(row.post_id);
    if (detail?.error) {
      return `<div class="muted text-xs">${esc(detail.error)}</div>`;
    }
    const comments = detail?.comments || [];
    const proposals = comments.filter((c) => (c.predictions?.response?.response_type || c.response_type) === "solution");
    const advices = comments.filter((c) => (c.predictions?.response?.response_type || c.response_type) === "advice");
    const probs = comments
      .filter((c) => c.predictions?.solution)
      .map((c) => ({
        comment_id: c.reddit_id,
        is_solution: Boolean(c.predictions.solution.is_solution),
        probability: c.predictions.solution.solution_probability,
      }));
    const modelPos = probs.filter((p) => p.is_solution).length;

    return `
      <div class="res-expand-grid">
        <div class="res-expand-col">
          <h4>Predicted solution proposals (${proposals.length})</h4>
          ${proposals.length
            ? proposals.slice(0, 6).map((s) => `<div class="res-quote-item"><span class="num res-quote-id">${esc(shortCommentLabel(s.reddit_id))} <code>solution</code></span><p>${esc(s.body)}</p></div>`).join("")
            : `<p class="muted text-xs">No predicted solution comments in this thread.</p>`}
        </div>
        <div class="res-expand-col">
          <h4>Predicted advice comments (${advices.length})</h4>
          ${advices.length
            ? advices.slice(0, 6).map((a) => `<div class="res-quote-item"><span class="num res-quote-id">${esc(shortCommentLabel(a.reddit_id))} <code>advice</code></span><p>${esc(a.body)}</p></div>`).join("")
            : `<p class="muted text-xs">No predicted advice comments in this thread.</p>`}
        </div>
        <div class="res-expand-col">
          <h4>Binary solution model output (${modelPos} / ${probs.length} positive)</h4>
          <div class="model-prob-chips">
            ${probs.slice(0, 12).map((p) => badge(`${shortCommentLabel(p.comment_id)}: ${p.is_solution ? "yes" : "no"} (${percent(p.probability, 0)})`, p.is_solution ? "positive" : "neutral")).join("")}
          </div>
          <div class="spacer-sm"></div>
          <a class="btn btn-primary btn-sm" href="#/thread/${encodeURIComponent(row.post_id)}">Inspect complete reply tree</a>
        </div>
      </div>
    `;
  }

  function paint() {
    const activeCountsByState = resState.engine === "v6" && hasV2
      ? Object.fromEntries(RESOLUTION_ORDER.map((k) => [k, Number(v2States[k] || 0)]))
      : legacyCountsByState;

    const q = resState.search.trim().toLowerCase();
    let filtered = items.filter((row) => {
      const effectiveState = resState.engine === "v6" && row.v6_resolution_state ? row.v6_resolution_state : row.state;
      if (resState.stateFilter !== "all" && effectiveState !== resState.stateFilter) return false;
      if (resState.issueFilter !== "all" && row.issue_key !== resState.issueFilter) return false;
      if (!q) return true;
      const hay = [row.title, row.post_id, row.issue, row.issue_key, effectiveState, row.v6_verdict || "", row.evidence_text || ""].join(" ").toLowerCase();
      return hay.includes(q);
    });

    if (resState.sortBy === "actionable") {
      filtered = [...filtered].sort((a, b) => Number(b.model_positive_count || 0) - Number(a.model_positive_count || 0));
    } else if (resState.sortBy === "contradictions") {
      filtered = [...filtered].sort((a, b) => Number(b.v6_contradictory_count || 0) - Number(a.v6_contradictory_count || 0));
    } else if (resState.sortBy === "evidence_first") {
      const rank = { reported_success: 0, reported_pending: 1, proposed_only: 2, insufficient_evidence: 3 };
      filtered = [...filtered].sort((a, b) => {
        const sa = resState.engine === "v6" && a.v6_resolution_state ? a.v6_resolution_state : a.state;
        const sb = resState.engine === "v6" && b.v6_resolution_state ? b.v6_resolution_state : b.state;
        return (rank[sa] ?? 9) - (rank[sb] ?? 9);
      });
    }

    const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    if (resState.page > totalPages) resState.page = totalPages;
    const startIdx = (resState.page - 1) * PAGE_SIZE;
    const pageRows = filtered.slice(startIdx, startIdx + PAGE_SIZE);

    const engineSwitcher = `
      <div class="method-bar">
        <div class="tag-row">
          <span class="method-bar-label">Resolution engine</span>
          <div class="segmented" role="group" aria-label="Select resolution analysis engine">
            <button type="button" class="seg-btn ${resState.engine === "v6" ? "is-active" : ""}" data-res-engine="v6">V6 Explainable Engine (Recommended)</button>
            <button type="button" class="seg-btn ${resState.engine === "v5" ? "is-active" : ""}" data-res-engine="v5">Legacy V5 Baseline (TF-IDF)</button>
          </div>
        </div>
        <div class="tag-row">
          ${badge(resState.engine === "v6" ? "Active: 8-Signal Audited Engine (0% False Claims)" : "Active: Historical V5 Baseline", resState.engine === "v6" ? "positive" : "warning")}
        </div>
      </div>
    `;

    const stateCards = `
      <div class="resolution-state-grid" role="group" aria-label="Filter threads by resolution evidence state">
        ${RESOLUTION_ORDER.map((stateKey) => {
          const meta = RESOLUTION_META[stateKey];
          const count = activeCountsByState[stateKey] || 0;
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
      count: activeCountsByState[k] || 0,
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

    const tableRows = pageRows.map((row) => {
      const expanded = resState.expandedIds.has(row.post_id);
      const effectiveState = resState.engine === "v6" && row.v6_resolution_state ? row.v6_resolution_state : row.state;
      const expandRow = expanded
        ? `<tr class="expand-detail-row"><td colspan="5">${renderExpandedBody(row)}</td></tr>`
        : "";

      return `
        <tr class="evidence-row ${expanded ? "is-selected" : ""}">
          <td>
            <a class="table-primary-link" href="#/thread/${encodeURIComponent(row.post_id)}">${esc(row.title)}</a>
            <div class="tag-row" style="margin-top:4px">
              <span class="muted text-xs">${esc(pretty(row.issue))}</span>
              ${copyIdChip(row.post_id)}
              ${row.v6_contradictory_count > 0 ? badge(`${row.v6_contradictory_count} contradictory`, "negative") : ""}
            </div>
          </td>
          <td>
            <div class="tag-row">
              ${resolutionBadge(effectiveState)}
              ${row.v6_verdict ? badge(row.v6_verdict, verdictTone(row.v6_verdict)) : ""}
            </div>
          </td>
          <td class="evidence-quote-col">
            ${row.evidence_text
              ? `<span class="table-evidence-quote">“${esc(row.evidence_text)}”</span>
                 <span class="table-evidence-sub num">${esc(row.evidence_type || row.state)}, comment ${esc(shortCommentLabel(row.evidence_comment_id))}</span>`
              : `<span class="muted">No explicit outcome statement in thread (unconfirmed).</span>`}
          </td>
          <td class="num">
            <div class="stacked-num-cell">
              <strong>${num(row.proposal_count || 0)} sol, ${num(row.advice_count || 0)} adv</strong>
              <span class="muted">${num(row.model_positive_count || 0)} model positive</span>
            </div>
          </td>
          <td class="num">
            <button type="button" class="btn btn-quiet btn-sm" data-toggle-expand="${esc(row.post_id)}" aria-expanded="${expanded}">
              ${expanded ? "Hide" : "Inspect"}
            </button>
          </td>
        </tr>
        ${expandRow}
      `;
    }).join("");

    const paginationBar = filtered.length > PAGE_SIZE
      ? `
        <div class="pagination-bar" role="navigation" aria-label="Audit table pagination">
          <span class="num">Rows ${num(startIdx + 1)}–${num(Math.min(startIdx + PAGE_SIZE, filtered.length))} of ${num(filtered.length)} (Page ${resState.page} of ${totalPages})</span>
          <div class="tag-row">
            <button type="button" class="btn btn-quiet btn-sm" data-res-page="${resState.page - 1}" ${resState.page <= 1 ? "disabled" : ""}>Previous</button>
            <button type="button" class="btn btn-quiet btn-sm" data-res-page="${resState.page + 1}" ${resState.page >= totalPages ? "disabled" : ""}>Next</button>
          </div>
        </div>
      `
      : "";

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
        ${paginationBar}
      `
      : emptyState({
          title: "No threads match these evidence filters",
          description: "Clear the search phrase or switch back to all 4 resolution states.",
          actionHtml: `<button type="button" class="btn btn-primary btn-sm" data-reset-res>Reset evidence filters</button>`,
        });

    const headlineConfirmed = activeCountsByState.reported_success || 0;
    container.innerHTML = `
      ${engineSwitcher}
      ${callout(
        `<strong>${glossaryTip("proposal_vs_resolution", "Proposed solution is not demonstrated resolution.")}</strong> Across ${num(totalThreads)} community threads, the corpus holds <strong>${num(totalProposedSolutions)}</strong> solution-role replies, <strong>${num(totalAdviceComments)}</strong> advice replies, and <strong>${num(totalModelPositives)}</strong> binary solution-positive predictions, while <strong>${num(headlineConfirmed)} threads (${share(headlineConfirmed, totalThreads)})</strong> contain verified first-person outcome confirmation.`,
        { tone: "info", title: "Research integrity guard:" }
      )}

      ${stateCards}

      ${resState.engine === "v6"
        ? renderV6ExplainableSection(resData.explainable_resolution_v2, resData.issue_solution_map_v2)
        : renderLegacyV5Section(resData.unresolved_matrix, resData.issue_solution_clusters, resData.linker_summary, funnelSegments, issueMatrixRows, totalThreads)}

      ${deck("Thread-level proposal &amp; outcome audit", {
        subtitle: "Paginated 365-thread evidence ledger — expand any row to lazily inspect predicted solutions, advice comments, and per-comment model probabilities",
        action: `
          <div class="export-bar">
            <button type="button" class="btn btn-quiet btn-sm" id="export-res-audit-csv">Export CSV (${num(filtered.length)})</button>
            ${resState.stateFilter !== "all" || resState.issueFilter !== "all" || resState.search
              ? `<button type="button" class="btn btn-quiet btn-sm" data-reset-res>Clear filters</button>`
              : ""}
          </div>
        `,
        body: `
          <div class="filter-dock" style="margin-bottom:var(--space-md)">
            <div class="filter-grid filter-grid-4">
              <div class="field">
                <label for="res-search">Search titles, evidence, or post ID</label>
                <input id="res-search" type="search" value="${esc(resState.search)}" placeholder="Search text or post ID (converted, waitlist, resolved)…">
              </div>
              <div class="field">
                <label for="res-state-sel">Evidence state</label>
                <select id="res-state-sel">
                  <option value="all" ${resState.stateFilter === "all" ? "selected" : ""}>All 4 states (${totalThreads})</option>
                  ${RESOLUTION_ORDER.map((k) => `<option value="${k}" ${resState.stateFilter === k ? "selected" : ""}>${esc(RESOLUTION_META[k].label)} (${activeCountsByState[k]})</option>`).join("")}
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
                  <option value="evidence_first" ${resState.sortBy === "evidence_first" ? "selected" : ""}>Explicit evidence first</option>
                  <option value="contradictions" ${resState.sortBy === "contradictions" ? "selected" : ""}>Contradictory follow-ups first</option>
                  <option value="actionable" ${resState.sortBy === "actionable" ? "selected" : ""}>Most solution-positive replies</option>
                  <option value="default" ${resState.sortBy === "default" ? "selected" : ""}>Corpus order</option>
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

    container.querySelectorAll("[data-res-engine]").forEach((btn) => {
      btn.addEventListener("click", () => {
        resState.engine = btn.dataset.resEngine;
        resState.page = 1;
        ctx.syncQueryParams?.({ engine: resState.engine, state: resState.stateFilter, issue: resState.issueFilter });
        paint();
      });
    });

    container.querySelectorAll("[data-triage-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        resState.triageFilter = btn.dataset.triageFilter;
        paint();
      });
    });

    container.querySelectorAll("[data-cluster-issue]").forEach((btn) => {
      btn.addEventListener("click", () => {
        resState.clusterIssue = btn.dataset.clusterIssue;
        paint();
      });
    });

    container.querySelectorAll("[data-res-state]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s = btn.dataset.resState;
        resState.stateFilter = resState.stateFilter === s ? "all" : s;
        resState.page = 1;
        ctx.syncQueryParams?.({ engine: resState.engine, state: resState.stateFilter, issue: resState.issueFilter });
        paint();
      });
    });

    const searchInput = container.querySelector("#res-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        resState.search = e.target.value;
        resState.page = 1;
        paint();
        const next = container.querySelector("#res-search");
        if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
      });
    }
    container.querySelector("#res-state-sel")?.addEventListener("change", (e) => {
      resState.stateFilter = e.target.value;
      resState.page = 1;
      ctx.syncQueryParams?.({ engine: resState.engine, state: resState.stateFilter, issue: resState.issueFilter });
      paint();
    });
    container.querySelector("#res-issue-sel")?.addEventListener("change", (e) => {
      resState.issueFilter = e.target.value;
      resState.page = 1;
      ctx.syncQueryParams?.({ engine: resState.engine, state: resState.stateFilter, issue: resState.issueFilter });
      paint();
    });
    container.querySelector("#res-sort-sel")?.addEventListener("change", (e) => {
      resState.sortBy = e.target.value;
      resState.page = 1;
      paint();
    });
    container.querySelectorAll("[data-res-page]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = Number(btn.dataset.resPage);
        if (target >= 1 && target <= totalPages) {
          resState.page = target;
          paint();
        }
      });
    });
    container.querySelectorAll("[data-toggle-expand]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.toggleExpand;
        if (resState.expandedIds.has(id)) {
          resState.expandedIds.delete(id);
          paint();
        } else {
          resState.expandedIds.add(id);
          ensureThreadDetail(id);
        }
      });
    });
    container.querySelectorAll("[data-reset-res]").forEach((btn) => {
      btn.addEventListener("click", () => {
        resState.stateFilter = "all";
        resState.issueFilter = "all";
        resState.search = "";
        resState.sortBy = "evidence_first";
        resState.page = 1;
        ctx.syncQueryParams?.({ engine: resState.engine, state: null, issue: null });
        paint();
      });
    });

    container.querySelector("#export-v6-verdicts-csv")?.addEventListener("click", () => {
      const rows = resData.explainable_resolution_v2?.by_issue || [];
      const ok = downloadCsv(
        rows,
        [
          { key: "issue_id", label: "Issue ID" },
          { key: "verdict", label: "Explainable Verdict" },
          { key: "threads", label: "Threads" },
          { key: "matched_solutions", label: "Matched E5 Solutions" },
          { key: "contradictory_threads", label: "Contradictory Threads" },
          { key: "mean_resolution_score", label: "Mean Resolution Score" },
        ],
        "rci_v6_issue_resolution_verdicts.csv"
      );
      if (ok) ctx.showToast?.("Exported V6 issue resolution verdicts CSV");
    });

    container.querySelector("#export-res-audit-csv")?.addEventListener("click", () => {
      const ok = downloadCsv(
        filtered,
        [
          { key: "post_id", label: "Post ID" },
          { key: "title", label: "Title" },
          { key: "issue", label: "Issue" },
          { key: "state", label: "Resolution State", get: (r) => (resState.engine === "v6" && r.v6_resolution_state ? r.v6_resolution_state : r.state) },
          { key: "v6_verdict", label: "V6 Verdict" },
          { key: "v6_contradictory_count", label: "Contradictory Signals" },
          { key: "proposal_count", label: "Solution Replies" },
          { key: "advice_count", label: "Advice Replies" },
          { key: "model_positive_count", label: "Binary Model Positives" },
          { key: "evidence_text", label: "Outcome Evidence" },
        ],
        `rci_resolution_audit_${resState.engine}.csv`
      );
      if (ok) ctx.showToast?.(`Exported ${filtered.length} audited threads to CSV`);
    });
  }

  paint();
}
