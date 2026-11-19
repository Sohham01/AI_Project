/**
 * View 1: Community Intelligence Studio (Overview — Built From Scratch on Dusk)
 * 1. Four Animated Ring-Gauge KPI Pods (cubic-out roll-ups + SVG arc sweeps)
 * 2. Full-Width Interactive 3-Stage Sankey Research Flow (Issues -> 6 Canonical Responses -> 4 Resolution States)
 * 3. Asymmetric Visual Split: Interactive Monthly Sentiment Spline (live crosshair) + Top Actionable Issues
 * 4. 3x2 Recent Thread Spotlight Grid + Compact Provenance Deck
 */
import { api } from "../api.js";
import {
  bindSankeyInteractions,
  bindSplineInteractions,
  horizontalBarChart,
  interactiveSplineAreaChart,
  sankeyFlowChart,
} from "../charts.js";
import { copyIdChip, deck, resolutionBadge, tallyStrip } from "../components.js";
import { dateText, esc, num, pretty } from "../format.js";

export async function renderOverview(container, ctx) {
  const [summary, sentimentRes] = await Promise.all([
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
    api.sentiment(),
  ]);
  ctx.summary = summary;

  const totalPosts = Number(summary.posts || 0);
  const totalComments = Number(summary.comments || 0);
  const resMap = Object.fromEntries(
    (summary.resolution_summary || []).map((r) => [r.label, Number(r.count || 0)])
  );
  const issues = summary.issue_distribution || [];
  const totalSolutions = issues.reduce((s, i) => s + Number(i.solutions_proposed || 0), 0);
  const totalAdvice = issues.reduce((s, i) => s + Number(i.advice_comments || 0), 0);
  const totalActionable = issues.reduce((s, i) => s + Number(i.actionable_recommendations || 0), 0);
  const reportedSuccess = resMap.reported_success || 0;
  const reportedPending = resMap.reported_pending || 0;
  const unconfirmedTotal = totalPosts - reportedSuccess - reportedPending;
  const actionablePct = totalComments ? (totalActionable / totalComments) * 100 : 0;
  const successPct = totalPosts ? (reportedSuccess / totalPosts) * 100 : 0;

  // Row 1: Dusk Ruled Tally Strip (micro-arcs strictly for real ratios, zero fake 100% rings)
  const tallyRow = tallyStrip([
    {
      label: "Synthetic corpus",
      valueText: num(totalPosts),
      countTo: totalPosts,
      unit: "threads",
      accent: true,
      note: `${num(totalComments)} comments across ${num(summary.issue_count)} balanced issue domains`,
      href: "#/explorer",
    },
    {
      label: "Actionable replies",
      valueText: `${actionablePct.toFixed(1)}%`,
      countTo: actionablePct,
      decimals: 1,
      suffix: "%",
      ratioPct: actionablePct,
      color: "var(--color-role-advice)",
      note: `${num(totalActionable)} of ${num(totalComments)} replies (${num(totalSolutions)} solution, ${num(totalAdvice)} advice)`,
      href: "#/responses",
    },
    {
      label: "Explicit outcome evidence",
      valueText: `${num(reportedSuccess + reportedPending)} / ${num(totalPosts)}`,
      ratioPct: successPct,
      color: "var(--color-positive)",
      note: `${num(reportedSuccess)} reported success, ${num(reportedPending)} pending, ${num(unconfirmedTotal)} unconfirmed`,
      href: "#/resolution",
    },
    {
      label: "Isolated baseline models",
      valueText: "5",
      countTo: 5,
      unit: "models",
      note: `Thread-grouped splits (896 / 192 / 192, zero leakage), ${num(summary.topic_count)} NMF topics`,
      href: "#/models",
    },
  ], "Synthetic corpus headline figures");

  // Row 2: Full-width Interactive 3-Stage Sankey Research Flow
  const sankeyDeck = deck("Interactive issue-to-evidence flow", {
    subtitle: "Hover any node or ribbon to trace synthetic discussions across 8 issue domains, the 6 fixed canonical response classes, and 4 resolution states; click any node to drill in",
    action: `<a class="btn btn-quiet btn-sm" href="#/resolution">Audit all 160 threads</a>`,
    body: sankeyFlowChart(summary),
  });

  // Row 3: Asymmetric Interactive Split (Monthly Sentiment Spline + Top Actionable Issues)
  const sortedIssues = [...issues].sort(
    (a, b) => Number(b.actionable_recommendations || 0) - Number(a.actionable_recommendations || 0)
  );
  const issueBars = sortedIssues.map((issue, idx) => ({
    label: pretty(issue.label),
    count: Number(issue.actionable_recommendations || 0),
    color: idx === 0 ? "var(--color-accent)" : idx < 3 ? "var(--color-role-advice)" : "var(--color-series-val)",
    href: `#/issues?selected=${encodeURIComponent(issue.key)}`,
  }));

  const splineDeck = deck("Monthly sentiment pulse", {
    subtitle: "Hover across Jan to Nov 2025 to scrub polarity counts; synthetic threads were generated sequentially by domain",
    action: `<a class="btn btn-quiet btn-sm" href="#/sentiment">Full affect studio</a>`,
    body: interactiveSplineAreaChart(sentimentRes.monthly || [], { chartId: "ov-spline" }),
  });

  const issuesDeck = deck("Actionable density by issue", {
    subtitle: "20 synthetic threads (160 comments) per domain, ranked by solution and advice count",
    action: `<a class="btn btn-quiet btn-sm" href="#/issues">Issues studio</a>`,
    body: horizontalBarChart(issueBars, { total: 160, maxCount: Math.max(...issueBars.map((b) => b.count)), unit: "actionable", showShare: true }),
  });

  // Row 4: Thread Spotlight Hairline List (flat, single containment layer)
  const spotlightRows = (summary.recent_discussions || []).map((post) => `
    <a class="spotlight-row" href="#/thread/${encodeURIComponent(post.id)}">
      <div class="spotlight-row-top">
        <span class="tag tag-neutral">${esc(pretty(post.issue))}</span>
        ${resolutionBadge(post.resolution_state)}
      </div>
      <div class="spotlight-row-title">${esc(post.title)}</div>
      <div class="spotlight-row-foot">
        <span class="num">${dateText(post.created_at)}</span>
        <span class="num">${num(post.comment_count)} replies</span>
      </div>
    </a>
  `).join("");

  const spotlightDeck = deck("Recent generated discussions", {
    subtitle: "Latest fictional discussions ordered by synthetic timestamp; open any thread to inspect nested parent-child replies",
    action: `<a class="btn btn-primary btn-sm" href="#/explorer">Open discussion workbench</a>`,
    body: `<div class="spotlight-grid">${spotlightRows}</div>`,
  });

  const prov = summary.provenance || {};
  const provenanceDeck = deck("Dataset provenance & synthetic boundaries", {
    subtitle: "Live metadata read from data/demo_synthetic/corpus.json",
    action: `<a class="btn btn-quiet btn-sm" href="#/models">Models &amp; evaluation</a>`,
    body: `
      <dl class="vault-kv-grid">
        <div><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "reddit-community-intelligence-demo")}</dd></div>
        <div><dt>Deterministic seed &amp; version</dt><dd class="num">Seed ${esc(prov.generation_seed)}, v${esc(prov.dataset_version || summary.dataset_version)}</dd></div>
        <div><dt>Authorization status</dt><dd><span class="tag tag-warning">${esc(prov.authorization_status || "synthetic_only")}</span></dd></div>
        <div><dt>Timestamp ordering</dt><dd>Sequential 2-day domain intervals (Jan to Nov 2025)</dd></div>
      </dl>
    `,
  });

  container.innerHTML = `
    ${tallyRow}
    ${sankeyDeck}
    <div class="studio-grid-asymmetric">
      ${splineDeck}
      ${issuesDeck}
    </div>
    ${spotlightDeck}
    ${provenanceDeck}
  `;

  bindSankeyInteractions(container);
  bindSplineInteractions(container);
}
