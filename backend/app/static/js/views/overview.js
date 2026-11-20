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
import { dateText, esc, monthLabel, num, pretty } from "../format.js";

export async function renderOverview(container, ctx) {
  const [summary, sentimentRes] = await Promise.all([
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
    api.sentiment(),
  ]);
  ctx.summary = summary;

  const totalPosts = Number(summary.posts || 0);
  const totalComments = Number(summary.comments || 0);
  const monthly = sentimentRes.monthly || [];
  const dateSpan = monthly.length >= 2
    ? `${monthLabel(monthly[0].month)} to ${monthLabel(monthly[monthly.length - 1].month)}`
    : (monthly.length === 1 ? monthLabel(monthly[0].month) : "Full archive");
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
  const communityCount = (summary.community_distribution || []).length;

  // Row 1: Dusk Ruled Tally Strip (micro-arcs strictly for real ratios, zero fake 100% rings)
  const tallyRow = tallyStrip([
    {
      label: "Community corpus",
      valueText: num(totalPosts),
      countTo: totalPosts,
      unit: "threads",
      accent: true,
      note: `${num(totalComments)} anonymized comments across ${num(communityCount)} subreddits`,
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
      label: "Active ML pipeline",
      valueText: "5",
      countTo: 5,
      unit: "models",
      note: `GPU MuRIL + Word+Char TF-IDF classifiers, ${num(summary.topic_count)} NMF topics`,
      href: "#/models",
    },
  ], "Community corpus headline figures");

  // Row 2: Full-width Interactive 3-Stage Sankey Research Flow
  const sankeyDeck = deck("Interactive issue-to-evidence flow", {
    subtitle: `Hover any node or ribbon to trace public discussions across ${num(issues.length)} issue domains, the 6 fixed canonical response classes, and 4 resolution states; click any node to drill in`,
    action: `<a class="btn btn-quiet btn-sm" href="#/resolution">Audit all ${num(totalPosts)} threads</a>`,
    body: sankeyFlowChart(summary),
  });

  // Row 3: Asymmetric Interactive Split (Monthly Sentiment Spline + Top Actionable Issues)
  const sortedIssues = [...issues].sort(
    (a, b) => Number(b.actionable_recommendations || 0) - Number(a.actionable_recommendations || 0)
  );
  const issueBars = sortedIssues.map((issue, idx) => ({
    label: pretty(issue.label),
    sublabel: `${num(issue.posts)} threads`,
    count: Number(issue.actionable_recommendations || 0),
    color: idx === 0 ? "var(--color-accent)" : idx < 3 ? "var(--color-role-advice)" : "var(--color-series-val)",
    href: `#/issues?selected=${encodeURIComponent(issue.key)}`,
  }));

  const splineDeck = deck("Monthly sentiment pulse", {
    subtitle: `Hover across ${dateSpan} to scrub predicted polarity counts by comment month`,
    action: `<a class="btn btn-quiet btn-sm" href="#/sentiment">Full affect studio</a>`,
    body: interactiveSplineAreaChart(monthly, { chartId: "ov-spline" }),
  });

  const issuesDeck = deck("Actionable density by issue", {
    subtitle: "Ranked by total predicted solution and advice replies across public community threads",
    action: `<a class="btn btn-quiet btn-sm" href="#/issues">Issues studio</a>`,
    body: horizontalBarChart(issueBars, {
      total: Math.max(1, totalActionable),
      maxCount: Math.max(1, ...issueBars.map((b) => b.count)),
      unit: "actionable",
      showShare: true,
    }),
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

  const spotlightDeck = deck("Recent community discussions", {
    subtitle: "Latest anonymized public Reddit discussions ordered by post date; open any thread to inspect nested parent-child replies",
    action: `<a class="btn btn-primary btn-sm" href="#/explorer">Open discussion workbench</a>`,
    body: `<div class="spotlight-grid">${spotlightRows}</div>`,
  });

  const prov = summary.provenance || {};
  const isSynthetic = Boolean(summary.synthetic_only);
  const provenanceDeck = deck("Dataset provenance & methodology", {
    subtitle: "Live corpus metadata read from /api/demo/summary",
    action: `<a class="btn btn-quiet btn-sm" href="#/models">Models &amp; evaluation</a>`,
    body: `
      <dl class="vault-kv-grid">
        <div><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "iim-sambalpur-arctic-shift-v2")}</dd></div>
        <div><dt>Source &amp; version</dt><dd class="num">${esc(prov.source_provider === "arctic_shift" ? "Arctic Shift public archive" : "Local corpus")}, v${esc(prov.dataset_version || summary.dataset_version)}</dd></div>
        <div><dt>Privacy &amp; access</dt><dd><span class="tag ${isSynthetic ? "tag-warning" : "tag-positive"}">${esc(isSynthetic ? (prov.authorization_status || "synthetic_only") : "Anonymized public Reddit")}</span></dd></div>
        <div><dt>Archive time span</dt><dd>${esc(dateSpan)} (${num(communityCount)} subreddits)</dd></div>
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
