/**
 * View 1: Community Intelligence Studio (Overview — Insightful Cross-Model Edition)
 * 1. Four Ruled Tally Strip KPI Pods (including calibrated prediction confidence)
 * 2. Cross-Model Insight Engine Briefings (synthesizing Issue x Sentiment x Emotion x Response x Resolution x Architecture)
 * 3. Full-Width Interactive 3-Stage Sankey Research Flow
 * 4. Issue Priority & Friction Matrix (Quadrant visual + Ranked Action Table)
 * 5. Asymmetric Interactive Split: Monthly Sentiment Spline + Top Actionable Issues
 * 6. Recent Thread Spotlight (with confidence pills) + Compact Provenance Deck
 */
import { api } from "../api.js";
import {
  bindSankeyInteractions,
  bindSplineInteractions,
  horizontalBarChart,
  interactiveSplineAreaChart,
  issueFrictionQuadrantChart,
  sankeyFlowChart,
} from "../charts.js";
import {
  badge,
  confidenceBadge,
  copyIdChip,
  deck,
  resolutionBadge,
  tallyStrip,
} from "../components.js";
import { dateText, esc, monthLabel, num, percent, pretty } from "../format.js";

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

  const conf = summary.confidence_summary || {};
  const meanConfPct = Number(conf.mean_confidence || 0) * 100;
  const insights = summary.cross_model_insights || {};
  const insightCards = insights.cards || [];
  const priorityMatrix = summary.issue_priority_matrix || [];

  // Row 1: Dusk Ruled Tally Strip (now featuring Calibrated Confidence)
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
      label: "Calibrated confidence",
      valueText: meanConfPct > 0 ? `${meanConfPct.toFixed(1)}%` : "5 models",
      countTo: meanConfPct > 0 ? meanConfPct : 5,
      decimals: meanConfPct > 0 ? 1 : 0,
      suffix: meanConfPct > 0 ? "%" : "",
      ratioPct: meanConfPct > 0 ? meanConfPct : null,
      color: meanConfPct >= 75 ? "var(--color-positive)" : "var(--color-accent)",
      note: meanConfPct > 0
        ? `${percent(conf.high_share, 0)} high (>=80%), ${num(conf.needs_review_count)} flagged for review`
        : `GPU MuRIL-DAPT v3 hybrid + ${num(summary.topic_count)} NMF topics`,
      href: "#/models",
    },
  ], "Community corpus headline figures");

  // Row 2: Cross-Model Insight Engine Briefings
  const insightsDeck = insightCards.length
    ? deck("Cross-model insight engine", {
        subtitle: "Synthesized evidence joining all 5 supervised NLP classifiers, 10 NMF topics, calibrated confidence tiers, and 4-architecture OOD telemetry",
        action: `<a class="btn btn-quiet btn-sm" href="#/models">Why Transformer &amp; Error Audit</a>`,
        body: `
          <div class="insight-grid">
            ${insightCards.map((card) => `
              <article class="insight-card insight-tone-${esc(card.tone || "info")}">
                <div class="insight-card-top">
                  ${badge(card.category, card.tone === "accent" ? "info" : (card.tone || "neutral"))}
                  <span class="insight-metric num">${esc(card.metric)}</span>
                </div>
                <h3 class="insight-headline">${esc(card.headline)}</h3>
                <p class="insight-detail">${esc(card.detail)}</p>
                ${card.href ? `<a class="insight-link" href="${esc(card.href)}">${esc(card.cta || "Drill in")} &rarr;</a>` : ""}
              </article>
            `).join("")}
          </div>
        `,
      })
    : "";

  // Row 3: Full-width Interactive 3-Stage Sankey Research Flow
  const sankeyDeck = deck("Interactive issue-to-evidence flow", {
    subtitle: `Hover any node or ribbon to trace public discussions across ${num(issues.length)} issue domains, the 6 fixed canonical response classes, and 4 resolution states; click any node to drill in`,
    action: `<a class="btn btn-quiet btn-sm" href="#/resolution">Audit all ${num(totalPosts)} threads</a>`,
    body: sankeyFlowChart(summary),
  });

  // Row 4: Issue Priority & Friction Synthesis (Quadrant + Ranked Multi-Signal Table)
  const priorityDeck = priorityMatrix.length
    ? deck("Issue friction &amp; priority matrix", {
        subtitle: "Cross-task triage ranking combining Negative Sentiment (40%), Unconfirmed Resolution (35%), and Corpus Volume (25%), paired with dominant emotion, NMF cluster, and prediction confidence",
        action: `<a class="btn btn-quiet btn-sm" href="#/issues">Deep-dive issues</a>`,
        body: `
          <div class="priority-split">
            <div>${issueFrictionQuadrantChart(priorityMatrix)}</div>
            <div class="table-scroll">
              <table class="data-table priority-table" aria-label="Cross-model issue friction priority ranking">
                <thead>
                  <tr>
                    <th scope="col">Issue domain</th>
                    <th scope="col" class="num">Friction</th>
                    <th scope="col" class="num">Neg %</th>
                    <th scope="col" class="num">Unresolved</th>
                    <th scope="col">Top emotion</th>
                    <th scope="col" class="num">Mean conf</th>
                  </tr>
                </thead>
                <tbody>
                  ${priorityMatrix.map((m) => `
                    <tr>
                      <td>
                        <a class="table-primary-link" href="#/issues?selected=${encodeURIComponent(m.key)}">${esc(m.label)}</a>
                        <span class="table-subcode">${num(m.posts)} threads · ${num(m.comments)} comments · ${esc(m.top_topic)}</span>
                      </td>
                      <td class="num">${badge(`${m.friction_score.toFixed(0)}`, m.priority_band === "critical" ? "danger" : m.priority_band === "elevated" ? "warning" : m.priority_band === "moderate" ? "info" : "positive")}</td>
                      <td class="num">${percent(m.negative_share, 0)}</td>
                      <td class="num">${percent(m.unresolved_share, 0)}</td>
                      <td><span class="tag tag-neutral">${esc(pretty(m.dominant_emotion))}</span></td>
                      <td class="num">${confidenceBadge(m.mean_confidence)}</td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          </div>
        `,
      })
    : "";

  // Row 5: Asymmetric Interactive Split (Monthly Sentiment Spline + Top Actionable Issues)
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

  // Row 6: Thread Spotlight Hairline List (with calibrated confidence badges)
  const spotlightRows = (summary.recent_discussions || []).map((post) => `
    <a class="spotlight-row" href="#/thread/${encodeURIComponent(post.id)}">
      <div class="spotlight-row-top">
        <span class="tag tag-neutral">${esc(pretty(post.issue))}</span>
        ${resolutionBadge(post.resolution_state)}
        ${confidenceBadge(post.mean_confidence, post.confidence_tier)}
      </div>
      <div class="spotlight-row-title">${esc(post.title)}</div>
      <div class="spotlight-row-foot">
        <span class="num">${dateText(post.created_at)}</span>
        <span class="num">${num(post.comment_count)} replies</span>
      </div>
    </a>
  `).join("");

  const spotlightDeck = deck("Recent community discussions", {
    subtitle: "Latest anonymized public Reddit discussions with calibrated prediction confidence; open any thread to inspect nested replies",
    action: `<a class="btn btn-primary btn-sm" href="#/explorer">Open discussion workbench</a>`,
    body: `<div class="spotlight-grid">${spotlightRows}</div>`,
  });

  const prov = summary.provenance || {};
  const isSynthetic = Boolean(summary.synthetic_only);
  const provenanceDeck = deck("Dataset provenance &amp; methodology", {
    subtitle: "Live corpus metadata read from /api/demo/summary",
    action: `<a class="btn btn-quiet btn-sm" href="#/models">Models &amp; evaluation</a>`,
    body: `
      <dl class="vault-kv-grid">
        <div><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "iim-sambalpur-arctic-shift-v4")}</dd></div>
        <div><dt>Source &amp; version</dt><dd class="num">${esc(prov.source_provider === "arctic_shift" ? "Arctic Shift public archive" : "Local corpus")}, v${esc(prov.dataset_version || summary.dataset_version)}</dd></div>
        <div><dt>Privacy &amp; access</dt><dd><span class="tag ${isSynthetic ? "tag-warning" : "tag-positive"}">${esc(isSynthetic ? (prov.authorization_status || "synthetic_only") : "Anonymized public Reddit")}</span></dd></div>
        <div><dt>Archive time span</dt><dd>${esc(dateSpan)} (${num(communityCount)} subreddits)</dd></div>
      </dl>
    `,
  });

  container.innerHTML = `
    ${tallyRow}
    ${insightsDeck}
    ${sankeyDeck}
    ${priorityDeck}
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
