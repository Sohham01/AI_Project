/**
 * View 1: Community Intelligence Studio (Overview — Calm, Honest, Scannable Edition)
 * Visual rhythm:
 * 1. Ruled 4-Metric Tally Strip (corpus, actionable replies, explicit outcomes, confidence)
 * 2. Key Cross-Model Findings (top 3 high-signal cards upfront + one-click reveal for all 5)
 * 3. Honest 3-Stage Corpus Pipeline (Issue Domains -> 6 Canonical Roles -> 4 Resolution States)
 * 4. Unified Honest 0–100% Linear Issue Triage Visual (no distorted axes, no duplicate table)
 * 5. Asymmetric 2-Column Split (Zero-Overshoot Monthly Sentiment Timeline + Actionable Density)
 * 6. Clean 2-Column Recent Discussions + Compact Single-Strip Provenance Bar
 */
import { api } from "../api.js?v=6.0";
import {
  bindChartTooltips,
  bindModelNetworkInteractions,
  bindNetworkInteractions,
  bindSankeyInteractions,
  bindSplineInteractions,
  communityNetworkDiagram,
  horizontalBarChart,
  interactiveSplineAreaChart,
  issueFrictionQuadrantChart,
  modelNetworkDiagram,
  sankeyFlowChart,
} from "../charts.js?v=6.0";
import {
  confidenceBadge,
  copyIdChip,
  deck,
  resolutionBadge,
  tallyStrip,
} from "../components.js?v=6.0";
import { dateText, esc, monthLabel, num, percent, plural, pretty } from "../format.js?v=6.0";

export async function renderOverview(container, ctx) {
  const [summary, sentimentRes, modelsData] = await Promise.all([
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
    api.sentiment(),
    api.models().catch(() => null),
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
  const communityNetwork = summary.community_network || null;

  // Row 1: Calm Ruled Tally Strip
  const tallyRow = tallyStrip([
    {
      label: "Community corpus",
      valueText: num(totalPosts),
      countTo: totalPosts,
      unit: "threads",
      accent: true,
      note: `${num(totalComments)} anonymized comments · ${num(communityCount)} subreddits`,
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
      note: `${num(totalSolutions)} solution, ${num(totalAdvice)} advice replies`,
      href: "#/responses",
    },
    {
      label: "Explicit outcomes",
      valueText: `${num(reportedSuccess + reportedPending)} / ${num(totalPosts)}`,
      ratioPct: successPct,
      color: "var(--color-positive)",
      note: `${num(reportedSuccess)} success · ${num(reportedPending)} pending · ${num(unconfirmedTotal)} unconfirmed`,
      href: "#/resolution",
    },
    {
      label: "Mean confidence",
      valueText: meanConfPct > 0 ? `${meanConfPct.toFixed(1)}%` : "5 models",
      countTo: meanConfPct > 0 ? meanConfPct : 5,
      decimals: meanConfPct > 0 ? 1 : 0,
      suffix: meanConfPct > 0 ? "%" : "",
      ratioPct: meanConfPct > 0 ? meanConfPct : null,
      color: meanConfPct >= 75 ? "var(--color-positive)" : "var(--color-accent)",
      note: meanConfPct > 0
        ? `${percent(conf.high_share, 0)} high (≥80%) · ${num(conf.low_count)} below 55%`
        : `V6 (MuRIL-WWM + XLM-R) + ${num(summary.topic_count)} NMF topics`,
      href: "#/models",
    },
  ], "Community corpus headline figures");

  // Row 2: Key Findings (Top 3 upfront to prevent card overload; optional reveal for all)
  const renderInsightItem = (card) => {
    const toneColor = card.tone === "danger"
      ? "var(--color-negative)"
      : card.tone === "warning"
        ? "var(--color-warning)"
        : card.tone === "positive"
          ? "var(--color-positive)"
          : card.tone === "accent"
            ? "var(--color-accent)"
            : "var(--color-info)";
    return `
      <article class="insight-card">
        <div class="insight-card-top">
          <span class="insight-cat"><span class="swatch-inline" style="background:${toneColor}" aria-hidden="true"></span>${esc(card.category)}</span>
          <span class="insight-metric num">${esc(card.metric)}</span>
        </div>
        <h3 class="insight-headline">${esc(card.headline)}</h3>
        <p class="insight-detail">${esc(card.detail)}</p>
        ${card.href ? `<a class="insight-link" href="${esc(card.href)}">${esc(card.cta || "Inspect")} &rarr;</a>` : ""}
      </article>
    `;
  };

  const hasExtraInsights = insightCards.length > 3;
  const insightsDeck = insightCards.length
    ? deck("Key cross-model findings", {
        subtitle: "Empirical takeaways joining all 5 supervised classifiers, calibrated confidence tiers, and real held-out benchmarks",
        action: `
          ${hasExtraInsights ? `<button type="button" class="btn btn-quiet btn-sm" id="toggle-all-insights" aria-expanded="false">Show all ${insightCards.length}</button>` : ""}
          <a class="btn btn-quiet btn-sm" href="#/models">Model benchmarks</a>
        `,
        body: `
          <div class="insight-grid" id="primary-insights-grid">
            ${insightCards.slice(0, 3).map(renderInsightItem).join("")}
          </div>
          ${hasExtraInsights ? `
            <div class="insight-grid insight-grid-extra" id="extra-insights-grid" hidden>
              ${insightCards.slice(3).map(renderInsightItem).join("")}
            </div>
          ` : ""}
        `,
      })
    : "";

  // Row 3: Interactive Observed Community & Discourse Network
  const networkDeck = communityNetwork?.nodes?.communities?.length
    ? deck("Community &amp; discourse network", {
        id: "overview-network-deck",
        subtitle: "Exact observed links from subreddits through issue hubs to canonical response roles or emotions — hover or click any node to isolate its neighborhood",
        action: `
          <div class="segmented" role="group" aria-label="Switch network right-hand layer">
            <button type="button" class="seg-btn is-active" data-net-mode="responses">Response roles</button>
            <button type="button" class="seg-btn" data-net-mode="emotions">Emotions</button>
          </div>
        `,
        body: `<div id="overview-network-slot">${communityNetworkDiagram(communityNetwork, "responses")}</div>`,
      })
    : "";

  // Row 4: Honest 3-Stage Corpus Breakdown Pipeline
  const pipelineDeck = deck("Corpus flow: issues, response roles &amp; outcomes", {
    subtitle: "Observed distribution at each stage of the community pipeline — hover for exact shares, click any bar to drill in",
    action: `<a class="btn btn-quiet btn-sm" href="#/resolution">Audit all ${num(totalPosts)} threads</a>`,
    body: sankeyFlowChart(summary),
  });

  // Row 4: Unified Honest 0–100% Linear Issue Triage Visual
  const priorityDeck = priorityMatrix.length
    ? deck("Issue friction &amp; triage comparison", {
        subtitle: "Side-by-side comparison on honest 0%–100% linear scales — hover any domain for full emotion, topic, and confidence context",
        action: `<a class="btn btn-quiet btn-sm" href="#/issues">Deep-dive issues</a>`,
        body: issueFrictionQuadrantChart(priorityMatrix),
      })
    : "";

  // Row 5: Asymmetric Interactive Split (Monthly Sentiment Timeline + Actionable Density)
  const sortedIssues = [...issues].sort(
    (a, b) => Number(b.actionable_recommendations || 0) - Number(a.actionable_recommendations || 0)
  );
  const issueBars = sortedIssues.map((issue, idx) => ({
    label: pretty(issue.label),
    sublabel: plural(issue.posts, "thread"),
    count: Number(issue.actionable_recommendations || 0),
    color: idx === 0 ? "var(--color-accent)" : idx < 3 ? "var(--color-role-advice)" : "var(--color-series-val)",
    href: `#/issues?selected=${encodeURIComponent(issue.key)}`,
  }));

  const splineDeck = deck("Monthly sentiment timeline", {
    subtitle: `${dateSpan} · zero-anchored monotone curves, hover to inspect monthly shares`,
    action: `<a class="btn btn-quiet btn-sm" href="#/sentiment">Sentiment studio</a>`,
    body: interactiveSplineAreaChart(monthly, { chartId: "ov-spline" }),
  });

  const issuesDeck = deck("Actionable replies by issue", {
    subtitle: "Predicted solution and advice replies ranked across issue domains",
    action: `<a class="btn btn-quiet btn-sm" href="#/issues">Issues studio</a>`,
    body: horizontalBarChart(issueBars, {
      total: Math.max(1, totalActionable),
      maxCount: Math.max(1, ...issueBars.map((b) => b.count)),
      unit: "actionable",
      showShare: true,
    }),
  });

  // Model Architecture & Inference Network Deck
  const modelNetDeck = modelsData?.tasks
    ? deck("Model architecture &amp; inference network", {
        id: "overview-model-network-deck",
        subtitle: "How anonymized Reddit replies flow through strictly-inductive MuRIL-DAPT-WWM (Issue/Sentiment/Emotion), XLM-RoBERTa-base (Response & Solution), 5 calibrated heads, and confidence routing",
        action: `<a class="btn btn-quiet btn-sm" href="#/models">Full benchmarks &amp; error audit</a>`,
        body: modelNetworkDiagram(modelsData, summary),
      })
    : "";

  // Row 6: Clean 2-Column Recent Discussions
  const spotlightRows = (summary.recent_discussions || []).slice(0, 6).map((post) => `
    <a class="spotlight-row" href="#/thread/${encodeURIComponent(post.id)}">
      <div class="spotlight-row-top">
        <span class="meta-issue-label">${esc(pretty(post.issue))}</span>
        <span class="meta-sep" aria-hidden="true">·</span>
        ${resolutionBadge(post.resolution_state)}
        ${confidenceBadge(post.mean_confidence, post.confidence_tier)}
      </div>
      <div class="spotlight-row-title">${esc(post.title)}</div>
      <div class="spotlight-row-foot">
        <span class="num">${dateText(post.created_at)}</span>
        <span class="num">${plural(post.comment_count, "reply", "replies")}</span>
      </div>
    </a>
  `).join("");

  const spotlightDeck = deck("Recent community discussions", {
    subtitle: "Latest anonymized public threads — open any discussion to inspect the nested reply tree and per-comment probabilities",
    action: `<a class="btn btn-primary btn-sm" href="#/explorer">Open workbench</a>`,
    body: `<div class="spotlight-grid">${spotlightRows}</div>`,
  });

  // Compact Provenance Strip (low visual weight instead of a full extra deck)
  const prov = summary.provenance || {};
  const provenanceStrip = `
    <aside class="compact-prov-bar anim-enter" aria-label="Dataset provenance summary">
      <div class="compact-prov-items">
        <span><span class="muted">Dataset</span> ${copyIdChip(prov.dataset_id || "iim-sambalpur-real-reddit-v6")}</span>
        <span class="meta-sep" aria-hidden="true">·</span>
        <span class="num"><span class="muted">Version</span> v${esc(prov.dataset_version || summary.dataset_version)}</span>
        <span class="meta-sep" aria-hidden="true">·</span>
        <span><span class="muted">Source</span> ${esc(prov.source_provider === "arctic_shift" ? "Arctic Shift (anonymized Reddit)" : "Local corpus")}</span>
        <span class="meta-sep" aria-hidden="true">·</span>
        <span><span class="muted">Span</span> ${esc(dateSpan)}</span>
      </div>
      <a class="row-open-link" href="#/models">Models &amp; audit &rarr;</a>
    </aside>
  `;

  container.innerHTML = `
    ${tallyRow}
    ${insightsDeck}
    ${networkDeck}
    ${pipelineDeck}
    ${priorityDeck}
    <div class="studio-grid-asymmetric">
      ${splineDeck}
      ${issuesDeck}
    </div>
    ${modelNetDeck}
    ${spotlightDeck}
    ${provenanceStrip}
  `;

  // Wire "Show all insights" progressive disclosure
  const toggleBtn = container.querySelector("#toggle-all-insights");
  const extraGrid = container.querySelector("#extra-insights-grid");
  if (toggleBtn && extraGrid) {
    toggleBtn.addEventListener("click", () => {
      const open = extraGrid.hidden;
      extraGrid.hidden = !open;
      toggleBtn.setAttribute("aria-expanded", String(open));
      toggleBtn.textContent = open ? "Show top 3" : `Show all ${insightCards.length}`;
    });
  }

  // Wire Network layer switch (Response roles <-> Emotions)
  container.querySelectorAll("[data-net-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.netMode;
      container.querySelectorAll("[data-net-mode]").forEach((b) => b.classList.toggle("is-active", b === btn));
      const slot = container.querySelector("#overview-network-slot");
      if (slot && communityNetwork) {
        slot.innerHTML = communityNetworkDiagram(communityNetwork, mode);
        bindNetworkInteractions(slot);
      }
    });
  });

  bindNetworkInteractions(container);
  bindModelNetworkInteractions(container);
  bindSankeyInteractions(container);
  bindSplineInteractions(container);
  bindChartTooltips(container);
}
