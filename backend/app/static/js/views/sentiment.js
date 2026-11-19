/**
 * View 6: Sentiment & Emotion Pulse Studio
 * Interactive smooth spline area chart with crosshair scrubbing, animated emotion bars,
 * and sortable Issue x Polarity normalized matrix.
 */
import { api } from "../api.js";
import {
  bindSplineInteractions,
  horizontalBarChart,
  interactiveSplineAreaChart,
  stackedCompositionBar,
  stackedMatrixRows,
} from "../charts.js";
import { callout, deck, tallyStrip } from "../components.js";
import {
  EMOTION_COLORS,
  SENTIMENT_META,
  SENTIMENT_ORDER,
  esc,
  issueLabelFromKey,
  num,
  pretty,
} from "../format.js";

const sentState = { issueSort: "negative_share" };

export async function renderSentiment(container, ctx) {
  const [data, summary] = await Promise.all([
    api.sentiment(),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
  ]);
  ctx.summary = summary;

  const sentMap = Object.fromEntries((data.sentiment || data.overall || []).map((s) => [s.label, Number(s.count || 0)]));
  const totalComments = SENTIMENT_ORDER.reduce((sum, k) => sum + (sentMap[k] || 0), 0) || Number(summary.comments || 1280);
  const emotions = [...(data.emotions || [])].sort((a, b) => Number(b.count) - Number(a.count));
  const totalEmotionAssignments = emotions.reduce((sum, e) => sum + Number(e.count || 0), 0);
  const avgEmotions = totalComments ? totalEmotionAssignments / totalComments : 0;

  const posPct = (sentMap.positive || 0) / totalComments * 100;
  const negPct = (sentMap.negative || 0) / totalComments * 100;
  const neuPct = (sentMap.neutral || 0) / totalComments * 100;

  const sentimentSegments = SENTIMENT_ORDER.map((k) => ({
    key: k,
    label: SENTIMENT_META[k].label,
    count: sentMap[k] || 0,
    color: SENTIMENT_META[k].color,
    href: `#/explorer?sentiment=${encodeURIComponent(k)}`,
  }));

  const emotionRows = emotions.map((e) => ({
    label: pretty(e.label),
    sublabel: e.label,
    count: Number(e.count || 0),
    color: EMOTION_COLORS[e.label] || "var(--color-accent)",
    href: `#/explorer?emotion=${encodeURIComponent(e.label)}`,
  }));

  const byIssueEnriched = (data.by_issue || []).map((row) => {
    const dist = Object.fromEntries((row.distribution || []).map((d) => [d.label, Number(d.count || 0)]));
    const total = SENTIMENT_ORDER.reduce((s, k) => s + (dist[k] || 0), 0);
    return {
      key: row.issue,
      label: issueLabelFromKey(row.issue, summary),
      dist,
      total,
      positiveShare: total ? dist.positive / total : 0,
      negativeShare: total ? dist.negative / total : 0,
      neutralShare: total ? dist.neutral / total : 0,
    };
  });

  function paint() {
    const sortedIssues = [...byIssueEnriched].sort((a, b) => {
      if (sentState.issueSort === "label") return a.label.localeCompare(b.label);
      if (sentState.issueSort === "positive_share") return b.positiveShare - a.positiveShare;
      if (sentState.issueSort === "neutral_share") return b.neutralShare - a.neutralShare;
      return b.negativeShare - a.negativeShare;
    });

    const matrixRows = sortedIssues.map((row) => ({
      key: row.key,
      label: row.label,
      total: row.total,
      actionHref: `#/issues?selected=${encodeURIComponent(row.key)}`,
      segments: SENTIMENT_ORDER.map((k) => ({
        key: k,
        label: SENTIMENT_META[k].label,
        short: SENTIMENT_META[k].label,
        count: row.dist[k] || 0,
        color: SENTIMENT_META[k].color,
        href: `#/explorer?issue=${encodeURIComponent(row.key)}&sentiment=${encodeURIComponent(k)}`,
      })),
    }));

    const tallyRow = tallyStrip([
      {
        label: "Predicted positive",
        valueText: `${posPct.toFixed(1)}%`,
        countTo: posPct,
        decimals: 1,
        suffix: "%",
        ratioPct: posPct,
        color: "var(--color-positive)",
        note: `${num(sentMap.positive || 0)} of ${num(totalComments)} comments (in-sample baseline)`,
        href: "#/explorer?sentiment=positive",
      },
      {
        label: "Predicted negative",
        valueText: `${negPct.toFixed(1)}%`,
        countTo: negPct,
        decimals: 1,
        suffix: "%",
        ratioPct: negPct,
        color: "var(--color-negative)",
        note: `${num(sentMap.negative || 0)} comments; select to inspect negative threads`,
        href: "#/explorer?sentiment=negative",
      },
      {
        label: "Predicted neutral",
        valueText: `${neuPct.toFixed(1)}%`,
        countTo: neuPct,
        decimals: 1,
        suffix: "%",
        ratioPct: neuPct,
        color: "var(--color-accent)",
        note: `${num(sentMap.neutral || 0)} comments (informational or verification replies)`,
        href: "#/explorer?sentiment=neutral",
      },
      {
        label: "Multi-label emotion density",
        valueText: avgEmotions.toFixed(2),
        countTo: avgEmotions,
        decimals: 2,
        unit: "/ comment",
        accent: true,
        note: `${num(totalEmotionAssignments)} predicted emotion tags across ${num(emotions.length)} classes`,
      },
    ], "Sentiment and emotion headline figures");

    container.innerHTML = `
      ${callout(
        "All polarity and emotion counts below are <strong>predicted outputs</strong> from the isolated synthetic TF-IDF baselines (in-sample corpus display), not verified human affect. Sentiment is single-label; emotions are multi-label (comments may carry multiple emotions simultaneously).",
        { tone: "warning", title: "Predicted vs. verified affect:" }
      )}

      ${tallyRow}

      ${deck("Interactive sentiment timeline (generated months)", {
        subtitle: "Hover across Jan to Nov 2025 to scrub exact polarity counts or filter curves; posts were generated sequentially by issue domain every 2 days",
        body: interactiveSplineAreaChart(data.monthly || [], { chartId: "sent-main-spline" }),
      })}

      <div class="studio-grid-2">
        ${deck("Predicted sentiment polarity", {
          subtitle: "Mutually exclusive single-label classification (n = 1,280); click any polarity to filter discussions",
          body: `
            ${stackedCompositionBar(sentimentSegments, { total: totalComments, unit: "comments", ariaLabel: "Overall predicted sentiment mix" })}
            <div class="spacer-md"></div>
            ${horizontalBarChart(sentimentSegments, { total: totalComments, unit: "comments", showShare: true })}
          `,
        })}
        ${deck("Predicted emotion prevalence", {
          subtitle: `Multi-label co-occurring affect (${num(totalEmotionAssignments)} assignments across ${num(totalComments)} comments; bars sum above 100%)`,
          body: `
            ${horizontalBarChart(emotionRows, { total: totalComments, maxCount: totalComments, unit: "comments", showShare: true })}
            <p class="chart-footnote">In the synthetic generator, positive comments commonly co-occur with <code>hope</code> and <code>gratitude</code> (or <code>confidence</code>), while negative comments co-occur with <code>stress</code> and <code>frustration</code>.</p>
          `,
        })}
      </div>

      ${deck("Predicted sentiment by issue domain", {
        subtitle: "100% normalized polarity mix per scenario issue category; click any sentiment count to filter discussions",
        action: `
          <div class="filter-pills" role="group" aria-label="Sort issues by sentiment share">
            <button type="button" class="pill-btn ${sentState.issueSort === "negative_share" ? "is-active" : ""}" data-sent-sort="negative_share">Highest negative</button>
            <button type="button" class="pill-btn ${sentState.issueSort === "positive_share" ? "is-active" : ""}" data-sent-sort="positive_share">Highest positive</button>
            <button type="button" class="pill-btn ${sentState.issueSort === "neutral_share" ? "is-active" : ""}" data-sent-sort="neutral_share">Highest neutral</button>
            <button type="button" class="pill-btn ${sentState.issueSort === "label" ? "is-active" : ""}" data-sent-sort="label">A to Z</button>
          </div>
        `,
        body: stackedMatrixRows(matrixRows, {
          legendItems: SENTIMENT_ORDER.map((k) => ({ label: SENTIMENT_META[k].label, color: SENTIMENT_META[k].color })),
          unit: "comments",
        }),
      })}
    `;

    bindSplineInteractions(container);
    container.querySelectorAll("[data-sent-sort]").forEach((btn) => {
      btn.addEventListener("click", () => {
        sentState.issueSort = btn.dataset.sentSort;
        paint();
      });
    });
  }

  paint();
}
