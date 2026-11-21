/**
 * Low-Cognitive-Load, Visually Honest SVG & Data Visualizations for RCI Studio.
 * Design rules enforced across every chart:
 * - Zero truncated bar/area baselines (all counts, shares, F1, and perplexity charts anchor honestly at 0).
 * - Zero redundant dual Y-axes (e.g., DAPT shows single-axis Held-Out Perplexity; MLM loss sits in tooltip).
 * - Zero non-linear axis distortions (linear 0-100% scales with explicit ticks and sample sizes n=).
 * - Zero spline overshoot (Fritsch-Carlson monotone cubic interpolation clamped strictly within [0, yMax]).
 * - Zero invented marginal cross-tab edges (3-stage pipeline presents true per-stage proportions cleanly).
 */
import {
  CANONICAL_RESPONSE_TYPES,
  EMOTION_COLORS,
  RESOLUTION_META,
  RESOLUTION_ORDER,
  RESPONSE_META,
  SENTIMENT_META,
  SENTIMENT_ORDER,
  esc,
  monthLabel,
  num,
  percent,
  plural,
  pretty,
  share,
} from "./format.js?v=5.6";
import { hideVizTooltip, showVizTooltip } from "./motion.js?v=5.6";

/* ==========================================================================
   0. Shared Interactive Floating Tooltip Binder (Used Across All Charts)
   ========================================================================== */
export function bindChartTooltips(container) {
  if (!container) return;
  container.querySelectorAll("[data-chart-tip-title]").forEach((el) => {
    if (el.dataset.tipBound === "1") return;
    el.dataset.tipBound = "1";
    const show = (e) => {
      const title = el.dataset.chartTipTitle || "";
      const body = el.dataset.chartTipBody || "";
      const rows = el.dataset.chartTipRows ? JSON.parse(el.dataset.chartTipRows) : null;
      const hint = el.dataset.chartTipHint || "";
      const rowsHtml = Array.isArray(rows)
        ? rows.map(([k, v]) => `<div class="viz-tip-row"><span>${esc(k)}</span><span class="num">${esc(v)}</span></div>`).join("")
        : "";
      const html = `
        <strong>${esc(title)}</strong>
        ${body ? `<div>${esc(body)}</div>` : ""}
        ${rowsHtml}
        ${hint ? `<div class="muted" style="margin-top:4px">${esc(hint)}</div>` : ""}
      `;
      showVizTooltip(html, e.clientX, e.clientY);
    };
    el.addEventListener("mouseenter", show);
    el.addEventListener("mousemove", show);
    el.addEventListener("mouseleave", hideVizTooltip);
  });
}

/* ==========================================================================
   1. Honest 3-Stage Corpus Pipeline (Overview Signature Visual)
   Replaces marginal-assumption spaghetti ribbons with 3 crisp, proportional,
   clickable stage breakdowns: Issue Domains -> Canonical Roles -> Resolution.
   ========================================================================== */
export function sankeyFlowChart(summary) {
  const issues = [...(summary.issue_distribution || [])].sort(
    (a, b) => Number(b.comments || 0) - Number(a.comments || 0)
  );
  const respMap = Object.fromEntries(
    (summary.response_distribution || []).map((r) => [r.label, Number(r.count || 0)])
  );
  const resMap = Object.fromEntries(
    (summary.resolution_summary || []).map((r) => [r.label, Number(r.count || 0)])
  );

  const totalComments = Math.max(1, Number(summary.comments || 1));
  const totalPosts = Math.max(1, Number(summary.posts || 1));
  const maxIssueComments = Math.max(1, ...issues.map((i) => Number(i.comments || 0)));
  const maxResp = Math.max(1, ...CANONICAL_RESPONSE_TYPES.map((k) => Number(respMap[k] || 0)));
  const maxRes = Math.max(1, ...RESOLUTION_ORDER.map((k) => Number(resMap[k] || 0)));

  const issueItems = issues.map((i, idx) => {
    const c = Number(i.comments || 0);
    const act = Number(i.actionable_recommendations || 0);
    const wPct = (c / maxIssueComments) * 100;
    const actPctOfBar = c > 0 ? (act / c) * 100 : 0;
    const rows = JSON.stringify([
      ["Comments", `${num(c)} (${share(c, totalComments)})`],
      ["Threads", `${num(i.posts)}`],
      ["Actionable replies", `${num(act)} (${share(act, c)})`],
      ["Reported success", `${num(i.reported_successes)} / ${num(i.posts)} threads`],
    ]);
    return `
      <a class="pipe-row"
         href="#/issues?selected=${encodeURIComponent(i.key)}"
         data-chart-tip-title="${esc(pretty(i.label))}"
         data-chart-tip-rows="${esc(rows)}"
         data-chart-tip-hint="Click to open Issues deep dive">
        <div class="pipe-row-head">
          <span class="pipe-row-label">${esc(pretty(i.label))}</span>
          <span class="pipe-row-stat num">${num(c)} <span class="muted">(${share(c, totalComments, 0)})</span></span>
        </div>
        <div class="pipe-track">
          <span class="pipe-fill" style="width:${wPct.toFixed(1)}%;background:${idx === 0 ? "var(--color-accent)" : "var(--color-series-val)"}">
            <span class="pipe-fill-inner" style="width:${actPctOfBar.toFixed(1)}%;background:var(--color-role-advice)" title="Actionable share"></span>
          </span>
        </div>
      </a>
    `;
  }).join("");

  const responseItems = CANONICAL_RESPONSE_TYPES.map((k, idx) => {
    const c = Number(respMap[k] || 0);
    const wPct = (c / maxResp) * 100;
    const meta = RESPONSE_META[k];
    const isActionable = meta.group === "actionable";
    const rows = JSON.stringify([
      ["Predicted replies", `${num(c)} (${share(c, totalComments)})`],
      ["Group", isActionable ? "Actionable intervention" : "Community discourse"],
      ["Canonical slot", `#${idx + 1} of 6`],
    ]);
    return `
      <a class="pipe-row"
         href="#/explorer?response=${encodeURIComponent(k)}"
         data-chart-tip-title="${esc(`${idx + 1}. ${meta.label}`)}"
         data-chart-tip-body="${esc(meta.definition)}"
         data-chart-tip-rows="${esc(rows)}"
         data-chart-tip-hint="Click to filter discussions">
        <div class="pipe-row-head">
          <span class="pipe-row-label">
            <span class="swatch-inline" style="background:${meta.color}" aria-hidden="true"></span>
            ${esc(`${idx + 1}. ${meta.label}`)}
          </span>
          <span class="pipe-row-stat num">${num(c)} <span class="muted">(${share(c, totalComments, 0)})</span></span>
        </div>
        <div class="pipe-track">
          <span class="pipe-fill" style="width:${wPct.toFixed(1)}%;background:${meta.color}"></span>
        </div>
      </a>
    `;
  }).join("");

  const resolutionItems = RESOLUTION_ORDER.map((k) => {
    const c = Number(resMap[k] || 0);
    const wPct = (c / maxRes) * 100;
    const meta = RESOLUTION_META[k];
    const rows = JSON.stringify([
      ["Threads", `${num(c)} of ${num(totalPosts)} (${share(c, totalPosts)})`],
    ]);
    return `
      <a class="pipe-row"
         href="#/resolution?state=${encodeURIComponent(k)}"
         data-chart-tip-title="${esc(meta.label)}"
         data-chart-tip-body="${esc(meta.description)}"
         data-chart-tip-rows="${esc(rows)}"
         data-chart-tip-hint="Click to audit threads in this state">
        <div class="pipe-row-head">
          <span class="pipe-row-label">
            <span class="swatch-inline" style="background:${meta.color}" aria-hidden="true"></span>
            ${esc(meta.label)}
          </span>
          <span class="pipe-row-stat num">${num(c)} <span class="muted">(${share(c, totalPosts, 0)})</span></span>
        </div>
        <div class="pipe-track">
          <span class="pipe-fill" style="width:${wPct.toFixed(1)}%;background:${meta.color}"></span>
        </div>
      </a>
    `;
  }).join("");

  return `
    <div class="pipeline-flow" data-sankey-root role="region" aria-label="Three-stage corpus summary from issue domains through canonical response roles to resolution evidence states">
      <div class="pipe-stage">
        <div class="pipe-stage-head">
          <span class="pipe-step-num num">01</span>
          <div>
            <h3 class="pipe-stage-title">Issue domains (${issues.length})</h3>
            <p class="pipe-stage-sub">By comment volume · teal inset = actionable replies</p>
          </div>
        </div>
        <div class="pipe-list">${issueItems}</div>
      </div>

      <div class="pipe-arrow" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M5 10H15M15 10L10.5 5.5M15 10L10.5 14.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>

      <div class="pipe-stage">
        <div class="pipe-stage-head">
          <span class="pipe-step-num num">02</span>
          <div>
            <h3 class="pipe-stage-title">Canonical response roles (6)</h3>
            <p class="pipe-stage-sub">All ${num(totalComments)} comments · fixed 1–6 order</p>
          </div>
        </div>
        <div class="pipe-list">${responseItems}</div>
      </div>

      <div class="pipe-arrow" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M5 10H15M15 10L10.5 5.5M15 10L10.5 14.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>

      <div class="pipe-stage">
        <div class="pipe-stage-head">
          <span class="pipe-step-num num">03</span>
          <div>
            <h3 class="pipe-stage-title">Resolution evidence (4 states)</h3>
            <p class="pipe-stage-sub">All ${num(totalPosts)} threads · explicit outcomes only</p>
          </div>
        </div>
        <div class="pipe-list">${resolutionItems}</div>
      </div>
    </div>
    <p class="chart-footnote">Each column shows true observed counts scaled from 0% within its stage (comments for stages 1–2; threads for stage 3). Click any bar to drill into the matching slice.</p>
  `;
}

export function bindSankeyInteractions(container) {
  bindChartTooltips(container);
}

/* ==========================================================================
   2. Zero-Overshoot Monotone Cubic Timeline (Fritsch-Carlson)
   Strictly prevents cubic spline overshoot below 0 or past local extrema.
   ========================================================================== */
function monotoneCubicPath(pts, minY, maxY) {
  const n = pts.length;
  if (!n) return "";
  if (n === 1) return `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  if (n === 2) return `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)} L ${pts[1].x.toFixed(1)} ${pts[1].y.toFixed(1)}`;

  const dx = new Array(n - 1);
  const dy = new Array(n - 1);
  const slope = new Array(n - 1);
  for (let i = 0; i < n - 1; i += 1) {
    dx[i] = Math.max(1e-6, pts[i + 1].x - pts[i].x);
    dy[i] = pts[i + 1].y - pts[i].y;
    slope[i] = dy[i] / dx[i];
  }

  const m = new Array(n);
  m[0] = slope[0];
  m[n - 1] = slope[n - 2];
  for (let i = 1; i < n - 1; i += 1) {
    if (slope[i - 1] * slope[i] <= 0) {
      m[i] = 0;
    } else {
      const w1 = 2 * dx[i] + dx[i - 1];
      const w2 = dx[i] + 2 * dx[i - 1];
      m[i] = (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i]);
    }
  }

  // Enforce Fritsch-Carlson monotonicity constraint
  for (let i = 0; i < n - 1; i += 1) {
    if (Math.abs(slope[i]) < 1e-9) {
      m[i] = 0;
      m[i + 1] = 0;
    } else {
      const a = m[i] / slope[i];
      const b = m[i + 1] / slope[i];
      const mag = Math.hypot(a, b);
      if (mag > 3) {
        const t = 3 / mag;
        m[i] = t * a * slope[i];
        m[i + 1] = t * b * slope[i];
      }
    }
  }

  const clampY = (y) => Math.max(minY, Math.min(maxY, y));
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < n - 1; i += 1) {
    const h = dx[i] / 3;
    const cp1x = pts[i].x + h;
    const cp1y = clampY(pts[i].y + m[i] * h);
    const cp2x = pts[i + 1].x - h;
    const cp2y = clampY(pts[i + 1].y - m[i + 1] * h);
    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${pts[i + 1].x.toFixed(1)} ${pts[i + 1].y.toFixed(1)}`;
  }
  return d;
}

function niceAxisMax(rawMax, steps = 4) {
  if (rawMax <= 0) return steps;
  const rough = rawMax / steps;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const frac = rough / pow;
  const niceFrac = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 2.5 ? 2.5 : frac <= 5 ? 5 : 10;
  return Math.ceil(rawMax / (niceFrac * pow)) * niceFrac * pow;
}

export function interactiveSplineAreaChart(monthly, { chartId = "sent-spline" } = {}) {
  if (!monthly || !monthly.length) {
    return `<div class="chart-empty">No monthly sentiment records available.</div>`;
  }

  const W = 720;
  const H = 252;
  const pad = { top: 20, right: 20, bottom: 36, left: 46 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const seriesMeta = [
    { key: "neutral", label: "Neutral", color: "var(--color-neutral-bar)" },
    { key: "positive", label: "Positive", color: "var(--color-positive)" },
    { key: "negative", label: "Negative", color: "var(--color-negative)" },
  ];

  const normalized = monthly.map((m) => {
    const distMap = Array.isArray(m.distribution)
      ? Object.fromEntries(m.distribution.map((d) => [d.label, Number(d.count || 0)]))
      : (m.counts || m);
    return {
      month: m.month,
      positive: Number(distMap.positive || 0),
      neutral: Number(distMap.neutral || 0),
      negative: Number(distMap.negative || 0),
    };
  });

  const maxVal = Math.max(
    1,
    ...normalized.flatMap((m) => seriesMeta.map((s) => Number(m[s.key] || 0)))
  );
  const yMax = niceAxisMax(maxVal, 4);
  const ticks = [0, Math.round(yMax * 0.25), Math.round(yMax * 0.5), Math.round(yMax * 0.75), yMax];

  const xAt = (idx) => pad.left + (normalized.length > 1 ? (idx / (normalized.length - 1)) * plotW : plotW / 2);
  const yAt = (val) => pad.top + plotH - (val / yMax) * plotH;

  const defs = seriesMeta.map((s) => `
    <linearGradient id="${esc(chartId)}-grad-${s.key}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${s.color}" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="${s.color}" stop-opacity="0"/>
    </linearGradient>
  `).join("");

  const gridLines = ticks.map((t, i) => {
    const y = yAt(t);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${i === 0 ? "none" : "2 4"}" opacity="${i === 0 ? "0.9" : "0.65"}"/>
      <text x="${pad.left - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${num(t)}</text>
    `;
  }).join("");

  const xStep = normalized.length > 9 ? Math.ceil(normalized.length / 7) : 1;
  const xLabels = normalized.map((m, idx) => {
    const show = idx === 0 || idx === normalized.length - 1 || idx % xStep === 0;
    if (!show) return "";
    // Avoid crowding the penultimate label against the final label
    if (idx !== normalized.length - 1 && normalized.length - 1 - idx < Math.max(1, Math.floor(xStep * 0.6))) return "";
    return `<text x="${xAt(idx).toFixed(1)}" y="${(H - 10).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${esc(monthLabel(m.month))}</text>`;
  }).join("");

  const seriesGroups = seriesMeta.map((s) => {
    const pts = normalized.map((m, idx) => ({
      x: xAt(idx),
      y: yAt(Number(m[s.key] || 0)),
      val: Number(m[s.key] || 0),
    }));
    const linePath = monotoneCubicPath(pts, pad.top, pad.top + plotH);
    const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} L ${pts[0].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} Z`;

    return `
      <g class="spline-series" data-spline-series="${s.key}">
        <path class="spline-area" d="${areaPath}" fill="url(#${esc(chartId)}-grad-${s.key})"/>
        <path class="spline-line" d="${linePath}" stroke="${s.color}" data-draw-path/>
        ${pts.map((p, idx) => `
          <circle class="spline-dot" data-dot-idx="${idx}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="${s.color}" stroke="var(--color-surface-solid)" stroke-width="2"/>
        `).join("")}
      </g>
    `;
  }).join("");

  const stepW = normalized.length > 1 ? plotW / (normalized.length - 1) : plotW;
  const hitCols = normalized.map((m, idx) => {
    const cx = xAt(idx);
    const x = Math.max(pad.left, cx - stepW / 2);
    const w = Math.min(W - pad.right - x, idx === 0 || idx === normalized.length - 1 ? stepW / 2 : stepW);
    const pos = m.positive;
    const neu = m.neutral;
    const neg = m.negative;
    const tot = pos + neu + neg;
    return `
      <rect
        class="spline-hit-col"
        x="${x.toFixed(1)}"
        y="${pad.top}"
        width="${Math.max(12, w).toFixed(1)}"
        height="${plotH}"
        data-hit-idx="${idx}"
        data-hit-x="${cx.toFixed(1)}"
        data-month="${esc(monthLabel(m.month))}"
        data-pos="${pos}"
        data-neu="${neu}"
        data-neg="${neg}"
        data-tot="${tot}"
      />
    `;
  }).join("");

  return `
    <div class="spline-container" data-spline-root="${esc(chartId)}">
      <div class="spline-controls" role="group" aria-label="Filter sentiment curves">
        <button type="button" class="series-pill is-active" data-series-filter="all">All polarities</button>
        ${seriesMeta.map((s) => `
          <button type="button" class="series-pill" data-series-filter="${s.key}">
            <span class="legend-swatch" style="background:${s.color}" aria-hidden="true"></span>
            ${s.label}
          </button>
        `).join("")}
      </div>
      <div class="spacer-xs"></div>
      <svg class="spline-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly predicted comment sentiment counts on a zero-anchored monotone line chart">
        <defs>${defs}</defs>
        ${gridLines}
        ${seriesGroups}
        <line class="spline-crosshair" data-crosshair x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotH}"/>
        ${xLabels}
        ${hitCols}
      </svg>
    </div>
  `;
}

export function bindSplineInteractions(container) {
  container.querySelectorAll("[data-spline-root]").forEach((root) => {
    const crosshair = root.querySelector("[data-crosshair]");
    const dots = [...root.querySelectorAll(".spline-dot")];
    const hits = [...root.querySelectorAll(".spline-hit-col")];
    const seriesEls = [...root.querySelectorAll("[data-spline-series]")];
    const filterBtns = [...root.querySelectorAll("[data-series-filter]")];

    filterBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.seriesFilter;
        filterBtns.forEach((b) => b.classList.toggle("is-active", b === btn));
        seriesEls.forEach((el) => {
          const isFocus = mode !== "all" && el.dataset.splineSeries === mode;
          el.classList.toggle("is-hidden", mode !== "all" && !isFocus);
          el.classList.toggle("is-isolated", isFocus);
        });
      });
    });

    const clearHover = () => {
      crosshair?.classList.remove("is-visible");
      dots.forEach((d) => d.classList.remove("is-visible"));
      hideVizTooltip();
    };

    hits.forEach((rect) => {
      const activate = (e) => {
        const idx = rect.dataset.hitIdx;
        const cx = rect.dataset.hitX;
        if (crosshair) {
          crosshair.setAttribute("x1", cx);
          crosshair.setAttribute("x2", cx);
          crosshair.classList.add("is-visible");
        }
        dots.forEach((d) => d.classList.toggle("is-visible", d.dataset.dotIdx === idx));
        const tot = Number(rect.dataset.tot || 1);
        showVizTooltip(
          `<strong>${esc(rect.dataset.month)} · ${num(tot)} comments</strong>
           <div class="viz-tip-row"><span>Positive</span><span class="num">${num(rect.dataset.pos)} (${share(rect.dataset.pos, tot, 0)})</span></div>
           <div class="viz-tip-row"><span>Neutral</span><span class="num">${num(rect.dataset.neu)} (${share(rect.dataset.neu, tot, 0)})</span></div>
           <div class="viz-tip-row"><span>Negative</span><span class="num">${num(rect.dataset.neg)} (${share(rect.dataset.neg, tot, 0)})</span></div>`,
          e.clientX,
          e.clientY
        );
      };
      rect.addEventListener("mouseenter", activate);
      rect.addEventListener("mousemove", activate);
      rect.addEventListener("mouseleave", clearHover);
    });
  });
}

/* ==========================================================================
   3. Honest Area-Proportional NMF Topic Grid (Zero Fake Orbital Spokes)
   Area of every circle scales directly with document_count (r = sqrt(c/max)*R).
   ========================================================================== */
export function topicConstellationSvg(topics, selectedId) {
  const sorted = [...topics].sort((a, b) => Number(b.document_count) - Number(a.document_count));
  const maxCount = Math.max(1, ...sorted.map((t) => Number(t.document_count || 0)));
  const totalDocs = Math.max(1, sorted.reduce((s, t) => s + Number(t.document_count || 0), 0));

  const cols = Math.min(5, Math.max(2, Math.ceil(sorted.length / 2)));
  const rows = Math.ceil(sorted.length / cols);
  const cellW = 112;
  const cellH = 136;
  const padX = 16;
  const padY = 18;
  const W = cols * cellW + padX * 2;
  const H = rows * cellH + padY * 2 + 12;
  const maxR = 40;

  const cells = sorted.map((t, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const cx = padX + col * cellW + cellW / 2;
    const cy = padY + row * cellH + 50;
    const count = Number(t.document_count || 0);
    // Strict area proportionality (r ~ sqrt(count)), minimum 11px for clickability
    const r = Math.max(11, Math.sqrt(count / maxCount) * maxR);
    const color = t.character.script === "devanagari"
      ? "var(--color-accent-2)"
      : t.character.tone === "info"
        ? "var(--color-role-advice)"
        : "var(--color-accent)";
    const isSel = Number(t.topic_id) === Number(selectedId);
    const kw1 = (t.keywords || [])[0] || "";
    const kw2 = (t.keywords || [])[1] || "";
    const kwLine = [kw1, kw2].filter(Boolean).join(" · ");
    const shortKw = kwLine.length > 16 ? `${kwLine.slice(0, 15)}…` : kwLine;

    return `
      <g class="topic-orb ${isSel ? "is-selected" : ""}" tabindex="0" role="button" data-select-topic="${t.topic_id}" aria-label="Topic ${t.topic_id}, ${num(count)} comments (${share(count, totalDocs)}), ${esc(kwLine)}">
        <circle class="topic-orb-ring" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${(r + 6).toFixed(1)}"/>
        <circle class="topic-orb-circle" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" fill-opacity="${isSel ? "0.28" : "0.16"}" stroke="${color}" stroke-width="${isSel ? "2.2" : "1.6"}"/>
        <text x="${cx.toFixed(1)}" y="${(cy + 4).toFixed(1)}" text-anchor="middle" fill="var(--color-ink)" font-size="11.5" font-weight="600" font-family="var(--font-body)">T${t.topic_id}</text>
        <text x="${cx.toFixed(1)}" y="${(padY + row * cellH + 104).toFixed(1)}" text-anchor="middle" fill="var(--color-ink)" font-size="11" font-weight="500" font-family="var(--font-body)">${esc(shortKw)}</text>
        <text x="${cx.toFixed(1)}" y="${(padY + row * cellH + 120).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">${num(count)} (${share(count, totalDocs, 0)})</text>
      </g>
    `;
  }).join("");

  return `
    <div class="constellation-stage">
      <svg class="constellation-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Area-proportional grid of ${sorted.length} unsupervised NMF topic clusters ordered by comment volume">
        ${cells}
      </svg>
      <p class="chart-footnote" style="margin-top:var(--space-xs)">Circle area is proportional to assigned comments (ordered largest to smallest). Positions are grid-aligned, not semantic distances.</p>
    </div>
  `;
}

/* ==========================================================================
   4. Animated Radial Canonical Response Wheel (6 Fixed Classes)
   ========================================================================== */
export function radialResponseWheel(orderedItems, total) {
  const r = 98;
  const circ = 2 * Math.PI * r;
  let offsetAcc = 0;

  const arcs = orderedItems.map((item) => {
    const frac = total > 0 ? item.count / total : 1 / 6;
    const segLen = Math.max(4, frac * circ - 4);
    const dashOffset = -offsetAcc;
    offsetAcc += frac * circ;
    return `
      <circle
        class="wheel-arc"
        cx="130"
        cy="130"
        r="${r}"
        stroke="${item.meta.color}"
        stroke-dasharray="${segLen.toFixed(1)} ${(circ - segLen).toFixed(1)}"
        stroke-dashoffset="${dashOffset.toFixed(1)}"
        data-wheel-role="${esc(item.key)}"
        data-wheel-label="${esc(`${item.rank}. ${item.meta.label}`)}"
        data-wheel-count="${num(item.count)}"
        data-wheel-share="${share(item.count, total)}"
      />
    `;
  }).join("");

  return `
    <div class="wheel-wrap" data-wheel-root>
      <svg class="wheel-svg" viewBox="0 0 260 260" role="img" aria-label="Canonical six-class response distribution ring">
        <circle cx="130" cy="130" r="${r}" fill="none" stroke="var(--color-track)" stroke-width="18"/>
        ${arcs}
      </svg>
      <div class="wheel-center">
        <strong data-wheel-center-val>${num(total)}</strong>
        <span data-wheel-center-label>Classified comments</span>
      </div>
    </div>
  `;
}

export function bindWheelInteractions(container, total) {
  const root = container.querySelector("[data-wheel-root]");
  if (!root) return;
  const arcs = [...root.querySelectorAll("[data-wheel-role]")];
  const centerVal = root.querySelector("[data-wheel-center-val]");
  const centerLabel = root.querySelector("[data-wheel-center-label]");

  arcs.forEach((arc) => {
    arc.addEventListener("mouseenter", () => {
      arcs.forEach((other) => {
        other.classList.toggle("is-active", other === arc);
        other.classList.toggle("is-dimmed", other !== arc);
      });
      if (centerVal) centerVal.textContent = arc.dataset.wheelShare;
      if (centerLabel) centerLabel.textContent = `${arc.dataset.wheelLabel} (${arc.dataset.wheelCount})`;
    });
    arc.addEventListener("mouseleave", () => {
      arcs.forEach((other) => other.classList.remove("is-active", "is-dimmed"));
      if (centerVal) centerVal.textContent = num(total);
      if (centerLabel) centerLabel.textContent = "Classified comments";
    });
    arc.addEventListener("click", () => {
      window.location.hash = `#/explorer?response=${encodeURIComponent(arc.dataset.wheelRole)}`;
    });
  });
}

/* ==========================================================================
   5. Shared Bars, Stacked Matrices & Mini Strips
   ========================================================================== */
export function stackedCompositionBar(segments, { total, unit = "items", ariaLabel = "Distribution bar" } = {}) {
  const safeTotal = total || segments.reduce((s, x) => s + Number(x.count || 0), 0);
  if (!safeTotal) return `<div class="chart-empty">No data available.</div>`;

  let cursor = 0;
  const viewW = 500;
  const rects = segments.map((seg) => {
    const val = Number(seg.count || 0);
    if (val <= 0) return "";
    const w = (val / safeTotal) * viewW;
    const rect = `
      <rect x="${cursor.toFixed(2)}" y="0" width="${Math.max(w, 1).toFixed(2)}" height="10" fill="${esc(seg.color)}">
        <title>${esc(seg.label)}: ${num(val)} ${esc(unit)} (${share(val, safeTotal)})</title>
      </rect>
    `;
    cursor += w;
    return rect;
  }).join("");

  const legend = segments.map((seg) => {
    const body = `
      <span class="legend-swatch" style="background:${esc(seg.color)}" aria-hidden="true"></span>
      <span>${esc(seg.label)}</span>
      <span class="num legend-stat">${num(seg.count)} <span class="legend-pct">(${share(seg.count, safeTotal)})</span></span>
    `;
    return seg.href
      ? `<a class="legend-link" href="${esc(seg.href)}">${body}</a>`
      : `<span class="legend-item">${body}</span>`;
  }).join("");

  return `
    <div class="stack-viz">
      <svg class="stack-bar-svg" viewBox="0 0 ${viewW} 10" preserveAspectRatio="none" role="img" aria-label="${esc(ariaLabel)}">
        <rect x="0" y="0" width="${viewW}" height="10" fill="var(--color-track)"/>
        ${rects}
      </svg>
      <div class="chart-legend">${legend}</div>
    </div>
  `;
}

export function horizontalBarChart(rows, { total = null, maxCount = null, unit = "comments", showShare = true } = {}) {
  if (!rows || !rows.length) return `<div class="chart-empty">No distribution records available.</div>`;
  const computedMax = maxCount || Math.max(1, ...rows.map((r) => Number(r.count || 0)));
  const computedTotal = total || rows.reduce((s, r) => s + Number(r.count || 0), 0);

  return `
    <div class="hbar-chart">
      ${rows.map((row) => {
        const count = Number(row.count || 0);
        const pctWidth = Math.max(0, Math.min(100, (count / computedMax) * 100));
        const labelHtml = row.href
          ? `<a class="hbar-label-link" href="${esc(row.href)}">${esc(row.label)}</a>`
          : `<span class="hbar-label-text">${esc(row.label)}</span>`;
        return `
          <div class="hbar-row">
            <div class="hbar-head">
              <div class="hbar-label-group">
                ${labelHtml}
                ${row.sublabel ? `<span class="hbar-sublabel">${esc(row.sublabel)}</span>` : ""}
              </div>
              <div class="num hbar-value">
                ${num(count)} ${esc(unit)}
                ${showShare && computedTotal > 0 ? `<span class="hbar-pct">(${share(count, computedTotal)})</span>` : ""}
              </div>
            </div>
            <div class="hbar-track">
              <span class="hbar-fill" style="width:${pctWidth.toFixed(1)}%;background:${esc(row.color || "var(--color-accent)")}"></span>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

export function stackedMatrixRows(rows, { legendItems = [], unit = "comments" } = {}) {
  if (!rows || !rows.length) return `<div class="chart-empty">No grouped distribution data available.</div>`;
  const legend = legendItems.length
    ? `<div class="chart-legend">${legendItems.map((l) => `<span class="legend-item"><span class="legend-swatch" style="background:${esc(l.color)}" aria-hidden="true"></span><span>${esc(l.label)}</span></span>`).join("")}</div>`
    : "";

  const body = rows.map((row) => {
    const total = Number(row.total || 0);
    let cursor = 0;
    const viewW = 400;
    const rects = (row.segments || []).map((seg) => {
      const val = Number(seg.count || 0);
      if (val <= 0 || total <= 0) return "";
      const w = (val / total) * viewW;
      const rect = `<rect x="${cursor.toFixed(2)}" y="0" width="${Math.max(w, 1).toFixed(2)}" height="8" fill="${esc(seg.color)}"><title>${esc(row.label)}, ${esc(seg.label)}: ${num(val)} (${share(val, total)})</title></rect>`;
      cursor += w;
      return rect;
    }).join("");

    const stats = (row.segments || []).map((seg) => {
      const content = `<span class="legend-swatch" style="background:${esc(seg.color)}" aria-hidden="true"></span><span>${esc(seg.short || seg.label)}</span> <strong class="num">${share(seg.count, total, 0)}</strong> <span class="num muted">(${num(seg.count)})</span>`;
      return seg.href
        ? `<a class="matrix-stat-link" href="${esc(seg.href)}">${content}</a>`
        : `<span class="matrix-stat">${content}</span>`;
    }).join("");

    const titleEl = row.actionHref
      ? `<a class="matrix-row-title" href="${esc(row.actionHref)}">${esc(row.label)}</a>`
      : `<span class="matrix-row-title">${esc(row.label)}</span>`;

    return `
      <div class="matrix-row">
        <div class="matrix-row-head">${titleEl}<span class="num matrix-row-total">${num(total)} ${esc(unit)}</span></div>
        <svg class="matrix-bar-svg" viewBox="0 0 ${viewW} 8" preserveAspectRatio="none" role="img" aria-label="${esc(row.label)}">
          <rect x="0" y="0" width="${viewW}" height="8" fill="var(--color-track)"/>
          ${rects}
        </svg>
        <div class="matrix-stats">${stats}</div>
      </div>
    `;
  }).join("");

  return `${legend}<div class="matrix-list">${body}</div>`;
}

export function responseMiniStrip(responseCounts) {
  const counts = Array.isArray(responseCounts)
    ? Object.fromEntries(responseCounts.map((r) => [r.label, Number(r.count || 0)]))
    : (responseCounts || {});
  const total = CANONICAL_RESPONSE_TYPES.reduce((s, k) => s + Number(counts[k] || 0), 0);
  if (!total) return `<span class="muted">No response predictions</span>`;

  let cursor = 0;
  const viewW = 100;
  const rects = CANONICAL_RESPONSE_TYPES.map((key) => {
    const val = Number(counts[key] || 0);
    if (!val) return "";
    const w = (val / total) * viewW;
    const r = `<rect x="${cursor.toFixed(1)}" y="0" width="${Math.max(w, 1).toFixed(1)}" height="5" fill="${RESPONSE_META[key].color}"><title>${RESPONSE_META[key].label}: ${val}</title></rect>`;
    cursor += w;
    return r;
  }).join("");
  const actionable = Number(counts.advice || 0) + Number(counts.solution || 0);
  return `
    <div class="mini-response-strip" title="${CANONICAL_RESPONSE_TYPES.map((k) => `${RESPONSE_META[k].label}: ${counts[k] || 0}`).join(", ")}">
      <svg class="mini-strip-svg" viewBox="0 0 ${viewW} 5" aria-hidden="true">${rects}</svg>
      <span class="num mini-strip-caption">${actionable} actionable / ${total}</span>
    </div>
  `;
}

export function modelComparisonSvgChart(tasks) {
  if (!tasks || !tasks.length) return `<div class="chart-empty">No model benchmarks available.</div>`;
  const W = 760;
  const rowH = 40;
  const pad = { top: 34, right: 112, bottom: 32, left: 112 };
  const H = pad.top + tasks.length * rowH + pad.bottom;
  const plotW = W - pad.left - pad.right;
  const xAt = (v) => pad.left + Math.max(0, Math.min(1, v)) * plotW;

  const grid = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    const x = xAt(t);
    return `
      <line x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${(H - pad.bottom).toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${t === 0 ? "none" : "2 4"}" opacity="0.7"/>
      <text x="${x.toFixed(1)}" y="${(H - pad.bottom + 18).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">${Math.round(t * 100)}%</text>
    `;
  }).join("");

  const rows = tasks.map((t, i) => {
    const cy = pad.top + i * rowH + rowH / 2;
    const vVal = Number.isFinite(Number(t.valF1)) ? Number(t.valF1) : 0;
    const tVal = Number.isFinite(Number(t.testF1)) ? Number(t.testF1) : 0;
    const x1 = xAt(Math.min(vVal, tVal));
    const x2 = xAt(Math.max(vVal, tVal));
    return `
      <g>
        <text x="${pad.left - 12}" y="${(cy + 4).toFixed(1)}" text-anchor="end" fill="var(--color-ink)" font-size="12" font-weight="500" font-family="var(--font-body)">${esc(t.label)}</text>
        <line x1="${pad.left}" y1="${cy.toFixed(1)}" x2="${W - pad.right}" y2="${cy.toFixed(1)}" stroke="var(--color-track)" stroke-width="1.5"/>
        <line x1="${x1.toFixed(1)}" y1="${cy.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${cy.toFixed(1)}" stroke="var(--color-accent)" stroke-width="3" opacity="0.4"/>
        <circle cx="${xAt(vVal).toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" fill="var(--color-surface-solid)" stroke="var(--color-muted)" stroke-width="2"/>
        <circle cx="${xAt(tVal).toFixed(1)}" cy="${cy.toFixed(1)}" r="5.5" fill="var(--color-accent)"/>
        <text x="${W - pad.right + 12}" y="${(cy + 4).toFixed(1)}" text-anchor="start" fill="var(--color-ink)" font-size="12" font-weight="600" font-family="var(--font-body)">${percent(t.testF1, 1, "N/A")}</text>
      </g>
    `;
  }).join("");

  return `
    <div>
      <svg class="responsive-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation versus held-out test macro F1 across all five active models">
        <text x="${pad.left}" y="16" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">Macro F1 on thread-grouped splits (honest 0%–100% scale)</text>
        ${grid}
        ${rows}
      </svg>
      <div class="chart-legend" style="margin-top:var(--space-xs)">
        <span class="legend-item"><span class="legend-swatch" style="background:transparent;border:2px solid var(--color-muted)"></span><span>Validation Macro F1</span></span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent)"></span><span>Held-out Test Macro F1</span></span>
      </div>
    </div>
  `;
}

export function confusionMatrixTable(matrix, labels, { openByDefault = false } = {}) {
  if (!Array.isArray(matrix) || !matrix.length || !Array.isArray(labels) || !labels.length) return "";
  const flat = matrix.flat().map((v) => Number(v || 0));
  const total = flat.reduce((a, b) => a + b, 0);
  if (total <= 0) return "";
  const maxCell = Math.max(1, ...flat);

  const head = labels.map((l) => `<th scope="col">${esc(pretty(l))}</th>`).join("");
  const rows = matrix.map((row, rIdx) => {
    const cells = row.map((val, cIdx) => {
      const n = Number(val || 0);
      const ratio = n / maxCell;
      let cls = "cm-zero";
      if (n > 0 && rIdx === cIdx) cls = ratio > 0.66 ? "cm-diag-3" : ratio > 0.33 ? "cm-diag-2" : "cm-diag-1";
      else if (n > 0 && rIdx !== cIdx) cls = ratio > 0.4 ? "cm-err-2" : "cm-err-1";
      return `<td class="num ${cls}">${num(n)}</td>`;
    }).join("");
    return `<tr><th scope="row">${esc(pretty(labels[rIdx] || `Class ${rIdx + 1}`))}</th>${cells}</tr>`;
  }).join("");

  return `
    <details class="confusion-details" ${openByDefault ? "open" : ""}>
      <summary>Confusion matrix · actual (rows) vs. predicted (columns), n = ${num(total)}</summary>
      <div class="table-scroll">
        <table class="confusion-table" aria-label="Held-out test confusion matrix">
          <thead><tr><th scope="col">Actual / Pred</th>${head}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>
  `;
}

/* ==========================================================================
   6. 4-Architecture Head-to-Head Dumbbell Dot-Plot (Honest 0%–100% Scale)
   Replaces the 25-bar visual wall with 5 crisp horizontal range tracks.
   Shows best non-Transformer baseline -> MuRIL-DAPT v3 gain in 0.5 seconds.
   ========================================================================== */
const ARCH_SERIES = [
  { key: "logistic_regression", short: "LogReg", color: "var(--color-muted)", shape: "hollow" },
  { key: "ann_mlp", short: "ANN (MLP)", color: "var(--color-warning)", shape: "diamond" },
  { key: "cnn_textcnn", short: "1D CNN", color: "var(--color-info)", shape: "square" },
  { key: "transformer_muril_dapt_v3", short: "MuRIL-DAPT v3", color: "var(--color-accent)", shape: "solid" },
  { key: "production_hybrid", short: "Prod Hybrid", color: "var(--color-positive)", shape: "ring" },
];

function renderArchMarker(shape, cx, cy, color) {
  const x = cx.toFixed(1);
  const y = cy.toFixed(1);
  if (shape === "hollow") {
    return `<circle cx="${x}" cy="${y}" r="4.5" fill="var(--color-surface-solid)" stroke="${color}" stroke-width="1.8"/>`;
  }
  if (shape === "diamond") {
    const r = 5.2;
    return `<polygon points="${x},${(cy - r).toFixed(1)} ${(cx + r).toFixed(1)},${y} ${x},${(cy + r).toFixed(1)} ${(cx - r).toFixed(1)},${y}" fill="var(--color-surface-solid)" stroke="${color}" stroke-width="1.6"/>`;
  }
  if (shape === "square") {
    const s = 8;
    return `<rect x="${(cx - s / 2).toFixed(1)}" y="${(cy - s / 2).toFixed(1)}" width="${s}" height="${s}" rx="1.5" fill="var(--color-surface-solid)" stroke="${color}" stroke-width="1.6"/>`;
  }
  if (shape === "ring") {
    return `<circle cx="${x}" cy="${y}" r="5.5" fill="var(--color-surface-solid)" stroke="${color}" stroke-width="2.4"/>`;
  }
  return `<circle cx="${x}" cy="${y}" r="6.2" fill="${color}" stroke="var(--color-surface-solid)" stroke-width="1.6"/>`;
}

export function architectureComparisonChart(archComp, mode = "test") {
  const perTask = archComp?.per_task || {};
  const taskOrder = ["issue", "solution", "sentiment", "response", "emotion"].filter((t) => t in perTask);
  if (!taskOrder.length) return `<div class="chart-empty">4-architecture comparison not emitted in this artifact.</div>`;

  const field = mode === "test" ? "test_macro_f1" : "probe_macro_f1";
  const subtitle = mode === "test"
    ? "Real held-out Reddit test Macro F1 (honest 0%–100% scale · n = 92 unseen-thread comments, 0 synthetic rows)"
    : "Auxiliary 54-probe stress diagnostic Macro F1 (honest 0%–100% scale · observational only, never used for selection)";

  const W = 880;
  const rowH = 48;
  const pad = { top: 44, right: 156, bottom: 36, left: 108 };
  const H = pad.top + taskOrder.length * rowH + pad.bottom;
  const plotW = W - pad.left - pad.right;
  const xAt = (v) => pad.left + Math.max(0, Math.min(1, v)) * plotW;

  const grid = [0, 0.25, 0.5, 0.75, 1.0].map((t) => {
    const x = xAt(t);
    return `
      <line x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${(H - pad.bottom).toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${t === 0 ? "none" : "2 4"}" opacity="${t === 0 ? "0.9" : "0.6"}"/>
      <text x="${x.toFixed(1)}" y="${(H - pad.bottom + 20).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">${Math.round(t * 100)}%</text>
    `;
  }).join("");

  const rows = taskOrder.map((taskKey, idx) => {
    const cy = pad.top + idx * rowH + rowH / 2;
    const tModels = perTask[taskKey] || {};
    const vals = Object.fromEntries(
      ARCH_SERIES.map((s) => [s.key, Math.max(0, Math.min(1, Number(tModels[s.key]?.[field] ?? 0)))])
    );

    const lr = vals.logistic_regression;
    const ann = vals.ann_mlp;
    const cnn = vals.cnn_textcnn;
    const muril = vals.transformer_muril_dapt_v3;
    const hyb = vals.production_hybrid;

    const minBase = Math.min(lr, ann, cnn);
    const maxBase = Math.max(lr, ann, cnn);
    const gainPp = (muril - maxBase) * 100;
    const gainSign = gainPp >= 0 ? "+" : "";

    // Gentle vertical micro-offset only when Hybrid sits right on top of MuRIL so both markers stay crisp
    const hybridOffsetY = Math.abs(hyb - muril) < 0.015 ? -7 : 0;
    const murilOffsetY = Math.abs(hyb - muril) < 0.015 ? 4 : 0;

    const tipRows = JSON.stringify([
      ["MuRIL-DAPT v3", `${(muril * 100).toFixed(1)}%`],
      ["Production Hybrid", `${(hyb * 100).toFixed(1)}%`],
      ["Logistic Regression", `${(lr * 100).toFixed(1)}%`],
      ["ANN (2-layer MLP)", `${(ann * 100).toFixed(1)}%`],
      ["1D TextCNN", `${(cnn * 100).toFixed(1)}%`],
      ["MuRIL vs. Best Non-Trans.", `${gainSign}${gainPp.toFixed(1)} pp`],
    ]);

    return `
      <g class="dumbbell-row"
         data-chart-tip-title="${esc(`${pretty(taskKey)} · ${mode === "test" ? "Real Held-Out Test" : "54-Probe Stress"}`)}"
         data-chart-tip-rows="${esc(tipRows)}">
        <rect class="dumbbell-hit" x="0" y="${(cy - rowH / 2 + 2).toFixed(1)}" width="${W}" height="${rowH - 4}" rx="8" fill="transparent"/>
        <text x="${pad.left - 14}" y="${(cy + 4).toFixed(1)}" text-anchor="end" fill="var(--color-ink)" font-size="12.5" font-weight="600" font-family="var(--font-body)">${esc(pretty(taskKey))}</text>

        <!-- Full 0-100% quiet track -->
        <line x1="${pad.left}" y1="${cy.toFixed(1)}" x2="${W - pad.right}" y2="${cy.toFixed(1)}" stroke="var(--color-track)" stroke-width="2" stroke-linecap="round"/>

        <!-- Baseline spread span -->
        <line x1="${xAt(minBase).toFixed(1)}" y1="${cy.toFixed(1)}" x2="${xAt(maxBase).toFixed(1)}" y2="${cy.toFixed(1)}" stroke="var(--color-rule-2)" stroke-width="3" stroke-linecap="round"/>

        <!-- Highlighted Transformer gain bridge from best baseline -> MuRIL -->
        <line x1="${xAt(Math.min(maxBase, muril)).toFixed(1)}" y1="${cy.toFixed(1)}" x2="${xAt(Math.max(maxBase, muril)).toFixed(1)}" y2="${cy.toFixed(1)}" stroke="${gainPp >= 0 ? "var(--color-accent)" : "var(--color-warning)"}" stroke-width="4" stroke-linecap="round" opacity="0.55"/>

        <!-- Non-transformer baseline markers (muted) -->
        ${renderArchMarker("hollow", xAt(lr), cy, "var(--color-muted)")}
        ${renderArchMarker("diamond", xAt(ann), cy, "var(--color-warning)")}
        ${renderArchMarker("square", xAt(cnn), cy, "var(--color-info)")}

        <!-- Highlighted MuRIL-DAPT v3 & Hybrid markers -->
        ${renderArchMarker("ring", xAt(hyb), cy + hybridOffsetY, "var(--color-positive)")}
        ${renderArchMarker("solid", xAt(muril), cy + murilOffsetY, "var(--color-accent)")}

        <!-- Direct right-margin readout (zero overlaps) -->
        <text x="${W - pad.right + 16}" y="${(cy + 4).toFixed(1)}" text-anchor="start" fill="var(--color-ink)" font-size="12.5" font-weight="600" font-family="var(--font-body)">${(muril * 100).toFixed(1)}%</text>
        <text x="${W - pad.right + 72}" y="${(cy + 4).toFixed(1)}" text-anchor="start" fill="${gainPp >= 0 ? "var(--color-positive)" : "var(--color-muted)"}" font-size="11.5" font-weight="600" font-family="var(--font-body)">${gainSign}${gainPp.toFixed(1)} pp</text>
      </g>
    `;
  }).join("");

  const legend = `
    <div class="chart-legend" style="margin-top:var(--space-xs)">
      <span class="legend-item">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" fill="var(--color-accent)"/></svg>
        <span><strong>MuRIL-DAPT v3</strong></span>
      </span>
      <span class="legend-item">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="4.2" fill="none" stroke="var(--color-positive)" stroke-width="2.2"/></svg>
        <span>Production Hybrid</span>
      </span>
      <span class="legend-item">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="3.8" fill="none" stroke="var(--color-muted)" stroke-width="1.8"/></svg>
        <span>LogReg</span>
      </span>
      <span class="legend-item">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><polygon points="6,1.5 10.5,6 6,10.5 1.5,6" fill="none" stroke="var(--color-warning)" stroke-width="1.6"/></svg>
        <span>ANN (MLP)</span>
      </span>
      <span class="legend-item">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="2" width="8" height="8" rx="1.5" fill="none" stroke="var(--color-info)" stroke-width="1.6"/></svg>
        <span>1D CNN</span>
      </span>
    </div>
  `;

  return `
    <div>
      <svg class="responsive-svg arch-dumbbell-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(subtitle)}">
        <text x="${pad.left}" y="16" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${esc(subtitle)}</text>
        <text x="${W - pad.right + 16}" y="16" text-anchor="start" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">MuRIL</text>
        <text x="${W - pad.right + 72}" y="16" text-anchor="start" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">vs. Best Base</text>
        ${grid}
        ${rows}
      </svg>
      ${legend}
    </div>
  `;
}

/* ==========================================================================
   7. 8-Category Auxiliary 54-Probe Stress Table (Restrained Tonal Contrast)
   ========================================================================== */
const PROBE_CAT_LABELS = {
  code_mixed_language: "Hinglish & code-mix",
  paraphrased_comments: "Paraphrased wording",
  informal_language: "Reddit slang & acronyms",
  parent_post_context: "Parent-post context",
  spelling_errors: "Spelling & typo noise",
  short_comments: "Terse short replies",
  ambiguous_responses: "Boundary / ambiguous",
  unseen_templates: "Unseen templates",
};

export function robustnessCategoryHeatmap(archComp) {
  const agg = archComp?.aggregate || {};
  const cats = Object.keys(PROBE_CAT_LABELS).filter((c) =>
    ARCH_SERIES.some((s) => agg[s.key]?.by_category_mean_f1?.[c] !== undefined)
  );
  if (!cats.length) return "";

  const rows = cats.map((cat) => {
    const scores = ARCH_SERIES.map((s) => Number(agg[s.key]?.by_category_mean_f1?.[cat] ?? 0));
    const maxScore = Math.max(...scores);
    const cells = ARCH_SERIES.map((s, idx) => {
      const val = scores[idx];
      const isWinner = val >= maxScore - 0.002;
      return `<td class="num hm-cell ${isWinner ? "hm-winner" : "hm-plain"}" title="${esc(PROBE_CAT_LABELS[cat])} · ${esc(s.short)}: ${(val * 100).toFixed(1)}%">${(val * 100).toFixed(1)}%</td>`;
    }).join("");
    const bestBaseline = Math.max(scores[0], scores[1], scores[2]);
    const transDelta = (scores[3] - bestBaseline) * 100;
    const deltaSign = transDelta >= 0 ? "+" : "";
    return `
      <tr>
        <th scope="row">${esc(PROBE_CAT_LABELS[cat])}</th>
        ${cells}
        <td class="num"><span class="delta-inline ${transDelta >= 0 ? "is-pos" : "is-muted"}">${deltaSign}${transDelta.toFixed(1)} pp</span></td>
      </tr>
    `;
  }).join("");

  return `
    <div class="table-scroll">
      <table class="data-table robustness-hm-table" aria-label="Mean 5-task F1 across all 8 unseen stress categories by architecture">
        <thead>
          <tr>
            <th scope="col">Linguistic stress category</th>
            ${ARCH_SERIES.map((s) => `<th scope="col" class="num">${esc(s.short)}</th>`).join("")}
            <th scope="col" class="num">MuRIL vs. Best Base</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

/* ==========================================================================
   8. DAPT Held-Out Validation Perplexity Curve (Single-Axis, Zero-Anchored)
   Eliminates redundant dual Y-axis (MLM loss = ln(perplexity) sits in tooltip).
   ========================================================================== */
export function daptConvergenceChart(daptSummary) {
  const hist = daptSummary?.history || [];
  if (!hist.length) return `<div class="chart-empty">DAPT epoch history not available.</div>`;

  const W = 520;
  const H = 236;
  const pad = { top: 34, right: 24, bottom: 38, left: 46 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const ppls = hist.map((h) => Number(h.perplexity));
  const maxPpl = niceAxisMax(Math.max(...ppls, 10), 4);
  const ticks = [0, Math.round(maxPpl * 0.25), Math.round(maxPpl * 0.5), Math.round(maxPpl * 0.75), maxPpl];

  const xOf = (i) => pad.left + (hist.length === 1 ? plotW / 2 : (i / (hist.length - 1)) * plotW);
  const yOf = (v) => pad.top + plotH - (Math.max(0, v) / maxPpl) * plotH;

  const pts = hist.map((h, i) => ({ x: xOf(i), y: yOf(Number(h.perplexity)), h }));
  const linePath = monotoneCubicPath(pts, pad.top, pad.top + plotH);
  const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} L ${pts[0].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} Z`;

  const initPpl = Number(daptSummary.initial_perplexity || ppls[0] || 0);
  const finalPpl = Number(daptSummary.final_perplexity || ppls[ppls.length - 1] || 0);
  const dropPct = initPpl > 0 ? ((initPpl - finalPpl) / initPpl) * 100 : 0;

  const grid = ticks.map((t, i) => {
    const y = yOf(t);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${i === 0 ? "none" : "2 4"}" opacity="${i === 0 ? "0.9" : "0.6"}"/>
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">${t}</text>
    `;
  }).join("");

  const dots = pts.map((p, i) => {
    const h = p.h;
    const epLabel = Number(h.epoch) === 0 ? "Pre-DAPT" : `Epoch ${h.epoch}`;
    const isFirstNearTop = i === 0 && p.y - 11 < pad.top + 8;
    const labelX = isFirstNearTop ? p.x + 22 : p.x;
    const labelY = isFirstNearTop ? p.y + 4 : (p.y - 11 < pad.top + 8 ? p.y + 18 : p.y - 11);
    const tipRows = JSON.stringify([
      ["Held-out Val Perplexity", Number(h.perplexity).toFixed(2)],
      ["Held-out MLM Loss", Number(h.mlm_loss).toFixed(4)],
      ["vs. Pre-DAPT", i === 0 ? "Baseline" : `${((Number(h.perplexity) - initPpl) / initPpl * 100).toFixed(1)}%`],
    ]);
    return `
      <g class="chart-point-hover"
         data-chart-tip-title="${esc(`MuRIL-DAPT · ${epLabel}`)}"
         data-chart-tip-rows="${esc(tipRows)}">
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="14" fill="transparent"/>
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="var(--color-accent)" stroke="var(--color-surface-solid)" stroke-width="2"/>
        <text x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="middle" fill="var(--color-ink)" font-size="11" font-weight="600" font-family="var(--font-body)">${Number(h.perplexity).toFixed(1)}</text>
        <text x="${p.x.toFixed(1)}" y="${(H - pad.bottom + 18).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">${epLabel}</text>
      </g>
    `;
  }).join("");

  const isHeldout = daptSummary?.perplexity_split === "heldout_validation";
  const caption = isHeldout
    ? `Held-out val perplexity (0-anchored · n = ${num(daptSummary.val_documents || 0)} unseen texts)`
    : "MLM Perplexity (lower = better domain fit)";

  return `
    <div>
      <svg class="responsive-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Single-axis zero-anchored held-out validation perplexity curve across DAPT epochs">
        <defs>
          <linearGradient id="dapt-ppl-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--color-accent)" stop-opacity="0.18"/>
            <stop offset="100%" stop-color="var(--color-accent)" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <text x="${pad.left}" y="15" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">${esc(caption)}</text>
        <text x="${W - pad.right}" y="15" text-anchor="end" fill="var(--color-positive)" font-size="11" font-weight="600" font-family="var(--font-body)">−${dropPct.toFixed(1)}%</text>
        ${grid}
        <path d="${areaPath}" fill="url(#dapt-ppl-grad)"/>
        <path d="${linePath}" fill="none" stroke="var(--color-accent)" stroke-width="2.4" stroke-linecap="round"/>
        ${dots}
      </svg>
      <p class="chart-footnote">Hover any epoch for held-out MLM cross-entropy loss (<code>${Number(hist[0].mlm_loss).toFixed(2)} → ${Number(hist[hist.length - 1].mlm_loss).toFixed(2)}</code>). Single axis avoids redundant dual-scale clutter.</p>
    </div>
  `;
}

/* ==========================================================================
   9. True 1:1 Square Calibration Reliability Diagram (Visible Bin Sample n=)
   Equal X/Y proportions guarantee a true 45-degree y = x calibration diagonal.
   ========================================================================== */
export function calibrationReliabilityChart(calData) {
  const bins = (calData?.bins || []).filter((b) => b.count > 0 && b.mean_confidence !== null);
  const rawBins = (calData?.uncalibrated_bins || []).filter((b) => b.count > 0 && b.mean_confidence !== null);
  if (!bins.length) return `<div class="chart-empty">Calibration bins not available.</div>`;

  // Square plotting area (220 x 220) so y = x is a true 45-degree diagonal
  const side = 204;
  const pad = { top: 28, right: 34, bottom: 42, left: 54 };
  const W = pad.left + side + pad.right;
  const H = pad.top + side + pad.bottom;

  const xOf = (v) => pad.left + Math.max(0, Math.min(1, v)) * side;
  const yOf = (v) => pad.top + side - Math.max(0, Math.min(1, v)) * side;

  const grid = [0, 0.25, 0.5, 0.75, 1.0].map((v, idx) => {
    const x = xOf(v);
    const y = yOf(v);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${pad.left + side}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${idx === 0 ? "none" : "2 4"}" opacity="0.6"/>
      <line x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${pad.top + side}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${idx === 0 ? "none" : "2 4"}" opacity="0.45"/>
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">${Math.round(v * 100)}%</text>
      <text x="${x.toFixed(1)}" y="${(pad.top + side + 16).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">${Math.round(v * 100)}%</text>
    `;
  }).join("");

  const calPts = bins.map((b) => `${xOf(b.mean_confidence).toFixed(1)},${yOf(b.empirical_accuracy).toFixed(1)}`).join(" ");
  const rawPts = rawBins.map((b) => `${xOf(b.mean_confidence).toFixed(1)},${yOf(b.empirical_accuracy).toFixed(1)}`).join(" ");

  // Gap whiskers visually show exact |accuracy - confidence| distance to y = x
  const whiskers = bins.map((b) => {
    const x = xOf(b.mean_confidence);
    const yAcc = yOf(b.empirical_accuracy);
    const yIdeal = yOf(b.mean_confidence);
    return `<line x1="${x.toFixed(1)}" y1="${yAcc.toFixed(1)}" x2="${x.toFixed(1)}" y2="${yIdeal.toFixed(1)}" stroke="var(--color-positive)" stroke-width="1.4" opacity="0.4"/>`;
  }).join("");

  const calDots = bins.map((b) => {
    const cx = xOf(b.mean_confidence);
    const cy = yOf(b.empirical_accuracy);
    const r = Math.min(7.5, 3.2 + Math.sqrt(b.count) * 0.55);
    const aboveIdeal = b.empirical_accuracy >= b.mean_confidence;
    const labelY = aboveIdeal ? Math.max(pad.top + 11, cy - r - 6) : Math.min(pad.top + side - 5, cy + r + 12);
    const gap = Math.abs(b.empirical_accuracy - b.mean_confidence) * 100;
    const tipRows = JSON.stringify([
      ["Mean predicted confidence", `${(b.mean_confidence * 100).toFixed(1)}%`],
      ["Empirical accuracy", `${(b.empirical_accuracy * 100).toFixed(1)}%`],
      ["Calibration gap", `${gap.toFixed(1)} pp`],
      ["Sample count", `n = ${num(b.count)}`],
    ]);
    return `
      <g class="chart-point-hover"
         data-chart-tip-title="${esc(`Confidence bin ${Math.round(b.lower * 100)}–${Math.round(b.upper * 100)}%`)}"
         data-chart-tip-rows="${esc(tipRows)}">
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="13" fill="transparent"/>
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="var(--color-positive)" stroke="var(--color-surface-solid)" stroke-width="1.6"/>
        <text x="${cx.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="9.5" font-weight="500" font-family="var(--font-body)">n=${b.count}</text>
      </g>
    `;
  }).join("");

  return `
    <div class="reliability-wrap">
      <svg class="reliability-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Square 1:1 calibration reliability diagram comparing predicted confidence against empirical accuracy with bin sample sizes">
        <text x="${(pad.left + side / 2).toFixed(1)}" y="14" text-anchor="middle" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">Empirical accuracy vs. confidence (1:1)</text>
        ${grid}
        <text x="${pad.left + 8}" y="${pad.top + 16}" fill="var(--color-muted)" font-size="9.5" opacity="0.7" font-family="var(--font-body)">Underconfident</text>
        <text x="${pad.left + side - 8}" y="${pad.top + side - 8}" text-anchor="end" fill="var(--color-muted)" font-size="9.5" opacity="0.7" font-family="var(--font-body)">Overconfident</text>
        <line x1="${xOf(0).toFixed(1)}" y1="${yOf(0).toFixed(1)}" x2="${xOf(1).toFixed(1)}" y2="${yOf(1).toFixed(1)}" stroke="var(--color-muted)" stroke-width="1.2" stroke-dasharray="4,4" opacity="0.7"/>
        ${whiskers}
        ${rawPts ? `<polyline points="${rawPts}" fill="none" stroke="var(--color-warning)" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.75"/>` : ""}
        <polyline points="${calPts}" fill="none" stroke="var(--color-positive)" stroke-width="2.2" stroke-linejoin="round"/>
        ${calDots}
        <text x="${(pad.left + side / 2).toFixed(1)}" y="${(H - 6).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">Predicted confidence (bin mean)</text>
      </svg>
      <div class="chart-legend" style="margin-top:var(--space-xs)">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-positive)"></span><span>Calibrated (ECE ${Number(calData.calibrated_ece ?? 0).toFixed(3)}, T=${esc(calData.temperature)})</span></span>
        ${calData.uncalibrated_ece !== undefined ? `<span class="legend-item"><span class="legend-swatch" style="background:var(--color-warning)"></span><span>Raw T=1.0 (ECE ${Number(calData.uncalibrated_ece).toFixed(3)})</span></span>` : ""}
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-muted)"></span><span>Ideal (y = x)</span></span>
      </div>
    </div>
  `;
}

/* ==========================================================================
   10. Corpus Confidence Distribution Histogram (Honest 0-Anchored Y-Axis)
   Calm background zone washes + clean headroom counts + zero floating boxes.
   ========================================================================== */
export function confidenceHistogramChart(confSummary) {
  const hist = confSummary?.histogram || [];
  if (!hist.length) return "";
  const W = 520;
  const H = 236;
  const pad = { top: 34, right: 16, bottom: 38, left: 46 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const totalCount = Math.max(1, hist.reduce((s, b) => s + Number(b.count || 0), 0));
  const rawMax = Math.max(1, ...hist.map((b) => Number(b.count || 0)));
  const yMax = niceAxisMax(rawMax, 4);
  const ticks = [0, Math.round(yMax * 0.5), yMax];

  const slotW = plotW / hist.length;
  const barW = slotW - 8;
  const xLow = pad.left + 5.5 * slotW;
  const xHigh = pad.left + 8.0 * slotW;
  const yOf = (v) => pad.top + plotH - (v / yMax) * plotH;

  const tierColor = (tier) =>
    tier === "high" ? "var(--color-positive)" : tier === "medium" ? "var(--color-accent)" : "var(--color-warning)";

  const grid = ticks.map((t, i) => {
    const y = yOf(t);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="${i === 0 ? "none" : "2 4"}" opacity="${i === 0 ? "0.9" : "0.55"}"/>
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="10.5" font-family="var(--font-body)">${num(t)}</text>
    `;
  }).join("");

  const bars = hist.map((b, i) => {
    const cnt = Number(b.count || 0);
    const bh = (cnt / yMax) * plotH;
    const x = pad.left + i * slotW + 4;
    const y = pad.top + plotH - bh;
    const cx = x + barW / 2;
    const tipRows = JSON.stringify([
      ["Comments", `${num(cnt)} (${share(cnt, totalCount)})`],
      ["Confidence tier", pretty(b.tier)],
    ]);
    return `
      <g class="chart-bar-hover"
         data-chart-tip-title="${esc(`Confidence ${b.label}`)}"
         data-chart-tip-rows="${esc(tipRows)}">
        <rect x="${(pad.left + i * slotW).toFixed(1)}" y="${pad.top}" width="${slotW.toFixed(1)}" height="${plotH}" fill="transparent"/>
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(cnt > 0 ? 2 : 0, bh).toFixed(1)}" rx="3" fill="${tierColor(b.tier)}" opacity="0.86"/>
        ${cnt > 0 ? `<text x="${cx.toFixed(1)}" y="${(y - 6).toFixed(1)}" text-anchor="middle" fill="var(--color-ink-2)" font-size="9.5" font-weight="500" font-family="var(--font-body)">${num(cnt)}</text>` : ""}
        <text x="${cx.toFixed(1)}" y="${(H - pad.bottom + 16).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="10" font-family="var(--font-body)">${Math.round(b.lower * 100)}%</text>
      </g>
    `;
  }).join("");

  return `
    <div>
      <svg class="responsive-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Zero-anchored histogram of calibrated prediction confidence across all live corpus comments">
        <text x="${pad.left}" y="15" fill="var(--color-muted)" font-size="11" font-family="var(--font-body)">Live comments by mean 5-task confidence (n = ${num(totalCount)})</text>
        <!-- Subtle zone background tints -->
        <rect x="${pad.left}" y="${pad.top}" width="${(xLow - pad.left).toFixed(1)}" height="${plotH}" fill="var(--color-warning)" opacity="0.06"/>
        <rect x="${xHigh.toFixed(1)}" y="${pad.top}" width="${(W - pad.right - xHigh).toFixed(1)}" height="${plotH}" fill="var(--color-positive)" opacity="0.06"/>
        ${grid}
        <line x1="${xLow.toFixed(1)}" y1="${pad.top}" x2="${xLow.toFixed(1)}" y2="${(pad.top + plotH).toFixed(1)}" stroke="var(--color-warning)" stroke-width="1" stroke-dasharray="3,3" opacity="0.7"/>
        <text x="${(xLow - 5).toFixed(1)}" y="${pad.top + 12}" text-anchor="end" fill="var(--color-warning)" font-size="9.5" font-weight="600" font-family="var(--font-body)">Review &lt;55%</text>
        <line x1="${xHigh.toFixed(1)}" y1="${pad.top}" x2="${xHigh.toFixed(1)}" y2="${(pad.top + plotH).toFixed(1)}" stroke="var(--color-positive)" stroke-width="1" stroke-dasharray="3,3" opacity="0.7"/>
        <text x="${(xHigh + 5).toFixed(1)}" y="${pad.top + 12}" text-anchor="start" fill="var(--color-positive)" font-size="9.5" font-weight="600" font-family="var(--font-body)">High &ge;80%</text>
        ${bars}
      </svg>
      <div class="chart-legend" style="margin-top:var(--space-xs)">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-positive)"></span><span>High &ge;80% (${num(confSummary.high_count)})</span></span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent)"></span><span>Medium 55–79% (${num(confSummary.medium_count)})</span></span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-warning)"></span><span>Review &lt;55% (${num(confSummary.low_count)})</span></span>
      </div>
    </div>
  `;
}

/* ==========================================================================
   11. Honest Linear Issue Triage Visual (0%–100% Paired Bars)
   Replaces the non-linear sqrt-X + zoomed-Y bubble cloud with a single unified,
   zero-distortion linear comparison of Negative Sentiment % and Unconfirmed %.
   ========================================================================== */
export function issueFrictionQuadrantChart(matrix) {
  if (!matrix || !matrix.length) return "";

  const bandColor = (b) =>
    b === "critical"
      ? "var(--color-negative)"
      : b === "elevated"
        ? "var(--color-warning)"
        : b === "moderate"
          ? "var(--color-accent)"
          : "var(--color-positive)";

  const rows = matrix.map((m) => {
    const negPct = Math.max(0, Math.min(100, Number(m.negative_share || 0) * 100));
    const unresPct = Math.max(0, Math.min(100, Number(m.unresolved_share || 0) * 100));
    const col = bandColor(m.priority_band);
    const tipRows = JSON.stringify([
      ["Friction index", `${Number(m.friction_score || 0).toFixed(1)} / 100 (${pretty(m.priority_band)})`],
      ["Negative sentiment", `${negPct.toFixed(1)}% of comments`],
      ["Unconfirmed resolution", `${unresPct.toFixed(0)}% of threads`],
      ["Volume", `${plural(m.posts, "thread")} · ${plural(m.comments, "comment")}`],
      ["Dominant emotion", pretty(m.dominant_emotion)],
      ["Mean confidence", percent(m.mean_confidence)],
      ["Top NMF cluster", m.top_topic || "—"],
    ]);
    return `
      <a class="triage-row"
         href="#/issues?selected=${encodeURIComponent(m.key)}"
         data-chart-tip-title="${esc(m.label)}"
         data-chart-tip-rows="${esc(tipRows)}"
         data-chart-tip-hint="Click to open Issues deep dive">
        <div class="triage-col-issue">
          <span class="triage-title">${esc(m.label)}</span>
          <span class="triage-meta num">${plural(m.posts, "thread")} · ${plural(m.comments, "comment")} · ${esc(pretty(m.dominant_emotion))}</span>
        </div>

        <div class="triage-col-bar">
          <div class="triage-bar-track" aria-hidden="true">
            <span class="triage-tick" style="left:25%"></span>
            <span class="triage-tick" style="left:50%"></span>
            <span class="triage-tick" style="left:75%"></span>
            <span class="triage-bar-fill" style="width:${negPct.toFixed(1)}%;background:var(--color-negative)"></span>
          </div>
          <span class="triage-val num">${negPct.toFixed(1)}%</span>
        </div>

        <div class="triage-col-bar">
          <div class="triage-bar-track" aria-hidden="true">
            <span class="triage-tick" style="left:25%"></span>
            <span class="triage-tick" style="left:50%"></span>
            <span class="triage-tick" style="left:75%"></span>
            <span class="triage-bar-fill" style="width:${unresPct.toFixed(1)}%;background:var(--color-series-val)"></span>
          </div>
          <span class="triage-val num">${unresPct.toFixed(0)}%</span>
        </div>

        <div class="triage-col-score num">
          <span class="swatch-inline" style="background:${col}" aria-hidden="true"></span>
          <strong>${Number(m.friction_score || 0).toFixed(0)}</strong>
        </div>
      </a>
    `;
  }).join("");

  return `
    <div class="triage-matrix" role="region" aria-label="Issue friction triage matrix on honest 0 to 100 percent linear scales">
      <div class="triage-head">
        <span>Issue domain &amp; dominant emotion</span>
        <span>Negative sentiment (0–100%)</span>
        <span>Unconfirmed resolution (0–100%)</span>
        <span class="triage-head-right">Friction</span>
      </div>
      <div class="triage-body">${rows}</div>
      <div class="triage-foot">
        <div class="chart-legend">
          <span class="legend-item"><span class="legend-swatch" style="background:var(--color-negative)"></span><span>Critical (&ge;62)</span></span>
          <span class="legend-item"><span class="legend-swatch" style="background:var(--color-warning)"></span><span>Elevated (48–61)</span></span>
          <span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent)"></span><span>Moderate (34–47)</span></span>
          <span class="legend-item"><span class="legend-swatch" style="background:var(--color-positive)"></span><span>Healthy (&lt;34)</span></span>
        </div>
        <span class="muted" style="font-size:var(--text-xs)">All tracks span true 0%–100% linear scale · Friction = 40% Neg + 35% Unconfirmed + 25% Volume</span>
      </div>
    </div>
  `;
}

/* ==========================================================================
   12. Interactive Tripartite Community & Discourse Network (Overview)
   100% observed co-occurrence counts (zero marginal guesses, zero hairball):
   Layer 1: Subreddit Communities (threads) -> Layer 2: Issue Domains (hubs)
   -> Layer 3: Canonical Response Roles OR Multi-Label Emotions (comments).
   Hover or click any node to isolate its true observed connections.
   ========================================================================== */
export function communityNetworkDiagram(network, mode = "responses") {
  const nodes = network?.nodes || {};
  const edges = network?.edges || {};
  const subs = nodes.communities || [];
  const issues = nodes.issues || [];
  const rightNodes = mode === "emotions" ? (nodes.emotions || []) : (nodes.responses || []);
  const leftEdges = edges.community_to_issue || [];
  const rightEdges = mode === "emotions" ? (edges.issue_to_emotion || []) : (edges.issue_to_response || []);

  if (!subs.length || !issues.length || !rightNodes.length) {
    return `<div class="chart-empty">Community co-occurrence network data unavailable.</div>`;
  }

  const W = 1040;
  const H = 446;
  const topY = 58;
  const botY = 414;
  const spanY = botY - topY;

  const pillW = 264;
  const pillH = 40;
  const pillLeftX = 388;
  const pillRightX = pillLeftX + pillW;
  const leftBaseX = 204;
  const rightBaseX = 836;

  const maxSubPosts = Math.max(1, ...subs.map((s) => Number(s.posts || 0)));
  const maxIssComments = Math.max(1, ...issues.map((i) => Number(i.comments || 0)));
  const maxRightComments = Math.max(1, ...rightNodes.map((r) => Number(r.comments || 0)));
  const maxLeftW = Math.max(1, ...leftEdges.map((e) => Number(e.threads || 0)));
  const maxRightW = Math.max(1, ...rightEdges.map((e) => Number(e.comments || 0)));

  const yAt = (idx, len) => (len <= 1 ? (topY + botY) / 2 : topY + (idx / (len - 1)) * spanY);

  // Subtle outward bow on outer node columns so curves fan cleanly without crowding
  const subLayout = subs.map((s, idx) => {
    const y = yAt(idx, subs.length);
    const t = subs.length > 1 ? idx / (subs.length - 1) : 0.5;
    const x = leftBaseX - Math.sin(t * Math.PI) * 14;
    const r = Math.max(4.5, Math.sqrt(Number(s.posts || 1) / maxSubPosts) * 11.5);
    return {
      ...s,
      layer: "community",
      x,
      y,
      r,
      color: "var(--color-info)",
      href: `#/explorer?community=${encodeURIComponent(s.key)}`,
      metricText: plural(s.posts, "thread"),
      subText: `${plural(s.posts, "thread")} · ${plural(s.comments, "comment")}`,
    };
  });

  const issueLayout = issues.map((iss, idx) => {
    const y = yAt(idx, issues.length);
    const dotR = Math.max(3.8, Math.sqrt(Number(iss.comments || 1) / maxIssComments) * 7.2);
    return {
      ...iss,
      layer: "issue",
      x: pillLeftX + pillW / 2,
      y,
      r: dotR,
      color: idx === 0 ? "var(--color-accent)" : idx === 1 ? "var(--color-accent-2)" : "var(--color-series-val)",
      href: `#/issues?selected=${encodeURIComponent(iss.key)}`,
      metricText: plural(iss.comments, "comment"),
      subText: `${plural(iss.posts, "thread")} · ${plural(iss.comments, "comment")}`,
    };
  });

  const rightLayout = rightNodes.map((rn, idx) => {
    const y = yAt(idx, rightNodes.length);
    const t = rightNodes.length > 1 ? idx / (rightNodes.length - 1) : 0.5;
    const x = rightBaseX + Math.sin(t * Math.PI) * 14;
    const r = Math.max(4.5, Math.sqrt(Number(rn.comments || 1) / maxRightComments) * 11.5);
    const color = mode === "emotions"
      ? (EMOTION_COLORS[rn.key] || "var(--color-accent)")
      : (RESPONSE_META[rn.key]?.color || "var(--color-accent)");
    const displayLabel = mode === "emotions"
      ? pretty(rn.key)
      : `${idx + 1}. ${RESPONSE_META[rn.key]?.label || pretty(rn.key)}`;
    const href = mode === "emotions"
      ? `#/explorer?emotion=${encodeURIComponent(rn.key)}`
      : `#/explorer?response=${encodeURIComponent(rn.key)}`;
    return {
      ...rn,
      displayLabel,
      layer: mode === "emotions" ? "emotion" : "response",
      x,
      y,
      r,
      color,
      href,
      metricText: plural(rn.comments, "comment"),
      subText: `${plural(rn.comments, "comment")}`,
    };
  });

  const nodeById = new Map([
    ...subLayout.map((n) => [n.id, n]),
    ...issueLayout.map((n) => [n.id, n]),
    ...rightLayout.map((n) => [n.id, n]),
  ]);

  // Build adjacency summary per node for the live connection inspector bar
  const adjByNode = new Map();
  const pushAdj = (nodeId, peerLabel, valueLabel, weightNum, drillHref) => {
    if (!adjByNode.has(nodeId)) adjByNode.set(nodeId, []);
    adjByNode.get(nodeId).push({ peerLabel, valueLabel, weightNum, drillHref });
  };

  const renderedLeftEdges = leftEdges.map((e) => {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) return "";
    const x1 = s.x + s.r + 2;
    const y1 = s.y;
    const x2 = pillLeftX;
    const y2 = t.y;
    const cx1 = x1 + (x2 - x1) * 0.48;
    const cx2 = x2 - (x2 - x1) * 0.48;
    const sw = 1.1 + Math.sqrt(Number(e.threads || 0) / maxLeftW) * 6.0;
    const major = Number(e.share_of_source || 0) >= 0.15 || Number(e.threads || 0) >= 10;
    const sharePct = `${(Number(e.share_of_source || 0) * 100).toFixed(0)}%`;
    const drillHref = `#/explorer?community=${encodeURIComponent(e.source_key)}&issue=${encodeURIComponent(e.target_key)}`;
    pushAdj(s.id, pretty(t.label), `${plural(e.threads, "thread")} (${sharePct})`, Number(e.threads), drillHref);
    pushAdj(t.id, s.label, `${plural(e.threads, "thread")}`, Number(e.threads), drillHref);

    const tipRows = JSON.stringify([
      ["Observed threads", `${num(e.threads)} (${sharePct} of ${s.label})`],
      ["Observed comments", num(e.comments)],
    ]);
    return `
      <path
        class="net-edge ${major ? "is-major" : "is-minor"}"
        data-source="${esc(e.source)}"
        data-target="${esc(e.target)}"
        data-href="${esc(drillHref)}"
        data-chart-tip-title="${esc(`${s.label} → ${pretty(t.label)}`)}"
        data-chart-tip-rows="${esc(tipRows)}"
        data-chart-tip-hint="Click edge to filter matching threads"
        d="M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${cx1.toFixed(1)} ${y1.toFixed(1)}, ${cx2.toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}"
        stroke="var(--color-info)"
        stroke-width="${sw.toFixed(1)}"
      />
    `;
  }).join("");

  const renderedRightEdges = rightEdges.map((e) => {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) return "";
    const x1 = pillRightX;
    const y1 = s.y;
    const x2 = t.x - t.r - 2;
    const y2 = t.y;
    const cx1 = x1 + (x2 - x1) * 0.48;
    const cx2 = x2 - (x2 - x1) * 0.48;
    const sw = 1.1 + Math.sqrt(Number(e.comments || 0) / maxRightW) * 6.5;
    const major = Number(e.share_of_source || 0) >= 0.14 || Number(e.comments || 0) >= 70;
    const sharePct = `${(Number(e.share_of_source || 0) * 100).toFixed(0)}%`;
    const drillHref = mode === "emotions"
      ? `#/explorer?issue=${encodeURIComponent(e.source_key)}&emotion=${encodeURIComponent(e.target_key)}`
      : `#/explorer?issue=${encodeURIComponent(e.source_key)}&response=${encodeURIComponent(e.target_key)}`;
    pushAdj(s.id, t.displayLabel, `${plural(e.comments, "comment")} (${sharePct})`, Number(e.comments), drillHref);
    pushAdj(t.id, pretty(s.label), `${plural(e.comments, "comment")}`, Number(e.comments), drillHref);

    const tipRows = JSON.stringify([
      ["Observed comments", `${num(e.comments)} (${sharePct} of ${pretty(s.label)})`],
      ["Target class", t.displayLabel],
    ]);
    return `
      <path
        class="net-edge ${major ? "is-major" : "is-minor"}"
        data-source="${esc(e.source)}"
        data-target="${esc(e.target)}"
        data-href="${esc(drillHref)}"
        data-chart-tip-title="${esc(`${pretty(s.label)} → ${t.displayLabel}`)}"
        data-chart-tip-rows="${esc(tipRows)}"
        data-chart-tip-hint="Click edge to filter matching threads"
        d="M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${cx1.toFixed(1)} ${y1.toFixed(1)}, ${cx2.toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}"
        stroke="${t.color}"
        stroke-width="${sw.toFixed(1)}"
      />
    `;
  }).join("");

  const encodeSummary = (n) => {
    const links = (adjByNode.get(n.id) || [])
      .sort((a, b) => b.weightNum - a.weightNum)
      .slice(0, 4);
    return JSON.stringify({
      label: n.displayLabel || pretty(n.label),
      subText: n.subText,
      href: n.href,
      links,
    });
  };

  const renderOuterNode = (n, side) => {
    const isLeft = side === "left";
    const labelX = isLeft ? n.x - n.r - 12 : n.x + n.r + 12;
    const anchor = isLeft ? "end" : "start";
    const labelText = n.displayLabel || n.label;
    return `
      <g class="net-node"
         tabindex="0"
         role="button"
         data-net-node="${esc(n.id)}"
         data-net-summary="${esc(encodeSummary(n))}"
         data-href="${esc(n.href)}"
         aria-label="${esc(labelText)}: ${esc(n.subText)}">
        <circle class="net-node-halo" cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${(n.r + 5).toFixed(1)}" fill="none" stroke="${n.color}" stroke-width="1.5"/>
        <circle class="net-node-dot" cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${n.r.toFixed(1)}" fill="${n.color}" stroke="var(--color-surface-solid)" stroke-width="2"/>
        <text x="${labelX.toFixed(1)}" y="${(n.y - 2).toFixed(1)}" text-anchor="${anchor}" class="net-label">${esc(labelText)}</text>
        <text x="${labelX.toFixed(1)}" y="${(n.y + 12).toFixed(1)}" text-anchor="${anchor}" class="net-sub">${esc(n.metricText)}</text>
      </g>
    `;
  };

  const renderCenterPill = (n) => {
    const py = n.y - pillH / 2;
    const dotCx = pillLeftX + 16;
    const labelX = pillLeftX + 30;
    const shortTitle = pretty(n.label).length > 36 ? `${pretty(n.label).slice(0, 35)}…` : pretty(n.label);
    return `
      <g class="net-node net-node-hub"
         tabindex="0"
         role="button"
         data-net-node="${esc(n.id)}"
         data-net-summary="${esc(encodeSummary(n))}"
         data-href="${esc(n.href)}"
         aria-label="${esc(pretty(n.label))}: ${esc(n.subText)}">
        <rect class="net-hub-rect" x="${pillLeftX}" y="${py.toFixed(1)}" width="${pillW}" height="${pillH}" rx="10" fill="var(--color-surface-solid)" stroke="var(--color-rule-2)" stroke-width="1"/>
        <circle cx="${dotCx}" cy="${n.y.toFixed(1)}" r="${n.r.toFixed(1)}" fill="${n.color}"/>
        <text x="${labelX}" y="${(n.y - 2).toFixed(1)}" text-anchor="start" class="net-label">${esc(shortTitle)}</text>
        <text x="${labelX}" y="${(n.y + 12).toFixed(1)}" text-anchor="start" class="net-sub">${esc(n.subText)}</text>
      </g>
    `;
  };

  const rightHeader = mode === "emotions"
    ? `Emotions (${rightLayout.length} · comments)`
    : `Response roles (${rightLayout.length} · comments)`;

  return `
    <div class="network-wrap" data-network-root>
      <div class="network-scroll">
        <svg class="network-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Interactive observed co-occurrence network linking ${subs.length} subreddits through ${issues.length} issue domains to ${rightLayout.length} ${mode}">
          <text x="${leftBaseX}" y="22" text-anchor="end" class="net-col-head">Communities (${subs.length} · threads)</text>
          <text x="${pillLeftX + pillW / 2}" y="22" text-anchor="middle" class="net-col-head">Issue domains (${issues.length} hubs)</text>
          <text x="${rightBaseX}" y="22" text-anchor="start" class="net-col-head">${esc(rightHeader)}</text>

          <g class="net-edges">
            ${renderedLeftEdges}
            ${renderedRightEdges}
          </g>

          <g class="net-nodes">
            ${subLayout.map((n) => renderOuterNode(n, "left")).join("")}
            ${issueLayout.map(renderCenterPill).join("")}
            ${rightLayout.map((n) => renderOuterNode(n, "right")).join("")}
          </g>
        </svg>
      </div>

      <div class="network-inspector-bar" data-net-inspector>
        <span class="muted">Hover any node or link to spotlight its exact observed connections · Click a node to pin focus, or click any edge to filter threads.</span>
      </div>

      <p class="chart-footnote">Every edge width represents exact observed counts (left: ${num(network.totals?.threads || 0)} threads; right: ${mode === "emotions" ? `${num(network.totals?.emotion_mentions || 0)} multi-label emotion tags` : `${num(network.totals?.comments || 0)} classified comments`}). Minor edges (&lt;15% share) stay muted until hovered to avoid hairball clutter.</p>
    </div>
  `;
}

export function bindNetworkInteractions(container) {
  const root = container.querySelector("[data-network-root]");
  if (!root) return;

  const nodes = [...root.querySelectorAll("[data-net-node]")];
  const edges = [...root.querySelectorAll(".net-edge")];
  const inspector = root.querySelector("[data-net-inspector]");
  const defaultInspectorHtml = inspector ? inspector.innerHTML : "";
  let pinnedNodeId = null;

  const paintInspector = (nodeEl, isPinned) => {
    if (!inspector || !nodeEl) return;
    try {
      const info = JSON.parse(nodeEl.dataset.netSummary || "{}");
      const chips = (info.links || []).map((lk) => `
        <a class="net-link-chip" href="${esc(lk.drillHref)}">
          <span>${esc(lk.peerLabel)}</span>
          <strong class="num">${esc(lk.valueLabel)}</strong>
        </a>
      `).join("");
      inspector.innerHTML = `
        <div class="net-insp-head">
          <div>
            <strong>${esc(info.label)}</strong>
            <span class="muted num"> · ${esc(info.subText)}</span>
          </div>
          <div class="net-insp-actions">
            ${isPinned ? `<button type="button" class="btn btn-quiet btn-sm" data-net-clear>Clear pin</button>` : ""}
            <a class="row-open-link" href="${esc(info.href)}">Open slice &rarr;</a>
          </div>
        </div>
        ${chips ? `<div class="net-insp-chips">${chips}</div>` : ""}
      `;
      inspector.querySelector("[data-net-clear]")?.addEventListener("click", () => {
        pinnedNodeId = null;
        resetVisuals();
      });
    } catch {
      // Keep quiet on malformed dataset
    }
  };

  const resetVisuals = () => {
    nodes.forEach((n) => n.classList.remove("is-dimmed", "is-active", "is-pinned"));
    edges.forEach((e) => e.classList.remove("is-dimmed", "is-active"));
    if (inspector) inspector.innerHTML = defaultInspectorHtml;
    hideVizTooltip();
  };

  const focusNode = (nodeId, isPinned = false) => {
    const connected = new Set([nodeId]);
    edges.forEach((e) => {
      const hit = e.dataset.source === nodeId || e.dataset.target === nodeId;
      e.classList.toggle("is-active", hit);
      e.classList.toggle("is-dimmed", !hit);
      if (hit) {
        connected.add(e.dataset.source);
        connected.add(e.dataset.target);
      }
    });
    let activeEl = null;
    nodes.forEach((n) => {
      const id = n.dataset.netNode;
      const isSelf = id === nodeId;
      if (isSelf) activeEl = n;
      n.classList.toggle("is-active", isSelf);
      n.classList.toggle("is-pinned", isPinned && isSelf);
      n.classList.toggle("is-dimmed", !connected.has(id));
    });
    if (activeEl) paintInspector(activeEl, isPinned);
  };

  nodes.forEach((node) => {
    const id = node.dataset.netNode;
    node.addEventListener("mouseenter", () => {
      if (!pinnedNodeId) focusNode(id, false);
    });
    node.addEventListener("mouseleave", () => {
      if (pinnedNodeId) focusNode(pinnedNodeId, true);
      else resetVisuals();
    });
    node.addEventListener("focus", () => {
      if (!pinnedNodeId) focusNode(id, false);
    });
    node.addEventListener("blur", () => {
      if (pinnedNodeId) focusNode(pinnedNodeId, true);
      else resetVisuals();
    });
    node.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pinnedNodeId === id) {
        pinnedNodeId = null;
        resetVisuals();
      } else {
        pinnedNodeId = id;
        focusNode(id, true);
      }
    });
    node.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (pinnedNodeId === id) {
          pinnedNodeId = null;
          resetVisuals();
        } else {
          pinnedNodeId = id;
          focusNode(id, true);
        }
      }
    });
  });

  edges.forEach((edge) => {
    edge.addEventListener("mouseenter", () => {
      if (pinnedNodeId) return;
      const src = edge.dataset.source;
      const dst = edge.dataset.target;
      edges.forEach((other) => {
        other.classList.toggle("is-active", other === edge);
        other.classList.toggle("is-dimmed", other !== edge);
      });
      nodes.forEach((n) => {
        const hit = n.dataset.netNode === src || n.dataset.netNode === dst;
        n.classList.toggle("is-dimmed", !hit);
        n.classList.toggle("is-active", hit);
      });
    });
    edge.addEventListener("mouseleave", () => {
      if (pinnedNodeId) focusNode(pinnedNodeId, true);
      else resetVisuals();
    });
    edge.addEventListener("click", () => {
      if (edge.dataset.href) window.location.hash = edge.dataset.href;
    });
  });

  root.querySelector(".network-svg")?.addEventListener("click", () => {
    if (pinnedNodeId) {
      pinnedNodeId = null;
      resetVisuals();
    }
  });

  bindChartTooltips(root);
}
