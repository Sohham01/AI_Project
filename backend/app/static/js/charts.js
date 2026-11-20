/**
 * Interactive Token-Driven SVG Visualizations for RCI Dusk Studio.
 * 1. Interactive 3-stage Sankey Flow (Issues -> 6 Canonical Responses -> 4 Resolution States)
 * 2. Smooth Multi-Series Spline Area Chart with live crosshair scrubbing & series filter
 * 3. Interactive NMF Topic Constellation (8 clickable script-colored proportional orbs)
 * 4. Animated Radial Canonical Response Wheel (6 fixed classes)
 * 5. Animated Stacked Bars, Horizontal Bars, Benchmark Columns & Confusion Heatmap
 */
import {
  CANONICAL_RESPONSE_TYPES,
  RESOLUTION_META,
  RESOLUTION_ORDER,
  RESPONSE_META,
  SENTIMENT_META,
  SENTIMENT_ORDER,
  esc,
  monthLabel,
  num,
  percent,
  pretty,
  share,
} from "./format.js";
import { hideVizTooltip, showVizTooltip } from "./motion.js";

/* ==========================================================================
   1. Interactive 3-Stage Sankey Research Flow (Overview Signature Visual)
   ========================================================================== */
export function sankeyFlowChart(summary) {
  const issues = [...(summary.issue_distribution || [])].sort(
    (a, b) => Number(b.actionable_recommendations || 0) - Number(a.actionable_recommendations || 0)
  );
  const respMap = Object.fromEntries(
    (summary.response_distribution || []).map((r) => [r.label, Number(r.count || 0)])
  );
  const resMap = Object.fromEntries(
    (summary.resolution_summary || []).map((r) => [r.label, Number(r.count || 0)])
  );

  const W = 1120;
  const H = 396;
  const topPad = 42;
  const botPad = 18;
  const usableH = H - topPad - botPad;
  const nodeW = 14;
  const midPillW = 188;
  const colX = { issue: 248, response: 468, resolution: 882 };

  function layoutNodes(list, x, gap) {
    const total = list.reduce((s, n) => s + n.weight, 0) || 1;
    const avail = usableH - gap * Math.max(0, list.length - 1);
    let y = topPad;
    return list.map((item) => {
      const h = Math.max(18, (item.weight / total) * avail);
      const node = { ...item, x, y, h, cy: y + h / 2 };
      y += h + gap;
      return node;
    });
  }

  const totalComments = Number(summary.comments || 1);
  const totalPosts = Number(summary.posts || 1);
  const issueNodes = layoutNodes(
    issues.map((i, idx) => ({
      id: `iss-${i.key}`,
      col: "issue",
      key: i.key,
      label: pretty(i.label),
      sub: `${num(i.actionable_recommendations)} actionable`,
      weight: Math.max(1, Number(i.comments || 1)),
      color: idx === 0 ? "var(--color-accent)" : idx < 3 ? "var(--color-info)" : "var(--color-series-val)",
      href: `#/issues?selected=${encodeURIComponent(i.key)}`,
      detail: `${num(i.posts)} threads, ${num(i.comments)} comments (${num(i.solutions_proposed)} solution, ${num(i.advice_comments)} advice, ${num(i.reported_successes)} reported success)`,
    })),
    colX.issue,
    7
  );

  const responseNodes = layoutNodes(
    CANONICAL_RESPONSE_TYPES.map((k, idx) => ({
      id: `resp-${k}`,
      col: "response",
      key: k,
      label: `${idx + 1}. ${RESPONSE_META[k].label}`,
      sub: `${num(respMap[k] || 0)} comments`,
      weight: respMap[k] || 1,
      color: RESPONSE_META[k].color,
      href: `#/explorer?response=${encodeURIComponent(k)}`,
      detail: `${num(respMap[k] || 0)} predicted comments (${share(respMap[k] || 0, totalComments)}) in canonical slot #${idx + 1}`,
    })),
    colX.response,
    10
  );

  const resolutionNodes = layoutNodes(
    RESOLUTION_ORDER.map((k) => ({
      id: `res-${k}`,
      col: "resolution",
      key: k,
      label: RESOLUTION_META[k].label,
      sub: `${num(resMap[k] || 0)} threads`,
      weight: Math.max(1, (resMap[k] || 0) * 8),
      color: RESOLUTION_META[k].color,
      href: `#/resolution?state=${encodeURIComponent(k)}`,
      detail: `${num(resMap[k] || 0)} of ${num(totalPosts)} threads (${share(resMap[k] || 0, totalPosts)}): ${RESOLUTION_META[k].description}`,
    })),
    colX.resolution,
    16
  );

  // Build smooth proportional Sankey ribbons (Issue -> Response, and Response -> Resolution)
  const ribbons = [];
  const respTotal = responseNodes.reduce((s, r) => s + r.weight, 0) || 1;
  const resTotal = resolutionNodes.reduce((s, r) => s + r.weight, 0) || 1;

  // Stage 1 -> Stage 2 ribbons (Issue -> top 2 actionable + aggregate flow ribbons cleanly distributed)
  const respInY = Object.fromEntries(responseNodes.map((n) => [n.id, n.y]));
  for (const iss of issueNodes) {
    let outY = iss.y;
    for (const resp of responseNodes) {
      const shareRatio = resp.weight / respTotal;
      const strokeW = Math.max(1.8, iss.h * shareRatio);
      const y1 = outY + strokeW / 2;
      const y2 = respInY[resp.id] + (resp.h * (iss.weight / (summary.comments || 1280))) / 2;
      outY += iss.h * shareRatio;
      respInY[resp.id] += resp.h * (iss.weight / (summary.comments || 1280));
      const x1 = iss.x + nodeW;
      const x2 = resp.x;
      const mx = (x1 + x2) / 2;
      ribbons.push({
        from: iss.id,
        to: resp.id,
        color: resp.color,
        strokeW,
        d: `M ${x1} ${y1.toFixed(1)} C ${mx} ${y1.toFixed(1)}, ${mx} ${y2.toFixed(1)}, ${x2} ${y2.toFixed(1)}`,
        title: `${iss.label} to ${resp.label}`,
        detail: `Proportional corpus flow across ${num(iss.weight)} issue comments`,
      });
    }
  }

  // Stage 2 -> Stage 3 ribbons (depart from RIGHT edge of Column-2 opaque pill so zero ribbons cross behind text!)
  const resInY = Object.fromEntries(resolutionNodes.map((n) => [n.id, n.y]));
  for (const resp of responseNodes) {
    let outY = resp.y;
    for (const res of resolutionNodes) {
      const shareRatio = res.weight / resTotal;
      const strokeW = Math.max(2.5, resp.h * shareRatio);
      const y1 = outY + strokeW / 2;
      const y2 = resInY[res.id] + (res.h * (resp.weight / respTotal)) / 2;
      outY += resp.h * shareRatio;
      resInY[res.id] += res.h * (resp.weight / respTotal);
      const x1 = resp.x + midPillW;
      const x2 = res.x;
      const mx = (x1 + x2) / 2;
      ribbons.push({
        from: resp.id,
        to: res.id,
        color: res.color,
        strokeW,
        d: `M ${x1} ${y1.toFixed(1)} C ${mx} ${y1.toFixed(1)}, ${mx} ${y2.toFixed(1)}, ${x2} ${y2.toFixed(1)}`,
        title: `${resp.label} to ${res.label}`,
        detail: `${res.label} (${res.sub}) across corpus threads`,
      });
    }
  }

  const renderNode = (n, align) => {
    if (align === "center") {
      return `
        <g class="sankey-node" tabindex="0" role="link" data-sankey-node="${esc(n.id)}" data-href="${esc(n.href)}" data-tip-title="${esc(n.label)}" data-tip-detail="${esc(n.detail)}" aria-label="${esc(n.label)}: ${esc(n.detail)}">
          <rect x="${n.x}" y="${n.y.toFixed(1)}" width="${midPillW}" height="${n.h.toFixed(1)}" rx="8" fill="var(--color-surface-solid)" stroke="var(--color-rule-2)" stroke-width="1"/>
          <rect x="${n.x}" y="${n.y.toFixed(1)}" width="8" height="${n.h.toFixed(1)}" rx="4" fill="${n.color}"/>
          <text x="${n.x + 18}" y="${(n.cy - 2).toFixed(1)}" text-anchor="start" class="sankey-node-label">${esc(n.label)}</text>
          <text x="${n.x + 18}" y="${(n.cy + 14).toFixed(1)}" text-anchor="start" class="sankey-node-count">${esc(n.sub)}</text>
        </g>
      `;
    }
    const labelX = align === "left" ? n.x - 12 : n.x + nodeW + 12;
    const anchor = align === "left" ? "end" : "start";
    return `
      <g class="sankey-node" tabindex="0" role="link" data-sankey-node="${esc(n.id)}" data-href="${esc(n.href)}" data-tip-title="${esc(n.label)}" data-tip-detail="${esc(n.detail)}" aria-label="${esc(n.label)}: ${esc(n.detail)}">
        <rect x="${n.x}" y="${n.y.toFixed(1)}" width="${nodeW}" height="${n.h.toFixed(1)}" rx="5" fill="${n.color}"/>
        <text x="${labelX}" y="${(n.cy - 2).toFixed(1)}" text-anchor="${anchor}" class="sankey-node-label">${esc(n.label)}</text>
        <text x="${labelX}" y="${(n.cy + 14).toFixed(1)}" text-anchor="${anchor}" class="sankey-node-count">${esc(n.sub)}</text>
      </g>
    `;
  };

  return `
    <div class="sankey-wrap" data-sankey-root>
      <svg class="sankey-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Interactive research flow diagram from ${issues.length} issue categories through 6 canonical community response classes to 4 explicit resolution evidence states">
        <text x="${colX.issue + nodeW}" y="20" text-anchor="end" class="sankey-col-head">1. Issue categories (${issues.length})</text>
        <text x="${colX.response}" y="20" text-anchor="start" class="sankey-col-head">2. Canonical response roles (1 to 6)</text>
        <text x="${colX.resolution}" y="20" text-anchor="start" class="sankey-col-head">3. Resolution evidence (4 states)</text>

        <g class="sankey-ribbons">
          ${ribbons.map((r) => `
            <path
              class="sankey-ribbon"
              data-from="${esc(r.from)}"
              data-to="${esc(r.to)}"
              data-tip-title="${esc(r.title)}"
              data-tip-detail="${esc(r.detail)}"
              data-draw-path
              d="${r.d}"
              stroke="${r.color}"
              stroke-width="${r.strokeW.toFixed(1)}"
            />
          `).join("")}
        </g>

        <g class="sankey-nodes">
          ${issueNodes.map((n) => renderNode(n, "left")).join("")}
          ${responseNodes.map((n) => renderNode(n, "center")).join("")}
          ${resolutionNodes.map((n) => renderNode(n, "right")).join("")}
        </g>
      </svg>
    </div>
  `;
}

export function bindSankeyInteractions(container) {
  const root = container.querySelector("[data-sankey-root]");
  if (!root) return;
  const nodes = [...root.querySelectorAll("[data-sankey-node]")];
  const ribbons = [...root.querySelectorAll(".sankey-ribbon")];

  const reset = () => {
    nodes.forEach((n) => n.classList.remove("is-dimmed"));
    ribbons.forEach((r) => r.classList.remove("is-dimmed", "is-active"));
    hideVizTooltip();
  };

  const highlightNode = (nodeId, evt) => {
    const connectedIds = new Set([nodeId]);
    ribbons.forEach((r) => {
      const hit = r.dataset.from === nodeId || r.dataset.to === nodeId;
      r.classList.toggle("is-active", hit);
      r.classList.toggle("is-dimmed", !hit);
      if (hit) {
        connectedIds.add(r.dataset.from);
        connectedIds.add(r.dataset.to);
      }
    });
    nodes.forEach((n) => {
      n.classList.toggle("is-dimmed", !connectedIds.has(n.dataset.sankeyNode));
    });
    if (evt) {
      const target = root.querySelector(`[data-sankey-node="${CSS.escape(nodeId)}"]`);
      if (target) {
        showVizTooltip(
          `<strong>${esc(target.dataset.tipTitle)}</strong><div>${esc(target.dataset.tipDetail)}</div><div class="muted">Click to drill into matching records</div>`,
          evt.clientX,
          evt.clientY
        );
      }
    }
  };

  nodes.forEach((node) => {
    const id = node.dataset.sankeyNode;
    node.addEventListener("mouseenter", (e) => highlightNode(id, e));
    node.addEventListener("mousemove", (e) => highlightNode(id, e));
    node.addEventListener("mouseleave", reset);
    node.addEventListener("focus", () => highlightNode(id, null));
    node.addEventListener("blur", reset);
    node.addEventListener("click", () => {
      if (node.dataset.href) window.location.hash = node.dataset.href;
    });
    node.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (node.dataset.href) window.location.hash = node.dataset.href;
      }
    });
  });

  ribbons.forEach((r) => {
    r.addEventListener("mouseenter", (e) => {
      ribbons.forEach((other) => {
        other.classList.toggle("is-active", other === r);
        other.classList.toggle("is-dimmed", other !== r);
      });
      showVizTooltip(
        `<strong>${esc(r.dataset.tipTitle)}</strong><div>${esc(r.dataset.tipDetail)}</div>`,
        e.clientX,
        e.clientY
      );
    });
    r.addEventListener("mousemove", (e) => {
      showVizTooltip(
        `<strong>${esc(r.dataset.tipTitle)}</strong><div>${esc(r.dataset.tipDetail)}</div>`,
        e.clientX,
        e.clientY
      );
    });
    r.addEventListener("mouseleave", reset);
  });
}

/* ==========================================================================
   2. Smooth Multi-Series Spline Area Chart with Live Crosshair Scrubbing
   ========================================================================== */
function catmullRomPath(pts) {
  if (!pts.length) return "";
  if (pts.length === 1) return `M ${pts[0].x} ${pts[0].y}`;
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

export function interactiveSplineAreaChart(monthly, { chartId = "sent-spline" } = {}) {
  if (!monthly || !monthly.length) {
    return `<div class="chart-empty">No monthly sentiment records available.</div>`;
  }

  const W = 720;
  const H = 264;
  const pad = { top: 24, right: 20, bottom: 38, left: 42 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const seriesMeta = [
    { key: "positive", label: "Positive", color: "var(--color-positive)" },
    { key: "neutral", label: "Neutral", color: "var(--color-accent)" },
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
  const yMax = Math.ceil(maxVal / 16) * 16;
  const ticks = [0, Math.round(yMax * 0.25), Math.round(yMax * 0.5), Math.round(yMax * 0.75), yMax];

  const xAt = (idx) => pad.left + (normalized.length > 1 ? (idx / (normalized.length - 1)) * plotW : plotW / 2);
  const yAt = (val) => pad.top + plotH - (val / yMax) * plotH;

  const defs = seriesMeta.map((s) => `
    <linearGradient id="${esc(chartId)}-grad-${s.key}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${s.color}" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="${s.color}" stop-opacity="0.01"/>
    </linearGradient>
  `).join("");

  const gridLines = ticks.map((t) => {
    const y = yAt(t);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1"/>
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="12" font-family="var(--font-body)">${t}</text>
    `;
  }).join("");

  const xStep = normalized.length > 10 ? Math.ceil(normalized.length / 8) : 1;
  const xLabels = normalized.map((m, idx) => {
    const show = idx === 0 || idx === normalized.length - 1 || idx % xStep === 0;
    if (!show) return "";
    return `<text x="${xAt(idx).toFixed(1)}" y="${(H - 12).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="12" font-family="var(--font-body)">${esc(monthLabel(m.month))}</text>`;
  }).join("");

  const seriesGroups = seriesMeta.map((s) => {
    const pts = normalized.map((m, idx) => ({
      x: xAt(idx),
      y: yAt(Number(m[s.key] || 0)),
      val: Number(m[s.key] || 0),
    }));
    const linePath = catmullRomPath(pts);
    const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} L ${pts[0].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} Z`;

    return `
      <g class="spline-series" data-spline-series="${s.key}">
        <path class="spline-area" d="${areaPath}" fill="url(#${esc(chartId)}-grad-${s.key})"/>
        <path class="spline-line" d="${linePath}" stroke="${s.color}" data-draw-path/>
        ${pts.map((p, idx) => `
          <circle class="spline-dot" data-dot-idx="${idx}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="${s.color}" stroke="var(--color-surface-solid)" stroke-width="2"/>
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
      <div class="spline-controls" role="group" aria-label="Filter sentiment spline curves">
        <button type="button" class="series-pill is-active" data-series-filter="all">All polarities</button>
        ${seriesMeta.map((s) => `
          <button type="button" class="series-pill" data-series-filter="${s.key}">
            <span class="legend-swatch" style="background:${s.color}" aria-hidden="true"></span>
            ${s.label}
          </button>
        `).join("")}
      </div>
      <div class="spacer-xs"></div>
      <svg class="spline-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Interactive smooth spline area chart of predicted comment sentiment counts by month">
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
          el.classList.toggle("is-hidden", mode !== "all" && el.dataset.splineSeries !== mode);
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
          `<strong>${esc(rect.dataset.month)} (${num(tot)} comments)</strong>
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
   3. Interactive NMF Topic Constellation SVG (8 Proportional Clickable Orbs)
   ========================================================================== */
export function topicConstellationSvg(topics, selectedId) {
  const W = 560;
  const H = 376;
  const cx = 280;
  const cy = 188;

  // Deterministic well-spaced orbital layout sized cleanly around largest topic
  const sorted = [...topics].sort((a, b) => Number(b.document_count) - Number(a.document_count));
  const maxCount = Math.max(1, ...sorted.map((t) => Number(t.document_count || 0)));

  const positions = [
    { x: 280, y: 188 },
    { x: 432, y: 122 },
    { x: 126, y: 124 },
    { x: 446, y: 254 },
    { x: 118, y: 262 },
    { x: 284, y: 318 },
    { x: 368, y: 58 },
    { x: 196, y: 58 },
    { x: 504, y: 186 },
    { x: 64, y: 194 },
  ];

  const orbs = sorted.map((t, idx) => {
    const pos = idx < positions.length
      ? positions[idx]
      : {
          x: Math.round(cx + 158 * Math.cos((idx / sorted.length) * Math.PI * 2)),
          y: Math.round(cy + 132 * Math.sin((idx / sorted.length) * Math.PI * 2)),
        };
    const ratio = Math.sqrt(Number(t.document_count || 1) / maxCount);
    const r = Math.round(24 + ratio * 30);
    const color = t.character.script === "devanagari"
      ? "var(--color-accent-2)"
      : t.character.tone === "info"
        ? "var(--color-role-advice)"
        : "var(--color-accent)";
    const isSel = Number(t.topic_id) === Number(selectedId);
    const rawKw = (t.keywords || []).slice(0, 2).join(", ");
    const shortKw = rawKw.length > 12 ? `${rawKw.slice(0, 11)}…` : rawKw;
    return { t, pos, r, color, isSel, shortKw };
  });

  // Subtle connecting spokes to center
  const spokes = orbs.slice(1).map((o) => `
    <line x1="${orbs[0].pos.x}" y1="${orbs[0].pos.y}" x2="${o.pos.x}" y2="${o.pos.y}" stroke="var(--color-rule)" stroke-width="1" stroke-dasharray="3 4" opacity="0.6"/>
  `).join("");

  return `
    <div class="constellation-stage">
      <svg class="constellation-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Interactive constellation of ${orbs.length} unsupervised NMF topic clusters sized by comment count">
        <circle cx="${cx}" cy="${cy}" r="96" fill="none" stroke="var(--color-rule)" stroke-width="1" opacity="0.45"/>
        <circle cx="${cx}" cy="${cy}" r="168" fill="none" stroke="var(--color-rule)" stroke-width="1" opacity="0.32"/>
        ${spokes}
        ${orbs.map(({ t, pos, r, color, isSel, shortKw }) => `
          <g class="topic-orb ${isSel ? "is-selected" : ""}" tabindex="0" role="button" data-select-topic="${t.topic_id}" aria-label="Topic ${t.topic_id}, ${num(t.document_count)} comments, ${esc(t.character.tag)}">
            <circle class="topic-orb-ring" cx="${pos.x}" cy="${pos.y}" r="${r + 7}"/>
            <circle class="topic-orb-circle" cx="${pos.x}" cy="${pos.y}" r="${r}" fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="2"/>
            <text x="${pos.x}" y="${pos.y - 5}" text-anchor="middle" fill="var(--color-ink)" font-size="12.5" font-weight="600" font-family="var(--font-body)">Topic ${t.topic_id}</text>
            <text x="${pos.x}" y="${pos.y + 11}" text-anchor="middle" fill="var(--color-ink-2)" font-size="12" font-family="var(--font-body)">${num(t.document_count)}</text>
          </g>
        `).join("")}
      </svg>
    </div>
  `;
}

/* ==========================================================================
   4. Animated Radial Canonical Response Wheel (6 Fixed Classes)
   ========================================================================== */
export function radialResponseWheel(orderedItems, total) {
  const r = 96;
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
        <circle cx="130" cy="130" r="${r}" fill="none" stroke="var(--color-track)" stroke-width="24"/>
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
   5. Shared Charts: Stacked Bar, Horizontal Bars, Matrix, Mini-Strip, Benchmarks
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
      <rect x="${cursor.toFixed(2)}" y="0" width="${Math.max(w, 1).toFixed(2)}" height="12" fill="${esc(seg.color)}">
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
      <svg class="stack-bar-svg" viewBox="0 0 ${viewW} 12" preserveAspectRatio="none" role="img" aria-label="${esc(ariaLabel)}">
        <rect x="0" y="0" width="${viewW}" height="12" fill="var(--color-track)"/>
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
      const rect = `<rect x="${cursor.toFixed(2)}" y="0" width="${Math.max(w, 1).toFixed(2)}" height="10" fill="${esc(seg.color)}"><title>${esc(row.label)}, ${esc(seg.label)}: ${num(val)} (${share(val, total)})</title></rect>`;
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
        <svg class="matrix-bar-svg" viewBox="0 0 ${viewW} 10" preserveAspectRatio="none" role="img" aria-label="${esc(row.label)}">
          <rect x="0" y="0" width="${viewW}" height="10" fill="var(--color-track)"/>
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
  const viewW = 108;
  const rects = CANONICAL_RESPONSE_TYPES.map((key) => {
    const val = Number(counts[key] || 0);
    if (!val) return "";
    const w = (val / total) * viewW;
    const r = `<rect x="${cursor.toFixed(1)}" y="0" width="${Math.max(w, 1).toFixed(1)}" height="6" fill="${RESPONSE_META[key].color}"><title>${RESPONSE_META[key].label}: ${val}</title></rect>`;
    cursor += w;
    return r;
  }).join("");
  const actionable = Number(counts.advice || 0) + Number(counts.solution || 0);
  return `
    <div class="mini-response-strip" title="${CANONICAL_RESPONSE_TYPES.map((k) => `${RESPONSE_META[k].label}: ${counts[k] || 0}`).join(", ")}">
      <svg class="mini-strip-svg" viewBox="0 0 ${viewW} 6" aria-hidden="true">${rects}</svg>
      <span class="num mini-strip-caption">${actionable} actionable / ${total}</span>
    </div>
  `;
}

export function modelComparisonSvgChart(tasks) {
  if (!tasks || !tasks.length) return `<div class="chart-empty">No model benchmarks available.</div>`;
  const W = 820;
  const H = 264;
  const pad = { top: 44, right: 24, bottom: 50, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  const grid = ticks.map((t) => {
    const y = pad.top + plotH - t * plotH;
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="var(--color-rule)" stroke-width="1"/>
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${Math.round(t * 100)}%</text>
    `;
  }).join("");

  const slotW = plotW / tasks.length;
  const barW = Math.min(28, slotW * 0.2);
  const gap = 20;

  const groups = tasks.map((t, idx) => {
    const cx = pad.left + slotW * idx + slotW / 2;
    const valRatio = Number.isFinite(Number(t.valF1)) ? Number(t.valF1) : 0;
    const testRatio = Number.isFinite(Number(t.testF1)) ? Number(t.testF1) : 0;
    const valH = valRatio * plotH;
    const testH = testRatio * plotH;
    const valY = pad.top + plotH - valH;
    const testY = pad.top + plotH - testH;
    const valX = cx - barW - gap / 2;
    const testX = cx + gap / 2;

    return `
      <g>
        <rect x="${valX.toFixed(1)}" y="${valY.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(valH, 2).toFixed(1)}" rx="5" fill="var(--color-series-val)">
          <title>${esc(t.label)}, validation macro F1: ${percent(t.valF1)}</title>
        </rect>
        <text x="${(valX + barW / 2 - 4).toFixed(1)}" y="${(valY - 7).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${percent(t.valF1, 1, "N/A")}</text>

        <rect x="${testX.toFixed(1)}" y="${testY.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(testH, 2).toFixed(1)}" rx="5" fill="var(--color-accent)">
          <title>${esc(t.label)}, held-out test macro F1: ${percent(t.testF1)}</title>
        </rect>
        <text x="${(testX + barW / 2 + 4).toFixed(1)}" y="${(testY - 7).toFixed(1)}" text-anchor="middle" fill="var(--color-ink)" font-size="11.5" font-weight="600" font-family="var(--font-body)">${percent(t.testF1, 1, "N/A")}</text>

        <text x="${cx.toFixed(1)}" y="${(H - pad.bottom + 18).toFixed(1)}" text-anchor="middle" fill="var(--color-ink)" font-size="12" font-weight="600" font-family="var(--font-body)">${esc(t.label)}</text>
        <text x="${cx.toFixed(1)}" y="${(H - pad.bottom + 34).toFixed(1)}" text-anchor="middle" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">${t.multilabel ? "Multi-label" : t.binary ? "Binary" : "Single-label"}</text>
      </g>
    `;
  }).join("");

  return `
    <div>
      <svg class="responsive-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Held-out validation and test macro F1 comparison across all five active NLP models">
        <text x="${pad.left}" y="15" fill="var(--color-muted)" font-size="11.5" font-family="var(--font-body)">Held-out macro F1 (%) on thread-grouped validation and test splits</text>
        ${grid}
        ${groups}
      </svg>
      <div class="spacer-xs"></div>
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-series-val)"></span><span>Validation macro F1</span></span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--color-accent)"></span><span>Held-out test macro F1</span></span>
      </div>
    </div>
  `;
}

export function confusionMatrixTable(matrix, labels) {
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
    <details class="confusion-details">
      <summary>Confusion matrix: actual (rows) vs. predicted (columns), n = ${num(total)}</summary>
      <div class="table-scroll">
        <table class="confusion-table" aria-label="Held-out test confusion matrix">
          <thead><tr><th scope="col">Actual / Predicted</th>${head}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>
  `;
}
