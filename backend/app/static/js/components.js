/**
 * Dusk Studio UI Component Primitives.
 * Smoked-glass decks, animated ring-gauge KPI pods, honest notices, badges,
 * copyable IDs, glossary tooltips, empty/error/skeleton states.
 */
import {
  GLOSSARY,
  LANGUAGE_LABELS,
  RESOLUTION_META,
  RESPONSE_META,
  SENTIMENT_META,
  esc,
  pretty,
} from "./format.js?v=5.6";
import { icons } from "./icons.js?v=5.6";

/**
 * Primary Dusk elevated deck container.
 */
export function deck(title, { subtitle = "", body = "", action = "", id = "", className = "" } = {}) {
  return `
    <section class="deck anim-enter ${esc(className)}" ${id ? `id="${esc(id)}"` : ""}>
      <header class="deck-header">
        <div class="deck-header-copy">
          <h2 class="deck-title">${title}</h2>
          ${subtitle ? `<p class="deck-subtitle">${subtitle}</p>` : ""}
        </div>
        ${action ? `<div class="deck-actions">${action}</div>` : ""}
      </header>
      <div class="deck-body">${body}</div>
    </section>
  `;
}

/**
 * Dusk Signature Horizontal Ruled Tally Strip (zero 4-tile box grids, zero fake 100% rings).
 * Displays animated count-up tabular figures parted by crisp vertical hairlines,
 * attaching a compact 36px SVG ratio ring ONLY when `ratioPct` is a genuine percentage.
 */
export function tallyStrip(items = [], ariaLabel = "Key indicators") {
  const r = 14;
  const circ = 2 * Math.PI * r;
  return `
    <section class="tally-strip anim-enter" aria-label="${esc(ariaLabel)}">
      ${items.map((item) => {
        const hasRatio = item.ratioPct !== undefined && item.ratioPct !== null && Number.isFinite(Number(item.ratioPct));
        const clamped = hasRatio ? Math.max(0, Math.min(100, Number(item.ratioPct))) : 0;
        const targetOffset = circ * (1 - clamped / 100);
        const countAttrs = item.countTo !== undefined && item.countTo !== null && Number.isFinite(Number(item.countTo))
          ? `data-count-to="${Number(item.countTo)}" data-count-decimals="${item.decimals || 0}" data-count-prefix="${esc(item.prefix || "")}" data-count-suffix="${esc(item.suffix || "")}"`
          : "";

        const ratioRing = hasRatio
          ? `
            <span class="mini-ring" aria-hidden="true">
              <svg width="36" height="36" viewBox="0 0 36 36">
                <circle class="mini-ring-track" cx="18" cy="18" r="${r}"/>
                <circle
                  class="mini-ring-arc"
                  cx="18"
                  cy="18"
                  r="${r}"
                  stroke="${esc(item.color || "var(--color-accent)")}"
                  stroke-dasharray="${circ.toFixed(2)}"
                  stroke-dashoffset="${targetOffset.toFixed(2)}"
                  data-ring-circ="${circ.toFixed(2)}"
                  data-ring-target="${targetOffset.toFixed(2)}"
                />
              </svg>
            </span>
          `
          : "";

        const inner = `
          <div class="tally-head">
            <span class="tally-label">${esc(item.label)}</span>
            ${ratioRing}
          </div>
          <strong class="tally-value ${item.accent ? "tally-value-accent" : ""}"><span class="num" ${countAttrs}>${esc(item.valueText)}</span>${item.unit ? ` <span class="tally-unit">${esc(item.unit)}</span>` : ""}</strong>
          ${item.note ? `<span class="tally-note">${item.note}</span>` : ""}
        `;

        return item.href
          ? `<a class="tally-cell tally-cell-link" href="${esc(item.href)}">${inner}</a>`
          : `<div class="tally-cell">${inner}</div>`;
      }).join("")}
    </section>
  `;
}

export function callout(text, { tone = "info", title = "" } = {}) {
  const icon = tone === "warning" || tone === "danger" ? icons.alert : icons.info;
  return `
    <div class="notice notice-${esc(tone)} anim-enter" role="${tone === "danger" ? "alert" : "note"}">
      <span class="notice-icon" aria-hidden="true">${icon}</span>
      <div class="notice-copy">
        ${title ? `<strong class="notice-title">${esc(title)}</strong> ` : ""}
        <span>${text}</span>
      </div>
    </div>
  `;
}

export function badge(label, tone = "neutral", { title = "" } = {}) {
  return `<span class="tag tag-${esc(tone)}" ${title ? `title="${esc(title)}"` : ""}>${esc(label)}</span>`;
}

export function confidenceBadge(score, tier = null, { needsReview = false, prefix = "" } = {}) {
  if (score === null || score === undefined || !Number.isFinite(Number(score))) return "";
  const val = Number(score);
  const resolvedTier = tier || (val >= 0.8 ? "high" : val >= 0.55 ? "medium" : "low");
  const isReview = Boolean(needsReview || resolvedTier === "low");
  const dotColor = resolvedTier === "high"
    ? "var(--color-positive)"
    : resolvedTier === "medium"
      ? "var(--color-info)"
      : "var(--color-warning)";
  const pct = `${(val * 100).toFixed(0)}%`;
  const suffix = isReview ? " · review" : "";
  const title = `Calibrated prediction confidence: ${(val * 100).toFixed(1)}% (${resolvedTier} tier${isReview ? "; flagged for human review" : ""})`;
  if (isReview) {
    return `<span class="tag tag-warning tag-conf" title="${esc(title)}"><span class="swatch-inline" style="background:${dotColor}" aria-hidden="true"></span>${esc(prefix)}${pct}${esc(suffix)}</span>`;
  }
  return `<span class="conf-meter" title="${esc(title)}"><span class="swatch-inline" style="background:${dotColor}" aria-hidden="true"></span><span class="num">${esc(prefix)}${pct}</span></span>`;
}

export function resolutionBadge(state) {
  const meta = RESOLUTION_META[state] || { label: pretty(state), cls: "neutral", description: "" };
  return badge(meta.label, meta.cls, { title: meta.description });
}

export function sentimentBadge(sentiment, prefix = "") {
  const key = String(sentiment || "").toLowerCase();
  const meta = SENTIMENT_META[key];
  if (!meta) return badge(`${prefix}${pretty(sentiment || "Unknown")}`, "neutral");
  return `<span class="tag tag-sentiment-${esc(key)}">${esc(prefix)}${esc(meta.label)}</span>`;
}

export function roleBadge(role, { prefix = "" } = {}) {
  const meta = RESPONSE_META[role];
  if (!meta) return badge(`${prefix}${pretty(role || "Unknown")}`, "neutral");
  return `<span class="tag tag-role" title="${esc(meta.definition)}"><span class="swatch-inline" style="background:${meta.color}" aria-hidden="true"></span>${esc(prefix)}${esc(meta.label)}</span>`;
}

export function languageBadge(language, script, isCodeMixed) {
  const langName = LANGUAGE_LABELS[language] || pretty(language || "unknown");
  const parts = [langName];
  if (script) parts.push(`${pretty(script)} script`);
  if (isCodeMixed) parts.push("code-mixed");
  return `<span class="tag tag-neutral" title="Detected language and script metadata">${esc(parts.join(", "))}</span>`;
}

export function copyIdChip(value, { label = "" } = {}) {
  const display = label ? `${label}: ${value}` : value;
  return `
    <button type="button" class="copy-chip" data-copy-value="${esc(value)}" title="Copy identifier ${esc(value)}">
      <code class="num">${esc(display)}</code>
      <span class="copy-chip-icon" aria-hidden="true">${icons.copy}</span>
    </button>
  `;
}

export function glossaryTip(termKey, displayLabel) {
  const definition = GLOSSARY[termKey];
  if (!definition) return esc(displayLabel);
  return `
    <span class="glossary-term" tabindex="0" aria-label="${esc(displayLabel)}: ${esc(definition)}">
      <span>${esc(displayLabel)}</span>
      <span class="glossary-icon" aria-hidden="true">${icons.info}</span>
      <span class="glossary-popover" role="tooltip">${esc(definition)}</span>
    </span>
  `;
}

export function emptyState({ title = "No matching records", description = "", actionHtml = "" }) {
  return `
    <div class="empty-state anim-enter">
      <h3>${esc(title)}</h3>
      ${description ? `<p>${esc(description)}</p>` : ""}
      ${actionHtml ? `<div class="empty-state-actions">${actionHtml}</div>` : ""}
    </div>
  `;
}

export function errorState(message, { canRetry = true } = {}) {
  return `
    <section class="error-state anim-enter" role="alert">
      <div class="error-state-head">
        <span aria-hidden="true">${icons.alert}</span>
        <h2>Data could not be loaded</h2>
      </div>
      <p>${esc(message)}</p>
      <p class="error-state-hint">Confirm the FastAPI service is reachable and corpus artifacts exist under <code>data/live_corpus/</code> and <code>ml/artifacts/live_corpus/</code>.</p>
      ${canRetry ? `<button type="button" class="btn btn-quiet" data-retry>Retry connection</button>` : ""}
    </section>
  `;
}

export function skeletonView() {
  return `
    <div class="skeleton-canvas" aria-busy="true" aria-label="Loading view">
      <div class="skeleton-row-4">
        <div class="skeleton-card h-120"></div>
        <div class="skeleton-card h-120"></div>
        <div class="skeleton-card h-120"></div>
        <div class="skeleton-card h-120"></div>
      </div>
      <div class="skeleton-card h-320"></div>
    </div>
  `;
}
