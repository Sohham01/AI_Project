/**
 * Reddit Community Intelligence (RCI) — Dusk Studio Router & Shell Controller.
 * Built from scratch: 204px labeled studio rail, command bar,
 * mobile bottom dock, Ctrl+K Command Palette, Dusk Midnight <-> Daylight Bone theme switcher,
 * and automatic GPU-safe motion hydration.
 */
import { api, clearApiCache } from "./js/api.js?v=5.4";
import { errorState, skeletonView } from "./js/components.js?v=5.4";
import { CANONICAL_RESPONSE_TYPES, RESPONSE_META, dateText, esc, num, pretty } from "./js/format.js?v=5.4";
import { hideVizTooltip, hydrateMotion } from "./js/motion.js?v=5.4";
import { renderExplorer } from "./js/views/explorer.js?v=5.4";
import { renderIssues } from "./js/views/issues.js?v=5.4";
import { renderModels } from "./js/views/models.js?v=5.4";
import { renderOverview } from "./js/views/overview.js?v=5.4";
import { renderResolution } from "./js/views/resolution.js?v=5.4";
import { renderResponses } from "./js/views/responses.js?v=5.4";
import { renderSentiment } from "./js/views/sentiment.js?v=5.4";
import { renderThread } from "./js/views/thread.js?v=5.4";
import { renderTopics } from "./js/views/topics.js?v=5.4";

const PAGE_META = {
  overview: {
    title: "Community intelligence studio",
    description: "Trace how public IIM Sambalpur Reddit discussions unfold across issue domains, six canonical response classes, actionable proposals, and explicit resolution signals.",
    crumb: "Overview",
    nav: "overview",
  },
  issues: {
    title: "Issues intelligence",
    description: "Compare community issue domains by actionable recommendation density, predicted polarity, and explicit outcome evidence.",
    crumb: "Issues",
    nav: "issues",
  },
  explorer: {
    title: "Discussion workbench",
    description: "Filter anonymized community threads, switch tabs inside the live split preview, or open the full parent-child conversation tree.",
    crumb: "Discussions",
    nav: "explorer",
  },
  thread: {
    title: "Conversation audit",
    description: "Inspect the original post, nested reply chains, language and script metadata, five-task probabilities, and resolution evidence.",
    crumb: "Conversation audit",
    nav: "explorer",
  },
  topics: {
    title: "Topic constellation",
    description: "Explore unsupervised NMF clusters fit directly on community comments, inspect weighted terms, and audit representative replies.",
    crumb: "Topic map",
    nav: "topics",
  },
  sentiment: {
    title: "Sentiment & emotion pulse",
    description: "Scrub interactive monthly polarity splines and compare single-label sentiment against multi-label predicted emotions.",
    crumb: "Sentiment",
    nav: "sentiment",
  },
  responses: {
    title: "Canonical community responses",
    description: "Inspect community reply roles in exact fixed canonical order: agreement, personal_experience, opinion, advice, solution, counterargument.",
    crumb: "Responses",
    nav: "responses",
  },
  resolution: {
    title: "Solutions & resolution evidence",
    description: "Audit proposed solutions and advice separately from explicit first-person resolution outcomes; missing follow-up never implies failure.",
    crumb: "Resolution",
    nav: "resolution",
  },
  models: {
    title: "Models & dataset provenance",
    description: "Inspect anonymized Reddit corpus provenance, model architectures, held-out thread-grouped validation/test scores, and robustness probes.",
    crumb: "Models & data",
    nav: "models",
  },
};

const dom = {
  content: document.querySelector("#page-content"),
  title: document.querySelector("#page-title"),
  description: document.querySelector("#page-description"),
  breadcrumb: document.querySelector("#breadcrumb"),
  serviceState: document.querySelector("#service-state"),
  stateDot: document.querySelector("#state-dot"),
  datasetSize: document.querySelector("#dataset-size"),
  generatedDate: document.querySelector("#generated-date"),
  sidebarVersion: document.querySelector("#sidebar-version"),
  sidebarDataset: document.querySelector("#sidebar-dataset"),
  sidebarCounts: document.querySelector("#sidebar-counts"),
  bannerNotice: document.querySelector("#banner-notice"),
  footerProvenance: document.querySelector("#footer-provenance"),
  sidebar: document.querySelector("#sidebar"),
  backdrop: document.querySelector("#sidebar-backdrop"),
  menuBtn: document.querySelector("#mobile-menu-btn"),
  dockMoreBtn: document.querySelector("#dock-more-btn"),
  closeBtn: document.querySelector("#sidebar-close"),
  globalSearchForm: document.querySelector("#global-search-form"),
  globalSearchInput: document.querySelector("#global-search-input"),
  openPaletteBtn: document.querySelector("#open-palette-btn"),
  paletteBackdrop: document.querySelector("#palette-backdrop"),
  paletteInput: document.querySelector("#palette-input"),
  paletteResults: document.querySelector("#palette-results"),
  paletteCloseBtn: document.querySelector("#palette-close-btn"),
  themeToggleBtn: document.querySelector("#theme-toggle-btn"),
  themeToggleLabel: document.querySelector("#theme-toggle-label"),
  provPillBtn: document.querySelector("#provenance-pill-btn"),
  provPillText: document.querySelector("#prov-pill-text"),
  provPopTitle: document.querySelector("#prov-pop-title"),
  provPopBody: document.querySelector("#prov-pop-body"),
  provPopover: document.querySelector("#provenance-popover"),
  closeProvBtn: document.querySelector("#close-prov-popover"),
  toastRegion: document.querySelector("#toast-region"),
};

const ctx = {
  health: null,
  summary: null,
};

let renderSeq = 0;
let toastTimer = null;
let paletteSearchTimer = null;
let paletteFocusIdx = 0;

/* ==========================================================================
   Theme Switcher (Dusk Midnight <-> Daylight Bone)
   ========================================================================== */
function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("rci-studio-theme", next); } catch {}
  if (dom.themeToggleLabel) {
    dom.themeToggleLabel.textContent = next === "light" ? "Dusk dark mode" : "Daylight mode";
  }
  if (dom.themeToggleBtn) {
    dom.themeToggleBtn.setAttribute("data-tip", next === "light" ? "Switch to Dusk dark" : "Switch to Daylight light");
  }
}

function initTheme() {
  let saved = "dark";
  try {
    const v = localStorage.getItem("rci-studio-theme");
    if (v === "light" || v === "dark") saved = v;
  } catch {}
  applyTheme(saved);
  dom.themeToggleBtn?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    applyTheme(cur === "light" ? "dark" : "light");
  });
}

/* ==========================================================================
   Routing & Active Nav Sync
   ========================================================================== */
function parseRoute() {
  const rawHash = (window.location.hash || "#/overview").replace(/^#/, "");
  const [pathPart, queryPart = ""] = rawHash.split("?");
  const segments = pathPart.replace(/^\/+/, "").split("/").filter(Boolean);
  const params = Object.fromEntries(new URLSearchParams(queryPart).entries());

  if (!segments.length) return { view: "overview", id: null, params };
  const head = segments[0].toLowerCase();
  if (head === "thread" && segments[1]) {
    return { view: "thread", id: decodeURIComponent(segments[1]), params };
  }
  if (PAGE_META[head] && head !== "thread") {
    return { view: head, id: null, params };
  }
  return { view: "overview", id: null, params };
}

function showToast(message) {
  if (!dom.toastRegion) return;
  dom.toastRegion.innerHTML = `<div class="toast-pill">${esc(message)}</div>`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    dom.toastRegion.innerHTML = "";
  }, 2400);
}

function setDrawerOpen(open) {
  if (!dom.sidebar || !dom.backdrop) return;
  dom.sidebar.classList.toggle("is-open", open);
  dom.backdrop.hidden = !open;
  document.body.classList.toggle("drawer-open", open);
  dom.menuBtn?.setAttribute("aria-expanded", String(open));
  if (open) dom.closeBtn?.focus();
}

function updateActiveNav(navKey) {
  document.querySelectorAll(".rail-link[data-view]").forEach((el) => {
    if (el.dataset.view === navKey) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  document.querySelectorAll(".dock-btn[data-dock]").forEach((el) => {
    if (el.dataset.dock === navKey) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
}

function updateHeader(route) {
  const meta = PAGE_META[route.view] || PAGE_META.overview;
  dom.title.textContent = meta.title;
  dom.description.textContent = meta.description;
  document.title = `${meta.crumb} | Reddit Community Intelligence`;
  updateActiveNav(meta.nav);

  if (route.view === "thread" && route.id) {
    dom.breadcrumb.innerHTML = `
      <a href="#/overview">RCI Studio</a>
      <span class="crumb-sep" aria-hidden="true">/</span>
      <a href="#/explorer">Discussions</a>
      <span class="crumb-sep" aria-hidden="true">/</span>
      <span class="crumb-active num">${esc(route.id)}</span>
    `;
  } else {
    dom.breadcrumb.innerHTML = `
      <a href="#/overview">RCI Studio</a>
      <span class="crumb-sep" aria-hidden="true">/</span>
      <span class="crumb-active">${esc(meta.crumb)}</span>
    `;
  }
}

/* ==========================================================================
   Command Palette Modal (Ctrl+K)
   ========================================================================== */
function getStaticPaletteItems() {
  const viewItems = Object.entries(PAGE_META)
    .filter(([k]) => k !== "thread")
    .map(([k, v]) => ({
      title: v.crumb,
      meta: "View",
      href: `#/${k}`,
      hay: `${v.crumb} ${v.title} ${v.description}`.toLowerCase(),
    }));
  const issueItems = (ctx.summary?.issue_distribution || []).map((i) => ({
    title: pretty(i.label),
    meta: "Issue domain",
    href: `#/issues?selected=${encodeURIComponent(i.key)}`,
    hay: `${i.label} ${i.key} issue`.toLowerCase(),
  }));
  const roleItems = CANONICAL_RESPONSE_TYPES.map((k, idx) => ({
    title: `${idx + 1}. ${RESPONSE_META[k].label}`,
    meta: "Canonical role",
    href: `#/explorer?response=${encodeURIComponent(k)}`,
    hay: `${k} ${RESPONSE_META[k].label} response`.toLowerCase(),
  }));
  return [...viewItems, ...issueItems, ...roleItems];
}

function paintPaletteList(items) {
  if (!dom.paletteResults) return;
  paletteFocusIdx = 0;
  if (!items.length) {
    dom.paletteResults.innerHTML = `<div class="palette-item"><span class="muted">No matching views or community threads</span></div>`;
    return;
  }
  dom.paletteResults.innerHTML = items.map((item, idx) => `
    <a class="palette-item ${idx === 0 ? "is-focused" : ""}" role="option" href="${esc(item.href)}" data-palette-idx="${idx}">
      <span>${esc(item.title)}</span>
      <span class="palette-item-meta">${esc(item.meta)}</span>
    </a>
  `).join("");
  dom.paletteResults.querySelectorAll(".palette-item").forEach((el) => {
    el.addEventListener("click", () => closePalette());
  });
}

function updatePalette(query) {
  const q = String(query || "").trim().toLowerCase();
  const base = getStaticPaletteItems().filter((i) => !q || i.hay.includes(q)).slice(0, 8);
  paintPaletteList(base);

  if (q.length >= 2) {
    clearTimeout(paletteSearchTimer);
    paletteSearchTimer = setTimeout(async () => {
      try {
        const res = await api.discussions({ q, page: 1, page_size: 5 });
        const threadItems = (res.items || []).map((p) => ({
          title: p.title,
          meta: `${pretty(p.issue)} (${p.id})`,
          href: `#/thread/${encodeURIComponent(p.id)}`,
        }));
        paintPaletteList([...threadItems, ...base].slice(0, 10));
      } catch {
        // Keep static results quietly
      }
    }, 160);
  }
}

function openPalette(prefill = "") {
  if (!dom.paletteBackdrop) return;
  dom.paletteBackdrop.hidden = false;
  document.body.classList.add("modal-open");
  if (dom.paletteInput) {
    dom.paletteInput.value = prefill;
    dom.paletteInput.focus();
  }
  updatePalette(prefill);
}

function closePalette() {
  if (!dom.paletteBackdrop) return;
  dom.paletteBackdrop.hidden = true;
  document.body.classList.remove("modal-open");
}

function movePaletteFocus(delta) {
  const items = [...(dom.paletteResults?.querySelectorAll(".palette-item[href]") || [])];
  if (!items.length) return;
  paletteFocusIdx = (paletteFocusIdx + delta + items.length) % items.length;
  items.forEach((el, idx) => el.classList.toggle("is-focused", idx === paletteFocusIdx));
  items[paletteFocusIdx]?.scrollIntoView({ block: "nearest" });
}

/* ==========================================================================
   Shell Hydration & Route Rendering
   ========================================================================== */
async function hydrateShell() {
  try {
    const [health, summary] = await Promise.all([api.health(), api.summary()]);
    ctx.health = health;
    ctx.summary = summary;

    dom.serviceState.textContent = "API ready";
    dom.stateDot.classList.add("ok");
    dom.stateDot.classList.remove("err");

    const prov = summary.provenance || {};
    const isSynthetic = Boolean(summary.synthetic_only);
    dom.datasetSize.textContent = `${num(summary.posts)} threads, ${num(summary.comments)} comments`;
    dom.generatedDate.textContent = dateText(summary.generated_at);
    dom.sidebarVersion.textContent = `v${summary.dataset_version}`;
    dom.sidebarDataset.textContent = isSynthetic
      ? `Seed ${prov.generation_seed || "20261116"}`
      : (prov.source_provider === "arctic_shift" ? "Arctic Shift archive" : "Public Reddit corpus");
    dom.sidebarCounts.textContent = `${num(summary.posts)} threads, ${num(summary.comments)} comments`;
    if (dom.bannerNotice && prov.notice) dom.bannerNotice.textContent = prov.notice;
    if (dom.provPillText) {
      dom.provPillText.textContent = isSynthetic ? "Archived sample corpus" : "Public Reddit corpus";
    }
    if (dom.provPopTitle) {
      dom.provPopTitle.textContent = isSynthetic ? "Archived sample provenance" : "Anonymized public Reddit corpus";
    }
    if (dom.provPopBody && prov.notice) {
      dom.provPopBody.textContent = prov.notice;
    }
    dom.footerProvenance.textContent = isSynthetic
      ? `Corpus v${summary.dataset_version}, read-only /api/demo`
      : `Anonymized public Reddit corpus v${summary.dataset_version} (Arctic Shift), read-only /api/demo`;
  } catch (err) {
    dom.serviceState.textContent = "API unavailable";
    dom.stateDot.classList.add("err");
    dom.stateDot.classList.remove("ok");
    throw err;
  }
}

async function renderCurrentRoute() {
  const mySeq = ++renderSeq;
  const route = parseRoute();
  hideVizTooltip();
  setDrawerOpen(false);
  if (dom.provPopover) dom.provPopover.hidden = true;
  dom.provPillBtn?.setAttribute("aria-expanded", "false");
  updateHeader(route);

  dom.content.innerHTML = skeletonView();

  try {
    if (!ctx.summary) {
      await hydrateShell();
    }
    if (mySeq !== renderSeq) return;

    const container = document.createElement("div");
    container.style.display = "contents";

    switch (route.view) {
      case "overview":
        await renderOverview(container, ctx, route.params);
        break;
      case "issues":
        await renderIssues(container, ctx, route.params);
        break;
      case "explorer":
        await renderExplorer(container, ctx, route.params);
        break;
      case "thread":
        await renderThread(container, ctx, route.id, route.params);
        break;
      case "topics":
        await renderTopics(container, ctx, route.params);
        break;
      case "sentiment":
        await renderSentiment(container, ctx, route.params);
        break;
      case "responses":
        await renderResponses(container, ctx, route.params);
        break;
      case "resolution":
        await renderResolution(container, ctx, route.params);
        break;
      case "models":
        await renderModels(container, ctx, route.params);
        break;
      default:
        await renderOverview(container, ctx, route.params);
        break;
    }

    if (mySeq !== renderSeq) return;
    dom.content.innerHTML = "";
    dom.content.appendChild(container);
    hydrateMotion(dom.content);
    window.scrollTo({ top: 0, behavior: "instant" });
  } catch (err) {
    if (mySeq !== renderSeq) return;
    dom.content.innerHTML = errorState(err?.message || "Failed to load data from /api/demo.");
    const retryBtn = dom.content.querySelector("[data-retry]");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        clearApiCache();
        ctx.summary = null;
        renderCurrentRoute();
      });
    }
  }
}

function bindGlobalInteractions() {
  dom.menuBtn?.addEventListener("click", () => setDrawerOpen(true));
  dom.dockMoreBtn?.addEventListener("click", () => setDrawerOpen(true));
  dom.closeBtn?.addEventListener("click", () => setDrawerOpen(false));
  dom.backdrop?.addEventListener("click", () => setDrawerOpen(false));

  dom.provPillBtn?.addEventListener("click", () => {
    const nextOpen = Boolean(dom.provPopover?.hidden);
    if (dom.provPopover) dom.provPopover.hidden = !nextOpen;
    dom.provPillBtn.setAttribute("aria-expanded", String(nextOpen));
  });
  dom.closeProvBtn?.addEventListener("click", () => {
    if (dom.provPopover) dom.provPopover.hidden = true;
    dom.provPillBtn?.setAttribute("aria-expanded", "false");
  });

  dom.openPaletteBtn?.addEventListener("click", () => openPalette(dom.globalSearchInput?.value || ""));
  dom.paletteCloseBtn?.addEventListener("click", () => closePalette());
  dom.paletteBackdrop?.addEventListener("click", (e) => {
    if (e.target === dom.paletteBackdrop) closePalette();
  });
  dom.paletteInput?.addEventListener("input", (e) => updatePalette(e.target.value));
  dom.paletteInput?.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); movePaletteFocus(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); movePaletteFocus(-1); }
    else if (e.key === "Enter") {
      const active = dom.paletteResults?.querySelector(".palette-item.is-focused[href]");
      if (active) {
        e.preventDefault();
        window.location.hash = active.getAttribute("href");
        closePalette();
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (dom.paletteBackdrop && !dom.paletteBackdrop.hidden) closePalette();
      if (dom.sidebar?.classList.contains("is-open")) setDrawerOpen(false);
      if (dom.provPopover && !dom.provPopover.hidden) {
        dom.provPopover.hidden = true;
        dom.provPillBtn?.setAttribute("aria-expanded", "false");
      }
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openPalette();
    } else if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      if (tag !== "input" && tag !== "textarea" && tag !== "select") {
        e.preventDefault();
        dom.globalSearchInput?.focus();
      }
    }
  });

  dom.globalSearchForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = (dom.globalSearchInput?.value || "").trim();
    window.location.hash = q ? `#/explorer?q=${encodeURIComponent(q)}` : "#/explorer";
    dom.globalSearchInput?.blur();
  });

  document.addEventListener("click", async (e) => {
    const chip = e.target.closest("[data-copy-value]");
    if (!chip) return;
    e.preventDefault();
    const val = chip.dataset.copyValue;
    try {
      await navigator.clipboard.writeText(val);
      showToast(`Copied ${val}`);
    } catch {
      showToast(`Identifier: ${val}`);
    }
  });

  window.addEventListener("hashchange", renderCurrentRoute);
}

initTheme();
bindGlobalInteractions();
renderCurrentRoute();
