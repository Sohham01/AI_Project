/**
 * Motion & Interaction Engine for RCI Dusk Studio.
 * Handles GPU-safe staggered entrances, cubic-out counter roll-ups, SVG ring sweeps,
 * stroke-dashoffset path draw-ins, and the shared cursor-tracking glass tooltip.
 * Respects prefers-reduced-motion: reduce cleanly.
 */

const REDUCED_MOTION_MQ = window.matchMedia("(prefers-reduced-motion: reduce)");

export function prefersReducedMotion() {
  return REDUCED_MOTION_MQ.matches;
}

/**
 * Shared floating frosted-glass tooltip positioned safely inside the viewport.
 */
export function showVizTooltip(html, clientX, clientY) {
  const el = document.querySelector("#viz-tooltip");
  if (!el) return;
  el.innerHTML = html;
  el.hidden = false;
  const pad = 14;
  const rect = el.getBoundingClientRect();
  let x = clientX + 16;
  let y = clientY - rect.height - 14;
  if (x + rect.width + pad > window.innerWidth) {
    x = Math.max(pad, clientX - rect.width - 16);
  }
  if (y < pad) {
    y = Math.min(window.innerHeight - rect.height - pad, clientY + 18);
  }
  el.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`;
  el.style.opacity = "1";
}

export function hideVizTooltip() {
  const el = document.querySelector("#viz-tooltip");
  if (!el) return;
  el.style.opacity = "0";
  el.hidden = true;
}

/**
 * Smooth cubic-out numeric count-up for elements tagged with data-count-to.
 */
export function animateCounters(root) {
  const nodes = root.querySelectorAll("[data-count-to]");
  if (!nodes.length) return;
  const reduced = prefersReducedMotion();

  nodes.forEach((node) => {
    const target = Number(node.dataset.countTo);
    const decimals = Number(node.dataset.countDecimals || 0);
    const suffix = node.dataset.countSuffix || "";
    const prefix = node.dataset.countPrefix || "";
    if (!Number.isFinite(target)) return;

    const formatVal = (v) => {
      const formatted = decimals > 0
        ? v.toFixed(decimals)
        : Math.round(v).toLocaleString("en-US");
      return `${prefix}${formatted}${suffix}`;
    };

    if (reduced) {
      node.textContent = formatVal(target);
      return;
    }

    const duration = 640;
    const start = performance.now();
    node.textContent = formatVal(0);

    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = formatVal(target * eased);
      if (t < 1) requestAnimationFrame(step);
      else node.textContent = formatVal(target);
    }
    requestAnimationFrame(step);
  });
}

/**
 * Smoothly sweeps SVG circular ring arcs from 0% to their target stroke-dashoffset.
 */
export function animateRings(root) {
  const arcs = root.querySelectorAll("[data-ring-target]");
  if (!arcs.length) return;
  const reduced = prefersReducedMotion();

  arcs.forEach((arc, idx) => {
    const targetOffset = Number(arc.dataset.ringTarget);
    const circ = Number(arc.dataset.ringCirc);
    if (!Number.isFinite(targetOffset) || !Number.isFinite(circ)) return;
    if (reduced) {
      arc.style.strokeDashoffset = String(targetOffset);
      return;
    }
    arc.style.transition = "none";
    arc.style.strokeDashoffset = String(circ);
    setTimeout(() => {
      arc.style.transition = "";
      arc.style.strokeDashoffset = String(targetOffset);
    }, 40 + idx * 70);
  });
}

/**
 * Smoothly draws SVG paths (spline curves & Sankey ribbons) using stroke-dashoffset.
 */
export function animateDrawPaths(root) {
  const paths = root.querySelectorAll("[data-draw-path]");
  if (!paths.length || prefersReducedMotion()) return;

  paths.forEach((path, idx) => {
    try {
      const len = path.getTotalLength();
      if (!Number.isFinite(len) || len <= 0) return;
      path.style.strokeDasharray = `${len} ${len}`;
      path.style.strokeDashoffset = String(len);
      path.style.transition = "none";
      requestAnimationFrame(() => {
        setTimeout(() => {
          path.style.transition = "stroke-dashoffset 780ms cubic-bezier(0.22, 1, 0.36, 1)";
          path.style.strokeDashoffset = "0";
          setTimeout(() => {
            path.style.strokeDasharray = "";
            path.style.strokeDashoffset = "";
            path.style.transition = "";
          }, 840);
        }, 25 + Math.min(idx, 12) * 35);
      });
    } catch {
      // Non-path geometry or detached node: skip safely
    }
  });
}

/**
 * Applies smooth staggered spring entrance indices to direct visual modules.
 */
export function hydrateMotion(root) {
  hideVizTooltip();
  const blocks = root.querySelectorAll(".anim-enter");
  blocks.forEach((el, idx) => {
    el.style.setProperty("--stagger-i", String(Math.min(idx, 8)));
  });
  animateCounters(root);
  animateRings(root);
  animateDrawPaths(root);
}
