/**
 * View 7: Canonical Community Responses Studio
 * STRICTLY preserves the exact 6 canonical response categories and order:
 * 1. agreement  2. personal_experience  3. opinion  4. advice  5. solution  6. counterargument
 * Features an interactive animated 6-segment radial wheel + 3-example interactive switcher per role.
 */
import { api } from "../api.js?v=6.2";
import { bindWheelInteractions, horizontalBarChart, radialResponseWheel } from "../charts.js?v=6.2";
import { badge, callout, deck, tallyStrip } from "../components.js?v=6.2";
import { CANONICAL_RESPONSE_TYPES, RESPONSE_META, esc, num, percent, share } from "../format.js?v=6.2";

export async function renderResponses(container, ctx) {
  const [res, summary] = await Promise.all([
    api.responses(),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
  ]);
  ctx.summary = summary;

  const byKey = Object.fromEntries((res.items || []).map((item) => [item.label || item.response_type, item]));
  const ordered = CANONICAL_RESPONSE_TYPES.map((key, index) => ({
    rank: index + 1,
    key,
    meta: RESPONSE_META[key],
    count: Number(byKey[key]?.count || 0),
    examples: byKey[key]?.examples || [],
  }));

  const total = ordered.reduce((s, r) => s + r.count, 0) || Number(summary.comments || 1);
  const adviceCount = byKey.advice?.count || 0;
  const solutionCount = byKey.solution?.count || 0;
  const actionableTotal = adviceCount + solutionCount;
  const discourseTotal = total - actionableTotal;
  const actionablePct = (actionableTotal / total) * 100;
  const discoursePct = (discourseTotal / total) * 100;

  const barRows = ordered.map((item) => ({
    label: `${item.rank}. ${item.meta.label}`,
    sublabel: item.key,
    count: item.count,
    color: item.meta.color,
    href: `#/explorer?response=${encodeURIComponent(item.key)}`,
  }));

  const tallyRow = tallyStrip([
    {
      label: "Canonical taxonomy",
      valueText: "6",
      countTo: 6,
      unit: "fixed roles",
      accent: true,
      note: "Exact names and order locked across API, training, and UI",
    },
    {
      label: "Classified comments",
      valueText: num(total),
      countTo: total,
      note: "6-class GPU MuRIL predictions across anonymized community replies",
    },
    {
      label: "Actionable roles",
      valueText: `${actionablePct.toFixed(1)}%`,
      countTo: actionablePct,
      decimals: 1,
      suffix: "%",
      ratioPct: actionablePct,
      color: "var(--color-positive)",
      note: `${num(adviceCount)} advice, ${num(solutionCount)} solution (distinct from verified resolution)`,
      href: "#/resolution",
    },
    {
      label: "Discourse & experience",
      valueText: `${discoursePct.toFixed(1)}%`,
      countTo: discoursePct,
      decimals: 1,
      suffix: "%",
      ratioPct: discoursePct,
      color: "var(--color-role-experience)",
      note: "Agreement, personal_experience, opinion, and counterargument",
    },
  ], "Canonical response summary");

  const exampleIndexes = Object.fromEntries(CANONICAL_RESPONSE_TYPES.map((k) => [k, 0]));

  function renderRoleCard(item) {
    const groupTone = item.meta.group === "actionable" ? "positive" : "neutral";
    const groupText = item.meta.group === "actionable" ? "Actionable intervention" : "Community discourse";
    const exIdx = exampleIndexes[item.key] || 0;
    const activeEx = item.examples[exIdx] || item.examples[0] || null;

    return `
      <article class="response-role-card anim-enter" data-canonical-role="${esc(item.key)}">
        <header class="response-role-head">
          <div>
            <div class="response-role-id">
              <span class="num role-order-num">0${item.rank}</span>
              <span class="swatch-inline" style="background:${item.meta.color}" aria-hidden="true"></span>
              <h3 style="margin:0;font-size:var(--text-md)">${esc(item.meta.label)}</h3>
              <code class="role-canonical-code">${esc(item.key)}</code>
            </div>
            <div style="margin-top:var(--space-xs);display:flex;gap:var(--space-xs);flex-wrap:wrap">
              ${badge(groupText, groupTone)}
            </div>
          </div>
          <div class="response-role-stat">
            <strong class="num">${num(item.count)}</strong>
            <span class="num">${share(item.count, total)}</span>
          </div>
        </header>

        <p class="response-role-def">${esc(item.meta.definition)}</p>

        <div class="example-switcher">
          <span class="muted" style="font-size:var(--text-xs)">Representative community replies (${item.examples.length} returned)</span>
          ${item.examples.length > 1
            ? `<div class="example-tabs" role="group" aria-label="Switch example for ${esc(item.key)}">
                ${item.examples.map((_, idx) => `<button type="button" class="ex-tab-btn ${idx === exIdx ? "is-active" : ""}" data-ex-role="${esc(item.key)}" data-ex-idx="${idx}">${idx + 1}</button>`).join("")}
              </div>`
            : ""}
        </div>

        <div class="preview-comment-item" data-ex-slot="${esc(item.key)}">
          ${activeEx
            ? `<p>“${esc(activeEx.text)}”</p>
               <div class="topic-example-foot">
                 <a href="#/thread/${encodeURIComponent(activeEx.post_id)}">${esc(activeEx.post_title)}</a>
                 <span class="num muted">Confidence ${percent(activeEx.confidence)}</span>
               </div>`
            : `<p class="muted">No classified examples returned.</p>`}
        </div>

        <details class="more-examples-details" style="font-size:var(--text-xs)">
          <summary style="cursor:pointer;color:var(--color-accent);font-weight:600">Inspect all ${item.examples.length} API examples simultaneously</summary>
          <div class="preview-comment-list" style="margin-top:var(--space-xs)">
            ${item.examples.map((ex) => `
              <div class="preview-comment-item">
                <p>“${esc(ex.text)}”</p>
                <div class="topic-example-foot">
                  <a href="#/thread/${encodeURIComponent(ex.post_id)}">${esc(ex.post_title)}</a>
                  <span class="num muted">Confidence ${percent(ex.confidence)}</span>
                </div>
              </div>
            `).join("")}
          </div>
        </details>

        <div style="margin-top:auto">
          <a class="btn btn-quiet btn-sm" href="#/explorer?response=${encodeURIComponent(item.key)}">Filter discussions with ${esc(item.key)}</a>
        </div>
      </article>
    `;
  }

  container.innerHTML = `
    ${callout(
      `The six canonical community response classes remain locked in exact order: <code>agreement</code>, <code>personal_experience</code>, <code>opinion</code>, <code>advice</code>, <code>solution</code>, <code>counterargument</code>. Counts below reflect GPU MuRIL 6-class predictions over anonymized public Reddit replies.`,
      { tone: "info", title: "Canonical taxonomy:" }
    )}
    ${tallyRow}
    ${deck("Canonical six-class distribution wheel", {
      subtitle: "Hover any arc on the wheel to inspect exact shares, or click any role to open matching threads in Discussion workbench",
      body: `
        <div class="wheel-showcase">
          ${radialResponseWheel(ordered, total)}
          ${horizontalBarChart(barRows, { total, unit: "comments", showShare: true })}
        </div>
      `,
    })}
    <div class="response-role-grid">
      ${ordered.map(renderRoleCard).join("")}
    </div>
    ${(res.difficult_boundaries || []).length ? deck("Difficult classification boundaries (held-out error analysis)", {
      subtitle: "Documented semantic overlaps observed across the 6 canonical community response roles on unseen Reddit discussions",
      body: `
        <div class="why-grid">
          ${(res.difficult_boundaries || []).map((b, i) => `
            <div class="why-card">
              <span class="why-num num">0${i + 1}</span>
              <div>
                <h4>${esc(b.boundary)}</h4>
                <p>${esc(b.reason)}</p>
                <p class="muted" style="font-size:var(--text-xs);margin-top:4px"><strong>Empirical finding:</strong> ${esc(b.frozen_test_note)}</p>
              </div>
            </div>
          `).join("")}
        </div>
      `,
    }) : ""}
  `;

  bindWheelInteractions(container, total);

  container.querySelectorAll("[data-ex-role]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const role = btn.dataset.exRole;
      const idx = Number(btn.dataset.exIdx || 0);
      exampleIndexes[role] = idx;
      const item = ordered.find((o) => o.key === role);
      const ex = item?.examples?.[idx];
      const card = container.querySelector(`[data-canonical-role="${CSS.escape(role)}"]`);
      if (card && ex) {
        card.querySelectorAll("[data-ex-role]").forEach((b) => {
          b.classList.toggle("is-active", Number(b.dataset.exIdx) === idx);
        });
        const slot = card.querySelector(`[data-ex-slot="${CSS.escape(role)}"]`);
        if (slot) {
          slot.innerHTML = `
            <p>“${esc(ex.text)}”</p>
            <div class="topic-example-foot">
              <a href="#/thread/${encodeURIComponent(ex.post_id)}">${esc(ex.post_title)}</a>
              <span class="num muted">Confidence ${percent(ex.confidence)}</span>
            </div>
          `;
        }
      }
    });
  });
}
