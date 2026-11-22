/**
 * View 10: Cross-IIM & Cross-Language Comparisons + Live Multilingual Analyzer
 * Directly answers Research Question #8 ("How do discussions differ across topics,
 * languages and IIMs?") plus Tasks 5 & 6 (LID failure study, preprocessing ablation,
 * and interactive live multilingual comment analysis).
 */
import { api } from "../api.js?v=6.0";
import { badge, callout, deck, emptyState } from "../components.js?v=6.0";
import { esc, num, percent, pretty } from "../format.js?v=6.0";

const cmpState = {
  tab: "iims", // iims | language | analyzer
  analyzerText: "Bhai IIM ABC ka summer placement scene kaisa hai 😭",
  analyzerResult: null,
  analyzerBusy: false,
  analyzerError: "",
};

const PDF_PRESETS = [
  {
    label: "Hinglish + Emoji (PDF p.1)",
    text: "Bhai IIM ABC ka summer placement scene kaisa hai 😭",
  },
  {
    label: "Odia + English Mixed (PDF p.1)",
    text: "ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro",
  },
  {
    label: "Banglish Romanized (PDF p.1)",
    text: "IIM e giye ki actually ROI pawa jay?",
  },
  {
    label: "Gen-Z Slang + Skull (PDF p.1)",
    text: "Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl.",
  },
  {
    label: "Actionable Solution (PDF p.6)",
    text: "Try applying for the need-based financial aid scholarship and reach out to seniors on LinkedIn.",
  },
];

function renderIimTab(crossIim) {
  const rows = crossIim?.iims || [];
  if (!rows.length) {
    return emptyState({
      title: "No cross-IIM mentions available",
      description: "Re-run the live corpus build to extract IIM entity comparisons.",
    });
  }

  const maxMentions = Math.max(...rows.map((r) => Number(r.direct_comment_mentions || 0)), 1);

  const tableRows = rows.map((r) => {
    const negPct = Number(r.sentiment?.negative || 0) * 100;
    const neuPct = Number(r.sentiment?.neutral || 0) * 100;
    const posPct = Number(r.sentiment?.positive || 0) * 100;
    const solPct = Number(r.solution_rate || 0) * 100;
    const volW = Math.max(4, (Number(r.direct_comment_mentions || 0) / maxMentions) * 100);
    return `
      <tr>
        <td>
          <strong>${esc(r.iim)}</strong>
          <div style="margin-top:4px;display:flex;align-items:center;gap:var(--space-xs)">
            <div class="kw-track" style="width:84px"><span class="kw-fill" style="width:${volW.toFixed(1)}%"></span></div>
            <span class="muted num" style="font-size:var(--text-xs)">${num(r.direct_comment_mentions)} replies · ${num(r.thread_mentions)} threads</span>
          </div>
        </td>
        <td><a class="table-primary-link" href="#/issues?selected=${encodeURIComponent(r.top_issue_key)}">${esc(r.top_issue_label)}</a></td>
        <td>
          <div style="display:flex;height:8px;border-radius:999px;overflow:hidden;min-width:110px;background:var(--color-surface-2)" title="Positive ${posPct.toFixed(1)}% · Neutral ${neuPct.toFixed(1)}% · Negative ${negPct.toFixed(1)}%">
            <span style="width:${posPct.toFixed(1)}%;background:var(--color-positive)"></span>
            <span style="width:${neuPct.toFixed(1)}%;background:var(--color-muted)"></span>
            <span style="width:${negPct.toFixed(1)}%;background:var(--color-negative)"></span>
          </div>
          <div class="muted num" style="font-size:var(--text-xs);margin-top:3px">
            <span style="color:var(--color-negative)">${negPct.toFixed(0)}% neg</span> · ${posPct.toFixed(0)}% pos
          </div>
        </td>
        <td>${badge(pretty(r.dominant_emotion), "neutral")}</td>
        <td class="num"><strong>${solPct.toFixed(1)}%</strong> <span class="muted">(${num(r.solution_count)})</span></td>
      </tr>
    `;
  }).join("");

  return `
    ${callout(
      `<strong>${num(crossIim.total_distinct_iims_detected)} distinct Indian IIMs</strong> are referenced across the anonymized corpus (showing all ${num(rows.length)} IIMs with &ge;${num(crossIim.min_mentions_threshold)} direct comment mentions). Admissions/waitlist dominates newer campuses (Sambalpur, Mumbai, Sirmaur), whereas comparative placement &amp; ROI discussions lead across peer IIM mentions.`,
      { tone: "info", title: "Cross-IIM contrast:" }
    )}
    ${deck("Discussions across mentioned IIMs", {
      subtitle: "Direct comment mentions, dominant issue domain, predicted polarity mix, top emotion, and XLM-R actionable solution share",
      body: `
        <div class="table-scroll">
          <table class="data-table" aria-label="Cross-IIM comparison across mentioned Indian Institutes of Management">
            <thead>
              <tr>
                <th scope="col">Mentioned IIM &amp; volume</th>
                <th scope="col">Top issue domain</th>
                <th scope="col">Sentiment mix (pos / neu / neg)</th>
                <th scope="col">Dominant emotion</th>
                <th scope="col" class="num">Solution rate</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      `,
    })}
  `;
}

function renderLanguageTab(crossLang, ablations) {
  const regs = crossLang?.registers || [];
  const rawCounts = crossLang?.raw_language_counts || {};
  const task5 = ablations?.task5_lid_study || null;
  const task6 = ablations?.task6_preprocessing_ablation || null;

  const regCards = regs.map((r) => {
    const negPct = (Number(r.sentiment?.negative || 0) * 100).toFixed(1);
    const posPct = (Number(r.sentiment?.positive || 0) * 100).toFixed(1);
    const solPct = (Number(r.solution_rate || 0) * 100).toFixed(1);
    const emPct = (Number(r.emoji_rate || 0) * 100).toFixed(1);
    const topRoles = (r.response_distribution || [])
      .slice()
      .sort((a, b) => b.count - a.count)
      .slice(0, 3)
      .map((x) => `${pretty(x.label)} (${percent(x.share, 0)})`)
      .join(" · ");
    return `
      <div class="deck" style="padding:var(--space-md)">
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:var(--space-xs)">
          <h4 style="margin:0">${esc(r.label)}</h4>
          ${badge(`${num(r.count)} (${percent(r.share, 1)})`, r.key === "code_mixed_indic" ? "warning" : "neutral")}
        </div>
        <p class="muted" style="font-size:var(--text-xs);margin:4px 0 var(--space-sm)">${esc(r.description)}</p>
        <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--space-xs);margin-bottom:var(--space-sm)">
          <div><span class="muted" style="font-size:var(--text-xs)">Negative polarity</span><div class="num" style="font-weight:700;color:var(--color-negative)">${negPct}%</div></div>
          <div><span class="muted" style="font-size:var(--text-xs)">Actionable solutions</span><div class="num" style="font-weight:700;color:var(--color-positive)">${solPct}%</div></div>
          <div><span class="muted" style="font-size:var(--text-xs)">Emoji density</span><div class="num" style="font-weight:600">${emPct}%</div></div>
          <div><span class="muted" style="font-size:var(--text-xs)">Avg length</span><div class="num" style="font-weight:600">${r.avg_word_count} words</div></div>
        </div>
        <div class="muted" style="font-size:var(--text-xs)">Top roles: ${esc(topRoles)}</div>
        <div class="muted" style="font-size:var(--text-xs);margin-top:2px">Positive share: <span class="num">${posPct}%</span></div>
      </div>
    `;
  }).join("");

  let task5Card = "";
  if (task5) {
    const probes = task5.controlled_probe_accuracy || {};
    const exRows = (task5.real_misclassification_examples || []).slice(0, 5).map((e) => `
      <tr>
        <td style="max-width:340px">“${esc(e.text)}”</td>
        <td><code>${esc(e.langdetect_pred)}</code> ${badge("bogus tag", "negative")}</td>
        <td><code>${esc(e.indian_reddit_lid_pred)}</code> ${badge(e.script, "positive")}</td>
      </tr>
    `).join("");
    task5Card = deck("Task 5 — Existing LID (langdetect) limitation study", {
      subtitle: `Evaluated across all ${num(task5.corpus_size)} real comments + ${num(probes.n_probes || 12)} multilingual probes (English, Hindi, Bengali, Odia, Hinglish, Banglish, Roman-Odia)`,
      body: `
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:var(--space-sm);margin-bottom:var(--space-md)">
          <div style="padding:var(--space-sm);border:1px solid var(--color-rule);border-radius:var(--radius-sm);background:var(--color-surface-2);display:flex;flex-direction:column;gap:4px">
            <span class="muted" style="font-size:var(--text-xs)">langdetect bogus tags (corpus)</span>
            <strong class="num" style="font-size:1.35rem;color:var(--color-negative)">${percent(task5.langdetect_bogus_non_indian_rate, 1)}</strong>
            <span class="muted" style="font-size:11px">${num(task5.langdetect_bogus_non_indian_total)} comments mislabeled (id/sw/so/cy…)</span>
          </div>
          <div style="padding:var(--space-sm);border:1px solid var(--color-rule);border-radius:var(--radius-sm);background:var(--color-surface-2);display:flex;flex-direction:column;gap:4px">
            <span class="muted" style="font-size:var(--text-xs)">langdetect bogus on code-mixed</span>
            <strong class="num" style="font-size:1.35rem;color:var(--color-negative)">${percent(task5.langdetect_bogus_on_indic_rate, 1)}</strong>
            <span class="muted" style="font-size:11px">${num(task5.langdetect_bogus_on_indic_count)} of ${num(task5.code_mixed_or_roman_indic_count)} Indic/Hinglish comments</span>
          </div>
          <div style="padding:var(--space-sm);border:1px solid var(--color-rule);border-radius:var(--radius-sm);background:var(--color-surface-2);display:flex;flex-direction:column;gap:4px">
            <span class="muted" style="font-size:var(--text-xs)">Multilingual probe accuracy</span>
            <strong class="num" style="font-size:1.35rem;color:var(--color-positive)">${percent(probes.indian_reddit_lid_accuracy, 1)}</strong>
            <span class="muted" style="font-size:11px">IndianRedditLID vs. ${percent(probes.langdetect_accuracy, 1)} langdetect</span>
          </div>
        </div>
        ${exRows ? `
          <div class="table-scroll">
            <table class="data-table" aria-label="Real Hinglish comments misclassified by langdetect">
              <thead><tr><th>Real anonymized Reddit comment</th><th>langdetect output</th><th>IndianRedditLID output</th></tr></thead>
              <tbody>${exRows}</tbody>
            </table>
          </div>
        ` : ""}
      `,
    });
  }

  let task6Card = "";
  if (task6) {
    const tfidf = task6.tfidf_5task_ablation || {};
    const muril = task6.muril_wwm_affect_ablation || {};
    const order = [
      ["raw", "Raw text (whitespace only)"],
      ["conventional_nlp", "Conventional NLP (stopwords + stemming + strip punct/emoji)"],
      ["emoji_removed", "Emoji removed (keep stopwords & punctuation)"],
      ["emoji_token_slang", "Semantic <EMOJI_name> tokens + Gen-Z slang expansion"],
      ["emoji_affect_slang", "Semantic affect phrases (sadness/disbelief) + Gen-Z slang"],
    ];
    const rowsHtml = order.map(([key, label]) => {
      const tf = tfidf[key] || {};
      const mu = muril[key] || {};
      const isConv = key === "conventional_nlp";
      return `
        <tr class="${isConv ? "" : ""}">
          <td><strong>${esc(label)}</strong></td>
          <td class="num"><strong>${percent(tf.mean_5task_oof_f1, 2, "—")}</strong></td>
          <td class="num">${percent(mu.sentiment?.oof_macro_f1, 2, "—")}</td>
          <td class="num">${percent(mu.emotion?.oof_macro_f1, 2, "—")}</td>
          <td class="num"><strong>${percent(mu.mean_affect_oof_f1, 2, "—")}</strong></td>
        </tr>
      `;
    }).join("");
    task6Card = deck("Task 6 — Empirical preprocessing & emoji ablation (5-fold OOF CV)", {
      subtitle: `${esc(task6.protocol || "5-fold post_id-grouped OOF CV on untouched V1 trainval (367)")}`,
      body: `
        <div class="table-scroll">
          <table class="data-table" style="font-size:var(--text-xs)" aria-label="Task 6 preprocessing ablation results">
            <thead>
              <tr>
                <th scope="col">Strategy</th>
                <th scope="col" class="num">TF-IDF 5-Task</th>
                <th scope="col" class="num">MuRIL Sent.</th>
                <th scope="col" class="num">MuRIL Emo.</th>
                <th scope="col" class="num">Affect Mean</th>
              </tr>
            </thead>
            <tbody>${rowsHtml}</tbody>
          </table>
        </div>
      `,
    });
  }

  return `
    ${callout(
      `<strong>Behavioral register divide:</strong> Code-mixed Hinglish &amp; Indic replies carry nearly <strong>2&times; higher negative sentiment</strong> and <strong>2.5&times; higher emoji usage</strong> than monolingual English, while actionable peer solutions are written predominantly in longer English comments. Detected raw tags: ${Object.entries(rawCounts).map(([k, v]) => `<code>${esc(k)}</code> (${num(v)})`).join(", ")}.`,
      { tone: "info", title: "Cross-language insight:" }
    )}
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:var(--space-md);margin-bottom:var(--space-md)">
      ${regCards}
    </div>
    ${task5Card}
    ${task6Card}
  `;
}

function renderAnalyzerTab() {
  const res = cmpState.analyzerResult;
  const presets = PDF_PRESETS.map((p, idx) => `
    <button type="button" class="btn btn-quiet btn-sm" data-preset-idx="${idx}">${esc(p.label)}</button>
  `).join("");

  let resultHtml = "";
  if (cmpState.analyzerError) {
    resultHtml = callout(esc(cmpState.analyzerError), { tone: "warning", title: "Analyzer notice:" });
  } else if (res) {
    const lid = res.language_detection || {};
    const pp = res.preprocessing || {};
    const preds = res.predictions || {};
    const emojis = pp.emojis_detected || [];
    const slang = pp.slang_detected || [];
    const iims = pp.iims_detected || [];
    const neighbors = res.nearest_v6_neighbors || [];

    resultHtml = `
      <div class="studio-grid-2" style="margin-top:var(--space-md)">
        ${deck("1. Language, script & informal text pipeline (Tasks 5–6)", {
          subtitle: "Deterministic real-time preprocessing — zero third-party LID hallucinations",
          body: `
            <div style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:var(--space-sm)">
              ${badge(`Language: ${lid.language || "und"}`, "info")}
              ${badge(`Script: ${lid.script || "unknown"}`, "neutral")}
              ${badge(lid.is_code_mixed ? "Code-mixed: YES" : "Code-mixed: NO", lid.is_code_mixed ? "warning" : "neutral")}
              ${iims.map((m) => badge(m, "positive")).join("")}
            </div>
            ${(lid.matched_markers || []).length ? `<p class="muted" style="font-size:var(--text-xs);margin-bottom:var(--space-xs)">Romanized Indic markers matched: ${lid.matched_markers.map((m) => `<code>${esc(m)}</code>`).join(" ")}</p>` : ""}
            <div style="margin-bottom:var(--space-sm)">
              <div class="muted" style="font-size:var(--text-xs);margin-bottom:2px">Semantic emoji token + slang normalization:</div>
              <div style="padding:var(--space-xs) var(--space-sm);border-radius:var(--radius-sm);background:var(--color-surface-2);font-family:var(--font-mono);font-size:var(--text-xs)">${esc(pp.normalized_emoji_token || "")}</div>
            </div>
            <div style="margin-bottom:var(--space-sm)">
              <div class="muted" style="font-size:var(--text-xs);margin-bottom:2px">Semantic affect verbalization (Task 6):</div>
              <div style="padding:var(--space-xs) var(--space-sm);border-radius:var(--radius-sm);background:var(--color-surface-2);font-family:var(--font-mono);font-size:var(--text-xs)">${esc(pp.normalized_semantic_affect || "")}</div>
            </div>
            ${emojis.length ? `<div style="margin-bottom:var(--space-xs)"><strong style="font-size:var(--text-xs)">Emojis decoded:</strong> ${emojis.map((e) => `${esc(e.emoji)} → <code>${esc(e.semantic_affect)}</code>`).join(" · ")}</div>` : ""}
            ${slang.length ? `<div><strong style="font-size:var(--text-xs)">Gen-Z / informal cues:</strong> ${slang.map((s) => `<code>${esc(s.term)}</code> (${esc(s.meaning)})`).join(" · ")}</div>` : ""}
          `,
        })}
        ${deck("2. Five-task V6 corpus-neighbor projection", {
          subtitle: "Projected from k=12 nearest precomputed V6 neural predictions (0 GPU, <15 ms)",
          body: `
            <div style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:var(--space-sm)">
              ${badge(`Sentiment: ${pretty(preds.sentiment?.label || "neutral")}`, preds.sentiment?.label === "negative" ? "negative" : preds.sentiment?.label === "positive" ? "positive" : "neutral")}
              ${badge(`Response role: ${pretty(preds.response?.label || "opinion")}`, "info")}
              ${badge(`Solution: ${preds.solution?.is_solution ? "YES" : "NO"} (${percent(preds.solution?.probability, 0)})`, preds.solution?.is_solution ? "positive" : "neutral")}
            </div>
            <div style="margin-bottom:var(--space-xs);font-size:var(--text-xs)">
              <strong>Issues:</strong> ${(preds.issues?.display_labels || []).length ? preds.issues.display_labels.map((l) => badge(l, "neutral")).join(" ") : `<span class="muted">None above 30%</span>`}
            </div>
            <div style="margin-bottom:var(--space-sm);font-size:var(--text-xs)">
              <strong>Emotions:</strong> ${(preds.emotions?.labels || []).length ? preds.emotions.labels.map((l) => badge(pretty(l), "warning")).join(" ") : `<span class="muted">Neutral / low intensity</span>`}
            </div>
            <p class="muted" style="font-size:var(--text-xs)">${esc(res.inference_notice || "")}</p>
          `,
        })}
      </div>
      ${neighbors.length ? deck("Closest real V6-predicted corpus replies (interpretable evidence)", {
        subtitle: "Actual anonymized Reddit comments driving the nearest-neighbor projection above",
        body: `
          <div style="display:grid;gap:var(--space-xs)">
            ${neighbors.map((n) => `
              <div style="padding:var(--space-sm);border:1px solid var(--color-rule);border-radius:var(--radius-sm)">
                <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:4px">
                  <a class="table-primary-link" href="#/thread/${encodeURIComponent(n.post_id)}">${esc(n.post_title)}</a>
                  <span class="num muted" style="font-size:var(--text-xs)">cos = ${Number(n.similarity).toFixed(3)} · ${esc(n.v6_sentiment)} · ${esc(n.v6_response)} · sol=${n.v6_is_solution ? "yes" : "no"}</span>
                </div>
                <p style="margin:0;font-size:var(--text-sm)">“${esc(n.text)}”</p>
              </div>
            `).join("")}
          </div>
        `,
      }) : ""}
    `;
  }

  return `
    ${deck("Live multilingual comment analyzer (Tasks 5 & 6 interactive lab)", {
      subtitle: "Test Hinglish, Banglish, Odia, Gen-Z slang, and emojis live — runs with zero GPU on Docker & Vercel",
      body: `
        <div style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:var(--space-sm)">
          ${presets}
        </div>
        <form id="live-analyzer-form" style="display:flex;flex-direction:column;gap:var(--space-sm)">
          <textarea id="live-analyzer-input" rows="3" style="width:100%;padding:var(--space-sm);border-radius:var(--radius-sm);border:1px solid var(--color-rule);background:var(--color-surface-2);color:var(--color-ink);font-family:var(--font-body);font-size:var(--text-sm)" placeholder="Enter any English, Hinglish, Banglish, Odia, emoji, or Gen-Z student Reddit comment...">${esc(cmpState.analyzerText)}</textarea>
          <div style="display:flex;align-items:center;gap:var(--space-sm)">
            <button type="submit" class="btn btn-primary btn-sm" ${cmpState.analyzerBusy ? "disabled" : ""}>${cmpState.analyzerBusy ? "Analyzing…" : "Analyze comment"}</button>
            <span class="muted" style="font-size:var(--text-xs)">Shows real-time language/script ID, emoji affect decoding, slang handling, and nearest V6 predictions</span>
          </div>
        </form>
      `,
    })}
    ${resultHtml}
  `;
}

export async function renderCompare(container, ctx, params = {}) {
  const data = await api.comparisons();
  if (params.tab && ["iims", "language", "analyzer"].includes(params.tab)) {
    cmpState.tab = params.tab;
  }

  async function runAnalyze(text) {
    const clean = String(text || "").trim();
    if (!clean) return;
    cmpState.analyzerText = clean;
    cmpState.analyzerBusy = true;
    cmpState.analyzerError = "";
    paint();
    try {
      cmpState.analyzerResult = await api.analyze(clean);
    } catch (err) {
      cmpState.analyzerError = err?.message || "Live analysis failed.";
    } finally {
      cmpState.analyzerBusy = false;
      paint();
    }
  }

  function paint() {
    const tabs = [
      ["iims", `Cross-IIM (${num(data.cross_iim?.iims?.length || 0)})`],
      ["language", "Languages & Tasks 5–6 Study"],
      ["analyzer", "Live Multilingual Analyzer"],
    ];

    const bodyHtml = cmpState.tab === "iims"
      ? renderIimTab(data.cross_iim)
      : cmpState.tab === "language"
        ? renderLanguageTab(data.cross_language, data.task5_task6_ablations)
        : renderAnalyzerTab();

    container.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-bottom:var(--space-md)" role="tablist" aria-label="Comparative analytics and live lab tabs">
        ${tabs.map(([id, label]) => `
          <button type="button" role="tab" aria-selected="${cmpState.tab === id}" class="btn ${cmpState.tab === id ? "btn-primary" : "btn-quiet"} btn-sm" data-cmp-tab="${id}">${label}</button>
        `).join("")}
      </div>
      ${bodyHtml}
    `;

    container.querySelectorAll("[data-cmp-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        cmpState.tab = btn.dataset.cmpTab;
        paint();
        if (cmpState.tab === "analyzer" && !cmpState.analyzerResult && !cmpState.analyzerBusy) {
          runAnalyze(cmpState.analyzerText);
        }
      });
    });

    container.querySelectorAll("[data-preset-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.presetIdx);
        const preset = PDF_PRESETS[idx];
        if (preset) runAnalyze(preset.text);
      });
    });

    container.querySelector("#live-analyzer-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      const input = container.querySelector("#live-analyzer-input");
      if (input) runAnalyze(input.value);
    });
  }

  paint();
  if (cmpState.tab === "analyzer" && !cmpState.analyzerResult && !cmpState.analyzerBusy) {
    runAnalyze(cmpState.analyzerText);
  }
}
