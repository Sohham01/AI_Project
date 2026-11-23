/**
 * View 10: Cross-IIM & Cross-Language Comparisons + Live Multilingual Analyzer
 * Clean SaaS product view covering cross-IIM contrast, language register analytics,
 * LID & preprocessing benchmarks, and real-time Span-LID + context-aware slang/emoji live lab.
 */
import { api } from "../api.js?v=6.2";
import { badge, callout, deck, emptyState } from "../components.js?v=6.2";
import { downloadCsv } from "../export.js?v=6.2";
import { esc, num, percent, pretty } from "../format.js?v=6.2";

const cmpState = {
  tab: "iims", // iims | language | analyzer
  analyzerText: "Bhai IIM ABC ka summer placement scene kaisa hai 😭",
  analyzerResult: null,
  analyzerBusy: false,
  analyzerError: "",
};

const LIVE_PRESETS = [
  {
    label: "Hinglish + Emoji",
    text: "Bhai IIM ABC ka summer placement scene kaisa hai 😭",
  },
  {
    label: "Odia + English Mixed",
    text: "ଆଜି କ୍ଲାସ୍ ବହୁତ hectic ଥିଲା bro",
  },
  {
    label: "Banglish Romanized",
    text: "IIM e giye ki actually ROI pawa jay?",
  },
  {
    label: "Gen-Z Slang + Skull",
    text: "Bro hostel life 💀... mess food toh literally mid... but seniors helped a lot ngl.",
  },
  {
    label: "Context Guard (Mid-Sem vs. Mid)",
    text: "After mid-semester exams my quiz score is totally cooked, though the placement cell is based on merit.",
  },
  {
    label: "Actionable Solution",
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
          <div class="tag-row" style="margin-top:4px">
            <div class="kw-track" style="width:84px"><span class="kw-fill" style="width:${volW.toFixed(1)}%"></span></div>
            <span class="muted num text-xs">${num(r.direct_comment_mentions)} replies · ${num(r.thread_mentions)} threads</span>
          </div>
        </td>
        <td><a class="table-primary-link" href="#/issues?selected=${encodeURIComponent(r.top_issue_key)}">${esc(r.top_issue_label)}</a></td>
        <td>
          <div class="sentiment-mini-bar" title="Positive ${posPct.toFixed(1)}% · Neutral ${neuPct.toFixed(1)}% · Negative ${negPct.toFixed(1)}%">
            <span style="width:${posPct.toFixed(1)}%;background:var(--color-positive)"></span>
            <span style="width:${neuPct.toFixed(1)}%;background:var(--color-muted)"></span>
            <span style="width:${negPct.toFixed(1)}%;background:var(--color-negative)"></span>
          </div>
          <div class="muted num text-xs" style="margin-top:3px">
            <span class="text-negative">${negPct.toFixed(0)}% neg</span> · ${posPct.toFixed(0)}% pos
          </div>
        </td>
        <td>${badge(pretty(r.dominant_emotion), "neutral")}</td>
        <td class="num"><strong>${solPct.toFixed(1)}%</strong> <span class="muted">(${num(r.solution_count)})</span></td>
      </tr>
    `;
  }).join("");

  return `
    ${callout(
      `<strong>${num(crossIim.total_distinct_iims_detected)} distinct Indian IIMs</strong> are referenced across the anonymized corpus (showing all ${num(rows.length)} IIMs with &ge;${num(crossIim.min_mentions_threshold)} direct comment mentions). Admissions &amp; waitlists dominate newer campuses (Sambalpur, Mumbai, Sirmaur), whereas comparative placement &amp; ROI discussions lead across peer IIM mentions.`,
      { tone: "info", title: "Cross-IIM contrast:" }
    )}
    ${deck("Discussions across mentioned IIMs", {
      subtitle: "Direct comment mentions, dominant issue domain, predicted polarity mix, top emotion, and XLM-R actionable solution share",
      action: `<button type="button" class="btn btn-quiet btn-sm" id="export-iim-csv">Export CSV</button>`,
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
  const lidStudy = ablations?.task5_lid_study || null;
  const prepAblation = ablations?.task6_preprocessing_ablation || null;

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
      <article class="surface-well">
        <div class="surface-well-head">
          <h4 class="text-sm" style="margin:0">${esc(r.label)}</h4>
          ${badge(`${num(r.count)} (${percent(r.share, 1)})`, r.key === "code_mixed_indic" ? "warning" : "neutral")}
        </div>
        <p class="muted text-xs">${esc(r.description)}</p>
        <div class="stat-strip-mini" style="margin-bottom:4px">
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Negative</span>
            <strong class="stat-mini-val is-negative num">${negPct}%</strong>
          </div>
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Solutions</span>
            <strong class="stat-mini-val is-positive num">${solPct}%</strong>
          </div>
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Emoji rate</span>
            <strong class="stat-mini-val num">${emPct}%</strong>
          </div>
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Avg length</span>
            <strong class="stat-mini-val num">${r.avg_word_count}w</strong>
          </div>
        </div>
        <div class="muted text-xs">Top roles: ${esc(topRoles)} · Positive: <span class="num">${posPct}%</span></div>
      </article>
    `;
  }).join("");

  let lidCard = "";
  if (lidStudy) {
    const probes = lidStudy.controlled_probe_accuracy || {};
    const exRows = (lidStudy.real_misclassification_examples || []).slice(0, 5).map((e) => `
      <tr>
        <td style="max-width:340px">“${esc(e.text)}”</td>
        <td><code>${esc(e.langdetect_pred)}</code> ${badge("bogus tag", "negative")}</td>
        <td><code>${esc(e.indian_reddit_lid_pred)}</code> ${badge(e.script, "positive")}</td>
      </tr>
    `).join("");
    lidCard = deck("Language identification benchmark (IndianRedditLID vs. langdetect)", {
      subtitle: `Evaluated across all ${num(lidStudy.corpus_size)} real comments + ${num(probes.n_probes || 12)} multilingual probes (English, Hindi, Bengali, Odia, Hinglish, Banglish, Roman-Odia)`,
      body: `
        <div class="stat-strip-mini">
          <div class="stat-mini-cell">
            <span class="stat-mini-label">langdetect bogus tags</span>
            <strong class="stat-mini-val is-negative num">${percent(lidStudy.langdetect_bogus_non_indian_rate, 1)}</strong>
            <span class="stat-mini-sub">${num(lidStudy.langdetect_bogus_non_indian_total)} comments mislabeled (id/sw/so/cy…)</span>
          </div>
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Bogus on code-mixed</span>
            <strong class="stat-mini-val is-negative num">${percent(lidStudy.langdetect_bogus_on_indic_rate, 1)}</strong>
            <span class="stat-mini-sub">${num(lidStudy.langdetect_bogus_on_indic_count)} of ${num(lidStudy.code_mixed_or_roman_indic_count)} Indic/Hinglish replies</span>
          </div>
          <div class="stat-mini-cell">
            <span class="stat-mini-label">Multilingual probe acc</span>
            <strong class="stat-mini-val is-positive num">${percent(probes.indian_reddit_lid_accuracy, 1)}</strong>
            <span class="stat-mini-sub">IndianRedditLID vs. ${percent(probes.langdetect_accuracy, 1)} langdetect</span>
          </div>
        </div>
        ${exRows ? `
          <div class="table-scroll">
            <table class="data-table" aria-label="Real Hinglish comments misclassified by langdetect">
              <thead><tr><th scope="col">Real anonymized Reddit comment</th><th scope="col">langdetect output</th><th scope="col">IndianRedditLID output</th></tr></thead>
              <tbody>${exRows}</tbody>
            </table>
          </div>
        ` : ""}
      `,
    });
  }

  let ablationCard = "";
  if (prepAblation) {
    const tfidf = prepAblation.tfidf_5task_ablation || {};
    const muril = prepAblation.muril_wwm_affect_ablation || {};
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
      return `
        <tr>
          <td><strong>${esc(label)}</strong></td>
          <td class="num"><strong>${percent(tf.mean_5task_oof_f1, 2, "—")}</strong></td>
          <td class="num">${percent(mu.sentiment?.oof_macro_f1, 2, "—")}</td>
          <td class="num">${percent(mu.emotion?.oof_macro_f1, 2, "—")}</td>
          <td class="num"><strong>${percent(mu.mean_affect_oof_f1, 2, "—")}</strong></td>
        </tr>
      `;
    }).join("");
    ablationCard = deck("Preprocessing, slang &amp; emoji ablation (5-fold OOF CV)", {
      subtitle: `${esc(prepAblation.protocol || "5-fold post_id-grouped OOF CV on untouched trainval split")}`,
      body: `
        <div class="table-scroll">
          <table class="data-table text-xs" aria-label="Preprocessing and emoji ablation results">
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
    <div class="studio-grid-2" style="margin-bottom:var(--space-md)">${regCards}</div>
    ${lidCard}
    ${ablationCard}
  `;
}

function renderAnalyzerTab() {
  const res = cmpState.analyzerResult;
  const presets = LIVE_PRESETS.map((p, idx) => `
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
    const ctxTransforms = pp.context_slang_transformations || [];
    const spans = lid.spans || [];
    const iims = pp.iims_detected || [];
    const neighbors = res.nearest_v6_neighbors || [];

    const spansHtml = spans.length
      ? `
        <div>
          <h4 class="subhead-xs">Span-level language &amp; script segmentation (${spans.length})</h4>
          <div class="tag-row">
            ${spans.map((sp, i) => `
              <span class="tag tag-neutral text-xs" title="${esc(sp.text || "")}">
                Span ${i + 1}: <strong>${esc(sp.language)}</strong>/${esc(sp.script)} (${Math.round(Number(sp.confidence || 0) * 100)}%)
              </span>
            `).join("")}
          </div>
        </div>
      `
      : "";

    const ctxSlangHtml = ctxTransforms.length
      ? `
        <div>
          <h4 class="subhead-xs">Context-aware slang disambiguation guard</h4>
          <div class="grid-stack-xs">
            ${ctxTransforms.map((tr) => `
              <div class="slang-diff-item">
                <code>${esc(tr.term)}</code>
                ${badge(tr.action === "literal_preserved" ? "Literal preserved" : "Slang expanded", tr.action === "literal_preserved" ? "positive" : "info")}
                ${tr.replacement && tr.replacement !== tr.term ? `<span>&rarr; <strong>${esc(tr.replacement)}</strong></span>` : ""}
                <span class="muted">${esc(tr.reason || "")}</span>
              </div>
            `).join("")}
          </div>
        </div>
      `
      : "";

    resultHtml = `
      <div class="studio-grid-2" style="margin-top:var(--space-md)">
        ${deck("1. Span-level LID, context slang &amp; emoji pipeline", {
          subtitle: "Deterministic real-time multilingual analysis — zero GPU, zero third-party LID hallucinations",
          body: `
            <div class="grid-stack-sm">
              <div class="tag-row">
                ${badge(`Language: ${lid.language || "und"}`, "info")}
                ${badge(`Script: ${lid.script || "unknown"}`, "neutral")}
                ${badge(lid.is_code_mixed ? "Code-mixed: YES" : "Code-mixed: NO", lid.is_code_mixed ? "warning" : "neutral")}
                ${lid.confidence !== undefined ? badge(`LID Conf: ${Math.round(Number(lid.confidence) * 100)}%`, "neutral") : ""}
                ${lid.abstain ? badge(`Abstained (${lid.uncertainty_flag || "low_signal"})`, "warning") : ""}
                ${iims.map((m) => badge(m, "positive")).join("")}
              </div>
              ${(lid.matched_markers || []).length ? `<p class="muted text-xs">Romanized Indic markers matched: ${lid.matched_markers.map((m) => `<code>${esc(m)}</code>`).join(" ")}</p>` : ""}
              ${spansHtml}
              ${ctxSlangHtml}
              ${pp.context_slang_normalized && pp.context_slang_normalized !== res.input_text ? `
                <div>
                  <h4 class="subhead-xs">Context-disambiguated text</h4>
                  <div class="code-well">${esc(pp.context_slang_normalized)}</div>
                </div>
              ` : ""}
              <div>
                <h4 class="subhead-xs">Semantic emoji token + slang normalization</h4>
                <div class="code-well">${esc(pp.normalized_emoji_token || "")}</div>
              </div>
              <div>
                <h4 class="subhead-xs">Affect-preserving semantic verbalization</h4>
                <div class="code-well">${esc(pp.normalized_semantic_affect || "")}</div>
              </div>
              ${emojis.length ? `<div class="text-xs"><strong>Emojis decoded:</strong> ${emojis.map((e) => `${esc(e.emoji)} &rarr; <code>${esc(e.semantic_affect)}</code>`).join(" · ")}</div>` : ""}
              ${slang.length ? `<div class="text-xs"><strong>Informal cues:</strong> ${slang.map((s) => `<code>${esc(s.term)}</code> (${esc(s.meaning)})`).join(" · ")}</div>` : ""}
            </div>
          `,
        })}
        ${deck("2. Five-task V6 corpus-neighbor projection", {
          subtitle: "Projected from k=12 nearest precomputed V6 neural predictions (0 GPU, <15 ms)",
          body: `
            <div class="grid-stack-sm">
              <div class="tag-row">
                ${badge(`Sentiment: ${pretty(preds.sentiment?.label || "neutral")}`, preds.sentiment?.label === "negative" ? "negative" : preds.sentiment?.label === "positive" ? "positive" : "neutral")}
                ${badge(`Response role: ${pretty(preds.response?.label || "opinion")}`, "info")}
                ${badge(`Solution: ${preds.solution?.is_solution ? "YES" : "NO"} (${percent(preds.solution?.probability, 0)})`, preds.solution?.is_solution ? "positive" : "neutral")}
              </div>
              <div class="text-xs">
                <strong>Predicted issues:</strong> ${(preds.issues?.display_labels || []).length ? preds.issues.display_labels.map((l) => badge(l, "neutral")).join(" ") : `<span class="muted">None above 30%</span>`}
              </div>
              <div class="text-xs">
                <strong>Predicted emotions:</strong> ${(preds.emotions?.labels || []).length ? preds.emotions.labels.map((l) => badge(pretty(l), "warning")).join(" ") : `<span class="muted">Neutral / low intensity</span>`}
              </div>
              <p class="muted text-xs">${esc(res.inference_notice || "")}</p>
            </div>
          `,
        })}
      </div>
      ${neighbors.length ? deck("Closest real V6-predicted corpus replies (interpretable evidence)", {
        subtitle: "Actual anonymized Reddit comments driving the nearest-neighbor projection above",
        body: `
          <div class="grid-stack-xs">
            ${neighbors.map((n) => `
              <div class="surface-well">
                <div class="surface-well-head">
                  <a class="surface-well-link" href="#/thread/${encodeURIComponent(n.post_id)}">${esc(n.post_title)}</a>
                  <span class="num muted text-xs">cos = ${Number(n.similarity).toFixed(3)} · ${esc(n.v6_sentiment)} · ${esc(n.v6_response)} · sol=${n.v6_is_solution ? "yes" : "no"}</span>
                </div>
                <p class="text-sm">“${esc(n.text)}”</p>
              </div>
            `).join("")}
          </div>
        `,
      }) : ""}
    `;
  }

  return `
    ${deck("Live multilingual comment analyzer", {
      subtitle: "Test Hinglish, Banglish, Odia, context-guarded Gen-Z slang, and emojis live — runs with zero GPU on Docker & Vercel",
      body: `
        <div class="tag-row tag-row-mb">${presets}</div>
        <form id="live-analyzer-form" class="grid-stack-sm">
          <label class="visually-hidden" for="live-analyzer-input">Comment text to analyze</label>
          <textarea id="live-analyzer-input" class="analyzer-textarea" rows="3" placeholder="Enter any English, Hinglish, Banglish, Odia, emoji, or Gen-Z student Reddit comment…">${esc(cmpState.analyzerText)}</textarea>
          <div class="tag-row">
            <button type="submit" class="btn btn-primary btn-sm" ${cmpState.analyzerBusy ? "disabled" : ""}>${cmpState.analyzerBusy ? "Analyzing…" : "Analyze comment"}</button>
            <span class="muted text-xs">Shows span-level language/script ID, context-aware slang guards, emoji affect decoding, and nearest V6 predictions</span>
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
      ["language", "Languages & NLP Benchmarks"],
      ["analyzer", "Live Multilingual Analyzer"],
    ];

    const bodyHtml = cmpState.tab === "iims"
      ? renderIimTab(data.cross_iim)
      : cmpState.tab === "language"
        ? renderLanguageTab(data.cross_language, data.task5_task6_ablations)
        : renderAnalyzerTab();

    container.innerHTML = `
      <div class="tag-row tag-row-md" role="tablist" aria-label="Comparative analytics and live lab tabs">
        ${tabs.map(([id, label]) => `
          <button type="button" role="tab" aria-selected="${cmpState.tab === id}" class="btn ${cmpState.tab === id ? "btn-primary" : "btn-quiet"} btn-sm" data-cmp-tab="${id}">${label}</button>
        `).join("")}
      </div>
      ${bodyHtml}
    `;

    container.querySelectorAll("[data-cmp-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        cmpState.tab = btn.dataset.cmpTab;
        ctx.syncQueryParams?.({ tab: cmpState.tab });
        paint();
        if (cmpState.tab === "analyzer" && !cmpState.analyzerResult && !cmpState.analyzerBusy) {
          runAnalyze(cmpState.analyzerText);
        }
      });
    });

    container.querySelector("#export-iim-csv")?.addEventListener("click", () => {
      const rows = data.cross_iim?.iims || [];
      const ok = downloadCsv(
        rows,
        [
          { key: "iim", label: "Mentioned IIM" },
          { key: "direct_comment_mentions", label: "Direct Replies" },
          { key: "thread_mentions", label: "Threads" },
          { key: "top_issue_label", label: "Top Issue" },
          { key: "dominant_emotion", label: "Dominant Emotion" },
          { key: "solution_rate", label: "Solution Rate" },
          { key: "negative", label: "Negative Share", get: (r) => r.sentiment?.negative ?? "" },
          { key: "positive", label: "Positive Share", get: (r) => r.sentiment?.positive ?? "" },
        ],
        "rci_cross_iim_comparison.csv"
      );
      if (ok) ctx.showToast?.(`Exported ${rows.length} IIM comparison rows to CSV`);
    });

    container.querySelectorAll("[data-preset-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.presetIdx);
        const preset = LIVE_PRESETS[idx];
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
