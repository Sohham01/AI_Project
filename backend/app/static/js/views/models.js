/**
 * View 9: Models, Architecture Proof, Error Analysis & Dataset Provenance Vault
 * Shows WHY we chose Transformer (LogReg vs ANN vs CNN vs MuRIL-DAPT v3 across
 * 5 tasks & 8 unseen stress categories), HOW DAPT MLM & temperature calibration
 * work, interactive Error Analysis (confusion matrices, recovery & failure cases),
 * automated Benchmark Verification, and 100% honest provenance/metric reporting.
 */
import { api } from "../api.js";
import {
  architectureComparisonChart,
  calibrationReliabilityChart,
  confidenceHistogramChart,
  confusionMatrixTable,
  daptConvergenceChart,
  modelComparisonSvgChart,
  robustnessCategoryHeatmap,
} from "../charts.js";
import { badge, callout, confidenceBadge, copyIdChip, deck, glossaryTip } from "../components.js";
import { dateTimeText, esc, num, percent, pretty } from "../format.js";

const TASK_ORDER = ["issue", "solution", "sentiment", "response", "emotion"];

const TASK_DESCRIPTIONS = {
  sentiment: "3-class mutually exclusive polarity classifier (negative, neutral, positive) powered by MuRIL-DAPT v3 + Word+Char TF-IDF hybrid.",
  emotion: "6-label multi-label affect classifier (confidence, curiosity, frustration, gratitude, hope, stress) powered by MuRIL-DAPT v3.",
  issue: "8-label multi-label issue detector using MuRIL-DAPT v3 + parent post title context + Word+Char TF-IDF hybrid.",
  response: "6-class multiclass classifier preserving the exact canonical community response taxonomy (MuRIL-DAPT v3 + Word+Char TF-IDF hybrid).",
  solution: "Binary actionable-solution detector (not_solution vs. solution) powered by MuRIL-DAPT v3 + Word+Char TF-IDF hybrid.",
};

function fmtParams(n) {
  const v = Number(n || 0);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return num(v);
}

function fmtPred(val) {
  if (Array.isArray(val)) return val.length ? val.map(pretty).join(", ") : "None";
  return pretty(String(val ?? "None"));
}

function renderPerClassSection(taskName, task) {
  const metrics = task.test?.metrics || {};
  const isMulti = Boolean(metrics.multilabel);
  const mapObj = Object.keys(metrics.per_class || {}).length ? metrics.per_class : (metrics.per_label || {});
  const entries = Object.entries(mapObj);

  if (!entries.length) return `<p class="muted">Per-class metrics were not emitted in the evaluation artifact.</p>`;

  const totalSupport = entries.reduce((sum, [, v]) => sum + Number(v?.support ?? 0), 0);
  const nSamples = Number(metrics.n_samples || 0);

  if (totalSupport === 0 && nSamples > 0) {
    return callout(
      `<strong>Per-class breakdown and confusion matrix unavailable in this frozen artifact (label-key mismatch):</strong> Held-out aggregate metrics remain valid (<code>n = ${num(nSamples)}</code>, accuracy <code>${percent(metrics.accuracy)}</code>, macro F1 <code>${percent(metrics.macro_f1)}</code>).`,
      { tone: "warning", title: "Artifact diagnostic:" }
    );
  }

  const rows = entries.map(([label, vals]) => {
    const f1 = vals?.f1 ?? vals?.f1_score;
    const f1Pct = f1 !== null && f1 !== undefined && Number.isFinite(Number(f1)) ? Number(f1) * 100 : 0;
    return `
      <tr>
        <td><code>${esc(label)}</code></td>
        <td class="num">${percent(vals?.precision, 1, "N/A")}</td>
        <td class="num">${percent(vals?.recall, 1, "N/A")}</td>
        <td class="num">
          <div style="display:inline-flex;align-items:center;gap:var(--space-xs);min-width:104px;justify-content:flex-end">
            <div class="kw-track" style="width:48px"><span class="kw-fill" style="width:${f1Pct.toFixed(1)}%"></span></div>
            <span class="num">${percent(f1, 1, "N/A")}</span>
          </div>
        </td>
        <td class="num">${num(vals?.support)}</td>
      </tr>
    `;
  }).join("");

  return `
    <div class="table-scroll">
      <table class="data-table" aria-label="${esc(pretty(taskName))} per-${isMulti ? "label" : "class"} test metrics">
        <thead>
          <tr>
            <th scope="col">${isMulti ? "Label" : "Class"}</th>
            <th scope="col" class="num">Precision</th>
            <th scope="col" class="num">Recall</th>
            <th scope="col" class="num">F1 score</th>
            <th scope="col" class="num">Test support</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderVerificationBanner(verification) {
  if (!verification || !Array.isArray(verification.checks)) return "";
  const passed = Number(verification.passed_checks || 0);
  const total = Number(verification.total_checks || 0);
  const allOk = Boolean(verification.verified);
  const rows = verification.checks.map((c) => `
    <div class="verify-check-row">
      <span class="tag ${c.passed ? "tag-positive" : "tag-danger"}">${c.passed ? "PASS" : "FAIL"}</span>
      <div>
        <strong>${esc(c.label)}</strong>
        <p class="muted" style="font-size:var(--text-xs);margin-top:2px">${esc(c.detail)}</p>
      </div>
    </div>
  `).join("");

  return `
    <details class="verify-banner ${allOk ? "verify-ok" : "verify-warn"} anim-enter">
      <summary class="verify-summary">
        <div class="verify-summary-left">
          <span class="tag ${allOk ? "tag-positive" : "tag-warning"}">${allOk ? "BENCHMARKS VERIFIED" : "AUDIT ATTENTION"}</span>
          <strong>${passed} / ${total} independent benchmark, anti-leakage, calibration &amp; zero-PII checks passed</strong>
        </div>
        <span class="num muted" style="font-size:var(--text-xs)">Verified ${dateTimeText(verification.verified_at)} · Inspect audit</span>
      </summary>
      <div class="verify-checks-grid">${rows}</div>
    </details>
  `;
}

function renderArchitectureSection(archComp) {
  if (!archComp || !archComp.aggregate) return "";
  const order = archComp.architectures_order || Object.keys(archComp.aggregate);
  const agg = archComp.aggregate || {};
  const why = archComp.why_transformer || {};

  const summaryRows = order.map((key) => {
    const row = agg[key];
    if (!row) return "";
    const isSelected = key === "transformer_muril_dapt_v3" || key === "production_hybrid";
    const gapPp = (Number(row.generalization_gap || 0) * 100).toFixed(1);
    return `
      <tr class="${isSelected ? "arch-row-highlight" : ""}">
        <td>
          <strong>${esc(row.label)}</strong>
          ${key === "transformer_muril_dapt_v3" ? badge("Core Encoder", "info") : key === "production_hybrid" ? badge("Deployed", "positive") : ""}
        </td>
        <td class="num">${percent(row.mean_test_macro_f1)}</td>
        <td class="num"><strong>${percent(row.mean_probe_macro_f1)}</strong></td>
        <td class="num"><span class="tag ${Number(row.generalization_gap) <= 0.28 ? "tag-positive" : "tag-warning"}">-${gapPp} pp</span></td>
        <td class="num">${Number(row.mean_calibration_ece ?? 0).toFixed(3)}</td>
        <td class="num">${fmtParams(row.param_count)}</td>
        <td class="num">${Number(row.total_train_seconds ?? 0).toFixed(1)}s</td>
        <td class="num">${Number(row.mean_inference_ms ?? 0).toFixed(2)} ms</td>
      </tr>
    `;
  }).join("");

  const reasonCards = (why.key_reasons || []).map((r, i) => `
    <div class="why-card">
      <span class="why-num num">0${i + 1}</span>
      <div>
        <h4>${esc(r.title)}</h4>
        <p>${esc(r.evidence)}</p>
      </div>
    </div>
  `).join("");

  return deck("Why we chose Transformer: 4-architecture empirical proof", {
    id: "why-transformer-deck",
    subtitle: "Apples-to-apples comparison of Logistic Regression, ANN (MLP), 1D TextCNN, and MuRIL-DAPT v3 Transformer on identical thread-grouped splits and the untouched 54-probe unseen 8-category stress benchmark",
    action: `
      <div class="segmented" role="group" aria-label="Architecture chart split toggle">
        <button type="button" class="seg-btn is-active" data-arch-mode="probe">54-Probe Unseen OOD</button>
        <button type="button" class="seg-btn" data-arch-mode="test">Held-out Test Split</button>
      </div>
    `,
    body: `
      ${why.gain_vs_logreg_pp !== undefined ? `
        <div class="arch-callout-strip">
          <div class="arch-delta-pill"><span>MuRIL-DAPT v3 vs. LogReg</span><strong class="num">+${Number(why.gain_vs_logreg_pp).toFixed(1)} pp OOD</strong></div>
          <div class="arch-delta-pill"><span>MuRIL-DAPT v3 vs. ANN (MLP)</span><strong class="num">+${Number(why.gain_vs_ann_pp).toFixed(1)} pp OOD</strong></div>
          <div class="arch-delta-pill"><span>MuRIL-DAPT v3 vs. 1D CNN</span><strong class="num">+${Number(why.gain_vs_cnn_pp).toFixed(1)} pp OOD</strong></div>
          <div class="arch-delta-pill arch-delta-accent"><span>Production Hybrid Mean OOD</span><strong class="num">${percent(why.mean_ood_f1_hybrid)}</strong></div>
        </div>
      ` : ""}
      <div id="arch-chart-slot">${architectureComparisonChart(archComp, "probe")}</div>
      <div class="spacer-md"></div>
      <h3 class="subpanel-title">Architecture tradeoff ledger (5-task aggregate: accuracy, generalization gap, calibration &amp; compute)</h3>
      <div class="table-scroll">
        <table class="data-table" aria-label="4-architecture aggregate comparison">
          <thead>
            <tr>
              <th scope="col">Architecture</th>
              <th scope="col" class="num">Held-out F1</th>
              <th scope="col" class="num">54-Probe OOD F1</th>
              <th scope="col" class="num">OOD Drop</th>
              <th scope="col" class="num">Mean ECE</th>
              <th scope="col" class="num">Params</th>
              <th scope="col" class="num">Train (5 tasks)</th>
              <th scope="col" class="num">Latency / text</th>
            </tr>
          </thead>
          <tbody>${summaryRows}</tbody>
        </table>
      </div>
      <div class="spacer-md"></div>
      <h3 class="subpanel-title">Where Transformer wins: 8 unseen robustness stress categories (mean 5-task F1)</h3>
      ${robustnessCategoryHeatmap(archComp)}
      ${reasonCards ? `<div class="spacer-md"></div><div class="why-grid">${reasonCards}</div>` : ""}
    `,
  });
}

function renderHowItWorksSection(dapt, errorAnalysis, confSummary) {
  if (!dapt && !errorAnalysis && !confSummary) return "";
  const calTask = errorAnalysis?.response?.calibration || errorAnalysis?.sentiment?.calibration || null;

  return deck("How MuRIL-DAPT v3 adapts &amp; calibrates prediction confidence", {
    subtitle: "Mechanism visuals explaining unsupervised Masked Language Modeling (DAPT) convergence, temperature-scaled reliability calibration, and live corpus confidence routing",
    body: `
      <div class="studio-grid-2">
        <div class="how-panel">
          <h3 class="subpanel-title">1. Unsupervised domain adaptation (DAPT / MLM)</h3>
          <p class="how-explain">Before any supervised fine-tuning, MuRIL masks 15% of tokens across <strong>${num(dapt?.unlabeled_documents || 0)}</strong> unlabeled anonymized Reddit texts to learn CAT/CAP/SAP/WL/LPA slang and Hinglish phrasing without fabricating labels.</p>
          ${daptConvergenceChart(dapt)}
        </div>
        <div class="how-panel">
          <h3 class="subpanel-title">2. Temperature scaling &amp; reliability calibration</h3>
          <p class="how-explain">Validation-only scalar temperature scaling (<code>T = ${esc(calTask?.temperature ?? "—")}</code>) aligns predicted probabilities with empirical accuracy so high-confidence scores can be trusted and uncertain replies route to human review.</p>
          ${calTask ? calibrationReliabilityChart(calTask) : `<div class="chart-empty">Calibration bins unavailable.</div>`}
        </div>
      </div>
      ${confSummary?.histogram?.length ? `
        <div class="spacer-md"></div>
        <div class="how-panel">
          <div class="how-panel-head">
            <div>
              <h3 class="subpanel-title">3. Live corpus prediction confidence distribution</h3>
              <p class="how-explain">Mean calibrated 5-task confidence across all live comments is <strong>${percent(confSummary.mean_confidence)}</strong> (median ${percent(confSummary.median_confidence)}). Comments below 55% or with low task margin (${num(confSummary.needs_review_count)} total) are flagged for Annotation Studio review.</p>
            </div>
            <a class="btn btn-quiet btn-sm" href="#/explorer?confidence=review">Inspect review queue</a>
          </div>
          ${confidenceHistogramChart(confSummary)}
        </div>
      ` : ""}
    `,
  });
}

function renderErrorAnalysisBody(taskKey, errTask) {
  if (!errTask) return `<p class="muted">No error analysis available for this task.</p>`;
  const probeCm = errTask.probe_confusion || {};
  const cal = errTask.calibration || {};
  const arch = errTask.architecture_analysis || {};
  const agree = arch.agreement_summary || {};
  const tiers = cal.confidence_tiers || {};

  const tierRows = ["high", "medium", "low"].map((t) => {
    const info = tiers[t] || {};
    return `
      <tr>
        <td>${badge(t === "high" ? "High (>=80%)" : t === "medium" ? "Medium (55-79%)" : "Low (<55% · Review)", t === "high" ? "positive" : t === "medium" ? "info" : "warning")}</td>
        <td class="num">${num(info.count || 0)}</td>
        <td class="num">${percent(info.share)}</td>
        <td class="num"><strong>${info.accuracy !== null && info.accuracy !== undefined ? percent(info.accuracy) : "—"}</strong></td>
        <td class="num">${info.error_rate !== null && info.error_rate !== undefined ? percent(info.error_rate) : "—"}</td>
      </tr>
    `;
  }).join("");

  const topPairsHtml = (probeCm.top_confusions || []).length
    ? (probeCm.top_confusions || []).map((p) => `
      <div class="confusion-pair-row">
        <div class="confusion-pair-head">
          <span><code>${esc(p.true_label)}</code> &rarr; <code>${esc(p.predicted_label)}</code></span>
          <span class="tag tag-warning num">${num(p.count)} errors (${percent(p.share_of_errors, 0)})</span>
        </div>
        <p class="muted" style="font-size:var(--text-xs);margin-top:4px">${esc(p.diagnosis)}</p>
      </div>
    `).join("")
    : `<p class="muted">Zero off-diagonal confusions recorded on the 54 unseen probes for this task.</p>`;

  const winsHtml = (arch.transformer_wins || []).length
    ? (arch.transformer_wins || []).slice(0, 4).map((w) => `
      <article class="err-case-card err-case-win">
        <div class="err-case-top">
          ${badge(pretty(w.category), "info")}
          ${confidenceBadge(w.transformer_confidence)}
          <span class="tag tag-positive">Beat ${num(w.baselines_failed)}/3 baselines</span>
        </div>
        <blockquote class="err-case-quote">“${esc(w.text)}”</blockquote>
        <div class="err-case-compare">
          <span><strong>Gold &amp; MuRIL:</strong> <code>${esc(fmtPred(w.true_label))}</code></span>
          <span class="muted">LogReg: <code>${esc(fmtPred(w.baseline_predictions?.logistic_regression))}</code> · ANN: <code>${esc(fmtPred(w.baseline_predictions?.ann_mlp))}</code> · CNN: <code>${esc(fmtPred(w.baseline_predictions?.cnn_textcnn))}</code></span>
        </div>
        <p class="err-case-reason">${esc(w.why_transformer_won)}</p>
      </article>
    `).join("")
    : `<p class="muted">All four architectures agreed on the correct probes for this task.</p>`;

  const residualHtml = (arch.residual_errors || []).length
    ? (arch.residual_errors || []).slice(0, 4).map((e) => `
      <article class="err-case-card err-case-miss">
        <div class="err-case-top">
          ${badge(e.failure_mode || pretty(e.category), "warning")}
          ${confidenceBadge(e.transformer_confidence, e.confidence_tier)}
        </div>
        <blockquote class="err-case-quote">“${esc(e.text)}”</blockquote>
        <div class="err-case-compare">
          <span><strong>True:</strong> <code>${esc(fmtPred(e.true_label))}</code></span>
          <span><strong>Predicted:</strong> <code>${esc(fmtPred(e.transformer_prediction))}</code></span>
        </div>
        <p class="err-case-reason">${esc(e.root_cause)}</p>
      </article>
    `).join("")
    : `<p class="muted">Zero residual Transformer errors on the 54 unseen probes for this task.</p>`;

  return `
    <div class="err-kpi-row">
      <div class="eval-cell"><span>54-Probe Error Rate</span><strong class="num">${percent(probeCm.error_rate)}</strong><small class="num">${num(probeCm.error_count)} / ${num(probeCm.n_samples)} probes</small></div>
      <div class="eval-cell"><span>Transformer Beat Majority Baselines</span><strong class="num">${num(agree.transformer_beats_majority_baselines || 0)}</strong><small>${num(agree.transformer_only_win || 0)} where ALL 3 baselines failed</small></div>
      <div class="eval-cell"><span>Calibrated ECE / Brier</span><strong class="num">${Number(cal.calibrated_ece ?? 0).toFixed(3)}</strong><small class="num">Raw T=1.0: ${Number(cal.uncalibrated_ece ?? 0).toFixed(3)} · Brier ${Number(cal.brier_score ?? 0).toFixed(3)}</small></div>
      <div class="eval-cell"><span>All 4 Architectures Failed</span><strong class="num">${num(agree.all_failed || 0)}</strong><small>Inherently ambiguous / review probes</small></div>
    </div>

    <div class="studio-grid-2" style="margin-top:var(--space-md)">
      <div>
        <h3 class="subpanel-title">Top confused class pairs (54 unseen probes)</h3>
        <div class="confusion-pair-list">${topPairsHtml}</div>
        ${!probeCm.multilabel && probeCm.matrix ? `<div class="spacer-sm"></div>${confusionMatrixTable(probeCm.matrix, probeCm.labels)}` : ""}
      </div>
      <div>
        <h3 class="subpanel-title">Error rate by prediction confidence tier</h3>
        <div class="table-scroll">
          <table class="data-table" aria-label="${esc(pretty(taskKey))} accuracy by confidence tier">
            <thead>
              <tr>
                <th scope="col">Confidence tier</th>
                <th scope="col" class="num">Probes</th>
                <th scope="col" class="num">Share</th>
                <th scope="col" class="num">Accuracy</th>
                <th scope="col" class="num">Error rate</th>
              </tr>
            </thead>
            <tbody>${tierRows}</tbody>
          </table>
        </div>
        <div class="spacer-sm"></div>
        ${calibrationReliabilityChart(cal)}
      </div>
    </div>

    <div class="studio-grid-2" style="margin-top:var(--space-lg)">
      <div>
        <h3 class="subpanel-title">Where Transformer beat LogReg / ANN / CNN (real probe recoveries)</h3>
        <div class="err-case-stack">${winsHtml}</div>
      </div>
      <div>
        <h3 class="subpanel-title">Hardest residual errors &amp; root-cause taxonomy</h3>
        <div class="err-case-stack">${residualHtml}</div>
      </div>
    </div>
  `;
}

export async function renderModels(container, ctx) {
  const [modelsData, summary, health] = await Promise.all([
    api.models(),
    ctx.summary ? Promise.resolve(ctx.summary) : api.summary(),
    api.health(),
  ]);
  ctx.summary = summary;

  const tasksObj = modelsData.tasks || {};
  const orderedNames = TASK_ORDER.filter((name) => name in tasksObj).concat(
    Object.keys(tasksObj).filter((name) => !TASK_ORDER.includes(name))
  );

  const prov = summary.provenance || {};
  const isSynthetic = Boolean(summary.synthetic_only);
  const filt = prov.filtering_summary || {};
  const dapt = modelsData.dapt_summary || prov.dapt_summary || null;
  const archComp = modelsData.architecture_comparison || null;
  const errorAnalysis = modelsData.error_analysis || null;
  const verification = modelsData.benchmark_verification || null;
  const confSummary = modelsData.confidence_summary || summary.confidence_summary || null;

  const subListText = (summary.community_distribution || [])
    .map((c) => `<code>r/${esc(c.label)}</code> (${num(c.count)})`)
    .join(", ") || "public Reddit communities";
  const firstArtifactPath = tasksObj.sentiment?.artifact_path || tasksObj.solution?.artifact_path || "";
  const activeArtifactDir = firstArtifactPath.includes("/")
    ? `${firstArtifactPath.slice(0, firstArtifactPath.lastIndexOf("/") + 1)}`
    : "ml/artifacts/live_corpus/";
  const avgReplies = summary.posts ? (Number(summary.comments || 0) / Number(summary.posts)).toFixed(1) : "0";

  const provenanceBody = `
    <div class="provenance-two-col">
      <div>
        <h3 class="subpanel-title">Corpus metadata &amp; privacy</h3>
        <dl class="vault-kv-grid">
          <div class="kv-span-full"><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "iim-sambalpur-arctic-shift-v4")}</dd></div>
          <div><dt>Dataset version</dt><dd class="num">v${esc(prov.dataset_version || health.dataset_version)}</dd></div>
          <div><dt>Source archive</dt><dd>${esc(prov.source_provider === "arctic_shift" ? "Arctic Shift (public Reddit)" : (prov.generation_seed ? `Seed ${prov.generation_seed}` : "Local archive"))}</dd></div>
          <div><dt>Updated at</dt><dd class="num">${dateTimeText(prov.generation_timestamp || summary.generated_at)}</dd></div>
          <div><dt>Corpus size</dt><dd class="num">${num(summary.posts)} threads, ${num(summary.comments)} comments (${avgReplies} / thread)</dd></div>
          <div><dt>Privacy &amp; access</dt><dd><span class="tag ${isSynthetic ? "tag-warning" : "tag-positive"}">${esc(isSynthetic ? (prov.authorization_status || "synthetic_only") : "Anonymized public Reddit")}</span></dd></div>
          <div><dt>Artifact directory</dt><dd class="num">${esc(activeArtifactDir)}</dd></div>
          ${dapt ? `<div class="kv-span-full"><dt>Domain-Adaptive MLM (DAPT)</dt><dd class="num">${num(dapt.unlabeled_documents)} real Reddit texts, ${num(dapt.epochs)} epochs (perplexity ${esc(dapt.initial_perplexity)} → ${esc(dapt.final_perplexity)})</dd></div>` : ""}
          ${filt.automod_boilerplate_filtered !== undefined ? `<div class="kv-span-full"><dt>Noise filtering</dt><dd class="num">${num(filt.automod_boilerplate_filtered)} AutoModerator sticky comments and ${num(filt.gif_only_filtered)} GIF-only embeds removed; usernames stripped and inline PII redacted</dd></div>` : ""}
        </dl>
      </div>

      <div>
        <h3 class="subpanel-title">Methodology &amp; epistemic boundaries</h3>
        <ul class="limitations-list">
          <li><strong>Anonymized real Reddit discussions:</strong> Public IIM Sambalpur discussions across ${subListText} collected via Arctic Shift with author usernames removed and inline contact/user mentions redacted.</li>
          <li><strong>4-Architecture Verified Selection:</strong> Benchmarked Logistic Regression, 2-layer ANN (MLP), 1D Multi-Kernel TextCNN, and MuRIL-DAPT v3 Transformer on identical <code>post_id</code>-grouped splits and 54 unseen stress probes before selecting the calibrated MuRIL-DAPT v3 hybrid.</li>
          <li><strong>Calibrated Confidence &amp; Human Review:</strong> Validation-only temperature scaling yields reliable probabilities; low-confidence predictions (&lt;55%) route directly to human adjudication in the local Annotation Studio.</li>
          <li><strong>Proposal vs. resolution separation:</strong> Predicted <code>advice</code> and <code>solution</code> replies are strictly separated from explicit first-person follow-up resolution signals.</li>
          <li><strong>Honest metric reporting:</strong> Multi-label tasks (<code>emotion</code>, <code>issue</code>) report <code>macro_f1</code>, <code>micro_f1</code>, <code>subset_accuracy</code>, and <code>hamming_loss</code>; any unemitted metric is displayed as <em>Not emitted</em> rather than 0%.</li>
        </ul>
      </div>
    </div>
  `;

  const chartTasks = orderedNames.map((name) => {
    const t = tasksObj[name];
    return {
      key: name,
      label: pretty(name),
      valF1: t.validation?.metrics?.macro_f1,
      testF1: t.test?.metrics?.macro_f1,
      multilabel: Boolean(t.test?.metrics?.multilabel),
      binary: (t.labels || []).length === 2 && !t.test?.metrics?.multilabel,
    };
  });

  const comparisonRows = orderedNames.map((name) => {
    const t = tasksObj[name];
    const tm = t.test?.metrics || {};
    const vm = t.validation?.metrics || {};
    const isMulti = Boolean(tm.multilabel);
    const probeF1 = t.robustness_54_probes?.overall_macro_f1;
    const accuracyCell = isMulti
      ? (tm.subset_accuracy !== null && tm.subset_accuracy !== undefined
          ? percent(tm.subset_accuracy)
          : `<span class="not-emitted-pill" title="subset_accuracy is null in evaluation.json">Not emitted</span>`)
      : percent(tm.accuracy);

    return `
      <tr>
        <td><a class="table-primary-link" href="#/models" data-scroll-task="${esc(name)}">${esc(pretty(name))}</a><span class="table-subcode">${esc(t.model_name)} ${esc(t.model_version)}</span></td>
        <td>${badge(isMulti ? "Multi-label" : (t.labels || []).length === 2 ? "Binary" : "Single-label", isMulti ? "info" : "neutral")}</td>
        <td class="num">${num(t.split?.train)} / ${num(t.split?.validation)} / ${num(t.split?.test)}</td>
        <td>${badge(t.leakage_report?.leakage_detected ? "Leakage detected" : "Zero leakage", t.leakage_report?.leakage_detected ? "danger" : "positive")}</td>
        <td class="num">${percent(vm.macro_f1)}</td>
        <td class="num"><strong>${percent(tm.macro_f1)}</strong></td>
        <td class="num"><strong>${probeF1 !== undefined && probeF1 !== null ? percent(probeF1) : "—"}</strong></td>
        <td class="num">${accuracyCell}</td>
      </tr>
    `;
  }).join("");

  const taskCards = orderedNames.map((name) => {
    const task = tasksObj[name];
    const tm = task.test?.metrics || {};
    const vm = task.validation?.metrics || {};
    const isMulti = Boolean(tm.multilabel);
    const split = task.split || {};
    const leakage = Boolean(task.leakage_report?.leakage_detected);
    const allWarnings = [...new Set([...(task.test?.warnings || []), ...(task.warnings || [])])];

    const hasSubsetAcc = tm.subset_accuracy !== null && tm.subset_accuracy !== undefined;
    const primaryMetricTiles = isMulti
      ? `
        <div class="eval-cell"><span>Test macro F1</span><strong class="num">${percent(tm.macro_f1)}</strong><small class="num">Val: ${percent(vm.macro_f1)}</small></div>
        <div class="eval-cell"><span>Test micro F1</span><strong class="num">${percent(tm.micro_f1)}</strong><small class="num">Val: ${percent(vm.micro_f1)}</small></div>
        <div class="eval-cell"><span>${glossaryTip("subset_accuracy", "Subset accuracy")}</span><strong class="${hasSubsetAcc ? "num" : "metric-muted"}">${percent(tm.subset_accuracy, 1, "Not emitted")}</strong><small class="num">${hasSubsetAcc ? `Hamming loss: ${percent(tm.hamming_loss, 2, "Not emitted")} · Val: ${percent(vm.subset_accuracy)}` : "Field is <code>null</code> in JSON"}</small></div>
        <div class="eval-cell"><span>Thread split (${esc(split.group_by || "post_id")})</span><strong class="num">${num(split.train)} / ${num(split.validation)} / ${num(split.test)}</strong><small>${leakage ? "Leakage detected" : "Zero cross-split leakage"}</small></div>
      `
      : `
        <div class="eval-cell"><span>Test accuracy</span><strong class="num">${percent(tm.accuracy)}</strong><small class="num">Val: ${percent(vm.accuracy)}</small></div>
        <div class="eval-cell"><span>Test macro F1</span><strong class="num">${percent(tm.macro_f1)}</strong><small class="num">Val: ${percent(vm.macro_f1)}</small></div>
        <div class="eval-cell"><span>${tm.micro_f1 !== null && tm.micro_f1 !== undefined ? "Test micro F1" : "Test weighted F1"}</span><strong class="num">${percent(tm.micro_f1 ?? tm.weighted_f1, 1, "N/A")}</strong><small class="num">n = ${num(tm.n_samples)} test records</small></div>
        <div class="eval-cell"><span>Thread split (${esc(split.group_by || "post_id")})</span><strong class="num">${num(split.train)} / ${num(split.validation)} / ${num(split.test)}</strong><small>${leakage ? "Leakage detected" : "Zero cross-split leakage"}</small></div>
      `;

    const multilabelNote = isMulti && (tm.subset_accuracy === null || tm.subset_accuracy === undefined)
      ? callout(
          `In <code>evaluation.json</code>, <code>subset_accuracy</code>, <code>hamming_loss</code>, and <code>hamming_accuracy</code> are <code>null</code> because <code>ml/evaluation/reports.py::_extract_metrics()</code> extracts only <code>macro_f1</code>, <code>micro_f1</code>, and <code>per_label</code> for multi-label tasks.`,
          { tone: "info", title: "Metric definition note:" }
        )
      : "";

    const warningsHtml = allWarnings.length
      ? `<div class="eval-warnings"><span class="eval-warnings-label">Evaluator &amp; split warnings (${allWarnings.length}):</span><ul>${allWarnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`
      : "";

    const labelDistHtml = Object.entries(task.label_counts || {})
      .map(([k, v]) => `<span class="tag tag-neutral">${esc(k)}: <span class="num">${num(v)}</span></span>`)
      .join("");

    const probeF1 = task.robustness_54_probes?.overall_macro_f1;
    const probeBadge = probeF1 !== undefined && probeF1 !== null
      ? badge(`54-probe unseen F1: ${percent(probeF1)}`, probeF1 >= 0.7 ? "positive" : "info")
      : "";

    return deck(`${esc(pretty(name))} classifier`, {
      id: `model-task-${esc(name)}`,
      className: "model-detail-card",
      subtitle: TASK_DESCRIPTIONS[name] || "",
      action: `
        ${badge(`${task.model_name} ${task.model_version}`, "info")}
        ${probeBadge}
        ${task.registry_id ? copyIdChip(task.registry_id, { label: "registry" }) : ""}
      `,
      body: `
        <div class="eval-row-grid">${primaryMetricTiles}</div>
        ${multilabelNote}
        ${warningsHtml}
        <div class="model-detail-split">
          <div>
            <h3 class="subpanel-title">Held-out test per-${isMulti ? "label" : "class"} breakdown (n = ${num(tm.n_samples)})</h3>
            ${renderPerClassSection(name, task)}
            ${confusionMatrixTable(tm.confusion_matrix, tm.confusion_matrix_labels)}
          </div>
          <div>
            <h3 class="subpanel-title">Live corpus predicted label distribution (n = ${num(summary.comments)})</h3>
            <div class="label-chip-wrap">${labelDistHtml}</div>
            <div class="spacer-md"></div>
            <h3 class="subpanel-title">Artifact &amp; scope</h3>
            <dl class="spec-kv">
              <div><dt>Artifact</dt><dd><code class="break-code">${esc(task.artifact_path || "None")}</code></dd></div>
              <div><dt>Split seed</dt><dd class="num">${esc(split.seed)}</dd></div>
              <div><dt>Stratified</dt><dd>${split.stratify_used ? "Yes" : "No (group-aware)"}</dd></div>
              <div><dt>Scope</dt><dd>${esc(task.inference_scope || "Batch inference over anonymized Reddit corpus")}</dd></div>
            </dl>
          </div>
        </div>
      `,
    });
  }).join("");

  let activeErrTask = errorAnalysis ? (TASK_ORDER.find((t) => t in errorAnalysis) || Object.keys(errorAnalysis)[0]) : null;
  const errorAnalysisDeck = errorAnalysis && activeErrTask
    ? deck("Interactive error analysis &amp; model failure diagnostics", {
        id: "error-analysis-deck",
        subtitle: "Inspect per-task confusion pairs, calibration reliability by confidence tier, real probe recoveries where Transformer beat LogReg/ANN/CNN, and residual errors",
        action: `
          <div class="segmented" role="tablist" aria-label="Select task for error analysis">
            ${TASK_ORDER.filter((t) => t in errorAnalysis).map((t) => `
              <button type="button" role="tab" class="seg-btn ${t === activeErrTask ? "is-active" : ""}" data-err-task="${esc(t)}" aria-selected="${t === activeErrTask ? "true" : "false"}">${esc(pretty(t))}</button>
            `).join("")}
          </div>
        `,
        body: `<div id="error-analysis-slot">${renderErrorAnalysisBody(activeErrTask, errorAnalysis[activeErrTask])}</div>`,
      })
    : "";

  container.innerHTML = `
    ${renderVerificationBanner(verification)}
    ${callout(`${esc(modelsData.evaluation_scope)}`, { tone: "info", title: "Evaluation & inference scope:" })}
    ${renderArchitectureSection(archComp)}
    ${renderHowItWorksSection(dapt, errorAnalysis, confSummary)}
    ${errorAnalysisDeck}
    ${deck("Dataset provenance, governance &amp; methodology", {
      subtitle: "Verified metadata read directly from /api/demo/summary and /api/demo/health",
      body: provenanceBody,
    })}
    ${deck("Production 5-task benchmark summary", {
      subtitle: "Side-by-side comparison of held-out validation, held-out test, and unseen 54-probe Macro F1 across all five active classifiers",
      body: `
        ${modelComparisonSvgChart(chartTasks)}
        <div class="spacer-md"></div>
        <div class="table-scroll">
          <table class="data-table" aria-label="Five active NLP models benchmark comparison">
            <thead>
              <tr>
                <th scope="col">Task &amp; model</th>
                <th scope="col">Paradigm</th>
                <th scope="col" class="num">Train / Val / Test</th>
                <th scope="col">${glossaryTip("group_split", "Leakage")}</th>
                <th scope="col" class="num">Val F1</th>
                <th scope="col" class="num">${glossaryTip("macro_f1", "Test F1")}</th>
                <th scope="col" class="num">54-Probe OOD</th>
                <th scope="col" class="num">${glossaryTip("subset_accuracy", "Accuracy / subset")}</th>
              </tr>
            </thead>
            <tbody>${comparisonRows}</tbody>
          </table>
        </div>
      `,
    })}
    <div class="model-cards-stack">${taskCards}</div>
  `;

  // Wire Architecture split toggle (54-Probe OOD vs Held-out Test)
  container.querySelectorAll("[data-arch-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.archMode;
      container.querySelectorAll("[data-arch-mode]").forEach((b) => b.classList.toggle("is-active", b === btn));
      const slot = container.querySelector("#arch-chart-slot");
      if (slot && archComp) slot.innerHTML = architectureComparisonChart(archComp, mode);
    });
  });

  // Wire Error Analysis task tabs
  container.querySelectorAll("[data-err-task]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeErrTask = btn.dataset.errTask;
      container.querySelectorAll("[data-err-task]").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      const slot = container.querySelector("#error-analysis-slot");
      if (slot && errorAnalysis) slot.innerHTML = renderErrorAnalysisBody(activeErrTask, errorAnalysis[activeErrTask]);
    });
  });

  // Wire smooth scroll to per-task cards
  container.querySelectorAll("[data-scroll-task]").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      container.querySelector(`#model-task-${CSS.escape(link.dataset.scrollTask)}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}
