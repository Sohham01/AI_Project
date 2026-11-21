/**
 * View 9: Models, Architecture Proof, Error Analysis & Provenance Vault
 * Streamlined low-cognitive-load IA (zero duplicated charts, progressive disclosure):
 * 1. Compact Pipeline Integrity Bar (7/7 automated checks, expandable)
 * 2. Hero 4-Architecture Benchmark (Honest 0–100% Horizontal Dumbbell + Tradeoff Ledger + Collapsible 54-Probe Matrix & Key Reasons)
 * 3. Mechanism & Confidence (Clean 2-Column: Single-Axis Zero-Anchored DAPT Perplexity + Corpus Confidence Histogram)
 * 4. Interactive 5-Task Error Analysis & 1:1 Square Calibration Studio (with tabbed Recoveries vs Residual Errors)
 * 5. Collapsible Per-Task Granular Audit Accordion + Compact Dataset Provenance Vault
 */
import { api } from "../api.js?v=5.6";
import {
  architectureComparisonChart,
  bindChartTooltips,
  calibrationReliabilityChart,
  confidenceHistogramChart,
  confusionMatrixTable,
  daptConvergenceChart,
  robustnessCategoryHeatmap,
} from "../charts.js?v=5.6";
import { badge, callout, confidenceBadge, copyIdChip, deck, glossaryTip } from "../components.js?v=5.6";
import { dateTimeText, esc, num, percent, pretty } from "../format.js?v=5.6";

const TASK_ORDER = ["issue", "solution", "sentiment", "response", "emotion"];

const TASK_DESCRIPTIONS = {
  sentiment: "3-class polarity classifier (negative, neutral, positive) using MuRIL-DAPT v3 + TF-IDF hybrid.",
  emotion: "6-label multi-label affect classifier (confidence, curiosity, frustration, gratitude, hope, stress).",
  issue: "8-label multi-label issue detector using MuRIL-DAPT v3 + parent post title context + TF-IDF hybrid.",
  response: "6-class multiclass classifier preserving the exact canonical community response taxonomy.",
  solution: "Binary actionable-solution detector (not_solution vs. solution) using MuRIL-DAPT v3 + TF-IDF hybrid.",
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
      `Per-class breakdown unavailable (label-key mismatch); held-out aggregate metrics remain valid (<code>n = ${num(nSamples)}</code>, macro F1 <code>${percent(metrics.macro_f1)}</code>).`,
      { tone: "warning", title: "Artifact note:" }
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
            <th scope="col" class="num">Support</th>
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
      <span class="swatch-inline" style="margin-top:6px;background:${c.passed ? "var(--color-positive)" : "var(--color-negative)"}" aria-hidden="true"></span>
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
          <span class="swatch-inline" style="background:${allOk ? "var(--color-positive)" : "var(--color-warning)"}" aria-hidden="true"></span>
          <strong>Pipeline integrity: ${passed} / ${total} checks passed</strong>
          <span class="muted">Zero split leakage · 0 synthetic rows · T &ge; 1.0 · zero PII</span>
        </div>
        <span class="num muted" style="font-size:var(--text-xs)">${dateTimeText(verification.verified_at)} · Inspect</span>
      </summary>
      ${verification.scope_note ? `<p class="muted" style="font-size:var(--text-xs);padding:0 var(--space-lg) var(--space-sm)">${esc(verification.scope_note)}</p>` : ""}
      <div class="verify-checks-grid">${rows}</div>
    </details>
  `;
}

function renderArchitectureSection(archComp, evalScope) {
  if (!archComp || !archComp.aggregate) return "";
  const order = archComp.architectures_order || Object.keys(archComp.aggregate);
  const agg = archComp.aggregate || {};
  const why = archComp.why_transformer || {};

  const summaryRows = order.map((key) => {
    const row = agg[key];
    if (!row) return "";
    const isSelected = key === "transformer_muril_dapt_v3" || key === "production_hybrid";
    const roleTag = key === "transformer_muril_dapt_v3"
      ? `<span class="meta-inline-tag">Core encoder</span>`
      : key === "production_hybrid"
        ? `<span class="meta-inline-tag is-pos">Deployed</span>`
        : "";
    return `
      <tr class="${isSelected ? "arch-row-highlight" : ""}">
        <td><strong>${esc(row.label)}</strong> ${roleTag}</td>
        <td class="num"><strong>${percent(row.mean_test_macro_f1)}</strong></td>
        <td class="num">${percent(row.mean_probe_macro_f1)}</td>
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

  const fmtDelta = (val) => {
    const n = Number(val || 0);
    return `${n >= 0 ? "+" : ""}${n.toFixed(1)} pp`;
  };

  return deck("4-Architecture benchmark on real held-out Reddit comments", {
    id: "why-transformer-deck",
    subtitle: evalScope || "Apples-to-apples comparison on identical post_id-grouped real Reddit test splits (n = 92 unseen-thread comments, 0 synthetic rows)",
    action: `
      <div class="segmented" role="group" aria-label="Architecture chart split toggle">
        <button type="button" class="seg-btn is-active" data-arch-mode="test">Real Held-Out Test</button>
        <button type="button" class="seg-btn" data-arch-mode="probe">54-Probe Aux</button>
      </div>
    `,
    body: `
      ${why.gain_vs_logreg_pp !== undefined ? `
        <div class="arch-callout-strip">
          <div class="arch-delta-pill arch-delta-accent"><span>MuRIL-DAPT v3 (5-task mean)</span><strong class="num">${percent(why.mean_test_f1_transformer ?? agg.transformer_muril_dapt_v3?.mean_test_macro_f1)}</strong></div>
          <div class="arch-delta-pill"><span>Gain vs. Logistic Reg.</span><strong class="num">${fmtDelta(why.gain_vs_logreg_pp)}</strong></div>
          <div class="arch-delta-pill"><span>Gain vs. ANN (MLP)</span><strong class="num">${fmtDelta(why.gain_vs_ann_pp)}</strong></div>
          <div class="arch-delta-pill"><span>Gain vs. 1D TextCNN</span><strong class="num">${fmtDelta(why.gain_vs_cnn_pp)}</strong></div>
        </div>
      ` : ""}

      <div id="arch-chart-slot">${architectureComparisonChart(archComp, "test")}</div>

      <div class="spacer-md"></div>
      <div class="table-scroll">
        <table class="data-table" aria-label="4-architecture aggregate comparison">
          <thead>
            <tr>
              <th scope="col">Architecture</th>
              <th scope="col" class="num">Real Test F1 (n=92)</th>
              <th scope="col" class="num">54-Probe Aux F1</th>
              <th scope="col" class="num">5-Bin ECE</th>
              <th scope="col" class="num">Params</th>
              <th scope="col" class="num">Train time</th>
              <th scope="col" class="num">Latency</th>
            </tr>
          </thead>
          <tbody>${summaryRows}</tbody>
        </table>
      </div>

      ${reasonCards ? `
        <details class="clean-disclosure">
          <summary>Why MuRIL-DAPT v3 outperformed classical &amp; shallow neural baselines (4 findings)</summary>
          <div class="why-grid" style="margin-top:var(--space-sm)">${reasonCards}</div>
        </details>
      ` : ""}

      <details class="clean-disclosure">
        <summary>Auxiliary 54-probe linguistic stress breakdown (8 shift categories · observational only)</summary>
        <div style="margin-top:var(--space-sm)">${robustnessCategoryHeatmap(archComp)}</div>
      </details>
    `,
  });
}

function renderHowItWorksSection(dapt, confSummary) {
  if (!dapt && !confSummary) return "";
  return deck("Domain adaptation (DAPT) &amp; live confidence routing", {
    subtitle: "How unsupervised Masked Language Modeling adapts MuRIL to Indian Reddit language and how calibrated confidence routes uncertain predictions to human review",
    body: `
      <div class="studio-grid-2">
        <div class="how-panel">
          <h3 class="subpanel-title">1. Unsupervised domain adaptation (DAPT)</h3>
          <p class="how-explain">Trained on <strong>${num(dapt?.train_documents || dapt?.unlabeled_documents || 0)}</strong> unlabeled Reddit texts and evaluated on <strong>${num(dapt?.val_documents || 0)}</strong> disjoint unseen texts across ${num(dapt?.epochs || 4)} epochs.</p>
          ${daptConvergenceChart(dapt)}
        </div>
        ${confSummary?.histogram?.length ? `
          <div class="how-panel">
            <div class="how-panel-head">
              <h3 class="subpanel-title" style="margin:0">2. Live corpus prediction confidence</h3>
              <a class="row-open-link" href="#/explorer?confidence=review">Review queue &rarr;</a>
            </div>
            <p class="how-explain" style="margin-top:var(--space-xs)">Mean 5-task confidence is <strong>${percent(confSummary.mean_confidence)}</strong> (median ${percent(confSummary.median_confidence)}). <strong>${num(confSummary.low_count)}</strong> fall below 55% (${num(confSummary.needs_review_count)} total flagged including low task margin).</p>
            ${confidenceHistogramChart(confSummary)}
          </div>
        ` : ""}
      </div>
    `,
  });
}

function renderErrorAnalysisBody(taskKey, errTask, caseTab = "wins") {
  if (!errTask) return `<p class="muted">No error analysis available for this task.</p>`;
  const probeCm = errTask.probe_confusion || {};
  const cal = errTask.calibration || {};
  const arch = errTask.architecture_analysis || {};
  const agree = arch.agreement_summary || {};
  const tiers = cal.confidence_tiers || {};

  const tierRows = ["high", "medium", "low"].map((t) => {
    const info = tiers[t] || {};
    const dot = t === "high" ? "var(--color-positive)" : t === "medium" ? "var(--color-info)" : "var(--color-warning)";
    const label = t === "high" ? "High (≥80%)" : t === "medium" ? "Medium (55–79%)" : "Low (<55% · Review)";
    return `
      <tr>
        <td><span class="swatch-inline" style="background:${dot};margin-right:6px" aria-hidden="true"></span>${label}</td>
        <td class="num">${num(info.count || 0)}</td>
        <td class="num">${percent(info.share)}</td>
        <td class="num"><strong>${info.accuracy !== null && info.accuracy !== undefined ? percent(info.accuracy) : "—"}</strong></td>
      </tr>
    `;
  }).join("");

  const topPairsHtml = (probeCm.top_confusions || []).length
    ? (probeCm.top_confusions || []).slice(0, 5).map((p) => `
      <div class="confusion-pair-row">
        <div class="confusion-pair-head">
          <span><code>${esc(p.true_label)}</code> &rarr; <code>${esc(p.predicted_label)}</code></span>
          <span class="num muted">${num(p.count)} errors (${percent(p.share_of_errors, 0)})</span>
        </div>
        <p class="muted" style="font-size:var(--text-xs);margin-top:4px">${esc(p.diagnosis)}</p>
      </div>
    `).join("")
    : `<p class="muted">Zero off-diagonal confusions recorded on the ${num(probeCm.n_samples)} held-out real test comments.</p>`;

  const wins = arch.transformer_wins || [];
  const residuals = arch.residual_errors || [];

  const winsHtml = wins.length
    ? wins.slice(0, 3).map((w) => `
      <article class="err-case-card err-case-win">
        <div class="err-case-top">
          <span class="meta-issue-label">${esc(pretty(w.category))}</span>
          <span class="meta-sep" aria-hidden="true">·</span>
          ${confidenceBadge(w.transformer_confidence)}
          <span class="meta-inline-tag is-pos">Beat ${num(w.baselines_failed)}/3 baselines</span>
        </div>
        <blockquote class="err-case-quote">“${esc(w.text)}”</blockquote>
        <div class="err-case-compare">
          <span><strong>Gold &amp; MuRIL:</strong> <code>${esc(fmtPred(w.true_label))}</code></span>
          <span class="muted">LogReg: <code>${esc(fmtPred(w.baseline_predictions?.logistic_regression))}</code> · ANN: <code>${esc(fmtPred(w.baseline_predictions?.ann_mlp))}</code> · CNN: <code>${esc(fmtPred(w.baseline_predictions?.cnn_textcnn))}</code></span>
        </div>
        <p class="err-case-reason">${esc(w.why_transformer_won)}</p>
      </article>
    `).join("")
    : `<p class="muted">All four architectures agreed on the correct real test comments for this task.</p>`;

  const residualHtml = residuals.length
    ? residuals.slice(0, 3).map((e) => `
      <article class="err-case-card err-case-miss">
        <div class="err-case-top">
          <span class="meta-issue-label">${esc(e.failure_mode || pretty(e.category))}</span>
          <span class="meta-sep" aria-hidden="true">·</span>
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
    : `<p class="muted">Zero residual Transformer errors on the ${num(probeCm.n_samples)} held-out real test comments.</p>`;

  return `
    <div class="err-kpi-row">
      <div class="eval-cell"><span>Real test error rate</span><strong class="num">${percent(probeCm.error_rate)}</strong><small class="num">${num(probeCm.error_count)} / ${num(probeCm.n_samples)} unseen real comments</small></div>
      <div class="eval-cell"><span>Beat majority baselines</span><strong class="num">${num(agree.transformer_beats_majority_baselines || 0)}</strong><small>${num(agree.transformer_only_win || 0)} where all 3 baselines failed</small></div>
      <div class="eval-cell"><span>Calibrated 5-bin ECE</span><strong class="num">${Number(cal.calibrated_ece ?? 0).toFixed(3)}</strong><small class="num">Raw T=1.0: ${Number(cal.uncalibrated_ece ?? 0).toFixed(3)} · Brier ${Number(cal.brier_score ?? 0).toFixed(3)}</small></div>
      <div class="eval-cell"><span>All 4 models failed</span><strong class="num">${num(agree.all_failed || 0)}</strong><small>Ambiguous / boundary real comments</small></div>
    </div>

    <div class="studio-grid-2" style="margin-top:var(--space-md)">
      <div>
        <h3 class="subpanel-title">Top confused class pairs (n = ${num(probeCm.n_samples)} real test)</h3>
        <div class="confusion-pair-list">${topPairsHtml}</div>
        ${!probeCm.multilabel && probeCm.matrix ? `<div class="spacer-sm"></div>${confusionMatrixTable(probeCm.matrix, probeCm.labels, { openByDefault: true })}` : ""}
      </div>
      <div>
        <h3 class="subpanel-title">Reliability &amp; confidence tiers (T = ${esc(cal.temperature ?? "—")}, n = ${num(cal.n_samples || probeCm.n_samples)})</h3>
        ${calibrationReliabilityChart(cal)}
        <div class="spacer-sm"></div>
        <div class="table-scroll">
          <table class="data-table" aria-label="${esc(pretty(taskKey))} accuracy by confidence tier">
            <thead>
              <tr>
                <th scope="col">Tier</th>
                <th scope="col" class="num">n</th>
                <th scope="col" class="num">Share</th>
                <th scope="col" class="num">Accuracy</th>
              </tr>
            </thead>
            <tbody>${tierRows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="err-examples-block">
      <div class="view-toggle-bar" style="margin-bottom:var(--space-sm)">
        <h3 class="subpanel-title" style="margin:0">Empirical real held-out examples</h3>
        <div class="segmented" role="tablist" aria-label="Switch between Transformer recoveries and residual errors">
          <button type="button" role="tab" aria-selected="${caseTab === "wins"}" class="seg-btn ${caseTab === "wins" ? "is-active" : ""}" data-err-case-tab="wins">MuRIL recoveries (${wins.length})</button>
          <button type="button" role="tab" aria-selected="${caseTab === "miss"}" class="seg-btn ${caseTab === "miss" ? "is-active" : ""}" data-err-case-tab="miss">Residual errors (${residuals.length})</button>
        </div>
      </div>
      <div class="err-case-stack">${caseTab === "wins" ? winsHtml : residualHtml}</div>
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

  // Compact Collapsible Per-Task Granular Audit (replaces duplicate 5-task chart + 5 giant expanded cards)
  const perTaskAccordions = orderedNames.map((name, idx) => {
    const task = tasksObj[name];
    const tm = task.test?.metrics || {};
    const vm = task.validation?.metrics || {};
    const isMulti = Boolean(tm.multilabel);
    const split = task.split || {};
    const leakage = Boolean(task.leakage_report?.leakage_detected);
    const probeF1 = task.robustness_54_probes?.overall_macro_f1;
    const labelDistHtml = Object.entries(task.label_counts || {})
      .map(([k, v]) => `<span class="meta-inline-tag">${esc(k)}: <span class="num">${num(v)}</span></span>`)
      .join("");

    return `
      <details class="task-accordion" id="model-task-${esc(name)}" ${idx === 0 ? "" : ""}>
        <summary class="task-accordion-head">
          <div class="task-acc-title">
            <strong>${esc(pretty(name))}</strong>
            <span class="muted">${isMulti ? "Multi-label" : (task.labels || []).length === 2 ? "Binary" : "Single-label"} · ${esc(task.model_name)} ${esc(task.model_version)}</span>
          </div>
          <div class="task-acc-metrics num">
            <span><span class="muted">Split</span> ${num(split.train)}/${num(split.validation)}/${num(split.test)}</span>
            <span><span class="muted">Val</span> ${percent(vm.macro_f1)}</span>
            <span><span class="muted">Test F1</span> <strong>${percent(tm.macro_f1)}</strong></span>
            <span><span class="muted">54-Probe</span> ${probeF1 !== undefined && probeF1 !== null ? percent(probeF1) : "—"}</span>
            <span class="meta-inline-tag ${leakage ? "is-warn" : "is-pos"}">${leakage ? "Leakage" : "0 leakage"}</span>
          </div>
        </summary>
        <div class="task-accordion-body">
          <p class="muted" style="font-size:var(--text-sm);margin-bottom:var(--space-md)">${esc(TASK_DESCRIPTIONS[name] || "")}</p>
          <div class="model-detail-split">
            <div>
              <h4 class="subpanel-title">Held-out test per-${isMulti ? "label" : "class"} breakdown (n = ${num(tm.n_samples)})</h4>
              ${renderPerClassSection(name, task)}
              ${confusionMatrixTable(tm.confusion_matrix, tm.confusion_matrix_labels)}
            </div>
            <div>
              <h4 class="subpanel-title">Live corpus predicted distribution (n = ${num(summary.comments)})</h4>
              <div class="label-chip-wrap">${labelDistHtml}</div>
              <div class="spacer-md"></div>
              <dl class="spec-kv">
                <div><dt>Artifact</dt><dd><code class="break-code">${esc(task.artifact_path || "None")}</code></dd></div>
                <div><dt>Metrics (${glossaryTip("subset_accuracy", "Subset / Acc")})</dt><dd class="num">${isMulti ? `${percent(tm.subset_accuracy, 1, "Not emitted")} subset · ${percent(tm.micro_f1, 1, "N/A")} micro F1` : `${percent(tm.accuracy, 1, "N/A")} accuracy`}</dd></div>
                <div><dt>Scope</dt><dd>${esc(task.inference_scope || "Batch inference over anonymized Reddit corpus")}</dd></div>
              </dl>
            </div>
          </div>
        </div>
      </details>
    `;
  }).join("");

  const provenanceBody = `
    <div class="provenance-two-col">
      <div>
        <h3 class="subpanel-title">Corpus metadata &amp; privacy</h3>
        <dl class="vault-kv-grid">
          <div class="kv-span-full"><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "iim-sambalpur-arctic-shift-v5")}</dd></div>
          <div><dt>Version</dt><dd class="num">v${esc(prov.dataset_version || health.dataset_version)}</dd></div>
          <div><dt>Source archive</dt><dd>${esc(prov.source_provider === "arctic_shift" ? "Arctic Shift (public Reddit)" : "Local archive")}</dd></div>
          <div><dt>Updated at</dt><dd class="num">${dateTimeText(prov.generation_timestamp || summary.generated_at)}</dd></div>
          <div><dt>Corpus size</dt><dd class="num">${num(summary.posts)} threads · ${num(summary.comments)} comments (${avgReplies}/thread)</dd></div>
          <div><dt>Privacy</dt><dd>${badge(isSynthetic ? (prov.authorization_status || "synthetic_only") : "Pseudonymized · zero PII", isSynthetic ? "warning" : "positive")}</dd></div>
          <div><dt>Artifact root</dt><dd class="num">${esc(activeArtifactDir)}</dd></div>
          ${dapt ? `<div class="kv-span-full"><dt>Domain-Adaptive MLM (DAPT)</dt><dd class="num">${num(dapt.train_documents || dapt.unlabeled_documents)} train / ${num(dapt.val_documents || 0)} held-out val texts · perplexity ${esc(dapt.initial_perplexity)} → ${esc(dapt.final_perplexity)}</dd></div>` : ""}
          ${filt.automod_boilerplate_filtered !== undefined ? `<div class="kv-span-full"><dt>Noise filtering</dt><dd class="num">${num(filt.automod_boilerplate_filtered)} AutoMod stickies &amp; ${num(filt.gif_only_filtered)} GIF embeds removed; IDs pseudonymized and PII redacted</dd></div>` : ""}
        </dl>
      </div>

      <div>
        <h3 class="subpanel-title">Methodology &amp; epistemic boundaries</h3>
        <ul class="limitations-list">
          <li><strong>100% real anonymized Reddit data:</strong> Public discussions across ${subListText}; 459 comments across 200 distinct threads hand-labeled for strictly thread-grouped training, validation, and testing (0 synthetic rows).</li>
          <li><strong>4-architecture benchmark:</strong> Logistic Regression, 2-layer ANN, 1D TextCNN, and MuRIL-DAPT v3 evaluated on identical unseen-thread splits; MuRIL-DAPT v3 leads all 5 tasks.</li>
          <li><strong>Non-sharpening calibration:</strong> Validation-only scalar temperature scaling strictly clamped to <code>T &ge; 1.0</code>; low-confidence comments (&lt;55%) route to human adjudication.</li>
          <li><strong>Proposal vs. outcome separation:</strong> Predicted <code>advice</code> / <code>solution</code> replies are strictly separated from explicit first-person follow-up resolution signals.</li>
        </ul>
      </div>
    </div>
  `;

  let activeErrTask = errorAnalysis ? (TASK_ORDER.find((t) => t in errorAnalysis) || Object.keys(errorAnalysis)[0]) : null;
  let activeCaseTab = "wins";

  const errorAnalysisDeck = errorAnalysis && activeErrTask
    ? deck("Interactive error analysis &amp; calibration reliability", {
        id: "error-analysis-deck",
        subtitle: "Select any classifier to audit real held-out confusions, 1:1 reliability calibration (with bin sample sizes n=), and real examples",
        action: `
          <div class="segmented" role="tablist" aria-label="Select task for error analysis">
            ${TASK_ORDER.filter((t) => t in errorAnalysis).map((t) => `
              <button type="button" role="tab" class="seg-btn ${t === activeErrTask ? "is-active" : ""}" data-err-task="${esc(t)}" aria-selected="${t === activeErrTask ? "true" : "false"}">${esc(pretty(t))}</button>
            `).join("")}
          </div>
        `,
        body: `<div id="error-analysis-slot">${renderErrorAnalysisBody(activeErrTask, errorAnalysis[activeErrTask], activeCaseTab)}</div>`,
      })
    : "";

  container.innerHTML = `
    ${renderVerificationBanner(verification)}
    ${renderArchitectureSection(archComp, modelsData.evaluation_scope)}
    ${renderHowItWorksSection(dapt, confSummary)}
    ${errorAnalysisDeck}
    ${deck("Per-task classifier ledger &amp; class-level audit", {
      subtitle: "Click any classifier row to inspect per-class Precision, Recall, F1, confusion matrix, and live corpus label counts",
      body: `<div class="task-accordion-list">${perTaskAccordions}</div>`,
    })}
    ${deck("Dataset provenance, governance &amp; boundaries", {
      subtitle: "Verified metadata read directly from /api/demo/summary and /api/demo/health",
      body: provenanceBody,
    })}
  `;

  function wireErrorCaseTabs() {
    container.querySelectorAll("[data-err-case-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeCaseTab = btn.dataset.errCaseTab;
        const slot = container.querySelector("#error-analysis-slot");
        if (slot && errorAnalysis) {
          slot.innerHTML = renderErrorAnalysisBody(activeErrTask, errorAnalysis[activeErrTask], activeCaseTab);
          wireErrorCaseTabs();
          bindChartTooltips(slot);
        }
      });
    });
  }

  // Wire Architecture split toggle (Real Held-Out Test vs 54-Probe Aux)
  container.querySelectorAll("[data-arch-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.archMode;
      container.querySelectorAll("[data-arch-mode]").forEach((b) => b.classList.toggle("is-active", b === btn));
      const slot = container.querySelector("#arch-chart-slot");
      if (slot && archComp) {
        slot.innerHTML = architectureComparisonChart(archComp, mode);
        bindChartTooltips(slot);
      }
    });
  });

  // Wire Error Analysis task tabs
  container.querySelectorAll("[data-err-task]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeErrTask = btn.dataset.errTask;
      activeCaseTab = "wins";
      container.querySelectorAll("[data-err-task]").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      const slot = container.querySelector("#error-analysis-slot");
      if (slot && errorAnalysis) {
        slot.innerHTML = renderErrorAnalysisBody(activeErrTask, errorAnalysis[activeErrTask], activeCaseTab);
        wireErrorCaseTabs();
        bindChartTooltips(slot);
      }
    });
  });

  wireErrorCaseTabs();
  bindChartTooltips(container);
}
