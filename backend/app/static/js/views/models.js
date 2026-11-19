/**
 * View 9: Models & Dataset Provenance Vault
 * Preserves 100% honest reporting: Emotion/Issue subset_accuracy: null -> "Not emitted",
 * Solution baseline zero-support artifact label-key mismatch disclosure,
 * held-out validation vs test F1 chart, warnings, and confusion heatmaps.
 */
import { api } from "../api.js";
import { confusionMatrixTable, modelComparisonSvgChart } from "../charts.js";
import { badge, callout, copyIdChip, deck, glossaryTip } from "../components.js";
import { dateTimeText, esc, num, percent, pretty } from "../format.js";

const TASK_ORDER = ["sentiment", "emotion", "issue", "response", "solution"];

const TASK_DESCRIPTIONS = {
  sentiment: "3-class mutually exclusive polarity classifier (negative, neutral, positive).",
  emotion: "6-label multi-label affect classifier (confidence, curiosity, frustration, gratitude, hope, stress).",
  issue: "8-label multi-label issue category detector trained on comment text alone.",
  response: "6-class multiclass classifier preserving exact canonical response taxonomy.",
  solution: "Binary actionable-solution detector (is_solution: true vs. false; advice and solution roles).",
};

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
      `<strong>Per-class breakdown and confusion matrix unavailable in this frozen artifact (label-key mismatch):</strong> In <code>backend/train_demo_baselines.py</code>, held-out per-class accounting compares gold and predicted sets (<code>{"true"}</code> / <code>{"false"}</code>) against string class names (<code>"not_solution"</code>, <code>"solution"</code>), yielding <code>support: 0</code> and <code>[[0,0],[0,0]]</code> despite valid aggregate held-out metrics above (<code>n = ${num(nSamples)}</code>, accuracy <code>${percent(metrics.accuracy)}</code>, macro F1 <code>${percent(metrics.macro_f1)}</code>). Existing artifacts are intentionally preserved unmodified.`,
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
  const firstArtifactPath = tasksObj.sentiment?.artifact_path || tasksObj.solution?.artifact_path || "";
  const activeArtifactDir = firstArtifactPath.includes("/")
    ? `${firstArtifactPath.slice(0, firstArtifactPath.lastIndexOf("/") + 1)}`
    : "ml/artifacts/demo_synthetic_v2/";
  const provenanceBody = `
    <div class="provenance-two-col">
      <div>
        <h3 class="subpanel-title">Corpus metadata &amp; isolation</h3>
        <dl class="vault-kv-grid">
          <div class="kv-span-full"><dt>Dataset identifier</dt><dd>${copyIdChip(prov.dataset_id || "reddit-community-intelligence-demo")}</dd></div>
          <div><dt>Dataset version</dt><dd class="num">v${esc(prov.dataset_version || health.dataset_version)}</dd></div>
          <div><dt>Deterministic seed</dt><dd class="num">${esc(prov.generation_seed)}</dd></div>
          <div><dt>Generated at</dt><dd class="num">${dateTimeText(prov.generation_timestamp || summary.generated_at)}</dd></div>
          <div><dt>Corpus size</dt><dd class="num">${num(summary.posts)} posts, ${num(summary.comments)} comments (8 / thread)</dd></div>
          <div><dt>Authorization status</dt><dd><span class="tag tag-warning">${esc(prov.authorization_status || "synthetic_only")}</span></dd></div>
          <div><dt>Artifact directory</dt><dd class="num">${esc(activeArtifactDir)}</dd></div>
        </dl>
      </div>

      <div>
        <h3 class="subpanel-title">Known pilot limitations</h3>
        <ul class="limitations-list">
          <li><strong>Synthetic generator family only:</strong> All posts, comments, IDs, labels, and outcome statements are fabricated. Not real Reddit data or real student testimony.</li>
          <li><strong>Inflated synthetic scores:</strong> Near-100% test F1 on sentiment, response, emotion, and solution reflects reproducible generator templates, not expected accuracy on noisy real discussions.</li>
          <li><strong>In-sample explorer display vs. held-out evaluation:</strong> Predictions on thread pages cover all 1,280 records; evaluation tables below strictly reflect the held-out 192-comment test split.</li>
          <li><strong>Partial multilingual coverage:</strong> Pilot includes English, Devanagari Hindi, and Hinglish; Bengali, Odia, Banglish, Gen-Z slang, and emoji semantics are not yet evaluated.</li>
          <li><strong>Task-appropriate metric families:</strong> Multi-label tasks (<code>emotion</code>, <code>issue</code>) report <code>subset_accuracy</code>, <code>hamming_loss</code>, <code>macro_f1</code>, and <code>micro_f1</code> (with single-label <code>accuracy</code> kept <code>null</code>); any uncomputed metric is displayed as <em>Not emitted</em> rather than 0%.</li>
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
        <td class="num">${percent(tm.micro_f1, 1, "N/A")}</td>
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
          `In <code>evaluation.json</code>, <code>subset_accuracy</code>, <code>hamming_loss</code>, and <code>hamming_accuracy</code> are <code>null</code> because <code>ml/evaluation/reports.py::_extract_metrics()</code> extracts only <code>macro_f1</code>, <code>micro_f1</code>, and <code>per_label</code> for multi-label tasks. Earlier prototype UI versions erroneously coerced <code>null</code> to <code>0.0%</code>.`,
          { tone: "info", title: "Metric definition note:" }
        )
      : "";

    const warningsHtml = allWarnings.length
      ? `<div class="eval-warnings"><span class="eval-warnings-label">Evaluator &amp; split warnings (${allWarnings.length}):</span><ul>${allWarnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`
      : "";

    const labelDistHtml = Object.entries(task.label_counts || {})
      .map(([k, v]) => `<span class="tag tag-neutral">${esc(k)}: <span class="num">${num(v)}</span></span>`)
      .join("");

    return deck(`${esc(pretty(name))} classifier`, {
      id: `model-task-${esc(name)}`,
      className: "model-detail-card",
      subtitle: TASK_DESCRIPTIONS[name] || "",
      action: `
        ${badge(`${task.model_name} ${task.model_version}`, "info")}
        ${copyIdChip(task.registry_id || "unregistered", { label: "registry" })}
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
            <h3 class="subpanel-title">Full corpus label distribution (n = ${num(task.dataset_records)})</h3>
            <div class="label-chip-wrap">${labelDistHtml}</div>
            <div class="spacer-md"></div>
            <h3 class="subpanel-title">Artifact &amp; scope</h3>
            <dl class="spec-kv">
              <div><dt>Artifact</dt><dd><code class="break-code">${esc(task.artifact_path || "None")}</code></dd></div>
              <div><dt>Split seed</dt><dd class="num">${esc(split.seed)}</dd></div>
              <div><dt>Stratified</dt><dd>${split.stratify_used ? "Yes" : "No (group-aware)"}</dd></div>
              <div><dt>Scope</dt><dd>${esc(task.inference_scope || "Synthetic pilot only")}</dd></div>
            </dl>
          </div>
        </div>
      `,
    });
  }).join("");

  container.innerHTML = `
    ${callout(
      `<strong>Synthetic training and evaluation data only.</strong> ${esc(modelsData.evaluation_scope)} High scores on synthetic template data must never be cited as real-world community NLP accuracy.`,
      { tone: "warning", title: "Evaluation scope:" }
    )}
    ${deck("Dataset provenance, governance & pilot limitations", {
      subtitle: "Verified metadata read directly from /api/demo/summary and /api/demo/health",
      body: provenanceBody,
    })}
    ${deck("Cross-task baseline performance summary", {
      subtitle: "Side-by-side comparison of held-out validation vs. held-out test macro F1 across all five TF-IDF baselines (896 / 192 / 192 thread-grouped splits)",
      body: `
        ${modelComparisonSvgChart(chartTasks)}
        <div class="spacer-md"></div>
        <div class="table-scroll">
          <table class="data-table" aria-label="Five synthetic TF-IDF baseline models comparison">
            <thead>
              <tr>
                <th scope="col">Task &amp; model</th>
                <th scope="col">Paradigm</th>
                <th scope="col" class="num">Train / Val / Test</th>
                <th scope="col">${glossaryTip("group_split", "Leakage")}</th>
                <th scope="col" class="num">Val macro F1</th>
                <th scope="col" class="num">${glossaryTip("macro_f1", "Test macro F1")}</th>
                <th scope="col" class="num">${glossaryTip("micro_f1", "Test micro F1")}</th>
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

  container.querySelectorAll("[data-scroll-task]").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      container.querySelector(`#model-task-${CSS.escape(link.dataset.scrollTask)}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}
