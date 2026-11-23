/**
 * Client-Side Research Export Utilities (Zero-Server, Vercel & Docker Safe).
 * Generates properly escaped RFC-4180 CSV, formatted JSON, and Executive
 * Evidence-Linked Markdown Briefs directly from loaded /api/demo payloads.
 */

function triggerBlobDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(val) {
  const s = val === null || val === undefined ? "" : String(val);
  if (/[",\r\n]/.test(s)) {
    return `"${s.replaceAll('"', '""')}"`;
  }
  return s;
}

export function downloadCsv(rows, columns, filename) {
  if (!Array.isArray(rows) || !rows.length) return false;
  const header = columns.map((c) => csvCell(c.label || c.key)).join(",");
  const body = rows.map((row) =>
    columns.map((c) => csvCell(typeof c.get === "function" ? c.get(row) : row[c.key])).join(",")
  );
  // UTF-8 BOM ensures Excel opens Devanagari/Odia/Bengali/emoji cells cleanly
  triggerBlobDownload(`\uFEFF${[header, ...body].join("\r\n")}`, filename, "text/csv;charset=utf-8");
  return true;
}

export function downloadJson(data, filename) {
  triggerBlobDownload(JSON.stringify(data, null, 2), filename, "application/json;charset=utf-8");
  return true;
}

export function downloadExecutiveMarkdownBrief(summary) {
  if (!summary) return false;
  const prov = summary.provenance || {};
  const resV2 = summary.resolution_v2_summary || {};
  const vd = resV2.verdict_distribution || {};
  const sd = resV2.state_distribution || {};
  const insights = summary.cross_model_insights?.cards || [];
  const issues = [...(summary.issue_distribution || [])].sort((a, b) => (b.comments || 0) - (a.comments || 0));

  const lines = [
    "# Reddit Community Intelligence (RCI) — Executive Insight Brief",
    "",
    `- **Dataset version:** v${summary.dataset_version || "6.1"}`,
    `- **Corpus scope:** ${summary.posts || 0} anonymized threads, ${summary.comments || 0} comments`,
    `- **Provenance:** ${prov.source_provider || "public_reddit"} (${prov.authorization_status || "anonymized"})`,
    `- **Generated:** ${new Date().toISOString().slice(0, 10)}`,
    "",
    "## Headline Findings (Observation vs. Interpretation)",
    "",
  ];

  for (const [idx, c] of insights.entries()) {
    lines.push(`### ${idx + 1}. ${c.headline}`);
    lines.push(`- **Category:** ${c.category} | **Metric:** ${c.metric}`);
    if (c.observation) lines.push(`- **Observation:** ${c.observation}`);
    if (c.interpretation) lines.push(`- **Interpretation:** ${c.interpretation}`);
    if (c.evidence_limitations) lines.push(`- **Limitation:** ${c.evidence_limitations}`);
    lines.push("");
  }

  lines.push("## Explainable Resolution Summary (365 Threads)", "");
  lines.push(`- **Resolved:** ${vd.Resolved ?? 0}`);
  lines.push(`- **Partially Resolved:** ${vd["Partially Resolved"] ?? 0}`);
  lines.push(`- **Unresolved:** ${vd.Unresolved ?? 0}`);
  lines.push(`- **Insufficient Evidence:** ${vd["Insufficient Evidence"] ?? 0}`);
  lines.push(`- **Confirmed Reported Success:** ${sd.reported_success ?? 0} | **Contradictory Follow-Ups:** ${resV2.threads_with_contradictory_followups ?? 0}`);
  lines.push("");

  lines.push("## Issue Domain Distribution", "");
  lines.push("| Issue Domain | Threads | Comments | Solution Role | Advice Role | Binary Actionable |");
  lines.push("| :--- | ---: | ---: | ---: | ---: | ---: |");
  for (const iss of issues) {
    lines.push(
      `| ${iss.label} | ${iss.posts} | ${iss.comments} | ${iss.solutions_proposed} | ${iss.advice_comments} | ${iss.actionable_recommendations} |`
    );
  }
  lines.push("");
  lines.push("---");
  lines.push("_Exported from Reddit Community Intelligence (RCI) Studio. Model classifications are distinct from human ground-truth labels._");

  triggerBlobDownload(lines.join("\n"), `rci_executive_brief_v${summary.dataset_version || "6.2"}.md`, "text/markdown;charset=utf-8");
  return true;
}
