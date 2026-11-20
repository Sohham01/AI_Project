const SENTIMENT_META = [
  { id: "negative", label: "Negative", short: "Neg", key: "1", tone: "negative" },
  { id: "neutral", label: "Neutral", short: "Neu", key: "2", tone: "neutral" },
  { id: "positive", label: "Positive", short: "Pos", key: "3", tone: "positive" },
];

const RESPONSE_META = [
  { id: "agreement", label: "Agreement", short: "Agr", key: "Q", dot: "var(--color-role-agreement)" },
  { id: "personal_experience", label: "Personal exp.", short: "Exp", key: "W", dot: "var(--color-role-experience)" },
  { id: "opinion", label: "Opinion", short: "Opn", key: "E", dot: "var(--color-role-opinion)" },
  { id: "advice", label: "Advice", short: "Adv", key: "R", dot: "var(--color-role-advice)" },
  { id: "solution", label: "Solution", short: "Sol", key: "T", dot: "var(--color-role-solution)" },
  { id: "counterargument", label: "Counterarg.", short: "Ctr", key: "Y", dot: "var(--color-role-counterargument)" },
];

const SOLUTION_META = [
  { id: "not_solution", label: "Not solution", short: "Not sol", key: "N", tone: "neutral" },
  { id: "solution", label: "Solution", short: "Solution", key: "S", tone: "solution" },
];

const EMOTION_META = [
  { id: "confidence", label: "Confidence", key: "A", dot: "var(--color-em-confidence)" },
  { id: "curiosity", label: "Curiosity", key: "D", dot: "var(--color-em-curiosity)" },
  { id: "frustration", label: "Frustration", key: "F", dot: "var(--color-em-frustration)" },
  { id: "gratitude", label: "Gratitude", key: "G", dot: "var(--color-em-gratitude)" },
  { id: "hope", label: "Hope", key: "H", dot: "var(--color-em-hope)" },
  { id: "stress", label: "Stress", key: "J", dot: "var(--color-em-stress)" },
];

const ISSUE_META = [
  { id: "academic_workload", label: "Academic workload", key: "⇧1" },
  { id: "administrative_communication", label: "Admin comm.", key: "⇧2" },
  { id: "campus_infrastructure", label: "Campus infra", key: "⇧3" },
  { id: "curriculum_teaching", label: "Curriculum", key: "⇧4" },
  { id: "extracurriculars", label: "Extracurriculars", key: "⇧5" },
  { id: "financial_concerns", label: "Financial / ROI", key: "⇧6" },
  { id: "hostel_facilities", label: "Hostel and mess", key: "⇧7" },
  { id: "placements", label: "Placements", key: "⇧8" },
];

const ABSTAIN_REASON_LABELS = {
  insufficient_context: "Insufficient context",
  missing_parent: "Missing parent",
  ambiguous_sarcasm: "Ambiguous or sarcasm",
  mixed_or_borderline: "Mixed or borderline",
  off_topic_noise: "Off-topic or bot noise",
  untranslatable: "Untranslatable",
  pii_risk: "PII risk",
  other: "Other",
};

const STATUS_LABELS = {
  unreviewed: "Unreviewed",
  labeled: "Labeled",
  uncertain: "Uncertain",
  abstain: "Abstained",
  excluded: "Excluded",
};

const state = {
  rows: [],
  validation: null,
  thresholds: { min_comments: 60, min_threads: 10, min_per_class: 3 },
  unreviewedNoise: 0,
  filterStatus: "unreviewed",
  filterSubreddit: "",
  filterSearch: "",
  sortMode: "balanced",
  activeDocId: null,
  lastSavedDocId: null,
  sessionSavedCount: 0,
  focusMode: false,
  draft: null,
  abstainOpen: false,
  abstainStatus: "uncertain",
  abstainReason: "insufficient_context",
  saving: false,
  trainPollTimer: null,
};

const $ = (id) => document.getElementById(id);

function showToast(msg, durationMs = 2000) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => {
    el.hidden = true;
  }, durationMs);
}

function getAnnotatorId() {
  return ($("annotator-id").value || "researcher_1").trim() || "researcher_1";
}

function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  const f = params.get("filter");
  if (["unreviewed", "all", "labeled", "adjudication"].includes(f)) {
    state.filterStatus = f;
  }
  const s = params.get("sort");
  if (["balanced", "original"].includes(s)) {
    state.sortMode = s;
  }
  state.filterSubreddit = params.get("sub") || "";
  state.filterSearch = params.get("q") || "";
  state.activeDocId = params.get("doc") || null;
}

function syncUrlState() {
  const params = new URLSearchParams();
  if (state.filterStatus !== "unreviewed") params.set("filter", state.filterStatus);
  if (state.sortMode !== "balanced") params.set("sort", state.sortMode);
  if (state.filterSubreddit) params.set("sub", state.filterSubreddit);
  if (state.filterSearch.trim()) params.set("q", state.filterSearch.trim());
  if (state.activeDocId) params.set("doc", state.activeDocId);
  const qs = params.toString();
  const nextUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  window.history.replaceState(null, "", nextUrl);
}

function filteredRows() {
  const q = state.filterSearch.trim().toLowerCase();
  const list = state.rows.filter((r) => {
    if (state.filterStatus === "unreviewed" && r.review_status !== "unreviewed") return false;
    if (state.filterStatus === "labeled" && r.review_status !== "labeled") return false;
    if (
      state.filterStatus === "adjudication" &&
      !["uncertain", "abstain", "excluded"].includes(r.review_status)
    ) {
      return false;
    }
    if (state.filterSubreddit && r.subreddit !== state.filterSubreddit) return false;
    if (q) {
      const hay = `${r.post_title} ${r.original_text} ${r.document_id} ${r.subreddit}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  if (state.sortMode === "balanced") {
    list.sort((a, b) => a.balanced_rank - b.balanced_rank);
  }
  return list;
}

function getActiveRow() {
  const list = filteredRows();
  if (!list.length) return { row: null, index: -1, list };
  let idx = list.findIndex((r) => r.document_id === state.activeDocId);
  if (idx < 0) {
    idx = 0;
    state.activeDocId = list[0].document_id;
  }
  return { row: list[idx], index: idx, list };
}

function loadDraftFromRow(row) {
  if (!row) {
    state.draft = null;
    return;
  }
  state.draft = {
    sentiment: row.sentiment || "",
    emotions: new Set(row.emotions || []),
    issues: new Set(row.issues || []),
    response_type: row.response_type || "",
    is_solution: row.is_solution || "",
    annotator_notes: row.annotator_notes || "",
  };
  state.abstainOpen = false;
  if (["uncertain", "abstain", "excluded"].includes(row.review_status)) {
    state.abstainStatus = row.review_status;
    state.abstainReason = row.abstain_reason || "insufficient_context";
  } else if (row.is_automod || row.is_gif_only) {
    state.abstainStatus = "excluded";
    state.abstainReason = "off_topic_noise";
  } else {
    state.abstainStatus = "uncertain";
    state.abstainReason = "insufficient_context";
  }
}

function buildStaticChipGroups() {
  const renderSingle = (containerId, items, field) => {
    const el = $(containerId);
    el.innerHTML = "";
    for (const item of items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.dataset.field = field;
      btn.dataset.value = item.id;
      if (item.tone) btn.dataset.tone = item.tone;
      btn.innerHTML = `${item.dot ? `<span class="chip-dot" style="background:${item.dot}" aria-hidden="true"></span>` : ""}<span>${item.label}</span><kbd>${item.key}</kbd>`;
      btn.addEventListener("click", () => {
        if (!state.draft) return;
        state.draft[field] = state.draft[field] === item.id ? "" : item.id;
        syncTriageUI();
      });
      el.appendChild(btn);
    }
  };

  const renderMulti = (containerId, items, field) => {
    const el = $(containerId);
    el.innerHTML = "";
    for (const item of items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.dataset.field = field;
      btn.dataset.value = item.id;
      btn.innerHTML = `${item.dot ? `<span class="chip-dot" style="background:${item.dot}" aria-hidden="true"></span>` : ""}<span>${item.label}</span><kbd>${item.key}</kbd>`;
      btn.addEventListener("click", () => {
        if (!state.draft) return;
        if (state.draft[field].has(item.id)) state.draft[field].delete(item.id);
        else state.draft[field].add(item.id);
        syncTriageUI();
      });
      el.appendChild(btn);
    }
  };

  renderSingle("group-sentiment", SENTIMENT_META, "sentiment");
  renderSingle("group-response", RESPONSE_META, "response_type");
  renderSingle("group-solution", SOLUTION_META, "is_solution");
  renderMulti("group-emotion", EMOTION_META, "emotions");
  renderMulti("group-issue", ISSUE_META, "issues");

  const reasonEl = $("abstain-reasons");
  reasonEl.innerHTML = "";
  for (const [key, label] of Object.entries(ABSTAIN_REASON_LABELS)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip abstain-reason-chip";
    btn.dataset.reason = key;
    btn.textContent = label;
    btn.addEventListener("click", () => {
      state.abstainReason = key;
      syncAbstainUI();
    });
    reasonEl.appendChild(btn);
  }

  document.querySelectorAll("[data-abstain-status]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.abstainStatus = btn.dataset.abstainStatus;
      syncAbstainUI();
    });
  });
}

function syncTriageUI() {
  if (!state.draft) return;

  document.querySelectorAll(".chip[data-field]").forEach((btn) => {
    const { field, value } = btn.dataset;
    const isSel =
      state.draft[field] instanceof Set
        ? state.draft[field].has(value)
        : state.draft[field] === value;
    btn.classList.toggle("selected", isSel);
    btn.setAttribute("aria-pressed", String(isSel));
  });

  const missing = [];
  if (!state.draft.sentiment) missing.push("Sentiment");
  if (!state.draft.response_type) missing.push("Response");
  if (!state.draft.is_solution) missing.push("Solution");

  const saveBtn = $("btn-save-next");
  const hint = $("validation-hint");
  if (missing.length) {
    saveBtn.disabled = true;
    hint.textContent = `Select required: ${missing.join(", ")}.`;
  } else {
    saveBtn.disabled = state.saving;
    hint.textContent = state.saving ? "Saving to CSV…" : "Ready. Press Enter to save and advance.";
  }
}

function syncAbstainUI() {
  $("abstain-panel").hidden = !state.abstainOpen;
  $("btn-toggle-abstain").setAttribute("aria-expanded", String(state.abstainOpen));
  document.querySelectorAll("[data-abstain-status]").forEach((btn) => {
    const active = btn.dataset.abstainStatus === state.abstainStatus;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll("[data-reason]").forEach((btn) => {
    const active = btn.dataset.reason === state.abstainReason;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  });
}

function syncFilterTabsUI() {
  document.querySelectorAll("#filter-tabs .filter-tab").forEach((tab) => {
    const isActive = tab.dataset.filter === state.filterStatus;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });
  $("sort-mode").value = state.sortMode;
  $("filter-search").value = state.filterSearch;
}

function setFocusMode(enabled) {
  state.focusMode = Boolean(enabled);
  $("workbench").classList.toggle("focus-mode", state.focusMode);
  $("btn-focus-mode").setAttribute("aria-pressed", String(state.focusMode));
  localStorage.setItem("rci_annotate_focus", state.focusMode ? "1" : "0");
}

function renderActiveView() {
  const { row, index, list } = getActiveRow();
  $("loading-state").hidden = true;
  $("error-state").hidden = true;

  $("queue-pos").textContent = list.length ? `${index + 1} of ${list.length}` : "0 of 0";
  $("btn-prev").disabled = index <= 0;
  $("btn-next").disabled = index < 0 || index >= list.length - 1;
  $("btn-undo-last").hidden = !state.lastSavedDocId;

  if (!row) {
    $("empty-state").hidden = false;
    $("reading-card").hidden = true;
    $("doc-id-chip").textContent = "No selection";
    $("status-badge").textContent = "Empty";
    $("status-badge").removeAttribute("data-status");
    renderQueueList(list);
    syncUrlState();
    return;
  }

  $("empty-state").hidden = true;
  $("reading-card").hidden = false;

  $("doc-id-chip").textContent = `${row.document_id} (${row.post_id})`;
  const niceStatus = STATUS_LABELS[row.review_status] || row.review_status;
  $("status-badge").textContent = row.abstain_reason
    ? `${niceStatus}: ${ABSTAIN_REASON_LABELS[row.abstain_reason] || row.abstain_reason}`
    : niceStatus;
  $("status-badge").dataset.status = row.review_status;

  $("meta-subreddit").textContent = `r/${row.subreddit}`;
  $("meta-date").textContent = (row.created_at || "").slice(0, 10) || "Unknown date";
  $("meta-score").textContent = row.score !== "" ? `Score ${row.score}` : "Score unrecorded";
  $("meta-lang").textContent = row.is_code_mixed
    ? `${row.language} (code-mixed)`
    : row.language || "Unknown";

  const noiseWrap = $("noise-flag-wrap");
  const noiseFlag = $("noise-flag");
  if (row.is_automod) {
    noiseFlag.textContent = "AutoModerator bot";
    noiseWrap.hidden = false;
  } else if (row.is_gif_only) {
    noiseFlag.textContent = "GIF-only embed";
    noiseWrap.hidden = false;
  } else {
    noiseWrap.hidden = true;
  }

  $("post-title").textContent = row.post_title || "Untitled thread";
  if (row.parent_text) {
    $("parent-quote").hidden = false;
    $("parent-body").textContent = row.parent_text;
  } else {
    $("parent-quote").hidden = true;
  }
  $("comment-body").textContent = row.original_text || row.normalized_text || "";

  loadDraftFromRow(row);
  $("annotator-notes").value = state.draft.annotator_notes;
  syncTriageUI();
  syncAbstainUI();
  renderQueueList(list);
  syncUrlState();
}

function renderQueueList(list) {
  const ul = $("queue-list");
  ul.innerHTML = "";
  const slice = list.slice(0, 120);
  for (const r of slice) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "queue-item" + (r.document_id === state.activeDocId ? " active" : "");
    const snippet = (r.original_text || "").replace(/\s+/g, " ").trim().slice(0, 68);
    const noiseSuffix = r.is_automod ? " (bot)" : r.is_gif_only ? " (GIF)" : "";
    const statusLabel = STATUS_LABELS[r.review_status] || r.review_status;
    const topDiv = document.createElement("div");
    topDiv.className = "queue-item-top";
    const subSpan = document.createElement("span");
    subSpan.textContent = `r/${r.subreddit || ""}${noiseSuffix}`;
    const statusSpan = document.createElement("span");
    statusSpan.className = "queue-item-status";
    statusSpan.dataset.status = r.review_status || "";
    statusSpan.textContent = statusLabel || "";
    topDiv.append(subSpan, statusSpan);

    const snippetDiv = document.createElement("div");
    snippetDiv.className = "queue-item-snippet";
    snippetDiv.textContent = snippet || "(empty comment)";
    btn.append(topDiv, snippetDiv);
    btn.addEventListener("click", () => {
      state.activeDocId = r.document_id;
      renderActiveView();
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
  if (list.length > slice.length) {
    const more = document.createElement("li");
    more.className = "queue-more";
    more.textContent = `${list.length - slice.length} more matching comments`;
    ul.appendChild(more);
  }
}

function renderReadiness() {
  const v = state.validation;
  if (!v) return;
  const th = state.thresholds;
  const labeled = v.labeled_comments || 0;
  const threads = v.labeled_threads || 0;
  const adj = v.adjudication_queue_count || 0;

  $("kpi-labeled").innerHTML = `${labeled}<span class="tally-denom">/${th.min_comments}</span>`;
  $("kpi-threads").innerHTML = `${threads}<span class="tally-denom">/${th.min_threads}</span>`;
  $("kpi-adj").textContent = String(adj);

  const pct = Math.min(100, Math.round((labeled / Math.max(th.min_comments, 1)) * 100));
  const bar = $("bar-comments");
  bar.style.setProperty("--progress", (pct / 100).toFixed(4));
  bar.classList.toggle("ready", Boolean(v.ready_for_training));
  $("progress-bar-role").setAttribute("aria-valuenow", String(pct));
  $("bar-comments-text").textContent = `${labeled} of ${th.min_comments} comments`;
  $("bar-threads-text").textContent = `${threads} of ${th.min_threads} threads`;

  const pill = $("ready-pill");
  pill.textContent = v.ready_for_training ? "Ready to train" : "Not ready";
  pill.classList.toggle("ready", Boolean(v.ready_for_training));

  const dist = v.task_label_distributions || {};
  const groups = [
    {
      name: "Sentiment",
      items: SENTIMENT_META.map((m) => ({
        short: m.short,
        full: `Sentiment: ${m.label}`,
        count: (dist.sentiment || {})[m.id] || 0,
      })),
    },
    {
      name: "Response",
      items: RESPONSE_META.map((m) => ({
        short: m.short,
        full: `Response: ${m.label}`,
        count: (dist.response || {})[m.id] || 0,
      })),
    },
    {
      name: "Solution",
      items: SOLUTION_META.map((m) => ({
        short: m.short,
        full: `Solution: ${m.label}`,
        count: (dist.solution || {})[m.id] || 0,
      })),
    },
  ];

  const allQuotas = groups.flatMap((g) => g.items);
  const unmet = allQuotas.filter((q) => q.count < th.min_per_class);
  const lowestQuota = unmet.length
    ? unmet.reduce((a, b) => (a.count <= b.count ? a : b))
    : null;

  const grid = $("quota-grid");
  grid.innerHTML = "";
  for (const g of groups) {
    const wrap = document.createElement("div");
    wrap.className = "quota-group";

    const heading = document.createElement("span");
    heading.className = "quota-group-name";
    heading.textContent = g.name;
    wrap.appendChild(heading);

    const pills = document.createElement("div");
    pills.className =
      g.items.length === 2
        ? "quota-pill-wrap quota-grid-2"
        : "quota-pill-wrap quota-grid-3";
    for (const q of g.items) {
      const span = document.createElement("span");
      const met = q.count >= th.min_per_class;
      const isLowest = !met && lowestQuota && q.full === lowestQuota.full && labeled > 0;
      span.className = "quota-pill" + (met ? " met" : isLowest ? " lowest" : "");
      span.title = `${q.full} (${q.count} of ${th.min_per_class})`;
      span.innerHTML = `<span>${q.short}</span><span>${q.count}/${th.min_per_class}</span>`;
      pills.appendChild(span);
    }
    wrap.appendChild(pills);
    grid.appendChild(wrap);
  }

  // Session rhythm hint
  const sessEl = $("session-hint");
  if (lowestQuota && labeled > 0) {
    sessEl.textContent = `${state.sessionSavedCount} saved this session. Lowest quota: ${lowestQuota.full} (${lowestQuota.count}/${th.min_per_class}).`;
  } else if (v.ready_for_training) {
    sessEl.textContent = `${state.sessionSavedCount} saved this session. All class quotas are met.`;
  } else {
    sessEl.textContent = `${state.sessionSavedCount} saved this session.`;
  }

  // Bot/GIF noise banner
  const unrevNoise = state.rows.filter(
    (r) => r.review_status === "unreviewed" && (r.is_automod || r.is_gif_only)
  ).length;
  state.unreviewedNoise = unrevNoise;
  $("noise-banner").hidden = unrevNoise === 0;
  $("noise-count-label").textContent = `${unrevNoise} unreviewed bot or GIF row${unrevNoise === 1 ? "" : "s"}`;

  // Training gate status
  const actions = $("train-actions");
  const note = $("train-note");
  if (v.ready_for_training) {
    actions.hidden = false;
    note.textContent =
      "All readiness quotas are satisfied. Start thread-grouped training into isolated ml/artifacts/real_arctic_shift_v1/.";
  } else {
    actions.hidden = true;
    const needParts = [];
    const needComments = Math.max(0, th.min_comments - labeled);
    const needThreads = Math.max(0, th.min_threads - threads);
    if (needComments > 0) needParts.push(`${needComments} more labeled comment${needComments === 1 ? "" : "s"}`);
    if (needThreads > 0) needParts.push(`${needThreads} more thread${needThreads === 1 ? "" : "s"}`);
    if (unmet.length > 0) needParts.push(`${unmet.length} class quota${unmet.length === 1 ? "" : "s"} under ${th.min_per_class}`);
    if (v.error_count > 0) needParts.push(`${v.error_count} schema error${v.error_count === 1 ? "" : "s"}`);
    note.textContent = needParts.length
      ? `Remaining before training: ${needParts.join(", ")}.`
      : "Meet every quota above to start supervised training.";
  }
}

async function saveCurrentRow(statusOverride = null) {
  const { row, index, list } = getActiveRow();
  if (!row || state.saving) return;

  const targetStatus = statusOverride || "labeled";
  const payload = {
    review_status: targetStatus,
    sentiment: targetStatus === "labeled" ? state.draft.sentiment : "",
    emotions: targetStatus === "labeled" ? Array.from(state.draft.emotions) : [],
    issues: targetStatus === "labeled" ? Array.from(state.draft.issues) : [],
    response_type: targetStatus === "labeled" ? state.draft.response_type : "",
    is_solution: targetStatus === "labeled" ? state.draft.is_solution : "",
    abstain_reason: ["uncertain", "abstain", "excluded"].includes(targetStatus)
      ? state.abstainReason
      : "",
    annotator_id: getAnnotatorId(),
    annotator_notes: ($("annotator-notes").value || "").trim(),
  };

  if (targetStatus === "labeled") {
    if (!payload.sentiment || !payload.response_type || !payload.is_solution) {
      showToast("Select Sentiment, Community response, and Solution before saving.");
      return;
    }
  }

  const nextDocInCurrentList =
    index + 1 < list.length
      ? list[index + 1].document_id
      : index - 1 >= 0
      ? list[index - 1].document_id
      : null;

  state.saving = true;
  syncTriageUI();
  try {
    const res = await fetch(`/api/annotate/row/${encodeURIComponent(row.document_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(`Save failed: ${data.detail || res.status}`, 3500);
      return;
    }

    const rIdx = state.rows.findIndex((r) => r.document_id === row.document_id);
    if (rIdx >= 0) state.rows[rIdx] = data.row;
    state.validation = data.validation;

    if (targetStatus !== "unreviewed") {
      state.lastSavedDocId = row.document_id;
      state.sessionSavedCount += 1;
      if (nextDocInCurrentList) {
        state.activeDocId = nextDocInCurrentList;
      }
    }
    showToast(
      targetStatus === "labeled"
        ? `Saved (${data.validation.labeled_comments} labeled across ${data.validation.labeled_threads} threads)`
        : targetStatus === "unreviewed"
        ? "Reset row to unreviewed"
        : `Routed to adjudication (${STATUS_LABELS[targetStatus] || targetStatus})`
    );
    renderReadiness();
    renderActiveView();
  } catch (err) {
    showToast(`Network error while saving: ${err.message}`, 3500);
  } finally {
    state.saving = false;
    syncTriageUI();
  }
}

function jumpToLastSaved() {
  if (!state.lastSavedDocId) return;
  const existsInCurrent = filteredRows().some((r) => r.document_id === state.lastSavedDocId);
  if (!existsInCurrent) {
    state.filterStatus = "all";
    syncFilterTabsUI();
  }
  state.activeDocId = state.lastSavedDocId;
  renderActiveView();
  showToast("Returned to your last saved comment.");
}

async function excludeBotNoise() {
  const btn = $("btn-exclude-noise");
  btn.disabled = true;
  try {
    const res = await fetch("/api/annotate/exclude-bot-noise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotator_id: getAnnotatorId() }),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast("Unable to exclude bot rows.", 3000);
      return;
    }
    await reloadState();
    showToast(`Excluded ${data.excluded_count} AutoModerator and GIF rows to adjudication.`);
  } catch (err) {
    showToast(`Request failed: ${err.message}`, 3500);
  } finally {
    btn.disabled = false;
  }
}

function renderTrainJob(job) {
  const box = $("train-status");
  if (!job || job.status === "idle") {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  if (job.status === "running") {
    box.textContent = job.include_muril
      ? "Training V1/V2 TF-IDF and GPU MuRIL in background (about 3 to 5 minutes on RTX 4060)…"
      : "Training V1/V2 Word+Char calibrated TF-IDF in background…";
    $("btn-train-classical").disabled = true;
    $("btn-train-muril").disabled = true;
    if (!state.trainPollTimer) {
      state.trainPollTimer = setInterval(pollTrainStatus, 2500);
    }
  } else if (job.status === "completed") {
    clearInterval(state.trainPollTimer);
    state.trainPollTimer = null;
    $("btn-train-classical").disabled = false;
    $("btn-train-muril").disabled = false;
    const tasks = job.summary?.supervised_models?.tasks || {};
    const lines = ["Training complete (held-out Macro F1):"];
    for (const [task, info] of Object.entries(tasks)) {
      const v1 = info.baseline_v1_tfidf?.held_out_test?.macro_f1;
      const v2 = info.upgraded_v2_word_char_calibrated?.held_out_test?.macro_f1;
      const mu = info.muril_v2_gpu?.held_out_test?.macro_f1;
      const muStr = mu !== undefined ? ` | MuRIL ${mu.toFixed(3)}` : "";
      lines.push(`${task.padEnd(10)} V1 ${v1?.toFixed(3) ?? "n/a"} | V2 ${v2?.toFixed(3) ?? "n/a"}${muStr}`);
    }
    box.textContent = lines.join("\n");
  } else if (job.status === "failed") {
    clearInterval(state.trainPollTimer);
    state.trainPollTimer = null;
    $("btn-train-classical").disabled = false;
    $("btn-train-muril").disabled = false;
    box.textContent = `Training failed (code ${job.return_code}): ${job.error || "Unknown error"}`;
  }
}

async function pollTrainStatus() {
  try {
    const res = await fetch("/api/annotate/train/status");
    if (res.ok) renderTrainJob(await res.json());
  } catch {
    // Ignore transient poll errors while training runs locally
  }
}

async function startTraining(includeMuril) {
  try {
    const res = await fetch("/api/annotate/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_muril: includeMuril, muril_epochs: 6 }),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail?.message || data.detail || "Training gate not met.", 3500);
      return;
    }
    renderTrainJob(data);
    showToast(includeMuril ? "Started V1/V2 and GPU MuRIL training." : "Started V1/V2 TF-IDF training.");
  } catch (err) {
    showToast(`Unable to start training: ${err.message}`, 3500);
  }
}

function navigate(delta) {
  const { index, list } = getActiveRow();
  if (!list.length) return;
  const next = Math.max(0, Math.min(list.length - 1, index + delta));
  if (next !== index) {
    state.activeDocId = list[next].document_id;
    renderActiveView();
  }
}

function bindEvents() {
  $("btn-prev").addEventListener("click", () => navigate(-1));
  $("btn-next").addEventListener("click", () => navigate(1));
  $("btn-undo-last").addEventListener("click", jumpToLastSaved);
  $("btn-save-next").addEventListener("click", () => saveCurrentRow("labeled"));
  $("btn-reset-row").addEventListener("click", () => saveCurrentRow("unreviewed"));
  $("btn-toggle-abstain").addEventListener("click", () => {
    state.abstainOpen = !state.abstainOpen;
    syncAbstainUI();
  });
  $("btn-cancel-abstain").addEventListener("click", () => {
    state.abstainOpen = false;
    syncAbstainUI();
  });
  $("btn-confirm-abstain").addEventListener("click", () => saveCurrentRow(state.abstainStatus));
  $("btn-exclude-noise").addEventListener("click", excludeBotNoise);
  $("btn-train-classical").addEventListener("click", () => startTraining(false));
  $("btn-train-muril").addEventListener("click", () => startTraining(true));
  $("btn-retry-load").addEventListener("click", reloadState);
  $("btn-show-all").addEventListener("click", () => {
    state.filterStatus = "all";
    state.filterSubreddit = "";
    state.filterSearch = "";
    $("filter-subreddit").value = "";
    syncFilterTabsUI();
    state.activeDocId = null;
    renderActiveView();
  });

  $("btn-focus-mode").addEventListener("click", () => setFocusMode(!state.focusMode));

  document.querySelectorAll("#filter-tabs .filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.filterStatus = tab.dataset.filter;
      syncFilterTabsUI();
      state.activeDocId = null;
      renderActiveView();
    });
  });

  $("filter-subreddit").addEventListener("change", (e) => {
    state.filterSubreddit = e.target.value;
    state.activeDocId = null;
    renderActiveView();
  });
  $("filter-search").addEventListener("input", (e) => {
    state.filterSearch = e.target.value;
    state.activeDocId = null;
    renderActiveView();
  });
  $("sort-mode").addEventListener("change", (e) => {
    state.sortMode = e.target.value;
    renderActiveView();
  });

  const savedAnnotator = localStorage.getItem("rci_annotator_id");
  if (savedAnnotator) $("annotator-id").value = savedAnnotator;
  $("annotator-id").addEventListener("change", (e) => {
    localStorage.setItem("rci_annotator_id", e.target.value.trim() || "researcher_1");
  });

  const savedFocus = localStorage.getItem("rci_annotate_focus") === "1";
  setFocusMode(savedFocus);

  const savedTheme = localStorage.getItem("rci_annotate_theme") || "dark";
  document.documentElement.dataset.theme = savedTheme;
  $("btn-theme").textContent = savedTheme === "dark" ? "Light" : "Dark";
  $("btn-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("rci_annotate_theme", next);
    $("btn-theme").textContent = next === "dark" ? "Light" : "Dark";
  });

  const dialog = $("guide-dialog");
  $("btn-guidelines").addEventListener("click", () => dialog.showModal());
  $("btn-close-guide").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close();
  });

  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) || "";
    const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

    if (e.key === "Escape") {
      if (dialog.open) {
        dialog.close();
      } else if (state.abstainOpen) {
        state.abstainOpen = false;
        syncAbstainUI();
      } else if (inField) {
        e.target.blur();
      }
      return;
    }

    if (inField || e.ctrlKey || e.metaKey || e.altKey) return;
    if (dialog.open) return;

    if (e.key === "?") {
      e.preventDefault();
      dialog.showModal();
      return;
    }
    if (e.key === "[" || e.key === "ArrowLeft") {
      e.preventDefault();
      navigate(-1);
      return;
    }
    if (e.key === "]" || e.key === "ArrowRight") {
      e.preventDefault();
      navigate(1);
      return;
    }

    const low = e.key.toLowerCase();
    if (!e.shiftKey && low === "m") {
      e.preventDefault();
      setFocusMode(!state.focusMode);
      return;
    }
    if (!e.shiftKey && low === "z" && state.lastSavedDocId) {
      e.preventDefault();
      jumpToLastSaved();
      return;
    }

    if (!state.draft) return;

    if (e.shiftKey && /^Digit[1-8]$/.test(e.code)) {
      e.preventDefault();
      const idx = Number(e.code.replace("Digit", "")) - 1;
      const issue = ISSUE_META[idx];
      if (issue) {
        if (state.draft.issues.has(issue.id)) state.draft.issues.delete(issue.id);
        else state.draft.issues.add(issue.id);
        syncTriageUI();
      }
      return;
    }

    if (e.shiftKey) return;

    if (low === "1") { state.draft.sentiment = "negative"; syncTriageUI(); return; }
    if (low === "2") { state.draft.sentiment = "neutral"; syncTriageUI(); return; }
    if (low === "3") { state.draft.sentiment = "positive"; syncTriageUI(); return; }

    const respMap = {
      q: "agreement",
      w: "personal_experience",
      e: "opinion",
      r: "advice",
      t: "solution",
      y: "counterargument",
    };
    if (respMap[low]) {
      state.draft.response_type = respMap[low];
      syncTriageUI();
      return;
    }

    if (low === "n") { state.draft.is_solution = "not_solution"; syncTriageUI(); return; }
    if (low === "s") { state.draft.is_solution = "solution"; syncTriageUI(); return; }

    const emoMap = {
      a: "confidence",
      d: "curiosity",
      f: "frustration",
      g: "gratitude",
      h: "hope",
      j: "stress",
    };
    if (emoMap[low]) {
      const id = emoMap[low];
      if (state.draft.emotions.has(id)) state.draft.emotions.delete(id);
      else state.draft.emotions.add(id);
      syncTriageUI();
      return;
    }

    if (low === "u") {
      e.preventDefault();
      state.abstainOpen = !state.abstainOpen;
      syncAbstainUI();
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      if (state.abstainOpen) {
        saveCurrentRow(state.abstainStatus);
      } else {
        saveCurrentRow("labeled");
      }
    }
  });
}

async function reloadState() {
  $("loading-state").hidden = false;
  $("error-state").hidden = true;
  $("empty-state").hidden = true;
  $("reading-card").hidden = true;

  try {
    const res = await fetch("/api/annotate/state");
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    const data = await res.json();
    state.rows = data.rows || [];
    state.validation = data.validation;
    state.thresholds = data.thresholds || state.thresholds;

    const subSel = $("filter-subreddit");
    subSel.innerHTML = '<option value="">All subreddits</option>';
    for (const sub of data.subreddits || []) {
      const opt = document.createElement("option");
      opt.value = sub;
      opt.textContent = `r/${sub}`;
      subSel.appendChild(opt);
    }
    subSel.value = state.filterSubreddit;

    syncFilterTabsUI();
    renderReadiness();
    renderActiveView();
    if (data.train_job) renderTrainJob(data.train_job);
  } catch (err) {
    $("loading-state").hidden = true;
    $("error-state").hidden = false;
    $("error-message").textContent = `Could not reach /api/annotate/state (${err.message}). Verify the local studio server is running and try again.`;
  }
}

async function init() {
  readUrlState();
  buildStaticChipGroups();
  bindEvents();
  await reloadState();
}

init();
