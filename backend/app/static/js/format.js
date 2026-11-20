/**
 * Formatting, escaping, canonical research taxonomies, and glossary definitions.
 * All colours reference Almanac tokens declared in tokens.css (zero literal colours).
 *
 * Research-integrity guarantees:
 * 1. Canonical 6 response categories remain strictly fixed in exact name and order.
 * 2. percent() NEVER coerces null, undefined, or empty string to "0.0%".
 */

export const CANONICAL_RESPONSE_TYPES = Object.freeze([
  "agreement",
  "personal_experience",
  "opinion",
  "advice",
  "solution",
  "counterargument",
]);

export const RESPONSE_META = Object.freeze({
  agreement: {
    label: "Agreement",
    short: "Agreement",
    color: "var(--color-role-agreement)",
    group: "discourse",
    definition: "Affirms, validates, or echoes a prior concern or observation without adding a concrete fix.",
  },
  personal_experience: {
    label: "Personal experience",
    short: "Experience",
    color: "var(--color-role-experience)",
    group: "discourse",
    definition: "Recounts a first-person encounter, timeline, or outcome with the discussed issue.",
  },
  opinion: {
    label: "Opinion",
    short: "Opinion",
    color: "var(--color-role-opinion)",
    group: "discourse",
    definition: "Expresses a personal judgment, preference, or viewpoint on how a process should work.",
  },
  advice: {
    label: "Advice",
    short: "Advice",
    color: "var(--color-role-advice)",
    group: "actionable",
    definition: "Offers guidance, tips, or recommended actions for navigating the issue individually.",
  },
  solution: {
    label: "Solution",
    short: "Solution",
    color: "var(--color-role-solution)",
    group: "actionable",
    definition: "Proposes a concrete fix, structured workaround, or systemic process change.",
  },
  counterargument: {
    label: "Counterargument",
    short: "Counter",
    color: "var(--color-role-counterargument)",
    group: "discourse",
    definition: "Challenges, qualifies, or presents an opposing perspective to the premise or claim.",
  },
});

export const SENTIMENT_ORDER = Object.freeze(["positive", "neutral", "negative"]);

export const SENTIMENT_META = Object.freeze({
  positive: { label: "Positive", color: "var(--color-positive)" },
  neutral:  { label: "Neutral",  color: "var(--color-neutral-bar)" },
  negative: { label: "Negative", color: "var(--color-negative)" },
});

export const RESOLUTION_ORDER = Object.freeze([
  "reported_success",
  "reported_pending",
  "proposed_only",
  "insufficient_evidence",
]);

export const RESOLUTION_META = Object.freeze({
  reported_success: {
    label: "Reported success",
    short: "Reported success",
    cls: "positive",
    color: "var(--color-positive)",
    description: "An explicit first-person follow-up comment indicates the issue or admission step was resolved.",
  },
  reported_pending: {
    label: "Reported pending",
    short: "Reported pending",
    cls: "warning",
    color: "var(--color-warning)",
    description: "An explicit first-person comment reports trying the steps while the request remains unresolved or pending.",
  },
  proposed_only: {
    label: "Proposal only",
    short: "Proposal only",
    cls: "info",
    color: "var(--color-info)",
    description: "Advice or solution comments were proposed, but no explicit outcome follow-up exists in the thread.",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    short: "Insufficient evidence",
    cls: "neutral",
    color: "var(--color-neutral-bar)",
    description: "Thread lacks explicit resolution outcome evidence; absence of evidence is never treated as proof of failure.",
  },
});

export const EMOTION_COLORS = Object.freeze({
  gratitude: "var(--color-em-gratitude)",
  hope: "var(--color-em-hope)",
  confidence: "var(--color-em-confidence)",
  curiosity: "var(--color-em-curiosity)",
  stress: "var(--color-em-stress)",
  frustration: "var(--color-em-frustration)",
});

export const LANGUAGE_LABELS = Object.freeze({
  en: "English",
  hi: "Hindi (Devanagari)",
  hinglish: "Hinglish (code-mixed)",
});

export const GLOSSARY = Object.freeze({
  macro_f1: "Unweighted mean of F1 scores across every class or label. Treats rare and frequent classes equally.",
  micro_f1: "Global F1 computed by pooling all true positives, false positives, and false negatives across labels.",
  subset_accuracy: "Strictest multi-label metric: requires the entire predicted label set to match the gold set exactly.",
  group_split: "All comments from a thread (post_id) stay inside a single split (train, validation, or test) to prevent cross-split thread leakage.",
  in_sample_vs_heldout: "Thread-level tags shown in the explorer are batch model predictions over anonymized Reddit comments; evaluation tables report held-out thread-grouped test and 54-probe robustness metrics.",
  nmf_topics: "Non-negative Matrix Factorization over TF-IDF unigrams and bigrams fit directly on the anonymized community comment corpus.",
  proposal_vs_resolution: "A comment classified as advice or solution is only a proposed intervention. Resolution requires an explicit follow-up outcome statement in the thread.",
});

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

export function pretty(value) {
  const str = String(value ?? "").replaceAll("_", " ").trim();
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function token(value) {
  return String(value ?? "").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
}

export function num(value) {
  if (value === null || value === undefined || value === "") return "None";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? new Intl.NumberFormat("en-US").format(parsed) : "None";
}

/**
 * Format a 0..1 ratio as a percentage.
 * Strictly guards null/undefined/empty strings so missing metrics NEVER coerce to 0.0%.
 */
export function percent(value, digits = 1, fallback = "Not emitted") {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return `${(parsed * 100).toFixed(digits)}%`;
}

export function share(part, total, digits = 1) {
  const p = Number(part);
  const t = Number(total);
  if (!Number.isFinite(p) || !Number.isFinite(t) || t <= 0) return "0.0%";
  return `${((p / t) * 100).toFixed(digits)}%`;
}

export function dateText(value) {
  if (!value) return "Date unavailable";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(d);
}

export function dateTimeText(value) {
  if (!value) return "Date unavailable";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(d) + " UTC";
}

export function monthLabel(ym) {
  if (!ym || !/^\d{4}-\d{2}$/.test(ym)) return String(ym ?? "");
  const [year, month] = ym.split("-").map(Number);
  const d = new Date(Date.UTC(year, month - 1, 1));
  return new Intl.DateTimeFormat("en-US", { month: "short", year: "2-digit", timeZone: "UTC" }).format(d);
}

export function truncate(text, maxLen = 140) {
  const clean = String(text ?? "").trim();
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, maxLen - 1).trimEnd()}…`;
}

export function shortCommentLabel(commentId) {
  const match = String(commentId ?? "").match(/-(\d+)$/);
  return match ? `#${match[1]}` : String(commentId ?? "");
}

export function issueLabelFromKey(issueKey, summaryData) {
  const hit = summaryData?.issue_distribution?.find((item) => item.key === issueKey);
  return hit ? pretty(hit.label) : pretty(issueKey);
}
