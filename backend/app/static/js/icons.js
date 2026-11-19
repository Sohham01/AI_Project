/**
 * Lightweight inline SVG icon helpers (zero external icon fonts or emojis).
 */
const SVG_ATTR = 'width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"';

export const icons = {
  arrowRight: `<svg ${SVG_ATTR}><path d="M4.5 10H15.5M11 5.5L15.5 10L11 14.5" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  arrowLeft: `<svg ${SVG_ATTR}><path d="M15.5 10H4.5M9 5.5L4.5 10L9 14.5" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  copy: `<svg width="13" height="13" viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="7" y="7" width="9.5" height="9.5" rx="1.5" stroke="currentColor" stroke-width="1.6"/><path d="M13 7V4.8A1.3 1.3 0 0 0 11.7 3.5H4.8A1.3 1.3 0 0 0 3.5 4.8V11.7A1.3 1.3 0 0 0 4.8 13H7" stroke="currentColor" stroke-width="1.6"/></svg>`,
  info: `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="7.5" stroke="currentColor" stroke-width="1.6"/><path d="M10 9.2V13.8M10 6.5H10.01" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  alert: `<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 3.2L17.6 16.5H2.4L10 3.2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M10 8.2V11.6M10 14.1H10.01" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`,
  check: `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4.5 10.5L8.2 14.2L15.8 6.4" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  reply: `<svg width="13" height="13" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M8 5L3.5 9.5L8 14" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 9.5H12.5C15 9.5 16.5 11 16.5 13.5V15" stroke="currentColor" stroke-width="1.65" stroke-linecap="round"/></svg>`,
  filter: `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 4.5H17L11.8 10.5V15.5L8.2 13.8V10.5L3 4.5Z" stroke="currentColor" stroke-width="1.55" stroke-linejoin="round"/></svg>`,
  close: `<svg width="12" height="12" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5.5 5.5L14.5 14.5M14.5 5.5L5.5 14.5" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>`,
  chevronDown: `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5.5 7.75L10 12.25L14.5 7.75" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  external: `<svg width="13" height="13" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M11.5 4H16V8.5M15.8 4.2L9.2 10.8M14 11.5V15.2A.8.8 0 0 1 13.2 16H4.8A.8.8 0 0 1 4 15.2V6.8A.8.8 0 0 1 4.8 6H8.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};
