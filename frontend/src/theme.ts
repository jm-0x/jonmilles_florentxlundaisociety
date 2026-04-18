// Single source of truth for colors. Both components (via inline styles / Nivo)
// and Tailwind (via the @theme block in index.css) draw from these values.
// To re-theme the app: change the hex values here + the matching --color-*
// custom properties in src/index.css. Two files, no component edits needed.

// Primary accent — selections, active states, buttons, chrome.
export const ACCENT = {
  primary: "#fbbf24", // amber-400
  primaryHover: "#fcd34d", // amber-300
  primaryMuted: "#92400e", // amber-800
  primaryBg: "rgb(251 191 36 / 0.15)",
};

// Data accent — heatmap heat scales, data bars.
// Kept amber: orange is intuitive for attention "heat".
export const DATA = {
  primary: "#fbbf24", // amber-400
  primaryHover: "#fcd34d", // amber-300
  primaryMuted: "#92400e", // amber-800
};

// Diff heatmap (compare mode, A − B).
export const DIFF = {
  negative: "#22d3ee", // cyan (A < B)
  positive: "#fbbf24", // amber (A > B)
  zero: "#0a0a0a", // near-black
};

// Intervention tab left-stripes.
export const INTERVENTION = {
  ablate: "#ef4444", // red-500 (necessity probe)
  patch: "#22d3ee", // cyan (sufficiency probe)
};

export const BG = {
  page: "#0a0a0a",
  panel: "#171717",
  panelHover: "#262626",
  border: "#262626",
};

export const TEXT = {
  primary: "#f5f5f5",
  secondary: "#a3a3a3",
  muted: "#525252",
};

export const MONO_FAMILY =
  "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace";
