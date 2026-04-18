// Nivo-specific theme + color interpolators. Sources all colors from
// theme.ts so there is a single re-theme point.

import { ACCENT, BG, DATA, DIFF, MONO_FAMILY, TEXT } from "./theme";

export { MONO_FAMILY };

// Back-compat named exports — prefer importing from theme.ts directly.
export const AMBER = DATA.primary;
export const AMBER_DEEP = DATA.primaryMuted;
export const CYAN = DIFF.negative;
export const NEAR_BLACK = BG.page;
export const PANEL_BG = BG.panel;
export const NEUTRAL_100 = TEXT.primary;
export const NEUTRAL_400 = TEXT.secondary;
export const NEUTRAL_500 = "#737373";
export const NEUTRAL_800 = BG.border;

// Nivo theme — tooltip border uses the chrome ACCENT so tooltips match
// the rest of the UI chrome.
export const nivoTheme = {
  background: "transparent",
  text: {
    fontFamily: MONO_FAMILY,
    fontSize: 11,
    fill: TEXT.primary,
    outlineWidth: 0,
    outlineColor: "transparent",
  },
  axis: {
    domain: { line: { stroke: BG.border, strokeWidth: 1 } },
    legend: {
      text: {
        fontFamily: MONO_FAMILY,
        fontSize: 10,
        fill: NEUTRAL_500,
      },
    },
    ticks: {
      line: { stroke: BG.border, strokeWidth: 1 },
      text: {
        fontFamily: MONO_FAMILY,
        fontSize: 10,
        fill: TEXT.secondary,
      },
    },
  },
  grid: { line: { stroke: BG.border, strokeWidth: 1 } },
  legends: {
    title: {
      text: { fontFamily: MONO_FAMILY, fontSize: 10, fill: TEXT.secondary },
    },
    text: { fontFamily: MONO_FAMILY, fontSize: 10, fill: TEXT.secondary },
  },
  labels: {
    text: {
      fontFamily: MONO_FAMILY,
      fontSize: 10,
      fill: BG.page,
      outlineWidth: 0,
    },
  },
  tooltip: {
    container: {
      background: BG.panel,
      color: TEXT.primary,
      fontSize: 11,
      fontFamily: MONO_FAMILY,
      border: `1px solid ${ACCENT.primary}`,
      borderRadius: 0,
      padding: "6px 8px",
      boxShadow: "none",
    },
  },
  annotations: {
    text: {
      fill: TEXT.primary,
      fontFamily: MONO_FAMILY,
      outlineWidth: 0,
      outlineColor: "transparent",
    },
    link: { stroke: ACCENT.primary, strokeWidth: 1 },
    outline: { stroke: ACCENT.primary, strokeWidth: 1 },
    symbol: { fill: ACCENT.primary },
  },
};

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace(/^#/, "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

const ZERO_RGB = hexToRgb(DIFF.zero);
const DATA_RGB = hexToRgb(DATA.primary);
const DATA_DEEP_RGB = hexToRgb(DATA.primaryMuted);
const DIFF_NEG_RGB = hexToRgb(DIFF.negative);
const DIFF_POS_RGB = hexToRgb(DIFF.positive);

function lerpRgb(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): string {
  const r = Math.round(lerp(a[0], b[0], t));
  const g = Math.round(lerp(a[1], b[1], t));
  const bl = Math.round(lerp(a[2], b[2], t));
  return `rgb(${r}, ${g}, ${bl})`;
}

// Diverging scale: cyan ← zero → amber. v in [-maxAbs, maxAbs].
export function divergingColor(v: number, maxAbs: number): string {
  const s = maxAbs <= 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
  if (s >= 0) return lerpRgb(ZERO_RGB, DIFF_POS_RGB, s);
  return lerpRgb(ZERO_RGB, DIFF_NEG_RGB, -s);
}

// Three-stop data-color ramp: near-black → data-muted → data-primary.
// Used by single-trace attention heatmaps (A-only / B-only / standalone).
export function amberColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  if (c < 0.5) return lerpRgb(ZERO_RGB, DATA_DEEP_RGB, c * 2);
  return lerpRgb(DATA_DEEP_RGB, DATA_RGB, (c - 0.5) * 2);
}
