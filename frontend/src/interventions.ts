import type { Intervention, Trace } from "./types";

// Deterministic sort key. Prefix ensures types group together in a stable order.
function key(iv: Intervention): string {
  if (iv.type === "zero_head") return `0:zero_head:${iv.layer}:${iv.head}`;
  if (iv.type === "zero_mlp") return `1:zero_mlp:${iv.layer}`;
  return `2:patch_head:${iv.layer}:${iv.head}:${iv.source_prompt}`;
}

export function sortedInterventions(ivs: Intervention[]): Intervention[] {
  return ivs.slice().sort((a, b) => key(a).localeCompare(key(b)));
}

export function traceId(prompt: string, ivs: Intervention[]): string {
  return `${prompt}|||${JSON.stringify(sortedInterventions(ivs))}`;
}

// Pick a short, recognisable tag for a source prompt: the first capitalised
// word after position 0 (which is usually the sentence-start capital).
// Falls back to the first word if no internal capital is found.
export function sourceTag(sourcePrompt: string): string {
  const words = sourcePrompt.trim().split(/\s+/);
  for (let i = 1; i < words.length; i++) {
    const raw = words[i].replace(/[^A-Za-z]/g, "");
    if (raw && raw[0] === raw[0].toUpperCase() && raw.toLowerCase() !== raw) {
      return raw.slice(0, 10);
    }
  }
  return (words[0] ?? "src").replace(/[^A-Za-z0-9]/g, "").slice(0, 10) || "src";
}

export function interventionBadge(iv: Intervention): string {
  if (iv.type === "zero_head") return `−L${iv.layer}.H${iv.head}`;
  if (iv.type === "zero_mlp") return `−L${iv.layer}.MLP`;
  return `L${iv.layer}.H${iv.head}←${sourceTag(iv.source_prompt)}`;
}

export function traceLabel(trace: Trace): string {
  const ivs = trace.interventions_applied ?? [];
  if (ivs.length === 0) return trace.prompt;
  const prefix = `[${ivs.map(interventionBadge).join(", ")}] `;
  return prefix + trace.prompt;
}

export type TraceAccent = "none" | "ablate" | "patch";

export function traceAccent(trace: Trace): TraceAccent {
  const ivs = trace.interventions_applied ?? [];
  if (ivs.some((iv) => iv.type === "patch_head")) return "patch";
  if (ivs.length > 0) return "ablate";
  return "none";
}

// Compact magnitude for tab badges: "1.2" when ≥1.0, ".13" otherwise, ".00"
// for near-zero effects (we show "did nothing" explicitly rather than hiding).
export function formatMagnitude(m: number): string {
  const a = Math.abs(m);
  if (a >= 1) return a.toFixed(1);
  const s = a.toFixed(2); // "0.13" | "1.00" if rounding
  return s.startsWith("0") ? s.slice(1) : s; // drop leading zero
}
