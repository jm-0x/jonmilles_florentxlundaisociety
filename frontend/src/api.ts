import type {
  Comparison,
  Intervention,
  SweepRequestBody,
  SweepResponse,
  Trace,
} from "./types";

const BACKEND = "http://localhost:8000";

export async function fetchTrace(
  prompt: string,
  interventions: Intervention[] = [],
  topK = 5
): Promise<Trace> {
  const payload: Record<string, unknown> = { prompt, top_k: topK };
  if (interventions.length > 0) payload.interventions = interventions;
  const res = await fetch(`${BACKEND}/trace`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`trace request failed (${res.status}): ${body || res.statusText}`);
  }
  return (await res.json()) as Trace;
}

export function comparisonKey(a: string, b: string): string {
  const [x, y] = [a, b].slice().sort();
  return `${x}|${y}`;
}

const _comparisonCache = new Map<string, Comparison>();

export function getCachedComparison(a: string, b: string): Comparison | undefined {
  return _comparisonCache.get(comparisonKey(a, b));
}

export async function runSweep(body: SweepRequestBody): Promise<SweepResponse> {
  const res = await fetch(`${BACKEND}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as SweepResponse;
}

export async function fetchComparison(a: string, b: string): Promise<Comparison> {
  const key = comparisonKey(a, b);
  const cached = _comparisonCache.get(key);
  if (cached) return cached;

  const res = await fetch(`${BACKEND}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt_a: a, prompt_b: b }),
  });
  if (!res.ok) {
    // Try to parse FastAPI error detail for clean messages.
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const comp = (await res.json()) as Comparison;
  _comparisonCache.set(key, comp);
  return comp;
}
