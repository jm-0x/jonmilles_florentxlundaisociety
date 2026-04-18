import { useMemo } from "react";
import type { Trace } from "../types";
import { displayToken, layerLabel } from "../util";

type Props = {
  trace: Trace;
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  onSelect: (idx: number) => void;
};

export function LayerList({
  trace,
  selectedTokenIdx,
  selectedLayerIdx,
  onSelect,
}: Props) {
  const finalTopToken =
    trace.logit_lens_top_k[trace.n_layers]?.[selectedTokenIdx]?.[0]?.token ??
    null;

  // Probability of the final-layer top-1 token at each layer's logit lens.
  // Caveat: we only have the top-5 per layer, so if the target falls out of
  // the top-5 at an early layer we report 0 — honest under-approximation,
  // shown as an empty bar. A proper fix would require the backend to also
  // expose the target token's probability at every layer.
  const probPerLayer = useMemo(() => {
    if (!finalTopToken) return trace.logit_lens_top_k.map(() => 0);
    return trace.logit_lens_top_k.map((layer) => {
      const row = layer[selectedTokenIdx] ?? [];
      const hit = row.find((p) => p.token === finalTopToken);
      return hit ? hit.prob : 0;
    });
  }, [trace, selectedTokenIdx, finalTopToken]);

  const maxProb = useMemo(
    () => probPerLayer.reduce((m, v) => Math.max(m, v), 0) || 1,
    [probPerLayer]
  );

  return (
    <div className="w-72 flex-shrink-0 border-r border-neutral-800 flex flex-col bg-neutral-950">
      <div className="px-3 py-2 border-b border-neutral-800 select-none">
        <div className="text-[10px] uppercase tracking-wider text-neutral-600">
          layers — logit lens @{" "}
          <span className="text-neutral-400 normal-case tracking-normal">
            '{displayToken(trace.tokens[selectedTokenIdx] ?? "")}'
          </span>
        </div>
        {finalTopToken && (
          <div className="text-[10px] text-neutral-600 normal-case tracking-normal mt-0.5">
            sparkline: p('
            <span className="text-data">
              {displayToken(finalTopToken)}
            </span>
            ') across layers
          </div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {trace.logit_lens_top_k.map((layer, layerIdx) => {
          const top1 = layer[selectedTokenIdx]?.[0];
          const selected = layerIdx === selectedLayerIdx;
          const p = probPerLayer[layerIdx] ?? 0;
          const pct = (p / maxProb) * 100;
          return (
            <button
              key={layerIdx}
              onClick={() => onSelect(layerIdx)}
              className={
                "w-full flex items-center gap-2 px-3 py-1 border-l-2 text-left transition-colors tabular-nums " +
                (selected
                  ? "border-accent bg-neutral-900 text-neutral-100"
                  : "border-transparent text-neutral-400 hover:bg-neutral-900/60")
              }
            >
              <span
                className={
                  "w-10 text-[11px] " +
                  (selected ? "text-accent" : "text-neutral-500")
                }
              >
                {layerLabel(layerIdx)}
              </span>
              <span className="flex-1 whitespace-pre overflow-hidden text-ellipsis min-w-0">
                {top1 ? displayToken(top1.token) : "—"}
              </span>
              <span
                className={
                  "text-[11px] w-10 text-right " +
                  (selected ? "text-neutral-300" : "text-neutral-600")
                }
              >
                {top1 ? top1.prob.toFixed(3) : ""}
              </span>
              <span
                className="relative h-1.5 w-[70px] bg-neutral-900 flex-shrink-0"
                title={`p=${p.toFixed(4)}`}
              >
                {p > 0 && (
                  <span
                    className="absolute inset-y-0 left-0 bg-data"
                    style={{ width: `${pct}%` }}
                  />
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
