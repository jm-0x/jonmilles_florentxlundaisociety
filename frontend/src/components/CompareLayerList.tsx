import type { Comparison } from "../types";
import { layerLabel } from "../util";

type Props = {
  comparison: Comparison;
  selectedLayerIdx: number;
  onSelect: (idx: number) => void;
};

export function CompareLayerList({
  comparison,
  selectedLayerIdx,
  onSelect,
}: Props) {
  const { kl_symmetric } = comparison;
  const max = kl_symmetric.reduce((m, v) => Math.max(m, v), 0) || 1;

  return (
    <div className="w-64 flex-shrink-0 border-r border-neutral-800 flex flex-col bg-neutral-950">
      <div className="px-3 py-2 border-b border-neutral-800 text-[10px] uppercase tracking-wider text-neutral-600 select-none">
        divergence (kl) by layer
      </div>
      <div className="flex-1 overflow-y-auto">
        {kl_symmetric.map((kl, layerIdx) => {
          const selected = layerIdx === selectedLayerIdx;
          const pct = (kl / max) * 100;
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
              <span
                className={
                  "w-20 text-[11px] " +
                  (selected ? "text-neutral-200" : "text-neutral-500")
                }
              >
                KL={kl.toFixed(2)}
              </span>
              <span className="flex-1 h-2.5 bg-neutral-900 border border-neutral-800 relative">
                <span
                  className="absolute inset-y-0 left-0 bg-data/80"
                  style={{ width: `${pct}%` }}
                />
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
