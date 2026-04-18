import type { SweepResultItem } from "../types";
import { displayToken } from "../util";

type Props = {
  results: SweepResultItem[];
  targetToken: string;
  baselineProb: number;
  onRowClick: (item: SweepResultItem) => void;
  onDismiss: () => void;
};

function formatEffect(e: number): string {
  const a = Math.abs(e);
  if (a >= 1) return a.toFixed(1);
  return a.toFixed(2).replace(/^0/, "");
}

export function SweepResultsPanel({
  results,
  targetToken,
  baselineProb,
  onRowClick,
  onDismiss,
}: Props) {
  const maxAbs = results.reduce((m, r) => Math.max(m, Math.abs(r.effect)), 0) || 1;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-neutral-800 bg-neutral-950">
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 select-none">
          Sweep results · tracking
          <span className="text-data normal-case tracking-normal ml-1.5">
            '{displayToken(targetToken)}'
          </span>
          <span className="text-neutral-700 mx-1.5">·</span>
          <span className="text-neutral-400 normal-case tracking-normal">
            baseline {baselineProb.toFixed(3)}
          </span>
          <span className="text-neutral-700 mx-1.5">·</span>
          <span className="normal-case tracking-normal">
            {results.length} ablations
          </span>
        </div>
        <button
          onClick={onDismiss}
          className="px-2.5 py-1 uppercase text-[10px] tracking-wider border border-neutral-700 text-neutral-300 hover:border-accent hover:text-accent transition-colors"
        >
          back to grid
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-[11px] font-mono tabular-nums">
          <thead className="sticky top-0 bg-neutral-950">
            <tr className="text-neutral-600 text-[10px] uppercase tracking-wider">
              <th className="text-left px-3 py-2 font-normal w-12">#</th>
              <th className="text-left px-3 py-2 font-normal w-14">layer</th>
              <th className="text-left px-3 py-2 font-normal w-14">head</th>
              <th className="text-left px-3 py-2 font-normal w-24">effect</th>
              <th className="text-left px-3 py-2 font-normal w-24">
                p(target)
              </th>
              <th className="text-left px-3 py-2 font-normal">magnitude</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => {
              const pct = (Math.abs(r.effect) / maxAbs) * 100;
              const isTop = i === 0;
              const sign = r.effect < 0 ? "↓" : r.effect > 0 ? "↑" : " ";
              return (
                <tr
                  key={`${r.layer}-${r.head}`}
                  onClick={() => onRowClick(r)}
                  className={
                    "cursor-pointer border-l-2 transition-colors " +
                    (isTop
                      ? "border-ablate bg-neutral-900/60 hover:bg-neutral-900"
                      : "border-transparent hover:bg-neutral-900/50")
                  }
                >
                  <td className="px-3 py-1 text-neutral-500">{i + 1}</td>
                  <td className="px-3 py-1 text-neutral-300">L{r.layer}</td>
                  <td className="px-3 py-1 text-neutral-300">H{r.head}</td>
                  <td className="px-3 py-1 text-ablate">
                    {sign} {formatEffect(r.effect)}
                  </td>
                  <td className="px-3 py-1 text-neutral-400">
                    {r.ablated_prob.toFixed(3)}
                  </td>
                  <td className="px-3 py-1">
                    <div className="h-2 bg-neutral-900 border border-neutral-800 max-w-[260px]">
                      <div
                        className="h-full bg-ablate/80"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
