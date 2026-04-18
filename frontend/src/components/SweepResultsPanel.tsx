import type { SweepResultItem } from "../types";
import { displayToken } from "../util";

type Props = {
  results: SweepResultItem[];
  targetToken: string;
  baselineProb: number;
  onRowClick: (item: SweepResultItem) => void;
  onDismiss: () => void;
};

const ZERO_EPSILON = 0.005;

type Direction = "neg" | "pos" | "zero";

function directionOf(effect: number): Direction {
  if (Math.abs(effect) < ZERO_EPSILON) return "zero";
  return effect < 0 ? "neg" : "pos";
}

function arrowOf(dir: Direction): string {
  if (dir === "neg") return "↓";
  if (dir === "pos") return "↑";
  return "·";
}

function textClassOf(dir: Direction): string {
  if (dir === "neg") return "text-ablate";
  if (dir === "pos") return "text-suppress";
  return "text-neutral-500";
}

function barClassOf(dir: Direction): string {
  if (dir === "neg") return "bg-ablate/80";
  if (dir === "pos") return "bg-suppress/80";
  return "bg-neutral-700";
}

function borderClassOf(dir: Direction): string {
  if (dir === "neg") return "border-ablate";
  if (dir === "pos") return "border-suppress";
  return "border-neutral-700";
}

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
  const topDir = results.length > 0 ? directionOf(results[0].effect) : "zero";

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
      <div className="px-4 py-1.5 border-b border-neutral-800 text-[10px] text-neutral-500 font-mono tracking-normal select-none flex gap-6 flex-wrap">
        <span>
          <span className="text-ablate">↓ red</span> = necessary (ablating hurts
          prediction)
        </span>
        <span>
          <span className="text-suppress">↑ green</span> = suppressive (ablating
          helps prediction)
        </span>
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
              const dir = directionOf(r.effect);
              const isTop = i === 0;
              return (
                <tr
                  key={`${r.layer}-${r.head}`}
                  onClick={() => onRowClick(r)}
                  className={
                    "cursor-pointer border-l-2 transition-colors " +
                    (isTop
                      ? `${borderClassOf(topDir)} bg-neutral-900/60 hover:bg-neutral-900`
                      : "border-transparent hover:bg-neutral-900/50")
                  }
                >
                  <td className="px-3 py-1 text-neutral-500">{i + 1}</td>
                  <td className="px-3 py-1 text-neutral-300">L{r.layer}</td>
                  <td className="px-3 py-1 text-neutral-300">H{r.head}</td>
                  <td className={"px-3 py-1 " + textClassOf(dir)}>
                    {arrowOf(dir)} {formatEffect(r.effect)}
                  </td>
                  <td className="px-3 py-1 text-neutral-400">
                    {r.ablated_prob.toFixed(3)}
                  </td>
                  <td className="px-3 py-1">
                    <div className="h-2 bg-neutral-900 border border-neutral-800 max-w-[260px]">
                      <div
                        className={"h-full " + barClassOf(dir)}
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
