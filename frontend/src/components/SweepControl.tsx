import { useState } from "react";
import type { SweepScope } from "../types";

type Props = {
  canSweep: boolean;
  isRunning: boolean;
  currentBlockIdx: number | null; // for the "this layer" option
  onRun: (scope: SweepScope) => void;
};

const HEADS_PER_LAYER = 16; // gpt2-medium
const RANGE_START = 10;
const RANGE_END = 20;
const TOTAL_LAYERS = 24;

export function SweepControl({
  canSweep,
  isRunning,
  currentBlockIdx,
  onRun,
}: Props) {
  const [scope, setScope] = useState<SweepScope>("current_layer");
  const disabled = isRunning || !canSweep;
  const rangeHeads = (RANGE_END - RANGE_START + 1) * HEADS_PER_LAYER;
  const allHeads = TOTAL_LAYERS * HEADS_PER_LAYER;
  const currentLabel =
    currentBlockIdx !== null
      ? `this layer (L${currentBlockIdx}, ${HEADS_PER_LAYER} heads)`
      : `this layer (${HEADS_PER_LAYER} heads)`;

  return (
    <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-neutral-500 select-none">
      <span>Sweep</span>
      <select
        value={scope}
        onChange={(e) => setScope(e.target.value as SweepScope)}
        disabled={disabled}
        className="bg-neutral-900 border border-neutral-800 text-neutral-200 text-[11px] normal-case tracking-normal px-1.5 py-0.5 outline-none focus:border-accent disabled:opacity-40"
      >
        <option value="current_layer">{currentLabel}</option>
        <option value="layer_range">
          layers {RANGE_START}-{RANGE_END} ({rangeHeads} heads)
        </option>
        <option value="all_layers">all layers ({allHeads} heads)</option>
      </select>
      <button
        onClick={() => onRun(scope)}
        disabled={disabled}
        className={
          "px-2.5 py-1 uppercase text-[10px] tracking-wider border transition-colors " +
          (disabled
            ? "border-neutral-800 text-neutral-600 cursor-not-allowed"
            : "border-ablate text-ablate hover:bg-ablate hover:text-neutral-950")
        }
      >
        {isRunning ? "sweeping…" : "run sweep"}
      </button>
      {!canSweep && !isRunning && (
        <span className="text-neutral-600 normal-case tracking-normal">
          exit compare mode to sweep
        </span>
      )}
    </div>
  );
}
