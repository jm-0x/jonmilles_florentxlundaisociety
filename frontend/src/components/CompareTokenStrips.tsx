import type { Trace } from "../types";
import { displayToken } from "../util";

type Props = {
  traceA: Trace;
  traceB: Trace;
  selectedIdx: number;
  onSelect: (idx: number) => void;
};

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function StripRow({
  label,
  prompt,
  tokens,
  otherTokens,
  selectedIdx,
  onSelect,
  labelColor,
}: {
  label: string;
  prompt: string;
  tokens: string[];
  otherTokens: string[];
  selectedIdx: number;
  onSelect: (i: number) => void;
  labelColor: string;
}) {
  return (
    <div className="flex items-start gap-3 px-4 py-1.5">
      <div
        className={
          "pt-1 w-20 flex-shrink-0 text-[10px] uppercase tracking-wider select-none " +
          labelColor
        }
        title={prompt}
      >
        {label}
        <div className="normal-case tracking-normal text-neutral-500 text-[10px] mt-0.5 font-normal">
          {truncate(prompt, 28)}
        </div>
      </div>
      <div className="flex gap-1 flex-wrap min-w-0">
        {tokens.map((tok, i) => {
          const selected = i === selectedIdx;
          const differs = otherTokens[i] !== undefined && otherTokens[i] !== tok;
          const border = selected
            ? "border-accent"
            : differs
              ? "border-data/70"
              : "border-neutral-800";
          const bg = selected
            ? "bg-accent text-neutral-950"
            : "bg-neutral-900 text-neutral-300 hover:border-neutral-600";
          return (
            <button
              key={i}
              onClick={() => onSelect(i)}
              className={`flex flex-col items-center border px-2 py-1 leading-tight transition-colors ${bg} ${border}`}
            >
              <span className="whitespace-pre">{displayToken(tok)}</span>
              <span
                className={
                  "text-[9px] " +
                  (selected ? "text-neutral-800" : "text-neutral-600")
                }
              >
                {i}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function CompareTokenStrips({
  traceA,
  traceB,
  selectedIdx,
  onSelect,
}: Props) {
  return (
    <div className="border-b border-neutral-800 bg-neutral-950">
      <StripRow
        label="Trace A"
        labelColor="text-data"
        prompt={traceA.prompt}
        tokens={traceA.tokens}
        otherTokens={traceB.tokens}
        selectedIdx={selectedIdx}
        onSelect={onSelect}
      />
      <div className="h-px bg-neutral-800" />
      <StripRow
        label="Trace B"
        labelColor="text-accent"
        prompt={traceB.prompt}
        tokens={traceB.tokens}
        otherTokens={traceA.tokens}
        selectedIdx={selectedIdx}
        onSelect={onSelect}
      />
    </div>
  );
}
