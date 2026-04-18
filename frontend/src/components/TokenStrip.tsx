import { displayToken } from "../util";

type Props = {
  tokens: string[];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
};

export function TokenStrip({ tokens, selectedIdx, onSelect }: Props) {
  return (
    <div className="border-b border-neutral-800 bg-neutral-950 px-4 py-2">
      <div className="text-neutral-600 text-[10px] uppercase tracking-wider mb-1.5 select-none">
        tokens
      </div>
      <div className="flex gap-1 flex-wrap">
        {tokens.map((tok, i) => {
          const selected = i === selectedIdx;
          return (
            <button
              key={i}
              onClick={() => onSelect(i)}
              className={
                "flex flex-col items-center border px-2 py-1 leading-tight transition-colors " +
                (selected
                  ? "bg-accent text-neutral-950 border-accent"
                  : "bg-neutral-900 text-neutral-300 border-neutral-800 hover:border-neutral-600")
              }
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
