import { useEffect, useState } from "react";
import type { RefObject } from "react";

type Props = {
  initialPrompt: string;
  onRun: (prompt: string) => void;
  loading: boolean;
  device: string | null;
  inputRef?: RefObject<HTMLInputElement | null>;
  /** Bumped by the parent to imperatively clear the draft input. */
  clearSignal?: number;
};

export function TopBar({
  initialPrompt,
  onRun,
  loading,
  device,
  inputRef,
  clearSignal = 0,
}: Props) {
  // Draft lives here so keystrokes don't re-render App (and 17+ heatmaps).
  const [draft, setDraft] = useState(initialPrompt);

  useEffect(() => {
    if (clearSignal > 0) setDraft("");
  }, [clearSignal]);

  return (
    <div className="flex items-center gap-4 px-4 py-2 border-b border-neutral-800 bg-neutral-950">
      <div className="text-neutral-400 tracking-wider text-[11px] uppercase select-none">
        neural trace
      </div>
      <input
        ref={inputRef}
        className="flex-1 bg-neutral-900 border border-neutral-800 px-3 py-1.5 text-neutral-100 placeholder-neutral-600 outline-none focus:border-accent caret-accent"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !loading) onRun(draft);
        }}
        placeholder="enter a prompt and press enter"
        spellCheck={false}
        autoCorrect="off"
        autoCapitalize="off"
      />
      <button
        onClick={() => !loading && onRun(draft)}
        disabled={loading}
        className="border border-accent text-accent px-3 py-1.5 uppercase text-[11px] tracking-wider hover:bg-accent hover:text-neutral-950 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-accent transition-colors"
      >
        {loading ? "running..." : "run"}
      </button>
      <div className="text-neutral-500 text-[11px] select-none">
        gpt2-medium / {device ?? "..."}
      </div>
    </div>
  );
}
