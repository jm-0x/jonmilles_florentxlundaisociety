import type {
  PatchHeadIntervention,
  Trace,
  ZeroHeadIntervention,
} from "../types";
import { displayToken, layerLabel } from "../util";
import { PredictionsTab } from "./PredictionsTab";
import { AttentionTab, type SourceTrace } from "./AttentionTab";

type DetailTab = "predictions" | "attention";

type Props = {
  trace: Trace;
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onAblateHead?: (iv: ZeroHeadIntervention) => void;
  onPatchHead?: (iv: PatchHeadIntervention) => void;
  sourceTraces?: SourceTrace[];
};

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "px-3 py-1.5 border-b-2 text-[11px] uppercase tracking-wider transition-colors " +
        (active
          ? "border-accent text-accent"
          : "border-transparent text-neutral-500 hover:text-neutral-300")
      }
    >
      {children}
    </button>
  );
}

export function DetailPanel({
  trace,
  selectedTokenIdx,
  selectedLayerIdx,
  tab,
  onTab,
  onAblateHead,
  onPatchHead,
  sourceTraces,
}: Props) {
  const lensPreds =
    trace.logit_lens_top_k[selectedLayerIdx][selectedTokenIdx] ?? [];
  const isFinalLayer = selectedLayerIdx === trace.n_layers;
  const finalPreds = isFinalLayer
    ? trace.final_top_k[selectedTokenIdx] ?? []
    : null;

  const tokenStr = trace.tokens[selectedTokenIdx] ?? "";

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-neutral-950">
      <div className="px-4 py-2 border-b border-neutral-800 flex items-baseline gap-3">
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider">
          layer
        </span>
        <span className="text-accent">{layerLabel(selectedLayerIdx)}</span>
        <span className="text-neutral-700">·</span>
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider">
          token
        </span>
        <span className="text-neutral-100 whitespace-pre">
          '{displayToken(tokenStr)}'
        </span>
        <span className="text-neutral-600 text-[11px]">
          [pos {selectedTokenIdx}]
        </span>
      </div>
      <div className="flex border-b border-neutral-800 px-2">
        <TabButton
          active={tab === "predictions"}
          onClick={() => onTab("predictions")}
        >
          predictions
        </TabButton>
        <TabButton
          active={tab === "attention"}
          onClick={() => onTab("attention")}
        >
          attention
        </TabButton>
      </div>
      <div className="flex-1 min-h-0">
        {tab === "predictions" ? (
          <PredictionsTab lensPreds={lensPreds} finalPreds={finalPreds} />
        ) : (
          <AttentionTab
            trace={trace}
            selectedLayerIdx={selectedLayerIdx}
            selectedTokenIdx={selectedTokenIdx}
            onAblateHead={onAblateHead}
            onPatchHead={onPatchHead}
            sourceTraces={sourceTraces}
          />
        )}
      </div>
    </div>
  );
}
