import type {
  AttentionCompareMode,
  DetailTab,
  PatchHeadIntervention,
  Trace,
  ZeroHeadIntervention,
} from "../types";
import { displayToken, layerLabel } from "../util";
import { PredBars } from "./PredBars";
import { AttentionTab, type SourceTrace } from "./AttentionTab";
import { AttentionDiff } from "./AttentionDiff";

type Props = {
  traceA: Trace;
  traceB: Trace;
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  detailTab: DetailTab;
  attentionCompareMode: AttentionCompareMode;
  onTab: (t: DetailTab) => void;
  onAttentionCompareMode: (m: AttentionCompareMode) => void;
  onAblateHeadA?: (iv: ZeroHeadIntervention) => void;
  onAblateHeadB?: (iv: ZeroHeadIntervention) => void;
  onPatchHeadA?: (iv: PatchHeadIntervention) => void;
  onPatchHeadB?: (iv: PatchHeadIntervention) => void;
  sourceTracesA?: SourceTrace[];
  sourceTracesB?: SourceTrace[];
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

function SegButton({
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
        "px-3 py-1 text-[11px] uppercase tracking-wider border transition-colors " +
        (active
          ? "bg-accent text-neutral-950 border-accent"
          : "bg-neutral-900 text-neutral-300 border-neutral-800 hover:border-neutral-600")
      }
    >
      {children}
    </button>
  );
}

function PredictionsSplit({
  traceA,
  traceB,
  selectedTokenIdx,
  selectedLayerIdx,
}: {
  traceA: Trace;
  traceB: Trace;
  selectedTokenIdx: number;
  selectedLayerIdx: number;
}) {
  const lensA = traceA.logit_lens_top_k[selectedLayerIdx]?.[selectedTokenIdx] ?? [];
  const lensB = traceB.logit_lens_top_k[selectedLayerIdx]?.[selectedTokenIdx] ?? [];
  const isFinalLayerA = selectedLayerIdx === traceA.n_layers;
  const isFinalLayerB = selectedLayerIdx === traceB.n_layers;
  const finalA = isFinalLayerA ? traceA.final_top_k[selectedTokenIdx] ?? [] : null;
  const finalB = isFinalLayerB ? traceB.final_top_k[selectedTokenIdx] ?? [] : null;

  // Shared x-axis scale across A and B (use max of their top-5 probs).
  const maxLens = Math.max(
    ...lensA.map((p) => p.prob),
    ...lensB.map((p) => p.prob),
    1e-6
  );
  const maxFinal =
    finalA || finalB
      ? Math.max(
          ...(finalA ?? []).map((p) => p.prob),
          ...(finalB ?? []).map((p) => p.prob),
          1e-6
        )
      : undefined;

  function Section({
    title,
    preds,
    maxValue,
  }: {
    title: string;
    preds: { token: string; prob: number }[];
    maxValue: number;
  }) {
    return (
      <section>
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
          {title}
        </div>
        <PredBars preds={preds} maxValue={maxValue} />
      </section>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-0 overflow-y-auto">
      <div className="p-4 border-r border-neutral-800 flex flex-col gap-5">
        <div className="text-[10px] uppercase tracking-wider text-data select-none">
          Trace A
        </div>
        <Section title="logit-lens top-5" preds={lensA} maxValue={maxLens} />
        {finalA && maxFinal !== undefined && (
          <Section
            title="actual final prediction"
            preds={finalA}
            maxValue={maxFinal}
          />
        )}
      </div>
      <div className="p-4 flex flex-col gap-5">
        <div className="text-[10px] uppercase tracking-wider text-accent select-none">
          Trace B
        </div>
        <Section title="logit-lens top-5" preds={lensB} maxValue={maxLens} />
        {finalB && maxFinal !== undefined && (
          <Section
            title="actual final prediction"
            preds={finalB}
            maxValue={maxFinal}
          />
        )}
      </div>
    </div>
  );
}

function AttentionCompareBody({
  traceA,
  traceB,
  selectedTokenIdx,
  selectedLayerIdx,
  mode,
  onAblateHeadA,
  onAblateHeadB,
  onPatchHeadA,
  onPatchHeadB,
  sourceTracesA,
  sourceTracesB,
}: {
  traceA: Trace;
  traceB: Trace;
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  mode: AttentionCompareMode;
  onAblateHeadA?: (iv: ZeroHeadIntervention) => void;
  onAblateHeadB?: (iv: ZeroHeadIntervention) => void;
  onPatchHeadA?: (iv: PatchHeadIntervention) => void;
  onPatchHeadB?: (iv: PatchHeadIntervention) => void;
  sourceTracesA?: SourceTrace[];
  sourceTracesB?: SourceTrace[];
}) {
  if (mode === "A") {
    return (
      <AttentionTab
        trace={traceA}
        selectedLayerIdx={selectedLayerIdx}
        selectedTokenIdx={selectedTokenIdx}
        onAblateHead={onAblateHeadA}
        onPatchHead={onPatchHeadA}
        sourceTraces={sourceTracesA}
      />
    );
  }
  if (mode === "B") {
    return (
      <AttentionTab
        trace={traceB}
        selectedLayerIdx={selectedLayerIdx}
        selectedTokenIdx={selectedTokenIdx}
        onAblateHead={onAblateHeadB}
        onPatchHead={onPatchHeadB}
        sourceTraces={sourceTracesB}
      />
    );
  }
  return (
    <AttentionDiff
      traceA={traceA}
      traceB={traceB}
      selectedLayerIdx={selectedLayerIdx}
      selectedTokenIdx={selectedTokenIdx}
    />
  );
}

export function CompareDetailPanel({
  traceA,
  traceB,
  selectedTokenIdx,
  selectedLayerIdx,
  detailTab,
  attentionCompareMode,
  onTab,
  onAttentionCompareMode,
  onAblateHeadA,
  onAblateHeadB,
  onPatchHeadA,
  onPatchHeadB,
  sourceTracesA,
  sourceTracesB,
}: Props) {
  const tokenLabelA = displayToken(traceA.tokens[selectedTokenIdx] ?? "");
  const tokenLabelB = displayToken(traceB.tokens[selectedTokenIdx] ?? "");
  const tokenHeader =
    tokenLabelA === tokenLabelB
      ? `'${tokenLabelA}'`
      : `A:'${tokenLabelA}'  B:'${tokenLabelB}'`;

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-neutral-950">
      <div className="px-4 py-2 border-b border-neutral-800 flex items-baseline gap-3 flex-wrap">
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider">
          layer
        </span>
        <span className="text-accent">{layerLabel(selectedLayerIdx)}</span>
        <span className="text-neutral-700">·</span>
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider">
          token
        </span>
        <span className="text-neutral-100 whitespace-pre">{tokenHeader}</span>
        <span className="text-neutral-600 text-[11px]">
          [pos {selectedTokenIdx}]
        </span>
      </div>
      <div className="flex border-b border-neutral-800 px-2">
        <TabButton
          active={detailTab === "predictions"}
          onClick={() => onTab("predictions")}
        >
          predictions
        </TabButton>
        <TabButton
          active={detailTab === "attention"}
          onClick={() => onTab("attention")}
        >
          attention
        </TabButton>
      </div>
      {detailTab === "attention" && (
        <div className="px-4 py-2 border-b border-neutral-800 flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 mr-1">
            view
          </span>
          <SegButton
            active={attentionCompareMode === "A"}
            onClick={() => onAttentionCompareMode("A")}
          >
            A only
          </SegButton>
          <SegButton
            active={attentionCompareMode === "B"}
            onClick={() => onAttentionCompareMode("B")}
          >
            B only
          </SegButton>
          <SegButton
            active={attentionCompareMode === "diff"}
            onClick={() => onAttentionCompareMode("diff")}
          >
            diff
          </SegButton>
        </div>
      )}
      <div className="flex-1 min-h-0">
        {detailTab === "predictions" ? (
          <PredictionsSplit
            traceA={traceA}
            traceB={traceB}
            selectedTokenIdx={selectedTokenIdx}
            selectedLayerIdx={selectedLayerIdx}
          />
        ) : (
          <AttentionCompareBody
            traceA={traceA}
            traceB={traceB}
            selectedTokenIdx={selectedTokenIdx}
            selectedLayerIdx={selectedLayerIdx}
            mode={attentionCompareMode}
            onAblateHeadA={onAblateHeadA}
            onAblateHeadB={onAblateHeadB}
            onPatchHeadA={onPatchHeadA}
            onPatchHeadB={onPatchHeadB}
            sourceTracesA={sourceTracesA}
            sourceTracesB={sourceTracesB}
          />
        )}
      </div>
    </div>
  );
}
