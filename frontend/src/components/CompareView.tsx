import type {
  AttentionCompareMode,
  Comparison,
  CompareViewState,
  DetailTab,
  PatchHeadIntervention,
  Trace,
  ZeroHeadIntervention,
} from "../types";
import { CompareTokenStrips } from "./CompareTokenStrips";
import { CompareLayerList } from "./CompareLayerList";
import { CompareDetailPanel } from "./CompareDetailPanel";
import type { SourceTrace } from "./AttentionTab";

type Props = {
  traceA: Trace;
  traceB: Trace;
  comparison: Comparison | null;
  loading: boolean;
  error: string | null;
  compareViewState: CompareViewState;
  attentionCompareMode: AttentionCompareMode;
  onUpdateView: (patch: Partial<CompareViewState>) => void;
  onAttentionCompareMode: (m: AttentionCompareMode) => void;
  onAblateHeadA?: (iv: ZeroHeadIntervention) => void;
  onAblateHeadB?: (iv: ZeroHeadIntervention) => void;
  onPatchHeadA?: (iv: PatchHeadIntervention) => void;
  onPatchHeadB?: (iv: PatchHeadIntervention) => void;
  sourceTracesA?: SourceTrace[];
  sourceTracesB?: SourceTrace[];
};

export function CompareView({
  traceA,
  traceB,
  comparison,
  loading,
  error,
  compareViewState,
  attentionCompareMode,
  onUpdateView,
  onAttentionCompareMode,
  onAblateHeadA,
  onAblateHeadB,
  onPatchHeadA,
  onPatchHeadB,
  sourceTracesA,
  sourceTracesB,
}: Props) {
  const { selectedTokenIdx, selectedLayerIdx, detailTab } = compareViewState;

  return (
    <>
      <CompareTokenStrips
        traceA={traceA}
        traceB={traceB}
        selectedIdx={selectedTokenIdx}
        onSelect={(idx) => onUpdateView({ selectedTokenIdx: idx })}
      />

      {error && (
        <div className="px-4 py-2 border-b border-red-900/50 bg-red-950/40 text-red-300 text-[12px]">
          {error}
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        {loading || !comparison ? (
          <>
            <div className="w-64 flex-shrink-0 border-r border-neutral-800 flex items-center justify-center bg-neutral-950 text-neutral-500 text-[11px] animate-pulse">
              computing divergence...
            </div>
            <div className="flex-1 flex items-center justify-center text-neutral-500 text-[12px] animate-pulse">
              computing divergence...
            </div>
          </>
        ) : (
          <>
            <CompareLayerList
              comparison={comparison}
              selectedLayerIdx={selectedLayerIdx}
              onSelect={(idx) => onUpdateView({ selectedLayerIdx: idx })}
            />
            <CompareDetailPanel
              traceA={traceA}
              traceB={traceB}
              selectedTokenIdx={selectedTokenIdx}
              selectedLayerIdx={selectedLayerIdx}
              detailTab={detailTab}
              attentionCompareMode={attentionCompareMode}
              onTab={(t: DetailTab) => onUpdateView({ detailTab: t })}
              onAttentionCompareMode={onAttentionCompareMode}
              onAblateHeadA={onAblateHeadA}
              onAblateHeadB={onAblateHeadB}
              onPatchHeadA={onPatchHeadA}
              onPatchHeadB={onPatchHeadB}
              sourceTracesA={sourceTracesA}
              sourceTracesB={sourceTracesB}
            />
          </>
        )}
      </div>
    </>
  );
}
