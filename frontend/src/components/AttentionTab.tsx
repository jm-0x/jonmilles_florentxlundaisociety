import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { ResponsiveHeatMap } from "@nivo/heatmap";
import type {
  PatchHeadIntervention,
  Trace,
  ZeroHeadIntervention,
} from "../types";
import { displayToken } from "../util";
import {
  AMBER,
  MONO_FAMILY,
  NEUTRAL_100,
  NEAR_BLACK,
  PANEL_BG,
  amberColor,
  nivoTheme,
} from "../nivoTheme";

export type SourceTrace = { prompt: string; label: string };

type Props = {
  trace: Trace;
  selectedLayerIdx: number;
  selectedTokenIdx: number;
  onAblateHead?: (iv: ZeroHeadIntervention) => void;
  onPatchHead?: (iv: PatchHeadIntervention) => void;
  /** Other open traces with matching token length — valid patch sources. */
  sourceTraces?: SourceTrace[];
};

type HeatDatum = { x: string; y: number };
type HeatRow = { id: string; data: HeatDatum[] };

function matrixToNivo(matrix: number[][], ids: string[]): HeatRow[] {
  return matrix.map((row, r) => ({
    id: ids[r],
    data: row.map((v, c) => ({ x: ids[c], y: v })),
  }));
}

// Zero out the BOS row and column so the color scale for remaining cells
// isn't compressed by the attention-sink spike.
function withBOSMasked(m: number[][]): number[][] {
  return m.map((row, r) => row.map((v, c) => (r === 0 || c === 0 ? 0 : v)));
}

const BOS_TOOLTIP =
  "Attention sink: most heads park significant attention on the BOS token (position 0) as a no-op. Masking it redistributes the color scale so real attention patterns are visible.";

function MaskBOSToggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-neutral-500 select-none">
      <label className="flex items-center gap-1.5 cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="w-3 h-3 cursor-pointer"
          style={{ accentColor: "var(--color-accent)" }}
        />
        <span>Mask BOS</span>
      </label>
      <div className="relative group">
        <span
          className="text-neutral-600 text-[12px] cursor-help leading-none"
          aria-label={BOS_TOOLTIP}
        >
          ⓘ
        </span>
        <div className="hidden group-hover:block absolute left-0 top-full mt-2 w-72 z-50 bg-neutral-900 border border-accent text-neutral-200 text-[11px] normal-case tracking-normal p-2.5 leading-relaxed pointer-events-none">
          {BOS_TOOLTIP}
        </div>
      </div>
    </div>
  );
}

function stripPrefix(s: string | number): string {
  return String(s).replace(/^\d+:/, "");
}

function rowAttnInfo(
  matrix: number[][],
  fromIdx: number
): {
  hasSignificant: boolean;
  maxValue: number;
  maxTo: number;
  bosAttn: number;
} {
  const row = matrix[fromIdx] ?? [];
  const bosAttn = row[0] ?? 0;
  let maxValue = -Infinity;
  let maxTo = -1;
  for (let c = 0; c < row.length; c++) {
    if (c === fromIdx) continue; // exclude self-attention
    if (row[c] > maxValue) {
      maxValue = row[c];
      maxTo = c;
    }
  }
  const hasSignificant = maxTo >= 0 && maxValue > 0;
  return {
    hasSignificant,
    maxValue: hasSignificant ? maxValue : 0,
    maxTo,
    bosAttn,
  };
}

type MiniHeatmapProps = {
  matrix: number[][]; // what gets rendered (may be BOS-masked)
  rawMatrix: number[][]; // original attention, used by the popover's info readout
  headIdx: number;
  blockIdx: number;
  highlightRow: number;
  tokens: string[];
  popoverOpen: boolean;
  bosMasked: boolean;
  sourceTraces: SourceTrace[];
  onToggleOpen: (head: number) => void;
  onAblate?: (head: number) => void;
  onPatch?: (head: number, sourcePrompt: string) => void;
};

const EMPTY_SOURCES: SourceTrace[] = [];

const MiniHeatmap = memo(function MiniHeatmap({
  matrix,
  rawMatrix,
  headIdx,
  blockIdx,
  highlightRow,
  tokens,
  popoverOpen,
  bosMasked,
  sourceTraces,
  onToggleOpen,
  onAblate,
  onPatch,
}: MiniHeatmapProps) {
  const [sourceSel, setSourceSel] = useState<string>("");
  useEffect(() => {
    if (!popoverOpen) setSourceSel("");
  }, [popoverOpen]);
  const seqLen = matrix.length;
  // Stable data reference; only recomputes when the head's matrix changes.
  const data = useMemo(() => {
    const ids = matrix.map((_, i) => String(i));
    return matrixToNivo(matrix, ids);
  }, [matrix]);
  // Per-head max — recomputed when the (possibly-masked) matrix changes.
  // Ensures the brightest non-BOS cell maps to full amber when BOS is masked.
  const maxValue = useMemo(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) if (v > m) m = v;
    return m || 1e-9;
  }, [matrix]);

  const BOX = 180;
  const rowH = BOX / seqLen;

  const rowInGrid = Math.floor(headIdx / 4);
  const colInGrid = headIdx % 4;
  const showAbove = rowInGrid >= 2;
  const alignRight = colInGrid >= 2;

  const popoverPosition: React.CSSProperties = showAbove
    ? alignRight
      ? { bottom: "calc(100% + 6px)", right: 0 }
      : { bottom: "calc(100% + 6px)", left: 0 }
    : alignRight
      ? { top: "calc(100% + 6px)", right: 0 }
      : { top: "calc(100% + 6px)", left: 0 };

  const info = popoverOpen ? rowAttnInfo(rawMatrix, highlightRow) : null;

  return (
    <div
      data-attn-mini
      onClick={(e) => {
        e.stopPropagation();
        onToggleOpen(headIdx);
      }}
      className={
        "relative border bg-neutral-950 cursor-pointer transition-colors " +
        (popoverOpen
          ? "border-accent"
          : "border-neutral-800 hover:border-neutral-600")
      }
      style={{ width: BOX, height: BOX }}
    >
      <div className="absolute top-0.5 left-1 text-[9px] text-neutral-500 z-10 select-none pointer-events-none">
        H{headIdx}
      </div>
      <ResponsiveHeatMap<HeatDatum, object>
        data={data}
        margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
        axisTop={null}
        axisRight={null}
        axisBottom={null}
        axisLeft={null}
        enableLabels={false}
        colors={({ value }) => amberColor((value ?? 0) / maxValue)}
        emptyColor={NEAR_BLACK}
        borderWidth={0}
        theme={nivoTheme}
        animate={false}
        tooltip={({ cell }) => (
          <div
            style={{
              fontFamily: MONO_FAMILY,
              background: PANEL_BG,
              color: NEUTRAL_100,
              border: `1px solid ${AMBER}`,
              padding: "4px 6px",
              fontSize: 10,
            }}
          >
            H{headIdx} &nbsp; from={String(cell.serieId)} &nbsp; to=
            {String(cell.data.x)} &nbsp; w=
            {cell.value !== null ? cell.value.toFixed(3) : "—"}
          </div>
        )}
      />
      {bosMasked && (
        <>
          <div
            className="absolute left-0 top-0 border border-data-muted pointer-events-none"
            style={{ width: BOX, height: rowH, boxSizing: "border-box" }}
          />
          <div
            className="absolute left-0 top-0 border border-data-muted pointer-events-none"
            style={{ width: rowH, height: BOX, boxSizing: "border-box" }}
          />
        </>
      )}
      <div
        className="absolute left-0 right-0 border border-accent pointer-events-none"
        style={{ top: highlightRow * rowH, height: rowH, boxSizing: "border-box" }}
      />
      {popoverOpen && info && (
        <div
          data-attn-popover
          onClick={(e) => e.stopPropagation()}
          className="absolute z-50 w-64 bg-neutral-900 border border-accent text-neutral-100 text-[11px] font-mono shadow-xl"
          style={popoverPosition}
        >
          <div className="px-3 py-2 border-b border-neutral-800 flex items-center justify-between">
            <span>
              <span className="text-neutral-500 uppercase tracking-wider text-[10px] mr-1.5">
                layer
              </span>
              <span className="text-accent">{blockIdx}</span>
              <span className="text-neutral-700 mx-1.5">·</span>
              <span className="text-neutral-500 uppercase tracking-wider text-[10px] mr-1.5">
                head
              </span>
              <span className="text-accent">{headIdx}</span>
            </span>
          </div>
          <div className="px-3 py-3 space-y-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onAblate?.(headIdx);
              }}
              disabled={!onAblate}
              className="w-full border border-accent text-accent px-3 py-1.5 uppercase text-[11px] tracking-wider hover:bg-accent hover:text-neutral-950 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-accent transition-colors"
            >
              ablate (zero out)
            </button>
            <div className="border-t border-neutral-800" />
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-neutral-500">
                patch from
              </div>
              {sourceTraces.length === 0 ? (
                <div className="text-[10px] text-neutral-600 leading-tight">
                  Open another trace with the same token length to enable
                  patching.
                </div>
              ) : (
                <>
                  <select
                    value={sourceSel}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      e.stopPropagation();
                      setSourceSel(e.target.value);
                    }}
                    className="w-full bg-neutral-950 border border-neutral-800 text-neutral-200 text-[11px] px-1.5 py-0.5 outline-none focus:border-patch"
                  >
                    <option value="">— select source trace —</option>
                    {sourceTraces.map((s) => (
                      <option key={s.prompt} value={s.prompt} title={s.prompt}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (sourceSel && onPatch) onPatch(headIdx, sourceSel);
                    }}
                    disabled={!sourceSel || !onPatch}
                    className="w-full border border-patch text-patch px-3 py-1.5 uppercase text-[11px] tracking-wider hover:bg-patch hover:text-neutral-950 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-patch transition-colors"
                  >
                    patch in
                  </button>
                </>
              )}
            </div>
          </div>
          <div className="px-3 pb-3 text-neutral-400 space-y-0.5">
            {info.hasSignificant ? (
              <div className="whitespace-pre-wrap break-words">
                <span className="text-neutral-500">Max from</span> '
                <span className="text-neutral-200">
                  {displayToken(tokens[highlightRow] ?? "")}
                </span>
                ':{" "}
                <span className="text-neutral-200 tabular-nums">
                  {info.maxValue.toFixed(2)}
                </span>{" "}
                → '
                <span className="text-neutral-200">
                  {displayToken(tokens[info.maxTo] ?? "")}
                </span>
                '
              </div>
            ) : (
              <div className="text-neutral-500">
                No significant attention from this token.
              </div>
            )}
            <div>
              <span className="text-neutral-500">BOS attn:</span>{" "}
              <span className="text-neutral-200 tabular-nums">
                {info.bosAttn.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

type MeanHeatmapProps = {
  matrix: number[][];
  tokens: string[];
  highlightRow: number;
  bosMasked: boolean;
};

const MeanHeatmap = memo(function MeanHeatmap({
  matrix,
  tokens,
  highlightRow,
  bosMasked,
}: MeanHeatmapProps) {
  const data = useMemo(() => {
    const ids = tokens.map((t, i) => `${i}:${displayToken(t)}`);
    return matrixToNivo(matrix, ids);
  }, [matrix, tokens]);
  const maxValue = useMemo(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) if (v > m) m = v;
    return m || 1e-9;
  }, [matrix]);

  const MARGIN = { top: 90, right: 24, bottom: 16, left: 140 };
  const WIDTH = 680;
  const HEIGHT = 560;
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const rowH = plotH / tokens.length;

  return (
    <div style={{ width: WIDTH, height: HEIGHT, position: "relative" }}>
      <ResponsiveHeatMap<HeatDatum, object>
        data={data}
        margin={MARGIN}
        axisTop={{
          tickSize: 4,
          tickPadding: 4,
          tickRotation: -45,
          format: stripPrefix,
          legend: "attending to →",
          legendOffset: -70,
          legendPosition: "middle",
        }}
        axisRight={null}
        axisBottom={null}
        axisLeft={{
          tickSize: 4,
          tickPadding: 4,
          tickRotation: 0,
          format: stripPrefix,
          legend: "↓ attending from",
          legendOffset: -120,
          legendPosition: "middle",
        }}
        enableLabels={false}
        colors={({ value }) => amberColor((value ?? 0) / maxValue)}
        emptyColor={NEAR_BLACK}
        borderWidth={0}
        theme={nivoTheme}
        animate={false}
        tooltip={({ cell }) => (
          <div
            style={{
              fontFamily: MONO_FAMILY,
              background: PANEL_BG,
              color: NEUTRAL_100,
              border: `1px solid ${AMBER}`,
              padding: "6px 8px",
              fontSize: 11,
            }}
          >
            from '{stripPrefix(cell.serieId)}' → to '{stripPrefix(String(cell.data.x))}'
            <br />
            w = {cell.value !== null ? cell.value.toFixed(4) : "—"}
          </div>
        )}
      />
      {bosMasked && (
        <>
          <div
            className="absolute border border-data-muted pointer-events-none"
            style={{
              top: MARGIN.top,
              left: MARGIN.left,
              right: MARGIN.right,
              height: rowH,
              boxSizing: "border-box",
            }}
          />
          <div
            className="absolute border border-data-muted pointer-events-none"
            style={{
              top: MARGIN.top,
              left: MARGIN.left,
              width: rowH,
              height: plotH,
              boxSizing: "border-box",
            }}
          />
        </>
      )}
      <div
        className="absolute border border-accent pointer-events-none"
        style={{
          top: MARGIN.top + highlightRow * rowH,
          left: MARGIN.left,
          right: MARGIN.right,
          height: rowH,
          boxSizing: "border-box",
        }}
      />
    </div>
  );
});

export function AttentionTab({
  trace,
  selectedLayerIdx,
  selectedTokenIdx,
  onAblateHead,
  onPatchHead,
  sourceTraces,
}: Props) {
  const [openHeadIdx, setOpenHeadIdx] = useState<number | null>(null);

  // Close popover when layer changes (stale attention data).
  useEffect(() => setOpenHeadIdx(null), [selectedLayerIdx]);

  // Escape closes.
  useEffect(() => {
    if (openHeadIdx === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenHeadIdx(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [openHeadIdx]);

  // Click outside any mini/popover closes.
  useEffect(() => {
    if (openHeadIdx === null) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (target.closest("[data-attn-mini]") || target.closest("[data-attn-popover]")) {
        return;
      }
      setOpenHeadIdx(null);
    };
    // Use capture to beat Nivo's own internal listeners.
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [openHeadIdx]);

  const [maskBOS, setMaskBOS] = useState(true);

  const blockIdx = selectedLayerIdx - 1;
  const heads = blockIdx >= 0 ? trace.attention_patterns[blockIdx] : null;
  const seqLen = trace.seq_len;

  // Memoize mean across heads so it's a stable reference across unrelated re-renders.
  const mean = useMemo(() => {
    if (!heads) return null;
    return Array.from({ length: seqLen }, (_, r) =>
      Array.from({ length: seqLen }, (_, c) => {
        let s = 0;
        for (let h = 0; h < heads.length; h++) s += heads[h][r][c];
        return s / heads.length;
      })
    );
  }, [heads, seqLen]);

  // Masked variants — separate refs so toggling doesn't recompute the clean matrices.
  const displayHeads = useMemo(
    () => (heads ? (maskBOS ? heads.map(withBOSMasked) : heads) : null),
    [heads, maskBOS]
  );
  const displayMean = useMemo(
    () => (mean ? (maskBOS ? withBOSMasked(mean) : mean) : null),
    [mean, maskBOS]
  );

  // Stable callbacks — identity doesn't change every render, so memo'd MiniHeatmaps
  // can skip when their other props haven't changed.
  const handleToggleOpen = useCallback((h: number) => {
    setOpenHeadIdx((prev) => (prev === h ? null : h));
  }, []);

  const handleAblate = useCallback(
    (h: number) => {
      if (!onAblateHead) return;
      onAblateHead({ type: "zero_head", layer: blockIdx, head: h });
      setOpenHeadIdx(null);
    },
    [onAblateHead, blockIdx]
  );

  const handlePatch = useCallback(
    (h: number, sourcePrompt: string) => {
      if (!onPatchHead) return;
      onPatchHead({
        type: "patch_head",
        layer: blockIdx,
        head: h,
        source_prompt: sourcePrompt,
      });
      setOpenHeadIdx(null);
    },
    [onPatchHead, blockIdx]
  );

  const effectiveSources = sourceTraces ?? EMPTY_SOURCES;

  if (
    selectedLayerIdx === 0 ||
    !heads ||
    !mean ||
    !displayHeads ||
    !displayMean
  ) {
    return (
      <div className="p-4 text-neutral-500 text-[12px]">
        Attention is computed by transformer blocks. No attention at the
        embedding layer.
      </div>
    );
  }

  return (
    <div className="p-4 overflow-auto">
      <div className="mb-3">
        <MaskBOSToggle checked={maskBOS} onChange={setMaskBOS} />
      </div>
      <div className="flex flex-wrap gap-12 items-start">
        <section className="flex-shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
            mean across heads (layer {blockIdx})
          </div>
          <MeanHeatmap
            matrix={displayMean}
            tokens={trace.tokens}
            highlightRow={selectedTokenIdx}
            bosMasked={maskBOS}
          />
        </section>
        <section className="flex-shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
            per-head attention (layer {blockIdx})
          </div>
          <div className="grid grid-cols-4 gap-2 w-fit">
            {displayHeads.map((m, h) => (
              <MiniHeatmap
                key={h}
                matrix={m}
                rawMatrix={heads[h]}
                headIdx={h}
                blockIdx={blockIdx}
                highlightRow={selectedTokenIdx}
                tokens={trace.tokens}
                popoverOpen={openHeadIdx === h}
                bosMasked={maskBOS}
                sourceTraces={
                  openHeadIdx === h ? effectiveSources : EMPTY_SOURCES
                }
                onToggleOpen={handleToggleOpen}
                onAblate={onAblateHead ? handleAblate : undefined}
                onPatch={onPatchHead ? handlePatch : undefined}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
