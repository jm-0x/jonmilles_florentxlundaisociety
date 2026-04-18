import { memo, useMemo, useState } from "react";
import { ResponsiveHeatMap } from "@nivo/heatmap";
import type { Trace } from "../types";
import { displayToken } from "../util";
import {
  AMBER,
  CYAN,
  MONO_FAMILY,
  NEAR_BLACK,
  NEUTRAL_100,
  PANEL_BG,
  divergingColor,
  nivoTheme,
} from "../nivoTheme";

type Props = {
  traceA: Trace;
  traceB: Trace;
  selectedLayerIdx: number; // 0 = embedding, 1..n_layers = block outputs
  selectedTokenIdx: number;
};

type HeatDatum = { x: string; y: number };
type HeatRow = { id: string; data: HeatDatum[] };

function matrixToNivo(matrix: number[][], ids: string[]): HeatRow[] {
  return matrix.map((row, r) => ({
    id: ids[r],
    data: row.map((v, c) => ({ x: ids[c], y: v })),
  }));
}

function stripPrefix(s: string | number): string {
  return String(s).replace(/^\d+:/, "");
}

function maxAbs(matrix: number[][]): number {
  let m = 0;
  for (const row of matrix)
    for (const v of row) if (Math.abs(v) > m) m = Math.abs(v);
  return m;
}

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

type MiniDiffProps = {
  matrix: number[][];
  headIdx: number;
  highlightRow: number;
  bosMasked: boolean;
};

const MiniDiff = memo(function MiniDiff({
  matrix,
  headIdx,
  highlightRow,
  bosMasked,
}: MiniDiffProps) {
  const seqLen = matrix.length;
  const data = useMemo(() => {
    const ids = matrix.map((_, i) => String(i));
    return matrixToNivo(matrix, ids);
  }, [matrix]);
  // Per-head symmetric diverging scale — each head rescales independently so
  // internal structure is visible regardless of the head's overall magnitude.
  const scale = useMemo(() => maxAbs(matrix) || 1e-9, [matrix]);

  const BOX = 180;
  const rowH = BOX / seqLen;

  return (
    <div
      className="relative border border-neutral-800 bg-neutral-950"
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
        colors={({ value }) => divergingColor(value ?? 0, scale)}
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
            {String(cell.data.x)} &nbsp; Δ=
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
        style={{
          top: highlightRow * rowH,
          height: rowH,
          boxSizing: "border-box",
        }}
      />
    </div>
  );
});

type MeanDiffProps = {
  matrix: number[][];
  tokensA: string[];
  tokensB: string[];
  highlightRow: number;
  scale: number;
  bosMasked: boolean;
};

const MeanDiff = memo(function MeanDiff({
  matrix,
  tokensA,
  tokensB,
  highlightRow,
  scale,
  bosMasked,
}: MeanDiffProps) {
  const data = useMemo(() => {
    const ids = tokensA.map((t, i) => {
      const a = displayToken(t);
      const b = displayToken(tokensB[i] ?? "");
      return a === b ? `${i}:${a}` : `${i}:${a} / ${b}`;
    });
    return matrixToNivo(matrix, ids);
  }, [matrix, tokensA, tokensB]);

  const MARGIN = { top: 90, right: 24, bottom: 16, left: 160 };
  const WIDTH = 700;
  const HEIGHT = 560;
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const rowH = plotH / tokensA.length;

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
          legendOffset: -140,
          legendPosition: "middle",
        }}
        enableLabels={false}
        colors={({ value }) => divergingColor(value ?? 0, scale)}
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
            Δ = {cell.value !== null ? cell.value.toFixed(4) : "—"}
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

function ColorLegend({ scale }: { scale: number }) {
  const stops = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1];
  return (
    <div className="flex items-center gap-3 text-[10px] text-neutral-500">
      <span style={{ color: CYAN }}>A &lt; B</span>
      <div className="flex border border-neutral-800">
        {stops.map((s, i) => (
          <div
            key={i}
            style={{
              width: 18,
              height: 10,
              backgroundColor: divergingColor(s * scale, scale),
            }}
          />
        ))}
      </div>
      <span style={{ color: AMBER }}>A &gt; B</span>
      <span className="ml-2 tabular-nums">
        scale ±{scale.toFixed(2)}
      </span>
    </div>
  );
}

export function AttentionDiff({
  traceA,
  traceB,
  selectedLayerIdx,
  selectedTokenIdx,
}: Props) {
  if (selectedLayerIdx === 0) {
    return (
      <div className="p-4 text-neutral-500 text-[12px]">
        Attention is computed by transformer blocks. No attention at the
        embedding layer.
      </div>
    );
  }

  const [maskBOS, setMaskBOS] = useState(true);

  const blockIdx = selectedLayerIdx - 1;
  const headsA = traceA.attention_patterns[blockIdx];
  const headsB = traceB.attention_patterns[blockIdx];
  const seqLen = traceA.seq_len;

  // Memoize so re-renders unrelated to the attention data don't recompute these arrays.
  const { diffs, meanDiff } = useMemo(() => {
    const nHeads = headsA.length;
    const diffs: number[][][] = headsA.map((hA, h) =>
      hA.map((row, r) => row.map((v, c) => v - headsB[h][r][c]))
    );
    const meanDiff: number[][] = Array.from({ length: seqLen }, (_, r) =>
      Array.from({ length: seqLen }, (_, c) => {
        let s = 0;
        for (let h = 0; h < nHeads; h++) s += diffs[h][r][c];
        return s / nHeads;
      })
    );
    return { diffs, meanDiff };
  }, [headsA, headsB, seqLen]);

  // Apply mask before computing color scale — BOS's attention-sink spike
  // otherwise dominates and flattens everything else to near-black.
  const displayDiffs = useMemo(
    () => (maskBOS ? diffs.map(withBOSMasked) : diffs),
    [diffs, maskBOS]
  );
  const displayMeanDiff = useMemo(
    () => (maskBOS ? withBOSMasked(meanDiff) : meanDiff),
    [meanDiff, maskBOS]
  );
  const headMaxAbs = useMemo(
    () => displayDiffs.reduce((m, mtx) => Math.max(m, maxAbs(mtx)), 0) || 1e-9,
    [displayDiffs]
  );
  const meanMaxAbs = useMemo(
    () => maxAbs(displayMeanDiff) || 1e-9,
    [displayMeanDiff]
  );

  return (
    <div className="flex flex-col gap-4 p-4 overflow-auto">
      <section className="flex items-center justify-between flex-shrink-0 gap-4 flex-wrap">
        <div className="flex items-center gap-6">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 select-none">
            attention diff (A − B) · layer {blockIdx}
          </div>
          <MaskBOSToggle checked={maskBOS} onChange={setMaskBOS} />
        </div>
        <ColorLegend scale={headMaxAbs} />
      </section>
      <div className="flex flex-wrap gap-12 items-start">
        <section className="flex-shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
            mean Δ across heads (scale ±{meanMaxAbs.toFixed(2)})
          </div>
          <MeanDiff
            matrix={displayMeanDiff}
            tokensA={traceA.tokens}
            tokensB={traceB.tokens}
            highlightRow={selectedTokenIdx}
            scale={meanMaxAbs}
            bosMasked={maskBOS}
          />
        </section>
        <section className="flex-shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
            per-head Δ (each head scaled independently; layer max ±
            {headMaxAbs.toFixed(2)})
          </div>
          <div className="grid grid-cols-4 gap-2 w-fit">
            {displayDiffs.map((m, h) => (
              <MiniDiff
                key={h}
                matrix={m}
                headIdx={h}
                highlightRow={selectedTokenIdx}
                bosMasked={maskBOS}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
