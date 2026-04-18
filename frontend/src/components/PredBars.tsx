import { ResponsiveBar } from "@nivo/bar";
import type { TokenPrediction } from "../types";
import { displayToken } from "../util";
import {
  AMBER,
  MONO_FAMILY,
  NEUTRAL_100,
  PANEL_BG,
  nivoTheme,
} from "../nivoTheme";

type BarDatum = { token: string; prob: number };

type Props = {
  preds: TokenPrediction[];
  maxValue?: number;
  height?: number;
};

export function PredBars({ preds, maxValue, height = 240 }: Props) {
  // Reverse so top-1 ends up visually at the TOP of the horizontal bar chart.
  const data: BarDatum[] = preds
    .slice()
    .reverse()
    .map((p) => ({ token: displayToken(p.token), prob: p.prob }));

  return (
    <div style={{ height }}>
      <ResponsiveBar<BarDatum>
        data={data}
        keys={["prob"]}
        indexBy="token"
        layout="horizontal"
        margin={{ top: 8, right: 70, bottom: 24, left: 160 }}
        padding={0.25}
        valueScale={{ type: "linear", min: 0, max: maxValue ?? "auto" }}
        indexScale={{ type: "band", round: true }}
        colors={[AMBER]}
        colorBy="id"
        borderRadius={0}
        borderWidth={0}
        enableLabel
        label={(d) => (typeof d.value === "number" ? d.value.toFixed(3) : "")}
        labelSkipWidth={28}
        labelTextColor="#0a0a0a"
        axisTop={null}
        axisBottom={null}
        axisRight={null}
        axisLeft={{ tickSize: 0, tickPadding: 8 }}
        enableGridX={false}
        enableGridY={false}
        tooltip={({ indexValue, value }) => (
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
            {String(indexValue)}  {typeof value === "number" ? value.toFixed(4) : value}
          </div>
        )}
        theme={nivoTheme}
        animate={false}
        isInteractive
      />
    </div>
  );
}
