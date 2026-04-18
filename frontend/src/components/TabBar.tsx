import type { Intervention, Trace, TraceMetadata } from "../types";
import {
  formatMagnitude,
  interventionBadge,
  sortedInterventions,
  traceAccent,
  traceLabel,
} from "../interventions";
import { displayToken } from "../util";
import { INTERVENTION } from "../theme";

type Props = {
  tabOrder: string[];
  activeTabId: string | null;
  traces: Record<string, Trace>;
  traceMetadata: Record<string, TraceMetadata>;
  onActivate: (id: string) => void;
  onClose: (id: string) => void;
  onAddTab: () => void;
  compareMode: boolean;
  compareLeftId: string | null;
  compareRightId: string | null;
  onCompareToggle: () => void;
  onCompareSelect: (side: "left" | "right", id: string | null) => void;
};

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function arrowFor(direction: "down" | "up"): string {
  return direction === "down" ? "↓" : "↑";
}

function accentClass(accent: "none" | "ablate" | "patch"): string {
  if (accent === "ablate") return "text-ablate";
  if (accent === "patch") return "text-patch";
  return "";
}

function latestInterventionLabel(iv: Intervention): string {
  if (iv.type === "zero_head") return `L${iv.layer} H${iv.head}`;
  if (iv.type === "zero_mlp") return `L${iv.layer} MLP`;
  return `L${iv.layer} H${iv.head}`;
}

function Tab({
  id,
  trace,
  metadata,
  active,
  accent,
  onActivate,
  onClose,
  traces,
}: {
  id: string;
  trace: Trace;
  metadata: TraceMetadata | undefined;
  active: boolean;
  accent: "none" | "ablate" | "patch";
  onActivate: (id: string) => void;
  onClose: (id: string) => void;
  traces: Record<string, Trace>;
}) {
  const stripeColor =
    accent === "patch"
      ? INTERVENTION.patch
      : accent === "ablate"
        ? INTERVENTION.ablate
        : null;

  const interventions = trace.interventions_applied ?? [];
  const hasInterventions = interventions.length > 0;
  const effect = metadata?.effect;

  const badgesText = hasInterventions
    ? interventions.map(interventionBadge).join(", ")
    : "";
  const effectText = effect
    ? ` ${arrowFor(effect.direction)}${formatMagnitude(effect.magnitude)}`
    : "";
  // Keep [ ... ] under ~20 chars; if chained-interventions blow it out, keep
  // only the most recent badge + effect.
  const fullBadge = hasInterventions ? `[${badgesText}${effectText}]` : "";
  const badgeTrimmed =
    fullBadge.length > 22 && interventions.length > 1
      ? `[…, ${interventionBadge(interventions[interventions.length - 1])}${effectText}]`
      : fullBadge;

  // Prompt part, truncated.
  const promptPart = truncate(trace.prompt, 30);

  // Tooltip content.
  const base = metadata?.baseTraceId ? traces[metadata.baseTraceId] : null;
  const lastIv =
    interventions.length > 0 ? interventions[interventions.length - 1] : null;
  let tooltipLines: string[] | null = null;
  if (lastIv && effect && base) {
    const trackedTok = effect.trackedToken;
    const baseProb =
      base.final_top_k[base.seq_len - 1].find((p) => p.token === trackedTok)
        ?.prob ?? 0;
    const intrProb =
      trace.final_top_k[trace.seq_len - 1].find((p) => p.token === trackedTok)
        ?.prob ?? 0;
    const arrow = arrowFor(effect.direction);
    const magAbs = Math.abs(effect.magnitude).toFixed(3);
    const probLine = `  p("${displayToken(trackedTok)}"): ${baseProb.toFixed(3)} → ${intrProb.toFixed(3)} (${arrow}${magAbs})`;

    if (lastIv.type === "patch_head") {
      tooltipLines = [
        `[Patch] ${latestInterventionLabel(lastIv)} from "${truncate(lastIv.source_prompt, 40)}"`,
        probLine,
        `  Base: "${truncate(base.prompt, 50)}"`,
      ];
    } else {
      tooltipLines = [
        `[Ablation] ${latestInterventionLabel(lastIv)}`,
        probLine,
        `  Base: "${truncate(base.prompt, 50)}"`,
      ];
    }
  }

  return (
    <div
      onClick={() => onActivate(id)}
      className={
        "group/tab relative flex items-center gap-2 pl-3 pr-1.5 py-1 border-r border-neutral-800 cursor-pointer select-none " +
        (active
          ? "bg-neutral-900 text-neutral-100 border-t-2 border-t-accent -mt-[2px]"
          : "text-neutral-500 hover:text-neutral-200 hover:bg-neutral-900/50 border-t-2 border-t-transparent -mt-[2px]")
      }
      style={
        stripeColor ? { boxShadow: `inset 2px 0 0 0 ${stripeColor}` } : undefined
      }
    >
      <span className="text-[11px] whitespace-nowrap">
        {hasInterventions ? (
          <>
            <span>
              {/* Split badge: neutral bracket+list, colored arrow+magnitude */}
              {badgeTrimmed && (() => {
                const effectIdx = effectText
                  ? badgeTrimmed.lastIndexOf(effectText)
                  : -1;
                if (effectIdx === -1) return <span>{badgeTrimmed}</span>;
                const pre = badgeTrimmed.slice(0, effectIdx);
                const post = badgeTrimmed.slice(effectIdx + effectText.length);
                return (
                  <>
                    <span>{pre}</span>
                    <span className={accentClass(accent)}>{effectText}</span>
                    <span>{post}</span>
                  </>
                );
              })()}
            </span>
            <span> {promptPart}</span>
          </>
        ) : (
          <span>{promptPart}</span>
        )}
      </span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClose(id);
        }}
        className={
          "w-4 h-4 flex items-center justify-center leading-none text-[12px] transition-colors " +
          (active
            ? "text-neutral-500 hover:text-accent"
            : "text-neutral-700 group-hover/tab:text-neutral-500 hover:!text-accent")
        }
        title="close tab"
      >
        ×
      </button>
      {tooltipLines && (
        <div
          className="hidden group-hover/tab:block absolute left-0 top-full mt-1 w-[440px] max-w-[80vw] z-50 bg-neutral-900 border border-neutral-700 text-neutral-200 text-[11px] font-mono normal-case tracking-normal p-2.5 leading-relaxed pointer-events-none whitespace-pre"
          style={{ borderColor: stripeColor ?? undefined }}
        >
          {tooltipLines.join("\n")}
        </div>
      )}
    </div>
  );
}

export function TabBar({
  tabOrder,
  activeTabId,
  traces,
  traceMetadata,
  onActivate,
  onClose,
  onAddTab,
  compareMode,
  compareLeftId,
  compareRightId,
  onCompareToggle,
  onCompareSelect,
}: Props) {
  void sortedInterventions; // keep import used if future expansion; no-op
  return (
    <div className="flex items-stretch border-b border-neutral-800 bg-neutral-950 min-h-[28px]">
      <div className="flex items-stretch overflow-x-auto">
        {tabOrder.map((id) => {
          const t = traces[id];
          if (!t) return null;
          return (
            <Tab
              key={id}
              id={id}
              trace={t}
              metadata={traceMetadata[id]}
              active={id === activeTabId}
              accent={traceAccent(t)}
              onActivate={onActivate}
              onClose={onClose}
              traces={traces}
            />
          );
        })}
        <button
          onClick={onAddTab}
          className="px-3 text-neutral-500 hover:text-accent text-[14px] leading-none select-none"
          title="new trace (focus prompt)"
        >
          +
        </button>
      </div>

      <div className="ml-auto flex items-center gap-2 px-3 py-1">
        <button
          onClick={onCompareToggle}
          className={
            "px-2.5 py-1 uppercase text-[10px] tracking-wider border transition-colors " +
            (compareMode
              ? "bg-accent text-neutral-950 border-accent"
              : "border-accent text-accent hover:bg-accent/10")
          }
          title="toggle compare mode"
        >
          compare
        </button>
        <label className="text-[10px] text-neutral-500 select-none">A</label>
        <select
          value={compareLeftId ?? ""}
          onChange={(e) =>
            onCompareSelect("left", e.target.value ? e.target.value : null)
          }
          title={
            compareLeftId ? traces[compareLeftId]?.prompt ?? undefined : undefined
          }
          className="bg-neutral-900 border border-neutral-800 text-neutral-200 text-[11px] px-1.5 py-0.5 outline-none focus:border-accent w-[240px] truncate"
        >
          <option value="">—</option>
          {tabOrder.map((id) => {
            const t = traces[id];
            const label = t ? traceLabel(t) : id;
            return (
              <option key={id} value={id} title={label}>
                {label}
              </option>
            );
          })}
        </select>
        <label className="text-[10px] text-neutral-500 select-none">B</label>
        <select
          value={compareRightId ?? ""}
          onChange={(e) =>
            onCompareSelect("right", e.target.value ? e.target.value : null)
          }
          title={
            compareRightId
              ? traces[compareRightId]?.prompt ?? undefined
              : undefined
          }
          className="bg-neutral-900 border border-neutral-800 text-neutral-200 text-[11px] px-1.5 py-0.5 outline-none focus:border-accent w-[240px] truncate"
        >
          <option value="">—</option>
          {tabOrder.map((id) => {
            const t = traces[id];
            const label = t ? traceLabel(t) : id;
            return (
              <option key={id} value={id} title={label}>
                {label}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}
