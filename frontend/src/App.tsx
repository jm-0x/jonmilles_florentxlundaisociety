import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import type {
  AttentionCompareMode,
  Comparison,
  CompareViewState,
  DetailTab,
  Intervention,
  PatchHeadIntervention,
  SweepResultItem,
  SweepScope,
  SweepState,
  Trace,
  TraceMetadata,
  TraceViewState,
  ZeroHeadIntervention,
} from "./types";
import {
  comparisonKey,
  fetchComparison,
  fetchTrace,
  getCachedComparison,
  runSweep,
} from "./api";
import { traceId } from "./interventions";
import type { SourceTrace } from "./components/AttentionTab";
import { TopBar } from "./components/TopBar";
import { TabBar } from "./components/TabBar";
import { TokenStrip } from "./components/TokenStrip";
import { LayerList } from "./components/LayerList";
import { DetailPanel } from "./components/DetailPanel";
import { CompareView } from "./components/CompareView";

const DEFAULT_PROMPT = "The capital of France is the city of";
const BACKEND = "http://localhost:8000";

function defaultViewState(trace: Trace): TraceViewState {
  return {
    selectedTokenIdx: trace.seq_len - 1,
    selectedLayerIdx: trace.n_layers,
    detailTab: "predictions",
  };
}

function defaultCompareViewState(traceA: Trace): CompareViewState {
  return {
    selectedTokenIdx: traceA.seq_len - 1,
    selectedLayerIdx: traceA.n_layers,
    detailTab: "predictions",
  };
}

function App() {
  const [traces, setTraces] = useState<Record<string, Trace>>({});
  const [viewStates, setViewStates] = useState<Record<string, TraceViewState>>({});
  const [traceMetadata, setTraceMetadata] = useState<
    Record<string, TraceMetadata>
  >({});
  const [sweepState, setSweepState] = useState<Record<string, SweepState>>({});
  const [tabOrder, setTabOrder] = useState<string[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);

  const [compareMode, setCompareMode] = useState(false);
  const [compareLeftId, setCompareLeftId] = useState<string | null>(null);
  const [compareRightId, setCompareRightId] = useState<string | null>(null);
  const [comparisons, setComparisons] = useState<Record<string, Comparison>>({});
  const [compareViewState, setCompareViewState] = useState<CompareViewState | null>(
    null
  );
  const [attentionCompareMode, setAttentionCompareMode] =
    useState<AttentionCompareMode>("A");
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [device, setDevice] = useState<string | null>(null);

  const [clearSignal, setClearSignal] = useState(0);
  const promptInputRef = useRef<HTMLInputElement>(null);
  const bootRef = useRef(false);

  async function openTrace(
    prompt: string,
    interventions: Intervention[]
  ): Promise<{ id: string; trace: Trace } | null> {
    const id = traceId(prompt, interventions);
    const existing = traces[id];
    if (existing) {
      setActiveTabId(id);
      return { id, trace: existing };
    }
    setLoading(true);
    setError(null);
    try {
      const trace = await fetchTrace(prompt, interventions);
      setTraces((prev) => ({ ...prev, [id]: trace }));
      setViewStates((prev) => ({ ...prev, [id]: defaultViewState(trace) }));
      setTabOrder((prev) => [...prev, id]);
      setActiveTabId(id);
      return { id, trace };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setLoading(false);
    }
  }

  async function runPrompt(raw: string) {
    const prompt = raw.trim();
    if (!prompt) return;
    await openTrace(prompt, []);
  }

  async function ablateHead(
    originalTraceId: string,
    newIntervention: ZeroHeadIntervention
  ) {
    const base = traces[originalTraceId];
    if (!base) return;
    const allInterventions: Intervention[] = [
      ...(base.interventions_applied ?? []),
      newIntervention,
    ];
    const result = await openTrace(base.prompt, allInterventions);
    if (!result) return;
    const { id: newId, trace: intervened } = result;

    // Effect: how much did the base's top-1 token's probability drop?
    const lastPosBase = base.seq_len - 1;
    const lastPosIntr = intervened.seq_len - 1;
    const baseTop = base.final_top_k[lastPosBase][0];
    const trackedToken = baseTop.token;
    const intrProb =
      intervened.final_top_k[lastPosIntr].find(
        (p) => p.token === trackedToken
      )?.prob ?? 0;
    const magnitude = intrProb - baseTop.prob;

    setTraceMetadata((prev) => ({
      ...prev,
      [newId]: {
        baseTraceId: originalTraceId,
        effect: { magnitude, trackedToken, direction: "down" },
      },
    }));

    setCompareLeftId(originalTraceId);
    setCompareRightId(newId);
    setCompareMode(true);
    setCompareError(null);
  }

  async function patchHead(
    originalTraceId: string,
    newIntervention: PatchHeadIntervention
  ) {
    const base = traces[originalTraceId];
    if (!base) return;
    const allInterventions: Intervention[] = [
      ...(base.interventions_applied ?? []),
      newIntervention,
    ];

    // Source trace lookup (for the tracked token) — sourceId may be null if
    // the source trace was closed; the patch still succeeds but we skip the
    // effect badge.
    const sourceId = tabOrder.find(
      (id) => traces[id]?.prompt === newIntervention.source_prompt
    );
    const source = sourceId ? traces[sourceId] : null;

    const result = await openTrace(base.prompt, allInterventions);
    if (!result) return;
    const { id: newId, trace: intervened } = result;

    if (source) {
      const sourceTop = source.final_top_k[source.seq_len - 1][0];
      const trackedToken = sourceTop.token;
      const baseProb =
        base.final_top_k[base.seq_len - 1].find(
          (p) => p.token === trackedToken
        )?.prob ?? 0;
      const intrProb =
        intervened.final_top_k[intervened.seq_len - 1].find(
          (p) => p.token === trackedToken
        )?.prob ?? 0;
      const magnitude = intrProb - baseProb;

      setTraceMetadata((prev) => ({
        ...prev,
        [newId]: {
          baseTraceId: originalTraceId,
          effect: { magnitude, trackedToken, direction: "up" },
        },
      }));
    } else {
      setTraceMetadata((prev) => ({
        ...prev,
        [newId]: { baseTraceId: originalTraceId },
      }));
    }

    setCompareLeftId(originalTraceId);
    setCompareRightId(newId);
    setCompareMode(true);
    setCompareError(null);
  }

  function activateTab(id: string) {
    setActiveTabId(id);
  }

  function closeTab(id: string) {
    const idx = tabOrder.indexOf(id);
    const nextOrder = tabOrder.filter((x) => x !== id);

    setTraces((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setViewStates((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setTraceMetadata((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setSweepState((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setTabOrder(nextOrder);

    if (activeTabId === id) {
      const fallback =
        (idx > 0 ? tabOrder[idx - 1] : null) ?? tabOrder[idx + 1] ?? null;
      setActiveTabId(fallback);
    }
    // If this tab was part of an active compare, disable compare mode.
    if (compareLeftId === id) setCompareLeftId(null);
    if (compareRightId === id) setCompareRightId(null);
    if (compareMode && (compareLeftId === id || compareRightId === id)) {
      setCompareMode(false);
      setCompareError(null);
    }
  }

  function updateActiveViewState(patch: Partial<TraceViewState>) {
    if (!activeTabId) return;
    setViewStates((prev) => ({
      ...prev,
      [activeTabId]: { ...prev[activeTabId], ...patch },
    }));
  }

  function updateCompareViewState(patch: Partial<CompareViewState>) {
    setCompareViewState((prev) => (prev ? { ...prev, ...patch } : prev));
  }

  function onAddTab() {
    setClearSignal((n) => n + 1);
    queueMicrotask(() => promptInputRef.current?.focus());
  }

  async function handleRunSweep(
    traceId: string,
    scope: SweepScope,
    currentBlockIdx: number | null
  ) {
    const trace = traces[traceId];
    if (!trace) return;

    setSweepState((prev) => ({
      ...prev,
      [traceId]: {
        isRunning: true,
        progress: null,
        results: null,
        targetToken: null,
        baselineProb: null,
        scopeUsed: scope,
      },
    }));

    try {
      const body = {
        prompt: trace.prompt,
        scope,
        ...(scope === "current_layer" && currentBlockIdx !== null
          ? { layer: currentBlockIdx }
          : {}),
        ...(scope === "layer_range" ? { layer_start: 10, layer_end: 20 } : {}),
        base_interventions: trace.interventions_applied ?? [],
      };
      const resp = await runSweep(body);
      setSweepState((prev) => ({
        ...prev,
        [traceId]: {
          isRunning: false,
          progress: null,
          results: resp.results,
          targetToken: resp.target_token,
          baselineProb: resp.target_token_baseline_prob,
          scopeUsed: scope,
        },
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSweepState((prev) => ({
        ...prev,
        [traceId]: {
          isRunning: false,
          progress: null,
          results: null,
          targetToken: null,
          baselineProb: null,
          scopeUsed: scope,
        },
      }));
    }
  }

  function handleSweepResultClick(traceId: string, item: SweepResultItem) {
    // Reuse the ablation flow — backend cache is already populated from the
    // sweep, so this opens the tab instantly.
    void ablateHead(traceId, {
      type: "zero_head",
      layer: item.layer,
      head: item.head,
    });
  }

  function handleDismissSweep(traceId: string) {
    setSweepState((prev) => {
      const next = { ...prev };
      delete next[traceId];
      return next;
    });
  }

  function onCompareToggle() {
    setCompareMode((prev) => {
      const next = !prev;
      if (next) {
        // If A/B aren't selected yet, prefill with the first two open tabs for convenience.
        if (!compareLeftId && tabOrder[0]) setCompareLeftId(tabOrder[0]);
        if (!compareRightId && tabOrder[1]) setCompareRightId(tabOrder[1]);
        // eslint-disable-next-line no-console
        console.log(
          `compare mode on, A=${compareLeftId ?? "null"}, B=${compareRightId ?? "null"}`
        );
      } else {
        setCompareError(null);
      }
      return next;
    });
  }

  function onCompareSelect(side: "left" | "right", id: string | null) {
    if (side === "left") setCompareLeftId(id);
    else setCompareRightId(id);
    setCompareError(null);
  }

  // When entering compare mode with both sides selected, fetch the divergence.
  useEffect(() => {
    if (!compareMode || !compareLeftId || !compareRightId) return;
    const traceA = traces[compareLeftId];
    const traceB = traces[compareRightId];
    if (!traceA || !traceB) return;

    const key = comparisonKey(traceA.prompt, traceB.prompt);
    // Seed compare view state if it hasn't been set or has become stale for a new pair.
    setCompareViewState((prev) => prev ?? defaultCompareViewState(traceA));

    const cached = getCachedComparison(traceA.prompt, traceB.prompt);
    if (cached) {
      setComparisons((prev) => ({ ...prev, [key]: cached }));
      setCompareError(null);
      setCompareLoading(false);
      return;
    }

    let cancelled = false;
    setCompareLoading(true);
    setCompareError(null);
    fetchComparison(traceA.prompt, traceB.prompt)
      .then((comp) => {
        if (cancelled) return;
        setComparisons((prev) => ({ ...prev, [key]: comp }));
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setCompareError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setCompareLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [compareMode, compareLeftId, compareRightId, traces]);

  // Re-seed the compare view when the selected pair changes.
  useEffect(() => {
    if (!compareMode || !compareLeftId || !compareRightId) return;
    const traceA = traces[compareLeftId];
    if (!traceA) return;
    setCompareViewState(defaultCompareViewState(traceA));
    setAttentionCompareMode("A");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareLeftId, compareRightId, compareMode]);

  useEffect(() => {
    if (bootRef.current) return;
    bootRef.current = true;
    fetch(`${BACKEND}/health`)
      .then((r) => r.json())
      .then((h) => setDevice(h.device ?? null))
      .catch(() => {});
    void runPrompt(DEFAULT_PROMPT);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeTrace = activeTabId ? traces[activeTabId] : null;
  const activeView = activeTabId ? viewStates[activeTabId] : null;

  // Other open traces whose prompt tokenizes to the same length as `forTrace` —
  // valid patch sources. Dedup by prompt (patching depends only on the source text).
  function computeSources(forTrace: Trace | null): SourceTrace[] {
    if (!forTrace) return [];
    const seen = new Set<string>();
    const out: SourceTrace[] = [];
    for (const id of tabOrder) {
      const t = traces[id];
      if (!t) continue;
      if (t.prompt === forTrace.prompt) continue;
      if (t.tokens.length !== forTrace.tokens.length) continue;
      if (seen.has(t.prompt)) continue;
      seen.add(t.prompt);
      out.push({ prompt: t.prompt, label: t.prompt });
    }
    return out;
  }

  const sourceTracesActive = useMemo(
    () => computeSources(activeTrace),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeTrace, tabOrder, traces]
  );

  const compareTraceA =
    compareLeftId && traces[compareLeftId] ? traces[compareLeftId] : null;
  const compareTraceB =
    compareRightId && traces[compareRightId] ? traces[compareRightId] : null;

  const sourceTracesA = useMemo(
    () => computeSources(compareTraceA),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [compareTraceA, tabOrder, traces]
  );
  const sourceTracesB = useMemo(
    () => computeSources(compareTraceB),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [compareTraceB, tabOrder, traces]
  );
  const compareReady = !!(
    compareMode &&
    compareTraceA &&
    compareTraceB &&
    compareViewState
  );
  const comparison =
    compareTraceA && compareTraceB
      ? comparisons[comparisonKey(compareTraceA.prompt, compareTraceB.prompt)] ??
        null
      : null;

  return (
    <div className="flex flex-col h-screen bg-neutral-950 text-neutral-100 relative">
      <TopBar
        initialPrompt={DEFAULT_PROMPT}
        onRun={runPrompt}
        loading={loading}
        device={device}
        inputRef={promptInputRef}
        clearSignal={clearSignal}
      />

      <TabBar
        tabOrder={tabOrder}
        activeTabId={activeTabId}
        traces={traces}
        traceMetadata={traceMetadata}
        onActivate={activateTab}
        onClose={closeTab}
        onAddTab={onAddTab}
        compareMode={compareMode}
        compareLeftId={compareLeftId}
        compareRightId={compareRightId}
        onCompareToggle={onCompareToggle}
        onCompareSelect={onCompareSelect}
      />

      {error && (
        <div className="px-4 py-2 border-b border-red-900/50 bg-red-950/40 text-red-300 text-[12px]">
          error: {error}
        </div>
      )}

      {compareReady && compareTraceA && compareTraceB && compareViewState ? (
        <CompareView
          traceA={compareTraceA}
          traceB={compareTraceB}
          comparison={comparison}
          loading={compareLoading}
          error={compareError}
          compareViewState={compareViewState}
          attentionCompareMode={attentionCompareMode}
          onUpdateView={updateCompareViewState}
          onAttentionCompareMode={setAttentionCompareMode}
          onAblateHeadA={
            compareLeftId
              ? (iv) => void ablateHead(compareLeftId, iv)
              : undefined
          }
          onAblateHeadB={
            compareRightId
              ? (iv) => void ablateHead(compareRightId, iv)
              : undefined
          }
          onPatchHeadA={
            compareLeftId
              ? (iv) => void patchHead(compareLeftId, iv)
              : undefined
          }
          onPatchHeadB={
            compareRightId
              ? (iv) => void patchHead(compareRightId, iv)
              : undefined
          }
          sourceTracesA={sourceTracesA}
          sourceTracesB={sourceTracesB}
        />
      ) : activeTrace && activeView ? (
        <>
          <TokenStrip
            tokens={activeTrace.tokens}
            selectedIdx={activeView.selectedTokenIdx}
            onSelect={(idx) => updateActiveViewState({ selectedTokenIdx: idx })}
          />
          <div className="flex flex-1 min-h-0">
            <LayerList
              trace={activeTrace}
              selectedTokenIdx={activeView.selectedTokenIdx}
              selectedLayerIdx={activeView.selectedLayerIdx}
              onSelect={(idx) =>
                updateActiveViewState({ selectedLayerIdx: idx })
              }
            />
            <DetailPanel
              trace={activeTrace}
              selectedTokenIdx={activeView.selectedTokenIdx}
              selectedLayerIdx={activeView.selectedLayerIdx}
              tab={activeView.detailTab}
              onTab={(t: DetailTab) => updateActiveViewState({ detailTab: t })}
              onAblateHead={
                activeTabId
                  ? (iv) => void ablateHead(activeTabId, iv)
                  : undefined
              }
              onPatchHead={
                activeTabId
                  ? (iv) => void patchHead(activeTabId, iv)
                  : undefined
              }
              sourceTraces={sourceTracesActive}
              sweepState={
                activeTabId ? sweepState[activeTabId] : undefined
              }
              canSweep={!!activeTabId && !compareMode}
              onRunSweep={
                activeTabId
                  ? (scope, blockIdx) =>
                      void handleRunSweep(activeTabId, scope, blockIdx)
                  : undefined
              }
              onSweepResultClick={
                activeTabId
                  ? (item) => handleSweepResultClick(activeTabId, item)
                  : undefined
              }
              onDismissSweep={
                activeTabId ? () => handleDismissSweep(activeTabId) : undefined
              }
            />
          </div>
          {loading && (
            <div className="absolute top-2 right-2 text-[10px] text-accent animate-pulse select-none">
              computing...
            </div>
          )}
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center text-neutral-500 text-sm select-none">
          {loading ? (
            <span className="animate-pulse">computing trace...</span>
          ) : compareMode ? (
            <span>Pick two tabs in A and B to compare.</span>
          ) : (
            <>Enter a prompt above to start a new trace.</>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
