# neural trace

A mechanistic interpretability tool for comparing forward passes across two
inputs on a small transformer, visualized in a dark, monospace IDA-flavored UI.
Cross-input comparison is a first-class, interactive default view — no
scripting per experiment.

Model: `gpt2-medium` (24 layers, 16 heads, 1024 d_model) via TransformerLens.

## Features

- **Single-trace view**: per-layer logit-lens top-5 for any token position,
  final-layer predictions, per-head attention heatmaps (4×4 grid) plus a
  mean heatmap with token-labelled axes.
- **Compare mode** (split view): side-by-side tokens strips with diff
  highlighting, a KL-divergence layer list between the two traces, split
  predictions panel, and an attention diff tab with per-head `A − B`
  heatmaps on a diverging cyan↔amber scale.
- **Interventions**: click a head cell → popover with:
  - **Ablate** — zero that head's output (necessity probe).
  - **Patch** — transplant that head's activations from another open trace of
    the same token length (sufficiency probe).
  Results open as a new tab and auto-activate compare mode (original vs
  intervened). Tab labels show a colored effect magnitude
  (`[−L15.H7 ↓.13]` / `[L15.H7←Italy ↑.03]`) and a rich hover tooltip with
  the probability transition.
- **Mask BOS toggle**: hides the attention-sink spike at position 0 so
  real per-head structure is visible, rescales colors to the non-BOS range.
- **Sparkline**: layer-list rows include a small bar of the final-layer
  top-1 token's probability across layers, so the emergence ("the moment
  of factual recall") is visible at a glance.
- **Trace ID is deterministic** on `(prompt, interventions)` — opening the
  same intervention again just switches tabs instead of re-fetching.

## Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Node 18+** and `npm` (Node 22 is what the project was developed on).
- **Optional**: NVIDIA GPU with a driver supporting CUDA 12.x. The app runs on
  CPU just fine — see the CPU-only install below.

## Setup

### 1. Clone and enter the project

```
git clone <this-repo> hackathon
cd hackathon
```

### 2. Install backend dependencies

The committed `pyproject.toml` pins the PyTorch CUDA 12.4 wheel index. Pick
the install path that matches your machine.

**With GPU (NVIDIA + CUDA 12.x driver)** — default, no edits needed:

```
uv sync
```

**CPU-only** — replace the torch index with the CPU wheel build before syncing:

```
# one-time: point pyproject at the CPU wheel index instead
sed -i 's|whl/cu124|whl/cpu|' pyproject.toml
uv sync
```

(Or edit `pyproject.toml`'s `[[tool.uv.index]]` block by hand to
`url = "https://download.pytorch.org/whl/cpu"`.)

The CPU wheels are ~200 MB; CUDA wheels are ~2 GB. Either way, `uv sync`
creates `.venv/` and installs `fastapi`, `uvicorn`, `torch`,
`transformer_lens`, `streamlit`, plus their transitive deps.

### 3. Install frontend dependencies

```
cd frontend
npm install
cd ..
```

## Running

Backend on `:8000`, frontend on `:5173`. Run each in its own terminal.

### Backend

```
uv run uvicorn server:app --reload --port 8000
```

First startup downloads GPT-2 medium weights (~1.4 GB) to
`~/.cache/huggingface` — subsequent runs are instant. When ready you'll see
`[server] gpt2-medium ready on cuda:0` (or `cpu`). Health check:

```
curl http://localhost:8000/health
# {"status":"ok","model":"gpt2-medium","device":"cuda:0"}   (or "cpu")
```

### Frontend

```
cd frontend
npm run dev
```

Open http://localhost:5173. The default prompt
`"The capital of France is the city of"` auto-loads as the first tab.

### Performance without a GPU

The app is fully functional on CPU — every feature works. Expect one forward
pass (~1 `/trace` request on `gpt2-medium`) to take roughly **0.5–2 seconds**
on a modern laptop CPU, versus ~100 ms on a mid-range consumer GPU
(RTX 3050 Ti). The 25-layer logit-lens and 24×16 attention patterns are all
computed server-side per request and cached, so interacting with an
already-loaded trace stays fast regardless of hardware. Only the diagnostic
sweeps (`find_critical_heads.py`, `test_patching.py` — 144 requests each)
noticeably slow down on CPU; budget ~5 minutes instead of ~30 seconds.

## Architecture

```
.
├── server.py              FastAPI app. /trace (+ interventions), /compare,
│                          /health. Loads the model once at startup, keeps
│                          hooks clean via try/finally + reset_hooks().
├── trace_core.py          compute_trace(prompt, model) → ForwardTrace:
│                          residual_stream, attention_patterns, per-layer
│                          per-position logit-lens top-k, final top-k.
├── diff_core.py           compute_divergence(a, b, model) → per-layer
│                          symmetric KL + cosine distance between two
│                          prompts at the final token.
├── validate_setup.py      One-shot smoke test (model loads, device, both
│                          prompts produce expected tokens).
├── find_critical_heads.py Sweep single-head ablations, rank by damage to
│                          p(' Paris') on the France prompt.
├── test_patching.py       Sweep single-head Italy→France patches, rank by
│                          p(' Rome') after patching.
└── frontend/
    ├── src/theme.ts       Colour constants (two-file rollback: here +
    │                      src/index.css @theme block).
    ├── src/nivoTheme.ts   Nivo chart theme, amber ramp, diverging scale.
    ├── src/api.ts         fetchTrace, fetchComparison (module-level cache).
    ├── src/interventions.ts  traceId hash, badge/label formatting,
    │                         magnitude formatter.
    ├── src/components/
    │  ├── TopBar, TabBar         Chrome.
    │  ├── TokenStrip             Single-trace token chips.
    │  ├── LayerList              Single-trace layer list with sparkline.
    │  ├── DetailPanel            Single-trace detail (predictions/attention).
    │  ├── PredictionsTab / PredBars   Top-5 bar chart (Nivo).
    │  ├── AttentionTab           Per-head grid + mean heatmap + popover
    │  │                          (Ablate / Patch), Mask-BOS toggle.
    │  ├── CompareView            Compare mode orchestration.
    │  ├── CompareTokenStrips     Two stacked strips with diff outlines.
    │  ├── CompareLayerList       KL-by-layer bars.
    │  ├── CompareDetailPanel     Split predictions + A/B/diff attention.
    │  └── AttentionDiff          Per-head Δ heatmaps + mean Δ heatmap.
    └── src/App.tsx              Top-level state: traces, viewStates, tabs,
                                 compare state, traceMetadata (effect).
```

## Diagnostic scripts

Run against a live backend (start it first):

```
uv run python validate_setup.py
uv run python find_critical_heads.py
uv run python test_patching.py
```

Representative findings (on this project's canonical prompts):

- **Baseline**: `France` → ` Paris` (0.338), `Italy` → ` Rome` (0.305).
- **Symmetric KL** between the two traces is flat (~0.01) through L14,
  spikes at L16 (7.35), peaks L20 (13.2). That's the "moment of factual
  recall."
- **Most-damaging ablation**: `L15.H7` crashes Paris 0.338 → 0.209 and
  promotes French cities (Lyon, Nice, Strasbourg, Marseille). No single
  head kills Paris outright — the circuit is distributed.
- **Most-effective patch**: `L16.H2` surfaces Rome to ~0.026. Single-head
  patching is too weak to fully flip France → Rome; you'd need multi-head
  or MLP patches for a clean demo flip.

## Honest notes

- Single-head patching is a weak intervention for factual recall in
  gpt2-medium. Expect subtle magnitude changes, not dramatic flips. The UI
  surfaces the magnitude so you can see "this did basically nothing" when
  that's the truth.
- The logit-lens sparkline looks up the final-layer top-1 token in each
  earlier layer's top-5 — if it's absent we show `0`. An honest
  under-approximation at early layers; a full fix would need the backend to
  return the target token's probability at every layer.
- Frontend typing is isolated to the top-bar's input so heatmaps don't
  re-render on every keystroke.
- Hooks are always cleaned up in a `try/finally`; the `/trace` endpoint
  with no interventions is byte-identical before and after any
  intervention request.

## Re-theming

Colors are centralized in two places. Edit both and the UI re-skins without
touching components:

- `frontend/src/theme.ts` — TypeScript constants (used by Nivo + inline).
- `frontend/src/index.css` — `@theme { --color-accent: ...; ... }` block
  (used by Tailwind utilities).

## Stack

Python 3.11, `uv`, FastAPI, uvicorn, PyTorch 2.6 + CUDA 12.4,
TransformerLens 3.0. Node 22, Vite 8, React 19, TypeScript 6,
Tailwind CSS v4, Nivo 0.99.
