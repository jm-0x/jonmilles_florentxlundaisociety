# neural trace

A GDB-style interactive debugger for transformer internals.

## Problem

Modern AI systems run on transformer models whose internal reasoning is
invisible. When these systems fail in unexpected ways — confidently
producing a wrong answer on an input nobody anticipated — engineers have no
system to look inside and ask why. Evals and logs tell you that the model
failed, but nothing more. They're black-box unit tests. They can't surface
root causes.

Mechanistic interpretability is the emerging standard way to do
root-cause analysis on transformer models. With techniques like activation
patching, ablation studies, and logit-lens analysis, engineers can reverse
engineer the internal computations of a trained model. Existing solutions
require developers to manually investigate, often writing custom
investigation code in Jupyter notebooks and drawing matplotlib plots. No
interactive interface exists. There is no persistent analysis and no
compare-and-iterate workflow, like we are used to from the world of reverse
engineering and debugging. Neural Trace is built to close this gap,
bringing a GDB-style interactive debugger to transformer internals.

## Solution

Neural Trace is a web-based debugger for transformer language models. It
treats a forward pass as a navigable trace. You can pause at any
(layer, token) pair, inspect internal state, run causal analysis, and
compare runs side-by-side.

Three capabilities:

**Inspection.** Navigate every layer's state for any prompt. See what the
model predicts at each layer (logit lens), watch predictions emerge across
layers, and inspect per-head attention patterns.

**Intervention.** Zero-ablate (zero out) any attention head with one click
to test whether it's necessary for a behavior. Patch activations from a
different prompt's forward pass to test sufficiency. Run a sweep across all
heads in a layer (or range of layers) to find the most critical components
automatically.

**Comparison.** Run two prompts side-by-side with a shared layer axis.
KL-divergence bars show where in the network the prompts start disagreeing.
Attention patterns can be diffed directly to surface heads doing
input-specific work.

A typical investigation takes minutes instead of hours. Example from the
demo: asking *"where does GPT-2 medium encode that Paris is the capital of
France?"*. The tool surfaces that L15.H7 is necessary for the specific
Paris prediction (ablating it drops Paris probability by 13 points and the
margin over other French cities disappears), and that the same head is
*not* what carries the Italy-to-France-prediction signal when patched. A
real finding reproduced in three clicks.

## Technical approach

Model: `gpt2-medium` via TransformerLens (24 layers, 16 heads per layer,
1024 d_model).

**Backend** — Python, FastAPI, PyTorch, TransformerLens.

- Single model loaded once at server startup; every request is a forward
  pass through the same in-memory `HookedTransformer`.
- `POST /trace`: one prompt → a full `ForwardTrace` with the residual stream
  at every layer+token, attention patterns per head, per-layer/per-position
  logit-lens top-k, and the final model top-k. Response keyed-cached on
  `(prompt, interventions)`.
- `POST /compare`: two equal-length prompts → per-layer symmetric KL and
  cosine distance at the final token position.
- `POST /sweep`: baseline + a (layer, head) grid of single-head ablations
  over a chosen scope, ranked by effect on a tracked target token. Each
  ablation populates the trace cache as a side effect, so clicking a sweep
  result opens the full ablated trace instantly.
- Interventions are Pydantic-discriminated-union types
  (`zero_head` / `zero_mlp` / `patch_head`). Each maps to a TransformerLens
  forward hook registered via `model.add_hook(...)` and always torn down in
  a `try/finally` — stale hooks can't leak between requests.
- Patching runs the source prompt's forward pass up-front, extracts
  `hook_z` for the requested (layer, head), and injects that slice into the
  target's forward pass at the same site. Source/target must tokenize to
  equal length; mismatch returns 400.

**Frontend** — Vite + React 19 + TypeScript 6 + Tailwind v4 + Nivo 0.99.

- Dark, monospace, IDA/Ghidra-flavored UI. Information density over
  whitespace; one consistent accent color.
- State shape: per-trace view state (selected token, selected layer, detail
  tab), per-trace sweep state, app-level compare state. Trace IDs are
  deterministic hashes of `(prompt, sorted-interventions)` so repeating an
  intervention just switches tabs instead of re-fetching.
- Heatmaps and bar charts use memoized data. Keystrokes in the prompt input
  are isolated to the TopBar component so they don't re-render the 17+
  Nivo charts that make up the attention view.
- Compare mode is a split view with two token strips, a KL-by-layer bar
  list, a shared-scale predictions panel, and an attention section with
  A-only / B-only / diff segmented controls. Diff heatmaps use a diverging
  cyan↔amber scale.
- Every intervention auto-activates compare mode (base trace ↔ intervened
  trace) so the user never has to set up the comparison manually. Tab
  labels surface the effect magnitude (`[−L15.H7 ↓.13]` / `[L15.H7←Italy ↑.03]`)
  color-coded by intervention type.

## How to run

### Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Node 18+** and `npm` (Node 22 is what the project was developed on).
- **Optional**: NVIDIA GPU with a driver supporting CUDA 12.x. The app runs
  on CPU just fine — see the CPU-only install below.

### Setup

```
git clone <this-repo> neural-trace
cd neural-trace
```

#### Install backend dependencies

The committed `pyproject.toml` pins the PyTorch CUDA 12.4 wheel index.
Pick the install path that matches your machine.

**With GPU (NVIDIA + CUDA 12.x driver)** — default, no edits needed:

```
uv sync
```

**CPU-only** — swap the torch index to the CPU wheel build before syncing:

```
sed -i 's|whl/cu124|whl/cpu|' pyproject.toml
uv sync
```

(Or edit `pyproject.toml`'s `[[tool.uv.index]]` block by hand to
`url = "https://download.pytorch.org/whl/cpu"`.)

CPU wheels are ~200 MB; CUDA wheels are ~2 GB. Either way, `uv sync`
creates `.venv/` and installs `fastapi`, `uvicorn`, `torch`,
`transformer_lens`, plus transitive deps.

#### Install frontend dependencies

```
cd frontend
npm install
cd ..
```

### Running

Backend on `:8000`, frontend on `:5173`. Run each in its own terminal.

**Backend:**

```
uv run uvicorn server:app --reload --port 8000
```

First startup downloads GPT-2 medium weights (~1.4 GB) to
`~/.cache/huggingface`. When ready you'll see
`[server] gpt2-medium ready on cuda:0` (or `cpu`). Health check:

```
curl http://localhost:8000/health
# {"status":"ok","model":"gpt2-medium","device":"cuda:0"}   (or "cpu")
```

**Frontend:**

```
cd frontend
npm run dev
```

Open http://localhost:5173. The default prompt
`"The capital of France is the city of"` auto-loads as the first tab.

### Performance without a GPU

Every feature works on CPU. Expect one forward pass to take roughly
**0.5–2 seconds** on a modern laptop CPU versus ~100 ms on a mid-range
consumer GPU (RTX 3050 Ti). Already-cached traces stay fast regardless.
The diagnostic sweep scripts noticeably slow down on CPU — budget minutes
instead of seconds.

## Architecture

```
.
├── server.py              FastAPI app. /trace (+ interventions), /compare,
│                          /sweep, /health. Loads the model once at startup,
│                          keeps hooks clean via try/finally + reset_hooks().
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
    ├── src/api.ts         fetchTrace, fetchComparison, runSweep.
    ├── src/interventions.ts  traceId hash, badge/label formatting,
    │                         magnitude formatter.
    ├── src/components/
    │  ├── TopBar, TabBar         Chrome.
    │  ├── TokenStrip             Single-trace token chips.
    │  ├── LayerList              Single-trace layer list with sparkline.
    │  ├── DetailPanel            Single-trace detail (predictions/attention).
    │  ├── PredictionsTab / PredBars   Top-5 bar chart (Nivo).
    │  ├── AttentionTab           Per-head grid + mean heatmap + popover
    │  │                          (Ablate / Patch), Mask-BOS toggle, sweep.
    │  ├── SweepControl           Scope dropdown + Run Sweep button.
    │  ├── SweepResultsPanel      Ranked ablation results, sign-colored.
    │  ├── CompareView            Compare mode orchestration.
    │  ├── CompareTokenStrips     Two stacked strips with diff outlines.
    │  ├── CompareLayerList       KL-by-layer bars.
    │  ├── CompareDetailPanel     Split predictions + A/B/diff attention.
    │  └── AttentionDiff          Per-head Δ heatmaps + mean Δ heatmap.
    └── src/App.tsx              Top-level state: traces, viewStates, tabs,
                                 compare state, traceMetadata, sweepState.
```

## Diagnostic scripts

Run against a live backend:

```
uv run python validate_setup.py
uv run python find_critical_heads.py
uv run python test_patching.py
```

Representative findings on the canonical France/Italy prompts:

- **Baseline**: `France` → ` Paris` (0.338), `Italy` → ` Rome` (0.305).
- **Symmetric KL** between the two traces is flat (~0.01) through L14,
  spikes at L16 (7.35), peaks L20 (13.2). That's the moment of factual
  recall.
- **Most-damaging ablation**: `L15.H7` crashes Paris 0.338 → 0.209 and
  promotes French cities (Lyon, Nice, Strasbourg, Marseille). No single
  head kills Paris outright — the circuit is distributed.
- **Most-effective single-head patch**: `L16.H2` surfaces Rome to ~0.026.
  Single-head patching is too weak to fully flip France → Rome; you'd need
  multi-head or MLP patches for a clean demo flip.

## Re-theming

Colors are centralized in two places. Edit both and the UI re-skins
without touching components:

- `frontend/src/theme.ts` — TypeScript constants (used by Nivo + inline).
- `frontend/src/index.css` — `@theme { --color-accent: ...; ... }` block
  (used by Tailwind utilities).

## Stack

Python 3.11, `uv`, FastAPI, uvicorn, PyTorch 2.6 + CUDA 12.4,
TransformerLens 3.0. Node 22, Vite 8, React 19, TypeScript 6,
Tailwind CSS v4, Nivo 0.99.
