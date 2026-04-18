"""FastAPI server exposing trace_core.compute_trace over HTTP."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from diff_core import compute_divergence, load_model
from trace_core import ForwardTrace, compute_trace


class ZeroHeadIntervention(BaseModel):
    type: Literal["zero_head"]
    layer: int
    head: int


class ZeroMLPIntervention(BaseModel):
    type: Literal["zero_mlp"]
    layer: int


class PatchHeadIntervention(BaseModel):
    type: Literal["patch_head"]
    layer: int
    head: int
    source_prompt: str


Intervention = Annotated[
    Union[ZeroHeadIntervention, ZeroMLPIntervention, PatchHeadIntervention],
    Field(discriminator="type"),
]


class TraceRequest(BaseModel):
    prompt: str
    top_k: int = 5
    interventions: list[Intervention] = []


class CompareRequest(BaseModel):
    prompt_a: str
    prompt_b: str


class CompareResponse(BaseModel):
    prompt_a: str
    prompt_b: str
    n_layers: int
    # length n_layers + 1 — index 0 is the embedding-layer (pre-layer-0) lens
    kl_symmetric: list[float]
    cosine_distance: list[float]


class TokenPrediction(BaseModel):
    token: str
    prob: float


class TraceResponse(BaseModel):
    prompt: str
    tokens: list[str]
    token_ids: list[int]
    n_layers: int
    n_heads: int
    d_model: int
    seq_len: int
    # [n_layers+1][seq_len][top_k] — index 0 is the pre-layer-0 (embedding) lens
    logit_lens_top_k: list[list[list[TokenPrediction]]]
    final_top_k: list[list[TokenPrediction]]
    # [n_layers][n_heads][seq_len][seq_len]
    attention_patterns: list[list[list[list[float]]]]
    interventions_applied: list[Intervention] = []


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_model("gpt2-medium")
    device = str(next(model.parameters()).device)
    _state["model"] = model
    _state["device"] = device
    _state["cache"] = {}
    _state["compare_cache"] = {}
    print(f"[server] gpt2-medium ready on {device}", flush=True)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _embedding_layer_divergence(
    prompt_a: str, prompt_b: str, model
) -> tuple[float, float]:
    """KL (symmetric) and cosine distance at the pre-layer-0 (embedding) residual,
    at the final token position. Run once per (a, b); cached at the endpoint layer."""
    with torch.no_grad():
        tokens_a = model.to_tokens(prompt_a)
        tokens_b = model.to_tokens(prompt_b)
        _, cache_a = model.run_with_cache(tokens_a)
        _, cache_b = model.run_with_cache(tokens_b)

        emb_a = (cache_a["hook_embed"][0] + cache_a["hook_pos_embed"][0])[-1]
        emb_b = (cache_b["hook_embed"][0] + cache_b["hook_pos_embed"][0])[-1]

        normed_a = model.ln_final(emb_a.unsqueeze(0)).squeeze(0)
        normed_b = model.ln_final(emb_b.unsqueeze(0)).squeeze(0)

        W_U = model.W_U
        b_U = getattr(model, "b_U", None)
        logits_a = normed_a @ W_U
        logits_b = normed_b @ W_U
        if b_U is not None:
            logits_a = logits_a + b_U
            logits_b = logits_b + b_U

        log_p_a = F.log_softmax(logits_a, dim=-1)
        log_p_b = F.log_softmax(logits_b, dim=-1)
        p_a = log_p_a.exp()
        p_b = log_p_b.exp()
        kl_ab = (p_a * (log_p_a - log_p_b)).sum().item()
        kl_ba = (p_b * (log_p_b - log_p_a)).sum().item()
        kl_sym = (kl_ab + kl_ba) / 2.0

        cos = F.cosine_similarity(
            emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=-1
        ).item()
        cos_dist = 1.0 - cos

    return kl_sym, cos_dist


def _embedding_layer_lens(
    trace: ForwardTrace, model, top_k: int
) -> list[list[tuple[str, float]]]:
    """Logit lens on the pre-layer-0 residual (embedding + pos_embed)."""
    with torch.no_grad():
        device = next(model.parameters()).device
        embed = trace.residual_stream[0].to(device)  # [seq, d_model]
        normed = model.ln_final(embed)
        logits = normed @ model.W_U
        b_U = getattr(model, "b_U", None)
        if b_U is not None:
            logits = logits + b_U
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

    rows: list[list[tuple[str, float]]] = []
    for pos in range(probs.shape[0]):
        row = [
            (model.to_string([tid]), float(p))
            for p, tid in zip(top_probs[pos].tolist(), top_ids[pos].tolist())
        ]
        rows.append(row)
    return rows


def _wrap_row(row: list[tuple[str, float]]) -> list[TokenPrediction]:
    return [TokenPrediction(token=t, prob=float(p)) for t, p in row]


# --- interventions --------------------------------------------------------
# Zero/patch hooks live at blocks.{L}.attn.hook_z (shape [batch, seq, n_heads, d_head])
# or blocks.{L}.hook_mlp_out (shape [batch, seq, d_model]). Adding a new
# intervention type = add a Pydantic class + a branch in _build_hooks (+ optional
# prep work in _prepare_patch_sources if the hook needs ahead-of-time tensors).

def _make_zero_head_hook(head_idx: int):
    def fn(z, hook):  # z: [batch, seq, n_heads, d_head]
        z[:, :, head_idx, :] = 0.0
        return z

    return fn


def _zero_mlp_hook(mlp_out, hook):  # mlp_out: [batch, seq, d_model]
    return torch.zeros_like(mlp_out)


def _make_patch_head_hook(head_idx: int, source_slice):
    """source_slice: [batch, seq, d_head] — overwrites the target's head output."""

    def fn(z, hook):  # z: [batch, seq, n_heads, d_head]
        z[:, :, head_idx, :] = source_slice
        return z

    return fn


def _prepare_patch_sources(
    interventions: list, model, target_len: int
) -> dict[str, dict[int, torch.Tensor]]:
    """For every unique (source_prompt, layer) referenced by patch interventions,
    run the source prompt through the clean model and cache its hook_z tensor.
    Returns {source_prompt: {layer: hook_z_tensor}}."""
    needed: dict[str, set[int]] = {}
    for iv in interventions:
        if isinstance(iv, PatchHeadIntervention):
            needed.setdefault(iv.source_prompt, set()).add(iv.layer)

    out: dict[str, dict[int, torch.Tensor]] = {}
    for source_prompt, layers in needed.items():
        tokens = model.to_tokens(source_prompt)
        if tokens.shape[1] != target_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Patch source must tokenize to the same length as the target "
                    f"(target={target_len}, source {source_prompt!r}={tokens.shape[1]})"
                ),
            )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        out[source_prompt] = {
            layer: cache[f"blocks.{layer}.attn.hook_z"].detach().clone()
            for layer in layers
        }
    return out


def _build_hooks(
    interventions: list,
    patch_sources: dict[str, dict[int, torch.Tensor]],
) -> list[tuple[str, object]]:
    hooks: list[tuple[str, object]] = []
    for iv in interventions:
        if isinstance(iv, ZeroHeadIntervention):
            hooks.append(
                (f"blocks.{iv.layer}.attn.hook_z", _make_zero_head_hook(iv.head))
            )
        elif isinstance(iv, ZeroMLPIntervention):
            hooks.append((f"blocks.{iv.layer}.hook_mlp_out", _zero_mlp_hook))
        elif isinstance(iv, PatchHeadIntervention):
            source_z = patch_sources[iv.source_prompt][iv.layer]
            # slice [batch, seq, d_head] for this head
            head_slice = source_z[:, :, iv.head, :].clone()
            hooks.append(
                (
                    f"blocks.{iv.layer}.attn.hook_z",
                    _make_patch_head_hook(iv.head, head_slice),
                )
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown intervention type: {iv!r}"
            )
    return hooks


def _trace_cache_key(prompt: str, interventions: list) -> str:
    intervention_key = json.dumps(
        [iv.model_dump() for iv in interventions], sort_keys=True
    )
    return f"{prompt}|||{intervention_key}"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "gpt2-medium",
        "device": _state.get("device", "unknown"),
    }


@app.post("/trace", response_model=TraceResponse)
def trace_endpoint(req: TraceRequest) -> TraceResponse:
    cache: dict[str, TraceResponse] = _state["cache"]
    key = _trace_cache_key(req.prompt, req.interventions)
    if key in cache:
        return cache[key]

    model = _state["model"]

    # Validate length for patch sources and extract their activations clean (no hooks).
    target_tokens = model.to_tokens(req.prompt)
    target_len = int(target_tokens.shape[1])
    patch_sources = _prepare_patch_sources(req.interventions, model, target_len)

    hooks = _build_hooks(req.interventions, patch_sources)

    try:
        for hook_name, hook_fn in hooks:
            model.add_hook(hook_name, hook_fn)
        trace = compute_trace(req.prompt, model, top_k=req.top_k)
        embed_lens = _embedding_layer_lens(trace, model, top_k=req.top_k)
    finally:
        # Always clean up — stale hooks would corrupt the next request.
        model.reset_hooks()

    all_lens = [embed_lens, *trace.logit_lens_top_k]

    resp = TraceResponse(
        prompt=trace.prompt,
        tokens=trace.tokens,
        token_ids=trace.token_ids,
        n_layers=trace.n_layers,
        n_heads=trace.n_heads,
        d_model=trace.d_model,
        seq_len=len(trace.tokens),
        logit_lens_top_k=[[_wrap_row(row) for row in layer] for layer in all_lens],
        final_top_k=[_wrap_row(row) for row in trace.final_top_k],
        attention_patterns=trace.attention_patterns.tolist(),
        interventions_applied=list(req.interventions),
    )
    cache[key] = resp
    return resp


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest) -> CompareResponse:
    cache: dict[tuple[str, str], dict] = _state["compare_cache"]
    key: tuple[str, str] = tuple(sorted([req.prompt_a, req.prompt_b]))  # type: ignore[assignment]
    if key in cache:
        c = cache[key]
        return CompareResponse(
            prompt_a=req.prompt_a,
            prompt_b=req.prompt_b,
            n_layers=c["n_layers"],
            kl_symmetric=c["kl_symmetric"],
            cosine_distance=c["cosine_distance"],
        )

    model = _state["model"]
    tokens_a = model.to_tokens(req.prompt_a)
    tokens_b = model.to_tokens(req.prompt_b)
    if tokens_a.shape[1] != tokens_b.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompts must tokenize to the same length "
                f"(got {tokens_a.shape[1]} vs {tokens_b.shape[1]})"
            ),
        )

    try:
        div = compute_divergence(req.prompt_a, req.prompt_b, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    emb_kl, emb_cos = _embedding_layer_divergence(req.prompt_a, req.prompt_b, model)
    kl_full = [float(emb_kl), *[float(x) for x in div["kl_symmetric"]]]
    cos_full = [float(emb_cos), *[float(x) for x in div["cosine_distance"]]]
    n_layers = len(div["layers"])

    cache[key] = {
        "n_layers": n_layers,
        "kl_symmetric": kl_full,
        "cosine_distance": cos_full,
    }
    return CompareResponse(
        prompt_a=req.prompt_a,
        prompt_b=req.prompt_b,
        n_layers=n_layers,
        kl_symmetric=kl_full,
        cosine_distance=cos_full,
    )
