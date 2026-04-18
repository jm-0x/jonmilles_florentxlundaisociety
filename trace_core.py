"""Single-prompt forward trace: the core abstraction.

One forward pass → one ForwardTrace capturing residuals, attention patterns,
per-layer/per-position logit-lens top-k, and final top-k predictions.
Everything is moved to CPU for downstream consumers (e.g. a UI).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from transformer_lens import HookedTransformer

from diff_core import load_model


@dataclass
class ForwardTrace:
    prompt: str
    tokens: list[str]
    token_ids: list[int]
    n_layers: int
    n_heads: int
    d_model: int
    residual_stream: Tensor           # [n_layers+1, seq_len, d_model]
    attention_patterns: Tensor        # [n_layers, n_heads, seq_len, seq_len]
    logit_lens_top_k: list            # [n_layers][seq_len] -> [(token_str, prob), ...]
    final_top_k: list                 # [seq_len] -> [(token_str, prob), ...]


def _top_k_per_position(
    logits_seq: Tensor, model: HookedTransformer, k: int
) -> list[list[tuple[str, float]]]:
    """logits_seq: [seq_len, vocab]. Returns seq_len rows of [(tok, prob)] of length k."""
    probs = F.softmax(logits_seq, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    out: list[list[tuple[str, float]]] = []
    for pos in range(probs.shape[0]):
        row: list[tuple[str, float]] = []
        for p, tid in zip(top_probs[pos].tolist(), top_ids[pos].tolist()):
            row.append((model.to_string([tid]), p))
        out.append(row)
    return out


def compute_trace(prompt: str, model: HookedTransformer, top_k: int = 5) -> ForwardTrace:
    with torch.no_grad():
        tokens = model.to_tokens(prompt)  # [1, seq]
        logits, cache = model.run_with_cache(tokens)

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        d_model = model.cfg.d_model

        # Residual stream: pre-layer-0 (embed + pos_embed) + each block's hook_resid_post
        embed = cache["hook_embed"][0] + cache["hook_pos_embed"][0]
        resid_layers = [cache[f"blocks.{i}.hook_resid_post"][0] for i in range(n_layers)]
        residual = torch.stack([embed, *resid_layers], dim=0)  # [n_layers+1, seq, d_model]

        attention = torch.stack(
            [cache[f"blocks.{i}.attn.hook_pattern"][0] for i in range(n_layers)],
            dim=0,
        )  # [n_layers, n_heads, seq, seq]

        # Logit lens at every layer, every position
        W_U = model.W_U
        b_U = getattr(model, "b_U", None)
        logit_lens_top_k: list[list[list[tuple[str, float]]]] = []
        for i in range(n_layers):
            resid = cache[f"blocks.{i}.hook_resid_post"][0]  # [seq, d_model]
            normed = model.ln_final(resid)
            layer_logits = normed @ W_U
            if b_U is not None:
                layer_logits = layer_logits + b_U
            logit_lens_top_k.append(_top_k_per_position(layer_logits, model, top_k))

        final_top_k = _top_k_per_position(logits[0], model, top_k)

        str_tokens = model.to_str_tokens(prompt)
        token_ids = tokens[0].tolist()

    return ForwardTrace(
        prompt=prompt,
        tokens=str_tokens,
        token_ids=token_ids,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        residual_stream=residual.cpu(),
        attention_patterns=attention.cpu(),
        logit_lens_top_k=logit_lens_top_k,
        final_top_k=final_top_k,
    )


if __name__ == "__main__":
    model = load_model("gpt2-medium")
    trace = compute_trace("The capital of France is the city of", model, top_k=5)

    print(f"prompt: {trace.prompt!r}")
    print(
        f"n_layers={trace.n_layers}  n_heads={trace.n_heads}  "
        f"d_model={trace.d_model}  seq_len={len(trace.tokens)}"
    )
    print(f"residual_stream shape: {tuple(trace.residual_stream.shape)}")
    print(f"attention_patterns shape: {tuple(trace.attention_patterns.shape)}")
    print(f"tokens: {trace.tokens}\n")

    print("logit-lens top-1 at FINAL position, across layers:")
    print(f"  {'layer':>5} | top-1")
    print("  " + "-" * 35)
    for i, layer_rows in enumerate(trace.logit_lens_top_k):
        tok, prob = layer_rows[-1][0]
        print(f"  {i:>5} | {tok!r} ({prob:.3f})")
    print()

    print("final model top-1 prediction per position:")
    print(f"  {'pos':>3} | {'input token':<18} | top-1 next")
    print("  " + "-" * 55)
    for pos, row in enumerate(trace.final_top_k):
        tok, prob = row[0]
        input_tok = trace.tokens[pos]
        print(f"  {pos:>3} | {input_tok!r:<18} | {tok!r} ({prob:.3f})")
