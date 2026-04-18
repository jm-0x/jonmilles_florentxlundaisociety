"""Cross-prompt divergence via logit lens + residual cosine distance.

For two equal-length prompts, walk every layer, project the final-token residual
through ln_final @ W_U (+ b_U), and record:
  - symmetric KL between the two logit-lens distributions
  - cosine distance between the two residual vectors
  - top-1 predicted token + prob for each prompt at that layer
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def load_model(name: str = "gpt2-medium") -> HookedTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(name)
    model.to(device)
    model.eval()
    return model


def compute_divergence(prompt_a: str, prompt_b: str, model: HookedTransformer) -> dict:
    tokens_a = model.to_tokens(prompt_a)
    tokens_b = model.to_tokens(prompt_b)
    if tokens_a.shape[1] != tokens_b.shape[1]:
        raise ValueError(
            "Prompts must tokenize to the same length.\n"
            f"  prompt_a ({tokens_a.shape[1]} tokens): {model.to_str_tokens(prompt_a)}\n"
            f"  prompt_b ({tokens_b.shape[1]} tokens): {model.to_str_tokens(prompt_b)}"
        )

    with torch.no_grad():
        _, cache_a = model.run_with_cache(tokens_a)
        _, cache_b = model.run_with_cache(tokens_b)

    W_U = model.W_U
    b_U = getattr(model, "b_U", None)
    n_layers = model.cfg.n_layers

    layers: list[int] = []
    kl_symmetric: list[float] = []
    cosine_distance: list[float] = []
    top_token_a: list[str] = []
    top_prob_a: list[float] = []
    top_token_b: list[str] = []
    top_prob_b: list[float] = []

    with torch.no_grad():
        for i in range(n_layers):
            resid_a = cache_a[f"blocks.{i}.hook_resid_post"][0, -1, :]
            resid_b = cache_b[f"blocks.{i}.hook_resid_post"][0, -1, :]

            normed_a = model.ln_final(resid_a.unsqueeze(0)).squeeze(0)
            normed_b = model.ln_final(resid_b.unsqueeze(0)).squeeze(0)

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
            kl_symmetric.append((kl_ab + kl_ba) / 2)

            cos = F.cosine_similarity(resid_a.unsqueeze(0), resid_b.unsqueeze(0), dim=-1).item()
            cosine_distance.append(1.0 - cos)

            pa_top, ia_top = p_a.max(dim=-1)
            pb_top, ib_top = p_b.max(dim=-1)
            top_token_a.append(model.to_string([ia_top.item()]))
            top_prob_a.append(pa_top.item())
            top_token_b.append(model.to_string([ib_top.item()]))
            top_prob_b.append(pb_top.item())

            layers.append(i)

    return {
        "layers": layers,
        "kl_symmetric": kl_symmetric,
        "cosine_distance": cosine_distance,
        "top_token_a": top_token_a,
        "top_prob_a": top_prob_a,
        "top_token_b": top_token_b,
        "top_prob_b": top_prob_b,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "tokens_a": model.to_str_tokens(prompt_a),
        "tokens_b": model.to_str_tokens(prompt_b),
    }


def _print_table(result: dict) -> None:
    print(f"prompt A: {result['prompt_a']!r}")
    print(f"prompt B: {result['prompt_b']!r}")
    print(f"tokens A: {result['tokens_a']}")
    print(f"tokens B: {result['tokens_b']}\n")

    header = f"{'Layer':>5} | {'KL sym':>8} | {'Cos dist':>8} | {'Top A':<24} | {'Top B':<24}"
    print(header)
    print("-" * len(header))
    for i, kl, cd, ta, pa, tb, pb in zip(
        result["layers"],
        result["kl_symmetric"],
        result["cosine_distance"],
        result["top_token_a"],
        result["top_prob_a"],
        result["top_token_b"],
        result["top_prob_b"],
    ):
        a_fmt = f"{ta!r} ({pa:.3f})"
        b_fmt = f"{tb!r} ({pb:.3f})"
        print(f"{i:>5} | {kl:>8.4f} | {cd:>8.4f} | {a_fmt:<24} | {b_fmt:<24}")


if __name__ == "__main__":
    model = load_model("gpt2-medium")
    print(f"device: {next(model.parameters()).device}\n")

    result = compute_divergence(
        "The capital of France is the city of",
        "The capital of Italy is the city of",
        model,
    )
    _print_table(result)
