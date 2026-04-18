"""Smoke test: load GPT-2 small via TransformerLens, cache activations on two
prompts, and confirm predictions look sensible. No UI, no abstractions."""

import torch
from transformer_lens import HookedTransformer

PROMPTS = [
    "The capital of France is the city of",
    "The capital of Italy is the city of",
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}\n")

    model = HookedTransformer.from_pretrained("gpt2-medium")
    model.to(device)
    model.eval()
    print(f"model loaded on: {next(model.parameters()).device}\n")

    last_cache = None
    for prompt in PROMPTS:
        with torch.no_grad():
            logits, cache = model.run_with_cache(prompt)
        last_cache = cache

        # logits: [batch=1, seq, vocab] — take last position
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5)

        print(f"prompt: {prompt!r}")
        for rank, (p, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
            tok = model.to_string([tid])
            print(f"  {rank}. {tok!r:<14} p={p:.4f}")
        print()

    # Inspect the activation cache from the last prompt so we know what's queryable.
    assert last_cache is not None
    keys = list(last_cache.keys())
    print(f"cache: {len(keys)} keys")
    print(f"first 5 keys: {keys[:5]}")
    sample_key = "blocks.0.hook_resid_pre"
    if sample_key in last_cache:
        t = last_cache[sample_key]
        print(f"{sample_key}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    else:
        # Fall back to the first key if the expected one is absent.
        k = keys[0]
        t = last_cache[k]
        print(f"{k}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")


if __name__ == "__main__":
    main()
