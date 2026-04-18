"""Sweep single-head ablations across layers 12..20 on gpt2-medium and rank
them by how much they damage p(' Paris') at the final position for
'The capital of France is the city of'. One-off diagnostic; not wired into
the endpoint path."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://localhost:8000/trace"
PROMPT = "The capital of France is the city of"
TARGET = " Paris"
LAYERS = range(12, 21)  # inclusive of 20
HEADS = range(16)


def post_trace(interventions: list[dict]) -> dict:
    body = json.dumps({"prompt": PROMPT, "interventions": interventions}).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def paris_prob(interventions: list[dict]) -> float:
    d = post_trace(interventions)
    for pred in d["final_top_k"][-1]:
        if pred["token"] == TARGET:
            return float(pred["prob"])
    return 0.0  # Paris fell out of the top-k entirely.


def main() -> None:
    baseline = paris_prob([])
    print(f"baseline p('{TARGET}') = {baseline:.4f}\n")

    results: list[tuple[int, int, float, float]] = []
    for layer in LAYERS:
        for head in HEADS:
            p = paris_prob(
                [{"type": "zero_head", "layer": layer, "head": head}]
            )
            drop = baseline - p
            results.append((layer, head, p, drop))
            print(f"  L{layer:>2} H{head:>2}: p={p:.4f}  drop={drop:+.4f}")

    results.sort(key=lambda r: -r[3])
    print("\ntop 10 most-damaging single-head ablations:")
    print(f"  {'layer':>5} {'head':>4} {'p(Paris)':>10} {'drop':>10}")
    for layer, head, p, drop in results[:10]:
        marker = "  *" if drop > 0.1 else ""
        print(f"  {layer:>5} {head:>4} {p:>10.4f} {drop:>+10.4f}{marker}")


if __name__ == "__main__":
    main()
