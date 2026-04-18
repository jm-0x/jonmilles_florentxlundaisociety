"""Sweep single-head Italy → France patches across layers 12..20.
Rank by p(' Rome') after patching. One-off diagnostic."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://localhost:8000/trace"
TARGET = "The capital of France is the city of"
SOURCE = "The capital of Italy is the city of"
LAYERS = range(12, 21)
HEADS = range(16)


def post_trace(interventions: list[dict]) -> dict:
    body = json.dumps({"prompt": TARGET, "interventions": interventions}).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def prob(final_top5: list, token: str) -> float:
    for p in final_top5:
        if p["token"] == token:
            return float(p["prob"])
    return 0.0


def main() -> None:
    baseline = post_trace([])
    base_paris = prob(baseline["final_top_k"][-1], " Paris")
    base_rome = prob(baseline["final_top_k"][-1], " Rome")
    print(f"baseline (France, clean)  p(' Paris')={base_paris:.4f}  p(' Rome')={base_rome:.4f}")
    print(f"patching from: {SOURCE!r}\n")

    results: list[tuple[int, int, float, float]] = []
    for layer in LAYERS:
        for head in HEADS:
            iv = {
                "type": "patch_head",
                "layer": layer,
                "head": head,
                "source_prompt": SOURCE,
            }
            d = post_trace([iv])
            final = d["final_top_k"][-1]
            p_paris = prob(final, " Paris")
            p_rome = prob(final, " Rome")
            results.append((layer, head, p_paris, p_rome))
            print(
                f"  L{layer:>2} H{head:>2}: p(Paris)={p_paris:.4f}  p(Rome)={p_rome:.4f}"
            )

    results.sort(key=lambda r: -r[3])  # by p(' Rome') descending
    print("\ntop 10 most-effective patches (ranked by p(' Rome')):")
    print(f"  {'layer':>5} {'head':>4} {'p(Paris)':>10} {'p(Rome)':>10} {'ΔParis':>10}")
    for layer, head, p_paris, p_rome in results[:10]:
        dp = p_paris - base_paris
        print(f"  {layer:>5} {head:>4} {p_paris:>10.4f} {p_rome:>10.4f} {dp:>+10.4f}")


if __name__ == "__main__":
    main()
