"""Measure one small attention configuration; never launches a shape sweep."""

import argparse
import json
from time import perf_counter

import torch


def dense_attention(q, k, v, causal):
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if causal:
        index = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(index[:, None] < index[None, :], -torch.inf)
    return scores.softmax(-1) @ v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--implementation", choices=["dense", "compiled", "pytorch", "triton"], default="dense")
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup-ms", type=int, default=5)
    p.add_argument("--rep-ms", type=int, default=10)
    a = p.parse_args()
    if min(a.length, a.head_dim, a.batch) < 1 or not (0 <= a.warmup_ms <= 100 and 1 <= a.rep_ms <= 100):
        p.error("Positive dimensions; 0..100 warmup ms; 1..100 repetition ms")
    device = torch.device(a.device)
    torch.manual_seed(0)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    inputs = [torch.randn(a.batch, a.length, a.head_dim, device=device, dtype=dtype).requires_grad_() for _ in range(3)]
    if a.implementation == "triton":
        from .flash_attention import FlashAttention2

        fn = FlashAttention2.apply
    elif a.implementation == "pytorch":
        from .attention import FlashAttentionPytorch

        fn = FlashAttentionPytorch.apply
    else:
        fn = torch.compile(dense_attention) if a.implementation == "compiled" else dense_attention

    def forward():
        return fn(*inputs, a.causal)

    output = forward()
    do = torch.randn_like(output)
    saved_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else None

    def backward():
        for t in inputs:
            t.grad = None
        output.backward(do, retain_graph=True)

    def combined():
        for t in inputs:
            t.grad = None
        forward().backward(do)

    def bench(call):
        call()  # compile/warm before timing
        if device.type == "cuda":
            import triton.testing

            torch.cuda.synchronize(device)
            return triton.testing.do_bench(call, warmup=a.warmup_ms, rep=a.rep_ms)
        samples = []
        for _ in range(2):
            start = perf_counter()
            call()
            samples.append((perf_counter() - start) * 1000)
        return sum(samples) / len(samples)

    result = {"settings": vars(a), "allocated_before_backward_bytes": saved_bytes, "forward_ms": bench(forward), "backward_ms": bench(backward)}
    # Free the retained graph before end-to-end measurement and peak accounting.
    output = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result["forward_backward_ms"] = bench(combined)
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
