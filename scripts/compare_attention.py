"""Controlled baseline/current comparison on one bounded GPU configuration."""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import triton.testing

from cs336_systems.flash_attention import FlashAttention2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, choices=[100, 128], default=100)
    p.add_argument("--sum-gradient", action="store_true")
    a = p.parse_args()
    if not 128 <= a.length <= 16384 or not 1 <= a.heads <= 24:
        p.error("Bounded comparison supports N=128..16384, H=1..24 only")
    spec = importlib.util.spec_from_file_location("attention_baseline", a.baseline)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    classes = {"baseline": module.FlashAttention2, "optimized": FlashAttention2}
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    shape = (a.heads, a.length, a.head_dim)
    inputs = [torch.randn(shape, device="cuda", dtype=torch.float16).requires_grad_() for _ in range(3)]
    do = torch.ones((), device="cuda", dtype=torch.float16).expand(shape) if a.sum_gradient else torch.randn_like(inputs[0])
    observations = {}
    reference = None
    for name, cls in classes.items():

        def combined(impl=cls):
            for t in inputs:
                t.grad = None
            output = impl.apply(*inputs)
            output.backward(do)
            return output

        output = combined()
        torch.cuda.synchronize()
        values = [output.detach()] + [t.grad.detach() for t in inputs]
        if reference is None:
            reference = [t.clone() for t in values]
        errors = {key: (actual.float() - expected.float()).abs().max().item() for key, actual, expected in zip(("o", "dq", "dk", "dv"), values, reference)}
        del output, values
        torch.cuda.reset_peak_memory_stats()
        # Three short trials, compilation excluded; same inputs/upstream gradient.
        samples = [triton.testing.do_bench(combined, warmup=10, rep=30, return_mode="median") for _ in range(3)]
        observations[name] = {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_difference_from_baseline": errors,
        }
    print(
        json.dumps(
            {
                "settings": vars(a),
                "gpu": torch.cuda.get_device_name(),
                "results": observations,
                "speedup": observations["baseline"]["median_ms"] / observations["optimized"]["median_ms"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
