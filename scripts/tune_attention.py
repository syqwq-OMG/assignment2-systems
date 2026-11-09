"""Bounded, explicit attention tile experiment; at most four modest configurations.

No automatic large-shape sweep. Results are observations, not production dispatch.
"""

import argparse
import json

import torch
import triton
import triton.testing

from cs336_systems.flash_attention import _backward_kv, _backward_q, _delta, _forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--head-dim", type=int, choices=[100, 128], default=100)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--sum-gradient", action="store_true")
    p.add_argument("--preset", choices=["wide", "small", "confirm"], default="wide")
    a = p.parse_args()
    if not 128 <= a.length <= 4096 or not 1 <= a.heads <= 4:
        p.error("This development experiment is limited to N=128..4096 and H=1..4")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    shape = (a.heads, a.length, a.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)]
    out, dq, dk, dv = [torch.empty_like(q) for _ in range(4)]
    do = torch.ones((), device="cuda", dtype=q.dtype).expand(shape) if a.sum_gradient else torch.randn_like(q)
    lse = torch.empty(shape[:2], device=q.device, dtype=torch.float32)
    delta = torch.empty_like(lse)
    n, d, h = a.length, a.head_dim, a.heads
    bd = triton.next_power_of_2(d)
    rows = []
    candidates = [(16, 32, 4, 1), (32, 32, 4, 1), (32, 32, 8, 1), (32, 64, 8, 1)] if a.preset == "small" else [(32, 64, 4, 2), (64, 64, 4, 2), (64, 64, 8, 2), (128, 64, 8, 2)]
    if a.preset == "confirm":
        candidates = [(32, 64, 4, 2), (32, 32, 4, 1)]
    for bm, bn, warps, stages in candidates:
        fargs = (q, k, v, out, lse, n, n, d, bd, bm, bn, False, d**-0.5)
        bargs = (n, n, d, *do.stride(), bd, bm, bn, False, d**-0.5)
        launches = {
            "forward": lambda bm=bm, fargs=fargs, warps=warps, stages=stages: _forward[(triton.cdiv(n, bm), h)](*fargs, num_warps=warps, num_stages=stages),
            "delta": lambda: _delta[(triton.cdiv(n, 32), h)](out, do, delta, n, d, *do.stride(), bd, 32, num_warps=4),
            "dq": lambda bm=bm, bargs=bargs, warps=warps, stages=stages: _backward_q[(triton.cdiv(n, bm), h)](
                q, k, v, do, lse, delta, dq, *bargs, num_warps=warps, num_stages=stages
            ),
            "dkv": lambda bn=bn, bargs=bargs, warps=warps, stages=stages: _backward_kv[(triton.cdiv(n, bn), h)](
                q, k, v, do, lse, delta, dk, dv, *bargs, num_warps=warps, num_stages=stages
            ),
        }
        result = {"tile": [bm, bn, warps, stages], "kernels": {}}
        for name, launch in launches.items():
            kernel = launch()
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(launch, warmup=5, rep=20, return_mode="median")
            result["kernels"][name] = {"ms": ms, "registers": kernel.n_regs, "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared}
        rows.append(result)
        print(json.dumps({"settings": vars(a), "result": result}), flush=True)
    print(json.dumps({"settings": vars(a), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
