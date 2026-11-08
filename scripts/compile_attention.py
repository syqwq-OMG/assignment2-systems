"""Compile Ampere kernels without a CUDA context, allocation, or kernel launch.

Run on Linux with Triton installed: CUDA_VISIBLE_DEVICES='' python -m scripts.compile_attention
This checks code generation/resource metadata, not numerical device correctness.
"""

import argparse

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from cs336_systems.flash_attention import _backward_kv, _backward_q, _delta, _forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--head-dim", type=int, default=100)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--sum-gradient", action="store_true", help="Compile zero-stride upstream gradient")
    args = parser.parse_args()
    d = args.head_dim
    n = 131072
    common = {
        "NQ": n,
        "NK": n,
        "N": n,
        "D": d,
        "BD": max(16, triton.next_power_of_2(d)),
        "BM": 32,
        "BN": 64,
        "CAUSAL": args.causal,
        "SCALE": d**-0.5,
        "SH": 0 if args.sum_gradient else n * d,
        "SN": 0 if args.sum_gradient else d,
        "SD": 0 if args.sum_gradient else 1,
    }
    for fn in (_forward, _delta, _backward_q, _backward_kv):
        constants = {name: common[name] for name in fn.arg_names if name in common}
        pointers = {name: "*fp32" if name in ("L", "DELTA") else "*" + args.dtype for name in fn.arg_names if name not in constants}
        kernel = triton.compile(ASTSource(fn, pointers, constexprs=constants), target=GPUTarget("cuda", 80, 32), options={"num_warps": 4, "num_stages": 2})
        print(f"{fn.__name__}: shared={kernel.metadata.shared} bytes", flush=True)


if __name__ == "__main__":
    main()
