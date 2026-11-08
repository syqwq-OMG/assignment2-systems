"""Single all-reduce size, launched with torchrun; tiny CPU defaults."""

import argparse
import json
import os
import statistics
from time import perf_counter

import torch
import torch.distributed as dist

from .benchmark import sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--megabytes", type=float, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    a = p.parse_args()
    if a.megabytes <= 0 or not 1 <= a.steps <= 100 or not 0 <= a.warmup <= 20:
        p.error("Positive size; 1..100 steps; 0..20 warmup steps")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if a.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if a.device == "cuda" else "gloo")
    data = torch.empty(max(1, int(a.megabytes * 1_000_000 / 4)), device=device)
    samples = []
    for i in range(a.warmup + a.steps):
        data.fill_(1)  # prevent repeated sum overflow
        dist.barrier()
        sync(device)
        start = perf_counter()
        dist.all_reduce(data)
        sync(device)
        if i >= a.warmup:
            samples.append((perf_counter() - start) * 1000)
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, {"rank": dist.get_rank(), "bytes": data.numel() * 4, "mean_ms": statistics.mean(samples), "samples_ms": samples})
    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
