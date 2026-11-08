"""One bounded model profiling run; tiny CPU defaults and no automated sweeps.

Use --help for model, precision, compile, checkpoint, and distributed options.
Distributed mode is launched with torchrun, not by this script spawning GPUs.
"""

import argparse
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM

from .checkpointing import checkpoint_layers
from .distributed import DDP, ShardedOptimizer
from .fsdp import FSDP


class CheckpointedLM(BasicsTransformerLM):
    def forward(self, x):
        x = self.token_embeddings(x)
        x = checkpoint_layers(self.layers, x, self.checkpoint_group, self.recursive_checkpoint)
        return self.lm_head(self.ln_final(x))


def annotated_attention(Q, K, V, mask=None):
    """Drop-in reference attention annotated at its three expensive operations."""
    with torch.autograd.profiler.record_function("attention_scores"):
        s = Q @ K.transpose(-1, -2) * Q.shape[-1] ** -0.5
        if mask is not None:
            s = s.masked_fill(~mask, -torch.inf)
    with torch.autograd.profiler.record_function("attention_softmax"):
        p = s.softmax(-1)
    with torch.autograd.profiler.record_function("attention_values"):
        return p @ V


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--batch", type=int, default=2, help="Local batch per rank")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--mode", choices=["forward", "backward", "train"], default="train")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--checkpoint-group", type=int, default=0)
    p.add_argument("--recursive-checkpoint", action="store_true")
    p.add_argument("--attention", choices=["reference", "triton"], default="reference")
    p.add_argument("--distributed", choices=["none", "individual", "flat", "overlap", "fsdp"], default="none")
    p.add_argument("--sharded-optimizer", action="store_true")
    p.add_argument("--memory-snapshot", type=Path)
    p.add_argument("--trace", type=Path, help="Chrome trace output; profiling affects timings")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not (1 <= a.steps <= 100 and 0 <= a.warmup <= 20):
        p.error("Use 1..100 measurement steps and 0..20 warmup steps per invocation")
    if min(a.batch, a.context, a.layers, a.heads, a.d_model, a.d_ff, a.vocab) <= 0 or a.d_model % a.heads:
        p.error("Dimensions must be positive and d-model divisible by heads")
    if a.sharded_optimizer and a.distributed in ("none", "fsdp"):
        p.error("Sharded optimizer is for DDP modes; FSDP already shards parameters")
    device = torch.device(a.device)
    if a.distributed != "none":
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(0)
    import cs336_basics.model as basics

    if a.attention == "triton":
        from .flash_attention import FlashAttention2

        def flash(Q, K, V, mask=None):
            return FlashAttention2.apply(Q, K, V, True)

        basics.scaled_dot_product_attention = flash
    else:
        basics.scaled_dot_product_attention = annotated_attention
    model = CheckpointedLM(vocab_size=a.vocab, context_length=a.context, d_model=a.d_model, num_layers=a.layers, num_heads=a.heads, d_ff=a.d_ff).to(device)
    model.checkpoint_group, model.recursive_checkpoint = a.checkpoint_group, a.recursive_checkpoint
    if a.distributed == "fsdp":
        model = FSDP(model, torch.bfloat16 if a.precision == "bf16" else None)
    elif a.distributed != "none":
        model = DDP(model, a.distributed)
    parallel_model = model
    if a.compile:
        model = torch.compile(model)
    optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3) if a.sharded_optimizer else torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(123 + rank)
    inputs = torch.randint(a.vocab, (a.batch, a.context), device=device)
    targets = torch.randint(a.vocab, inputs.shape, device=device)
    phase_times = {key: [] for key in ("forward", "backward", "communication_wait", "optimizer", "total")}
    memory = {}

    def allocated(label):
        if device.type == "cuda":
            memory[label] = {"allocated": torch.cuda.memory_allocated(device), "peak_allocated": torch.cuda.max_memory_allocated(device)}

    def timed_step(measure):
        optimizer.zero_grad(set_to_none=True)
        if dist.is_initialized():
            dist.barrier()
        sync(device)
        begin = perf_counter()
        last = begin

        def mark(phase):
            nonlocal last
            sync(device)
            now = perf_counter()
            if measure:
                phase_times[phase].append((now - last) * 1000)
            last = now

        amp = torch.autocast(device.type, dtype=torch.bfloat16) if a.precision == "bf16" else nullcontext()
        with torch.set_grad_enabled(a.mode != "forward"), amp, torch.autograd.profiler.record_function("forward"):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        mark("forward")
        if a.mode != "forward":
            with torch.autograd.profiler.record_function("backward"):
                loss.backward()
            mark("backward")
            if a.distributed != "none":
                parallel_model.finish_gradient_synchronization()
            mark("communication_wait")
            allocated("before_optimizer")
            if a.mode == "train":
                with torch.autograd.profiler.record_function("optimizer"):
                    optimizer.step()
                mark("optimizer")
                allocated("after_optimizer")
        if measure:
            phase_times["total"].append((perf_counter() - begin) * 1000)

    allocated("initialization")
    for _ in range(a.warmup):
        timed_step(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if a.memory_snapshot:
        if device.type != "cuda":
            p.error("Memory snapshots require CUDA")
        torch.cuda.memory._record_memory_history(max_entries=100000)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) if a.trace else nullcontext()
    # NVTX can be captured with nsys --capture-range=nvtx --nvtx-capture=measurement.
    nvtx = torch.cuda.nvtx.range("measurement") if device.type == "cuda" else nullcontext()
    with profiler as prof, nvtx:
        for _ in range(a.steps):
            timed_step(True)
    if a.trace:
        path = a.trace.with_name(f"{a.trace.stem}.rank{rank}{a.trace.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
    if a.memory_snapshot:
        path = a.memory_snapshot.with_name(f"{a.memory_snapshot.stem}.rank{rank}{a.memory_snapshot.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)
    result = {
        "rank": rank,
        "settings": vars(a),
        "milliseconds": {key: {"mean": statistics.mean(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0} for key, values in phase_times.items() if values},
        "memory_bytes": memory,
    }
    results = [result]
    if dist.is_initialized():
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result)
    if rank == 0:
        output = json.dumps(results, indent=2, default=str)
        print(output)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(output + "\n")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
