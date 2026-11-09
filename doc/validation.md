# Validation record, 2026-11-08

This is the original CPU-only record. See [the November 9 GPU report](gpu_optimization_2026-11-09.md) for subsequent CUDA validation and benchmark optimization.

Implementation and verification were performed without CUDA kernel execution.
The server's GPU was in use, and Sun Yuquan's reservation starts November 9.
All remote writes were confined to `/ai/lmfs2026/sunyuquan/assignment2-codex`.

## Exact server environment

- Python 3.13, PyTorch 2.11.0+cu128, Triton 3.6.0.
- Observed hardware: NVIDIA A800 80GB PCIe, Ampere compute capability 8.0.
- Existing shared environment: `/lmfs2026/env/a1`; no dependency modifications.
- CPU runs: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.
- Tests import Basics from `/lmfs2026/data/assignment2-systems/cs336-basics` read-only.

## Results

| Check | Result |
| --- | --- |
| Full test suite, CPU, exact server runtime | **23 passed, 13 skipped**, 34.48 seconds |
| Attention edge tests, Triton CPU interpreter | **14 passed, 1 skipped**, 2.77 seconds |
| Ruff over implementation, scripts, adapters, added tests | Passed |
| `git diff --check` | Passed |
| Tiny model training with grouped checkpointing | Passed |
| Tiny recursive-checkpoint BF16 CPU backward and Chrome trace | Passed on development Mac |
| Tiled-reference attention timing, N=33, D=17 | Passed on server |
| Two-rank Gloo all-reduce, 1 MB, one measured step | Passed |
| Two-rank CPU BF16 FSDP training, one measured step | Passed |
| Two-rank CPU overlapped DDP + sharded AdamW, one measured step | Passed |

The full CPU suite skips the four provided CUDA attention cases and nine added
Triton cases. Interpreter mode executes those nine added cases except BF16.
Triton's documented BF16 interpreter limitation is the reason for that skip;
BF16 device numerical correctness is not established. Framework deprecation
warnings about `torch.jit.script_method` were emitted by the server runtime.

The tests cover forward/LSE, arbitrary-gradient backward, summed-loss backward,
causal/noncausal attention, uneven sequence tails, rectangular query/key counts,
D=17/100 padding, noncontiguous inputs, tied weights, frozen parameters,
multi-step optimizer equivalence, dynamic optimizer groups, empty ownership,
checkpoint restore, dropout RNG preservation, odd-size FSDP shards, and compiled
backward graph capture. The compiled graph regression uses the eager backend;
it does not benchmark Inductor.

A private minimal Mac environment used PyTorch 2.15. Its `torchrun --standalone`
rendezvous stalled before worker startup and was terminated. The corresponding
Linux server collective and distributed model smoke commands completed. The
specified-runtime verification results above come from the server, not that
newer Mac runtime.

## Offline CUDA compilation

Explicit `GPUTarget('cuda',80,32)` compilation succeeded for all four kernels in
these configurations. The compiler was invoked with CUDA hidden; kernels were
not launched and tensors of the declared large sequence shape were not allocated.

| Type / D / mask / upstream gradient | Forward shared bytes | Delta shared bytes | dQ shared bytes | dK/dV shared bytes |
| --- | ---: | ---: | ---: | ---: |
| FP16 / 100 / noncausal / zero strides | 32768 | 8192 | 32768 | 32768 |
| FP32 / 128 / causal / contiguous | 32768 | 512 | 34816 | 32768 |
| BF16 / 128 / noncausal / contiguous | 32768 | 512 | 32768 | 32768 |

Compilation verifies code generation for the target, not numerical accuracy,
occupancy, register spilling, race freedom, throughput, or device peak memory.

## Still requiring the reserved GPU session

1. Small CUDA correctness cases, including BF16 and arbitrary upstream gradients.
2. Full course A/B/C comparisons against original FP32 references and the worst
   absolute-error coefficient, beginning with A only.
3. Measured device peak memory and single-configuration efficiency.
4. NCCL multi-GPU execution and trace-based overlap validation on suitable hardware.
5. Empirical assignment writeup tables, memory screenshots, and any leaderboard work.

The declared FP16 input choice and 64-bit address arithmetic address analytical
memory/indexing hazards; no full-size accuracy or performance claim is made.
No formal shared-directory course submission has been made by this session.
