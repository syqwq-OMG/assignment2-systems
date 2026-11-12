# Assignment 2: implementation and learning guide

**November 9 follow-up:** see [the real-GPU optimization report](gpu_optimization_2026-11-09.md) for 29 passing CUDA tests, an 8.63x A-sized speedup, full-A accuracy at the 1e-2 tier, and bounded B/C head checks. The original November 8 verification record below is historical.

**November 13 submission:** the standalone file is now in
`/ai/lmfs2026/data/assignment2/sunyuquan.py`, and the personal working directory
is `/ai/lmfs2026/sunyuquan/assignment2`. See [the submission audit](submission_status.md)
for the completion assessment and remaining full B/C evaluation.

## 1. Scope and completion status

This guide accompanies the code implementation, not a completed empirical writeup. Sources are the repository `README.md` (there is no `readmd.md`), the 48-page `cs336_assignment2_systems.pdf`, and both original files under `doc/`: `FA2.md` and `login_method.md`.

The ECNU course task in `FA2.md` takes priority for the standalone submission: Triton forward **and backward**, noncausal attention, `[heads, sequence, head_dim]` inputs, `FlashAttention2.apply(q,k,v)`, and no full attention matrix. The upstream Stanford assignment additionally asks for profiling, checkpointing, DDP, optimizer sharding, and FSDP. Those implementations and small profiling entry points are included here as well.

| Requirement | Implementation |
| --- | --- |
| Standalone course submission | `cs336_systems/flash_attention.py` |
| Tiled PyTorch reference and optional compiled backward | `cs336_systems/attention.py` |
| Naive, flat, and overlapped DDP | `cs336_systems/distributed.py`: `DDP(mode=...)` |
| Optimizer state sharding | `cs336_systems/distributed.py`: `ShardedOptimizer` |
| Layer-wise FSDP, optional compute precision | `cs336_systems/fsdp.py` |
| Flat and recursive activation checkpointing | `cs336_systems/checkpointing.py` |
| Model timing, autocast, compile, NVTX, memory, traces | `python -m cs336_systems.benchmark` |
| Dense/compiled/tiled/Triton attention timing | `python -m cs336_systems.benchmark_attention` |
| All-reduce timing | `python -m cs336_systems.benchmark_collective` |
| Compilation with no GPU execution | `python -m scripts.compile_attention` |
| Test integration | `tests/adapters.py` |

No shared evaluator/data files or shared dependencies were modified. The November 8 implementation was verified on CPU only. On November 9, all 29 CUDA attention tests passed, including BF16; bounded GPU benchmarks and full-A/partial-B/C reference checks were performed. Full B/C evaluation, official efficiency rankings, NCCL execution, and leaderboard results remain unmeasured; see the GPU report for precise coverage and the measured 1e-2 accuracy tier on full A. The B200 leaderboard is a separate optimization exercise, not a performance result supplied by this implementation. The upstream `torch.compile` backward exercise is available as `FlashAttentionCompiledBackward` and attention benchmark `--implementation compiled-backward`; the standalone course submission uses the explicit tiled Triton backward kernels. The ordinary PyTorch reference stays eager for transparent debugging. Full graph capture of the tensor backward is tested with the eager compiler backend; optimized Inductor timing remains unmeasured.

## 2. Step-by-step reproduction

### Step 1: inspect and protect existing work

```bash
git status --short
```

Do not discard existing changes. This session began with user deletions of `AGENTS.md`/`CLAUDE.md` and untracked course notes. Stage only files you intentionally change. Commit and push after attention, distributed implementation, and final verification milestones. The original notes are left untouched.

### Step 2: connect to the correct server

Connect to the university network/VPN, then:

```bash
ssh sunyuquan@172.23.166.144 -p 32343 -i ~/.ssh/lmfs_class_key
source /lmfs2026/env/a1/bin/activate
python -c 'import torch,triton; print(torch.__version__,torch.version.cuda,triton.__version__)'
```

The note spells the key path `lmfs_class_ke`; the existing local key is actually `lmfs_class_key`. Never copy a private key into the repository. The assignment-1 SSH port is different.

The inspected server runs PyTorch 2.11.0+cu128 and Triton 3.6.0, and exposes one NVIDIA A800 80GB PCIe. The course note says A100; use the actual machine when interpreting measurements. At 2026-11-08 16:05 CST the GPU was occupied. The timetable assigns Sun Yuquan 2026-11-09 14:00 for 24 hours. Check the current schedule and machine state before a GPU run.

Use the persistent personal disk:

```bash
mkdir -p /ai/lmfs2026/sunyuquan/assignment2
cd /ai/lmfs2026/sunyuquan/assignment2
```

This is the personal working copy used during development. `/lmfs2026/data/assignment2-systems` and `/lmfs2026/data/2-fa2-data` are shared read-only inputs. Do not run package installation commands against `/lmfs2026/env/a1`.

### Step 3: run CPU correctness tests first

On the server, after copying this repository to your personal directory:

```bash
source /lmfs2026/env/a1/bin/activate
export PYTHONPATH="$PWD/cs336-basics:$PWD"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest tests -q
```

CUDA visibility must be set **before** importing PyTorch: the provided distributed test helper otherwise chooses a GPU automatically. The suite uses two Gloo processes and small toy models. The CUDA-only attention cases should skip in this command.

For a private local environment, `uv run pytest` uses the repository's declared dependencies. On macOS, CUDA/Triton execution is unavailable; the CPU reference and Gloo tests still run. The development Mac environment was a private minimal environment and had PyTorch 2.15; server CPU verification is authoritative for the specified PyTorch 2.11 version.

### Step 4: execute the kernels in Triton's CPU interpreter

Linux with Triton installed:

```bash
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest tests/test_attention_edges.py -q
```

This runs the kernel operations through NumPy without using CUDA. Cases cover odd query/key counts, rectangular attention, D=17/100 padding, causal masks, sliced input storage, and summed-loss gradients in FP16 and FP32. BF16 is skipped in interpreter mode because Triton documents BF16 arithmetic as unsupported there. The source uses `while` loops, avoiding an interpreter scalar-index conversion incompatibility with recent NumPy versions.

Interpreter execution checks arithmetic and indexing at small sizes. It cannot prove device race freedom, tensor-core rounding, GPU resource use, or speed.

### Step 5: compile for Ampere without launching a kernel

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.compile_attention --dtype fp16 --head-dim 100 --sum-gradient
CUDA_VISIBLE_DEVICES='' python -m scripts.compile_attention --dtype fp32 --head-dim 128 --causal
CUDA_VISIBLE_DEVICES='' python -m scripts.compile_attention --dtype bf16 --head-dim 128
```

The target is explicitly CUDA compute capability 8.0, so these commands need the Triton toolchain but no CUDA allocation/context. All four kernels compile. Compilation is an additional check, not GPU numerical validation. The script reports static shared-memory metadata; it does not measure runtime or tune configurations.

### Step 6: during your slot, perform small device checks

Use these small checks to reproduce the GPU validation; the November 9 report records the commands and coverage actually exercised:

```bash
OMP_NUM_THREADS=2 python -m pytest tests/test_attention.py tests/test_attention_edges.py -q
python -m cs336_systems.benchmark_attention --device cuda --implementation triton \
  --length 128 --head-dim 100 --dtype fp16 --warmup-ms 5 --rep-ms 10
```

Start with correctness, then one small timing. Compare output and **all three gradients** against the dense FP32 reference. Testing only `output.sum()` is insufficient: arbitrary upstream gradients are needed to catch derivative mistakes. Check both causal modes for Stanford compatibility even though ECNU uses noncausal attention.

Only after small device tests pass should the full course evaluator be considered. Do not invoke it as part of a development smoke test: configurations B/C are very expensive even with linear auxiliary memory. Locate the official evaluator read-only, choose a single configuration (`--index 0` first), and use `--mode accuracy` before efficiency. The evaluator's default 1000ms warmup/10000ms repetition settings are not suitable for routine shared-GPU development. Do not automate a sweep over A/B/C.

### Step 7: prepare the standalone course file

The file is self-contained; it imports only Python standard-library modules, PyTorch, and Triton:

```bash
cp cs336_systems/flash_attention.py /ai/lmfs2026/sunyuquan/assignment2/sunyuquan.py
```

The evaluator can load it with `importlib` and call `FlashAttention2.apply(q,k,v)`. It also accepts an optional fourth causal flag for the upstream adapters. `INPUT_DTYPE=torch.float16` controls evaluator input conversion; the function itself supports FP32, FP16, and BF16 CUDA tensors. The reviewed file was placed at the course's stated shared hand-in location `/ai/lmfs2026/data/assignment2/sunyuquan.py` on November 13. Its checksum matches the implementation and the local submission snapshot. See the submission audit for remaining evaluation coverage.

For the upstream assignment, run tests directly and inspect the exit status before using `test_and_make_submission.sh`. The supplied packaging script intentionally has `pytest ... || true`, so producing a zip does not imply passing tests. Empirical written answers and screenshots belong in the later writeup; do not invent them from analytical expectations.

## 3. Attention: the mathematical foundation

For one head, let Q have shape `[Nq,D]`, K/V `[Nk,D]`. Define:

```text
S = Q K^T / sqrt(D)             [Nq,Nk]
P[i,j] = exp(S[i,j]) / sum_j exp(S[i,j])
O = P V                        [Nq,D]
```

Q asks what a token is looking for, K describes what a token offers, and V carries the information to mix. The scale `1/sqrt(D)` prevents dot-product variance from increasing with head dimension when features have roughly unit variance. Softmax normalizes independently across keys for each query. Causal masking permits only `j <= i`; noncausal attention permits all real keys.

Dense attention stores scores or probabilities proportional to `H*Nq*Nk`. This becomes prohibitive with long sequences. FlashAttention computes the same attention up to floating-point rounding, but stores only output, inputs, and one normalization scalar per query. Its arithmetic still scales quadratically with sequence length: reducing memory does not eliminate the all-pairs computation.

### Online softmax, derived

For a query row and the keys processed so far, maintain:

```text
m = maximum score seen so far
ell = sum_j exp(score_j - m)
a = sum_j exp(score_j - m) * V_j
```

When a new score tile arrives, let `m_new=max(m,max(tile))`. Re-express the old sums relative to this new maximum:

```text
alpha = exp(m - m_new)
p_tile = exp(score_tile - m_new)
a_new = alpha*a + p_tile @ V_tile
ell_new = alpha*ell + sum(p_tile)
```

Start with `m=-inf`, `ell=0`, `a=0`. Finish with `O=a/ell` and `L=m+log(ell)`. Subtracting the largest score keeps exponentials bounded by one. Merely applying softmax separately to each tile and adding tile outputs is wrong: each tile would have a different denominator. Rescaling old sums is the step that makes streaming exact in real arithmetic.

`L` is natural-log logsumexp, not base-2 logsumexp. Reconstruct probabilities later as `exp(S-L)`; changing to `exp2` for performance requires converting scores and normalizers consistently.

### Forward kernel mapping

Each Triton program owns one `(query_tile,head)` pair. It loads Q once, walks through K/V tiles, updates the three online buffers, and writes its output and L once. The implementation uses masked pointer grids rather than block pointers to support D=100 and uneven sequence tails explicitly.

`BM=32`, `BN=64`, four warps, and two pipeline stages remain the forward/dQ defaults. Following measured Ampere tuning on November 9, dK/dV uses `BM=32`, `BN=32`, four warps, and one stage for FP16/BF16 with `64 < D <= 128`, reducing register spills. Other dK/dV cases keep the original configuration. D is padded to `max(16,next_power_of_2(D))` within the tile; padding lanes are zeros, not extra features in the attention scale. For D=100 the internal width is 128, but the scale remains `1/sqrt(100)` and only 100 features are stored.

Triton `tl.constexpr` parameters are compile-time specializations, permitting static tile shapes and eliminating the unused causal branch. `program_id(0)` selects a query or key tile; `program_id(1)` selects a flattened batch/head index. Masked loads supply zeros outside real storage, and invalid score columns are set to negative infinity. Output stores mask both sequence and feature tails. Head indices are cast to int64 **before** multiplying strides: configuration C has more than 2^31 elements, so int32 element offsets overflow even when the byte pointers themselves are 64-bit.

### Backward, derived

Let G=`dLoss/dO`. Then:

```text
dV = P^T G
dP = G V^T
Delta_i = sum_d O[i,d] G[i,d]
dS[i,j] = P[i,j] * (dP[i,j] - Delta_i)
dQ = dS K / sqrt(D)
dK = dS^T Q / sqrt(D)
```

For softmax, `dS_j=P_j*(dP_j-sum_k P_k*dP_k)`. The row correction equals `sum_d O_d*G_d`, because expanding `O=P V` yields the same weighted sum. This identity lets `_delta` compute one scalar per query without constructing P.

The backward kernels load L and recompute `P_tile=exp(S_tile-L)`. `_backward_q` owns a Q tile and loops over keys. `_backward_kv` owns a K/V tile and loops over queries, accumulating both dK and dV. Recomputing P twice costs arithmetic, but each program has exclusive ownership of its output region. No atomic additions, cross-program reductions, or N-by-N global buffers are needed. Causal loops skip wholly future tiles and mask diagonal boundary tiles. The code returns actual gradients for arbitrary G; it does not exploit the evaluator's special all-ones gradient to change the mathematical task.

Autograd calls `forward(ctx,...)` without tracking its internal operations. `ctx.save_for_backward` saves Q/K/V/O/L and checks their version counters. `ctx.is_causal` saves the Python flag. Backward must return one entry per supplied argument: three tensor gradients for the course call, or three gradients plus `None` for an explicitly supplied Boolean. This is why the returned tuple is sliced using `ctx.needs_input_grad`.

## 4. Precision and memory choices

FP16 uses 1 sign bit, 5 exponent bits, and 10 fraction bits; BF16 uses 1/8/7. FP32 uses 1/8/23. BF16 has a wider range than FP16 but less precision near one. FP32 accumulation is valuable even when multiplication inputs are low precision: repeatedly adding a small value into a large FP16 partial sum can eventually stop changing it. Converting a previously rounded input to FP32 improves accumulation, but does not restore the lost input bits.

The kernels keep maxima, denominators, output accumulators, Delta, and gradient accumulators in FP32. For FP16/BF16 inputs, probability/score-gradient tiles are converted to the multiplication input dtype before tensor-core dot products. For FP32 inputs, `input_precision='tf32x3'` requests a more accurate decomposition than a single TF32 multiply. The real-device tests passed on November 9, while the original FP32 course references exposed FP16 quantization error in dK. The interpreter does not model all tensor-core rounding.

For config C, `E=256*131072*100=3,355,443,200` elements. One FP32 tensor is exactly 12.5 GiB, and one FP16 tensor is 6.25 GiB. The three inputs, output, and three gradients require seven such tensors:

| Storage | FP32 | FP16 |
| --- | ---: | ---: |
| Q/K/V | 37.5 GiB | 18.75 GiB |
| O | 12.5 GiB | 6.25 GiB |
| dQ/dK/dV | 37.5 GiB | 18.75 GiB |
| Seven tensors total | 87.5 GiB | 43.75 GiB |
| L plus Delta, always FP32 | 0.25 GiB | 0.25 GiB |

The FP16 total leaves room for autograd temporaries, allocator/context overhead, and per-head reference comparisons. It is an analytical accounting, not a measured peak or proof against every caller's allocations. Contiguous course inputs are reused as views. Sliced inputs are copied into contiguous storage; users retaining other large tensors increase memory needs.

`O.sum().backward()` usually supplies a tensor backed by a scalar with zero strides. Calling `.contiguous()` would allocate another entire output-sized tensor. The kernels instead receive the actual batch/row/feature strides, including zero, and load G directly. For general multi-dimensional leading layouts, `reshape` may still need a copy; the evaluator's `[H,N,D]` layout does not.

The ECNU metric uses absolute error with `rtol=0` and chooses the worst threshold across outputs/configurations. Comparing FP16 computation against already FP16-rounded inputs is only a kernel check. To assess the course grade, compare against the original FP32 reference as well: input conversion itself introduces error. FP16 is a memory-conscious choice; the full-A GPU check measured a dK error of 0.001153 and therefore passes 1e-2 rather than 1e-3. BF16 often introduces more input rounding; full FP32 cannot fit the largest case with ordinary FP32 output/gradients.

## 5. DDP and collective communication

Data parallelism splits the batch, while each rank stores a full model. Broadcast rank-0 parameters initially, compute each local mean loss, average gradients, then update. Equal-sized local batches make the average of local mean gradients equal the full-batch mean gradient. Unequal batches require weighting by sample count; this wrapper assumes equal batches.

All-reduce produces a summed tensor on every rank. All-gather assembles shards into a full tensor on every rank. Reduce-scatter sums full local contributions and returns only one shard per rank. Gloo is used for CPU debugging; NCCL is used for CUDA. A rank is a process ID in the group, and local rank selects the GPU within its machine. A collective requires every participating rank to call compatible operations in the same order; mismatched branches can hang or reduce unrelated tensors.

The three DDP modes demonstrate different tradeoffs:

- `individual`: wait for backward, all-reduce each gradient, divide by world size.
- `flat`: concatenate gradients per dtype/device, all-reduce the flat buffer, unpack. Fewer collective calls reduce latency overhead but require a large temporary buffer.
- `overlap`: post-accumulate hooks issue asynchronous all-reduces as each parameter finishes backward. `finish_gradient_synchronization()` waits and installs averaged results before the optimizer runs.

Communication buffers are cloned so autograd does not mutate in-flight data. Tied parameters are deduplicated through `module.parameters()`, and their post-accumulate hook sees the completed gradient. Frozen parameters are broadcast but need no gradient collective. Module buffers are synchronized initially.

Call finish after every backward and before step/zero_grad. This educational wrapper assumes one backward per step, consistent parameter use/order across ranks, and no rank-dependent unused parameters. Gradient accumulation, `no_sync`, uneven batches, and production fault tolerance are not implemented. These are explicit interface limits, not properties supplied by passing the toy tests.

## 6. Optimizer state sharding

For P FP32 parameters, ordinary AdamW stores approximately `4P` bytes of weights, `4P` of gradients, and `8P` of first/second moments: `16P` bytes excluding activations and temporaries. Adam state is allocated lazily on the first update. Measuring immediately after optimizer construction will miss most of its cost.

`ShardedOptimizer` assigns whole tensors greedily to the least-loaded rank. An owner updates its tensors and keeps their optimizer state; it then broadcasts updated tensors so replicas agree. Memory becomes approximately `8P+8P/W` bytes per rank. Tensor-granularity assignment is not perfectly balanced when a single parameter is very large. Gradients must already be averaged by DDP; the optimizer does not do that part.

Public parameter groups include all parameters so `zero_grad` and learning-rate schedulers work normally. The local optimizer sees only owned parameters. Ranks with no owned tensor use empty groups. `add_param_group` assigns new tensors deterministically and preserves per-group hyperparameters. Save one checkpoint per rank; restore the same rank/world size/group ordering. No repartitioning or global checkpoint consolidation is supplied.

This is a simple ZeRO-1-style implementation, not a full production ZeRO package. It retains replicated gradients, communicates whole updated tensors, and does not flatten/pipeline updates or reduce gradients directly to owners. Real ZeRO implementations can schedule and partition communication more efficiently.

## 7. FSDP

FSDP also shards master weights and gradients. In this implementation, Basics `Linear` and `Embedding` layers are sharded; norm parameters stay replicated. Each weight is flattened and padded to an equal shard length. Padding is excluded when reconstructing the original shape. The optimizer is constructed **after** wrapping because parameters are replaced with shards.

A custom Linear autograd function gathers weight shards, computes `X @ W.T`, and saves X plus the master shard. It does not save the full weight. Backward gathers W again for dX, computes dW from G and X, then reduce-scatters dW. The Embedding function saves token IDs and accumulates repeated-token gradients with `index_add_`. Tied layers share one master shard. Replicated norm gradients use asynchronous all-reduce followed by division by world size.

When requested, shards are cast before all-gather, so communicated compute weights are low precision while master weights and optimizer updates remain FP32. Gathered weights are released after use. For CPU portability, Gloo uses all-reduce followed by slicing; CUDA/NCCL uses `reduce_scatter_tensor`. `gather_full_params()` is a collective debug/checkpoint helper and temporarily reconstructs full parameters, so it does not represent steady-state FSDP memory.

The implementation gathers on demand and does not prefetch future layers. This bounds gathered-weight lifetimes, but communication can remain on the critical path. It is a correctness-oriented baseline; overlap improvements need a later NCCL trace. Ignoring temporary gathers and replicated small layers, FP32 AdamW model/gradient/state storage tends toward `16P/W` bytes per rank.

## 8. Activation checkpointing

Ordinary autograd keeps each layer's residuals for backward. Checkpointing saves a region's input and reruns its forward during backward to reconstruct internal residuals. It trades compute for activation memory, not parameter/optimizer memory. Non-reentrant checkpointing is used; PyTorch's RNG preservation makes dropout recomputation consistent.

Let N be layer count, A bytes per saved boundary activation, and R bytes per layer's residuals. With flat groups of k layers, a rough peak is `(N/k)*A+k*R`. Treating k continuously gives `k≈sqrt(N*A/R)`, clipped to valid group sizes. Since transformer residuals can greatly exceed one boundary activation, the optimum is not necessarily `sqrt(N)`. Measure neighboring integer k values at the actual model/sequence size. Flat grouping recomputes each region roughly once.

Recursive bisection stores a logarithmic stack of boundaries and adds recomputation at each level: approximately `O(A log N+R)` activation storage and `O(N log N)` block work. It is provided as a useful schedule, not an absolute optimum with unlimited compute. If minimizing activation memory regardless of compute, one can save only the stack input, recompute every layer's input prefix afresh while traversing layers backwards, then differentiate just that layer. This uses a constant number of activation-sized values plus one layer's residuals, and `O(N^2)` forward work. Such a prefix strategy requires managing parameter gradients and reproducible randomness carefully; the guide distinguishes it from the implemented recursive schedule.

The default grouped checkpoint closure binds its start/end indices immediately. Forgetting to bind them makes all delayed recomputation closures use the final loop indices and silently changes gradients. Regression tests compare all parameter/input gradients and dropout behavior against an ordinary five-layer stack.

## 9. Profiling workflow and interpretation

All tools default to small CPU inputs and a single configuration. They never sweep model sizes, launch a GPU worker pool, or run the huge evaluator. Their timing limits keep routine repetitions short, but deliberately requesting a large model/sequence can still make one step expensive.

```bash
# Tiny CPU smoke runs
OMP_NUM_THREADS=2 python -m cs336_systems.benchmark --steps 2 --checkpoint-group 1
OMP_NUM_THREADS=2 python -m cs336_systems.benchmark_attention --length 128
OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 \
  -m cs336_systems.benchmark_collective --megabytes 1 --steps 3
OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 \
  -m cs336_systems.benchmark --distributed overlap --sharded-optimizer --steps 2
```

For model experiments, `--d-model`, `--d-ff`, `--layers`, `--heads`, `--vocab`, `--context`, and `--batch` select the configuration. The upstream table is:

| Name | d_model | d_ff | layers | heads |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

The upstream reference uses vocabulary 10,000, batch 4, context 512 unless otherwise specified. These sizes are provided for translating the handout, not a recommendation to execute them today.

`--mode forward` runs inference with gradient tracking disabled; `backward` runs forward and backward without updates; `train` also updates AdamW. The forward phase includes loss computation. `--precision bf16` enables autocast; it does not convert master parameters to BF16. `--compile` wraps the model with `torch.compile`; warmup must absorb compilation, and the first backward can compile additional code. `--attention triton` replaces the Basics causal attention operation; the reference model otherwise remains unchanged.

Warmup exercises allocator caches, libraries, compiled kernels, and optimizer state creation. GPU calls are asynchronous, so CPU call-return time alone underestimates device completion. The model timer synchronizes at phase boundaries, and the collective timer synchronizes before/after all-reduce. Those phase fences perturb natural pipelining. For overlapped DDP, the reported backward phase can include communication and `communication_wait` is only the exposed wait after backward, **not total collective time**. Use a trace to see real overlap. Do not add isolated forward/backward timings and assume that equals unfenced throughput.

The attention timer uses `triton.testing.do_bench` on CUDA, with only 5ms warmup and 10ms repetition by default; CPU mode uses short wall-clock samples. Backward-only timing retains one forward graph and clears input gradients before each repeat. The combined measurement releases that graph and recomputes forward every time. Output reports allocated bytes before backward and peak allocation during combined timing. Compilation/profiling overhead and short sample windows can distort comparisons; record settings with any reported result.

### Traces and memory snapshots

During a later small authorized GPU run:

```bash
python -m cs336_systems.benchmark --device cuda --steps 2 \
  --trace output/tiny-trace.json --memory-snapshot output/tiny-memory.pickle
```

The actual paths get `.rank0` suffixes. Chrome traces can be opened in Perfetto; memory snapshots can be opened in PyTorch's memory visualizer. Distinguish allocated memory (live tensors) from reserved memory (allocator-held segments). Snapshots use PyTorch's private memory-history APIs and may need adjustment if the runtime changes. Memory phase counters describe the configured run after warmup; they are not an isolated measurement of Adam's first allocation.

Nsight example for later use:

```bash
nsys profile --trace=cuda,nvtx --capture-range=nvtx --nvtx-capture=measurement \
  python -m cs336_systems.benchmark --device cuda --steps 2
```

The outer `measurement` NVTX range excludes warmup. Use Nsight's PyTorch tracing options from the handout to label individual framework operations, and compare the attention score/softmax/value ranges in the Chrome trace. Matmuls can dominate FLOPs while softmax, normalization, copies, and optimizer elementwise updates still take substantial runtime because they have lower arithmetic intensity or launch overhead. Profiling changes timing, so collect final timing without trace recording.

## 10. Communication and parallelism calculations

These analytical models explain the later written questions without inventing measurements. Let C be FLOP/s and W be bytes/s of per-device egress bandwidth. Let `r(n)=(n-1)/n`. For a full tensor of S bytes, ideal ring all-gather and reduce-scatter each take `r(n)*S/W`; all-reduce takes `2*r(n)*S/W`. Sending every rank's full tensor around the ring instead takes `(n-1)*S/W`, because each of n-1 rounds sends S bytes rather than S/n. Real collectives also pay latency per round and topology/contention costs.

For the handout's gated FFN, the three weights contain `3*D*F` elements (`F=d_ff`). Forward has three matmuls, `6*B*D*F` FLOPs. Backward has six matmuls, `12*B*D*F` FLOPs, ignoring elementwise activations. All communicated tensors in the following simplified model are FP16.

| Strategy | Forward compute/device | Backward compute/device | Forward communication | Backward communication |
| --- | --- | --- | --- | --- |
| DP, n devices | `6BDF/n` | `12BDF/n` | 0 | `12*r(n)*DF/W` |
| FSDP, n devices | `6BDF/n` | `12BDF/n` | `6*r(n)*DF/W` | `12*r(n)*DF/W` |
| TP, n devices | `6BDF/n` | `12BDF/n` | `4*r(n)*BD/W` | `4*r(n)*BD/W` |

DP averages full weight gradients. FSDP gathers 6DF weight bytes in forward and again in backward, then reduce-scatters 6DF gradient bytes. TP shards the two expanding weights by output columns and the contracting weight by input rows, so intermediate activations stay sharded. Its forward sums partial outputs and backward sums partial input gradients, each an all-reduce of 2BD bytes.

For TP backward on rank i:

```text
dz_i = dy @ W3_i^T
dx2_i = dz_i * f(x1_i)
dx1_i = dz_i * f'(x1_i) * x2_i
dW3_i = z_i^T @ dy
dW2_i = x^T @ dx2_i
dW1_i = x^T @ dx1_i
dx = all_reduce(dx1_i @ W1_i^T + dx2_i @ W2_i^T)
```

Requiring communication time <= compute time gives exact ring-model bounds:

```text
DP backward:    n - 1 <= B*W/C
FSDP forward:   n - 1 <= B*W/C
FSDP backward:  n - 1 <= B*W/C
TP forward:    n - 1 <= (3/2)*F*W/C
TP backward:   n - 1 <= 3*F*W/C
```

For 2D parallelism, let f be FSDP degree and t TP degree. Forward compute is `6BDF/(f*t)`. The FSDP-axis communication is `6*r(f)*DF/(t*W)` and TP-axis communication is `4*r(t)*BD/(f*W)`. If axes overlap perfectly, communication time is their maximum; if they share the same bottleneck network, it is their sum.

Define `a=B*W/C` and `b=(3/2)*F*W/C`. With ideal overlapping axes, compute-bound operation requires `f-1<=a` and `t-1<=b`, giving maximum continuous device count `(1+a)*(1+b)`. With shared communication resources, the constraint becomes `(f-1)/a+(t-1)/b<=1`. For an interior continuous optimum, `f=(1+a+a/b)/2`, `t=(1+b+b/a)/2`; if either falls below one, the optimum lies on the corresponding boundary. In the large-degree approximation (ignore the minus-one ring factors), the familiar bounds become `N<=a*b` for overlap and `N<=a*b/4` without overlap. State which approximation you use in a written answer.

These bandwidth models explain why larger batches help DP/FSDP and larger FFN widths help TP. They omit network startup, hidden overlap constraints, optimizer overhead, and memory limits, so they are not predictions of measured end-to-end speed.

## 11. A controlled tuning plan for a later GPU session

First establish small device correctness and a baseline. Change one parameter at a time, such as BM=32 to 64, keeping a single small representative shape. Larger tiles reduce loop and memory-traffic overhead but increase register/shared-memory use and may lower occupancy or spill. More warps distribute work but consume resources; more stages can overlap loads with compute but require buffers. Inspect compilation metadata, then measure. Use a small hand-picked candidate list, not unrestricted `@triton.autotune` over all large evaluator cases.

Check D=100 and D=128, sequence tails, both masks, arbitrary upstream gradients, and input-rounding error after each change. A speedup that lowers the worst accuracy coefficient can reduce the final course score. Do not switch normalization log bases or reduce precision without a new numerical comparison. Do not infer that successful compilation means the chosen tile is fast, or that a tiny interpreter case proves the largest evaluator case fits.

## 12. References

- Assignment handout: `../cs336_assignment2_systems.pdf`, especially sections 2-8 and Algorithms 1-2.
- Course interface, configurations, scoring, and evaluator: `FA2.md`.
- Login, storage, and reservation schedule: `login_method.md`.
- [Triton dot-product precision options](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
- [Triton interpreter and debugging limitations](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
- [FlashAttention-2 paper](https://tridao.me/publications/flash2/flash2.pdf), for tiling and program ownership concepts.

Implementation was written from the assignment equations; tutorial implementations are not vendored as submission code.

## 13. Verification record

See [validation.md](validation.md) for the exact environment, commands exercised,
23-passed CPU suite, 14-passed interpreter run, offline compilation metadata,
smoke checks, and the remaining GPU validation work.
